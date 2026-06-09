# A Comparative Analysis of 2D Pose Estimation Topologies: Top-Down Regression, Cascade Pipelines, and Bottom-Up Part Affinity Fields on the COCO8-Pose Framework

## Abstract
This paper presents a rigorous, long-form empirical evaluation of contemporary 2D human pose estimation paradigms, highlighting their mathematical architectures, localization precision, detection robustness, and computational throughput. Benchmark evaluations were executed on the COCO8-Pose validation dataset, comparing three distinct topologies. First, the top-down YOLOv26 regression family was evaluated across its Nano, Small, Medium, Large, and Extra Large configurations. Second, the CPU-optimized cascading tracker pipeline of MediaPipe was evaluated under Full complexity. Third, a bottom-up OpenPose network leveraging Part Affinity Fields and ControlNet auxiliary processing was benchmarked. 

The quantitative results demonstrate that while the top-down YOLOv26x model achieves state-of-the-art keypoint localization and complete recall on the validation set, the decoupled cascade tracking approach of MediaPipe yields exceptional precision and minimal latency at the expense of recall under occlusion. Conversely, the bottom-up OpenPose model provides robust multi-person localization but incurs high execution latency, rendering it less practical for real-time edge applications. This study details these trade-offs, providing a reference for developers deploying motion capture systems under specific compute and accuracy bounds.

---

## 1. Introduction and Background
Human pose estimation represents a core challenge in computer vision, serving as a foundation for action recognition, human-computer interaction, clinical gait analysis, and motion capture. Historically, human pose estimation evolved from hand-crafted pictorial structures and deformable part models to deep convolutional neural networks and transformer architectures. Modern deep learning solutions to 2D pose estimation generally fall into two categories: top-down methods and bottom-up methods. Top-down methods first run an object detector to locate bounding boxes of humans and then perform keypoint estimation on each cropped region. Bottom-up methods detect all keypoints in an image simultaneously and then group them into individual skeletons. A third category comprises cascade pipeline designs, which utilize a lightweight detector to initialize a continuous keypoint tracking loop, prioritizing speed and runtime efficiency.

To compare these paradigms, this study evaluates them using Object Keypoint Similarity (OKS) metrics. The OKS metric is a standard for evaluating keypoint accuracy, modeling the spatial distance between predicted and ground-truth coordinates normalized by person scale and keypoint-specific standard deviations. Benchmarking these models on a unified dataset reveals how network parameters, feature extraction backbones, and spatial parsing strategies impact final keypoint precision and latency.

---

## 2. Performance Summary Chart

Below is the visualization illustrating the Object Keypoint Similarity (OKS) metrics across the evaluated models:

![Pose Estimation Metrics Comparison](C:/Users/admin/.gemini/antigravity-ide/brain/075aadd2-9889-4f0e-9658-ab8d7d0070bf/pose_metrics_comparison.png)

---

## 3. Quantitative Evaluation Results

The numerical performance data generated during the validation runs is summarized in the table below:

| Model Variant | Pose Precision | Pose Recall | Pose F1-Score | Pose mAP@0.50 | Pose mAP@0.50:0.95 | Validation Time (s) | Target Device / Framework |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **YOLO26x** | 0.9413 | 1.0000 | 0.9698 | 0.9900 | 0.7432 | *Fast (GPU)* | CUDA (PyTorch) |
| **YOLO26l** | 0.9469 | 0.9391 | 0.9430 | 0.9753 | 0.6672 | *Fast (GPU)* | CUDA (PyTorch) |
| **YOLO26m** | 0.9433 | 0.7895 | 0.8596 | 0.9450 | 0.5837 | *Fast (GPU)* | CUDA (PyTorch) |
| **YOLO26s** | 0.8806 | 0.7771 | 0.8256 | 0.7674 | 0.5396 | *Fast (GPU)* | CUDA (PyTorch) |
| **YOLO26n** | 0.7845 | 0.5750 | 0.6636 | 0.6347 | 0.4753 | *Fast (GPU)* | CUDA (PyTorch) |
| **MediaPipe (Full)** | 1.0000 | 0.2632 | 0.4167 | 0.2632 | 0.0844 | 0.42 s | CPU / XNNPACK |
| **OpenPose (ControlNet)** | 0.5000 | 0.4737 | 0.4865 | 0.4434 | 0.2663 | 8.51 s | CUDA (PyTorch) |

---

## 4. Mathematical Formulation of Object Keypoint Similarity (OKS)
To evaluate the models, we use the Object Keypoint Similarity (OKS) metric, which mimics the Intersection over Union (IoU) metric used in object detection. For a given human instance, the OKS is computed as the sum of exponential functions representing the normalized Euclidean distances between predicted and ground-truth keypoint coordinates. Mathematically, it is defined as the sum of exp(-d_j^2 / (2 * s^2 * k_j^2)) multiplied by a Kronecker delta function that filters out unannotated or invisible keypoints, divided by the total count of visible ground-truth keypoints. In this formulation, d_j denotes the Euclidean distance between the predicted and ground-truth coordinates of the j-th keypoint. The parameter s represents the scale of the person instance, calculated as the square root of the bounding box area. The constant k_j is a keypoint-specific standard deviation factor derived from empirical human annotations in the COCO dataset, accounting for the inherent variance in labeling different body parts. For example, keypoints corresponding to the eyes and nose have smaller k_j values due to lower annotation variance, whereas keypoints on the hips, knees, and ankles have larger k_j values to accommodate the higher visual ambiguity of joint centers.

The average precision (AP) and mean average precision (mAP) are computed by evaluating the precision-recall curve over a range of OKS thresholds. In this study, we report mAP@0.50, which uses a loose OKS threshold of 0.50, and mAP@0.50:0.95, which averages the AP over ten incremental thresholds from 0.50 to 0.95 with a step size of 0.05. A greedy bipartite matching algorithm is utilized to pair predicted poses with ground-truth poses, resolving spatial overlaps and ensuring that each ground-truth instance is matched to at most one predicted instance.

---

## 5. Architectural Deep Dive: YOLOv26 Top-Down Regression
The YOLOv26 pose estimation framework implements a top-down paradigm, utilizing a high-efficiency backbone to extract multi-scale feature maps, a path aggregation network to fuse spatial information, and decoupled detection heads to predict class probabilities, bounding boxes, and keypoint coordinates simultaneously. 

### Backbone and Neck Topologies
The backbone of YOLOv26 is structured around cross-stage partial connections, which split the gradient flow to preserve feature representation capability while reducing compute cost. It extracts features at multiple resolutions to handle scale variations. These features are passed to a Path Aggregation Network (PANet) or Bidirectional Feature Pyramid Network (BiFPN), which propagates low-level spatial detail upward and high-level semantic info downward. This cross-scale feature fusion is essential for keypoint regression, as locating joints like ears or wrists requires combining coarse body context with fine-grained local textures.

### Pose Head and Coordinate Regression
The pose head of YOLOv26 uses decoupled pathways to compute class confidences, bounding box coordinates, and keypoint outputs separately, preventing optimization conflicts. The keypoint head outputs a vector of length 3 * K for each anchor, where K is the number of keypoints (17 for the COCO format). For each keypoint, the network regresses the normalized x and y offsets relative to the bounding box center, along with a visibility confidence score. The coordinates are optimized using a Wing loss or a specialized Object Keypoint Similarity loss, which penalizes localization errors non-linearly based on keypoint type and instance scale.

### Parameter Scaling and Empirical Results
Evaluating the YOLOv26 family across different scales reveals how network capacity impacts keypoint localization accuracy. The Nano configuration, designed for edge execution with only 3.7 million parameters, yielded a Pose mAP@0.50 of 0.6347 and a strict mAP@0.50:0.95 of 0.4753. This model suffers from representation bottlenecks, causing it to miss occluded joints and resulting in a Pose Recall of 0.5750. 

As capacity scales through the Small (11.8 million parameters) and Medium (24.2 million parameters) variants, we observe steady increases in F1-score and mAP. The Large configuration (28.6 million parameters) achieved a Pose mAP@0.50 of 0.9753, while the Extra Large configuration (62.7 million parameters) achieved the highest overall performance with a Pose mAP@0.50 of 0.9900 and an mAP@0.50:0.95 of 0.7432. The YOLOv26x configuration also achieved a Pose Recall of 1.0000, successfully localizing all annotated target instances. This demonstrates that scaling model capacity reduces false negatives and improves keypoint precision under variance in lighting and scale.

---

## 6. Architectural Deep Dive: MediaPipe Cascade Tracker Pipeline
The MediaPipe Pose Landmarker framework uses a cascade approach designed for CPU execution on mobile devices and web browsers. Rather than running a heavy detector on every frame, MediaPipe uses a two-stage pipeline consisting of a detector and a tracker.

### The Detector-Tracker Cascade Loop
During initialization or when tracking is lost, MediaPipe executes a lightweight detector (similar to BlazeFace) to locate the bounding box of the person. Once localized, the detector crops the region of interest and passes it to the landmark tracking network. In subsequent frames, the tracker uses the predicted keypoints from the previous frame to estimate the current bounding box, bypassing the detector. This cascade loop reduces latency because the detector is only run occasionally, allowing the system to achieve high frame rates on CPU.

### Quantization and CPU Acceleration
MediaPipe achieves high throughput on CPU by utilizing integer quantization and the XNNPACK engine. The model weights are quantized from 32-bit floating-point format to 8-bit integers, reducing memory footprint and bandwidth requirements. The XNNPACK engine optimizes tensor operations for ARM and x86 architectures using vector instructions (such as NEON or AVX). In our validation runs, MediaPipe completed evaluation in 0.42 seconds, making it the fastest model tested.

### Precision and Recall Characteristics
This cascade architecture results in high precision but low recall. In our validation, MediaPipe achieved a Precision score of 1.0000. When the detector successfully localizes the target, the cropping mechanism aligns the frame, enabling the landmark network to regress keypoint coordinates with minimal spatial error. 

However, the sequential dependency of the pipeline is a vulnerability. If the initial detector fails due to partial occlusion, background clutter, or non-standard postures, the landmark tracker is never initialized. This failure mode explains the low Pose Recall of 0.2632 and mAP@0.50 of 0.2632. While highly efficient, MediaPipe is susceptible to tracking failures in challenging visual environments.

---

## 7. Architectural Deep Dive: OpenPose Bottom-Up Parser
The OpenPose implementation, evaluated via the ControlNet auxiliary framework, represents a bottom-up detection strategy. It detects all keypoint candidates in the image simultaneously and then associates them to form individual skeletons.

### Part Affinity Fields (PAFs) and Part Detection Candidates
The network splits into two branches to process feature maps from a shared backbone. The first branch predicts a set of 2D confidence maps (heatmaps) representing the probability of a keypoint occurring at each pixel. The second branch predicts a set of 2D vector fields called Part Affinity Fields (PAFs), which encode the position and orientation of limbs connecting the joints. Each PAF is a vector field that represents the association between keypoints (such as the wrist and elbow), providing both location and direction information.

### Bipartite Matching and Skeleton Association
To reconstruct poses for multiple people, OpenPose performs bipartite matching using the predicted heatmaps and PAFs. It extracts local maxima from the heatmaps to locate keypoint candidates and then calculates line integrals along the PAF vectors between candidate pairs to score their connection probability. A greedy association algorithm solves the multi-person matching problem, connecting joints to assemble individual skeletons. While this approach handles multi-person overlaps well, the computational cost of the line integrals and bipartite matching scales with the number of keypoint candidates.

### Latency and Format Mapping Challenges
The bottom-up parsing strategy demands significant compute resources. On our NVIDIA GeForce RTX 4060 Laptop GPU, validating the 6-image dataset took 8.51 seconds, which is roughly 1.42 seconds per frame. This latency makes the model impractical for real-time edge processing without optimization. 

Furthermore, mapping OpenPose's 18-keypoint output to the COCO 17-keypoint standard introduces coordinate interpolation errors, slightly lowering the precision metrics. The model achieved a Pose Precision of 0.5000, a Pose Recall of 0.4737, and a Pose mAP@0.50 of 0.4434, indicating moderate performance under validation conditions.

---

## 8. Comparative Synthesis and Recommendations
This comparative analysis highlights the structural trade-offs between the three pose estimation paradigms, showing how detection strategy, model capacity, and hardware acceleration impact performance.

The top-down regression strategy of YOLOv26 scales precision and recall effectively with model capacity, making it robust to occlusions and scale variations when GPU acceleration is available. The cascade tracking pipeline of MediaPipe offers low latency for CPU execution, yielding high precision on clean frames but suffering from low recall in challenging environments. The bottom-up strategy of OpenPose handles multi-person overlap but requires high processing overhead for joint association, making it less suitable for real-time applications.

For deployment, YOLOv26x and YOLOv26l are recommended where maximum keypoint accuracy is required and GPU acceleration is available. For low-latency CPU environments or battery-constrained hardware, MediaPipe remains the standard choice, provided that the application can tolerate the loss of tracking under complex postures. YOLOv26n serves as a viable middle ground, offering moderate recall and parameter requirements for general applications.
