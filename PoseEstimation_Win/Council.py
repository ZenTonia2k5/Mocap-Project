import cv2
import torch
import time
import os
import argparse
import numpy as np
import yaml
import urllib.request
from datetime import datetime

# Import models
from ultralytics import YOLO
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import draw_poses
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# COCO Keypoint standards
COCO_K = np.array([
    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
])

COCO_CONNECTIONS = [
    (0, 1), (1, 3), (0, 2), (2, 4),  # Face
    (5, 6),                          # Shoulders
    (5, 7), (7, 9),                  # Left arm
    (6, 8), (8, 10),                 # Right arm
    (11, 12),                        # Hips
    (5, 11), (6, 12),                # Torso
    (11, 13), (13, 15),              # Left leg
    (12, 14), (14, 16)               # Right leg
]

COCO_CLASSES = ["nose", "L-eye", "R-eye", "L-ear", "R-ear", "L-shoulder", "R-shoulder", 
                "L-elbow", "R-elbow", "L-wrist", "R-wrist", "L-hip", "R-hip", "L-knee", 
                "R-knee", "L-ankle", "R-ankle"]

# Mappings to COCO
OPENPOSE_TO_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
MEDIAPIPE_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

MODEL_COLORS = {
    "council": (255, 255, 0),     # Neon Cyan
    "yolo26x": (0, 0, 255),       # Bright Red
    "yolo26l": (0, 128, 255),     # Orange
    "yolo26m": (0, 255, 255),     # Yellow
    "yolo26s": (255, 0, 255),     # Magenta
    "yolo26n": (128, 0, 128),     # Purple
    "mediapipe": (0, 255, 0),     # Green
    "openpose": (255, 0, 0),      # Blue
}

def log_to_file(filepath, summary_dict, mode):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n========================================\n")
        f.write(f"Model/System: AI COUNCIL (7 Models Fused)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Mode: {mode}\n")
        for k, v in summary_dict.items():
            f.write(f"{k}: {v}\n")
        f.write(f"========================================\n")
    print(f"Metrics saved to log file: {os.path.abspath(filepath)}")

def compute_ap(tp_list, conf_list, num_gt):
    if num_gt == 0:
        return 0.0
    if len(tp_list) == 0:
        return 0.0
        
    sort_indices = np.argsort(conf_list)[::-1]
    tp = np.array(tp_list)[sort_indices]
    
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(1 - tp)
    
    recalls = tp_cumsum / num_gt
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))
    
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])
    return ap

def get_bbox_from_keypoints(kpts):
    """Computes a normalized bounding box [x_min, y_min, w, h] from a keypoint array (17, 3)."""
    valid_kpts = kpts[kpts[:, 2] > 0.05]
    if len(valid_kpts) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    x_min = np.min(valid_kpts[:, 0])
    y_min = np.min(valid_kpts[:, 1])
    x_max = np.max(valid_kpts[:, 0])
    y_max = np.max(valid_kpts[:, 1])
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def pose_distance(pose1, pose2):
    """Computes mean joint distance between two standardized poses (17, 3) for matching."""
    vis1 = pose1[:, 2] > 0.05
    vis2 = pose2[:, 2] > 0.05
    both = vis1 & vis2
    if np.sum(both) < 3:  # Need at least 3 shared keypoints to avoid random matches
        return 1.0
    dist = np.sqrt(np.sum((pose1[both, :2] - pose2[both, :2])**2, axis=1))
    return np.mean(dist)

def compute_oks(pred_pose, gt_pose):
    """Calculates Object Keypoint Similarity between a predicted and target pose."""
    pred_kpts = pred_pose["keypoints"]
    gt_kpts = gt_pose["keypoints"]
    bbox = gt_pose["bbox"]
    
    s = np.sqrt(bbox[2] * bbox[3])
    if s <= 0:
        s = 1e-5
        
    d_sq = (pred_kpts[:, 0] - gt_kpts[:, 0])**2 + (pred_kpts[:, 1] - gt_kpts[:, 1])**2
    vis = gt_kpts[:, 2] > 0
    if not np.any(vis):
        return 0.0
        
    oks_vals = np.exp(-d_sq / (2 * (s**2) * (COCO_K**2)))
    return np.sum(oks_vals[vis]) / np.sum(vis)

def match_poses_oks(pred_poses, target_poses, oks_threshold=0.50):
    """Performs bipartite matching of predicted poses to targets based on OKS."""
    M = len(target_poses)
    N = len(pred_poses)
    if M == 0 or N == 0:
        return []
        
    oks_matrix = np.zeros((M, N))
    for i in range(M):
        for j in range(N):
            oks_matrix[i, j] = compute_oks(pred_poses[j], target_poses[i])
            
    matched_targets = set()
    matched_preds = set()
    pairs = []
    
    for i in range(M):
        for j in range(N):
            if oks_matrix[i, j] >= oks_threshold:
                pairs.append((oks_matrix[i, j], i, j))
    pairs.sort(key=lambda x: x[0], reverse=True)
    
    matches = []
    for OKS, i, j in pairs:
        if i not in matched_targets and j not in matched_preds:
            matched_targets.add(i)
            matched_preds.add(j)
            matches.append((j, i, OKS))
            
    return matches

def cluster_and_average_poses(model_poses_dict, match_threshold=0.10):
    """
    Groups detections from the active models representing the same physical person
    and returns a list of Council Consensus average poses.
    """
    num_active_models = len(model_poses_dict)
    flat_poses = []
    for model_name, poses in model_poses_dict.items():
        for p in poses:
            flat_poses.append({
                "model": model_name,
                "keypoints": p["keypoints"],
                "conf": p["conf"],
                "bbox": p["bbox"]
            })
            
    clusters = []
    for p in flat_poses:
        best_cluster_idx = -1
        best_dist = match_threshold
        
        for idx, cluster in enumerate(clusters):
            dists = [pose_distance(p["keypoints"], member["keypoints"]) for member in cluster]
            mean_dist = np.mean(dists)
            if mean_dist < best_dist:
                best_dist = mean_dist
                best_cluster_idx = idx
                
        if best_cluster_idx != -1:
            existing_idx = -1
            for idx, member in enumerate(clusters[best_cluster_idx]):
                if member["model"] == p["model"]:
                    existing_idx = idx
                    break
            if existing_idx != -1:
                if p["conf"] > clusters[best_cluster_idx][existing_idx]["conf"]:
                    clusters[best_cluster_idx][existing_idx] = p
            else:
                clusters[best_cluster_idx].append(p)
        else:
            clusters.append([p])
            
    consensus_poses = []
    for cluster in clusters:
        consensus_kpts = np.zeros((17, 3))
        
        for j in range(17):
            coords = []
            confs = []
            for member in cluster:
                kp = member["keypoints"][j]
                if kp[2] > 0.05 and (kp[0] > 0 or kp[1] > 0):
                    coords.append(kp[:2])
                    confs.append(kp[2])
                    
            if len(coords) > 0:
                mean_coord = np.mean(coords, axis=0)
                mean_conf = np.mean(confs)
                consensus_kpts[j] = [mean_coord[0], mean_coord[1], mean_conf]
            else:
                consensus_kpts[j] = [0.0, 0.0, 0.0]
                
        consensus_bbox = get_bbox_from_keypoints(consensus_kpts)
        
        avg_model_conf = np.mean([member["conf"] for member in cluster])
        participation_ratio = len(cluster) / float(num_active_models) if num_active_models > 0 else 1.0
        consensus_conf = avg_model_conf * participation_ratio
        
        consensus_poses.append({
            "keypoints": consensus_kpts,
            "conf": consensus_conf,
            "bbox": consensus_bbox,
            "models_present": [member["model"] for member in cluster]
        })
        
    return consensus_poses

def draw_skeleton(frame, keypoints, color, thickness=2):
    """Draws a pose skeleton and joint circles on the frame."""
    H, W, _ = frame.shape
    for p1, p2 in COCO_CONNECTIONS:
        kp1, kp2 = keypoints[p1], keypoints[p2]
        if kp1[2] > 0.05 and kp2[2] > 0.05:
            pt1 = (int(kp1[0] * W), int(kp1[1] * H))
            pt2 = (int(kp2[0] * W), int(kp2[1] * H))
            cv2.line(frame, pt1, pt2, color, thickness, cv2.LINE_AA)
            
    for j in range(17):
        kp = keypoints[j]
        if kp[2] > 0.05:
            pt = (int(kp[0] * W), int(kp[1] * H))
            cv2.circle(frame, pt, thickness + 2, (0, 0, 0), -1)
            cv2.circle(frame, pt, thickness + 1, color, -1)

def main():
    parser = argparse.ArgumentParser(description="2D Pose Estimation AI Council")
    parser.add_argument("--mode", type=str, default="val", choices=["predict", "val"], 
                        help="Run mode: 'predict' for live stream/video or 'val' for dataset validation")
    parser.add_argument("--source", type=str, default="0", 
                        help="Path to video file or camera index (e.g., '0') for prediction")
    parser.add_argument("--data", type=str, default="my_coco8-pose.yaml", 
                        help="Path to dataset.yaml (required for validation mode)")
    parser.add_argument("--log", type=str, default="inference_log.txt", 
                        help="Path to the log file")
    parser.add_argument("--no-show", action="store_true", 
                        help="Disable displaying the GUI video stream window")
    parser.add_argument("--max-frames", type=int, default=None, 
                        help="Limit the number of frames to process in prediction")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold for predictions")
    parser.add_argument("--models", type=str, default="01234567",
                        help="Choose which models to run (digits 0-7, default '01234567'): "
                             "0=Council Consensus, 1=YOLOv26x, 2=YOLOv26l, 3=YOLOv26m, "
                             "4=YOLOv26s, 5=YOLOv26n, 6=MediaPipe, 7=OpenPose")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    device = '0' if cuda_available else 'cpu'
    
    print(f"\n========================================")
    print(f"Initializing AI Council in Pose Estimation")
    print(f"Device: {device_name} (CUDA={cuda_available})")
    
    # Map
    digit_map = {
        '1': 'yolo26x',
        '2': 'yolo26l',
        '3': 'yolo26m',
        '4': 'yolo26s',
        '5': 'yolo26n',
        '6': 'mediapipe',
        '7': 'openpose'
    }
    
    selected_digits = [c for c in args.models if c in '01234567']
    if not selected_digits:
        selected_digits = list('01234567')
        
    active_models = [digit_map[c] for c in selected_digits if c in digit_map]
    if not active_models:
        print("Error: No models selected to run. Please specify at least one digit in range 1-7.")
        exit(1)
        
    print(f"Selected Models to Run: {active_models}")
    print(f"========================================\n")

    # 1. Load YOLOv26 models if selected
    yolo_models = {}
    yolo_names = ['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x']
    for name in yolo_names:
        if name in active_models:
            pt_path = f"{name}-pose.pt"
            if not os.path.exists(pt_path):
                print(f"Downloading missing weights: {pt_path}...")
            yolo_models[name] = YOLO(pt_path)
            print(f"Loaded {name} pose model.")

    # 2. Load OpenPose if selected
    openpose = None
    if 'openpose' in active_models:
        print("Loading PyTorch OpenPose (ControlNet-based)...")
        try:
            openpose_device = 'cuda' if cuda_available else 'cpu'
            openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(openpose_device)
            print("Loaded OpenPose model.")
        except Exception as e:
            print(f"Failed to load OpenPose: {e}")
            exit(1)

    # 3. Load MediaPipe if selected
    mp_detector = None
    if 'mediapipe' in active_models:
        mp_task_path = os.path.join("models", "pose_landmarker_full.task")
        if not os.path.exists(mp_task_path):
            print(f"Downloading MediaPipe model weights to {mp_task_path}...")
            os.makedirs("models", exist_ok=True)
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
            urllib.request.urlretrieve(url, mp_task_path)
        
        print(f"Loading MediaPipe Pose Landmarker: {mp_task_path}...")
        base_options = python.BaseOptions(model_asset_path=mp_task_path)
        num_poses = 5 if args.mode == "val" else 1
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            min_pose_detection_confidence=args.conf,
            min_pose_presence_confidence=args.conf,
            min_tracking_confidence=args.conf,
            num_poses=num_poses,
            output_segmentation_masks=False
        )
        mp_detector = vision.PoseLandmarker.create_from_options(options)
        print("Loaded MediaPipe model.")

    model_names = ["council"] + active_models
    
    # Validation Mode
    if args.mode == "val":
        print(f"\n--- Running Validation on {args.data} ---")
        with open(args.data, 'r') as f:
            data_cfg = yaml.safe_load(f)
            
        dataset_path = data_cfg.get('path', '')
        if not os.path.isabs(dataset_path):
            data_dir = os.path.dirname(os.path.abspath(args.data))
            dataset_path = os.path.abspath(os.path.join(data_dir, dataset_path))
            
        val_rel = data_cfg.get('val', '')
        val_images_dir = os.path.join(dataset_path, val_rel)
        val_labels_dir = val_images_dir.replace("images", "labels")
        
        if not os.path.exists(val_images_dir):
            raise FileNotFoundError(f"Validation images directory not found: {val_images_dir}")
            
        image_extensions = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(val_images_dir) if f.lower().endswith(image_extensions)]
        print(f"Found {len(image_files)} validation images.")

        val_out_dir = os.path.join("runs", "pose", "council_val")
        os.makedirs(val_out_dir, exist_ok=True)
        print(f"Saving annotated predictions to: {os.path.abspath(val_out_dir)}")

        thresholds = np.arange(0.5, 1.0, 0.05)
        
        # Accumulators for OKS evaluation against dataset Ground Truth (GT)
        gt_eval = {m: {t: {"tp": 0, "fp": 0, "fn": 0, "tp_list": [], "conf_list": []} for t in thresholds} for m in model_names}
        total_num_gt = 0
        
        start_val_time = time.time()
        
        for img_idx, img_file in enumerate(image_files):
            img_path = os.path.join(val_images_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
            H, W, _ = frame.shape

            # Parse GT Poses
            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(val_labels_dir, label_file)
            gt_poses = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    for line in lf:
                        parts = list(map(float, line.strip().split()))
                        if not parts or int(parts[0]) != 0:
                            continue
                        bbox = parts[1:5] # [x, y, w, h] normalized
                        kpts = parts[5:] # [x, y, v, ...] normalized
                        kpts = np.array(kpts).reshape(17, 3)
                        gt_poses.append({
                            "bbox": bbox,
                            "keypoints": kpts
                        })
            total_num_gt += len(gt_poses)
            
            # Predict from each model on this image
            predictions = {m: [] for m in model_names}
            
            # A. Run YOLOv26 models
            for name, yolo in yolo_models.items():
                res = yolo.predict(source=frame, conf=args.conf, device=device, verbose=False)[0]
                if len(res.boxes) > 0:
                    xywhn = res.boxes.xywhn.cpu().numpy()
                    xyn = res.keypoints.xyn.cpu().numpy()
                    confs = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
                    box_confs = res.boxes.conf.cpu().numpy()
                    
                    for i in range(len(xywhn)):
                        kpts_std = np.zeros((17, 3))
                        kpts_std[:, :2] = xyn[i]
                        for j in range(17):
                            kpts_std[j, 2] = confs[i, j] if confs is not None else box_confs[i]
                        predictions[name].append({
                            "keypoints": kpts_std,
                            "conf": box_confs[i],
                            "bbox": xywhn[i]
                        })

            # B. Run OpenPose
            if "openpose" in active_models and openpose is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    poses = openpose.detect_poses(rgb_frame, include_hand=False, include_face=False)
                    for pose in poses:
                        body_kpts = pose.body.keypoints
                        coco_kpts = np.zeros((17, 3))
                        valid_count = 0
                        for coco_idx in range(17):
                            op_idx = OPENPOSE_TO_COCO[coco_idx]
                            kp = body_kpts[op_idx]
                            if kp is not None:
                                coco_kpts[coco_idx] = [kp.x, kp.y, 1.0]
                                valid_count += 1
                        
                        conf = valid_count / 17.0
                        predictions["openpose"].append({
                            "keypoints": coco_kpts,
                            "conf": conf,
                            "bbox": get_bbox_from_keypoints(coco_kpts)
                        })
                except Exception as e:
                    print(f"OpenPose error on {img_file}: {e}")

            # C. Run MediaPipe
            if "mediapipe" in active_models and mp_detector is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    mp_results = mp_detector.detect(mp_image)
                    if mp_results.pose_landmarks:
                        for pose_landmarks in mp_results.pose_landmarks:
                            coco_kpts = np.zeros((17, 3))
                            for coco_idx, mp_idx in enumerate(MEDIAPIPE_TO_COCO):
                                landmark = pose_landmarks[mp_idx]
                                coco_kpts[coco_idx] = [landmark.x, landmark.y, landmark.visibility]
                            predictions["mediapipe"].append({
                                "keypoints": coco_kpts,
                                "conf": np.mean(coco_kpts[:, 2]),
                                "bbox": get_bbox_from_keypoints(coco_kpts)
                            })
                except Exception as e:
                    print(f"MediaPipe error on {img_file}: {e}")

            # D. Compute Council Consensus (Fusion)
            predictions["council"] = cluster_and_average_poses(
                {k: v for k, v in predictions.items() if k != "council"}
            )

            # Evaluate each model's output vs Ground Truth
            for name in model_names:
                preds = predictions[name]
                M = len(gt_poses)
                N = len(preds)
                
                # Compute OKS Matrix
                oks_matrix = np.zeros((M, N))
                for i in range(M):
                    for j in range(N):
                        oks_matrix[i, j] = compute_oks(preds[j], gt_poses[i])
                
                # Check metrics at all thresholds
                for t in thresholds:
                    matched_gt = set()
                    matched_pred = set()
                    pairs = []
                    for i in range(M):
                        for j in range(N):
                            if oks_matrix[i, j] >= t:
                                pairs.append((oks_matrix[i, j], i, j))
                    pairs.sort(key=lambda x: x[0], reverse=True)
                    
                    tps = 0
                    for OKS, i, j in pairs:
                        if i not in matched_gt and j not in matched_pred:
                            if preds[j]["conf"] >= args.conf:
                                matched_gt.add(i)
                                matched_pred.add(j)
                                tps += 1
                            
                    fps = sum(1 for j in range(N) if preds[j]["conf"] >= args.conf and j not in matched_pred)
                    fns = M - len(matched_gt)
                    
                    gt_eval[name][t]["tp"] += tps
                    gt_eval[name][t]["fp"] += fps
                    gt_eval[name][t]["fn"] += fns
                    
                    for j in range(N):
                        is_tp = 1 if j in matched_pred else 0
                        gt_eval[name][t]["tp_list"].append(is_tp)
                        gt_eval[name][t]["conf_list"].append(preds[j]["conf"])

            # Save visualization of Council vs GT
            vis_img = frame.copy()
            # Draw GT in gold
            for gt in gt_poses:
                draw_skeleton(vis_img, gt["keypoints"], (0, 165, 255), thickness=1)
            # Draw Council average in neon cyan
            for cp in predictions["council"]:
                draw_skeleton(vis_img, cp["keypoints"], (255, 255, 0), thickness=3)
                
            # Add simple info text
            cv2.putText(vis_img, f"GT Count: {len(gt_poses)} | Council Count: {len(predictions['council'])}", 
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(val_out_dir, f"val_{os.path.splitext(img_file)[0]}_comparison.png"), vis_img)

            if (img_idx + 1) % 2 == 0 or (img_idx + 1) == len(image_files):
                print(f"Validated {img_idx + 1}/{len(image_files)} images...")

        # Calculate and print validation summaries
        validation_summary = {}
        print("\n" + "="*80)
        print(f"{'Pose Estimation AI Council Validation Report':^80}")
        print("="*80)
        print(f"{'Model Variant':<22} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'mAP@0.50':<9} | {'mAP50-95':<9}")
        print("-"*80)
        
        for name in model_names:
            # Stats at t = 0.50
            t_50 = gt_eval[name][0.50]
            tp = t_50["tp"]
            fp = t_50["fp"]
            fn = t_50["fn"]
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            map50 = compute_ap(t_50["tp_list"], t_50["conf_list"], total_num_gt)
            
            map50_95_vals = []
            for t in thresholds:
                map50_95_vals.append(compute_ap(gt_eval[name][t]["tp_list"], gt_eval[name][t]["conf_list"], total_num_gt))
            map50_95 = np.mean(map50_95_vals)
            
            display_name = f"**{name}**" if name == "council" else name
            print(f"{display_name:<22} | {prec:.4f}    | {rec:.4f}    | {f1:.4f}    | {map50:.4f}    | {map50_95:.4f}")
            
            validation_summary[f"{name}_Precision"] = f"{prec:.4f}"
            validation_summary[f"{name}_Recall"] = f"{rec:.4f}"
            validation_summary[f"{name}_F1"] = f"{f1:.4f}"
            validation_summary[f"{name}_mAP50"] = f"{map50:.4f}"
            validation_summary[f"{name}_mAP50_95"] = f"{map50_95:.4f}"
            
        print("="*80)
        
        validation_summary["Total Images Validated"] = len(image_files)
        validation_summary["Total Ground Truth Poses"] = total_num_gt
        validation_summary["Total Validation Time (s)"] = f"{time.time() - start_val_time:.2f}"
        
        log_to_file(args.log, validation_summary, "Validation Mode")
        if mp_detector is not None:
            mp_detector.close()
        return

    # Prediction Mode
    else:
        source_val = args.source
        if source_val.isdigit():
            source_val = int(source_val)
            
        print(f"\n--- Running Prediction on Source: {source_val} ---")
        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{source_val}'")
            if mp_detector is not None:
                mp_detector.close()
            exit(1)

        # Metrics for evaluating individual models against Council Consensus (Consensus acts as GT)
        # We track TP, FP, FN at OKS threshold = 0.50
        model_eval = {m: {"tp": 0, "fp": 0, "fn": 0, "keypoint_errors_px": []} for m in model_names if m != "council"}
        
        frame_count = 0
        fps_list = []
        
        # Interactive GUI Toggles:
        # '0': Council Consensus, '1': yolo26x, '2': yolo26l, '3': yolo26m, '4': yolo26s, '5': yolo26n, '6': mediapipe, '7': openpose
        toggles = {
            "council": True,
            "yolo26x": '1' in selected_digits,
            "yolo26l": '2' in selected_digits,
            "yolo26m": '3' in selected_digits,
            "yolo26s": '4' in selected_digits,
            "yolo26n": '5' in selected_digits,
            "mediapipe": '6' in selected_digits,
            "openpose": '7' in selected_digits
        }
        if args.models == "01234567":
            for k in toggles:
                toggles[k] = (k == "council")
                
        toggle_keys = {
            ord('0'): "council",
            ord('1'): "yolo26x",
            ord('2'): "yolo26l",
            ord('3'): "yolo26m",
            ord('4'): "yolo26s",
            ord('5'): "yolo26n",
            ord('6'): "mediapipe",
            ord('7'): "openpose"
        }

        if not args.no_show:
            cv2.namedWindow('Pose Estimation AI Council', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Pose Estimation AI Council', 1280, 720)
            print("\n" + "="*50)
            print("Interactive GUI Shortcuts:")
            print("  '0' - Toggle Council Consensus Skeleton (Cyan)")
            print("  '1' - Toggle YOLOv26x Skeleton (Red)")
            print("  '2' - Toggle YOLOv26l Skeleton (Orange)")
            print("  '3' - Toggle YOLOv26m Skeleton (Yellow)")
            print("  '4' - Toggle YOLOv26s Skeleton (Magenta)")
            print("  '5' - Toggle YOLOv26n Skeleton (Purple)")
            print("  '6' - Toggle MediaPipe Skeleton (Green)")
            print("  '7' - Toggle OpenPose Skeleton (Blue)")
            print("  'c' - Save current screenshot")
            print("  'q' - Quit stream")
            print("="*50 + "\n")
        else:
            print("Running prediction in headless mode (no GUI window)...")
            
        start_session_time = time.time()

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            if args.max_frames is not None and frame_count > args.max_frames:
                frame_count -= 1
                break
                
            H, W, _ = frame.shape
            t_start = time.time()
            
            # Predict from each model on this frame
            frame_predictions = {m: [] for m in model_names}
            
            # A. Run YOLOv26 models
            for name, yolo in yolo_models.items():
                res = yolo.predict(source=frame, conf=args.conf, device=device, verbose=False)[0]
                if len(res.boxes) > 0:
                    xywhn = res.boxes.xywhn.cpu().numpy()
                    xyn = res.keypoints.xyn.cpu().numpy()
                    confs = res.keypoints.conf.cpu().numpy() if res.keypoints.conf is not None else None
                    box_confs = res.boxes.conf.cpu().numpy()
                    
                    for i in range(len(xywhn)):
                        kpts_std = np.zeros((17, 3))
                        kpts_std[:, :2] = xyn[i]
                        for j in range(17):
                            kpts_std[j, 2] = confs[i, j] if confs is not None else box_confs[i]
                        frame_predictions[name].append({
                            "keypoints": kpts_std,
                            "conf": box_confs[i],
                            "bbox": xywhn[i]
                        })

            # B. Run OpenPose
            if "openpose" in active_models and openpose is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    poses = openpose.detect_poses(rgb_frame, include_hand=False, include_face=False)
                    for pose in poses:
                        body_kpts = pose.body.keypoints
                        coco_kpts = np.zeros((17, 3))
                        valid_count = 0
                        for coco_idx in range(17):
                            op_idx = OPENPOSE_TO_COCO[coco_idx]
                            kp = body_kpts[op_idx]
                            if kp is not None:
                                coco_kpts[coco_idx] = [kp.x, kp.y, 1.0]
                                valid_count += 1
                        
                        conf = valid_count / 17.0
                        frame_predictions["openpose"].append({
                            "keypoints": coco_kpts,
                            "conf": conf,
                            "bbox": get_bbox_from_keypoints(coco_kpts)
                        })
                except Exception as e:
                    pass

            # C. Run MediaPipe
            if "mediapipe" in active_models and mp_detector is not None:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    mp_results = mp_detector.detect(mp_image)
                    if mp_results.pose_landmarks:
                        for pose_landmarks in mp_results.pose_landmarks:
                            coco_kpts = np.zeros((17, 3))
                            for coco_idx, mp_idx in enumerate(MEDIAPIPE_TO_COCO):
                                landmark = pose_landmarks[mp_idx]
                                coco_kpts[coco_idx] = [landmark.x, landmark.y, landmark.visibility]
                            frame_predictions["mediapipe"].append({
                                "keypoints": coco_kpts,
                                "conf": np.mean(coco_kpts[:, 2]),
                                "bbox": get_bbox_from_keypoints(coco_kpts)
                            })
                except Exception as e:
                    pass

            # D. Compute Council Consensus (Fusion)
            frame_predictions["council"] = cluster_and_average_poses(
                {k: v for k, v in frame_predictions.items() if k != "council"}
            )
            
            t_end = time.time()
            fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0.0
            fps_list.append(fps)

            # Evaluate each individual model against Council Consensus
            consensus_poses = [p for p in frame_predictions["council"] if p["conf"] >= args.conf]
            for name in model_names:
                if name == "council":
                    continue
                preds = [p for p in frame_predictions[name] if p["conf"] >= args.conf]
                
                # Match predictions to Council Consensus
                matches = match_poses_oks(preds, consensus_poses, oks_threshold=0.50)
                
                tps = len(matches)
                fps_count = len(preds) - tps
                fns = len(consensus_poses) - tps
                
                model_eval[name]["tp"] += tps
                model_eval[name]["fp"] += fps_count
                model_eval[name]["fn"] += fns
                
                # Compute Keypoint Coordinate Errors for matched poses
                for pred_idx, target_idx, OKS in matches:
                    pred_kpts = preds[pred_idx]["keypoints"]
                    target_kpts = consensus_poses[target_idx]["keypoints"]
                    
                    vis_pred = pred_kpts[:, 2] > 0.05
                    vis_target = target_kpts[:, 2] > 0.05
                    both = vis_pred & vis_target
                    
                    if np.any(both):
                        # Calculate Euclidean distance in pixels
                        dx_px = (pred_kpts[both, 0] - target_kpts[both, 0]) * W
                        dy_px = (pred_kpts[both, 1] - target_kpts[both, 1]) * H
                        dists_px = np.sqrt(dx_px**2 + dy_px**2)
                        model_eval[name]["keypoint_errors_px"].extend(dists_px.tolist())

            # Draw skeletons on the frame based on active toggles
            annotated_frame = frame.copy()
            
            for key, active in toggles.items():
                if active and key in frame_predictions:
                    for p in frame_predictions[key]:
                        draw_skeleton(annotated_frame, p["keypoints"], MODEL_COLORS[key], thickness=3 if key == "council" else 2)

            # HUD Display
            y_offset = 40
            cv2.putText(annotated_frame, f"AI COUNCIL - Frame: {frame_count} | FPS: {fps:.1f}", 
                        (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            y_offset += 30
            
            # Active Skeleton Indicators in HUD
            indicator_text = "Active overlays: "
            cv2.putText(annotated_frame, indicator_text, (20, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
            x_offset = 160
            for name in model_names:
                active = toggles.get(name, False)
                text = f"[{name.upper()}] " if active else f"{name} "
                color = MODEL_COLORS[name] if active else (128, 128, 128)
                
                # Check for line wrap based on W
                text_width = len(text) * 11
                if x_offset + text_width > W - 20:
                    x_offset = 20
                    y_offset += 20
                    
                cv2.putText(annotated_frame, text, (x_offset, y_offset), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2 if active else 1, cv2.LINE_AA)
                x_offset += text_width

            # GUI Window Event Loop
            if not args.no_show:
                cv2.imshow('Pose Estimation AI Council', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') or key == ord('C'):
                    capture_dir = os.path.join("runs", "captures")
                    os.makedirs(capture_dir, exist_ok=True)
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    capture_path = os.path.join(capture_dir, f"council_{timestamp_str}.jpg")
                    cv2.imwrite(capture_path, annotated_frame)
                    print(f"Captured and saved frame to: {os.path.abspath(capture_path)}")
                elif key in toggle_keys:
                    toggled_name = toggle_keys[key]
                    toggles[toggled_name] = not toggles[toggled_name]
                    print(f"Toggled skeleton visibility of: {toggled_name.upper()} = {toggles[toggled_name]}")
            else:
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count} frames... Current FPS: {fps:.1f}")

        total_session_time = time.time() - start_session_time
        cap.release()
        if mp_detector is not None:
            mp_detector.close()
        if not args.no_show:
            cv2.destroyAllWindows()
            
        # Compute and output the final prediction report
        avg_fps = np.mean(fps_list) if fps_list else 0.0
        
        prediction_summary = {}
        print("\n" + "="*80)
        print(f"{'Individual Model Evaluation Relative to Council Consensus':^80}")
        print("="*80)
        print(f"{'Model Variant':<20} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Mean Error (px)':<15}")
        print("-"*80)
        
        for name in model_names:
            if name == "council":
                continue
            tp = model_eval[name]["tp"]
            fp = model_eval[name]["fp"]
            fn = model_eval[name]["fn"]
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            errs = model_eval[name]["keypoint_errors_px"]
            mean_err = np.mean(errs) if errs else 0.0
            
            print(f"{name:<20} | {prec:.4f}    | {rec:.4f}    | {f1:.4f}    | {mean_err:.2f} px")
            
            prediction_summary[f"{name}_Precision_vs_Council"] = f"{prec:.4f}"
            prediction_summary[f"{name}_Recall_vs_Council"] = f"{rec:.4f}"
            prediction_summary[f"{name}_F1_vs_Council"] = f"{f1:.4f}"
            prediction_summary[f"{name}_Mean_Error_px_vs_Council"] = f"{mean_err:.2f}px"
            
        print("="*80)
        
        prediction_summary["Source"] = args.source
        prediction_summary["Total Frames Processed"] = frame_count
        prediction_summary["Average Inference FPS"] = f"{avg_fps:.2f}"
        prediction_summary["Total Session Time (s)"] = f"{total_session_time:.2f}"
        
        log_to_file(args.log, prediction_summary, "Prediction Mode")

if __name__ == '__main__':
    main()
