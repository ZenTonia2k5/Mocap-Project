import cv2
import torch
import time
import os
import argparse
import numpy as np
import yaml
from datetime import datetime
from controlnet_aux import OpenposeDetector
from controlnet_aux.open_pose import draw_poses

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

def log_to_file(filepath, model, mode, summary_dict, device_name, cuda_available):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n========================================\n")
        f.write(f"Model: {model}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Device: {device_name} (CUDA={cuda_available})\n")
        for k, v in summary_dict.items():
            f.write(f"{k}: {v}\n")
        f.write(f"========================================\n")
    print(f"Metrics saved to log file: {os.path.abspath(filepath)}")

def main():
    parser = argparse.ArgumentParser(description="OpenPose Estimation and Validation with FPS and Metrics Logging")
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
    parser.add_argument("--hand", action="store_true", 
                        help="Include hand keypoints in detection")
    parser.add_argument("--face", action="store_true", 
                        help="Include face keypoints in detection")
    parser.add_argument("--max-frames", type=int, default=None, 
                        help="Limit the number of frames to process (useful for testing)")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    device = "cuda" if cuda_available else "cpu"
    
    print(f"CUDA available: {cuda_available}")
    print(f"Device name: {device_name}")
    print(f"--- Running on: {device_name} ---")

    print("Loading PyTorch OpenPose model...")
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(device)

    if args.mode == "val":
        if args.data is None:
            raise ValueError("Error: Validation mode requires a dataset YAML file path via the '--data' argument.")
        
        print(f"--- Running Validation on {args.data} ---")

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

        val_out_base = os.path.join("runs", "pose")
        os.makedirs(val_out_base, exist_ok=True)
        val_idx = 1
        while True:
            suffix = "" if val_idx == 1 else f"-{val_idx}"
            val_out_dir = os.path.join(val_out_base, f"openpose_val{suffix}")
            if not os.path.exists(val_out_dir):
                break
            val_idx += 1
        os.makedirs(val_out_dir, exist_ok=True)
        print(f"Saving validation images to: {os.path.abspath(val_out_dir)}")
        
        thresholds = np.arange(0.5, 1.0, 0.05)
        
        pose_tp_list = {t: [] for t in thresholds}
        pose_conf_list = {t: [] for t in thresholds}
        
        total_num_gt = 0

        openpose_to_coco = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]
        
        COCO_CONNECTIONS = [
            (0, 1), (1, 3), (0, 2), (2, 4),  
            (5, 6),                          
            (5, 7), (7, 9),                 
            (6, 8), (8, 10),               
            (11, 12),                      
            (5, 11), (6, 12),               
            (11, 13), (13, 15),            
            (12, 14), (14, 16)             
        ]
        
        COCO_K = np.array([
            0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
        ])
        
        start_val_time = time.time()
        
        for img_idx, img_file in enumerate(image_files):
            img_path = os.path.join(val_images_dir, img_file)
            frame = cv2.imread(img_path)
            if frame is None:
                continue
                
            H, W, _ = frame.shape

            label_file = os.path.splitext(img_file)[0] + ".txt"
            label_path = os.path.join(val_labels_dir, label_file)
            gt_poses = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as lf:
                    for line in lf:
                        parts = list(map(float, line.strip().split()))
                        if not parts or int(parts[0]) != 0:
                            continue
                        bbox = parts[1:5] 
                        kpts = parts[5:]  
                        kpts = np.array(kpts).reshape(17, 3)
                        gt_poses.append({
                            "bbox": bbox,
                            "keypoints": kpts
                        })
            
            total_num_gt += len(gt_poses)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            poses = openpose.detect_poses(rgb_frame, include_hand=args.hand, include_face=args.face)
            
            pred_poses = []
            for pose in poses:
                body_kpts = pose.body.keypoints  
                coco_kpts = []
                valid_count = 0
                for coco_idx in range(17):
                    op_idx = openpose_to_coco[coco_idx]
                    kp = body_kpts[op_idx]
                    if kp is not None:
                        coco_kpts.append([kp.x, kp.y, 1.0])
                        valid_count += 1
                    else:
                        coco_kpts.append([0.0, 0.0, 0.0])
                
                coco_kpts = np.array(coco_kpts)
                conf = valid_count / 17.0 if valid_count > 0 else 0.0
                
                pred_poses.append({
                    "keypoints": coco_kpts,
                    "conf": conf
                })
            
            M = len(gt_poses)
            N = len(pred_poses)
            
            oks_matrix = np.zeros((M, N))
            for i in range(M):
                for j in range(N):
                    gt_kpts = gt_poses[i]["keypoints"]
                    bbox = gt_poses[i]["bbox"]
                    pred_kpts = pred_poses[j]["keypoints"]
                    
                    s = np.sqrt(bbox[2] * bbox[3])
                    if s <= 0:
                        s = 1e-5
                        
                    d_sq = (pred_kpts[:, 0] - gt_kpts[:, 0])**2 + (pred_kpts[:, 1] - gt_kpts[:, 1])**2
                    vis = gt_kpts[:, 2] > 0
                    
                    if not np.any(vis):
                        oks_matrix[i, j] = 0.0
                    else:
                        oks_vals = np.exp(-d_sq / (2 * (s**2) * (COCO_K**2)))
                        oks_matrix[i, j] = np.sum(oks_vals[vis]) / np.sum(vis)
            
            for t in thresholds:
                matched_gt = set()
                matched_pred = set()
                pairs = []
                for i in range(M):
                    for j in range(N):
                        if oks_matrix[i, j] >= t:
                            pairs.append((oks_matrix[i, j], i, j))
                pairs.sort(key=lambda x: x[0], reverse=True)
                
                tp_set = set()
                for OKS, i, j in pairs:
                    if i not in matched_gt and j not in matched_pred:
                        matched_gt.add(i)
                        matched_pred.add(j)
                        tp_set.add(j)
                
                for j in range(N):
                    if j in tp_set:
                        pose_tp_list[t].append(1)
                    else:
                        pose_tp_list[t].append(0)
                    pose_conf_list[t].append(pred_poses[j]["conf"])
            
            gt_img = frame.copy()
            for gt_pose in gt_poses:
                kpts = gt_pose["keypoints"]
                for p1, p2 in COCO_CONNECTIONS:
                    if kpts[p1, 2] > 0 and kpts[p2, 2] > 0:
                        pt1 = (int(kpts[p1, 0] * W), int(kpts[p1, 1] * H))
                        pt2 = (int(kpts[p2, 0] * W), int(kpts[p2, 1] * H))
                        cv2.line(gt_img, pt1, pt2, (0, 255, 255), 2)  
                for i in range(17):
                    if kpts[i, 2] > 0:
                        pt = (int(kpts[i, 0] * W), int(kpts[i, 1] * H))
                        cv2.circle(gt_img, pt, 5, (0, 0, 255), -1)  
            
            img_base = os.path.splitext(img_file)[0]
            cv2.imwrite(os.path.join(val_out_dir, f"val_{img_base}_labels.png"), gt_img)
            
            pred_img = frame.copy()
            skeleton_canvas = draw_poses(poses, H, W, draw_body=True, draw_hand=args.hand, draw_face=args.face)
            skeleton_canvas_bgr = cv2.cvtColor(skeleton_canvas, cv2.COLOR_RGB2BGR)
            mask = np.any(skeleton_canvas_bgr > 0, axis=-1)
            pred_img[mask] = skeleton_canvas_bgr[mask]
            cv2.imwrite(os.path.join(val_out_dir, f"val_{img_base}_pred.png"), pred_img)
            
            if (img_idx + 1) % 2 == 0 or (img_idx + 1) == len(image_files):
                print(f"Validated {img_idx + 1}/{len(image_files)} images...")
        
        pose_ap = {}
        for t in thresholds:
            pose_ap[t] = compute_ap(pose_tp_list[t], pose_conf_list[t], total_num_gt)
            
        pose_tp_50 = sum(pose_tp_list[0.50])
        pose_fp_50 = len(pose_tp_list[0.50]) - pose_tp_50
        pose_fn_50 = total_num_gt - pose_tp_50
        
        pose_precision_50 = pose_tp_50 / (pose_tp_50 + pose_fp_50) if (pose_tp_50 + pose_fp_50) > 0 else 0.0
        pose_recall_50 = pose_tp_50 / (pose_tp_50 + pose_fn_50) if (pose_tp_50 + pose_fn_50) > 0 else 0.0
        pose_f1_50 = (2 * pose_precision_50 * pose_recall_50) / (pose_precision_50 + pose_recall_50) if (pose_precision_50 + pose_recall_50) > 0 else 0.0
        
        pose_map50 = pose_ap[0.50]
        pose_map50_95 = np.mean([pose_ap[t] for t in thresholds])
        
        total_val_time = time.time() - start_val_time
        
        validation_summary = {
            "Dataset": args.data,
            "Total Images Validated": len(image_files),
            "Pose Precision": f"{pose_precision_50:.4f}",
            "Pose Recall": f"{pose_recall_50:.4f}",
            "Pose F1-Score": f"{pose_f1_50:.4f}",
            "Pose mAP50": f"{pose_map50:.4f}",
            "Pose mAP50-95": f"{pose_map50_95:.4f}",
            "Total Validation Time (s)": f"{total_val_time:.2f}"
        }
        
        print(f"\nResults saved to {os.path.abspath(val_out_dir)}")
        log_to_file(args.log, "openpose-controlnet", "Validation", validation_summary, device_name, cuda_available)
        return

    else:
        source_val = args.source
        if source_val.isdigit():
            source_val = int(source_val)
            
        print(f"--- Running OpenPose Prediction on Source: {source_val} ---")
        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{source_val}'")
            exit(1)

        prev_time = 0
        fps_list = []
        detections_count = 0
        frame_count = 0
        
        if not args.no_show:
            cv2.namedWindow('OpenPose Stream', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('OpenPose Stream', 1280, 720)
            print("Press 'q' to quit, 'c' to capture a frame...")
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
            H, W, C = frame.shape
            
            t_start = time.time()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            poses = openpose.detect_poses(rgb_frame, include_hand=args.hand, include_face=args.face)
            
            t_end = time.time()
            inference_time = t_end - t_start
            fps = 1.0 / inference_time if inference_time > 0 else 0.0
            fps_list.append(fps)

            num_detections = len(poses)
            detections_count += num_detections

            skeleton_canvas = draw_poses(poses, H, W, draw_body=True, draw_hand=args.hand, draw_face=args.face)

            skeleton_canvas_bgr = cv2.cvtColor(skeleton_canvas, cv2.COLOR_RGB2BGR)

            annotated_frame = frame.copy()
            mask = np.any(skeleton_canvas_bgr > 0, axis=-1)
            annotated_frame[mask] = skeleton_canvas_bgr[mask]
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            if not args.no_show:
                cv2.imshow('OpenPose Stream', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') or key == ord('C'):
                    capture_dir = os.path.join("runs", "captures")
                    os.makedirs(capture_dir, exist_ok=True)
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    capture_path = os.path.join(capture_dir, f"openpose_capture_{timestamp_str}.jpg")
                    cv2.imwrite(capture_path, annotated_frame)
                    print(f"Captured and saved frame to: {os.path.abspath(capture_path)}")
            else:
                if frame_count % 10 == 0:
                    print(f"Processed {frame_count} frames... Current FPS: {fps:.1f}")

        total_session_time = time.time() - start_session_time
        cap.release()
        if not args.no_show:
            cv2.destroyAllWindows()

        avg_fps = np.mean(fps_list) if fps_list else 0.0

        prediction_summary = {
            "Source": args.source,
            "Total Frames Processed": frame_count,
            "Total Poses Detected": detections_count,
            "Average Inference FPS": f"{avg_fps:.2f}",
            "Total Execution Time (s)": f"{total_session_time:.2f}"
        }

        log_to_file(args.log, "openpose-controlnet", "Prediction", prediction_summary, device_name, cuda_available)

if __name__ == '__main__':
    main()
