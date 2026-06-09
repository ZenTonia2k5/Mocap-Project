# Verify CUDA availability and load YOLO Pose Estimation
import torch
import os
import cv2
import time
import argparse
import numpy as np
from datetime import datetime
from ultralytics import YOLO

def log_to_file(filepath,model, mode, summary_dict, device_name, cuda_available):
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
    #argparse
    parser = argparse.ArgumentParser(description="YOLO Pose Estimation with FPS and Metrics Logging")
    parser.add_argument("--mode", type=str, default="val", choices=["predict", "val"], 
                        help="Run mode: 'predict' for live stream/video or 'val' for dataset validation")
    parser.add_argument("--source", type=str, default="0", 
                        help="Path to video file or camera index (e.g., '0') for prediction")
    parser.add_argument("--data", type=str, default="my_coco8-pose.yaml", 
                        help="Path to dataset.yaml (required for validation mode)")
    parser.add_argument("--log", type=str, default="inference_log.txt", 
                        help="Path to the log file")
    parser.add_argument("--conf", type=float, default=0.25, 
                        help="Confidence threshold for prediction")
    parser.add_argument("--no-show", action="store_true", 
                        help="Disable displaying the GUI video stream window")
    parser.add_argument("--model", type=str, default='yolo26x', choices=['yolo26n', 'yolo26s', 'yolo26m', 'yolo26l', 'yolo26x'],
                        help="Choose model")
    args = parser.parse_args()

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"
    device = '0' if cuda_available else 'cpu'

    print(f"CUDA available: {cuda_available}")
    print(f"Device name: {device_name}")
    print(f"--- Running on: {device_name} ---")

    #Load YOLO model
    if args.model =="yolo26n":
        model_path = 'yolo26n-pose.pt'
    elif args.model =="yolo26s":
        model_path = 'yolo26s-pose.pt'
    elif args.model =="yolo26m":
        model_path = 'yolo26m-pose.pt'
    elif args.model =="yolo26l":
        model_path = 'yolo26l-pose.pt'
    elif args.model =="yolo26x":
        model_path = 'yolo26x-pose.pt'
    if not os.path.exists(model_path):
        print(f"Warning: Model file '{model_path}' not found. Downloading default...")
    model = YOLO(model_path)

    if args.mode == "val":
        if args.data is None:
            raise ValueError("Error: Validation mode requires a dataset YAML file path via the '--data' argument.")
        
        print(f"--- Running Validation on {args.data} ---")
        
        results = model.val(data=args.data, device=device, plots=True)
        
        r_dict = results.results_dict

        precision_pose = r_dict.get('metrics/precision(P)', r_dict.get('metrics/precision(pose)', 0.0))
        recall_pose = r_dict.get('metrics/recall(P)', r_dict.get('metrics/recall(pose)', 0.0))
        map50_pose = r_dict.get('metrics/mAP50(P)', r_dict.get('metrics/mAP50(pose)', 0.0))
        map50_95_pose = r_dict.get('metrics/mAP50-95(P)', r_dict.get('metrics/mAP50-95(pose)', 0.0))
        
        precision_box = r_dict.get('metrics/precision(B)', 0.0)
        recall_box = r_dict.get('metrics/recall(B)', 0.0)
        map50_box = r_dict.get('metrics/mAP50(B)', 0.0)
        map50_95_box = r_dict.get('metrics/mAP50-95(B)', 0.0)
        
        # Calculate F1 scores
        f1_pose = (2 * precision_pose * recall_pose / (precision_pose + recall_pose)) if (precision_pose + recall_pose) > 0 else 0.0
        f1_box = (2 * precision_box * recall_box / (precision_box + recall_box)) if (precision_box + recall_box) > 0 else 0.0
        
        metrics_summary = {
            "Dataset": args.data,
            "Pose Precision": f"{precision_pose:.4f}",
            "Pose Recall": f"{recall_pose:.4f}",
            "Pose F1-Score": f"{f1_pose:.4f}",
            "Pose mAP50": f"{map50_pose:.4f}",
            "Pose mAP50-95": f"{map50_95_pose:.4f}",
            "Box Precision": f"{precision_box:.4f}",
            "Box Recall": f"{recall_box:.4f}",
            "Box F1-Score": f"{f1_box:.4f}",
            "Box mAP50": f"{map50_box:.4f}",
            "Box mAP50-95": f"{map50_95_box:.4f}",
        }
        
        log_to_file(args.log, args.model, "Validation", metrics_summary, device_name, cuda_available)

    else:
        source_val = args.source
        if source_val.isdigit():
            source_val = int(source_val)
        
        print(f"--- Running Prediction on Source: {source_val} ---")
        
        cap = cv2.VideoCapture(source_val)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{source_val}'")
            exit(1)
            
        prev_time = 0
        fps_list = []
        conf_list = []
        detections_count = 0
        frame_count = 0
        
        if not args.no_show:
            cv2.namedWindow('YOLO Pose Estimation Stream', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('YOLO Pose Estimation Stream', 1280, 720)
            print("Press 'q' in the window to quit...")
        else:
            print("Running prediction in headless mode (no GUI window)...")
        
        start_session_time = time.time()
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
                
            frame_count += 1
            
            t_start = time.time()
            
            results = model.predict(source=frame, conf=args.conf, device=device, verbose=False)
            
            t_end = time.time()
            inference_time = t_end - t_start
            fps = 1.0 / inference_time if inference_time > 0 else 0.0
            fps_list.append(fps)
            
            result = results[0]
            num_detections = len(result.boxes) if result.boxes is not None else 0
            detections_count += num_detections
            
            if num_detections > 0:
                confs = result.boxes.conf.cpu().numpy()
                conf_list.extend(confs)
                
            annotated_frame = result.plot()
            
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if not args.no_show:
                cv2.imshow('YOLO Pose Estimation Stream', annotated_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c') or key == ord('C'):
                    capture_dir = os.path.join("runs", "captures")
                    os.makedirs(capture_dir, exist_ok=True)
                    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                    capture_path = os.path.join(capture_dir, f"capture_{timestamp_str}.jpg")
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
        avg_conf = np.mean(conf_list) if conf_list else 0.0
        
        prediction_summary = {
            "Source": args.source,
            "Total Frames Processed": frame_count,
            "Total Poses Detected": detections_count,
            "Average Inference FPS": f"{avg_fps:.2f}",
            "Average Confidence": f"{avg_conf:.4f}",
            "Total Execution Time (s)": f"{total_session_time:.2f}"
        }
        
        log_to_file(args.log, args.model, "Prediction", prediction_summary, device_name, cuda_available)

if __name__ == '__main__':
    main()
