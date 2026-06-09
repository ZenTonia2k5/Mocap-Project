# OpenMMLab MMPose Estimation with FPS and Metrics Logging
import cv2
import torch
import time
import os
import argparse
import numpy as np
from datetime import datetime
from mmpose.apis import MMPoseInferencer

# Helper function to append logs
def log_to_file(filepath, model, mode, summary_dict, device_name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(f"\n========================================\n")
        f.write(f"Model: {model}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Device: {device_name}\n")
        for k, v in summary_dict.items():
            f.write(f"{k}: {v}\n")
        f.write(f"========================================\n")
    print(f"Metrics saved to log file: {os.path.abspath(filepath)}")

def main():
    # 1. Setup argparse for flexible configurations
    parser = argparse.ArgumentParser(description="MMPose Estimation with FPS and Metrics Logging")
    parser.add_argument("--model", type=str, default="vitpose-b",
                        choices=["human", "vitpose-s", "vitpose-b", "vitpose-l", "rtmpose-s", "rtmpose-m", "rtmpose-l", "hrnet"],
                        help="Choose MMPose model alias (e.g. 'human', 'vitpose-b', 'rtmpose-m', 'hrnet')")
    parser.add_argument("--source", type=str, default="0", 
                        help="Path to video file or camera index (e.g., '0') for prediction")
    parser.add_argument("--log", type=str, default="inference_log.txt", 
                        help="Path to the log file")
    parser.add_argument("--no-show", action="store_true", 
                        help="Disable displaying the GUI video stream window")
    args = parser.parse_args()

    # 2. CUDA Device check
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device name: {device_name}")
    print(f"--- Running on: {device_name} ---")

    # 3. Load MMPose model
    print(f"Loading MMPose model: {args.model}...")
    model_name = args.model
    if model_name == "hrnet":
        model_name = "td-hm_hrnet-w32_8xb64-210e_coco-256x192"
    inferencer = MMPoseInferencer(pose2d=model_name, device=device)

    # 4. Open video source
    source_val = args.source
    if source_val.isdigit():
        source_val = int(source_val)
        
    print(f"--- Running Prediction on Source: {source_val} ---")
    cap = cv2.VideoCapture(source_val)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source_val}'")
        exit(1)

    # Variables for FPS calculation and statistics
    prev_time = 0
    fps_list = []
    conf_list = []
    detections_count = 0
    frame_count = 0
    
    if not args.no_show:
        cv2.namedWindow('Pose Stream', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Pose Stream', 1280, 720)
        print("Press 'q' to quit, 'c' to capture a frame...")
    else:
        print("Running prediction in headless mode (no GUI window)...")
        
    start_session_time = time.time()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        frame_count += 1
        
        # Calculate inference FPS
        t_start = time.time()
        
        # 5. Run prediction
        # MMPoseInferencer yields results via generator
        result_it = inferencer(inputs=frame, return_vis=True)
        result = next(result_it)
        
        t_end = time.time()
        inference_time = t_end - t_start
        fps = 1.0 / inference_time if inference_time > 0 else 0.0
        fps_list.append(fps)
        
        # Get frame count & confidences
        # Structure: result['predictions'] = [[dict_instance_1, dict_instance_2, ...]]
        predictions = result.get('predictions', [[]])[0]
        num_detections = len(predictions)
        detections_count += num_detections
        
        for p in predictions:
            if 'keypoint_scores' in p:
                conf_list.extend(p['keypoint_scores'])
                
        # Lấy tấm ảnh đã được vẽ khung xương từ AI (in RGB format)
        viz_frame = result['visualization'][0]
        
        # Overlay current FPS on frame
        cv2.putText(viz_frame, f"FPS: {fps:.1f}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Convert RGB to BGR for OpenCV display
        viz_frame_bgr = cv2.cvtColor(viz_frame, cv2.COLOR_RGB2BGR)

        # Display the frame if enabled
        if not args.no_show:
            cv2.imshow('Pose Stream', viz_frame_bgr)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c') or key == ord('C'):
                # Capture and save frame
                capture_dir = os.path.join("runs", "captures")
                os.makedirs(capture_dir, exist_ok=True)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
                capture_path = os.path.join(capture_dir, f"mmpose_capture_{timestamp_str}.jpg")
                cv2.imwrite(capture_path, viz_frame_bgr)
                print(f"Captured and saved frame to: {os.path.abspath(capture_path)}")
        else:
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames... Current FPS: {fps:.1f}")

    total_session_time = time.time() - start_session_time
    cap.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    # Calculate averages
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

    # Save prediction statistics
    log_to_file(args.log, f"mmpose-{args.model}", "Prediction", prediction_summary, device_name)

if __name__ == '__main__':
    main()