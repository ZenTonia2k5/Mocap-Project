# Verify CUDA availability
import torch
import os
import cv2
import time
import pandas
import numpy as np
from ultralytics import YOLO

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")

device = '0' if torch.cuda.is_available() else 'cpu'
print(f"--- Đang chạy trên: {torch.cuda.get_device_name(0) if device == '0' else 'CPU'} ---")
print(os.path.exists(r'D:/PoseEstimation/test.mp4'))
# 2. Load model YOLO26
model = YOLO('yolo26x-pose.pt')

# Source = 0 => Camera; Source = video.mp4 => video r'./test.mp4'
results = model.predict(source= 'D:/PoseEstimation/test.mp4', 
                        show=True,
                        conf=0.25,
                        device=device)

model.info()
print(f"Model Classes Trained: {model.names}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
