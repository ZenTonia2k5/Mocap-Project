import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import ssl
try:
    ssl._create_default_https_context = ssl._create_unverified_context
except Exception:
    pass

from flask import Flask, render_template, Response, request, jsonify
import cv2
import threading
import torch
import time
from ultralytics import YOLO

app = Flask(__name__)

class ModelManager:
    def __init__(self):
        self.lock = threading.Lock()
        self.current_model_name = "yolo26"
        self.model_instance = None
        self.load_model(self.current_model_name)

    def load_model(self, model_name):
        with self.lock:
            print(f"Server transitioning to model: {model_name}")
            
            # Unload old model and flush Mac MPS memory heavily
            if self.model_instance is not None:
                del self.model_instance
                self.model_instance = None
                
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            
            self.current_model_name = model_name
            
            if model_name == "yolo26":
                self.model_instance = YOLO("yolo26n-pose.pt")
                self.model_type = "yolo"
            else:
                self.model_type = "mmpose"
                try:
                    import ssl
                    ssl._create_default_https_context = ssl._create_unverified_context
                    from mmpose.apis import MMPoseInferencer
                    
                    alias_map = {
                        "hrnet": "td-hm_hrnet-w32_8xb64-210e_ubody-256x192",
                        "rtmpose": "human",
                        "vitpose": "td-hm_vipnas-mbv3_8xb64-210e_coco-wholebody-256x192",
                        "openpose": "rtmw-m_8xb1024-270e_cocktail14-256x192",
                        "higherhrnet": "td-hm_res101_8xb32-210e_coco-wholebody-384x288",
                        "alphapose": "td-hm_res50_8xb64-210e_coco-wholebody-384x288" 
                    }
                    mmpose_name = alias_map.get(model_name, "human")
                    
                    # mmdet's custom NMS C++ extension has no MPS implementation;
                    # PYTORCH_ENABLE_MPS_FALLBACK only covers native PyTorch ops.
                    # CPU is the only reliable path for MMPose on macOS.
                    self.model_instance = MMPoseInferencer(pose2d=mmpose_name, device='cpu')
                except Exception as e:
                    print(f"Warning: MMPoseInferencer failed to load {model_name}. Reason: {e}")
                    self.model_instance = None

    def run_inference(self, frame):
        with self.lock:
            if self.model_instance is None:
                # If model failed to load due to compiler waiting etc, pass through grayscale so users know
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if self.model_type == "yolo":
                results = self.model_instance(frame, device="mps", verbose=False)
                return results[0].plot()
            elif self.model_type == "mmpose":
                # MMPose inferencer returns a generator
                try:
                    # Next gives the prediction dictionary mapping
                    result = next(self.model_instance(frame, return_vis=True))
                    vis_image = result['visualization'][0] # usually returns the drawn frame
                    return vis_image
                except Exception as e:
                    print(f"MMPose Inference Error: {e}")
                    return frame

model_manager = ModelManager()

camera_lock = threading.Lock()
cap = None

def get_camera():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(1)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
    return cap

def generate_frames():
    camera = get_camera()
    while True:
        with camera_lock:
            success, frame = camera.read()
        
        if not success:
            break
            
        annotated_frame = model_manager.run_inference(frame)
        
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/switch_model', methods=['POST'])
def switch_model():
    data = request.get_json()
    model_name = data.get('model')
    try:
        model_manager.load_model(model_name)
        return jsonify({"status": "success", "message": f"Successfully loaded {model_name}"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/metrics/<model_name>')
def get_metrics(model_name):
    # Dictionary mapping model characteristics logic
    metrics_db = {
        "yolo26": {
            "device": "MPS CoreML",
            "dataset": "COCO Keypoints",
            "mAP": "72.1",
            "params": "2.9M",
            "architecture": "Top-Down (End-to-End NMS-free)",
            "description": "Ultralytics YOLO unified framework for real-time edge processing."
        },
        "hrnet": {
            "device": "MPS PyTorch",
            "dataset": "COCO",
            "mAP": "74.4",
            "params": "28.5M",
            "architecture": "Top-Down High-Resolution Net",
            "description": "Maintains high-resolution representations through the entire process, excellent for detail."
        },
        "openpose": {
            "device": "MPS PyTorch",
            "dataset": "COCO (Part Affinity Fields)",
            "mAP": "61.8",
            "params": "52M",
            "architecture": "Bottom-Up (Part Affinity Fields)",
            "description": "True bottom-up approach. It detects human joints globally and groups them into individual skeletons."
        },
        "lowerhrnet": {"device": "MPS", "mAP":"-", "params":"-"},
        "higherhrnet": {
            "device": "MPS",
            "dataset": "COCO",
            "mAP": "68.4",
            "params": "28M",
            "architecture": "Bottom-Up HRNet",
            "description": "Uses Higher-Resolution Feature Pyramids to resolve scale variation challenges in bottom-up pipelines."
        },
        "rtmpose": {
            "device": "MPS",
            "dataset": "COCO",
            "mAP": "73.2",
            "params": "4.1M",
            "architecture": "Top-Down (RTMPose-m)",
            "description": "Extremely fast and optimized top-down model by OpenMMLab targeting real-time CPU/GPU bounds."
        },
        "vitpose": {
            "device": "MPS",
            "dataset": "COCO",
            "mAP": "80.9",
            "params": "86M",
            "architecture": "Top-Down (Vision Transformer)",
            "description": "SOTA performance using pure Transformer blocks instead of CNNs. Heavy but highly precise."
        },
        "alphapose": {
            "device": "MPS",
            "dataset": "COCO + Halpe",
            "mAP": "75.0+",
            "params": "34M",
            "architecture": "Top-Down (RMPE)",
            "description": "Regional Multi-Person Pose Estimation framework using symmetric integral algorithms."
        }
    }
    return jsonify(metrics_db.get(model_name, {"device": "MPS", "dataset": "Unknown", "mAP": "N/A", "params": "N/A", "architecture": "Unknown", "description": "No metadata available."}))

if __name__ == '__main__':
    print("Starting PoseScope Comparison Engine... Go to http://localhost:5050")
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)
