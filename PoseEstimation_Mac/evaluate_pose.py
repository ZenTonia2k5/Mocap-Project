from ultralytics import YOLO

def main():
    # Initialize the YOLOv26 Pose model
    print("Loading YOLOv26n-pose model...")
    model = YOLO("yolo26n-pose.pt")

    # Evaluate the model on the small coco8-pose dataset using Metal (MPS)
    print("Evaluating model with MPS backend...")
    metrics = model.val(data="coco8-pose.yaml", device="mps")

    # Extract and report metrics
    print("\n" + "="*50)
    print("EVALUATION METRICS REPORT")
    print("="*50)
    
    # Bounding Box metrics
    print("\n[Bounding Box Metrics]")
    print(f"mAP@50-95: {metrics.box.map:.4f}")
    print(f"mAP@50:    {metrics.box.map50:.4f}")
    print(f"mAP@75:    {metrics.box.map75:.4f}")
    
    # Pose / Keypoint metrics
    print("\n[Keypoint Metrics]")
    print(f"mAP@50-95: {metrics.pose.map:.4f}")
    print(f"mAP@50:    {metrics.pose.map50:.4f}")
    print(f"mAP@75:    {metrics.pose.map75:.4f}")

    print("\nDone!")

if __name__ == "__main__":
    main()
