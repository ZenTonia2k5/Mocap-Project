import os
import shutil
from ultralytics import YOLO

# Load model
model = YOLO('yolo26x-pose.pt')
dataset_dir = r'D:\BKU\Mocap\PoseEstimation_Win\dataset\coco8-pose'

# Configure files to process
images = ["siu.jpg"]
modes = ["val"]

for img_name, mode in zip(images, modes):
    img_path = os.path.join(dataset_dir, img_name)
    if not os.path.exists(img_path):
        print(f"File not found: {img_path}")
        continue
        
    print(f"Processing: {img_path} for {mode}...")
    results = model(img_path)
    result = results[0]
    
    if len(result.boxes) == 0:
        print(f"No persons detected in {img_name}!")
        continue
        
    lines = []
    boxes = result.boxes.xywhn.cpu().numpy()
    
    if result.keypoints is not None:
        kpts = result.keypoints.xyn.cpu().numpy()
    else:
        kpts = None
        
    for i in range(len(boxes)):
        box = boxes[i]
        line_elements = [0] # class 0 (person)
        line_elements.extend(box)
        
        if kpts is not None and i < len(kpts):
            for j in range(17):
                x = kpts[i, j, 0]
                y = kpts[i, j, 1]
                # visibility: 2 if keypoint has valid coordinates, else 0
                v = 2 if (x > 0 or y > 0) else 0
                line_elements.extend([x, y, v])
                
        lines.append(" ".join(map(str, line_elements)))
        
    # Write label file
    base_name = os.path.splitext(img_name)[0]
    txt_name = base_name + ".txt"
    txt_path = os.path.join(dataset_dir, txt_name)
    
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"Created label file: {txt_path}")
    
    # Destination directories
    img_dest_dir = os.path.join(dataset_dir, 'images', mode)
    lbl_dest_dir = os.path.join(dataset_dir, 'labels', mode)
    
    os.makedirs(img_dest_dir, exist_ok=True)
    os.makedirs(lbl_dest_dir, exist_ok=True)
    
    # Move files
    shutil.move(img_path, os.path.join(img_dest_dir, img_name))
    shutil.move(txt_path, os.path.join(lbl_dest_dir, txt_name))
    print(f"Moved {img_name} to {img_dest_dir}")
    print(f"Moved {txt_name} to {lbl_dest_dir}")

# Clear stale cache files to force dataset scan update
cache_dir = os.path.join(dataset_dir, 'labels')
for cache_file in ['train.cache', 'val.cache']:
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"Removed cache file: {cache_path}")
