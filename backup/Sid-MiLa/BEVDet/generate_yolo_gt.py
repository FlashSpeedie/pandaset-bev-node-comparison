import pickle
import json
import cv2
import os
from ultralytics import YOLO
from tqdm import tqdm

# 1. Load the heaviest, most accurate YOLO model (it will auto-download)
print("Loading YOLOv8x model...")
model = YOLO('yolov8x.pt') 

# YOLOv8 (COCO dataset) class mapping to our target vehicle/pedestrian classes
# 0: person, 2: car, 3: motorcycle, 5: bus, 7: truck
TARGET_CLASSES = {0: 'pedestrian', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

# 2. Load your custom dataset infos
info_path = 'custom_nuscenes_infos_clean.pkl'
print(f"Loading dataset infos from {info_path}...")
with open(info_path, 'rb') as f:
    data = pickle.load(f)

infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data

# 3. Process frames and cameras
yolo_ground_truth = {}
cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

print("Running YOLO inference on all cameras across 1805 frames...")
for info in tqdm(infos):
    token = info['token']
    yolo_ground_truth[token] = {}
    
    for cam in cams:
        # Get the image path
        img_path = info['cams'][cam]['data_path']
        
        # Make sure path exists (adjust prefix if necessary based on your data_root)
        if not os.path.exists(img_path):
            img_path = os.path.join('data/nuscenes', img_path)
            
        if not os.path.exists(img_path):
            print(f"Warning: Could not find image {img_path}")
            continue

        # Run YOLO inference (conf=0.25 filters out background noise)
        results = model(img_path, conf=0.25, verbose=False)
        
        boxes = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0].item())
                
                # Only keep vehicles and pedestrians
                if cls_id in TARGET_CLASSES:
                    class_name = TARGET_CLASSES[cls_id]
                    # Get box coordinates [x1, y1, x2, y2]
                    xyxy = box.xyxy[0].tolist() 
                    score = box.conf[0].item()
                    
                    boxes.append({
                        'class': class_name,
                        'bbox_2d': xyxy,
                        'score': score
                    })
                    
        yolo_ground_truth[token][cam] = boxes

# 4. Save the YOLO Ground Truth
output_file = 'yolo_2d_ground_truth.json'
with open(output_file, 'w') as f:
    json.dump(yolo_ground_truth, f, indent=4)

print(f"\nSuccess! YOLO 2D ground truth saved to {output_file}")
