import json
import pickle
import cv2
import os
import numpy as np
from pyquaternion import Quaternion

print("Loading data...")
with open("yolo_2d_ground_truth.json", "r") as f:
    yolo_gt = json.load(f)

preds_path = "work_dirs/bevdet_custom_1805/pts_bbox/results_nusc.json"
with open(preds_path, "r") as f:
    preds = json.load(f)['results']

with open("custom_nuscenes_infos_clean.pkl", "rb") as f:
    infos = pickle.load(f)

if isinstance(infos, dict) and 'infos' in infos:
    infos = infos['infos']

def project_ego_to_image(pt_ego, info, cam_name):
    """
    Projects a 3D point from ego vehicle coordinates directly into 2D camera coordinates.
    """
    cam_info = info['cams'][cam_name]
    
    # 1. Ego to Camera Sensor Transformation
    sensor_trans = np.array(cam_info['sensor2ego_translation'])
    sensor_rot = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
    
    pt_cam = sensor_rot.T @ (np.array(pt_ego) - sensor_trans)
    
    # Check if point is behind the camera (nuScenes camera Z is forward)
    if pt_cam[2] <= 0:
        return None
        
    # 2. Camera Sensor to Image Plane (Intrinsics)
    K = np.array(cam_info['cam_intrinsic'])
    pt_img_homo = K @ pt_cam
    
    u = pt_img_homo[0] / pt_img_homo[2]
    v = pt_img_homo[1] / pt_img_homo[2]
    
    return int(u), int(v)

print("Searching for a frame with 3D predictions...")
found = False
cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

for info in infos:
    token = info['token']
    if token not in preds:
        continue
    
    pred_boxes = [p for p in preds[token] if p.get('detection_score', 0) > 0.25]
    if len(pred_boxes) == 0:
        continue
        
    for cam in cams:
        if token in yolo_gt and cam in yolo_gt[token] and len(yolo_gt[token][cam]) > 0:
            img_path = info['cams'][cam]['data_path']
            if not os.path.exists(img_path):
                img_path = os.path.join('data/nuscenes', img_path)
            
            img = cv2.imread(img_path)
            if img is None: continue
            
            # Draw YOLO (Green Boxes)
            for box in yolo_gt[token][cam]:
                x1, y1, x2, y2 = map(int, box['bbox_2d'])
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw 3D Model Centers (Red Dots)
            drawn_count = 0
            for p in pred_boxes:
                pt = project_ego_to_image(p['translation'], info, cam)
                if pt and 0 <= pt[0] < 1600 and 0 <= pt[1] < 1200:
                    cv2.circle(img, pt, 12, (0, 0, 255), -1)
                    drawn_count += 1
            
            if drawn_count > 0:
                cv2.imwrite("debug_projection.jpg", img)
                print(f"Success! Drew {drawn_count} red dots on token {token}, camera {cam}.")
                print("Open 'debug_projection.jpg' to check alignment.")
                found = True
                break
    if found:
        break

if not found:
    print("No 3D predictions projected inside camera bounds.")
