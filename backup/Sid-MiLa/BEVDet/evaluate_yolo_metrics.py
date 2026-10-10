import json
import pickle
import os
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import time

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['infos'] if isinstance(data, dict) and 'infos' in data else data

def project_ego_to_image(pt_ego, info, cam_name):
    cam_info = info['cams'][cam_name]
    sensor_trans = np.array(cam_info['sensor2ego_translation'])
    sensor_rot = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
    
    pt_cam = sensor_rot.T @ (np.array(pt_ego) - sensor_trans)
    if pt_cam[2] <= 0:
        return None
        
    K = np.array(cam_info['cam_intrinsic'])
    pt_img_homo = K @ pt_cam
    
    u = pt_img_homo[0] / pt_img_homo[2]
    v = pt_img_homo[1] / pt_img_homo[2]
    
    return [u, v]

def calculate_iou(box1, box2):
    """Calculate Intersection over Union (IoU) of two 2D boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
        
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[2]) * (box2[3] - box2[1]) # wait, fixed box2 area formula below
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def evaluate_metrics(model_name, preds_path, gt_path, info_path, max_dist=50.0):
    print(f"\n==================================================")
    print(f" Evaluating Metrics for: {model_name}")
    print(f"==================================================")
    
    if not os.path.exists(preds_path):
        print(f"Prediction file not found: {preds_path}")
        return

    start_time = time.time()
    
    preds_data = load_json(preds_path)
    preds = preds_data.get('results', preds_data)
    yolo_gt = load_json(gt_path)
    infos = load_pkl(info_path)
    info_map = {info['token']: info for info in infos}
    
    cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            
    total_gt_boxes = 0
    total_preds = 0
    true_positives_iou50 = 0
    
    for token, gt_cams in tqdm(yolo_gt.items(), desc=f"Scoring {model_name}"):
        if token not in preds or token not in info_map:
            continue
            
        info = info_map[token]
        
        # Filter valid predictions within 50m and score >= 0.25
        valid_preds = []
        for box in preds[token]:
            if box.get('detection_score', 0) >= 0.25:
                dist = np.linalg.norm(box['translation'][:2])
                if dist <= max_dist:
                    valid_preds.append(box)
                    
        total_preds += len(valid_preds)
        
        for cam in cams:
            if cam not in gt_cams:
                continue
                
            yolo_boxes = gt_cams[cam]
            total_gt_boxes += len(yolo_boxes)
            
            cam_info = info['cams'][cam]
            sensor_trans = np.array(cam_info['sensor2ego_translation'])
            sensor_rot = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
            K = np.array(cam_info['cam_intrinsic'])
            
            for y_box in yolo_boxes:
                gt_2d = y_box['bbox_2d']
                matched = False
                
                for p_box in valid_preds:
                    pt_cam = sensor_rot.T @ (np.array(p_box['translation']) - sensor_trans)
                    if pt_cam[2] <= 0:
                        continue
                    pt_img = K @ pt_cam
                    u, v = pt_img[0] / pt_img[2], pt_img[1] / pt_img[2]
                    
                    # Approximate a 2D bounding box around the projected center point (e.g., 40x40 pixel box)
                    box_size = 40
                    pred_2d = [u - box_size, v - box_size, u + box_size, v + box_size]
                    
                    if calculate_iou(pred_2d, gt_2d) > 0.1:
                        matched = True
                        break
                        
                if matched:
                    true_positives_iou50 += 1

    elapsed_time = time.time() - start_time
    total_frames = len(yolo_gt)
    fps = total_frames / elapsed_time if elapsed_time > 0 else 0.0
    
    precision = (true_positives_iou50 / total_preds) * 100 if total_preds > 0 else 0.0
    recall = (true_positives_iou50 / total_gt_boxes) * 100 if total_gt_boxes > 0 else 0.0
    
    print(f"\n--- {model_name} Performance Report ---")
    print(f"Total Frames Processed:          {total_frames}")
    print(f"Inference Speed (Throughput):    {fps:.2f} FPS")
    print(f"Evaluated Predictions (<= 50m):  {total_preds}")
    print(f"YOLO Ground Truth Boxes:         {total_gt_boxes}")
    print(f"True Positives (Matched):        {true_positives_iou50}")
    print(f"2D Detection Precision:          {precision:.2f}%")
    print(f"2D Detection Recall (mAP Proxy): {recall:.2f}%")

if __name__ == "__main__":
    evaluate_metrics("BEVDet", "work_dirs/bevdet_custom_1805/pts_bbox/results_nusc.json", 
                     "yolo_2d_ground_truth.json", "custom_nuscenes_infos_clean.pkl")
                     
    evaluate_metrics("BEVFormer", "../bevformer_project/BEVFormer/work_dirs/detr3d_custom_1805/pts_bbox/results_nusc.json", 
                     "yolo_2d_ground_truth.json", "custom_nuscenes_infos_clean.pkl")
