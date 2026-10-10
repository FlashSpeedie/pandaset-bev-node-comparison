import json
import pickle
import os
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

def evaluate_model(model_name, preds_path, gt_path, info_path, max_dist=50.0):
    print(f"\n==========================================")
    print(f" Evaluating: {model_name} (Range <= {max_dist}m)")
    print(f"==========================================")
    
    if not os.path.exists(preds_path):
        print(f"Prediction file not found: {preds_path}")
        return

    with open(preds_path, 'r') as f:
        preds_data = json.load(f)
    preds = preds_data.get('results', preds_data)
    
    with open(gt_path, 'r') as f:
        yolo_gt = json.load(f)
        
    with open(info_path, 'rb') as f:
        data = pickle.load(f)
    infos = data['infos'] if isinstance(data, dict) and 'infos' in data else data
    info_map = {info['token']: info for info in infos}
    
    cams = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT', 
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
            
    total_matches = 0
    total_preds_evaluated = 0

    for token, gt_cams in tqdm(yolo_gt.items(), desc=f"Evaluating {model_name}"):
        if token not in preds or token not in info_map:
            continue
            
        info = info_map[token]
        
        # Filter 3D boxes within 50m and score >= 0.25
        valid_preds = []
        for box in preds[token]:
            if box.get('detection_score', 0) >= 0.25:
                dist = np.linalg.norm(box['translation'][:2])
                if dist <= max_dist:
                    valid_preds.append(box)
                    
        total_preds_evaluated += len(valid_preds)
        
        for cam in cams:
            if cam not in gt_cams:
                continue
                
            cam_info = info['cams'][cam]
            sensor_trans = np.array(cam_info['sensor2ego_translation'])
            sensor_rot = Quaternion(cam_info['sensor2ego_rotation']).rotation_matrix
            K = np.array(cam_info['cam_intrinsic'])
            
            yolo_boxes = gt_cams[cam]
            
            for p_box in valid_preds:
                # Project 3D prediction to 2D
                pt_cam = sensor_rot.T @ (np.array(p_box['translation']) - sensor_trans)
                if pt_cam[2] <= 0:
                    continue
                pt_img = K @ pt_cam
                u, v = pt_img[0] / pt_img[2], pt_img[1] / pt_img[2]
                
                # Check if this prediction falls inside any YOLO 2D box for this camera
                for y_box in yolo_boxes:
                    x1, y1, x2, y2 = y_box['bbox_2d']
                    pad = 15 # pixel tolerance
                    if (x1 - pad) <= u <= (x2 + pad) and (y1 - pad) <= v <= (y2 + pad):
                        total_matches += 1
                        break

    print(f"\n--- {model_name} Final Summary ---")
    print(f"Evaluated 3D Predictions (<= {max_dist}m): {total_preds_evaluated}")
    print(f"Successfully Matched YOLO 2D Boxes:     {total_matches}")
    if total_preds_evaluated > 0:
        precision_proxy = (total_matches / total_preds_evaluated) * 100
        print(f"Prediction Alignment Accuracy:          {precision_proxy:.2f}%")

if __name__ == "__main__":
    evaluate_model("BEVDet", "work_dirs/bevdet_custom_1805/pts_bbox/results_nusc.json", 
                   "yolo_2d_ground_truth.json", "custom_nuscenes_infos_clean.pkl")
                   
    evaluate_model("BEVFormer", "../bevformer_project/BEVFormer/work_dirs/detr3d_custom_1805/pts_bbox/results_nusc.json", 
                   "yolo_2d_ground_truth.json", "custom_nuscenes_infos_clean.pkl")
