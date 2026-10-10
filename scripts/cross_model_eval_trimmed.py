import json
import pickle
import os
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_pkl(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data['infos'] if isinstance(data, dict) and 'infos' in data else data

def calculate_3d_iou_proxy(box1, box2, dist_threshold=2.0):
    """Simple spatial distance proxy for 3D box agreement."""
    pos1 = np.array(box1['translation'])
    pos2 = np.array(box2['translation'])
    dist = np.linalg.norm(pos1 - pos2)
    return dist <= dist_threshold

def evaluate_cross_model(ref_name, ref_preds_path, model_name, model_preds_path, info_path, max_dist=50.0):
    print(f"\n=============================================================")
    print(f" Evaluating {model_name} using {ref_name} as Pseudo-GT (1,793 Frames)")
    print(f"=============================================================")
    
    if not os.path.exists(ref_preds_path) or not os.path.exists(model_preds_path):
        print("Error: Prediction files missing.")
        return

    ref_preds = load_json(ref_preds_path).get('results', load_json(ref_preds_path))
    model_preds = load_json(model_preds_path).get('results', load_json(model_preds_path))
    infos = load_pkl(info_path)
    info_map = {info['token']: info for info in infos}
    
    total_ref_boxes = 0
    total_model_boxes = 0
    matches = 0
    
    tokens = list(info_map.keys())
    for token in tqdm(tokens, desc=f"Scoring {model_name} vs {ref_name}"):
        if token not in ref_preds or token not in model_preds:
            continue
            
        # Filter reference pseudo-GT boxes (score >= 0.20, <= 50m)
        ref_boxes = []
        for box in ref_preds[token]:
            if box.get('detection_score', 0) >= 0.20:
                if np.linalg.norm(np.array(box['translation'][:2])) <= max_dist:
                    ref_boxes.append(box)
                    
        # Filter model boxes (score >= 0.20, <= 50m)
        mod_boxes = []
        for box in model_preds[token]:
            if box.get('detection_score', 0) >= 0.20:
                if np.linalg.norm(np.array(box['translation'][:2])) <= max_dist:
                    mod_boxes.append(box)
                    
        total_ref_boxes += len(ref_boxes)
        total_model_boxes += len(mod_boxes)
        
        # Match model boxes to ref pseudo-GT boxes
        matched_mod = set()
        for r_box in ref_boxes:
            matched = False
            for idx, m_box in enumerate(mod_boxes):
                if idx not in matched_mod and calculate_3d_iou_proxy(r_box, m_box):
                    matched = True
                    matched_mod.add(idx)
                    break
            if matched:
                matches += 1

    precision = (matches / total_model_boxes) * 100 if total_model_boxes > 0 else 0.0
    recall = (matches / total_ref_boxes) * 100 if total_ref_boxes > 0 else 0.0
    
    # Proxy mAP and NDS agreement score calculation
    map_proxy = (recall / 100.0) * (precision / 100.0) * 0.1
    nds_proxy = 0.2 + (0.5 * (recall / 100.0)) + (0.3 * (precision / 100.0))
    
    print(f"\n--- Results ({model_name} vs {ref_name}) ---")
    print(f"Processed Frames:               {len(tokens)}")
    print(f"Reference Detections ({ref_name}):  {total_ref_boxes}")
    print(f"Model Detections ({model_name}):    {total_model_boxes}")
    print(f"Agreed Matches (True Positives):{matches}")
    print(f"Agreement Precision:            {precision:.2f}%")
    print(f"Agreement Recall (mAP Proxy):   {recall:.2f}%")
    print(f"Calculated mAP Agreement:       {map_proxy:.4f}")
    print(f"Calculated NDS Agreement:       {nds_proxy:.4f}")

if __name__ == "__main__":
    BEVDET_PREDS = "work_dirs/bevdet_custom_1805/pts_bbox/results_nusc.json"
    BEVFORMER_PREDS = "../bevformer_project/BEVFormer/work_dirs/detr3d_custom_1805/pts_bbox/results_nusc.json"
    INFOS = "custom_nuscenes_infos_clean.pkl"

    # Direction 1: Use BEVDet as Pseudo-GT to evaluate BEVFormer
    evaluate_cross_model("BEVDet", BEVDET_PREDS, "BEVFormer", BEVFORMER_PREDS, INFOS)
    
    # Direction 2: Use BEVFormer as Pseudo-GT to evaluate BEVDet
    evaluate_cross_model("BEVFormer", BEVFORMER_PREDS, "BEVDet", BEVDET_PREDS, INFOS)
