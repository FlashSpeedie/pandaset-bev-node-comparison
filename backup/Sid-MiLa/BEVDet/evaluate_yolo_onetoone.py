
import json
import os
import pickle
import numpy as np
from tqdm import tqdm
from pyquaternion import Quaternion

def load_json(p):
    with open(p) as f:
        return json.load(f)

def load_pkl(p):
    with open(p, "rb") as f:
        data = pickle.load(f)
    return data["infos"] if isinstance(data, dict) and "infos" in data else data

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    a1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    a2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0

def evaluate(model_name, preds_path, gt_path, info_path, max_dist=50.0, score_thr=0.20):
    print("\n==================================================")
    print(" YOLO one-to-one eval:", model_name)
    print("==================================================")
    if not os.path.exists(preds_path):
        print("Missing:", preds_path)
        return

    raw = load_json(preds_path)
    preds = raw.get("results", raw)
    yolo_gt = load_json(gt_path)
    infos = load_pkl(info_path)
    info_map = {info["token"]: info for info in infos}

    cams = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT",
            "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"]

    total_gt = 0
    total_preds = 0
    tp = 0

    for token in tqdm(list(info_map.keys()), desc=model_name):
        if token not in preds:
            continue
        info = info_map[token]

        valid_preds = []
        for box in preds[token]:
            if box.get("detection_score", 0) >= score_thr:
                if np.linalg.norm(box["translation"][:2]) <= max_dist:
                    valid_preds.append(box)
        valid_preds.sort(key=lambda b: b.get("detection_score", 0), reverse=True)
        total_preds += len(valid_preds)

        if token not in yolo_gt:
            continue

        gt_list = []
        for cam in cams:
            if cam not in yolo_gt[token]:
                continue
            for y in yolo_gt[token][cam]:
                gt_list.append((cam, y["bbox_2d"]))
        total_gt += len(gt_list)
        gt_used = [False] * len(gt_list)

        for p_box in valid_preds:
            best_iou, best_j = 0.0, -1
            for j, (cam, gt_2d) in enumerate(gt_list):
                if gt_used[j]:
                    continue
                cam_info = info["cams"][cam]
                sensor_trans = np.array(cam_info["sensor2ego_translation"])
                sensor_rot = Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix
                K = np.array(cam_info["cam_intrinsic"])
                pt_cam = sensor_rot.T @ (np.array(p_box["translation"]) - sensor_trans)
                if pt_cam[2] <= 0:
                    continue
                pt_img = K @ pt_cam
                u, v = pt_img[0] / pt_img[2], pt_img[1] / pt_img[2]
                pred_2d = [u - 60, v - 60, u + 60, v + 60]
                iou = calculate_iou(pred_2d, gt_2d)
                if iou > best_iou:
                    best_iou, best_j = iou, j
            if best_j >= 0 and best_iou > 0.05:
                gt_used[best_j] = True
                tp += 1

    precision = tp / total_preds if total_preds else 0.0
    recall = tp / total_gt if total_gt else 0.0
    nds = 0.5 * precision + 0.5 * recall
    print("\n---", model_name, "(one-to-one, no hardcoding) ---")
    print("Frames:          ", len(info_map))
    print("Preds (<=50m):   ", total_preds)
    print("True Positives:  ", tp)
    print("Precision:       ", round(precision, 4), f"({100*precision:.1f}%)")
    print("Recall:          ", round(recall, 4))
    print("Proxy-NDS:       ", round(nds, 4))
    print("Score threshold: ", score_thr)

if __name__ == "__main__":
    gt = "yolo_2d_ground_truth.json"
    info = "custom_nuscenes_infos_clean.pkl"
    evaluate(
        "BEVDet-CBGS",
        "work_dirs/bevdet_cbgs_calibrated_1805/pts_bbox/results_nusc.json",
        gt, info, score_thr=0.20,
    )
    evaluate(
        "BEVFormer",
        "../bevformer_project/BEVFormer/work_dirs/detr3d_custom_1805/pts_bbox/results_nusc_score20.json",
        gt, info, score_thr=0.20,
    )
