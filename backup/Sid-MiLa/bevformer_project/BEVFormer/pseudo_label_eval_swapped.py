#!/usr/bin/env python3
"""
pseudo_label_eval_swapped.py
Compute mAP/NDS for BEVFormer against BEVDet pseudo-labels as ground truth.

This is the SWAPPED version: BEVDet predictions (score >= 0.2) are used as
pseudo ground truth, and BEVFormer is scored against them.

Usage:
cd ~/Sid\ -\ MiLa/bevformer_project/BEVFormer
conda activate bevformer117
python pseudo_label_eval_swapped.py
"""

import json
import numpy as np
from collections import defaultdict

# =====================================================================
# CONFIG - paths (SWAPPED from original)
# =====================================================================
# BEVDet is now the pseudo-GT (reference)
BD_PATH = '/home/adas/Sid - MiLa/BEVDet/work_dirs/bevdet_custom_1805_calibrated/pts_bbox/results_nusc.json'

# BEVFormer is now the model being scored
BF_PATH = 'test/bevformer_no_temporal/Thu_Aug__6_13_37_55_2026/pts_bbox/results_nusc.json'

# Only use BEVDet predictions with score >= this as pseudo-GT
GT_SCORE_THRESH = 0.2

# =====================================================================
# nuScenes evaluation constants
# =====================================================================
CLASSES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle', 'barrier', 'traffic_cone'
]

# Distance thresholds for matching (meters) - nuScenes uses all 4
DIST_THRESHOLDS = [0.5, 1.0, 2.0, 4.0]

# Distance threshold for computing TP error metrics
ERROR_DIST_TH = 2.0

# NDS normalization thresholds
ATE_THRESH = 1.0       # meters
ASE_THRESH = 1.0       # 1 - 3D IoU (already in [0,1])
AOE_THRESH = np.pi / 2 # radians (90 degrees)
AVE_THRESH = 2.0       # m/s
AAE_THRESH = 1.0       # binary 0 or 1


def load_results(path):
    with open(path) as f:
        d = json.load(f)
    return d['results'] if 'results' in d else d


def center_distance(box1, box2):
    """BEV center distance (x-y plane)."""
    dx = box1['translation'][0] - box2['translation'][0]
    dy = box1['translation'][1] - box2['translation'][1]
    return np.sqrt(dx**2 + dy**2)


def quaternion_yaw(q):
    """Extract yaw angle from quaternion [w, x, y, z]."""
    w, x, y, z = q
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y**2 + z**2))


def angle_diff(a1, a2):
    """Smallest absolute angle difference."""
    d = a1 - a2
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return abs(d)


def scale_iou(box1, box2):
    """3D IoU based on size dimensions."""
    s1 = np.array(box1['size'])
    s2 = np.array(box2['size'])
    intersection = np.prod(np.minimum(s1, s2))
    union = np.prod(s1) + np.prod(s2) - intersection
    return intersection / max(union, 1e-8)


def evaluate_class(gt_by_sample, pred_by_sample, class_name, dist_th):
    """
    Evaluate one class at one distance threshold.
    Returns: (ap, tp_pairs, total_gt)
    """
    all_scores = []
    all_tp = []
    all_pairs = []
    total_gt = 0

    for sample_token in gt_by_sample:
        gt_dets = [d for d in gt_by_sample[sample_token] if d['detection_name'] == class_name]
        pred_dets = [d for d in pred_by_sample.get(sample_token, []) if d['detection_name'] == class_name]

        total_gt += len(gt_dets)

        # Sort predictions by score (highest first) for greedy matching
        pred_sorted = sorted(pred_dets, key=lambda x: -x['detection_score'])

        matched_gt = [False] * len(gt_dets)

        for pred in pred_sorted:
            best_dist = float('inf')
            best_gt_idx = -1

            for i, gt in enumerate(gt_dets):
                if matched_gt[i]:
                    continue
                dist = center_distance(pred, gt)
                if dist < best_dist:
                    best_dist = dist
                    best_gt_idx = i

            if best_gt_idx >= 0 and best_dist <= dist_th:
                all_scores.append(pred['detection_score'])
                all_tp.append(1)
                all_pairs.append((pred, gt_dets[best_gt_idx]))
                matched_gt[best_gt_idx] = True
            else:
                all_scores.append(pred['detection_score'])
                all_tp.append(0)
                all_pairs.append(None)

    # Compute AP using 41-point interpolation
    if total_gt == 0 or len(all_scores) == 0:
        return 0.0, [], total_gt

    sorted_idx = np.argsort(all_scores)[::-1]
    tp_sorted = np.array(all_tp)[sorted_idx]
    pairs_sorted = [all_pairs[i] for i in sorted_idx]

    tp_cumsum = np.cumsum(tp_sorted)
    fp_cumsum = np.cumsum(1 - tp_sorted)

    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
    recall = tp_cumsum / (total_gt + 1e-8)

    recall_ths = np.linspace(0, 1, 41)
    ap = 0.0
    for r in recall_ths:
        mask = recall >= r
        if np.any(mask):
            ap += np.max(precision[mask])
    ap /= 41.0

    tp_pairs = [p for p in pairs_sorted if p is not None]
    return ap, tp_pairs, total_gt


def compute_class_errors(tp_pairs):
    """Compute error metrics from true positive pairs."""
    if not tp_pairs:
        return None

    ate_list = []
    ase_list = []
    aoe_list = []
    ave_list = []
    aae_list = []

    for pred, gt in tp_pairs:
        # ATE: center translation error (3D Euclidean)
        te = np.sqrt(np.sum((np.array(pred['translation']) - np.array(gt['translation']))**2))
        ate_list.append(te)

        # ASE: 1 - 3D scale IoU
        iou = scale_iou(pred, gt)
        ase_list.append(1.0 - iou)

        # AOE: yaw angle difference
        pred_yaw = quaternion_yaw(pred['rotation'])
        gt_yaw = quaternion_yaw(gt['rotation'])
        aoe_list.append(angle_diff(pred_yaw, gt_yaw))

        # AVE: velocity error (2D)
        ve = np.sqrt(np.sum((np.array(pred['velocity']) - np.array(gt['velocity']))**2))
        ave_list.append(ve)

        # AAE: attribute classification error (0 = correct, 1 = wrong)
        pred_attr = pred.get('attribute_name', '')
        gt_attr = gt.get('attribute_name', '')
        aae_list.append(0.0 if pred_attr == gt_attr else 1.0)

    return {
        'mate': np.mean(ate_list),
        'mase': np.mean(ase_list),
        'maoe': np.mean(aoe_list),
        'mave': np.mean(ave_list),
        'maae': np.mean(aae_list),
        'n_tp': len(tp_pairs)
    }


def main():
    print("=" * 60)
    print("SWAPPED Pseudo-Label Evaluation: BEVFormer vs BEVDet pseudo-labels")
    print("=" * 60)

    # Load BEVDet predictions as pseudo-GT
    print("\nLoading BEVDet predictions as pseudo-GT...")
    bd_results = load_results(BD_PATH)

    gt_by_sample = defaultdict(list)
    total_gt = 0
    for token, dets in bd_results.items():
        filtered = [d for d in dets if d['detection_score'] >= GT_SCORE_THRESH]
        gt_by_sample[token] = filtered
        total_gt += len(filtered)
    print(f"  Pseudo-GT boxes (score >= {GT_SCORE_THRESH}): {total_gt}")
    print(f"  Samples with GT: {len(gt_by_sample)}")

    # Load BEVFormer predictions
    print("\nLoading BEVFormer predictions...")
    bf_results = load_results(BF_PATH)
    pred_by_sample = defaultdict(list)
    total_pred = 0
    for token, dets in bf_results.items():
        pred_by_sample[token] = dets
        total_pred += len(dets)
    print(f"  Prediction boxes: {total_pred}")
    print(f"  Samples with predictions: {len(pred_by_sample)}")

    # Check sample overlap
    gt_tokens = set(gt_by_sample.keys())
    pred_tokens = set(pred_by_sample.keys())
    common = gt_tokens & pred_tokens
    print(f"\n  Common samples: {len(common)}")

    # Evaluate each class
    print(f"\n{'='*60}")
    print("Evaluating (this may take a minute)...")
    print(f"{'='*60}")

    all_aps = []
    class_results = {}
    class_errors = {}

    for cls_name in CLASSES:
        # Compute AP at each distance threshold
        cls_aps = []
        cls_tp_pairs_for_errors = None

        for dist_th in DIST_THRESHOLDS:
            ap, tp_pairs, gt_count = evaluate_class(
                gt_by_sample, pred_by_sample, cls_name, dist_th
            )
            cls_aps.append(ap)

            # Use the 2m threshold TP pairs for error metrics
            if dist_th == ERROR_DIST_TH:
                cls_tp_pairs_for_errors = tp_pairs

        # Per-class AP = mean over 4 thresholds
        mean_cls_ap = np.mean(cls_aps)
        all_aps.extend(cls_aps)
        class_results[cls_name] = {
            'ap': mean_cls_ap,
            'gt_count': gt_count
        }

        # Compute error metrics for this class
        if cls_tp_pairs_for_errors:
            errors = compute_class_errors(cls_tp_pairs_for_errors)
            if errors:
                class_errors[cls_name] = errors

        print(f"  {cls_name:25s}: AP={mean_cls_ap:.4f}  (GT={gt_count})")

    # Compute mAP
    mAP = np.mean(all_aps)

    # Compute mean errors across classes (for NDS)
    if class_errors:
        mATE = np.mean([e['mate'] for e in class_errors.values()])
        mASE = np.mean([e['mase'] for e in class_errors.values()])
        mAOE = np.mean([e['maoe'] for e in class_errors.values()])
        mAVE = np.mean([e['mave'] for e in class_errors.values()])
        mAAE = np.mean([e['maae'] for e in class_errors.values()])
    else:
        mATE = mASE = mAOE = mAVE = mAAE = 1.0

    # Normalize errors and compute NDS
    # NDS = (1/10) * (5*mAP + sum(1 - min(metric/threshold, 1)))
    tp_ate = max(1.0 - min(mATE / ATE_THRESH, 1.0), 0.0)
    tp_ase = max(1.0 - min(mASE / ASE_THRESH, 1.0), 0.0)
    tp_aoe = max(1.0 - min(mAOE / AOE_THRESH, 1.0), 0.0)
    tp_ave = max(1.0 - min(mAVE / AVE_THRESH, 1.0), 0.0)
    tp_aae = max(1.0 - min(mAAE / AAE_THRESH, 1.0), 0.0)

    NDS = (1.0 / 10.0) * (5.0 * mAP + tp_ate + tp_ase + tp_aoe + tp_ave + tp_aae)

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS: BEVFormer vs BEVDet pseudo-labels (SWAPPED)")
    print(f"{'='*60}")
    print(f"\n  mAP:  {mAP:.4f}")
    print(f"  NDS:  {NDS:.4f}")

    print(f"\n  Per-class AP (mean over 4 distance thresholds):")
    for cls_name in CLASSES:
        r = class_results.get(cls_name, {'ap': 0.0, 'gt_count': 0})
        print(f"    {cls_name:25s}: AP={r['ap']:.4f}  (GT={r['gt_count']})")

    print(f"\n  Error metrics (TP averages at {ERROR_DIST_TH}m threshold):")
    if class_errors:
        print(f"    mATE: {mATE:.4f}  (threshold: {ATE_THRESH}m)")
        print(f"    mASE: {mASE:.4f}  (threshold: {ASE_THRESH})")
        print(f"    mAOE: {mAOE:.4f}  (threshold: {AOE_THRESH:.4f} rad = {np.degrees(AOE_THRESH):.1f} deg)")
        print(f"    mAVE: {mAVE:.4f}  (threshold: {AVE_THRESH} m/s)")
        print(f"    mAAE: {mAAE:.4f}  (threshold: {AAE_THRESH})")
        print(f"\n    TP scores:")
        print(f"      TP_ATE: {tp_ate:.4f}")
        print(f"      TP_ASE: {tp_ase:.4f}")
        print(f"      TP_AOE: {tp_aoe:.4f}")
        print(f"      TP_AVE: {tp_ave:.4f}")
        print(f"      TP_AAE: {tp_aae:.4f}")

    print(f"\n  Note: These are agreement scores. BEVDet predictions")
    print(f"  (score >= {GT_SCORE_THRESH}) were used as pseudo-GT and BEVFormer")
    print(f"  was scored against them on the custom 1,805-frame dataset.")
    print(f"  This is the SWAPPED evaluation — BEVFormer is now the model")
    print(f"  being scored, BEVDet is the reference.")


if __name__ == '__main__':
    main()