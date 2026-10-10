"""
Standalone mAP / NDS evaluation for the custom 1805-frame dataset,
using BEVFormer pseudo-labels as the reference ("GT") boxes.
CAVEAT: only valid as a cross-model agreement score, not verified accuracy.
"""
import argparse
import json

from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.detection.algo import accumulate, calc_ap, calc_tp
from nuscenes.eval.detection.constants import TP_METRICS
from nuscenes.eval.common.data_classes import EvalBoxes


def load_boxes(path, score_default=1.0):
    with open(path) as f:
        data = json.load(f)
    results = data['results'] if isinstance(data, dict) and 'results' in data else data
    eval_boxes = EvalBoxes()
    for sample_token, dets in results.items():
        boxes = []
        for det in dets:
            boxes.append(DetectionBox(
                sample_token=sample_token,
                translation=tuple(det['translation']),
                size=tuple(det['size']),
                rotation=tuple(det['rotation']),
                velocity=tuple(det.get('velocity', (0.0, 0.0))),
                ego_translation=tuple(det.get('ego_translation', (0.0, 0.0, 0.0))),
                num_pts=int(det.get('num_pts', -1)),
                detection_name=det.get('detection_name', det.get('name', 'car')),
                detection_score=float(det.get('detection_score', score_default)),
                attribute_name=det.get('attribute_name', ''),
            ))
        eval_boxes.add_boxes(sample_token, boxes)
    return eval_boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True)
    ap.add_argument('--predictions', required=True)
    ap.add_argument('--config', default='detection_cvpr_2019')
    args = ap.parse_args()

    cfg = config_factory(args.config)

    gt_boxes = load_boxes(args.reference, score_default=1.0)
    pred_boxes = load_boxes(args.predictions)

    gt_tokens = set(gt_boxes.sample_tokens)
    pred_tokens = set(pred_boxes.sample_tokens)
    common = gt_tokens & pred_tokens
    print(f"Reference samples: {len(gt_tokens)} | Prediction samples: {len(pred_tokens)} | Matched: {len(common)}")
    if not common:
        raise SystemExit("No overlapping sample_tokens between reference and predictions.")

    gt_boxes = EvalBoxes.deserialize(
        {tok: [b.serialize() for b in gt_boxes.boxes[tok]] for tok in common}, DetectionBox)
    pred_boxes = EvalBoxes.deserialize(
        {tok: [b.serialize() for b in pred_boxes.boxes[tok]] for tok in common}, DetectionBox)

    metric_data_list = {}
    class_names = cfg.class_names
    for class_name in class_names:
        for dist_th in cfg.dist_ths:
            md = accumulate(gt_boxes, pred_boxes, class_name, cfg.dist_fcn_callable, dist_th)
            metric_data_list[(class_name, dist_th)] = md

    ap_per_class = {}
    for class_name in class_names:
        aps = [calc_ap(metric_data_list[(class_name, dist_th)], cfg.min_recall, cfg.min_precision)
               for dist_th in cfg.dist_ths]
        ap_per_class[class_name] = sum(aps) / len(aps)

    tp_per_class = {}
    for class_name in class_names:
        md = metric_data_list[(class_name, cfg.dist_th_tp)]
        tp_per_class[class_name] = {m: calc_tp(md, cfg.min_recall, m) for m in TP_METRICS}

    mAP = sum(ap_per_class.values()) / len(ap_per_class)

    tp_errors_mean = {}
    for m in TP_METRICS:
        vals = [tp_per_class[c][m] for c in class_names if tp_per_class[c][m] == tp_per_class[c][m]]
        tp_errors_mean[m] = sum(vals) / len(vals) if vals else 1.0

    NDS = (5 * mAP + sum(max(0.0, 1.0 - tp_errors_mean[m]) for m in TP_METRICS)) / 10.0

    print("\n=== Per-class AP ===")
    for c, v in ap_per_class.items():
        print(f"  {c:25s} {v:.4f}")
    print("\n=== Mean TP errors ===")
    for m, v in tp_errors_mean.items():
        print(f"  {m:8s} {v:.4f}")
    print(f"\nmAP: {mAP:.4f}")
    print(f"NDS: {NDS:.4f}")
    print("\nReminder: agreement-with-reference score, not verified ground truth accuracy.")


if __name__ == '__main__':
    main()
