#!/usr/bin/env python3
"""Evaluate saved PandaSet predictions against the included GT infos.

Run from the repository root:
    python scripts/eval_holdout.py

Protocol:
    Detection score >= 0.30
    Planar range <= 50 m
    Class-agnostic center matching within <= 2 m
    One-to-one greedy matching, predictions considered by descending score

This script evaluates the PandaSet center-match metrics. It does not
recompute official nuScenes mAP/NDS or occupancy IoU.
"""

import gzip
import json
import pickle
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"

RESULT_FILES = [
    ("BEVFormer stock", RES / "bevformer_stock_hold12.json.gz"),
    ("BEVFormer FT ep8", RES / "bevformer_ft_ep8_hold12.json"),
    ("BEVDet stock", RES / "bevdet_stock_hold12.json"),
    ("BEVDet FT ep8", RES / "bevdet_ft_ep8_hold12.json"),
]

INFO_FILE = RES / "pandaset_hold12_infos_sameconv.pkl"

SCORE_THRESHOLD = 0.30
MAX_RANGE_M = 50.0
MATCH_DISTANCE_M = 2.0


def read_json(path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as stream:
        data = json.load(stream)

    if isinstance(data, dict) and "results" in data:
        data = data["results"]

    if not isinstance(data, dict):
        raise ValueError(f"Unexpected result format: {path}")

    return data


def read_infos(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing ground-truth infos: {path}")

    with path.open("rb") as stream:
        raw = pickle.load(stream)

    infos = raw["infos"] if isinstance(raw, dict) and "infos" in raw else raw

    if not isinstance(infos, (list, tuple)):
        raise ValueError("Unexpected infos pickle format.")

    by_token = {str(info["token"]): info for info in infos}

    if not by_token:
        raise ValueError("The infos pickle contains no samples.")

    return by_token


def get_predictions(results):
    """Return score-sorted (score, xy) predictions for each frame."""
    output = {}

    for token, boxes in results.items():
        frame_predictions = []

        for box in boxes:
            score = float(box.get("detection_score", 0.0))

            if score < SCORE_THRESHOLD:
                continue

            if "translation" not in box:
                raise ValueError(f"Prediction has no translation: {token}")

            xy = np.asarray(box["translation"][:2], dtype=float)

            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                continue

            if np.linalg.norm(xy) <= MAX_RANGE_M:
                frame_predictions.append((score, xy))

        if frame_predictions:
            # Python sorting is stable, including when scores are tied.
            frame_predictions.sort(key=lambda item: -item[0])
            output[str(token)] = frame_predictions

    return output


def get_ground_truth(infos):
    """Read GT centers and retain objects within the planar range."""
    output = {}
    total_boxes_seen = 0

    for token, info in infos.items():
        boxes = info.get("gt_boxes")

        if boxes is None:
            continue

        boxes = np.asarray(boxes, dtype=float)

        if boxes.size == 0:
            continue

        if boxes.ndim != 2 or boxes.shape[1] < 2:
            raise ValueError(f"Unexpected GT box shape for {token}: {boxes.shape}")

        total_boxes_seen += len(boxes)
        xy = boxes[:, :2]
        valid = np.all(np.isfinite(xy), axis=1)
        valid &= np.linalg.norm(xy, axis=1) <= MAX_RANGE_M
        xy = xy[valid]

        if len(xy):
            output[token] = xy

    if total_boxes_seen == 0:
        raise ValueError(
            "No gt_boxes were found in the infos pickle. "
            "Do not trust the metrics until the expected GT field is identified."
        )

    return output


def match_frame(predictions, ground_truth):
    """Greedily match predictions to distinct GT centers."""
    if len(ground_truth) == 0:
        return 0

    matched_gt = np.zeros(len(ground_truth), dtype=bool)
    true_positives = 0

    for _score, xy in predictions:
        distances = np.linalg.norm(ground_truth - xy, axis=1)
        distances[matched_gt] = np.inf

        nearest = int(np.argmin(distances))

        # Inclusive 2 m boundary. Each GT can be matched only once.
        if distances[nearest] <= MATCH_DISTANCE_M:
            matched_gt[nearest] = True
            true_positives += 1

    return true_positives


def evaluate(predictions, ground_truth):
    total_predictions = sum(len(v) for v in predictions.values())
    total_ground_truth = sum(len(v) for v in ground_truth.values())
    true_positives = 0

    # Include every frame represented in either set.
    for token in sorted(set(predictions) | set(ground_truth)):
        pred = predictions.get(token, [])
        gt = ground_truth.get(token)

        if gt is None:
            continue

        true_positives += match_frame(pred, gt)

    recall = (
        true_positives / total_ground_truth
        if total_ground_truth else 0.0
    )
    precision = (
        true_positives / total_predictions
        if total_predictions else 0.0
    )

    return total_predictions, total_ground_truth, true_positives, recall, precision


def main():
    missing = [str(path) for _, path in RESULT_FILES if not path.is_file()]

    if missing:
        raise FileNotFoundError(
            "Required result files are missing:\n  " + "\n  ".join(missing)
        )

    infos = read_infos(INFO_FILE)
    ground_truth = get_ground_truth(infos)

    print(f"Hold-out info frames: {len(infos)}")
    print(f"Ground-truth centers within {MAX_RANGE_M:.0f} m: "
          f"{sum(len(v) for v in ground_truth.values())}")
    print(f"Score threshold: {SCORE_THRESHOLD:.2f}")
    print(f"Planar range limit: {MAX_RANGE_M:.1f} m")
    print(f"Center-match distance: <= {MATCH_DISTANCE_M:.1f} m")
    print("\nResults")
    print("-" * 91)

    for name, path in RESULT_FILES:
        predictions = get_predictions(read_json(path))
        npred, ngt, tp, recall, precision = evaluate(
            predictions, ground_truth
        )

        print(
            f"{name:20s} predictions={npred:6d} "
            f"GT={ngt:6d} TP={tp:6d} "
            f"recall={100 * recall:5.1f}% "
            f"precision={100 * precision:5.1f}%"
        )


if __name__ == "__main__":
    main()
