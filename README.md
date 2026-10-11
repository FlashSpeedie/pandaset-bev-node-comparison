# Spatial Cross-Attention vs. Lift-Splat-Shoot for Multi-Camera 3D Object Detection

This repository contains evaluation results and analysis comparing two camera-to-BEV approaches for multi-camera 3D object detection:

- **BEVFormer-base**, which uses spatial cross-attention.
- **BEVDet-R50-CBGS**, which uses the Lift-Splat-Shoot (LSS) approach.

The study includes an evaluation on nuScenes mini and a separate experiment on a 12-sequence PandaSet hold-out split. The PandaSet experiment uses six-camera data, official cuboids, and a custom center-matching evaluation protocol.

## Results

### Official nuScenes Mini Evaluation

Results on 81 nuScenes mini validation frames using the official evaluation metrics:

| Method | mAP | NDS |
|---|---:|---:|
| BEVDet-R50-CBGS | 0.2916 | 0.3510 |
| BEVFormer-base | 0.3691 | 0.4216 |

### PandaSet Hold-Out Evaluation

The hold-out experiment covers 12 sequences and 960 frames.

Evaluation settings:
- Minimum detection confidence: 0.30
- Maximum planar distance: 50 m
- Maximum center distance for a match: 2 m
- Reported ground-truth boxes: 24,410

| Method | Predictions | True Positives | Recall | Precision |
|---|---:|---:|---:|---:|
| BEVFormer stock | 21,734 | 8,238 | 33.7% | 37.9% |
| BEVFormer fine-tuned, epoch 8 | 23,943 | 12,034 | 49.3% | 50.3% |
| BEVDet-R50-CBGS stock | 42,659 | 2,313 | 9.5% | 5.4% |
| BEVDet fine-tuned, epoch 8 | 18,628 | 12,852 | 52.7% | 69.0% |

These PandaSet values use the study's custom center-matching protocol. They are not official nuScenes mAP or NDS scores.

### Range and Occupancy Analysis

Recall and precision are reported in that order. Occupancy IoU is the reported overall value for each model.

| Method | Range | Recall | Precision | Overall Occupancy IoU |
|---|---|---:|---:|---:|
| BEVFormer fine-tuned | 0-20 m | 69.9% | 61.5% | 12.8% |
| BEVFormer fine-tuned | 20-50 m | 40.0% | 43.4% | 12.8% |
| BEVDet fine-tuned | 0-20 m | 88.4% | 87.1% | 23.9% |
| BEVDet fine-tuned | 20-50 m | 37.9% | 56.9% | 23.9% |

## Quick Start

The evaluation script is intended to calculate PandaSet center-matching metrics from the saved predictions and hold-out information in `results/`. It does not train models or rerun inference.

### 1. Clone the repository

```bash
git clone https://github.com/FlashSpeedie/pandaset-bev-node-comparison.git
cd pandaset-bev-node-comparison
```

If the repository is private, authenticate with GitHub before cloning.

### 2. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 3. Run the evaluator

```bash
python scripts/eval_holdout.py
```

Compare the actual output with the tables above. The script reproduces all four rows of the PandaSet center-match table from the included result files.

## Datasets

See [`datasets.txt`](datasets.txt) for download links and expected directory layouts.

### nuScenes Mini
- Download: https://www.nuscenes.org/nuscenes
- Expected location: `data/nuscenes/v1.0-mini/`
- Used for the official mAP and NDS evaluation.

### PandaSet
- Download: https://scale.com/open-datasets/pandaset
- Expected location: `data/pandaset/`
- Used for the hold-out center-matching and range and occupancy analyses.

The saved evaluation files can be analyzed without downloading raw datasets. Raw data and the appropriate model environment are needed to rerun inference or training.

## Repository Contents

- `results/` - Saved prediction results and hold-out information.
- `figures/` - BEV visualizations used in the paper.
- `scripts/eval_holdout.py` - PandaSet center-matching evaluation script.
- `requirements.txt` - Python dependencies.
- `datasets.txt` - Dataset download links and expected layouts.

## Not Included

- Model checkpoints (`.pth` files).
- Raw nuScenes and PandaSet data.
- Training logs and intermediate checkpoints.
- Every configuration and dependency needed to recreate the original model environments.

Reproducing inference or training requires the appropriate model repositories, configurations, checkpoints, and datasets.

## Limitations and Reproducibility Notes

- Fine-tuning used 73 PandaSet training sequences for domain adaptation. No new model layer was added.
- Intermediate checkpoints were inspected using the same 12-sequence hold-out split. This means the split should be treated as an exploratory evaluation set, not an untouched final test set.
- PandaSet recall and precision use custom center matching, not the official nuScenes evaluation protocol.
- The models differ in their backbones and detection heads as well as their camera-to-BEV approaches. The results compare complete model pipelines and do not isolate the causal effect of the view transformation method.
- The custom vehicle recordings do not have verified 3D ground-truth annotations for the same evaluation protocol and are not included in the final quantitative tables.
- KITTI and WoodScape are not used for the reported quantitative results.
