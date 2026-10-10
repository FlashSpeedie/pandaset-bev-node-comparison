# Spatial Cross-Attention vs Lift-Splat-Shoot on PandaSet

Comparison of two multi-camera bird's-eye-view (BEV) view nodes for 3D object detection.

- Spatial cross-attention (BEVFormer-base)
- Lift-Splat-Shoot (BEVDet-R50-CBGS)

Official protocol on nuScenes mini. Second test on a held-out split of PandaSet (12 sequences, 960 frames) with official cameras and cuboids.

## Locked Results

Score >= 0.30, planar range <= 50 m, 2 m center match against official cuboid centers (24,410 ground-truth boxes).

| Method               | Recall (%) | Precision (%) |
|----------------------|------------|---------------|
| BEVFormer stock      | 33.7       | 37.9          |
| BEVFormer FT epoch 8 | 49.3       | 50.3          |
| BEVDet-CBGS stock    | 9.5        | 5.4           |
| BEVDet FT epoch 8    | 52.7       | 69.0          |

Official nuScenes mini (81 frames):
- BEVDet-R50-CBGS: mAP 0.2916, NDS 0.3510
- BEVFormer-base: mAP 0.3691, NDS 0.4216

Range split (same filters):
- BEVFormer FT: 0-20 m 69.9 / 61.5, 20-50 m 40.0 / 43.4, occupancy IoU 12.8%
- BEVDet FT: 0-20 m 88.4 / 87.1, 20-50 m 37.9 / 56.9, occupancy IoU 23.9%

## Files

- results/bevformer_ft_ep8_hold12.json
- results/bevdet_ft_ep8_hold12.json
- results/bevdet_stock_hold12.json
- results/pandaset_hold12_infos_sameconv.pkl
- figures/ : fused BEV examples used in the paper
- scripts/ : evaluation code that produced the center-match and occupancy numbers

Only the files that produced the locked paper numbers are included. Intermediate training runs and failed experiments are excluded.

## Notes

- Fine-tuning is domain adaptation on the 73 PandaSet training sequences. No new layer was added.
- PandaSet recall and precision are not official mAP or NDS.
- The two models differ in backbone and head in addition to the view node.
