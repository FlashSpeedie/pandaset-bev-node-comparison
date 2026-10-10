import sys
import os

# 1. FORCE THE PATH
sys.path.insert(0, '/scratch/kaustubh/bevformer_project/mmdetection3d')
sys.path.insert(0, '/scratch/kaustubh/bevformer_project/BEVFormer')

import mmcv
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmcv.utils import import_modules_from_strings
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset

print("Loading config...")
cfg = Config.fromfile('projects/configs/bevformer/bevformer_base.py')

print("Loading plugins...")
if cfg.get('custom_imports', None):
    import_modules_from_strings(**cfg['custom_imports'])

# 2. Configure dataset for custom infos
cfg.data.test.ann_file = 'custom_nuscenes_infos.pkl'

print("Building dataset (Teacher)...")
dataset = build_dataset(cfg.data.test)
dataset.version = 'v1.0-mini'

print("Loading your predictions (Student)...")
# --- THE PERMANENT FILE PATH FIX ---
outputs = mmcv.load('/scratch/kaustubh/bevformer_project/BEVFormer/custom_car_predictions.pkl')
# -----------------------------------

print("Grading...")
dataset.evaluate(outputs, metric='bbox')