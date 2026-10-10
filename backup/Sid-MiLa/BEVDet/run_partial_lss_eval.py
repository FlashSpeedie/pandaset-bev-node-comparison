import sys, os
sys.path.insert(0, os.getcwd())

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset, build_dataloader
from mmdet3d.models import build_model
from mmdet3d.apis import single_gpu_test

from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.loaders import (
    load_prediction, load_gt, add_center_dist, filter_eval_boxes
)

def _patched_init(self, nusc, config, result_path, eval_set, output_dir=None, verbose=True):
    self.nusc = nusc
    self.eval_set = eval_set
    self.result_path = result_path
    self.output_dir = output_dir
    self.verbose = verbose
    self.cfg = config

    self.pred_boxes, self.meta = load_prediction(
        self.result_path, self.cfg.max_boxes_per_sample, DetectionBox, verbose=verbose)
    self.gt_boxes = load_gt(self.nusc, self.eval_set, DetectionBox, verbose=verbose)

    common = set(self.pred_boxes.sample_tokens) & set(self.gt_boxes.sample_tokens)
    print('Evaluating on', len(common), 'common samples (partial val)')

    pred_f, gt_f = EvalBoxes(), EvalBoxes()
    for t in common:
        pred_f.add_boxes(t, self.pred_boxes[t])
        gt_f.add_boxes(t, self.gt_boxes[t])
    self.pred_boxes = pred_f
    self.gt_boxes = gt_f

    self.pred_boxes = add_center_dist(nusc, self.pred_boxes)
    self.gt_boxes = add_center_dist(nusc, self.gt_boxes)
    self.pred_boxes = filter_eval_boxes(nusc, self.pred_boxes, self.cfg.class_range, verbose=verbose)
    self.gt_boxes = filter_eval_boxes(nusc, self.gt_boxes, self.cfg.class_range, verbose=verbose)

NuScenesEval.__init__ = _patched_init

cfg = Config.fromfile('configs/bevdet/bevdet-r50_clean.py')
cfg.data.test.ann_file = 'data/nuscenes/partial_infos_val_available.pkl'
cfg.data.test.test_mode = True
cfg.model.train_cfg = None

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset, samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
load_checkpoint(model, 'ckpts/bevdet-r50.pth', map_location='cpu')
model = MMDataParallel(model, device_ids=[0])
model.eval()

print('Running inference on', len(dataset), 'samples...')
outputs = single_gpu_test(model, data_loader)

print('Evaluating...')
metrics = dataset.evaluate(outputs, metric='bbox')
print('==== PARTIAL VAL LSS RESULTS ====')
for k, v in metrics.items():
    print(f'{k}: {v}')
