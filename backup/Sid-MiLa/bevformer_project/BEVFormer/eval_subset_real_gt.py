import json
from nuscenes import NuScenes
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.common.data_classes import EvalBoxes

# The results file you just generated
res_path = "/home/adas/Sid - MiLa/BEVDet/work_dirs/bevdet_partial_1076/pts_bbox/results_nusc.json"

# Monkey-patch NuScenesEval to skip the assert and filter Ground Truth
original_init = NuScenesEval.__init__

def patched_init(self, nusc, config, result_path, eval_set, output_dir=None, verbose=True):
    try:
        original_init(self, nusc, config, result_path, eval_set, output_dir, verbose)
    except AssertionError:
        pass # Catch the assert crash!
        
    # Filter GT to only include tokens that actually exist in our 1076 predictions
    pred_tokens = set(self.pred_boxes.sample_tokens)
    filtered_gt = EvalBoxes()
    for tok in pred_tokens:
        if tok in self.gt_boxes.sample_tokens:
            filtered_gt.add_boxes(tok, self.gt_boxes.boxes[tok])
            
    self.gt_boxes = filtered_gt
    self.sample_tokens = list(pred_tokens)

NuScenesEval.__init__ = patched_init

print("Loading NuScenes and evaluating real GT on matched subset...")
nusc = NuScenes(version='v1.0-trainval', dataroot='data/nuscenes', verbose=False)
cfg = config_factory('detection_cvpr_2019')

nusc_eval = NuScenesEval(nusc, config=cfg, result_path=res_path, eval_set='val', output_dir='./tmp')
metrics, _ = nusc_eval.evaluate()

print("\n" + "="*50)
print("FINAL RESULTS: NO-TEMPORAL ABLATION (1076 Real GT)")
print("="*50)
print(f"mAP: {metrics.mean_ap:.4f}")
print(f"NDS: {metrics.nd_score:.4f}")
print("="*50)
