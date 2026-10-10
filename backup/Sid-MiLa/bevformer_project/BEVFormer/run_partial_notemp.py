import sys, os, runpy
from types import ModuleType
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader
import mmcv

# numba shim
try:
    import numba.core.errors as numba_errors
    sys.modules["numba.errors"] = numba_errors
except Exception:
    dummy = ModuleType("numba.errors")
    dummy.NumbaPerformanceWarning = Warning
    dummy.NumbaDeprecationWarning = Warning
    dummy.NumbaWarning = Warning
    sys.modules["numba.errors"] = dummy

# force registry
_orig = mmcv.utils.registry.Registry._register_module
def force_register(self, module, module_name=None, force=False):
    return _orig(self, module=module, module_name=module_name, force=True)
mmcv.utils.registry.Registry._register_module = force_register

class DummyClass:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self
    def __iter__(self): return iter([])
    def __getitem__(self, key): return self

class DummyModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = "/tmp/dummy.py"
        self.__path__ = []
    def __getattr__(self, item):
        if item.startswith("__"): raise AttributeError(item)
        if item == "color_palette":
            return lambda *a, **k: [(0.1, 0.5, 0.9)] * 20
        return DummyClass()

class GeneralMockFinder(MetaPathFinder, Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("detectron2") or fullname in ("IPython", "seaborn"):
            return spec_from_loader(fullname, self)
        if fullname.startswith("mmdet3d.ops."):
            leaf = fullname.split(".")[-1]
            if leaf.endswith(("_ext", "_cuda")) or leaf in ("voxel_layer", "sparse_conv_ext", "ball_query_ext"):
                return spec_from_loader(fullname, self)
        return None
    def create_module(self, spec):
        return DummyModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, GeneralMockFinder())

# subset NuScenesEval patch
from nuscenes.eval.detection.evaluate import NuScenesEval
from nuscenes.eval.common.data_classes import EvalBoxes
from nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes.eval.common.loaders import load_prediction, load_gt, add_center_dist, filter_eval_boxes

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
    print("Evaluating on", len(common), "common samples (partial val)")
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

sys.argv = [
    "tools/test.py",
    "projects/configs/bevformer/bevformer_no_temporal.py",
    "ckpts/bevformer_r101_dcn_24ep.pth",
    "--eval", "bbox",
    "--cfg-options",
    "data.test.ann_file=data/nuscenes/bevformer_partial_val_1076.pkl",
]
print("=== No-Temporal ablation on partial 1076 ===")
runpy.run_path("tools/test.py", run_name="__main__")
