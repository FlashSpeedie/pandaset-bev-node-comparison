import os, sys
from types import ModuleType
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader

try:
    import numba.core.errors as numba_errors
    sys.modules["numba.errors"] = numba_errors
except Exception:
    dummy_errors = ModuleType("numba.errors")
    dummy_errors.NumbaPerformanceWarning = Warning
    dummy_errors.NumbaDeprecationWarning = Warning
    dummy_errors.NumbaWarning = Warning
    sys.modules["numba.errors"] = dummy_errors

class DummyClass:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
    def __iter__(self): return iter([])

class DummyList:
    def __getitem__(self, item): return (0.1, 0.5, 0.9)
    def __iter__(self): return iter([(0.1, 0.5, 0.9)] * 20)

class DummyModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = "/tmp/dummy.py"
        self.__path__ = []
    def __getattr__(self, item):
        if item.startswith("__"): raise AttributeError(item)
        if item == "color_palette": return lambda *args, **kwargs: DummyList()
        return DummyClass()

class GeneralMockFinder(MetaPathFinder, Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("detectron2") or fullname == "IPython" or fullname == "seaborn":
            return spec_from_loader(fullname, self)
        if fullname.startswith("mmdet3d.ops."):
            leaf = fullname.split(".")[-1]
            if leaf.endswith("_ext") or leaf.endswith("_cuda") or leaf in ("voxel_layer", "sparse_conv_ext"):
                return spec_from_loader(fullname, self)
        return None
    def create_module(self, spec):
        return DummyModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, GeneralMockFinder())

PROJECT_ROOT = os.path.expanduser("~/Sid - MiLa/bevformer_project")
sys.path.insert(0, os.path.join(PROJECT_ROOT, "BEVFormer/mmdetection3d"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "BEVFormer"))

import mmcv.utils.registry
_orig_register = mmcv.utils.registry.Registry._register_module
def _patched_register(self, module, module_name=None, force=True):
    return _orig_register(self, module, module_name=module_name, force=True)
mmcv.utils.registry.Registry._register_module = _patched_register

import mmcv
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmdet3d.apis import single_gpu_test
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmcv.runner import load_checkpoint
from projects.mmdet3d_plugin.datasets.nuscenes_dataset import CustomNuScenesDataset, NuScenesEval_custom

nusc_eval_mod = sys.modules[NuScenesEval_custom.__module__]

def patched_init(self, nusc, config, result_path, eval_set, output_dir, verbose=True, *args, **kwargs):
    self.nusc = nusc; self.result_path = result_path; self.eval_set = eval_set
    self.output_dir = output_dir; self.verbose = verbose; self.cfg = config
    
    import inspect
    actual_box_cls = None
    for name, obj in inspect.getmembers(nusc_eval_mod):
        if inspect.isclass(obj) and hasattr(obj, "sample_token") and hasattr(obj, "translation"):
            actual_box_cls = obj
            break
    if not actual_box_cls:
        from nuscenes.eval.detection.data_classes import DetectionBox
        actual_box_cls = DetectionBox
    
    self.gt_boxes = nusc_eval_mod.load_gt(self.nusc, self.eval_set, actual_box_cls, verbose=self.verbose)
    self.pred_boxes, self.meta = nusc_eval_mod.load_prediction(self.result_path, self.cfg.max_boxes_per_sample, actual_box_cls, verbose=self.verbose)
    val_tokens = set(self.gt_boxes.sample_tokens)
    self.pred_boxes.boxes = {k: v for k, v in self.pred_boxes.boxes.items() if k in val_tokens}
    assert set(self.pred_boxes.sample_tokens) == set(self.gt_boxes.sample_tokens), "Samples mismatch"
    self.sample_tokens = self.gt_boxes.sample_tokens
NuScenesEval_custom.__init__ = patched_init

print("Loading configuration...")
cfg = Config.fromfile("projects/configs/bevformer/bevformer_base.py")
cfg.data.test.ann_file = "custom_nuscenes_infos.pkl"
cfg.data.test.test_mode = True

print("Building dataset...")
dataset = build_dataset(cfg.data.test)
dataset.version = "v1.0-mini"

print("Building dataloader...")
data_loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

print("Building BEVFormer model...")
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "ckpts/bevformer_r101_dcn_24ep.pth", map_location="cpu")
model = MMDataParallel(model, device_ids=[0])

print("Starting inference on RTX 3090 GPU...")
outputs = single_gpu_test(model, data_loader)

print("=" * 50)
print("INFERENCE COMPLETE. GRADING PREDICTIONS IN MEMORY...")
print("=" * 50)
dataset.evaluate(outputs, metric="bbox")