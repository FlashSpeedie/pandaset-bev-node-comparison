import sys
import os
import mmcv
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

original_register = mmcv.utils.registry.Registry._register_module
def force_register(self, module, module_name=None, force=False):
    return original_register(self, module=module, module_name=module_name, force=True)
mmcv.utils.registry.Registry._register_module = force_register

class DummyClass:
    def __init__(self, *args, **kwargs): pass
    def __call__(self, *args, **kwargs): return self
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
            return lambda *args, **kwargs: [(0.1, 0.5, 0.9)] * 20
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
    def create_module(self, spec): return DummyModule(spec.name)
    def exec_module(self, module): pass

sys.meta_path.insert(0, GeneralMockFinder())


# ---- FPS timing (INT8) with plugin registration ----
import os
import time
import importlib
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

cfg = Config.fromfile("projects/configs/bevformer/bevformer_base.py")

# Register CustomNuScenesDataset and BEVFormer modules
if hasattr(cfg, "plugin") and cfg.plugin:
    import sys
    plugin_dir = cfg.plugin_dir if hasattr(cfg, "plugin_dir") else "projects/mmdet3d_plugin/"
    # e.g. projects/mmdet3d_plugin/ -> import projects.mmdet3d_plugin
    plg = plugin_dir.strip("/").replace("/", ".")
    importlib.import_module(plg)
else:
    importlib.import_module("projects.mmdet3d_plugin")

cfg.data.test.test_mode = True
cfg.model.train_cfg = None

dataset = build_dataset(cfg.data.test)
data_loader = build_dataloader(
    dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False
)
model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "ckpts/bevformer_r101_dcn_24ep_INT8_optimized.pth", map_location="cuda:0")
model = MMDataParallel(model.cuda(), device_ids=[0]).eval()

# warmup
it = iter(data_loader)
for _ in range(3):
    data = next(it)
    with torch.no_grad():
        model(return_loss=False, rescale=True, **data)

n = 20
it = iter(data_loader)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(n):
    data = next(it)
    with torch.no_grad():
        model(return_loss=False, rescale=True, **data)
torch.cuda.synchronize()
t1 = time.time()
print("=== Running INT8 FPS ===")
print("INT8 FPS: {:.2f}".format(n / (t1 - t0)))
print("INT8 Latency ms: {:.1f}".format((t1 - t0) / n * 1000))
