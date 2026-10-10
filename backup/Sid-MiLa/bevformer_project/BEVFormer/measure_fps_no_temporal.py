import sys, os, time, torch
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

import mmcv
original_register = mmcv.utils.registry.Registry._register_module
def force_register(self, module, module_name=None, force=False):
    return original_register(self, module=module, module_name=module_name, force=True)
mmcv.utils.registry.Registry._register_module = force_register

class DummyClass:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return self
    def __iter__(self): return iter([])
    def __getitem__(self, k): return self

class DummyModule(ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__file__ = "/tmp/dummy.py"
        self.__path__ = []
    def __getattr__(self, item):
        if item.startswith("__"): raise AttributeError(item)
        return DummyClass()

class GeneralMockFinder(MetaPathFinder, Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith("detectron2") or fullname in ("IPython", "seaborn"):
            return spec_from_loader(fullname, self)
        if fullname.startswith("mmdet3d.ops."):
            leaf = fullname.split(".")[-1]
            if leaf.endswith(("_ext", "_cuda")) or leaf in ("voxel_layer", "sparse_conv_ext", "ball_query_ext", "iou3d_cuda"):
                return spec_from_loader(fullname, self)
        return None
    def create_module(self, spec):
        return DummyModule(spec.name)
    def exec_module(self, module):
        pass

sys.meta_path.insert(0, GeneralMockFinder())

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint

import projects.mmdet3d_plugin
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model

cfg = Config.fromfile("projects/configs/bevformer/bevformer_no_temporal.py")
cfg.data.test.test_mode = True
cfg.model.train_cfg = None

dataset = build_dataset(cfg.data.test)
loader = build_dataloader(dataset, samples_per_gpu=1, workers_per_gpu=0, dist=False, shuffle=False)

model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
load_checkpoint(model, "ckpts/bevformer_r101_dcn_24ep.pth", map_location="cuda:0")
model = MMDataParallel(model.cuda(), device_ids=[0])
model.eval()

it = iter(loader)
for _ in range(3):
    data = next(it)
    with torch.no_grad():
        model(return_loss=False, rescale=True, **data)

n = 20
it = iter(loader)
torch.cuda.synchronize()
t0 = time.time()
for i in range(n):
    data = next(it)
    with torch.no_grad():
        model(return_loss=False, rescale=True, **data)
torch.cuda.synchronize()
t1 = time.time()

print("=" * 40)
print("No-temporal (mini) FPS: {:.2f}".format(n / (t1 - t0)))
print("Latency ms/frame: {:.1f}".format((t1 - t0) / n * 1000))
print("=" * 40)
