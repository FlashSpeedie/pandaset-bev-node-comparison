import sys, os, runpy
from types import ModuleType
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader
import mmcv

try:
    import numba.core.errors as numba_errors
    sys.modules["numba.errors"] = numba_errors
except Exception:
    dummy = ModuleType("numba.errors")
    dummy.NumbaPerformanceWarning = Warning
    dummy.NumbaDeprecationWarning = Warning
    dummy.NumbaWarning = Warning
    sys.modules["numba.errors"] = dummy

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
print("=== BEVFormer on CLEAN custom 1805 ===")
sys.argv = [
    "tools/test.py",
    "projects/configs/bevformer/bevformer_no_temporal.py",
    "ckpts/bevformer_r101_dcn_24ep.pth",
    "--eval", "bbox",
    "--cfg-options",
    "data.test.ann_file=custom_nuscenes_infos_calibrated.pkl",
    "data.workers_per_gpu=0",
]
runpy.run_path("tools/test.py", run_name="__main__")
