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

import runpy
sys.argv = [
    "tools/test.py",
    "projects/configs/bevformer/bevformer_no_temporal.py",
    "ckpts/bevformer_r101_dcn_24ep.pth",
    "--eval", "bbox"
]
print("=== Running NO-TEMPORAL Ablation (queue_length=1) ===")
runpy.run_path("tools/test.py", run_name="__main__")
