import sys, os, runpy
from types import ModuleType

# Mock numba.core.errors to prevent any Numba version mismatch crashes
try:
    import numba.core.errors as numba_errors
    sys.modules["numba.core.errors"] = numba_errors
except Exception:
    dummy = ModuleType("numba.core.errors")
    dummy.NumbaPerformanceWarning = Warning
    dummy.NumbaDeprecationWarning = Warning
    dummy.NumbaWarning = Warning
    sys.modules["numba.core.errors"] = dummy

print("=== Starting Timed BEVDet Inference on Custom 1805 ===")
sys.argv = [
    "tools/test.py",
    "configs/bevdet/bevdet-r50.py",
    "ckpts/bevdet-r50.pth",
    "--eval", "bbox",
    "--cfg-options",
    "data.test.ann_file=/home/adas/Sid - MiLa/bevformer_project/BEVFormer/custom_nuscenes_infos_clean.pkl",
    "data.workers_per_gpu=0",
]
runpy.run_path("tools/test.py", run_name="__main__")
