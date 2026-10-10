
import sys
from types import ModuleType

try:
    import numba.core.errors as numba_errors
    sys.modules["numba.errors"] = numba_errors
except Exception:
    dummy = ModuleType("numba.errors")
    dummy.NumbaPerformanceWarning = Warning
    dummy.NumbaDeprecationWarning = Warning
    dummy.NumbaWarning = Warning
    sys.modules["numba.errors"] = dummy

import torch

# Patch empty temporal queue before train imports finish using the model class
import runpy

# After plugin loads, patch BEVFormer.obtain_history_bev
def _install_patch():
    from projects.mmdet3d_plugin.bevformer.detectors.bevformer import BEVFormer
    orig = BEVFormer.obtain_history_bev
    def safe_obtain_history_bev(self, imgs_queue, img_metas_list):
        if imgs_queue is None or (isinstance(imgs_queue, torch.Tensor) and (imgs_queue.numel() == 0 or imgs_queue.shape[0] == 0 or (imgs_queue.dim() >= 2 and imgs_queue.shape[1] == 0))):
            return None
        return orig(self, imgs_queue, img_metas_list)
    BEVFormer.obtain_history_bev = safe_obtain_history_bev
    print("Patched BEVFormer.obtain_history_bev for empty queue")

# Import plugin path by loading config machinery lightly
sys.path.insert(0, ".")
sys.path.insert(0, "mmdetection3d")
try:
    _install_patch()
except Exception as e:
    print("Patch deferred:", e)

sys.argv = [
    "tools/train.py",
    "projects/configs/bevformer/bevformer_pseudo_ft.py",
    "--work-dir",
    "work_dirs/bevformer_pseudo_ft",
]

# Install patch again right before main via import of detector after mmcv registry
import importlib
try:
    import projects.mmdet3d_plugin.bevformer.detectors.bevformer as bevmod
    orig = bevmod.BEVFormer.obtain_history_bev
    def safe_obtain_history_bev(self, imgs_queue, img_metas_list):
        if imgs_queue is None:
            return None
        if isinstance(imgs_queue, torch.Tensor):
            if imgs_queue.numel() == 0:
                return None
            if imgs_queue.dim() >= 1 and imgs_queue.shape[0] == 0:
                return None
            if imgs_queue.dim() >= 2 and imgs_queue.shape[1] == 0:
                return None
        return orig(self, imgs_queue, img_metas_list)
    bevmod.BEVFormer.obtain_history_bev = safe_obtain_history_bev
    print("Patched obtain_history_bev OK")
except Exception as e:
    print("Could not pre-patch:", e)

runpy.run_path("tools/train.py", run_name="__main__")
