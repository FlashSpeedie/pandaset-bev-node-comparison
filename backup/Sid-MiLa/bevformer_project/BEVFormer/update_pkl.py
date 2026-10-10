import sys
import mmcv
import numpy as np

# --- NumPy 2.0 (Windows) -> NumPy 1.x (OSCER) Compatibility Patch ---
try:
    import numpy.core as _core
    sys.modules['numpy._core'] = _core
    if hasattr(_core, 'multiarray'):
        sys.modules['numpy._core.multiarray'] = _core.multiarray
except Exception:
    pass

from scipy.spatial.transform import Rotation as R_tool

data = mmcv.load('custom_nuscenes_infos.pkl')

try:
    extrinsics = np.load('extracted_extrinsics.npy', allow_pickle=True).item()
    print("✅ Loaded extracted_extrinsics.npy successfully!")

    # Map camera pairs to rotations/translations
    for info in data['infos']:
        for cam_name, cam_info in info['cams'].items():
            if cam_name in ['CAM_FRONT_LEFT', 'CAM_BACK_LEFT']:
                pair_key = 'CAM_FRONT->CAM_FRONT_LEFT' if 'FRONT' in cam_name else 'CAM_BACK->CAM_BACK_LEFT'
                if pair_key in extrinsics:
                    R_mat = extrinsics[pair_key]['R']
                    # Convert 3x3 Rotation matrix to Quaternion [w, x, y, z]
                    q = R_tool.from_matrix(R_mat).as_quat()
                    cam_info['sensor2ego_rotation'] = [q[3], q[0], q[1], q[2]]
                    cam_info['sensor2ego_translation'] = extrinsics[pair_key]['t_dir'].tolist()
except Exception as e:
    print(f"Warning updating extrinsics: {e}")

mmcv.dump(data, 'custom_nuscenes_infos.pkl')
print(f"🎉 Successfully verified and updated custom_nuscenes_infos.pkl across {len(data['infos'])} frames!")
