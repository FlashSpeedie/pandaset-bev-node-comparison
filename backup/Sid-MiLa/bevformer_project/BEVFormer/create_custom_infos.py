import os
import numpy as np
import mmcv

base_path = '/scratch/kaustubh/bevformer_project/processed_frames'
cams = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

# Load camera intrinsics
calibs = {}
for i, cam_name in enumerate(cams, start=1):
    data = np.load(f"No{i}_camera.npz")
    calibs[cam_name] = {
        'cam_intrinsic': data['camera_matrix'].tolist(),
    }

frame_files = sorted([f for f in os.listdir(os.path.join(base_path, 'CAM_FRONT')) if f.endswith(('.jpg', '.png'))])

infos = []
scene_token = "scene_custom_001"

for idx, frame_name in enumerate(frame_files):
    # Standard dummy CAN bus vector expected by BEVFormer spatial cross-attention
    can_bus = np.zeros(18)
    can_bus[0:3] = [0.0, 0.0, 0.0]  # translation
    can_bus[3:7] = [1.0, 0.0, 0.0, 0.0]  # rotation quaternion
    
    info = {
        'token': f"frame_{idx:04d}",
        'frame_idx': idx,
        'timestamp': idx * 100000,
        'scene_token': scene_token,
        'can_bus': can_bus,
        'cams': {}
    }
    
    for cam_name in cams:
        rel_path = os.path.join('../processed_frames', cam_name, frame_name)
        info['cams'][cam_name] = {
            'data_path': rel_path,
            'type': cam_name,
            'sample_data_token': f"{cam_name}_{idx:04d}",
            'sensor2ego_translation': [0.0, 0.0, 1.5],
            'sensor2ego_rotation': [1.0, 0.0, 0.0, 0.0],
            'ego2global_translation': [0.0, 0.0, 0.0],
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],
            'cam_intrinsic': calibs[cam_name]['cam_intrinsic']
        }
    infos.append(info)

data_dict = {'infos': infos, 'metadata': {'version': 'v1.0-trainval'}}
mmcv.dump(data_dict, 'custom_nuscenes_infos.pkl')
print(f"✅ Re-created custom_nuscenes_infos.pkl with {len(infos)} valid frames!")
