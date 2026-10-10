import numpy as np
import pickle
import os

# Define the 6 cameras required by BEVFormer / nuScenes standard
CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT'
]

def build_camera_calibration_dict(npz_folder_path):
    """
    Reads Dr. Xu's .npz camera intrinsics and structures them 
    into the BEVFormer dataset metadata format.
    """
    calibration_data = {}
    
    for cam_name in CAMERAS:
        # Standard filename pattern (e.g., No1_camera.npz, No2_camera.npz, etc.)
        # Modify the filename mapping below if Dr. Xu named them differently:
        cam_id = CAMERAS.index(cam_name) + 1
        npz_file = os.path.join(npz_folder_path, f'No{cam_id}_camera.npz')
        
        if os.path.exists(npz_file):
            data = np.load(npz_file)
            intrinsics = data['camera_matrix']
            dist_coeffs = data['dist_coeffs']
            print(f" Loaded intrinsics for {cam_name} from {os.path.basename(npz_file)}")
        else:
            print(f" Warning: {npz_file} not found! Using default camera matrix.")
            intrinsics = np.array([
                [1800.0, 0.0, 960.0],
                [0.0, 1729.0, 540.0],
                [0.0, 0.0, 1.0]
            ])
            dist_coeffs = np.zeros((1, 5))
            
        # Extrinsic rigid transformation placeholder (Camera-to-Ego vehicle frame)
        # Identity matrix for rotation, zero vector for translation until physical rig offsets are set
        sensor2ego_translation = [0.0, 0.0, 1.5]  # Nominal 1.5m height above ground
        sensor2ego_rotation = [1.0, 0.0, 0.0, 0.0]  # Quaternion representation [w, x, y, z]

        calibration_data[cam_name] = {
            'camera_intrinsic': intrinsics.tolist(),
            'distortion_coefficients': dist_coeffs.tolist(),
            'sensor2ego_translation': sensor2ego_translation,
            'sensor2ego_rotation': sensor2ego_rotation,
        }
        
    return calibration_data

if __name__ == '__main__':
    # Directory where you stored/uploaded the camera .npz files
    NPZ_DIR = '.'
    OUTPUT_PKL = 'car_camera_calibration_infos.pkl'
    
    print("--- Generating BEVFormer Calibration Metadata ---")
    calib_info = build_camera_calibration_dict(NPZ_DIR)
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(calib_info, f)
        
    print(f"\n Successfully saved calibration metadata to: {OUTPUT_PKL}")
