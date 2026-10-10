import numpy as np
import mmcv

camera_config = {}

# Store intrinsically calibrated camera parameters
for i in range(1, 7):
    filename = f"No{i}_camera.npz"
    data = np.load(filename)
    
    camera_config[f"cam_{i}"] = {
        "camera_matrix": data["camera_matrix"].tolist(),
        "dist_coeffs": data["dist_coeffs"].flatten().tolist(),
        "fx": float(data["camera_matrix"][0, 0]),
        "fy": float(data["camera_matrix"][1, 1]),
        "cx": float(data["camera_matrix"][0, 2]),
        "cy": float(data["camera_matrix"][1, 2]),
    }

mmcv.dump(camera_config, "camera_config.pkl")
print("✅ Successfully saved camera parameters to camera_config.pkl")
