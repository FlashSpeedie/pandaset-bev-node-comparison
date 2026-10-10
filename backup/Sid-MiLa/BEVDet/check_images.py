import pickle
import cv2
import os

with open("custom_nuscenes_infos_clean.pkl", "rb") as f:
    infos = pickle.load(f)
if isinstance(infos, dict) and 'infos' in infos:
    infos = infos['infos']

print("Checking first 5 front camera images...")
for i in range(5):
    img_path = infos[i]['cams']['CAM_FRONT']['data_path']
    if not os.path.exists(img_path):
        img_path = os.path.join('data/nuscenes', img_path)
    
    print(f"\nFrame {i}: Path -> {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        print("  -> ERROR: cv2 could not read this image! File might be missing or corrupt.")
    else:
        print(f"  -> SUCCESS: Image loaded. Shape: {img.shape}. Mean pixel value: {img.mean():.2f}")
        # A normal image should have a mean pixel value somewhere between 50 and 200.
        # If it's 250+, it's completely washed out/white.
