import pickle
import numpy as np

pkl_path = 'custom_car_predictions.pkl'

try:
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)

    print("=" * 50)
    print("      BEVFormer Benchmark Analysis Summary      ")
    print("=" * 50)
    print(f"✅ Total frames processed: {len(results)}")

    total_boxes = 0
    scores = []

    for frame in results:
        if 'pts_bbox' in frame:
            boxes = frame['pts_bbox']['boxes_3d']
            frame_scores = frame['pts_bbox']['scores_3d']
            total_boxes += len(boxes)
            scores.extend(frame_scores.numpy() if hasattr(frame_scores, 'numpy') else frame_scores)

    print(f"📦 Total 3D objects detected: {total_boxes}")
    if len(results) > 0:
        print(f"📊 Average objects per frame: {total_boxes / len(results):.2f}")
    if len(scores) > 0:
        print(f"🎯 Average detection confidence: {np.mean(scores):.4f}")
    print("=" * 50)

except FileNotFoundError:
    print(f"⏳ '{pkl_path}' not found yet. Run this after Job 32701852 finishes!")
