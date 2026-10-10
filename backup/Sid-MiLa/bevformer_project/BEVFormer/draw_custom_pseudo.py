#!/usr/bin/env python3
import os, json, argparse
import numpy as np
import cv2
import mmcv
from pyquaternion import Quaternion

CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
COLORS = {
    'car': (0, 255, 0), 'truck': (255, 128, 0), 'bus': (255, 0, 0),
    'trailer': (128, 0, 255), 'construction_vehicle': (0, 128, 255),
    'pedestrian': (0, 255, 255), 'motorcycle': (255, 255, 0),
    'bicycle': (255, 0, 255), 'traffic_cone': (0, 165, 255),
    'barrier': (128, 128, 128),
}
EDGES = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]

def box_corners_3d(translation, size, rotation_quat):
    w, l, h = size
    x, y, z = l/2, w/2, h/2
    corners = np.array([
        [ x,  y, 0], [ x, -y, 0], [-x, -y, 0], [-x,  y, 0],
        [ x,  y, h], [ x, -y, h], [-x, -y, h], [-x,  y, h],
    ], dtype=np.float64)
    R = Quaternion(rotation_quat).rotation_matrix
    return corners @ R.T + np.array(translation)

def lidar_to_cam(corners_lidar, sensor2lidar_r, sensor2lidar_t):
    R = np.array(sensor2lidar_r).reshape(3, 3)
    t = np.array(sensor2lidar_t).reshape(3)
    return (corners_lidar - t) @ R

def project_cam(corners_cam, K):
    K = np.array(K).reshape(3, 3)
    z = corners_cam[:, 2]
    valid = z > 0.1
    uv = np.zeros((len(corners_cam), 2))
    uv[valid] = (corners_cam[valid] @ K.T)[:, :2] / z[valid, None]
    return uv, valid

def draw_box(img, uv, valid, color, score, name):
    h, w = img.shape[:2]
    for i, j in EDGES:
        if not (valid[i] and valid[j]):
            continue
        p1 = (int(uv[i, 0]), int(uv[i, 1]))
        p2 = (int(uv[j, 0]), int(uv[j, 1]))
        cv2.line(img, p1, p2, color, 2)
    vis = uv[valid]
    if len(vis):
        x, y = int(vis[:, 0].mean()), int(max(15, vis[:, 1].min() - 5))
        cv2.putText(img, f'{name[:6]} {score:.2f}', (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infos', default='custom_nuscenes_infos_abs.pkl')
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', default='visual_results/custom_1805_pseudo')
    ap.add_argument('--score-thr', type=float, default=0.25)
    ap.add_argument('--save-single', action='store_true',
                    help='also save each of 6 cameras separately (10830 files)')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.save_single:
        for c in CAMS:
            os.makedirs(os.path.join(args.out, 'per_cam', c), exist_ok=True)

    data = mmcv.load(args.infos)
    with open(args.results) as f:
        res = json.load(f)
    results = res['results'] if isinstance(res, dict) and 'results' in res else res

    n_grid = 0
    n_single = 0
    for idx, info in enumerate(data['infos']):
        token = info['token']
        boxes = results.get(token, [])
        views = []
        for cam in CAMS:
            cam_info = info['cams'][cam]
            path = cam_info['data_path']
            if not os.path.exists(path):
                path = os.path.join('processed_frames', cam, os.path.basename(path))
            img = cv2.imread(path)
            if img is None:
                img = np.zeros((900, 1600, 3), dtype=np.uint8)
            K = cam_info['cam_intrinsic']
            R = cam_info['sensor2lidar_rotation']
            t = cam_info['sensor2lidar_translation']
            for b in boxes:
                score = float(b.get('detection_score', b.get('score', 0)))
                if score < args.score_thr:
                    continue
                name = b.get('detection_name', b.get('name', 'car'))
                corners = box_corners_3d(b['translation'], b['size'], b['rotation'])
                corners_cam = lidar_to_cam(corners, R, t)
                uv, valid = project_cam(corners_cam, K)
                draw_box(img, uv, valid, COLORS.get(name, (0, 255, 0)), score, name)
            cv2.putText(img, cam, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)
            if args.save_single:
                sp = os.path.join(args.out, 'per_cam', cam, f'{idx:04d}_{token[:8]}.jpg')
                cv2.imwrite(sp, img)
                n_single += 1
            views.append(cv2.resize(img, (640, 360)))
        grid = np.vstack([np.hstack(views[0:3]), np.hstack(views[3:6])])
        cv2.imwrite(os.path.join(args.out, f'{idx:04d}_{token[:8]}.jpg'), grid)
        n_grid += 1
        if (idx + 1) % 50 == 0:
            print(f'Progress: {idx+1}/{len(data["infos"])} frames')

    print('Done grids:', n_grid)
    if args.save_single:
        print('Done single-cam images:', n_single)
    print('Output dir:', args.out)

if __name__ == '__main__':
    main()
