
import numpy as np
import cv2

def rotate_iou_gpu_eval(boxes, query_boxes, criterion=-1, device_id=0):
    """CPU stand-in. criterion: -1 iou, 0 inter/a1, 1 inter/a2, 2 raw inter area."""
    boxes = np.asarray(boxes, dtype=np.float32)
    query_boxes = np.asarray(query_boxes, dtype=np.float32)
    N = 0 if boxes.size == 0 else boxes.shape[0]
    K = 0 if query_boxes.size == 0 else query_boxes.shape[0]
    out = np.zeros((N, K), dtype=np.float64)
    if N == 0 or K == 0:
        return out
    for i in range(N):
        r1 = ((float(boxes[i, 0]), float(boxes[i, 1])),
              (float(boxes[i, 2]), float(boxes[i, 3])),
              float(np.rad2deg(boxes[i, 4])))
        a1 = float(max(boxes[i, 2] * boxes[i, 3], 0.0))
        for j in range(K):
            r2 = ((float(query_boxes[j, 0]), float(query_boxes[j, 1])),
                  (float(query_boxes[j, 2]), float(query_boxes[j, 3])),
                  float(np.rad2deg(query_boxes[j, 4])))
            a2 = float(max(query_boxes[j, 2] * query_boxes[j, 3], 0.0))
            inter = 0.0
            try:
                ret, contour = cv2.rotatedRectangleIntersection(r1, r2)
                if ret > 0 and contour is not None:
                    inter = float(cv2.contourArea(contour))
            except cv2.error:
                inter = 0.0
            if criterion == -1:
                union = a1 + a2 - inter
                out[i, j] = inter / union if union > 1e-8 else 0.0
            elif criterion == 0:
                out[i, j] = inter / a1 if a1 > 1e-8 else 0.0
            elif criterion == 1:
                out[i, j] = inter / a2 if a2 > 1e-8 else 0.0
            else:
                out[i, j] = inter
    return out
