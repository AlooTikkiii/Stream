import numpy as np
from typing import List, Tuple, Any, Optional, Iterable, Set
from queue import Full, Queue, Empty


Det = Tuple[float, float, float, float, int, float]  # x1, y1, x2, y2, class_id, confidence
Match = Tuple[Det, Det, int, float]  # left_det, right_det

def extract_topk_detections(detections, k : int):

    if detections.boxes is None or len(detections.boxes) == 0:
        return []
    
    confidences = detections.boxes.conf.cpu().numpy()
    boxes = detections.boxes.xyxy.cpu().numpy()
    class_ids = detections.boxes.cls.cpu().numpy().astype(int)

    n = len(confidences)
    if n <= k:
        idxs = np.argsort(-confidences)
    else:
        top_k = np.argpartition(-confidences, k)[:k]
        idxs = top_k[np.argsort(-confidences[top_k])]

    dets = []
    for i in idxs:
        x1, y1, x2, y2 = boxes[i]
        dets.append(
            (float(x1), float(y1), float(x2), float(y2), int(class_ids[i]), float(confidences[i]))
        )
    return dets

def postprocess_detections( dets_combined: List[Any], H: int, W: int, iou_threshold: float = 0.5, area_ratio_thresh: float = 0.90, edge_tol: float = 3.0, classes: Optional[Iterable[int]] = None) -> List[Match]:
    def clamp_box(x1, y1, x2, y2, w, h):
        x1 = max(0.0, min(float(w), x1))
        x2 = max(0.0, min(float(w), x2))
        y1 = max(0.0, min(float(h), y1))
        y2 = max(0.0, min(float(h), y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    def box_iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        iw = max(0.0, inter_x2 - inter_x1)
        ih = max(0.0, inter_y2 - inter_y1)
        inter = iw * ih
        if inter <= 0.0:
            return 0.0
        a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        denom = a_area + b_area - inter
        return float(inter / denom) if denom > 0.0 else 0.0

    def is_full_image_box(
        x1: float, y1: float, x2: float, y2: float,
        img_w: int, img_h: int,
        area_ratio_thresh: float = 0.90,
        edge_tol: float = 3.0
    ) -> bool:
        x1 = max(0.0, min(float(img_w), x1))
        x2 = max(0.0, min(float(img_w), x2))
        y1 = max(0.0, min(float(img_h), y1))
        y2 = max(0.0, min(float(img_h), y2))

        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if bw <= 0.0 or bh <= 0.0:
            return False

        area_ratio = (bw * bh) / float(img_w * img_h)

        near_left   = x1 <= edge_tol
        near_top    = y1 <= edge_tol
        near_right  = (img_w - x2) <= edge_tol
        near_bottom = (img_h - y2) <= edge_tol

        return (area_ratio >= area_ratio_thresh) and near_left and near_top and near_right and near_bottom

    def match_stereo(left: List[Det], right: List[Det]) -> List[Tuple[Det, Det, int, float]]:
        """
        Greedy one-to-one matching by (class_id + IoU).
        Returns: (left_det, right_det, class_id, iou)
        """
        candidates: List[Tuple[float, int, int]] = []

        for iL, (lx1, ly1, lx2, ly2, lcls, lconf) in enumerate(left):
            for iR, (rx1, ry1, rx2, ry2, rcls, rconf) in enumerate(right):
                if lcls != rcls:
                    continue
                iou = box_iou(
                    (lx1, ly1, lx2, ly2),
                    (rx1, ry1, rx2, ry2),
                )
                if iou >= iou_threshold:
                    candidates.append((iou, iL, iR))

        if not candidates:
            return []

        # Highest IoU first
        candidates.sort(key=lambda t: t[0], reverse=True)

        usedL, usedR = set(), set()
        matches: List[Tuple[Det, Det, int, float]] = []

        for iou, iL, iR in candidates:
            if iL in usedL or iR in usedR:
                continue

            usedL.add(iL)
            usedR.add(iR)

            L = left[iL]
            R = right[iR]
            cls_id = int(L[4])  # same as R[4]

            matches.append((L, R, cls_id, float(iou)))

        return matches

    # -------------------------
    # Split dets into left/right
    # -------------------------
    left: List[Det] = []
    right: List[Det] = []

    for det in dets_combined:
        x1, y1, x2, y2, class_id, conf = det
        cx = (x1 + x2) / 2.0

        if cx < W:
            lx1, ly1, lx2, ly2 = clamp_box(x1, y1, x2, y2, W, H)
            if not is_full_image_box(lx1, ly1, lx2, ly2, W, H, area_ratio_thresh, edge_tol):
                left.append((lx1, ly1, lx2, ly2, int(class_id), float(conf)))
        else:
            rx1, ry1, rx2, ry2 = clamp_box(x1 - W, y1, x2 - W, y2, W, H)
            if not is_full_image_box(rx1, ry1, rx2, ry2, W, H, area_ratio_thresh, edge_tol):
                right.append((rx1, ry1, rx2, ry2, int(class_id), float(conf)))

    if not left or not right:
        return []

    # -------------------------
    # Match and optional class filter
    # -------------------------
    matched_pairs = match_stereo(left, right)

    if classes is not None:
        cls_set: Set[int] = set(int(c) for c in classes)
        matched_pairs = [
            (L, R, cls_id, iou)
            for (L, R, cls_id, iou) in matched_pairs
            if cls_id in cls_set
        ]

    return matched_pairs

