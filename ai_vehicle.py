# ai_vehicle.py

import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from ultralytics import YOLO

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8x.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.45"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5"))
HOLD_SECONDS = float(os.getenv("HOLD_SECONDS", "0.7"))

# Keep only one main pass for speed
IMG_SIZE_MAIN = int(os.getenv("IMG_SIZE_MAIN", "640"))
MIN_VEHICLE_BOX_AREA_RATIO = float(os.getenv("MIN_VEHICLE_BOX_AREA_RATIO", "0.0025"))
MAX_VEHICLE_BOX_AREA_RATIO = float(os.getenv("MAX_VEHICLE_BOX_AREA_RATIO", "0.98"))

# Plate-guided fallback
ENABLE_PLATE_GUIDED_VEHICLE_FALLBACK = (
    str(os.getenv("ENABLE_PLATE_GUIDED_VEHICLE_FALLBACK", "true")).lower() == "true"
)
PLATE_GUIDED_CONF_THRESHOLD = float(os.getenv("PLATE_GUIDED_CONF_THRESHOLD", "0.18"))
PLATE_GUIDED_IMG_SIZE = int(os.getenv("PLATE_GUIDED_IMG_SIZE", "640"))
FORCED_VEHICLE_SYNTHETIC_CONF = float(os.getenv("FORCED_VEHICLE_SYNTHETIC_CONF", "0.35"))

VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    21: "pickup",
    22: "utility vehicle",
    23: "van",
    24: "jeep",
    25: "suv",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"[AI_VEHICLE] Loading YOLO model on device: {DEVICE}")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(DEVICE)

last_vehicle_detections_by_stream: Dict[str, List["VehicleDetection"]] = {}
last_vehicle_ts_by_stream: Dict[str, float] = {}


class VehicleBBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    width: float
    height: float
    cx: float
    cy: float
    nx: float
    ny: float
    nwidth: float
    nheight: float


class VehicleDetection(BaseModel):
    bbox: VehicleBBox
    confidence: float
    class_id: int
    class_name: str


def build_vehicle_bbox(x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int) -> VehicleBBox:
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0

    nx = x1 / image_w
    ny = y1 / image_h
    nwidth = width / image_w
    nheight = height / image_h

    return VehicleBBox(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        width=width,
        height=height,
        cx=cx,
        cy=cy,
        nx=nx,
        ny=ny,
        nwidth=nwidth,
        nheight=nheight,
    )


def _get_last_or_empty(stream_id: str) -> List["VehicleDetection"]:
    now = time.time()
    last_dets = last_vehicle_detections_by_stream.get(stream_id, [])
    last_ts = last_vehicle_ts_by_stream.get(stream_id, 0.0)
    if last_dets and (now - last_ts) <= HOLD_SECONDS:
        return last_dets
    return []


def _clip_box(x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(x1), float(image_w - 1)))
    y1 = max(0.0, min(float(y1), float(image_h - 1)))
    x2 = max(x1 + 1.0, min(float(x2), float(image_w)))
    y2 = max(y1 + 1.0, min(float(y2), float(image_h)))
    return x1, y1, x2, y2


def _box_area_ratio(x1: float, y1: float, x2: float, y2: float, image_w: int, image_h: int) -> float:
    box_area = max(1.0, (x2 - x1) * (y2 - y1))
    img_area = max(1.0, image_w * image_h)
    return box_area / img_area


def _iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0

    return inter_area / union


def _dedupe_candidates(candidates: List[dict], iou_threshold: float = 0.55) -> List[dict]:
    if not candidates:
        return []

    candidates = sorted(
        candidates,
        key=lambda x: (x["confidence"], x["area_ratio"]),
        reverse=True,
    )

    kept: List[dict] = []

    for cand in candidates:
        duplicate = False
        for existing in kept:
            if cand["class_id"] != existing["class_id"]:
                continue
            if _iou_xyxy(cand["bbox_xyxy"], existing["bbox_xyxy"]) >= iou_threshold:
                duplicate = True
                break

        if not duplicate:
            kept.append(cand)

    return kept


def _extract_candidates(
    result,
    image_w: int,
    image_h: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> List[dict]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    candidates: List[dict] = []

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        if cls_id not in VEHICLE_CLASS_IDS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        x1 += offset_x
        y1 += offset_y
        x2 += offset_x
        y2 += offset_y

        x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, image_w, image_h)
        area_ratio = _box_area_ratio(x1, y1, x2, y2, image_w, image_h)

        if area_ratio < MIN_VEHICLE_BOX_AREA_RATIO:
            continue
        if area_ratio > MAX_VEHICLE_BOX_AREA_RATIO:
            continue

        candidates.append(
            {
                "bbox_xyxy": (x1, y1, x2, y2),
                "confidence": conf,
                "class_id": cls_id,
                "class_name": VEHICLE_CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
                "area_ratio": area_ratio,
            }
        )

    return candidates


def _predict_once(
    img_np: np.ndarray,
    image_w: int,
    image_h: int,
    conf_threshold: float,
    imgsz: int,
    offset_x: int = 0,
    offset_y: int = 0,
) -> List[dict]:
    results = yolo_model.predict(
        img_np,
        conf=conf_threshold,
        iou=IOU_THRESHOLD,
        imgsz=imgsz,
        device=DEVICE,
        verbose=False,
    )

    if not results:
        return []

    return _extract_candidates(
        results[0],
        image_w=image_w,
        image_h=image_h,
        offset_x=offset_x,
        offset_y=offset_y,
    )


def _plate_to_vehicle_search_roi(
    plate_bbox: dict,
    image_w: int,
    image_h: int,
) -> Tuple[int, int, int, int]:
    """
    Build a larger ROI around detected plate where the vehicle likely exists.
    Fast and avoids full-frame re-run.
    """
    px1 = int(plate_bbox.get("x1", 0))
    py1 = int(plate_bbox.get("y1", 0))
    px2 = int(plate_bbox.get("x2", 0))
    py2 = int(plate_bbox.get("y2", 0))

    pw = max(1, px2 - px1)
    ph = max(1, py2 - py1)
    pcx = px1 + pw / 2.0

    x1 = int(round(pcx - (pw * 3.0)))
    x2 = int(round(pcx + (pw * 3.0)))
    y1 = int(round(py1 - (ph * 6.0)))
    y2 = int(round(py2 + (ph * 6.5)))

    x1, y1, x2, y2 = _clip_box(x1, y1, x2, y2, image_w, image_h)
    return int(x1), int(y1), int(x2), int(y2)


def _guess_vehicle_class_from_plate_roi(roi_w: float, roi_h: float) -> Tuple[int, str]:
    aspect = roi_w / max(1.0, roi_h)

    if aspect < 0.95:
        return 3, VEHICLE_CLASS_NAMES.get(3, "motorcycle")

    return 2, VEHICLE_CLASS_NAMES.get(2, "car")


def _build_synthetic_vehicle_from_plate(
    plate_bbox: dict,
    image_w: int,
    image_h: int,
) -> VehicleDetection:
    x1, y1, x2, y2 = _plate_to_vehicle_search_roi(plate_bbox, image_w, image_h)
    class_id, class_name = _guess_vehicle_class_from_plate_roi(x2 - x1, y2 - y1)

    bbox = build_vehicle_bbox(x1, y1, x2, y2, image_w, image_h)

    return VehicleDetection(
        bbox=bbox,
        confidence=FORCED_VEHICLE_SYNTHETIC_CONF,
        class_id=class_id,
        class_name=class_name,
    )


def force_vehicle_from_plates(
    img_np: np.ndarray,
    plates: Optional[List[dict]],
    stream_id: str,
) -> List[VehicleDetection]:
    """
    Called only when:
    - normal vehicle detection returned nothing
    - at least one plate exists

    Strategy:
    1. Use the best plate bbox
    2. Run one small local YOLO search around that plate
    3. If still nothing, synthesize a vehicle box around the plate
    """
    if not ENABLE_PLATE_GUIDED_VEHICLE_FALLBACK:
        return []

    if not plates:
        return []

    image_h, image_w = img_np.shape[:2]

    valid_plates = [p for p in plates if isinstance(p, dict) and p.get("bbox")]
    if not valid_plates:
        return []

    def plate_sort_key(p: dict):
        bbox = p.get("bbox") or {}
        area = max(0, int(bbox.get("x2", 0)) - int(bbox.get("x1", 0))) * max(
            0, int(bbox.get("y2", 0)) - int(bbox.get("y1", 0))
        )
        conf = float((p.get("ocr") or {}).get("confidence") or 0.0)
        return (conf, area)

    valid_plates.sort(key=plate_sort_key, reverse=True)
    best_plate = valid_plates[0]
    plate_bbox = best_plate["bbox"]

    rx1, ry1, rx2, ry2 = _plate_to_vehicle_search_roi(plate_bbox, image_w, image_h)
    roi = img_np[ry1:ry2, rx1:rx2]

    if roi.size > 0:
        local_candidates = _predict_once(
            img_np=roi,
            image_w=image_w,
            image_h=image_h,
            conf_threshold=PLATE_GUIDED_CONF_THRESHOLD,
            imgsz=PLATE_GUIDED_IMG_SIZE,
            offset_x=rx1,
            offset_y=ry1,
        )

        local_candidates = _dedupe_candidates(local_candidates, iou_threshold=0.55)

        if local_candidates:
            detections: List[VehicleDetection] = []
            for item in local_candidates:
                x1, y1, x2, y2 = item["bbox_xyxy"]
                bbox = build_vehicle_bbox(x1, y1, x2, y2, image_w, image_h)

                detections.append(
                    VehicleDetection(
                        bbox=bbox,
                        confidence=float(item["confidence"]),
                        class_id=int(item["class_id"]),
                        class_name=str(item["class_name"]),
                    )
                )

            detections.sort(
                key=lambda d: (
                    float(d.confidence),
                    float(d.bbox.width * d.bbox.height),
                ),
                reverse=True,
            )

            last_vehicle_detections_by_stream[stream_id] = detections
            last_vehicle_ts_by_stream[stream_id] = time.time()

            print(
                f"[AI_VEHICLE] stream={stream_id} plate-guided local detection "
                f"count={len(detections)} top={detections[0].class_name} "
                f"conf={detections[0].confidence:.3f}"
            )
            return detections

    synthetic = _build_synthetic_vehicle_from_plate(plate_bbox, image_w, image_h)
    detections = [synthetic]

    last_vehicle_detections_by_stream[stream_id] = detections
    last_vehicle_ts_by_stream[stream_id] = time.time()

    print(
        f"[AI_VEHICLE] stream={stream_id} synthetic vehicle box from plate "
        f"class={synthetic.class_name} conf={synthetic.confidence:.3f}"
    )

    return detections


def run_yolo_vehicles(img_np: np.ndarray, stream_id: str) -> List[VehicleDetection]:
    image_h, image_w = img_np.shape[:2]

    candidates = _predict_once(
        img_np=img_np,
        image_w=image_w,
        image_h=image_h,
        conf_threshold=CONF_THRESHOLD,
        imgsz=IMG_SIZE_MAIN,
    )

    if not candidates:
        return _get_last_or_empty(stream_id)

    candidates = _dedupe_candidates(candidates, iou_threshold=0.55)

    if not candidates:
        return _get_last_or_empty(stream_id)

    detections: List[VehicleDetection] = []

    for item in candidates:
        x1, y1, x2, y2 = item["bbox_xyxy"]
        bbox = build_vehicle_bbox(x1, y1, x2, y2, image_w, image_h)

        detections.append(
            VehicleDetection(
                bbox=bbox,
                confidence=float(item["confidence"]),
                class_id=int(item["class_id"]),
                class_name=str(item["class_name"]),
            )
        )

    detections.sort(
        key=lambda d: (
            float(d.confidence),
            float(d.bbox.width * d.bbox.height),
        ),
        reverse=True,
    )

    last_vehicle_detections_by_stream[stream_id] = detections
    last_vehicle_ts_by_stream[stream_id] = time.time()

    if detections:
        top = detections[0]
        print(
            f"[AI_VEHICLE] stream={stream_id} detections={len(detections)} "
            f"top={top.class_name} conf={top.confidence:.3f}"
        )

    return detections