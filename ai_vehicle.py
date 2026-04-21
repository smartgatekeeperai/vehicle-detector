import os
import time
from typing import Dict, List

import numpy as np
import torch
from pydantic import BaseModel
from ultralytics import YOLO

YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8x.pt")
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5"))
HOLD_SECONDS = float(os.getenv("HOLD_SECONDS", "0.7"))

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


def run_yolo_vehicles(img_np: np.ndarray, stream_id: str) -> List[VehicleDetection]:
    image_h, image_w = img_np.shape[:2]

    results = yolo_model.predict(
        img_np,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        verbose=False,
    )

    if not results:
        now = time.time()
        last_dets = last_vehicle_detections_by_stream.get(stream_id, [])
        last_ts = last_vehicle_ts_by_stream.get(stream_id, 0.0)
        if last_dets and (now - last_ts) <= HOLD_SECONDS:
            return last_dets
        return []

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        now = time.time()
        last_dets = last_vehicle_detections_by_stream.get(stream_id, [])
        last_ts = last_vehicle_ts_by_stream.get(stream_id, 0.0)
        if last_dets and (now - last_ts) <= HOLD_SECONDS:
            return last_dets
        return []

    detections: List[VehicleDetection] = []

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())

        if cls_id not in VEHICLE_CLASS_IDS:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = build_vehicle_bbox(x1, y1, x2, y2, image_w, image_h)

        detections.append(
            VehicleDetection(
                bbox=bbox,
                confidence=conf,
                class_id=cls_id,
                class_name=VEHICLE_CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            )
        )

    detections.sort(key=lambda d: d.confidence, reverse=True)
    last_vehicle_detections_by_stream[stream_id] = detections
    last_vehicle_ts_by_stream[stream_id] = time.time()

    return detections