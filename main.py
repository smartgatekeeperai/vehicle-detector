import io
import time
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from ultralytics import YOLO
import torch


# -------------------------------
# Config
# -------------------------------

YOLO_MODEL_PATH = "yolov8n.pt"

# COCO vehicle class IDs (YOLOv8 + COCO)
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

VEHICLE_CLASS_NAMES = {
    2: "car",
    3: "motorcycle",
    4: "airplane",
    5: "bus",
    6: "train",
    7: "truck",
    21: "pickup",
    22: "utility vehicle",
    23: "van",
    24: "jeep",
    25: "suv",
}

CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.5
HOLD_SECONDS = 0.7

# Downscale very large images before inference
MAX_SIDE = 960  # px


# -------------------------------
# FastAPI app
# -------------------------------

app = FastAPI(
    title="YOLOv8 Vehicle Detection API",
    description="Vehicle detector (car, truck, bus, motorcycle) using YOLOv8.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Model loading (on startup)
# -------------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Loading YOLOv8 model on device: {DEVICE}")
model = YOLO(YOLO_MODEL_PATH)
model.to(DEVICE)


# -------------------------------
# Pydantic response models
# -------------------------------

class BBox(BaseModel):
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
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


class DetectionResponse(BaseModel):
    success: bool
    num_vehicles: int
    vehicles: List[VehicleDetection]
    inference_time_ms: float
    total_time_ms: float
    image_width: int
    image_height: int
    model: str


# -------------------------------
# Smoothing state (global)
# -------------------------------

LAST_DETECTION: Optional[DetectionResponse] = None
LAST_DETECTION_TS: float = 0.0


# -------------------------------
# Utils
# -------------------------------

def load_and_resize_image(data: bytes):
    """
    Load bytes -> PIL -> optional resize -> numpy RGB.
    Returns: (np_array, width, height)
    """
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")

    w, h = image.size
    max_side = max(w, h)
    if max_side > MAX_SIDE:
        scale = MAX_SIDE / max_side
        new_size = (int(w * scale), int(h * scale))
        image = image.resize(new_size, Image.LANCZOS)
        w, h = image.size

    return np.array(image), w, h


def build_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_w: int,
    image_h: int,
) -> BBox:
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0

    nx = x1 / image_w
    ny = y1 / image_h
    nwidth = width / image_w
    nheight = height / image_h

    return BBox(
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


def make_detection_from_box(box, image_w: int, image_h: int) -> VehicleDetection:
    cls_id = int(box.cls.item())
    conf = float(box.conf.item())
    x1, y1, x2, y2 = box.xyxy[0].tolist()
    bbox = build_bbox(x1, y1, x2, y2, image_w, image_h)
    class_name = VEHICLE_CLASS_NAMES.get(cls_id, f"class_{cls_id}")
    return VehicleDetection(
        bbox=bbox,
        confidence=conf,
        class_id=cls_id,
        class_name=class_name,
    )


# -------------------------------
# Routes
# -------------------------------

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok", "device": DEVICE, "model": YOLO_MODEL_PATH}


@app.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Detect vehicles in an image",
)
async def detect_vehicles(file: UploadFile = File(...)):
    global LAST_DETECTION, LAST_DETECTION_TS

    request_start = time.time()

    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    img_np, width, height = load_and_resize_image(data)

    # Inference
    t0 = time.time()
    try:
        results = model.predict(
            img_np,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")
    t1 = time.time()
    inference_ms = (t1 - t0) * 1000.0

    # Safety: if model returned nothing at all
    if not results:
        now = time.time()
        if LAST_DETECTION and (now - LAST_DETECTION_TS) <= HOLD_SECONDS:
            return LAST_DETECTION
        total_ms = (time.time() - request_start) * 1000.0
        return DetectionResponse(
            success=True,
            num_vehicles=0,
            vehicles=[],
            inference_time_ms=inference_ms,
            total_time_ms=total_ms,
            image_width=width,
            image_height=height,
            model=YOLO_MODEL_PATH,
        )

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        now = time.time()
        if LAST_DETECTION and (now - LAST_DETECTION_TS) <= HOLD_SECONDS:
            return LAST_DETECTION
        total_ms = (time.time() - request_start) * 1000.0
        return DetectionResponse(
            success=True,
            num_vehicles=0,
            vehicles=[],
            inference_time_ms=inference_ms,
            total_time_ms=total_ms,
            image_width=width,
            image_height=height,
            model=YOLO_MODEL_PATH,
        )

    vehicle_detections: List[VehicleDetection] = []
    all_detections: List[VehicleDetection] = []

    for box in boxes:
        det = make_detection_from_box(box, width, height)
        all_detections.append(det)
        if det.class_id in VEHICLE_CLASS_IDS:
            vehicle_detections.append(det)

    if vehicle_detections:
        detections = vehicle_detections
    elif all_detections:
        detections = all_detections
    else:
        now = time.time()
        if LAST_DETECTION and (now - LAST_DETECTION_TS) <= HOLD_SECONDS:
            return LAST_DETECTION
        total_ms = (time.time() - request_start) * 1000.0
        return DetectionResponse(
            success=True,
            num_vehicles=0,
            vehicles=[],
            inference_time_ms=inference_ms,
            total_time_ms=total_ms,
            image_width=width,
            image_height=height,
            model=YOLO_MODEL_PATH,
        )

    detections.sort(key=lambda d: d.confidence, reverse=True)

    total_ms = (time.time() - request_start) * 1000.0

    response = DetectionResponse(
        success=True,
        num_vehicles=len(detections),
        vehicles=detections,
        inference_time_ms=inference_ms,
        total_time_ms=total_ms,
        image_width=width,
        image_height=height,
        model=YOLO_MODEL_PATH,
    )

    LAST_DETECTION = response
    LAST_DETECTION_TS = time.time()

    return response
