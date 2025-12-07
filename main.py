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

YOLO_MODEL_PATH = "yolov8x.pt"

# COCO vehicle class IDs (YOLOv8 + COCO)
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
VEHICLE_CLASS_IDS = {2, 3, 5, 7}

# YOLO's known classes + extra slots for future / custom models
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

# Detection thresholds – tune for your use case
CONF_THRESHOLD = 0.5   # higher -> fewer, more confident detections
IOU_THRESHOLD = 0.5

# How long (in seconds) we keep using the last detection
# when a new frame has no boxes (temporal smoothing).
HOLD_SECONDS = 0.7  # ~700ms


# -------------------------------
# FastAPI app
# -------------------------------

app = FastAPI(
    title="YOLOv8 Vehicle Detection API",
    description="Very accurate vehicle detector (car, truck, bus, motorcycle) using YOLOv8.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # change to your domains in production
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
    # raw xyxy (pixels)
    x1: float
    y1: float
    x2: float
    y2: float

    # size (pixels)
    width: float
    height: float

    # center (pixels)
    cx: float
    cy: float

    # normalized [0,1] for canvas scaling
    nx: float      # x1 / image_width
    ny: float      # y1 / image_height
    nwidth: float  # width / image_width
    nheight: float # height / image_height


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
    image_width: int
    image_height: int
    model: str


# -------------------------------
# Smoothing state (global)
# -------------------------------

# Last non-empty detection and when it was seen
LAST_DETECTION: Optional[DetectionResponse] = None
LAST_DETECTION_TS: float = 0.0


# -------------------------------
# Utils
# -------------------------------

def read_imagefile_to_numpy(data: bytes):
    """
    Safely load an uploaded image (bytes) into a numpy RGB array.
    Returns: (np_array, width, height)
    """
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file.")
    return np.array(image), image.width, image.height


def build_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_w: int,
    image_h: int,
) -> BBox:
    """
    Build full box structure with absolute + normalized coordinates.
    Perfect for drawing directly on a canvas.
    """
    width = x2 - x1
    height = y2 - y1
    cx = x1 + width / 2.0
    cy = y1 + height / 2.0

    # normalized [0,1]
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
    description="""
Upload an image. The API returns detected vehicles (car, truck, bus, motorcycle)
with bounding boxes, confidence scores, and class names.

For frontend canvas:
- Use bbox.x1, bbox.y1, bbox.width, bbox.height in pixels
- Or use bbox.nx, bbox.ny, bbox.nwidth, bbox.nheight for normalized coords
""",
)
async def detect_vehicles(file: UploadFile = File(...)):
    global LAST_DETECTION, LAST_DETECTION_TS

    # Basic validation
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    # Decode image
    img_np, width, height = read_imagefile_to_numpy(data)

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

    detections: List[VehicleDetection] = []

    # Safety: if model returned nothing at all
    if not results:
        # Try smoothing: reuse last detection if still "fresh"
        now = time.time()
        if LAST_DETECTION and (now - LAST_DETECTION_TS) <= HOLD_SECONDS:
            return LAST_DETECTION

        return DetectionResponse(
            success=True,
            num_vehicles=0,
            vehicles=[],
            inference_time_ms=inference_ms,
            image_width=width,
            image_height=height,
            model=YOLO_MODEL_PATH,
        )

    result = results[0]  # batch size = 1
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        # No boxes this frame -> maybe use previous detection
        now = time.time()
        if LAST_DETECTION and (now - LAST_DETECTION_TS) <= HOLD_SECONDS:
            return LAST_DETECTION

        return DetectionResponse(
            success=True,
            num_vehicles=0,
            vehicles=[],
            inference_time_ms=inference_ms,
            image_width=width,
            image_height=height,
            model=YOLO_MODEL_PATH,
        )

    for box in boxes:
        cls_id = int(box.cls.item())
        conf = float(box.conf.item())
        if cls_id not in VEHICLE_CLASS_IDS:
            continue

        # xyxy format
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        bbox = build_bbox(x1, y1, x2, y2, width, height)

        detections.append(
            VehicleDetection(
                bbox=bbox,
                confidence=conf,
                class_id=cls_id,
                class_name=VEHICLE_CLASS_NAMES.get(cls_id, f"class_{cls_id}"),
            )
        )

    # If no vehicles after filtering class IDs -> smoothing again
    if not detections:
        now = time.time()
        if LAST_DETECTION and (now - LAST_DETECTION_TS) <= HOLD_SECONDS:
            return LAST_DETECTION

        return DetectionResponse(
            success=True,
            num_vehicles=0,
            vehicles=[],
            inference_time_ms=inference_ms,
            image_width=width,
            image_height=height,
            model=YOLO_MODEL_PATH,
        )

    # Sort by confidence (descending) – helps downstream logic pick the best one
    detections.sort(key=lambda d: d.confidence, reverse=True)

    response = DetectionResponse(
        success=True,
        num_vehicles=len(detections),
        vehicles=detections,
        inference_time_ms=inference_ms,
        image_width=width,
        image_height=height,
        model=YOLO_MODEL_PATH,
    )

    # 🔹 Update smoothing state with this non-empty detection
    LAST_DETECTION = response
    LAST_DETECTION_TS = time.time()

    return response
