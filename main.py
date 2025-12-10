import io
import os
import time
import json
from functools import lru_cache
from typing import List, Optional, Literal, get_args, Any, Dict

from dotenv import load_dotenv  # Load env vars from .env

# ---------------------------------------------------------
# Load .env (current dir + parent, to catch root .env)
# ---------------------------------------------------------
# First load from current directory (vehicle-detector/.env)
load_dotenv()
# Also try parent directory (smartgatekeeperai/.env)
parent_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(parent_env_path):
    load_dotenv(parent_env_path, override=False)

import numpy as np
import asyncpg
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from PIL import Image

from ultralytics import YOLO
import torch

from pusher import Pusher

from fast_alpr import ALPR
from fast_alpr.default_detector import PlateDetectorModel
from fast_alpr.default_ocr import OcrModel

# =========================================================
# YOLO CONFIG (VEHICLES)
# =========================================================

YOLO_MODEL_PATH = "yolov8x.pt"

# COCO vehicle class IDs (YOLOv8 + COCO)
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
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

CONF_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5
HOLD_SECONDS = 0.7  # smoothing window for vehicles (YOLO)

# =========================================================
# FASTALPR CONFIG (PLATES)
# =========================================================

DETECTOR_MODELS: List[PlateDetectorModel] = list(get_args(PlateDetectorModel))
OCR_MODELS: List[OcrModel] = list(get_args(OcrModel))

# (optional) Move a specific OCR model to front if desired
if "cct-s-v1-global-model" in OCR_MODELS:
    OCR_MODELS.remove("cct-s-v1-global-model")
    OCR_MODELS.insert(0, "cct-s-v1-global-model")

DetectorName = Literal[tuple(DETECTOR_MODELS)]  # type: ignore
OcrName = Literal[tuple(OCR_MODELS)]  # type: ignore


@lru_cache(maxsize=8)
def get_alpr(detector_model: str, ocr_model: str) -> ALPR:
    if detector_model not in DETECTOR_MODELS:
        raise ValueError(f"Unknown detector_model {detector_model}")
    if ocr_model not in OCR_MODELS:
        raise ValueError(f"Unknown ocr_model {ocr_model}")
    return ALPR(detector_model=detector_model, ocr_model=ocr_model)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Smart Gate Keeper - YOLOv8x + FastALPR",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# DEVICE + YOLO MODEL
# =========================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INIT] Loading YOLOv8 model on device: {DEVICE}")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(DEVICE)

# =========================================================
# GATE STATE + DB + PUSHER (Node.js logic port)
# =========================================================

# Global gate state equivalent to Node's gateState
gate_state: Dict[str, Any] = {
    "vehicleFound": False,  # boolean (true/false)
    "plate": None,
    "driver": None,         # DB JSONB "Driver"
    "vehicle": None,        # { brand, model, type }
    "lastUpdate": None,     # ms epoch
}

# When a registered plate is detected, we "lock" that vehicle for a while.
current_registered_gate_state: Optional[Dict[str, Any]] = None
current_registered_ts: int = 0  # ms epoch when registered plate was confirmed

# How long we keep the registered vehicle active WHILE a vehicle is present
PASS_LOCK_MS = 10_000  # 10 seconds

# How long after NO vehicle is seen we still keep last gate_state before reset
CLEAR_AFTER_MS = 5_000  # 5 seconds

# =========================================================
# LATEST FRAMES (Node-style `latestFrames = new Map()`)
# =========================================================

# Python equivalent of: const latestFrames = new Map();
# Key: stream_id (str) → Value: { "buffer": bytes, "mimetype": str, "ts": int }
latest_frames: Dict[str, Dict[str, Any]] = {}

# --- DB (PostgreSQL) ---
# Try direct DATABASE_URL first
DATABASE_URL = os.getenv("DATABASE_URL")

# If not set, build it from DB_* vars like your Node app
if not DATABASE_URL:
    DB_USER = os.getenv("DB_USER", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "")

    if DB_USER and DB_NAME:
        # asyncpg uses standard postgresql:// URI
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        print("[DB] Constructed DATABASE_URL from DB_* env vars:", DATABASE_URL)
    else:
        print("[DB] Missing DB_USER or DB_NAME; cannot construct DATABASE_URL.")

if DATABASE_URL:
    print("[DB] Using DATABASE_URL:", DATABASE_URL)
else:
    print("[DB] DATABASE_URL is still not set. DB queries will fail.")


@app.on_event("startup")
async def startup():
    if not DATABASE_URL:
        print("[WARN] DATABASE_URL is not set; DB queries will fail.")
        return
    app.state.db_pool = await asyncpg.create_pool(DATABASE_URL)
    print("[INIT] asyncpg pool created")


async def db_query(sql: str, params: List[Any] = None) -> List[Dict[str, Any]]:
    """Equivalent to Node dbQuery using asyncpg."""
    params = params or []
    pool = getattr(app.state, "db_pool", None)
    if pool is None:
        raise RuntimeError(
            "DB pool is not initialized. Set DATABASE_URL and ensure startup event runs."
        )

    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
        return [dict(r) for r in rows]


# --- Pusher client (Node-style env logic: cloud + optional local) ---

PUSHER_APP_ID = os.getenv("PUSHER_APP_ID")
PUSHER_KEY = os.getenv("PUSHER_KEY", "")
PUSHER_SECRET = os.getenv("PUSHER_SECRET", "")
PUSHER_CLUSTER = os.getenv("PUSHER_CLUSTER", "ap1")
PUSHER_HOST = os.getenv("PUSHER_HOST")
PUSHER_PORT = os.getenv("PUSHER_PORT")
USE_LOCAL_PUSHER = os.getenv("USE_LOCAL_PUSHER", "false").lower() == "true"

# Same logic as Node:
# const isLocalEnv = process.env.VERCEL !== '1' && process.env.NODE_ENV !== 'production';
is_local_env = os.getenv("VERCEL") != "1" and os.getenv("NODE_ENV") != "production"

# Debug-print Pusher env values
print(
    "[Pusher] env values:",
    "APP_ID=", repr(PUSHER_APP_ID),
    "KEY=", repr(PUSHER_KEY),
    "SECRET set=", bool(PUSHER_SECRET),
    "CLUSTER=", repr(PUSHER_CLUSTER),
    "HOST=", repr(PUSHER_HOST),
    "PORT=", repr(PUSHER_PORT),
    "USE_LOCAL_PUSHER=", USE_LOCAL_PUSHER,
    "is_local_env=", is_local_env,
)

if not PUSHER_APP_ID or PUSHER_APP_ID.strip() == "":
    # No valid APP_ID → create dummy client so the app can still start
    print("[WARN] PUSHER_APP_ID is not set or invalid. Using DummyPusher (no real-time events).")

    class DummyPusher:
        def trigger(self, channel, event, data):
            print(f"[DummyPusher] trigger → channel={channel}, event={event}, data={data}")

    pusher_client: Any = DummyPusher()
else:
    # Default: Pusher CLOUD (like Node)
    client_kwargs: Dict[str, Any] = {
        "app_id": PUSHER_APP_ID,
        "key": PUSHER_KEY,
        "secret": PUSHER_SECRET,
        "cluster": PUSHER_CLUSTER,
        "ssl": True,  # useTLS: true
    }

    # Optional local mode for Soketi/other compatible server
    # if (isLocalEnv && USE_LOCAL_PUSHER && PUSHER_HOST)
    if is_local_env and USE_LOCAL_PUSHER and PUSHER_HOST:
        host = PUSHER_HOST
        port = int(PUSHER_PORT or "6001")
        print(f"[Pusher] Using LOCAL server at {host}:{port}")
        # For local: HTTP (no TLS), host/port; cluster is not used
        client_kwargs.pop("cluster", None)
        client_kwargs.update(
            {
                "host": host,
                "port": port,
                "ssl": False,
            }
        )
    else:
        # Cloud mode (works on localhost + prod)
        print(f"[Pusher] Using CLOUD (cluster={PUSHER_CLUSTER}, TLS=True)")

    pusher_client = Pusher(**client_kwargs)
    print("[INIT] Pusher client initialized:", client_kwargs)


# =========================================================
# MODELS (Pydantic)
# =========================================================

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


LAST_VEHICLE_DETECTIONS: List[VehicleDetection] = []
LAST_VEHICLE_TS: float = 0.0


# =========================================================
# HELPERS
# =========================================================

def load_image_to_numpy(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    return np.array(img)


def build_vehicle_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_w: int,
    image_h: int,
) -> VehicleBBox:
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


def run_yolo_vehicles(img_np: np.ndarray) -> List[VehicleDetection]:
    global LAST_VEHICLE_DETECTIONS, LAST_VEHICLE_TS

    image_h, image_w = img_np.shape[:2]

    try:
        results = yolo_model.predict(
            img_np,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            verbose=False,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLO inference error: {e}")

    if not results:
        now = time.time()
        if LAST_VEHICLE_DETECTIONS and (now - LAST_VEHICLE_TS) <= HOLD_SECONDS:
            return LAST_VEHICLE_DETECTIONS
        return []

    result = results[0]
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        now = time.time()
        if LAST_VEHICLE_DETECTIONS and (now - LAST_VEHICLE_TS) <= HOLD_SECONDS:
            return LAST_VEHICLE_DETECTIONS
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

    # sort by confidence (highest first)
    detections.sort(key=lambda d: d.confidence, reverse=True)

    LAST_VEHICLE_DETECTIONS = detections
    LAST_VEHICLE_TS = time.time()

    return detections


def serialize_alpr_result(result) -> dict:
    ocr_obj = getattr(result, "ocr", None)
    ocr = None
    if ocr_obj is not None:
        raw_conf = getattr(ocr_obj, "confidence", 0.0)
        if isinstance(raw_conf, (list, tuple)):
            conf_val = float(sum(raw_conf) / len(raw_conf)) if raw_conf else 0.0
        else:
            conf_val = float(raw_conf or 0.0)
        ocr = {
            "text": getattr(ocr_obj, "text", None),
            "confidence": conf_val,
        }

    bbox = None
    detection = getattr(result, "detection", None)
    if detection is not None:
        bb = getattr(detection, "bounding_box", None)
        if bb is not None:
            bbox = {
                "x1": int(getattr(bb, "x1", 0)),
                "y1": int(getattr(bb, "y1", 0)),
                "x2": int(getattr(bb, "x2", 0)),
                "y2": int(getattr(bb, "y2", 0)),
            }

    return {"ocr": ocr, "bbox": bbox}


def run_alpr(img_np: np.ndarray, detector_name: str, ocr_name: str) -> dict:
    try:
        alpr = get_alpr(detector_name, ocr_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        results = alpr.predict(img_np)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ALPR error: {e}")

    plates = [serialize_alpr_result(r) for r in results] if results else []

    return {
        "model": {
            "detector_model": detector_name,
            "ocr_model": ocr_name,
        },
        "plates": plates,
    }


def normalize_plate_text(plates: List[dict]) -> List[str]:
    """
    Convert FastALPR plate results into normalized strings:
    - take plate['ocr']['text']
    - remove spaces and dashes
    - lowercase
    """
    cleaned: List[str] = []
    for p in plates:
        ocr = p.get("ocr") or {}
        text = ocr.get("text")
        if not text:
            continue
        normalized = (
            text.replace(" ", "")
            .replace("-", "")
            .replace("\t", "")
            .replace("\n", "")
            .lower()
        )
        cleaned.append(normalized)
    print("[GateState] cleaned_plates:", cleaned)
    return cleaned


# =========================================================
# GATE STATE LOGIC (with plate lock + smooth clearing)
# =========================================================

async def update_gate_state_and_push(
    vehicles: List[VehicleDetection],
    plates: List[dict],
) -> Dict[str, Any]:
    """
    Logic with:
    1) Plate lock: if a registered plate is found, keep returning that same
       driver/vehicle for up to PASS_LOCK_MS while a vehicle is present.
    2) Smooth clear: after no vehicle is detected, keep last state for up to
       CLEAR_AFTER_MS, then reset for the next vehicle.
    """
    global gate_state, current_registered_gate_state, current_registered_ts

    now_ms = int(time.time() * 1000)
    vehicle_count = len(vehicles)

    # ---------------------------------------------
    # A. If we already have a registered vehicle, apply lock rules
    # ---------------------------------------------
    if current_registered_gate_state is not None:
        lock_age = now_ms - current_registered_ts

        # A1: Vehicle still present & within PASS_LOCK_MS -> keep registered state
        if vehicle_count > 0 and lock_age <= PASS_LOCK_MS:
            gate_state = {
                **current_registered_gate_state,
                "vehicleFound": True,
                "lastUpdate": now_ms,
            }
            try:
                pusher_client.trigger("gate-channel", "gate-update", gate_state)
            except Exception as e:
                print("[Pusher] Error (lock, vehicle present):", e)
            print("[GateState] LOCK active (vehicle present). gate_state:", gate_state)
            return gate_state

        # A2: No vehicle present, but we keep showing last state for CLEAR_AFTER_MS
        if vehicle_count == 0:
            last_update = gate_state.get("lastUpdate") or current_registered_ts
            if now_ms - last_update <= CLEAR_AFTER_MS:
                # Keep last gate_state, just rebroadcast
                try:
                    pusher_client.trigger("gate-channel", "gate-update", gate_state)
                except Exception as e:
                    print("[Pusher] Error (lock, smooth clear window):", e)
                print("[GateState] LOCK smooth clear (no vehicle, but within CLEAR_AFTER_MS). gate_state:", gate_state)
                return gate_state

        # A3: Lock expired + either no vehicle or a new scenario -> clear lock and continue
        print("[GateState] LOCK expired or new cycle. Clearing current_registered_*")
        current_registered_gate_state = None
        current_registered_ts = 0
        # fall through to normal logic below

    # ---------------------------------------------
    # B. No active lock (normal logic)
    # ---------------------------------------------

    # B1: No vehicles & no plates at all
    print("# B1) No vehicles & no plates")
    if (not vehicles) and (not plates):
        last_update = gate_state.get("lastUpdate")
        # If we had a previous vehicle, keep it for CLEAR_AFTER_MS
        if gate_state.get("vehicleFound") and last_update and (now_ms - last_update <= CLEAR_AFTER_MS):
            try:
                pusher_client.trigger("gate-channel", "gate-update", gate_state)
            except Exception as e:
                print("[Pusher] Error (B1 smooth no-vehicle window):", e)
            print("[GateState] B1 smooth hold (no vehicles/plates but within CLEAR_AFTER_MS). gate_state:", gate_state)
            return gate_state

        # Otherwise, hard reset
        gate_state = {
            "vehicleFound": False,
            "plate": None,
            "driver": None,
            "vehicle": None,
            "lastUpdate": None,
        }
        try:
            pusher_client.trigger("gate-channel", "gate-update", gate_state)
        except Exception as e:
            print("[Pusher] Error (B1 reset):", e)
        print("[GateState] B1 reset. gate_state:", gate_state)
        return gate_state

    # B2: Vehicles found but no plates
    print("# B2) Vehicles found but no plates")
    if vehicles and not plates:
        # Just mark vehicle present but unknown
        gate_state = {
            "vehicleFound": True,
            "plate": gate_state.get("plate"),   # keep last plate if any
            "driver": gate_state.get("driver"), # keep last driver if any
            "vehicle": gate_state.get("vehicle"),
            "lastUpdate": now_ms,
        }
        try:
            pusher_client.trigger("gate-channel", "gate-update", gate_state)
        except Exception as e:
            print("[Pusher] Error (B2 vehicles only):", e)
        print("[GateState] B2 vehicles only. gate_state:", gate_state)
        return gate_state

    # B3: We have plates; normalize for DB search
    cleaned_plates = normalize_plate_text(plates)

    print("# B3) We have plates; normalize for DB search")
    if not cleaned_plates:
        # behave like "vehicles only" if plate texts are empty
        gate_state = {
            "vehicleFound": vehicle_count > 0,
            "plate": gate_state.get("plate"),
            "driver": gate_state.get("driver"),
            "vehicle": gate_state.get("vehicle"),
            "lastUpdate": now_ms,
        }
        try:
            pusher_client.trigger("gate-channel", "gate-update", gate_state)
        except Exception as e:
            print("[Pusher] Error (B3 no cleaned plates):", e)
        print("[GateState] B3 no cleaned plates. gate_state:", gate_state)
        return gate_state

    # B4: DB lookup (PostgreSQL, same logic as Node with UNNEST)
    print("# B4) DB lookup (PostgreSQL, same logic as Node with UNNEST)")
    try:
        sql = """
            SELECT v.*, p.cleaned_input
            FROM dbo."Vehicles" v
            CROSS JOIN UNNEST($1::text[]) AS p(cleaned_input)
            WHERE 
                LOWER(REPLACE(REPLACE(v."PlateNumber", ' ', ''), '-', '')) = p.cleaned_input
                AND v."Active" = true
            LIMIT 1
        """
        print("[GateState] running DB query with:", cleaned_plates)
        rows = await db_query(sql, [cleaned_plates])
        print("[GateState] DB rows count:", len(rows))
    except Exception as db_err:
        print("[GateState] DB error:", db_err)
        gate_state = {
            "vehicleFound": vehicle_count > 0,
            "plate": gate_state.get("plate"),
            "driver": gate_state.get("driver"),
            "vehicle": gate_state.get("vehicle"),
            "lastUpdate": now_ms,
        }
        try:
            pusher_client.trigger("gate-channel", "gate-update", gate_state)
        except Exception as e:
            print("[Pusher] Error (B4 DB error):", e)
        print("[GateState] B4 DB error. gate_state:", gate_state)
        return gate_state

    if not rows:
        # No registered vehicle found for these plates
        gate_state = {
            "vehicleFound": vehicle_count > 0,
            "plate": None,
            "driver": None,
            "vehicle": None,
            "lastUpdate": now_ms,
        }
        try:
            pusher_client.trigger("gate-channel", "gate-update", gate_state)
        except Exception as e:
            print("[Pusher] Error (B4 no rows):", e)
        print("[GateState] B4 no rows. gate_state:", gate_state)
        return gate_state

    # B5: Registered vehicle found -> set LOCK and gate_state
    vehicle_details = rows[0]
    print("[GateState] vehicle_details row:", vehicle_details)

    raw_driver = vehicle_details.get("Driver")
    if isinstance(raw_driver, str):
        try:
            raw_driver = json.loads(raw_driver)
        except Exception:
            print("[GateState] Warning: Driver JSON could not be parsed")

    # Build new state
    new_state = {
        "vehicleFound": True,
        "plate": vehicle_details.get("PlateNumber"),
        "driver": raw_driver,
        "vehicle": {
            "brand": vehicle_details.get("Brand"),
            "model": vehicle_details.get("Model"),
            "type": vehicle_details.get("Type"),
        },
        "lastUpdate": now_ms,
    }

    gate_state = new_state
    current_registered_gate_state = new_state
    current_registered_ts = now_ms

    try:
        pusher_client.trigger("gate-channel", "gate-update", gate_state)
    except Exception as e:
        print("[Pusher] Error (B5 final trigger):", e)

    print("[GateState] B5 registered & LOCK set. gate_state:", gate_state)
    return gate_state


# =========================================================
# ROUTES
# =========================================================

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "yolo_model": YOLO_MODEL_PATH,
        "database_url": DATABASE_URL is not None,
    }


@app.get("/alpr/models")
async def alpr_models():
    return {"detector_models": DETECTOR_MODELS, "ocr_models": OCR_MODELS}


# ---------------------------------------------------------
# /stream-frame (mobile → server frame upload)
# Node streamFrameHandler equivalent
# ---------------------------------------------------------

@app.post("/stream-frame")
async def stream_frame(
    frame: UploadFile = File(...),
    stream_id: Optional[str] = None,
):
    """
    Receives a single image frame from mobile, stores it in memory per stream_id,
    and notifies listeners via Pusher with metadata only.

    Node JS equivalent:

      const { buffer, mimetype } = req.file;
      const streamId = req.body.stream_id || "mobile-1";
      const ct = mimetype || "image/jpeg";
      const ts = Date.now();
      latestFrames.set(streamId, { buffer, mimetype: ct, ts });
      await pusher.trigger("video-channel", "frame", { stream_id: streamId, ts });
    """
    if not frame:
        raise HTTPException(status_code=400, detail="frame is required")

    data = await frame.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty frame")

    sid = stream_id or "mobile-1"
    ct = frame.content_type or "image/jpeg"
    ts = int(time.time() * 1000)

    # 1) Store latest frame in memory
    latest_frames[sid] = {
        "buffer": data,
        "mimetype": ct,
        "ts": ts,
    }

    # 2) Notify via Pusher (metadata only)
    try:
        pusher_client.trigger("video-channel", "frame", {
            "stream_id": sid,
            "ts": ts,
        })
    except Exception as e:
        print("[/stream-frame] Pusher error:", e)

    return JSONResponse({"success": True})


# ---------------------------------------------------------
# /latest-frame (web → server)
# Node getLatestFrameHandler equivalent
# ---------------------------------------------------------

@app.get("/latest-frame")
async def get_latest_frame(stream_id: str = Query("mobile-1")):
    """
    Returns the raw bytes of the latest frame for given stream_id.

    Node JS equivalent:

      const streamId = (req.query.stream_id || "mobile-1").toString();
      const frame = latestFrames.get(streamId);
      if (!frame) 404 "No frame yet";
      res.setHeader("Content-Type", frame.mimetype || "image/jpeg");
      res.setHeader("Cache-Control", "no-store");
      res.send(frame.buffer);
    """
    frame = latest_frames.get(stream_id)
    if not frame:
        raise HTTPException(status_code=404, detail="No frame yet")

    mimetype = frame.get("mimetype") or "image/jpeg"
    buffer = frame.get("buffer") or b""

    headers = {
        "Cache-Control": "no-store",
        # Add if dashboard is on a different origin:
        # "Access-Control-Allow-Origin": "*",
    }

    return Response(content=buffer, media_type=mimetype, headers=headers)


# ---------------------------------------------------------
# /detect (unchanged; ALPR + YOLO)
# ---------------------------------------------------------

@app.post("/detect")
async def detect_all(
    file: UploadFile = File(...),
    detector_model: Optional[DetectorName] = None,
    ocr_model: Optional[OcrName] = None,
    # Optional stream_id to mirror Node signature (not used in logic yet)
    stream_id: Optional[str] = None,
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    detector_name = detector_model or DETECTOR_MODELS[0]
    ocr_name = ocr_model or OCR_MODELS[0]

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    img_np = load_image_to_numpy(data)
    h, w = img_np.shape[:2]

    t0 = time.time()
    vehicles = run_yolo_vehicles(img_np)
    alpr_result = run_alpr(img_np, detector_name, ocr_name)
    plates = alpr_result["plates"]

    # apply gateState logic and push update BEFORE responding
    gate = await update_gate_state_and_push(vehicles, plates)

    t1 = time.time()

    return JSONResponse(
        {
            "success": True,
            "image_width": w,
            "image_height": h,
            "vehicles": [v.dict() for v in vehicles],
            "plates": plates,
            "alpr_model": alpr_result["model"],
            "yolo_model": YOLO_MODEL_PATH,
            "total_time_ms": (t1 - t0) * 1000.0,
            "gate_state": gate,
            "stream_id": stream_id,
        }
    )
