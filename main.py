import io
import os
import time
import json
import socket
import tempfile
import traceback
import threading
from functools import lru_cache
from typing import List, Optional, Literal, get_args, Any, Dict, Tuple

from dotenv import load_dotenv

# ---------------------------------------------------------
# Load .env (current dir + parent, to catch root .env)
# ---------------------------------------------------------
load_dotenv()
parent_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(parent_env_path):
    load_dotenv(parent_env_path, override=False)

import numpy as np
import asyncpg
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, Form
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
# OPTIONAL SERIAL (PC -> Pico)
# =========================================================
try:
    import serial  # pip install pyserial
except Exception:
    serial = None


# =========================================================
# CONFIG
# =========================================================
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8x.pt")

# COCO vehicle class IDs
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

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.5"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.5"))
HOLD_SECONDS = float(os.getenv("HOLD_SECONDS", "0.7"))

# Gate smoothing / lock behavior
PASS_LOCK_MS = int(os.getenv("PASS_LOCK_MS", "10000"))
CLEAR_AFTER_MS = int(os.getenv("CLEAR_AFTER_MS", "5000"))

# Logging window rule (5 minutes)
LOG_WINDOW_MS = int(os.getenv("LOG_WINDOW_MS", str(5 * 60 * 1000)))

# Used for ImagePreview in Logs
PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")

# Stream registry timing
STREAM_ONLINE_WINDOW_MS = int(os.getenv("STREAM_ONLINE_WINDOW_MS", "5000"))
STREAM_STALE_REMOVE_MS = int(os.getenv("STREAM_STALE_REMOVE_MS", "60000"))


# =========================================================
# SERIAL CONFIG (PICO CONTROL)
# =========================================================
SERIAL_ENABLED = os.getenv("SERIAL_ENABLED", "false").lower() == "true"
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM5")
SERIAL_BAUDRATE = int(os.getenv("SERIAL_BAUDRATE", "115200"))
SERIAL_TIMEOUT = float(os.getenv("SERIAL_TIMEOUT", "1"))
SERIAL_OPEN_DELAY = float(os.getenv("SERIAL_OPEN_DELAY", "2.0"))

last_serial_command: Optional[str] = None
_serial_lock = threading.Lock()
_serial_conn = None


# =========================================================
# FASTALPR CONFIG (PLATES)
# =========================================================
DETECTOR_MODELS: List[PlateDetectorModel] = list(get_args(PlateDetectorModel))
OCR_MODELS: List[OcrModel] = list(get_args(OcrModel))

if "cct-xs-v2-global-model" in OCR_MODELS:
    OCR_MODELS.remove("cct-xs-v2-global-model")
    OCR_MODELS.insert(0, "cct-xs-v2-global-model")
elif "cct-s-v2-global-model" in OCR_MODELS:
    OCR_MODELS.remove("cct-s-v2-global-model")
    OCR_MODELS.insert(0, "cct-s-v2-global-model")

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
    title="Smart Gate Keeper - YOLOv8 + FastALPR",
    version="1.2.0",
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
print(f"[INIT] Loading YOLO model on device: {DEVICE}")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(DEVICE)


# =========================================================
# PER-STREAM GATE STATE
# =========================================================
def create_empty_gate_state(stream_id: str) -> Dict[str, Any]:
    return {
        "stream_id": stream_id,
        "vehicleFound": False,
        "plate": None,
        "driver": None,
        "vehicle": None,
        "lastUpdate": None,
    }


gate_states_by_stream: Dict[str, Dict[str, Any]] = {}
registered_gate_state_by_stream: Dict[str, Optional[Dict[str, Any]]] = {}
registered_gate_ts_by_stream: Dict[str, int] = {}

# Per-stream YOLO hold memory
last_vehicle_detections_by_stream: Dict[str, List["VehicleDetection"]] = {}
last_vehicle_ts_by_stream: Dict[str, float] = {}


def get_gate_state_for_stream(stream_id: str) -> Dict[str, Any]:
    if stream_id not in gate_states_by_stream:
        gate_states_by_stream[stream_id] = create_empty_gate_state(stream_id)
    return gate_states_by_stream[stream_id]


def set_gate_state_for_stream(stream_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {
        "stream_id": stream_id,
        "vehicleFound": bool(state.get("vehicleFound")),
        "plate": state.get("plate"),
        "driver": state.get("driver"),
        "vehicle": state.get("vehicle"),
        "lastUpdate": state.get("lastUpdate"),
    }
    gate_states_by_stream[stream_id] = normalized
    return normalized


def clear_registered_lock_for_stream(stream_id: str) -> None:
    registered_gate_state_by_stream[stream_id] = None
    registered_gate_ts_by_stream[stream_id] = 0


# =========================================================
# LATEST FRAMES + ACTIVE STREAMS
# =========================================================
latest_frames: Dict[str, Dict[str, Any]] = {}
active_streams: Dict[str, Dict[str, Any]] = {}


def now_ms() -> int:
    return int(time.time() * 1000)


def touch_stream(stream_id: str, ts: Optional[int] = None) -> Dict[str, Any]:
    sid = (stream_id or "").strip() or "mobile-1"
    ts_ms = ts if ts is not None else now_ms()

    existing = active_streams.get(sid) or {}
    record = {
        "stream_id": sid,
        "last_seen": ts_ms,
        "first_seen": existing.get("first_seen", ts_ms),
    }
    active_streams[sid] = record
    return record


def cleanup_streams(current_ts: Optional[int] = None) -> None:
    ts_ms = current_ts if current_ts is not None else now_ms()

    stale_stream_ids = [
        sid
        for sid, info in active_streams.items()
        if (ts_ms - int(info.get("last_seen", 0))) > STREAM_STALE_REMOVE_MS
    ]
    for sid in stale_stream_ids:
        active_streams.pop(sid, None)

    stale_frame_ids = [
        sid
        for sid, info in latest_frames.items()
        if (ts_ms - int(info.get("ts", 0))) > STREAM_STALE_REMOVE_MS
    ]
    for sid in stale_frame_ids:
        latest_frames.pop(sid, None)

    stale_state_ids = [
        sid
        for sid, state in gate_states_by_stream.items()
        if state.get("lastUpdate") and (ts_ms - int(state["lastUpdate"])) > STREAM_STALE_REMOVE_MS
    ]
    for sid in stale_state_ids:
        gate_states_by_stream.pop(sid, None)
        registered_gate_state_by_stream.pop(sid, None)
        registered_gate_ts_by_stream.pop(sid, None)
        last_vehicle_detections_by_stream.pop(sid, None)
        last_vehicle_ts_by_stream.pop(sid, None)


def build_stream_list(current_ts: Optional[int] = None) -> List[Dict[str, Any]]:
    ts_ms = current_ts if current_ts is not None else now_ms()
    cleanup_streams(ts_ms)

    items: List[Dict[str, Any]] = []
    for sid, info in active_streams.items():
        last_seen = int(info.get("last_seen", 0))
        items.append(
            {
                "stream_id": sid,
                "last_seen": last_seen,
                "is_online": (ts_ms - last_seen) <= STREAM_ONLINE_WINDOW_MS,
                "has_frame": sid in latest_frames,
            }
        )

    items.sort(key=lambda x: x["last_seen"], reverse=True)
    return items


# =========================================================
# DB (PostgreSQL)
# =========================================================
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DB_USER = os.getenv("DB_USER", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "")

    if DB_USER and DB_NAME:
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        print("[DB] Constructed DATABASE_URL from DB_* env vars")
    else:
        print("[DB] Missing DB_USER or DB_NAME; cannot construct DATABASE_URL.")

if DATABASE_URL:
    print("[DB] Using DATABASE_URL")
else:
    print("[DB] DATABASE_URL is not set. DB queries will fail.")


@app.on_event("startup")
async def startup():
    if not DATABASE_URL:
        print("[WARN] DATABASE_URL is not set; DB queries will fail.")
        app.state.db_pool = None
    else:
        try:
            app.state.db_pool = await asyncpg.create_pool(DATABASE_URL)
            print("[INIT] asyncpg pool created")
        except Exception as e:
            print("[DB] Failed to create pool:", e)
            app.state.db_pool = None

    # Pre-open serial once on startup if enabled
    if SERIAL_ENABLED:
        try:
            ser = get_serial_connection()
            if ser is not None:
                print(f"[SERIAL] Ready on {SERIAL_PORT}")
        except Exception as e:
            print("[SERIAL] Startup open failed:", e)


@app.on_event("shutdown")
async def shutdown():
    pool = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()
        print("[DB] Pool closed")

    close_serial_connection()
    print("[SERIAL] Port closed")


async def db_query(sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    params = params or []
    pool = getattr(app.state, "db_pool", None)
    if pool is None:
        raise RuntimeError("DB pool is not initialized.")
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
        return [dict(r) for r in rows]


async def db_execute(sql: str, params: Optional[List[Any]] = None) -> None:
    params = params or []
    pool = getattr(app.state, "db_pool", None)
    if pool is None:
        raise RuntimeError("DB pool is not initialized.")
    async with pool.acquire() as conn:
        await conn.execute(sql, *params)


async def get_light_by_stream_id(stream_id: str) -> Optional[Dict[str, Any]]:
    if not stream_id:
        return None

    sql = """
        SELECT
            "Name",
            "SecretKey",
            "CameraStreamId"
        FROM dbo."Lights"
        WHERE "Active" = true
          AND "CameraStreamId" = $1
        ORDER BY "Name" ASC
    """
    rows = await db_query(sql, [stream_id])
    if not rows:
        return None

    if len(rows) > 1:
        print(f"[LIGHT WARNING] Multiple active Lights rows found for stream_id='{stream_id}': {rows}")

    return rows[0]


# =========================================================
# PUSHER CLIENT
# =========================================================
PUSHER_APP_ID = (os.getenv("PUSHER_APP_ID") or "").strip()
PUSHER_KEY = (os.getenv("PUSHER_KEY") or "").strip()
PUSHER_SECRET = (os.getenv("PUSHER_SECRET") or "").strip()
PUSHER_CLUSTER = (os.getenv("PUSHER_CLUSTER") or "ap1").strip()
PUSHER_HOST = (os.getenv("PUSHER_HOST") or "").strip()
PUSHER_PORT = int((os.getenv("PUSHER_PORT") or "6001").strip())

USE_LOCAL_PUSHER = (os.getenv("USE_LOCAL_PUSHER") or "false").strip().lower() == "true"
PUSHER_RETRY_SECONDS = int((os.getenv("PUSHER_RETRY_SECONDS") or "10").strip())
PUSHER_CONNECT_TIMEOUT = float((os.getenv("PUSHER_CONNECT_TIMEOUT") or "0.8").strip())

IS_HUGGING_FACE = bool(os.getenv("SPACE_ID"))
IS_VERCEL = os.getenv("VERCEL") == "1"
IS_PRODUCTION = (os.getenv("NODE_ENV") or "").strip().lower() == "production"

USE_LOCAL_PUSHER_EFFECTIVE = USE_LOCAL_PUSHER and not IS_HUGGING_FACE and not IS_VERCEL and not IS_PRODUCTION


def can_connect_socket(host: str, port: int, timeout: float = 0.8) -> bool:
    if not host or not port:
        return False

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
        return True
    except Exception:
        return False
    finally:
        try:
            sock.close()
        except Exception:
            pass


class DummyPusher:
    def __init__(self, reason: str = "disabled"):
        self.reason = reason
        self.last_error = None
        self.disabled_until = 0.0

    def trigger(self, channel, event, data):
        return None


class SafePusher:
    def __init__(self, client, mode: str, retry_seconds: int = 10):
        self.client = client
        self.mode = mode
        self.retry_seconds = max(1, retry_seconds)
        self.disabled_until = 0.0
        self.last_error = None

    def trigger(self, channel, event, data):
        now = time.time()

        if now < self.disabled_until:
            return None

        try:
            return self.client.trigger(channel, event, data)
        except Exception as e:
            self.last_error = str(e)
            self.disabled_until = now + self.retry_seconds
            print(
                f"[Pusher] trigger failed in {self.mode} mode; "
                f"muted for {self.retry_seconds}s. Error: {e}"
            )
            return None


pusher_mode = "dummy"
pusher_reason = ""
pusher_client: Any = DummyPusher("uninitialized")

print(
    "[Pusher] env values:",
    "APP_ID=", repr(PUSHER_APP_ID),
    "KEY=", repr(PUSHER_KEY),
    "SECRET set=", bool(PUSHER_SECRET),
    "CLUSTER=", repr(PUSHER_CLUSTER),
    "HOST=", repr(PUSHER_HOST),
    "PORT=", repr(PUSHER_PORT),
    "USE_LOCAL_PUSHER=", USE_LOCAL_PUSHER,
    "USE_LOCAL_PUSHER_EFFECTIVE=", USE_LOCAL_PUSHER_EFFECTIVE,
    "IS_HUGGING_FACE=", IS_HUGGING_FACE,
    "IS_VERCEL=", IS_VERCEL,
    "IS_PRODUCTION=", IS_PRODUCTION,
)

if not PUSHER_APP_ID:
    pusher_mode = "dummy"
    pusher_reason = "missing PUSHER_APP_ID"
    print(f"[Pusher] Dummy mode: {pusher_reason}")
    pusher_client = DummyPusher(pusher_reason)

else:
    if USE_LOCAL_PUSHER_EFFECTIVE and PUSHER_HOST:
        local_reachable = can_connect_socket(
            PUSHER_HOST,
            PUSHER_PORT,
            timeout=PUSHER_CONNECT_TIMEOUT,
        )

        if local_reachable:
            try:
                local_kwargs: Dict[str, Any] = {
                    "app_id": PUSHER_APP_ID,
                    "key": PUSHER_KEY,
                    "secret": PUSHER_SECRET,
                    "host": PUSHER_HOST,
                    "port": PUSHER_PORT,
                    "ssl": False,
                }
                raw_local_client = Pusher(**local_kwargs)
                pusher_client = SafePusher(
                    raw_local_client,
                    mode="local",
                    retry_seconds=PUSHER_RETRY_SECONDS,
                )
                pusher_mode = "local"
                pusher_reason = f"connected to local server at {PUSHER_HOST}:{PUSHER_PORT}"
                print(f"[Pusher] Using LOCAL server at {PUSHER_HOST}:{PUSHER_PORT}")
            except Exception as e:
                pusher_mode = "dummy"
                pusher_reason = f"local init failed: {e}"
                pusher_client = DummyPusher(pusher_reason)
                print(f"[Pusher] Dummy mode: {pusher_reason}")
        else:
            pusher_mode = "dummy"
            pusher_reason = f"local server not reachable at {PUSHER_HOST}:{PUSHER_PORT}"
            pusher_client = DummyPusher(pusher_reason)
            print(f"[Pusher] Dummy mode: {pusher_reason}")

    else:
        try:
            cloud_kwargs: Dict[str, Any] = {
                "app_id": PUSHER_APP_ID,
                "key": PUSHER_KEY,
                "secret": PUSHER_SECRET,
                "cluster": PUSHER_CLUSTER,
                "ssl": True,
            }
            raw_cloud_client = Pusher(**cloud_kwargs)
            pusher_client = SafePusher(
                raw_cloud_client,
                mode="cloud",
                retry_seconds=PUSHER_RETRY_SECONDS,
            )
            pusher_mode = "cloud"
            pusher_reason = f"using cloud cluster={PUSHER_CLUSTER}"
            print(f"[Pusher] Using CLOUD (cluster={PUSHER_CLUSTER}, TLS=True)")
        except Exception as e:
            pusher_mode = "dummy"
            pusher_reason = f"cloud init failed: {e}"
            pusher_client = DummyPusher(pusher_reason)
            print(f"[Pusher] Dummy mode: {pusher_reason}")

print(f"[INIT] Realtime mode = {pusher_mode} ({pusher_reason})")


# =========================================================
# Pydantic Models
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


# =========================================================
# HELPERS (Image / YOLO / ALPR)
# =========================================================
def load_image_to_numpy(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    return np.array(img)


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

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name

        Image.fromarray(img_np).save(temp_path, format="JPEG")
        results = alpr.predict(temp_path)

    except Exception as e:
        print("[ALPR] Full traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ALPR error: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    plates = [serialize_alpr_result(r) for r in results] if results else []
    return {
        "model": {
            "detector_model": detector_name,
            "ocr_model": ocr_name,
        },
        "plates": plates,
    }


def normalize_plate_text(plates: List[dict]) -> List[str]:
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

    return cleaned


def pick_best_plate_text_and_conf(plates: List[dict]) -> Tuple[Optional[str], float]:
    best_text = None
    best_conf = 0.0

    for p in plates or []:
        ocr = p.get("ocr") or {}
        text = ocr.get("text")
        conf = ocr.get("confidence")
        if not text:
            continue

        try:
            conf_val = float(conf) if conf is not None else 0.0
        except Exception:
            conf_val = 0.0

        if conf_val > best_conf:
            best_conf = conf_val
            best_text = text

    return best_text, best_conf


def to_ai_confidence_bigint(best_conf_0_1: float) -> int:
    if best_conf_0_1 < 0:
        best_conf_0_1 = 0.0
    if best_conf_0_1 > 1:
        best_conf_0_1 = 1.0
    return int(round(best_conf_0_1 * 100))


def build_image_preview_url(stream_id: str) -> str:
    suffix = f"/latest-frame?stream_id={stream_id}"
    return (PUBLIC_BASE_URL + suffix) if PUBLIC_BASE_URL else suffix


# =========================================================
# SERIAL HELPERS
# =========================================================
def close_serial_connection() -> None:
    global _serial_conn
    with _serial_lock:
        try:
            if _serial_conn is not None and _serial_conn.is_open:
                _serial_conn.close()
        except Exception:
            pass
        _serial_conn = None


def get_serial_connection():
    global _serial_conn

    if not SERIAL_ENABLED:
        return None

    if serial is None:
        print("[SERIAL] pyserial is not installed.")
        return None

    try:
        if _serial_conn is not None and _serial_conn.is_open:
            return _serial_conn
    except Exception:
        _serial_conn = None

    try:
        print(f"[SERIAL] Opening port {SERIAL_PORT} ...")
        _serial_conn = serial.Serial(
            SERIAL_PORT,
            SERIAL_BAUDRATE,
            timeout=SERIAL_TIMEOUT,
            write_timeout=SERIAL_TIMEOUT,
        )

        # Give MicroPython/USB serial a moment only when opening
        time.sleep(SERIAL_OPEN_DELAY)

        try:
            _serial_conn.reset_input_buffer()
        except Exception:
            pass

        try:
            _serial_conn.reset_output_buffer()
        except Exception:
            pass

        print(f"[SERIAL] Port opened: {SERIAL_PORT}")
        return _serial_conn

    except Exception as e:
        print(f"[SERIAL] Failed to open port {SERIAL_PORT}: {e}")
        _serial_conn = None
        return None


def send_serial_command_to_pico(command_line: str) -> bool:
    global last_serial_command

    if not SERIAL_ENABLED:
        print(f"[SERIAL] Skipped (disabled): {command_line}")
        return False

    command_line = (command_line or "").strip()
    if not command_line:
        print("[SERIAL] Empty command line")
        return False

    with _serial_lock:
        ser = get_serial_connection()
        if ser is None:
            return False

        try:
            payload = (command_line + "\n").encode("utf-8")
            ser.write(payload)
            ser.flush()
            last_serial_command = command_line
            print(f"[SERIAL] Sent to Pico: {command_line}")
            return True

        except Exception as e:
            print(f"[SERIAL] Failed to send '{command_line}' to Pico: {e}")
            close_serial_connection()
            return False


# =========================================================
# LIGHT COMMAND HELPERS
# =========================================================
def is_gate_verified_open(gate: Dict[str, Any]) -> bool:
    return bool(
        gate.get("vehicleFound") is True
        and gate.get("plate")
    )


def get_light_command_from_gate(gate: Dict[str, Any]) -> str:
    if is_gate_verified_open(gate):
        return "green"
    return "red"


def build_light_serial_command(light: Dict[str, Any], command: str) -> Optional[str]:
    if not light:
        return None

    name = str(light.get("Name") or "").strip()
    secret = str(light.get("SecretKey") or "").strip()
    cmd = str(command or "").strip().lower()

    if not name or not secret or not cmd:
        return None

    return f"{name} {secret} {cmd}"


# =========================================================
# LOGGING HELPERS
# =========================================================
async def get_last_log_time_ms_for_plate(plate_text: str) -> Optional[int]:
    try:
        sql = """
            SELECT EXTRACT(EPOCH FROM "CreatedAt") * 1000 AS ts_ms
            FROM dbo."Logs"
            WHERE "PlateNumber" = $1
            ORDER BY "CreatedAt" DESC
            LIMIT 1
        """
        rows = await db_query(sql, [plate_text])
        if not rows:
            return None

        ts_ms = rows[0].get("ts_ms")
        return int(ts_ms) if ts_ms is not None else None
    except Exception as e:
        print("[Logs] Warning: get_last_log_time_ms_for_plate failed:", e)
        return None


async def insert_log_row(
    plate_number: str,
    driver_json: Optional[dict],
    vehicle_json: Optional[dict],
    role_type: Optional[str],
    verification: str,
    camera_source: str,
    image_preview: str,
    ai_confidence: int,
) -> None:
    sql = """
        INSERT INTO dbo."Logs"
            ("PlateNumber", "Driver", "Vehicle", "RoleType", "Verification", "CameraSource", "ImagePreview", "AIConfidence")
        VALUES
            ($1, $2::jsonb, $3::jsonb, $4, $5, $6, $7, $8)
    """
    await db_execute(sql, [
        plate_number,
        json.dumps(driver_json) if driver_json is not None else None,
        json.dumps(vehicle_json) if vehicle_json is not None else None,
        role_type,
        verification,
        camera_source,
        image_preview,
        int(ai_confidence),
    ])


# =========================================================
# GATE STATE LOGIC (PER STREAM)
# =========================================================
def push_gate_update(stream_id: str, gate: Dict[str, Any]) -> None:
    payload = {
        **gate,
        "stream_id": stream_id,
    }
    try:
        print(f"[GateState] pushing gate-update: {json.dumps(payload, default=str)}")
        pusher_client.trigger("gate-channel", "gate-update", payload)
    except Exception as e:
        print("[Pusher] Error sending gate-update:", e)


async def update_gate_state_and_push(
    vehicles: List[VehicleDetection],
    plates: List[dict],
    camera_source: str,
) -> Dict[str, Any]:
    stream_id = (camera_source or "").strip() or "mobile-1"
    now_ts = now_ms()
    vehicle_count = len(vehicles)

    gate_state = get_gate_state_for_stream(stream_id)
    current_registered_gate_state = registered_gate_state_by_stream.get(stream_id)
    current_registered_ts = registered_gate_ts_by_stream.get(stream_id, 0)

    print(
        f"[GateState] update start | stream={stream_id} "
        f"| vehicles={vehicle_count} | plates={len(plates)}"
    )

    if current_registered_gate_state is not None:
        lock_age = now_ts - current_registered_ts

        if vehicle_count > 0 and lock_age <= PASS_LOCK_MS:
            gate_state = set_gate_state_for_stream(
                stream_id,
                {
                    **current_registered_gate_state,
                    "stream_id": stream_id,
                    "vehicleFound": True,
                    "lastUpdate": now_ts,
                },
            )
            push_gate_update(stream_id, gate_state)
            return gate_state

        if vehicle_count == 0:
            last_update = gate_state.get("lastUpdate") or current_registered_ts
            if last_update and (now_ts - int(last_update) <= CLEAR_AFTER_MS):
                push_gate_update(stream_id, gate_state)
                return gate_state

        clear_registered_lock_for_stream(stream_id)

    if (not vehicles) and (not plates):
        last_update = gate_state.get("lastUpdate")
        if gate_state.get("vehicleFound") and last_update and (now_ts - int(last_update) <= CLEAR_AFTER_MS):
            push_gate_update(stream_id, gate_state)
            return gate_state

        gate_state = set_gate_state_for_stream(
            stream_id,
            {
                "stream_id": stream_id,
                "vehicleFound": False,
                "plate": None,
                "driver": None,
                "vehicle": None,
                "lastUpdate": None,
            },
        )
        push_gate_update(stream_id, gate_state)
        return gate_state

    if vehicles and not plates:
        existing_plate = None
        existing_driver = None
        existing_vehicle = None

        if current_registered_gate_state is not None:
            lock_age = now_ts - current_registered_ts
            if lock_age <= PASS_LOCK_MS:
                existing_plate = current_registered_gate_state.get("plate")
                existing_driver = current_registered_gate_state.get("driver")
                existing_vehicle = current_registered_gate_state.get("vehicle")

        gate_state = set_gate_state_for_stream(
            stream_id,
            {
                "stream_id": stream_id,
                "vehicleFound": True,
                "plate": existing_plate,
                "driver": existing_driver,
                "vehicle": existing_vehicle,
                "lastUpdate": now_ts,
            },
        )
        push_gate_update(stream_id, gate_state)
        return gate_state

    cleaned_plates = normalize_plate_text(plates)
    print(f"[GateState] stream={stream_id} cleaned_plates={cleaned_plates}")

    best_plate_text, best_plate_conf = pick_best_plate_text_and_conf(plates)
    ai_conf_bigint = to_ai_confidence_bigint(best_plate_conf)
    image_preview = build_image_preview_url(stream_id)

    if not cleaned_plates:
        gate_state = set_gate_state_for_stream(
            stream_id,
            {
                "stream_id": stream_id,
                "vehicleFound": vehicle_count > 0,
                "plate": None,
                "driver": None,
                "vehicle": None,
                "lastUpdate": now_ts,
            },
        )
        push_gate_update(stream_id, gate_state)
        return gate_state

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
        rows = await db_query(sql, [cleaned_plates])
    except Exception as db_err:
        print("[GateState] DB error:", db_err)
        gate_state = set_gate_state_for_stream(
            stream_id,
            {
                "stream_id": stream_id,
                "vehicleFound": vehicle_count > 0,
                "plate": None,
                "driver": None,
                "vehicle": None,
                "lastUpdate": now_ts,
            },
        )
        push_gate_update(stream_id, gate_state)
        return gate_state

    async def should_insert_log(plate_number: str) -> bool:
        try:
            last_ts_ms = await get_last_log_time_ms_for_plate(plate_number)
            if last_ts_ms is not None and (now_ts - last_ts_ms) <= LOG_WINDOW_MS:
                return False
            return True
        except Exception as e:
            print("[Logs] 5-min check failed; allowing insert:", e)
            return True

    if not rows:
        if best_plate_text:
            try:
                if await should_insert_log(best_plate_text):
                    await insert_log_row(
                        plate_number=best_plate_text,
                        driver_json=None,
                        vehicle_json=None,
                        role_type="Visitors",
                        verification="NOT REGISTERED",
                        camera_source=stream_id,
                        image_preview=image_preview,
                        ai_confidence=ai_conf_bigint,
                    )
                    print(f"[Logs] NOT REGISTERED log inserted for {best_plate_text}")
                else:
                    print(f"[Logs] NOT REGISTERED log skipped (<5min) for {best_plate_text}")
            except Exception as e:
                print("[Logs] Failed to write NOT REGISTERED Logs row:", e)

        gate_state = set_gate_state_for_stream(
            stream_id,
            {
                "stream_id": stream_id,
                "vehicleFound": vehicle_count > 0,
                "plate": None,
                "driver": None,
                "vehicle": None,
                "lastUpdate": now_ts,
            },
        )
        push_gate_update(stream_id, gate_state)
        return gate_state

    vehicle_details = rows[0]

    raw_driver = vehicle_details.get("Driver")
    if isinstance(raw_driver, str):
        try:
            raw_driver = json.loads(raw_driver)
        except Exception:
            raw_driver = None

    registered_plate = vehicle_details.get("PlateNumber") or best_plate_text or ""
    vehicle_json = {
        "brand": vehicle_details.get("Brand"),
        "model": vehicle_details.get("Model"),
        "type": vehicle_details.get("Type"),
    }

    try:
        if registered_plate and await should_insert_log(registered_plate):
            await insert_log_row(
                plate_number=registered_plate,
                driver_json=raw_driver,
                vehicle_json=vehicle_json,
                role_type=(raw_driver or {}).get("RoleType") if isinstance(raw_driver, dict) else None,
                verification="REGISTERED",
                camera_source=stream_id,
                image_preview=image_preview,
                ai_confidence=ai_conf_bigint,
            )
            print(f"[Logs] REGISTERED log inserted for {registered_plate}")
        else:
            print(f"[Logs] REGISTERED log skipped (<5min) for {registered_plate}")
    except Exception as e:
        print("[Logs] Failed to write REGISTERED Logs row:", e)

    new_state = set_gate_state_for_stream(
        stream_id,
        {
            "stream_id": stream_id,
            "vehicleFound": True,
            "plate": registered_plate,
            "driver": raw_driver,
            "vehicle": vehicle_json,
            "lastUpdate": now_ts,
        },
    )

    registered_gate_state_by_stream[stream_id] = dict(new_state)
    registered_gate_ts_by_stream[stream_id] = now_ts

    print(f"[GateState] stream={stream_id} registered match -> {json.dumps(new_state, default=str)}")
    push_gate_update(stream_id, new_state)

    return new_state


# =========================================================
# ROUTES
# =========================================================
@app.get("/health")
async def health():
    safe_last_error = getattr(pusher_client, "last_error", None)
    safe_disabled_until = getattr(pusher_client, "disabled_until", 0.0)

    serial_open = False
    try:
        serial_open = bool(_serial_conn is not None and _serial_conn.is_open)
    except Exception:
        serial_open = False

    return {
        "status": "ok",
        "device": DEVICE,
        "yolo_model": YOLO_MODEL_PATH,
        "database_url": DATABASE_URL is not None,
        "public_base_url": PUBLIC_BASE_URL or None,
        "serial_enabled": SERIAL_ENABLED,
        "serial_port": SERIAL_PORT if SERIAL_ENABLED else None,
        "serial_open": serial_open,
        "last_serial_command": last_serial_command,
        "stream_online_window_ms": STREAM_ONLINE_WINDOW_MS,
        "stream_stale_remove_ms": STREAM_STALE_REMOVE_MS,
        "active_gate_states": list(gate_states_by_stream.keys()),
        "realtime": {
            "mode": pusher_mode,
            "reason": pusher_reason,
            "use_local_requested": USE_LOCAL_PUSHER,
            "use_local_effective": USE_LOCAL_PUSHER_EFFECTIVE,
            "host": PUSHER_HOST or None,
            "port": PUSHER_PORT,
            "cluster": PUSHER_CLUSTER,
            "last_error": safe_last_error,
            "muted_until_epoch": safe_disabled_until if safe_disabled_until else None,
        },
    }


@app.get("/alpr/models")
async def alpr_models():
    return {
        "detector_models": DETECTOR_MODELS,
        "ocr_models": OCR_MODELS,
    }


@app.get("/streams")
async def get_streams():
    ts = now_ms()
    return {
        "success": True,
        "streams": build_stream_list(ts),
    }


@app.post("/stream-frame")
async def stream_frame(
    frame: UploadFile = File(...),
    stream_id: Optional[str] = Form(None),
):
    if not frame:
        raise HTTPException(status_code=400, detail="frame is required")

    data = await frame.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty frame")

    sid = (stream_id or "").strip() or "mobile-1"
    ct = frame.content_type or "image/jpeg"
    ts = now_ms()

    latest_frames[sid] = {
        "buffer": data,
        "mimetype": ct,
        "ts": ts,
    }
    touch_stream(sid, ts)
    cleanup_streams(ts)

    try:
        pusher_client.trigger("video-channel", "frame", {"stream_id": sid, "ts": ts})
    except Exception as e:
        print("[/stream-frame] Pusher error:", e)

    return JSONResponse({"success": True, "stream_id": sid, "ts": ts})


@app.get("/latest-frame")
async def get_latest_frame(stream_id: str = Query("mobile-1")):
    sid = (stream_id or "").strip() or "mobile-1"
    frame = latest_frames.get(sid)
    if not frame:
        raise HTTPException(status_code=404, detail="No frame yet")

    mimetype = frame.get("mimetype") or "image/jpeg"
    buffer = frame.get("buffer") or b""

    headers = {"Cache-Control": "no-store"}
    return Response(content=buffer, media_type=mimetype, headers=headers)


@app.post("/test-pusher")
async def test_pusher():
    payload = {
        "stream_id": "test-stream",
        "vehicleFound": True,
        "plate": "TEST-123",
        "driver": {"fullName": "Test Driver"},
        "vehicle": {"type": "Sedan", "brand": "Toyota", "model": "Vios"},
        "lastUpdate": int(time.time() * 1000),
    }
    print("[/test-pusher] sending gate-update:", payload)
    pusher_client.trigger("gate-channel", "gate-update", payload)
    return {
        "success": True,
        "mode": pusher_mode,
        "reason": pusher_reason,
        "payload": payload,
    }


@app.post("/detect")
async def detect_all(
    file: UploadFile = File(...),
    detector_model: Optional[DetectorName] = Form(None),
    ocr_model: Optional[OcrName] = Form(None),
    stream_id: Optional[str] = Form(None),
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    detector_name = detector_model or DETECTOR_MODELS[0]
    ocr_name = ocr_model or OCR_MODELS[0]

    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file.")

    img_np = load_image_to_numpy(data)
    h, w = img_np.shape[:2]
    sid = (stream_id or "").strip() or "mobile-1"
    ts = now_ms()

    touch_stream(sid, ts)
    cleanup_streams(ts)

    t0 = time.time()

    vehicles = run_yolo_vehicles(img_np, sid)
    alpr_result = run_alpr(img_np, detector_name, ocr_name)
    plates = alpr_result["plates"]

    print(f"[/detect] stream_id={sid}")
    print(f"[/detect] vehicles_count={len(vehicles)}")
    print(f"[/detect] plates_count={len(plates)}")
    if vehicles:
        print(f"[/detect] top_vehicle={vehicles[0].class_name} conf={vehicles[0].confidence:.4f}")
    if plates:
        print(f"[/detect] top_plate={plates[0]}")

    gate = await update_gate_state_and_push(vehicles, plates, camera_source=sid)

    print(f"[/detect] gate_state={json.dumps(gate, default=str)}")

    light = await get_light_by_stream_id(sid)
    print(f"[/detect] requested_stream_id={sid}")
    print(f"[/detect] light_mapping={json.dumps(light, default=str) if light else None}")

    light_command = get_light_command_from_gate(gate)
    serial_command = build_light_serial_command(light, light_command)

    print(f"[/detect] light_command={light_command}")
    print(f"[/detect] serial_command={serial_command}")

    serial_sent = False
    if serial_command:
        serial_sent = send_serial_command_to_pico(serial_command)
    else:
        print(f"[/detect] No active light mapping found for stream_id='{sid}', skipping serial send.")

    t1 = time.time()

    return JSONResponse(
        {
            "success": True,
            "image_width": w,
            "image_height": h,
            "vehicles": [v.model_dump() for v in vehicles],
            "plates": plates,
            "alpr_model": alpr_result["model"],
            "yolo_model": YOLO_MODEL_PATH,
            "total_time_ms": (t1 - t0) * 1000.0,
            "gate_state": gate,
            "light": light,
            "light_command": light_command,
            "serial_command": serial_command,
            "serial_sent": serial_sent,
            "stream_id": sid,
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)