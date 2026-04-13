# main.py
import io
import os
import time
import json
import tempfile
import traceback
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


# =========================================================
# SERIAL CONFIG (PICO CONTROL)
# =========================================================
SERIAL_ENABLED = os.getenv("SERIAL_ENABLED", "false").lower() == "true"
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM5")
SERIAL_BAUDRATE = int(os.getenv("SERIAL_BAUDRATE", "115200"))
SERIAL_TIMEOUT = float(os.getenv("SERIAL_TIMEOUT", "1"))
SERIAL_OPEN_DELAY = float(os.getenv("SERIAL_OPEN_DELAY", "2.0"))


# =========================================================
# FASTALPR CONFIG (PLATES)
# =========================================================
DETECTOR_MODELS: List[PlateDetectorModel] = list(get_args(PlateDetectorModel))
OCR_MODELS: List[OcrModel] = list(get_args(OcrModel))

# Prefer newer v2 OCR models first
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
print(f"[INIT] Loading YOLO model on device: {DEVICE}")
yolo_model = YOLO(YOLO_MODEL_PATH)
yolo_model.to(DEVICE)


# =========================================================
# GATE STATE (GLOBAL)
# =========================================================
gate_state: Dict[str, Any] = {
    "vehicleFound": False,
    "plate": None,
    "driver": None,
    "vehicle": None,
    "lastUpdate": None,
}

current_registered_gate_state: Optional[Dict[str, Any]] = None
current_registered_ts: int = 0


# =========================================================
# LATEST FRAMES
# =========================================================
latest_frames: Dict[str, Dict[str, Any]] = {}


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
        return

    try:
        app.state.db_pool = await asyncpg.create_pool(DATABASE_URL)
        print("[INIT] asyncpg pool created")
    except Exception as e:
        print("[DB] Failed to create pool:", e)
        app.state.db_pool = None


@app.on_event("shutdown")
async def shutdown():
    pool = getattr(app.state, "db_pool", None)
    if pool is not None:
        await pool.close()
        print("[DB] Pool closed")


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


# =========================================================
# PUSHER CLIENT
# =========================================================
PUSHER_APP_ID = os.getenv("PUSHER_APP_ID")
PUSHER_KEY = os.getenv("PUSHER_KEY", "")
PUSHER_SECRET = os.getenv("PUSHER_SECRET", "")
PUSHER_CLUSTER = os.getenv("PUSHER_CLUSTER", "ap1")
PUSHER_HOST = os.getenv("PUSHER_HOST")
PUSHER_PORT = os.getenv("PUSHER_PORT")
USE_LOCAL_PUSHER = os.getenv("USE_LOCAL_PUSHER", "false").lower() == "true"

# Hugging Face is not local
is_local_env = (
    os.getenv("SPACE_ID") is None
    and os.getenv("VERCEL") != "1"
    and os.getenv("NODE_ENV") != "production"
)

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
    print("[WARN] PUSHER_APP_ID is not set or invalid. Using DummyPusher.")

    class DummyPusher:
        def trigger(self, channel, event, data):
            print(f"[DummyPusher] trigger → channel={channel}, event={event}, data={data}")

    pusher_client: Any = DummyPusher()
else:
    client_kwargs: Dict[str, Any] = {
        "app_id": PUSHER_APP_ID,
        "key": PUSHER_KEY,
        "secret": PUSHER_SECRET,
        "cluster": PUSHER_CLUSTER,
        "ssl": True,
    }

    if is_local_env and USE_LOCAL_PUSHER and PUSHER_HOST:
        host = PUSHER_HOST
        port = int(PUSHER_PORT or "6001")
        print(f"[Pusher] Using LOCAL server at {host}:{port}")
        client_kwargs.pop("cluster", None)
        client_kwargs.update({"host": host, "port": port, "ssl": False})
    else:
        print(f"[Pusher] Using CLOUD (cluster={PUSHER_CLUSTER}, TLS=True)")

    pusher_client = Pusher(**client_kwargs)
    print("[INIT] Pusher client initialized")


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


LAST_VEHICLE_DETECTIONS: List[VehicleDetection] = []
LAST_VEHICLE_TS: float = 0.0


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

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name

        Image.fromarray(img_np).save(temp_path, format="JPEG")

        # Hugging Face-safe path input for FastALPR
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

    print("[GateState] cleaned_plates:", cleaned)
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
last_serial_command: Optional[str] = None


def is_gate_verified_open(gate: Dict[str, Any]) -> bool:
    return bool(
        gate.get("vehicleFound") is True
        and gate.get("plate")
    )


def get_pico_command_from_gate(gate: Dict[str, Any]) -> str:
    if is_gate_verified_open(gate):
        return "green"
    return "red"


def send_serial_command_to_pico(command: str) -> None:
    global last_serial_command

    if not SERIAL_ENABLED:
        print(f"[SERIAL] Skipped (disabled): {command}")
        return

    if serial is None:
        print("[SERIAL] pyserial is not installed.")
        return

    command = (command or "").strip().lower()
    if not command:
        return

    try:
        with serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT) as ser:
            time.sleep(SERIAL_OPEN_DELAY)
            ser.write((command + "\n").encode("utf-8"))
            ser.flush()
            last_serial_command = command
            print(f"[SERIAL] Sent to Pico: {command}")
    except Exception as e:
        print(f"[SERIAL] Failed to send '{command}' to Pico: {e}")


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
        print('[Logs] Warning: get_last_log_time_ms_for_plate failed:', e)
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
# GATE STATE LOGIC
# =========================================================
async def update_gate_state_and_push(
    vehicles: List[VehicleDetection],
    plates: List[dict],
    camera_source: str,
) -> Dict[str, Any]:
    global gate_state, current_registered_gate_state, current_registered_ts

    now_ms = int(time.time() * 1000)
    vehicle_count = len(vehicles)

    if current_registered_gate_state is not None:
        lock_age = now_ms - current_registered_ts

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
            return gate_state

        if vehicle_count == 0:
            last_update = gate_state.get("lastUpdate") or current_registered_ts
            if now_ms - last_update <= CLEAR_AFTER_MS:
                try:
                    pusher_client.trigger("gate-channel", "gate-update", gate_state)
                except Exception as e:
                    print("[Pusher] Error (lock, smooth clear window):", e)
                return gate_state

        current_registered_gate_state = None
        current_registered_ts = 0

    if (not vehicles) and (not plates):
        last_update = gate_state.get("lastUpdate")
        if gate_state.get("vehicleFound") and last_update and (now_ms - last_update <= CLEAR_AFTER_MS):
            try:
                pusher_client.trigger("gate-channel", "gate-update", gate_state)
            except Exception as e:
                print("[Pusher] Error (B1 smooth no-vehicle window):", e)
            return gate_state

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
        return gate_state

    if vehicles and not plates:
        gate_state = {
            "vehicleFound": True,
            "plate": gate_state.get("plate"),
            "driver": gate_state.get("driver"),
            "vehicle": gate_state.get("vehicle"),
            "lastUpdate": now_ms,
        }
        try:
            pusher_client.trigger("gate-channel", "gate-update", gate_state)
        except Exception as e:
            print("[Pusher] Error (B2 vehicles only):", e)
        return gate_state

    cleaned_plates = normalize_plate_text(plates)
    best_plate_text, best_plate_conf = pick_best_plate_text_and_conf(plates)
    ai_conf_bigint = to_ai_confidence_bigint(best_plate_conf)
    image_preview = build_image_preview_url(camera_source)

    if not cleaned_plates:
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
        return gate_state

    async def should_insert_log(plate_number: str) -> bool:
        try:
            last_ts_ms = await get_last_log_time_ms_for_plate(plate_number)
            if last_ts_ms is not None and (now_ms - last_ts_ms) <= LOG_WINDOW_MS:
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
                        camera_source=camera_source,
                        image_preview=image_preview,
                        ai_confidence=ai_conf_bigint,
                    )
                    print(f"[Logs] NOT REGISTERED log inserted for {best_plate_text}")
                else:
                    print(f"[Logs] NOT REGISTERED log skipped (<5min) for {best_plate_text}")
            except Exception as e:
                print("[Logs] Failed to write NOT REGISTERED Logs row:", e)

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
                camera_source=camera_source,
                image_preview=image_preview,
                ai_confidence=ai_conf_bigint,
            )
            print(f"[Logs] REGISTERED log inserted for {registered_plate}")
        else:
            print(f"[Logs] REGISTERED log skipped (<5min) for {registered_plate}")
    except Exception as e:
        print("[Logs] Failed to write REGISTERED Logs row:", e)

    new_state = {
        "vehicleFound": True,
        "plate": registered_plate,
        "driver": raw_driver,
        "vehicle": vehicle_json,
        "lastUpdate": now_ms,
    }

    gate_state = new_state
    current_registered_gate_state = new_state
    current_registered_ts = now_ms

    try:
        pusher_client.trigger("gate-channel", "gate-update", gate_state)
    except Exception as e:
        print("[Pusher] Error (registered trigger):", e)

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
        "public_base_url": PUBLIC_BASE_URL or None,
        "serial_enabled": SERIAL_ENABLED,
        "serial_port": SERIAL_PORT if SERIAL_ENABLED else None,
    }


@app.get("/alpr/models")
async def alpr_models():
    return {
        "detector_models": DETECTOR_MODELS,
        "ocr_models": OCR_MODELS,
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

    sid = stream_id or "mobile-1"
    ct = frame.content_type or "image/jpeg"
    ts = int(time.time() * 1000)

    latest_frames[sid] = {
        "buffer": data,
        "mimetype": ct,
        "ts": ts,
    }

    try:
        pusher_client.trigger("video-channel", "frame", {"stream_id": sid, "ts": ts})
    except Exception as e:
        print("[/stream-frame] Pusher error:", e)

    return JSONResponse({"success": True})


@app.get("/latest-frame")
async def get_latest_frame(stream_id: str = Query("mobile-1")):
    frame = latest_frames.get(stream_id)
    if not frame:
        raise HTTPException(status_code=404, detail="No frame yet")

    mimetype = frame.get("mimetype") or "image/jpeg"
    buffer = frame.get("buffer") or b""

    headers = {"Cache-Control": "no-store"}
    return Response(content=buffer, media_type=mimetype, headers=headers)


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

    sid = stream_id or "mobile-1"

    t0 = time.time()

    vehicles = run_yolo_vehicles(img_np)
    alpr_result = run_alpr(img_np, detector_name, ocr_name)
    plates = alpr_result["plates"]

    gate = await update_gate_state_and_push(vehicles, plates, camera_source=sid)

    pico_command = get_pico_command_from_gate(gate)
    send_serial_command_to_pico(pico_command)

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
            "pico_command": pico_command,
            "stream_id": sid,
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)