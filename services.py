# services.py

import asyncio
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

import asyncpg
import serial
from dotenv import load_dotenv

from ws_client import publish_event

load_dotenv()
parent_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(parent_env_path):
    load_dotenv(parent_env_path, override=False)


# ---------------------------------------------------------
# Environment
# ---------------------------------------------------------
IMAGE_SAVE_DIR = os.getenv("IMAGE_SAVE_DIR", "images")
STREAM_ONLINE_WINDOW_MS = int(os.getenv("STREAM_ONLINE_WINDOW_MS", "5000"))
STREAM_STALE_REMOVE_MS = int(os.getenv("STREAM_STALE_REMOVE_MS", "60000"))

SERIAL_ENABLED = str(os.getenv("SERIAL_ENABLED", "false")).lower() == "true"
SERIAL_PORT = os.getenv("SERIAL_PORT", "")
SERIAL_BAUDRATE = int(os.getenv("SERIAL_BAUDRATE", "115200"))
SERIAL_TIMEOUT = float(os.getenv("SERIAL_TIMEOUT", "1"))
SERIAL_OPEN_DELAY = float(os.getenv("SERIAL_OPEN_DELAY", "2.0"))

DB_USER = os.getenv("DB_USER", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_SSL = str(os.getenv("DB_SSL", "false")).lower() == "true"

LOG_DEDUP_SECONDS = int(os.getenv("LOG_DEDUP_SECONDS", "300"))
PLATE_CACHE_REFRESH_SECONDS = int(os.getenv("PLATE_CACHE_REFRESH_SECONDS", "10"))


# ---------------------------------------------------------
# Runtime state
# ---------------------------------------------------------
db_pool: Optional[asyncpg.Pool] = None
latest_frames: Dict[str, Dict[str, Any]] = {}
active_streams: Dict[str, Dict[str, Any]] = {}
active_gate_states: Dict[str, Dict[str, Any]] = {}

_plate_registry_lock = threading.Lock()
_plate_registry: Dict[str, Dict[str, Any]] = {}
_plate_cache_task: Optional[asyncio.Task] = None

_serial_lock = threading.Lock()
_serial_conn: Optional[serial.Serial] = None


# ---------------------------------------------------------
# Time / stream helpers
# ---------------------------------------------------------
def now_ms() -> int:
    return int(time.time() * 1000)


def touch_stream(stream_id: str, ts: Optional[int] = None) -> None:
    ts = ts or now_ms()
    current = active_streams.get(stream_id) or {}
    active_streams[stream_id] = {
        **current,
        "stream_id": stream_id,
        "last_seen": ts,
    }


def cleanup_streams(ts: Optional[int] = None) -> None:
    ts = ts or now_ms()
    cutoff = ts - STREAM_STALE_REMOVE_MS

    stale_keys = [sid for sid, info in active_streams.items() if int(info.get("last_seen", 0)) < cutoff]
    for sid in stale_keys:
        active_streams.pop(sid, None)
        latest_frames.pop(sid, None)
        active_gate_states.pop(sid, None)


def build_stream_list(ts: Optional[int] = None) -> list[dict]:
    ts = ts or now_ms()
    cleanup_streams(ts)

    out = []
    for sid, info in sorted(active_streams.items(), key=lambda x: x[0]):
        last_seen = int(info.get("last_seen", 0))
        out.append(
            {
                "stream_id": sid,
                "last_seen": last_seen,
                "is_online": (ts - last_seen) <= STREAM_ONLINE_WINDOW_MS,
            }
        )
    return out


# ---------------------------------------------------------
# Image saving
# ---------------------------------------------------------
def save_image_bytes(data: bytes, stream_id: str, ext: str = ".jpg") -> str:
    save_dir = Path(IMAGE_SAVE_DIR)
    save_dir.mkdir(parents=True, exist_ok=True)

    safe_stream = "".join(ch for ch in (stream_id or "stream") if ch.isalnum() or ch in ("-", "_")) or "stream"
    filename = f"{safe_stream}_{now_ms()}_{uuid.uuid4().hex[:8]}{ext}"
    file_path = save_dir / filename
    file_path.write_bytes(data)
    return filename


# ---------------------------------------------------------
# Serial helpers
# ---------------------------------------------------------
def _open_serial_if_needed() -> Optional[serial.Serial]:
    global _serial_conn

    if not SERIAL_ENABLED or not SERIAL_PORT:
        return None

    with _serial_lock:
        if _serial_conn and _serial_conn.is_open:
            return _serial_conn

        try:
            _serial_conn = serial.Serial(
                SERIAL_PORT,
                SERIAL_BAUDRATE,
                timeout=SERIAL_TIMEOUT,
                write_timeout=SERIAL_TIMEOUT,
            )
            time.sleep(SERIAL_OPEN_DELAY)
            print(f"[SERIAL] opened {SERIAL_PORT} @ {SERIAL_BAUDRATE}")
            return _serial_conn
        except Exception as e:
            print(f"[SERIAL] open failed: {e}")
            _serial_conn = None
            return None


def send_serial_command_to_pico(command: str) -> bool:
    if not SERIAL_ENABLED:
        return False

    command = str(command or "").strip()
    if not command:
        return False

    ser = _open_serial_if_needed()
    if not ser:
        return False

    try:
        with _serial_lock:
            ser.write((command + "\n").encode("utf-8"))
            ser.flush()
        print(f"[SERIAL] Pico: Send {command} OK")
        return True
    except Exception as e:
        print(f"[SERIAL] write failed: {e}")
        try:
            ser.close()
        except Exception:
            pass
        return False


# ---------------------------------------------------------
# DB helpers
# ---------------------------------------------------------
def _database_dsn() -> str:
    ssl_mode = "require" if DB_SSL else "disable"
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={ssl_mode}"


async def init_db() -> None:
    global db_pool, _plate_cache_task

    if db_pool is None:
        db_pool = await asyncpg.create_pool(
            dsn=_database_dsn(),
            min_size=1,
            max_size=10,
            command_timeout=30,
        )
        print("[INIT] asyncpg pool created")

    await refresh_plate_registry()

    if _plate_cache_task is None or _plate_cache_task.done():
        _plate_cache_task = asyncio.create_task(_plate_registry_refresher(), name="plate-registry-refresher")


async def close_db() -> None:
    global db_pool, _plate_cache_task

    if _plate_cache_task:
        _plate_cache_task.cancel()
        try:
            await _plate_cache_task
        except asyncio.CancelledError:
            pass
        except Exception:
            pass
        _plate_cache_task = None

    if db_pool is not None:
        await db_pool.close()
        db_pool = None

    global _serial_conn
    if _serial_conn:
        try:
            _serial_conn.close()
        except Exception:
            pass
        _serial_conn = None


async def _plate_registry_refresher() -> None:
    while True:
        try:
            await asyncio.sleep(PLATE_CACHE_REFRESH_SECONDS)
            await refresh_plate_registry()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[PLATE CACHE] refresh failed: {e}")


def normalize_plate_key(value: Optional[str]) -> str:
    if not value:
        return ""
    cleaned = "".join(ch for ch in str(value).upper() if ch.isalnum())
    return cleaned


async def refresh_plate_registry() -> None:
    global _plate_registry

    if db_pool is None:
        return

    sql = """
        SELECT
            "PlateNumber",
            "Driver",
            "Type",
            "Model",
            "Brand"
        FROM dbo."Vehicles"
        WHERE "Active" = true
          AND NULLIF(TRIM("PlateNumber"), '') IS NOT NULL
    """

    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql)

    next_registry: Dict[str, Dict[str, Any]] = {}

    for row in rows:
        plate_number = row.get("PlateNumber")
        plate_key = normalize_plate_key(plate_number)
        if not plate_key:
            continue

        driver_value = row.get("Driver")
        if isinstance(driver_value, str):
            try:
                driver_value = json.loads(driver_value)
            except Exception:
                driver_value = None

        next_registry[plate_key] = {
            "plate": plate_number,
            "driver": driver_value,
            "vehicle": {
                "type": row.get("Type"),
                "model": row.get("Model"),
                "brand": row.get("Brand"),
            },
        }

    with _plate_registry_lock:
        _plate_registry = next_registry

    print(f"[PLATE CACHE] loaded {len(next_registry)} registered plates")


# ---------------------------------------------------------
# Lights
# ---------------------------------------------------------
async def get_light_by_stream_id(stream_id: str) -> Optional[Dict[str, Any]]:
    if db_pool is None:
        return None

    sql = """
        SELECT
            "Name",
            "SecretKey",
            "CameraStreamId"
        FROM dbo."Lights"
        WHERE "Active" = true
          AND "CameraStreamId" = $1
        ORDER BY COALESCE("UpdatedAt", "CreatedAt") DESC
        LIMIT 1
    """

    async with db_pool.acquire() as conn:
        row = await conn.fetchrow(sql, stream_id)

    if not row:
        return None

    return {
        "name": row.get("Name"),
        "secretKey": row.get("SecretKey"),
        "cameraStreamId": row.get("CameraStreamId"),
    }


def build_light_serial_command(light: Optional[Dict[str, Any]], command: Optional[str]) -> Optional[str]:
    if not light or not command:
        return None

    name = str(light.get("name") or "").strip().upper()
    secret_key = str(light.get("secretKey") or "").strip().upper()
    command = str(command or "").strip().lower()

    if not name or not secret_key or not command:
        return None

    return f"{name} {secret_key} {command}"


def get_light_command_from_gate(gate: Dict[str, Any]) -> str:
    if not gate.get("vehicleFound"):
        return "off"

    if gate.get("driver") and gate.get("vehicle") and gate.get("plate"):
        return "green"

    return "red"


# ---------------------------------------------------------
# Gate state logic
# ---------------------------------------------------------
def _top_detected_vehicle(vehicles_payload: list) -> Optional[Dict[str, Any]]:
    if not vehicles_payload:
        return None

    top = vehicles_payload[0] or {}
    return {
        "type": top.get("class_name"),
        "brand": None,
        "model": None,
    }


def _lookup_registered_plate_sync(plate_text: Optional[str]) -> Optional[Dict[str, Any]]:
    plate_key = normalize_plate_key(plate_text)
    if not plate_key:
        return None

    with _plate_registry_lock:
        return _plate_registry.get(plate_key)


def compute_gate_state_fast(
    vehicles_payload: list,
    plates: list,
    camera_source: str,
) -> Dict[str, Any]:
    vehicle_found = len(vehicles_payload or []) > 0
    detected_vehicle = _top_detected_vehicle(vehicles_payload)
    plate_text = None
    ai_conf = None

    if plates:
        top_plate = plates[0] or {}
        ocr = top_plate.get("ocr") or {}
        plate_text = ocr.get("text")
        ai_conf = ocr.get("confidence")

    matched = _lookup_registered_plate_sync(plate_text)

    gate = {
        "stream_id": camera_source,
        "camera_source": camera_source,
        "vehicleFound": vehicle_found,
        "plate": plate_text,
        "driver": matched.get("driver") if matched else None,
        "vehicle": matched.get("vehicle") if matched else detected_vehicle,
        "lastUpdate": now_ms(),
        "imagePreview": None,
        "verification": "registered" if matched else ("not-registered" if vehicle_found else "none"),
        "aiConfidence": ai_conf,
    }

    active_gate_states[camera_source] = gate
    return gate


# ---------------------------------------------------------
# Logs + final realtime update
# ---------------------------------------------------------
async def _find_recent_duplicate_log(
    conn: asyncpg.Connection,
    plate_number: Optional[str],
    camera_source: str,
) -> bool:
    if not plate_number:
        return False

    sql = """
        SELECT 1
        FROM dbo."Logs"
        WHERE "PlateNumber" = $1
          AND "CameraSource" = $2
          AND "CreatedAt" >= NOW() - ($3 * INTERVAL '1 second')
        LIMIT 1
    """
    row = await conn.fetchrow(sql, plate_number, camera_source, LOG_DEDUP_SECONDS)
    return row is not None

async def _insert_log_if_needed(
    gate: Dict[str, Any],
    image_preview_filename: Optional[str],
) -> None:
    if db_pool is None:
        return

    plate_number = gate.get("plate")
    if not gate.get("vehicleFound"):
        return

    async with db_pool.acquire() as conn:
        if await _find_recent_duplicate_log(conn, plate_number, gate.get("camera_source") or ""):
            return

        sql = """
            INSERT INTO dbo."Logs" (
                "PlateNumber",
                "Driver",
                "Vehicle",
                "RoleType",
                "Verification",
                "CameraSource",
                "ImagePreview",
                "AIConfidence"
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """

        driver = gate.get("driver")
        vehicle = gate.get("vehicle")
        role_type = None
        if isinstance(driver, dict):
            role_type = driver.get("roleType")

        ai_confidence = gate.get("aiConfidence")
        try:
            ai_confidence = int(round(float(ai_confidence or 0) * 100))
        except Exception:
            ai_confidence = 0

        await conn.execute(
            sql,
            plate_number,
            json.dumps(driver) if driver is not None else None,
            json.dumps(vehicle) if vehicle is not None else None,
            role_type,
            "REGISTERED" if gate.get("driver") else "NOT_REGISTERED",
            gate.get("camera_source"),
            image_preview_filename,
            ai_confidence,
        )


async def update_gate_state_and_push(
    vehicles_payload: list,
    plates: list,
    camera_source: str,
    image_preview_filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Kept same function name to avoid changing the rest of your app.
    Internally this now does DB/log work + websocket publish.
    """
    gate = compute_gate_state_fast(
        vehicles_payload=vehicles_payload,
        plates=plates,
        camera_source=camera_source,
    )

    if image_preview_filename:
        gate["imagePreview"] = image_preview_filename

    active_gate_states[camera_source] = gate

    try:
        await _insert_log_if_needed(gate, image_preview_filename)
    except Exception as e:
        print(f"[LOGS] insert failed: {e}")

    try:
        payload = {
            "stream_id": gate.get("stream_id"),
            "vehicleFound": gate.get("vehicleFound"),
            "plate": gate.get("plate"),
            "driver": gate.get("driver"),
            "vehicle": gate.get("vehicle"),
            "lastUpdate": gate.get("lastUpdate"),
            "imagePreview": gate.get("imagePreview"),
            "verification": gate.get("verification"),
            "aiConfidence": gate.get("aiConfidence"),
        }
        await publish_event("gate-update", payload)
    except Exception as e:
        print(f"[WS] gate-update publish failed: {e}")

    return gate