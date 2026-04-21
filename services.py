# services.py

import json
import os
import socket
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from pusher import Pusher

try:
    import serial
except Exception:
    serial = None

DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    DB_USER = os.getenv("DB_USER", "")
    DB_PASSWORD = os.getenv("DB_PASSWORD", "")
    DB_HOST = os.getenv("DB_HOST", "localhost")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME", "")
    if DB_USER and DB_NAME:
        DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

PUBLIC_BASE_URL = (os.getenv("PUBLIC_BASE_URL") or "").rstrip("/")

PASS_LOCK_MS = int(os.getenv("PASS_LOCK_MS", "10000"))
CLEAR_AFTER_MS = int(os.getenv("CLEAR_AFTER_MS", "5000"))

# 1 minute duplicate suppression window, PlateNumber only
LOG_WINDOW_MS = int(os.getenv("LOG_WINDOW_MS", str(60 * 1000)))

STREAM_ONLINE_WINDOW_MS = int(os.getenv("STREAM_ONLINE_WINDOW_MS", "5000"))
STREAM_STALE_REMOVE_MS = int(os.getenv("STREAM_STALE_REMOVE_MS", "60000"))

SERIAL_ENABLED = os.getenv("SERIAL_ENABLED", "false").lower() == "true"
SERIAL_PORT = os.getenv("SERIAL_PORT", "COM5")
SERIAL_BAUDRATE = int(os.getenv("SERIAL_BAUDRATE", "115200"))
SERIAL_TIMEOUT = float(os.getenv("SERIAL_TIMEOUT", "1"))
SERIAL_OPEN_DELAY = float(os.getenv("SERIAL_OPEN_DELAY", "2.0"))

IMAGE_SAVE_DIR = os.getenv("IMAGE_SAVE_DIR", os.path.join(os.getcwd(), "storage", "logs"))
IMAGE_ROUTE_PREFIX = os.getenv("IMAGE_ROUTE_PREFIX", "/images").strip()

Path(IMAGE_SAVE_DIR).mkdir(parents=True, exist_ok=True)

latest_frames: Dict[str, Dict[str, Any]] = {}
active_streams: Dict[str, Dict[str, Any]] = {}

gate_states_by_stream: Dict[str, Dict[str, Any]] = {}
registered_gate_state_by_stream: Dict[str, Optional[Dict[str, Any]]] = {}
registered_gate_ts_by_stream: Dict[str, int] = {}

db_pool: Optional[asyncpg.Pool] = None


def now_ms() -> int:
    return int(time.time() * 1000)


def create_empty_gate_state(stream_id: str) -> Dict[str, Any]:
    return {
        "stream_id": stream_id,
        "vehicleFound": False,
        "plate": None,
        "driver": None,
        "vehicle": None,
        "lastUpdate": None,
    }


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


async def init_db() -> None:
    global db_pool
    if not DATABASE_URL:
        db_pool = None
        print("[SERVICES] DATABASE_URL is not set")
        return

    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        print("[SERVICES] asyncpg pool created")
    except Exception as e:
        db_pool = None
        print(f"[SERVICES] Failed to create DB pool: {e}")


async def close_db() -> None:
    global db_pool
    if db_pool is not None:
        await db_pool.close()
        db_pool = None
        print("[SERVICES] DB pool closed")


async def db_query(sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
    global db_pool
    params = params or []
    if db_pool is None:
        print("[SERVICES DB] skipped query: DB pool is not initialized")
        return []
    async with db_pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)
        return [dict(r) for r in rows]


async def db_execute(sql: str, params: Optional[List[Any]] = None) -> None:
    global db_pool
    params = params or []
    if db_pool is None:
        print("[SERVICES DB] skipped execute: DB pool is not initialized")
        return
    async with db_pool.acquire() as conn:
        await conn.execute(sql, *params)


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
            print(f"[SERVICES] Pusher muted {self.retry_seconds}s: {e}")
            return None


pusher_mode = "dummy"
pusher_reason = ""
pusher_client: Any = DummyPusher("uninitialized")

if not PUSHER_APP_ID:
    pusher_mode = "dummy"
    pusher_reason = "missing PUSHER_APP_ID"
    pusher_client = DummyPusher(pusher_reason)
else:
    if USE_LOCAL_PUSHER_EFFECTIVE and PUSHER_HOST:
        local_reachable = can_connect_socket(PUSHER_HOST, PUSHER_PORT, timeout=PUSHER_CONNECT_TIMEOUT)
        if local_reachable:
            try:
                raw_local_client = Pusher(
                    app_id=PUSHER_APP_ID,
                    key=PUSHER_KEY,
                    secret=PUSHER_SECRET,
                    host=PUSHER_HOST,
                    port=PUSHER_PORT,
                    ssl=False,
                )
                pusher_client = SafePusher(raw_local_client, mode="local", retry_seconds=PUSHER_RETRY_SECONDS)
                pusher_mode = "local"
                pusher_reason = f"connected to local server at {PUSHER_HOST}:{PUSHER_PORT}"
            except Exception as e:
                pusher_mode = "dummy"
                pusher_reason = f"local init failed: {e}"
                pusher_client = DummyPusher(pusher_reason)
        else:
            pusher_mode = "dummy"
            pusher_reason = f"local server not reachable at {PUSHER_HOST}:{PUSHER_PORT}"
            pusher_client = DummyPusher(pusher_reason)
    else:
        try:
            raw_cloud_client = Pusher(
                app_id=PUSHER_APP_ID,
                key=PUSHER_KEY,
                secret=PUSHER_SECRET,
                cluster=PUSHER_CLUSTER,
                ssl=True,
            )
            pusher_client = SafePusher(raw_cloud_client, mode="cloud", retry_seconds=PUSHER_RETRY_SECONDS)
            pusher_mode = "cloud"
            pusher_reason = f"using cloud cluster={PUSHER_CLUSTER}"
        except Exception as e:
            pusher_mode = "dummy"
            pusher_reason = f"cloud init failed: {e}"
            pusher_client = DummyPusher(pusher_reason)

print(f"[SERVICES] Realtime mode = {pusher_mode} ({pusher_reason})")


def sanitize_stream_id(stream_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in (stream_id or "mobile-1"))
    return safe[:80] or "mobile-1"


def generate_image_filename(stream_id: str, ext: str = ".jpg") -> str:
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    ms = f"{int(now.microsecond / 1000):03d}"
    safe_stream = sanitize_stream_id(stream_id)
    return f"{ts}_{ms}_{safe_stream}{ext}"


def save_image_bytes(data: bytes, stream_id: str, ext: str = ".jpg") -> Optional[str]:
    try:
        filename = generate_image_filename(stream_id, ext=ext)
        full_path = Path(IMAGE_SAVE_DIR) / filename
        with open(full_path, "wb") as f:
            f.write(data)
        return filename
    except Exception as e:
        print(f"[SERVICES IMAGE] save failed: {e}")
        return None


def build_image_url(filename: Optional[str]) -> Optional[str]:
    if not filename:
        return None
    route_prefix = IMAGE_ROUTE_PREFIX.rstrip("/")
    return f"{route_prefix}/{filename}"


async def get_light_by_stream_id(stream_id: str) -> Optional[Dict[str, Any]]:
    if not stream_id:
        return None

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
        ORDER BY "Name" ASC
    """
    rows = await db_query(sql, [stream_id])
    if not rows:
        return None
    return rows[0]


def get_light_command_from_gate(gate: Dict[str, Any]) -> str:
    return "green" if gate.get("vehicleFound") and gate.get("plate") else "red"


def build_light_serial_command(light: Dict[str, Any], command: str) -> Optional[str]:
    if not light:
        return None

    name = str(light.get("Name") or "").strip()
    secret = str(light.get("SecretKey") or "").strip()
    cmd = str(command or "").strip().lower()

    if not name or not secret or not cmd:
        return None

    return f"{name} {secret} {cmd}"


def send_serial_command_to_pico(command_line: str) -> None:
    if not SERIAL_ENABLED:
        print(f"[SERVICES SERIAL] skipped: {command_line}")
        return

    if serial is None:
        print("[SERVICES SERIAL] pyserial not installed")
        return

    command_line = (command_line or "").strip()
    if not command_line:
        return

    try:
        with serial.Serial(SERIAL_PORT, SERIAL_BAUDRATE, timeout=SERIAL_TIMEOUT) as ser:
            time.sleep(SERIAL_OPEN_DELAY)
            ser.write((command_line + "\n").encode("utf-8"))
            ser.flush()
            print(f"[SERVICES SERIAL] sent: {command_line}")
    except Exception as e:
        print(f"[SERVICES SERIAL] failed: {e}")


async def get_last_log_time_ms_for_plate(plate_text: str) -> Optional[int]:
    """
    Checks only PlateNumber, ignoring Verification and all other columns.
    """
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
        print("[SERVICES LOG] warning:", e)
        return None


async def insert_log_row(
    plate_number: str,
    driver_json: Optional[dict],
    vehicle_json: Optional[dict],
    role_type: Optional[str],
    verification: str,
    camera_source: str,
    image_preview: Optional[str],
    ai_confidence: int,
) -> None:
    sql = """
        INSERT INTO dbo."Logs"
            ("PlateNumber", "Driver", "Vehicle", "RoleType", "Verification", "CameraSource", "ImagePreview", "AIConfidence")
        VALUES
            ($1, $2::jsonb, $3::jsonb, $4, $5, $6, $7, $8)
    """
    await db_execute(
        sql,
        [
            plate_number,
            json.dumps(driver_json) if driver_json is not None else None,
            json.dumps(vehicle_json) if vehicle_json is not None else None,
            role_type,
            verification,
            camera_source,
            image_preview,
            int(ai_confidence),
        ],
    )


def push_gate_update(stream_id: str, gate: Dict[str, Any]) -> None:
    payload = {**gate, "stream_id": stream_id}
    try:
        pusher_client.trigger("gate-channel", "gate-update", payload)
    except Exception as e:
        print("[SERVICES PUSHER] gate-update failed:", e)


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
    best_conf = -1.0

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

    return best_text, max(0.0, best_conf)


def to_ai_confidence_bigint(best_conf_0_1: float) -> int:
    if best_conf_0_1 < 0:
        best_conf_0_1 = 0.0
    if best_conf_0_1 > 1:
        best_conf_0_1 = 1.0
    return int(round(best_conf_0_1 * 100))


def compute_gate_state_fast(
    vehicles: List[Any],
    plates: List[dict],
    camera_source: str,
) -> Dict[str, Any]:
    stream_id = (camera_source or "").strip() or "mobile-1"
    now_ts = now_ms()
    vehicle_count = len(vehicles)

    best_plate_text, _ = pick_best_plate_text_and_conf(plates)

    state = set_gate_state_for_stream(
        stream_id,
        {
            "stream_id": stream_id,
            "vehicleFound": vehicle_count > 0 or bool(best_plate_text),
            "plate": best_plate_text,
            "driver": None,
            "vehicle": None,
            "lastUpdate": now_ts,
        },
    )
    return state


async def update_gate_state_and_push(
    vehicles: List[Any],
    plates: List[dict],
    camera_source: str,
    image_preview_filename: Optional[str] = None,
) -> Dict[str, Any]:
    stream_id = (camera_source or "").strip() or "mobile-1"
    now_ts = now_ms()
    vehicle_count = len(vehicles)

    gate_state = get_gate_state_for_stream(stream_id)
    current_registered_gate_state = registered_gate_state_by_stream.get(stream_id)
    current_registered_ts = registered_gate_ts_by_stream.get(stream_id, 0)

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
        gate_state = set_gate_state_for_stream(
            stream_id,
            {
                "stream_id": stream_id,
                "vehicleFound": True,
                "plate": None,
                "driver": None,
                "vehicle": None,
                "lastUpdate": now_ts,
            },
        )
        push_gate_update(stream_id, gate_state)
        return gate_state

    cleaned_plates = normalize_plate_text(plates)
    best_plate_text, best_plate_conf = pick_best_plate_text_and_conf(plates)
    ai_conf_bigint = to_ai_confidence_bigint(best_plate_conf)
    image_preview = image_preview_filename

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
        print("[SERVICES DB] error:", db_err)
        gate_state = set_gate_state_for_stream(
            stream_id,
            {
                "stream_id": stream_id,
                "vehicleFound": vehicle_count > 0,
                "plate": best_plate_text,
                "driver": None,
                "vehicle": None,
                "lastUpdate": now_ts,
            },
        )
        push_gate_update(stream_id, gate_state)
        return gate_state

    async def should_insert_log(plate_number: str) -> bool:
        """
        Duplicate suppression uses only PlateNumber.
        If same plate exists less than LOG_WINDOW_MS ago, skip insert.
        """
        try:
            last_ts_ms = await get_last_log_time_ms_for_plate(plate_number)
            if last_ts_ms is not None and (now_ts - last_ts_ms) <= LOG_WINDOW_MS:
                print(f"[SERVICES LOG] skip duplicate plate={plate_number} within {LOG_WINDOW_MS}ms")
                return False
            return True
        except Exception:
            return True

    if not rows:
        if best_plate_text and await should_insert_log(best_plate_text):
            try:
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
            except Exception as e:
                print("[SERVICES LOG] insert failed:", e)

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
    except Exception as e:
        print("[SERVICES LOG] insert failed:", e)

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

    push_gate_update(stream_id, new_state)
    return new_state