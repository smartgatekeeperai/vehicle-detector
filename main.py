# main.py

import io
import json
import os
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------
# Load .env FIRST, before importing modules that read env
# ---------------------------------------------------------
load_dotenv()
parent_env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
if os.path.exists(parent_env_path):
    load_dotenv(parent_env_path, override=False)

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageDraw, ImageFont

from ai_plate import DETECTOR_MODELS, OCR_MODELS, DetectorName, OcrName, load_image_to_numpy, run_alpr
from ai_vehicle import DEVICE, YOLO_MODEL_PATH, force_vehicle_from_plates, run_yolo_vehicles
from services import (
    IMAGE_SAVE_DIR,
    SERIAL_ENABLED,
    SERIAL_PORT,
    STREAM_ONLINE_WINDOW_MS,
    STREAM_STALE_REMOVE_MS,
    active_streams,
    build_light_serial_command,
    build_stream_list,
    cleanup_streams,
    close_db,
    compute_gate_state_fast,
    get_light_by_stream_id,
    get_light_command_from_gate,
    init_db,
    latest_frames,
    now_ms,
    save_image_bytes,
    send_serial_command_to_pico,
    touch_stream,
    update_gate_state_and_push,
)
from ws_client import publish_event, start_realtime_ws, stop_realtime_ws, ws_status

app = FastAPI(
    title="Smart Gate Keeper - YOLOv8 + FastALPR + Fast PaddleOCR",
    version="2.4.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_font(size: int = 24):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _draw_label(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, font):
    if not text:
        return

    try:
        bbox = draw.textbbox((x, y), text, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw = max(40, len(text) * 8)
        th = 18

    pad_x = 6
    pad_y = 4

    rx1 = x
    ry1 = max(0, y - th - (pad_y * 2) - 4)
    rx2 = x + tw + (pad_x * 2)
    ry2 = ry1 + th + (pad_y * 2)

    draw.rectangle([rx1, ry1, rx2, ry2], fill=(0, 0, 0))
    draw.text((rx1 + pad_x, ry1 + pad_y), text, fill=(255, 255, 255), font=font)


def _safe_bbox_coords(bbox: dict, img_w: int, img_h: int):
    x1 = int(max(0, min(bbox.get("x1", 0), img_w - 1)))
    y1 = int(max(0, min(bbox.get("y1", 0), img_h - 1)))
    x2 = int(max(x1 + 1, min(bbox.get("x2", 0), img_w)))
    y2 = int(max(y1 + 1, min(bbox.get("y2", 0), img_h)))
    return x1, y1, x2, y2


def build_annotated_preview_bytes(
    original_image_bytes: bytes,
    plates: list,
    vehicles: list,
) -> bytes:
    img = Image.open(io.BytesIO(original_image_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _load_font(24)

    img_w, img_h = img.size

    for vehicle in vehicles or []:
        vb = vehicle.get("bbox") if isinstance(vehicle, dict) else None
        if not vb:
            continue

        x1 = int(vb.get("x1", 0))
        y1 = int(vb.get("y1", 0))
        x2 = int(vb.get("x2", 0))
        y2 = int(vb.get("y2", 0))

        x1 = max(0, min(x1, img_w - 1))
        y1 = max(0, min(y1, img_h - 1))
        x2 = max(x1 + 1, min(x2, img_w))
        y2 = max(y1 + 1, min(y2, img_h))

        draw.rectangle([x1, y1, x2, y2], outline=(255, 215, 0), width=4)

        label = vehicle.get("class_name") or "vehicle"
        conf = vehicle.get("confidence")
        if conf is not None:
            try:
                label = f"{label} {float(conf):.2f}"
            except Exception:
                pass

        _draw_label(draw, x1, y1, label, font)

    for plate in plates or []:
        bbox = plate.get("bbox")
        ocr = plate.get("ocr") or {}
        text = ocr.get("text") or ""

        if not bbox:
            continue

        x1, y1, x2, y2 = _safe_bbox_coords(bbox, img_w, img_h)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=5)
        _draw_label(draw, x1, y1, text, font)

    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)
    return out.getvalue()


async def finalize_detect_side_effects(
    sid: str,
    original_image_bytes: bytes,
    vehicles_payload: list,
    plates: list,
):
    """
    Non-critical path:
    - build/save preview image
    - DB/log state update
    - websocket update through services
    """
    image_preview_filename = None

    try:
        preview_bytes = build_annotated_preview_bytes(
            original_image_bytes=original_image_bytes,
            plates=plates,
            vehicles=vehicles_payload,
        )
        image_preview_filename = save_image_bytes(preview_bytes, sid, ext=".jpg")
        print(f"[/detect bg] image_preview_filename={image_preview_filename}")
    except Exception as e:
        print(f"[/detect bg] preview annotation/save failed: {e}")

    try:
        await update_gate_state_and_push(
            vehicles_payload,
            plates,
            camera_source=sid,
            image_preview_filename=image_preview_filename,
        )
    except Exception as e:
        print(f"[/detect bg] update_gate_state_and_push failed: {e}")


@app.on_event("startup")
async def startup():
    await init_db()
    await start_realtime_ws()

    try:
        import numpy as np

        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = run_yolo_vehicles(dummy, "__warmup__")
        _ = run_alpr(dummy, DETECTOR_MODELS[0], OCR_MODELS[0], vehicles=[])
        print("[INIT] AI warmup complete")
    except Exception as e:
        print(f"[INIT] AI warmup failed: {e}")


@app.on_event("shutdown")
async def shutdown():
    await stop_realtime_ws()
    await close_db()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "yolo_model": YOLO_MODEL_PATH,
        "image_save_dir": IMAGE_SAVE_DIR,
        "serial_enabled": SERIAL_ENABLED,
        "serial_port": SERIAL_PORT if SERIAL_ENABLED else None,
        "stream_online_window_ms": STREAM_ONLINE_WINDOW_MS,
        "stream_stale_remove_ms": STREAM_STALE_REMOVE_MS,
        "active_gate_states": list(active_streams.keys()),
        "realtime": ws_status(),
    }


@app.get("/alpr/models")
async def alpr_models():
    return {
        "detector_models": DETECTOR_MODELS,
        "ocr_models": OCR_MODELS,
        "active_recognizer": "paddleocr_fast",
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
        await publish_event("video-frame", {"stream_id": sid, "ts": ts})
    except Exception as e:
        print("[MAIN] /stream-frame websocket publish error:", e)

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


@app.get("/images/{filename}")
async def get_saved_image(filename: str):
    safe_name = Path(filename).name
    file_path = Path(IMAGE_SAVE_DIR) / safe_name

    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found")

    return FileResponse(str(file_path))


@app.post("/test-ws")
async def test_ws():
    payload = {
        "stream_id": "test-stream",
        "vehicleFound": True,
        "plate": "TEST-123",
        "driver": {"fullName": "Test Driver"},
        "vehicle": {"type": "Sedan", "brand": "Toyota", "model": "Vios"},
        "lastUpdate": int(time.time() * 1000),
    }
    await publish_event("gate-update", payload)
    return {
        "success": True,
        "realtime": ws_status(),
        "payload": payload,
    }


@app.post("/detect")
async def detect_all(
    background_tasks: BackgroundTasks,
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

    sid = (stream_id or "").strip() or "mobile-1"
    ts = now_ms()

    touch_stream(sid, ts)
    cleanup_streams(ts)

    t0 = time.time()

    t_load_0 = time.time()
    img_np = load_image_to_numpy(data)
    h, w = img_np.shape[:2]
    t_load_1 = time.time()

    t_vehicle_0 = time.time()
    vehicles = run_yolo_vehicles(img_np, sid)
    vehicles_payload = [v.model_dump() for v in vehicles]
    t_vehicle_1 = time.time()

    t_plate_0 = time.time()
    alpr_result = run_alpr(img_np, detector_name, ocr_name, vehicles=vehicles_payload)
    plates = alpr_result["plates"]
    t_plate_1 = time.time()

    t_forced_vehicle_0 = time.time()
    if not vehicles_payload and plates:
        forced_vehicles = force_vehicle_from_plates(img_np, plates, sid)
        if forced_vehicles:
            vehicles_payload = [v.model_dump() for v in forced_vehicles]
    t_forced_vehicle_1 = time.time()

    t_gate_0 = time.time()
    gate = compute_gate_state_fast(
        vehicles_payload,
        plates,
        camera_source=sid,
    )

    light = await get_light_by_stream_id(sid)
    light_command = get_light_command_from_gate(gate)
    serial_command = build_light_serial_command(light, light_command)

    if serial_command:
        send_serial_command_to_pico(serial_command)
    t_gate_1 = time.time()

    background_tasks.add_task(
        finalize_detect_side_effects,
        sid,
        data,
        vehicles_payload,
        plates,
    )

    t1 = time.time()

    print(f"[/detect] stream_id={sid}")
    print(f"[/detect] vehicles_count={len(vehicles_payload)}")
    print(f"[/detect] plates_count={len(plates)}")
    if vehicles_payload:
        print(f"[/detect] top_vehicle={vehicles_payload[0].get('class_name')} conf={vehicles_payload[0].get('confidence')}")
    if plates:
        print(f"[/detect] top_plate={json.dumps(plates[0], default=str)}")
    print(f"[/detect] serial_command={serial_command}")

    print(f"[/detect timing] load_ms={(t_load_1 - t_load_0)*1000:.1f}")
    print(f"[/detect timing] vehicle_ms={(t_vehicle_1 - t_vehicle_0)*1000:.1f}")
    print(f"[/detect timing] plate_ms={(t_plate_1 - t_plate_0)*1000:.1f}")
    print(f"[/detect timing] forced_vehicle_ms={(t_forced_vehicle_1 - t_forced_vehicle_0)*1000:.1f}")
    print(f"[/detect timing] gate_ms={(t_gate_1 - t_gate_0)*1000:.1f}")
    print(f"[/detect timing] total_ms={(t1 - t0)*1000:.1f}")

    return JSONResponse(
        {
            "success": True,
            "image_width": w,
            "image_height": h,
            "vehicles": vehicles_payload,
            "plates": plates,
            "alpr_model": alpr_result["model"],
            "yolo_model": YOLO_MODEL_PATH,
            "total_time_ms": (t1 - t0) * 1000.0,
            "gate_state": gate,
            "light": light,
            "light_command": light_command,
            "serial_command": serial_command,
            "stream_id": sid,
        }
    )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)