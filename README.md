---
title: YOLO Vehicle Detector API
emoji: 🚗
colorFrom: blue
colorTo: pink
sdk: gradio
sdk_version: "4.36.0"
python_version: "3.10"
app_file: start.py
pinned: false
---

# YOLO FastAPI Vehicle Detection (Hugging Face Space)

This Space runs a **FastAPI server** (no Gradio UI) for YOLOv8 vehicle detection.

## Endpoints

- `GET /health`
- `POST /detect` — Upload an image to detect vehicles.

## Notes

- Uses `yolov8n.pt` for better perform
