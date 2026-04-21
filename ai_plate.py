import io
import os
import tempfile
import traceback
from functools import lru_cache
from typing import Dict, List, Literal, Optional, Tuple, get_args

import numpy as np
import torch
from fastapi import HTTPException
from PIL import Image, ImageOps
from paddleocr import PaddleOCR

from fast_alpr import ALPR
from fast_alpr.default_detector import PlateDetectorModel
from fast_alpr.default_ocr import OcrModel

FULL_IMAGE_MAX_SIDE = int(os.getenv("FULL_IMAGE_MAX_SIDE", "1280"))

PLATE_REFINEMENT_PADDING_LEFT = float(os.getenv("PLATE_REFINEMENT_PADDING_LEFT", "0.06"))
PLATE_REFINEMENT_PADDING_RIGHT = float(os.getenv("PLATE_REFINEMENT_PADDING_RIGHT", "0.10"))
PLATE_REFINEMENT_PADDING_TOP = float(os.getenv("PLATE_REFINEMENT_PADDING_TOP", "0.08"))
PLATE_REFINEMENT_PADDING_BOTTOM = float(os.getenv("PLATE_REFINEMENT_PADDING_BOTTOM", "0.08"))

MIN_PLATE_TEXT_LEN = int(os.getenv("MIN_PLATE_TEXT_LEN", "6"))
PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "en")

DETECTOR_MODELS: List[PlateDetectorModel] = list(get_args(PlateDetectorModel))
OCR_MODELS: List[str] = list(get_args(OcrModel))

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


@lru_cache(maxsize=1)
def get_paddle_ocr() -> PaddleOCR:
    use_gpu = torch.cuda.is_available()
    return PaddleOCR(
        use_angle_cls=False,
        lang=PADDLEOCR_LANG,
        show_log=False,
        use_gpu=use_gpu,
    )


def load_image_to_numpy(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {e}")
    return np.array(img)


def resize_if_needed(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    return img.resize(
        (max(1, int(round(w * scale))), max(1, int(round(h * scale)))),
        Image.LANCZOS,
    )


def pad_image_horizontal(
    img: Image.Image,
    left_ratio: float = 0.08,
    right_ratio: float = 0.16,
    top_ratio: float = 0.06,
    bottom_ratio: float = 0.06,
    fill=(255, 255, 255),
) -> Image.Image:
    w, h = img.size
    left = max(1, int(round(w * left_ratio)))
    right = max(1, int(round(w * right_ratio)))
    top = max(1, int(round(h * top_ratio)))
    bottom = max(1, int(round(h * bottom_ratio)))
    return ImageOps.expand(img, border=(left, top, right, bottom), fill=fill)


def clean_plate_text(text: Optional[str]) -> str:
    if not text:
        return ""
    return "".join(ch for ch in text.upper() if ch.isalnum())


def extract_best_numeric_plate(text: str) -> str:
    cleaned = clean_plate_text(text)
    if not cleaned:
        return ""

    digit_runs = []
    current = []

    for ch in cleaned:
        if ch.isdigit():
            current.append(ch)
        else:
            if current:
                digit_runs.append("".join(current))
                current = []
    if current:
        digit_runs.append("".join(current))

    if not digit_runs:
        return cleaned

    digit_runs.sort(key=len, reverse=True)
    best = digit_runs[0]

    # conservative fix for one extra leading noise digit
    if len(best) == 11 and best[1] == "0":
        best = best[1:]

    return best


def count_digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text or "")


def score_numeric_candidate(text: str, conf: float) -> float:
    cleaned = clean_plate_text(text)
    if not cleaned:
        return -9999.0

    digits = "".join(ch for ch in cleaned if ch.isdigit())
    alpha_count = sum(ch.isalpha() for ch in cleaned)
    digit_count = len(digits)

    score = float(conf)
    score += digit_count * 0.22
    score -= alpha_count * 0.30

    if digit_count >= 8:
        score += 0.60
    if digit_count >= 10:
        score += 0.90
    if digit_count == 0:
        score -= 2.0

    return score


def clamp_box(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))
    return x1, y1, x2, y2


def expand_local_plate_bbox(
    bbox: Dict[str, int],
    img_w: int,
    img_h: int,
    pad_left_ratio: float = PLATE_REFINEMENT_PADDING_LEFT,
    pad_right_ratio: float = PLATE_REFINEMENT_PADDING_RIGHT,
    pad_top_ratio: float = PLATE_REFINEMENT_PADDING_TOP,
    pad_bottom_ratio: float = PLATE_REFINEMENT_PADDING_BOTTOM,
) -> Tuple[int, int, int, int]:
    x1 = int(bbox.get("x1", 0))
    y1 = int(bbox.get("y1", 0))
    x2 = int(bbox.get("x2", 0))
    y2 = int(bbox.get("y2", 0))

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)

    pad_left = int(round(width * pad_left_ratio))
    pad_right = int(round(width * pad_right_ratio))
    pad_top = int(round(height * pad_top_ratio))
    pad_bottom = int(round(height * pad_bottom_ratio))

    return clamp_box(
        x1 - pad_left,
        y1 - pad_top,
        x2 + pad_right,
        y2 + pad_bottom,
        img_w,
        img_h,
    )


def crop_number_band(plate_img: Image.Image, mode: str = "tight") -> Image.Image:
    w, h = plate_img.size

    if mode == "tight":
        top = 0.22
        bottom = 0.60
    elif mode == "mid":
        top = 0.24
        bottom = 0.66
    elif mode == "low":
        top = 0.28
        bottom = 0.72
    else:
        top = 0.22
        bottom = 0.60

    x1 = int(round(w * 0.02))
    x2 = int(round(w * 0.98))
    y1 = int(round(h * top))
    y2 = int(round(h * bottom))

    x1 = max(0, min(x1, w - 1))
    x2 = max(x1 + 1, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(y1 + 1, min(y2, h))

    band = plate_img.crop((x1, y1, x2, y2))
    band = pad_image_horizontal(
        band,
        left_ratio=0.02,
        right_ratio=0.03,
        top_ratio=0.03,
        bottom_ratio=0.03,
    )
    return band


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


def run_plate_detector_on_pil(pil_img: Image.Image, detector_name: str, ocr_name: str) -> List[dict]:
    try:
        alpr = get_alpr(detector_name, ocr_name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name

        pil_img.save(temp_path, format="PNG", optimize=False)
        results = alpr.predict(temp_path)
    except Exception as e:
        print("[AI_PLATE] Full traceback:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ALPR error: {e}")
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass

    return [serialize_alpr_result(r) for r in results] if results else []


def recognize_with_paddle_boxes(img: Image.Image) -> List[dict]:
    """
    Returns OCR pieces with text/conf/bbox so we can score full-image fallback.
    """
    ocr = get_paddle_ocr()
    img_np = np.array(img.convert("RGB"))
    try:
        result = ocr.ocr(img_np, cls=False)
    except Exception as e:
        print(f"[AI_PLATE] PaddleOCR error: {e}")
        return []

    if not result or not result[0]:
        return []

    pieces = []
    for item in result[0]:
        try:
            box = item[0]
            rec = item[1]
            txt = str(rec[0] or "").strip()
            conf = float(rec[1] or 0.0)
            if not txt:
                continue

            xs = [float(pt[0]) for pt in box]
            ys = [float(pt[1]) for pt in box]

            pieces.append(
                {
                    "text": clean_plate_text(txt),
                    "confidence": conf,
                    "bbox": {
                        "x1": int(min(xs)),
                        "y1": int(min(ys)),
                        "x2": int(max(xs)),
                        "y2": int(max(ys)),
                    },
                }
            )
        except Exception:
            continue

    return pieces


def recognize_with_paddle(img: Image.Image) -> Tuple[str, float]:
    pieces = recognize_with_paddle_boxes(img)
    if not pieces:
        return "", 0.0

    pieces.sort(key=lambda p: p["bbox"]["x1"])
    joined = "".join(p["text"] for p in pieces)
    conf = float(sum(p["confidence"] for p in pieces) / len(pieces))
    numeric = extract_best_numeric_plate(joined)

    print(f"[AI_PLATE RAW] joined={joined}")
    print(f"[AI_PLATE CLEAN] numeric={numeric}")

    return numeric, conf


def fallback_full_image_ocr(base_full: Image.Image) -> List[dict]:
    """
    Cheap fallback when detector fails:
    OCR whole image once, score numeric candidates.
    """
    pieces = recognize_with_paddle_boxes(base_full)
    if not pieces:
        return []

    candidates = []
    for p in pieces:
        raw_text = p["text"]
        digits = extract_best_numeric_plate(raw_text)
        if not digits:
            continue

        score = score_numeric_candidate(digits, p["confidence"])
        print(
            f"[AI_PLATE OCR-FALLBACK] raw={raw_text} digits={digits} "
            f"conf={p['confidence']:.4f} score={score:.4f}"
        )
        candidates.append(
            {
                "ocr": {
                    "text": digits,
                    "confidence": p["confidence"],
                },
                "bbox": p["bbox"],
                "_score": score,
            }
        )

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["_score"], reverse=True)
    best = candidates[0]

    if count_digits(best["ocr"]["text"]) < MIN_PLATE_TEXT_LEN:
        return []

    return [
        {
            "ocr": best["ocr"],
            "bbox": best["bbox"],
        }
    ]


def run_alpr(img_np: np.ndarray, detector_name: str, ocr_name: str) -> dict:
    """
    Fastest robust flow:
    1. detector once on original image
    2. if found -> tight OCR first, fallback mid/low only if needed
    3. if detector fails -> full-image OCR fallback once
    """
    base_full = Image.fromarray(img_np).convert("RGB")
    base_full = resize_if_needed(base_full, FULL_IMAGE_MAX_SIDE)

    detected_plates = run_plate_detector_on_pil(base_full, detector_name, ocr_name)

    # Detector failed -> use full-image OCR fallback once
    if not detected_plates:
        plates = fallback_full_image_ocr(base_full)
        return {
            "model": {
                "detector_model": detector_name,
                "ocr_model": "paddleocr_fast",
            },
            "plates": plates,
        }

    def bbox_area(p: dict) -> int:
        bbox = p.get("bbox") or {}
        return max(0, int(bbox.get("x2", 0)) - int(bbox.get("x1", 0))) * max(
            0, int(bbox.get("y2", 0)) - int(bbox.get("y1", 0))
        )

    detected_plates.sort(key=bbox_area, reverse=True)
    best = detected_plates[0]
    bbox = best.get("bbox")

    if not bbox:
        return {
            "model": {
                "detector_model": detector_name,
                "ocr_model": "paddleocr_fast",
            },
            "plates": [],
        }

    img_w, img_h = base_full.size
    x1, y1, x2, y2 = expand_local_plate_bbox(bbox, img_w, img_h)
    plate_crop = base_full.crop((x1, y1, x2, y2))

    # 1st OCR attempt: tight only
    tight_band = crop_number_band(plate_crop, mode="tight")
    tight_text, tight_conf = recognize_with_paddle(tight_band)
    tight_digits = extract_best_numeric_plate(tight_text)
    tight_score = score_numeric_candidate(tight_digits, tight_conf)

    print(
        f"[AI_PLATE BAND] mode=tight raw={tight_text} digits={tight_digits} "
        f"conf={tight_conf:.4f} score={tight_score:.4f}"
    )

    if count_digits(tight_digits) >= 8:
        text = tight_digits
        conf = tight_conf
    else:
        candidates = []
        if tight_digits:
            candidates.append(
                {
                    "text": tight_digits,
                    "confidence": tight_conf,
                    "score": tight_score,
                }
            )

        for mode in ("mid", "low"):
            band = crop_number_band(plate_crop, mode=mode)
            text_try, conf_try = recognize_with_paddle(band)
            cleaned_digits = extract_best_numeric_plate(text_try)
            score = score_numeric_candidate(cleaned_digits, conf_try)

            print(
                f"[AI_PLATE BAND] mode={mode} raw={text_try} digits={cleaned_digits} "
                f"conf={conf_try:.4f} score={score:.4f}"
            )

            if cleaned_digits:
                candidates.append(
                    {
                        "text": cleaned_digits,
                        "confidence": conf_try,
                        "score": score,
                    }
                )

        if not candidates:
            return {
                "model": {
                    "detector_model": detector_name,
                    "ocr_model": "paddleocr_fast",
                },
                "plates": [],
            }

        candidates.sort(key=lambda x: x["score"], reverse=True)
        best_candidate = candidates[0]
        text = best_candidate["text"]
        conf = best_candidate["confidence"]

    if not text or len(text) < MIN_PLATE_TEXT_LEN:
        return {
            "model": {
                "detector_model": detector_name,
                "ocr_model": "paddleocr_fast",
            },
            "plates": [],
        }

    plates = [
        {
            "ocr": {
                "text": text,
                "confidence": conf,
            },
            "bbox": bbox,
        }
    ]

    return {
        "model": {
            "detector_model": detector_name,
            "ocr_model": "paddleocr_fast",
        },
        "plates": plates,
    }