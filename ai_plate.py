# ai_plate.py

import io
import os
import re
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

# -------------------------------------------------------------------
# Faster defaults but still tuned for short-character plate support
# -------------------------------------------------------------------
FULL_IMAGE_MAX_SIDE = int(os.getenv("FULL_IMAGE_MAX_SIDE", "1024"))

PLATE_REFINEMENT_PADDING_LEFT = float(os.getenv("PLATE_REFINEMENT_PADDING_LEFT", "0.05"))
PLATE_REFINEMENT_PADDING_RIGHT = float(os.getenv("PLATE_REFINEMENT_PADDING_RIGHT", "0.08"))
PLATE_REFINEMENT_PADDING_TOP = float(os.getenv("PLATE_REFINEMENT_PADDING_TOP", "0.08"))
PLATE_REFINEMENT_PADDING_BOTTOM = float(os.getenv("PLATE_REFINEMENT_PADDING_BOTTOM", "0.08"))

MIN_PLATE_TEXT_LEN = int(os.getenv("MIN_PLATE_TEXT_LEN", "4"))
MAX_PLATE_TEXT_LEN = int(os.getenv("MAX_PLATE_TEXT_LEN", "12"))
PADDLEOCR_LANG = os.getenv("PADDLEOCR_LANG", "en")

FAST_ACCEPT_CONF = float(os.getenv("FAST_ACCEPT_CONF", "0.72"))
SECOND_ACCEPT_CONF = float(os.getenv("SECOND_ACCEPT_CONF", "0.66"))

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

# -------------------------------------------------------------------
# OCR noise words commonly merged with Philippine temporary plates
# -------------------------------------------------------------------
NOISE_TOKENS = sorted(
    {
        "REGISTERED",
        "REGISTER",
        "REGIST",
        "TEMPORARYPLATE",
        "TEMPORARY",
        "PLATE",
        "REGION",
        "MOTORCYCLECITY",
        "MOTORCYCLE",
        "CITY",
        "HONDA",
        "YAMAHA",
        "KAWASAKI",
        "SUZUKI",
        "NMAX",
        "CLICK",
        "MIO",
        "TMX",
        "WAVE",
    },
    key=len,
    reverse=True,
)

# Fragments that often appear when OCR only catches part of the printed word
NOISE_EDGE_FRAGMENTS = sorted(
    {
        "REGIST",
        "REGIS",
        "REGI",
        "REG",
        "REGION",
        "TEMPORARY",
        "PLATE",
    },
    key=len,
    reverse=True,
)


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
    return np.ascontiguousarray(np.array(img))


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
    left_ratio: float = 0.05,
    right_ratio: float = 0.08,
    top_ratio: float = 0.04,
    bottom_ratio: float = 0.04,
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


def count_digits(text: str) -> int:
    return sum(ch.isdigit() for ch in text or "")


def count_letters(text: str) -> int:
    return sum(ch.isalpha() for ch in text or "")


def normalize_final_plate_text(text: str) -> str:
    cleaned = clean_plate_text(text)
    if not cleaned:
        return ""
    if len(cleaned) > MAX_PLATE_TEXT_LEN:
        cleaned = cleaned[:MAX_PLATE_TEXT_LEN]
    return cleaned


def longest_digit_run(text: str) -> str:
    cleaned = clean_plate_text(text)
    if not cleaned:
        return ""

    runs = re.findall(r"\d+", cleaned)
    if not runs:
        return ""

    runs.sort(key=len, reverse=True)
    best = runs[0]

    if len(best) == 11 and len(best) >= 2 and best[1] == "0":
        best = best[1:]

    if len(best) > MAX_PLATE_TEXT_LEN:
        best = best[:MAX_PLATE_TEXT_LEN]

    return best


def _remove_known_noise_tokens(text: str) -> str:
    cleaned = clean_plate_text(text)
    if not cleaned:
        return ""

    out = cleaned
    for token in NOISE_TOKENS:
        out = out.replace(token, "")
    return out


def _trim_edge_noise_fragments(text: str) -> str:
    """
    Removes partial OCR leftovers on the left/right side only,
    without destroying the middle of the actual plate.
    """
    value = clean_plate_text(text)
    if not value:
        return ""

    changed = True
    while changed and value:
        changed = False

        for frag in NOISE_EDGE_FRAGMENTS:
            if value.startswith(frag) and len(value) > len(frag):
                value = value[len(frag):]
                changed = True
                break

        if changed:
            continue

        for frag in NOISE_EDGE_FRAGMENTS:
            if value.endswith(frag) and len(value) > len(frag):
                value = value[: -len(frag)]
                changed = True
                break

    return value


def sanitize_ocr_text_for_plate(text: str) -> str:
    """
    Main sanitizer for OCR text before candidate extraction.

    Handles cases like:
    - REGION7929GOR      -> 7929GOR
    - 123ABCREGIST       -> 123ABC
    - REGISTERED123ABC   -> 123ABC
    """
    cleaned = clean_plate_text(text)
    if not cleaned:
        return ""

    cleaned = _remove_known_noise_tokens(cleaned)
    cleaned = _trim_edge_noise_fragments(cleaned)
    cleaned = clean_plate_text(cleaned)
    return cleaned


def trim_numeric_plate_noise(candidate: str) -> str:
    if not candidate:
        return ""

    cleaned = normalize_final_plate_text(candidate)
    if not cleaned:
        return ""

    digit_count = count_digits(cleaned)
    letter_count = count_letters(cleaned)

    m = re.match(r"^(\d{7,12})([A-Z]{1,2})$", cleaned)
    if m and digit_count >= 7 and letter_count <= 2:
        return m.group(1)

    return cleaned


def _extract_candidates_from_text(cleaned: str) -> List[str]:
    candidates: List[str] = []

    patterns = [
        r"[A-Z]{2,4}\d{2,4}",
        r"[A-Z]{1,3}\d{2,4}",
        r"\d{4,12}",
        r"\d{3,6}[A-Z]{1,4}",
        r"[A-Z]{1,4}\d{3,6}",
        r"[A-Z0-9]{4,12}",
    ]

    for pat in patterns:
        matches = re.findall(pat, cleaned)
        for m in matches:
            m2 = trim_numeric_plate_noise(m)
            if MIN_PLATE_TEXT_LEN <= len(m2) <= MAX_PLATE_TEXT_LEN:
                candidates.append(m2)

    return candidates


def extract_best_plate_candidate(text: str) -> str:
    raw_cleaned = clean_plate_text(text)
    if not raw_cleaned:
        return ""

    sanitized = sanitize_ocr_text_for_plate(raw_cleaned)

    # Strong preference for long numeric temporary plate
    best_digits = longest_digit_run(sanitized)
    if 7 <= len(best_digits) <= 12:
        return best_digits

    candidates = _extract_candidates_from_text(sanitized)

    # If sanitizer removed too much, fall back to raw cleaned OCR
    if not candidates:
        best_digits_raw = longest_digit_run(raw_cleaned)
        if 7 <= len(best_digits_raw) <= 12:
            return best_digits_raw
        candidates = _extract_candidates_from_text(raw_cleaned)

    if candidates:
        def rank_key(v: str):
            d = count_digits(v)
            a = count_letters(v)

            # prefer realistic mixed temp plate / short mixed plate
            if a >= 1 and d >= 2 and 4 <= len(v) <= 8:
                return (7, len(v), d, -a)
            if a == 0 and 7 <= d <= 12:
                return (6, len(v), d, 0)
            if a >= 1 and d >= 2:
                return (5, len(v), d, -a)
            if a == 0 and 4 <= d <= 6:
                return (4, len(v), d, 0)
            return (1, len(v), d, -a)

        candidates = sorted(set(candidates), key=rank_key, reverse=True)
        return candidates[0]

    if best_digits:
        return best_digits

    return ""


def length_bonus(total_len: int) -> float:
    if total_len < MIN_PLATE_TEXT_LEN:
        return -1.2
    if MIN_PLATE_TEXT_LEN <= total_len <= MAX_PLATE_TEXT_LEN:
        return 1.2
    return -1.0


def score_plate_candidate(
    text: str,
    conf: float,
    bbox: Optional[dict] = None,
    image_size: Optional[Tuple[int, int]] = None,
) -> float:
    cleaned = normalize_final_plate_text(text)
    if not cleaned:
        return -9999.0

    digit_count = count_digits(cleaned)
    letter_count = count_letters(cleaned)
    total_len = len(cleaned)

    score = float(conf)
    score += length_bonus(total_len)
    score += digit_count * 0.10
    score += letter_count * 0.12

    if digit_count >= 2 and letter_count >= 1:
        score += 0.45

    if letter_count == 0 and 7 <= digit_count <= 12:
        score += 0.70

    if digit_count >= 7 and letter_count >= 1:
        score -= 0.35 * letter_count

    # Penalize obvious noise words surviving in candidate
    upper_text = clean_plate_text(text)
    for token in NOISE_TOKENS:
        if token in upper_text:
            score -= 1.25

    if bbox is not None:
        bw = max(1, int(bbox["x2"]) - int(bbox["x1"]))
        bh = max(1, int(bbox["y2"]) - int(bbox["y1"]))
        aspect = bw / float(bh)
        area = bw * bh
        score += min(aspect, 8.0) * 0.04
        score += min(area / 5000.0, 1.0) * 0.18

        if image_size is not None:
            iw, ih = image_size
            cx = (bbox["x1"] + bbox["x2"]) / 2.0
            cy = (bbox["y1"] + bbox["y2"]) / 2.0
            dx = abs(cx - (iw / 2.0)) / max(iw, 1)
            dy = abs(cy - (ih / 2.0)) / max(ih, 1)
            score -= (dx + dy) * 0.30

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
        top = 0.18
        bottom = 0.72
    elif mode == "mid":
        top = 0.15
        bottom = 0.78
    else:
        top = 0.18
        bottom = 0.72

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
        left_ratio=0.03,
        right_ratio=0.04,
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
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            temp_path = tmp.name

        pil_img.save(temp_path, format="JPEG", quality=92, optimize=False)
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
    ocr = get_paddle_ocr()
    img_np = np.ascontiguousarray(np.array(img.convert("RGB")))
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
    plate_text = extract_best_plate_candidate(joined)

    return plate_text, conf


def _same_row(a: dict, b: dict) -> bool:
    ay = (a["bbox"]["y1"] + a["bbox"]["y2"]) / 2.0
    by = (b["bbox"]["y1"] + b["bbox"]["y2"]) / 2.0
    ah = max(1, a["bbox"]["y2"] - a["bbox"]["y1"])
    bh = max(1, b["bbox"]["y2"] - b["bbox"]["y1"])
    tol = max(ah, bh) * 0.65
    return abs(ay - by) <= tol


def _merge_bbox(items: List[dict]) -> dict:
    return {
        "x1": min(i["bbox"]["x1"] for i in items),
        "y1": min(i["bbox"]["y1"] for i in items),
        "x2": max(i["bbox"]["x2"] for i in items),
        "y2": max(i["bbox"]["y2"] for i in items),
    }


def fallback_full_image_ocr(base_full: Image.Image) -> List[dict]:
    pieces = recognize_with_paddle_boxes(base_full)
    if not pieces:
        return []

    img_w, img_h = base_full.size

    filtered = []
    for p in pieces:
        plate_text = extract_best_plate_candidate(p["text"])
        if not plate_text:
            continue
        filtered.append(
            {
                "text": p["text"],
                "plate": plate_text,
                "confidence": p["confidence"],
                "bbox": p["bbox"],
            }
        )

    if not filtered:
        return []

    filtered.sort(key=lambda p: (p["bbox"]["y1"], p["bbox"]["x1"]))
    candidates = []

    for p in filtered:
        score = score_plate_candidate(
            p["plate"],
            p["confidence"],
            bbox=p["bbox"],
            image_size=(img_w, img_h),
        )
        candidates.append(
            {
                "ocr": {
                    "text": p["plate"],
                    "confidence": p["confidence"],
                },
                "bbox": p["bbox"],
                "_score": score,
            }
        )

    n = len(filtered)
    for i in range(n):
        group = [filtered[i]]
        merged_text = filtered[i]["text"]
        merged_conf_vals = [filtered[i]["confidence"]]

        for j in range(i + 1, min(i + 3, n)):
            if not _same_row(filtered[j - 1], filtered[j]):
                break

            gap = filtered[j]["bbox"]["x1"] - filtered[j - 1]["bbox"]["x2"]
            prev_h = max(1, filtered[j - 1]["bbox"]["y2"] - filtered[j - 1]["bbox"]["y1"])
            if gap > prev_h * 2.0:
                break

            group.append(filtered[j])
            merged_text += filtered[j]["text"]
            merged_conf_vals.append(filtered[j]["confidence"])

            plate_text = extract_best_plate_candidate(merged_text)
            if not plate_text:
                continue

            merged_bbox = _merge_bbox(group)
            merged_conf = float(sum(merged_conf_vals) / len(merged_conf_vals))
            score = score_plate_candidate(
                plate_text,
                merged_conf,
                bbox=merged_bbox,
                image_size=(img_w, img_h),
            )

            candidates.append(
                {
                    "ocr": {
                        "text": plate_text,
                        "confidence": merged_conf,
                    },
                    "bbox": merged_bbox,
                    "_score": score,
                }
            )

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["_score"], reverse=True)
    best = candidates[0]

    if len(best["ocr"]["text"]) < MIN_PLATE_TEXT_LEN:
        return []

    return [
        {
            "ocr": best["ocr"],
            "bbox": best["bbox"],
        }
    ]


def _vehicle_front_plate_roi(vehicle_bbox: dict, img_w: int, img_h: int) -> Optional[Tuple[int, int, int, int]]:
    try:
        vx1 = int(vehicle_bbox["x1"])
        vy1 = int(vehicle_bbox["y1"])
        vx2 = int(vehicle_bbox["x2"])
        vy2 = int(vehicle_bbox["y2"])
    except Exception:
        return None

    vw = max(1, vx2 - vx1)
    vh = max(1, vy2 - vy1)

    x1 = vx1 + int(vw * 0.28)
    x2 = vx1 + int(vw * 0.72)
    y1 = vy1 + int(vh * 0.50)
    y2 = vy1 + int(vh * 0.82)

    return clamp_box(x1, y1, x2, y2, img_w, img_h)


def _vehicle_crop_ocr_fallback(base_full: Image.Image, vehicles: List[dict]) -> List[dict]:
    if not vehicles:
        return []

    img_w, img_h = base_full.size
    top_vehicle = vehicles[0]
    vehicle_bbox = top_vehicle.get("bbox")
    if not vehicle_bbox:
        return []

    roi = _vehicle_front_plate_roi(vehicle_bbox, img_w, img_h)
    if not roi:
        return []

    rx1, ry1, rx2, ry2 = roi
    roi_img = base_full.crop((rx1, ry1, rx2, ry2))
    roi_img = pad_image_horizontal(roi_img, left_ratio=0.03, right_ratio=0.03, top_ratio=0.03, bottom_ratio=0.03)

    pieces = recognize_with_paddle_boxes(roi_img)
    if not pieces:
        return []

    candidates = []

    for p in pieces:
        plate_text = extract_best_plate_candidate(p["text"])
        if not plate_text:
            continue

        local_bbox = p["bbox"]
        global_bbox = {
            "x1": rx1 + local_bbox["x1"],
            "y1": ry1 + local_bbox["y1"],
            "x2": rx1 + local_bbox["x2"],
            "y2": ry1 + local_bbox["y2"],
        }

        score = score_plate_candidate(
            plate_text,
            p["confidence"],
            bbox=global_bbox,
            image_size=(img_w, img_h),
        )

        candidates.append(
            {
                "ocr": {
                    "text": plate_text,
                    "confidence": p["confidence"],
                },
                "bbox": global_bbox,
                "_score": score,
            }
        )

    if not candidates:
        return []

    candidates.sort(key=lambda x: x["_score"], reverse=True)
    best = candidates[0]

    if len(best["ocr"]["text"]) < MIN_PLATE_TEXT_LEN:
        return []

    return [
        {
            "ocr": best["ocr"],
            "bbox": best["bbox"],
        }
    ]


def run_alpr(img_np: np.ndarray, detector_name: str, ocr_name: str, vehicles: Optional[List[dict]] = None) -> dict:
    """
    Faster robust flow:
    1. detector once on resized image
    2. if found -> OCR on plate crop only (2 band modes max)
    3. if detector fails and vehicle exists -> vehicle ROI OCR fallback
    4. only last resort -> full-image OCR fallback
    """
    base_full = Image.fromarray(img_np).convert("RGB")
    base_full = resize_if_needed(base_full, FULL_IMAGE_MAX_SIDE)

    detected_plates = run_plate_detector_on_pil(base_full, detector_name, ocr_name)

    if not detected_plates:
        if vehicles:
            plates = _vehicle_crop_ocr_fallback(base_full, vehicles)
            if plates:
                return {
                    "model": {
                        "detector_model": detector_name,
                        "ocr_model": "paddleocr_fast",
                    },
                    "plates": plates,
                }

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

    modes = ["tight", "mid"]
    candidates = []

    for idx, mode in enumerate(modes):
        band = crop_number_band(plate_crop, mode=mode)
        raw_text, conf = recognize_with_paddle(band)
        plate_text = extract_best_plate_candidate(raw_text)

        cand_bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
        score = score_plate_candidate(
            plate_text,
            conf,
            bbox=cand_bbox,
            image_size=(img_w, img_h),
        )

        if plate_text:
            candidates.append(
                {
                    "text": plate_text,
                    "confidence": conf,
                    "score": score,
                }
            )

        total_len = len(plate_text)

        if idx == 0 and MIN_PLATE_TEXT_LEN <= total_len <= MAX_PLATE_TEXT_LEN and conf >= FAST_ACCEPT_CONF:
            return {
                "model": {
                    "detector_model": detector_name,
                    "ocr_model": "paddleocr_fast",
                },
                "plates": [
                    {
                        "ocr": {
                            "text": plate_text,
                            "confidence": conf,
                        },
                        "bbox": bbox,
                    }
                ],
            }

        if idx == 1 and MIN_PLATE_TEXT_LEN <= total_len <= MAX_PLATE_TEXT_LEN and conf >= SECOND_ACCEPT_CONF:
            break

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

    return {
        "model": {
            "detector_model": detector_name,
            "ocr_model": "paddleocr_fast",
        },
        "plates": [
            {
                "ocr": {
                    "text": text,
                    "confidence": conf,
                },
                "bbox": bbox,
            }
        ],
    }