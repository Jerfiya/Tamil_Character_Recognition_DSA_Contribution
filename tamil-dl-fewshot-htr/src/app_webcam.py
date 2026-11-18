"""Webcam (or still image) recogniser: finds handwritten characters by contours and labels them.

Each detected character is cropped, passed through the shared preprocessing and the few-shot
recogniser; the Tamil label is drawn with Pillow (cv2.putText cannot render Tamil).

    python src/app_webcam.py                    # camera 0, press q to quit, s to save a frame
    python src/app_webcam.py --image page.png   # run once on a photo, write *_annotated.png
If no camera can be opened the script says so and exits cleanly.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image

import tamil_text as tt
from config import force_utf8, setup_logging
from predict import Prediction, Recognizer
from preprocessing import read_image_gray
from support_registry import DEFAULT_MODELS, DEFAULT_SUPPORT

log = logging.getLogger("app_webcam")
Box = Tuple[int, int, int, int]


def detect_characters(gray: np.ndarray, min_area_frac: float = 0.0015, merge_px: int = 0) -> List[Box]:
    """Bounding boxes of ink blobs (dark ink on light paper or the reverse), merged per character.

    Tamil letters have detached parts (dots, vowel signs), so ink is dilated before contours
    are taken; boxes are returned left-to-right, top-to-bottom.
    """
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if (mask > 0).mean() > 0.5:
        mask = 255 - mask
    h, w = gray.shape
    k = merge_px or max(3, int(min(h, w) * 0.012))
    blob = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)), iterations=2)
    cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, bw, bh = cv2.boundingRect(c)
        if bw * bh < min_area_frac * h * w or bw > 0.95 * w or bh > 0.95 * h:
            continue
        boxes.append((x, y, bw, bh))
    return sorted(boxes, key=lambda b: (b[1] // max(1, h // 8), b[0]))


def annotate(frame_bgr: np.ndarray, rec: Recognizer) -> Tuple[np.ndarray, List[Tuple[Box, Prediction]]]:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
    boxes = detect_characters(gray)
    results = []
    if boxes:
        crops = []
        for x, y, bw, bh in boxes:
            pad = int(0.1 * max(bw, bh))
            crops.append(rec.model.preprocess(gray[max(0, y - pad):y + bh + pad, max(0, x - pad):x + bw + pad]))
        results = list(zip(boxes, rec.predict_u8(np.stack(crops))))
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if frame_bgr.ndim == 2 else frame_bgr.copy()
    for (x, y, bw, bh), pr in results:
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), (0, 180, 0) if pr.known else (0, 0, 255), 2)
    pil = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
    for (x, y, bw, bh), pr in results:
        text = pr.label if pr.known else "?"
        px = max(18, min(48, bh // 2))
        tt.draw_text_on_image(pil, (x, max(0, y - px - 6)), text, px, (0, 160, 0) if pr.known else (220, 0, 0))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), results


def run_image(path: Path, rec: Recognizer) -> Path:
    gray = read_image_gray(path)
    vis, res = annotate(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), rec)
    out = path.with_name(path.stem + "_annotated.png")
    cv2.imencode(".png", vis)[1].tofile(str(out))
    log.info("%d characters detected: %s -> %s", len(res), " ".join(p.label for _, p in res), out)
    return out


def run_camera(rec: Recognizer, index: int = 0) -> int:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
    if not cap.isOpened():
        log.error("No camera available (index %d). Use --image <file> to run on a photo instead.", index)
        return 1
    log.info("camera opened - press q to quit, s to save the annotated frame")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                log.error("camera stopped delivering frames")
                break
            vis, _ = annotate(frame, rec)
            cv2.imshow("Tamil few-shot recogniser", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                cv2.imencode(".png", vis)[1].tofile(f"webcam_{n:03d}.png"); n += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()
    return 0


def main() -> None:
    force_utf8()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default=None)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--support-dir", default=str(DEFAULT_SUPPORT))
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    rec = Recognizer(Path(a.models), Path(a.support_dir), a.backend)
    if a.image:
        run_image(Path(a.image), rec)
    else:
        raise SystemExit(run_camera(rec, a.camera))


if __name__ == "__main__":
    main()
