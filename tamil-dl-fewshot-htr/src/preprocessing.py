"""Image preprocessing shared by training AND inference (single source of truth).

Pipeline: read (unicode-safe, alpha on white) -> grayscale -> light denoise -> polarity
normalisation (ink white on black) -> Otsu/adaptive threshold (optional soft variant) ->
crop to the ink bounding box -> pad to square -> resize -> uint8.
``to_tensor`` then scales to [0, 1] and normalises.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Optional

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger(__name__)

DEFAULTS = {"size": 48, "margin_frac": 0.08, "denoise": True, "threshold": "otsu", "soft": True,
            "min_ink_frac": 0.002, "mean": 0.5, "std": 0.5}


def _opt(cfg: Optional[Mapping[str, Any]], key: str) -> Any:
    return cfg[key] if cfg is not None and key in cfg else DEFAULTS[key]


def composite_to_gray(arr: np.ndarray) -> np.ndarray:
    """Convert a decoded image (any channel count / depth) to uint8 grayscale, alpha on white."""
    if arr.dtype == np.uint16:
        arr = (arr / 257.0).astype(np.uint8)
    elif arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return arr
    if arr.shape[2] == 4:
        alpha = arr[:, :, 3:4].astype(np.float32) / 255.0
        rgb = arr[:, :, :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
        return cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    if arr.shape[2] == 2:  # gray + alpha
        alpha = arr[:, :, 1].astype(np.float32) / 255.0
        return (arr[:, :, 0].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)).astype(np.uint8)
    if arr.shape[2] == 1:
        return arr[:, :, 0]
    return cv2.cvtColor(arr[:, :, :3], cv2.COLOR_BGR2GRAY)


def read_image_gray(path: str | Path) -> np.ndarray:
    """Read an image from a (possibly non-ASCII) path as uint8 grayscale.

    Uses ``np.fromfile`` + ``cv2.imdecode`` because ``cv2.imread`` fails on Unicode paths on
    Windows; falls back to Pillow. Raises ``ValueError`` for unreadable files.
    """
    path = Path(path)
    try:
        buf = np.fromfile(str(path), dtype=np.uint8)
    except OSError as exc:
        raise ValueError(f"cannot read file: {path}: {exc}") from exc
    if buf.size == 0:
        raise ValueError(f"empty file: {path}")
    arr = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    if arr is None:
        try:
            with Image.open(path) as im:
                im.load()
                arr = np.asarray(im.convert("RGBA"))[:, :, [2, 1, 0, 3]]
        except Exception as exc:
            raise ValueError(f"corrupt or unsupported image: {path}: {exc}") from exc
    if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ValueError(f"degenerate image: {path}")
    return composite_to_gray(arr)


def ink_is_dark(gray: np.ndarray) -> bool:
    """Decide polarity: True when the background is bright (dark ink on light paper).

    The border ring is mostly background; if it is ambiguous the global majority decides.
    """
    h, w = gray.shape
    b = max(1, min(h, w) // 16)
    border = np.concatenate([gray[:b].ravel(), gray[-b:].ravel(), gray[:, :b].ravel(), gray[:, -b:].ravel()])
    thr, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    frac_bright_border = float((border > thr).mean())
    if 0.35 < frac_bright_border < 0.65:
        return float((gray > thr).mean()) >= 0.5
    return frac_bright_border >= 0.5


def remove_specks(mask: np.ndarray) -> np.ndarray:
    """Drop noise components so they cannot stretch the ink bounding box.

    Removed: components of 1-2 pixels (scaled with resolution), and components that touch the
    image border while holding under 10% of the ink (ruled lines, page edges, corner shadows).
    Real diacritics such as the pulli dot are larger and/or interior, so they are kept.
    """
    n, lab, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), connectivity=8)
    if n <= 2:
        return mask
    h, w = mask.shape
    areas = stats[1:, cv2.CC_STAT_AREA]
    total = float(areas.sum())
    tiny = max(2, int(round(h * w / 1200.0)))
    keep = np.ones(n, bool)
    keep[0] = False
    for i in range(1, n):
        x, y, bw, bh, a = stats[i]
        on_border = x <= 0 or y <= 0 or x + bw >= w or y + bh >= h
        if a <= tiny or (on_border and a < 0.10 * total):
            keep[i] = False
    if not keep[1:].any():
        return mask
    return np.where(keep[lab], 255, 0).astype(np.uint8)


def preprocess_gray(gray: np.ndarray, cfg: Optional[Mapping[str, Any]] = None,
                    size: Optional[int] = None) -> np.ndarray:
    """Normalise a grayscale image to a ``size`` x ``size`` uint8 array, ink white on black."""
    size = int(size or _opt(cfg, "size"))
    g = gray
    if g.dtype != np.uint8:
        g = np.clip(g, 0, 255).astype(np.uint8)
    if _opt(cfg, "denoise") and min(g.shape) >= 48:
        g = cv2.medianBlur(g, 3)
    if ink_is_dark(g):
        g = 255 - g                                    # ink is now bright
    if _opt(cfg, "threshold") == "adaptive" and min(g.shape) >= 32:
        block = max(15, (min(g.shape) // 8) | 1)
        mask = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block, -8)
    else:
        _, mask = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if g.max() - g.min() < 16:                         # flat image: nothing to segment
        mask = np.zeros_like(g)
    mask = remove_specks(mask)
    if _opt(cfg, "soft"):
        lo = float(g[mask == 0].mean()) if (mask == 0).any() else 0.0
        hi = float(g[mask > 0].mean()) if (mask > 0).any() else 255.0
        soft = np.clip((g.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1) * 255.0
        ink = np.where(cv2.dilate(mask, np.ones((3, 3), np.uint8)) > 0, soft, 0).astype(np.uint8)
    else:
        ink = mask
    ys, xs = np.where(mask > 0)
    if ys.size < _opt(cfg, "min_ink_frac") * mask.size or ys.size == 0:
        return np.zeros((size, size), np.uint8)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = ink[y0:y1, x0:x1]
    h, w = crop.shape
    inner = max(8, int(round(size * (1.0 - 2.0 * float(_opt(cfg, "margin_frac"))))))
    scale = inner / max(h, w)
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (nw, nh), interpolation=interp)
    out = np.zeros((size, size), np.uint8)
    oy, ox = (size - nh) // 2, (size - nw) // 2
    out[oy:oy + nh, ox:ox + nw] = resized
    return out


def preprocess_path(path: str | Path, cfg: Optional[Mapping[str, Any]] = None,
                    size: Optional[int] = None) -> np.ndarray:
    """Read and preprocess one image file."""
    return preprocess_gray(read_image_gray(path), cfg, size)


def to_tensor(img_u8: np.ndarray, cfg: Optional[Mapping[str, Any]] = None) -> np.ndarray:
    """uint8 (..., H, W) -> float32 (..., 1, H, W), scaled to [0, 1] then normalised."""
    x = img_u8.astype(np.float32) / 255.0
    x = (x - float(_opt(cfg, "mean"))) / float(_opt(cfg, "std"))
    return x[..., None, :, :]


def normalize_batch(x01, cfg: Optional[Mapping[str, Any]] = None):
    """Normalise a torch batch that is already scaled to [0, 1]."""
    return (x01 - float(_opt(cfg, "mean"))) / float(_opt(cfg, "std"))
