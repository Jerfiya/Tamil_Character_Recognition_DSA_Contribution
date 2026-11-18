"""Recognise handwritten Tamil letters in an image or a folder of images.

Output per image: top-5 letters with temperature-calibrated confidence, or "unknown" when the
best score is below the open-set threshold calibrated on validation data.

    python src/predict.py path/to/image.png
    python src/predict.py path/to/folder --backend torch --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from config import ROOT, force_utf8, setup_logging
from support_registry import DEFAULT_MODELS, DEFAULT_SUPPORT, IMG_EXTS, EmbeddingModel, load_prototypes

log = logging.getLogger("predict")


@dataclass
class Prediction:
    label: str                       # best letter or "unknown"
    top: List[Tuple[str, float]]     # top-k (letter, calibrated confidence)
    score: float                     # max logit (open-set score)
    known: bool


class Recognizer:
    """Registry prototypes + embedding model + calibration = a ready-to-use classifier."""

    def __init__(self, models_dir: Path = DEFAULT_MODELS, support_dir: Path = DEFAULT_SUPPORT,
                 backend: str = "onnx", top_k: int = 5, threshold: Optional[float] = None) -> None:
        self.model = EmbeddingModel(models_dir, backend)
        self.support_dir = Path(support_dir)
        self.top_k = top_k
        self.temp = float(self.model.spec["calibration_temperature"])
        self.threshold = float(self.model.spec["openset_threshold_max_logit"] if threshold is None else threshold)
        self.reload()

    def reload(self) -> None:
        reg = load_prototypes(self.support_dir)
        self.labels = [str(x) for x in reg["labels"]]
        self.protos = reg["prototypes"]

    def predict_u8(self, imgs_u8: np.ndarray) -> List[Prediction]:
        """Preprocessed uint8 images [B, S, S] -> predictions."""
        lg = self.model.logits(self.model.embed(imgs_u8), self.protos)
        z = lg / self.temp
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        out = []
        for i in range(len(lg)):
            order = np.argsort(-p[i])[: self.top_k]
            top = [(self.labels[j], float(p[i, j])) for j in order]
            score = float(lg[i].max())
            known = score >= self.threshold and imgs_u8[i].max() > 0
            out.append(Prediction(top[0][0] if known else "unknown", top, score, known))
        return out

    def predict_gray(self, gray: np.ndarray) -> Prediction:
        return self.predict_u8(self.model.preprocess(gray)[None])[0]

    def predict_files(self, paths: List[Path]) -> List[Prediction]:
        return self.predict_u8(np.stack([self.model.preprocess_file(p) for p in paths]))


def main() -> None:
    force_utf8()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", help="image file or folder")
    p.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--support-dir", default=str(DEFAULT_SUPPORT))
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--threshold", type=float, default=None, help="override the calibrated open-set threshold")
    p.add_argument("--csv", default=None, help="also write results to this CSV")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    src = Path(a.input)
    paths = sorted(f for f in src.rglob("*") if f.suffix.lower() in IMG_EXTS) if src.is_dir() else [src]
    if not paths:
        raise SystemExit(f"no images found at {src}")
    rec = Recognizer(Path(a.models), Path(a.support_dir), a.backend, a.top_k, a.threshold)
    preds = rec.predict_files(paths)
    rows = []
    for path, pr in zip(paths, preds):
        tops = "  ".join(f"{lab} {c:.2f}" for lab, c in pr.top)
        log.info("%s -> %s (score %.2f) | %s", path.name, pr.label, pr.score, tops)
        rows.append({"path": str(path), "prediction": pr.label, "known": pr.known, "score": pr.score,
                     **{f"top{i + 1}": lab for i, (lab, _) in enumerate(pr.top)},
                     **{f"conf{i + 1}": round(c, 4) for i, (_, c) in enumerate(pr.top)}})
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        log.info("wrote %s", a.csv)


if __name__ == "__main__":
    main()
