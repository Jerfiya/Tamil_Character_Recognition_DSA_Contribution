"""Support registry: one prototype per enrolled letter, built from ``data/support_set/<label>/``.

Enrolling a letter never retrains anything: drop 1-5 (3-5 recommended) images of the letter
into ``data/support_set/<label>/`` and refresh. Prototypes are cached in
``data/support_set/prototypes.npz``.

    python src/support_registry.py init --per-class 5        # seed the support set from TRAIN images
    python src/support_registry.py refresh [--backend onnx]  # (re)build prototypes.npz
    python src/support_registry.py enroll தௌ img1.png img2.png ...
    python src/support_registry.py list

Every image goes through ``preprocessing.preprocess_path`` - the same function that built the
training cache - so deployment and training see identical inputs.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

import tamil_text as tt
from config import ROOT, force_utf8, setup_logging
from preprocessing import preprocess_gray, preprocess_path

log = logging.getLogger("registry")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
DEFAULT_MODELS = ROOT / "models"
DEFAULT_SUPPORT = ROOT / "data" / "support_set"


# --- embedding backends ----------------------------------------------------------------------
class EmbeddingModel:
    """Preprocessed uint8 images -> embeddings, with ONNX or PyTorch backend (models/deploy.json)."""

    def __init__(self, models_dir: Path = DEFAULT_MODELS, backend: str = "onnx") -> None:
        self.models_dir = Path(models_dir)
        spec_path = self.models_dir / "deploy.json"
        if not spec_path.exists():
            raise FileNotFoundError(f"{spec_path} missing - run `python src/export_onnx.py --run <run_dir>` first")
        self.spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.pre = self.spec["preprocess"]
        self.size = int(self.spec["input_size"])
        self.backend = backend
        if backend == "onnx":
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(self.models_dir / self.spec["onnx"]), providers=["CPUExecutionProvider"])
            self._run: Callable[[np.ndarray], np.ndarray] = lambda x: self.sess.run(None, {"image": x})[0]
        elif backend == "torch":
            import torch
            from fewshot import build_method
            from config import _wrap
            ck = torch.load(self.models_dir / self.spec["torch"], map_location="cpu", weights_only=False)
            cfg = _wrap(ck["config"])
            cfg["method"]["init_from"] = None
            cfg["model"]["pretrained"] = False
            net = build_method(cfg, len(ck["train_classes"]))
            net.load_state_dict(ck["state_dict"])
            net.eval()

            def run(x: np.ndarray) -> np.ndarray:
                with torch.no_grad():
                    return net.features(torch.from_numpy(x)).numpy()
            self._run = run
        else:
            raise ValueError("backend must be onnx or torch")

    def preprocess(self, gray: np.ndarray) -> np.ndarray:
        return preprocess_gray(gray, self.pre, self.size)

    def preprocess_file(self, path: Path) -> np.ndarray:
        return preprocess_path(path, self.pre, self.size)

    def embed(self, imgs_u8: np.ndarray, chunk: int = 256) -> np.ndarray:
        """uint8 [B, S, S] (already preprocessed) -> float32 [B, D]."""
        x = imgs_u8.astype(np.float32)[:, None] / 255.0
        x = (x - float(self.pre["mean"])) / float(self.pre["std"])
        return np.concatenate([self._run(x[i:i + chunk]) for i in range(0, len(x), chunk)]).astype(np.float32)

    def logits(self, q: np.ndarray, protos: np.ndarray) -> np.ndarray:
        """Class scores, the same formula as the training method's ``episode_logits``."""
        if self.spec["geometry"] == "cosine":
            qn = q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)
            pn = protos / np.maximum(np.linalg.norm(protos, axis=1, keepdims=True), 1e-12)
            return float(self.spec["scale"]) * qn @ pn.T
        d = (q ** 2).sum(1, keepdims=True) - 2 * q @ protos.T + (protos ** 2).sum(1)[None]
        return -float(self.spec["scale"]) * np.maximum(d, 0)


# --- registry --------------------------------------------------------------------------------
def list_support(support_dir: Path) -> Dict[str, List[Path]]:
    """label (NFC) -> image files. Empty folders are reported and skipped."""
    out: Dict[str, List[Path]] = {}
    if not support_dir.exists():
        return out
    for d in sorted(p for p in support_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in d.rglob("*") if f.suffix.lower() in IMG_EXTS)
        if files:
            out.setdefault(tt.nfc(d.name), []).extend(files)
        else:
            log.warning("support folder %s has no images - skipped", d.name)
    return out


def build_prototypes(model: EmbeddingModel, support_dir: Path = DEFAULT_SUPPORT,
                     out: Optional[Path] = None) -> Dict[str, np.ndarray]:
    """Embed every support image and average per letter; write prototypes.npz."""
    files = list_support(support_dir)
    if not files:
        raise FileNotFoundError(f"no support images under {support_dir} (run `support_registry.py init`)")
    labels, protos, counts, bad = [], [], [], []
    for lab, paths in files.items():
        imgs = []
        for p in paths:
            try:
                imgs.append(model.preprocess_file(p))
            except Exception as exc:
                bad.append(str(p))
                log.warning("skipping unreadable support image %s: %s", p, exc)
        if not imgs:
            continue
        e = model.embed(np.stack(imgs))
        labels.append(lab); protos.append(e.mean(0)); counts.append(len(imgs))
    reg = {"labels": np.array(labels), "prototypes": np.stack(protos).astype(np.float32), "counts": np.array(counts)}
    out = out or support_dir / "prototypes.npz"
    np.savez(out, **reg, model_run=np.array(model.spec["run"]))
    log.info("registry: %d letters, %d images -> %s", len(labels), int(sum(counts)), out)
    return reg


def load_prototypes(support_dir: Path = DEFAULT_SUPPORT) -> Dict[str, np.ndarray]:
    z = np.load(support_dir / "prototypes.npz", allow_pickle=False)
    return {k: z[k] for k in z.files}


def enroll(model: EmbeddingModel, label: str, images: Sequence[Path] | Sequence[np.ndarray],
           support_dir: Path = DEFAULT_SUPPORT) -> Dict[str, np.ndarray]:
    """Add images of ``label`` (files or grey arrays) to the support set and update the cache in place."""
    label = tt.nfc(label)
    d = support_dir / label
    d.mkdir(parents=True, exist_ok=True)
    start = len(list(d.glob("*")))
    import cv2
    for i, im in enumerate(images):
        target = d / f"enrolled_{start + i:04d}.png"
        if isinstance(im, (str, Path)):
            shutil.copyfile(im, target.with_suffix(Path(im).suffix.lower()))
        else:
            ok, buf = cv2.imencode(".png", np.asarray(im))
            buf.tofile(str(target))
    paths = sorted(f for f in d.glob("*") if f.suffix.lower() in IMG_EXTS)
    proto = model.embed(np.stack([model.preprocess_file(p) for p in paths])).mean(0)
    cache = support_dir / "prototypes.npz"
    reg = load_prototypes(support_dir) if cache.exists() else {"labels": np.array([], dtype="<U8"),
                                                              "prototypes": np.zeros((0, len(proto)), np.float32),
                                                              "counts": np.array([], dtype=int)}
    labels = [str(x) for x in reg["labels"]]
    protos, counts = list(reg["prototypes"]), list(reg["counts"])
    if label in labels:
        j = labels.index(label)
        protos[j], counts[j] = proto, len(paths)
    else:
        labels.append(label); protos.append(proto); counts.append(len(paths))
    reg = {"labels": np.array(labels), "prototypes": np.stack(protos).astype(np.float32), "counts": np.array(counts)}
    np.savez(cache, **reg, model_run=np.array(model.spec["run"]))
    log.info("enrolled %s from %d images (registry now %d letters)", label, len(paths), len(labels))
    return reg


def init_from_dataset(per_class: int, support_dir: Path = DEFAULT_SUPPORT, seed: int = 0) -> int:
    """Copy ``per_class`` Protocol-A TRAIN images per class (never val/test) into the support set.

    The dataset itself is only read. Returns the number of files copied.
    """
    import pandas as pd
    man = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv", encoding="utf-8-sig", keep_default_na=False)
    tr = man[man.split_A == "train"]
    rng = np.random.default_rng(seed)
    n = 0
    for lab, g in tr.groupby("label"):
        d = support_dir / tt.nfc(lab)
        d.mkdir(parents=True, exist_ok=True)
        pick = g.iloc[rng.permutation(len(g))[:per_class]]
        for _, r in pick.iterrows():
            src = Path(r.path)
            dst = d / f"train_{int(r.idx):05d}{src.suffix.lower()}"
            if not dst.exists():
                shutil.copyfile(src, dst)
                n += 1
    log.info("support set initialised: %d files copied (%d per class) into %s", n, per_class, support_dir)
    return n


def main() -> None:
    force_utf8()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["init", "refresh", "enroll", "list"])
    p.add_argument("label", nargs="?", help="letter to enroll")
    p.add_argument("images", nargs="*", help="image files for enroll")
    p.add_argument("--per-class", type=int, default=5)
    p.add_argument("--support-dir", default=str(DEFAULT_SUPPORT))
    p.add_argument("--models", default=str(DEFAULT_MODELS))
    p.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    sd = Path(a.support_dir)
    if a.command == "init":
        init_from_dataset(a.per_class, sd)
        build_prototypes(EmbeddingModel(Path(a.models), a.backend), sd)
    elif a.command == "refresh":
        build_prototypes(EmbeddingModel(Path(a.models), a.backend), sd)
    elif a.command == "enroll":
        if not a.label or not a.images:
            p.error("enroll needs a label and at least one image")
        enroll(EmbeddingModel(Path(a.models), a.backend), a.label, [Path(x) for x in a.images], sd)
    else:
        reg = load_prototypes(sd)
        for lab, c in zip(reg["labels"], reg["counts"]):
            log.info("%s\t%d images", lab, c)
        log.info("%d letters enrolled", len(reg["labels"]))


if __name__ == "__main__":
    main()
