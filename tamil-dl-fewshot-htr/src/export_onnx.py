"""Export the deployable embedding network of a run to ONNX and verify it against PyTorch.

The exported graph maps a normalised grey image batch [B, 1, S, S] to the per-image
embedding used for prototypes (``model.features``). Everything a deployment needs besides the
weights (preprocessing settings, scoring geometry, temperature, open-set threshold) is written
to ``models/deploy.json`` so ``predict.py`` / the apps never import the training config.

    python src/export_onnx.py --run runs/<run_dir> [--ckpt best_val_loss.pt]

Writes models/embedder.onnx, models/embedder.pt (TorchScript-free state copy), models/deploy.json
and reports/onnx_export.json (max |diff|, latency PyTorch vs onnxruntime).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from calibrate import load_or_calibrate
from config import ROOT, setup_logging, to_plain
from runmodel import RunModel

log = logging.getLogger("export_onnx")


class FeatureWrapper(nn.Module):
    """Exposes ``method.features`` (embedding, normalised where the method normalises)."""

    def __init__(self, method: nn.Module) -> None:
        super().__init__()
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.method.features(x)


def scoring_spec(rm: RunModel) -> Dict[str, Any]:
    """How to turn (query embedding, prototypes) into logits, mirroring ``episode_logits``."""
    m = rm.model
    if rm.method == "protonet":
        if getattr(m, "distance", "euclidean") == "cosine":
            return {"geometry": "cosine", "scale": float(m.temperature)}
        return {"geometry": "euclidean", "scale": 16.0 * float(m.temperature) / rm.embedding_dim()}
    if rm.method in ("arcface", "baselinepp"):
        return {"geometry": "cosine", "scale": float(getattr(m, "s", 10.0))}
    if rm.method == "supervised":
        return {"geometry": "euclidean", "scale": 16.0 / rm.embedding_dim()}
    raise ValueError(f"method {rm.method} is not deployable as a prototype embedder (use protonet/arcface/baselinepp)")


def export(run_dir: Path, ckpt: str, out_dir: Path, opset: int = 17) -> Dict[str, Any]:
    rm = RunModel(run_dir, ckpt, threads=4)
    spec = scoring_spec(rm)
    cal = load_or_calibrate(rm)
    rm.model.to("cpu").eval()
    wrapper = FeatureWrapper(rm.model).eval()
    size = int(rm.data.images.shape[-1])
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "embedder.onnx"
    dummy = torch.zeros(1, 1, size, size)
    torch.onnx.export(wrapper, dummy, str(onnx_path), input_names=["image"], output_names=["embedding"],
                      dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=opset,
                      dynamo=False)
    import onnx
    import onnxruntime as ort
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    idx = np.random.default_rng(0).choice(len(rm.data.labels), 256, replace=False)
    x = rm.data.normalize(rm.data.batch(idx, torch.device("cpu")))
    with torch.no_grad():
        ref = wrapper(x).numpy()
    got = sess.run(None, {"image": x.numpy()})[0]
    max_diff = float(np.abs(ref - got).max())
    rel = float(np.abs(ref - got).max() / max(np.abs(ref).max(), 1e-8))
    cos = float((ref * got).sum(1).mean() / (np.linalg.norm(ref, axis=1) * np.linalg.norm(got, axis=1)).mean())

    def bench(fn, n: int = 200) -> float:
        for i in range(10):
            fn(x[i:i + 1])
        t0 = time.perf_counter()
        for i in range(n):
            fn(x[i % len(x):i % len(x) + 1])
        return 1000 * (time.perf_counter() - t0) / n

    with torch.no_grad():
        ms_torch = bench(lambda b: wrapper(b))
    ms_onnx = bench(lambda b: sess.run(None, {"image": b.numpy()}))
    torch.save({"state_dict": rm.model.state_dict(), "config": to_plain(rm.cfg), "train_classes": rm.train_classes},
               out_dir / "embedder.pt")
    deploy = {"run": run_dir.name, "run_dir": str(run_dir), "checkpoint": ckpt, "checkpoint_epoch": rm.ckpt_epoch,
              "method": rm.method, "backbone": str(rm.cfg.model.backbone), "input_size": size,
              "embedding_dim": rm.embedding_dim(), "preprocess": to_plain(rm.cfg.preprocess), **spec,
              "calibration_temperature": float(cal["temperature"]),
              "openset_threshold_max_logit": float(cal["openset_threshold"]),
              "openset_threshold_note": f"max-logit threshold accepting {cal['openset_target_tpr']:.0%} of enrolled val "
                                        f"queries at K={cal['k_shot']} (validation only)",
              "onnx": onnx_path.name, "torch": "embedder.pt"}
    (out_dir / "deploy.json").write_text(json.dumps(deploy, indent=2, ensure_ascii=False), encoding="utf-8")
    report = {"run": run_dir.name, "checkpoint": ckpt, "onnx_path": str(onnx_path), "opset": opset,
              "samples_compared": int(len(idx)), "max_abs_diff": max_diff, "max_rel_diff": rel, "mean_cosine": cos,
              "parity_ok": bool(max_diff < 1e-3 or rel < 1e-4), "torch_cpu_ms_per_image": ms_torch,
              "onnxruntime_cpu_ms_per_image": ms_onnx, "onnx_size_kb": onnx_path.stat().st_size / 1024}
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "onnx_export.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("ONNX export: max|diff|=%.2e (rel %.2e) parity_ok=%s; latency torch %.2f ms vs onnxruntime %.2f ms",
             max_diff, rel, report["parity_ok"], ms_torch, ms_onnx)
    if not report["parity_ok"]:
        raise SystemExit("ONNX outputs do not match PyTorch")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="run directory to export")
    p.add_argument("--ckpt", default="best_val_loss.pt")
    p.add_argument("--out", default=str(ROOT / "models"))
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    export(Path(a.run), a.ckpt, Path(a.out))


if __name__ == "__main__":
    main()
