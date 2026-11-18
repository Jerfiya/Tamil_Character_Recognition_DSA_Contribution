"""Grad-CAM for few-shot predictions (own implementation, works for every cacheable method).

The class score is the method's episode logit of a query against the prototypes of the same
closed-set problem used by ``evaluate.py`` (identical support draw). The gradient of the
chosen class score w.r.t. the backbone's last feature map gives channel weights; the
weighted, ReLU'd map is upsampled onto the 48x48 input.

    python src/explain.py runs/<run_dir>            # 16 correct + 16 misclassified test queries

Writes runs/<run>/plots/gradcam.png and runs/<run>/test/gradcam_samples.csv (reads only the
stored predictions.csv to choose queries; no new test-set decisions are made).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import tamil_text as tt  # noqa: E402
from config import setup_logging  # noqa: E402
from runmodel import RunModel, fullset_split  # noqa: E402

log = logging.getLogger("explain")


def gradcam(rm: RunModel, support_feats: torch.Tensor, n_way: int, k_shot: int, x: torch.Tensor,
            target: int) -> np.ndarray:
    """Grad-CAM heatmap in [0, 1] (input resolution) for one normalised image ``x`` [1, 1, S, S]."""
    model, bb = rm.model, rm.model.backbone
    holder: Dict[str, torch.Tensor] = {}
    orig = bb.features

    def hooked(inp: torch.Tensor) -> torch.Tensor:
        f = orig(inp)
        f.retain_grad()
        holder["f"] = f
        return f

    bb.features = hooked
    try:
        with torch.enable_grad():
            model.zero_grad(set_to_none=True)
            qf = model.features(x)
            logits = model.episode_logits(support_feats, qf, n_way, k_shot)
            logits[0, target].backward()
    finally:
        del bb.features                        # drop the instance override -> class method again
    f, g = holder["f"].detach(), holder["f"].grad.detach()
    w = g.mean((2, 3), keepdim=True)
    cam = F.relu((w * f).sum(1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def run(run_dir: Path, n_each: int = 16, seed: int = 0) -> Path:
    rm = RunModel(run_dir, "best_val_loss.pt")
    if rm.needs_images:
        raise SystemExit("Grad-CAM needs a cacheable method (MAML adapts per episode)")
    pred = pd.read_csv(run_dir / "test" / "predictions.csv", encoding="utf-8-sig")
    e = rm.cfg.eval
    fs = fullset_split(rm.data, rm.protocol, "test", int(e.key_shot), int(e.seed) + 31)   # same draw as evaluate.py
    n, k = fs["support"].shape
    with torch.no_grad():
        sf = rm.features(fs["support"].reshape(-1))
    rng = np.random.default_rng(seed)
    ok, bad = pred[pred.correct], pred[~pred.correct]
    pick = pd.concat([ok.iloc[rng.permutation(len(ok))[:n_each]], bad.sort_values("confidence", ascending=False).head(n_each)])
    name_to_col = {rm.data.class_names[c]: j for j, c in enumerate(fs["classes"])}
    rows: List[Dict] = []
    fig, axes = plt.subplots(4, 8, figsize=(16, 9.5))
    tt.setup_matplotlib()
    for ax, (_, r) in zip(axes.ravel(), pick.iterrows()):
        idx = int(r.idx)
        x = rm.data.normalize(rm.data.batch([idx], rm.device))
        cam = gradcam(rm, sf, n, k, x, name_to_col[r.pred])
        img = rm.data.images[idx]
        ax.imshow(255 - img, cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.45)
        ax.set_xticks([]); ax.set_yticks([])
        col = (0, 120, 0) if r.correct else (200, 0, 0)
        tt.mpl_text(ax, 0.5, 1.02, f"{r.true} > {r.pred}", px=26, color=col, zoom=0.5, xycoords="axes fraction",
                    box_alignment=(0.5, 0.0))
        ax.set_xlabel(f"conf {r.confidence:.2f}", fontsize=8)
        rows.append({"idx": idx, "true": r.true, "pred": r.pred, "correct": bool(r.correct), "confidence": r.confidence,
                     "cam_peak_row": int(np.unravel_index(cam.argmax(), cam.shape)[0]),
                     "cam_peak_col": int(np.unravel_index(cam.argmax(), cam.shape)[1])})
    fig.suptitle(f"Grad-CAM ({rm.method}/{rm.cfg.model.backbone}, protocol {rm.protocol}): top 2 rows correct, "
                 f"bottom 2 rows most-confident errors (map for the PREDICTED class)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = run_dir / "plots" / "gradcam.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160); plt.close(fig)
    pd.DataFrame(rows).to_csv(run_dir / "test" / "gradcam_samples.csv", index=False, encoding="utf-8-sig")
    log.info("wrote %s", out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    run(Path(a.run_dir), a.n)


if __name__ == "__main__":
    main()
