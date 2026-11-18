"""Enrolment without retraining: evidence on real data plus the missing-letter demo.

Part 1 (real handwriting): for each Protocol-B NOVEL letter in --letters (never seen by the model in
training), remove it from a copy of the registry, measure how its TEST images are handled
(rejected as unknown / confused), enrol it from 5 TRAIN images, and measure top-1/top-5 again.
Part 2 (missing letter தௌ): no handwritten samples exist, so 5 font-rendered + augmented glyphs are
enrolled (clearly a demo) and 5 other renders are recognised.

    python src/demo_enroll.py [--letters லீ ரே மி] [--backend onnx]

Writes reports/enrollment_demo.json and reports/figures/enrollment_demo.png.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

import tamil_text as tt
from augment import Augmenter
from config import ROOT, force_utf8, load_config, setup_logging
from predict import Recognizer
from support_registry import DEFAULT_MODELS, DEFAULT_SUPPORT, build_prototypes, enroll

log = logging.getLogger("demo_enroll")


def synthetic_glyphs(text: str, n: int, seed: int) -> List[np.ndarray]:
    """Font-rendered letter, augmented (rotation/elastic/stroke) and returned as dark-on-light grey images."""
    from PIL import Image
    cfg = load_config()
    aug = Augmenter(cfg.augment, seed=seed)
    out = []
    for i in range(n):
        px = 60 + 6 * (i % 4)
        patch = tt.render_text(text, px, (255, 255, 255))
        canvas = Image.new("L", (128, 128), 0)
        canvas.paste(patch.split()[-1], ((128 - patch.width) // 2, (128 - patch.height) // 2))
        x = torch.from_numpy(np.asarray(canvas, np.float32) / 255.0)[None, None]
        x = aug(x)[0, 0].clamp(0, 1).numpy()
        out.append((255 - 255 * x).astype(np.uint8))
    return out


def run(letters: List[str], backend: str) -> Dict:
    man = pd.read_csv(ROOT / "data" / "processed" / "manifest.csv", encoding="utf-8-sig", keep_default_na=False)
    split = json.loads((ROOT / "data" / "processed" / "class_split.json").read_text(encoding="utf-8"))
    novel = set(split["novel"])
    tmp = Path(tempfile.mkdtemp(prefix="enroll_demo_"))
    sup = tmp / "support_set"
    shutil.copytree(DEFAULT_SUPPORT, sup, ignore=shutil.ignore_patterns("prototypes.npz"))
    for lab in letters:
        if lab not in novel:
            log.warning("%s is not a Protocol-B novel letter; the model saw it in training", lab)
        shutil.rmtree(sup / lab, ignore_errors=True)
    rec0 = None
    res: Dict = {"letters": {}, "note": "Part 1 uses real handwriting of novel letters; Part 2 (தௌ) uses font-rendered glyphs."}
    from support_registry import EmbeddingModel
    model = EmbeddingModel(DEFAULT_MODELS, backend)
    build_prototypes(model, sup)
    rec0 = Recognizer(DEFAULT_MODELS, sup, backend)
    for lab in letters:
        g = man[man.label == lab]
        test = [Path(p) for p in g[g.split_A == "test"].path] + [Path(p) for p in g[g.split_A == "val"].path]
        train = [Path(p) for p in g[g.split_A == "train"].path][:5]
        before = rec0.predict_files(test)
        enroll(model, lab, train, sup)
        rec0.reload()
        after = rec0.predict_files(test)
        res["letters"][lab] = {
            "enrolled_from_train_images": len(train), "evaluated_images (val+test pools)": len(test),
            "before_rejected_as_unknown": float(np.mean([not p.known for p in before])),
            "before_top_predictions": pd.Series([p.top[0][0] for p in before]).value_counts().head(3).to_dict(),
            "after_top1": float(np.mean([p.top[0][0] == lab for p in after])),
            "after_top5": float(np.mean([lab in [t for t, _ in p.top] for p in after])),
            "after_accepted_and_correct": float(np.mean([p.known and p.label == lab for p in after]))}
        log.info("%s: before enrolment %.0f%% rejected as unknown; after: top-1 %.0f%%, top-5 %.0f%%", lab,
                 100 * res["letters"][lab]["before_rejected_as_unknown"], 100 * res["letters"][lab]["after_top1"],
                 100 * res["letters"][lab]["after_top5"])
    # Part 2: the real registry gains தௌ (font-rendered demo images)
    missing = "தௌ"
    enr = synthetic_glyphs(missing, 5, seed=1)
    probe = synthetic_glyphs(missing, 5, seed=99)
    rec = Recognizer(DEFAULT_MODELS, DEFAULT_SUPPORT, backend)
    before = [rec.predict_gray(g) for g in probe]
    enroll(model, missing, enr, DEFAULT_SUPPORT)
    rec.reload()
    after = [rec.predict_gray(g) for g in probe]
    res["missing_letter"] = {"label": missing, "enrolled_images": 5, "source": "font-rendered + augmented (demo)",
                             "before": [p.label for p in before], "after": [p.label for p in after],
                             "after_top1": float(np.mean([p.top[0][0] == missing for p in after])),
                             "registry_size_after": len(rec.labels)}
    log.info("%s enrolled; probes predicted as %s", missing, [p.label for p in after])
    out = ROOT / "reports" / "enrollment_demo.json"
    out.write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    _figure(enr, probe, after)
    shutil.rmtree(tmp, ignore_errors=True)
    return res


def _figure(enr, probe, after) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tt.setup_matplotlib()
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.6))
    for ax, g in zip(axes[0], enr):
        ax.imshow(g, cmap="gray"); ax.axis("off")
    for ax, g, p in zip(axes[1], probe, after):
        ax.imshow(g, cmap="gray"); ax.axis("off")
        tt.mpl_text(ax, 0.5, -0.05, f"{p.label} {p.top[0][1]:.2f}", px=26, zoom=0.5, xycoords="axes fraction",
                    box_alignment=(0.5, 1.0), color=(0, 120, 0) if p.label == "தௌ" else (200, 0, 0))
    axes[0, 0].set_title("enrolled (5 images)", loc="left", fontsize=9)
    axes[1, 0].set_title("new probes -> prediction", loc="left", fontsize=9)
    fig.suptitle("Enrolling the missing letter without retraining (font-rendered demo glyphs)", fontsize=10)
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports" / "figures" / "enrollment_demo.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    force_utf8()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--letters", nargs="+", default=None, help="novel letters to hold out and re-enrol")
    p.add_argument("--backend", default="onnx", choices=["onnx", "torch"])
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    split = json.loads((ROOT / "data" / "processed" / "class_split.json").read_text(encoding="utf-8"))
    letters = a.letters or split["novel"][:6]
    run([tt.nfc(x) for x in letters], a.backend)


if __name__ == "__main__":
    main()
