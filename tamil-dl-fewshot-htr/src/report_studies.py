"""Write reports/epoch_study.md and reports/ablations.md from stored VALIDATION metrics only.

    python src/report_studies.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import ROOT, parse_run_name, setup_logging

log = logging.getLogger("studies")
RUNS, REPORTS = ROOT / "runs", ROOT / "reports"

ABL_DESC = {"lr1e-3": "baseline: 20-way 5-shot, lr 1e-3, emb 128, Euclidean, EMA, 48 px, Conv4",
            "lr3e-3": "lr 3e-3", "lr3e-4": "lr 3e-4", "lr1e-4": "lr 1e-4",
            "abl_way5": "train 5-way", "abl_way10": "train 10-way", "abl_way30": "train 30-way",
            "abl_shot1": "train 1-shot", "abl_shotmix": "train mixed shots {1,2,3,5}",
            "abl_emb64": "embedding dim 64", "abl_emb256": "embedding dim 256", "abl_cosine": "cosine distance",
            "abl_noema": "EMA off", "abl_in64": "input 64 px (upsampled)", "abl_resnet12": "ResNet-12 (width 16), scratch",
            "abl_mnv3_pre96": "MobileNetV3-L ImageNet-pretrained, 96 px", "abl_mnv3_scr96": "MobileNetV3-L from scratch, 96 px",
            "abl_resnet18": "ResNet-18 (width 16), scratch", "abl_effb0_pre96": "EfficientNet-B0 ImageNet-pretrained, 96 px"}


def tag(rd: Path) -> str:
    return parse_run_name(rd.name)["tag"]


def load(rd: Path):
    df = pd.read_csv(rd / "metrics_epoch.csv")
    h = json.loads((rd / "history.json").read_text(encoding="utf-8")) if (rd / "history.json").exists() else {}
    env = json.loads((rd / "env.json").read_text(encoding="utf-8")) if (rd / "env.json").exists() else {}
    return df, h, env


def epoch_study() -> Optional[str]:
    runs = sorted(r for r in RUNS.glob("*_epochstudy") if (r / "metrics_epoch.csv").exists())
    if not runs:
        return None
    rd = runs[-1]
    df, h, _ = load(rd)
    v = df.val_loss.to_numpy()
    smooth = pd.Series(v).rolling(3, center=True, min_periods=1).mean().to_numpy()
    e_min = int(df.epoch.iat[int(np.argmin(v))])
    e_smin = int(df.epoch.iat[int(np.argmin(smooth))])
    within = lambda tol: int(df.epoch.iat[int(np.argmax(smooth <= smooth.min() * (1 + tol)))])  # noqa: E731
    e1, e2, e5 = within(0.01), within(0.02), within(0.05)
    acc = df.val_episode_acc.to_numpy()
    rows = []
    for e in sorted({5, 10, 15, 20, 25, 30, 35, 40, e_min} & set(df.epoch)):
        r = df[df.epoch == e].iloc[0]
        rows.append(f"| {e} | {r.train_loss:.4f} | {r.val_loss:.4f} | {100 * r.val_episode_acc:.2f} ± {100 * r.val_acc_ci95:.2f} | "
                    f"{r.lr:.2e} | {int(r.overfit_flag)} |")
    md = [f"# Epoch-budget study (ProtoNet, Conv4, Protocol A, validation only)", "",
          f"Run: `{rd.name}`: {len(df)} epochs × 50 episodes, cosine schedule over the whole budget, **early stopping off**. "
          "Validation is a fixed list of 100 10-way 5-shot episodes (no augmentation), so the val curve is comparable epoch to epoch.",
          "", "![epoch study](figures/epoch_study.png)", "",
          "| epoch | train loss | val loss | val acc % (±95% CI) | lr | overfit flag |", "|---|---|---|---|---|---|", *rows, "",
          "## Reading the curve", "",
          f"- Raw val-loss minimum: epoch **{e_min}** ({v.min():.4f}); 3-epoch-smoothed minimum: epoch **{e_smin}**.",
          f"- Smoothed val loss first comes within 1% / 2% / 5% of its minimum at epochs **{e1} / {e2} / {e5}**.",
          f"- Best val accuracy {100 * acc.max():.2f}% at epoch {int(df.epoch.iat[int(np.argmax(acc))])}; "
          f"at epoch 15 it is {100 * acc[min(14, len(acc) - 1)]:.2f}%.",
          f"- Overfitting flags (train ↓ while val ↑ for 3 epochs): {h.get('overfit_epochs') or 'none'}. "
          f"Plateau flags: {len(h.get('plateau_epochs', []))} epochs.",
          "- Train loss stays above val loss throughout because training episodes are augmented and 20-way, "
          "while validation is clean 10-way. The gap is therefore not a sign of underfitting.", "",
          "## Decision", ""]
    if e5 <= 15:
        md.append(f"The curve gets within 5% of its minimum by epoch {e5}. Most of the gain arrives by then. A cosine schedule "
                  "**shortened to the budget** also anneals sooner, so a 15-epoch run reaches its low-LR phase at epoch 15 and does "
                  "better than epoch 15 of this 40-epoch run. **Budget: 15 epochs** (the final runs; their val loss still falls "
                  "slowly at the end, so 15 is a CPU-time compromise, not a fully converged optimum). With a GPU, use the "
                  f"smoothed-minimum epoch ({e_smin}) plus a 20% margin, with early stopping on val loss as the safety net.")
    else:
        md.append(f"The val loss is still improving well past epoch 15 (within 5% of the minimum only at epoch {e5}). "
                  f"15 epochs under-trains. Recommended budget: ~{int(round(e_smin * 1.2))} epochs (smoothed minimum + 20% margin) "
                  "with early stopping on val loss. The 15-epoch final runs are a CPU-time compromise and are reported as such.")
    md += ["", "All final runs keep early stopping on val loss (patience 10) and checkpoint `best_val_loss.pt`, so a too-generous "
           "budget cannot hurt the selected model."]
    return "\n".join(md) + "\n"


def ablations() -> Optional[str]:
    runs = [r for r in sorted(RUNS.iterdir()) if (tag(r).startswith(("abl", "lr"))) and (r / "metrics_epoch.csv").exists()]
    if not runs:
        return None
    rows: List[Dict] = []
    for rd in runs:
        df, h, env = load(rd)
        if df.empty:
            continue
        rows.append({"tag": tag(rd), "setting": ABL_DESC.get(tag(rd), tag(rd)), "epochs": len(df),
                     "best_val_loss": df.val_loss.min(), "best_val_loss_epoch": int(df.epoch.iat[int(df.val_loss.idxmin())]),
                     "best_val_acc": df.val_episode_acc.max(), "ci": float(df.val_acc_ci95.iat[int(df.val_episode_acc.idxmax())]),
                     "final_train_loss": df.train_loss.iat[-1], "params": env.get("parameter_count"),
                     "sec_per_epoch": df.epoch_time_s.median(), "pretrained_loaded": "pre96" in tag(rd),
                     "status": h.get("status")})
    if not rows:
        return None
    t = pd.DataFrame(rows)
    t.to_csv(REPORTS / "ablations.csv", index=False)
    base = t[t.tag == "lr1e-3"]
    b_loss = float(base.best_val_loss.iat[0]) if len(base) else np.nan
    b_acc = float(base.best_val_acc.iat[0]) if len(base) else np.nan

    def table(sel: pd.DataFrame) -> List[str]:
        out = ["| setting | best val loss (epoch) | Δ loss vs baseline | best val acc % (±CI) | Δ acc (pp) | params | s/epoch |",
               "|---|---|---|---|---|---|---|"]
        for _, r in sel.sort_values("best_val_loss").iterrows():
            out.append(f"| {r.setting}{'' if r.status == 'finished' else ' *(' + str(r.status) + ')*'} | {r.best_val_loss:.4f} "
                       f"({r.best_val_loss_epoch}/{r.epochs}) | {r.best_val_loss - b_loss:+.4f} | {100 * r.best_val_acc:.2f} ± {100 * r.ci:.2f} | "
                       f"{100 * (r.best_val_acc - b_acc):+.2f} | {r.params / 1e3:.0f}k | {r.sec_per_epoch:.0f} |")
        return out
    groups = [("Learning rate (judged by val-loss curves; see figures/lr_comparison.png)", ["lr1e-3", "lr3e-3", "lr3e-4", "lr1e-4"]),
              ("Train-time way", ["lr1e-3", "abl_way5", "abl_way10", "abl_way30"]),
              ("Train-time shots", ["lr1e-3", "abl_shot1", "abl_shotmix"]),
              ("Embedding dimension", ["lr1e-3", "abl_emb64", "abl_emb256"]),
              ("Distance", ["lr1e-3", "abl_cosine"]), ("EMA", ["lr1e-3", "abl_noema"]),
              ("Input size", ["lr1e-3", "abl_in64"]),
              ("Backbone / pretraining", ["lr1e-3", "abl_resnet12", "abl_mnv3_pre96", "abl_mnv3_scr96"])]
    md = ["# Ablations (ProtoNet, Protocol B, validation only)", "",
          "Each row is one 8-epoch run (50 episodes/epoch, cosine schedule over 8 epochs) that changes one thing from the baseline. "
          "Scores are the best epoch on the FIXED validation episodes (100 × 10-way 5-shot over the 37 val classes, never trained on). "
          "Test data was not used. Differences smaller than ~1.5 pp are inside the 95% CI and are treated as ties.", "",
          "![ablations](figures/ablations_val_loss.png)", "", "![lr comparison](figures/lr_comparison.png)", ""]
    for name, tags in groups:
        sel = t[t.tag.isin(tags)]
        if len(sel) > 1 or (len(sel) == 1 and tags[0] != "lr1e-3"):
            md += [f"## {name}", "", *table(sel), ""]
    missing = [ABL_DESC[k] for k in ABL_DESC if k.startswith("abl_") and k not in set(t.tag)]
    if missing:
        md += ["## Not run", "", "Dropped for CPU time and memory (see decisions.md, session 3); `run_batch3.sh` still lists them:", "",
               *[f"- {m}" for m in missing], ""]
    arc = _arcface_vs_scratch()
    if arc:
        md += ["## ArcFace pretraining vs episodic training from scratch (15-epoch final runs, Protocol B val)", "", *arc, ""]
    return "\n".join(md) + "\n"


def _arcface_vs_scratch() -> List[str]:
    out = []
    pairs = [("ProtoNet from scratch", "_protonet_conv4_B_final"), ("ArcFace head only (cosine protos)", "_arcface_conv4_B_final"),
             ("ArcFace pretrain -> ProtoNet fine-tune", "_protonet_conv4_B_arcft_final")]
    rows = []
    for name, suf in pairs:
        rs = sorted(r for r in RUNS.glob(f"*{suf}") if (r / "metrics_epoch.csv").exists())
        if rs and not load(rs[-1])[0].empty:
            df, h, _ = load(rs[-1])
            rows.append(f"| {name} | {df.val_loss.min():.4f} | {100 * df.val_episode_acc.max():.2f} ± "
                        f"{100 * df.val_acc_ci95.iat[int(df.val_episode_acc.idxmax())]:.2f} | {len(df)} |")
    if rows:
        out = ["| recipe | best val loss | best val acc % | epochs |", "|---|---|---|---|", *rows,
               "", "Note: the ArcFace val loss is its cosine-prototype episode CE (scale s = 24) and is not on the same scale as "
               "ProtoNet's. Compare accuracy across recipes."]
    return out


def main() -> None:
    setup_logging("INFO")
    es = epoch_study()
    if es:
        (REPORTS / "epoch_study.md").write_text(es, encoding="utf-8")
        log.info("wrote reports/epoch_study.md")
    ab = ablations()
    if ab:
        (REPORTS / "ablations.md").write_text(ab, encoding="utf-8")
        log.info("wrote reports/ablations.md")


if __name__ == "__main__":
    main()
