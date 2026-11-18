"""Collect every evaluated run into comparison tables (reads stored results only).

Outputs:
    reports/model_comparison.csv / .md   one row per run (primary checkpoint best_val_loss.pt)
    reports/results.csv                  every few-shot cell of every run
    reports/seeds.csv                    ProtoNet Protocol B seeds (mean +- std in model_comparison.md)

    python src/summarize.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from config import ROOT, parse_run_name, setup_logging

log = logging.getLogger("summarize")
RUNS, REPORTS = ROOT / "runs", ROOT / "reports"
NICE = {"protonet": "ProtoNet", "matchingnet": "Matching Net", "relationnet": "Relation Net", "siamese": "Siamese (triplet)",
        "arcface": "ArcFace (cosine protos)", "baselinepp": "Baseline++", "maml": "FO-MAML", "supervised": "Supervised CE"}
PRIMARY = "best_val_loss.pt"
_LAT_CSV = REPORTS / "latency.csv"      # idle-CPU re-measurement (src/measure_latency.py) overrides the stored value
_LAT = dict(pd.read_csv(_LAT_CSV)[["run", "cpu_ms_per_image"]].values) if _LAT_CSV.exists() else {}


def _cell(fr: Optional[pd.DataFrame], split: str, way: str, k: int, col: str = "acc") -> float:
    if fr is None:
        return np.nan
    s = fr[(fr.split == split) & (fr.n_way.astype(str) == way) & (fr.k_shot == k)]
    return float(s[col].iat[0]) if len(s) else np.nan


def run_row(rd: Path) -> Optional[Dict[str, Any]]:
    ts_path = rd / "test" / "test_summary.json"
    if not ts_path.exists():
        return None
    ts = json.loads(ts_path.read_text(encoding="utf-8"))
    prim = next((r for r in ts if r["checkpoint"] == PRIMARY), None)
    if prim is None:
        return None
    h = json.loads((rd / "history.json").read_text(encoding="utf-8")) if (rd / "history.json").exists() else {}
    fr = pd.read_csv(rd / "test" / "fewshot_results.csv") if (rd / "test" / "fewshot_results.csv").exists() else None
    main = prim["split"]
    other = {r["checkpoint"]: r for r in ts}
    row = {"run": rd.name, "method": prim["method"], "backbone": prim["backbone"], "protocol": prim["protocol"],
           "tag": parse_run_name(rd.name)["tag"], "split": main, "params": prim["params"],
           "cpu_ms_per_image": _LAT.get(rd.name, prim["cpu_ms_per_image"]), "epochs_run": len(h.get("epochs", [])),
           "best_epoch": h.get("best_val_loss_epoch"), "best_val_loss": h.get("best_val_loss"),
           "best_val_acc": h.get("best_val_acc"), "best_val_acc_epoch": h.get("best_val_acc_epoch"),
           "stop_reason": h.get("stop_reason"), "overfit_epochs": " ".join(map(str, h.get("overfit_epochs", []))),
           "plateau_epochs": len(h.get("plateau_epochs", [])),
           "test_loss_5w5s": prim.get("episode_5w5s_loss"),
           "test_acc_5w1s": _cell(fr, main, "5", 1), "test_ci_5w1s": _cell(fr, main, "5", 1, "ci95"),
           "test_acc_5w5s": prim.get("episode_5w5s_acc"), "test_ci_5w5s": prim.get("episode_5w5s_ci95"),
           "test_acc_20w5s": _cell(fr, main, "20", 5), "test_acc_allw5s": _cell(fr, main, "all", 5),
           "hard_novel_5w1s": _cell(fr, "B_hard_novel", "5", 1), "hard_novel_5w5s": _cell(fr, "B_hard_novel", "5", 5),
           "fullset_n_classes": prim.get("fullset_n_classes"), "fullset_acc": prim.get("fullset_acc"),
           "fullset_top5": prim.get("fullset_top5"), "macro_f1": prim.get("macro_f1"), "weighted_f1": prim.get("weighted_f1"),
           "ece_before": prim.get("ece_before"), "ece_after": prim.get("ece_after"), "temperature": prim.get("temperature"),
           "openset_auroc": prim.get("openset_auroc"), "openset_unknown_rejection": prim.get("openset_unknown_rejection_rate"),
           "openset_false_reject": prim.get("openset_false_reject_rate")}
    for ck, short in (("best_val_acc.pt", "bestacc"), ("last.pt", "last")):
        if ck in other:
            row[f"{short}_ckpt_epoch"] = other[ck]["checkpoint_epoch"]
            row[f"{short}_test_acc_5w5s"] = other[ck].get("episode_5w5s_acc")
    for f, cols in (("transductive.csv", None), ("source_aware.csv", None)):
        p = rd / "test" / f
        if p.exists() and p.stat().st_size > 2:
            d = pd.read_csv(p)
            if f == "transductive.csv" and len(d):
                d = d[d.split == main]
                row["transductive_gain_5w1s"] = float(d[(d.n_way == 5) & (d.k_shot == 1)].gain.mean())
                row["transductive_gain_5w5s"] = float(d[(d.n_way == 5) & (d.k_shot == 5)].gain.mean())
            if f == "source_aware.csv" and len(d):
                row["cross_source_acc"] = float(d.cross_source_acc.iat[0])
                row["cross_source_drop"] = float(d["drop"].iat[0])
    return row


def fmt(x: Any, pct: bool = False, nd: int = 3) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    if pct:
        return f"{100 * x:.1f}"
    return f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def main() -> None:
    setup_logging("INFO")
    rows, cells = [], []
    for rd in sorted(RUNS.iterdir()):
        r = run_row(rd)
        if r is None:
            continue
        rows.append(r)
        fr = rd / "test" / "fewshot_results.csv"
        if fr.exists():
            cells.append(pd.read_csv(fr).assign(run=rd.name, method=r["method"], backbone=r["backbone"], protocol=r["protocol"], tag=r["tag"]))
    if not rows:
        raise SystemExit("no evaluated runs (run src/evaluate.py first)")
    df = pd.DataFrame(rows)
    df.to_csv(REPORTS / "model_comparison.csv", index=False, encoding="utf-8-sig")
    allc = pd.concat(cells) if cells else pd.DataFrame()
    if len(allc):
        allc.to_csv(REPORTS / "results.csv", index=False, encoding="utf-8-sig")
    seeds = allc[(allc.method == "protonet") & (allc.protocol == "B") & allc.tag.str.match(r"^final") & (allc.n_way.astype(str) == "5")
                 & allc.k_shot.isin([1, 5])] if len(allc) else pd.DataFrame()
    if len(seeds):
        seeds.to_csv(REPORTS / "seeds.csv", index=False, encoding="utf-8-sig")

    md = ["# Model comparison (TEST set)", "",
          "One row per run; numbers from the PRIMARY checkpoint (`best_val_loss.pt`, chosen on validation loss). "
          "Accuracy = mean over >= 600 test episodes (± 95% CI) on the run's own test split: Protocol A = known classes "
          "(support from train images, queries from held-out test images); Protocol B = 37 novel classes never seen in "
          "training. *Full-set* = every test query classified among all classes of the split with 5 support images each. "
          "Latency = CPU, batch 1, embedding only.", ""]
    cols = ["method", "backbone", "protocol", "tag", "params", "cpu_ms_per_image", "best_epoch", "best_val_loss",
            "test_acc_5w1s", "test_acc_5w5s", "test_acc_20w5s", "test_acc_allw5s", "hard_novel_5w5s", "fullset_acc", "macro_f1",
            "ece_before", "ece_after", "openset_auroc"]
    md += ["| method | backbone | proto | tag | params | CPU ms | best ep / run | best val loss | 5w1s % | 5w5s % (±CI) | 20w5s % | "
           "all-way 5s % | hard-novel 5w5s % | full-set % | macro F1 | ECE before → after | open-set AUROC |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    order = df.assign(_m=df.tag.str.startswith(("final", "arcft")).map({True: 0, False: 1})).sort_values(
        ["_m", "protocol", "test_acc_5w5s"], ascending=[True, False, False])
    for _, r in order.iterrows():
        md.append(f"| {NICE.get(r.method, r.method)} | {r.backbone} | {r.protocol} | {r.tag} | {r.params / 1e3:.0f}k | "
                  f"{r.cpu_ms_per_image:.1f} | {r.best_epoch} / {r.epochs_run} | {fmt(r.best_val_loss, nd=4)} | "
                  f"{fmt(r.test_acc_5w1s, True)} | {fmt(r.test_acc_5w5s, True)} ± {fmt(r.test_ci_5w5s, True)} | "
                  f"{fmt(r.test_acc_20w5s, True)} | {fmt(r.test_acc_allw5s, True)} | {fmt(r.hard_novel_5w5s, True)} | "
                  f"{fmt(r.fullset_acc, True)} | {fmt(r.macro_f1)} | {fmt(r.ece_before)} → {fmt(r.ece_after)} | {fmt(r.openset_auroc)} |")
    if len(seeds):
        g = seeds.groupby(["split", "k_shot"]).acc.agg(["mean", "std", "count"]).reset_index()
        md += ["", "## ProtoNet (Protocol B) across seeds — 5-way", "", "| split | K | mean % | std % | seeds |", "|---|---|---|---|---|"]
        md += [f"| {r.split} | {r.k_shot} | {100 * r['mean']:.1f} | {100 * r['std']:.1f} | {r['count']} |" for _, r in g.iterrows()]
    md += ["", "## Checkpoint transparency (5-way 5-shot test accuracy)", "",
           "| run | best_val_loss.pt (epoch) | best_val_acc.pt (epoch) | last.pt (epoch) |", "|---|---|---|---|"]
    for _, r in order.iterrows():
        md.append(f"| {r.run} | {fmt(r.test_acc_5w5s, True)} ({r.best_epoch}) | {fmt(r.get('bestacc_test_acc_5w5s'), True)} "
                  f"({r.get('bestacc_ckpt_epoch')}) | {fmt(r.get('last_test_acc_5w5s'), True)} ({r.get('last_ckpt_epoch')}) |")
    (REPORTS / "model_comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log.info("wrote reports/model_comparison.md (%d runs)", len(df))


if __name__ == "__main__":
    main()
