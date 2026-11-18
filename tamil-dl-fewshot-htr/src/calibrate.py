"""Calibration on VALIDATION data only: temperature scaling and the open-set threshold.

Closed set: K support images per val class (Protocol B: val classes; Protocol A: all classes,
support from train, queries from val) -> logits for every query. A temperature T is fitted by
minimising NLL; ECE is reported before and after.

Open set: val classes are split in half (enrolled / unseen). Queries of both halves are scored
against the enrolled prototypes only; the score is the maximum logit (an N-independent
similarity). The threshold accepts ``target_tpr`` of the enrolled queries; anything below is
"unknown".

    python src/calibrate.py runs/<run_dir> [--ckpt best_val_loss.pt]

Writes runs/<run>/calibration_<ckpt-stem>.json.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from config import setup_logging
from runmodel import RunModel, auroc, ece_bins, fit_temperature, fullset_split, openset_split

log = logging.getLogger("calibrate")
TARGET_TPR = 0.95


def calibrate(rm: RunModel, target_tpr: float = TARGET_TPR) -> Dict[str, Any]:
    e = rm.cfg.eval
    k, seed = int(e.key_shot), int(e.seed) + 7
    fs = fullset_split(rm.data, rm.protocol, "val", k, seed)
    lg = rm.logits(fs["support"], fs["query"]).cpu().float()
    y = torch.as_tensor(fs["query_y"])
    t = fit_temperature(lg, y)
    p0, p1 = F.softmax(lg, 1), F.softmax(lg / t, 1)
    correct = (lg.argmax(1) == y).numpy()
    ece0, _ = ece_bins(p0.max(1).values.numpy(), correct)
    ece1, _ = ece_bins(p1.max(1).values.numpy(), correct)
    enrolled, unseen = openset_split(fs["classes"], seed)
    keep = np.isin(fs["classes"], enrolled)
    sup = fs["support"][keep]
    q_cls = fs["classes"][fs["query_y"]]
    in_q, out_q = fs["query"][np.isin(q_cls, enrolled)], fs["query"][np.isin(q_cls, unseen)]
    s_in = rm.logits(sup, in_q).max(1).values.cpu().numpy()
    s_out = rm.logits(sup, out_q).max(1).values.cpu().numpy()
    thr = float(np.quantile(s_in, 1.0 - target_tpr))
    res = {"run": rm.run_dir.name, "checkpoint": rm.ckpt_path.name, "split": f"{rm.protocol}/val", "k_shot": k,
           "n_classes": int(len(fs["classes"])), "n_queries": int(len(fs["query"])),
           "val_acc": float(correct.mean()), "val_nll_before": float(F.cross_entropy(lg, y)),
           "val_nll_after": float(F.cross_entropy(lg / t, y)), "temperature": t, "ece_before": ece0, "ece_after": ece1,
           "openset_score": "max_logit", "openset_threshold": thr, "openset_target_tpr": target_tpr,
           "openset_val_auroc": auroc(s_in, s_out), "openset_val_unknown_rejection": float((s_out < thr).mean()),
           "openset_enrolled_classes": int(len(enrolled)), "openset_unseen_classes": int(len(unseen))}
    log.info("%s %s: T=%.3f ECE %.4f -> %.4f, open-set thr=%.3f (val AUROC %.3f)", rm.run_dir.name, rm.ckpt_path.name,
             t, ece0, ece1, thr, res["openset_val_auroc"])
    return res


def calibration_path(run_dir: Path, ckpt: str) -> Path:
    return Path(run_dir) / f"calibration_{Path(ckpt).stem}.json"


def load_or_calibrate(rm: RunModel) -> Dict[str, Any]:
    path = calibration_path(rm.run_dir, rm.ckpt_path.name)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    res = calibrate(rm)
    path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--ckpt", default="best_val_loss.pt")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    rm = RunModel(Path(a.run_dir), a.ckpt)
    res = calibrate(rm)
    calibration_path(rm.run_dir, rm.ckpt_path.name).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
