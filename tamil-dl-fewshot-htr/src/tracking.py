"""Run tracking: RunLogger, training safeguards, checkpoints, environment capture.

Every training script logs through ``RunLogger`` so every run directory has the same files:
config_snapshot.yaml, env.json, model_summary.txt, class_split.json, metrics_epoch.csv,
metrics_step.csv, history.json, train.log, checkpoints/, test/, plots/.
"""
from __future__ import annotations

import csv
import json
import logging
import math
import platform
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml

from config import ROOT, to_plain

log = logging.getLogger("tracking")

EPOCH_FIELDS = ["epoch", "train_loss", "val_loss", "train_obj", "train_episode_acc", "val_episode_acc", "val_acc_ci95",
                "val_top3_acc", "lr", "epoch_time_s", "gpu_mem_mb", "ram_mb", "grad_norm_mean", "grad_norm_max",
                "clipped_frac", "emb_norm", "inter_dist", "intra_spread", "sep_ratio", "temperature",
                "loss_pos", "loss_neg", "dist_pos", "dist_neg", "active_frac", "gen_gap",
                "overfit_flag", "plateau_flag", "is_best_loss", "is_best_acc"]
STEP_FIELDS = ["step", "epoch", "loss", "running_loss", "lr"]


class NonFiniteLossError(RuntimeError):
    """Raised (after a crash checkpoint is saved) when the loss becomes NaN or Inf."""


def check_finite(loss: float, where: str = "") -> None:
    """Abort with a clear message when ``loss`` is NaN/Inf."""
    if not math.isfinite(float(loss)):
        raise NonFiniteLossError(
            f"Non-finite loss ({loss}) {where}. Training aborted; a crash checkpoint was saved. "
            "Typical causes: learning rate too high, exploding temperature/scale, or bad input values.")


class OverfitDetector:
    """Flags an epoch when train loss has fallen while val loss has risen for ``patience`` epochs in a row."""

    def __init__(self, patience: int = 3) -> None:
        self.patience, self.streak = int(patience), 0
        self.prev_train: Optional[float] = None
        self.prev_val: Optional[float] = None

    def update(self, train_loss: float, val_loss: float) -> bool:
        if self.prev_train is not None and train_loss < self.prev_train and val_loss > self.prev_val:
            self.streak += 1
        else:
            self.streak = 0
        self.prev_train, self.prev_val = train_loss, val_loss
        return self.streak >= self.patience

    def state_dict(self) -> dict:
        return dict(self.__dict__)

    def load_state_dict(self, s: dict) -> None:
        self.__dict__.update(s)


class PlateauDetector:
    """Flags when train loss improved by less than ``eps`` over the last ``patience`` epochs."""

    def __init__(self, patience: int = 5, eps: float = 1e-3) -> None:
        self.patience, self.eps = int(patience), float(eps)
        self.history: List[float] = []

    def update(self, train_loss: float) -> bool:
        self.history.append(float(train_loss))
        if len(self.history) <= self.patience:
            return False
        before = min(self.history[:-self.patience])
        return before - min(self.history[-self.patience:]) < self.eps

    def state_dict(self) -> dict:
        return {"patience": self.patience, "eps": self.eps, "history": list(self.history)}

    def load_state_dict(self, s: dict) -> None:
        self.__dict__.update(s)


class EarlyStopping:
    """Stops when val loss has not improved for ``patience`` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, enabled: bool = True) -> None:
        self.patience, self.min_delta, self.enabled = int(patience), float(min_delta), bool(enabled)
        self.best = math.inf
        self.bad = 0

    def update(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best, self.bad = float(val_loss), 0
        else:
            self.bad += 1
        return self.enabled and self.bad >= self.patience

    def state_dict(self) -> dict:
        return dict(self.__dict__)

    def load_state_dict(self, s: dict) -> None:
        self.__dict__.update(s)


class EMA:
    """Exponential moving average of model weights (buffers are copied)."""

    def __init__(self, model: nn.Module, decay: float = 0.995) -> None:
        import copy
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        for pe, pm in zip(self.module.parameters(), model.parameters()):
            pe.mul_(d).add_(pm.detach(), alpha=1.0 - d)
        for be, bm in zip(self.module.buffers(), model.buffers()):
            be.copy_(bm)


# --- model accounting -----------------------------------------------------------------------
def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def count_flops(model: nn.Module, example: torch.Tensor) -> int:
    """Multiply-accumulate count of conv and linear layers for one forward pass."""
    total = [0]

    def conv_hook(m: nn.Conv2d, _inp, out) -> None:
        total[0] += int(out.numel() / out.shape[0] * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1])

    def lin_hook(m: nn.Linear, _inp, out) -> None:
        total[0] += int(out.numel() / out.shape[0] * m.in_features)

    hooks = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            model(example)
    finally:
        for h in hooks:
            h.remove()
        model.train(was)
    return total[0]


def git_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return "not-a-git-repo"


def env_info() -> Dict[str, Any]:
    """Package versions and hardware for env.json."""
    import importlib.metadata as md
    vers = {}
    for pkg in ["torch", "torchvision", "timm", "numpy", "pandas", "scikit-learn", "opencv-python", "Pillow",
                "matplotlib", "seaborn", "imagehash", "onnx", "onnxruntime", "PyYAML", "tensorboard", "uharfbuzz"]:
        try:
            vers[pkg] = md.version(pkg)
        except Exception:
            vers[pkg] = "missing"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "processor": platform.processor(),
            "packages": vers, "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "torch_threads": torch.get_num_threads(), "git_hash": git_hash()}


def ram_mb() -> float:
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2 ** 20
    except Exception:
        return 0.0


def make_run_dir(runs_root: Path, method: str, backbone: str, protocol: str, tag: str = "") -> Path:
    """runs/<YYYYMMDD_HHMMSS>_<method>_<backbone>_<protocol>[_tag], unique even for parallel starts."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{method}_{backbone}_{protocol}" + (f"_{tag}" if tag else "")
    path, i = runs_root / name, 1
    while True:
        try:
            path.mkdir(parents=True, exist_ok=False)
            return path
        except FileExistsError:
            i += 1
            path = runs_root / f"{name}-{i}"


# --- checkpoints ----------------------------------------------------------------------------
def rng_state() -> Dict[str, Any]:
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def set_rng_state(state: Mapping[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, payload: Dict[str, Any]) -> None:
    """Atomic checkpoint write."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


# --- the logger -----------------------------------------------------------------------------
class RunLogger:
    """Writes every per-run artefact and applies the overfitting / plateau detectors."""

    def __init__(self, run_dir: Path, cfg: Mapping[str, Any], resume: bool = False) -> None:
        self.dir = Path(run_dir)
        for sub in ("checkpoints", "test", "plots"):
            (self.dir / sub).mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        t = cfg["train"]
        self.overfit = OverfitDetector(t.get("overfit_patience", 3))
        self.plateau = PlateauDetector(t.get("plateau_patience", 5), t.get("plateau_eps", 1e-3))
        self.history: Dict[str, Any] = {"status": "running", "epochs": [], "overfit_epochs": [], "plateau_epochs": [],
                                        "best_val_loss": None, "best_val_loss_epoch": None, "best_val_acc": None,
                                        "best_val_acc_epoch": None, "stop_reason": None, "warnings": []}
        self._tb = None
        self._rows = 0
        self._handler: Optional[logging.Handler] = None
        hist_path = self.dir / "history.json"
        if resume and hist_path.exists():
            self.history = json.loads(hist_path.read_text(encoding="utf-8"))
            self.history["status"] = "running"
        else:
            (self.dir / "config_snapshot.yaml").write_text(yaml.safe_dump(to_plain(cfg), allow_unicode=True, sort_keys=False), encoding="utf-8")
            self._init_csv("metrics_epoch.csv", EPOCH_FIELDS)
            self._init_csv("metrics_step.csv", STEP_FIELDS)
        self._attach_log_file()
        if t.get("tensorboard", True):
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb = SummaryWriter(str(self.dir / "tb"))
            except Exception as exc:  # pragma: no cover
                log.warning("TensorBoard disabled: %s", exc)

    def _init_csv(self, name: str, fields: List[str]) -> None:
        with open(self.dir / name, "w", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(fields)

    def _attach_log_file(self) -> None:
        self._handler = logging.FileHandler(self.dir / "train.log", encoding="utf-8")
        self._handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
        logging.getLogger().addHandler(self._handler)

    def truncate_to(self, epoch: int) -> None:
        """On resume, drop logged rows after ``epoch`` so the continued run does not duplicate them."""
        self.history["epochs"] = [r for r in self.history["epochs"] if r["epoch"] <= epoch]
        for key in ("overfit_epochs", "plateau_epochs"):
            self.history[key] = [e for e in self.history[key] if e <= epoch]
        for name in ("metrics_epoch.csv", "metrics_step.csv"):
            p = self.dir / name
            with open(p, "r", newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            fields = EPOCH_FIELDS if name == "metrics_epoch.csv" else STEP_FIELDS
            with open(p, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=fields)
                w.writeheader()
                w.writerows(r for r in rows if int(float(r["epoch"])) <= epoch)

    def start(self, model: nn.Module, example: torch.Tensor, extra: Optional[Dict[str, Any]] = None,
              class_split: Optional[Dict[str, Any]] = None) -> None:
        """Write env.json, model_summary.txt and the class split copy."""
        env = env_info()
        env.update(seed=self.cfg.get("seed"), parameter_count=count_params(model))
        try:
            env["flops_per_image"] = count_flops(getattr(model, "backbone", model), example)
        except Exception as exc:
            env["flops_per_image"] = None
            log.warning("FLOP count failed: %s", exc)
        env.update(extra or {})
        (self.dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        lines = [repr(model), "", f"parameters: {env['parameter_count']:,}", f"MACs per image: {env['flops_per_image']}", ""]
        lines += [f"{n:60s} {tuple(p.shape)!s:24s} {p.numel():>10,}" for n, p in model.named_parameters()]
        (self.dir / "model_summary.txt").write_text("\n".join(lines), encoding="utf-8")
        if class_split is not None:
            (self.dir / "class_split.json").write_text(json.dumps(class_split, ensure_ascii=False, indent=1), encoding="utf-8")

    def warn(self, msg: str) -> None:
        log.warning(msg)
        self.history["warnings"].append(msg)

    def log_step(self, step: int, epoch: int, loss: float, running: float, lr: float) -> None:
        with open(self.dir / "metrics_step.csv", "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow([step, epoch, f"{loss:.6f}", f"{running:.6f}", f"{lr:.8f}"])
        if self._tb:
            self._tb.add_scalar("step/loss", loss, step)
            self._tb.add_scalar("step/lr", lr, step)

    def log_epoch(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Apply detectors, mark bests, append to CSV/history/TensorBoard, print a table row."""
        h = self.history
        ep = int(row["epoch"])
        row["gen_gap"] = row["val_loss"] - row["train_loss"]
        row["overfit_flag"] = int(self.overfit.update(row["train_loss"], row["val_loss"]))
        row["plateau_flag"] = int(self.plateau.update(row["train_loss"]))
        row["is_best_loss"] = int(h["best_val_loss"] is None or row["val_loss"] < h["best_val_loss"])
        row["is_best_acc"] = int(h["best_val_acc"] is None or row["val_episode_acc"] > h["best_val_acc"])
        if row["is_best_loss"]:
            h["best_val_loss"], h["best_val_loss_epoch"] = row["val_loss"], ep
            h["train_loss_at_best"] = row["train_loss"]
        if row["is_best_acc"]:
            h["best_val_acc"], h["best_val_acc_epoch"] = row["val_episode_acc"], ep
        if row["overfit_flag"]:
            h["overfit_epochs"].append(ep)
            self.warn(f"epoch {ep}: possible OVERFITTING - train loss falling while val loss rising for "
                      f"{self.overfit.patience}+ consecutive epochs")
        if row["plateau_flag"]:
            h["plateau_epochs"].append(ep)
            if len(h["plateau_epochs"]) == 1 or ep - h["plateau_epochs"][-2] > 1:
                self.warn(f"epoch {ep}: train loss PLATEAU - improved < {self.plateau.eps} over {self.plateau.patience} epochs")
        clean = {k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in row.items()}
        h["epochs"].append(clean)
        with open(self.dir / "metrics_epoch.csv", "a", newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerow(["" if row.get(k) is None else (f"{row[k]:.6g}" if isinstance(row[k], float) else row[k])
                                     for k in EPOCH_FIELDS])
        self._write_history()
        if self._tb:
            for k, v in row.items():
                if isinstance(v, (int, float)) and v is not None and k != "epoch":
                    self._tb.add_scalar(f"epoch/{k}", v, ep)
            self._tb.flush()
        if self._rows % 20 == 0:
            log.info("%5s | %9s %9s | %7s %16s %6s | %8s %6s %7s %5s %6s %5s | %s", "epoch", "train_loss", "val_loss",
                     "tr_acc", "val_acc (95% CI)", "top3", "lr", "time", "gnorm", "clip", "sep", "temp", "flags")
        self._rows += 1
        flags = "".join(["*L" if row["is_best_loss"] else "", "*A" if row["is_best_acc"] else "",
                         " OVERFIT" if row["overfit_flag"] else "", " PLATEAU" if row["plateau_flag"] else ""])
        log.info("%5d | %9.4f %9.4f | %7.4f %8.4f +-%6.4f %6.4f | %8.2e %5.0fs %7.3f %5.2f %6.2f %5.2f | %s",
                 ep, row["train_loss"], row["val_loss"], row["train_episode_acc"], row["val_episode_acc"],
                 row["val_acc_ci95"], row["val_top3_acc"], row["lr"], row["epoch_time_s"], row["grad_norm_mean"],
                 row["clipped_frac"], row.get("sep_ratio") or 0.0, row.get("temperature") or 0.0, flags)
        return row

    def _write_history(self) -> None:
        tmp = self.dir / "history.json.tmp"
        tmp.write_text(json.dumps(self.history, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.dir / "history.json")

    def finish(self, stop_reason: str, status: str = "finished", extra: Optional[Dict[str, Any]] = None) -> None:
        self.history.update(status=status, stop_reason=stop_reason, **(extra or {}))
        self._write_history()
        log.info("run %s: %s (best val loss %.4f at epoch %s; best val acc %.4f at epoch %s)", status, stop_reason,
                 self.history["best_val_loss"] or float("nan"), self.history["best_val_loss_epoch"],
                 self.history["best_val_acc"] or float("nan"), self.history["best_val_acc_epoch"])
        self.close()

    def close(self) -> None:
        if self._tb:
            self._tb.close()
            self._tb = None
        if self._handler:
            logging.getLogger().removeHandler(self._handler)
            self._handler.close()
            self._handler = None

    def state_dict(self) -> dict:
        return {"overfit": self.overfit.state_dict(), "plateau": self.plateau.state_dict()}

    def load_state_dict(self, s: dict) -> None:
        self.overfit.load_state_dict(s["overfit"])
        self.plateau.load_state_dict(s["plateau"])
