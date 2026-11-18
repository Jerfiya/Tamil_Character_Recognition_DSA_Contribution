"""Training entry point for every few-shot method (and, via train_baseline.py, the supervised baseline).

One "epoch" is a fixed number of episodes (``train.episodes_per_epoch``) so curves are
comparable across methods. Validation scores a FIXED list of episodes without augmentation.

Examples:
    python src/train_fewshot.py --set method.name=protonet train.protocol=B
    python src/train_fewshot.py --resume runs/<run_dir>
    python src/train_fewshot.py --lr-find --set method.name=protonet
"""
from __future__ import annotations

import argparse
import csv
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

from augment import Augmenter
from config import Cfg, _wrap, add_common_args, apply_overrides, get_device, load_config, resolve, seed_everything, setup_logging
from datasets import ProcessedData, cap_pools, load_processed
from episodes import Episode, EpisodeSampler
from fewshot import EpisodeInfo, build_method
from losses import separation_stats
from tracking import (EMA, EarlyStopping, NonFiniteLossError, RunLogger, check_finite, load_checkpoint, make_run_dir,
                      ram_mb, rng_state, save_checkpoint, set_rng_state)

log = logging.getLogger("train")
COMP_KEYS = ("loss_pos", "loss_neg", "dist_pos", "dist_neg", "active_frac")


class Trainer:
    """Builds data, model, optimiser and logger from a config and runs the epoch loop."""

    def __init__(self, cfg: Cfg, run_dir: Optional[Path] = None, resume: bool = False, tag: str = "",
                 use_logger: bool = True, data: Optional[ProcessedData] = None) -> None:
        self.cfg = cfg
        t = cfg.train
        torch.set_num_threads(int(cfg.get("threads", 4)))
        seed_everything(int(cfg.seed))
        self.device = get_device()
        self.data = data or load_processed(cfg)
        self.protocol = str(t.protocol)
        if cfg.method.name == "supervised" and self.protocol != "A":
            raise ValueError("the supervised baseline is defined for Protocol A only")
        sp, _ = self.data.episode_pools(self.protocol, "train")
        self.train_classes = sorted(sp)
        self.global_map = np.full(self.data.n_classes, -1, dtype=np.int64)
        self.global_map[self.train_classes] = np.arange(len(self.train_classes))
        k = list(t.k_shot) if isinstance(t.k_shot, (list, tuple)) else int(t.k_shot)
        self.sampler = EpisodeSampler(sp, None, int(t.n_way), k, int(t.q_query), seed=int(cfg.seed), name="train")
        self.aug = Augmenter(cfg.augment, seed=int(cfg.seed) + 1, device=self.device)
        self.model = build_method(cfg, len(self.train_classes)).to(self.device)
        self.ema = EMA(self.model, float(t.ema_decay)) if t.ema else None
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=float(t.lr), weight_decay=float(t.weight_decay))
        self.epe = int(t.episodes_per_epoch)
        total, warm = int(t.epochs) * self.epe, int(t.warmup_epochs) * self.epe
        self.sched_kind = str(t.scheduler)
        if self.sched_kind == "cosine":
            def lr_lambda(step: int) -> float:
                if step < warm:
                    return (step + 1) / max(warm, 1)
                return 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warm) / max(total - warm, 1))))
            self.sched: Any = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)
        elif self.sched_kind == "plateau":
            self.sched = torch.optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=0.5, patience=3)
        else:
            self.sched = None
        self.amp = self.device.type == "cuda" and t.amp in (True, "auto")
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.early = EarlyStopping(int(t.patience), 0.0, bool(t.early_stopping))
        self.epoch, self.global_step, self.running = 0, 0, None
        self._build_val()
        self.logger: Optional[RunLogger] = None
        self.run_dir = run_dir
        if use_logger:
            if self.run_dir is None:
                self.run_dir = make_run_dir(resolve(cfg, "runs"), cfg.method.name, cfg.model.backbone, self.protocol, tag)
            self.logger = RunLogger(self.run_dir, cfg, resume=resume)
            if resume:
                self._load(self.run_dir / "checkpoints" / "last.pt")
            else:
                example = self.data.normalize(self.data.batch([0], self.device))
                self.logger.start(self.model, example, {"protocol": self.protocol, "n_train_classes": len(self.train_classes),
                                                        "amp": self.amp, "episodes_per_epoch": self.epe},
                                  self.data.class_split)
                log.info("run dir: %s", self.run_dir)

    # -- validation -------------------------------------------------------------------------
    def _build_val(self) -> None:
        v = self.cfg.val
        vs, vq = self.data.episode_pools(self.protocol, "val")
        if vq is not None:
            vs = cap_pools(vs, 10, int(v.seed))        # fixed support bank keeps per-epoch embedding cheap
        samp = EpisodeSampler(vs, vq, int(v.n_way), int(v.k_shot), int(v.q_query), seed=int(v.seed), name="val")
        n = int(v.episodes) if not self.model.needs_images else min(int(v.episodes), 30)
        self.val_eps: List[Episode] = samp.fixed(n)
        used = [np.concatenate([e.support.reshape(-1), e.query]) for e in self.val_eps]
        self.val_pool_idx = np.concatenate(list(vq.values())) if (vq is not None and self.cfg.method.name == "supervised") else None
        if self.val_pool_idx is not None:
            used.append(self.val_pool_idx)
        self.val_idx = np.unique(np.concatenate(used))
        self.val_pos = np.full(len(self.data.labels), -1, dtype=np.int64)
        self.val_pos[self.val_idx] = np.arange(len(self.val_idx))

    @torch.no_grad()
    def embed(self, net, idx: np.ndarray, chunk: int = 512) -> torch.Tensor:
        """Features of images ``idx`` with ``net`` in eval mode (no augmentation)."""
        out = [net.features(self.data.normalize(self.data.batch(idx[i:i + chunk], self.device))) for i in range(0, len(idx), chunk)]
        return torch.cat(out)

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        net = self.ema.module if self.ema is not None else self.model
        net.eval()
        losses, accs, top3s = [], [], []
        stats: Dict[str, float] = {}
        feats = None
        if not net.needs_images:
            feats = self.embed(net, self.val_idx)
            stats = separation_stats(feats, torch.as_tensor(self.data.labels[self.val_idx], device=feats.device))
        for ep in self.val_eps:
            n, k = ep.n_way, ep.k_shot
            qy = torch.as_tensor(ep.query_y, device=self.device)
            if feats is not None:
                sf, qf = feats[self.val_pos[ep.support.reshape(-1)]], feats[self.val_pos[ep.query]]
                logits = net.episode_logits(sf, qf, n, k)
                losses.append(net.val_loss(sf, qf, qy, n, k, logits))
            else:
                xs = self.data.normalize(self.data.batch(ep.support.reshape(-1), self.device))
                xq = self.data.normalize(self.data.batch(ep.query, self.device))
                logits = net.episode_logits_images(xs, xq, n, k)
                losses.append(float(torch.nn.functional.cross_entropy(logits, qy)))
            accs.append(float((logits.argmax(1) == qy).float().mean()))
            top3s.append(float((logits.topk(min(3, n), dim=1).indices == qy.unsqueeze(1)).any(1).float().mean()))
        out = {"val_loss": float(np.mean(losses)), "val_episode_acc": float(np.mean(accs)),
               "val_acc_ci95": float(1.96 * np.std(accs) / math.sqrt(len(accs))), "val_top3_acc": float(np.mean(top3s)), **stats}
        if self.val_pool_idx is not None and feats is not None:
            y = torch.as_tensor(self.global_map[self.data.labels[self.val_pool_idx]], device=self.device)
            pv = net.pool_val(feats[self.val_pos[self.val_pool_idx]], y)
            if pv is not None:      # supervised baseline: val loss is the CE of its own head on the val pool
                out.update(val_loss=pv["loss"], val_pool_acc=pv["acc"], val_top3_acc=pv["top3"])
        out.update(net.stats())
        return out

    # -- training ---------------------------------------------------------------------------
    def _episode_batch(self, ep: Episode, augment: bool):
        idx = np.concatenate([ep.support.reshape(-1), ep.query])
        x = self.data.batch(idx, self.device)
        if augment:
            x = self.aug(x)
        x = self.data.normalize(x)
        n, k = ep.n_way, ep.k_shot
        qy = torch.as_tensor(ep.query_y, device=self.device)
        local = torch.cat([torch.arange(n, device=self.device).repeat_interleave(k), qy])
        glob = torch.as_tensor(self.global_map[self.data.labels[idx]], device=self.device)
        return x, EpisodeInfo(n, k, qy, local, glob, float(self.cfg.train.label_smoothing))

    def train_step(self, lr_override: Optional[float] = None) -> Dict[str, float]:
        """One optimisation step on one episode."""
        t = self.cfg.train
        x, info = self._episode_batch(self.sampler.sample(), bool(t.augment))
        if lr_override is not None:
            for g in self.opt.param_groups:
                g["lr"] = lr_override
        with torch.autocast(self.device.type, enabled=self.amp):
            out = self.model.train_loss_images(x, info) if self.model.needs_images else self.model.train_loss(self.model.features(x), info)
        loss = out["loss"]
        if not torch.isfinite(loss):
            if self.logger is not None:
                save_checkpoint(self.run_dir / "checkpoints" / "crash.pt", self._payload({"reason": "non-finite loss"}))
            check_finite(float(loss), f"at epoch {self.epoch + 1}, step {self.global_step + 1}")
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        clip = float(t.grad_clip)
        gn = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip if clip > 0 else float("inf")))
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.sched_kind == "cosine" and lr_override is None:
            self.sched.step()
        if self.ema is not None:
            self.ema.update(self.model)
        res = {"obj": float(loss.detach()), "plain": out["plain"], "acc": out["acc"], "gn": gn,
               "clipped": float(clip > 0 and gn > clip)}
        res.update({k: out[k] for k in COMP_KEYS if k in out})
        return res

    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        acc: Dict[str, List[float]] = {}
        every = int(self.cfg.train.step_log_every)
        for _ in range(self.epe):
            r = self.train_step()
            self.global_step += 1
            self.running = r["plain"] if self.running is None else 0.95 * self.running + 0.05 * r["plain"]
            for k, v in r.items():
                acc.setdefault(k, []).append(v)
            if self.logger is not None and self.global_step % every == 0:
                self.logger.log_step(self.global_step, self.epoch, r["plain"], self.running, self.opt.param_groups[0]["lr"])
        out = {"train_loss": float(np.mean(acc["plain"])), "train_obj": float(np.mean(acc["obj"])),
               "train_episode_acc": float(np.mean(acc["acc"])), "grad_norm_mean": float(np.mean(acc["gn"])),
               "grad_norm_max": float(np.max(acc["gn"])), "clipped_frac": float(np.mean(acc["clipped"]))}
        out.update({k: float(np.mean(acc[k])) for k in COMP_KEYS if k in acc})
        return out

    def fit(self, stop_after: Optional[int] = None) -> Path:
        """Run the epoch loop. ``stop_after`` ends the call early (used to test exact resume)."""
        assert self.logger is not None
        t = self.cfg.train
        ck = self.run_dir / "checkpoints"
        reason = "max_epochs"
        try:
            while self.epoch < int(t.epochs):
                self.epoch += 1
                t0 = time.time()
                if self.device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                row: Dict[str, Any] = {"epoch": self.epoch, **self.train_epoch(), **self.validate()}
                row.update(lr=float(self.opt.param_groups[0]["lr"]), epoch_time_s=time.time() - t0, ram_mb=ram_mb(),
                           gpu_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20 if self.device.type == "cuda" else 0.0)
                row = self.logger.log_epoch(row)
                if self.sched_kind == "plateau":
                    self.sched.step(row["val_loss"])
                stop = self.early.update(row["val_loss"])
                payload = self._payload(row)
                save_checkpoint(ck / "last.pt", payload)
                if row["is_best_loss"]:
                    save_checkpoint(ck / "best_val_loss.pt", payload)
                if row["is_best_acc"]:
                    save_checkpoint(ck / "best_val_acc.pt", payload)
                if int(t.ckpt_every) > 0 and self.epoch % int(t.ckpt_every) == 0:
                    save_checkpoint(ck / f"epoch_{self.epoch:03d}.pt", payload)
                if stop:
                    reason = f"early_stopping: val loss did not improve for {self.early.patience} epochs"
                    break
                if stop_after is not None and self.epoch >= stop_after:
                    self.logger.close()
                    return self.run_dir
        except NonFiniteLossError as exc:
            log.error(str(exc))
            self.logger.finish(f"crashed: {exc}", status="crashed")
            raise
        except KeyboardInterrupt:
            self.logger.finish("interrupted", status="interrupted")
            raise
        self.logger.finish(reason, extra={"epochs_run": self.epoch})
        return self.run_dir

    # -- checkpoints ------------------------------------------------------------------------
    def _payload(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {"model": self.model.state_dict(), "ema": self.ema.module.state_dict() if self.ema else None,
                "ema_updates": self.ema.updates if self.ema else 0, "optimizer": self.opt.state_dict(),
                "scheduler": self.sched.state_dict() if self.sched is not None else None,
                "scaler": self.scaler.state_dict(), "epoch": self.epoch, "global_step": self.global_step,
                "running": self.running, "metrics": {k: v for k, v in metrics.items()},
                "rng": rng_state(), "sampler": self.sampler.state_dict(), "aug": self.aug.state_dict(),
                "early": self.early.state_dict(), "logger": self.logger.state_dict() if self.logger else None,
                "train_classes": self.train_classes, "class_names": self.data.class_names,
                "config": yaml.safe_load(yaml.safe_dump(_plain(self.cfg)))}

    def _load(self, path: Path) -> None:
        ck = load_checkpoint(path)
        self.model.load_state_dict(ck["model"])
        if self.ema is not None and ck["ema"] is not None:
            self.ema.module.load_state_dict(ck["ema"])
            self.ema.updates = ck["ema_updates"]
        self.opt.load_state_dict(ck["optimizer"])
        if self.sched is not None and ck["scheduler"] is not None:
            self.sched.load_state_dict(ck["scheduler"])
        self.scaler.load_state_dict(ck["scaler"])
        self.epoch, self.global_step, self.running = ck["epoch"], ck["global_step"], ck["running"]
        set_rng_state(ck["rng"])
        self.sampler.load_state_dict(ck["sampler"])
        self.aug.load_state_dict(ck["aug"])
        self.early.load_state_dict(ck["early"])
        self.logger.load_state_dict(ck["logger"])
        self.logger.truncate_to(self.epoch)
        log.info("resumed from %s at epoch %d", path, self.epoch)

    # -- LR range test ----------------------------------------------------------------------
    def lr_find(self, out_csv: Path, lr_min: float = 1e-5, lr_max: float = 0.3, steps: int = 80) -> float:
        """Exponentially increase the LR each step, record the smoothed loss, suggest an LR."""
        self.model.train()
        rows, smooth, best = [], None, float("inf")
        for i in range(steps):
            lr = lr_min * (lr_max / lr_min) ** (i / (steps - 1))
            try:
                r = self.train_step(lr_override=lr)
            except NonFiniteLossError:
                break
            smooth = r["plain"] if smooth is None else 0.8 * smooth + 0.2 * r["plain"]
            rows.append((lr, r["plain"], smooth))
            best = min(best, smooth)
            if i > 10 and smooth > 3.0 * best:
                break
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["lr", "loss", "smoothed_loss"])
            w.writerows(rows)
        lrs, sm = np.array([r[0] for r in rows]), np.array([r[2] for r in rows])
        grad = np.gradient(sm, np.log10(lrs))
        suggestion = float(lrs[int(np.argmin(grad[3:-1])) + 3]) if len(rows) > 8 else float(lrs[len(lrs) // 2])
        log.info("LR finder: steepest descent at lr=%.2e, loss minimum at lr=%.2e (csv: %s)", suggestion,
                 float(lrs[int(np.argmin(sm))]), out_csv)
        return suggestion


def _plain(cfg: Any) -> Any:
    from config import to_plain
    return to_plain(cfg)


def run_from_args(args: argparse.Namespace, forced: Optional[List[str]] = None) -> Path:
    """Shared CLI logic for train_fewshot.py and train_baseline.py."""
    setup_logging(args.log_level)
    if args.resume:
        run_dir = Path(args.resume)
        cfg = _wrap(yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8")))
        cfg = apply_overrides(cfg, args.set)
        trainer = Trainer(cfg, run_dir=run_dir, resume=True)
    else:
        cfg = load_config(args.config, list(forced or []) + list(args.set))
        if args.lr_find:
            trainer = Trainer(cfg, use_logger=False)
            out = resolve(cfg, "reports") / "lr_finder" / f"{cfg.method.name}_{cfg.model.backbone}_{cfg.train.protocol}.csv"
            trainer.lr_find(out)
            return out
        trainer = Trainer(cfg, tag=args.tag)
    try:
        run_dir = trainer.fit()
    except NonFiniteLossError:
        sys.exit(2)
    print(f"RUN_DIR={run_dir}", flush=True)
    return run_dir


def build_parser(desc: str) -> argparse.ArgumentParser:
    parser = add_common_args(argparse.ArgumentParser(description=desc, formatter_class=argparse.RawDescriptionHelpFormatter))
    parser.add_argument("--resume", default=None, help="run directory to resume exactly from checkpoints/last.pt")
    parser.add_argument("--tag", default="", help="suffix for the run directory name")
    parser.add_argument("--lr-find", action="store_true", help="run the learning-rate range test and exit")
    return parser


def main() -> None:
    run_from_args(build_parser(__doc__).parse_args())


if __name__ == "__main__":
    main()
