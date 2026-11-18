"""Final TEST evaluation of one run. The only script that reads test data; one look per checkpoint.

For each of best_val_loss.pt, best_val_acc.pt and last.pt (transparency) it writes a row of
``test_summary``. The PRIMARY checkpoint (best_val_loss.pt; early stopping is on val loss)
additionally gets every detailed result file. Temperature and the open-set threshold come
from ``calibrate.py`` (validation only) and are never tuned here.

Test splits: Protocol A runs -> A test pool (support from train pool). Protocol B runs ->
novel classes and the hard-novel subset.

Files in runs/<run>/test/:
    test_summary.json / .csv      per checkpoint: episode loss/acc/top-3 (key cell), full-set acc,
                                  top-3/5, macro & weighted F1, NLL, ECE before/after temperature
    fewshot_results.csv           N-way x K-shot grid, >= 600 episodes per cell (see decisions.md)
    per_class_metrics.csv         precision, recall, F1, support, mean loss per class
    confusion_matrix.npy (+ confusion_labels.json), top_confused_pairs.csv
    category_accuracy.csv, consonant_row_accuracy.csv, vowel_sign_accuracy.csv
    source_aware.csv              key cell with support/query from different source groups
    openset.csv (+ openset_scores.npz), transductive.csv, predictions.csv, calibration_bins.csv
    embeddings.npz, example_episodes.json

    python src/evaluate.py runs/<run_dir>
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from calibrate import load_or_calibrate
from config import setup_logging
from datasets import load_processed
from episodes import EpisodeSampler
from fewshot.transductive import transductive_logits
from runmodel import RunModel, auroc, ci95, ece_bins, fullset_split, openset_split

log = logging.getLogger("evaluate")
CKPTS = ("best_val_loss.pt", "best_val_acc.pt", "last.pt")
PRIMARY = "best_val_loss.pt"
EMBED_METHODS = ("protonet", "supervised", "arcface", "siamese", "matchingnet", "baselinepp")


def test_stages(protocol: str) -> List[str]:
    return ["test"] if protocol == "A" else ["test", "hard"]


def stage_name(protocol: str, stage: str) -> str:
    return {"A": {"test": "A_test"}, "B": {"test": "B_novel", "hard": "B_hard_novel"}}[protocol][stage]


class TestEvaluator:
    def __init__(self, rm: RunModel) -> None:
        self.rm, self.cfg, self.data = rm, rm.cfg, rm.data
        self.e = rm.cfg.eval
        self.groups = rm.data.manifest.source_group.to_numpy()
        self._feat: Dict[str, Any] = {}

    # -- features for a stage ---------------------------------------------------------------
    def stage_feats(self, stage: str):
        if stage not in self._feat:
            sp, qp = self.data.episode_pools(self.rm.protocol, stage)
            idx = np.concatenate(list(sp.values()) + ([] if qp is None else list(qp.values())))
            self._feat[stage] = (sp, qp) + ((None, None) if self.rm.needs_images else self.rm.cached_features(idx))
        return self._feat[stage]

    def episode_logits(self, stage: str, ep, transductive: bool = False) -> torch.Tensor:
        _, _, feats, pos = self.stage_feats(stage)
        if not transductive:
            return self.rm.logits(ep.support, ep.query, feats, pos)
        sf, qf = feats[pos[ep.support.reshape(-1)]], feats[pos[ep.query]]
        return transductive_logits(sf, qf, ep.n_way, ep.k_shot, int(self.e.transductive_steps), *self._trans_geom())

    def _trans_geom(self):
        m = self.rm.model
        if self.rm.method in ("arcface", "baselinepp") or getattr(m, "distance", "euclidean") == "cosine":
            return "cosine", float(getattr(m, "s", 10.0))
        temp = float(m.temperature) if hasattr(m, "temperature") else 1.0
        return "euclidean", 16.0 * temp / self.rm.embedding_dim()

    def cell(self, stage: str, n_way: Any, k: int, episodes: int, seed: int, transductive: bool = False,
             cross_group: bool = False, keep: int = 0) -> Dict[str, Any]:
        sp, qp, _, _ = self.stage_feats(stage)
        samp = EpisodeSampler(sp, qp, n_way, k, int(self.e.q_query), seed=seed, groups=self.groups, name=stage_name(self.rm.protocol, stage))
        accs, top3, losses, nws, kept = [], [], [], [], []
        for i in range(episodes):
            ep = samp.sample(cross_group=cross_group)
            qy = torch.as_tensor(ep.query_y)
            lg = self.episode_logits(stage, ep, transductive).cpu().float()
            pred = lg.argmax(1)
            accs.append(float((pred == qy).float().mean()))
            top3.append(float((lg.topk(min(3, ep.n_way), 1).indices == qy.unsqueeze(1)).any(1).float().mean()))
            losses.append(float(F.cross_entropy(lg, qy)))
            nws.append(ep.n_way)
            if i < keep:
                kept.append({"classes": ep.classes.tolist(), "support": ep.support.tolist(), "query": ep.query.tolist(),
                             "query_y": ep.query_y.tolist(), "pred": pred.tolist()})
        return {"split": stage_name(self.rm.protocol, stage), "n_way": str(n_way), "effective_n_way": int(np.median(nws)),
                "k_shot": int(k), "episodes": episodes, "acc": float(np.mean(accs)), "ci95": ci95(accs),
                "top3": float(np.mean(top3)), "loss": float(np.mean(losses)), "loss_ci95": ci95(losses), "_kept": kept}

    def _n_eps(self, w: Any) -> int:
        n = int(self.e.episodes)
        if self.rm.needs_images:              # MAML adapts per episode on raw images (~1-13 s/episode on CPU)
            return int(self.e.maml_episodes)
        if self.rm.method == "relationnet" and w == "all":
            n_cls = len(self.stage_feats("test")[0])
            n = max(100, min(n, int(self.e.relation_pair_budget) // max(1, n_cls * n_cls * int(self.e.q_query))))
        if self.rm.method == "baselinepp" and w == "all":
            n = min(n, 200)
        return n

    # -- results ----------------------------------------------------------------------------
    def grid(self) -> pd.DataFrame:
        ways, shots = list(self.e.ways), list(self.e.shots)
        if self.rm.needs_images:              # MAML adapts per episode on raw images: reduced grid
            ways, shots = [5, 10, 20], [1, 5]
        rows = []
        for stage in test_stages(self.rm.protocol):
            n_cls = len(self.stage_feats(stage)[0])
            for i, w in enumerate(ways):
                if w != "all" and int(w) > n_cls:
                    continue
                for j, k in enumerate(shots):
                    t0 = time.time()
                    r = self.cell(stage, w if w == "all" else int(w), int(k), self._n_eps(w), int(self.e.seed) + 100 * i + j)
                    r.pop("_kept")
                    rows.append(r)
                    log.info("%-12s %4s-way %2d-shot: %.4f +- %.4f (%d eps, %.0fs)", r["split"], w, k, r["acc"], r["ci95"],
                             r["episodes"], time.time() - t0)
        return pd.DataFrame(rows)

    def transductive(self) -> pd.DataFrame:
        if self.rm.needs_images or self.rm.method not in EMBED_METHODS:
            return pd.DataFrame()
        rows = []
        for stage in test_stages(self.rm.protocol):
            for w in (5, 10):
                for k in (1, 5):
                    seed = int(self.e.seed) + 5000 + 10 * w + k
                    a = self.cell(stage, w, k, int(self.e.episodes), seed)
                    b = self.cell(stage, w, k, int(self.e.episodes), seed, transductive=True)
                    rows.append({"split": a["split"], "n_way": w, "k_shot": k, "episodes": a["episodes"],
                                 "inductive_acc": a["acc"], "inductive_ci95": a["ci95"], "transductive_acc": b["acc"],
                                 "transductive_ci95": b["ci95"], "gain": b["acc"] - a["acc"]})
        return pd.DataFrame(rows)

    def source_aware(self) -> pd.DataFrame:
        w, k = int(self.e.key_way), int(self.e.key_shot)
        n = int(self.e.episodes) if not self.rm.needs_images else int(self.e.maml_episodes)
        seed = int(self.e.seed) + 777
        base = self.cell("test", w, k, n, seed)
        try:
            cross = self.cell("test", w, k, n, seed, cross_group=True)
        except ValueError as exc:
            log.warning("source-aware evaluation impossible: %s", exc)
            return pd.DataFrame()
        return pd.DataFrame([{"split": base["split"], "n_way": w, "k_shot": k, "episodes": n,
                              "unconstrained_acc": base["acc"], "unconstrained_ci95": base["ci95"],
                              "cross_source_acc": cross["acc"], "cross_source_ci95": cross["ci95"],
                              "drop": base["acc"] - cross["acc"], "cross_source_n_way": cross["effective_n_way"]}])

    def fullset(self, stage: str = "test") -> Dict[str, Any]:
        """Every query of the stage classified among all its classes (K=key_shot support)."""
        fs = fullset_split(self.data, self.rm.protocol, stage, int(self.e.key_shot), int(self.e.seed) + 31)
        _, _, feats, pos = self.stage_feats(stage)
        lg = self.rm.logits(fs["support"], fs["query"], feats, pos).cpu().float()
        fs["logits"] = lg
        return fs

    def openset(self, cal: Dict[str, Any]) -> Dict[str, Any]:
        fs = fullset_split(self.data, self.rm.protocol, "test", int(self.e.key_shot), int(self.e.seed) + 41)
        enrolled, unseen = openset_split(fs["classes"], int(self.e.seed) + 41)
        _, _, feats, pos = self.stage_feats("test")
        sup = fs["support"][np.isin(fs["classes"], enrolled)]
        q_cls = fs["classes"][fs["query_y"]]
        mi, mo = np.isin(q_cls, enrolled), np.isin(q_cls, unseen)
        lg_in = self.rm.logits(sup, fs["query"][mi], feats, pos).cpu()
        s_in = lg_in.max(1).values.numpy()
        s_out = self.rm.logits(sup, fs["query"][mo], feats, pos).max(1).values.cpu().numpy()
        thr = float(cal["openset_threshold"])
        y_in = np.searchsorted(np.sort(enrolled), q_cls[mi])
        acc_in = lg_in.argmax(1).numpy() == y_in
        return {"row": {"split": stage_name(self.rm.protocol, "test"), "enrolled_classes": len(enrolled),
                        "unseen_classes": len(unseen), "enrolled_queries": int(mi.sum()), "unseen_queries": int(mo.sum()),
                        "threshold_from_val": thr, "unknown_rejection_rate": float((s_out < thr).mean()),
                        "false_reject_rate": float((s_in < thr).mean()), "auroc": auroc(s_in, s_out),
                        "closed_set_acc_enrolled": float(acc_in.mean()),
                        "accepted_and_correct_rate": float(((s_in >= thr) & acc_in).mean())},
                "scores_in": s_in, "scores_out": s_out}


def summarize_fullset(fs: Dict[str, Any], temp: float) -> Dict[str, Any]:
    lg, y = fs["logits"], torch.as_tensor(fs["query_y"])
    n = lg.shape[1]
    pred = lg.argmax(1).numpy()
    yy = y.numpy()
    p0, p1 = F.softmax(lg, 1), F.softmax(lg / temp, 1)
    correct = pred == yy
    ece0, bins0 = ece_bins(p0.max(1).values.numpy(), correct)
    ece1, bins1 = ece_bins(p1.max(1).values.numpy(), correct)
    _, _, f1m, _ = precision_recall_fscore_support(yy, pred, labels=range(n), average="macro", zero_division=0)
    _, _, f1w, _ = precision_recall_fscore_support(yy, pred, labels=range(n), average="weighted", zero_division=0)
    tk = lambda t: float((lg.topk(min(t, n), 1).indices == y.unsqueeze(1)).any(1).float().mean())  # noqa: E731
    return {"fullset_n_classes": n, "fullset_n_queries": int(len(yy)), "fullset_acc": float(correct.mean()),
            "fullset_top3": tk(3), "fullset_top5": tk(5), "macro_f1": float(f1m), "weighted_f1": float(f1w),
            "nll_uncalibrated": float(F.cross_entropy(lg, y)), "nll_calibrated": float(F.cross_entropy(lg / temp, y)),
            "ece_before": ece0, "ece_after": ece1, "temperature": temp, "_bins": (bins0, bins1), "_p1": p1}


def detailed_files(ev: TestEvaluator, fs: Dict[str, Any], sm: Dict[str, Any], out: Path) -> None:
    data = ev.data
    names = data.class_names
    cls = fs["classes"]
    lg, yy = fs["logits"], fs["query_y"]
    p1 = sm["_p1"]
    pred = lg.argmax(1).numpy()
    n = len(cls)
    loss = F.cross_entropy(lg / sm["temperature"], torch.as_tensor(yy), reduction="none").numpy()
    # predictions
    top5 = p1.topk(min(5, n), 1)
    man = data.manifest
    rows = []
    for i, qi in enumerate(fs["query"]):
        rows.append({"idx": int(qi), "path": man.path.iat[qi], "source_group": man.source_group.iat[qi],
                     "true": names[cls[yy[i]]], "pred": names[cls[pred[i]]],
                     "top5": " ".join(names[cls[j]] for j in top5.indices[i].tolist()),
                     "top5_conf": " ".join(f"{v:.3f}" for v in top5.values[i].tolist()),
                     "confidence": float(top5.values[i, 0]), "loss": float(loss[i]), "correct": bool(pred[i] == yy[i])})
    pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False, encoding="utf-8-sig")
    # per class
    pr, rc, f1, sup = precision_recall_fscore_support(yy, pred, labels=range(n), zero_division=0)
    attrs = man.groupby("class_idx").first()
    pc = pd.DataFrame({"label": [names[c] for c in cls], "class_idx": cls, "category": [attrs.category_en.get(c) for c in cls],
                       "consonant": [attrs.consonant.get(c) for c in cls], "vowel_sign": [attrs.vowel.get(c) for c in cls],
                       "precision": pr, "recall": rc, "f1": f1, "support": sup,
                       "mean_loss": [float(loss[yy == j].mean()) if (yy == j).any() else np.nan for j in range(n)]})
    pc.to_csv(out / "per_class_metrics.csv", index=False, encoding="utf-8-sig")
    # confusion
    cm = np.zeros((n, n), dtype=np.int64)
    np.add.at(cm, (yy, pred), 1)
    np.save(out / "confusion_matrix.npy", cm)
    (out / "confusion_labels.json").write_text(json.dumps([names[c] for c in cls], ensure_ascii=False), encoding="utf-8")
    pairs = []
    for a in range(n):
        for b in range(a + 1, n):
            if cm[a, b] + cm[b, a]:
                pairs.append({"letter_a": names[cls[a]], "letter_b": names[cls[b]], "a_as_b": int(cm[a, b]),
                              "b_as_a": int(cm[b, a]), "total": int(cm[a, b] + cm[b, a]),
                              "rate": float((cm[a, b] + cm[b, a]) / max(1, cm[a].sum() + cm[b].sum()))})
    pd.DataFrame(pairs).sort_values("total", ascending=False).head(50).to_csv(out / "top_confused_pairs.csv", index=False, encoding="utf-8-sig")
    # grouped accuracy
    qdf = pd.DataFrame({"correct": pred == yy, "category": pc.category.to_numpy()[yy],
                        "consonant": pc.consonant.to_numpy()[yy], "vowel_sign": pc.vowel_sign.to_numpy()[yy]})
    for col, fname in (("category", "category_accuracy.csv"), ("consonant", "consonant_row_accuracy.csv"),
                       ("vowel_sign", "vowel_sign_accuracy.csv")):
        g = qdf[qdf[col].astype(str).str.len() > 0].groupby(col).correct.agg(["mean", "count"]).reset_index()
        g.columns = [col, "accuracy", "queries"]
        g.to_csv(out / fname, index=False, encoding="utf-8-sig")
    grid = qdf[(qdf.consonant.astype(str).str.len() > 0)].groupby(["consonant", "vowel_sign"]).correct.mean().reset_index()
    grid.to_csv(out / "accuracy_consonant_x_vowel.csv", index=False, encoding="utf-8-sig")
    # calibration bins
    b0, b1 = sm["_bins"]
    pd.concat([pd.DataFrame(b0).assign(stage="before"), pd.DataFrame(b1).assign(stage="after")]).to_csv(
        out / "calibration_bins.csv", index=False)


def save_embeddings(ev: TestEvaluator, fs: Dict[str, Any], out: Path, max_q: int = 4000) -> None:
    rm, data = ev.rm, ev.data
    rng = np.random.default_rng(0)
    q = fs["query"]
    if len(q) > max_q:
        q = np.sort(rng.choice(q, max_q, replace=False))
    train_pools = data.episode_pools(rm.protocol, "train")[0]
    base = np.concatenate([rng.choice(v, min(8, len(v)), replace=False) for v in train_pools.values()])
    if rm.protocol == "A":
        base = base[:0]
    idx = np.concatenate([q, base])
    f = rm.features(idx).flatten(1).cpu().numpy().astype(np.float32)
    sup = fs["support"]
    sf = rm.features(sup.reshape(-1)).flatten(1).cpu().numpy().reshape(sup.shape[0], sup.shape[1], -1).mean(1)
    np.savez_compressed(out / "embeddings.npz", feats=f, idx=idx, labels=data.labels[idx],
                        part=np.r_[np.full(len(q), "test" if rm.protocol == "A" else "novel"), np.full(len(base), "base")],
                        proto_classes=fs["classes"], prototypes=sf.astype(np.float32))


def run(run_dir: Path, force: bool = False) -> None:
    out = run_dir / "test"
    out.mkdir(exist_ok=True)
    done_path = out / "DONE.json"
    done = json.loads(done_path.read_text(encoding="utf-8")) if done_path.exists() else {}
    data = None
    summary: List[Dict[str, Any]] = json.loads((out / "test_summary.json").read_text(encoding="utf-8")) if (out / "test_summary.json").exists() else []
    for ck in CKPTS:
        if ck in done and not force:
            log.info("%s already evaluated on test (%s) - skipping (one look per checkpoint)", ck, done[ck])
            continue
        if not (run_dir / "checkpoints" / ck).exists():
            continue
        t0 = time.time()
        rm = RunModel(run_dir, ck, data=data)
        data = rm.data
        cal = load_or_calibrate(rm)                       # validation only
        ev = TestEvaluator(rm)
        w, k = int(rm.cfg.eval.key_way), int(rm.cfg.eval.key_shot)
        key = ev.cell("test", w, k, int(rm.cfg.eval.episodes) if not rm.needs_images else int(rm.cfg.eval.maml_episodes),
                      int(rm.cfg.eval.seed) + 1, keep=6)
        fs = ev.fullset("test")
        sm = summarize_fullset(fs, float(cal["temperature"]))
        row = {"run": run_dir.name, "method": rm.method, "backbone": rm.cfg.model.backbone, "protocol": rm.protocol,
               "split": key["split"], "checkpoint": ck, "checkpoint_epoch": rm.ckpt_epoch,
               "val_loss_at_ckpt": rm.ckpt_metrics.get("val_loss"), "val_acc_at_ckpt": rm.ckpt_metrics.get("val_episode_acc"),
               f"episode_{w}w{k}s_loss": key["loss"], f"episode_{w}w{k}s_acc": key["acc"], f"episode_{w}w{k}s_ci95": key["ci95"],
               f"episode_{w}w{k}s_top3": key["top3"], **{k_: v for k_, v in sm.items() if not k_.startswith("_")},
               "params": rm.n_params(), "cpu_ms_per_image": rm.latency_ms()}
        if ck == PRIMARY:
            detailed_files(ev, fs, sm, out)
            save_embeddings(ev, fs, out)
            (out / "example_episodes.json").write_text(json.dumps(key["_kept"]), encoding="utf-8")
            ev.grid().to_csv(out / "fewshot_results.csv", index=False)
            ev.transductive().to_csv(out / "transductive.csv", index=False)
            ev.source_aware().to_csv(out / "source_aware.csv", index=False)
            os_ = ev.openset(cal)
            pd.DataFrame([os_["row"]]).to_csv(out / "openset.csv", index=False)
            np.savez(out / "openset_scores.npz", enrolled=os_["scores_in"], unseen=os_["scores_out"],
                     threshold=cal["openset_threshold"])
            row.update({f"openset_{k_}": v for k_, v in os_["row"].items() if k_ in ("auroc", "unknown_rejection_rate", "false_reject_rate")})
        row["eval_seconds"] = round(time.time() - t0, 1)
        summary = [r for r in summary if r["checkpoint"] != ck] + [row]
        log.info("%s %s: key %d-way %d-shot %.4f +- %.4f | full-set acc %.4f, macro-F1 %.4f, ECE %.3f -> %.3f",
                 run_dir.name, ck, w, k, key["acc"], key["ci95"], sm["fullset_acc"], sm["macro_f1"], sm["ece_before"], sm["ece_after"])
        (out / "test_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        pd.DataFrame(summary).to_csv(out / "test_summary.csv", index=False, encoding="utf-8-sig")
        done[ck] = time.strftime("%Y-%m-%d %H:%M:%S")
        done_path.write_text(json.dumps(done, indent=2), encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--force", action="store_true", help="re-evaluate checkpoints already marked in test/DONE.json")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    for rd in a.run_dirs:
        run(Path(rd), a.force)


if __name__ == "__main__":
    main()
