"""Every figure of the project, regenerated from STORED results only (CSV / JSON / NPY / NPZ).

    python src/visualize.py --run runs/<run_dir>     # plots of one run -> runs/<run>/plots/
    python src/visualize.py --all                    # every run + cross-run figures + reports/dashboard.html

Nothing here trains or evaluates; images for the episode / error grids come from the cached
preprocessed array data/processed/images_<size>.npy. Tamil labels are drawn as shaped glyph
images (tamil_text.render_text) because matplotlib has no Tamil text shaping.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

import tamil_text as tt  # noqa: E402
from config import ROOT, parse_run_name, setup_logging  # noqa: E402

log = logging.getLogger("visualize")
DPI = 160
RUNS, REPORTS = ROOT / "runs", ROOT / "reports"
FIG = REPORTS / "figures"
PROC = ROOT / "data" / "processed"
NICE = {"protonet": "ProtoNet", "matchingnet": "Matching Net", "relationnet": "Relation Net", "siamese": "Siamese (triplet)",
        "arcface": "ArcFace (cosine protos)", "baselinepp": "Baseline++", "maml": "FO-MAML", "supervised": "Supervised CE"}
CAT_COLORS = {"vowel": "#d62728", "consonant": "#1f77b4", "consonant-vowel": "#7f7f7f", "aytham": "#2ca02c", "variant": "#9467bd"}


# --- helpers -----------------------------------------------------------------------------------
def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def _read_csv(p: Path) -> Optional[pd.DataFrame]:
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        df = pd.read_csv(p, encoding="utf-8-sig", keep_default_na=False, na_values=[""])
    except pd.errors.EmptyDataError:
        return None
    return df if len(df) else None


def _json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


_IMG_CACHE: Dict[int, np.ndarray] = {}


def images(size: int = 48) -> np.ndarray:
    if size not in _IMG_CACHE:
        _IMG_CACHE[size] = np.load(PROC / f"images_{size}.npy", mmap_mode="r")
    return _IMG_CACHE[size]


def class_table() -> pd.DataFrame:
    cls = pd.DataFrame(_json(PROC / "class_names.json"))
    parts = cls.label.map(tt.parse_label)
    cls["consonant"] = parts.map(lambda p: p["consonant"] or "")
    cls["vowel"] = parts.map(lambda p: p["vowel"] if p["vowel"] is not None else "-")
    cls["kind"] = parts.map(lambda p: p["kind"])
    return cls


def run_label(rd: Path) -> str:
    pr = parse_run_name(rd.name)
    method, backbone, proto, tag = pr["method"], pr["backbone"], pr["protocol"], pr["tag"]
    return f"{NICE.get(method, method)} / {backbone} / {proto}" + (f" [{tag}]" if tag and tag != "final" else "")


def _mark_epochs(ax, h: Dict[str, Any], df: pd.DataFrame) -> None:
    for e in (h or {}).get("overfit_epochs", []):
        ax.axvspan(e - 0.5, e + 0.5, color="tab:red", alpha=0.12, lw=0)
    for e in (h or {}).get("plateau_epochs", []):
        ax.axvspan(e - 0.5, e + 0.5, color="tab:orange", alpha=0.08, lw=0)
    be = (h or {}).get("best_val_loss_epoch")
    if be:
        ax.axvline(be, color="k", ls="--", lw=1, alpha=0.7)


# --- per-run training plots (1-6, 8) ----------------------------------------------------------
def plot_training(rd: Path, out: Path) -> List[Path]:
    df = _read_csv(rd / "metrics_epoch.csv")
    if df is None:
        return []
    h = _json(rd / "history.json") or {}
    title = run_label(rd)
    made = []
    # 1 loss curves (linear + log)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, logy in zip(axes, (False, True)):
        ax.plot(df.epoch, df.train_loss, "-o", ms=3, label="train loss")
        ax.plot(df.epoch, df.val_loss, "-o", ms=3, label="val loss (fixed episodes)")
        _mark_epochs(ax, h, df)
        if logy:
            ax.set_yscale("log")
        ax.set_xlabel("epoch"); ax.set_ylabel("loss" + (" (log)" if logy else "")); ax.grid(alpha=.3)
    be = h.get("best_val_loss_epoch")
    axes[0].legend(title=f"best val loss {h.get('best_val_loss', float('nan')):.4f} @ epoch {be}" if be else None, fontsize=8)
    fig.suptitle(f"{title} — loss (dashed: best val-loss epoch; red: overfitting flag; orange: plateau)", fontsize=10)
    made.append(_save(fig, out / "loss_curve.png"))
    # 2 accuracy with CI band
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df.epoch, 100 * df.train_episode_acc, "-o", ms=3, label="train episode acc (augmented)")
    ax.plot(df.epoch, 100 * df.val_episode_acc, "-o", ms=3, label="val episode acc")
    ax.fill_between(df.epoch, 100 * (df.val_episode_acc - df.val_acc_ci95), 100 * (df.val_episode_acc + df.val_acc_ci95),
                    alpha=.25, color="tab:orange", label="val 95% CI")
    if "val_top3_acc" in df:
        ax.plot(df.epoch, 100 * df.val_top3_acc, ":", label="val top-3")
    _mark_epochs(ax, h, df)
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    ax.set_title(f"{title} — episode accuracy", fontsize=10)
    made.append(_save(fig, out / "accuracy_curve.png"))
    # 3 diagnostics
    cols = [("lr", "learning rate", True), ("grad_norm_mean", "grad norm (mean / max)", False),
            ("clipped_frac", "clipped-step fraction", False), ("temperature", "ProtoNet temperature", False),
            ("emb_norm", "embedding norm", False), ("epoch_time_s", "epoch time (s)", False)]
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, (c, name, logy) in zip(axes.ravel(), cols):
        if c in df and df[c].notna().any():
            ax.plot(df.epoch, df[c], "-o", ms=3)
            if c == "grad_norm_mean" and "grad_norm_max" in df:
                ax.plot(df.epoch, df.grad_norm_max, "--", alpha=.7)
            if logy:
                ax.set_yscale("log")
        else:
            ax.text(0.5, 0.5, "n/a for this method", ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title(name, fontsize=9); ax.grid(alpha=.3); ax.set_xlabel("epoch")
    fig.suptitle(f"{title} — optimisation diagnostics", fontsize=10); fig.tight_layout()
    made.append(_save(fig, out / "diagnostics.png"))
    # 4 separation
    if "sep_ratio" in df and df.sep_ratio.notna().any():
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df.epoch, df.inter_dist, "-o", ms=3, label="inter-class prototype distance")
        ax.plot(df.epoch, df.intra_spread, "-o", ms=3, label="intra-class spread")
        ax2 = ax.twinx()
        ax2.plot(df.epoch, df.sep_ratio, "-s", ms=3, color="tab:green", label="separation ratio")
        ax2.set_ylabel("ratio", color="tab:green")
        ax.set_xlabel("epoch"); ax.set_ylabel("distance (val embeddings)"); ax.grid(alpha=.3)
        ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="lower right", fontsize=8)
        ax.set_title(f"{title} — prototype separation", fontsize=10)
        made.append(_save(fig, out / "separation.png"))
    # 5 step loss
    st = _read_csv(rd / "metrics_step.csv")
    if st is not None:
        fig, ax = plt.subplots(figsize=(8, 3.6))
        ax.plot(st.step, st.loss, alpha=.35, lw=.8, label="episode loss")
        ax.plot(st.step, st.running_loss, lw=1.5, label="running average (EMA 0.95)")
        ax.plot(st.step, st.loss.rolling(20, min_periods=1).mean(), lw=1.2, label="moving average (20 logs)")
        ax.set_xlabel("step (episode)"); ax.set_ylabel("loss"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title(f"{title} — step-level loss", fontsize=10)
        made.append(_save(fig, out / "step_loss.png"))
    # 6 loss components
    if "loss_pos" in df and df.loss_pos.notna().any():
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        method = parse_run_name(rd.name)["method"]
        lab = ("1 - cos(target)", "relu(cos hardest non-target)", "mean target cosine", "mean hardest non-target cosine") \
            if method in ("arcface",) else ("hardest-positive distance", "negative hinge term", "mean positive distance", "mean negative distance")
        axes[0].plot(df.epoch, df.loss_pos, "-o", ms=3, label=lab[0])
        axes[0].plot(df.epoch, df.loss_neg, "-o", ms=3, label=lab[1])
        if "active_frac" in df and df.active_frac.notna().any():
            axes[0].plot(df.epoch, df.active_frac, "--", label="active (violating) fraction")
        axes[1].plot(df.epoch, df.dist_pos, "-o", ms=3, label=lab[2])
        axes[1].plot(df.epoch, df.dist_neg, "-o", ms=3, label=lab[3])
        for ax in axes:
            ax.grid(alpha=.3); ax.legend(fontsize=8); ax.set_xlabel("epoch")
        axes[0].set_title("loss components"); axes[1].set_title("positive vs negative distance / margin")
        fig.suptitle(title, fontsize=10)
        made.append(_save(fig, out / "loss_components.png"))
    # 8 generalisation gap
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(df.epoch, df.val_loss - df.train_loss, "-o", ms=3)
    ax.axhline(0, color="k", lw=.8)
    _mark_epochs(ax, h, df)
    ax.set_xlabel("epoch"); ax.set_ylabel("val loss - train loss"); ax.grid(alpha=.3)
    ax.set_title(f"{title} — generalisation gap (negative: train episodes are augmented and harder)", fontsize=9)
    made.append(_save(fig, out / "generalization_gap.png"))
    return made


# --- per-run test plots (10, 12-15, 17-19) ------------------------------------------------------
def plot_fewshot(t: Path, out: Path, title: str) -> List[Path]:
    fr = _read_csv(t / "fewshot_results.csv")
    if fr is None:
        return []
    fr["n_way"] = fr.n_way.astype(str)
    made = []
    splits = list(dict.fromkeys(fr.split))
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        for w, g in fr[fr.split == sp].groupby("n_way", sort=False):
            g = g.sort_values("k_shot")
            eff = int(g.effective_n_way.median())
            ax.errorbar(g.k_shot, 100 * g.acc, yerr=100 * g.ci95, marker="o", capsize=3,
                        label=f"{w}-way" + (f" (= {eff})" if w == "all" else ""))
        ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10]); ax.set_xticklabels(["1", "2", "3", "5", "10"])
        ax.set_xlabel("K (shots)"); ax.set_ylabel("test accuracy (%)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title(sp)
    fig.suptitle(f"{title} — accuracy vs K-shot (±95% CI)", fontsize=10)
    made.append(_save(fig, out / "fewshot_vs_shot.png"))
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), squeeze=False)
    for ax, sp in zip(axes[0], splits):
        s = fr[fr.split == sp]
        for k, g in s.groupby("k_shot"):
            ax.errorbar(range(len(g)), 100 * g.acc, yerr=100 * g.ci95, marker="o", capsize=3, label=f"{k}-shot")
            ax.set_xticks(range(len(g))); ax.set_xticklabels([f"{w}\n({e})" for w, e in zip(g.n_way, g.effective_n_way)])
        ax.set_xlabel("N-way (effective)"); ax.set_ylabel("test accuracy (%)"); ax.grid(alpha=.3); ax.legend(fontsize=8)
        ax.set_title(sp)
    fig.suptitle(f"{title} — accuracy vs N-way (±95% CI)", fontsize=10)
    made.append(_save(fig, out / "fewshot_vs_way.png"))
    return made


def plot_confusion(t: Path, out: Path, title: str) -> List[Path]:
    if not (t / "confusion_matrix.npy").exists():
        return []
    cm = np.load(t / "confusion_matrix.npy")
    labels = _json(t / "confusion_labels.json")
    n = len(labels)
    made = []
    rown = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    big = max(8, n * 0.11)
    fig, ax = plt.subplots(figsize=(big + 2, big))
    im = ax.imshow(rown, cmap="viridis", vmin=0, vmax=1, interpolation="nearest")
    zoom = 0.28 if n > 100 else 0.45
    tt.tamil_ticks(ax, labels, "x", px=22, zoom=zoom, offset=0.002)
    tt.tamil_ticks(ax, labels, "y", px=22, zoom=zoom, offset=0.002)
    ax.set_title(f"{title} — row-normalised confusion matrix ({n} classes)", fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, fraction=0.03)
    made.append(_save(fig, out / "confusion_full.png"))
    pairs = _read_csv(t / "top_confused_pairs.csv")
    if pairs is not None:
        sel: List[str] = []
        for _, r in pairs.iterrows():
            for x in (r.letter_a, r.letter_b):
                if x not in sel:
                    sel.append(x)
            if len(sel) >= 20:
                break
        ids = [labels.index(x) for x in sel if x in labels]
        sub = cm[np.ix_(ids, ids)]
        fig, ax = plt.subplots(figsize=(9, 8))
        sns.heatmap(sub, annot=True, fmt="d", cmap="rocket_r", cbar=False, ax=ax, annot_kws={"size": 8})
        tt.tamil_ticks(ax, [labels[i] for i in ids], "x", positions=np.arange(len(ids)) + 0.5, px=26, zoom=0.36)
        tt.tamil_ticks(ax, [labels[i] for i in ids], "y", positions=np.arange(len(ids)) + 0.5, px=26, zoom=0.5)
        ax.set_title(f"{title} — confusion among the most-confused letters (counts; rows = true)", fontsize=10, pad=14)
        made.append(_save(fig, out / "confusion_zoom.png"))
        top = pairs.head(15)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(range(len(top))[::-1], top.total, color="tab:red")
        ax.set_yticks(range(len(top))[::-1]); ax.set_yticklabels([""] * len(top))
        for i, (_, r) in enumerate(top.iterrows()):
            tt.mpl_text(ax, -0.01, len(top) - 1 - i, f"{r.letter_a} - {r.letter_b}", px=24, zoom=0.5,
                        xycoords=ax.get_yaxis_transform(), box_alignment=(1.0, 0.5))
        ax.set_xlabel("confusions (both directions)"); ax.set_title(f"{title} — top confused pairs", fontsize=10)
        made.append(_save(fig, out / "top_confused_pairs.png"))
    return made


def plot_per_class(t: Path, out: Path, title: str) -> List[Path]:
    pc = _read_csv(t / "per_class_metrics.csv")
    made = []
    if pc is not None:
        w = pc.sort_values("f1").head(30)
        fig, ax = plt.subplots(figsize=(12, 4.2))
        ax.bar(range(len(w)), w.f1, color=[CAT_COLORS.get(c, "gray") for c in w.category])
        tt.tamil_ticks(ax, w.label.tolist(), "x", px=26, zoom=0.5)
        ax.set_ylabel("F1"); ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.3)
        ax.set_title(f"{title} — 30 weakest letters by test F1 (colour = category)", fontsize=10)
        made.append(_save(fig, out / "per_class_f1_worst30.png"))
    pr = _read_csv(t / "predictions.csv")
    if pr is not None:
        fig, ax = plt.subplots(figsize=(7, 3.8))
        bins = np.linspace(0, max(1e-3, np.percentile(pr.loss, 99.5)), 60)
        ax.hist(pr[pr.correct].loss, bins=bins, alpha=.6, label="correct")
        ax.hist(pr[~pr.correct].loss, bins=bins, alpha=.6, label="misclassified")
        ax.set_yscale("log"); ax.set_xlabel("per-sample test loss (calibrated NLL)"); ax.set_ylabel("count")
        ax.legend(); ax.grid(alpha=.3); ax.set_title(f"{title} — per-sample loss histogram", fontsize=10)
        made.append(_save(fig, out / "loss_histogram.png"))
    grid = _read_csv(t / "accuracy_consonant_x_vowel.csv")
    if grid is not None:
        piv = grid.pivot_table(index="consonant", columns="vowel_sign", values="correct")
        order_c = [c for c in tt.CONSONANTS if c in piv.index] if hasattr(tt, "CONSONANTS") else list(piv.index)
        piv = piv.reindex(order_c)
        fig, ax = plt.subplots(figsize=(max(6, 0.8 * piv.shape[1] + 2), max(5, 0.45 * piv.shape[0] + 1)))
        sns.heatmap(100 * piv, annot=True, fmt=".0f", cmap="RdYlGn", vmin=0, vmax=100, ax=ax, cbar_kws={"label": "accuracy %"},
                    annot_kws={"size": 7})
        tt.tamil_ticks(ax, [tt.vowel_sign_name(v) if hasattr(tt, "vowel_sign_name") else str(v) for v in piv.columns], "x",
                       positions=np.arange(piv.shape[1]) + 0.5, px=22, zoom=0.45)
        tt.tamil_ticks(ax, list(piv.index), "y", positions=np.arange(piv.shape[0]) + 0.5, px=24, zoom=0.5)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_title(f"{title} — accuracy by consonant row × vowel sign (blank = no test class)", fontsize=10, pad=16)
        made.append(_save(fig, out / "accuracy_consonant_x_vowel.png"))
    cat = _read_csv(t / "category_accuracy.csv")
    if cat is not None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.bar(cat.category, 100 * cat.accuracy, color=[CAT_COLORS.get(c, "gray") for c in cat.category])
        for i, (a, q) in enumerate(zip(cat.accuracy, cat.queries)):
            ax.text(i, 100 * a + 1, f"{100 * a:.1f}%\n(n={q})", ha="center", fontsize=8)
        ax.set_ylim(0, 110); ax.set_ylabel("accuracy (%)"); ax.grid(axis="y", alpha=.3)
        ax.set_title(f"{title} — accuracy by category", fontsize=10)
        made.append(_save(fig, out / "category_accuracy.png"))
    return made


def plot_embeddings(t: Path, out: Path, title: str, max_points: int = 2500) -> List[Path]:
    p = t / "embeddings.npz"
    if not p.exists():
        return []
    from sklearn.manifold import TSNE
    z = np.load(p, allow_pickle=False)
    f, lab, part = z["feats"], z["labels"], z["part"].astype(str)
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(len(f), min(max_points, len(f)), replace=False))
    f, lab, part = f[keep], lab[keep], part[keep]
    cache = t / "tsne_2d.npy"
    if cache.exists() and np.load(cache).shape[0] == len(f):
        xy = np.load(cache)
    else:
        xy = TSNE(2, init="pca", perplexity=30, random_state=0, learning_rate="auto").fit_transform(f.astype(np.float64))
        np.save(cache, xy)
    cls = class_table()
    kinds = cls.kind.to_numpy()[lab]
    kinds = np.where(kinds == "compound", "consonant-vowel", kinds)
    made = []
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    markers = {"base": "x", "novel": "o", "test": "o"}
    for k in np.unique(kinds):
        for prt in np.unique(part):
            m = (kinds == k) & (part == prt)
            if m.any():
                axes[0].scatter(xy[m, 0], xy[m, 1], s=8, c=CAT_COLORS.get(k, "gray"), marker=markers.get(prt, "o"),
                                alpha=.6, label=f"{k} ({prt})")
    axes[0].legend(fontsize=7, markerscale=2); axes[0].set_title("by category (x = base classes, o = test/novel)")
    cons = cls.consonant.to_numpy()[lab]
    vow = cls.vowel.to_numpy()[lab]
    for ax, key, name in ((axes[1], cons, "consonant row"), (axes[2], vow, "vowel sign")):
        vals = sorted(set(key))
        pal = sns.color_palette("tab20", len(vals))
        for v, col in zip(vals, pal):
            m = key == v
            ax.scatter(xy[m, 0], xy[m, 1], s=6, color=col, alpha=.55)
        ax.set_title(f"coloured by {name}")
    # annotate class centroids with shaped Tamil labels
    for c in np.unique(lab):
        m = lab == c
        if m.sum() >= 3:
            cx, cy = np.median(xy[m], 0)
            tt.mpl_text(axes[0], cx, cy, cls.label.iat[int(c)], px=22, zoom=0.42, color=(0, 0, 0))
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(f"{title} — t-SNE of embeddings (labels at class medians)", fontsize=11)
    made.append(_save(fig, out / "tsne.png"))
    # 14 prototype distance heatmap for the most-confused letters
    pairs = _read_csv(t / "top_confused_pairs.csv")
    if pairs is not None and "prototypes" in z.files:
        names = list(cls.label)
        pcls = list(z["proto_classes"])
        sel: List[int] = []
        for _, r in pairs.iterrows():
            for x in (r.letter_a, r.letter_b):
                i = names.index(x) if x in names else None
                if i is not None and i in pcls and i not in sel:
                    sel.append(i)
            if len(sel) >= 16:
                break
        P = z["prototypes"][[pcls.index(i) for i in sel]]
        D = np.sqrt(np.maximum(((P[:, None] - P[None]) ** 2).sum(-1), 0))
        fig, ax = plt.subplots(figsize=(8.5, 7.5))
        sns.heatmap(D, annot=True, fmt=".1f", cmap="mako", ax=ax, annot_kws={"size": 7}, cbar_kws={"label": "Euclidean distance"})
        tt.tamil_ticks(ax, [names[i] for i in sel], "x", positions=np.arange(len(sel)) + 0.5, px=26, zoom=0.38)
        tt.tamil_ticks(ax, [names[i] for i in sel], "y", positions=np.arange(len(sel)) + 0.5, px=26, zoom=0.5)
        ax.set_title(f"{title} — prototype distances of the most-confused letters", fontsize=10, pad=14)
        made.append(_save(fig, out / "prototype_distances.png"))
    return made


def _imshow(ax, img, border=None):
    ax.imshow(255 - np.asarray(img), cmap="gray", vmin=0, vmax=255)
    ax.set_xticks([]); ax.set_yticks([])
    if border:
        for s in ax.spines.values():
            s.set_edgecolor(border); s.set_linewidth(2.5)


def plot_episodes(t: Path, out: Path, title: str, n_ep: int = 3) -> List[Path]:
    eps = _json(t / "example_episodes.json")
    if not eps:
        return []
    imgs, names = images(), list(class_table().label)
    made = []
    for e_i, ep in enumerate(eps[:n_ep]):
        sup, q, qy, pred = np.array(ep["support"]), np.array(ep["query"]), np.array(ep["query_y"]), np.array(ep["pred"])
        n, k = sup.shape
        qpc = int(np.bincount(qy).max())
        fig, axes = plt.subplots(n, k + qpc + 1, figsize=(1.15 * (k + qpc + 1), 1.35 * n), squeeze=False)
        for j in range(n):
            for s in range(k):
                _imshow(axes[j, s], imgs[sup[j, s]])
            axes[j, k].axis("off")
            tt.mpl_text(axes[j, k], 0.5, 0.5, names[ep["classes"][j]], px=30, zoom=0.5, xycoords="axes fraction")
            qs = np.where(qy == j)[0]
            for c in range(qpc):
                ax = axes[j, k + 1 + c]
                if c < len(qs):
                    ok = pred[qs[c]] == j
                    _imshow(ax, imgs[q[qs[c]]], "green" if ok else "red")
                    if not ok:
                        tt.mpl_text(ax, 0.5, -0.02, ">" + names[ep["classes"][pred[qs[c]]]], px=22, zoom=0.45,
                                    color=(200, 0, 0), xycoords="axes fraction", box_alignment=(0.5, 1.0))
                else:
                    ax.axis("off")
        axes[0, 0].set_title("support", fontsize=8, loc="left")
        axes[0, k + 1].set_title("queries (green = correct, red = wrong → predicted)", fontsize=8, loc="left")
        fig.suptitle(f"{title} — test episode {e_i + 1}: {n}-way {k}-shot", fontsize=10)
        fig.tight_layout()
        made.append(_save(fig, out / f"episode_{e_i + 1}.png"))
    return made


def plot_misclassified(t: Path, out: Path, title: str, n: int = 40) -> List[Path]:
    pr = _read_csv(t / "predictions.csv")
    if pr is None:
        return []
    bad = pr[~pr.correct.astype(bool)].sort_values("confidence", ascending=False).head(n)
    if bad.empty:
        return []
    imgs = images()
    cols = 8
    rows = int(np.ceil(len(bad) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.1), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (_, r) in zip(axes.ravel(), bad.iterrows()):
        ax.axis("on")
        _imshow(ax, imgs[int(r.idx)], "red")
        tt.mpl_text(ax, 0.5, 1.02, f"{r.true} > {r.pred}", px=26, zoom=0.45, xycoords="axes fraction", box_alignment=(0.5, 0.0))
        ax.set_xlabel(f"{r.confidence:.2f} · {r.source_group}", fontsize=7)
    fig.suptitle(f"{title} — most confident test errors (true → predicted, confidence · source)", fontsize=10)
    fig.tight_layout()
    return [_save(fig, out / "misclassified.png")]


def plot_openset(t: Path, out: Path, title: str) -> List[Path]:
    p = t / "openset_scores.npz"
    if not p.exists():
        return []
    from sklearn.metrics import roc_curve
    z = np.load(p)
    s_in, s_out, thr = z["enrolled"], z["unseen"], float(z["threshold"])
    y = np.r_[np.ones(len(s_in)), np.zeros(len(s_out))]
    fpr, tpr, _ = roc_curve(y, np.r_[s_in, s_out])
    os_ = _read_csv(t / "openset.csv")
    au = float(os_.auroc.iat[0]) if os_ is not None else float("nan")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    axes[0].plot(fpr, tpr, label=f"AUROC {au:.3f}")
    axes[0].plot([0, 1], [0, 1], "k:", lw=.8)
    if os_ is not None:
        axes[0].scatter([1 - os_.unknown_rejection_rate.iat[0]], [1 - os_.false_reject_rate.iat[0]], c="red", zorder=3,
                        label="val-calibrated threshold")
    axes[0].set_xlabel("unseen accepted (FPR)"); axes[0].set_ylabel("enrolled accepted (TPR)"); axes[0].legend(); axes[0].grid(alpha=.3)
    axes[0].set_title("open-set ROC (score = max logit)")
    bins = np.linspace(min(s_in.min(), s_out.min()), max(s_in.max(), s_out.max()), 60)
    axes[1].hist(s_in, bins=bins, alpha=.6, density=True, label="enrolled letters")
    axes[1].hist(s_out, bins=bins, alpha=.6, density=True, label="unseen letters")
    axes[1].axvline(thr, color="red", ls="--", label="threshold (from val)")
    axes[1].legend(); axes[1].set_xlabel("max logit"); axes[1].grid(alpha=.3); axes[1].set_title("score distributions")
    fig.suptitle(f"{title} — open-set rejection", fontsize=10)
    return [_save(fig, out / "openset.png")]


def plot_reliability(t: Path, out: Path, title: str) -> List[Path]:
    b = _read_csv(t / "calibration_bins.csv")
    if b is None:
        return []
    ts = (_json(t / "test_summary.json") or [{}])
    row = next((r for r in ts if r.get("checkpoint") == "best_val_loss.pt"), ts[0] if ts else {})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, stage, ece in zip(axes, ("before", "after"), (row.get("ece_before"), row.get("ece_after"))):
        s = b[(b.stage == stage) & (b["count"] > 0)]
        mid = (s.lo + s.hi) / 2
        ax.bar(mid, s.acc, width=(s.hi - s.lo) * .95, alpha=.7, label="accuracy", edgecolor="k")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.plot(mid, s.conf, "o-", color="tab:red", ms=3, label="mean confidence")
        ax.set_xlabel("confidence"); ax.set_title(f"{stage} temperature scaling — ECE {ece:.3f}" if ece is not None else stage)
        ax.legend(fontsize=8); ax.grid(alpha=.3)
    axes[0].set_ylabel("accuracy")
    fig.suptitle(f"{title} — reliability diagram (T = {row.get('temperature', float('nan')):.3f}, fitted on val)", fontsize=10)
    return [_save(fig, out / "reliability.png")]


def plot_run(rd: Path) -> List[Path]:
    """All per-run figures (training + test) for one run directory."""
    tt.setup_matplotlib()
    out = rd / "plots"
    title = run_label(rd)
    made = plot_training(rd, out)
    t = rd / "test"
    for fn in (plot_fewshot, plot_confusion, plot_per_class, plot_embeddings, plot_episodes, plot_misclassified,
               plot_openset, plot_reliability):
        try:
            made += fn(t, out, title)
        except Exception as exc:  # keep going: one broken input should not block the rest
            log.exception("%s failed for %s: %s", fn.__name__, rd.name, exc)
    log.info("%s: %d figures", rd.name, len(made))
    return made


# --- cross-run figures ------------------------------------------------------------------------
def list_runs() -> List[Path]:
    return sorted(p for p in RUNS.iterdir() if (p / "metrics_epoch.csv").exists())


def tag_of(rd: Path) -> str:
    return parse_run_name(rd.name)["tag"]


def overlay(runs: List[Path], fname: str, title: str, label_fn=run_label, col: str = "val_loss") -> Optional[Path]:
    if not runs:
        return None
    fig, ax = plt.subplots(figsize=(9, 5))
    for rd in runs:
        df = _read_csv(rd / "metrics_epoch.csv")
        if df is not None:
            ax.plot(df.epoch, df[col], "-o", ms=2.5, label=label_fn(rd))
    ax.set_xlabel("epoch"); ax.set_ylabel(col.replace("_", " ")); ax.grid(alpha=.3); ax.legend(fontsize=7)
    if col.endswith("loss"):
        ax.set_yscale("log")
    ax.set_title(title, fontsize=10)
    return _save(fig, FIG / fname)


def cross_run_figures() -> List[Path]:
    tt.setup_matplotlib()
    runs = list_runs()
    made: List[Path] = []
    finals = [r for r in runs if tag_of(r).startswith(("final", "arcft"))]
    for proto in ("A", "B"):
        sel = [r for r in finals if parse_run_name(r.name)["protocol"] == proto and "seed" not in tag_of(r)]
        for col in ("val_loss", "val_episode_acc"):
            p = overlay(sel, f"overlay_{col}_protocol{proto}.png", f"Protocol {proto}: {col} of every method (final runs)", col=col)
            made += [p] if p else []
    lr = [r for r in runs if tag_of(r).startswith("lr")]
    p = overlay(lr, "lr_comparison.png", "LR comparison (ProtoNet, Protocol B, 8 epochs): validation loss",
                label_fn=lambda r: "lr = " + tag_of(r)[2:])
    made += [p] if p else []
    abl = [r for r in runs if tag_of(r).startswith("abl")] + [r for r in lr if tag_of(r) == "lr1e-3"]
    p = overlay(abl, "ablations_val_loss.png", "Ablations (ProtoNet, Protocol B, 8 epochs): validation loss",
                label_fn=lambda r: "baseline (20-way 5-shot, lr 1e-3)" if tag_of(r) == "lr1e-3" else tag_of(r)[4:])
    made += [p] if p else []
    es = [r for r in runs if tag_of(r) == "epochstudy"]
    if es:
        made += plot_epoch_study(es[-1])
    made += plot_lr_finder()
    made += plot_data_figures()
    comp = _read_csv(REPORTS / "model_comparison.csv")
    if comp is not None:
        made += plot_comparison(comp)
    made += plot_seeds()
    return made


def plot_epoch_study(rd: Path) -> List[Path]:
    df = _read_csv(rd / "metrics_epoch.csv")
    h = _json(rd / "history.json") or {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
    axes[0].plot(df.epoch, df.train_loss, "-o", ms=3, label="train")
    axes[0].plot(df.epoch, df.val_loss, "-o", ms=3, label="val")
    axes[0].plot(df.epoch, df.val_loss.rolling(3, center=True, min_periods=1).mean(), "k:", label="val (3-epoch smooth)")
    _mark_epochs(axes[0], h, df)
    axes[0].set_yscale("log"); axes[0].legend(); axes[0].grid(alpha=.3); axes[0].set_xlabel("epoch")
    axes[0].set_title("loss (dashed: val-loss minimum)")
    axes[1].plot(df.epoch, 100 * df.val_episode_acc, "-o", ms=3, color="tab:orange")
    axes[1].fill_between(df.epoch, 100 * (df.val_episode_acc - df.val_acc_ci95), 100 * (df.val_episode_acc + df.val_acc_ci95), alpha=.25,
                         color="tab:orange")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("val episode accuracy (%)"); axes[1].grid(alpha=.3)
    fig.suptitle(f"Epoch-budget study: {run_label(rd)} ({len(df)} epochs, early stopping off)", fontsize=10)
    return [_save(fig, FIG / "epoch_study.png")]


def plot_lr_finder() -> List[Path]:
    made = []
    for p in sorted((REPORTS / "lr_finder").glob("*.csv")) if (REPORTS / "lr_finder").exists() else []:
        df = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(df.lr, df.loss, alpha=.35, label="loss")
        ax.plot(df.lr, df.smoothed_loss, label="smoothed")
        i = int(np.argmin(np.gradient(df.smoothed_loss.to_numpy(), np.log10(df.lr.to_numpy()))[3:-1])) + 3 if len(df) > 8 else 0
        ax.axvline(df.lr.iat[i], color="red", ls="--", label=f"steepest descent {df.lr.iat[i]:.1e}")
        ax.set_xscale("log"); ax.set_xlabel("learning rate"); ax.set_ylabel("episode loss"); ax.grid(alpha=.3); ax.legend()
        ax.set_title(f"LR range test — {p.stem}", fontsize=10)
        made.append(_save(fig, FIG / f"lr_finder_{p.stem}.png"))
    return made


def plot_data_figures() -> List[Path]:
    made = []
    cc = _read_csv(REPORTS / "audit_class_counts.csv")
    if cc is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        bins = np.arange(0, max(cc.raw.max(), cc.clean.max()) + 5, 4)
        axes[0].hist(cc.raw, bins=bins, alpha=.55, label="before dedupe")
        axes[0].hist(cc.clean, bins=bins, alpha=.55, label="after dedupe")
        axes[0].set_xlabel("images per class"); axes[0].set_ylabel("classes"); axes[0].legend(); axes[0].grid(alpha=.3)
        axes[0].set_title(f"class-size histogram ({len(cc)} classes)")
        s = cc.sort_values("clean")
        axes[1].bar(range(len(s)), s.clean, color="tab:blue", width=1.0)
        axes[1].bar(range(len(s)), s.raw - s.clean, bottom=s.clean, color="tab:red", width=1.0, label="removed duplicates")
        axes[1].set_xlabel("classes (sorted)"); axes[1].set_ylabel("images"); axes[1].legend(); axes[1].grid(alpha=.3)
        axes[1].set_title("class imbalance (sorted counts)")
        made.append(_save(fig, FIG / "class_counts.png"))
    man = _read_csv(PROC / "manifest.csv")
    if man is not None:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        vc = man.source_group.value_counts()
        axes[0].bar(vc.index, vc.values, color="tab:purple")
        for i, v in enumerate(vc.values):
            axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
        axes[0].set_yscale("log"); axes[0].set_title("source groups (from file names)"); axes[0].grid(axis="y", alpha=.3)
        ct = pd.crosstab(man.split_A, man.source_group, normalize="index") * 100
        ct.plot.bar(stacked=True, ax=axes[1], colormap="tab20", width=.7)
        axes[1].set_ylabel("% of pool"); axes[1].set_title("source-group share per Protocol A pool"); axes[1].legend(fontsize=7, ncol=2)
        made.append(_save(fig, FIG / "source_groups.png"))
    return made


def plot_comparison(comp: pd.DataFrame) -> List[Path]:
    made = []
    c = comp[comp.test_acc_5w5s.notna()].copy()
    if c.empty:
        return made
    fig, ax = plt.subplots(figsize=(10, 6))
    for proto, mk in (("A", "s"), ("B", "o")):
        s = c[c.protocol == proto]
        sc = ax.scatter(s.params / 1e3, 100 * s.test_acc_5w5s, s=40 + 40 * s.cpu_ms_per_image, marker=mk, alpha=.6,
                        c=s.cpu_ms_per_image, cmap="viridis", edgecolor="k", label=f"Protocol {proto}")
        for _, r in s.iterrows():
            ax.annotate(f"{NICE.get(r.method, r.method)}\n{r.backbone} {r.tag}".strip(), (r.params / 1e3, 100 * r.test_acc_5w5s),
                        fontsize=6.5, xytext=(4, 4), textcoords="offset points")
    plt.colorbar(sc, ax=ax, label="CPU ms / image (bubble size)")
    ax.set_xscale("log"); ax.set_xlabel("parameters (thousands, log)"); ax.set_ylabel("test accuracy 5-way 5-shot (%)")
    ax.legend(); ax.grid(alpha=.3); ax.set_title("Accuracy vs size vs latency (test)", fontsize=10)
    made.append(_save(fig, FIG / "model_bubble.png"))
    # learning curves (accuracy vs shots) for every method, 5-way, per split
    rows = []
    for rd in list_runs():
        fr = _read_csv(rd / "test" / "fewshot_results.csv")
        if fr is None:
            continue
        fr = fr[fr.n_way.astype(str) == "5"].assign(run=rd.name, label=run_label(rd))
        rows.append(fr)
    if rows:
        allfr = pd.concat(rows)
        splits = list(dict.fromkeys(allfr.split))
        fig, axes = plt.subplots(1, len(splits), figsize=(6.2 * len(splits), 4.6), squeeze=False, sharey=True)
        for ax, sp in zip(axes[0], splits):
            for lab, g in allfr[allfr.split == sp].groupby("label"):
                g = g.sort_values("k_shot")
                ax.errorbar(g.k_shot, 100 * g.acc, yerr=100 * g.ci95, marker="o", ms=3, capsize=2, label=lab)
            ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10]); ax.set_xticklabels(["1", "2", "3", "5", "10"])
            ax.set_title(f"{sp} — 5-way"); ax.set_xlabel("K (shots)"); ax.grid(alpha=.3)
        axes[0, 0].set_ylabel("test accuracy (%)")
        axes[0, -1].legend(fontsize=6.5, loc="lower right")
        fig.suptitle("Learning curves: accuracy vs number of shots, every method", fontsize=10)
        made.append(_save(fig, FIG / "learning_curves_all_methods.png"))
        # Protocol A vs B novel vs B hard-novel (5-way 1/5-shot) for methods that exist in several
        key = allfr[allfr.k_shot.isin([1, 5])].copy()
        key["method"] = key.run.map(lambda n: parse_run_name(n)["method"])
        key = key[~key.run.str.contains("seed|abl|lr|epochstudy")]
        piv = key.pivot_table(index=["method", "k_shot"], columns="split", values="acc")
        if not piv.empty:
            fig, ax = plt.subplots(figsize=(12, 4.5))
            (100 * piv).plot.bar(ax=ax, width=.8)
            ax.set_xticklabels([f"{NICE.get(m, m)}\n{k}-shot" for m, k in piv.index], rotation=0, fontsize=7)
            ax.set_ylabel("test accuracy (%)"); ax.set_xlabel(""); ax.grid(axis="y", alpha=.3)
            ax.set_title("5-way accuracy: Protocol A (known classes) vs B novel vs B hard-novel", fontsize=10)
            made.append(_save(fig, FIG / "protocol_gap.png"))
    return made


def plot_seeds() -> List[Path]:
    seeds = _read_csv(REPORTS / "seeds.csv")
    if seeds is None:
        return []
    fig, ax = plt.subplots(figsize=(7, 4))
    for (sp, k), g in seeds.groupby(["split", "k_shot"]):
        ax.errorbar([f"{sp}\n{k}-shot"], [100 * g.acc.mean()], yerr=[100 * g.acc.std(ddof=1)], fmt="o", capsize=6)
        ax.scatter([f"{sp}\n{k}-shot"] * len(g), 100 * g.acc, s=12, c="gray", zorder=3)
    ax.set_ylabel("test accuracy, 5-way (%)"); ax.grid(alpha=.3)
    ax.set_title("ProtoNet (Protocol B) across 3 seeds: mean ± std (grey = individual seeds)", fontsize=10)
    return [_save(fig, FIG / "seeds.png")]


# --- dashboard --------------------------------------------------------------------------------
def copy_run_figures(rd: Path) -> List[Path]:
    dst = FIG / "runs" / rd.name
    dst.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted((rd / "plots").glob("*.png")):
        shutil.copy2(p, dst / p.name)
        out.append(dst / p.name)
    return out


def build_dashboard() -> Path:
    def img(p: Path, w: str = "100%") -> str:
        rel = p.relative_to(REPORTS).as_posix()
        return f'<a href="{html.escape(rel)}"><img src="{html.escape(rel)}" style="width:{w}" loading="lazy"></a>'

    parts = ["<!doctype html><html><head><meta charset='utf-8'><title>Tamil few-shot HTR dashboard</title>",
             "<style>body{font-family:Segoe UI,Nirmala UI,sans-serif;margin:24px;max-width:1500px;background:#fafafa;color:#222}"
             "h2{border-bottom:2px solid #ccc;padding-bottom:4px;margin-top:40px}.grid{display:grid;grid-template-columns:"
             "repeat(auto-fill,minmax(460px,1fr));gap:12px}.card{background:#fff;padding:8px;border:1px solid #ddd;border-radius:6px}"
             "table{border-collapse:collapse;font-size:13px;background:#fff}td,th{border:1px solid #ccc;padding:3px 7px;text-align:right}"
             "th{background:#eee}td:first-child,th:first-child{text-align:left}details{margin:10px 0}</style></head><body>",
             "<h1>Handwritten Tamil character recognition — few-shot deep learning</h1>",
             "<p>Static dashboard generated by <code>src/visualize.py --all</code> from stored results only. "
             "Click any figure for full resolution. Test numbers come from a single evaluation per checkpoint.</p>"]
    comp = REPORTS / "model_comparison.csv"
    if comp.exists():
        df = pd.read_csv(comp, encoding="utf-8-sig")
        cols = [c for c in ["method", "backbone", "protocol", "tag", "params", "cpu_ms_per_image", "best_epoch", "best_val_loss",
                            "test_acc_5w1s", "test_acc_5w5s", "test_ci_5w5s", "fullset_acc", "macro_f1", "ece_before", "ece_after",
                            "openset_auroc"] if c in df]
        parts += ["<h2>Model comparison (test)</h2>", df[cols].to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0)]
    sections = [("Data", ["class_counts.png", "source_groups.png"], ["preprocess_samples.png"]),
                ("Epoch check: LR finder, LR comparison, epoch budget, ablations",
                 [p.name for p in sorted(FIG.glob("lr_finder_*.png"))] + ["lr_comparison.png", "epoch_study.png", "ablations_val_loss.png"], []),
                ("Method comparison", ["overlay_val_loss_protocolB.png", "overlay_val_episode_acc_protocolB.png",
                                       "overlay_val_loss_protocolA.png", "overlay_val_episode_acc_protocolA.png",
                                       "learning_curves_all_methods.png", "protocol_gap.png", "model_bubble.png", "seeds.png"], [])]
    for name, figs, rep in sections:
        cards = [f"<div class='card'>{img(FIG / f)}</div>" for f in figs if (FIG / f).exists()]
        cards += [f"<div class='card'>{img(REPORTS / f)}</div>" for f in rep if (REPORTS / f).exists()]
        if cards:
            parts += [f"<h2>{html.escape(name)}</h2><div class='grid'>", *cards, "</div>"]
    parts.append("<h2>Per-run figures</h2>")
    for d in sorted((FIG / "runs").glob("*")) if (FIG / "runs").exists() else []:
        pngs = sorted(d.glob("*.png"))
        if not pngs:
            continue
        open_ = " open" if ("final" in d.name and "seed" not in d.name) else ""
        parts += [f"<details{open_}><summary><b>{html.escape(d.name)}</b> ({len(pngs)} figures)</summary><div class='grid'>",
                  *[f"<div class='card'>{img(p)}<div>{html.escape(p.stem)}</div></div>" for p in pngs], "</div></details>"]
    parts.append("</body></html>")
    out = REPORTS / "dashboard.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    log.info("dashboard: %s", out)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", action="append", default=[], help="run directory (repeatable)")
    p.add_argument("--all", action="store_true", help="every run + cross-run figures + dashboard")
    p.add_argument("--skip-runs", action="store_true", help="with --all: only cross-run figures and the dashboard")
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()
    setup_logging(a.log_level)
    for r in a.run:
        plot_run(Path(r))
        copy_run_figures(Path(r))
    if a.all:
        if not a.skip_runs:
            for rd in list_runs():
                plot_run(rd)
        for rd in list_runs():
            copy_run_figures(rd)
        cross_run_figures()
        build_dashboard()


if __name__ == "__main__":
    main()
