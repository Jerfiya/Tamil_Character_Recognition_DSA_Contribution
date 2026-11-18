"""Build the derived dataset: manifest, preprocessed image caches, and the two protocols.

Outputs (all under ``data/processed``): ``manifest.csv``, ``class_names.json``,
``images_<size>.npy``, ``splits_A.csv`` (Protocol A image split) and ``class_split.json``
(Protocol B class split, with the hard-novel subset). Duplicates are removed by the audit
BEFORE any split is made.
"""
from __future__ import annotations

import argparse
import json
import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

import tamil_text as tt
from config import Cfg, add_common_args, load_config, resolve, setup_logging
from dataset_audit import render_markdown, run_audit
from preprocessing import preprocess_gray, read_image_gray

log = logging.getLogger("prepare")


def build_manifest(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Keep usable files and attach class indices and parsed label parts."""
    df = audit_df[audit_df.status == "ok"].copy()
    df["label"] = df.label.map(tt.nfc)
    order = df.drop_duplicates("label").sort_values(["category", "label"]).label.tolist()
    idx = {lab: i for i, lab in enumerate(order)}
    df["class_idx"] = df.label.map(idx)
    parts = df.label.map(tt.parse_label)
    df["kind"] = parts.map(lambda p: p["kind"])
    df["consonant"] = parts.map(lambda p: p["consonant"] or "")
    df["vowel"] = parts.map(lambda p: "-" if p["vowel"] is None else p["vowel"])
    df["category_en"] = df.label.map(tt.english_category)
    return df.sort_values(["class_idx", "rel_path"]).reset_index(drop=True)


def split_protocol_a(df: pd.DataFrame, fr: Dict[str, float], seed: int) -> np.ndarray:
    """Per-class 70/15/15 image split, stratified by source group.

    Images of a class are ordered by source group (shuffled inside each group); test and val
    members are then taken at evenly spaced positions, so each pool gets every source group
    in proportion. Classes with fewer than 3 images go entirely to train.
    """
    rng = np.random.default_rng(seed)
    split = np.array(["train"] * len(df), dtype=object)
    for _, g in df.groupby("class_idx"):
        n = len(g)
        if n < 3:
            continue
        order = g.assign(_r=rng.random(n)).sort_values(["source_group", "_r"]).index.to_numpy()
        n_test = max(1, int(round(fr["test"] * n)))
        n_val = max(1, int(round(fr["val"] * n)))
        if n - n_test - n_val < 1:
            n_val = max(0, n - n_test - 1)
        off = rng.random()
        test_pos = np.unique(np.floor((np.arange(n_test) + off) * n / n_test).astype(int) % n)
        rest = np.setdiff1d(np.arange(n), test_pos)
        val_pos = rest[np.unique(np.floor((np.arange(n_val) + rng.random()) * len(rest) / max(n_val, 1)).astype(int) % len(rest))] if n_val else np.array([], int)
        split[order[test_pos]] = "test"
        split[order[val_pos]] = "val"
    return split


def split_protocol_b(classes: pd.DataFrame, fr: Dict[str, Any], seed: int) -> Dict[str, Any]:
    """Split CLASSES into base / val / novel, stratified by category.

    Classes with fewer than ``min_images_eval`` images cannot form evaluation episodes, so
    they are forced into base. ``hard_novel`` = novel compounds whose consonant AND vowel sign
    each occur in some base compound.
    """
    rng = np.random.default_rng(seed)
    out: Dict[str, List[str]] = {"base": [], "val": [], "novel": []}
    forced: List[str] = []
    for _, g in classes.groupby("category"):
        small = g[g["count"] < int(fr["min_images_eval"])].label.tolist()
        forced += small
        out["base"] += small
        elig = g[g["count"] >= int(fr["min_images_eval"])].label.tolist()
        elig = [elig[i] for i in rng.permutation(len(elig))]
        n = len(g)
        n_novel = min(int(round(fr["novel"] * n)), max(0, len(elig) - 1)) if n >= 4 else 0
        n_val = min(int(round(fr["val"] * n)), max(0, len(elig) - n_novel - 1)) if n >= 4 else 0
        out["novel"] += elig[:n_novel]
        out["val"] += elig[n_novel:n_novel + n_val]
        out["base"] += elig[n_novel + n_val:]
    base_parts = [tt.parse_label(x) for x in out["base"]]
    base_cons = {p["consonant"] for p in base_parts if p["kind"] in ("compound", "variant")}
    base_vow = {p["vowel"] for p in base_parts if p["kind"] in ("compound", "variant")}
    hard = []
    for lab in out["novel"]:
        p = tt.parse_label(lab)
        if p["kind"] == "compound" and p["consonant"] in base_cons and p["vowel"] in base_vow:
            hard.append(lab)
    result: Dict[str, Any] = {k: sorted(v) for k, v in out.items()}
    result.update(hard_novel=sorted(hard), forced_base_small=sorted(forced), seed=seed,
                  note="hard_novel: novel compounds whose consonant and vowel sign both occur in base compounds")
    return result


def preprocess_all(df: pd.DataFrame, cfg: Cfg, size: int) -> np.ndarray:
    """Preprocess every manifest row with the shared pipeline."""
    out = np.zeros((len(df), size, size), np.uint8)
    for i, p in enumerate(df.path):
        out[i] = preprocess_gray(read_image_gray(p), cfg.preprocess, size)
    return out


def save_sample_grid(df: pd.DataFrame, images: np.ndarray, path, seed: int, n: int = 24) -> None:
    """Raw vs preprocessed samples with Tamil labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tt.setup_matplotlib()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(df), size=min(n, len(df)), replace=False)
    fig, axes = plt.subplots(4, 12, figsize=(18, 7.2))
    for j, i in enumerate(pick):
        r, c = divmod(j, 6)
        a_raw, a_pre = axes[r, 2 * c], axes[r, 2 * c + 1]
        a_raw.imshow(read_image_gray(df.path.iloc[i]), cmap="gray", vmin=0, vmax=255)
        a_pre.imshow(images[i], cmap="gray", vmin=0, vmax=255)
        tt.mpl_text(a_raw, 0.5, 1.04, df.label.iloc[i], px=30, zoom=0.5, xycoords=a_raw.transAxes, box_alignment=(0.5, 0))
        a_pre.set_title(df.source_group.iloc[i], fontsize=7)
        for a in (a_raw, a_pre):
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle("Preprocessing samples: raw (left, Tamil label above) -> normalised (right, source group above)", y=0.995)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, args.set)
    out = resolve(cfg, "processed")
    out.mkdir(parents=True, exist_ok=True)
    reports = resolve(cfg, "reports")

    audit_df, summary = run_audit(cfg, write=True)
    df = build_manifest(audit_df)
    size = int(cfg.preprocess.size)
    images = preprocess_all(df, cfg, size)
    blank = images.reshape(len(df), -1).max(1) == 0
    if blank.any():
        log.warning("dropping %d images with no detectable ink: %s", int(blank.sum()), df.rel_path[blank].tolist()[:10])
        pd.DataFrame({"rel_path": df.rel_path[blank]}).to_csv(out / "blank_dropped.csv", index=False, encoding="utf-8-sig")
        df = df[~blank].reset_index(drop=True)
        images = images[~blank]
    df.insert(0, "idx", np.arange(len(df)))
    np.save(out / f"images_{size}.npy", images)
    for s in cfg.preprocess.extra_sizes:
        np.save(out / f"images_{int(s)}.npy", preprocess_all(df, cfg, int(s)))

    counts = df.groupby("label").size()
    classes = df.drop_duplicates("class_idx").sort_values("class_idx")[["class_idx", "label", "category", "category_en", "kind", "consonant", "vowel"]].copy()
    classes["count"] = classes.label.map(counts).astype(int)
    (out / "class_names.json").write_text(json.dumps(
        [{"index": int(r.class_idx), "label": r.label, "category": r.category, "category_en": r.category_en,
          "kind": r.kind, "consonant": r.consonant, "vowel": r.vowel, "count": int(r["count"])}
         for _, r in classes.iterrows()], ensure_ascii=False, indent=1), encoding="utf-8")

    df["split_A"] = split_protocol_a(df, cfg.split.A, int(cfg.split.seed))
    df[["idx", "rel_path", "label", "class_idx", "source_group", "writer", "split_A"]].to_csv(out / "splits_A.csv", index=False, encoding="utf-8-sig")
    csplit = split_protocol_b(classes, cfg.split.B, int(cfg.split.seed))
    (out / "class_split.json").write_text(json.dumps(csplit, ensure_ascii=False, indent=1), encoding="utf-8")
    df.to_csv(out / "manifest.csv", index=False, encoding="utf-8-sig")
    save_sample_grid(df, images, reports / "preprocess_samples.png", int(cfg.seed))

    lab2n = counts.to_dict()
    lines = [render_markdown(summary), "", "# Split summary", "",
             f"Images in manifest: {len(df)} across {len(classes)} classes; cached sizes: {[size] + list(cfg.preprocess.extra_sizes)}", "",
             "## Protocol A (image split, all classes)", "", "| pool | images | classes | min/class | median/class |", "|---|---|---|---|---|"]
    for part in ("train", "val", "test"):
        c = df[df.split_A == part].groupby("label").size()
        lines.append(f"| {part} | {int(c.sum())} | {len(c)} | {int(c.min())} | {c.median():.0f} |")
    lines += ["", "Source-group share per pool: " + "; ".join(
        f"{part}: " + ", ".join(f"{k} {v:.0%}" for k, v in df[df.split_A == part].source_group.value_counts(normalize=True).items())
        for part in ("train", "val", "test")), "",
        "## Protocol B (class split)", "", "| part | classes | images | by category |", "|---|---|---|---|"]
    cat = classes.set_index("label").category_en
    for part in ("base", "val", "novel", "hard_novel"):
        labs = csplit[part]
        bc = pd.Series([cat[x] for x in labs]).value_counts().to_dict() if labs else {}
        lines.append(f"| {part} | {len(labs)} | {sum(lab2n[x] for x in labs)} | {bc} |")
    lines += ["", f"Small classes forced into base: {', '.join(csplit['forced_base_small']) or 'none'}",
              f"Novel classes: {' '.join(csplit['novel'])}", f"Val classes: {' '.join(csplit['val'])}", ""]
    text = "\n".join(lines)
    (reports / "data_summary.md").write_text(text, encoding="utf-8")
    log.info("\n%s", text)


if __name__ == "__main__":
    main()
