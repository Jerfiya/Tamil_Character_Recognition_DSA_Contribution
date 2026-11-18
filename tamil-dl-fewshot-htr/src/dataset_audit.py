"""Read-only audit of the raw dataset.

Locates the dataset tree by recursive scan, discovers classes dynamically (a class is any
folder that directly contains image files), and reports counts, empties, nesting, zips,
formats, corrupt files, perceptual-hash duplicates, source groups, image sizes, missing
letters of the 18 x 12 grid and spelling variants. Nothing inside the dataset is modified;
zip archives are extracted to ``data/interim/extracted``.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import re
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import tamil_text as tt
from config import ROOT, Cfg, add_common_args, load_config, resolve, setup_logging
from preprocessing import ink_is_dark, read_image_gray

log = logging.getLogger("audit")

SKIP_DIRS = {".venv", "venv", ".git", "__pycache__", "node_modules", ".pytest_cache"}
SOURCE_GROUPS = ["whatsapp", "screenshot", "scanned", "pxl", "img_date", "resized", "numeric", "other"]


@dataclass
class ClassDir:
    """One folder that directly contains images (or zips) for a class."""
    category: str
    label: str
    path: Path
    files: List[Path] = field(default_factory=list)
    zips: List[Path] = field(default_factory=list)
    nested: bool = False


def source_group(filename: str) -> str:
    """Coarse capture source derived from a file name."""
    s = Path(filename).name.lower()
    stem = re.sub(r"(\.[a-z0-9_ #=]{2,12})+$", "", s)
    if "whatsapp" in stem or re.match(r"img-\d{8}-wa", stem):
        return "whatsapp"
    if "screenshot" in stem or "screen shot" in stem:
        return "screenshot"
    if "scan" in stem:
        return "scanned"
    if stem.startswith("pxl_"):
        return "pxl"
    if re.match(r"img[_-]?\d{6,}", stem):
        return "img_date"
    if "resiz" in stem:
        return "resized"
    if re.fullmatch(r"[\d\s()_\-]+", stem):
        return "numeric"
    return "other"


def writer_key(filename: str, group: str, origin: str = "") -> str:
    """Finer grouping: a normalised writer name for name-style files, else group + origin."""
    if group != "other":
        return f"{group}:{origin}" if origin else group
    stem = re.sub(r"(\.[A-Za-z0-9_ #=]{2,12})+$", "", Path(filename).name).lower()
    stem = re.sub(r"[஀-௿]+", " ", stem)           # drop Tamil letters (the class label)
    stem = re.sub(r"\(\d+\)|\d+", " ", stem)                # drop counters
    stem = re.sub(r"(_|\b)(png|jpg|jpeg|loose|tight)\b", " ", stem)
    stem = re.sub(r"[\W_]+", "", stem)
    return stem or "unnamed"


def _is_excluded(path: Path, exclude: Sequence[Path]) -> bool:
    rp = path.resolve()
    return any(rp == e or e in rp.parents for e in exclude)


def _walk(scan_root: Path, exclude: Sequence[Path]) -> Iterable[Tuple[Path, List[str], List[str]]]:
    for dirpath, dirnames, filenames in os.walk(scan_root):
        d = Path(dirpath)
        dirnames[:] = sorted(n for n in dirnames if n not in SKIP_DIRS and not _is_excluded(d / n, exclude))
        yield d, dirnames, sorted(filenames)


def locate_dataset_root(scan_root: Path, exts: Sequence[str], exclude: Sequence[Path] = ()) -> Path:
    """Find the dataset tree under ``scan_root`` without assuming its sub-path.

    Folders holding images/zips are grouped by their top-level ancestor below ``scan_root``;
    the group with the most files wins and its deepest common ancestor is the dataset root
    (so the layout below it is ``category / class / images``).
    """
    scan_root = scan_root.resolve()
    exclude = [Path(e).resolve() for e in exclude]
    exts = {e.lower() for e in exts}
    groups: Dict[str, List[Tuple[Path, int]]] = defaultdict(list)
    for d, _, files in _walk(scan_root, exclude):
        n = sum(1 for f in files if Path(f).suffix.lower() in exts or f.lower().endswith(".zip"))
        if n and d != scan_root:
            groups[d.relative_to(scan_root).parts[0]].append((d, n))
    if not groups:
        raise FileNotFoundError(f"no image folders found under {scan_root}")
    best = max(groups.values(), key=lambda items: sum(n for _, n in items))
    dirs = [d for d, _ in best]
    common = Path(os.path.commonpath([str(d) for d in dirs]))
    if len(dirs) == 1 or common in dirs:          # degenerate: a single class or single category
        common = dirs[0].parent.parent if len(dirs) == 1 else common.parent
    return common


def discover_classes(root: Path, exts: Sequence[str]) -> Tuple[List[ClassDir], List[Path]]:
    """Return (class folders, empty leaf folders). Label = NFC name of the folder holding images."""
    exts = {e.lower() for e in exts}
    classes: List[ClassDir] = []
    empty: List[Path] = []
    root = root.resolve()
    for d, subdirs, files in _walk(root, ()):
        if d == root:
            continue
        rel = d.relative_to(root).parts
        imgs = [d / f for f in files if Path(f).suffix.lower() in exts]
        zips = [d / f for f in files if f.lower().endswith(".zip")]
        if imgs or zips:
            if len(rel) < 2:
                log.warning("images directly inside category folder %s are unlabeled; skipped", d)
                continue
            names = [tt.nfc(p) for p in rel[1:]]
            nested = len(rel) > 2
            classes.append(ClassDir(tt.nfc(rel[0]), names[-1], d, imgs, zips, nested))
        elif not subdirs and len(rel) >= 2:
            empty.append(d)
    return classes, empty


def extract_zips(classes: List[ClassDir], out_root: Path, exts: Sequence[str]) -> int:
    """Extract image members of zip archives to ``out_root/<category>/<class>/`` (never in the dataset)."""
    exts = {e.lower() for e in exts}
    count = 0
    for c in classes:
        for z in c.zips:
            dest = out_root / c.category / c.label / z.stem
            dest.mkdir(parents=True, exist_ok=True)
            try:
                with zipfile.ZipFile(z) as zf:
                    for i, info in enumerate(zf.infolist()):
                        name = Path(info.filename).name          # basename only: no zip-slip
                        if info.is_dir() or Path(name).suffix.lower() not in exts:
                            continue
                        target = dest / f"{i:05d}_{name}"
                        if not target.exists():
                            target.write_bytes(zf.read(info))
                        c.files.append(target)
                        count += 1
            except (zipfile.BadZipFile, OSError) as exc:
                log.warning("bad zip %s: %s", z, exc)
    return count


def load_original_names(root: Path) -> Dict[str, str]:
    """Optional mapping relative-path -> original file name from a ``metadata.csv`` shipped with the data."""
    meta = root / "metadata.csv"
    out: Dict[str, str] = {}
    if not meta.exists():
        return out
    try:
        with open(meta, "r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("new_path") and row.get("original_path"):
                    key = tt.nfc(row["new_path"].replace("\\", "/"))
                    out[key] = row["original_path"].replace("\\", "/").split("/")[-1]
    except Exception as exc:
        log.warning("could not read %s: %s", meta, exc)
    return out


def scan_files(classes: List[ClassDir], root: Path, originals: Dict[str, str]) -> pd.DataFrame:
    """Decode every image once and collect per-file facts (size, hashes, source group, status)."""
    import imagehash
    rows: List[Dict[str, Any]] = []
    for c in classes:
        for f in c.files:
            try:
                rel = tt.nfc(f.resolve().relative_to(root).as_posix())
            except ValueError:
                rel = tt.nfc(f.as_posix())
            orig = originals.get(rel, f.name)
            origin = (re.search(r"_(ds\d+)_", f.name) or [None, ""])[1]
            grp = source_group(orig)
            row: Dict[str, Any] = {"path": str(f), "rel_path": rel, "category": c.category, "label": c.label,
                                   "ext": f.suffix.lower(), "original_name": orig, "origin": origin,
                                   "source_group": grp, "writer": writer_key(orig, grp, origin),
                                   "nested": c.nested, "status": "ok", "error": ""}
            try:
                gray = read_image_gray(f)
                row.update(width=gray.shape[1], height=gray.shape[0], dark_ink=bool(ink_is_dark(gray)),
                           md5=hashlib.md5(gray.tobytes() + str(gray.shape).encode()).hexdigest(),
                           phash=str(imagehash.phash(Image.fromarray(gray))))
            except ValueError as exc:
                row.update(status="corrupt", error=str(exc)[:200])
            rows.append(row)
    return pd.DataFrame(rows)


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def mark_duplicates(df: pd.DataFrame, threshold: int) -> pd.DataFrame:
    """Flag duplicates: pixel-identical images in different classes are label conflicts (all
    copies dropped); within a class, images whose pHash differs by <= ``threshold`` bits are
    near-duplicates (first kept)."""
    df = df.copy()
    status = df.status.to_numpy().astype(object)
    dup_of = np.array([""] * len(df), dtype=object)
    rel = df.rel_path.to_numpy()
    pos = {ix: i for i, ix in enumerate(df.index)}
    ok = df[df.status == "ok"]
    for _, g in ok.groupby("md5"):
        if len(g) > 1 and g.label.nunique() > 1:
            status[[pos[i] for i in g.index]] = "conflict"
    ok = df[status == "ok"]
    for _, g in ok.groupby(["category", "label"]):
        idx = [pos[i] for i in g.index]
        hashes = [int(h, 16) for h in g.phash]
        md5s = list(g.md5)
        kept: List[int] = []
        for i in range(len(idx)):
            dup = next((k for k in kept if md5s[k] == md5s[i] or _hamming(hashes[k], hashes[i]) <= threshold), None)
            if dup is None:
                kept.append(i)
            else:
                status[idx[i]] = "duplicate"
                dup_of[idx[i]] = rel[idx[dup]]
    df["status"], df["dup_of"] = status, dup_of
    return df


def run_audit(cfg: Cfg, write: bool = True) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run the whole audit; returns the per-file table and the summary dict."""
    exts = cfg.audit.image_exts
    scan_root = resolve(cfg, "scan_root")
    root = locate_dataset_root(scan_root, exts, exclude=[ROOT])
    log.info("dataset root located at: %s", root)
    classes, empty = discover_classes(root, exts)
    interim = resolve(cfg, "interim")
    n_zip = extract_zips(classes, interim / "extracted", exts) if any(c.zips for c in classes) else 0
    df = mark_duplicates(scan_files(classes, root, load_original_names(root)), int(cfg.audit.phash_threshold))

    ok = df[df.status == "ok"]
    raw_counts = df.groupby("label").size()
    counts = ok.groupby("label").size()
    labels = sorted(df.label.unique())
    dirs_per_class = Counter((c.category, c.label) for c in classes)
    grid = tt.grid_report(labels)
    # near-duplicate sensitivity: how many would be removed at other thresholds
    sens = {}
    base = df[df.status.isin(["ok", "duplicate"])].assign(status="ok")
    for t in (0, 2, 4, 6, 8, 10):
        sens[str(t)] = int((mark_duplicates(base, t).status == "duplicate").sum())
    summary: Dict[str, Any] = {
        "scan_root": str(scan_root), "dataset_root": str(root),
        "categories": {k: int(v) for k, v in df.groupby("category").label.nunique().items()},
        "n_classes_with_images": int(len(labels)),
        "n_files_seen": int(len(df)), "n_ok_after_dedupe": int(len(ok)),
        "raw_count_stats": {"min": int(raw_counts.min()), "median": float(raw_counts.median()), "max": int(raw_counts.max())},
        "clean_count_stats": {"min": int(counts.min()), "median": float(counts.median()), "max": int(counts.max())},
        "small_classes_lt15": {k: int(v) for k, v in counts[counts < 15].sort_values().items()},
        "classes_lost_entirely": sorted(set(labels) - set(counts.index)),
        "empty_class_folders": [str(p.relative_to(root)) for p in empty],
        "nested_class_folders": [str(c.path.relative_to(root)) for c in classes if c.nested],
        "classes_from_multiple_folders": [f"{k[0]}/{k[1]}" for k, v in dirs_per_class.items() if v > 1],
        "zip_archives": [str(z.relative_to(root)) for c in classes for z in c.zips], "zip_images_extracted": n_zip,
        "formats": {k: int(v) for k, v in df.ext.value_counts().items()},
        "corrupt_files": df[df.status == "corrupt"].rel_path.tolist(),
        "label_conflicts_removed": int((df.status == "conflict").sum()),
        "near_duplicates_removed": int((df.status == "duplicate").sum()),
        "duplicates_removed_per_class": {k: int(v) for k, v in df[df.status == "duplicate"].groupby("label").size().sort_values(ascending=False).items()},
        "duplicate_threshold_sensitivity": sens,
        "source_groups": {k: int(v) for k, v in ok.source_group.value_counts().items()},
        "origins": {k or "-": int(v) for k, v in ok.origin.value_counts().items()},
        "n_writers": int(ok.writer.nunique()),
        "image_size": {"width": _stats(ok.width), "height": _stats(ok.height),
                       "distinct_sizes": int(ok.groupby(["width", "height"]).ngroups)},
        "polarity": {"dark_ink_on_light": int(ok.dark_ink.sum()), "light_ink_on_dark": int((~ok.dark_ink.astype(bool)).sum())},
        "grid": grid,
        "enrollable_later": sorted(set(grid["missing_compounds"] + grid["missing_vowels"] + grid["missing_consonants"]
                                       + grid["missing_aytham"] + [tt.nfc(p.name) for p in empty])),
        "episode_eligibility": {f"K+Q>={n}": int((counts >= n).sum()) for n in cfg.audit.kq_table},
        "original_names_from_metadata_csv": bool((root / "metadata.csv").exists()),
    }
    if write:
        interim.mkdir(parents=True, exist_ok=True)
        df.to_csv(interim / "audit_files.csv", index=False, encoding="utf-8-sig")
        reports = resolve(cfg, "reports")
        reports.mkdir(parents=True, exist_ok=True)
        (reports / "audit.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        cc = pd.DataFrame({"raw": raw_counts, "clean": counts}).fillna(0).astype(int).reset_index()
        cc.insert(0, "category", cc.label.map(df.drop_duplicates("label").set_index("label").category))
        cc.to_csv(reports / "audit_class_counts.csv", index=False, encoding="utf-8-sig")
        (reports / "audit.md").write_text(render_markdown(summary), encoding="utf-8")
    return df, summary


def _stats(s: pd.Series) -> Dict[str, float]:
    return {"min": int(s.min()), "median": float(s.median()), "max": int(s.max())}


def render_markdown(s: Dict[str, Any]) -> str:
    """Human-readable audit summary."""
    g = s["grid"]
    j = lambda xs: ", ".join(xs) if xs else "none"  # noqa: E731
    lines = [
        "# Dataset audit", "",
        f"- Dataset root (found by scan): `{s['dataset_root']}`",
        f"- Classes with images: **{s['n_classes_with_images']}** — per category: {s['categories']}",
        f"- Files seen: {s['n_files_seen']}; usable after cleaning: **{s['n_ok_after_dedupe']}**",
        f"- Images per class (raw): min {s['raw_count_stats']['min']} / median {s['raw_count_stats']['median']:.0f} / max {s['raw_count_stats']['max']}",
        f"- Images per class (after dedupe): min {s['clean_count_stats']['min']} / median {s['clean_count_stats']['median']:.0f} / max {s['clean_count_stats']['max']}",
        f"- Small classes (<15 images after dedupe): {s['small_classes_lt15'] or 'none'}",
        f"- Empty class folders (skipped, enrollable later): {j(s['empty_class_folders'])}",
        f"- Nested class folders: {j(s['nested_class_folders'])}",
        f"- Classes merged from several folders: {j(s['classes_from_multiple_folders'])}",
        f"- Zip archives: {j(s['zip_archives'])} ({s['zip_images_extracted']} images extracted)",
        f"- Formats: {s['formats']}",
        f"- Corrupt files (skipped): {len(s['corrupt_files'])} {j(s['corrupt_files'][:10])}",
        f"- Cross-class identical images removed as label conflicts: {s['label_conflicts_removed']}",
        f"- Within-class near-duplicates removed (pHash): {s['near_duplicates_removed']}",
        f"- Near-duplicate count by Hamming threshold: {s['duplicate_threshold_sensitivity']}",
        f"- Source groups: {s['source_groups']}; origins: {s['origins']}; distinct writer keys: {s['n_writers']}",
        f"- Image size: width {s['image_size']['width']}, height {s['image_size']['height']}, distinct sizes {s['image_size']['distinct_sizes']}",
        f"- Polarity: {s['polarity']}",
        "", "## Alphabet coverage (18 x 12 grid)", "",
        f"- Missing compounds: {j(g['missing_compounds'])}",
        f"- Missing vowels: {j(g['missing_vowels'])}; missing consonants: {j(g['missing_consonants'])}; missing aytham: {j(g['missing_aytham'])}",
        f"- **Possible spelling variants (kept as distinct classes, NOT merged):** {j(g['spelling_variants'])} "
        f"— possible variants of {j(g['variant_of'])}",
        f"- Unrecognised labels: {j(g['unrecognised'])}",
        f"- Enrollable later: {j(s['enrollable_later'])}",
        "", "## Episode eligibility (classes with at least K+Q images)", "",
        "| requirement | classes |", "|---|---|",
    ] + [f"| {k} | {v} |" for k, v in s["episode_eligibility"].items()] + [
        "", "## Duplicates removed per class (top 15)", "",
        j([f"{k}:{v}" for k, v in list(s["duplicates_removed_per_class"].items())[:15]]), ""]
    return "\n".join(lines)


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser(description=__doc__))
    args = parser.parse_args()
    setup_logging(args.log_level)
    cfg = load_config(args.config, args.set)
    _, summary = run_audit(cfg)
    log.info("\n%s", render_markdown(summary))


if __name__ == "__main__":
    main()
