"""Fingerprint the read-only dataset tree (relative path, size, md5) and compare later.

    python src/check_dataset_untouched.py --record    # writes data/interim/dataset_fingerprint.json
    python src/check_dataset_untouched.py             # compares, exit 1 on any difference
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path
from typing import Dict

from config import add_common_args, load_config, resolve, setup_logging

log = logging.getLogger("fingerprint")


def fingerprint(root: Path) -> Dict[str, list]:
    """{relative posix path: [size, md5]} for every file under ``root``."""
    out = {}
    for p in sorted(root.rglob("*")):
        if p.is_file():
            out[p.relative_to(root).as_posix()] = [p.stat().st_size, hashlib.md5(p.read_bytes()).hexdigest()]
    return out


def dataset_root(cfg) -> Path:
    audit = json.loads((resolve(cfg, "reports") / "audit.json").read_text(encoding="utf-8"))
    return Path(audit["dataset_root"])


def main() -> None:
    p = add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    p.add_argument("--record", action="store_true")
    a = p.parse_args()
    setup_logging(a.log_level)
    cfg = load_config(a.config, a.set)
    root = dataset_root(cfg)
    ref = resolve(cfg, "interim") / "dataset_fingerprint.json"
    fp = fingerprint(root)
    if a.record:
        ref.write_text(json.dumps({"root": str(root), "files": fp}, ensure_ascii=False), encoding="utf-8")
        log.info("recorded %d files under %s", len(fp), root)
        return
    old = json.loads(ref.read_text(encoding="utf-8"))["files"]
    added, removed = set(fp) - set(old), set(old) - set(fp)
    changed = [k for k in set(fp) & set(old) if fp[k] != old[k]]
    log.info("dataset %s: %d files; added %d, removed %d, changed %d", root, len(fp), len(added), len(removed), len(changed))
    sys.exit(1 if (added or removed or changed) else 0)


if __name__ == "__main__":
    main()
