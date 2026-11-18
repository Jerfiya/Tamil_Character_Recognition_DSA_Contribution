"""Re-measure CPU inference latency of every evaluated run under identical, idle conditions.

The latency stored in test_summary.json was timed while other training jobs shared the CPU, so the
same architecture showed 6 ms in one run and 47 ms in another. This script times every run's PRIMARY
checkpoint the same way (batch 1, ``threads`` from the config, median of 3 repeats of 200 images) and
writes reports/latency.csv, which summarize.py prefers over the stored values. Run it with no other jobs.

    python src/measure_latency.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import ROOT, setup_logging
from runmodel import RunModel

log = logging.getLogger("latency")


def main() -> None:
    setup_logging("INFO")
    rows, data = [], None
    torch.set_num_threads(2)
    warm = torch.nn.Conv2d(1, 64, 3)                    # bring the CPU to a steady clock before the first timed run
    with torch.inference_mode():
        for _ in range(300):
            warm(torch.zeros(8, 1, 96, 96))
    for rd in sorted((ROOT / "runs").iterdir()):
        if not (rd / "test" / "test_summary.json").exists() or not (rd / "checkpoints" / "best_val_loss.pt").exists():
            continue
        rm = RunModel(rd, "best_val_loss.pt", data=data)
        data = rm.data
        torch.set_num_threads(int(rm.cfg.get("threads", 2)))
        with torch.inference_mode():
            ms = float(np.median([rm.latency_ms(200) for _ in range(3)]))
        rows.append({"run": rd.name, "method": rm.method, "backbone": rm.cfg.model.backbone,
                     "input_size": int(rm.cfg.model.input_size), "threads": torch.get_num_threads(), "cpu_ms_per_image": ms})
        log.info("%-62s %7.2f ms/image", rd.name, ms)
    out = ROOT / "reports" / "latency.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("wrote %s (%d runs)", out, len(rows))


if __name__ == "__main__":
    main()
