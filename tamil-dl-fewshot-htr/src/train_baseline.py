"""Train the plain supervised cross-entropy classifier (non-few-shot reference, Protocol A only).

It uses the same Trainer, RunLogger, class-balanced batches, augmentation and safeguards as
the few-shot methods; a "step" is one class-balanced batch the size of a training episode.

    python src/train_baseline.py
    python src/train_baseline.py --resume runs/<run_dir>
"""
from __future__ import annotations

from train_fewshot import build_parser, run_from_args


def main() -> None:
    args = build_parser(__doc__).parse_args()
    run_from_args(args, forced=["method.name=supervised", "train.protocol=A"])


if __name__ == "__main__":
    main()
