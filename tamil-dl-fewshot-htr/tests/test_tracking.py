"""RunLogger artefacts, safeguards (NaN abort, overfit/plateau detectors, early stopping), exact resume."""
import json

import pandas as pd
import pytest
import torch

from conftest import needs_data
from config import load_config
from tracking import (EarlyStopping, NonFiniteLossError, OverfitDetector, PlateauDetector, RunLogger, check_finite)

EXPECTED = ["config_snapshot.yaml", "env.json", "model_summary.txt", "class_split.json", "metrics_epoch.csv",
            "metrics_step.csv", "history.json", "train.log", "checkpoints", "test", "plots"]


def _tiny(tmp_path, *extra):
    return load_config(None, [f"paths.runs={tmp_path.as_posix()}", "threads=2", "train.epochs=3",
                              "train.episodes_per_epoch=3", "train.n_way=5", "train.k_shot=2", "train.q_query=2",
                              "val.episodes=4", "val.n_way=5", "train.step_log_every=1", "train.tensorboard=false",
                              "train.warmup_epochs=1", *extra])


def test_overfit_detector_synthetic_curve():
    det = OverfitDetector(patience=3)
    train = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    val = [1.0, 0.95, 0.9, 0.92, 0.95, 0.99, 1.05]   # bottoms out at epoch 3 then rises
    flags = [det.update(t, v) for t, v in zip(train, val)]
    assert flags == [False, False, False, False, False, True, True]


def test_plateau_detector():
    det = PlateauDetector(patience=3, eps=0.01)
    flags = [det.update(x) for x in [1.0, 0.8, 0.6, 0.599, 0.598, 0.597, 0.596]]
    assert flags[-1] and not flags[3]


def test_early_stopping():
    es = EarlyStopping(patience=2)
    assert [es.update(v) for v in [1.0, 0.9, 0.95, 0.96]] == [False, False, False, True]


def test_check_finite():
    check_finite(1.0)
    for bad in (float("nan"), float("inf")):
        with pytest.raises(NonFiniteLossError):
            check_finite(bad)


def test_runlogger_writes_all_files(tmp_path):
    cfg = _tiny(tmp_path)
    lg = RunLogger(tmp_path / "run", cfg)
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 3), torch.nn.Flatten(), torch.nn.LazyLinear(3))
    model(torch.zeros(1, 1, 8, 8))
    lg.start(model, torch.zeros(1, 1, 8, 8), {"protocol": "B"}, {"base": ["அ"]})
    lg.log_step(1, 1, 0.5, 0.5, 1e-3)
    base = dict(train_loss=1.0, val_loss=1.1, train_episode_acc=0.5, val_episode_acc=0.4, val_acc_ci95=0.01,
                val_top3_acc=0.8, lr=1e-3, epoch_time_s=1.0, grad_norm_mean=1.0, clipped_frac=0.0)
    for ep, (tl, vl) in enumerate([(1.0, 1.1), (0.9, 1.0), (0.8, 1.2)], 1):
        lg.log_epoch({**base, "epoch": ep, "train_loss": tl, "val_loss": vl})
    lg.finish("max_epochs")
    for name in EXPECTED:
        assert (tmp_path / "run" / name).exists(), name
    h = json.loads((tmp_path / "run" / "history.json").read_text(encoding="utf-8"))
    assert h["best_val_loss_epoch"] == 2 and h["status"] == "finished" and len(h["epochs"]) == 3
    df = pd.read_csv(tmp_path / "run" / "metrics_epoch.csv")
    assert list(df.train_loss) == [1.0, 0.9, 0.8] and "gen_gap" in df


@needs_data
def test_nan_loss_aborts_with_crash_checkpoint(tmp_path, monkeypatch):
    from train_fewshot import Trainer
    tr = Trainer(_tiny(tmp_path), run_dir=tmp_path / "nanrun")
    orig = tr.model.train_loss

    def bad(feats, info):
        out = orig(feats, info)
        out["loss"] = out["loss"] * float("nan")
        return out
    monkeypatch.setattr(tr.model, "train_loss", bad)
    with pytest.raises(NonFiniteLossError):
        tr.fit()
    assert (tmp_path / "nanrun" / "checkpoints" / "crash.pt").exists()
    assert json.loads((tmp_path / "nanrun" / "history.json").read_text(encoding="utf-8"))["status"] == "crashed"


@needs_data
def test_resume_is_exact(tmp_path):
    from train_fewshot import Trainer
    full = Trainer(_tiny(tmp_path), run_dir=tmp_path / "full")
    full.fit()
    part = Trainer(_tiny(tmp_path), run_dir=tmp_path / "part")
    part.fit(stop_after=1)
    resumed = Trainer(_tiny(tmp_path), run_dir=tmp_path / "part", resume=True)
    assert resumed.epoch == 1
    resumed.fit()
    a = pd.read_csv(tmp_path / "full" / "metrics_epoch.csv")
    b = pd.read_csv(tmp_path / "part" / "metrics_epoch.csv")
    assert len(a) == len(b) == 3
    for col in ("train_loss", "val_loss", "val_episode_acc", "lr"):
        assert a[col].tolist() == pytest.approx(b[col].tolist(), rel=1e-4, abs=1e-6), col
    for name in ("best_val_loss.pt", "best_val_acc.pt", "last.pt"):
        assert (tmp_path / "part" / "checkpoints" / name).exists()
