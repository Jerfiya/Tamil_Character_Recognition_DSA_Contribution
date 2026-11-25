"""The UI must predict exactly what ``src/predict.py`` predicts (same preprocessing, same model)."""
from __future__ import annotations

from pathlib import Path

from .conftest import ROOT, needs_ui_data, sample_images

pytestmark = needs_ui_data


def test_ui_matches_predict_py():
    from fastapi.testclient import TestClient
    from predict import Recognizer
    from server import create_app
    from service import RecognitionService
    from support_registry import DEFAULT_MODELS, DEFAULT_SUPPORT

    samples = sample_images(20, seed=7)
    assert len(samples) == 20
    paths = [Path(s["path"]) for s in samples]
    ref = Recognizer(DEFAULT_MODELS, DEFAULT_SUPPORT, "onnx", 5).predict_files(paths)

    svc = RecognitionService()          # real support set, read-only use (no enrolment here)
    svc.load()
    with TestClient(create_app(svc)) as c:
        for p, want in zip(paths, ref):
            got = c.post("/api/predict", files={"image": (p.name, p.read_bytes(), "image/png")}).json()
            assert got["label"] == want.label, p
            assert got["known"] == want.known
            assert [t["letter"] for t in got["top"]] == [lab for lab, _ in want.top]
            for t, (_, conf) in zip(got["top"], want.top):
                assert abs(t["confidence"] - conf) < 1e-4
            assert abs(got["score"] - want.score) < 1e-3
