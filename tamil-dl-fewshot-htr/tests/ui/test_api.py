"""API tests for the web UI (FastAPI TestClient; temp support set, dataset read-only)."""
from __future__ import annotations

from pathlib import Path

import pytest

from .conftest import needs_ui_data, one_per_kind, png_bytes, square_png, sample_images

pytestmark = needs_ui_data


def _post(client, path, blob: bytes, name="x.png", ctype="image/png", **extra):
    return client.post(path, files={"image": (name, blob, ctype)}, **extra)


def test_health(client):
    h = client.get("/api/health").json()
    assert h["ready"] and h["status"] == "ready"
    assert h["model_id"] == "mnv3-onnx" and h["backend"] == "onnx" and h["device"] == "cpu"
    assert h["class_count"] == 247 and h["registry_size"] >= 247 and h["input_size"] == 96


def test_default_model_is_best_deployable(service):
    assert service.default_model_id() == "mnv3-onnx"


def test_classes(client):
    cs = client.get("/api/classes").json()
    by = {c["letter"]: c for c in cs}
    assert sum(c["kind"] == "vowel" for c in cs) == 12
    assert sum(c["kind"] == "consonant" for c in cs) == 18
    assert sum(c["kind"] in ("compound",) and c["in_chart"] for c in cs) == 18 * 12
    assert by["க"]["translit"] == "ka" and by["க"]["category_ta"] == "உயிர்மெய்"
    assert by["க்"]["category_ta"] == "மெய்" and by["அ"]["category_ta"] == "உயிர்"
    assert by["ஃ"]["category_ta"] == "ஆய்தம்"
    assert by["கௌ"]["vowel"] == "ஔ" and by["கௌ"]["consonant_row"] == "க்"
    assert by["தெள"]["variant_pair"] == "தௌ" and by["தௌ"]["variant_pair"] == "தெள"
    assert all("enrolled" in c and "n_support" in c for c in cs)
    assert not by["லௌ"]["trained"]        # missing from the dataset


@pytest.mark.parametrize("kind", ["vowel", "consonant", "compound", "aytham"])
def test_predict_each_category(client, kind):
    s = one_per_kind()[kind]
    r = _post(client, "/api/predict", Path(s["path"]).read_bytes())
    assert r.status_code == 200, r.text
    d = r.json()
    assert len(d["top"]) == 5 and d["preprocessed"].startswith("data:image/png;base64,")
    assert 0 < d["confidence"] <= 1 and d["latency_ms"] > 0
    assert {"translit", "category_ta", "category_en"} <= set(d["top"][0])
    assert d["best"]["kind"] in ("vowel", "consonant", "compound", "variant", "aytham")


def test_top5_sanity_on_test_images(client):
    """Not a benchmark (see reports/), just a guard that the served model is the right one."""
    samples = sample_images(20, seed=3)
    hits = 0
    for s in samples:
        d = _post(client, "/api/predict", Path(s["path"]).read_bytes()).json()
        hits += s["label"] in [t["letter"] for t in d["top"]]
    assert hits / len(samples) >= 0.7, hits


def test_predict_gradcam(client):
    s = one_per_kind()["vowel"]
    d = _post(client, "/api/predict?gradcam=true", Path(s["path"]).read_bytes()).json()
    assert d["gradcam"] and d["gradcam"].startswith("data:image/png")


def test_unknown_and_blank(client):
    d = _post(client, "/api/predict", square_png()).json()
    assert d["known"] is False and d["label"] == "unknown"
    b = _post(client, "/api/predict", png_bytes(lambda g: None)).json()
    assert b["blank"] is True and b["known"] is False


def test_stream_stable_flag(client, service):
    data = Path(one_per_kind()["vowel"]["path"]).read_bytes()
    flags = [_post(client, "/api/predict/stream", data, data={"session": "s1"}).json() for _ in range(service.stable_frames)]
    assert [f["stable"] for f in flags] == [False] * (service.stable_frames - 1) + [True]
    assert len(flags[0]["top"]) == 3 and "preprocessed" not in flags[0]
    other = _post(client, "/api/predict/stream", data, data={"session": "s2"}).json()
    assert other["stable"] is False
    blank = _post(client, "/api/predict/stream", png_bytes(lambda g: None), data={"session": "s1"}).json()
    assert blank["stable"] is False and blank["blank"] is True


def test_rejects_oversize_and_bad_types(client):
    big = b"\x89PNG" + b"0" * (5 * 1024 * 1024 + 10)
    assert _post(client, "/api/predict", big).status_code == 413
    assert _post(client, "/api/predict", b"hello", name="a.txt", ctype="text/plain").status_code == 415
    assert _post(client, "/api/predict", b"not an image at all", name="a.png").status_code == 400
    assert _post(client, "/api/predict", b"", name="a.png").status_code == 400
    r = _post(client, "/api/predict", b"garbage")
    assert "Traceback" not in r.text


@pytest.mark.parametrize("label", ["../evil", "..\\evil", "அ/../../x", "/etc", "C:\\x", "abc", "", "அஅஅஅஅஅ", ".", "க\x00"])
def test_enroll_rejects_bad_labels(client, service, label):
    before = set(p.name for p in service.support_dir.parent.iterdir())
    r = client.post("/api/enroll", data={"label": label}, files=[("images", ("a.png", square_png(), "image/png"))])
    assert r.status_code in (400, 422)
    assert set(p.name for p in service.support_dir.parent.iterdir()) == before
    assert not (service.support_dir / label).exists() or label in ("", ".")


def test_practice_check(client):
    s = one_per_kind()["vowel"]
    data = Path(s["path"]).read_bytes()
    ok = client.post("/api/practice/check", data={"target": s["label"]}, files={"image": ("a.png", data, "image/png")}).json()
    assert ok["correct"] is True and ok["message_ta"] == "சரி!"
    bad = client.post("/api/practice/check", data={"target": "ழ"}, files={"image": ("a.png", data, "image/png")}).json()
    assert bad["correct"] is False and "முயற்சி" in bad["message_ta"]


def test_enroll_new_letter_recognised_without_retraining(client, service):
    """Enrol லௌ (absent from the dataset) from 3 squares; a square is then recognised as லௌ."""
    assert _post(client, "/api/predict", square_png()).json()["label"] == "unknown"
    files = [("images", (f"s{i}.png", square_png(), "image/png")) for i in range(3)]
    r = client.post("/api/enroll", data={"label": "லௌ"}, files=files)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["label"] == "லௌ" and d["total_images"] == 3
    assert (service.support_dir / "லௌ").is_dir()
    after = _post(client, "/api/predict", square_png()).json()
    assert after["known"] and after["label"] == "லௌ"
    cs = {c["letter"]: c for c in client.get("/api/classes").json()}
    assert cs["லௌ"]["enrolled"] and cs["லௌ"]["n_support"] == 3


def test_models_and_switch(client):
    ms = client.get("/api/models").json()
    ids = {m["id"] for m in ms}
    assert {"mnv3-onnx", "mnv3-torch", "conv4-protonet"} <= ids
    assert [m["id"] for m in ms if m["active"]] == ["mnv3-onnx"]
    assert client.post("/api/models/select", json={"id": "nope"}).status_code == 404
    data = Path(one_per_kind()["compound"]["path"]).read_bytes()
    try:
        h = client.post("/api/models/select", json={"id": "conv4-protonet"}).json()
        assert h["model_id"] == "conv4-protonet" and h["ready"] and h["input_size"] == 48
        d = _post(client, "/api/predict?gradcam=true", data).json()
        assert len(d["top"]) == 5 and d["input_size"] == 48 and d["gradcam"]
    finally:
        h = client.post("/api/models/select", json={"id": "mnv3-onnx"}).json()
    assert h["model_id"] == "mnv3-onnx"


def test_index_and_fonts_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "app.js" in r.text
    for f in ("NotoSansTamil.ttf", "NotoSerifTamil.ttf"):
        assert client.get(f"/static/fonts/{f}").status_code == 200
