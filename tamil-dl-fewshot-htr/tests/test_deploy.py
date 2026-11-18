"""ONNX parity and enrolment of a new letter without retraining."""
import json

import numpy as np
import torch

from config import load_config
from fewshot import build_method


def _export(tmp_path):
    from export_onnx import FeatureWrapper
    cfg = load_config(None, ["model.pretrained=false", "threads=2"])
    torch.manual_seed(0)
    model = build_method(cfg, 5).eval()
    w = FeatureWrapper(model).eval()
    path = tmp_path / "embedder.onnx"
    torch.onnx.export(w, torch.zeros(1, 1, 48, 48), str(path), input_names=["image"], output_names=["embedding"],
                      dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False)
    spec = {"run": "test", "onnx": "embedder.onnx", "torch": "embedder.pt", "input_size": 48, "geometry": "euclidean",
            "scale": 0.125, "preprocess": dict(cfg.preprocess), "calibration_temperature": 1.0,
            "openset_threshold_max_logit": -1e9}
    (tmp_path / "deploy.json").write_text(json.dumps(spec), encoding="utf-8")
    from config import to_plain
    torch.save({"state_dict": model.state_dict(), "config": to_plain(cfg), "train_classes": list(range(5))},
               tmp_path / "embedder.pt")
    return w, path


def test_onnx_matches_pytorch(tmp_path):
    import onnxruntime as ort
    w, path = _export(tmp_path)
    x = torch.randn(8, 1, 48, 48)
    with torch.no_grad():
        ref = w(x).numpy()
    got = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"]).run(None, {"image": x.numpy()})[0]
    assert np.abs(ref - got).max() < 1e-4


def _letter_img(shift):
    g = np.full((64, 64), 255, np.uint8)
    g[10 + shift:50, 20:26] = 0
    g[30:36, 10:55 - shift] = 0
    return g


def test_enroll_new_letter_and_recognise(tmp_path):
    from predict import Recognizer
    from support_registry import EmbeddingModel, build_prototypes, enroll
    _export(tmp_path)
    import cv2
    sup = tmp_path / "support"
    for lab in ("அ", "க"):
        (sup / lab).mkdir(parents=True)
    a = np.full((64, 64), 255, np.uint8)
    cv2.circle(a, (32, 32), 18, 0, 5)
    k = np.full((64, 64), 255, np.uint8)
    cv2.line(k, (5, 5), (58, 58), 0, 5)
    cv2.imencode(".png", a)[1].tofile(str(sup / "அ" / "1.png"))
    cv2.imencode(".png", k)[1].tofile(str(sup / "க" / "1.png"))
    for backend in ("onnx", "torch"):
        model = EmbeddingModel(tmp_path, backend)
        reg = build_prototypes(model, sup)
        assert sorted(reg["labels"].tolist()) == ["அ", "க"]
    reg = enroll(model, "தௌ", [_letter_img(s) for s in (0, 2, 4)], sup)       # a missing letter
    assert "தௌ" in reg["labels"].tolist() and len(list((sup / "தௌ").glob("*.png"))) == 3
    rec = Recognizer(tmp_path, sup, "onnx")
    pr = rec.predict_gray(_letter_img(1))
    assert pr.label == "தௌ" and len(pr.top) == 3
