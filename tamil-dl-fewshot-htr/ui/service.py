"""ML service behind the web UI: a thin, thread-safe wrapper around the project's own code.

Nothing here re-implements preprocessing or scoring:

* images are decoded exactly like ``preprocessing.read_image_gray`` and normalised with
  ``preprocessing.preprocess_gray`` (the function that built the training cache);
* the deployed model is ``support_registry.EmbeddingModel`` and classification is
  ``predict.Recognizer.predict_u8`` (temperature-calibrated softmax + VAL open-set threshold);
* enrolment is ``support_registry.enroll`` / ``build_prototypes`` (no retraining);
* Grad-CAM is ``explain.gradcam``.

Other trained runs (Conv4 ProtoNet, ArcFace) can be selected; their calibration comes from the
run's own ``calibration_best_val_loss.json`` and their prototypes are built from the same
support set into ``ui/cache/<model id>/prototypes.npz``.
"""
from __future__ import annotations

import base64
import collections
import json
import logging
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

UI_DIR = Path(__file__).resolve().parent
SRC = UI_DIR.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tamil_text as tt  # noqa: E402
from config import ROOT, load_config  # noqa: E402
from predict import Prediction, Recognizer  # noqa: E402
from preprocessing import composite_to_gray, preprocess_gray  # noqa: E402
from support_registry import EmbeddingModel, build_prototypes, enroll, list_support  # noqa: E402

log = logging.getLogger("ui.service")

# --- Tamil letter knowledge ------------------------------------------------------------------
_CONS_ROMAN = ["k", "ṅ", "c", "ñ", "ṭ", "ṇ", "t", "n", "p", "m", "y", "r", "l", "v", "ḻ", "ḷ", "ṟ", "ṉ"]
_VOWEL_ROMAN = ["a", "ā", "i", "ī", "u", "ū", "e", "ē", "ai", "o", "ō", "au"]
CATEGORY_TA = {"vowel": "உயிர்", "consonant": "மெய்", "compound": "உயிர்மெய்", "variant": "உயிர்மெய்",
               "aytham": "ஆய்தம்", "other": "பிற"}
CATEGORY_EN = {"vowel": "Vowel", "consonant": "Consonant", "compound": "Consonant-vowel",
               "variant": "Consonant-vowel (variant spelling)", "aytham": "Aytham", "other": "Other"}
_TAMIL_RE = re.compile(r"^[஀-௿]{1,4}$")


def transliterate(letter: str) -> str:
    """ISO 15919-style romanisation of one Tamil letter (consonant row + vowel sign)."""
    p = tt.parse_label(letter)
    s = tt.nfc(letter)
    if p["kind"] == "aytham":
        return "ḵ"
    if p["kind"] == "vowel":
        return _VOWEL_ROMAN[tt.VOWELS.index(s)]
    if p["consonant"] is None:
        return s
    c = _CONS_ROMAN[tt.CONSONANTS.index(p["consonant"])]
    if p["kind"] == "consonant":
        return c
    if p["kind"] == "variant":
        return c + "eḷ"
    return c + _VOWEL_ROMAN[tt.VOWEL_SIGNS.index(p["vowel"])]


def letter_info(letter: str) -> Dict[str, Any]:
    """Category, consonant row, vowel sign and transliteration of a letter."""
    p = tt.parse_label(letter)
    kind = p["kind"] or "other"
    vowel_col = None
    if kind == "vowel":
        vowel_col = letter
    elif kind in ("compound",) and p["vowel"] is not None:
        vowel_col = tt.vowel_sign_name(p["vowel"])
    elif kind == "variant":
        vowel_col = "ஔ"
    return {"letter": tt.nfc(letter), "kind": kind, "category_ta": CATEGORY_TA[kind],
            "category_en": CATEGORY_EN[kind], "translit": transliterate(letter),
            "consonant_row": (p["consonant"] + tt.PULLI) if p["consonant"] else None,
            "vowel": vowel_col,
            "variant_of": tt.nfc(p["consonant"] + "ௌ") if kind == "variant" else None}


def variant_pair(label: str) -> Optional[str]:
    """The other spelling of a ெள / ௌ letter (தெள <-> தௌ), else None."""
    p = tt.parse_label(label)
    if p["kind"] == "variant":
        return tt.nfc(p["consonant"] + "ௌ")
    if p["kind"] == "compound" and p["vowel"] == "ௌ":
        return tt.nfc(p["consonant"] + tt.VARIANT_AU)
    return None


class LabelError(ValueError):
    """Raised for labels that are not a single Tamil letter (or would escape the support folder)."""


def sanitize_label(raw: str, support_dir: Path) -> str:
    """NFC-normalise and validate an enrol label; returns the safe label or raises LabelError."""
    label = tt.nfc((raw or "").strip())
    if not label or not _TAMIL_RE.match(label) or (tt.parse_label(label)["kind"] or "other") == "other":
        raise LabelError("label must be one Tamil letter")
    target = (support_dir / label).resolve()
    if target.parent != support_dir.resolve():
        raise LabelError("invalid label")
    return label


# --- image helpers ---------------------------------------------------------------------------
class ImageError(ValueError):
    """Raised for uploads that are not decodable images."""


def decode_gray(data: bytes) -> np.ndarray:
    """Bytes -> uint8 grayscale, the same decode steps as ``preprocessing.read_image_gray``."""
    if not data:
        raise ImageError("empty file")
    arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if arr is None:
        try:
            import io
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                arr = np.asarray(im.convert("RGBA"))[:, :, [2, 1, 0, 3]]
        except Exception as exc:
            raise ImageError("not a readable image") from exc
    if arr.size == 0 or arr.shape[0] < 2 or arr.shape[1] < 2:
        raise ImageError("image too small")
    return composite_to_gray(arr)


def png_b64(img: np.ndarray, upscale: int = 1) -> str:
    """uint8 gray/RGB array -> data URL (nearest-neighbour upscaled for a crisp preview)."""
    if upscale > 1:
        img = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_NEAREST)
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", img)
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


# --- models ----------------------------------------------------------------------------------
class TorchRunEmbedder(EmbeddingModel):
    """A trained run's checkpoint exposed through the ``EmbeddingModel`` interface."""

    def __init__(self, net: Any, spec: Dict[str, Any]) -> None:  # noqa: D107 - no super(): no deploy.json
        import torch
        self.models_dir = None
        self.spec, self.pre, self.size, self.backend, self.net = spec, spec["preprocess"], int(spec["input_size"]), "torch", net

        def run(x: np.ndarray) -> np.ndarray:
            with torch.no_grad():
                return net.features(torch.from_numpy(x)).numpy()
        self._run = run


def _build_net(cfg: Dict[str, Any], state: Dict[str, Any], n_train: int) -> Any:
    from config import _wrap
    from fewshot import build_method
    cfg = _wrap(cfg)
    cfg["method"]["init_from"] = None
    cfg["model"]["pretrained"] = False
    net = build_method(cfg, n_train)
    net.load_state_dict(state)
    net.eval()
    return net


def load_run_model(run_dir: Path) -> TorchRunEmbedder:
    """Checkpoint + config snapshot + VAL calibration of a run -> embedder with a deploy-style spec."""
    import torch
    import yaml
    from export_onnx import scoring_spec
    cfg = yaml.safe_load((run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))
    ck = torch.load(run_dir / "checkpoints" / "best_val_loss.pt", map_location="cpu", weights_only=False)
    net = _build_net(cfg, ck["ema"] if ck.get("ema") is not None else ck["model"], len(ck["train_classes"]))
    size = int(cfg["model"]["input_size"])
    with torch.no_grad():
        dim = int(net.features(torch.zeros(1, 1, size, size)).shape[1])
    scoring = scoring_spec(SimpleNamespace(model=net, method=str(cfg["method"]["name"]), embedding_dim=lambda: dim))
    cal = json.loads((run_dir / "calibration_best_val_loss.json").read_text(encoding="utf-8"))
    spec = {"run": run_dir.name, "method": cfg["method"]["name"], "backbone": cfg["model"]["backbone"],
            "input_size": size, "embedding_dim": dim, "preprocess": cfg["preprocess"], **scoring,
            "calibration_temperature": float(cal["temperature"]),
            "openset_threshold_max_logit": float(cal["openset_threshold"])}
    return TorchRunEmbedder(net, spec)


class UIRecognizer(Recognizer):
    """``predict.Recognizer`` with a configurable prototype cache (so other models never
    overwrite the deployed model's ``prototypes.npz``)."""

    def __init__(self, model: EmbeddingModel, support_dir: Path, proto_path: Path, top_k: int = 5) -> None:  # noqa: D107
        self.model, self.support_dir, self.proto_path, self.top_k = model, support_dir, proto_path, top_k
        self.temp = float(model.spec["calibration_temperature"])
        self.threshold = float(model.spec["openset_threshold_max_logit"])
        self.reload()

    def reload(self) -> None:
        """Load prototypes, rebuilding them when missing or built by another model."""
        ok = False
        if self.proto_path.exists():
            z = np.load(self.proto_path, allow_pickle=False)
            ok = "model_run" in z.files and str(z["model_run"]) == self.model.spec["run"]
        if not ok:
            self.proto_path.parent.mkdir(parents=True, exist_ok=True)
            build_prototypes(self.model, self.support_dir, out=self.proto_path)
            z = np.load(self.proto_path, allow_pickle=False)
        self.labels = [str(x) for x in z["labels"]]
        self.protos = z["prototypes"]
        self.counts = [int(c) for c in z["counts"]]


@dataclass
class ModelEntry:
    id: str
    name: str
    source: str
    backend: str
    run: Optional[str] = None
    available: bool = True
    reason: str = ""
    test_acc_5w5s: Optional[float] = None


@dataclass
class ServiceState:
    status: str = "loading"            # loading | ready | switching | error
    message: str = "Loading model…"
    latencies: Deque[float] = field(default_factory=lambda: collections.deque(maxlen=500))


class RecognitionService:
    """Loads the model once, keeps it warm, and serves predictions / enrolment thread-safely."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.cfg = cfg or load_config()
        ui = self.cfg["ui"]
        self.models_dir = ROOT / self.cfg["paths"]["models"]
        self.support_dir = ROOT / self.cfg["paths"]["support_set"]
        self.cache_dir = ROOT / ui["cache_dir"]
        self.top_k = int(self.cfg["deploy"]["top_k"])
        self.stable_frames = int(ui["stable_frames"])
        self.max_upload = int(float(ui["max_upload_mb"]) * 1024 * 1024)
        self.state = ServiceState()
        self.lock = threading.RLock()
        self.rec: Optional[Recognizer] = None
        self.active: Optional[ModelEntry] = None
        self._gradnet: Optional[Tuple[str, Any]] = None
        self._streams: "collections.OrderedDict[str, Deque[str]]" = collections.OrderedDict()
        self.models = self._discover_models()
        self.trained_labels = set(tt.nfc(x["label"] if isinstance(x, dict) else x) for x in json.loads(
            (ROOT / self.cfg["paths"]["processed"] / "class_names.json").read_text(encoding="utf-8")))

    # -- model catalogue ------------------------------------------------------------------------
    def _discover_models(self) -> List[ModelEntry]:
        acc: Dict[str, float] = {}
        mc = ROOT / "reports" / "model_comparison.csv"
        if mc.exists():
            import pandas as pd
            df = pd.read_csv(mc, encoding="utf-8-sig")
            acc = dict(zip(df["run"], df["test_acc_5w5s"]))
        deploy_run = None
        if (self.models_dir / "deploy.json").exists():
            deploy_run = json.loads((self.models_dir / "deploy.json").read_text(encoding="utf-8"))["run"]
        out = []
        for m in self.cfg["ui"]["models"]:
            e = ModelEntry(id=m["id"], name=m["name"], source=m["source"], backend=m.get("backend", "torch"), run=m.get("run"))
            if e.source == "deploy":
                e.run = deploy_run
                f = self.models_dir / (("embedder.onnx") if e.backend == "onnx" else "embedder.pt")
                if deploy_run is None or not f.exists():
                    e.available, e.reason = False, f"{f.name} / deploy.json missing"
            else:
                ck = ROOT / self.cfg["paths"]["runs"] / str(e.run) / "checkpoints" / "best_val_loss.pt"
                if not ck.exists():
                    e.available, e.reason = False, "checkpoint not present (runs/ is not in the repository)"
            e.test_acc_5w5s = acc.get(e.run) if e.run else None
            out.append(e)
        return out

    def default_model_id(self) -> str:
        """Best deployable model by test 5-way 5-shot accuracy; ONNX preferred over PyTorch on ties."""
        avail = [m for m in self.models if m.available]
        if not avail:
            raise RuntimeError("no deployable model found (models/deploy.json and runs/ are missing)")
        avail.sort(key=lambda m: (-(m.test_acc_5w5s or 0.0), 0 if m.backend == "onnx" else 1))
        return avail[0].id

    def _entry(self, model_id: str) -> ModelEntry:
        for m in self.models:
            if m.id == model_id:
                return m
        raise KeyError(model_id)

    def _make_recognizer(self, e: ModelEntry) -> Recognizer:
        if e.source == "deploy":
            if e.backend == "onnx":
                # exactly what predict.py builds -> identical predictions
                return Recognizer(self.models_dir, self.support_dir, "onnx", self.top_k)
            model = EmbeddingModel(self.models_dir, "torch")
            return UIRecognizer(model, self.support_dir, self.cache_dir / e.id / "prototypes.npz", self.top_k)
        model = load_run_model(ROOT / self.cfg["paths"]["runs"] / str(e.run))
        return UIRecognizer(model, self.support_dir, self.cache_dir / e.id / "prototypes.npz", self.top_k)

    def load(self, model_id: Optional[str] = None) -> None:
        """Load (or switch to) a model and warm it up. Keeps the previous model on failure."""
        mid = model_id or self.default_model_id()
        e = self._entry(mid)
        if not e.available:
            raise RuntimeError(f"model {mid} unavailable: {e.reason}")
        prev = (self.state.status, self.state.message)
        self.state.status, self.state.message = ("switching" if self.rec else "loading"), f"Loading {e.name}…"
        t0 = time.perf_counter()
        try:
            rec = self._make_recognizer(e)
            rec.predict_u8(np.zeros((2, rec.model.size, rec.model.size), np.uint8))       # warm-up
            blank = np.full((128, 128), 255, np.uint8)
            cv2.line(blank, (30, 64), (98, 64), 0, 8)
            rec.predict_gray(blank)
        except Exception:
            self.state.status, self.state.message = prev if self.rec else ("error", "Model failed to load")
            log.exception("loading %s failed", mid)
            raise
        with self.lock:
            self.rec, self.active, self._gradnet = rec, e, None
            self._streams.clear()
        self.state.status, self.state.message = "ready", "Ready"
        log.info("model %s ready in %.1fs (%d letters enrolled)", mid, time.perf_counter() - t0, len(rec.labels))

    # -- info --------------------------------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        lat = sorted(self.state.latencies)
        pct = (lambda q: round(lat[min(len(lat) - 1, int(q * len(lat)))], 2)) if lat else (lambda q: None)
        rec = self.rec
        return {"status": self.state.status, "message": self.state.message, "ready": self.state.status == "ready",
                "model": self.active.name if self.active else None, "model_id": self.active.id if self.active else None,
                "backend": self.active.backend if self.active else None, "device": "cpu",
                "run": rec.model.spec["run"] if rec else None,
                "input_size": rec.model.size if rec else None,
                "class_count": len(self.trained_labels), "registry_size": len(rec.labels) if rec else 0,
                "latency_ms": {"p50": pct(0.5), "p95": pct(0.95), "n": len(lat)}}

    def models_info(self) -> List[Dict[str, Any]]:
        return [{"id": m.id, "name": m.name, "backend": m.backend, "run": m.run, "available": m.available,
                 "reason": m.reason, "test_acc_5w5s": m.test_acc_5w5s,
                 "active": self.active is not None and m.id == self.active.id} for m in self.models]

    def classes(self) -> List[Dict[str, Any]]:
        """Every letter of the standard chart plus any extra enrolled/trained labels."""
        rec = self.rec
        enrolled = dict(zip(rec.labels, getattr(rec, "counts", [0] * len(rec.labels)))) if rec else {}
        if rec and not hasattr(rec, "counts"):
            z = np.load(self.support_dir / "prototypes.npz", allow_pickle=False)
            enrolled = dict(zip([str(x) for x in z["labels"]], [int(c) for c in z["counts"]]))
        chart = list(tt.VOWELS) + [tt.AYTHAM] + [tt.nfc(c + tt.PULLI) for c in tt.CONSONANTS] + tt.expected_grid()
        extra = sorted((set(enrolled) | self.trained_labels) - set(chart))
        out = []
        for lab in chart + extra:
            info = letter_info(lab)
            info.update({"enrolled": lab in enrolled, "n_support": int(enrolled.get(lab, 0)),
                         "trained": lab in self.trained_labels, "in_chart": lab in chart,
                         "variant_pair": variant_pair(lab)})
            out.append(info)
        return out

    # -- prediction ----------------------------------------------------------------------------
    def check_size(self, data: bytes) -> None:
        if len(data) > self.max_upload:
            raise ImageError(f"image larger than {self.max_upload // (1024 * 1024)} MB")

    def _record(self, ms: float) -> None:
        self.state.latencies.append(ms)
        n = len(self.state.latencies)
        if n % 50 == 0:
            lat = sorted(self.state.latencies)
            log.info("latency over last %d: p50 %.1f ms, p95 %.1f ms", n, lat[n // 2], lat[int(0.95 * (n - 1))])

    def _require(self) -> Recognizer:
        if self.rec is None or self.state.status not in ("ready", "switching"):
            raise RuntimeError("model not ready")
        return self.rec

    def _format(self, pr: Prediction, k: int) -> Dict[str, Any]:
        top = [{"letter": lab, "confidence": round(c, 4), **{x: letter_info(lab)[x] for x in ("translit", "category_ta", "category_en")}}
               for lab, c in pr.top[:k]]
        best = letter_info(pr.top[0][0])
        return {"label": pr.label, "known": bool(pr.known), "score": round(pr.score, 4),
                "threshold": round(self.rec.threshold, 4) if self.rec else None,
                "confidence": top[0]["confidence"], "best": best, "top": top}

    def predict_bytes(self, data: bytes, gradcam: bool = False, k: int = 5) -> Dict[str, Any]:
        """Decode + shared preprocessing + calibrated classification (+ optional Grad-CAM)."""
        self.check_size(data)
        gray = decode_gray(data)
        t0 = time.perf_counter()
        with self.lock:
            rec = self._require()
            pre = rec.model.preprocess(gray)
            pr = rec.predict_u8(pre[None])[0]
            ms = (time.perf_counter() - t0) * 1000
            out = self._format(pr, k)
            out["blank"] = bool(pre.max() == 0)
            out["preprocessed"] = png_b64(pre, max(1, 192 // pre.shape[0]))
            out["input_size"] = int(pre.shape[0])
            if gradcam and not out["blank"]:
                try:
                    out["gradcam"] = self._gradcam(pre, rec.labels.index(pr.top[0][0]))
                except Exception:
                    log.exception("Grad-CAM failed")
                    out["gradcam"] = None
        self._record(ms)
        out["latency_ms"] = round(ms, 2)
        out["model"] = self.active.name if self.active else None
        return out

    def predict_stream(self, data: bytes, session: str) -> Dict[str, Any]:
        """Light live-camera prediction: top-3 and a 'stable' flag over the last N frames."""
        res = self.predict_bytes(data, gradcam=False, k=3)
        key = (session or "default")[:64]
        with self.lock:
            hist = self._streams.setdefault(key, collections.deque(maxlen=self.stable_frames))
            self._streams.move_to_end(key)
            while len(self._streams) > 64:
                self._streams.popitem(last=False)
            hist.append(res["label"] if (res["known"] and not res["blank"]) else "∅")
            run = 0
            for x in reversed(hist):
                if x != hist[-1] or x == "∅":
                    break
                run += 1
        return {"label": res["label"], "known": res["known"], "blank": res["blank"], "confidence": res["confidence"],
                "best": res["best"], "top": res["top"], "stable": run >= self.stable_frames, "streak": run,
                "latency_ms": res["latency_ms"]}

    def practice_check(self, target: str, data: bytes) -> Dict[str, Any]:
        target = tt.nfc(target)
        res = self.predict_bytes(data, k=5)
        letters = [t["letter"] for t in res["top"]]
        correct = res["known"] and letters[0] == target
        close = (not correct) and target in letters[:3]
        if res["blank"]:
            msg_ta, msg_en = "எழுத்தை வரையவும்", "Draw the letter first"
        elif correct:
            msg_ta, msg_en = "சரி!", "Correct!"
        elif close:
            msg_ta, msg_en = "அருகில்! மீண்டும் முயற்சி", "Close! Try once more"
        else:
            msg_ta, msg_en = "மீண்டும் முயற்சி", "Try again"
        return {"target": target, "correct": bool(correct), "close": bool(close), "message_ta": msg_ta,
                "message_en": msg_en, "prediction": res}

    def _gradcam(self, pre: np.ndarray, target: int) -> str:
        """Grad-CAM (``explain.gradcam``) of the predicted class against all prototypes."""
        import torch
        from explain import gradcam
        rec = self.rec
        mid = self.active.id
        if self._gradnet is None or self._gradnet[0] != mid:
            net = getattr(rec.model, "net", None)
            if net is None:                         # ONNX backend: same weights from embedder.pt
                ck = torch.load(self.models_dir / rec.model.spec["torch"], map_location="cpu", weights_only=False)
                net = _build_net(ck["config"], ck["state_dict"], len(ck["train_classes"]))
            self._gradnet = (mid, net)
        net = self._gradnet[1]
        pre_cfg = rec.model.pre
        x = (pre.astype(np.float32) / 255.0 - float(pre_cfg["mean"])) / float(pre_cfg["std"])
        protos = torch.from_numpy(np.asarray(rec.protos, np.float32))
        cam = gradcam(SimpleNamespace(model=net), protos, len(protos), 1, torch.from_numpy(x)[None, None], target)
        heat = cv2.applyColorMap((np.clip(cam, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        base = np.repeat(pre[:, :, None], 3, axis=2)
        mix = (0.55 * base + 0.45 * heat).astype(np.uint8)
        return png_b64(mix, max(1, 192 // pre.shape[0]))

    # -- enrolment -----------------------------------------------------------------------------
    def enroll_images(self, raw_label: str, blobs: List[bytes]) -> Dict[str, Any]:
        """Save images to data/support_set/<label>/ and refresh prototypes (no retraining)."""
        label = sanitize_label(raw_label, self.support_dir)
        if not 1 <= len(blobs) <= 10:
            raise ImageError("send 1-10 images")
        grays = []
        for b in blobs:
            self.check_size(b)
            g = decode_gray(b)
            if preprocess_gray(g, self.rec.model.pre if self.rec else None, 48).max() == 0:
                raise ImageError("one of the images is blank")
            grays.append(g)
        t0 = time.perf_counter()
        with self.lock:
            rec = self._require()
            deploy_onnx = isinstance(rec, Recognizer) and not isinstance(rec, UIRecognizer)
            if deploy_onnx:
                enroll(rec.model, label, grays, self.support_dir)          # project code; updates prototypes.npz
            else:
                # Save the files and refresh the deployed cache with the deployed ONNX model, then
                # rebuild the active model's cache so both stay consistent.
                enroll(EmbeddingModel(self.models_dir, "onnx"), label, grays, self.support_dir)
                rec.proto_path.unlink(missing_ok=True)
            for m in self.models:                                          # other caches are now stale
                p = self.cache_dir / m.id / "prototypes.npz"
                if (self.active is None or m.id != self.active.id) and p.exists():
                    p.unlink()
            rec.reload()
            n = len(list_support(self.support_dir).get(label, []))
        log.info("enrolled %s (+%d images, %d total) in %.0f ms", label, len(blobs), n, (time.perf_counter() - t0) * 1000)
        return {"label": label, "added": len(blobs), "total_images": n, "registry_size": len(rec.labels),
                "info": letter_info(label), "variant_pair": variant_pair(label)}
