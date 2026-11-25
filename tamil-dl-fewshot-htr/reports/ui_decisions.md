# Interactive UI: decisions log

| # | Decision | Why |
|---|---|---|
| 1 | Default model = ProtoNet MobileNetV3-L **ONNX**, chosen in code from `model_comparison.csv` | Best test 5w5s (95.6%); ONNX is the deployed export and the fastest on CPU. |
| 2 | Default model is served by `predict.Recognizer` itself | Guarantees identical outputs to `predict.py` (verified: `tests/ui/test_parity.py`, 20 images, labels/top-5/confidences/scores match). |
| 3 | Upload bytes are decoded with the same steps as `read_image_gray` (`cv2.imdecode` → Pillow fallback → `composite_to_gray`) | `read_image_gray` only takes a path; writing temp files per request would be slower. The steps are identical, and the parity test proves it. |
| 4 | Selectable models: MNv3 ONNX, MNv3 PyTorch, Conv4 ProtoNet, Conv4 ProtoNet←ArcFace, Conv4 ArcFace | Only prototype-embedding methods can serve an open registry. Matching/Relation/Siamese/MAML need episode-specific heads (`scoring_spec` rejects them). |
| 5 | Non-default models keep prototypes in `ui/cache/<id>/prototypes.npz` | So switching never overwrites the deployed `data/support_set/prototypes.npz`. Built from the same support images on first use (about 13 s for Conv4), then cached. |
| 6 | Calibration per model: deployed = `deploy.json`; runs = `calibration_best_val_loss.json` | No thresholds are invented. The VAL-calibrated open-set threshold is used as stored. |
| 7 | Enrolment always goes through `support_registry.enroll` with the deployed ONNX model, then the active model's cache is rebuilt and other caches dropped | The deployed registry and every model stay consistent. No retraining. |
| 8 | Labels: NFC, 1–4 Tamil code points, must parse as a Tamil letter, resolved path must sit directly under the support folder | Prevents path traversal and junk folders (tested with `../`, `..\\`, absolute paths, NUL, Latin text). |
| 9 | Upload limit 5 MB and content-type allow-list (PNG/JPEG/WebP/BMP) server-side, plus decode check; the same checks client-side | Spec. Raw errors are never returned (a generic handler replaces tracebacks). |
| 10 | "How the model saw it" shows the model's real input size (96×96 for MNv3, 48×48 for Conv4), not a fixed 64×64 | Showing the actual tensor is more honest than resizing to 64. |
| 11 | Live camera "stable" = the same known letter for `ui.stable_frames` (5) consecutive frames, kept per client session on the server (LRU of 64 sessions) | The spec wants the flag from `/api/predict/stream`. |
| 12 | Live mode sends a 224 px JPEG crop of the guide box about 2.5×/s (380 ms after each reply, never overlapping) | Small payload; no request pile-up on slow phones. |
| 13 | The crop sent from the camera is the guide-box region of the raw (un-mirrored) frame | The mirror is display-only; the model must see real handwriting orientation. |
| 14 | Auto-predict 600 ms after the last stroke, Recognize mode only. Practice uses an explicit Check | Multi-part Tamil letters (கொ, ஔ…) have natural pauses; auto-checking in Practice would mark half-written letters wrong. |
| 15 | Tablet layout also applies to touch screens up to 1194 px (`pointer: coarse`) | iPad landscape is 1080–1194 px wide; the three-column laptop layout is cramped there and its targets are too small for fingers. |
| 16 | The drawing canvas stays light ("paper") in dark mode | Ink on paper is what the model was trained on; the preprocessing handles polarity, but users expect paper. |
| 17 | Fonts: Noto Sans/Serif Tamil (OFL), instanced to wght 400–700 and subset to Tamil + Latin-1 as WOFF2 (42 KB + 46 KB); the original TTFs are kept as fallback | Works offline. On slow 3G the 760 KB TTFs delayed first paint by more than 15 s. |
| 18 | Spelling-variant warning (தெள/லெள/ளெள ↔ தௌ/லௌ/ளௌ) shown only when the other spelling exists among the classes; saving requires a tick | Avoids nagging for every ௌ letter while protecting the three known ambiguous pairs. |
| 19 | History = last 20 predictions in `localStorage` (64 px thumbnails, no Grad-CAM), every access in try/catch | Small, private, survives reloads; the UI works when storage is blocked. |
| 20 | `--lan` binds 0.0.0.0 over HTTPS with a self-signed certificate in `ui/cache/tls/` (git-ignored) and prints a warning | Phones only allow `getUserMedia` on HTTPS or localhost. The default stays 127.0.0.1. |
| 21 | Tests use a temporary copy of `data/support_set` | Enrolment tests must not change the real registry; dataset and checkpoints are read-only. |
| 22 | Grad-CAM for the ONNX model uses the identical weights from `models/embedder.pt` (PyTorch), loaded lazily | ONNX Runtime has no gradients. |

## Known gaps / degraded
- The open-set threshold is the deployed VAL one. It rejects clear non-letters (a square) but accepts many
  scribbles with low confidence (red bars). This matches the reported weakness: against the full 246-letter
  registry, rejection is weak.
- Canvas strokes and phone photos differ from the scanned 48×48 training images, so accuracy on drawings is
  lower than the reported test accuracy.
- Zoom and torch controls appear only where the browser exposes them (mostly Chrome on Android).
- `runs/` is not in the GitHub repository, so on a fresh clone only the two MobileNetV3 models (ONNX/PyTorch)
  are selectable. The others show as "unavailable" with the reason.
