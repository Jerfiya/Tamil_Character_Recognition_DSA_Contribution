# Interactive UI: plan

## What exists and is reused (no re-implementation)
| Need | Project code used |
|---|---|
| Image decode + normalisation | `preprocessing.composite_to_gray` + `preprocessing.preprocess_gray` (the function that built the training cache) |
| Embedding model | `support_registry.EmbeddingModel` reading `models/deploy.json` (ONNX, or PyTorch from `embedder.pt`) |
| Classification, calibration, open set | `predict.Recognizer.predict_u8` (temperature from `calibrate.py`, VAL max-logit threshold) |
| Letter registry / enrolment | `support_registry.enroll`, `build_prototypes`, `list_support` (`data/support_set/<label>/`) |
| Letter knowledge | `tamil_text.parse_label`, `VOWELS`, `CONSONANTS`, `expected_grid`, `nfc` |
| Grad-CAM | `explain.gradcam` |
| Other runs' scoring | `export_onnx.scoring_spec` + each run's `calibration_best_val_loss.json` |
| Paths, settings | `config.yaml` (new `ui:` section) |

## Default model
Picked automatically: the best test 5-way 5-shot accuracy among available models in
`reports/model_comparison.csv`, ONNX preferred on ties → **ProtoNet · MobileNetV3-L (ONNX)**
(`runs/20261117_020113_protonet_mobilenetv3_large_B_abl_mnv3_pre96`, 95.6%).

## Architecture
```
ui/run.py        launcher: 127.0.0.1:8000, or --lan → HTTPS 0.0.0.0:8443 with a self-signed cert
ui/server.py     FastAPI: validation, HTTP errors, static files (thin)
ui/service.py    RecognitionService: model load/warm-up/switch, predict, stream, practice, enrol (thread-safe)
ui/static/       index.html, style.css, app.js, fonts/ (Noto Sans/Serif Tamil, OFL)
tests/ui/        test_api.py, test_parity.py, test_browser.py (Playwright)
```

## API
`GET /api/health`, `GET /api/classes`, `POST /api/predict[?gradcam=true]`, `POST /api/predict/stream`,
`POST /api/enroll`, `POST /api/practice/check`, `GET /api/models`, `POST /api/models/select`.

## Build order
1. Service + API + pytest (incl. parity with `predict.py`).
2. Front end: laptop → tablet → mobile, then camera mode, then loading states; screenshot each.
3. Playwright suite on iPhone 13, small Android, iPad portrait/landscape, 1440 laptop, fake camera,
   slow-3G + 4× CPU throttling; screenshots to `reports/ui_screenshots/`.
