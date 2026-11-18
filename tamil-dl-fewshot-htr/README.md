# Handwritten Tamil character recognition with few-shot deep learning

Deep backbones (Conv4, ResNet-12, ResNet-18, ImageNet-pretrained MobileNetV3 / EfficientNet, a small ViT)
trained with episodic and metric-learning methods: Prototypical, Matching and Relation Networks,
Siamese/triplet, ArcFace pretraining followed by episodic fine-tuning, Baseline++, transductive
prototype refinement and first-order MAML. A letter can be recognised from 1-5 examples, and a new
letter can be enrolled **without retraining**. A plain supervised classifier serves as the reference.
The layout was inspired by *Handwritten-Digit-Recognition-using-Deep-Learning* (softmax / MLP / CNN +
live recognition); all code here is original.

Key reports: `reports/audit.md` (data), `reports/decisions.md` (every choice and why),
`reports/epoch_study.md`, `reports/ablations.md`, `reports/model_comparison.md`, `reports/summary.md`,
and the static dashboard `reports/dashboard.html`.

## Setup

```powershell
cd tamil-dl-fewshot-htr
python -m venv .venv
.venv\Scripts\activate          # Linux/WSL: source .venv/bin/activate
pip install -r requirements.txt
python -c "import torch, timm, cv2, onnxruntime; print(torch.__version__, torch.cuda.is_available())"
```
CPU works out of the box (this project was run on a laptop CPU). For an NVIDIA GPU, first install the
CUDA build of torch/torchvision from https://download.pytorch.org/whl/cu121 (or newer). On Windows,
WSL2 with the CUDA driver works too. AMP turns on automatically on CUDA. With a GPU, raise
`train.episodes_per_epoch` to 200 and use wider ResNets (`model.width: 64`).

All CLIs accept `--config config.yaml` and `--set key.sub=value` overrides.

## Data preparation (the dataset folder is only read, never written)

```powershell
python src/check_dataset_untouched.py --record   # fingerprint the dataset tree
python src/dataset_audit.py                      # -> reports/audit.md, audit.json, audit_class_counts.csv
python src/prepare_data.py                       # -> data/processed/: manifest.csv, class_names.json,
                                                 #    splits_A.csv, class_split.json, images_{48,64,96,128}.npy
```
The dataset tree is found by a recursive scan of `paths.scan_root` (default `..`). The class label is the
NFC-normalised name of the folder that directly contains the images. Empty folders are skipped and
listed as enrollable. Zip-only folders are extracted to `data/interim/extracted/`. Corrupt files are
skipped and listed. Near-duplicates are removed by perceptual hash before any split.

## Training

```powershell
python src/train_fewshot.py --set method.name=protonet train.protocol=B          # main method
python src/train_fewshot.py --set method.name=matchingnet                         # relationnet | siamese | arcface | baselinepp | maml
python src/train_fewshot.py --set method.name=protonet method.init_from=runs/<arcface_run>/checkpoints/best_val_loss.pt train.lr=0.0003
python src/train_baseline.py                                                      # supervised CE, Protocol A
python src/train_fewshot.py --lr-find --set method.name=protonet                  # LR range test -> reports/lr_finder/
bash run_all.sh ; bash run_batch1.sh ; bash run_batch2.sh ; bash run_batch3.sh    # the exact runs of this project
bash eval_queue.sh                                                                # test-evaluate every finished run once
```
Each run writes `runs/<YYYYMMDD_HHMMSS>_<method>_<backbone>_<protocol>[_tag]/` with `config_snapshot.yaml`,
`env.json`, `model_summary.txt`, `class_split.json`, `metrics_epoch.csv` (train/val loss, accuracy ± CI,
top-3, lr, gradient norms, clipped fraction, embedding norm, separation ratio, temperature, loss
components, flags), `metrics_step.csv`, `history.json` (best epochs, overfitting/plateau flags, stop
reason), `train.log`, TensorBoard logs in `tb/`, and `checkpoints/` (`best_val_loss.pt`, `best_val_acc.pt`,
`last.pt`, every 10 epochs, `crash.pt` on a NaN loss).

**Resume exactly** (model, optimiser, scheduler, EMA, RNG streams, sampler and augmenter state):
```powershell
python src/train_fewshot.py --resume runs/<run_dir>
```

## Evaluation (test data: once per checkpoint)

```powershell
python src/calibrate.py runs/<run_dir>      # temperature + open-set threshold on VALIDATION (evaluate.py calls it too)
python src/evaluate.py runs/<run_dir>       # -> runs/<run>/test/*.csv|json|npy ; refuses a second look (DONE.json)
python src/explain.py runs/<run_dir>        # Grad-CAM grid
python src/measure_latency.py               # idle-CPU latency of every run -> reports/latency.csv (used by summarize)
python src/summarize.py                     # -> reports/model_comparison.md|csv, results.csv, seeds.csv
python src/report_studies.py                # -> reports/epoch_study.md, ablations.md (validation data only)
```

## Re-plotting without retraining

```powershell
python src/visualize.py --run runs/<run_dir>   # every per-run plot -> runs/<run>/plots/
python src/visualize.py --all                  # all runs + cross-run figures + reports/dashboard.html
```

## Deployment: ONNX, registry, prediction, apps

```powershell
python src/export_onnx.py --run runs/<best_run>        # shipped: the MobileNetV3 ProtoNet run (lowest val loss); models/embedder.onnx + deploy.json, parity + latency check
python src/support_registry.py init --per-class 5      # support set from TRAIN images -> data/support_set/prototypes.npz
python src/predict.py some_letter.png                  # top-5 + calibrated confidence + "unknown" rejection
python src/predict.py folder\ --backend torch --csv out.csv
python src/app_canvas.py                               # draw with the mouse; Predict / Clear / Enroll as new letter
python src/app_webcam.py                               # webcam; or --image photo.png ; exits cleanly without a camera
```

**Enrolling a missing letter (e.g. தௌ):** put 3-5 images in `data/support_set/தௌ/` and run
`python src/support_registry.py refresh`, or `python src/support_registry.py enroll தௌ a.png b.png c.png`,
or draw it in the canvas app and press *Enroll as new letter*. No retraining is involved. Only the
letter's prototype is added.

## Tests

```powershell
python -m pytest -q
```
Covers nested/empty/zip class discovery, unicode paths, NFC normalisation, preprocessing contract and
train/inference parity, the episode sampler (N/K/Q, no support/query overlap, small classes, mixed shots,
cross-source), Protocol B disjointness, split leakage, prototype math, every backbone and method on a
dummy batch, RunLogger files, NaN abort, overfitting/plateau detectors, exact resume, ONNX parity and
enrolment.

## Known limitations
- Trained on a laptop CPU: one epoch is 50 episodes, the backbones are small, and final runs last 15 epochs.
  The 40-epoch study shows val loss still falling until epoch ~36, so **every final run is under-trained**.
  The recommended budget is ~44 epochs at lr 3e-3 (the LR comparison's winner). The numbers here are lower bounds.
- The deployed model (ImageNet-pretrained MobileNetV3, 96 px) was trained for only 8 epochs, as an ablation run.
  It still has the lowest validation loss of all runs. A full 44-epoch run of it is the obvious next step.
- Several ablations (train-way, train-shot, embedding dim, distance, EMA, input size, ResNet-12/18,
  EfficientNet-B0) were dropped for time and memory. See `reports/ablations.md` ("Not run").
- Open-set rejection works at the episode level (AUROC ~0.92), but against the full 246-letter registry an
  unseen letter is nearly always close to some enrolled letter. In the enrolment demo, 0-4% of a novel letter's
  images were rejected as "unknown" before it was enrolled. Treat "unknown" as a weak signal.
- Every source image is 48 x 48 and many are low-contrast photo crops. Some samples contain shadows or
  dark backgrounds that the preprocessing cannot separate from ink (see the misclassification grids).
  More and cleaner samples matter more than model size.
- Source groups come from file names. 78% fall into "other" (named files), so the source-aware test is
  coarse. A real per-writer split would need writer metadata.
- தெள / லெள / ளெள are kept as separate classes pending your review (see `reports/decisions.md`).
- The demo enrolment of தௌ uses font-rendered, augmented glyphs because no handwritten samples exist.
  Replace them with real handwriting in `data/support_set/தௌ/`.
- Matplotlib cannot shape Tamil, so labels in the figures are pre-rendered with HarfBuzz/FreeType
  (`src/tamil_text.py`, Nirmala UI on Windows, Noto Sans Tamil elsewhere).
