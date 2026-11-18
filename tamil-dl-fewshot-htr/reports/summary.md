# Final summary: few-shot handwritten Tamil character recognition

All numbers are on held-out TEST data, read once per checkpoint, from the checkpoint picked on **validation
loss**. Full tables: `model_comparison.md` / `.csv`. Figures: `dashboard.html`. Every choice: `decisions.md`.

## Data (from the audit)
247 classes, 14,463 images (all 48 × 48 PNG), 258 near-duplicates removed before splitting, and 3 blank images
dropped. Nothing was corrupt, zipped, nested or empty. Missing from the 18 × 12 grid: **தௌ, லௌ, ளௌ**.
**தெள / லெள / ளெள** are kept as separate classes and flagged as possible spelling variants of those three.
8 classes have fewer than 15 images (ணே ணெ ழா ழ: 4; ணு ணூ: 5; ழி ழீ: 6).

## Best model
**ProtoNet + ImageNet-pretrained MobileNetV3-L at 96 px** (`runs/…_abl_mnv3_pre96`). It has the lowest val
loss of any run (0.2204), after only 8 epochs. Exported to `models/embedder.onnx`. ONNX matches PyTorch
(max |diff| 1.1e-5). CPU latency is 1.4 ms/image with onnxruntime and about 5 ms with PyTorch.

| Protocol B, novel letters (never trained on) | 1-shot | 5-shot |
|---|---|---|
| 5-way | 89.7% | **95.6% ± 0.4** |
| 20-way | — | 88.0% |
| all 37 novel letters | — | 83.5% |
| hard-novel 5-way (consonant & vowel sign seen in base) | — | 95.3% |

## Method comparison (Conv4 backbone, Protocol B novel classes, 15 epochs)

| method | 5w1s | 5w5s | all-way 5s | macro F1 | ECE (before → after) |
|---|---|---|---|---|---|
| Matching Net | 88.3 | 95.0 ± 0.4 | 80.6 | 0.786 | 0.166 → 0.032 |
| ProtoNet (3 seeds) | 87.6 ± 0.1 | 94.4 ± 0.3 | ~80 | ~0.79 | ~0.13 → 0.045 |
| ArcFace → ProtoNet fine-tune | 87.2 | 94.6 ± 0.5 | 81.2 | 0.783 | 0.107 → 0.046 |
| Siamese (triplet) | 89.0 | 94.1 ± 0.5 | 76.4 | 0.759 | 0.028 → 0.026 |
| Relation Net | 86.8 | 94.0 ± 0.5 | 76.3 | 0.750 | 0.052 → 0.029 |
| ArcFace (cosine prototypes) | 84.9 | 93.2 ± 0.5 | 76.8 | 0.750 | 0.046 → 0.039 |
| Baseline++ | 84.3 | 92.4 ± 0.5 | 75.4 | 0.734 | 0.072 → 0.040 |
| FO-MAML (8 epochs, 100 eps/cell) | 86.1 | 71.3 ± 5.4 | — | 0.622 | 0.415 → 0.049 |

The episodic metric methods are within about 1 point of each other at 5-way 5-shot. The differences show
up at all-way, where Matching Net and the ProtoNet variants lead. FO-MAML is unstable: its 5-shot result is
worse than its 1-shot result.

**Protocol A vs B:** ProtoNet Conv4 scores 95.3% (A, known letters) vs 94.4% (B, novel letters) at 5-way
5-shot, a gap of under 1 point. At all-way, A is much harder: all 247 letters vs 37 (60.9% vs 80.3%). The
supervised CE baseline (A only) reaches 93.5% 5-way and 54.4% all-way, below ProtoNet's 95.3% / 60.9%.
The 40-epoch ProtoNet (epoch study) reaches 96.6% 5-way and 68.8% all-way on A, so longer training helps.

## Training findings (validation only)
- Best epoch was the last or second-to-last epoch in **every** run. No run tripped the overfitting detector.
- The 40-epoch study bottoms out at epoch 36. **The 15-epoch final runs under-train.** Recommended: ~44 epochs.
- LR: 3e-3 is better than the 1e-3 used. Pretraining is the largest single gain: +4.5 pp over Conv4,
  and +18.8 pp over the same network trained from scratch.
- The negative generalisation gap (train loss > val loss) comes from augmentation plus 20-way training
  vs clean 10-way validation. It is not underfitting.

## Calibration and open set (best model)
- Temperature scaling (fitted on val): ECE 0.164 → 0.044.
- Episode-level open set (half the test letters enrolled): AUROC 0.938. The val threshold rejects 59% of
  unseen letters and falsely rejects 3.5% of enrolled ones.
- Against the full 246-letter registry, rejection is weak: an unseen letter is almost always close to
  some enrolled letter.
- Transductive refinement: +3.6 pp at 5-way 1-shot and +5.5 pp at 10-way 1-shot. It adds nothing measurable at 5-shot.
- Source-aware: support and queries drawn from different source groups showed no drop. Source groups come
  from file names only (78% "other"), so this test is coarse.

## Weakest letters and confusions (best model, novel letters)
Weakest F1: லெள 0.30, மௌ 0.32, வௌ 0.35, றோ 0.61, ஞௌ 0.64, ணௌ 0.67, டி 0.71.
Most-confused pairs: லெள↔வௌ, னீ↔ளீ, மௌ↔வௌ, பௌ↔மௌ, தோ↔றோ, டி↔ழ், ஞௌ↔நௌ.
Most errors are in **two-part letters** (ொ ோ ௌ) and in similar left-hand consonant shapes. Several
test images are unusable (solid bars, dark shaded backgrounds) and should be cleaned from the dataset.

## Enrolment without retraining
6 novel letters, enrolled from 5 images each: top-1 50–96%, top-5 79–96%. The missing letter
**தௌ** was enrolled from 5 font-rendered glyphs (a demo; no handwriting samples exist). 5/5 probe renders
were recognised, and 4 of the 5 enrolled images classify back as தௌ.

## Next steps
1. Retrain the best recipe at ~44 epochs and lr 3e-3. Combine ArcFace pretraining with the
   MobileNetV3/EfficientNet backbone.
2. Collect more samples for the 8 classes under 15 images (ணே ணெ ழா ழ ணு ணூ ழி ழீ) and for the
   ௌ/ொ/ோ family.
3. Decide whether தெள / லெள / ளெள are தௌ / லௌ / ளௌ, and rename the folders if so.
4. Run the dropped ablations (`run_batch3.sh`) on a GPU or overnight.
