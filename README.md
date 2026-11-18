# Tamil Handwritten Character Recognition (Few-Shot)

Recognise handwritten Tamil letters from only **1 or 5 example images** per letter, including letters the model never saw during training.

![Results](tamil-dl-fewshot-htr/reports/figures/results_simple.png)

## Best model

**Prototypical Network + ImageNet-pretrained MobileNetV3-Large** (96 px input, exported to ONNX)

| Metric (on letters never seen in training) | Score |
|---|---|
| Accuracy, 5-way 5-shot | **95.6% ± 0.4** |
| Accuracy, 5-way 1-shot | 89.7% |
| Accuracy, all 37 unseen letters, 5-shot | 83.5% |
| Macro F1 | 0.82 |
| Open-set AUROC (rejecting unknown letters) | 0.94 |
| CPU time per image (ONNX Runtime) | 1.4 ms |

## Models compared (5-way 5-shot accuracy, unseen letters)

| Model | 1-shot | 5-shot | Macro F1 | Open-set AUROC |
|---|---|---|---|---|
| **ProtoNet + MobileNetV3 (pretrained)** | **89.7%** | **95.6%** | **0.82** | **0.94** |
| Matching Network | 88.3% | 95.0% | 0.79 | 0.71 |
| Prototypical Network | 87.5% | 94.4% | 0.78 | 0.92 |
| Siamese Network (triplet) | 89.0% | 94.1% | 0.76 | 0.91 |
| Relation Network | 86.8% | 94.0% | 0.75 | 0.87 |
| ArcFace | 84.9% | 93.2% | 0.75 | 0.90 |
| Baseline++ | 84.3% | 92.4% | 0.73 | 0.73 |
| MAML (first-order) | 86.1% | 71.3% | 0.62 | 0.74 |

Unless named otherwise, models use a small Conv4 backbone (~0.2 M parameters). Every score is on the test set, averaged over at least 600 random episodes.

### Evaluation metrics

- **N-way K-shot accuracy:** pick the correct letter out of N, given K example images of each.
- **95% confidence interval:** how much the accuracy could vary across test episodes.
- **Macro F1:** precision and recall, averaged so every letter counts equally.
- **Open-set AUROC:** how well the model says "I don't know" for letters it wasn't given.
- **ECE (calibration error):** whether the model's confidence matches how often it is right.
- **Latency:** time to process one image on a CPU.

## Dataset

`Tamil Letters Combined/`: **247 letter classes, 14,463 images** (48 × 48 PNG), grouped as vowels (உயிர்), consonant-vowel combinations (உயிர்மெய்) and the āytam (ஆய்தம்).

- **Protocol A:** letters seen in training.
- **Protocol B:** 37 letters held out completely, used to test new letters.

## Repository layout

```
Tamil Letters Combined/        dataset (category / letter / images)
tamil-dl-fewshot-htr/
├── src/                       data prep, training, evaluation, export, demo apps
├── tests/                     pytest suite
├── models/                    embedder.onnx (deployed model), embedder.pt, deploy.json
├── reports/                   PDF/HTML reports, comparison tables, figures
├── data/support_set/          enrolled example images used for prediction
└── config.yaml                all settings
```

## Quick start

```bash
cd tamil-dl-fewshot-htr
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python src/predict.py some_letter.png      # top-5 letters + confidence
python src/app_canvas.py                   # draw a letter with the mouse
```

To retrain, first rebuild the preprocessed arrays with `python src/prepare_data.py`, then run `python src/train_fewshot.py --set method.name=protonet`. See [`tamil-dl-fewshot-htr/README.md`](tamil-dl-fewshot-htr/README.md) for the full pipeline.

## Reports

- [Full report (PDF)](tamil-dl-fewshot-htr/reports/Tamil_FewShot_HTR_Report.pdf)
- [Summary](tamil-dl-fewshot-htr/reports/summary.md)
- [Detailed model comparison](tamil-dl-fewshot-htr/reports/model_comparison.md)
- [Detailed results figure](tamil-dl-fewshot-htr/reports/figures/results_summary.png)

## Limitations

- Most models trained for only 8–15 epochs. Accuracy was still improving, and about 44 epochs is recommended.
- 8 letters have fewer than 15 images.
- The two-part vowel-sign letters (ொ ோ ௌ) are the hardest to recognise.
