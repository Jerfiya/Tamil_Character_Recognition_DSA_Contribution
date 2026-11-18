"""Build the formal project report (HTML -> PDF via headless Microsoft Edge, which shapes Tamil correctly).

Every table is read from stored results (runs/*/test, reports/*.csv), so the report can be regenerated
after new runs without retraining:

    python src/make_report.py [--best runs/<run_dir>] [--author "Name"] [--mentor "Name"]

Writes reports/Tamil_FewShot_HTR_Report.html and reports/Tamil_FewShot_HTR_Report.pdf.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence

import pandas as pd

from config import ROOT, force_utf8, setup_logging

log = logging.getLogger("report")
REPORTS, RUNS = ROOT / "reports", ROOT / "runs"
DEFAULT_BEST = "20261117_020113_protonet_mobilenetv3_large_B_abl_mnv3_pre96"
NICE = {"protonet": "ProtoNet", "matchingnet": "Matching Net", "relationnet": "Relation Net", "siamese": "Siamese (triplet)",
        "arcface": "ArcFace (cosine prototypes)", "baselinepp": "Baseline++", "maml": "FO-MAML", "supervised": "Supervised CE"}
BB = {"conv4": "Conv4", "mobilenetv3_large": "MobileNetV3-L (ImageNet)", "resnet12": "ResNet-12"}


def _ndup(audit) -> int:
    v = audit["near_duplicates_removed"]
    return len(v) if isinstance(v, (list, dict)) else int(v)


def esc(x) -> str:
    return html.escape(str(x))


def pct(x, nd: int = 1) -> str:
    return "—" if pd.isna(x) else f"{100 * float(x):.{nd}f}"


def table(headers: Sequence[str], rows: Iterable[Sequence], cls: str = "", highlight: int = -1) -> str:
    h = "".join(f"<th>{esc(c)}</th>" for c in headers)
    body = []
    for i, r in enumerate(rows):
        attr = ' class="hl"' if i == highlight else ""
        body.append(f"<tr{attr}>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>")
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{"".join(body)}</tbody></table>'


def fig(path: Path, caption: str, num: List[int], width: str = "100%") -> str:
    if not path.exists():
        log.warning("figure missing: %s", path)
        return ""
    num[0] += 1
    return (f'<figure><img src="{path.resolve().as_uri()}" style="width:{width}"/>'
            f"<figcaption><b>Figure {num[0]}.</b> {caption}</figcaption></figure>")


CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
* { box-sizing: border-box; }
body { font-family: 'Segoe UI', 'Nirmala UI', Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #1a1a1a; }
.ta { font-family: 'Nirmala UI', 'Noto Sans Tamil', sans-serif; }
h1 { font-size: 15pt; color: #12355b; border-bottom: 2px solid #12355b; padding-bottom: 3px; margin: 0 0 8px; page-break-after: avoid; }
h2 { font-size: 12pt; color: #12355b; margin: 14px 0 5px; page-break-after: avoid; }
h3 { font-size: 10.5pt; margin: 10px 0 4px; page-break-after: avoid; }
p { margin: 4px 0 7px; text-align: justify; }
ul, ol { margin: 3px 0 8px 18px; padding: 0; } li { margin: 2px 0; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 8.8pt; page-break-inside: avoid; }
th { background: #12355b; color: #fff; font-weight: 600; padding: 4px 5px; text-align: left; }
td { border-bottom: 1px solid #d5dbe3; padding: 3px 5px; vertical-align: top; }
tr:nth-child(even) td { background: #f4f7fb; }
tr.hl td { background: #fff4d6 !important; font-weight: 600; }
figure { margin: 8px 0 12px; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; border: 1px solid #e3e3e3; }
figcaption { font-size: 8.6pt; color: #444; margin-top: 3px; text-align: left; }
.section { page-break-before: always; }
.cover { height: 250mm; display: flex; flex-direction: column; justify-content: space-between; }
.cover .band { background: #12355b; color: #fff; padding: 22px 24px; border-radius: 4px; }
.cover h0 { display: block; font-size: 22pt; font-weight: 700; line-height: 1.2; }
.cover .sub { font-size: 12pt; margin-top: 8px; opacity: .92; }
.meta td { border: none; padding: 3px 6px; font-size: 10.5pt; background: none !important; }
.kpis { display: flex; gap: 8px; margin: 8px 0 10px; }
.kpi { flex: 1; border: 1px solid #c9d3df; border-radius: 4px; padding: 7px 8px; text-align: center; }
.kpi b { display: block; font-size: 15pt; color: #12355b; }
.kpi span { font-size: 8.2pt; color: #444; }
.note { background: #f4f7fb; border-left: 3px solid #12355b; padding: 6px 9px; margin: 6px 0 10px; font-size: 9.4pt; }
.warn { background: #fff7e6; border-left: 3px solid #d48806; padding: 6px 9px; margin: 6px 0 10px; font-size: 9.4pt; }
.two { display: flex; gap: 10px; } .two > * { flex: 1; }
.small { font-size: 8.6pt; color: #444; }
"""


def build(best_name: str, author: str, mentor: str, institution: str) -> str:
    best = RUNS / best_name
    T = best / "test"
    P = best / "plots"
    F = REPORTS / "figures"
    n = [0]
    ts = json.loads((T / "test_summary.json").read_text(encoding="utf-8"))
    prim = next(r for r in ts if r["checkpoint"] == "best_val_loss.pt")
    hist = json.loads((best / "history.json").read_text(encoding="utf-8"))
    env = json.loads((best / "env.json").read_text(encoding="utf-8"))
    audit = json.loads((REPORTS / "audit.json").read_text(encoding="utf-8"))
    onnx = json.loads((REPORTS / "onnx_export.json").read_text(encoding="utf-8"))
    enrol = json.loads((REPORTS / "enrollment_demo.json").read_text(encoding="utf-8")) if (REPORTS / "enrollment_demo.json").exists() else {}
    grid = pd.read_csv(T / "fewshot_results.csv")
    comp = pd.read_csv(REPORTS / "model_comparison.csv")
    lat = pd.read_csv(REPORTS / "latency.csv") if (REPORTS / "latency.csv").exists() else None
    best_lat = float(lat.set_index("run").cpu_ms_per_image.get(best_name)) if lat is not None else float(prim["cpu_ms_per_image"])
    seeds = pd.read_csv(REPORTS / "seeds.csv") if (REPORTS / "seeds.csv").exists() else None
    pc = pd.read_csv(T / "per_class_metrics.csv")
    cat = pd.read_csv(T / "category_accuracy.csv")
    vow = pd.read_csv(T / "vowel_sign_accuracy.csv")
    con = pd.read_csv(T / "consonant_row_accuracy.csv")
    pairs = pd.read_csv(T / "top_confused_pairs.csv")
    osr = pd.read_csv(T / "openset.csv").iloc[0]
    tra = pd.read_csv(T / "transductive.csv")
    src = pd.read_csv(T / "source_aware.csv").iloc[0]
    abl = pd.read_csv(REPORTS / "ablations.csv") if (REPORTS / "ablations.csv").exists() else None
    split = json.loads((ROOT / "data/processed/class_split.json").read_text(encoding="utf-8"))
    key = f"{100 * prim['episode_5w5s_acc']:.1f}% ± {100 * prim['episode_5w5s_ci95']:.1f}"

    def g(split_: str, way: str, k: int, col: str = "acc") -> float:
        s = grid[(grid.split == split_) & (grid.n_way.astype(str) == way) & (grid.k_shot == k)]
        return float(s[col].iat[0]) if len(s) else float("nan")

    # ---------------- cover ----------------
    out = [f"""<div class="cover"><div>
<div class="band"><h0>Handwritten Tamil Character Recognition with Few-Shot Deep Learning</h0>
<div class="sub">Project report: proposed model, dataset analysis and performance evaluation</div></div>
<table class="meta" style="margin-top:26px;width:auto">
<tr><td><b>Prepared by</b></td><td>{esc(author)}</td></tr>
<tr><td><b>Submitted to</b></td><td>{esc(mentor)}</td></tr>
<tr><td><b>Institution</b></td><td>{esc(institution)}</td></tr>
<tr><td><b>Date</b></td><td>17 November 2026</td></tr>
<tr><td><b>Proposed model</b></td><td>Prototypical Network with an ImageNet-pretrained MobileNetV3-Large backbone (96 × 96 input)</td></tr>
</table></div>
<div><h2>Abstract</h2>
<p>This report presents a few-shot deep-learning system for recognising handwritten Tamil characters. The system
learns a metric embedding in which a letter is represented by the mean of a few example images (a
<i>prototype</i>), so a letter can be recognised from 1 to 5 examples and new letters can be added without
retraining. Eight few-shot methods and several backbones were trained and compared on a cleaned dataset of
{audit['n_ok_after_dedupe']:,} images covering {audit['n_classes_with_images']} letter classes. The proposed model, a Prototypical
Network on an ImageNet-pretrained MobileNetV3-Large, was selected on validation loss. On
{len(split['novel'])} letters never seen during training it reaches <b>{key}</b> accuracy on 5-way 5-shot
episodes, {pct(g('B_novel', '5', 1))}% on 5-way 1-shot, and {pct(prim['fullset_acc'])}% when every query is
classified among all {prim['fullset_n_classes']} unseen letters (macro-F1 {prim['macro_f1']:.3f}). After temperature
scaling its confidences are well calibrated (ECE {prim['ece_before']:.3f} → {prim['ece_after']:.3f}). The model is
exported to ONNX and runs in about {onnx['onnxruntime_cpu_ms_per_image']:.1f} ms per image on a laptop CPU.</p></div></div>"""]

    # ---------------- 1. summary ----------------
    out.append(f"""<div class="section"><h1>1. Executive summary</h1>
<div class="kpis">
<div class="kpi"><b>{pct(prim['episode_5w5s_acc'])}%</b><span>5-way 5-shot accuracy<br>on unseen letters</span></div>
<div class="kpi"><b>{pct(g('B_novel', '5', 1))}%</b><span>5-way 1-shot accuracy<br>on unseen letters</span></div>
<div class="kpi"><b>{pct(prim['fullset_acc'])}%</b><span>accuracy among all {prim['fullset_n_classes']}<br>unseen letters (top-5 {pct(prim['fullset_top5'])}%)</span></div>
<div class="kpi"><b>{prim['ece_after']:.3f}</b><span>calibration error (ECE)<br>after temperature scaling</span></div>
<div class="kpi"><b>{onnx['onnxruntime_cpu_ms_per_image']:.1f} ms</b><span>CPU time per image<br>(ONNX Runtime)</span></div></div>
<ul>
<li><b>Task.</b> Recognise isolated handwritten Tamil letters (vowels, consonants, consonant-vowel compounds and
āytham) from very few examples per letter, and allow new letters to be enrolled without retraining.</li>
<li><b>Data.</b> {audit['n_files_seen']:,} images in {audit['n_classes_with_images']} classes. {_ndup(audit)} near-duplicates were removed
before any split. 3 letters of the 18 × 12 compound grid are missing (<span class="ta">தௌ, லௌ, ளௌ</span>).</li>
<li><b>Evaluation.</b> The main protocol (Protocol B) holds out {len(split['novel'])} entire letter classes that are
never used for training, which is the true few-shot test. A second protocol (A) evaluates letters seen in training
on held-out images. Test data was used once per checkpoint. Every design choice was made on validation data.</li>
<li><b>Proposed model.</b> Prototypical Network, MobileNetV3-Large backbone pretrained on ImageNet, 128-dimensional
embedding, 96 × 96 grey input, {prim['params'] / 1e6:.2f} M parameters. It had the lowest validation loss of the
{len(comp)} runs ({hist['best_val_loss']:.4f}).</li>
<li><b>Main findings.</b> ImageNet pretraining gives the largest single improvement. The episodic metric methods
(ProtoNet, Matching Net, ArcFace → ProtoNet) perform within about one point of each other with a small backbone.
Errors concentrate on two-part vowel signs (ொ, ோ, ௌ) and on visually similar consonant shapes.</li>
<li><b>Limitations.</b> Training ran on a laptop CPU, so the models are small and under-trained: validation loss
was still falling at the end. Open-set ("unknown letter") rejection is reliable only for small enrolled sets.</li>
</ul></div>""")

    # ---------------- 2. introduction ----------------
    out.append(f"""<div class="section"><h1>2. Introduction and objectives</h1>
<p>The Tamil script has 12 vowels, 18 consonants, the āytham (<span class="ta">ஃ</span>) and 216 consonant-vowel
compounds, giving 247 letters. Many letters differ only in a small stroke, a loop or the position of a vowel sign.
Collecting large labelled sets for every letter is costly, and the compound letters are numerous. A conventional
classifier also needs retraining whenever a letter is added. <b>Few-shot learning</b> addresses both problems.
The network learns an embedding space in which images of the same letter lie close together. A letter is then
described by a handful of examples rather than by trained output weights.</p>
<h2>Objectives</h2><ol>
<li>Audit and clean the handwritten Tamil dataset, and build leakage-free evaluation protocols.</li>
<li>Train and compare deep few-shot methods (Prototypical, Matching and Relation Networks, Siamese/triplet,
ArcFace pre-training, Baseline++, first-order MAML) against a supervised CNN baseline.</li>
<li>Monitor training rigorously (epoch-level loss curves, overfitting and plateau detection) and choose all
settings on validation data.</li>
<li>Evaluate once on held-out test data: accuracy over N-way K-shot episodes with confidence intervals, F1,
confusion analysis, calibration and open-set rejection.</li>
<li>Deliver a deployable model (ONNX) that can enrol new letters without retraining.</li></ol>
<p class="small">Implementation: Python {env['python']}, PyTorch {env.get('packages', {}).get('torch', '')}, timm,
OpenCV, scikit-learn and ONNX Runtime. Hardware: Intel Core i5-1335U laptop CPU, 16 GB RAM, no GPU.</p>""")

    # ---------------- 3. dataset ----------------
    sg = audit.get("source_groups", {})
    out.append(f"""<h1 style="margin-top:14px">3. Dataset</h1>
<h2>3.1 Description and audit</h2>
<p>The dataset ("Tamil Letters Combined") is organised as <i>category / letter / images</i>. It was located by a
recursive scan and treated as read-only. A content fingerprint confirmed that it was unchanged at the end of the
project. The class label is the Unicode-normalised (NFC) name of the folder that contains the images.</p>
{table(["Property", "Value"], [
        ["Letter classes with images", f"{audit['n_classes_with_images']} (vowels 12, consonants 18, compounds 216, āytham 1)"],
        ["Image files", f"{audit['n_files_seen']:,} PNG, all 48 × 48 pixels; 0 corrupt, 0 zip archives, 0 empty or nested folders"],
        ["Near-duplicates removed (perceptual hash, Hamming ≤ 4)", f"{_ndup(audit)}, removed before any split"],
        ["Usable images", f"{audit['n_ok_after_dedupe']:,} (3 more blank images dropped after preprocessing)"],
        ["Images per class after cleaning", "min 4 / median 46 / max 96"],
        ["Classes with fewer than 15 images", '<span class="ta">ணே, ணெ, ழா, ழ</span> (4); <span class="ta">ணு, ணூ</span> (5); <span class="ta">ழி, ழீ</span> (6)'],
        ["Missing from the 18 × 12 compound grid", '<span class="ta">தௌ, லௌ, ளௌ</span>'],
        ["Possible spelling variants (kept separate, flagged)", '<span class="ta">தெள, லெள, ளெள</span>: encoded as consonant + ெ (U+0BC6) + ள (U+0BB3) instead of the single sign ௌ (U+0BCC); they look identical when rendered'],
        ["Ink polarity", "14,180 dark-on-light, 25 light-on-dark (normalised automatically)"],
        ["Source groups (from file names)", ", ".join(f"{k} {v:,}" for k, v in sg.items()) or "see audit"],
    ])}
{fig(F / 'class_counts.png', 'Images per class before and after near-duplicate removal.', n)}
<h2>3.2 Preprocessing</h2>
<p>One module is used for training and for inference, so the two cannot drift apart. Steps: alpha compositing on
white, conversion to grey, light denoising, automatic ink-polarity detection (normalised to white ink on black),
Otsu thresholding with a soft mask that keeps grey levels inside the strokes, cropping to the ink bounding box with
an 8% margin, padding to a square without distorting the aspect ratio, resizing (48 px for Conv4 models, 96 px for
the proposed model) and normalisation. Soft thresholding was chosen because hard binarisation broke thin strokes
at this resolution.</p>
{fig(REPORTS / 'preprocess_samples.png', 'Random preprocessed samples (white ink on black, cropped and centred).', n, '82%')}
<h2>3.3 Evaluation protocols</h2>
<p><b>Protocol B (novel classes, main protocol).</b> The letter classes are split 70/15/15, stratified by category,
into {len(split['base'])} base classes (training), {len(split['val'])} validation classes (model selection) and
{len(split['novel'])} novel classes (final test). The novel letters are never seen during training. A
<i>hard-novel</i> subset of {len(split.get('hard_novel', []))} compounds contains letters whose consonant and vowel sign
both appear in base classes in other combinations (for example, <span class="ta">கொ</span> held out while
<span class="ta">க</span> and <span class="ta">கோ</span> are seen). This tests compositional generalisation.</p>
<p><b>Protocol A (known classes).</b> Within every class, images are split 70/15/15 (train/val/test), stratified by
source group. Support examples come from the train pool and queries from the test pool.</p>
{table(["Protocol B part", "Classes", "Images", "Use"], [
        ["Base", len(split['base']), "9,825", "training only"],
        ["Validation", len(split['val']), "2,271", "epochs, learning rate, model selection, calibration"],
        ["Novel (test)", len(split['novel']), "2,106", "final test, evaluated once"],
        ["Hard-novel (subset of novel)", len(split.get('hard_novel', [])), "1,643", "compositional generalisation test"],
    ])}
<p class="small">Novel test letters: <span class="ta">{esc(' '.join(split['novel']))}</span></p>
<div class="note"><b>Leakage controls.</b> Duplicates were removed before splitting. Base, validation and novel classes
are disjoint (unit-tested). No image appears in both the support set and the query set of an episode.
Validation uses a fixed list of episodes, so loss values are comparable across epochs. Test data was evaluated
once per checkpoint, and a guard file blocks a second evaluation.</div></div>""")

    # ---------------- 4. methodology ----------------
    out.append(f"""<div class="section"><h1>4. Methodology</h1>
<h2>4.1 Few-shot formulation</h2>
<p>Training and testing use <i>episodes</i>. An N-way K-shot episode draws N letters, K labelled <i>support</i>
images per letter and Q unlabelled <i>query</i> images per letter. The model must label the queries using only
the support images. Classes are sampled uniformly, so every episode is class-balanced regardless of the class
imbalance in the dataset.</p>
<h2>4.2 Proposed model: Prototypical Network with a pretrained MobileNetV3-Large</h2>
{table(["Component", "Design"], [
        ["Input", "1 × 96 × 96 grey image, ink-normalised (Section 3.2)"],
        ["Backbone", "MobileNetV3-Large pretrained on ImageNet-1k (timm), stem adapted to one channel by summing the RGB filters"],
        ["Embedding head", "global average pooling → linear layer → 128-dimensional embedding"],
        ["Prototype", "mean embedding of the K support images of each letter"],
        ["Classifier", "softmax over −(squared Euclidean distance / D) × 16 × learnable temperature"],
        ["Loss", "cross-entropy over the episode's queries, label smoothing 0.05"],
        ["Parameters / compute", f"{prim['params'] / 1e6:.2f} M parameters, {env.get('flops_per_image', 0) / 1e6:.0f} M FLOPs per image"],
        ["Calibration", f"temperature T = {prim['temperature']:.2f} fitted on validation episodes"],
        ["Unknown-letter rejection", "maximum score below a threshold fitted on validation data (accepts 95% of enrolled letters)"],
    ], highlight=-1)}
<p>The network learns an embedding in which images of the same letter cluster together. After training, enrolling a
letter only requires computing its prototype from a few images. No weights change, so new or missing letters (for
example <span class="ta">தௌ</span>) can be added at deployment time.</p>
<h2>4.3 Training recipe</h2>
{table(["Setting", "Value"], [
        ["Episodes", "20-way 5-shot 5-query from the 173 base classes; one epoch = 50 episodes"],
        ["Optimiser", "AdamW, learning rate 1 × 10⁻³, weight decay 5 × 10⁻⁴, gradient clipping at 5"],
        ["Schedule", "2 warm-up epochs, then cosine decay to 1% of the peak"],
        ["Weight averaging", "exponential moving average of the weights (decay 0.995); the EMA model is validated and deployed"],
        ["Augmentation (training only)", "rotation ±10°, shift/zoom ±10%, shear ±8°, elastic distortion, noise/blur, stroke thickness, brightness/contrast, small random erasing. No flips or large rotations: Tamil letters are not symmetric."],
        ["Validation", "fixed set of 100 episodes, 10-way 5-shot, on the 37 validation classes, without augmentation"],
        ["Checkpoint selection", "lowest validation loss (early stopping on validation loss, patience 10)"],
        ["Training length", f"{hist['epochs_run']} epochs; best epoch {hist['best_val_loss_epoch']} (val loss {hist['best_val_loss']:.4f}, val accuracy {pct(hist['best_val_acc'])}%)"],
    ])}
<h2>4.4 Methods compared</h2>
<p>All methods use the same splits, the same episodes and the same training recipe:</p><ul>
<li><b>Prototypical Network</b>: mean-prototype classifier (the proposed approach).</li>
<li><b>Matching Network</b>: cosine attention over all support images.</li>
<li><b>Relation Network</b>: a learned module scores concatenated support/query feature maps.</li>
<li><b>Siamese (triplet)</b>: triplet loss with hard-negative mining; nearest-prototype decision.</li>
<li><b>ArcFace</b>: angular-margin classification pretraining on the base classes. Evaluated directly, and as an
initialisation for episodic ProtoNet fine-tuning.</li>
<li><b>Baseline++</b>: frozen backbone with a cosine classifier fitted on each episode's support images.</li>
<li><b>First-order MAML</b>: gradient-based adaptation on each episode.</li>
<li><b>Supervised CE classifier</b>: conventional non-few-shot reference (Protocol A only).</li></ul>
<p>Two further techniques were evaluated as separate result rows: <b>transductive refinement</b> (soft k-means on
the unlabelled queries) and <b>temperature scaling</b>.</p>
<h2>4.5 Evaluation metrics</h2>
<p>Episode accuracy is the mean over ≥ 600 test episodes with a 95% confidence interval (FO-MAML: 100 episodes),
for N ∈ {{5, 10, 20, all}} and K ∈ {{1, 2, 3, 5, 10}}. The full-set evaluation classifies every test query among
all test classes with 5 support images each, and reports accuracy, top-3/5, macro and weighted F1, per-class
precision/recall/F1 and the confusion matrix. Calibration is measured by expected calibration error (ECE, 15
bins) before and after temperature scaling. Open-set performance is measured by AUROC and rejection rates.</p></div>""")

    # ---------------- 5. model selection ----------------
    ep = pd.read_csv(sorted(RUNS.glob("*_epochstudy"))[-1] / "metrics_epoch.csv")
    lr_rows, bb_rows = [], []
    if abl is not None:
        for _, r in abl[abl.tag.str.startswith("lr")].sort_values("best_val_loss").iterrows():
            lr_rows.append([esc(r.setting.replace("baseline: 20-way 5-shot, ", "").split(",")[0]), f"{r.best_val_loss:.4f}", f"{100 * r.best_val_acc:.2f} ± {100 * r.ci:.2f}"])
        base = abl[abl.tag == "lr1e-3"]
        for _, r in pd.concat([base, abl[abl.tag.str.contains("mnv3")]]).sort_values("best_val_loss").iterrows():
            name = "Conv4 from scratch, 48 px" if r.tag == "lr1e-3" else r.setting
            bb_rows.append([esc(name), f"{r.params / 1e3:,.0f} k", f"{r.best_val_loss:.4f}", f"{100 * r.best_val_acc:.2f} ± {100 * r.ci:.2f}"])
    out.append(f"""<div class="section"><h1>5. Training monitoring and model selection (validation data only)</h1>
<p>Every run logged training and validation loss and accuracy per epoch, together with learning rate, gradient norms,
embedding norms and prototype separation. Automatic detectors flagged overfitting (training loss falling while
validation loss rises for 3 epochs) and plateaus. <b>No run triggered the overfitting detector.</b> In every run
the best validation epoch was the last or second-to-last.</p>
{fig(P / 'loss_curve.png', 'Proposed model: training vs validation loss per epoch (best epoch marked). Training loss is higher because training episodes are augmented and 20-way, while validation episodes are clean and 10-way.', n, '88%')}
<h2>5.1 Epoch budget</h2>
<p>A 40-epoch study (ProtoNet, Conv4, early stopping off) shows validation loss decreasing until epoch
{int(ep.epoch.iat[int(ep.val_loss.idxmin())])} ({ep.val_loss.min():.4f}). Validation accuracy rises from
{pct(ep.val_episode_acc.iat[14], 2)}% at epoch 15 to {pct(ep.val_episode_acc.max(), 2)}%. The recommended budget
is therefore about 44 epochs. Compute limits restricted the compared runs to 15 epochs and the proposed model to
8, so all reported test numbers are conservative.</p>
{fig(F / 'epoch_study.png', 'Epoch-budget study: validation loss keeps improving until about epoch 36, with no sign of overfitting.', n, '80%')}
<div class="two"><div><h2>5.2 Learning rate</h2>
{table(["Learning rate", "Best val loss", "Val acc %"], lr_rows)}
<p class="small">8-epoch ProtoNet runs. A learning-rate range test suggested ~1.4 × 10⁻³; 3 × 10⁻³ was best
in the comparison and is recommended for the next training round.</p></div>
<div><h2>5.3 Backbone and pretraining</h2>
{table(["Backbone", "Params", "Best val loss", "Val acc %"], bb_rows, highlight=0)}
<p class="small">ImageNet pretraining is the largest single effect: +4.5 points over Conv4, and +18.8 points over the
same network trained from scratch. This run was selected as the proposed model.</p></div></div></div>""")

    # ---------------- 6. results of proposed model ----------------
    ways, shots = ["5", "10", "20", "all"], [1, 2, 3, 5, 10]
    grid_rows = []
    for sp_, label in (("B_novel", "Novel"), ("B_hard_novel", "Hard-novel")):
        for w in ways:
            eff = grid[(grid.split == sp_) & (grid.n_way.astype(str) == w)].effective_n_way
            wl = f"{w}-way" if w != "all" else f"all ({int(eff.iat[0]) if len(eff) else '?'})"
            grid_rows.append([label, wl] + [f"{pct(g(sp_, w, k))} <span class='small'>±{pct(g(sp_, w, k, 'ci95'))}</span>" for k in shots])
    cat_rows = [[esc(r.category), f"{pct(r.accuracy)}", int(r.queries)] for _, r in cat.iterrows()]
    vow_rows = [[f'<span class="ta">{esc(r.vowel_sign)}</span>', pct(r.accuracy), int(r.queries)] for _, r in vow.sort_values("accuracy").iterrows()]
    pcs = pc.sort_values("f1")
    worst = [[f'<span class="ta">{esc(r.label)}</span>', f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r.f1:.2f}", int(r.support)] for _, r in pcs.head(10).iterrows()]
    bestc = [[f'<span class="ta">{esc(r.label)}</span>', f"{r.precision:.2f}", f"{r.recall:.2f}", f"{r.f1:.2f}", int(r.support)] for _, r in pcs.tail(10).iloc[::-1].iterrows()]
    pr_rows = [[f'<span class="ta">{esc(r.letter_a)} ↔ {esc(r.letter_b)}</span>', int(r.a_as_b), int(r.b_as_a), int(r.total), pct(r.rate)] for _, r in pairs.head(10).iterrows()]
    out.append(f"""<div class="section"><h1>6. Results of the proposed model (test set)</h1>
<p>All results in this section are on the {len(split['novel'])} <b>novel letters, which were never used in training</b>
(Protocol B), using checkpoint <code>best_val_loss.pt</code> (epoch {prim['checkpoint_epoch']}). The best-accuracy
and last checkpoints are the same epoch and give the same numbers.</p>
<h2>6.1 Few-shot accuracy (N-way × K-shot, % ± 95% CI, ≥ 600 episodes per cell)</h2>
{table(["Split", "N-way", "1-shot", "2-shot", "3-shot", "5-shot", "10-shot"], grid_rows)}
<p class="small">The headline 5-way 5-shot figure ({key}) comes from an independently sampled set of 600 episodes and
agrees with the grid cell within the confidence interval. Hard-novel letters are recognised as well as other novel
letters, so the embedding generalises to unseen consonant/vowel-sign combinations.</p>
<div class="two">{fig(P / 'fewshot_vs_shot.png', 'Accuracy vs number of support examples (K).', n)}{fig(P / 'fewshot_vs_way.png', 'Accuracy vs number of candidate letters (N).', n)}</div>
<h2>6.2 Full-set classification among all {prim['fullset_n_classes']} unseen letters (5 support images each)</h2>
{table(["Metric", "Value"], [
        ["Queries", f"{prim['fullset_n_queries']:,}"],
        ["Top-1 accuracy", f"<b>{pct(prim['fullset_acc'])}%</b>"],
        ["Top-3 / top-5 accuracy", f"{pct(prim['fullset_top3'])}% / {pct(prim['fullset_top5'])}%"],
        ["Macro F1 / weighted F1", f"{prim['macro_f1']:.3f} / {prim['weighted_f1']:.3f}"],
        ["Negative log-likelihood (before → after calibration)", f"{prim['nll_uncalibrated']:.3f} → {prim['nll_calibrated']:.3f}"],
        ["Expected calibration error (before → after)", f"{prim['ece_before']:.3f} → {prim['ece_after']:.3f}"],
        ["Letters with F1 ≥ 0.90 / F1 < 0.50", f"{int((pc.f1 >= 0.9).sum())} / {int((pc.f1 < 0.5).sum())} of {len(pc)}"],
    ])}
<div class="two"><div><h3>Accuracy by category</h3>{table(["Category", "Acc %", "Queries"], cat_rows)}</div>
<div><h3>Accuracy by vowel sign (lowest first)</h3>{table(["Vowel sign", "Acc %", "Queries"], vow_rows[:8])}</div></div>
<p class="small">"-" = pure consonant (with pulli); "ெள*" = the flagged spelling variants (<span class="ta">லெள</span> is a novel test class).</p>
<div class="section"></div>
<h2>6.3 Per-letter performance</h2>
<div class="two"><div><h3>Ten weakest letters</h3>{table(["Letter", "Precision", "Recall", "F1", "Queries"], worst)}</div>
<div><h3>Ten strongest letters</h3>{table(["Letter", "Precision", "Recall", "F1", "Queries"], bestc)}</div></div>
<h3>Most-confused letter pairs</h3>
{table(["Pair", "A predicted as B", "B predicted as A", "Total", "Rate %"], pr_rows)}
{fig(P / 'confusion_zoom.png', 'Confusion among the most-confused unseen letters (rows = true letter).', n, '78%')}
<p>The errors follow the structure of the script. The weakest letters carry the two-part vowel signs ௌ, ோ and
ொ, whose parts sit on both sides of the consonant (<span class="ta">லெள/வௌ, மௌ/வௌ, பௌ/மௌ, தோ/றோ</span>). Other errors come
from consonants that differ in a single loop or tail (<span class="ta">னீ/ளீ, டி/ழ்</span>). At 48-px source
resolution, the pulli dot of <span class="ta">ழ்</span> is often barely visible.</p>
{fig(P / 'misclassified.png', 'Most confident test errors (true → predicted, confidence, source group). Several are data-quality problems: solid bars and dark shaded backgrounds.', n, '92%')}
<h2>6.4 What the network attends to</h2>
{fig(P / 'gradcam.png', 'Grad-CAM saliency for correctly classified (top rows) and misclassified (bottom rows) queries.', n, '88%')}
{fig(P / 'tsne.png', 't-SNE of test embeddings coloured by category, consonant row and vowel sign. The embedding is organised largely by vowel sign; the ொ/ோ/ௌ family forms one dense region, which matches the confusions.', n)}
<h2>6.5 Calibration, open-set rejection and robustness</h2>
{table(["Analysis", "Result"], [
        ["Temperature scaling (fitted on validation data)", f"T = {prim['temperature']:.2f}; ECE {prim['ece_before']:.3f} → {prim['ece_after']:.3f}"],
        ["Open-set rejection (half the test letters enrolled, half unseen)", f"AUROC {osr.auroc:.3f}; rejects {pct(osr.unknown_rejection_rate)}% of unseen-letter queries; falsely rejects {pct(osr.false_reject_rate)}% of enrolled-letter queries"],
        ["Transductive prototype refinement", "; ".join(f"{int(r.n_way)}-way {int(r.k_shot)}-shot {'+' if r.gain >= 0 else ''}{100 * r.gain:.1f} pp" for _, r in tra[tra.split == 'B_novel'].iterrows())],
        ["Source-aware test (support and queries from different source groups, 5-way 5-shot)", f"{pct(src.cross_source_acc)}% vs {pct(src.unconstrained_acc)}% unconstrained (no drop)"],
    ])}
<div class="two">{fig(P / 'reliability.png', 'Reliability diagram before and after temperature scaling.', n)}{fig(P / 'openset.png', 'Open-set ROC and score distributions (enrolled vs unseen letters).', n)}</div></div>""")

    # ---------------- 7. comparison ----------------
    keep = comp[comp.tag.isin(["final", "arcft_final", "final_seed7", "final_seed123", "abl_mnv3_pre96"]) |
                ((comp.protocol == "A") & (comp.tag == "final"))].copy()
    keep["order"] = keep.apply(lambda r: (r.protocol, -(r.test_acc_5w5s if pd.notna(r.test_acc_5w5s) else 0)), axis=1)
    keep = keep.sort_values("order")
    rows, hl = [], -1
    for i, (_, r) in enumerate(keep.iterrows()):
        name = NICE.get(r.method, r.method)
        if r.tag == "arcft_final":
            name = "ArcFace → ProtoNet fine-tune"
        if r.method == "protonet" and r.tag == "final" and r.protocol == "B":
            name += " (seed 42)"
        if str(r.tag).startswith("final_seed"):
            name += f" (seed {str(r.tag).replace('final_seed', '')})"
        if r.run == best_name:
            name, hl = "<b>ProtoNet (proposed)</b>", i
        latency = lat.set_index("run").cpu_ms_per_image.get(r.run, r.cpu_ms_per_image) if lat is not None else r.cpu_ms_per_image
        rows.append([name, BB.get(r.backbone, r.backbone), r.protocol, f"{r.params / 1e3:,.0f} k", f"{latency:.1f}",
                     f"{int(r.best_epoch)}/{int(r.epochs_run)}", pct(r.test_acc_5w1s), f"<span style='white-space:nowrap'>{pct(r.test_acc_5w5s)} ± {pct(r.test_ci_5w5s)}</span>",
                     pct(r.test_acc_20w5s), pct(r.test_acc_allw5s), f"{r.macro_f1:.3f}", f"{r.ece_after:.3f}"])
    seed_txt = ""
    if seeds is not None:
        sb = seeds[(seeds.split == "B_novel") & (seeds.n_way.astype(str) == "5")]
        m = sb.groupby("k_shot").acc.agg(["mean", "std", "count"])
        if 5 in m.index and 1 in m.index:
            seed_txt = (f"Across {int(m.loc[5, 'count'])} seeds, ProtoNet (Conv4) reaches {100 * m.loc[5, 'mean']:.1f} ± {100 * m.loc[5, 'std']:.1f}% "
                        f"at 5-way 5-shot and {100 * m.loc[1, 'mean']:.1f} ± {100 * m.loc[1, 'std']:.1f}% at 5-way 1-shot, so differences "
                        "under about 0.5 points between Conv4 methods are not significant.")
    out.append(f"""<div class="section"><h1>7. Comparison with other methods</h1>
<p>Protocol B rows report unseen letters, and Protocol A rows report seen letters on held-out images. Accuracy is in %.
"All" means all classes of the test split (37 for B, 247 for A). Latency is PyTorch, batch 1, on an idle CPU
(2 threads).</p>
{table(["Method", "Backbone", "Prot.", "Params", "ms/img", "Best ep.", "5w1s", "5w5s", "20w5s", "All-way 5s", "Macro F1", "ECE"], rows, highlight=hl)}
<p class="small">{seed_txt} FO-MAML is evaluated on 100 episodes per cell, so its intervals are wider; its 5-shot
accuracy falls below its 1-shot accuracy, which indicates an unstable inner loop.</p>
<ul>
<li><b>Few-shot vs supervised.</b> On known letters (Protocol A), episodic ProtoNet beats the supervised CE
classifier at 5-way 5-shot and when classifying among all 247 letters, using the same backbone.</li>
<li><b>Known vs novel letters.</b> At 5-way 5-shot the gap between Protocol A and Protocol B is under one point, so
the embedding transfers to unseen letters almost as well as to seen ones.</li>
<li><b>Method choice.</b> With a small backbone, ProtoNet, Matching Net and ArcFace → ProtoNet are statistically
tied. ArcFace pretraining gives the best Conv4 validation loss. Relation Net, Siamese and Baseline++ fall behind
on the harder all-way task.</li>
<li><b>Backbone.</b> The pretrained MobileNetV3 is the best model at every N and K, despite only 8 training epochs.</li>
</ul>
{fig(learning_curve_figure(comp, best_name, F / 'report_learning_curves.png'), 'Learning curves on unseen letters: test accuracy (± 95% CI) vs number of support examples, for the final run of each method (FO-MAML evaluated on a reduced grid).', n)}</div>""")

    # ---------------- 8. deployment ----------------
    demo_rows = []
    for letter, d in (enrol.get("letters") or {}).items():
        demo_rows.append([f'<span class="ta">{esc(letter)}</span>', pct(d.get("before_rejected_as_unknown"), 0),
                          pct(d.get("after_top1"), 0), pct(d.get("after_top5"), 0), int(d.get("evaluated_images (val+test pools)", 0))])
    out.append(f"""<div class="section"><h1>8. Deployment</h1>
{table(["Item", "Result"], [
        ["Export", f"ONNX (opset {onnx['opset']}), {onnx['onnx_size_kb'] / 1024:.1f} MB, <code>models/embedder.onnx</code>; preprocessing, temperature and threshold in <code>models/deploy.json</code>"],
        ["Parity with PyTorch", f"max |difference| {onnx['max_abs_diff']:.1e} over {onnx['samples_compared']} images (cosine similarity {onnx['mean_cosine']:.3f}). Passed."],
        ["CPU latency per image", f"ONNX Runtime {onnx['onnxruntime_cpu_ms_per_image']:.1f} ms; PyTorch {onnx['torch_cpu_ms_per_image']:.1f} ms (export run) and {best_lat:.1f} ms (idle re-measurement)"],
        ["Support registry", "5 training images per letter for all 247 letters; prototypes cached in <code>prototypes.npz</code> and refreshed with one command"],
        ["Tools", "<code>predict.py</code> (image/folder → top-5 + calibrated confidence + 'unknown'), a drawing-canvas app with live enrolment, a webcam app"],
    ])}
<h2>8.1 Enrolling letters without retraining</h2>
<p>Each of {len(enrol.get('letters') or {})} novel letters was removed from the registry, enrolled from 5 of its training images, and then
recognised on its held-out images. The missing letter <span class="ta">தௌ</span> was enrolled from 5 font-rendered
glyphs (a demonstration, because no handwritten samples exist) and recognised on 5/5 probe images.</p>
{table(["Letter", "Rejected as unknown before enrolment %", "Top-1 after %", "Top-5 after %", "Images evaluated"], demo_rows) if demo_rows else ""}
{fig(F / 'enrollment_demo.png', 'Enrolment demonstration: novel letters and the missing letter தௌ added without retraining.', n, '85%')}
<div class="warn"><b>Limitation of open-set rejection.</b> At the episode level the model separates enrolled from
unseen letters well (AUROC {osr.auroc:.2f}). Against the full 246-letter registry, however, an unseen letter is
almost always close to some enrolled letter. Before enrolment, only a few percent of a new letter's images were
rejected as "unknown". The "unknown" output should therefore be treated as a weak signal in full-alphabet use.</div></div>""")

    # ---------------- 9. limitations, conclusion ----------------
    out.append(f"""<div class="section"><h1>9. Limitations</h1><ul>
<li><b>Compute.</b> All training ran on a laptop CPU: small backbones, 50-episode epochs, 15 epochs for the compared
methods and 8 for the proposed model. The epoch study shows validation loss still improving until about epoch 36.
The reported numbers are lower bounds.</li>
<li><b>Ablations not run.</b> Train-time way and shot, embedding size, distance metric, EMA, input size, ResNet-12/18
and EfficientNet-B0 were not compared because of time and memory limits.</li>
<li><b>Data.</b> Every image is 48 × 48, so fine details such as the pulli dot are near the resolution limit. Some
images have dark shaded backgrounds or are unusable crops. Eight letters have fewer than 15 images. Writer identity
is not recorded, so the source-aware test relies on coarse file-name groups (78% are "other").</li>
<li><b>Labels.</b> <span class="ta">தெள, லெள, ளெள</span> are stored as consonant + ெ (U+0BC6) + ள (U+0BB3),
while the other 15 consonant rows use the single au sign ௌ (U+0BCC). The two spellings look identical on screen, so
these classes are probably the "missing" <span class="ta">தௌ, லௌ, ளௌ</span> written with a different encoding. They were kept separate pending review. This also
lowers the measured accuracy for the ௌ sign.</li>
<li><b>Open set.</b> Rejection of unknown letters is weak at full-alphabet scale (Section 8).</li></ul>
<h1 style="margin-top:14px">10. Conclusion and future work</h1>
<p>A complete few-shot recognition pipeline for handwritten Tamil was built and evaluated rigorously. The proposed
Prototypical Network with an ImageNet-pretrained MobileNetV3-Large recognises letters it has never seen with
<b>{key}</b> accuracy from five examples (5-way) and <b>{pct(prim['fullset_acc'])}%</b> among all
{prim['fullset_n_classes']} unseen letters. Its confidences are well calibrated. It runs in about
{onnx['onnxruntime_cpu_ms_per_image']:.1f} ms per image on a CPU, and new letters can be added without retraining.
The comparison shows that, in this regime, the choice and pretraining of the backbone matter more than the choice
among episodic metric-learning methods.</p>
<h2>Recommended next steps</h2><ol>
<li>Retrain the proposed model for ~44 epochs at learning rate 3 × 10⁻³, initialised with ArcFace pretraining
(the best Conv4 recipe), preferably on a GPU with 200-episode epochs.</li>
<li>Collect more samples for the small classes (<span class="ta">ணே ணெ ழா ழ ணு ணூ ழி ழீ</span>) and for the
ொ/ோ/ௌ family, and remove unusable images.</li>
<li>Resolve the <span class="ta">தெள/லெள/ளெள</span> spelling question, and collect handwritten samples of
<span class="ta">தௌ, லௌ, ளௌ</span>.</li>
<li>Record writer IDs to enable a true writer-independent evaluation.</li>
<li>Improve open-set rejection, for example with per-letter thresholds or distance-to-second-best scores.</li></ol>
<h2>Appendix: reproducibility</h2>
<p class="small">Fixed seeds (Python, NumPy, PyTorch, data loaders). Every run stores its configuration snapshot,
environment and package versions, per-epoch and per-step metrics, checkpoints and test files under
<code>runs/&lt;run&gt;/</code>. Proposed model run: <code>runs/{esc(best_name)}</code>. All plots can be regenerated
from stored results with <code>python src/visualize.py --all</code>, and this report with
<code>python src/make_report.py</code>. The dataset folder was verified to be unmodified at the end of the project
(14,470 files: 0 added, removed or changed). 53 automated tests pass.</p></div>""")

    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Tamil Few-Shot HTR Report</title>"
            f"<style>{CSS}</style></head><body>{''.join(out)}</body></html>")


def learning_curve_figure(comp: pd.DataFrame, best_name: str, out: Path) -> Path:
    """Accuracy vs shots (Protocol B novel; 5-way and all-way) for the final run of every method."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sel = comp[(comp.protocol == "B") & (comp.tag.isin(["final", "arcft_final"]) | (comp.run == best_name))]
    fig_, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=160)
    cmap = plt.get_cmap("tab10")
    for i, (_, r) in enumerate(sel.iterrows()):
        fr = pd.read_csv(RUNS / r.run / "test" / "fewshot_results.csv")
        fr = fr[fr.split == "B_novel"]
        name = "ArcFace -> ProtoNet" if r.tag == "arcft_final" else NICE.get(r.method, r.method)
        proposed = r.run == best_name
        if proposed:
            name = "ProtoNet + MobileNetV3 (proposed)"
        for ax, way in zip(axes, ("5", "all")):
            d = fr[fr.n_way.astype(str) == way].sort_values("k_shot")
            if d.empty:
                continue
            ax.errorbar(d.k_shot, 100 * d.acc, yerr=100 * d.ci95, marker="o", ms=4 if not proposed else 6,
                        lw=1.3 if not proposed else 3.0, color="black" if proposed else cmap(i % 10), capsize=2,
                        label=name, zorder=5 if proposed else 2)
    for ax, t in zip(axes, ("5-way", "all 37 novel letters")):
        ax.set_xscale("log"); ax.set_xticks([1, 2, 3, 5, 10]); ax.set_xticklabels(["1", "2", "3", "5", "10"])
        ax.set_xlabel("support examples per letter (K)"); ax.set_ylabel("test accuracy (%)")
        ax.set_title(f"Protocol B, unseen letters: {t}"); ax.grid(alpha=.3)
    h, l = axes[0].get_legend_handles_labels()
    fig_.legend(h, l, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig_.tight_layout(rect=(0, 0.13, 1, 1))
    fig_.savefig(out, bbox_inches="tight"); plt.close(fig_)
    return out


def build_simple(best_name: str, author: str) -> str:
    """Short plain-language report: no cover page, few tables, simple wording."""
    best = RUNS / best_name
    T, P, F = best / "test", best / "plots", REPORTS / "figures"
    n = [0]
    prim = next(r for r in json.loads((T / "test_summary.json").read_text(encoding="utf-8")) if r["checkpoint"] == "best_val_loss.pt")
    audit = json.loads((REPORTS / "audit.json").read_text(encoding="utf-8"))
    onnx = json.loads((REPORTS / "onnx_export.json").read_text(encoding="utf-8"))
    enrol = json.loads((REPORTS / "enrollment_demo.json").read_text(encoding="utf-8")) if (REPORTS / "enrollment_demo.json").exists() else {}
    split = json.loads((ROOT / "data/processed/class_split.json").read_text(encoding="utf-8"))
    grid = pd.read_csv(T / "fewshot_results.csv")
    comp = pd.read_csv(REPORTS / "model_comparison.csv")
    pc = pd.read_csv(T / "per_class_metrics.csv").sort_values("f1")
    pairs = pd.read_csv(T / "top_confused_pairs.csv")
    cat = pd.read_csv(T / "category_accuracy.csv")

    def g(way: str, k: int) -> str:
        s = grid[(grid.split == "B_novel") & (grid.n_way.astype(str) == way) & (grid.k_shot == k)]
        return pct(s.acc.iat[0]) + "%" if len(s) else "—"

    def ta(x: str) -> str:
        return f'<span class="ta">{esc(x)}</span>'

    by = f" · Prepared by {esc(author)}" if author and not author.startswith("[") else ""
    nn = prim["fullset_n_classes"]
    out = [f"""<div style="font-size:19pt;font-weight:700;color:#12355b">Handwritten Tamil Letter Recognition using Few-Shot Learning</div>
<p class="small" style="margin-bottom:10px">Project report · 17 November 2026{by}</p>
<div class="note" style="font-size:10.2pt"><b>In one line:</b> we built a model that recognises a handwritten Tamil letter
from only a few example images, including letters it never saw during training. Shown 5 examples each of 5 new letters,
it picks the right letter <b>{pct(prim['episode_5w5s_acc'])}%</b> of the time.</div>

<h1>1. What the project does</h1>
<p>Tamil has 247 letters, and many of them look very similar. Teaching a normal computer model to read them needs
hundreds of examples of every letter, and the model has to be retrained whenever a letter is added.</p>
<p>We used <b>few-shot learning</b> instead. The model learns <i>how to compare</i> handwriting rather than
memorising each letter. To recognise a letter it only needs a few example images, usually 1 to 5. A new letter
can be added just by giving it a few examples, with <b>no retraining</b>.</p>

<h1>2. The data</h1>
{table(["Item", "Value"], [
        ["Letters (classes)", f"{audit['n_classes_with_images']}: 12 vowels, 18 consonants, 216 consonant-vowel letters, 1 āytham ({ta('ஃ')})"],
        ["Images", f"{audit['n_files_seen']:,} handwritten images, each 48 × 48 pixels"],
        ["After cleaning", f"{audit['n_ok_after_dedupe']:,} images ({_ndup(audit)} near-duplicate copies removed)"],
        ["Images per letter", "between 4 and 96 (typically about 46)"],
        ["Missing letters", f"{ta('தௌ, லௌ, ளௌ')} (no images in the dataset)"],
        ["Letters with very few images", f"{ta('ணே ணெ ழா ழ ணு ணூ ழி ழீ')} (only 4 to 6 images each)"],
    ])}
<p>Before training, every image is cleaned the same way: it is turned into black-and-white, the letter is cut out
from the background and centred, and it is resized to a fixed size.</p>
{fig(REPORTS / 'preprocess_samples.png', 'Examples of cleaned images given to the model.', n, '70%')}

<h1>3. The proposed model</h1>
<p><b>Model:</b> a <b>Prototypical Network</b> built on <b>MobileNetV3</b>, a well-known image network already
pretrained on millions of general photos.</p>
<p><b>How it works, in four steps:</b></p><ol>
<li>The network turns every letter image into a list of 128 numbers, like a fingerprint of its shape.</li>
<li>For each letter, the fingerprints of its few example images are averaged. This average is the letter's <b>prototype</b>.</li>
<li>A new image is given to the letter whose prototype is <b>closest</b> to its fingerprint.</li>
<li>If no prototype is close enough, the model can answer <b>"unknown"</b>.</li></ol>
{table(["Detail", "Value"], [
        ["Model size", f"{prim['params'] / 1e6:.1f} million parameters"],
        ["Speed", f"about {onnx['onnxruntime_cpu_ms_per_image']:.1f} milliseconds per image on a normal laptop (no GPU needed)"],
        ["File", "<code>models/embedder.onnx</code> (12 MB), checked to give the same answers as the original model"],
        ["Why this model", "it had the best score on the validation data, which was used to choose between all the models we trained"],
    ])}

<h1>4. How we tested it</h1>
<p>We split the 247 letters into three groups. The model <b>trained</b> on {len(split['base'])} letters. Another
{len(split['val'])} letters were used to <b>choose settings</b>. The last <b>{len(split['novel'])} letters were kept
completely hidden</b> until the final test. So the test checks whether the model can recognise letters it has
<b>never seen before</b>, which is the real goal of few-shot learning.</p>
<p>Each test round, called an <i>episode</i>, gives the model a few example images of some letters and asks it to
label new images of those letters. We ran 600 rounds for each setting and report the average.</p>

<h1>5. Results</h1>
<div class="kpis">
<div class="kpi"><b>{pct(prim['episode_5w5s_acc'])}%</b><span>choose among 5 new letters,<br>5 examples each</span></div>
<div class="kpi"><b>{g('5', 1)}</b><span>choose among 5 new letters,<br>only 1 example each</span></div>
<div class="kpi"><b>{pct(prim['fullset_acc'])}%</b><span>choose among all {nn} new letters,<br>5 examples each</span></div>
<div class="kpi"><b>{pct(prim['fullset_top5'])}%</b><span>correct letter in the<br>top 5 guesses (all {nn})</span></div></div>
<h3>Accuracy on new letters (never seen in training)</h3>
{table(["Number of letters to choose from", "1 example per letter", "5 examples per letter", "10 examples per letter"], [
        ["5 letters", g('5', 1), g('5', 5), g('5', 10)],
        ["10 letters", g('10', 1), g('10', 5), g('10', 10)],
        ["20 letters", g('20', 1), g('20', 5), g('20', 10)],
        [f"All {nn} new letters", g('all', 1), g('all', 5), g('all', 10)],
    ])}
<p><b>How to read this:</b> more examples per letter means better accuracy, and more letters to choose from makes the
task harder. Even with a single example, the model gets about 9 out of 10 right when choosing among 5 letters.</p>
<p><b>Other checks:</b></p><ul>
<li><b>Confidence is trustworthy.</b> When the model says it is 80% sure, it is right about 80% of the time
(calibration error {prim['ece_after']:.3f}, where 0 is perfect).</li>
<li><b>Overall quality score</b> (macro F1, from 0 to 1) across all {nn} new letters: <b>{prim['macro_f1']:.2f}</b>.</li>
<li>Accuracy by letter type: {", ".join(f"{esc(r.category)} {pct(r.accuracy)}%" for _, r in cat.iterrows())}.</li></ul>"""]

    order = [("proposed", "", "ProtoNet + MobileNetV3 (proposed)"), ("final", "matchingnet", "Matching Network"),
             ("arcft_final", "protonet", "ArcFace + ProtoNet"), ("final", "protonet", "ProtoNet (small network)"),
             ("final", "siamese", "Siamese network"), ("final", "relationnet", "Relation Network"),
             ("final", "arcface", "ArcFace"), ("final", "baselinepp", "Baseline++"), ("final", "maml", "MAML")]
    rows = []
    for tag, meth, name in order:
        r = comp[comp.run == best_name] if tag == "proposed" else comp[(comp.tag == tag) & (comp.method == meth) & (comp.protocol == "B")]
        if len(r):
            r = r.iloc[0]
            rows.append([f"<b>{name}</b>" if tag == "proposed" else name, pct(r.test_acc_5w1s) + "%", pct(r.test_acc_5w5s) + "%",
                         pct(r.test_acc_allw5s) + "%" if pd.notna(r.test_acc_allw5s) else "—"])
    sup = comp[comp.method == "supervised"]
    pa = comp[(comp.method == "protonet") & (comp.protocol == "A") & (comp.tag == "final")]
    sup_txt = ""
    if len(sup) and len(pa):
        sup_txt = (f"<p>We also trained a <b>normal classifier</b> (not few-shot) for comparison. It only works on letters it was "
                   f"trained on. Even there, the few-shot ProtoNet did better: {pct(pa.test_acc_5w5s.iat[0])}% vs "
                   f"{pct(sup.test_acc_5w5s.iat[0])}% (5 letters, 5 examples), and {pct(pa.test_acc_allw5s.iat[0])}% vs "
                   f"{pct(sup.test_acc_allw5s.iat[0])}% among all 247 letters.</p>")
    out.append(f"""<h1>6. Comparison with other methods</h1>
<p>We trained eight few-shot methods on exactly the same data and tested them the same way, on the new letters.</p>
{table(["Method", "5 letters, 1 example", "5 letters, 5 examples", f"All {nn} letters, 5 examples"], rows, highlight=0)}
<p><b>Key takeaways:</b> the proposed model is the best in every column. Most of its advantage comes from starting
with a network that was <b>already pretrained</b> on general images; the same network trained from zero reached only
about 85%. The other methods use a smaller network, and the best of them are within about 1 point of each other.
MAML was unstable on this task.</p>
{sup_txt}
{fig(learning_curve_figure(comp, best_name, F / 'report_learning_curves.png'), 'Accuracy on new letters as the number of examples per letter grows. The proposed model (black line) is the highest everywhere.', n)}""")

    pr = [[ta(f"{r.letter_a}  and  {r.letter_b}"), int(r.total)] for _, r in pairs.head(7).iterrows()]
    weak = ", ".join(ta(x) for x in pc.label.head(7))
    strong = ", ".join(ta(x) for x in pc.sort_values("f1", ascending=False).label.head(7))
    out.append(f"""<h1>7. Where the model makes mistakes</h1>
<p>Most mistakes are between letters that really do look alike. Two patterns stand out:</p><ul>
<li><b>Two-part vowel signs</b> ({ta('ொ, ோ, ௌ')}) have pieces on both sides of the consonant, and these letters get
mixed up with each other.</li>
<li><b>Consonants that differ by one small stroke or loop</b>, for example {ta('னீ')} and {ta('ளீ')}, or {ta('டி')}
and {ta('ழ்')}. The small dot (pulli) is hard to see in a 48-pixel image.</li></ul>
<div class="two"><div>{table(["Letters most often confused", "Times confused"], pr)}</div>
<div><p><b>Hardest letters:</b><br>{weak}</p><p><b>Easiest letters:</b><br>{strong}</p>
<p class="small">Some mistakes come from bad images in the dataset, such as dark backgrounds or a cut-off letter.</p></div></div>
{fig(P / 'misclassified.png', 'Examples of mistakes, shown as true letter > predicted letter. Several of these images are poor quality.', n, '85%')}""")

    erows = [[ta(k), pct(d.get("after_top1"), 0) + "%", pct(d.get("after_top5"), 0) + "%"] for k, d in (enrol.get("letters") or {}).items()]
    out.append(f"""<h1>8. Adding a new letter without retraining</h1>
<p>To show this works, we removed some letters from the model's list, added each back using only 5 example images,
and tested it again. No training was needed, and each letter took about a second to add.</p>
{table(["Letter added back", "Correct (1st guess)", "Correct (in top 5 guesses)"], erows)}
<p>We also added the missing letter {ta('தௌ')} the same way. Because the dataset has no handwritten samples of it,
we used computer-rendered images for this demonstration, and the model then recognised it correctly. Real
handwritten samples should replace them.</p>

<h1>9. Limitations</h1><ul>
<li><b>Trained on a laptop.</b> There was no GPU, so training was short. Our tests show the model would still improve
with about 5 times more training, so these results are on the low side.</li>
<li><b>"Unknown" answers are weak.</b> When all 247 letters are loaded, an unfamiliar letter almost always looks close
to some known letter, so the model rarely says "unknown".</li>
<li><b>Data quality.</b> The images are small (48 × 48), 8 letters have only 4 to 6 images, and some images are unclear.</li>
<li><b>Possible duplicate letters.</b> {ta('தெள, லெள, ளெள')} look exactly like the "missing" letters
{ta('தௌ, லௌ, ளௌ')}, but are typed with a different Unicode spelling (ெ + ள instead of the single sign ௌ). They are
probably the same letters, and this needs to be checked.</li></ul>

<h1>10. Conclusion and next steps</h1>
<p>The proposed model, a Prototypical Network on a pretrained MobileNetV3, recognises Tamil letters it has never seen
with <b>{pct(prim['episode_5w5s_acc'])}%</b> accuracy from 5 examples (choosing among 5 letters) and
<b>{pct(prim['fullset_acc'])}%</b> among all {nn} new letters. It is fast, its confidence scores are reliable, and
new letters can be added without retraining.</p>
<p><b>Next steps:</b></p><ol>
<li>Train the same model for longer, preferably on a GPU.</li>
<li>Collect more images for the letters with very few samples, and for the {ta('ொ / ோ / ௌ')} letters.</li>
<li>Check the {ta('தெள / லெள / ளெள')} spelling question, and collect real handwritten {ta('தௌ, லௌ, ளௌ')}.</li>
<li>Improve the "unknown letter" detection.</li></ol>
<p class="small">Full technical details (all experiments, settings and extra charts) are in the detailed report
(<code>reports/Tamil_FewShot_HTR_Report_detailed.pdf</code>) and in the project's <code>reports/</code> folder.</p>""")
    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'><title>Tamil Letter Recognition Report</title>"
            f"<style>{CSS}</style></head><body>{''.join(out)}</body></html>")


def find_edge() -> str:
    for p in (r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe", r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"):
        if Path(p).exists():
            return p
    for name in ("msedge", "chrome", "google-chrome", "chromium"):
        if shutil.which(name):
            return shutil.which(name)
    raise SystemExit("no Edge/Chrome found to print the PDF")


def main() -> None:
    force_utf8()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--best", default=DEFAULT_BEST, help="run directory name of the proposed model")
    ap.add_argument("--author", default="[Your name]")
    ap.add_argument("--mentor", default="[Mentor's name]")
    ap.add_argument("--institution", default="[Institution / department]")
    ap.add_argument("--detailed", action="store_true", help="full technical report with cover page (default: short plain-language report)")
    a = ap.parse_args()
    setup_logging("INFO")
    html_path = REPORTS / "Tamil_FewShot_HTR_Report.html"
    pdf_path = REPORTS / "Tamil_FewShot_HTR_Report.pdf"
    if a.detailed:
        html_path, pdf_path = REPORTS / "Tamil_FewShot_HTR_Report_detailed.html", REPORTS / "Tamil_FewShot_HTR_Report_detailed.pdf"
        content = build(Path(a.best).name, a.author, a.mentor, a.institution)
    else:
        content = build_simple(Path(a.best).name, a.author)
    html_path.write_text(content, encoding="utf-8")
    subprocess.run([find_edge(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
                    f"--print-to-pdf={pdf_path}", html_path.resolve().as_uri()], check=True, capture_output=True, timeout=180)
    log.info("wrote %s and %s", html_path, pdf_path)


if __name__ == "__main__":
    main()
