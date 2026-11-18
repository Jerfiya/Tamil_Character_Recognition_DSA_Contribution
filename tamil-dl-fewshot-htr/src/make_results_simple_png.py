"""Simple one-page results PNG: best model, metrics explained, model comparison."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "reports" / "model_comparison.csv", encoding="utf-8-sig")
OUT = ROOT / "reports" / "figures" / "results_simple.png"

SURF, INK, INK2, MUTED, GRID, TILE = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0", "#f1f0ec"
BLUE, GRAY = "#2a78d6", "#c4c3bd"
plt.rcParams.update({"font.family": "Segoe UI", "font.size": 12, "text.color": INK,
                     "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "xtick.color": INK2, "ytick.color": INK2})

def pick(method, tag="final", backbone="conv4"):
    q = (df.method == method) & (df.tag == tag) & (df.backbone == backbone) & (df.protocol == "B")
    return df[q].iloc[0]

models = [
    ("ProtoNet + MobileNetV3 (best)", pick("protonet", "abl_mnv3_pre96", "mobilenetv3_large")),
    ("Matching Network", pick("matchingnet")),
    ("Prototypical Network", pick("protonet")),
    ("Siamese Network", pick("siamese")),
    ("Relation Network", pick("relationnet")),
    ("ArcFace", pick("arcface")),
    ("Baseline++", pick("baselinepp")),
    ("MAML", pick("maml")),
]

fig = plt.figure(figsize=(14, 19.5), dpi=150)
fig.text(0.05, 0.975, "Tamil Handwriting Recognition: Results", fontsize=28, weight="bold", va="top")
fig.text(0.05, 0.952, "Task: recognise new handwritten Tamil letters from only 1 or 5 example images.\n"
         "All scores are on test letters the model never saw during training.",
         fontsize=13.5, color=INK2, va="top", linespacing=1.5)

# ---- 1. best model ----
fig.text(0.05, 0.905, "1.  Best model", fontsize=18, weight="bold", va="top")
fig.text(0.05, 0.883, "Prototypical Network with a pretrained MobileNetV3 backbone",
         fontsize=13.5, color=INK2, va="top")
tiles = [("95.6%", "accuracy with\n5 examples"), ("89.7%", "accuracy with\n1 example"),
         ("0.82", "macro F1\nscore"), ("1.4 ms", "time per image\non a CPU")]
for i, (big, small) in enumerate(tiles):
    x = 0.05 + i * 0.228
    ax = fig.add_axes([x, 0.785, 0.21, 0.085]); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.08",
                                transform=ax.transAxes, fc=TILE, ec="none"))
    ax.text(0.08, 0.66, big, fontsize=28, weight="bold", color=BLUE if i == 0 else INK, va="center")
    ax.text(0.08, 0.25, small, fontsize=11.5, color=INK2, va="center", linespacing=1.3)

# ---- 2. metrics ----
fig.text(0.05, 0.765, "2.  How the models were evaluated", fontsize=18, weight="bold", va="top")
metrics = [
    ("Accuracy (5-way, 1-shot / 5-shot)",
     "Pick the right letter out of 5, given 1 or 5 example images of each. Higher is better."),
    ("95% confidence interval (±)",
     "Accuracy averaged over 600 random tests; ± is how much the result could vary."),
    ("Macro F1 score (0–1)",
     "Balance of precision and recall, averaged so every letter counts equally."),
    ("Open-set AUROC (0–1)",
     "How well the model says \"I don't know this letter\" for unknown letters."),
    ("Calibration error (ECE)",
     "Whether confidence matches reality (90% sure ≈ 90% right). Lower is better."),
    ("Latency (ms per image)", "How fast one image is processed on a normal CPU."),
]
ax = fig.add_axes([0.05, 0.555, 0.9, 0.195]); ax.axis("off")
t = ax.table(cellText=[[a, b] for a, b in metrics], colLabels=["Metric", "What it means"],
             colWidths=[0.33, 0.67], loc="upper left", cellLoc="left")
t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 2.3)
for (i, j), c in t.get_celld().items():
    c.set_edgecolor(GRID); c.PAD = 0.03
    c.set_facecolor(TILE if i == 0 else SURF)
    if i == 0 or j == 0: c.set_text_props(weight="bold")

# ---- 3. comparison chart ----
fig.text(0.05, 0.545, "3.  Model comparison (accuracy with 5 examples)", fontsize=18, weight="bold", va="top")
ax = fig.add_axes([0.31, 0.335, 0.6, 0.19])
order = models[::-1]
vals = [r.test_acc_5w5s * 100 for _, r in order]
ax.barh(range(len(order)), vals, height=0.62,
        color=[BLUE if "best" in n else GRAY for n, _ in order])
for i, v in enumerate(vals):
    ax.text(v + 0.4, i, f"{v:.1f}%", va="center", fontsize=12, weight="bold" if i == len(order) - 1 else None)
ax.set_yticks(range(len(order)), [n for n, _ in order], fontsize=12.5)
ax.set_xlim(60, 100); ax.set_xlabel("accuracy (%)", color=INK2)
for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID); ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)

# ---- 4. full metric table ----
fig.text(0.05, 0.285, "4.  All metrics per model", fontsize=18, weight="bold", va="top")
ax = fig.add_axes([0.05, 0.095, 0.9, 0.175]); ax.axis("off")
cols = ["Model", "1-shot acc.", "5-shot acc.", "Macro F1", "Open-set AUROC", "ECE*", "ms / image"]
cells = [[n, f"{r.test_acc_5w1s*100:.1f}%", f"{r.test_acc_5w5s*100:.1f} ± {r.test_ci_5w5s*100:.1f}%",
          f"{r.macro_f1:.2f}", f"{r.openset_auroc:.2f}", f"{r.ece_after:.3f}",
          f"{r.cpu_ms_per_image:.1f}"] for n, r in models]
t = ax.table(cellText=cells, colLabels=cols, loc="upper left", cellLoc="center",
             colWidths=[0.3, 0.11, 0.15, 0.1, 0.13, 0.09, 0.12])
t.auto_set_font_size(False); t.set_fontsize(12); t.scale(1, 1.95)
for (i, j), c in t.get_celld().items():
    c.set_edgecolor(GRID)
    c.set_facecolor(TILE if i == 0 else ("#eef4fc" if i == 1 else SURF))
    if i == 0: c.set_text_props(weight="bold")
    if j == 0 and i > 0: c.set_text_props(ha="left"); c._loc = "left"
    if i == 1 and j in (0, 1, 2, 3, 4): c.set_text_props(weight="bold", color=BLUE if j else INK)
fig.text(0.05, 0.09, "* calibration error after temperature scaling. Time per image measured in PyTorch "
         "(the best model runs at 1.4 ms in ONNX). Bold blue = best in that column.\n"
         "Conv4 models are small (~0.2 M parameters); "
         "MobileNetV3 has 4.3 M.", fontsize=10.5, color=MUTED, va="top", linespacing=1.4)

# ---- 5. takeaways ----
fig.text(0.05, 0.065, "Key takeaways", fontsize=16, weight="bold", va="top")
fig.text(0.05, 0.043,
         "•  Starting from an ImageNet-pretrained network gave the biggest boost (+10.6 points over training from scratch).\n"
         "•  Most few-shot methods score about 92–95%. MAML was unstable and did worst (71%).",
         fontsize=12.5, va="top", linespacing=1.7)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=SURF)
print(OUT)
