"""Render a one-page results summary PNG from reports/model_comparison.csv."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "reports" / "model_comparison.csv", encoding="utf-8-sig")
OUT = ROOT / "reports" / "figures" / "results_summary.png"

SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
plt.rcParams.update({
    "font.family": "Segoe UI", "font.size": 11, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
})

NAMES = {"matchingnet": "Matching Net", "protonet": "ProtoNet", "relationnet": "Relation Net",
         "siamese": "Siamese (triplet)", "arcface": "ArcFace", "baselinepp": "Baseline++",
         "maml": "FO-MAML", "supervised": "Supervised CE"}

def pick(tag, method=None, backbone="conv4", proto="B"):
    q = (df.tag == tag) & (df.backbone == backbone) & (df.protocol == proto)
    if method: q &= df.method == method
    return df[q].iloc[0]

rows = [("ProtoNet + MobileNetV3-L (pretrained, 96px)  (best)", pick("abl_mnv3_pre96", backbone="mobilenetv3_large"))]
for m in ["matchingnet", "protonet", "siamese", "relationnet", "arcface", "baselinepp", "maml"]:
    rows.append((f"{NAMES[m]} · Conv4", pick("final", m)))
rows.insert(3, ("ProtoNet ← ArcFace init · Conv4", pick("arcft_final", "protonet")))
rows.append(("ProtoNet + MobileNetV3-L (scratch, 96px)", pick("abl_mnv3_scr96", backbone="mobilenetv3_large")))

fig = plt.figure(figsize=(18, 25), dpi=150)
gs = fig.add_gridspec(5, 2, height_ratios=[0.55, 1.35, 1.15, 1.05, 0.55], hspace=0.38, wspace=0.42,
                      left=0.11, right=0.97, top=0.95, bottom=0.03)

# ---------- header ----------
fig.text(0.02, 0.985, "Few-Shot Handwritten Tamil Character Recognition: Results",
         fontsize=26, weight="bold", va="top")
fig.text(0.02, 0.968, "247 classes · 14,463 images (48×48) · TEST set, ≥600 episodes per cell, 95% CI · "
         "Protocol B = 37 novel letters never seen in training · CPU-only",
         fontsize=12.5, color=INK2, va="top")
hdr = fig.add_subplot(gs[0, :]); hdr.axis("off")
best = rows[0][1]
tiles = [("95.6%", "5-way 5-shot accuracy\n(novel letters, ±0.4)"),
         ("89.7%", "5-way 1-shot accuracy\n(novel letters)"),
         ("83.5%", "all-37-way 5-shot\n(novel letters)"),
         ("0.938", "open-set AUROC\n(unknown-letter rejection)"),
         ("1.4 ms", "CPU latency / image\n(ONNX Runtime)")]
w = 1 / len(tiles)
for i, (big, small) in enumerate(tiles):
    x = i * w + 0.006
    hdr.add_patch(FancyBboxPatch((x, 0.02), w - 0.012, 0.78, boxstyle="round,pad=0,rounding_size=0.02",
                                 transform=hdr.transAxes, fc="#f1f0ec", ec="none"))
    hdr.text(x + 0.02, 0.62, big, fontsize=30, weight="bold", color=BLUE if i == 0 else INK,
             transform=hdr.transAxes, va="center")
    hdr.text(x + 0.02, 0.25, small, fontsize=11.5, color=INK2, transform=hdr.transAxes, va="center")
hdr.text(0, 0.86, "Best / deployed model: ProtoNet + ImageNet-pretrained MobileNetV3-Large @ 96 px "
         "(4.3 M params, 8 epochs) → models/embedder.onnx", fontsize=13.5, weight="bold",
         transform=hdr.transAxes, va="bottom")

# ---------- models table ----------
ax = fig.add_subplot(gs[1, :]); ax.axis("off")
ax.set_title("Models used and benchmark results (Protocol B, novel classes)", loc="left",
             fontsize=16, weight="bold", pad=8)
cols = ["Model", "Params", "CPU ms*", "Epochs", "5w-1s", "5w-5s (±CI)", "20w-5s", "All-way 5s",
        "Macro F1", "ECE → calib.", "Open-set AUROC"]
cell = []
for name, r in rows:
    allw = "—" if pd.isna(r.test_acc_allw5s) else f"{r.test_acc_allw5s*100:.1f}"
    cell.append([name, f"{r.params/1e3:,.0f}k" if r.params < 1e6 else f"{r.params/1e6:.2f}M",
                 f"{r.cpu_ms_per_image:.1f}", f"{int(r.epochs_run)}",
                 f"{r.test_acc_5w1s*100:.1f}", f"{r.test_acc_5w5s*100:.1f} ± {r.test_ci_5w5s*100:.1f}",
                 f"{r.test_acc_20w5s*100:.1f}", allw, f"{r.macro_f1:.3f}",
                 f"{r.ece_before:.3f} → {r.ece_after:.3f}", f"{r.openset_auroc:.3f}"])
t = ax.table(cellText=cell, colLabels=cols, loc="upper center", cellLoc="center",
             colWidths=[0.27, 0.06, 0.06, 0.055, 0.06, 0.1, 0.06, 0.075, 0.065, 0.11, 0.09])
t.auto_set_font_size(False); t.set_fontsize(11.5); t.scale(1, 2.35)
numcols = {4: "test_acc_5w1s", 5: "test_acc_5w5s", 6: "test_acc_20w5s", 7: "test_acc_allw5s",
           8: "macro_f1", 10: "openset_auroc"}
colbest = {c: max(r[k] for _, r in rows if not pd.isna(r[k])) for c, k in numcols.items()}
for (i, j), c in t.get_celld().items():
    c.set_edgecolor(GRID); c.set_linewidth(0.6)
    if i == 0:
        c.set_facecolor("#ecebe6"); c.set_text_props(weight="bold", color=INK)
    else:
        c.set_facecolor("#eef4fc" if i == 1 else SURF)
        if j == 0: c.set_text_props(ha="left"); c._loc = "left"
        r = rows[i - 1][1]
        if j in numcols and not pd.isna(r[numcols[j]]) and r[numcols[j]] == colbest[j]:
            c.set_text_props(weight="bold", color=BLUE)
ax.text(0, -0.02, "* embedding only, batch 1, PyTorch, 2 threads (ONNX Runtime: 1.4 ms for the best model). "
        "Bold blue = best in column. FO-MAML trained 8 epochs and is unstable (5-shot < 1-shot). "
        "ProtoNet across 3 seeds: 94.4 ± 0.3 (5w-5s).", transform=ax.transAxes, fontsize=10.5, color=MUTED, va="top")

# ---------- 5w5s bars with CI ----------
ax = fig.add_subplot(gs[2, 0])
order = sorted(rows, key=lambda x: x[1].test_acc_5w5s)
y = range(len(order))
vals = [r.test_acc_5w5s * 100 for _, r in order]
ci = [r.test_ci_5w5s * 100 for _, r in order]
colors = [BLUE if "(best)" in n else "#b9b8b2" for n, _ in order]
ax.barh(y, vals, xerr=ci, color=colors, height=0.62, error_kw=dict(ecolor=INK2, lw=1.2, capsize=3))
ax.set_yticks(list(y), [n.replace("  (best)", " (best)").replace("ProtoNet + MobileNetV3-L ", "MNv3 ").replace(", 96px)", ")").replace(" · Conv4", "") for n, _ in order], fontsize=10.5)
for yi, v, c in zip(y, vals, ci):
    ax.text(v + c + 0.5, yi, f"{v:.1f}", va="center", fontsize=10.5, color=INK)
ax.set_xlim(60, 100); ax.set_xlabel("5-way 5-shot test accuracy (%)")
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.set_title("Benchmark: 5-way 5-shot on novel letters (±95% CI)", loc="left", fontsize=14, weight="bold")

# ---------- benchmark heatmap ----------
ax = fig.add_subplot(gs[2, 1])
bench = [("5w-1s", "test_acc_5w1s"), ("5w-5s", "test_acc_5w5s"), ("20w-5s", "test_acc_20w5s"),
         ("All-way 5s", "test_acc_allw5s"), ("Full-set", "fullset_acc")]
hm_rows = [(n.replace("  (best)", " (best)").replace(" · Conv4", "").replace("ProtoNet + MobileNetV3-L ", "MNv3 ").replace(", 96px)", ")"), r)
           for n, r in rows]
M = [[r[k] * 100 if not pd.isna(r[k]) else float("nan") for _, k in bench] for _, r in hm_rows]
cmap = LinearSegmentedColormap.from_list("seqblue", ["#eef4fc", "#9cc2ef", BLUE, "#0f3f7a"])
im = ax.imshow(M, cmap=cmap, vmin=50, vmax=100, aspect="auto")
for i, row in enumerate(M):
    for j, v in enumerate(row):
        txt = "—" if v != v else f"{v:.1f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=10.5,
                color="white" if v == v and v > 82 else INK)
ax.set_xticks(range(len(bench)), [b for b, _ in bench]); ax.xaxis.tick_top()
ax.set_yticks(range(len(hm_rows)), [n for n, _ in hm_rows], fontsize=10.5)
for s in ax.spines.values(): s.set_visible(False)
ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.outline.set_visible(False)
cb.set_label("accuracy (%)")
ax.set_title("Benchmark grid: accuracy across task difficulty", loc="left", fontsize=14, weight="bold", pad=28)

# ---------- ablations ----------
ax = fig.add_subplot(gs[3, 0])
lrs = [("1e-4", "lr1e-4"), ("3e-4", "lr3e-4"), ("1e-3", "lr1e-3"), ("3e-3", "lr3e-3")]
lv = [pick(t, "protonet").test_acc_5w5s * 100 for _, t in lrs]
ax.plot(range(4), lv, color=BLUE, lw=2, marker="o", ms=8, mec=SURF, mew=2, label="Conv4 ProtoNet, LR sweep (8 ep)")
for i, v in enumerate(lv): ax.text(i, v + 0.6, f"{v:.1f}", ha="center", fontsize=10.5)
ax.set_xticks(range(4), [f"lr {a}" for a, _ in lrs])
scr, pre = rows[-1][1].test_acc_5w5s * 100, best.test_acc_5w5s * 100
ax.scatter([4.3], [scr], color=ORANGE, s=90, zorder=3, ec=SURF, lw=2, label="MobileNetV3-L from scratch")
ax.scatter([4.3], [pre], color=AQUA, s=90, zorder=3, ec=SURF, lw=2, label="MobileNetV3-L ImageNet-pretrained")
ax.annotate("", xy=(4.3, pre - 0.4), xytext=(4.3, scr + 0.4), arrowprops=dict(arrowstyle="->", color=INK2, lw=1.4))
ax.text(4.45, (pre + scr) / 2, f"+{pre - scr:.1f} pp\nfrom\npretraining", fontsize=10.5, va="center", color=INK)
ax.text(4.45, pre, f"{pre:.1f}", va="center", fontsize=10.5); ax.text(4.45, scr, f"{scr:.1f}", va="center", fontsize=10.5)
ax.set_xlim(-0.4, 5.3); ax.set_ylim(83, 98)
ax.set_ylabel("5-way 5-shot test acc. (%)"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=10)
ax.set_title("Ablations: learning rate and pretraining", loc="left", fontsize=14, weight="bold")

# ---------- protocol A vs B ----------
ax = fig.add_subplot(gs[3, 1])
groups = [("ProtoNet\n(Prot. A, 15 ep)", pick("final", "protonet", proto="A")),
          ("ProtoNet\n(Prot. A, 40 ep)", pick("epochstudy", "protonet", proto="A")),
          ("Supervised CE\n(Prot. A)", pick("final", "supervised", proto="A")),
          ("ProtoNet\n(Prot. B, 15 ep)", pick("final", "protonet")),
          ("ProtoNet+MNv3\n(Prot. B, best)", best)]
import numpy as np
x = np.arange(len(groups)); bw = 0.36
a = [g.test_acc_5w5s * 100 for _, g in groups]; b = [g.test_acc_allw5s * 100 for _, g in groups]
ax.bar(x - bw / 2 - 0.01, a, bw, color=BLUE, label="5-way 5-shot")
ax.bar(x + bw / 2 + 0.01, b, bw, color=ORANGE, label="All-way 5-shot (A: 247 letters · B: 37)")
for xi, v in zip(x - bw / 2, a): ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=10)
for xi, v in zip(x + bw / 2, b): ax.text(xi, v + 1, f"{v:.1f}", ha="center", fontsize=10)
ax.set_xticks(x, [n for n, _ in groups], fontsize=10); ax.set_ylim(0, 108)
ax.set_ylabel("test accuracy (%)"); ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, 1.0))
ax.set_title("Known letters (A) vs novel letters (B)", loc="left", fontsize=14, weight="bold")

# ---------- takeaways ----------
ax = fig.add_subplot(gs[4, :]); ax.axis("off")
ax.set_title("Key findings", loc="left", fontsize=16, weight="bold")
notes = [
    "1. ImageNet pretraining is the single biggest gain: MobileNetV3 pretrained 95.6% vs 85.0% from scratch (+10.6 pp); +1.2 pp over Conv4 ProtoNet at 5w-5s and +3.2 pp at all-way.",
    "2. Episodic metric learners (Matching Net, ProtoNet, Siamese, Relation Net) land within ~1 pp at 5-way 5-shot; harder all-way tasks separate them (Matching Net / ProtoNet lead).",
    "3. FO-MAML is unstable on this data (71.3 ± 5.4% at 5-shot); Baseline++ and ArcFace trail the episodic methods by 1–2 pp.",
    "4. Temperature scaling cuts calibration error ~3–8× (best model ECE 0.164 → 0.044). LR 3e-3 beats the default 1e-3, and 40 epochs beats 15 (96.6% vs 95.3% on A), so the 15-epoch runs under-train.",
    "5. Errors cluster on two-part vowel-sign letters (the -o / -ō / -au series) and look-alike consonants (e.g. ṉ/ḷ, m/v, t/ṟ); the weakest letters (-au forms) score F1 0.30–0.35.",
]
for i, n in enumerate(notes):
    ax.text(0, 0.86 - i * 0.2, n, fontsize=12, transform=ax.transAxes, va="top",
            fontfamily=["Segoe UI", "Nirmala UI"])

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, facecolor=SURF)
print(OUT)
