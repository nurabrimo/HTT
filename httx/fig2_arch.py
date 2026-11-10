"""Fig. 2 redrawn (same content as the original): HTT architecture and its Baseline. Vector PDF + 600 dpi PNG.
Usage: fig2_arch.py <out_dir>"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
BLUE, BLUE_BG = "#2a78d6", "#e8f0fb"
ORANGE, ORANGE_BG = "#eb6834", "#fdebe3"
GREEN, GREEN_BG = "#2e8b57", "#e5f3ea"
INK, INK2, MUTED, GREY_BG = "#0b0b0b", "#3d3c39", "#8a8984", "#f1f1ef"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.2, "mathtext.fontset": "dejavusans"})

fig = plt.figure(figsize=(7.2, 4.15))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 72); ax.set_ylim(0, 41.5); ax.axis("off")


def box(x, y, w, h, text, ec=INK2, fc="white", lw=0.9, fs=7.2, weight="normal", color=INK, r=0.9, z=2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}", ec=ec, fc=fc, lw=lw, zorder=z))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight,
                zorder=z + 1, linespacing=1.35)
    return (x, y, w, h)


def arrow(p, q, c=INK2, lw=0.9, ms=7):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=c, lw=lw, zorder=1,
                                 shrinkA=0, shrinkB=0))


def label(x, y, letter, title):
    ax.text(x, y, letter, fontsize=10, weight="bold", color=INK, va="baseline")
    ax.text(x + 2.0, y, title, fontsize=8.2, color=INK2, va="baseline")


# ----------------------------------------------------------------------------------------------- panel a
label(0.8, 39.3, "a", "HTT: two-level hierarchical model")
# inputs
ry = [31.6, 25.6, 16.0]
for i, (y, sub) in enumerate(zip(ry, ["1", "2", "n"])):
    box(0.8, y, 12.6, 4.4, f"Report $R_{{{sub}}}$  (day $t_{{{sub}}}$)\n[CANCER:type] token", ec=MUTED, fc=GREY_BG)
ax.text(7.1, 21.6, "⋮", ha="center", va="center", fontsize=13, color=INK2)

# Level 1 container
L1 = (15.6, 13.6, 16.4, 23.2)
box(*L1, None, ec=BLUE, fc=BLUE_BG, lw=1.2, r=1.2, z=0)
ax.text(L1[0] + L1[2] / 2, L1[1] + L1[3] - 2.0, "Level 1: report encoder", ha="center", fontsize=7.8, weight="bold", color=BLUE)
b1 = box(16.8, 26.2, 14.0, 5.6, "BiomedBERT\n+ LoRA (r = 16)\nshared across reports", ec=BLUE)
b2 = box(16.8, 19.6, 14.0, 4.6, "mean pooling\n768-d → 512-d projection", ec=BLUE)
b3 = box(16.8, 14.8, 14.0, 3.2, "report embedding $e_k$", ec=BLUE)
arrow((23.8, 26.2), (23.8, 24.2), BLUE); arrow((23.8, 19.6), (23.8, 18.0), BLUE)
for y in ry:
    arrow((13.4, y + 2.2), (16.8, 29.0 if y > 30 else (28.0 if y > 25 else 27.0)), MUTED)

# Level 2 container
L2 = (34.4, 13.6, 20.6, 23.2)
box(*L2, None, ec=BLUE, fc=BLUE_BG, lw=1.2, r=1.2, z=0)
ax.text(L2[0] + L2[2] / 2, L2[1] + L2[3] - 2.0, "Level 2: temporal transformer", ha="center", fontsize=7.8, weight="bold", color=BLUE)
c1 = box(35.8, 14.8, 17.8, 3.6, "$e_k$ + time code + type embedding", ec=BLUE)
c2 = box(35.8, 21.0, 17.8, 4.8, "[CLS] + 4 transformer layers\n8 heads, pre-LayerNorm", ec=BLUE)
c3 = box(35.8, 28.4, 17.8, 3.6, "CLS output  |  visit outputs", ec=BLUE)
arrow((44.7, 18.4), (44.7, 21.0), BLUE); arrow((44.7, 25.8), (44.7, 28.4), BLUE)
arrow((30.8, 16.4), (35.8, 16.6), BLUE)

# inputs to Level 2 from below
t1 = box(34.4, 2.4, 7.6, 7.6, "cancer-type\nembedding", ec=INK2)
t2 = box(42.6, 2.4, 14.4, 7.6, "time code\ninitial: TE(elapsed $t_k$)\nHTT-IA: TE(interval $t_k - t_{k-1}$)\nperiods 7 d → 5 y",
         ec=ORANGE, fc=ORANGE_BG, lw=1.1, fs=6.5)
arrow((38.2, 10.0), (39.2, 14.8), INK2); arrow((49.6, 10.0), (49.4, 14.8), ORANGE, lw=1.1)

# heads
h1 = box(58.4, 27.0, 13.2, 6.2, "patient head\n512→256→2\nprimary loss", ec=GREEN, fc=GREEN_BG, lw=1.1)
h2 = box(58.4, 17.4, 13.2, 6.6, "visit head\n512→2\nauxiliary loss (0.3),\ntraining only", ec=GREEN, lw=1.1)
ax.text(65.0, 36.1, "P(label)", ha="center", fontsize=8, weight="bold", color=GREEN)
arrow((65.0, 33.2), (65.0, 35.3), GREEN, lw=1.1)
arrow((53.6, 30.6), (58.4, 30.6), GREEN, lw=1.1)
arrow((53.6, 29.2), (58.4, 21.4), GREEN, lw=1.1)

# ----------------------------------------------------------------------------------------------- panel b
ax.plot([0.8, 32.0], [11.6, 11.6], color="#e4e3df", lw=0.8)
label(0.8, 8.6, "b", "Baseline")
box(0.8, 1.4, 12.6, 5.0, "last report only", ec=MUTED, fc=GREY_BG)
box(15.6, 1.4, 16.4, 5.0, "same Level 1 encoder\n+ single linear head", ec=INK2)
arrow((13.4, 3.9), (15.6, 3.9), INK2)
ax.text(58.4, 6.2, "(b) Baseline head is also applied\nto every report in training with\nthe same visit loss (weight 0.3).",
        fontsize=6.4, color=MUTED, va="center", linespacing=1.35)

for ext, kw in (("pdf", {}), ("png", {"dpi": 600})):
    fig.savefig(os.path.join(OUT, f"Fig2_architecture.{ext}"), bbox_inches="tight", pad_inches=0.04, **kw)
print("saved")
