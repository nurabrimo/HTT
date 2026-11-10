"""Figures 13-15 of the second revision, from summary2.json (aggregates only).
Usage: figs2.py <summary2.json> <summary.json (first revision, for HTT/ceilings)> <out_dir>"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

S = json.load(open(sys.argv[1])); S1 = json.load(open(sys.argv[2])); OUT = sys.argv[3]
os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
                     "savefig.dpi": 300})
CENT = ["MSK", "DFCI", "VICC", "UHN"]


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)


def panel_label(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=10, fontweight="bold", color=INK, ha="right")


# ------------------------------------------------------------------ Fig 13: GENIE real timelines
G = S["genie"]
rows = [("lr_notime", "Logistic regression, no time"), ("lr_time", "Logistic regression + time features"),
        ("gbm_notime", "Gradient boosting, no time"), ("gbm_time", "Gradient boosting + time features"),
        ("htt_none", "HTT, no time"), ("htt_order", "HTT, order only"),
        ("htt_orig_elapsed", "HTT, elapsed (original band)"), ("htt_clin_elapsed", "HTT, elapsed (clinical band)"),
        ("htt_clin_interval", "HTT-IA, interval (clinical band)"), ("htt_clin_both", "HTT, elapsed + interval")]
col = {"htt_orig_elapsed": BLUE, "htt_clin_interval": ORANGE, "lr_time": AQUA, "gbm_time": AQUA}
fig = plt.figure(figsize=(7.2, 5.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1], hspace=0.55, wspace=0.08)
for j, (task, title) in enumerate([("nextprog", "a  Next scan curated as progressing"),
                                   ("mort12", "b  Death within 12 months (1-year landmark)")]):
    ax = fig.add_subplot(gs[0, j])
    for i, (k, lab) in enumerate(rows):
        e = G[task][k]; y = len(rows) - 1 - i
        c = col.get(k, INK2)
        ax.scatter([e["per_centre"][cc] for cc in CENT], [y] * 4, s=10, color=c, alpha=0.35, linewidths=0, zorder=2)
        ax.scatter([e["auroc"]], [y], s=34, color=c, zorder=3, edgecolors="white", linewidths=0.8)
        ax.text(1.0, y, f"{e['auroc']:.3f}", transform=ax.get_yaxis_transform(), va="center", ha="right",
                fontsize=7, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows][::-1] if j == 0 else [])
    ax.tick_params(axis="y", length=0)
    lo = min(min(G[task][k]["per_centre"].values()) for k, _ in rows)
    hi = max(max(G[task][k]["per_centre"].values()) for k, _ in rows)
    ax.set_xlim(lo - 0.01, hi + 0.035)
    ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlabel("AUROC on the held-out institution")
    ax.set_title(title, loc="left", fontsize=8)
# C: contrasts with CIs (next-scan progression)
ax = fig.add_subplot(gs[1, :])
con = G["nextprog"]["contrasts"]
items = [("htt_order - htt_none", "Order only vs no time"),
         ("htt_orig_elapsed - htt_none", "Elapsed (original band) vs no time"),
         ("htt_clin_interval - htt_none", "HTT-IA interval vs no time"),
         ("htt_orig_elapsed - lr_time", "HTT elapsed vs logistic regression + time"),
         ("htt_orig_elapsed - gbm_time", "HTT elapsed vs gradient boosting + time"),
         ("gbm_time - gbm_notime", "Time features in gradient boosting"),
         ("lr_time - lr_notime", "Time features in logistic regression"),
         ("lr_time: shuffle - intact", "Logistic regression + time: gaps shuffled"),
         ("htt_orig_elapsed: shuffle - intact", "HTT elapsed: gaps shuffled")]
for i, (k, lab) in enumerate(items):
    v = con[k]; y = len(items) - 1 - i
    c = BLUE if v["lo"] > 0 else (ORANGE if v["hi"] < 0 else MUTED)
    ax.plot([v["lo"], v["hi"]], [y, y], color=c, lw=2, solid_capstyle="round")
    ax.scatter([v["diff"]], [y], color=c, s=26, zorder=3, edgecolors="white", linewidths=0.8)
    ax.text(1.02, y, f"{v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]   {v['better_centres']}/4",
            va="center", fontsize=7, color=INK2, transform=ax.get_yaxis_transform())
ax.axvline(0, color=MUTED, lw=0.8)
ax.set_yticks(range(len(items))); ax.set_yticklabels([lab for _, lab in items][::-1]); ax.tick_params(axis="y", length=0)
ax.set_xlim(-0.035, 0.07)
ax.set_xlabel("Difference in AUROC, next-scan progression (95% CI; institutions where the first model is higher)")
ax.set_title("c  Paired contrasts across institutions", loc="left", fontsize=8)
ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
panel_label(ax, "")
save(fig, "Fig13_genie_timelines")

# ------------------------------------------------------------------ Fig 14: label provenance
A = S["cdr_audit"]; C = S["chord_vs_genie"]; X = S["chord_vs_genie_crosstab"]
fig = plt.figure(figsize=(7.2, 2.9))
gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.25, 1.2], wspace=1.05)
ax = fig.add_subplot(gs[0, 0])
vals = [A["label_paper_vs_registry"]["kappa"], A["label_corr_vs_registry"]["kappa"]]
ax.bar([0, 1], vals, color=[MUTED, BLUE], width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 0.02, f"κ {v:.2f}", ha="center", fontsize=7, color=INK2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Released\nrule", "Corrected\nrule"], fontsize=7)
ax.set_ylim(0, 1.1); ax.set_ylabel("Agreement with registry grade (κ)")
ax.set_title("a  TCGA grade vs registry", loc="left", fontsize=8)
ax.text(0.5, -0.36, f"n = {A['label_paper_vs_registry']['n']:,} / {A['label_corr_vs_registry']['n']:,} patients",
        transform=ax.transAxes, ha="center", fontsize=6.5, color=MUTED)
# B: crosstab curated vs NLP (row-normalised)
ax = fig.add_subplot(gs[0, 1])
cur_rows = ["Progressing/Worsening/", "Stable/No change", "Improving/Responding", "Not stated/Indetermina"]
cur_lab = ["Progressing", "Stable", "Responding", "Not stated /\nindeterminate"]
nlp_cols = ["Y", "N", "Indeterminate"]
M = np.array([[X.get(c, {}).get(r, 0) for c in nlp_cols] for r in cur_rows], float)
Mn = M / M.sum(1, keepdims=True)
from matplotlib.colors import LinearSegmentedColormap
cm = LinearSegmentedColormap.from_list("b", ["#ffffff", BLUE])
ax.imshow(Mn, cmap=cm, vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{Mn[i, j]*100:.0f}%\n{int(M[i, j]):,}", ha="center", va="center", fontsize=6,
                color="white" if Mn[i, j] > 0.6 else INK)
ax.set_xticks(range(3)); ax.set_xticklabels(["NLP\nyes", "NLP\nno", "NLP\nindet."])
ax.set_yticks(range(4)); ax.set_yticklabels(cur_lab)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("b  MSK-CHORD NLP vs curation", loc="left", fontsize=8)
ax.text(0.5, -0.3, f"{C['matched_reports']:,} matched reports, {C['matched_patients']:,} patients",
        transform=ax.transAxes, ha="center", fontsize=6.5, color=MUTED)
# C: redaction collapse, held-out macro AUROC raw vs redacted (HTT and LLMs)
ax = fig.add_subplot(gs[0, 2])
L = S["llm"]
tc = {r["key"]: r for r in S1["tcga"]}
pairs = [("HTT, 128 tokens", tc["corr_raw_biomedbert"]["held_macro_mean"], tc["corr_redacted_biomedbert"]["held_macro_mean"]),
         ("HTT, 512 tokens", tc["corr_raw_biomedbert_512"]["held_macro_mean"], tc["corr_redacted_biomedbert_512"]["held_macro_mean"])]
for k, lab in [("q8", "Qwen3-8B"), ("q14", "Qwen3-14B"), ("q27", "Qwen3.6-27B"), ("med8", "II-Medical-8B"), ("oss20", "gpt-oss-20b")]:
    for mode, ml in [("logprob", ""), ("reason", ", reasoning")]:
        a, b = L.get(f"grade_{k}_{mode}_raw"), L.get(f"grade_{k}_{mode}_redacted")
        if a and b and a["parse_fail"] < 0.05 * a["n"] and b["parse_fail"] < 0.05 * b["n"]:
            pairs.append((lab + (" (R)" if ml else ""), a["held_macro"], b["held_macro"]))
for i, (lab, r, d) in enumerate(pairs):
    y = len(pairs) - 1 - i
    ax.plot([d, r], [y, y], color=GRID, lw=2.5, zorder=1)
    ax.scatter([r], [y], color=BLUE, s=24, zorder=3, label="Raw text" if i == 0 else None)
    ax.scatter([d], [y], color=ORANGE, s=24, zorder=3, label="Grade phrases masked" if i == 0 else None)
ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (2, 2)))
ax.set_yticks(range(len(pairs))); ax.set_yticklabels([p[0] for p in pairs][::-1]); ax.tick_params(axis="y", length=0)
ax.set_xlim(0.45, 1.0); ax.set_xlabel("Held-out-type macro AUROC")
ax.set_title("c  Grade transfer, label phrase masked", loc="left", fontsize=8)
ax.text(0.0, -0.33, "(R): with reasoning", transform=ax.transAxes, fontsize=6.3, color=MUTED)
ax.legend(loc="upper left", fontsize=6.5, bbox_to_anchor=(0.0, -0.2), ncol=2, handletextpad=0.2)
ax.set_ylim(-0.6, len(pairs) - 0.4)
save(fig, "Fig15_label_provenance")

# ------------------------------------------------------------------ Fig 14: LLMs on the growth benchmark
ce = S1["ceilings"]["test180"]
ST = S.get("llm_strict", {}); SC = S.get("llm_strict_contrasts", {})
MODS = [("q8", "Qwen3-8B"), ("q14", "Qwen3-14B"), ("q27", "Qwen3.6-27B"), ("med8", "II-Medical-8B"), ("oss20", "gpt-oss-20b")]
def okd(n):
    e = L.get(n); return e is not None and e["parse_fail"] <= 0.05 * e["n"]
def val(k, mode, v):
    if mode == "logprob":
        n = f"growth_{k}_logprob_{v}"
        return (L[n]["sets"]["test180"], None) if okd(n) else (None, None)
    n = f"growth_{k}_reasonS_{v}"
    return (ST[n]["auroc_completed"], ST[n]["completion"]) if n in ST else (None, None)
cols = [(k, m, f"{lab}, {'direct' if m == 'logprob' else 'reasoning'}") for k, lab in MODS for m in ("logprob", "reason")
        if val(k, m, "dates")[0] is not None or val(k, m, "nodates")[0] is not None]
fig = plt.figure(figsize=(7.2, 3.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.75, 1], wspace=0.42)
ax = fig.add_subplot(gs[0, 0])
VW = [("nodates", "No dates", AQUA, -0.16), ("dates", "True dates", BLUE, 0.0), ("redate21", "All visits 21 d apart", ORANGE, 0.16)]
seen = set()
for xi, (k, mode, lab) in enumerate(cols):
    ys = []
    for v, vl, c, dx in VW:
        y, comp = val(k, mode, v)
        if y is None:
            continue
        ys.append(y)
        ax.scatter([xi + dx], [y], color=c, s=22, zorder=3, edgecolors="white", linewidths=0.5,
                   label=None if vl in seen else vl); seen.add(vl)
        if comp is not None and comp < 0.95:
            ax.text(xi + dx, y - 0.035, f"{comp*100:.0f}%", fontsize=5, color=MUTED, ha="center")
    if ys:
        ax.plot([xi, xi], [min(ys), max(ys)], color=GRID, lw=1.4, zorder=1)
ax.scatter([], [], color=INK2, s=22, label="Bayes reader, 21 d apart")
ax.axhline(ce["nodates"], color=AQUA, lw=0.8, ls=(0, (3, 2)))
ax.axhline(ce["dates"], color=BLUE, lw=0.8, ls=(0, (3, 2)))
if "ceil_redate" in S:
    ax.axhline(S["ceil_redate"]["redate21"], color=ORANGE, lw=0.8, ls=(0, (1, 2)))
ax.axhline(0.5, color=MUTED, lw=0.6)
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c[2] for c in cols], fontsize=6.2, rotation=35, ha="right", rotation_mode="anchor")
ax.set_ylabel("AUROC, growth benchmark test180"); ax.set_ylim(0.2, 1.0)
ax.set_title("a  Zero-shot language models", loc="left", fontsize=8)
h, l = ax.get_legend_handles_labels()
ax.legend(h[:3], l[:3], fontsize=6.3, loc="upper left", bbox_to_anchor=(0.0, -0.45), ncol=3, handletextpad=0.2, columnspacing=1.0)
ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
ax = fig.add_subplot(gs[0, 1])
bars = [("Bayes reader", 0.324, INK2)]
g = {r["key"]: r for r in S1["growth"]}
bars += [("HTT-IA (trained)", g["clin_interval"]["shift"][0], BLUE), ("Initial HTT (trained)", g["orig_elapsed"]["shift"][0], MUTED)]
sh = S["llm_redate_shift"]
for k, lab in MODS:
    if f"growth_{k}_logprob" in sh and okd(f"growth_{k}_logprob_redate21"):
        bars.append((f"{lab}, direct", sh[f"growth_{k}_logprob"], MUTED))
    c = SC.get(f"{k}:redate21-redate90")
    if c:
        bars.append((f"{lab}, reasoning", c["shift"], ORANGE))
for i_, (lab, v, c) in enumerate(bars):
    ax.barh(len(bars) - 1 - i_, v, color=c, height=0.62)
    ax.text(v + 0.008, len(bars) - 1 - i_, f"{v:.3f}", va="center", fontsize=6.3, color=INK2)
ax.set_yticks(range(len(bars))); ax.set_yticklabels([b[0] for b in bars][::-1], fontsize=6.3); ax.tick_params(axis="y", length=0)
ax.axvline(0.324, color=INK2, lw=0.6, ls=(0, (2, 2)))
ax.set_xlabel("Re-dating shift |P(21 d) − P(90 d)|"); ax.set_xlim(0, 0.62)
ax.set_title("b  Response to visit spacing", loc="left", fontsize=8)
save(fig, "Fig14_llm_growth")
print("figures written")
