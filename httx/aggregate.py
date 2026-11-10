"""Aggregate all new runs into tables (JSON/CSV) and figures (Fig. 11, Fig. 12, Fig. S1).
Usage: python httx/aggregate.py RESULTS_DIR"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from httx import stats as St

R = sys.argv[1]
FIG = os.path.join(R, "figures"); os.makedirs(FIG, exist_ok=True)
OUT = {}

# palette: categorical slots 1-3 of the reference palette; neutrals for references/ceilings
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8.5,
                     "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False,
                     "savefig.dpi": 300})


def fmt(m, s=None, d=3):
    return f"{m:.{d}f}" if s is None else f"{m:.{d}f} ± {s:.{d}f}"


def fci(x, d=3):
    return f"{x['diff']:+.{d}f} [{x['lo']:+.{d}f}, {x['hi']:+.{d}f}]"


# --------------------------------------------------------------------------------------
# Growth benchmark
# --------------------------------------------------------------------------------------
CEIL = json.load(open(os.path.join(R, "ceilings", "ceilings_sets.json")))
POP = json.load(open(os.path.join(R, "ceilings", "ceilings_pop.json")))
G = os.path.join(R, "growth")
LABEL = {
    "baseline": "Baseline (last report)", "none": "HTT, no time input", "order": "HTT, order only",
    "orig_elapsed": "HTT, original band, elapsed (initial HTT)",
    "orig_interval": "HTT, original band, interval",
    "clin_elapsed": "HTT, clinical band, elapsed",
    "clin_interval": "HTT, clinical band, interval (HTT-IA)",
    "clin_both": "HTT, clinical band, elapsed + interval",
    "none_num": "HTT + parsed sizes, no time",
    "orig_elapsed_num": "HTT + parsed sizes, initial code",
    "clin_interval_num": "HTT-IA + parsed sizes",
    "struct_none": "Structured reader, no time",
    "struct_orig_elapsed": "Structured reader, initial code",
    "struct_clin_interval": "Structured reader, interval code",
}
runs = {k: St.load(G, k) for k in LABEL}
runs = {k: v for k, v in runs.items() if v}
SETS = ["val", "test60", "test120", "test180", "PANC100", "PANC300"]
rows = []
for k, r in runs.items():
    row = dict(key=k, model=LABEL[k], n_seeds=len(r))
    for s in SETS:
        m, sd, _ = St.summary(r, s)
        row[s] = fmt(m, sd); row[s + "_mean"] = m
    if "orig_elapsed" in runs and k != "orig_elapsed":
        row["d_vs_initial"] = St.paired_diff(r, runs["orig_elapsed"], "test180")
    if "clin_interval" in runs and k != "clin_interval":
        row["d_vs_IA"] = St.paired_diff(r, runs["clin_interval"], "test180")
    if "baseline" in runs and k != "baseline":
        row["d_vs_base"] = St.paired_diff(r, runs["baseline"], "test180")
    if k not in ("baseline", "none", "none_num", "struct_none"):
        row["shift"] = St.redate_shift(r)
    if "freq_scale_mean" in next(iter(r.values())):
        row["freq_scale"] = float(np.mean([x["freq_scale_mean"] for x in r.values()]))
    row["best_epoch"] = float(np.mean([x["fit"]["best_epoch"] for x in r.values()]))
    rows.append(row)
OUT["growth"] = rows
OUT["ceilings"] = CEIL; OUT["ceilings_pop"] = POP
gt = {r["key"]: r for r in rows}

# ---- Fig. 11
fig = plt.figure(figsize=(7.2, 6.4))
gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.55)

# (a) the ladder: where the gap to the ceiling goes (test180)
ax = fig.add_subplot(gs[0, 0])
ladder = [k for k in ["struct_clin_interval", "clin_interval_num", "orig_elapsed_num", "clin_interval",
                      "orig_interval", "clin_elapsed", "orig_elapsed", "none", "baseline"] if k in gt]
ys = np.arange(len(ladder))[::-1]
for y, k in zip(ys, ladder):
    _, _, a = St.summary(runs[k], "test180")
    col = AQUA if k.startswith("struct") else (ORANGE if k.endswith("_num") else BLUE)
    ax.scatter(a, np.full(len(a), y) + np.random.default_rng(0).uniform(-0.18, 0.18, len(a)), s=8,
               color=col, alpha=0.45, linewidths=0)
    ax.plot([a.mean()] * 2, [y - 0.32, y + 0.32], color=col, lw=2, solid_capstyle="round")
ax.set_yticks(ys); ax.set_yticklabels([LABEL[k].replace("HTT, ", "").replace(" (initial HTT)", "\n(initial HTT)") for k in ladder], fontsize=6.5)
for v, lab, ls in [(CEIL["test180"]["last"], "last report", ":"), (CEIL["test180"]["nodates"], "no dates", "--"),
                   (CEIL["test180"]["dates"], "with dates", "-")]:
    ax.axvline(v, color=MUTED, lw=0.9, ls=ls, zorder=0)
    ax.text(v, 1.01, lab, rotation=40, va="bottom", ha="left", fontsize=5.8, color=INK2,
            transform=matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes))
ax.set_xlabel("test180 AUROC (10 seeds)")
ax.set_title("a  Where the gap to the ceiling goes", loc="left", pad=26)
ax.set_xlim(0.72, 1.005)

# (b) 2x2 factorial: frequency band x time input
ax = fig.add_subplot(gs[0, 1])
cells = [("orig_elapsed", "orig_interval"), ("clin_elapsed", "clin_interval")]
for i, (ke, ki) in enumerate(cells):
    xs = [0, 1]
    ms = [gt[ke]["test180_mean"] if ke in gt else np.nan, gt[ki]["test180_mean"] if ki in gt else np.nan]
    col = [MUTED, BLUE][i]
    ax.plot(xs, ms, "-o", color=col, lw=2, ms=6, markeredgecolor="white", markeredgewidth=1.5)
    ax.text(1.08, ms[1], ["6.3–126 y band", "7 d–5 y band"][i], va="center", ha="left", fontsize=6.5, color=INK2)
if "none" in gt:
    ax.axhline(gt["none"]["test180_mean"], color=MUTED, ls=":", lw=0.9)
    ax.text(-0.3, gt["none"]["test180_mean"] + 0.0006, "no time", va="bottom", ha="left", fontsize=6, color=INK2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["elapsed", "interval"]); ax.set_xlim(-0.35, 1.75)
ax.set_xlabel("time input"); ax.set_ylabel("test180 AUROC (mean of 10 seeds)")
ax.set_title("b  Frequency band × time input", loc="left")

# (c) re-dating shift
ax = fig.add_subplot(gs[1, 0])
sk = [k for k in ["orig_elapsed", "orig_interval", "clin_elapsed", "clin_interval", "clin_interval_num",
                  "struct_clin_interval"] if k in gt and "shift" in gt[k]]
vals = [gt[k]["shift"][0] for k in sk]; errs = [gt[k]["shift"][1] for k in sk]
cols = [AQUA if k.startswith("struct") else (ORANGE if k.endswith("_num") else BLUE) for k in sk]
ax.bar(range(len(sk)), vals, yerr=errs, color=cols, width=0.62, error_kw=dict(lw=0.8, ecolor=INK2, capsize=2))
for i, (v, e) in enumerate(zip(vals, errs)):
    ax.text(i, v + e + 0.008, f"{v:.2f}", ha="center", fontsize=6, color=INK2)
ax.axhline(CEIL["redate_shift_bayes_21_90"], color=INK2, ls="--", lw=0.9)
ax.text(len(sk) - 0.5, CEIL["redate_shift_bayes_21_90"] + 0.01, "Bayes reader", ha="right", fontsize=6, color=INK2)
short = {"orig_elapsed": "orig.\nelapsed", "orig_interval": "orig.\ninterval", "clin_elapsed": "clin.\nelapsed",
         "clin_interval": "HTT-IA", "clin_interval_num": "HTT-IA\n+sizes", "struct_clin_interval": "struct.\ninterval"}
ax.set_xticks(range(len(sk))); ax.set_xticklabels([short[k] for k in sk], fontsize=5.6)
ax.set_ylabel("mean |P(21 d) − P(90 d)|")
ax.set_title("c  Response to re-dating", loc="left")

# (d) scaling
ax = fig.add_subplot(gs[1, 1])
S = os.path.join(R, "scale")
scale_rows = []; ends = []
for k, col, lab in [("orig_elapsed", MUTED, "initial HTT"), ("clin_interval", BLUE, "HTT-IA"),
                    ("clin_interval_num", ORANGE, "HTT-IA + sizes"), ("struct_clin_interval", AQUA, "structured"),
                    ("none", INK2, "no time")]:
    xs, ms, sds = [], [], []
    for ntr in [70, 280, 700, 1750]:
        rr = runs.get(k) if ntr == 70 else St.load(S, f"{k}_n{ntr}")
        if ntr == 70 and rr:
            rr = {s: v for s, v in rr.items() if s in (42, 1, 2, 3, 4)}
        if rr:
            m, sd, _ = St.summary(rr, "test180")
            xs.append(4 * ntr); ms.append(m); sds.append(sd)
            scale_rows.append(dict(model=lab, n_train=4 * ntr, seeds=len(rr), test180=m, sd=sd,
                                   shift=St.redate_shift(rr)[0] if k != "none" else None))
    if xs:
        ax.errorbar(xs, ms, yerr=sds, color=col, lw=1.6 if col != INK2 else 1.0, marker="o", ms=4,
                    ls="-" if col != INK2 else ":", capsize=0)
        ends.append([ms[-1], xs[-1] * 1.12, lab])
ends.sort()
lo_ = ax.get_ylim()
gap = 0.008
for i in range(1, len(ends)):
    ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
for y, x, lab in ends:
    ax.text(x, y, lab, va="center", fontsize=6.2, color=INK2)
ax.axhline(CEIL["test180"]["dates"], color=MUTED, lw=0.9); ax.axhline(CEIL["test180"]["nodates"], color=MUTED, lw=0.9, ls="--")
ax.set_xscale("log"); ax.set_xlim(200, 7000 * 3.2)
ax.set_xticks([280, 1120, 2800, 7000]); ax.set_xticklabels(["280", "1,120", "2,800", "7,000"])
ax.set_xlabel("training patients"); ax.set_ylabel("test180 AUROC (mean ± SD, 5 seeds)")
ax.set_title("d  More training data", loc="left")
OUT["scaling"] = scale_rows
fig.savefig(os.path.join(FIG, "Fig11_growth_gap.png"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "Fig11_growth_gap.pdf"), bbox_inches="tight")
plt.close(fig)

# long runs (20 epochs) and phrase benchmark
L = os.path.join(R, "long"); OUT["long"] = []
lr = {k: St.load(L, f"{k}_ep20") for k in ["orig_elapsed", "none", "clin_interval", "baseline"]}
for k, r in lr.items():
    if r:
        m, sd, _ = St.summary(r, "test180")
        row = dict(key=k, test180=fmt(m, sd), mean=m)
        if k != "clin_interval" and lr.get("clin_interval"):
            row["IA_minus_this"] = St.paired_diff(lr["clin_interval"], r, "test180")
        if k not in ("none", "baseline"):
            row["shift"] = St.redate_shift(r)
        OUT["long"].append(row)
P = os.path.join(R, "phrase"); OUT["phrase"] = []
pr = {k: St.load(P, k) for k in ["baseline", "orig_elapsed", "clin_interval"]}
for k, r in pr.items():
    if not r:
        continue
    row = dict(key=k)
    for s in SETS:
        m, sd, _ = St.summary(r, s); row[s] = fmt(m, sd)
    row["test180_per_seed"] = [float(x) for x in St.summary(r, "test180")[2]]
    if k == "clin_interval" and pr.get("orig_elapsed"):
        for s in ["test180", "PANC300", "test60"]:
            row[f"IA_minus_initial_{s}"] = St.paired_diff(r, pr["orig_elapsed"], s)
    if k != "baseline" and pr.get("baseline"):
        for s in ["test180", "PANC300"]:
            row[f"minus_base_{s}"] = St.paired_diff(r, pr["baseline"], s)
    if k != "baseline":
        row["shift"] = St.redate_shift(r)
    OUT["phrase"].append(row)

# --------------------------------------------------------------------------------------
# TCGA
# --------------------------------------------------------------------------------------
T = os.path.join(R, "tcga")
TF = json.load(open(os.path.join(R, "tfidf", "tfidf.json")))
AUD = json.load(open(os.path.join(R, "tcga_audit.json")))
OUT["audit"] = AUD
HELD = ["THCA", "SARC", "LUSC"]


def within_type_macro(sets_entry):
    """Macro AUROC over types inside the in-distribution test set (types with both classes)."""
    df = pd.DataFrame(dict(y=sets_entry["y"], p=sets_entry["p"], t=sets_entry["ctype"]))
    a = [St.fast_auc(g.y.values, g.p.values) for _, g in df.groupby("t")]
    a = [x for x in a if x == x]
    return float(np.mean(a))


def tcga_row(name, r, label):
    row = dict(key=name, model=label, n_seeds=len(r))
    seeds = sorted(r)
    row["test_pooled"] = fmt(*St.summary(r, "test")[:2])
    wm = [within_type_macro(r[s]["sets"]["test"]) for s in seeds]
    row["test_within_type_macro"] = fmt(np.mean(wm), np.std(wm)); row["test_wtm_mean"] = float(np.mean(wm))
    macro = []
    for t in HELD:
        key = f"held_{t}"
        if key in r[seeds[0]]["sets"]:
            m, lo, hi = St.mean_ci(r, key, B=1000, stratified=True)
            row[t] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"; row[t + "_mean"] = m
            macro.append(np.array(St.summary(r, key)[2]))
    mac = np.mean(np.vstack(macro), 0)
    row["held_macro"] = fmt(mac.mean(), mac.std()); row["held_macro_mean"] = float(mac.mean())
    row["n"] = r[seeds[0]]["n"]
    return row


TCOND = [("ref_paper", "Paper labels, regex types (reproduction)"),
         ("paperlab_truetype", "Paper labels, true types"),
         ("corr_raw_biomedbert", "Corrected; BiomedBERT"), ("corr_raw_bert", "Corrected; general BERT"),
         ("corr_raw_random", "Corrected; random init"),
         ("corr_redacted_biomedbert", "Corrected, redacted; BiomedBERT"),
         ("corr_redacted_bert", "Corrected, redacted; general BERT"),
         ("corr_redacted_random", "Corrected, redacted; random init"),
         ("corr_raw_biomedbert_512", "Corrected; BiomedBERT, 512 tokens"),
         ("corr_redacted_biomedbert_512", "Corrected, redacted; BiomedBERT, 512 tokens")]
truns = {k: St.load(T, k) for k, _ in TCOND}
OUT["tcga"] = [tcga_row(k, truns[k], lab) for k, lab in TCOND if truns[k]]
# key paired contrasts on held-out types (class-stratified bootstrap)
contr = {}
pairs = [("corr_raw_biomedbert", "corr_redacted_biomedbert"), ("corr_redacted_biomedbert", "corr_redacted_random"),
         ("corr_redacted_biomedbert", "corr_redacted_bert"), ("corr_raw_biomedbert", "corr_raw_random"),
         ("corr_raw_biomedbert_512", "corr_raw_biomedbert"), ("corr_redacted_biomedbert_512", "corr_redacted_biomedbert")]
for a, b in pairs:
    if truns.get(a) and truns.get(b):
        contr[f"{a} - {b}"] = {t: St.paired_diff(truns[a], truns[b], f"held_{t}", B=1000, stratified=True) for t in HELD}
OUT["tcga_contrasts"] = contr
# TF-IDF rows
OUT["tfidf"] = {k: {s: v["auroc"] for s, v in d["sets"].items()} for k, d in TF.items()}

# LOTO
LO = os.path.join(R, "loto"); LTYPES = ["BRCA", "COAD", "HNSC", "LIHC", "LUAD", "SARC", "UCEC", "LUSC"]
loto = []
for t in LTYPES:
    row = dict(type=t)
    for txt in ["raw", "redacted"]:
        r = St.load(LO, f"loto_{t}_{txt}")
        if r:
            m, lo, hi = St.mean_ci(r, f"held_{t}", B=1000, stratified=True)
            row[f"htt_{txt}"] = m; row[f"htt_{txt}_ci"] = (lo, hi); row["seeds"] = len(r)
            row["n"] = r[sorted(r)[0]]["n"]["held"][t]
        row[f"tfidf_{txt}"] = TF[f"loto_{t}_{txt}"]["sets"][f"held_{t}"]["auroc"]
    loto.append(row)
OUT["loto"] = loto

# ---- Fig. 12
fig = plt.figure(figsize=(7.2, 6.6))
gs = fig.add_gridspec(2, 2, hspace=0.95, wspace=0.6)
# (a) audit of the published labels
ax = fig.add_subplot(gs[0, 0])
lo_ = AUD["paper_low_relabelled_by_corrected"]; hi_ = AUD["paper_high_relabelled_by_corrected"]
cats = ["low", "unlabelled", "high"]; cc = [BLUE, GRID, ORANGE]
for i, (name, d) in enumerate([("published 'low'\n(n = %d)" % sum(lo_.values()), lo_),
                               ("published 'high'\n(n = %d)" % sum(hi_.values()), hi_)]):
    left = 0; tot = sum(d.values())
    for c, col in zip(cats, cc):
        w = d.get(c, 0) / tot
        ax.barh(i, w, left=left, color=col, height=0.55, edgecolor="white", linewidth=2)
        if w > 0.12:
            ax.text(left + w / 2, i, f"{c}\n{d.get(c, 0)}", ha="center", va="center", fontsize=6,
                    color="white" if col != GRID else INK)
        left += w
ax.set_yticks([0, 1]); ax.set_yticklabels([n for n, _ in [("published 'low'\n(n = %d)" % sum(lo_.values()), 0),
                                                         ("published 'high'\n(n = %d)" % sum(hi_.values()), 0)]], fontsize=6.5)
ax.set_xlim(0, 1); ax.set_xlabel("share of reports, by corrected label")
ax.set_title("a  Published labels under the corrected rule", loc="left")
pass

# (b) held-out types: raw vs redacted, encoders
ax = fig.add_subplot(gs[0, 1])
enc = [("biomedbert", "BiomedBERT", BLUE), ("bert", "general BERT", ORANGE), ("random", "random init", AQUA)]
xpos = {t: i for i, t in enumerate(HELD)}
for j, (e, lab, col) in enumerate(enc):
    for txt, mk in [("raw", "o"), ("redacted", "s")]:
        r = truns.get(f"corr_{txt}_{e}")
        if not r:
            continue
        for t in HELD:
            _, _, a = St.summary(r, f"held_{t}")
            x = xpos[t] + (j - 1) * 0.22 + (0.06 if txt == "redacted" else -0.06)
            ax.scatter([x], [a.mean()], marker=mk, s=26, color=col if txt == "raw" else "white",
                       edgecolors=col, linewidths=1.4, zorder=3)
    ax.plot([], [], "o", color=col, label=lab, ms=5)
for t in HELD:
    ax.plot([xpos[t] - 0.42, xpos[t] + 0.42], [TF["main_raw"]["sets"][f"held_{t}"]["auroc"]] * 2, color=MUTED, lw=1)
    ax.plot([xpos[t] - 0.42, xpos[t] + 0.42], [TF["main_redacted"]["sets"][f"held_{t}"]["auroc"]] * 2, color=MUTED, lw=1, ls="--")
ax.plot([], [], "o", color=INK2, label="raw text", ms=5); ax.plot([], [], "s", mfc="white", mec=INK2, label="grade phrases redacted", ms=5)
ax.plot([], [], color=MUTED, label="TF-IDF raw"); ax.plot([], [], color=MUTED, ls="--", label="TF-IDF redacted")
ax.axhline(0.5, color=GRID, lw=0.8, zorder=0)
ax.set_xticks(range(3)); ax.set_xticklabels(HELD); ax.set_ylim(0.2, 1.02)
ax.set_ylabel("held-out AUROC (mean of 5 seeds)")
ax.set_title("b  Held-out types: what survives redaction", loc="left")
ax.legend(fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, handlelength=1.4)

# (c) LOTO raw vs redacted
ax = fig.add_subplot(gs[1, :])
xs = np.arange(len(loto))
for off, key, col, lab in [(-0.2, "raw", BLUE, "HTT, raw"), (0.2, "redacted", BLUE, "HTT, redacted")]:
    v = [r.get(f"htt_{key}", np.nan) for r in loto]
    lo = [r.get(f"htt_{key}_ci", (np.nan, np.nan))[0] for r in loto]; hi = [r.get(f"htt_{key}_ci", (np.nan, np.nan))[1] for r in loto]
    ax.errorbar(xs + off, v, yerr=[np.array(v) - np.array(lo), np.array(hi) - np.array(v)], fmt="o" if key == "raw" else "s",
                color=col, mfc=col if key == "raw" else "white", ms=5, lw=0.9, capsize=0, label=lab)
    tv = [r[f"tfidf_{key}"] for r in loto]
    ax.scatter(xs + off, tv, marker="_", s=140, color=MUTED, linewidths=1.8, zorder=4,
               label="TF-IDF (same position)" if key == "raw" else None)
ax.axhline(0.5, color=GRID, lw=0.8, zorder=0)
ax.set_xticks(xs); ax.set_xticklabels([f"{r['type']}\n(n = {r.get('n', '')})" for r in loto], fontsize=6.5)
ax.set_ylabel("AUROC on the withheld type\n(mean of 3 seeds, 95% CI)"); ax.set_ylim(0.15, 1.03)
ax.set_title("c  Leave-one-type-out over eight evaluable types (left of each pair: raw text; right: redacted)", loc="left")
ax.legend(fontsize=6, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.2))
fig.savefig(os.path.join(FIG, "Fig12_tcga_audit.png"), bbox_inches="tight")
fig.savefig(os.path.join(FIG, "Fig12_tcga_audit.pdf"), bbox_inches="tight")
plt.close(fig)

json.dump(OUT, open(os.path.join(R, "summary.json"), "w"), indent=1, default=float)
print("ok", {k: len(v) if hasattr(v, "__len__") else v for k, v in OUT.items()})
