"""Statistics used throughout (Section 2.11 of the manuscript): mean +- SD over seeds; paired
patient bootstrap (2,000 resamples) of the mean difference over seeds; Wilcoxon across seeds."""
import glob
import json
import os
import numpy as np
from scipy.stats import wilcoxon, rankdata


def fast_auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(p)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def load(dirname, name):
    """All seeds of one condition -> {seed: result}"""
    out = {}
    for f in glob.glob(os.path.join(dirname, f"{name}__s*.json")):
        s = int(f.rsplit("__s", 1)[1][:-5])
        out[s] = json.load(open(f))
    return out


def matrix(runs, setname, key="p"):
    """-> pids, y, P (seeds x patients) aligned on patient id, seeds sorted."""
    seeds = sorted(runs)
    ref = runs[seeds[0]]["sets"][setname]
    pids = ref["pid"]; y = np.array(ref["y"])
    P = []
    for s in seeds:
        d = runs[s]["sets"][setname]
        m = dict(zip(d["pid"], d[key]))
        P.append([m[i] for i in pids])
    return seeds, pids, y, np.array(P)


def summary(runs, setname):
    _, _, y, P = matrix(runs, setname)
    a = np.array([fast_auc(y, p) for p in P])
    return float(np.nanmean(a)), float(np.nanstd(a)), a


def boot_idx(y, rng, stratified):
    n = len(y)
    if not stratified:
        return rng.integers(0, n, n)
    i1 = np.where(y == 1)[0]; i0 = np.where(y == 0)[0]
    return np.concatenate([rng.choice(i1, len(i1)), rng.choice(i0, len(i0))])


def mean_ci(runs, setname, B=2000, seed=0, stratified=False):
    _, _, y, P = matrix(runs, setname)
    rng = np.random.default_rng(seed)
    point = np.nanmean([fast_auc(y, p) for p in P])
    bs = []
    for _ in range(B):
        i = boot_idx(y, rng, stratified)
        bs.append(np.nanmean([fast_auc(y[i], p[i]) for p in P]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def paired_diff(ra, rb, setname, B=2000, seed=0, stratified=False):
    """Mean over shared seeds of AUROC(A)-AUROC(B), patient bootstrap CI, seeds A>B, Wilcoxon p."""
    shared = sorted(set(ra) & set(rb))
    ra = {s: ra[s] for s in shared}; rb = {s: rb[s] for s in shared}
    _, pa, y, PA = matrix(ra, setname)
    _, pb, yb, PB0 = matrix(rb, setname)
    m = {pid: j for j, pid in enumerate(pb)}
    PB = PB0[:, [m[i] for i in pa]]
    da = np.array([fast_auc(y, p) for p in PA]); db = np.array([fast_auc(y, p) for p in PB])
    d = da - db
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(B):
        i = boot_idx(y, rng, stratified)
        bs.append(np.nanmean([fast_auc(y[i], a[i]) - fast_auc(y[i], b[i]) for a, b in zip(PA, PB)]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    try:
        p = float(wilcoxon(d).pvalue) if np.any(d != 0) and len(d) >= 5 else float("nan")
    except ValueError:
        p = float("nan")
    return dict(diff=float(np.nanmean(d)), lo=float(lo), hi=float(hi), better=int((d > 0).sum()),
                n=len(d), wilcoxon_p=p)


def redate_shift(runs, g1="21", g2="90"):
    """Mean over models of mean |P(g1) - P(g2)| on test180, with SD over models."""
    v = [np.mean(np.abs(np.array(r["redate"][g1]) - np.array(r["redate"][g2]))) for r in runs.values()]
    return float(np.mean(v)), float(np.std(v))
