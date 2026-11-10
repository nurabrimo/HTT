"""Second revision: aggregate E15-E20 (registry audit, label-source runs, survival, GENIE BPC timelines,
MSK-CHORD vs curation, open-weight LLMs). Runs on DCC; writes only aggregate numbers and figures
(no patient-level GENIE data leave the cluster).  Usage: aggregate2.py <runs_dir> <data_dir> <out_dir>"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from httx.stats import fast_auc

R, DATA, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)
B = 2000
S = {}


def J(p):
    return json.load(open(p))


# ---------------------------------------------------------------- E15 registry audit
S["cdr_audit"] = {k: v for k, v in J(os.path.join(DATA, "cdr_audit.json")).items()}

# ---------------------------------------------------------------- E16 label source (train corr/reg, score both)
rows = {}
for name in ["labcorr_raw", "labcorr_redacted", "labreg_raw", "labreg_redacted", "labcorrmatched_raw", "labcorrmatched_redacted"]:
    runs = [J(f) for f in sorted(glob.glob(os.path.join(R, "tcga_reg", f"{name}__s*.json")))]
    if not runs:
        continue
    t = [r["sets"]["test"] for r in runs]
    # common evaluation set: test patients with BOTH a corrected-rule and a registry label
    both = []
    for v in t:
        y_tr = np.array(v["y"]); y_alt = np.array([np.nan if a is None else a for a in v["y_alt"]], float)
        m = ~np.isnan(y_alt)
        y_corr, y_reg = (y_tr, y_alt) if name.startswith("labcorr") else (y_alt, y_tr)
        p = np.array(v["p"])
        both.append((fast_auc(y_corr[m], p[m]), fast_auc(y_reg[m], p[m]), int(m.sum())))
    both = np.array(both)
    rows[name] = dict(seeds=len(runs), auroc_own=float(np.mean([v["auroc"] for v in t])),
                      auroc_own_sd=float(np.std([v["auroc"] for v in t])),
                      common_vs_corr=float(np.mean(both[:, 0])), common_vs_reg=float(np.mean(both[:, 1])),
                      common_n=int(both[0, 2]),
                      held_macro=float(np.nanmean([r.get("held_macro", np.nan) for r in runs])))
S["label_source"] = rows

# ---------------------------------------------------------------- E17 survival
surv = {}
for f in sorted(glob.glob(os.path.join(R, "surv", "*.json"))):
    r = J(f)
    surv[os.path.basename(f)[:-5]] = dict(n=r["n"], events=r["events"],
                                          models={k: dict(c=v["c"], c_strat=v["c_strat"], sd=v["c_strat_sd"])
                                                  for k, v in r["models"].items()},
                                          deltas={k: v for k, v in r.items() if k.startswith("delta")})
S["survival"] = surv

# ---------------------------------------------------------------- E18 GENIE timelines
CENT = ["MSK", "DFCI", "VICC", "UHN"]
G = defaultdict(lambda: defaultdict(dict))        # (task,name) -> center -> seed -> sets
for f in glob.glob(os.path.join(R, "genie", "*.json")):
    r = J(f); a = r["config"]
    G[(a["task"], a["name"])][a["test_center"]][a["seed"]] = r["sets"]


def center_pred(task, name, c, setname="test"):
    d = G[(task, name)][c]
    seeds = sorted(d)
    ref = d[seeds[0]][setname]
    pids, y = ref["pid"], np.array(ref["y"])
    P = np.array([[dict(zip(d[s][setname]["pid"], d[s][setname]["p"]))[i] for i in pids] for s in seeds])
    return pids, y, P


def genie_point(task, name, setname="test"):
    per = {}
    for c in CENT:
        if c in G[(task, name)]:
            _, y, P = center_pred(task, name, c, setname)
            per[c] = float(np.mean([fast_auc(y, p) for p in P]))
    return float(np.mean(list(per.values()))), per


def genie_contrast(task, n1, n0, setname1="test", setname0="test"):
    """Mean over centres of the seed-averaged AUROC difference; centre-stratified patient bootstrap."""
    data = []
    for c in CENT:
        pids, y, P1 = center_pred(task, n1, c, setname1)
        p0ids, y0, P0 = center_pred(task, n0, c, setname0)
        m0 = {i: k for k, i in enumerate(p0ids)}
        P0 = P0[:, [m0[i] for i in pids]]
        data.append((y, P1, P0))
    def stat(idx_list):
        return np.mean([np.mean([fast_auc(y[i], p[i]) for p in P1]) - np.mean([fast_auc(y[i], p[i]) for p in P0])
                        for (y, P1, P0), i in zip(data, idx_list)])
    point = stat([np.arange(len(d[0])) for d in data])
    bs = [stat([rng.integers(0, len(d[0]), len(d[0])) for d in data]) for _ in range(500)]
    return dict(diff=float(point), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)),
                better_centres=int(sum(1 for (y, P1, P0) in data
                                       if np.mean([fast_auc(y, p) for p in P1]) > np.mean([fast_auc(y, p) for p in P0]))))


NAMES = ["lr_notime", "lr_time", "gbm_notime", "gbm_time", "htt_none", "htt_order", "htt_orig_elapsed", "htt_clin_elapsed",
         "htt_clin_interval", "htt_clin_both"]
gen = {}
for task in ["nextprog", "mort12"]:
    t = {}
    for n in NAMES:
        if (task, n) not in G:
            continue
        m, per = genie_point(task, n)
        e = dict(auroc=m, per_centre=per)
        for pr in ["shuffle", "nogap"]:
            e[pr] = genie_point(task, n, pr)[0]
            e[pr + "_shift"] = float(np.mean([np.mean([G[(task, n)][c][s][pr]["mean_abs_shift"] for s in G[(task, n)][c]])
                                              for c in CENT]))
        t[n] = e
    con = {}
    for n1, n0 in [("htt_order", "htt_none"), ("htt_orig_elapsed", "htt_none"), ("htt_clin_interval", "htt_none"),
                   ("htt_orig_elapsed", "htt_clin_interval"), ("htt_orig_elapsed", "lr_time"), ("lr_time", "lr_notime"),
                   ("gbm_time", "gbm_notime"), ("htt_orig_elapsed", "gbm_time")]:
        if (task, n1) not in G or (task, n0) not in G:
            continue
        con[f"{n1} - {n0}"] = genie_contrast(task, n1, n0)
    for n in ["lr_time", "htt_orig_elapsed", "htt_clin_interval"]:
        con[f"{n}: shuffle - intact"] = genie_contrast(task, n, n, "shuffle", "test")
    t["contrasts"] = con
    t["n"] = {c: len(center_pred(task, "htt_none", c)[1]) for c in CENT}
    t["pos"] = {c: float(center_pred(task, "htt_none", c)[1].mean()) for c in CENT}
    gen[task] = t
S["genie"] = gen
S["genie_summary"] = J(os.path.join(DATA, "genie_bpc", "genie_summary.json"))
S["chord_vs_genie"] = {k: v for k, v in J(os.path.join(DATA, "genie_bpc", "chord_vs_genie.json")).items()
                       if k != "crosstab_curated_vs_nlp"}
S["chord_vs_genie_crosstab"] = J(os.path.join(DATA, "genie_bpc", "chord_vs_genie.json"))["crosstab_curated_vs_nlp"]

# ---------------------------------------------------------------- E19 LLMs
llm = {}
for f in sorted(glob.glob(os.path.join(R, "llm", "*.json"))):
    r = J(f); n = os.path.basename(f)[:-5]
    e = dict(parse_fail=r["parse_fail"], n=r["n"], sets={k: v["auroc"] for k, v in r["sets"].items()})
    if "held_macro" in r:
        e["held_macro"] = r["held_macro"]
    if "reason" in n:
        e["median_gen_tokens"] = float(np.median([x for v in r["sets"].values() for x in v.get("n_gen_tokens", [0])]))
    llm[n] = e
# re-dating shift for LLM growth readers: mean |P(21 d) - P(90 d)| on test180
shift = {}
for f in glob.glob(os.path.join(R, "llm", "growth_*_redate21.json")):
    g = f.replace("redate21", "redate90")
    if os.path.exists(g):
        a, b = J(f)["sets"]["test180"], J(g)["sets"]["test180"]
        mb = dict(zip(b["pid"], b["p"]))          # probability shift (bounded), as for HTT
        shift[os.path.basename(f).replace("_redate21.json", "")] = float(np.mean([abs(p - mb[i]) for i, p in zip(a["pid"], a["p"])]))
S["llm"] = llm
S["llm_redate_shift"] = shift
# dates vs no-dates contrast (paired bootstrap on test180)
dvn = {}
for f in glob.glob(os.path.join(R, "llm", "growth_*_dates.json")):
    g = f.replace("_dates.json", "_nodates.json")
    if not os.path.exists(g):
        continue
    a, b = J(f)["sets"]["test180"], J(g)["sets"]["test180"]
    mb = dict(zip(b["pid"], b["score"])); y = np.array(a["y"]); pa = np.array(a["score"]); pb = np.array([mb[i] for i in a["pid"]])
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(y), len(y))
        bs.append(fast_auc(y[i], pa[i]) - fast_auc(y[i], pb[i]))
    dvn[os.path.basename(f).replace("_dates.json", "")] = dict(diff=float(fast_auc(y, pa) - fast_auc(y, pb)),
                                                               lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)))
S["llm_dates_minus_nodates"] = dvn

json.dump(S, open(os.path.join(OUT, "summary2.json"), "w"), indent=1, default=float)
print(json.dumps({k: (v if k in ("label_source", "llm_redate_shift", "llm_dates_minus_nodates") else "...") for k, v in S.items()},
                 indent=1, default=float)[:4000])
for task in ["nextprog", "mort12"]:
    print("==", task)
    for n, e in S["genie"][task].items():
        if n in ("contrasts", "n", "pos"):
            continue
        print(f"  {n:18s} {e['auroc']:.3f} shuffle {e['shuffle']:.3f} nogap {e['nogap']:.3f} shift {e['shuffle_shift']:.3f}/{e['nogap_shift']:.3f}")
    for k, v in S["genie"][task]["contrasts"].items():
        print(f"  {k:40s} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] better {v['better_centres']}/4")

# ---------------------------------------------------------------- review additions
def holm(pv):
    """Holm-adjusted p-values for a dict of p-values."""
    ks = sorted(pv, key=pv.get); m = len(ks); out = {}; run = 0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, (m - i) * pv[k])); out[k] = run
    return out


# bootstrap-based two-sided p for GENIE contrasts, then Holm across the nextprog family
for task in ["nextprog", "mort12"]:
    con = S["genie"][task]["contrasts"]
    pv = {}
    for k, v in con.items():
        if ":" in k:
            continue
        se = (v["hi"] - v["lo"]) / (2 * 1.96)
        from math import erf, sqrt
        z = abs(v["diff"]) / se if se > 0 else 0
        pv[k] = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    adj = holm(pv)
    for k in pv:
        con[k]["p"] = pv[k]; con[k]["p_holm"] = adj[k]
    # patient-weighted mean AUROC (sensitivity to equal institution weighting)
    for n, e in S["genie"][task].items():
        if isinstance(e, dict) and "per_centre" in e:
            w = S["genie"][task]["n"]
            e["auroc_weighted"] = float(sum(e["per_centre"][c] * w[c] for c in CENT) / sum(w[c] for c in CENT))

# GENIE second pass: extra probes and time-since-diagnosis-only baseline
G2 = defaultdict(lambda: defaultdict(dict))
for f in glob.glob(os.path.join(R, "genie2", "*.json")):
    r = J(f); a = r["config"]
    G2[(a["task"], a["name"])][a["test_center"]][a["seed"]] = r["sets"]
g2 = {}
for (task, name), d in G2.items():
    e = {}
    for pr in ["test", "shuffle", "nogap", "scale2", "shift365"]:
        per = [np.mean([d[c][s][pr]["auroc"] for s in d[c]]) for c in CENT if c in d and pr in d[c][min(d[c])]]
        if per:
            e[pr] = float(np.mean(per))
        sh = [np.mean([d[c][s][pr].get("mean_abs_shift", 0) for s in d[c]]) for c in CENT if c in d and pr in d[c][min(d[c])]]
        if sh and pr != "test":
            e[pr + "_shift"] = float(np.mean(sh))
    e["n_centres"] = len(d)
    g2[f"{task}:{name}"] = e
S["genie2"] = g2

# strict reasoning reruns (test180, 24k tokens): completion and AUROC
strict = {}
for f in glob.glob(os.path.join(R, "llm", "growth_*_reasonS_*.json")):
    r = J(f); n = os.path.basename(f)[:-5]; st = r["sets"]["test180"]
    strict[n] = dict(auroc_imputed=st["auroc"], auroc_completed=st.get("auroc_completed"),
                     completion=float(np.mean(st["completed"])), truncated=r.get("truncated"),
                     median_tokens=float(np.median(st["n_gen_tokens"])))
S["llm_strict"] = strict


def comp_pair(fa, fb, key="score"):
    a, b = J(fa)["sets"]["test180"], J(fb)["sets"]["test180"]
    ca = dict(zip(a["pid"], a.get("completed", [True] * len(a["pid"]))))
    cb = dict(zip(b["pid"], b.get("completed", [True] * len(b["pid"]))))
    sa, sb = dict(zip(a["pid"], a[key])), dict(zip(b["pid"], b[key]))
    ids = [i for i in a["pid"] if ca[i] and cb.get(i, False)]
    y = np.array([dict(zip(a["pid"], a["y"]))[i] for i in ids])
    pa = np.array([sa[i] for i in ids]); pb = np.array([sb[i] for i in ids])
    bs = []
    for _ in range(2000):
        k = rng.integers(0, len(y), len(y)); bs.append(fast_auc(y[k], pa[k]) - fast_auc(y[k], pb[k]))
    return dict(n=len(ids), a=float(fast_auc(y, pa)), b=float(fast_auc(y, pb)), diff=float(fast_auc(y, pa) - fast_auc(y, pb)),
                lo=float(np.nanpercentile(bs, 2.5)), hi=float(np.nanpercentile(bs, 97.5)),
                shift=float(np.mean(np.abs(np.array([dict(zip(a["pid"], a["p"]))[i] for i in ids]) -
                                           np.array([dict(zip(b["pid"], b["p"]))[i] for i in ids])))))


sc = {}
for k in ["q8", "q14", "q27", "med8", "oss20"]:
    base = os.path.join(R, "llm", f"growth_{k}_reasonS_")
    if os.path.exists(base + "dates.json") and os.path.exists(base + "nodates.json"):
        sc[f"{k}:dates-nodates"] = comp_pair(base + "dates.json", base + "nodates.json")
    if os.path.exists(base + "redate21.json") and os.path.exists(base + "redate90.json"):
        sc[f"{k}:redate21-redate90"] = comp_pair(base + "redate21.json", base + "redate90.json")
    if os.path.exists(base + "dates.json") and os.path.exists(base + "redate21.json"):
        sc[f"{k}:dates-redate21"] = comp_pair(base + "dates.json", base + "redate21.json")
    d0 = os.path.join(R, "llm", f"growth_{k}_logprob_nodates.json")
    if os.path.exists(base + "dates.json") and os.path.exists(d0):
        sc[f"{k}:reasondates-directnodates"] = comp_pair(base + "dates.json", d0)
S["llm_strict_contrasts"] = sc

for nm, fn in [("registry_common", "registry_common.json"), ("chord_extra", "chord_extra.json"),
               ("surv_ci", "surv_ci.json"), ("ceil_redate", "ceil_redate.json")]:
    pth = os.path.join(OUT, fn)
    if os.path.exists(pth):
        S[nm] = J(pth)
json.dump(S, open(os.path.join(OUT, "summary2.json"), "w"), indent=1, default=float)
print("review additions:", {k: len(v) if hasattr(v, "__len__") else v for k, v in
                            [("genie2", g2), ("llm_strict", strict), ("llm_strict_contrasts", sc)]})


# ---------------------------------------------------------------- extensions: calibration, subgroups, GENIE LLMs, all-type grading
def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def calib(y, p):
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(y); p = np.asarray(p, float)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return dict(brier=float(np.mean((p - y) ** 2)), slope=float(m.coef_[0][0]), intercept=float(m.intercept_[0]),
                mean_p=float(p.mean()), prevalence=float(y.mean()))


# patient attributes (sex, race) from the GENIE downloads
attr = {}
for f in glob.glob(os.path.join(DATA, "genie_bpc", "*_public_genie_bpc.json")):
    dd = J(f)
    for r in dd["patient_clinical"]:
        if r["clinicalAttributeId"] in ("SEX", "PRIMARY_RACE"):
            attr.setdefault(r["patientId"], {})[r["clinicalAttributeId"]] = r["value"]
cal, sub = {}, {}
for task in ["nextprog", "mort12"]:
    for n in ["htt_none", "htt_orig_elapsed", "htt_clin_interval", "lr_notime", "lr_time", "gbm_time"]:
        if (task, n) not in G:
            continue
        ys, ps, pids = [], [], []
        per = {}
        for c in CENT:
            if c not in G[(task, n)]:
                continue
            pid_, y_, P_ = center_pred(task, n, c)
            pm = P_.mean(0)
            per[c] = calib(y_, pm)
            ys += list(y_); ps += list(pm); pids += pid_
        cal[f"{task}:{n}"] = dict(pooled=calib(ys, ps), per_centre=per)
        ys, ps = np.array(ys), np.array(ps)
        g = {}
        for key in ["SEX", "PRIMARY_RACE"]:
            vals = np.array([attr.get(i, {}).get(key, "Unknown") for i in pids])
            for v in np.unique(vals):
                m = vals == v
                if m.sum() >= 50 and len(set(ys[m])) > 1:
                    g[f"{key}={v}"] = dict(n=int(m.sum()), auroc=float(fast_auc(ys[m], ps[m])))
        sub[f"{task}:{n}"] = g
S["genie_calibration"] = cal; S["genie_subgroups"] = sub

gl = {}
for f in glob.glob(os.path.join(R, "genie_llm", "*.json")):
    r = J(f); a = r["config"]
    gl[f"{a['task']}:{a['name']}"] = dict(mean=r["auroc_mean_centres"], per_centre={c: v["auroc"] for c, v in r["centres"].items()},
                                          parse_fail=r["parse_fail"], n=r["n"])
def glcontrast(task, n1, n0):
    f1 = os.path.join(R, "genie_llm", f"{task}_{n1}.json"); f0 = os.path.join(R, "genie_llm", f"{task}_{n0}.json")
    if not (os.path.exists(f1) and os.path.exists(f0)):
        return None
    A1, A0 = J(f1)["centres"], J(f0)["centres"]
    data = []
    for c in CENT:
        if c in A1 and c in A0:
            m0 = dict(zip(A0[c]["pid"], A0[c]["score"]))
            data.append((np.array(A1[c]["y"]), np.array(A1[c]["score"]), np.array([m0[i] for i in A1[c]["pid"]])))
    st = lambda idx: np.mean([fast_auc(y[i], a1[i]) - fast_auc(y[i], a0[i]) for (y, a1, a0), i in zip(data, idx)])
    pt = st([np.arange(len(d[0])) for d in data])
    bs = [st([rng.integers(0, len(d[0]), len(d[0])) for d in data]) for _ in range(1000)]
    return dict(diff=float(pt), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
glc = {}
for task in ["nextprog", "mort12"]:
    for k in ["q8", "q14", "q27"]:
        for v in ["nodates", "shuffle"]:
            c = glcontrast(task, f"{k}_dates", f"{k}_{v}")
            if c:
                glc[f"{task}:{k}:dates-{v}"] = c
S["genie_llm"] = gl; S["genie_llm_contrasts"] = glc

ga = {}
for f in glob.glob(os.path.join(R, "llm", "gradeall_*.json")):
    r = J(f); per = {k[5:]: v["auroc"] for k, v in r["sets"].items() if v["auroc"] == v["auroc"]}
    ga[os.path.basename(f)[:-5]] = dict(per_type=per, macro=float(np.mean(list(per.values()))), n_types=len(per),
                                         parse_fail=r["parse_fail"])
S["llm_grade_alltypes"] = ga
json.dump(S, open(os.path.join(OUT, "summary2.json"), "w"), indent=1, default=float)
print("extensions:", len(cal), len(gl), len(glc), len(ga))
