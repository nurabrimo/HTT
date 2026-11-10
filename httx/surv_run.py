"""Prognostic value of pathology-report text beyond registry covariates (TCGA-CDR endpoints).

Cross-validated Cox models (5 folds stratified by cancer type and event, repeated over --repeats seeds):
  clin        cancer type + age + registry AJCC stage + registry grade (missing indicators)
  text        report features only            (--feat tfidf | emb | llm)
  clin+text   both
Text views: raw, or with all grade AND stage/TNM language masked (text_redacted_all).
Out-of-fold risks are pooled per repeat; metrics are Harrell's C (pan-cancer), type-stratified C
(pairs within the same cancer type) and per-type C; paired bootstrap for the difference
(clin+text) - clin.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.metrics import concordance_index_censored
from sksurv.util import Surv

ALPHAS = [1.0, 10.0, 100.0, 1000.0]


def cindex(e, t, r):
    return float(concordance_index_censored(e.astype(bool), t, r)[0])


def strat_cindex(e, t, r, typ):
    """Harrell's C over comparable pairs within the same cancer type."""
    num = den = 0.0
    for ty in np.unique(typ):
        m = typ == ty
        if m.sum() < 5 or e[m].sum() < 2:
            continue
        res = concordance_index_censored(e[m].astype(bool), t[m], r[m])
        num += res[1] + 0.5 * res[3]; den += res[1] + res[2] + res[3]
    return float(num / den) if den else float("nan")


def clin_matrix(d, types):
    X = pd.get_dummies(pd.Categorical(d.type_true, categories=types), prefix="t", drop_first=True).astype(float)
    X["age"] = d.age.fillna(d.age.median()).values
    X["age_na"] = d.age.isna().astype(float).values
    for s in [2, 3, 4]:
        X[f"stage{s}"] = (d.stage_reg == s).astype(float).values
    X["stage_na"] = d.stage_reg.isna().astype(float).values
    X["grade_high"] = (d.label_reg == 1).astype(float).values
    X["grade_na"] = d.label_reg.isna().astype(float).values
    return X.values


def fit_cox(Xtr, ytr, Xva, yva):
    """Ridge Cox; penalty chosen on an inner validation split by Harrell's C."""
    sc = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = sc.transform(Xtr), sc.transform(Xva)
    best, best_a = -1, ALPHAS[0]
    for a in ALPHAS:
        try:
            m = CoxPHSurvivalAnalysis(alpha=a, n_iter=200).fit(Xtr_s, ytr)
            c = cindex(yva["e"], yva["t"], m.predict(Xva_s))
        except Exception:
            continue
        if c > best:
            best, best_a = c, a
    return best_a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/path/to/work/htt/data/tcga_cdr.pkl")
    ap.add_argument("--endpoint", choices=["OS", "PFI"], default="OS")
    ap.add_argument("--feat", choices=["tfidf", "emb", "llm"], default="tfidf")
    ap.add_argument("--text", choices=["raw", "redacted_all"], default="raw")
    ap.add_argument("--emb", default="/path/to/work/htt/data/emb_biomedbert_{text}.npy")
    ap.add_argument("--llm", default="/path/to/work/htt/runs/llm/tcga_surv_{model}_{text}.json")
    ap.add_argument("--llm_model", default="q14")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--svd", type=int, default=128)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_path = os.path.join(a.out, f"{a.name}.json")
    os.makedirs(a.out, exist_ok=True)

    df = pd.read_pickle(a.data)
    df = df.reset_index(drop=True)
    df["_row"] = np.arange(len(df))
    ev, tm = a.endpoint, f"{a.endpoint}.time"
    d = df[df[tm].notna() & (df[tm] > 0) & df[ev].notna() & df.type_true.notna()].copy().reset_index(drop=True)
    types = sorted(d.type_true.unique())
    e = d[ev].values.astype(int); t = d[tm].values.astype(float); typ = d.type_true.values
    y = Surv.from_arrays(e.astype(bool), t, name_event="e", name_time="t")
    Xc = clin_matrix(d, types)
    txtcol = "text" if a.text == "raw" else "text_redacted_all"

    if a.feat == "emb":
        E = np.load(a.emb.format(text=a.text))[d._row.values]
    elif a.feat == "llm":
        L = json.load(open(a.llm.format(model=a.llm_model, text=a.text)))
        m = {}
        for s in L["sets"].values():
            m.update(dict(zip(s["pid"], s["p"])))
        E = np.array([[m.get(p, np.nan)] for p in d.patient_id])
        E = np.where(np.isnan(E), np.nanmean(E), E)

    risks = {k: np.zeros((a.repeats, len(d))) for k in ["clin", "text", "clin+text", "type_age"]}
    strat = np.array([f"{ty}_{x}" for ty, x in zip(typ, e)])
    for rep in range(a.repeats):
        skf = StratifiedKFold(5, shuffle=True, random_state=rep)
        for tr, te in skf.split(np.zeros(len(d)), strat):
            rng = np.random.default_rng(rep)
            inner = rng.random(len(tr)) < 0.2
            itr, iva = tr[~inner], tr[inner]
            if a.feat == "tfidf":
                vec = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_features=50000, sublinear_tf=True)
                Ttr = vec.fit_transform(d[txtcol].iloc[tr].astype(str))
                svd = TruncatedSVD(a.svd, random_state=rep).fit(Ttr)
                Tall = np.zeros((len(d), a.svd))
                Tall[tr] = svd.transform(Ttr)
                Tall[te] = svd.transform(vec.transform(d[txtcol].iloc[te].astype(str)))
            elif a.feat == "emb":
                from sklearn.decomposition import PCA
                pca = PCA(a.svd, random_state=rep).fit(E[tr])
                Tall = pca.transform(E)
            else:
                Tall = E
            n_typ = len(types) - 1
            feats = {"clin": Xc, "text": Tall, "clin+text": np.hstack([Xc, Tall]),
                     "type_age": Xc[:, :n_typ + 2]}
            for k, X in feats.items():
                alpha = fit_cox(X[itr], y[itr], X[iva], y[iva])
                sc = StandardScaler().fit(X[tr])
                mdl = CoxPHSurvivalAnalysis(alpha=alpha, n_iter=200).fit(sc.transform(X[tr]), y[tr])
                risks[k][rep, te] = mdl.predict(sc.transform(X[te]))

    res = dict(config=vars(a), n=len(d), events=int(e.sum()), types=types, models={})
    for k, R in risks.items():
        c = [cindex(e, t, R[r]) for r in range(a.repeats)]
        cs = [strat_cindex(e, t, R[r], typ) for r in range(a.repeats)]
        per = {}
        for ty in types:
            mk = typ == ty
            if e[mk].sum() >= 10:
                per[ty] = float(np.mean([cindex(e[mk], t[mk], R[r][mk]) for r in range(a.repeats)]))
        res["models"][k] = dict(c=float(np.mean(c)), c_sd=float(np.std(c)), c_strat=float(np.mean(cs)),
                                c_strat_sd=float(np.std(cs)), per_type=per)
    # paired bootstrap of the type-stratified C difference, averaged over repeats
    rng = np.random.default_rng(0)
    for k1, k0 in [("clin+text", "clin"), ("text", "clin"), ("clin", "type_age")]:
        diffs = []
        for b in range(300):
            idx = np.concatenate([rng.choice(np.where(typ == ty)[0], (typ == ty).sum()) for ty in types])
            dd = [strat_cindex(e[idx], t[idx], risks[k1][r][idx], typ[idx]) -
                  strat_cindex(e[idx], t[idx], risks[k0][r][idx], typ[idx]) for r in range(min(a.repeats, 2))]
            diffs.append(np.mean(dd))
        res[f"delta_{k1}_vs_{k0}"] = dict(diff=res["models"][k1]["c_strat"] - res["models"][k0]["c_strat"],
                                          lo=float(np.percentile(diffs, 2.5)), hi=float(np.percentile(diffs, 97.5)))
    res["oof_risk_repeat0"] = {k: [round(float(x), 5) for x in R[0]] for k, R in risks.items()}
    res["pid"] = d.patient_id.tolist(); res["type"] = typ.tolist(); res["e"] = e.tolist(); res["t"] = t.tolist()
    json.dump(res, open(out_path, "w"))
    print("done", a.name, {k: (round(v["c"], 3), round(v["c_strat"], 3)) for k, v in res["models"].items()},
          {k: v for k, v in res.items() if k.startswith("delta")})


if __name__ == "__main__":
    main()
