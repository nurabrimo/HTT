"""TF-IDF + logistic regression (word 1-2 grams), same frames/splits as tcga_run. CPU.
C chosen on the in-distribution validation split by AUROC."""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from httx.tcga_data import make_frame, split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def run(df, text, held, labels="corr", types="true", max_tokens=None):
    d = make_frame(df, labels, types, text)
    if max_tokens:
        d["report_text"] = d.report_text.str.split().str[:max_tokens].str.join(" ")
    tr, va, te, ho = split(d, set(held))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, sublinear_tf=True, max_features=200000)
    Xtr = vec.fit_transform(tr.report_text)
    best = None
    for C in [0.3, 1, 3, 10, 30]:
        clf = LogisticRegression(C=C, max_iter=2000, class_weight="balanced").fit(Xtr, tr.label)
        a = roc_auc_score(va.label, clf.predict_proba(vec.transform(va.report_text))[:, 1])
        if best is None or a > best[0]:
            best = (a, C, clf)
    clf = best[2]
    out = dict(C=best[1], val=best[0], sets={})
    for name, x in [("test", te)] + [(f"held_{t}", ho[ho.cancer_type == t]) for t in held]:
        if len(x) == 0:
            continue
        p = clf.predict_proba(vec.transform(x.report_text))[:, 1]
        out["sets"][name] = dict(auroc=float(roc_auc_score(x.label, p)) if x.label.nunique() > 1 else float("nan"),
                                 pid=x.patient_id.tolist(), y=x.label.tolist(), p=p.round(5).tolist(),
                                 ctype=x.cancer_type.tolist())
    return out


if __name__ == "__main__":
    df = pd.read_pickle(sys.argv[1]); outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
    res = {}
    for text in ["raw", "redacted"]:
        res[f"main_{text}"] = run(df, text, ["THCA", "SARC", "LUSC"])
        res[f"ref_paper_{text}"] = run(df, text, ["THCA", "SARC", "LUSC"], labels="paper", types="regex")
        for t in ["BRCA", "COAD", "HNSC", "LIHC", "LUAD", "SARC", "UCEC", "LUSC"]:
            res[f"loto_{t}_{text}"] = run(df, text, [t])
        print(text, {k: {s: round(v["auroc"], 3) for s, v in r["sets"].items()} for k, r in res.items() if k.endswith(text)}, flush=True)
    json.dump(res, open(os.path.join(outdir, "tfidf.json"), "w"))
