"""TCGA frame construction and the OncoLocus split (shared by HTT and TF-IDF runners)."""
import random
import pandas as pd

TYPES14 = ["BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LIHC", "LUAD", "PRAD", "STAD", "UCEC",
           "THCA", "SARC", "LUSC"]


def make_frame(df, labels, types, text):
    lab = {"paper": "label_paper", "corr": "label_corr", "reg": "label_reg"}[labels]
    typ = "type_regex" if types == "regex" else "type_true"
    txt = {"raw": "text", "redacted": "text_redacted", "redacted_all": "text_redacted_all"}[text]
    d = df[df[lab].notna() & df[typ].isin(TYPES14)].copy()
    d["cancer_type"] = d[typ]
    d["label"] = d[lab].astype(int)
    d["report_text"] = "[CANCER:" + d[typ] + "] " + d[txt].astype(str).str.strip()
    d["visit_index"] = 0
    d["days_from_start"] = 0.0
    return d[["patient_id", "cancer_type", "visit_index", "days_from_start", "report_text", "label"]]


def split(d, held):
    """OncoLocus build_datasets: per-type patient shuffle with random.seed(42), 70/15/15."""
    random.seed(42)
    tr, va, te = [], [], []
    for ct in sorted(d.cancer_type.unique()):
        if ct in held:
            continue
        sub = d[d.cancer_type == ct]
        ids = sub.patient_id.unique().tolist()
        random.shuffle(ids)
        n = len(ids); a, b = int(n * 0.7), int(n * 0.7) + int(n * 0.15)
        tr.append(sub[sub.patient_id.isin(ids[:a])])
        va.append(sub[sub.patient_id.isin(ids[a:b])])
        te.append(sub[sub.patient_id.isin(ids[b:])])
    return pd.concat(tr), pd.concat(va), pd.concat(te), d[d.cancer_type.isin(held)]


