"""Merge TCGA-Reports with the TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR; Liu et al., Cell 2018).

Adds to tcga.pkl:
  label_reg      registry grade from CDR histological_grade: G1 / 'Low Grade' -> 0; G3, G4 / 'High Grade' -> 1;
                 G2, GX and missing -> NaN (mirrors the corrected text rule, which leaves grade 2 unlabelled)
  reg_grade_raw  the CDR string
  stage_reg      AJCC pathologic stage group from CDR (I-IV or NaN); adv_reg = 1 if III/IV
  stage_text     stage group stated in the report (regex), adv_text likewise
  text_redacted_all  grade AND stage/TNM language masked (for prognosis experiments)
  age, gender, OS, OS.time, PFI, PFI.time, DSS, DSS.time
and writes cdr_audit.json (registry vs text agreement).
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from httx.tcga_prep import REDACT_RE, TYPES14

REG_LOW = {"G1", "LOW GRADE"}
REG_HIGH = {"G3", "G4", "HIGH GRADE"}

STAGE_RE = re.compile(r"\b(?:ajcc\s+)?(?:pathologic(?:al)?\s+)?stage\s*(?:group(?:ing)?)?\s*[:\-]?\s*(iv|iii|ii|i|4|3|2|1)\s*[abc]?\b",
                      re.I)
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "1": 1, "2": 2, "3": 3, "4": 4}
STAGE_REDACT = [
    r"\b(?:ajcc\s+)?(?:pathologic(?:al)?\s+|clinical\s+)?stage\s*(?:group(?:ing)?)?\s*[:\-]?\s*(?:iv|iii|ii|i|[0-4])\s*[abc]?\d?\b",
    r"\b[pcy]?r?[pcy]?t\s*(?:is|x|[0-4][abcd]?)\s*(?:\(\w+\))?\s*n\s*(?:x|[0-3][abc]?)\s*(?:\(\w+\))?\s*(?:m\s*(?:x|[01][abc]?))?\b",
    r"\b[pcy]?[tnm]\s*(?:is|x|[0-4][abcd]?)(?:\s*\(\w+\))?(?=[\s,.;:)]|$)",
    r"\bajcc\b[^.\n]{0,40}", r"\btnm\b[^.\n]{0,40}", r"\bfigo\b[^.\n]{0,40}",
    r"\b(?:primary\s+tumou?r|regional\s+lymph\s+nodes|distant\s+metastasis)\s*\(\s*[ptnm]{1,2}\s*\)\s*[:\-]?\s*[^.\n]{0,60}",
]
STAGE_REDACT_RE = re.compile("|".join(STAGE_REDACT), re.I)


def stage_from_cdr(s):
    s = str(s).upper()
    m = re.match(r"STAGE\s+(IV|III|II|I)", s)
    return ROMAN[m.group(1).lower()] if m else np.nan


def stage_from_text(t):
    v = [ROMAN[m.group(1).lower()] for m in STAGE_RE.finditer(t or "")]
    return max(v) if v else np.nan


def main(pkl, cdr_xlsx):
    df = pd.read_pickle(pkl)
    cdr = pd.read_excel(cdr_xlsx, sheet_name=0)
    cdr = cdr.rename(columns={"bcr_patient_barcode": "patient_id"})
    keep = ["patient_id", "type", "age_at_initial_pathologic_diagnosis", "gender", "ajcc_pathologic_tumor_stage",
            "histological_grade", "histological_type", "OS", "OS.time", "DSS", "DSS.time", "PFI", "PFI.time"]
    cdr = cdr[[c for c in keep if c in cdr.columns]].drop_duplicates("patient_id")
    for c in ["OS", "OS.time", "DSS", "DSS.time", "PFI", "PFI.time", "age_at_initial_pathologic_diagnosis"]:
        cdr[c] = pd.to_numeric(cdr[c], errors="coerce")
    df = df.drop(columns=[c for c in cdr.columns if c != "patient_id" and c in df.columns]).merge(cdr, on="patient_id", how="left")
    g = df.histological_grade.astype(str).str.strip().str.upper()
    df["reg_grade_raw"] = df.histological_grade
    df["label_reg"] = np.where(g.isin(REG_HIGH), 1.0, np.where(g.isin(REG_LOW), 0.0, np.nan))
    df["stage_reg"] = df.ajcc_pathologic_tumor_stage.apply(stage_from_cdr)
    df["adv_reg"] = np.where(df.stage_reg.isna(), np.nan, (df.stage_reg >= 3).astype(float))
    df["stage_text"] = df.text.apply(stage_from_text)
    df["adv_text"] = np.where(df.stage_text.isna(), np.nan, (df.stage_text >= 3).astype(float))
    df["text_redacted_all"] = df.text.astype(str).apply(lambda t: STAGE_REDACT_RE.sub(" [MASK] ", REDACT_RE.sub(" [MASK] ", t)))
    df["age"] = df.age_at_initial_pathologic_diagnosis
    df.to_pickle(pkl.replace(".pkl", "_cdr.pkl"))

    A = {}
    A["n_reports"] = len(df)
    A["matched_to_cdr"] = int(df.type.notna().sum())
    A["cdr_type_equals_project"] = float((df.type == df.type_true)[df.type.notna()].mean())
    sub = df[df.type_true.isin(TYPES14)]
    A["registry_grade_values_14types"] = sub.reg_grade_raw.astype(str).value_counts().head(25).to_dict()
    A["registry_grade_defined_14types"] = int(sub.label_reg.notna().sum())
    per = {}
    for t, s in sub.groupby("type_true"):
        r = {}
        for lab in ["label_paper", "label_corr"]:
            b = s[s[lab].notna() & s.label_reg.notna()]
            r[lab] = dict(n=len(b), agree=float((b[lab] == b.label_reg).mean()) if len(b) else None,
                          text_high_reg_low=int(((b[lab] == 1) & (b.label_reg == 0)).sum()),
                          text_low_reg_high=int(((b[lab] == 0) & (b.label_reg == 1)).sum()))
        r["reg_high_low"] = [int((s.label_reg == 1).sum()), int((s.label_reg == 0).sum())]
        r["reg_G2"] = int((s.reg_grade_raw.astype(str).str.upper().str.strip() == "G2").sum())
        per[t] = r
    A["grade_text_vs_registry_per_type"] = per
    for lab in ["label_paper", "label_corr"]:
        b = sub[sub[lab].notna() & sub.label_reg.notna()]
        from sklearn.metrics import cohen_kappa_score
        A[f"{lab}_vs_registry"] = dict(n=len(b), agree=float((b[lab] == b.label_reg).mean()),
                                       kappa=float(cohen_kappa_score(b[lab], b.label_reg)))
    # registry G2 cases: what did the text rules call them?
    g2 = sub[sub.reg_grade_raw.astype(str).str.upper().str.strip() == "G2"]
    A["registry_G2_text_labels"] = {lab: g2[lab].map({1.0: "high", 0.0: "low"}).fillna("unlabelled").value_counts().to_dict()
                                    for lab in ["label_paper", "label_corr"]}
    st = df[df.stage_reg.notna()]
    A["stage_reg_defined"] = len(st)
    A["stage_text_defined"] = int(df.stage_text.notna().sum())
    b = df[df.stage_reg.notna() & df.stage_text.notna()]
    A["stage_text_vs_registry"] = dict(n=len(b), exact=float((b.stage_text == b.stage_reg).mean()),
                                       adv_agree=float((b.adv_text == b.adv_reg).mean()))
    A["os_defined"] = int(df["OS.time"].notna().sum()); A["os_events"] = int(df.OS.fillna(0).sum())
    A["pfi_defined"] = int(df["PFI.time"].notna().sum()); A["pfi_events"] = int(df.PFI.fillna(0).sum())
    A["still_matching_grade_or_stage_after_full_redaction"] = int(
        df.text_redacted_all.apply(lambda t: bool(STAGE_RE.search(t))).sum())
    json.dump(A, open(os.path.join(os.path.dirname(pkl), "cdr_audit.json"), "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in A.items() if k != "grade_text_vs_registry_per_type"}, indent=1, default=str))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
