"""Build the TCGA grade table with both label schemes and both cancer-type sources.

label_paper : OncoLocus _extract_grade_label as released (substring-style regexes; 'grade i'
              also fires on 'grade ii' / 'grade iv' / 'grade is').
label_corr  : corrected. Word-boundary patterns; Roman numerals must be whole tokens;
              WHO/histologic grade IV or 4 counts as high; 'grade 1-2' is not low;
              precursor lesions (high-grade dysplasia / PIN / SIL) do not count; Gleason
              'grade 3+4' style scores are excluded. Mixed high and low -> high (paper rule).
type_regex  : OncoLocus _extract_cancer_site as released.
type_true   : TCGA project of the patient (tcga_patient_to_cancer_type.csv).
"""
import os
import re
import sys
import json
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from httx.oncolocus_ref import _extract_cancer_site, _extract_grade_label

TYPES14 = ["BLCA", "BRCA", "COAD", "GBM", "HNSC", "KIRC", "LIHC", "LUAD", "PRAD", "STAD", "UCEC",
           "THCA", "SARC", "LUSC"]

_NOT_PRECURSOR = r"(?![\s-]*(?:dysplasia|prostatic\s+intraepithelial|squamous\s+intraepithelial|intraepithelial|pin\b|sil\b|cin\b))"
HIGH = [
    r"\bgrade\s*[:\-]?\s*(?:iii|3|iv|4)\b(?!\s*[+/])",
    r"\b(?:who|histologic(?:al)?|nuclear|fuhrman|fnclcc|nottingham|tumou?r|overall)\s+grade\s*[:\-]?\s*(?:iii|3|iv|4)\b(?!\s*[+/])",
    r"\bpoorly[\s-]*differentiated\b",
    r"\bundifferentiated\b",
    r"\bhigh[\s-]*grade\b" + _NOT_PRECURSOR,
]
LOW = [
    r"\bgrade\s*[:\-]?\s*(?:i|1)\b(?!\s*(?:[+/]|-\s*(?:ii|2|iii|3)|to\s+(?:ii|2)))",
    r"\b(?:who|histologic(?:al)?|nuclear|fuhrman|fnclcc|nottingham|tumou?r|overall)\s+grade\s*[:\-]?\s*(?:i|1)\b(?!\s*[+/\-])",
    r"\bwell[\s-]*differentiated\b",
    r"\blow[\s-]*grade\b" + _NOT_PRECURSOR,
]
HIGH_RE = re.compile("|".join(HIGH), re.I)
LOW_RE = re.compile("|".join(LOW), re.I)
GLEASON_CTX = re.compile(r"gleason[^.\n]{0,60}$", re.I)


def _hits(rx, t):
    out = []
    for m in rx.finditer(t):
        if GLEASON_CTX.search(t[max(0, m.start() - 60):m.start()]):
            continue                      # a Gleason pattern/grade number, not a tumour grade
        out.append(m)
    return out


def label_corrected(t):
    h, l = _hits(HIGH_RE, t), _hits(LOW_RE, t)
    if h:
        return 1
    if l:
        return 0
    return np.nan


# Redaction of all grade/differentiation language (level 2 of the earlier controls, extended
# with the corrected patterns). Matched spans become the neutral token "[MASK]".
R_EXTRA = [
    r"\b(?:histologic(?:al)?|nuclear|tumou?r|who|fuhrman(?:'?s)?|fnclcc|isup|nottingham|bloom[\s-]*richardson|"
    r"elston|combined|overall|architectural|mitotic|gleason|cytologic(?:al)?)?\s*grade[sd]?\b[\s:]*(?:is|of)?\s*"
    r"[ivx0-9]{0,4}(?:\s*(?:/|of|out of|-|to)\s*[ivx0-9]{1,4})?",
    r"\b(?:well|moderately|moderate|poorly|poor|intermediate|inter|un|de)[\s-]*differentiat\w*",
    r"\bdifferentiat\w*",
    r"\bgleason\b[^.\n]{0,40}", r"\bfuhrman\b[^.\n]{0,30}", r"\bfnclcc\b[^.\n]{0,30}",
    r"\bnottingham\b[^.\n]{0,40}", r"\bbloom[\s-]*richardson\b[^.\n]{0,40}", r"\bisup\b[^.\n]{0,20}",
    r"\b(?:low|high|intermediate)[\s-]*grade\b", r"\bg[1-4]\b", r"\banaplastic\b",
]
REDACT_RE = re.compile("|".join(HIGH + LOW + R_EXTRA), re.I)


def redact(t):
    return REDACT_RE.sub(" [MASK] ", t)


def build(root, out):
    df = pd.read_csv(os.path.join(root, "data", "TCGA_Reports.csv"))
    meta = pd.read_csv(os.path.join(root, "tcga-path-reports", "data", "tcga_metadata",
                                    "tcga_patient_to_cancer_type.csv"))
    df["patient_id"] = df.patient_filename.str.split(".").str[0]
    df = df.merge(meta.rename(columns={"cancer_type": "type_true"}), on="patient_id", how="left")
    df["type_regex"] = df.text.apply(_extract_cancer_site)
    df["label_paper"] = df.text.apply(_extract_grade_label)
    df["label_corr"] = df.text.apply(label_corrected)
    df["text_redacted"] = df.text.apply(redact)
    df["still_matches_after_redaction"] = df.text_redacted.apply(
        lambda t: bool(HIGH_RE.search(t) or LOW_RE.search(t)))
    df.to_pickle(out)
    return df


def audit(df):
    a = {}
    a["n_reports"] = len(df); a["n_patients"] = int(df.patient_id.nunique())
    pap = df[df.label_paper.notna() & df.type_regex.isin(TYPES14)]
    a["paper_labelled_in_14_regex_types"] = len(pap)
    a["paper_high_low"] = [int((pap.label_paper == 1).sum()), int((pap.label_paper == 0).sum())]
    a["paper_type_regex_vs_true_agreement"] = float((pap.type_regex == pap.type_true).mean())
    a["paper_heldout_composition"] = {
        t: pap[pap.type_regex == t].type_true.value_counts().head(5).to_dict() for t in ["THCA", "SARC", "LUSC"]}
    # how many paper 'low' labels mention grade II / IV only
    low = pap[pap.label_paper == 0]
    a["paper_low_n"] = len(low)
    a["paper_low_relabelled_by_corrected"] = low.label_corr.map({1.0: "high", 0.0: "low"}).fillna("unlabelled").value_counts().to_dict()
    hi = pap[pap.label_paper == 1]
    a["paper_high_relabelled_by_corrected"] = hi.label_corr.map({1.0: "high", 0.0: "low"}).fillna("unlabelled").value_counts().to_dict()
    gbm = pap[pap.type_true == "GBM"]
    a["GBM_paper_label_counts"] = gbm.label_paper.value_counts().to_dict()
    cor = df[df.label_corr.notna() & df.type_true.isin(TYPES14)]
    a["corrected_labelled_in_14_true_types"] = len(cor)
    a["corrected_high_low"] = [int((cor.label_corr == 1).sum()), int((cor.label_corr == 0).sum())]
    a["corrected_per_type"] = {t: [int((g.label_corr == 1).sum()), int((g.label_corr == 0).sum())]
                               for t, g in cor.groupby("type_true")}
    both = df[df.label_paper.notna() & df.label_corr.notna()]
    a["agreement_paper_vs_corrected_where_both_defined"] = float((both.label_paper == both.label_corr).mean())
    a["corrected_still_matching_after_redaction"] = int(cor.still_matches_after_redaction.sum())
    return a


if __name__ == "__main__":
    root, out = sys.argv[1], sys.argv[2]
    df = build(root, out)
    a = audit(df)
    print(json.dumps(a, indent=1, default=str))
    json.dump(a, open(out.replace(".pkl", "_audit.json"), "w"), indent=1, default=str)
