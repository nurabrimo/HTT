"""NLP-derived vs human-curated labels on the same radiology reports.

MSK patients appear in both MSK-CHORD (NLP outputs per radiology report: PROGRESSION Y/N/Indeterminate with
probability; HAS_CANCER with probability) and GENIE BPC (PRISSMM human curation per report: curated trend,
impression cancer status). Timelines are aligned through the shared tumour sample: offset = GENIE sequencing
day (days since diagnosis) - MSK-CHORD sequencing day for the same sample ID.  Reports are matched on
patient, aligned day and modality (CT / MR / PET).
Writes chord_vs_genie.json.  Usage: chord_vs_genie.py <genie_dir> <chord_dir> <out.json>
"""
import glob
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def mod_genie(v):
    v = (v or "").lower()
    return "CT" if v == "ct" else "MR" if v.startswith("mri") else "PET" if "pet" in v else None


def main(gdir, cdir, out):
    prog = pd.read_csv(os.path.join(cdir, "data_timeline_progression.txt"), sep="\t", low_memory=False)
    pres = pd.read_csv(os.path.join(cdir, "data_timeline_cancer_presence.txt"), sep="\t", low_memory=False)
    # Date alignment: the two resources use different day-0 anchors. For each patient the offset is the
    # shift that makes the most GENIE imaging dates coincide with MSK-CHORD radiology-report dates (dates only,
    # never labels). Patients are kept if >= 3 dates and >= 50% of GENIE scans coincide under that shift.
    cdays = pd.concat([prog[["PATIENT_ID", "START_DATE"]], pres[["PATIENT_ID", "START_DATE"]]]) \
        .groupby("PATIENT_ID").START_DATE.apply(lambda s: np.array(sorted(set(s.astype(int)))))
    rows = []
    align = dict(candidates=0, kept=0)
    for f in sorted(glob.glob(os.path.join(gdir, "*_public_genie_bpc.json"))):
        d = json.load(open(f))
        coh = d["study"].split("_")[0]
        for pid in d["patients"]:
            if "-MSK-" not in pid:
                continue
            cpid = pid.replace("GENIE-MSK-", "")
            evs = d["events"].get(pid, [])
            gd = sorted({round(float(e["startNumberOfDaysSinceDiagnosis"])) for e in evs
                         if e["eventType"].upper() == "IMAGING" and e.get("startNumberOfDaysSinceDiagnosis") is not None})
            if cpid not in cdays.index or len(gd) < 3:
                continue
            align["candidates"] += 1
            cnt = Counter(int(x - y) for x in gd for y in cdays[cpid])
            off, hits = cnt.most_common(1)[0]
            if hits < 3 or hits < 0.5 * len(gd):
                continue
            align["kept"] += 1
            for e in evs:
                if e["eventType"].upper() != "IMAGING":
                    continue
                a = {z["key"]: z["value"] for z in e["attributes"]}
                m = mod_genie(a.get("IMAGE_SCAN_TYPE"))
                if m is None:
                    continue
                rows.append(dict(cohort=coh, pid=cpid, day=round(float(e["startNumberOfDaysSinceDiagnosis"]) - off),
                                 mod=m, curated=a.get("CURATED_CANCER_STATUS", ""), impression=a.get("CANCER_STATUS", "")))
    g = pd.DataFrame(rows)
    A = dict(genie_msk_imaging_reports=len(g), genie_msk_patients=int(g.pid.nunique()),
             alignment=align)
    prog["mod"] = prog.PROCEDURE_TYPE.astype(str).map(lambda v: v if v in ("CT", "MR", "PET") else None)
    pres["mod"] = pres.PROCEDURE_TYPE.astype(str).map(lambda v: v if v in ("CT", "MR", "PET") else None)
    key = ["pid", "day", "mod"]
    p2 = prog.rename(columns={"PATIENT_ID": "pid", "START_DATE": "day"})[key + ["PROGRESSION", "NLP_PROGRESSION_PROBABILITY"]]
    c2 = pres.rename(columns={"PATIENT_ID": "pid", "START_DATE": "day"})[key + ["HAS_CANCER", "NLP_HAS_CANCER_PROBABILITY"]]
    # one NLP row per (pid, day, modality): keep exact matches only, drop days with several same-modality reports
    p2 = p2.drop_duplicates(key, keep=False)
    c2 = c2.drop_duplicates(key, keep=False)
    gu = g.drop_duplicates(key, keep=False)
    m = gu.merge(p2, on=key, how="inner").merge(c2, on=key, how="left")
    A["matched_reports"] = len(m); A["matched_patients"] = int(m.pid.nunique())
    A["matched_by_cohort"] = m.cohort.value_counts().to_dict()
    cur = m.curated.str.lower()
    m["cur_prog"] = np.where(cur.str.startswith("progress"), 1,
                             np.where(cur.str.startswith(("stable", "improv")), 0, np.nan))
    m["nlp_prog"] = m.PROGRESSION.map({"Y": 1, "N": 0})
    A["crosstab_curated_vs_nlp"] = pd.crosstab(m.curated.str[:22], m.PROGRESSION.astype(str)).to_dict()
    b = m[m.cur_prog.notna() & m.nlp_prog.notna()]
    A["progression"] = dict(n=len(b), agree=float((b.cur_prog == b.nlp_prog).mean()),
                            kappa=float(cohen_kappa_score(b.cur_prog, b.nlp_prog)),
                            nlp_prob_auroc_vs_curated=float(roc_auc_score(m[m.cur_prog.notna()].cur_prog,
                                                                          m[m.cur_prog.notna()].NLP_PROGRESSION_PROBABILITY)),
                            nlp_Y_but_curated_not=int(((b.nlp_prog == 1) & (b.cur_prog == 0)).sum()),
                            nlp_N_but_curated_prog=int(((b.nlp_prog == 0) & (b.cur_prog == 1)).sum()),
                            curated_prog_rate=float(b.cur_prog.mean()), nlp_prog_rate=float(b.nlp_prog.mean()),
                            nlp_indeterminate_share=float((m.PROGRESSION == "Indeterminate").mean()))
    per = {}
    for coh, x in b.groupby("cohort"):
        per[coh] = dict(n=len(x), agree=float((x.cur_prog == x.nlp_prog).mean()),
                        kappa=float(cohen_kappa_score(x.cur_prog, x.nlp_prog)))
    A["progression_by_cohort"] = per
    imp = m.impression.str.lower().str.strip('"')
    m["imp_yes"] = np.where(imp.str.startswith("yes"), 1, np.where(imp.str.startswith("no"), 0, np.nan))
    m["nlp_cancer"] = m.HAS_CANCER.map({"Y": 1, "N": 0})
    c = m[m.imp_yes.notna() & m.nlp_cancer.notna()]
    A["cancer_presence"] = dict(n=len(c), agree=float((c.imp_yes == c.nlp_cancer).mean()),
                                kappa=float(cohen_kappa_score(c.imp_yes, c.nlp_cancer)),
                                nlp_prob_auroc=float(roc_auc_score(c.imp_yes, c.NLP_HAS_CANCER_PROBABILITY)))
    m[["cohort", "pid", "day", "mod", "curated", "impression", "PROGRESSION", "NLP_PROGRESSION_PROBABILITY",
       "HAS_CANCER", "NLP_HAS_CANCER_PROBABILITY"]].to_csv(out.replace(".json", "_matched.csv"), index=False)
    json.dump(A, open(out, "w"), indent=1, default=str)
    print(json.dumps(A, indent=1, default=str))


if __name__ == "__main__":
    main(*sys.argv[1:4])
