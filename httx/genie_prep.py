"""GENIE BPC (six public cohorts) -> per-patient real clinical timelines and two prediction tasks.

Events (days since diagnosis of the index cancer): imaging reports (curated trend + impression status +
scan type), oncologist notes (curated trend + impression status), pathology reports, systemic/radiation
treatment starts and ends, performance status, tumour-marker labs, diagnosis (stage, age).

Tasks (one sample per patient; inputs are events strictly BEFORE the prediction day, then a query token
placed at the prediction day):
  nextprog : will the next imaging report be curated as progressing?  The target scan is drawn uniformly
             (seed 0) from scans with >= 2 earlier scans and a determinate curated status.
  mort12   : death within 365 days after a landmark 365 days after diagnosis (alive at landmark, >= 3 prior
             events, known status at landmark + 365).
Writes genie.pkl with dict(F=feature names, samples=list of dicts) and genie_summary.json.
"""
import glob
import json
import math
import os
import sys

import numpy as np

COHORT = {"nsclc": "NSCLC", "crc": "CRC", "brca": "BrCa", "panc": "PANC", "prostate": "Prostate", "bladder": "Bladder"}
KINDS = ["imaging", "medonc", "pathology", "tx_start", "tx_end", "perf", "lab", "diagnosis", "query"]
CUR = ["prog", "stable", "resp", "mixed", "indet"]
IMP = ["yes", "no", "uncertain", "notmention"]
SCAN = ["ct", "mri", "pet", "bone", "other"]
STAGE = ["0", "I", "II", "III", "IV"]
F = (["k_" + k for k in KINDS] + ["img_" + c for c in CUR] + ["imgimp_" + c for c in IMP] +
     ["onc_" + c for c in CUR] + ["oncimp_" + c for c in IMP] + ["scan_" + s for s in SCAN] +
     ["tx_systemic", "tx_radiation", "tx_regimen", "ecog", "lab_logratio", "lab_high"] +
     ["stage_" + s for s in STAGE] + ["age"])
IX = {f: i for i, f in enumerate(F)}
MAXEV = 63


def cur(v):
    v = (v or "").lower()
    return ("prog" if v.startswith("progress") else "stable" if v.startswith("stable") else
            "resp" if v.startswith("improv") else "mixed" if v.startswith("mixed") else "indet")


def imp(v):
    v = (v or "").lower().strip('"')
    return ("yes" if v.startswith("yes") else "no" if v.startswith("no,") or v.startswith("no ") else
            "uncertain" if "uncertain" in v else "notmention")


def scan(v):
    v = (v or "").lower()
    return "ct" if v == "ct" else "mri" if v.startswith("mri") else "pet" if "pet" in v else \
        "bone" if v.startswith("bone") else "other"


def vec(**on):
    x = np.zeros(len(F), np.float32)
    for k, v in on.items():
        x[IX[k]] = v
    return x


def patient_events(evs):
    out = []                       # (day, kind, x, curated_img_status or None)
    for e in evs:
        t = e["eventType"].upper()
        a = {z["key"]: z["value"] for z in e["attributes"]}
        d = e.get("startNumberOfDaysSinceDiagnosis")
        if d is None:
            continue
        d = float(d)
        if t == "IMAGING":
            c, i = cur(a.get("CURATED_CANCER_STATUS")), imp(a.get("CANCER_STATUS"))
            out.append((d, "imaging", vec(k_imaging=1, **{"img_" + c: 1, "imgimp_" + i: 1,
                                                        "scan_" + scan(a.get("IMAGE_SCAN_TYPE")): 1}), c))
        elif t == "MEDONC":
            c, i = cur(a.get("CURATED_CANCER_STATUS")), imp(a.get("CANCER_STATUS"))
            out.append((d, "medonc", vec(k_medonc=1, **{"onc_" + c: 1, "oncimp_" + i: 1}), None))
        elif t == "PATHOLOGY":
            out.append((d, "pathology", vec(k_pathology=1), None))
        elif t == "TREATMENT":
            sysrt = "tx_radiation" if "radiation" in a.get("TREATMENT_TYPE", "").lower() else "tx_systemic"
            try:
                reg = min(float(a.get("REGIMEN_NUMBER", 1)), 10) / 10
            except ValueError:
                reg = 0.1
            out.append((d, "tx_start", vec(k_tx_start=1, **{sysrt: 1, "tx_regimen": reg}), None))
            end = e.get("endNumberOfDaysSinceDiagnosis")
            if end is not None and float(end) > d:
                out.append((float(end), "tx_end", vec(k_tx_end=1, **{sysrt: 1, "tx_regimen": reg}), None))
        elif t == "PERFORMANCE STATUS":
            try:
                r = float(a.get("RESULT"))
            except (TypeError, ValueError):
                continue
            ecog = r if "ECOG" in a.get("SCORE_TYPE", "").upper() or r <= 4 else (100 - r) / 20
            out.append((d, "perf", vec(k_perf=1, ecog=min(max(ecog, 0), 5) / 4), None))
        elif t == "LAB_TEST":
            try:
                r, u = float(a.get("RESULT")), float(a.get("TM_NORMAL_RANGE_UPPER"))
            except (TypeError, ValueError):
                continue
            if r < 0 or u <= 0:
                continue
            lr = math.log((r + 1e-3) / u)
            out.append((d, "lab", vec(k_lab=1, lab_logratio=max(min(lr, 6), -6) / 3, lab_high=float(r > u)), None))
        elif t == "DIAGNOSIS":
            if a.get("INDEX_CANCER", "Yes") != "Yes":
                continue
            st = a.get("STAGE_DX", "")
            s = st.replace("Stage ", "").split()[0] if st.startswith("Stage") else ""
            s = s if s in STAGE else None
            try:
                age = float(a.get("AGE_DX")) / 100
            except (TypeError, ValueError):
                age = 0.65
            on = {"k_diagnosis": 1, "age": age}
            if s:
                on["stage_" + s] = 1
            out.append((d, "diagnosis", vec(**on), None))
    out.sort(key=lambda z: (z[0], KINDS.index(z[1])))
    return out


def make_sample(evs, day, extra):
    prior = [e for e in evs if e[0] < day]
    prior = prior[-MAXEV:]
    days = [e[0] for e in prior] + [day]
    X = np.stack([e[2] for e in prior] + [vec(k_query=1)])
    t0 = min(0.0, days[0])
    s = dict(days=np.array(days, np.float32) - t0, feats=X, kinds=[e[1] for e in prior] + ["query"],
             cur=[e[3] for e in prior] + [None], query_day=day - t0)
    s.update(extra)
    return s


def main(indir, out):
    rng = np.random.default_rng(0)
    samples, summ = [], {}
    for f in sorted(glob.glob(os.path.join(indir, "*_public_genie_bpc.json"))):
        d = json.load(open(f))
        coh = COHORT[d["study"].split("_")[0]]
        pc = {}
        for r in d["patient_clinical"]:
            pc.setdefault(r["patientId"], {})[r["clinicalAttributeId"]] = r["value"]
        n_np = n_m = 0
        for pid in d["patients"]:
            a = pc.get(pid, {})
            center = a.get("CENTER") or pid.split("-")[1]
            evs = patient_events(d["events"].get(pid, []))
            if not evs:
                continue
            base = dict(pid=pid, cohort=coh, center=center)
            # nextprog
            scans = [k for k, e in enumerate(evs) if e[1] == "imaging"]
            cand = [k for j, k in enumerate(scans) if j >= 2 and evs[k][3] in ("prog", "stable", "resp", "mixed")]
            if cand:
                k = cand[rng.integers(len(cand))]
                s = make_sample(evs, evs[k][0], dict(base, task="nextprog", label=int(evs[k][3] == "prog"),
                                                     target_status=evs[k][3]))
                if sum(1 for z in s["kinds"] if z == "imaging") >= 2:
                    samples.append(s); n_np += 1
            # mort12
            try:
                os_d = float(a["OS_MONTHS"]) * 30.4375
                dead = str(a.get("OS_STATUS", "")).startswith("1")
            except (KeyError, ValueError):
                continue
            L = 365.0
            if os_d > L and (dead and os_d <= 2 * L or os_d > 2 * L):
                y = int(dead and os_d <= 2 * L)
                s = make_sample(evs, L, dict(base, task="mort12", label=y))
                if len(s["kinds"]) >= 4:
                    samples.append(s); n_m += 1
        summ[coh] = dict(patients=len(d["patients"]), nextprog=n_np, mort12=n_m)
    for task in ["nextprog", "mort12"]:
        ss = [s for s in samples if s["task"] == task]
        summ[task] = dict(n=len(ss), pos=float(np.mean([s["label"] for s in ss])),
                          centers={c: sum(1 for s in ss if s["center"] == c) for c in sorted({s["center"] for s in ss})},
                          median_events=float(np.median([len(s["kinds"]) for s in ss])),
                          median_query_gap_days=float(np.median([s["days"][-1] - s["days"][-2] for s in ss])))
    import pickle
    pickle.dump(dict(F=F, samples=samples), open(out, "wb"))
    json.dump(summ, open(out.replace(".pkl", "_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
