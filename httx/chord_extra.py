"""(1) Agreement over the full curated x NLP table; (2) null for the date alignment: pair each GENIE MSK patient
with a random other MSK-CHORD patient and apply the same offset search and acceptance rule."""
import glob, json, os, sys
from collections import Counter
import numpy as np, pandas as pd
gdir, cdir, out = sys.argv[1:4]
m = pd.read_csv(os.path.join(gdir, "chord_vs_genie_matched.csv"))
cur = m.curated.fillna("").str.lower()
m["c3"] = np.where(cur.str.startswith("progress"), "prog", np.where(cur.str.startswith(("stable", "improv")), "noprog",
          np.where(cur.str.startswith("mixed"), "mixed", "indet")))
tab = pd.crosstab(m.c3, m.PROGRESSION.fillna("NA")).to_dict()
# agreement if mixed counted as progression (a curator flags new or growing disease) and indeterminate kept separate
a = m[m.PROGRESSION.isin(["Y", "N"])]
full = dict(n_all=len(m), n_nlp_yes_no=len(a),
            agree_mixed_as_prog=float(((a.c3.isin(["prog", "mixed"])) == (a.PROGRESSION == "Y"))[a.c3 != "indet"].mean()),
            n_mixed=int((m.c3 == "mixed").sum()), mixed_called_yes=int(((m.c3 == "mixed") & (m.PROGRESSION == "Y")).sum()),
            share_nlp_indet=float((m.PROGRESSION == "Indeterminate").mean()),
            share_cur_indet=float((m.c3 == "indet").mean()))
prog = pd.read_csv(os.path.join(cdir, "data_timeline_progression.txt"), sep="\t", low_memory=False)
pres = pd.read_csv(os.path.join(cdir, "data_timeline_cancer_presence.txt"), sep="\t", low_memory=False)
cdays = pd.concat([prog[["PATIENT_ID", "START_DATE"]], pres[["PATIENT_ID", "START_DATE"]]]) \
    .groupby("PATIENT_ID").START_DATE.apply(lambda s: np.array(sorted(set(s.astype(int)))))
gd_all = {}
for f in sorted(glob.glob(os.path.join(gdir, "*_public_genie_bpc.json"))):
    d = json.load(open(f))
    for pid in d["patients"]:
        if "-MSK-" not in pid:
            continue
        gd = sorted({round(float(e["startNumberOfDaysSinceDiagnosis"])) for e in d["events"].get(pid, [])
                     if e["eventType"].upper() == "IMAGING" and e.get("startNumberOfDaysSinceDiagnosis") is not None})
        if pid.replace("GENIE-MSK-", "") in cdays.index and len(gd) >= 3:
            gd_all[pid.replace("GENIE-MSK-", "")] = gd
rng = np.random.default_rng(0)
ids = list(gd_all); others = list(cdays.index)
def accept(gd, cd):
    off, hits = Counter(int(x - y) for x in gd for y in cd).most_common(1)[0]
    return hits >= 3 and hits >= 0.5 * len(gd), hits / len(gd)
true_acc = [accept(gd_all[i], cdays[i]) for i in ids]
null_acc = [accept(gd_all[i], cdays[others[rng.integers(len(others))]]) for i in ids]
nulls = dict(n=len(ids), true_accept=float(np.mean([a for a, _ in true_acc])), null_accept=float(np.mean([a for a, _ in null_acc])),
             true_median_hit_share=float(np.median([h for _, h in true_acc])), null_median_hit_share=float(np.median([h for _, h in null_acc])))
res = dict(full_table=full, crosstab3=tab, alignment_null=nulls)
json.dump(res, open(out, "w"), indent=1, default=str); print(json.dumps(res, indent=1, default=str))
