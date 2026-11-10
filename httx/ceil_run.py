"""Growth-benchmark ceilings. Usage:
  python httx/ceil_run.py sets OUTDIR            # sample ceilings for each evaluation set + OLS + re-dating
  python httx/ceil_run.py pop OUTDIR CHUNK NCHUNK  # population ceilings, one chunk of 5,000 patients
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from httx import growth as G
from sklearn.metrics import roc_auc_score

mode, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
if mode == "sets":
    sets = G.build_benchmark(70, 42)
    res = {}
    for k in ["val", "test60", "test120", "test180", "PANC100", "PANC300"]:
        c = G.ceilings(sets[k], bank_size=1500)
        c.to_csv(os.path.join(out, f"ceil_{k}.csv"), index=False)
        o = G.ols_rate_reader(sets[k])
        res[k] = dict(n=len(c), pos=float(c.label.mean()),
                      last=roc_auc_score(c.label, c.p_last), nodates=roc_auc_score(c.label, c.p_nodates),
                      dates=roc_auc_score(c.label, c.p_dates), ols=roc_auc_score(o.label, o.score))
        print(k, res[k], flush=True)
    res["redate_shift_bayes_21_90"] = G.redate_ceiling_shift(sets["test180"], 21, 90)
    print(res["redate_shift_bayes_21_90"])
    json.dump(res, open(os.path.join(out, "ceilings_sets.json"), "w"), indent=1)
else:
    chunk, nchunk = int(sys.argv[3]), int(sys.argv[4])
    P = G.generate(1250, ["NSCLC", "BrCa", "CRC", "Prostate"], 777, "P")   # 5,000 patients
    ids = P.patient_id.unique()
    mine = set(ids[chunk::nchunk])
    c = G.ceilings(P[P.patient_id.isin(mine)], bank_size=1500, seed=123)
    c.to_csv(os.path.join(out, f"pop_{chunk}.csv"), index=False)
    print("chunk", chunk, len(c))
