"""Bayes reader (with dates) applied to re-dated inputs: AUROC on test180 when every visit k is placed on day k*g."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from httx.growth import _obs_patient, _post, _loglik_given_times
from sklearn.metrics import roc_auc_score
sets = pd.read_pickle(sys.argv[1]); df = sets["test180"]
res = {}
for g in [21, 42, 66, 90]:
    y, p = [], []
    for pid, gg in df.groupby("patient_id", sort=False):
        gg = gg.sort_values("visit_index")
        yy, s2, idx = _obs_patient(gg)
        k = np.arange(len(gg))[idx]
        p.append(_post(_loglik_given_times(yy, s2, (k * g)[None, :].astype(float))))
        y.append(int(gg.label.iloc[0]))
    res[f"redate{g}"] = float(roc_auc_score(y, p))
print(json.dumps(res)); json.dump(res, open(sys.argv[2], "w"))
