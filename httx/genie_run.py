"""Real-timeline experiments on GENIE BPC: leave-one-institution-out (train on three centres, test on the fourth).

--model htt : HTT structured reader (same Level 2, time codes and training protocol as Sections 2.2/2.6;
              per-event input = projection of the curated event features; cohort embedding).
--model lr  : logistic regression on hand-built summary features, with (--timefeat) or without time features.
Probes on the test centre (same model, inputs altered at evaluation only):
  shuffle : inter-event gaps permuted within each patient (order kept, total span kept)
  nogap   : query moved onto the last observed event (time-to-prediction set to 0)
  scale2  : all times doubled
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from httx import engine as E
from httx.models import HTT

COHORTS = ["NSCLC", "CRC", "BrCa", "PANC", "Prostate", "Bladder"]


class DS:
    def __init__(self, items, max_visits=64):
        self.items, self.max_visits, self.max_len = items, max_visits, 1

    def __len__(self):
        return len(self.items)


def to_item(s):
    n = len(s["days"])
    return dict(pid=s["pid"], ids=None, mask=None, days=torch.tensor(s["days"]),
                vlab=torch.full((n,), -1, dtype=torch.long), plab=int(s["label"]),
                type_id=COHORTS.index(s["cohort"]), ctype=s["cohort"], feats=torch.tensor(s["feats"]))


def probe(samples, kind, rng):
    out = []
    for s in samples:
        s = dict(s); d = s["days"].astype(np.float64)
        if kind == "shuffle":
            g = np.diff(d); g = rng.permutation(g); d = np.concatenate([[d[0]], d[0] + np.cumsum(g)])
        elif kind == "nogap":
            d = d.copy(); d[-1] = d[-2]
        elif kind == "scale2":
            d = d * 2
        elif kind == "shift365":
            d = d + 365.0
        s["days"] = d.astype(np.float32)
        out.append(s)
    return out


def summary_feats(s, F, timefeat, dxonly=False):
    X, kinds, d = s["feats"], s["kinds"], s["days"]
    ix = {f: i for i, f in enumerate(F)}
    q = d[-1]
    img = [k for k, z in enumerate(kinds) if z == "imaging"]
    v = []
    last = X[img[-1]] if img else np.zeros(len(F))
    v += [last[ix["img_" + c]] for c in ["prog", "stable", "resp", "mixed", "indet"]]
    v += [last[ix["imgimp_" + c]] for c in ["yes", "no", "uncertain", "notmention"]]
    last3 = [X[k] for k in img[-3:]]
    v += [sum(x[ix["img_prog"]] for x in last3), sum(x[ix["img_resp"]] for x in last3), len(img) / 20]
    onc = [k for k, z in enumerate(kinds) if z == "medonc"]
    lo = X[onc[-1]] if onc else np.zeros(len(F))
    v += [lo[ix["onc_" + c]] for c in ["prog", "stable", "resp", "mixed", "indet"]]
    perf = [X[k][ix["ecog"]] for k, z in enumerate(kinds) if z == "perf"]
    v += [perf[-1] if perf else 0.25, float(bool(perf))]
    lab = [X[k] for k, z in enumerate(kinds) if z == "lab"]
    v += [lab[-1][ix["lab_logratio"]] if lab else 0.0, lab[-1][ix["lab_high"]] if lab else 0.0,
          (lab[-1][ix["lab_logratio"]] - lab[-2][ix["lab_logratio"]]) if len(lab) > 1 else 0.0]
    v += [max([x[ix["tx_regimen"]] for x in X] + [0])]
    dx = [X[k] for k, z in enumerate(kinds) if z == "diagnosis"]
    v += ([dx[0][ix["stage_" + st]] for st in ["0", "I", "II", "III", "IV"]] + [dx[0][ix["age"]]]) if dx else [0] * 5 + [0.65]
    v += [float(c == s["cohort"]) for c in COHORTS]
    if dxonly:
        v += [np.log1p(q)]
    if timefeat:
        lastimg = d[img[-1]] if img else 0.0
        progd = [d[k] for k in img if X[k][ix["img_prog"]] > 0]
        txs = [d[k] for k, z in enumerate(kinds) if z == "tx_start"]
        v += [np.log1p(q - d[-2]), np.log1p(q - lastimg), np.log1p(q - progd[-1]) if progd else 0.0, float(bool(progd)),
              np.log1p(q - txs[-1]) if txs else 0.0, np.log1p(q), sum(1 for x in d[:-1] if q - x <= 90) / 20,
              np.log1p(np.median(np.diff(d[img])) if len(img) > 1 else 0.0)]
    return np.array(v, np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/path/to/work/htt/data/genie_bpc/genie.pkl")
    ap.add_argument("--task", choices=["nextprog", "mort12"], required=True)
    ap.add_argument("--test_center", required=True)
    ap.add_argument("--model", choices=["htt", "lr", "gbm"], default="htt")
    ap.add_argument("--time_band", default="clinical")
    ap.add_argument("--time_input", default="interval")
    ap.add_argument("--timefeat", action="store_true")
    ap.add_argument("--dxonly", action="store_true", help="LR: time since diagnosis as the only time feature")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_path = os.path.join(a.out, f"{a.task}_{a.name}__{a.test_center}__s{a.seed}.json")
    if os.path.exists(out_path):
        print("exists", out_path); return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    D = pickle.load(open(a.data, "rb"))
    F = D["F"]
    S = [s for s in D["samples"] if s["task"] == a.task]
    test = [s for s in S if s["center"] == a.test_center]
    trall = [s for s in S if s["center"] != a.test_center]
    rng = np.random.default_rng(a.seed)
    isval = rng.random(len(trall)) < 0.15
    tr = [s for s, v in zip(trall, isval) if not v]
    va = [s for s, v in zip(trall, isval) if v]
    probes = {k: probe(test, k, np.random.default_rng(100 + a.seed)) for k in ["shuffle", "nogap", "scale2", "shift365"]}
    res = dict(config=vars(a), n=dict(train=len(tr), val=len(va), test=len(test)), sets={})

    def record(name, p, ss):
        y = [s["label"] for s in ss]
        res["sets"][name] = dict(auroc=E.auc(y, p), pid=[s["pid"] for s in ss], y=y,
                                 cohort=[s["cohort"] for s in ss], p=[round(float(x), 5) for x in p])

    if a.model == "gbm":
        from sklearn.ensemble import HistGradientBoostingClassifier
        Xf = lambda ss: np.stack([summary_feats(s, F, a.timefeat, a.dxonly) for s in ss])
        Xtr, ytr = Xf(tr), [s["label"] for s in tr]
        best, bp = -1, None
        for lr_ in [0.03, 0.1]:
            for leaves in [15, 31]:
                m = HistGradientBoostingClassifier(learning_rate=lr_, max_leaf_nodes=leaves, max_iter=400,
                                                   early_stopping=True, validation_fraction=0.15, random_state=a.seed).fit(Xtr, ytr)
                v = E.auc([s["label"] for s in va], m.predict_proba(Xf(va))[:, 1])
                if v > best:
                    best, bp = v, (lr_, leaves)
        m = HistGradientBoostingClassifier(learning_rate=bp[0], max_leaf_nodes=bp[1], max_iter=400, early_stopping=True,
                                           validation_fraction=0.15, random_state=a.seed).fit(Xf(tr + va), [s["label"] for s in tr + va])
        record("test", m.predict_proba(Xf(test))[:, 1], test)
        for k, ss in probes.items():
            record(k, m.predict_proba(Xf(ss))[:, 1], ss)
        res["params"] = bp
    elif a.model == "lr":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xf = lambda ss: np.stack([summary_feats(s, F, a.timefeat, a.dxonly) for s in ss])
        sc = StandardScaler().fit(Xf(tr))
        best, bc = -1, 1.0
        for C in [0.01, 0.1, 1.0, 10.0]:
            m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(Xf(tr)), [s["label"] for s in tr])
            v = E.auc([s["label"] for s in va], m.predict_proba(sc.transform(Xf(va)))[:, 1])
            if v > best:
                best, bc = v, C
        m = LogisticRegression(C=bc, max_iter=2000).fit(sc.transform(Xf(tr + va)), [s["label"] for s in tr + va])
        record("test", m.predict_proba(sc.transform(Xf(test)))[:, 1], test)
        for k, ss in probes.items():
            record(k, m.predict_proba(sc.transform(Xf(ss)))[:, 1], ss)
        res["C"] = bc
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if dev.type == "cpu":
            torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
        mk = lambda ss: DS([to_item(s) for s in ss])
        E.seed_all(a.seed)
        model = HTT(reader="struct", time_band=a.time_band, time_input=a.time_input, n_types=len(COHORTS),
                    n_feat=len(F))
        fit = E.train(model, mk(tr), mk(va), dev, a.seed, epochs=a.epochs, patience=5, visit_w=0.0, log=lambda *_: None)
        res["fit"] = fit
        record("test", E.predict(model, mk(test), dev), test)
        for k, ss in probes.items():
            record(k, E.predict(model, mk(ss), dev), ss)
        res["device"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else f"cpu x{torch.get_num_threads()}"
    p0 = np.array(res["sets"]["test"]["p"])
    for k in probes:
        res["sets"][k]["mean_abs_shift"] = float(np.mean(np.abs(np.array(res["sets"][k]["p"]) - p0)))
    res["seconds"] = time.time() - t0
    json.dump(res, open(out_path, "w"))
    print("done", a.name, a.test_center, {k: round(v["auroc"], 4) for k, v in res["sets"].items()},
          {k: round(v.get("mean_abs_shift", 0), 3) for k, v in res["sets"].items()}, f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
