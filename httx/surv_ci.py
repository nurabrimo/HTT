"""Consistent point estimate and CI for (registry+text) - registry within-type C, from the saved out-of-fold
risks of repeat 0: type-stratified patient bootstrap (1,000). Usage: surv_ci.py <surv_dir> <out.json>"""
import glob, json, os, sys
import numpy as np
from multiprocessing import Pool
from sksurv.metrics import concordance_index_censored


def strat(e, t, r, typ):
    num = den = 0.0
    for ty in np.unique(typ):
        m = typ == ty
        if m.sum() < 5 or e[m].sum() < 2:
            continue
        c = concordance_index_censored(e[m].astype(bool), t[m], r[m])
        num += c[1] + 0.5 * c[3]; den += c[1] + c[2] + c[3]
    return num / den


def one(f):
    r = json.load(open(f))
    e, t, typ = np.array(r["e"]), np.array(r["t"]), np.array(r["type"])
    R = {k: np.array(v) for k, v in r["oof_risk_repeat0"].items()}
    rng = np.random.default_rng(0)
    idx_t = {ty: np.where(typ == ty)[0] for ty in np.unique(typ)}
    out = {}
    for a, b in [("clin+text", "clin"), ("text", "clin")]:
        point = strat(e, t, R[a], typ) - strat(e, t, R[b], typ)
        bs = []
        for _ in range(1000):
            i = np.concatenate([rng.choice(v, len(v)) for v in idx_t.values()])
            bs.append(strat(e[i], t[i], R[a][i], typ[i]) - strat(e[i], t[i], R[b][i], typ[i]))
        out[f"{a} - {b}"] = dict(diff=float(point), lo=float(np.percentile(bs, 2.5)), hi=float(np.percentile(bs, 97.5)))
    out["c_strat_rep0"] = {k: float(strat(e, t, v, typ)) for k, v in R.items()}
    return os.path.basename(f)[:-5], out


if __name__ == "__main__":
    files = [f for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))) if "_emb_" not in f]
    with Pool(min(8, len(files))) as p:
        res = dict(p.map(one, files))
    json.dump(res, open(sys.argv[2], "w"), indent=1)
    for k, v in res.items():
        print(k, {a: (round(b["diff"], 3), round(b["lo"], 3), round(b["hi"], 3)) for a, b in v.items() if " - " in a})
