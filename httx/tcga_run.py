"""One TCGA grade run (HTT, one report per patient) with switchable label scheme, type source,
redaction, encoder and window. Writes per-patient predictions for the in-distribution test set
and every held-out type."""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from httx import engine as E
from httx.models import HTT, BIOMEDBERT, GENERAL_BERT

from httx.tcga_data import TYPES14, make_frame, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--labels", choices=["paper", "corr", "reg"], default="corr")
    ap.add_argument("--types", choices=["regex", "true"], default="true")
    ap.add_argument("--text", choices=["raw", "redacted", "redacted_all"], default="raw")
    ap.add_argument("--encoder", choices=["biomedbert", "bert", "random"], default="biomedbert")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--held", nargs="+", default=["THCA", "SARC", "LUSC"])
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--restrict_reg", action="store_true")
    ap.add_argument("--eval_labels", choices=["paper", "corr", "reg"], default=None)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_path = os.path.join(a.out, f"{a.name}__s{a.seed}.json")
    if os.path.exists(out_path):
        print("exists", out_path); return
    lock = out_path + ".running"
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 3 * 3600:
        print("running elsewhere", out_path); return
    os.makedirs(a.out, exist_ok=True)
    open(lock, "w").write(os.environ.get("SLURM_JOB_ID", "") + " " + os.uname().nodename)
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    df = pd.read_pickle(a.data)
    if a.restrict_reg:                      # size-matched: only patients that also have a registry label
        df = df[df.label_reg.notna()]
    d = make_frame(df, a.labels, a.types, a.text)
    tr, va, te, ho = split(d, set(a.held))
    if a.eval_labels:                       # score the same patients against another label source
        alt = df.set_index("patient_id")[{"paper": "label_paper", "corr": "label_corr", "reg": "label_reg"}[a.eval_labels]]
    seen = sorted(tr.cancer_type.unique())
    type_ids = {t: i for i, t in enumerate(seen)}          # held-out types -> reserved index len(seen)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(GENERAL_BERT if a.encoder == "bert" else BIOMEDBERT)
    mk = lambda x: E.Patients(x, tok, type_ids, a.max_len, 1, unseen_id=len(seen))
    ds_tr, ds_va, ds_te = mk(tr), mk(va), mk(te)
    ds_ho = {t: mk(ho[ho.cancer_type == t]) for t in a.held if (ho.cancer_type == t).any()}

    E.seed_all(a.seed)
    model = HTT(reader="text", encoder=a.encoder, time_band="orig", time_input="elapsed",
                n_types=len(seen))
    lr = a.lr if a.lr is not None else (1e-4 if a.encoder == "random" else 2e-4)
    fit = E.train(model, ds_tr, ds_va, dev, a.seed, epochs=a.epochs, patience=a.patience, lr=lr, visit_w=0.3)

    res = dict(config=vars(a), lr=lr, n=dict(train=len(ds_tr), val=len(ds_va), test=len(ds_te),
                                             held={t: len(v) for t, v in ds_ho.items()}),
               fit=fit, sets={})
    for name, ds in [("test", ds_te)] + [(f"held_{t}", v) for t, v in ds_ho.items()]:
        p = E.predict(model, ds, dev)
        y = [it["plab"] for it in ds.items]
        res["sets"][name] = dict(auroc=E.auc(y, p), pid=[it["pid"] for it in ds.items], y=y,
                                 ctype=[it["ctype"] for it in ds.items],
                                 p=[round(float(x), 5) for x in p])
    if a.eval_labels:
        for name, v in res["sets"].items():
            ya = [alt.get(pid, np.nan) for pid in v["pid"]]
            m = [i for i, x in enumerate(ya) if x == x]
            v["y_alt"] = [None if x != x else int(x) for x in ya]
            v["auroc_alt"] = E.auc([int(ya[i]) for i in m], [v["p"][i] for i in m]) if m else float("nan")
            v["n_alt"] = len(m)
    held_aucs = [v["auroc"] for k, v in res["sets"].items() if k.startswith("held_") and v["auroc"] == v["auroc"]]
    res["held_macro"] = float(np.mean(held_aucs)) if held_aucs else float("nan")
    res["seconds"] = time.time() - t0
    res["device"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else f"cpu x{torch.get_num_threads()}"
    json.dump(res, open(out_path, "w"))
    os.remove(lock)
    print("done", out_path, "test", round(res["sets"]["test"]["auroc"], 4), "held_macro",
          round(res["held_macro"], 4), {k: round(v["auroc"], 3) for k, v in res["sets"].items()},
          f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
