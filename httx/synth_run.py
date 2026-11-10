"""One training run on a synthetic benchmark; writes a JSON with metrics, per-patient
predictions on every evaluation set and the re-dating probe on test180."""
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
from httx.models import HTT, Baseline, BIOMEDBERT

TYPE_IDS = {"NSCLC": 0, "BrCa": 1, "CRC": 2, "PANC": 3, "Prostate": 4}   # OncoLocus CANCER_TYPE_TO_ID
TRAIN_TYPES = ["NSCLC", "BrCa", "CRC", "Prostate"]


def _load(cache):
    try:
        return pd.read_pickle(cache)
    except Exception:
        return None


def _save(obj, cache):
    tmp = f"{cache}.{os.getpid()}.tmp"
    pd.to_pickle(obj, tmp)
    os.replace(tmp, cache)          # atomic: concurrent tasks never see a partial file


def phrase_benchmark(cache):
    """Released phrase benchmark: OncoLocus generator, 100 patients/type, seed 42, split as in
    build_datasets (per-type shuffle, 70/15/15, PANC held out). Extra test patients from the same
    generator with the disjoint seed 1042."""
    if os.path.exists(cache) and _load(cache) is not None:
        return _load(cache)
    from httx.oncolocus_ref import generate_synthetic_dataset
    data = generate_synthetic_dataset(n_patients_per_type=100, seed=42)
    random.seed(42)
    sets = {"train": [], "val": [], "test60": []}
    for ct, df in data.items():
        if ct == "PANC":
            continue
        ids = df.patient_id.unique().tolist()
        random.shuffle(ids)
        n = len(ids); a, b = int(n * 0.7), int(n * 0.7) + int(n * 0.15)
        sets["train"].append(df[df.patient_id.isin(ids[:a])])
        sets["val"].append(df[df.patient_id.isin(ids[a:b])])
        sets["test60"].append(df[df.patient_id.isin(ids[b:])])
    sets = {k: pd.concat(v, ignore_index=True) for k, v in sets.items()}
    sets["PANC100"] = data["PANC"]
    extra = generate_synthetic_dataset(n_patients_per_type=200, seed=1042)
    ex = {}
    for ct, df in extra.items():
        df = df.copy(); df["patient_id"] = "X" + df.patient_id
        ids = df.patient_id.unique().tolist()
        ex[ct] = (df, ids)
    add15 = pd.concat([d[d.patient_id.isin(i[:15])] for c, (d, i) in ex.items() if c != "PANC"])
    add30 = pd.concat([d[d.patient_id.isin(i[:30])] for c, (d, i) in ex.items() if c != "PANC"])
    sets["test120"] = pd.concat([sets["test60"], add15], ignore_index=True)
    sets["test180"] = pd.concat([sets["test60"], add30], ignore_index=True)
    d, i = ex["PANC"]
    sets["PANC300"] = pd.concat([sets["PANC100"], d[d.patient_id.isin(i[:200])]], ignore_index=True)
    _save(sets, cache)
    return sets


def growth_benchmark(cache, n_train):
    if os.path.exists(cache) and _load(cache) is not None:
        return _load(cache)
    from httx.growth import build_benchmark
    sets = build_benchmark(n_train_per_type=n_train, seed=42)
    _save(sets, cache)
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", choices=["phrase", "growth"], required=True)
    ap.add_argument("--model", choices=["htt", "baseline"], default="htt")
    ap.add_argument("--reader", choices=["text", "struct"], default="text")
    ap.add_argument("--time_band", default="orig")
    ap.add_argument("--time_input", default="elapsed")
    ap.add_argument("--numbers", action="store_true")
    ap.add_argument("--visit_w", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--n_train", type=int, default=70, help="training patients per type (growth only)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    a = ap.parse_args()

    out_path = os.path.join(a.out, f"{a.name}__s{a.seed}.json")
    if os.path.exists(out_path):
        print("exists", out_path); return
    lock = out_path + ".running"
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 3 * 3600:
        print("running elsewhere", out_path); return
    os.makedirs(a.out, exist_ok=True)
    open(lock, "w").write(os.environ.get("SLURM_JOB_ID", "") + " " + os.uname().nodename)
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.cache, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()

    if a.bench == "phrase":
        sets = phrase_benchmark(os.path.join(a.cache, "phrase.pkl"))
    else:
        sets = growth_benchmark(os.path.join(a.cache, f"growth_n{a.n_train}.pkl"), a.n_train)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BIOMEDBERT) if a.reader == "text" else None
    feats = a.bench == "growth" and (a.numbers or a.reader == "struct")
    ds = {k: E.Patients(v, tok, TYPE_IDS, 128, 10, with_feats=feats) for k, v in sets.items()}

    E.seed_all(a.seed)
    if a.model == "baseline":
        model = Baseline()
    else:
        model = HTT(reader=a.reader, time_band=a.time_band, time_input=a.time_input, numbers=a.numbers,
                    n_types=len(TYPE_IDS))
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    fit = E.train(model, ds["train"], ds["val"], dev, a.seed, epochs=a.epochs, patience=a.patience,
                  visit_w=a.visit_w)

    res = dict(config=vars(a), params=dict(trainable=n_tr, total=n_all), fit=fit, sets={}, redate={})
    for k in ["val", "test60", "test120", "test180", "PANC100", "PANC300"]:
        p = E.predict(model, ds[k], dev)
        y = [it["plab"] for it in ds[k].items]
        res["sets"][k] = dict(auroc=E.auc(y, p), pid=[it["pid"] for it in ds[k].items], y=y,
                              p=[round(float(x), 5) for x in p],
                              ctype=[it["ctype"] for it in ds[k].items])
    # re-dating probe: same text, visit k placed on day k*g
    for g in [21, 42, 66, 90, 0]:
        p = E.predict(model, ds["test180"], dev, days_override=lambda n, g=g: torch.arange(n).float() * g)
        res["redate"][str(g)] = [round(float(x), 5) for x in p]
    if hasattr(model, "time"):
        res["freq_scale_mean"] = float(model.time.freq_scale.abs().mean())
    res["seconds"] = time.time() - t0
    res["device"] = torch.cuda.get_device_name(0) if dev.type == "cuda" else f"cpu x{torch.get_num_threads()}"
    with open(out_path, "w") as f:
        json.dump(res, f)
    os.remove(lock)
    print("done", out_path, {k: round(v["auroc"], 4) for k, v in res["sets"].items()}, f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
