"""Zero-shot open-weight LLM readers on GENIE BPC timelines (run on DCC GPUs only; data never leave the cluster).
Each patient's curated events before the prediction day are written as text lines; the model answers A/B and the
score is the log-odds of the answer letters (no reasoning). Views: dates (day since diagnosis on every line),
nodates (events in order, no days), shuffle (inter-event gaps permuted within patient, order kept).
Scored per institution for comparison with the leave-one-institution-out models."""
import argparse
import json
import math
import os
import pickle
import time

import numpy as np

SYS = "You are an experienced medical oncologist reviewing a patient's record."
CUR = {"prog": "progressing", "stable": "stable", "resp": "responding", "mixed": "mixed", "indet": "not stated or indeterminate"}
IMP = {"yes": "cancer present", "no": "no evidence of cancer", "uncertain": "uncertain", "notmention": "cancer not mentioned"}
SCAN = {"ct": "CT", "mri": "MRI", "pet": "PET", "bone": "bone scan", "other": "other imaging"}
STG = ["0", "I", "II", "III", "IV"]
TYPE = {"NSCLC": "non-small-cell lung cancer", "CRC": "colorectal cancer", "BrCa": "breast cancer",
        "PANC": "pancreatic cancer", "Prostate": "prostate cancer", "Bladder": "bladder cancer"}


def line(kind, x, ix):
    on = lambda pre, keys: next((k for k in keys if x[ix[pre + k]] > 0), None)
    if kind == "imaging":
        return (f"{SCAN[on('scan_', SCAN) or 'other']} report: {CUR[on('img_', CUR) or 'indet']}, "
                f"impression {IMP[on('imgimp_', IMP) or 'notmention']}")
    if kind == "medonc":
        return f"oncologist note: {CUR[on('onc_', CUR) or 'indet']}, {IMP[on('oncimp_', IMP) or 'notmention']}"
    if kind == "pathology":
        return "pathology report"
    if kind in ("tx_start", "tx_end"):
        what = "radiation therapy" if x[ix["tx_radiation"]] > 0 else "systemic therapy"
        return f"{'started' if kind == 'tx_start' else 'stopped'} {what} (regimen {int(round(x[ix['tx_regimen']] * 10))})"
    if kind == "perf":
        return f"performance status ECOG {int(round(x[ix['ecog']] * 4))}"
    if kind == "lab":
        r = math.exp(x[ix["lab_logratio"]] * 3)
        return f"tumour marker {r:.1f} times the upper limit of normal"
    if kind == "diagnosis":
        s = on("stage_", STG)
        return f"diagnosis{', stage ' + s if s else ''}, age {int(round(x[ix['age']] * 100))}"
    return kind


def render(s, F, view, rng):
    ix = {f: i for i, f in enumerate(F)}
    d = s["days"].astype(float).copy()
    if view == "shuffle":
        g = rng.permutation(np.diff(d)); d = np.concatenate([[d[0]], d[0] + np.cumsum(g)])
    rows = []
    for k, kind in enumerate(s["kinds"][:-1]):
        txt = line(kind, s["feats"][k], ix)
        rows.append(f"Day {int(round(d[k]))}: {txt}" if view != "nodates" else f"- {txt}")
    q = d[-1]
    head = (f"Patient with {TYPE[s['cohort']]}. Events in chronological order"
            + (", with days counted from diagnosis" if view != "nodates" else " (dates not available)") + ":\n")
    if s["task"] == "nextprog":
        ask = (("On day %d" % int(round(q))) if view != "nodates" else "Next") + \
              ", the patient has a new imaging study. Will it be curated as progressing (worsening or enlarging disease)?"
        let = "Answer with a single letter: A if progressing, B if not."
    else:
        ask = (("At day %d after diagnosis" % int(round(q))) if view != "nodates" else "Now") + \
              ", the patient is alive. Will the patient die within the next 12 months?"
        let = "Answer with a single letter: A if the patient will die within 12 months, B if not."
    return head + "\n".join(rows) + "\n\n" + ask + "\n" + let


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/path/to/work/htt/data/genie_bpc/genie.pkl")
    ap.add_argument("--task", choices=["nextprog", "mort12"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--view", choices=["dates", "nodates", "shuffle"], default="dates")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_path = os.path.join(a.out, f"{a.task}_{a.name}.json")
    if os.path.exists(out_path):
        print("exists", out_path); return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    D = pickle.load(open(a.data, "rb")); F = D["F"]
    S = [s for s in D["samples"] if s["task"] == a.task]
    if a.limit:
        S = S[:a.limit]
    rng = np.random.default_rng(100)
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    llm = LLM(model=a.model, max_model_len=12288, gpu_memory_utilization=0.90, seed=0, max_num_seqs=128,
              enable_prefix_caching=True)
    def prompt(q):
        p = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
        if not p.rstrip().endswith("</think>") and "qwen3" in a.model.lower():
            p += "<think>\n\n</think>\n\n"
        return p
    prompts = [prompt(render(s, F, a.view, rng)) for s in S]
    outs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
    ida = {tok.convert_tokens_to_ids(t) for t in ["A", "ĠA"]} - {None, tok.unk_token_id}
    idb = {tok.convert_tokens_to_ids(t) for t in ["B", "ĠB"]} - {None, tok.unk_token_id}
    sc, miss = [], 0
    for o in outs:
        lp = o.outputs[0].logprobs[0]
        la = max([lp[i].logprob for i in ida if i in lp], default=None)
        lb = max([lp[i].logprob for i in idb if i in lp], default=None)
        if la is None and lb is None:
            miss += 1; sc.append(0.0)
        else:
            sc.append((la if la is not None else -50.0) - (lb if lb is not None else -50.0))
    from sklearn.metrics import roc_auc_score
    res = dict(config=vars(a), n=len(S), parse_fail=miss, centres={})
    for c in sorted({s["center"] for s in S}):
        m = [i for i, s in enumerate(S) if s["center"] == c]
        y = [S[i]["label"] for i in m]
        res["centres"][c] = dict(auroc=float(roc_auc_score(y, [sc[i] for i in m])), n=len(m),
                                 pid=[S[i]["pid"] for i in m], y=y, score=[round(sc[i], 5) for i in m])
    res["auroc_mean_centres"] = float(np.mean([v["auroc"] for v in res["centres"].values()]))
    import torch
    res["seconds"] = time.time() - t0; res["device"] = torch.cuda.get_device_name(0)
    res["example_prompt"] = render(S[0], F, a.view, np.random.default_rng(0))[:1500]
    json.dump(res, open(out_path, "w"))
    print("done", a.task, a.name, round(res["auroc_mean_centres"], 4), {c: round(v["auroc"], 3) for c, v in res["centres"].items()},
          "miss", miss, f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
