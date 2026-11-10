"""Zero-shot open-weight LLM readers (vLLM, local GPU; no data leaves the cluster).

Tasks
  tcga_grade : one pathology report -> P(high grade). Corrected labels, true TCGA types, the same
               in-distribution test set and held-out types as the HTT runs. --text raw | redacted.
  growth     : a patient's serial imaging reports -> P(rapid progression), growth benchmark test180
               and PANC300. --view dates | nodates | last | redate21 | redate90.
Modes
  logprob : no reasoning; P = softmax over the answer letters A/B from the first-token logprobs.
  reason  : the model reasons freely, then states a probability (0-100) on a final line.
"""
import argparse
import json
import math
import os
import re
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from sklearn.metrics import roc_auc_score

MAX_CHARS = 12000
ALL_TYPES = False          # TCGA reports: keep the first 12k characters (>99% of reports are shorter)

GRADE_SYS = "You are an expert surgical pathologist."
GRADE_Q = ("Read the pathology report below and decide whether the tumour is HIGH grade (grade 3 or 4, poorly "
           "differentiated or undifferentiated, or high grade) or LOW grade (grade 1, well differentiated, or low "
           "grade).{mask}\n\nReport (cancer type {ctype}):\n\"\"\"\n{text}\n\"\"\"\n\n")
GRADE_MASK = (" Words that state the grade or the degree of differentiation have been replaced by [MASK]; infer "
              "the grade from the other findings in the report.")

GROWTH_SYS = "You are an oncologist reviewing a patient's serial imaging reports."
GROWTH_Q = ("In this cohort every patient has one measurable tumour. A RAPIDLY progressing tumour doubles its "
            "diameter in about 120 days; a SLOWLY progressing tumour doubles its diameter in about 600 days "
            "(both vary by roughly 35% between patients). About 40% of patients are rapid progressors. Reported "
            "sizes carry about 8% measurement noise and are rounded to 0.1 cm.{when}\n\n{reports}\n\n"
            "Is this patient's tumour rapidly progressing?")
WHEN = {"dates": " Each report is labelled with the day of the examination, counted from the first examination.",
        "nodates": " The reports are in chronological order, but the dates of the examinations are not available.",
        "last": " Only the most recent report is available; its date and earlier reports are not.",
        "redate21": " Each report is labelled with the day of the examination, counted from the first examination.",
        "redate90": " Each report is labelled with the day of the examination, counted from the first examination."}

ASK_LETTER = {"tcga_surv": "Answer with a single letter (A, B, C, D or E).","tcga_grade": "Answer with a single letter: A if HIGH grade, B if LOW grade.",
              "growth": "Answer with a single letter: A if rapidly progressing, B if slowly progressing."}
ASK_PROB = {"tcga_grade": ("Think it through, then end your answer with one line exactly of the form "
                           "'Probability of high grade: NN%'."),
            "growth": ("Think it through, then end your answer with one line exactly of the form "
                       "'Probability of rapid progression: NN%'.")}


def tcga_items(data, text, labels="corr"):
    from httx.tcga_data import make_frame, split
    df = pd.read_pickle(data)
    d = make_frame(df, labels, "true", text)
    _, _, te, ho = split(d, {"THCA", "SARC", "LUSC"})
    out = []
    groups = [("test", te)] + [(f"held_{t}", ho[ho.cancer_type == t]) for t in ["THCA", "SARC", "LUSC"]]
    if ALL_TYPES:
        groups = [(f"type_{t}", d[d.cancer_type == t]) for t in sorted(d.cancer_type.unique())]
    for setname, sub in groups:
        for r in sub.itertuples():
            body = re.sub(r"^\[CANCER:\w+\]\s*", "", r.report_text)[:MAX_CHARS]
            q = GRADE_Q.format(mask=GRADE_MASK if text != "raw" else "", ctype=r.cancer_type, text=body)
            out.append(dict(set=setname, pid=r.patient_id, ctype=r.cancer_type, y=int(r.label), q=q))
    return out


SURV_SYS = "You are an experienced oncologist."
SURV_Q = ("Below is the diagnostic pathology report of a patient with {ctype} ({name}). Patient age at diagnosis: "
          "{age}.{mask}\n\nReport:\n\"\"\"\n{text}\n\"\"\"\n\nEstimate the risk that this patient dies within 5 years "
          "of diagnosis.\nA: below 10%\nB: 10-30%\nC: 30-50%\nD: 50-70%\nE: above 70%\n")
SURV_MASK = (" Words stating tumour grade, differentiation, stage or TNM category have been replaced by [MASK].")
SURV_W = {"A": 0.05, "B": 0.20, "C": 0.40, "D": 0.60, "E": 0.85}
TCGA_NAMES = {"ACC": "adrenocortical carcinoma", "BLCA": "bladder urothelial carcinoma", "BRCA": "breast invasive carcinoma",
              "CESC": "cervical cancer", "CHOL": "cholangiocarcinoma", "COAD": "colon adenocarcinoma",
              "DLBC": "diffuse large B-cell lymphoma", "ESCA": "oesophageal carcinoma", "GBM": "glioblastoma",
              "HNSC": "head and neck squamous cell carcinoma", "KICH": "kidney chromophobe",
              "KIRC": "kidney renal clear cell carcinoma", "KIRP": "kidney renal papillary cell carcinoma",
              "LGG": "lower grade glioma", "LIHC": "liver hepatocellular carcinoma", "LUAD": "lung adenocarcinoma",
              "LUSC": "lung squamous cell carcinoma", "MESO": "mesothelioma", "OV": "ovarian serous cystadenocarcinoma",
              "PAAD": "pancreatic adenocarcinoma", "PCPG": "pheochromocytoma and paraganglioma",
              "PRAD": "prostate adenocarcinoma", "READ": "rectum adenocarcinoma", "SARC": "sarcoma",
              "SKCM": "skin cutaneous melanoma", "STAD": "stomach adenocarcinoma", "TGCT": "testicular germ cell tumour",
              "THCA": "thyroid carcinoma", "THYM": "thymoma", "UCEC": "uterine corpus endometrial carcinoma",
              "UCS": "uterine carcinosarcoma", "UVM": "uveal melanoma"}


def surv_items(data, text):
    df = pd.read_pickle(data.replace(".pkl", "_cdr.pkl") if not data.endswith("_cdr.pkl") else data)
    d = df[df["OS.time"].notna() & df.type_true.notna()]
    col = "text" if text == "raw" else "text_redacted_all"
    out = []
    for r in d.itertuples():
        age = "unknown" if r.age != r.age else f"{int(r.age)} years"
        q = SURV_Q.format(ctype=r.type_true, name=TCGA_NAMES.get(r.type_true, r.type_true), age=age,
                          mask=SURV_MASK if text != "raw" else "", text=str(getattr(r, col))[:MAX_CHARS])
        out.append(dict(set="all", pid=r.patient_id, ctype=r.type_true, y=int(r.OS), q=q))
    return out


def growth_items(cache, view):
    path = os.path.join(cache, "growth_n70.pkl")        # same cached benchmark as the HTT runs
    if os.path.exists(path):
        sets = pd.read_pickle(path)
    else:
        from httx.growth import build_benchmark
        sets = build_benchmark(n_train_per_type=70, seed=42)
    out = []
    for setname in ["test180", "PANC300"]:
        for pid, g in sets[setname].groupby("patient_id", sort=False):
            g = g.sort_values("visit_index").iloc[:10]
            txt = [re.sub(r"^\[CANCER:\w+\]\s*", "", t) for t in g.report_text]
            days = g.days_from_start.values
            if view == "redate21":
                days = np.arange(len(txt)) * 21.0
            elif view == "redate90":
                days = np.arange(len(txt)) * 90.0
            if view == "last":
                reps = f"Most recent report: {txt[-1]}"
            elif view == "nodates":
                reps = "\n".join(f"Report {k + 1}: {t}" for k, t in enumerate(txt))
            else:
                reps = "\n".join(f"Day {int(round(d))}: {t}" for d, t in zip(days, txt))
            q = GROWTH_Q.format(when=WHEN[view], reports=reps)
            out.append(dict(set=setname, pid=pid, ctype=g.cancer_type.iloc[0], y=int(g.label.iloc[0]), q=q,
                            n_visits=len(txt)))
    return out


STRICT_RE = re.compile(r"probability of (?:rapid progression|high grade)\W{0,6}(\d{1,3}(?:\.\d+)?)\s*%", re.I)
PROB_RE = re.compile(r"probability[^0-9\n]{0,40}?(\d{1,3}(?:\.\d+)?)\s*%", re.I)


def parse_prob(text):
    m = PROB_RE.findall(text or "")
    if not m:
        m = re.findall(r"(\d{1,3}(?:\.\d+)?)\s*%", text or "")
    if not m:
        return None
    v = float(m[-1])
    return min(max(v / 100.0, 0.0), 1.0)


def auc(y, p):
    y = np.asarray(y)
    return float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["tcga_grade", "growth", "tcga_surv"], required=True)
    ap.add_argument("--labels", choices=["corr", "reg"], default="corr")
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["logprob", "reason"], default="logprob")
    ap.add_argument("--text", choices=["raw", "redacted", "redacted_all"], default="raw")
    ap.add_argument("--view", default="dates")
    ap.add_argument("--data", default="/path/to/work/htt/data/tcga.pkl")
    ap.add_argument("--cache", default="/path/to/work/htt/cache")
    ap.add_argument("--max_tokens", type=int, default=6144)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sets", nargs="*", default=None, help="restrict to these evaluation sets (e.g. test180)")
    ap.add_argument("--max_model_len", type=int, default=16384)
    ap.add_argument("--all_types", action="store_true", help="tcga_grade: score every labelled patient, by type")
    ap.add_argument("--strict", action="store_true", help="reasoning: count truncated or line-less outputs as failures")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out_path = os.path.join(a.out, f"{a.name}.json")
    if os.path.exists(out_path):
        print("exists", out_path); return
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()

    global ALL_TYPES
    ALL_TYPES = a.all_types
    if a.task == "tcga_grade":
        items = tcga_items(a.data if a.labels == "corr" else a.data.replace(".pkl", "_cdr.pkl"), a.text, a.labels)
    elif a.task == "tcga_surv":
        items = surv_items(a.data, a.text)
    else:
        items = growth_items(a.cache, a.view)
    if a.sets:
        items = [it for it in items if it["set"] in a.sets]
    if a.limit:
        items = items[:a.limit]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    is_oss = "gpt-oss" in a.model
    llm = LLM(model=a.model, max_model_len=a.max_model_len, gpu_memory_utilization=0.90, seed=0, max_num_seqs=128,
              enable_prefix_caching=True, trust_remote_code=False)
    sysmsg = {"tcga_grade": GRADE_SYS, "growth": GROWTH_SYS, "tcga_surv": SURV_SYS}[a.task]

    def prompt(q, ask, think):
        msgs = [{"role": "system", "content": sysmsg}, {"role": "user", "content": q + ask}]
        kw = {"enable_thinking": think}
        if is_oss:
            kw = {"reasoning_effort": "medium"}
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)
        thinker = "qwen3" in a.model.lower() or "ii-medical" in a.model.lower()
        if not think and thinker and not p.rstrip().endswith("</think>"):
            p += "<think>\n\n</think>\n\n"          # templates that ignore enable_thinking=False (e.g. II-Medical)
        if not think and "ii-medical" in a.model.lower():
            p += "Answer:"                              # II-Medical does not open with the letter otherwise
        return p

    if a.mode == "logprob":
        prompts = [prompt(it["q"], ASK_LETTER[a.task], False) for it in items]
        sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        outs = llm.generate(prompts, sp)
        ids_a = {tok.convert_tokens_to_ids(t) for t in ["A", "ĠA"]} - {None, tok.unk_token_id}
        ids_b = {tok.convert_tokens_to_ids(t) for t in ["B", "ĠB"]} - {None, tok.unk_token_id}
        miss = 0
        if a.task == "tcga_surv":
            lid = {L: {tok.convert_tokens_to_ids(t) for t in [L, "Ġ" + L]} - {None, tok.unk_token_id} for L in SURV_W}
            for it, o in zip(items, outs):
                lp = o.outputs[0].logprobs[0]
                pr = {L: max([math.exp(lp[i].logprob) for i in ids if i in lp], default=0.0) for L, ids in lid.items()}
                z = sum(pr.values())
                if z == 0:
                    it["p"], miss = 0.4, miss + 1
                else:
                    it["p"] = sum(SURV_W[L] * v for L, v in pr.items()) / z
                it["s"] = it["p"]
            items_done = True
        for it, o in zip(items if a.task != "tcga_surv" else [], outs):
            lp = o.outputs[0].logprobs[0]
            la = max([lp[i].logprob for i in ids_a if i in lp], default=None)
            lb = max([lp[i].logprob for i in ids_b if i in lp], default=None)
            if la is None and lb is None:
                sc, miss = 0.0, miss + 1
            else:
                sc = (la if la is not None else -50.0) - (lb if lb is not None else -50.0)
            it["s"] = sc                                    # log-odds A vs B (unsaturated ranking score)
            it["p"] = 1.0 / (1.0 + math.exp(-max(min(sc, 50), -50)))
        parse_fail = miss
    else:
        prompts = [prompt(it["q"], ASK_PROB[a.task], True) for it in items]
        sp = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=a.max_tokens, seed=0)
        outs = llm.generate(prompts, sp)
        parse_fail = 0
        truncated = 0
        for it, o in zip(items, outs):
            txt = o.outputs[0].text
            fin = o.outputs[0].finish_reason
            closed = ("</think>" in txt) if not is_oss else ("final" in txt)
            final = txt.split("</think>")[-1] if "</think>" in txt else txt
            if is_oss and "final" in final:
                final = final.split("final")[-1]
            if a.strict:
                m = STRICT_RE.findall(final) if (closed and fin != "length") else []
                p = min(max(float(m[-1]) / 100.0, 0.0), 1.0) if m else None
            else:
                p = parse_prob(final)
            it["finish"] = fin
            it["completed"] = p is not None
            truncated += int(fin == "length")
            if p is None:
                p, parse_fail = 0.5, parse_fail + 1
            it["p"] = p
            it["s"] = p
            it["answer"] = final[-600:]
            it["n_gen_tokens"] = len(o.outputs[0].token_ids)

    res = dict(config=vars(a), n=len(items), parse_fail=parse_fail, sets={})
    if a.mode == "reason":
        res["truncated"] = truncated
    for s in sorted({it["set"] for it in items}):
        sub = [it for it in items if it["set"] == s]
        res["sets"][s] = dict(auroc=auc([i["y"] for i in sub], [i["s"] for i in sub]),
                              score=[round(float(i["s"]), 6) for i in sub],
                              pid=[i["pid"] for i in sub], y=[i["y"] for i in sub],
                              ctype=[i["ctype"] for i in sub], p=[round(i["p"], 5) for i in sub])
        if a.mode == "reason":
            res["sets"][s]["answer_tail"] = [i["answer"] for i in sub]
            res["sets"][s]["n_gen_tokens"] = [i["n_gen_tokens"] for i in sub]
            res["sets"][s]["completed"] = [i["completed"] for i in sub]
            done = [i for i in sub if i["completed"]]
            res["sets"][s]["auroc_completed"] = auc([i["y"] for i in done], [i["s"] for i in done]) if done else float("nan")
            res["sets"][s]["n_completed"] = len(done)
    held = [v["auroc"] for k, v in res["sets"].items() if k.startswith("held_") and v["auroc"] == v["auroc"]]
    if held:
        res["held_macro"] = float(np.mean(held))
    import torch
    res["seconds"] = time.time() - t0
    res["device"] = torch.cuda.get_device_name(0)
    json.dump(res, open(out_path, "w"))
    print("done", out_path, {k: round(v["auroc"], 4) for k, v in res["sets"].items()},
          "parse_fail", parse_fail, f"{res['seconds']:.0f}s")


if __name__ == "__main__":
    main()
