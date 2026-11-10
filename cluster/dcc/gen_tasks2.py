"""Task lists for the open-LLM, registry-label and survival experiments (E15-E18)."""
R = "/path/to/work/htt/runs"
M = {"q8": "Qwen/Qwen3-8B", "med8": "Intelligent-Internet/II-Medical-8B", "q14": "Qwen/Qwen3-14B",
     "q27": "Qwen/Qwen3.6-27B-FP8", "oss20": "openai/gpt-oss-20b"}
small, big = [], []          # small: fits 24-32 GB; big: needs 48 GB
for k in ["q8", "med8", "q14", "q27"]:
    L = small if k in ("q8", "med8") else big
    for v in ["dates", "nodates", "last", "redate21", "redate90"]:
        L.append(f"httx/llm_run.py --task growth --model {M[k]} --mode logprob --view {v} --name growth_{k}_logprob_{v} --out {R}/llm")
    for t in ["raw", "redacted"]:
        L.append(f"httx/llm_run.py --task tcga_grade --model {M[k]} --mode logprob --text {t} --name grade_{k}_logprob_{t} --out {R}/llm")
for k in ["q8", "q14", "q27", "oss20"]:
    L = small if k == "q8" else big
    for v in ["dates", "nodates", "redate21", "redate90"]:
        L.append(f"httx/llm_run.py --task growth --model {M[k]} --mode reason --view {v} --name growth_{k}_reason_{v} --out {R}/llm")
for k in ["q14", "q27", "oss20"]:
    for t in ["raw", "redacted"]:
        big.append(f"httx/llm_run.py --task tcga_grade --model {M[k]} --mode reason --text {t} --name grade_{k}_reason_{t} --out {R}/llm")
for t in ["raw", "redacted"]:
    big.append(f"httx/llm_run.py --task tcga_grade --labels reg --model {M['q27']} --mode logprob --text {t} --name gradereg_q27_logprob_{t} --out {R}/llm")
for k in ["q14", "q27"]:
    for t in ["raw", "redacted_all"]:
        big.append(f"httx/llm_run.py --task tcga_surv --model {M[k]} --mode logprob --text {t} --name tcga_surv_{k}_{t} --out {R}/llm")
open("dcc/tasks_llm_small.txt", "w").write("\n".join(small) + "\n")
open("dcc/tasks_llm_big.txt", "w").write("\n".join(big) + "\n")

D = "/path/to/work/htt/data/tcga_cdr.pkl"
reg = []
for lab, ev in [("reg", "corr"), ("corr", "reg")]:
    for t in ["raw", "redacted"]:
        for s in [1, 2, 3, 4, 5]:
            reg.append(f"httx/tcga_run.py --data {D} --labels {lab} --eval_labels {ev} --text {t} --seed {s} "
                       f"--name lab{lab}_{t} --out {R}/tcga_reg")
open("dcc/tasks_tcga_reg.txt", "w").write("\n".join(reg) + "\n")

surv = []
for ep in ["OS", "PFI"]:
    for t in ["raw", "redacted_all"]:
        for f in ["tfidf", "emb"]:
            surv.append(f"httx/surv_run.py --endpoint {ep} --feat {f} --text {t} --name surv_{ep}_{f}_{t} --out {R}/surv")
        for k in ["q14", "q27"]:
            surv.append(f"httx/surv_run.py --endpoint {ep} --feat llm --llm_model {k} --text {t} --name surv_{ep}_llm{k}_{t} --out {R}/surv")
open("dcc/tasks_surv.txt", "w").write("\n".join(surv) + "\n")
print(len(small), len(big), len(reg), len(surv))
