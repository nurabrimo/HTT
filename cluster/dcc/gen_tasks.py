"""Write the task lists for all new experiments (one python argument line per run)."""
R = "/path/to/work/htt/runs"; C = "/path/to/work/htt/cache"; D = "/path/to/work/htt/data/tcga.pkl"
SEEDS10 = [42, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SEEDS5 = [42, 1, 2, 3, 4]
S = "httx/synth_run.py"; T = "httx/tcga_run.py"

GROWTH = {   # name: extra args
    "baseline": "--model baseline",
    "orig_elapsed": "--time_band orig --time_input elapsed",
    "none": "--time_input none",
    "order": "--time_input order",
    "orig_interval": "--time_band orig --time_input interval",
    "clin_elapsed": "--time_band clinical --time_input elapsed",
    "clin_interval": "--time_band clinical --time_input interval",
    "clin_both": "--time_band clinical --time_input both",
    "orig_elapsed_num": "--time_band orig --time_input elapsed --numbers",
    "none_num": "--time_input none --numbers",
    "clin_interval_num": "--time_band clinical --time_input interval --numbers",
    "struct_none": "--reader struct --time_input none",
    "struct_orig_elapsed": "--reader struct --time_band orig --time_input elapsed",
    "struct_clin_interval": "--reader struct --time_band clinical --time_input interval",
}
lines = {"growth": [], "scale": [], "long": [], "phrase": [], "tcga": [], "loto": []}
for s in SEEDS10:
    for n, x in GROWTH.items():
        lines["growth"].append(f"{S} --bench growth {x} --seed {s} --name {n} --out {R}/growth --cache {C}")
for ntr in [280, 700, 1750]:
    for s in SEEDS5:
        for n in ["orig_elapsed", "none", "clin_interval", "clin_interval_num", "struct_clin_interval", "struct_none"]:
            lines["scale"].append(f"{S} --bench growth {GROWTH[n]} --n_train {ntr} --seed {s} --name {n}_n{ntr} --out {R}/scale --cache {C}")
for s in SEEDS10:
    for n in ["orig_elapsed", "none", "clin_interval", "baseline"]:
        lines["long"].append(f"{S} --bench growth {GROWTH[n]} --epochs 20 --patience 0 --seed {s} --name {n}_ep20 --out {R}/long --cache {C}")
    for n, x in [("baseline", "--model baseline"), ("orig_elapsed", "--time_band orig --time_input elapsed"),
                 ("clin_interval", "--time_band clinical --time_input interval")]:
        lines["phrase"].append(f"{S} --bench phrase {x} --seed {s} --name {n} --out {R}/phrase --cache {C}")
TC = [("ref_paper", "--labels paper --types regex --text raw --encoder biomedbert", SEEDS10)]
for enc in ["biomedbert", "bert", "random"]:
    for txt in ["raw", "redacted"]:
        TC.append((f"corr_{txt}_{enc}", f"--labels corr --types true --text {txt} --encoder {enc}", SEEDS5))
for txt in ["raw", "redacted"]:
    TC.append((f"corr_{txt}_biomedbert_512", f"--labels corr --types true --text {txt} --encoder biomedbert --max_len 512", SEEDS5))
TC.append(("paperlab_truetype", "--labels paper --types true --text raw --encoder biomedbert", SEEDS5))
for n, x, seeds in TC:
    for s in seeds:
        lines["tcga"].append(f"{T} --data {D} {x} --seed {s} --name {n} --out {R}/tcga")
for t in ["BRCA", "COAD", "HNSC", "LIHC", "LUAD", "SARC", "UCEC", "LUSC"]:
    for txt in ["raw", "redacted"]:
        for s in [42, 1, 2]:
            lines["loto"].append(f"{T} --data {D} --labels corr --types true --text {txt} --encoder biomedbert --held {t} --seed {s} --name loto_{t}_{txt} --out {R}/loto")
for k, v in lines.items():
    open(f"dcc/tasks_{k}.txt", "w").write("\n".join(v) + "\n")
    print(k, len(v))
