# HTT: do clinical sequence models and language models use time?

Code, synthetic data and results for the manuscript *Do Clinical Sequence Models and Language Models Use
Time? Bayes-Optimal Benchmarks, Multi-Institution Timelines and Label Provenance in Oncology*.

The Hierarchical Temporal Transformer (HTT) reads a patient's sequence of cancer reports. It encodes each report
with BiomedBERT + LoRA, then a cancer-type-conditioned temporal transformer reads the sequence. The repository
evaluates it, and five open-weight language models, in four settings:

| Setting | Data | Main question |
|---|---|---|
| Synthetic phrase and growth benchmarks | generated here (`data/synthetic`) | Does a model use visit dates, measured against exact Bayes-optimal AUROC? |
| TCGA pathology reports | TCGA-Reports, TCGA-CDR (public) | Are text-mined grade labels right, and what transfers once grade phrases are masked? |
| Real curated timelines, 4 institutions | AACR GENIE BPC (registration required) | Does timing help on real records, validated at an institution not used for training? |
| NLP vs curated labels | MSK-CHORD × GENIE BPC | Do published NLP progression labels agree with human curation? |

## Layout

```
httx/        analysis package (models, training, benchmarks, runners, aggregation, figures)
cluster/     Slurm job scripts (Duke Compute Cluster) and NVIDIA Brev worker scripts
data/        synthetic benchmarks (CSV) and instructions/scripts to obtain the public datasets
results/     aggregate results (summary_rev1.json, summary_rev2.json), audits, Bayes ceilings,
             and run-level outputs for the synthetic and TCGA experiments
figures/     Figures 11-15 (PNG and PDF)
```

### Key modules

| File | What it does |
|---|---|
| `httx/models.py`, `httx/engine.py` | HTT, time codes (original and clinical band), training loop |
| `httx/growth.py`, `httx/ceil_run.py`, `httx/ceil_redate.py` | Growth-benchmark generator and exact Bayes readers |
| `httx/synth_run.py` | Runner for the phrase and growth benchmarks |
| `httx/tcga_prep.py`, `httx/cdr_prep.py` | TCGA label audit, corrected grade rule, redaction, registry linkage |
| `httx/tcga_run.py`, `httx/tfidf_run.py`, `httx/surv_run.py` | TCGA grade runs, lexical baseline, survival models |
| `httx/genie_fetch.py`, `httx/genie_prep.py`, `httx/genie_run.py` | GENIE BPC download (your token), timelines, leave-one-institution-out runs |
| `httx/chord_vs_genie.py`, `httx/chord_extra.py` | MSK-CHORD date alignment and NLP-vs-curation comparison |
| `httx/llm_run.py`, `httx/genie_llm.py` | Open-weight language-model readers (vLLM, local GPUs) |
| `httx/aggregate.py`, `httx/aggregate2.py`, `httx/figs2.py` | Paired bootstrap contrasts, tables and figures |

`httx/oncolocus_ref.py` reproduces functions from the original OncoLocus code
(https://github.com/rajneeshanand/OncoLocus, commit 47719d1) so the published labels and phrase benchmark can be
regenerated exactly.

## Reproducing

1. Environment: Python 3.11, `pip install -r requirements.txt` (for language models, a separate environment
   with `vllm==0.33.0`).
2. Data: follow `data/README.md`. The synthetic benchmarks are in `data/synthetic` and can be regenerated with
   `httx/growth.py` and `httx/synth_run.py`.
3. Paths: the scripts use `/path/to/work` (Python defaults) and `$HTT_WORK` (Slurm scripts) for the work
   directory; set these to your own locations.
4. Runs: `cluster/dcc/gen_tasks*.py` writes one command line per run; `cluster/dcc/array.sbatch` executes one
   line per Slurm array index. Aggregate with `httx/aggregate.py` and `httx/aggregate2.py`.

Training runs are not bitwise reproducible across GPU and CPU types: re-running 27 seeds on other hardware changed
set-level AUROCs by a median of 0.008, which is why results are reported over ten seeds.

## Data use

GENIE BPC and MSK-CHORD data are **not** included. Their terms do not permit redistribution (GENIE terms of
access; MSK-CHORD CC BY-NC-ND 4.0). `results/summary_rev2.json` contains only aggregate statistics for those
analyses. GENIE BPC patient-level outputs are available on request to users with GENIE access.
