# Data

## Included
- `synthetic/` — the phrase and growth benchmarks used in the paper, one CSV per split
  (`export_growth_n70_<split>.csv`, `export_phrase_<split>.csv`). Growth columns include the true doubling
  time (`dt_true`) and the reported sizes; `days_from_start` is the visit day.

## Public, download yourself (`get_public_data.sh`)
- **TCGA-Reports** (Kefeli & Tatonetti, Patterns 2024): `TCGA_Reports.csv` and the patient-to-project table from
  the TCGA-Reports release (https://github.com/tatonettilab/tcga-path-reports). Then run
  `python httx/tcga_prep.py <root> tcga.pkl`.
- **TCGA-CDR** (Liu et al., Cell 2018): `TCGA-CDR-SupplementalTableS1.xlsx` from the GDC PanCanAtlas page.
  Then run `python httx/cdr_prep.py tcga.pkl TCGA-CDR.xlsx`.
- **MSK-CHORD** (Jee et al., Nature 2024; CC BY-NC-ND 4.0): `msk_chord_2024.tar.gz` from the cBioPortal datahub.

## Access-controlled
- **AACR GENIE BPC** public cohorts (NSCLC, CRC, BrCa, PANC, Prostate, Bladder): register at
  https://genie.cbioportal.org, accept the terms, create a data access token, then run
  `python httx/genie_fetch.py <token_file> <out_dir>` and `python httx/genie_prep.py <out_dir> genie.pkl`.
  Do not redistribute the downloaded data.
