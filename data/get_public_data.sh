#!/bin/bash
# Download the public datasets used in the paper into ./raw (TCGA-CDR and MSK-CHORD; see README for TCGA-Reports).
set -e
mkdir -p raw && cd raw
curl -L -o TCGA-CDR.xlsx "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
curl -L -o msk_chord_2024.tar.gz "https://datahub.assets.cbioportal.org/msk_chord_2024.tar.gz" && tar xzf msk_chord_2024.tar.gz
echo "TCGA-Reports: clone https://github.com/tatonettilab/tcga-path-reports and follow its instructions."
