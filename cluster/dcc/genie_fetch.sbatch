#!/bin/bash
#SBATCH --job-name=htt-genie
#SBATCH --partition=common
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%j.out
umask 077
cd $HOME/htt
$HTT_WORK/htt/venv_llm/bin/python httx/genie_fetch.py $HOME/.secrets/genie_token.txt $HTT_WORK/htt/data/genie_bpc
