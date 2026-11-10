#!/bin/bash
#SBATCH --job-name=htt_cpu
#SBATCH --partition=common,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-N%K dcc/array.sbatch tasks.txt   (one python argument line per task)
export OMP_NUM_THREADS=4 HF_HOME=$HTT_WORK/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
cd $HOME/htt
echo "host $(hostname) task: $LINE"
$HTT_WORK/htt/venv/bin/python $LINE
