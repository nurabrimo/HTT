#!/bin/bash
#SBATCH --job-name=htt
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --requeue
#SBATCH --exclude=dcc-core-gpu-ferc-s-h36-6,dcc-gehmlab-gpu-ferc-s-z25-18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --array=0-N%K dcc/array.sbatch tasks.txt   (one python argument line per task)
export HF_HOME=$HTT_WORK/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
cd $HOME/htt
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) task: $LINE"
$HTT_WORK/htt/venv/bin/python $LINE
