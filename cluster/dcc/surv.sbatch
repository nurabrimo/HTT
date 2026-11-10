#!/bin/bash
#SBATCH --job-name=htt-surv
#SBATCH --partition=common,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
export OMP_NUM_THREADS=8 PYTHONUNBUFFERED=1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
cd $HOME/htt
echo "host $(hostname) task: $LINE"
$HTT_WORK/htt/venv_llm/bin/python $LINE
