#!/bin/bash
#SBATCH --job-name=htt_ceil
#SBATCH --partition=common,scavenger
#SBATCH --requeue
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%A_%a.out
cd $HOME/htt
export OMP_NUM_THREADS=2
if [ "$SLURM_ARRAY_TASK_ID" = "20" ]; then
  $HTT_WORK/htt/venv/bin/python httx/ceil_run.py sets $HTT_WORK/htt/runs/ceilings
else
  $HTT_WORK/htt/venv/bin/python httx/ceil_run.py pop $HTT_WORK/htt/runs/ceilings $SLURM_ARRAY_TASK_ID 20
fi
