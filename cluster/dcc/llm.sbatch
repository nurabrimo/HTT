#!/bin/bash
#SBATCH --job-name=htt-llm
#SBATCH --partition=gpu-common,scavenger-gpu
#SBATCH --requeue
#SBATCH --exclude=dcc-core-gpu-ferc-s-h36-6,dcc-gehmlab-gpu-ferc-s-z25-18,dcc-majoroslab-gpu-ferc-a-r32-34
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%A_%a.out
# Usage: sbatch --gres=gpu:6000_ada:1 --array=0-N%K dcc/llm.sbatch tasks.txt   (local vLLM; data stay on DCC)
export VLLM_USE_DEEP_GEMM=0 PATH=$HTT_WORK/htt/venv_llm/bin:$PATH HF_HOME=$HTT_WORK/hf_cache TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$1")
cd $HOME/htt
echo "host $(hostname) gpu $(nvidia-smi --query-gpu=name --format=csv,noheader) task: $LINE"
$HTT_WORK/htt/venv_llm/bin/python $LINE
