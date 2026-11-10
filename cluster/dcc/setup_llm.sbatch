#!/bin/bash
#SBATCH --job-name=htt-setup
#SBATCH --partition=common,scavenger
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out
# One-off: vLLM venv, open-weight models, TCGA-CDR clinical table.
set -x
export HF_HOME=$HTT_WORK/hf_cache PIP_CACHE_DIR=$HTT_WORK/pip_cache
V=$HTT_WORK/htt/venv_llm
if [ ! -x $V/bin/vllm ]; then
  $HTT_WORK/conda/recast/bin/python -m venv $V
  $V/bin/pip install -q --upgrade pip
  $V/bin/pip install -q vllm pandas scikit-learn scikit-survival lifelines openpyxl
fi
$V/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"
for m in Qwen/Qwen3-8B Intelligent-Internet/II-Medical-8B Qwen/Qwen3-14B Qwen/Qwen3.6-27B-FP8 openai/gpt-oss-20b; do
  $V/bin/hf download $m --quiet || $V/bin/huggingface-cli download $m --quiet
  echo "downloaded $m"
done
mkdir -p $HTT_WORK/htt/data
cd $HTT_WORK/htt/data
[ -s TCGA-CDR.xlsx ] || curl -sL -o TCGA-CDR.xlsx "https://api.gdc.cancer.gov/data/1b5f413e-a8d1-4d10-92eb-7c4ae739ed81"
ls -la $HTT_WORK/htt/data
du -sh $HTT_WORK/hf_cache/hub/models--*Qwen* $HTT_WORK/hf_cache/hub/models--*gpt-oss* $HTT_WORK/hf_cache/hub/models--*II-Medical* 2>/dev/null
echo SETUP_DONE
