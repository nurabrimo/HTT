#!/bin/bash
# usage: worker.sh <gpu> <taskfile>  — runs each llm_run task line sequentially on one GPU
G=$1; F=$2; W=$HOME/w
export VLLM_USE_DEEP_GEMM=0 VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=$G HF_HOME=$HOME/hf TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
export PATH=$HOME/venv/bin:$PATH
cd $HOME/htt
while read -r line; do
  [ -z "$line" ] && continue
  l=$(echo "$line" | sed "s#$HTT_WORK/htt/runs/llm#$W/runs/llm#")
  m=$(echo "$line" | grep -o -- "--model [^ ]*" | cut -d" " -f2)
  until grep -q "downloaded $m" $HOME/dl2.log; do sleep 20; done
  echo "[$(date +%H:%M:%S)] GPU$G start: $l"
  python $l --data $W/data/tcga.pkl --cache $W/cache 2>&1 | grep -E "^done|Error|Traceback|exists" | tail -3
done < "$F"
echo "[$(date +%H:%M:%S)] GPU$G finished"
