cd $HOME
mkdir -p w/runs/llm hf
pip3 install -q --user --break-system-packages uv > setup.log 2>&1
$HOME/.local/bin/uv venv $HOME/venv --python 3.12 >> setup.log 2>&1
VIRTUAL_ENV=$HOME/venv $HOME/.local/bin/uv pip install -q "vllm==0.33.0" pandas scikit-learn pyarrow ninja "huggingface_hub[hf_xet]" >> setup.log 2>&1
$HOME/venv/bin/python -c "import vllm; print('vllm', vllm.__version__)" >> setup.log 2>&1
cat > $HOME/dl2.sh <<'X'
export HF_HOME=$HOME/hf
for m in Qwen/Qwen3.6-27B-FP8 Qwen/Qwen3-14B Qwen/Qwen3-8B Intelligent-Internet/II-Medical-8B openai/gpt-oss-20b; do $HOME/venv/bin/hf download $m --quiet && echo "downloaded $m"; done
echo DL_DONE
X
nohup bash $HOME/dl2.sh > $HOME/dl2.log 2>&1 < /dev/null &
nohup $HOME/worker.sh 0 $HOME/t2_gpu0.txt > $HOME/w0.log 2>&1 < /dev/null &
nohup $HOME/worker.sh 1 $HOME/t2_gpu1.txt > $HOME/w1.log 2>&1 < /dev/null &
sleep 3; tail -n 2 setup.log; pgrep -c -f "worker[.]sh"
