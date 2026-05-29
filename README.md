# ai-guard
Testing multi-slm architectures for the best outcomes in safety based on token consumption.

# How to run

1. Pull the repo
```
git clone https://github.com/aligned-ai-stack/ai-guard.git
```

2. Install requirements
```
pip install -r requirements.txt
```

3. Set up env based on local ollama models; e.g.:
```
export LLM_BACKEND=ollama

export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=EMPTY

export DEFENDER_MODEL=dolphin-llama3:8b
export ATTACKER_MODEL=dolphin-llama3:8b
export JUDGE_MODEL=dolphin-llama3:8b

ollama run dolphin-llama3:8b

in case ollama does not use the GPU:
sudo lsof -i :11434
sudo kill -9 (INSERT PID HERE FROM PREVIOUS COMMAND)
then rerun
```
If vllm is not installed, it can be left like this.

4. In case the database is not writing to disk, delete and rerun the script.

## Run on a cluster
1. Run this script template
```
cd data/ai-guard-storage/ai-guard
source .venv/bin/activate

# install once (or in case of a new version)
pip install vllm


#terminal 1
vllm serve "solidrust/dolphin-2.9.4-llama3.1-8b-AWQ" --max-model-len 4096 --gpu-memory-utilization 0.8 --port 8000
# wait till this line is done

#terminal 2
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
export DEFENDER_MODEL=solidrust/dolphin-2.9.4-llama3.1-8b-AWQ
export ATTACKER_MODEL=solidrust/dolphin-2.9.4-llama3.1-8b-AWQ
export JUDGE_MODEL=solidrust/dolphin-2.9.4-llama3.1-8b-AWQ

python apps/app.py


#reset
vllm serve solidrust/dolphin-2.9.4-llama3.1-8b-AWQ --port 8000 --max-model-len 4096 --gpu-memory-utilization 0.8
```

---

# Running benchmarks correctly
## JBB Benchmark
1. Use vllm
2. Use the models mentioned below for the defender and judge (more parameters, more accurate):
```
defender1: lmsys/vicuna-13b-v1.5
defender2: meta-llama/Llama-2-7b-chat-hf    # needs license

judge1: meta-llama/Meta-Llama-3-70B-Instruct    # needs license
judge2: meta-llama/Meta-Llama-3-8B-Instruct     # needs license
```
3. Set conversation length to 1 in the main function in app.py