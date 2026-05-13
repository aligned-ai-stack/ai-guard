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
LLM_BACKEND=ollama

VLLM_BASE_URL=http://localhost:8000/v1
VLLM_API_KEY=EMPTY

DEFENDER_MODEL=llama3.1:8b
ATTACKER_MODEL=dolphin-llama3:8b
JUDGE_MODEL=dolphin-llama3:8b
```
If vllm is not installed, it can be left like this.

4. In case the database is not writing to disk, delete and rerun the script.

## Run on a cluster
1. Run this script template
```
# install once, never redownload
pip install vllm


#terminal 1
vllm serve dphn/Dolphin3.0-Llama3.1-8B --port 8000
# wait till this line is done

#terminal 2
export LLM_BACKEND=vllm
export VLLM_BASE_URL=http://localhost:8000/v1
export DEFENDER_MODEL=dphn/Dolphin3.0-Llama3.1-8B
export ATTACKER_MODEL=dphn/Dolphin3.0-Llama3.1-8B
export JUDGE_MODEL=dphn/Dolphin3.0-Llama3.1-8B

python apps/app.py


#reset
vllm serve dphn/Dolphin3.0-Llama3.1-8B --port 8000 --max-model-len 8192
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