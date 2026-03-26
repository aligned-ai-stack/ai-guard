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
RESPONSE_MODEL=gemma3:270m
AUDITOR_MODEL=llama3.1:8b
ATTACKER_MODEL=hf.co/UnfilteredAI/DAN-L3-R1-8B
JUDGE_MODEL=llama3.1:8b
```