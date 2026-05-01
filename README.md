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
DEFENDER_MODEL=dolphin-llama3:8b
ATTACKER_MODEL=dolphin-llama3:8b
JUDGE_MODEL=dolphin-llama3:8b
```
4. In case the database is not writing to disk, delete and rerun the script.