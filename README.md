# AI Guard

A research testbed for comparing language-model defenses during adversarial conversations. It runs attacker, defender, and judge configurations and records their outputs, token use, and timing.

The working question is whether additional checking or tracking of intent improves resistance to attacks enough to justify its computational cost. This repository contains experiments and analysis tools; results depend on the models, prompts, datasets, and judging procedure used.

Part of [Aligned AI Stack](https://github.com/aligned-ai-stack).

## What is implemented

- Defender variants: a basic chatbot, a system-prompt baseline, cross-examination, and an intent tracker.
- Attacker variants: persuasion pipelines and a foot-in-the-door strategy.
- Judges for safety and task relevance.
- Records for runs, conversation traces, and individual turns, stored through SQLite.
- Analysis scripts for attack success, latency, defender behavior, and guard degradation.

## Repository guide

| Path | Contents |
| --- | --- |
| `apps/app.py` | Experiment runner and current configuration choices. |
| `modules/defenders/` | Defender implementations. |
| `bench/attackers/`, `bench/judges/` | Attack and judging procedures. |
| `bench/test_sets/` | Experiment inputs. |
| `bench/analysis/` | Analysis scripts and saved figures. |
| `core/` | Model client, record schemas, and database handling. |
| `docs/` | Working research notes and diagrams. |

## Run an experiment

From the repository root, create a Python environment and install `requirements.txt`. Review `.env.example` and configure a running Ollama or vLLM server with the models you intend to compare.

For example, with a model already available in Ollama:

```bash
export LLM_BACKEND=ollama
export DEFENDER_MODEL=llama3.1:8b
export ATTACKER_MODEL=llama3.1:8b
export JUDGE_MODEL=llama3.1:8b
python -m apps.app
```

This is an example configuration, not a benchmark reference setting. Select a mode when prompted. The main block in `apps/app.py` currently controls the module choices, dataset paths, task limits, and conversation lengths. Modes 1–3 use one item by default; mode 4 compares several defenders on five tasks with five exchanges.

For vLLM, set `LLM_BACKEND=vllm`, `VLLM_BASE_URL`, and the three model names to match your server configuration. Run the model server separately.

## Interpreting results

Record the model versions, judge, prompts, turn limits, and dataset with every comparison. Separate safety judgments, task completion, and token cost. LLM judgments need their own validation, and changing a model or prompt can change the meaning of a score.

JailbreakBench inputs are included. A run against those inputs alone does not establish comparability with its leaderboard protocol. The working notes in `docs/` describe exploratory runs; this repository does not establish a general safety guarantee.
