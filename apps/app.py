from pathlib import Path

import uuid
from dotenv import load_dotenv

from bench.judges.judge_v1 import run_judgement
from bench.red_team.persuader_v1 import run_persuader_v1
from bench.red_team.persuader_v2 import run_persuader_v2
from core.observability.run_database import RunsDatabase
from core.observability.trace_database import TraceDatabase
from modules.agents.basic_chatbot_v1 import run_basic_chatbot

from modules.agents.cross_exam_v1 import run_cross_exam
import json
import os

#load the modules functions
module_agents_registry = {
    "cross_exam_v1": run_cross_exam,
    "basic_chatbot_v1": run_basic_chatbot
}

module_red_team_registry = {
    "persuader_v1": run_persuader_v1,
    "persuader_v2": run_persuader_v2
}

module_judges_registry = {
    "judge_v1": run_judgement,
}

load_dotenv()

"""
The structure:
1. Test sets. run_benchmark()
    * The run is initialized
    * For each test, we take the trace and run the model with the test prompt, 
    a judge reviews the final response.
    * We then calculate precision, F1, ...

2. Agent. run_agent()
    * An agentic loop starts(limit = 5)
    * Instead of test sets, we have a red_team agent with a goal in mind, based on the restrictions
    it has, it will try to in the end get what it wants.
    * A judge reviews if the red_team got what it wanted.
    * The rest of the process is similar
    
3. User
    * Just for testing, on the back burner for now
"""
def run_agent(red_team: str, module_type: str, judge_type: str):
    #initialize the databases
    traces_db = TraceDatabase()
    runs_db = RunsDatabase()

    convo_length = 3

    run_id = f"{uuid.uuid4().hex[:8]}"
    gen_model = os.getenv("RESPONSE_MODEL", "llama3.1:8b")
    audit_model = os.getenv("AUDITOR_MODEL", "llama3.1:8b")
    attacker_model = os.getenv("ATTACKER_MODEL", "llama3.1:8b")
    judge_model = os.getenv("JUDGE_MODEL", "llama3.1:8b")
    stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "tokens": 0, "latency": []}

    runs_db.start_run(
        run_id = run_id,
        test_type = "red_team",
        test_name = red_team,
        framework = module_type,
        gen_model = gen_model,
        audit_model = audit_model
    )
    print(f"Starting Benchmark Session: {run_id}")

    print(f"MODELS INSERTED:"
          f"\nATTACKER MODEL: {attacker_model}"
          f"\nGENERATION MODEL: {gen_model}"
          f"\nAUDITOR MODEL: {audit_model}"
          f"\nJUDGE MODEL: {judge_model}")

    #set up attacker
    red_team_history = []
    defender_history = []

    #start agentic conversation
    for nb in range(convo_length):
        print(f"\n--- EXCHANGE NB: {nb + 1}: ---\n")

        current_attack_prompt, request_signal = module_red_team_registry[red_team](
            red_team_history,
            attacker_model
        )

        result, trace = module_agents_registry[module_type](
            current_attack_prompt,
            run_id,
            expected_status="unsafe",
            history=defender_history
        )

        expected = trace.expected_status
        actual = module_judges_registry[judge_type](trace.output_content, judge_model)

        if expected == "unsafe" and (actual == "safe" or actual == "guarded"):
            stats["tp"] += 1
        elif expected == "safe" and actual == "safe":
            stats["tn"] += 1
        elif expected == "safe" and actual == "guarded":
            stats["fp"] += 1
        elif actual == "unsafe":
            stats["fn"] += 1

        stats["tokens"] += trace.total_tokens
        stats["latency"].append(trace.total_duration_ms)
        traces_db.save_trace(trace)
        print("Trace Saved.")

        red_team_history.append({"role": "assistant", "content": current_attack_prompt})
        red_team_history.append({"role": "user", "content": trace.output_content})

        defender_history.append({"role": "user", "content": current_attack_prompt})
        defender_history.append({"role": "assistant", "content": trace.output_content})


    #calculate the stats of the run
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    avg_latency = sum(stats["latency"]) / len(stats["latency"]) if stats["latency"] else 0

    final_stats = {
        "total": convo_length,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_latency": avg_latency,
        "tokens": stats["tokens"]
    }

    runs_db.finalize_run(run_id, final_stats)
    print(f"\nBenchmark Completed. F1 Score: {f1:.2f}")


def run_benchmark(dataset_path: str, module_type: str, judge_type: str):
    #initialize the databases
    traces_db = TraceDatabase()
    runs_db = RunsDatabase()

    test_set = dataset_path.split("/")[-1].split(".")[0]
    run_id = f"{uuid.uuid4().hex[:8]}"
    gen_model = os.getenv("RESPONSE_MODEL", "llama3.1:8b")
    audit_model = os.getenv("AUDITOR_MODEL", "llama3.1:8b")
    judge_model = os.getenv("JUDGE_MODEL", "llama3.1:8b")
    stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "tokens": 0, "latency": []}

    runs_db.start_run(
        run_id=run_id,
        test_type="test_set",
        test_name=test_set,
        framework=module_type,
        gen_model=gen_model,
        audit_model=audit_model
    )
    print(f"Starting Benchmark Session: {run_id}")

    print(f"MODELS INSERTED:"
          f"\nGENERATION MODEL: {gen_model}"
          f"\nAUDITOR MODEL: {audit_model}"
          f"\nJUDGE MODEL: {judge_model}")

    with open(Path(__file__).resolve().parents[1] / dataset_path, "r") as f:
        tests = json.load(f)

    for test in tests:
        print(f"\n--- Testing ID: {test['id']} ---")

        #function
        result, trace =module_agents_registry.get(module_type)(
            test['prompt'],
            run_id,
            test['expected_status']
        )

        expected = trace.expected_status
        actual = module_judges_registry[judge_type](trace.output_content, judge_model)

        if expected == "unsafe" and (actual == "safe" or actual == "guarded"):
            stats["tp"] += 1
        elif expected == "safe" and actual == "safe":
            stats["tn"] += 1
        elif expected == "safe" and actual == "guarded":
            stats["fp"] += 1
        elif actual == "unsafe":
            stats["fn"] += 1

        stats["tokens"] += trace.total_tokens
        stats["latency"].append(trace.total_duration_ms)
        traces_db.save_trace(trace)
        print("Trace Saved.")


    #calculate the stats of the run
    tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    avg_latency = sum(stats["latency"]) / len(stats["latency"]) if stats["latency"] else 0

    final_stats = {
        "total": len(tests),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "avg_latency": avg_latency,
        "tokens": stats["tokens"]
    }

    runs_db.finalize_run(run_id, final_stats)
    print(f"\nBenchmark Completed. F1 Score: {f1:.2f}")


if __name__ == "__main__":
    mode = input("Input number:\n\t1 - Test_Set,\n\t2 - Agent,\n\t3 - User (not added)\n")

    module = "basic_chatbot_v1"
    red_team_agent = "persuader_v2"
    judge = "judge_v1"
    test_set = "bench/test_sets/set_test_1.json"

    if mode == "1":
        run_benchmark(
            test_set,
            module,
            judge
                      )

    elif mode == "2":
        run_agent(
            red_team_agent,
            module,
            judge
                      )

    elif mode == "3":
        print("To be added")