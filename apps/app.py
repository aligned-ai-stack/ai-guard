from pathlib import Path

import uuid
from dotenv import load_dotenv

from bench.red_team.charmer_v1 import run_charmer
from core.observability.run_database import RunsDatabase
from core.observability.trace_database import TraceDatabase

from modules.agents.cross_exam_v1 import run_cross_exam
import json
import os

#load the modules functions
module_agents_registry = {
    "cross_exam_v1": run_cross_exam,
}

module_red_team_registry = {
    "charmer_v1": run_charmer,
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
def run_agent(red_team: str, module_type: str):
    #initialize the databases
    traces_db = TraceDatabase()
    runs_db = RunsDatabase()

    convo_length = 5
    run_id = f"{uuid.uuid4().hex[:8]}"
    gen_model = os.getenv("RESPONSE_MODEL", "llama3.1:8b")
    audit_model = os.getenv("AUDITOR_MODEL", "llama3.1:8b")
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

    #start agentic conversation
    for nb in range(convo_length):
        print(f"\n--- Back n Forth nb. {nb + 1}: ---")

        print("hm")


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


def run_benchmark(dataset_path: str, module_type: str):
    #initialize the databases
    traces_db = TraceDatabase()
    runs_db = RunsDatabase()

    test_set = dataset_path.split("/")[-1].split(".")[0]
    run_id = f"{uuid.uuid4().hex[:8]}"
    gen_model = os.getenv("RESPONSE_MODEL", "llama3.1:8b")
    audit_model = os.getenv("AUDITOR_MODEL", "llama3.1:8b")
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

        stats["tokens"] += trace.total_tokens
        stats["latency"].append(trace.total_duration_ms)

        expected = trace.expected_status
        actual = trace.predicted_status

        #check the prediction
        if expected == "unsafe" and actual == "unsafe":
            stats["tn"] += 1
        elif expected == "safe" and actual == "safe":
            stats["tp"] += 1
        elif expected == "safe" and actual == "unsafe":
            stats["fp"] += 1
        elif expected == "unsafe" and actual == "safe":
            stats["fn"] += 1

        traces_db.save_trace(trace)


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
    mode = input("Input number: 1 - Test_Set, 2 - Agent, 3 - User (not added)\n")
    module = "cross_exam_v1"
    red_team_agent = "charmer_v1"

    if mode == "1":
        run_benchmark(
            "bench/test_sets/set_test_1.json",
            module
                      )

    elif mode == "2":
        run_agent(
            red_team_agent,
            module
                      )

    elif mode == "3":
        print("To be added")