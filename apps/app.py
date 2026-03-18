from pathlib import Path

import uuid
from dotenv import load_dotenv

from core.observability.run_database import RunsDatabase
from core.observability.trace_database import TraceDatabase
from modules.cross_exam_v1 import run_cross_exam
import json
import os

load_dotenv()

def run_benchmark(dataset_path: str):
    #initialize the databases
    traces_db = TraceDatabase()
    runs_db = RunsDatabase()

    test_set = dataset_path.split("/")[-1].split(".")[0]

    run_id = f"{uuid.uuid4().hex[:8]}"
    res_model = os.getenv("RESPONSE_MODEL", "llama3.1:8b")
    audit_model = os.getenv("AUDITOR_MODEL", "llama3.1:8b")
    framework = "cross_exam_v1"
    stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "tokens": 0, "latency": []}
    runs_db.start_run(run_id, test_set, framework, res_model, audit_model, {})

    print(f"Starting Benchmark Session: {run_id}")

    with open(Path(__file__).resolve().parents[1] / dataset_path, "r") as f:
        tests = json.load(f)

    for test in tests:
        print(f"\n--- Testing ID: {test['id']} ---")
        result, trace = run_cross_exam(
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
    run_benchmark("bench/test_sets/set_test_2.json")