from pathlib import Path

import uuid
from dotenv import load_dotenv

from core.observability.trace_database import TraceDatabase
from modules.cross_exam_v1 import run_cross_exam
import json

load_dotenv()

def run_benchmark(dataset_path: str):
    db = TraceDatabase()

    run_id = f"{uuid.uuid4().hex[:8]}"
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
        if trace.status == "SUCCESS":
            print(result.output_content)
        else:
            print(f"Trace failed with status: {trace.status}")

        db.save_trace(trace)


    print(f"Benchmark Session Completed: {run_id}")

if __name__ == "__main__":
    run_benchmark("bench/test_sets/set_test_1.json")