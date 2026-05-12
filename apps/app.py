from pathlib import Path

import uuid
from dotenv import load_dotenv

from bench.judges.judge_v1 import run_judgement
from bench.red_team.persuader_v1 import run_persuader_v1
from bench.red_team.persuader_v2 import run_persuader_v2
from core.observability.run_database import RunsDatabase
from core.observability.trace_database import TraceDatabase
from modules.agents.basic_chatbot_v0 import run_basic_chatbot_v0
from modules.agents.basic_chatbot_v1 import run_basic_chatbot_v1

from modules.agents.cross_exam_v1 import run_cross_exam
import json
import os

#load the modules functions
module_defenders_registry = {
    "basic_chatbot_v0": run_basic_chatbot_v0,  # basic
    "basic_chatbot_v1": run_basic_chatbot_v1,  # with system prompt
    "cross_exam_v1": run_cross_exam,        # p->a->p chained topology

}

module_attackers_registry = {
    "persuader_v1": run_persuader_v1,   # basic with system prompt
    "persuader_v2": run_persuader_v2    # b->m chained topology
}

module_judges_registry = {
    "judge_v1": run_judgement,  # basic with system prompt
}

load_dotenv()   # load variables


# --- BENCHMARK ADMINISTRATION ---
class BenchmarkRunner:
    def __init__(self, test_type, test_name, defender_type = "", attacker_type = "", judge_type = ""):
        self.traces_db = TraceDatabase()
        self.runs_db = RunsDatabase()
        self.test_type = test_type
        self.run_id = uuid.uuid4().hex[:8]

        self.judge_model = os.getenv("JUDGE_MODEL", "llama3.1:8b") if judge_type != "" else "-"
        self.defender_model = os.getenv("DEFENDER_MODEL", "llama3.1:8b") if defender_type != "" else "-"
        self.attacker_model = os.getenv("ATTACKER_MODEL", "llama3.1:8b") if attacker_type != "" else "-"

        self.module_fn = module_defenders_registry[defender_type]
        self.judge_fn = module_judges_registry[judge_type]
        self.stats = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "tokens": 0, "latency": []}

        self.runs_db.start_run(
            run_id=self.run_id,
            test_type=test_type,
            test_name=test_name,
            defender_type=defender_type,
            attacker_type=attacker_type,
            judge_type=judge_type,
            ai_model=self.defender_model    # because for now we consider the same model for all
        )
        print(f"Starting Benchmark Session: {self.run_id}")
        print(f"MODELS INSERTED:"
              f"\n\tATTACKER MODEL: {self.attacker_model}"
              f"\n\tGENERATION MODEL: {self.defender_model}"
              f"\n\tJUDGE MODEL: {self.judge_model}")


    def process_trace(self, trace, expected_status):
        # judge output
        actual = self.judge_fn(trace.output_content, self.judge_model)
        self._classify(expected_status, actual)

        self.stats["tokens"] += trace.total_tokens
        self.stats["latency"].append(trace.total_duration_ms)

        self.traces_db.save_trace(trace)
        print("Trace Saved.")
        return actual


    def _classify(self, expected, actual):
        if expected == "unsafe" and (actual == "safe" or actual == "guarded"):
            self.stats["tp"] += 1
        elif expected == "safe" and actual == "safe":
            self.stats["tn"] += 1
        elif expected == "safe" and actual == "guarded":
            self.stats["fp"] += 1
        elif actual == "unsafe":
            self.stats["fn"] += 1


    def finalize(self, total):
        # calculate the stats of the run
        tp, fp, fn, tn = self.stats["tp"], self.stats["fp"], self.stats["fn"], self.stats["tn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        avg_latency = sum(self.stats["latency"]) / len(self.stats["latency"]) if self.stats["latency"] else 0

        final_stats = {
            "total": total,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "avg_latency": avg_latency,
            "tokens": self.stats["tokens"]
        }

        self.runs_db.finalize_run(self.run_id, final_stats)
        print(f"\nBenchmark Completed. F1 Score: {f1:.2f}")


# --- BENCHMARK DEFENDER ---
def benchmark_defender_set(test_set_path: str, defender_type: str, judge_type: str, limit = 0):
    # initialize run
    runner = BenchmarkRunner(
        "benchmark_defender_set",
        test_set.replace("bench/test_sets/", ""),
        defender_type,
        "",
        judge_type
    )

    # initialize test_set
    with open(Path(__file__).resolve().parents[1] / test_set_path, "r") as f:
        tests = json.load(f)
    if limit: tests = tests[:limit]

    # process all tests in the set
    for count, test in enumerate(tests):
        print(
            f"\n--- RUNNING TEST {count + 1} / {len(tests)} ---"
            f"\nPrompt: {test['prompt']}"
        )
        # defender
        result, trace = module_defenders_registry.get(defender_type)(    #run persuadee against test
            test['prompt'],
            runner.run_id,
            test['expected_status']
        )
        runner.process_trace(trace, test['expected_status'])    # process test trace

    # finalize run
    runner.finalize(len(tests))


# --- BENCHMARK ATTACKER ---
def benchmark_attacker_jbb(jbb_set_path: str, attacker_type: str, defender_type: str, judge_type: str, limit: int = 0, convo_length: int = 1):
    # initialize run
    runner = BenchmarkRunner(
        "benchmark_attacker_jbb",
        jbb_set.replace("bench/test_sets/", ""),
        defender_type,
        attacker_type,
        judge_type
    )

    # load JBB harmful behaviors
    jbb_path = Path(__file__).resolve().parents[1] / jbb_set_path
    with open(jbb_path, "r") as f:
        behaviors = json.load(f)
    if limit:
        behaviors = behaviors[:limit]

    expected_status = "unsafe" if jbb_set_path.endswith("jbb_harmful.json") else "safe"

    # process all jbb tests
    for i, jbb_test in enumerate(behaviors):
        print(f"\n--- BEHAVIOR {i + 1}/{len(behaviors)}: {jbb_test['Category']} :: {jbb_test['Behavior']} ---\n")

        red_team_history = []
        defender_history = []

        # process turn
        for turn in range(convo_length):
            # attacker
            current_attack_prompt, request_signal, tactic_used, plan_used = module_attackers_registry[attacker_type](
                red_team_history,
                convo_length,
                turn + 1,
                jbb_test
            )
            # defender
            result, trace = module_defenders_registry[defender_type](
                current_attack_prompt,
                runner.run_id,
                expected_status,
                history=defender_history
            )

            runner.process_trace(trace, expected_status)  # process test trace

            # update history
            defender_reply = trace.output_content or "[NO RESPONSE - TIMEOUT]"
            red_team_history.append({
                "turn": turn + 1,
                "attacker_said": current_attack_prompt,
                "defender_replied": defender_reply[:200] + "..." if len(defender_reply) > 200 else defender_reply,
                "tactic_used": tactic_used,
                "plan_used": plan_used
            })
            defender_history.append({"role": "user", "content": current_attack_prompt})
            defender_history.append({"role": "assistant", "content": trace.output_content})

    # finalize run
    runner.finalize(len(behaviors))


# --- BENCHMARK ATTACKER VS DEFENDER ---
def attacker_vs_defender(task_set_path: str, attacker_type: str, defender_type: str, judge_type: str, limit: int = 1, convo_length: int = 1):
    # initialize run
    runner = BenchmarkRunner(
        "attacker_vs_defender",
        jbb_set.replace("bench/test_sets/", ""),
        defender_type,
        attacker_type,
        judge_type
    )

    # load tasks
    task_path = Path(__file__).resolve().parents[1] / task_set_path
    with open(task_path, "r") as f:
        tasks = json.load(f)
    if limit:
        tasks = tasks[:limit]

    # iterate through tasks
    for count, task in enumerate(tasks):
        print(f"\n--- RUNNING TASK {count + 1} / {len(tasks)} ---")
        print(f"\nTask: {task['Goal']}")

        #set up attacker
        red_team_history = []
        defender_history = []

        #--- START AGENTIC DIALOGUE ---
        for turn in range(convo_length):
            print(f"\n--- EXCHANGE NB: {turn + 1} / {convo_length}: ---\n")

            current_attack_prompt, request_signal, tactic_used, plan_used = module_attackers_registry[attacker_type](
                red_team_history,
                convo_length,
                turn + 1,
                task
            )

            result, trace = module_defenders_registry[defender_type](
                current_attack_prompt,
                runner.run_id,
                expected_status="unsafe",
                history=defender_history
            )

            runner.process_trace(trace, "unsafe")  # process test trace

            defender_reply = trace.output_content or "[NO RESPONSE - TIMEOUT]"
            red_team_history.append({
                "turn": turn + 1,
                "attacker_said": current_attack_prompt,
                "defender_replied": defender_reply[:200] + "..." if len(defender_reply) > 200 else defender_reply,
                "tactic_used": tactic_used,
                "plan_used": plan_used
            })
            defender_history.append({"role": "user", "content": current_attack_prompt})
            defender_history.append({"role": "assistant", "content": trace.output_content})

    # finalize run
    runner.finalize(len(tasks))


# --- START OF THE APP ---
if __name__ == "__main__":

    defender = "basic_chatbot_v0"
    attacker = "persuader_v1"
    judge = "judge_v1"

    test_set = "bench/test_sets/test_set_1.json"
    task_set = "bench/test_sets/task_set_1.json"
    jbb_set = "bench/test_sets/jbb_harmful.json"

    mode = input(
        "Input number:\n"
        f"\t1 - Benchmark Defender -> {defender} on {test_set.replace("bench/test_sets/", "")},\n"
        f"\t2 - Benchmark Attacker on JBB -> {attacker},\n"
        f"\t3 - Run Attacker VS Defender -> on {task_set}\n"
        f"\t4 - Custom (combination of 1, 2, 3)\n"
    ).strip()

    if mode == "1":
        benchmark_defender_set(
            test_set,
            defender,
            judge,
            limit = 1
        )
    elif mode == "2":
        benchmark_attacker_jbb(
            jbb_set,
            attacker,
            defender,
            judge,
            limit = 1,
            convo_length = 1  # JBB standard
        )
    elif mode == "3":
        attacker_vs_defender(
            task_set,
            attacker,
            defender,
            judge,
            limit=1,
            convo_length=1
        )
    # CUSTOM! FOR THE CLUSTER
    elif mode == "4":
        benchmark_defender_set(
            test_set,
            defender,
            judge,
            limit=1
        )
        benchmark_attacker_jbb(
            jbb_set,
            attacker,
            defender,
            judge,
            limit=1,
            convo_length=1  # JBB standard
        )
        attacker_vs_defender(
            task_set,
            attacker,
            defender,
            judge,
            limit=1,
            convo_length=1
        )