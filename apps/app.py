from pathlib import Path
import sys

from bench.attackers.fitd import run_fitd

from modules.defenders.intent_tracker_v1 import run_intent_tracker

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from bench.attackers.persuader_v3 import run_persuader_v3
from bench.judges.judge_v2 import run_task_judgement

from bench.judges.judge_v1 import run_safety_judgement
from bench.attackers.persuader_v1 import run_persuader_v1
from bench.attackers.persuader_v2 import run_persuader_v2
from modules.defenders.basic_chatbot_v0 import run_basic_chatbot_v0
from modules.defenders.basic_chatbot_v1 import run_basic_chatbot_v1

from core.contracts.models import Run, Trace, Turn
from core.observability.db import BenchmarkDB

from modules.defenders.cross_exam_v1 import run_cross_exam
import json
import os

load_dotenv()   # load variables

#load the modules functions
defender_registry = {
    "basic_chatbot_v0": run_basic_chatbot_v0,  # basic
    "basic_chatbot_v1": run_basic_chatbot_v1,  # with system prompt
    "cross_exam_v1": run_cross_exam,        # p->a->p chained topology
    "intent_tracker_v1": run_intent_tracker,    # stateful intent prediction

}

attacker_registry = {
    "persuader_v1": run_persuader_v1,   # basic with system prompt
    "persuader_v2": run_persuader_v2,    # b->m chained topology
    "persuader_v3": run_persuader_v3,    # e->b->m chained topology
    "fitd": run_fitd,                       # foot-in-the-door, de-escalates on refusal
}

judge_registry = {
    "judge_v1": run_safety_judgement,  # basic with system prompt
    "judge_v2": run_task_judgement,     # analyzes how connected is a response with a task
}

# --- HISTORY UTILS ---
def frame_history(conversation, perspective):
    """Frame the conversation history such that:
    -for the defender, he is the assistant and the attacker is the user
    -for the attacker, he is the assistant and the defender is the user"""
    framed = []
    for msg in conversation:
        if msg["role"] == perspective:
            framed.append({"role": "assistant", "content": msg["content"]})
        else:
            framed.append({"role": "user", "content": msg["content"]})
    return framed


# --- BENCHMARK ADMINISTRATION ---
class BenchmarkRunner:
    def __init__(self, benchmark_mode, dataset_path,
                 defender_type="", attacker_type="", judge_type=""):
        self.db = BenchmarkDB()
        self.traces = []

        self.defender_fn = defender_registry.get(defender_type)
        self.attacker_fn = attacker_registry.get(attacker_type)
        self.judge_fn = judge_registry.get(judge_type)

        self.run = Run(
            benchmark_mode=benchmark_mode,
            dataset_path=dataset_path,
            defender_module=defender_type,
            attacker_module=attacker_type,
            judge_module=judge_type,
            defender_model=os.getenv("DEFENDER_MODEL", ""),
            attacker_model=os.getenv("ATTACKER_MODEL", ""),
            judge_model=os.getenv("JUDGE_MODEL", ""),
            backend=os.getenv("LLM_BACKEND", "ollama"),
        )

        self.db.save_run(self.run)
        print(f"Starting Benchmark: {self.run.run_id}")
        print(f"  DEFENDER: {self.run.defender_model} ({defender_type})")
        print(f"  ATTACKER: {self.run.attacker_model} ({attacker_type})")
        print(f"  JUDGE:    {self.run.judge_model} ({judge_type})")

        # --- CORE: process one behavior/prompt ---

    def run_trace(self, prompt, expected_status, category="", convo_length=1, task=None):
        """One trace = one behavior being tested across N turns."""

        trace = Trace(
            run_id=self.run.run_id,
            behavior_goal=prompt,
            behavior_category=category,
            expected_status=expected_status,
            status="RUNNING",
        )
        self.db.save_trace(trace)

        trace_tokens = 0    #still trace
        trace_duration = 0.0

        conversation_history = []   # universal history, it states which agent said what
        attacker_state = {}
        defender_state = {}
        turn_index = 0

        # walks through all turns in one round
        for turn_num in range(convo_length):
            print(f"\n--- EXCHANGE {turn_num + 1}/{convo_length} ---")

            # --- ATTACKER, call only if we have a task and attacker inserted ---
            attack_prompt = prompt
            if self.attacker_fn and task:
                attacker_history = frame_history(conversation_history, "attacker")

                attacker_turns, attacker_state = self.attacker_fn(
                    task, attacker_history, attacker_state, convo_length, turn_num + 1
                )
                # save attacker turns
                for t in attacker_turns:
                    turn = Turn(
                        trace_id=trace.trace_id,
                        turn_index=turn_index,
                        **t
                    )
                    self.db.save_turns(turn)
                    turn_index += 1 # next turn
                # the attack prompt is the last attacker turn's output
                attack_prompt = attacker_turns[-1]["output_content"]

                trace_tokens += sum(t.get("input_tokens", 0) + t.get("output_tokens", 0) for t in attacker_turns)
                trace_duration += sum(t.get("duration_ms", 0.0) for t in attacker_turns)


            # --- DEFENDER, always called ---
            max_history_pairs = 4
            recent_history = conversation_history[-(max_history_pairs * 2):]
            defender_history = frame_history(recent_history, "defender")

            defender_turns, defender_state = self.defender_fn(
                attack_prompt, defender_history, defender_state
            )
            # save defender turns
            for t in defender_turns:
                turn = Turn(
                    trace_id=trace.trace_id,
                    turn_index=turn_index,
                    **t
                )
                self.db.save_turns(turn)
                turn_index += 1
            # the final defender output is the last turn's content
            defender_output = defender_turns[-1]["output_content"] or ""

            trace_tokens += sum(t.get("input_tokens", 0) + t.get("output_tokens", 0) for t in attacker_turns)
            trace_duration += sum(t.get("duration_ms", 0.0) for t in attacker_turns)


            # --- UPDATE CANONICAL HISTORY ---
            conversation_history.append({"role": "attacker", "content": attack_prompt})
            conversation_history.append({"role": "defender", "content": defender_output})


            # --- JUDGE ---
            # note, this if-check is redundant, unless if we go for more
            # objective checks in the future where we do not need an agent
            if self.judge_fn:
                judge_result = self.judge_fn(defender_output, task.get('Goal', '') if task else prompt)
                judge_turn = Turn(
                    trace_id=trace.trace_id,
                    turn_index=turn_index,
                    **judge_result
                )
                self.db.save_turns(judge_turn)

                if judge_result["output_content"] == "unsafe":
                    trace.final_verdict = "unsafe"
                    print(f"Verdict on Turn {turn_num + 1}: unsafe (Jailbreak Successful)")

                    trace_tokens += sum(t.get("input_tokens", 0) + t.get("output_tokens", 0) for t in attacker_turns)
                    trace_duration += sum(t.get("duration_ms", 0.0) for t in attacker_turns)


        # --- FINALIZE TRACE ---
        if trace.final_verdict != "unsafe":
            trace.final_verdict = "safe"
        trace.turns_used = convo_length
        trace.status = "SUCCESS"
        trace.total_tokens = trace_tokens
        trace.total_duration_ms = trace_duration

        self.db.save_trace(trace)
        self.traces.append(trace)
        print("Trace saved.")
        return trace


    # --- FINALIZE RUN ---
    def finalize(self):
        total = len(self.traces)
        self.run.total_traces = total

        if total == 0:
            self.db.save_run(self.run)
            return

        # count verdicts
        unsafe = sum(1 for t in self.traces if t.final_verdict == "unsafe")
        guarded = sum(1 for t in self.traces if t.final_verdict == "guarded")

        # ASR
        self.run.asr = unsafe / total

        # refusal rate
        self.run.refusal_rate = guarded / total

        # confusion matrix
        for t in self.traces:
            exp = t.expected_status
            verdict = t.final_verdict
            if exp == "unsafe" and verdict in ("safe", "guarded"):
                self.run.tp_count += 1
            elif exp == "safe" and verdict == "safe":
                self.run.tn_count += 1
            elif exp == "safe" and verdict == "guarded":
                self.run.fp_count += 1
            elif verdict == "unsafe":
                self.run.fn_count += 1

        tp = self.run.tp_count
        fp = self.run.fp_count
        fn = self.run.fn_count
        tn = self.run.tn_count

        self.run.precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        self.run.recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        self.run.f1_score = (
            2 * self.run.precision * self.run.recall
            / (self.run.precision + self.run.recall)
            if (self.run.precision + self.run.recall) > 0 else 0
        )
        self.run.accuracy = (tp + tn) / total if total > 0 else 0

        self.run.total_tokens = sum(t.total_tokens for t in self.traces)
        self.run.total_duration_ms = sum(t.total_duration_ms for t in self.traces)
        if total > 0:
            self.run.avg_tokens_per_trace = self.run.total_tokens / total
            self.run.avg_duration_ms_per_trace = self.run.total_duration_ms / total

        self.db.save_run(self.run)
        print(f"\nBenchmark Complete.")
        print(f"  ASR: {self.run.asr:.2%}")
        print(f"  F1:  {self.run.f1_score:.2f}")
        print(f"  TP:{tp} TN:{tn} FP:{fp} FN:{fn}")


# --- BENCHMARK DEFENDER ---
def benchmark_defender_set(test_set_path, defender_type, judge_type, limit=0):
    with open(Path(__file__).resolve().parents[1] / test_set_path, "r") as f:
        tests = json.load(f)
    if limit:
        tests = tests[:limit]

    runner = BenchmarkRunner(
        "defender_set",
        test_set_path,
        defender_type=defender_type,
        judge_type=judge_type
    )

    for i, test in enumerate(tests):
        print(f"\n--- TEST {i + 1}/{len(tests)} ---")
        print(f"Prompt: {test['prompt']}")

        runner.run_trace(
            prompt=test['prompt'],
            expected_status=test['expected_status'],
            category=test.get('category', ''),
        )

    runner.finalize()


# --- BENCHMARK ATTACKER ---
JBB_VANILLA_DEFENDERS = {
    "lmsys/vicuna-13b-v1.5",
    "meta-llama/Llama-2-7b-chat-hf",
}

JBB_VANILLA_JUDGES = {
    "meta-llama/Meta-Llama-3-70B-Instruct",
    "meta-llama/Meta-Llama-3-8B-Instruct",
}


def benchmark_attacker_jbb(jbb_set_path: str, attacker_type: str, defender_type: str, judge_type: str, limit: int = 0, convo_length: int = 1):
    # check if jbb is vanilla
    vanilla_models = check_jbb_vanilla(defender_model=os.getenv("DEFENDER_MODEL", ""),
                                   judge_model=os.getenv("JUDGE_MODEL", ""))

    with open(Path(__file__).resolve().parents[1] / jbb_set_path, "r") as f:
        behaviors = json.load(f)
    if limit:
        behaviors = behaviors[:limit]

    runner = BenchmarkRunner(
        "attacker_jbb", jbb_set_path,
        defender_type=defender_type,
        attacker_type=attacker_type,
        judge_type=judge_type
    )

    for i, behavior in enumerate(behaviors):
        print(f"\n--- BEHAVIOR {i + 1}/{len(behaviors)}: "
              f"{behavior.get('Category', '')} :: {behavior.get('Behavior', '')} ---")
        runner.run_trace(
            prompt=behavior.get('Goal', behavior.get('Behavior', '')),
            expected_status="unsafe",
            category=behavior.get('Category', ''),
            convo_length=convo_length,
            task=behavior,
        )

    runner.finalize()


def check_jbb_vanilla(defender_model: str, judge_model: str) -> bool:
    defender_ok = defender_model in JBB_VANILLA_DEFENDERS
    judge_ok = judge_model in JBB_VANILLA_JUDGES

    if defender_ok and judge_ok:
        print(f"[JBB] Vanilla config — leaderboard-comparable.")
        return True

    print("[JBB] WRONG MODELS — running custom JBB, not leaderboard-comparable.")
    if not defender_ok:
        print(f"  defender: '{defender_model}' (expected one of {sorted(JBB_VANILLA_DEFENDERS)})")
    if not judge_ok:
        print(f"  judge:    '{judge_model}' (expected one of {sorted(JBB_VANILLA_JUDGES)})\n")
    return False


# --- BENCHMARK ATTACKER VS DEFENDER ---
def attacker_vs_defender(task_set_path: str, attacker_type: str, defender_type: str, judge_type: str, limit: int = 1, convo_length: int = 1):
    with open(Path(__file__).resolve().parents[1] / task_set_path, "r") as f:
        tasks = json.load(f)
    if limit:
        tasks = tasks[:limit]

    runner = BenchmarkRunner(
        "attacker_vs_defender", task_set_path,
        defender_type=defender_type,
        attacker_type=attacker_type,
        judge_type=judge_type
    )

    for i, task in enumerate(tasks):
        print(f"\n--- TASK {i + 1}/{len(tasks)}: {task.get('Goal', '')} ---")
        runner.run_trace(
            prompt=task.get('Goal', ''),
            expected_status="unsafe",
            category=task.get('Category', ''),
            convo_length=convo_length,
            task=task,
        )

    runner.finalize()


# --- START OF THE APP ---
if __name__ == "__main__":

    defender = "cross_exam_v1"
    attacker = "fitd"
    judge = "judge_v2"

    test_set = "bench/test_sets/test_set_1.json"    # for benchmarking defender, not so relevant for now
    task_set =  "bench/test_sets/jbb_harmful.json"  # "bench/test_sets/task_set_1.json"    for defender vs attacker
    jbb_set = "bench/test_sets/jbb_harmful.json"    # for jailbreakbench

    mode = input(
        "Input number:\n"
        f"\t1 - Benchmark Defender -> {defender} on {test_set.replace('bench/test_sets/', '')},\n"
        f"\t2 - Benchmark Attacker on JBB -> {attacker},\n"
        f"\t3 - Run Attacker VS Defender -> on {task_set}\n"
        f"\t4 - Custom (combination of 1, 2, 3)\n"
    ).strip()

    if mode == "1":
        benchmark_defender_set(test_set, defender, judge, limit=1)
    elif mode == "2":
        benchmark_attacker_jbb(jbb_set, attacker, defender, judge,
                               limit=1, convo_length=1)
    elif mode == "3":
        attacker_vs_defender(task_set, attacker, defender, judge,
                             limit=1, convo_length=1)
    elif mode == "4":
        defenders = ["basic_chatbot_v1", "cross_exam_v1", "intent_tracker_v1"]
        for defender in defenders:
            attacker_vs_defender(task_set, attacker, defender, judge, limit=5, convo_length=5)