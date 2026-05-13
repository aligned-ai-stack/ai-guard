import sqlite3
import json
from pathlib import Path
from core.contracts.models import Run, Trace, Turn


class BenchmarkDB:
    def __init__(self, db_rel_path: str = "bench/benchmark.db"):
        self.db_path = Path(__file__).resolve().parents[2] / db_rel_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init()

    def _init(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    benchmark_mode TEXT,
                    dataset_path TEXT,
                    defender_module TEXT,
                    attacker_module TEXT,
                    judge_module TEXT,
                    defender_model TEXT,
                    attacker_model TEXT,
                    judge_model TEXT,
                    backend TEXT,
                    total_traces INTEGER DEFAULT 0,
                    asr REAL,
                    refusal_rate REAL,
                    f1_score REAL,
                    accuracy REAL,
                    precision REAL,
                    recall REAL,
                    tp_count INTEGER DEFAULT 0,
                    tn_count INTEGER DEFAULT 0,
                    fp_count INTEGER DEFAULT 0,
                    fn_count INTEGER DEFAULT 0,
                    avg_tokens_per_trace REAL DEFAULT 0,
                    avg_duration_ms_per_trace REAL DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_duration_ms REAL DEFAULT 0,
                    config_json TEXT
                );
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    run_id TEXT REFERENCES runs(run_id),
                    behavior_goal TEXT,
                    behavior_category TEXT,
                    expected_status TEXT,
                    final_verdict TEXT,
                    turns_used INTEGER DEFAULT 0,
                    total_tokens INTEGER DEFAULT 0,
                    total_duration_ms REAL DEFAULT 0,
                    status TEXT,
                    error_report TEXT,
                    metadata_json TEXT
                );
                CREATE TABLE IF NOT EXISTS turns (
                    turn_id TEXT PRIMARY KEY,
                    trace_id TEXT REFERENCES traces(trace_id),
                    turn_index INTEGER,
                    role TEXT,
                    model_used TEXT,
                    output_content TEXT,
                    input_tokens INTEGER DEFAULT 0,
                    output_tokens INTEGER DEFAULT 0,
                    duration_ms REAL DEFAULT 0,
                    status TEXT,
                    error_report TEXT,
                    execution_data_json TEXT
                );
            """)

    def save_run(self, run: Run):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run.run_id, run.timestamp.isoformat(), run.benchmark_mode,
                run.dataset_path, run.defender_module, run.attacker_module,
                run.judge_module, run.defender_model, run.attacker_model,
                run.judge_model, run.backend, run.total_traces,
                run.asr, run.refusal_rate, run.f1_score,
                run.accuracy, run.precision, run.recall,
                run.tp_count, run.tn_count, run.fp_count, run.fn_count,
                run.avg_tokens_per_trace, run.avg_duration_ms_per_trace,
                run.total_tokens, run.total_duration_ms,
                json.dumps(run.config)
            ))

    def save_trace(self, trace: Trace):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO traces VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                trace.trace_id, trace.run_id, trace.behavior_goal,
                trace.behavior_category, trace.expected_status,
                trace.final_verdict,
                trace.turns_used, trace.total_tokens, trace.total_duration_ms,
                trace.status, trace.error_report,
                json.dumps(trace.metadata)
            ))

    def save_turns(self, turn: Turn):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO turns VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                turn.turn_id, turn.trace_id, turn.turn_index,
                turn.role, turn.model_used, turn.output_content,
                turn.input_tokens, turn.output_tokens, turn.duration_ms,
                turn.status, turn.error_report,
                json.dumps(turn.execution_data)
            ))