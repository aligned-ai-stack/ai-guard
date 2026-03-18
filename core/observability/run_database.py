import sqlite3
import json
from pathlib import Path
from datetime import datetime

class RunsDatabase:
    def __init__(self, db_rel_path: str = "bench/runs.db"):
        self.db_path = Path(__file__).resolve().parents[2] / db_rel_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    test_set TEXT,
                    framework_version TEXT,
                    gen_model TEXT,
                    audit_model TEXT,
                    system_prompts TEXT,
                    total_traces INTEGER,
                    tp_count INTEGER DEFAULT 0,
                    tn_count INTEGER DEFAULT 0,
                    fp_count INTEGER DEFAULT 0,
                    fn_count INTEGER DEFAULT 0,
                    precision REAL,
                    recall REAL,
                    f1_score REAL,
                    avg_latency_ms REAL,
                    total_tokens INTEGER
                )
            """)
            conn.commit()

    def start_run(self, run_id, test_set, framework, gen_model, audit_model, system_prompts):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO runs (run_id, timestamp, test_set, framework_version, gen_model, audit_model, system_prompts)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (run_id, datetime.now().isoformat(), test_set, framework, gen_model, audit_model, json.dumps(system_prompts)))
            conn.commit()

    def finalize_run(self, run_id, stats: dict):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE runs SET 
                    total_traces = ?,
                    tp_count = ?, tn_count = ?, fp_count = ?, fn_count = ?,
                    precision = ?, recall = ?, f1_score = ?,
                    avg_latency_ms = ?, total_tokens = ?
                WHERE run_id = ?
            """, (
                stats['total'],
                stats['tp'], stats['tn'], stats['fp'], stats['fn'],
                stats['precision'], stats['recall'], stats['f1'],
                stats['avg_latency'], stats['tokens'],
                run_id
            ))
            conn.commit()