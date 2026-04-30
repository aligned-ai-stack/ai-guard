import sqlite3
import json
from contextlib import closing
from pathlib import Path
from datetime import datetime

class RunsDatabase:
    def __init__(self, db_rel_path: str = "bench/runs.db"):
        self.db_path = Path(__file__).resolve().parents[2] / db_rel_path
        print(f"[RunsDB] Writing to: {self.db_path.resolve()}")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()

    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    test_type TEXT,
                    test_name TEXT,
                    framework TEXT,
                    gen_model TEXT,
                    audit_model TEXT,
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

    def start_run(self, run_id, test_type, test_name, framework, gen_model, audit_model):
        with closing(sqlite3.connect(self.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO runs (run_id, timestamp, test_type, test_name, framework, gen_model, audit_model)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (run_id, datetime.now().isoformat(), test_type, test_name, framework, gen_model, audit_model))
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