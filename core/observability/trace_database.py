import sqlite3
import json
from pathlib import Path

from core.contracts.trace import Trace

class TraceDatabase:
    #constructor
    def __init__(self, db_rel_path: str = "bench/traces.db"):
        self.db_path = Path(__file__).resolve().parents[2] / db_rel_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_db()


    #initialize if empty
    def _initialize_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    run_id TEXT,
                    timestamp TEXT,
                    input_query TEXT,
                    expected_status TEXT,
                    input_tokens INTEGER,
                    framework_version TEXT,
                    gen_model TEXT,
                    audit_model TEXT,
                    execution_data TEXT,
                    predicted_status TEXT,
                    output_content TEXT,
                    status TEXT,
                    error_report TEXT,
                    total_duration_ms REAL,
                    total_tokens INTEGER,
                    tags TEXT
                )
            """)
            conn.commit()


    #add an element
    def save_trace(self, trace: Trace):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            placeholders = ", ".join(["?"] * 17)
            sql_query = f"""
                INSERT INTO traces (
                    trace_id, run_id, timestamp, input_query, expected_status,
                    input_tokens, framework_version, gen_model, audit_model,
                    execution_data, predicted_status, output_content, status, error_report,
                    total_duration_ms, total_tokens, tags
                ) VALUES ({placeholders})
            """
            data_to_save = (
                trace.trace_id,
                trace.run_id,
                trace.timestamp.isoformat(),
                trace.input_query,
                trace.expected_status,
                trace.input_tokens,
                trace.framework_version,
                trace.gen_model,
                trace.audit_model,
                json.dumps(trace.execution_data),
                trace.predicted_status,
                trace.output_content,
                trace.status,
                trace.error_report,
                trace.total_duration_ms,
                trace.total_tokens,
                json.dumps(trace.tags)
            )

            cursor.execute(sql_query, data_to_save)
            conn.commit()