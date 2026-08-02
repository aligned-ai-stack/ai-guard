"""
Export Conversations to CSV
---------------------------
For a given date range, pull every exchange and write one row per judged turn:

    attacker_output , defender_output , judge_verdict

- attacker_output : the LAST attacker turn of the exchange = the muscle/actual
                    message sent to the defender (not the brain's strategy note).
- defender_output : the LAST defender turn of the exchange = the rewritten answer
                    if cross_exam flagged+fixed it, otherwise the original draft.
- judge_verdict   : the judge turn's output_content ("safe"/"unsafe"/"guarded").

Multi-turn traces (convo_length > 1) produce multiple rows, one per judge verdict.
'audit' turns are internal and skipped.

Usage:
    python export_conversations.py
    python export_conversations.py 2026-05-14 2026-05-15
    python export_conversations.py 2026-05-14 2026-05-15 out.csv
"""
import sys
import csv
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "bench" / "cluster_benchmark.db"

# defaults; override via argv
DATE_START = "2026-07-21"
DATE_END = "2026-07-24"

# include leading trace_id / exchange columns? set False for pure 3-col output
INCLUDE_METADATA = True
WRITE_HEADER = True

# False -> one row per exchange (multiple rows per trace).
# True  -> one row per trace = the FINAL exchange only (conversation end-state).
LAST_EXCHANGE_ONLY = False


def _norm(d: str) -> str:
    """Accept 'YYYY-MM-DD' or a full ISO string; timestamps are stored as ISO."""
    return d if "T" in d else f"{d}T00:00:00"


def export(db_path: Path, date_start: str, date_end: str, out_path: Path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    query = """
        SELECT
            r.run_id        AS run_id,
            r.timestamp     AS timestamp,
            t.trace_id      AS trace_id,
            tn.turn_index   AS turn_index,
            tn.role         AS role,
            tn.output_content AS output_content,
            tn.rowid        AS insertion_order
        FROM turns tn
        JOIN traces t ON tn.trace_id = t.trace_id
        JOIN runs   r ON t.run_id    = r.run_id
        WHERE r.timestamp >= ? AND r.timestamp < ?
        ORDER BY r.timestamp, t.trace_id, tn.turn_index, tn.rowid
    """
    cur = conn.execute(query, (_norm(date_start), _norm(date_end)))

    rows = []
    last_trace = None
    exchange_idx = 0
    pending = {"attacker": None, "defender": None}
    meta = {"run_id": None, "trace_id": None}

    def reset_exchange():
        pending["attacker"] = None
        pending["defender"] = None

    for rec in cur:
        trace_id = rec["trace_id"]
        role = rec["role"]
        content = rec["output_content"] or ""

        if trace_id != last_trace:
            last_trace = trace_id
            exchange_idx = 0
            reset_exchange()

        meta["run_id"] = rec["run_id"]
        meta["trace_id"] = trace_id

        if role == "attacker":
            pending["attacker"] = content       # keep last -> muscle message
        elif role == "defender":
            pending["defender"] = content        # keep last -> post-rewrite answer
        elif role == "audit":
            continue                             # internal, not exported
        elif role == "judge":
            exchange_idx += 1
            row = [pending["attacker"] or "", pending["defender"] or "", content]
            if INCLUDE_METADATA:
                row = [meta["run_id"], meta["trace_id"], exchange_idx] + row
            rows.append((trace_id, row))         # tag with trace for optional filtering
            reset_exchange()                     # ready for next exchange in trace

    conn.close()

    if LAST_EXCHANGE_ONLY:
        # keep only the final exchange per trace; dict preserves first-seen order,
        # and later writes overwrite earlier ones -> value is the last exchange.
        final_by_trace = {}
        for tid, row in rows:
            final_by_trace[tid] = row
        rows = [(tid, row) for tid, row in final_by_trace.items()]

    rows = [row for _tid, row in rows]

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)  # handles commas / quotes / newlines inside cells
        if WRITE_HEADER:
            header = ["attacker_output", "defender_output", "judge_verdict"]
            if INCLUDE_METADATA:
                header = ["run_id", "trace_id", "exchange", *header]
            w.writerow(header)
        w.writerows(rows)

    print(f"Wrote {len(rows)} exchange rows to {out_path}")
    if not rows:
        print("  (no data — check the date range and DB path)")


if __name__ == "__main__":
    start = sys.argv[1] if len(sys.argv) > 1 else DATE_START
    end = sys.argv[2] if len(sys.argv) > 2 else DATE_END
    out = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(f"conversations_{start}_to_{end}.csv")

    if not DB_PATH.exists():
        print(f"DB not found at {DB_PATH}")
        sys.exit(1)

    export(DB_PATH, start, end, out)