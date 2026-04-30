"""
Diagnose why runs in runs.db have null metrics.

Run from project root:
    python diagnose_runs.py

Checks:
  1. Which runs never finalized (total_traces IS NULL)
  2. For each unfinalized run, how many traces actually saved + their status
  3. Diagnosis: did the loop never start, die mid-way, or finish but skip finalize?
"""
import sqlite3
from pathlib import Path

RUNS_DB = Path("bench/runs.db")
TRACES_DB = Path("bench/traces.db")


def main():
    if not RUNS_DB.exists():
        print(f"runs.db not found at {RUNS_DB.resolve()}")
        print("Run this from your project root.")
        return
    if not TRACES_DB.exists():
        print(f"traces.db not found at {TRACES_DB.resolve()}")
        return

    # 1. find unfinalized runs
    with sqlite3.connect(RUNS_DB) as conn:
        conn.row_factory = sqlite3.Row
        unfinalized = conn.execute("""
            SELECT run_id, timestamp, test_type, test_name, framework, gen_model
            FROM runs
            WHERE total_traces IS NULL
            ORDER BY timestamp DESC
        """).fetchall()

        total_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]

    print(f"Total runs in db: {total_runs}")
    print(f"Unfinalized runs: {len(unfinalized)}")
    print("=" * 80)

    if not unfinalized:
        print("All runs have been finalized. No diagnostic needed.")
        return

    # 2. for each unfinalized run, look at its traces
    with sqlite3.connect(TRACES_DB) as conn:
        conn.row_factory = sqlite3.Row

        for run in unfinalized:
            run_id = run["run_id"]
            print(f"\nRUN: {run_id}")
            print(f"  timestamp : {run['timestamp']}")
            print(f"  test_type : {run['test_type']}  ({run['test_name']})")
            print(f"  framework : {run['framework']}")

            traces = conn.execute("""
                SELECT trace_id, status, predicted_status, expected_status,
                       error_report, total_duration_ms, total_tokens
                FROM traces
                WHERE run_id = ?
                ORDER BY timestamp ASC
            """, (run_id,)).fetchall()

            print(f"  traces saved: {len(traces)}")

            if not traces:
                # cause 2 most likely: loop died on turn 1 before any trace saved
                print("  >>> DIAGNOSIS: loop died before any trace was saved.")
                print("      Most likely the persuader/attacker crashed on turn 1.")
                print("      Check: persuader_v2 returns 3-tuple on except, app unpacks 4.")
            else:
                statuses = [t["status"] for t in traces]
                print(f"  trace statuses: {statuses}")

                # show errors if any
                errored = [t for t in traces if t["status"] in ("FAILED", "TIMEOUT")]
                for t in errored:
                    err = (t["error_report"] or "")[:120]
                    print(f"    {t['trace_id'][:8]} [{t['status']}]: {err}")

                if all(s == "SUCCESS" for s in statuses):
                    print("  >>> DIAGNOSIS: all traces succeeded but finalize never ran.")
                    print("      Something crashed AFTER the last trace saved but")
                    print("      BEFORE finalize_run — likely in the metric calc block,")
                    print("      OR the unpacking on the next iteration's persuader call.")
                else:
                    print("  >>> DIAGNOSIS: loop ran partially. Check the errors above.")
                    print("      Either way, no try/finally means finalize is skipped.")

    print("\n" + "=" * 80)
    print("FIX SUMMARY:")
    print("  1. persuader_v2.py except branch: return 4 values, not 3.")
    print("  2. Wrap run_agent / run_benchmark loops in try/finally so")
    print("     finalize_run runs even on partial completion.")


if __name__ == "__main__":
    main()