"""
Run Summary Table
-----------------
Prints a formatted table of all runs with key metrics.
Good for quick overview before diving into specific charts.

Usage: python print_run_summary.py
"""
import sqlite3
import pandas as pd
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "benchmark.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT run_id, benchmark_mode, defender_module, attacker_module,
           total_traces, asr, refusal_rate, f1_score,
           tp_count, tn_count, fp_count, fn_count,
           avg_tokens_per_trace, avg_duration_ms_per_trace
    FROM runs
    ORDER BY timestamp DESC
""", conn)
conn.close()

if df.empty:
    print("No runs found.")
    exit()

# format for display
df["asr"] = df["asr"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
df["refusal_rate"] = df["refusal_rate"].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "-")
df["f1_score"] = df["f1_score"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "-")
df["avg_tokens"] = df["avg_tokens_per_trace"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
df["avg_ms"] = df["avg_duration_ms_per_trace"].apply(lambda x: f"{x:.0f}" if pd.notna(x) else "-")
df["confusion"] = df.apply(lambda r: f"TP:{r.tp_count} TN:{r.tn_count} FP:{r.fp_count} FN:{r.fn_count}", axis=1)

display_cols = ["run_id", "benchmark_mode", "defender_module", "attacker_module",
                "total_traces", "asr", "f1_score", "confusion", "avg_tokens", "avg_ms"]

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
pd.set_option("display.max_colwidth", 25)
print(df[display_cols].to_string(index=False))