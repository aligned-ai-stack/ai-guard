"""
Guard Degradation Over Conversation Depth
-------------------------------------------
Line chart: ASR per turn index, one line per defender.
Tests the hypothesis: "the guard of a module lowers based on the depth of the chat."

Uses the turns table to determine per-turn outcomes. Since the judge
only gives a final_verdict per trace, we approximate per-turn ASR by
grouping traces by turns_used (conversation length) and comparing ASR.

Usage: python plot_guard_degradation.py
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "benchmark.db"

DATE_START = "2026-05-14T00:00:00"
DATE_END = "2026-05-15T00:00:00"

conn = sqlite3.connect(DB_PATH)

query = """
    SELECT t.trace_id, t.final_verdict, t.expected_status,
           t.turns_used, r.defender_module, r.attacker_module
    FROM traces t
    JOIN runs r ON t.run_id = r.run_id
    WHERE t.expected_status = 'unsafe'
      AND t.status = 'SUCCESS'
      AND t.turns_used > 0
"""

if DATE_START and DATE_END:
    query += f" AND r.timestamp >= '{DATE_START}' AND r.timestamp < '{DATE_END}'"

df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("No data found.")
    exit()

df["attack_success"] = (df["final_verdict"] == "unsafe").astype(int)

# ASR grouped by (defender, turns_used)
asr = df.groupby(["defender_module", "turns_used"]).agg(
    asr=("attack_success", "mean"),
    n=("attack_success", "count")
).reset_index()

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(8, 5))

sns.lineplot(
    data=asr, x="turns_used", y="asr",
    hue="defender_module", marker="o",
    linewidth=2.5, markersize=9, ax=ax,
    palette=["#E8453C", "#F4A261", "#2A9D8F"]
)

ax.set_xlabel("Conversation Length (turns)")
ax.set_ylabel("Attack Success Rate (ASR)")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Guard Degradation Over Conversation Depth", fontweight="bold")
ax.legend(title="Defender", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.axhline(y=0.5, color="#999", linestyle="--", linewidth=0.7, alpha=0.5)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "guard_degradation.png", dpi=300)
print("Saved guard_degradation.png")