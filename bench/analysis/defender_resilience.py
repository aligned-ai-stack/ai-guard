"""
Defender Resilience Curve
--------------------------
Line chart: ASR at each conversation length (turns_used), one line per defender.
Answers: "How many turns of sustained pressure does it take to break each defender?"

Unlike guard_degradation (which looks at per-turn judge verdicts),
this groups by the TOTAL conversation length of completed traces.

Usage: python defender_resilience.py
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "cluster_benchmark.db"

DATE_START = None
DATE_END = None

conn = sqlite3.connect(DB_PATH)

query = """
    SELECT t.final_verdict, t.turns_used, r.defender_module, r.attacker_module, r.timestamp
    FROM traces t
    JOIN runs r ON t.run_id = r.run_id
    WHERE t.expected_status = 'unsafe'
      AND r.timestamp >= '2026-07-21T00:00:00'
      AND r.timestamp < '2026-07-24T00:00:00'
      AND t.status = 'SUCCESS'
      AND t.turns_used > 0
      AND r.attacker_module IS NOT NULL
      AND r.attacker_module != ''
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
grouped = df.groupby(["defender_module", "turns_used"]).agg(
    asr=("attack_success", "mean"),
    n=("attack_success", "count")
).reset_index()

# only keep groups with enough samples
grouped = grouped[grouped["n"] >= 1]

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(10, 6))

sns.lineplot(
    data=grouped, x="turns_used", y="asr",
    hue="defender_module", marker="o",
    linewidth=2.5, markersize=9, ax=ax,
    palette="Set2"
)

ax.set_xlabel("Conversation Length (total turns)")
ax.set_ylabel("Attack Success Rate (ASR)")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Defender Resilience Over Conversation Length", fontweight="bold")
ax.legend(title="Defender", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.axhline(y=0.5, color="#999", linestyle="--", linewidth=0.7, alpha=0.5)
ax.grid(alpha=0.3)

# annotate sample sizes
for _, row in grouped.iterrows():
    ax.annotate(
        f"n={int(row['n'])}",
        (row["turns_used"], row["asr"]),
        textcoords="offset points", xytext=(0, 10),
        fontsize=7, color="#666", ha="center"
    )

plt.tight_layout()
plt.savefig(Path(__file__).parent / "defender_resilience.png", dpi=300)
print("Saved defender_resilience.png")
