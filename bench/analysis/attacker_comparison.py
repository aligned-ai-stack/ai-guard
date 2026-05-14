"""
Attacker Effectiveness: persuader_v1 vs persuader_v2
------------------------------------------------------
Grouped bar chart showing ASR for each attacker against each defender.
Directly answers: "Is the brain+muscle split (v2) actually better at
breaking defenders than the single-shot (v1)?"

Usage: python plot_attacker_comparison.py
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "benchmark.db"

DATE_START = "2026-05-14T00:00:00"
DATE_END = "2026-05-15T00:00:00"

query = """
    SELECT defender_module, attacker_module, asr
    FROM runs
    WHERE attacker_module IS NOT NULL
      AND attacker_module != ''
      AND asr IS NOT NULL
"""

if DATE_START and DATE_END:
    query += f" AND timestamp >= '{DATE_START}' AND timestamp < '{DATE_END}'"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("No data found.")
    exit()

# average ASR per (defender, attacker) pair
grouped = df.groupby(["defender_module", "attacker_module"])["asr"].mean().reset_index()

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(9, 5))

sns.barplot(
    data=grouped, x="defender_module", y="asr",
    hue="attacker_module", ax=ax,
    palette=["#E76F51", "#264653"],
    edgecolor="white", linewidth=0.8
)

ax.set_xlabel("Defender Module")
ax.set_ylabel("Attack Success Rate (ASR)")
ax.set_ylim(0, 1.05)
ax.set_title("Attacker Effectiveness: v1 (single-shot) vs v2 (brain+muscle)", fontweight="bold")
ax.legend(title="Attacker", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.axhline(y=0.5, color="#999", linestyle="--", linewidth=0.7, alpha=0.5)
ax.grid(axis="y", alpha=0.3)

# annotate bars
for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=9, fontweight="bold", padding=3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "attacker_comparison.png", dpi=300)
print("Saved attacker_comparison.png")