"""
Latency Distribution per Defender
-----------------------------------
Violin plot of per-trace latency for each defender module.
Shows not just mean but the full distribution — useful for spotting
timeout-prone defenders (long tail).

Usage: python plot_latency_violin.py
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent / "benchmark.db"

conn = sqlite3.connect(DB_PATH)

DATE_START = "2026-05-14T00:00:00"
DATE_END = "2026-05-15T00:00:00"

query = """
    SELECT t.total_duration_ms, r.defender_module
    FROM traces t
    JOIN runs r ON t.run_id = r.run_id
    WHERE t.status = 'SUCCESS'
      AND t.total_duration_ms > 0
"""

if DATE_START and DATE_END:
    query += f" AND r.timestamp >= '{DATE_START}' AND r.timestamp < '{DATE_END}'"

df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("No data found.")
    exit()

# convert to seconds for readability
df["duration_s"] = df["total_duration_ms"] / 1000.0

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(9, 5))

sns.violinplot(
    data=df, x="defender_module", y="duration_s",
    palette=["#E8453C", "#F4A261", "#2A9D8F"],
    inner="box", cut=0, ax=ax, linewidth=1.2
)

# overlay strip for individual points
sns.stripplot(
    data=df, x="defender_module", y="duration_s",
    color="black", alpha=0.07, size=2, jitter=True, ax=ax
)

ax.set_xlabel("Defender Module")
ax.set_ylabel("Trace Duration (seconds)")
ax.set_title("Latency Distribution per Defender", fontweight="bold")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "latency_violin.png", dpi=300)
print("Saved latency_violin.png")