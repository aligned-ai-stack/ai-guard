"""
ASR per Behavior Category
--------------------------
Grouped bar chart. One bar group per behavior category, one bar per defender.
Shows which categories are hardest to defend against.

Usage: python plot_asr_per_category.py
TO BE BUILT
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
    SELECT t.final_verdict, t.behavior_category, r.defender_module
    FROM traces t
    JOIN runs r ON t.run_id = r.run_id
    WHERE t.expected_status = 'unsafe'
      AND t.status = 'SUCCESS'
      AND t.behavior_category != ''
"""
if DATE_START and DATE_END:
    query += f" AND r.timestamp >= '{DATE_START}' AND r.timestamp < '{DATE_END}'"

df = pd.read_sql_query(query, conn)
conn.close()

if df.empty:
    print("No data found.")
    exit()

# ASR = fraction where judge said "unsafe" (attack succeeded)
grouped = df.groupby(["behavior_category", "defender_module"]).apply(
    lambda g: (g["final_verdict"] == "unsafe").mean()
).reset_index(name="asr")

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(12, 5.5))

sns.barplot(
    data=grouped, x="behavior_category", y="asr",
    hue="defender_module", ax=ax,
    palette=["#E8453C", "#F4A261", "#2A9D8F"],
    edgecolor="white", linewidth=0.8
)

ax.set_xlabel("Behavior Category")
ax.set_ylabel("Attack Success Rate (ASR)")
ax.set_ylim(0, 1.05)
ax.set_title("ASR per Behavior Category × Defender", fontweight="bold")
ax.legend(title="Defender", bbox_to_anchor=(1.02, 1), loc="upper left")
ax.axhline(y=0.5, color="#999", linestyle="--", linewidth=0.7, alpha=0.5)
plt.xticks(rotation=35, ha="right")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "asr_per_category.png", dpi=300)
print("Saved asr_per_category.png")