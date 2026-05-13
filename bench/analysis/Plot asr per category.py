"""
ASR per Category
----------------
Grouped bar chart. One bar group per behavior category, one bar per defender.
Shows which categories are hardest to defend against.

Usage: python plot_asr_per_category.py
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "benchmark.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT t.final_verdict, t.behavior_category, r.defender_module
    FROM traces t
    JOIN runs r ON t.run_id = r.run_id
    WHERE t.expected_status = 'unsafe'
      AND t.status = 'SUCCESS'
      AND t.behavior_category != ''
""", conn)
conn.close()

if df.empty:
    print("No data found.")
    exit()

# compute ASR per (category, defender)
grouped = df.groupby(["behavior_category", "defender_module"]).apply(
    lambda g: (g["final_verdict"] == "unsafe").mean()
).reset_index(name="asr")

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, ax = plt.subplots(figsize=(10, 5))

sns.barplot(
    data=grouped, x="behavior_category", y="asr",
    hue="defender_module", ax=ax
)

ax.set_xlabel("Category")
ax.set_ylabel("Attack Success Rate (ASR)")
ax.set_ylim(0, 1.05)
ax.set_title("ASR per behavior category")
ax.legend(title="Defender", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.xticks(rotation=35, ha="right")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "asr_per_category.pdf", dpi=300)
plt.savefig(Path(__file__).parent / "asr_per_category.png", dpi=300)
print("Saved asr_per_category.pdf/.png")