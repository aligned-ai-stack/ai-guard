"""
Tactic Effectiveness Heatmap
-----------------------------
Rows = manipulation tactics, Columns = behavior categories.
Cell color = success rate (how often the attacker broke through).
Requires tactic info stored in turns.execution_data_json or traces.metadata_json.

Usage: python plot_tactic_heatmap.py
"""
import sqlite3
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "benchmark.db"

conn = sqlite3.connect(DB_PATH)

# try to get tactic from the attacker turns' execution_data
df = pd.read_sql_query("""
    SELECT t.trace_id, t.behavior_category, t.final_verdict, t.metadata_json,
           tu.execution_data_json
    FROM traces t
    LEFT JOIN turns tu ON t.trace_id = tu.trace_id AND tu.role = 'attacker'
    WHERE t.expected_status = 'unsafe'
      AND t.status = 'SUCCESS'
      AND t.behavior_category != ''
""", conn)
conn.close()

if df.empty:
    print("No data found.")
    exit()

# extract tactic from execution_data or metadata
def get_tactic(row):
    for field in [row.get("execution_data_json"), row.get("metadata_json")]:
        if field:
            try:
                data = json.loads(field) if isinstance(field, str) else field
                for key in ["tactic_used", "tactic_chosen", "tactic"]:
                    if key in data:
                        return data[key]
            except (json.JSONDecodeError, TypeError):
                pass
    return "unknown"

df["tactic"] = df.apply(get_tactic, axis=1)

# drop duplicates (multiple attacker turns per trace)
df = df.drop_duplicates(subset=["trace_id"])
df = df[df["tactic"] != "unknown"]

if df.empty:
    print("No tactic data found in execution_data or metadata.")
    exit()

# compute success rate per (tactic, category)
df["broke_through"] = (df["final_verdict"] == "unsafe").astype(int)
pivot = df.pivot_table(
    values="broke_through",
    index="tactic",
    columns="behavior_category",
    aggfunc="mean"
).fillna(0)

sns.set_theme(style="white", font_scale=0.95)
fig, ax = plt.subplots(figsize=(10, max(4, len(pivot) * 0.6 + 1.5)))

sns.heatmap(
    pivot, annot=True, fmt=".2f", cmap="YlOrRd",
    vmin=0, vmax=1, linewidths=0.5, ax=ax,
    cbar_kws={"label": "Attack success rate"}
)

ax.set_xlabel("Behavior category")
ax.set_ylabel("Manipulation tactic")
ax.set_title("Tactic effectiveness by category")
plt.xticks(rotation=35, ha="right")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "tactic_heatmap.pdf", dpi=300)
plt.savefig(Path(__file__).parent / "tactic_heatmap.png", dpi=300)
print("Saved tactic_heatmap.pdf/.png")