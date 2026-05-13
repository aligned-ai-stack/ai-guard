"""
Token Efficiency
----------------
Scatter plot. Each point is one run. X = avg tokens per trace, Y = ASR.
Shows the cost-effectiveness of different attacker/defender combos.

Usage: python plot_token_efficiency.py
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "benchmark.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT run_id, asr, avg_tokens_per_trace,
           defender_module, attacker_module
    FROM runs
    WHERE asr IS NOT NULL
      AND avg_tokens_per_trace > 0
""", conn)
conn.close()

if df.empty:
    print("No data found.")
    exit()

df["config"] = df["attacker_module"] + " vs " + df["defender_module"]

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, ax = plt.subplots(figsize=(7, 5))

sns.scatterplot(
    data=df, x="avg_tokens_per_trace", y="asr",
    hue="config", s=80, ax=ax
)

ax.set_xlabel("Avg. tokens per trace")
ax.set_ylabel("Attack Success Rate (ASR)")
ax.set_ylim(-0.05, 1.05)
ax.set_title("Token efficiency: ASR vs. cost")
ax.legend(title="Config", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "token_efficiency.pdf", dpi=300)
plt.savefig(Path(__file__).parent / "token_efficiency.png", dpi=300)
print("Saved token_efficiency.pdf/.png")