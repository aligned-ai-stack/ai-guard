"""
ASR vs Conversation Length
--------------------------
One line per defender module. X-axis = turns allowed, Y-axis = attack success rate.
Shows how defender resilience degrades over longer conversations.

Usage: python plot_asr_vs_convo_length.py
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "benchmark.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT t.run_id, t.final_verdict, t.turns_used, r.defender_module, r.attacker_module
    FROM traces t
    JOIN runs r ON t.run_id = r.run_id
    WHERE t.expected_status = 'unsafe'
      AND t.status = 'SUCCESS'
""", conn)
conn.close()

if df.empty:
    print("No data found. Run some benchmarks first.")
    exit()

# compute ASR per (defender, turns_used)
grouped = df.groupby(["defender_module", "turns_used"]).apply(
    lambda g: (g["final_verdict"] == "unsafe").mean()
).reset_index(name="asr")

# plot
sns.set_theme(style="whitegrid", font_scale=1.1)
fig, ax = plt.subplots(figsize=(7, 4.5))

for defender, sub in grouped.groupby("defender_module"):
    sub = sub.sort_values("turns_used")
    ax.plot(sub["turns_used"], sub["asr"], marker="o", label=defender, linewidth=2)

ax.set_xlabel("Conversation length (turns)")
ax.set_ylabel("Attack Success Rate (ASR)")
ax.set_ylim(-0.05, 1.05)
ax.legend(title="Defender")
ax.set_title("ASR vs. conversation length")

plt.tight_layout()
plt.savefig(Path(__file__).parent / "asr_vs_convo_length.pdf", dpi=300)
plt.savefig(Path(__file__).parent / "asr_vs_convo_length.png", dpi=300)
print("Saved asr_vs_convo_length.pdf/.png")