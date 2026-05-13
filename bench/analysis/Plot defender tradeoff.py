"""
Defender Trade-off Curve
------------------------
X = refusal rate on safe prompts (false positive rate — over-cautiousness)
Y = ASR on unsafe prompts (false negative rate — vulnerability)
Each point = one run/defender config. Ideal is bottom-left (low refusal, low ASR).
This is your paper's equivalent of an ROC curve for defenders.

Usage: python plot_defender_tradeoff.py

Note: requires runs with BOTH safe and unsafe prompts to be meaningful.
      If you only have unsafe prompts, the X axis will be empty.
"""
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "benchmark.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query("""
    SELECT t.final_verdict, t.expected_status, r.run_id,
           r.defender_module, r.attacker_module
    FROM traces t
    JOIN runs r ON t.run_id = r.run_id
    WHERE t.status = 'SUCCESS'
""", conn)
conn.close()

if df.empty:
    print("No data found.")
    exit()

results = []
for (run_id, defender, attacker), group in df.groupby(["run_id", "defender_module", "attacker_module"]):
    unsafe_set = group[group["expected_status"] == "unsafe"]
    safe_set = group[group["expected_status"] == "safe"]

    asr = (unsafe_set["final_verdict"] == "unsafe").mean() if len(unsafe_set) > 0 else None
    refusal = (safe_set["final_verdict"] == "guarded").mean() if len(safe_set) > 0 else None

    results.append({
        "defender": defender,
        "attacker": attacker,
        "asr": asr,
        "refusal_rate": refusal,
    })

plot_df = pd.DataFrame(results).dropna()

if plot_df.empty:
    print("Need runs with both safe and unsafe prompts for this chart.")
    print("Tip: create a test set with mix of safe/unsafe expected_status.")
    exit()

sns.set_theme(style="whitegrid", font_scale=1.1)
fig, ax = plt.subplots(figsize=(6, 5))

sns.scatterplot(
    data=plot_df, x="refusal_rate", y="asr",
    hue="defender", style="attacker", s=100, ax=ax
)

ax.set_xlabel("Refusal rate on safe prompts (over-caution)")
ax.set_ylabel("ASR on unsafe prompts (vulnerability)")
ax.set_xlim(-0.05, 1.05)
ax.set_ylim(-0.05, 1.05)
ax.set_title("Defender trade-off: caution vs. vulnerability")
ax.legend(fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")

# ideal region
ax.annotate("Ideal", xy=(0.05, 0.05), fontsize=10, color="green",
            fontstyle="italic", alpha=0.7)

plt.tight_layout()
plt.savefig(Path(__file__).parent / "defender_tradeoff.pdf", dpi=300)
plt.savefig(Path(__file__).parent / "defender_tradeoff.png", dpi=300)
print("Saved defender_tradeoff.pdf/.png")