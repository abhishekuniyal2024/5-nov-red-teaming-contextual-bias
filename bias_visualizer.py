#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bias_visualizer.py
--------------------------------
Visual summary of contextual bias mitigation results.
Reads output CSV from pair_bias_wrapper.py and generates:
1. Bias score comparison (before vs after)
2. Category-wise average bias reduction
3. Distribution of bias changes across all pairs
4. Optional: bar chart of mitigation status counts
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------------- SETUP ----------------
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "wrapper_output"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------------- LOAD DATA ----------------
file_path = OUT_DIR / "pair_wrapper_results.csv"
df = pd.read_csv(file_path)

# Compute bias reduction
df["bias_reduction"] = df["bias_score_before"] - df["bias_score_after"]

# ---------------- 1. Overall Scatter Comparison ----------------
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x="bias_score_before", y="bias_score_after",
                hue="category", palette="viridis", s=80)
plt.plot([0, df["bias_score_before"].max()], [0, df["bias_score_before"].max()],
         color="red", linestyle="--", label="No change line")
plt.title("Bias Score Before vs After Wrapper")
plt.xlabel("Bias Score (Before Wrapper)")
plt.ylabel("Bias Score (After Wrapper)")
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "bias_before_vs_after.png", dpi=300)
plt.close()

# ---------------- 2. Category-wise Average Reduction ----------------
plt.figure(figsize=(7, 5))
avg_df = df.groupby("category")[["bias_score_before", "bias_score_after"]].mean().reset_index()
avg_df["reduction"] = avg_df["bias_score_before"] - avg_df["bias_score_after"]
sns.barplot(data=avg_df, x="category", y="reduction", palette="coolwarm")
plt.title("Average Bias Reduction by Category")
plt.ylabel("Average Reduction (Before - After)")
plt.xlabel("Category")
plt.tight_layout()
plt.savefig(OUT_DIR / "avg_bias_reduction_by_category.png", dpi=300)
plt.close()

# ---------------- 3. Distribution of Reductions ----------------
plt.figure(figsize=(8, 5))
sns.histplot(df["bias_reduction"], bins=10, kde=True, color="teal")
plt.title("Distribution of Bias Reduction Across All Pairs")
plt.xlabel("Bias Reduction (Before - After)")
plt.ylabel("Count of Pairs")
plt.tight_layout()
plt.savefig(OUT_DIR / "bias_reduction_distribution.png", dpi=300)
plt.close()

# ---------------- 4. Mitigation Status Summary ----------------
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="status", palette="crest")
plt.title("Bias Mitigation Outcomes")
plt.xlabel("Status")
plt.ylabel("Number of Pairs")
plt.tight_layout()
plt.savefig(OUT_DIR / "bias_mitigation_status.png", dpi=300)
plt.close()

print("\nVisualization complete.")
print(f"Charts saved to: {OUT_DIR}")
print("Generated: bias_before_vs_after.png, avg_bias_reduction_by_category.png, "
      "bias_reduction_distribution.png, bias_mitigation_status.png")

# ---------------- 5. Generate Summary Report ----------------
summary_path = OUT_DIR / "summary_report.txt"

total_pairs = len(df)
mitigated_count = (df["status"] == "Bias mitigated").sum()
no_change_count = (df["status"] == "No improvement").sum()
avg_before = df["bias_score_before"].mean()
avg_after = df["bias_score_after"].mean()
avg_drop = avg_before - avg_after
percent_reduction = (avg_drop / avg_before * 100) if avg_before > 0 else 0

with open(summary_path, "w", encoding="utf-8") as f:
    f.write("=== BIAS MITIGATION WRAPPER SUMMARY REPORT ===\n")
    f.write(f"Total pairs analyzed: {total_pairs}\n")
    f.write(f"Bias mitigated pairs: {mitigated_count} ({mitigated_count/total_pairs*100:.1f}%)\n")
    f.write(f"Average bias score drop: {avg_drop:.3f} ({percent_reduction:.1f}% reduction)\n")
    f.write(f"Final average bias score after mitigation: {avg_after:.3f}\n")
    f.write("==============================================\n")

print(f"\nSummary report saved to: {summary_path}")
