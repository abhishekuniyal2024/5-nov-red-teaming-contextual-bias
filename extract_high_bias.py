# extract_high_bias.py

import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
OUT_DIR = ROOT / "bias_analysis_output"

# 1. Paths to the two analysis outputs
comparison_path = OUT_DIR / "contextual_bias_comparison_with_bias_notes.csv"
scores_path     = OUT_DIR / "contextual_bias_results.csv"

# 2. Load files
comp_df   = pd.read_csv(comparison_path)
scores_df = pd.read_csv(scores_path)

# 3. Collapse scores to one per (pair_id, model)
pair_scores = (
    scores_df
    .groupby(["pair_id", "model"], as_index=False)["contextual_bias_score"]
    .mean()
)

# 4. Merge scores into comparison table
merged = comp_df.merge(pair_scores, on=["pair_id", "model"], how="left")

# 5. Filter high-bias examples (tune threshold if you want)
THRESHOLD = 0.4
high_bias = merged[merged["contextual_bias_score"] > THRESHOLD].copy()

# 6. Save final candidates for fine-tuning
out_path = OUT_DIR / "fine_tune_candidates.csv"
high_bias.to_csv(out_path, index=False)

print(f"Saved {len(high_bias)} high-bias examples to {out_path}")
