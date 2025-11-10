# prepare_fine_tune_dataset.py
# produces a JSONL file you can feed into any fine-tuning pipeline

import pandas as pd, json

df = pd.read_csv("bias_analysis_output/fine_tune_candidates.csv")

data = []
for _, r in df.iterrows():
    data.append({
        "prompt": r["prompt_a"],
        "response": r["response_a"],
        "comparison_response": r["response_b"],
        "contextual_bias_score": r["contextual_bias_score"]
    })

with open("bias_analysis_output/fine_tune_dataset.jsonl", "w", encoding="utf-8") as f:
    for d in data:
        f.write(json.dumps(d, ensure_ascii=False) + "\n")

print("Saved fine_tune_dataset.jsonl ready for training")
