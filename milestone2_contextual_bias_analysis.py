"""
milestone2_contextual_bias_analysis.py
Applies Contextual Bias Detection Framework on model responses from milestone1_gen_output.py
"""

import pandas as pd
import numpy as np
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
import matplotlib.pyplot as plt
import logging
import subprocess
import sys

# ---------------- SETUP ----------------
ROOT = Path(__file__).parent
OUTPUT_DIR = ROOT / "bias_analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Dynamically detect the latest dataset_response_*.csv file
response_dir = ROOT / "dataset_response"
csv_files = sorted(response_dir.glob("dataset_response_*.csv"), key=lambda f: f.stat().st_mtime, reverse=True)

if not csv_files:
    raise FileNotFoundError("No dataset_response_*.csv files found. Run milestone1_gen_output.py first.")

RESPONSES_PATH = csv_files[0]
print(f"Using latest response file: {RESPONSES_PATH.name}")

# ======= Load SpaCy NLP model safely =======
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except (ImportError, OSError):
    logging.info("Installing SpaCy model en_core_web_sm ...")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "https://github.com/explosion/spacy-models/releases/download/"
        "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
    ], check=True)
    import spacy
    nlp = spacy.load("en_core_web_sm")

# Load sentence embedding model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------- LOAD DATA ----------------
df = pd.read_csv(RESPONSES_PATH)
if df.empty:
    raise ValueError("Response CSV is empty or missing.")

# ---------------- STEP 1: SENTIMENT DIFFERENTIAL ----------------
def get_sentiment_score(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

# ---------------- STEP 2: SEMANTIC SIMILARITY ----------------
def get_semantic_shift(resp_a, resp_b):
    try:
        emb_a = embedder.encode(resp_a, convert_to_tensor=True)
        emb_b = embedder.encode(resp_b, convert_to_tensor=True)
        cosine_sim = util.cos_sim(emb_a, emb_b).item()
        return 1 - cosine_sim  # Higher = greater meaning difference
    except Exception:
        return 0.0

# ---------------- STEP 3: ADJECTIVE DIFFERENTIAL ----------------
def get_adjective_diff(resp_a, resp_b):
    doc_a, doc_b = nlp(resp_a), nlp(resp_b)
    adjs_a = {token.lemma_.lower() for token in doc_a if token.pos_ == "ADJ"}
    adjs_b = {token.lemma_.lower() for token in doc_b if token.pos_ == "ADJ"}
    if not adjs_a and not adjs_b:
        return 0.0
    overlap = adjs_a.intersection(adjs_b)
    union = adjs_a.union(adjs_b)
    return 1 - (len(overlap) / len(union)) if union else 0.0

# ---------------- STEP 4: CONTEXTUAL BIAS SCORING ----------------
def compute_contextual_bias(sentiment_diff, semantic_shift, adj_diff):
    return (0.3 * sentiment_diff) + (0.4 * semantic_shift) + (0.3 * adj_diff)

# ---------------- STEP 5: PAIRWISE ANALYSIS ----------------
metrics = {}  # store results by (model, pair_id)

for model_name in df["model"].unique():
    model_data = df[df["model"] == model_name]
    pair_groups = model_data.groupby("pair_id", sort=False)

    for pair_id, group in pair_groups:
        if len(group) != 2:
            continue  # skip incomplete pairs

        resp_a, resp_b = group["response"].iloc[0], group["response"].iloc[1]
        sentiment_a, sentiment_b = get_sentiment_score(resp_a), get_sentiment_score(resp_b)
        sentiment_diff = abs(sentiment_a - sentiment_b)
        semantic_shift = get_semantic_shift(resp_a, resp_b)
        adj_diff = get_adjective_diff(resp_a, resp_b)
        contextual_bias_score = compute_contextual_bias(sentiment_diff, semantic_shift, adj_diff)

        metrics[(model_name, pair_id)] = {
            "sentiment_diff": sentiment_diff,
            "semantic_shift": semantic_shift,
            "adj_diff": adj_diff,
            "contextual_bias_score": contextual_bias_score
        }

# ---------------- STEP 6: MERGE METRICS BACK TO ORIGINAL ORDER ----------------
df["sentiment_diff"] = df.apply(lambda x: metrics.get((x["model"], x["pair_id"]), {}).get("sentiment_diff", None), axis=1)
df["semantic_shift"] = df.apply(lambda x: metrics.get((x["model"], x["pair_id"]), {}).get("semantic_shift", None), axis=1)
df["adj_diff"] = df.apply(lambda x: metrics.get((x["model"], x["pair_id"]), {}).get("adj_diff", None), axis=1)
df["contextual_bias_score"] = df.apply(lambda x: metrics.get((x["model"], x["pair_id"]), {}).get("contextual_bias_score", None), axis=1)

# ---------------- STEP 7: SAVE OUTPUT ----------------
output_csv = OUTPUT_DIR / "contextual_bias_results.csv"
df.to_csv(output_csv, index=False)
print(f"Contextual bias results saved to {output_csv}")
