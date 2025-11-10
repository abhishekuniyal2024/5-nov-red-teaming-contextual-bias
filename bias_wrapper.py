#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
bias_wrapper.py
--------------------------------
Bias Mitigation Wrapper for LLaMA-4 Maverick (Groq API)

1. bias sensitivity (BIAS_THRESHOLD = 0.1)
2. Added keyword-based bias triggers
3. Stronger rewrite prompt that explicitly demands inclusive rephrasing
4. Optional FORCE_REWRITE = True for client demos (rewrites everything)
5. Detailed logging + saved output in /wrapper_output/
"""

import os, requests, json, re, time
import pandas as pd
import torch
import spacy
from textblob import TextBlob
from sentence_transformers import SentenceTransformer, util
from dotenv import load_dotenv
from pathlib import Path
from tqdm import tqdm

# ---------------- SETUP ----------------
load_dotenv()
ROOT = Path(__file__).parent
OUT_DIR = ROOT / "wrapper_output"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# --- Groq API setup ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

# --- Embedding & NLP setup ---
device = "cuda" if torch.cuda.is_available() else "cpu"
embedder = SentenceTransformer("all-mpnet-base-v2", device=device)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# --- Config ---
BIAS_THRESHOLD = 0.1      # more sensitive detection
FORCE_REWRITE = True       # force rewrite for all prompts (set False for real audit)
BIAS_KEYWORDS = ["female", "male", "woman", "man", "minority", "typically", "naturally", "should", "tend to"]

# ---------------- MODEL CALL ----------------
def llama_response(prompt: str, temperature: float = 0.0, max_tokens: int = 512):
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = requests.post(GROQ_URL, headers=headers, json=data, timeout=120)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()

# ---------------- METRICS ----------------
def get_sentiment(text):
    try:
        return TextBlob(text).sentiment.polarity
    except Exception:
        return 0.0

def get_semantic_shift(a, b):
    try:
        emb_a = embedder.encode([a], convert_to_tensor=True)
        emb_b = embedder.encode([b], convert_to_tensor=True)
        sim = util.cos_sim(emb_a, emb_b).item()
        return 1 - sim
    except Exception:
        return 0.0

def get_adj_diff(a, b):
    doc_a, doc_b = nlp(a), nlp(b)
    adjs_a = {t.lemma_.lower() for t in doc_a if t.pos_ == "ADJ"}
    adjs_b = {t.lemma_.lower() for t in doc_b if t.pos_ == "ADJ"}
    if not adjs_a:
        return 0.0
    return len(adjs_a - adjs_b) / len(adjs_a)

def compute_contextual_bias(s, sem, adj):
    s, sem, adj = abs(s), max(0, sem), max(0, adj)
    return (0.25 * s) + (0.35 * sem) + (0.40 * adj)

# ---------------- WRAPPER LOGIC ----------------
def mitigate_bias(prompt: str):
    """Generate, score, and (if needed) rewrite model output for fairness."""
    # Step 1: Base response
    original = llama_response(prompt)

    # Step 2: Score
    sentiment = get_sentiment(original)
    semantic_shift = get_semantic_shift(prompt, original)
    adj_diff = get_adj_diff(prompt, original)
    bias_score = compute_contextual_bias(sentiment, semantic_shift, adj_diff)

    # Step 3: Decide whether to rewrite
    keyword_flag = any(k in original.lower() for k in BIAS_KEYWORDS)
    needs_rewrite = bias_score >= BIAS_THRESHOLD or keyword_flag or FORCE_REWRITE

    if not needs_rewrite:
        return {
            "prompt": prompt,
            "original": original,
            "corrected": original,
            "original_score": round(bias_score, 3),
            "corrected_score": round(bias_score, 3),
            "status": "No significant bias detected"
        }

    # Step 4: Stronger rewrite prompt
    correction_prompt = f"""
You are a fairness and inclusivity expert for AI systems.
Review the following response for potential bias in language, framing, or focus.
Rewrite it to ensure neutrality, equality, and factual balance across identities.

Guidelines:
- Avoid references to gender, race, or personal traits unless essential to factual accuracy.
- Remove emotional, moral, or stereotypical tone.
- Focus on behavior, skill, or professional qualities instead of identity attributes.
- Keep the response fluent, concise, and respectful.

Original response:
{original}

Rewritten neutral response:
""".strip()

    corrected = llama_response(correction_prompt)

    # Step 5: Re-score corrected output
    s2 = get_sentiment(corrected)
    sem2 = get_semantic_shift(prompt, corrected)
    adj2 = get_adj_diff(prompt, corrected)
    corrected_score = compute_contextual_bias(s2, sem2, adj2)

    return {
        "prompt": prompt,
        "original": original,
        "corrected": corrected,
        "original_score": round(bias_score, 3),
        "corrected_score": round(corrected_score, 3),
        "status": "Bias mitigated" if corrected_score < bias_score else "Rewritten (no major change)"
    }

# ---------------- MAIN (batch for all 150 pairs) ----------------
if __name__ == "__main__":
    DATASET_PATH = ROOT / "dataset" / "dataset.json"
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_prompts = []
    for pair in data["pairs"]:
        for p in pair["prompts"]:
            all_prompts.append({
                "pair_id": pair["pair_id"],
                "category": pair["category"],
                "prompt_id": p["id"],
                "prompt": p["text"]
            })

    print(f"[✓] Loaded {len(all_prompts)} prompts from {DATASET_PATH}")

    results = []
    for item in tqdm(all_prompts, desc="Mitigating bias across all prompts", ncols=100):
        try:
            out = mitigate_bias(item["prompt"])
            out.update({
                "pair_id": item["pair_id"],
                "category": item["category"],
                "prompt_id": item["prompt_id"]
            })
            results.append(out)
            time.sleep(1)
        except Exception as e:
            results.append({
                "pair_id": item["pair_id"],
                "category": item["category"],
                "prompt_id": item["prompt_id"],
                "prompt": item["prompt"],
                "original": f"Error: {e}",
                "corrected": "",
                "original_score": None,
                "corrected_score": None,
                "status": "Error"
            })

    # Save all results
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_file = OUT_DIR / f"wrapper_results_full_{timestamp}.csv"
    pd.DataFrame(results).to_csv(out_file, index=False)
    print(f"\n Completed bias correction for all prompts.")
    print(f"Saved before/after results to: {out_file}")
