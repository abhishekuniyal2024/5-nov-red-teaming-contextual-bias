"""
milestone1_gen_output.py
Collect model responses from gemini-2.0-flash, GPT-3.5 turbo, and LLaMA-4 Maverick, 
and save them in a CSV file for contextual bias analysis.
"""

import os
import json
import time
import logging
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import pandas as pd
import openai
import google.generativeai as genai
from tqdm import tqdm

# ---------------- SETUP ----------------
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

openai.api_key = OPENAI_API_KEY
genai.configure(api_key=GEMINI_API_KEY)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"

ROOT = Path(__file__).parent
DATASET_PATH = ROOT / "dataset" / "dataset.json"
OUTPUT_DIR = ROOT / "dataset_response"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# ---------------- SAFE DATASET LOADING ----------------

def extract_json_objects(text):
    objs, brace_count, start_idx = [], 0, None
    in_string, escape = False, False
    for i, ch in enumerate(text):
        if ch == '"' and not escape:
            in_string = not in_string
        if ch == "\\" and not escape:
            escape = True
            continue
        escape = False
        if in_string:
            continue
        if ch == '{':
            if brace_count == 0:
                start_idx = i
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx is not None:
                objs.append(text[start_idx:i + 1])
                start_idx = None
    return objs

def load_dataset_safely(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    blocks = extract_json_objects(raw)
    merged_pairs = []
    for block in blocks:
        try:
            obj = json.loads(block)
            if isinstance(obj, dict) and "pairs" in obj:
                merged_pairs.extend(obj["pairs"])
        except json.JSONDecodeError:
            continue
    return {"pairs": merged_pairs}

# ---------------- MODEL CALLS ----------------

def get_gpt_response(prompt_text):
    """GPT-3.5 Turbo (OpenAI 0.28.x style)"""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt_text}],
            temperature=0.0,
            max_tokens=512
        )
        return response.choices[0].message["content"]
    except Exception as e:
        return f"API Error: {str(e)}"

def get_gemini_response(prompt_text):
    """Gemini 2.0 Flash"""
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt_text,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=512
            )
        )
        if getattr(response, "text", None):
            return response.text
        elif hasattr(response, "candidates") and response.candidates:
            texts = []
            for c in response.candidates:
                if hasattr(c, "content") and getattr(c.content, "parts", None):
                    for p in c.content.parts:
                        if hasattr(p, "text"):
                            texts.append(p.text)
            return " ".join(texts) if texts else "API Error: Empty Gemini response"
        else:
            return "API Error: Empty Gemini response"
    except Exception as e:
        return f"API Error: {str(e)}"

def get_llama_response(prompt_text):
    """LLaMA via Groq API"""
    try:
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "model": GROQ_MODEL,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": 0.0,
            "max_tokens": 512
        }
        response = requests.post(GROQ_API_URL, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"API Error: {str(e)}"

def run_all_models(prompt_text):
    """Run GPT, Gemini, and LLaMA concurrently."""
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(get_gpt_response, prompt_text): "gpt-3.5-turbo",
            executor.submit(get_gemini_response, prompt_text): "gemini-2.0-flash",
            executor.submit(get_llama_response, prompt_text): "llama-4-maverick"
        }
        responses = {}
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                responses[model_name] = future.result()
            except Exception as e:
                responses[model_name] = f"API Error: {str(e)}"
        return responses

# ---------------- MAIN ----------------

if __name__ == "__main__":
    logging.info(f"Loading dataset from {DATASET_PATH}")
    data = load_dataset_safely(DATASET_PATH)
    if not data["pairs"]:
        logging.error("No pairs loaded from dataset.json. Check the file.")
        raise SystemExit(1)

    rows = []
    total_prompts = sum(len(pair.get("prompts", [])) for pair in data["pairs"])
    with tqdm(total=total_prompts, desc="Processing prompts", ncols=100) as pbar:
        for pair in data["pairs"]:
            pair_id, category = pair.get("pair_id", ""), pair.get("category", "")
            for prompt_obj in pair.get("prompts", []):
                pid, text = prompt_obj.get("id", ""), prompt_obj.get("text", "")
                responses = run_all_models(text)
                for model, resp in responses.items():
                    rows.append({
                        "pair_id": pair_id,
                        "category": category,
                        "model": model,
                        "id": pid,
                        "prompt": text,
                        "response": resp
                    })
                pbar.update(1)

    timestamp = int(time.time())
    out_csv = OUTPUT_DIR / f"dataset_response_{timestamp}.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    logging.info(f"Saved {len(rows)} responses to {out_csv}")
