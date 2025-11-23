# app.py
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import traceback
import time

# ---------- PERMANENT (fixed) file paths ----------
# These are your Windows files (fixed - no sidebar editing)
DEFAULT_RUBRIC_PATH = r"C:\Users\potnu\Desktop\CS Nirmaann\Case study for interns.xlsx"
DEFAULT_TRANSCRIPT_PATH = r"C:\Users\potnu\Desktop\CS Nirmaann\Sample text for case study.txt"

# This is the screenshot that exists in the chat runtime; optional to display.
SCREENSHOT_PATH = "/mnt/data/ddaf18b9-c7c6-4d08-9772-e08fab72dfce.png"

# ---------- UI / App title ----------
st.set_page_config(page_title="Nirmaan — Intro Scorer", layout="centered")
st.title("Nirmaan — Intro Scorer (Case Study Demo)")
st.markdown(
    "Paste transcript or upload a `.txt` file. Uses rubric Excel to compute per-criterion scores and overall score."
)

# ---------- Lazy model holder ----------
_model = None

def load_model_lazy(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load sentence-transformers model lazily. This prevents blocking the initial UI render.
    """
    global _model
    if _model is None:
        # Import here to avoid heavy import at top-level
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(model_name)
    return _model

# ---------- Rubric loader ----------
def load_rubric(path: str) -> pd.DataFrame:
    """
    Load rubric Excel robustly. Raises descriptive errors if file not found or invalid.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rubric file not found at: {path}")
    try:
        df = pd.read_excel(p)
    except Exception as e:
        raise RuntimeError(f"Failed to read Excel file: {e}")
    # Normalize column names (strip)
    df.columns = [str(c).strip() for c in df.columns]
    return df

# ---------- Helpers ----------
def extract_keywords(kw_cell) -> List[str]:
    if pd.isna(kw_cell) or kw_cell is None:
        return []
    if isinstance(kw_cell, (list, tuple)):
        return [str(k).strip().lower() for k in kw_cell if str(k).strip()]
    s = str(kw_cell)
    for sep in [",", ";", "\n", "|"]:
        if sep in s:
            parts = [p.strip().lower() for p in s.split(sep) if p.strip()]
            if parts:
                return parts
    return [s.strip().lower()] if s.strip() else []

def keyword_score(transcript: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    t = transcript.lower()
    found = sum(1 for k in keywords if k in t)
    return found / len(keywords)

def length_score(transcript: str, min_w, max_w) -> float:
    words = len(transcript.split())
    # if both min and max are missing => full score
    try:
        has_min = not pd.isna(min_w)
    except Exception:
        has_min = False
    try:
        has_max = not pd.isna(max_w)
    except Exception:
        has_max = False

    if not has_min and not has_max:
        return 1.0

    try:
        min_w_val = float(min_w) if has_min else None
    except Exception:
        min_w_val = None
    try:
        max_w_val = float(max_w) if has_max else None
    except Exception:
        max_w_val = None

    if min_w_val and words < min_w_val:
        return max(0.0, words / (min_w_val))
    if max_w_val and words > max_w_val:
        return max(0.0, (max_w_val / words))
    return 1.0

def semantic_score(model, transcript: str, criterion_text: str) -> float:
    # Guard: if either is empty, return conservative defaults
    if not criterion_text.strip():
        return 1.0
    if not transcript.strip():
        return 0.0
    emb_t = model.encode(transcript, convert_to_tensor=True)
    emb_c = model.encode(criterion_text, convert_to_tensor=True)
    from sentence_transformers import util
    sim = util.cos_sim(emb_t, emb_c).item()
    sim = float(np.clip(sim, 0.0, 1.0))
    return sim

def compute_scores(df_rubric: pd.DataFrame, transcript: str, model) -> dict:
    """
    Return dict with overall_score (0-100), words, and per_criterion list.
    Expects df_rubric with columns that may include: criterion/description/keywords/weight/min_words/max_words
    """
    # Decide mapping for columns - support several common names
    col_map = {}
    for name in ["criterion", "criteria", "title"]:
        for c in df_rubric.columns:
            if name in c.lower():
                col_map['criterion'] = c
                break
        if 'criterion' in col_map:
            break
    for name in ["description", "criterion_description", "details"]:
        for c in df_rubric.columns:
            if name in c.lower():
                col_map['description'] = c
                break
        if 'description' in col_map:
            break
    for name in ["keyword", "keywords", "keyphrases"]:
        for c in df_rubric.columns:
            if name in c.lower():
                col_map['keywords'] = c
                break
        if 'keywords' in col_map:
            break
    for name in ["weight", "points", "score_weight"]:
        for c in df_rubric.columns:
            if name in c.lower():
                col_map['weight'] = c
                break
        if 'weight' in col_map:
            break
    for name in ["min_word", "min_words", "min_word_limit"]:
        for c in df_rubric.columns:
            if name in c.lower():
                col_map['min_words'] = c
                break
        if 'min_words' in col_map:
            break
    for name in ["max_word", "max_words", "max_word_limit"]:
        for c in df_rubric.columns:
            if name in c.lower():
                col_map['max_words'] = c
                break
        if 'max_words' in col_map:
            break

    # Fill defaults if missing
    if 'criterion' not in col_map:
        col_map['criterion'] = df_rubric.columns[0]
    if 'description' not in col_map:
        col_map['description'] = df_rubric.columns[0]
    if 'keywords' not in col_map:
        df_rubric['__keywords__'] = ""
        col_map['keywords'] = '__keywords__'
    if 'weight' not in col_map:
        df_rubric['__weight__'] = 1.0
        col_map['weight'] = '__weight__'

    alpha, beta, gamma = 0.5, 0.4, 0.1
    results = []
    weighted_sum = 0.0

    # compute total weight (safe)
    try:
        total_weight = float(df_rubric[col_map['weight']].astype(float).sum())
    except Exception:
        total_weight = len(df_rubric) * 1.0

    for _, row in df_rubric.iterrows():
        crit = str(row.get(col_map['criterion'], "")).strip()
        desc = str(row.get(col_map['description'], "")).strip()
        kws = extract_keywords(row.get(col_map['keywords'], ""))
        try:
            w = float(row.get(col_map['weight'], 1.0))
        except Exception:
            w = 1.0
        min_w = row.get(col_map.get('min_words', None), np.nan)
        max_w = row.get(col_map.get('max_words', None), np.nan)

        ks = keyword_score(transcript, kws)
        ls = length_score(transcript, min_w, max_w)
        try:
            ss = semantic_score(model, transcript, desc) if desc else 1.0
        except Exception:
            ss = 0.0

        raw = alpha * ks + beta * ss + gamma * ls
        weighted = raw * w
        weighted_sum += weighted

        suggestion = ""
        if ks < 0.5:
            suggestion += "Mention more of the expected keywords. "
        if ss < 0.5:
            suggestion += "Make your answer closer to the criterion description. "
        if ls < 0.8:
            suggestion += "Adjust length to meet recommended bounds. "

        feedback = {
            "criterion": crit or "Unnamed criterion",
            "description": desc,
            "keywords": kws,
            "keyword_score": round(ks, 3),
            "semantic_score": round(ss, 3),
            "length_score": round(ls, 3),
            "raw_score": round(raw, 3),
            "weight": w,
            "suggestion": suggestion.strip()
        }
        results.append(feedback)

    overall = (weighted_sum / total_weight) * 100 if total_weight > 0 else 0.0
    return {"overall_score": round(overall, 2), "per_criterion": results, "words": len(transcript.split())}

# ---------- Sidebar: show fixed paths and debug option ----------
st.sidebar.header("Configuration (fixed)")
st.sidebar.write("Rubric (fixed):")
st.sidebar.code(DEFAULT_RUBRIC_PATH)
st.sidebar.write("Transcript (fixed):")
st.sidebar.code(DEFAULT_TRANSCRIPT_PATH)
st.sidebar.checkbox("Show debug logs / tracebacks", value=True, key="show_debug_checkbox")

# ---------- Show uploaded screenshot (if available) ----------
if Path(SCREENSHOT_PATH).exists():
    st.sidebar.image(SCREENSHOT_PATH, caption="Runtime screenshot (optional)", use_column_width=True)

# ---------- Main UI: transcript input / uploader ----------
st.subheader("Transcript input")
uploaded = st.file_uploader("Upload transcript (.txt) — optional", type=["txt"])
transcript_text = ""

# If user uploaded file, prefer that
if uploaded is not None:
    try:
        transcript_text = uploaded.getvalue().decode("utf-8")
    except Exception:
        try:
            transcript_text = uploaded.getvalue().decode("latin-1")
        except Exception:
            transcript_text = ""

# If no uploaded file, attempt to auto-load the fixed transcript path
if not transcript_text:
    p = Path(DEFAULT_TRANSCRIPT_PATH)
    if p.exists():
        try:
            transcript_text = p.read_text()
            st.info(f"Auto-loaded transcript from fixed path.")
        except Exception as e:
            st.warning(f"Could not auto-load transcript: {e}")
    else:
        st.warning("No transcript found at fixed path and no file uploaded. Paste transcript in the box below.")

transcript_text = st.text_area("Transcript (paste/edit here)", value=transcript_text, height=220)

# ---------- Buttons ----------
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Load Rubric (test)"):
        try:
            df_loaded = load_rubric(DEFAULT_RUBRIC_PATH)
            st.success(f"Rubric loaded: {len(df_loaded)} rows. Columns: {list(df_loaded.columns)}")
            st.write(df_loaded.head(10))
        except Exception as e:
            st.error(f"Failed to load rubric: {e}")
            st.text(traceback.format_exc())

with col2:
    score_click = st.button("Score")

# ---------- Scoring flow ----------
if score_click:
    status = st.empty()
    try:
        status.info("Loading rubric (fixed path)...")
        df_rubric = load_rubric(DEFAULT_RUBRIC_PATH)

        status.info("Rubric loaded. Preparing scoring engine...")

        status.info("Loading model (this may take a minute on first run)...")
        t0 = time.time()
        model = load_model_lazy("all-MiniLM-L6-v2")
        t1 = time.time()
        status.success(f"Model loaded in {t1 - t0:.1f}s")

        status.info("Computing scores...")
        out = compute_scores(df_rubric, transcript_text or "", model)

        # Display results
        st.metric("Overall Score (0-100)", out['overall_score'])
        st.write("Word count:", out['words'])
        st.subheader("Per-criterion breakdown")
        for pc in out['per_criterion']:
            st.markdown(f"**{pc['criterion']}** (weight: {pc['weight']})")
            st.write({
                "keyword_score": pc['keyword_score'],
                "semantic_score": pc['semantic_score'],
                "length_score": pc['length_score'],
                "raw_score": pc['raw_score'],
                "suggestion": pc['suggestion']
            })
        st.subheader("Full JSON output")
        st.json(out)

    except Exception as e:
        st.error(f"Scoring failed: {e}")
        st.text(traceback.format_exc())
        status.empty()
