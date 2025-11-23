# Nirmaan — Intro Scorer (Case Study)

## What this is
A small Streamlit app that scores a spoken-introduction transcript using a rubric Excel.  
Outputs: overall_score (0–100), per_criterion array (keyword/semantic/length scores), and word count.

## Files
- `app.py` — Streamlit app
- `requirements.txt` — Python deps
- `Case study for interns.xlsx` — rubric (optional, provide locally)
- `Sample text for case study.txt` — sample transcript (optional)

## How scoring works (formula)
For each rubric item:
- `keyword_score` = (#keywords present) / (#keywords listed)  (if no keywords → 1.0)
- `semantic_score` = cosine(embedding(transcript), embedding(criterion_description)) if model available; else a Jaccard heuristic
- `length_score` = 1 if within min/max word bounds; otherwise linear penalty
- `raw_score` = 0.5 * keyword_score + 0.4 * semantic_score + 0.1 * length_score
- `weighted` = raw_score * rubric_weight
Overall score = (sum of weighted) / (sum of weights) * 100

## Run locally (short)
1. `git clone <your-repo-url>`
2. `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `streamlit run app.py`

## Notes
- The app will attempt to download a small embedding model the first time. If your environment can't use `sentence-transformers` (Keras/TF incompatibility), the app falls back to a lightweight semantic heuristic.
- If you want reproducible results in offline mode, pre-download or remove the model usage in `app.py`.

