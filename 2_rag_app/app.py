# -*- coding: utf-8 -*-
"""
app.py — Congressional Bills AI · Streamlit RAG Chatbot

Semantic search + passage-prediction over 2,200+ bills from the
118th U.S. Congress (2023-2024).

Run:  streamlit run app.py
Requires GROQ_API_KEY in .env
"""

import os
import re
import pickle
import concurrent.futures

import chromadb
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

from config import (
    GROQ_API_KEY,
    CHROMA_PATH,
    PREDICTOR_PATH,
    CHUNKS_CSV_PATH,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_TEMPERATURE,
    FEATURES,
    BILL_TYPE_MAP,
    RESOLUTION_TYPES,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LegisAI — Congressional Bills Intelligence",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS — LIGHT PROFESSIONAL THEME
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── Base ──────────────────────────────────────────────────────────────────── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: #f0f4fb;
    min-height: 100vh;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1100px; }

/* ── Hero ──────────────────────────────────────────────────────────────────── */
.hero {
    background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 50%, #4f46e5 100%);
    border-radius: 18px;
    padding: 2.2rem 2.8rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    right: -60px; top: -60px;
    width: 260px; height: 260px;
    border-radius: 50%;
    background: rgba(255,255,255,0.05);
}
.hero::before {
    content: '';
    position: absolute;
    right: 80px; bottom: -80px;
    width: 180px; height: 180px;
    border-radius: 50%;
    background: rgba(255,255,255,0.04);
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.2);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 0.7rem;
    font-weight: 600;
    color: rgba(255,255,255,0.85);
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.1rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.4rem 0;
    line-height: 1.2;
}
.hero-sub {
    color: rgba(255,255,255,0.65);
    font-size: 0.9rem;
    margin: 0;
}

/* ── Search ────────────────────────────────────────────────────────────────── */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #c7d4f0 !important;
    border-radius: 10px !important;
    color: #1a1f36 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1.1rem !important;
    box-shadow: 0 1px 3px rgba(60,80,180,0.06) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #4f46e5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder { color: #9ca3af !important; }
.stTextInput label {
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 0.83rem !important;
    letter-spacing: 0.03em !important;
}

/* ── AI Answer ──────────────────────────────────────────────────────────────── */
.ai-box {
    background: #ffffff;
    border: 1px solid #e0e7ff;
    border-left: 3px solid #4f46e5;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    color: #1e293b;
    font-size: 0.94rem;
    line-height: 1.75;
    margin: 0.5rem 0 1.5rem 0;
    box-shadow: 0 1px 4px rgba(79,70,229,0.07);
}
.ai-label {
    font-size: 0.68rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #4f46e5;
    margin-bottom: 0.5rem;
    display: flex;
    align-items: center;
    gap: 5px;
}

/* ── Summary box ────────────────────────────────────────────────────────────── */
.summary-box {
    background: #f8faff;
    border: 1px solid #e0e7ff;
    border-radius: 10px;
    padding: 0.9rem 1.2rem;
    color: #374151;
    font-size: 0.84rem;
    line-height: 1.6;
    margin-bottom: 1.2rem;
}
.summary-box strong { color: #1e293b; }

/* ── KPI cards ──────────────────────────────────────────────────────────────── */
.kpi-row {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.9rem;
    margin: 1rem 0 1.5rem 0;
}
.kpi {
    background: #ffffff;
    border: 1px solid #e2e8f4;
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    box-shadow: 0 1px 3px rgba(60,80,180,0.06);
}
.kpi-label {
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 4px;
}
.kpi-val {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem;
    font-weight: 700;
    color: #0f172a;
    line-height: 1.1;
}
.kpi-sub { font-size: 0.72rem; color: #94a3b8; margin-top: 2px; }

/* ── Section header ─────────────────────────────────────────────────────────── */
.section-head {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1rem;
    font-weight: 600;
    color: #1e293b;
    margin: 1.8rem 0 0.9rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f4;
    display: flex;
    align-items: center;
    gap: 7px;
}

/* ── Bill cards ──────────────────────────────────────────────────────────────── */
.bill-card {
    background: #ffffff;
    border: 1px solid #e2e8f4;
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 1px 4px rgba(60,80,180,0.06);
    position: relative;
    overflow: hidden;
    transition: box-shadow 0.2s, transform 0.15s;
}
.bill-card:hover {
    box-shadow: 0 4px 16px rgba(60,80,180,0.12);
    transform: translateY(-1px);
}
.bill-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 4px; height: 100%;
}
.bill-card.passed::before  { background: linear-gradient(180deg, #059669, #34d399); }
.bill-card.failed::before  { background: linear-gradient(180deg, #dc2626, #f87171); }
.bill-card.unknown::before { background: #d1d5db; }

.bill-top {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.8rem;
}
.bill-id {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    background: #eef2ff;
    border: 1px solid #c7d2fe;
    border-radius: 5px;
    padding: 2px 7px;
    color: #4338ca;
    display: inline-block;
    margin-bottom: 5px;
}
.match {
    font-size: 0.72rem;
    font-weight: 500;
    color: #94a3b8;
    white-space: nowrap;
}
.bill-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.93rem;
    font-weight: 600;
    color: #0f172a;
    line-height: 1.4;
}
.pills {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-bottom: 0.8rem;
}
.pill {
    font-size: 0.7rem;
    font-weight: 500;
    border-radius: 20px;
    padding: 3px 9px;
    white-space: nowrap;
}
.pill.passed { background:#dcfce7; border:1px solid #bbf7d0; color:#15803d; }
.pill.failed { background:#fee2e2; border:1px solid #fecaca; color:#b91c1c; }
.pill.d      { background:#dbeafe; border:1px solid #bfdbfe; color:#1d4ed8; }
.pill.r      { background:#fce7f3; border:1px solid #fbcfe8; color:#be185d; }
.pill.i      { background:#f3e8ff; border:1px solid #e9d5ff; color:#7e22ce; }
.pill.dim    { background:#f1f5f9; border:1px solid #e2e8f0; color:#64748b; }

/* ── Probability bar ────────────────────────────────────────────────────────── */
.prob-wrap { margin: 0.7rem 0; }
.prob-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 5px;
}
.prob-lbl {
    font-size: 0.72rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: #94a3b8;
}
.prob-val { font-family:'Space Grotesk',sans-serif; font-size:0.85rem; font-weight:700; }
.prob-val.high   { color: #059669; }
.prob-val.medium { color: #d97706; }
.prob-val.low    { color: #dc2626; }
.prob-track {
    width: 100%;
    height: 6px;
    background: #f1f5f9;
    border-radius: 3px;
    overflow: hidden;
}
.prob-bar {
    height: 100%;
    border-radius: 3px;
}
.prob-bar.high   { background: linear-gradient(90deg,#059669,#34d399); }
.prob-bar.medium { background: linear-gradient(90deg,#d97706,#fbbf24); }
.prob-bar.low    { background: linear-gradient(90deg,#dc2626,#f87171); }

/* ── Signals ────────────────────────────────────────────────────────────────── */
.signals { display:flex; flex-wrap:wrap; gap:5px; margin-top:0.5rem; }
.sig {
    font-size: 0.69rem;
    font-weight: 500;
    border-radius: 4px;
    padding: 2px 7px;
}
.sig.pos { background:#f0fdf4; border:1px solid #bbf7d0; color:#15803d; }
.sig.neu { background:#f8fafc; border:1px solid #e2e8f0; color:#64748b; }

/* ── Snippet ────────────────────────────────────────────────────────────────── */
.snippet {
    background: #f8faff;
    border: 1px solid #e0e7ff;
    border-radius: 8px;
    padding: 0.7rem 0.9rem;
    font-size: 0.8rem;
    color: #475569;
    line-height: 1.6;
    margin: 0.6rem 0;
    font-style: italic;
}
.action-line {
    font-size: 0.72rem;
    color: #94a3b8;
    border-top: 1px solid #f1f5f9;
    padding-top: 0.4rem;
    margin-top: 0.4rem;
}
.action-line b { color: #64748b; }

/* ── Sidebar ────────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #e2e8f4 !important;
}
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label {
    color: #374151 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}
.sb-brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.15rem;
    font-weight: 700;
    color: #0f172a;
}
.sb-desc { font-size:0.78rem; color:#6b7280; line-height:1.5; margin-bottom:0.8rem; }
.sb-section {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #4f46e5;
    margin: 1.2rem 0 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 1px solid #e0e7ff;
}

/* ── Empty state ────────────────────────────────────────────────────────────── */
.empty {
    text-align: center;
    padding: 3.5rem 0;
    color: #94a3b8;
}
.empty-icon { font-size:2.8rem; margin-bottom:0.8rem; }
.empty-title { font-family:'Space Grotesk',sans-serif; font-size:1rem; font-weight:600; color:#64748b; }
.empty-hint  { font-size:0.8rem; margin-top:0.4rem; }

/* ── Misc ────────────────────────────────────────────────────────────────────── */
.stSpinner > div { border-top-color: #4f46e5 !important; }
::-webkit-scrollbar { width:5px; }
::-webkit-scrollbar-track { background:#f1f5f9; }
::-webkit-scrollbar-thumb { background:#c7d2fe; border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embedder():
    return SentenceTransformer(EMBEDDING_MODEL)

@st.cache_resource(show_spinner=False)
def load_chroma():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(CHROMA_COLLECTION)

@st.cache_resource(show_spinner=False)
def load_predictor():
    with open(PREDICTOR_PATH, "rb") as f:
        return pickle.load(f)

@st.cache_data(show_spinner=False)
def load_chunks():
    return pd.read_csv(CHUNKS_CSV_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────────────────────────────────────

with st.spinner("Loading models…"):
    embedder   = load_embedder()
    collection = load_chroma()
    predictor  = load_predictor()
    chunks_df  = load_chunks()

api_key = GROQ_API_KEY or os.environ.get("GROQ_API_KEY", "")
if not api_key:
    st.error("**GROQ_API_KEY not set.** Copy `.env.example` → `.env`, add your key, restart.")
    st.stop()

groq_client = Groq(api_key=api_key)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def esc(text: str) -> str:
    """
    Escape text for safe embedding inside HTML within st.markdown().
    Prevents Streamlit's markdown parser from interpreting bill text
    (e.g. [H.R. 598] notation) as markdown links or other constructs.
    """
    if not text:
        return ""
    text = str(text)
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    text = text.replace("[", "&#91;")
    text = text.replace("]", "&#93;")
    text = text.replace("(", "&#40;")
    text = text.replace(")", "&#41;")
    text = text.replace("*", "&#42;")
    text = text.replace("_", "&#95;")
    text = text.replace("`", "&#96;")
    text = text.replace("#", "&#35;")
    return text


def prob_tier(pct: float) -> str:
    if pct >= 60: return "high"
    if pct >= 30: return "medium"
    return "low"


def predict_passage(meta: dict) -> tuple[float, list[dict]]:
    model    = predictor["model"]
    features = predictor["features"]
    title    = meta.get("title", "")
    text     = meta.get("text", "")

    row = {
        "is_senate":          1 if meta.get("chamber") == "Senate" else 0,
        "is_democrat":        1 if meta.get("party") == "D" else 0,
        "is_republican":      1 if meta.get("party") == "R" else 0,
        "is_resolution":      1 if meta.get("bill_type", "") in RESOLUTION_TYPES else 0,
        "bill_type_enc":      BILL_TYPE_MAP.get(meta.get("bill_type", ""), 8),
        "bypassed_committee": int(meta.get("bypassed_committee", 0)),
        "num_committees":     int(meta.get("num_committees", 0)),
        "text_length":        len(text),
        "word_count":         len(text.split()),
        "title_word_count":   len(title.split()),
        "is_naming_bill":     1 if re.search(r'\bnaming\b|\bpost office\b|\bdesignate\b', title, re.I) else 0,
        "has_appropriation":  1 if re.search(r'approp|fund|billion|million|\$\s*\d', text, re.I) else 0,
        "has_health":         1 if re.search(r'health|medical|medicare|medicaid|hospital', text, re.I) else 0,
        "has_defense":        1 if re.search(r'defense|military|armed forces|veteran|national security', text, re.I) else 0,
        "has_education":      1 if re.search(r'education|school|student|university|college', text, re.I) else 0,
    }

    X    = pd.DataFrame([row])[features]
    prob = float(model.predict_proba(X)[0][1])

    signals = []
    if row["bypassed_committee"]: signals.append({"l": "Bypassed committee (fast-tracked)", "t": "pos"})
    if row["is_resolution"]:      signals.append({"l": "Simple resolution (high pass rate)", "t": "pos"})
    if row["is_senate"]:          signals.append({"l": "Senate bill",                        "t": "pos"})
    if row["num_committees"] > 1: signals.append({"l": f"{row['num_committees']} committees", "t": "pos"})
    if row["is_naming_bill"]:     signals.append({"l": "Naming / designation bill",           "t": "pos"})
    if row["word_count"] > 2000:  signals.append({"l": "Detailed legislation (>2k words)",    "t": "pos"})
    if row["is_democrat"]:        signals.append({"l": "Democrat sponsor",                     "t": "neu"})
    if row["is_republican"]:      signals.append({"l": "Republican sponsor",                   "t": "neu"})
    if not signals:               signals.append({"l": "No strong signals found",              "t": "neu"})

    return prob, signals


def query_bills(question: str, n_results: int = 5, filters: dict | None = None) -> dict:
    emb = embedder.encode([question])[0].tolist()
    kwargs = {
        "query_embeddings": [emb],
        "n_results":        n_results,
        "include":          ["documents", "metadatas", "distances"],
    }
    if filters:
        kwargs["where"] = filters
    return collection.query(**kwargs)


def generate_answer(question: str, docs: list, metas: list) -> str:
    context = ""
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        context += (
            f"\n[Bill {i+1}: {meta['title']} "
            f"({meta['bill_id'].upper()}) — {meta.get('passed_label','?').upper()}]\n"
            f"{doc[:800]}\n"
        )
    prompt = f"""You are an expert on U.S. Congressional legislation.
Answer the user's question based ONLY on the bill excerpts below.
Be specific — reference bill IDs and titles. Keep to 3-5 sentences.
If bills don't contain enough info, say so honestly.

BILL EXCERPTS:
{context}

QUESTION: {question}

ANSWER:"""
    resp = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    return resp.choices[0].message.content.strip()


def summarize_snippet(title: str, doc: str) -> str:
    """Uses Groq to generate a 1-sentence plain English summary of a bill snippet."""
    prompt = (
        "Write a clear, one-sentence plain English summary (max 25 words) explaining the practical purpose of this bill. "
        "Do not use introductory phrases like 'This bill aims to'. Just state what it does.\n\n"
        f"Title: {title}\n\nText: {doc[:1200]}"
    )
    try:
        resp = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=45,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip(' "')
    except Exception:
        # Fallback if API fails
        return doc[:250] + "..."


def results_summary(metas: list, docs: list) -> str:
    """Auto-generate a plain-English summary of the result set."""
    n = len(metas)
    passed = sum(1 for m in metas if m.get("passed_label") == "passed")
    party_d = sum(1 for m in metas if m.get("party") == "D")
    party_r = sum(1 for m in metas if m.get("party") == "R")
    senate  = sum(1 for m in metas if m.get("chamber") == "Senate")
    house   = sum(1 for m in metas if m.get("chamber") == "House")

    parts = [f"<strong>{n} bill{'s' if n != 1 else ''}</strong> retrieved."]

    if passed == 0:
        parts.append("None have been enacted into law.")
    elif passed == n:
        parts.append("All have been enacted.")
    else:
        parts.append(f"<strong>{passed}</strong> of {n} passed into law.")

    if party_d > party_r and party_d > 0:
        parts.append(f"Predominantly <strong>Democrat-sponsored</strong> ({party_d}/{n}).")
    elif party_r > party_d and party_r > 0:
        parts.append(f"Predominantly <strong>Republican-sponsored</strong> ({party_r}/{n}).")
    elif party_d == party_r and party_d > 0:
        parts.append("Bipartisan mix of sponsors.")

    if senate > 0 and house > 0:
        parts.append(f"<strong>{senate} Senate</strong> and <strong>{house} House</strong> bills.")
    elif senate > house:
        parts.append(f"All from the <strong>Senate</strong>.")
    elif house > senate:
        parts.append(f"All from the <strong>House</strong>.")

    return " ".join(parts)


def render_bill_card(doc: str, meta: dict, dist: float, summary: str) -> None:
    similarity = round((1 - dist) * 100, 1)
    prob, signals = predict_passage({**meta, "text": doc})
    prob_pct = round(prob * 100, 1)
    tier     = prob_tier(prob_pct)
    status   = meta.get("passed_label", "unknown")
    party    = meta.get("party", "?")

    status_cls   = "passed" if status == "passed" else ("failed" if status == "failed" else "unknown")
    status_label = "✓ Passed" if status == "passed" else "✗ Not Passed"
    party_cls    = {"D": "d", "R": "r"}.get(party, "i")

    sig_html = "".join(
        f'<span class="sig {s["t"]}">{esc(s["l"])}</span>' for s in signals
    )
    snippet = esc(summary)
    action  = esc(meta.get("latest_action", "")[:160])
    title   = esc(meta.get("title", "Untitled")[:120])
    bill_id = esc(meta.get("bill_id", "?").upper())
    btype   = esc(meta.get("bill_type", "?").upper())
    chamber = esc(meta.get("chamber", "?"))
    years   = esc(meta.get("congress_years", "?"))

    html = f"""
<div class="bill-card {status_cls}">
  <div class="bill-top">
    <div style="flex:1;">
      <span class="bill-id">{bill_id}</span>
      <div class="bill-title">{title}</div>
    </div>
    <span class="match">Match&nbsp;{similarity}%</span>
  </div>

  <div class="pills">
    <span class="pill {status_cls}">{status_label}</span>
    <span class="pill {party_cls}">{party}</span>
    <span class="pill dim">{chamber}</span>
    <span class="pill dim">{btype}</span>
    <span class="pill dim">{years}</span>
  </div>

  <div class="prob-wrap">
    <div class="prob-head">
      <span class="prob-lbl">Passage Probability</span>
      <span class="prob-val {tier}">{prob_pct}%</span>
    </div>
    <div class="prob-track">
      <div class="prob-bar {tier}" style="width:{prob_pct}%;"></div>
    </div>
  </div>

  <div class="signals">{sig_html}</div>

  <div class="snippet">{snippet}…</div>

  <div class="action-line"><b>Latest action:</b> {action if action else "—"}</div>
</div>"""

    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sb-brand">🏛️ LegisAI</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-desc">Semantic search + ML passage prediction over the '
        '118th U.S. Congress (2023-2024).</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sb-section">Filter Results</div>', unsafe_allow_html=True)
    party_filter  = st.selectbox("Sponsor Party",  ["All","Democrat (D)","Republican (R)","Independent (I)"])
    chamber_filter= st.selectbox("Chamber",         ["All","Senate","House"])
    status_filter = st.selectbox("Bill Status",     ["All","Passed","Not Passed"])
    btype_filter  = st.multiselect("Bill Type",     ["hr","s","hres","sres","hjres","sjres"], default=[])

    st.markdown('<div class="sb-section">Search Options</div>', unsafe_allow_html=True)
    n_results = st.slider("Results to retrieve", 3, 10, 5)

    st.markdown('<div class="sb-section">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sb-desc">Built with ChromaDB · Sentence-Transformers · '
        'XGBoost · Groq LLaMA-3.3</div>',
        unsafe_allow_html=True,
    )

# ── Chroma filter ─────────────────────────────────────────────────────────────
where_conds = []
if party_filter != "All":
    where_conds.append({"party": {"$eq": party_filter.split("(")[1].replace(")", "")}})
if chamber_filter != "All":
    where_conds.append({"chamber": {"$eq": chamber_filter}})
if status_filter == "Passed":
    where_conds.append({"passed_label": {"$eq": "passed"}})
elif status_filter == "Not Passed":
    where_conds.append({"passed_label": {"$ne": "passed"}})
if btype_filter:
    where_conds.append({"bill_type": {"$in": btype_filter}})

chroma_filter = None
if len(where_conds) == 1:   chroma_filter = where_conds[0]
elif len(where_conds) > 1:  chroma_filter = {"$and": where_conds}


# ─────────────────────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-badge">🏛️ &nbsp;118th U.S. Congress · 2023–2024</div>
  <h1 class="hero-title">Congressional Bills Intelligence</h1>
  <p class="hero-sub">Search 2,200+ bills using natural language &middot; Powered by RAG + XGBoost passage prediction</p>
</div>
""", unsafe_allow_html=True)

question = st.text_input(
    "Ask about legislation",
    placeholder="e.g.  healthcare for veterans,  climate legislation,  education funding…",
    key="q",
)

if not question:
    st.markdown("""
<div class="empty">
  <div class="empty-icon">🔍</div>
  <div class="empty-title">Enter a query to search the legislative database</div>
  <div class="empty-hint">Try: "education funding bills" &middot; "defense authorization" &middot; "climate change"</div>
</div>""", unsafe_allow_html=True)
else:
    with st.spinner("Searching legislation…"):
        results = query_bills(question, n_results=n_results, filters=chroma_filter)

    docs  = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    if not docs:
        st.warning("No results found. Try adjusting filters or rephrasing your question.")
    else:
        # ── AI Answer ─────────────────────────────────────────────────────────
        with st.spinner("Generating AI summary…"):
            answer = generate_answer(question, docs, metas)

        st.markdown('<div class="ai-label">💬 &nbsp;AI Answer</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="ai-box">{answer}</div>', unsafe_allow_html=True)

        # ── Results overview ───────────────────────────────────────────────────
        summary_html = results_summary(metas, docs)
        st.markdown(
            f'<div class="summary-box">📊 &nbsp;<strong>Results overview:</strong> {summary_html}</div>',
            unsafe_allow_html=True,
        )

        # ── KPI row ────────────────────────────────────────────────────────────
        passed_count = sum(1 for m in metas if m.get("passed_label") == "passed")
        avg_prob = round(
            np.mean([predict_passage({**m, "text": d})[0] for m, d in zip(metas, docs)]) * 100, 1
        )
        st.markdown(f"""
<div class="kpi-row">
  <div class="kpi">
    <div class="kpi-label">Bills Retrieved</div>
    <div class="kpi-val">{len(docs)}</div>
    <div class="kpi-sub">from {collection.count():,} indexed chunks</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Already Passed</div>
    <div class="kpi-val">{passed_count}<span style="font-size:1.1rem;color:#94a3b8">/{len(docs)}</span></div>
    <div class="kpi-sub">in this result set</div>
  </div>
  <div class="kpi">
    <div class="kpi-label">Avg Pass Probability</div>
    <div class="kpi-val">{avg_prob}<span style="font-size:1.1rem;color:#94a3b8">%</span></div>
    <div class="kpi-sub">XGBoost model estimate</div>
  </div>
</div>""", unsafe_allow_html=True)

        # ── Bill cards ─────────────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-head">📋 Retrieved Bills '
            f'<span style="font-size:0.75rem;font-weight:400;color:#94a3b8">— {len(docs)} results</span></div>',
            unsafe_allow_html=True,
        )
        
        with st.spinner("Generating bill summaries…"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, len(docs))) as executor:
                futures = [
                    executor.submit(summarize_snippet, m.get("title", ""), d)
                    for m, d in zip(metas, docs)
                ]
                card_summaries = [f.result() for f in futures]

        for doc, meta, dist, summary in zip(docs, metas, dists, card_summaries):
            render_bill_card(doc, meta, dist, summary)
