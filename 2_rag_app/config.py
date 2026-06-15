"""
config.py — Centralized configuration for the RAG app.

Load sensitive keys from environment variables (set in .env or your shell).
"""
import os
from pathlib import Path

# ── Root / Data paths ────────────────────────────────────────────────────────
# Automatically detects if running locally (data in parent folder) 
# or on Hugging Face Spaces (data uploaded alongside app.py).
CURRENT_DIR = Path(__file__).parent

if (CURRENT_DIR / "passage_predictor.pkl").exists():
    # HF Spaces deployment mode
    DATA_DIR = CURRENT_DIR
    CHROMA_PATH = str(CURRENT_DIR / "chroma_db")
else:
    # Local dual-project mode
    DATA_DIR = CURRENT_DIR.parent
    CHROMA_PATH = str(CURRENT_DIR.parent / "chroma_db")
PREDICTOR_PATH   = DATA_DIR / "passage_predictor.pkl"
CHUNKS_CSV_PATH  = DATA_DIR / "bills_chunks.csv"
LABELED_BILLS    = DATA_DIR / "labeled_bills_data.jsonl"
CLEAN_CSV        = DATA_DIR / "bills_clean.csv"

# ── API keys ─────────────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

# ── Model / RAG settings ─────────────────────────────────────────────────────
EMBEDDING_MODEL     = "all-MiniLM-L6-v2"
CHROMA_COLLECTION   = "congressional_bills"
LLM_MODEL           = "llama-3.3-70b-versatile"
LLM_MAX_TOKENS      = 400
LLM_TEMPERATURE     = 0.2
EMBED_BATCH_SIZE    = 64
CHROMA_BATCH_SIZE   = 500
MAX_CHUNK_CHARS     = 8_000

RESOLUTION_TYPES = {"hres", "sres", "hjres", "sjres", "hconres", "sconres"}
BILL_TYPE_MAP = {
    "hr": 0, "s": 1, "hres": 2, "sres": 3,
    "hjres": 4, "sjres": 5, "hconres": 6, "sconres": 7,
}

FEATURES = [
    "is_senate", "is_democrat", "is_republican", "is_resolution",
    "bill_type_enc", "bypassed_committee", "num_committees",
    "text_length", "word_count", "title_word_count",
    "is_naming_bill", "has_appropriation", "has_health",
    "has_defense", "has_education",
]
