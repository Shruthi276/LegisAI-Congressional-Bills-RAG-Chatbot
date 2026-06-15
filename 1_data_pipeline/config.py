"""
config.py — Centralized configuration for the data pipeline.

Load API keys from environment variables (set in .env or your shell).
Never commit real keys to version control.
"""
import os
from pathlib import Path

# ── Root paths ──────────────────────────────────────────────────────────────
# Data files live in the parent Project_bill/ folder (where they were originally
# created by the fetch/preprocess scripts). We point there directly.
# To use a different location, change DATA_DIR here.
DATA_DIR = Path(__file__).parent.parent

# ── Congress API ─────────────────────────────────────────────────────────────
CONGRESS_API_KEY = os.environ.get("CONGRESS_API_KEY", "")
CONGRESS_BASE_URL = "https://api.congress.gov/v3"

# ── Data files ───────────────────────────────────────────────────────────────
RAW_BILLS_FILE      = DATA_DIR / "bills_data.jsonl"
LABELED_BILLS_FILE  = DATA_DIR / "labeled_bills_data.jsonl"
CLEAN_CSV_FILE      = DATA_DIR / "bills_clean.csv"
MODEL_OUTPUT_FILE   = DATA_DIR / "passage_predictor.pkl"

# ── Fetch settings ───────────────────────────────────────────────────────────
CONGRESS_SESSION    = 118           # 118th Congress (2023-2024)
FETCH_BATCH_SIZE    = 250           # Max allowed by Congress API
FETCH_TOTAL         = 2000          # Total bills to fetch in 'all' mode
TARGET_PASSED_COUNT = 500           # Extra passed bills to balance dataset
API_RATE_LIMIT_SLEEP = 0.5          # Seconds between requests
MAX_TEXT_LENGTH     = 100_000       # Cap bill text at 100k chars

# ── Model settings ───────────────────────────────────────────────────────────
RANDOM_STATE        = 42
TEST_SIZE           = 0.2
CV_FOLDS            = 5
XGB_N_ESTIMATORS    = 300
XGB_MAX_DEPTH       = 4
XGB_LEARNING_RATE   = 0.05

FEATURES = [
    "is_senate",
    "is_democrat",
    "is_republican",
    "is_resolution",
    "bill_type_enc",
    "bypassed_committee",
    "num_committees",
    "text_length",
    "word_count",
    "title_word_count",
    "is_naming_bill",
    "has_appropriation",
    "has_health",
    "has_defense",
    "has_education",
]

BILL_TYPE_MAP = {
    "hr": 0, "s": 1, "hres": 2, "sres": 3,
    "hjres": 4, "sjres": 5, "hconres": 6, "sconres": 7,
}
RESOLUTION_TYPES = {"hres", "sres", "hjres", "sjres", "hconres", "sconres"}
