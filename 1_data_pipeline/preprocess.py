"""
preprocess.py — Clean and label raw bill data for ML training.

Reads bills_data.jsonl (output of fetch_bills.py) and produces
labeled_bills_data.jsonl with:
  - Parsed sponsor → chamber, party, state
  - Extracted bill_type and congress session
  - Cleaned bill text (boilerplate removed, capped)
  - Binary passed_label derived from latest_action

Usage:
  python preprocess.py
  python preprocess.py --input data/bills_data.jsonl --output data/labeled_bills_data.jsonl
"""

import json
import re
import logging
import argparse
from pathlib import Path

from config import (
    RAW_BILLS_FILE,
    LABELED_BILLS_FILE,
    MAX_TEXT_LENGTH,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LABEL DERIVATION
# ─────────────────────────────────────────────────────────────────────────────

_PASS_INDICATORS = [
    "became public law",
    "signed by president",
    "passed/agreed to",
    "agreed to",
    "passed senate",
    "passed house",
    "enacted",
]


def derive_label(latest_action: str | None) -> str:
    """
    Derive a binary passage label from the bill's latest action text.

    Args:
        latest_action: Free-text description of the most recent bill action.

    Returns:
        'passed' if the action indicates passage, else 'failed'.
    """
    if not latest_action:
        return "failed"
    action_lower = latest_action.lower()
    return "passed" if any(kw in action_lower for kw in _PASS_INDICATORS) else "failed"


# ─────────────────────────────────────────────────────────────────────────────
# SPONSOR PARSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_sponsor(sponsor_str: str | None) -> tuple[str | None, str | None, str | None]:
    """
    Parse a sponsor string into (chamber, party, state).

    Handles formats like:
      "Sen. Schumer, Charles E. [D-NY]"
      "Rep. Langworthy, Nicholas A. [R-NY-23]"

    Args:
        sponsor_str: Raw sponsor full-name string from the API.

    Returns:
        Tuple of (chamber, party, state). Any field may be None if not parseable.
    """
    if not sponsor_str:
        return None, None, None

    chamber = None
    if sponsor_str.startswith("Sen."):
        chamber = "Senate"
    elif sponsor_str.startswith("Rep."):
        chamber = "House"

    match = re.search(r'\[([A-Z]+)-([A-Z]{2})', sponsor_str)
    party = match.group(1) if match else None
    state = match.group(2) if match else None

    return chamber, party, state


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

_BOILERPLATE_PATTERNS = [
    (re.compile(r'\[Congressional Bills.*?\]', re.IGNORECASE), ''),
    (re.compile(r'<DOC>', re.IGNORECASE), ''),
    (re.compile(r'</DOC>', re.IGNORECASE), ''),
    (re.compile(r'\[From the U\.S\. Government Publishing Office\]', re.IGNORECASE), ''),
    (re.compile(r'\r\n'), '\n'),
    (re.compile(r' {3,}'), '  '),
    (re.compile(r'\n{3,}'), '\n\n'),
]


def clean_text(text: str | None, title: str = "") -> str:
    """
    Strip boilerplate from bill text, normalize whitespace, and cap length.

    Falls back to the bill title if the text is empty after cleaning.

    Args:
        text:  Raw bill text from the API/scraper.
        title: Bill title used as fallback.

    Returns:
        Cleaned text string, capped at MAX_TEXT_LENGTH characters.
    """
    if not text or not text.strip():
        return title

    for pattern, replacement in _BOILERPLATE_PATTERNS:
        text = pattern.sub(replacement, text)

    text = text.strip()

    if not text:
        return title

    return text[:MAX_TEXT_LENGTH]


# ─────────────────────────────────────────────────────────────────────────────
# BILL ID PARSING
# ─────────────────────────────────────────────────────────────────────────────

def extract_bill_type(bill_id: str | None) -> str | None:
    """
    Extract the bill type prefix from a bill_id string.

    Example: "hr9566-118" → "hr"

    Args:
        bill_id: Bill identifier string.

    Returns:
        Lowercase bill type string, or None.
    """
    if not bill_id:
        return None
    match = re.match(r'^([a-zA-Z]+)', bill_id)
    return match.group(1).lower() if match else None


def extract_congress(bill_id: str | None) -> tuple[str | None, str | None]:
    """
    Extract congress number and approximate year range from a bill_id.

    Example: "hr9566-118" → ("118", "2023-2024")

    Args:
        bill_id: Bill identifier string.

    Returns:
        Tuple of (congress_number_str, years_str). Either may be None.
    """
    if not bill_id:
        return None, None
    match = re.search(r'-(\d+)$', bill_id)
    if not match:
        return None, None
    congress = match.group(1)
    try:
        c_num = int(congress)
        start_year = 1789 + (c_num - 1) * 2
        return congress, f"{start_year}-{start_year + 1}"
    except ValueError:
        return congress, None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PROCESSING PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def process_bills(input_file: Path, output_file: Path) -> None:
    """
    Read raw JSONL bills, enrich with derived features, and write labeled JSONL.

    Args:
        input_file:  Path to raw bills_data.jsonl.
        output_file: Path to write labeled_bills_data.jsonl.
    """
    processed = passed = failed = skipped = 0

    with open(input_file, encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:

        for line in infile:
            if not line.strip():
                continue

            try:
                bill = json.loads(line)
            except json.JSONDecodeError:
                log.warning("Skipping invalid JSON line.")
                skipped += 1
                continue

            # ── Sponsor fields ───────────────────────────────────────────────
            chamber, party, state = parse_sponsor(bill.get("sponsor"))
            bill["sponsor_chamber"] = chamber
            bill["sponsor_party"]   = party
            bill["sponsor_state"]   = state

            # ── Bill type + congress ─────────────────────────────────────────
            bill_id = bill.get("bill_id", "")
            bill["bill_type"]      = extract_bill_type(bill_id)
            congress, years        = extract_congress(bill_id)
            bill["congress"]       = congress
            bill["congress_years"] = years

            # ── Committee features ───────────────────────────────────────────
            committees = bill.get("committees", [])
            if not isinstance(committees, list):
                committees = []
            bill["num_committees"]      = len(committees)
            bill["bypassed_committee"]  = 1 if len(committees) == 0 else 0
            bill["committee_names_str"] = " | ".join(committees)

            # ── Naming bill flag ─────────────────────────────────────────────
            title_lower = bill.get("title", "").lower()
            bill["is_naming_bill"] = int(
                "designate the facility" in title_lower or "post office" in title_lower
            )

            # ── Clean text ───────────────────────────────────────────────────
            bill["actual_text"] = clean_text(bill.get("actual_text"), bill.get("title", ""))

            # ── Passage label ────────────────────────────────────────────────
            label = derive_label(bill.get("latest_action"))
            bill["passed_label"] = label

            if label == "passed":
                passed += 1
            else:
                failed += 1

            outfile.write(json.dumps(bill) + "\n")
            processed += 1

    log.info("Processed %d bills  |  Passed: %d  |  Failed: %d  |  Skipped: %d",
             processed, passed, failed, skipped)
    log.info("Output saved to: %s", output_file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean and label raw congressional bill data."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=RAW_BILLS_FILE,
        help="Path to raw bills_data.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=LABELED_BILLS_FILE,
        help="Path to write labeled output JSONL",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Run fetch_bills.py first."
        )

    log.info("Processing %s ...", args.input)
    process_bills(args.input, args.output)


if __name__ == "__main__":
    main()
