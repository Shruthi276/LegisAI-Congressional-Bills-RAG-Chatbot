"""
fetch_bills.py — Fetch U.S. Congressional bill data from the Congress.gov API.

Modes:
  all         — Fetch up to FETCH_TOTAL bills (general dataset)
  passed_only — Scan for passed bills to balance the dataset

Usage:
  python fetch_bills.py --mode all
  python fetch_bills.py --mode passed_only

Set your API key in a .env file:
  CONGRESS_API_KEY=your_key_here
"""

import json
import logging
import time
import argparse
import os
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ── Local imports ────────────────────────────────────────────────────────────
from config import (
    CONGRESS_API_KEY,
    CONGRESS_BASE_URL,
    CONGRESS_SESSION,
    FETCH_BATCH_SIZE,
    FETCH_TOTAL,
    TARGET_PASSED_COUNT,
    API_RATE_LIMIT_SLEEP,
    RAW_BILLS_FILE,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# API HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _api_params(**extra) -> dict:
    """Return base Congress API query parameters merged with extras."""
    return {"api_key": CONGRESS_API_KEY, "format": "json", **extra}


def get_bills(congress: int = CONGRESS_SESSION, limit: int = 250, offset: int = 0) -> list:
    """
    Fetch a paginated list of bills from Congress.gov.

    Args:
        congress: Congress session number (e.g. 118).
        limit: Number of bills per page (max 250).
        offset: Pagination offset.

    Returns:
        List of bill summary dicts from the API.
    """
    url = f"{CONGRESS_BASE_URL}/bill/{congress}"
    response = requests.get(url, params=_api_params(limit=limit, offset=offset))
    response.raise_for_status()
    return response.json().get("bills", [])


def get_bill_details(congress: int, bill_type: str, bill_number: str) -> dict:
    """
    Fetch full metadata for a single bill.

    Args:
        congress: Congress session number.
        bill_type: Bill type code (e.g. 'hr', 's', 'hres').
        bill_number: Bill number string.

    Returns:
        Bill detail dict from the API.
    """
    url = f"{CONGRESS_BASE_URL}/bill/{congress}/{bill_type}/{bill_number}"
    response = requests.get(url, params=_api_params())
    response.raise_for_status()
    return response.json().get("bill", {})


def get_bill_text_url(congress: int, bill_type: str, bill_number: str) -> str | None:
    """
    Fetch the URL for the most recent text version of a bill.

    Prefers 'Formatted Text' (HTML), falls back to 'Formatted XML'.

    Returns:
        URL string if text is available, else None.
    """
    url = f"{CONGRESS_BASE_URL}/bill/{congress}/{bill_type}/{bill_number}/text"
    response = requests.get(url, params=_api_params())
    if response.status_code != 200:
        return None

    versions = response.json().get("textVersions", [])
    if not versions:
        return None

    for fmt in versions[0].get("formats", []):
        if fmt.get("type") == "Formatted Text":
            return fmt.get("url")
    for fmt in versions[0].get("formats", []):
        if fmt.get("type") == "Formatted XML":
            return fmt.get("url")
    return None


def fetch_bill_text(text_url: str) -> str | None:
    """
    Download and parse bill text from a congress.gov HTML/XML URL.

    Args:
        text_url: URL returned by get_bill_text_url.

    Returns:
        Plain text string, or None on failure.
    """
    try:
        response = requests.get(text_url, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        return soup.get_text(separator="\n", strip=True)
    except Exception as exc:
        log.warning("Could not fetch bill text from %s: %s", text_url, exc)
        return None


def fetch_committees(committees_data: dict) -> list[str]:
    """
    Fetch committee names for a bill from the nested committees URL.

    Args:
        committees_data: The 'committees' sub-dict from bill details.

    Returns:
        List of committee name strings.
    """
    if not committees_data.get("count") or not committees_data.get("url"):
        return []
    try:
        resp = requests.get(committees_data["url"], params=_api_params())
        if resp.status_code == 200:
            return [c.get("name") for c in resp.json().get("committees", []) if c.get("name")]
    except Exception as exc:
        log.warning("Could not fetch committees: %s", exc)
    return []


def build_bill_record(
    bill_id: str,
    title: str | None,
    sponsor_name: str | None,
    committees: list,
    latest_action: str | None,
    text_url: str | None,
    actual_text: str | None,
) -> dict:
    """Assemble a clean bill record dict ready for JSONL serialization."""
    return {
        "bill_id":      bill_id,
        "title":        title,
        "sponsor":      sponsor_name,
        "committees":   committees,
        "latest_action": latest_action,
        "text_url":     text_url,
        "actual_text":  actual_text,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PASSAGE CHECK HELPER
# ─────────────────────────────────────────────────────────────────────────────

def is_passed(latest_action: str) -> bool:
    """Return True if the latest_action string indicates the bill passed."""
    if not latest_action:
        return False
    action_lower = latest_action.lower()
    return any(kw in action_lower for kw in [
        "became public law", "signed by president",
        "passed/agreed to", "agreed to",
        "passed senate", "passed house", "enacted",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# MODE 1 — FETCH ALL BILLS
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_bills(output_file: Path) -> None:
    """
    Fetch up to FETCH_TOTAL bills from the 118th Congress and save to JSONL.

    Resumes automatically if the output file already exists.

    Args:
        output_file: Path to the output .jsonl file.
    """
    fetched_count = 0
    offset = 0

    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            fetched_count = sum(1 for _ in f)
        log.info("Resuming — %d bills already fetched.", fetched_count)
        offset = fetched_count

    log.info("Fetching up to %d bills (118th Congress)...", FETCH_TOTAL)

    while fetched_count < FETCH_TOTAL:
        limit = min(FETCH_BATCH_SIZE, FETCH_TOTAL - fetched_count)
        log.info("Page: offset=%d, limit=%d", offset, limit)

        try:
            bills = get_bills(limit=limit, offset=offset)
        except Exception as exc:
            log.error("Failed to fetch bill list: %s. Retrying in 5s...", exc)
            time.sleep(5)
            continue

        if not bills:
            log.info("No more bills returned — stopping.")
            break

        for bill in bills:
            if fetched_count >= FETCH_TOTAL:
                break

            congress    = bill.get("congress")
            bill_type   = bill.get("type", "").lower()
            bill_number = bill.get("number")
            bill_id     = f"{bill_type}{bill_number}-{congress}"

            log.info("[%d/%d] %s", fetched_count + 1, FETCH_TOTAL, bill_id.upper())

            try:
                details     = get_bill_details(congress, bill_type, bill_number)
                sponsors    = details.get("sponsors", [])
                committees  = fetch_committees(details.get("committees", {}))
                text_url    = get_bill_text_url(congress, bill_type, bill_number)
                actual_text = fetch_bill_text(text_url) if text_url else None

                record = build_bill_record(
                    bill_id      = bill_id,
                    title        = details.get("title"),
                    sponsor_name = sponsors[0].get("fullName") if sponsors else None,
                    committees   = committees,
                    latest_action= details.get("latestAction", {}).get("text"),
                    text_url     = text_url,
                    actual_text  = actual_text,
                )

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                fetched_count += 1
                time.sleep(API_RATE_LIMIT_SLEEP)

            except Exception as exc:
                log.error("Error processing %s: %s", bill_id, exc)
                time.sleep(2)

        offset += len(bills)

    log.info("Done. Saved %d bills to %s", fetched_count, output_file)


# ─────────────────────────────────────────────────────────────────────────────
# MODE 2 — FETCH PASSED BILLS ONLY (dataset balancing)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_passed_only(output_file: Path) -> None:
    """
    Scan the Congress API for passed bills to supplement an existing dataset.

    Reads the existing output file to avoid duplicates, then pages through
    bills starting at offset 2000 (beyond what fetch_all_bills already got),
    saving only bills where the latest action indicates passage.

    Args:
        output_file: Path to the output .jsonl file (appended to).
    """
    existing_passed_ids: set[str] = set()

    if output_file.exists():
        with open(output_file, encoding="utf-8") as f:
            for line in f:
                try:
                    bill = json.loads(line)
                    if is_passed(bill.get("latest_action", "")):
                        existing_passed_ids.add(bill["bill_id"])
                except Exception:
                    pass

    log.info("Found %d passed bills already in dataset.", len(existing_passed_ids))
    passed_needed = TARGET_PASSED_COUNT - len(existing_passed_ids)

    if passed_needed <= 0:
        log.info("Dataset already has enough passed bills. Nothing to do.")
        return

    log.info("Need %d more passed bills.", passed_needed)
    offset = 2000  # skip bills already fetched by fetch_all_bills

    while passed_needed > 0:
        try:
            bills = get_bills(limit=250, offset=offset)
        except Exception as exc:
            log.error("Error fetching bill list: %s", exc)
            break

        if not bills:
            break

        for bill in bills:
            latest_action = bill.get("latestAction", {}).get("text", "")
            bill_type     = bill.get("type", "").lower()
            bill_number   = bill.get("number")
            congress      = bill.get("congress")
            bill_id       = f"{bill_type}{bill_number}-{congress}"

            if not is_passed(latest_action) or bill_id in existing_passed_ids:
                continue

            log.info("Found passed bill: %s — fetching details...", bill_id)

            try:
                details     = get_bill_details(congress, bill_type, bill_number)
                sponsors    = details.get("sponsors", [])
                committees  = fetch_committees(details.get("committees", {}))
                text_url    = get_bill_text_url(congress, bill_type, bill_number)
                actual_text = fetch_bill_text(text_url) if text_url else None

                record = build_bill_record(
                    bill_id      = bill_id,
                    title        = details.get("title"),
                    sponsor_name = sponsors[0].get("fullName") if sponsors else None,
                    committees   = committees,
                    latest_action= details.get("latestAction", {}).get("text"),
                    text_url     = text_url,
                    actual_text  = actual_text,
                )

                with open(output_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                existing_passed_ids.add(bill_id)
                passed_needed -= 1
                log.info("Saved. Still need %d more.", passed_needed)

                if passed_needed <= 0:
                    break

                time.sleep(API_RATE_LIMIT_SLEEP)

            except Exception as exc:
                log.error("Error processing %s: %s", bill_id, exc)

        offset += len(bills)
        log.info("Next page: offset=%d", offset)

    log.info("Done fetching passed bills.")


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    if not CONGRESS_API_KEY:
        raise EnvironmentError(
            "CONGRESS_API_KEY is not set.\n"
            "Copy .env.example to .env and add your key."
        )

    parser = argparse.ArgumentParser(
        description="Fetch congressional bills from the Congress.gov API."
    )
    parser.add_argument(
        "--mode",
        choices=["all", "passed_only"],
        default="all",
        help="'all' fetches FETCH_TOTAL bills; 'passed_only' supplements with passed bills.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RAW_BILLS_FILE,
        help="Path to output .jsonl file.",
    )
    args = parser.parse_args()

    if args.mode == "all":
        fetch_all_bills(args.output)
    else:
        fetch_passed_only(args.output)


if __name__ == "__main__":
    main()
