"""
rag_pipeline.py — Build and index the ChromaDB vector store for the RAG app.

Pipeline:
  1. Load labeled bill data (from 1_data_pipeline)
  2. Clean text + derive binary passed label
  3. Chunk bills by legislative section
  4. Embed chunks with sentence-transformers
  5. Index into ChromaDB
  6. Run smoke-test queries
  7. Save bills_clean.csv and bills_chunks.csv

Usage:
  python rag_pipeline.py
  python rag_pipeline.py --input data/labeled_bills_data.jsonl
"""

import argparse
import json
import logging
import re
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

from config import (
    LABELED_BILLS,
    CLEAN_CSV,
    CHUNKS_CSV_PATH,
    CHROMA_PATH,
    CHROMA_COLLECTION,
    EMBEDDING_MODEL,
    EMBED_BATCH_SIZE,
    CHROMA_BATCH_SIZE,
    MAX_CHUNK_CHARS,
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Bill IDs that are placeholder "Reserved" bills — not real legislation
_PLACEHOLDER_IDS = {
    "hr9-118", "hr10-118", "hr8-118", "hr6-118",
    "hr4-118", "hr3-118", "hr19-118", "hr18-118", "hr13-118",
}

_META_COLS = [
    "bill_id", "title", "sponsor_party", "sponsor_chamber",
    "sponsor_state", "bill_type", "committee_names_str",
    "bypassed_committee", "num_committees", "passed_label",
    "passed", "latest_action", "congress_years",
]


# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str | None, title: str = "") -> str:
    """
    Clean boilerplate headers and normalize whitespace in bill text.

    Falls back to the bill title if text is empty or very short.

    Args:
        text:  Raw bill text.
        title: Bill title used as fallback.

    Returns:
        Cleaned text string.
    """
    if not text or len(text.strip()) < 100:
        return title

    text = re.sub(r'\[.*?Congress.*?\]\n', '', text)
    text = re.sub(r'\d+th CONGRESS\s+\d+\w+ Session', '', text)
    text = re.sub(r'_{5,}', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# CHUNKING
# ─────────────────────────────────────────────────────────────────────────────

def chunk_bill(row: pd.Series) -> list[dict]:
    """
    Split a bill's text into retrievable chunks by legislative section.

    Splits on 'SEC. N.' headers. Falls back to a single chunk for short bills
    or those without section headers.

    Args:
        row: A DataFrame row with 'clean_text' and 'bill_id' fields.

    Returns:
        List of chunk dicts, each with chunk_id, bill_id, chunk_index,
        text, and is_intro fields.
    """
    text    = row["clean_text"]
    bill_id = row["bill_id"]

    sections = re.split(r'\n(?=SEC\.\s+\d+\.)', text)

    if len(sections) <= 1 or len(text) < 2000:
        return [{
            "chunk_id":    f"{bill_id}_chunk_0",
            "bill_id":     bill_id,
            "chunk_index": 0,
            "text":        text[:MAX_CHUNK_CHARS],
            "is_intro":    True,
        }]

    chunks = []
    for i, section in enumerate(sections[:10]):
        section = section.strip()
        if len(section) < 50:
            continue
        chunks.append({
            "chunk_id":    f"{bill_id}_chunk_{i}",
            "bill_id":     bill_id,
            "chunk_index": i,
            "text":        section[:MAX_CHUNK_CHARS],
            "is_intro":    i == 0,
        })
    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# EMBEDDING
# ─────────────────────────────────────────────────────────────────────────────

def embed_texts(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """
    Embed a list of text strings in batches.

    Args:
        model: Loaded SentenceTransformer model.
        texts: List of text strings to embed.

    Returns:
        List of embedding vectors (each a list of floats).
    """
    embeddings = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i: i + EMBED_BATCH_SIZE]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        embeddings.extend(batch_embeddings.tolist())
        if i % 500 == 0:
            log.info("  Embedded %d / %d chunks...", i, len(texts))
    return embeddings


# ─────────────────────────────────────────────────────────────────────────────
# CHROMADB INDEXING
# ─────────────────────────────────────────────────────────────────────────────

def index_into_chroma(
    chunks_df: pd.DataFrame,
    embeddings: list[list[float]],
    chroma_path: str,
) -> chromadb.Collection:
    """
    Create (or replace) the ChromaDB collection and index all chunks.

    Args:
        chunks_df:   DataFrame of chunks with metadata columns.
        embeddings:  Embedding vectors aligned with chunks_df rows.
        chroma_path: Filesystem path for persistent ChromaDB storage.

    Returns:
        The populated ChromaDB Collection object.
    """
    client = chromadb.PersistentClient(path=chroma_path)

    try:
        client.delete_collection(CHROMA_COLLECTION)
        log.info("Deleted existing collection '%s'.", CHROMA_COLLECTION)
    except Exception:
        pass

    collection = client.create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    for i in range(0, len(chunks_df), CHROMA_BATCH_SIZE):
        batch = chunks_df.iloc[i: i + CHROMA_BATCH_SIZE]
        collection.add(
            ids        = batch["chunk_id"].tolist(),
            embeddings = embeddings[i: i + CHROMA_BATCH_SIZE],
            documents  = batch["text"].tolist(),
            metadatas  = [
                {
                    "bill_id":            str(row["bill_id"]),
                    "title":              str(row["title"]),
                    "party":              str(row["sponsor_party"]),
                    "chamber":            str(row["sponsor_chamber"]),
                    "state":              str(row["sponsor_state"]),
                    "bill_type":          str(row["bill_type"]),
                    "committees":         str(row["committee_names_str"]),
                    "bypassed_committee": int(row["bypassed_committee"]),
                    "passed_label":       str(row["passed_label"]),
                    "latest_action":      str(row["latest_action"] if pd.notna(row["latest_action"]) else ""),
                    "congress_years":     str(row["congress_years"]),
                    "chunk_index":        int(row["chunk_index"]),
                    "is_intro":           bool(row["is_intro"]),
                }
                for _, row in batch.iterrows()
            ],
        )
        log.info("  Indexed %d / %d chunks...",
                 min(i + CHROMA_BATCH_SIZE, len(chunks_df)), len(chunks_df))

    return collection


# ─────────────────────────────────────────────────────────────────────────────
# SMOKE-TEST QUERIES
# ─────────────────────────────────────────────────────────────────────────────

def run_test_queries(collection: chromadb.Collection, embed_model: SentenceTransformer) -> None:
    """Run a few sanity-check queries and print top results."""
    test_queries = [
        ("healthcare and medical coverage for veterans", None),
        ("climate change and clean energy", {"party": {"$eq": "D"}}),
        ("government spending and appropriations", {"passed_label": {"$eq": "passed"}}),
    ]

    log.info("\n=== SMOKE-TEST QUERIES ===")
    for question, filters in test_queries:
        embedding = embed_model.encode([question])[0].tolist()
        kwargs = {
            "query_embeddings": [embedding],
            "n_results":        3,
            "include":          ["documents", "metadatas", "distances"],
        }
        if filters:
            kwargs["where"] = filters

        results = collection.query(**kwargs)
        log.info("\nQuery: '%s'  Filters: %s", question, filters)
        log.info("-" * 60)
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            log.info(
                "  [%s] %s | Party: %s | Status: %s | Similarity: %.3f",
                meta["bill_id"],
                meta["title"][:55],
                meta["party"],
                meta["passed_label"],
                1 - dist,
            )


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def build_index(input_file: Path) -> None:
    """
    End-to-end pipeline: load → clean → chunk → embed → index → save.

    Args:
        input_file: Path to labeled_bills_data.jsonl.
    """
    # 1. Load
    log.info("Loading bills from %s ...", input_file)
    bills = [json.loads(line) for line in open(input_file, encoding="utf-8")]
    df = pd.DataFrame(bills)
    log.info("Loaded %d bills.", len(df))

    # 2. Clean
    log.info("Cleaning data...")
    df = df[~df["bill_id"].isin(_PLACEHOLDER_IDS)].copy()
    df["sponsor_chamber"] = df["sponsor_chamber"].fillna("Delegate")
    df["clean_text"]      = df.apply(
        lambda r: clean_text(r["actual_text"], r["title"]), axis=1
    )
    df["passed"] = (df["passed_label"] == "passed").astype(int)
    log.info("After cleanup: %d bills  |  Pass rate: %.1f%%",
             len(df), df["passed"].mean() * 100)

    # 3. Chunk
    log.info("Chunking bills by section...")
    all_chunks = []
    for _, row in df.iterrows():
        all_chunks.extend(chunk_bill(row))
    chunks_df = pd.DataFrame(all_chunks)
    log.info("Total chunks: %d  |  Avg per bill: %.1f",
             len(chunks_df), len(chunks_df) / len(df))

    # 4. Attach metadata
    chunks_df = chunks_df.merge(df[_META_COLS], on="bill_id", how="left")

    # 5. Embed
    log.info("Loading embedding model: %s ...", EMBEDDING_MODEL)
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    log.info("Embedding %d chunks...", len(chunks_df))
    embeddings = embed_texts(embed_model, chunks_df["text"].tolist())
    log.info("Embedding complete.")

    # 6. Index
    log.info("Indexing into ChromaDB at %s ...", CHROMA_PATH)
    collection = index_into_chroma(chunks_df, embeddings, CHROMA_PATH)
    log.info("ChromaDB collection size: %d chunks.", collection.count())

    # 7. Smoke tests
    run_test_queries(collection, embed_model)

    # 8. Save CSVs
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    CHUNKS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    chunks_df.to_csv(CHUNKS_CSV_PATH, index=False)
    log.info("Saved: %s", CLEAN_CSV)
    log.info("Saved: %s", CHUNKS_CSV_PATH)
    log.info("RAG pipeline complete! Bills: %d | Chunks: %d", len(df), collection.count())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the ChromaDB vector index for the RAG chatbot."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=LABELED_BILLS,
        help="Path to labeled_bills_data.jsonl",
    )
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Input file not found: {args.input}\n"
            "Run 1_data_pipeline/preprocess.py first."
        )

    build_index(args.input)


if __name__ == "__main__":
    main()
