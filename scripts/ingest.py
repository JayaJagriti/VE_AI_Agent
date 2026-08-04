"""
scripts/ingest.py

Builds the RAG knowledge base for the agent:

  1. Fetch content — from settings.ve_source_urls (live scrape) AND/OR any
     .txt/.md files already sitting in data/raw/ (manual fallback, useful
     if a site is behind bot-protection and scraping fails).
  2. Clean + chunk the text into LangChain Documents.
  3. Save the chunks to data/processed/ (for inspection/debugging).
  4. Build a FAISS index from the chunks and persist it to data/vector_store/.

Run from the project root:
    python -m scripts.ingest

Note: virtualemployee.com currently serves a "JavaScript required"
interstitial to non-browser HTTP clients on at least some pages. This
script detects that stub page and skips it with a warning rather than
silently indexing junk. If a page keeps getting skipped, copy its visible
text into a .txt file under data/raw/ instead — local files are always
included regardless of scrape success.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import List

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.vector_store import build_index, save_index
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)

REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
REQUEST_TIMEOUT_SECONDS = 15

# Substring that identifies VE's bot-protection interstitial page, so we
# don't accidentally index it as real content.
JS_CHALLENGE_MARKER = "javascript is required"


def fetch_url_text(url: str) -> str | None:
    """Fetch a URL and return its cleaned visible text, or None if the
    fetch failed or returned the JS-challenge interstitial."""
    try:
        response = requests.get(url, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None

    text = extract_visible_text(response.text)

    if JS_CHALLENGE_MARKER in text.lower():
        logger.warning(
            "Skipping %s — got a JS-challenge interstitial instead of real content. "
            "Consider copying this page's text manually into data/raw/.",
            url,
        )
        return None

    if len(text.strip()) < 200:
        logger.warning("Skipping %s — extracted text looked too short to be useful (%d chars).", url, len(text))
        return None

    return text


def extract_visible_text(html: str) -> str:
    """Strip scripts/styles/nav chrome and return readable page text."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def load_local_raw_files() -> List[Document]:
    """Pick up any .txt/.md files already placed in data/raw/ as a manual
    fallback source, independent of whether scraping succeeds."""
    docs = []
    for path in sorted(settings.raw_data_dir.glob("*")):
        if path.suffix.lower() not in (".txt", ".md"):
            continue
        content = path.read_text(encoding="utf-8", errors="ignore").strip()
        if not content:
            continue
        docs.append(Document(page_content=content, metadata={"source": str(path)}))
        logger.info("Loaded local raw file: %s (%d chars)", path.name, len(content))
    return docs


def scrape_source_urls(urls: List[str]) -> List[Document]:
    docs = []
    for url in urls:
        logger.info("Fetching %s", url)
        text = fetch_url_text(url)
        if text is None:
            continue
        docs.append(Document(page_content=text, metadata={"source": url}))
        # Also cache the raw scrape to data/raw/ so re-runs don't need
        # network access, and so it's inspectable.
        cache_name = url.rstrip("/").split("/")[-1] or "index"
        cache_path = settings.raw_data_dir / f"{cache_name}.txt"
        cache_path.write_text(text, encoding="utf-8")
    return docs


def chunk_documents(documents: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split %d source documents into %d chunks", len(documents), len(chunks))
    return chunks



def save_processed_chunks(chunks: List[Document]) -> Path:
    """Persist chunks as JSON under data/processed/ for inspection."""
    out_path = settings.processed_data_dir / "chunks.json"
    payload = [{"content": c.page_content, "metadata": c.metadata} for c in chunks]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Saved %d processed chunks to %s", len(chunks), out_path)
    return out_path


def _deduplicate_documents(documents: List[Document]) -> List[Document]:
    """Drop documents whose content is (near-)identical to one already
    kept. Needed because scrape_source_urls() auto-caches every scraped
    page to data/raw/, and load_local_raw_files() then picks that same
    file back up — without this, every page ends up indexed twice,
    wasting retrieval slots on exact duplicates instead of diverse
    content."""
    seen_hashes = set()
    deduped = []
    for doc in documents:
        normalized = " ".join(doc.page_content.split()).lower()
        content_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            logger.info("Skipping duplicate content from %s", doc.metadata.get("source"))
            continue
        seen_hashes.add(content_hash)
        deduped.append(doc)
    return deduped

def _deduplicate_chunks(chunks: List[Document]) -> List[Document]:
    """Drop chunks whose content is (near-)identical to a chunk already
    kept. Complements _deduplicate_documents(): that function catches
    whole duplicate pages, but repeated boilerplate paragraphs (e.g. a
    marketing blurb that appears verbatim across several distinct pages)
    survive page-level dedup and end up indexed multiple times, skewing
    retrieval toward whichever generic chunk happens to be duplicated."""
    seen_hashes = set()
    deduped = []
    for chunk in chunks:
        normalized = " ".join(chunk.page_content.split()).lower()
        content_hash = hashlib.md5(normalized.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            logger.info(
                "Skipping duplicate chunk from %s: %.60r...",
                chunk.metadata.get("source"),
                chunk.page_content,
            )
            continue
        seen_hashes.add(content_hash)
        deduped.append(chunk)
    return deduped

def run_ingestion(skip_scrape: bool = False) -> None:
    documents: List[Document] = []

    if not skip_scrape:
        documents.extend(scrape_source_urls(settings.ve_source_urls))
    else:
        logger.info("Skipping live scrape (--skip-scrape).")

    documents.extend(load_local_raw_files())

    documents = _deduplicate_documents(documents)

    if not documents:
        logger.error(
            "No content collected — scraping failed and data/raw/ is empty. "
            "Add .txt/.md files to data/raw/ manually, then re-run."
        )
        sys.exit(1)

    chunks = chunk_documents(documents)
    chunks = _deduplicate_chunks(chunks)
    save_processed_chunks(chunks)

    vector_store = build_index(chunks)
    save_index(vector_store)
    logger.info("Ingestion complete. FAISS index is ready in %s", settings.vector_store_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the VE RAG knowledge base.")
    parser.add_argument(
        "--skip-scrape",
        action="store_true",
        help="Skip live scraping and only ingest files already in data/raw/.",
    )
    args = parser.parse_args()
    run_ingestion(skip_scrape=args.skip_scrape)