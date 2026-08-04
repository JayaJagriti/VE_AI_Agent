"""
scripts/test_retrieval.py

Sanity-checks the RAG retrieval layer WITHOUT calling the LLM — fast and
free, since it only exercises the embedding model + FAISS index. Use this
to verify the right chunks come back for a question before worrying about
whether the LLM's phrasing of the answer is accurate.

Run from the project root:
    python -m scripts.test_retrieval
    python -m scripts.test_retrieval "your own question here"
"""

import sys

from app.rag.retriever import VEKnowledgeRetriever
from config import settings

DEFAULT_TEST_QUESTIONS = [
    "What is Virtual Employee's pricing model?",
    "Do I need to be technical to hire through VE?",
    "How does VE handle employee supervision?",
    "What services does Virtual Employee offer?",
    "Is Virtual Employee GDPR certified?",
]


def run(questions: list[str]) -> None:
    retriever = VEKnowledgeRetriever()

    if not retriever.is_ready():
        print("No FAISS index found. Run `python -m scripts.ingest` first.")
        sys.exit(1)

    for question in questions:
        print(f"\n{'=' * 70}")
        print(f"Q: {question}")
        print("=" * 70)
        chunks = retriever.get_relevant_chunks(question)
        if not chunks:
            print("  (no chunks retrieved)")
            continue
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("source", "unknown")
            preview = chunk.page_content[:200].replace("\n", " ")
            print(f"\n  [{i}] source: {source}")
            print(f"      {preview}...")

    print(f"\n\nManually check: does each retrieved chunk actually contain")
    print("information relevant to its question? If chunks look off-topic")
    print("or thin, try adjusting settings.chunk_size / chunk_overlap in")
    print("config.py and re-running `python -m scripts.ingest`.")


if __name__ == "__main__":
    questions = sys.argv[1:] if len(sys.argv) > 1 else DEFAULT_TEST_QUESTIONS
    run(questions)
