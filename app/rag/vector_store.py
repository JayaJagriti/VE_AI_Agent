"""
vector_store.py

Build, persist, and load the FAISS index that backs the RAG knowledge base
about Virtual Employee. This module only handles index *mechanics*
(create/save/load) — deciding what content goes into it is a separate
ingestion step (a future script that reads data/raw/, chunks it into
data/processed/, and calls build_index() here).
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.rag.embeddings import get_embedding_model
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


def _index_path() -> Path:
    return settings.vector_store_dir / settings.faiss_index_name


def build_index(documents: List[Document]) -> FAISS:
    """Create a new FAISS index in memory from a list of LangChain Documents.
    Does not persist to disk — call save_index() separately."""
    if not documents:
        raise ValueError("Cannot build a FAISS index from an empty document list.")

    embeddings = get_embedding_model()
    logger.info("Building FAISS index from %d documents", len(documents))
    return FAISS.from_documents(documents, embeddings)


def save_index(vector_store: FAISS) -> Path:
    """Persist a FAISS index to disk under data/vector_store/."""
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(path))
    logger.info("Saved FAISS index to %s", path)
    return path


def load_index() -> Optional[FAISS]:
    """Load a previously persisted FAISS index, or None if it doesn't exist yet."""
    path = _index_path()
    if not path.exists():
        logger.warning("No FAISS index found at %s — has it been built yet?", path)
        return None

    embeddings = get_embedding_model()
    return FAISS.load_local(
        str(path), embeddings, allow_dangerous_deserialization=True
    )


def index_exists() -> bool:
    return _index_path().exists()
