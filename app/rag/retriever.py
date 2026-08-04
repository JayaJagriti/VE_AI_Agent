"""
retriever.py

Thin wrapper around the FAISS vector store that the agent calls when it
needs to answer "about Virtual Employee" questions. Isolated from
vector_store.py so the rest of the app doesn't need to know about index
build/load mechanics — it just asks for relevant chunks.
"""

from typing import List

from langchain_core.documents import Document

from app.rag.vector_store import load_index
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class VEKnowledgeRetriever:
    """Retrieves relevant chunks from the Virtual Employee knowledge base
    for use in RAG-style question answering."""

    def __init__(self, top_k: int = None):
        self.top_k = top_k or settings.retriever_top_k
        self._vector_store = load_index()

        if self._vector_store is None:
            logger.warning(
                "VEKnowledgeRetriever initialized without a built index. "
                "Run the ingestion step (build_index + save_index) first — "
                "get_relevant_chunks() will raise until then."
            )

    def is_ready(self) -> bool:
        """Whether a FAISS index has been loaded successfully."""
        return self._vector_store is not None

    def get_relevant_chunks(self, query: str) -> List[Document]:
        """Return the top-k most relevant document chunks for a query."""
        if self._vector_store is None:
            raise RuntimeError(
                "No FAISS index loaded. Build and save one via "
                "app.rag.vector_store.build_index()/save_index() first."
            )
        return self._vector_store.similarity_search(query, k=self.top_k)

    def as_langchain_retriever(self):
        """Expose as a standard LangChain retriever object, for use inside
        a RetrievalQA / LCEL chain later."""
        if self._vector_store is None:
            raise RuntimeError("No FAISS index loaded.")
        return self._vector_store.as_retriever(search_kwargs={"k": self.top_k})
