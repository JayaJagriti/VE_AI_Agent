"""
embeddings.py

Builds the embedding model used to vectorize both the Virtual Employee
knowledge base (at indexing time) and incoming queries (at retrieval
time). Reads provider/model choice from config.py so nothing else in the
app hardcodes an embedding model name.
"""

from langchain_core.embeddings import Embeddings

from config import settings


def get_embedding_model() -> Embeddings:
    """Factory that returns a LangChain-compatible embeddings object based
    on settings.embedding_provider. Providers are imported lazily so the
    app doesn't require every optional dependency to be installed at once.
    """

    provider = settings.embedding_provider.lower()

    if provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model_name,
            api_key=settings.openai_api_key or None,
        )

    if provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

    raise ValueError(
        f"Unsupported embedding_provider '{settings.embedding_provider}'. "
        "Expected 'openai' or 'huggingface' — check config.py."
    )
