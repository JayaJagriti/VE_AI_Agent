"""
config.py

Central configuration for the Virtual Employee AI Requirement Discovery Agent.

This file ONLY defines settings/constants — paths, model names, chunking
parameters, etc. It contains no business logic, no LangChain chains, and no
Streamlit UI code. Every other module should import `settings` from here
instead of hardcoding values, so the whole app has a single source of truth.
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    # ---------------------------------------------------------------
    # App metadata
    # ---------------------------------------------------------------
    app_name: str = "VE Requirement Discovery Agent"
    company_name: str = "Virtual Employee"
    company_website: str = "https://www.virtualemployee.com/"

    # Seed pages for the RAG knowledge base. Add/remove as needed —
    # scripts/ingest.py reads this list.
    ve_source_urls: list[str] = Field(
        default_factory=lambda: [
            "https://www.virtualemployee.com/about-us",
            "https://www.virtualemployee.com/our-mission",
            "https://www.virtualemployee.com/our-culture",
            "https://www.virtualemployee.com/our-journey",
            "https://www.virtualemployee.com/how-does-ve-work",
            "https://www.virtualemployee.com/cost",
            "https://www.virtualemployee.com/faqs",
            "https://www.virtualemployee.com/services/hire-developers",
            "https://www.virtualemployee.com/services/software-developers",
        ]
    )

    # ---------------------------------------------------------------
    # Filesystem paths (all resolved relative to project root)
    # ---------------------------------------------------------------
    base_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = base_dir / "data"
    raw_data_dir: Path = data_dir / "raw"              # scraped VE site content, brochures, FAQs
    processed_data_dir: Path = data_dir / "processed"  # cleaned/chunked text ready for embedding
    vector_store_dir: Path = data_dir / "vector_store"  # persisted FAISS index

    # ---------------------------------------------------------------
    # LLM settings
    # ---------------------------------------------------------------
    llm_provider: str = "groq"            # "groq" (free tier) | "openai" | "gemini"
    llm_model_name: str = "openai/gpt-oss-20b"  # generous free-tier limits; use "llama-3.3-70b-versatile" for better quality at a lower daily cap
    llm_temperature: float = 0.3
    llm_max_tokens: int = 1024

    # API keys are read from environment / .env — never hardcoded here
    openai_api_key: str = Field(default="", env="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", env="GROQ_API_KEY")
    gemini_api_key: str = Field(default="", env="GEMINI_API_KEY")

    # ---------------------------------------------------------------
    # Embeddings / RAG settings
    # ---------------------------------------------------------------
    embedding_provider: str = "huggingface"  # runs locally — free, no API key, no billing
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # small, fast, well-tested; downloads once (~90MB) then runs offline
    chunk_size: int = 800
    chunk_overlap: int = 120
    retriever_top_k: int = 4

    # ---------------------------------------------------------------
    # FAISS index settings
    # ---------------------------------------------------------------
    faiss_index_name: str = "ve_knowledge_base"

    # ---------------------------------------------------------------
    # Structured requirement capture
    # ---------------------------------------------------------------
    requirement_store_path: Path = data_dir / "requirements_store.json"

    # ---------------------------------------------------------------
    # Streamlit / conversation settings
    # ---------------------------------------------------------------
    max_conversation_turns: int = 30
    session_timeout_minutes: int = 30

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

# Ensure required data directories exist at import time (safe, no business logic)
for _dir in (
    settings.raw_data_dir,
    settings.processed_data_dir,
    settings.vector_store_dir,
):
    _dir.mkdir(parents=True, exist_ok=True)