"""
client.py

Provider-agnostic LLM wrapper. Every other module (ConversationManager,
summary_generator) should call get_chat_model() instead of importing
ChatOpenAI/ChatGroq/etc. directly — so switching providers (OpenAI, Groq,
Gemini) is a one-line change in config.py, not a code change anywhere else.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from config import settings


def get_chat_model() -> BaseChatModel:
    """Factory that returns a LangChain-compatible chat model based on
    settings.llm_provider. Providers are imported lazily so the app
    doesn't require every optional SDK installed at once."""

    provider = settings.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.openai_api_key or None,
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            api_key=settings.groq_api_key or None,
        )

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=settings.llm_model_name,
            temperature=settings.llm_temperature,
            max_output_tokens=settings.llm_max_tokens,
            google_api_key=settings.gemini_api_key or None,
        )

    raise ValueError(
        f"Unsupported llm_provider '{settings.llm_provider}'. "
        "Expected 'openai', 'groq', or 'gemini' — check config.py."
    )
