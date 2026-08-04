"""
text.py

Small shared text helpers used across the agent — kept here so
conversation_manager.py and summary_generator.py don't duplicate the same
"strip markdown code fences off an LLM's JSON reply" logic.
"""


def strip_code_fences(text: str) -> str:
    """Remove a leading/trailing ``` or ```json fence some LLMs wrap
    JSON responses in, so json.loads() doesn't choke on it."""
    text = text.strip()
    if not text.startswith("```"):
        return text

    text = text.strip("`")
    if text.lower().startswith("json"):
        text = text[4:]
    return text.strip()
