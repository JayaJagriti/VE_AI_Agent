"""
state_store.py

Persists the agent's structured memory of each conversation — a
RequirementState (see app/models/requirement_schema.py) — keyed by
session_id. Renamed from storage/ per review: this is the agent's
structured memory, not generic file storage.

Simple JSON-on-disk implementation (settings.requirement_store_path).
Swap this for a real DB later without changing the function signatures,
since ConversationManager only depends on this interface.
"""

import json
from typing import Dict, Optional

from app.models.requirement_schema import RequirementState
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


def _load_all() -> Dict[str, dict]:
    path = settings.requirement_store_path
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read requirement store at %s: %s", path, exc)
        return {}


def _write_all(data: Dict[str, dict]) -> None:
    path = settings.requirement_store_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def save_state(state: RequirementState) -> None:
    """Persist (create or update) a RequirementState."""
    data = _load_all()
    data[state.session_id] = json.loads(state.model_dump_json())
    _write_all(data)
    logger.info("Saved state for session %s", state.session_id)


def load_state(session_id: str) -> Optional[RequirementState]:
    """Load a previously saved RequirementState, or None if not found."""
    data = _load_all()
    raw = data.get(session_id)
    if raw is None:
        return None
    return RequirementState(**raw)


def delete_state(session_id: str) -> None:
    """Remove a stored RequirementState (e.g. once summary is delivered)."""
    data = _load_all()
    if session_id in data:
        del data[session_id]
        _write_all(data)
        logger.info("Deleted state for session %s", session_id)
