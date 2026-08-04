"""
summary_generator.py

Turns a completed RequirementState into a RequirementSummary using the
LLM. Falls back to a deterministic, no-LLM summary if the call or JSON
parsing fails, so the user always gets something usable.
"""

import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from app.llm.client import get_chat_model
from app.models.requirement_schema import RequirementState, RequirementSummary
from app.prompts.summary_prompt import SUMMARY_PROMPT_TEMPLATE
from app.prompts.system_prompt import SYSTEM_PROMPT
from app.utils.logger import get_logger
from app.utils.text import strip_code_fences

logger = get_logger(__name__)


def generate_summary(state: RequirementState) -> RequirementSummary:
    prompt = SUMMARY_PROMPT_TEMPLATE.format(
        requirement_state_json=state.model_dump_json(indent=2)
    )
    llm = get_chat_model()
    try:
        response = llm.invoke([SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)])
        text = response.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("LLM call failed while generating summary: %s", exc)
        return _fallback_summary(state)

    parsed = _parse_json(text)
    if not parsed:
        return _fallback_summary(state)

    return RequirementSummary(
        session_id=state.session_id,
        summary_text=parsed.get("summary_text", "") or _fallback_summary(state).summary_text,
        key_points=parsed.get("key_points", []),
        recommended_next_step=parsed.get(
            "recommended_next_step", "Schedule a call with a Virtual Employee account manager."
        ),
    )


def _parse_json(text: str) -> Optional[dict]:
    text = strip_code_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Could not parse summary JSON: %s", text[:200])
        return None


def _fallback_summary(state: RequirementState) -> RequirementSummary:
    """Deterministic summary built without the LLM — used if the LLM call
    or JSON parsing fails."""
    lines = [
        f"Client: {state.client_name or 'Not provided'} ({state.company_name or 'Company not provided'})",
        f"Role: {state.role_title or state.role_category}",
        f"Engagement: {state.engagement_type}, {state.number_of_resources or '?'} resource(s)",
    ]
    if state.estimated_budget:
        lines.append(
            f"Budget: {state.estimated_budget.min_amount}-{state.estimated_budget.max_amount} "
            f"{state.estimated_budget.currency}/{state.estimated_budget.period}"
        )
    if state.project_description:
        lines.append(f"Project: {state.project_description}")

    return RequirementSummary(
        session_id=state.session_id,
        summary_text="\n".join(lines),
        key_points=lines,
        recommended_next_step="Schedule a call with a Virtual Employee account manager.",
    )
