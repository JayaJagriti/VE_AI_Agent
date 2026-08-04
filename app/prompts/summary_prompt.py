"""
summary_prompt.py

Instructions used to turn a completed RequirementState into the final
human-readable RequirementSummary handed to the client / sales team.
"""

SUMMARY_PROMPT_TEMPLATE = """\
Using the structured requirement data below, write a clear, professional
summary suitable for a Virtual Employee account manager to act on
immediately.

Requirement data:
{requirement_state_json}

Respond with ONLY a JSON object matching this schema, no markdown fences:

{{
  "summary_text": "3-5 sentence narrative summary in plain English",
  "key_points": ["short bullet point", "short bullet point", "..."],
  "recommended_next_step": "one sentence, e.g. 'Schedule a call with a VE account manager.'"
}}
"""
