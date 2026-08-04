"""
extraction_prompt.py

Instructions used when asking the LLM to extract structured fields (for
RequirementState) out of a client's free-text message. Kept separate from
system_prompt.py since this is a narrower, single-purpose instruction used
only during the extraction step of ConversationManager.

The explicit JSON schema in the template matters: it's what lets
ConversationManager reliably json.loads() the response and merge it into
RequirementState via Pydantic validation.
"""

EXTRACTION_PROMPT_TEMPLATE = """\
Extract any client profile or project requirement details present in the
message below. Only extract fields the client actually stated or clearly
implied — set anything unmentioned to null. Do not guess or fill in
plausible-sounding values.

Conversation so far (for context):
{conversation_history}

Client's latest message:
{message}

Respond with ONLY a JSON object matching this exact schema (all keys
optional/nullable — omit nothing, use null for anything not mentioned):

{{
  "client_name": string or null,
  "client_email": string or null,
  "client_phone": string or null,
  "company_name": string or null,
  "company_industry": string or null,
  "country": string or null,
  "technical_level": "technical" | "non_technical" | null,
  "role_category": "software_development" | "web_design_development" | "mobile_app_development" | "data_science_ai" | "qa_testing" | "graphic_design" | "digital_marketing_seo" | "content_writing" | "accounting_bookkeeping" | "data_entry" | "admin_virtual_assistant" | "customer_support" | "recruitment_hr" | "legal_process" | "other" | null,
  "role_title": string or null,
  "required_skills": [{{"skill_name": string, "is_mandatory": true, "years_experience": number or null}}] or null,
  "experience_level": "junior" | "mid" | "senior" | "lead" | null,
  "engagement_type": "full_time" | "part_time" | "project_based" | null,
  "number_of_resources": number or null,
  "estimated_budget": {{"min_amount": number or null, "max_amount": number or null, "currency": string, "period": "monthly" | "hourly" | "project_total"}} or null,
  "urgency": "immediate" | "short_term" | "flexible" | null,
  "required_timezone_overlap": string or null,
  "project_description": string or null,
  "additional_notes": string or null
}}

No explanation, no markdown fences — just the JSON object.
"""
