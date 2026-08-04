"""
intent_prompt.py

Instructions used to classify each incoming client message into one of
ConversationManager's IntentType values, before deciding how to route it.
"""

INTENT_PROMPT_TEMPLATE = """\
Classify the client's latest message into exactly one of these four categories.

You are NOT classifying the latest message in isolation.
You are deciding how the assistant should CONTINUE the conversation based on:
- the assistant's current pending question,
- the assistant's previous question,
- the conversation history,
- and the client's latest message.

Intent categories:

- "company_info": asking about Virtual Employee itself — services, pricing,
  hiring process, security, company background, policies, etc.

- "project_info": providing, answering, updating, clarifying or discussing
  the client's own hiring requirement (role, skills, experience, engagement,
  budget, timeline, team size, project details, contact information, etc.).

- "both": the client is BOTH updating their hiring requirement AND asking a
  factual question about Virtual Employee.

- "general": greetings, thanks, small talk, or messages unrelated to either
  the pending question and Virtual Employee.

IMPORTANT ROUTING RULES (highest priority first):

1. ALWAYS consider the assistant's current pending field and previous question
before making your decision.

2. If the assistant is currently waiting for a hiring requirement, and the
client's latest message could plausibly be interpreted as answering,
continuing, or clarifying that question, classify it as "project_info".

This rule takes precedence over every other rule — EXCEPT when the message is
clearly unrelated to BOTH the pending question AND Virtual Employee entirely
(e.g. random trivia, off-topic questions, jokes, testing messages like "asdf"
or "test"). In that case, classify it as "general", even though a field is
pending. Do not force an unrelated message into "project_info" just because a
question happens to be pending — the assistant should acknowledge the
off-topic message rather than silently ignore it.

3. Short replies such as:
"yes", "no", "okay", "sure", "mid", "senior", "junior", "lead",
"1", "2", "part-time", "full-time", "project-based",
names, email addresses, phone numbers, countries, currencies,
or similar concise replies should usually be treated as
"project_info" IF they answer the assistant's previous hiring question.

4. Do NOT classify a message as "company_info" simply because it contains
words that are also related to Virtual Employee services (such as pricing,
part-time, developer, budget, etc.) if the client is primarily answering
the assistant's previous hiring question.

5. Use "both" ONLY when the client is simultaneously:
   - continuing their hiring requirement, AND
   - asking a factual question about Virtual Employee.

6. Use "company_info" only when the client's primary goal has shifted away
from the current requirement-gathering conversation.

Examples:

Assistant: Do you have a monthly budget range?
Client: No, what would you recommend?
→ {{"intent": "project_info"}}

Assistant: What experience level are you after?
Client: Mid
→ {{"intent": "project_info"}}

Assistant: Are you looking for a full-time dedicated resource, part-time help, or something project-based?
Client: Part-time
→ {{"intent": "project_info"}}

Assistant: How many developers do you need?
Client: Around 3. Also, how does your hiring process work?
→ {{"intent": "both"}}

Client: What services do you offer?
→ {{"intent": "company_info"}}

Client: I need a senior React developer.
→ {{"intent": "project_info"}}

Assistant: What role or type of resource are you looking to hire?
Client: what is the color of sky
→ {{"intent": "general"}}

Assistant: What role or type of resource are you looking to hire?
Client: test
→ {{"intent": "general"}}

Assistant is currently waiting for:
{pending_field}

Assistant's previous question:
{previous_question}

Conversation so far:
{conversation_history}

Client's latest message:
{message}

Respond with ONLY a JSON object.

{{"intent": "company_info" | "project_info" | "both" | "general"}}
"""