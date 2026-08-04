# VirtualEmployee AI Requirement Discovery Agent

🚀 **Live Demo:** https://veaiagent-10.streamlit.app

# VE Requirement Discovery Agent

An AI chat agent for Virtual Employee that:

- Answers questions about Virtual Employee using RAG (retrieval-augmented generation) over the company's own site content
- Gathers a client's hiring requirement through natural conversation, adapting its questions depending on whether the client reads as technical or non-technical
- Captures everything into a structured, validated data model as the conversation goes
- Generates a final requirement summary for a Virtual Employee account manager to act on

**Stack:** Python · Streamlit · LangChain · FAISS · Pydantic · Groq (LLM, free tier) · HuggingFace sentence-transformers (embeddings, local & free)

No paid API keys are required to run this project.

---

## Table of contents

- [How it works](#how-it-works)
- [Project structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup — step by step](#setup--step-by-step)
- [Building the knowledge base](#building-the-knowledge-base)
- [Running the app](#running-the-app)
- [Configuration reference](#configuration-reference)
- [Troubleshooting](#troubleshooting)
- [Extending the project](#extending-the-project)

---

## How it works

Every incoming chat message goes through `ConversationManager`:

1. **Classify intent** (LLM call) into one of:
   - `company_info` — a question about Virtual Employee itself
   - `project_info` — details about the client's own hiring need
   - `both` — a bit of each
   - `general` — greetings / small talk
2. **Route:**
   - `company_info` / `both` → retrieve relevant chunks from the FAISS knowledge base and answer using RAG
   - `project_info` / `both` → extract structured fields (LLM call) and merge them into the session's `RequirementState`
   - `general` → a plain conversational reply
3. **Decide the next question** — a deterministic (no LLM call) check of which `RequirementState` fields are still empty, phrased differently for technical vs. non-technical clients
4. **Persist** the updated state to `data/requirements_store.json`

Once every required field is captured, the sidebar's **Generate Summary** button produces a final `RequirementSummary` (also LLM-generated, with a deterministic fallback if that call fails).

---

## Project structure

```
ve-requirement-agent/
├── app/
│   ├── main.py                       # Streamlit entry point — run this
│   ├── agent/
│   │   ├── conversation_manager.py   # Intent classification, routing, next-question logic
│   │   └── summary_generator.py      # Turns a completed state into the final summary
│   ├── llm/
│   │   └── client.py                 # Provider-agnostic chat model factory (Groq/OpenAI/Gemini)
│   ├── memory/
│   │   └── state_store.py            # JSON persistence of RequirementState, keyed by session
│   ├── models/
│   │   ├── enums.py                  # Shared enums (RoleCategory, EngagementType, etc.)
│   │   └── requirement_schema.py     # RequirementState / RequirementSummary Pydantic models
│   ├── prompts/
│   │   ├── system_prompt.py
│   │   ├── intent_prompt.py
│   │   ├── extraction_prompt.py
│   │   └── summary_prompt.py         # All LLM prompt text, kept out of Python logic
│   ├── rag/
│   │   ├── embeddings.py             # Embedding model factory
│   │   ├── vector_store.py           # FAISS build/save/load
│   │   └── retriever.py              # Retrieval wrapper used by the agent
│   └── utils/
│       ├── logger.py
│       └── text.py
├── scripts/
│   └── ingest.py                     # Scrapes VE pages (+ local files) → builds the FAISS index
├── data/
│   ├── raw/                          # Scraped/manual source text
│   ├── processed/                    # Chunked text (chunks.json), for inspection
│   ├── vector_store/                 # Persisted FAISS index
│   └── requirements_store.json       # Persisted conversation states (created on first run)
├── config.py                         # Single source of truth for all settings
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## Prerequisites

- **Python 3.10–3.12** (check with `python3 --version`)
- A free **Groq** account and API key — sign up at [console.groq.com](https://console.groq.com) (no credit card required)
- ~500MB free disk space (mostly for the local embedding model, downloaded once)

---

## Setup — step by step

### 1. Get the project into a folder

Unzip the project, then open a terminal (or VS Code's integrated terminal) inside the project folder:

```bash
cd ve-requirement-agent
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
```

Activate it:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat
```

Your terminal prompt should now show `(.venv)` at the start of the line.

### 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This installs Streamlit, LangChain, FAISS, Pydantic, the Groq SDK, and `sentence-transformers` (for local embeddings). The first time `sentence-transformers` runs it will download the embedding model (~90MB) — this happens automatically the first time you run ingestion or the app, and is cached afterward.

### 4. Set up your environment variables

```bash
cp .env.example .env
```

Open `.env` and add your Groq key:

```
GROQ_API_KEY=your_key_here
```

(Leave `OPENAI_API_KEY` / `GEMINI_API_KEY` blank unless you switch providers — see [Configuration reference](#configuration-reference).)

### 5. Verify the setup

```bash
python3 -c "
from config import settings
from app.models.requirement_schema import RequirementState
from app.agent.conversation_manager import ConversationManager, IntentType
print('Setup OK —', settings.app_name)
"
```

You should see:

```
Setup OK — VE Requirement Discovery Agent
```

If you get a `ModuleNotFoundError`, re-run `pip install -r requirements.txt` and make sure your venv is activated.

---

## Building the knowledge base

Before the agent can answer questions about Virtual Employee, it needs a FAISS index built from the company's site content.

```bash
python -m scripts.ingest
```

This will:
1. Scrape the pages listed in `config.settings.ve_source_urls`
2. Chunk the text and save it to `data/processed/chunks.json` (for inspection)
3. Build and save a FAISS index to `data/vector_store/`

**Note:** `virtualemployee.com` currently serves a "JavaScript required" interstitial to some non-browser HTTP clients. The script detects this automatically and skips any page that comes back as that stub, rather than indexing junk. If a page keeps getting skipped:

1. Open the page in your browser
2. Copy the visible text
3. Save it as a `.txt` file under `data/raw/` (e.g. `data/raw/faqs.txt`)
4. Re-run ingestion using the local-file-only mode:

```bash
python -m scripts.ingest --skip-scrape
```

Local files in `data/raw/` are **always** picked up regardless of scrape success, so you can mix scraped + manually-added content freely.

---

## Running the app

With your venv active and the knowledge base built:

```bash
streamlit run app/main.py
```

or if python version issue run:

```bash
python3 -m streamlit run app/main.py
```

This opens the chat interface in your browser (usually `http://localhost:8501`). The sidebar shows:
- Fields captured from the conversation so far, live
- Whether the knowledge base is loaded
- A **Generate Summary** button once enough information is captured
- A **Start over** button to reset the session

To stop the app, go back to the terminal and press `Ctrl+C`.

---

## Configuration reference

Everything tunable lives in `config.py`. The most relevant settings:

| Setting | Default | Notes |
|---|---|---|
| `llm_provider` | `"groq"` | `"groq"` \| `"openai"` \| `"gemini"` |
| `llm_model_name` | `"llama-3.1-8b-instant"` | Try `"llama-3.3-70b-versatile"` for higher quality (lower daily free-tier cap) |
| `embedding_provider` | `"huggingface"` | `"huggingface"` (local, free) \| `"openai"` (paid) |
| `embedding_model_name` | `"sentence-transformers/all-MiniLM-L6-v2"` | Downloads once, then runs offline |
| `ve_source_urls` | list of VE pages | Add/remove pages for `scripts/ingest.py` to scrape |
| `chunk_size` / `chunk_overlap` | `800` / `120` | RAG chunking parameters |
| `retriever_top_k` | `4` | How many chunks are retrieved per question |

### Switching to a different LLM provider

**OpenAI:**
```
# .env
OPENAI_API_KEY=sk-...
```
```python
# config.py
llm_provider: str = "openai"
llm_model_name: str = "gpt-4o-mini"
```

**Gemini:**
```
# .env
GEMINI_API_KEY=...
```
```python
# config.py
llm_provider: str = "gemini"
llm_model_name: str = "gemini-1.5-pro"
```

No other code changes are needed — `app/llm/client.py` handles the switch.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'email_validator'`**
Run `pip install -r requirements.txt` again — it's listed there, but if you installed packages manually before, you may have missed it.

**Ingestion reports "Skipping [url] — got a JS-challenge interstitial"**
Expected on some VE pages — see [Building the knowledge base](#building-the-knowledge-base) for the manual fallback.

**"No content collected" error from `scripts.ingest`**
All scrape attempts failed and `data/raw/` is empty. Add at least one `.txt` file to `data/raw/` and re-run.

**Sidebar shows "Knowledge base not built yet"**
Run `python -m scripts.ingest` (see above), then restart the Streamlit app.

**Groq API errors / rate limits**
Groq's free tier has daily token/request caps that vary by model. If you hit them, either wait for the daily reset or switch to a smaller/faster model in `config.py` (e.g. stay on `llama-3.1-8b-instant` rather than a 70B model).

**LLM responses aren't valid JSON, agent seems to skip fields**
`ConversationManager` and `summary_generator.py` are built to fail gracefully (they fall back to "keep asking" or a deterministic summary rather than crashing) — but if this happens often, try a larger/more capable Groq model in `config.py`.

**Streamlit is slow / reloads the embedding model on every message**
The retriever is cached with `@st.cache_resource` in `app/main.py`, so this shouldn't happen after the first load. If it does, restart the app (`Ctrl+C` then `streamlit run app/main.py` again).

---

## Extending the project

- **Add more VE source pages**: edit `config.settings.ve_source_urls`, then re-run `python -m scripts.ingest`
- **Add a new requirement field**: extend `RequirementState` in `app/models/requirement_schema.py`, add it to `_REQUIRED_FIELD_ORDER` and `_QUESTIONS` in `app/agent/conversation_manager.py`, and add it to the JSON schema in `app/prompts/extraction_prompt.py`
- **Swap JSON storage for a real database**: `app/memory/state_store.py` exposes `save_state` / `load_state` / `delete_state` — reimplement these against your DB of choice; nothing else needs to change
- **Tune prompt wording**: everything is in `app/prompts/` as plain string constants, separate from orchestration logic
