"""
main.py

Streamlit entry point for the Virtual Employee Requirement Discovery
Agent. Run with (from the project root):

    streamlit run app/main.py
"""

import base64
import sys
import textwrap
from pathlib import Path

# Streamlit adds this script's own folder (app/) to sys.path, not the
# project root — so "from app.agent... import ..." can't resolve without
# this. Inserting the parent dir explicitly makes it work regardless of
# the current working directory or how streamlit is invoked.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.agent.conversation_manager import ConversationManager
from app.agent.summary_generator import generate_summary
from app.models.requirement_schema import RequirementState
from app.rag.retriever import VEKnowledgeRetriever
from config import settings

st.set_page_config(
    page_title="VirtualEmployee AI Consultant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Presentational-only constants (no effect on state/logic).
# Place bot_avatar.png / client_avatar.png / ve_logo.png in an
# "assets" folder next to this file (i.e. app/assets/).
ASSETS_DIR = Path(__file__).resolve().parent / "assets"
AVATARS = {
    "assistant": str(ASSETS_DIR / "bot_avatar.png"),
    "user": str(ASSETS_DIR / "client_avatar.png"),
}


@st.cache_data(show_spinner=False)
def _logo_data_uri() -> str:
    """Base64-embed the logo so the header doesn't depend on the
    browser being able to fetch a local file path directly."""
    logo_path = ASSETS_DIR / "ve_logo.png"
    if not logo_path.exists():
        return ""
    encoded = base64.b64encode(logo_path.read_bytes()).decode()
    return f"data:image/png;base64,{encoded}"


LOGO_DATA_URI = _logo_data_uri()

# All colors below are set EXPLICITLY on every custom element (never left
# to inherit from Streamlit's theme). This app forces white/light card
# backgrounds regardless of whether the visitor's Streamlit theme is set
# to light or dark — if text color were left to inherit, a dark-theme
# visitor would get near-invisible light-gray-on-white text. Explicit
# color rules here make the app look identical no matter the visitor's
# theme setting.
st.markdown("""
<style>

:root{
--accent:#14B8A6;
--accent-dark:#0F9488;
--accent-light:#E6FBF7;
--navy:#0B1B3F;
--light:#F5F7FB;
--border:#E4E8F0;
--text:#1F2937;
--muted:#5F6B7A;
}

/* Force the light palette for the app body. */
html{
color-scheme:light !important;
--background-color:#F7F9FC !important;
--secondary-background-color:#FFFFFF !important;
--text-color:#1F2937 !important;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"]{
background:#F7F9FC !important;
}

.stApp{
background:#F7F9FC !important;
}

/* Streamlit's own dev toolbar (top "Deploy" bar) and the sidebar
   collapse control live inside this header. Rather than fight to
   force it light (which also erases its white icon/text and needs
/* Streamlit's own dev toolbar (top "Deploy" bar) lives in this
   header. Keep it on the app's light palette (matching the original
   scheme) rather than a separate navy bar. */
[data-testid*="Header"],
[data-testid*="Toolbar"]{
background:#F7F9FC !important;
background-image:none !important;
}

[data-testid*="Header"] *,
[data-testid*="Toolbar"] *{
color:var(--navy) !important;
fill:var(--navy) !important;
}

[data-testid="stDecoration"]{
background:linear-gradient(90deg,var(--accent),var(--accent-dark)) !important;
}

/* Fixed bottom bar that hosts the chat input: Streamlit fades it in
   with a gradient that ends in the (dark) theme color, which is what
   painted the black corners flanking the input. Force a flat light
   background on the bar and every element inside it, then re-apply
   white specifically to the input pill afterwards (further down,
   with higher specificity) so only the bar itself, not the input, is
   affected. */
[data-testid*="Bottom"]{
background:#F7F9FC !important;
}

[data-testid*="Bottom"] *{
background-color:#F7F9FC !important;
background-image:none !important;
}

/* Match the main content column's width so the input bar lines up
   with the cards above it instead of stretching edge-to-edge. */
[data-testid="stBottomBlockContainer"]{
max-width:1200px !important;
margin:0 auto !important;
padding-left:1rem;
padding-right:1rem;
}

/* The little control that expands a collapsed sidebar — give it its
   own teal accent chip so it stays easy to spot against the light
   header, without needing to darken the whole bar. */
[data-testid*="CollapsedControl"],
[data-testid*="collapsedControl"],
button[title*="sidebar" i],
button[aria-label*="sidebar" i]{
background:var(--accent) !important;
border-radius:8px !important;
opacity:1 !important;
visibility:visible !important;
z-index:999999 !important;
}

[data-testid*="CollapsedControl"] svg,
[data-testid*="collapsedControl"] svg,
button[title*="sidebar" i] svg,
button[aria-label*="sidebar" i] svg{
fill:#FFFFFF !important;
stroke:#FFFFFF !important;
color:#FFFFFF !important;
}

[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea{
background-color:#FFFFFF !important;
}

[data-testid="stChatInput"]{
border:1px solid var(--border) !important;
border-radius:26px !important;
box-shadow:0 2px 10px rgba(0,0,0,.05);
}

[data-testid="stChatInput"] textarea{
color:var(--text) !important;
-webkit-text-fill-color:var(--text) !important;
caret-color:var(--text) !important;
border:none !important;
box-shadow:none !important;
}

[data-testid="stChatInput"] textarea::placeholder{
color:var(--muted) !important;
opacity:1 !important;
}

[data-testid="stChatInput"]:focus-within,
[data-testid="stChatInput"] textarea:focus{
outline:none !important;
border-color:var(--accent) !important;
}

[data-testid="stChatInputSubmitButton"]{
background:var(--accent) !important;
border-radius:50% !important;
}

[data-testid="stChatInputSubmitButton"]:hover{
background:var(--accent-dark) !important;
}

[data-testid="stChatInputSubmitButton"] svg{
fill:#FFFFFF !important;
}

.block-container{
padding-top:2.2rem;
padding-bottom:2rem;
max-width:1200px;
}

/* ---------- Top header bar ---------- */
.ve-header{
display:flex;
align-items:center;
justify-content:space-between;
background:#FFFFFF;
border-radius:16px;
padding:14px 22px;
box-shadow:0 4px 18px rgba(0,0,0,.06);
border:1px solid var(--border);
margin-bottom:20px;
}

.ve-header-left{
display:flex;
align-items:center;
gap:14px;
}

.ve-logo{
width:42px;
height:42px;
border-radius:12px;
background:linear-gradient(135deg,var(--accent),var(--accent-dark));
display:flex;
align-items:center;
justify-content:center;
font-size:20px;
overflow:hidden;
}

.ve-logo img{
width:100%;
height:100%;
object-fit:contain;
padding:6px;
box-sizing:border-box;
}

.ve-header-title{
font-size:17px;
font-weight:700;
color:var(--navy) !important;
line-height:1.2;
}

.ve-header-subtitle{
font-size:13px;
color:var(--muted) !important;
line-height:1.2;
}

.ve-status{
display:flex;
align-items:center;
gap:6px;
font-size:13px;
font-weight:600;
color:var(--accent-dark) !important;
}

.ve-status-dot{
width:8px;
height:8px;
border-radius:50%;
background:var(--accent);
box-shadow:0 0 0 3px var(--accent-light);
}

/* ---------- Hero card ---------- */
.hero{
background:#FFFFFF;
border-radius:18px;
padding:28px 30px;
box-shadow:0 4px 18px rgba(0,0,0,.06);
border:1px solid var(--border);
margin-bottom:20px;
}

.hero-title{
font-size:30px;
font-weight:700;
color:var(--navy) !important;
}

.hero-subtitle{
font-size:16px;
margin-top:6px;
color:var(--muted) !important;
}

/* ---------- Quick topic pills ---------- */
div[data-testid="stHorizontalBlock"] .stButton>button{
background:#FFFFFF;
color:var(--navy) !important;
border:1px solid var(--border);
border-radius:999px;
font-weight:600;
padding:.55rem 1rem;
box-shadow:0 2px 6px rgba(0,0,0,.04);
}

div[data-testid="stHorizontalBlock"] .stButton>button:hover{
border-color:var(--accent);
color:var(--accent-dark) !important;
background:var(--accent-light);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"]{
background:#FFFFFF;
border-right:1px solid var(--border);
}

section[data-testid="stSidebar"] *{
color:var(--text) !important;
}

.progress-item{
display:flex;
align-items:center;
gap:12px;
padding:10px 12px;
border-radius:12px;
border:1px solid var(--border);
margin-bottom:8px;
background:#FFFFFF;
}

.progress-item.done{
background:var(--accent-light);
border-color:#BEEFE6;
}

.progress-icon{
width:34px;
height:34px;
min-width:34px;
border-radius:9px;
background:var(--light);
display:flex;
align-items:center;
justify-content:center;
font-size:16px;
}

.progress-item.done .progress-icon{
background:#FFFFFF;
}

.progress-text{
flex:1;
min-width:0;
}

.progress-label{
font-size:13px;
font-weight:700;
color:var(--navy) !important;
}

.progress-value{
font-size:12px;
color:var(--muted) !important;
white-space:nowrap;
overflow:hidden;
text-overflow:ellipsis;
}

.progress-check{
width:20px;
height:20px;
min-width:20px;
border-radius:50%;
display:flex;
align-items:center;
justify-content:center;
font-size:12px;
font-weight:700;
color:#FFFFFF !important;
}

.progress-item.done .progress-check{
background:var(--accent);
}

.progress-item:not(.done) .progress-check{
border:2px solid var(--border);
}

.progress-caption{
font-size:12px;
color:var(--muted) !important;
margin:6px 0 14px 0;
}

div[data-testid="stSidebar"] [data-testid="stProgress"] > div{
background:var(--border) !important;
border-radius:999px;
}

div[data-testid="stSidebar"] [data-testid="stProgress"] > div > div{
background:var(--accent) !important;
border-radius:999px;
}

/* ---------- Chat area ---------- */
div[data-testid="stChatMessage"]{
padding:14px 16px;
border-radius:16px;
margin-bottom:14px;
background:#FFFFFF;
border:1px solid var(--border);
box-shadow:0 2px 8px rgba(0,0,0,.03);
max-width:80%;
}

div[data-testid="stChatMessage"] p,
div[data-testid="stChatMessage"] li,
div[data-testid="stChatMessage"] span{
color:var(--text) !important;
}

/* User bubbles: right-aligned, teal tint */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-user"]){
background:var(--accent-light);
border-color:#BEEFE6;
margin-left:auto;
flex-direction:row-reverse;
}

/* Assistant bubbles: left-aligned, white */
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]){
background:#FFFFFF;
margin-right:auto;
}

.stButton>button{
background:var(--accent);
color:#FFFFFF !important;
border-radius:10px;
border:none;
font-weight:600;
padding:.6rem;
}

.stButton>button:hover{
background:var(--accent-dark);
}

.stDownloadButton>button{
background:var(--accent);
color:#FFFFFF !important;
border:none;
border-radius:10px;
}

.stDownloadButton>button:hover{
background:var(--accent-dark);
}

div[data-testid="stExpander"]{
border-radius:12px;
border:1px solid var(--border);
background:#FFFFFF;
}

div[data-testid="stExpander"] *{
color:var(--text) !important;
}

div[data-testid="stAlert"]{
border-radius:12px;
}

.ve-disclaimer{
text-align:center;
font-size:12px;
color:var(--muted) !important;
margin-top:8px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# Top header bar
# ----------------------------------------------------------------------
# NOTE: built as a single flat string with no leading indentation and no
# blank lines between tags. Streamlit runs st.markdown() content through
# a Markdown parser before injecting raw HTML — a blank line followed by
# 4+ spaces of indentation is Markdown's trigger for a preformatted code
# block, which silently breaks HTML rendering. Keep this flat if you
# edit it — do not reintroduce indentation or blank lines between tags.
st.markdown(
    '<div class="ve-header">'
    '<div class="ve-header-left">'
    '<div class="ve-logo">'
    + (f'<img src="{LOGO_DATA_URI}" alt="VirtualEmployee logo">' if LOGO_DATA_URI else "💼")
    + '</div>'
    '<div>'
    '<div class="ve-header-title">VirtualEmployee</div>'
    '<div class="ve-header-subtitle">AI Hiring Assistant</div>'
    '</div>'
    '</div>'
    '<div class="ve-status">'
    '<span class="ve-status-dot"></span>Online'
    '</div>'
    '</div>',
    unsafe_allow_html=True,
)

# NOTE: same indentation/blank-line trap applies here as above — keep
# this flat too.
st.markdown(
    '<div class="hero">'
    '<div class="hero-title">VirtualEmployee AI Assistant</div>'
    '<div class="hero-subtitle">'
    "Ask about our services, hiring process, pricing, or simply describe "
    "the team you're looking to build. I'll answer your questions and "
    "help gather your hiring requirements through one conversation."
    "</div>"
    "</div>",
    unsafe_allow_html=True,
)

# Same indentation/blank-line trap applies to plain Markdown (bold text,
# bullets), not just raw HTML — so this is built with textwrap.dedent()
# rather than a raw indented triple-quoted string, to keep the Python
# source readable without leaking leading whitespace into the rendered
# message.
WELCOME_MESSAGE = textwrap.dedent("""\
    👋 **Welcome to the VirtualEmployee AI Assistant!**

    I'm here to help you with anything related to VirtualEmployee.

    I can help you:

    • 💼 Learn about our services and engagement models
    • 💰 Explain pricing and hiring processes
    • 🌍 Answer company-related questions
    • 👥 Understand your hiring requirements
    • 📋 Prepare a structured requirement summary

    How can I assist you today?
""").strip()

QUICK_QUESTIONS = [
    "I need two Python developers",
    "Tell me about your pricing",
    "How does your hiring process work?",
    "We need an AI engineer immediately",
]


@st.cache_resource(show_spinner=False)
def _load_retriever() -> VEKnowledgeRetriever:
    """Cached across reruns — loading the FAISS index / embedding model
    is too slow to redo on every keystroke."""
    return VEKnowledgeRetriever()

CHOICE_FIELDS = {
    "experience_level": [
        ("Junior", "junior"),
        ("Mid", "mid"),
        ("Senior", "senior"),
        ("Lead", "lead"),
    ],
    "engagement_type": [
        ("Full-time", "full_time"),
        ("Part-time", "part_time"),
        ("Project-based", "project_based"),
    ],
    "urgency": [
        ("Immediately", "immediate"),
        ("Within a month", "short_term"),
        ("Flexible", "flexible"),
    ],
}

def _init_session():
    if "requirement_state" not in st.session_state:
        st.session_state.requirement_state = RequirementState()

    if "manager" not in st.session_state:
        st.session_state.manager = ConversationManager(
            st.session_state.requirement_state
        )

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": WELCOME_MESSAGE,
            }
        ]


_init_session()
manager: ConversationManager = st.session_state.manager

# ----------------------------------------------------------------------
# Sidebar: captured requirement, knowledge base status, summary
# ----------------------------------------------------------------------
with st.sidebar:
    st.subheader("Captured requirement")
    state = manager.state
    skip_keys = {"session_id", "created_at", "updated_at"}
    empty_values = (None, [], "", "unknown", "unspecified")
    st.markdown("## 📋 Hiring Progress")
    st.caption("I'll update this automatically as we chat.")

    progress_items = [
        ("👤", "Client", state.client_name),
        ("🏢", "Company", state.company_name),
        ("💼", "Role", state.role_title),
        ("🛠", "Skills", state.required_skills),
        ("👥", "Team Size", state.number_of_resources),
        ("💰", "Budget", state.estimated_budget),
        ("⚡", "Urgency", state.urgency if state.urgency != "unspecified" else None),
        ("🌍", "Timezone", state.required_timezone_overlap),
    ]

    completed = 0
    rows_html = []

    for icon, label, value in progress_items:
        filled = value not in (None, "", [], "unknown")

        if filled:
            completed += 1
            display_value = ", ".join(str(v) for v in value) if isinstance(value, list) else str(value)
        else:
            display_value = "Waiting for your input"

        rows_html.append(
            '<div class="progress-item{done_class}">'
            '<div class="progress-icon">{icon}</div>'
            '<div class="progress-text">'
            '<div class="progress-label">{label}</div>'
            '<div class="progress-value">{value}</div>'
            '</div>'
            '<div class="progress-check">{check}</div>'
            '</div>'.format(
                done_class=" done" if filled else "",
                icon=icon,
                label=label,
                value=display_value,
                check="✓" if filled else "",
            )
        )

    progress = completed / len(progress_items)

    st.progress(progress)
    st.markdown(
        '<div class="progress-caption">Collected <b>{completed}/{total}</b> '
        'requirement details ({pct}%)</div>'.format(
            completed=completed, total=len(progress_items), pct=int(progress * 100)
        ),
        unsafe_allow_html=True,
    )

    st.markdown("".join(rows_html), unsafe_allow_html=True)

    st.divider()

    retriever = _load_retriever()
    if retriever.is_ready():
        st.success("Ready to answer company questions ")
    else:
        st.warning("Knowledge base not built yet.\nRun `python -m scripts.ingest` first.")

    st.divider()

    if state.status in ("pending_review", "complete"):
        if st.button("📋 Generate Summary", use_container_width=True):
            with st.spinner("Generating summary..."):
                st.session_state.summary = generate_summary(state)

    if "summary" in st.session_state:
        summary = st.session_state.summary
        st.subheader("Summary")
        st.write(summary.summary_text)
        for point in summary.key_points:
            st.markdown(f"- {point}")
        st.caption(f"Next step: {summary.recommended_next_step}")
        st.download_button(
            "⬇️ Download summary (JSON)",
            data=summary.model_dump_json(indent=2),
            file_name=f"requirement_summary_{state.session_id[:8]}.json",
            mime="application/json",
            use_container_width=True,
        )

    st.divider()
    if st.button("🔄 Start over", use_container_width=True):
        for key in ("requirement_state", "manager", "messages", "summary"):
            st.session_state.pop(key, None)
        st.rerun()



# ----------------------------------------------------------------------
# Main chat area
# ----------------------------------------------------------------------
def _render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"📚 Sources used for this answer ({len(sources)} chunk(s))"):
        for i, src in enumerate(sources, 1):
            st.caption(f"[{i}] {src['source']}")
            st.text(src["content"])

def process_message(message: str):
    PRETTY = {
        "junior": "Junior",
        "mid": "Mid",
        "senior": "Senior",
        "lead": "Lead",
        "full_time": "Full-time",
        "part_time": "Part-time",
        "project_based": "Project-based",
        "immediate": "Immediately",
        "short_term": "Within a month",
        "flexible": "Flexible",
    }

    pretty = PRETTY.get(message, message)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": pretty,
        }
    )

    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(pretty)

    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        with st.spinner("Thinking..."):
            reply = manager.handle_message(message)

        st.markdown(reply)

        sources = [
            {
                "source": c.metadata.get("source", "unknown"),
                "content": c.page_content,
            }
            for c in manager.last_retrieved_chunks
        ]

        _render_sources(sources)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": reply,
            "sources": sources,
        }
    )


if len(st.session_state.messages) == 1:
    st.markdown(
        '<div class="progress-caption" style="margin-top:-4px;">'
        'Popular questions</div>',
        unsafe_allow_html=True,
    )
    quick_cols = st.columns(len(QUICK_QUESTIONS))
    for col, question in zip(quick_cols, QUICK_QUESTIONS):
        with col:
            if st.button(question, key=f"quick_{question}", use_container_width=True):
                process_message(question)
                st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS.get(msg["role"])):
        st.markdown(msg["content"])
        _render_sources(msg.get("sources", []))

pending = manager.pending_field

if pending in CHOICE_FIELDS:
    cols = st.columns(len(CHOICE_FIELDS[pending]))

    for col, (label, value) in zip(cols, CHOICE_FIELDS[pending]):
        with col:
            if st.button(label, key=f"{pending}_{value}", use_container_width=True):
                process_message(value)
                st.rerun()

typed_input = st.chat_input(
    "Describe your hiring requirement or ask about VirtualEmployee..."
)

if typed_input:
    process_message(typed_input)
    st.rerun()

st.markdown(
    '<div class="ve-disclaimer">AI can make mistakes. Please verify important information.</div>',
    unsafe_allow_html=True,
)