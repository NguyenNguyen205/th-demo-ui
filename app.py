"""TH Group Nutrition Advisor — Streamlit chat interface."""

import json
import os
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# ── Config ─────────────────────────────────────────────────────────────────────

REGION = os.getenv("AWS_REGION", "ap-northeast-1")
AGENT_ARN = os.getenv(
    "AGENT_ARN",
    "arn:aws:bedrock-agentcore:ap-northeast-1:627617970708:runtime/langgraph_agent_web_search-uVGXiw97pa",
)
COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "66fkvph0jpae1tn0p4fh3ukf8p")
COGNITO_USERNAME = os.getenv("COGNITO_USERNAME", "USER")
COGNITO_PASSWORD = os.getenv("COGNITO_PASSWORD", "DemoAgent123!@")
ACTOR_ID = "nutrition-advisor"  # fixed across all sessions

MODEL_OPTIONS = {
    "Claude Haiku 4.5":  "haiku",
    "Claude Sonnet 4.5": "sonnet",
    "Amazon Nova Pro":   "nova",
    "MiniMax M2":        "minimax",
}

SESSIONS_FILE = Path(__file__).parent / ".sessions.json"
S3_REGION = os.getenv("AWS_REGION", "ap-northeast-1")

# ── Token management ───────────────────────────────────────────────────────────

def _fetch_token() -> dict:
    """Fetch a fresh Cognito access token. Returns {token, expires_at}."""
    import boto3
    client = boto3.client("cognito-idp", region_name=REGION)
    resp = client.initiate_auth(
        ClientId=COGNITO_CLIENT_ID,
        AuthFlow="USER_PASSWORD_AUTH",
        AuthParameters={
            "USERNAME": COGNITO_USERNAME,
            "PASSWORD": COGNITO_PASSWORD,
        },
    )
    token = resp["AuthenticationResult"]["AccessToken"]
    expires_in = resp["AuthenticationResult"].get("ExpiresIn", 3600)
    expires_at = time.time() + expires_in - 60  # refresh 60s early
    return {"token": token, "expires_at": expires_at}


def get_token() -> str:
    """Return a valid token, refreshing silently if needed."""
    creds = st.session_state.get("_cognito")
    if not creds or time.time() >= creds["expires_at"]:
        with st.spinner("Authenticating…"):
            st.session_state["_cognito"] = _fetch_token()
    return st.session_state["_cognito"]["token"]


# ── Session persistence (local JSON file) ─────────────────────────────────────

def _load_sessions() -> dict:
    if SESSIONS_FILE.exists():
        try:
            return json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _save_sessions(sessions: dict) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions, ensure_ascii=False, indent=2), encoding="utf-8")


def _new_session_id() -> str:
    return str(uuid.uuid4())


def _session_label(session_id: str, sessions: dict) -> str:
    meta = sessions.get(session_id, {})
    label = meta.get("label") or session_id[:8]
    return label


# ── Citation helpers ───────────────────────────────────────────────────────────

def _s3_to_https(uri: str) -> str:
    """Convert s3://bucket/key to virtual-hosted HTTPS URL (public bucket)."""
    if not uri.startswith("s3://"):
        return uri
    without_scheme = uri[5:]
    bucket, _, key = without_scheme.partition("/")
    return f"https://{bucket}.s3.{S3_REGION}.amazonaws.com/{key}"


def _render_disclaimer(text: str) -> None:
    """Render the nutrition disclaimer as a styled info card."""
    st.info(f"⚠️ {text}", icon=None)


def _render_citations(citations: list[dict]) -> None:
    """Render a list of citation dicts as formatted academic-style references."""
    for i, c in enumerate(citations, 1):
        title = c.get("title") or "Unknown source"
        # Strip file extension for display
        display_title = title.rsplit(".", 1)[0] if "." in title else title
        score = c.get("score", 0.0)
        excerpt = c.get("excerpt", "")
        url = _s3_to_https(c.get("uri", ""))

        st.markdown(
            f"**[{i}] [{display_title}]({url})**  \n"
            f"Relevance: {score:.0%}"
        )
        if excerpt:
            st.markdown(f"> *\"{excerpt}\"*")
        st.divider()


# ── Agent invocation ───────────────────────────────────────────────────────────

def invoke_streaming(prompt: str, session_id: str, model_id: str = "sonnet") -> tuple[str, str, list[dict], str]:
    """Send prompt to AgentCore Runtime, stream tokens, return (response, thinking, citations, disclaimer)."""
    token = get_token()
    escaped = urllib.parse.quote(AGENT_ARN, safe="")
    url = (
        f"https://bedrock-agentcore.{REGION}.amazonaws.com"
        f"/runtimes/{escaped}/invocations?qualifier=DEFAULT"
    )
    # url = (
    #     "http://localhost:8080/invocations?qualifier=DEFAULT"
    # )
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "X-Amzn-Bedrock-AgentCore-Runtime-Session-Id": session_id,
        "X-Amzn-Bedrock-AgentCore-Runtime-Custom-ActorId": ACTOR_ID,
    }

    resp = requests.post(
        url,
        headers=headers,
        data=json.dumps({"prompt": prompt, "model_id": model_id}),
        stream=True,
        timeout=300,
    )

    if resp.status_code != 200:
        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:400]}")

    visible_parts: list[str] = []
    citation_list: list[dict] = []
    disclaimer_text: str = ""
    response_placeholder = st.empty()

    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8").strip()
        if not decoded.startswith("data: "):
            continue
        data_str = decoded[6:].strip()
        if not data_str:
            continue
        try:
            event = json.loads(data_str)
        except json.JSONDecodeError:
            continue

        # Handle structured metadata epilog
        if isinstance(event, dict) and event.get("type") == "metadata":
            disclaimer_text = event.get("disclaimer", "")
            citation_list.extend(event.get("citations", []))
            continue

        text = _extract_text(event)
        if not text:
            continue
        visible_parts.append(text)

        current = "".join(visible_parts).strip()
        if current:
            response_placeholder.markdown(current + "▌")

    response_placeholder.empty()
    return "".join(visible_parts).strip(), "", citation_list, disclaimer_text


def _extract_text(event) -> str:
    if isinstance(event, str):
        return event
    if not isinstance(event, dict):
        return ""
    # Structured metadata event — not a text token
    if event.get("type") == "metadata":
        return ""
    import base64
    chunk = event.get("chunk")
    if isinstance(chunk, dict):
        raw = chunk.get("bytes")
        if raw:
            try:
                return base64.b64decode(raw).decode("utf-8")
            except Exception:
                return str(raw)
    output = event.get("output")
    if isinstance(output, str):
        return output
    content = event.get("content")
    if isinstance(content, str):
        return content
    for v in event.values():
        if isinstance(v, str) and v.strip():
            return v
    return ""


# ── UI ─────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="TH Nutrition Advisor",
    page_icon="🥛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Sidebar session list */
    .session-active { font-weight: 700; color: #1a73e8; }
    /* Remove default top padding */
    .block-container { padding-top: 1rem; }
    /* Chat bubbles */
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ── State bootstrap ────────────────────────────────────────────────────────────

if "sessions" not in st.session_state:
    st.session_state.sessions = _load_sessions()          # {session_id: {label, messages, created_at}}

if "active_session" not in st.session_state:
    sessions = st.session_state.sessions
    if sessions:
        st.session_state.active_session = next(iter(sessions))
    else:
        sid = _new_session_id()
        st.session_state.sessions[sid] = {
            "label": "New chat",
            "messages": [],
            "model": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        st.session_state.active_session = sid
        _save_sessions(st.session_state.sessions)

# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🥛 TH Nutrition Advisor")
    st.divider()

    if st.button("＋  New chat", use_container_width=True, type="primary"):
        sid = _new_session_id()
        st.session_state.sessions[sid] = {
            "label": "New chat",
            "messages": [],
            "model": None,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        st.session_state.active_session = sid
        _save_sessions(st.session_state.sessions)
        st.rerun()

    st.divider()

    # Session list — newest first
    sessions = st.session_state.sessions
    sorted_ids = sorted(
        sessions.keys(),
        key=lambda s: sessions[s].get("created_at", ""),
        reverse=True,
    )

    for sid in sorted_ids:
        meta = sessions[sid]
        label = meta.get("label") or "New chat"
        is_active = sid == st.session_state.active_session
        col1, col2 = st.columns([6, 1])
        with col1:
            btn_type = "primary" if is_active else "secondary"
            if st.button(label, key=f"sel_{sid}", use_container_width=True, type=btn_type):
                st.session_state.active_session = sid
                st.rerun()
        with col2:
            if st.button("🗑", key=f"del_{sid}", help="Delete session"):
                del st.session_state.sessions[sid]
                if st.session_state.active_session == sid:
                    remaining = [s for s in sorted_ids if s != sid]
                    if remaining:
                        st.session_state.active_session = remaining[0]
                    else:
                        new_sid = _new_session_id()
                        st.session_state.sessions[new_sid] = {
                            "label": "New chat",
                            "messages": [],
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        }
                        st.session_state.active_session = new_sid
                _save_sessions(st.session_state.sessions)
                st.rerun()

    st.divider()
    _active_model = st.session_state.sessions.get(st.session_state.active_session, {}).get("model")
    if _active_model:
        _label = next((k for k, v in MODEL_OPTIONS.items() if v == _active_model), _active_model)
        st.caption(f"Model: **{_label}** (locked)")
    else:
        selected_label = st.selectbox("Model", options=list(MODEL_OPTIONS.keys()), index=0)
        st.session_state["_pending_model"] = MODEL_OPTIONS[selected_label]
    show_thinking = st.toggle("Show thinking", value=False)
    st.caption(f"Session ID: `{st.session_state.active_session[:8]}…`")

# ── Main chat area ─────────────────────────────────────────────────────────────

active_sid = st.session_state.active_session
active_meta = st.session_state.sessions[active_sid]
messages: list[dict] = active_meta["messages"]

st.markdown("# TH Group Nutrition Advisor")

# Render history
for msg in messages:
    with st.chat_message(msg["role"]):
        if show_thinking and msg.get("thinking"):
            with st.expander("💭 Thinking", expanded=False):
                st.markdown(msg["thinking"])
        st.markdown(msg["content"])
        if msg.get("disclaimer"):
            _render_disclaimer(msg["disclaimer"])
        if msg.get("citations"):
            with st.expander("📚 Sources", expanded=False):
                _render_citations(msg["citations"])

# Chat input
if prompt := st.chat_input("Ask about nutrition, TH products, or healthy eating…"):
    # Add user message
    messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Lock model on first message
    if not active_meta.get("model"):
        active_meta["model"] = st.session_state.get("_pending_model", "sonnet")

    # Auto-label session from first user message
    if len(messages) == 1:
        active_meta["label"] = prompt[:40] + ("…" if len(prompt) > 40 else "")

    # Stream assistant response
    with st.chat_message("assistant"):
        try:
            response, thinking, citations, disclaimer = invoke_streaming(prompt, active_sid, active_meta.get("model", "sonnet"))
        except Exception as e:
            response, thinking, citations, disclaimer = f"Sorry, something went wrong: {e}", "", [], ""
            st.error(response)
        else:
            if show_thinking and thinking:
                with st.expander("💭 Thinking", expanded=True):
                    st.markdown(thinking)
            st.markdown(response)
            if disclaimer:
                _render_disclaimer(disclaimer)
            if citations:
                with st.expander("📚 Sources", expanded=False):
                    _render_citations(citations)

    messages.append({"role": "assistant", "content": response, "thinking": thinking, "citations": citations, "disclaimer": disclaimer})
    _save_sessions(st.session_state.sessions)
    st.rerun()
