import streamlit as st
import pandas as pd
from typing import Dict, Any, List

from rag_backend import (
    start_new_session,
    answer_with_mlflow_tracking,
)

st.set_page_config(
    page_title="Agentic Domain-Specific RAG System for Film Analysis",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(circle at top left, rgba(191, 219, 254, 0.22), transparent 28%),
            radial-gradient(circle at bottom right, rgba(125, 211, 252, 0.12), transparent 24%),
            linear-gradient(180deg, #f8fbff 0%, #eef6ff 100%);
        color: #0f172a;
    }

    .block-container {
        padding-top: 1.2rem;
        max-width: 1400px;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #dff2ff 0%, #cfeaff 100%);
        border-right: 1px solid rgba(37, 99, 235, 0.12);
    }

    [data-testid="stSidebar"] * {
        color: #0f172a !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
    }

    [data-testid="stAppViewContainer"] {
        overflow: visible !important;
    }

    .full-header {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000;
        background: linear-gradient(90deg, #fefce8 0%, #fff7ed 50%, #ecfeff 100%);
        border-bottom: 1px solid rgba(148, 163, 184, 0.18);
        padding: 14px 0;
        text-align: center;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        backdrop-filter: blur(10px);
    }

    .full-header-title {
        font-size: 18px;
        font-weight: 800;
        color: #0f172a;
        margin: 0;
    }

    .header-spacer {
        height: 64px;
    }

    .msg-wrap {
        margin-bottom: 0.9rem;
        display: flex;
        width: 100%;
    }

    .msg-wrap.user {
        justify-content: flex-end;
    }

    .msg-wrap.assistant {
        justify-content: flex-start;
    }

    .msg {
        max-width: 80%;
        padding: 1rem 1.05rem;
        border-radius: 18px;
        line-height: 1.6;
        font-size: 0.98rem;
        white-space: pre-wrap;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
        word-wrap: break-word;
        overflow-wrap: break-word;
    }

    .msg.user {
        background: linear-gradient(135deg, #1d4ed8 0%, #2563eb 100%);
        color: #ffffff;
        border-bottom-right-radius: 6px;
    }

    .msg.assistant {
        background: linear-gradient(135deg, #fbcfe8 0%, #c7d2fe 100%);
        color: #0f172a;
        border: 1px solid rgba(79, 70, 229, 0.25);
        border-radius: 20px;
        border-bottom-left-radius: 6px;
        box-shadow: 0 12px 28px rgba(79, 70, 229, 0.12);
    }

    .msg.assistant a {
        color: #3730a3  !important;
        font-weight: 800;
        text-decoration: underline;
    }

    .msg.assistant a:hover {
        color: #064e3b !important;
    }

    .msg.assistant strong,
    .msg.assistant b {
        color: #0f172a !important;
    }

    .section-card {
        background: #ffffff;
        border: 1px solid rgba(148, 163, 184, 0.20);
        border-radius: 14px;
        padding: 0.95rem;
        margin-top: 0.6rem;
        color: #0f172a;
        box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
    }

    .stInfo {
        background: rgba(255,255,255,0.90) !important;
        color: #0f172a !important;
        border: 1px solid rgba(148, 163, 184, 0.16) !important;
    }

    .stChatInputContainer {
        background: transparent;
    }

    [data-testid="stChatInput"] {
        background: #ecfdf5 !important;
        border: 1px solid #86efac !important;
        border-radius: 18px !important;
        box-shadow: 0 8px 24px rgba(34, 197, 94, 0.08) !important;
    }

    [data-testid="stChatInput"] textarea {
        color: #14532d !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #15803d !important;
        opacity: 0.8 !important;
    }

    [data-testid="stChatInputSubmitButton"] {
        background: #22c55e !important;
        border-radius: 999px !important;
        border: none !important;
        box-shadow: 0 6px 16px rgba(34, 197, 94, 0.22) !important;
    }

    [data-testid="stChatInputSubmitButton"] svg {
        fill: white !important;
    }

    .stButton > button {
        border-radius: 12px;
        border: 1px solid rgba(37, 99, 235, 0.12);
        background: rgba(255,255,255,0.86);
        color: #0f172a;
    }

    .stButton > button:hover {
        border-color: rgba(37, 99, 235, 0.22);
        background: rgba(255,255,255,0.98);
        color: #0f172a;
    }

    .stExpander {
        background: rgba(255,255,255,0.82);
        border-radius: 14px;
    }

    /* Sidebar toggle arrow: visible and below header */
    button[kind="header"],
    [data-testid="collapsedControl"] {
        margin-top: 82px !important;
        z-index: 3000 !important;
        background: rgba(255, 255, 255, 0.96) !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.12) !important;
        opacity: 1 !important;
        visibility: visible !important;
    }

    button[kind="header"] svg,
    [data-testid="collapsedControl"] svg {
        fill: #0f172a !important;
        color: #0f172a !important;
        opacity: 1 !important;
        visibility: visible !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_history_from_backend():
    try:
        df = pd.read_csv("/tmp/rag_chatbot_ui/memory/session_logs.csv")
    except Exception:
        return {}

    if df.empty:
        return {}

    sessions = {}

    question_col = "user_question" if "user_question" in df.columns else "question"
    answer_col = "answer" if "answer" in df.columns else "final_answer"

    if question_col not in df.columns or answer_col not in df.columns:
        return {}

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp")

    for _, row in df.iterrows():
        sid = row.get("session_id")
        question = row.get(question_col, "")
        answer = row.get(answer_col, "")

        if not sid:
            continue

        if sid not in sessions:
            title = str(question)[:35] if isinstance(question, str) and question else "Chat"
            sessions[sid] = {
                "title": title,
                "messages": []
            }

        sessions[sid]["messages"].append({
            "role": "user",
            "content": str(question)
        })

        sessions[sid]["messages"].append({
            "role": "assistant",
            "content": str(answer),
            "result": {}
        })

    return sessions


def init_state():
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = start_new_session()

    if "sessions" not in st.session_state:
        loaded_sessions = load_history_from_backend()

        if loaded_sessions:
            st.session_state.sessions = loaded_sessions
            st.session_state.active_session_id = list(loaded_sessions.keys())[-1]
        else:
            st.session_state.sessions = {
                st.session_state.active_session_id: {
                    "title": "New Chat",
                    "messages": []
                }
            }


def get_active_chat() -> Dict[str, Any]:
    return st.session_state.sessions[st.session_state.active_session_id]


def make_title(question: str) -> str:
    q = " ".join(question.strip().split())
    if len(q) <= 35:
        return q
    return q[:32] + "..."


def new_chat():
    session_id = start_new_session()
    st.session_state.active_session_id = session_id
    st.session_state.sessions[session_id] = {
        "title": "New Chat",
        "messages": []
    }


def delete_chat(session_id: str):
    if session_id in st.session_state.sessions:
        del st.session_state.sessions[session_id]

    if not st.session_state.sessions:
        new_chat()
        return

    if st.session_state.active_session_id == session_id:
        st.session_state.active_session_id = list(st.session_state.sessions.keys())[-1]


def render_sidebar():
    with st.sidebar:
        st.markdown("## 💬 Chats")

        if st.button("➕ New Chat", use_container_width=True):
            new_chat()
            st.rerun()

        st.markdown("---")
        st.markdown("### History")

        session_items = list(st.session_state.sessions.items())[::-1]
        for session_id, data in session_items:
            label = data["title"] if data["title"] else "New Chat"

            col1, col2 = st.columns([5, 1])

            with col1:
                if st.button(label, key=f"chat_{session_id}", use_container_width=True):
                    st.session_state.active_session_id = session_id
                    st.rerun()

            with col2:
                if st.button("🗑️", key=f"delete_{session_id}", use_container_width=True):
                    delete_chat(session_id)
                    st.rerun()


def render_message(role: str, content: str):
    role_class = "user" if role == "user" else "assistant"
    st.markdown(
        f"""
        <div class="msg-wrap {role_class}">
            <div class="msg {role_class}">{content}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def safe_list(v) -> List[Any]:
    if v is None:
        return []
    if isinstance(v, list):
        return v
    return []


def render_result_panels(result: Dict[str, Any]):
    decision = result.get("decision_action", "-")
    reason = result.get("decision_reason", "-")
    latency = result.get("latency_seconds", "-")

    with st.expander("Sources and metadata"):
        st.write(f"**Decision:** {decision}")
        st.write(f"**Reason:** {reason}")
        st.write(f"**Latency:** {latency}")

        internal_sources = safe_list(result.get("internal_sources"))
        google_results = safe_list(result.get("google_results"))
        youtube_results = safe_list(result.get("youtube_results"))

        st.markdown("### Internal Sources")
        if internal_sources:
            for item in internal_sources:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <b>{item.get("file_name", "Unknown file")}</b><br>
                        Chunk ID: {item.get("chunk_id", "-")}<br>
                        Preview: {item.get("preview", "")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("No internal sources.")

        st.markdown("### Google Results")
        if google_results:
            for item in google_results:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <b>{item.get("title", "")}</b><br>
                        {item.get("link", "")}<br>
                        {item.get("snippet", "")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("No Google results.")

        st.markdown("### YouTube Results")
        if youtube_results:
            for item in youtube_results:
                st.markdown(
                    f"""
                    <div class="section-card">
                        <b>{item.get("title", "")}</b><br>
                        {item.get("link", "")}<br>
                        {item.get("snippet", "")}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        else:
            st.write("No YouTube results.")

        st.markdown("### Raw Response")
        st.json(result)


def main():
    inject_css()
    init_state()
    render_sidebar()

    st.markdown(
        """
        <div class="full-header">
            <div class="full-header-title">
                Agentic Domain-Specific RAG System for Film Analysis
            </div>
        </div>
        <div class="header-spacer"></div>
        """,
        unsafe_allow_html=True,
    )

    active_chat = get_active_chat()
    messages = active_chat["messages"]

    st.markdown('<div class="chat-shell">', unsafe_allow_html=True)

    if not messages:
        st.info("Start a new conversation.")

    for msg in messages:
        render_message(msg["role"], msg["content"])
        if msg["role"] == "assistant" and msg.get("result"):
            render_result_panels(msg["result"])

    st.markdown("</div>", unsafe_allow_html=True)

    prompt = st.chat_input("Ask your question...")
    if prompt:
        if active_chat["title"] == "New Chat":
            active_chat["title"] = make_title(prompt)

        active_chat["messages"].append({
            "role": "user",
            "content": prompt
        })

        with st.spinner("Thinking..."):
            try:
                result = answer_with_mlflow_tracking(
                    question=prompt,
                    session_id=st.session_state.active_session_id
                )
                answer = result.get("answer", "No answer returned.")
            except Exception as e:
                result = {
                    "decision_action": "ERROR",
                    "decision_reason": str(e),
                    "answer": f"Error: {e}",
                    "internal_sources": [],
                    "google_results": [],
                    "youtube_results": [],
                    "latency_seconds": None,
                }
                answer = result["answer"]

        active_chat["messages"].append({
            "role": "assistant",
            "content": answer,
            "result": result,
        })

        st.rerun()


if __name__ == "__main__":
    main()