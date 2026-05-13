import streamlit as st
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "agentic_rag"))

# pyrefly: ignore [missing-import]
from backend.graph import build_graph
# pyrefly: ignore [missing-import]
from backend.core.logger import log_agentic_state

st.set_page_config(
    page_title="Agentic RAG",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* ─────────────────────────────────────────────
       FONT IMPORT — IBM Plex Mono + Syne
    ───────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    /* ─────────────────────────────────────────────
       CSS VARIABLES
    ───────────────────────────────────────────── */
    :root {
        --bg-base:        #09090b;
        --bg-surface:     #0e1015;
        --bg-raised:      #13161d;
        --bg-glass:       rgba(14, 16, 21, 0.72);

        --border-subtle:  rgba(255,255,255,0.055);
        --border-mid:     rgba(255,255,255,0.09);
        --border-accent:  rgba(6, 182, 212, 0.35);

        --cyan:           #06b6d4;
        --cyan-dim:       rgba(6, 182, 212, 0.18);
        --cyan-glow:      rgba(6, 182, 212, 0.08);
        --violet:         #7c3aed;
        --violet-dim:     rgba(124, 58, 237, 0.18);
        --green:          #10b981;
        --green-dim:      rgba(16, 185, 129, 0.15);
        --amber:          #f59e0b;
        --amber-dim:      rgba(245, 158, 11, 0.15);
        --red:            #ef4444;
        --red-dim:        rgba(239, 68, 68, 0.15);

        --text-primary:   #f1f5f9;
        --text-secondary: #94a3b8;
        --text-muted:     #4a5568;

        --radius-sm:      6px;
        --radius-md:      10px;
        --radius-lg:      14px;

        --font-ui:        'Syne', sans-serif;
        --font-mono:      'IBM Plex Mono', monospace;

        --transition:     all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    }

    /* ─────────────────────────────────────────────
       GLOBAL BASE
    ───────────────────────────────────────────── */
    html, body, .stApp {
        background-color: var(--bg-base) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-ui) !important;
    }

    /* Subtle scanline texture overlay */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,0,0,0.08) 2px,
            rgba(0,0,0,0.08) 4px
        );
        pointer-events: none;
        z-index: 0;
        opacity: 0.4;
    }

    .stApp > header { background: transparent !important; }

    /* ─────────────────────────────────────────────
       SIDEBAR
    ───────────────────────────────────────────── */
    [data-testid="stSidebar"] {
        background-color: var(--bg-surface) !important;
        border-right: 1px solid var(--border-subtle) !important;
        backdrop-filter: blur(12px);
    }
    [data-testid="stSidebar"]::after {
        content: '';
        position: absolute;
        top: 0; right: 0;
        width: 1px; height: 100%;
        background: linear-gradient(
            180deg,
            transparent 0%,
            var(--cyan) 30%,
            var(--violet) 70%,
            transparent 100%
        );
        opacity: 0.3;
    }

    /* ─────────────────────────────────────────────
       HIDE STREAMLIT CHROME
    ───────────────────────────────────────────── */
    #MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden !important; }

    /* ─────────────────────────────────────────────
       TYPOGRAPHY
    ───────────────────────────────────────────── */
    h1, h2, h3 {
        font-family: var(--font-ui) !important;
        font-weight: 800 !important;
        letter-spacing: -0.03em !important;
        color: var(--text-primary) !important;
    }
    h2 {
        background: linear-gradient(135deg, #f1f5f9 0%, var(--cyan) 60%, var(--violet) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    /* ─────────────────────────────────────────────
       METRIC CARDS — Glassmorphism Telemetry
    ───────────────────────────────────────────── */
    .metric-card {
        position: relative;
        background: var(--bg-glass);
        border: 1px solid var(--border-mid);
        border-radius: var(--radius-lg);
        padding: 20px 22px;
        margin-bottom: 10px;
        overflow: hidden;
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        transition: var(--transition);
        box-shadow:
            0 1px 0 0 rgba(255,255,255,0.04) inset,
            0 4px 24px rgba(0,0,0,0.35);
    }
    /* Top edge glint */
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 10%; right: 10%;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent);
    }
    /* Corner accent dot */
    .metric-card::after {
        content: '';
        position: absolute;
        top: 14px; right: 16px;
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--cyan);
        box-shadow: 0 0 8px var(--cyan), 0 0 16px var(--cyan-dim);
        animation: pulse-dot 2.4s ease-in-out infinite;
    }
    .metric-card:hover {
        border-color: var(--border-accent);
        transform: translateY(-2px);
        box-shadow:
            0 1px 0 0 rgba(255,255,255,0.06) inset,
            0 8px 32px rgba(0,0,0,0.5),
            0 0 20px var(--cyan-glow);
    }
    .metric-card .val {
        font-family: var(--font-mono) !important;
        font-size: 2rem;
        font-weight: 600;
        line-height: 1;
        margin-bottom: 6px;
        background: linear-gradient(135deg, #e2e8f0 0%, var(--cyan) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.04em;
        font-variant-numeric: tabular-nums;
    }
    .metric-card .lbl {
        font-family: var(--font-mono) !important;
        font-size: 0.62rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 0.14em;
        margin-top: 2px;
    }

    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 6px var(--cyan), 0 0 12px var(--cyan-dim); }
        50%       { opacity: 0.4; box-shadow: 0 0 2px var(--cyan); }
    }

    /* ─────────────────────────────────────────────
       STATUS PILLS — System Badges
    ───────────────────────────────────────────── */
    .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 5px 13px;
        border-radius: 4px;
        font-family: var(--font-mono) !important;
        font-size: 0.68rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        transition: var(--transition);
        cursor: default;
        position: relative;
        overflow: hidden;
    }
    /* Shimmer sweep on hover */
    .pill::after {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(105deg, transparent 40%, rgba(255,255,255,0.08) 50%, transparent 60%);
        transform: translateX(-100%);
        transition: transform 0.5s ease;
    }
    .pill:hover::after { transform: translateX(100%); }

    .pill-web {
        background: linear-gradient(135deg, rgba(239,68,68,0.12), rgba(239,68,68,0.06));
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.35);
        box-shadow: 0 0 12px rgba(239,68,68,0.1), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .pill-web:hover {
        background: linear-gradient(135deg, rgba(239,68,68,0.2), rgba(239,68,68,0.1));
        border-color: rgba(239,68,68,0.6);
        box-shadow: 0 0 20px rgba(239,68,68,0.2);
    }

    .pill-local {
        background: linear-gradient(135deg, rgba(16,185,129,0.12), rgba(16,185,129,0.06));
        color: #6ee7b7;
        border: 1px solid rgba(16,185,129,0.35);
        box-shadow: 0 0 12px rgba(16,185,129,0.1), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .pill-local:hover {
        background: linear-gradient(135deg, rgba(16,185,129,0.2), rgba(16,185,129,0.1));
        border-color: rgba(16,185,129,0.6);
        box-shadow: 0 0 20px rgba(16,185,129,0.2);
    }

    .pill-warn {
        background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(245,158,11,0.06));
        color: #fcd34d;
        border: 1px solid rgba(245,158,11,0.35);
        box-shadow: 0 0 12px rgba(245,158,11,0.1), inset 0 1px 0 rgba(255,255,255,0.04);
    }
    .pill-warn:hover {
        background: linear-gradient(135deg, rgba(245,158,11,0.2), rgba(245,158,11,0.1));
        border-color: rgba(245,158,11,0.6);
        box-shadow: 0 0 20px rgba(245,158,11,0.2);
    }

    /* ─────────────────────────────────────────────
       CHAT MESSAGES
    ───────────────────────────────────────────── */
    [data-testid="stChatMessage"] {
        border-radius: var(--radius-md) !important;
        padding: 1rem 1.1rem !important;
        margin-bottom: 10px !important;
        transition: var(--transition) !important;
        position: relative;
    }

    /* User bubble */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
        background: linear-gradient(135deg, #0f172a 0%, #111827 100%) !important;
        border: 1px solid rgba(99, 102, 241, 0.25) !important;
        box-shadow: 0 2px 16px rgba(0,0,0,0.3) !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]):hover {
        border-color: rgba(99, 102, 241, 0.45) !important;
    }

    /* AI bubble — verified response glow on left border */
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
        background: linear-gradient(135deg, #0a0f16 0%, #0e1318 100%) !important;
        border: 1px solid var(--border-subtle) !important;
        border-left: 3px solid var(--cyan) !important;
        box-shadow:
            -4px 0 20px rgba(6, 182, 212, 0.12),
            0 2px 20px rgba(0,0,0,0.4) !important;
    }
    div[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]):hover {
        border-color: var(--border-subtle) !important;
        border-left-color: var(--cyan) !important;
        box-shadow:
            -6px 0 28px rgba(6, 182, 212, 0.2),
            0 4px 28px rgba(0,0,0,0.5) !important;
    }

    /* ─────────────────────────────────────────────
       CHAT INPUT
    ───────────────────────────────────────────── */
    [data-testid="stChatInput"] {
        background: transparent !important;
    }
    [data-testid="stChatInput"] textarea {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border-mid) !important;
        border-radius: var(--radius-md) !important;
        color: var(--text-primary) !important;
        font-family: var(--font-ui) !important;
        transition: var(--transition) !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: var(--cyan) !important;
        box-shadow:
            0 0 0 3px var(--cyan-glow),
            0 0 20px rgba(6,182,212,0.06) !important;
        outline: none !important;
    }
    [data-testid="stChatInput"] textarea::placeholder {
        color: var(--text-muted) !important;
        font-family: var(--font-mono) !important;
        font-size: 0.82rem !important;
    }

    /* ─────────────────────────────────────────────
       EXPANDERS
    ───────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--bg-surface) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-md) !important;
        transition: var(--transition) !important;
        overflow: hidden;
    }
    [data-testid="stExpander"]:hover {
        border-color: var(--border-mid) !important;
        box-shadow: 0 4px 20px rgba(0,0,0,0.35) !important;
    }
    [data-testid="stExpander"] summary {
        color: var(--text-secondary) !important;
        font-size: 0.8rem !important;
        letter-spacing: 0.04em !important;
        transition: color 0.2s ease !important;
        padding: 10px 14px !important;
    }
    [data-testid="stExpander"] summary:hover {
        color: var(--cyan) !important;
    }
    /* Animated expand indicator */
    [data-testid="stExpander"] summary svg {
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    }

    /* ─────────────────────────────────────────────
       DIVIDER
    ───────────────────────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg,
            transparent 0%,
            var(--border-mid) 20%,
            var(--border-mid) 80%,
            transparent 100%) !important;
        margin: 14px 0 !important;
    }

    /* ─────────────────────────────────────────────
       SIDEBAR BUTTON
    ───────────────────────────────────────────── */
    .stButton > button {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border-mid) !important;
        color: var(--text-secondary) !important;
        border-radius: var(--radius-sm) !important;
        width: 100% !important;
        font-family: var(--font-mono) !important;
        font-size: 0.76rem !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        padding: 8px 16px !important;
        transition: var(--transition) !important;
        position: relative;
        overflow: hidden;
    }
    .stButton > button::before {
        content: '';
        position: absolute;
        inset: 0;
        background: linear-gradient(135deg, var(--cyan-dim), transparent);
        opacity: 0;
        transition: opacity 0.3s ease;
    }
    .stButton > button:hover {
        border-color: var(--cyan) !important;
        color: var(--cyan) !important;
        background: rgba(6, 182, 212, 0.04) !important;
        box-shadow: 0 0 16px rgba(6,182,212,0.1), inset 0 1px 0 rgba(255,255,255,0.05) !important;
        transform: translateY(-1px) !important;
    }
    .stButton > button:hover::before { opacity: 1; }
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: none !important;
    }

    /* ─────────────────────────────────────────────
       SOURCE TAG (monospace chip)
    ───────────────────────────────────────────── */
    .src-tag {
        font-family: var(--font-mono) !important;
        font-size: 0.7rem;
        background: linear-gradient(135deg, rgba(6,182,212,0.1), rgba(124,58,237,0.08));
        color: #67e8f9;
        padding: 2px 8px;
        border-radius: 4px;
        border: 1px solid rgba(6,182,212,0.2);
        letter-spacing: 0.04em;
        transition: var(--transition);
    }
    .src-tag:hover {
        border-color: rgba(6,182,212,0.45);
        box-shadow: 0 0 10px rgba(6,182,212,0.1);
    }

    /* ─────────────────────────────────────────────
       STREAMLIT METRICS (inside expander)
    ───────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-raised) !important;
        border: 1px solid var(--border-subtle) !important;
        border-radius: var(--radius-sm) !important;
        padding: 12px 14px !important;
        transition: var(--transition) !important;
    }
    [data-testid="stMetric"]:hover {
        border-color: var(--border-mid) !important;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3) !important;
    }
    [data-testid="stMetricLabel"] {
        font-family: var(--font-mono) !important;
        font-size: 0.65rem !important;
        color: var(--text-muted) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.1em !important;
    }
    [data-testid="stMetricValue"] {
        font-family: var(--font-mono) !important;
        color: var(--text-primary) !important;
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.03em !important;
    }

    /* ─────────────────────────────────────────────
       CAPTIONS & SMALL TEXT
    ───────────────────────────────────────────── */
    [data-testid="stCaptionContainer"], .stCaption {
        font-family: var(--font-mono) !important;
        color: var(--text-muted) !important;
        font-size: 0.72rem !important;
        line-height: 1.55 !important;
    }

    /* ─────────────────────────────────────────────
       CODE BLOCKS (failed queries log)
    ───────────────────────────────────────────── */
    code, pre {
        font-family: var(--font-mono) !important;
        background: rgba(6,182,212,0.05) !important;
        border: 1px solid rgba(6,182,212,0.12) !important;
        border-radius: var(--radius-sm) !important;
        color: #7dd3fc !important;
        font-size: 0.76rem !important;
    }

    /* ─────────────────────────────────────────────
       SCROLLBAR
    ───────────────────────────────────────────── */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg-base); }
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, var(--cyan), var(--violet));
        border-radius: 2px;
    }
    ::-webkit-scrollbar-thumb:hover { background: var(--cyan); }

    /* ─────────────────────────────────────────────
       SPINNER
    ───────────────────────────────────────────── */
    [data-testid="stSpinner"] > div {
        border-top-color: var(--cyan) !important;
    }

    /* ─────────────────────────────────────────────
       PAGE-LOAD FADE-IN
    ───────────────────────────────────────────── */
    .main .block-container {
        animation: fadeInUp 0.5s cubic-bezier(0.4, 0, 0.2, 1) both;
        padding-top: 2rem !important;
    }
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_agent():
    return build_graph()


def confidence_color(val: float) -> str:
    if val >= 0.75:
        return "#4ade80"
    if val >= 0.5:
        return "#fbbf24"
    return "#f87171"


def render_telemetry(state: dict, latency: float):
    cols = st.columns(4)
    metrics = [
        ("Retry Loops",       str(state.get("loop_count", 0))),
        ("Critic Confidence", f"{state.get('critic_confidence', 0.0):.2f}"),
        ("Context Relevance", f"{state.get('relevance_score', 0.0):.2f}"),
        ("Latency",           f"{latency:.2f}s"),
    ]
    for col, (lbl, val) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="val">{val}</div>
                <div class="lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)


def render_details(state: dict):
    docs = state.get("approved_context", [])
    is_web = any(
        str(doc.metadata.get("source", "")).startswith("Internet_")
        for doc in docs
    )

    if is_web:
        st.markdown('<span class="pill pill-web">🌐 Web fallback triggered</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="pill pill-local">🔒 Local hybrid retrieval</span>', unsafe_allow_html=True)

    loops = state.get("loop_count", 0)
    if loops > 1:
        st.markdown(
            f'&nbsp;&nbsp;<span class="pill pill-warn">⟳ {loops} retry loops</span>',
            unsafe_allow_html=True
        )

    st.write("")

    with st.expander(f"📄 Context documents ({len(docs)})"):
        if not docs:
            st.caption("No context approved.")
        for i, doc in enumerate(docs):
            src = doc.metadata.get("source", "Unknown")
            st.markdown(f'**{i+1}.** <span class="src-tag">{src}</span>', unsafe_allow_html=True)
            st.caption(doc.page_content[:600] + ("…" if len(doc.page_content) > 600 else ""))
            if i < len(docs) - 1:
                st.divider()

    with st.expander("🧠 Critic analysis"):
        st.markdown(f"**Reasoning:** {state.get('critic_reasoning', 'N/A')}")
        st.markdown(f"**Feedback:** {state.get('critic_feedback', 'N/A')}")
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Agreement score",      f"{state.get('document_agreement_score', 1.0):.2f}")
            st.metric("Reranking conf (RRF)",  f"{state.get('reranking_confidence', 0.0):.2f}")
        with c2:
            st.metric("Hallucination risk",   f"{state.get('hallucination_risk_score', 0.0):.2f}")
            st.metric("Contradictory docs",   "Yes" if state.get("docs_contradictory") else "No")

    history = state.get("retry_history", [])
    if history:
        with st.expander(f"🔄 Retry log ({len(history)} loops)"):
            for h in history:
                cc = confidence_color(h.get("confidence", 0))
                st.markdown(
                    f"**Loop {h['loop']}** — relevance `{h['relevance_score']}` "
                    f"— confidence <span style='color:{cc}'>`{h['confidence']}`</span>",
                    unsafe_allow_html=True
                )
            failed = state.get("failed_queries", [])
            if failed:
                st.divider()
                st.caption("Failed queries in memory:")
                for fq in failed:
                    st.code(fq, language=None)


def main():
    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🤖 Agentic RAG")
        st.caption("Critic-guided hybrid retrieval with web fallback.")
        st.divider()

        st.markdown("**Pipeline**")
        for step in [
            "Hybrid Search (ChromaDB + BM25)",
            "Cross-encoder Reranker",
            "Context Critic",
            "Web Fallback (DuckDuckGo)",
            "Answer Generator",
        ]:
            st.markdown(
                f"<span style='color:#475569;font-size:0.8rem'>→ {step}</span>",
                unsafe_allow_html=True
            )

        st.divider()
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.rerun()

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("## Critic-Guided Agentic RAG")
    st.caption("Multi-agent hybrid search · Context critic · Failure memory · Web fallback")
    st.divider()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.spinner("Loading agents…"):
        agent_app = load_agent()

    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "state" in msg:
                st.divider()
                render_details(msg["state"])
                render_telemetry(msg["state"], msg["latency"])

    # New input
    if prompt := st.chat_input("Ask something…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agents coordinating…"):
                t0 = time.time()
                final_state = agent_app.invoke({
                    "original_query": prompt,
                    "loop_count": 0
                }, config={"recursion_limit": 100})
                latency = time.time() - t0

            log_agentic_state(final_state, latency)

            answer = final_state.get("final_answer", "No answer generated.")
            st.markdown(answer)
            st.divider()
            render_details(final_state)
            render_telemetry(final_state, latency)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "state": final_state,
                "latency": latency,
            })


if __name__ == "__main__":
    main()