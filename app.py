"""Vidya Offline — Premium UI · fully on-device AI tutor.

Run with:  streamlit run app.py   |   or double-click run.bat on Windows
"""
from __future__ import annotations

import dataclasses

import streamlit as st

from vidya.config import load_config
from vidya.embeddings import EmbeddingError, OllamaEmbedder
from vidya.ingest import SUPPORTED, IngestError
from vidya.library import Library
from vidya.llm import LLM, LLMError, is_embed_model, pick_default
from vidya.pipeline import index_document, ingest_file
from vidya.rag import CLASS_LEVELS, Tutor
from vidya.store import VectorStore, slug
from vidya.study import (StudySource, flashcards_to_csv, gather_context,
                          make_flashcards, make_mindmap, make_quiz,
                          mindmap_to_dot, summary_messages, translate_cards,
                          translate_mindmap, translate_quiz)
from vidya.textrank import top_sentences
from vidya.translate import ENGLISH, LANGUAGES, Translator, looks_non_english

cfg = load_config()

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Vidya Offline",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL STYLES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Typography ─────────────────────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stApp {
    font-family: 'Plus Jakarta Sans', system-ui, sans-serif !important;
}

/* ── App shell ──────────────────────────────────────────────────────────── */
.stApp { background: #0f0e17 !important; }
.block-container { padding-top: 1.4rem !important; padding-bottom: 2rem !important; }
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ────────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #13122a 0%, #0f0e1f 100%) !important;
    border-right: 1px solid rgba(129,140,248,.18) !important;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── HERO BANNER ────────────────────────────────────────────────────────── */
.hero {
    position: relative;
    overflow: hidden;
    background: linear-gradient(135deg,#312e81 0%,#4c1d95 40%,#1e1b4b 100%);
    border-radius: 24px;
    padding: 36px 44px 30px;
    margin-bottom: 28px;
    box-shadow: 0 20px 60px rgba(99,102,241,.35), inset 0 1px 0 rgba(255,255,255,.08);
}
.hero::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 60% 80% at 80% 20%,
        rgba(167,139,250,.25) 0%, transparent 60%);
    pointer-events: none;
}
.hero-glow {
    position: absolute; right: -20px; top: -20px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(139,92,246,.4) 0%, transparent 65%);
    pointer-events: none;
}
.hero-title {
    font-size: 2.6rem; font-weight: 800; letter-spacing: -1px;
    color: #fff; margin: 0 0 6px; line-height: 1.1;
    text-shadow: 0 2px 20px rgba(0,0,0,.4);
}
.hero-title span { color: #c4b5fd; }
.hero-sub {
    font-size: .93rem; color: rgba(255,255,255,.72);
    margin: 0 0 20px; line-height: 1.6;
}
.hero-badges { display: flex; gap: 10px; flex-wrap: wrap; }
.hbadge {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 5px 14px; border-radius: 999px;
    font-size: .72rem; font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
    backdrop-filter: blur(8px);
    border: 1px solid rgba(255,255,255,.15);
    color: rgba(255,255,255,.88);
    background: rgba(255,255,255,.08);
}

/* ── GLASS CARDS ────────────────────────────────────────────────────────── */
.glass-card {
    background: rgba(26,24,48,.7);
    border: 1px solid rgba(129,140,248,.18);
    border-radius: 20px;
    padding: 24px 28px;
    backdrop-filter: blur(16px);
    box-shadow: 0 4px 32px rgba(0,0,0,.3), inset 0 1px 0 rgba(255,255,255,.05);
    margin-bottom: 16px;
}

/* ── SECTION HEADERS ────────────────────────────────────────────────────── */
.sec-hdr {
    display: flex; align-items: center; gap: 10px;
    font-size: 1.05rem; font-weight: 700; color: #c4b5fd;
    margin: 20px 0 12px; letter-spacing: -.2px;
}
.sec-hdr-line {
    flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(129,140,248,.35), transparent);
}
.sidebar-sec {
    font-size: .78rem; font-weight: 700; letter-spacing: 1.2px;
    text-transform: uppercase; color: #818cf8; margin: 18px 0 10px;
}

/* ── PILL BADGE ─────────────────────────────────────────────────────────── */
.pill {
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: .7rem; font-weight: 700; letter-spacing: .5px; text-transform: uppercase;
}
.pill-indigo { background: rgba(99,102,241,.2); color: #a5b4fc; border: 1px solid rgba(99,102,241,.3); }
.pill-green  { background: rgba(16,185,129,.15); color: #6ee7b7; border: 1px solid rgba(16,185,129,.25); }
.pill-amber  { background: rgba(245,158,11,.15); color: #fcd34d; border: 1px solid rgba(245,158,11,.25); }
.pill-red    { background: rgba(239,68,68,.15);  color: #fca5a5; border: 1px solid rgba(239,68,68,.25); }

/* ── BOOK CARD (sidebar) ────────────────────────────────────────────────── */
.book-row {
    display: flex; align-items: center; gap: 10px;
    background: rgba(129,140,248,.07);
    border: 1px solid rgba(129,140,248,.14);
    border-radius: 12px; padding: 10px 12px;
    margin-bottom: 8px; transition: border-color .2s;
}
.book-row:hover { border-color: rgba(129,140,248,.4); }
.book-ico { font-size: 1.5rem; flex-shrink: 0; }
.book-name {
    font-weight: 600; font-size: .84rem; color: #e0e7ff;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis; flex: 1;
}
.book-meta { font-size: .72rem; color: #6b7280; margin-top: 1px; }

/* ── TABS ───────────────────────────────────────────────────────────────── */
[data-testid="stTabs"] {
    border-bottom: 1px solid rgba(129,140,248,.2);
    margin-bottom: 4px;
}
[data-testid="stTabs"] [role="tablist"] { gap: 2px; }
[data-testid="stTabs"] [role="tab"] {
    font-size: .84rem; font-weight: 600;
    color: #6b7280; padding: 10px 20px;
    border-radius: 10px 10px 0 0;
    border: none !important; background: transparent !important;
    transition: color .15s, background .15s;
}
[data-testid="stTabs"] [role="tab"]:hover { color: #c4b5fd; background: rgba(129,140,248,.08) !important; }
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #a5b4fc !important; font-weight: 800 !important;
    background: rgba(99,102,241,.15) !important;
    box-shadow: inset 0 -2px 0 #818cf8;
}

/* ── CHAT ───────────────────────────────────────────────────────────────── */
[data-testid="stChatMessage"] {
    background: rgba(26,24,48,.55) !important;
    border: 1px solid rgba(129,140,248,.12) !important;
    border-radius: 16px !important;
    backdrop-filter: blur(8px);
    margin-bottom: 10px !important;
}
[data-testid="stChatInput"] textarea {
    background: rgba(26,24,48,.8) !important;
    border: 1.5px solid rgba(129,140,248,.3) !important;
    border-radius: 14px !important;
    color: #e8e6f0 !important;
    font-family: inherit !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,.18) !important;
}

/* ── FLASHCARD ──────────────────────────────────────────────────────────── */
.flashcard-wrap {
    perspective: 1000px;
    margin: 24px auto;
    max-width: 760px;
}
.flashcard {
    position: relative;
    min-height: 260px;
    background: linear-gradient(135deg, #1e1b4b 0%, #2e1065 100%);
    border: 1.5px solid rgba(167,139,250,.35);
    border-radius: 24px;
    padding: 48px 40px;
    text-align: center;
    display: flex; flex-direction: column; justify-content: center; align-items: center;
    box-shadow: 0 20px 60px rgba(0,0,0,.5), inset 0 1px 0 rgba(255,255,255,.07);
    overflow: hidden;
}
.flashcard::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(ellipse 70% 60% at 50% 0%, rgba(167,139,250,.18), transparent 60%);
}
.fc-side-label {
    font-size: .65rem; font-weight: 800; letter-spacing: 2px; text-transform: uppercase;
    color: #818cf8; margin-bottom: 20px;
    display: flex; align-items: center; gap: 7px;
}
.fc-side-label::before, .fc-side-label::after {
    content: ''; flex: 1; height: 1px; background: rgba(129,140,248,.3);
}
.fc-text {
    font-size: 1.5rem; font-weight: 700; color: #e8e6f0;
    line-height: 1.55; position: relative; z-index: 1;
    max-width: 640px;
}
.fc-counter {
    position: absolute; bottom: 18px; right: 22px;
    font-size: .7rem; color: rgba(255,255,255,.3); font-weight: 600;
}
.fc-dot-row {
    display: flex; justify-content: center; gap: 6px; margin: 16px 0 4px;
}
.fc-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: rgba(129,140,248,.2); transition: background .2s;
}
.fc-dot.active { background: #818cf8; }

/* ── QUIZ ───────────────────────────────────────────────────────────────── */
.quiz-card {
    background: rgba(26,24,48,.65);
    border: 1px solid rgba(129,140,248,.18);
    border-radius: 18px; padding: 24px 28px;
    margin-bottom: 16px;
    backdrop-filter: blur(10px);
    transition: border-color .2s;
}
.quiz-card:hover { border-color: rgba(129,140,248,.35); }
.quiz-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: linear-gradient(135deg,#4f46e5,#7c3aed);
    font-size: .78rem; font-weight: 800; color: #fff;
    margin-right: 10px; flex-shrink: 0;
    box-shadow: 0 2px 8px rgba(99,102,241,.4);
}
.quiz-q-text { font-size: .97rem; font-weight: 600; color: #e0e7ff; }

/* ── SCORE CARD ─────────────────────────────────────────────────────────── */
.score-wrap {
    background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%);
    border-radius: 20px; padding: 32px 24px;
    text-align: center; color: #fff;
    box-shadow: 0 12px 40px rgba(99,102,241,.45);
    position: relative; overflow: hidden;
}
.score-wrap::before {
    content: '';
    position: absolute; inset: 0;
    background: radial-gradient(circle at 30% 20%, rgba(255,255,255,.12), transparent 50%);
}
.score-emoji { font-size: 2.4rem; display: block; margin-bottom: 8px; }
.score-num { font-size: 3.2rem; font-weight: 900; line-height: 1; }
.score-denom { font-size: 1.3rem; font-weight: 500; opacity: .7; }
.score-pct { font-size: .9rem; opacity: .8; margin-top: 6px; letter-spacing: .5px; }
.score-msg { font-size: 1rem; font-weight: 700; margin-top: 10px; }

/* ── MIND MAP ───────────────────────────────────────────────────────────── */
[data-testid="stGraphvizChart"] svg {
    border-radius: 16px;
    background: rgba(26,24,48,.5) !important;
    padding: 16px;
}

/* ── BUTTONS ────────────────────────────────────────────────────────────── */
[data-testid="stButton"] > button {
    border-radius: 12px !important;
    font-weight: 600 !important; font-size: .85rem !important;
    transition: all .18s cubic-bezier(.4,0,.2,1) !important;
    letter-spacing: .2px !important;
}
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg,#4f46e5 0%,#7c3aed 100%) !important;
    border: none !important; color: #fff !important;
    box-shadow: 0 4px 16px rgba(99,102,241,.45) !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 24px rgba(99,102,241,.55) !important;
}
[data-testid="stButton"] > button:not([kind="primary"]) {
    background: rgba(129,140,248,.1) !important;
    border: 1.5px solid rgba(129,140,248,.25) !important;
    color: #a5b4fc !important;
}
[data-testid="stButton"] > button:not([kind="primary"]):hover {
    background: rgba(129,140,248,.2) !important;
    border-color: rgba(129,140,248,.5) !important;
    transform: translateY(-1px) !important;
}

/* ── DOWNLOAD BUTTON ────────────────────────────────────────────────────── */
[data-testid="stDownloadButton"] > button {
    border-radius: 12px !important; font-weight: 600 !important;
    background: rgba(16,185,129,.12) !important;
    border: 1.5px solid rgba(16,185,129,.3) !important;
    color: #6ee7b7 !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: rgba(16,185,129,.22) !important;
    transform: translateY(-1px) !important;
}

/* ── FORM WIDGETS ───────────────────────────────────────────────────────── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(26,24,48,.8) !important;
    border: 1.5px solid rgba(129,140,248,.2) !important;
    border-radius: 10px !important;
}
[data-testid="stSlider"] [data-testid="stThumbValue"] { color: #a5b4fc !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background: #818cf8 !important;
    box-shadow: 0 0 0 4px rgba(129,140,248,.2) !important;
}
[data-testid="stFileUploadDropzone"] {
    background: rgba(99,102,241,.06) !important;
    border: 2px dashed rgba(129,140,248,.35) !important;
    border-radius: 16px !important;
    transition: border-color .2s, background .2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
    background: rgba(99,102,241,.12) !important;
    border-color: rgba(129,140,248,.7) !important;
}
[data-testid="stTextInput"] input {
    background: rgba(26,24,48,.8) !important;
    border: 1.5px solid rgba(129,140,248,.2) !important;
    border-radius: 10px !important; color: #e8e6f0 !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #818cf8 !important;
    box-shadow: 0 0 0 3px rgba(129,140,248,.18) !important;
}
[data-testid="stNumberInput"] input {
    background: rgba(26,24,48,.8) !important;
    border: 1.5px solid rgba(129,140,248,.2) !important;
    border-radius: 10px !important; color: #e8e6f0 !important;
}
[data-testid="stCheckbox"] span { color: #c4b5fd !important; }
[data-testid="stRadio"] label { color: #c4b5fd !important; font-weight: 500 !important; }

/* ── EXPANDER ───────────────────────────────────────────────────────────── */
[data-testid="stExpander"] {
    background: rgba(26,24,48,.5) !important;
    border: 1px solid rgba(129,140,248,.18) !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary { font-weight: 600 !important; color: #a5b4fc !important; }

/* ── ALERTS ─────────────────────────────────────────────────────────────── */
[data-testid="stAlert"] {
    border-radius: 14px !important;
    border-left-width: 3px !important;
}

/* ── PROGRESS BAR ───────────────────────────────────────────────────────── */
[data-testid="stProgress"] > div { background: rgba(129,140,248,.18) !important; border-radius: 999px; }
[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7) !important;
    border-radius: 999px;
    box-shadow: 0 0 10px rgba(129,140,248,.5);
}

/* ── SPINNER ────────────────────────────────────────────────────────────── */
[data-testid="stSpinner"] p { color: #818cf8 !important; font-weight: 600 !important; }

/* ── HORIZONTAL RULE ────────────────────────────────────────────────────── */
hr { border-color: rgba(129,140,248,.15) !important; }

/* ── METRIC ─────────────────────────────────────────────────────────────── */
[data-testid="stMetric"] { background: rgba(26,24,48,.5); border-radius: 14px; padding: 16px 20px; }
[data-testid="stMetricValue"] { color: #a5b4fc !important; font-weight: 800 !important; }

/* ── CAPTION ────────────────────────────────────────────────────────────── */
[data-testid="stCaptionContainer"] { color: #6b7280 !important; }

/* ── SOURCE EXPANDER ────────────────────────────────────────────────────── */
.src-row {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 12px 0; border-bottom: 1px solid rgba(129,140,248,.1);
}
.src-row:last-child { border-bottom: none; }
.src-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; margin-top: 4px; }

/* ── EMPTY STATE ────────────────────────────────────────────────────────── */
.empty-state {
    text-align: center; padding: 32px 16px;
    color: #4b5563;
}
.empty-state .icon { font-size: 3rem; margin-bottom: 10px; opacity: .6; }
.empty-state .msg { font-size: .85rem; line-height: 1.6; }

/* ── STUDY PICKER ───────────────────────────────────────────────────────── */
.picker-wrap {
    background: rgba(26,24,48,.55);
    border: 1px solid rgba(129,140,248,.14);
    border-radius: 16px; padding: 20px 22px;
    margin-bottom: 20px;
    backdrop-filter: blur(8px);
}

/* ── STAT ROW ───────────────────────────────────────────────────────────── */
.stat-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
.stat-box {
    flex: 1; min-width: 90px;
    background: rgba(99,102,241,.1); border: 1px solid rgba(129,140,248,.2);
    border-radius: 12px; padding: 10px 14px; text-align: center;
}
.stat-box .n { font-size: 1.3rem; font-weight: 800; color: #a5b4fc; }
.stat-box .l { font-size: .68rem; color: #6b7280; font-weight: 600; text-transform: uppercase; letter-spacing: .5px; }

/* ── MIND MAP LABEL ─────────────────────────────────────────────────────── */
.map-wrap {
    background: rgba(26,24,48,.55);
    border: 1px solid rgba(129,140,248,.18);
    border-radius: 20px; padding: 20px;
    margin-top: 16px;
}

/* ── OLLAMA STATUS ──────────────────────────────────────────────────────── */
.status-pill {
    display: flex; align-items: center; gap: 7px;
    background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.25);
    border-radius: 999px; padding: 7px 14px; margin-top: 12px;
}
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: #10b981;
    box-shadow: 0 0 6px #10b981; flex-shrink: 0; }
.status-txt { font-size: .75rem; font-weight: 600; color: #6ee7b7; }

/* ── KEYPOINTS LIST ─────────────────────────────────────────────────────── */
.kp-item {
    display: flex; align-items: flex-start; gap: 10px;
    padding: 10px 0; border-bottom: 1px solid rgba(129,140,248,.08);
}
.kp-item:last-child { border-bottom: none; }
.kp-bullet {
    width: 6px; height: 6px; border-radius: 50%;
    background: #818cf8; flex-shrink: 0; margin-top: 7px;
}
.kp-txt { font-size: .9rem; color: #c4b5fd; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)


# ── Cached services ────────────────────────────────────────────────────────────
@st.cache_resource
def get_library() -> Library:
    return Library(cfg.library_dir)

@st.cache_resource
def get_store(em: str) -> VectorStore:
    return VectorStore(cfg.lancedb_dir, em)

@st.cache_resource
def get_translator() -> Translator:
    return Translator(cfg.nllb_ct2_dir, cfg.nllb_hf_dir)


# ── Hero Banner ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-glow"></div>
  <div class="hero-title">📖 <span>Vidya</span> Offline</div>
  <div class="hero-sub">
    Your 100% on-device AI tutor — works without internet.<br>
    Your books and questions <strong>never leave this computer.</strong>
  </div>
  <div class="hero-badges">
    <span class="hbadge">🔒 Fully Private</span>
    <span class="hbadge">⚡ Local AI</span>
    <span class="hbadge">🌐 Multi-language</span>
    <span class="hbadge">📚 RAG-powered</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── Ollama health check ────────────────────────────────────────────────────────
probe = LLM(cfg.chat_model, host=cfg.ollama_host)
if not probe.is_up():
    st.markdown("""
<div style="background:rgba(239,68,68,.12);border:1px solid rgba(239,68,68,.3);
            border-radius:16px;padding:24px 28px;">
  <div style="font-size:1.1rem;font-weight:700;color:#fca5a5;margin-bottom:8px;">
    🔴 Ollama is not running
  </div>
  <div style="color:#f87171;font-size:.9rem;line-height:1.7;">
    1. Install Ollama from <a href="https://ollama.com/download" style="color:#fbbf24;">ollama.com/download</a><br>
    2. Start it — open the Ollama app or run <code>ollama serve</code> in a terminal<br>
    3. Reload this page
  </div>
</div>
""", unsafe_allow_html=True)
    st.stop()

installed    = probe.list_models()
chat_models  = [m for m in installed if not is_embed_model(m)]
embed_models = [m for m in installed if is_embed_model(m)]

if not chat_models or not embed_models:
    missing = []
    if not chat_models:  missing.append(f"ollama pull {cfg.chat_model}")
    if not embed_models: missing.append(f"ollama pull {cfg.embed_model}")
    st.error("⚠️ Required models missing. Open a terminal and run:")
    st.code("\n".join(missing), language="bash")
    st.stop()

library = get_library()


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Settings ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-sec">⚙️ AI Settings</div>', unsafe_allow_html=True)

    chat_model = st.selectbox(
        "Chat model", chat_models,
        index=pick_default(chat_models, cfg.chat_model),
        help="Smaller = faster. llama3.2:1b for 4 GB RAM, llama3.2 for 8 GB.",
    )
    embed_model = st.selectbox(
        "Embedding model", embed_models,
        index=pick_default(embed_models, cfg.embed_model),
    )
    grade = st.selectbox(
        "Student level", CLASS_LEVELS, index=2,
        help="Answers are tailored to this education level.",
    )
    language = st.selectbox(
        "Language", list(LANGUAGES), index=0,
        help="Ask and read in this language.",
    )
    lang_code = LANGUAGES[language]

    with st.expander("🔧 Advanced"):
        top_k = st.slider("Passages per answer", 1, 8, cfg.top_k)
        max_distance = st.slider(
            "Match strictness", 0.2, 1.0, cfg.max_distance, 0.05,
            help="Lower = stricter book matching.",
        )
        num_ctx = st.select_slider(
            "Context window", [2048, 3072, 4096, 6144, 8192], value=cfg.num_ctx,
            help="2048 is fastest; 8192 remembers more text.",
        )
        ocr_on = st.checkbox("OCR scanned pages", value=cfg.ocr_enabled)

    translator = get_translator()
    if language != "English":
        st.info(f"🌐 {translator.describe()}", icon="🌐")

    # ── Library ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sidebar-sec">📚 My Library</div>', unsafe_allow_html=True)

    store    = get_store(embed_model)
    embedder = OllamaEmbedder(embed_model, host=cfg.ollama_host)
    llm      = LLM(chat_model, host=cfg.ollama_host, num_ctx=num_ctx, temperature=cfg.temperature)
    translator.llm = llm

    uploads = st.file_uploader(
        "Drop files here",
        type=sorted(e.lstrip(".") for e in SUPPORTED),
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    for kind, msg in st.session_state.pop("notes", []):
        getattr(st, kind)(msg)

    if uploads and st.button("⚡ Process & Index", type="primary", use_container_width=True):
        notes = st.session_state.setdefault("notes", [])
        for up in uploads:
            bar = st.progress(0.0, text=f"📄 {up.name}")
            try:
                run_cfg = dataclasses.replace(cfg, ocr_enabled=ocr_on)
                doc, warns, dup = ingest_file(
                    up.name, up.getvalue(), run_cfg, library, store, embedder,
                    progress=lambda f, msg, n=up.name: bar.progress(min(f, 1.0), text=f"📄 {n}: {msg}"),
                )
                bar.empty()
                if doc is None:
                    notes.append(("error", f"❌ {up.name}: " + " ".join(warns))); continue
                stat = "already in library" if dup else f"{doc.pages} pages · {doc.chunks} passages"
                notes.append(("success", f"✅ **{up.name}** – {stat}"))
                notes.extend(("warning", w) for w in warns)
            except (IngestError, EmbeddingError) as e:
                bar.empty(); notes.append(("error", f"❌ {up.name}: {e}"))
        st.rerun()

    docs  = library.list_docs()
    stale = [d for d in docs if slug(embed_model) not in d.indexed_with]
    if stale:
        st.warning(f"⚠️ {len(stale)} book(s) not indexed for **{embed_model}**")
        if st.button("🔄 Re-index", use_container_width=True):
            bar = st.progress(0.0, text="Re-indexing…")
            try:
                for i, d in enumerate(stale):
                    index_document(library, store, embedder, d.doc_id,
                                   progress=lambda f, m, i=i: bar.progress((i+f)/len(stale), text=m))
            except EmbeddingError as e:
                st.session_state.setdefault("notes", []).append(("error", str(e)))
            st.rerun()

    if not docs:
        st.markdown("""
<div class="empty-state">
  <div class="icon">📂</div>
  <div class="msg">No books yet.<br>Upload a PDF, Word, PowerPoint,<br>text file, or a photo.</div>
</div>""", unsafe_allow_html=True)
    else:
        for d in docs:
            c1, c2 = st.columns([7, 1])
            with c1:
                st.markdown(
                    f'<div class="book-row">'
                    f'<div class="book-ico">📗</div>'
                    f'<div><div class="book-name" title="{d.name}">{d.name}</div>'
                    f'<div class="book-meta">{d.pages} pg · {d.chunks} passages</div></div>'
                    f'</div>', unsafe_allow_html=True)
            with c2:
                if st.button("🗑", key=f"del_{d.doc_id}", help="Remove"):
                    store.delete_doc(d.doc_id); library.delete(d.doc_id); st.rerun()

    doc_names = {d.doc_id: d.name for d in docs}
    scope = (
        st.multiselect("Search in", list(doc_names), default=list(doc_names),
                       format_func=lambda i: doc_names[i],
                       placeholder="All books…")
        if docs else []
    )

    # ── Status pill ───────────────────────────────────────────────────────────
    model_short = chat_model.split(":")[0]
    st.markdown(
        f'<div class="status-pill"><div class="status-dot"></div>'
        f'<div class="status-txt">Ollama · {model_short}</div></div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════
tutor = Tutor(embedder, store)
state = st.session_state
state.setdefault("messages", [])


def tr_or_stop(fn, *a):
    try:
        return fn(*a)
    except (RuntimeError, LLMError) as e:
        st.error(f"Translation failed: {e}"); st.stop()


def to_english(text: str) -> str:
    if lang_code == ENGLISH or not looks_non_english(text):
        return text
    return tr_or_stop(translator.translate, text, lang_code, ENGLISH)


def sources_box(hits: list[dict]) -> None:
    if not hits:
        return
    with st.expander(f"📚 {len(hits)} source{'s' if len(hits) != 1 else ''} used", expanded=False):
        for h in hits:
            dist = h.get("_distance", 0)
            dot_color = ("#10b981" if dist < 0.4 else "#f59e0b" if dist < 0.6 else "#ef4444")
            st.markdown(
                f'<div class="src-row">'
                f'<div class="src-dot" style="background:{dot_color};box-shadow:0 0 6px {dot_color};"></div>'
                f'<div><strong style="color:#e0e7ff;">{h["source"]}</strong>'
                f'<span style="color:#6b7280;font-size:.78rem;"> · page {h["page"]}</span>'
                f' <span class="pill pill-indigo">dist {dist:.2f}</span><br>'
                f'<span style="color:#9ca3af;font-size:.8rem;">'
                f'{h["text"][:360]}{"…" if len(h["text"]) > 360 else ""}</span></div>'
                f'</div>', unsafe_allow_html=True)


def _picker_header(label: str, icon: str) -> None:
    st.markdown(
        f'<div class="sec-hdr">{icon} {label}<div class="sec-hdr-line"></div></div>',
        unsafe_allow_html=True,
    )


def study_picker(key: str) -> StudySource | None:
    if not docs:
        st.markdown('<div class="empty-state"><div class="icon">📂</div>'
                    '<div class="msg">Upload a book in the sidebar first.</div></div>',
                    unsafe_allow_html=True)
        return None
    with st.container():
        st.markdown('<div class="picker-wrap">', unsafe_allow_html=True)
        c1, c2 = st.columns([2, 3])
        doc_id = c1.selectbox("Book", list(doc_names), format_func=lambda i: doc_names[i], key=f"{key}_doc")
        doc = library.get(doc_id)
        mode_label = c2.radio(
            "Source", ["Whole book", "Page range", "Topic"],
            horizontal=True, key=f"{key}_mode",
        )
        st.markdown('</div>', unsafe_allow_html=True)
    src = StudySource(doc_id=doc_id)
    if mode_label == "Page range":
        a, b = st.columns(2)
        src.mode     = "pages"
        src.page_from = a.number_input("From page", 1, max(doc.pages, 1), 1, key=f"{key}_pf")
        src.page_to   = b.number_input("To page",   1, max(doc.pages, 1), min(doc.pages, 5), key=f"{key}_pt")
    elif mode_label == "Topic":
        src.mode  = "topic"
        src.topic = st.text_input("Topic / chapter", key=f"{key}_topic", placeholder="e.g. Photosynthesis")
    return src


def get_material(src: StudySource) -> str | None:
    if src.mode == "topic" and not src.topic.strip():
        st.warning("Please enter a topic first."); return None
    try:
        mat = gather_context(library, store, embedder, src,
                             max_chars=min(6000, int(num_ctx * 1.2)))
    except EmbeddingError as e:
        st.error(str(e)); return None
    if len(mat.strip()) < 80:
        st.warning("Not enough text found. Try a different page range or topic."); return None
    return mat


def new_gen(kind: str) -> None:
    state[kind + "_gen"] = state.get(kind + "_gen", 0) + 1


def localized(kind: str, items_en, translate_fn):
    if lang_code == ENGLISH:
        return items_en
    key = f"{kind}_{lang_code}_{state.get(kind + '_gen', 0)}"
    if key not in state:
        with st.spinner(f"🌐 Translating to {language}…"):
            state[key] = tr_or_stop(translate_fn, items_en, translator, lang_code)
    return state[key]


# ══════════════════════════════════════════════════════════════════════════════
#  TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_ask, tab_sum, tab_cards, tab_quiz, tab_map = st.tabs([
    "💬  Ask my Books",
    "📋  Summary",
    "🗂️  Flashcards",
    "📝  Quiz",
    "🧠  Mind Map",
])


# ════════════════════════════════════  TAB 1 — ASK  ══════════════════════════
with tab_ask:
    _picker_header("Chat with your books", "💬")

    if not docs:
        st.markdown("""
<div class="glass-card" style="text-align:center;padding:36px;">
  <div style="font-size:2.8rem;margin-bottom:12px;">💬</div>
  <div style="font-size:1.1rem;font-weight:700;color:#c4b5fd;margin-bottom:6px;">No books yet</div>
  <div style="color:#6b7280;font-size:.88rem;">
    Add a textbook in the sidebar and ask questions about it.<br>
    You can also chat freely without uploading a book.
  </div>
</div>""", unsafe_allow_html=True)

    for m in state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("notice"):
                st.caption("⚠️ " + m["notice"])
            sources_box(m.get("sources", []))

    if q := st.chat_input("Ask anything about your books…"):
        q_en       = to_english(q)
        history_en = [{"role": m["role"], "content": m["en"]} for m in state.messages]
        state.messages.append({"role": "user", "content": q, "en": q_en})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Searching your library…"):
                    plan = tutor.plan(q_en, history_en, grade, scope or None,
                                      top_k, max_distance, bool(docs))
                holder, answer = st.empty(), ""
                for piece in llm.chat_stream(plan.messages):
                    answer += piece
                    holder.markdown(answer + "▌")
                answer = answer.strip()
                holder.markdown(answer)
            except (LLMError, EmbeddingError) as e:
                st.error(str(e)); state.messages.pop(); st.stop()
            shown = answer
            if lang_code != ENGLISH and answer:
                with st.spinner(f"🌐 Translating to {language}…"):
                    shown = tr_or_stop(translator.translate, answer, ENGLISH, lang_code)
                holder.markdown(shown)
            if plan.notice:
                st.caption("⚠️ " + plan.notice)
            sources_box(plan.hits)
        state.messages.append({
            "role": "assistant", "content": shown, "en": answer,
            "sources": plan.hits, "notice": plan.notice,
        })

    if state.messages:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🧹 Clear conversation"):
            state.messages = []; st.rerun()


# ════════════════════════════════════  TAB 2 — SUMMARY  ══════════════════════
with tab_sum:
    _picker_header("Generate a Summary", "📋")
    src = study_picker("sum")

    if src:
        col_ai, col_fast = st.columns(2)
        gen_ai   = col_ai.button("✨ AI Summary",        type="primary", use_container_width=True)
        gen_fast = col_fast.button("⚡ Instant Key Points", use_container_width=True)

        if gen_ai:
            mat = get_material(src)
            if mat:
                holder, text = st.empty(), ""
                try:
                    for piece in llm.chat_stream(summary_messages(mat, grade), temperature=0.2):
                        text += piece; holder.markdown(text + "▌")
                    state["summary_en"] = text.strip()
                    state["summary_tr"] = {}
                    holder.empty()
                except LLMError as e:
                    st.error(str(e))

        if gen_fast:
            mat = get_material(src)
            if mat:
                state["keypoints"] = top_sentences(mat, max_sentences=8)

    if state.get("keypoints"):
        _picker_header("Key Points", "🔑")
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.caption("Extracted by TextRank — instant, works even on slow hardware")
        for s in state["keypoints"]:
            st.markdown(
                f'<div class="kp-item"><div class="kp-bullet"></div>'
                f'<div class="kp-txt">{s}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    if state.get("summary_en"):
        shown = state["summary_en"]
        if lang_code != ENGLISH:
            cache = state.setdefault("summary_tr", {})
            if lang_code not in cache:
                with st.spinner(f"🌐 Translating to {language}…"):
                    cache[lang_code] = tr_or_stop(
                        translator.translate_markdown, state["summary_en"], lang_code)
            shown = cache[lang_code]
        _picker_header("AI Summary", "📄")
        with st.container():
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(shown)
            st.markdown('</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download (.md)", shown, file_name="summary.md", mime="text/markdown")


# ════════════════════════════════════  TAB 3 — FLASHCARDS  ═══════════════════
with tab_cards:
    _picker_header("Flashcards", "🗂️")
    src = study_picker("fc")

    col_n, col_btn = st.columns([3, 2])
    n_cards = col_n.slider("Cards to generate", 4, 15, 8, key="fc_n")
    if src and col_btn.button("🗂️ Generate Cards", type="primary", use_container_width=True):
        mat = get_material(src)
        if mat:
            with st.spinner("✍️ Writing flashcards…"):
                try:
                    state["cards_en"]  = make_flashcards(llm, mat, grade, n_cards)
                    state["card_i"]    = 0
                    state["card_flip"] = False
                    new_gen("cards")
                except (LLMError, ValueError) as e:
                    st.error(str(e))

    cards_en = state.get("cards_en")
    if cards_en:
        cards = localized("cards", cards_en, lambda it, tr, tg: translate_cards(it, tr, tg))
        i     = state.get("card_i", 0) % len(cards)
        flip  = state.get("card_flip", False)
        card  = cards[i]

        # Dot navigation row
        dots = "".join(
            f'<div class="fc-dot{"  active" if j == i else ""}"></div>'
            for j in range(len(cards))
        )
        st.markdown(f'<div class="fc-dot-row">{dots}</div>', unsafe_allow_html=True)

        # Card face
        label = "ANSWER" if flip else "QUESTION"
        icon  = "💡" if flip else "❓"
        text  = card.back if flip else card.front
        st.markdown(f"""
<div class="flashcard-wrap">
  <div class="flashcard">
    <div class="fc-side-label">{icon} {label}</div>
    <div class="fc-text">{text}</div>
    <div class="fc-counter">{i+1} / {len(cards)}</div>
  </div>
</div>""", unsafe_allow_html=True)

        # Nav buttons
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("⬅️ Prev",   use_container_width=True):
            state["card_i"] = (i - 1) % len(cards); state["card_flip"] = False; st.rerun()
        if b2.button("🔄 Flip",   use_container_width=True, type="primary"):
            state["card_flip"] = not flip; st.rerun()
        if b3.button("Next ➡️",   use_container_width=True):
            state["card_i"] = (i + 1) % len(cards); state["card_flip"] = False; st.rerun()
        b4.download_button("⬇️ CSV", flashcards_to_csv(cards),
                           file_name="flashcards.csv", mime="text/csv",
                           use_container_width=True)


# ════════════════════════════════════  TAB 4 — QUIZ  ═════════════════════════
with tab_quiz:
    _picker_header("Mock Quiz", "📝")
    src = study_picker("qz")

    col_n, col_btn = st.columns([3, 2])
    n_q = col_n.slider("Questions", 3, 10, 5, key="qz_n")
    if src and col_btn.button("📝 Create Quiz", type="primary", use_container_width=True):
        mat = get_material(src)
        if mat:
            with st.spinner("✍️ Writing questions…"):
                try:
                    state["quiz_en"]   = make_quiz(llm, mat, grade, n_q)
                    state["quiz_done"] = False
                    new_gen("quiz")
                    for k in [k for k in state if str(k).startswith("qa_")]:
                        del state[k]
                except (LLMError, ValueError) as e:
                    st.error(str(e))

    quiz_en = state.get("quiz_en")
    if quiz_en:
        quiz  = localized("quiz", quiz_en, lambda it, tr, tg: translate_quiz(it, tr, tg))
        done  = state.get("quiz_done", False)
        score = 0

        for qi, q in enumerate(quiz):
            st.markdown(
                f'<div class="quiz-card">'
                f'<div style="display:flex;align-items:center;margin-bottom:12px;">'
                f'<div class="quiz-num">{qi+1}</div>'
                f'<div class="quiz-q-text">{q.question}</div>'
                f'</div>', unsafe_allow_html=True)
            choice = st.radio(
                f"q_{qi}", q.options, index=None,
                key=f"qa_{qi}", disabled=done,
                label_visibility="collapsed",
            )
            if done:
                right = q.options[q.answer_index]
                if choice == right:
                    score += 1
                    st.success("✅ Correct!")
                else:
                    st.error(f"❌ Correct answer: **{right}**")
                if q.explanation:
                    st.info(f"💡 {q.explanation}", icon="💡")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        if not done:
            if st.button("✅ Submit & Check", type="primary"):
                state["quiz_done"] = True; st.rerun()
        else:
            pct  = int(score / len(quiz) * 100)
            emoji = "🏆" if pct >= 80 else "🌟" if pct >= 60 else "👍" if pct >= 40 else "📚"
            msg   = ("Outstanding!" if pct >= 80 else "Great work!" if pct >= 60
                     else "Keep going!" if pct >= 40 else "Keep studying!")
            col_sc, col_act = st.columns([1, 2])
            with col_sc:
                st.markdown(f"""
<div class="score-wrap">
  <span class="score-emoji">{emoji}</span>
  <div><span class="score-num">{score}</span>
       <span class="score-denom"> / {len(quiz)}</span></div>
  <div class="score-pct">{pct}% correct</div>
  <div class="score-msg">{msg}</div>
</div>""", unsafe_allow_html=True)
            with col_act:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🔁 Try Again", use_container_width=True):
                    state["quiz_done"] = False
                    for k in [k for k in state if str(k).startswith("qa_")]:
                        del state[k]
                    st.rerun()


# ════════════════════════════════════  TAB 5 — MIND MAP  ═════════════════════
with tab_map:
    _picker_header("Visual Mind Map", "🧠")
    src = study_picker("mm")

    if src and st.button("🧠 Generate Mind Map", type="primary"):
        mat = get_material(src)
        if mat:
            with st.spinner("🤔 Building your mind map…"):
                try:
                    state["map_en"] = make_mindmap(llm, mat, grade)
                    new_gen("map")
                except (LLMError, ValueError) as e:
                    st.error(str(e))

    if state.get("map_en"):
        mm  = localized("map", state["map_en"], lambda it, tr, tg: translate_mindmap(it, tr, tg))
        dot = mindmap_to_dot(mm)
        st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
        st.graphviz_chart(dot, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Download graph (.dot)", dot, file_name="mindmap.dot")
