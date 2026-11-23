"""Vidya Offline – improved UI/UX dashboard (Streamlit).

Run with:  streamlit run app.py     (or double-click run.bat on Windows)
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
from vidya.study import (StudySource, flashcards_to_csv, gather_context, make_flashcards, make_mindmap,
                         make_quiz, mindmap_to_dot, summary_messages, translate_cards, translate_mindmap,
                         translate_quiz)
from vidya.textrank import top_sentences
from vidya.translate import ENGLISH, LANGUAGES, Translator, looks_non_english

cfg = load_config()

# ─── Page config (must be first Streamlit call) ───────────────────────────────
st.set_page_config(
    page_title="Vidya Offline",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Global CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---- imports ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ---- base ---- */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ---- hero header ---- */
.vidya-hero {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a78bfa 100%);
    border-radius: 20px;
    padding: 32px 36px 24px;
    margin-bottom: 24px;
    color: #fff;
    box-shadow: 0 8px 32px rgba(99,102,241,.25);
    position: relative;
    overflow: hidden;
}
.vidya-hero::after {
    content: "📖";
    font-size: 120px;
    position: absolute;
    right: 32px; top: -8px;
    opacity: .15;
    line-height: 1;
}
.vidya-hero h1 { margin: 0 0 4px; font-size: 2.2rem; font-weight: 700; letter-spacing: -0.5px; }
.vidya-hero p  { margin: 0; font-size: .95rem; opacity: .88; }

/* ---- pill badges ---- */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: .4px;
    text-transform: uppercase;
}
.badge-green  { background:#d1fae5; color:#065f46; }
.badge-purple { background:#ede9fe; color:#5b21b6; }
.badge-blue   { background:#dbeafe; color:#1e40af; }
.badge-amber  { background:#fef3c7; color:#92400e; }

/* ---- book cards ---- */
.book-card {
    background: #fff;
    border: 1.5px solid #e5e7eb;
    border-radius: 14px;
    padding: 14px 16px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
    box-shadow: 0 1px 4px rgba(0,0,0,.04);
    transition: box-shadow .15s;
}
.book-card:hover { box-shadow: 0 4px 16px rgba(99,102,241,.12); }
.book-icon { font-size: 2rem; flex-shrink: 0; }
.book-info { flex: 1; min-width: 0; }
.book-name { font-weight: 600; font-size: .93rem; color: #1e1b4b; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.book-meta { font-size: .78rem; color: #6b7280; margin-top: 2px; }

/* ---- tabs custom ---- */
[data-testid="stTabs"] [role="tab"] {
    font-size: 0.88rem;
    font-weight: 500;
    padding: 8px 18px;
    border-radius: 10px 10px 0 0;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: #ede9fe;
    color: #5b21b6;
    font-weight: 700;
}

/* ---- flashcard ---- */
.flashcard {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 100%);
    border: 2px solid #c4b5fd;
    border-radius: 20px;
    padding: 40px 32px;
    text-align: center;
    min-height: 200px;
    display: flex;
    flex-direction: column;
    justify-content: center;
    box-shadow: 0 4px 20px rgba(99,102,241,.12);
}
.flashcard-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #7c3aed;
    margin-bottom: 16px;
}
.flashcard-text {
    font-size: 1.35rem;
    font-weight: 600;
    color: #1e1b4b;
    line-height: 1.5;
}
.flashcard-nav { display: flex; justify-content: center; gap: 12px; margin-top: 20px; }

/* ---- quiz item ---- */
.quiz-q {
    background: #fff;
    border: 1.5px solid #e5e7eb;
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 14px;
}

/* ---- source chip ---- */
.source-chip {
    display: inline-block;
    background: #f3f4f6;
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    padding: 4px 10px;
    font-size: 0.78rem;
    color: #374151;
    margin: 2px 4px 2px 0;
}

/* ---- loading shimmer ---- */
@keyframes shimmer { 0%{background-position:-400px 0} 100%{background-position:400px 0} }
.shimmer {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 800px 100%;
    animation: shimmer 1.4s infinite;
    border-radius: 8px;
    height: 20px;
    margin: 6px 0;
}

/* ---- sidebar cleanup ---- */
[data-testid="stSidebar"] { background: #faf9ff; border-right: 1px solid #e5e7eb; }
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 { font-size: 1rem !important; color: #5b21b6; }

/* ---- section headers ---- */
.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1e1b4b;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* ---- progress bar colour ---- */
[data-testid="stProgress"] > div > div { background: linear-gradient(90deg, #6366f1, #8b5cf6); }

/* ---- scrollable chat ---- */
.chat-wrap { max-height: 520px; overflow-y: auto; padding-right: 6px; }

/* ---- subtle divider ---- */
.vdivider { height: 1px; background: #e5e7eb; margin: 16px 0; }

/* ---- score card ---- */
.score-card {
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    border-radius: 16px;
    padding: 24px;
    text-align: center;
    color: #fff;
    box-shadow: 0 4px 20px rgba(99,102,241,.3);
}
.score-card .num { font-size: 3rem; font-weight: 800; line-height: 1; }
.score-card .lbl { font-size: .85rem; opacity: .85; margin-top: 4px; }

/* ---- upload area ---- */
[data-testid="stFileUploadDropzone"] {
    border: 2px dashed #c4b5fd !important;
    border-radius: 14px !important;
    background: #faf9ff !important;
}

/* ---- primary buttons ---- */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    border: none !important;
    color: #fff !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 10px rgba(99,102,241,.3) !important;
    transition: opacity .15s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover { opacity: .9 !important; }

/* ---- secondary buttons ---- */
[data-testid="stButton"] > button:not([kind="primary"]) {
    border-radius: 10px !important;
    font-weight: 500 !important;
    border: 1.5px solid #d1d5db !important;
}

/* hide default streamlit watermark */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ─── Cached services ──────────────────────────────────────────────────────────
@st.cache_resource
def get_library() -> Library:
    return Library(cfg.library_dir)


@st.cache_resource
def get_store(embed_model: str) -> VectorStore:
    return VectorStore(cfg.lancedb_dir, embed_model)


@st.cache_resource
def get_translator() -> Translator:
    return Translator(cfg.nllb_ct2_dir, cfg.nllb_hf_dir)


# ─── Hero Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="vidya-hero">
  <h1>Vidya Offline</h1>
  <p>100% on-device AI tutor &nbsp;·&nbsp; works without internet &nbsp;·&nbsp;
     your books and questions never leave this computer</p>
</div>
""", unsafe_allow_html=True)


# ─── Ollama / model check ─────────────────────────────────────────────────────
probe = LLM(cfg.chat_model, host=cfg.ollama_host)
if not probe.is_up():
    st.error("🔴 **Ollama is not running.** Start it, then reload this page.")
    with st.expander("How to fix this", expanded=True):
        st.markdown("""
1. **Install Ollama** from [ollama.com/download](https://ollama.com/download) *(one-time, needs internet)*
2. **Start it** – open the Ollama app, or run `ollama serve` in a terminal
3. **Reload** this page
        """)
    st.stop()

installed   = probe.list_models()
chat_models  = [m for m in installed if not is_embed_model(m)]
embed_models = [m for m in installed if is_embed_model(m)]

if not chat_models or not embed_models:
    missing = []
    if not chat_models:
        missing.append(f"ollama pull {cfg.chat_model}")
    if not embed_models:
        missing.append(f"ollama pull {cfg.embed_model}")
    st.error("⚠️ **Required models are missing.** Open a terminal and run:")
    st.code("\n".join(missing), language="bash")
    st.stop()

library = get_library()


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Settings section
    st.markdown('<div class="section-header">⚙️ Settings</div>', unsafe_allow_html=True)

    chat_model = st.selectbox(
        "Chat model", chat_models,
        index=pick_default(chat_models, cfg.chat_model),
        help="Smaller = faster. llama3.2:1b suits 4 GB RAM; llama3.2 (3B) suits 8 GB."
    )
    embed_model = st.selectbox(
        "Embedding model", embed_models,
        index=pick_default(embed_models, cfg.embed_model)
    )
    grade = st.selectbox(
        "Student level", CLASS_LEVELS, index=2,
        help="Answers, summaries and quizzes are written at this level."
    )
    language = st.selectbox(
        "Language", list(LANGUAGES), index=0,
        help="Ask questions and read answers in this language."
    )
    lang_code = LANGUAGES[language]

    with st.expander("🔧 Advanced settings"):
        top_k = st.slider("Passages used per answer", 1, 8, cfg.top_k)
        max_distance = st.slider(
            "Book-match strictness", 0.2, 1.0, cfg.max_distance, 0.05,
            help="Lower = stricter. Raise if real questions get 'not found'."
        )
        num_ctx = st.select_slider(
            "Context window (RAM ↔ memory)",
            [2048, 3072, 4096, 6144, 8192], value=cfg.num_ctx
        )
        ocr_on = st.checkbox("OCR scanned pages / images", value=cfg.ocr_enabled)

    translator = get_translator()
    if language != "English":
        backend = translator.describe()
        st.info(f"🌐 **Translation active**\n\n{backend}", icon="🌐")

    st.markdown('<div class="vdivider"></div>', unsafe_allow_html=True)

    # Library section
    st.markdown('<div class="section-header">📚 My Library</div>', unsafe_allow_html=True)

    store    = get_store(embed_model)
    embedder = OllamaEmbedder(embed_model, host=cfg.ollama_host)
    llm      = LLM(chat_model, host=cfg.ollama_host, num_ctx=num_ctx, temperature=cfg.temperature)
    translator.llm = llm  # LLM fallback for translation

    uploads = st.file_uploader(
        "Upload textbooks / notes",
        type=sorted(e.lstrip(".") for e in SUPPORTED),
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    for kind, msg in st.session_state.pop("notes", []):
        getattr(st, kind)(msg)

    if uploads and st.button("⚡ Process & Index", type="primary", use_container_width=True):
        notes = st.session_state.setdefault("notes", [])
        for up in uploads:
            bar = st.progress(0.0, text=f"📄 {up.name}: starting…")
            try:
                run_cfg = dataclasses.replace(cfg, ocr_enabled=ocr_on)
                doc, warns, dup = ingest_file(
                    up.name, up.getvalue(), run_cfg, library, store, embedder,
                    progress=lambda f, msg, n=up.name: bar.progress(min(f, 1.0), text=f"📄 {n}: {msg}")
                )
                bar.empty()
                if doc is None:
                    notes.append(("error", f"❌ {up.name}: " + " ".join(warns)))
                    continue
                status = "already in your library." if dup else f"{doc.pages} pages · {doc.chunks} passages indexed."
                notes.append(("success", f"✅ **{up.name}** – {status}"))
                notes.extend(("warning", w) for w in warns)
            except (IngestError, EmbeddingError) as e:
                bar.empty()
                notes.append(("error", f"❌ {up.name}: {e}"))
        st.rerun()

    # Book list
    docs = library.list_docs()
    stale = [d for d in docs if slug(embed_model) not in d.indexed_with]
    if stale:
        st.warning(f"⚠️ {len(stale)} book(s) not indexed for **{embed_model}**.")
        if st.button("🔄 Re-index", use_container_width=True):
            bar = st.progress(0.0, text="Re-indexing…")
            try:
                for i, d in enumerate(stale):
                    index_document(library, store, embedder, d.doc_id,
                                   progress=lambda f, m, i=i: bar.progress((i + f) / len(stale), text=m))
            except EmbeddingError as e:
                st.session_state.setdefault("notes", []).append(("error", str(e)))
            st.rerun()

    if not docs:
        st.markdown("""
<div style="text-align:center;padding:20px 0;color:#9ca3af;">
  <div style="font-size:2.5rem;">📂</div>
  <div style="font-size:.85rem;margin-top:8px;">No books yet.<br>Upload a PDF, Word, PowerPoint,<br>text file or a photo.</div>
</div>
""", unsafe_allow_html=True)
    else:
        for d in docs:
            col_a, col_b = st.columns([6, 1])
            with col_a:
                st.markdown(
                    f'<div class="book-card">'
                    f'<div class="book-icon">📗</div>'
                    f'<div class="book-info">'
                    f'<div class="book-name" title="{d.name}">{d.name}</div>'
                    f'<div class="book-meta">{d.pages} pages · {d.chunks} passages</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )
            with col_b:
                if st.button("🗑", key=f"del_{d.doc_id}", help="Remove this book"):
                    store.delete_doc(d.doc_id)
                    library.delete(d.doc_id)
                    st.rerun()

    doc_names = {d.doc_id: d.name for d in docs}
    scope = (
        st.multiselect(
            "🔍 Search in", list(doc_names), default=list(doc_names),
            format_func=lambda i: doc_names[i],
            placeholder="Select books to search…",
        )
        if docs else []
    )

    # Status footer
    st.markdown('<div class="vdivider"></div>', unsafe_allow_html=True)
    model_short = chat_model.split(":")[0]
    st.markdown(
        f'<div style="font-size:.75rem;color:#9ca3af;text-align:center;">'
        f'🟢 Ollama connected &nbsp;·&nbsp; <b>{model_short}</b></div>',
        unsafe_allow_html=True,
    )


# ─── Helpers ──────────────────────────────────────────────────────────────────
tutor = Tutor(embedder, store)
state = st.session_state
state.setdefault("messages", [])


def tr_or_stop(fn, *a):
    """Run a translation, turning a missing engine into a friendly message."""
    try:
        return fn(*a)
    except (RuntimeError, LLMError) as e:
        st.error(f"Translation failed: {e}")
        st.stop()


def to_english(text: str) -> str:
    if lang_code == ENGLISH or not looks_non_english(text):
        return text
    return tr_or_stop(translator.translate, text, lang_code, ENGLISH)


def sources_box(hits: list[dict]) -> None:
    if not hits:
        return
    with st.expander(f"📚 Sources ({len(hits)})", expanded=False):
        for h in hits:
            dist = h.get("_distance", 0)
            quality = "🟢" if dist < 0.4 else "🟡" if dist < 0.6 else "🔴"
            st.markdown(
                f"{quality} **{h['source']}** · page {h['page']} "
                f'<span class="badge badge-purple">dist {dist:.2f}</span>',
                unsafe_allow_html=True,
            )
            st.caption(h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))
            st.markdown('<div class="vdivider"></div>', unsafe_allow_html=True)


def study_picker(key: str) -> StudySource | None:
    if not docs:
        st.info("👈 Upload a book in the sidebar first.", icon="📂")
        return None
    c1, c2 = st.columns([2, 3])
    doc_id = c1.selectbox("Book", list(doc_names), format_func=lambda i: doc_names[i], key=f"{key}_doc")
    doc = library.get(doc_id)
    mode_label = c2.radio(
        "Material", ["Whole book (key sentences)", "Page range", "Topic"],
        horizontal=True, key=f"{key}_mode"
    )
    src = StudySource(doc_id=doc_id)
    if mode_label == "Page range":
        a, b = st.columns(2)
        src.mode = "pages"
        src.page_from = a.number_input("From page", 1, max(doc.pages, 1), 1, key=f"{key}_pf")
        src.page_to   = b.number_input("To page", 1, max(doc.pages, 1), min(doc.pages, 5), key=f"{key}_pt")
    elif mode_label == "Topic":
        src.mode  = "topic"
        src.topic = st.text_input("Topic or chapter name", key=f"{key}_topic", placeholder="e.g. Photosynthesis")
    return src


def get_material(src: StudySource) -> str | None:
    if src.mode == "topic" and not src.topic.strip():
        st.warning("Please type a topic first.")
        return None
    try:
        material = gather_context(library, store, embedder, src, max_chars=min(6000, int(num_ctx * 1.2)))
    except EmbeddingError as e:
        st.error(str(e))
        return None
    if len(material.strip()) < 80:
        st.warning("Not enough text found for that selection. Try a different page range or topic.")
        return None
    return material


def new_gen(kind: str) -> None:
    state[kind + "_gen"] = state.get(kind + "_gen", 0) + 1


def localized(kind: str, items_en, translate_fn):
    """Translate generated study items once per language and cache them for this session."""
    if lang_code == ENGLISH:
        return items_en
    key = f"{kind}_{lang_code}_{state.get(kind + '_gen', 0)}"
    if key not in state:
        with st.spinner(f"Translating to {language}…"):
            state[key] = tr_or_stop(translate_fn, items_en, translator, lang_code)
    return state[key]


# ─── Tabs ─────────────────────────────────────────────────────────────────────
tab_ask, tab_sum, tab_cards, tab_quiz, tab_map = st.tabs([
    "💬 Ask my books",
    "📋 Summary",
    "🗂️ Flashcards",
    "📝 Mock Quiz",
    "🧠 Mind Map",
])


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Ask
# ══════════════════════════════════════════════════════════════════════════════
with tab_ask:
    if not docs:
        col_tip, _ = st.columns([2, 1])
        with col_tip:
            st.info(
                "**No books uploaded yet.**\n\n"
                "Add a textbook in the sidebar and start asking questions about it. "
                "You can also chat without a book.",
                icon="📂",
            )

    # Render conversation history
    for m in state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("notice"):
                st.caption("⚠️ " + m["notice"])
            sources_box(m.get("sources", []))

    # Input
    if q := st.chat_input("Ask a question about your books…"):
        q_en        = to_english(q)
        history_en  = [{"role": m["role"], "content": m["en"]} for m in state.messages]
        state.messages.append({"role": "user", "content": q, "en": q_en})

        with st.chat_message("user"):
            st.markdown(q)

        with st.chat_message("assistant"):
            try:
                with st.spinner("🔍 Searching your books…"):
                    plan = tutor.plan(q_en, history_en, grade, scope or None, top_k, max_distance, bool(docs))
                holder, answer = st.empty(), ""
                for piece in llm.chat_stream(plan.messages):
                    answer += piece
                    holder.markdown(answer + "▌")
                answer = answer.strip()
                holder.markdown(answer)
            except (LLMError, EmbeddingError) as e:
                st.error(str(e))
                state.messages.pop()
                st.stop()

            shown = answer
            if lang_code != ENGLISH and answer:
                with st.spinner(f"🌐 Translating to {language}…"):
                    shown = tr_or_stop(translator.translate, answer, ENGLISH, lang_code)
                holder.markdown(shown)

            if plan.notice:
                st.caption("⚠️ " + plan.notice)
            sources_box(plan.hits)

        state.messages.append({
            "role": "assistant",
            "content": shown,
            "en": answer,
            "sources": plan.hits,
            "notice": plan.notice,
        })

    # Clear button
    if state.messages:
        if st.button("🧹 Clear conversation", use_container_width=False):
            state.messages = []
            st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Summary
# ══════════════════════════════════════════════════════════════════════════════
with tab_sum:
    st.markdown('<div class="section-header">📋 Chapter Summary</div>', unsafe_allow_html=True)
    src = study_picker("sum")
    if src:
        col_ai, col_fast = st.columns(2)
        with col_ai:
            gen_ai = st.button("✨ Generate AI Summary", type="primary", use_container_width=True)
        with col_fast:
            gen_fast = st.button("⚡ Instant Key Points (no AI)", use_container_width=True)

        if gen_ai:
            material = get_material(src)
            if material:
                holder, text = st.empty(), ""
                try:
                    for piece in llm.chat_stream(summary_messages(material, grade), temperature=0.2):
                        text += piece
                        holder.markdown(text + "▌")
                    state["summary_en"] = text.strip()
                    state["summary_tr"] = {}
                    holder.empty()
                except LLMError as e:
                    st.error(str(e))

        if gen_fast:
            material = get_material(src)
            if material:
                state["keypoints"] = top_sentences(material, max_sentences=8)

    # Key points display
    if state.get("keypoints"):
        st.markdown('<div class="section-header">🔑 Key Sentences</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.caption("Extracted by TextRank — instant, works even on slow PCs")
            for s in state["keypoints"]:
                st.markdown(f"- {s}")

    # AI summary display
    if state.get("summary_en"):
        shown = state["summary_en"]
        if lang_code != ENGLISH:
            cache = state.setdefault("summary_tr", {})
            if lang_code not in cache:
                with st.spinner(f"🌐 Translating to {language}…"):
                    cache[lang_code] = tr_or_stop(translator.translate_markdown, state["summary_en"], lang_code)
            shown = cache[lang_code]
        st.markdown('<div class="section-header">📄 Summary</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(shown)
        st.download_button("⬇️ Download (.md)", shown, file_name="summary.md", mime="text/markdown")


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 – Flashcards
# ══════════════════════════════════════════════════════════════════════════════
with tab_cards:
    st.markdown('<div class="section-header">🗂️ Flashcards</div>', unsafe_allow_html=True)
    src = study_picker("fc")

    col_n, col_btn = st.columns([2, 1])
    n_cards = col_n.slider("Number of cards", 4, 15, 8, key="fc_n")

    if src and col_btn.button("🗂️ Make flashcards", type="primary", use_container_width=True):
        material = get_material(src)
        if material:
            with st.spinner("✍️ Writing flashcards…"):
                try:
                    state["cards_en"] = make_flashcards(llm, material, grade, n_cards)
                    state["card_i"]   = 0
                    state["card_flip"] = False
                    new_gen("cards")
                except (LLMError, ValueError) as e:
                    st.error(str(e))

    cards_en = state.get("cards_en")
    if cards_en:
        cards = localized("cards", cards_en, lambda items, tr, tgt: translate_cards(items, tr, tgt))
        i     = state.get("card_i", 0) % len(cards)
        flip  = state.get("card_flip", False)
        card  = cards[i]

        # Progress bar
        progress_pct = (i + 1) / len(cards)
        st.progress(progress_pct, text=f"Card {i + 1} of {len(cards)}")

        # Card face
        label = "ANSWER" if flip else "QUESTION"
        text  = card.back if flip else card.front
        st.markdown(
            f'<div class="flashcard">'
            f'<div class="flashcard-label">{"💡 " + label}</div>'
            f'<div class="flashcard-text">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)

        # Navigation
        b1, b2, b3, b4 = st.columns(4)
        if b1.button("⬅️ Prev", use_container_width=True):
            state["card_i"]    = (i - 1) % len(cards)
            state["card_flip"] = False
            st.rerun()
        if b2.button("🔄 Flip", use_container_width=True, type="primary"):
            state["card_flip"] = not flip
            st.rerun()
        if b3.button("Next ➡️", use_container_width=True):
            state["card_i"]    = (i + 1) % len(cards)
            state["card_flip"] = False
            st.rerun()
        b4.download_button(
            "⬇️ CSV", flashcards_to_csv(cards),
            file_name="flashcards.csv", mime="text/csv",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 – Quiz
# ══════════════════════════════════════════════════════════════════════════════
with tab_quiz:
    st.markdown('<div class="section-header">📝 Mock Quiz</div>', unsafe_allow_html=True)
    src = study_picker("qz")

    col_n, col_btn = st.columns([2, 1])
    n_q = col_n.slider("Number of questions", 3, 10, 5, key="qz_n")

    if src and col_btn.button("📝 Create Quiz", type="primary", use_container_width=True):
        material = get_material(src)
        if material:
            with st.spinner("✍️ Writing questions…"):
                try:
                    state["quiz_en"]   = make_quiz(llm, material, grade, n_q)
                    state["quiz_done"] = False
                    new_gen("quiz")
                    for k in [k for k in state if str(k).startswith("qa_")]:
                        del state[k]
                except (LLMError, ValueError) as e:
                    st.error(str(e))

    quiz_en = state.get("quiz_en")
    if quiz_en:
        quiz  = localized("quiz", quiz_en, lambda items, tr, tgt: translate_quiz(items, tr, tgt))
        done  = state.get("quiz_done", False)
        score = 0

        for qi, q in enumerate(quiz):
            with st.container(border=True):
                choice = st.radio(
                    f"**{qi + 1}. {q.question}**",
                    q.options, index=None, key=f"qa_{qi}", disabled=done
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

        st.markdown("<br>", unsafe_allow_html=True)

        if not done:
            if st.button("✅ Check Answers", type="primary"):
                state["quiz_done"] = True
                st.rerun()
        else:
            pct = int(score / len(quiz) * 100)
            emoji = "🏆" if pct >= 80 else "👍" if pct >= 50 else "📚"
            col_sc, col_btn2 = st.columns([1, 2])
            with col_sc:
                st.markdown(
                    f'<div class="score-card">'
                    f'<div style="font-size:2rem;">{emoji}</div>'
                    f'<div class="num">{score}/{len(quiz)}</div>'
                    f'<div class="lbl">{pct}% correct</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with col_btn2:
                st.markdown("<br><br>", unsafe_allow_html=True)
                if st.button("🔁 Try Again", use_container_width=True):
                    state["quiz_done"] = False
                    for k in [k for k in state if str(k).startswith("qa_")]:
                        del state[k]
                    st.rerun()


# ══════════════════════════════════════════════════════════════════════════════
#  TAB 5 – Mind Map
# ══════════════════════════════════════════════════════════════════════════════
with tab_map:
    st.markdown('<div class="section-header">🧠 Visual Mind Map</div>', unsafe_allow_html=True)
    src = study_picker("mm")
    if src and st.button("🧠 Draw Mind Map", type="primary"):
        material = get_material(src)
        if material:
            with st.spinner("🤔 Generating mind map…"):
                try:
                    state["map_en"] = make_mindmap(llm, material, grade)
                    new_gen("map")
                except (LLMError, ValueError) as e:
                    st.error(str(e))

    if state.get("map_en"):
        mm  = localized("map", state["map_en"], lambda item, tr, tgt: translate_mindmap(item, tr, tgt))
        dot = mindmap_to_dot(mm)
        st.graphviz_chart(dot, use_container_width=True)
        st.download_button("⬇️ Download graph (.dot)", dot, file_name="mindmap.dot")
