"""Vidya Offline - Stage 5: the all-in-one dashboard (Streamlit).

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
st.set_page_config(page_title="Vidya Offline", page_icon="📖", layout="wide")


# ----------------------------------------------------------------------------- cached services
@st.cache_resource
def get_library() -> Library:
    return Library(cfg.library_dir)


@st.cache_resource
def get_store(embed_model: str) -> VectorStore:
    return VectorStore(cfg.lancedb_dir, embed_model)


@st.cache_resource
def get_translator() -> Translator:
    return Translator(cfg.nllb_ct2_dir, cfg.nllb_hf_dir)


# ----------------------------------------------------------------------------- header
st.title("📖 Vidya Offline")
st.caption("100% on-device AI tutor · works without internet · your books and questions never leave this computer")

# ----------------------------------------------------------------------------- Ollama check
probe = LLM(cfg.chat_model, host=cfg.ollama_host)
if not probe.is_up():
    st.error("**Ollama is not running.**")
    st.markdown("1. Install Ollama from https://ollama.com/download (one-time, needs internet).\n"
                "2. Start it (open the *Ollama* app, or run `ollama serve` in a terminal).\n"
                "3. Reload this page.")
    st.stop()

installed = probe.list_models()
chat_models = [m for m in installed if not is_embed_model(m)]
embed_models = [m for m in installed if is_embed_model(m)]
if not chat_models or not embed_models:
    st.error("Required models are missing.")
    st.markdown("Open a terminal and run (needs internet once):")
    cmds = []
    if not chat_models:
        cmds.append(f"ollama pull {cfg.chat_model}")
    if not embed_models:
        cmds.append(f"ollama pull {cfg.embed_model}")
    st.code("\n".join(cmds), language="bash")
    st.stop()

library = get_library()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    chat_model = st.selectbox("Chat model", chat_models, index=pick_default(chat_models, cfg.chat_model),
                              help="Smaller = faster and lighter. llama3.2:1b suits 4 GB PCs; llama3.2 (3B) suits 8 GB.")
    embed_model = st.selectbox("Embedding model", embed_models,
                               index=pick_default(embed_models, cfg.embed_model))
    grade = st.selectbox("Student level", CLASS_LEVELS, index=2,
                         help="Answers, summaries and quizzes are written at this level.")
    language = st.selectbox("Language", list(LANGUAGES), index=0,
                            help="Ask in this language and read answers in this language.")
    lang_code = LANGUAGES[language]

    with st.expander("Advanced"):
        top_k = st.slider("Passages used per answer", 1, 8, cfg.top_k)
        max_distance = st.slider("Book-match strictness", 0.2, 1.0, cfg.max_distance, 0.05,
                                 help="Lower = stricter. If real book questions get 'not found', raise it; "
                                      "if unrelated questions get book answers, lower it.")
        num_ctx = st.select_slider("Context window (RAM vs. memory of the text)", [2048, 3072, 4096, 6144, 8192],
                                   value=cfg.num_ctx)
        ocr_on = st.checkbox("OCR scanned pages and images", value=cfg.ocr_enabled)

    translator = get_translator()
    if language != "English":
        st.caption(f"🌐 Translation: {translator.describe()}")

    st.divider()
    st.header("📚 My books")
    store = get_store(embed_model)
    embedder = OllamaEmbedder(embed_model, host=cfg.ollama_host)
    llm = LLM(chat_model, host=cfg.ollama_host, num_ctx=num_ctx, temperature=cfg.temperature)
    translator.llm = llm  # last-resort fallback

    uploads = st.file_uploader("Add textbooks / notes", type=sorted(e.lstrip(".") for e in SUPPORTED),
                               accept_multiple_files=True)
    for kind, msg in st.session_state.pop("notes", []):
        getattr(st, kind)(msg)

    if uploads and st.button("⚡ Process & index", type="primary"):
        notes = st.session_state.setdefault("notes", [])
        for up in uploads:
            bar = st.progress(0.0, text=f"{up.name}: starting")
            try:
                run_cfg = dataclasses.replace(cfg, ocr_enabled=ocr_on)
                doc, warns, dup = ingest_file(up.name, up.getvalue(), run_cfg, library, store, embedder,
                                              progress=lambda f, msg, n=up.name: bar.progress(min(f, 1.0), text=f"{n}: {msg}"))
                bar.empty()
                if doc is None:
                    notes.append(("error", f"{up.name}: " + " ".join(warns)))
                    continue
                notes.append(("success", f"{up.name}: " + ("already in your library." if dup
                              else f"{doc.pages} pages, {doc.chunks} passages indexed.")))
                notes.extend(("info", w) for w in warns)
            except (IngestError, EmbeddingError) as e:
                bar.empty()
                notes.append(("error", f"{up.name}: {e}"))
        st.rerun()

    docs = library.list_docs()
    stale = [d for d in docs if slug(embed_model) not in d.indexed_with]
    if stale:
        st.warning(f"{len(stale)} book(s) are not indexed for **{embed_model}**.")
        if st.button("🔄 Re-index them"):
            bar = st.progress(0.0, text="Re-indexing")
            try:
                for i, d in enumerate(stale):
                    index_document(library, store, embedder, d.doc_id,
                                   progress=lambda f, m, i=i: bar.progress((i + f) / len(stale), text=m))
            except EmbeddingError as e:
                st.session_state.setdefault("notes", []).append(("error", str(e)))
            st.rerun()

    if not docs:
        st.info("No books yet. Upload a PDF, Word, PowerPoint, text file or a photo of a page.")
    for d in docs:
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"**{d.name}**  \n<small>{d.pages} pages · {d.chunks} passages</small>", unsafe_allow_html=True)
        if c2.button("🗑", key=f"del_{d.doc_id}", help="Remove this book"):
            store.delete_doc(d.doc_id)
            library.delete(d.doc_id)
            st.rerun()

    doc_names = {d.doc_id: d.name for d in docs}
    scope = st.multiselect("Search in", list(doc_names), default=list(doc_names),
                           format_func=lambda i: doc_names[i]) if docs else []

# ----------------------------------------------------------------------------- helpers
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
    with st.expander(f"📚 Sources ({len(hits)})"):
        for h in hits:
            st.markdown(f"**{h['source']} · page {h['page']}** <small>(distance {h.get('_distance', 0):.2f})</small>",
                        unsafe_allow_html=True)
            st.caption(h["text"][:400] + ("…" if len(h["text"]) > 400 else ""))


# ----------------------------------------------------------------------------- study-material picker
tab_ask, tab_sum, tab_cards, tab_quiz, tab_map = st.tabs(
    ["💬 Ask my books", "📋 Summary", "🗂️ Flashcards", "📝 Mock quiz", "🧠 Mind map"])


def study_picker(key: str) -> StudySource | None:
    if not docs:
        st.info("Upload a book in the sidebar first.")
        return None
    c1, c2 = st.columns([2, 3])
    doc_id = c1.selectbox("Book", list(doc_names), format_func=lambda i: doc_names[i], key=f"{key}_doc")
    doc = library.get(doc_id)
    mode_label = c2.radio("Material", ["Whole book (key sentences)", "Page range", "Topic"], horizontal=True,
                          key=f"{key}_mode")
    src = StudySource(doc_id=doc_id)
    if mode_label == "Page range":
        a, b = st.columns(2)
        src.mode = "pages"
        src.page_from = a.number_input("From page", 1, max(doc.pages, 1), 1, key=f"{key}_pf")
        src.page_to = b.number_input("To page", 1, max(doc.pages, 1), min(doc.pages, 5), key=f"{key}_pt")
    elif mode_label == "Topic":
        src.mode = "topic"
        src.topic = st.text_input("Topic or chapter name", key=f"{key}_topic", placeholder="e.g. Photosynthesis")
    return src


def get_material(src: StudySource) -> str | None:
    if src.mode == "topic" and not src.topic.strip():
        st.warning("Type a topic first.")
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


# ============================================================================= TAB 1: ask
with tab_ask:
    if not docs:
        st.info("👈 Add a textbook in the sidebar to ask questions about it. You can still chat without one.")
    for m in state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])
            if m.get("notice"):
                st.caption("⚠️ " + m["notice"])
            sources_box(m.get("sources", []))

    if q := st.chat_input("Ask a question about your books…"):
        q_en = to_english(q)
        history_en = [{"role": m["role"], "content": m["en"]} for m in state.messages]
        state.messages.append({"role": "user", "content": q, "en": q_en})
        with st.chat_message("user"):
            st.markdown(q)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your books…"):
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
                with st.spinner(f"Translating to {language}…"):
                    shown = tr_or_stop(translator.translate, answer, ENGLISH, lang_code)
                holder.markdown(shown)
            if plan.notice:
                st.caption("⚠️ " + plan.notice)
            sources_box(plan.hits)
        state.messages.append({"role": "assistant", "content": shown, "en": answer,
                               "sources": plan.hits, "notice": plan.notice})
    if state.messages and st.button("🧹 Clear chat"):
        state.messages = []
        st.rerun()

# ============================================================================= TAB 2: summary
with tab_sum:
    st.subheader("Chapter summary")
    src = study_picker("sum")
    if src:
        c1, c2 = st.columns(2)
        if c1.button("✨ Generate AI summary", type="primary"):
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
        if c2.button("⚡ Instant key points (no AI model)"):
            material = get_material(src)
            if material:
                state["keypoints"] = top_sentences(material, max_sentences=8)

    if state.get("keypoints"):
        with st.container(border=True):
            st.markdown("**Key sentences (TextRank — works even on very slow PCs)**")
            for s in state["keypoints"]:
                st.markdown(f"- {s}")
    if state.get("summary_en"):
        shown = state["summary_en"]
        if lang_code != ENGLISH:
            cache = state.setdefault("summary_tr", {})
            if lang_code not in cache:
                with st.spinner(f"Translating to {language}…"):
                    cache[lang_code] = tr_or_stop(translator.translate_markdown, state["summary_en"], lang_code)
            shown = cache[lang_code]
        with st.container(border=True):
            st.markdown(shown)
        st.download_button("⬇️ Download (.md)", shown, file_name="summary.md", mime="text/markdown")

# ============================================================================= TAB 3: flashcards
with tab_cards:
    st.subheader("Flashcards")
    src = study_picker("fc")
    n_cards = st.slider("Number of cards", 4, 15, 8, key="fc_n")
    if src and st.button("🗂️ Make flashcards", type="primary"):
        material = get_material(src)
        if material:
            with st.spinner("Writing flashcards…"):
                try:
                    state["cards_en"], state["card_i"], state["card_flip"] = make_flashcards(llm, material, grade, n_cards), 0, False
                    new_gen("cards")
                except (LLMError, ValueError) as e:
                    st.error(str(e))
    cards_en = state.get("cards_en")
    if cards_en:
        cards = localized("cards", cards_en, lambda items, tr, tgt: translate_cards(items, tr, tgt))
        i = state.get("card_i", 0) % len(cards)
        with st.container(border=True):
            st.caption(f"Card {i + 1} of {len(cards)} — {'answer' if state.get('card_flip') else 'question'}")
            st.markdown(f"## {cards[i].back if state.get('card_flip') else cards[i].front}")
        b1, b2, b3 = st.columns(3)
        if b1.button("⬅️ Previous"):
            state["card_i"], state["card_flip"] = (i - 1) % len(cards), False
            st.rerun()
        if b2.button("🔄 Flip"):
            state["card_flip"] = not state.get("card_flip", False)
            st.rerun()
        if b3.button("Next ➡️"):
            state["card_i"], state["card_flip"] = (i + 1) % len(cards), False
            st.rerun()
        st.download_button("⬇️ Download as CSV (Anki-ready)", flashcards_to_csv(cards), file_name="flashcards.csv",
                           mime="text/csv")

# ============================================================================= TAB 4: quiz
with tab_quiz:
    st.subheader("Mock quiz")
    src = study_picker("qz")
    n_q = st.slider("Number of questions", 3, 10, 5, key="qz_n")
    if src and st.button("📝 Create quiz", type="primary"):
        material = get_material(src)
        if material:
            with st.spinner("Writing questions…"):
                try:
                    state["quiz_en"] = make_quiz(llm, material, grade, n_q)
                    new_gen("quiz")
                    state["quiz_done"] = False
                    for k in [k for k in state if str(k).startswith("qa_")]:
                        del state[k]
                except (LLMError, ValueError) as e:
                    st.error(str(e))
    quiz_en = state.get("quiz_en")
    if quiz_en:
        quiz = localized("quiz", quiz_en, lambda items, tr, tgt: translate_quiz(items, tr, tgt))
        done = state.get("quiz_done", False)
        score = 0
        for qi, q in enumerate(quiz):
            choice = st.radio(f"**{qi + 1}. {q.question}**", q.options, index=None, key=f"qa_{qi}", disabled=done)
            if done:
                right = q.options[q.answer_index]
                if choice == right:
                    score += 1
                    st.success("Correct ✅")
                else:
                    st.error(f"Answer: {right}")
                if q.explanation:
                    st.caption(q.explanation)
        if not done:
            if st.button("✅ Check answers"):
                state["quiz_done"] = True
                st.rerun()
        else:
            st.metric("Your score", f"{score} / {len(quiz)}")
            if st.button("🔁 Try again"):
                state["quiz_done"] = False
                for k in [k for k in state if str(k).startswith("qa_")]:
                    del state[k]
                st.rerun()

# ============================================================================= TAB 5: mind map
with tab_map:
    st.subheader("Visual mind map")
    src = study_picker("mm")
    if src and st.button("🧠 Draw mind map", type="primary"):
        material = get_material(src)
        if material:
            with st.spinner("Thinking…"):
                try:
                    state["map_en"] = make_mindmap(llm, material, grade)
                    new_gen("map")
                except (LLMError, ValueError) as e:
                    st.error(str(e))
    if state.get("map_en"):
        mm = localized("map", state["map_en"], lambda item, tr, tgt: translate_mindmap(item, tr, tgt))
        dot = mindmap_to_dot(mm)
        st.graphviz_chart(dot)
        st.download_button("⬇️ Download graph (.dot)", dot, file_name="mindmap.dot")
