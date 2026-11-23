"""Headless UI smoke tests (Streamlit AppTest) with Ollama mocked out."""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from conftest import FakeEmbedder
from vidya.study import Branch, Flashcard, FlashcardSet, MCQ, MindMap, Quiz

APP = str(Path(__file__).resolve().parent.parent / "app.py")
BOOK = ("Photosynthesis is the process by which green plants make food using sunlight, water and carbon dioxide. "
        "It takes place in the chloroplasts, which contain chlorophyll. Oxygen is released as a by-product. " * 4)


@pytest.fixture
def env(tmp_path, monkeypatch):
    import streamlit as st
    st.cache_resource.clear()
    monkeypatch.setenv("VIDYA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VIDYA_MODELS_DIR", str(tmp_path / "models"))

    import vidya.embeddings as emb
    import vidya.llm as llm
    fe = FakeEmbedder()
    monkeypatch.setattr(emb.OllamaEmbedder, "_embed", lambda self, texts: [fe._vec(t) for t in texts])
    monkeypatch.setattr(llm.LLM, "is_up", lambda self: True)
    monkeypatch.setattr(llm.LLM, "list_models", lambda self: ["llama3.2:latest", "nomic-embed-text:latest"])
    monkeypatch.setattr(llm.LLM, "chat_stream",
                        lambda self, messages, temperature=None: iter(["Plants ", "make ", "food."]))

    def structured(self, messages, model_cls, temperature=0.2, retries=2):
        if model_cls is FlashcardSet:
            return FlashcardSet(cards=[Flashcard(front=f"Q{i}", back=f"A{i}") for i in range(3)])
        if model_cls is Quiz:
            return Quiz(questions=[MCQ(question="What do plants make?", options=["Food", "Rocks", "Glass", "Iron"],
                                       answer_index=0, explanation="Plants make food.")])
        return MindMap(topic="Photosynthesis", branches=[Branch(name="Inputs", children=["Light", "Water"])])
    monkeypatch.setattr(llm.LLM, "structured", structured)

    # pre-load a book (file upload widgets can't be driven by AppTest)
    from vidya.config import load_config
    from vidya.library import Library
    from vidya.pipeline import ingest_file
    from vidya.store import VectorStore
    cfg = load_config()
    lib, store = Library(cfg.library_dir), VectorStore(cfg.lancedb_dir, "nomic-embed-text:latest")
    doc, *_ = ingest_file("biology.txt", BOOK.encode(), cfg, lib, store, emb.OllamaEmbedder("nomic-embed-text:latest"))
    assert doc
    return doc


def run():
    at = AppTest.from_file(APP, default_timeout=60)
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    return at


def test_ollama_down_shows_instructions(env, monkeypatch):
    import vidya.llm as llm
    monkeypatch.setattr(llm.LLM, "is_up", lambda self: False)
    at = run()
    assert any("Ollama is not running" in e.value for e in at.error)


def test_models_missing_shows_pull_commands(env, monkeypatch):
    import vidya.llm as llm
    monkeypatch.setattr(llm.LLM, "list_models", lambda self: ["llama3.2:latest"])
    at = run()
    assert any("missing" in e.value for e in at.error)
    assert any("ollama pull nomic-embed-text" in c.value for c in at.code)


def test_chat_book_answer_with_sources(env):
    at = run()
    at.sidebar.expander[0].slider[1].set_value(1.0).run()  # toy embedder has larger distances than a real one
    at.chat_input[0].set_value("What is photosynthesis?").run()
    assert not at.exception, [e.value for e in at.exception]
    msgs = at.session_state["messages"]
    assert msgs[-1]["content"] == "Plants make food." and msgs[-1]["sources"]
    assert msgs[-1]["sources"][0]["source"] == "biology.txt"


def test_chat_unrelated_goes_general_with_notice(env):
    at = run()
    at.sidebar.expander[0].slider[1].set_value(0.2).run()  # very strict matching
    at.chat_input[0].set_value("Tell me about the moon landing in 1969").run()
    assert not at.exception, [e.value for e in at.exception]
    last = at.session_state["messages"][-1]
    assert last["notice"] and not last["sources"]


def test_study_tools(env):
    at = run()
    # Summary
    next(b for b in at.button if "Generate AI summary" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["summary_en"] == "Plants make food."
    next(b for b in at.button if "Instant key points" in b.label).click().run()
    assert at.session_state["keypoints"]
    # Flashcards
    next(b for b in at.button if "Make flashcards" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.session_state["cards_en"]) == 3
    next(b for b in at.button if "Flip" in b.label).click().run()
    assert at.session_state["card_flip"] is True
    # Quiz
    next(b for b in at.button if "Create quiz" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    radio = next(r for r in at.radio if r.label.startswith("**1."))
    right = at.session_state["quiz_en"][0].options[at.session_state["quiz_en"][0].answer_index]
    radio.set_value(right).run()
    next(b for b in at.button if "Check answers" in b.label).click().run()
    assert any("1 / 1" in m.value for m in at.metric)
    # Mind map
    next(b for b in at.button if "Draw mind map" in b.label).click().run()
    assert not at.exception, [e.value for e in at.exception]
    assert at.session_state["map_en"].topic == "Photosynthesis"


def test_hindi_flow_uses_translator(env, monkeypatch):
    from vidya.translate import Translator
    monkeypatch.setattr(Translator, "backend_name", lambda self: "ct2")
    monkeypatch.setattr(Translator, "_load", lambda self: "ct2")
    monkeypatch.setattr(Translator, "_nllb_batch",
                        lambda self, batch, src, tgt, be: [f"[{tgt}] {s}" for s in batch])
    at = run()
    at.sidebar.selectbox[3].set_value("Hindi (हिन्दी)").run()
    at.sidebar.expander[0].slider[1].set_value(1.0).run()
    at.chat_input[0].set_value("प्रकाश संश्लेषण क्या है?").run()
    assert not at.exception, [e.value for e in at.exception]
    last = at.session_state["messages"][-1]
    assert last["en"] == "Plants make food." and last["content"] == "[hin_Deva] Plants make food."
    # the Hindi question was translated to English before retrieval/history
    assert at.session_state["messages"][-2]["en"].startswith("[eng_Latn]")
