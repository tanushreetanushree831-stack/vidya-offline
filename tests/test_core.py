from pydantic import BaseModel

from conftest import FakeEmbedder, FakeLLM
from vidya.config import load_config
from vidya.library import DocInfo, Library
from vidya.llm import ThinkStripper, parse_json_model, pick_default, strip_think, is_embed_model
from vidya.pipeline import index_document, ingest_file
from vidya.rag import Tutor, build_plan, contextualize, is_chitchat
from vidya.store import VectorStore
from vidya.study import (Branch, MCQ, MindMap, Quiz, StudySource, clean_quiz, flashcards_to_csv, gather_context,
                         mindmap_to_dot, Flashcard)
from vidya.textrank import compress, keywords, top_sentences
from vidya.translate import Translator, looks_non_english

BOOK = ("Photosynthesis is the process by which green plants make food using sunlight, water and carbon dioxide. "
        "It takes place in the chloroplasts, which contain chlorophyll. Oxygen is released as a by-product. "
        "The water cycle describes how water evaporates from oceans, condenses into clouds and falls as rain. "
        "Evaporation is driven by heat from the sun. Condensation turns vapour into tiny droplets. "
        "Newton's first law says an object stays at rest unless a force acts on it. "
        "Force equals mass times acceleration according to the second law. ") * 3


# ---------------------------------------------------------------- llm helpers
def test_think_stripper_across_chunks():
    s = ThinkStripper()
    pieces = ["Hel", "lo <thi", "nk>secret reason", "ing</th", "ink> world", "!"]
    out = "".join(s.feed(p) for p in pieces) + s.flush()
    assert out == "Hello  world!"


def test_strip_think_variants():
    assert strip_think("<think>x</think>Answer") == "Answer"
    assert strip_think("Answer<think>never closed") == "Answer"


class Tiny(BaseModel):
    a: int


def test_parse_json_with_noise():
    assert parse_json_model('Sure! ```json\n{"a": 3}\n```', Tiny).a == 3
    assert parse_json_model('<think>hm</think>{"a": 1}', Tiny).a == 1


def test_model_helpers():
    assert is_embed_model("nomic-embed-text:latest") and not is_embed_model("llama3.2:latest")
    assert pick_default(["gemma:2b", "llama3.2:latest"], "llama3.2") == 1
    assert pick_default(["gemma:2b", "llama3.2:1b"], "llama3.2") == 1


# ---------------------------------------------------------------- textrank
def test_textrank_selects_and_compresses():
    top = top_sentences(BOOK, max_sentences=3)
    assert 1 <= len(top) <= 3
    assert len(compress(BOOK, 400)) <= 450
    assert "photosynthesis" in keywords(BOOK, 10) or "water" in keywords(BOOK, 10)


# ---------------------------------------------------------------- store + library + pipeline
def make_env(tmp_path):
    lib = Library(tmp_path / "lib")
    store = VectorStore(tmp_path / "db", "fake-embed")
    return lib, store, FakeEmbedder()


def test_ingest_search_filter_delete(tmp_path):
    cfg = load_config()
    lib, store, emb = make_env(tmp_path)
    doc, warns, dup = ingest_file("science.txt", BOOK.encode(), cfg, lib, store, emb)
    assert doc and not dup and doc.chunks >= 1 and store.embed_slug in doc.indexed_with
    doc2, _, dup2 = ingest_file("other.txt", ("Algebra uses letters for unknown numbers. " * 20).encode(),
                                cfg, lib, store, emb)
    assert doc2.doc_id != doc.doc_id

    hits = store.search(emb.embed_query("What is photosynthesis?"), k=3)
    assert hits and "hotosynthesis" in hits[0]["text"]
    assert hits[0]["_distance"] < hits[-1]["_distance"] + 1e-9
    assert "vector" not in hits[0]

    only2 = store.search(emb.embed_query("photosynthesis"), k=3, doc_ids=[doc2.doc_id])
    assert only2 and all(h["doc_id"] == doc2.doc_id for h in only2)

    # duplicate upload is detected
    _, _, dup3 = ingest_file("science-copy.txt", BOOK.encode(), cfg, lib, store, emb)
    assert dup3
    assert len(lib.list_docs()) == 2

    # reindex is idempotent
    n_before = store.count()
    index_document(lib, store, emb, doc.doc_id)
    assert store.count() == n_before

    store.delete_doc(doc.doc_id)
    lib.delete(doc.doc_id)
    assert all(h["doc_id"] != doc.doc_id for h in store.search(emb.embed_query("photosynthesis"), k=10))
    assert lib.get(doc.doc_id) is None and not lib.load_chunks(doc.doc_id)


def test_apostrophes_in_filenames_dont_break_filters(tmp_path):
    lib, store, emb = make_env(tmp_path)
    store.add([{"chunk_id": "a", "doc_id": "o'brien", "source": "x", "page": 1, "text": "hello world",
                "vector": emb.embed_query("hello world")}])
    assert store.search(emb.embed_query("hello"), 2, doc_ids=["o'brien"])
    store.delete_doc("o'brien")
    assert store.count() == 0


def test_reindex_for_new_embedding_model(tmp_path):
    cfg = load_config()
    lib, store, emb = make_env(tmp_path)
    doc, *_ = ingest_file("s.txt", BOOK.encode(), cfg, lib, store, emb)
    store2 = VectorStore(tmp_path / "db", "another-model")
    assert store2.embed_slug not in lib.get(doc.doc_id).indexed_with
    assert store2.search(emb.embed_query("water"), 3) == []
    index_document(lib, store2, emb, doc.doc_id)
    assert store2.embed_slug in lib.get(doc.doc_id).indexed_with
    assert store2.search(emb.embed_query("water"), 3)


# ---------------------------------------------------------------- router
def test_chitchat_detection():
    for q in ["hi", "Hello!", "thanks", "who are you?", "Good morning"]:
        assert is_chitchat(q), q
    for q in ["hi, what is photosynthesis", "What is gravity?"]:
        assert not is_chitchat(q), q


def test_followup_contextualization():
    hist = [{"role": "user", "content": "What is photosynthesis?"}, {"role": "assistant", "content": "..."}]
    assert "photosynthesis" in contextualize("why?", hist)
    assert contextualize("Explain the water cycle in detail please", hist) == "Explain the water cycle in detail please"


def test_three_layer_routing(tmp_path):
    cfg = load_config()
    lib, store, emb = make_env(tmp_path)
    ingest_file("science.txt", BOOK.encode(), cfg, lib, store, emb)
    tutor = Tutor(emb, store)
    kw = dict(history_en=[], grade="Class 6-8", doc_ids=None, top_k=3, has_docs=True)

    p = tutor.plan(max_distance=0.7, query_en="hello", **kw)
    assert p.route == "chitchat" and not p.hits

    p = tutor.plan(max_distance=0.9, query_en="What is photosynthesis and where does it take place?", **kw)
    assert p.route == "book" and p.hits
    assert "TEXTBOOK EXCERPTS" in p.messages[0]["content"] and "science.txt" in p.messages[0]["content"]

    p = tutor.plan(max_distance=0.05, query_en="Who won the 1983 cricket world cup?", **kw)
    assert p.route == "general" and "general knowledge" in p.notice and not p.hits

    p = build_plan("What is gravity?", [], "Class 1-3", [], 0.6, has_docs=False)
    assert p.route == "general" and "No book uploaded" in p.notice
    assert "6-9 years" in p.messages[0]["content"]


# ---------------------------------------------------------------- study
def test_clean_quiz_filters_and_keeps_answer():
    quiz = Quiz(questions=[
        MCQ(question="Q1?", options=["a", "b", "c", "d"], answer_index=2, explanation="e"),
        MCQ(question="Bad index", options=["a", "b", "c", "d"], answer_index=9),
        MCQ(question="Too few", options=["a", "b"], answer_index=0),
        MCQ(question="Dupes", options=["a", "a", "a", "b"], answer_index=0),
    ])
    for _ in range(20):  # shuffling must never lose the right answer
        qs = clean_quiz(quiz, 5)
        assert len(qs) == 1 and qs[0].options[qs[0].answer_index] == "c"


def test_mindmap_dot_is_escaped():
    mm = MindMap(topic='Plants "life"', branches=[Branch(name="Photosynthesis", children=["Needs sunlight & water"])])
    dot = mindmap_to_dot(mm)
    assert dot.startswith("digraph") and '"life"' not in dot and "b0 -> b0_0" in dot


def test_gather_context_modes(tmp_path):
    cfg = load_config()
    lib, store, emb = make_env(tmp_path)
    doc, *_ = ingest_file("s.txt", BOOK.encode(), cfg, lib, store, emb)
    whole = gather_context(lib, store, emb, StudySource(doc.doc_id, "whole"), max_chars=500)
    assert 0 < len(whole) <= 600
    topic = gather_context(lib, store, emb, StudySource(doc.doc_id, "topic", topic="Newton force"), max_chars=2000)
    assert "force" in topic.lower()
    assert gather_context(lib, store, emb, StudySource(doc.doc_id, "pages", 1, 1)).strip()
    assert gather_context(lib, store, emb, StudySource(doc.doc_id, "pages", 50, 60)) == ""


def test_flashcard_csv_quotes():
    out = flashcards_to_csv([Flashcard(front="Define, 'x'", back='He said "hi"')])
    assert '"Define, \'x\'"' in out


# ---------------------------------------------------------------- translation
def test_translation_llm_fallback_and_markdown(tmp_path):
    llm = FakeLLM(reply="TRANSLATED")
    tr = Translator(tmp_path / "ct2", tmp_path / "hf", llm=llm)
    assert tr.backend_name() == "llm"
    md = "## Main Points\n- Plants make food\n1. Second point\n\n**Bold** line"
    out = tr.translate_markdown(md, "hin_Deva")
    lines = out.splitlines()
    assert lines[0] == "## TRANSLATED" and lines[1] == "- TRANSLATED" and lines[2] == "1. TRANSLATED"
    assert lines[3] == "" and lines[4] == "TRANSLATED"
    assert tr.translate("same", "eng_Latn", "eng_Latn") == "same"
    n = len(llm.calls)
    tr.translate("Plants make food", "eng_Latn", "hin_Deva")  # cached
    assert len(llm.calls) == n


def test_no_backend_available(tmp_path):
    assert Translator(tmp_path / "a", tmp_path / "b").backend_name() == "none"


def test_script_detection():
    assert looks_non_english("ಪ್ರಕಾಶ ಸಂಶ್ಲೇಷಣೆ ಎಂದರೇನು?") and looks_non_english("प्रकाश संश्लेषण क्या है")
    assert not looks_non_english("What is photosynthesis?")
