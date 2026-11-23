"""Stage 5 outputs: summaries, flashcards, mock quizzes, mind maps."""
from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass

from pydantic import BaseModel, Field

from .library import Library
from .llm import LLM
from .rag import GRADE_STYLE
from .store import VectorStore
from .embeddings import OllamaEmbedder
from .textrank import compress


# ----------------------------------------------------------------------------- schemas
class Flashcard(BaseModel):
    front: str
    back: str


class FlashcardSet(BaseModel):
    cards: list[Flashcard] = Field(default_factory=list)


class MCQ(BaseModel):
    question: str
    options: list[str]
    answer_index: int
    explanation: str = ""


class Quiz(BaseModel):
    questions: list[MCQ] = Field(default_factory=list)


class Branch(BaseModel):
    name: str
    children: list[str] = Field(default_factory=list)


class MindMap(BaseModel):
    topic: str
    branches: list[Branch] = Field(default_factory=list)


# ----------------------------------------------------------------------------- context
@dataclass
class StudySource:
    doc_id: str
    mode: str = "whole"        # "whole" | "pages" | "topic"
    page_from: int = 1
    page_to: int = 1
    topic: str = ""


def gather_context(library: Library, store: VectorStore, embedder: OllamaEmbedder,
                   src: StudySource, max_chars: int = 5000, k: int = 8) -> str:
    chunks = library.load_chunks(src.doc_id)
    if src.mode == "pages":
        sel = [c for c in chunks if src.page_from <= c["page"] <= src.page_to]
    elif src.mode == "topic" and src.topic.strip():
        hits = store.search(embedder.embed_query(src.topic), k=k, doc_ids=[src.doc_id])
        sel = sorted(hits, key=lambda h: (h["page"], h["chunk_id"]))
    else:
        sel = chunks
    text = "\n".join(c["text"] for c in sel)
    return compress(text, max_chars)


# ----------------------------------------------------------------------------- generators
def _sys(grade: str, task: str) -> str:
    return ("You are Vidya, an offline tutor. " + GRADE_STYLE.get(grade, "") + " " + task +
            " Use ONLY the study material provided; do not add outside facts.")


def summary_messages(material: str, grade: str) -> list[dict]:
    task = ("Write a structured revision summary in Markdown with exactly these sections:\n"
            "## Big Idea (1-2 sentences)\n## Main Points (5-8 short bullets)\n"
            "## Key Terms (4-6 bullets as 'term: simple meaning')\n## Remember (one line)")
    return [{"role": "system", "content": _sys(grade, task)},
            {"role": "user", "content": f"STUDY MATERIAL:\n{material}"}]


def make_flashcards(llm: LLM, material: str, grade: str, n: int = 8) -> list[Flashcard]:
    task = (f"Create {n} flashcards. 'front' is a short question or a term; 'back' is a clear answer "
            "in at most 25 words. Return JSON only.")
    out = llm.structured([{"role": "system", "content": _sys(grade, task)},
                          {"role": "user", "content": f"STUDY MATERIAL:\n{material}"}], FlashcardSet)
    cards = [Flashcard(front=c.front.strip(), back=c.back.strip())
             for c in out.cards if c.front.strip() and c.back.strip()]
    if not cards:
        raise ValueError("The model returned no usable flashcards. Try again or choose a bigger passage.")
    return cards[:n]


def clean_quiz(quiz: Quiz, n: int) -> list[MCQ]:
    good: list[MCQ] = []
    for q in quiz.questions:
        opts = []
        for o in q.options:
            o = o.strip()
            if o and o not in opts:
                opts.append(o)
        if not q.question.strip() or not (3 <= len(opts) <= 5):
            continue
        if not (0 <= q.answer_index < len(q.options)):
            continue
        correct = q.options[q.answer_index].strip()
        if correct not in opts:
            continue
        random.shuffle(opts)  # small models love putting the answer at A
        good.append(MCQ(question=q.question.strip(), options=opts,
                        answer_index=opts.index(correct), explanation=q.explanation.strip()))
    return good[:n]


def make_quiz(llm: LLM, material: str, grade: str, n: int = 5) -> list[MCQ]:
    task = (f"Create {n} multiple-choice questions. Each has exactly 4 options, 'answer_index' "
            "(0-3) pointing at the single correct option, and a one-sentence 'explanation'. Return JSON only.")
    out = llm.structured([{"role": "system", "content": _sys(grade, task)},
                          {"role": "user", "content": f"STUDY MATERIAL:\n{material}"}], Quiz)
    qs = clean_quiz(out, n)
    if not qs:
        raise ValueError("The model returned no usable questions. Try again or choose a bigger passage.")
    return qs


def make_mindmap(llm: LLM, material: str, grade: str) -> MindMap:
    task = ("Build a mind map. 'topic' is the central idea (max 5 words). Give 4-6 'branches', each with a short "
            "'name' (max 4 words) and 2-4 'children' (max 6 words each). Return JSON only.")
    mm = llm.structured([{"role": "system", "content": _sys(grade, task)},
                         {"role": "user", "content": f"STUDY MATERIAL:\n{material}"}], MindMap)
    mm.branches = [Branch(name=b.name.strip(), children=[c.strip() for c in b.children if c.strip()][:5])
                   for b in mm.branches if b.name.strip()][:8]
    if not mm.topic.strip() or not mm.branches:
        raise ValueError("The model returned an empty mind map. Try again.")
    return mm


# ----------------------------------------------------------------------------- rendering / export
def _wrap(label: str, width: int = 22) -> str:
    words, lines, cur = label.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return "\\n".join(lines)


def _esc(s: str) -> str:
    return s.replace("\\", "").replace('"', "'")


PALETTE = ["#fde68a", "#bfdbfe", "#bbf7d0", "#fbcfe8", "#ddd6fe", "#fed7aa", "#a5f3fc", "#e9d5ff"]


def mindmap_to_dot(mm: MindMap) -> str:
    """Graphviz DOT. Streamlit draws it in the browser (viz.js) - no Graphviz install needed."""
    lines = ["digraph G {", "rankdir=LR;", 'bgcolor="transparent";',
             'node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, color="#475569"];',
             'edge [color="#94a3b8"];',
             f'root [label="{_wrap(_esc(mm.topic), 18)}", fillcolor="#fca5a5", fontsize=15, shape=ellipse];']
    for i, b in enumerate(mm.branches):
        color = PALETTE[i % len(PALETTE)]
        lines.append(f'b{i} [label="{_wrap(_esc(b.name))}", fillcolor="{color}"];')
        lines.append(f"root -> b{i};")
        for j, c in enumerate(b.children):
            lines.append(f'b{i}_{j} [label="{_wrap(_esc(c))}", fillcolor="white"];')
            lines.append(f"b{i} -> b{i}_{j};")
    lines.append("}")
    return "\n".join(lines)


def flashcards_to_csv(cards: list[Flashcard]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["front", "back"])
    for c in cards:
        w.writerow([c.front, c.back])
    return buf.getvalue()


# ----------------------------------------------------------------------------- translation helpers
def translate_cards(cards: list[Flashcard], tr, tgt: str) -> list[Flashcard]:
    fronts = tr.translate_many([c.front for c in cards], "eng_Latn", tgt)
    backs = tr.translate_many([c.back for c in cards], "eng_Latn", tgt)
    return [Flashcard(front=f, back=b) for f, b in zip(fronts, backs)]


def translate_quiz(qs: list[MCQ], tr, tgt: str) -> list[MCQ]:
    out = []
    for q in qs:
        texts = [q.question, q.explanation, *q.options]
        t = tr.translate_many(texts, "eng_Latn", tgt)
        out.append(MCQ(question=t[0], explanation=t[1], options=t[2:], answer_index=q.answer_index))
    return out


def translate_mindmap(mm: MindMap, tr, tgt: str) -> MindMap:
    names = tr.translate_many([mm.topic] + [b.name for b in mm.branches], "eng_Latn", tgt)
    branches = []
    for i, b in enumerate(mm.branches):
        kids = tr.translate_many(b.children, "eng_Latn", tgt) if b.children else []
        branches.append(Branch(name=names[i + 1], children=kids))
    return MindMap(topic=names[0], branches=branches)
