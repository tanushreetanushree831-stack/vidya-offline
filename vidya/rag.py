"""Stage 3 (reasoning): the 3-layer router + grounded prompts.

Layer 1  chit-chat / greetings        -> answer directly, no retrieval
Layer 2  question found in the books  -> answer ONLY from retrieved passages, with page citations
Layer 3  not found in the books       -> general knowledge, clearly flagged to the student
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .embeddings import OllamaEmbedder
from .store import VectorStore

CLASS_LEVELS = ["Class 1-3", "Class 4-5", "Class 6-8", "Class 9-10", "Class 11-12", "College"]

GRADE_STYLE = {
    "Class 1-3": "The student is 6-9 years old. Use very short sentences, simple everyday words and one friendly example. Maximum 4 sentences.",
    "Class 4-5": "The student is 9-11 years old. Use simple words, short sentences and one familiar example from daily life.",
    "Class 6-8": "The student is 11-14 years old. Explain clearly in simple language; define any technical term the first time you use it.",
    "Class 9-10": "The student is 14-16 years old. Be clear and accurate; use correct technical terms with brief explanations and steps where useful.",
    "Class 11-12": "The student is 16-18 years old. Be precise and thorough; include formulas, reasoning steps and exam-relevant points.",
    "College": "The student is at undergraduate level. Be rigorous and concise, with correct terminology.",
}

_CHITCHAT = re.compile(
    r"^\s*(hi+|hello+|hey+|namaste|namaskara|good\s+(morning|afternoon|evening|night)|thanks?(\s+you)?|"
    r"thank\s+you(\s+so\s+much)?|ok(ay)?|bye|goodbye|who\s+are\s+you|what\s+can\s+you\s+do|how\s+are\s+you|help)"
    r"[\s!.?]*$", re.I)

MAX_HISTORY = 6


def is_chitchat(query: str) -> bool:
    return bool(_CHITCHAT.match(query))


def contextualize(query: str, history: list[dict]) -> str:
    """Very short follow-ups ('why?', 'explain more') borrow the previous question for retrieval."""
    if len(query.split()) <= 4 and history:
        for m in reversed(history):
            if m["role"] == "user":
                return f"{m['content']} {query}"
    return query


@dataclass
class Plan:
    route: str                       # "chitchat" | "book" | "general"
    messages: list[dict]
    hits: list[dict] = field(default_factory=list)
    notice: str = ""


def _history(history: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in history[-MAX_HISTORY:]]


def build_plan(query: str, history: list[dict], grade: str, hits: list[dict],
               max_distance: float, has_docs: bool) -> Plan:
    style = GRADE_STYLE.get(grade, GRADE_STYLE["Class 6-8"])
    persona = "You are Vidya, a kind, patient offline tutor for students in India. " + style

    if is_chitchat(query):
        sys = persona + " Reply briefly and warmly. If asked, say you can answer questions from the student's own textbooks, and make summaries, flashcards, quizzes and mind maps."
        return Plan("chitchat", [{"role": "system", "content": sys}, *_history(history),
                                 {"role": "user", "content": query}])

    good = [h for h in hits if h.get("_distance", 9.0) <= max_distance]
    if good:
        blocks = "\n\n".join(f"[{i}] ({h['source']}, page {h['page']})\n{h['text']}"
                             for i, h in enumerate(good, 1))
        sys = (persona + "\n\nAnswer the question using ONLY the textbook excerpts below. "
               "If the excerpts do not contain the answer, say you could not find it in the book. "
               "Do not invent facts, page numbers or quotes.\n\nTEXTBOOK EXCERPTS:\n" + blocks)
        return Plan("book", [{"role": "system", "content": sys}, *_history(history),
                             {"role": "user", "content": query}], good)

    sys = persona + " The question could not be matched to the student's textbooks, so answer from general knowledge. Be accurate, and say when you are unsure."
    notice = ("No matching passage in your books" if has_docs else "No book uploaded yet") + \
             " - this answer comes from the AI's general knowledge. Please check it with your teacher or textbook."
    if hits:  # helps teachers tune the strictness slider
        best = min(h.get("_distance", 9.0) for h in hits)
        notice += f" (closest passage distance {best:.2f}, limit {max_distance:.2f})"
    return Plan("general", [{"role": "system", "content": sys}, *_history(history),
                            {"role": "user", "content": query}], [], notice)


class Tutor:
    def __init__(self, embedder: OllamaEmbedder, store: VectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def plan(self, query_en: str, history_en: list[dict], grade: str, doc_ids: list[str] | None,
             top_k: int, max_distance: float, has_docs: bool) -> Plan:
        hits: list[dict] = []
        if has_docs and not is_chitchat(query_en):
            q = contextualize(query_en, history_en)
            hits = self.store.search(self.embedder.embed_query(q), k=top_k, doc_ids=doc_ids)
        return build_plan(query_en, history_en, grade, hits, max_distance, has_docs)
