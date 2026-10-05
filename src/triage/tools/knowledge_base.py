"""Tool: keyword search over the support knowledge base."""
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..data_loader import read_json
from .base import Tool

_WORD = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {"the", "and", "for", "with", "not", "are", "can", "you", "your", "has", "have",
     "this", "that", "from", "but", "was", "when", "what", "how", "its", "any", "all"}
)

TAG_WEIGHT = 3
TITLE_WEIGHT = 2
BODY_WEIGHT = 1


def _tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall(text.lower()) if len(w) > 2 and w not in _STOPWORDS}


class Article(BaseModel):
    id: str
    title: str
    tags: list[str]
    content: str


class KnowledgeBaseArgs(BaseModel):
    query: str = Field(
        min_length=2,
        description="Short English keywords describing the problem, e.g. 'duplicate charge'.",
    )
    max_results: int = Field(default=3, ge=1, le=5, description="How many articles to return.")


class SearchKnowledgeBase(Tool):
    name = "search_knowledge_base"
    description = (
        "Search the support FAQ and help articles. Returns the most relevant articles with "
        "their full text. Always write the query in English, even if the ticket is not."
    )
    args_model = KnowledgeBaseArgs

    def __init__(self, articles: list[Article]) -> None:
        self._articles = articles

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "SearchKnowledgeBase":
        raw = read_json(data_dir / "kb.json")
        return cls([Article.model_validate(item) for item in raw])

    @staticmethod
    def _score(article: Article, query: str, query_tokens: set[str]) -> int:
        tag_hits = sum(1 for tag in article.tags if tag in query)
        title_hits = len(query_tokens & _tokens(article.title))
        body_hits = len(query_tokens & _tokens(article.content))
        return tag_hits * TAG_WEIGHT + title_hits * TITLE_WEIGHT + body_hits * BODY_WEIGHT

    def run(self, args: KnowledgeBaseArgs) -> dict[str, Any]:
        query = args.query.lower()
        query_tokens = _tokens(query)
        scored = [(self._score(a, query, query_tokens), a) for a in self._articles]
        ranked = sorted((pair for pair in scored if pair[0] > 0), key=lambda p: p[0], reverse=True)
        return {
            "query": args.query,
            "results": [
                {"id": a.id, "title": a.title, "content": a.content, "relevance": score}
                for score, a in ranked[: args.max_results]
            ],
        }
