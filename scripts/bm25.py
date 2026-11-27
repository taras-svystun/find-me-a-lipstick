"""A5: BM25 keyword search over item text, written out so every weight can be explained.

score(q, d) = sum over query words w of  idf(w) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(d) / avg_len))
idf(w) = ln(1 + (N - n_w + 0.5) / (n_w + 0.5))      (Lucene / Elasticsearch form, never negative)
"""

import math
import re
from collections import Counter, defaultdict

K1 = 1.2  # how fast repeats saturate: 1.2 means the 2nd "lipstick" adds much less than the 1st
B = 0.75  # how much a long text is penalised: 0 = not at all, 1 = fully by length


def tokenize(text: str) -> list[str]:
    """Lower case, split on anything that is not a letter or digit. No stemming, like Elasticsearch's
    standard analyzer: "lipsticks" and "lipstick" are different words."""
    return re.findall(r"[a-z0-9]+", text.lower())


def item_text(item: dict) -> str:
    """The fields both BM25 and the text side of dense search index (A5 in docs/PLAN.md)."""
    parts = [item.get("brand"), item["title"], *item["categories"], *item["features"], item.get("description")]
    return " ".join(p for p in parts if p)


class BM25:
    def __init__(self, ids: list[str], texts: list[str], k1: float = K1, b: float = B):
        self.ids, self.k1, self.b = ids, k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)  # word -> [(doc, count)]
        self.lengths = []
        for i, text in enumerate(texts):
            counts = Counter(tokenize(text))
            self.lengths.append(sum(counts.values()))
            for word, tf in counts.items():
                self.postings[word].append((i, tf))
        self.avg_len = sum(self.lengths) / len(self.lengths)

    @classmethod
    def from_items(cls, items: list[dict]) -> "BM25":
        return cls([it["id"] for it in items], [item_text(it) for it in items])

    def idf(self, word: str) -> float:
        n = len(self.postings.get(word, ()))
        if n == 0:
            return 0.0
        return math.log(1 + (len(self.ids) - n + 0.5) / (n + 0.5))

    def term_score(self, idf: float, tf: int, doc: int) -> float:
        norm = 1 - self.b + self.b * self.lengths[doc] / self.avg_len
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)

    def search(self, query: str, k: int = 10, with_scores: bool = False) -> list:
        """Top k item ids with a score above 0; ties go to the earlier item, so runs are reproducible.
        with_scores=True returns (id, score) pairs instead."""
        scores: dict[int, float] = defaultdict(float)
        for word in set(tokenize(query)):  # a word typed twice counts once
            idf = self.idf(word)
            for doc, tf in self.postings.get(word, ()):
                scores[doc] += self.term_score(idf, tf, doc)
        best = sorted(scores, key=lambda d: (-scores[d], d))[:k]
        if with_scores:
            return [(self.ids[d], scores[d]) for d in best]
        return [self.ids[d] for d in best]
