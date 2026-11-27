"""A7: hybrid search. Reciprocal rank fusion (RRF) merges BM25 and the dense text+photo list:

  score(d) = sum over lists L that hold d in their top `depth` of  1 / (k + rank_L(d))

Only ranks enter the sum, so BM25 points and cosines, which live on different scales, never need
to be made comparable.

Two lists, not three (BM25, text, photo): with three, the two dense lists would vote twice.

Usage:
  .venv/bin/python scripts/hybrid.py --tune   # RRF k on the dev queries
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import dense
from bm25 import BM25

ROOT = Path(__file__).resolve().parent.parent
K = 60  # the RRF paper's value (Cormack et al., 2009); dev nDCG@10 is flat over k 1-300 (0.720-0.729),
#        so the best cell (k=1) would be fitted to noise (docs/journal.md, A7)
DEPTH = 100  # items taken from each list; a deeper item gets nothing from that list
K_GRID = [1, 10, 30, 60, 100, 300]


class RRF:
    def __init__(self, engines: list, k: float = K, depth: int = DEPTH):
        self.engines, self.k, self.depth = engines, k, depth

    @classmethod
    def from_items(cls, items: list[dict], k: float = K) -> "RRF":
        return cls([BM25.from_items(items), dense.Dense.from_items(items, dense.MODEL, w=dense.W)], k=k)

    def fuse(self, lists: list[list[str]], n: int, with_scores: bool = False) -> list:
        """Ties go to the item seen first (first list, then lower rank), so runs are reproducible."""
        scores: dict[str, float] = defaultdict(float)  # dict keeps first-seen order
        for ids in lists:
            for rank, item in enumerate(ids, start=1):
                scores[item] += 1 / (self.k + rank)
        best = sorted(scores, key=lambda d: -scores[d])[:n]  # sorted() is stable
        return [(d, scores[d]) for d in best] if with_scores else best

    def search(self, query: str, k: int = 10, with_scores: bool = False) -> list:
        return self.fuse([e.search(query, self.depth) for e in self.engines], k, with_scores)


def tune(items: list[dict], queries: list[dict]) -> None:
    """Print dev nDCG@10 (all queries) and p@10 (no brand) for each k; K is set by hand from it."""
    from metrics import evaluate
    dev = [q for q in queries if q["split"] == "dev"]
    rrf = RRF.from_items(items)
    lists = {q["qid"]: [e.search(q["query"], rrf.depth) for e in rrf.engines] for q in dev}  # search once
    print("k      nDCG@10 all   p@10 no brand")
    for k in K_GRID:
        rrf.k = k
        r = evaluate({qid: rrf.fuse(ls, 10) for qid, ls in lists.items()}, queries, "dev")
        print(f"{k:<6} {r['all']['ndcg@10']:<13.3f} {r['no brand']['p@10']:.3f}")


def main() -> None:
    if "--tune" in sys.argv:
        items = [json.loads(line) for line in open(ROOT / "data" / "items.jsonl")]
        tune(items, [json.loads(line) for line in open(ROOT / "eval" / "queries.jsonl")])


if __name__ == "__main__":
    main()
