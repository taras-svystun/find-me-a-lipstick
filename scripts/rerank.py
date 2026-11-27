"""A8: reranking. The hybrid (A7) finds `depth` candidates; a cross-encoder rescores each one and the
list is reordered by that score.

A bi-encoder (SigLIP2, A6) encodes the query and the item apart, so item vectors are computed once
and a search is a dot product. A cross-encoder reads "query [SEP] item text" as one input: every
query word attends to every item word, and the output is one number. Nothing can be precomputed,
so each (query, item) pair costs a full forward pass; that is why it only sees the top `depth`.

Usage:
  .venv/bin/python scripts/rerank.py --tune   # model and depth on the dev queries
"""

import json
import os
import sys
import time
from functools import lru_cache
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))  # downloaded once (README), then used offline
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import hybrid  # noqa: E402
from bm25 import item_text  # noqa: E402  (the same fields BM25 indexes; the cross-encoder sees up to MAX_TOKENS)
from dense import normalize_text  # noqa: E402

MODELS = {"minilm": "cross-encoder/ms-marco-MiniLM-L-6-v2", "bge": "BAAI/bge-reranker-base"}
MODEL = "minilm"  # set from --tune on dev
DEPTH = 50  # the plan's value; --tune checks 10-100 on dev
MAX_TOKENS = 256  # query + item text; the dense text encoder stops at 64
DEPTH_GRID = [10, 20, 50, 100]


class Scorer:
    """A cross-encoder on CPU: (query, texts) -> one relevance score per text."""

    def __init__(self, model: str):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(MODELS[model])
        self.model = AutoModelForSequenceClassification.from_pretrained(MODELS[model]).eval()
        # transformers maps the weight file into memory without a copy. MiniLM's file puts every tensor
        # 2 bytes off a 4-byte boundary, and float maths on such addresses crashes on the M1 (SIGBUS) or
        # gives NaN. A copy moves the weights to aligned memory (docs/journal.md, A8).
        for p in self.model.parameters():
            p.data = p.data.clone()

    def __call__(self, query: str, texts: list[str]) -> list[float]:
        with self.torch.no_grad():
            x = self.tokenizer([query] * len(texts), texts, truncation="only_second", max_length=MAX_TOKENS,
                               padding=True, return_tensors="pt")
            return self.model(**x).logits[:, 0].tolist()


@lru_cache(maxsize=None)
def get_scorer(model: str) -> Scorer:
    return Scorer(model)


class Rerank:
    def __init__(self, first, scorer, texts: dict[str, str], depth: int = DEPTH):
        self.first, self.scorer, self.texts, self.depth = first, scorer, texts, depth

    @classmethod
    def from_items(cls, items: list[dict], model: str = MODEL) -> "Rerank":
        return cls(hybrid.RRF.from_items(items), get_scorer(model), {it["id"]: item_text(it) for it in items})

    def order(self, query: str, ids: list[str]) -> list[tuple[str, float]]:
        """Candidates sorted by cross-encoder score; ties keep the first-stage order (stable sort)."""
        scores = self.scorer(normalize_text(query), [self.texts[d] for d in ids])
        return sorted(zip(ids, scores), key=lambda p: -p[1])

    def search(self, query: str, k: int = 10, with_scores: bool = False) -> list:
        best = self.order(query, self.first.search(query, self.depth))[:k]
        return best if with_scores else [d for d, _ in best]


def tune(items: list[dict], queries: list[dict]) -> None:
    """Dev nDCG@10 (all), p@10 (no brand), source@10 (all) and median ms per model and depth.
    A pair's score does not depend on the other candidates, so each query's top 100 is scored once
    and every depth reuses those scores; latency is timed apart, on the first 10 dev queries."""
    from metrics import evaluate
    dev = [q for q in queries if q["split"] == "dev"]
    texts = {it["id"]: item_text(it) for it in items}
    first = hybrid.RRF.from_items(items)
    cands = {q["qid"]: first.search(q["query"], max(DEPTH_GRID)) for q in dev}
    r = evaluate({qid: c[:10] for qid, c in cands.items()}, queries, "dev")
    print("model   depth  nDCG@10 all  p@10 no brand  source@10 all  median ms (rerank only)")
    print(f"{'hybrid':<15} {r['all']['ndcg@10']:<12.3f} {r['no brand']['p@10']:<14.3f} {r['all']['source@10']:.3f}")
    for model in MODELS:
        rr = Rerank(first, get_scorer(model), texts)
        ordered = {}
        for q in dev:
            scored = dict(rr.order(q["query"], cands[q["qid"]]))
            ordered[q["qid"]] = [scored[d] for d in cands[q["qid"]]]  # back in first-stage order
        for depth in DEPTH_GRID:
            run = {}
            for qid, c in cands.items():
                top = sorted(zip(c[:depth], ordered[qid][:depth]), key=lambda p: -p[1])
                run[qid] = [d for d, _ in top][:10]
            times = []
            for q in dev[:10]:
                start = time.perf_counter()
                rr.order(q["query"], cands[q["qid"]][:depth])
                times.append((time.perf_counter() - start) * 1000)
            r = evaluate(run, queries, "dev")
            print(f"{model:<7} {depth:<6} {r['all']['ndcg@10']:<12.3f} {r['no brand']['p@10']:<14.3f} "
                  f"{r['all']['source@10']:<14.3f} {median(times):.0f}")


def main() -> None:
    if "--tune" in sys.argv:
        items = [json.loads(line) for line in open(ROOT / "data" / "items.jsonl")]
        tune(items, [json.loads(line) for line in open(ROOT / "eval" / "queries.jsonl")])


if __name__ == "__main__":
    main()
