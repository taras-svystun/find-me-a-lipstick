"""A4: score a ranked list against graded relevance labels, and summarise a run over the test queries.

A run is {qid: [item ids, best first]}; queries are the rows of eval/queries.jsonl.
"""

import math
import random
from statistics import mean

K = 10
LOW_OVERLAP = 0.3  # below this share of title words, a query cannot win by copying the title
LOW_TEXT_OVERLAP = 0.7  # the same for the whole item text; 0.7 gives a slice as big as the title one
N_BOOT = 1000


def check_ranking(ranked: list[str], k: int = K) -> list[str]:
    top = ranked[:k]
    if len(set(top)) != len(top):  # a repeated item would count twice in precision
        raise ValueError(f"duplicate item ids in the top {k}: {top}")
    return top


def precision_at_k(ranked: list[str], relevant: dict[str, int], k: int = K) -> float:
    """Share of the top k that is relevant (grade 1 or 2). Divides by k even when fewer came back,
    so a system cannot score higher by returning less."""
    return sum(1 for i in check_ranking(ranked, k) if relevant.get(i, 0) > 0) / k


def dcg(grades: list[int]) -> float:
    """Discounted cumulative gain: each grade divided by log2(rank + 1), so rank 1 counts in full,
    rank 3 counts half. Linear gain (the grade itself), not 2^grade - 1: see docs/journal.md."""
    return sum(g / math.log2(rank + 1) for rank, g in enumerate(grades, start=1))


def ndcg_at_k(ranked: list[str], relevant: dict[str, int], k: int = K) -> float:
    """DCG of this top k divided by the DCG of the best possible top k, so 1.0 means perfect."""
    ideal = dcg(sorted(relevant.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return dcg([relevant.get(i, 0) for i in check_ranking(ranked, k)]) / ideal


def source_in_top_k(ranked: list[str], source_id: str, k: int = K) -> float:
    """Diagnostic only: 1.0 if the item Qwen read to write the query is in the top k."""
    return float(source_id in check_ranking(ranked, k))


SLICES = {
    "all": lambda q: True,
    "no brand": lambda q: not q["brand_in_query"],
    "no brand, no fragrance": lambda q: not q["brand_in_query"] and q["category"] != "Fragrance",
    "brand": lambda q: q["brand_in_query"],
    f"title overlap < {LOW_OVERLAP}": lambda q: q["title_overlap"] < LOW_OVERLAP,
    f"text overlap < {LOW_TEXT_OVERLAP}": lambda q: q["text_overlap"] < LOW_TEXT_OVERLAP,
}


def _percentile_ci(values: list[float], n_boot: int, seed: int) -> tuple[float, float]:
    """Resample the queries with replacement n_boot times, take the mean each time, and keep the
    middle 95% of those means: how much the score would move on another draw of queries."""
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(n_boot))
    return means[int(n_boot * 0.025)], means[int(n_boot * 0.975)]


def bootstrap_ci(scores: list[float], n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float] | None:
    """95% interval of the mean per-query score."""
    return _percentile_ci(scores, n_boot, seed) if scores else None


def paired_diff_ci(a: list[float], b: list[float], n_boot: int = N_BOOT, seed: int = 0) -> tuple[float, float]:
    """95% interval of mean(a - b), where a[i] and b[i] are two systems on the same query i.
    Resampling the differences keeps each query's difficulty on both sides, so it cancels out."""
    if len(a) != len(b):
        raise ValueError(f"paired scores need the same queries: {len(a)} against {len(b)}")
    return _percentile_ci([x - y for x, y in zip(a, b)], n_boot, seed)


def evaluate(run: dict[int, list[str]], queries: list[dict], split: str = "test") -> dict[str, dict]:
    """Mean metrics per slice. precision@10 averages only the slice's no-brand queries: a brand
    query can have one relevant item, so its precision@10 cannot pass 0.1."""
    run = {int(qid): ranked for qid, ranked in run.items()}  # a run loaded from JSON has string keys
    if not run.keys() & {q["qid"] for q in queries if q["split"] == split}:
        raise ValueError(f"the run answers none of the {split} queries: wrong keys or wrong split?")
    rows = []
    for q in queries:
        if q["split"] != split:
            continue
        ranked = run.get(q["qid"], [])  # a query the system answered with nothing scores 0
        rows.append((q, {
            "ndcg@10": ndcg_at_k(ranked, q["relevant"]),
            "source@10": source_in_top_k(ranked, q["source_id"]),
            "p@10": None if q["brand_in_query"] else precision_at_k(ranked, q["relevant"]),
        }))
    report = {}
    for name, keep in SLICES.items():
        scores = [s for q, s in rows if keep(q)]
        per_metric = {
            "p@10": [s["p@10"] for s in scores if s["p@10"] is not None],
            "ndcg@10": [s["ndcg@10"] for s in scores],
            "source@10": [s["source@10"] for s in scores],
        }
        report[name] = {"n": len(scores), "n p@10": len(per_metric["p@10"])}
        for metric, values in per_metric.items():
            ci = bootstrap_ci(values)
            report[name][metric] = round(mean(values), 3) if values else None
            report[name][f"{metric} ci"] = [round(x, 3) for x in ci] if ci else None
    return report
