"""The ablation table: one row per search system, scored on the test queries.

Usage:
  .venv/bin/python scripts/ablation.py              # all systems on the 149 test queries -> eval/ablation.md
  .venv/bin/python scripts/ablation.py --split dev  # the 49 dev queries, for tuning (not written to disk)
"""

import json
import sys
import time
from pathlib import Path
from statistics import median

import dense
import hybrid
from bm25 import BM25
from metrics import SLICES, evaluate, paired_diff_ci, per_query

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = {  # name -> build(items) -> object with .search(query, k)
    "bm25": BM25.from_items,
    "vectors text": lambda items: dense.Dense.from_items(items, dense.MODEL, w=1.0),
    "vectors photo": lambda items: dense.Dense.from_items(items, dense.MODEL, w=0.0),
    "vectors text+photo": lambda items: dense.Dense.from_items(items, dense.MODEL, w=dense.W),
    "hybrid": hybrid.RRF.from_items,
}


def run_systems(items: list[dict], queries: list[dict], systems: list[str], split: str = "test") -> tuple[dict, dict]:
    """Each system's top 10 per query of the split, and its median latency in ms."""
    runs, latency = {}, {}
    for name in systems:
        engine = SYSTEMS[name](items)
        run, times = {}, []
        for q in queries:
            if q["split"] != split:
                continue
            start = time.perf_counter()
            run[q["qid"]] = engine.search(q["query"], 10)
            times.append((time.perf_counter() - start) * 1000)
        runs[name], latency[name] = run, round(median(times), 2)
    return runs, latency


def build_table(items: list[dict], queries: list[dict], systems: list[str], split: str = "test") -> dict:
    runs, latency = run_systems(items, queries, systems, split)
    return {n: {**evaluate(runs[n], queries, split), "latency_ms": latency[n]} for n in systems}


ROWS = [  # (label, slice, metric): one row per number, one column per system
    ("p@10 no brand", "no brand", "p@10"),
    ("p@10 no brand, no fragrance", "no brand, no fragrance", "p@10"),
    ("nDCG@10 no brand", "no brand", "ndcg@10"),
    ("nDCG@10 brand", "brand", "ndcg@10"),
    ("source@10 all", "all", "source@10"),
    ("p@10 text overlap < 0.7", "text overlap < 0.7", "p@10"),
    ("source@10 text overlap < 0.7", "text overlap < 0.7", "source@10"),
]


PAIRS = [  # (a, b): rows compared query by query, a - b
    ("vectors text+photo", "bm25"),
    ("vectors photo", "vectors text"),
    ("vectors text+photo", "vectors photo"),
    ("hybrid", "bm25"),
    ("hybrid", "vectors text+photo"),
]
PAIR_ROWS = [r for r in ROWS if r[0] in
             {"p@10 no brand", "nDCG@10 no brand", "source@10 all", "source@10 text overlap < 0.7"}]


def paired_rows(runs: dict, queries: list[dict], pairs: list[tuple[str, str]], split: str = "test") -> dict:
    """Mean of a - b per query and its 95% paired bootstrap interval, for each pair and PAIR_ROWS metric."""
    scores = {n: per_query(run, queries, split) for n, run in runs.items()}
    out = {}
    for a, b in pairs:
        out[f"{a} - {b}"] = {}
        for label, s, m in PAIR_ROWS:
            keep = [i for i, (q, sc) in enumerate(scores[a]) if SLICES[s](q) and sc[m] is not None]
            x = [scores[a][i][1][m] for i in keep]
            y = [scores[b][i][1][m] for i in keep]
            out[f"{a} - {b}"][label] = {"diff": round(sum(x) / len(x) - sum(y) / len(y), 3),
                                        "ci": [round(v, 3) for v in paired_diff_ci(x, y)], "n": len(x)}
    return out


def pairs_markdown(pairs: dict) -> str:
    """One row per pair; an interval that excludes 0 means the difference is more than query noise."""
    labels = [r[0] for r in PAIR_ROWS]
    lines = ["Paired differences (a - b, same queries, 95% bootstrap interval):", "",
             "| pair | " + " | ".join(labels) + " |", "|---" * (len(labels) + 1) + "|"]
    for name, cells in pairs.items():
        lines.append(f"| {name} | " + " | ".join(
            f'{cells[l]["diff"]:+.3f} [{cells[l]["ci"][0]:+.3f}, {cells[l]["ci"][1]:+.3f}]' for l in labels) + " |")
    return "\n".join(lines) + "\n"


def to_markdown(table: dict, split: str) -> str:
    """Each cell is the mean and its 95% bootstrap interval over the slice's queries."""
    first = next(iter(table.values()))
    sizes = ", ".join(f'{s} {r["n"]}' + (f' (p@10 on {r["n p@10"]})' if r["n p@10"] != r["n"] else "")
                      for s, r in first.items() if isinstance(r, dict))
    names = list(table)
    lines = [f"Split: {split}. Queries per slice: {sizes}.", "",
             "| metric | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
    for label, s, m in ROWS + [("median ms", None, None)]:
        if s is None:
            cells = [str(table[n]["latency_ms"]) for n in names]
        else:
            cells = [f'{table[n][s][m]:.3f} [{table[n][s][m + " ci"][0]:.3f}, {table[n][s][m + " ci"][1]:.3f}]'
                     for n in names]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    split = sys.argv[sys.argv.index("--split") + 1] if "--split" in sys.argv else "test"
    items = [json.loads(line) for line in open(ROOT / "data" / "items.jsonl")]
    queries = [json.loads(line) for line in open(ROOT / "eval" / "queries.jsonl")]
    runs, latency = run_systems(items, queries, list(SYSTEMS), split)
    table = {n: {**evaluate(runs[n], queries, split), "latency_ms": latency[n]} for n in SYSTEMS}
    pairs = paired_rows(runs, queries, PAIRS, split)
    text = to_markdown(table, split) + "\n" + pairs_markdown(pairs)
    print(text)
    if split == "test":
        (ROOT / "eval" / "ablation.md").write_text(text)
        (ROOT / "eval" / "ablation.json").write_text(json.dumps({"rows": table, "paired": pairs}, indent=1) + "\n")


if __name__ == "__main__":
    main()
