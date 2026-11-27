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

from bm25 import BM25
from metrics import evaluate

ROOT = Path(__file__).resolve().parent.parent
SYSTEMS = {"bm25": BM25.from_items}  # name -> build(items) -> object with .search(query, k)


def build_table(items: list[dict], queries: list[dict], systems: list[str], split: str = "test") -> dict:
    table = {}
    for name in systems:
        engine = SYSTEMS[name](items)
        run, times = {}, []
        for q in queries:
            if q["split"] != split:
                continue
            start = time.perf_counter()
            run[q["qid"]] = engine.search(q["query"], 10)
            times.append((time.perf_counter() - start) * 1000)
        table[name] = {**evaluate(run, queries, split), "latency_ms": round(median(times), 2)}
    return table


ROWS = [  # (label, slice, metric): one row per number, one column per system
    ("p@10 no brand", "no brand", "p@10"),
    ("p@10 no brand, no fragrance", "no brand, no fragrance", "p@10"),
    ("nDCG@10 no brand", "no brand", "ndcg@10"),
    ("nDCG@10 brand", "brand", "ndcg@10"),
    ("source@10 all", "all", "source@10"),
    ("p@10 title overlap < 0.3", "title overlap < 0.3", "p@10"),
    ("source@10 title overlap < 0.3", "title overlap < 0.3", "source@10"),
    ("p@10 text overlap < 0.7", "text overlap < 0.7", "p@10"),
    ("source@10 text overlap < 0.7", "text overlap < 0.7", "source@10"),
]


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
    table = build_table(items, queries, list(SYSTEMS), split)
    text = to_markdown(table, split)
    print(text)
    if split == "test":
        (ROOT / "eval" / "ablation.md").write_text(text)
        (ROOT / "eval" / "ablation.json").write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main()
