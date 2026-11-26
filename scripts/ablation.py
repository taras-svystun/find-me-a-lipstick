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


COLUMNS = [  # (header, slice, metric)
    ("p@10 no brand", "no brand", "p@10"),
    ("p@10 no brand, no fragrance", "no brand, no fragrance", "p@10"),
    ("nDCG@10 no brand", "no brand", "ndcg@10"),
    ("nDCG@10 brand", "brand", "ndcg@10"),
    ("source@10 all", "all", "source@10"),
    ("p@10 overlap < 0.3", "title overlap < 0.3", "p@10"),
    ("nDCG@10 overlap < 0.3", "title overlap < 0.3", "ndcg@10"),
]


def to_markdown(table: dict, split: str) -> str:
    first = next(iter(table.values()))
    sizes = ", ".join(f'{s} {first[s]["n"]}' for s in first if isinstance(first[s], dict))
    lines = [f"Split: {split}. Queries per slice: {sizes}.", "",
             "| system | " + " | ".join(h for h, _, _ in COLUMNS) + " | median ms |",
             "|---" * (len(COLUMNS) + 2) + "|"]
    for name, row in table.items():
        cells = [f"{row[s][m]:.3f}" for _, s, m in COLUMNS]
        lines.append(f"| {name} | " + " | ".join(cells) + f' | {row["latency_ms"]} |')
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
