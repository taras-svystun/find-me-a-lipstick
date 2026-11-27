"""A3: write shopper-style test queries with a local Qwen (Ollama) and label them from category and brand.

Usage:
  ollama serve                                     # if Ollama is not already running
  .venv/bin/python scripts/make_queries.py         # writes eval/queries.jsonl (~200 queries)
  .venv/bin/python scripts/make_queries.py --show 10   # prints 10 random queries for a spot check
  .venv/bin/python scripts/make_queries.py --relabel   # recomputes labels after a change to labels()

The queries are committed (eval/queries.jsonl): an LLM does not give the same text twice on another
machine or model version, so the file itself is the test set, not this script. After a label
change, rerun only --relabel; a plain run writes a new, different set of queries.
"""

import json
import random
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

from bm25 import item_text

ROOT = Path(__file__).resolve().parent.parent
ITEMS = ROOT / "data" / "items.jsonl"
OUT = ROOT / "eval" / "queries.jsonl"

MODEL = "qwen3.5:9b"
OLLAMA = "http://localhost:11434/api/chat"
N_QUERIES = 200
BRAND_SHARE = 0.25  # share of queries that name the brand, like a shopper who knows what they want
MIN_LEAF = 20  # skip leaf categories with fewer items: too few relevant items to measure anything
SKIP_LEAVES = {"Sets & Kits"}  # a mix of anything, so "same category" says nothing about relevance
SKIP_SOURCES = {"B071QWWMH1", "B078VJ83ZY"}  # misfiled by Amazon: a shampoo in Body Butters, a knife in Eyeliner
SEED = 0
MAX_WORDS = 12  # a longer answer is a keyword dump, not a shopper's query: ask again
TRIES = 3
# Amazon files the same bottle under any of these leaves (a "cologne" query can come from an
# Eau de Toilette), so for labels they are one shelf. Fragrance "Sets" stay apart: a gift box.
FRAGRANCE_LEAVES = {"Cologne", "Eau de Parfum", "Eau de Toilette", "Body Sprays", "Eau Fraiche"}
N_DEV = 50  # the first 50 queries tune choices (model, RRF k, rerank depth); the other 150 fill the table

PROMPT = """You write one search query that a real shopper would type into a beauty store's search box
to find a product like this one.

Product:
- title: {title}
- category: {path}
- details: {details}

Rules:
- {brand_rule}
- Name the product type and one or two things this shopper cares about most. Describe the need,
  not this exact product. Use your own everyday words, not phrases copied from the title.
- 3 to 10 words, lower case, no quotes. Short, like a real search box.
Return JSON: {{"query": "..."}}"""


def load_items() -> list[dict]:
    return [json.loads(line) for line in open(ITEMS)]


def leaf(item: dict) -> str:
    return item["categories"][-1]


def label_group(item: dict) -> str:
    """The shelf used for labels: the leaf category, with the fragrance leaves merged."""
    return "Fragrance" if leaf(item) in FRAGRANCE_LEAVES else leaf(item)


def norm_brand(brand: str | None) -> str:
    """'K i e h l's', "Kiehl's" and 'KIEHLS' all become 'kiehls'; 'Tatum & Shea' becomes 'tatumandshea'."""
    return re.sub(r"[^a-z0-9]", "", (brand or "").lower().replace("&", "and"))


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def title_overlap(query: str, title: str) -> float:
    """Share of query words that also appear in the source title."""
    q = words(query)
    return round(len(q & words(title)) / len(q), 2) if q else 0.0


def text_overlap(query: str, item: dict) -> float:
    """Share of query words found anywhere in the text BM25 indexes for the item. The prompt showed
    Qwen the category path, features and description too, so copying is not only from the title."""
    return title_overlap(query, item_text(item))


def sample_sources(items: list[dict], rng: random.Random) -> list[dict]:
    """Pick source items spread over leaf categories, in proportion to their size."""
    by_leaf = defaultdict(list)
    for it in items:
        if len(it["categories"]) >= 3 and leaf(it) not in SKIP_LEAVES and it["id"] not in SKIP_SOURCES:
            by_leaf[leaf(it)].append(it)
    pool = [it for group in by_leaf.values() if len(group) >= MIN_LEAF for it in group]
    return rng.sample(pool, N_QUERIES + 50)  # spares replace sources whose queries stay too long


def ask_qwen(prompt: str, seed: int) -> str:
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "format": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]},
        "think": False,
        "stream": False,
        "options": {"temperature": 0.7, "seed": seed},
    }
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(json.loads(r.read())["message"]["content"])["query"].strip().lower()


def labels(query: str, source: dict, items: list[dict]) -> dict[str, int]:
    """Graded relevance: 2 for the source item, 1 for items in its label group
    (and of its brand, when the query names the brand)."""
    brand = norm_brand(source["brand"])
    named = bool(brand) and brand in norm_brand(query)
    grades = {
        it["id"]: 1
        for it in items
        if len(it["categories"]) >= 3
        and label_group(it) == label_group(source)
        and (not named or norm_brand(it["brand"]) == brand)
    }
    grades[source["id"]] = 2
    return grades


def make() -> None:
    rng = random.Random(SEED)
    items = load_items()
    OUT.parent.mkdir(exist_ok=True)
    with open(OUT, "w") as f:
        n = 0
        for k, src in enumerate(sample_sources(items, rng)):
            if n == N_QUERIES:
                break
            wants_brand = bool(src["brand"]) and rng.random() < BRAND_SHARE
            brand_rule = (
                f'Include the brand "{src["brand"]}".' if wants_brand else "Do not name any brand."
            )
            details = "; ".join(src["features"][:3])[:400] or (src["description"] or "")[:400]
            prompt = PROMPT.format(
                title=src["title"], path=" > ".join(src["categories"]), details=details, brand_rule=brand_rule
            )
            for t in range(TRIES):
                query = ask_qwen(prompt, seed=SEED + 1000 * t + k)
                if len(query.split()) <= MAX_WORDS:
                    break
            else:
                print(f"skip {src['id']}: still {len(query.split())} words after {TRIES} tries")
                continue
            grades = labels(query, src, items)
            named = bool(src["brand"]) and norm_brand(src["brand"]) in norm_brand(query)
            row = {"qid": n, "split": "dev" if n < N_DEV else "test", "query": query,
                   "source_id": src["id"], "category": label_group(src), "brand_in_query": named,
                   "title_overlap": title_overlap(query, src["title"]),
                   "text_overlap": text_overlap(query, src), "relevant": grades}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"{n:3d} [{leaf(src)}] {query}  ({len(grades)} relevant)", flush=True)
            n += 1
    print(f"wrote {OUT}")


def relabel() -> None:
    """Recompute labels for the committed queries without asking Qwen again."""
    items = load_items()
    by_id = {it["id"]: it for it in items}
    rows = [json.loads(line) for line in open(OUT)]
    with open(OUT, "w") as f:
        for row in rows:
            src = by_id[row["source_id"]]
            row["category"] = label_group(src)
            row["brand_in_query"] = bool(src["brand"]) and norm_brand(src["brand"]) in norm_brand(row["query"])
            row["relevant"] = labels(row["query"], src, items)
            row["text_overlap"] = text_overlap(row["query"], src)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"relabelled {len(rows)} queries in {OUT}")


def show(k: int) -> None:
    items = {it["id"]: it for it in load_items()}
    rows = [json.loads(line) for line in open(OUT)]
    for row in random.Random(1).sample(rows, k):
        src = items[row["source_id"]]
        print(f'{row["qid"]:3d} "{row["query"]}"\n    source: {src["title"][:90]}\n'
              f'    label: {row["category"]}{" + brand " + src["brand"] if row["brand_in_query"] else ""}'
              f' -> {len(row["relevant"])} relevant; title overlap {row["title_overlap"]}')


if __name__ == "__main__":
    if "--show" in sys.argv:
        show(int(sys.argv[2]))
    elif "--relabel" in sys.argv:
        relabel()
    else:
        make()
