# find-me-a-lipstick

Hybrid text + photo search over ~15k beauty products (Amazon Reviews 2023 metadata):
BM25 + CLIP/SigLIP vectors, merged with RRF, reranked by a cross-encoder.
Runs locally on a MacBook M1 with open models.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Run

Commands are added here as each step lands.

```sh
.venv/bin/python scripts/fetch_items.py   # A1: download ~135 MB, keep ~15.6k items -> data/items.jsonl
.venv/bin/python scripts/image_list.py    # A2: writes data/images.curl
curl --fail --parallel --parallel-max 16 -sS -K data/images.curl   # A2: download (outside the sandbox)
.venv/bin/python scripts/image_list.py --check   # A2: counts images, lists missing ones
.venv/bin/python scripts/make_queries.py   # A3: needs `ollama pull qwen3.5:9b`; writes eval/queries.jsonl (~20 min)
.venv/bin/python scripts/make_queries.py --show 10   # A3: spot-check 10 queries and their labels
.venv/bin/python scripts/make_queries.py --relabel   # A3: recompute labels for the committed queries
.venv/bin/pytest -q   # all tests (A4: metrics on hand-computed cases)
```
