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
```
