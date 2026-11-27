# find-me-a-lipstick

Hybrid text + photo search over ~15k beauty products (Amazon Reviews 2023 metadata):
BM25 + CLIP/SigLIP vectors, merged with RRF, reranked by a cross-encoder.
Runs locally on a MacBook M1 with open models.

## Setup

```sh
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# A6 models (~2 GB) into models/; scripts then run offline
HF_HOME=models .venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('openai/clip-vit-base-patch32', allow_patterns=['*.json', '*.txt', 'pytorch_model.bin']); d('google/siglip2-base-patch16-224', allow_patterns=['*.json', '*.model', '*.safetensors'])"
# A8 cross-encoders (~1.2 GB)
HF_HOME=models .venv/bin/python -c "from huggingface_hub import snapshot_download as d; d('cross-encoder/ms-marco-MiniLM-L-6-v2', allow_patterns=['*.json', '*.txt', 'model.safetensors']); d('BAAI/bge-reranker-base', allow_patterns=['*.json', '*.model', 'model.safetensors'])"
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
git config core.hooksPath hooks   # once: turn on the pre-commit hook (tests-first check)
.venv/bin/pytest -q   # all tests (A4: metrics on hand-computed cases)
.venv/bin/python scripts/dense.py --embed clip      # A6: item text and photo vectors -> data/vectors/ (~12 min on CPU)
.venv/bin/python scripts/dense.py --embed siglip2   # A6: the same for SigLIP2 (~25 min on CPU)
.venv/bin/python scripts/dense.py --tune            # A6: model and text weight on the dev queries
.venv/bin/python scripts/hybrid.py --tune           # A7: RRF k on the dev queries
.venv/bin/python scripts/rerank.py --tune           # A8: cross-encoder model and rerank depth on the dev queries
.venv/bin/python scripts/a8_trap.py                  # A8: lost source items the reranker brings back (test)
.venv/bin/python scripts/ablation.py   # A5+: the ablation table on the test queries -> eval/ablation.md
.venv/bin/python scripts/ablation.py --split dev   # the same on the dev queries, for tuning
```
