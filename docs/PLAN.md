# find-me-a-lipstick: plan

Agreed with Taras on 2026-11-20 and 2026-11-26.

## Goal
A proof-of-work project for product search. The point is not that Taras built it. The point is
that he **understands every part and can defend it with his own numbers**: Claude Code builds
it, and Taras gets grilled on the theory.

It covers: computer vision and photo search, hybrid search, reranking, caching, model routing,
and "agent or deterministic pipeline".

Private context (who it is for, dates, test-query sources) lives outside this public repo,
see `CLAUDE.md`.

## Constraints
- $0. Everything runs locally on a MacBook M1 with open models. A paid API is allowed (max
  $1-2) only if it gives a big jump in the result. Current verdict: no paid API.
- Public repo. Nothing from the private context enters it.
- Only what is needed. No extra features.

## Dataset (checked 2026-11-20, size updated 2026-11-26)
- Use **Amazon Reviews 2023, `raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl`**
  (Hugging Face `McAuley-Lab/Amazon-Reviews-2023`, 2.8 GB). Take only the first ~135 MB with an
  HTTP range request, then keep Skin Care, Makeup and Fragrance (~35% of records), for ~15k items.
- In a sample of 1,294 items: category 100%, features 78%, brand 82%, image 100%,
  description 51%, price 52%. Top categories: Hair Care, Skin Care, Makeup, Fragrance.
- Rejected: `meta_All_Beauty.jsonl`. In a sample of 2,053 items: category 0%, description
  20%, and off-topic items such as leather conditioner.
- Images are on `m.media-amazon.com`. The sandbox blocks this host, so Taras runs the bulk
  image download himself (one command).
- Free relevance labels for evaluation: category and brand. Test queries come from a local
  Qwen model.

## Steps (in order; do B and C only if there is time)
1. **A. Hybrid text + photo search.** BM25 (keyword search) + CLIP/SigLIP embeddings (text
   and photos in one vector space) + RRF (merge two ranked lists by rank) + a cross-encoder
   reranker on the top 50. Deliverable: an ablation table (BM25 / vectors / hybrid /
   hybrid + reranker) with recall@10, nDCG@10 and latency.
   Check: the table is reproducible with one command.
2. **B. Mini-assistant on top of A.** RAG assistant with a local LLM: query rewriting, metadata
   filters (brand, price), structured outputs, tool calling, a semantic cache, model routing
   (cheap model for easy queries), Langfuse tracing, LLM-as-judge evaluation.
   Check: cache hit rate and the routing split are measured on the test queries.
3. **C. Fine-tune the embedding model or the reranker** on synthetic pairs.
   Check: a before/after row in the ablation table.

## How Taras learns each part (max ~20 minutes per part)
Breadth over depth: teach only the ideas an interviewer would ask about (mechanisms and
trade-offs). Plumbing steps (download, parsing) get no lesson.
1. A toy example of about 5 lines, with only what is needed to understand it.
2. Taras guesses the result first (the prediction effect).
3. A trap example that looks simple but behaves in a surprising way.
4. One question the Team Lead would ask.
Before every number in the ablation table, Taras guesses it first.

## Decisions (2026-11-26)
- Step A is done in any case: it closes the biggest gaps, so he can say "I tried this, here is
  what I measured", and show the demo if it is ready.
- Local LLM: Ollama or llama.cpp, either is fine.
- ~15k items. Learning time per day: no fixed limit.

## Step A in small steps (2026-11-26)
Each step adds code and a check; steps with an interview-relevant idea also get a lesson. Every
ablation row also records latency (median ms per query on the M1). Traps are not written here,
so Taras can guess them.

- ✅ **A1. Data slice.** Range-request the first ~135 MB, parse it, keep Skin Care, Makeup and
  Fragrance, write `data/items.jsonl`.
  Check: item count and field coverage printed; coverage close to the 2026-11-20 sample.
- ✅ **A2. Images.** A script writes the URL list; Taras downloads it with one command.
  Check: one non-empty image per item with an image URL, missing ones listed.
- **A3. Test queries and relevance labels.** A local Qwen (Ollama) writes ~200 shopper-style
  queries from sampled items; labels come from category and brand.
  Check: query count; Taras and Claude spot-check 10 queries and their labels.
  Lesson: how a label design can favour one system over another.
- **A4. Metrics.** recall@10 and nDCG@10, with tests on hand-made rankings.
  Check: tests pass on cases computed by hand.
  Lesson: nDCG on a 5-item toy ranking.
- **A5. BM25 row.** Check: row in the table. Lesson: BM25 term weights on a toy corpus.
- **A6. Vectors row.** CLIP/SigLIP embeddings of item photos and item text; the model is
  chosen in this step. Check: row in the table. Lesson: one space for text and photos.
- **A7. Hybrid row.** RRF over BM25 and vectors. Check: row in the table. Lesson: RRF and its `k`.
- **A8. Reranker row.** Cross-encoder over the hybrid top 50. Check: row in the table.
  Lesson: cross-encoder versus bi-encoder, and what it costs in latency.
- **A9. One command.** `make ablation` (or one Python command) rebuilds the whole table.
  Check: run from a clean state gives the same numbers; README has the command.
