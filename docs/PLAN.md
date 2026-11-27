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
- Free relevance labels for evaluation: leaf category, plus brand when the query names it.
  Test queries come from a local Qwen model.

## Steps (in order; do B and C only if there is time)
1. **A. Hybrid text + photo search.** BM25 (keyword search) + CLIP/SigLIP embeddings (text
   and photos in one vector space) + RRF (merge two ranked lists by rank) + a cross-encoder
   reranker on the top 50. Deliverable: an ablation table (BM25 / vectors / hybrid /
   hybrid + reranker) with precision@10, nDCG@10 and latency, on the 150 test queries.
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

## Decisions (2026-11-26, A3)
- precision@10 replaces recall@10: a query without a brand has 31-1,640 relevant items, so recall@10 could never
  exceed 10/31 and mostly stays near 0.01.
- Dev/test split: model choice (A6), RRF `k` (A7) and rerank depth (A8) are tuned on the dev set
  only, so the table is not fitted to its own test.
- Brand-named queries are reported as a separate group: their relevant set can be one item.

## Step A in small steps (2026-11-26)
Each step adds code and a check; steps with an interview-relevant idea also get a lesson. Every
ablation row also records latency (median ms per query on the M1). Traps are not written here,
so Taras can guess them.

- ✅ **A1. Data slice.** Range-request the first ~135 MB, parse it, keep Skin Care, Makeup and
  Fragrance, write `data/items.jsonl`.
  Check: item count and field coverage printed; coverage close to the 2026-11-20 sample.
- ✅ **A2. Images.** A script writes the URL list; Taras downloads it with one command.
  Check: one non-empty image per item with an image URL, missing ones listed.
- ✅ **A3. Test queries and relevance labels.** A local Qwen (`qwen3.5:9b`, Ollama) writes 200
  shopper-style queries, one per sampled source item, 25% of them naming the brand. Labels are
  graded: 2 for the source item, 1 for its label group (the leaf category, with the five perfume
  leaves merged into "Fragrance") and brand, when named. Items with a
  category path shorter than 3 are left out. Two queries with a misfiled source item were dropped, so 198
  remain: qids 1-49 are the dev set (49, for tuning choices), qids 50-199 the test set (149, for the table). Queries are committed in `eval/queries.jsonl`.
  Check: query count; Taras and Claude spot-check 10 queries and their labels.
  Lesson: how a label design can favour one system over another.
- ✅ **A4. Metrics.** precision@10 and nDCG@10 (graded), plus "source item in top 10" as a
  diagnostic only, with tests on hand-made rankings.
  precision@10 is reported for no-brand queries only (a brand query can have one relevant item,
  so its precision@10 cannot pass 0.1); brand queries get nDCG@10 and "source item in top 10".
  Every row is also reported on queries with `title_overlap` below 0.3, to show how much of
  BM25's score comes from words the query copied from the title. no-brand precision@10 is also
  shown without the 22 Fragrance queries, whose 1,640 relevant items make it near 1.0 for any system.
  Every cell carries a 95% bootstrap interval (resample the queries 1,000 times); two rows are
  compared with a paired bootstrap on the same queries. A second low-overlap slice uses the whole
  item text (`text_overlap` < 0.7), since Qwen also saw the category path, features and description.
  Check: tests pass on cases computed by hand.
  Lesson: nDCG on a 5-item toy ranking.
- ✅ **A5. BM25 row.** BM25 and the text side of dense search index the same fields (title,
  categories, features, description, plus brand: it is missing from 4k of 15.6k titles), so
  the rows differ by method, not by input (except that the dense text encoder cuts
  the text at 64-77 tokens, so it mostly sees brand, title and categories). Lucene's BM25 (k1 1.2, b 0.75), no stemming.
  Check: row in the table. Lesson: BM25 term weights on a toy corpus.
- ✅ **A6. Vectors row.** CLIP/SigLIP embeddings of item photos and item text; the model is
  chosen in this step. Three rows: vectors text, vectors photo, vectors text+photo (z-scored mix,
  text weight w). Model and w chosen on dev: SigLIP2, w 0.3. Photo queries are left for the demo.
  Check: rows in the table. Lesson: one space for text and photos.
- **A7. Hybrid row.** RRF over BM25 and vectors. Check: row in the table. Lesson: RRF and its `k`.
- **A8. Reranker row.** Cross-encoder over the hybrid top 50. Check: row in the table.
  Lesson: cross-encoder versus bi-encoder, and what it costs in latency.
- **A9. One command.** `make ablation` (or one Python command) rebuilds the whole table.
  Check: run from a clean state gives the same numbers; README has the command.
