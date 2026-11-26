# Journal

Newest at the bottom.

- 2026-11-20, research: dataset choice. Verdict: Amazon Reviews 2023 `meta_Beauty_and_Personal_Care.jsonl`,
  not `meta_All_Beauty.jsonl` (category 0%, off-topic items). Source: samples of 1,294 and 2,053 items, see `docs/PLAN.md`.
- 2026-11-26, research: can the sandbox fetch the dataset? Verdict: yes. A range request to huggingface.co
  redirects to `us.aws.cdn.hf.co` and returns 206 Partial Content. Images on `m.media-amazon.com` stay blocked.
- 2026-11-26, setup: CLAUDE.md, GLOSSARY.md, docs/PLAN.md (was `project_plan.md`), journal, README, .gitignore.
- 2026-11-26, A1 data slice: `scripts/fetch_items.py` range-requests the first 135 MB (44,224 records)
  and keeps 15,624 items (Skin Care 7,953, Makeup 5,611, Fragrance 2,060). Coverage: brand 97% (falls
  back to `store`), features 76%, description 64%, price 48%, image 100%. Taras guessed 10k kept from
  16k records in the 50 MB test slice; the truth was 35%. The CDN can close the connection early,
  so the download resumes where it stopped.
- 2026-11-26, research: is the head of the file a fair sample? Verdict: for categories yes, for text
  length not quite. Five 5 MB slices at 0/25/50/75/95% of the file keep 32-39% of records, with the
  same top categories, but hold 1,612 records at the start versus 2,046 at the end, so later records
  are shorter. The head is a bit richer in text than the whole catalogue. Source: range requests, this session.
- 2026-11-26, privacy: the private context moved to ask-bro, and the public repo carries none of it.
  Pseudonyms were rejected: the context is easy to decode, and git history is permanent.
- 2026-11-26, A2 images: Taras downloaded 15,624 main images (369 MB) with one parallel curl. 19 are
  60x40 "no photo" placeholders (13 GIF, 6 JPEG) that Amazon serves instead of a 404; `--check`
  detects them by pixel size, since 12 real photos are just as small in bytes (~1-2 KB thumbnails).
  Result: 15,605 real photos. 59 URLs are shared by several items. Items without a photo are handled in A6.
- 2026-11-26, review of plan + A1-A2: two blockers. (1) Labels "category and brand" would either
  leave one relevant item per query (brand + full path: median group size 1) or cap recall@10 near
  0.01 (leaf alone: 31-836 items). Fixed in A3 below. (2) The words "interview" and "Team Lead" are
  in the public repo; Taras decided they may stay, no history rewrite. Also fixed: `--check` no
  longer crashes on an unreadable image, curl gets `--fail`.
- 2026-11-26, A3 test queries: `scripts/make_queries.py` asks `qwen3.5:9b` (Ollama, ~5 s per query on
  the M1) for one query per sampled source item and writes `eval/queries.jsonl` (committed: an LLM
  will not give the same text twice). 200 queries, 50 dev + 150 test, 56 name the brand, 5-12 words
  (median 8). Labels: 2 for the source item, 1 for its leaf category (and normalised brand when
  named); 33 brand queries have one relevant item, so brand queries are reported apart. Mean share of
  query words found in the source title: 0.47 without brand, 0.55 with.
  First run failed: 61 of 200 queries were over 10 words, 8 were 30-118-word keyword dumps. Cause:
  the prompt listed six attributes and the model filled them all ("eau de parfum for dry skin").
  Fix: ask for one or two needs, and retry any query over 12 words.
  Review fixes: brand "Tatum & Shea" was not matched in "tatum and shea" (698 relevant instead of 1);
  `norm_brand` now maps "&" to "and", and `--relabel` recomputes labels without asking Qwen again. Two
  queries dropped whose source item Amazon misfiled (a shampoo in Body Butters, a Remington knife in
  Eyeliner): 198 left, 49 dev + 149 test.
  Fragrance merge (Taras decided): 22 of 198 queries are perfume, and some name a sibling leaf
  ("... cedar cologne spray" from an Eau de Toilette), so leaf labels measured Amazon's shelving, not
  search. Cologne, Eau de Parfum, Eau de Toilette, Body Sprays and Eau Fraiche are now one label group
  "Fragrance" (1,640 items in the labelled pool; fragrance Sets stay apart). Cost: precision@10 on perfume
  queries becomes easy for every system. 20 queries changed labels.
  Known label noise left: leaf labels ignore Men's/Women's, so a "men cologne" query counts women's
  perfume as relevant.
  Spot-check of 10 queries (Taras): all read like a person wrote them, but some are more detailed than a
  real shopper types. Effect: long queries share more words with titles, which helps BM25; the
  `title_overlap` < 0.3 slice in A4 shows how much.
  Lesson guesses: toy "red matte lipstick", labels = any lipstick, system A returns 10 nude, B 10 red;
  Taras guessed precision@10 A 0, B 1.0; truth both 1.0 (the label cannot see colour). Trap: which
  system gains from the grade-2 source item? Taras guessed BM25, because Qwen reuses title words: right.
