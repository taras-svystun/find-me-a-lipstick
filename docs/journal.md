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
- 2026-11-26, A4 metrics: `scripts/metrics.py` gives precision@10, nDCG@10 and "source item in top 10",
  and `evaluate()` averages them over the slices in the plan; `tests/test_metrics.py` has 11 cases
  computed by hand (`pytest -q`: 11 passed). Decision: linear gain (the grade), not 2^grade - 1, which
  would make the source item worth 3 shelf items instead of 2 and so favour BM25, which finds it most
  easily. Cost: our nDCG is not comparable with tools that default to 2^grade - 1.
  Evidence the tests bite: 8 deliberate bugs (exponential gain, 1/rank discount, ideal from retrieved
  items only, dividing precision by the returned count, ...) each made at least one test fail.
  Tests-first was not followed for the first 9 tests (written with the code); it was for the qid fix
  below (2 failed before, 11 passed after) and is a rule from now on (`CLAUDE.md`, Tests).
  Random ranking on the 149 test queries: no-brand precision@10 0.04 against 0.036 expected from label
  sizes, nDCG@10 0.023. Reviewer: no blockers; fixed a silent-zero bug (a run loaded from JSON has string
  qids, so every query would score 0); `evaluate()` now converts keys and raises if no query matches.
  Note: nDCG on the "brand" and "no brand" slices is on different scales (a source-only system scores
  0.81 against 0.36), so compare systems within one slice, never across.
  Lesson guesses: toy ranking L1, S, X, L2, Y: Taras guessed 0.85-0.95, truth 0.86. Interactive page
  with 4 puzzles: Taras got 2 of 4 right before the reveal. Team Lead question "why divide by the ideal
  DCG?": Taras said "to get 0..1"; the deeper reason is that each query has its own maximum (2 for a
  one-item brand query, 5.5 for a full shelf), so a mean of raw DCG would rank queries, not systems.
- 2026-11-27, A5 BM25: `scripts/bm25.py` (Lucene BM25, k1 1.2, b 0.75, no stemming) over title,
  brand, categories, features and description; `scripts/ablation.py` writes `eval/ablation.md` and
  `eval/ablation.json`. Tests were committed first (`c54be73`, failing: modules missing), code in
  `8a31b39` (22 passed). One test changed after the code: its comment claimed a document of average
  length, and it was not.
  BM25 on the 149 test queries, Taras's guess -> measured: p@10 no brand 0.4 -> 0.643; nDCG@10 brand
  0.6 -> 0.857; p@10 on `title_overlap` < 0.3 "halves" -> 0.621 (almost no drop). Random gives 0.04.
  Why the overlap slice barely moves precision: the label is the shelf (leaf category), and the query
  names the product type ("lipstick"), which BM25 also finds in the categories field; copied title
  words help find the source item, not the shelf. Evidence: source@10 drops 0.812 -> 0.632 on that
  slice while p@10 drops only 0.643 -> 0.621. So our shelf labels make precision@10 easy for a
  keyword system; the dense rows must win on source@10 and low-overlap queries, or not at all.
  Review (no blockers) limits this reading: (1) the 0.022 gap is inside the noise: the p@10 slice
  averages 33 queries (standard error ~0.05), and dev gives 0.567 on the same slice; (2) the slice
  removes only words copied from the title, while Qwen also saw the category path, features and
  description, which BM25 indexes too ("matte lipstick long lasting ..." is in the slice). Open:
  bootstrap confidence intervals before A6, and an overlap slice against the full item text.
  Median latency 10 ms per query on the M1.
  Lesson: interactive page with k1, b and four riddles, the last one the no-stemming trap; Taras
  solved all four.
- 2026-11-27, A5 follow-up: 95% bootstrap intervals in every cell (resample the 149 test queries
  1,000 times, seed 0), `paired_diff_ci` for comparing rows in A6, the p@10 query count per slice,
  and a `text_overlap` slice: share of query words found anywhere in the source's indexed text
  (threshold 0.7, picked by size: 43 test queries, 31 without brand; 18 are also in the title
  slice). Tests first (`e120678`), code `73945c1`, 29 passed; `--relabel` changed only the new field.
  Taras's guess -> measured (BM25): source@10 on text overlap < 0.7: 0.6 -> 0.488 [0.349, 0.628];
  width of the p@10 no-brand interval: +-0.05 -> +-0.056 [0.587, 0.699], right.
  Reading: copying from the item card explains much of BM25's source@10 (0.812 on all queries); its
  p@10 on the slice stays 0.642, because the shelf word is in the categories of many items. The
  slice is chosen by word overlap, BM25's own mechanism, so it is a diagnostic of copying, not a
  set of "hard" queries; describe it that way. Overlapping intervals of two rows do not mean "no
  difference": compare rows with the paired interval. Accepted: the upper percentile index leaves
  24 of 1,000 means above it against 25 below the lower one (~0.1% coverage, invisible at 3
  places). Cut candidate after A6 starts: the title-overlap slice, which the text slice contains.
- 2026-11-27, A6 vectors: `scripts/dense.py` embeds each item's text (the BM25 fields, lower-cased, cut at
  the token limit) and photo with CLIP ViT-B/32 and SigLIP2 base-224, on CPU (the sandbox has no GPU:
  12 and 22 min for 15.6k items). Three rows: text, photo, and w * z(text) + (1 - w) * z(photo).
  Tests first (`0553a62`, failing: module missing), code `93ead89` (46 passed). One test changed after
  the code: "red lipstick" puts the nude lipstick first on both models (a long item text dilutes "red"),
  so the direction test now checks that the word moves the score gap, not the top rank.
  Setup notes: pip needs `--use-deprecated=legacy-certs` and Python downloads need `SSL_CERT_FILE` set to
  certifi in the sandbox (Keychain blocked); Hugging Face's xet transfer is blocked, `HF_HUB_DISABLE_XET=1`.
  Why z: measured on dev, text cosines average 0.48-0.63 and photo 0.02-0.22 (the modality gap, a
  constant that cannot change a ranking), but text varies 2-2.6 times more across items (std 0.061-0.093
  against 0.035-0.038), so a raw mix gives the text side more say than w. Commit `93ead89` blamed the gap;
  the spread is the real reason.
  Dev choice (49 queries, nDCG@10 all), Taras's guess -> measured: SigLIP2 -> SigLIP2 (best 0.648 against
  CLIP 0.510); w 0.5 -> 0.3 (0.2-0.5 within 0.03, inside the noise). SigLIP2 photo alone 0.610 beats its
  text alone 0.499; for CLIP photo 0.405 and text 0.423.
  Test (149 queries), Taras's guess -> measured: p@10 no brand text 0.55 -> 0.586, photo 0.4 -> 0.646,
  text+photo 0.6 -> 0.679 (BM25 0.643); source@10 on text overlap < 0.7, text+photo 0.5 -> 0.419 (BM25 0.488).
  Paired bootstrap, p@10 no brand: photo - text +0.060 [0.016, 0.106]; text+photo - photo +0.033
  [0.010, 0.057]; text+photo - BM25 +0.036 [-0.005, 0.082], so no proven win over BM25. nDCG@10 no brand
  text+photo - BM25 -0.028 [-0.067, 0.013]; source@10 all -0.282 [-0.362, -0.201].
  Reading: vectors find the shelf as well as BM25 but much more rarely the exact item (-0.282 on all
  queries). Where the query shares few words with the card, the gap is -0.070 [-0.186, +0.023]: not
  shown either way on 43 queries. The A5 hope "dense wins on source@10" fails. The
  case for A7 is that the two make different mistakes; that is what RRF can use. Why the photo beats the
  text (estimate, not checked): the label is the shelf, a photo shows the product type plainly, and the
  item text the encoder sees is the first 64 tokens of brand and a long marketing title. A check: embed the
  title alone and see if text catches up, judged on dev (`--tune`), not on the test table.
  Median latency 46-49 ms per query (CPU query encoding plus 15.6k dot products), BM25 10 ms; reruns move
  it ~20%, so the 1-3 ms between dense rows is noise.
  Review (no blockers): the paired intervals were computed by hand, so `ablation.py` now writes them
  (tests first `8ee296f`); `eval/ablation.json` is now {"rows", "paired"}. The low-overlap claim above was
  cut back to what its interval shows. The title-overlap rows left the table (the text-overlap slice
  contains them; the slice stays in `metrics.py`). The real-model invariance test was cut: query
  normalisation is checked with a toy encoder, and the cut saves two model loads per commit.
  Lesson: interactive page (w slider, z switch) with four riddles.
  Taras's results: riddle 1 (size of the text-photo cosine gap) missed, he did not expect such a
  large gap; riddle 2 (the trap) right, by intuition, he could not yet explain why; riddle 3 (long
  keyword query against a short one) missed, he had not noticed the models were CLIP and SigLIP2;
  riddle 4 (why SigLIP2 beats CLIP here) right, without knowing the training difference before.
  To revisit in A7: explain the trap in his own words.
- 2026-11-27, A7 hybrid, code and dev choice (table waits for Taras's guesses): `scripts/hybrid.py`
  fuses two lists, BM25 and vectors text+photo, with RRF, `1 / (k + rank)` over each list's top 100.
  Two lists, not three (BM25, text, photo): with three, the dense side would vote twice. Tests first
  (`3f482d3`, failing: module missing), 53 passed with the code.
  Dev (49 queries), k -> nDCG@10 all / p@10 no brand: 1 0.729/0.678, 10 0.728/0.692, 30 0.722/0.714,
  60 0.723/0.724, 100 0.724/0.727, 300 0.720/0.724. nDCG@10 is flat (0.009 range, well inside the noise
  of 49 queries), so k stays at the paper's 60 instead of the best cell; p@10 grows with k up to ~60.
  Lesson riddle (X rank 1 in BM25 only, Y rank 3 in both): Taras said Y wins at k=60 and X at k=0
  (unsure k=0 is allowed: it is, ranks start at 1). Right: X 1/61 = 0.016 < Y 2/63 = 0.032; at k=0 X 1
  > Y 0.67. A large k flattens the reward per rank, so agreement between lists beats one first place.
  Test (149 queries), Taras's guess -> measured: p@10 no brand 0.70 -> 0.698 [0.640, 0.758]; source@10
  all 0.85 -> 0.711 [0.638, 0.779] (BM25 0.812). nDCG@10 brand 0.895 (BM25 0.857). Median 59 ms.
  Paired: hybrid - BM25 p@10 no brand +0.055 [0.023, 0.090], the first row with a proven win over BM25;
  nDCG@10 no brand +0.018 [-0.010, 0.047], not shown; source@10 all -0.101 [-0.161, -0.040], a proven
  loss. hybrid - vectors text+photo: source@10 +0.181 [0.121, 0.242], nDCG@10 no brand +0.046 [0.020, 0.072].
  Reading: held back until Taras answers the lesson trap (why source@10 fell); it goes in then.
  Evidence for it, counted with a one-off loop over the test runs (BM25 and dense top 100, fused at
  k=60): the hybrid loses BM25's source item from the top 10 on 19 test queries; BM25 had it at rank
  1-8, and on 15 of them the dense list does not hold it in its top 100 (the other 4: dense rank 27-86).
  Dev source@10 by k (added to `--tune` after the review): 1 0.796, 10 0.796, 30 0.714, 60 0.673,
  100 0.673, 300 0.673. So on dev too a smaller k keeps the exact item and a larger one fills the shelf;
  k=60 was chosen before this column existed and stays for the table. Which side matters is a product
  question, open for Taras. Depth 100 is not tuned; it sets how many items can sit in both lists.
  A8's reranker is the planned way to bring the exact item up. Latency: 59 ms runs BM25 and dense one after another.
