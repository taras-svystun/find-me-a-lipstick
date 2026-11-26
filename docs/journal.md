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
