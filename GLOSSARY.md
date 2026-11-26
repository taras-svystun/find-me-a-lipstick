# Glossary: find-me-a-lipstick

Hybrid text + photo search over beauty products, with an ablation table that measures each part.

- **item**: one product from the dataset (title, brand, category, features, description, price, image).
- **query**: a search request in natural language, for example "matte red lipstick that lasts all day".
- **relevance label**: which items count as correct answers for a query, like the answer key to a test.
  Here built for free: grade 2 for the source item, 1 for its label group (and brand, when the query
  names it).
- **label group**: the shelf used for labels: the leaf category, except that Cologne, Eau de Parfum,
  Eau de Toilette, Body Sprays and Eau Fraiche are one group "Fragrance", because Amazon files the
  same bottle under any of them.
- **BM25**: keyword search. Scores an item by how many query words it contains, giving rare words
  more weight and damping repeats and long texts.
- **embedding**: a vector that stands for a text or a photo, so that similar meanings sit close together.
- **CLIP / SigLIP**: models that put texts and photos into one vector space, so a text query can find
  a photo.
- **dense search** (vectors): search by nearest embeddings.
- **RRF** (reciprocal rank fusion): merges several ranked lists by summing `1 / (k + rank)` for each item.
- **hybrid search**: BM25 and dense search merged with RRF.
- **cross-encoder reranker**: a model that reads the query and one item together and scores the pair.
  Slow, so it only reorders the top 50.
- **source item**: the item Qwen read to write a test query. Grade 2 in its labels.
- **dev set / test set**: qids 1-49 (49 queries) tune choices (model, RRF `k`); qids 50-199 (149) fill the
  ablation table. Tuning on the test set would make the table flatter the choices.
- **precision@10**: the share of the top 10 that is relevant.
- **recall@10**: the share of all relevant items that appear in the top 10. Not used: with hundreds
  of relevant items per query it cannot rise above a few percent.
- **nDCG@10**: a top-10 score that rewards relevant items more the higher they are ranked, scaled to 0..1.
  Each grade is divided by log2(rank + 1), then the sum (DCG) is divided by the best possible sum for
  that query, so every query weighs the same however many relevant items it has.
- **gain**: what one item adds to DCG before the rank discount. Here the grade itself (2 or 1), not
  2^grade - 1, so the source item is not weighted up.
- **mutation check**: break the code on purpose and see that some test fails; a test no mutant breaks
  protects nothing.
- **ablation table**: one row per system variant (BM25 / vectors / hybrid / hybrid + reranker), so the
  gain of each part is visible.
