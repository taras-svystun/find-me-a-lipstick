# Glossary: find-me-a-lipstick

Hybrid text + photo search over beauty products, with an ablation table that measures each part.

- **item**: one product from the dataset (title, brand, category, features, description, price, image).
- **query**: a search request in natural language, for example "matte red lipstick that lasts all day".
- **relevance label**: which items count as correct answers for a query. Here built for free from
  category and brand.
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
- **recall@10**: the share of all relevant items that appear in the top 10.
- **nDCG@10**: a top-10 score that rewards relevant items more the higher they are ranked, scaled to 0..1.
- **ablation table**: one row per system variant (BM25 / vectors / hybrid / hybrid + reranker), so the
  gain of each part is visible.
