"""A5 BM25: unit cases computed by hand, invariance and direction checks, and the table end to end.
BM25 here: idf = ln(1 + (N - n + 0.5) / (n + 0.5)), k1 = 1.2, b = 0.75 (Lucene's defaults)."""

import math

import pytest

from ablation import build_table
from bm25 import BM25, item_text, tokenize

DOCS = ["red lipstick", "nude lipstick", "dry hair shampoo", "red nail polish"]  # avg length 2.5
IDS = ["a", "b", "c", "d"]


def test_tokenize_lowercases_and_splits_on_punctuation():
    assert tokenize("Matte-Red LIPSTICK, 24h!") == ["matte", "red", "lipstick", "24h"]


def test_idf_rare_word_weighs_more():
    bm = BM25(IDS, DOCS)
    assert bm.idf("shampoo") == pytest.approx(math.log(1 + 3.5 / 1.5))  # in 1 of 4 docs: 1.204
    assert bm.idf("lipstick") == pytest.approx(math.log(2))  # in 2 of 4 docs: 0.693
    assert bm.idf("glitter") == 0.0  # unknown word adds nothing


def test_score_one_term_by_hand():
    bm = BM25(IDS, DOCS)
    # "shampoo" in doc c: tf 1, length 3, avg 2.5
    expected = math.log(1 + 3.5 / 1.5) * 1 * 2.2 / (1 + 1.2 * (0.25 + 0.75 * 3 / 2.5))  # 1.113
    assert bm.score("shampoo", 2) == pytest.approx(expected)


def test_repeats_saturate():
    bm = BM25(["x", "y"], ["gloss " * 10, "matte " * 10])  # both length 10 = the average: no length penalty
    idf = bm.idf("gloss")
    assert bm.score("gloss", 0) == pytest.approx(idf * 10 * 2.2 / (10 + 1.2))
    assert bm.score("gloss", 0) < idf * 2.2  # ten repeats never pass k1 + 1 = 2.2 times one idf


def test_longer_doc_with_same_count_scores_lower():
    bm = BM25(["short", "long", "other"], ["red lipstick", "red lipstick with a soft satin finish", "shampoo"])
    assert bm.score("lipstick", 0) > bm.score("lipstick", 1)


def test_search_ranks_and_breaks_ties_by_item_order():
    bm = BM25(IDS, DOCS)
    assert bm.search("red lipstick") == ["a", "b", "d"]  # a has both words; b and d one each
    assert BM25(["p", "q"], ["red gloss", "red gloss"]).search("red") == ["p", "q"]


def test_search_with_no_known_word_returns_nothing():
    assert BM25(IDS, DOCS).search("glitter") == []


# invariance: changes that must not move the result; direction: a change that must move it

def test_case_and_spaces_do_not_change_the_ranking():
    bm = BM25(IDS, DOCS)
    assert bm.search("  RED   Lipstick ") == bm.search("red lipstick")


def test_adding_a_word_moves_its_item_up():
    bm = BM25(IDS, DOCS)
    assert bm.search("red")[0] == "a"  # "red lipstick" is shorter than "red nail polish"
    assert bm.search("red polish")[0] == "d"


# integration and end to end

ITEMS = [
    {"id": "a", "title": "Velvet Matte Lipstick", "brand": "Rouge Co", "categories": ["Makeup", "Lips", "Lipstick"],
     "features": ["long lasting red"], "description": "A matte red lipstick."},
    {"id": "b", "title": "Glitter Gloss", "brand": None, "categories": ["Makeup", "Lips", "Lip Gloss"],
     "features": [], "description": None},
    {"id": "c", "title": "Hydrating Shampoo", "brand": "Hairy", "categories": ["Hair Care", "Shampoo", "Shampoo"],
     "features": ["for dry hair"], "description": "Gentle shampoo."},
]


def test_item_text_uses_brand_title_categories_features_description():
    text = item_text(ITEMS[0])
    for part in ["Rouge Co", "Velvet Matte Lipstick", "Lips", "long lasting red", "A matte red lipstick."]:
        assert part in text
    assert item_text(ITEMS[1]) == "Glitter Gloss Makeup Lips Lip Gloss"  # missing fields are skipped


def test_build_table_end_to_end():
    queries = [
        {"qid": 7, "split": "test", "query": "rouge co red lipstick", "source_id": "a", "brand_in_query": True,
         "category": "Lipstick", "title_overlap": 0.25, "relevant": {"a": 2}},
        {"qid": 8, "split": "test", "query": "shampoo for dry hair", "source_id": "c", "brand_in_query": False,
         "category": "Shampoo", "title_overlap": 0.25, "relevant": {"c": 2}},
    ]
    rows = build_table(ITEMS, queries, systems=["bm25"])
    row = rows["bm25"]
    assert row["all"]["ndcg@10"] == 1.0  # both source items come first
    assert row["no brand"]["p@10"] == pytest.approx(0.1)
    assert row["latency_ms"] >= 0
