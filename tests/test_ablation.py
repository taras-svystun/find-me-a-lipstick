"""Paired differences between two rows of the ablation table, computed by hand on a tiny run.
A paired difference compares two systems query by query, so each query's difficulty cancels out."""

import pytest

from ablation import paired_rows, pairs_markdown

QUERIES = [
    {"qid": 1, "split": "test", "query": "x", "source_id": "s1", "brand_in_query": False, "category": "Lipstick",
     "title_overlap": 0.5, "text_overlap": 0.5, "relevant": {"s1": 2, "r1": 1}},
    {"qid": 2, "split": "test", "query": "y", "source_id": "s2", "brand_in_query": False, "category": "Shampoo",
     "title_overlap": 0.5, "text_overlap": 0.9, "relevant": {"s2": 2}},
]
RUNS = {
    "new": {1: ["s1", "r1"], 2: ["s2"]},  # p@10: 0.2, 0.1; source@10: 1, 1
    "old": {1: ["zz"], 2: ["s2"]},        # p@10: 0.0, 0.1; source@10: 0, 1
}


def test_paired_rows_by_hand():
    rows = paired_rows(RUNS, QUERIES, [("new", "old")], "test")
    pair = rows["new - old"]
    # p@10 differences per query: +0.2 and 0.0 -> mean +0.1; resampling 2 values gives means in {0, 0.1, 0.2}
    assert pair["p@10 no brand"]["diff"] == pytest.approx(0.1)
    lo, hi = pair["p@10 no brand"]["ci"]
    assert 0.0 <= lo <= 0.1 <= hi <= 0.2
    # source@10: +1 on query 1 only, which is the one query with text overlap < 0.7
    assert pair["source@10 text overlap < 0.7"]["diff"] == pytest.approx(1.0)
    assert pair["source@10 all"]["diff"] == pytest.approx(0.5)


def test_markdown_lists_the_paired_differences():
    md = pairs_markdown(paired_rows(RUNS, QUERIES, [("new", "old")], "test"))
    assert "| new - old |" in md
    assert "+0.100 [" in md
