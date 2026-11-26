"""Hand-computed cases for scripts/metrics.py. log2(2) = 1, log2(3) = 1.58496."""

import pytest

from metrics import evaluate, ndcg_at_k, precision_at_k, source_in_top_k

L3 = 1.584962500721156  # log2(3)


def test_precision_counts_both_grades_and_divides_by_k():
    relevant = {"a": 2, "b": 1, "c": 1}
    assert precision_at_k(["a", "x", "b", "y", "c"] + list("pqrst"), relevant) == pytest.approx(0.3)


def test_precision_short_list_still_divides_by_k():
    assert precision_at_k(["a"], {"a": 1}) == pytest.approx(0.1)


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k(["a", "b"], {"a": 2, "b": 1}) == pytest.approx(1.0)


def test_ndcg_swapped_pair():
    # DCG = 1/1 + 2/log2(3); ideal = 2/1 + 1/log2(3)
    expected = (1 + 2 / L3) / (2 + 1 / L3)  # 0.8597
    assert ndcg_at_k(["b", "a"], {"a": 2, "b": 1}) == pytest.approx(expected)


def test_ndcg_ideal_uses_all_labels_not_only_retrieved():
    # only "b" is returned, but the ideal top 10 holds "a" too
    assert ndcg_at_k(["b"], {"a": 2, "b": 1}) == pytest.approx(1 / (2 + 1 / L3))


def test_ndcg_no_relevant_retrieved_is_zero():
    assert ndcg_at_k(["x", "y"], {"a": 2}) == 0.0


def test_source_in_top_k():
    assert source_in_top_k(["x", "s"], "s") == 1.0
    assert source_in_top_k([str(i) for i in range(10)] + ["s"], "s") == 0.0


def test_duplicates_are_rejected():
    with pytest.raises(ValueError):
        precision_at_k(["a", "a"], {"a": 1})


def query(qid, brand, category="Lipstick", overlap=0.5, split="test"):
    return {"qid": qid, "split": split, "source_id": f"s{qid}", "brand_in_query": brand,
            "category": category, "title_overlap": overlap,
            "relevant": {f"s{qid}": 2, f"r{qid}": 1}}


def test_evaluate_slices():
    queries = [
        query(0, brand=False),                       # perfect: p 0.2, ndcg 1, source 1
        query(1, brand=True),                        # nothing returned: all 0
        query(2, brand=False, category="Fragrance", overlap=0.1),  # only r2: p 0.1
        query(3, brand=False, split="dev"),          # not counted
    ]
    run = {0: ["s0", "r0"], 2: ["r2"], 3: ["s3"]}
    r = evaluate(run, queries)
    ndcg2 = 1 / (2 + 1 / L3)
    assert r["all"]["n"] == 3
    assert r["all"]["ndcg@10"] == pytest.approx((1 + 0 + ndcg2) / 3, abs=1e-3)
    assert r["all"]["p@10"] == pytest.approx(0.15)       # brand query left out
    assert r["no brand, no fragrance"]["p@10"] == pytest.approx(0.2)
    assert r["brand"] == {"n": 1, "p@10": None, "ndcg@10": 0.0, "source@10": 0.0}
    assert r["title overlap < 0.3"]["n"] == 1
    assert r["title overlap < 0.3"]["source@10"] == 0.0


def test_evaluate_accepts_string_qids_from_json():
    # json.dump turns int keys into strings; the scores must not silently drop to 0
    assert evaluate({"0": ["s0", "r0"]}, [query(0, brand=False)])["all"]["ndcg@10"] == 1.0


def test_evaluate_rejects_a_run_that_matches_no_query():
    with pytest.raises(ValueError):
        evaluate({99: ["x"]}, [query(0, brand=False)])
