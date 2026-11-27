"""A7 hybrid search: reciprocal rank fusion (RRF) of several ranked lists, computed by hand on fake
engines, plus the table end to end on a tiny corpus.

RRF score of item d = sum over lists L that contain d of 1 / (k + rank_L(d)), rank starting at 1.
Only ranks count: the engines' own scores (BM25 points, cosines) never enter the sum."""

import pytest

from hybrid import RRF


class Fake:
    """An engine that always returns the same ranked ids and records the k it was asked for."""

    def __init__(self, ids):
        self.ids, self.asked = ids, []

    def search(self, query, k=10):
        self.asked.append(k)
        return self.ids[:k]


def test_rrf_scores_by_hand():
    # A = [p, q, r], B = [q, s], k = 60
    # p 1/61 = 0.016393; q 1/62 + 1/61 = 0.032522; r 1/63 = 0.015873; s 1/62 = 0.016129
    rrf = RRF([Fake(["p", "q", "r"]), Fake(["q", "s"])], k=60)
    assert rrf.search("x", 4, with_scores=True) == [
        ("q", pytest.approx(0.032522, abs=1e-6)),
        ("p", pytest.approx(0.016393, abs=1e-6)),
        ("s", pytest.approx(0.016129, abs=1e-6)),
        ("r", pytest.approx(0.015873, abs=1e-6)),
    ]


def test_search_returns_top_k_ids():
    rrf = RRF([Fake(["p", "q", "r"]), Fake(["q", "s"])], k=60)
    assert rrf.search("x", 2) == ["q", "p"]


def test_each_engine_is_asked_for_depth_items_not_k():
    a, b = Fake(["p"]), Fake(["q"])
    RRF([a, b], k=60, depth=100).search("x", 10)
    assert a.asked == [100] and b.asked == [100]


def test_an_item_below_depth_gets_nothing_from_that_list():
    # depth 2: r (rank 3 in A) is never seen, so it cannot appear at all
    rrf = RRF([Fake(["p", "q", "r"]), Fake(["s"])], k=60, depth=2)
    assert "r" not in rrf.search("x", 10)


def test_swapping_the_lists_keeps_the_scores():
    # invariance: fusion is a sum, so the order of the engines must not move any score
    one = dict(RRF([Fake(["p", "q", "r"]), Fake(["q", "s"])], k=10).search("x", 4, with_scores=True))
    two = dict(RRF([Fake(["q", "s"]), Fake(["p", "q", "r"])], k=10).search("x", 4, with_scores=True))
    assert one == pytest.approx(two)


def test_ties_go_to_the_earlier_list():
    # p and s both score 1/61; p comes from the first engine, so runs are reproducible
    assert RRF([Fake(["p"]), Fake(["s"])], k=60).search("x", 2) == ["p", "s"]


def test_build_table_with_the_hybrid_row(tmp_path, monkeypatch):
    # end to end on 2 items: BM25 and SigLIP2 both put the shampoo first, so the hybrid must too
    from PIL import Image

    import dense
    from ablation import build_table
    items = [
        {"id": "a", "title": "Velvet Matte Lipstick", "brand": "Rouge Co", "categories": ["Makeup", "Lips", "Lipstick"],
         "features": ["long lasting red"], "description": "A matte red lipstick."},
        {"id": "c", "title": "Hydrating Shampoo", "brand": "Hairy", "categories": ["Hair Care", "Shampoo", "Shampoo"],
         "features": ["for dry hair"], "description": "Gentle shampoo."},
    ]
    for item_id, rgb in {"a": (200, 0, 0), "c": (0, 60, 200)}.items():
        Image.new("RGB", (224, 224), rgb).save(tmp_path / f"{item_id}.jpg")
    monkeypatch.setattr(dense, "IMAGE_DIR", tmp_path)
    queries = [{"qid": 8, "split": "test", "query": "shampoo for dry hair", "source_id": "c", "brand_in_query": False,
                "category": "Shampoo", "title_overlap": 0.25, "text_overlap": 0.5, "relevant": {"c": 2}}]
    rows = build_table(items, queries, systems=["hybrid"])
    assert rows["hybrid"]["all"]["ndcg@10"] == 1.0
