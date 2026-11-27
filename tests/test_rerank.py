"""A8 reranker: a cross-encoder rescores the first stage's top `depth` items and reorders them.

Unit cases use a fake first stage and a fake scorer, so the expected order is computed by hand.
One direction check runs the real cross-encoder, and the table is built end to end on a tiny corpus.

A cross-encoder reads the query and one item text together and returns one number per pair, so
it can only reorder what the first stage handed it: an item below `depth` never comes back."""

from rerank import Rerank


class Fake:
    """A first stage that returns fixed ranked ids and records the k it was asked for."""

    def __init__(self, ids):
        self.ids, self.asked = ids, []

    def search(self, query, k=10):
        self.asked.append(k)
        return self.ids[:k]


class Scorer:
    """Scores a text by a fixed table; records every (query, texts) call."""

    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, query, texts):
        self.calls.append((query, list(texts)))
        return [self.table.get(t, 0.0) for t in texts]


TEXTS = {"p": "text p", "q": "text q", "r": "text r", "s": "text s"}


def test_reorders_the_candidates_by_the_scorer():
    # first stage p, q, r; scores p 0.1, q 0.9, r 0.5 -> q, r, p
    rr = Rerank(Fake(["p", "q", "r"]), Scorer({"text p": 0.1, "text q": 0.9, "text r": 0.5}), TEXTS, depth=3)
    assert rr.search("x", 3) == ["q", "r", "p"]


def test_asks_the_first_stage_for_depth_items_and_returns_k():
    first = Fake(["p", "q", "r", "s"])
    rr = Rerank(first, Scorer({"text s": 1.0}), TEXTS, depth=4)
    assert rr.search("x", 2) == ["s", "p"]
    assert first.asked == [4]


def test_an_item_below_depth_never_comes_back():
    # s would score highest, but the first stage ranks it 4th and depth is 3
    rr = Rerank(Fake(["p", "q", "r", "s"]), Scorer({"text s": 9.0}), TEXTS, depth=3)
    assert "s" not in rr.search("x", 10)


def test_ties_keep_the_first_stage_order():
    rr = Rerank(Fake(["r", "p", "q"]), Scorer({}), TEXTS, depth=3)
    assert rr.search("x", 3) == ["r", "p", "q"]


def test_scorer_gets_all_candidates_in_one_call():
    # one batch per query: a forward pass per pair, but not a Python call per pair
    scorer = Scorer({})
    Rerank(Fake(["p", "q", "r"]), scorer, TEXTS, depth=3).search("x", 3)
    assert scorer.calls == [("x", ["text p", "text q", "text r"])]


def test_case_and_extra_spaces_do_not_move_the_result():
    # invariance: the query is normalised before scoring
    scorer = Scorer({})
    rr = Rerank(Fake(["p"]), scorer, TEXTS, depth=1)
    rr.search("  Red   LIPSTICK ", 1)
    rr.search("red lipstick", 1)
    assert scorer.calls[0] == scorer.calls[1]


def test_real_cross_encoder_prefers_the_matching_item():
    # direction: "red lipstick" must score a lipstick card above a shampoo card
    from rerank import MODEL, get_scorer
    s = get_scorer(MODEL)("red lipstick", ["Rouge Co Velvet Matte Lipstick Makeup Lips Lipstick long lasting red",
                                           "Hairy Hydrating Shampoo Hair Care Shampoo for dry hair"])
    assert s[0] > s[1]


def test_build_table_with_the_rerank_row(tmp_path, monkeypatch):
    # end to end on 2 items: the shampoo query must end with the shampoo first
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
    rows = build_table(items, queries, systems=["hybrid + rerank"])
    assert rows["hybrid + rerank"]["all"]["ndcg@10"] == 1.0
