"""A6 dense search: unit cases computed by hand with toy vectors, invariance and direction checks on
the real models, and the table end to end on a tiny corpus.

Dense score of item d for query q (vectors are unit length, so a dot product is a cosine):
  text  = cos(q, text_d)     photo = cos(q, photo_d)
  text+photo = w * z(text) + (1 - w) * z(photo), where z standardises a score over all items
  (subtract the mean, divide by the standard deviation), because text-text and text-photo cosines
  live on different scales."""

import numpy as np
import pytest
from PIL import Image

from ablation import build_table
from dense import Dense, embed_items, normalize_text, zscore

IDS = ["a", "b", "c"]
TEXT = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])  # unit vectors
PHOTO = np.array([[0.0, 1.0], [1.0, 0.0], [0.8, 0.6]])


def toy(w):
    return Dense(IDS, TEXT, PHOTO, encode_query=lambda q: np.array([1.0, 0.0]), w=w)


def test_normalize_text_lowercases_and_collapses_spaces():
    assert normalize_text("  Red   LIPSTICK\n") == "red lipstick"


def test_zscore_by_hand():
    # mean 2, population std sqrt(2/3) = 0.8165
    assert zscore(np.array([1.0, 2.0, 3.0])) == pytest.approx([-1.2247, 0.0, 1.2247], abs=1e-4)


def test_text_only_ranks_by_text_cosine():
    assert toy(1.0).search("anything", 3) == ["a", "c", "b"]  # cos 1, 0.6, 0


def test_photo_only_ranks_by_photo_cosine():
    assert toy(0.0).search("anything", 3) == ["b", "c", "a"]  # cos 1, 0.8, 0


def test_text_plus_photo_by_hand():
    # text [1, 0, 0.6]: mean 0.5333, std 0.4110 -> z [1.1355, -1.2978, 0.1622]
    # photo [0, 1, 0.8]: mean 0.6, std 0.4320 -> z [-1.3887, 0.9258, 0.4629]
    # w = 0.5: a -0.1266, b -0.1860, c 0.3126
    bm = toy(0.5)
    assert bm.search("anything", 3, with_scores=True) == [
        ("c", pytest.approx(0.3126, abs=1e-3)),
        ("a", pytest.approx(-0.1266, abs=1e-3)),
        ("b", pytest.approx(-0.1860, abs=1e-3)),
    ]


def test_ties_go_to_the_earlier_item():
    d = Dense(["p", "q"], np.array([[1.0, 0.0]] * 2), np.array([[1.0, 0.0]] * 2),
              encode_query=lambda q: np.array([1.0, 0.0]), w=1.0)
    assert d.search("x", 2) == ["p", "q"]


def test_query_is_normalised_before_encoding():
    seen = []
    d = Dense(IDS, TEXT, PHOTO, encode_query=lambda q: seen.append(q) or np.array([1.0, 0.0]), w=1.0)
    d.search("  RED   Lipstick ", 3)
    assert seen == ["red lipstick"]


# real models: invariance, direction, one space for text and photos

ITEMS = [
    {"id": "a", "title": "Velvet Matte Lipstick", "brand": "Rouge Co", "categories": ["Makeup", "Lips", "Lipstick"],
     "features": ["long lasting red"], "description": "A matte red lipstick."},
    {"id": "b", "title": "Nude Satin Lipstick", "brand": None, "categories": ["Makeup", "Lips", "Lipstick"],
     "features": [], "description": None},
    {"id": "c", "title": "Hydrating Shampoo", "brand": "Hairy", "categories": ["Hair Care", "Shampoo", "Shampoo"],
     "features": ["for dry hair"], "description": "Gentle shampoo."},
]
COLOURS = {"a": (200, 0, 0), "b": (225, 190, 160), "c": (0, 60, 200)}  # red, beige, blue squares


@pytest.fixture(scope="module")
def image_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("images")
    for item_id, rgb in COLOURS.items():
        Image.new("RGB", (224, 224), rgb).save(d / f"{item_id}.jpg")
    return d


@pytest.mark.parametrize("model", ["clip", "siglip2"])
def test_embed_items_gives_unit_vectors_per_item(model, image_dir):
    v = embed_items(ITEMS, model, image_dir)
    assert v["ids"] == ["a", "b", "c"]
    assert v["text"].shape == v["photo"].shape and v["text"].shape[0] == 3
    assert np.linalg.norm(v["text"], axis=1) == pytest.approx(1.0, abs=1e-4)
    assert np.linalg.norm(v["photo"], axis=1) == pytest.approx(1.0, abs=1e-4)


def test_embed_items_fails_loudly_on_a_missing_photo(tmp_path):
    with pytest.raises(FileNotFoundError):
        embed_items(ITEMS[:1], "clip", tmp_path)


@pytest.mark.parametrize("model", ["clip", "siglip2"])
def test_case_and_spaces_do_not_change_the_ranking(model, image_dir):
    d = Dense.from_items(ITEMS, model, w=0.5, image_dir=image_dir)
    assert d.search("  RED   Lipstick ", 3) == d.search("red lipstick", 3)


@pytest.mark.parametrize("model", ["clip", "siglip2"])
def test_adding_a_word_moves_its_item_up_in_text_search(model, image_dir):
    # direction, not rank: the word must move the score gap its way (top 1 is not asserted)
    d = Dense.from_items(ITEMS, model, w=1.0, image_dir=image_dir)
    assert d.search("shampoo", 1) == ["c"]

    def gap(query):  # red item a minus nude item b
        s = dict(d.search(query, 3, with_scores=True))
        return s["a"] - s["b"]

    assert gap("red lipstick") > gap("lipstick")
    assert gap("nude lipstick") < gap("lipstick")


@pytest.mark.parametrize("model", ["clip", "siglip2"])
def test_a_text_query_finds_the_photo_of_its_colour(model, image_dir):
    # photo only: the item texts play no part, so the text query must land near the right photo
    d = Dense.from_items(ITEMS, model, w=0.0, image_dir=image_dir)
    assert d.search("a red square", 1) == ["a"]
    assert d.search("a blue square", 1) == ["c"]


def test_build_table_with_dense_rows(image_dir, monkeypatch):
    import dense
    monkeypatch.setattr(dense, "IMAGE_DIR", image_dir)
    queries = [{"qid": 8, "split": "test", "query": "shampoo for dry hair", "source_id": "c", "brand_in_query": False,
                "category": "Shampoo", "title_overlap": 0.25, "text_overlap": 0.5, "relevant": {"c": 2}}]
    rows = build_table(ITEMS, queries, systems=["vectors text", "vectors photo", "vectors text+photo"])
    assert rows["vectors text"]["all"]["ndcg@10"] == 1.0  # shampoo text comes first
    assert set(rows) == {"vectors text", "vectors photo", "vectors text+photo"}
