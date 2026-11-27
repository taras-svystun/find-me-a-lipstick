"""A6: dense search. CLIP and SigLIP2 put texts and photos into one vector space, so a text query
can be scored against an item's text and against its photo.

  text       = cos(q, text_d)         photo = cos(q, photo_d)
  text+photo = w * z(text) + (1 - w) * z(photo)

z standardises one query's scores over all items (subtract the mean, divide by the standard
deviation). Text-text cosines sit near 0.7 and text-photo cosines near 0.1-0.2 (the modality gap),
so without z the weight w would mostly correct the scale instead of saying how much each side counts.

Usage:
  .venv/bin/python scripts/dense.py --embed clip      # item vectors -> data/vectors/clip.npz (~12 min on CPU)
  .venv/bin/python scripts/dense.py --embed siglip2   # (~25 min on CPU)
  .venv/bin/python scripts/dense.py --tune            # model and w on the dev queries
"""

import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("HF_HOME", str(ROOT / "models"))  # models are downloaded once (README), then used offline
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from bm25 import item_text  # noqa: E402  (same fields as BM25, so rows differ by method, not input)

MODELS = {"clip": "openai/clip-vit-base-patch32", "siglip2": "google/siglip2-base-patch16-224"}
MODEL = "siglip2"  # chosen on the dev queries with --tune (docs/journal.md, A6)
W = 0.5  # weight of the text side in "vectors text+photo", chosen with MODEL
IMAGE_DIR = ROOT / "data" / "images"
VECTOR_DIR = ROOT / "data" / "vectors"
W_GRID = [round(0.1 * i, 1) for i in range(11)]


def normalize_text(text: str) -> str:
    """Lower case and single spaces. CLIP's tokenizer lower-cases by itself; SigLIP2's (Gemma) does not,
    so "Red Lipstick" and "red lipstick" would get different vectors (cosine 0.83)."""
    return " ".join(text.lower().split())


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


class Encoder:
    """One model's text and photo towers, on CPU (the sandbox has no GPU access)."""

    def __init__(self, model: str):
        import torch
        from transformers import AutoModel, AutoProcessor
        self.torch, self.name = torch, MODELS[model]
        self.model = AutoModel.from_pretrained(self.name).eval()
        self.processor = AutoProcessor.from_pretrained(self.name)
        # SigLIP2 was trained on texts padded to 64 tokens; CLIP stops at 77 tokens
        self.text_args = (dict(padding="max_length", max_length=64) if model == "siglip2" else dict(padding=True))

    def _unit(self, out) -> np.ndarray:
        out = out if self.torch.is_tensor(out) else out.pooler_output
        return self.torch.nn.functional.normalize(out, dim=-1).numpy().astype(np.float32)

    def texts(self, texts: list[str]) -> np.ndarray:
        """Unit vectors. A text longer than the token limit is cut: an item vector sees the brand, the
        title and the first categories, and rarely the description."""
        with self.torch.no_grad():
            x = self.processor(text=[normalize_text(t) for t in texts], truncation=True,
                               return_tensors="pt", **self.text_args)
            return self._unit(self.model.get_text_features(**x))

    def photos(self, paths: list[Path]) -> np.ndarray:
        from PIL import Image
        with self.torch.no_grad():
            x = self.processor(images=[Image.open(p).convert("RGB") for p in paths], return_tensors="pt")
            return self._unit(self.model.get_image_features(**x))


@lru_cache(maxsize=None)
def get_encoder(model: str) -> Encoder:
    return Encoder(model)


def embed_items(items: list[dict], model: str, image_dir: Path, batch: int = 64, log: bool = False) -> dict:
    paths = [Path(image_dir) / f'{it["id"]}.jpg' for it in items]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} photos missing, first {missing[0]}")
    enc = get_encoder(model)
    text, photo = [], []
    for start in range(0, len(items), batch):
        text.append(enc.texts([item_text(it) for it in items[start:start + batch]]))
        photo.append(enc.photos(paths[start:start + batch]))
        if log and start % (batch * 20) == 0:
            print(f"{model}: {start + batch}/{len(items)}", flush=True)
    return {"ids": [it["id"] for it in items], "text": np.concatenate(text), "photo": np.concatenate(photo)}


def load_vectors(items: list[dict], model: str, image_dir: Path) -> dict:
    """The saved vectors when they cover exactly these items; otherwise embed them now (tiny test corpora)."""
    path = VECTOR_DIR / f"{model}.npz"
    ids = [it["id"] for it in items]
    if path.exists():
        saved = np.load(path)
        if saved["ids"].tolist() == ids:
            return {"ids": ids, "text": saved["text"], "photo": saved["photo"]}
    return embed_items(items, model, image_dir)


class Dense:
    def __init__(self, ids: list[str], text: np.ndarray, photo: np.ndarray, encode_query, w: float):
        self.ids, self.text, self.photo, self.encode_query, self.w = ids, text, photo, encode_query, w

    @classmethod
    def from_items(cls, items: list[dict], model: str, w: float, image_dir: Path | None = None) -> "Dense":
        v = load_vectors(items, model, image_dir or IMAGE_DIR)
        return cls(v["ids"], v["text"], v["photo"], lambda q: get_encoder(model).texts([q])[0], w)

    def scores(self, query: str) -> np.ndarray:
        q = self.encode_query(normalize_text(query))
        if self.w == 1.0:
            return self.text @ q
        if self.w == 0.0:
            return self.photo @ q
        return self.w * zscore(self.text @ q) + (1 - self.w) * zscore(self.photo @ q)

    def search(self, query: str, k: int = 10, with_scores: bool = False) -> list:
        """Top k item ids; ties go to the earlier item (a stable sort), so runs are reproducible."""
        s = self.scores(query)
        best = np.argsort(-s, kind="stable")[:k]
        if with_scores:
            return [(self.ids[i], float(s[i])) for i in best]
        return [self.ids[i] for i in best]


def tune(items: list[dict], queries: list[dict]) -> None:
    """nDCG@10 on all dev queries for each model and w. The pick is the best text+photo cell; ties
    go to the larger w (fewer photo terms to explain)."""
    from metrics import evaluate
    dev = [q for q in queries if q["split"] == "dev"]
    print("model   " + " ".join(f"w={w:<4}" for w in W_GRID))
    for model in MODELS:
        v = load_vectors(items, model, IMAGE_DIR)
        enc = get_encoder(model)
        qv = {q["qid"]: enc.texts([q["query"]])[0] for q in dev}
        cells = []
        for w in W_GRID:
            d = Dense(v["ids"], v["text"], v["photo"], None, w)
            run = {}
            for q in dev:
                d.encode_query = lambda _, vec=qv[q["qid"]]: vec  # each dev query is encoded once
                run[q["qid"]] = d.search(q["query"], 10)
            cells.append(evaluate(run, queries, "dev")["all"]["ndcg@10"])
        print(f"{model:<8}" + " ".join(f"{c:<6.3f}" for c in cells))


def main() -> None:
    items = [json.loads(line) for line in open(ROOT / "data" / "items.jsonl")]
    if "--embed" in sys.argv:
        model = sys.argv[sys.argv.index("--embed") + 1]
        v = embed_items(items, model, IMAGE_DIR, log=True)
        VECTOR_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(VECTOR_DIR / f"{model}.npz", ids=np.array(v["ids"]), text=v["text"], photo=v["photo"])
    elif "--tune" in sys.argv:
        tune(items, [json.loads(line) for line in open(ROOT / "eval" / "queries.jsonl")])


if __name__ == "__main__":
    main()
