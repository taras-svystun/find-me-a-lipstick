"""A1: download the first ~135 MB of the Beauty metadata and keep Skin Care, Makeup, Fragrance.

Usage: .venv/bin/python scripts/fetch_items.py
Writes data/raw_head.jsonl (the byte range, cached) and data/items.jsonl (kept items).
"""

import http.client
import json
import urllib.request
from collections import Counter
from pathlib import Path

URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/resolve/main/"
    "raw/meta_categories/meta_Beauty_and_Personal_Care.jsonl"
)
N_BYTES = 135_000_000  # ~35% of records are kept, so this gives ~15k items
KEEP = {"Skin Care", "Makeup", "Fragrance"}
DATA = Path(__file__).resolve().parent.parent / "data"


def download_head(path: Path) -> bytes:
    raw = bytearray(path.read_bytes()[:N_BYTES] if path.exists() else b"")
    for _ in range(10):  # the CDN sometimes closes the connection early: resume where it stopped
        if len(raw) >= N_BYTES:
            break
        req = urllib.request.Request(URL, headers={"Range": f"bytes={len(raw)}-{N_BYTES - 1}"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206, f"expected 206 Partial Content, got {resp.status}"
            try:
                raw += resp.read()
            except http.client.IncompleteRead as e:
                raw += e.partial
    assert len(raw) == N_BYTES, f"got {len(raw):,} of {N_BYTES:,} bytes"
    path.write_bytes(raw)
    return bytes(raw)


def main_image(images: list[dict]) -> str | None:
    for img in images:
        if img.get("variant") == "MAIN" and img.get("large"):
            return img["large"]
    return images[0].get("large") if images else None


def to_item(rec: dict) -> dict | None:
    cats = rec.get("categories") or []
    if len(cats) < 2 or cats[1] not in KEEP:
        return None
    details = rec.get("details") or {}
    return {
        "id": rec["parent_asin"],
        "title": rec.get("title") or "",
        "brand": details.get("Brand") or rec.get("store") or None,
        "categories": cats[1:],  # drop the root "Beauty & Personal Care"
        "features": rec.get("features") or [],
        "description": " ".join(rec.get("description") or []),
        "price": rec.get("price"),
        "image_url": main_image(rec.get("images") or []),
    }


def main() -> None:
    DATA.mkdir(exist_ok=True)
    raw = download_head(DATA / "raw_head.jsonl")
    lines = raw.split(b"\n")
    complete, cut = lines[:-1], lines[-1]  # a range request ends mid-record
    records = [json.loads(line) for line in complete if line]

    items = [it for rec in records if (it := to_item(rec))]
    with open(DATA / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    print(f"bytes {len(raw):,}  records {len(records):,}  dropped cut tail {len(cut)} bytes")
    print(f"kept {len(items):,} items: {dict(Counter(it['categories'][0] for it in items))}")
    for field in ["brand", "features", "description", "price", "image_url"]:
        share = sum(bool(it[field]) for it in items) / len(items)
        print(f"  {field:12s} {share:5.0%}")


if __name__ == "__main__":
    main()
