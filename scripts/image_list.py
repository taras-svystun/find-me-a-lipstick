"""A2: write a curl config that downloads one main image per item, and check what arrived.

Usage:
  .venv/bin/python scripts/image_list.py          # writes data/images.curl
  curl --parallel --parallel-max 16 -sS -K data/images.curl   # run in your own terminal
  .venv/bin/python scripts/image_list.py --check  # counts images, lists missing or placeholder ones
"""

import json
import sys
from pathlib import Path

from PIL import Image, UnidentifiedImageError

DATA = Path(__file__).resolve().parent.parent / "data"
IMAGES = DATA / "images"
PLACEHOLDER = (60, 40)  # Amazon serves a 60x40 "no photo" image (GIF or JPEG) instead of a 404


def items() -> list[dict]:
    return [json.loads(line) for line in open(DATA / "items.jsonl")]


def write_config() -> None:
    IMAGES.mkdir(exist_ok=True)
    with open(DATA / "images.curl", "w") as f:
        for it in items():
            if it["image_url"]:
                f.write(f'url = "{it["image_url"]}"\noutput = "{IMAGES / it["id"]}.jpg"\n')
    print(f"wrote {DATA / 'images.curl'}")


def is_real_photo(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with Image.open(path) as img:
            return img.size != PLACEHOLDER
    except UnidentifiedImageError:  # an empty file or a saved HTTP error page
        return False


def check() -> None:
    wanted = [it["id"] for it in items() if it["image_url"]]
    missing = [i for i in wanted if not is_real_photo(IMAGES / f"{i}.jpg")]
    print(f"images {len(wanted) - len(missing):,} of {len(wanted):,}; missing {len(missing)}")
    if missing:
        (DATA / "images_missing.txt").write_text("\n".join(missing) + "\n")
        print(f"missing ids in {DATA / 'images_missing.txt'}")


if __name__ == "__main__":
    check() if "--check" in sys.argv else write_config()
