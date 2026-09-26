"""Record the pins of the Widget Captioning sample shard in src/pix2struct_ui_captioning_pipeline/samples.py.

Needs Hub access. Downloads `CORPUS_FILE["path"]` of `CORPUS_REPO` at the immutable `CORPUS_REVISION`,
checks its size against the committed pin, computes its SHA-256, counts its rows and distinct screens,
rewrites the `sha256`, `rows` and `screens` entries of `CORPUS_FILE`, and prints the realised split counts
of the default sample. Regenerate the notebook afterwards (`python tools/build_notebook.py`), because the
carried module changed. Run it once; re-running on an already-pinned file only confirms the pins.

    python tools/pin_corpus.py [--cache weights/widget-captioning]
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pix2struct_ui_captioning_pipeline import samples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=str(ROOT / "weights" / "widget-captioning"))
    args = parser.parse_args()
    cache = Path(args.cache)
    path = samples._download_corpus(cache)
    size = path.stat().st_size
    if size != samples.CORPUS_FILE["bytes"]:
        pinned_size = samples.CORPUS_FILE["bytes"]
        raise SystemExit(f"{path}: {size} bytes, committed pin {pinned_size}; refusing to pin")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = samples.read_corpus(path)
    screens = len({r["screen_id"] for r in rows})
    pinned = samples.CORPUS_FILE.get("sha256")
    if pinned is not None and pinned != digest:
        raise SystemExit(f"{path}: sha256 {digest} differs from the recorded pin {pinned}; investigate")
    module = ROOT / "src" / "pix2struct_ui_captioning_pipeline" / "samples.py"
    text = module.read_text(encoding="utf-8")
    new, count = re.subn(
        r'("sha256": )(None|"[0-9a-f]{64}"),\n(\s*"rows": )(None|\d+),\n(\s*"screens": )(None|\d+),',
        lambda m: f'{m.group(1)}"{digest}",\n{m.group(3)}{len(rows)},\n{m.group(5)}{screens},',
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("could not find the CORPUS_FILE pin entries in samples.py")
    module.write_text(new, encoding="utf-8")
    samples.CORPUS_FILE.update({"sha256": digest, "rows": len(rows), "screens": screens})
    splits = samples.build_sample_dataset(rows, image_dir=cache / "images")
    print({"sha256": digest, "rows": len(rows), "screens": screens})
    for name, part in splits.items():
        apps = len({r["group"] for r in part})
        print({name: {"widgets": len(part), "screens": len({r["image_id"] for r in part}), "apps": apps}})
    print("pins written to", module.relative_to(ROOT), "- now run: python tools/build_notebook.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
