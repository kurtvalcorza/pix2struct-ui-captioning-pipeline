"""Widget-captioning dataset contract for fine-tuning: the pinned Widget Captioning sample, validation, seeded
app-disjoint splitting, BYOD loaders and JSONL export.

The default dataset is **real** and from the checkpoint's own task: widgets from the test split of Widget
Captioning (Li et al., EMNLP 2020; RICO screenshots, Deka et al. 2017), as mirrored on the Hugging Face Hub in
`bevaya/RICO-WidgetCaptioning` under **CC BY 4.0**. `google/pix2struct-widget-captioning-base` was fine-tuned
on Widget Captioning's *training* widgets, so this is continued adaptation inside the task on apps it was
not trained on, not a distribution shift. The sample is one pinned parquet shard
(`data/test-00000-of-00002.parquet`, 95,313,640 bytes) downloaded whole at the pinned dataset revision and
refused unless its SHA-256 matches the pin before `pyarrow` reads a byte of it; only the screen id, caption,
box, app-package and screenshot columns are read, and each screenshot is written to the cache once.

A record is ``{id, image_id, image, box, captions, group, category}`` — the path of the screenshot, the
widget box in pixels ``[x0, y0, x1, y1]`` (the corpus stores it relative to the image size), one or more
reference captions, the app package the screen belongs to, and a category (`small-widget` when the box covers
under `SMALL_WIDGET_AREA` of the screen — mostly icons — `large-widget` otherwise; BYOD records may carry any
label, `other` by default). Several widgets share a screen and several screens share an app; records of the
same app (`group`) are always kept in one split, as in the original benchmark split.

The shard's SHA-256 and the counts it yields are recorded by `tools/pin_corpus.py` (it needs Hub access);
until they are recorded, `fetch_corpus` refuses to read the shard rather than read an unpinned file.
"""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from .pipeline import MODEL_ID, validate_box, validate_image

CORPUS_NAME = "Widget Captioning (RICO screens, test widgets)"
CORPUS_REPO = "bevaya/RICO-WidgetCaptioning"
CORPUS_REVISION = "6ec57b56bebd722b9c646c78d0f34e1199b6d7a9"
CORPUS_RELEASE = (
    "Widget Captioning test split (3,621 widgets) as mirrored on the Hugging Face Hub, "
    "dataset revision 6ec57b56"
)
CORPUS_LICENSE = "CC BY 4.0 (declared by the Hub mirror; Li et al. 2020 Widget Captioning over RICO screens)"
CORPUS_COLUMNS = ("screenId", "captions", "bbox", "app_package_name", "image")
# The shard's pins. `sha256`, `rows` and `screens` are written by tools/pin_corpus.py from a verified
# download; while `sha256` is None the reader refuses to run.
CORPUS_FILE: dict[str, Any] = {
    "path": "data/test-00000-of-00002.parquet",
    "bytes": 95_313_640,
    "sha256": None,
    "rows": None,
    "screens": None,
}
DEFAULT_CACHE_DIR = Path("weights") / "widget-captioning"
SAMPLE_SEED = 42
# Target widget counts per split; whole apps are allocated until each target is reached, so the realised
# counts can exceed a target by the widgets of the last app added.
SAMPLE_WIDGETS = {"train": 320, "validation": 64, "test": 160}
SMALL_WIDGET_AREA = 0.01
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MIN_CAPTIONS = 1
MAX_CAPTION_CHARS = 200
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def corpus_pinned() -> bool:
    """Whether the shard's SHA-256 has been recorded (see tools/pin_corpus.py)."""
    return isinstance(CORPUS_FILE.get("sha256"), str) and len(CORPUS_FILE["sha256"]) == 64


def _hub_download(cache: Path) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            CORPUS_REPO,
            CORPUS_FILE["path"],
            repo_type="dataset",
            revision=CORPUS_REVISION,
            local_dir=str(cache),
        )
    )


def fetch_corpus(
    *, cache_dir: str | Path | None = None, downloader: Callable[[Path], Path] | None = None
) -> Path:
    """Return the path of the pinned shard, downloading it at the pinned revision when the cached copy is
    absent or drifted; refused on any size or SHA-256 mismatch, and outright while no pin is recorded."""
    if not corpus_pinned():
        raise RuntimeError(
            f"{CORPUS_REPO}@{CORPUS_REVISION[:8]} {CORPUS_FILE['path']}: no SHA-256 pin is recorded; run "
            "tools/pin_corpus.py with Hub access to record it before the sample can be read"
        )
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    local = cache / CORPUS_FILE["path"]

    def ok(path: Path) -> bool:
        return (
            path.is_file()
            and path.stat().st_size == CORPUS_FILE["bytes"]
            and _sha256_file(path) == CORPUS_FILE["sha256"]
        )

    if ok(local):
        return local
    fetched = (downloader or _hub_download)(cache)
    if not ok(fetched):
        size = fetched.stat().st_size if fetched.is_file() else None
        raise ValueError(
            f"{CORPUS_FILE['path']}: fetched {size} bytes, pinned {CORPUS_FILE['bytes']} / "
            f"{CORPUS_FILE['sha256'][:16]}…; refusing to read it"
        )
    return fetched


def read_corpus(path: str | Path) -> list[dict[str, Any]]:
    """The shard's widgets as ``{screen_id, captions, bbox, app, image_bytes}`` (bbox relative, 0..1)."""
    import pyarrow.parquet as pq

    rows = pq.read_table(str(path), columns=list(CORPUS_COLUMNS)).to_pylist()
    out = []
    for index, row in enumerate(rows):
        image = row["image"]
        data = image.get("bytes") if isinstance(image, Mapping) else None
        if not data:
            raise ValueError(f"row {index}: no screenshot bytes")
        out.append(
            {
                "screen_id": str(row["screenId"]),
                "captions": [str(c) for c in row["captions"] or []],
                "bbox": [float(v) for v in row["bbox"]],
                "app": str(row["app_package_name"]),
                "image_bytes": bytes(data),
            }
        )
    if CORPUS_FILE.get("rows") is not None and len(out) != CORPUS_FILE["rows"]:
        raise ValueError(f"shard has {len(out)} rows, pinned {CORPUS_FILE['rows']}")
    return out


def widget_category(box: Sequence[int], image_size: Sequence[int]) -> str:
    """`small-widget` when the box covers under SMALL_WIDGET_AREA of the screen, else `large-widget`."""
    x0, y0, x1, y1 = box
    area = (x1 - x0) * (y1 - y0) / float(image_size[0] * image_size[1])
    return "small-widget" if area < SMALL_WIDGET_AREA else "large-widget"


def _pixel_box(bbox: Sequence[float], size: Sequence[int]) -> list[int]:
    width, height = size
    x0, y0, x1, y1 = bbox
    return [
        max(0, round(x0 * width)),
        max(0, round(y0 * height)),
        min(width, round(x1 * width)),
        min(height, round(y1 * height)),
    ]


def _image_suffix(fmt: str | None) -> str:
    return {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get((fmt or "png").lower(), ".png")


def build_sample_dataset(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
    image_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Group the widgets by app, shuffle the apps with `seed`, and allocate whole apps to test, validation
    and train until each split's widget target is reached. Widgets without a reference caption or whose
    box is out of range or under MIN_BOX_SIDE are skipped; the screenshots of the chosen widgets are written
    to `image_dir` under their screen id."""
    sizes = dict(sizes or SAMPLE_WIDGETS)
    out_dir = Path(image_dir) if image_dir is not None else DEFAULT_CACHE_DIR / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    if CORPUS_FILE.get("screens") is not None:
        screens = len({r["screen_id"] for r in rows})
        if screens != CORPUS_FILE["screens"]:
            raise ValueError(f"shard has {screens} distinct screens, pinned {CORPUS_FILE['screens']}")
    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row["app"]), []).append(row)
    order = sorted(groups)
    random.Random(seed).shuffle(order)
    out: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    written: dict[str, tuple[str, list[int]]] = {}
    for app in order:
        open_splits = [name for name in ("test", "validation", "train") if len(out[name]) < sizes[name]]
        if not open_splits:
            break
        target = open_splits[0]
        for row in groups[app]:
            captions = [" ".join(c.split()) for c in row["captions"] if c and c.strip()]
            if not captions:
                continue
            screen = str(row["screen_id"])
            if screen not in written:
                with Image.open(io.BytesIO(row["image_bytes"])) as image:
                    size, fmt = list(image.size), image.format
                path = out_dir / f"{screen}{_image_suffix(fmt)}"
                if not path.is_file() or _sha256_file(path) != _sha256_bytes(row["image_bytes"]):
                    path.write_bytes(row["image_bytes"])
                written[screen] = (str(path), size)
            path, size = written[screen]
            box = _pixel_box(row["bbox"], size)
            try:
                validate_box(box, (size[0], size[1]))
            except (TypeError, ValueError):
                continue
            out[target].append(
                {
                    "id": f"{target}-{len(out[target]):04d}",
                    "image_id": screen,
                    "image": path,
                    "box": box,
                    "captions": captions,
                    "group": str(app),
                    "category": widget_category(box, size),
                }
            )
    short = {name: (len(out[name]), sizes[name]) for name in out if len(out[name]) < sizes[name]}
    if short:
        raise ValueError(f"the shard's apps do not fill the split targets: {short}")
    return {"train": out["train"], "validation": out["validation"], "test": out["test"]}


def fetch_sample_dataset(
    *,
    cache_dir: str | Path | None = None,
    downloader: Callable[[Path], Path] | None = None,
    seed: int = SAMPLE_SEED,
    sizes: Mapping[str, int] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """The tutorial splits from the pinned shard."""
    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    rows = read_corpus(fetch_corpus(cache_dir=cache, downloader=downloader))
    return build_sample_dataset(rows, seed=seed, sizes=sizes, image_dir=cache / "images")


def _check_record(record: Any, index: int, *, base_dir: Path | None) -> dict[str, Any]:
    label = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{label} must be a mapping with id/image/box/captions")
    for key in ("id", "image", "box", "captions"):
        if key not in record:
            raise ValueError(f"{label} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{label}: id must match {_ID_RE.pattern}")
    image_ref = record["image"]
    if not isinstance(image_ref, (str, Path)) or not str(image_ref).strip():
        raise ValueError(f"{label}: image must be a file path")
    path = Path(image_ref)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    if not path.is_file():
        raise ValueError(f"{label}: image file not found: {path}")
    try:
        with Image.open(path) as handle:
            handle.load()
            validate_image(handle)
            width, height = handle.size
        box = validate_box(record["box"], (width, height))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label}: {exc}") from exc
    except OSError as exc:
        raise ValueError(f"{label}: image cannot be decoded: {exc}") from exc
    captions = record["captions"]
    if isinstance(captions, str) or not isinstance(captions, Sequence) or len(captions) < MIN_CAPTIONS:
        raise ValueError(f"{label}: captions must be a list of at least {MIN_CAPTIONS} reference caption(s)")
    checked_captions = [" ".join(str(c).split()) for c in captions]
    if not all(checked_captions):
        raise ValueError(f"{label}: every reference caption must be a non-empty string")
    if any(len(c) > MAX_CAPTION_CHARS for c in checked_captions):
        raise ValueError(f"{label}: a reference caption exceeds MAX_CAPTION_CHARS={MAX_CAPTION_CHARS}")
    image_id = str(record.get("image_id", path.name))
    return {
        "id": rid,
        "image_id": image_id,
        "image": str(path),
        "image_size": [width, height],
        "box": box,
        "captions": checked_captions,
        "group": str(record.get("group", image_id)),
        "category": str(record.get("category", "other")),
    }


def validate_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    min_records: int = MIN_RECORDS,
    max_records: int = MAX_RECORDS,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Structural validation of a widget-captioning dataset (every screenshot opened and decoded, every box
    held to the same checks `caption` applies); raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, (str, bytes)):
        raise ValueError("records must be a list of {id, image, box, captions} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    base = Path(base_dir) if base_dir is not None else None
    checked = []
    ids: set[str] = set()
    for index, record in enumerate(records):
        item = _check_record(record, index, base_dir=base)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        checked.append(item)
    return {
        "records": checked,
        "n_records": len(checked),
        "unique_images": len({r["image_id"] for r in checked}),
        "unique_groups": len({r["group"] for r in checked}),
        "categories": dict(Counter(r["category"] for r in checked)),
        "captions_per_widget": {
            "min": min(len(r["captions"]) for r in checked),
            "max": max(len(r["captions"]) for r in checked),
        },
        "caption_words": {
            "min": min(len(c.split()) for r in checked for c in r["captions"]),
            "max": max(len(c.split()) for r in checked for c in r["captions"]),
        },
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    payload = [
        [
            r["id"],
            r["image_id"],
            list(r["box"]),
            list(r["captions"]),
            r.get("group", ""),
            r.get("category", ""),
        ]
        for r in records
    ]
    return _sha256_bytes(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def reference_captions(record: Mapping[str, Any]) -> list[str]:
    """The reference captions of a record."""
    return [str(c) for c in record["captions"]]


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no app (`group`) and no screenshot appears in two splits (leakage check)."""
    seen: dict[tuple[str, str], str] = {}
    for name, records in splits.items():
        for record in records:
            image_id = str(record.get("image_id", record["id"]))
            for key in (("app", str(record.get("group", image_id))), ("screen", image_id)):
                if key in seen and seen[key] != name:
                    raise ValueError(f"{key[0]} {key[1]!r} appears in both {seen[key]} and {name}")
                seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]],
    *,
    val_fraction: float = 0.15,
    test_fraction: float = 0.2,
    seed: int = 0,
    base_dir: str | Path | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Seeded split of a BYOD dataset into train/validation/test **by group** (app, defaulting to the
    screenshot): every widget of the same group lands in the same split."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records, base_dir=base_dir)["records"]
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in checked:
        groups.setdefault(record["group"], []).append(record)
    order = list(groups.values())
    random.Random(seed).shuffle(order)
    n_test = max(1, round(len(checked) * test_fraction))
    n_val = round(len(checked) * val_fraction)
    splits: dict[str, list[dict[str, Any]]] = {"test": [], "validation": [], "train": []}
    for group in order:
        if len(splits["test"]) < n_test:
            splits["test"].extend(group)
        elif len(splits["validation"]) < n_val:
            splits["validation"].extend(group)
        else:
            splits["train"].extend(group)
    if len(splits["train"]) < MIN_RECORDS:
        raise ValueError(
            f"split leaves {len(splits['train'])} training records; at least {MIN_RECORDS} are required"
        )
    return splits


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Read records from a JSON array or a JSONL file of ``{id, image, box, captions}`` objects; `image`
    paths are resolved relative to the file's directory by `validate_dataset(..., base_dir=...)`."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"dataset not found: {file_path}")
    suffix = file_path.suffix.lower()
    text = file_path.read_text(encoding="utf-8")
    if suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON dataset must be an array of records")
        return data
    raise ValueError("BYOD datasets must be .json or .jsonl")


def write_dataset_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """One record per line in the shape `load_byod_dataset` reads back (image paths as given)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ("id", "image_id", "image", "box", "captions", "group", "category")
    with open(out, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({k: record[k] for k in keys if k in record}, ensure_ascii=False) + "\n")
    return out
