"""Offline checks of the widget-captioning dataset contract, the captioning metrics and baselines, the
pinned-shard reader (refusal while unpinned, digest checks when pinned) and the adaptation surface — no
model weights, no network."""

from __future__ import annotations

import hashlib
import io
import json

import pytest
from PIL import Image, ImageDraw

from pix2struct_ui_captioning_pipeline import (
    MAX_RECORDS,
    MIN_RECORDS,
    Pix2StructWidgetCaptioningPipeline,
    bleu4,
    build_sample_dataset,
    caption_metrics,
    check_split_disjoint,
    cider_d,
    colour_neighbour_baseline,
    constant_caption_baseline,
    corpus_pinned,
    dataset_digest,
    fetch_corpus,
    load_byod_dataset,
    read_corpus,
    rouge_l,
    split_dataset,
    validate_dataset,
    widget_category,
    write_dataset_jsonl,
)
from pix2struct_ui_captioning_pipeline import samples as samples_module

COLOURS = ["red", "green", "blue", "yellow", "purple", "orange"]


def _screen(colour: str, size=(120, 200)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    ImageDraw.Draw(image).rectangle([10, 10, 60, 40], fill=colour)
    return image


@pytest.fixture
def records(tmp_path):
    out = []
    for i in range(12):
        path = tmp_path / f"s{i // 2}.png"
        if not path.is_file():
            _screen(COLOURS[(i // 2) % len(COLOURS)]).save(path)
        out.append(
            {
                "id": f"w{i:02d}",
                "image_id": f"s{i // 2}",
                "image": str(path),
                "box": [10, 10, 60, 40] if i % 2 == 0 else [70, 150, 110, 190],
                "captions": [f"{COLOURS[(i // 2) % len(COLOURS)]} button", "go back"],
                "group": f"app{i // 4 if i < 4 else i // 2}",
                "category": "large-widget",
            }
        )
    return out


# ---- metrics and baselines ------------------------------------------------------------------------------


def test_caption_metrics_are_perfect_on_the_references_and_zero_when_disjoint():
    refs = [["search bar", "search"], ["go back", "back button"]]
    perfect = caption_metrics(["search bar", "go back"], refs)
    assert perfect["bleu4"] == pytest.approx(0.0, abs=1.0) and perfect["rouge_l"] == pytest.approx(1.0)
    assert all(score > 0 for score in cider_d(["search bar", "go back"], refs))
    assert rouge_l("zebra", ["search bar"]) == 0.0 and bleu4(["zebra"], [["search bar"]]) == 0.0


def test_baselines_use_the_training_split_and_the_widget_crop(records):
    constant = constant_caption_baseline(records[:8], records[8:])
    assert constant["n"] == 4 and constant["baseline"].startswith("constant caption")
    neighbour = colour_neighbour_baseline(records[:8], records[8:])
    assert neighbour["n"] == 4 and "training widgets" in neighbour["baseline"]


def test_widget_category_splits_on_the_area_threshold():
    assert widget_category([0, 0, 10, 10], [100, 200]) == "small-widget"
    assert widget_category([0, 0, 50, 50], [100, 200]) == "large-widget"


# ---- dataset contract -----------------------------------------------------------------------------------


def test_validate_dataset_reports_structure_and_digest(records):
    manifest = validate_dataset(records)
    assert manifest["n_records"] == 12 and manifest["unique_images"] == 6 and manifest["unique_groups"] == 5
    assert manifest["captions_per_widget"] == {"min": 2, "max": 2}
    assert manifest["digest"] == dataset_digest(manifest["records"])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: {**r, "id": "bad id!"}, "id must match"),
        (lambda r: {**r, "image": "/does/not/exist.png"}, "image file not found"),
        (lambda r: {**r, "box": [0, 0, 500, 500]}, "outside"),
        (lambda r: {**r, "box": [0, 0, 2, 2]}, "MIN_BOX_SIDE"),
        (lambda r: {**r, "box": "0,0,1,1"}, "four numbers"),
        (lambda r: {**r, "captions": []}, "at least 1"),
        (lambda r: {**r, "captions": ["  "]}, "non-empty"),
        (lambda r: {**r, "captions": ["x" * 201]}, "MAX_CAPTION_CHARS"),
        (lambda r: {k: v for k, v in r.items() if k != "box"}, "missing 'box'"),
    ],
)
def test_validate_dataset_refuses_contract_violations(records, mutate, message):
    broken = [mutate(records[0]), *records[1:]]
    with pytest.raises(ValueError, match=message):
        validate_dataset(broken)


def test_validate_dataset_refuses_duplicates_and_size_bounds(records):
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], *records[:8]])
    with pytest.raises(ValueError, match=f"{MIN_RECORDS}..{MAX_RECORDS}"):
        validate_dataset(records[:3])


def test_split_dataset_keeps_each_app_in_one_split(records):
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=1)
    counts = check_split_disjoint(splits)
    assert sum(counts.values()) == 12
    for name, part in splits.items():
        others = {r["group"] for other, rest in splits.items() if other != name for r in rest}
        assert not {r["group"] for r in part} & others


def test_check_split_disjoint_refuses_a_shared_app_or_screen(records):
    with pytest.raises(ValueError, match="app 'app0' appears in both"):
        check_split_disjoint({"train": records[:1], "test": records[2:3]})
    moved = {**records[1], "group": "elsewhere"}
    with pytest.raises(ValueError, match="screen 's0' appears in both"):
        check_split_disjoint({"train": records[:1], "test": [moved]})


def test_byod_jsonl_round_trip(records, tmp_path):
    path = write_dataset_jsonl(records, tmp_path / "out" / "records.jsonl")
    again = load_byod_dataset(path)
    assert [r["id"] for r in again] == [r["id"] for r in records]
    assert validate_dataset(again)["digest"] == validate_dataset(records)["digest"]


# ---- pinned shard ---------------------------------------------------------------------------------------


def _shard(tmp_path, n_apps=20, screens_per_app=2, widgets_per_screen=3):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = []
    for a in range(n_apps):
        for s in range(screens_per_app):
            buffer = io.BytesIO()
            _screen(COLOURS[(a + s) % len(COLOURS)], size=(100 + a, 200)).save(buffer, format="PNG")
            data = buffer.getvalue()
            for w in range(widgets_per_screen):
                bbox = [0.1, 0.1 + 0.2 * w, 0.5 + 0.1 * w, 0.25 + 0.2 * w]
                captions = [f"widget {w}", "open menu"] if w else []  # the first widget has no caption
                rows.append(
                    {
                        "screenId": a * 10 + s,
                        "captions": captions,
                        "bbox": bbox,
                        "app_package_name": f"com.example.app{a}",
                        "image": {"bytes": data, "path": None},
                    }
                )
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.Table.from_pylist(rows), str(path))
    return path


def test_fetch_corpus_refuses_while_no_pin_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setitem(samples_module.CORPUS_FILE, "sha256", None)
    assert not corpus_pinned()
    with pytest.raises(RuntimeError, match="no SHA-256 pin is recorded"):
        fetch_corpus(cache_dir=tmp_path, downloader=lambda cache: pytest.fail("must not download"))


def test_fetch_corpus_checks_size_and_digest(tmp_path, monkeypatch):
    shard = _shard(tmp_path)
    data = shard.read_bytes()
    monkeypatch.setitem(samples_module.CORPUS_FILE, "bytes", len(data))
    monkeypatch.setitem(samples_module.CORPUS_FILE, "sha256", hashlib.sha256(data).hexdigest())
    assert corpus_pinned()
    assert fetch_corpus(cache_dir=tmp_path / "cache", downloader=lambda cache: shard) == shard
    monkeypatch.setitem(samples_module.CORPUS_FILE, "sha256", "0" * 64)
    with pytest.raises(ValueError, match="refusing to read it"):
        fetch_corpus(cache_dir=tmp_path / "cache", downloader=lambda cache: shard)


def test_read_corpus_and_build_sample_split_by_app(tmp_path, monkeypatch):
    shard = _shard(tmp_path)
    monkeypatch.setitem(samples_module.CORPUS_FILE, "rows", None)
    monkeypatch.setitem(samples_module.CORPUS_FILE, "screens", None)
    rows = read_corpus(shard)
    assert len(rows) == 120 and rows[1]["captions"] == ["widget 1", "open menu"]
    sizes = {"train": 30, "validation": 8, "test": 16}
    splits = build_sample_dataset(rows, seed=3, sizes=sizes, image_dir=tmp_path / "images")
    counts = check_split_disjoint(splits)
    assert all(counts[name] >= sizes[name] for name in sizes)
    assert all(counts[name] % 4 == 0 for name in sizes)  # whole apps only (2 screens x 2 captioned widgets)
    record = splits["train"][0]
    with Image.open(record["image"]) as image:
        width, height = image.size
    assert record["box"][2] <= width and record["box"][3] <= height
    again = build_sample_dataset(rows, seed=3, sizes=sizes, image_dir=tmp_path / "images")
    assert [r["id"] for r in again["test"]] == [r["id"] for r in splits["test"]]
    for part in splits.values():
        validate_dataset(part)
    monkeypatch.setitem(samples_module.CORPUS_FILE, "screens", 3)
    with pytest.raises(ValueError, match="pinned 3"):
        build_sample_dataset(rows, sizes=sizes, image_dir=tmp_path / "images")
    monkeypatch.setitem(samples_module.CORPUS_FILE, "screens", None)
    too_many = {"train": 200, "validation": 1, "test": 1}
    with pytest.raises(ValueError, match="do not fill"):
        build_sample_dataset(rows, sizes=too_many, image_dir=tmp_path / "i2")


# ---- adaptation surface without a model -------------------------------------------------------------------


def _runner(annotated, max_new_tokens):
    return {"caption": "go back", "new_tokens": 2}


def test_evaluate_with_an_injected_runner_scores_captions(records):
    pipe = Pix2StructWidgetCaptioningPipeline(_runner)
    metrics = pipe.evaluate(records)
    assert metrics["n"] == 12 and metrics["rouge_l"] == pytest.approx(1.0)
    assert metrics["verdict"] == "measured-small-sample" and metrics["adapted"] is False
    assert pipe.predict(records[:2]) == ["go back", "go back"]


def test_adaptation_needs_a_loaded_model(records, tmp_path):
    pipe = Pix2StructWidgetCaptioningPipeline(_runner)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(records)
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)
    with pytest.raises(ValueError, match="1..12"):
        pipe._trainable_names(13)


def test_load_artifact_refuses_a_foreign_manifest(tmp_path):
    pipe = Pix2StructWidgetCaptioningPipeline(_runner)
    (tmp_path / "manifest.json").write_text(json.dumps({"format": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
