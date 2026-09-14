"""Role-helper contract: validate_inputs (validation stage) and evaluation_report (evaluation stage)."""

from __future__ import annotations

import pytest
from PIL import Image

from pix2struct_ui_captioning_pipeline import (
    DEFAULT_MAX_NEW_TOKENS,
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    caption_tokens,
    evaluation_report,
    keyword_hits,
    normalize_caption,
    unigram_f1,
    validate_inputs,
)

BOXES = [[90, 820, 450, 880], [30, 120, 510, 170]]


def _image(width: int = 540, height: int = 960) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _result(caption: str, box: list[int] | None = None, truncated: bool = False) -> dict:
    return {"caption": caption, "box": box or BOXES[0], "truncated": truncated}


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), BOXES, names=["screen.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["inputs"] == [{"id": "screen.png", "mode": "RGB", "size": [540, 960]}]
    assert manifest["boxes"] == BOXES
    assert manifest["generation"] == {
        "max_new_tokens": DEFAULT_MAX_NEW_TOKENS,
        "do_sample": False,
        "decoding": "greedy",
    }
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_request() -> None:
    manifest = validate_inputs(_image(), [(10.2, 10, 50, 50.4)], max_new_tokens=8)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["boxes"] == [[10, 10, 50, 50]]
    assert manifest["generation"]["max_new_tokens"] == 8


def test_validate_inputs_rejects_like_caption() -> None:
    with pytest.raises(TypeError, match="non-empty sequence"):
        validate_inputs(_image(), [])
    with pytest.raises(TypeError, match="not a single box"):
        validate_inputs(_image(), BOXES[0])
    with pytest.raises(ValueError, match="outside"):
        validate_inputs(_image(), [[0, 0, 600, 100]])
    with pytest.raises(ValueError, match="MAX_NEW_TOKENS"):
        validate_inputs(_image(), BOXES, max_new_tokens=0)
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8), [[0, 0, 4, 4]])
    with pytest.raises(ValueError, match="exactly one entry"):
        validate_inputs(_image(), BOXES, names=["a", "b"])


def test_normalize_caption_tokens_and_keyword_hits() -> None:
    assert normalize_caption("  Go to New Message! ") == "go to new message"
    assert caption_tokens("Search bar.") == ["search", "bar"]
    assert keyword_hits("go to new message", ["message", "new message", "search"]) == {
        "message": True,
        "new message": True,
        "search": False,
    }


def test_unigram_f1_best_reference() -> None:
    assert unigram_f1("search bar", ["search bar"]) == 1.0
    assert unigram_f1("go to profile", ["profile tab", "open profile"]) == pytest.approx(0.4)
    assert unigram_f1("", ["search bar"]) == 0.0
    with pytest.raises(ValueError, match="references"):
        unigram_f1("x", [])


def test_evaluation_report_not_measurable_without_references() -> None:
    report = evaluation_report([_result("search bar")], sample_kind="synthetic")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_widgets"] == 1 and report["truncated"] == [False]
    assert "CIDEr" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "no score" in report["score_semantics"]


def test_evaluation_report_sample_sanity_with_references() -> None:
    results = [_result("search bar"), _result("go to next", BOXES[1], truncated=True)]
    report = evaluation_report(results, [["search bar"], ["open settings", "settings"]])
    assert report["verdict"] == "sample-sanity"
    assert report["truncated"] == [False, True]
    by_id = {metric["id"]: metric for metric in report["metrics"]}
    assert by_id["unigram_f1"]["value"] == pytest.approx(0.5)
    assert "CIDEr" in by_id["unigram_f1"]["relation_to_benchmarks"]
    assert [entry["unigram_f1"] for entry in report["per_widget"]] == [1.0, 0.0]
    assert report["per_widget"][1]["box"] == BOXES[1]


def test_evaluation_report_rejects_mismatched_or_empty_references() -> None:
    with pytest.raises(ValueError, match="references has"):
        evaluation_report([_result("a")], [["a"], ["b"]])
    with pytest.raises(ValueError, match="non-empty sequence"):
        evaluation_report([_result("a")], [[]])
    with pytest.raises(ValueError, match="results"):
        evaluation_report([], None)
