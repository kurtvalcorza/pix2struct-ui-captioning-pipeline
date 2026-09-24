"""Adaptation checks on a real Pix2Struct network: a small randomly initialised model built offline from the
committed snapshot config, processor (header rendering off) and tokenizer (always), and the pinned checkpoint
itself where its weights are staged (local pre-flight). They cover a referenced evaluation, a one-epoch
adaptation of the last decoder block, the trainable-tensor scope, the artifact round trip, the loader's
tensor-set check, the transactional guarantee and — where CUDA is visible — the same path on the
accelerator."""

from __future__ import annotations

import hashlib
import json
import shutil

import pytest
from PIL import Image, ImageDraw

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from pix2struct_ui_captioning_pipeline import (  # noqa: E402  -- after the importorskip guards
    DEFAULT_WEIGHTS_DIR,
    WEIGHT_FILE,
    Pix2StructWidgetCaptioningPipeline,
)

COLOURS = ["red", "green", "blue", "yellow"]
REAL_ONE_BLOCK_TRAINABLE = 9_440_256  # decoder.layer.11.* (9,439,488) + decoder.final_layer_norm (768)


def _tiny_pipeline(seed: int = 0) -> Pix2StructWidgetCaptioningPipeline:
    """A 3-layer, 32-wide Pix2Struct with the checkpoint's real vocabulary, processor and tokenizer."""
    from transformers import Pix2StructConfig, Pix2StructForConditionalGeneration, Pix2StructProcessor

    processor = Pix2StructProcessor.from_pretrained(str(DEFAULT_WEIGHTS_DIR), local_files_only=True)
    processor.image_processor.max_patches = 64
    config = Pix2StructConfig.from_pretrained(str(DEFAULT_WEIGHTS_DIR), local_files_only=True)
    text, vision = config.text_config, config.vision_config
    text.hidden_size, text.d_kv, text.num_heads, text.d_ff, text.num_layers = 32, 8, 4, 64, 3
    vision.hidden_size, vision.d_kv, vision.num_attention_heads, vision.d_ff = 32, 8, 4, 64
    vision.num_hidden_layers = 2
    config.text_config, config.vision_config = text, vision
    torch.manual_seed(seed)
    model = Pix2StructForConditionalGeneration(config)
    return Pix2StructWidgetCaptioningPipeline._from_model(model, processor, "cpu", "tiny-random")


def _real_pipeline(device: str = "cpu") -> Pix2StructWidgetCaptioningPipeline:
    return Pix2StructWidgetCaptioningPipeline.from_pretrained(device=device)


def _real_staged() -> bool:
    return (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file()


def _screen(path, colour, label):
    image = Image.new("RGB", (180, 320), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, 180, 40], fill=(30, 30, 30))
    draw.rectangle([20, 100, 160, 140], fill=colour)
    draw.text((30, 110), label, fill="white")
    image.save(path, format="PNG")
    return path


@pytest.fixture(scope="module")
def records(tmp_path_factory):
    root = tmp_path_factory.mktemp("screens")
    out = []
    for i in range(12):
        colour, label = COLOURS[i % 4], ["Search", "Back", "Menu", "Share"][i % 4]
        path = _screen(root / f"screen{i}.png", colour, label)
        out.append(
            {
                "id": f"w{i:02d}",
                "image_id": f"screen{i}",
                "image": str(path),
                "box": [20, 100, 160, 140],
                "captions": [f"{label.lower()} button", f"tap to {label.lower()}"],
                "group": f"app{i}",
                "category": "large-widget",
            }
        )
    return out


def _captions(pipe, records):
    out = []
    for r in records:
        with Image.open(r["image"]) as image:
            image.load()
            out.append(pipe.caption(image, r["box"], max_new_tokens=8)["caption"])
    return out


@pytest.fixture(params=["tiny", "real"])
def kind(request):
    if request.param == "real" and not _real_staged():
        pytest.skip("pinned snapshot not staged")
    return request.param


@pytest.fixture
def pipe(kind):
    return _tiny_pipeline() if kind == "tiny" else _real_pipeline()


def _fresh(kind):
    return _tiny_pipeline() if kind == "tiny" else _real_pipeline()


def _load(kind, artifact):
    fresh = _fresh(kind)
    fresh.load_artifact(artifact)
    return fresh


def test_evaluate_scores_reference_captions_of_widgets(pipe, records):
    metrics = pipe.evaluate(records[:5], max_new_tokens=8)
    assert metrics["n"] == 5 and 0.0 <= metrics["bleu4"] <= 1.0 and metrics["adapted"] is False
    assert set(metrics) >= {"rouge_l", "cider_d", "unigram_f1", "empty_rate", "mean_words", "seconds"}


def test_one_epoch_adaptation_scope_and_artifact_round_trip(kind, pipe, records, tmp_path):
    result = pipe.adapt(records[:8], records[8:], epochs=1, trainable_decoder_layers=1, batch_size=4)
    last = pipe._model.config.text_config.num_layers - 1
    assert all(
        name.startswith(f"decoder.layer.{last}.") or name.startswith("decoder.final_layer_norm.")
        for name in result["trainable_names"]
    )
    assert "decoder.lm_head.weight" not in result["trainable_names"]
    assert not any(name.startswith("encoder.") for name in result["trainable_names"])
    if kind == "real":
        assert result["n_trainable"] == REAL_ONE_BLOCK_TRAINABLE
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    expected_val = {"bleu4", "rouge_l", "cider_d", "unigram_f1", "mean_words", "n"}
    assert set(result["history"][1]["val"]) == expected_val
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"])
    assert manifest["base_model"]["weight_file"] == WEIGHT_FILE
    reloaded = _load(kind, artifact)
    assert _captions(pipe, records[:4]) == _captions(reloaded, records[:4])
    assert reloaded.adapter["best_epoch"] == result["best_epoch"]
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_training_moves_the_trainable_tensors_only(kind, pipe, records):
    if kind == "real":
        pytest.skip("the tiny model is enough to check what moves")
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
    # A batch is widgets (each carrying all its references), so batch_size=2 gives four steps an epoch.
    result = pipe.adapt(records[:8], None, epochs=3, lr=1e-3, trainable_decoder_layers=1, batch_size=2)
    after = pipe._model.state_dict()
    moved = {k for k in before if not torch.equal(before[k], after[k])}
    assert moved and moved <= set(result["trainable_names"])
    assert result["history"][3]["train_loss"] < result["history"][1]["train_loss"]


def test_no_validation_keeps_the_final_epoch_and_reloads_it(kind, pipe, records, tmp_path):
    result = pipe.adapt(records[:8], None, epochs=2, trainable_decoder_layers=1, batch_size=4)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = _load(kind, artifact)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["best_epoch"] == 2 and reloaded.adapter["trainable_decoder_layers"] == 1


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(
    kind, pipe, records, tmp_path
):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:8], None, epochs=1, trainable_decoder_layers=1, batch_size=4)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        _load(kind, fewer)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["zz.extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [
        {**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}
    ]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        _load(kind, extra)
    other_layers = tmp_path / "other_layers"
    shutil.copytree(artifact, other_layers)
    adapter = {**manifest["adapter"], "trainable_decoder_layers": 2}
    (other_layers / "manifest.json").write_text(json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        _load(kind, other_layers)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:8], None, epochs=2, trainable_decoder_layers=1, batch_size=4, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_caption_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    """Every tensor the runner, the encoder cache and the trainer build must land on the model's device."""
    if not _real_staged():
        pytest.skip("pinned snapshot not staged")
    cuda = _real_pipeline(device="cuda:0")
    assert cuda.device == "cuda:0"
    with Image.open(records[0]["image"]) as image:
        image.load()
        first = cuda.caption(image, records[0]["box"])
    assert first["device"] == "cuda:0" and isinstance(first["caption"], str)
    result = cuda.adapt(records[:8], records[8:], epochs=1, trainable_decoder_layers=1, batch_size=4)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = Pix2StructWidgetCaptioningPipeline.from_artifact(artifact, device="cuda:0")
    assert _captions(cuda, records[:4]) == _captions(reloaded, records[:4])
