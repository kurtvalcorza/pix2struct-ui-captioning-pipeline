"""Widget captioning with the pinned ``google/pix2struct-widget-captioning-base`` checkpoint.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the Pix2Struct architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed. The target widget is indicated
the way the upstream preprocessing did — a blue outline drawn on the screenshot, no header text — and
the model generates a short caption for it.

The adaptation contract (`evaluate`, `adapt`, `save_artifact`, `load_artifact`, `from_artifact`) fine-tunes
the caption decoder's last blocks on validated widget records with reference captions, selects the epoch on
validation CIDEr-D and exports the trained tensors as a safetensors adapter bound to the pinned base.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

MODEL_ID = "google/pix2struct-widget-captioning-base"
MODEL_REVISION = "7e99642f87127dd3aee97ea82616bbfda5e610bb"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "pix2struct-widget-captioning-base"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
WEIGHT_FILE = "model.safetensors"
WEIGHT_SHA256 = "0b096c351692854237aa9fca73e973e3f764f376afc84999357614bac36e2b92"
PARAMETER_COUNT = 282_285_696  # 18,879,744 of them train by default (2 decoder blocks + final norm)
DECODER_LAYERS = 12
DEFAULT_TRAINABLE_DECODER_LAYERS = 2
# Evaluation bounds: a split larger than MAX_EVAL_RECORDS is refused (captioning is one encoder pass per
# widget); below MIN_SCORED_RECORDS the verdict says the sample is small.
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 50
ARTIFACT_FORMAT = "org.valcorza.pix2struct-widget-captioning-base.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"

# Generation ceilings. Widget captions are a few words ("search bar", "go to profile"; the checkpoint's
# text_config max_length is 20); the default leaves room for a phrase, the ceiling bounds runaway
# generation.
MAX_NEW_TOKENS = 64
DEFAULT_MAX_NEW_TOKENS = 20
DECODING = "greedy"
# Widget box rendering: the upstream preprocessing (pix2struct/preprocessing/convert_widget_captioning.py)
# draws the target widget's bounds as a blue rectangle with a transparent fill and no header text.
BOX_COLOR = (0, 0, 255)
BOX_WIDTH = 3
MIN_BOX_SIDE = 4
# Input ceilings. The processor extracts at most MAX_PATCHES 16x16 patches (preprocessor_config.json)
# after scaling the image to fill that budget (aspect ratio preserved), so pixel count only guards
# memory during resizing.
MAX_PATCHES = 2048
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
_PUNCT_RE = re.compile(r"[^\w\s]")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def normalize_caption(text: str) -> str:
    """COCO-caption-style normalisation: lower-case, punctuation removed, whitespace collapsed."""
    return " ".join(_PUNCT_RE.sub(" ", text.lower()).split())


def caption_tokens(text: str) -> list[str]:
    return normalize_caption(text).split()


def unigram_f1(prediction: str, references: Sequence[str]) -> float:
    """Bag-of-words F1 between the normalised prediction and the best-matching reference.

    A plumbing check, not a captioning metric: CIDEr, BLEU-4 and SPICE need several references per
    image and corpus-level statistics. Multiset overlap counts repeated words once per occurrence.
    """
    if not references:
        raise ValueError("references must contain at least one caption")
    pred = caption_tokens(prediction)
    best = 0.0
    for reference in references:
        ref = caption_tokens(reference)
        if not pred or not ref:
            continue
        ref_counts: dict[str, int] = {}
        for token in ref:
            ref_counts[token] = ref_counts.get(token, 0) + 1
        overlap = 0
        for token in pred:
            if ref_counts.get(token, 0) > 0:
                overlap += 1
                ref_counts[token] -= 1
        if overlap:
            precision, recall = overlap / len(pred), overlap / len(ref)
            best = max(best, 2 * precision * recall / (precision + recall))
    return best


def keyword_hits(caption: str, keywords: Sequence[str]) -> dict[str, bool]:
    """Which of the caller's keywords (normalised, whole-token match) appear in the caption."""
    tokens = set(caption_tokens(caption))
    return {keyword: all(part in tokens for part in caption_tokens(keyword)) for keyword in keywords}


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def validate_box(box: Any, image_size: tuple[int, int]) -> list[int]:
    """Check one widget box ``[x0, y0, x1, y1]`` in pixels: inside the image, at least MIN_BOX_SIDE a side."""
    if isinstance(box, (str, bytes)) or not isinstance(box, Sequence) or len(box) != 4:
        raise TypeError("box must be a sequence of four numbers [x0, y0, x1, y1]")
    values = []
    for value in box:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("box coordinates must be numbers")
        values.append(int(round(value)))
    x0, y0, x1, y1 = values
    width, height = image_size
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
        raise ValueError(f"box {values} lies outside the {width}x{height} image")
    if x1 - x0 < MIN_BOX_SIDE or y1 - y0 < MIN_BOX_SIDE:
        raise ValueError(f"box {values} is smaller than MIN_BOX_SIDE {MIN_BOX_SIDE} px on a side")
    return values


def annotate_widget(image: Image.Image, box: Sequence[int]) -> Image.Image:
    """Return a copy of the RGB image with the widget outlined the way the upstream preprocessing did.

    Upstream (``pix2struct/preprocessing/convert_widget_captioning.py``) draws the target widget's
    bounds as a blue rectangle with a transparent fill and no header text; this package draws the same
    blue outline at BOX_WIDTH pixels.
    """
    rgb = validate_image(image)
    x0, y0, x1, y1 = validate_box(box, rgb.size)
    annotated = rgb.copy()
    ImageDraw.Draw(annotated).rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=BOX_WIDTH)
    return annotated


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one screenshot as PIL.Image.Image (any mode, converted to RGB) plus one widget box "
        "[x0, y0, x1, y1] in pixels"
    ),
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "box": f"pixel coordinates inside the image, each side at least {MIN_BOX_SIDE} px",
    "max_new_tokens": [1, MAX_NEW_TOKENS],
    "decoding": f"{DECODING} (do_sample=False, num_beams=1), deterministic on a fixed device and dtype",
    "preprocessing": (
        f"the widget box is drawn on a copy of the screenshot as a blue outline ({BOX_WIDTH} px, the "
        "upstream widget-captioning convention; no header text is rendered); the annotated screenshot is "
        "scaled to fill at most MAX_PATCHES 16x16 patches (aspect ratio preserved), normalised per image "
        "and flattened into patch tokens with row/column positions; the decoder generates the caption"
    ),
    "output": "one short caption string for the boxed widget (the model's decoded text), no score",
}


def _check_inputs(image: Any, box: Any, max_new_tokens: Any) -> tuple[Image.Image, list[int], int]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``caption`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    checked_box = validate_box(box, rgb.size)
    if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
        raise TypeError("max_new_tokens must be an int")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS:
        raise ValueError(f"max_new_tokens must be between 1 and MAX_NEW_TOKENS={MAX_NEW_TOKENS}")
    return rgb, checked_box, max_new_tokens


def validate_inputs(
    image: Image.Image,
    boxes: Sequence[Sequence[int]],
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Every box is checked exactly as ``caption`` would check it; rejection is reported by raising, and
    a caller that wants the finding recorded catches the exception and stores ``str(exc)`` under
    ``findings``.
    """
    if isinstance(boxes, (str, bytes)) or not isinstance(boxes, Sequence) or not boxes:
        raise TypeError("boxes must be a non-empty sequence of [x0, y0, x1, y1] boxes")
    if isinstance(boxes[0], (int, float)):
        raise TypeError("boxes must be a sequence of boxes, not a single box")
    checked = [_check_inputs(image, box, max_new_tokens)[1] for box in boxes]
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (caption takes one screenshot)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [{"id": names[0] if names else "image-0", "mode": image.mode, "size": list(image.size)}],
        "boxes": checked,
        "generation": {"max_new_tokens": int(max_new_tokens), "do_sample": False, "decoding": DECODING},
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    results: Sequence[Mapping[str, Any]],
    references: Sequence[Sequence[str]] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``references`` (one sequence of reference captions per result, in order) the report carries
    the mean ``unigram_f1`` over the widgets plus one per-widget entry, verdict ``sample-sanity``;
    without references it is ``not-measurable`` and says what labelled data would make the task
    measurable. Neither is a captioning benchmark.
    """
    if not results:
        raise ValueError("results must contain at least one caption result")
    base = {
        "task": "screenshot + widget box -> short caption of the widget (widget captioning)",
        "score_semantics": (
            "the caption is generated text and carries no score, probability or correctness signal; a "
            "fluent caption is not evidence that it describes the boxed widget. Greedy decoding makes the "
            "output reproducible on a fixed device and dtype, a reproducibility property, not a quality one"
        ),
        "sample_kind": sample_kind,
        "n_widgets": len(results),
        "truncated": [bool(result.get("truncated")) for result in results],
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if references is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no reference captions were supplied for the captioned widgets",
            "needs": (
                "several human-written reference captions per widget from the deployment's own screens "
                "(Widget Captioning-style annotations) scored with CIDEr / BLEU-4 over a corpus; no such "
                "labelled set ships with this repository"
            ),
        }
    if len(references) != len(results):
        raise ValueError(f"references has {len(references)} entries for {len(results)} results")
    per_widget = []
    for result, refs in zip(results, references, strict=True):
        if isinstance(refs, str) or not refs:
            raise ValueError("each references entry must be a non-empty sequence of captions")
        prediction = str(result["caption"])
        per_widget.append(
            {
                "box": result.get("box"),
                "prediction": prediction,
                "references": list(refs),
                "unigram_f1": unigram_f1(prediction, refs),
            }
        )
    metrics = [
        {
            "id": "unigram_f1",
            "value": sum(entry["unigram_f1"] for entry in per_widget) / len(per_widget),
            "normalisation": "lower-cased, punctuation removed, whitespace collapsed; best reference",
            "relation_to_benchmarks": (
                "bag-of-words overlap with the best reference; not CIDEr, BLEU-4 or SPICE, which need "
                "several references per widget and corpus-level statistics"
            ),
            "estimation": f"{len(per_widget)} widget(s) on one screenshot, no dispersion estimate",
        }
    ]
    return {
        **base,
        "metrics": metrics,
        "per_widget": per_widget,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(per_widget)} widget(s) with caller-written reference captions; plumbing evidence, not a "
            "captioning benchmark"
        ),
        "needs": (
            "several human-written reference captions per widget from the deployment's own screens scored "
            "with CIDEr / BLEU-4 over a corpus for any quality claim; the Widget Captioning dataset is not "
            "bundled"
        ),
    }


@dataclass
class Pix2StructWidgetCaptioningPipeline:
    """``_runner(annotated_image, max_new_tokens)`` returns ``{"caption": str, "new_tokens": int}``;
    injectable so the offline tests run without the model."""

    _runner: Callable[..., dict[str, Any]]
    device: str = "cpu"
    dtype: str = "float32"
    source: str = "injected"
    adapter: dict[str, Any] | None = field(default=None, repr=False)
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Pix2StructWidgetCaptioningPipeline:
        root = Path(weights_dir or DEFAULT_WEIGHTS_DIR)
        common: dict[str, Any] = {"trust_remote_code": False}
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            location, common["local_files_only"], source = str(root), True, "local-snapshot"
        elif allow_download:
            location, common["revision"], source = MODEL_ID, MODEL_REVISION, "hf-hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage it with: hf download {MODEL_ID} --revision {MODEL_REVISION} --local-dir {root}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import Pix2StructForConditionalGeneration, Pix2StructProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = Pix2StructProcessor.from_pretrained(location, **common)
        model = Pix2StructForConditionalGeneration.from_pretrained(location, dtype=torch.float32, **common)
        return cls._from_model(model, processor, resolved_device, source)

    @classmethod
    def _from_model(
        cls, model: Any, processor: Any, device: str, source: str
    ) -> Pix2StructWidgetCaptioningPipeline:
        """Wrap a constructed model and processor (every parameter frozen, eval mode, header rendering off) in
        a pipeline; the offline tests use it with a small randomly initialised Pix2Struct model."""
        import torch

        # The snapshot's preprocessor_config declares is_vqa=True, which would render a text header
        # above the screenshot and fetch a font from the Hub. Upstream's widget-captioning preprocessing
        # renders no header - only the blue widget box - so the header path is disabled here; nothing
        # but the box is added to the image and no font is involved.
        processor.image_processor.is_vqa = False
        model = model.eval().to(device)
        for param in model.parameters():
            param.requires_grad_(False)

        def runner(annotated: Image.Image, max_new_tokens: int) -> dict[str, Any]:
            inputs = processor.image_processor(annotated, return_tensors="pt").to(device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False, num_beams=1
                )
            # Encoder-decoder: the output holds only decoder tokens (decoder_start + caption + eos).
            ids = generated[0]
            decoded = processor.tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
            return {"caption": decoded, "new_tokens": max(int(ids.shape[0]) - 1, 0)}

        return cls(runner, device, "float32", source, _model=model, _processor=processor)

    def caption(
        self,
        image: Image.Image,
        box: Sequence[int],
        *,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> dict[str, Any]:
        """Caption the widget inside ``box`` on one screenshot; ``caption`` is the decoded text, stripped."""
        rgb, checked_box, checked_tokens = _check_inputs(image, box, max_new_tokens)
        annotated = annotate_widget(rgb, checked_box)
        raw = self._runner(annotated, checked_tokens)
        if not isinstance(raw, dict) or "caption" not in raw:
            raise RuntimeError("runner must return a dict with 'caption'")
        new_tokens = int(raw.get("new_tokens", 0))
        return {
            "caption": str(raw["caption"]).strip(),
            "box": checked_box,
            "image_size": list(rgb.size),
            "new_tokens": new_tokens,
            "truncated": new_tokens >= checked_tokens,
            "generation": {"max_new_tokens": checked_tokens, "do_sample": False, "decoding": DECODING},
            "device": self.device,
            "dtype": self.dtype,
            "source": self.source,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    # ---- adaptation contract ---------------------------------------------------------------------------

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._processor

    def predict(
        self, records: Sequence[Mapping[str, Any]], *, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> list[str]:
        """Caption every validated record's widget, in order."""
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        out = []
        for record in checked:
            with Image.open(record["image"]) as image:
                image.load()
                out.append(self.caption(image, record["box"], max_new_tokens=max_new_tokens)["caption"])
        return out

    def evaluate(
        self, records: Sequence[Mapping[str, Any]], *, max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    ) -> dict[str, Any]:
        """Caption every record's widget (greedy) and score the predictions against its reference captions
        (BLEU-4, ROUGE-L, CIDEr-D, unigram F1)."""
        from .metrics import caption_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        predictions = self.predict(checked, max_new_tokens=max_new_tokens)
        metrics = caption_metrics(predictions, [[str(c) for c in r["captions"]] for r in checked])
        metrics.update(
            {
                "max_new_tokens": max_new_tokens,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _decoder_layers(self) -> int:
        model, _ = self._require_model()
        return int(model.config.text_config.num_layers)

    def _trainable_names(self, trainable_decoder_layers: int) -> list[str]:
        """The last `trainable_decoder_layers` blocks of the caption decoder plus the decoder's final layer
        norm. The untied output projection (`decoder.lm_head`, vocabulary x hidden) and every embedding stay
        frozen, as does the whole image encoder."""
        if (
            isinstance(trainable_decoder_layers, bool)
            or not isinstance(trainable_decoder_layers, int)
            or not 1 <= trainable_decoder_layers <= DECODER_LAYERS
        ):
            raise ValueError(f"trainable_decoder_layers must be an int in 1..{DECODER_LAYERS}")
        model, _ = self._require_model()
        n_layers = self._decoder_layers()
        if trainable_decoder_layers > n_layers:
            raise ValueError(f"trainable_decoder_layers must be an int in 1..{n_layers} for this model")
        first = n_layers - trainable_decoder_layers
        prefixes = tuple(f"decoder.layer.{k}." for k in range(first, n_layers))
        prefixes += ("decoder.final_layer_norm.",)
        return [name for name, _p in model.named_parameters() if name.startswith(prefixes)]

    def _encode_batch(self, records: Sequence[Mapping[str, Any]]) -> tuple[Any, Any]:
        """The frozen encoder's output and patch mask for a batch of boxed screenshots. The box is part of the
        image, so every widget is its own encoder input; it is recomputed per step under `no_grad` instead of
        cached (2,048 x 768 floats per widget)."""
        import torch

        model, processor = self._require_model()
        device = next(model.parameters()).device
        images = []
        for record in records:
            with Image.open(record["image"]) as image:
                images.append(annotate_widget(image.convert("RGB"), record["box"]))
        inputs = processor.image_processor(images, return_tensors="pt").to(device)
        with torch.no_grad():
            hidden = model.encoder(
                flattened_patches=inputs["flattened_patches"], attention_mask=inputs["attention_mask"]
            ).last_hidden_state
        return hidden, inputs["attention_mask"]

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 3,
        lr: float = 1e-5,
        batch_size: int = 4,
        trainable_decoder_layers: int = DEFAULT_TRAINABLE_DECODER_LAYERS,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised fine-tuning on validated widget records with reference captions.

        Only the last `trainable_decoder_layers` blocks of the caption decoder and the decoder's final layer
        norm train (2 blocks by default); the image encoder, every embedding and the untied output projection
        stay frozen. Each step takes `batch_size` widgets, draws their boxes on their screenshots exactly as
        `caption` does, runs the frozen encoder once per widget and trains on **every** (widget, reference
        caption) pair of the batch against that encoder output; the target is the tokenised caption with its
        end-of-sequence token, decoded with teacher forcing and scored with the model's own cross-entropy
        (padding ignored); AdamW at a fixed learning rate with gradient clipping at 1.0, no scheduler.
        Epoch 0 records the frozen model's validation metrics; the epoch with the highest validation CIDEr-D
        is kept."""
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not (0.0 < lr <= 1e-3):
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        names = self._trainable_names(trainable_decoder_layers)
        train_checked = validate_dataset(train)["records"]
        val_checked = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"] if val else []
        )
        import torch

        torch.manual_seed(seed)
        model, processor = self._require_model()
        tokenizer = processor.tokenizer
        pad_id = int(tokenizer.pad_token_id)
        started = time.perf_counter()
        wanted = set(names)
        for name, param in model.named_parameters():
            param.requires_grad_(name in wanted)
        params = [p for p in model.parameters() if p.requires_grad]
        n_trainable = sum(p.numel() for p in params)
        optimiser = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
        device = next(model.parameters()).device
        n_pairs = sum(len(r["captions"]) for r in train_checked)

        def score_val() -> dict[str, Any] | None:
            if not val_checked:
                return None
            model.eval()
            return {
                k: v
                for k, v in self.evaluate(val_checked).items()
                if k in ("bleu4", "rouge_l", "cider_d", "unigram_f1", "mean_words", "n")
            }

        history: list[dict[str, Any]] = []
        entry: dict[str, Any] = {"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}
        history.append(entry)
        if progress:
            progress(entry)
        best_score = entry["val"]["cider_d"] if entry["val"] else -math.inf
        best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
        initial_state = {k: v.clone() for k, v in best_state.items()}
        best_epoch = 0
        generator = torch.Generator().manual_seed(seed)
        try:
            for epoch in range(1, epochs + 1):
                model.train()
                order = torch.randperm(len(train_checked), generator=generator).tolist()
                losses = []
                for start in range(0, len(order), batch_size):
                    chosen = [train_checked[j] for j in order[start : start + batch_size]]
                    hidden, mask = self._encode_batch(chosen)
                    owners = [i for i, r in enumerate(chosen) for _c in r["captions"]]
                    targets = tokenizer(
                        [str(c) for r in chosen for c in r["captions"]],
                        padding=True,
                        truncation=True,
                        max_length=MAX_NEW_TOKENS + 1,
                        return_tensors="pt",
                    ).to(device)
                    labels = targets["input_ids"].masked_fill(targets["input_ids"] == pad_id, -100)
                    index = torch.tensor(owners, device=hidden.device)
                    out = model(
                        encoder_outputs=(hidden.index_select(0, index),),
                        attention_mask=mask.index_select(0, index),
                        labels=labels,
                    )
                    optimiser.zero_grad(set_to_none=True)
                    out.loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimiser.step()
                    losses.append(float(out.loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": score_val()}
                history.append(entry)
                if progress:
                    progress(entry)
                current = entry["val"]["cider_d"] if entry["val"] else math.inf
                if current > best_score or not entry["val"]:
                    best_score = current
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in wanted}
                    best_epoch = epoch
        except BaseException:
            # Transactional: a failure in training, validation or the progress callback leaves the base
            # exactly as it was, with every parameter frozen again.
            restore = dict(model.state_dict())
            restore.update(initial_state)
            model.load_state_dict(restore, strict=True)
            model.eval()
            for param in model.parameters():
                param.requires_grad_(False)
            self.adapter = None
            raise
        merged = dict(model.state_dict())
        merged.update(best_state)
        model.load_state_dict(merged, strict=True)
        model.eval()
        for param in model.parameters():
            param.requires_grad_(False)
        self.adapter = {
            "trainable_decoder_layers": trainable_decoder_layers,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest validation CIDEr-D" if val_checked else "final epoch (no validation split)",
            "lr": lr,
            "batch_size": batch_size,
            "n_train": len(train_checked),
            "n_pairs": n_pairs,
            "n_val": len(val_checked),
            "seed": seed,
            "history": history,
            "seconds": round(time.perf_counter() - started, 2),
        }
        return dict(self.adapter)

    # ---- artifacts ------------------------------------------------------------------------------------

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the adapted decoder-block and final-norm tensors as safetensors plus a base manifest."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter["trainable_names"])
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHT_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def _check_artifact_manifest(self, root: Path, manifest: Mapping[str, Any]) -> Path:
        """Refuse an artifact whose manifest is not exactly the one this pipeline writes: the supported format
        and version, the pinned base (id, revision, weight file, digest), exactly one file entry named
        `adapter.safetensors` that resolves inside the artifact directory, and a recorded
        `trainable_decoder_layers` in range. Nothing is deserialised here. The digest check that follows
        detects corruption or drift of the weights relative to the adjacent manifest; it is not authenticity
        against an actor who can replace both files."""
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(
                f"artifact format_version {manifest.get('format_version')!r} is not the supported "
                f"{ARTIFACT_FORMAT_VERSION!r}"
            )
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        if base.get("weight_file", WEIGHT_FILE) != WEIGHT_FILE:
            raise ValueError("artifact was adapted from a different base weight file")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must list exactly one file")
        entry = files[0]
        if not isinstance(entry, Mapping) or entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError(f"artifact manifest must name exactly {ARTIFACT_WEIGHTS_NAME!r}")
        weights_path = (root / entry["path"]).resolve()
        if weights_path.parent != root.resolve():
            raise ValueError("artifact weight path must resolve inside the artifact directory")
        adapter = manifest.get("adapter")
        layers = adapter.get("trainable_decoder_layers") if isinstance(adapter, Mapping) else None
        if isinstance(layers, bool) or not isinstance(layers, int) or not 1 <= layers <= DECODER_LAYERS:
            raise ValueError("artifact manifest does not record an in-range integer trainable_decoder_layers")
        if not isinstance(manifest.get("tensors"), list):
            raise ValueError("artifact manifest must list its tensors")
        return weights_path

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest, digest and exact tensor set **before** deserialising, then overwrite
        exactly the tensors it carries."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        weights_path = self._check_artifact_manifest(root, manifest)
        entry = manifest["files"][0]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        # The exact tensor set the recorded configuration implies — no subset, no extra, no other layer.
        expected = sorted(self._trainable_names(manifest["adapter"]["trainable_decoder_layers"]))
        if sorted(manifest["tensors"]) != expected:
            raise ValueError("artifact tensor list does not match its recorded configuration")
        model, _ = self._require_model()
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != expected:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith("decoder."):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable caption-decoder tensor of the base"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key} has shape {tuple(value.shape)}, "
                    f"base has {tuple(state[key].shape)}"
                )
        merged = dict(state)
        merged.update({k: v.to(state[k].dtype) for k, v in tensors.items()})
        model.load_state_dict(merged, strict=True)
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": manifest["tensors"],
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> Pix2StructWidgetCaptioningPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
