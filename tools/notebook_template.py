"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "pix2struct_ui_captioning_pipeline",
    "repo_name": "pix2struct-ui-captioning-pipeline",
    "stem": "pix2struct_ui_captioning",
    "notebook_name": "pix2struct_ui_captioning_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "Pix2StructWidgetCaptioningPipeline",
    "weights_key": "pix2struct-widget-captioning-base",
    "runtime_imports": ["torch", "transformers"],
    "title": "Pix2Struct widget-captioning-base — DIMER UI widget captioning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/pix2struct-ui-captioning-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/pix2struct-ui-captioning-pipeline/blob/main/tutorials/pix2struct_ui_captioning_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-google%2Fpix2struct--widget--captioning--base-ffcc4d?style=flat",
            "https://huggingface.co/google/pix2struct-widget-captioning-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-google--research%2Fpix2struct-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/google-research/pix2struct",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2210.03347-b31b1b.svg", "https://arxiv.org/abs/2210.03347"),
    ],
    "capability": "UI widget captioning — one app screenshot plus one widget bounding box → one short caption of that widget's purpose — using the pinned `google/pix2struct-widget-captioning-base` weights",
    "intro": (
        "At inference the Pix2Struct image-encoder/text-decoder (a ViT-style encoder over variable-resolution 16×16 patches "
        "and a 12-layer text decoder, 282M parameters, pretrained by parsing masked web screenshots into HTML and fine-tuned "
        "on Widget Captioning, a set of Android screenshots from the Rico corpus with human-written descriptions of individual "
        "UI elements) receives the screenshot with the **target widget outlined in blue** — the upstream preprocessing "
        "convention, reproduced by the carried module; no header text is rendered — scales it to fill at most 2048 patches, "
        "and generates a short caption of the widget's role (`search bar`, `go to profile`) token by token. Decoding is greedy "
        "(`do_sample=False`, one beam) under a caller-owned `max_new_tokens` budget. **No adaptation occurs:** no training, "
        "fine-tuning, in-context conditioning, or preprocessing fitting happens in this notebook — the upstream checkpoint "
        "supplies the weights, processor and tokenizer, and the carried module adds snapshot verification, the input contract "
        "(image side ceilings, a box inside the image with sides of at least 4 px, the token budget), the box rendering, a "
        "fixed output contract, and the `annotate_widget`, `keyword_hits`, `unigram_f1`, `validate_inputs` and "
        "`evaluation_report` helpers. The default sample is a flat mock of a messaging app drawn in code with five widget "
        "boxes and no reference captions, so the evaluation report is `not-measurable` by design (the fleet matrix lists this "
        "row as report-only) and the printed keyword checks are observations, not a Widget Captioning benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic app screen with widget boxes (or upload your own screenshot and "
        "type box coordinates) and validate it into an input manifest, choose a token budget, run the supported task, read "
        "the captions correctly (generated text, no score, a `truncated` flag), exercise an optional BYOD path, produce an "
        "evaluation report that is `not-measurable` without reference captions and `sample-sanity` with a bag-of-words "
        "`unigram_f1` when you supply some, and export the captions, the annotated screenshot and provenance."
    ),
    "exclusions": (
        "Widget detection (the box is the caller's input; nothing is located), whole-screen summarisation (a separate "
        "checkpoint), reading UI text back as a transcript, captions in languages other than English, batch throughput, "
        "sampling or beam search (greedy decoding for reproducibility), evaluation on the Widget Captioning benchmark (not "
        "bundled; CIDEr and BLEU-4 need several references per widget and are not computed here), and any training. The "
        "model was fine-tuned on 2017-era Android screenshots at phone resolution; flat mocks, desktop or web UIs, dark "
        "themes and non-English interfaces are outside what this notebook measures, and a fluent wrong caption carries no "
        "signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 5.4 s to load and 2.0–2.4 s per widget on the 540×960 mock in the Windows venv (Intel Core Ultra 9 275HX) — the 2048-patch encoder pass dominates. The pinned `torch==2.14.0` install and the 1.13 GB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; why a bounding box drawn on the input is part of the model's contract; what reference-based caption metrics (CIDEr, BLEU) need; that a confident caption is not a correct one.",
        "- **Data:** the default sample is a deterministic 540×960 mock of a messaging app drawn in code with Pillow's bundled font (a blue header bar with a gear icon, a search field, three conversation rows with avatars, a green `New message` button, a four-tab bottom bar) with five widget boxes and **no reference captions**, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one screenshot decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px, plus widget boxes typed as `x0, y0, x1, y1` lines in pixels. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic app screen or optional BYOD\n\n"
                "The default sample is **synthetic**: a flat mock of a messaging app — a blue header reading `Messages` with a "
                "gear icon, a `Search conversations` field, three conversation rows with round avatars, a green `New message` "
                "button and a `Home / Chats / Calls / Profile` tab bar — is drawn with Pillow at 540×960 (a phone aspect "
                "ratio), the same mock the repository's smoke run used. Five widget boxes are authored in pixel coordinates "
                "(the button, the search field, the gear icon, Ana's avatar, the Profile tab), each with a few **expected "
                "keywords** the notebook checks as an observation. **No reference captions are authored**, because a caption "
                "you write yourself is not an annotation standard: the evaluation report will therefore be `not-measurable`. "
                "The image digest is printed for the record; it depends on the Pillow build's bundled font rendering. BYOD is "
                "optional and disabled by default; when enabled, upload one screenshot and type your boxes.\n\n"
                "The token budget is a **caller-owned request parameter**: `max_new_tokens` bounds the caption "
                "(`DEFAULT_MAX_NEW_TOKENS = 20` fits any widget phrase; `MAX_NEW_TOKENS = 64` is the ceiling). Nothing is "
                "validated in this cell — the next section hands the image and the boxes to the pipeline's own validation "
                "stage, which is the only checker. Look for a dictionary naming the sample kind, the image size and digest, the "
                "budget and the number of boxes."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "byod_boxes = '90, 820, 450, 880'  # @param {{type:\"string\"}}\n"
                "max_new_tokens = 20  # @param {{type:\"integer\"}}\n\n\n"
                "def synthetic_screen(width=540, height=960):\n"
                "    \"\"\"A flat messaging-app mock drawn with Pillow; returns image + [(widget name, box, expected keywords)].\"\"\"\n"
                "    image = Image.new('RGB', (width, height), (245, 246, 250))\n"
                "    d = ImageDraw.Draw(image)\n"
                "    title, body = ImageFont.load_default(size=26), ImageFont.load_default(size=20)\n"
                "    d.rectangle([0, 0, 540, 90], fill=(33, 90, 200))  # header bar\n"
                "    d.text((30, 30), 'Messages', fill='white', font=title)\n"
                "    d.rectangle([470, 25, 510, 65], outline='white', width=3)  # gear-like icon\n"
                "    d.ellipse([482, 37, 498, 53], fill='white')\n"
                "    d.rounded_rectangle([30, 120, 510, 170], radius=12, fill='white', outline=(200, 200, 200))  # search field\n"
                "    d.text((50, 133), 'Search conversations', fill=(150, 150, 150), font=body)\n"
                "    for index, (name, message) in enumerate([('Ana', 'See you at 6?'), ('Ben', 'Sent the files'), ('Cara', 'Happy birthday!')]):\n"
                "        y = 200 + index * 90\n"
                "        d.ellipse([30, y, 90, y + 60], fill=(120, 160, 220))  # avatar\n"
                "        d.text((110, y + 5), name, fill='black', font=title)\n"
                "        d.text((110, y + 38), message, fill=(110, 110, 110), font=body)\n"
                "    d.rounded_rectangle([90, 820, 450, 880], radius=30, fill=(33, 150, 90))  # primary button\n"
                "    d.text((270, 850), 'New message', fill='white', font=title, anchor='mm')\n"
                "    d.rectangle([0, 900, 540, 960], fill='white')  # tab bar\n"
                "    for index, label in enumerate(['Home', 'Chats', 'Calls', 'Profile']):\n"
                "        d.text((67 + index * 135, 930), label, fill=(80, 80, 80), font=body, anchor='mm')\n"
                "    widgets = [\n"
                "        ('new-message button', [90, 820, 450, 880], ['message']),\n"
                "        ('search field', [30, 120, 510, 170], ['search']),\n"
                "        ('gear icon', [470, 25, 510, 65], ['settings']),  # smoke run: `go to next` (recorded miss)\n"
                "        ('Ana avatar', [30, 200, 90, 260], ['ana']),\n"
                "        ('Profile tab', [420, 905, 540, 955], ['profile']),\n"
                "    ]\n"
                "    return image, widgets\n\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    widgets = []\n"
                "    for index, line in enumerate(byod_boxes.splitlines()):\n"
                "        parts = [part.strip() for part in line.split(',') if part.strip()]\n"
                "        if len(parts) == 4:\n"
                "            widgets.append((f'widget-{{index}}', [float(part) for part in parts], []))\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic mock: no randomness, so no seed is needed; the digest depends on the Pillow build's bundled font.\n"
                "    image, widgets = synthetic_screen()\n"
                "    image_name = 'synthetic_messaging_screen_540x960.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "widget_names = [name for name, _, _ in widgets]\n"
                "boxes = [box for _, box, _ in widgets]\n"
                "expected_keywords = [keywords for _, _, keywords in widgets]\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'max_new_tokens': max_new_tokens, 'n_boxes': len(boxes)}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `caption` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, each box four numbers inside the image with sides "
                "of at least `MIN_BOX_SIDE` px, and `max_new_tokens` in `[1, MAX_NEW_TOKENS]` — and returns an **input "
                "manifest** naming the schema (including the blue-outline rendering, the patch budget and the decoding rule), "
                "the input's observed mode and size, the checked boxes, the budget and the verdict. The manifest is written to "
                "`outputs/{stem}_input_manifest.json`. To show what rejection looks like, the cell also validates a box that "
                "runs past the screenshot's edge and records the pipeline's own error message as a finding. Inside the pipeline "
                "the image is converted to RGB, the box is drawn on a copy, and the composite is scaled to the patch budget; "
                "nothing else is dropped or altered. The pipeline cannot tell whether the box encloses a widget or whether the "
                "image is a screenshot at all: that contract is the caller's."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PATCHES': MAX_PATCHES, 'MIN_BOX_SIDE': MIN_BOX_SIDE, 'BOX_COLOR': BOX_COLOR, 'BOX_WIDTH': BOX_WIDTH, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING}}}})\n"
                "input_manifest = validate_inputs(image, boxes, max_new_tokens=max_new_tokens, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, [[0, 0, image.width + 20, 100]])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'box-outside-image-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Caption the widgets and read the output correctly\n\n"
                "`caption` returns, per widget, a dict with `caption` (the decoded text, stripped), the checked `box`, "
                "`image_size`, `new_tokens`, a `truncated` flag that is true when the budget was exhausted, the generation "
                "settings and the model identity. **No score exists**: the caption is generated text with no probability and no "
                "correctness signal, and a fluent caption is not evidence that it describes the boxed widget. Greedy decoding is "
                "deterministic on a fixed device and dtype; CUDA kernel selection can change a token and therefore the rest of "
                "the caption, so GPU and CPU outputs need not match. Each call draws the box and re-encodes the screenshot at up "
                "to 2048 patches, so cost is per widget (about 2.0–2.4 s each on the reference CPU). As recorded in the model "
                "card, the repository's CPU smoke on this same mock captioned the five widgets `go to new message`, `search "
                "bar`, `go to next`, `select ana` and `profile` — the gear icon is the recorded miss — and captioned a box on a "
                "blank white image `select the image` and on noise `go to next`: the model always produces a caption, whether or "
                "not the box encloses anything. The cell also records which expected keywords appear in each caption; that is "
                "an observation, not a metric."
            ),
            "code": (
                "import time\n\n"
                "results, seconds = [], []\n"
                "for name, box in zip(widget_names, boxes):\n"
                "    t0 = time.time()\n"
                "    result = pipe.caption(image, box, max_new_tokens=max_new_tokens)\n"
                "    result['widget'] = name\n"
                "    results.append(result)\n"
                "    seconds.append(round(time.time() - t0, 2))\n"
                "print({{'device': pipe.device, 'dtype': pipe.dtype, 'seconds_per_widget': seconds, 'any_truncated': any(r['truncated'] for r in results)}})\n"
                "keyword_observations = []\n"
                "for result, keywords in zip(results, expected_keywords):\n"
                "    hits = keyword_hits(result['caption'], keywords) if keywords else {{}}\n"
                "    keyword_observations.append({{'widget': result['widget'], 'box': result['box'], 'expected_keywords': keywords, 'hits': hits}})\n"
                "    print(f\"{{result['widget']}} {{result['box']}}\\n   caption: {{result['caption']!r}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})\\n   keywords: {{hits}}\")\n"
                "if any(r['truncated'] for r in results):\n"
                "    print('A budget was exhausted: that caption is incomplete. Raise max_new_tokens (ceiling MAX_NEW_TOKENS) and rerun.')"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No quality is "
                "reported by default: widget-caption metrics (CIDEr, BLEU-4) need several human-written reference captions per "
                "widget from the deployment's own screens and corpus-level statistics, and this repository ships none (the "
                "Widget Captioning dataset is not bundled). The repository's helper `unigram_f1` — bag-of-words F1 after "
                "normalisation (lower-case, punctuation removed, whitespace collapsed) against the best-matching reference — "
                "exists so that a caller who does supply references gets a `sample-sanity` report with one entry per widget; it "
                "is explicitly **not** a captioning metric. On the default path no references are supplied, the verdict is "
                "`not-measurable`, and the report states what would make the task measurable; the keyword observations from "
                "the previous section are attached to the report file under `observations` for the record. The report is "
                "written to `outputs/{stem}_evaluation_report.json`. To see the other branch, set `references` below to one "
                "list of reference captions per widget."
            ),
            "code": (
                "references = None  # e.g. [['compose a new message'], ['search conversations'], ['open settings'], ['open Ana chat'], ['go to profile']]\n"
                "report = evaluation_report(results, references, sample_kind=sample_kind)\n"
                "report['observations'] = keyword_observations\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k not in ('metrics', 'per_widget', 'observations')}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:12}} {{metric['value']:.3f}}  ({{metric['estimation']}})\")\n"
                "for entry in report.get('per_widget', []):\n"
                "    print(f\"  unigram_f1 {{entry['unigram_f1']:.2f}}  {{entry['box']}} -> {{entry['prediction']!r}} (references: {{entry['references']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference captions exist for these widgets, so nothing is scored; read the captions against the screen yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves every result (widget name, box, caption, `new_tokens`, `truncated`, the budget), "
                "the evaluation report with the keyword observations, the input manifest, the sample identity and digest, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the "
                "immutable model revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, "
                "device). The captions are also written as CSV with explicit `image`, `widget`, `box`, `caption`, "
                "`new_tokens`, `truncated` columns, and an annotated PNG shows the screenshot with every widget box outlined "
                "and its caption printed in a panel beside it for visual inspection — a supplement to, not a replacement for, "
                "the machine-readable files. No credentials are recorded."
            ),
            "code": (
                "import csv\n\n"
                "panel_width = 420\n"
                "annotated = Image.new('RGB', (image.width + panel_width, image.height), 'white')\n"
                "annotated.paste(image.convert('RGB'), (0, 0))\n"
                "draw = ImageDraw.Draw(annotated)\n"
                "panel_font = ImageFont.load_default(size=15)\n"
                "for index, result in enumerate(results):\n"
                "    x0, y0, x1, y1 = result['box']\n"
                "    draw.rectangle([x0, y0, x1, y1], outline=BOX_COLOR, width=BOX_WIDTH)\n"
                "    draw.text((x0 + 4, max(y0 - 18, 0)), str(index + 1), fill=BOX_COLOR, font=panel_font)\n"
                "    draw.text((image.width + 16, 20 + 26 * index), f\"{{index + 1}}. {{result['widget']}}: {{result['caption']}}\", fill=(40, 90, 220), font=panel_font)\n"
                "annotated.save('outputs/{stem}_annotated.png')\n"
                "payload = {{\n"
                "    'predictions': results,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'widgets': widget_names, 'boxes': boxes, 'expected_keywords': expected_keywords}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "with open('outputs/{stem}_captions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'widget', 'box', 'caption', 'new_tokens', 'truncated'])\n"
                "    for result in results:\n"
                "        writer.writerow([image_name, result['widget'], ' '.join(str(v) for v in result['box']), result['caption'], result['new_tokens'], result['truncated']])\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The captions are the text the model generates for a screenshot with one widget outlined in blue; nothing in the "
        "output scores that text, the model returns no location beyond the box you gave it, and it captions every box — "
        "including one on a blank image — with equal fluency. On the drawn mock the evaluation report is `not-measurable` by "
        "design: no reference captions exist, and the keyword observations (the repository's smoke run found `message`, "
        "`search`, `ana` and `profile` and called the gear icon `go to next`) are what you can check by eye, not a metric; they "
        "say nothing about real Android screenshots, desktop or web interfaces, dark themes, icons without text, or "
        "non-English UIs, and a BYOD result is a per-widget observation with the same verdict. **The model captions any "
        "box on any image** and stops only at end-of-sequence or the token budget: check `truncated`, and treat a plausible "
        "caption for a box that encloses nothing as the expected failure mode, not an exception. The box is part of the "
        "request — a box around the whole screen produced `select the message` in the smoke run — so a wrong box is a wrong "
        "input, not a model error. The pipeline provides no widget detection, no screen summarisation, no benchmark evaluation "
        "and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** move a box a few pixels off its widget and watch the caption change; box Ben's or Cara's row "
        "instead of Ana's avatar; lower `max_new_tokens` to 1 and watch `truncated` turn true on `go`; write one reference "
        "caption per widget into `references` and see the verdict switch to `sample-sanity` with a `unigram_f1` you should not "
        "mistake for CIDEr; enable `USE_BYOD` with a screenshot you know and type its widget boxes.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/pix2struct-ui-captioning-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/pix2struct-ui-captioning-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/pix2struct-ui-captioning-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code (widget-box preprocessing): https://github.com/google-research/pix2struct/blob/main/pix2struct/preprocessing/convert_widget_captioning.py\n"
        "- Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding (Lee et al., 2022): https://arxiv.org/abs/2210.03347\n"
        "- Widget Captioning: Generating Natural Language Description for Mobile User Interface Elements (Li et al., 2020): https://arxiv.org/abs/2010.04295\n"
        "- Rico: A Mobile App Dataset for Building Data-Driven Design Applications (Deka et al., 2017): https://doi.org/10.1145/3126594.3126651"
    ),
}
