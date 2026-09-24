"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E widget-captioning workflow: the pinned google/pix2struct-widget-captioning-base
snapshot is digest-verified and loaded, one digest-pinned parquet shard of Widget Captioning test widgets is
downloaded, validated and split by app, a drawn app screen with five boxed widgets is captioned through the
inference contract, the frozen model is scored on the held-out widgets beside two non-neural baselines, a
bounded fine-tuning of the caption decoder's last blocks runs in the kernel, the held-out split is scored again
per category, the adapted model re-captions the drawn screen, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "pix2struct_ui_captioning_pipeline",
    "repo_name": "pix2struct-ui-captioning-pipeline",
    "stem": "pix2struct_ui_captioning",
    "notebook_name": "pix2struct_ui_captioning_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `google/pix2struct-widget-captioning-base` snapshot (a 1.13 GB `model.safetensors`), downloads one "
        "digest-pinned parquet shard of Widget Captioning test widgets from the Hugging Face Hub (95 MB, no credential, "
        "refused on any size or SHA-256 mismatch), cuts a seeded subset of whole apps into training, validation and test "
        "widgets so no app or screen is shared, captions five boxed widgets on a drawn messaging-app screen through the "
        "inference contract with an input manifest and a rejection probe, scores the frozen model on the test widgets with "
        "BLEU-4, ROUGE-L, CIDEr-D and unigram F1 beside the constant-caption and colour-nearest-neighbour baselines, runs a "
        "bounded fine-tuning of the caption decoder's last blocks with validation-CIDEr-D epoch selection, scores the "
        "held-out widgets again per category, re-captions the drawn screen with the adapted model, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify caption parity. The default "
        "path needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration "
        "edit (NOTEBOOK_SPEC 2.0 §5). A CUDA runtime is used automatically when present; the CPU path works but is slow "
        "(every widget draws its own box and is encoded at up to 2,048 patches), and the timings of the first clean run are "
        "recorded in `docs/release-verification.md`."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "holding a `records.jsonl` (or `records.json`) of `{{id, image, box, captions}}` objects — `image` a screenshot file "
        "name inside the zip, `box` the widget's `[x0, y0, x1, y1]` in pixels, `captions` one or more reference captions, "
        "optional `image_id`, `group` (the app, so its screens stay in one split) and `category` — beside the image files. "
        "They pass through the same validation, seeded app-disjoint split, baselines, fine-tuning, held-out evaluation, "
        "artifact export and reload-parity cells as the Widget Captioning sample. The expected schema and the ceilings are "
        "stated in the Prerequisites and in Section 4, and uploaded files stay inside this runtime. BYOD is optional and "
        "never part of the default path."
    ),
    "pipeline_class": "Pix2StructWidgetCaptioningPipeline",
    "weights_key": "pix2struct-widget-captioning-base",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "runtime_imports": ["torch", "transformers"],
    "title": "Pix2Struct widget-captioning-base — DIMER E2E UI widget captioning fine-tuning tutorial (standalone)",
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
    "capability": "UI widget captioning and bounded supervised fine-tuning of the caption decoder's last blocks on a screenshot/widget-box/reference-captions dataset, using the pinned `google/pix2struct-widget-captioning-base` weights",
    "intro": (
        "`google/pix2struct-widget-captioning-base` is the Pix2Struct model of Lee et al. (2023) — a ViT-style image encoder "
        "over variable-resolution 16×16 patches (up to 2,048 per image) and a 12-layer text decoder that cross-attends to "
        "them; 282,285,696 parameters, pretrained by parsing masked web screenshots into simplified HTML and fine-tuned on "
        "Widget Captioning, Android screenshots from the Rico corpus with human-written descriptions of individual UI "
        "elements — published under the **Apache-2.0** licence. It receives the screenshot with the **target widget "
        "outlined in blue** (the upstream preprocessing convention, reproduced by the carried module; no header text is "
        "rendered) and generates a short phrase for the widget's role (`search bar`, `go to profile`) with greedy decoding "
        "under a caller-owned `max_new_tokens` budget. **No score exists**: the caption is generated text with no "
        "probability and no correctness signal, and a fluent caption is **not evidence that it describes the boxed "
        "widget**.\n\n"
        "What this notebook adds to inference is **adaptation with reference captions**. The dataset is real and from the "
        "checkpoint's own task: widgets from the Widget Captioning test split (Li et al., EMNLP 2020; **CC BY 4.0** as "
        "declared by the Hub mirror), whose apps the checkpoint was not trained on — so this is continued adaptation inside "
        "the task, and the honest question is a narrow one: does a bounded adaptation of the caption decoder's last blocks on "
        "a few hundred more widgets move the consensus metric on an app-disjoint test split at all, and on which kind of "
        "widget? Small widgets (mostly icons, under 1% of the screen) and larger ones (buttons, fields, rows) are read "
        "separately (`small-widget` / `large-widget`). The notebook downloads **one pinned parquet shard** (95 MB, SHA-256 "
        "pinned in the carried module) and draws a seeded subset of whole apps from it. Three captioning metrics are "
        "implemented in pure Python in the carried modules (**BLEU-4**, **ROUGE-L**, **CIDEr-D** — own implementations of the "
        "`coco-caption` definitions, with CIDEr-D's document frequencies taken from the evaluated set) beside the plumbing "
        "check `unigram_f1`, and two **non-neural baselines** — the corpus-medoid constant caption and a colour nearest "
        "neighbour over the widget's own crop — show where a system with no model sits. Nothing here is a quality claim "
        "about your screens: it is one seeded split of one shard.\n\n"
        "**Weight-format note:** the pinned revision ships the model as SafeTensors (`model.safetensors`, digest-pinned in the "
        "manifest); the snapshot's processor declares the VQA variant, which the carried module switches off so that nothing "
        "but the box is added to the screenshot and no font is downloaded. Section 3 stages and digest-verifies the snapshot "
        "before the processor or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; download a digest-pinned shard of screenshots with widget boxes and "
        "reference captions, validate it and split it by app without leakage; caption through the public API over a drawn "
        "app screen and read `caption`, `box`, `new_tokens` and `truncated` correctly (generated text, no score); score the "
        "frozen model against several reference captions per widget with BLEU-4, ROUGE-L and CIDEr-D beside two non-neural "
        "baselines and read the `small-widget` / `large-widget` breakdown; run a bounded fine-tuning with explicit "
        "hyperparameters and validation-based epoch selection; evaluate on an app-disjoint test split; re-caption a screen "
        "from a different image family with the adapted model; and export a safetensors adapter that reloads against the "
        "pinned base with verified parity."
    ),
    "exclusions": (
        "Widget detection (the caller supplies the box), screen summarisation or question answering (separate checkpoints), "
        "OCR of the screen, captions in languages other than English, batch throughput, sampling, beam search or repetition "
        "penalties (the notebook decodes greedily for reproducibility), SPICE (needs a scene-graph parser), evaluation on the "
        "Widget Captioning benchmark proper (only a seeded subset of one test shard is scored here), fine-tuning of the image "
        "encoder, the embeddings or the output projection, training on screens that are not the pinned sample or your own "
        "uploads, and any claim that a Rico split stands in for your app's screens. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available; a GPU runtime is recommended for Sections 6–8. Every widget draws its own box on its screenshot and is encoded at up to 2,048 patches, so captioning costs seconds per widget on CPU. The pinned `torch==2.14.0` install and the 1.13 GB checkpoint are the large downloads of the run; the widget shard adds 95 MB.",
        "- **Knowledge:** basic Python and PIL; what an encoder–decoder model's generated tokens are; what BLEU-4, ROUGE-L and CIDEr-D measure (n-gram precision with a brevity penalty, longest-common-subsequence F-measure, TF-IDF-weighted n-gram consensus) and why none is a human judgement; why a confident caption is not a correct one.",
        "- **Data contract:** records are `{{id, image, box, captions}}` — a screenshot decodable by Pillow with sides between `MIN_IMAGE_SIDE` (16) and `MAX_IMAGE_SIDE` (4096) px, a widget box `[x0, y0, x1, y1]` in pixels inside the image with sides of at least `MIN_BOX_SIDE` (4) px, and one or more non-empty reference captions of at most `MAX_CAPTION_CHARS` (200) characters (`MIN_CAPTIONS` = 1); optional `image_id` names the screen, optional `group` names the app (BYOD defaults it to the screen) and optional `category` labels the breakdown. Ids match `[A-Za-z0-9_.:-]{{1,64}}` and are unique; a dataset needs 8..5,000 records; every widget of the same app lands in the same split, so a test app's screens are never trained on; every (widget, reference caption) pair is one training target. BYOD accepts one zip of screenshots plus a `records.jsonl` / `records.json` in that shape.",
        "- **Validation is structural, not semantic:** every screenshot is opened and decoded and every box and caption checked, but nothing checks that a reference caption describes the boxed widget — a mislabelled widget is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — app screenshots can show names, messages and account details. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path downloads one object from the Hub dataset repository `bevaya/RICO-WidgetCaptioning` at the immutable revision `6ec57b56…` (`data/test-00000-of-00002.parquet`, 95,313,640 bytes) and refuses it unless its size and SHA-256 match the pins carried in `samples.py`; only the screen id, caption, box, app-package and screenshot columns are read. The mirror declares CC BY 4.0 (Li et al., 2020, over Rico screens, Deka et al., 2017).",
    ],
    "cells": [
        {
            "md": (
                "## 4. Widget Captioning screens, widgets and split\n\n"
                "`fetch_corpus` returns the pinned shard from the cache under `weights/widget-captioning/` or downloads it at "
                "the pinned dataset revision, and refuses it unless its byte size and SHA-256 equal the pins in the carried "
                "module (it also refuses to run at all while no SHA-256 pin is recorded). `read_corpus` reads the screen id, "
                "captions, relative box, app package and screenshot columns with `pyarrow`. `build_sample_dataset` groups the "
                "widgets by app, shuffles the apps with `SPLIT_SEED` and allocates **whole apps** to the test, validation and "
                "training splits until each reaches its widget target (`SAMPLE_WIDGETS`), converting each box to pixels, "
                "skipping widgets with no caption or a box under `MIN_BOX_SIDE`, and labelling each `small-widget` (under 1% of "
                "the screen) or `large-widget`. `validate_dataset` then opens and decodes every screenshot and checks every "
                "record against the contract, `check_split_disjoint` asserts no app and no screen is shared, and the training "
                "split is written to `outputs/{stem}_train.jsonl` in the shape BYOD expects.\n\n"
                "Look for: the shard's row and screen counts, the widget, screen and app counts per split, the category mix, "
                "captions per widget, three digests, and four refusal probes — a duplicate id, a missing image file, a box "
                "outside the screenshot and a dataset too small to split — each rejected before `torch` does anything."
            ),
            "code": (
                "import collections\n"
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import zipfile\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_dir = Path('work') / 'byod'\n"
                "    byod_dir.mkdir(parents=True, exist_ok=True)\n"
                "    with zipfile.ZipFile(io.BytesIO(payload)) as archive:\n"
                "        for member in archive.infolist():\n"
                "            name = Path(member.filename).name\n"
                "            if member.is_dir() or not name or name.startswith('.'):\n"
                "                continue\n"
                "            (byod_dir / name).write_bytes(archive.read(member))\n"
                "    records_file = next(p for p in (byod_dir / 'records.jsonl', byod_dir / 'records.json') if p.is_file())\n"
                "    records = load_byod_dataset(records_file)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED, base_dir=byod_dir)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    shard_path = fetch_corpus(cache_dir='weights/widget-captioning')\n"
                "    corpus_rows = read_corpus(shard_path)\n"
                "    raw_rows = {{'widgets': len(corpus_rows), 'screens': len({{r['screen_id'] for r in corpus_rows}}), 'apps': len({{r['app'] for r in corpus_rows}})}}\n"
                "    splits = build_sample_dataset(corpus_rows, seed=SPLIT_SEED, image_dir='weights/widget-captioning/images')\n"
                "    data_source = f'{{CORPUS_NAME}} — {{CORPUS_RELEASE}} ({{CORPUS_LICENSE}})'\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "categories = {{name: manifest['categories'] for name, manifest in dataset_manifests.items()}}\n"
                "write_dataset_jsonl(splits['train'], 'outputs/{stem}_train.jsonl')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'shard_sha256': str(CORPUS_FILE['sha256'])[:16] + '...'}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'widgets': manifest['n_records'], 'screens': manifest['unique_images'], 'apps': manifest['unique_groups'], 'categories': manifest['categories'], 'captions_per_widget': manifest['captions_per_widget'], 'caption_words': manifest['caption_words'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = splits['train'][0]\n"
                "print({{'example': {{'id': example['id'], 'image': Path(example['image']).name, 'size': example['image_size'], 'box': example['box'], 'category': example['category'], 'captions': example['captions']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in splits['train'][:8]],\n"
                "    'missing image file': [{{**splits['train'][0], 'image': 'work/does-not-exist.png'}}, *splits['train'][1:8]],\n"
                "    'box outside the screenshot': [{{**splits['train'][0], 'box': [0, 0, 99999, 10]}}, *splits['train'][1:8]],\n"
                "    'too small': splits['train'][:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Caption through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a flat mock of a messaging app "
                "drawn in code at 540×960 — a blue header reading `Messages` with a gear icon, a `Search conversations` field, "
                "three conversation rows with round avatars, a green `New message` button and a `Home / Chats / Calls / Profile` "
                "tab bar — with five widget boxes authored in pixel coordinates and a few **expected keywords** per widget. It is "
                "a different image family from the Rico screenshots, and the adapted model will be asked to caption the same "
                "boxes in Section 9. `validate_inputs` applies exactly the checks `caption` applies (image sides "
                "`MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, each box inside the image with sides of at least `MIN_BOX_SIDE`, "
                "`max_new_tokens` in `[1, MAX_NEW_TOKENS]`) and returns an input manifest; a box that runs past the screenshot's "
                "edge is validated too and its rejection recorded as a finding. `caption` draws the blue outline on a copy and "
                "returns the decoded text, the checked box, `new_tokens`, a `truncated` flag and the model identity. **No score "
                "exists.** As recorded in the model card, the inference-only smoke captioned these five widgets `go to new "
                "message`, `search bar`, `go to next`, `select ana` and `profile` (the gear icon is the recorded miss). No "
                "reference captions are authored for the mock, so its `evaluation_report` is `not-measurable` by design and the "
                "keyword checks are observations; whether captions are *right* is what Section 6 measures on the test widgets "
                "with several human references each. The image digest depends on the Pillow build's bundled font rendering."
            ),
            "code": (
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "CAPTION_MAX_TOKENS = 20  # @param {{type:\"integer\"}}\n\n\n"
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
                "screen, widgets = synthetic_screen()\n"
                "screen_name = 'synthetic_messaging_screen_540x960.png'\n"
                "widget_names = [name for name, _, _ in widgets]\n"
                "boxes = [box for _, box, _ in widgets]\n"
                "expected_keywords = [keywords for _, _, keywords in widgets]\n"
                "screen_sha256 = hashlib.sha256(np.asarray(screen.convert('RGB')).tobytes()).hexdigest()\n"
                "ceilings = {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PATCHES': MAX_PATCHES, 'MIN_BOX_SIDE': MIN_BOX_SIDE, 'BOX_COLOR': BOX_COLOR, 'BOX_WIDTH': BOX_WIDTH, 'MAX_NEW_TOKENS': MAX_NEW_TOKENS, 'DEFAULT_MAX_NEW_TOKENS': DEFAULT_MAX_NEW_TOKENS, 'DECODING': DECODING, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'MIN_CAPTIONS': MIN_CAPTIONS, 'MAX_CAPTION_CHARS': MAX_CAPTION_CHARS}}\n"
                "print(ceilings)\n"
                "input_manifest = validate_inputs(screen, boxes, max_new_tokens=CAPTION_MAX_TOKENS, names=[screen_name])\n"
                "try:\n"
                "    validate_inputs(screen, [[0, 0, screen.width + 20, 100]])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'box-outside-image-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'screen': screen_name, 'rgb_sha256': screen_sha256[:16] + '...', 'boxes': len(boxes), 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n"
                "results = []\n"
                "for name, box, keywords in widgets:\n"
                "    started = time.perf_counter()\n"
                "    result = pipe.caption(screen, box, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "    result['widget'] = name\n"
                "    results.append({{'seconds': round(time.perf_counter() - started, 3), **result}})\n"
                "    print(f\"{{name}} {{box}}\\n   caption: {{result['caption']!r}}  ({{result['new_tokens']}} tokens{{', TRUNCATED' if result['truncated'] else ''}})  keywords: {{keyword_hits(result['caption'], keywords)}}\")\n"
                "checks = {{\n"
                "    'one_result_per_widget': len(results) == len(widgets),\n"
                "    'captions_are_text': all(isinstance(r['caption'], str) for r in results),\n"
                "    'budget_respected': all(r['new_tokens'] <= CAPTION_MAX_TOKENS for r in results),\n"
                "    'box_echoed': all(r['box'] == [int(v) for v in box] for r, box in zip(results, boxes, strict=True)),\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'caption output failed a sanity check: {{checks}}')\n"
                "frozen_screen = evaluation_report(results, None, sample_kind='synthetic')\n"
                "frozen_keywords = [keyword_hits(r['caption'], k) for r, k in zip(results, expected_keywords, strict=True)]\n"
                "print({{'checks': checks, 'frozen_screen_verdict': frozen_screen['verdict'], 'keyword_hits': frozen_keywords}})"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model's score on the test widgets\n\n"
                "Three systems frame the adaptation, each read four ways. The **constant-caption baseline** answers every "
                "widget with the one training caption that scores highest against all other training references — the corpus "
                "medoid, a phrase that is safe everywhere and right nowhere. The **colour-nearest-neighbour baseline** answers "
                "with the first reference caption of the training widget whose 3×3 mean-colour grid over its box crop is "
                "closest — a lookup that knows the widget through 27 numbers. The **frozen model** captions every test widget "
                "with the budget from Section 5 and is scored with the same metrics: **BLEU-4** (corpus-level clipped n-gram "
                "precision with a brevity penalty), **ROUGE-L** (longest-common-subsequence F-measure against the best "
                "reference), **CIDEr-D** (TF-IDF-weighted n-gram consensus over all references, the metric Widget Captioning is "
                "ranked by) and the plumbing check **unigram F1**, all after lower-casing and punctuation removal. The "
                "checkpoint was fine-tuned on Widget Captioning's training apps, so expect it well above both baselines; the "
                "cell asserts only that it beats the constant caption. Read the per-category breakdown: icons (`small-widget`) "
                "carry no text for the model to read. The measured values of the first clean run are recorded in "
                "`docs/release-verification.md` and the model card."
            ),
            "code": (
                "baseline_constant = constant_caption_baseline(train_records, test_records)\n"
                "baseline_neighbour = colour_neighbour_baseline(train_records, test_records)\n"
                "METRICS = ('bleu4', 'rouge_l', 'cider_d', 'unigram_f1')\n"
                "print({{'constant_caption_baseline': {{k: round(baseline_constant[k], 3) for k in METRICS}}, 'n': baseline_constant['n'], 'caption': baseline_constant['baseline']}})\n"
                "print({{'colour_neighbour_baseline': {{k: round(baseline_neighbour[k], 3) for k in METRICS}}, 'note': baseline_neighbour['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'mean_words': round(frozen_test['mean_words'], 1), 'n': frozen_test['n'], 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n\n\n"
                "def by_category(predictions, records):\n"
                "    \"\"\"CIDEr-D per category, with document frequencies from the whole evaluated set (as in `evaluate`).\"\"\"\n"
                "    scores = cider_d(list(predictions), [reference_captions(r) for r in records])\n"
                "    groups = collections.defaultdict(list)\n"
                "    for record, score in zip(records, scores, strict=True):\n"
                "        groups[record['category']].append(score)\n"
                "    return {{category: {{'n': len(values), 'cider_d': round(sum(values) / len(values), 3)}} for category, values in sorted(groups.items())}}\n\n\n"
                "medoid = medoid_caption(train_records)\n"
                "constant_fields = by_category([medoid] * len(test_records), test_records)\n"
                "frozen_predictions = pipe.predict(test_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "frozen_fields = by_category(frozen_predictions, test_records)\n"
                "print({{'by_category': {{'constant': constant_fields, 'frozen': frozen_fields}}}})\n"
                "for record, prediction in list(zip(test_records, frozen_predictions, strict=True))[:3]:\n"
                "    print({{'category': record['category'], 'frozen': prediction, 'references': reference_captions(record)[:2]}})\n"
                "assert frozen_test['cider_d'] > baseline_constant['cider_d']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the caption decoder's last blocks\n\n"
                "`pipe.adapt` trains only the last `TRAINABLE_DECODER_LAYERS` blocks of the caption decoder plus the decoder's "
                "final layer norm — two blocks by default, 18,879,744 of 282,285,696 parameters; the image encoder, every "
                "embedding and the untied output projection (a 50,244 × 768 matrix) stay frozen. Each step takes "
                "`BATCH_SIZE` widgets, draws their boxes on their screenshots exactly as `caption` does, runs the frozen "
                "encoder once per widget (recomputed each step without gradients — the box makes every widget its own image) "
                "and trains on **every** (widget, reference caption) pair of the batch against that encoder output; the target "
                "is the tokenised caption with its end-of-sequence token, decoded with teacher forcing and scored with the "
                "model's own cross-entropy (padding ignored); AdamW at a fixed learning rate, gradient clipping at 1.0, seeded "
                "shuffling and no scheduler. Epoch 0 records the frozen model's validation metrics; every epoch is scored on the "
                "validation widgets, and the epoch with the highest validation CIDEr-D is kept — a validation split of about "
                "sixty widgets makes that selection noisy, which is why the held-out split in Section 8 is what the numbers are "
                "read from. If no epoch beats the frozen model on validation, the selector keeps epoch 0 and the adapter "
                "reproduces the frozen captions; that outcome is reported, not hidden."
            ),
            "code": (
                "EPOCHS = 3  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 1e-5  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 4  # @param {{type:\"integer\"}}\n"
                "TRAINABLE_DECODER_LAYERS = 2  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "        row['val_mean_words'] = round(entry['val']['mean_words'], 1)\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, trainable_decoder_layers=TRAINABLE_DECODER_LAYERS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'training_widgets': adapt_result['n_train'], 'training_pairs': adapt_result['n_pairs'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test widgets were never used for training or epoch selection, and no test app or screen appears in the "
                "training or validation splits. The adapted model is scored exactly as the frozen model was in Section 6, the "
                "four systems are put side by side on all four metrics, and the per-category CIDEr-D is repeated. Read it in "
                "this order: **CIDEr-D** first (the consensus metric the epoch was selected on), then BLEU-4 and ROUGE-L, which "
                "can move the other way when the adapted captions change length, then the `small-widget` / `large-widget` "
                "split. `adapted_beats_frozen` records whether the held-out CIDEr-D rose. About 160 widgets from one seeded "
                "draw of one shard give **no dispersion estimate**; the deltas are sample-sanity evidence that the adaptation "
                "contract works, not a benchmark, and a gain on Rico apps says nothing about your app until you measure it there."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "adapted_val = pipe.evaluate(val_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "adapted_predictions = pipe.predict(test_records, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "adapted_fields = by_category(adapted_predictions, test_records)\n"
                "comparison = {{metric: {{'constant': round(baseline_constant[metric], 3), 'neighbour': round(baseline_neighbour[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['mean_words'] = {{'constant': round(baseline_constant['mean_words'], 1), 'neighbour': round(baseline_neighbour['mean_words'], 1), 'frozen': round(frozen_test['mean_words'], 1), 'adapted': round(adapted_test['mean_words'], 1)}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['by_category'] = {{category: {{'n': frozen_fields[category]['n'], 'constant': constant_fields[category]['cider_d'], 'frozen': frozen_fields[category]['cider_d'], 'adapted': adapted_fields[category]['cider_d']}} for category in frozen_fields}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "for record, before, after in list(zip(test_records, frozen_predictions, adapted_predictions, strict=True))[:3]:\n"
                "    print({{'category': record['category'], 'frozen': before, 'adapted': after, 'references': reference_captions(record)[:2]}})\n"
                "adapted_beats_frozen = adapted_test['cider_d'] > frozen_test['cider_d']\n"
                "print({{'adapted_beats_frozen': adapted_beats_frozen}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'categories': categories,\n"
                "    'max_new_tokens': CAPTION_MAX_TOKENS,\n"
                "    'baselines': {{'constant_caption': baseline_constant, 'colour_neighbour': baseline_neighbour}},\n"
                "    'frozen_test': frozen_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "    'adapted_beats_frozen': adapted_beats_frozen,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Re-caption the drawn screen, export the adapter and reload it\n\n"
                "The five widgets on the drawn screen from Section 5 are captioned again by the adapted model and their keyword "
                "observations repeated — a mock from a different image family than the Rico screenshots it was tuned on, so this "
                "is a small look at whether the adaptation changed the model's behaviour *outside* its sample (five widgets of "
                "evidence, not a measurement; a different caption here is a finding to record, not a failure). Both caption "
                "sets are written as CSV.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the caption decoder's last two blocks and its final layer norm, "
                "about 76 MB — as `adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id "
                "and revision, the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the "
                "training configuration and the epoch history (OUT8). `Pix2StructWidgetCaptioningPipeline.from_artifact` "
                "re-verifies the base snapshot, checks the artifact manifest, its digest and its exact tensor set **before** "
                "deserialising, refuses any tensor that is not a caption-decoder tensor, and overlays the tensors onto a freshly "
                "loaded base — a new object from files, not the in-memory model (VER2). The cell asserts identical captions on "
                "eight test widgets (VER4)."
            ),
            "code": (
                "import csv\n"
                "import shutil\n\n"
                "adapted_results = []\n"
                "for name, box, keywords in widgets:\n"
                "    result = pipe.caption(screen, box, max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "    result['widget'] = name\n"
                "    adapted_results.append(result)\n"
                "adapted_screen = evaluation_report(adapted_results, None, sample_kind='synthetic')\n"
                "adapted_keywords = [keyword_hits(r['caption'], k) for r, k in zip(adapted_results, expected_keywords, strict=True)]\n"
                "for before, after, hits in zip(results, adapted_results, adapted_keywords, strict=True):\n"
                "    print({{'widget': before['widget'], 'frozen': before['caption'], 'adapted': after['caption'], 'adapted_keywords': hits}})\n"
                "with open('outputs/{stem}_captions.csv', 'w', encoding='utf-8', newline='') as handle:\n"
                "    writer = csv.writer(handle)\n"
                "    writer.writerow(['image', 'widget', 'box', 'frozen_caption', 'adapted_caption', 'adapted_new_tokens', 'expected_keywords'])\n"
                "    for before, after, keywords in zip(results, adapted_results, expected_keywords, strict=True):\n"
                "        writer.writerow([screen_name, before['widget'], ' '.join(str(v) for v in before['box']), before['caption'], after['caption'], after['new_tokens'], ' | '.join(keywords)])\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = Pix2StructWidgetCaptioningPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = pipe.predict(test_records[:8], max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "after = reloaded.predict(test_records[:8], max_new_tokens=CAPTION_MAX_TOKENS)\n"
                "parity = {{'identical_captions': sum(a == b for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_captions'] == parity['of']\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHT_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'SafeTensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'release': CORPUS_RELEASE, 'license': CORPUS_LICENSE, 'file': CORPUS_FILE, 'sample_widgets': SAMPLE_WIDGETS}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'screen': {{'name': screen_name, 'size': list(screen.size), 'rgb_sha256': screen_sha256, 'widgets': widget_names, 'boxes': boxes, 'expected_keywords': expected_keywords}}, 'items': [{{k: r[k] for k in ('widget', 'caption', 'new_tokens', 'truncated', 'seconds')}} for r in results], 'frozen_report': frozen_screen, 'frozen_keywords': frozen_keywords, 'adapted_items': [{{k: r[k] for k in ('widget', 'caption', 'new_tokens', 'truncated')}} for r in adapted_results], 'adapted_report': adapted_screen, 'adapted_keywords': adapted_keywords}},\n"
                "    'comparison': comparison,\n"
                "    'adapted_beats_frozen': adapted_beats_frozen,\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The frozen model is a Widget Captioning checkpoint scored on apps it was not trained on, beside two non-neural "
        "baselines, and a bounded fine-tuning of the caption decoder's last two blocks on a few hundred more widgets is then "
        "scored on an app-disjoint test split — overall, and separately on small and larger widgets — with an adapter that "
        "reloads to identical captions. That is the claim: the adaptation contract works end to end on a real widget-captioning "
        "corpus, and the numbers it produces are read on four metrics and per category against two non-neural baselines and "
        "the frozen model rather than in isolation. Whether the held-out CIDEr-D rose is recorded as `adapted_beats_frozen`, "
        "not assumed.\n\n"
        "The test split is about 160 widgets on whole apps from one seeded draw of one shard, the validation split that picks "
        "the epoch is about 60, the metrics are four reference-based scores (own pure-Python implementations of the "
        "`coco-caption` definitions, with CIDEr-D's document frequencies from the evaluated set — so its absolute value is not "
        "comparable to the benchmark's published numbers — and none a human judgement). Because the checkpoint already saw "
        "Widget Captioning's training apps, a small or zero gain is the expected outcome and not a failure of the contract; a "
        "learning rate that is too high overfits this little data within an epoch, which the validation-based selector reports "
        "by keeping epoch 0. So a gain here says the contract works, not that the adapted model describes your app's widgets "
        "better; it still captions every box — including one around nothing — with a fluent phrase. Fine-tuning on a narrow "
        "sample can also erode the model elsewhere; the drawn screen re-captioned in Section 9 is five widgets of evidence about "
        "that, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** the constant-caption and colour-neighbour baselines and the "
        "frozen model's score on *your* references are the numbers to read before any adapted one, per category and on "
        "CIDEr-D. **Leakage:** keep every widget of an app in one split (the contract does this), never split at random over "
        "screens of the same app. **The box is part of the input:** a wrong or loose box is a wrong request, and the model "
        "will caption it anyway.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real widget-captioning "
        "corpus, validate the demonstrated dataset contract without leakage, execute the inference contract and a bounded "
        "fine-tuning, evaluate against two trivial baselines and the frozen model on an app-disjoint split, and emit the shown "
        "machine-readable artifacts — without the repository being reachable. It does **not** establish benchmark superiority, "
        "caption quality on any other app population or platform, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** raise `LEARNING_RATE` and watch the training loss fall "
        "while the validation CIDEr-D drops and the selector keeps an early epoch; set `TRAINABLE_DECODER_LAYERS = 1` and "
        "compare the artifact size and the held-out score; loosen a box on the drawn screen and watch the caption follow it; or "
        "bring your own screenshots through BYOD and read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/pix2struct-ui-captioning-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/pix2struct-ui-captioning-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/pix2struct-ui-captioning-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model (Google, Apache-2.0): https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/google-research/pix2struct\n"
        "- Pix2Struct: Screenshot Parsing as Pretraining for Visual Language Understanding (Lee et al., ICML 2023): https://arxiv.org/abs/2210.03347\n"
        "- Widget Captioning: Generating Natural Language Description for Mobile User Interface Elements (Li et al., EMNLP 2020): https://arxiv.org/abs/2010.04295 — data: https://github.com/google-research-datasets/widget-caption\n"
        "- Rico: A Mobile App Dataset for Building Data-Driven Design Applications (Deka et al., UIST 2017): https://dl.acm.org/doi/10.1145/3126594.3126651\n"
        "- Widget Captioning as mirrored on the Hugging Face Hub (CC BY 4.0): https://huggingface.co/datasets/bevaya/RICO-WidgetCaptioning\n"
        "- BLEU: a Method for Automatic Evaluation of Machine Translation (Papineni et al., 2002): https://aclanthology.org/P02-1040/\n"
        "- ROUGE: A Package for Automatic Evaluation of Summaries (Lin, 2004): https://aclanthology.org/W04-1013/\n"
        "- CIDEr: Consensus-based Image Description Evaluation (Vedantam et al., 2015): https://arxiv.org/abs/1411.5726\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
