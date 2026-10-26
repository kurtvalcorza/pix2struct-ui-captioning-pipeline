# Release verification

`tutorials/pix2struct_ui_captioning_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the single kernel cell that builds (or reuses, by lock digest) the isolated hash-locked uv environment and routes every later cell to it, with no `pip install` into the kernel and no restart request; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the pinned
  Widget Captioning dataset revision is the one other 40-hex string allowed);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `Pix2StructWidgetCaptioningPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` and `read_corpus` from
  the pinned cache path, `build_sample_dataset(..., seed=SPLIT_SEED, image_dir=...)` / `load_byod_dataset`,
  `validate_dataset` per split, `check_split_disjoint`, `write_dataset_jsonl`, the ceiling print, `validate_inputs`
  with the box-outside-image refusal probe, `pipe.caption` with the sanity checks and keyword observations,
  `evaluation_report` on the drawn screen, `constant_caption_baseline`, `colour_neighbour_baseline`, `pipe.evaluate` on the frozen
  model and on the validation and test splits after adaptation, the per-category breakdown through `cider_d` and
  `reference_captions`, `pipe.adapt` with its explicit hyperparameters, `evaluation_report` on the screen after
  adaptation, `pipe.save_artifact`, `Pix2StructWidgetCaptioningPipeline.from_artifact` and the reload-parity assertion,
  the recorded `adapted_beats_frozen` flag, and the provenance fields `weight_format`, `weight_sha256` and the
  `corpus` block), the six expected `outputs/` paths, the learner-facing statements and the gated-off BYOD default;
  forbidden patterns (credential-in-URL, any `git clone` / `github.com` / repository import on the primary path, a
  mutable `revision='main'`, direct `from transformers import` / `Pix2StructForConditionalGeneration` /
  `Pix2StructProcessor` / `.generate(` / `from huggingface_hub import` / `urllib.request` / `pyarrow` /
  `safetensors` / `torch.optim` / `.backward(` / `pipe._model` use **outside the carried module cells**,
  `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `numpy`, `pillow`,
`huggingface-hub` and `pyarrow`, runs `ruff check src tests tools`, `tools/build_notebook.py --check`, and the
offline suites (`tests/test_pipeline.py`, `tests/test_adaptation.py`, `tests/test_role_helpers.py`,
`tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected runner and shard downloader,
tiny PIL drawings, a small generated parquet shard, temporary manifests, no weights).
`tests/test_adaptation_model.py` builds a 3-layer, 32-wide random Pix2Struct from the committed config,
processor (header rendering off) and tokenizer and runs the real `evaluate` / `adapt` /
`save_artifact` / `load_artifact` path on it offline; its pinned-checkpoint and CUDA cases skip unless
`model.safetensors` is staged and a GPU is visible. These are source/provenance and unit checks. They are **not**
execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; a GPU runtime is recommended for Sections 6–8) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/pix2struct-widget-captioning-base/` or the corpus cache `weights/widget-captioning/`, and with
   the shard's SHA-256 pin recorded in the committed `samples.py` (the reader refuses to run otherwise);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `CAPTION_MAX_TOKENS = 20`, `EPOCHS = 3`, `LEARNING_RATE = 1e-5`,
   `BATCH_SIZE = 4`, `TRAINABLE_DECODER_LAYERS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`; they are installed into the isolated environment Section 1 builds, so no interpreter restart is
   expected);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `google/pix2struct-widget-captioning-base` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (8 files), and
     `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with no further Hub access (a
     font fetch in the logs after staging is a finding);
   - Section 4: `fetch_corpus` downloading `data/test-00000-of-00002.parquet` at the pinned dataset revision and
     accepting it only with the pinned size and SHA-256; `read_corpus` returning the pinned row count; the seeded
     allocation of whole apps into about 320 / 64 / 160 training, validation and test widgets with
     `check_split_disjoint` reporting no shared app or screen, the category mix and the three dataset digests printed;
     `outputs/pix2struct_ui_captioning_train.jsonl` written; the four dataset refusal probes each raising
     `ValueError`;
   - Section 5: the ceilings surfaced; the messaging-app screen drawn; `validate_inputs` writing
     `outputs/pix2struct_ui_captioning_input_manifest.json` (verdict `accepted`, one recorded rejection finding from
     the box-outside-image probe); `pipe.caption` on the five boxes with every sanity check `True`, the keyword
     observations printed and `evaluation_report` verdict `not-measurable` (the inference-only notebook's runs
     captioned `go to new message`, `search bar`, `go to next`, `select ana` and `profile`; a different caption on
     another runtime is a finding to record, not a failure);
   - Section 6: the constant-caption and colour-neighbour baselines and the frozen model's test score with the
     per-category breakdown, and the recorded verdict `frozen_beats_constant` (the frozen CIDEr-D against the constant caption's);
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 18,879,744 trainable of 282,285,696
     parameters (29 tensors: two decoder blocks and the final layer norm), the training widget and pair counts, and
     the epoch history with validation CIDEr-D;
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison on four metrics,
     the per-category breakdown, `adapted_beats_frozen` printed and
     `outputs/pix2struct_ui_captioning_evaluation_report.json` written (the gain is **recorded, not asserted**, until
     a measured recipe is on file);
   - Section 9: the five widgets captioned by the adapted model with their keyword observations,
     `outputs/pix2struct_ui_captioning_captions.csv` written; `pipe.save_artifact` writing
     `outputs/pix2struct_ui_captioning_adapter/{adapter.safetensors,manifest.json}` and
     `Pix2StructWidgetCaptioningPipeline.from_artifact` reloading it with identical captions on eight test widgets
     (the cell asserts it); `outputs/pix2struct_ui_captioning_result.json` written with `NOTEBOOK_SOURCE`, the model
     identity and licence, the snapshot block, the `corpus` block, the inference-contract items, the comparison, the
     artifact digest, the reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the corpus cache were clean,
   outcome, produced outputs, the observed metrics (as observations, not a benchmark), the value of
   `adapted_beats_frozen` and any warning or applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Recorded executions

Notebook identity is the Git blob id of `tutorials/pix2struct_ui_captioning_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/pix2struct_ui_captioning_colab.ipynb`). Wall times, when recorded,
are the sum of per-cell times reported by the executor and include installs and the model download;
they are measurements for the stated runtime, not general estimates.

### `E2E` notebook

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-25 | `0383932` / `a17c3510` (PR #10 head) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `038393256ce84fda994883e17f3a135a161c5eb3`, git blob SHA-1 `a17c3510018af157673e31d2514e124acf45eb3d` checked, input unchanged; captured executed-notebook SHA-256 `14aa614f68341818cb3baf6eccdbfdaef01729a6e6a4118be9fa8027cdefb865` equals `run_summary.json` `executed_sha256` | 1161.8 s | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells (`exec.log`); isolated environment built in 48 s (CPython 3.12.12 from the 48-package lock, kernel Python 3.13.16, `cuda:0`); splits 330 / 77 / 164 widgets; held-out test CIDEr-D constant 0.138, colour neighbour 0.142, frozen 1.318 (`large-widget` 2.246, `small-widget` 0.934); validation CIDEr-D 0.777 → 0.882 → 0.881 → 0.894, `best_epoch` 3 (18,879,744 of 282,285,696 parameters, 893 training pairs, 640.0 s); adapted test BLEU-4 0.365 → 0.232, ROUGE-L 0.564 → 0.567, CIDEr-D 1.318 → 1.356, unigram F1 0.577 → 0.578, `delta_vs_frozen` BLEU-4 −0.134, `adapted_beats_frozen` `True` (CIDEr-D only; printed `Mixed result`); per category `large-widget` 2.246 → 2.291, `small-widget` 0.934 → 0.969; adapter exported (29 tensors, 75,522,608 bytes); `reload_parity` 8 of 8 identical captions, 0 reloaded captions differing from frozen. Not exercised: BYOD (REL12) and a browser `Run all` (REL2). Evidence in `docs/execution-evidence/2026-10-25-0383932/` |
| 2026-10-25 | `c45a12e` / `820076d3` (PR #10 head) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `c45a12e4d5a00fd1a54f44b71d723b319bee5799`, git blob SHA-1 `820076d362813ae7bd780a85728abf4b020a3bc2` checked, input unchanged; captured executed-notebook SHA-256 `7f3fc6aecc70fa25c28c949732d69d030ffa22a791b170bd662fd996cf6020bb` equals `run_summary.json` `executed_sha256` | 1180.8 s | **PASSED in one pass, no restart, 0 errors** — 13/13 code cells (`exec.log`); isolated environment built in 46 s; test (164 widgets) constant caption `go back` CIDEr-D 0.138, colour neighbour 0.142, frozen 1.318 (BLEU-4 0.365, ROUGE-L 0.564, unigram F1 0.577; `large-widget` 2.246, `small-widget` 0.934); validation CIDEr-D 0.777 → 0.882 → 0.881 → 0.894, `best_epoch` 3, fine-tune 646.6 s; adapted CIDEr-D 1.356 (+0.038), ROUGE-L 0.567, unigram F1 0.578, BLEU-4 0.232 (−0.134); `large-widget` 2.291, `small-widget` 0.969; adapter 29 tensors, 75,522,608 B; `reload_parity` `{'identical_captions': 8, 'of': 8, 'reloaded_captions_differing_from_frozen': 0}` (0 is inconclusive about the adapter being applied, as the notebook says). Not exercised: BYOD (REL12) and a browser `Run all` (REL2). Evidence in `docs/execution-evidence/2026-10-25-c45a12e/` |
| 2026-10-08 (03:39–03:58) | `c020a97` / `cf909dbcd4e5` (`NOTEBOOK_SOURCE.repository_revision` = `cbb199d…`, the source revision the generator embedded; `cbb199d..c020a97` changes only the isolated worker's `google.colab` stubs, which now carry a module spec, plus their regression test; generator `build_notebook.py/2.2`) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM, committed blob fetched at the 40-char SHA and Git-blob verified; kernel Python 3.13.15; isolated uv environment from the 48-package hash lock (Python 3.12.12, 55 s), torch 2.14.0+cu130, transformers 4.57.6, device `cuda:0`, float32, source `local-snapshot` | Default path, form parameters at their defaults, 13/13 code cells in order (cells 3–5 are the carried module definitions and print nothing). Staging: 8 manifest files (1,133,308,959 B) from `google/pix2struct-widget-captioning-base` @ `7e99642f…`, `verify_snapshot` 8 files. Section 4: shard sha256 `91d31536…` (1,811 widgets, 646 screens), split 330 / 77 / 164 widgets over 75 / 13 / 24 whole apps (test 48 `large-widget` + 116 `small-widget`), the four refusal probes rejected. Section 6 (n = 164): constant caption CIDEr-D 0.138, colour neighbour 0.142, frozen 1.318 (BLEU-4 0.365, ROUGE-L 0.564; `large-widget` 2.246, `small-widget` 0.934). Section 7: 18,879,744 of 282,285,696 parameters trainable, 893 pairs from 330 widgets; validation CIDEr-D 0.777 → 0.882 → 0.881 → 0.894, best epoch 3; 654.8 s. Section 8: adapted CIDEr-D 1.356 (+0.038), ROUGE-L 0.567, BLEU-4 0.232 (−0.134); `large-widget` 2.291, `small-widget` 0.969; `adapted_beats_frozen` `True` on CIDEr-D only (recorded, not asserted). Section 9: adapter 29 tensors, 75,522,608 B, sha256 `f1fe5cb24a21…`; reload parity 8/8 identical; 0 of the 8 reloaded captions differ from the frozen ones. Not exercised: BYOD upload, the box experiment (skipped by default) and the other activities. Byte-exact evidence in `docs/execution-evidence/2026-10-08-c020a97/`: executed notebook sha256 `36a08f9e7316…`, `exec.log` `a5ad8f647c99…`, `run_summary.json` `48e34aaeac06…` | 1188.3 s (one pass, no restart) | **PASSED** — one pass, no restart, 0 errors; evidence for this blob; status stays Candidate pending review |
| 2026-09-26 | `0edb5de` / `2e28896c0f42` (the PR head; `NOTEBOOK_SOURCE.repository_revision` = `metadata.dimer.generated_from` = `6583937…`) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pix2struct-ui-captioning` v3; image `gcr.io/kaggle-gpu-images/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461`, `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` (CUDA 13.0) / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | Default sample path (`USE_BYOD = False`, `SPLIT_SEED = 42`, all form defaults), `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 1292.2 s | **PASSED** — 11/11 code cells ok (1 restart after install cell: the pins replaced the loaded numpy and cuda-bindings); `stage_missing_files` fetched all 8 manifest entries (1,133,308,959 bytes) at `7e99642f…`, `verify_snapshot` 8 files, no font fetch after staging; the pinned shard `data/test-00000-of-00002.parquet` fetched and accepted (1,811 rows, 646 screens, 391 apps); split 330 / 77 / 164 widgets over 75 / 13 / 24 whole apps, disjoint; the four dataset refusal probes each rejected; input manifest `accepted` with the box-outside-image finding; frozen screen captions `go to new message`, `search bar`, `go to next`, `select ana`, `profile` (identical to the inference-only runs; keywords 4/5, `settings` missing), verdict `not-measurable`; test (164 widgets) CIDEr-D constant 0.138 / colour-neighbour 0.142 / frozen 1.318 / adapted 1.356, BLEU-4 0.0 / 0.0 / 0.365 / 0.232, ROUGE-L 0.075 / 0.087 / 0.564 / 0.567, unigram F1 0.077 / 0.087 / 0.577 / 0.578; by category CIDEr-D `large-widget` (48) 2.246 → 2.291, `small-widget` (116) 0.934 → 0.969; 18,879,744 of 282,285,696 parameters trained over 893 (widget, reference) pairs from 330 widgets; validation CIDEr-D by epoch 0.777 / 0.882 / 0.881 / 0.894 (best epoch 3); adaptation 612.4 s; `adapted_beats_frozen` true; adapted screen captions `go to new message`, `search for a conversation`, `go to next`, `select ana`, `go to profile` (keywords 4/5); adapter 29 tensors, reload parity 8/8 identical captions; preserved output SHA-256: `pix2struct_ui_captioning_result.json` `cbc9c323b9bb…`, `pix2struct_ui_captioning_evaluation_report.json` `3b431f346ceb…`, `pix2struct_ui_captioning_adapter/adapter.safetensors` `f1fe5cb24a21…` (75,522,608 bytes); 205 files / 1248 MB staged; run summary and executed notebook archived under `.agent/backups/kaggle-batch-2026-09-26/out/dimer-nb2-pix2struct-ui-captioning/v3/evidence/` in the workspace |
| 2026-09-26 | `fe3385a` / `4e74c5a4f655` (earlier blob, superseded) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-pix2struct-ui-captioning` v2; same image) | Default sample path, as above | 212.4 s | **FAILED** — 4/11 code cells ok; code cell 11 (Section 3, staging) raised `TypeError: _hub_download() takes 1 positional argument but 2 were given`: in the standalone notebook the carried `samples.py` cell's `_hub_download` shadowed the carried `pipeline.py` helper of the same name. Fixed by `6583937` (rename in `samples.py`) and `0edb5de` (notebook regenerated); the row above is the re-run of the fixed blob |

The rows below are the earlier inference-only notebook's runs; they are history and are **not** evidence for the `E2E` blob.

### Superseded `TASK-INFERENCE` notebook — local pre-flight (not a supported runtime)

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | notebook blob `5baeb8f48104` (commit `e764380`, generated at `2fe75b2`; `NOTEBOOK_SOURCE.repository_revision` = `2fe75b2…`) | Local Windows-venv harness (`run_nb_local.py`: nbclient 0.11.0, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=-1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6 | Default synthetic path, all 8 code cells: pinned install skipped (pre-installed), `stage_missing_files` fetched all 8 manifest entries (1.13 GB) from the Hub cache at the pinned revision into the scratch `weights/`, `verify_snapshot` PASS (8 files), no font or other download in the log after staging, five `caption` calls → `go to new message`, `search bar`, `go to next`, `select ana`, `profile` (2–5 tokens, none truncated, 2.03–2.59 s), keyword observations 4/5 (`settings` absent — the recorded miss), `evaluation_report` `not-measurable` (no references, by design), mock digest `364146e9…` (Pillow 11.3.0 bundled font), 5 outputs written | 123.3 s | PASS — pre-flight only; not promotion evidence |

### Superseded `TASK-INFERENCE` notebook — manual clean-runtime evidence

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-14 | `d32697f` / `bc29545d855d` | Kaggle CPU (`kurtvalcorza/dimer-nb2-pix2struct-ui-captioning` v1) | Default sample path | 302.6 s | **PASSED** — 8/8 ok code cells executed cleanly, 18 files, 1133 MB staged |
| 2026-10-25 (18:45:06 UTC start) | `4f0aeee` / `7b6c2aa1` (PR #10 head; historical wording revision; final confirming run; fetched at the full SHA and blob-verified before the session) | Colab CLI sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`. Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log | Default sample path, no overrides, input unchanged; captured executed-notebook SHA-256 `16a3c565e8e533a2c300844701491f01e477123d5cd2a78a44d418bc32fd7fb0` equals `run_summary.json` `executed_sha256` | 1206.8 s session wall | **PASSED — one pass, no restart, 0 errors, 13/13 code cells.** Isolated environment built in 47 s (CPython 3.12.12, 48-package lock); splits 330 / 77 / 164 widgets; baselines constant `go back` CIDEr-D 0.138, colour neighbour 0.142; frozen BLEU-4 0.365 / ROUGE-L 0.564 / CIDEr-D 1.318 / unigram F1 0.577 (84.0 s); validation CIDEr-D 0.777 → 0.882 → 0.881 → 0.894 (epoch 3 kept), 893 pairs from 330 widgets, 665.7 s of fine-tuning; adapted BLEU-4 0.232 / ROUGE-L 0.567 / CIDEr-D 1.356 / unigram F1 0.578 (`delta_vs_frozen` BLEU-4 −0.134, CIDEr-D +0.038), `adapted_beats_frozen` True (CIDEr-D only); reload parity 8 of 8 identical, 0 of 8 reloaded captions differ from the frozen ones. Not exercised: BYOD on a hosted runtime (REL12), the box-loosening activity, a browser Run all (REL2). Final confirming run under the 2026-10-26 historical-wording ruling (drift recorded under that revision). Evidence `docs/execution-evidence/2026-10-25-4f0aeee/` — SHA-256 captured → committed: `pix2struct_ui_captioning_colab_4f0aeee_colab-cli-t4_default.ipynb` `16a3c565e8e533a2c300844701491f01e477123d5cd2a78a44d418bc32fd7fb0` → unchanged (no local path); `exec.log` `c6cf445739bf64f87d9a1ac8729712f6afb55398cb2e15aa95585201aa44ccf7` → `4b271ec421ddf4178d8fcc577e05f47c006b0e86a3ebd6a42e5f2810492f20a6` (2 local-path replacements); `run_summary.json` `62101b793ef621785327c87ac9b6ac7d142397e3176ff4e30318c5d7e9608af5` → unchanged (no local path) |
| 2026-10-26 (00:09:42 UTC start) | `6aa1bed` / `d523f506` (PR #10 head; final prose pass; confirming run; fetched at the full SHA and blob-verified before the session) | Colab CLI sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`. Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log | Default sample path, no overrides; input unchanged; captured executed-notebook SHA-256 `058f4287d31e10bcd168fa884a5e14ae059a73e47d6a199c0c64cbb4686e3c4f` equals `run_summary.json` `executed_sha256` | 1138.5 s session wall | **PASSED — one pass, no restart, 0 errors, 13/13 code cells.** Isolated environment built in 45 s (CPython 3.12.12, 48-package lock); splits 330 / 77 / 164 widgets; baselines constant `go back` CIDEr-D 0.138, colour neighbour 0.142; frozen BLEU-4 0.365 / ROUGE-L 0.564 / CIDEr-D 1.318 / unigram F1 0.577 (80.5 s); validation CIDEr-D 0.777 → 0.882 → 0.881 → 0.894 (epoch 3 kept), 893 pairs from 330 widgets, 630.7 s of fine-tuning; adapted BLEU-4 0.232 / ROUGE-L 0.567 / CIDEr-D 1.356 / unigram F1 0.578 (`delta_vs_frozen` BLEU-4 −0.134, CIDEr-D +0.038), `adapted_beats_frozen` True (CIDEr-D only); reload parity 8 of 8 identical, 0 of 8 reloaded captions differ from the frozen ones. Not exercised: BYOD on a hosted runtime (REL12), the box-loosening activity, a browser Run all (REL2). Confirming run of the final prose pass (drift recorded under the 2026-10-26 current status). Evidence `docs/execution-evidence/2026-10-26-6aa1bed/` — SHA-256 captured → committed: `pix2struct_ui_captioning_colab_6aa1bed_colab-cli-t4_default.ipynb` `058f4287d31e10bcd168fa884a5e14ae059a73e47d6a199c0c64cbb4686e3c4f` → unchanged (no local path); `exec.log` `4b096e53d66d15cabfad50f800ceed8885aea5a88ff71a4badb19056d32574ba` → `6c46ae0f3ce8b7bc2d2e3b6b240047a6bafcebc17d0c7fe144eab0d2662a057f` (2 local-path replacements); `run_summary.json` `6c3af97d5b75d87ac86ace295ee1fb373537d70c859e8952c98761c10cfc49b0` → unchanged (no local path) |

## Current status

**Current status (2026-10-26, DAT19/DAT20 refusal display): Candidate.** The current blob `c9c65eea` (code commit `fe19a58`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results and `outputs/` files and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD=True;BYOD_PATH='/etc/hostname'`) are pending, and the hosted runs of earlier blobs, including `d523f506`, are history for it (REL14).

**Previous status (2026-10-26, final prose pass): Candidate, hosted run recorded.** **Code commit `a027f5f`, notebook commit `4fdb32b`.** The current blob of `tutorials/pix2struct_ui_captioning_colab.ipynb` is `d523f5060b93257cdfcc0e49aef6eb5e265cfda3`, **Candidate**; that exact blob (PR #10 head `6aa1bed`) passed its confirming hosted run on 2026-10-26: Colab CLI sequential execution on a fresh Colab Tesla T4, one pass, no restart, 0 errors (13/13 code cells, 1138.5 s wall; test CIDEr-D frozen 1.318 / adapted 1.356, BLEU-4 0.365 → 0.232; reload parity 8/8). The paths this blob's runs cover: the default path only. Remaining gates: REL12 (a hosted BYOD run, one input accepted and one refused) and REL2 (a browser `Run all`). Runs of earlier blobs are history (REL14). See its row under "Recorded executions". Kurt's 2026-10-26 ruling: fix only the prose rule breaks the final evidence flagged; recorded drift stays recorded, not fixed. Markdown only: the Section 6 intro no longer says to expect the checkpoint well above both baselines. The runs of blob `7b6c2aa1` and earlier blobs are history (REL14); the records below are kept.

Evidence files for the 2026-10-26 default run of blob `d523f506` (confirming run of the final prose pass), copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-26-6aa1bed/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `pix2struct_ui_captioning_colab_6aa1bed_colab-cli-t4_default.ipynb` | `058f4287d31e10bcd168fa884a5e14ae059a73e47d6a199c0c64cbb4686e3c4f` | `058f4287d31e10bcd168fa884a5e14ae059a73e47d6a199c0c64cbb4686e3c4f` (unchanged, no local path) |
| `exec.log` | `4b096e53d66d15cabfad50f800ceed8885aea5a88ff71a4badb19056d32574ba` | `6c46ae0f3ce8b7bc2d2e3b6b240047a6bafcebc17d0c7fe144eab0d2662a057f` (2 local-path replacements) |
| `run_summary.json` | `6c3af97d5b75d87ac86ace295ee1fb373537d70c859e8952c98761c10cfc49b0` | `6c3af97d5b75d87ac86ace295ee1fb373537d70c859e8952c98761c10cfc49b0` (unchanged, no local path) |

**Final prose pass: drift record (2026-10-26).** Under Kurt's 2026-10-26 ruling the notebook prose of blob `d523f506` is not changed again: a value this run printed that differs from the values the prose attributes to the named earlier runs is recorded here as drift, not fixed. Checked against this run's output: the splits (330 / 77 / 164), the baselines (CIDEr-D 0.138 / 0.142, BLEU-4 0.0 / 0.0), the frozen 0.365 / 0.564 / 1.318 / 0.577, the validation CIDEr-D path 0.777 → 0.882 → 0.881 → 0.894 with epoch 3 kept, the 893 training pairs, the adapted CIDEr-D 1.356 and ROUGE-L 0.567, unigram F1 0.578, BLEU-4 0.232 with the printed `delta_vs_frozen` −0.134, and the reload parity (8 of 8, 0 differing from frozen) all equal the values the prose gives for the named hosted runs.

Drift (this run versus the prose's named-run values):

- Duration (header md0, prerequisites md2, Section 7 answer md25): the prose gives the named Colab T4 runs' whole default path as 1,188.3 s, 1,180.8 s and 1,161.8 s and their fine-tuning as 654.8 s, 646.6 s and 640.0 s (md25: "640.0–654.8 s of fine-tuning in the three Colab T4 runs"). This run took 1,138.5 s for the whole path and 630.7 s of fine-tuning, both below every named Colab run's value; the fine-tuning lies inside the prerequisites' "612.4–654.8 s on a T4", which includes the Kaggle run.

Rule check: no expectation range, timing promise, run-status sentence or unlabelled run-derived number was found in the notebook prose.

**Historical wording revision (2026-10-26, code commit `e431d1a`, regenerated in `1a7d890`).** Markdown only, code cells unchanged. Following the 2026-10-26 ruling that run-derived numbers in the notebook are records, never expectations: the header and prerequisites give the recorded walls and fine-tuning times of every hosted run (Colab T4 `cf909dbc` 1,188.3 s / 654.8 s, `820076d3` 1,180.8 s / 646.6 s, `a17c3510` 1,161.8 s / 640.0 s; Kaggle T4 `2e28896c` 1,292.2 s / 612.4 s) in place of "about 11 minutes" and "about 21½ minutes"; "the recorded run" now means each of those four runs, which printed the same metrics; the Section 8 BLEU-4 delta is the printed `delta_vs_frozen` −0.134 (it read −0.133, the difference of rounded scores); the CPU sentence states that the recorded CPU runs are inference-only and gives the estimate without a "not timed" status; Section 10 no longer calls a small gain "the expected outcome". The regenerated notebook blob `7b6c2aa1a91517c44f28762e7fe96f1ed5ca4c9f` was **Candidate**; that exact blob (commit `4f0aeee`) ran in one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 13/13 code cells, 1206.8 s; test CIDEr-D frozen 1.318 / adapted 1.356, BLEU-4 0.365 → 0.232; reload parity 8/8; see its row under "Recorded executions"). The paths this blob's runs cover: the default path only. Remaining gates: REL12 (a hosted BYOD run) and REL2 (a browser `Run all`). The run of `a17c3510` and every earlier record are kept as history (REL14).

Evidence files for the 2026-10-25 default run of blob `7b6c2aa1` (final confirming run), copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-4f0aeee/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `pix2struct_ui_captioning_colab_4f0aeee_colab-cli-t4_default.ipynb` | `16a3c565e8e533a2c300844701491f01e477123d5cd2a78a44d418bc32fd7fb0` | `16a3c565e8e533a2c300844701491f01e477123d5cd2a78a44d418bc32fd7fb0` (unchanged, no local path) |
| `exec.log` | `c6cf445739bf64f87d9a1ac8729712f6afb55398cb2e15aa95585201aa44ccf7` | `4b271ec421ddf4178d8fcc577e05f47c006b0e86a3ebd6a42e5f2810492f20a6` (2 local-path replacements) |
| `run_summary.json` | `62101b793ef621785327c87ac9b6ac7d142397e3176ff4e30318c5d7e9608af5` | `62101b793ef621785327c87ac9b6ac7d142397e3176ff4e30318c5d7e9608af5` (unchanged, no local path) |

**Final confirming run: drift record (2026-10-26).** Under the 2026-10-26 ruling ("historical wording, then stop") the notebook prose is not changed again; a value this run printed that differs from the values the prose attributes to the named earlier runs is recorded here as drift, not fixed. Checked against this run's output: the splits (330 / 77 / 164), the baselines (CIDEr-D 0.138 / 0.142, BLEU-4 0.0 / 0.0), the frozen 0.365 / 0.564 / 1.318 / 0.577, the validation CIDEr-D path 0.777 → 0.882 → 0.881 → 0.894 with epoch 3 kept, the adapted CIDEr-D 1.356 and ROUGE-L 0.567, unigram F1 0.578, BLEU-4 0.232 with the printed `delta_vs_frozen` −0.134, and the reload parity (8 of 8, 0 differing from frozen) all equal the values the prose gives for "the recorded run", which the how-to-use paragraph defines as each of the named hosted runs.

Drift (this run versus the prose's named-run values):

- Duration (header and prerequisites): the prose gives the named Colab T4 runs' whole default path as 1,188.3 s, 1,180.8 s and 1,161.8 s and their fine-tuning as 654.8 s, 646.6 s and 640.0 s (Section 7 answer: "640.0–654.8 s of fine-tuning in the three Colab T4 runs"; prerequisites: "612.4–654.8 s on a T4"). This run took 1,206.8 s for the whole path and 665.7 s of fine-tuning, both above every named run's value.

Prose statements that break the historical rules themselves (reported, not fixed here):

- md20 (Section 6 intro): "The checkpoint was fine-tuned on Widget Captioning's training apps, so expect it well above both baselines". A claim about which item wins, written as an expectation rather than as what the named runs showed (rule 2); the answer in md22 gives the named runs' values.

**Previous blob (2026-10-26 record; history).** The previous blob is `tutorials/pix2struct_ui_captioning_colab.ipynb` `a17c3510018af157673e31d2514e124acf45eb3d` (commit `0383932`, regenerated at `22cab29`; markdown only). The fine-tune timings are now approximate — about 11 minutes on a Colab T4 (`cf909dbc` 2026-10-08: 654.8 s; `820076d3` 2026-10-25: 646.6 s) and about 10 minutes of a 21½-minute wall on the 2026-09-26 Kaggle T4 run of `2e28896c` (612.4 s of 1,292.2 s). No quoted metric differed between the runs. That exact blob completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 13/13 code cells, 1161.8 s wall; fine-tuning 640.0 s; held-out CIDEr-D frozen 1.318 → adapted 1.356, BLEU-4 0.365 → 0.232, epoch 3 kept; reload parity 8/8; evidence in `docs/execution-evidence/2026-10-25-0383932/`) — the same metrics as the runs of `820076d3` and `cf909dbc`. The paths this blob's runs cover: the default path only. The remaining gates are REL12 (a hosted BYOD run) and REL2 (a browser `Run all`). The 2026-10-25 run of `820076d3` described next and every earlier run are history (REL14).

Evidence files for the 2026-10-25 default run of blob `a17c3510`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-0383932/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `pix2struct_ui_captioning_colab_0383932_colab-cli-t4_default.ipynb` | `14aa614f68341818cb3baf6eccdbfdaef01729a6e6a4118be9fa8027cdefb865` | `14aa614f68341818cb3baf6eccdbfdaef01729a6e6a4118be9fa8027cdefb865` (unchanged, no local path) |
| `exec.log` | `7ce2d64a627b8327e6fbb34cc6b881c4bf1668feea99f01da87edccb5f5c0761` | `c95693dee982022236a1822910939454d85fc5b7f19200c55eb973d9883e0916` (2 local-path replacements) |
| `run_summary.json` | `cb97cda2774ed44be280e1d080f8be4537d360fd44a313726174676022bf65a3` | `cb97cda2774ed44be280e1d080f8be4537d360fd44a313726174676022bf65a3` (unchanged, no local path) |

**Prose check (2026-10-25, not fixed in this evidence commit).** The notebook prose was compared number by number with what this run printed. These quoted figures differ from this run's output (the notebook is unchanged here; a later generator change can re-quote them):

- md cell "What changed after fine-tuning" (Section 7 reading guide): "BLEU-4 **down** 0.365 → 0.232 (−0.133, about a third)" — this run printed `delta_vs_frozen` `bleu4` **−0.134** (so did the 2026-10-25 run of `820076d3`); −0.133 is the difference of the two rounded scores, not a printed value.


**Previous blob (history, REL14).** The previous notebook `tutorials/pix2struct_ui_captioning_colab.ipynb` is blob `820076d362813ae7bd780a85728abf4b020a3bc2` (commit `c45a12e`), regenerated from generator commit `2217961` for the tier C review findings of row 65. That exact blob completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 13/13 code cells, 1180.8 s wall; held-out CIDEr-D frozen 1.318 → adapted 1.356, BLEU-4 0.365 → 0.232, epoch 3 kept; reload parity 8/8; evidence in `docs/execution-evidence/2026-10-25-c45a12e/`) — the same metrics as the run of blob `cf909dbc`. The paths this blob's runs cover: the default path only. The remaining gates are REL12 (a hosted BYOD run) and REL2 (a browser `Run all`). Every run recorded for earlier blobs is history (REL14). Changes: T65-1 (REL13) defines the recorded run as the Colab T4 run of blob `cf909dbc` (`c020a97`, 2026-10-08: fine-tune 654.8 s, 0 of 8 reloaded captions differing from frozen) and labels the 2026-09-26 Kaggle run of blob `2e28896c`; T65-2 says that a zero differing-caption count is inconclusive (the recorded run kept epoch 3 and printed 0), so parity there shows a faithful reload, not an applied adapter; T65-3 declares Notebook Specification 2.5 with `notebook_kind: TUTORIAL`, a use notice, IEEE references and the documented `DIMER_ISOLATED_ENV`. Local checks (generator `--check`, `tools/validate_release_assets.py`, ruff, pytest against the `main` baseline) are source checks, not clean-runtime evidence.

Evidence files for the 2026-10-25 default run of blob `820076d3`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-c45a12e/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `pix2struct_ui_captioning_colab_c45a12e_colab-cli-t4_default.ipynb` | `7f3fc6aecc70fa25c28c949732d69d030ffa22a791b170bd662fd996cf6020bb` | `7f3fc6aecc70fa25c28c949732d69d030ffa22a791b170bd662fd996cf6020bb` (unchanged, no local path) |
| `exec.log` | `601c0217cd1188f559ec4adb1d12a51fd28ccda48e9e63f9f6fbe109b52b56ca` | `aee0ea9e14edb685592ff6e2b01f4c4d27e880d03fb19f2d4a053eb4919b1e01` (2 local-path replacements) |
| `run_summary.json` | `0049a465bfe1c07ef7fc0a1ed76a76e83762df1f7cda5c5e06349d967266e9e6` | `0049a465bfe1c07ef7fc0a1ed76a76e83762df1f7cda5c5e06349d967266e9e6` (unchanged, no local path) |

**Prose check (2026-10-25, not fixed in this evidence commit).** The notebook prose was compared number by number with what this run printed. These quoted figures differ from this run's output (the notebook is unchanged here; a later generator change can re-quote them):

- Section 7 worked answer: 'after 654.8 s on a T4' (the run of blob `cf909dbc`, which the notebook defines as the recorded run); this run's fine-tune took 646.6 s. Every metric the worked answers quote (baselines 0.138 / 0.142, frozen CIDEr-D 1.318, validation 0.777 → 0.882 → 0.881 → 0.894 with epoch 3 kept, adapted 1.356, ROUGE-L 0.564 → 0.567, BLEU-4 0.365 → 0.232, per category 2.246 → 2.291 / 0.934 → 0.969, reload 8 of 8 with 0 differing from frozen, 29 tensors, 75,522,608 bytes) matched this run.

**Previous blob — Candidate.** The notebook was regenerated at `c020a97` (blob `cf909dbcd4e5`) and that exact blob passed the Colab CLI Tesla T4
run recorded above (2026-10-08 UTC, 13/13 code cells, 1188.3 s, one pass, no restart, 0 errors, default path only), reproducing
the recorded held-out numbers; promotion to Release-grade is a review decision for the exact release revision. The earlier
Release-grade record identifies blob `2e28896c` (committed at `0edb5de`, generated at `6583937`) only and remains history. The Widget Captioning shard is pinned (95,313,640 bytes, SHA-256
`91d31536466cc5e6f1a15e284d766e80d1de0a94cd90bebe431135e1b51c9cb3`, 1,811 rows over 646 screens; realised default split
330 / 77 / 164 widgets over 75 / 13 / 24 whole apps). Any later change to the carried modules or the notebook yields a new
blob and returns the status to Candidate.

Facts a reviewer should weigh before promotion: the checkpoint was fine-tuned on Widget Captioning's training apps, so
this is continued adaptation inside the task and a small or zero gain is the expected outcome, not a defect; the
fine-tuning recipe (`LEARNING_RATE = 1e-5`, three epochs, two blocks, batches of four widgets with all their references)
has been run once on this checkpoint (the T4 row above): held-out CIDEr-D rose 1.318 → 1.356, ROUGE-L 0.564 → 0.567 and
unigram F1 0.577 → 0.578, while corpus BLEU-4 fell 0.365 → 0.232 — margins on 164 widgets from one seed and one split,
too small to assert, so the notebook keeps recording `adapted_beats_frozen` rather than asserting a gain; the Section 6
assertion that the frozen model beats the constant caption held on this sample (CIDEr-D 1.318 against 0.138); the
snapshot's `is_vqa` header path is disabled at load to match the upstream widget-captioning preprocessing (blue box, no
header), and training uses the same rendering; the pinned sample's screenshots are 1080×1920 or 540×960, so every widget costs a
full 2,048-patch encoder pass in training as well as evaluation and a GPU runtime is recommended; and the ~64-widget
validation and ~160-widget test splits carry no dispersion estimate (the realised splits are 77 and 164 widgets).
