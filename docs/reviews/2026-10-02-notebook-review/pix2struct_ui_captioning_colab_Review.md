# Notebook review: `tutorials/pix2struct_ui_captioning_colab.ipynb`

Notebook Review Framework v1 review against DIMER Notebook Specification 2.2. Review only: nothing in the
repository was changed. Finding prefix: **PSU**.

## 1. Scope and evidence

### Review contract

| Item | Value |
|---|---|
| Repository | `kurtvalcorza/pix2struct-ui-captioning-pipeline` |
| Notebook | `tutorials/pix2struct_ui_captioning_colab.ipynb` |
| Reviewed revision | `origin/main` = `5d8750d4c00ace5b0eb4d8bc458fb13568333ccf` (notebook blob `2e28896c0f4222e890a0ac50d1e21a29e98b24e5`) |
| Generated from | `metadata.dimer.generated_from.revision` `6583937f32dd…`, `tools/build_notebook.py` + `tools/notebook_template.py`. `build_notebook.py --check` reports the notebook up to date |
| Requirements baseline | NOTEBOOK_SPEC **2.2** (ml-worker `origin/main` `b1cfe13`). The notebook declares **2.0** |
| Profile / mode | `E2E` / `GUIDED` (metadata and opening cell) |
| Intended audience | Basic Python and PIL; what an encoder–decoder's generated tokens are; what BLEU-4, ROUGE-L and CIDEr-D measure (Prerequisites) |
| Supported runtime | Colab or Jupyter, Python 3.12. CPU float32 is stated to work; CUDA used when present and recommended for Sections 6–8 |
| Promised outcomes | Pinned install; stage and digest-verify the 8-file snapshot; one digest-pinned Widget Captioning test shard (95 MB) split by whole app into training / validation / test widgets, four refusal probes; inference contract on a drawn five-widget messaging screen with input manifest, rejection probe and keyword observations; constant-caption and colour-neighbour baselines plus the frozen model on four metrics and per category (`small-widget` / `large-widget`); bounded adaptation of the last two decoder blocks with validation-CIDEr-D selection; held-out four-way comparison; drawn screen re-captioned; safetensors adapter export, fresh reload and 8-caption parity; optional BYOD through the same cells |

### Existing execution evidence

`docs/release-verification.md` records one **PASSED** Kaggle Tesla T4 run of blob `2e28896c0f42` — the blob under
review — at `0edb5de` (2026-09-26, 1292.2 s): "11/11 code cells ok (**1 restart after install cell**: the pins replaced
the loaded numpy and cuda-bindings)". Split 330 / 77 / 164 widgets over 75 / 13 / 24 apps. Held-out test (164 widgets):
CIDEr-D constant 0.138, colour-neighbour 0.142, frozen 1.318, adapted 1.356; BLEU-4 0.0 / 0.0 / **0.365 / 0.232**;
ROUGE-L 0.564 → 0.567; unigram F1 0.577 → 0.578; `large-widget` 2.246 → 2.291, `small-widget` 0.934 → 0.969;
validation CIDEr-D by epoch 0.777 / 0.882 / 0.881 / 0.894 (best epoch 3); adaptation 612.4 s; reload parity 8/8. An
earlier attempt on blob `4e74c5a4f655` failed (superseded). No Colab run, no CPU run of the E2E blob, no BYOD run and
no run of an optional experiment are recorded. STATUS.md, README.md and `tutorials/README.md` record
**Release-grade**.

### Journeys and evidence basis

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection of all 25 cells (11 code) | Barriers: PSU-M3, PSU-m1, PSU-m3, PSU-m4, PSU-m6 |
| Clean default | Documented execution evidence (Kaggle T4, exact blob). Not run here: no local GPU by workspace rule, and the full model is 1.13 GB | Completes only after a manual restart (PSU-M1). The comparison has headroom (frozen 1.318 vs constant 0.138; no ceiling). Held-out CIDEr-D +0.038 while BLEU-4 falls 0.133 (PSU-m3). **Colab not verified; CPU path not verified** |
| Active learning | Source inspection, plus direct execution at **reduced scale**: a random 3-decoder-block, 32-wide Pix2Struct built from the committed config (as `tests/test_adaptation_model.py`), CPU, `adapt` called twice as a rerun of Section 7 does | No rerun instructions. A rerun stacks adaptation silently, epoch 0 is still labelled `frozen model`, and the artifact mismatches the in-memory model on 14 tensors (PSU-M2). The learning-rate prediction is unrecorded (PSU-m4). Full scale: not verified |
| Reuse and recovery | Direct execution of the BYOD branch's verbatim unzip/load/split/validate logic with a stubbed upload (no model). Artifact reload: documented evidence (8/8) | A flat 60-record zip is accepted (39 / 9 / 12). Missing `captions` and an undecodable image give actionable messages. A cancelled upload and a zip without a records file give a bare `StopIteration`; a zip with folders is rejected with a misleading path; 30- and 45-record datasets inside the stated 8..5,000 contract fail with a message naming no split (PSU-m2). The documented BYOD rerun silently treats the adapted model as frozen (PSU-M2). BYOD with the real model: **not verified** |

Limitations: no learner observation; no Colab or CPU execution of the full notebook; full-scale numbers come from
the repository record and were not reproduced. Repository checks run here are source checks, not execution
evidence (REL8): `build_notebook.py --check` exit 0; `validate_release_assets.py` PASS (exit 0);
`PYTHONPATH=src pytest -q -o addopts= tests` 60 passed / 2 failed / 7 skipped — both failures are
`ModuleNotFoundError: pyarrow` in the probe environment (an environment artifact, not a finding). Probe
environment: torch 2.13.0+cpu, transformers 4.57.6, Pillow 12.3.0 (not the notebook's pins), `CUDA_VISIBLE_DEVICES=-1`.

### Promise → evidence trace

| Claim | Implementation | Observable result | Learner interpretation |
|---|---|---|---|
| Run all completes in a fresh runtime (opening cell) | Cell 3 pinned in-kernel install plus stale-module guard | Kaggle T4: 1 restart after the install cell | Contradicted (PSU-M1) |
| Pinned, digest-verified snapshot | Cell 11 | Documented: 8 entries, 1,133,308,959 bytes, verified | Delivered |
| Digest-pinned shard, app-disjoint split, 4 refusals | Cell 13 | Documented: 1,811 rows; 330 / 77 / 164 widgets on 75 / 13 / 24 apps, disjoint; probes rejected | Delivered |
| Inference contract on the drawn screen, no score | Cell 15 | Documented: five captions, keywords 4/5, `not-measurable` | Delivered; well explained |
| Baselines and frozen model, four metrics, per category | Cell 17 | Documented: constant 0.138 / neighbour 0.142 / frozen 1.318 CIDEr-D | Delivered |
| Bounded adaptation, validation selection | Cell 19 | Documented: best epoch 3, 18,879,744 trainable | Delivered on the default path; breaks on rerun (PSU-M2) |
| Held-out comparison, gain recorded not asserted | Cell 21 | Documented: CIDEr-D +0.038, BLEU-4 −0.133, `adapted_beats_frozen` true | Delivered; the mixed result is not interpreted (PSU-m3) |
| Drawn screen re-captioned, export, fresh reload, parity | Cell 23 | Documented: 2 of 5 captions changed; 29 tensors; 8/8 | Delivered |
| "Optional experiments" (learning rate, 1 block, loosen a box, BYOD) | Interpretation cell prose only | No code; rerun stacks (PSU-M2); LR outcome unrecorded (PSU-m4) | Not delivered as an activity |
| BYOD after the default completes, "re-run from that cell" | Cell 13 `USE_BYOD` branch | Flat zip reaches the split (probe); then Sections 5–7 score the adapted model as frozen | Partly delivered (PSU-M2, PSU-m2) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| Install, stage and verify identities | Printed dicts, Sections 1 and 3 | Shown; no question asks the learner to read them (PSU-M3) |
| Download, validate, split by app without leakage | Section 4 manifests and refusal probes | Shown; no activity (PSU-M3) |
| Read `caption`, `box`, `new_tokens`, `truncated` correctly | Section 5 prints and prose | Explained well; no check-your-reading prompt |
| Score frozen model beside baselines, read per category | Section 6 | Exercised on the default path; explained |
| Run bounded fine-tuning with explicit hyperparameters | Section 7 form fields | Exercised once; changing a value and rerunning gives invalid results (PSU-M2) |
| Evaluate on an app-disjoint test split | Section 8 | Exercised; reading order explained, BLEU-4 drop not discussed (PSU-m3) |
| Re-caption a different image family | Section 9 | Exercised (five widgets, labelled as observation) |
| Export and reload with verified parity | Section 9 | Exercised (documented 8/8) |

## 2. Separate judgments

- **Technical correctness:** the default path is carefully engineered: immutable revisions for model and dataset,
  per-file and whole-shard digests, `trust_remote_code=False`, the VQA header switched off so no font is fetched, a
  transactional `adapt`, and manifest, digest and exact-tensor-set checks before an adapter loads. Two defects: the
  documented install restart (a `MUST` failure), and `adapt` training the shared `pipe` in place from its current
  weights, so every documented rerun (BYOD, optional experiments) silently corrupts the frozen-vs-adapted comparison
  and can make the exported artifact misdescribe the in-memory model.
- **Promise fulfilment:** identity, corpus, split, inference contract, baselines, adaptation, comparison, export and
  reload are met by documented evidence on the default path. "Run all completes" is not met. The corpus leaves
  headroom (frozen 1.318 vs constant 0.138), so unlike the sibling `pix2struct-docvqa` row the central demonstration
  is valid. The optional experiments and the BYOD rerun are not completable as written.
- **Learner experience:** the prose is unusually careful about what a caption is not, the box as part of the input,
  app-level leakage and what the result does not establish. But the GUIDED learner meets about 74,000 characters of
  unlabelled carried code before any model step; there is no How-to-use section, roadmap, glossary, troubleshooting,
  prediction, checkpoint, coded activity or conclusion scaffold; template braces leak into two markdown cells; the
  recorded 36 % BLEU-4 drop beside a 3 % CIDEr-D gain is left to the learner; and two numbers in the prose (validation
  "about sixty", the learning-rate outcome) are not backed by the run.
- **Spec conformance:** fails the `MUST`s RUN1, RUN10 and ENV6 (restart) and DAT13 on the documented BYOD rerun (the
  "frozen" baseline is not frozen); DAT19 for the cancelled-upload, missing-records-file, nested-path and small-split
  BYOD cases; SRC3 for the knowingly wrong id pattern in the data contract. Declares spec 2.0, not 2.2. GDL1–GDL14,
  EXE2 and EXE5 `SHOULD` deviations are not recorded as deviations. Other applicable `MUST`s checked by source
  inspection appear met on the default path: ST, MOD, DAT1–DAT9, VAL, SPL, FT, EVAL1–EVAL7, EVAL14, ART and VER1–VER3.

## 3. Findings

### Major

**PSU-M1 — Section 1 install cell: the default `Run all` requires a manual restart.**
Cell 3 `pip install`s exact pins (`torch==2.14.0`, `numpy==2.5.3`, `transformers==4.57.6`, …) into the running
kernel and raises "Restart the runtime, then rerun from the top" if a pinned distribution was already imported. The
only clean run of this blob hit it ("1 restart after install cell: the pins replaced the loaded numpy and
cuda-bindings"). `docs/release-verification.md` step 4 calls the restart "expected", and the repository records
Release-grade.
- *Consequence:* a learner choosing Run all on a fresh hosted runtime stops at cell 3; the Release-grade label
  overstates the evidence for a one-pass Run all.
- *Evidence:* documented execution evidence (release-verification.md row 2026-09-26; STATUS.md; README.md);
  source inspection of cell 3 and `tools/build_notebook.py:61–69`. Colab: not verified.
- *Recommended correction:* adopt the fleet's uv isolated-environment pattern: the setup cell bootstraps uv, creates
  an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked
  `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs
  the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can
  be required. Reference implementations on `main`:
  `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and
  `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. Do not add
  another in-kernel install guard or loosen pins to dodge the restart. Implement it in `tools/build_notebook.py`,
  regenerate, re-qualify with a one-pass hosted Run all, and correct release-verification.md step 4 so a
  restart-dependent run is not reported as a `Run all` PASS.
- *Acceptance check:* a fresh Colab (or Kaggle) runtime executes every code cell in order in one kernel with no
  restart and no error output, recorded against the new blob; release-verification.md no longer accepts a restart.
- *Spec:* RUN1, RUN10, ENV6 (MUST); REL11.

**PSU-M2 — Sections 3–9: no rerun instructions; the documented BYOD rerun and every optional experiment silently
stack adaptation onto the already-adapted model.**
`pipe` is built only in cell 11 (Section 3) and never rebuilt or deleted (static probe: `pipe_built_only_in_cell:
[11]`, `del_pipe_anywhere: false`). `pipe.adapt` (carried `pipeline.py`, `adapt`, line 574) trains in place from the
pipeline's *current* weights and labels epoch 0 `"frozen model"` (line 637).
- The BYOD text says "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and **re-run from that
  cell**". Doing so runs Section 5 ("frozen" screen captions), Section 6 (`frozen_test`, the "frozen" per-category row)
  and Section 7 (epoch 0 "frozen model") on the Widget-Captioning-adapted model, then trains it further. Section 8's
  `delta_vs_frozen` and `adapted_beats_frozen` then compare two adapted models. Nothing errors.
- "Optional experiments": "raise `LEARNING_RATE` …; set `TRAINABLE_DECODER_LAYERS = 1` and compare the artifact size
  and the held-out score; loosen a box on the drawn screen …" have no code and no rerun route. Editing Section 7 and
  rerunning it continues from the adapted weights; with one block, the first run's other block stays modified in
  memory but is not exported, so the artifact no longer reproduces the in-memory model. Loosening a box after
  Section 7 and rerunning Section 5 captions with the adapted model under the label "frozen".
- *Consequence:* the learner's frozen-vs-adapted, learning-rate and 1-vs-2-block conclusions are wrong without any
  signal, and the exported adapter misdescribes its lineage. Framework dimensions 3, 7 and 8. Same defect as the
  sibling rows (PSA-M2, PSD-M3, PST-M2); here, as in textcaps, the BYOD route is silent (no `del pipe`).
- *Evidence:* static probe as above plus `byod_rerun_text`. Direct execution at **reduced scale**
  (`stacked_adaptation`): after `adapt(layers=2)` then `adapt(layers=1)`, run 2 did not start from base
  (`run2_started_from_base: false`), reported epoch 0 as `"frozen model"`, all 14 `decoder.layer.1.*` tensors (the
  first-run-only block) differ from base, and the artifact reloaded onto a fresh base mismatches the in-memory model on
  14 tensors. Rebuilding the pipeline, as Section 3 does, restores base (`rebuild_pipeline_restores_base: true`).
- *Recommended correction:* make every experiment start from the verified base and say exactly what to rerun. Either
  rebuild `pipe` with `Pix2StructWidgetCaptioningPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)` at the top of
  Section 5 (no download; snapshot already verified), or snapshot the adaptable tensors after Section 3 and restore
  them before Sections 5 and 7 (refusing to run Section 6 when `pipe.adapter is not None`). Change the BYOD text
  (`tools/notebook_template.py:40`) and the optional-experiments text (`:526`) to "set the value, then run from
  Section N to the end".
- *Acceptance check:* after a completed default run, following the written BYOD instruction gives a Section 6
  `frozen_test` with `adapted: False` and Section 7 epoch-0 validation equal to the frozen score; following the
  written 1-block instruction leaves every non-exported decoder tensor equal to base. A reduced-scale probe like
  `stacked_adaptation` reports 0 mismatched tensors.
- *Spec:* DAT13 (MUST), RUN9, UX7, GDL10, VER3–VER5.

**PSU-M3 — Whole notebook: the declared `GUIDED` layer is largely missing, and about 74,000 characters of carried
code are unlabelled.**
Static probe: no "How to use this notebook", roadmap, glossary, troubleshooting, collapsible answers, "what to
notice" guidance or conclusion scaffold (`predict_prompt: true` is a false positive: the matches are the code calls
`pipe.predict`). 0 `cellView: form` cells, no "Infrastructure" label. The three carried module cells hold 41,833,
12,602 and 19,651 characters and sit between Section 1 and the first model step. No code cell is a learner activity;
the "Optional experiments" are one prose paragraph.
- *Consequence:* the stated learner can follow the default path but is never asked to apply the objectives (predict
  where icons vs larger widgets stand, read the per-category split, explain a caption error such as the gear icon).
  The infrastructure dominates the scroll before any learning content.
- *Evidence:* source inspection; static probe `guided_markers`, `embedded_module_cell_chars`,
  `optional_experiments_have_code: false`.
- *Recommended correction:* in `tools/notebook_template.py`, add a How-to-use cell and roadmap after the opening,
  a short glossary (patch, teacher forcing, CIDEr-D document frequency, corpus medoid, app-disjoint split), label the
  three module cells Infrastructure with `cellView: form` and a one-line "what to look for", add one Predict → Change
  → Run → Observe → Explain activity (e.g. predict which category gains from adaptation before Section 8, or loosen a
  box on the drawn screen) with a collapsible answer, checkpoints after Sections 4, 6 and 8, a troubleshooting table
  (restart, CUDA absent, upload cancelled), and a conclusion scaffold.
- *Acceptance check:* a static probe finds How-to-use, roadmap, glossary, troubleshooting, at least one prediction
  prompt with a collapsible answer, Infrastructure labels on every carried module cell, and a conclusion scaffold.
- *Spec:* GDL1–GDL14, UX5, UX8 (SHOULD); spec 2.2 notes existing notebooks are not nonconformant solely for lacking
  them, so this is a learner-impact Major, not a `MUST` failure.

### Minor

**PSU-m1 — Opening cell and Prerequisites: template brace escaping leaks `{{…}}` into learner markdown, including a
wrong id pattern.** The rendered text shows `{{id, image, box, captions}}` twice and the id pattern
`[A-Za-z0-9_.:-]{{1,64}}`; the code's pattern is `{1,64}`. The markdown strings at `tools/notebook_template.py:41`
and `:130` keep doubled braces that are never substituted.
- *Consequence:* a BYOD user copying the pattern gets a regex that matches `{1,64}` literally; the record shape looks
  like a templating error.
- *Evidence:* static probe `double_brace_in_markdown` (cells 0 and 1, 3 occurrences).
- *Recommended correction:* single braces in those markdown strings (or format them), then regenerate.
- *Acceptance check:* no `{{` or `}}` in any rendered markdown cell.
- *Spec:* SRC3, UX2.

**PSU-m2 — Section 4 BYOD: four failure cases are not actionable.**
Probe replay of the verbatim branch: a cancelled upload and a zip with no `records.jsonl`/`records.json` raise a bare
`StopIteration`; a zip whose records reference `images/img0.png` (the natural layout) is flattened by basename and
then rejected with "image file not found: …/byod/images/img0.png", although the file was in the zip; datasets of 30
and 45 records — inside the stated "8..5,000 records" — fail with "6 records; 8..5000 are required" / "7 records; …"
because each split is re-validated with the default `min_records=8`, and the message names no split. The effective
minimum is about 50 records when every widget is its own screen (probe). There is no location field (upload dialog
only), and the Section 6 `assert frozen_test['cider_d'] > baseline_constant['cider_d']` gives a bare `AssertionError`
if a BYOD corpus defeats it. Missing `captions` and an undecodable image are reported clearly.
- *Consequence:* a learner with valid-looking data gets an unexplained error and no corrective action.
- *Evidence:* direct execution (`byod_replay`), source inspection of cell 13 and `samples.py`
  `split_dataset` / `validate_dataset`.
- *Recommended correction:* in the template's BYOD branch, check the upload and the records file with messages naming
  the expected file; preserve relative paths when extracting (or resolve by basename consistently); validate splits
  with the split name and state the effective minimum in the Prerequisites; add a `BYOD_ZIP_PATH` form field; give the
  Section 6 assert a message.
- *Acceptance check:* each probe case ends in a `ValueError` naming the failed condition and the fix; a 60-record
  nested-folder zip is accepted.
- *Spec:* DAT19 (MUST), UX10, EXE1, EXE2.

**PSU-m3 — Section 8 and Interpretation: the recorded held-out result is mixed, and the notebook leaves it
uninterpreted.** The recorded run gives CIDEr-D 1.318 → 1.356 (+0.038, +3 %) but BLEU-4 0.365 → 0.232 (−0.133,
−36 %), with ROUGE-L and unigram F1 essentially flat (+0.003, +0.001). Section 8 says only that BLEU-4 and ROUGE-L
"can move the other way when the adapted captions change length"; the Interpretation cell concludes "a gain here says
the contract works", and `adapted_beats_frozen` is computed from CIDEr-D alone. The length explanation is not checked
(`mean_words` is printed but the record does not report it, and nothing tells the learner to compare it).
- *Consequence:* a learner reads `adapted_beats_frozen: True` and "a gain" and concludes the adaptation helped, when
  one of the four reported metrics fell by a third and the one that rose was also the selection metric. Framework
  dimensions 3 and 5.
- *Evidence:* documented execution evidence (release-verification.md 2026-09-26 row; STATUS.md also records "BLEU-4
  fell") against cell 20 and cell 24 text (`tools/notebook_template.py` Section 8 and Interpretation strings).
- *Recommended correction:* in Section 8 print `mean_words` beside `delta_vs_frozen` and add a sentence telling the
  learner what a large BLEU-4 drop with a small CIDEr-D gain means (shorter or reworded captions, higher-order n-gram
  matches lost, CIDEr-D chosen as the selection metric); in the Interpretation, state that the recorded run was mixed
  rather than "a gain".
- *Acceptance check:* the notebook text names the direction of every metric in the recorded run and tells the learner
  how to check the length explanation against `mean_words`.
- *Spec:* EVAL15 (related; four metrics are reported, so not a `MUST` failure), EVAL14, UX2.

**PSU-m4 — Sections 7 and Interpretation: prose numbers and predictions not backed by the run.** Section 7 says "a
validation split of about sixty widgets" and the Interpretation "the validation split that picks the epoch is about
60"; the recorded default split is **77** widgets. The optional experiment predicts "raise `LEARNING_RATE` and watch
the training loss fall while the validation CIDEr-D drops and the selector keeps an early epoch", and the
Interpretation says a too-high rate "overfits this little data within an epoch, which the validation-based selector
reports by keeping epoch 0". No run at any learning rate other than 1e-5 is recorded in this repository (the only
record shows validation CIDEr-D rising 0.777 → 0.894 to the last epoch). The sibling `pix2struct-textcaps` row's
recorded sweep contradicted the same prediction (PST-m3).
- *Consequence:* a learner who tries the natural values (2e-5, 5e-5) has no expected range; if validation rises they
  may conclude the notebook is broken.
- *Evidence:* source inspection (static probe `validation_size_claims`, `lr_experiment_claim`, `lr_overfit_claim`)
  against the recorded split and epoch history; `tools/notebook_template.py:506`, `:510`, `:526`.
- *Recommended correction:* print the split sizes the prose relies on instead of hard-coding them (or say 77), and
  either measure and state a learning rate at which the selector keeps an early epoch, or drop the prediction.
- *Acceptance check:* every number and predicted experiment outcome in the prose matches a recorded run.
- *Spec:* GDL10, UX2.

**PSU-m5 — Metadata and opening: declares NOTEBOOK_SPEC 2.0 rather than the current 2.2.**
- *Evidence:* `metadata.dimer.notebook_spec: "2.0"`, opening cell, `NOTEBOOK_SOURCE`, References.
- *Recommended correction:* re-baseline against 2.2 in `tools/build_notebook.py` and record any `SHOULD` deviations.
- *Acceptance check:* metadata, opening cell and `tutorials/README.md` declare 2.2.
- *Spec:* §3.4, §32.

**PSU-m6 — Opening and Prerequisites: no runtime estimate, and the CPU path is called workable without one.**
The notebook says "the CPU path works but is slow" and points to `docs/release-verification.md` for timings; no
markdown cell states a duration (static probe). Inferred CPU cost: about 1,070 greedy caption calls (5 + 328 in
Section 6 + 308 validation scorings over epochs 0–3 + 405 in Section 8 + 21 in Section 9) at the recorded 2.0–2.6 s
per call, i.e. 35–45 minutes of captioning alone, plus three epochs over 893 (widget, reference) pairs that took
612 s on a T4 — likely well over an hour on CPU, not verified. `tutorials/README.md` gives "~22 min on a Kaggle
Tesla T4" but the notebook does not.
- *Consequence:* a learner on a CPU runtime cannot plan, and may abandon a run that is working.
- *Evidence:* source inspection; static probe `runtime_estimate_in_notebook_markdown: false`.
- *Recommended correction:* state the recorded T4 wall time (labelled with its environment) and a labelled CPU
  estimate, or recommend GPU as required for Sections 6–8.
- *Acceptance check:* the opening cell states a labelled runtime estimate per supported runtime.
- *Spec:* RUN12, UX12, GDL1.

**PSU-m7 — Section 1: `DIMER_NOTEBOOK_CI_PREINSTALLED` is read but undocumented.** Setting it skips the pinned
install (`tools/build_notebook.py:470`).
- *Evidence:* static probe (`env_var_…_in_code: true`, `env_var_documented_in_markdown: false`).
- *Recommended correction:* one sentence in the Section 1 markdown.
- *Acceptance check:* the variable is named and explained in markdown.
- *Spec:* EXE5.

### Suggestions

- **PSU-S1** — Print the `adapted` flag that `pipe.evaluate` already returns (`pipeline.py:524`) in Sections 6 and 8,
  so a stale-state run (PSU-M2) is visible. Spec: none.
- **PSU-S2** — Section 6 captions the 164 test widgets twice (`pipe.evaluate`, then `pipe.predict` for the
  per-category table) and Section 8 does the same, plus re-evaluates the 77 validation widgets already scored at the
  best epoch inside `adapt` (static probe `test_photos_captioned_twice`): about 405 redundant generate calls, roughly
  15 minutes on CPU (inferred). Return per-record predictions from `evaluate` and reuse them. Spec: RUN12.
- **PSU-S3** — The held-out gain is +0.038 CIDEr-D on 164 widgets with one seed. A paired bootstrap interval over the
  test widgets (overall and per category, and for BLEU-4) would give "recorded, not asserted" a number the learner
  can read. Spec: EVAL15 (related).
- **PSU-S4** — Provide the optional experiments (learning rate, one block, loosen a box) as optional code cells (after
  the PSU-M2 fix), each following Predict → Change → Run → Observe → Explain. Spec: GDL10, UX5, UX9.
- **PSU-S5** — Add DOI/APA references for Lee et al. 2023, Li et al. 2020, Deka et al. 2017 (DOI present), Papineni
  et al. 2002, Lin 2004 and Vedantam et al. 2015 beside the arXiv/ACL links. Spec: UX2, SRC10.
- **PSU-S6** — In the Section 9 parity check, also report how many of the 8 reloaded captions differ from
  `frozen_predictions[:8]`, so parity demonstrably shows the adapter applied rather than the base reproduced. Not
  verified whether the current 8 differ. Spec: VER4, VER5.

## 4. Readiness

**Needs revision.** Three Majors are open (PSU-M1 restart, PSU-M2 silent stacked adaptation on the documented reruns,
PSU-M3 missing GUIDED layer), and the `MUST`s RUN1/RUN10/ENV6, DAT13 and DAT19 are unresolved. Execution evidence
for the default path exists (Kaggle T4, exact blob, one run) but records the restart. Remaining gates after fixes: a
one-pass hosted Run all on the new blob (Colab preferred), a recorded BYOD run, and a rerun of one optional
experiment through the written instructions.

Sibling consistency (pix2struct-ai2d PSA, -docvqa PSD, -textcaps PST): PSU-M1 = PSA-M1 / PSD-M1 / PST-M1; PSU-M2 =
PSA-M2 / PSD-M3 / PST-M2 (silent here, as in textcaps); PSU-M3 = PSA-M3 / PSD-M4 / PST-M3; PSU-m1 = PSA-m1 / PST-m1
(brace leak recurs); PSU-m2 = PSA-m2 / PSD-m2 / PST-m2 (BYOD wording recurs); PSU-m5–m7 = PST-m4–m6. The docvqa
ceiling finding (PSD-M2) does **not** recur: frozen test CIDEr-D 1.318 sits far above the 0.138 constant baseline,
and validation rose 0.777 → 0.894. New in this row: PSU-m3 (mixed held-out result uninterpreted, BLEU-4 −36 %).
PSU-m4 is the textcaps learning-rate finding (PST-m3) without a contradicting record here, so it is "unbacked"
rather than "contradicted".

## 5. Verified versus inferred

- **Verified by direct execution (CPU, reduced scale or no model):** BYOD branch outcomes (8 cases); stacked
  adaptation semantics on a 3-block random Pix2Struct (14 mismatched tensors); static notebook properties; generator
  parity, release validator and the offline test suite (60 passed; 2 failures are the probe env's missing pyarrow).
- **From documented evidence only:** every full-scale number, the restart, reload parity 8/8.
- **Inferred:** the CPU runtime estimate (PSU-m6) and redundant-call cost (PSU-S2); that the BYOD rerun reproduces the
  stacking at full scale (same code path as the probe, not run with the real checkpoint).
- **Most likely to be wrong:** PSU-M3's severity — the guided layer is `SHOULD`, and spec 2.2 says existing notebooks
  are not nonconformant solely for lacking it; given how well the prose explains the results, it could fairly be
  Minor. It is kept Major for consistency with the sibling rows.

Probe bundle: `pix2struct_ui_captioning_colab_Review_Probes.zip` (`run_probes.py`, `results.json`,
`source_manifest.json`).
