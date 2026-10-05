# pix2struct_ui_captioning_colab — fleet-sweep fixes (2026-10-05)

Targeted fix of the 2026-10-05 fleet sweep findings. There is no full Notebook Review Framework v1 report for this
notebook; each flag was first confirmed in the cell source on `main` (`5d8750d`). All changes are made in the
generator (`tools/build_notebook.py`, `tools/notebook_template*.py`) and the notebook is regenerated. STATUS and the
release labels are unchanged. **Readiness: Verification pending** (hosted Run all not yet done).

## Findings and fixes

| ID | Status | Change | Cells / files touched | Evidence |
|---|---|---|---|---|
| SWP-R (restart guard) | Fixed — hosted confirmation pending | Confirmed (the recorded Kaggle T4 run of 2026-09-26 needed a restart after the install cell): Section 1 ran `pip install` into the kernel and raised "Restart the runtime" on stale modules. Generator upgraded to `build_notebook.py/2.2` (the fleet isolated runtime): one kernel cell downloads the pinned `uv` 0.12.15 wheel (size + SHA-256), builds a managed CPython 3.12.12 environment from the new hash lock `tutorials/requirements-colab.lock.txt` (`--require-hashes --only-binary :all:`), and routes every later cell to one persistent worker. The environment folder is keyed on the lock digest and reused by a re-run or a second Run all; re-running Section 1 keeps the live worker and its variables; the worker gets `MPLBACKEND=Agg` and no `PYTHONPATH`/`PYTHONHOME`/`PYTHONSTARTUP`. | Section 1 (kernel cell + "Record the runtime"); `tools/build_notebook.py`; `tutorials/requirements-colab.lock.txt`; `tools/validate_release_assets.py` (install markers, bootstrap check, kernel cell excluded from the library-use scan); `docs/release-verification.md` (the line describing that check); the release-verification step that expected an interpreter restart | `test_swp_r_no_pip_install_or_restart_in_any_cell`, `test_swp_r_lock_is_carried_hash_locked_and_matches_pins`, `test_swp_r_environment_keyed_on_lock_and_child_env_cleaned`, `test_swp_r_section1_reuses_environment_and_worker_when_rerun` (executes the notebook's own kernel cell with a stand-in IPython shell; the worker runs on the test interpreter) |
| SWP-G (guided layer) | Fixed | Confirmed: GUIDED mode with 0 of 9 guided markers. Added audience and Input → Model → Output table, How to use this notebook, roadmap, a Learner prerequisite, five Predict / Check your reasoning pairs (Sections 4, 6–9) quoting the recorded Kaggle T4 run (CIDEr-D constant 0.138 / neighbour 0.142 / frozen 1.318 / adapted 1.356; BLEU-4 0.365 → 0.232; large 2.246 → 2.291, small 0.934 → 0.969; validation 0.777 → 0.894, epoch 3 kept; parity 8/8), Troubleshooting, Glossary and a Conclusion template. Sections 1–3 labelled Infrastructure and collapsed. Literal doubled braces in the BYOD declaration and the data-contract prerequisite (neither is passed through `str.format`) now render as single braces. | Template opening, BYOD declaration, prerequisites, Sections 4 and 6–9, closing | `test_swp_g_guided_layer_present`, `test_swp_g_infrastructure_cells_labelled_and_collapsed`, `test_swp_g_no_leftover_placeholders` |
| SWP-A (quality asserts) | Fixed | Section 6 ended with `assert frozen_test['cider_d'] > baseline_constant['cider_d']`, which aborted a run (e.g. BYOD) before adaptation, export and reload. It is now the recorded verdict `frozen_beats_constant` (printed and written to the evaluation report); the reload-parity assert stays (contract). | Section 6, evaluation report; validator marker; `docs/release-verification.md` | `test_swp_a_frozen_vs_constant_is_a_verdict` |
| SWP-F (frozen re-run) | Fixed | Confirmed: `pipe.adapt` keeps the trained decoder blocks inside `pipe` (restored only on an exception), so a re-run of Section 6 scored the adapted model as frozen and a re-run of Section 7 (the closing's experiments) continued from it with epoch 0 labelled frozen. Sections 6 and 7 now reload the frozen pipeline from the pinned, digest-verified snapshot through `from_pretrained` when `pipe.adapter` is set; the default first pass loads nothing extra. | Sections 6 and 7 | `test_swp_f_adapted_pipeline_is_reloaded_from_the_pinned_snapshot` (both cells) |
| SWP-B (BYOD) | Fixed | BYOD worked only through `files.upload()` (an empty upload raised `StopIteration`), and a zip without `records.jsonl` raised another bare `StopIteration`. Added a `BYOD_PATH` form field with the Colab upload as a guarded fallback, and a named refusal for a zip without a records file. | Section 4, BYOD declaration | `test_swp_b_byod_path_reads_file_and_refuses_with_names`, `test_swp_b_cancelled_colab_upload_gives_a_clear_message`, `test_swp_b_byod_path_fields_default_off`, `test_swp_b_zip_without_records_file_is_named` |

## User-visible changes

- Section 1 no longer installs into the notebook's Python and never asks for a restart; it builds (first run) or reuses `dimer_isolated_env_<lock digest>/` and every later code cell runs there. Linux x86_64 runtimes only.
- Section 6 records `frozen_beats_constant` instead of stopping.
- Sections 6 and 7 reload the frozen pipeline when `pipe` was already adapted (re-runs only).
- New `BYOD_PATH` form field in Section 4; on Colab an empty path still opens the upload dialog.
- Guided material added; Sections 1–3 collapsed.

## Verification (offline; not clean-runtime evidence)

- Real input: none of the model stages could run here (the Hugging Face Hub is unreachable and torch is not installed).
- Stand-ins: the kernel-cell test runs the generated bootstrap against a pre-built environment folder whose `python` is the test interpreter (routing, reuse and idempotence are real; the managed CPython and locked packages are stand-ins); the SWP-F test executes the reload block with a stand-in pipeline class.
- `python tools/build_notebook.py --check`: OK. `python tools/validate_release_assets.py`: PASS. `ruff check src tests tools`: clean.
- `pytest` with CI's dependencies except torch (CI installs CPU torch; torch-gated tests skip here): 56 passed, 1 skipped before → 72 passed, 1 skipped after.
- Every code cell of the regenerated notebook parses.

## Remaining gates

- A hosted **Run all in one pass** on a fresh runtime (expected: no restart prompt; Section 1 builds the environment; a second Run all reports `'reused': True`).
- The REL12 BYOD run with the BYOD gates and path fields set.
- A full Notebook Review Framework v1 review has not been done.
