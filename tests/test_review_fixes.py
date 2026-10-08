"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings (prefix PSU) fixed in the generator on
top of the fleet-sweep fixes. They need only CI's dependencies: the Section 4 BYOD branch is executed with the
notebook's own source and stand-in inputs (synthetic PNGs, no model), and the rest are static checks on the
generated notebook."""
# ruff: noqa: E501  -- assertion messages and notebook source fragments are kept on single lines

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import re
import shutil
import textwrap
import zipfile
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook_review", TOOLS / "build_notebook.py")
TEMPLATE = _load("notebook_template_review", TOOLS / "notebook_template.py").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]


@pytest.fixture(scope="module")
def nb() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _src(cell: dict) -> str:
    s = cell["source"]
    return "".join(s) if isinstance(s, list) else s


def _code_cells(nb: dict) -> list[dict]:
    return [c for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell_with(nb: dict, marker: str) -> str:
    found = [_src(c) for c in _code_cells(nb) if marker in _src(c)]
    assert len(found) == 1, f"exactly one code cell must contain {marker!r}"
    return found[0]


# --- PSU-m4: spec 2.2 everywhere -------------------------------------------------------------------------------------


def test_psu_m5_spec_2_2_declared_everywhere(nb: dict) -> None:
    """PSU-m5: metadata, opening cell, registry, validator and generator agree on NOTEBOOK_SPEC 2.2 (§32 item 6)."""
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert build.NOTEBOOK_SPEC == "2.2"
    validator = _load("validate_release_assets_review", TOOLS / "validate_release_assets.py")
    assert validator.NOTEBOOK_SPEC == "2.2"
    assert "DIMER Notebook Specification 2.2 — **standalone** (§4)" in _src(nb["cells"][0])
    assert "DIMER Notebook Specification 2.2" in (ROOT / "tutorials" / "README.md").read_text(encoding="utf-8")
    text = _markdown(nb) + "\n".join(_src(c) for c in _code_cells(nb))
    assert "Specification 2.0" not in text and "NOTEBOOK_SPEC 2.0" not in text


# --- PSU-M3: Infrastructure titles, guided markers, a coded default-off experiment -----------------------------------


def test_psu_m3_every_carried_and_install_cell_is_titled_infrastructure(nb: dict) -> None:
    """PSU-M3 acceptance: every carried-module, install and snapshot cell starts with `# @title Infrastructure`, is
    collapsed, and the carried text after the title line still equals the module (PAR1 via embedded_module_text)."""
    titled = 0
    for cell in _code_cells(nb):
        s = _src(cell)
        if "# dimer: kernel cell" in s or cell["metadata"].get("dimer", {}).get("embedded_module") or "MANIFEST = {" in s:
            assert s.startswith("# @title Infrastructure"), s[:80]
            assert cell["metadata"].get("cellView") == "form"
            titled += 1
    assert titled >= 5
    ctx = build.load_context(ROOT, TEMPLATE, nb["metadata"]["dimer"]["generated_from"]["revision"])
    for cell, module in zip([c for c in _code_cells(nb) if c["metadata"].get("dimer", {}).get("embedded_module")], ctx["modules"], strict=True):
        assert _src(cell).startswith(build.EMBEDDED_TITLE_PREFIX)
        assert build.embedded_module_text(_src(cell)).rstrip("\n") + "\n" == ctx["embedded"][module]
    assert build.embedded_module_text("x = 1\n") == "x = 1\n"


def test_psu_m3_guided_markers_all_present(nb: dict) -> None:
    """PSU-M3 acceptance: the review's static probe markers are all true (the prediction prompt is a real markdown
    prompt, not the `def predict` false positive) and at least four cells are form-collapsed."""
    text = "\n".join(_src(c) for c in nb["cells"])
    markers = {
        "how_to_use": r"how to use this notebook", "roadmap": r"roadmap", "glossary": r"glossary", "troubleshooting": r"troubleshoot",
        "infrastructure_label": r"infrastructure", "check_your_reasoning": r"check your reasoning|<details",
        "what_to_notice": r"what to notice|expected result|look for", "conclusion_template": r"conclusion template|your conclusion|write.*conclusion",
    }
    missing = [k for k, p in markers.items() if not re.search(p, text, re.I)]
    assert not missing, missing
    assert _markdown(nb).count("**Predict before running:**") >= 5
    assert sum(c.get("metadata", {}).get("cellView") == "form" for c in nb["cells"]) >= 4


def test_psu_m3_coded_experiment_cell_is_off_by_default_and_trains_nothing(nb: dict) -> None:
    """PSU-M3 acceptance / PSU-S4: a coded Predict → change → run → observe experiment (loosen / shift the box) exists,
    is gated off by default, only prints `skipped` under Run all, and trains and writes nothing when on."""
    source = _cell_with(nb, "RUN_BOX_EXPERIMENT = False  # @param")
    assert ".adapt(" not in source and "save_artifact" not in source and "open(" not in source
    assert "'loosened'" in source and "'shifted'" in source and "MIN_BOX_SIDE" in source
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(source, "<experiment>", "exec"), {})
    assert "skipped" in out.getvalue()


# --- PSU-M2: scored weights printed; rerun routes named ---------------------------------------------------------------


def test_psu_m2_adapted_flag_is_printed_and_rerun_instructions_name_cells(nb: dict) -> None:
    """PSU-M2 / PSU-S1: Section 6 prints `frozen_test['adapted']` and refuses adapted scores after the SWP-F reload;
    Section 8 prints which weights were scored; the BYOD and optional-experiment texts name the cells to re-run."""
    six = _cell_with(nb, "frozen_test = pipe.evaluate(test_records")
    assert "'adapted': frozen_test['adapted']" in six and "if frozen_test['adapted']:" in six
    assert six.index("if pipe.adapter is not None:") < six.index("frozen_test = pipe.evaluate")
    eight = _cell_with(nb, "adapted_test = pipe.evaluate(test_records")
    assert "'adapted_test_adapted': adapted_test['adapted']" in eight
    md = _markdown(nb)
    assert re.search(r"re-?run (from )?(section|cell)", md, re.I)
    assert "re-run Section 4 and every code cell of Sections 5–9 in order" in md
    assert "re-run the Section 7, 8 and 9 cells" in md and "No run at a rate other than 1e-5 is recorded" in md


# --- PSU-m3 / PSU-m4: the mixed recorded result is interpreted; prose numbers match the run ------------------------


def test_psu_m3_mixed_result_is_named_and_reported(nb: dict) -> None:
    """PSU-m3: Section 8 names the direction of every metric in the recorded run, explains a BLEU-4 drop beside a
    CIDEr-D gain, prints `metric_directions` and `length_change`, and the Interpretation calls the run mixed."""
    md = _markdown(nb)
    assert "BLEU-4 **down** 0.365 → 0.232" in md and "CIDEr-D 1.318 → 1.356" in md and "the recorded run was **mixed**" in md
    eight = _cell_with(nb, "adapted_test = pipe.evaluate(test_records")
    assert "metric_directions = {" in eight and "length_change = {" in eight and "'metric_directions': metric_directions" in eight
    block = "adapted_beats_frozen = " + eight.split("adapted_beats_frozen = ", 1)[1].split("evaluation_report_payload", 1)[0]
    ns = {
        "adapted_test": {"cider_d": 1.356, "adapted": True}, "frozen_test": {"cider_d": 1.318, "adapted": False}, "METRICS": ("bleu4", "rouge_l", "cider_d", "unigram_f1"),
        "comparison": {"delta_vs_frozen": {"bleu4": -0.133, "rouge_l": 0.003, "cider_d": 0.038, "unigram_f1": 0.0}, "mean_words": {"frozen": 2.1, "adapted": 2.9}},
    }
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        exec(compile(block, "<eight>", "exec"), ns)
    assert ns["metric_directions"] == {"bleu4": "down", "rouge_l": "up", "cider_d": "up", "unigram_f1": "flat"}
    assert ns["length_change"]["delta"] == 0.8 and "Mixed result: CIDEr-D rose while bleu4 fell" in out.getvalue()


def test_psu_m4_prose_numbers_match_the_recorded_run(nb: dict) -> None:
    """PSU-m4: the validation split is stated as 77 (not 'about sixty'), the learning-rate overfit prediction is gone,
    and no unrecorded learning-rate outcome is claimed."""
    md = _markdown(nb)
    assert not re.search(r"validation split[^.]{0,40}about (sixty|60)", md) and "sixty widgets" not in md
    assert "a validation split of 77 widgets" in md and "the epoch is 77" in md
    assert not re.search(r"a learning rate that is too high[^,]*,", md)
    assert "watch the training loss fall while the validation CIDEr-D drops" not in md
    assert "No run at a rate other than 1e-5 is recorded" in md
    record = (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")
    assert "330 / 77 / 164" in record


# --- PSU-m6: environment-labelled durations -----------------------------------------------------------------------------


def test_psu_m6_prerequisites_state_t4_and_cpu_durations(nb: dict) -> None:
    """PSU-m6: the Prerequisites give the recorded T4 wall time and a CPU figure labelled as an estimate."""
    prereq = next(_src(c) for c in nb["cells"] if c["cell_type"] == "markdown" and _src(c).startswith("## Prerequisites"))
    assert "**1,292.2 s**" in prereq and "Tesla T4" in prereq
    assert "CPU has not been timed" in prereq and "not a measurement" in prereq
    assert re.search(r"(minutes|hours?)\b.*CPU|CPU.*\b(minutes|hours?)", prereq)
    assert "1292.2 s" in (ROOT / "docs" / "release-verification.md").read_text(encoding="utf-8")


# --- PSU-m1 / PSU-m7 -----------------------------------------------------------------------------------------------------


def test_psu_m1_no_doubled_braces_in_markdown(nb: dict) -> None:
    """PSU-m1: no `{{`/`}}` reaches the learner; the id pattern renders with single braces."""
    md = _markdown(nb)
    assert "{{" not in md and "}}" not in md and "[A-Za-z0-9_.:-]{1,64}" in md


def test_psu_m7_preinstalled_variable_documented(nb: dict) -> None:
    """PSU-m7: DIMER_NOTEBOOK_CI_PREINSTALLED is explained in markdown wherever code reads it."""
    assert "DIMER_NOTEBOOK_CI_PREINSTALLED" in "\n".join(_src(c) for c in _code_cells(nb))
    assert "`DIMER_NOTEBOOK_CI_PREINSTALLED=1` lets an" in _markdown(nb)


# --- PSU-S6: parity also counts captions that differ from the frozen ones --------------------------------------------


def test_psu_s6_parity_counts_reloaded_captions_differing_from_frozen(nb: dict) -> None:
    """PSU-S6: Section 9 reports how many of the 8 reloaded captions differ from the frozen captions."""
    nine = _cell_with(nb, "reloaded = Pix2StructWidgetCaptioningPipeline.from_artifact(")
    assert "'reloaded_captions_differing_from_frozen': sum(f != b for f, b in zip(frozen_predictions[:8], after, strict=True))" in nine
    assert "assert parity['identical_captions'] == parity['of']" in nine


# --- PSU-m2: BYOD branch replayed with the review's inputs ------------------------------------------------------------


def _png(colour: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (64, 64), colour).save(buf, format="PNG")
    return buf.getvalue()


def _zip(entries: list[tuple[str, bytes | str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for n, d in entries:
            z.writestr(n, d)
    return buf.getvalue()


def _jsonl(n: int, folder: str = "", drop: str | None = None) -> str:
    rows = []
    for i in range(n):
        row = {"id": f"r{i}", "image": f"{folder}img{i}.png", "box": [8, 8, 40, 40], "captions": [f"button number {i}"], "category": "large-widget"}
        if drop:
            row.pop(drop)
        rows.append(json.dumps(row))
    return "\n".join(rows)


def _imgs(n: int, folder: str = "") -> list[tuple[str, bytes]]:
    return [(f"{folder}img{i}.png", _png("red" if i % 2 else "blue")) for i in range(n)]


def _byod_branch(nb: dict) -> str:
    source = _cell_with(nb, "BYOD_PATH = ''  # @param")
    body = source.split("\nif USE_BYOD:\n", 1)[1].split("\nelse:\n", 1)[0]
    return textwrap.dedent(body)


def _run_byod(nb: dict, work: Path, payload: bytes) -> dict:
    from pix2struct_ui_captioning_pipeline.samples import (
        MAX_RECORDS,
        MIN_RECORDS,
        load_byod_dataset,
        split_dataset,
    )

    zip_path = work / "mine.zip"
    zip_path.write_bytes(payload)
    cwd = Path.cwd()
    os.chdir(work)
    try:
        ns = {
            "Path": Path, "io": io, "zipfile": zipfile, "shutil": shutil, "MAX_RECORDS": MAX_RECORDS, "MIN_RECORDS": MIN_RECORDS, "SPLIT_SEED": 42,
            "load_byod_dataset": load_byod_dataset, "split_dataset": split_dataset, "BYOD_PATH": str(zip_path),
            "byod_file": lambda path, kind, suffixes=(): Path(path),
        }
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(_byod_branch(nb), "<byod>", "exec"), ns)
    finally:
        os.chdir(cwd)
    return ns


def test_psu_m2_byod_valid_flat_and_nested_zips_are_split(nb: dict, tmp_path: Path) -> None:
    """PSU-m2 acceptance (review `valid_60_flat`, `valid_60_nested_folder_paths`): both are accepted and split 12/9/39;
    folders inside the zip are kept so `images/img0.png` resolves."""
    ns = _run_byod(nb, tmp_path, _zip([("records.jsonl", _jsonl(60))] + _imgs(60)))
    assert {k: len(v) for k, v in ns["splits"].items()} == {"test": 12, "validation": 9, "train": 39}
    ns = _run_byod(nb, tmp_path, _zip([("data/records.jsonl", _jsonl(60, folder="images/"))] + _imgs(60, "data/images/")))
    assert {k: len(v) for k, v in ns["splits"].items()} == {"test": 12, "validation": 9, "train": 39}


def test_psu_m2_byod_effective_minimum_is_stated_and_enforced(nb: dict, tmp_path: Path) -> None:
    """PSU-m2: 50 single-caption records pass every split check; 30, 45 and 49 are refused with the split named."""
    ns = _run_byod(nb, tmp_path, _zip([("records.jsonl", _jsonl(50))] + _imgs(50)))
    assert sum(len(v) for v in ns["splits"].values()) == 50
    for n, split in ((30, "test"), (45, "validation"), (49, "validation")):
        with pytest.raises(ValueError, match=rf"mine\.zip: split\(s\) \{{[^}}]*'{split}': \d+[^}}]*\}} hold fewer than MIN_RECORDS = 8 records .*50 records when every widget is its own screen and app"):
            _run_byod(nb, tmp_path, _zip([("records.jsonl", _jsonl(n))] + _imgs(n)))
    assert "50 records when every widget is its own screen and app" in _markdown(nb)


@pytest.mark.parametrize(
    ("name", "entries", "match"),
    [
        ("no_records_file", _imgs(60), r"mine\.zip: the zip must hold exactly one records\.jsonl.*found none"),
        ("two_records_files", [("records.jsonl", _jsonl(60)), ("sub/records.json", "[]")] + _imgs(60), r"exactly one records\.jsonl.*found \['records\.jsonl', 'sub/records\.json'\]"),
        ("missing_captions_key", [("records.jsonl", _jsonl(60, drop="captions"))] + _imgs(60), r"records\[0\] is missing 'captions'"),
        ("non_image_file", [("records.jsonl", _jsonl(60))] + _imgs(59) + [("img59.png", b"not an image")], r"records\[59\]: image cannot be decoded"),
        ("path_traversal", [("records.jsonl", _jsonl(60)), ("../escape.png", _png("red"))] + _imgs(60), r"mine\.zip: member '\.\./escape\.png' points outside the zip"),
    ],
)
def test_psu_m2_byod_refusals_name_the_zip_and_the_rule(nb: dict, tmp_path: Path, name: str, entries: list, match: str) -> None:
    """PSU-m2 acceptance: each expected failure raises ValueError naming the failed condition (no StopIteration)."""
    with pytest.raises(ValueError, match=match):
        _run_byod(nb, tmp_path, _zip(entries))


def test_psu_m2_byod_member_count_and_expanded_size_are_capped(nb: dict, tmp_path: Path) -> None:
    """PSU-m2 (§20): more members than MAX_RECORDS + 2, or more than 2 GiB extracted, is refused before extraction."""
    from pix2struct_ui_captioning_pipeline.samples import MAX_RECORDS

    too_many = _zip([("records.jsonl", _jsonl(8))] + [(f"f{i}.txt", b"x") for i in range(MAX_RECORDS + 2)])
    with pytest.raises(ValueError, match=rf"mine\.zip: {MAX_RECORDS + 3} files .* exceed the BYOD ceiling of {MAX_RECORDS + 2} files"):
        _run_byod(nb, tmp_path, too_many)
    assert not any((tmp_path / "work" / "byod").glob("f*.txt"))
    branch = _byod_branch(nb)
    assert "BYOD_MAX_EXPANDED_BYTES = MAX_RECORDS + 2, 2 * 1024 ** 3" in branch and "expanded > BYOD_MAX_EXPANDED_BYTES" in branch


def test_psu_m2_cancelled_upload_is_named_not_stopiteration(nb: dict) -> None:
    """PSU-m2 (review `cancelled_upload`): the helper checks the upload count before `next(iter(...))`."""
    source = _cell_with(nb, "BYOD_PATH = ''  # @param")
    assert "Upload exactly one" in source and source.index("if len(uploaded) != 1:") < source.index("next(iter(uploaded.items()))")
