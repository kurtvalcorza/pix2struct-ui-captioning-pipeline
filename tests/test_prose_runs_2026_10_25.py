"""Prose numbers stay true across every recorded hosted run (Kurt, 2026-10-25: point to output + ranges).

Fine-tune times: Kaggle T4 `2e28896c` 612.4 s (2026-09-26, 1,292.2 s wall), Colab T4 `cf909dbc` 654.8 s
(2026-10-08), Colab T4 `820076d3` 646.6 s (2026-10-25). Timings are approximate with the runtime named.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "pix2struct_ui_captioning_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_timings_are_approximate_and_name_the_runtime():
    md = _markdown()
    for exact in ("654.8 s", "646.6 s", "612.4 s", "612 s", "1,292.2 s"):
        assert exact not in md, exact
    assert "about 11 minutes of fine-tuning on a Colab T4" in md and "`820076d3`" in md
