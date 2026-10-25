"""Final prose pass (Kurt's ruling, 2026-10-26): the flagged expectation sentences are removed or
rewritten as records of named runs.
pix2struct: Section 6 intro.
Fails on PR head 347b6da.
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / 'tutorials/pix2struct_ui_captioning_colab.ipynb'


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


GONE = [
    'so expect it well above both baselines',
]
PRESENT = [
    "was fine-tuned on Widget Captioning's training apps; the cell records",
]


def test_flagged_expectations_are_gone():
    md = _markdown()
    for text in GONE:
        assert text not in md, text


def test_replacement_records_are_present():
    md = _markdown()
    for text in PRESENT:
        assert text in md, text
