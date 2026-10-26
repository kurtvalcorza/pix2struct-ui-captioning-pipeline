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
    # Historical wording (2026-10-26): exact recorded times are allowed again, each labelled with its run.
    assert "640.0–654.8 s of fine-tuning in the three Colab T4 runs" in md and "`820076d3`" in md


def test_historical_wording_records_every_hosted_run():
    # Historical wording (2026-10-26): timings are records of every named hosted run; the BLEU-4 delta is the
    # printed -0.134; no "has not been timed" status. Fails on PR #10 head 372548d.
    md = _markdown()
    for gone in ("about 11 minutes", "about 21½ minutes", "has not been timed", "−0.133",
                 "the expected outcome"):
        assert gone not in md, gone
    for quoted in ("1,188.3 s, 1,180.8 s and 1,161.8 s", "`a17c3510` (commit `0383932`, 25 October 2026)",
                   "printed `delta_vs_frozen` −0.134", "Your run may take a different time."):
        assert quoted in md, quoted
