"""Tests for the two documents a newcomer actually reads first.

The README badge and the Colab notebook are the front door. Both have drifted
before — the badge claimed 229 tests when there were 454, and the notebook told
people to expect a number that had not been true for several commits. A stale
count is a small lie that costs trust in every other number on the page, and it
is trivially machine-checkable, so it is checked here.

The notebook is also validated structurally: Colab refuses to open a notebook
whose code cells are missing ``outputs``, and a repo whose one-click badge leads
to a broken notebook is worse than one with no badge.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
NOTEBOOK = ROOT / "notebooks" / "Yaobi_Harness_Colab.ipynb"


def discovered_test_count() -> int:
    return unittest.defaultTestLoader.discover(str(ROOT / "tests")).countTestCases()


class TestCountClaimsTests(unittest.TestCase):
    """Every place that states a test count must state the real one."""

    def setUp(self):
        self.actual = discovered_test_count()

    def test_the_readme_badge_is_current(self):
        badge = re.search(r"tests-(\d+)%20passing", README.read_text(encoding="utf-8"))
        self.assertIsNotNone(badge, "the README lost its test badge")
        self.assertEqual(
            int(badge.group(1)), self.actual,
            f"README badge says {badge.group(1)}, discovery finds {self.actual}",
        )

    def test_every_readme_count_agrees_with_the_badge(self):
        text = README.read_text(encoding="utf-8")
        for claim in re.findall(r"(\d+) 个(?:用例|测试)", text):
            self.assertEqual(int(claim), self.actual,
                             f"README says {claim} 个用例, discovery finds {self.actual}")

    def test_the_notebook_count_is_current(self):
        text = NOTEBOOK.read_text(encoding="utf-8")
        claims = re.findall(r"(\d+) 个(?:用例|测试)", text)
        self.assertTrue(claims, "the notebook no longer states how many tests there are")
        for claim in claims:
            self.assertEqual(int(claim), self.actual,
                             f"notebook says {claim} 个用例, discovery finds {self.actual}")


class NotebookStructureTests(unittest.TestCase):
    def setUp(self):
        self.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

    def test_it_is_a_notebook_colab_will_open(self):
        self.assertEqual(self.nb.get("nbformat"), 4)
        for index, cell in enumerate(self.nb["cells"]):
            self.assertIn(cell["cell_type"], ("code", "markdown"), index)
            if cell["cell_type"] == "code":
                self.assertIn("outputs", cell, f"cell {index} has no outputs key")
                self.assertIn("execution_count", cell, f"cell {index} has no execution_count")

    def test_no_output_is_committed(self):
        """Outputs would carry whatever data the last run happened to hold."""
        for index, cell in enumerate(self.nb["cells"]):
            if cell["cell_type"] == "code":
                self.assertEqual(cell["outputs"], [], f"cell {index} has committed output")

    def test_the_walkthrough_covers_every_shipped_capability(self):
        """A feature with no cell is a feature nobody will find."""
        text = "".join("".join(c["source"]) for c in self.nb["cells"])
        for marker in ("interview", "ConsultPanel", "vision", "Journal", "parse_plan",
                       "ConsoleService", "panel_concurrency", "record_journal"):
            self.assertIn(marker, text, f"the notebook never mentions {marker}")


if __name__ == "__main__":
    unittest.main()
