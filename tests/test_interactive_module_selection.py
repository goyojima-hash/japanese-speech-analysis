from __future__ import annotations

import unittest
from unittest.mock import patch

from jgrade_eval.fact_modules import INTERACTIVE_FACT_MODULES
from jgrade_eval.interactive import prompt_fact_module_selection


class InteractiveModuleSelectionTests(unittest.TestCase):
    def test_enter_selects_all_five_modules(self) -> None:
        with patch("builtins.input", return_value=""):
            self.assertEqual(prompt_fact_module_selection(), INTERACTIVE_FACT_MODULES)

    def test_number_list_selects_only_requested_modules(self) -> None:
        with patch("builtins.input", return_value="1,3,5"):
            self.assertEqual(prompt_fact_module_selection(), frozenset({"fluency", "accuracy", "interaction"}))


if __name__ == "__main__":
    unittest.main()
