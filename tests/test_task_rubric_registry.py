from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from jgrade_eval.task_rubrics import TaskRubric, load_task_rubrics


class TaskRubricRegistryTests(unittest.TestCase):
    def test_missing_directory_is_empty(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(load_task_rubrics(Path(directory) / "missing"), {})

    def test_loads_versioned_experimental_rules_from_admin_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "date-test.json"
            path.write_text(json.dumps({
                "rubric_id": "date-test", "version": "test-v1",
                "task_type": "date_confirmation", "approval_status": "experimental",
                "rules": [{"rule_id": "r1", "all_features": {"act_type": "confirmation_request"},
                           "rating": "○"}],
            }), encoding="utf-8")
            loaded = load_task_rubrics(Path(directory))
        self.assertEqual(loaded["date-test"].version, "test-v1")

    def test_approved_rule_requires_source_and_approver(self) -> None:
        with self.assertRaisesRegex(ValueError, "approved rubric"):
            TaskRubric.from_dict({
                "rubric_id": "date-test", "version": "v1",
                "task_type": "date_confirmation", "approval_status": "approved",
                "rules": [{"rule_id": "r1", "all_features": {"act_type": "confirmation_request"},
                           "rating": "○"}],
            })


if __name__ == "__main__":
    unittest.main()
