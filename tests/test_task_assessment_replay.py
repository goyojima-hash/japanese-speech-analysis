import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from jgrade_eval.task_assessment_replay import replay_cases
from jgrade_eval.task_rubrics import TaskRubric


def case():
    return {
        "case_id": "synthetic-1", "transcript": "お茶をください。", "duration_sec": 3,
        "task_context": {
            "prompts": [{"prompt_id": "p1", "text": "飲み物を頼んでください。",
                         "task_type": "synthetic", "rubric_id": "test-only"}],
            "whole_recording_answer_prompt_id": "p1",
        },
        "observations": {"p1": {
            "prompt_id": "p1", "segment_id": "whole-recording", "act_type": "request",
            "target": "tea", "quote": "お茶をください。", "start_offset": 0, "end_offset": 8,
            "features": {"explicit": True}, "judge_id": "O1", "model": "synthetic",
        }},
    }


def rubric(version="1", rating="○"):
    return TaskRubric.from_dict({
        "rubric_id": "test-only", "version": version, "task_type": "synthetic",
        "approval_status": "experimental", "rules": [
            {"rule_id": "request", "all_features": {"act_type": "request"}, "rating": rating},
        ],
    })


class ReplayTests(unittest.TestCase):
    def test_no_labels_no_accuracy_and_input_unchanged(self):
        value = case()
        before = copy.deepcopy(value)
        report = replay_cases([value], [rubric()])
        self.assertIsNone(report["observation_metrics"]["accuracy"])
        self.assertIsNone(report["rubric_runs"][0]["metrics"]["accuracy"])
        self.assertEqual(value, before)
        self.assertEqual(len(report["input_sha256"]), 64)

    def test_same_observation_different_rubric_versions(self):
        value = case()
        value["labels"] = {"p1": {"observation": {"act_type": "request"}, "rating": "○"}}
        report = replay_cases([value], [rubric(), rubric("2", "△")])
        self.assertEqual(report["observation_metrics"]["accuracy"], 1)
        self.assertEqual([r["metrics"]["accuracy"] for r in report["rubric_runs"]], [1, 0])
        self.assertEqual(report["observations"][0]["observation"]["model"], "synthetic")

    def test_invalid_quote_counts_in_denominator(self):
        value = case()
        value["observations"]["p1"]["quote"] = "別の発話"
        value["labels"] = {"p1": {"observation": {"act_type": "request"}, "rating": "○"}}
        report = replay_cases([value], [rubric()])
        self.assertEqual(report["observation_metrics"], {"labeled": 1, "predicted": 0, "correct": 0, "accuracy": 0})
        self.assertEqual(report["rubric_runs"][0]["results"][0]["status"], "observation_failed")
        self.assertEqual(report["rubric_runs"][0]["metrics"]["labeled"], 1)

    def test_no_rubric_keeps_valid_observation(self):
        report = replay_cases([case()], [])
        self.assertIsNotNone(report["observations"][0]["observation"])
        self.assertEqual(report["rubric_runs"][0]["results"][0]["status"], "rubric_unavailable")

    def test_unconfirmed_answer_not_observed(self):
        value = case()
        del value["task_context"]["whole_recording_answer_prompt_id"]
        report = replay_cases([value], [rubric()])
        self.assertIsNone(report["observations"][0]["observation"])
        self.assertEqual(report["rubric_runs"][0]["results"][0]["status"], "insufficient_context")

    def test_feature_types_are_not_coerced(self):
        value = case()
        value["labels"] = {"p1": {"observation": {"features": {"explicit": 1}}}}
        self.assertEqual(replay_cases([value], [rubric()])["observation_metrics"]["accuracy"], 0)

    def test_rejects_bad_labels_and_duplicates(self):
        for labels in [{"unknown": {"rating": "○"}}, {"p1": {"observation": {}}},
                       {"p1": {"observation": {"score": 1}}}, {"p1": {"rating": "invalid"}}]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                replay_cases([{**case(), "labels": labels}], [rubric()])
        with self.assertRaises(ValueError):
            replay_cases([case(), case()], [rubric()])
        with self.assertRaises(ValueError):
            replay_cases([case()], [rubric(), rubric()])

    def test_cli_outputs_report_without_changing_input(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "cases.json"
            output = Path(directory) / "report.json"
            source.write_text(json.dumps({"cases": [case()]}), encoding="utf-8")
            before = source.read_bytes()
            command = [sys.executable, "-m", "jgrade_eval.task_assessment_replay", str(source), "--output", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(output.read_text())["schema_version"], "task-replay.v1")
            self.assertEqual(source.read_bytes(), before)
            failed = subprocess.run(command[:-1] + [str(source)], capture_output=True, text=True)
            self.assertNotEqual(failed.returncode, 0)
            self.assertEqual(source.read_bytes(), before)
