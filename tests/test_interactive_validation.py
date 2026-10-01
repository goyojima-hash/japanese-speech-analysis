import io
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

from jgrade_eval.interactive_validation import prompt_validation_mode, run_validation
from jgrade_eval.live_judges import ProviderSpec
from jgrade_eval.task_observation import TaskObservation
from jgrade_eval.tuning_samples import review_record_to_item


DATA = {"raw_transcript_hiragana": "お茶をください。", "fluency_metrics": {"audio_duration_sec": 3}}
CONTEXT = {"prompts": [{"prompt_id": "q1", "text": "注文してください。"}]}


class InteractiveValidationTests(unittest.TestCase):
    def test_prompt_defaults_off_and_requires_explicit_on(self):
        for text, expected in [("", False), ("n", False), ("y", True), ("on", True)]:
            with patch("builtins.input", return_value=text), redirect_stdout(io.StringIO()):
                self.assertEqual(prompt_validation_mode(), expected)
        with patch("builtins.input", side_effect=EOFError), redirect_stdout(io.StringIO()):
            self.assertFalse(prompt_validation_mode())

    def test_off_never_loads_rules_prompts_or_observes(self):
        with patch("builtins.input") as input_mock, patch("jgrade_eval.interactive_validation.load_default_task_rubrics") as rules, patch("jgrade_eval.interactive_validation.observe_task_with_provider") as observe:
            self.assertIsNone(run_validation(enabled=False, objective_data=DATA, interaction_context=CONTEXT,
                                             judge_mode="live", provider_specs=[], timeout_sec=60))
        input_mock.assert_not_called()
        rules.assert_not_called()
        observe.assert_not_called()

    def test_live_confirmed_observation_without_rubric(self):
        def observe(prompt, answer, spec, *, timeout_sec):
            self.assertEqual(timeout_sec, 20)
            return TaskObservation.from_dict({
                "prompt_id": prompt.prompt_id, "segment_id": answer.segment_id,
                "act_type": "request", "target": "tea", "quote": answer.transcript_text,
                "start_offset": 0, "end_offset": 8, "features": {}, "judge_id": "O1", "model": spec.model,
            }, answer=answer)
        with patch("builtins.input", return_value="y"), patch("jgrade_eval.interactive_validation.load_default_task_rubrics", return_value={}), patch("jgrade_eval.interactive_validation.observe_task_with_provider", side_effect=observe) as observer, redirect_stdout(io.StringIO()):
            result = run_validation(enabled=True, objective_data=DATA, interaction_context=CONTEXT,
                                    judge_mode="live", provider_specs=[ProviderSpec("openai", "test-model")], timeout_sec=60)
        observer.assert_called_once()
        self.assertEqual(result["results"][0]["status"], "rubric_unavailable")
        self.assertEqual(result["results"][0]["observation"]["act_type"], "request")
        self.assertIsNone(result["results"][0]["rating"])

    def test_missing_prompt_unconfirmed_or_mock_never_observes(self):
        for context, answer, mode, status in [({}, "", "live", "insufficient_context"),
                                               (CONTEXT, "", "live", "insufficient_context"),
                                               (CONTEXT, "y", "mock", "mock_unavailable")]:
            with patch("builtins.input", return_value=answer), patch("jgrade_eval.interactive_validation.observe_task_with_provider") as observer, redirect_stdout(io.StringIO()):
                result = run_validation(enabled=True, objective_data=DATA, interaction_context=context,
                                        judge_mode=mode, provider_specs=[ProviderSpec("openai", "test")], timeout_sec=1)
            observer.assert_not_called()
            self.assertEqual(result["results"][0]["status"], status)

    def test_failures_do_not_raise_or_change_input(self):
        for rules_fail in [False, True]:
            with patch("builtins.input", return_value="y"), patch("jgrade_eval.interactive_validation.load_default_task_rubrics", side_effect=ValueError("broken") if rules_fail else None, return_value={}), patch("jgrade_eval.interactive_validation.observe_task_with_provider", side_effect=RuntimeError("provider")), redirect_stdout(io.StringIO()):
                result = run_validation(enabled=True, objective_data=DATA, interaction_context=CONTEXT,
                                        judge_mode="live", provider_specs=[ProviderSpec("openai", "test")], timeout_sec=1)
            self.assertEqual(result["results"][0]["status"], "observation_failed")
            self.assertEqual(DATA["raw_transcript_hiragana"], "お茶をください。")

    def test_prediction_history_stores_validation_only_when_present(self):
        record = {"sample_id": "test", "audio_path": "sample.wav", "objective_data": DATA}
        self.assertNotIn("task_validation", review_record_to_item(record, source="interactive")["last_prediction"])
        record["task_validation"] = {"mode": "shadow", "results": []}
        item = review_record_to_item(record, source="interactive")
        self.assertEqual(item["last_prediction"]["task_validation"], record["task_validation"])
        self.assertEqual(item["human_rating"], "")
