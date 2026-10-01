from __future__ import annotations

from pathlib import Path
import unittest
from unittest.mock import patch

from jgrade_eval.api_service import evaluate_speech_level
from jgrade_eval.live_judges import ProviderSpec
from jgrade_eval.mock_judges import judge_auto_cefr_with_mock_panel
from jgrade_eval.task_observation import TaskObservation
from jgrade_eval.task_rubrics import TaskRubric


class FakeExtractor:
    def extract(self, audio_path: Path) -> dict:
        return {
            "audio_path": str(audio_path),
            "raw_transcript_hiragana": "らいしゅうのきんようびでいいですか",
            "raw_transcript_romaji": "",
            "fluency_metrics": {
                "audio_duration_sec": 4.0, "speech_sec": 3.0,
                "speech_ratio_pct": 75.0, "pause_total_sec": 1.0,
                "pause_count": 1, "avg_pause_sec": 1.0,
                "max_pause_sec": 1.0, "mora_count": 17,
                "mora_per_sec": 4.25,
            },
            "top_pauses": [], "speech_segments": [], "extraction_time_sec": 0.01,
            "stt_model": "fake", "vad_model": "fake",
        }


def _context() -> dict:
    return {
        "prompts": [{"prompt_id": "q1", "text": "日付を確認してください",
                     "task_type": "date_confirmation", "rubric_id": "date-test"}],
        "whole_recording_answer_prompt_id": "q1",
    }


def _rubric() -> TaskRubric:
    return TaskRubric.from_dict({
        "rubric_id": "date-test", "version": "test-v1", "task_type": "date_confirmation",
        "approval_status": "experimental",
        "rules": [{"rule_id": "confirmed", "all_features": {"act_type": "confirmation_request"},
                   "rating": "○"}],
    })


class TaskAssessmentApiTests(unittest.TestCase):
    def test_off_keeps_legacy_result_and_never_observes(self) -> None:
        with patch("jgrade_eval.api_service.observe_task_with_provider") as observer:
            result = evaluate_speech_level(
                Path("sample.mp3"), extractor=FakeExtractor(), judge_mode="mock",
                selected_modules=("accuracy",), task_context=_context(),
                assessment_mode="off", task_rubrics={"date-test": _rubric()},
            )
        self.assertNotIn("task_assessment_shadow", result)
        self.assertNotIn("interaction_data", result["objective_data"])
        observer.assert_not_called()

    def test_shadow_applies_only_separate_rating_without_interaction_axis(self) -> None:
        baseline = evaluate_speech_level(
            Path("sample.mp3"), extractor=FakeExtractor(), judge_mode="mock",
            selected_modules=("accuracy",),
        )
        captured = []
        def fake_observe(prompt, answer, spec, *, timeout_sec):
            captured.append((prompt.prompt_id, answer.mapping_status, spec.model, timeout_sec))
            return TaskObservation.from_dict({
                "prompt_id": "q1", "segment_id": "whole-recording",
                "act_type": "confirmation_request", "target": "date",
                "quote": "らいしゅう", "start_offset": 0, "end_offset": 5,
                "features": {}, "judge_id": "O1", "model": "fake",
            }, answer=answer)

        with (
            patch("jgrade_eval.api_service.judge_auto_cefr_with_live_panel_partial",
                  side_effect=lambda roleplay_input, *args, **kwargs: (
                      judge_auto_cefr_with_mock_panel(roleplay_input), [])),
            patch("jgrade_eval.api_service.observe_task_with_provider", side_effect=fake_observe),
        ):
            result = evaluate_speech_level(
                Path("sample.mp3"), extractor=FakeExtractor(), judge_mode="live",
                provider_specs=[ProviderSpec("openai", "fake-model")],
                selected_modules=("accuracy",), task_context=_context(),
                assessment_mode="shadow", task_rubrics={"date-test": _rubric()},
            )
        self.assertEqual(result["task_rating"], baseline["task_rating"])
        self.assertEqual(result["final_cefr_level"], baseline["final_cefr_level"])
        self.assertEqual(result["task_assessment_shadow"]["rating"], "○")
        self.assertEqual(captured[0][:3], ("q1", "confirmed", "fake-model"))
        self.assertNotIn("interaction_data", result["objective_data"])

    def test_shadow_with_mock_exposes_no_fake_scoring(self) -> None:
        result = evaluate_speech_level(
            Path("sample.mp3"), extractor=FakeExtractor(), judge_mode="mock",
            selected_modules=("accuracy",), task_context=_context(),
            assessment_mode="shadow", task_rubrics={"date-test": _rubric()},
        )
        self.assertEqual(result["task_assessment_shadow"]["status"], "mock_unavailable")
        self.assertIsNone(result["task_assessment_shadow"]["rating"])

    def test_shadow_keeps_observation_without_registered_rubric(self) -> None:
        def fake_observe(prompt, answer, spec, *, timeout_sec):
            return TaskObservation.from_dict({
                "prompt_id": prompt.prompt_id, "segment_id": answer.segment_id,
                "act_type": "question", "target": None,
                "quote": answer.transcript_text[:5], "start_offset": answer.start_offset,
                "end_offset": answer.start_offset + 5,
                "features": {}, "judge_id": "O1", "model": spec.model,
            }, answer=answer)

        with (
            patch("jgrade_eval.api_service.judge_auto_cefr_with_live_panel_partial",
                  side_effect=lambda roleplay_input, *args, **kwargs: (
                      judge_auto_cefr_with_mock_panel(roleplay_input), [])),
            patch("jgrade_eval.api_service.observe_task_with_provider", side_effect=fake_observe),
        ):
            result = evaluate_speech_level(
                Path("sample.mp3"), extractor=FakeExtractor(), judge_mode="live",
                provider_specs=[ProviderSpec("openai", "fake-model")],
                selected_modules=("accuracy",), task_context=_context(),
                assessment_mode="shadow", task_rubrics={},
            )
        shadow = result["task_assessment_shadow"]
        self.assertEqual(shadow["status"], "rubric_unavailable")
        self.assertEqual(shadow["observation"]["act_type"], "question")
        self.assertIsNone(shadow["rating"])

    def test_shadow_keeps_two_confirmed_answers_separate(self) -> None:
        transcript = "らいしゅうのきんようびでいいですか"
        task_context = {
            "prompts": [
                {"prompt_id": "q1", "text": "最初の確認", "task_type": "date_confirmation", "rubric_id": "date-test"},
                {"prompt_id": "q2", "text": "次の確認", "task_type": "date_confirmation", "rubric_id": "date-test"},
            ],
            "answers": [
                {"prompt_id": "q1", "start_offset": 0, "end_offset": 8,
                 "transcript_text": transcript[:8], "confirmed_by": "teacher-1"},
                {"prompt_id": "q2", "start_offset": 8, "end_offset": len(transcript),
                 "transcript_text": transcript[8:], "confirmed_by": "teacher-1"},
            ],
        }
        def fake_observe(prompt, answer, spec, *, timeout_sec):
            return TaskObservation.from_dict({
                "prompt_id": prompt.prompt_id, "segment_id": answer.segment_id,
                "act_type": "confirmation_request", "target": "date",
                "quote": answer.transcript_text,
                "start_offset": answer.start_offset, "end_offset": answer.end_offset,
                "features": {}, "judge_id": "O1", "model": "fake",
            }, answer=answer)
        with (
            patch("jgrade_eval.api_service.judge_auto_cefr_with_live_panel_partial",
                  side_effect=lambda roleplay_input, *args, **kwargs: (
                      judge_auto_cefr_with_mock_panel(roleplay_input), [])),
            patch("jgrade_eval.api_service.observe_task_with_provider", side_effect=fake_observe),
        ):
            result = evaluate_speech_level(
                Path("sample.mp3"), extractor=FakeExtractor(), judge_mode="live",
                provider_specs=[ProviderSpec("openai", "fake-model")],
                selected_modules=("accuracy",), task_context=task_context,
                assessment_mode="shadow", task_rubrics={"date-test": _rubric()},
            )
        self.assertEqual([item["prompt_id"] for item in result["task_assessments_shadow"]],
                         ["q1", "q2"])
        self.assertNotIn("task_assessment_shadow", result)

    def test_bad_rubric_registry_does_not_erase_legacy_result(self) -> None:
        def fake_observe(prompt, answer, spec, *, timeout_sec):
            return TaskObservation.from_dict({
                "prompt_id": prompt.prompt_id, "segment_id": answer.segment_id,
                "act_type": "question", "target": None,
                "quote": answer.transcript_text[:5], "start_offset": answer.start_offset,
                "end_offset": answer.start_offset + 5,
                "features": {}, "judge_id": "O1", "model": spec.model,
            }, answer=answer)
        with (
            patch("jgrade_eval.api_service.judge_auto_cefr_with_live_panel_partial",
                  side_effect=lambda roleplay_input, *args, **kwargs: (
                      judge_auto_cefr_with_mock_panel(roleplay_input), [])),
            patch("jgrade_eval.api_service.load_default_task_rubrics",
                  side_effect=ValueError("bad operator config")),
            patch("jgrade_eval.api_service.observe_task_with_provider", side_effect=fake_observe),
        ):
            result = evaluate_speech_level(
                Path("sample.mp3"), extractor=FakeExtractor(), judge_mode="live",
                provider_specs=[ProviderSpec("openai", "fake-model")],
                selected_modules=("accuracy",), task_context=_context(),
                assessment_mode="shadow",
            )
        self.assertIn(result["task_rating"], {"◎", "○", "△", "×"})
        self.assertEqual(result["task_assessment_shadow"]["status"], "rubric_registry_failed")
        self.assertEqual(result["task_assessment_shadow"]["observation"]["act_type"], "question")


if __name__ == "__main__":
    unittest.main()
