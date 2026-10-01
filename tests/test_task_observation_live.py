from __future__ import annotations

import unittest
from unittest.mock import patch

from jgrade_eval.live_judges import ProviderSpec
from jgrade_eval.task_context import TaskContext
from jgrade_eval.task_observation import observe_task_with_provider


class TaskObservationLiveTests(unittest.TestCase):
    def setUp(self) -> None:
        context = TaskContext.from_inputs(
            roleplay_task="unknown", interaction_context=None,
            task_context={
                "prompts": [{"prompt_id": "q1", "text": "日付を確認してください",
                             "task_type": "date_confirmation", "rubric_id": "test"}],
                "whole_recording_answer_prompt_id": "q1",
            },
            transcript="らいしゅうのきんようびでいいですか", duration_sec=4,
        )
        self.prompt = context.prompts[0]
        self.answer = context.confirmed_answer_for("q1")

    def test_provider_observes_without_returning_a_rating(self) -> None:
        response = ('{"act_type":"confirmation_request","target":"date",'
                    '"quote":"きんようびでいいですか",'
                    '"features":{"date_anchor_present":true}}')
        with patch("jgrade_eval.task_observation._call_provider", return_value=response) as call:
            observation = observe_task_with_provider(
                self.prompt, self.answer, ProviderSpec("openai", "test-model"), timeout_sec=8,
            )
        self.assertEqual(observation.quote, "きんようびでいいですか")
        self.assertEqual((observation.start_offset, observation.end_offset), (6, 17))
        self.assertEqual(observation.model, "test-model")
        self.assertIn("達成度", call.call_args.args[1]["system"])
        self.assertNotIn("rating", observation.to_dict())

    def test_ambiguous_repeated_quote_is_rejected(self) -> None:
        context = TaskContext.from_inputs(
            roleplay_task="unknown", interaction_context=None,
            task_context={"prompts": [{"prompt_id": "q1", "text": "日付を確認"}],
                          "whole_recording_answer_prompt_id": "q1"},
            transcript="きんようびきんようび", duration_sec=3,
        )
        response = '{"act_type":"statement","quote":"きんようび","features":{}}'
        with patch("jgrade_eval.task_observation._call_provider", return_value=response):
            with self.assertRaisesRegex(ValueError, "ambiguous"):
                observe_task_with_provider(
                    context.prompts[0], context.confirmed_answer_for("q1"),
                    ProviderSpec("openai", "test-model"), timeout_sec=8,
                )


if __name__ == "__main__":
    unittest.main()
