import unittest

from jgrade_eval.prompts import AUTO_CEFR_SYSTEM_PROMPT, build_auto_cefr_judge_messages


class AccuracyPromptTests(unittest.TestCase):
    def test_accuracy_facts_are_not_instructions_to_assign_errors(self) -> None:
        messages = build_auto_cefr_judge_messages(
            {
                "raw_transcript_hiragana": "わたしはすしです",
                "fluency_metrics": {},
                "accuracy_data": {
                    "asr_observations": [{"calibration_status": "unavailable"}],
                    "unavailable_capabilities": ["asr_posterior_confidence"],
                },
            },
            judge_id="A",
            model_family="openai",
        )

        self.assertIn("`accuracy_data`", AUTO_CEFR_SYSTEM_PROMPT)
        self.assertIn("正誤", AUTO_CEFR_SYSTEM_PROMPT)
        self.assertIn("未較正", AUTO_CEFR_SYSTEM_PROMPT)
        self.assertIn("accuracy_data", messages["user"])

    def test_coherence_facts_are_not_instructions_to_assign_quality_or_error(self) -> None:
        messages = build_auto_cefr_judge_messages(
            {
                "raw_transcript_hiragana": "すしですだからたべます",
                "fluency_metrics": {},
                "coherence_data": {
                    "candidate_units": [{"boundary_derivations": ["connective:causal"]}],
                    "unavailable_capabilities": ["coreference_resolution"],
                },
            },
            judge_id="A",
            model_family="openai",
        )

        self.assertIn("`coherence_data`", AUTO_CEFR_SYSTEM_PROMPT)
        self.assertIn("候補", AUTO_CEFR_SYSTEM_PROMPT)
        self.assertIn("単独", AUTO_CEFR_SYSTEM_PROMPT)
        self.assertIn("coherence_data", messages["user"])


if __name__ == "__main__":
    unittest.main()
