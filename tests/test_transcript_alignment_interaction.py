from __future__ import annotations

import unittest

from jgrade_eval.evidence.models import (
    EvidenceBundle, LinguisticEvidence, MoraTiming, SourceEvidence, SpeechEvidence, TimedSpan,
)
from jgrade_eval.interaction import InteractionModule
from jgrade_eval.fact_modules import run_fact_modules


def bundle(*, mismatch: bool = False) -> EvidenceBundle:
    timings = (
        MoraTiming("あ", 0.0, 0.2), MoraTiming("い", 0.2, 0.4),
        MoraTiming("え" if mismatch else "う", 2.0, 2.2),
        MoraTiming("え", 2.2, 2.4), MoraTiming("お", 2.4, 2.6),
    )
    return EvidenceBundle(
        source=SourceEvidence("sample.wav", None),
        speech=SpeechEvidence(
            raw_transcript_hiragana="あいうえお", raw_transcript_romaji="",
            duration_sec=4.0,
            speech_segments=(TimedSpan(0.0, 0.4), TimedSpan(2.0, 2.6)),
            pause_segments=(TimedSpan(0.4, 2.0),),
            mora_timings=timings, factual_metrics=(), provenance=(("stt_model", "fake-ctc"),),
        ),
        linguistic=LinguisticEvidence(tokens=(), tokenizer_version="fake", split_mode="A"),
    )


class AlignmentInteractionTests(unittest.TestCase):
    def test_pause_candidates_gain_unconfirmed_text_ranges(self) -> None:
        data = InteractionModule().collect(bundle(), interaction_context={
            "prompts": [{"prompt_id": "q1", "text": "質問1"},
                        {"prompt_id": "q2", "text": "質問2"}],
        }).to_dict()
        segments = data["candidate_answer_segments"]
        self.assertEqual([item["transcript_text"] for item in segments], ["あい", "うえお"])
        self.assertEqual([(item["start_offset"], item["end_offset"]) for item in segments],
                         [(0, 2), (2, 5)])
        self.assertTrue(all(item["mapping_status"] == "candidate" for item in segments))
        self.assertTrue(all("confirmed_by" not in item and "prompt_id" not in item
                            for item in segments))

    def test_mismatch_keeps_time_candidates_without_text_guess(self) -> None:
        data = InteractionModule().collect(bundle(mismatch=True)).to_dict()
        self.assertTrue(data["candidate_answer_segments"])
        self.assertTrue(all("transcript_text" not in item
                            for item in data["candidate_answer_segments"]))

    def test_new_candidates_do_not_change_legacy_judge_packet(self) -> None:
        run = run_fact_modules(bundle(), selected_modules=("interaction",))
        public_segments = run.packets["interaction_data"]["candidate_answer_segments"]
        judge_segments = run.add_packets_to_roleplay_input({})["interaction_data"]["candidate_answer_segments"]
        self.assertIn("transcript_text", public_segments[0])
        self.assertEqual([item["mapping_status"] for item in public_segments],
                         [item["mapping_status"] for item in judge_segments])
        self.assertTrue(all("transcript_text" not in item and "alignment_status" not in item
                            for item in judge_segments))


if __name__ == "__main__":
    unittest.main()
