from __future__ import annotations

import unittest

from jgrade_eval.evidence.models import (
    EvidenceBundle,
    LinguisticEvidence,
    SourceEvidence,
    SpeechEvidence,
    TimedSpan,
    TokenEvidence,
)


class FactModuleRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bundle = EvidenceBundle(
            source=SourceEvidence(audio_path="sample.wav", audio_sha256=None),
            speech=SpeechEvidence(
                raw_transcript_hiragana="すしですだからすしをたべます",
                raw_transcript_romaji="",
                duration_sec=3.0,
                speech_segments=(TimedSpan(0.0, 2.0),),
                pause_segments=(),
                mora_timings=(),
                factual_metrics=(),
                provenance=(("stt_model", "fake-stt"),),
            ),
            linguistic=LinguisticEvidence(
                tokens=(
                    TokenEvidence("すし", "寿司", "すし", ("名詞",), 0, 2),
                    TokenEvidence("です", "です", "です", ("助動詞",), 2, 4),
                    TokenEvidence("だから", "だから", "だから", ("接続詞",), 4, 7),
                    TokenEvidence("すし", "寿司", "すし", ("名詞",), 7, 9),
                    TokenEvidence("を", "を", "を", ("助詞",), 9, 10),
                    TokenEvidence("たべます", "食べる", "たべます", ("動詞",), 10, 14),
                ),
                tokenizer_version="fake-tokenizer",
                split_mode="A",
            ),
        )

    def test_runner_collects_selected_packets_and_copies_them_to_both_consumers(self) -> None:
        from jgrade_eval.fact_modules import run_fact_modules

        result = run_fact_modules(
            self.bundle,
            selected_modules=("range", "accuracy", "coherence"),
            range_extractor=_FakeRangeExtractor(),
        )

        objective_data = result.merge_objective_data({"raw_transcript_hiragana": "すしですだからすしをたべます"})
        roleplay_input = result.add_packets_to_roleplay_input({"sample_id": "sample"})
        self.assertEqual(result.active_modules, frozenset({"range", "accuracy", "coherence"}))
        self.assertEqual(objective_data["range_data"]["source"], "shared-evidence")
        self.assertEqual(objective_data["accuracy_data"]["module_id"], "accuracy")
        self.assertEqual(objective_data["coherence_data"]["module_id"], "coherence")
        self.assertEqual(roleplay_input["coherence_data"], objective_data["coherence_data"])
        self.assertEqual(roleplay_input["range_data"], objective_data["range_data"])

    def test_runner_default_selects_all_implemented_modules_and_rejects_unknown_modules(self) -> None:
        from jgrade_eval.fact_modules import DEFAULT_FACT_MODULES, SUPPORTED_FACT_MODULES, run_fact_modules

        default_result = run_fact_modules(self.bundle, range_extractor=_FakeRangeExtractor())
        self.assertEqual(default_result.active_modules, DEFAULT_FACT_MODULES)
        self.assertEqual(default_result.active_modules, SUPPORTED_FACT_MODULES)
        self.assertIn("accuracy_data", default_result.packets)
        self.assertIn("coherence_data", default_result.packets)
        with self.assertRaisesRegex(ValueError, "unsupported fact module"):
            run_fact_modules(self.bundle, selected_modules=("unknown",))

    def test_runner_reports_each_selected_collector_in_deterministic_order(self) -> None:
        from jgrade_eval.fact_modules import run_fact_modules

        started: list[str] = []
        completed: list[str] = []
        run_fact_modules(
            self.bundle,
            selected_modules=("coherence", "accuracy", "range"),
            range_extractor=_FakeRangeExtractor(),
            on_module_start=started.append,
            on_module_result=lambda module_id, _packet: completed.append(module_id),
        )

        self.assertEqual(started, ["range", "accuracy", "coherence"])
        self.assertEqual(completed, ["range", "accuracy", "coherence"])


class _FakeRangeExtractor:
    def analyze_linguistic_evidence(self, linguistic: LinguisticEvidence) -> dict:
        return {"source": "shared-evidence", "token_count": len(linguistic.tokens)}


if __name__ == "__main__":
    unittest.main()
