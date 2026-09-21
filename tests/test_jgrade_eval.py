import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from jgrade_eval.consensus import AutoCefrConsensus, ConsensusGate
from jgrade_eval.deliberation import deliberate_auto_cefr
from jgrade_eval.jfs_samples import (
    build_jfs_tuning_dataset,
    download_jfs_assets,
    load_jfs_catalog,
    validate_jfs_catalog,
)
from jgrade_eval.interactive import (
    _format_provider_key_state,
    _normalize_audio_path_input,
    _upsert_env_value,
    apply_and_save_user_level_correction,
    list_audio_files,
    print_objective_data,
    prompt_user_cefr_level,
    prompt_provider_specs,
    run_interactive,
)
from jgrade_eval.range import RangeExtractor
from jgrade_eval.live_judges import (
    ProviderKeyStatus,
    ProviderKeyValidationHTTPError,
    ProviderSpec,
    _extract_json_object,
    judge_auto_cefr_with_live_panel_partial,
    missing_key_envs,
    parse_provider_specs,
    provider_key_status,
    validate_provider_key,
)
from jgrade_eval.judge_config import load_judge_console_config
from jgrade_eval.metrics import evaluate_against_humans
from jgrade_eval.mock_judges import judge_auto_cefr_with_mock_panel, judge_with_mock_panel
from jgrade_eval.models import AutoLevelJudgeResult, BenchmarkItem, JudgeResult, Rating
from jgrade_eval.prompts import build_auto_cefr_judge_messages, build_judge_messages
from jgrade_eval.tuning_profile import (
    TuningProfile,
    apply_level_correction,
    compose_auto_cefr_system_prompt,
    load_profile,
    restore_history,
    save_profile,
    with_history,
)
from jgrade_eval.tuning_runner import (
    compute_cefr_metrics,
    is_adjacent_level,
    run_tuning_dataset,
)
from jgrade_eval.tuning_samples import append_judged_sample, make_review_sample_id


def judge(judge_id: str, rating: str) -> JudgeResult:
    return JudgeResult(
        judge_id=judge_id,
        model_family={"A": "claude", "B": "gpt", "C": "gemini"}[judge_id],
        rating=Rating.parse(rating),
        confidence=0.8,
        rationale="test",
    )


class ConsensusGateTests(unittest.TestCase):
    def test_roleplay_uses_strict_majority(self) -> None:
        decision = ConsensusGate().decide_roleplay(
            "rp-1",
            [judge("A", "○"), judge("B", "○"), judge("C", "△")],
        )

        self.assertEqual(decision.consensus_rating, Rating.PASS)
        self.assertTrue(decision.has_strict_majority)
        self.assertFalse(decision.needs_human_review)

    def test_all_different_ratings_use_median_and_flag_review(self) -> None:
        decision = ConsensusGate().decide_roleplay(
            "rp-1",
            [judge("A", "◎"), judge("B", "△"), judge("C", "×")],
        )

        self.assertEqual(decision.consensus_rating, Rating.NEAR_FAIL)
        self.assertFalse(decision.has_strict_majority)
        self.assertTrue(decision.needs_human_review)

    def test_level_passes_when_two_roleplays_pass(self) -> None:
        gate = ConsensusGate()
        decisions = [
            gate.decide_roleplay("rp-1", [judge("A", "○"), judge("B", "○"), judge("C", "△")]),
            gate.decide_roleplay("rp-2", [judge("A", "△"), judge("B", "△"), judge("C", "○")]),
            gate.decide_roleplay("rp-3", [judge("A", "◎"), judge("B", "○"), judge("C", "○")]),
        ]

        level = gate.decide_level("B1", decisions)

        self.assertTrue(level.passed)
        self.assertEqual(level.final_level, "B1")

    def test_level_downgrades_when_two_roleplays_fail(self) -> None:
        gate = ConsensusGate()
        decisions = [
            gate.decide_roleplay("rp-1", [judge("A", "△"), judge("B", "△"), judge("C", "○")]),
            gate.decide_roleplay("rp-2", [judge("A", "×"), judge("B", "△"), judge("C", "×")]),
            gate.decide_roleplay("rp-3", [judge("A", "○"), judge("B", "○"), judge("C", "△")]),
        ]

        level = gate.decide_level("B1", decisions)

        self.assertFalse(level.passed)
        self.assertEqual(level.final_level, "A2")

    def test_two_judge_roleplay_disagreement_uses_conservative_rating(self) -> None:
        decision = ConsensusGate().decide_roleplay(
            "rp-1",
            [judge("A", "○"), judge("B", "△")],
        )

        self.assertEqual(decision.consensus_rating, Rating.NEAR_FAIL)
        self.assertFalse(decision.has_strict_majority)
        self.assertTrue(decision.needs_human_review)


class AutoCefrConsensusTests(unittest.TestCase):
    def test_auto_cefr_uses_majority_level(self) -> None:
        results = [
            AutoLevelJudgeResult("A", "anthropic", "B1", Rating.PASS, 0.8, "test"),
            AutoLevelJudgeResult("B", "openai", "B1", Rating.PASS, 0.7, "test"),
            AutoLevelJudgeResult("C", "gemini", "A2", Rating.PASS, 0.6, "test"),
        ]

        decision = AutoCefrConsensus().decide(results)

        self.assertEqual(decision.final_cefr_level, "B1")
        self.assertTrue(decision.has_strict_majority)
        self.assertFalse(decision.needs_human_review)

    def test_auto_cefr_uses_median_when_all_differ(self) -> None:
        results = [
            AutoLevelJudgeResult("A", "anthropic", "A2", Rating.PASS, 0.8, "test"),
            AutoLevelJudgeResult("B", "openai", "B2", Rating.PASS, 0.7, "test"),
            AutoLevelJudgeResult("C", "gemini", "B1", Rating.PASS, 0.6, "test"),
        ]

        decision = AutoCefrConsensus().decide(results)

        self.assertEqual(decision.final_cefr_level, "B1")
        self.assertFalse(decision.has_strict_majority)
        self.assertTrue(decision.needs_human_review)

    def test_auto_cefr_one_judge_runs_with_review_flag(self) -> None:
        results = [
            AutoLevelJudgeResult("A", "anthropic", "A2", Rating.PASS, 0.8, "test"),
        ]

        decision = AutoCefrConsensus().decide(results)

        self.assertEqual(decision.final_cefr_level, "A2")
        self.assertFalse(decision.has_strict_majority)
        self.assertTrue(decision.needs_human_review)

    def test_auto_cefr_two_judge_disagreement_uses_lower_level(self) -> None:
        results = [
            AutoLevelJudgeResult("A", "anthropic", "B2", Rating.PASS, 0.8, "test"),
            AutoLevelJudgeResult("B", "openai", "B1", Rating.PASS, 0.7, "test"),
        ]

        decision = AutoCefrConsensus().decide(results)

        self.assertEqual(decision.final_cefr_level, "B1")
        self.assertFalse(decision.has_strict_majority)
        self.assertTrue(decision.needs_human_review)

    def test_deliberation_uses_similar_human_correction_as_calibration(self) -> None:
        results = [
            AutoLevelJudgeResult("A", "anthropic", "A2", Rating.PASS, 0.7, "test"),
            AutoLevelJudgeResult("B", "openai", "A1", Rating.NEAR_FAIL, 0.6, "test"),
            AutoLevelJudgeResult("C", "gemini", "A2", Rating.PASS, 0.7, "test"),
        ]
        objective_data = {
            "raw_transcript_hiragana": "あ" * 580,
            "fluency_metrics": {
                "speech_ratio_pct": 65.0,
                "mora_per_sec": 3.35,
                "max_pause_sec": 4.8,
            },
        }
        profile = TuningProfile(
            tuning_examples=(
                {
                    "sample_id": "example-1",
                    "previous_predicted_cefr": "A2",
                    "corrected_cefr": "B1",
                    "raw_transcript_hiragana": "い" * 587,
                    "fluency_metrics": {
                        "speech_ratio_pct": 65.0,
                        "mora_per_sec": 3.35,
                        "max_pause_sec": 4.8,
                    },
                },
            )
        )

        deliberation = deliberate_auto_cefr(
            results,
            objective_data=objective_data,
            profile=profile,
        )

        self.assertEqual(deliberation.raw_decision.final_cefr_level, "A2")
        self.assertEqual(deliberation.final_decision.final_cefr_level, "B1")
        self.assertTrue(deliberation.applied_calibration)
        self.assertIn("A2->B1", deliberation.conclusion)
        self.assertEqual(len(deliberation.judge_summaries), 3)


class MetricsTests(unittest.TestCase):
    def test_evaluate_against_humans_reports_core_metrics(self) -> None:
        report = evaluate_against_humans(
            [
                BenchmarkItem("s1", "B1", "rp-1", Rating.PASS, Rating.PASS),
                BenchmarkItem("s2", "B1", "rp-2", Rating.NEAR_FAIL, Rating.NEAR_FAIL),
                BenchmarkItem("s3", "B1", "rp-3", Rating.PASS, Rating.NEAR_FAIL, True),
            ]
        )

        self.assertEqual(report["n"], 3)
        self.assertAlmostEqual(report["accuracy"], 2 / 3)
        self.assertEqual(report["boundary_confusions_pass_vs_near_fail"], ["s3"])
        self.assertAlmostEqual(report["judge_disagreement_rate"], 1 / 3)


class PromptTests(unittest.TestCase):
    def test_prompt_contains_output_contract_and_no_correction_rule(self) -> None:
        messages = build_judge_messages(
            {
                "roleplay_task": "道を尋ねる",
                "raw_transcript_hiragana": "えきはどこですか",
                "fluency_metrics": {"mora_per_sec": 5.0},
            },
            judge_id="A",
            model_family="claude",
        )

        self.assertIn("漢字変換", messages["system"])
        self.assertIn('"rating": "◎|○|△|×"', messages["system"])
        self.assertIn("えきはどこですか", messages["user"])

    def test_auto_cefr_prompt_contains_level_contract(self) -> None:
        messages = build_auto_cefr_judge_messages(
            {
                "roleplay_task": "不明",
                "raw_transcript_hiragana": "えきはどこですか",
                "fluency_metrics": {"mora_per_sec": 5.0},
            },
            judge_id="A",
            model_family="anthropic",
        )

        self.assertIn("predicted_cefr_level", messages["system"])
        self.assertIn("A1, A2, B1, B2, C1, C2", messages["system"])
        self.assertIn("transcript:", messages["system"])
        self.assertIn("metrics:", messages["system"])
        self.assertIn("boundary:", messages["system"])
        self.assertIn("えきはどこですか", messages["user"])


class OfficialJfsSampleTests(unittest.TestCase):
    def test_official_catalog_is_valid_and_documents_level_coverage(self) -> None:
        catalog = load_jfs_catalog()

        validate_jfs_catalog(catalog)

        levels = {sample["human_cefr"] for sample in catalog["samples"]}
        self.assertEqual(levels, {"A2", "B1", "B2", "C1"})
        self.assertEqual(len(catalog["samples"]), 13)
        self.assertIn("C2", catalog["coverage_note"])
        self.assertTrue(
            all(
                sample["audio_url"].startswith("https://www.jfstandard.jpf.go.jp/")
                for sample in catalog["samples"]
            )
        )

    def test_download_jfs_assets_uses_local_external_paths(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = download_jfs_assets(
                output_dir=root,
                audio_dir=root / "audio",
                include_pdfs=False,
                fetch_bytes=lambda url: f"payload for {url}".encode("utf-8"),
            )

            self.assertEqual(result["sample_count"], 13)
            self.assertEqual(result["asset_count"], 13)
            first = root / "audio" / "A2_nijumaru01_JFS_RPT.mp3"
            self.assertTrue(first.exists())
            self.assertEqual(first.read_bytes()[:7], b"payload")

    def test_build_jfs_tuning_dataset_keeps_source_metadata(self) -> None:
        from tempfile import TemporaryDirectory

        catalog = load_jfs_catalog()
        first = catalog["samples"][0]
        with TemporaryDirectory() as tmp:
            audio_dir = Path(tmp)
            (audio_dir / first["audio_filename"]).write_bytes(b"fake mp3")

            dataset = build_jfs_tuning_dataset(
                audio_dir=audio_dir,
                require_audio=False,
            )

        self.assertEqual(len(dataset["items"]), 1)
        item = dataset["items"][0]
        self.assertEqual(item["sample_id"], first["sample_id"])
        self.assertEqual(item["human_cefr"], "A2")
        self.assertEqual(item["human_rating"], "◎")
        self.assertIn("source", item)
        self.assertEqual(len(dataset["missing_audio"]), 12)


class MockJudgeTests(unittest.TestCase):
    def test_mock_judges_return_three_schema_compatible_results(self) -> None:
        results = judge_with_mock_panel(
            {
                "raw_transcript_hiragana": "すみませんえきはどこですかまっすぐですか",
                "fluency_metrics": {
                    "speech_ratio_pct": 70,
                    "mora_per_sec": 5.0,
                    "max_pause_sec": 1.2,
                },
                "expected_hiragana_keywords": ["えき", "どこ"],
            }
        )

        self.assertEqual(len(results), 3)
        self.assertEqual({result.judge_id for result in results}, {"A", "B", "C"})
        self.assertTrue(all(result.risk_flags for result in results))

    def test_mock_auto_cefr_returns_three_level_predictions(self) -> None:
        results = judge_auto_cefr_with_mock_panel(
            {
                "raw_transcript_hiragana": "わたしはきのうともだちとえきにいきました" * 10,
                "fluency_metrics": {
                    "speech_ratio_pct": 70,
                    "mora_per_sec": 4.5,
                    "max_pause_sec": 1.5,
                },
            }
        )

        self.assertEqual(len(results), 3)
        self.assertEqual({result.judge_id for result in results}, {"A", "B", "C"})
        self.assertTrue(all(result.predicted_cefr_level for result in results))


class LiveJudgeConfigTests(unittest.TestCase):
    def test_parse_provider_specs_accepts_unique_one_to_three_entries(self) -> None:
        specs = parse_provider_specs(
            "anthropic:claude-sonnet-4-6,openai:gpt-5.4-mini"
        )

        self.assertEqual(
            specs,
            [
                ProviderSpec("anthropic", "claude-sonnet-4-6"),
                ProviderSpec("openai", "gpt-5.4-mini"),
            ],
        )

    def test_parse_provider_specs_rejects_too_many_entries(self) -> None:
        with self.assertRaises(ValueError):
            parse_provider_specs(
                "anthropic:a,openai:b,gemini:c,xai:d"
            )

    def test_parse_provider_specs_rejects_duplicate_providers(self) -> None:
        with self.assertRaises(ValueError):
            parse_provider_specs("openai:gpt-5.4-mini,openai:gpt-5.4")

    def test_load_judge_console_config_reads_three_enabled_judges(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge_llms.json"
            path.write_text(
                json.dumps(
                    {
                        "judge_mode": "live",
                        "audio_dir": "audio",
                        "profile": "tuning_profiles/base.json",
                        "live_judges": [
                            {
                                "judge_id": "A",
                                "enabled": True,
                                "provider": "anthropic",
                                "model": "claude-sonnet-4-6",
                            },
                            {
                                "judge_id": "B",
                                "enabled": True,
                                "provider": "openai",
                                "model": "gpt-5.4-mini",
                            },
                            {
                                "judge_id": "C",
                                "enabled": True,
                                "provider": "gemini",
                                "model": "gemini-3.1-pro-preview",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_judge_console_config(path)

        self.assertEqual(config.judge_mode, "live")
        self.assertEqual(config.audio_dir, Path("audio"))
        self.assertEqual(
            list(config.judge_providers),
            [
                ProviderSpec("anthropic", "claude-sonnet-4-6"),
                ProviderSpec("openai", "gpt-5.4-mini"),
                ProviderSpec("gemini", "gemini-3.1-pro-preview"),
            ],
        )

    def test_load_judge_console_config_skips_disabled_judges(self) -> None:
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "judge_llms.json"
            path.write_text(
                json.dumps(
                    {
                        "judge_mode": "live",
                        "live_judges": [
                            {
                                "judge_id": "A",
                                "enabled": True,
                                "provider": "anthropic",
                                "model": "claude-sonnet-4-6",
                            },
                            {
                                "judge_id": "B",
                                "enabled": False,
                                "provider": "openai",
                                "model": "gpt-5.4-mini",
                            },
                        ],
                    }
                ),
                encoding="utf-8",
            )

            config = load_judge_console_config(path)

        self.assertEqual(
            list(config.judge_providers),
            [ProviderSpec("anthropic", "claude-sonnet-4-6")],
        )

    def test_extract_json_object_accepts_fenced_response(self) -> None:
        self.assertEqual(
            _extract_json_object('```json\n{"rating": "○"}\n```'),
            '{"rating": "○"}',
        )

    def test_missing_key_envs_reports_provider_keys(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {}, clear=True):
            missing = missing_key_envs(
                [
                    ProviderSpec("anthropic", "claude-sonnet-4-6"),
                    ProviderSpec("openai", "gpt-5.4-mini"),
                    ProviderSpec("gemini", "gemini-3.1-pro-preview"),
                ]
            )

        self.assertEqual(
            missing,
            ["ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY/GOOGLE_API_KEY"],
        )

    def test_provider_key_status_flags_invalid_gemini_key_shape(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"GEMINI_API_KEY": "not-google-ai-studio-key"}, clear=True):
            status = provider_key_status("gemini")

        self.assertFalse(status.ok)
        self.assertEqual(status.state, "invalid")
        self.assertIn("Google AI Studio", status.message)

    def test_provider_key_state_display_shows_set_missing_and_invalid(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"OPENAI_API_KEY": "ok"}, clear=True):
            with patch("jgrade_eval.live_judges._get_json", return_value={"data": []}):
                self.assertEqual(
                    _format_provider_key_state(ProviderSpec("openai", "gpt-5.4-mini")),
                    "key=valid (OPENAI_API_KEY)",
                )
            self.assertEqual(
                _format_provider_key_state(ProviderSpec("anthropic", "claude-sonnet-4-6")),
                "key=missing (ANTHROPIC_API_KEY)",
            )
        with patch.dict(os.environ, {"GEMINI_API_KEY": "not-google-ai-studio-key"}, clear=True):
            self.assertEqual(
                _format_provider_key_state(ProviderSpec("gemini", "gemini-3.1-pro-preview")),
                "key=invalid (GEMINI_API_KEY)",
            )

    def test_validate_provider_key_confirms_openai_key_with_provider_api(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"OPENAI_API_KEY": "ok"}, clear=True):
            with patch("jgrade_eval.live_judges._get_json", return_value={"data": []}) as get_json:
                status = validate_provider_key(ProviderSpec("openai", "gpt-5.4-mini"))

        self.assertTrue(status.ok)
        self.assertEqual(status.state, "valid")
        self.assertEqual(status.env_name, "OPENAI_API_KEY")
        self.assertEqual(get_json.call_count, 1)

    def test_validate_provider_key_marks_auth_failure_invalid(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "bad"}, clear=True):
            with patch(
                "jgrade_eval.live_judges._get_json",
                side_effect=ProviderKeyValidationHTTPError(401, "unauthorized"),
            ):
                status = validate_provider_key(ProviderSpec("anthropic", "claude-sonnet-4-6"))

        self.assertFalse(status.ok)
        self.assertEqual(status.state, "invalid")

    def test_validate_provider_key_keeps_network_errors_unchecked(self) -> None:
        import os
        from unittest.mock import patch

        with patch.dict(os.environ, {"GROQ_API_KEY": "ok"}, clear=True):
            with patch("jgrade_eval.live_judges._get_json", side_effect=RuntimeError("timeout")):
                status = validate_provider_key(ProviderSpec("groq", "llama-3.3-70b-versatile"))

        self.assertFalse(status.ok)
        self.assertEqual(status.state, "unchecked")

    def test_partial_live_panel_keeps_successful_judges(self) -> None:
        from unittest.mock import patch

        def fake_call_provider(spec, messages, *, timeout_sec):
            if spec.provider == "gemini":
                raise RuntimeError("Provider HTTP error 401: UNAUTHENTICATED")
            return (
                '{"predicted_cefr_level":"B1","task_rating":"○",'
                '"confidence":0.7,"rationale":"test","evidence":[],"risk_flags":[]}'
            )

        with patch("jgrade_eval.live_judges._call_provider", side_effect=fake_call_provider):
            results, failures = judge_auto_cefr_with_live_panel_partial(
                {
                    "roleplay_task": "不明",
                    "raw_transcript_hiragana": "えきはどこですか",
                    "fluency_metrics": {"mora_per_sec": 5.0},
                },
                [
                    ProviderSpec("openai", "gpt-5.4-mini"),
                    ProviderSpec("gemini", "gemini-3.1-pro-preview"),
                ],
                timeout_sec=1,
            )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].judge_id, "A")
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].judge_id, "B")
        self.assertIn("Gemini認証に失敗", failures[0].message)

    def test_partial_live_panel_retries_invalid_json_once(self) -> None:
        from unittest.mock import patch

        calls = {"gemini": 0}
        gemini_user_messages = []

        def fake_call_provider(spec, messages, *, timeout_sec):
            if spec.provider == "gemini":
                calls["gemini"] += 1
                gemini_user_messages.append(messages["user"])
                if calls["gemini"] == 1:
                    return '{"predicted_cefr_level":"C1",'
            return (
                '{"predicted_cefr_level":"C1","task_rating":"○",'
                '"confidence":0.7,"rationale":"test","evidence":[],"risk_flags":[]}'
            )

        with patch("jgrade_eval.live_judges._call_provider", side_effect=fake_call_provider):
            results, failures = judge_auto_cefr_with_live_panel_partial(
                {
                    "roleplay_task": "不明",
                    "raw_transcript_hiragana": "しゃかいもんだいについてながくはなします",
                    "fluency_metrics": {"mora_per_sec": 6.0},
                },
                [
                    ProviderSpec("anthropic", "claude-sonnet-4-6"),
                    ProviderSpec("openai", "gpt-5.4-mini"),
                    ProviderSpec("gemini", "gemini-3.1-pro-preview"),
                ],
                timeout_sec=1,
            )

        self.assertEqual(len(results), 3)
        self.assertEqual(failures, [])
        self.assertEqual(calls["gemini"], 2)
        self.assertIn("前回の応答はJSONとして解析できませんでした", gemini_user_messages[-1])


class InteractiveCliTests(unittest.TestCase):
    def test_list_audio_files_filters_supported_extensions(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.mp3").write_bytes(b"")
            (root / "b.wav").write_bytes(b"")
            (root / "note.txt").write_text("x")

            self.assertEqual(
                [path.name for path in list_audio_files(root)],
                ["a.mp3", "b.wav"],
            )

    def test_normalize_audio_path_input_handles_quotes_and_escaped_spaces(self) -> None:
        self.assertEqual(
            _normalize_audio_path_input("'/Users/test/Desktop/my audio.mp3'"),
            Path("/Users/test/Desktop/my audio.mp3"),
        )
        self.assertEqual(
            _normalize_audio_path_input("/Users/test/Desktop/my\\ audio.mp3"),
            Path("/Users/test/Desktop/my audio.mp3"),
        )

    def test_prompt_provider_specs_hides_used_provider_and_can_skip(self) -> None:
        import io
        import os
        from contextlib import redirect_stdout
        from unittest.mock import patch

        # A: anthropic model 1. B: first remaining provider (openai), model 1.
        # C: skip. If anthropic appeared again for B, this input would select it
        # and the assertion below would fail.
        env = {"ANTHROPIC_API_KEY": "ok", "OPENAI_API_KEY": "ok"}
        def valid_status(spec):
            env_name = f"{spec.provider.upper()}_API_KEY"
            if spec.provider == "anthropic":
                env_name = "ANTHROPIC_API_KEY"
            return ProviderKeyStatus(spec.provider, env_name, "valid", "ok")

        with patch.dict(os.environ, env, clear=True):
            with patch("jgrade_eval.interactive.validate_provider_key", side_effect=valid_status):
                with patch("builtins.input", side_effect=["1", "1", "1", "1", "4"]):
                    output = io.StringIO()
                    with redirect_stdout(output):
                        specs = prompt_provider_specs()

        self.assertEqual(
            specs,
            [
                ProviderSpec("anthropic", "claude-sonnet-4-6"),
                ProviderSpec("openai", "gpt-5.4-mini"),
            ],
        )
        self.assertIn("anthropic (ANTHROPIC_API_KEY) [key=valid", output.getvalue())
        self.assertIn("openai (OPENAI_API_KEY) [key=valid", output.getvalue())

    def test_prompt_provider_specs_can_skip_invalid_provider_key(self) -> None:
        import io
        import os
        from contextlib import redirect_stdout
        from unittest.mock import patch

        with patch.dict(os.environ, {"GEMINI_API_KEY": "bad-key", "OPENAI_API_KEY": "ok"}, clear=True):
            def status_for_skip(spec):
                if spec.provider == "gemini":
                    return ProviderKeyStatus("gemini", "GEMINI_API_KEY", "invalid", "bad")
                return ProviderKeyStatus(spec.provider, f"{spec.provider.upper()}_API_KEY", "valid", "ok")

            with patch("jgrade_eval.interactive.validate_provider_key", side_effect=status_for_skip):
                with patch("builtins.input", side_effect=["3", "1", "2", "2", "1", "5"]):
                    with redirect_stdout(io.StringIO()):
                        specs = prompt_provider_specs()

        self.assertEqual(specs, [ProviderSpec("openai", "gpt-5.4-mini")])

    def test_upsert_env_value_updates_existing_key_without_printing_value(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("GEMINI_API_KEY=old\nOPENAI_API_KEY=x\n", encoding="utf-8")
            _upsert_env_value(path, "GEMINI_API_KEY", "AIza-new")

            text = path.read_text(encoding="utf-8")

        self.assertIn("GEMINI_API_KEY=AIza-new", text)
        self.assertIn("OPENAI_API_KEY=x", text)
        self.assertNotIn("GEMINI_API_KEY=old", text)

    def test_print_objective_data_separates_fluency_and_range_evidence(self) -> None:
        import io
        from contextlib import redirect_stdout

        objective_data = {
            "audio_path": "sample.wav",
            "stt_model": "test-stt",
            "vad_model": "test-vad",
            "raw_transcript_hiragana": "わたしはすしがすきです",
            "fluency_metrics": {
                "audio_duration_sec": 3.0,
                "speech_sec": 2.0,
                "speech_ratio_pct": 66.7,
                "pause_count": 1,
                "avg_pause_sec": 0.4,
                "max_pause_sec": 0.4,
                "mora_count": 11,
                "mora_per_sec": 5.5,
                "fluency_grade": "B",
            },
            "top_pauses": [],
            "range_data": {
                "tokens": [
                    {
                        "surface": "わたし",
                        "dictionary_form": "私",
                        "reading_hiragana": "わたし",
                        "part_of_speech": ["代名詞"],
                        "is_lexical": True,
                        "jlpt_candidates": [{"surface": "私", "jlpt_level": "N5"}],
                        "selected_jlpt_level": "N5",
                    }
                ],
                "statistics": {
                    "token_count": 6,
                    "lexical_token_count": 3,
                    "unique_lemma_count": 3,
                    "ttr": 1.0,
                    "known_token_count": 2,
                    "unknown_token_count": 1,
                    "unknown_token_rate": 0.3333,
                },
                "jlpt_distribution": {
                    "N5": {"token_count": 2, "unique_lemma_count": 2},
                    "N4": {"token_count": 0, "unique_lemma_count": 0},
                    "N3": {"token_count": 0, "unique_lemma_count": 0},
                    "N2": {"token_count": 0, "unique_lemma_count": 0},
                    "N1": {"token_count": 0, "unique_lemma_count": 0},
                },
                "ambiguity_count": 1,
                "dictionary_version": "test-dict",
                "tokenizer_version": "test-tokenizer",
                "split_mode": "A",
            },
        }

        output = io.StringIO()
        with redirect_stdout(output):
            print_objective_data(objective_data)

        rendered = output.getvalue()
        self.assertIn("=== Fluency客観データ ===", rendered)
        self.assertIn("=== Range客観データ ===", rendered)
        self.assertIn("語彙TTR: 1.0", rendered)
        self.assertIn("JLPT N5: token=2 / unique_lemma=2", rendered)
        self.assertNotIn("=== 客観データ ===", rendered)

    def test_interactive_prints_shared_and_accuracy_evidence_without_prompting_for_user_cefr(self) -> None:
        import io
        from contextlib import redirect_stdout

        objective_data = {
            "audio_path": "sample.wav",
            "stt_model": "test-stt",
            "vad_model": "test-vad",
            "raw_transcript_hiragana": "わたしはすしがすきです",
            "fluency_metrics": {
                "audio_duration_sec": 3.0,
                "speech_sec": 2.0,
                "speech_ratio_pct": 66.7,
                "pause_count": 1,
                "avg_pause_sec": 0.4,
                "max_pause_sec": 0.4,
                "mora_count": 11,
                "mora_per_sec": 5.5,
                "fluency_grade": "B",
            },
            "top_pauses": [],
        }
        range_data = {
            "tokens": [],
            "statistics": {
                "token_count": 0,
                "lexical_token_count": 0,
                "unique_lemma_count": 0,
                "ttr": 0.0,
                "known_token_count": 0,
                "unknown_token_count": 0,
                "unknown_token_rate": 0.0,
            },
            "jlpt_distribution": {level: {"token_count": 0, "unique_lemma_count": 0} for level in ("N5", "N4", "N3", "N2", "N1")},
            "ambiguity_count": 0,
            "dictionary_version": "test-dict",
            "tokenizer_version": "test-tokenizer",
            "split_mode": "A",
        }
        fluency_extractor = Mock()
        fluency_extractor.extract.return_value = objective_data
        range_extractor = Mock()
        range_extractor.analyze.return_value = range_data
        judge_inputs: list[dict] = []

        def capture_judge_input(roleplay_input: dict):
            judge_inputs.append(roleplay_input)
            return judge_auto_cefr_with_mock_panel(roleplay_input)

        with (
            patch("jgrade_eval.interactive.prompt_judge_setup", return_value=("mock", [])),
            patch("jgrade_eval.interactive.prompt_audio_choice", return_value=Path("sample.wav")),
            patch("jgrade_eval.interactive.FluencyExtractor", return_value=fluency_extractor),
            patch("jgrade_eval.fact_modules.RangeExtractor.default", return_value=range_extractor),
            patch("jgrade_eval.interactive.judge_auto_cefr_with_mock_panel", side_effect=capture_judge_input),
            patch("jgrade_eval.interactive.prompt_user_cefr_level", return_value=None) as prompt_level,
        ):
            output = io.StringIO()
            with redirect_stdout(output):
                run_interactive(
                    audio_dir=Path("."),
                    judge_mode="mock",
                    provider_specs=[],
                    timeout_sec=1.0,
                    review_store_path=None,
                )

        range_extractor.analyze.assert_called_once_with("わたしはすしがすきです")
        self.assertEqual(judge_inputs[0]["range_data"], range_data)
        self.assertEqual(judge_inputs[0]["accuracy_data"]["module_id"], "accuracy")
        self.assertEqual(judge_inputs[0]["coherence_data"]["module_id"], "coherence")
        prompt_level.assert_not_called()
        self.assertIn("[1/7] 共通Evidence層", output.getvalue())
        self.assertIn("[2/7] Fluencyモジュール", output.getvalue())
        self.assertIn("[3/7] Rangeモジュール", output.getvalue())
        self.assertIn("[4/7] Accuracyモジュール", output.getvalue())
        self.assertIn("[5/7] Coherenceモジュール", output.getvalue())
        self.assertIn("=== 共通Evidence層客観データ ===", output.getvalue())
        self.assertIn("=== Accuracy客観データ ===", output.getvalue())
        self.assertIn("=== Coherence客観データ ===", output.getvalue())


class TuningTests(unittest.TestCase):
    def test_tuning_runner_uses_embedded_objective_data_and_reports_metrics(self) -> None:
        dataset = {
            "name": "unit",
            "items": [
                {
                    "sample_id": "s1",
                    "audio_path": "unused.mp3",
                    "human_cefr": "B1",
                    "human_rating": "○",
                    "objective_data": {
                        "raw_transcript_hiragana": "わたしはきのうともだちとえきにいきました" * 14,
                        "fluency_metrics": {
                            "audio_duration_sec": 60,
                            "speech_ratio_pct": 70,
                            "mora_per_sec": 4.5,
                            "max_pause_sec": 1.5,
                        },
                    },
                }
            ],
        }

        result = run_tuning_dataset(dataset, TuningProfile.default(), base_dir=Path.cwd())

        self.assertEqual(result["metrics"]["n_evaluable"], 1)
        self.assertEqual(result["records"][0]["status"], "ok")
        self.assertEqual(result["records"][0]["predicted_cefr"], "B1")
        self.assertEqual(result["metrics"]["accuracy"], 1.0)

    def test_tuning_metrics_counts_adjacent_predictions(self) -> None:
        metrics = compute_cefr_metrics(
            [
                {"status": "ok", "sample_id": "s1", "human_cefr": "B1", "predicted_cefr": "A2"},
                {"status": "ok", "sample_id": "s2", "human_cefr": "B2", "predicted_cefr": "B2"},
                {"status": "error", "sample_id": "s3"},
            ]
        )

        self.assertEqual(metrics["n_total"], 3)
        self.assertEqual(metrics["n_failed"], 1)
        self.assertEqual(metrics["accuracy"], 0.5)
        self.assertEqual(metrics["adjacent_accuracy"], 1.0)
        self.assertTrue(is_adjacent_level("B1", "A2"))

    def test_tuning_profile_round_trips(self) -> None:
        from tempfile import TemporaryDirectory

        profile = TuningProfile(
            name="p1",
            judge_mode="live",
            judge_providers=(ProviderSpec("openai", "gpt-5.4-mini"),),
            notes="test",
        )
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "profile.json"
            save_profile(profile, path)
            loaded = load_profile(path)

        self.assertEqual(loaded.name, "p1")
        self.assertEqual(loaded.judge_mode, "live")
        self.assertEqual(loaded.judge_providers, (ProviderSpec("openai", "gpt-5.4-mini"),))

    def test_composed_profile_prompt_adds_required_evidence_contract(self) -> None:
        prompt = compose_auto_cefr_system_prompt(
            TuningProfile(auto_cefr_system_prompt="古い保存済みプロンプト")
        )

        self.assertIn("transcript:", prompt)
        self.assertIn("metrics:", prompt)
        self.assertIn("boundary:", prompt)

    def test_level_correction_is_used_as_calibration_on_next_run(self) -> None:
        dataset = {
            "name": "unit",
            "items": [
                {
                    "sample_id": "s1",
                    "audio_path": "unused.mp3",
                    "human_cefr": "B2",
                    "objective_data": {
                        "raw_transcript_hiragana": "わたしはきのうともだちとえきにいきました" * 14,
                        "fluency_metrics": {
                            "audio_duration_sec": 60,
                            "speech_ratio_pct": 70,
                            "mora_per_sec": 4.5,
                            "max_pause_sec": 1.5,
                        },
                    },
                }
            ],
        }
        first = run_tuning_dataset(dataset, TuningProfile.default(), base_dir=Path.cwd())
        tuned = apply_level_correction(
            TuningProfile.default(),
            record=first["records"][0],
            corrected_cefr="B2",
            note="teacher correction",
        )

        second = run_tuning_dataset(dataset, tuned, base_dir=Path.cwd())

        self.assertEqual(second["records"][0]["raw_predicted_cefr"], "B1")
        self.assertEqual(second["records"][0]["predicted_cefr"], "B2")
        self.assertTrue(second["records"][0]["calibration_applied"])
        self.assertIn("B1->B2", second["records"][0]["deliberation_conclusion"])
        self.assertEqual(second["metrics"]["accuracy"], 1.0)
        self.assertIn("teacher correction", compose_auto_cefr_system_prompt(tuned))
        self.assertIn("B1->B2", compose_auto_cefr_system_prompt(tuned))

    def test_prompt_user_cefr_level_can_select_or_skip(self) -> None:
        import io
        from contextlib import redirect_stdout
        from unittest.mock import patch

        with patch("builtins.input", return_value="4"):
            with redirect_stdout(io.StringIO()):
                selected = prompt_user_cefr_level("B1")
        with patch("builtins.input", return_value="7"):
            with redirect_stdout(io.StringIO()):
                skipped = prompt_user_cefr_level("B1")

        self.assertEqual(selected, "B2")
        self.assertIsNone(skipped)

    def test_console_level_feedback_saves_calibration_example(self) -> None:
        from tempfile import TemporaryDirectory

        record = {
            "sample_id": "sample-1",
            "predicted_cefr": "B1",
            "raw_predicted_cefr": "B1",
            "objective_data": {
                "raw_transcript_hiragana": "こんにちは",
                "fluency_metrics": {"mora_per_sec": 4.0},
            },
        }
        with TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "profile.json"
            updated = apply_and_save_user_level_correction(
                profile=TuningProfile.default(),
                profile_path=profile_path,
                record=record,
                corrected_level="B2",
            )
            loaded = load_profile(profile_path)

        self.assertNotIn("sample-1", updated.level_overrides)
        self.assertNotIn("sample-1", loaded.level_overrides)
        self.assertEqual(loaded.metadata["level_correction_stats"]["B1->B2"], 1)
        self.assertEqual(loaded.tuning_examples[-1]["corrected_cefr"], "B2")
        self.assertEqual(loaded.tuning_examples[-1]["note"], "console final level feedback")

    def test_profile_history_keeps_last_ten_and_can_restore(self) -> None:
        profile = TuningProfile.default()
        for index in range(12):
            changed = TuningProfile(name=f"p{index}")
            profile = with_history(changed, previous=profile, summary=f"change {index}")

        self.assertEqual(len(profile.change_history), 10)
        self.assertEqual(profile.change_history[0]["summary"], "change 11")

        restored = restore_history(profile, 0)

        self.assertEqual(restored.name, "p10")
        self.assertEqual(restored.change_history[0]["summary"][:11], "rollback to")

    def test_review_dataset_upserts_judged_samples(self) -> None:
        from tempfile import TemporaryDirectory

        objective = {
            "raw_transcript_hiragana": "こんにちはきょうはえきでともだちにあいました",
            "fluency_metrics": {"mora_per_sec": 4.0},
        }
        sample_id = make_review_sample_id(Path("/tmp/audio.wav"), objective)
        record = {
            "sample_id": sample_id,
            "status": "ok",
            "audio_path": "/tmp/audio.wav",
            "predicted_cefr": "A2",
            "raw_predicted_cefr": "A2",
            "objective_data": objective,
            "judge_results": [],
            "judge_failures": [],
        }

        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "review_samples.json"
            first = append_judged_sample(path, record=record, source="test")
            second_record = {**record, "predicted_cefr": "B1"}
            second = append_judged_sample(path, record=second_record, source="test")

        self.assertEqual(len(first["items"]), 1)
        self.assertEqual(len(second["items"]), 1)
        self.assertEqual(second["items"][0]["last_prediction"]["predicted_cefr"], "B1")
        self.assertEqual(len(second["items"][0]["prediction_history"]), 2)


if __name__ == "__main__":
    unittest.main()
