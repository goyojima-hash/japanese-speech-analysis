from pathlib import Path
import unittest
from unittest.mock import patch

from starlette.testclient import TestClient

from jgrade_eval.api import EVALUATIONS, app
from jgrade_eval.api_service import evaluate_speech_level
from jgrade_eval.mock_judges import judge_auto_cefr_with_mock_panel


class FakeExtractor:
    def extract(self, audio_path: Path) -> dict:
        return {
            "audio_path": str(audio_path),
            "raw_transcript_hiragana": "わたしはきのうともだちとえきにいきました" * 14,
            "raw_transcript_romaji": "watashi",
            "fluency_metrics": {
                "audio_duration_sec": 60.0,
                "speech_sec": 42.0,
                "speech_ratio_pct": 70.0,
                "pause_total_sec": 3.0,
                "pause_count": 3,
                "avg_pause_sec": 1.0,
                "max_pause_sec": 1.5,
                "mora_count": 270,
                "mora_per_sec": 4.5,
                "fluency_grade": "B",
            },
            "top_pauses": [],
            "speech_segments": [],
            "extraction_time_sec": 0.01,
            "stt_model": "fake",
            "vad_model": "fake",
        }


class FakeDownloadResponse:
    content = b"fake audio"

    def raise_for_status(self) -> None:
        return None


class FakeRangeExtractor:
    def __init__(self) -> None:
        self.transcripts: list[str] = []

    def analyze(self, transcript: str) -> dict:
        self.transcripts.append(transcript)
        return {
            "tokens": [],
            "statistics": {"token_count": 7, "ttr": 0.5},
            "jlpt_distribution": {},
            "ambiguity_count": 0,
            "dictionary_version": "fake-range-dictionary",
            "tokenizer_version": "fake-tokenizer",
            "split_mode": "A",
        }


class JGradeApiTests(unittest.TestCase):
    def test_service_exposes_range_data_to_response_and_judges(self) -> None:
        range_extractor = FakeRangeExtractor()
        judge_inputs: list[dict] = []

        def capture_judge_input(roleplay_input: dict):
            judge_inputs.append(roleplay_input)
            return judge_auto_cefr_with_mock_panel(roleplay_input)

        with patch("jgrade_eval.api_service.judge_auto_cefr_with_mock_panel", side_effect=capture_judge_input):
            result = evaluate_speech_level(
                Path("sample.mp3"),
                judge_mode="mock",
                extractor=FakeExtractor(),
                range_extractor=range_extractor,
            )

        self.assertEqual(range_extractor.transcripts, [
            "わたしはきのうともだちとえきにいきました" * 14
        ])
        self.assertEqual(result["objective_data"]["range_data"]["dictionary_version"], "fake-range-dictionary")
        self.assertEqual(judge_inputs[0]["range_data"], result["objective_data"]["range_data"])
        self.assertNotIn("fluency_grade", result["objective_data"]["fluency_metrics"])
        self.assertEqual(result["objective_data"]["evidence_schema_version"], "evidence.v1")
        self.assertEqual(result["fact_modules"], ["accuracy", "coherence", "fluency", "range"])
        self.assertIn("accuracy_data", result["objective_data"])
        self.assertIn("coherence_data", result["objective_data"])
        self.assertIn("accuracy_data", judge_inputs[0])
        self.assertIn("coherence_data", judge_inputs[0])

    def test_service_can_select_accuracy_without_running_range(self) -> None:
        range_extractor = FakeRangeExtractor()
        judge_inputs: list[dict] = []

        def capture_judge_input(roleplay_input: dict):
            judge_inputs.append(roleplay_input)
            return judge_auto_cefr_with_mock_panel(roleplay_input)

        with patch("jgrade_eval.api_service.judge_auto_cefr_with_mock_panel", side_effect=capture_judge_input):
            result = evaluate_speech_level(
                Path("sample.mp3"),
                judge_mode="mock",
                extractor=FakeExtractor(),
                range_extractor=range_extractor,
                selected_modules=("accuracy",),
            )

        self.assertEqual(range_extractor.transcripts, [])
        self.assertIn("accuracy_data", result["objective_data"])
        self.assertNotIn("range_data", result["objective_data"])
        self.assertIn("accuracy_data", judge_inputs[0])

    def test_service_can_select_coherence_without_running_range(self) -> None:
        range_extractor = FakeRangeExtractor()
        judge_inputs: list[dict] = []

        def capture_judge_input(roleplay_input: dict):
            judge_inputs.append(roleplay_input)
            return judge_auto_cefr_with_mock_panel(roleplay_input)

        with patch("jgrade_eval.api_service.judge_auto_cefr_with_mock_panel", side_effect=capture_judge_input):
            result = evaluate_speech_level(
                Path("sample.mp3"),
                judge_mode="mock",
                extractor=FakeExtractor(),
                range_extractor=range_extractor,
                selected_modules=("coherence",),
            )

        self.assertEqual(range_extractor.transcripts, [])
        self.assertIn("coherence_data", result["objective_data"])
        self.assertNotIn("range_data", result["objective_data"])
        self.assertIn("coherence_data", judge_inputs[0])

    def test_service_rejects_unknown_fact_module(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported fact module"):
            evaluate_speech_level(
                Path("sample.mp3"),
                judge_mode="mock",
                extractor=FakeExtractor(),
                selected_modules=("not-a-module",),
            )

    def test_create_speech_level_evaluation_returns_completed_result(self) -> None:
        EVALUATIONS.clear()
        client = TestClient(app)

        with patch("jgrade_eval.api_service.FluencyExtractor", return_value=FakeExtractor()):
            response = client.post(
                "/api/v1/speech-level-evaluations",
                data={
                    "external_id": "sample-1",
                    "language": "ja",
                    "judge_mode": "mock",
                    "roleplay_task": "駅で友達との出来事を説明する。",
                },
                files={"audio": ("sample.mp3", b"fake audio", "audio/mpeg")},
            )

        self.assertEqual(response.status_code, 201)
        data = response.json()["data"]
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["external_id"], "sample-1")
        self.assertEqual(data["final_cefr_level"], "B1")
        self.assertEqual(data["consensus"]["judge_count"], 3)
        self.assertTrue(data["objective_data"]["raw_transcript_hiragana"].startswith("わたし"))
        self.assertEqual(data["reasons"][0]["type"], "transcript")

        fetched = client.get(f"/api/v1/speech-level-evaluations/{data['id']}")
        self.assertEqual(fetched.status_code, 200)
        self.assertEqual(fetched.json()["data"]["id"], data["id"])

    def test_create_speech_level_evaluation_rejects_non_japanese_language(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/api/v1/speech-level-evaluations",
            data={"language": "en", "judge_mode": "mock"},
            files={"audio": ("sample.mp3", b"fake audio", "audio/mpeg")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "validation_error")

    def test_create_speech_level_evaluation_accepts_json_audio_url(self) -> None:
        EVALUATIONS.clear()
        client = TestClient(app)

        with (
            patch("jgrade_eval.api.requests.get", return_value=FakeDownloadResponse()),
            patch("jgrade_eval.api_service.FluencyExtractor", return_value=FakeExtractor()),
        ):
            response = client.post(
                "/api/v1/speech-level-evaluations",
                json={
                    "audio_url": "https://example.com/sample.mp3",
                    "external_id": "json-sample",
                    "judge_mode": "mock",
                    "include_objective_data": False,
                },
            )

        self.assertEqual(response.status_code, 201)
        data = response.json()["data"]
        self.assertEqual(data["external_id"], "json-sample")
        self.assertEqual(data["final_cefr_level"], "B1")
        self.assertNotIn("objective_data", data)

    def test_http_api_can_select_accuracy_fact_module(self) -> None:
        EVALUATIONS.clear()
        client = TestClient(app)

        with (
            patch("jgrade_eval.api.requests.get", return_value=FakeDownloadResponse()),
            patch("jgrade_eval.api_service.FluencyExtractor", return_value=FakeExtractor()),
        ):
            response = client.post(
                "/api/v1/speech-level-evaluations",
                json={
                    "audio_url": "https://example.com/sample.mp3",
                    "judge_mode": "mock",
                    "fact_modules": ["accuracy"],
                },
            )

        self.assertEqual(response.status_code, 201)
        data = response.json()["data"]
        self.assertEqual(data["fact_modules"], ["accuracy"])
        self.assertIn("accuracy_data", data["objective_data"])
        self.assertNotIn("range_data", data["objective_data"])


if __name__ == "__main__":
    unittest.main()
