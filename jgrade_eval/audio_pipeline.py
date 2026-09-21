from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .consensus import ConsensusGate
from .live_judges import ProviderSpec, judge_with_live_panel
from .mock_judges import judge_with_mock_panel


class ObjectiveExtractionError(RuntimeError):
    """Raised when the upstream fluency/STT pipeline cannot produce objective data."""


class FluencyExtractor:
    """Extract objective transcript and timing metrics once models are loaded."""

    def __init__(self) -> None:
        # Keep the heavy STT/VAD imports lazy so unit tests and CLI help stay fast.
        try:
            from fluency import load_silero_vad, load_vumichien

            self.vad_model = load_silero_vad()
            self.v_model, self.v_proc, self.v_device = load_vumichien()
            self.vocab = self.v_proc.tokenizer.convert_ids_to_tokens(
                range(self.v_proc.tokenizer.vocab_size)
            )
            self.blank_id = self.v_proc.tokenizer.pad_token_id or 0
        except Exception as exc:
            raise ObjectiveExtractionError(_format_model_load_error(exc)) from exc

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "stt_model": "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
            "vad_model": "silero-vad",
        }

    def extract(self, audio_path: Path) -> dict[str, Any]:
        from fluency import (
            detect_speech_and_pauses,
            get_mora_timings,
            to_romaji,
            transcribe,
        )

        start = time.perf_counter()
        try:
            hiragana, all_logits, duration, audio = transcribe(
                self.v_model,
                self.v_proc,
                self.v_device,
                str(audio_path),
            )
            mora_timings = get_mora_timings(all_logits, self.blank_id, self.vocab)
            speech_ts, pauses = detect_speech_and_pauses(self.vad_model, audio, duration)
        except Exception as exc:
            raise ObjectiveExtractionError(
                f"音声ファイルの客観データ抽出に失敗しました: {exc}"
            ) from exc

        speech_sec = sum(segment["end"] - segment["start"] for segment in speech_ts)
        pause_total = sum(pause["duration"] for pause in pauses)
        speech_ratio_pct = speech_sec / duration * 100 if duration else 0.0
        mora_per_sec = len(mora_timings) / duration if duration else 0.0
        max_pause = max((pause["duration"] for pause in pauses), default=0.0)
        avg_pause = pause_total / len(pauses) if pauses else 0.0
        hiragana_clean = "".join(hiragana.split())

        return {
            "audio_path": str(audio_path),
            "raw_transcript_hiragana": hiragana_clean,
            "raw_transcript_romaji": to_romaji(hiragana_clean),
            "fluency_metrics": {
                "audio_duration_sec": round(duration, 2),
                "speech_sec": round(speech_sec, 2),
                "speech_ratio_pct": round(speech_ratio_pct, 2),
                "pause_total_sec": round(pause_total, 2),
                "pause_count": len(pauses),
                "avg_pause_sec": round(avg_pause, 2),
                "max_pause_sec": round(max_pause, 2),
                "mora_count": len(mora_timings),
                "mora_per_sec": round(mora_per_sec, 2),
            },
            "top_pauses": [
                {
                    "start": round(pause["start"], 2),
                    "end": round(pause["end"], 2),
                    "duration": round(pause["duration"], 2),
                }
                for pause in sorted(pauses, key=lambda item: item["duration"], reverse=True)[:3]
            ],
            "speech_segments": [
                {
                    "start": round(segment["start"], 2),
                    "end": round(segment["end"], 2),
                    "duration": round(segment["end"] - segment["start"], 2),
                }
                for segment in speech_ts
            ],
            "pause_segments": [
                {
                    "start": round(pause["start"], 2),
                    "end": round(pause["end"], 2),
                    "duration": round(pause["duration"], 2),
                }
                for pause in pauses
            ],
            "mora_timings": [
                {
                    "mora": str(timing["mora"]),
                    "start": round(float(timing["start"]), 4),
                    "end": round(float(timing["end"]), 4),
                }
                for timing in mora_timings
            ],
            "extraction_time_sec": round(time.perf_counter() - start, 2),
            "stt_model": "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
            "vad_model": "silero-vad",
        }


def evaluate_audio_manifest(
    manifest: dict[str, Any],
    *,
    base_dir: Path,
    judge_mode: str,
    provider_specs: list[ProviderSpec] | None = None,
    timeout_sec: float = 60.0,
) -> dict[str, Any]:
    if judge_mode not in {"mock", "live"}:
        raise ValueError("judge_mode must be 'mock' or 'live'.")
    if judge_mode == "live" and not provider_specs:
        raise ValueError("provider_specs are required when judge_mode='live'.")

    extractor = FluencyExtractor()
    roleplay_payloads = []
    roleplay_decisions = []
    gate = ConsensusGate()

    for roleplay in manifest["roleplays"]:
        audio_path = _resolve_audio_path(base_dir, Path(roleplay["audio_path"]))
        objective_data = extractor.extract(audio_path)
        roleplay_input = {
            "sample_id": manifest["sample_id"],
            "target_cefr_level": manifest["tested_level"],
            "roleplay_id": roleplay["roleplay_id"],
            "roleplay_task": roleplay["roleplay_task"],
            "jfs_can_do_criteria": roleplay.get("jfs_can_do_criteria", []),
            "raw_transcript_hiragana": objective_data["raw_transcript_hiragana"],
            "fluency_metrics": objective_data["fluency_metrics"],
            "speaker_metadata": {
                **manifest.get("speaker_metadata", {}),
                **roleplay.get("speaker_metadata", {}),
            },
            "optional_expected_information": roleplay.get("optional_expected_information", []),
            "expected_hiragana_keywords": roleplay.get("expected_hiragana_keywords", []),
        }
        if judge_mode == "mock":
            judge_results = judge_with_mock_panel(roleplay_input)
        else:
            judge_results = judge_with_live_panel(
                roleplay_input,
                provider_specs or [],
                timeout_sec=timeout_sec,
            )
        decision = gate.decide_roleplay(roleplay["roleplay_id"], judge_results)
        roleplay_decisions.append(decision)
        roleplay_payloads.append(
            {
                "roleplay_id": roleplay["roleplay_id"],
                "audio_path": str(audio_path),
                "objective_data": objective_data,
                "roleplay_input": roleplay_input,
                "judge_results": [result.to_dict() for result in judge_results],
                "consensus": decision.to_dict(),
            }
        )

    level_decision = gate.decide_level(manifest["tested_level"], roleplay_decisions)
    return {
        "sample_id": manifest["sample_id"],
        "tested_level": manifest["tested_level"],
        "judge_mode": judge_mode,
        "warning": _judge_mode_warning(judge_mode),
        "level_decision": level_decision.to_dict(),
        "roleplays": roleplay_payloads,
    }


def extract_objective_manifest(
    manifest: dict[str, Any],
    *,
    base_dir: Path,
) -> dict[str, Any]:
    """Extract only objective transcript/timing data from manifest audio files."""

    extractor = FluencyExtractor()
    roleplay_payloads = []

    for roleplay in manifest["roleplays"]:
        audio_path = _resolve_audio_path(base_dir, Path(roleplay["audio_path"]))
        objective_data = extractor.extract(audio_path)
        roleplay_payloads.append(
            {
                "roleplay_id": roleplay["roleplay_id"],
                "audio_path": str(audio_path),
                "roleplay_task": roleplay.get("roleplay_task", ""),
                "jfs_can_do_criteria": roleplay.get("jfs_can_do_criteria", []),
                "optional_expected_information": roleplay.get(
                    "optional_expected_information",
                    [],
                ),
                "objective_data": objective_data,
            }
        )

    return {
        "sample_id": manifest["sample_id"],
        "tested_level": manifest["tested_level"],
        "stage": "objective_data_extraction",
        "source_pipeline": {
            "module": "fluency.py",
            "transcript_function": "transcribe",
            "timing_functions": [
                "get_mora_timings",
                "detect_speech_and_pauses",
            ],
            "stt_model": "vumichien/wav2vec2-large-xlsr-japanese-hiragana",
            "vad_model": "silero-vad",
        },
        "roleplays": roleplay_payloads,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _resolve_audio_path(base_dir: Path, audio_path: Path) -> Path:
    resolved = audio_path if audio_path.is_absolute() else base_dir / audio_path
    if not resolved.exists():
        raise FileNotFoundError(f"audio file not found: {resolved}")
    return resolved


def _judge_mode_warning(judge_mode: str) -> str | None:
    if judge_mode == "mock":
        return (
            "mock judge is for local pipeline smoke tests only; use live "
            "provider adapters for valid JFS decisions"
        )
    return None


def _format_model_load_error(exc: Exception) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else repr(exc)
    return (
        "客観データ抽出モデルの読み込みに失敗しました。\n"
        "このCLIは先に fluency.py のWav2Vec2/STTモデルとsilero-VADを読み込みます。\n"
        "初回実行時はHugging Faceからモデルを取得できるネットワーク接続が必要です。\n"
        "一度キャッシュ済みなら、以後はローカルキャッシュを優先して読み込みます。\n"
        f"原因: {message}"
    )
