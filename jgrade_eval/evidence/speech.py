from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from .models import MoraTiming, SpeechEvidence, TimedSpan


_NON_FACTUAL_METRIC_KEYS = frozenset({"fluency_grade"})


class LegacySpeechExtractor(Protocol):
    def extract(self, audio_path: Path) -> dict[str, Any]: ...


class FluencySpeechEvidenceExtractor:
    """Adapt the current Fluency extractor to fact-only common evidence."""

    def __init__(self, extractor: LegacySpeechExtractor) -> None:
        self._extractor = extractor

    @property
    def provenance(self) -> dict[str, str]:
        legacy_provenance = getattr(self._extractor, "provenance", {})
        if not isinstance(legacy_provenance, dict):
            legacy_provenance = {}
        return {
            "adapter": type(self._extractor).__name__,
            **{str(key): str(value) for key, value in legacy_provenance.items()},
        }

    def extract(self, audio_path: Path) -> SpeechEvidence:
        return speech_evidence_from_legacy(self._extractor.extract(audio_path))


def speech_evidence_from_legacy(data: dict[str, Any]) -> SpeechEvidence:
    metrics = dict(data.get("fluency_metrics", {}))
    segments = tuple(_span(item) for item in data.get("speech_segments", []))
    pause_records = data.get("pause_segments", data.get("top_pauses", []))
    pauses = tuple(_span(item) for item in pause_records)
    timings = tuple(
        MoraTiming(mora=str(item["mora"]), start=float(item["start"]), end=float(item["end"]))
        for item in data.get("mora_timings", [])
    )
    provenance = {
        "stt_model": str(data.get("stt_model", "unknown")),
        "vad_model": str(data.get("vad_model", "unknown")),
    }
    factual_metrics = {
        str(key): value
        for key, value in metrics.items()
        if key not in _NON_FACTUAL_METRIC_KEYS
        and isinstance(value, (int, float))
        and not isinstance(value, bool)
    }
    return SpeechEvidence(
        raw_transcript_hiragana=str(data.get("raw_transcript_hiragana", "")),
        raw_transcript_romaji=str(data.get("raw_transcript_romaji", "")),
        duration_sec=float(metrics.get("audio_duration_sec", 0.0)),
        speech_segments=segments,
        pause_segments=pauses,
        mora_timings=timings,
        factual_metrics=tuple(sorted(factual_metrics.items())),
        provenance=tuple(sorted(provenance.items())),
    )


def objective_data_from_evidence(data: SpeechEvidence, *, audio_path: str) -> dict[str, Any]:
    """Expose compatibility-shaped facts without restoring module-level grades."""
    pauses = [span.to_dict() for span in data.pause_segments]
    return {
        "audio_path": audio_path,
        "raw_transcript_hiragana": data.raw_transcript_hiragana,
        "raw_transcript_romaji": data.raw_transcript_romaji,
        "fluency_metrics": dict(data.factual_metrics),
        "top_pauses": sorted(pauses, key=lambda item: item["duration"], reverse=True)[:3],
        "pause_segments": pauses,
        "speech_segments": [span.to_dict() for span in data.speech_segments],
        "mora_timings": [timing.to_dict() for timing in data.mora_timings],
        **dict(data.provenance),
    }


def _span(value: dict[str, Any]) -> TimedSpan:
    start = float(value["start"])
    end = float(value.get("end", start + float(value.get("duration", 0.0))))
    return TimedSpan(start=start, end=end)
