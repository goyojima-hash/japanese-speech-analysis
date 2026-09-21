from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


SCHEMA_VERSION = "evidence.v1"


@dataclass(frozen=True)
class TimedSpan:
    start: float
    end: float

    def to_dict(self) -> dict[str, float]:
        return {"start": self.start, "end": self.end, "duration": round(self.end - self.start, 4)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TimedSpan":
        return cls(start=float(value["start"]), end=float(value["end"]))


@dataclass(frozen=True)
class MoraTiming:
    mora: str
    start: float
    end: float

    def to_dict(self) -> dict[str, Any]:
        return {"mora": self.mora, "start": self.start, "end": self.end}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MoraTiming":
        return cls(mora=str(value["mora"]), start=float(value["start"]), end=float(value["end"]))


@dataclass(frozen=True)
class TokenEvidence:
    surface: str
    dictionary_form: str
    reading_hiragana: str
    part_of_speech: tuple[str, ...]
    start_offset: int = -1
    end_offset: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "dictionary_form": self.dictionary_form,
            "reading_hiragana": self.reading_hiragana,
            "part_of_speech": list(self.part_of_speech),
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TokenEvidence":
        return cls(
            surface=str(value["surface"]),
            dictionary_form=str(value["dictionary_form"]),
            reading_hiragana=str(value["reading_hiragana"]),
            part_of_speech=tuple(str(item) for item in value["part_of_speech"]),
            start_offset=int(value.get("start_offset", -1)),
            end_offset=int(value.get("end_offset", -1)),
        )


@dataclass(frozen=True)
class SourceEvidence:
    audio_path: str
    audio_sha256: str | None

    def to_dict(self) -> dict[str, str | None]:
        return {"audio_path": self.audio_path, "audio_sha256": self.audio_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SourceEvidence":
        sha256 = value.get("audio_sha256")
        return cls(audio_path=str(value["audio_path"]), audio_sha256=str(sha256) if sha256 else None)


@dataclass(frozen=True)
class SpeechEvidence:
    raw_transcript_hiragana: str
    raw_transcript_romaji: str
    duration_sec: float
    speech_segments: tuple[TimedSpan, ...]
    pause_segments: tuple[TimedSpan, ...]
    mora_timings: tuple[MoraTiming, ...]
    factual_metrics: tuple[tuple[str, float | int], ...]
    provenance: tuple[tuple[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_transcript_hiragana": self.raw_transcript_hiragana,
            "raw_transcript_romaji": self.raw_transcript_romaji,
            "duration_sec": self.duration_sec,
            "speech_segments": [segment.to_dict() for segment in self.speech_segments],
            "pause_segments": [segment.to_dict() for segment in self.pause_segments],
            "mora_timings": [timing.to_dict() for timing in self.mora_timings],
            "factual_metrics": dict(self.factual_metrics),
            "provenance": dict(self.provenance),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SpeechEvidence":
        return cls(
            raw_transcript_hiragana=str(value["raw_transcript_hiragana"]),
            raw_transcript_romaji=str(value["raw_transcript_romaji"]),
            duration_sec=float(value["duration_sec"]),
            speech_segments=tuple(TimedSpan.from_dict(item) for item in value["speech_segments"]),
            pause_segments=tuple(TimedSpan.from_dict(item) for item in value["pause_segments"]),
            mora_timings=tuple(MoraTiming.from_dict(item) for item in value["mora_timings"]),
            factual_metrics=tuple(
                sorted(
                    (str(key), _number(item))
                    for key, item in dict(value.get("factual_metrics", {})).items()
                )
            ),
            provenance=tuple(sorted((str(key), str(item)) for key, item in dict(value["provenance"]).items())),
        )


def _number(value: Any) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("factual_metrics values must be numeric.")
    return value


@dataclass(frozen=True)
class LinguisticEvidence:
    tokens: tuple[TokenEvidence, ...]
    tokenizer_version: str
    split_mode: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": [token.to_dict() for token in self.tokens],
            "tokenizer_version": self.tokenizer_version,
            "split_mode": self.split_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LinguisticEvidence":
        return cls(
            tokens=tuple(TokenEvidence.from_dict(item) for item in value["tokens"]),
            tokenizer_version=str(value["tokenizer_version"]),
            split_mode=str(value["split_mode"]),
        )


@dataclass(frozen=True)
class EvidenceBundle:
    source: SourceEvidence
    speech: SpeechEvidence
    linguistic: LinguisticEvidence
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "speech": self.speech.to_dict(),
            "linguistic": self.linguistic.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EvidenceBundle":
        return cls(
            source=SourceEvidence.from_dict(value["source"]),
            speech=SpeechEvidence.from_dict(value["speech"]),
            linguistic=LinguisticEvidence.from_dict(value["linguistic"]),
            schema_version=str(value["schema_version"]),
        )
