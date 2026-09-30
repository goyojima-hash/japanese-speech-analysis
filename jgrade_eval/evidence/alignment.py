"""Conservative text/time links derived from existing CTC argmax labels.

These are candidates, not forced alignment or proof that ASR text is correct.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

from .models import SpeechEvidence


ALIGNMENT_SCHEMA_VERSION = "transcript-alignment.v1"
ALIGNMENT_METHOD = "ctc_argmax_exact_prefix"
_DURATION_TOLERANCE_SEC = 0.05  # Legacy duration is rounded to two decimal places.


@dataclass(frozen=True)
class TimedTextUnit:
    text: str
    start_offset: int
    end_offset: int
    start_sec: float
    end_sec: float

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "text": self.text,
            "start_offset": self.start_offset,
            "end_offset": self.end_offset,
            "start_sec": self.start_sec,
            "end_sec": self.end_sec,
        }


@dataclass(frozen=True)
class TranscriptAlignment:
    transcript: str
    units: tuple[TimedTextUnit, ...]
    status: str
    reason: str | None
    source_model: str
    schema_version: str = ALIGNMENT_SCHEMA_VERSION
    method: str = ALIGNMENT_METHOD

    @property
    def aligned_chars(self) -> int:
        return self.units[-1].end_offset if self.units else 0

    def candidate_for_interval(self, start_sec: float, end_sec: float) -> dict[str, Any] | None:
        if not (isfinite(start_sec) and isfinite(end_sec) and 0 <= start_sec < end_sec):
            return None
        selected = tuple(unit for unit in self.units
                         if unit.start_sec >= start_sec - 0.0001 and unit.end_sec <= end_sec + 0.0001)
        if not selected or any(left.end_offset != right.start_offset
                               for left, right in zip(selected, selected[1:])):
            return None
        first, last = selected[0], selected[-1]
        return {
            "start_offset": first.start_offset,
            "end_offset": last.end_offset,
            "transcript_text": self.transcript[first.start_offset:last.end_offset],
            "alignment_status": self.status,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "method": self.method,
            "status": self.status,
            "reason": self.reason,
            "source_model": self.source_model,
            "timing_quality": "unverified_ctc_argmax",
            "aligned_chars": self.aligned_chars,
            "transcript_chars": len(self.transcript),
            "units": [unit.to_dict() for unit in self.units],
        }


def build_transcript_alignment(speech: SpeechEvidence) -> TranscriptAlignment:
    """Link only an exact prefix; never repair or guess a mismatched transcript."""

    transcript = speech.raw_transcript_hiragana
    model = dict(speech.provenance).get("stt_model", "unknown")

    def unavailable(reason: str) -> TranscriptAlignment:
        return TranscriptAlignment(transcript, (), "unavailable", reason, model)

    if not transcript:
        return unavailable("empty_transcript")
    if not speech.mora_timings:
        return unavailable("missing_mora_timings")
    if not isfinite(speech.duration_sec) or speech.duration_sec <= 0:
        return unavailable("invalid_duration")

    units: list[TimedTextUnit] = []
    offset = 0
    previous_end = 0.0
    for timing in speech.mora_timings:
        label = timing.mora
        if (not label or label == "?" or not transcript.startswith(label, offset)
                or offset + len(label) > len(transcript)):
            return unavailable("label_transcript_mismatch")
        if (not isfinite(timing.start) or not isfinite(timing.end)
                or timing.start < previous_end or timing.start < 0
                or timing.end <= timing.start
                or timing.end > speech.duration_sec + _DURATION_TOLERANCE_SEC):
            return unavailable("invalid_or_nonmonotonic_timing")
        units.append(TimedTextUnit(label, offset, offset + len(label), timing.start, timing.end))
        offset += len(label)
        previous_end = timing.end

    status = "complete" if offset == len(transcript) else "partial"
    reason = None if status == "complete" else "trailing_transcript_unaligned"
    return TranscriptAlignment(transcript, tuple(units), status, reason, model)
