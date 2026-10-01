"""Opt-in, local CTC Viterbi alignment for experiment use only.

This finds the most likely CTC path for supplied text. It does not establish
that the text was spoken, nor does it measure boundary accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


SCHEMA_VERSION = "forced-alignment.v2"
ALGORITHM = "ctc_viterbi"
_MAX_PATH_CELLS = 30_000_000


@dataclass(frozen=True)
class CTCLogitChunk:
    start_sample: int
    sample_count: int
    logits: np.ndarray


@dataclass(frozen=True)
class AlignedCharacter:
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
class ForcedAlignmentResult:
    transcript: str
    units: tuple[AlignedCharacter, ...]
    status: str
    reason: str | None
    model_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "algorithm": ALGORITHM,
            "status": self.status,
            "reason": self.reason,
            "model_id": self.model_id,
            "timing_quality": "unverified_ctc_forced_alignment",
            "transcript_chars": len(self.transcript),
            "aligned_chars": len(self.units),
            "units": [unit.to_dict() for unit in self.units],
        }


def align_ctc_chunks(
    chunks: Sequence[CTCLogitChunk],
    transcript: str,
    char_to_id: Mapping[str, int],
    *,
    blank_id: int,
    sample_rate: int,
    model_id: str = "unknown",
) -> ForcedAlignmentResult:
    """Align exact transcript characters to existing model logits.

    No text normalization or implicit deletion occurs. An impossible CTC path
    returns ``unavailable`` rather than fabricated timestamps.
    """

    def unavailable(reason: str) -> ForcedAlignmentResult:
        return ForcedAlignmentResult(transcript, (), "unavailable", reason, model_id)

    if not transcript:
        return unavailable("empty_transcript")
    if not chunks or sample_rate <= 0:
        return unavailable("invalid_audio_geometry")

    emissions: list[np.ndarray] = []
    frame_starts: list[int] = []
    frame_ends: list[int] = []
    expected_start = 0
    vocab_size: int | None = None
    for chunk in chunks:
        scores = np.asarray(chunk.logits)
        if (scores.ndim != 2 or scores.shape[0] == 0 or scores.shape[1] == 0
                or not np.issubdtype(scores.dtype, np.number)
                or not np.isfinite(scores).all()):
            return unavailable("invalid_logits")
        if (chunk.start_sample != expected_start or chunk.sample_count <= 0
                or scores.shape[0] > chunk.sample_count):
            return unavailable("invalid_audio_geometry")
        if vocab_size is not None and scores.shape[1] != vocab_size:
            return unavailable("invalid_logits")
        vocab_size = scores.shape[1]
        frames = scores.shape[0]
        for index in range(frames):
            frame_starts.append(chunk.start_sample + index * chunk.sample_count // frames)
            frame_ends.append(chunk.start_sample + (index + 1) * chunk.sample_count // frames)
        emissions.append(scores.astype(np.float64, copy=False))
        expected_start += chunk.sample_count

    assert vocab_size is not None
    if not isinstance(blank_id, int) or not 0 <= blank_id < vocab_size:
        return unavailable("invalid_blank_id")
    ids: list[int] = []
    for character in transcript:
        token_id = char_to_id.get(character)
        if not isinstance(token_id, int) or token_id == blank_id or not 0 <= token_id < vocab_size:
            return unavailable("unsupported_character")
        ids.append(token_id)

    scores = np.concatenate(emissions, axis=0)
    state_count = 2 * len(ids) + 1
    if scores.shape[0] * state_count > _MAX_PATH_CELLS:
        return unavailable("sequence_too_large")
    states = np.full(state_count, blank_id, dtype=np.int32)
    states[1::2] = ids
    previous = np.full(state_count, -np.inf, dtype=np.float64)
    previous[0] = scores[0, blank_id]
    previous[1] = scores[0, ids[0]]
    back = np.full((scores.shape[0], state_count), -1, dtype=np.int32)
    positions = np.arange(state_count)
    skip_allowed = np.zeros(state_count, dtype=bool)
    skip_allowed[3::2] = states[3::2] != states[1:-2:2]

    for frame in range(1, scores.shape[0]):
        advance = np.concatenate(([-np.inf], previous[:-1]))
        skip = np.concatenate(([-np.inf, -np.inf], previous[:-2]))
        skip[~skip_allowed] = -np.inf
        options = np.stack((previous, advance, skip))
        choice = np.argmax(options, axis=0)
        best = options[choice, positions]
        previous = best + scores[frame, states]
        back[frame] = positions - choice

    final_state = state_count - 1
    if previous[-2] > previous[-1]:
        final_state -= 1
    if not np.isfinite(previous[final_state]):
        return unavailable("ctc_path_unavailable")

    path = np.empty(scores.shape[0], dtype=np.int32)
    path[-1] = final_state
    for frame in range(scores.shape[0] - 1, 0, -1):
        path[frame - 1] = back[frame, path[frame]]

    units: list[AlignedCharacter] = []
    for offset, character in enumerate(transcript):
        aligned_frames = np.flatnonzero(path == 2 * offset + 1)
        if aligned_frames.size == 0:
            return unavailable("ctc_path_unavailable")
        first, last = int(aligned_frames[0]), int(aligned_frames[-1])
        units.append(AlignedCharacter(
            character, offset, offset + 1,
            frame_starts[first] / sample_rate,
            frame_ends[last] / sample_rate,
        ))
    return ForcedAlignmentResult(transcript, tuple(units), "complete", None, model_id)
