"""Fact-only Accuracy observations derived from shared speech evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from .evidence.models import EvidenceBundle, TokenEvidence


class ReferenceTokenizer(Protocol):
    version: str
    split_mode: str

    def tokenize(self, text: str) -> list[TokenEvidence]: ...


@dataclass(frozen=True)
class AsrObservation:
    unit: str
    start: float
    end: float
    derivation: str
    calibration_status: str
    selected_posterior_mean: float | None = None
    competing_margin_mean: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit": self.unit,
            "start": self.start,
            "end": self.end,
            "derivation": self.derivation,
            "calibration_status": self.calibration_status,
            "selected_posterior_mean": self.selected_posterior_mean,
            "competing_margin_mean": self.competing_margin_mean,
        }


@dataclass(frozen=True)
class MorphologyObservation:
    token_index: int
    token: TokenEvidence
    pattern_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            **self.token.to_dict(),
            "pattern_ids": list(self.pattern_ids),
        }


@dataclass(frozen=True)
class ReferenceDifference:
    operation: str
    observed: str | None
    reference: str | None
    observed_token_index: int | None
    reference_token_index: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "observed": self.observed,
            "reference": self.reference,
            "observed_token_index": self.observed_token_index,
            "reference_token_index": self.reference_token_index,
        }


@dataclass(frozen=True)
class AccuracyFactPacket:
    module_id: str
    input_provenance: tuple[tuple[str, str], ...]
    asr_observations: tuple[AsrObservation, ...]
    morphology_observations: tuple[MorphologyObservation, ...]
    reference_differences: tuple[ReferenceDifference, ...]
    unavailable_capabilities: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "input_provenance": dict(self.input_provenance),
            "asr_observations": [item.to_dict() for item in self.asr_observations],
            "morphology_observations": [item.to_dict() for item in self.morphology_observations],
            "reference_differences": [item.to_dict() for item in self.reference_differences],
            "unavailable_capabilities": list(self.unavailable_capabilities),
        }


class AccuracyModule:
    """Collect transcript and morphology observations without judging correctness."""

    def __init__(self, *, reference_tokenizer: ReferenceTokenizer | None = None) -> None:
        self._reference_tokenizer = reference_tokenizer

    def collect(
        self,
        evidence: EvidenceBundle,
        *,
        reference_transcript: str | None = None,
    ) -> AccuracyFactPacket:
        source_tokens = evidence.linguistic.tokens
        return AccuracyFactPacket(
            module_id="accuracy",
            input_provenance=tuple(
                sorted(
                    {
                        "evidence_schema_version": evidence.schema_version,
                        "stt_model": dict(evidence.speech.provenance).get("stt_model", "unknown"),
                        "tokenizer_version": evidence.linguistic.tokenizer_version,
                    }.items()
                )
            ),
            asr_observations=tuple(
                AsrObservation(
                    unit=timing.mora,
                    start=timing.start,
                    end=timing.end,
                    derivation="ctc_argmax_mora_timing",
                    calibration_status="unavailable",
                )
                for timing in evidence.speech.mora_timings
            ),
            morphology_observations=tuple(
                MorphologyObservation(
                    token_index=index,
                    token=token,
                    pattern_ids=_pattern_ids(source_tokens, index),
                )
                for index, token in enumerate(source_tokens)
            ),
            reference_differences=self._reference_differences(
                source_tokens,
                reference_transcript=reference_transcript,
            ),
            unavailable_capabilities=(
                "asr_posterior_confidence",
                "phoneme_alignment",
                "pronunciation_diagnosis",
            ),
        )

    def _reference_differences(
        self,
        observed_tokens: tuple[TokenEvidence, ...],
        *,
        reference_transcript: str | None,
    ) -> tuple[ReferenceDifference, ...]:
        if reference_transcript is None:
            return ()
        reference_tokens = self._tokenize_reference(reference_transcript)
        return tuple(_align_token_surfaces(observed_tokens, reference_tokens))

    def _tokenize_reference(self, text: str) -> list[TokenEvidence]:
        if self._reference_tokenizer is None:
            from .evidence.linguistic import SudachiTokenizer

            self._reference_tokenizer = SudachiTokenizer()
        return self._reference_tokenizer.tokenize(text)


def _pattern_ids(tokens: tuple[TokenEvidence, ...], index: int) -> tuple[str, ...]:
    token = tokens[index]
    pos = token.part_of_speech[0] if token.part_of_speech else "unknown"
    patterns = [f"pos:{pos}"]
    previous_pos = (
        tokens[index - 1].part_of_speech[0]
        if index > 0 and tokens[index - 1].part_of_speech
        else None
    )
    if pos == "助詞" and previous_pos == "名詞":
        patterns.append("particle_after_noun")
    if pos == "助動詞" and previous_pos == "名詞":
        patterns.append("auxiliary_after_noun")
    if pos == "助動詞" and previous_pos == "動詞":
        patterns.append("auxiliary_after_verb")
    return tuple(patterns)


def _align_token_surfaces(
    observed: tuple[TokenEvidence, ...],
    reference: list[TokenEvidence],
) -> list[ReferenceDifference]:
    rows = len(observed) + 1
    columns = len(reference) + 1
    costs = [[0] * columns for _ in range(rows)]
    for observed_index in range(rows):
        costs[observed_index][0] = observed_index
    for reference_index in range(columns):
        costs[0][reference_index] = reference_index
    for observed_index in range(1, rows):
        for reference_index in range(1, columns):
            replace_cost = (
                0
                if observed[observed_index - 1].surface
                == reference[reference_index - 1].surface
                else 1
            )
            costs[observed_index][reference_index] = min(
                costs[observed_index - 1][reference_index - 1] + replace_cost,
                costs[observed_index - 1][reference_index] + 1,
                costs[observed_index][reference_index - 1] + 1,
            )

    differences: list[ReferenceDifference] = []
    observed_index, reference_index = len(observed), len(reference)
    while observed_index or reference_index:
        if (
            observed_index
            and reference_index
            and costs[observed_index][reference_index]
            == costs[observed_index - 1][reference_index - 1]
            + (observed[observed_index - 1].surface != reference[reference_index - 1].surface)
        ):
            observed_index -= 1
            reference_index -= 1
            differences.append(
                ReferenceDifference(
                    operation=(
                        "equal"
                        if observed[observed_index].surface == reference[reference_index].surface
                        else "replace"
                    ),
                    observed=observed[observed_index].surface,
                    reference=reference[reference_index].surface,
                    observed_token_index=observed_index,
                    reference_token_index=reference_index,
                )
            )
        elif (
            observed_index
            and costs[observed_index][reference_index]
            == costs[observed_index - 1][reference_index] + 1
        ):
            observed_index -= 1
            differences.append(
                ReferenceDifference("delete", observed[observed_index].surface, None, observed_index, None)
            )
        else:
            reference_index -= 1
            differences.append(
                ReferenceDifference("insert", None, reference[reference_index].surface, None, reference_index)
            )
    return list(reversed(differences))
