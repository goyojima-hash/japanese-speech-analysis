"""Fact-only Coherence observations derived from shared speech evidence."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from .evidence.models import EvidenceBundle, TimedSpan, TokenEvidence


COHERENCE_LEXICON_VERSION = "coherence-lexicon.v1"
UNIT_POLICY_VERSION = "coherence-candidate-units.v1"


@dataclass(frozen=True)
class ConnectiveObservation:
    token_index: int
    surface: str
    dictionary_form: str
    category: str
    lexicon_version: str = COHERENCE_LEXICON_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_index": self.token_index,
            "surface": self.surface,
            "dictionary_form": self.dictionary_form,
            "category": self.category,
            "lexicon_version": self.lexicon_version,
        }


@dataclass(frozen=True)
class CandidateUnit:
    unit_index: int
    token_start_index: int
    token_end_index: int
    text_start_offset: int | None
    text_end_offset: int | None
    boundary_derivations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "unit_index": self.unit_index,
            "token_start_index": self.token_start_index,
            "token_end_index": self.token_end_index,
            "boundary_derivations": list(self.boundary_derivations),
        }
        if self.text_start_offset is not None and self.text_end_offset is not None:
            data["text_start_offset"] = self.text_start_offset
            data["text_end_offset"] = self.text_end_offset
        return data


@dataclass(frozen=True)
class RepetitionObservation:
    dictionary_form: str
    unit_indexes: tuple[int, ...]
    occurrence_count: int
    unit_distances: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "dictionary_form": self.dictionary_form,
            "unit_indexes": list(self.unit_indexes),
            "occurrence_count": self.occurrence_count,
            "unit_distances": list(self.unit_distances),
        }


@dataclass(frozen=True)
class CoherenceFactPacket:
    input_provenance: tuple[tuple[str, str], ...]
    connective_observations: tuple[ConnectiveObservation, ...]
    candidate_units: tuple[CandidateUnit, ...]
    repetition_observations: tuple[RepetitionObservation, ...]
    pause_observations: tuple[TimedSpan, ...]
    unavailable_capabilities: tuple[str, ...]
    module_id: str = "coherence"

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "input_provenance": dict(self.input_provenance),
            "connective_observations": [item.to_dict() for item in self.connective_observations],
            "candidate_units": [item.to_dict() for item in self.candidate_units],
            "repetition_observations": [item.to_dict() for item in self.repetition_observations],
            "pause_observations": [item.to_dict() for item in self.pause_observations],
            "unavailable_capabilities": list(self.unavailable_capabilities),
        }


class CoherenceModule:
    """Collect deterministic discourse observations without rating their quality."""

    def collect(self, evidence: EvidenceBundle) -> CoherenceFactPacket:
        tokens = evidence.linguistic.tokens
        observations = tuple(
            observation
            for index, token in enumerate(tokens)
            if (observation := _connective_observation(index, token)) is not None
        )
        units, offsets_unavailable = _candidate_units(tokens, observations)
        repetitions = _repetition_observations(tokens, units)
        unavailable = [
            "dependency_parsing",
            "coreference_resolution",
            "implicit_discourse_relations",
            "semantic_similarity",
            "token_pause_alignment",
        ]
        if not tokens:
            unavailable.insert(0, "no_linguistic_tokens")
        if offsets_unavailable:
            unavailable.append("token_offsets_unavailable")
        return CoherenceFactPacket(
            input_provenance=tuple(
                sorted(
                    {
                        "evidence_schema_version": evidence.schema_version,
                        "stt_model": dict(evidence.speech.provenance).get("stt_model", "unknown"),
                        "tokenizer_version": evidence.linguistic.tokenizer_version,
                        "split_mode": evidence.linguistic.split_mode,
                        "connective_lexicon_version": COHERENCE_LEXICON_VERSION,
                        "unit_policy_version": UNIT_POLICY_VERSION,
                    }.items()
                )
            ),
            connective_observations=observations,
            candidate_units=units,
            repetition_observations=repetitions,
            pause_observations=evidence.speech.pause_segments,
            unavailable_capabilities=tuple(unavailable),
        )


_CONNECTIVE_CATEGORIES = {
    "だから": "causal",
    "ので": "causal",
    "しかし": "contrast",
    "けど": "contrast",
    "けれど": "contrast",
    "でも": "contrast",
    "そして": "additive",
    "また": "additive",
    "まず": "sequence",
    "次に": "sequence",
    "それから": "sequence",
    "もし": "condition",
    "つまり": "conclusion",
}


def _connective_observation(index: int, token: TokenEvidence) -> ConnectiveObservation | None:
    category = _CONNECTIVE_CATEGORIES.get(token.dictionary_form) or _CONNECTIVE_CATEGORIES.get(token.surface)
    if category is None:
        return None
    return ConnectiveObservation(
        token_index=index,
        surface=token.surface,
        dictionary_form=token.dictionary_form,
        category=category,
    )


def _candidate_units(
    tokens: tuple[TokenEvidence, ...], observations: tuple[ConnectiveObservation, ...]
) -> tuple[tuple[CandidateUnit, ...], bool]:
    if not tokens:
        return (), False

    connective_categories = {item.token_index: item.category for item in observations}
    starts: list[tuple[int, tuple[str, ...]]] = [(0, ("transcript_start",))]
    for index, token in enumerate(tokens):
        if index in connective_categories and index > starts[-1][0]:
            starts.append((index, (f"connective:{connective_categories[index]}",)))
            continue
        if _is_terminal_pattern(token) and index + 1 < len(tokens) and index + 1 > starts[-1][0]:
            if index + 1 not in connective_categories:
                starts.append((index + 1, (f"terminal_pattern:{_terminal_pattern_id(token)}",)))

    offsets_unavailable = False
    units: list[CandidateUnit] = []
    for unit_index, (start, derivations) in enumerate(starts):
        end = starts[unit_index + 1][0] if unit_index + 1 < len(starts) else len(tokens)
        if unit_index + 1 == len(starts):
            derivations = (*derivations, "transcript_end")
        start_offset = tokens[start].start_offset
        end_offset = tokens[end - 1].end_offset
        if start_offset < 0 or end_offset < 0:
            offsets_unavailable = True
            text_start_offset = text_end_offset = None
        else:
            text_start_offset, text_end_offset = start_offset, end_offset
        units.append(
            CandidateUnit(
                unit_index=unit_index,
                token_start_index=start,
                token_end_index=end,
                text_start_offset=text_start_offset,
                text_end_offset=text_end_offset,
                boundary_derivations=derivations,
            )
        )
    return tuple(units), offsets_unavailable


def _is_terminal_pattern(token: TokenEvidence) -> bool:
    return _terminal_pattern_id(token) is not None


def _terminal_pattern_id(token: TokenEvidence) -> str | None:
    if token.surface.endswith(("ます", "ました")):
        return "polite_masu"
    if token.surface.endswith(("です", "でした")):
        return "polite_desu"
    if token.dictionary_form in {"だ", "である"}:
        return "plain_copula"
    return None


def _repetition_observations(
    tokens: tuple[TokenEvidence, ...], units: tuple[CandidateUnit, ...]
) -> tuple[RepetitionObservation, ...]:
    occurrences: dict[str, list[int]] = defaultdict(list)
    for unit in units:
        for token in tokens[unit.token_start_index : unit.token_end_index]:
            if _is_content_word(token):
                occurrences[token.dictionary_form].append(unit.unit_index)
    observations = []
    for dictionary_form, unit_indexes in sorted(occurrences.items()):
        distinct_indexes = tuple(sorted(set(unit_indexes)))
        if len(distinct_indexes) < 2:
            continue
        observations.append(
            RepetitionObservation(
                dictionary_form=dictionary_form,
                unit_indexes=distinct_indexes,
                occurrence_count=len(unit_indexes),
                unit_distances=tuple(right - left for left, right in combinations(distinct_indexes, 2)),
            )
        )
    return tuple(observations)


def _is_content_word(token: TokenEvidence) -> bool:
    return bool(token.part_of_speech) and token.part_of_speech[0] in {"名詞", "動詞", "形容詞", "副詞", "連体詞"}
