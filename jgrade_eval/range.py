from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Mapping
from unicodedata import normalize

from .evidence.linguistic import SudachiTokenizer, Tokenizer
from .evidence.models import LinguisticEvidence, TokenEvidence


JLPT_LEVELS = ("N5", "N4", "N3", "N2", "N1")
_FUNCTION_POS = frozenset({"助詞", "助動詞", "補助記号", "空白"})
_DATA_DIRECTORY = Path(__file__).with_name("data")
_DEFAULT_VOCABULARY_PATH = _DATA_DIRECTORY / "jlpt_vocab_all.json"


class TokenizedWord(TokenEvidence):
    """Backward-compatible Range token record.

    Shared tokenisation now returns ``TokenEvidence``; this subclass retains the
    former convenience property for callers that imported it from this module.
    """

    @property
    def is_lexical(self) -> bool:
        return _is_lexical(self)


@dataclass(frozen=True)
class JLPTCandidate:
    surface: str
    reading_hiragana: str
    level_number: int

    @property
    def level_name(self) -> str:
        return f"N{self.level_number}"

    def to_dict(self) -> dict[str, str]:
        return {
            "surface": self.surface,
            "reading_hiragana": self.reading_hiragana,
            "jlpt_level": self.level_name,
        }


class JLPTVocabulary:
    """Reading-indexed JLPT vocabulary with replace-by-reading overrides."""

    def __init__(
        self,
        candidates_by_reading: Mapping[str, tuple[JLPTCandidate, ...]],
        *,
        version: str,
    ) -> None:
        self._candidates_by_reading = dict(candidates_by_reading)
        self.version = version

    @classmethod
    def from_entries(
        cls,
        entries: Iterable[Mapping[str, object]],
        *,
        overrides: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
        version: str,
    ) -> "JLPTVocabulary":
        candidates_by_reading = _group_candidates(entries)
        for reading, override_entries in (overrides or {}).items():
            candidates_by_reading[_normalise_hiragana(reading)] = _sort_candidates(
                _parse_candidates(override_entries)
            )
        return cls(candidates_by_reading, version=version)

    @classmethod
    def from_json_files(
        cls,
        base_path: Path,
        *,
        override_path: Path | None = None,
        version: str,
    ) -> "JLPTVocabulary":
        with base_path.open(encoding="utf-8") as file:
            base_data = json.load(file)
        if not isinstance(base_data, dict):
            raise ValueError("JLPT vocabulary data must be a JSON object.")

        entries = [
            {"surface": surface, "reading": candidate["reading"], "level": candidate["level"]}
            for surface, candidates in base_data.items()
            if isinstance(candidates, list)
            for candidate in candidates
            if isinstance(candidate, dict)
        ]
        overrides = _load_overrides(override_path) if override_path else None
        return cls.from_entries(entries, overrides=overrides, version=version)

    def lookup(self, reading_hiragana: str) -> tuple[JLPTCandidate, ...]:
        return self._candidates_by_reading.get(_normalise_hiragana(reading_hiragana), ())


class RangeExtractor:
    """Build deterministic lexical Range evidence from a hiragana transcript."""

    def __init__(self, tokenizer: Tokenizer, vocabulary: JLPTVocabulary) -> None:
        self._tokenizer = tokenizer
        self._vocabulary = vocabulary

    @classmethod
    def default(cls, *, override_path: Path | None = None) -> "RangeExtractor":
        vocabulary = JLPTVocabulary.from_json_files(
            _DEFAULT_VOCABULARY_PATH,
            override_path=override_path,
            version="Bluskyo/JLPT_Vocabulary@4358f932937ad0232194a36e9f4f875094910c6b",
        )
        return cls(SudachiTokenizer(), vocabulary)

    def analyze(self, raw_transcript_hiragana: str) -> dict[str, object]:
        tokens = self._tokenizer.tokenize(raw_transcript_hiragana)
        return self._analyze_tokens(
            tokens,
            tokenizer_version=getattr(self._tokenizer, "version", "unknown"),
            split_mode=getattr(self._tokenizer, "split_mode", "unknown"),
        )

    def analyze_linguistic_evidence(self, evidence: LinguisticEvidence) -> dict[str, object]:
        """Calculate Range facts from a shared, already-tokenised evidence bundle."""
        return self._analyze_tokens(
            evidence.tokens,
            tokenizer_version=evidence.tokenizer_version,
            split_mode=evidence.split_mode,
        )

    def _analyze_tokens(
        self,
        tokens: Iterable[TokenEvidence],
        *,
        tokenizer_version: str,
        split_mode: str,
    ) -> dict[str, object]:
        token_records = [self._analyse_token(token) for token in tokens]
        lexical_records = [record for record in token_records if record["is_lexical"]]
        known_records = [record for record in lexical_records if record["selected_jlpt_level"]]
        unknown_records = [record for record in lexical_records if not record["selected_jlpt_level"]]
        lemma_count = len({str(record["dictionary_form"]) for record in lexical_records})
        lexical_token_count = len(lexical_records)

        return {
            "tokens": token_records,
            "statistics": {
                "token_count": len(token_records),
                "lexical_token_count": lexical_token_count,
                "unique_lemma_count": lemma_count,
                "ttr": round(lemma_count / lexical_token_count, 4) if lexical_token_count else 0.0,
                "known_token_count": len(known_records),
                "unknown_token_count": len(unknown_records),
                "unknown_token_rate": round(len(unknown_records) / lexical_token_count, 4)
                if lexical_token_count
                else 0.0,
            },
            "jlpt_distribution": _distribution(known_records),
            "ambiguity_count": sum(1 for record in lexical_records if len(record["jlpt_candidates"]) > 1),
            "dictionary_version": self._vocabulary.version,
            "tokenizer_version": tokenizer_version,
            "split_mode": split_mode,
        }

    def _analyse_token(self, token: TokenEvidence) -> dict[str, object]:
        is_lexical = _is_lexical(token)
        candidates = self._vocabulary.lookup(token.reading_hiragana) if is_lexical else ()
        selected = candidates[0] if candidates else None
        return {
            "surface": token.surface,
            "dictionary_form": token.dictionary_form,
            "reading_hiragana": _normalise_hiragana(token.reading_hiragana),
            "part_of_speech": list(token.part_of_speech),
            "is_lexical": is_lexical,
            "jlpt_candidates": [candidate.to_dict() for candidate in candidates],
            "selected_jlpt_level": selected.level_name if selected else None,
        }


def _is_lexical(token: TokenEvidence) -> bool:
    return not token.part_of_speech or token.part_of_speech[0] not in _FUNCTION_POS


def _group_candidates(entries: Iterable[Mapping[str, object]]) -> dict[str, tuple[JLPTCandidate, ...]]:
    grouped: dict[str, list[JLPTCandidate]] = {}
    for candidate in _parse_candidates(entries):
        grouped.setdefault(candidate.reading_hiragana, []).append(candidate)
    return {reading: _sort_candidates(candidates) for reading, candidates in grouped.items()}


def _parse_candidates(entries: Iterable[Mapping[str, object]]) -> list[JLPTCandidate]:
    candidates: list[JLPTCandidate] = []
    for entry in entries:
        level_number = int(entry["level"])
        if level_number not in range(1, 6):
            raise ValueError(f"JLPT level must be an integer from 1 to 5, got {level_number}.")
        candidates.append(
            JLPTCandidate(
                surface=str(entry["surface"]),
                reading_hiragana=_normalise_hiragana(str(entry["reading"])),
                level_number=level_number,
            )
        )
    return candidates


def _sort_candidates(candidates: Iterable[JLPTCandidate]) -> tuple[JLPTCandidate, ...]:
    return tuple(sorted(candidates, key=lambda candidate: (-candidate.level_number, candidate.surface)))


def _distribution(records: list[dict[str, object]]) -> dict[str, dict[str, int]]:
    levels = [str(record["selected_jlpt_level"]) for record in records]
    token_counts = Counter(levels)
    lemma_sets = {
        level: {
            str(record["dictionary_form"])
            for record in records
            if record["selected_jlpt_level"] == level
        }
        for level in JLPT_LEVELS
    }
    return {
        level: {
            "token_count": token_counts[level],
            "unique_lemma_count": len(lemma_sets[level]),
        }
        for level in JLPT_LEVELS
    }


def _normalise_hiragana(value: str) -> str:
    normalised = normalize("NFKC", value).strip()
    return "".join(
        chr(ord(character) - 0x60) if "ァ" <= character <= "ヶ" else character
        for character in normalised
    )


def _load_overrides(path: Path) -> Mapping[str, Iterable[Mapping[str, object]]]:
    with path.open(encoding="utf-8") as file:
        overrides = json.load(file)
    if not isinstance(overrides, dict):
        raise ValueError("Range vocabulary overrides must be a JSON object keyed by reading.")
    return overrides
