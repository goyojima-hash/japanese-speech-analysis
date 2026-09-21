from __future__ import annotations

from importlib.metadata import version
from typing import Protocol
from unicodedata import normalize

from .models import LinguisticEvidence, TokenEvidence


class Tokenizer(Protocol):
    version: str
    split_mode: str

    def tokenize(self, text: str) -> list[TokenEvidence]: ...


class SudachiTokenizer:
    """Convert a fixed SudachiPy analysis into position-preserving token facts."""

    split_mode = "A"

    def __init__(self) -> None:
        from sudachipy import Dictionary, SplitMode

        self._tokenizer = Dictionary(dict="core").create()
        self._split_mode = SplitMode.A
        self.version = f"SudachiPy {version('SudachiPy')}; SudachiDict-core {version('SudachiDict-core')}"

    def tokenize(self, text: str) -> list[TokenEvidence]:
        if not text.strip():
            return []
        return [
            TokenEvidence(
                surface=morpheme.surface(),
                dictionary_form=morpheme.dictionary_form(),
                reading_hiragana=_normalise_hiragana(morpheme.reading_form()),
                part_of_speech=tuple(morpheme.part_of_speech()),
                start_offset=morpheme.begin(),
                end_offset=morpheme.end(),
            )
            for morpheme in self._tokenizer.tokenize(text, self._split_mode)
        ]


class LinguisticEvidenceExtractor:
    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self._tokenizer = tokenizer or SudachiTokenizer()

    @property
    def provenance(self) -> dict[str, str]:
        return {
            "tokenizer_version": self._tokenizer.version,
            "split_mode": self._tokenizer.split_mode,
        }

    def analyze(self, text: str) -> LinguisticEvidence:
        return LinguisticEvidence(
            tokens=tuple(self._tokenizer.tokenize(text)),
            tokenizer_version=self._tokenizer.version,
            split_mode=self._tokenizer.split_mode,
        )


def _normalise_hiragana(value: str) -> str:
    normalised = normalize("NFKC", value).strip()
    return "".join(
        chr(ord(character) - 0x60) if "ァ" <= character <= "ヶ" else character
        for character in normalised
    )
