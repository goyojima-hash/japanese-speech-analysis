from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping

from .cache import EvidenceCache
from .linguistic import LinguisticEvidenceExtractor
from .models import EvidenceBundle, SCHEMA_VERSION, SourceEvidence, SpeechEvidence


class EvidencePipeline:
    """Compose reusable, fact-only evidence once for selected modules."""

    def __init__(
        self,
        *,
        speech_extractor: object,
        linguistic_extractor: LinguisticEvidenceExtractor,
        cache: EvidenceCache | None = None,
        preprocessing_config: Mapping[str, str] | None = None,
    ) -> None:
        self._speech_extractor = speech_extractor
        self._linguistic_extractor = linguistic_extractor
        self._cache = cache
        self._preprocessing_config = dict(preprocessing_config or {})

    def build(self, audio_path: Path) -> EvidenceBundle:
        source = SourceEvidence(audio_path=str(audio_path), audio_sha256=_hash_file_if_present(audio_path))
        key = self._cache_key(source)
        if self._cache:
            cached = self._cache.load(key)
            if cached is not None:
                return cached

        speech = self._extract_speech(audio_path)
        linguistic = self._linguistic_extractor.analyze(speech.raw_transcript_hiragana)
        bundle = EvidenceBundle(source=source, speech=speech, linguistic=linguistic)
        if self._cache:
            self._cache.store(key, bundle)
        return bundle

    def _extract_speech(self, audio_path: Path) -> SpeechEvidence:
        extract = getattr(self._speech_extractor, "extract")
        value = extract(audio_path)
        if not isinstance(value, SpeechEvidence):
            raise TypeError("Speech evidence extractors must return SpeechEvidence.")
        return value

    def _cache_key(self, source: SourceEvidence) -> str:
        speech_provenance = getattr(self._speech_extractor, "provenance", {})
        payload = {
            "schema_version": SCHEMA_VERSION,
            "audio_sha256": source.audio_sha256,
            "audio_path": source.audio_path if source.audio_sha256 is None else None,
            "speech_provenance": dict(speech_provenance),
            "linguistic_provenance": self._linguistic_extractor.provenance,
            "preprocessing_config": self._preprocessing_config,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        return sha256(encoded).hexdigest()


def _hash_file_if_present(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()
