"""Fact-only Interaction observations from shared evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .evidence.models import EvidenceBundle


INTERACTION_LEXICON_VERSION = "interaction-lexicon.v1"
_MARKERS = {
    "すみません": "repair_or_attention",
    "もういちど": "clarification_request",
    "どういう": "clarification_request",
    "そうです": "confirmation_or_acknowledgement",
    "はい": "acknowledgement",
    "ありがとう": "closing_or_acknowledgement",
}


@dataclass(frozen=True)
class InteractionFactPacket:
    input_provenance: tuple[tuple[str, str], ...]
    recording_observation: dict[str, Any]
    candidate_answer_segments: tuple[dict[str, Any], ...]
    interaction_marker_candidates: tuple[dict[str, Any], ...]
    unavailable_capabilities: tuple[str, ...]
    module_id: str = "interaction"

    def to_dict(self) -> dict[str, Any]:
        return {
            "module_id": self.module_id,
            "input_provenance": dict(self.input_provenance),
            "recording_observation": self.recording_observation,
            "candidate_answer_segments": list(self.candidate_answer_segments),
            "interaction_marker_candidates": list(self.interaction_marker_candidates),
            "unavailable_capabilities": list(self.unavailable_capabilities),
        }


class InteractionModule:
    """Collect observations only; never infer CEFR, intent, or conversation quality."""

    def collect(
        self,
        evidence: EvidenceBundle,
        *,
        prompt_text: str | None = None,
        interaction_context: Mapping[str, Any] | None = None,
    ) -> InteractionFactPacket:
        tokens = evidence.linguistic.tokens
        markers = tuple(
            {
                "token_index": index,
                "surface": token.surface,
                "dictionary_form": token.dictionary_form,
                "category": _MARKERS[token.dictionary_form],
                "lexicon_version": INTERACTION_LEXICON_VERSION,
            }
            for index, token in enumerate(tokens)
            if token.dictionary_form in _MARKERS
        )
        context = _normalize_context(
            interaction_context, prompt_text=prompt_text, duration=evidence.speech.duration_sec,
            pauses=tuple((pause.start, pause.end) for pause in evidence.speech.pause_segments),
            transcript=evidence.speech.raw_transcript_hiragana,
        )
        has_prompt = bool(context["prompts"])
        duration = evidence.speech.duration_sec
        unavailable = [
            "speaker_turns",
            "turn_transition_timing",
            "overlap_detection",
            "speaker_role_identification",
        ]
        if not has_prompt:
            unavailable.append("prompt_text")
        if context["recording_mode"] == "monologue":
            unavailable.append("dialogue_audio_not_present")
        if not tokens:
            unavailable.append("no_linguistic_tokens")
        return InteractionFactPacket(
            input_provenance=tuple(sorted({
                "evidence_schema_version": evidence.schema_version,
                "stt_model": dict(evidence.speech.provenance).get("stt_model", "unknown"),
                "tokenizer_version": evidence.linguistic.tokenizer_version,
                "interaction_lexicon_version": INTERACTION_LEXICON_VERSION,
            }.items())),
            recording_observation={
                "recording_mode": context["recording_mode"],
                "mode_source": context["mode_source"],
                "detected_speaker_count": context["detected_speaker_count"],
                "learner_speaker_id": context["learner_speaker_id"],
                "prompt_available": has_prompt,
            },
            candidate_answer_segments=tuple(context["answer_segments"]),
            interaction_marker_candidates=markers,
            unavailable_capabilities=tuple(unavailable),
        )


def _normalize_context(
    value: Mapping[str, Any] | None,
    *,
    prompt_text: str | None,
    duration: float,
    pauses: tuple[tuple[float, float], ...],
    transcript: str,
) -> dict[str, Any]:
    raw = dict(value or {})
    prompts = raw.get("prompts", [])
    if not prompts and prompt_text and prompt_text.strip() and prompt_text.strip() != "unknown":
        prompts = [{"prompt_id": "rp-1", "text": prompt_text.strip(), "display_order": 1}]
    normalized_prompts = []
    ids: set[str] = set()
    for index, item in enumerate(prompts, 1):
        if not isinstance(item, Mapping):
            raise ValueError("interaction_context.prompts items must be objects.")
        prompt_id = str(item.get("prompt_id") or "").strip()
        text = str(item.get("text") or "").strip()
        if not prompt_id or not text or prompt_id in ids:
            raise ValueError("interaction_context.prompts requires unique prompt_id and text.")
        ids.add(prompt_id)
        normalized_prompts.append({"prompt_id": prompt_id, "text": text, "display_order": int(item.get("display_order", index))})
    mode = str(raw.get("recording_mode") or "monologue")
    if mode not in {"monologue", "interview_dialogue", "multi_party_dialogue", "unknown"}:
        raise ValueError("interaction_context.recording_mode is invalid.")
    supplied_segments = raw.get("answer_segments") or []
    segments = []
    for index, item in enumerate(supplied_segments, 1):
        if not isinstance(item, Mapping):
            raise ValueError("interaction_context.answer_segments items must be objects.")
        start, end = float(item.get("start_sec", -1)), float(item.get("end_sec", -1))
        if start < 0 or end <= start or end > duration + 0.01:
            raise ValueError("interaction_context.answer_segments contains an invalid time range.")
        prompt_id = item.get("prompt_id")
        if prompt_id is not None and str(prompt_id) not in ids:
            raise ValueError("interaction_context.answer_segments references an unknown prompt_id.")
        segment = {
            "segment_id": str(item.get("segment_id") or f"segment-{index}"),
            "start_sec": start, "end_sec": end, "prompt_id": prompt_id,
            "mapping_status": "confirmed" if prompt_id else "candidate",
            "boundary_evidence": list(item.get("boundary_evidence") or ["user_supplied"]),
        }
        text_fields = {"start_offset", "end_offset", "transcript_text", "confirmed_by"}
        if any(key in item for key in text_fields):
            start_offset = item.get("start_offset")
            end_offset = item.get("end_offset")
            text = item.get("transcript_text")
            confirmed_by = str(item.get("confirmed_by") or "").strip()
            if (type(start_offset) is not int or type(end_offset) is not int
                    or not 0 <= start_offset < end_offset <= len(transcript)
                    or text != transcript[start_offset:end_offset]
                    or not confirmed_by or prompt_id is None):
                raise ValueError("interaction_context.answer_segments transcript span must be confirmed and match shared transcript.")
            segment.update({"start_offset": start_offset, "end_offset": end_offset,
                            "transcript_text": text, "confirmed_by": confirmed_by})
        segments.append(segment)
    if not segments:
        boundaries = [end for start, end in pauses if end - start >= 1.0 and 0 < end < duration]
        starts = [0.0, *boundaries]
        ends = [*boundaries, duration]
        segments = [{"segment_id": f"candidate-{index}", "start_sec": start, "end_sec": end,
                     "mapping_status": "candidate" if normalized_prompts else "unassigned",
                     "boundary_evidence": (["recording_start"] if index == 1 else ["vad_pause>=1.0s"])
                     + (["recording_end"] if index == len(starts) else [])}
                    for index, (start, end) in enumerate(zip(starts, ends), 1) if end > start]
    return {
        "prompts": normalized_prompts, "answer_segments": segments, "recording_mode": mode,
        "mode_source": "user_declared" if "recording_mode" in raw else "single_stream_baseline",
        "detected_speaker_count": int(raw.get("detected_speaker_count", 1)),
        "learner_speaker_id": raw.get("learner_speaker_id"),
    }
