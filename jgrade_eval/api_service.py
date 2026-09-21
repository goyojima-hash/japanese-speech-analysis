from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from .audio_pipeline import FluencyExtractor
from .deliberation import deliberate_auto_cefr
from .evidence import EvidencePipeline, FluencySpeechEvidenceExtractor, LinguisticEvidenceExtractor
from .evidence.speech import objective_data_from_evidence
from .fact_modules import DEFAULT_FACT_MODULES, SUPPORTED_FACT_MODULES, run_fact_modules
from .live_judges import (
    JudgeFailure,
    ProviderSpec,
    judge_auto_cefr_with_live_panel_partial,
)
from .mock_judges import judge_auto_cefr_with_mock_panel
from .models import AutoLevelJudgeResult, Rating
from .range import RangeExtractor
from .tuning_profile import TuningProfile, compose_auto_cefr_system_prompt


TASK_RATING_ORDER = {
    Rating.FAIL: 0,
    Rating.NEAR_FAIL: 1,
    Rating.PASS: 2,
    Rating.EXCELLENT: 3,
}

def evaluate_speech_level(
    audio_path: Path,
    *,
    external_id: str | None = None,
    language: str = "ja",
    roleplay_task: str = "unknown",
    jfs_can_do_criteria: list[str] | None = None,
    speaker_metadata: dict[str, Any] | None = None,
    judge_mode: str = "mock",
    provider_specs: list[ProviderSpec] | None = None,
    timeout_sec: float = 60.0,
    include_objective_data: bool = True,
    profile: TuningProfile | None = None,
    extractor: FluencyExtractor | None = None,
    range_extractor: RangeExtractor | None = None,
    evidence_pipeline: EvidencePipeline | None = None,
    selected_modules: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Evaluate one speech file and return an API-shaped completed result."""

    if language != "ja":
        raise ValueError("language must be 'ja'.")
    if judge_mode not in {"mock", "live"}:
        raise ValueError("judge_mode must be 'mock' or 'live'.")
    if judge_mode == "live" and not provider_specs:
        raise ValueError("live judge_mode requires 1 to 3 judge providers.")

    evaluation_id = f"eval_{uuid4().hex[:12]}"
    created_at = _now_iso()
    active_pipeline = evidence_pipeline or EvidencePipeline(
        speech_extractor=FluencySpeechEvidenceExtractor(extractor or FluencyExtractor()),
        linguistic_extractor=LinguisticEvidenceExtractor(),
    )
    evidence = active_pipeline.build(audio_path)
    fact_modules = run_fact_modules(
        evidence,
        selected_modules=selected_modules,
        range_extractor=range_extractor,
    )
    active_modules = fact_modules.active_modules
    objective_data = fact_modules.merge_objective_data(
        objective_data_from_evidence(evidence.speech, audio_path=str(audio_path))
    )
    objective_data["evidence_schema_version"] = evidence.schema_version
    sample_id = external_id or evaluation_id
    roleplay_input = {
        "sample_id": sample_id,
        "target_cefr_level": "auto",
        "roleplay_id": "rp-1",
        "roleplay_task": roleplay_task or "unknown",
        "jfs_can_do_criteria": jfs_can_do_criteria or [],
        "raw_transcript_hiragana": objective_data["raw_transcript_hiragana"],
        "fluency_metrics": objective_data["fluency_metrics"],
        "speaker_metadata": speaker_metadata or {},
        "optional_expected_information": [],
    }
    roleplay_input = fact_modules.add_packets_to_roleplay_input(roleplay_input)

    judge_failures: list[JudgeFailure] = []
    if judge_mode == "mock":
        judge_results = judge_auto_cefr_with_mock_panel(roleplay_input)
    else:
        active_profile = profile or TuningProfile.default()
        judge_results, judge_failures = judge_auto_cefr_with_live_panel_partial(
            roleplay_input,
            provider_specs or [],
            timeout_sec=timeout_sec,
            system_prompt=compose_auto_cefr_system_prompt(active_profile),
        )
        if not judge_results:
            raise RuntimeError("all LLM Judges failed.")

    deliberation = deliberate_auto_cefr(
        judge_results,
        objective_data=objective_data,
        profile=profile,
    )
    decision = deliberation.final_decision
    task_rating = _aggregate_task_rating(judge_results)
    confidence = _aggregate_confidence(judge_results, decision.final_cefr_level)
    reasons = _build_reasons(
        final_level=decision.final_cefr_level,
        objective_data=objective_data,
        judge_results=judge_results,
        needs_human_review=decision.needs_human_review or bool(judge_failures),
    )

    payload: dict[str, Any] = {
        "id": evaluation_id,
        "external_id": external_id,
        "status": "completed",
        "language": language,
        "judge_mode": judge_mode,
        "fact_modules": sorted(active_modules),
        "final_cefr_level": decision.final_cefr_level,
        "task_rating": task_rating.value,
        "confidence": confidence,
        "needs_human_review": decision.needs_human_review or bool(judge_failures),
        "summary": _build_summary(decision.final_cefr_level, task_rating, judge_results),
        "reasons": reasons,
        "consensus": {
            "method": "auto_cefr_deliberation",
            "raw_cefr_level": deliberation.raw_decision.final_cefr_level,
            "has_strict_majority": deliberation.raw_decision.has_strict_majority,
            "judge_count": len(judge_results),
            "applied_calibration": deliberation.applied_calibration,
            "judge_summaries": list(deliberation.judge_summaries),
            "calibration_matches": [
                {
                    "sample_id": match.sample_id,
                    "previous_level": match.previous_level,
                    "corrected_level": match.corrected_level,
                    "score": match.score,
                    "summary": match.summary,
                }
                for match in deliberation.calibration_matches
            ],
            "conclusion": deliberation.conclusion,
        },
        "judge_results": [result.to_dict() for result in judge_results],
        "judge_failures": [_judge_failure_to_dict(failure) for failure in judge_failures],
        "created_at": created_at,
        "completed_at": _now_iso(),
    }
    if include_objective_data:
        payload["objective_data"] = _public_objective_data(objective_data)
    return payload


def _aggregate_task_rating(results: list[AutoLevelJudgeResult]) -> Rating:
    counts = Counter(result.task_rating for result in results)
    rating, count = counts.most_common(1)[0]
    if count > len(results) / 2:
        return rating
    sorted_ratings = sorted((result.task_rating for result in results), key=TASK_RATING_ORDER.get)
    if len(sorted_ratings) == 2:
        return sorted_ratings[0]
    return sorted_ratings[len(sorted_ratings) // 2]


def _aggregate_confidence(results: list[AutoLevelJudgeResult], final_level: str) -> float:
    matching = [result.confidence for result in results if result.predicted_cefr_level == final_level]
    values = matching or [result.confidence for result in results]
    return round(sum(values) / len(values), 2) if values else 0.0


def _build_summary(
    final_level: str,
    task_rating: Rating,
    judge_results: list[AutoLevelJudgeResult],
) -> str:
    matching = [result for result in judge_results if result.predicted_cefr_level == final_level]
    rationale = (matching or judge_results)[0].rationale if judge_results else ""
    return f"{final_level}相当 / task={task_rating.value}。{rationale}".strip()


def _build_reasons(
    *,
    final_level: str,
    objective_data: dict[str, Any],
    judge_results: list[AutoLevelJudgeResult],
    needs_human_review: bool,
) -> list[dict[str, str]]:
    evidence = [
        item
        for result in judge_results
        if result.predicted_cefr_level == final_level
        for item in result.evidence
    ]
    if not evidence:
        evidence = [item for result in judge_results for item in result.evidence]

    transcript_reason = _first_prefixed(evidence, "transcript:")
    metrics_reason = _first_prefixed(evidence, "metrics:")
    boundary_reason = _first_prefixed(evidence, "boundary:")
    metrics = objective_data.get("fluency_metrics", {})
    transcript = str(objective_data.get("raw_transcript_hiragana", ""))

    return [
        {
            "type": "transcript",
            "text": transcript_reason
            or f"ひらがなTranscriptは{len(transcript)}文字で、発話内容の量とまとまりを判定材料にした。",
        },
        {
            "type": "metrics",
            "text": metrics_reason
            or (
                "流暢性指標として "
                f"speech_ratio_pct={metrics.get('speech_ratio_pct')}, "
                f"mora_per_sec={metrics.get('mora_per_sec')}, "
                f"max_pause_sec={metrics.get('max_pause_sec')} を判定材料にした。"
            ),
        },
        {
            "type": "boundary",
            "text": boundary_reason
            or (
                "Judge間の一致度と上下レベルとの境界から最終レベルを決めた。"
                + (" Judge結果に不一致または失敗があるため人間レビューを推奨する。" if needs_human_review else "")
            ),
        },
    ]


def _first_prefixed(evidence: list[str], prefix: str) -> str | None:
    for item in evidence:
        if item.lower().startswith(prefix):
            return item.split(":", 1)[1].strip()
    return None


def _public_objective_data(objective_data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in objective_data.items() if key != "audio_path"}


def _judge_failure_to_dict(failure: JudgeFailure) -> dict[str, str]:
    return {
        "judge_id": failure.judge_id,
        "provider": failure.provider_spec.provider,
        "model": failure.provider_spec.model,
        "message": failure.message,
    }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
