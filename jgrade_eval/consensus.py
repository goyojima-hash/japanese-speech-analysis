from __future__ import annotations

from collections import Counter
from typing import Any

from .models import (
    CEFR_LEVELS,
    AutoLevelDecision,
    AutoLevelJudgeResult,
    JudgeResult,
    LevelDecision,
    Rating,
    RoleplayDecision,
)


TASK_RATING_ORDER = {
    Rating.FAIL: 0,
    Rating.NEAR_FAIL: 1,
    Rating.PASS: 2,
    Rating.EXCELLENT: 3,
}


def aggregate_task_rating(results: list[AutoLevelJudgeResult]) -> Rating:
    """Use one task-rating rule across API, batch, and interactive entrypoints."""

    if not results:
        raise ValueError("at least one Judge result is required.")
    counts = Counter(result.task_rating for result in results)
    rating, count = counts.most_common(1)[0]
    if count > len(results) / 2:
        return rating

    sorted_ratings = sorted(
        (result.task_rating for result in results),
        key=TASK_RATING_ORDER.get,
    )
    if len(sorted_ratings) == 2:
        return sorted_ratings[0]
    return sorted_ratings[len(sorted_ratings) // 2]


class ConsensusGate:
    """Aggregate independent LLM judge ratings into a CEFR level decision."""

    severity_order = {
        Rating.FAIL: 0,
        Rating.NEAR_FAIL: 1,
        Rating.PASS: 2,
        Rating.EXCELLENT: 3,
    }

    def __init__(self, levels: list[str] | None = None) -> None:
        self.levels = levels or CEFR_LEVELS

    def decide_roleplay(
        self,
        roleplay_id: str,
        judge_results: list[JudgeResult] | tuple[JudgeResult, ...],
    ) -> RoleplayDecision:
        if not 1 <= len(judge_results) <= 3:
            raise ValueError("1 to 3 judge results are required per roleplay.")

        results = tuple(judge_results)
        counts = Counter(result.rating for result in results)
        rating, count = counts.most_common(1)[0]

        if len(results) == 1:
            return RoleplayDecision(
                roleplay_id=roleplay_id,
                consensus_rating=rating,
                judge_results=results,
                has_strict_majority=False,
                needs_human_review=True,
            )

        if count > len(results) / 2:
            return RoleplayDecision(
                roleplay_id=roleplay_id,
                consensus_rating=rating,
                judge_results=results,
                has_strict_majority=True,
                needs_human_review=False,
            )

        sorted_ratings = sorted(
            (result.rating for result in results),
            key=lambda rating_value: self.severity_order[rating_value],
        )
        median_rating = sorted_ratings[len(sorted_ratings) // 2]
        if len(sorted_ratings) == 2:
            median_rating = sorted_ratings[0]
        return RoleplayDecision(
            roleplay_id=roleplay_id,
            consensus_rating=median_rating,
            judge_results=results,
            has_strict_majority=False,
            needs_human_review=True,
        )

    def decide_level(
        self,
        tested_level: str,
        roleplay_decisions: list[RoleplayDecision] | tuple[RoleplayDecision, ...],
    ) -> LevelDecision:
        decisions = tuple(roleplay_decisions)
        pass_count = sum(decision.consensus_rating.is_passing for decision in decisions)
        fail_count = len(decisions) - pass_count

        if pass_count >= 2:
            return LevelDecision(tested_level, tested_level, True, decisions)
        if fail_count >= 2:
            return LevelDecision(tested_level, self._downgrade(tested_level), False, decisions)

        # Fallback for incomplete or unusual test sets. Production callers should
        # keep the human-review flag visible instead of silently passing.
        return LevelDecision(tested_level, self._downgrade(tested_level), False, decisions)

    def decide_payload(self, payload: dict[str, Any]) -> LevelDecision:
        tested_level = str(payload["tested_level"])
        roleplay_decisions = []
        for roleplay in payload["roleplays"]:
            results = [
                JudgeResult.from_dict(result)
                for result in roleplay["judge_results"]
            ]
            roleplay_decisions.append(
                self.decide_roleplay(str(roleplay["roleplay_id"]), results)
            )
        return self.decide_level(tested_level, roleplay_decisions)

    def _downgrade(self, level: str) -> str:
        if level not in self.levels:
            allowed = ", ".join(self.levels)
            raise ValueError(f"tested_level must be one of: {allowed}")
        index = self.levels.index(level)
        return self.levels[max(0, index - 1)]


class AutoCefrConsensus:
    """Aggregate independent Judge CEFR predictions."""

    def __init__(self, levels: list[str] | None = None) -> None:
        self.levels = levels or CEFR_LEVELS

    def decide(
        self,
        judge_results: list[AutoLevelJudgeResult] | tuple[AutoLevelJudgeResult, ...],
    ) -> AutoLevelDecision:
        if not 1 <= len(judge_results) <= 3:
            raise ValueError("1 to 3 judge results are required.")

        results = tuple(judge_results)
        counts = Counter(result.predicted_cefr_level for result in results)
        level, count = counts.most_common(1)[0]

        if len(results) == 1:
            return AutoLevelDecision(
                final_cefr_level=level,
                judge_results=results,
                has_strict_majority=False,
                needs_human_review=True,
            )

        if count > len(results) / 2:
            return AutoLevelDecision(
                final_cefr_level=level,
                judge_results=results,
                has_strict_majority=True,
                needs_human_review=False,
            )

        sorted_levels = sorted(
            (result.predicted_cefr_level for result in results),
            key=self.levels.index,
        )
        median_level = sorted_levels[len(sorted_levels) // 2]
        if len(sorted_levels) == 2:
            median_level = sorted_levels[0]
        return AutoLevelDecision(
            final_cefr_level=median_level,
            judge_results=results,
            has_strict_majority=False,
            needs_human_review=True,
        )
