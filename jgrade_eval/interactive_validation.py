"""Opt-in shadow validation after normal terminal evaluation; no audio upload."""

from __future__ import annotations

import json
from typing import Any

from .live_judges import ProviderSpec
from .task_assessment import assess_shadow_tasks
from .task_context import TaskContext
from .task_observation import observe_task_with_provider
from .task_rubrics import TaskAssessment, load_default_task_rubrics


def _explicit_yes(message: str) -> bool:
    try:
        return input(message).strip().lower() in {"y", "yes", "on"}
    except EOFError:
        return False


def prompt_validation_mode() -> bool:
    print("\n=== 発話行為・基準変換の並行検証 ===")
    print("既定OFF。ON時だけ観測AIを追加呼出しします（追加費用・待ち時間あり）。")
    print("通常のCEFR結果は変更しません。音声は送らず、観測・基準版は判定履歴に別保存します。")
    return _explicit_yes("検証をONにしますか？ [y/N]: ")


def run_validation(
    *,
    enabled: bool,
    objective_data: dict[str, Any],
    interaction_context: dict[str, Any],
    judge_mode: str,
    provider_specs: list[ProviderSpec],
    timeout_sec: float,
) -> dict[str, Any] | None:
    if not enabled:
        return None
    print("\n=== 並行検証（通常のCEFRとは別・精度は未測定） ===")
    try:
        result = _validate(objective_data, interaction_context, judge_mode, provider_specs, timeout_sec)
    except Exception as error:
        # This optional diagnostic must never replace a completed normal result.
        result = {"mode": "shadow", "results": [{"status": "validation_failed", "reason": type(error).__name__}]}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def _validate(objective_data, interaction_context, judge_mode, provider_specs, timeout_sec):
    raw_context = {"prompts": interaction_context.get("prompts", []),
                   "answers": interaction_context.get("answer_segments", [])}
    if len(raw_context["prompts"]) == 1 and not raw_context["answers"] and judge_mode == "live":
        print("回答範囲の確認: 録音全体が、この１設問に対する回答ですか？")
        if _explicit_yes("確認できた場合のみ y [y/N]: "):
            raw_context["whole_recording_answer_prompt_id"] = raw_context["prompts"][0]["prompt_id"]
    context = TaskContext.from_inputs(
        roleplay_task=None, interaction_context=None, task_context=raw_context,
        transcript=objective_data["raw_transcript_hiragana"],
        duration_sec=objective_data["fluency_metrics"]["audio_duration_sec"],
    )
    if not context.prompts:
        results = [TaskAssessment(status="insufficient_context", reason="no prompt")]
    elif judge_mode == "mock":
        results = [TaskAssessment(status="mock_unavailable", prompt_id=p.prompt_id,
                                  reason="mock Judge is not a speech-act model") for p in context.prompts]
    elif not provider_specs:
        results = [TaskAssessment(status="provider_unavailable", prompt_id=p.prompt_id) for p in context.prompts]
    else:
        registry_error = None
        # No confirmed answers means no rubric loading and no AI calls.
        registry = {}
        if any(context.confirmed_answer_for(p.prompt_id) for p in context.prompts):
            try:
                registry = load_default_task_rubrics()
            except Exception as error:
                registry_error = type(error).__name__
        spec = provider_specs[0]

        def observe(prompt, answer):
            return observe_task_with_provider(prompt, answer, spec, timeout_sec=min(timeout_sec, 20.0))

        results = list(assess_shadow_tasks(context, registry, observe=observe))
        if registry_error:
            results = [TaskAssessment(status="rubric_registry_failed", prompt_id=r.prompt_id,
                                      observation=r.observation, reason=registry_error)
                       if r.status == "rubric_unavailable" else r for r in results]
    return {"mode": "shadow", "task_context": context.to_dict(),
            "results": [result.to_dict() for result in results]}
