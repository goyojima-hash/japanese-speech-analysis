from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from . import api_service
from .audio_pipeline import (
    ObjectiveExtractionError,
    evaluate_audio_manifest,
    extract_objective_manifest,
)
from .consensus import ConsensusGate
from .interactive import run_interactive
from .judge_config import DEFAULT_JUDGE_CONFIG_PATH, load_judge_console_config
from .jfs_samples import (
    DEFAULT_CATALOG_PATH,
    DEFAULT_DOWNLOAD_DIR,
    DEFAULT_TEST_AUDIO_DIR,
    DEFAULT_TUNING_DATASET_PATH,
    build_jfs_tuning_dataset,
    download_jfs_assets,
    load_jfs_catalog,
    validate_jfs_catalog,
)
from .key_setup import configure_keys
from .live_judges import load_env_file, parse_provider_specs
from .live_judges import JudgeProviderError
from .metrics import evaluate_against_humans, load_benchmark_items
from .prompts import build_judge_messages
from .tuning_samples import DEFAULT_REVIEW_DATASET_PATH


def main() -> None:
    parser = argparse.ArgumentParser(description="J-GRADE local evaluation tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prompt_parser = subparsers.add_parser("prompt", help="Build a judge prompt payload")
    prompt_parser.add_argument("--input", required=True, type=Path)
    prompt_parser.add_argument("--judge", required=True, choices=["A", "B", "C"])
    prompt_parser.add_argument(
        "--model-family",
        required=True,
        choices=["anthropic", "openai", "gemini", "xai", "groq", "claude", "gpt"],
    )
    prompt_parser.add_argument("--output", type=Path)

    consensus_parser = subparsers.add_parser("consensus", help="Run consensus gate")
    consensus_parser.add_argument("--input", required=True, type=Path)
    consensus_parser.add_argument("--output", type=Path)

    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Compare AI labels against human teacher labels",
    )
    benchmark_parser.add_argument("--input", required=True, type=Path)
    benchmark_parser.add_argument("--output", type=Path)

    audio_parser = subparsers.add_parser(
        "evaluate-audio",
        help="Extract objective audio data and run J-GRADE consensus",
    )
    audio_parser.add_argument("--manifest", required=True, type=Path)
    audio_parser.add_argument(
        "--judge-mode",
        default="mock",
        choices=["mock", "live"],
        help="mock is local only; live calls the selected LLM providers",
    )
    audio_parser.add_argument(
        "--judge-providers",
        help=(
            "comma-separated provider:model list, 1 to 3 unique entries; "
            "example: anthropic:claude-sonnet-4-6,openai:gpt-5.4-mini,gemini:gemini-3.1-pro-preview"
        ),
    )
    audio_parser.add_argument(
        "--env-file",
        type=Path,
        help="optional .env file containing provider API keys; defaults to ./.env",
    )
    audio_parser.add_argument("--timeout-sec", type=float, default=60.0)
    audio_parser.add_argument(
        "--base-dir",
        type=Path,
        help="base directory for relative audio_path entries; defaults to cwd",
    )
    audio_parser.add_argument("--output", type=Path)

    speech_parser = subparsers.add_parser(
        "evaluate-speech",
        help="Run the same five-module and Judge path as the API for one audio file",
    )
    speech_parser.add_argument("--audio", required=True, type=Path)
    speech_parser.add_argument("--judge-mode", choices=["mock", "live"], default="mock")
    speech_parser.add_argument("--judge-providers", help="comma-separated provider:model list for live mode")
    speech_parser.add_argument("--env-file", type=Path)
    speech_parser.add_argument("--timeout-sec", type=float, default=60.0)
    speech_parser.add_argument("--roleplay-task", default="unknown")
    speech_parser.add_argument("--modules", help="comma-separated fact module names; default is all five")
    speech_parser.add_argument("--task-context-file", type=Path,
                               help="JSON task context; confirmed transcript spans are required for shadow observation")
    speech_parser.add_argument("--assessment-mode", choices=["off", "shadow"], default="off")
    speech_parser.add_argument("--output", type=Path, help="save the legacy and shadow results together as JSON")

    objective_parser = subparsers.add_parser(
        "extract-objective",
        help="Extract raw hiragana transcript and timing metrics only",
    )
    objective_parser.add_argument("--manifest", required=True, type=Path)
    objective_parser.add_argument(
        "--base-dir",
        type=Path,
        help="base directory for relative audio_path entries; defaults to cwd",
    )
    objective_parser.add_argument("--output", type=Path)

    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Choose one audio file, show objective data, Judge results, and consensus",
    )
    interactive_parser.add_argument(
        "--judge-config",
        type=Path,
        default=DEFAULT_JUDGE_CONFIG_PATH,
        help="JSON config for collaborator console testing",
    )
    interactive_parser.add_argument("--audio-dir", type=Path)
    interactive_parser.add_argument(
        "--judge-mode",
        default=None,
        choices=["mock", "live"],
        help="override judge_mode from the config; mock is local only; live calls LLM providers",
    )
    interactive_parser.add_argument(
        "--judge-providers",
        help=(
            "comma-separated provider:model list, 1 to 3 unique entries; "
            "example: anthropic:claude-sonnet-4-6,openai:gpt-5.4-mini,gemini:gemini-3.1-pro-preview"
        ),
    )
    interactive_parser.add_argument(
        "--env-file",
        type=Path,
        help="optional .env file containing provider API keys; defaults to ./.env",
    )
    interactive_parser.add_argument("--timeout-sec", type=float, default=60.0)
    interactive_parser.add_argument(
        "--profile",
        type=Path,
        help="override profile from the config; used for prompt tuning and exact CEFR overrides",
    )
    interactive_parser.add_argument(
        "--review-store",
        type=Path,
        default=DEFAULT_REVIEW_DATASET_PATH,
        help="where judged samples are accumulated for the tuning UI",
    )

    keys_parser = subparsers.add_parser(
        "configure-keys",
        help="Prompt for provider API keys and write a local .env file",
    )
    keys_parser.add_argument("--env-file", type=Path, default=Path(".env"))

    catalog_parser = subparsers.add_parser(
        "show-jfs-catalog",
        help="Validate and print the official JF Standard role-play sample catalog",
    )
    catalog_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    catalog_parser.add_argument("--output", type=Path)

    download_parser = subparsers.add_parser(
        "download-jfs-samples",
        help="Download official JF Standard role-play audio locally without tracking it in git",
    )
    download_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    download_parser.add_argument("--output-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    download_parser.add_argument("--audio-dir", type=Path, default=DEFAULT_TEST_AUDIO_DIR)
    download_parser.add_argument(
        "--skip-pdfs",
        action="store_true",
        help="download only audio files; evaluation PDFs are downloaded by default",
    )
    download_parser.add_argument("--force", action="store_true")
    download_parser.add_argument("--output", type=Path)

    dataset_parser = subparsers.add_parser(
        "prepare-jfs-dataset",
        help="Build a tuning dataset from downloaded official JF Standard samples",
    )
    dataset_parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    dataset_parser.add_argument(
        "--audio-dir",
        type=Path,
        default=DEFAULT_TEST_AUDIO_DIR,
    )
    dataset_parser.add_argument("--allow-missing", action="store_true")
    dataset_parser.add_argument("--output", type=Path, default=DEFAULT_TUNING_DATASET_PATH)

    args = parser.parse_args()

    if args.command == "prompt":
        roleplay_input = _read_json(args.input)
        result = build_judge_messages(roleplay_input, args.judge, args.model_family)
    elif args.command == "consensus":
        payload = _read_json(args.input)
        result = ConsensusGate().decide_payload(payload).to_dict()
    elif args.command == "benchmark":
        payload = _read_json(args.input)
        result = evaluate_against_humans(load_benchmark_items(payload))
    elif args.command == "evaluate-audio":
        load_env_file(args.env_file)
        manifest = _read_json(args.manifest)
        try:
            result = evaluate_audio_manifest(
                manifest,
                base_dir=args.base_dir or Path.cwd(),
                judge_mode=args.judge_mode,
                provider_specs=(
                    parse_provider_specs(args.judge_providers)
                    if args.judge_mode == "live"
                    else None
                ),
                timeout_sec=args.timeout_sec,
            )
        except ObjectiveExtractionError as exc:
            parser.exit(2, f"[エラー] 客観データ抽出に失敗しました。\n{exc}\n")
        except JudgeProviderError as exc:
            parser.exit(2, f"[エラー] LLM Judge呼び出しに失敗しました。\n{exc}\n")
    elif args.command == "evaluate-speech":
        load_env_file(args.env_file)
        context = _read_json(args.task_context_file) if args.task_context_file else None
        if context is not None and not isinstance(context, dict):
            parser.error("--task-context-file must contain a JSON object")
        roleplay_task = args.roleplay_task
        if roleplay_task == "unknown" and context is not None:
            prompts = context.get("prompts")
            if isinstance(prompts, list) and len(prompts) == 1 and isinstance(prompts[0], dict):
                roleplay_task = str(prompts[0].get("text") or "unknown")
        modules = tuple(item.strip() for item in args.modules.split(",")) if args.modules is not None else None
        if modules is not None and (not modules or any(not item for item in modules)):
            parser.error("--modules requires comma-separated module names")
        result = api_service.evaluate_speech_level(
            args.audio,
            roleplay_task=roleplay_task,
            judge_mode=args.judge_mode,
            provider_specs=(parse_provider_specs(args.judge_providers)
                            if args.judge_mode == "live" and args.judge_providers else None),
            timeout_sec=args.timeout_sec,
            selected_modules=modules,
            task_context=context,
            assessment_mode=args.assessment_mode,
        )
    elif args.command == "extract-objective":
        manifest = _read_json(args.manifest)
        try:
            result = extract_objective_manifest(
                manifest,
                base_dir=args.base_dir or Path.cwd(),
            )
        except ObjectiveExtractionError as exc:
            parser.exit(2, f"[エラー] 客観データ抽出に失敗しました。\n{exc}\n")
    elif args.command == "interactive":
        load_env_file(args.env_file)
        judge_config = load_judge_console_config(args.judge_config)
        judge_mode = args.judge_mode or judge_config.judge_mode
        provider_specs = (
            parse_provider_specs(args.judge_providers)
            if args.judge_providers
            else list(judge_config.judge_providers)
        )
        run_interactive(
            audio_dir=args.audio_dir or judge_config.audio_dir,
            judge_mode=judge_mode,
            provider_specs=provider_specs,
            timeout_sec=args.timeout_sec,
            profile_path=args.profile or judge_config.profile_path,
            review_store_path=args.review_store,
        )
        return
    elif args.command == "configure-keys":
        configure_keys(args.env_file)
        return
    elif args.command == "show-jfs-catalog":
        result = load_jfs_catalog(args.catalog)
        validate_jfs_catalog(result)
    elif args.command == "download-jfs-samples":
        result = download_jfs_assets(
            catalog_path=args.catalog,
            output_dir=args.output_dir,
            audio_dir=args.audio_dir,
            include_pdfs=not args.skip_pdfs,
            force=args.force,
        )
    elif args.command == "prepare-jfs-dataset":
        result = build_jfs_tuning_dataset(
            catalog_path=args.catalog,
            audio_dir=args.audio_dir,
            require_audio=not args.allow_missing,
        )
    else:
        parser.error(f"unsupported command: {args.command}")

    _write_or_print(result, args.output)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def _write_or_print(data: Any, output_path: Path | None) -> None:
    text = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
        print(f"wrote {output_path}")
        return
    print(text, end="")
