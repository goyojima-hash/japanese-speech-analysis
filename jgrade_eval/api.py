from __future__ import annotations

import json
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from starlette.applications import Starlette
from starlette.datastructures import UploadFile
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .api_service import evaluate_speech_level
from .audio_pipeline import ObjectiveExtractionError
from .live_judges import load_env_file, parse_provider_specs


SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a"}
EVALUATIONS: dict[str, dict[str, Any]] = {}


async def health(_: Request) -> JSONResponse:
    return JSONResponse({"data": {"status": "ok"}})


async def create_speech_level_evaluation(request: Request) -> JSONResponse:
    temp_dir = Path(tempfile.mkdtemp(prefix="jgrade-api-"))
    try:
        payload, audio_path = await _read_request_payload(request, temp_dir)
        provider_specs = (
            parse_provider_specs(payload.get("judge_providers"))
            if payload.get("judge_mode", "mock") == "live"
            else None
        )
        load_env_file(Path(payload["env_file"])) if payload.get("env_file") else load_env_file(None)
        result = evaluate_speech_level(
            audio_path,
            external_id=payload.get("external_id"),
            language=payload.get("language", "ja"),
            roleplay_task=payload.get("roleplay_task", "unknown"),
            jfs_can_do_criteria=payload.get("jfs_can_do_criteria", []),
            speaker_metadata=payload.get("speaker_metadata", {}),
            judge_mode=payload.get("judge_mode", "mock"),
            provider_specs=provider_specs,
            timeout_sec=float(payload.get("timeout_sec", 60.0)),
            include_objective_data=_parse_bool(payload.get("include_objective_data"), default=True),
            selected_modules=_parse_optional_module_list(payload.get("fact_modules")),
        )
        EVALUATIONS[result["id"]] = result
        return JSONResponse({"data": result}, status_code=201)
    except ValueError as exc:
        return _error_response("validation_error", str(exc), status_code=422)
    except ObjectiveExtractionError as exc:
        return _error_response("audio_processing_failed", str(exc), status_code=422)
    except Exception as exc:
        return _error_response("evaluation_failed", str(exc), status_code=500)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


async def get_speech_level_evaluation(request: Request) -> JSONResponse:
    evaluation_id = request.path_params["evaluation_id"]
    result = EVALUATIONS.get(evaluation_id)
    if result is None:
        return _error_response("not_found", f"evaluation not found: {evaluation_id}", status_code=404)
    return JSONResponse({"data": result})


async def _read_request_payload(request: Request, temp_dir: Path) -> tuple[dict[str, Any], Path]:
    content_type = request.headers.get("content-type", "")
    if content_type.startswith("multipart/form-data"):
        form = await request.form()
        upload = form.get("audio")
        if not isinstance(upload, UploadFile):
            raise ValueError("multipart request requires an audio file field named 'audio'.")
        audio_path = temp_dir / _safe_filename(upload.filename or "audio.mp3")
        _validate_audio_path(audio_path)
        try:
            audio_path.write_bytes(await upload.read())
        finally:
            await upload.close()
        payload = {
            "external_id": _optional_form_value(form.get("external_id")),
            "language": str(form.get("language") or "ja"),
            "roleplay_task": str(form.get("roleplay_task") or "unknown"),
            "jfs_can_do_criteria": _parse_list_field(form.get("jfs_can_do_criteria")),
            "speaker_metadata": _parse_json_object_field(form.get("speaker_metadata")),
            "judge_mode": str(form.get("judge_mode") or "mock"),
            "judge_providers": _optional_form_value(form.get("judge_providers")),
            "timeout_sec": float(form.get("timeout_sec") or 60.0),
            "include_objective_data": _parse_bool(form.get("include_objective_data"), default=True),
            "fact_modules": _parse_optional_module_list(form.get("fact_modules")),
            "env_file": _optional_form_value(form.get("env_file")),
        }
        return payload, audio_path

    if content_type.startswith("application/json"):
        payload = dict(await request.json())
        audio_url = str(payload.get("audio_url") or "")
        if not audio_url:
            raise ValueError("JSON request requires audio_url.")
        audio_path = temp_dir / _filename_from_url(audio_url)
        _validate_audio_path(audio_path)
        response = requests.get(audio_url, timeout=60)
        response.raise_for_status()
        audio_path.write_bytes(response.content)
        payload.setdefault("language", "ja")
        payload.setdefault("roleplay_task", "unknown")
        payload.setdefault("jfs_can_do_criteria", [])
        payload.setdefault("speaker_metadata", {})
        payload.setdefault("judge_mode", "mock")
        payload["include_objective_data"] = _parse_bool(
            payload.get("include_objective_data"),
            default=True,
        )
        return payload, audio_path

    raise ValueError("content-type must be multipart/form-data or application/json.")


def _safe_filename(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", Path(filename).name).strip(".-")
    return cleaned or "audio.mp3"


def _filename_from_url(audio_url: str) -> str:
    parsed = urlparse(audio_url)
    return _safe_filename(Path(parsed.path).name or "audio.mp3")


def _validate_audio_path(path: Path) -> None:
    if path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
        allowed = ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        raise ValueError(f"audio file extension must be one of: {allowed}")


def _parse_list_field(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    text = str(value).strip()
    if not text:
        return []
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise ValueError("jfs_can_do_criteria must be a JSON array.")
        return [str(item) for item in parsed]
    return [part.strip() for part in text.splitlines() if part.strip()]


def _parse_optional_module_list(value: Any) -> list[str] | None:
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = _parse_list_field(value)
    return parsed or None


def _parse_json_object_field(value: Any) -> dict[str, Any]:
    if value is None or value == "":
        return {}
    parsed = json.loads(str(value))
    if not isinstance(parsed, dict):
        raise ValueError("speaker_metadata must be a JSON object.")
    return parsed


def _parse_bool(value: Any, *, default: bool) -> bool:
    if value is None or value == "":
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _optional_form_value(value: Any) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def _error_response(code: str, message: str, *, status_code: int) -> JSONResponse:
    return JSONResponse(
        {"error": {"code": code, "message": message}},
        status_code=status_code,
    )


app = Starlette(
    debug=False,
    routes=[
        Route("/health", health, methods=["GET"]),
        Route(
            "/api/v1/speech-level-evaluations",
            create_speech_level_evaluation,
            methods=["POST"],
        ),
        Route(
            "/api/v1/speech-level-evaluations/{evaluation_id}",
            get_speech_level_evaluation,
            methods=["GET"],
        ),
    ],
)
