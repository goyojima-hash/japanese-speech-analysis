from __future__ import annotations

import json
import os
import ssl
from dataclasses import dataclass
from pathlib import Path
from urllib import error, parse, request

from .models import AutoLevelJudgeResult, JudgeResult
from .prompts import build_auto_cefr_judge_messages, build_judge_messages


DEFAULT_LIVE_PROVIDER_SPECS = (
    "anthropic:claude-sonnet-4-6",
    "openai:gpt-5.4-mini",
    "gemini:gemini-3.1-pro-preview",
)

PROVIDER_MODEL_OPTIONS: dict[str, list[str]] = {
    "anthropic": ["claude-sonnet-4-6", "claude-opus-4-8", "claude-haiku-4-5"],
    "openai": ["gpt-5.4-mini", "gpt-5.4", "gpt-5.5"],
    "gemini": ["gemini-3.1-pro-preview", "gemini-3.5-flash", "gemini-flash-latest"],
    "xai": ["grok-4.3", "latest", "grok-build-0.1"],
    "groq": ["openai/gpt-oss-120b", "llama-3.3-70b-versatile"],
}

PROVIDER_KEY_ENVS: dict[str, tuple[str, ...]] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "openai": ("OPENAI_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "xai": ("XAI_API_KEY",),
    "groq": ("GROQ_API_KEY",),
}


@dataclass(frozen=True)
class ProviderSpec:
    provider: str
    model: str


class JudgeProviderError(RuntimeError):
    """Raised when a live LLM Judge call fails."""


@dataclass(frozen=True)
class JudgeFailure:
    judge_id: str
    provider_spec: ProviderSpec
    message: str


@dataclass(frozen=True)
class ProviderKeyStatus:
    provider: str
    env_name: str
    state: str
    message: str

    @property
    def ok(self) -> bool:
        return self.state in {"ok", "valid"}


class ProviderKeyValidationHTTPError(RuntimeError):
    """Raised when provider key validation receives an HTTP error."""

    def __init__(self, code: int, body: str):
        self.code = code
        self.body = body
        super().__init__(f"Provider HTTP error {code}: {body}")


def parse_provider_specs(value: str | None) -> list[ProviderSpec]:
    raw_specs = (
        DEFAULT_LIVE_PROVIDER_SPECS
        if value is None or not value.strip()
        else tuple(part.strip() for part in value.split(",") if part.strip())
    )
    specs: list[ProviderSpec] = []
    for raw_spec in raw_specs:
        if ":" not in raw_spec:
            raise ValueError(f"judge provider must be provider:model, got {raw_spec!r}")
        provider, model = raw_spec.split(":", 1)
        provider = provider.strip().lower()
        model = model.strip()
        if not provider or not model:
            raise ValueError(f"judge provider must be provider:model, got {raw_spec!r}")
        specs.append(ProviderSpec(provider=provider, model=model))
    _validate_provider_specs(specs)
    return specs


def format_provider_spec(spec: ProviderSpec) -> str:
    return f"{spec.provider}:{spec.model}"


def missing_key_envs(provider_specs: list[ProviderSpec]) -> list[str]:
    missing: list[str] = []
    for spec in provider_specs:
        env_names = PROVIDER_KEY_ENVS.get(spec.provider, ())
        if env_names and not any(os.environ.get(name) for name in env_names):
            missing.append("/".join(env_names))
    return missing


def provider_key_status(provider: str) -> ProviderKeyStatus:
    env_names = PROVIDER_KEY_ENVS.get(provider, ())
    if not env_names:
        return ProviderKeyStatus(provider, "", "ok", "APIキーは不要です。")

    for env_name in env_names:
        value = os.environ.get(env_name, "").strip()
        if value:
            if provider == "gemini" and not value.startswith("AIza"):
                return ProviderKeyStatus(
                    provider,
                    env_name,
                    "invalid",
                    (
                        f"{env_name} は設定済みですが、Google AI StudioのAPIキー形式に見えません。"
                        "通常は AIza で始まるキーです。"
                    ),
                )
            return ProviderKeyStatus(provider, env_name, "ok", f"{env_name} は設定済みです。")

    return ProviderKeyStatus(
        provider,
        env_names[0],
        "missing",
        f"{'/'.join(env_names)} が未設定です。",
    )


def validate_provider_key(
    spec: ProviderSpec,
    *,
    timeout_sec: float = 6,
) -> ProviderKeyStatus:
    status = provider_key_status(spec.provider)
    if status.state in {"missing", "invalid"}:
        return status
    if not status.env_name:
        return status

    endpoint = _key_validation_endpoint(spec, status.env_name)
    if endpoint is None:
        return ProviderKeyStatus(
            spec.provider,
            status.env_name,
            "unchecked",
            f"{format_provider_spec(spec)} はAPIキーの疎通確認に未対応です。",
        )

    try:
        _get_json(endpoint["url"], headers=endpoint["headers"], timeout_sec=timeout_sec)
    except ProviderKeyValidationHTTPError as exc:
        if exc.code in {400, 401, 403} or _looks_like_auth_error(exc.body):
            return ProviderKeyStatus(
                spec.provider,
                status.env_name,
                "invalid",
                f"{status.env_name} は設定済みですが、プロバイダAPIの認証に失敗しました。",
            )
        return ProviderKeyStatus(
            spec.provider,
            status.env_name,
            "unchecked",
            f"{status.env_name} は設定済みですが、API疎通確認を完了できませんでした。",
        )
    except Exception as exc:
        return ProviderKeyStatus(
            spec.provider,
            status.env_name,
            "unchecked",
            f"{status.env_name} は設定済みですが、API疎通確認を完了できませんでした: {exc}",
        )

    return ProviderKeyStatus(
        spec.provider,
        status.env_name,
        "valid",
        f"{status.env_name} はプロバイダAPIで有効確認済みです。",
    )


def load_env_file(path: Path | None) -> None:
    env_path = path or Path(".env")
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def judge_with_live_panel(
    roleplay_input: dict,
    provider_specs: list[ProviderSpec],
    *,
    timeout_sec: float,
) -> list[JudgeResult]:
    _validate_provider_specs(provider_specs)
    judge_ids = ("A", "B", "C")
    results = []
    for judge_id, spec in zip(judge_ids, provider_specs):
        try:
            messages = build_judge_messages(
                roleplay_input,
                judge_id=judge_id,
                model_family=spec.provider,
            )
            response_text = _call_provider(spec, messages, timeout_sec=timeout_sec)
            results.append(_parse_judge_response(response_text, judge_id, spec))
        except Exception as exc:
            raise JudgeProviderError(
                f"Judge {judge_id} ({format_provider_spec(spec)}) failed: {exc}"
            ) from exc
    return results


def judge_auto_cefr_with_live_panel(
    roleplay_input: dict,
    provider_specs: list[ProviderSpec],
    *,
    timeout_sec: float,
    system_prompt: str | None = None,
) -> list[AutoLevelJudgeResult]:
    _validate_provider_specs(provider_specs)
    judge_ids = ("A", "B", "C")
    results = []
    for judge_id, spec in zip(judge_ids, provider_specs):
        try:
            messages = build_auto_cefr_judge_messages(
                roleplay_input,
                judge_id=judge_id,
                model_family=spec.provider,
                system_prompt=system_prompt,
            )
            results.append(
                _call_and_parse_auto_level(
                    spec,
                    messages,
                    judge_id=judge_id,
                    timeout_sec=timeout_sec,
                )
            )
        except Exception as exc:
            raise JudgeProviderError(
                f"Judge {judge_id} ({format_provider_spec(spec)}) failed: {exc}"
            ) from exc
    return results


def judge_auto_cefr_with_live_panel_partial(
    roleplay_input: dict,
    provider_specs: list[ProviderSpec],
    *,
    timeout_sec: float,
    system_prompt: str | None = None,
) -> tuple[list[AutoLevelJudgeResult], list[JudgeFailure]]:
    _validate_provider_specs(provider_specs)
    judge_ids = ("A", "B", "C")
    results: list[AutoLevelJudgeResult] = []
    failures: list[JudgeFailure] = []
    for judge_id, spec in zip(judge_ids, provider_specs):
        try:
            messages = build_auto_cefr_judge_messages(
                roleplay_input,
                judge_id=judge_id,
                model_family=spec.provider,
                system_prompt=system_prompt,
            )
            results.append(
                _call_and_parse_auto_level(
                    spec,
                    messages,
                    judge_id=judge_id,
                    timeout_sec=timeout_sec,
                )
            )
        except Exception as exc:
            failures.append(
                JudgeFailure(
                    judge_id=judge_id,
                    provider_spec=spec,
                    message=_format_provider_failure(spec, exc),
                )
            )
    return results, failures


def _call_provider(
    spec: ProviderSpec,
    messages: dict[str, str],
    *,
    timeout_sec: float,
) -> str:
    if spec.provider == "anthropic":
        return _call_anthropic(spec, messages, timeout_sec=timeout_sec)
    if spec.provider == "gemini":
        return _call_gemini(spec, messages, timeout_sec=timeout_sec)
    if spec.provider == "openai":
        return _call_openai_responses(spec, messages, timeout_sec=timeout_sec)
    if spec.provider in {"xai", "groq"}:
        return _call_openai_compatible(spec, messages, timeout_sec=timeout_sec)
    if spec.provider == "elevenlabs":
        raise NotImplementedError(
            "ElevenLabs is not configured as a text Judge provider in this pipeline."
        )
    raise ValueError(f"Unsupported judge provider: {spec.provider}")


def _validate_provider_specs(provider_specs: list[ProviderSpec]) -> None:
    if not 1 <= len(provider_specs) <= 3:
        raise ValueError("1 to 3 judge providers are required.")
    providers = [spec.provider for spec in provider_specs]
    duplicates = sorted({provider for provider in providers if providers.count(provider) > 1})
    if duplicates:
        raise ValueError(f"Judge providers must be unique: {', '.join(duplicates)}")


def _key_validation_endpoint(spec: ProviderSpec, env_name: str) -> dict[str, object] | None:
    api_key = os.environ.get(env_name, "").strip()
    if spec.provider == "anthropic":
        return {
            "url": "https://api.anthropic.com/v1/models",
            "headers": {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
            },
        }
    if spec.provider == "openai":
        return {
            "url": "https://api.openai.com/v1/models",
            "headers": {"Authorization": f"Bearer {api_key}"},
        }
    if spec.provider == "gemini":
        return {
            "url": (
                "https://generativelanguage.googleapis.com/v1beta/models"
                f"?key={parse.quote(api_key, safe='')}"
            ),
            "headers": {},
        }
    if spec.provider == "groq":
        return {
            "url": "https://api.groq.com/openai/v1/models",
            "headers": {"Authorization": f"Bearer {api_key}"},
        }
    if spec.provider == "xai":
        return {
            "url": "https://api.x.ai/v1/models",
            "headers": {"Authorization": f"Bearer {api_key}"},
        }
    return None


def _looks_like_auth_error(text: str) -> bool:
    auth_markers = (
        "unauthorized",
        "unauthenticated",
        "invalid api key",
        "invalid_api_key",
        "permission_denied",
        "api key not valid",
        "access_token_type_unsupported",
    )
    lowered = text.lower()
    return any(marker in lowered for marker in auth_markers)


def _format_provider_failure(spec: ProviderSpec, exc: Exception) -> str:
    message = str(exc)
    if spec.provider == "gemini" and (
        "401" in message
        or "UNAUTHENTICATED" in message
        or "ACCESS_TOKEN_TYPE_UNSUPPORTED" in message
    ):
        return (
            f"{format_provider_spec(spec)}: Gemini認証に失敗しました。"
            "GEMINI_API_KEYにはGoogle AI Studioで発行したAPIキーを設定してください。"
            "現在の値はOAuthトークン/別種の認証情報、または無効なキーの可能性があります。"
        )
    return f"{format_provider_spec(spec)}: {message}"


def _call_openai_responses(
    spec: ProviderSpec,
    messages: dict[str, str],
    *,
    timeout_sec: float,
) -> str:
    payload = {
        "model": spec.model,
        "instructions": messages["system"],
        "input": [{"role": "user", "content": messages["user"]}],
        "text": {"format": {"type": "json_object"}},
        "max_output_tokens": 1200,
    }
    data = _post_json(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {_required_env('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout_sec=timeout_sec,
    )
    return _extract_response_output_text(data)


def _call_openai_compatible(
    spec: ProviderSpec,
    messages: dict[str, str],
    *,
    timeout_sec: float,
) -> str:
    api_key = _required_env(_api_key_env(spec.provider))
    base_urls = {
        "openai": "https://api.openai.com/v1/chat/completions",
        "xai": "https://api.x.ai/v1/chat/completions",
        "groq": "https://api.groq.com/openai/v1/chat/completions",
    }
    payload = {
        "model": spec.model,
        "messages": [
            {"role": "system", "content": messages["system"]},
            {"role": "user", "content": messages["user"]},
        ],
        "temperature": 0,
        "max_tokens": 1200,
    }
    if spec.provider == "openai":
        payload["response_format"] = {"type": "json_object"}

    data = _post_json(
        base_urls[spec.provider],
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout_sec=timeout_sec,
    )
    return data["choices"][0]["message"]["content"]


def _call_anthropic(
    spec: ProviderSpec,
    messages: dict[str, str],
    *,
    timeout_sec: float,
) -> str:
    payload = {
        "model": spec.model,
        "max_tokens": 1200,
        "system": messages["system"],
        "messages": [{"role": "user", "content": messages["user"]}],
    }
    data = _post_json(
        "https://api.anthropic.com/v1/messages",
        headers={
            "x-api-key": _required_env("ANTHROPIC_API_KEY"),
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        },
        payload=payload,
        timeout_sec=timeout_sec,
    )
    text_blocks = [
        block.get("text", "")
        for block in data.get("content", [])
        if block.get("type") == "text"
    ]
    return "\n".join(text_blocks)


def _call_gemini(
    spec: ProviderSpec,
    messages: dict[str, str],
    *,
    timeout_sec: float,
) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY or GOOGLE_API_KEY.")
    model = parse.quote(spec.model, safe="")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    payload = {
        "systemInstruction": {"parts": [{"text": messages["system"]}]},
        "contents": [{"role": "user", "parts": [{"text": messages["user"]}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": 2000,
            "responseMimeType": "application/json",
        },
    }
    data = _post_json(
        f"{url}?key={parse.quote(api_key, safe='')}",
        headers={"Content-Type": "application/json"},
        payload=payload,
        timeout_sec=timeout_sec,
    )
    parts = data["candidates"][0]["content"]["parts"]
    return "\n".join(part.get("text", "") for part in parts)


def _post_json(
    url: str,
    *,
    headers: dict[str, str],
    payload: dict,
    timeout_sec: float,
) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec, context=_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise RuntimeError(
            f"Provider HTTP error {exc.code}: {response_body}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Provider request failed: {exc.reason}") from exc


def _get_json(
    url: str,
    *,
    headers: dict[str, str],
    timeout_sec: float,
) -> dict:
    req = request.Request(url, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_sec, context=_ssl_context()) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")[:2000]
        raise ProviderKeyValidationHTTPError(exc.code, response_body) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Provider request failed: {exc.reason}") from exc


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _extract_response_output_text(data: dict) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    text_parts: list[str] = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if text:
                    text_parts.append(text)
    if text_parts:
        return "\n".join(text_parts)
    raise RuntimeError(f"OpenAI response did not contain output text: {data!r}")


def _parse_judge_response(
    response_text: str,
    judge_id: str,
    spec: ProviderSpec,
) -> JudgeResult:
    parsed = json.loads(_extract_json_object(response_text))
    parsed["judge_id"] = judge_id
    parsed["model_family"] = spec.provider
    return JudgeResult.from_dict(parsed)


def _parse_auto_level_response(
    response_text: str,
    judge_id: str,
    spec: ProviderSpec,
) -> AutoLevelJudgeResult:
    parsed = json.loads(_extract_json_object(response_text))
    parsed["judge_id"] = judge_id
    parsed["model_family"] = spec.provider
    return AutoLevelJudgeResult.from_dict(parsed)


def _call_and_parse_auto_level(
    spec: ProviderSpec,
    messages: dict[str, str],
    *,
    judge_id: str,
    timeout_sec: float,
) -> AutoLevelJudgeResult:
    retry_messages = messages
    last_parse_error: Exception | None = None
    for attempt in range(3):
        response_text = _call_provider(spec, retry_messages, timeout_sec=timeout_sec)
        try:
            return _parse_auto_level_response(response_text, judge_id, spec)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            last_parse_error = exc
            if attempt == 2:
                break
            retry_messages = {
                **messages,
                "user": (
                    f"{messages['user']}\n\n"
                    "前回の応答はJSONとして解析できませんでした。"
                    "説明文、Markdown、コードブロックを一切付けず、"
                    "指定schemaを満たす1個の完全なJSON objectだけを返してください。"
                    "必須キー: judge_id, model_family, predicted_cefr_level, "
                    "task_rating, confidence, rationale, evidence, risk_flags。"
                    "途中で切れないよう、rationaleは120文字以内、"
                    "evidenceは3件以内、risk_flagsは3件以内にしてください。"
                ),
            }
    assert last_parse_error is not None
    raise last_parse_error


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end < start:
        raise ValueError(f"Judge response did not contain a JSON object: {text[:200]}")
    return cleaned[start : end + 1]


def _api_key_env(provider: str) -> str:
    env_names = {
        "openai": "OPENAI_API_KEY",
        "xai": "XAI_API_KEY",
        "groq": "GROQ_API_KEY",
    }
    return env_names[provider]


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing {name}. Set it in the environment or .env.")
    return value
