from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ModelConfig:
    api_key: str
    base_url: str
    model_name: str

    @property
    def chat_completions_url(self) -> str:
        cleaned = self.base_url.rstrip("/")
        if cleaned.endswith("/chat/completions"):
            return cleaned
        return f"{cleaned}/chat/completions"


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens

    @classmethod
    def from_api_usage(cls, usage: dict[str, Any] | None) -> "TokenUsage":
        usage = usage or {}
        return cls(
            prompt_tokens=int(usage.get("prompt_tokens", 0) or 0),
            completion_tokens=int(usage.get("completion_tokens", 0) or 0),
            total_tokens=int(usage.get("total_tokens", 0) or 0),
        )


@dataclass
class ModelResponse:
    content: str
    usage: TokenUsage
    raw: dict[str, Any]


def _strip_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def read_dotenv(path: str | Path = ".env") -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        raise FileNotFoundError(f"Missing .env file at {env_path.resolve()}")

    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _strip_quotes(value.strip())
    return values


def load_model_config(path: str | Path = ".env") -> ModelConfig:
    dotenv_values = read_dotenv(path)
    api_key = os.environ.get("apikey") or os.environ.get("APIKEY") or dotenv_values.get("apikey")
    base_url = os.environ.get("base") or os.environ.get("BASE") or dotenv_values.get("base")
    model_name = os.environ.get("model") or os.environ.get("MODEL") or dotenv_values.get("model")

    missing = [name for name, value in [("apikey", api_key), ("base", base_url), ("model", model_name)] if not value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required model config in .env: {joined}")

    return ModelConfig(api_key=api_key, base_url=base_url, model_name=model_name)


class ModelClient:
    def __init__(
        self,
        env_path: str | Path = ".env",
        timeout: int = 120,
        max_retries: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        self.config = load_model_config(env_path)
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self._usage = TokenUsage()

    @property
    def model_name(self) -> str:
        return self.config.model_name

    def generate(self, system_prompt: str, user_prompt: str, temperature: float = 0.2) -> ModelResponse:
        import requests

        payload = {
            "model": self.config.model_name,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        retryable_status_codes = {408, 409, 425, 429, 500, 502, 503, 504}

        last_exc: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = requests.post(
                    self.config.chat_completions_url,
                    headers=headers,
                    json=payload,
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                break
            except requests.HTTPError as exc:
                status_code = exc.response.status_code if exc.response is not None else None
                is_retryable = status_code in retryable_status_codes
                if attempt >= self.max_retries or not is_retryable:
                    raise
                last_exc = exc
            except requests.RequestException as exc:
                if attempt >= self.max_retries:
                    raise
                last_exc = exc

            sleep_seconds = self.retry_backoff_seconds * (2**attempt)
            time.sleep(sleep_seconds)
        else:
            raise RuntimeError(f"Model request failed after retries: {last_exc}")

        choices = data.get("choices") or []
        if not choices:
            raise ValueError("Model response did not include any choices.")
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            text_parts = [part.get("text", "") for part in content if isinstance(part, dict)]
            content = "\n".join(part for part in text_parts if part)
        if content is None:
            content = ""
        usage = TokenUsage.from_api_usage(data.get("usage"))
        self._usage.add(usage)
        return ModelResponse(content=str(content), usage=usage, raw=data)

    def get_usage(self) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self._usage.prompt_tokens,
            completion_tokens=self._usage.completion_tokens,
            total_tokens=self._usage.total_tokens,
        )

    def save_usage(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model_name,
            "usage": asdict(self.get_usage()),
        }
        output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
