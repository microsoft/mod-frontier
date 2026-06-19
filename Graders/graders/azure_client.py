"""Thin async wrapper around Azure OpenAI shared by every grader.

Auth uses Azure AD via DefaultAzureCredential, so no API keys are stored.
Endpoint / model config lives in config/endpoint.yaml.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

import yaml
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider
from openai import AsyncAzureOpenAI
from openai import APIError, RateLimitError, APITimeoutError, APIConnectionError

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "endpoint.yaml"


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    with open(path or CONFIG_PATH) as f:
        return yaml.safe_load(f)


def is_reasoning_model(model_alias: str) -> bool:
    """Reasoning models (o*/gpt-5) reject `temperature` and use max_completion_tokens."""
    return model_alias.startswith("o") or model_alias.startswith("gpt-5")


class AzureChatClient:
    """Minimal async Azure OpenAI chat client with retry + concurrency control."""

    def __init__(self, config: dict[str, Any] | None = None, workers: int = 16):
        self.config = dict(config or load_config())
        # Optional per-process overrides so the same spec can be sharded across
        # several Azure resources (each with its own quota) concurrently.
        endpoint_override = os.environ.get("GRADERS_AZURE_ENDPOINT")
        if endpoint_override:
            self.config["azure_endpoint"] = endpoint_override
        api_version_override = os.environ.get("GRADERS_API_VERSION")
        if api_version_override:
            self.config["api_version"] = api_version_override
        self._deployment_override = os.environ.get("GRADERS_DEPLOYMENT")
        self._models = self.config["models"]
        self._sem = asyncio.Semaphore(workers)
        self._credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(self._credential, self.config["scope"])
        self.client = AsyncAzureOpenAI(
            azure_ad_token_provider=token_provider,
            api_version=self.config["api_version"],
            azure_endpoint=self.config["azure_endpoint"],
            max_retries=4,  # SDK retries 429s honoring Retry-After before raising
        )

    def deployment(self, model_alias: str) -> str:
        if self._deployment_override:
            return self._deployment_override
        return self._models.get(model_alias, model_alias)

    async def chat(
        self,
        model_alias: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = 0.0,
        max_retries: int = 6,
        **kwargs: Any,
    ) -> str:
        deployment = self.deployment(model_alias)
        call_kwargs: dict[str, Any] = dict(kwargs)
        if not is_reasoning_model(model_alias) and temperature is not None:
            call_kwargs["temperature"] = temperature

        delay = 1.0
        last_exc: Exception | None = None
        for _attempt in range(max_retries):
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(
                        model=deployment, messages=messages, **call_kwargs
                    )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_exc = e
                await asyncio.sleep(min(delay, 12))
                delay *= 1.7
            except APIError as e:
                last_exc = e
                # 5xx transient gateway errors: retry; 4xx: give up fast.
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise
                await asyncio.sleep(min(delay, 12))
                delay *= 1.7
        raise RuntimeError(f"Chat failed after {max_retries} retries: {last_exc}")

    async def aclose(self):
        await self.client.close()
        await self._credential.close()
