#!/usr/bin/env python3
"""Optional OpenAI-key transport for the ``Graders`` package.

The graders in ``Graders/`` authenticate to Azure OpenAI with Azure AD
(``DefaultAzureCredential`` + ``config/endpoint.yaml``). Environments without
an Azure AD credential can route the same grader specs through plain OpenAI
(``api.openai.com``) keyed by ``OPENAI_API_KEY``: this module provides a
drop-in client with the interface ``graders.core`` uses
(``AzureChatClient(config=None, workers=...)``, ``.chat(model_alias,
messages, ...)``, ``.deployment()``, ``.aclose()``) and an :func:`install`
hook that swaps it into ``graders.core`` without modifying the package.

The grader specs, prompts, ensemble aggregation, parsing, and caching are
untouched -- only transport/auth changes. Model aliases (``gpt-4.1``,
``gpt-4o``, ...) are used as the OpenAI model names directly, so labels come
from the same model families the specs pin.

Selection: ``rewriter/eval_e2e.py`` calls :func:`install` when
``GRADERS_AUTH=openai`` is set (or ``--grader-auth openai`` is passed). With
Azure AD available, don't install anything -- the stock client runs as-is.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any

from openai import APIConnectionError, APIError, APITimeoutError, AsyncOpenAI, RateLimitError


def is_reasoning_model(model_alias: str) -> bool:
    """Reasoning models (o*/gpt-5) reject ``temperature``."""
    return model_alias.startswith("o") or model_alias.startswith("gpt-5")


class OpenAIChatClient:
    """Async OpenAI chat client with the retry/concurrency behavior of
    ``graders.azure_client.AzureChatClient``."""

    def __init__(self, config: dict[str, Any] | None = None, workers: int = 16):
        self.config = dict(config or {})
        self._deployment_override = os.environ.get("GRADERS_DEPLOYMENT")
        self._sem = asyncio.Semaphore(workers)
        api_key = os.environ.get("GRADERS_OPENAI_API_KEY") or os.environ["OPENAI_API_KEY"]
        base_url = os.environ.get("GRADERS_OPENAI_BASE_URL")  # None -> api.openai.com
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url, max_retries=4)

    def deployment(self, model_alias: str) -> str:
        """With plain OpenAI the alias is the model name (override still wins)."""
        return self._deployment_override or model_alias

    async def chat(
        self,
        model_alias: str,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = 0.0,
        max_retries: int = 6,
        **kwargs: Any,
    ) -> str:
        model = self._deployment_override or model_alias
        call_kwargs: dict[str, Any] = dict(kwargs)
        # Guard on the EFFECTIVE model, not the alias: with GRADERS_DEPLOYMENT
        # redirecting an alias to a reasoning model (o*/gpt-5*), temperature
        # and max_tokens are rejected by the API.
        if not is_reasoning_model(model) and temperature is not None:
            call_kwargs["temperature"] = temperature
        if is_reasoning_model(model) and "max_tokens" in call_kwargs:
            call_kwargs["max_completion_tokens"] = max(
                int(call_kwargs.pop("max_tokens")), 2000
            )

        delay = 1.0
        last_exc: Exception | None = None
        for _attempt in range(max_retries):
            try:
                async with self._sem:
                    resp = await self.client.chat.completions.create(
                        model=model, messages=messages, **call_kwargs
                    )
                return resp.choices[0].message.content or ""
            except (RateLimitError, APITimeoutError, APIConnectionError) as e:
                last_exc = e
                await asyncio.sleep(min(delay, 12))
                delay *= 1.7
            except APIError as e:
                last_exc = e
                status = getattr(e, "status_code", None)
                if status is not None and 400 <= status < 500:
                    raise
                await asyncio.sleep(min(delay, 12))
                delay *= 1.7
        raise RuntimeError(f"Chat failed after {max_retries} retries: {last_exc}")

    async def aclose(self) -> None:
        await self.client.close()


def install() -> None:
    """Swap :class:`OpenAIChatClient` into ``graders.core`` for this process.

    Import the ``graders`` package first (``rewriter.eval_e2e`` handles the
    path setup). Idempotent.

    When ``GRADERS_DEPLOYMENT`` redirects every alias to one model, the
    graders' disk cache — which keys results by the spec's model *alias* —
    would silently serve override-produced labels to later runs without the
    override (and vice versa). The cache key is therefore extended with the
    effective deployment whenever an override is active.
    """
    from graders import core

    core.AzureChatClient = OpenAIChatClient  # type: ignore[attr-defined]

    deployment = os.environ.get("GRADERS_DEPLOYMENT")
    if deployment and not getattr(core, "_deployment_keyed_cache", None):
        original_cache_key = core._cache_key

        def _deployment_cache_key(model: str, prompt: str, sample: Any) -> str:
            return original_cache_key(f"{model}@deployment={deployment}", prompt, sample)

        core._cache_key = _deployment_cache_key  # type: ignore[attr-defined]
        core._deployment_keyed_cache = deployment  # type: ignore[attr-defined]
        print(f"Grader cache keys scoped to deployment override {deployment!r}", flush=True)
