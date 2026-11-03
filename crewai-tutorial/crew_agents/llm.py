"""The one place the CrewAI build creates its model, so notebooks and agents use identical settings.

CrewAI 1.15 reaches Ollama through its OpenAI-compatible API (`/v1`). That API turns thinking off with
`reasoning_effort="none"` and ignores a per-request context size, which is why the shared settings are
built into the `qwen3-agents` model itself (DECISIONS.md 12). Temperature and seed are also sent.

CrewAI also needs to be told the context size: for a model it does not know it assumes 8,192 tokens and
trims conversations to fit, which would give CrewAI less room than the other two frameworks.
"""


from __future__ import annotations

import logging

from crewai import LLM

from agent_contracts.harness import ModelSettings, load_settings

# CrewAI warns on every call that it estimates prompt tokens itself; the real counts come from Ollama.
logging.getLogger("crewai.llms.base_llm").setLevel(logging.ERROR)


def make_llm(settings: ModelSettings | None = None) -> LLM:
    s = settings or load_settings()
    return LLM(
        model=f"ollama/{s.model}",
        base_url=s.host,
        temperature=s.temperature,
        seed=s.seed,
        reasoning_effort="none",  # thinking off on the OpenAI-style API
        context_window_size=s.num_ctx,  # otherwise CrewAI assumes 8,192 and trims early
    )
