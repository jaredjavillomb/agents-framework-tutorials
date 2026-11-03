"""The one place the LangGraph build creates its chat model, so notebooks and agents use identical settings.

Settings come from `agent_contracts.harness.load_settings()` (ShipFast SPEC.md section 2). LangChain's
`ChatOllama` talks to Ollama's own API, so it can send every setting per request; `qwen3-agents` also
carries them in the model itself, which is what keeps the other two frameworks level.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from agent_contracts.harness import ModelSettings, load_settings


def make_llm(settings: ModelSettings | None = None) -> ChatOllama:
    s = settings or load_settings()
    return ChatOllama(
        model=s.model,
        base_url=s.host,
        temperature=s.temperature,
        seed=s.seed,
        num_ctx=s.num_ctx,
        reasoning=s.think,  # False: thinking off (Ollama's `think`)
    )
