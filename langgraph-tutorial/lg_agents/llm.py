"""The one place the LangGraph build creates its chat model, so notebooks and agents use identical settings.

Settings come from `agent_contracts.harness.load_settings()` (ShipFast SPEC.md section 2). LangChain's
`ChatOllama` talks to Ollama's own API, so it can send every setting per request; `qwen3-agents` also
carries them in the model itself, which is what keeps the other two frameworks level.
"""

from __future__ import annotations

from langchain_ollama import ChatOllama

from agent_contracts.harness import ModelSettings, load_settings


def make_llm(settings: ModelSettings | None = None) -> ChatOllama:
    # The shared settings (or the ones passed in). MODEL and OLLAMA_HOST can override the first two.
    s = settings or load_settings()
    return ChatOllama(
        model=s.model,  # "qwen3-agents" unless MODEL says otherwise
        base_url=s.host,  # where Ollama listens, http://localhost:11434 by default
        temperature=s.temperature,  # 0: always the likeliest next token, so runs are close to repeatable
        seed=s.seed,  # a fixed seed (42), for the same reason
        num_ctx=s.num_ctx,  # the context window in tokens (16,384), sent with every request
        reasoning=s.think,  # False: thinking off (Ollama's `think`)
    )
