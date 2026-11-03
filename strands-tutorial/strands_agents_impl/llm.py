"""The one place the Strands build creates its model, so notebooks and agents use identical settings.

Strands' `OllamaModel` talks to Ollama's own API, so every setting can be sent per request: temperature,
seed and context size in `options`, and thinking off with Ollama's `think` flag. `qwen3-agents` also
carries the settings in the model itself (DECISIONS.md 12). Strands defaults to Amazon Bedrock when no
model is given, so every agent here is given this one.
"""

from __future__ import annotations

from strands.models.ollama import OllamaModel

from agent_contracts.harness import ModelSettings, load_settings


def make_model(settings: ModelSettings | None = None) -> OllamaModel:
    s = settings or load_settings()
    return OllamaModel(
        s.host,
        model_id=s.model,
        temperature=s.temperature,
        options={"seed": s.seed, "num_ctx": s.num_ctx},
        additional_args={"think": s.think},  # False: thinking off
    )
