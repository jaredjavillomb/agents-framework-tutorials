"""Reliable structured output from Strands' Ollama provider, with token counts.

Strands offers two ways to get a structured result from Ollama:

- Agent level, `agent(prompt, structured_output_model=Model)`: the schema becomes a tool the model must
  call. Ollama cannot force a tool call, so a small model sometimes answers in text instead, and Strands
  raises `StructuredOutputException`.
- Model level, `OllamaModel.structured_output(...)`: sends Ollama's native JSON-schema `format`, which
  constrains the reply to the schema. Reliable, but it reports no token usage.

For single structured calls this build uses the model-level request, built with the provider's own
`format_request` so it matches what Strands would send, and reads Ollama's token counts from the reply
so the shared harness can count the call.
"""

from __future__ import annotations

from typing import TypeVar

import ollama
from pydantic import BaseModel
from strands.models.ollama import OllamaModel

from agent_contracts.harness import RunMetrics

T = TypeVar("T", bound=BaseModel)


def structured_call(model: OllamaModel, schema: type[T], prompt: str | None = None, system: str | None = None,
                    metrics: RunMetrics | None = None, messages: list[dict] | None = None) -> T:
    """One structured answer. Pass `prompt` for a single user message, or `messages` (Strands message dicts)
    for a whole conversation."""
    messages = messages or [{"role": "user", "content": [{"text": prompt}]}]
    request = model.format_request(messages=messages, system_prompt=system)
    request["format"] = schema.model_json_schema()
    request["stream"] = False
    request.pop("tools", None)
    response = ollama.Client(model.host, **(model.client_args or {})).chat(**request)
    if metrics is not None:
        metrics.add_llm_call(response.prompt_eval_count, response.eval_count)
    return schema.model_validate_json(response.message.content.strip())
