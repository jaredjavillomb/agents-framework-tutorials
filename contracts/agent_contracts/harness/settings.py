"""Model settings every build uses (DECISIONS.md 4, 10 and 11).

The model name is written the plain Ollama way (`qwen3:8b`); each build adds its framework's prefix
itself. Every build sends `options()` explicitly on every call and turns thinking off, so no framework
falls back to a different default.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_MODEL = "qwen3:8b"
DEFAULT_HOST = "http://localhost:11434"


@dataclass(frozen=True)
class ModelSettings:
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 16384  # Ollama's own default on the reference Mac is 4096: too small for agents.
    think: bool = False

    def options(self) -> dict:
        """Ollama `options` for every request."""
        return {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx}


def load_settings() -> ModelSettings:
    """Read `MODEL` and `OLLAMA_HOST` from the environment; everything else is fixed by the spec."""
    return ModelSettings(
        model=os.environ.get("MODEL", DEFAULT_MODEL),
        host=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
    )
