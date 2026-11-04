"""Model settings every build uses (DECISIONS.md 4, 10, 11 and 12).

The shared model is `qwen3-agents`, built from `qwen3:8b` by `contracts/ollama/Modelfile` with the
context size, temperature and seed written into the model itself. That is what makes the settings hold
on every API, including the OpenAI-style one CrewAI uses, which ignores a per-request `num_ctx`.
Builds still send `options()` where their framework can, and turn thinking off on every call, since a
Modelfile cannot.

The model name is written the plain Ollama way; each build adds its framework's prefix itself.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass

DEFAULT_MODEL = "qwen3-agents"
BASE_MODEL = "qwen3:8b"
# Smaller stand-ins for computers that cannot run the 8B (DECISIONS.md 20). Each Modelfile builds the
# same name, qwen3-agents, with the same settings; results are only comparable on BASE_MODEL.
SMALLER_MODELFILES = {"Modelfile.4b": "qwen3:4b-q4_K_M", "Modelfile.1.7b": "qwen3:1.7b"}
DEFAULT_HOST = "http://localhost:11434"
CREATE_HINT = "ollama create qwen3-agents -f contracts/ollama/Modelfile"


@dataclass(frozen=True)
class ModelSettings:
    model: str = DEFAULT_MODEL
    host: str = DEFAULT_HOST
    temperature: float = 0.0
    seed: int = 42
    num_ctx: int = 16384  # Ollama's own default on the reference Mac is 4096: too small for agents.
    think: bool = False

    def options(self) -> dict:
        """Ollama `options` for builds whose framework can send them per request."""
        return {"temperature": self.temperature, "seed": self.seed, "num_ctx": self.num_ctx}


def load_settings() -> ModelSettings:
    """Read `MODEL` and `OLLAMA_HOST` from the environment; everything else is fixed by the spec."""
    return ModelSettings(
        model=os.environ.get("MODEL", DEFAULT_MODEL),
        host=os.environ.get("OLLAMA_HOST", DEFAULT_HOST),
    )


def _model_parameters(show_response: dict) -> dict[str, str]:
    """Parse the `parameters` text of Ollama's /api/show into {name: value} (last value wins)."""
    params = {}
    for line in (show_response.get("parameters") or "").splitlines():
        name, _, value = line.strip().partition(" ")
        if name:
            params[name] = value.strip().strip('"')
    return params


def verify_model(settings: ModelSettings | None = None, timeout: float = 10.0) -> None:
    """Fail fast, before a run, if Ollama is down or the model lacks the shared settings.

    Raises RuntimeError with the fix. Call it at the top of every `run_eval.py`.
    """
    s = settings or load_settings()
    req = urllib.request.Request(f"{s.host.rstrip('/')}/api/show", data=json.dumps({"model": s.model}).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            shown = json.loads(r.read())
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama has no model {s.model!r}. Create it: {CREATE_HINT}") from exc
    except OSError as exc:
        raise RuntimeError(f"Ollama is not reachable at {s.host}. Start it: ollama serve") from exc
    params = _model_parameters(shown)
    want = {"num_ctx": str(s.num_ctx), "temperature": "0", "seed": str(s.seed)}
    wrong = {k: params.get(k) for k, v in want.items() if params.get(k) != v}
    if wrong:
        raise RuntimeError(f"Model {s.model!r} does not carry the shared settings {want} (has {wrong}). "
                           f"Recreate it: {CREATE_HINT}")
