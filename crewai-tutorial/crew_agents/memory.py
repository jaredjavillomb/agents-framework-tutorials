"""CrewAI memory, kept local.

CrewAI's unified `Memory` defaults its own LLM to an OpenAI model (`gpt-5.4-mini`) and needs an embedding
model for semantic search. Both are set explicitly here: the shared local model, and `nomic-embed-text`
served by Ollama. Storage is a LanceDB folder inside the tutorial (git ignores it), not a hidden system
folder, so it is easy to inspect and delete.
"""

from __future__ import annotations

from pathlib import Path

from crewai.memory.unified_memory import Memory

from agent_contracts.harness import load_settings
from crew_agents.llm import make_llm

EMBED_MODEL = "nomic-embed-text"
DEFAULT_PATH = Path(__file__).resolve().parents[1] / ".crewai_memory"


def embedder() -> dict:
    host = load_settings().host.rstrip("/")
    return {"provider": "ollama", "config": {"model_name": EMBED_MODEL, "url": f"{host}/api/embeddings"}}


def make_memory(path: str | Path | None = None, read_only: bool = False) -> Memory:
    """A local CrewAI Memory. `read_only=True` recalls without saving new memories after each task, which
    on this machine saves roughly a minute or more per task (each save is analysed by the model first)."""
    return Memory(
        llm=make_llm(),  # the local model analyses each fact before it is saved (CrewAI's default is OpenAI's)
        embedder=embedder(),  # nomic-embed-text in Ollama turns text into vectors for search by meaning
        storage=str(path or DEFAULT_PATH),  # a LanceDB folder: the vector database the facts live in
        read_only=read_only,  # True: recall only, never save
    )
