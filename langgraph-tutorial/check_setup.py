"""Check that this tutorial is ready to run: `uv run python check_setup.py` from this folder.

Each check prints ok or FAIL with the fix. It ends with one short call to the model, so the first run can
take a few seconds while Ollama loads it. Exits with status 1 if anything failed.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from importlib.metadata import PackageNotFoundError, version

TUTORIAL = "LangGraph"
PACKAGES = ("langgraph", "langchain", "langchain-ollama")  # the framework, at the versions in uv.lock
MODULE = "lg_agents"  # this tutorial's agents
SYNC = "run `uv sync` in this folder, and make sure the notebook kernel is this folder's .venv"

failures = 0


def check(name: str, ok: bool, detail: str = "", fix: str = "") -> bool:
    global failures
    print(f"{'ok  ' if ok else 'FAIL'}  {name}{': ' + detail if detail else ''}")
    if not ok:
        failures += 1
        print(f"      fix: {fix}")
    return ok


def ollama(path: str, body: dict | None = None, timeout: float = 10.0) -> dict:
    req = urllib.request.Request(f"{host}{path}", data=json.dumps(body).encode() if body else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


print(f"Checking the {TUTORIAL} tutorial\n")

check("python", sys.version_info[:2] in ((3, 12), (3, 13)), sys.version.split()[0],
      "use Python 3.12 or 3.13; `uv sync` picks one for you")

for package in PACKAGES:
    try:
        check(package, True, version(package))
    except PackageNotFoundError:
        check(package, False, "not installed", SYNC)

try:
    from agent_contracts.harness import load_settings, verify_model
    check("agent_contracts", True, "the shared contract is installed")
except ImportError as exc:
    check("agent_contracts", False, str(exc), SYNC)
    sys.exit(1)

try:
    __import__(MODULE)
    check(MODULE, True, "this tutorial's agents import")
except Exception as exc:  # an import error inside the framework counts too
    check(MODULE, False, f"{type(exc).__name__}: {exc}", SYNC)

settings = load_settings()
host = settings.host.rstrip("/")
try:
    check("ollama", True, f"version {ollama('/api/version')['version']} at {host}")
except OSError:
    check("ollama", False, f"not reachable at {host}", "start it: `ollama serve` or `brew services start ollama`")
    print("\nThe model checks need Ollama; stopping here.")
    sys.exit(1)

try:
    verify_model(settings)
    check("model", True, f"{settings.model} with the shared settings")
    model_ok = True
except RuntimeError as exc:
    model_ok = check("model", False, str(exc).split(". ")[0], str(exc).split(": ", 1)[-1].replace(" contracts/", " ../contracts/"))

if model_ok:
    try:
        reply = ollama("/api/chat", {"model": settings.model, "stream": False, "think": False, "options": settings.options(),
                                     "messages": [{"role": "user", "content": "Reply with the single word: ready"}]},
                       timeout=180)["message"]["content"].strip()
        check("model call", bool(reply), repr(reply[:40]))
    except OSError as exc:
        check("model call", False, str(exc), "close other memory-hungry apps and try again; the model needs about 6 GB")

print("\nAll set: open notebooks/00_introduction.ipynb." if not failures else f"\n{failures} check(s) failed.")
sys.exit(1 if failures else 0)
