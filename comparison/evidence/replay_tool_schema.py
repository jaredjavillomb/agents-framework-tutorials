"""Replay the request that decides S3a, with and without per-parameter descriptions (COMPARISON.md).

The two JSON files are the exact requests LangGraph and Strands sent to Ollama for the specialist's fourth
model call in scenario S3a, captured with a recording relay on 2026-11-03: same system prompt, options,
messages, tool calls and tool results; the only difference is that Strands adds a placeholder
description ("Parameter order_id") to every tool parameter. Needs Ollama running with qwen3-agents.

    python3 comparison/evidence/replay_tool_schema.py
"""

import copy
import json
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent


def send(body: dict) -> str:
    body = dict(body, stream=False)
    req = urllib.request.Request("http://127.0.0.1:11434/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    message = json.loads(urllib.request.urlopen(req, timeout=600).read())["message"]
    calls = [c["function"]["name"] for c in message.get("tool_calls") or []]
    return f"calls {calls}" if calls else "text: " + (message.get("content") or "")[:70]


def with_descriptions(body: dict, on: bool) -> dict:
    body = copy.deepcopy(body)
    for t in body["tools"]:
        for name, p in t["function"]["parameters"]["properties"].items():
            if on:
                p["description"] = f"Parameter {name}"
            else:
                p.pop("description", None)
    return body


strands = json.loads((HERE / "s3a_decision_request_strands.json").read_text())
langgraph = json.loads((HERE / "s3a_decision_request_langgraph.json").read_text())
print("Strands request as sent               ->", send(strands))
print("Strands request, descriptions removed ->", send(with_descriptions(strands, False)))
print("LangGraph request as sent             ->", send(langgraph))
print("LangGraph request, descriptions added ->", send(with_descriptions(langgraph, True)))
