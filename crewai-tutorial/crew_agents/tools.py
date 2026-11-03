"""CrewAI tool wrappers around the shared tools in agent_contracts.

The tool logic lives in `agent_contracts` (CLAUDE.md: never in a framework). This module only wraps the
plain functions in CrewAI's tool type; the function name becomes the tool name and its docstring the
description the model sees.
"""

from __future__ import annotations

from crewai.tools import BaseTool, tool

from agent_contracts.research import tools as research
from agent_contracts.shipfast import tools as shipfast


def _wrap(functions) -> list[BaseTool]:
    return [tool(f.__name__)(f) for f in functions]


def shipfast_tools() -> list[BaseTool]:
    return _wrap(shipfast.TOOLS)


def research_tools() -> list[BaseTool]:
    return _wrap(research.TOOLS)


def function_schemas(tools: list[BaseTool]) -> list[dict]:
    """Tool descriptions in the OpenAI function format that `LLM.call(tools=...)` expects."""
    return [{"type": "function", "function": {"name": t.name, "description": t.description,
                                              "parameters": t.args_schema.model_json_schema()}} for t in tools]
