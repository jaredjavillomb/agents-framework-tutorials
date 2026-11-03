"""LangChain tool wrappers around the shared tools in agent_contracts.

The tool logic lives in `agent_contracts` (CLAUDE.md: never in a framework). This module only wraps the
plain functions in LangChain's tool type; the function name becomes the tool name and its docstring the
description the model sees.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from agent_contracts.research import tools as research
from agent_contracts.shipfast import tools as shipfast


def shipfast_tools() -> list[BaseTool]:
    return [tool(f) for f in shipfast.TOOLS]


def research_tools() -> list[BaseTool]:
    return [tool(f) for f in research.TOOLS]
