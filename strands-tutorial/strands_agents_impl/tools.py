"""Strands tool wrappers around the shared tools in agent_contracts.

The tool logic lives in `agent_contracts` (CLAUDE.md: never in a framework). This module only wraps the
plain functions with Strands' `tool`; the function name becomes the tool name and its docstring the
description the model sees.
"""

from __future__ import annotations

from strands import tool

from agent_contracts.research import tools as research
from agent_contracts.shipfast import tools as shipfast


def shipfast_tools() -> list:
    # shipfast.TOOLS is the five shared functions: lookup_order, track_shipment, search_policy,
    # issue_refund and escalate_to_human. tool(f) reads each one's name, type hints and docstring into a
    # tool spec; calling the tool still runs the shared function, unchanged.
    return [tool(f) for f in shipfast.TOOLS]


def research_tools() -> list:
    return [tool(f) for f in research.TOOLS]
