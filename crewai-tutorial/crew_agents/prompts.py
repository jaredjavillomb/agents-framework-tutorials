"""ShipFast specialist instructions for CrewAI, matching lg_agents/prompts.py in the LangGraph build.

CrewAI describes an agent with a role, a goal and a backstory rather than one system prompt. The
working steps go in the backstory: the shared SPECIALIST text, so all three frameworks give the same words.
"""

from agent_contracts.shipfast.prompts import SPECIALIST, TRIAGE  # shared by all builds (DECISIONS.md 18)

ROLE = "ShipFast support specialist"
GOAL = "Resolve each customer message correctly under ShipFast's policy, using the tools to check every fact"
BACKSTORY = SPECIALIST
