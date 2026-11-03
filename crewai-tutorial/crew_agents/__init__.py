"""CrewAI build of the shared agents.

This project runs locally (DECISIONS.md 4), so CrewAI's anonymous usage telemetry and its cloud tracing
prompt are switched off before CrewAI is imported. These are defaults: set the variables in `.env` or the
shell to override them.

CrewAI also prints boxed panels for every tool run. They follow a crew's `verbose` flag, but a standalone
`Agent.kickoff` always prints them, so they are off by default here; `show_panels(True)` turns them on.
"""

import os

os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_DISABLE_TRACKING", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")

from crewai.events.event_listener import event_listener  # noqa: E402  (must follow the settings above)


def show_panels(on: bool = True) -> None:
    """Turn CrewAI's console panels on or off for standalone agent runs."""
    event_listener.formatter.verbose = on


show_panels(False)
