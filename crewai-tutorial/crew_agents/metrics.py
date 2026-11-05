"""Feed CrewAI's event bus into the shared harness (agent_contracts.harness.RunMetrics).

CrewAI publishes an `LLMCallCompletedEvent` with token usage after every model call, from agents, crews,
flows and direct `LLM.call` alike. This listener adds each one to the `RunMetrics` currently being
measured. CrewAI delivers events asynchronously, so `measuring` flushes the bus before it stops
counting. Tool calls are not counted here: the shared tools record them themselves, so all three
frameworks count them the same way.
"""

from __future__ import annotations

from contextlib import contextmanager

from crewai.events import BaseEventListener, LLMCallCompletedEvent, crewai_event_bus

from agent_contracts.harness import RunMetrics


def _usage(usage) -> tuple[int, int]:
    if usage is None:
        return 0, 0
    get = usage.get if isinstance(usage, dict) else (lambda k: getattr(usage, k, None))
    return int(get("prompt_tokens") or 0), int(get("completion_tokens") or 0)


class HarnessListener(BaseEventListener):
    def __init__(self) -> None:
        self.metrics: RunMetrics | None = None  # None: not measuring right now, so events are ignored
        super().__init__()  # BaseEventListener's __init__ calls setup_listeners below

    def setup_listeners(self, crewai_event_bus) -> None:
        # Subscribe to one event type: CrewAI publishes LLMCallCompletedEvent after every model call.
        @crewai_event_bus.on(LLMCallCompletedEvent)
        def on_llm_call(source, event: LLMCallCompletedEvent) -> None:
            if self.metrics is not None:
                self.metrics.add_llm_call(*_usage(event.usage))  # (input tokens, output tokens)

    # Used as `with listener.measuring(metrics):`. Only calls made inside the block are counted.
    @contextmanager
    def measuring(self, metrics: RunMetrics):
        """Count every model call made inside the `with` block into `metrics`."""
        self.metrics = metrics
        try:
            yield metrics  # the body of the `with` block runs here
        finally:
            crewai_event_bus.flush()  # events are delivered asynchronously; wait for the last ones
            self.metrics = None


harness_listener = HarnessListener()
