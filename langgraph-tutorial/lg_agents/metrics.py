"""Feed LangChain's callback events into the shared harness (agent_contracts.harness.RunMetrics).

Pass `HarnessCallback(metrics)` in the run's config (`config={"callbacks": [cb]}`); LangChain hands it to
every nested model call, including the agent inside a workflow node. Tool calls are not counted here: the
shared tools record them themselves, so all three frameworks count them the same way.
"""

from __future__ import annotations

from langchain_core.callbacks import BaseCallbackHandler

from agent_contracts.harness import RunMetrics


class HarnessCallback(BaseCallbackHandler):
    def __init__(self, metrics: RunMetrics) -> None:
        self.metrics = metrics  # the shared counters for this run

    # LangChain calls on_llm_end after every model call it makes, however deep in the run.
    def on_llm_end(self, response, **kwargs) -> None:
        # response.generations is a list (one entry per prompt) of lists (one entry per reply). An agent
        # sends one prompt and gets one reply, so this records one model call.
        for generations in response.generations:
            for generation in generations:
                # The reply message carries the token counts Ollama reported; {} if a provider sends none.
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                self.metrics.add_llm_call(usage.get("input_tokens"), usage.get("output_tokens"))
