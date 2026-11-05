"""Feed Strands' model calls into the shared harness (agent_contracts.harness.RunMetrics).

`HarnessHook` is a Strands hook provider; give the same hook to every agent in a run. Strands updates an
agent's token totals only after a model call's events have fired, so the hook settles at the end of each
agent invocation (`AfterInvocationEvent`): it adds the model calls (event-loop cycles) and the tokens used
since the last time it looked. The run's totals and call count are exact; the harness records no per-call
breakdown. Tool calls are not counted here: the shared tools record them themselves, so all three
frameworks count them the same way.
"""

from __future__ import annotations

from strands.hooks import AfterInvocationEvent, HookProvider, HookRegistry

from agent_contracts.harness import RunMetrics


class HarnessHook(HookProvider):
    def __init__(self, metrics: RunMetrics) -> None:
        self.metrics = metrics  # the shared counters for this run
        self._seen: dict[int, tuple[int, int, int]] = {}  # per agent: cycles, input, output already counted

    # Strands calls register_hooks once, when the hook is given to an Agent: subscribe to one event here.
    def register_hooks(self, registry: HookRegistry, **kwargs) -> None:
        registry.add_callback(AfterInvocationEvent, self.after_invocation)

    # Runs after every agent call (every agent("...")), once the agent's whole loop has finished.
    def after_invocation(self, event: AfterInvocationEvent) -> None:
        # Strands keeps running totals per agent: cycles (one model call each) and tokens, since it was made.
        m = event.agent.event_loop_metrics
        cycles = int(m.cycle_count)
        tokens_in, tokens_out = int(m.accumulated_usage.get("inputTokens", 0)), int(m.accumulated_usage.get("outputTokens", 0))
        # Subtract what was already counted for this agent, so a second call to it adds only its own share.
        seen_cycles, seen_in, seen_out = self._seen.get(id(event.agent), (0, 0, 0))
        self._seen[id(event.agent)] = (cycles, tokens_in, tokens_out)
        new_calls = cycles - seen_cycles
        if new_calls <= 0:
            return
        # The totals cannot be split per call, so all new tokens go on the first new call and the rest
        # are counted with zero tokens: the call count and the token total both come out right.
        self.metrics.add_llm_call(tokens_in - seen_in, tokens_out - seen_out)  # first call carries the tokens
        for _ in range(new_calls - 1):
            self.metrics.add_llm_call(0, 0)
