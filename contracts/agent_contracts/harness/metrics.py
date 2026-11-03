"""One definition of tokens, LLM calls and latency.

- Tokens: input and output tokens summed over every LLM call in one scenario or question. Each build
  feeds them in from its framework's callbacks or responses with `add_llm_call()`. With Ollama these
  are the response's `prompt_eval_count` and `eval_count`.
- LLM calls: the number of `add_llm_call()` calls.
- Latency: wall clock for the whole scenario or question, measured by `measure()`.

Usage in a runner:

    metrics = RunMetrics()
    with metrics.measure():
        run_the_agent(..., on_llm_call=metrics.add_llm_call)
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field


@dataclass
class RunMetrics:
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)

    def add_llm_call(self, input_tokens: int | None, output_tokens: int | None) -> None:
        with self._lock:
            self.llm_calls += 1
            self.input_tokens += int(input_tokens or 0)
            self.output_tokens += int(output_tokens or 0)

    @contextmanager
    def measure(self):
        start = time.perf_counter()
        try:
            yield self
        finally:
            self.latency_s = round(time.perf_counter() - start, 3)

    def as_dict(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_s": self.latency_s,
        }
