"""Research tools (SPEC.md section 3): `web_search` over fixed articles and `calculator`, as JSON strings.

Both share one tool-call budget of 5 per question, enforced here so it means the same thing in every
framework. Call 6 onward returns `budget_exhausted` and does not raise. The runner calls `reset()`
before every question.
"""

from __future__ import annotations

import ast
import json
import operator
import re
from pathlib import Path

from agent_contracts.harness import ToolCallLog

TOOL_CALL_LIMIT = 5
FIXTURES: list[dict] = json.loads((Path(__file__).parent / "data" / "search_fixtures.json").read_text())
FIXTURE_IDS = {d["id"] for d in FIXTURES}

STOPWORDS = frozenset("""
a an the and or of in on at to for from by with about as is are was were be been has have had do does
did how what when where which who whom why many much it its this that these those now i you we they me
my our your their there can will would should could
""".split())

log = ToolCallLog(limit=TOOL_CALL_LIMIT)


def _budget_error() -> str:
    return json.dumps({"error": "budget_exhausted", "limit": TOOL_CALL_LIMIT})


def tokens(text: str) -> set[str]:
    """The matching rule's normalization (SPEC.md section 3)."""
    text = re.sub(r"(?<=\d),(?=\d)", "", text.lower())
    out = set()
    for word in re.split(r"[^a-z0-9.]+", text):
        word = word.strip(".")
        if not word or word in STOPWORDS:
            continue
        if len(word) >= 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        out.add(word)
    return out


_INDEX = [(doc, tokens(doc["title"] + " " + doc["snippet"])) for doc in FIXTURES]


def search(query: str) -> list[dict]:
    """The top 3 matching articles, without touching the budget (for tests and notebooks)."""
    wanted = tokens(query)
    scored = [(len(wanted & toks), i, doc) for i, (doc, toks) in enumerate(_INDEX)]
    scored = sorted((s for s in scored if s[0] > 0), key=lambda s: (-s[0], s[1]))
    return [dict(doc) for _, _, doc in scored[:3]]


def web_search(query: str) -> str:
    """Search news and company articles. Returns up to 3 results, each with an id to cite."""
    if not log.attempt("web_search"):
        return _budget_error()
    return json.dumps({"query": query, "results": search(query)})


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.USub: operator.neg, ast.UAdd: operator.pos}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("not arithmetic")


def calculate(expression: str) -> float:
    """Evaluate plain arithmetic, without touching the budget. Raises ValueError otherwise."""
    try:
        result = _eval(ast.parse(re.sub(r"(?<=\d),(?=\d)", "", expression), mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError, TypeError) as exc:
        raise ValueError(str(exc)) from exc
    return float(result)


def calculator(expression: str) -> str:
    """Evaluate arithmetic: numbers, + - * / ** and parentheses. Example: (1.56 - 1.20) / 1.20 * 100"""
    if not log.attempt("calculator"):
        return _budget_error()
    try:
        result = calculate(expression)
    except ValueError:
        return json.dumps({"error": "invalid_expression", "expression": expression})
    return json.dumps({"expression": expression, "result": round(result, 10)})


TOOLS = [web_search, calculator]


def tool_calls() -> list[str]:
    """Every attempted call since the last reset, refused ones included."""
    return log.names()


def budget_exhausted() -> bool:
    return log.budget_exhausted()


def reset() -> None:
    """Reset the tool-call budget. Call before every question."""
    log.reset()
