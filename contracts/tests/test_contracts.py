"""Deterministic tests for agent_contracts. No model and no network needed."""

import json

import pytest

from agent_contracts.harness import (RunMetrics, ToolCallLog, build_results, load_settings, verify_model,
                                     write_results)
from agent_contracts.research import tools as rt
from agent_contracts.research.schema import Critique, Report
from agent_contracts.research.scoring import load_questions, matches, normalize, score_report
from agent_contracts.shipfast import tools as st
from agent_contracts.shipfast.schema import Resolution, Triage
from agent_contracts.shipfast.scoring import check_ledger, check_resolution, load_scenarios


@pytest.fixture(autouse=True)
def clean_state():
    st.reset()
    rt.reset()
    yield
    st.reset()
    rt.reset()


# ---------------------------------------------------------------- ShipFast tools

def test_lookup_order_hides_tracking_and_handles_unknown():
    order = json.loads(st.lookup_order("A1001"))
    assert order["order_id"] == "A1001" and "tracking" not in order
    assert json.loads(st.lookup_order("a1001"))["order_id"] == "A1001"
    assert json.loads(st.lookup_order("A9999")) == {"error": "order_not_found", "order_id": "A9999"}


def test_track_shipment_shape():
    t = json.loads(st.track_shipment("A1002"))
    assert set(t) == {"order_id", "status", "carrier", "eta", "delivered_on", "tracking"}
    assert t["status"] == "lost"


def test_search_policy_matches_and_falls_back_to_all():
    hits = json.loads(st.search_policy("my package was damaged"))
    assert hits["today"] == "2026-11-01"
    assert "P3" in {s["id"] for s in hits["sections"]}
    everything = json.loads(st.search_policy("xyz"))
    assert [s["id"] for s in everything["sections"]] == ["P1", "P2", "P3", "P4", "P5", "P6", "P7"]


def test_issue_refund_ledger_and_limits():
    assert json.loads(st.issue_refund("A1002", 45, "lost")) == {
        "refund_id": "RF-A1002", "order_id": "A1002", "amount": 45.0, "status": "issued"}
    assert json.loads(st.issue_refund("A1002", 120, "lost"))["error"] == "amount_exceeds_order_total"
    assert json.loads(st.issue_refund("A1002", 0, "lost"))["error"] == "invalid_amount"
    assert json.loads(st.issue_refund("A9999", 10, "x"))["error"] == "order_not_found"
    assert [(r["order_id"], r["amount"]) for r in st.refunds()] == [("A1002", 45.0)]


def test_escalate_and_tool_log_and_reset():
    assert json.loads(st.escalate_to_human("", "poem request"))["ticket_id"] == "TCK-GENERAL"
    assert json.loads(st.escalate_to_human("A1003", "rejected"))["ticket_id"] == "TCK-A1003"
    st.lookup_order("A1001")
    assert st.tool_calls() == ["escalate_to_human", "escalate_to_human", "lookup_order"]
    st.reset()
    assert st.tool_calls() == [] and st.tickets() == [] and st.refunds() == []


def test_shipfast_scoring():
    s2 = next(s for s in load_scenarios() if s["id"] == "S2_lost_refund_auto")
    good = Resolution(intent="refund_request", order_id="A1002", action="refunded", refund_amount=45.0,
                      reply="Refunded.")
    assert check_resolution(good, s2["expected"]) == []
    assert check_resolution(good.model_copy(update={"refund_amount": 45.001}), s2["expected"]) == []
    assert check_resolution(good.model_copy(update={"refund_amount": 120.0}), s2["expected"])
    assert check_ledger(good, []) == ["action is 'refunded' but the ledger has no matching refund"]
    st.issue_refund("A1002", 45, "lost")
    assert check_ledger(good, st.refunds()) == []
    assert Triage(intent="other", order_id=None).order_id is None


# ---------------------------------------------------------------- research tools

def test_every_question_is_reachable_by_a_plain_search():
    queries = {"Q01_headcount": "Veltara employees", "Q02_founding": "Veltara founders",
               "Q03_revenue_growth": "Veltara revenue FY2024", "Q04_series_b": "Veltara Series B",
               "Q05_capacity": "Veltara production capacity per year", "Q06_daily_energy": "TidePure 40 energy",
               "Q07_deployment": "Veltara units deployed provinces", "Q08_recall": "TidePure Mini recall",
               "Q09_mindanao": "Veltara Mindanao", "Q10_order_cost": "TidePure price list 2026"}
    want = {"Q01_headcount": "F02", "Q02_founding": "F01", "Q03_revenue_growth": "F05", "Q04_series_b": "F06",
            "Q05_capacity": "F08", "Q06_daily_energy": "F10", "Q07_deployment": "F04", "Q08_recall": "F12",
            "Q09_mindanao": "F13", "Q10_order_cost": "F28"}
    assert set(queries) == {q["id"] for q in load_questions()}
    for qid, query in queries.items():
        assert want[qid] in [d["id"] for d in rt.search(query)], qid


def test_stale_article_surfaces_beside_the_current_one():
    ids = [d["id"] for d in rt.search("How many employees does Veltara Aquatech have?")]
    assert ids[:2] == ["F02", "F03"]


def test_search_matching_rules():
    assert rt.tokens("Employees, 1,200 units. The plant") == {"employee", "1200", "unit", "plant"}
    assert rt.search("zzz qqq") == []
    assert json.loads(rt.web_search("zzz"))["results"] == []
    assert len(rt.search("Veltara")) == 3


def test_calculator():
    assert json.loads(rt.calculator("(1.56 - 1.20) / 1.20 * 100"))["result"] == pytest.approx(30.0)
    assert json.loads(rt.calculator("3 * 385,000 + 10 * 48,000"))["result"] == 1635000
    for bad in ("__import__('os')", "1/0", "abs(-1)", "2 ** 1000", "x + 1", ""):
        rt.reset()  # each case on a fresh budget
        assert json.loads(rt.calculator(bad))["error"] == "invalid_expression", bad


def test_budget_shared_across_tools_and_refusal_does_not_raise():
    for _ in range(3):
        rt.web_search("Veltara")
    rt.calculator("1+1")
    rt.calculator("2+2")
    assert not rt.budget_exhausted()
    assert json.loads(rt.web_search("Veltara")) == {"error": "budget_exhausted", "limit": 5}
    assert json.loads(rt.calculator("1+1")) == {"error": "budget_exhausted", "limit": 5}
    assert rt.budget_exhausted() and len(rt.tool_calls()) == 7
    rt.reset()
    assert rt.tool_calls() == [] and not rt.budget_exhausted()
    assert "results" in json.loads(rt.web_search("Veltara"))


# ---------------------------------------------------------------- research scoring

def _report(answer, citations=("F02",), verdicts=("pass",), revisions=0, qid="Q01_headcount"):
    return Report(question_id=qid, plan=["search"], answer=answer, citations=list(citations),
                  critiques=[Critique(verdict=v) for v in verdicts], revisions=revisions)


def _q(qid):
    return next(q for q in load_questions() if q["id"] == qid)


def test_scoring_pass_and_stale_fail():
    assert score_report(_report("Veltara has 350 employees [F02]."), _q("Q01_headcount"))["passed"]
    stale = score_report(_report("350 employees, up from 200 in 2023."), _q("Q01_headcount"))
    assert not stale["passed"] and not stale["scores"]["contradiction_free"]


def test_scoring_whole_token_matching_and_normalization():
    assert matches("14", normalize("in 14 provinces")) and not matches("14", normalize("founded 2014"))
    assert not matches("30%", normalize("grew 130%"))
    q10 = _report("Total PHP 1,635,000: 3 x 385,000 plus 10 x 48,000.", ["F28"], qid="Q10_order_cost")
    assert score_report(q10, _q("Q10_order_cost"))["passed"]


def test_scoring_citations_and_loop_rule():
    q = _q("Q01_headcount")
    assert not score_report(_report("350 employees", citations=()), q)["scores"]["citations_valid"]
    assert not score_report(_report("350 employees", citations=("F99",)), q)["passed"]
    assert score_report(_report("350", verdicts=("revise", "pass"), revisions=1), q)["passed"]
    assert score_report(_report("350", verdicts=("revise", "revise"), revisions=2), q)["passed"]
    bad = score_report(_report("350", verdicts=("revise",), revisions=0), q)
    assert not bad["passed"] and "invalid critique loop" in bad["problems"][0]


def test_every_question_passes_with_its_own_aliases():
    for q in load_questions():
        answer = "; ".join(f["aliases"][0] for f in q["expected_facts"])
        assert score_report(_report(answer, qid=q["id"]), q)["passed"], q["id"]


# ---------------------------------------------------------------- harness

def test_tool_call_log_limit():
    log = ToolCallLog(limit=1)
    assert log.attempt("a") and not log.attempt("b")
    assert log.names() == ["a", "b"] and log.budget_exhausted()


def test_metrics_and_results(tmp_path, monkeypatch):
    m = RunMetrics()
    with m.measure():
        m.add_llm_call(100, 10)
        m.add_llm_call(None, 5)
    assert (m.llm_calls, m.input_tokens, m.output_tokens) == (2, 100, 15) and m.latency_s >= 0
    items = [{"id": "Q01", "passed": True, **m.as_dict()}, {"id": "Q02", "passed": False, **m.as_dict()}]
    path = write_results("research", "langgraph", "1.0", "qwen3:8b", items, results_dir=tmp_path)
    data = json.loads(path.read_text())
    assert path == tmp_path / "research" / "langgraph.json"
    assert data["summary"]["passed"] == 1 and data["summary"]["total"] == 2
    assert data["summary"]["total_tokens"] == 230
    assert "questions" in data and "scenarios" in build_results("shipfast", "crewai", "1", "m", [])
    with pytest.raises(ValueError):
        build_results("other", "crewai", "1", "m", [])


def test_settings(monkeypatch):
    monkeypatch.delenv("MODEL", raising=False)
    s = load_settings()
    assert s.model == "qwen3-agents" and s.think is False
    assert s.options() == {"temperature": 0.0, "seed": 42, "num_ctx": 16384}
    monkeypatch.setenv("MODEL", "qwen3:14b")
    assert load_settings().model == "qwen3:14b"


def test_modelfile_matches_settings():
    from pathlib import Path
    from agent_contracts.harness.settings import BASE_MODEL
    text = (Path(__file__).parents[1] / "ollama" / "Modelfile").read_text()
    s = load_settings()
    assert f"FROM {BASE_MODEL}" in text
    assert f"PARAMETER num_ctx {s.num_ctx}" in text and f"PARAMETER seed {s.seed}" in text
    assert "PARAMETER temperature 0" in text


def test_verify_model_parses_parameters_and_reports_unreachable():
    from agent_contracts.harness.settings import ModelSettings, _model_parameters
    params = _model_parameters({"parameters": 'num_ctx                        16384\nseed 42\nstop "<|im_end|>"'})
    assert params["num_ctx"] == "16384" and params["seed"] == "42" and params["stop"] == "<|im_end|>"
    with pytest.raises(RuntimeError, match="not reachable"):
        verify_model(ModelSettings(host="http://127.0.0.1:9"), timeout=1)
