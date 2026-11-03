"""The shared Resolution rules and eval runners, driven by a scripted fake agent (no model needed)."""

import asyncio
import json

from agent_contracts.harness.runners import Turn, run_question, run_research_eval, run_shipfast_eval
from agent_contracts.research.schema import Critique, Report
from agent_contracts.research.scoring import load_questions
from agent_contracts.shipfast import tools as st
from agent_contracts.shipfast.resolution import TurnMarker, derive_resolution
from agent_contracts.shipfast.schema import Outcome, Triage

TRIAGE = {  # what a correct triage returns for each opening message, by order id in the text
    "A1001": Triage(intent="order_status", order_id="A1001"), "A1002": Triage(intent="refund_request", order_id="A1002"),
    "A1003": Triage(intent="refund_request", order_id="A1003"), "A1004": Triage(intent="complaint", order_id="A1004"),
    "A1005": Triage(intent="refund_request", order_id="A1005"), "A9999": Triage(intent="order_status", order_id="A9999"),
}


class ScriptedShipFast:
    """Behaves the way a correct agent would, using the real shared tools, so only the runner is tested."""

    async def start(self):
        return {"triage": None, "marker": None}

    async def send(self, conv, text, metrics):
        metrics.add_llm_call(10, 2)
        oid = next((k for k in TRIAGE if k in text), None)
        conv["triage"] = TRIAGE[oid] if oid else (conv["triage"] or Triage(intent="other", order_id=None))
        conv["marker"] = TurnMarker.now()
        t = conv["triage"]
        if t.intent == "other":
            return Turn(derive_resolution(t, conv["marker"], "Sorry, out of scope."))
        if t.order_id == "A1002":
            st.issue_refund("A1002", 45, "lost")
        elif t.order_id == "A1003":
            return Turn(derive_resolution(t, conv["marker"], "Waiting for a supervisor.", paused=True), paused=True)
        elif t.order_id == "A1004" and "photo" in text:
            st.issue_refund("A1004", 60, "broken, photo provided")
        outcome = {"A1004": "needs_info", "A9999": "needs_info", "A1005": "declined"}.get(t.order_id, "answered")
        return Turn(derive_resolution(t, conv["marker"], "reply", outcome=Outcome(action=outcome)))

    async def resume(self, conv, decision, metrics):
        t = conv["triage"]
        if decision == "approve":
            st.issue_refund(t.order_id, 250, "late")
        else:
            st.escalate_to_human(t.order_id, "refund rejected")
        return Turn(derive_resolution(t, conv["marker"], "done"))


def test_derive_resolution_rules():
    st.reset()
    t = Triage(intent="refund_request", order_id="A1002")
    m = TurnMarker.now()
    assert derive_resolution(t, m, "r", outcome=Outcome(action="declined")).action == "declined"
    assert derive_resolution(t, m, "r", paused=True).action == "pending_approval"
    st.issue_refund("A1002", 45, "lost")
    r = derive_resolution(t, m, "r")
    assert (r.action, r.refund_amount) == ("refunded", 45.0)
    later = TurnMarker.now()  # an earlier turn's refund must not count again
    assert derive_resolution(t, later, "r", outcome=Outcome(action="answered")).action == "answered"
    st.escalate_to_human("A1002", "x")
    e = derive_resolution(t, later, "r")
    assert (e.action, e.ticket_id) == ("escalated", "TCK-A1002")
    assert derive_resolution(Triage(intent="other", order_id=None), later, "r").action == "declined"
    st.reset()


def test_shipfast_runner_scores_every_scenario(tmp_path):
    out = asyncio.run(run_shipfast_eval(ScriptedShipFast(), "langgraph", "test", "fake", results_dir=tmp_path,
                                        log=lambda line: None))
    items = {i["id"]: i for i in out["items"]}
    assert all(i["passed"] for i in items.values()), {k: v["problems"] for k, v in items.items() if not v["passed"]}
    assert items["S3a_late_refund_approved"]["resolution"]["refund_amount"] == 250.0
    assert items["S3b_late_refund_rejected"]["resolution"]["ticket_id"] == "TCK-A1003"
    assert [t["action"] for t in items["S4_damaged_needs_photo_then_refund"]["turn_resolutions"]] == ["needs_info", "refunded"]
    data = json.loads(out["path"].read_text())
    assert data["summary"]["passed"] == 8 and data["summary"]["total"] == 8
    assert data["scenarios"][0]["llm_calls"] >= 1


def test_shipfast_runner_reports_failures():
    class Wrong(ScriptedShipFast):
        async def resume(self, conv, decision, metrics):
            st.issue_refund(conv["triage"].order_id, 250, "late")  # refunds even when rejected
            return Turn(derive_resolution(conv["triage"], conv["marker"], "done"))
    out = asyncio.run(run_shipfast_eval(Wrong(), "langgraph", "t", "f", scenario_ids=["S3b_late_refund_rejected"],
                                        write=False, log=lambda line: None))
    item = out["items"][0]
    assert not item["passed"] and any("action" in p for p in item["problems"])


def test_research_runner():
    q = {x["id"]: x for x in load_questions()}["Q01_headcount"]

    async def answer(qid, question, metrics):
        metrics.add_llm_call(5, 5)
        return Report(question_id=qid, plan=["search"], answer="Veltara has 350 employees [F02].",
                      citations=["F02"], critiques=[Critique(verdict="pass")], revisions=0)

    async def broken(qid, question, metrics):
        raise RuntimeError("model unavailable")

    assert asyncio.run(run_question(answer, q))["passed"]
    failed = asyncio.run(run_question(broken, q))
    assert not failed["passed"] and failed["error"].startswith("RuntimeError")
    out = asyncio.run(run_research_eval(answer, "strands", "t", "f", question_ids=["Q01_headcount"], write=False,
                                        log=lambda line: None))
    assert out["items"][0]["passed"] and out["path"] is None
