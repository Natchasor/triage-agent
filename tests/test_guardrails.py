from itertools import product

import pytest

from triage.config import Settings
from triage.data_loader import load_customers, load_kb_ids, load_tickets
from triage.guardrails import (
    DEFAULT_RULES,
    MIN_CONFIDENCE,
    GuardrailContext,
    apply_guardrails,
)
from triage.models import (
    Action,
    IssueType,
    Sentiment,
    SpecialistTeam,
    Ticket,
    TriageDecision,
    Urgency,
)

DATA_DIR = Settings().data_dir
TICKETS = {t.ticket_id: t for t in load_tickets(DATA_DIR)}
CUSTOMERS = load_customers(DATA_DIR)
KB_IDS = load_kb_ids(DATA_DIR)

DARK_MODE_TICKET = TICKETS["T-1003"]  # friendly, no dispute words
BILLING_TICKET = TICKETS["T-1001"]  # mentions "disputing" charges with the bank


def make_decision(**overrides) -> TriageDecision:
    """A harmless auto_respond decision. Tests change only what they care about."""
    fields = {
        "urgency": Urgency.LOW,
        "product": "Appearance",
        "issue_type": IssueType.HOW_TO,
        "sentiment": Sentiment.NEUTRAL,
        "language": "en",
        "summary": "Customer asks about dark mode.",
        "action": Action.AUTO_RESPOND,
        "specialist_team": None,
        "reasoning": "KB answers it.",
        "confidence": 0.9,
        "kb_article_ids": ["kb_006"],
        "draft_reply": "Hi!",
    }
    return TriageDecision(**{**fields, **overrides})


def make_context(ticket: Ticket = DARK_MODE_TICKET) -> GuardrailContext:
    return GuardrailContext(ticket, CUSTOMERS.get(ticket.customer_id), KB_IDS)


def run(decision: TriageDecision, ticket: Ticket = DARK_MODE_TICKET):
    return apply_guardrails(decision, make_context(ticket))


# ---------- no interference ----------


def test_a_clean_auto_respond_is_left_alone():
    decision = make_decision()
    final, log = run(decision)

    assert final == decision
    assert log == []


def test_original_decision_is_never_mutated():
    decision = make_decision(urgency=Urgency.CRITICAL)
    run(decision)

    assert decision.action is Action.AUTO_RESPOND


# ---------- individual rules ----------


def test_critical_ticket_is_escalated_to_a_human():
    final, log = run(make_decision(urgency=Urgency.CRITICAL, issue_type=IssueType.OUTAGE))

    assert final.action is Action.ESCALATE_HUMAN
    assert final.specialist_team is SpecialistTeam.INFRASTRUCTURE
    assert log[0].startswith("critical_urgency")


def test_dispute_threat_overrides_route_specialist():
    decision = make_decision(
        action=Action.ROUTE_SPECIALIST,
        specialist_team=SpecialistTeam.BILLING,
        issue_type=IssueType.BILLING,
    )
    final, log = run(decision, BILLING_TICKET)

    assert final.action is Action.ESCALATE_HUMAN
    assert final.specialist_team is SpecialistTeam.BILLING
    assert log[0].startswith("dispute_threat")


def test_dispute_words_in_thai_are_detected():
    ticket = TICKETS["T-1002"].model_copy(deep=True)
    ticket.messages[-1].text = "ถ้าไม่แก้ไข เราจะฟ้องบริษัท"
    final, log = run(make_decision(), ticket)

    assert final.action is Action.ESCALATE_HUMAN
    assert any(entry.startswith("dispute_threat") for entry in log)


def test_ticket_without_dispute_words_is_not_flagged():
    _, log = run(make_decision(), DARK_MODE_TICKET)

    assert log == []


def test_enterprise_with_high_urgency_needs_a_human():
    decision = make_decision(
        urgency=Urgency.HIGH, action=Action.ROUTE_SPECIALIST, specialist_team=SpecialistTeam.PRODUCT
    )
    final, log = run(decision, TICKETS["T-1002"])

    assert final.action is Action.ESCALATE_HUMAN
    assert log[0].startswith("enterprise_sla")


def test_high_urgency_on_a_non_enterprise_plan_is_not_forced():
    decision = make_decision(
        urgency=Urgency.HIGH, action=Action.ROUTE_SPECIALIST, specialist_team=SpecialistTeam.BILLING
    )
    final, log = run(decision, DARK_MODE_TICKET)  # Pro plan

    assert final.action is Action.ROUTE_SPECIALIST
    assert log == []


def test_angry_customer_is_not_auto_answered():
    final, log = run(make_decision(sentiment=Sentiment.ANGRY))

    assert final.action is Action.ESCALATE_HUMAN
    assert log[0].startswith("angry_customer")


@pytest.mark.parametrize(
    ("confidence", "escalated"), [(0.3, True), (MIN_CONFIDENCE - 0.01, True), (MIN_CONFIDENCE, False)]
)
def test_low_confidence_threshold(confidence, escalated):
    final, _ = run(make_decision(confidence=confidence))

    assert (final.action is Action.ESCALATE_HUMAN) is escalated


@pytest.mark.parametrize("cited", [[], ["kb_999"]])
def test_auto_respond_needs_a_real_knowledge_base_article(cited):
    final, log = run(make_decision(kb_article_ids=cited, issue_type=IssueType.BUG))

    assert final.action is Action.ROUTE_SPECIALIST
    assert final.specialist_team is SpecialistTeam.TECHNICAL_SUPPORT
    assert log[0].startswith("ungrounded_answer")


def test_one_real_article_among_invented_ones_is_enough():
    final, _ = run(make_decision(kb_article_ids=["kb_999", "kb_006"]))

    assert final.action is Action.AUTO_RESPOND


def test_missing_team_is_filled_in_for_human_handling():
    decision = make_decision(action=Action.ROUTE_SPECIALIST, issue_type=IssueType.BILLING)
    final, log = run(decision)

    assert final.specialist_team is SpecialistTeam.BILLING
    assert log[0].startswith("team_consistency")


def test_team_is_cleared_for_auto_respond():
    final, log = run(make_decision(specialist_team=SpecialistTeam.PRODUCT))

    assert final.specialist_team is None
    assert log[0].startswith("team_consistency")


def test_several_problems_produce_a_single_escalation():
    decision = make_decision(urgency=Urgency.CRITICAL, confidence=0.2, sentiment=Sentiment.ANGRY)
    final, log = run(decision)

    assert final.action is Action.ESCALATE_HUMAN
    assert len(log) == 1


def test_unknown_customer_does_not_crash_customer_rules():
    ticket = DARK_MODE_TICKET.model_copy(update={"customer_id": "c_unknown"})
    final, log = run(make_decision(), ticket)

    assert final.action is Action.AUTO_RESPOND
    assert log == []


# ---------- the safety property ----------

CAUTION = {Action.AUTO_RESPOND: 0, Action.ROUTE_SPECIALIST: 1, Action.ESCALATE_HUMAN: 2}


def test_guardrails_never_make_a_decision_less_cautious():
    """Whatever the LLM says, the final action is at least as cautious as its own."""
    grid = product(
        Action, Urgency, Sentiment, [0.2, 0.9], [["kb_006"], []], [DARK_MODE_TICKET, BILLING_TICKET]
    )
    checked = 0
    for action, urgency, sentiment, confidence, kb_ids, ticket in grid:
        team = None if action is Action.AUTO_RESPOND else SpecialistTeam.BILLING
        decision = make_decision(
            action=action,
            urgency=urgency,
            sentiment=sentiment,
            confidence=confidence,
            kb_article_ids=kb_ids,
            specialist_team=team,
        )
        final, _ = apply_guardrails(decision, make_context(ticket), DEFAULT_RULES)

        assert CAUTION[final.action] >= CAUTION[decision.action]
        assert (final.specialist_team is None) == (final.action is Action.AUTO_RESPOND)
        checked += 1
    assert checked == 3 * 4 * 4 * 2 * 2 * 2
