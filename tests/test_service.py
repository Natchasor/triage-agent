from triage.config import Settings
from triage.data_loader import load_customers, load_kb_ids, load_tickets
from triage.models import (
    Action,
    IssueType,
    Sentiment,
    SpecialistTeam,
    TriageDecision,
    TriageResult,
    Urgency,
)
from triage.service import TriageService

DATA_DIR = Settings().data_dir
BILLING_TICKET = next(t for t in load_tickets(DATA_DIR) if t.ticket_id == "T-1001")


class StubAgent:
    """Returns a fixed result, standing in for the LLM agent."""

    def __init__(self, decision: TriageDecision) -> None:
        self._decision = decision

    def triage(self, ticket):
        return TriageResult(
            ticket_id=ticket.ticket_id, decision=self._decision, tools_used=["search_knowledge_base"]
        )


def make_service(decision: TriageDecision) -> TriageService:
    return TriageService(
        StubAgent(decision), load_customers(DATA_DIR), load_kb_ids(DATA_DIR)
    )


def decision(**overrides) -> TriageDecision:
    fields = {
        "urgency": Urgency.MEDIUM,
        "product": "Billing",
        "issue_type": IssueType.BILLING,
        "sentiment": Sentiment.NEUTRAL,
        "language": "en",
        "summary": "Payment problem.",
        "action": Action.AUTO_RESPOND,
        "specialist_team": None,
        "reasoning": "KB covers it.",
        "confidence": 0.9,
        "kb_article_ids": ["kb_001"],
        "draft_reply": "Hello",
    }
    return TriageDecision(**{**fields, **overrides})


def test_service_corrects_an_unsafe_llm_decision_and_records_why():
    # The LLM wrongly auto-responds to a ticket where the customer threatens a bank dispute.
    result = make_service(decision()).triage(BILLING_TICKET)

    assert result.decision.action is Action.ESCALATE_HUMAN
    assert result.decision.specialist_team is SpecialistTeam.BILLING
    assert result.guardrail_overrides[0].startswith("dispute_threat")


def test_service_keeps_agent_facts_such_as_tools_used():
    result = make_service(decision()).triage(BILLING_TICKET)

    assert result.ticket_id == "T-1001"
    assert result.tools_used == ["search_knowledge_base"]


def test_service_passes_a_safe_decision_through_untouched():
    safe = decision(
        action=Action.ESCALATE_HUMAN, specialist_team=SpecialistTeam.BILLING, urgency=Urgency.HIGH
    )
    result = make_service(safe).triage(BILLING_TICKET)

    assert result.decision == safe
    assert result.guardrail_overrides == []
