from triage.cli import main, render
from triage.models import TriageDecision, TriageResult


def _result() -> TriageResult:
    decision = TriageDecision(
        urgency="high",
        product="Billing",
        issue_type="billing",
        sentiment="angry",
        language="en",
        summary="Charged three times.",
        action="escalate_human",
        specialist_team="billing",
        reasoning="Money taken and a chargeback is threatened.",
        confidence=0.9,
        kb_article_ids=["kb_002"],
        draft_reply="We are looking into it.",
    )
    return TriageResult(
        ticket_id="T-1001", decision=decision, tools_used=["get_billing_transactions"]
    )


def test_render_shows_the_key_fields():
    text = render(_result())

    assert "T-1001" in text
    assert "escalate_human -> billing" in text
    assert "get_billing_transactions" in text
    assert "We are looking into it." in text


def test_unknown_ticket_id_exits_with_code_2_before_any_api_call(capsys):
    assert main(["--ticket", "T-9999"]) == 2
    assert "No ticket with ID T-9999" in capsys.readouterr().err


def test_render_lists_guardrail_overrides():
    result = _result().model_copy(update={"guardrail_overrides": ["dispute_threat: escalated"]})

    assert "dispute_threat: escalated" in render(result)
    assert "none triggered" in render(_result())
