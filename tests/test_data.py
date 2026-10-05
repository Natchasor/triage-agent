import pytest
from pydantic import ValidationError

from triage.config import Settings
from triage.data_loader import load_customers, load_tickets, read_json
from triage.models import Message, TriageDecision

DATA_DIR = Settings().data_dir


def _decision_kwargs(**overrides):
    base = {
        "urgency": "low",
        "product": "Appearance settings",
        "issue_type": "how_to",
        "sentiment": "positive",
        "language": "en",
        "summary": "Customer asks about dark mode.",
        "action": "auto_respond",
        "specialist_team": None,
        "reasoning": "Simple how-to answered by the knowledge base.",
        "confidence": 0.9,
        "kb_article_ids": ["kb_006"],
        "draft_reply": "Hi! Dark mode is in Settings > Appearance.",
    }
    return {**base, **overrides}


def test_three_sample_tickets_load():
    assert [t.ticket_id for t in load_tickets(DATA_DIR)] == ["T-1001", "T-1002", "T-1003"]


def test_every_ticket_has_a_known_customer():
    customers = load_customers(DATA_DIR)
    assert all(t.customer_id in customers for t in load_tickets(DATA_DIR))


def test_thai_ticket_contains_thai_text():
    ticket = next(t for t in load_tickets(DATA_DIR) if t.ticket_id == "T-1002")
    assert any("\u0e00" <= char <= "\u0e7f" for char in ticket.transcript())


def test_message_age_labels():
    assert Message(text="x", minutes_ago=0).age_label == "just now"
    assert Message(text="x", minutes_ago=45).age_label == "45 min ago"
    assert Message(text="x", minutes_ago=180).age_label == "3 h ago"
    assert Message(text="x", minutes_ago=2880).age_label == "2 d ago"


def test_knowledge_base_ids_are_unique():
    ids = [article["id"] for article in read_json(DATA_DIR / "kb.json")]
    assert len(ids) == len(set(ids))


def test_valid_decision_is_accepted():
    assert TriageDecision(**_decision_kwargs()).confidence == 0.9


@pytest.mark.parametrize("bad_confidence", [-0.1, 1.5])
def test_confidence_must_be_between_zero_and_one(bad_confidence):
    with pytest.raises(ValidationError):
        TriageDecision(**_decision_kwargs(confidence=bad_confidence))


def test_unknown_urgency_is_rejected():
    with pytest.raises(ValidationError):
        TriageDecision(**_decision_kwargs(urgency="extreme"))
