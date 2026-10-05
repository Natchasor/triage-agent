"""Domain models: ticket inputs and triage outputs."""
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class Plan(StrEnum):
    FREE = "free"
    PRO = "pro"
    ENTERPRISE = "enterprise"


class Urgency(StrEnum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    ANGRY = "angry"


class IssueType(StrEnum):
    BILLING = "billing"
    OUTAGE = "outage"
    BUG = "bug"
    FEATURE_REQUEST = "feature_request"
    HOW_TO = "how_to"
    ACCOUNT = "account"
    OTHER = "other"


class Action(StrEnum):
    AUTO_RESPOND = "auto_respond"
    ROUTE_SPECIALIST = "route_specialist"
    ESCALATE_HUMAN = "escalate_human"


class SpecialistTeam(StrEnum):
    BILLING = "billing"
    TECHNICAL_SUPPORT = "technical_support"
    INFRASTRUCTURE = "infrastructure"
    PRODUCT = "product"


# ---------- Inputs ----------


class Message(BaseModel):
    text: str
    minutes_ago: int = Field(ge=0)

    @property
    def age_label(self) -> str:
        if self.minutes_ago == 0:
            return "just now"
        if self.minutes_ago < 60:
            return f"{self.minutes_ago} min ago"
        if self.minutes_ago < 1440:
            return f"{self.minutes_ago // 60} h ago"
        return f"{self.minutes_ago // 1440} d ago"


class Customer(BaseModel):
    customer_id: str
    plan: Plan
    region: str
    seats: int
    tenure_months: int
    previous_tickets: int
    notes: str = ""


class Ticket(BaseModel):
    ticket_id: str
    customer_id: str
    messages: list[Message]

    def transcript(self) -> str:
        """All messages, oldest first, each prefixed with its age."""
        return "\n".join(f"[{m.age_label}] {m.text}" for m in self.messages)


# ---------- Outputs ----------


class TriageDecision(BaseModel):
    """The structured answer the LLM must return for each ticket."""

    urgency: Urgency
    product: str = Field(description="Product or feature the ticket is about.")
    issue_type: IssueType
    sentiment: Sentiment
    language: str = Field(description="ISO 639-1 code of the customer's language, e.g. 'en'.")
    summary: str = Field(description="One-sentence summary of the customer's problem.")
    action: Action
    specialist_team: SpecialistTeam | None = Field(
        description="Team to route to, or null when action is auto_respond."
    )
    reasoning: str = Field(description="Short explanation of the urgency and action chosen.")
    confidence: float = Field(description="Confidence in this decision, from 0.0 to 1.0.")
    kb_article_ids: list[str] = Field(description="IDs of knowledge-base articles relied on.")
    draft_reply: str = Field(description="Reply to the customer, written in their language.")

    @field_validator("confidence")
    @classmethod
    def _confidence_in_range(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0.0 and 1.0")
        return value


class TriageResult(BaseModel):
    """Final output: the LLM decision plus facts recorded by our own code."""

    ticket_id: str
    decision: TriageDecision
    tools_used: list[str]
    guardrail_overrides: list[str] = Field(default_factory=list)
