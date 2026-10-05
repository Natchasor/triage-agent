"""Deterministic guardrails applied after the LLM decision.

The LLM proposes; these rules have the last word. A rule can only make a decision more
cautious (auto_respond -> route_specialist -> escalate_human), never less. Rules rely on
facts from our own data (customer record, knowledge-base IDs, ticket text), not on the
LLM's claims about them.
"""
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from .models import (
    Action,
    Customer,
    IssueType,
    Plan,
    Sentiment,
    SpecialistTeam,
    Ticket,
    TriageDecision,
    Urgency,
)

MIN_CONFIDENCE = 0.6

DEFAULT_TEAM = {
    IssueType.BILLING: SpecialistTeam.BILLING,
    IssueType.OUTAGE: SpecialistTeam.INFRASTRUCTURE,
    IssueType.BUG: SpecialistTeam.TECHNICAL_SUPPORT,
    IssueType.FEATURE_REQUEST: SpecialistTeam.PRODUCT,
    IssueType.HOW_TO: SpecialistTeam.TECHNICAL_SUPPORT,
    IssueType.ACCOUNT: SpecialistTeam.TECHNICAL_SUPPORT,
    IssueType.OTHER: SpecialistTeam.TECHNICAL_SUPPORT,
}

# Words that signal a payment dispute or legal threat (English and Thai). A false positive
# only means a human reads the ticket, which is the safe direction to be wrong.
_DISPUTE_PATTERN = re.compile(
    r"charge\s?back|disput(?:e|es|ed|ing)|lawyer|attorney|lawsuit|legal action|\bsue\b|ฟ้อง|ทนาย",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class GuardrailContext:
    """Facts from our own systems that rules may check the LLM against."""

    ticket: Ticket
    customer: Customer | None
    known_kb_ids: frozenset[str]


@dataclass(frozen=True)
class Override:
    message: str
    changes: dict[str, Any]


Rule = Callable[[TriageDecision, GuardrailContext], Override | None]


def _not_escalated(decision: TriageDecision) -> bool:
    return decision.action is not Action.ESCALATE_HUMAN


def _escalate(rule: str, decision: TriageDecision, reason: str) -> Override:
    team = decision.specialist_team or DEFAULT_TEAM[decision.issue_type]
    return Override(
        message=f"{rule}: {decision.action.value} -> escalate_human ({reason})",
        changes={"action": Action.ESCALATE_HUMAN, "specialist_team": team},
    )


def critical_needs_human(decision: TriageDecision, ctx: GuardrailContext) -> Override | None:
    if decision.urgency is Urgency.CRITICAL and _not_escalated(decision):
        return _escalate("critical_urgency", decision, "critical tickets always need a human")
    return None


def dispute_needs_human(decision: TriageDecision, ctx: GuardrailContext) -> Override | None:
    if _not_escalated(decision) and _DISPUTE_PATTERN.search(ctx.ticket.transcript()):
        return _escalate("dispute_threat", decision, "customer mentions a dispute or legal action")
    return None


def enterprise_priority_needs_human(
    decision: TriageDecision, ctx: GuardrailContext
) -> Override | None:
    customer = ctx.customer
    is_enterprise = customer is not None and customer.plan is Plan.ENTERPRISE
    is_urgent = decision.urgency in (Urgency.CRITICAL, Urgency.HIGH)
    if is_enterprise and is_urgent and _not_escalated(decision):
        return _escalate("enterprise_sla", decision, "urgent ticket from an Enterprise customer")
    return None


def angry_customer_needs_human(decision: TriageDecision, ctx: GuardrailContext) -> Override | None:
    if decision.sentiment is Sentiment.ANGRY and decision.action is Action.AUTO_RESPOND:
        return _escalate("angry_customer", decision, "angry customers are not auto-answered")
    return None


def low_confidence_needs_human(decision: TriageDecision, ctx: GuardrailContext) -> Override | None:
    if decision.confidence < MIN_CONFIDENCE and _not_escalated(decision):
        reason = f"confidence {decision.confidence:.2f} is below {MIN_CONFIDENCE}"
        return _escalate("low_confidence", decision, reason)
    return None


def auto_respond_needs_grounding(
    decision: TriageDecision, ctx: GuardrailContext
) -> Override | None:
    """An automatic answer must rest on at least one real knowledge-base article."""
    grounded = bool(set(decision.kb_article_ids) & ctx.known_kb_ids)
    if decision.action is Action.AUTO_RESPOND and not grounded:
        return Override(
            message="ungrounded_answer: auto_respond -> route_specialist "
            "(no valid knowledge-base article cited)",
            changes={
                "action": Action.ROUTE_SPECIALIST,
                "specialist_team": DEFAULT_TEAM[decision.issue_type],
            },
        )
    return None


def team_matches_action(decision: TriageDecision, ctx: GuardrailContext) -> Override | None:
    """Keep specialist_team consistent: set for human handling, empty for auto-respond."""
    if decision.action is Action.AUTO_RESPOND and decision.specialist_team is not None:
        return Override("team_consistency: cleared team for auto_respond", {"specialist_team": None})
    if decision.action is not Action.AUTO_RESPOND and decision.specialist_team is None:
        team = DEFAULT_TEAM[decision.issue_type]
        return Override(
            f"team_consistency: assigned {team.value} (none given)", {"specialist_team": team}
        )
    return None


DEFAULT_RULES: tuple[Rule, ...] = (
    critical_needs_human,
    dispute_needs_human,
    enterprise_priority_needs_human,
    angry_customer_needs_human,
    low_confidence_needs_human,
    auto_respond_needs_grounding,
    team_matches_action,  # last, so it sees the final action
)


def apply_guardrails(
    decision: TriageDecision,
    ctx: GuardrailContext,
    rules: Sequence[Rule] = DEFAULT_RULES,
) -> tuple[TriageDecision, list[str]]:
    """Run the rules in order. Returns the final decision and a log of what changed."""
    log: list[str] = []
    for rule in rules:
        override = rule(decision, ctx)
        if override is not None:
            decision = decision.model_copy(update=override.changes)
            log.append(override.message)
    return decision, log
