"""The triage service: the LLM agent followed by the deterministic guardrails."""
from collections.abc import Sequence
from typing import Protocol

from openai import OpenAI

from .agent import TriageAgent
from .config import Settings
from .data_loader import load_customers, load_kb_ids
from .guardrails import DEFAULT_RULES, GuardrailContext, Rule, apply_guardrails
from .models import Customer, Ticket, TriageResult
from .tools import build_registry


class Triager(Protocol):
    def triage(self, ticket: Ticket) -> TriageResult: ...


class TriageService:
    def __init__(
        self,
        agent: Triager,
        customers: dict[str, Customer],
        known_kb_ids: frozenset[str],
        rules: Sequence[Rule] = DEFAULT_RULES,
    ) -> None:
        self._agent = agent
        self._customers = customers
        self._known_kb_ids = known_kb_ids
        self._rules = rules

    def triage(self, ticket: Ticket) -> TriageResult:
        result = self._agent.triage(ticket)
        context = GuardrailContext(
            ticket=ticket,
            customer=self._customers.get(ticket.customer_id),
            known_kb_ids=self._known_kb_ids,
        )
        decision, overrides = apply_guardrails(result.decision, context, self._rules)
        return result.model_copy(update={"decision": decision, "guardrail_overrides": overrides})


def build_service(client: OpenAI, settings: Settings) -> TriageService:
    """Wire the production object graph: tools -> agent -> guardrails."""
    agent = TriageAgent(client, build_registry(settings.data_dir), settings)
    return TriageService(
        agent,
        customers=load_customers(settings.data_dir),
        known_kb_ids=load_kb_ids(settings.data_dir),
    )
