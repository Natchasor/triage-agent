"""The agent loop: the LLM calls tools to gather facts, then submits a structured decision.

Uses the OpenAI Responses API. The conversation is a list of "input items": our messages,
the model's own output items (reasoning, function calls) and our function_call_output items.
"""
import json
import logging
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from .config import Settings
from .models import Ticket, TriageDecision, TriageResult
from .prompts import SYSTEM_PROMPT, build_ticket_message
from .schema_utils import inline_refs
from .tools import ToolRegistry

logger = logging.getLogger(__name__)

SUBMIT_TOOL_NAME = "submit_triage_decision"
FORCE_SUBMIT = {"type": "function", "name": SUBMIT_TOOL_NAME}
NUDGE = f"Call your tools to gather facts, then submit your decision with {SUBMIT_TOOL_NAME}."


class TriageError(Exception):
    """The agent could not produce a valid decision for a ticket."""


def submit_tool_spec() -> dict[str, Any]:
    """The final-answer tool. Its argument schema is the TriageDecision model."""
    return {
        "type": "function",
        "name": SUBMIT_TOOL_NAME,
        "description": (
            "Submit the final triage decision for this ticket. Call it exactly once, "
            "after gathering what you need from the other tools."
        ),
        "parameters": inline_refs(TriageDecision.model_json_schema()),
        "strict": False,
    }


class TriageAgent:
    def __init__(self, client: OpenAI, registry: ToolRegistry, settings: Settings) -> None:
        self._client = client
        self._registry = registry
        self._settings = settings
        self._tool_specs = [*registry.specs(), submit_tool_spec()]

    def triage(self, ticket: Ticket) -> TriageResult:
        input_items: list[Any] = [{"role": "user", "content": build_ticket_message(ticket)}]
        tools_used: list[str] = []
        total_rounds = self._settings.max_tool_rounds

        for round_number in range(1, total_rounds + 1):
            is_last_round = round_number == total_rounds
            response = self._client.responses.create(
                **self._request(input_items, force_submit=is_last_round)
            )
            # Keep the model's own output (including reasoning) in the conversation.
            input_items.extend(response.output)

            calls = [item for item in response.output if item.type == "function_call"]
            if not calls:
                input_items.append({"role": "user", "content": NUDGE})
                continue

            decision = self._run_tool_calls(calls, input_items, tools_used)
            if decision is not None:
                return TriageResult(
                    ticket_id=ticket.ticket_id, decision=decision, tools_used=tools_used
                )

        raise TriageError(f"No valid decision for {ticket.ticket_id} after {total_rounds} rounds")

    def _request(self, input_items: list[Any], *, force_submit: bool) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self._settings.model,
            "instructions": SYSTEM_PROMPT,
            "input": input_items,
            "tools": self._tool_specs,
            "tool_choice": FORCE_SUBMIT if force_submit else "required",
        }
        if self._settings.reasoning_effort:
            request["reasoning"] = {"effort": self._settings.reasoning_effort}
        return request

    def _run_tool_calls(
        self, calls: list[Any], input_items: list[Any], tools_used: list[str]
    ) -> TriageDecision | None:
        """Answer every function call. Returns the decision if a valid one was submitted."""
        decision: TriageDecision | None = None
        for call in calls:
            logger.info("tool call: %s %s", call.name, call.arguments)

            if call.name == SUBMIT_TOOL_NAME:
                try:
                    decision = TriageDecision.model_validate_json(call.arguments)
                    output: dict[str, Any] = {"status": "accepted"}
                except ValidationError as exc:
                    output = {"error": f"Decision rejected. Fix and resubmit. Details: {exc}"}
            else:
                tools_used.append(call.name)
                output = self._registry.execute(call.name, call.arguments)

            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(output, ensure_ascii=False),
                }
            )
        return decision
