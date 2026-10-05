"""Command-line entry point: triage the sample tickets."""
import argparse
import logging
import sys
from textwrap import indent

from openai import OpenAI, OpenAIError

from .agent import TriageError
from .config import Settings
from .data_loader import load_tickets
from .models import TriageResult
from .service import build_service


def render(result: TriageResult) -> str:
    d = result.decision
    team = f" -> {d.specialist_team.value}" if d.specialist_team else ""
    lines = [
        f"=== {result.ticket_id} " + "=" * 50,
        f"Urgency:     {d.urgency.value}",
        f"Action:      {d.action.value}{team}",
        f"Issue:       {d.issue_type.value} | {d.product}",
        f"Sentiment:   {d.sentiment.value}",
        f"Language:    {d.language}",
        f"Confidence:  {d.confidence:.2f}",
        f"Summary:     {d.summary}",
        f"Reasoning:   {d.reasoning}",
        f"KB articles: {', '.join(d.kb_article_ids) or '-'}",
        f"Tools used:  {', '.join(result.tools_used) or '-'}",
        f"Guardrails:  {'; '.join(result.guardrail_overrides) or 'none triggered'}",
        "Draft reply:",
        indent(d.draft_reply, "    "),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Triage support tickets with an LLM agent.")
    parser.add_argument("--ticket", help="Only process this ticket ID, e.g. T-1002.")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of a summary.")
    parser.add_argument("-v", "--verbose", action="store_true", help="Log every tool call.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING, format="  [log] %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)  # hide per-request HTTP lines
    settings = Settings.from_env()

    try:
        tickets = load_tickets(settings.data_dir)
    except (OSError, ValueError) as exc:
        print(f"Cannot read {settings.data_dir / 'tickets.json'}: {exc}", file=sys.stderr)
        return 2
    if args.ticket:
        tickets = [t for t in tickets if t.ticket_id == args.ticket]
        if not tickets:
            print(f"No ticket with ID {args.ticket}", file=sys.stderr)
            return 2

    try:
        client = OpenAI(timeout=settings.request_timeout_s, max_retries=3)
    except OpenAIError as exc:
        print(f"Cannot create the OpenAI client: {exc}", file=sys.stderr)
        print("Set OPENAI_API_KEY in your .env file.", file=sys.stderr)
        return 2

    service = build_service(client, settings)
    failures = 0
    for ticket in tickets:
        try:
            result = service.triage(ticket)
        except (TriageError, OpenAIError) as exc:
            print(f"{ticket.ticket_id}: FAILED - {exc}", file=sys.stderr)
            failures += 1
            continue
        print(result.model_dump_json(indent=2) if args.json else render(result))
        print()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
