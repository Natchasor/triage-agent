"""Helpers for reading the mock data files."""
import json
from pathlib import Path
from typing import Any

from .models import Customer, Ticket


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tickets(data_dir: Path) -> list[Ticket]:
    return [Ticket.model_validate(item) for item in read_json(data_dir / "tickets.json")]


def load_customers(data_dir: Path) -> dict[str, Customer]:
    customers = [Customer.model_validate(item) for item in read_json(data_dir / "customers.json")]
    return {customer.customer_id: customer for customer in customers}


def load_kb_ids(data_dir: Path) -> frozenset[str]:
    return frozenset(article["id"] for article in read_json(data_dir / "kb.json"))
