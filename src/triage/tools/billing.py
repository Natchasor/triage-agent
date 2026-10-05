"""Tool: list a customer's recent payment transactions."""
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..data_loader import read_json
from .base import Tool


class Transaction(BaseModel):
    transaction_id: str
    amount_usd: float
    status: str
    description: str
    minutes_ago: int
    refund_status: str


class BillingArgs(BaseModel):
    customer_id: str = Field(description="Customer ID from the ticket, e.g. 'c_1001'.")


class GetBillingTransactions(Tool):
    name = "get_billing_transactions"
    description = (
        "List a customer's recent payment transactions (amount, status, age, refund status). "
        "Use it for failed payments, duplicate charges and refund questions. Totals and "
        "duplicate detection are computed for you, so do not do the arithmetic yourself."
    )
    args_model = BillingArgs

    def __init__(self, transactions: dict[str, list[Transaction]]) -> None:
        self._transactions = transactions

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "GetBillingTransactions":
        raw = read_json(data_dir / "billing.json")
        return cls(
            {cid: [Transaction.model_validate(t) for t in txns] for cid, txns in raw.items()}
        )

    def run(self, args: BillingArgs) -> dict[str, Any]:
        txns = self._transactions.get(args.customer_id, [])
        repeats = Counter(t.description for t in txns)
        return {
            "customer_id": args.customer_id,
            "transaction_count": len(txns),
            "total_usd": round(sum(t.amount_usd for t in txns), 2),
            "duplicate_charges_suspected": any(count > 1 for count in repeats.values()),
            "transactions": [t.model_dump() for t in txns],
        }
