"""Tool: look up a customer's account profile and support history."""
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from ..data_loader import load_customers
from ..models import Customer
from .base import Tool


class CustomerHistoryArgs(BaseModel):
    customer_id: str = Field(description="Customer ID from the ticket, e.g. 'c_1001'.")


class GetCustomerHistory(Tool):
    name = "get_customer_history"
    description = (
        "Look up a customer's plan, region, seats, tenure, number of previous support "
        "tickets and account notes. Use it to judge how important the customer is and "
        "whether this is a repeat contact."
    )
    args_model = CustomerHistoryArgs

    def __init__(self, customers: dict[str, Customer]) -> None:
        self._customers = customers

    @classmethod
    def from_data_dir(cls, data_dir: Path) -> "GetCustomerHistory":
        return cls(load_customers(data_dir))

    def run(self, args: CustomerHistoryArgs) -> dict[str, Any]:
        customer = self._customers.get(args.customer_id)
        if customer is None:
            return {"found": False, "customer_id": args.customer_id}
        return {"found": True, **customer.model_dump(mode="json")}
