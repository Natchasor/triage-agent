"""Tool registry wiring. To add a tool: create its module and add one line below."""
from pathlib import Path

from .base import Tool, ToolRegistry
from .billing import GetBillingTransactions
from .customer_history import GetCustomerHistory
from .knowledge_base import SearchKnowledgeBase
from .system_status import CheckSystemStatus

__all__ = ["Tool", "ToolRegistry", "build_registry"]


def build_registry(data_dir: Path) -> ToolRegistry:
    return ToolRegistry(
        [
            GetCustomerHistory.from_data_dir(data_dir),
            SearchKnowledgeBase.from_data_dir(data_dir),
            GetBillingTransactions.from_data_dir(data_dir),
            CheckSystemStatus.from_data_dir(data_dir),
        ]
    )
