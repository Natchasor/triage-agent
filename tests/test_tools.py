import json

import pytest
from pydantic import BaseModel

from triage.config import Settings
from triage.tools import Tool, ToolRegistry, build_registry

DATA_DIR = Settings().data_dir


@pytest.fixture(scope="module")
def registry() -> ToolRegistry:
    return build_registry(DATA_DIR)


def call(registry: ToolRegistry, name: str, **arguments):
    return registry.execute(name, json.dumps(arguments))


# ---------- registry ----------


def test_registry_exposes_the_core_tools_as_openai_function_specs(registry):
    specs = registry.specs()
    core_tools = {
        "get_customer_history",
        "search_knowledge_base",
        "get_billing_transactions",
        "check_system_status",
    }
    assert core_tools <= {s["name"] for s in specs}
    for spec in specs:
        assert spec["type"] == "function"
        assert spec["description"]
        assert spec["strict"] is False
        assert spec["parameters"]["type"] == "object"
        assert spec["parameters"]["required"]


def test_duplicate_tool_names_are_rejected():
    from triage.tools.customer_history import GetCustomerHistory

    tool = GetCustomerHistory({})
    with pytest.raises(ValueError, match="Duplicate"):
        ToolRegistry([tool, tool])


def test_unknown_tool_returns_error_instead_of_raising(registry):
    result = registry.execute("delete_everything", "{}")
    assert "Unknown tool" in result["error"]


def test_invalid_json_arguments_return_error(registry):
    result = registry.execute("get_customer_history", "{not json")
    assert "Invalid arguments" in result["error"]


def test_missing_required_argument_returns_error(registry):
    result = registry.execute("get_customer_history", "{}")
    assert "customer_id" in result["error"]


def test_tool_crash_is_contained():
    class _Args(BaseModel):
        pass

    class Exploding(Tool):
        name = "explode"
        description = "always fails"
        args_model = _Args

        def run(self, args):
            raise RuntimeError("boom")

    result = ToolRegistry([Exploding()]).execute("explode", "{}")
    assert result == {"error": "Tool 'explode' failed: boom"}


# ---------- get_customer_history ----------


def test_customer_history_known_customer(registry):
    result = call(registry, "get_customer_history", customer_id="c_1002")
    assert result["found"] is True
    assert result["plan"] == "enterprise"
    assert result["seats"] == 45


def test_customer_history_unknown_customer(registry):
    assert call(registry, "get_customer_history", customer_id="nope") == {
        "found": False,
        "customer_id": "nope",
    }


# ---------- search_knowledge_base ----------


@pytest.mark.parametrize(
    ("query", "expected_top_id"),
    [
        ("payment failed upgrade pro card", "kb_001"),
        ("duplicate charges refund dispute", "kb_002"),
        ("error 500 cannot access server", "kb_004"),
        ("dark mode system default theme", "kb_006"),
    ],
)
def test_knowledge_base_ranks_the_right_article_first(registry, query, expected_top_id):
    result = call(registry, "search_knowledge_base", query=query)
    assert result["results"][0]["id"] == expected_top_id


def test_knowledge_base_returns_nothing_for_unrelated_query(registry):
    assert call(registry, "search_knowledge_base", query="quantum banana")["results"] == []


def test_knowledge_base_respects_max_results(registry):
    result = call(registry, "search_knowledge_base", query="billing refund charge", max_results=1)
    assert len(result["results"]) == 1


def test_knowledge_base_rejects_out_of_range_max_results(registry):
    result = call(registry, "search_knowledge_base", query="refund", max_results=50)
    assert "Invalid arguments" in result["error"]


# ---------- get_billing_transactions ----------


def test_billing_detects_three_duplicate_charges(registry):
    result = call(registry, "get_billing_transactions", customer_id="c_1001")
    assert result["transaction_count"] == 3
    assert result["total_usd"] == 89.97
    assert result["duplicate_charges_suspected"] is True


def test_billing_customer_without_transactions(registry):
    result = call(registry, "get_billing_transactions", customer_id="c_1003")
    assert result["transaction_count"] == 0
    assert result["duplicate_charges_suspected"] is False


# ---------- check_system_status ----------


def test_status_detects_incident_hidden_by_public_page(registry):
    result = call(registry, "check_system_status", region="TH")
    assert result["internal"]["region"] == "ap-southeast"
    assert result["internal"]["health"] == "degraded"
    assert result["status_page_mismatch"] is True


def test_status_healthy_region_has_no_mismatch(registry):
    result = call(registry, "check_system_status", region="us-east")
    assert result["internal"]["health"] == "healthy"
    assert result["status_page_mismatch"] is False


def test_status_unknown_region_lists_known_ones(registry):
    result = call(registry, "check_system_status", region="mars")
    assert result["found"] is False
    assert "ap-southeast" in result["known_regions"]


def test_every_tool_result_is_json_serialisable(registry):
    calls = [
        ("get_customer_history", {"customer_id": "c_1002"}),
        ("search_knowledge_base", {"query": "error 500"}),
        ("get_billing_transactions", {"customer_id": "c_1001"}),
        ("check_system_status", {"region": "TH"}),
    ]
    for name, arguments in calls:
        json.dumps(call(registry, name, **arguments), ensure_ascii=False)
