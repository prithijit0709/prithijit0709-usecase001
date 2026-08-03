import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.agents.sttm_agent import STTMAgent


def _agent() -> STTMAgent:
    return STTMAgent(run_id="run-test", llm=MagicMock())


def test_invalid_layer_raises_value_error():
    agent = _agent()
    with pytest.raises(ValueError):
        agent.run(layer="platinum", tables=[{"name": "sales"}], profiles={}, intent="test")


def test_bronze_success_returns_structured_proposed_output():
    agent = _agent()
    payload = (
        '{"table":"sales","rationale":"raw-first mapping","proposals":['
        '{"source_column":"sale_id","target_column":"sale_id","action":"pass_through",'
        '"proposed_type":"string","standardization":"trim","reason":"retain fidelity",'
        '"confidence_score":"0.91","risk_level":"low"}]}'
    )
    with patch.object(agent, "invoke_llm", return_value=payload):
        result = agent.run(
            layer="bronze",
            tables=[{"name": "sales"}],
            profiles={"sales": {"columns": ["sale_id"]}},
            intent="land sales data",
        )

    assert result["layer"] == "bronze"
    assert result["status"] == "Proposed"
    assert result["rules"][0]["table"] == "sales"
    assert result["rules"][0]["proposals"][0]["status"] == "Proposed"
    assert result["rules"][0]["proposals"][0]["confidence_score"] == pytest.approx(0.91)


def test_silver_fallback_on_llm_failure():
    agent = _agent()
    with patch.object(agent, "invoke_llm", side_effect=RuntimeError("llm unavailable")):
        result = agent.run(
            layer="silver",
            tables=[{"name": "customers"}],
            profiles={"customers": {"columns": ["customer_id"]}},
            intent="clean customer rows",
        )

    assert result["status"] == "Proposed"
    assert result["rules"][0]["fallback"] is True
    assert result["rules"][0]["table"] == "customers"
    assert result["summary"]["fallbacks"] == 1


def test_gold_structured_output_contains_required_target_fields():
    agent = _agent()
    payload = (
        '{"target":"sales_kpi","sources":["sales","products"],'
        '"join_plan":[{"left":"sales","right":"products","condition":"sales.product_id = products.product_id","join_type":"left"}],'
        '"dimensions":[{"name":"category","expression":"products.category"}],'
        '"metrics":[{"name":"total_revenue","calculation":"sum(sales.amount)","kpi":"Revenue"}],'
        '"filters":["sales.amount > 0"],"rationale":"Aligned to business KPI",'
        '"confidence_score":"0.88","risk_level":"medium"}'
    )
    with patch.object(agent, "invoke_llm", return_value=payload):
        result = agent.run(
            layer="gold",
            tables=[{"name": "sales_kpi"}],
            profiles={"sales": {"columns": ["amount"]}},
            intent="build revenue KPI",
        )

    rule = result["rules"][0]
    assert rule["target"] == "sales_kpi"
    assert isinstance(rule["sources"], list)
    assert isinstance(rule["join_plan"], list)
    assert isinstance(rule["dimensions"], list)
    assert isinstance(rule["metrics"], list)
    assert isinstance(rule["filters"], list)
    assert rule["status"] == "Proposed"


def test_malformed_output_normalization_extracts_json_object():
    agent = _agent()
    payload = (
        "Result follows:\n```json\n"
        "{\"table\":\"orders\",\"proposals\":[{\"source_column\":\"id\",\"target_column\":\"id\"}]}\n"
        "```"
    )
    with patch.object(agent, "invoke_llm", return_value=payload):
        result = agent.run(
            layer="bronze",
            tables=[{"name": "orders"}],
            profiles={"orders": {"columns": ["id"]}},
            intent="ingest orders",
        )

    proposal = result["rules"][0]["proposals"][0]
    assert proposal["source_column"] == "id"
    assert proposal["target_column"] == "id"
    assert proposal["status"] == "Proposed"


def test_partial_failure_resilience_continues_other_tables():
    agent = _agent()
    first_payload = (
        '{"table":"sales","proposals":[{"source_column":"id","target_column":"id",'
        '"action":"pass_through","proposed_type":"int","standardization":"none","reason":"ok"}]}'
    )
    with patch.object(agent, "invoke_llm", side_effect=[first_payload, RuntimeError("table-specific failure")]):
        result = agent.run(
            layer="bronze",
            tables=[{"name": "sales"}, {"name": "products"}],
            profiles={"sales": {}, "products": {}},
            intent="ingest everything",
        )

    assert len(result["rules"]) == 2
    assert result["rules"][0]["table"] == "sales"
    assert result["rules"][1]["table"] == "products"
    assert result["rules"][1]["fallback"] is True
    assert result["summary"]["fallbacks"] == 1
