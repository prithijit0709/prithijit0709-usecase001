import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from agents.sttm_generator import (
    _prepare_bronze_context,
    _prepare_silver_context,
    _prepare_gold_context,
    _extract_sttm_rows,
    generate_bronze_sttm,
    generate_silver_sttm,
    generate_gold_sttm,
)


BRONZE_ROW = {
    "source_schema": "CSV", "source_table": "data.csv",
    "source_column": "id", "target_schema": "Bronze",
    "target_table": "bronze_data", "target_column": "id",
    "transformation_type": "Direct", "transformation_logic": "Passthrough",
}
SILVER_ROW = {
    "source_schema": "Bronze", "source_table": "bronze_data",
    "source_column": "id", "target_schema": "Silver",
    "target_table": "silver_data", "target_column": "id",
    "transformation_type": "Direct", "transformation_logic": "Passthrough",
}
GOLD_ROW = {
    "source_schema": "Silver", "source_table": "silver_data",
    "source_column": "id", "target_schema": "Gold",
    "target_table": "gold_data", "target_column": "id",
    "transformation_type": "Direct", "transformation_logic": "Passthrough",
}


def _mock_agent(sttm_rows: list[dict], context_tool=None):
    """Return a fake create_agent that optionally calls the context tool then returns STTM rows."""
    def fake_create_agent(llm, tools, **kwargs):
        mock_agent = MagicMock()
        def invoke(inputs):
            messages = []
            if context_tool is not None:
                tool_result = context_tool.invoke({})
                messages.append(MagicMock(content=tool_result))
            messages.append(MagicMock(content=json.dumps(sttm_rows)))
            return {"messages": messages}
        mock_agent.invoke = invoke
        return mock_agent
    return fake_create_agent


def _mock_llm(sttm_rows: list[dict]):
    llm = MagicMock()
    llm.invoke.return_value = MagicMock(content=json.dumps(sttm_rows))
    return llm


# ---------------------------------------------------------------------------
# _prepare_bronze_context
# ---------------------------------------------------------------------------
class TestPrepareBronzeContext:
    def test_returns_profile_dict(self, tmp_path):
        profile = {"files": ["a.csv"], "datasets": {"a": {"columns": {"id": {}}}}}
        p = tmp_path / "profile.json"
        p.write_text(json.dumps(profile))
        result = _prepare_bronze_context(str(p))
        assert result == profile

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _prepare_bronze_context(str(tmp_path / "missing.json"))


# ---------------------------------------------------------------------------
# _prepare_silver_context
# ---------------------------------------------------------------------------
class TestPrepareSilverContext:
    def test_filters_to_approved_columns(self, tmp_path):
        df = pd.DataFrame({"approved_col": [1, 2], "dropped_col": [3, 4], "_meta": ["a", "b"]})
        parquet_path = tmp_path / "bronze.parquet"
        df.to_parquet(str(parquet_path), index=False)

        sttm = pd.DataFrame({"target_column": ["approved_col"]})
        sttm_path = tmp_path / "sttm.csv"
        sttm.to_csv(str(sttm_path), index=False)

        result = _prepare_silver_context([str(parquet_path)], str(sttm_path))
        assert len(result) == 1
        assert "approved_col" in result[0]["columns"]
        assert "_meta" in result[0]["columns"]
        assert "dropped_col" not in result[0]["columns"]

    def test_returns_filename_not_full_path(self, tmp_path):
        df = pd.DataFrame({"col": [1]})
        p = tmp_path / "bronze_data.parquet"
        df.to_parquet(str(p), index=False)
        sttm = pd.DataFrame({"target_column": ["col"]})
        sttm_path = tmp_path / "sttm.csv"
        sttm.to_csv(str(sttm_path), index=False)

        result = _prepare_silver_context([str(p)], str(sttm_path))
        assert result[0]["filename"] == "bronze_data.parquet"


# ---------------------------------------------------------------------------
# _prepare_gold_context
# ---------------------------------------------------------------------------
class TestPrepareGoldContext:
    def test_filters_to_approved_columns(self, tmp_path):
        df = pd.DataFrame({"revenue": [10], "cost": [5], "_sk": [1]})
        p = tmp_path / "silver.parquet"
        df.to_parquet(str(p), index=False)
        sttm = pd.DataFrame({"target_column": ["revenue"]})
        sttm_path = tmp_path / "sttm.csv"
        sttm.to_csv(str(sttm_path), index=False)

        result = _prepare_gold_context([str(p)], str(sttm_path))
        assert "revenue" in result[0]["columns"]
        assert "_sk" in result[0]["columns"]
        assert "cost" not in result[0]["columns"]


# ---------------------------------------------------------------------------
# _extract_sttm_rows
# ---------------------------------------------------------------------------
class TestExtractSttmRows:
    def test_extracts_plain_json_array(self):
        result = {"messages": [MagicMock(content=json.dumps([BRONZE_ROW]))]}
        rows = _extract_sttm_rows(result)
        assert rows == [BRONZE_ROW]

    def test_extracts_json_with_fences(self):
        content = f"```json\n{json.dumps([BRONZE_ROW])}\n```"
        result = {"messages": [MagicMock(content=content)]}
        rows = _extract_sttm_rows(result)
        assert rows == [BRONZE_ROW]

    def test_returns_empty_list_when_no_array(self):
        result = {"messages": [MagicMock(content="No rules generated.")]}
        assert _extract_sttm_rows(result) == []

    def test_prefers_last_message(self):
        early = MagicMock(content=json.dumps([{"source_column": "old"}]))
        late = MagicMock(content=json.dumps([BRONZE_ROW]))
        result = {"messages": [early, late]}
        rows = _extract_sttm_rows(result)
        assert rows == [BRONZE_ROW]


# ---------------------------------------------------------------------------
# generate_bronze_sttm
# ---------------------------------------------------------------------------
class TestGenerateBronzeSttm:
    def test_generation_does_not_use_outer_react_agent(self, tmp_path):
        profile = {"files": ["a.csv"], "datasets": {}}
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(json.dumps(profile))
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=json.dumps([BRONZE_ROW]))

        with patch("agents.sttm_generator.create_agent", side_effect=AssertionError("agent created")), \
             patch("agents.sttm_generator._make_llm", return_value=llm), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_bronze_sttm(
                str(profile_path), "run-direct", "Generate Bronze STTM."
            )

        assert Path(path).exists()
        assert len(pd.read_csv(path)) == 1

    def test_saves_csv_and_returns_path(self, tmp_path):
        profile = {"files": ["a.csv"], "datasets": {}}
        p = tmp_path / "profile.json"
        p.write_text(json.dumps(profile))

        with patch("agents.sttm_generator._make_llm", return_value=_mock_llm([BRONZE_ROW])), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_bronze_sttm(
            str(p), "run-b1", "Generate Bronze STTM for run-b1."
            )

        assert Path(path).exists()
        df = pd.read_csv(path)
        assert "source_column" in df.columns
        assert len(df) == 1

    def test_rejects_empty_bronze_sttm(self, tmp_path):
        profile = {"files": [], "datasets": {}}
        p = tmp_path / "profile.json"
        p.write_text(json.dumps(profile))

        with patch("agents.sttm_generator._make_llm", return_value=_mock_llm([])), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            with pytest.raises(RuntimeError, match="no valid rows"):
                generate_bronze_sttm(str(p), "run-b2", "Generate Bronze STTM.")


# ---------------------------------------------------------------------------
# generate_silver_sttm
# ---------------------------------------------------------------------------
class TestGenerateSilverSttm:
    def test_generation_calls_llm_once_and_maps_every_column(self, tmp_path):
        sales = pd.DataFrame({
            "sale_date": ["2026-08-01"],
            "store_id": [1],
            "store_name": ["Central"],
            "total_amount": [125.5],
        })
        parquet = tmp_path / "sales_bronze.parquet"
        sales.to_parquet(str(parquet), index=False)
        sttm_csv = tmp_path / "bronze_sttm.csv"
        pd.DataFrame({"target_column": list(sales.columns)}).to_csv(sttm_csv, index=False)
        llm = _mock_llm([{
            **SILVER_ROW,
            "source_table": "sales_bronze",
            "source_column": "total_amount",
            "target_table": "sales_silver",
            "target_column": "total_amount",
            "transformation_type": "Direct",
            "transformation_logic": "Fill null with median and cast to decimal",
        }])

        with patch("agents.sttm_generator._make_llm", return_value=llm), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_silver_sttm(
                [str(parquet)], str(sttm_csv), "run-local", "Generate Silver STTM."
            )

        result = pd.read_csv(path).fillna("")
        llm.invoke.assert_called_once()
        assert result.iloc[0]["target_column"] == "pk_sales_silver_id"
        assert set(sales.columns).issubset(set(result["source_column"]))
        assert len(result) == len(sales.columns) + 1
        amount_rule = result[result["source_column"] == "total_amount"].iloc[0]
        assert amount_rule["transformation_logic"] == "Fill null with median and cast to decimal"

    def test_malformed_llm_response_falls_back_to_type_safe_complete_rows(self, tmp_path):
        bronze = pd.DataFrame({
            "event_date": ["2026-08-01"],
            "customer_id": [1],
            "amount": [10.5],
        })
        parquet = tmp_path / "events_bronze.parquet"
        bronze.to_parquet(parquet, index=False)
        sttm_csv = tmp_path / "bronze_sttm.csv"
        pd.DataFrame({"target_column": list(bronze.columns)}).to_csv(sttm_csv, index=False)
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='[{"source_column": "amount"')

        with patch("agents.sttm_generator._make_llm", return_value=llm), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_silver_sttm(
                [str(parquet)], str(sttm_csv), "run-safe", "Generate Silver STTM."
            )

        result = pd.read_csv(path).fillna("")
        llm.invoke.assert_called_once()
        assert len(result) == 4
        assert "date" in result.loc[result["source_column"] == "event_date", "transformation_logic"].iloc[0].lower()
        assert "integer" in result.loc[result["source_column"] == "customer_id", "transformation_logic"].iloc[0].lower()
        assert "decimal" in result.loc[result["source_column"] == "amount", "transformation_logic"].iloc[0].lower()

    def test_incompatible_ai_cast_keeps_dtype_safe_rule(self, tmp_path):
        bronze = pd.DataFrame({"amount": [10.5]})
        parquet = tmp_path / "sales_bronze.parquet"
        bronze.to_parquet(parquet, index=False)
        sttm_csv = tmp_path / "bronze_sttm.csv"
        pd.DataFrame({"target_column": ["amount"]}).to_csv(sttm_csv, index=False)
        llm = _mock_llm([{
            **SILVER_ROW,
            "source_table": "sales_bronze",
            "source_column": "amount",
            "transformation_logic": "Cast to date format YYYY-MM-DD",
        }])

        with patch("agents.sttm_generator._make_llm", return_value=llm), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_silver_sttm(
                [str(parquet)], str(sttm_csv), "run-types", "Generate Silver STTM."
            )

        result = pd.read_csv(path).fillna("")
        amount_logic = result.loc[
            result["source_column"] == "amount", "transformation_logic"
        ].iloc[0].lower()
        assert "decimal" in amount_logic
        assert "date" not in amount_logic

    def test_saves_csv_and_returns_path(self, tmp_path):
        df = pd.DataFrame({"id": [1], "val": [2]})
        parquet = tmp_path / "bronze.parquet"
        df.to_parquet(str(parquet), index=False)
        sttm_csv = tmp_path / "bronze_sttm.csv"
        pd.DataFrame({"target_column": ["id", "val"]}).to_csv(str(sttm_csv), index=False)

        with patch("agents.sttm_generator._make_llm", return_value=_mock_llm([SILVER_ROW])), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_silver_sttm(
            [str(parquet)], str(sttm_csv), "run-s1", "Generate Silver STTM for run-s1."
            )

        assert Path(path).exists()
        df_out = pd.read_csv(path)
        assert len(df_out) == 3
        assert set(df_out["target_column"]) == {"pk_bronze_silver_id", "id", "val"}


# ---------------------------------------------------------------------------
# generate_gold_sttm
# ---------------------------------------------------------------------------
class TestGenerateGoldSttm:
    def test_saves_csv_and_returns_path(self, tmp_path):
        df = pd.DataFrame({"revenue": [100], "region": ["EU"]})
        parquet = tmp_path / "silver.parquet"
        df.to_parquet(str(parquet), index=False)
        sttm_csv = tmp_path / "silver_sttm.csv"
        pd.DataFrame({"target_column": ["revenue", "region"]}).to_csv(str(sttm_csv), index=False)

        with patch("agents.sttm_generator._make_llm", return_value=_mock_llm([GOLD_ROW])), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_gold_sttm(
                [str(parquet)], str(sttm_csv), "intent", "run-g1",
                task_description="Generate Gold STTM for run-g1.",
            )

        assert Path(path).exists()
        df_out = pd.read_csv(path)
        assert len(df_out) == 1

    def test_malformed_llm_response_falls_back_to_complete_passthrough_rows(self, tmp_path):
        silver = pd.DataFrame({"store_id": [1], "store_name": ["Central"], "total_amount": [125.5]})
        parquet = tmp_path / "sample_sales_silver.parquet"
        silver.to_parquet(parquet, index=False)
        sttm_csv = tmp_path / "silver_sttm.csv"
        pd.DataFrame({"target_column": list(silver.columns)}).to_csv(sttm_csv, index=False)
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content='[{"source_column": "total_amount"')

        with patch("agents.sttm_generator._make_llm", return_value=llm), \
             patch("agents.sttm_generator.STTM_DIR", tmp_path), \
             patch("agents.sttm_generator.AuditLogger"):
            path = generate_gold_sttm(
                [str(parquet)], str(sttm_csv),
                "Which store had the lowest total sales amount?", "run-safe-gold",
                task_description="Generate Gold STTM.",
            )

        result = pd.read_csv(path).fillna("")
        assert len(result) == len(silver.columns) + 1
        assert result.iloc[0]["target_column"] == "pk_gold_id"
        assert set(silver.columns) == set(result.iloc[1:]["source_column"])
        assert set(result.iloc[1:]["transformation_type"]) == {"Direct"}
        assert set(result["target_table"]) == {"sample_sales"}

