import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pytest
import pandas as pd
from unittest.mock import patch, MagicMock

from agents.reporter import (
    generate_chart_from_spec,
    _make_reporter_tools,
    _extract_analysis,
    generate_report,
)


ANALYSIS_JSON = {
    "direct_answer": {
        "question": "Revenue by region?",
        "answer": "West: $400, East: $300",
        "why": "Direct sum from query",
        "approach": "SELECT region, SUM(revenue) GROUP BY region",
    },
    "charts": [],
    "detailed_analysis": "West leads East in revenue.",
}


def _mock_reporter_agent(gold_parquet_path: str, analysis: dict = ANALYSIS_JSON):
    """Mock create_agent that calls the real load + execute tools, then returns analysis JSON."""
    table_stem = Path(gold_parquet_path).stem.replace("-", "_").replace(" ", "_")

    def fake_create_agent(llm, tools, **kwargs):
        mock_agent = MagicMock()

        def invoke(inputs):
            load_tool, query_tool = tools[0], tools[1]
            messages = [
                MagicMock(content=load_tool.invoke({})),
                MagicMock(content=query_tool.invoke({"sql_query": f"SELECT * FROM {table_stem}"})),
                MagicMock(content=json.dumps(analysis)),
            ]
            return {"messages": messages}

        mock_agent.invoke = invoke
        return mock_agent

    return fake_create_agent


def _mock_reporter_llm(table_name: str, analysis: dict = ANALYSIS_JSON):
    llm = MagicMock()
    llm.invoke.side_effect = [
        MagicMock(content=json.dumps({"sql_query": f"SELECT * FROM {table_name}"})),
        MagicMock(content=json.dumps(analysis)),
    ]
    return llm


# ---------------------------------------------------------------------------
# generate_chart_from_spec
# ---------------------------------------------------------------------------
class TestGenerateChartFromSpec:
    def _df(self):
        return pd.DataFrame({
            "region": ["East", "West", "North"],
            "revenue": [300, 400, 200],
        })

    def test_bar_chart_returns_html(self):
        spec = {"type": "bar", "title": "Revenue", "x_column": "region", "y_column": "revenue"}
        html = generate_chart_from_spec(self._df(), spec, 1)
        assert "<div" in html

    def test_pie_chart_returns_html(self):
        spec = {"type": "pie", "title": "Share", "labels_column": "region", "values_column": "revenue"}
        html = generate_chart_from_spec(self._df(), spec, 2)
        assert "<div" in html

    def test_unknown_chart_type_returns_empty(self):
        spec = {"type": "heatmap", "title": "Unknown", "x_column": "region"}
        html = generate_chart_from_spec(self._df(), spec, 3)
        assert html == ""

    def test_bad_column_does_not_raise(self):
        spec = {"type": "bar", "title": "Bad", "x_column": "nonexistent", "y_column": "revenue"}
        # Should return empty string on error, not raise
        html = generate_chart_from_spec(self._df(), spec, 4)
        assert isinstance(html, str)


# ---------------------------------------------------------------------------
# _make_reporter_tools
# ---------------------------------------------------------------------------
class TestMakeReporterTools:
    def test_load_tool_returns_catalog(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        p = tmp_path / "gold_sales.parquet"
        df.to_parquet(str(p), index=False)

        _, load_tool, _, scratchpad, conn = _make_reporter_tools([str(p)], "run-r1")
        try:
            result = load_tool.invoke({})
            catalog = json.loads(result)
            assert "gold_sales" in catalog
            assert "columns" in catalog["gold_sales"]
            assert "id" in catalog["gold_sales"]["columns"]
        finally:
            conn.close()

    def test_execute_tool_stores_result_in_scratchpad(self, tmp_path):
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        p = tmp_path / "gold_data.parquet"
        df.to_parquet(str(p), index=False)

        _, load_tool, query_tool, scratchpad, conn = _make_reporter_tools([str(p)], "run-r2")
        try:
            load_tool.invoke({})
            query_tool.invoke({"sql_query": "SELECT * FROM gold_data"})
            assert "result_df" in scratchpad
            assert len(scratchpad["result_df"]) == 2
            assert scratchpad["sql_query"] == "SELECT * FROM gold_data"
        finally:
            conn.close()

    def test_execute_tool_returns_error_on_bad_sql(self, tmp_path):
        df = pd.DataFrame({"id": [1]})
        p = tmp_path / "gold_x.parquet"
        df.to_parquet(str(p), index=False)

        _, load_tool, query_tool, scratchpad, conn = _make_reporter_tools([str(p)], "run-r3")
        try:
            load_tool.invoke({})
            result = query_tool.invoke({"sql_query": "SELECT * FROM nonexistent_table"})
            parsed = json.loads(result)
            assert "error" in parsed
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# _extract_analysis
# ---------------------------------------------------------------------------
class TestExtractAnalysis:
    def test_extracts_direct_answer_json(self):
        result = {"messages": [MagicMock(content=json.dumps(ANALYSIS_JSON))]}
        analysis = _extract_analysis(result)
        assert analysis["direct_answer"]["answer"] == "West: $400, East: $300"

    def test_handles_json_fences(self):
        content = f"```json\n{json.dumps(ANALYSIS_JSON)}\n```"
        result = {"messages": [MagicMock(content=content)]}
        analysis = _extract_analysis(result)
        assert "direct_answer" in analysis

    def test_returns_empty_dict_when_no_direct_answer(self):
        result = {"messages": [MagicMock(content='{"other_key": "value"}')]}
        assert _extract_analysis(result) == {}

    def test_returns_empty_dict_on_no_messages(self):
        assert _extract_analysis({"messages": []}) == {}


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------
class TestGenerateReport:
    def test_entity_id_query_result_is_enriched_with_human_readable_name(self, tmp_path):
        sales = pd.DataFrame({
            "store_id": ["S001", "S003"],
            "store_name": ["Downtown", "Suburb"],
            "total_amount": [1000.0, 500.0],
        })
        gold_path = tmp_path / "sales.parquet"
        sales.to_parquet(gold_path, index=False)
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content=json.dumps({
                "sql_query": (
                    'SELECT "store_id", SUM("total_amount") AS "total_sales_amount" '
                    'FROM "sales" GROUP BY "store_id" ORDER BY "total_sales_amount" LIMIT 1'
                )
            })),
            MagicMock(content=json.dumps({
                "direct_answer": {
                    "answer": "Suburb (S003) had the lowest total sales amount of 500.",
                    "approach": "Grouped sales by store.",
                },
                "charts": [],
                "detailed_analysis": "Suburb is the lowest-sales store.",
            })),
        ]

        with patch("agents.reporter._make_llm", return_value=llm), patch(
            "agents.reporter.REPORTS_DIR", tmp_path
        ), patch("agents.reporter.store_document"):
            generate_report(
                [str(gold_path)],
                "Which store had the lowest total sales amount?",
                "run-store-name",
                task_description="Generate report.",
            )

        analysis_prompt = llm.invoke.call_args_list[1].args[0]
        assert '"store_name": "Suburb"' in analysis_prompt
        assert "Prefer human-readable name columns" in analysis_prompt

    def test_malformed_analysis_uses_data_driven_fallback(self, tmp_path):
        df = pd.DataFrame({"region": ["East", "West"], "revenue": [300, 400]})
        gold_path = tmp_path / "gold_output.parquet"
        df.to_parquet(str(gold_path), index=False)
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"sql_query": "SELECT * FROM gold_output"})),
            MagicMock(content="I could not return valid JSON."),
        ]

        with patch("agents.reporter._make_llm", return_value=llm), patch(
            "agents.reporter.REPORTS_DIR", tmp_path
        ), patch("agents.reporter.store_document"):
            path = generate_report(
                [str(gold_path)],
                "Analyse revenue by region",
                "run-fallback",
                task_description="Generate report.",
            )

        content = Path(path).read_text(encoding="utf-8")
        assert "Analysis could not be structured" not in content
        assert "2 records" in content
        assert "revenue" in content
        assert "Detailed Analysis" in content
        assert "chart_1" in content

    def test_accepts_string_direct_answer(self, tmp_path):
        df = pd.DataFrame({"region": ["West"], "revenue": [400]})
        gold_path = tmp_path / "gold_output.parquet"
        df.to_parquet(str(gold_path), index=False)
        analysis = {
            "direct_answer": "West generated $400 in revenue.",
            "charts": [],
            "detailed_analysis": "West is the only region in the result.",
        }

        with patch(
            "agents.reporter._make_llm",
            return_value=_mock_reporter_llm("gold_output", analysis),
        ), patch("agents.reporter.REPORTS_DIR", tmp_path), patch(
            "agents.reporter.store_document"
        ):
            path = generate_report(
                [str(gold_path)],
                "Analyse revenue by region",
                "run-string-answer",
                task_description="Generate report.",
            )

        assert "West generated $400 in revenue." in Path(path).read_text(encoding="utf-8")

    def test_reporter_uses_two_direct_llm_calls(self, tmp_path):
        df = pd.DataFrame({"region": ["East", "West"], "revenue": [300, 400]})
        gold_path = tmp_path / "gold_output.parquet"
        df.to_parquet(str(gold_path), index=False)
        llm = MagicMock()
        llm.invoke.side_effect = [
            MagicMock(content=json.dumps({"sql_query": "SELECT * FROM gold_output"})),
            MagicMock(content=json.dumps(ANALYSIS_JSON)),
        ]

        with patch("agents.reporter.create_agent", side_effect=AssertionError("agent created")), \
             patch("agents.reporter._make_llm", return_value=llm), \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            path = generate_report(
                [str(gold_path)], "Analyse revenue by region", "run-direct", "Generate report"
            )

        assert llm.invoke.call_count == 2
        assert Path(path).exists()

    def test_saves_html_and_returns_path(self, tmp_path):
        df = pd.DataFrame({"region": ["East", "West"], "revenue": [300, 400]})
        gold_path = tmp_path / "gold_output.parquet"
        df.to_parquet(str(gold_path), index=False)

        with patch("agents.reporter._make_llm", return_value=_mock_reporter_llm("gold_output")), \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            path = generate_report(
                [str(gold_path)], "Analyse revenue by region", "run-rpt1",
                task_description="Generate report for run-rpt1.",
            )

        assert Path(path).exists()
        content = Path(path).read_text(encoding="utf-8")
        assert "Executive Report" in content
        assert "Analyse revenue by region" in content

    def test_analysis_prompt_contains_query_results(self, tmp_path):
        df = pd.DataFrame({"region": ["East"], "revenue": [300]})
        gold_path = tmp_path / "gold_p.parquet"
        df.to_parquet(str(gold_path), index=False)

        llm = _mock_reporter_llm("gold_p")
        with patch("agents.reporter._make_llm", return_value=llm), \
             patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            generate_report(
                [str(gold_path)], "intent", "run-rpt2",
                task_description="Generate report.",
            )

        assert "East" in llm.invoke.call_args_list[1].args[0]

    def test_returns_empty_string_for_no_files(self, tmp_path):
        with patch("agents.reporter.REPORTS_DIR", tmp_path), \
             patch("agents.reporter.store_document"):
            result = generate_report([], "intent", "run-rpt3",
                                     task_description="No files.")
        assert result == ""

