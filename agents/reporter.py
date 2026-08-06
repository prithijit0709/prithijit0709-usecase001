"""Bounded two-request reporting pipeline.

The agent receives a goal from the orchestrator, inspects available Gold tables
first to understand their schemas, forms an analytical plan, writes and executes
SQL to answer the business question, and renders an HTML report.

I/O contract (UNCHANGED — UI and orchestrator safe):
    generate_report(gold_files, business_intent, run_id, task_description) -> str
"""

import html
import json
import pandas as pd
import duckdb
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from core.config import LLM_PROVIDER, GROQ_API_KEY, GROQ_MODEL, GOOGLE_API_KEY, GEMINI_MODEL, REPORTS_DIR
from core.audit import AuditLogger
from core.llm import make_llm
from core.observability import AgentTrace
from core.memory import store_document


REPORTER_AGENT_PROMPT = """You are an autonomous Senior Data Analyst and Business Intelligence Engineer
specialising in business-intent-driven reporting from Medallion Gold layer data.
You operate independently: you receive a goal from the orchestrator, inspect the Gold
tables, form an analytical plan, write and execute SQL, and return structured analysis.

## Your operating mode — follow this EXACT sequence every time

1. THINK: Read the task. Identify the business question, the Gold files available,
   and what kind of analysis (aggregation, trend, comparison, ranking) will answer it.

2. INSPECT: Call inspect_gold_tables_tool FIRST. This gives you a lightweight preview
   of each Gold table — column names, dtypes, row count, and 3 sample rows — without
   loading full data into DuckDB. State your observations: which tables are relevant,
   which columns can answer the business question, what joins may be needed.

3. PLAN: Based on the inspection, write your analytical plan:
   - Which Gold tables will you query?
   - What SQL approach will directly answer the business question?
   - What chart type(s) will best visualise the answer?

4. ACT — two sub-steps in order:
   a. Call load_gold_data_tool to register Gold tables in DuckDB and get the full schema catalog.
   b. Call execute_query_tool(sql_query=<your_sql>) to execute your SQL and get results.

5. VERIFY & RESPOND: Analyse the query results and return ONLY a valid JSON object
   as your final answer (no markdown fences, no prose before or after).

## Available tools

- **inspect_gold_tables_tool**: Quickly previews Gold Parquet files — table names,
  column names, dtypes, row count, and 3 sample rows per table. Call this FIRST
  to understand what is available before loading into DuckDB. Returns a JSON summary.

- **load_gold_data_tool**: Loads Gold Parquet files into an in-memory DuckDB database
  and returns a full catalog of table names, column names, types, row counts, and
  sample data. Call this before execute_query_tool.

- **execute_query_tool**: Executes a SQL SELECT query against the loaded Gold tables
  in DuckDB. Pass your SQL as the sql_query parameter. Returns query results as a
  JSON array. On error returns {"error": "..."}.

## Output format
Return ONLY a valid JSON object — no markdown fences, no prose:
{
  "direct_answer": {
    "question": "Restate the business question clearly",
    "answer": "Direct answer with specific numbers from the query results",
    "why": "Evidence and reasoning from the data",
    "approach": "Describe the SQL query and analytical method used"
  },
  "charts": [
    {
      "type": "bar|line|pie|scatter",
      "title": "Chart title",
      "x_column": "column from query result",
      "y_column": "column from query result (bar/line/scatter)",
      "labels_column": "column from query result (pie only)",
      "values_column": "column from query result (pie only)",
      "reason": "Why this chart directly answers the question"
    }
  ],
  "detailed_analysis": "2-3 paragraphs of additional insights and patterns"
}

## Rules
- Include only 1-2 charts that directly answer the business question.
- Use ACTUAL column names from the query result — not from the original Gold tables.
- Be specific with numbers in the direct_answer.
- Write standard ANSI SQL compatible with DuckDB.
- If execute_query_tool returns an error, fix the SQL and retry once."""


# ---------------------------------------------------------------------------
# Pure Python helpers — no LLM
# ---------------------------------------------------------------------------

def _inspect_gold_tables(gold_files: list[str]) -> dict:
    """Quick preview of Gold Parquet tables: schema + 3 sample rows. No LLM, no DuckDB."""
    summary = {}
    for fp in gold_files:
        try:
            df = pd.read_parquet(fp)
            stem = Path(fp).stem.replace("-", "_").replace(" ", "_")
            summary[stem] = {
                "file": fp,
                "table_name": stem,
                "row_count": len(df),
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample_rows": df.head(3).to_dict(orient="records"),
            }
        except Exception as e:
            summary[Path(fp).stem] = {"file": fp, "error": str(e)}
    return summary


def _enrich_entity_labels(result_df: pd.DataFrame, source_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    enriched = result_df.copy()
    for id_column in [column for column in enriched.columns if str(column).endswith("_id")]:
        name_column = f"{str(id_column)[:-3]}_name"
        if name_column in enriched.columns:
            continue

        mapping_parts = [
            source_df[[id_column, name_column]].dropna().drop_duplicates()
            for source_df in source_dfs
            if id_column in source_df.columns and name_column in source_df.columns
        ]
        if not mapping_parts:
            continue

        mapping = pd.concat(mapping_parts, ignore_index=True).drop_duplicates()
        unambiguous_ids = mapping.groupby(id_column)[name_column].nunique()
        mapping = mapping[mapping[id_column].isin(unambiguous_ids[unambiguous_ids == 1].index)]
        mapping = mapping.drop_duplicates(subset=[id_column])
        enriched = enriched.merge(mapping, on=id_column, how="left", sort=False)
    return enriched


def generate_chart_from_spec(df: pd.DataFrame, chart_spec: dict, chart_id: int) -> str:
    """Render a single Plotly chart from an LLM-specified chart spec dict. Returns embedded HTML."""
    try:
        chart_type = chart_spec.get("type", "bar").lower()
        title = chart_spec.get("title", f"Chart {chart_id}")

        if chart_type == "bar":
            x_col = chart_spec.get("x_column")
            y_col = chart_spec.get("y_column")
            if y_col and y_col in df.columns:
                agg_data = df.groupby(x_col)[y_col].sum().sort_values(ascending=False).head(10)
                fig = go.Figure(data=[go.Bar(x=agg_data.index, y=agg_data.values, marker_color="#667eea")])
            else:
                value_counts = df[x_col].value_counts().head(10)
                fig = go.Figure(data=[go.Bar(x=value_counts.index, y=value_counts.values, marker_color="#667eea")])
            fig.update_layout(title=title, xaxis_title=x_col, yaxis_title=y_col or "Count",
                              height=450, template="plotly_white")
            return fig.to_html(include_plotlyjs="cdn", div_id=f"chart_{chart_id}")

        elif chart_type == "line":
            x_col = chart_spec.get("x_column")
            y_col = chart_spec.get("y_column")
            fig = px.line(df, x=x_col, y=y_col, title=title)
            fig.update_layout(height=450, template="plotly_white")
            return fig.to_html(include_plotlyjs="cdn", div_id=f"chart_{chart_id}")

        elif chart_type == "pie":
            labels_col = chart_spec.get("labels_column")
            values_col = chart_spec.get("values_column")
            agg_data = df.groupby(labels_col)[values_col].sum()
            fig = go.Figure(data=[go.Pie(labels=agg_data.index, values=agg_data.values)])
            fig.update_layout(title=title, height=450)
            return fig.to_html(include_plotlyjs="cdn", div_id=f"chart_{chart_id}")

        elif chart_type == "scatter":
            x_col = chart_spec.get("x_column")
            y_col = chart_spec.get("y_column")
            fig = px.scatter(df, x=x_col, y=y_col, title=title, trendline="ols")
            fig.update_layout(height=450, template="plotly_white")
            return fig.to_html(include_plotlyjs="cdn", div_id=f"chart_{chart_id}")

        return ""
    except Exception as e:
        print(f"[REPORTER] Error generating chart {chart_id}: {e}")
        return ""


def _extract_analysis(result: dict) -> dict:
    """Scan agent message history (reverse order) for a JSON object with 'direct_answer' key."""
    for msg in reversed(result.get("messages", [])):
        content = getattr(msg, "content", "")
        if not isinstance(content, str):
            continue
        text = content
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end == -1:
            continue
        try:
            parsed = json.loads(text[start: end + 1])
            if isinstance(parsed, dict) and "direct_answer" in parsed:
                return parsed
        except (json.JSONDecodeError, ValueError):
            continue
    return {}


def _normalize_analysis(analysis: dict, business_intent: str) -> dict:
    direct_answer = analysis.get("direct_answer")
    if isinstance(direct_answer, str):
        direct_answer = {"question": business_intent, "answer": direct_answer}
    elif not isinstance(direct_answer, dict):
        direct_answer = {}

    charts = analysis.get("charts", [])
    if not isinstance(charts, list):
        charts = []

    detailed_analysis = analysis.get("detailed_analysis", "No additional analysis provided.")
    if not isinstance(detailed_analysis, str):
        detailed_analysis = json.dumps(detailed_analysis, default=str)

    return {
        **analysis,
        "direct_answer": direct_answer,
        "charts": [chart for chart in charts if isinstance(chart, dict)],
        "detailed_analysis": detailed_analysis,
    }


def _format_metric(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{value:,.2f}".rstrip("0").rstrip(".")
    return str(value)


def _build_data_driven_analysis(
    result_df: pd.DataFrame,
    business_intent: str,
    query_code: str,
) -> dict:
    row_count, column_count = result_df.shape
    numeric_columns = [
        column for column in result_df.columns
        if pd.api.types.is_numeric_dtype(result_df[column])
    ]
    category_columns = [
        column for column in result_df.columns
        if column not in numeric_columns and not pd.api.types.is_datetime64_any_dtype(result_df[column])
    ]

    evidence = [f"The query returned {row_count:,} records across {column_count:,} fields."]
    details = [
        f"The result contains {row_count:,} records with these fields: "
        f"{', '.join(map(str, result_df.columns))}."
    ]

    for column in numeric_columns[:3]:
        values = pd.to_numeric(result_df[column], errors="coerce").dropna()
        if values.empty:
            continue
        summary = (
            f"{column} totals {_format_metric(values.sum())}, averages "
            f"{_format_metric(values.mean())}, and ranges from "
            f"{_format_metric(values.min())} to {_format_metric(values.max())}."
        )
        details.append(summary)
        if len(evidence) == 1:
            evidence.append(summary)

    for column in category_columns[:2]:
        counts = result_df[column].dropna().astype(str).value_counts()
        if counts.empty:
            continue
        details.append(
            f"The most frequent {column} value is {counts.index[0]} "
            f"with {int(counts.iloc[0]):,} records."
        )

    charts = []
    if category_columns and numeric_columns:
        charts.append({
            "type": "bar",
            "title": f"{numeric_columns[0]} by {category_columns[0]}",
            "x_column": category_columns[0],
            "y_column": numeric_columns[0],
        })
    elif category_columns:
        charts.append({
            "type": "bar",
            "title": f"Records by {category_columns[0]}",
            "x_column": category_columns[0],
        })

    return {
        "direct_answer": {
            "question": business_intent,
            "answer": " ".join(evidence),
            "why": "This summary is calculated directly from the executed query result.",
            "approach": f"Executed the report query and profiled its returned data: {query_code}",
        },
        "charts": charts,
        "detailed_analysis": " ".join(details),
    }


# ---------------------------------------------------------------------------
# Tool factory
# ---------------------------------------------------------------------------

def _make_reporter_tools(gold_files: list[str], run_id: str):
    """Returns inspect + load + query tools sharing a DuckDB connection via closure."""
    conn = duckdb.connect(":memory:")
    scratchpad: dict = {}

    @tool
    def inspect_gold_tables_tool(confirmation: str = "execute") -> str:
        """Preview Gold Parquet tables before loading into DuckDB.

        Returns a JSON summary of each Gold table: table name, file path, row count,
        column names, dtypes, and 3 sample rows. Call this FIRST to understand what
        data is available and form your analytical plan.
        """
        return json.dumps(_inspect_gold_tables(gold_files), default=str)

    @tool
    def load_gold_data_tool(confirmation: str = "execute") -> str:
        """Load Gold Parquet files into DuckDB and return the full table catalog.

        Registers each Gold file as a DuckDB table and returns a catalog mapping table
        names to column names, types, row counts, and sample data — everything needed
        to write a precise SQL query. Call this before execute_query_tool.
        """
        catalog: dict = {}
        source_dfs = []
        for fp in gold_files:
            df = pd.read_parquet(fp)
            source_dfs.append(df)
            stem = Path(fp).stem.replace("-", "_").replace(" ", "_")
            conn.register(stem, df)
            catalog[stem] = {
                "table_name": stem,
                "columns": list(df.columns),
                "dtypes": {c: str(t) for c, t in df.dtypes.items()},
                "sample": df.head(5).to_dict(orient="records"),
                "row_count": len(df),
            }
        scratchpad["catalog"] = catalog
        scratchpad["source_dfs"] = source_dfs
        return json.dumps(catalog, default=str)

    @tool
    def execute_query_tool(sql_query: str) -> str:
        """Execute a SQL SELECT query against the loaded Gold tables in DuckDB.

        Call this after load_gold_data_tool. Pass your SQL as sql_query.
        Returns the query result as a JSON array of records (up to 100 rows).
        On SQL error returns {"error": "..."} — fix the SQL and retry once.
        """
        try:
            result_df = conn.execute(sql_query).fetchdf()
            result_df = _enrich_entity_labels(result_df, scratchpad.get("source_dfs", []))
            scratchpad["result_df"] = result_df
            scratchpad["sql_query"] = sql_query
            return json.dumps(result_df.head(100).to_dict(orient="records"), default=str)
        except Exception as e:
            return json.dumps({"error": str(e)})

    return inspect_gold_tables_tool, load_gold_data_tool, execute_query_tool, scratchpad, conn


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def _make_llm():
    return make_llm()


# ---------------------------------------------------------------------------
# Public entry point — I/O contract UNCHANGED
# ---------------------------------------------------------------------------

def generate_report(
    gold_files: list[str],
    business_intent: str,
    run_id: str,
    task_description: str,
) -> str:
    """Reporter AI agent entry point — autonomous ReAct version.

    The agent inspects Gold tables, plans its SQL analysis, loads tables into
    DuckDB, executes the query, and renders a self-contained HTML report.

    Args:
        gold_files: Gold Parquet file paths to analyse.
        business_intent: The business question driving the analysis.
        run_id: Unique identifier for this pipeline run.
        task_description: High-level goal message from the orchestrator.

    Returns:
        str: Path to the saved HTML report.
    """
    trace = AgentTrace("reporter", run_id)
    trace.set_input(gold_files=gold_files, business_intent=business_intent)

    print(f"[REPORTER] Starting report generation for run_id: {run_id}")
    audit = AuditLogger(run_id)
    audit.log("reporter", "started", gold_files=gold_files, intent=business_intent)

    if not gold_files:
        audit.log("reporter", "error", detail="No gold files to report on")
        trace.fail("No gold files provided")
        return ""

    inspect_tool, load_tool, query_tool, scratchpad, conn = _make_reporter_tools(gold_files, run_id)
    llm = _make_llm()

    try:
        catalog_text = load_tool.invoke({})
        catalog = json.loads(catalog_text)
        sql_prompt = (
            "Return only JSON with one key, sql_query. Write one DuckDB SELECT query "
            "that directly answers the business question. Use only exact table and column "
            "names from the catalog, quote identifiers, and never invent a name.\n"
            f"Question: {business_intent}\nCatalog: {catalog_text[:4500]}"
        )
        sql_response = llm.invoke(sql_prompt)
        sql_text = sql_response.content if isinstance(sql_response.content, str) else ""
        if "```json" in sql_text:
            sql_text = sql_text.split("```json")[1].split("```")[0]
        sql_query = json.loads(sql_text).get("sql_query", "")
        if not sql_query:
            raise RuntimeError("Reporter did not produce a SQL query")

        query_result_text = query_tool.invoke({"sql_query": sql_query})
        query_result = json.loads(query_result_text)
        if isinstance(query_result, dict) and query_result.get("error"):
            first_table = next(iter(catalog), "")
            if not first_table:
                raise RuntimeError(query_result["error"])
            query_result_text = query_tool.invoke(
                {"sql_query": f'SELECT * FROM "{first_table}" LIMIT 100'}
            )

        analysis_prompt = (
            "Return only one compact valid JSON object under 700 words with this exact shape: "
            '{"direct_answer":{"answer":"specific answer with numbers","approach":"method"},'
            '"charts":[{"type":"bar","title":"title","x_column":"exact result column",'
            '"y_column":"exact result column"}],"detailed_analysis":"evidence and insights"}. '
            "Use only actual query-result columns and at most two charts. Prefer human-readable name "
            "columns over ID columns in answers and chart labels; when both are available, present the "
            "name first and include the ID parenthetically. Do not use markdown.\n"
            f"Question: {business_intent}\nSQL: {scratchpad.get('sql_query', sql_query)}\n"
            f"Results: {query_result_text[:5000]}"
        )
        analysis_response = llm.invoke(analysis_prompt)
        analysis_result = _extract_analysis({"messages": [analysis_response]})
    except Exception as e:
        trace.fail(str(e))
        raise
    finally:
        conn.close()

    result_df: pd.DataFrame = scratchpad.get("result_df")  # type: ignore[assignment]
    query_code: str = scratchpad.get("sql_query", "-- No query executed")

    # Fallback: agent did not call execute_query_tool or query returned nothing
    if result_df is None or result_df.empty:
        print("[REPORTER] No query result in scratchpad — falling back to combined gold data")
        fallback_dfs = [pd.read_parquet(fp) for fp in gold_files]
        result_df = pd.concat(fallback_dfs, ignore_index=True) if fallback_dfs else pd.DataFrame()
        query_code = "-- Fallback: combined all Gold tables"

    # Fallback: agent response was not parseable as structured analysis
    if not analysis_result:
        analysis_result = _build_data_driven_analysis(result_df, business_intent, query_code)
    analysis_result = _normalize_analysis(analysis_result, business_intent)

    print(f"[REPORTER] Query result: {result_df.shape[0]} rows x {result_df.shape[1]} columns")

    # Generate charts from agent-specified chart specs
    charts_html = []
    for idx, chart_spec in enumerate(analysis_result.get("charts", []), 1):
        chart_html = generate_chart_from_spec(result_df, chart_spec, idx)
        if chart_html:
            charts_html.append(chart_html)
    print(f"[REPORTER] Generated {len(charts_html)} charts")

    direct_answer = analysis_result.get("direct_answer", {})
    detailed_analysis = analysis_result.get("detailed_analysis", "No additional analysis provided.")
    answer_text = html.escape(str(direct_answer.get("answer", "No answer provided")))
    approach_text = html.escape(str(direct_answer.get("approach", "No methodology provided")))
    detailed_analysis_html = html.escape(str(detailed_analysis)).replace("\n", "<br>")

    answer_html = f"""
    <div class="answer-section">
        <p>{answer_text}</p>
    </div>
    """

    query_code_escaped = query_code.replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    approach_html = f"""
    <div class="approach-section">
        <h3>Query Code</h3>
        <pre class="code-block"><code>{query_code_escaped}</code></pre>
        <h3>Query Description</h3>
        <p>{approach_text}</p>
    </div>
    """

    charts_section = "\n".join(charts_html) if charts_html else "<p>No charts generated.</p>"

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Executive Report - {run_id[:8]}</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
                color: #333;
            }}
            .header {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                border-radius: 10px;
                margin-bottom: 30px;
            }}
            .header h1 {{ margin: 0; font-size: 32px; }}
            .header p {{ margin: 10px 0 0 0; opacity: 0.9; }}
            .section {{
                background: white;
                padding: 25px;
                border-radius: 8px;
                margin-bottom: 20px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }}
            .section h2 {{
                color: #667eea;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            .answer-section {{
                background: #e8f4f8;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #28a745;
            }}
            .answer-section p {{ margin: 0; line-height: 1.6; font-size: 16px; color: #333; }}
            .approach-section {{ margin: 20px 0; }}
            .approach-section h3 {{ color: #667eea; font-size: 16px; margin: 20px 0 10px 0; }}
            .code-block {{
                background: #f5f5f5;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 15px;
                overflow-x: auto;
                font-family: 'Courier New', monospace;
                font-size: 13px;
                line-height: 1.4;
                color: #333;
                margin: 0 0 15px 0;
            }}
            .code-block code {{ color: #667eea; }}
            .approach-section p {{ line-height: 1.6; color: #555; margin: 0 0 15px 0; }}
            .chart-container {{ margin: 20px 0; }}
            .footer {{
                text-align: center;
                color: #999;
                margin-top: 40px;
                padding-top: 20px;
                border-top: 1px solid #eee;
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>&#128202; Executive Report</h1>
            <p><strong>Business Question:</strong> {business_intent}</p>
        </div>
        <div class="section">
            <h2>&#9989; Answer</h2>
            {answer_html}
        </div>
        <div class="section">
            <h2>&#128202; Approach &amp; Query</h2>
            {approach_html}
        </div>
        <div class="section">
            <h2>&#128201; Visual Evidence</h2>
            <div class="chart-container">
                {charts_section}
            </div>
        </div>
        <div class="section">
            <h2>Detailed Analysis</h2>
            <p>{detailed_analysis_html}</p>
        </div>
        <div class="footer">
            <p>Generated by IDAMP (Intent-Driven Agentic Medallion Pipeline)</p>
            <p>Report Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
    </body>
    </html>
    """

    report_path = str(REPORTS_DIR / f"report_{run_id[:8]}.html")
    print(f"[REPORTER] Saving HTML report → {report_path}")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    json_path = str(REPORTS_DIR / f"report_{run_id[:8]}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(analysis_result, f, indent=2)

    store_document(
        doc_id=f"report_{run_id}",
        text=json.dumps(analysis_result),
        metadata={"type": "report", "run_id": run_id, "intent": business_intent},
    )

    audit.log("reporter", "completed", report_path=report_path)
    trace.set_output(report_path=report_path).complete()
    print(f"[REPORTER] Done — {report_path}")
    return report_path
