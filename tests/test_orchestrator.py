from pathlib import Path
from unittest.mock import MagicMock, patch

from agents.orchestrator import (
    run_bronze_to_silver_sttm,
    run_gold_and_report,
    run_silver_to_gold_sttm,
    run_until_bronze_sttm,
)


def _state(tmp_path: Path) -> dict:
    return {
        "run_id": "run-test",
        "status": "ready",
        "uploaded_files": [str(tmp_path / "sales.csv")],
        "business_intent": "Summarize sales",
        "profile_path": str(tmp_path / "profile.json"),
        "sttm_bronze_path": str(tmp_path / "bronze.csv"),
        "sttm_silver_path": str(tmp_path / "silver.csv"),
        "sttm_gold_path": str(tmp_path / "gold.csv"),
        "bronze_sttm_approved": False,
        "silver_sttm_approved": False,
        "gold_sttm_approved": False,
        "bronze_output_paths": [str(tmp_path / "bronze.parquet")],
        "silver_output_paths": [str(tmp_path / "silver.parquet")],
        "gold_output_paths": [],
        "report_path": "",
        "error": "",
    }


def test_phase1_sequences_profiler_then_bronze_sttm_without_supervisor(tmp_path):
    profile_path = tmp_path / "profile.json"
    bronze_sttm_path = tmp_path / "bronze.csv"
    profile_path.touch()
    bronze_sttm_path.touch()

    with (
        patch("agents.orchestrator.AuditLogger", return_value=MagicMock()),
        patch("agents.orchestrator.store_document"),
        patch("agents.orchestrator._run_supervisor", side_effect=AssertionError("supervisor called")),
        patch("agents.orchestrator.profile_multiple_datasets", return_value=str(profile_path)) as profile,
        patch("agents.orchestrator.generate_bronze_sttm", return_value=str(bronze_sttm_path)) as sttm,
    ):
        state = run_until_bronze_sttm(["sales.csv"], "Summarize sales")

    profile.assert_called_once()
    sttm.assert_called_once()
    assert state["status"] == "awaiting_bronze_sttm_approval"
    assert state["sttm_bronze_path"] == str(bronze_sttm_path)


def test_phase2_sequences_bronze_then_silver_sttm_without_supervisor(tmp_path):
    state = _state(tmp_path)
    silver_sttm_path = tmp_path / "generated_silver.csv"

    with (
        patch("agents.orchestrator.AuditLogger", return_value=MagicMock()),
        patch("agents.orchestrator._run_supervisor", side_effect=AssertionError("supervisor called")),
        patch("agents.orchestrator.execute_bronze", return_value=["bronze.parquet"]) as bronze,
        patch("agents.orchestrator.generate_silver_sttm", return_value=str(silver_sttm_path)) as sttm,
    ):
        result = run_bronze_to_silver_sttm(state)

    bronze.assert_called_once()
    sttm.assert_called_once()
    assert result["status"] == "awaiting_silver_sttm_approval"
    assert result["sttm_silver_path"] == str(silver_sttm_path)


def test_phase3_sequences_silver_then_gold_sttm_without_supervisor(tmp_path):
    state = _state(tmp_path)
    gold_sttm_path = tmp_path / "generated_gold.csv"

    with (
        patch("agents.orchestrator.AuditLogger", return_value=MagicMock()),
        patch("agents.orchestrator._run_supervisor", side_effect=AssertionError("supervisor called")),
        patch("agents.orchestrator.execute_silver", return_value=["silver.parquet"]) as silver,
        patch("agents.orchestrator.generate_gold_sttm", return_value=str(gold_sttm_path)) as sttm,
    ):
        result = run_silver_to_gold_sttm(state)

    silver.assert_called_once()
    sttm.assert_called_once()
    assert result["status"] == "awaiting_gold_sttm_approval"
    assert result["sttm_gold_path"] == str(gold_sttm_path)


def test_phase4_sequences_gold_then_report_without_supervisor(tmp_path):
    state = _state(tmp_path)
    gold_path = tmp_path / "gold.parquet"
    report_path = tmp_path / "report.html"
    gold_path.touch()
    report_path.touch()

    with (
        patch("agents.orchestrator.AuditLogger", return_value=MagicMock()),
        patch("agents.orchestrator._run_supervisor", side_effect=AssertionError("supervisor called")),
        patch("agents.orchestrator.execute_gold", return_value=[str(gold_path)]) as gold,
        patch("agents.orchestrator.generate_report", return_value=str(report_path)) as report,
    ):
        result = run_gold_and_report(state)

    gold.assert_called_once()
    report.assert_called_once()
    assert result["status"] == "completed"
    assert result["report_path"] == str(report_path)


def test_phase4_rejects_empty_gold_output_before_reporter(tmp_path):
    state = _state(tmp_path)

    with (
        patch("agents.orchestrator.AuditLogger", return_value=MagicMock()),
        patch("agents.orchestrator.execute_gold", return_value=[]),
        patch("agents.orchestrator.generate_report") as report,
    ):
        result = run_gold_and_report(state)

    report.assert_not_called()
    assert result["status"] == "failed"
    assert "no output files" in result["error"].lower()
