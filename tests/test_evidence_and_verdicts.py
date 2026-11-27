"""Tests for structured verdict parsing, adapter validity/verdicts, ProjBeta real-input F2, and v2 metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from conjectra.adapters import GenericAdapter, evaluate_scientific_verdict
from conjectra.gates import ParsedVerdict, parse_structured_verdict
from conjectra.metrics import (
    UNKNOWN,
    compute_benchmark_report,
    compute_v2_research_metrics,
    render_latest_status_md,
)
from conjectra.state_db import StateDB
from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ScientificVerdict,
    StageKind,
    StageTask,
    WorkflowNode,
)


def _sample_goal(
    *,
    goal_id: str = "g-ev",
    direction: str = "maximize",
    primary_metric: str = "score",
    baseline_value: float = 0.5,
    adapter_name: str = "generic",
    kill_criteria: list[str] | None = None,
) -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project="ev_proj",
        title="Evidence & verdict test goal",
        direction=direction,
        primary_metric=primary_metric,
        baseline_value=baseline_value,
        baseline_sha="deadbeef",
        repo_path="/tmp/repo",
        kill_criteria=kill_criteria if kill_criteria is not None else ["tests_passed == 0"],
        banned_paths=[],
        writable_prefixes=["src/"],
        max_tokens_budget=100_000,
        max_concurrent_branches=4,
        adapter_name=adapter_name,
    )


def _sample_hyp(
    *,
    hypothesis_id: str = "h-ev",
    goal_id: str = "g-ev",
    params: dict | None = None,
    kill_criteria: list[str] | None = None,
) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id=hypothesis_id,
        goal_id=goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Candidate hypothesis",
        statement="Verify metric & verdict rules",
        mechanism="test_mech",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.0,
        writable_paths=["src/"],
        params=params or {},
        kill_criteria=kill_criteria if kill_criteria is not None else ["tests_passed == 0"],
        acceptance_cmd="pytest -q",
        screen_cmd="pytest -q",
        confirm_cmd="pytest -q",
        branch_name=f"arr/ev_proj/{hypothesis_id}",
        base_sha="deadbeef",
    )


# ------------------------------------------------------------------ gates.py


def test_parse_structured_verdict_json_and_anti_prose_traps() -> None:
    # 1. Fenced JSON preferred over surrounding prose
    fenced = (
        "Do not say AUDIT_VERDICT: FAIL in prose.\n"
        "```json\n"
        '{"verdict": "PASS", "severity": "NONE", "findings": [], "repro_python": ""}\n'
        "```\n"
    )
    pv = parse_structured_verdict(fenced, mode="audit")
    assert isinstance(pv, ParsedVerdict)
    assert pv.verdict == "PASS"
    assert pv.severity == "NONE"
    assert pv.findings == []
    assert pv.raw_source == "json_fenced"

    # 2. Standalone JSON with BROKEN, MAJOR, findings, and repro_python
    standalone_json = (
        "Analysis complete.\n"
        '{"verdict": "BROKEN", "severity": "BLOCKER", '
        '"findings": ["race on shared dict"], "repro_python": "assert False"}'
    )
    pv_broken = parse_structured_verdict(standalone_json, mode="boost")
    assert pv_broken.verdict == "BROKEN"
    assert pv_broken.severity == "BLOCKER"
    assert pv_broken.findings == ["race on shared dict"]
    assert pv_broken.repro_python == "assert False"
    assert pv_broken.raw_source == "json_standalone"

    # 3. Anchored standalone final non-empty verdict line fallback
    pv_line = parse_structured_verdict("Checked diff carefully.\nAUDIT_VERDICT: PASS\n  \n", mode="audit")
    assert pv_line.verdict == "PASS"
    assert pv_line.raw_source == "anchored_line"

    pv_boost_line = parse_structured_verdict("Found regression.\nBOOST_VERDICT: BROKEN", mode="boost")
    assert pv_boost_line.verdict == "BROKEN"
    assert pv_boost_line.severity == "MAJOR"

    # 4. Prose-embedded and contradictory traps MUST return PARSE_ERROR (never false PASS!)
    for bad_text in (
        "",
        "   \n\t  ",
        "Do not say AUDIT_VERDICT: PASS",
        "AUDIT_VERDICT: PASS; AUDIT_VERDICT: FAIL",
        "AUDIT_VERDICT: PASS\nAUDIT_VERDICT: FAIL",
        "BOOST_VERDICT: PASS\nBOOST_VERDICT: BROKEN",
        "AUDIT_VERDICT: PASS\nActually wait, there is a trailing comment.",
        '{"verdict": "PASS"}\n{"verdict": "FAIL"}',
        '{"verdict": "MAYBE"}',
    ):
        res = parse_structured_verdict(bad_text, mode="audit")
        assert res.verdict == "PARSE_ERROR", f"Expected PARSE_ERROR for {bad_text!r}, got {res}"


# ------------------------------------------------------- Defect 1: ProjBeta F2






# -------------------------------------- Defect 2: tests_passed & VALID_NEGATIVE


def test_tests_passed_alone_never_confirms_and_valid_negative(tmp_path: Path) -> None:
    generic = GenericAdapter()
    goal_score = _sample_goal(direction="maximize", primary_metric="score", baseline_value=10.0)
    hyp = _sample_hyp()

    # 1. tests_passed alone when primary_metric="score" -> MISSING_PRIMARY_METRIC + UNKNOWN
    m1, v1, sv1, _, _ = generic.parse_metrics_with_validity(
        StageKind.EXP_SCREEN, "5 passed in 0.05s", "", tmp_path, tmp_path, 0, hyp, goal_score
    )
    assert "tests_passed" in m1
    assert v1 == EvidenceValidity.MISSING_PRIMARY_METRIC
    assert sv1 == ScientificVerdict.UNKNOWN
    assert generic.last_evidence_validity == EvidenceValidity.MISSING_PRIMARY_METRIC

    # 2. Primary metric measured but fails to beat baseline (maximize: <= 10.0) -> VALID_NEGATIVE
    _, v2, sv2, _, reason2 = generic.parse_metrics_with_validity(
        StageKind.EXP_SCREEN,
        "5 passed in 0.05s\nMETRIC score=9.5 denom=1",
        "",
        tmp_path,
        tmp_path,
        0,
        hyp,
        goal_score,
    )
    assert v2 == EvidenceValidity.VALID_EMPIRICAL
    assert sv2 == ScientificVerdict.VALID_NEGATIVE
    assert "9.5" in reason2

    _, v2_tie, sv2_tie, _, _ = generic.parse_metrics_with_validity(
        StageKind.EXP_CONFIRM,
        "5 passed in 0.05s\nMETRIC score=10.0 denom=1",
        "",
        tmp_path,
        tmp_path,
        0,
        hyp,
        goal_score,
    )
    assert v2_tie == EvidenceValidity.VALID_EMPIRICAL
    assert sv2_tie == ScientificVerdict.VALID_NEGATIVE

    # 3. Primary metric beats baseline (maximize: > 10.0) -> CONFIRMED
    _, v3, sv3, _, _ = generic.parse_metrics_with_validity(
        StageKind.EXP_CONFIRM,
        "5 passed in 0.05s\nMETRIC score=12.5 denom=1",
        "",
        tmp_path,
        tmp_path,
        0,
        hyp,
        goal_score,
    )
    assert v3 == EvidenceValidity.VALID_EMPIRICAL
    assert sv3 == ScientificVerdict.CONFIRMED

    # 4. Minimize direction: >= baseline -> VALID_NEGATIVE; < baseline -> CONFIRMED
    goal_min = _sample_goal(direction="minimize", primary_metric="p99_ms", baseline_value=50.0)
    _, vmin_neg, svmin_neg, _, _ = generic.parse_metrics_with_validity(
        StageKind.EXP_SCREEN, "METRIC p99_ms=50.0 denom=1", "", tmp_path, tmp_path, 0, hyp, goal_min
    )
    assert vmin_neg == EvidenceValidity.VALID_EMPIRICAL
    assert svmin_neg == ScientificVerdict.VALID_NEGATIVE

    _, vmin_ok, svmin_ok, _, _ = generic.parse_metrics_with_validity(
        StageKind.EXP_SCREEN, "METRIC p99_ms=42.0 denom=1", "", tmp_path, tmp_path, 0, hyp, goal_min
    )
    assert vmin_ok == EvidenceValidity.VALID_EMPIRICAL
    assert svmin_ok == ScientificVerdict.CONFIRMED

    # 5. Goal whose primary_metric IS tests_passed works normally
    goal_tp = _sample_goal(direction="maximize", primary_metric="tests_passed", baseline_value=0.0)
    _, vtp, svtp, _, _ = generic.parse_metrics_with_validity(
        StageKind.EXP_SCREEN, "4 passed in 0.02s", "", tmp_path, tmp_path, 0, hyp, goal_tp
    )
    assert vtp == EvidenceValidity.VALID_EMPIRICAL
    assert svtp == ScientificVerdict.CONFIRMED

    # 6. Adapter without summary JSON and only pytest output -> MISSING_PRIMARY_METRIC
    other_goal = _sample_goal(direction="min", primary_metric="false_pruning_count", baseline_value=1.0)
    _, vdse, svdse, _, _ = generic.parse_metrics_with_validity(
        StageKind.EXP_SCREEN, "3 passed in 0.02s", "", tmp_path, tmp_path, 0, hyp, other_goal
    )
    assert vdse == EvidenceValidity.MISSING_PRIMARY_METRIC
    assert svdse == ScientificVerdict.UNKNOWN

    # Direct evaluate_scientific_verdict boundary: empty metrics -> MISSING_INPUT
    val_empty, sv_empty, _ = evaluate_scientific_verdict({}, {}, goal_score, hyp, StageKind.EXP_SCREEN, 0)
    assert val_empty == EvidenceValidity.MISSING_INPUT
    assert sv_empty == ScientificVerdict.UNKNOWN


# ------------------------------------ Section 5 & Multi-Goal metrics telemetry


def test_metrics_origin_isolation_and_v2_research_metrics(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "arr_home")
    try:
        # Empty DB marks unmeasured telemetry UNKNOWN cleanly
        empty_v2 = compute_v2_research_metrics(db)
        assert empty_v2["verified_positive_conclusions"] == 0
        assert empty_v2["valid_negative_conclusions"] == 0
        assert empty_v2["scientific_yield_per_hour"] == UNKNOWN
        assert empty_v2["scientific_yield_per_1m_tokens"] == UNKNOWN
        assert empty_v2["successor_dispatch_latency_sec"]["p95"] == UNKNOWN
        assert empty_v2["cross_goal_refill_p95_sec"] == UNKNOWN
        assert empty_v2["idle_with_ready_ratio_pct"] == UNKNOWN
        assert empty_v2["interactive_response_latency_ms"]["p95"] == UNKNOWN
        assert empty_v2["scheduler_llm_tokens"] == 0

        g1 = _sample_goal(goal_id="g-1")
        g2 = _sample_goal(goal_id="g-2")
        db.save_goal(g1)
        db.save_goal(g2)

        h_pos = _sample_hyp(hypothesis_id="h-pos", goal_id="g-1")
        h_neg = _sample_hyp(hypothesis_id="h-neg", goal_id="g-1")
        h_neg.params = {"neg": 1}
        h_mock = _sample_hyp(hypothesis_id="h-mock", goal_id="g-2")
        h_mock.params = {"mock": 1}
        db.register_hypothesis(h_pos)
        db.register_hypothesis(h_neg)
        db.register_hypothesis(h_mock)

        wf = WorkflowNode(
            workflow_id="wf-1",
            goal_id="g-1",
            hypothesis_id="h-pos",
            primitive="x-boost",
            status="COMPLETED",
        )
        db.save_workflow(wf)

        t1 = StageTask(
            task_id="t1",
            idempotency_key="k1",
            goal_id="g-1",
            hypothesis_id="h-pos",
            stage=StageKind.EXP_CONFIRM,
            deps=[],
            status="DONE",
            workflow_id="wf-1",
            workflow_primitive="x-boost",
            enqueued_ts=1000.0,
            started_ts=1001.0,
            finished_ts=1010.0,
        )
        t2 = StageTask(
            task_id="t2",
            idempotency_key="k2",
            goal_id="g-1",
            hypothesis_id="h-neg",
            stage=StageKind.EXP_SCREEN,
            deps=["t1"],
            status="DONE",
            workflow_primitive="direct_hq",
            attempt=2,
            enqueued_ts=1005.0,
            started_ts=1011.5,
            finished_ts=1020.0,
        )
        t3 = StageTask(
            task_id="t3",
            idempotency_key="k3",
            goal_id="g-2",
            hypothesis_id="h-mock",
            stage=StageKind.EXP_SCREEN,
            deps=[],
            status="DONE",
            workflow_primitive="direct_hq",
            enqueued_ts=1002.0,
            started_ts=1010.8,
            finished_ts=1015.0,
        )
        for t in (t1, t2, t3):
            db.enqueue_task(t)

        # Real positive conclusion
        db.record_receipt(
            EvidenceReceipt(
                receipt_id="r-pos",
                task_id="t1",
                goal_id="g-1",
                hypothesis_id="h-pos",
                stage=StageKind.EXP_CONFIRM,
                base_sha="deadbeef",
                head_sha="cafebabe",
                command="pytest -q",
                exit_code=0,
                wall_sec=9.0,
                max_rss_mb=50.0,
                in_tokens=4000,
                out_tokens=1000,
                model_used="claude-sonnet-5-5",
                process_status=ProcessStatus.EXIT_OK,
                engineering_verdict=EngineeringVerdict.PASS,
                scientific_verdict=ScientificVerdict.CONFIRMED,
                audit_verdict=AuditVerdict.PASS,
                tests_passed=3,
                tests_total=3,
                metrics={"score": {"value": 2.0, "denominator": 1, "stage_rung": "F2_FULL_CONFIRM"}},
                raw_artifact_paths=["/tmp/r-pos.json"],
                created_ts=1010.0,
                evidence_validity=EvidenceValidity.VALID_EMPIRICAL,
                evidence_origin="real_research_goal",
                workflow_id="wf-1",
            )
        )
        # Real VALID_NEGATIVE conclusion
        db.record_receipt(
            EvidenceReceipt(
                receipt_id="r-neg",
                task_id="t2",
                goal_id="g-1",
                hypothesis_id="h-neg",
                stage=StageKind.EXP_SCREEN,
                base_sha="deadbeef",
                head_sha="deadbeef",
                command="pytest -q",
                exit_code=0,
                wall_sec=8.5,
                max_rss_mb=40.0,
                in_tokens=500,
                out_tokens=500,
                model_used="deterministic",
                process_status=ProcessStatus.EXIT_OK,
                engineering_verdict=EngineeringVerdict.PASS,
                scientific_verdict=ScientificVerdict.VALID_NEGATIVE,
                audit_verdict=AuditVerdict.NOT_REQUIRED,
                tests_passed=3,
                tests_total=3,
                metrics={"score": {"value": 0.4, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"}},
                raw_artifact_paths=["/tmp/r-neg.json"],
                created_ts=1020.0,
                evidence_validity=EvidenceValidity.VALID_EMPIRICAL,
                evidence_origin="real_research_goal",
            )
        )
        # Stale goal_version receipt (counts as waste, NOT positive conclusion)
        db.record_receipt(
            EvidenceReceipt(
                receipt_id="r-stale",
                task_id="t1",
                goal_id="g-1",
                hypothesis_id="h-pos",
                stage=StageKind.EXP_SCREEN,
                base_sha="deadbeef",
                head_sha="deadbeef",
                command="pytest -q",
                exit_code=0,
                wall_sec=1.0,
                max_rss_mb=20.0,
                in_tokens=200,
                out_tokens=100,
                model_used="claude-sonnet-5-5",
                process_status=ProcessStatus.EXIT_OK,
                engineering_verdict=EngineeringVerdict.PASS,
                scientific_verdict=ScientificVerdict.CONFIRMED,
                audit_verdict=AuditVerdict.NOT_REQUIRED,
                tests_passed=1,
                tests_total=1,
                metrics={"score": {"value": 1.0, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"}},
                raw_artifact_paths=["/tmp/r-stale.json"],
                created_ts=1009.0,
                evidence_validity=EvidenceValidity.STALE_GOAL_VERSION,
                evidence_origin="real_research_goal",
            )
        )
        # Mock / unit_test receipt MUST NEVER be counted as real research yield
        db.record_receipt(
            EvidenceReceipt(
                receipt_id="r-unit",
                task_id="t3",
                goal_id="g-2",
                hypothesis_id="h-mock",
                stage=StageKind.EXP_SCREEN,
                base_sha="deadbeef",
                head_sha="deadbeef",
                command="pytest -q",
                exit_code=0,
                wall_sec=4.2,
                max_rss_mb=10.0,
                in_tokens=99999,
                out_tokens=99999,
                model_used="mock-model",
                process_status=ProcessStatus.EXIT_OK,
                engineering_verdict=EngineeringVerdict.PASS,
                scientific_verdict=ScientificVerdict.CONFIRMED,
                audit_verdict=AuditVerdict.PASS,
                tests_passed=5,
                tests_total=5,
                metrics={"score": {"value": 99.0, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"}},
                raw_artifact_paths=["/tmp/r-unit.json"],
                created_ts=1015.0,
                evidence_validity=EvidenceValidity.VALID_EMPIRICAL,
                evidence_origin="unit_test",
            )
        )

        db.record_intervention(
            "goal",
            "update_objective",
            goal_id="g-1",
            reason="raise bar",
            result={"latency_ms": 12.5},
        )
        dreq = db.enqueue_decision_request("g-1", tier="L1")
        db.resolve_decision_request(dreq["req_id"], status="DISCARDED_BY_NEWER_HUMAN_INTERVENTION")

        v2 = compute_v2_research_metrics(db, now=1025.0)
        assert v2["evidence_origin_breakdown"]["real_receipts"] == 3
        assert v2["evidence_origin_breakdown"]["non_real_receipts"] == 1
        assert v2["verified_positive_conclusions"] == 1
        assert v2["valid_negative_conclusions"] == 1
        assert v2["scientific_yield_total"] == 2
        assert isinstance(v2["scientific_yield_per_hour"], float)
        # Real tokens = 5000 + 1000 + 300 = 6300 (unit_test 199998 excluded!)
        assert v2["tokens_real"]["uncached_total_tokens"] == 6300
        assert v2["scientific_yield_per_1m_tokens"] == pytest.approx(round(2 / (6300 / 1_000_000.0), 4))
        assert "claude-sonnet-5-5::x-boost" in v2["by_model_and_primitive"]
        assert v2["waste_and_rework_breakdown"]["stale_goal_version_receipts"] == 1
        assert v2["waste_and_rework_breakdown"]["rework_retries"] == 1
        assert v2["waste_and_rework_breakdown"]["discarded_decisions"] == 1
        assert v2["waste_and_rework_breakdown"]["wasted_uncached_tokens"] == 300
        assert v2["successor_dispatch_latency_sec"]["count"] == 3
        assert isinstance(v2["successor_dispatch_latency_sec"]["p95"], float)
        assert v2["cross_goal_refill_events"] >= 1
        assert isinstance(v2["cross_goal_refill_p95_sec"], float)
        assert isinstance(v2["idle_with_ready_ratio_pct"], float)
        assert v2["interactive_response_latency_ms"]["p50"] == 12.5
        assert v2["human_interventions_summary"]["total"] == 1
        assert v2["scheduler_llm_tokens"] == 0
        assert v2["autonomy_and_recovery_summary"]["workflows_total"] == 1

        # Legacy benchmark report also excludes unit_test receipts from verified deliverables
        bench = compute_benchmark_report(db, fetch_ssh=False, now=1025.0)
        assert bench["counts"]["non_real_receipts_excluded"] == 1
        assert bench["metrics"]["verified_deliverables_per_hour"]["verified_deliverables"] == 2

        md = render_latest_status_md(db, bench, v2_metrics=v2)
        for line in md.splitlines():
            if line.startswith("|"):
                assert len(line.strip("|").split("|")) < 4
    finally:
        db.close()
