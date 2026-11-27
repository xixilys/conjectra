"""Tests for types backward-compat, StateDB v2 tables, and GlobalGoalManager HITL."""

from __future__ import annotations

import json
import math

import pytest

from conjectra.goal_manager import (
    DEFAULT_FOREGROUND_RESERVE_SLOTS,
    DEFAULT_GLOBAL_MAX_TOKENS_BUDGET,
    GlobalGoalManager,
)
from conjectra.state_db import StateDB, StateTransitionError
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
    ResourceSpec,
    ScientificVerdict,
    StageKind,
    StageTask,
    WorkflowNode,
)


class FakeHQ:
    """Minimal stub recording HQ job cancellations."""

    def __init__(self) -> None:
        self.cancelled: list[tuple[list[int], str | None]] = []

    def cancel_jobs(self, job_ids, reason: str | None = None) -> dict[int, bool]:
        ids = [int(j) for j in job_ids]
        self.cancelled.append((ids, reason))
        return {jid: True for jid in ids}


def _make_goal(
    goal_id: str = "goal-a",
    *,
    primary_metric: str = "p99_ms",
    direction: str = "minimize",
    baseline_value: float = 100.0,
    priority_weight: float = 1.0,
    min_guaranteed_slots: int = 1,
    max_tokens_budget: int = 100_000,
    kill_criteria: list[str] | None = None,
) -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project="arr-test",
        title=f"Optimize {goal_id}",
        direction=direction,
        primary_metric=primary_metric,
        baseline_value=baseline_value,
        baseline_sha="abc1234",
        repo_path="/tmp/repo",
        kill_criteria=list(kill_criteria) if kill_criteria is not None else ["p99_ms > 120"],
        banned_paths=["secret_holdout/"],
        writable_prefixes=["src/", "tests/"],
        max_tokens_budget=max_tokens_budget,
        max_concurrent_branches=4,
        adapter_name="generic",
        priority_weight=priority_weight,
        min_guaranteed_slots=min_guaranteed_slots,
    )


def _make_hyp(
    hyp_id: str,
    goal_id: str = "goal-a",
    *,
    parent_id: str | None = None,
    mechanism: str = "cache-tiling",
    status: str = "ACTIVE",
    scientific_verdict: ScientificVerdict = ScientificVerdict.UNKNOWN,
) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id=hyp_id,
        goal_id=goal_id,
        parent_id=parent_id,
        kind=HypothesisKind.COMPETING_ROUTE,
        title=f"Hyp {hyp_id}",
        statement="Improve metric",
        mechanism=mechanism,
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.8,
        writable_paths=["src/"],
        params={"tile": 32},
        kill_criteria=["p99_ms > 120"],
        acceptance_cmd="pytest -q",
        screen_cmd="python3 bench.py",
        confirm_cmd="python3 bench.py --full",
        branch_name=f"arr/arr-test/{hyp_id}",
        base_sha="abc1234",
        status=status,
        scientific_verdict=scientific_verdict,
    )


def _make_task(
    task_id: str,
    goal_id: str,
    hyp_id: str,
    *,
    stage: StageKind = StageKind.EXP_SCREEN,
    status: str = "READY",
    hq_job_id: int | None = None,
    workflow_id: str | None = None,
    goal_version: int = 1,
) -> StageTask:
    return StageTask(
        task_id=task_id,
        idempotency_key=f"{goal_id}:{hyp_id}:{stage.value}:{task_id}",
        goal_id=goal_id,
        hypothesis_id=hyp_id,
        stage=stage,
        deps=[],
        resources=ResourceSpec(cpus=1),
        hq_job_id=hq_job_id,
        status=status,
        workflow_id=workflow_id,
        goal_version=goal_version,
    )


def _make_receipt(
    receipt_id: str,
    task_id: str,
    goal_id: str,
    hyp_id: str,
    *,
    stage: StageKind = StageKind.EXP_SCREEN,
    scientific_verdict: ScientificVerdict = ScientificVerdict.CONFIRMED,
    evidence_validity: EvidenceValidity = EvidenceValidity.VALID_EMPIRICAL,
    metrics: dict | None = None,
    goal_version: int = 1,
    workflow_id: str | None = None,
) -> EvidenceReceipt:
    return EvidenceReceipt(
        receipt_id=receipt_id,
        task_id=task_id,
        goal_id=goal_id,
        hypothesis_id=hyp_id,
        stage=stage,
        base_sha="abc1234",
        head_sha="def5678",
        command="python3 bench.py",
        exit_code=0,
        wall_sec=1.5,
        max_rss_mb=64.0,
        in_tokens=100,
        out_tokens=50,
        model_used="none",
        process_status=ProcessStatus.EXIT_OK,
        engineering_verdict=EngineeringVerdict.PASS,
        scientific_verdict=scientific_verdict,
        audit_verdict=AuditVerdict.NOT_REQUIRED,
        tests_passed=3,
        tests_total=3,
        metrics=metrics
        if metrics is not None
        else {"p99_ms": {"value": 85.0, "denominator": 100, "stage_rung": "F1_FAST_SCREEN"}},
        raw_artifact_paths=["/tmp/bench.json"],
        evidence_validity=evidence_validity,
        evidence_origin="real_research_goal",
        goal_version=goal_version,
        workflow_id=workflow_id,
    )


def test_types_backward_compatibility_and_workflow_node_roundtrip():
    legacy_goal = {
        "goal_id": "g-legacy",
        "project": "proj",
        "title": "Legacy",
        "direction": "minimize",
        "primary_metric": "loss",
        "baseline_value": 1.0,
        "baseline_sha": "1111111",
        "repo_path": "/tmp/r",
        "kill_criteria": [],
        "banned_paths": [],
        "writable_prefixes": ["src/"],
        "max_tokens_budget": 5000,
    }
    g = ResearchGoal.from_dict(legacy_goal)
    assert g.version == 1
    assert g.human_epoch == 0
    assert g.autonomy_mode == "unattended"
    assert g.priority_weight == 1.0
    assert g.min_guaranteed_slots == 1
    assert g.target_value is None
    assert g.wall_budget_sec == 21600
    assert ResearchGoal.from_dict(g.to_dict()) == g

    legacy_task = {
        "task_id": "t-legacy",
        "idempotency_key": "k1",
        "goal_id": "g-legacy",
        "hypothesis_id": "h1",
        "stage": "CODE",
        "deps": [],
    }
    t = StageTask.from_dict(legacy_task)
    assert t.workflow_id is None
    assert t.goal_version == 1
    assert t.workflow_primitive == "direct_hq"
    assert t.step_key == ""
    assert StageTask.from_dict(t.to_dict()) == t

    legacy_rcpt = {
        "receipt_id": "r-legacy",
        "task_id": "t-legacy",
        "goal_id": "g-legacy",
        "hypothesis_id": "h1",
        "stage": "EXP_SCREEN",
        "base_sha": "1111111",
        "head_sha": "2222222",
        "command": "pytest",
        "exit_code": 0,
        "wall_sec": 1.0,
        "max_rss_mb": 32.0,
        "in_tokens": 0,
        "out_tokens": 0,
        "model_used": "none",
        "process_status": "EXIT_OK",
        "engineering_verdict": "PASS",
        "scientific_verdict": "VALID_NEGATIVE",
        "audit_verdict": "PARSE_ERROR",
        "tests_passed": 1,
        "tests_total": 1,
    }
    r = EvidenceReceipt.from_dict(legacy_rcpt)
    assert r.evidence_validity == EvidenceValidity.VALID_ENGINEERING_ONLY
    assert r.evidence_origin == "real_research_goal"
    assert r.goal_version == 1
    assert r.workflow_id is None
    assert r.scientific_verdict == ScientificVerdict.VALID_NEGATIVE
    assert r.audit_verdict == AuditVerdict.PARSE_ERROR
    assert EvidenceReceipt.from_dict(r.to_dict()) == r

    wf = WorkflowNode(
        workflow_id="wf-1",
        goal_id="g-legacy",
        hypothesis_id="h1",
        parent_workflow_id="wf-root",
        primitive="x-boost",
        status="RUNNING",
        goal_version=2,
        budget_tokens=25000,
        config={"max_repairs": 2},
        state={"step": "adversary", "round": 1},
        created_ts=10.0,
        updated_ts=20.0,
    )
    assert WorkflowNode.from_dict(json.loads(json.dumps(wf.to_dict()))) == wf


def test_state_db_v2_tables_and_crud(tmp_path):
    with StateDB(tmp_path) as db:
        # Global config
        assert db.get_global_config() == {}
        assert db.get_global_config("missing", 42) == 42
        db.set_global_config("global_max_tokens_budget", 900_000, now=100.0)
        db.set_global_config("foreground_reserve_slots", 2, now=101.0)
        assert db.get_global_config("global_max_tokens_budget") == 900_000
        assert db.get_global_config() == {
            "foreground_reserve_slots": 2,
            "global_max_tokens_budget": 900_000,
        }

        # Goals status preservation & filtering
        g1 = _make_goal("g1")
        db.save_goal(g1, status="PAUSED")
        g1.title = "Updated title"
        db.save_goal(g1)  # status=None preserves PAUSED
        assert db.get_goal_status("g1") == "PAUSED"
        assert len(db.list_goals(status="PAUSED")) == 1
        assert len(db.list_goals(status="ACTIVE")) == 0

        # Workflows CRUD
        wf_root = WorkflowNode(
            workflow_id="wf-root",
            goal_id="g1",
            hypothesis_id="h1",
            primitive="x-teamwork",
            status="READY",
            config={"slices": 2},
        )
        saved_root = db.save_workflow(wf_root, now=200.0)
        assert saved_root.created_ts == 200.0
        wf_child = WorkflowNode(
            workflow_id="wf-child",
            goal_id="g1",
            hypothesis_id="h1",
            parent_workflow_id="wf-root",
            primitive="x-boost",
        )
        db.save_workflow(wf_child, now=201.0)
        assert db.get_workflow("wf-child").parent_workflow_id == "wf-root"
        assert [w.workflow_id for w in db.list_workflows(goal_id="g1")] == ["wf-root", "wf-child"]
        assert [w.workflow_id for w in db.list_workflows(hypothesis_id="h1", parent_workflow_id="wf-root")] == ["wf-child"]

        upd = db.update_workflow_state(
            "wf-root",
            status="RUNNING",
            state_patch={"milestone": 1},
            expected_statuses="READY",
            now=205.0,
        )
        assert upd.status == "RUNNING"
        assert upd.state == {"milestone": 1}
        with pytest.raises(StateTransitionError):
            db.update_workflow_state("wf-root", status="DONE", expected_statuses="READY")
        with pytest.raises(KeyError):
            db.update_workflow_state("wf-missing", status="DONE")

        # Goal versions
        db.save_goal_version(g1, change_summary="v1", actor="user", reason="init", now=300.0)
        g1_v2 = ResearchGoal.from_dict({**g1.to_dict(), "version": 2, "human_epoch": 1})
        db.save_goal_version(g1_v2, change_summary="v2", actor="user", reason="tighten", now=310.0)
        versions = db.list_goal_versions("g1")
        assert [v["version"] for v in versions] == [1, 2]

        # Goal knowledge upsert
        k1 = db.upsert_knowledge_item("g1", "lesson", "tile_32", {"gain": 0.12}, hypothesis_id="h1", now=400.0)
        k2 = db.upsert_knowledge_item("g1", "lesson", "tile_32", {"gain": 0.15}, hypothesis_id="h1", now=405.0)
        assert k2["item_id"] == k1["item_id"]
        assert db.list_knowledge_items("g1", category="lesson")[0]["content"] == {"gain": 0.15}

        # Decision requests (coalescing + ready_before_ts + resolve)
        d1 = db.enqueue_decision_request(
            "g1",
            tier="L1",
            trigger_kind="EVIDENCE",
            coalesce_key="g1:evolve",
            coalesce_window_s=10.0,
            payload={"receipts": ["r1"]},
            now=500.0,
        )
        d1_coalesced = db.enqueue_decision_request(
            "g1",
            tier="L2",
            trigger_kind="EVIDENCE",
            coalesce_key="g1:evolve",
            coalesce_window_s=10.0,
            payload={"receipts": ["r1", "r2"]},
            now=503.0,
        )
        assert d1_coalesced["req_id"] == d1["req_id"]
        assert d1_coalesced["tier"] == "L2"
        assert db.list_decision_requests("g1", status="PENDING", ready_before_ts=505.0) == []
        assert len(db.list_decision_requests("g1", status="PENDING", ready_before_ts=511.0)) == 1

        resolved = db.resolve_decision_request(
            d1["req_id"],
            status="RESOLVED",
            resolution={"spawned": ["h2"]},
            hq_job_id=77,
            expected_statuses="PENDING",
            now=512.0,
        )
        assert resolved["status"] == "RESOLVED"
        assert resolved["hq_job_id"] == 77
        assert resolved["resolved_ts"] == 512.0
        with pytest.raises(StateTransitionError):
            db.resolve_decision_request(d1["req_id"], status="CANCELLED", expected_statuses="PENDING")
        with pytest.raises(KeyError):
            db.resolve_decision_request("missing-req")

        # Interventions
        db.record_intervention("goal", "pause", goal_id="g1", before_epoch=0, after_epoch=1, now=600.0)
        assert len(db.list_interventions(goal_id="g1")) == 1


def test_multigoal_isolated_pause_resume_stop_and_rebalance(tmp_path):
    hq = FakeHQ()
    with StateDB(tmp_path) as db:
        mgr = GlobalGoalManager(db, hq=hq, notify_reactor=False)
        assert mgr.get_global_config() == {
            "global_max_tokens_budget": DEFAULT_GLOBAL_MAX_TOKENS_BUDGET,
            "foreground_reserve_slots": DEFAULT_FOREGROUND_RESERVE_SLOTS,
        }
        mgr.set_global_config("global_max_tokens_budget", 500_000)
        mgr.apply_intervention(
            "set_global_budget",
            scope="global",
            payload={"foreground_reserve_slots": 2},
        )
        assert mgr.get_global_config()["foreground_reserve_slots"] == 2

        ga = mgr.create_or_update_goal(_make_goal("goal-a"))
        gb = mgr.create_or_update_goal(_make_goal("goal-b"))
        gc = mgr.create_or_update_goal(_make_goal("goal-c"))

        db.register_hypothesis(_make_hyp("ha-1", "goal-a"))
        db.register_hypothesis(_make_hyp("hb-1", "goal-b"))
        db.enqueue_task(_make_task("ta-run", "goal-a", "ha-1", status="RUNNING", hq_job_id=101))
        db.enqueue_task(_make_task("tb-run", "goal-b", "hb-1", status="RUNNING", hq_job_id=202))
        db.save_workflow(WorkflowNode(workflow_id="wfa", goal_id="goal-a", hypothesis_id="ha-1", status="RUNNING"))
        db.save_workflow(WorkflowNode(workflow_id="wfb", goal_id="goal-b", hypothesis_id="hb-1", status="RUNNING"))

        # Pausing goal-a cancels ONLY goal-a running HQ task (leaving goal-b and goal-c untouched)
        pause_res = mgr.apply_intervention("pause", scope="goal", goal_id="goal-a", reason="focus elsewhere")
        assert db.get_goal_status("goal-a") == "PAUSED"
        assert db.get_goal_status("goal-b") == "ACTIVE"
        assert db.get_goal_status("goal-c") == "ACTIVE"
        assert db.get_goal("goal-a").human_epoch == ga.human_epoch + 1
        assert db.get_goal("goal-b").human_epoch == gb.human_epoch
        assert pause_res["cancelled_task_ids"] == ["ta-run"]
        assert pause_res["cancelled_hq_job_ids"] == [101]
        assert db.get_task("ta-run").status == "CANCELLED"
        assert db.get_task("tb-run").status == "RUNNING"

        # Resuming goal-a spawns fresh attempt+1 READY replacement task cleanly
        res_out = mgr.apply_intervention("resume", scope="goal", goal_id="goal-a")
        assert db.get_goal_status("goal-a") == "ACTIVE"
        assert len(res_out["spawned_replacement_task_ids"]) == 1
        repl_tid = res_out["spawned_replacement_task_ids"][0]
        assert db.get_task(repl_tid).attempt == 2

        # Stopping goal-a cancels ONLY goal-a replacement/active tasks & workflows
        stop_res = mgr.apply_intervention("stop", scope="goal", goal_id="goal-a", reason="user stop")
        assert db.get_goal_status("goal-a") == "STOPPED"
        assert db.get_goal_status("goal-b") == "ACTIVE"
        assert stop_res["cancelled_task_ids"] == [repl_tid]
        assert hq.cancelled == [([101], "focus elsewhere")]
        assert db.get_task("ta-run").status == "CANCELLED"
        assert db.get_task(repl_tid).status == "CANCELLED"
        assert db.get_task("tb-run").status == "RUNNING"
        assert db.get_workflow("wfa").status == "CANCELLED"
        assert db.get_workflow("wfb").status == "RUNNING"

        # Cannot resume STOPPED goal without force
        with pytest.raises(StateTransitionError):
            mgr.apply_intervention("resume", scope="goal", goal_id="goal-a")

        # pause_all / resume_all skips STOPPED goal-a
        p_all = mgr.apply_intervention("pause_all", scope="global")
        assert set(p_all["paused_goal_ids"]) == {"goal-b", "goal-c"}
        assert db.get_goal_status("goal-a") == "STOPPED"
        r_all = mgr.apply_intervention("resume_all", scope="global")
        assert set(r_all["resumed_goal_ids"]) == {"goal-b", "goal-c"}
        assert db.get_goal_status("goal-a") == "STOPPED"

        # Rebalance weights/slots/budgets across goals
        reb = mgr.apply_intervention(
            "rebalance_goals",
            scope="global",
            payload={
                "weights": {"goal-b": 3.5, "goal-c": 0.5},
                "min_slots": {"goal-b": 2},
                "budgets": {"goal-c": 250_000},
            },
        )
        assert len(reb["updated_goals"]) == 2
        assert db.get_goal("goal-b").priority_weight == 3.5
        assert db.get_goal("goal-b").min_guaranteed_slots == 2
        assert db.get_goal("goal-c").priority_weight == 0.5
        assert db.get_goal("goal-c").max_tokens_budget == 250_000
        assert db.get_goal("goal-c").version == gc.version  # rebalance doesn't bump scientific version


def test_update_objective_impact_recheck_and_epoch_validation(tmp_path):
    hq = FakeHQ()
    with StateDB(tmp_path) as db:
        mgr = GlobalGoalManager(db, hq=hq, notify_reactor=False)
        mgr.create_or_update_goal(
            _make_goal("goal-1", primary_metric="p99_ms", direction="minimize", baseline_value=100.0)
        )

        # Two hypotheses with stored receipts: 85.0 (beats 100) and 92.0 (beats 100)
        h1 = _make_hyp("h1", "goal-1", mechanism="m1", status="COMPLETED", scientific_verdict=ScientificVerdict.CONFIRMED)
        h2 = _make_hyp("h2", "goal-1", mechanism="m2", status="COMPLETED", scientific_verdict=ScientificVerdict.CONFIRMED)
        h3 = _make_hyp("h3", "goal-1", mechanism="m3", status="ACTIVE")
        db.register_hypothesis(h1)
        db.register_hypothesis(h2)
        db.register_hypothesis(h3)

        t1 = _make_task("t1", "goal-1", "h1", status="DONE", goal_version=1)
        t2 = _make_task("t2", "goal-1", "h2", status="DONE", goal_version=1)
        t3_inflight = _make_task("t3", "goal-1", "h3", status="RUNNING", hq_job_id=303, goal_version=1)
        for t in (t1, t2, t3_inflight):
            db.enqueue_task(t)

        r1 = _make_receipt(
            "r1", "t1", "goal-1", "h1",
            metrics={"p99_ms": {"value": 85.0, "denominator": 100, "stage_rung": "F1_FAST_SCREEN"}},
            goal_version=1,
        )
        r2 = _make_receipt(
            "r2", "t2", "goal-1", "h2",
            metrics={"p99_ms": {"value": 92.0, "denominator": 100, "stage_rung": "F1_FAST_SCREEN"}},
            goal_version=1,
        )
        db.record_receipt(r1)
        db.record_receipt(r2)

        # Dry-run impact analysis tightening baseline from 100.0 to 88.0: h1 stays CONFIRMED, h2 -> VALID_NEGATIVE
        impact = mgr.analyze_version_impact("goal-1", {"baseline_value": 88.0})
        assert impact["objective_changed"] is True
        assert impact["affected_task_ids"] == ["t3"]
        assert impact["recommendation"] == "CANCEL_INFLIGHT_AND_RECHECK"
        by_hyp = {r["hypothesis_id"]: r for r in impact["recheck_preview"]["results"]}
        assert by_hyp["h1"]["new_verdict"] == "CONFIRMED"
        assert by_hyp["h2"]["new_verdict"] == "VALID_NEGATIVE"
        assert db.get_goal("goal-1").version == 1  # dry-run did not mutate DB

        # Apply update_objective -> bumps version to 2, cancels stale inflight t3, rechecks stored receipts
        upd = mgr.apply_intervention(
            "update_objective",
            scope="goal",
            goal_id="goal-1",
            payload={"baseline_value": 88.0, "target_value": 85.0},
            reason="tighten SLA",
        )
        g_v2 = db.get_goal("goal-1")
        assert g_v2.version == 2
        assert upd["cancelled_task_ids"] == ["t3"]
        assert upd["cancelled_hq_job_ids"] == [303]
        assert db.get_hypothesis("h1").scientific_verdict == ScientificVerdict.CONFIRMED
        assert db.get_hypothesis("h2").scientific_verdict == ScientificVerdict.VALID_NEGATIVE
        assert len(db.list_goal_versions("goal-1")) == 2
        assert len(db.list_knowledge_items("goal-1", category="rechecked_conclusion")) == 2

        # Epoch validation catches stale v1 task/receipt against v2 goal
        assert mgr.validate_receipt_epoch(g_v2, t1, r1) == EvidenceValidity.STALE_GOAL_VERSION

        # Changing primary_metric to an unmeasured metric rechecks to MISSING_PRIMARY_METRIC / UNKNOWN
        mgr.apply_intervention(
            "update_objective",
            scope="goal",
            goal_id="goal-1",
            payload={"primary_metric": "throughput_qps", "direction": "maximize", "baseline_value": 500.0},
        )
        assert db.get_hypothesis("h1").scientific_verdict == ScientificVerdict.UNKNOWN


def test_branch_workflow_task_interventions_and_boundary_errors(tmp_path):
    hq = FakeHQ()
    with StateDB(tmp_path) as db:
        mgr = GlobalGoalManager(db, hq=hq, notify_reactor=False)
        mgr.create_or_update_goal(_make_goal("goal-1"))

        # Inject hypothesis (valid and invalid)
        with pytest.raises(ValueError):
            mgr.apply_intervention(
                "inject_hypothesis",
                scope="branch",
                goal_id="goal-1",
                payload={
                    "title": "Bad path",
                    "mechanism": "peek",
                    "writable_paths": ["secret_holdout/foo.py"],
                },
            )

        inj = mgr.apply_intervention(
            "inject_hypothesis",
            scope="branch",
            goal_id="goal-1",
            payload={
                "hypothesis_id": "h-root",
                "title": "Human seed",
                "mechanism": "human-guided prefetch",
                "writable_paths": ["src/"],
            },
        )
        assert inj["created"] is True

        # Child hypothesis + workflow hierarchy + tasks
        h_child = _make_hyp("h-child", "goal-1", parent_id="h-root", mechanism="child-opt")
        db.register_hypothesis(h_child)
        wf_parent = WorkflowNode(workflow_id="wf-p", goal_id="goal-1", hypothesis_id="h-child", status="RUNNING")
        wf_sub = WorkflowNode(workflow_id="wf-sub", goal_id="goal-1", hypothesis_id="h-child", parent_workflow_id="wf-p", status="READY")
        db.save_workflow(wf_parent)
        db.save_workflow(wf_sub)
        t_sub = _make_task("t-sub", "goal-1", "h-child", status="RUNNING", hq_job_id=501, workflow_id="wf-sub")
        db.enqueue_task(t_sub)

        # focus_branch
        foc = mgr.apply_intervention("focus_branch", scope="branch", goal_id="goal-1", target_id="h-root")
        assert foc["priority_evoi"] >= 2.0
        assert db.get_hypothesis("h-root").params["human_focused"] is True

        # cancel_workflow cascades to child workflows & tasks & HQ
        cw = mgr.apply_intervention("cancel_workflow", scope="workflow", target_id="wf-p")
        assert set(cw["cancelled_workflow_ids"]) == {"wf-p", "wf-sub"}
        assert cw["cancelled_task_ids"] == ["t-sub"]
        assert cw["cancelled_hq_job_ids"] == [501]
        assert db.get_workflow("wf-sub").status == "CANCELLED"
        assert mgr.validate_receipt_epoch(db.get_goal("goal-1"), db.get_task("t-sub"), None) == EvidenceValidity.HUMAN_INVALIDATED

        # redesign_experiment replaces hypothesis and cancels active tasks
        t_root = _make_task("t-root", "goal-1", "h-root", status="RUNNING", hq_job_id=601)
        db.enqueue_task(t_root)
        red = mgr.apply_intervention(
            "redesign_experiment",
            scope="branch",
            target_id="h-root",
            payload={"title": "Redesigned prefetch", "mechanism": "prefetch-v2"},
        )
        assert db.get_hypothesis("h-root").status == "PRUNED"
        assert red["cancelled_task_ids"] == ["t-root"]
        assert red["cancelled_hq_job_ids"] == [601]
        new_hid = red["new_hypothesis_id"]
        assert db.get_hypothesis(new_hid).parent_id == "h-root"

        # prune_branch prunes new_hid and cancels its active task
        t_new = _make_task("t-new", "goal-1", new_hid, status="READY", hq_job_id=701)
        db.enqueue_task(t_new)
        pruned = mgr.apply_intervention("prune_branch", scope="branch", target_id=new_hid)
        assert new_hid in pruned["pruned_hypothesis_ids"]
        assert pruned["cancelled_hq_job_ids"] == [701]
        rcpt_new = _make_receipt("r-new", "t-new", "goal-1", new_hid)
        assert mgr.validate_receipt_epoch(db.get_goal("goal-1"), db.get_task("t-new"), rcpt_new) == EvidenceValidity.HUMAN_INVALIDATED

        # cancel_task on standalone task
        h_solo = _make_hyp("h-solo", "goal-1", mechanism="solo-opt")
        db.register_hypothesis(h_solo)
        t_solo = _make_task("t-solo", "goal-1", "h-solo", status="RUNNING", hq_job_id=801)
        db.enqueue_task(t_solo)
        ct = mgr.apply_intervention("cancel_task", scope="task", target_id="t-solo")
        assert ct["cancelled_task_ids"] == ["t-solo"]
        assert ct["cancelled_hq_job_ids"] == [801]

        # Operational updates & boundary validations
        mgr.apply_intervention("update_budget", scope="goal", goal_id="goal-1", payload={"max_tokens_budget": 300_000})
        mgr.apply_intervention("set_priority", scope="goal", goal_id="goal-1", payload={"priority_weight": 2.5, "min_guaranteed_slots": 2})
        mgr.apply_intervention("set_autonomy", scope="goal", goal_id="goal-1", payload={"autonomy_mode": "supervised"})
        g_final = db.get_goal("goal-1")
        assert g_final.max_tokens_budget == 300_000
        assert g_final.priority_weight == 2.5
        assert g_final.autonomy_mode == "supervised"

        with pytest.raises(ValueError):
            mgr.create_or_update_goal(_make_goal("bad-dir", direction="sideways"))
        with pytest.raises(ValueError):
            mgr.create_or_update_goal(_make_goal("bad-nan", baseline_value=math.nan))
        with pytest.raises(ValueError):
            mgr.apply_intervention("set_autonomy", scope="goal", goal_id="goal-1", payload={"autonomy_mode": "yolo"})
        with pytest.raises(ValueError):
            mgr.apply_intervention("unknown_action", scope="goal", goal_id="goal-1")
        with pytest.raises(ValueError):
            mgr.apply_intervention("pause", scope="invalid_scope", goal_id="goal-1")
        with pytest.raises(KeyError):
            mgr.analyze_version_impact("nonexistent", {"baseline_value": 1.0})
        with pytest.raises(ValueError):
            mgr.analyze_version_impact("goal-1", {})
