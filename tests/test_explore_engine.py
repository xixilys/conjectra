"""Tests for conjectra.explore: hypothesis tree, EVOI, anti-filler, graveyard dedup, and AI ingestion."""

from __future__ import annotations

import os

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from conjectra.explore import (
    DEFAULT_AI_MODEL,
    ExploreEngine,
    check_anti_filler_guard,
    compute_evoi,
    compute_fingerprint,
    import_proposals_json,
    propose_hypotheses,
    ready_queue_watermark,
    reconcile_receipt,
    validate_hypothesis_candidate,
)
from conjectra.state_db import StateDB
from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ResourceSpec,
    ScientificVerdict,
    StageKind,
    StageTask,
)


def _make_goal(
    *,
    goal_id: str = "goal_demo",
    project: str = "demo",
    adapter_name: str = "demo3",
    max_tokens_budget: int = 100_000,
    max_concurrent_branches: int = 4,
) -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project=project,
        title="Minimize validation-loss delta while keeping >=50% throughput saved",
        direction="minimize",
        primary_metric="val_loss_delta_mean",
        baseline_value=0.05,
        baseline_sha="deadbeef1234",
        repo_path="/tmp/conjectra-demo-repo",
        kill_criteria=["ub95 > 0.0148", "throughput_saved_pct < 50.0"],
        banned_paths=["holdout_out/", "frozen_baseline/", "sealed_eval/", "lockbox/"],
        writable_prefixes=["src/demo/"],
        max_tokens_budget=max_tokens_budget,
        max_concurrent_branches=max_concurrent_branches,
        adapter_name=adapter_name,
    )


def _make_receipt(
    hyp: HypothesisNode,
    *,
    receipt_id: str = "rcpt_1",
    task_id: str = "task_1",
    stage: StageKind = StageKind.EXP_SCREEN,
    process_status: ProcessStatus = ProcessStatus.EXIT_OK,
    engineering_verdict: EngineeringVerdict = EngineeringVerdict.PASS,
    scientific_verdict: ScientificVerdict = ScientificVerdict.CONFIRMED,
    audit_verdict: AuditVerdict = AuditVerdict.NOT_REQUIRED,
    metrics: dict[str, Any] | None = None,
    failure_reason: str = "",
    in_tokens: int = 0,
    out_tokens: int = 0,
) -> EvidenceReceipt:
    return EvidenceReceipt(
        receipt_id=receipt_id,
        task_id=task_id,
        goal_id=hyp.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=stage,
        base_sha=hyp.base_sha,
        head_sha=hyp.head_sha or hyp.base_sha,
        command=hyp.screen_cmd,
        exit_code=0 if process_status == ProcessStatus.EXIT_OK else 1,
        wall_sec=12.5,
        max_rss_mb=256.0,
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        model_used="none",
        process_status=process_status,
        engineering_verdict=engineering_verdict,
        scientific_verdict=scientific_verdict,
        audit_verdict=audit_verdict,
        tests_passed=5,
        tests_total=5,
        metrics=metrics or {"loss_delta": {"value": 0.01, "denominator": 12, "stage_rung": "F1_FAST_SCREEN"}},
        raw_artifact_paths=["/tmp/summary.json"],
        failure_reason=failure_reason,
    )


def test_f1_to_f2_promotion_and_auto_ablation_falsification(tmp_path: Path) -> None:
    db = StateDB(tmp_path)
    goal = _make_goal()
    db.save_goal(goal)

    batch = propose_hypotheses(goal, db=db, mode="seeds", dispatchable_slots=4)
    assert len(batch.accepted) == 3
    kinds = {h.kind for h in batch.accepted}
    assert HypothesisKind.COMPETING_ROUTE in kinds
    assert HypothesisKind.PARAMETER_RACE in kinds

    winner = batch.accepted[0]
    assert winner.fidelity == FidelityRung.F1_FAST_SCREEN

    # F1 CONFIRMED -> promotes to F2_FULL_CONFIRM AND auto-spawns ABLATION + FALSIFICATION.
    rcpt_f1 = _make_receipt(winner, scientific_verdict=ScientificVerdict.CONFIRMED)
    decision = reconcile_receipt(db, goal, winner, rcpt_f1)

    assert decision.action == "PROMOTE_F2"
    assert decision.promoted_to == FidelityRung.F2_FULL_CONFIRM
    assert decision.new_status == "PROMOTED_F2"
    assert len(decision.spawned_hypotheses) == 2
    spawned_kinds = {h.kind for h in decision.spawned_hypotheses}
    assert spawned_kinds == {HypothesisKind.ABLATION, HypothesisKind.FALSIFICATION}
    for child in decision.spawned_hypotheses:
        assert child.parent_id == winner.hypothesis_id
        assert child.fidelity == FidelityRung.F1_FAST_SCREEN
        assert child.fingerprint
        assert db.get_hypothesis(child.hypothesis_id) is not None

    stored_winner = db.get_hypothesis(winner.hypothesis_id)
    assert stored_winner is not None
    assert stored_winner.fidelity == FidelityRung.F2_FULL_CONFIRM

    # F2 CONFIRMED -> promotes to F3_MILESTONE_AUDIT; F3 PASS -> COMPLETED.
    rcpt_f2 = _make_receipt(
        stored_winner,
        receipt_id="rcpt_f2",
        stage=StageKind.EXP_CONFIRM,
        scientific_verdict=ScientificVerdict.CONFIRMED,
    )
    dec_f2 = reconcile_receipt(db, goal, stored_winner, rcpt_f2)
    assert dec_f2.action == "PROMOTE_F3"
    assert dec_f2.promoted_to == FidelityRung.F3_MILESTONE_AUDIT

    stored_f3 = db.get_hypothesis(winner.hypothesis_id)
    assert stored_f3 is not None
    rcpt_f3 = _make_receipt(
        stored_f3,
        receipt_id="rcpt_f3",
        stage=StageKind.BLIND_AUDIT,
        scientific_verdict=ScientificVerdict.CONFIRMED,
        audit_verdict=AuditVerdict.PASS,
    )
    dec_f3 = reconcile_receipt(db, goal, stored_f3, rcpt_f3)
    assert dec_f3.action == "COMPLETE"
    assert dec_f3.new_status == "COMPLETED"
    db.close()


def test_refutation_prunes_descendants_cancels_tasks_and_dedups_graveyard(tmp_path: Path) -> None:
    db = StateDB(tmp_path)
    goal = _make_goal()
    db.save_goal(goal)

    parent = HypothesisNode(
        hypothesis_id="hyp_parent",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Parent route",
        statement="Test parent route",
        mechanism="mech_parent_x",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.7,
        writable_paths=["src/demo/tuned_scheduler.py"],
        params={"tau_hi": 0.18, "alpha": 1.0},
        kill_criteria=["ub95 > 0.0148"],
        acceptance_cmd="pytest -q src/demo/test_tuned_scheduler.py",
        screen_cmd="python3 -B -m src.demo.fast_race --dry-run-cpu",
        confirm_cmd=None,
        branch_name="arr/projbeta/hyp_parent",
        base_sha=goal.baseline_sha,
    )
    parent.fingerprint = compute_fingerprint(
        parent.kind, parent.mechanism, parent.params, parent.writable_paths, project=goal.project
    )
    db.register_hypothesis(parent)

    child = HypothesisNode(
        hypothesis_id="hyp_child",
        goal_id=goal.goal_id,
        parent_id=parent.hypothesis_id,
        kind=HypothesisKind.REFINEMENT,
        title="Child refinement",
        statement="Downstream child",
        mechanism="mech_child_y",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.5,
        writable_paths=["src/demo/tuned_scheduler.py"],
        params={"tau_hi": 0.22},
        kill_criteria=["ub95 > 0.0148"],
        acceptance_cmd="pytest -q src/demo/test_tuned_scheduler.py",
        screen_cmd="python3 -B -m src.demo.fast_race --dry-run-cpu",
        confirm_cmd=None,
        branch_name="arr/projbeta/hyp_child",
        base_sha=goal.baseline_sha,
    )
    child.fingerprint = compute_fingerprint(
        child.kind, child.mechanism, child.params, child.writable_paths, project=goal.project
    )
    db.register_hypothesis(child)

    child_task = StageTask(
        task_id="task_child_1",
        idempotency_key="idem_child_1",
        goal_id=goal.goal_id,
        hypothesis_id=child.hypothesis_id,
        stage=StageKind.EXP_SCREEN,
        deps=[],
        resources=ResourceSpec(),
        status="READY",
    )
    db.enqueue_task(child_task)

    # Trigger kill-criterion hit even if receipt initially said UNKNOWN!
    rcpt = _make_receipt(
        parent,
        scientific_verdict=ScientificVerdict.UNKNOWN,
        metrics={"ub95": {"value": 0.0210, "denominator": 201, "stage_rung": "F1_FAST_SCREEN"}},
    )
    decision = reconcile_receipt(db, goal, parent, rcpt)

    assert decision.action == "REFUTE_AND_PRUNE"
    assert decision.new_status == "REFUTED"
    assert decision.scientific_verdict == ScientificVerdict.REFUTED
    assert decision.kill_criterion_hit == "ub95 > 0.0148"
    assert decision.graveyard_recorded is True
    assert child.hypothesis_id in decision.pruned_hypothesis_ids
    assert child_task.task_id in decision.cancelled_task_ids

    assert db.get_hypothesis(parent.hypothesis_id).status == "REFUTED"
    assert db.get_hypothesis(child.hypothesis_id).status == "PRUNED"
    assert db.get_task(child_task.task_id).status == "CANCELLED"
    assert len(db.list_graveyard(goal_id=goal.goal_id)) == 1

    # Re-proposing the equivalent config with reordered params & ./ prefix MUST be deduplicated by graveyard!
    reproposal = [
        {
            "kind": "COMPETING_ROUTE",
            "title": "Renamed duplicate parent",
            "statement": "Retry same mechanism with reordered params",
            "mechanism": "  MECH_PARENT_X ",
            "writable_paths": ["./src/demo/tuned_scheduler.py"],
            "params": {"alpha": 1, "tau_hi": 0.18000000},
            "kill_criteria": ["ub95 > 0.0148"],
            "screen_cmd": "python3 -B -m src.demo.fast_race --dry-run-cpu",
            "acceptance_cmd": "pytest -q src/demo/test_tuned_scheduler.py",
        }
    ]
    res = import_proposals_json(goal, reproposal, db=db)
    assert res.accepted == []
    assert len(res.rejected) == 1
    assert "graveyard" in res.rejected[0]["reasons"][0].lower()
    db.close()


def test_unknown_never_fabricates_pass_and_escalates_on_streak(tmp_path: Path) -> None:
    db = StateDB(tmp_path)
    goal = _make_goal(max_tokens_budget=50_000)
    db.save_goal(goal)

    engine = ExploreEngine(db=db, max_unknown_refinements=1)
    batch = engine.propose(goal, mode="seeds", dispatchable_slots=4)
    hyp = batch.accepted[0]

    # Even if scientific_verdict says CONFIRMED, FAIL_TEST must NEVER promote or fabricate pass!
    bad_gate_rcpt = _make_receipt(
        hyp,
        receipt_id="rcpt_bad",
        engineering_verdict=EngineeringVerdict.FAIL_TEST,
        scientific_verdict=ScientificVerdict.CONFIRMED,
    )
    dec1 = engine.reconcile(goal, hyp, bad_gate_rcpt)
    assert dec1.scientific_verdict == ScientificVerdict.UNKNOWN
    assert dec1.action == "DIAGNOSTIC_REFINEMENT"
    assert len(dec1.spawned_hypotheses) == 1
    diag = dec1.spawned_hypotheses[0]
    assert diag.kind == HypothesisKind.REFINEMENT

    # Second UNKNOWN on diagnostic child exceeds max_unknown_refinements=1 -> ESCALATED.
    rcpt_diag = _make_receipt(
        diag,
        receipt_id="rcpt_diag_unknown",
        scientific_verdict=ScientificVerdict.UNKNOWN,
    )
    dec2 = engine.reconcile(goal, diag, rcpt_diag)
    assert dec2.action == "ESCALATE_UNKNOWN"
    assert dec2.escalate is True
    assert dec2.new_status == "ESCALATED"
    db.close()


def test_anti_filler_watermark_budget_and_evoi_ranking(tmp_path: Path) -> None:
    goal = _make_goal(max_tokens_budget=10_000, max_concurrent_branches=4)
    assert ready_queue_watermark(4) == pytest.approx(6.0)

    # Queue at watermark (6 >= 1.5 * 4) refuses new proposals.
    res_full = propose_hypotheses(goal, mode="seeds", ready_queue_count=6, dispatchable_slots=4, persist=False)
    assert res_full.accepted == []
    assert res_full.stopped_reason is not None
    assert "WATERMARK" in res_full.stopped_reason

    # Queue at 4 has headroom of 2 under watermark=6 -> trims 3 seeds down to top 2 by EVOI.
    res_headroom = propose_hypotheses(goal, mode="seeds", ready_queue_count=4, dispatchable_slots=4, persist=False)
    assert len(res_headroom.accepted) == 2
    assert res_headroom.accepted[0].priority_evoi >= res_headroom.accepted[1].priority_evoi

    # Token budget fuse blocks generation when budget nearly exhausted.
    allowed, reason = check_anti_filler_guard(goal, ready_queue_count=0, dispatchable_slots=4, used_tokens=9_800)
    assert allowed is False
    assert "TOKEN_BUDGET_FUSE" in (reason or "")

    # Diminishing returns fuse blocks generation after streak.
    allowed_dr, reason_dr = check_anti_filler_guard(
        goal, ready_queue_count=0, dispatchable_slots=4, used_tokens=100, consecutive_non_improving=4
    )
    assert allowed_dr is False
    assert "DIMINISHING_RETURNS_FUSE" in (reason_dr or "")

    # EVOI ranking prefers high gain, high novelty, and lower cost.
    base_existing = res_headroom.accepted[0]
    cheap_novel = HypothesisNode(
        hypothesis_id="h_cheap_novel",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Novel low-cost sparse quantization route",
        statement="Novel mechanism",
        mechanism="novel_sparse_quant",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.0,
        writable_paths=["src/demo/fused_engine.py"],
        params={"expected_delta": 0.04, "estimated_tokens": 500, "estimated_wall_s": 30.0},
        kill_criteria=["ub95 > 0.0148"],
        acceptance_cmd="pytest -q",
        screen_cmd="pytest -q",
        confirm_cmd=None,
        branch_name="arr/projbeta/h_cheap_novel",
        base_sha=goal.baseline_sha,
    )
    expensive_dup = HypothesisNode(
        hypothesis_id="h_exp_near_dup",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.REFINEMENT,
        title=base_existing.title + " variant",
        statement="Near duplicate",
        mechanism=base_existing.mechanism + " v2",
        fidelity=FidelityRung.F2_FULL_CONFIRM,
        priority_evoi=0.0,
        writable_paths=list(base_existing.writable_paths),
        params={"expected_delta": 0.005, "estimated_tokens": 9000, "estimated_wall_s": 900.0},
        kill_criteria=["ub95 > 0.0148"],
        acceptance_cmd="pytest -q",
        screen_cmd="pytest -q",
        confirm_cmd=None,
        branch_name="arr/projbeta/h_exp_near_dup",
        base_sha=goal.baseline_sha,
    )
    evoi_high = compute_evoi(goal, cheap_novel, [base_existing])
    evoi_low = compute_evoi(goal, expensive_dup, [base_existing])
    assert evoi_high > evoi_low > 0.0


def test_ai_proposer_json_parsing_usage_and_strict_rejection(tmp_path: Path) -> None:
    db = StateDB(tmp_path)
    goal = _make_goal(adapter_name="projalpha", project="projalpha", goal_id="goal_projalpha")
    goal.writable_prefixes = ["harness/core/"]
    goal.banned_paths = ["sealed_eval/", "lockbox/", "holdout_out/"]
    db.save_goal(goal)

    captured_cmds: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured_cmds.append(cmd)
        usage_idx = cmd.index("--usage-out") + 1
        usage_path = Path(cmd[usage_idx])
        usage_path.write_text(
            json.dumps(
                {
                    "route": "gemini_cli",
                    "host": "local",
                    "model": DEFAULT_AI_MODEL,
                    "usage": {"input_tokens": 1234, "output_tokens": 567},
                }
            ),
            encoding="utf-8",
        )
        # Return 1 valid proposal + 4 invalid proposals (banned path, outside prefix, no kill_criteria, main branch)
        ai_payload = [
            {
                "kind": "COMPETING_ROUTE",
                "title": "Valid PROJALPHA calibration invariant route",
                "statement": "Enforce strict calibration domain check",
                "mechanism": "strict_calib_domain_check",
                "writable_paths": ["harness/core/calibration/policy.py"],
                "params": {"strict": True, "estimated_tokens": 2000},
                "kill_criteria": ["false_pruning > 0"],
                "screen_cmd": "pytest -q tests/test_calibration_spotcheck.py",
                "confirm_cmd": "python3 -m unittest core.calibration.test_policy",
                "acceptance_cmd": "./check.sh --related --base main",
            },
            {
                "kind": "COMPETING_ROUTE",
                "title": "Banned sealed_eval access",
                "statement": "Should be rejected",
                "mechanism": "bad_sealed_eval",
                "writable_paths": ["sealed_eval/secret.py"],
                "params": {},
                "kill_criteria": ["x > 0"],
                "screen_cmd": "cat sealed_eval/secret.py",
                "acceptance_cmd": "pytest -q",
            },
            {
                "kind": "COMPETING_ROUTE",
                "title": "Outside writable prefix",
                "statement": "Should be rejected",
                "mechanism": "outside_prefix",
                "writable_paths": ["gem5_sim/core.cc"],
                "params": {},
                "kill_criteria": ["x > 0"],
                "screen_cmd": "pytest -q",
                "acceptance_cmd": "pytest -q",
            },
            {
                "kind": "COMPETING_ROUTE",
                "title": "Missing kill criteria & acceptance",
                "statement": "Should be rejected",
                "mechanism": "no_kill_crit",
                "writable_paths": ["harness/core/design/decision.py"],
                "params": {},
                "kill_criteria": [],
                "screen_cmd": "pytest -q",
                "acceptance_cmd": "",
            },
            {
                "kind": "COMPETING_ROUTE",
                "title": "Targets main branch",
                "statement": "Should be rejected",
                "mechanism": "touches_main",
                "writable_paths": ["harness/core/design/decision.py"],
                "params": {},
                "kill_criteria": ["err > 0"],
                "screen_cmd": "pytest -q",
                "acceptance_cmd": "pytest -q",
                "branch_name": "main",
            },
        ]
        stdout = "```json\n" + json.dumps(ai_payload) + "\n```"
        return subprocess.CompletedProcess(cmd, 0, stdout=stdout, stderr="")

    res = propose_hypotheses(
        goal,
        db=db,
        mode="ai",
        subprocess_runner=fake_runner,
        max_ai_proposals=5,
        dispatchable_slots=4,
    )

    assert len(captured_cmds) == 1
    cmd = captured_cmds[0]
    assert "gemini" in cmd
    assert "--host" in cmd and cmd[cmd.index("--host") + 1] == "local"
    assert "--model" in cmd and cmd[cmd.index("--model") + 1] == DEFAULT_AI_MODEL
    assert "--readonly" in cmd

    assert res.in_tokens == 1234
    assert res.out_tokens == 567
    assert res.model_used == DEFAULT_AI_MODEL
    assert len(res.accepted) == 1
    assert res.accepted[0].title == "Valid PROJALPHA calibration invariant route"
    assert len(res.rejected) == 4

    # Malformed AI stdout is rejected cleanly without crashing.
    def broken_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(cmd, 0, stdout="not json at all!!!", stderr="")

    res_bad = propose_hypotheses(goal, db=db, mode="ai", subprocess_runner=broken_runner)
    assert res_bad.accepted == []
    assert res_bad.stopped_reason is not None
    assert "parse error" in res_bad.stopped_reason.lower()

    # Validate direct validator rejection on protected / lockbox.
    os.environ["CONJECTRA_PROTECTED_PATHS"] = "harness/protected"
    vr_protected = validate_hypothesis_candidate(
        {
            "kind": "ABLATION",
            "title": "Touch protected",
            "statement": "Bad",
            "mechanism": "bad",
            "writable_paths": ["/home/user/workspaces/harness/protected/foo.py"],
            "kill_criteria": ["fail"],
            "screen_cmd": "pytest -q",
            "acceptance_cmd": "pytest -q",
        },
        goal,
    )
    assert vr_protected.valid is False
    os.environ.pop("CONJECTRA_PROTECTED_PATHS", None)
    db.close()




def test_import_proposals_from_protected_cwd_and_reject_protected_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CONJECTRA_PROTECTED_PATHS", "harness/protected")
    protected_dir = tmp_path / "harness" / "protected"
    protected_dir.mkdir(parents=True)
    target_repo = tmp_path / "target_repo"
    target_repo.mkdir()
    monkeypatch.chdir(protected_dir)

    goal = ResearchGoal(
        goal_id="goal_cwd_test",
        project="demo",
        title="Demo goal",
        direction="maximize",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha="abc1234",
        repo_path=str(target_repo),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["sealed_eval/", "lockbox/"],
        writable_prefixes=["src/"],
        max_tokens_budget=50_000,
        max_concurrent_branches=4,
        adapter_name="generic",
    )
    proposals = [
        {
            "kind": "COMPETING_ROUTE",
            "title": "Valid relative src/m0.py route",
            "statement": "Improve score safely",
            "mechanism": "safe_m0_route",
            "writable_paths": ["src/m0.py"],
            "params": {"alpha": 2},
            "kill_criteria": ["tests_passed == 0"],
            "screen_cmd": "python3 -m pytest -q && true",
            "acceptance_cmd": "python3 -m pytest -q",
        }
    ]
    res = import_proposals_json(goal, proposals, persist=False)
    assert len(res.accepted) == 1
    assert res.rejected == []

    # Explicit protected target must still be rejected.
    bad_vr = validate_hypothesis_candidate(
        {
            "kind": "COMPETING_ROUTE",
            "title": "Bad protected command",
            "statement": "Reads protected",
            "mechanism": "bad_protected_cmd",
            "writable_paths": ["src/m0.py"],
            "kill_criteria": ["tests_passed == 0"],
            "screen_cmd": "cat /home/user/workspaces/harness/protected/README.md",
            "acceptance_cmd": "python3 -m pytest -q",
        },
        goal,
    )
    assert bad_vr.valid is False
