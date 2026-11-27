"""Comprehensive V4 tests for SubGoal evaluation, prompt injection, generation caps, portfolio actions, and two-tier DecisionEngine."""

from __future__ import annotations

import json
import pathlib
from typing import Any

import pytest

from conjectra.adapters import GenericAdapter, evaluate_scientific_verdict
from conjectra.decision_engine import (
    MAX_SUBGOAL_DIGEST_BYTES,
    DecisionEngine,
    build_portfolio_sandtable,
    build_subgoal_digest,
    parse_lead_decision_payload,
)
from conjectra.explore import (
    apply_scientific_actions,
    import_proposals_json,
    propose_hypotheses,
)
from conjectra.state_db import StateDB
from conjectra.types import (
    ActionKind,
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    LeadAction,
    LeadDecisionEnvelope,
    ProcessStatus,
    ResearchGoal,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
    SubGoalKind,
    SubGoalNode,
    SubGoalStatus,
    WorkflowNode,
)
from conjectra.workflow_runtime import WorkflowRuntime


def _make_goal(tmp_path: pathlib.Path, goal_id: str = "g_v4_sg") -> ResearchGoal:
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir(parents=True, exist_ok=True)
    return ResearchGoal(
        goal_id=goal_id,
        project="projx",
        title="Improve end-to-end accuracy above 0.90",
        direction="maximize",
        primary_metric="e2e_accuracy",
        baseline_value=0.70,
        baseline_sha="aaaa1111bbbb",
        repo_path=str(repo_dir),
        kill_criteria=["e2e_accuracy < 0.50"],
        banned_paths=["te_sealed/", "lockbox/", "secrets/"],
        writable_prefixes=["src/", "tests/"],
        max_tokens_budget=200_000,
        version=1,
        human_epoch=1,
    )


def _make_subgoal(
    goal: ResearchGoal,
    sg_id: str = "sg_1",
    *,
    local_metric: str = "cache_hit_rate",
    baseline_value: float = 0.50,
    target_value: float | None = 0.85,
    direction: str = "maximize",
    kill_criteria: list[str] | None = None,
    status: str = "ACTIVE",
    depends_on: list[str] | None = None,
    sealed_sha: str | None = None,
    max_auto_generations: int = 1,
    priority_weight: float = 1.0,
    max_tokens_budget: int = 40_000,
    spec_paths: list[str] | None = None,
) -> SubGoalNode:
    return SubGoalNode(
        subgoal_id=sg_id,
        goal_id=goal.goal_id,
        title=f"Track {sg_id}",
        direction_thesis=f"Optimize {local_metric} cleanly",
        direct_contribution=f"Implement algorithmic improvement for {local_metric}",
        local_metric=local_metric,
        kind=SubGoalKind.DIRECTION_TRACK,
        status=status,
        baseline_value=baseline_value,
        target_value=target_value,
        direction=direction,
        kill_criteria=list(kill_criteria if kill_criteria is not None else [f"{local_metric} < 0.20"]),
        depends_on_subgoals=list(depends_on or []),
        track_branch=f"track/{goal.project}/{sg_id}",
        base_sha=goal.baseline_sha,
        sealed_sha=sealed_sha,
        spec_paths=list(spec_paths or []),
        writable_prefixes=["src/", "tests/"],
        priority_weight=priority_weight,
        max_tokens_budget=max_tokens_budget,
        max_auto_generations=max_auto_generations,
    )


def _make_hyp(
    goal: ResearchGoal,
    hid: str,
    *,
    subgoal_id: str | None = None,
    parent_id: str | None = None,
    head_sha: str | None = None,
    params: dict[str, Any] | None = None,
    kill_criteria: list[str] | None = None,
) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id=hid,
        goal_id=goal.goal_id,
        parent_id=parent_id,
        kind=HypothesisKind.COMPETING_ROUTE,
        title=f"Hyp {hid}",
        statement=f"Improve metric via {hid}",
        mechanism=f"mech_{hid}",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.8,
        writable_paths=["src/mod.py", "tests/test_mod.py"],
        params=dict(params or {"requires_llm_code": True}),
        kill_criteria=list(kill_criteria if kill_criteria is not None else ["tests_passed == 0"]),
        acceptance_cmd="pytest -q tests/test_mod.py",
        screen_cmd="python3 -m src.run_screen",
        confirm_cmd="python3 -m src.run_confirm",
        branch_name=f"conjectra/{goal.project}/{hid}",
        base_sha=goal.baseline_sha,
        head_sha=head_sha,
        subgoal_id=subgoal_id,
    )


def test_subgoal_local_metric_verdict_and_adapter_parse_metrics(tmp_path: pathlib.Path) -> None:
    goal = _make_goal(tmp_path)
    sg = _make_subgoal(
        goal,
        "sg_cache",
        local_metric="cache_hit_rate",
        baseline_value=0.60,
        target_value=0.85,
        direction="maximize",
        kill_criteria=["cache_hit_rate < 0.30"],
    )
    hyp_no_override = _make_hyp(goal, "h_cache_1", subgoal_id=sg.subgoal_id, kill_criteria=["mem_mb > 4096"])

    # 1. With subgoal=sg passed, evaluates cache_hit_rate instead of failing MISSING_PRIMARY_METRIC on e2e_accuracy.
    val, sv, reason = evaluate_scientific_verdict(
        {"cache_hit_rate": {"value": 0.78, "denominator": 100, "stage_rung": "F1_FAST_SCREEN"}},
        {"cache_hit_rate": 0.78},
        goal,
        hyp_no_override,
        StageKind.EXP_SCREEN,
        0,
        subgoal=sg,
    )
    assert val == EvidenceValidity.VALID_EMPIRICAL
    assert sv == ScientificVerdict.CONFIRMED, reason

    # 2. Below baseline -> VALID_NEGATIVE against subgoal.baseline_value
    val_neg, sv_neg, reason_neg = evaluate_scientific_verdict(
        {"cache_hit_rate": {"value": 0.55, "denominator": 100, "stage_rung": "F1_FAST_SCREEN"}},
        {},
        goal,
        hyp_no_override,
        StageKind.EXP_SCREEN,
        0,
        subgoal=sg,
    )
    assert val_neg == EvidenceValidity.VALID_EMPIRICAL
    assert sv_neg == ScientificVerdict.VALID_NEGATIVE
    assert "0.6" in reason_neg

    # 3. SubGoal kill criterion triggers REFUTED; hyp.kill_criteria also triggers REFUTED
    val_kill_sg, sv_kill_sg, reason_kill_sg = evaluate_scientific_verdict(
        {"cache_hit_rate": {"value": 0.25, "denominator": 100, "stage_rung": "F1_FAST_SCREEN"}},
        {},
        goal,
        hyp_no_override,
        StageKind.EXP_SCREEN,
        0,
        subgoal=sg,
    )
    assert val_kill_sg == EvidenceValidity.VALID_EMPIRICAL
    assert sv_kill_sg == ScientificVerdict.REFUTED
    assert "cache_hit_rate < 0.30" in reason_kill_sg

    val_kill_hyp, sv_kill_hyp, reason_kill_hyp = evaluate_scientific_verdict(
        {
            "cache_hit_rate": {"value": 0.80, "denominator": 100, "stage_rung": "F1_FAST_SCREEN"},
            "mem_mb": {"value": 8192, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"},
        },
        {},
        goal,
        hyp_no_override,
        StageKind.EXP_SCREEN,
        0,
        subgoal=sg,
    )
    assert val_kill_hyp == EvidenceValidity.VALID_EMPIRICAL
    assert sv_kill_hyp == ScientificVerdict.REFUTED
    assert "mem_mb > 4096" in reason_kill_hyp

    # 4. When subgoal is None, hyp.params["subgoal_local_metric"] override works via GenericAdapter.parse_metrics_with_validity
    adapter = GenericAdapter()
    hyp_params_sg = _make_hyp(
        goal,
        "h_lat",
        subgoal_id="sg_lat",
        params={
            "subgoal_local_metric": "p99_ms",
            "subgoal_baseline_value": 50.0,
            "subgoal_direction": "minimize",
            "subgoal_kill_criteria": ["p99_ms > 100"],
        },
    )
    metrics, v_adp, sv_adp, _, r_adp = adapter.parse_metrics_with_validity(
        StageKind.EXP_SCREEN,
        "4 passed in 0.1s\nMETRIC p99_ms=32.5 denom=1",
        "",
        tmp_path,
        tmp_path,
        0,
        hyp_params_sg,
        goal,
    )
    assert "p99_ms" in metrics
    assert v_adp == EvidenceValidity.VALID_EMPIRICAL
    assert sv_adp == ScientificVerdict.CONFIRMED, r_adp

    # Passing explicit subgoal to parse_metrics_with_validity also confirms without goal.primary_metric
    _, v_sub, sv_sub, _, r_sub = adapter.parse_metrics_with_validity(
        StageKind.EXP_SCREEN,
        "METRIC cache_hit_rate=0.81 denom=100",
        "",
        tmp_path,
        tmp_path,
        0,
        hyp_no_override,
        goal,
        subgoal=sg,
    )
    assert v_sub == EvidenceValidity.VALID_EMPIRICAL
    assert sv_sub == ScientificVerdict.CONFIRMED, r_sub


def test_build_code_prompt_injects_bounded_safe_spec_excerpts_and_acceptance(tmp_path: pathlib.Path) -> None:
    goal = _make_goal(tmp_path)
    repo_dir = pathlib.Path(goal.repo_path)
    wt_dir = tmp_path / "worktree"
    (wt_dir / "specs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "docs").mkdir(parents=True, exist_ok=True)
    (repo_dir / "te_sealed").mkdir(parents=True, exist_ok=True)
    outside_secret = tmp_path / "outside_secret.txt"
    outside_secret.write_text("OUTSIDE_SECRET_DO_NOT_LEAK", encoding="utf-8")

    (wt_dir / "specs" / "slice_spec.md").write_text(
        "SPEC_HEADER: Implement bounded ring buffer.\n" + ("A" * 6000),
        encoding="utf-8",
    )
    (repo_dir / "docs" / "extra_spec.md").write_text(
        "REPO_SPEC_CONTENT: Keep complexity O(1).",
        encoding="utf-8",
    )
    (repo_dir / "te_sealed" / "holdout.md").write_text(
        "BANNED_HOLDOUT_LEAK",
        encoding="utf-8",
    )

    hyp = _make_hyp(
        goal,
        "h_spec",
        subgoal_id="sg_spec",
        params={
            "spec_paths": [
                "specs/slice_spec.md",
                "docs/extra_spec.md",
                "te_sealed/holdout.md",
                "../outside_secret.txt",
                str(outside_secret),
            ],
            "subgoal_local_metric": "ring_hit_rate",
            "subgoal_baseline_value": 0.4,
            "subgoal_direction": "maximize",
        },
    )
    adapter = GenericAdapter()
    prompt = adapter.build_code_prompt(hyp, goal, wt_dir)

    assert "SPEC_HEADER: Implement bounded ring buffer." in prompt
    assert "REPO_SPEC_CONTENT: Keep complexity O(1)." in prompt
    assert "BANNED_HOLDOUT_LEAK" not in prompt
    assert "OUTSIDE_SECRET_DO_NOT_LEAK" not in prompt
    assert hyp.acceptance_cmd in prompt
    assert str(wt_dir) in prompt
    # Total spec excerpt portion is strictly < 4KB
    spec_start = prompt.index("Spec excerpts (<4KB):")
    instr_start = prompt.index("\nInstructions:")
    assert len(prompt[spec_start:instr_start].encode("utf-8")) < 4096


def test_subgoal_propagation_and_max_auto_generations_cap(tmp_path: pathlib.Path) -> None:
    goal = _make_goal(tmp_path)
    with StateDB(tmp_path / "state_home") as db:
        db.save_goal(goal)
        sg = _make_subgoal(
            goal,
            "sg_gen",
            local_metric="f1_local",
            baseline_value=0.35,
            direction="maximize",
            kill_criteria=["f1_local < 0.10"],
            max_auto_generations=1,
            spec_paths=["specs/a.md"],
        )
        sg.base_sha = "sgbase9999"
        db.save_subgoal(sg)

        root_prop = {
            "kind": "COMPETING_ROUTE",
            "title": "Root slice under sg_gen",
            "statement": "Implement primary route",
            "mechanism": "sg_gen_root_v1",
            "writable_paths": ["src/a.py"],
            "screen_cmd": "python3 -m src.run_a",
            "acceptance_cmd": "pytest -q tests/test_a.py",
        }
        batch1 = import_proposals_json(goal, [root_prop], db=db, subgoal_id=sg.subgoal_id)
        assert len(batch1.accepted) == 1
        root_hyp = batch1.accepted[0]
        assert root_hyp.subgoal_id == sg.subgoal_id
        assert root_hyp.base_sha == "sgbase9999"
        assert root_hyp.params["spec_paths"] == ["specs/a.md"]
        assert root_hyp.params["subgoal_local_metric"] == "f1_local"
        assert root_hyp.params["subgoal_baseline_value"] == 0.35
        assert root_hyp.params["subgoal_direction"] == "maximize"
        assert root_hyp.params["subgoal_kill_criteria"] == ["f1_local < 0.10"]

        # Child proposal (generation 2 > max_auto_generations=1) must be rejected cleanly
        child_prop = {
            "kind": "REFINEMENT",
            "parent_id": root_hyp.hypothesis_id,
            "title": "Child refinement exceeding cap",
            "statement": "Should hit SUBGOAL_GENERATION_CAP_REACHED",
            "mechanism": "sg_gen_child_v2",
            "writable_paths": ["src/a.py"],
            "kill_criteria": ["f1_local < 0.10"],
            "screen_cmd": "python3 -m src.run_a",
            "acceptance_cmd": "pytest -q tests/test_a.py",
        }
        batch_cap = propose_hypotheses(goal, db=db, mode="import", external_proposals=[child_prop])
        assert batch_cap.accepted == []
        assert len(batch_cap.rejected) == 1
        assert "SUBGOAL_GENERATION_CAP_REACHED" in batch_cap.rejected[0]["reasons"][0]

        # Raise subgoal.max_auto_generations to 2 -> gen 2 allowed, gen 3 rejected
        db.update_subgoal_fields(sg.subgoal_id, {"max_auto_generations": 2})
        batch_ok = import_proposals_json(goal, [child_prop], db=db)
        assert len(batch_ok.accepted) == 1
        child_hyp = batch_ok.accepted[0]
        assert child_hyp.subgoal_id == sg.subgoal_id

        grandchild_prop = {
            "kind": "REFINEMENT",
            "parent_id": child_hyp.hypothesis_id,
            "title": "Grandchild gen 3",
            "statement": "Exceeds cap=2",
            "mechanism": "sg_gen_grandchild_v3",
            "writable_paths": ["src/a.py"],
            "kill_criteria": ["f1_local < 0.10"],
            "screen_cmd": "python3 -m src.run_a",
            "acceptance_cmd": "pytest -q tests/test_a.py",
        }
        batch_cap2 = import_proposals_json(goal, [grandchild_prop], db=db)
        assert batch_cap2.accepted == []
        assert "SUBGOAL_GENERATION_CAP_REACHED" in batch_cap2.rejected[0]["reasons"][0]

        # When no DB SubGoalNode exists, hyp.params["max_auto_generations"] configures cap cleanly
        orphan_parent = _make_hyp(
            goal,
            "h_orphan_p",
            subgoal_id="sg_transient",
            params={"subgoal_id": "sg_transient", "max_auto_generations": 2},
        )
        db.register_hypothesis(orphan_parent)
        orphan_child = {
            "kind": "REFINEMENT",
            "parent_id": orphan_parent.hypothesis_id,
            "title": "Orphan gen 2",
            "statement": "Allowed under hyp.params cap=2",
            "mechanism": "orphan_gen2",
            "writable_paths": ["src/b.py"],
            "kill_criteria": ["tests_passed == 0"],
            "screen_cmd": "pytest -q",
            "acceptance_cmd": "pytest -q",
        }
        batch_orphan = import_proposals_json(goal, [orphan_child], db=db)
        assert len(batch_orphan.accepted) == 1


def test_apply_scientific_actions_all_five_portfolio_actions(tmp_path: pathlib.Path) -> None:
    goal = _make_goal(tmp_path)
    other_goal = _make_goal(tmp_path, goal_id="g_other")
    with StateDB(tmp_path / "state_home") as db:
        db.save_goal(goal)
        db.save_goal(other_goal)
        db.save_subgoal(_make_subgoal(other_goal, "sg_foreign"))

        # 1. SPAWN_SUBGOAL anti-filler rejections + valid spawn (ACTIVE + LOCKED_DEP + initial_slices)
        env_spawn = LeadDecisionEnvelope(
            role=ScientificRole.RESEARCH_DIRECTOR,
            goal_id=goal.goal_id,
            goal_version=goal.version,
            human_epoch=goal.human_epoch,
            actions=[
                LeadAction(
                    action=ActionKind.SPAWN_SUBGOAL,
                    parameters={
                        "subgoal_id": "sg_filler_docs",
                        "title": "Doc update track",
                        "direction_thesis": "Sync docs",
                        "direct_contribution": "Update README and record commit hash",
                        "local_metric": "docs_synced",
                    },
                ),
                LeadAction(
                    action=ActionKind.SPAWN_SUBGOAL,
                    parameters={
                        "subgoal_id": "sg_missing_metric",
                        "title": "No metric track",
                        "direction_thesis": "Explore X",
                        "direct_contribution": "Implement new kernel",
                        "local_metric": "",
                    },
                ),
                LeadAction(
                    action=ActionKind.SPAWN_SUBGOAL,
                    parameters={
                        "subgoal_id": "sg_track_a",
                        "title": "Track A: Sparse KV",
                        "direction_thesis": "Quantize host KV pages to INT4",
                        "direct_contribution": "Implement symmetric INT4 KV tiering kernel",
                        "local_metric": "kv_saved_pct",
                        "baseline_value": 20.0,
                        "target_value": 55.0,
                        "direction": "maximize",
                        "kill_criteria": ["kv_saved_pct < 10.0"],
                        "priority_weight": 2.0,
                        "max_tokens_budget": 60_000,
                        "spec_paths": ["specs/kv.md"],
                        "initial_slices": [
                            {
                                "kind": "COMPETING_ROUTE",
                                "title": "INT4 host pack slice",
                                "statement": "Pack KV cache into symmetric INT4 blocks",
                                "mechanism": "int4_pack_v1",
                                "writable_paths": ["src/kv_pack.py"],
                                "screen_cmd": "python3 -m src.bench_kv",
                                "acceptance_cmd": "pytest -q tests/test_kv.py",
                            }
                        ],
                    },
                ),
                LeadAction(
                    action=ActionKind.SPAWN_SUBGOAL,
                    parameters={
                        "subgoal_id": "sg_track_b",
                        "title": "Track B: Fused Attention",
                        "direction_thesis": "Fuse sparse attention decode kernel",
                        "direct_contribution": "Implement fused sparse attention decode kernel",
                        "local_metric": "decode_latency_ms",
                        "baseline_value": 18.0,
                        "target_value": 10.0,
                        "direction": "minimize",
                        "priority_weight": 1.0,
                        "max_tokens_budget": 30_000,
                    },
                ),
                LeadAction(
                    action=ActionKind.SPAWN_SUBGOAL,
                    parameters={
                        "subgoal_id": "sg_track_c_dep",
                        "title": "Track C: End-to-end Tuning",
                        "direction_thesis": "Tune decode threshold after Track A seals",
                        "direct_contribution": "Implement adaptive threshold calibration over Track A",
                        "local_metric": "e2e_accuracy",
                        "depends_on_subgoals": ["sg_track_a"],
                        "priority_weight": 1.0,
                        "max_tokens_budget": 20_000,
                    },
                ),
            ],
        )
        sum_spawn = apply_scientific_actions(db, goal, env_spawn)
        assert sum_spawn["spawned_subgoal_ids"] == ["sg_track_a", "sg_track_b", "sg_track_c_dep"]
        assert len(sum_spawn["rejected"]) == 2
        assert "ANTI_FILLER" in sum_spawn["rejected"][0]["reason"]
        assert len(sum_spawn["accepted_hypothesis_ids"]) == 1

        sg_a = db.get_subgoal("sg_track_a")
        sg_b = db.get_subgoal("sg_track_b")
        sg_c = db.get_subgoal("sg_track_c_dep")
        assert sg_a is not None and sg_a.status == "ACTIVE"
        assert sg_a.track_branch == f"track/{goal.project}/sg_track_a"
        assert sg_a.base_sha == goal.baseline_sha
        assert sg_b is not None and sg_b.status == "ACTIVE"
        assert sg_c is not None and sg_c.status == "LOCKED_DEP"

        slice_a = db.get_hypothesis(sum_spawn["accepted_hypothesis_ids"][0])
        assert slice_a is not None
        assert slice_a.subgoal_id == "sg_track_a"
        assert slice_a.params["subgoal_local_metric"] == "kv_saved_pct"

        # 2. RETUNE_SUBGOAL updates fields, resets AWAITING_APEX -> ACTIVE, bumps plan_epoch
        db.transition_subgoal_status("sg_track_b", "ACTIVE", SubGoalStatus.AWAITING_APEX.value)
        prev_epoch = int((db.get_goal(goal.goal_id) or goal).plan_epoch)
        env_retune = LeadDecisionEnvelope(
            role=ScientificRole.RESEARCH_DIRECTOR,
            goal_id=goal.goal_id,
            actions=[
                LeadAction(
                    action=ActionKind.RETUNE_SUBGOAL,
                    target_subgoal_id="sg_track_b",
                    parameters={
                        "local_metric": "p99_decode_ms",
                        "target_value": 12.0,
                        "baseline_value": 20.0,
                        "direction": "minimize",
                        "kill_criteria": ["p99_decode_ms > 30.0"],
                        "direction_thesis": "Optimize p99 decode latency with warp tiling",
                        "compute_target": "local_gpu",
                        "priority_weight": 1.5,
                        "max_tokens_budget": 45_000,
                    },
                )
            ],
        )
        sum_retune = apply_scientific_actions(db, db.get_goal(goal.goal_id) or goal, env_retune)
        assert sum_retune["retuned_subgoal_ids"] == ["sg_track_b"]
        sg_b_after = db.get_subgoal("sg_track_b")
        assert sg_b_after is not None
        assert sg_b_after.status == "ACTIVE"
        assert sg_b_after.local_metric == "p99_decode_ms"
        assert sg_b_after.target_value == 12.0
        assert sg_b_after.baseline_value == 20.0
        assert sg_b_after.compute_target == "local_gpu"
        assert sg_b_after.priority_weight == 1.5
        assert sg_b_after.max_tokens_budget == 45_000
        fresh_goal = db.get_goal(goal.goal_id)
        assert fresh_goal is not None and fresh_goal.plan_epoch == prev_epoch + 1

        # 3. SYNC_SUBGOAL_BASE updates base_sha & unlocks LOCKED_DEP -> ACTIVE once upstream is SEALED
        db.transition_subgoal_status("sg_track_a", "ACTIVE", SubGoalStatus.SEALED.value, sealed_sha="sealed_sha_a1")
        env_sync = LeadDecisionEnvelope(
            role=ScientificRole.RESEARCH_DIRECTOR,
            goal_id=goal.goal_id,
            actions=[
                LeadAction(
                    action=ActionKind.SYNC_SUBGOAL_BASE,
                    target_subgoal_id="sg_track_c_dep",
                    parameters={"upstream_subgoal_id": "sg_track_a"},
                )
            ],
        )
        sum_sync = apply_scientific_actions(db, fresh_goal, env_sync)
        assert sum_sync["synced_subgoal_ids"] == ["sg_track_c_dep"]
        sg_c_after = db.get_subgoal("sg_track_c_dep")
        assert sg_c_after is not None
        assert sg_c_after.base_sha == "sealed_sha_a1"
        assert sg_c_after.status == "ACTIVE"

        # 4. MERGE_SUBGOALS creates INTEGRATION_MERGE SubGoal + registers & enqueues integration hypothesis
        hyp_b = _make_hyp(goal, "hyp_b1", subgoal_id="sg_track_b", head_sha="head_sha_b1")
        db.register_hypothesis(hyp_b)
        rt = WorkflowRuntime(db)
        env_merge = LeadDecisionEnvelope(
            role=ScientificRole.RESEARCH_DIRECTOR,
            goal_id=goal.goal_id,
            actions=[
                LeadAction(
                    action=ActionKind.MERGE_SUBGOALS,
                    parameters={
                        "subgoal_id": "sg_merge_ab",
                        "source_subgoal_ids": ["sg_track_a", "sg_track_b"],
                        "acceptance_cmd": "pytest -q tests/test_integ.py",
                    },
                )
            ],
        )
        sum_merge = apply_scientific_actions(db, fresh_goal, env_merge, workflow_runtime=rt)
        assert sum_merge["merged_subgoal_ids"] == ["sg_merge_ab"]
        assert len(sum_merge["accepted_hypothesis_ids"]) == 1
        assert len(sum_merge["emitted_task_ids"]) >= 1
        sg_merge = db.get_subgoal("sg_merge_ab")
        assert sg_merge is not None
        assert sg_merge.kind == SubGoalKind.INTEGRATION_MERGE
        assert sg_merge.status == "ACTIVE"
        assert sg_merge.source_subgoal_ids == ["sg_track_a", "sg_track_b"]
        assert sg_merge.base_sha == "sealed_sha_a1"
        merge_hyp = db.get_hypothesis(sum_merge["accepted_hypothesis_ids"][0])
        assert merge_hyp is not None
        assert merge_hyp.subgoal_id == "sg_merge_ab"
        assert merge_hyp.params["merge_source_shas"] == ["sealed_sha_a1", "head_sha_b1"]
        assert merge_hyp.params["merge_subgoal_ids"] == ["sg_track_a", "sg_track_b"]
        assert merge_hyp.params["requires_llm_code"] is True
        assert merge_hyp.branch_name not in (sg_a.track_branch, sg_b_after.track_branch)

        # 5. KILL_AND_REALLOCATE kills target, cancels its workflows & tasks, reallocates weight & budget
        wf_b = WorkflowNode(
            workflow_id="wf_kill_b",
            goal_id=goal.goal_id,
            hypothesis_id=hyp_b.hypothesis_id,
            primitive="direct_hq",
            status="RUNNING",
        )
        db.save_workflow(wf_b)
        tsk_b = StageTask(
            task_id="tsk_kill_b",
            idempotency_key="idem_kill_b",
            goal_id=goal.goal_id,
            hypothesis_id=hyp_b.hypothesis_id,
            workflow_id=wf_b.workflow_id,
            stage=StageKind.CODE,
            deps=[],
            status="READY",
        )
        db.enqueue_task(tsk_b)

        env_kill = LeadDecisionEnvelope(
            role=ScientificRole.RESEARCH_DIRECTOR,
            goal_id=goal.goal_id,
            actions=[
                LeadAction(
                    action=ActionKind.KILL_AND_REALLOCATE,
                    target_subgoal_id="sg_track_b",
                    parameters={"reallocate_to": ["sg_track_c_dep"]},
                )
            ],
        )
        sum_kill = apply_scientific_actions(db, fresh_goal, env_kill)
        assert sum_kill["killed_subgoal_ids"] == ["sg_track_b"]
        sg_b_killed = db.get_subgoal("sg_track_b")
        assert sg_b_killed is not None and sg_b_killed.status == "KILLED"
        assert sg_b_killed.priority_weight == 0.0
        assert sg_b_killed.max_tokens_budget == 0
        assert db.get_workflow("wf_kill_b").status == "CANCELLED"
        assert db.get_task("tsk_kill_b").status == "CANCELLED"
        sg_c_boosted = db.get_subgoal("sg_track_c_dep")
        assert sg_c_boosted is not None
        assert sg_c_boosted.priority_weight == pytest.approx(2.5)
        assert sg_c_boosted.max_tokens_budget == 65_000

        # Cross-goal SubGoal ownership guard rejects foreign subgoal reference cleanly
        env_foreign = LeadDecisionEnvelope(
            role=ScientificRole.RESEARCH_DIRECTOR,
            goal_id=goal.goal_id,
            actions=[
                LeadAction(
                    action=ActionKind.KILL_AND_REALLOCATE,
                    target_subgoal_id="sg_foreign",
                )
            ],
        )
        sum_foreign = apply_scientific_actions(db, fresh_goal, env_foreign)
        assert sum_foreign["cross_goal_violation"] is True
        assert db.get_subgoal("sg_foreign").status == "ACTIVE"


def test_subgoal_digest_isolation_and_sandtable_compute_snapshot(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goal = _make_goal(tmp_path)
    monkeypatch.setenv("HOME", str(tmp_path))
    cache_dir = tmp_path / ".cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "slurm_sinfo.json").write_text(
        json.dumps({"partition": "gpu", "nodes_idle": 3}),
        encoding="utf-8",
    )
    (cache_dir / "slurm_queue_state.json").write_text(
        json.dumps({"pending_jobs": 1, "running_jobs": 2}),
        encoding="utf-8",
    )
    cloud_dir = tmp_path / "cloud_state"
    cloud_dir.mkdir(parents=True, exist_ok=True)
    (cloud_dir / "cloud_slot_state_1.json").write_text(
        json.dumps({"runtime_id": "rt-g4", "status": "ready", "available_slots": 2, "active_jobs": 1}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CONJECTRA_CLOUD_SLOT_STATE_PATH", str(cloud_dir / "cloud_slot_state_1.json"))

    with StateDB(tmp_path / "state_home") as db:
        db.save_goal(goal)
        sg_alpha = _make_subgoal(
            goal,
            "sg_alpha",
            local_metric="alpha_score",
            baseline_value=1.0,
            target_value=2.0,
            spec_paths=["specs/alpha.md"],
        )
        sg_beta = _make_subgoal(
            goal,
            "sg_beta",
            local_metric="beta_secret_metric",
            baseline_value=10.0,
            target_value=5.0,
            direction="minimize",
            status="AWAITING_APEX",
            spec_paths=["specs/beta.md"],
        )
        db.save_subgoal(sg_alpha)
        db.save_subgoal(sg_beta)

        h_alpha = _make_hyp(goal, "h_alpha_1", subgoal_id="sg_alpha", head_sha="sha_a1")
        h_beta = _make_hyp(goal, "h_beta_1", subgoal_id="sg_beta")
        h_beta.status = "BLOCKED_ENV"
        db.register_hypothesis(h_alpha)
        db.register_hypothesis(h_beta)

        db.record_receipt(
            EvidenceReceipt(
                receipt_id="rcpt_alpha_1",
                task_id="t_alpha_1",
                goal_id=goal.goal_id,
                hypothesis_id=h_alpha.hypothesis_id,
                stage=StageKind.EXP_SCREEN,
                base_sha=goal.baseline_sha,
                head_sha="sha_a1",
                command="python3 run_alpha.py",
                exit_code=0,
                wall_sec=5.0,
                max_rss_mb=128.0,
                in_tokens=1100,
                out_tokens=400,
                model_used="gemini:gemini-pro",
                process_status=ProcessStatus.EXIT_OK,
                engineering_verdict=EngineeringVerdict.PASS,
                scientific_verdict=ScientificVerdict.CONFIRMED,
                audit_verdict=AuditVerdict.NOT_REQUIRED,
                tests_passed=3,
                tests_total=3,
                metrics={"alpha_score": {"value": 1.65, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"}},
                raw_artifact_paths=[],
                evidence_validity=EvidenceValidity.VALID_EMPIRICAL,
            )
        )
        db.record_receipt(
            EvidenceReceipt(
                receipt_id="rcpt_beta_1",
                task_id="t_beta_1",
                goal_id=goal.goal_id,
                hypothesis_id=h_beta.hypothesis_id,
                stage=StageKind.EXP_SCREEN,
                base_sha=goal.baseline_sha,
                head_sha=goal.baseline_sha,
                command="python3 run_beta.py",
                exit_code=1,
                wall_sec=2.0,
                max_rss_mb=64.0,
                in_tokens=300,
                out_tokens=100,
                model_used="gemini:gemini-pro",
                process_status=ProcessStatus.EXIT_NONZERO,
                engineering_verdict=EngineeringVerdict.FAIL_TEST,
                scientific_verdict=ScientificVerdict.UNKNOWN,
                audit_verdict=AuditVerdict.NOT_REQUIRED,
                tests_passed=0,
                tests_total=1,
                metrics={"beta_secret_metric": {"value": 12.0, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"}},
                raw_artifact_paths=[],
                failure_reason="BETA_CROSS_TRACK_NOISE_MARKER",
            )
        )
        db.upsert_knowledge_item(
            goal.goal_id,
            "valid_negative",
            "k_beta",
            {"detail": "BETA_SECRET_LESSON"},
            hypothesis_id=h_beta.hypothesis_id,
        )
        db.upsert_knowledge_item(
            goal.goal_id,
            "confirmed_finding",
            "k_alpha",
            {"detail": "ALPHA_OWN_LESSON"},
            hypothesis_id=h_alpha.hypothesis_id,
        )

        # build_subgoal_digest strictly <3KB and contains zero cross-track noise from sg_beta
        digest_a = build_subgoal_digest(db, goal, sg_alpha, max_bytes=MAX_SUBGOAL_DIGEST_BYTES)
        raw_digest = json.dumps(digest_a, sort_keys=True, ensure_ascii=False)
        assert len(raw_digest.encode("utf-8")) <= 3072
        assert digest_a["subgoal_id"] == "sg_alpha"
        assert digest_a["local_metric"] == "alpha_score"
        assert digest_a["target_value"] == 2.0
        assert digest_a["spec_paths"] == ["specs/alpha.md"]
        assert len(digest_a["slices"]) == 1
        assert digest_a["slices"][0]["id"] == "h_alpha_1"
        assert digest_a["slices"][0]["metric"] == 1.65
        assert "ALPHA_OWN_LESSON" in raw_digest
        for forbidden_noise in ("sg_beta", "h_beta_1", "beta_secret_metric", "BETA_CROSS_TRACK_NOISE_MARKER", "BETA_SECRET_LESSON"):
            assert forbidden_noise not in raw_digest

        # build_portfolio_sandtable summarizes ALL subgoals + 0-Token Live Compute Snapshot
        sandtable = build_portfolio_sandtable(db, goal)
        by_sg = {row["subgoal_id"]: row for row in sandtable["subgoals"]}
        assert set(by_sg.keys()) == {"sg_alpha", "sg_beta"}
        assert by_sg["sg_alpha"]["best_metric"] == 1.65
        assert by_sg["sg_alpha"]["token_spend"] == 1500
        assert by_sg["sg_beta"]["best_metric"] == 12.0
        assert "h_beta_1:BLOCKED_ENV" in by_sg["sg_beta"]["blocker"]
        assert "BETA_CROSS_TRACK_NOISE_MARKER" in by_sg["sg_beta"]["blocker"]

        snap = sandtable["compute_snapshot"]
        assert snap["cloud"]["available_slots"] == 2
        assert snap["cloud"]["stale"] is False
        assert snap["local"]["cpus"] >= 1
        assert snap["slurm"]["slurm_sinfo"]["present"] is True
        assert snap["slurm"]["slurm_sinfo"]["data"]["nodes_idle"] == 3
        assert snap["slurm"]["slurm_queue_state"]["present"] is True
        assert snap["slurm"]["slurm_queue_state"]["data"]["running_jobs"] == 2


def test_two_tier_track_lead_and_apex_director_decision_engine(tmp_path: pathlib.Path) -> None:
    goal = _make_goal(tmp_path)
    with StateDB(tmp_path / "state_home") as db:
        db.save_goal(goal)
        eng = DecisionEngine(db)

        # 1. ApexDirector INITIAL_SUBGOAL_SPLIT prompt includes Portfolio Sandtable & spawns SubGoal in dry-run
        req_split = eng.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_DIRECTOR,
            tier="L2",
            trigger_kind="INITIAL_SUBGOAL_SPLIT",
            urgency="IMMEDIATE",
        )
        assert req_split["role"] == ScientificRole.RESEARCH_DIRECTOR.value
        assert "portfolio_sandtable" in req_split["payload"]

        apex_prompt = eng._build_role_prompt(
            req_split,
            goal,
            db.list_hypotheses(goal_id=goal.goal_id),
            db.list_receipts(goal_id=goal.goal_id),
        )
        assert "ApexDirector" in apex_prompt
        assert "SPAWN_SUBGOAL" in apex_prompt
        assert "MERGE_SUBGOALS" in apex_prompt
        assert "compute_snapshot" in apex_prompt

        res_split = eng.process_request(req_split["req_id"], dry_run_llm=True)
        assert res_split["status"] == "RESOLVED"
        spawned_ids = res_split["resolution"]["action_summary"]["spawned_subgoal_ids"]
        assert len(spawned_ids) == 1
        sg1_id = spawned_ids[0]

        # Add a sibling subgoal to verify TrackLead isolation
        db.save_subgoal(_make_subgoal(goal, "sg_sibling", local_metric="sib_metric"))
        h_sib = _make_hyp(goal, "h_sib_only", subgoal_id="sg_sibling")
        db.register_hypothesis(h_sib)

        # 2. TrackLead scoped to sg1_id persists subgoal_id, hides sibling, blocks sibling mutation & Apex actions
        req_tl = eng.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            tier="L1",
            subgoal_id=sg1_id,
            trigger_kind="TRACK_WAVE_REVIEW",
            urgency="COALESCED",
        )
        assert req_tl["subgoal_id"] == sg1_id
        tl_prompt = eng._build_role_prompt(
            req_tl,
            goal,
            db.list_hypotheses(goal_id=goal.goal_id),
            db.list_receipts(goal_id=goal.goal_id),
        )
        assert "TrackLead" in tl_prompt
        assert sg1_id in tl_prompt
        assert "FORBIDDEN from mutating" in tl_prompt
        assert "REQUEST_DIRECTOR_ADJUDICATION" in tl_prompt
        assert "h_sib_only" not in tl_prompt

        # TrackLead attempting to spawn a SubGoal or prune sibling hypothesis must be rejected cleanly
        def _rogue_track_lead(prov: Any, prompt: str) -> dict[str, Any]:
            env = {
                "role": "research_lead",
                "goal_id": goal.goal_id,
                "goal_version": goal.version,
                "human_epoch": goal.human_epoch,
                "uncertainty_summary": "Trying forbidden cross-track and apex actions",
                "actions": [
                    {
                        "action": "PRUNE_BRANCH",
                        "target_hypothesis_id": "h_sib_only",
                        "rationale": "Cross-track prune attempt",
                    },
                    {
                        "action": "SPAWN_SUBGOAL",
                        "parameters": {
                            "title": "Rogue",
                            "direction_thesis": "Rogue",
                            "direct_contribution": "Implement rogue feature",
                            "local_metric": "x",
                        },
                    },
                    {
                        "action": "REQUEST_DIRECTOR_ADJUDICATION",
                        "target_subgoal_id": sg1_id,
                        "rationale": "Need Apex decision",
                    },
                ],
            }
            return {
                "exit_code": 0,
                "response": json.dumps(env),
                "in_tokens": 50,
                "out_tokens": 30,
                "model_used": prov.label(),
            }

        res_tl = eng.process_request(req_tl["req_id"], llm_caller=_rogue_track_lead)
        assert res_tl["status"] == "RESOLVED"
        assert res_tl["resolution"]["subgoal_id"] == sg1_id
        tl_sum = res_tl["resolution"]["action_summary"]
        assert tl_sum["pruned_hypothesis_ids"] == []
        assert tl_sum["spawned_subgoal_ids"] == []
        assert len(tl_sum["director_requests"]) == 1
        assert len(tl_sum["rejected"]) == 2
        assert db.get_hypothesis("h_sib_only").status == "PENDING"

        # 3. ApexDirector APEX_PORTFOLIO_REVIEW with 2 sealed subgoals -> dry-run emits MERGE_SUBGOALS
        db.transition_subgoal_status(sg1_id, "ACTIVE", SubGoalStatus.SEALED.value, sealed_sha="sha_sg1")
        db.transition_subgoal_status("sg_sibling", "ACTIVE", SubGoalStatus.SEALED.value, sealed_sha="sha_sib")
        req_apex = eng.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_DIRECTOR,
            tier="L2",
            trigger_kind="APEX_PORTFOLIO_REVIEW",
            urgency="IMMEDIATE",
        )
        res_apex = eng.process_request(req_apex["req_id"], dry_run_llm=True)
        assert res_apex["status"] == "RESOLVED"
        assert len(res_apex["resolution"]["action_summary"]["merged_subgoal_ids"]) == 1
        assert len(res_apex["resolution"]["action_summary"]["accepted_hypothesis_ids"]) == 1


def test_portfolio_boundary_cases_even_reallocation_and_partial_deps(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    goal = _make_goal(tmp_path, goal_id="g_v4_bounds")
    monkeypatch.setenv("HOME", str(tmp_path / "empty_home"))
    (tmp_path / "empty_home" / ".cache").mkdir(parents=True, exist_ok=True)
    (tmp_path / "empty_home" / ".cache" / "slurm_sinfo.json").write_text("{not-json", encoding="utf-8")

    with StateDB(tmp_path / "state_home") as db:
        db.save_goal(goal)
        db.save_subgoal(_make_subgoal(goal, "sg_d1", status="SEALED", sealed_sha="sha_d1", priority_weight=1.0, max_tokens_budget=20_000))
        db.save_subgoal(_make_subgoal(goal, "sg_d2", status="ACTIVE", priority_weight=1.0, max_tokens_budget=20_000))
        db.save_subgoal(_make_subgoal(goal, "sg_d3", status="ACTIVE", priority_weight=1.0, max_tokens_budget=20_000))
        db.save_subgoal(
            _make_subgoal(
                goal,
                "sg_multi_dep",
                status="LOCKED_DEP",
                depends_on=["sg_d1", "sg_d2"],
                priority_weight=1.0,
                max_tokens_budget=10_000,
            )
        )
        db.save_subgoal(_make_subgoal(goal, "sg_kill_src", status="ACTIVE", priority_weight=2.0, max_tokens_budget=25_001))

        # 1. Partial dependency sync updates base_sha but keeps LOCKED_DEP while sg_d2 is still ACTIVE
        env_partial = parse_lead_decision_payload(
            json.dumps(
                {
                    "role": "research_director",
                    "goal_id": goal.goal_id,
                    "actions": [
                        {
                            "action": "SYNC_SUBGOAL_BASE",
                            "target_subgoal_id": "sg_multi_dep",
                            "parameters": {"upstream_subgoal_id": "sg_d1"},
                        },
                        {
                            "action": "MERGE_SUBGOALS",
                            "parameters": {"source_subgoal_ids": ["sg_d2", "sg_d3"]},
                        },
                        {
                            "action": "KILL_AND_REALLOCATE",
                            "target_subgoal_id": "sg_kill_src",
                        },
                    ],
                }
            ),
            goal=goal,
            default_role=ScientificRole.RESEARCH_DIRECTOR,
        )
        res = apply_scientific_actions(db, goal, env_partial)
        assert res["synced_subgoal_ids"] == ["sg_multi_dep"]
        assert res["merged_subgoal_ids"] == []
        assert any("sealed_sha or head_sha" in r.get("reason", "") for r in res["rejected"])
        assert res["killed_subgoal_ids"] == ["sg_kill_src"]

        sg_m = db.get_subgoal("sg_multi_dep")
        assert sg_m is not None
        assert sg_m.base_sha == "sha_d1"
        assert sg_m.status == "LOCKED_DEP"

        # 2. Even reallocation across remaining ACTIVE siblings (sg_d2 and sg_d3 only)
        sg_d2 = db.get_subgoal("sg_d2")
        sg_d3 = db.get_subgoal("sg_d3")
        assert sg_d2 is not None and sg_d3 is not None
        assert sg_d2.priority_weight == pytest.approx(2.0)
        assert sg_d3.priority_weight == pytest.approx(2.0)
        assert sg_d2.max_tokens_budget + sg_d3.max_tokens_budget == 40_000 + 25_001

        # 3. Sandtable tolerates missing/invalid slurm cache and digest respects strict small max_bytes
        sandtable = build_portfolio_sandtable(db, goal)
        assert sandtable["compute_snapshot"]["slurm"]["slurm_sinfo"]["error"] == "invalid_json"
        assert sandtable["compute_snapshot"]["slurm"]["slurm_queue_state"]["present"] is False

        tiny_digest = build_subgoal_digest(db, goal, sg_d2, max_bytes=420)
        assert len(json.dumps(tiny_digest, sort_keys=True, ensure_ascii=False).encode("utf-8")) <= 420
