"""Unit and live-snapshot migration tests for V3 schema, plans, failure domains, and quorum."""

from __future__ import annotations

import concurrent.futures
import json
import pathlib
import sqlite3

import pytest

from conjectra.gates import (
    check_no_internal_ids,
    classify_failure_domain,
    evaluate_kill_criteria,
    is_authenticated_negative_empirical_outcome,
    is_valid_empirical_outcome,
    map_timeout_verdict,
    parse_structured_verdict,
    validate_experiment_plan_gate,
    validate_receipt_epoch,
)
from conjectra.planning import (
    INITIAL_QUORUM_DEADLINE_S,
    TIEBREAKER_DEADLINE_S,
    PlanLifecycleManager,
    ReviewQuorumReducer,
    build_design_review_prompt,
    build_impl_review_prompt,
    build_result_review_prompt,
    build_synthesis_review_prompt,
    classify_review_tier,
    parse_review_output,
)
from conjectra.state_db import FencedStaleWriteError, StateDB
from conjectra.types import (
    ActionKind,
    AuditVerdict,
    ConstraintStrength,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    ExecutionPlan,
    ExperimentPlan,
    FailureDomain,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    LeadAction,
    LeadDecision,
    LeadDecisionEnvelope,
    ModelPolicy,
    ProcessStatus,
    ResearchGoal,
    ResearchPlan,
    ResolvedModelRoute,
    ResourceSpec,
    ReviewKind,
    ReviewReceipt,
    ReviewTier,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
)


def _sample_goal(repo_path: str = "/tmp/sample-repo") -> ResearchGoal:
    return ResearchGoal(
        goal_id="goal-v3-test",
        project="projgamma",
        title="PROJGAMMA Spectral Bound",
        direction="Derive exact spectral bound",
        primary_metric="spectral_gap",
        baseline_value=0.42,
        baseline_sha="abcdef1234567",
        repo_path=repo_path,
        kill_criteria=["spectral_gap < 0.30"],
        banned_paths=["/secret/freeze"],
        writable_prefixes=["src/", "tests/"],
        max_tokens_budget=200_000,
        version=2,
        human_epoch=3,
        routing_policy_nl="PROJGAMMA 理论创新用 Opus xhigh，重大理论必须跨家族审查",
    )


def _sample_hyp(goal_id: str = "goal-v3-test", *, tags: list[str] | None = None) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id="hyp-v3-1",
        goal_id=goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Chebyshev Spectral Route",
        statement="Chebyshev polynomial filter improves spectral_gap above 0.50",
        mechanism="3-term recurrence operator",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.85,
        writable_paths=["src/spectral.py", "tests/test_spectral.py"],
        params={"degree": 4},
        kill_criteria=["spectral_gap < 0.30"],
        acceptance_cmd="pytest -q tests/test_spectral.py",
        screen_cmd="python -m src.eval_screen",
        confirm_cmd="python -m src.eval_confirm",
        branch_name="arr/projgamma/hyp-v3-1",
        base_sha="abcdef1234567",
        tags=tags if tags is not None else ["theory"],
    )


def _sample_receipt(
    *,
    stage: StageKind = StageKind.EXP_SCREEN,
    process_status: ProcessStatus = ProcessStatus.EXIT_OK,
    engineering_verdict: EngineeringVerdict = EngineeringVerdict.PASS,
    scientific_verdict: ScientificVerdict = ScientificVerdict.CONFIRMED,
    audit_verdict: AuditVerdict = AuditVerdict.NOT_REQUIRED,
    evidence_validity: EvidenceValidity = EvidenceValidity.VALID_EMPIRICAL,
    failure_domain: FailureDomain = FailureDomain.NONE,
    metrics: dict | None = None,
    failure_reason: str = "",
    goal_version: int = 2,
    human_epoch: int = 3,
    exit_code: int = 0,
) -> EvidenceReceipt:
    return EvidenceReceipt(
        receipt_id="rcpt-v3-1",
        task_id="tsk-v3-1",
        goal_id="goal-v3-test",
        hypothesis_id="hyp-v3-1",
        stage=stage,
        base_sha="abcdef1234567",
        head_sha="1234567abcdef",
        command="python -m src.eval_screen",
        exit_code=exit_code,
        wall_sec=12.5,
        max_rss_mb=256.0,
        in_tokens=1200,
        out_tokens=400,
        model_used="claude:claude-opus-5-5:xhigh",
        process_status=process_status,
        engineering_verdict=engineering_verdict,
        scientific_verdict=scientific_verdict,
        audit_verdict=audit_verdict,
        tests_passed=8,
        tests_total=8,
        metrics={"spectral_gap": 0.54} if metrics is None else metrics,
        raw_artifact_paths=["/tmp/art.json"],
        failure_reason=failure_reason,
        evidence_validity=evidence_validity,
        goal_version=goal_version,
        human_epoch=human_epoch,
        failure_domain=failure_domain,
        effort_used="xhigh",
    )


# --------------------------- 1. Types & Round-Trip Serialization


def test_v3_types_roundtrip_and_resource_spec_backward_compat() -> None:
    # Default ResourceSpec must NOT inject string/dict keys into V2 dict representation
    default_res = ResourceSpec(cpus=2, ram_mb=2048, llm_pool="llm_claude")
    assert set(default_res.to_dict().keys()) == {
        "cpus",
        "ram_mb",
        "gpu",
        "llm_pool",
        "eda_sim",
        "cloud_slot",
        "wall_timeout_s",
    }

    route = ResolvedModelRoute(
        backend_mode="claude_cli",
        execution_host="local",
        multipool_host_arg="remote",
        pool="claude",
        canonical_model="claude-opus-5-5",
        requested_effort="xhigh",
        wire_effort="xhigh",
        family="anthropic",
        logical_pool="llm_claude",
        hq_resource="llmclaude",
        strength=ConstraintStrength.HARD,
        matched_rule_id="rule_projgamma_theory",
    )
    assert route.canonical_spec() == "claude:claude-opus-5-5:xhigh"
    assert ResolvedModelRoute.from_dict(route.to_dict()) == route

    v3_res = ResourceSpec(
        cpus=1,
        hq_llm_resource="llmclaude",
        pinned_route=route.to_dict(),
    )
    restored_res = ResourceSpec.from_dict(v3_res.to_dict())
    assert restored_res.hq_llm_resource == "llmclaude"
    assert restored_res.pinned_route is not None
    assert restored_res.pinned_route["wire_effort"] == "xhigh"

    policy = ModelPolicy(
        source_nl="PROJGAMMA theory uses Opus xhigh",
        project="projgamma",
        rules=[{"id": "r1", "role": "theorist", "model": "claude-opus-5-5", "effort": "xhigh"}],
        review_overrides={"theory": "RED_TEAM_TIEBREAKER"},
        default_strength=ConstraintStrength.HARD,
    )
    assert ModelPolicy.from_dict(policy.to_dict()) == policy

    # ExperimentPlan clamps gate_timeout_s strictly into [30, 180]
    eplan_high = ExperimentPlan(
        exp_plan_id="ep1",
        goal_id="g1",
        hypothesis_id="h1",
        falsifiable_claim="claim",
        gate_timeout_s=900,
    )
    assert eplan_high.gate_timeout_s == 180
    eplan_low = ExperimentPlan(
        exp_plan_id="ep2",
        goal_id="g1",
        hypothesis_id="h1",
        falsifiable_claim="claim",
        gate_timeout_s=5,
    )
    assert eplan_low.gate_timeout_s == 30

    envelope = LeadDecisionEnvelope(
        role=ScientificRole.RESEARCH_LEAD,
        goal_id="goal-v3-test",
        goal_version=2,
        human_epoch=3,
        uncertainty_summary="Need discriminating ablation on Chebyshev order",
        actions=[
            LeadAction(
                action=ActionKind.ADD_DISCRIMINATING_EXPERIMENT,
                target_hypothesis_id="hyp-v3-1",
                uncertainty_id="u1",
                rationale="Isolate boundary condition effect",
                workflow_primitive="x-boost",
                review_tier=ReviewTier.RED_TEAM_TIEBREAKER,
            ),
            LeadAction(
                action=ActionKind.REPAIR_IMPLEMENTATION,
                target_hypothesis_id="hyp-v3-1",
            ),
        ],
    )
    assert LeadDecision is LeadDecisionEnvelope
    assert LeadDecision.from_dict(envelope.to_dict()) == envelope


# ------------------ 2. Cross-Process SQLite V3 Migration & Live Snapshot


def test_v3_migration_idempotency_concurrency_and_legacy_role_derivation(
    tmp_path: pathlib.Path,
) -> None:
    arr_home = tmp_path / "arr_home"
    arr_home.mkdir()
    db_path = arr_home / "state.db"

    # Simulate a pre-V3 database (`user_version = 0`) with legacy L1 & L2 decision_requests
    raw_conn = sqlite3.connect(str(db_path))
    raw_conn.executescript(
        """
        CREATE TABLE decision_requests (
            req_id TEXT PRIMARY KEY,
            goal_id TEXT NOT NULL,
            tier TEXT NOT NULL,
            trigger_kind TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'PENDING',
            goal_version INTEGER NOT NULL DEFAULT 1,
            human_epoch INTEGER NOT NULL DEFAULT 0,
            coalesce_key TEXT NOT NULL DEFAULT '',
            available_ts REAL NOT NULL,
            payload_json TEXT NOT NULL,
            resolution_json TEXT NOT NULL DEFAULT '{}',
            hq_job_id INTEGER,
            created_ts REAL NOT NULL,
            updated_ts REAL NOT NULL,
            resolved_ts REAL
        );
        INSERT INTO decision_requests (
            req_id, goal_id, tier, trigger_kind, status,
            goal_version, human_epoch, coalesce_key, available_ts,
            payload_json, resolution_json, hq_job_id, created_ts, updated_ts
        ) VALUES
            ('dreq_legacy_l2', 'g1', 'L2', 'ESCALATION', 'PENDING', 1, 0, '', 100.0, '{"reason":"stuck"}', '{}', NULL, 100.0, 100.0),
            ('dreq_legacy_l1', 'g1', 'L1', 'EVIDENCE_UPDATE', 'PENDING', 1, 0, '', 101.0, '{"reason":"done"}', '{}', NULL, 101.0, 101.0);
        """
    )
    raw_conn.close()

    # Open 6 concurrent StateDB instances against the same pre-V3 database
    def _open_and_check(_idx: int) -> int:
        with StateDB(arr_home) as db:
            ver = db._conn.execute("PRAGMA user_version").fetchone()[0]
            return int(ver)

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        versions = list(pool.map(_open_and_check, range(6)))
    assert versions == [8, 8, 8, 8, 8, 8]

    with StateDB(arr_home) as db:
        req_l2 = db.get_decision_request("dreq_legacy_l2")
        req_l1 = db.get_decision_request("dreq_legacy_l1")
        assert req_l2 is not None and req_l2["role"] == ScientificRole.RESEARCH_DIRECTOR.value
        assert req_l1 is not None and req_l1["role"] == ScientificRole.RESEARCH_LEAD.value

        # Explicit V3 role in payload_json overrides legacy tier projection
        req_v3 = db.enqueue_decision_request(
            "g1",
            tier="L1",
            role=ScientificRole.THEORIST,
            model_spec="claude:claude-opus-5-5:xhigh",
            attempt=2,
            payload={"note": "derive lemma"},
        )
        assert req_v3["role"] == ScientificRole.THEORIST.value
        assert req_v3["attempt"] == 2
        assert req_v3["model_spec"] == "claude:claude-opus-5-5:xhigh"




# ----------- 3. Atomic Plans, RW Leases, and (goal_version, human_epoch, attempt, lease) Fencing


def test_plan_version_allocation_rw_leases_and_atomic_tx_fencing(
    tmp_path: pathlib.Path,
) -> None:
    with StateDB(tmp_path / "arr") as db:
        goal = _sample_goal(str(tmp_path / "repo"))
        db.save_goal(goal)
        hyp, created = db.register_hypothesis(_sample_hyp(goal.goal_id))
        assert created is True

        mgr = PlanLifecycleManager(db)
        rp1 = mgr.create_research_plan(goal, thesis="Initial spectral thesis", status="APPROVED")
        rp2 = mgr.create_research_plan(goal, thesis="Refined spectral thesis", status="APPROVED")
        assert rp1.plan_version == 1
        assert rp2.plan_version == 2
        reloaded_rp1 = db.get_research_plan(rp1.plan_id)
        assert reloaded_rp1 is not None and reloaded_rp1.status == "SUPERSEDED"
        assert db.get_latest_research_plan(goal.goal_id, status="APPROVED") == rp2

        ep1 = mgr.create_or_revise_experiment_plan(goal, hyp)
        ep2 = mgr.create_or_revise_experiment_plan(goal, hyp)
        assert ep1.plan_version == 1
        assert ep2.plan_version == 2
        assert db.get_hypothesis(hyp.hypothesis_id).exp_plan_id == ep2.exp_plan_id

        xp = mgr.create_execution_plan(
            goal,
            hyp,
            workflow_id="wf-1",
            primitive="x-boost",
            role_routes={
                "theorist": {"family": "anthropic", "canonical_model": "claude-opus-5-5"},
                "engineer_builder": {"family": "google", "canonical_model": "gemini-pro"},
            },
        )
        assert xp.review_tier == ReviewTier.RED_TEAM_TIEBREAKER
        assert xp.contributor_families == ["anthropic", "google"]

        # Shared branch_ro leases allow parallel reviewers while blocking writers!
        ro1 = db.acquire_branch_ro_lease(goal.goal_id, hyp.hypothesis_id, "deadbeef", "rev-0", "w0", now=100.0)
        ro2 = db.acquire_branch_ro_lease(goal.goal_id, hyp.hypothesis_id, "deadbeef", "rev-1", "w1", now=100.0)
        assert ro1 is not None and ro2 is not None
        assert db.has_active_branch_ro_leases(goal.goal_id, hyp.hypothesis_id, now=101.0) is True

        # Writer MUST be blocked while RO leases are active
        wr_blocked = db.acquire_branch_writer_lease(
            goal.goal_id, hyp.hypothesis_id, "code-1", "w2", now=101.0
        )
        assert wr_blocked is None

        # Release both RO leases -> writer succeeds, and new RO lease is blocked while writer active
        assert db.release_lease(ro1["lease_id"], now=102.0) is True
        assert db.release_lease(ro2["lease_id"], now=102.0) is True
        wr = db.acquire_branch_writer_lease(
            goal.goal_id, hyp.hypothesis_id, "code-1", "w2", now=103.0
        )
        assert wr is not None
        assert (
            db.acquire_branch_ro_lease(
                goal.goal_id, hyp.hypothesis_id, "deadbeef", "rev-2", "w3", now=104.0
            )
            is None
        )
        assert db.release_lease(wr["lease_id"], now=105.0) is True

        # Atomic `(goal_version, human_epoch, attempt, lease_id)` fencing on tasks & decisions
        task = db.enqueue_task(
            StageTask(
                task_id="tsk-fence-1",
                idempotency_key="ik-fence-1",
                goal_id=goal.goal_id,
                hypothesis_id=hyp.hypothesis_id,
                stage=StageKind.CODE,
                deps=[],
                attempt=1,
                goal_version=goal.version,
                human_epoch=goal.human_epoch,
                pinned_route={
                    "pool": "claude",
                    "canonical_model": "claude-sonnet-5-5",
                    "requested_effort": "medium",
                    "hq_resource": "llmclaude",
                },
            )
        )
        t_lease = db.acquire_branch_writer_lease(
            goal.goal_id, hyp.hypothesis_id, task.task_id, "runner-1", now=200.0
        )
        assert t_lease is not None

        running = db.fenced_transition_task(
            task.task_id,
            "READY",
            "RUNNING",
            expected_goal_version=2,
            expected_human_epoch=3,
            expected_attempt=1,
            expected_lease_id=t_lease["lease_id"],
            now=201.0,
        )
        assert running.status == "RUNNING"

        # Wrong attempt -> STALE_ATTEMPT
        with pytest.raises(FencedStaleWriteError) as exc_att:
            db.fenced_transition_task(
                task.task_id,
                "RUNNING",
                "SUCCEEDED",
                expected_goal_version=2,
                expected_human_epoch=3,
                expected_attempt=2,
                expected_lease_id=t_lease["lease_id"],
                now=202.0,
            )
        assert exc_att.value.reason_code == "STALE_ATTEMPT"

        # Wrong lease_id -> LEASE_MISMATCH
        with pytest.raises(FencedStaleWriteError) as exc_lease:
            db.fenced_transition_task(
                task.task_id,
                "RUNNING",
                "SUCCEEDED",
                expected_goal_version=2,
                expected_human_epoch=3,
                expected_attempt=1,
                expected_lease_id="lease_wrong",
                now=202.0,
            )
        assert exc_lease.value.reason_code == "LEASE_MISMATCH"

        # Bump human_epoch on Goal -> strict equality fence rejects stale task & decision!
        goal.human_epoch = 4
        db.save_goal(goal)
        with pytest.raises(FencedStaleWriteError) as exc_ep:
            db.fenced_transition_task(
                task.task_id,
                "RUNNING",
                "SUCCEEDED",
                expected_goal_version=2,
                expected_human_epoch=3,
                expected_attempt=1,
                expected_lease_id=t_lease["lease_id"],
                now=203.0,
            )
        assert exc_ep.value.reason_code == "HUMAN_INVALIDATED"

        dreq = db.enqueue_decision_request(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            attempt=1,
            goal_version=2,
            human_epoch=4,
        )
        db.resolve_decision_request(dreq["req_id"], status="RUNNING")
        d_lease = db.acquire_lease(
            f"decision:{goal.goal_id}:{dreq['req_id']}",
            f"dec-{dreq['req_id']}",
            "dec-worker",
            now=204.0,
        )
        assert d_lease is not None
        with pytest.raises(FencedStaleWriteError) as exc_dec_ep:
            db.fenced_resolve_decision_request(
                dreq["req_id"],
                expected_goal_version=2,
                expected_human_epoch=3,
                expected_attempt=1,
                expected_lease_id=d_lease["lease_id"],
                now=205.0,
            )
        assert exc_dec_ep.value.reason_code == "HUMAN_INVALIDATED"

        resolved_dec = db.fenced_resolve_decision_request(
            dreq["req_id"],
            expected_goal_version=2,
            expected_human_epoch=4,
            expected_attempt=1,
            expected_lease_id=d_lease["lease_id"],
            resolution={"actions": []},
            now=205.0,
        )
        assert resolved_dec["status"] == "RESOLVED"


# -------- 4. FailureDomain Isolation & Empirical Outcome Predicate


def test_failure_domain_and_empirical_predicate_isolate_all_non_scientific_failures(
    tmp_path: pathlib.Path,
) -> None:
    goal = _sample_goal(str(tmp_path / "repo"))

    # 1. Valid empirical confirmation (both scalar & structured dict metric) -> SCIENTIFIC_OUTCOME & valid
    ok_rcpt = _sample_receipt(failure_domain=FailureDomain.SCIENTIFIC_OUTCOME)
    assert classify_failure_domain(ok_rcpt, goal=goal) == FailureDomain.SCIENTIFIC_OUTCOME
    assert is_valid_empirical_outcome(ok_rcpt, goal) is True
    struct_rcpt = _sample_receipt(
        metrics={"spectral_gap": {"value": 0.58, "denominator": 10, "stage_rung": "F1_FAST_SCREEN"}}
    )
    assert classify_failure_domain(struct_rcpt, goal=goal) == FailureDomain.SCIENTIFIC_OUTCOME
    assert is_valid_empirical_outcome(struct_rcpt, goal) is True

    # 2. ProjBeta forensic case: cuda_unavailable -> ENV_CONFIG_MISMATCH, NEVER empirical
    cuda_rcpt = _sample_receipt(
        process_status=ProcessStatus.EXIT_NONZERO,
        engineering_verdict=EngineeringVerdict.FAIL_TEST,
        scientific_verdict=ScientificVerdict.REFUTED,
        evidence_validity=EvidenceValidity.MISSING_INPUT,
        metrics={"skip_reason": "cuda_unavailable"},
        failure_reason="RuntimeError: cuda_unavailable on CPU slot",
        exit_code=2,
    )
    assert classify_failure_domain(cuda_rcpt, goal=goal) == FailureDomain.ENV_CONFIG_MISMATCH
    assert is_valid_empirical_outcome(cuda_rcpt, goal) is False

    # 3. PROJALPHA forensic case: unrecognized arguments: --targeted-count -> ENV_CONFIG_MISMATCH
    cli_rcpt = _sample_receipt(
        process_status=ProcessStatus.EXIT_NONZERO,
        engineering_verdict=EngineeringVerdict.FAIL_TEST,
        evidence_validity=EvidenceValidity.MISSING_PRIMARY_METRIC,
        metrics={},
        failure_reason="eval.py: error: unrecognized arguments: --targeted-count 50",
        exit_code=2,
    )
    assert classify_failure_domain(cli_rcpt, goal=goal) == FailureDomain.ENV_CONFIG_MISMATCH
    assert is_valid_empirical_outcome(cli_rcpt, goal) is False

    # 4. PROJGAMMA forensic case: BLIND_AUDIT FAIL -> ENGINEERING_DEFECT, NEVER empirical
    audit_fail_rcpt = _sample_receipt(
        stage=StageKind.IMPL_REVIEW,
        audit_verdict=AuditVerdict.FAIL,
        scientific_verdict=ScientificVerdict.UNKNOWN,
        evidence_validity=EvidenceValidity.VALID_ENGINEERING_ONLY,
    )
    assert classify_failure_domain(audit_fail_rcpt, goal=goal) == FailureDomain.ENGINEERING_DEFECT
    assert is_valid_empirical_outcome(audit_fail_rcpt, goal) is False

    # 5. CPU_GATE TIMEOUT -> INFRA_TRANSIENT + UNKNOWN / BLOCKED_TIMEOUT
    timeout_rcpt = _sample_receipt(
        stage=StageKind.CPU_GATE,
        process_status=ProcessStatus.TIMEOUT,
        engineering_verdict=EngineeringVerdict.FAIL_TEST,
        scientific_verdict=ScientificVerdict.UNKNOWN,
        evidence_validity=EvidenceValidity.VALID_ENGINEERING_ONLY,
        exit_code=124,
    )
    assert classify_failure_domain(timeout_rcpt, goal=goal) == FailureDomain.INFRA_TRANSIENT
    assert is_valid_empirical_outcome(timeout_rcpt, goal) is False
    assert map_timeout_verdict(ProcessStatus.TIMEOUT) == (
        ScientificVerdict.UNKNOWN,
        "BLOCKED_TIMEOUT",
    )

    # 6. Genuine domain empirical kill metric crossing with EXIT_NONZERO + primary metric present
    kill_rcpt = _sample_receipt(
        process_status=ProcessStatus.EXIT_NONZERO,
        engineering_verdict=EngineeringVerdict.FAIL_TEST,
        scientific_verdict=ScientificVerdict.VALID_NEGATIVE,
        evidence_validity=EvidenceValidity.VALID_EMPIRICAL,
        failure_domain=FailureDomain.SCIENTIFIC_OUTCOME,
        metrics={"spectral_gap": 0.18},
        exit_code=1,
    )
    assert classify_failure_domain(kill_rcpt, goal=goal) == FailureDomain.SCIENTIFIC_OUTCOME
    assert is_valid_empirical_outcome(kill_rcpt, goal) is True

    # 7. External `cd /other/path` in ExperimentPlan / Hypothesis is blocked by gate
    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()
    bad_hyp = _sample_hyp(goal.goal_id)
    bad_hyp.screen_cmd = "cd /tmp/escape_outside && python run.py"
    plan = ExperimentPlan(
        exp_plan_id="ep-escape",
        goal_id=goal.goal_id,
        hypothesis_id=bad_hyp.hypothesis_id,
        falsifiable_claim="Test escape detection",
    )
    gate = validate_experiment_plan_gate(plan, goal, bad_hyp, repo_path=repo_dir)
    assert gate.passed is False
    assert any("escapes worktree via external cd" in f for f in gate.failures)


# ------ 5. 3-Way Verdicts, ReviewTier, & Exhaustive RED_TEAM_TIEBREAKER Quorum


def test_three_way_verdict_parsing_and_exhaustive_red_team_quorum_matrix() -> None:
    # Anchored standalone verdict lines for DESIGN_VERDICT, IMPL_VERDICT, RESULT_VERDICT
    for tag, rk in (
        ("DESIGN_VERDICT", ReviewKind.DESIGN_REVIEW),
        ("IMPL_VERDICT", ReviewKind.IMPLEMENTATION_REVIEW),
        ("RESULT_VERDICT", ReviewKind.RESULT_REVIEW),
    ):
        v, sev, _, _ = parse_review_output(f"All invariants verified.\n{tag}: PASS\n", review_kind=rk)
        assert v == AuditVerdict.PASS and sev == "NONE"
        vf, sevf, _, _ = parse_review_output(f"Found silent bug.\n{tag}: FAIL\n", review_kind=rk)
        assert vf == AuditVerdict.FAIL and sevf == "MAJOR"

    # Prose-only mention never yields false PASS
    assert parse_structured_verdict("I think IMPL_VERDICT: PASS might happen.").verdict == "PARSE_ERROR"

    goal = _sample_goal()
    theory_hyp = _sample_hyp(tags=["theory"])
    assert classify_review_tier(goal, theory_hyp, diff_lines=10) == ReviewTier.RED_TEAM_TIEBREAKER

    small_goal = ResearchGoal.from_dict({**goal.to_dict(), "routing_policy_nl": ""})
    small_hyp = HypothesisNode.from_dict(
        {
            **theory_hyp.to_dict(),
            "kind": HypothesisKind.REFINEMENT.value,
            "fidelity": FidelityRung.F0_SANITY.value,
            "tags": ["config_tweak"],
        }
    )
    assert classify_review_tier(small_goal, small_hyp, diff_lines=15) == ReviewTier.FAST_LANE

    # Verify prompts build cleanly without leaking internal IDs
    p1 = build_design_review_prompt(goal, theory_hyp)
    p2 = build_impl_review_prompt(goal, theory_hyp, diff_text="+def f(): return 1")
    p3 = build_result_review_prompt(goal, theory_hyp, receipts=[_sample_receipt()])
    assert "DESIGN_VERDICT:" in p1 and "IMPL_VERDICT:" in p2 and "RESULT_VERDICT:" in p3

    def _mk_rev(
        slot: int,
        family: str,
        verdict: AuditVerdict,
        pstatus: ProcessStatus = ProcessStatus.EXIT_OK,
    ) -> ReviewReceipt:
        return ReviewReceipt(
            review_id=f"rev-{slot}-{family}",
            goal_id=goal.goal_id,
            hypothesis_id=theory_hyp.hypothesis_id,
            workflow_id="wf-rt",
            task_id=f"tsk-rev-{slot}",
            review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
            review_tier=ReviewTier.RED_TEAM_TIEBREAKER,
            target_version=1,
            head_sha="1234567abcdef",
            slot_index=slot,
            author_families=["anthropic"],
            reviewer_route={"family": family, "canonical_model": f"{family}-model"},
            verdict=verdict,
            process_status=pstatus,
            findings=[f"{family}:{verdict.value}"],
            created_ts=100.0 + slot,
        )

    base_state = ReviewQuorumReducer.init_quorum_state(
        review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
        review_tier=ReviewTier.RED_TEAM_TIEBREAKER,
        target_version=1,
        head_sha="1234567abcdef",
        contributor_families=["anthropic"],
        now=100.0,
    )
    assert base_state["quorum_deadline_ts"] == 100.0 + INITIAL_QUORUM_DEADLINE_S

    # Case A: (PASS, PASS) on 2 distinct non-contributor families -> PASS
    dec_pp = ReviewQuorumReducer.evaluate_quorum(
        base_state,
        [_mk_rev(0, "openai", AuditVerdict.PASS), _mk_rev(1, "google", AuditVerdict.PASS)],
        now=120.0,
    )
    assert dec_pp.status == "PASS"

    # Case B: (FAIL, NON_VERDICT [OOM]) -> fail-closed FAIL immediately without wasting tie-breaker!
    dec_fn = ReviewQuorumReducer.evaluate_quorum(
        base_state,
        [
            _mk_rev(0, "openai", AuditVerdict.FAIL),
            _mk_rev(1, "google", AuditVerdict.PARSE_ERROR, ProcessStatus.OOM),
        ],
        now=120.0,
    )
    assert dec_fn.status == "FAIL"

    # Case C: (PASS, FAIL) -> NEED_TIEBREAKER (slot 2, deadline +300s) -> Director PASS / FAIL / NON_VERDICT
    dec_pf = ReviewQuorumReducer.evaluate_quorum(
        base_state,
        [_mk_rev(0, "openai", AuditVerdict.PASS), _mk_rev(1, "google", AuditVerdict.FAIL)],
        now=150.0,
    )
    assert dec_pf.status == "NEED_TIEBREAKER"
    assert dec_pf.slots_to_spawn == [2]
    assert dec_pf.updated_state["quorum_deadline_ts"] == 150.0 + TIEBREAKER_DEADLINE_S
    assert dec_pf.prefer_avoid_families == ["anthropic", "google", "openai"]

    dec_tb_pass = ReviewQuorumReducer.evaluate_quorum(
        dec_pf.updated_state,
        [
            _mk_rev(0, "openai", AuditVerdict.PASS),
            _mk_rev(1, "google", AuditVerdict.FAIL),
            _mk_rev(2, "deepseek", AuditVerdict.PASS),
        ],
        now=200.0,
    )
    assert dec_tb_pass.status == "PASS"

    dec_tb_nonverdict = ReviewQuorumReducer.evaluate_quorum(
        dec_pf.updated_state,
        [
            _mk_rev(0, "openai", AuditVerdict.PASS),
            _mk_rev(1, "google", AuditVerdict.FAIL),
            _mk_rev(2, "deepseek", AuditVerdict.PARSE_ERROR, ProcessStatus.TIMEOUT),
        ],
        now=200.0,
    )
    assert dec_tb_nonverdict.status == "BLOCKED_REVIEW_QUORUM"
    assert dec_tb_nonverdict.escalate_owner is True

    # Case D: Unschedulable tie-breaker -> BLOCKED_REVIEW_QUORUM + OWNER_ESCALATION
    dec_unsched = ReviewQuorumReducer.evaluate_quorum(
        base_state,
        [_mk_rev(0, "openai", AuditVerdict.PASS), _mk_rev(1, "google", AuditVerdict.FAIL)],
        now=150.0,
        tiebreaker_schedulable=False,
    )
    assert dec_unsched.status == "BLOCKED_REVIEW_QUORUM"
    assert dec_unsched.escalate_owner is True

    # Case E: (NON_VERDICT, NON_VERDICT) -> 1 retry -> still (NON_VERDICT, NON_VERDICT) -> BLOCKED_REVIEW_QUORUM
    dec_nn1 = ReviewQuorumReducer.evaluate_quorum(
        base_state,
        [
            _mk_rev(0, "openai", AuditVerdict.PARSE_ERROR),
            _mk_rev(1, "google", AuditVerdict.PARSE_ERROR, ProcessStatus.LEASE_LOST),
        ],
        now=130.0,
    )
    assert dec_nn1.status == "NEED_RETRY_SLOTS"
    assert dec_nn1.slots_to_spawn == [0, 1]

    dec_nn2 = ReviewQuorumReducer.evaluate_quorum(
        dec_nn1.updated_state,
        [
            _mk_rev(0, "deepseek", AuditVerdict.PARSE_ERROR, ProcessStatus.EXIT_NONZERO),
            _mk_rev(1, "xai", AuditVerdict.PARSE_ERROR, ProcessStatus.ENV_ERROR),
        ],
        now=180.0,
    )
    assert dec_nn2.status == "BLOCKED_REVIEW_QUORUM"
    assert dec_nn2.escalate_owner is True

    # Case F: Persisted deadline expiry across restart -> BLOCKED_REVIEW_QUORUM
    dec_expired = ReviewQuorumReducer.evaluate_quorum(
        base_state,
        [_mk_rev(0, "openai", AuditVerdict.PASS)],
        now=100.0 + INITIAL_QUORUM_DEADLINE_S + 1.0,
    )
    assert dec_expired.status == "BLOCKED_REVIEW_QUORUM"
    assert dec_expired.escalate_owner is True


def test_statedb_review_quorum_inflight_retry_and_glob_underscore_and_prestamped_domain(
    tmp_path: pathlib.Path,
) -> None:
    goal = _sample_goal(str(tmp_path / "repo"))
    with StateDB(tmp_path / "arr") as db:
        db.save_goal(goal)
        h_ab = _sample_hyp(goal.goal_id)
        h_ab.hypothesis_id = "hyp_a_b"
        h_axb = HypothesisNode.from_dict(
            {
                **h_ab.to_dict(),
                "hypothesis_id": "hyp_axb",
                "mechanism": "distinct_mech_axb",
                "fingerprint": "",
            }
        )
        db.register_hypothesis(h_ab)
        db.register_hypothesis(h_axb)

        # 1. Underscore `_` in `hyp_a_b` must match literally in GLOB (not wildcard `hyp_axb`!)
        ro_lease = db.acquire_branch_ro_lease(
            goal.goal_id, "hyp_a_b", "sha1", "tsk-r0", "w0", now=100.0
        )
        assert ro_lease is not None
        assert db.has_active_branch_ro_leases(goal.goal_id, "hyp_a_b", now=101.0) is True
        assert db.has_active_branch_ro_leases(goal.goal_id, "hyp_axb", now=101.0) is False
        assert (
            db.acquire_branch_writer_lease(goal.goal_id, "hyp_axb", "tsk-w-axb", "w1", now=101.0)
            is not None
        )
        assert (
            db.acquire_branch_writer_lease(goal.goal_id, "hyp_a_b", "tsk-w-ab", "w2", now=101.0)
            is None
        )

        # 2. Mis-stamped `failure_domain=SCIENTIFIC_OUTCOME` with `cuda_unavailable` MUST still fail `is_valid_empirical_outcome`!
        misstamped = _sample_receipt(
            process_status=ProcessStatus.EXIT_NONZERO,
            engineering_verdict=EngineeringVerdict.FAIL_TEST,
            scientific_verdict=ScientificVerdict.REFUTED,
            evidence_validity=EvidenceValidity.VALID_EMPIRICAL,
            failure_domain=FailureDomain.SCIENTIFIC_OUTCOME,
            metrics={"spectral_gap": 0.10, "skip_reason": "cuda_unavailable"},
            failure_reason="RuntimeError: cuda_unavailable",
            exit_code=1,
        )
        assert is_valid_empirical_outcome(misstamped, goal) is False

        # 3. StateDB-backed in-flight retry sweep & staggered slot-0 / slot-1 retry completion
        q0 = ReviewQuorumReducer.init_quorum_state(
            review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
            review_tier=ReviewTier.RED_TEAM_TIEBREAKER,
            target_version=1,
            head_sha="deadbeef",
            contributor_families=["anthropic"],
            now=100.0,
        )
        db.save_review_receipt(
            ReviewReceipt(
                review_id="rev-s0-a1",
                goal_id=goal.goal_id,
                hypothesis_id="hyp_a_b",
                workflow_id="wf1",
                task_id="tsk-s0-a1",
                review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                review_tier=ReviewTier.RED_TEAM_TIEBREAKER,
                target_version=1,
                head_sha="deadbeef",
                slot_index=0,
                reviewer_route={"family": "openai"},
                verdict=AuditVerdict.PARSE_ERROR,
                process_status=ProcessStatus.TIMEOUT,
                created_ts=110.0,
            )
        )
        db.save_review_receipt(
            ReviewReceipt(
                review_id="rev-s1-a1",
                goal_id=goal.goal_id,
                hypothesis_id="hyp_a_b",
                workflow_id="wf1",
                task_id="tsk-s1-a1",
                review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                review_tier=ReviewTier.RED_TEAM_TIEBREAKER,
                target_version=1,
                head_sha="deadbeef",
                slot_index=1,
                reviewer_route={"family": "google"},
                verdict=AuditVerdict.PARSE_ERROR,
                process_status=ProcessStatus.LEASE_LOST,
                created_ts=112.0,
            )
        )

        d1 = ReviewQuorumReducer.evaluate_quorum(
            q0, db.list_review_receipts(workflow_id="wf1"), now=115.0
        )
        assert d1.status == "NEED_RETRY_SLOTS"
        assert d1.slots_to_spawn == [0, 1]

        # Sweep while retries are in flight MUST return WAIT_SLOTS (NOT premature BLOCKED_REVIEW_QUORUM!)
        d_inflight = ReviewQuorumReducer.evaluate_quorum(
            d1.updated_state, db.list_review_receipts(workflow_id="wf1"), now=116.0
        )
        assert d_inflight.status == "WAIT_SLOTS"

        # Staggered completion: retry slot 0 finishes PASS (`created_ts=140`) while retry slot 1 is still running!
        db.save_review_receipt(
            ReviewReceipt(
                review_id="rev-s0-a2",
                goal_id=goal.goal_id,
                hypothesis_id="hyp_a_b",
                workflow_id="wf1",
                task_id="tsk-s0-a2",
                review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                review_tier=ReviewTier.RED_TEAM_TIEBREAKER,
                target_version=1,
                head_sha="deadbeef",
                slot_index=0,
                reviewer_route={"family": "deepseek"},
                verdict=AuditVerdict.PASS,
                process_status=ProcessStatus.EXIT_OK,
                created_ts=140.0,
            )
        )
        d_staggered = ReviewQuorumReducer.evaluate_quorum(
            d1.updated_state, db.list_review_receipts(workflow_id="wf1"), now=141.0
        )
        assert d_staggered.status == "WAIT_SLOTS"

        # Now retry slot 1 finishes PASS (`created_ts=155`) -> quorum resolves PASS cleanly!
        db.save_review_receipt(
            ReviewReceipt(
                review_id="rev-s1-a2",
                goal_id=goal.goal_id,
                hypothesis_id="hyp_a_b",
                workflow_id="wf1",
                task_id="tsk-s1-a2",
                review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                review_tier=ReviewTier.RED_TEAM_TIEBREAKER,
                target_version=1,
                head_sha="deadbeef",
                slot_index=1,
                reviewer_route={"family": "xai"},
                verdict=AuditVerdict.PASS,
                process_status=ProcessStatus.EXIT_OK,
                created_ts=155.0,
            )
        )
        d_done = ReviewQuorumReducer.evaluate_quorum(
            d_staggered.updated_state, db.list_review_receipts(workflow_id="wf1"), now=156.0
        )
        assert d_done.status == "PASS"

        # 4. Reentrant `db._tx()` rolls back ALL nested mutations & events atomically on failure
        with pytest.raises(RuntimeError, match="abort_outer_tx"):
            with db._tx():
                db.save_research_plan(
                    ResearchPlan(
                        plan_id="rp_rollback_test",
                        goal_id=goal.goal_id,
                        thesis="Should roll back completely",
                    )
                )
                raise RuntimeError("abort_outer_tx")
        assert db.get_research_plan("rp_rollback_test") is None


def test_v4_schema_migration_backfill_epoch_review_uniqueness_and_budget_reservations(
    tmp_path: pathlib.Path,
) -> None:
    arr_home = tmp_path / "v3_to_v4"
    arr_home.mkdir()
    db_file = arr_home / "state.db"

    # Seed a v3 database (`user_version = 3`) with historical receipt + decision token spend
    raw = sqlite3.connect(str(db_file))
    raw.executescript(
        """
        CREATE TABLE goals (
            goal_id TEXT PRIMARY KEY, project TEXT NOT NULL, title TEXT NOT NULL,
            direction TEXT NOT NULL, primary_metric TEXT NOT NULL, baseline_value REAL NOT NULL,
            baseline_sha TEXT NOT NULL, repo_path TEXT NOT NULL, max_tokens_budget INTEGER NOT NULL,
            max_concurrent_branches INTEGER NOT NULL, adapter_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ACTIVE', payload_json TEXT NOT NULL,
            goal_version INTEGER NOT NULL DEFAULT 1, human_epoch INTEGER NOT NULL DEFAULT 0,
            risk_profile TEXT NOT NULL DEFAULT 'standard', created_ts REAL NOT NULL, updated_ts REAL NOT NULL
        );
        CREATE TABLE evidence_receipts (
            receipt_id TEXT PRIMARY KEY, task_id TEXT NOT NULL, goal_id TEXT NOT NULL,
            hypothesis_id TEXT NOT NULL, stage TEXT NOT NULL, exit_code INTEGER NOT NULL,
            in_tokens INTEGER NOT NULL, out_tokens INTEGER NOT NULL, wall_sec REAL NOT NULL,
            max_rss_mb REAL NOT NULL DEFAULT 0.0,
            process_status TEXT NOT NULL, engineering_verdict TEXT NOT NULL,
            scientific_verdict TEXT NOT NULL, audit_verdict TEXT NOT NULL,
            evidence_validity TEXT NOT NULL, goal_version INTEGER NOT NULL DEFAULT 1,
            workflow_id TEXT, failure_domain TEXT NOT NULL DEFAULT 'NONE',
            effort_used TEXT NOT NULL DEFAULT '', reviewer_family TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL, created_ts REAL NOT NULL
        );
        CREATE TABLE decision_requests (
            req_id TEXT PRIMARY KEY, goal_id TEXT NOT NULL, tier TEXT NOT NULL,
            trigger_kind TEXT NOT NULL, status TEXT NOT NULL, goal_version INTEGER NOT NULL,
            human_epoch INTEGER NOT NULL, coalesce_key TEXT NOT NULL DEFAULT '',
            available_ts REAL NOT NULL, role TEXT NOT NULL DEFAULT 'research_lead',
            attempt INTEGER NOT NULL DEFAULT 1, model_spec TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL, resolution_json TEXT NOT NULL DEFAULT '{}',
            hq_job_id INTEGER, created_ts REAL NOT NULL, updated_ts REAL NOT NULL, resolved_ts REAL
        );
        CREATE TABLE review_records (
            review_id TEXT PRIMARY KEY, goal_id TEXT NOT NULL, hypothesis_id TEXT NOT NULL,
            workflow_id TEXT NOT NULL DEFAULT '', task_id TEXT NOT NULL DEFAULT '',
            review_kind TEXT NOT NULL, review_tier TEXT NOT NULL,
            target_version INTEGER NOT NULL DEFAULT 1, head_sha TEXT NOT NULL DEFAULT '',
            slot_index INTEGER NOT NULL DEFAULT 0, verdict TEXT NOT NULL,
            process_status TEXT NOT NULL DEFAULT 'EXIT_OK', severity TEXT NOT NULL DEFAULT 'NONE',
            tiebreaker_used INTEGER NOT NULL DEFAULT 0, in_tokens INTEGER NOT NULL DEFAULT 0,
            out_tokens INTEGER NOT NULL DEFAULT 0, payload_json TEXT NOT NULL, created_ts REAL NOT NULL,
            UNIQUE (workflow_id, review_kind, target_version, head_sha, slot_index)
        );
        CREATE VIEW review_receipts AS SELECT * FROM review_records;
        INSERT INTO goals VALUES (
            'g_mig', 'projgamma', 'Mig', 'maximize', 'm', 0.1, 'sha0', '/tmp/r', 20000, 2, 'generic',
            'ACTIVE', '{"goal_id":"g_mig","project":"projgamma","title":"Mig","direction":"maximize","primary_metric":"m","baseline_value":0.1,"baseline_sha":"sha0","repo_path":"/tmp/r","max_tokens_budget":20000}',
            1, 0, 'standard', 100.0, 100.0
        );
        INSERT INTO evidence_receipts VALUES (
            'rcpt_hist_1', 'tsk_h1', 'g_mig', 'hyp_h1', 'CODE', 0, 4000, 1000, 5.0, 64.0,
            'EXIT_OK', 'PASS', 'NOT_APPLICABLE', 'NOT_REQUIRED', 'VALID_ENGINEERING_ONLY',
            1, 'wf_h1', 'NONE', 'high', '', '{"model_used":"claude-opus-5-5"}', 110.0
        );
        INSERT INTO decision_requests VALUES (
            'dec_hist_1', 'g_mig', 'L1', 'INITIAL_PLANNING', 'RESOLVED', 1, 0, '',
            100.0, 'research_lead', 1, 'codex:gpt-5-codex:high', '{}',
            '{"in_tokens": 3000, "out_tokens": 500, "cached_in_tokens": 200}', NULL, 100.0, 115.0, 115.0
        );
        PRAGMA user_version = 3;
        """
    )
    raw.close()

    with StateDB(arr_home) as db:
        assert int(db._conn.execute("PRAGMA user_version").fetchone()[0]) == 8
        usage = db.get_goal_token_usage("g_mig")
        # Both historical receipt (5000) AND historical decision (3500) are accounted!
        assert usage["in_tokens"] == 7000
        assert usage["out_tokens"] == 1500
        assert usage["total_tokens"] == 8500
        assert usage["decision_tokens"] == 3500

        # Reserve 9000 tokens (8500 + 9000 = 17500 <= 20000) -> allowed; second 5000 reservation -> denied!
        ok1, comm1, _cap1 = db.reserve_goal_budget("g_mig", "dec:d2:a1", entity_kind="decision", entity_id="d2", reserve_tokens=9000)
        assert ok1 is True
        assert comm1 == 17500
        assert db.get_goal_token_usage("g_mig")["committed_tokens"] == 17500
        ok2, _comm2, _cap2 = db.reserve_goal_budget("g_mig", "task:t2:a1", entity_kind="stage_task", entity_id="t2", reserve_tokens=5000)
        assert ok2 is False

        # Settling reservation replaces reserved_tokens cleanly with actual tokens (no double-count)
        db.settle_goal_budget("g_mig", "dec:d2:a1", entity_kind="decision", entity_id="d2", input_tokens=1200, output_tokens=300)
        usage_settled = db.get_goal_token_usage("g_mig")
        assert usage_settled["reserved_tokens"] == 0
        assert usage_settled["total_tokens"] == 10000

        # Epoch-bound review uniqueness: distinct `(plan_epoch, human_epoch)` on same `(workflow_id, review_kind, target_version, head_sha, slot_index)` BOTH persist!
        for rev_id, pe, he, verd in (("rev_pe0", 0, 1, AuditVerdict.FAIL), ("rev_pe1", 1, 2, AuditVerdict.PASS)):
            db.save_review_receipt(
                ReviewReceipt(
                    review_id=rev_id,
                    goal_id="g_mig",
                    hypothesis_id="hyp_h1",
                    workflow_id="wf_same",
                    task_id=f"tsk_{rev_id}",
                    review_kind=ReviewKind.DESIGN_REVIEW,
                    review_tier=ReviewTier.FAST_LANE,
                    target_version=1,
                    plan_epoch=pe,
                    human_epoch=he,
                    head_sha="sha0",
                    slot_index=0,
                    reviewer_route={"family": "openai"},
                    verdict=verd,
                    process_status=ProcessStatus.EXIT_OK,
                    created_ts=200.0 + pe,
                )
            )
        revs = db.list_review_receipts(workflow_id="wf_same")
        assert len(revs) == 2
        q_pe1 = ReviewQuorumReducer.init_quorum_state(
            review_kind=ReviewKind.DESIGN_REVIEW,
            review_tier=ReviewTier.FAST_LANE,
            target_version=1,
            plan_epoch=1,
            human_epoch=2,
            head_sha="sha0",
            contributor_families=["anthropic"],
            now=199.0,
        )
        q_dec = ReviewQuorumReducer.evaluate_quorum(q_pe1, revs, now=205.0)
        assert q_dec.status == "PASS"

        syn_prompt = build_synthesis_review_prompt(
            _sample_goal(),
            [_sample_hyp("g_mig")],
            [_sample_receipt()],
            "Combined spectral bound",
        )
        assert "SYNTHESIS" in syn_prompt.upper()


def test_authenticated_negative_predicate_epoch_fencing_and_cross_goal_ownership(
    tmp_path: pathlib.Path,
) -> None:
    goal_a = _sample_goal(str(tmp_path / "repo_a"))
    goal_a.plan_epoch = 2
    hyp_a = _sample_hyp(goal_a.goal_id)

    # 1. `is_authenticated_negative_empirical_outcome` rejects CONFIRMED and non-kill UNKNOWN
    r_pos = _sample_receipt(
        scientific_verdict=ScientificVerdict.CONFIRMED,
        failure_domain=FailureDomain.SCIENTIFIC_OUTCOME,
        metrics={"spectral_gap": 0.62},
    )
    r_unk = _sample_receipt(
        scientific_verdict=ScientificVerdict.UNKNOWN,
        failure_domain=FailureDomain.SCIENTIFIC_OUTCOME,
        metrics={"spectral_gap": 0.45},
    )
    r_kill = _sample_receipt(
        scientific_verdict=ScientificVerdict.UNKNOWN,
        failure_domain=FailureDomain.SCIENTIFIC_OUTCOME,
        metrics={"spectral_gap": 0.18},
    )
    r_ref = _sample_receipt(
        scientific_verdict=ScientificVerdict.REFUTED,
        failure_domain=FailureDomain.SCIENTIFIC_OUTCOME,
        metrics={"spectral_gap": 0.18},
    )
    assert is_authenticated_negative_empirical_outcome(r_pos, hyp_a, goal=goal_a) is False
    assert is_authenticated_negative_empirical_outcome(r_unk, hyp_a, goal=goal_a) is False
    assert evaluate_kill_criteria(r_kill, hyp_a, goal_a) == "spectral_gap < 0.30"
    assert is_authenticated_negative_empirical_outcome(r_kill, hyp_a, goal=goal_a) is True
    assert is_authenticated_negative_empirical_outcome(r_ref, hyp_a, goal=goal_a) is True

    # 2. `validate_receipt_epoch` fences `plan_epoch` and `human_epoch` even when `task` is provided!
    stale_task = StageTask(
        task_id="tsk_pe1",
        idempotency_key="ik_pe1",
        goal_id=goal_a.goal_id,
        hypothesis_id=hyp_a.hypothesis_id,
        stage=StageKind.DESIGN_REVIEW,
        deps=[],
        goal_version=goal_a.version,
        human_epoch=goal_a.human_epoch,
        plan_epoch=1,
    )
    stale_rcpt = _sample_receipt()
    stale_rcpt.plan_epoch = 1
    assert validate_receipt_epoch(stale_rcpt, goal_a, task=stale_task) == EvidenceValidity.SUPERSEDED_EPOCH
    assert (
        validate_receipt_epoch(
            stale_rcpt,
            goal_a,
            task=stale_task,
            preserved_receipt_ids={stale_rcpt.receipt_id},
        )
        == EvidenceValidity.VALID_EMPIRICAL
    )

    # 3. Cross-Goal StateDB ownership rejections
    goal_b = ResearchGoal.from_dict({**goal_a.to_dict(), "goal_id": "goal-foreign-b"})
    hyp_b = HypothesisNode.from_dict({**hyp_a.to_dict(), "hypothesis_id": "hyp-b-1", "goal_id": goal_b.goal_id, "fingerprint": ""})
    with StateDB(tmp_path / "own_arr") as db:
        db.save_goal(goal_a)
        db.save_goal(goal_b)
        db.register_hypothesis(hyp_a)
        db.register_hypothesis(hyp_b)

        with pytest.raises(ValueError, match="Cross-goal ownership violation"):
            db.get_goal_hypothesis(goal_a.goal_id, hyp_b.hypothesis_id)
        with pytest.raises(Exception, match="belongs to goal"):
            db.register_hypothesis(
                HypothesisNode.from_dict(
                    {
                        **hyp_a.to_dict(),
                        "hypothesis_id": "hyp-cross-child",
                        "parent_id": hyp_b.hypothesis_id,
                        "mechanism": "cross_parent_mech",
                        "fingerprint": "",
                    }
                )
            )
        with pytest.raises(Exception, match="Cross-goal ownership violation"):
            db.save_stage_task(
                StageTask(
                    task_id="tsk-cross",
                    idempotency_key="ik-cross",
                    goal_id=goal_a.goal_id,
                    hypothesis_id=hyp_b.hypothesis_id,
                    stage=StageKind.CODE,
                    deps=[],
                )
            )


def test_r2_primary_key_collisions_promotion_intents_cas_task_leases_and_budget_release(
    tmp_path: pathlib.Path,
) -> None:
    from conjectra.types import WorkflowNode

    ga = _sample_goal(str(tmp_path / "repo_a"))
    gb = ResearchGoal.from_dict({**ga.to_dict(), "goal_id": "goal-b", "max_tokens_budget": 8_000})
    ha1 = _sample_hyp(ga.goal_id)
    ha2 = HypothesisNode.from_dict(
        {**ha1.to_dict(), "hypothesis_id": "hyp-a-2", "mechanism": "mech_a2", "fingerprint": ""}
    )
    hb1 = HypothesisNode.from_dict(
        {**ha1.to_dict(), "hypothesis_id": "hyp-b-1", "goal_id": gb.goal_id, "mechanism": "mech_b1", "fingerprint": ""}
    )

    with StateDB(tmp_path / "r2_arr") as db:
        assert int(db._conn.execute("PRAGMA user_version").fetchone()[0]) == 8
        db.save_goal(ga)
        db.save_goal(gb)
        for h in (ha1, ha2, hb1):
            db.register_hypothesis(h)

        # 1. Primary-key collision guards across ResearchPlan, ExperimentPlan, ExecutionPlan, WorkflowNode, StageTask
        rp_a = db.save_research_plan(ResearchPlan(plan_id="rp_shared_id", goal_id=ga.goal_id, thesis="Goal A thesis"))
        with pytest.raises(ValueError, match="Cross-goal ResearchPlan collision"):
            db.save_research_plan(ResearchPlan(plan_id=rp_a.plan_id, goal_id=gb.goal_id, thesis="Hijack"))
        assert db.get_research_plan(rp_a.plan_id).goal_id == ga.goal_id

        ep_a = db.save_experiment_plan(
            ExperimentPlan(
                exp_plan_id="ep_shared_id",
                goal_id=ga.goal_id,
                hypothesis_id=ha1.hypothesis_id,
                falsifiable_claim="Original claim A1",
            )
        )
        with pytest.raises(ValueError, match="ExperimentPlan"):
            db.save_experiment_plan(
                ExperimentPlan(
                    exp_plan_id=ep_a.exp_plan_id,
                    goal_id=gb.goal_id,
                    hypothesis_id=hb1.hypothesis_id,
                    falsifiable_claim="Hijack B1",
                )
            )
        with pytest.raises(ValueError, match="ExperimentPlan collision"):
            db.save_experiment_plan(
                ExperimentPlan(
                    exp_plan_id=ep_a.exp_plan_id,
                    goal_id=ga.goal_id,
                    hypothesis_id=ha2.hypothesis_id,
                    falsifiable_claim="Hijack A2",
                )
            )
        assert db.get_experiment_plan(ep_a.exp_plan_id).falsifiable_claim == "Original claim A1"

        wf_a = db.save_workflow(WorkflowNode(workflow_id="wf_shared_id", goal_id=ga.goal_id, hypothesis_id=ha1.hypothesis_id))
        with pytest.raises(ValueError, match="workflow_id collision"):
            db.save_workflow(WorkflowNode(workflow_id=wf_a.workflow_id, goal_id=gb.goal_id, hypothesis_id=hb1.hypothesis_id))

        xp_a = db.save_execution_plan(
            ExecutionPlan(
                exec_plan_id="xp_shared_id",
                workflow_id=wf_a.workflow_id,
                goal_id=ga.goal_id,
                hypothesis_id=ha1.hypothesis_id,
            )
        )
        with pytest.raises(ValueError, match="ExecutionPlan"):
            db.save_execution_plan(
                ExecutionPlan(
                    exec_plan_id=xp_a.exec_plan_id,
                    workflow_id="wf_other",
                    goal_id=gb.goal_id,
                    hypothesis_id=hb1.hypothesis_id,
                )
            )

        t_a = db.enqueue_task(
            StageTask(
                task_id="tsk_shared_id",
                idempotency_key="ik_shared_key",
                goal_id=ga.goal_id,
                hypothesis_id=ha1.hypothesis_id,
                stage=StageKind.CODE,
                deps=[],
            )
        )
        with pytest.raises(ValueError, match="task_id collision"):
            db.enqueue_task(
                StageTask(
                    task_id=t_a.task_id,
                    idempotency_key="ik_other_key",
                    goal_id=gb.goal_id,
                    hypothesis_id=hb1.hypothesis_id,
                    stage=StageKind.CODE,
                    deps=[],
                )
            )
        with pytest.raises(ValueError, match="idempotency_key collision"):
            db.enqueue_task(
                StageTask(
                    task_id="tsk_other_id",
                    idempotency_key=t_a.idempotency_key,
                    goal_id=gb.goal_id,
                    hypothesis_id=hb1.hypothesis_id,
                    stage=StageKind.CODE,
                    deps=[],
                )
            )

        # 2. `verify_and_renew_task_leases` checks ownership + expiry and renews TTL inside `db._tx()`
        owner_id = f"task:{t_a.task_id}:a1"
        l_row = db.acquire_branch_writer_lease(ga.goal_id, ha1.hypothesis_id, t_a.task_id, owner_id, ttl_s=30.0, now=100.0)
        assert l_row is not None
        ok_l, msg_l = db.verify_and_renew_task_leases(
            t_a.task_id, 1, owner_id, [l_row["resource_key"]], ttl_sec=60.0, now=110.0
        )
        assert ok_l is True and msg_l == "OK"
        ok_exp, msg_exp = db.verify_and_renew_task_leases(
            t_a.task_id, 1, owner_id, [l_row["resource_key"]], now=200.0
        )
        assert ok_exp is False and "LEASE_EXPIRED_OR_LOST" in msg_exp

        # 3. `promotion_intents` two-phase CAS advances `hypothesis.head_sha` atomically
        pi = db.prepare_promotion_intent(
            goal_id=ga.goal_id,
            hypothesis_id=ha1.hypothesis_id,
            task_id=t_a.task_id,
            attempt=1,
            expected_base_sha=ha1.base_sha,
            candidate_sha="feedface1234",
        )
        assert pi["state"] == "PREPARED"
        pi_comm = db.commit_promotion_intent(pi["intent_id"], candidate_sha="feedface1234")
        assert pi_comm["state"] == "COMMITTED"
        assert db.get_hypothesis(ha1.hypothesis_id).head_sha == "feedface1234"

        # 4. Budget reservation release on cancellation (no permanent leak!) & `exclude_usage_key` self-block guard
        t_b1 = db.enqueue_task(
            StageTask(
                task_id="tsk_b1",
                idempotency_key="ik_b1",
                goal_id=gb.goal_id,
                hypothesis_id=hb1.hypothesis_id,
                stage=StageKind.CODE,
                deps=[],
            )
        )
        t_b2 = db.enqueue_task(
            StageTask(
                task_id="tsk_b2",
                idempotency_key="ik_b2",
                goal_id=gb.goal_id,
                hypothesis_id=hb1.hypothesis_id,
                stage=StageKind.EXP_SCREEN,
                deps=[],
            )
        )
        assert db.reserve_goal_budget(gb.goal_id, "task:tsk_b1:a1", entity_kind="stage_task", entity_id="tsk_b1", reserve_tokens=4000)[0] is True
        assert db.reserve_goal_budget(gb.goal_id, "task:tsk_b2:a1", entity_kind="stage_task", entity_id="tsk_b2", reserve_tokens=4000)[0] is True
        assert db.is_budget_exhausted(gb.goal_id) is True
        # Own reservation NEVER self-blocks `tsk_b1`, while a 3rd key `tsk_b3` IS blocked!
        assert db.is_budget_exhausted(gb.goal_id, exclude_usage_key="task:tsk_b1:a1") is False
        assert db.is_budget_exhausted(gb.goal_id, exclude_usage_key="task:tsk_b3:a1") is True

        # Cancelling `tsk_b1` & `tsk_b2` without receipts releases BOTH reservations (`reserved_tokens == 0`)!
        db.transition_task_status("tsk_b1", "READY", "CANCELLED")
        db.transition_task_status("tsk_b2", "READY", "CANCELLED")
        usage_b = db.get_goal_token_usage(gb.goal_id)
        assert usage_b["reserved_tokens"] == 0
        assert usage_b["budget_exhausted"] is False

        # Settled actual tokens (`state='SETTLED'`) are NEVER erased by a later cancellation!
        db.settle_goal_budget(gb.goal_id, "task:tsk_settled:a1", entity_kind="stage_task", entity_id="tsk_settled", input_tokens=2500, output_tokens=500)
        db.release_goal_budget_reservation(gb.goal_id, "task:tsk_settled:a1", reason="late_cancel")
        assert db.get_goal_token_usage(gb.goal_id)["total_tokens"] == 3000

        # Late attempt-1 receipt (`-a1-<hex>`) arriving after task advanced to `attempt=2` settles `a1`
        # WITHOUT clobbering `a2`'s live reservation; `fenced_transition_task(..., "CANCELLED")` releases `a2`!
        db.enqueue_task(
            StageTask(
                task_id="tsk_retry",
                idempotency_key="ik_retry",
                goal_id=gb.goal_id,
                hypothesis_id=hb1.hypothesis_id,
                stage=StageKind.CODE,
                deps=[],
                attempt=2,
                goal_version=gb.version,
                human_epoch=gb.human_epoch,
            )
        )
        assert db.reserve_goal_budget(
            gb.goal_id,
            "task:tsk_retry:a2",
            entity_kind="stage_task",
            entity_id="tsk_retry",
            attempt=2,
            reserve_tokens=4000,
        )[0] is True
        r_late_a1 = _sample_receipt()
        r_late_a1.receipt_id = "rcpt-tsk_retry-a1-deadbeef"
        r_late_a1.goal_id = gb.goal_id
        r_late_a1.hypothesis_id = hb1.hypothesis_id
        r_late_a1.task_id = "tsk_retry"
        r_late_a1.in_tokens = 600
        r_late_a1.out_tokens = 150
        db.record_receipt(r_late_a1)
        usage_retry = db.get_goal_token_usage(gb.goal_id)
        assert usage_retry["reserved_tokens"] == 4000
        assert usage_retry["total_tokens"] == 3750
        db.fenced_transition_task(
            "tsk_retry",
            "READY",
            "CANCELLED",
            expected_goal_version=gb.version,
            expected_human_epoch=gb.human_epoch,
            expected_attempt=2,
        )
        assert db.get_goal_token_usage(gb.goal_id)["reserved_tokens"] == 0

    # 5. When BOTH `plan_epoch` and `human_epoch` advance (`update_plan`), `validate_receipt_epoch` returns `SUPERSEDED_EPOCH`
    g_both = ResearchGoal.from_dict({**ga.to_dict(), "human_epoch": 4, "plan_epoch": 2})
    r_old = _sample_receipt(human_epoch=3)
    r_old.plan_epoch = 1
    assert validate_receipt_epoch(r_old, g_both) == EvidenceValidity.SUPERSEDED_EPOCH


def test_r3_v6_migration_legacy_s3_revoked_leases_dual_key_exec_plan_and_attempt_cancel_confirm(
    tmp_path: pathlib.Path,
) -> None:
    from conjectra.state_db import FencedStaleWriteError
    from conjectra.types import ExecutionPlan, ReviewTier, WorkflowNode

    arr_home = tmp_path / "arr_r3_s1"
    with StateDB(arr_home) as db:
        ga = _sample_goal("goal_r3_a")
        ga.max_tokens_budget = 16_000
        db.save_goal(ga)
        ha, _ = db.register_hypothesis(_sample_hyp(ga.goal_id))

        # Seed a legacy pre-R3 S3 workflow without `source_lineage_snapshot` + READY review task + 125 revoked leases
        legacy_wf = WorkflowNode(
            workflow_id="wf_legacy_s3_nosnap",
            goal_id=ga.goal_id,
            hypothesis_id=ha.hypothesis_id,
            primitive="direct_hq",
            status="RUNNING",
            config={
                "target_kind": "goal_synthesis",
                "synthesized_hypothesis_ids": [ha.hypothesis_id],
                "synthesis_key": "syn:legacy",
            },
            state={"step": "synthesis_review"},
        )
        db.save_workflow(legacy_wf)
        db.enqueue_task(
            StageTask(
                task_id="tsk_legacy_s3_rev",
                idempotency_key="ik_legacy_s3_rev",
                goal_id=ga.goal_id,
                hypothesis_id=ha.hypothesis_id,
                workflow_id=legacy_wf.workflow_id,
                stage=StageKind.RESULT_REVIEW,
                deps=[],
            )
        )
        many_revoked = [f"lease_hist_{i:03d}" for i in range(125)]
        db.set_global_config(
            "maintenance_barrier_fences",
            {"barrier_generation": 9, "active_lease_id": "", "revoked_lease_ids": many_revoked},
        )
        # Simulate pre-v6 database state (`user_version = 5`, no `revoked_barrier_leases`)
        with db._tx() as (conn, _p):
            conn.execute("DROP TABLE IF EXISTS revoked_barrier_leases")
            conn.execute("PRAGMA user_version = 5")

    # Re-open -> `_migrate_v6_schema` upgrades to `user_version == 6`, backfills ALL 125 revoked leases,
    # and fails closed on the snapshot-less S3 workflow (`SYNTHESIS_SUPERSEDED`) + cancels its READY task!
    with StateDB(arr_home) as db:
        with db._lock:
            uv8 = int(db._conn.execute("PRAGMA user_version").fetchone()[0])
        assert uv8 == 8
        assert len(db.list_revoked_barrier_leases()) == 125
        assert db.is_barrier_lease_revoked("lease_hist_000") is True
        assert db.is_barrier_lease_revoked("lease_hist_124") is True
        migrated_wf = db.get_workflow("wf_legacy_s3_nosnap")
        assert migrated_wf is not None and migrated_wf.status == "SYNTHESIS_SUPERSEDED"
        assert db.get_task("tsk_legacy_s3_rev").status == "CANCELLED"
        # §5 P2 #1: Migrated snapshot-less S3 workflow on ACTIVE goal automatically spawns a replacement S3 workflow!
        repl_s3_id = migrated_wf.state.get("replacement_workflow_id")
        assert repl_s3_id
        repl_s3_wf = db.get_workflow(str(repl_s3_id))
        assert repl_s3_wf is not None and repl_s3_wf.config.get("source_lineage_snapshot")

        # §5 P2 #1: `save_execution_plan` rejects workflow_id mismatch on existing `exec_plan_id` AND dual-key ambiguity!
        wf1 = db.save_workflow(
            WorkflowNode(workflow_id="wf_xp_1", goal_id=ga.goal_id, hypothesis_id=ha.hypothesis_id, primitive="direct_hq")
        )
        wf2 = db.save_workflow(
            WorkflowNode(workflow_id="wf_xp_2", goal_id=ga.goal_id, hypothesis_id=ha.hypothesis_id, primitive="direct_hq")
        )
        wf3 = db.save_workflow(
            WorkflowNode(workflow_id="wf_xp_3", goal_id=ga.goal_id, hypothesis_id=ha.hypothesis_id, primitive="direct_hq")
        )
        db.save_execution_plan(
            ExecutionPlan(
                exec_plan_id="xplan_fixed_1",
                workflow_id=wf1.workflow_id,
                goal_id=ga.goal_id,
                hypothesis_id=ha.hypothesis_id,
                goal_version=ga.version,
                human_epoch=ga.human_epoch,
                primitive="direct_hq",
                review_tier=ReviewTier.FAST_LANE,
            )
        )
        db.save_execution_plan(
            ExecutionPlan(
                exec_plan_id="xplan_fixed_2",
                workflow_id=wf2.workflow_id,
                goal_id=ga.goal_id,
                hypothesis_id=ha.hypothesis_id,
                goal_version=ga.version,
                human_epoch=ga.human_epoch,
                primitive="direct_hq",
                review_tier=ReviewTier.FAST_LANE,
            )
        )
        with pytest.raises(ValueError, match="workflow_id mismatch"):
            db.save_execution_plan(
                ExecutionPlan(
                    exec_plan_id="xplan_fixed_1",
                    workflow_id=wf3.workflow_id,
                    goal_id=ga.goal_id,
                    hypothesis_id=ha.hypothesis_id,
                    goal_version=ga.version,
                    human_epoch=ga.human_epoch,
                    primitive="direct_hq",
                    review_tier=ReviewTier.FAST_LANE,
                )
            )
        with pytest.raises(ValueError, match="Ambiguous dual-key ExecutionPlan collision"):
            db.save_execution_plan(
                ExecutionPlan(
                    exec_plan_id="xplan_fixed_1",
                    workflow_id=wf2.workflow_id,
                    goal_id=ga.goal_id,
                    hypothesis_id=ha.hypothesis_id,
                    goal_version=ga.version,
                    human_epoch=ga.human_epoch,
                    primitive="direct_hq",
                    review_tier=ReviewTier.FAST_LANE,
                )
            )

        # `P1-04-R3` + §5 P2 #2: Attempt-scoped `confirm_hq_cancellation` releases `a1` without touching `a2`!
        db.record_pending_hq_cancellation(
            hq_job_id=50101,
            entity_kind="task",
            entity_id="tsk_att_cancel",
            attempt=1,
            goal_id=ga.goal_id,
            hypothesis_id=ha.hypothesis_id,
            state="CANCEL_PENDING",
        )
        assert db.reserve_goal_budget(ga.goal_id, "task:tsk_att_cancel:a1", entity_kind="stage_task", entity_id="tsk_att_cancel", attempt=1, reserve_tokens=4000)[0] is True
        assert db.reserve_goal_budget(ga.goal_id, "task:tsk_att_cancel:a2", entity_kind="stage_task", entity_id="tsk_att_cancel", attempt=2, reserve_tokens=4000)[0] is True
        db.confirm_hq_cancellation(50101, entity_kind="task", entity_id="tsk_att_cancel", attempt=1)
        assert db.get_goal_token_usage(ga.goal_id)["reserved_tokens"] == 4000

        # `P1-07-R3`: `allow_same_holder_renew=False` rejects duplicate claim; `fenced_resolve_decision_request` rejects `None`/empty lease!
        l_excl = db.acquire_lease("decision:g:d_excl", "dec-d_excl", "same_holder", allow_same_holder_renew=False)
        assert l_excl is not None
        assert db.acquire_lease("decision:g:d_excl", "dec-d_excl", "same_holder", allow_same_holder_renew=False) is None
        dreq = db.enqueue_decision_request(ga.goal_id, role="research_lead", tier="L1", trigger_kind="TEST")
        db.resolve_decision_request(dreq["req_id"], status="RUNNING")
        with pytest.raises(FencedStaleWriteError, match="lease"):
            db.fenced_resolve_decision_request(
                dreq["req_id"],
                expected_goal_version=ga.version,
                expected_human_epoch=ga.human_epoch,
                expected_attempt=1,
                expected_lease_id=None,
                status="RESOLVED",
            )


def test_r4_v7_four_col_cancel_key_terminal_synthesis_superseded_and_task_cancel_reservation_retention(
    tmp_path: pathlib.Path,
) -> None:
    from conjectra.workflow_runtime import TERMINAL_WORKFLOW_STATUSES

    assert "SYNTHESIS_SUPERSEDED" in TERMINAL_WORKFLOW_STATUSES

    with StateDB(tmp_path / "arr_r4_s1") as db:
        ga = _sample_goal("goal_r4_a")
        ga.max_tokens_budget = 20_000
        db.save_goal(ga)
        ha, _ = db.register_hypothesis(_sample_hyp(ga.goal_id))

        # 1. §5 P2 #2: 4-column `(hq_job_id, entity_kind, entity_id, attempt)` conflict identity
        # preserves `a1` (`CANCEL_CONFIRMED`) without overwriting `attempt` or clobbering `a2` (`CANCEL_PENDING`)!
        db.record_pending_hq_cancellation(
            hq_job_id=60601,
            entity_kind="task",
            entity_id="tsk_4col",
            attempt=1,
            goal_id=ga.goal_id,
            hypothesis_id=ha.hypothesis_id,
            state="CANCEL_CONFIRMED",
        )
        db.record_pending_hq_cancellation(
            hq_job_id=60601,
            entity_kind="task",
            entity_id="tsk_4col",
            attempt=2,
            goal_id=ga.goal_id,
            hypothesis_id=ha.hypothesis_id,
            state="CANCEL_PENDING",
        )
        all_phq = [
            r
            for r in db.list_pending_hq_cancellations(goal_id=ga.goal_id, state=None)
            if r["entity_id"] == "tsk_4col"
        ]
        by_att = {int(r["attempt"]): r["state"] for r in all_phq}
        assert by_att == {1: "CANCEL_CONFIRMED", 2: "CANCEL_PENDING"}

        # 2. `P1-04-R4`: HQ-submitted task cancelled logically retains `'RESERVED'` until `confirm_hq_cancellation`,
        # whereas never-submitted `READY` task (`hq_job_id=None`) releases `'RESERVED'` immediately!
        t_ready_only = db.enqueue_task(
            StageTask(
                task_id="tsk_never_hq",
                idempotency_key="ik_never_hq",
                goal_id=ga.goal_id,
                hypothesis_id=ha.hypothesis_id,
                stage=StageKind.CODE,
                status="READY",
                deps=[],
            )
        )
        assert db.reserve_goal_budget(
            ga.goal_id, "task:tsk_never_hq:a1", entity_kind="stage_task", entity_id=t_ready_only.task_id, attempt=1, reserve_tokens=3000
        )[0] is True
        db.transition_task_status(t_ready_only.task_id, "READY", "CANCELLED")
        assert db.get_goal_token_usage(ga.goal_id)["reserved_tokens"] == 0

        t_hq = db.enqueue_task(
            StageTask(
                task_id="tsk_hq_submitted",
                idempotency_key="ik_hq_submitted",
                goal_id=ga.goal_id,
                hypothesis_id=ha.hypothesis_id,
                stage=StageKind.CODE,
                status="RUNNING",
                hq_job_id=60602,
                deps=[],
            )
        )
        assert db.reserve_goal_budget(
            ga.goal_id, "task:tsk_hq_submitted:a1", entity_kind="stage_task", entity_id=t_hq.task_id, attempt=1, reserve_tokens=4000
        )[0] is True
        db.record_pending_hq_cancellation(
            hq_job_id=60602,
            entity_kind="task",
            entity_id=t_hq.task_id,
            attempt=1,
            goal_id=ga.goal_id,
            hypothesis_id=ha.hypothesis_id,
            state="CANCEL_PENDING",
        )
        db.transition_task_status(t_hq.task_id, "RUNNING", "CANCELLED")
        assert db.get_goal_token_usage(ga.goal_id)["reserved_tokens"] == 4000
        db.confirm_hq_cancellation(60602, entity_kind="task", entity_id=t_hq.task_id, attempt=1)
        assert db.get_goal_token_usage(ga.goal_id)["reserved_tokens"] == 0


def test_stage5_v8_schema_migration_and_nonce_scoped_cancellation(tmp_path: pathlib.Path) -> None:
    """P1-03-R5: schema v8 adds usage_key to pending_hq_cancellations so confirming nonce A keeps nonce B RESERVED."""
    arr_home = tmp_path / "arr_v8_mig"
    with StateDB(arr_home) as db:
        ga = ResearchGoal(
            goal_id="g_v8",
            project="projgamma",
            title="V8 schema goal",
            direction="maximize",
            primary_metric="acc",
            baseline_value=0.5,
            baseline_sha="deadbeef",
            repo_path="/tmp/repo_v8",
            kill_criteria=[],
            banned_paths=["main"],
            writable_prefixes=["src/"],
            max_tokens_budget=50_000,
        )
        db.save_goal(ga)
        ha = HypothesisNode.from_dict(
            {
                "hypothesis_id": "h_v8",
                "goal_id": ga.goal_id,
                "kind": HypothesisKind.COMPETING_ROUTE.value,
                "mechanism": "mech_v8",
                "fidelity": FidelityRung.F1_FAST_SCREEN.value,
                "priority_evoi": 1.0,
                "writable_paths": ["src/"],
                "params": {},
                "kill_criteria": [],
                "acceptance_cmd": "pytest -q",
                "screen_cmd": "pytest -q",
                "confirm_cmd": None,
                "branch_name": "arr/v8",
                "base_sha": "deadbeef",
                "title": "H v8",
                "statement": "test v8 nonce isolation",
            }
        )
        db.register_hypothesis(ha)

    # Simulate v7 database lacking usage_key on pending_hq_cancellations
    import sqlite3 as _sqlite3

    raw = _sqlite3.connect(str(arr_home / "state.db"))
    raw.executescript(
        """
        DROP INDEX IF EXISTS idx_pending_hq_cancel_usage_key;
        DROP INDEX IF EXISTS idx_pending_hq_cancel_state;
        DROP TABLE pending_hq_cancellations;
        CREATE TABLE pending_hq_cancellations (
            cancel_id TEXT PRIMARY KEY,
            hq_job_id INTEGER NOT NULL,
            entity_kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,
            goal_id TEXT NOT NULL,
            hypothesis_id TEXT NOT NULL DEFAULT '',
            workflow_id TEXT NOT NULL DEFAULT '',
            epoch INTEGER NOT NULL DEFAULT 0,
            plan_epoch INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'CANCEL_PENDING',
            reason TEXT NOT NULL DEFAULT '',
            created_ts REAL NOT NULL,
            updated_ts REAL NOT NULL,
            confirmed_ts REAL,
            UNIQUE (hq_job_id, entity_kind, entity_id, attempt)
        );
        INSERT INTO pending_hq_cancellations VALUES
            ('hqc_8801', 8801, 'task', 't_old', 1, 'g_v8', 'h_v8', '', 0, 0, 'CANCEL_PENDING', 'legacy_v7', 10.0, 10.0, NULL);
        PRAGMA user_version = 7;
        """
    )
    raw.close()

    with StateDB(arr_home) as db:
        with db._lock:
            assert int(db._conn.execute("PRAGMA user_version").fetchone()[0]) == 8
            cols = {
                str(r[1])
                for r in db._conn.execute("PRAGMA table_info(pending_hq_cancellations)").fetchall()
            }
            assert "usage_key" in cols
            idxs = {
                str(r[1])
                for r in db._conn.execute("PRAGMA index_list(pending_hq_cancellations)").fetchall()
            }
            assert "idx_pending_hq_cancel_usage_key" in idxs

        # Same req_id + attempt=1, two distinct nonces
        ukey_a = "decision:dreq_n:a1:nonce_aaa"
        ukey_b = "decision:dreq_n:a1:nonce_bbb"
        assert db.reserve_goal_budget(
            "g_v8", ukey_a, entity_kind="decision", entity_id="dreq_n", attempt=1, reserve_tokens=1500
        )[0] is True
        assert db.reserve_goal_budget(
            "g_v8", ukey_b, entity_kind="decision", entity_id="dreq_n", attempt=1, reserve_tokens=1500
        )[0] is True
        assert db.get_goal_token_usage("g_v8")["reserved_tokens"] == 3000

        db.record_pending_hq_cancellation(
            hq_job_id=8802,
            entity_kind="decision",
            entity_id="dreq_n",
            attempt=1,
            goal_id="g_v8",
            state="CANCEL_PENDING",
            usage_key=ukey_a,
        )
        with db._lock:
            assert db._has_unconfirmed_hq_cancel_in_conn(db._conn, "decision", "dreq_n", 1, usage_key=ukey_a) is True
            assert db._has_unconfirmed_hq_cancel_in_conn(db._conn, "decision", "dreq_n", 1, usage_key=ukey_b) is False

        db.confirm_hq_cancellation(
            8802, entity_kind="decision", entity_id="dreq_n", attempt=1, usage_key=ukey_a
        )
        assert db.get_goal_token_usage("g_v8")["reserved_tokens"] == 1500
        with db._lock:
            rows = {
                str(r["usage_key"]): str(r["state"])
                for r in db._conn.execute(
                    "SELECT usage_key, state FROM token_usage_ledger WHERE goal_id = 'g_v8'"
                ).fetchall()
            }
        assert rows[ukey_a] == "RELEASED"
        assert rows[ukey_b] == "RESERVED"

        # R6 #3: stale nonce_A CANCEL_PENDING (attempt=1) MUST NOT block live nonce_B (attempt=1)
        # from releasing on 0-token FAILED / BUDGET_EXHAUSTED in resolve_decision_request / fenced_resolve_decision_request!
        r_f = db.enqueue_decision_request(
            "g_v8",
            tier="L1",
            role=ScientificRole.RESEARCH_LEAD,
            trigger_kind="EVIDENCE_UPDATE",
        )
        uk_sa = f"decision:{r_f['req_id']}:a1:stale_na"
        uk_lb = f"decision:{r_f['req_id']}:a1:live_nb"
        assert db.reserve_goal_budget(
            "g_v8", uk_sa, entity_kind="decision_request", entity_id=r_f["req_id"], attempt=1, reserve_tokens=1000
        )[0] is True
        assert db.reserve_goal_budget(
            "g_v8", uk_lb, entity_kind="decision_request", entity_id=r_f["req_id"], attempt=1, reserve_tokens=1000
        )[0] is True
        db.record_pending_hq_cancellation(
            hq_job_id=8803,
            entity_kind="decision",
            entity_id=r_f["req_id"],
            attempt=1,
            goal_id="g_v8",
            state="CANCEL_PENDING",
            usage_key=uk_sa,
        )
        with db._tx() as (conn, _p):
            conn.execute(
                "UPDATE decision_requests SET status = 'RUNNING', hq_job_id = 8804, payload_json = ? WHERE req_id = ?",
                (
                    json.dumps({**r_f["payload"], "submit_nonce": "live_nb", "usage_key": uk_lb}),
                    r_f["req_id"],
                ),
            )
        db.resolve_decision_request(
            r_f["req_id"], status="FAILED", resolution={"reason": "err", "in_tokens": 0, "out_tokens": 0}
        )
        with db._lock:
            led_rf = {
                str(r["usage_key"]): str(r["state"])
                for r in db._conn.execute(
                    "SELECT usage_key, state FROM token_usage_ledger WHERE entity_id = ?",
                    (r_f["req_id"],),
                ).fetchall()
            }
        assert led_rf[uk_lb] == "RELEASED"
        assert led_rf[uk_sa] == "RESERVED"

        # Same check for `fenced_resolve_decision_request`
        r_f2 = db.enqueue_decision_request(
            "g_v8",
            tier="L1",
            role=ScientificRole.RESEARCH_LEAD,
            trigger_kind="EVIDENCE_UPDATE",
        )
        uk_sa2 = f"decision:{r_f2['req_id']}:a1:stale_na2"
        uk_lb2 = f"decision:{r_f2['req_id']}:a1:live_nb2"
        assert db.reserve_goal_budget(
            "g_v8", uk_sa2, entity_kind="decision_request", entity_id=r_f2["req_id"], attempt=1, reserve_tokens=900
        )[0] is True
        assert db.reserve_goal_budget(
            "g_v8", uk_lb2, entity_kind="decision_request", entity_id=r_f2["req_id"], attempt=1, reserve_tokens=900
        )[0] is True
        db.record_pending_hq_cancellation(
            hq_job_id=8805,
            entity_kind="decision",
            entity_id=r_f2["req_id"],
            attempt=1,
            goal_id="g_v8",
            state="CANCEL_PENDING",
            usage_key=uk_sa2,
        )
        with db._tx() as (conn, _p):
            conn.execute(
                "UPDATE decision_requests SET status = 'RUNNING', hq_job_id = 8806, payload_json = ? WHERE req_id = ?",
                (
                    json.dumps({**r_f2["payload"], "submit_nonce": "live_nb2", "usage_key": uk_lb2}),
                    r_f2["req_id"],
                ),
            )
        ls_rf2 = db.acquire_lease(
            f"decision:g_v8:{r_f2['req_id']}",
            f"dec-{r_f2['req_id']}",
            holder="test-rf2",
            ttl_s=60.0,
            goal_id="g_v8",
        )
        assert ls_rf2 is not None
        db.fenced_resolve_decision_request(
            r_f2["req_id"],
            expected_goal_version=ga.version,
            expected_human_epoch=ga.human_epoch,
            expected_attempt=1,
            expected_lease_id=ls_rf2["lease_id"],
            expected_statuses=("RUNNING",),
            status="BUDGET_EXHAUSTED",
            resolution={"reason": "exhausted", "in_tokens": 0, "out_tokens": 0},
        )
        with db._lock:
            led_rf2 = {
                str(r["usage_key"]): str(r["state"])
                for r in db._conn.execute(
                    "SELECT usage_key, state FROM token_usage_ledger WHERE entity_id = ?",
                    (r_f2["req_id"],),
                ).fetchall()
            }
        assert led_rf2[uk_lb2] == "RELEASED"
        assert led_rf2[uk_sa2] == "RESERVED"


def test_r6_v7_legacy_decision_cancellation_migration_never_releases_sibling_nonce(
    tmp_path: pathlib.Path,
) -> None:
    """P1-01-R6: migrating v7 empty-usage_key decision cancellation and confirming A must keep B RESERVED."""
    import sqlite3

    arr_home = tmp_path / "arr_r6_v7_mig"
    db_seed = StateDB(arr_home)
    ga = ResearchGoal(
        goal_id="g_r6_v7",
        project="projgamma",
        title="R6 v7 legacy cancel test",
        direction="maximize",
        primary_metric="score",
        baseline_value=0.5,
        baseline_sha="deadbeef",
        repo_path=str(tmp_path),
        max_tokens_budget=100_000,
        kill_criteria=[],
        banned_paths=[],
        writable_prefixes=["src/"],
    )
    db_seed.save_goal(ga)
    dreq_m = db_seed.enqueue_decision_request(
        "g_r6_v7",
        tier="L1",
        role=ScientificRole.RESEARCH_LEAD,
        trigger_kind="EVIDENCE_UPDATE",
    )
    rid = str(dreq_m["req_id"])
    uk_a = f"decision:{rid}:a1:nonce_a"
    uk_b = f"decision:{rid}:a1:nonce_b"
    assert db_seed.reserve_goal_budget(
        "g_r6_v7", uk_a, entity_kind="decision", entity_id=rid, attempt=1, reserve_tokens=1100
    )[0] is True
    assert db_seed.reserve_goal_budget(
        "g_r6_v7", uk_b, entity_kind="decision", entity_id=rid, attempt=1, reserve_tokens=1300
    )[0] is True
    db_seed.close()

    # Downgrade pending_hq_cancellations to v7 schema (without usage_key column) + legacy CANCEL_PENDING row for A
    raw = sqlite3.connect(str(arr_home / "state.db"))
    raw.executescript(
        f"""
        DROP TABLE pending_hq_cancellations;
        CREATE TABLE pending_hq_cancellations (
            cancel_id TEXT PRIMARY KEY,
            hq_job_id INTEGER NOT NULL,
            entity_kind TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 1,
            goal_id TEXT NOT NULL,
            hypothesis_id TEXT NOT NULL DEFAULT '',
            workflow_id TEXT NOT NULL DEFAULT '',
            epoch INTEGER NOT NULL DEFAULT 0,
            plan_epoch INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'CANCEL_PENDING',
            reason TEXT NOT NULL DEFAULT '',
            created_ts REAL NOT NULL,
            updated_ts REAL NOT NULL,
            confirmed_ts REAL,
            UNIQUE (hq_job_id, entity_kind, entity_id, attempt)
        );
        INSERT INTO pending_hq_cancellations VALUES
            ('hqc_r6_legacy', 9901, 'decision', '{rid}', 1, 'g_r6_v7', '', '', 0, 0, 'CANCEL_PENDING', 'legacy_v7', 10.0, 10.0, NULL);
        PRAGMA user_version = 7;
        """
    )
    raw.commit()
    raw.close()

    with StateDB(arr_home) as db_mig:
        with db_mig._lock:
            row_mig = db_mig._conn.execute(
                "SELECT usage_key FROM pending_hq_cancellations WHERE cancel_id = 'hqc_r6_legacy'"
            ).fetchone()
            assert row_mig is not None
            assert str(row_mig["usage_key"]).startswith("legacy_ambiguous:")
        db_mig.confirm_hq_cancellation(
            9901, entity_kind="decision", entity_id=rid, attempt=1, usage_key=uk_a
        )
        with db_mig._lock:
            led_after = {
                str(r["usage_key"]): str(r["state"])
                for r in db_mig._conn.execute(
                    "SELECT usage_key, state FROM token_usage_ledger WHERE entity_id = ?",
                    (rid,),
                ).fetchall()
            }
        assert led_after[uk_a] == "RELEASED"
        assert led_after[uk_b] == "RESERVED"


def test_slice_1_sources_contain_no_prohibited_internal_ids() -> None:
    staging_root = pathlib.Path(__file__).resolve().parents[1]
    for rel in (
        "conjectra/types.py",
        "conjectra/state_db.py",
        "conjectra/gates.py",
        "conjectra/planning.py",
    ):
        text = (staging_root / rel).read_text(encoding="utf-8")
        # Exclude regex definition literals in gates.py itself
        if rel == "conjectra/gates.py":
            lines = [
                ln
                for ln in text.splitlines()
                if "re.compile(" not in ln
            ]
            text = "\n".join(lines)
        gate = check_no_internal_ids(text)
        assert gate.passed is True, f"{rel} failed check_no_internal_ids: {gate.failures}"
