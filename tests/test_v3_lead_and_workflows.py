"""Comprehensive V3 tests for role-typed ResearchLead/Director, failure-isolated Explore, and 3-way WorkflowRuntime."""

from __future__ import annotations

import json
import pathlib
from typing import Any, Sequence

import pytest

from conjectra.backends import DEFAULT_BACKEND_FOR_POOL
from conjectra.decision_engine import (
    DECISION_INNER_TIMEOUT_S,
    DECISION_OUTER_WALL_TIMEOUT_S,
    DecisionEngine,
    parse_lead_decision_payload,
)
from conjectra.explore import (
    apply_scientific_actions,
    check_kill_criteria,
    reconcile_receipt,
)
from conjectra.gates import check_no_internal_ids
from conjectra.state_db import StateDB
from conjectra.types import (
    ActionKind,
    AuditVerdict,
    ConstraintStrength,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    FailureDomain,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    LeadAction,
    LeadDecisionEnvelope,
    ProcessStatus,
    ResearchGoal,
    ResolvedModelRoute,
    ReviewKind,
    ReviewTier,
    ScientificRole,
    ScientificVerdict,
    StageKind,
)
from conjectra.workflow_runtime import WorkflowRuntime


class _FakeModelRouter:
    """Deterministic `RouteResolverProtocol` test double verifying zero-import DI."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._candidates = [
            ("codex", "gpt-5-codex", "high", "openai", "llmcodexaudit"),
            ("gemini", "gemini-pro", "high", "google", "llmgemini"),
            ("deepseek", "deepseek-chat", "high", "deepseek", "llmdeepseek"),
            ("claude", "claude-opus-5-5", "xhigh", "anthropic", "llmclaude"),
        ]

    def resolve_route(
        self,
        role: ScientificRole | str,
        *,
        goal: ResearchGoal | None = None,
        hyp: HypothesisNode | None = None,
        explicit_spec: str | None = None,
        avoid_families: Sequence[str] = (),
        exclude_specs: Sequence[str] = (),
        execution_host: str | None = None,
    ) -> ResolvedModelRoute:
        r_str = role.value if isinstance(role, ScientificRole) else str(role)
        avoid = {str(f).lower() for f in avoid_families}
        excl = {str(s).lower() for s in exclude_specs}
        self.calls.append({"role": r_str, "avoid_families": sorted(avoid), "explicit_spec": explicit_spec})
        for pool, model, effort, fam, hq_res in self._candidates:
            spec = f"{pool}:{model}:{effort}"
            if fam in avoid or spec.lower() in excl:
                continue
            return ResolvedModelRoute(
                backend_mode=DEFAULT_BACKEND_FOR_POOL.get(pool, "dispatch_script"),
                execution_host=execution_host or "local",
                multipool_host_arg="local",
                pool=pool,
                canonical_model=model,
                requested_effort=effort,
                wire_effort=effort,
                family=fam,
                logical_pool=f"llm_{pool}",
                hq_resource=hq_res,
                strength=ConstraintStrength.HARD,
                matched_rule_id=f"mock_{r_str}",
            )
        raise RuntimeError(f"No route outside {sorted(avoid)}")


def _goal(goal_id: str = "g_v3_lw", *, risk_profile: str = "standard") -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project="projgamma",
        title="Derive spectral convergence rate",
        direction="maximize",
        primary_metric="spectral_gap",
        baseline_value=0.40,
        baseline_sha="1111111aaaa",
        repo_path="/tmp/projgamma_repo",
        kill_criteria=["spectral_gap < 0.20", "tests_passed == 0"],
        banned_paths=["sealed_eval/", "lockbox/"],
        writable_prefixes=["src/", "tests/"],
        max_tokens_budget=250_000,
        version=1,
        human_epoch=1,
        risk_profile=risk_profile,
        routing_policy_nl="PROJGAMMA 理论创新用 Opus xhigh，重大理论必须跨家族审查",
    )


def _hyp(
    goal: ResearchGoal,
    hid: str = "h_line1",
    *,
    tags: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id=hid,
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title=f"Line {hid}",
        statement="Improve spectral_gap above 0.55",
        mechanism=f"mech_{hid}",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.9,
        writable_paths=["src/line1.py", "tests/test_line1.py"],
        params=dict(params or {"requires_llm_code": True}),
        kill_criteria=["spectral_gap < 0.20", "tests_passed == 0"],
        acceptance_cmd="pytest -q tests/test_line1.py",
        screen_cmd="python3 -m src.run_screen",
        confirm_cmd="python3 -m src.run_confirm",
        branch_name=f"arr/projgamma/{hid}",
        base_sha=goal.baseline_sha,
        tags=list(tags or []),
    )


def _rcpt(
    hyp: HypothesisNode,
    *,
    rid: str,
    tid: str,
    stage: StageKind,
    pstatus: ProcessStatus = ProcessStatus.EXIT_OK,
    everdict: EngineeringVerdict = EngineeringVerdict.PASS,
    sverdict: ScientificVerdict = ScientificVerdict.CONFIRMED,
    averdict: AuditVerdict = AuditVerdict.NOT_REQUIRED,
    validity: EvidenceValidity = EvidenceValidity.VALID_EMPIRICAL,
    fdomain: FailureDomain = FailureDomain.NONE,
    metrics: dict[str, Any] | None = None,
    reason: str = "",
) -> EvidenceReceipt:
    return EvidenceReceipt(
        receipt_id=rid,
        task_id=tid,
        goal_id=hyp.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=stage,
        base_sha=hyp.base_sha,
        head_sha="2222222bbbb",
        command="pytest -q",
        exit_code=0 if pstatus == ProcessStatus.EXIT_OK else 1,
        wall_sec=8.0,
        max_rss_mb=128.0,
        in_tokens=300,
        out_tokens=120,
        model_used="codex:gpt-5-codex:high",
        process_status=pstatus,
        engineering_verdict=everdict,
        scientific_verdict=sverdict,
        audit_verdict=averdict,
        tests_passed=4 if everdict == EngineeringVerdict.PASS else 0,
        tests_total=4,
        metrics={"spectral_gap": 0.62} if metrics is None else metrics,
        raw_artifact_paths=["/tmp/out.json"],
        failure_reason=reason,
        evidence_validity=validity,
        goal_version=1,
        human_epoch=1,
        failure_domain=fdomain,
    )


# ---- 1. Failure-Isolated Explore (`check_kill_criteria`, `reconcile_receipt`, budgets)


def test_explore_isolates_audit_fail_cuda_cli_timeout_from_graveyard(
    tmp_path: pathlib.Path,
) -> None:
    with StateDB(tmp_path / "arr") as db:
        goal = _goal()
        db.save_goal(goal)

        # Case 1: BLIND_AUDIT / IMPL_REVIEW FAIL (PROJGAMMA Lines 1/2/5 forensic bug) -> NEEDS_REPAIR, 0 graveyard!
        h_audit = _hyp(goal, "h_projgamma_audit")
        db.register_hypothesis(h_audit)
        r_audit = _rcpt(
            h_audit,
            rid="r_aud",
            tid="t_aud",
            stage=StageKind.IMPL_REVIEW,
            sverdict=ScientificVerdict.UNKNOWN,
            averdict=AuditVerdict.FAIL,
            validity=EvidenceValidity.VALID_ENGINEERING_ONLY,
            reason="Adversary found missing normalization factor",
        )
        assert check_kill_criteria(goal, h_audit, r_audit) is None
        dec_aud = reconcile_receipt(db, goal, h_audit, r_audit)
        assert dec_aud.new_status == "NEEDS_REPAIR"
        assert dec_aud.graveyard_recorded is False
        assert dec_aud.spawned_hypotheses == []
        assert db.list_graveyard(goal.goal_id) == []

        # Case 2: ProjBeta cuda_unavailable + tests_passed == 0 -> BLOCKED_ENV, 0 graveyard!
        h_cuda = _hyp(goal, "h_pb_cuda")
        db.register_hypothesis(h_cuda)
        r_cuda = _rcpt(
            h_cuda,
            rid="r_cuda",
            tid="t_cuda",
            stage=StageKind.EXP_SCREEN,
            pstatus=ProcessStatus.EXIT_NONZERO,
            everdict=EngineeringVerdict.FAIL_TEST,
            sverdict=ScientificVerdict.REFUTED,
            validity=EvidenceValidity.MISSING_INPUT,
            metrics={"tests_passed": 0, "skip_reason": "cuda_unavailable"},
            reason="RuntimeError: cuda_unavailable",
        )
        assert check_kill_criteria(goal, h_cuda, r_cuda) is None
        dec_cuda = reconcile_receipt(db, goal, h_cuda, r_cuda)
        assert dec_cuda.new_status == "BLOCKED_ENV"
        assert dec_cuda.scientific_verdict == ScientificVerdict.UNKNOWN
        assert dec_cuda.graveyard_recorded is False
        assert db.list_graveyard(goal.goal_id) == []

        # Case 3: CPU_GATE TIMEOUT -> BLOCKED_TIMEOUT + UNKNOWN, 0 graveyard, 0 child spawns!
        h_to = _hyp(goal, "h_gate_timeout")
        db.register_hypothesis(h_to)
        r_to = _rcpt(
            h_to,
            rid="r_to",
            tid="t_to",
            stage=StageKind.CPU_GATE,
            pstatus=ProcessStatus.TIMEOUT,
            everdict=EngineeringVerdict.FAIL_TEST,
            sverdict=ScientificVerdict.UNKNOWN,
            validity=EvidenceValidity.VALID_ENGINEERING_ONLY,
        )
        dec_to = reconcile_receipt(db, goal, h_to, r_to)
        assert dec_to.new_status == "BLOCKED_TIMEOUT"
        assert dec_to.scientific_verdict == ScientificVerdict.UNKNOWN
        assert dec_to.spawned_hypotheses == []
        assert db.list_graveyard(goal.goal_id) == []


# ---- 2. Role-Typed DecisionEngine: Single-Attempt, No Live Dummy Fallback, All 13 Actions


def test_decision_engine_single_attempt_no_live_dummy_fallback_and_13_actions(
    tmp_path: pathlib.Path,
) -> None:
    assert DECISION_INNER_TIMEOUT_S == 180
    assert DECISION_OUTER_WALL_TIMEOUT_S == 225

    with StateDB(tmp_path / "arr") as db:
        goal = _goal()
        db.save_goal(goal)
        h1 = _hyp(goal, "h_target_1")
        h2 = _hyp(goal, "h_target_2")
        h3 = _hyp(goal, "h_unverified_refute")
        h4 = _hyp(goal, "h_verified_refute")
        for h in (h1, h2, h3, h4):
            db.register_hypothesis(h)

        # Record a genuine empirical refutation receipt ONLY on h4
        db.record_receipt(
            _rcpt(
                h4,
                rid="r_h4_emp",
                tid="t_h4_emp",
                stage=StageKind.EXP_SCREEN,
                sverdict=ScientificVerdict.REFUTED,
                fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
                metrics={"spectral_gap": 0.12},
                reason="spectral_gap < 0.20",
            )
        )

        router = _FakeModelRouter()
        rt = WorkflowRuntime(db, router=router)
        engine = DecisionEngine(db, router=router, workflow_runtime=rt)

        # 1. Live LLM failure (`exit_code=1`) runs AT MOST 1 attempt and NEVER injects dummy "pytest -q" proposals!
        req_fail = engine.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            trigger_kind="EVIDENCE_UPDATE",
            urgency="IMMEDIATE",
        )
        call_count = 0

        def _failing_llm(_prov: Any, _prompt: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"exit_code": 1, "response": "", "in_tokens": 50, "out_tokens": 0}

        out_fail = engine.process_request(req_fail["req_id"], llm_caller=_failing_llm, dry_run_llm=False)
        assert call_count == 1
        assert out_fail["status"] == "FAILED"
        assert out_fail["resolution"]["used_fallback"] is False
        assert out_fail["resolution"]["accepted_hypothesis_ids"] == []
        assert len(db.list_hypotheses(goal.goal_id)) == 4

        # 2. Hypothesis-scoped triage budget (`max_triage_decisions = 2`) & coalesce_key deduplication:
        # Coalescing into an existing PENDING request MUST NOT double-increment `triage_decisions_used`!
        t1 = engine.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            hypothesis_id=h1.hypothesis_id,
            head_sha="shaA",
            failure_domain=FailureDomain.ENGINEERING_DEFECT,
            now=100.0,
        )
        t2 = engine.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            hypothesis_id=h1.hypothesis_id,
            head_sha="shaA",
            failure_domain=FailureDomain.ENGINEERING_DEFECT,
            now=101.0,
        )
        assert t1["req_id"] == t2["req_id"]
        assert db.get_hypothesis(h1.hypothesis_id).budget_state["triage_decisions_used"] == 1

        # Second distinct triage key charges slot 2/2; third distinct key is suppressed!
        t_second = engine.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            hypothesis_id=h1.hypothesis_id,
            head_sha="shaB",
            failure_domain=FailureDomain.ENGINEERING_DEFECT,
            now=102.0,
        )
        assert t_second["req_id"] != t1["req_id"]
        assert db.get_hypothesis(h1.hypothesis_id).budget_state["triage_decisions_used"] == 2

        t3 = engine.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            hypothesis_id=h1.hypothesis_id,
            head_sha="shaC",
            failure_domain=FailureDomain.ENGINEERING_DEFECT,
            now=103.0,
        )
        assert t3["status"] == "SUPPRESSED_BY_TRIAGE_BUDGET"

        # Standalone DecisionEngine (workflow_runtime omitted) still attaches REPAIR_IMPLEMENTATION
        # onto a live WorkflowNode with non-null `workflow_id`, `llm_pool`, and `pinned_route`!
        standalone_engine = DecisionEngine(db, router=router)
        assert standalone_engine.workflow_runtime is not None
        h_repair_only = _hyp(goal, "h_repair_only")
        db.register_hypothesis(h_repair_only)
        rep_summary = apply_scientific_actions(
            db,
            goal,
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=goal.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.REPAIR_IMPLEMENTATION,
                        target_hypothesis_id=h_repair_only.hypothesis_id,
                        rationale="Repair normalization constant",
                    )
                ],
            ),
            workflow_runtime=standalone_engine.workflow_runtime,
        )
        assert len(rep_summary["emitted_task_ids"]) == 1
        rep_task = db.get_task(rep_summary["emitted_task_ids"][0])
        assert rep_task is not None
        assert rep_task.workflow_id is not None
        assert rep_task.resources.llm_pool is not None
        assert rep_task.pinned_route is not None
        assert rep_task.pinned_route["family"] in {"openai", "google", "deepseek", "anthropic"}

        # 3. Execute all 13 typed `ActionKind`s through `LeadDecisionEnvelope`!
        db.record_receipt(
            _rcpt(
                h2,
                rid="r_h2_conf",
                tid="t_h2_conf",
                stage=StageKind.EXP_CONFIRM,
                sverdict=ScientificVerdict.CONFIRMED,
                fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
                metrics={"spectral_gap": 0.68},
            )
        )
        envelope = LeadDecisionEnvelope(
            role=ScientificRole.RESEARCH_LEAD,
            goal_id=goal.goal_id,
            goal_version=1,
            human_epoch=1,
            uncertainty_summary="Synthesizing Line 1-4 outcomes and dispatching bounded repairs",
            actions=[
                LeadAction(action=ActionKind.CONTINUE, rationale="Keep baseline control active"),
                LeadAction(
                    action=ActionKind.REVISE_HYPOTHESIS,
                    target_hypothesis_id=h1.hypothesis_id,
                    parameters={
                        "params_patch": {"degree": 6},
                        "experiment_plan": {"gate_timeout_s": 90, "screen_timeout_s": 360},
                    },
                ),
                LeadAction(
                    action=ActionKind.PROPOSE_COMPETING_ROUTE,
                    uncertainty_id="u_lanczos",
                    parameters={
                        "proposal": {
                            "title": "Lanczos tridiagonal route",
                            "statement": "Lanczos achieves faster spectral separation",
                            "mechanism": "lanczos_tridiag_v1",
                            "writable_paths": ["src/lanczos.py"],
                            "kill_criteria": ["spectral_gap < 0.20"],
                            "acceptance_cmd": "pytest -q tests/test_lanczos.py",
                            "screen_cmd": "python3 -m src.run_lanczos",
                        }
                    },
                ),
                LeadAction(
                    action=ActionKind.ADD_THEORETICAL_DERIVATION,
                    uncertainty_id="u_bound",
                    parameters={
                        "proposal": {
                            "title": "Tight eigenvalue perturbation lemma",
                            "statement": "Prove O(1/k^2) error bound",
                            "mechanism": "weyl_perturbation_lemma",
                            "writable_paths": ["src/lemma.py"],
                            "kill_criteria": ["spectral_gap < 0.20"],
                            "acceptance_cmd": "pytest -q tests/test_lemma.py",
                            "screen_cmd": "python3 -m src.verify_lemma",
                        }
                    },
                ),
                LeadAction(
                    action=ActionKind.ADD_DISCRIMINATING_EXPERIMENT,
                    target_hypothesis_id=h2.hypothesis_id,
                    uncertainty_id="u_disc",
                    parameters={
                        "proposal": {
                            "title": "Ill-conditioned matrix stress test",
                            "statement": "Discriminate Chebyshev vs Lanczos on kappa=1e6",
                            "mechanism": "kappa_1e6_stress",
                            "writable_paths": ["tests/test_stress.py"],
                            "kill_criteria": ["spectral_gap < 0.20"],
                            "acceptance_cmd": "pytest -q tests/test_stress.py",
                            "screen_cmd": "python3 -m src.run_stress",
                        }
                    },
                ),
                LeadAction(
                    action=ActionKind.REPAIR_IMPLEMENTATION,
                    target_hypothesis_id=h1.hypothesis_id,
                    rationale="Fix boundary normalization bug",
                ),
                LeadAction(
                    action=ActionKind.SWITCH_WORKFLOW,
                    target_hypothesis_id=h1.hypothesis_id,
                    workflow_primitive="x-teamwork",
                    rationale="Split into 2 disjoint slices",
                    parameters={
                        "config": {
                            "require_design_review": False,
                            "phases": [
                                {
                                    "name": "p1",
                                    "slices": [
                                        {"name": "s1", "writable_paths": ["src/line1.py"]},
                                        {"name": "s2", "writable_paths": ["tests/test_line1.py"]},
                                    ],
                                }
                            ],
                        }
                    },
                ),
                LeadAction(
                    action=ActionKind.SYNTHESIZE_RESULTS,
                    uncertainty_id="u_syn",
                    rationale="Combine F2 findings into updated ResearchPlan",
                    parameters={"thesis": "Lanczos + Chebyshev hybrid dominates", "uncertainties": [{"id": "u_syn"}]},
                ),
                LeadAction(
                    action=ActionKind.REQUEST_DIRECTOR_ADJUDICATION,
                    target_hypothesis_id=h2.hypothesis_id,
                    rationale="Confirm milestone readiness",
                ),
                # Unverified refutation on h3 (no empirical receipt) MUST NOT write graveyard!
                LeadAction(
                    action=ActionKind.TERMINATE_REFUTED_ROUTE,
                    target_hypothesis_id=h3.hypothesis_id,
                    rationale="Try terminating without empirical receipt",
                ),
                # Verified empirical refutation on h4 writes graveyard cleanly!
                LeadAction(
                    action=ActionKind.TERMINATE_REFUTED_ROUTE,
                    target_hypothesis_id=h4.hypothesis_id,
                    rationale="Empirically refuted at spectral_gap=0.12",
                ),
                LeadAction(
                    action=ActionKind.COMPLETE_MILESTONE,
                    target_hypothesis_id=h2.hypothesis_id,
                    parameters={"milestone_id": "m_line2"},
                ),
                LeadAction(
                    action=ActionKind.PRUNE_BRANCH,
                    target_hypothesis_id=h3.hypothesis_id,
                    rationale="Redundant branch",
                ),
                LeadAction(
                    action=ActionKind.REQUEST_OWNER_DECISION,
                    rationale="Need GPU budget sign-off for scale run",
                ),
            ],
        )

        # Round-trip parse through `parse_lead_decision_payload`
        parsed_env = parse_lead_decision_payload(
            "```json\n" + json.dumps(envelope.to_dict()) + "\n```",
            goal=goal,
        )
        assert len(parsed_env.actions) == 14

        summary = apply_scientific_actions(db, goal, parsed_env, workflow_runtime=rt)
        assert summary["continued"] == 1
        assert h1.hypothesis_id in summary["revised_hypothesis_ids"]
        assert len(summary["accepted_hypothesis_ids"]) == 3
        assert h1.hypothesis_id in summary["repaired_hypothesis_ids"]
        assert len(summary["switched_workflows"]) == 1
        assert summary["switched_workflows"][0]["primitive"] == "x-teamwork"
        assert summary["syntheses_recorded"] == 1
        assert len(summary["director_requests"]) == 1
        assert summary["blocked_unverified_refutations"] == [h3.hypothesis_id]
        assert summary["terminated_hypothesis_ids"] == [h4.hypothesis_id]
        assert summary["completed_milestones"] == ["m_line2"]
        assert len(summary["owner_escalations"]) == 1

        # Verify graveyard ONLY contains h4 (NOT h3!)
        gy_ids = [g["hypothesis_id"] for g in db.list_graveyard(goal.goal_id)]
        assert gy_ids == [h4.hypothesis_id]


# ---- 3. Unified WorkflowRuntime: R1 Design Review, x-boost Max-Repair Re-Audit, EXP_CONFIRM & Red-Team Tie-Breaker


def test_xboost_mandatory_reaudit_at_max_repairs_and_full_confirm_pipeline(
    tmp_path: pathlib.Path,
) -> None:
    with StateDB(tmp_path / "arr") as db:
        goal = _goal()
        db.save_goal(goal)
        hyp = _hyp(goal, "h_xb_strict", params={"requires_llm_code": True, "max_repairs": 1})
        db.register_hypothesis(hyp)

        router = _FakeModelRouter()
        rt = WorkflowRuntime(db, router=router)
        wf, init_tasks = rt.create_workflow_for_hypothesis(
            goal,
            hyp,
            primitive="x-boost",
            config_override={"max_repairs": 1, "require_design_review": False},
        )
        assert len(init_tasks) == 1 and init_tasks[0].step_key == "boost_build"

        # Build -> Gate -> Adversary 0 (FAIL) -> Repair 1 (`repairs_used == max_repairs == 1`) -> Gate r1
        t_gate0 = rt.advance_workflow(
            wf, init_tasks[0], _rcpt(hyp, rid="r1", tid=init_tasks[0].task_id, stage=StageKind.CODE), goal
        )[0]
        t_adv0 = rt.advance_workflow(
            wf, t_gate0, _rcpt(hyp, rid="r2", tid=t_gate0.task_id, stage=StageKind.CPU_GATE), goal
        )[0]
        assert t_adv0.step_key == "boost_adversary_0"
        assert t_adv0.stage == StageKind.IMPL_REVIEW

        t_rep1 = rt.advance_workflow(
            wf,
            t_adv0,
            _rcpt(hyp, rid="r3", tid=t_adv0.task_id, stage=StageKind.IMPL_REVIEW, averdict=AuditVerdict.FAIL),
            goal,
        )[0]
        assert t_rep1.step_key == "boost_repair_1"

        t_gate1 = rt.advance_workflow(
            wf, t_rep1, _rcpt(hyp, rid="r4", tid=t_rep1.task_id, stage=StageKind.CODE), goal
        )[0]
        assert t_gate1.step_key == "boost_gate_r1"

        # CRITICAL V3 FIX: Passing `boost_gate_r1` at `repairs_used == max_repairs` MUST emit `boost_adversary_1`
        # (NEVER jumping directly to `exp_screen` unaudited!)
        after_gate1 = rt.advance_workflow(
            wf, t_gate1, _rcpt(hyp, rid="r5", tid=t_gate1.task_id, stage=StageKind.CPU_GATE), goal
        )
        assert len(after_gate1) == 1
        assert after_gate1[0].step_key == "boost_adversary_1"
        assert after_gate1[0].stage == StageKind.IMPL_REVIEW

        # Now `boost_adversary_1` PASSES -> emits `exp_screen` -> `exp_confirm` -> `boost_result_review` -> `COMPLETED`!
        t_scr = rt.advance_workflow(
            wf,
            after_gate1[0],
            _rcpt(hyp, rid="r6", tid=after_gate1[0].task_id, stage=StageKind.IMPL_REVIEW, averdict=AuditVerdict.PASS),
            goal,
        )[0]
        assert t_scr.stage == StageKind.EXP_SCREEN

        t_cnf = rt.advance_workflow(
            wf,
            t_scr,
            _rcpt(hyp, rid="r7", tid=t_scr.task_id, stage=StageKind.EXP_SCREEN, sverdict=ScientificVerdict.CONFIRMED),
            goal,
        )[0]
        assert t_cnf.stage == StageKind.EXP_CONFIRM

        t_r3 = rt.advance_workflow(
            wf,
            t_cnf,
            _rcpt(hyp, rid="r8", tid=t_cnf.task_id, stage=StageKind.EXP_CONFIRM, sverdict=ScientificVerdict.CONFIRMED),
            goal,
        )[0]
        assert t_r3.stage == StageKind.RESULT_REVIEW
        assert t_r3.step_key == "boost_result_review"

        done_tasks = rt.advance_workflow(
            wf,
            t_r3,
            _rcpt(hyp, rid="r9", tid=t_r3.task_id, stage=StageKind.RESULT_REVIEW, averdict=AuditVerdict.PASS),
            goal,
        )
        assert done_tasks == []
        assert db.get_workflow(wf.workflow_id).status == "COMPLETED"


def test_red_team_design_review_tiebreaker_and_deadline_sweep(
    tmp_path: pathlib.Path,
) -> None:
    with StateDB(tmp_path / "arr") as db:
        goal = _goal(risk_profile="high_stakes")
        db.save_goal(goal)
        hyp = _hyp(
            goal,
            "h_theory_rt",
            tags=["theory"],
            params={"requires_llm_code": True, "proposer_family": "anthropic"},
        )
        db.register_hypothesis(hyp)

        router = _FakeModelRouter()
        rt = WorkflowRuntime(db, router=router)

        # High-stakes `theory` hypothesis starts with 2 parallel cross-family R1 `DESIGN_REVIEW` tasks (`slot 0` & `slot 1`)
        # and BOTH exclude upstream `proposer_family = anthropic`!
        wf, dr_tasks = rt.create_workflow_for_hypothesis(goal, hyp, primitive="direct_hq")
        assert len(dr_tasks) == 2
        assert {t.stage for t in dr_tasks} == {StageKind.DESIGN_REVIEW}
        families = {t.pinned_route["family"] for t in dr_tasks if t.pinned_route}
        assert "anthropic" not in families
        assert len(families) == 2

        # Slot 0 PASS + Slot 1 FAIL -> spawns Director tie-breaker `design_review_s2` on a 3rd distinct family!
        assert (
            rt.advance_workflow(
                wf,
                dr_tasks[0],
                _rcpt(hyp, rid="dr0", tid=dr_tasks[0].task_id, stage=StageKind.DESIGN_REVIEW, averdict=AuditVerdict.PASS),
                goal,
            )
            == []
        )
        tb_tasks = rt.advance_workflow(
            wf,
            dr_tasks[1],
            _rcpt(hyp, rid="dr1", tid=dr_tasks[1].task_id, stage=StageKind.DESIGN_REVIEW, averdict=AuditVerdict.FAIL),
            goal,
        )
        assert len(tb_tasks) == 1
        assert tb_tasks[0].step_key == "design_review_s2"
        assert tb_tasks[0].pinned_route is not None
        assert tb_tasks[0].pinned_route["family"] not in (families | {"anthropic"})

        # Director tie-breaker PASSES -> approves ExperimentPlan and emits `CODE` build task with `pinned_route`!
        code_tasks = rt.advance_workflow(
            wf,
            tb_tasks[0],
            _rcpt(hyp, rid="dr2", tid=tb_tasks[0].task_id, stage=StageKind.DESIGN_REVIEW, averdict=AuditVerdict.PASS),
            goal,
        )
        assert len(code_tasks) == 1 and code_tasks[0].stage == StageKind.CODE
        assert code_tasks[0].pinned_route is not None
        assert any(c["role"] == ScientificRole.THEORIST.value for c in router.calls)
        latest_ep = db.get_latest_experiment_plan(hyp.hypothesis_id)
        assert latest_ep is not None and latest_ep.review_status == "APPROVED"

        # Restart-durable deadline sweep transitions expired quorum to BLOCKED_REVIEW_QUORUM + owner_escalation
        h_exp = _hyp(goal, "h_quorum_exp", tags=["theory"])
        db.register_hypothesis(h_exp)
        wf_exp, _ = rt.create_workflow_for_hypothesis(goal, h_exp, primitive="direct_hq")
        expired = rt.sweep_expired_review_quorums(goal.goal_id, now=1e12)
        assert len(expired) == 1 and expired[0].status == "BLOCKED_REVIEW_QUORUM"
        assert db.get_hypothesis(h_exp.hypothesis_id).status == "BLOCKED_REVIEW_QUORUM"
        esc = db.list_knowledge_items(goal.goal_id, category="owner_escalation")
        assert any(e["key"] == f"quorum:{wf_exp.workflow_id}" for e in esc)


def test_code_pinned_route_policy_blocked_propagation_and_retry_review_role(
    tmp_path: pathlib.Path,
) -> None:
    class PolicyBlockedError(RuntimeError):
        def __init__(self, msg: str, *, waitable: bool = False) -> None:
            super().__init__(msg)
            self.waitable = waitable

    class _PolicyTestRouter(_FakeModelRouter):
        def __init__(self, *, block_builder: bool = False) -> None:
            super().__init__()
            self.block_builder = block_builder

        def resolve_route(
            self,
            role: ScientificRole | str,
            **kwargs: Any,
        ) -> ResolvedModelRoute:
            r_str = role.value if isinstance(role, ScientificRole) else str(role)
            if self.block_builder and r_str in ("engineer_builder", "theorist"):
                raise PolicyBlockedError("HARD Opus xhigh unavailable", waitable=False)
            return super().resolve_route(role, **kwargs)

    with StateDB(tmp_path / "arr") as db:
        goal = _goal()
        db.save_goal(goal)
        hyp = _hyp(goal, "h_pol_1", params={"requires_llm_code": True, "proposer_family": "openai"})
        db.register_hypothesis(hyp)

        # 1. `PolicyBlockedError` on HARD `CODE` build route MUST propagate (never swallowed to None)
        rt_block = WorkflowRuntime(db, router=_PolicyTestRouter(block_builder=True))
        with pytest.raises(PolicyBlockedError, match="HARD Opus xhigh unavailable"):
            rt_block.create_workflow_for_hypothesis(
                goal, hyp, primitive="direct_hq", config_override={"require_design_review": False}
            )

        # 2. Pre-code R1 `DESIGN_REVIEW` excludes only `proposer_family` ("openai"), NOT unrun builder;
        #    and `NEED_RETRY_SLOTS` on `DESIGN_REVIEW` invokes `ScientificRole.DESIGN_REVIEWER`!
        router_ok = _PolicyTestRouter(block_builder=False)
        rt_ok = WorkflowRuntime(db, router=router_ok)
        wf, dr_tasks = rt_ok.create_workflow_for_hypothesis(
            goal,
            hyp,
            primitive="direct_hq",
            config_override={"require_design_review": True},
        )
        assert len(dr_tasks) == 1 and dr_tasks[0].stage == StageKind.DESIGN_REVIEW
        dr_call = next(c for c in router_ok.calls if c["role"] == ScientificRole.DESIGN_REVIEWER.value)
        assert dr_call["avoid_families"] == ["openai"]

        # Trigger `NEED_RETRY_SLOTS` on `DESIGN_REVIEW` (`PARSE_ERROR` on slot 0) -> must request `DESIGN_REVIEWER`!
        router_ok.calls.clear()
        retry_dr = rt_ok.advance_workflow(
            wf,
            dr_tasks[0],
            _rcpt(hyp, rid="dr_nv", tid=dr_tasks[0].task_id, stage=StageKind.DESIGN_REVIEW, averdict=AuditVerdict.PARSE_ERROR),
            goal,
        )
        assert len(retry_dr) == 1 and retry_dr[0].attempt == 2
        assert any(c["role"] == ScientificRole.DESIGN_REVIEWER.value for c in router_ok.calls)
        assert all(c["role"] != ScientificRole.IMPL_REVIEWER.value for c in router_ok.calls)


def test_terminate_refuted_route_blocks_confirmed_and_unknown_receipts(
    tmp_path: pathlib.Path,
) -> None:
    with StateDB(tmp_path / "arr") as db:
        goal = _goal("g_ref_guard")
        db.save_goal(goal)
        h_pos = _hyp(goal, "h_pos_emp")
        h_unk = _hyp(goal, "h_unk_emp")
        h_kill = _hyp(goal, "h_kill_emp")
        for h in (h_pos, h_unk, h_kill):
            db.register_hypothesis(h)

        rt = WorkflowRuntime(db, router=_FakeModelRouter())
        _w_pos, pos_tasks = rt.create_workflow_for_hypothesis(
            goal, h_pos, primitive="direct_hq", config_override={"require_design_review": False}
        )
        _w_unk, unk_tasks = rt.create_workflow_for_hypothesis(
            goal, h_unk, primitive="direct_hq", config_override={"require_design_review": False}
        )

        db.record_receipt(
            _rcpt(
                h_pos,
                rid="r_pos_1",
                tid="t_pos_1",
                stage=StageKind.EXP_SCREEN,
                sverdict=ScientificVerdict.CONFIRMED,
                fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
                metrics={"spectral_gap": 0.66},
            )
        )
        db.record_receipt(
            _rcpt(
                h_unk,
                rid="r_unk_1",
                tid="t_unk_1",
                stage=StageKind.EXP_SCREEN,
                sverdict=ScientificVerdict.UNKNOWN,
                fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
                metrics={"spectral_gap": 0.44},
            )
        )
        db.record_receipt(
            _rcpt(
                h_kill,
                rid="r_kill_1",
                tid="t_kill_1",
                stage=StageKind.EXP_SCREEN,
                sverdict=ScientificVerdict.UNKNOWN,
                fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
                metrics={"spectral_gap": 0.11},
            )
        )

        out = apply_scientific_actions(
            db,
            goal,
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=goal.goal_id,
                actions=[
                    LeadAction(action=ActionKind.TERMINATE_REFUTED_ROUTE, target_hypothesis_id=h_pos.hypothesis_id),
                    LeadAction(action=ActionKind.TERMINATE_REFUTED_ROUTE, target_hypothesis_id=h_unk.hypothesis_id),
                    LeadAction(action=ActionKind.TERMINATE_REFUTED_ROUTE, target_hypothesis_id=h_kill.hypothesis_id),
                ],
            ),
            workflow_runtime=rt,
        )
        assert h_pos.hypothesis_id in out["blocked_unverified_refutations"]
        assert h_unk.hypothesis_id in out["blocked_unverified_refutations"]
        assert out["terminated_hypothesis_ids"] == [h_kill.hypothesis_id]

        gy_ids = {g["hypothesis_id"] for g in db.list_graveyard(goal_id=goal.goal_id)}
        assert gy_ids == {h_kill.hypothesis_id}
        assert db.get_hypothesis(h_pos.hypothesis_id).status != "REFUTED"
        assert db.get_hypothesis(h_unk.hypothesis_id).status != "REFUTED"
        assert db.get_task(pos_tasks[0].task_id).status == "READY"
        assert db.get_task(unk_tasks[0].task_id).status == "READY"


def test_two_goal_cross_ownership_isolation_across_all_mutating_actions(
    tmp_path: pathlib.Path,
) -> None:
    with StateDB(tmp_path / "arr") as db:
        ga = _goal("goal-alpha")
        gb = _goal("goal-beta")
        db.save_goal(ga)
        db.save_goal(gb)
        ha = _hyp(ga, "ha_1")
        hb = _hyp(gb, "hb_1")
        db.register_hypothesis(ha)
        db.register_hypothesis(hb)
        db.record_receipt(
            _rcpt(
                hb,
                rid="rb_ref",
                tid="tb_ref",
                stage=StageKind.EXP_SCREEN,
                sverdict=ScientificVerdict.REFUTED,
                fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
                metrics={"spectral_gap": 0.10},
            )
        )
        rt = WorkflowRuntime(db, router=_FakeModelRouter())
        wfb, tb_tasks = rt.create_workflow_for_hypothesis(
            gb, hb, primitive="direct_hq", config_override={"require_design_review": False}
        )

        # Mixed batch with a valid action on `ha_1` + cross-goal action on `hb_1` rolls back atomically!
        bad_envelopes = [
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=ga.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.REVISE_HYPOTHESIS,
                        target_hypothesis_id=ha.hypothesis_id,
                        parameters={"params_patch": {"should_rollback": True}},
                    ),
                    LeadAction(
                        action=ActionKind.PRUNE_BRANCH,
                        target_hypothesis_id=hb.hypothesis_id,
                    ),
                ],
            ),
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=ga.goal_id,
                actions=[LeadAction(action=ActionKind.REPAIR_IMPLEMENTATION, target_hypothesis_id=hb.hypothesis_id)],
            ),
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=ga.goal_id,
                actions=[LeadAction(action=ActionKind.TERMINATE_REFUTED_ROUTE, target_hypothesis_id=hb.hypothesis_id)],
            ),
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=ga.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.SWITCH_WORKFLOW,
                        target_hypothesis_id=hb.hypothesis_id,
                        workflow_primitive="x-boost",
                    )
                ],
            ),
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=ga.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.PROPOSE_COMPETING_ROUTE,
                        parameters={
                            "proposal": {
                                "title": "Cross parent",
                                "statement": "Cross",
                                "mechanism": "cross_m",
                                "parent_id": hb.hypothesis_id,
                                "writable_paths": ["src/c.py"],
                                "kill_criteria": ["spectral_gap < 0.20"],
                                "acceptance_cmd": "pytest -q",
                                "screen_cmd": "pytest -q",
                            }
                        },
                    )
                ],
            ),
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=gb.goal_id,
                actions=[LeadAction(action=ActionKind.CONTINUE)],
            ),
        ]

        for env in bad_envelopes:
            with pytest.raises(ValueError):
                apply_scientific_actions(db, ga, env, workflow_runtime=rt, raise_on_cross_goal=True)
            res = apply_scientific_actions(db, ga, env, workflow_runtime=rt, raise_on_cross_goal=False)
            assert res["cross_goal_violation"] is True

        # Verify ZERO partial writes to either Goal!
        assert "should_rollback" not in db.get_hypothesis(ha.hypothesis_id).params
        assert db.get_hypothesis(hb.hypothesis_id).status == "PENDING"
        assert db.list_graveyard(goal_id=gb.goal_id) == []
        assert db.get_workflow(wfb.workflow_id).status == "RUNNING"
        assert db.get_task(tb_tasks[0].task_id).status == "READY"
        assert len(db.list_hypotheses(ga.goal_id)) == 1


def test_s3_cross_experiment_synthesis_review_quorum_promotion_and_rejection(
    tmp_path: pathlib.Path,
) -> None:
    with StateDB(tmp_path / "arr") as db:
        goal = _goal("g_s3_syn")
        db.save_goal(goal)
        h_s1 = _hyp(goal, "h_syn_1", params={"requires_llm_code": True, "proposer_family": "anthropic"})
        h_s2 = _hyp(goal, "h_syn_2", params={"requires_llm_code": True, "proposer_family": "openai"})
        db.register_hypothesis(h_s1)
        db.register_hypothesis(h_s2)

        router = _FakeModelRouter()
        rt = WorkflowRuntime(db, router=router)
        out = apply_scientific_actions(
            db,
            goal,
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=goal.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.SYNTHESIZE_RESULTS,
                        uncertainty_id="u_cross_syn",
                        rationale="Combine h_syn_1 and h_syn_2 into unified theorem",
                        parameters={
                            "thesis": "Unified spectral theorem across h_syn_1 and h_syn_2",
                            "synthesized_hypothesis_ids": [h_s1.hypothesis_id, h_s2.hypothesis_id],
                        },
                    )
                ],
            ),
            workflow_runtime=rt,
        )
        assert len(out["synthesis_review_workflows"]) == 1
        assert len(out["emitted_task_ids"]) >= 1
        syn_wf = db.get_workflow(out["synthesis_review_workflows"][0])
        assert syn_wf is not None
        qstate = syn_wf.state["review_quorum"]
        assert qstate["review_kind"] == ReviewKind.SYNTHESIS_REVIEW.value
        assert qstate["target_kind"] == "goal_synthesis"
        assert set(qstate["synthesized_hypothesis_ids"]) == {h_s1.hypothesis_id, h_s2.hypothesis_id}

        # Reviewer routes MUST exclude contributor families from BOTH synthesized hypotheses (`anthropic` & `openai`)!
        syn_tasks = [db.get_task(tid) for tid in out["emitted_task_ids"]]
        for st in syn_tasks:
            assert st is not None and st.pinned_route is not None
            assert st.pinned_route["family"] not in {"anthropic", "openai"}

        # Before review completes, synthesis is held in `synthesis_pending` and ResearchPlan is `DRAFT`
        assert db.list_knowledge_items(goal.goal_id, category="synthesis") == []
        assert len(db.list_knowledge_items(goal.goal_id, category="synthesis_pending")) == 1
        assert db.get_latest_research_plan(goal.goal_id).status == "DRAFT"

        # Passing all synthesis review slots promotes synthesis to `VERIFIED_SYNTHESIS` + `APPROVED` ResearchPlan!
        for idx, st in enumerate(syn_tasks):
            assert st is not None
            rt.advance_workflow(
                syn_wf,
                st,
                _rcpt(h_s1, rid=f"syn_r_{idx}", tid=st.task_id, stage=st.stage, averdict=AuditVerdict.PASS),
                goal,
            )
            syn_wf = db.get_workflow(syn_wf.workflow_id) or syn_wf

        assert syn_wf.status == "COMPLETED"
        syn_items = db.list_knowledge_items(goal.goal_id, category="synthesis")
        assert len(syn_items) == 1
        assert syn_items[0]["content"]["status"] == "VERIFIED_SYNTHESIS"
        assert db.get_latest_research_plan(goal.goal_id).status == "APPROVED"
        syn_revs = db.list_review_receipts(workflow_id=syn_wf.workflow_id, review_kind=ReviewKind.SYNTHESIS_REVIEW)
        assert len(syn_revs) >= 1


def test_r2_nested_experiment_plan_sealing_and_foreign_id_envelope_rejection(
    tmp_path: pathlib.Path,
) -> None:
    from conjectra.types import ExperimentPlan

    with StateDB(tmp_path / "arr") as db:
        ga = _goal("goal-alpha-r2")
        gb = _goal("goal-beta-r2")
        db.save_goal(ga)
        db.save_goal(gb)
        ha = _hyp(ga, "ha_own")
        hb = _hyp(gb, "hb_foreign")
        db.register_hypothesis(ha)
        db.register_hypothesis(hb)
        ep_b = db.save_experiment_plan(
            ExperimentPlan(
                exp_plan_id="eplan_foreign_b",
                goal_id=gb.goal_id,
                hypothesis_id=hb.hypothesis_id,
                falsifiable_claim="Original Goal B claim",
            )
        )

        # Exact R2 counterexample: Lead for goal A targets its own H_A, but nested `experiment_plan`
        # supplies foreign `goal_id`, `hypothesis_id`, or `exp_plan_id` -> whole envelope rejected!
        for bad_nested in (
            {"goal_id": gb.goal_id, "hypothesis_id": hb.hypothesis_id, "falsifiable_claim": "Hijack B"},
            {"exp_plan_id": ep_b.exp_plan_id, "falsifiable_claim": "Overwrite B plan"},
            {"hypothesis_id": hb.hypothesis_id, "falsifiable_claim": "Redirect to H_B"},
        ):
            hijack_env = LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=ga.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.CONTINUE,
                        rationale="Benign sibling action that must also roll back",
                    ),
                    LeadAction(
                        action=ActionKind.REVISE_HYPOTHESIS,
                        target_hypothesis_id=ha.hypothesis_id,
                        parameters={"experiment_plan": bad_nested},
                    ),
                ],
            )
            with pytest.raises(ValueError, match="CROSS_GOAL_OWNERSHIP_VIOLATION"):
                apply_scientific_actions(db, ga, hijack_env, raise_on_cross_goal=True)
            res = apply_scientific_actions(db, ga, hijack_env, raise_on_cross_goal=False)
            assert res["cross_goal_violation"] is True
            assert res["continued"] == 0

        # Verify Goal B's plan and hypothesis linkage are 100% untouched!
        assert db.get_experiment_plan(ep_b.exp_plan_id).falsifiable_claim == "Original Goal B claim"
        assert db.get_hypothesis(hb.hypothesis_id).exp_plan_id == ep_b.exp_plan_id
        assert db.list_knowledge_items(ga.goal_id, category="lead_continue") == []


def test_r2_stale_empirical_not_authoritative_after_revision_and_s3_multi_source_fencing(
    tmp_path: pathlib.Path,
) -> None:
    with StateDB(tmp_path / "arr") as db:
        goal = _goal("g_r2_lineage")
        db.save_goal(goal)
        h_neg = _hyp(goal, "h_neg_ep0")
        h_pos = _hyp(goal, "h_pos_ep0")
        h_s2 = _hyp(goal, "h_syn_dep2", params={"requires_llm_code": True, "proposer_family": "openai"})
        for h in (h_neg, h_pos, h_s2):
            db.register_hypothesis(h)

        # Record valid epoch-0 negative on `h_neg` and valid epoch-0 positive on `h_pos`
        db.record_receipt(
            _rcpt(
                h_neg,
                rid="rcpt_neg_ep0",
                tid="t_neg_0",
                stage=StageKind.EXP_SCREEN,
                sverdict=ScientificVerdict.REFUTED,
                fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
                metrics={"spectral_gap": 0.11},
            )
        )
        db.record_receipt(
            _rcpt(
                h_pos,
                rid="rcpt_pos_ep0",
                tid="t_pos_0",
                stage=StageKind.EXP_CONFIRM,
                sverdict=ScientificVerdict.CONFIRMED,
                fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
                metrics={"spectral_gap": 0.72},
            )
        )

        rt = WorkflowRuntime(db, router=_FakeModelRouter())
        # Spawn S3 synthesis over `[h_pos, h_s2]` anchored on `h_pos`
        syn_out = apply_scientific_actions(
            db,
            goal,
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=goal.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.SYNTHESIZE_RESULTS,
                        uncertainty_id="u_s3_dep",
                        rationale="Synthesize h_pos + h_s2",
                        parameters={
                            "thesis": "Initial synthesis",
                            "synthesized_hypothesis_ids": [h_pos.hypothesis_id, h_s2.hypothesis_id],
                        },
                    )
                ],
            ),
            workflow_runtime=rt,
        )
        syn_wf_id = syn_out["synthesis_review_workflows"][0]
        syn_task_id = syn_out["emitted_task_ids"][0]
        assert db.get_workflow(syn_wf_id).status == "RUNNING"

        # Autonomous `REVISE_HYPOTHESIS` on `h_neg`, `h_pos`, and NON-ANCHOR `h_s2`:
        # bumps `plan_epoch`, invalidates active S3 synthesis on `[h_pos, h_s2]`, and makes epoch-0 empirical receipts non-authoritative!
        apply_scientific_actions(
            db,
            goal,
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=goal.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.REVISE_HYPOTHESIS,
                        target_hypothesis_id=h_s2.hypothesis_id,
                        parameters={"experiment_plan": {"falsifiable_claim": "Revised non-anchor h_s2 claim"}},
                    ),
                    LeadAction(
                        action=ActionKind.REVISE_HYPOTHESIS,
                        target_hypothesis_id=h_neg.hypothesis_id,
                        parameters={"experiment_plan": {"falsifiable_claim": "Revised h_neg control"}},
                    ),
                    LeadAction(
                        action=ActionKind.REVISE_HYPOTHESIS,
                        target_hypothesis_id=h_pos.hypothesis_id,
                        parameters={"experiment_plan": {"falsifiable_claim": "Revised h_pos control"}},
                    ),
                ],
            ),
            workflow_runtime=rt,
        )
        assert db.get_workflow(syn_wf_id).status == "CANCELLED"
        assert db.get_task(syn_task_id).status == "CANCELLED"

        # Now `TERMINATE_REFUTED_ROUTE` on `h_neg` and `COMPLETE_MILESTONE` on `h_pos` MUST reject stale epoch-0 receipts!
        stale_attempt = apply_scientific_actions(
            db,
            db.get_goal(goal.goal_id),
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=goal.goal_id,
                actions=[
                    LeadAction(action=ActionKind.TERMINATE_REFUTED_ROUTE, target_hypothesis_id=h_neg.hypothesis_id),
                    LeadAction(
                        action=ActionKind.COMPLETE_MILESTONE,
                        target_hypothesis_id=h_pos.hypothesis_id,
                        parameters={"milestone_id": "ms_stale_check"},
                    ),
                ],
            ),
            workflow_runtime=rt,
        )
        assert h_neg.hypothesis_id in stale_attempt["blocked_unverified_refutations"]
        assert stale_attempt["terminated_hypothesis_ids"] == []
        assert stale_attempt["completed_milestones"] == []
        assert db.list_graveyard(goal_id=goal.goal_id) == []
        assert db.get_hypothesis(h_pos.hypothesis_id).status != "COMPLETED"

        # Descendant hypothesis lineage stamping + S3 `"GRAVEYARD"` source fence (`#179`):
        cur_goal = db.get_goal(goal.goal_id)
        assert cur_goal is not None
        h_child = _hyp(cur_goal, "h_child_neg", params={"lineage_plan_epoch": cur_goal.plan_epoch})
        h_child.parent_id = h_neg.hypothesis_id
        db.register_hypothesis(h_child)
        r_child_neg = _rcpt(
            h_child,
            rid="r_child_pre_rev",
            tid="t_child_pre",
            stage=StageKind.EXP_SCREEN,
            sverdict=ScientificVerdict.REFUTED,
            averdict=AuditVerdict.PASS,
            fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
            metrics={"spectral_gap": 0.11},
        )
        r_child_neg.plan_epoch = cur_goal.plan_epoch
        r_child_neg.evidence_validity = EvidenceValidity.VALID_EMPIRICAL
        db.record_receipt(r_child_neg)
        assert len(db.get_authoritative_empirical_receipts(cur_goal, db.get_hypothesis("h_child_neg"))) == 1

        apply_scientific_actions(
            db,
            cur_goal,
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=cur_goal.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.REVISE_HYPOTHESIS,
                        target_hypothesis_id=h_neg.hypothesis_id,
                        parameters={"experiment_plan": {"falsifiable_claim": "Parent revision cascades to child"}},
                    )
                ],
            ),
            workflow_runtime=rt,
        )
        post_goal = db.get_goal(cur_goal.goal_id)
        post_child = db.get_hypothesis("h_child_neg")
        assert post_goal is not None and post_child is not None
        assert post_child.params["lineage_plan_epoch"] == post_goal.plan_epoch
        assert db.get_authoritative_empirical_receipts(post_goal, post_child) == []

        # S3 promotion blocks (`SYNTHESIS_SUPERSEDED`) if a synthesized source transitions to `"GRAVEYARD"`!
        syn_grav_wf, syn_grav_tasks = rt.spawn_synthesis_review(
            post_goal,
            db.get_hypothesis(h_pos.hypothesis_id),
            synthesized_hypothesis_ids=[h_pos.hypothesis_id, h_s2.hypothesis_id],
            synthesis_key="syn:grav_guard",
            thesis="Synthesis must reject graveyarded source",
        )
        db.transition_hypothesis_status(
            h_s2.hypothesis_id,
            db.get_hypothesis(h_s2.hypothesis_id).status,
            "GRAVEYARD",
        )
        for idx, st in enumerate(syn_grav_tasks):
            rt.advance_workflow(
                db.get_workflow(syn_grav_wf.workflow_id),
                st,
                _rcpt(h_pos, rid=f"syn_grav_r_{idx}", tid=st.task_id, stage=st.stage, averdict=AuditVerdict.PASS),
                post_goal,
            )
        assert db.get_workflow(syn_grav_wf.workflow_id).status == "SYNTHESIS_SUPERSEDED"


def test_r3_unaffected_sibling_opposing_outcomes_legacy_s3_fail_closed_and_decision_submit_cas(
    tmp_path: pathlib.Path,
) -> None:
    from conjectra.decision_engine import DecisionEngine
    from conjectra.gates import validate_receipt_epoch
    from conjectra.planning import PlanLifecycleManager
    from conjectra.state_db import FencedStaleWriteError
    from conjectra.types import StageTask, WorkflowNode

    with StateDB(tmp_path / "arr_r3_s3") as db:
        goal = _goal("g_r3_sibling")
        goal.max_tokens_budget = 32_000
        db.save_goal(goal)
        h_a = _hyp(goal, "h_rev_a")
        h_b = _hyp(goal, "h_unaffected_b")
        db.register_hypothesis(h_a)
        db.register_hypothesis(h_b)

        pmgr = PlanLifecycleManager(db)
        pmgr.create_or_revise_experiment_plan(goal, h_a)
        pmgr.create_or_revise_experiment_plan(goal, h_b)

        # `h_b` has an earlier rev-0 negative receipt AND an active `EXP_CONFIRM` task
        r_b_neg0 = _rcpt(
            h_b,
            rid="rcpt_b_neg_rev0",
            tid="t_b_screen0",
            stage=StageKind.EXP_SCREEN,
            sverdict=ScientificVerdict.REFUTED,
            fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
            metrics={"spectral_gap": 0.12},
        )
        db.record_receipt(r_b_neg0)
        t_b_act = db.enqueue_task(
            StageTask(
                task_id="tsk_b_confirm_act",
                idempotency_key="ik_b_confirm_act",
                goal_id=goal.goal_id,
                hypothesis_id=h_b.hypothesis_id,
                stage=StageKind.EXP_CONFIRM,
                status="RUNNING",
                plan_epoch=goal.plan_epoch,
                human_epoch=goal.human_epoch,
                deps=[],
            )
        )

        rt = WorkflowRuntime(db, router=_FakeModelRouter())
        # 1. `P1-03-R3` exact counterexample: revise `h_a` -> `h_b` is unaffected (`sci_revision == 0`, `exp_plan.plan_epoch == 0`),
        # while `t_b_act`'s execution-fence epoch is synced to `goal.plan_epoch == 1`.
        apply_scientific_actions(
            db,
            goal,
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=goal.goal_id,
                actions=[
                    LeadAction(
                        action=ActionKind.REVISE_HYPOTHESIS,
                        target_hypothesis_id=h_a.hypothesis_id,
                        parameters={"experiment_plan": {"falsifiable_claim": "Revised A claim only"}},
                    )
                ],
            ),
            workflow_runtime=rt,
        )
        g_after = db.get_goal(goal.goal_id)
        t_b_synced = db.get_task(t_b_act.task_id)
        assert g_after is not None and g_after.plan_epoch == 1
        assert t_b_synced is not None and t_b_synced.plan_epoch == 1
        assert t_b_synced.pinned_route["exec_fence_plan_epoch"] == 1

        # `t_b_act` finishes with a new POSITIVE receipt stamped with `h_b`'s branch revision (`plan_epoch=0, sci_revision=0`)
        # and synced execution-fence metadata (`exec_fence_plan_epoch=1`).
        r_b_pos = _rcpt(
            h_b,
            rid="rcpt_b_pos_rev0_new",
            tid=t_b_act.task_id,
            stage=StageKind.EXP_CONFIRM,
            sverdict=ScientificVerdict.CONFIRMED,
            fdomain=FailureDomain.SCIENTIFIC_OUTCOME,
            metrics={
                "spectral_gap": 0.91,
                "sci_revision": 0,
                "exec_fence_plan_epoch": g_after.plan_epoch,
                "exec_fence_human_epoch": g_after.human_epoch,
            },
        )
        r_b_pos.plan_epoch = 0
        r_b_pos.human_epoch = goal.human_epoch
        db.record_receipt(r_b_pos)

        assert validate_receipt_epoch(r_b_pos, g_after, task=t_b_synced) == EvidenceValidity.VALID_EMPIRICAL
        auth_b = db.get_authoritative_empirical_receipts(g_after, db.get_hypothesis(h_b.hypothesis_id))
        assert [r.receipt_id for r in auth_b][-1] == "rcpt_b_pos_rev0_new"

        # `TERMINATE_REFUTED_ROUTE` on `h_b` MUST refuse refutation (latest authoritative is `CONFIRMED`!),
        # and `COMPLETE_MILESTONE` on `h_b` MUST succeed!
        out_b = apply_scientific_actions(
            db,
            g_after,
            LeadDecisionEnvelope(
                role=ScientificRole.RESEARCH_LEAD,
                goal_id=g_after.goal_id,
                actions=[
                    LeadAction(action=ActionKind.TERMINATE_REFUTED_ROUTE, target_hypothesis_id=h_b.hypothesis_id),
                    LeadAction(
                        action=ActionKind.COMPLETE_MILESTONE,
                        target_hypothesis_id=h_b.hypothesis_id,
                        parameters={"milestone_id": "ms_b_positive"},
                    ),
                ],
            ),
            workflow_runtime=rt,
        )
        assert h_b.hypothesis_id in out_b["blocked_unverified_refutations"]
        assert out_b["terminated_hypothesis_ids"] == []
        assert db.get_hypothesis(h_b.hypothesis_id).status == "COMPLETED"

        # 2. `P1-03-R3` runtime fail-closed legacy S3 without `source_lineage_snapshot`:
        # never publishes `VERIFIED_SYNTHESIS` with `{}`; marks `SYNTHESIS_SUPERSEDED` and respawns a snapshot-bound S3 workflow!
        legacy_s3 = WorkflowNode(
            workflow_id="wf_s3_missing_snap_rt",
            goal_id=g_after.goal_id,
            hypothesis_id=h_b.hypothesis_id,
            primitive="direct_hq",
            status="RUNNING",
            plan_epoch=g_after.plan_epoch,
            config={
                "target_kind": "goal_synthesis",
                "synthesized_hypothesis_ids": [h_b.hypothesis_id],
                "synthesis_key": "syn:missing_snap_rt",
                "thesis": "Legacy S3 without snapshot",
            },
            state={
                "step": "synthesis_review",
                "review_quorum": {
                    "review_kind": ReviewKind.SYNTHESIS_REVIEW.value,
                    "review_tier": "STANDARD_CROSS_FAMILY",
                    "target_kind": "goal_synthesis",
                    "target_version": 1,
                    "plan_epoch": g_after.plan_epoch,
                    "human_epoch": g_after.human_epoch,
                    "head_sha": h_b.base_sha,
                    "synthesized_hypothesis_ids": [h_b.hypothesis_id],
                    "contributor_families": ["anthropic"],
                    "required_reviewers": 1,
                    "slots": {"0": {"slot_index": 0, "status": "DISPATCHED"}},
                    "invalidated_slots": [],
                },
            },
        )
        db.save_workflow(legacy_s3)
        t_leg_rev = db.enqueue_task(
            StageTask(
                task_id="tsk_s3_missing_snap_s0",
                idempotency_key="ik_s3_missing_snap_s0",
                goal_id=g_after.goal_id,
                hypothesis_id=h_b.hypothesis_id,
                workflow_id=legacy_s3.workflow_id,
                stage=StageKind.RESULT_REVIEW,
                step_key="synthesis_review_s0",
                plan_epoch=g_after.plan_epoch,
                deps=[],
            )
        )
        respawned = rt.advance_workflow(
            legacy_s3,
            t_leg_rev,
            _rcpt(h_b, rid="r_leg_s3_pass", tid=t_leg_rev.task_id, stage=StageKind.RESULT_REVIEW, averdict=AuditVerdict.PASS),
            g_after,
        )
        assert db.get_workflow(legacy_s3.workflow_id).status == "SYNTHESIS_SUPERSEDED"
        assert db.list_knowledge_items(g_after.goal_id, category="synthesis") == []
        assert len(respawned) >= 1
        repl_wf_id = db.get_workflow(legacy_s3.workflow_id).state.get("replacement_workflow_id")
        assert repl_wf_id and db.get_workflow(repl_wf_id).config.get("source_lineage_snapshot")

        # 3. `P1-07-R3` Decision submission exclusive lease, concurrent pause cancel, and fast-completion preservation:
        class _MiniHQ:
            def __init__(self) -> None:
                self.submitted: list[int] = []
                self.cancelled: list[int] = []
                self._next = 900

            def ensure_running(self) -> None:
                return None

            def submit_task(self, *_a: object, **_kw: object) -> int:
                self._next += 1
                self.submitted.append(self._next)
                return self._next

            def cancel_jobs(self, jids: list[int], reason: str = "") -> dict[int, bool]:
                _ = reason
                self.cancelled.extend(int(x) for x in jids)
                return {int(x): True for x in jids}

        mini_hq = _MiniHQ()
        eng = DecisionEngine(db, hq=mini_hq, router=_FakeModelRouter(), workflow_runtime=rt)  # type: ignore[arg-type]

        # (a) Conflicting active lease -> `acquire_lease` returns None -> `submit_decision_to_hq` aborts with 0 reservations & 0 HQ submits!
        req_conf = db.enqueue_decision_request(
            g_after.goal_id,
            role="research_lead",
            tier="L1",
            trigger_kind="R3_CONFLICT",
            goal_version=g_after.version,
            human_epoch=g_after.human_epoch,
            coalesce_key="k_conf",
        )
        held_l = db.acquire_lease(f"decision:{g_after.goal_id}:{req_conf['req_id']}", f"dec-{req_conf['req_id']}", "other-holder")
        assert held_l is not None
        with pytest.raises(FencedStaleWriteError, match="lease"):
            eng.submit_decision_to_hq(req_conf["req_id"])
        assert mini_hq.submitted == []
        assert db.get_goal_token_usage(g_after.goal_id)["reserved_tokens"] == 0
        db.release_lease(held_l["lease_id"])

        # (b) Concurrent human `pause` / `CANCELLED` during `submit_decision_to_hq` -> preserves `CANCELLED`,
        # cancels HQ job, and RETAINS `RESERVED` tokens until authoritative HQ termination confirmation!
        req_pause = db.enqueue_decision_request(
            g_after.goal_id,
            role="research_lead",
            tier="L1",
            trigger_kind="R3_PAUSE",
            goal_version=g_after.version,
            human_epoch=g_after.human_epoch,
            coalesce_key="k_pause",
        )

        def _pause_mid_submit(_jid: int) -> None:
            db.resolve_decision_request(req_pause["req_id"], status="CANCELLED", resolution={"reason": "human_pause_mid_submit"})

        jid_paused = eng.submit_decision_to_hq(req_pause["req_id"], post_submit_hook=_pause_mid_submit)
        assert db.get_decision_request(req_pause["req_id"])["status"] == "CANCELLED"
        assert jid_paused in mini_hq.cancelled
        assert db.get_goal_token_usage(g_after.goal_id)["reserved_tokens"] > 0
        db.confirm_hq_cancellation(jid_paused, entity_kind="decision", entity_id=req_pause["req_id"], attempt=1)
        assert db.get_goal_token_usage(g_after.goal_id)["reserved_tokens"] == 0

        # (c) Fast worker finishes (`RESOLVED`) before `submit_decision_to_hq` post-submit step -> preserves `RESOLVED` without cancelling HQ job!
        req_fast = db.enqueue_decision_request(
            g_after.goal_id,
            role="research_lead",
            tier="L1",
            trigger_kind="R3_FAST",
            goal_version=g_after.version,
            human_epoch=g_after.human_epoch,
            coalesce_key="k_fast",
        )
        prev_cancelled = list(mini_hq.cancelled)

        def _fast_complete_mid_submit(_jid: int) -> None:
            cur_d = db.get_decision_request(req_fast["req_id"])
            assert cur_d is not None and cur_d["status"] == "SUBMITTING"
            lid_fast = cur_d["payload"]["lease_id"]
            db.fenced_resolve_decision_request(
                req_fast["req_id"],
                expected_goal_version=g_after.version,
                expected_human_epoch=g_after.human_epoch,
                expected_attempt=1,
                expected_lease_id=lid_fast,
                expected_statuses=("SUBMITTING", "RUNNING"),
                status="RESOLVED",
                resolution={"in_tokens": 500, "out_tokens": 100, "accepted_hypothesis_ids": []},
            )

        jid_fast = eng.submit_decision_to_hq(req_fast["req_id"], post_submit_hook=_fast_complete_mid_submit)
        after_fast = db.get_decision_request(req_fast["req_id"])
        assert after_fast is not None and after_fast["status"] == "RESOLVED"
        assert after_fast["hq_job_id"] == jid_fast
        assert mini_hq.cancelled == prev_cancelled


def test_stage4_reversible_f1_skips_llm_r1_and_advisory_confound_checks(
    tmp_path: pathlib.Path,
) -> None:
    """Priority 4: reversible F1 uses 0-LLM R1 contract check and treats theoretical critiques as advisory."""
    with StateDB(tmp_path / "arr_r1") as db:
        goal = _goal("goal-projgamma-test", risk_profile="standard")
        db.save_goal(goal)
        rt = WorkflowRuntime(db, router=_FakeModelRouter())

        # 1. Reversible F1 theory hypothesis on standard goal with 跨家族审查 skips LLM DESIGN_REVIEW
        h_f1 = _hyp(
            goal,
            "h_rev_f1",
            tags=["theory"],
            params={"requires_llm_code": True, "proposer_family": "anthropic"},
        )
        db.register_hypothesis(h_f1)
        wf_f1, tasks_f1 = rt.create_workflow_for_hypothesis(goal, h_f1, primitive="direct_hq")
        assert len(tasks_f1) == 1 and tasks_f1[0].stage == StageKind.CODE
        assert wf_f1.config.get("design_review_mode") == "FAST_LANE_0_LLM"
        ep_f1 = db.get_latest_experiment_plan(h_f1.hypothesis_id)
        if ep_f1 is not None:
            assert ep_f1.review_status == "APPROVED"

        # 2. Explicit LLM DESIGN_REVIEW FAIL with purely theoretical critique -> advisory confound_checks, advances to CODE!
        h_adv = _hyp(
            goal,
            "h_r1_advisory",
            tags=[],
            params={"requires_llm_code": True, "require_design_review": True},
        )
        db.register_hypothesis(h_adv)
        wf_adv, dr_adv = rt.create_workflow_for_hypothesis(goal, h_adv, primitive="direct_hq")
        assert len(dr_adv) == 1 and dr_adv[0].stage == StageKind.DESIGN_REVIEW
        code_adv = rt.advance_workflow(
            wf_adv,
            dr_adv[0],
            _rcpt(
                h_adv,
                rid="r1_theory_crit",
                tid=dr_adv[0].task_id,
                stage=StageKind.DESIGN_REVIEW,
                averdict=AuditVerdict.FAIL,
                reason="Theoretical assumption of Lipschitz smoothness needs empirical confound check under F1",
            ),
            goal,
        )
        assert len(code_adv) == 1 and code_adv[0].stage == StageKind.CODE
        ep_adv = db.get_latest_experiment_plan(h_adv.hypothesis_id)
        assert ep_adv is not None and ep_adv.review_status == "APPROVED"
        assert any("Lipschitz" in c for c in ep_adv.confound_checks)
        assert db.get_hypothesis(h_adv.hypothesis_id).status != "NEEDS_PLAN_REVISION"  # type: ignore[union-attr]

        # Strict R3 RESULT_REVIEW on EXP_CONFIRM still blocks (`RESULT_REJECTED`)
        r_code = _rcpt(h_adv, rid="adv_code", tid=code_adv[0].task_id, stage=StageKind.CODE)
        t_gate = rt.advance_workflow(wf_adv, code_adv[0], r_code, goal)[0]
        r_gate = _rcpt(h_adv, rid="adv_gate", tid=t_gate.task_id, stage=StageKind.CPU_GATE)
        t_next = rt.advance_workflow(wf_adv, t_gate, r_gate, goal)[0]
        if t_next.stage == StageKind.IMPL_REVIEW:
            r_r2 = _rcpt(
                h_adv,
                rid="adv_r2",
                tid=t_next.task_id,
                stage=StageKind.IMPL_REVIEW,
                averdict=AuditVerdict.PASS,
            )
            t_scr = rt.advance_workflow(wf_adv, t_next, r_r2, goal)[0]
        else:
            t_scr = t_next
        assert t_scr.stage == StageKind.EXP_SCREEN
        r_scr = _rcpt(
            h_adv,
            rid="adv_scr",
            tid=t_scr.task_id,
            stage=StageKind.EXP_SCREEN,
            sverdict=ScientificVerdict.CONFIRMED,
        )
        t_cnf = rt.advance_workflow(wf_adv, t_scr, r_scr, goal)[0]
        assert t_cnf.stage == StageKind.EXP_CONFIRM
        r_cnf = _rcpt(
            h_adv,
            rid="adv_cnf",
            tid=t_cnf.task_id,
            stage=StageKind.EXP_CONFIRM,
            sverdict=ScientificVerdict.CONFIRMED,
        )
        t_r3 = rt.advance_workflow(wf_adv, t_cnf, r_cnf, goal)[0]
        assert t_r3.stage == StageKind.RESULT_REVIEW
        r3_out = rt.advance_workflow(
            wf_adv,
            t_r3,
            _rcpt(
                h_adv,
                rid="adv_r3_fail",
                tid=t_r3.task_id,
                stage=StageKind.RESULT_REVIEW,
                averdict=AuditVerdict.FAIL,
                reason="Result unsupported by measured baseline",
            ),
            goal,
        )
        assert r3_out == []
        assert db.get_workflow(wf_adv.workflow_id).status == "RESULT_REJECTED"  # type: ignore[union-attr]

        # 3. Hard violation (touches holdout) STILL blocks as DESIGN_REJECTED -> NEEDS_PLAN_REVISION
        h_hard = _hyp(
            goal,
            "h_r1_hard",
            params={"requires_llm_code": True, "require_design_review": True},
        )
        db.register_hypothesis(h_hard)
        wf_hard, dr_hard = rt.create_workflow_for_hypothesis(goal, h_hard, primitive="direct_hq")
        out_hard = rt.advance_workflow(
            wf_hard,
            dr_hard[0],
            _rcpt(
                h_hard,
                rid="r1_hard_fail",
                tid=dr_hard[0].task_id,
                stage=StageKind.DESIGN_REVIEW,
                averdict=AuditVerdict.FAIL,
                reason="Plan accesses holdout split directly causing holdout leakage",
            ),
            goal,
        )
        assert out_hard == []
        assert db.get_workflow(wf_hard.workflow_id).status == "DESIGN_REJECTED"  # type: ignore[union-attr]
        assert db.get_hypothesis(h_hard.hypothesis_id).status == "NEEDS_PLAN_REVISION"  # type: ignore[union-attr]

        # 4. Crashed / PARSE_ERROR DESIGN_REVIEW on F1 is NEVER auto-approved as advisory
        h_parse = _hyp(
            goal,
            "h_r1_parse_err",
            tags=[],
            params={"requires_llm_code": True, "require_design_review": True},
        )
        db.register_hypothesis(h_parse)
        wf_parse, dr_parse = rt.create_workflow_for_hypothesis(goal, h_parse, primitive="direct_hq")
        # Exhaust retries / crashed process_status=ENV_ERROR -> must block!
        out_p1 = rt.advance_workflow(
            wf_parse,
            dr_parse[0],
            _rcpt(
                h_parse,
                rid="r1_pe_1",
                tid=dr_parse[0].task_id,
                stage=StageKind.DESIGN_REVIEW,
                pstatus=ProcessStatus.ENV_ERROR,
                averdict=AuditVerdict.PARSE_ERROR,
                reason="Worker crashed before structured JSON verdict",
            ),
            goal,
        )
        if out_p1:
            out_p1 = rt.advance_workflow(
                wf_parse,
                out_p1[0],
                _rcpt(
                    h_parse,
                    rid="r1_pe_2",
                    tid=out_p1[0].task_id,
                    stage=StageKind.DESIGN_REVIEW,
                    pstatus=ProcessStatus.ENV_ERROR,
                    averdict=AuditVerdict.PARSE_ERROR,
                    reason="Worker crashed on retry too",
                ),
                goal,
            )
        assert out_p1 == []
        assert db.get_workflow(wf_parse.workflow_id).status in ("DESIGN_REJECTED", "BLOCKED_REVIEW_QUORUM")  # type: ignore[union-attr]

        # Crashed FAIL (process_status=ENV_ERROR, averdict=AuditVerdict.FAIL) is also rejected, never advisory!
        h_crash = _hyp(
            goal,
            "h_r1_crash_fail",
            tags=[],
            params={"requires_llm_code": True, "require_design_review": True},
        )
        db.register_hypothesis(h_crash)
        wf_crash, dr_crash = rt.create_workflow_for_hypothesis(goal, h_crash, primitive="direct_hq")
        out_crash = rt.advance_workflow(
            wf_crash,
            dr_crash[0],
            _rcpt(
                h_crash,
                rid="r1_cr_1",
                tid=dr_crash[0].task_id,
                stage=StageKind.DESIGN_REVIEW,
                pstatus=ProcessStatus.ENV_ERROR,
                averdict=AuditVerdict.FAIL,
                reason="Process exited with infra failure",
            ),
            goal,
        )
        if out_crash:
            out_crash = rt.advance_workflow(
                wf_crash,
                out_crash[0],
                _rcpt(
                    h_crash,
                    rid="r1_cr_2",
                    tid=out_crash[0].task_id,
                    stage=StageKind.DESIGN_REVIEW,
                    pstatus=ProcessStatus.ENV_ERROR,
                    averdict=AuditVerdict.FAIL,
                    reason="Process exited with infra failure on retry",
                ),
                goal,
            )
        assert out_crash == []
        assert db.get_workflow(wf_crash.workflow_id).status in ("DESIGN_REJECTED", "BLOCKED_REVIEW_QUORUM")  # type: ignore[union-attr]

        # 5. F3_MILESTONE_AUDIT & high-risk goals stay strictly blocking even on non-keyword critiques
        h_f3 = HypothesisNode.from_dict(
            {
                **_hyp(goal, "h_r1_f3", tags=[], params={"requires_llm_code": True, "require_design_review": True}).to_dict(),
                "fidelity": FidelityRung.F3_MILESTONE_AUDIT.value,
            }
        )
        db.register_hypothesis(h_f3)
        wf_f3, dr_f3 = rt.create_workflow_for_hypothesis(goal, h_f3, primitive="direct_hq")
        for idx_dr, t_dr in enumerate(dr_f3):
            out_f3 = rt.advance_workflow(
                wf_f3,
                t_dr,
                _rcpt(
                    h_f3,
                    rid=f"r1_f3_{idx_dr}",
                    tid=t_dr.task_id,
                    stage=StageKind.DESIGN_REVIEW,
                    averdict=AuditVerdict.FAIL,
                    reason="Theoretical assumption requires tighter proof",
                ),
                goal,
            )
        assert out_f3 == []
        assert db.get_workflow(wf_f3.workflow_id).status == "DESIGN_REJECTED"  # type: ignore[union-attr]

        g_high = _goal("goal-high-risk", risk_profile="high")
        db.save_goal(g_high)
        h_high = _hyp(g_high, "h_r1_high", tags=[], params={"requires_llm_code": True, "require_design_review": True})
        db.register_hypothesis(h_high)
        wf_hi, dr_hi = rt.create_workflow_for_hypothesis(g_high, h_high, primitive="direct_hq")
        for idx_dr, t_dr in enumerate(dr_hi):
            out_hi = rt.advance_workflow(
                wf_hi,
                t_dr,
                _rcpt(
                    h_high,
                    rid=f"r1_hi_{idx_dr}",
                    tid=t_dr.task_id,
                    stage=StageKind.DESIGN_REVIEW,
                    averdict=AuditVerdict.FAIL,
                    reason="Methodological critique without hard keyword",
                ),
                g_high,
            )
        assert out_hi == []
        assert db.get_workflow(wf_hi.workflow_id).status == "DESIGN_REJECTED"  # type: ignore[union-attr]


def test_stage5_p1_02_decision_worker_revalidates_fresh_policy_before_llm(
    tmp_path: pathlib.Path,
) -> None:
    """P1-02-R5: policy change between submit and worker start aborts LLM call and cleanly requeues or blocks."""
    from conjectra.model_router import ModelRouter

    with StateDB(tmp_path / "arr_p102") as db:
        goal = _goal("g_p102", risk_profile="standard")
        goal.routing_policy_nl = ""
        goal.routing_policy = {}
        goal.max_tokens_budget = 50_000
        db.save_goal(goal)

        # Allow OpenAI at submit time
        db.set_global_config(
            "routing_policy",
            {
                "rules": [
                    {
                        "rule_id": "allow_sol",
                        "roles": ["research_lead"],
                        "strength": "HARD",
                        "allow_effort_alias": False,
                        "candidates": [{"provider_key": "gpt-5-codex", "effort": "high"}],
                    }
                ]
            },
        )
        router = ModelRouter(db=db)
        rt = WorkflowRuntime(db, router=router)
        engine = DecisionEngine(db, router=router, workflow_runtime=rt)

        class _FakeSubmitHQ:
            def __init__(self) -> None:
                self._next_id = 7700

            def ensure_running(self) -> None:
                return None

            def submit_task(self, *_a: Any, **_kw: Any) -> int:
                self._next_id += 1
                return self._next_id

            def cancel_job(self, *_a: Any, **_kw: Any) -> None:
                return None

        fake_hq = _FakeSubmitHQ()
        engine_hq = DecisionEngine(db, router=router, workflow_runtime=rt, hq=fake_hq)
        req = engine_hq.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            trigger_kind="EVIDENCE_UPDATE",
            urgency="IMMEDIATE",
        )
        job_1 = engine_hq.submit_decision_to_hq(req["req_id"])
        assert int(job_1) == 7701
        submitted_1 = db.get_decision_request(req["req_id"])
        assert submitted_1 is not None and submitted_1["status"] == "RUNNING"
        assert submitted_1["payload"]["pinned_route"]["canonical_model"] == "gpt-5-codex"
        assert db.get_goal_token_usage(goal.goal_id)["reserved_tokens"] > 0

        # HITL changes global policy to force Anthropic Opus BEFORE worker runs -> requeues PENDING with 0 LLM calls!
        db.set_global_config(
            "routing_policy",
            {
                "rules": [
                    {
                        "rule_id": "force_opus",
                        "roles": ["research_lead"],
                        "strength": "HARD",
                        "allow_effort_alias": False,
                        "candidates": [{"provider_key": "claude-opus-5-5", "effort": "high"}],
                    }
                ]
            },
        )
        llm_calls = 0

        def _should_not_run(_prov: Any, _prompt: str) -> dict[str, Any]:
            nonlocal llm_calls
            llm_calls += 1
            return {"exit_code": 0, "response": "{}", "in_tokens": 100, "out_tokens": 10}

        out_requeue = engine_hq.process_request(req["req_id"], llm_caller=_should_not_run, dry_run_llm=False)
        assert llm_calls == 0
        assert out_requeue["status"] == "PENDING"
        assert out_requeue["attempt"] == 2
        assert "pinned_route" not in out_requeue["payload"]
        assert db.get_goal_token_usage(goal.goal_id)["reserved_tokens"] == 0

        # Now submit second request pinned to gpt-5-codex, then set an un-satisfiable hard model ban -> BLOCKED_POLICY cleanly!
        req_blk = engine_hq.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            trigger_kind="EVIDENCE_UPDATE",
            urgency="IMMEDIATE",
        )
        # Switch policy back to allow gpt-5-codex at submit time, submit, then switch to unsatisfiable rule
        db.set_global_config(
            "routing_policy",
            {
                "rules": [
                    {
                        "rule_id": "allow_sol_2",
                        "roles": ["research_lead"],
                        "strength": "HARD",
                        "allow_effort_alias": False,
                        "candidates": [{"provider_key": "gpt-5-codex", "effort": "high"}],
                    }
                ]
            },
        )
        job_2 = engine_hq.submit_decision_to_hq(req_blk["req_id"])
        assert int(job_2) == 7702
        submitted_2 = db.get_decision_request(req_blk["req_id"])
        assert submitted_2 is not None and submitted_2["status"] == "RUNNING"
        assert db.get_goal_token_usage(goal.goal_id)["reserved_tokens"] > 0
        db.set_global_config(
            "routing_policy",
            {
                "rules": [
                    {
                        "rule_id": "sonnet_xhigh_unsupported_hard",
                        "roles": ["research_lead"],
                        "strength": "HARD",
                        "allow_effort_alias": False,
                        "candidates": [{"provider_key": "claude-sonnet-5-5", "effort": "xhigh"}],
                    }
                ]
            },
        )
        out_blocked = engine_hq.process_request(req_blk["req_id"], llm_caller=_should_not_run, dry_run_llm=False)
        assert llm_calls == 0
        assert out_blocked["status"] == "BLOCKED_POLICY"
        assert db.get_goal_token_usage(goal.goal_id)["reserved_tokens"] == 0

        # R6 #2: hypothesis/workflow-scoped policy change triggers revalidation + fenced lease/goal checks!
        from conjectra.state_db import FencedStaleWriteError
        from conjectra.types import WorkflowNode

        db.set_global_config(
            "routing_policy",
            {
                "rules": [
                    {
                        "rule_id": "global_sol",
                        "roles": ["research_lead"],
                        "strength": "HARD",
                        "allow_effort_alias": False,
                        "candidates": [{"provider_key": "gpt-5-codex", "effort": "high"}],
                    }
                ]
            },
        )
        h_scoped = _hyp(goal, "h_p102_scoped", tags=["theory"])
        db.register_hypothesis(h_scoped)
        wf_sc = db.save_workflow(
            WorkflowNode(
                workflow_id="wf_p102_sc",
                goal_id=goal.goal_id,
                hypothesis_id=h_scoped.hypothesis_id,
                primitive="direct_hq",
            )
        )
        req_sc = engine_hq.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            hypothesis_id=h_scoped.hypothesis_id,
            trigger_kind="EVIDENCE_UPDATE",
            urgency="IMMEDIATE",
            payload={"workflow_id": wf_sc.workflow_id},
        )
        engine_hq.submit_decision_to_hq(req_sc["req_id"])
        assert db.get_decision_request(req_sc["req_id"])["payload"]["pinned_route"]["canonical_model"] == "gpt-5-codex"

        # Add a workflow/tag-scoped rule requiring Opus xhigh for `theory` (leaving global gpt-5-codex rule intact!)
        db.update_hypothesis_params(
            h_scoped.hypothesis_id,
            {
                "workflow_routing_policies": {
                    wf_sc.workflow_id: {
                        "rules": [
                            {
                                "rule_id": "wf_theory_opus",
                                "tags": ["theory"],
                                "roles": ["research_lead"],
                                "strength": "HARD",
                                "allow_effort_alias": False,
                                "candidates": [{"provider_key": "claude-opus-5-5", "effort": "xhigh"}],
                            }
                        ]
                    }
                }
            },
        )
        # First verify mid-flight lease loss during revalidation aborts fenced requeue CAS with FencedStaleWriteError!
        sc_row = db.get_decision_request(req_sc["req_id"])
        assert sc_row is not None
        real_lease = str(sc_row["payload"]["lease_id"])
        orig_reval = router.revalidate_pinned_route

        def _release_mid_reval(*args: Any, **kwargs: Any) -> Any:
            db.release_lease(real_lease)
            return orig_reval(*args, **kwargs)

        router.revalidate_pinned_route = _release_mid_reval  # type: ignore[method-assign]
        try:
            with pytest.raises(FencedStaleWriteError, match="LEASE_MISMATCH|lease"):
                engine_hq.process_request(req_sc["req_id"], llm_caller=_should_not_run, dry_run_llm=False)
        finally:
            router.revalidate_pinned_route = orig_reval  # type: ignore[method-assign]
        assert db.get_decision_request(req_sc["req_id"])["status"] == "RUNNING"
        assert db.get_goal_token_usage(goal.goal_id)["reserved_tokens"] > 0

        # Re-acquire valid lease -> hypothesis+workflow-scoped rule forces clean PENDING requeue with 0 LLM calls!
        re_ls = db.acquire_lease(
            f"decision:{goal.goal_id}:{req_sc['req_id']}",
            f"dec-{req_sc['req_id']}",
            holder="test-holder-reval",
            ttl_s=120.0,
            goal_id=goal.goal_id,
        )
        out_sc = engine_hq.process_request(
            req_sc["req_id"],
            llm_caller=_should_not_run,
            dry_run_llm=False,
            expected_lease_id=re_ls["lease_id"],
        )
        assert llm_calls == 0
        assert out_sc["status"] == "PENDING"
        assert out_sc["attempt"] == 2
        assert db.get_goal_token_usage(goal.goal_id)["reserved_tokens"] == 0


def test_stage5_p1_06_r5_mandatory_r1_strict_contract_and_s3_transient_retry(
    tmp_path: pathlib.Path,
) -> None:
    """P1-06-R5 + R5 P2 #1: mandatory triggers beat overrides, R1 fast-lane requires baseline/claim/cpu, unknown fail closes."""
    from conjectra.planning import (
        PlanLifecycleManager,
        classify_design_findings,
        validate_r1_fast_lane_contract,
    )
    from conjectra.types import ExperimentPlan, WorkflowNode

    with StateDB(tmp_path / "arr_p106") as db:
        router = _FakeModelRouter()
        rt = WorkflowRuntime(db, router=router)
        goal_std = _goal("g_p106_std", risk_profile="standard")
        goal_high = _goal("g_p106_high", risk_profile="high")
        db.save_goal(goal_std)
        db.save_goal(goal_high)

        # Trigger A: require_design_review=False & FAST_LANE CANNOT bypass F2, F3, high risk, holdout, paid GPU, cloud, irreversible!
        mandatory_cases = [
            ("h_f2_over", goal_std, FidelityRung.F2_FULL_CONFIRM, [], {"requires_llm_code": True, "require_design_review": False}),
            ("h_f3_over", goal_std, FidelityRung.F3_MILESTONE_AUDIT, [], {"requires_llm_code": True, "require_design_review": False}),
            ("h_high_risk", goal_high, FidelityRung.F1_FAST_SCREEN, [], {"requires_llm_code": True, "require_design_review": False}),
            ("h_holdout", goal_std, FidelityRung.F1_FAST_SCREEN, ["holdout"], {"requires_llm_code": True, "require_design_review": False, "touches_holdout": True}),
            ("h_paid_gpu", goal_std, FidelityRung.F1_FAST_SCREEN, [], {"requires_llm_code": True, "require_design_review": False, "requires_paid_gpu": True}),
            ("h_cloud_gpu", goal_std, FidelityRung.F1_FAST_SCREEN, [], {"requires_llm_code": True, "require_design_review": False, "required_hardware": "cloud_gpu"}),
            ("h_irrev", goal_std, FidelityRung.F1_FAST_SCREEN, [], {"requires_llm_code": True, "require_design_review": False, "irreversible": True}),
        ]
        for hid, gg, fid, tags, prm in mandatory_cases:
            hh = HypothesisNode.from_dict(
                {
                    **_hyp(gg, hid, tags=tags, params=prm).to_dict(),
                    "fidelity": fid.value,
                }
            )
            db.register_hypothesis(hh)
            wf, tasks = rt.create_workflow_for_hypothesis(
                gg,
                hh,
                primitive="direct_hq",
                config_override={"require_design_review": False, "review_tier": ReviewTier.FAST_LANE.value},
            )
            assert tasks and tasks[0].stage == StageKind.DESIGN_REVIEW, f"{hid} bypassed R1 DESIGN_REVIEW!"
            assert wf.status == "RUNNING"

        # Trigger B: F1 with require_design_review=False MUST still fail closed on missing baseline, missing claim, or cloud_gpu!
        h_no_base = _hyp(goal_std, "h_no_base", params={"requires_llm_code": True, "require_design_review": False})
        db.register_hypothesis(h_no_base)
        db.save_experiment_plan(
            ExperimentPlan(
                exp_plan_id="ep_no_base",
                goal_id=goal_std.goal_id,
                hypothesis_id=h_no_base.hypothesis_id,
                plan_version=1,
                falsifiable_claim="Claim without baseline",
                control_baseline={},
                required_hardware="cpu_only",
            )
        )
        wf_nb, t_nb = rt.create_workflow_for_hypothesis(
            goal_std, h_no_base, primitive="direct_hq", config_override={"require_design_review": False}
        )
        assert t_nb == []
        assert db.get_workflow(wf_nb.workflow_id).status == "DESIGN_REJECTED"  # type: ignore[union-attr]

        h_no_claim = _hyp(goal_std, "h_no_claim", params={"requires_llm_code": True, "require_design_review": False})
        db.register_hypothesis(h_no_claim)
        db.save_experiment_plan(
            ExperimentPlan(
                exp_plan_id="ep_no_claim",
                goal_id=goal_std.goal_id,
                hypothesis_id=h_no_claim.hypothesis_id,
                plan_version=1,
                falsifiable_claim="",
                control_baseline={"primary_metric": "spectral_gap", "baseline_value": 0.25},
                required_hardware="cpu_only",
            )
        )
        wf_nc, t_nc = rt.create_workflow_for_hypothesis(
            goal_std, h_no_claim, primitive="direct_hq", config_override={"require_design_review": False}
        )
        assert t_nc == []
        assert db.get_workflow(wf_nc.workflow_id).status == "DESIGN_REJECTED"  # type: ignore[union-attr]

        # Direct contract also blocks cloud_gpu
        gate_cloud = validate_r1_fast_lane_contract(
            ExperimentPlan(
                exp_plan_id="ep_cloud",
                goal_id=goal_std.goal_id,
                hypothesis_id=h_no_base.hypothesis_id,
                falsifiable_claim="Cloud run",
                control_baseline={"primary_metric": "spectral_gap", "baseline_value": 0.25},
                required_hardware="cloud_gpu",
            ),
            goal_std,
            h_no_base,
        )
        assert gate_cloud.passed is False

        # Trigger C: structured finding classification fails closed on unknown reasons, allows structured THEORY advisory on reversible F1
        cls_unk = classify_design_findings(["Unexpected uncategorized failure string"])
        assert cls_unk["unknown_blockers"] and not cls_unk["advisory_checks"]

        h_rev_r1 = _hyp(goal_std, "h_rev_r1_cls", params={"requires_llm_code": True, "require_design_review": True})
        db.register_hypothesis(h_rev_r1)
        PlanLifecycleManager(db).create_or_revise_experiment_plan(goal_std, h_rev_r1)
        wf_unk, dr_unk = rt.create_workflow_for_hypothesis(goal_std, h_rev_r1, primitive="direct_hq")
        for idx, td in enumerate(dr_unk):
            out_unk = rt.advance_workflow(
                wf_unk,
                td,
                _rcpt(
                    h_rev_r1,
                    rid=f"r_unk_{idx}",
                    tid=td.task_id,
                    stage=StageKind.DESIGN_REVIEW,
                    averdict=AuditVerdict.FAIL,
                    reason="Completely opaque anomaly",
                ),
                goal_std,
            )
        assert out_unk == []
        assert db.get_workflow(wf_unk.workflow_id).status == "DESIGN_REJECTED"  # type: ignore[union-attr]

        # Structured THEORY critique on valid reversible F1 converts cleanly to advisory and enters build!
        h_rev_adv = _hyp(goal_std, "h_rev_adv_cls", params={"requires_llm_code": True, "require_design_review": True})
        db.register_hypothesis(h_rev_adv)
        PlanLifecycleManager(db).create_or_revise_experiment_plan(goal_std, h_rev_adv)
        wf_adv, dr_adv = rt.create_workflow_for_hypothesis(goal_std, h_rev_adv, primitive="direct_hq")
        for idx, td in enumerate(dr_adv):
            out_adv = rt.advance_workflow(
                wf_adv,
                td,
                _rcpt(
                    h_rev_adv,
                    rid=f"r_adv_{idx}",
                    tid=td.task_id,
                    stage=StageKind.DESIGN_REVIEW,
                    averdict=AuditVerdict.FAIL,
                    reason="Theoretical derivation caveat in bound",
                    metrics={"findings": [{"category": "THEORY", "message": "Check second-order remainder"}]},
                ),
                goal_std,
            )
        assert out_adv and out_adv[0].stage == StageKind.CODE
        wf_adv_done = db.get_workflow(wf_adv.workflow_id)
        assert wf_adv_done is not None and wf_adv_done.state.get("design_review_advisory") is True

        # R5 P2 #1: transient failure during legacy S3 respawn retries instead of setting migration_respawn_attempted permanently!
        h_s3_retry = _hyp(goal_std, "h_s3_retry")
        db.register_hypothesis(h_s3_retry)
        db.save_workflow(
            WorkflowNode(
                workflow_id="wf_leg_s3_retry",
                goal_id=goal_std.goal_id,
                hypothesis_id=h_s3_retry.hypothesis_id,
                primitive="direct_hq",
                status="SYNTHESIS_SUPERSEDED",
                config={
                    "synthesis_mode": True,
                    "synthesized_hypothesis_ids": [h_s3_retry.hypothesis_id],
                    "uncertainties": [{"id": "u1"}],
                    "thesis": "Retry synthesis",
                },
                state={
                    "superseded_reason": "LEGACY_S3_SYNTHESIS_MISSING_SOURCE_SNAPSHOT",
                    "stale_sources": ["missing_source_lineage_snapshot:legacy_s3"],
                },
            )
        )
        orig_spawn = rt.spawn_synthesis_review
        calls = 0

        def _flaky_spawn(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("transient sqlite lock during respawn")
            return orig_spawn(*args, **kwargs)

        rt.spawn_synthesis_review = _flaky_spawn  # type: ignore[method-assign]
        try:
            first_try = rt.respawn_superseded_legacy_synthesis_workflows(goal_id=goal_std.goal_id)
            assert first_try == []
            wf_mid = db.get_workflow("wf_leg_s3_retry")
            assert wf_mid is not None and not wf_mid.state.get("migration_respawn_attempted")
            assert wf_mid.state.get("respawn_transient_attempts") == 1
            second_try = rt.respawn_superseded_legacy_synthesis_workflows(goal_id=goal_std.goal_id)
            assert len(second_try) == 1
            wf_done = db.get_workflow("wf_leg_s3_retry")
            assert wf_done is not None and wf_done.state.get("migration_respawn_attempted") is True
        finally:
            rt.spawn_synthesis_review = orig_spawn  # type: ignore[method-assign]


def test_slice_3_sources_contain_no_prohibited_internal_ids() -> None:
    staging_root = pathlib.Path(__file__).resolve().parents[1]
    for rel in (
        "conjectra/decision_engine.py",
        "conjectra/explore.py",
        "conjectra/workflow_runtime.py",
    ):
        text = (staging_root / rel).read_text(encoding="utf-8")
        gate = check_no_internal_ids(text)
        assert gate.passed is True, f"{rel} failed check_no_internal_ids: {gate.failures}"


@pytest.mark.parametrize("primitive", ["direct_hq", "x-boost", "x-swarm"])
def test_failed_cpu_gate_repair_tasks_are_immediately_schedulable(
    tmp_path: pathlib.Path,
    primitive: str,
) -> None:
    """Repair tasks emitted after a FAILED CPU_GATE must not depend on the failed task ID."""
    with StateDB(tmp_path / "arr") as db:
        goal = _goal()
        db.save_goal(goal)
        router = _FakeModelRouter()
        rt = WorkflowRuntime(db, router=router)

        hyp = _hyp(goal, f"h_{primitive}_fail_gate", params={"requires_llm_code": True, "max_repairs": 1})
        db.register_hypothesis(hyp)
        cfg: dict[str, Any] = {"max_repairs": 1, "require_design_review": False}
        expected_step = "code_repair_1"
        if primitive == "x-boost":
            expected_step = "boost_repair_1"
        elif primitive == "x-swarm":
            cfg = {
                "max_ticks": 2,
                "require_design_review": False,
                "workers": [{"name": "s1", "llm_pool": "claude:claude-sonnet-5-5", "writable_paths": ["src/"]}],
            }
            expected_step = "swarm_worker:s1:t2"

        wf, init_tasks = rt.create_workflow_for_hypothesis(
            goal, hyp, primitive=primitive, config_override=cfg
        )
        t_gate = rt.advance_workflow(
            wf, init_tasks[0], _rcpt(hyp, rid="r_c", tid=init_tasks[0].task_id, stage=StageKind.CODE), goal
        )[0]
        db.transition_task_status(t_gate.task_id, "READY", "FAILED", finished_ts=1700000000.0)
        rep_tasks = rt.advance_workflow(
            wf,
            t_gate,
            _rcpt(
                hyp,
                rid="r_g",
                tid=t_gate.task_id,
                stage=StageKind.CPU_GATE,
                pstatus=ProcessStatus.EXIT_NONZERO,
                everdict=EngineeringVerdict.FAIL_TEST,
                reason="pytest failed",
            ),
            goal,
        )
        assert len(rep_tasks) == 1 and rep_tasks[0].step_key == expected_step
        assert rep_tasks[0].deps == []
        assert rep_tasks[0].task_id in {t.task_id for t in db.list_ready_tasks(goal_id=goal.goal_id)}


def test_failed_review_slot_retry_is_immediately_schedulable(
    tmp_path: pathlib.Path,
) -> None:
    """Retry review tasks emitted after a crashed (FAILED) review slot must not depend on the failed task ID."""
    with StateDB(tmp_path / "arr") as db:
        goal = _goal()
        db.save_goal(goal)
        router = _FakeModelRouter()
        rt = WorkflowRuntime(db, router=router)
        hyp = _hyp(goal, "h_rev_retry", params={"requires_llm_code": True, "require_design_review": True})
        db.register_hypothesis(hyp)
        wf, dr_tasks = rt.create_workflow_for_hypothesis(goal, hyp, primitive="direct_hq")
        t_slot0 = dr_tasks[0]
        db.transition_task_status(t_slot0.task_id, "READY", "FAILED", finished_ts=1700000000.0)
        retry_tasks = rt.advance_workflow(
            wf,
            t_slot0,
            _rcpt(
                hyp,
                rid="r_dr_err",
                tid=t_slot0.task_id,
                stage=StageKind.DESIGN_REVIEW,
                pstatus=ProcessStatus.ENV_ERROR,
                averdict=AuditVerdict.FAIL,
                reason="transient review failure",
            ),
            goal,
        )
        assert len(retry_tasks) == 1
        assert retry_tasks[0].deps == []
        assert retry_tasks[0].task_id in {t.task_id for t in db.list_ready_tasks(goal_id=goal.goal_id)}


def test_r6_typed_unknown_finding_fails_closed_and_default_router_passes_arbiter(
    tmp_path: pathlib.Path,
) -> None:
    """P1-04-R6 + R6 P2: unrecognized typed category fails closed; DecisionEngine._try_build_default_router receives arbiter."""
    from conjectra.arbiter import ResourceArbiter
    from conjectra.decision_engine import DecisionEngine, _try_build_default_router
    from conjectra.planning import classify_design_findings, extract_hard_design_blockers

    cls_res = classify_design_findings(
        [{"category": "UNRECOGNIZED_CATEGORY", "message": "check seed variance and methodology ablation"}]
    )
    assert cls_res["advisory_checks"] == []
    assert len(cls_res["unknown_blockers"]) == 1
    assert "UNRECOGNIZED_CATEGORY" in cls_res["unknown_blockers"][0]
    assert len(extract_hard_design_blockers([{"category": "CUSTOM_FAIL", "message": "seed"}])) == 1

    arb = ResourceArbiter()
    built = _try_build_default_router(arbiter=arb, db=None)
    assert built is not None

    with StateDB(tmp_path / "arr_de") as db:
        goal = _goal()
        db.save_goal(goal)
        eng = DecisionEngine(db, arbiter=arb)
        dreq = eng.enqueue_decision(
            goal.goal_id,
            role=ScientificRole.RESEARCH_LEAD,
            tier="L1",
            trigger_kind="INITIAL_PLANNING",
        )
        out = eng.process_request(str(dreq["req_id"]), dry_run_llm=True)
        assert out["status"] == "RESOLVED"
