"""Versioned research/experiment/execution planning and 3-way review quorum reducer."""

from __future__ import annotations

import json
import pathlib
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from conjectra.gates import (
    GateResult,
    parse_structured_verdict,
    validate_experiment_plan_gate,
)
from conjectra.state_db import StateDB
from conjectra.types import (
    AuditVerdict,
    EvidenceReceipt,
    ExecutionPlan,
    ExperimentPlan,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ResearchPlan,
    ReviewKind,
    ReviewReceipt,
    ReviewTier,
    StageTask,
)

INITIAL_QUORUM_DEADLINE_S: float = 600.0
TIEBREAKER_DEADLINE_S: float = 300.0

HIGH_RISK_REVIEW_TAGS: frozenset[str] = frozenset(
    {
        "theory",
        "backbone",
        "milestone",
        "architecture",
        "theorem",
        "closeout",
        "red_team",
    }
)

FAST_LANE_SAFE_TAGS: frozenset[str] = frozenset(
    {
        "typo",
        "doc",
        "docs",
        "formatting",
        "test_only",
        "config_tweak",
    }
)

NON_VERDICT_PROCESS_STATUSES: frozenset[ProcessStatus] = frozenset(
    {
        ProcessStatus.EXIT_NONZERO,
        ProcessStatus.TIMEOUT,
        ProcessStatus.OOM,
        ProcessStatus.ENV_ERROR,
        ProcessStatus.LEASE_LOST,
    }
)

AUTHOR_ROLE_KEYS: frozenset[str] = frozenset(
    {
        "research_director",
        "research_lead",
        "theorist",
        "experiment_designer",
        "engineer_builder",
        "proposer",
        "builder",
    }
)


def infer_model_family(model_or_spec: str) -> str:
    """Infer normalized provider family from a model identifier or canonical spec."""
    token = (model_or_spec or "").strip().lower()
    if not token:
        return ""
    if any(k in token for k in ("claude", "opus", "sonnet", "haiku", "anthropic")):
        return "anthropic"
    if any(k in token for k in ("gpt-", "codex", "openai", "webchat")):
        return "openai"
    if any(k in token for k in ("gemini", "google")):
        return "google"
    if "deepseek" in token:
        return "deepseek"
    if any(k in token for k in ("grok", "xai")):
        return "xai"
    return f"custom_{token.split(':', 1)[0]}"


def collect_contributor_families(
    *,
    exec_plan: ExecutionPlan | None = None,
    tasks: Sequence[StageTask] = (),
    receipts: Sequence[EvidenceReceipt] = (),
    extra_families: Sequence[str] = (),
) -> list[str]:
    """Collect normalized model families of every upstream proposer, designer, and builder."""
    families: set[str] = {
        str(f).strip().lower() for f in extra_families if str(f).strip()
    }
    if exec_plan is not None:
        for fam in exec_plan.contributor_families:
            if str(fam).strip():
                families.add(str(fam).strip().lower())
        for role_key, route in exec_plan.role_routes.items():
            if str(role_key).strip().lower() not in AUTHOR_ROLE_KEYS:
                continue
            if isinstance(route, Mapping):
                fam = str(route.get("family") or "").strip().lower()
                if not fam:
                    fam = infer_model_family(
                        str(route.get("canonical_model") or route.get("pool") or "")
                    )
                if fam:
                    families.add(fam)

    for task in tasks:
        if task.stage.value not in ("PROPOSE", "CODE"):
            continue
        if isinstance(task.pinned_route, Mapping):
            fam = str(task.pinned_route.get("family") or "").strip().lower()
            if not fam:
                fam = infer_model_family(
                    str(task.pinned_route.get("canonical_model") or "")
                )
            if fam:
                families.add(fam)
        elif task.model_spec:
            fam = infer_model_family(task.model_spec)
            if fam:
                families.add(fam)

    for rec in receipts:
        if rec.stage.value not in ("PROPOSE", "CODE"):
            continue
        fam = infer_model_family(rec.model_used)
        if fam:
            families.add(fam)

    return sorted(families)


def classify_review_tier(
    goal: ResearchGoal,
    hyp: HypothesisNode,
    *,
    diff_lines: int = 0,
    compiled_policy: Mapping[str, Any] | None = None,
    workflow_primitive: str | None = None,
) -> ReviewTier:
    """Select risk/cost-adaptive ReviewTier for a hypothesis execution."""
    explicit_hyp = hyp.params.get("review_tier")
    if isinstance(explicit_hyp, str) and explicit_hyp.strip():
        return ReviewTier(explicit_hyp.strip().upper())

    norm_tags = {str(t).strip().lower() for t in hyp.tags if str(t).strip()}
    for extra_tag in hyp.params.get("tags", []) if isinstance(hyp.params.get("tags"), list) else []:
        if str(extra_tag).strip():
            norm_tags.add(str(extra_tag).strip().lower())

    policy = dict(compiled_policy) if isinstance(compiled_policy, Mapping) else dict(goal.routing_policy or {})
    overrides = policy.get("review_overrides")
    if isinstance(overrides, Mapping):
        for tag in sorted(norm_tags):
            if tag in overrides:
                return ReviewTier(str(overrides[tag]).strip().upper())
        if "*" in overrides:
            return ReviewTier(str(overrides["*"]).strip().upper())

    nl_source = f"{goal.routing_policy_nl} {policy.get('source_nl', '')}".lower()
    requires_cross_family_theory = (
        "跨家族审查" in nl_source or "cross-family" in nl_source or "red-team" in nl_source
    )

    prim = (workflow_primitive or str(hyp.params.get("workflow_primitive") or "")).strip().lower()

    if (
        bool(norm_tags & HIGH_RISK_REVIEW_TAGS)
        or hyp.fidelity == FidelityRung.F3_MILESTONE_AUDIT
        or prim == "x-teamwork"
        or goal.risk_profile.lower() in ("high", "critical", "red_team", "high_stakes")
        or (requires_cross_family_theory and ("theory" in norm_tags or "理论" in hyp.statement))
    ):
        return ReviewTier.RED_TEAM_TIEBREAKER

    if (
        goal.risk_profile.lower() not in ("strict",)
        and not requires_cross_family_theory
        and hyp.kind in (HypothesisKind.REFINEMENT, HypothesisKind.PARAMETER_RACE)
        and hyp.fidelity == FidelityRung.F0_SANITY
        and 0 < int(diff_lines) <= 40
        and (not norm_tags or norm_tags <= FAST_LANE_SAFE_TAGS)
    ):
        return ReviewTier.FAST_LANE

    return ReviewTier.STANDARD_CROSS_FAMILY


HARD_DESIGN_BLOCKER_TERMS: tuple[str, ...] = (
    "holdout",
    "banned_path",
    "banned path",
    "read_ban",
    "read-ban",
    "safety",
    "redline",
    "red_line",
    "missing_baseline",
    "missing baseline",
    "no baseline",
    "missing_metric",
    "missing metric",
    "no primary metric",
    "paid_gpu",
    "paid_compute",
    "cloud_gpu",
    "unapproved_gpu",
    "cross_goal",
    "main_branch",
    "protected_branch",
    "secret",
    "credential",
)


THEORY_METHODOLOGY_ADVISORY_TERMS: tuple[str, ...] = (
    "theor",
    "assump",
    "lipschitz",
    "proof",
    "derivation",
    "lemma",
    "theorem",
    "bound",
    "convergen",
    "confound",
    "methodolog",
    "sensitivity",
    "ablation",
    "hyperparameter",
    "variance",
    "seed",
    "sample",
    "generaliz",
    "empirical",
    "statistical",
    "mechanism",
    "calibrat",
    "prior art",
)


def is_reversible_f1_exploratory(
    goal: ResearchGoal,
    hyp: HypothesisNode,
    cfg: Mapping[str, Any] | None = None,
    *,
    ignore_explicit_design_review: bool = False,
) -> bool:
    """Return True ONLY for validated reversible F0/F1 exploratory hypotheses."""
    params = dict(hyp.params or {})
    cfg_map = dict(cfg or {})
    if bool(params.get("require_design_review_strict")) or bool(cfg_map.get("require_design_review_strict")):
        return False
    if hyp.fidelity not in (FidelityRung.F0_SANITY, FidelityRung.F1_FAST_SCREEN):
        return False
    if goal.risk_profile.lower() in ("high", "critical", "red_team", "high_stakes"):
        return False
    if any(
        bool(params.get(k))
        for k in ("touches_holdout", "requires_paid_gpu", "paid_cloud_gpu", "irreversible")
    ):
        return False
    if str(params.get("required_hardware") or "").strip().lower() == "cloud_gpu":
        return False
    norm_tags = {str(t).strip().lower() for t in hyp.tags if str(t).strip()}
    for extra_tag in params.get("tags", []) if isinstance(params.get("tags"), list) else []:
        if str(extra_tag).strip():
            norm_tags.add(str(extra_tag).strip().lower())
    non_reversible_tags = {
        "holdout",
        "banned_path",
        "paid_gpu",
        "cloud_gpu",
        "destructive",
        "irreversible",
        "security",
        "milestone",
    }
    if norm_tags & non_reversible_tags:
        return False
    if not ignore_explicit_design_review and (
        bool(cfg_map.get("require_design_review")) or bool(params.get("require_design_review"))
    ):
        return False
    return True


def classify_design_findings(
    findings: Sequence[Any],
    *,
    failure_reason: str = "",
    severity: str = "",
) -> dict[str, list[str]]:
    """Classify R1 review findings into structured categories; fail closed on UNKNOWN."""
    raw_items: list[tuple[str, str]] = []
    for f in findings:
        if isinstance(f, Mapping):
            cat = str(f.get("category") or f.get("type") or "").strip().upper()
            msg = str(f.get("message") or f.get("reason") or f.get("detail") or f).strip()
            if msg:
                raw_items.append((cat, msg))
        else:
            msg = str(f).strip()
            if msg:
                raw_items.append(("", msg))
    fr = str(failure_reason or "").strip()
    if fr and not any(m == fr for _, m in raw_items):
        raw_items.append(("", fr))

    hard_blockers: list[str] = []
    advisory_checks: list[str] = []
    unknown_blockers: list[str] = []

    if not raw_items:
        unknown_blockers.append("R1_UNCLASSIFIED_EMPTY_FAILURE")
        return {
            "hard_blockers": hard_blockers,
            "advisory_checks": advisory_checks,
            "unknown_blockers": unknown_blockers,
        }

    advisory_cats = {"THEORY", "THEORETICAL", "METHODOLOGY", "METHODOLOGICAL", "CONFOUND", "ADVISORY"}
    hard_cats = {"SAFETY", "HOLDOUT", "BANNED_PATH", "MISSING_BASELINE", "MISSING_METRIC", "PAID_COMPUTE", "CROSS_GOAL"}

    for cat, msg in raw_items:
        low = msg.lower()
        if cat in hard_cats or any(term in low for term in HARD_DESIGN_BLOCKER_TERMS):
            hard_blockers.append(msg)
        elif cat:
            if cat in advisory_cats:
                advisory_checks.append(msg)
            else:
                unknown_blockers.append(f"UNCLASSIFIED_R1_BLOCKER({cat}): {msg}")
        elif any(term in low for term in THEORY_METHODOLOGY_ADVISORY_TERMS):
            advisory_checks.append(msg)
        else:
            unknown_blockers.append(f"UNCLASSIFIED_R1_BLOCKER(UNKNOWN): {msg}")

    return {
        "hard_blockers": hard_blockers,
        "advisory_checks": advisory_checks,
        "unknown_blockers": unknown_blockers,
    }


def extract_hard_design_blockers(
    findings: Sequence[Any],
    *,
    failure_reason: str = "",
    severity: str = "",
) -> list[str]:
    """Return all blocking findings (hard policy/safety violations plus unknown unclassified failures)."""
    classified = classify_design_findings(findings, failure_reason=failure_reason, severity=severity)
    return [*classified["hard_blockers"], *classified["unknown_blockers"]]


def validate_r1_fast_lane_contract(
    plan: ExperimentPlan,
    goal: ResearchGoal,
    hyp: HypothesisNode,
    *,
    repo_path: str | pathlib.Path | None = None,
) -> GateResult:
    """Strict R1 fast-lane gate enforcing claim, baseline, primary metric, non-paid local hardware, and safety."""
    res = validate_experiment_plan_gate(
        plan,
        goal,
        hyp,
        repo_path=repo_path or (goal.repo_path or None),
        require_existing_venv=False,
    )
    if plan.required_hardware == "cloud_gpu":
        res.failures.append("R1 fast lane prohibits cloud_gpu paid cloud hardware")
    if not is_reversible_f1_exploratory(goal, hyp, ignore_explicit_design_review=True):
        res.failures.append("Hypothesis is not eligible for reversible F0/F1 fast-lane approval")
    res.passed = len(res.failures) == 0
    return res


def validate_experiment_plan(
    plan: ExperimentPlan,
    goal: ResearchGoal | None = None,
    hyp: HypothesisNode | None = None,
    *,
    repo_path: str | pathlib.Path | None = None,
    require_existing_venv: bool = False,
    extra_bans: Sequence[str] = (),
) -> GateResult:
    """Validate an ExperimentPlan via deterministic safety/timeout/hardware gates."""
    return validate_experiment_plan_gate(
        plan,
        goal,
        hyp,
        repo_path=repo_path,
        require_existing_venv=require_existing_venv,
        extra_bans=extra_bans,
    )


class PlanLifecycleManager:
    """Manage versioned ResearchPlan, ExperimentPlan, and ExecutionPlan records in StateDB."""

    def __init__(self, db: StateDB) -> None:
        self.db = db

    def create_research_plan(
        self,
        goal: ResearchGoal,
        *,
        thesis: str,
        uncertainties: Sequence[Mapping[str, Any]] = (),
        competing_routes: Sequence[Mapping[str, Any]] = (),
        milestone_criteria: Sequence[str] = (),
        status: str = "APPROVED",
        plan_id: str | None = None,
    ) -> ResearchPlan:
        """Create and persist a new versioned ResearchPlan for a Goal."""
        plan = ResearchPlan(
            plan_id=plan_id or f"rplan_{uuid.uuid4().hex[:12]}",
            goal_id=goal.goal_id,
            plan_version=0,
            goal_version=int(goal.version),
            human_epoch=int(goal.human_epoch),
            plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
            thesis=thesis,
            uncertainties=[dict(u) for u in uncertainties],
            competing_routes=[dict(r) for r in competing_routes],
            milestone_criteria=[str(c) for c in milestone_criteria],
            status=status,
        )
        return self.db.save_research_plan(plan)

    def build_default_experiment_plan(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
    ) -> ExperimentPlan:
        """Synthesize a baseline ExperimentPlan from Goal & Hypothesis metadata."""
        cmd_blob = f"{hyp.screen_cmd} {hyp.confirm_cmd or ''}".lower()
        needs_gpu = (
            bool(hyp.params.get("gpu"))
            or "--device cuda" in cmd_blob
            or "cuda" in cmd_blob
            or str(hyp.params.get("required_hardware", "")).lower() == "local_gpu"
        )
        req_hw = str(
            hyp.params.get(
                "required_hardware",
                "local_gpu" if needs_gpu else "cpu_only",
            )
        )
        min_vram = int(hyp.params.get("min_vram_mb", 1500 if req_hw == "local_gpu" else 0))
        gate_timeout = int(hyp.params.get("gate_timeout_s", 90))
        screen_timeout = int(hyp.params.get("screen_timeout_s", 300))
        confirm_timeout = int(hyp.params.get("confirm_timeout_s", 1200))
        venv_python = str(hyp.params.get("venv_python", ""))
        return ExperimentPlan(
            exp_plan_id=f"eplan_{uuid.uuid4().hex[:12]}",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            plan_version=0,
            goal_version=int(goal.version),
            human_epoch=int(goal.human_epoch),
            plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
            falsifiable_claim=hyp.statement or hyp.title,
            control_baseline={
                "primary_metric": goal.primary_metric,
                "baseline_value": float(goal.baseline_value),
                "baseline_sha": goal.baseline_sha,
            },
            venv_python=venv_python,
            env_vars={
                str(k): str(v)
                for k, v in dict(hyp.params.get("env_vars") or {}).items()
            },
            required_hardware=req_hw,
            min_vram_mb=min_vram,
            gate_timeout_s=gate_timeout,
            screen_timeout_s=screen_timeout,
            confirm_timeout_s=confirm_timeout,
            confound_checks=list(goal.kill_criteria),
            review_status="PENDING",
        )

    def create_or_revise_experiment_plan(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        plan: ExperimentPlan | None = None,
        repo_path: str | pathlib.Path | None = None,
    ) -> ExperimentPlan:
        """Validate and persist a new version of a hypothesis ExperimentPlan."""
        if hyp.goal_id != goal.goal_id:
            raise ValueError(
                f"Cross-goal ownership violation: hypothesis {hyp.hypothesis_id!r} belongs to {hyp.goal_id!r}, not {goal.goal_id!r}"
            )
        if plan is not None:
            if plan.goal_id and plan.goal_id != goal.goal_id:
                raise ValueError(
                    f"Cross-goal ExperimentPlan ownership violation: plan.goal_id={plan.goal_id!r} != {goal.goal_id!r}"
                )
            if plan.hypothesis_id and plan.hypothesis_id != hyp.hypothesis_id:
                raise ValueError(
                    f"ExperimentPlan hypothesis mismatch: plan.hypothesis_id={plan.hypothesis_id!r} != {hyp.hypothesis_id!r}"
                )
            baseline_spec = dict(plan.control_baseline or {})
            if not baseline_spec:
                latest_ep = self.db.get_latest_experiment_plan(hyp.hypothesis_id)
                if latest_ep is not None and latest_ep.control_baseline:
                    baseline_spec = dict(latest_ep.control_baseline)
                elif str(goal.primary_metric or "").strip():
                    baseline_spec = {
                        "primary_metric": goal.primary_metric,
                        "baseline_value": float(goal.baseline_value),
                        "baseline_sha": goal.baseline_sha,
                    }
            candidate = ExperimentPlan.from_dict(
                {
                    **plan.to_dict(),
                    "goal_id": goal.goal_id,
                    "hypothesis_id": hyp.hypothesis_id,
                    "control_baseline": baseline_spec,
                }
            )
        else:
            candidate = self.build_default_experiment_plan(goal, hyp)
        gate = validate_experiment_plan(candidate, goal, hyp, repo_path=repo_path)
        if not gate.passed:
            raise ValueError(
                f"ExperimentPlan validation failed for {hyp.hypothesis_id}: {'; '.join(gate.failures)}"
            )
        return self.db.save_experiment_plan(candidate)

    def create_execution_plan(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        workflow_id: str,
        primitive: str = "direct_hq",
        review_tier: ReviewTier | None = None,
        slices: Sequence[Mapping[str, Any]] = (),
        role_routes: Mapping[str, Mapping[str, Any]] | None = None,
        contributor_families: Sequence[str] = (),
        budget_tokens: int = 50000,
        diff_lines: int = 0,
        exec_plan_id: str | None = None,
    ) -> ExecutionPlan:
        """Create and persist a workflow ExecutionPlan with full contributor lineage."""
        eff_tier = review_tier or classify_review_tier(
            goal,
            hyp,
            diff_lines=diff_lines,
            workflow_primitive=primitive,
        )
        route_map = {
            str(k): dict(v) for k, v in (role_routes or {}).items()
        }
        temp_plan = ExecutionPlan(
            exec_plan_id=exec_plan_id or f"xplan_{uuid.uuid4().hex[:12]}",
            workflow_id=workflow_id,
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            goal_version=int(goal.version),
            human_epoch=int(goal.human_epoch),
            plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
            primitive=primitive,
            review_tier=eff_tier,
            slices=[dict(s) for s in slices],
            role_routes=route_map,
            contributor_families=list(contributor_families),
            budget_tokens=int(budget_tokens),
        )
        temp_plan.contributor_families = collect_contributor_families(
            exec_plan=temp_plan,
            extra_families=contributor_families,
        )
        return self.db.save_execution_plan(temp_plan)


# ------------------------------------ Context-Stripped 3-Way Review Prompts


def build_design_review_prompt(
    goal: ResearchGoal,
    hyp: HypothesisNode,
    *,
    research_plan: ResearchPlan | None = None,
    experiment_plan: ExperimentPlan | None = None,
) -> str:
    """Build a context-stripped prompt for independent Research/Experiment Design Review (R1)."""
    exp_dict = experiment_plan.to_dict() if experiment_plan is not None else {}
    rplan_thesis = research_plan.thesis if research_plan is not None else ""
    return (
        "You are an independent cross-family Scientific Design Reviewer (R1).\n"
        "Author rationale and chain-of-thought have been intentionally stripped.\n"
        "Audit whether this hypothesis and experiment plan form a fair, falsifiable,\n"
        "confound-controlled test without holdout leakage or baseline manipulation.\n\n"
        f"Goal ID: {goal.goal_id} (v{goal.version})\n"
        f"Primary Metric: {goal.primary_metric} (baseline={goal.baseline_value})\n"
        f"Kill Criteria: {json.dumps(goal.kill_criteria, ensure_ascii=False)}\n"
        f"Research Plan Thesis: {rplan_thesis}\n"
        f"Hypothesis ({hyp.hypothesis_id}, kind={hyp.kind.value}, fidelity={hyp.fidelity.value}):\n"
        f"  Title: {hyp.title}\n"
        f"  Claim: {hyp.statement}\n"
        f"  Mechanism: {hyp.mechanism}\n"
        f"  Writable Paths: {json.dumps(hyp.writable_paths, ensure_ascii=False)}\n"
        f"  Acceptance Cmd: {hyp.acceptance_cmd}\n"
        f"  Screen Cmd: {hyp.screen_cmd}\n"
        f"  Confirm Cmd: {hyp.confirm_cmd or ''}\n"
        f"ExperimentPlan: {json.dumps(exp_dict, sort_keys=True, ensure_ascii=False)}\n\n"
        "Respond with a fenced JSON object containing keys "
        '{"verdict": "PASS"|"FAIL", "severity": "BLOCKER"|"MAJOR"|"MINOR"|"NONE", "findings": [...]} '
        "followed by a final standalone line `DESIGN_VERDICT: PASS` or `DESIGN_VERDICT: FAIL`.\n"
    )


def build_impl_review_prompt(
    goal: ResearchGoal,
    hyp: HypothesisNode,
    *,
    diff_text: str,
    gate_summary: str = "",
    experiment_plan: ExperimentPlan | None = None,
) -> str:
    """Build a context-stripped prompt for independent Engineering Implementation Review (R2)."""
    exp_claim = (
        experiment_plan.falsifiable_claim
        if experiment_plan is not None
        else hyp.statement
    )
    return (
        "You are an independent cross-family Engineering Implementation Reviewer (R2).\n"
        "Author chain-of-thought has been stripped. Audit the exact git diff against the\n"
        "stated mechanism and falsifiable claim for silent bugs, deadlocks, metric spoofing,\n"
        "synthetic fallback data, weakened assertions, or scope violations.\n\n"
        f"Goal ID: {goal.goal_id} | Primary Metric: {goal.primary_metric}\n"
        f"Hypothesis ({hyp.hypothesis_id}): {hyp.title}\n"
        f"Falsifiable Claim: {exp_claim}\n"
        f"Mechanism: {hyp.mechanism}\n"
        f"Deterministic Gate Summary: {gate_summary}\n\n"
        "--- BEGIN GIT DIFF ---\n"
        f"{diff_text}\n"
        "--- END GIT DIFF ---\n\n"
        "Respond with a fenced JSON object containing keys "
        '{"verdict": "PASS"|"FAIL", "severity": "BLOCKER"|"MAJOR"|"MINOR"|"NONE", '
        '"findings": [...], "repro_python": "..."} '
        "followed by a final standalone line `IMPL_VERDICT: PASS` or `IMPL_VERDICT: FAIL`.\n"
    )


def build_result_review_prompt(
    goal: ResearchGoal,
    hyp: HypothesisNode,
    *,
    receipts: Sequence[EvidenceReceipt],
    experiment_plan: ExperimentPlan | None = None,
    diff_stat: str = "",
) -> str:
    """Build a context-stripped prompt for independent Scientific Result/Conclusion Review (R3)."""
    serialized_receipts = [
        {
            "receipt_id": r.receipt_id,
            "stage": r.stage.value,
            "head_sha": r.head_sha,
            "exit_code": r.exit_code,
            "process_status": r.process_status.value,
            "engineering_verdict": r.engineering_verdict.value,
            "scientific_verdict": r.scientific_verdict.value,
            "evidence_validity": r.evidence_validity.value,
            "metrics": r.metrics,
            "tests_passed": r.tests_passed,
            "tests_total": r.tests_total,
            "raw_artifact_paths": r.raw_artifact_paths,
        }
        for r in receipts
    ]
    baseline_info = (
        experiment_plan.control_baseline
        if experiment_plan is not None
        else {"primary_metric": goal.primary_metric, "baseline_value": goal.baseline_value}
    )
    return (
        "You are an independent cross-family Scientific Result Auditor (R3).\n"
        "Verify whether the measured empirical receipts genuinely support the hypothesis claim\n"
        "over the control baseline without metric mismatch, cherry-picking, or unverified extrapolation.\n\n"
        f"Goal ID: {goal.goal_id} (v{goal.version}) | Primary Metric: {goal.primary_metric}\n"
        f"Control Baseline: {json.dumps(baseline_info, sort_keys=True, ensure_ascii=False)}\n"
        f"Hypothesis ({hyp.hypothesis_id}): {hyp.statement}\n"
        f"Diff Stat: {diff_stat}\n"
        f"Empirical Receipts: {json.dumps(serialized_receipts, sort_keys=True, ensure_ascii=False)}\n\n"
        "Respond with a fenced JSON object containing keys "
        '{"verdict": "PASS"|"FAIL", "severity": "BLOCKER"|"MAJOR"|"MINOR"|"NONE", "findings": [...]} '
        "followed by a final standalone line `RESULT_VERDICT: PASS` or `RESULT_VERDICT: FAIL`.\n"
    )


def build_synthesis_review_prompt(
    goal: ResearchGoal,
    hypotheses: Sequence[HypothesisNode],
    receipts: Sequence[EvidenceReceipt],
    synthesis_claim: str,
) -> str:
    """Build cross-experiment S3 synthesis review prompt gating multi-hypothesis SYNTHESIZE_RESULTS."""
    hyp_summary = [
        {
            "hypothesis_id": h.hypothesis_id,
            "kind": h.kind.value,
            "status": h.status,
            "scientific_verdict": h.scientific_verdict.value,
            "statement": h.statement,
        }
        for h in hypotheses
    ]
    rcpt_summary = [
        {
            "receipt_id": r.receipt_id,
            "hypothesis_id": r.hypothesis_id,
            "stage": r.stage.value,
            "scientific_verdict": r.scientific_verdict.value,
            "evidence_validity": r.evidence_validity.value,
            "metrics": r.metrics,
            "head_sha": r.head_sha,
        }
        for r in receipts
    ]
    return (
        "You are an independent cross-family Synthesis Auditor (S3_CROSS_EXPERIMENT).\n"
        "Audit whether the proposed cross-experiment synthesis conclusion is strictly supported\n"
        "across all referenced hypotheses and verified empirical receipts without confounds or overreach.\n\n"
        f"Goal ID: {goal.goal_id} (v{goal.version}, plan_epoch={getattr(goal, 'plan_epoch', 0)}) | Primary Metric: {goal.primary_metric}\n"
        f"Proposed Synthesis Claim: {synthesis_claim}\n"
        f"Hypotheses: {json.dumps(hyp_summary, sort_keys=True, ensure_ascii=False)}\n"
        f"Empirical Receipts: {json.dumps(rcpt_summary, sort_keys=True, ensure_ascii=False)}\n\n"
        "Respond with a fenced JSON object containing keys "
        '{"verdict": "PASS"|"FAIL", "severity": "BLOCKER"|"MAJOR"|"MINOR"|"NONE", "findings": [...]} '
        "followed by a final standalone line `SYNTHESIS_VERDICT: PASS` or `SYNTHESIS_VERDICT: FAIL`.\n"
    )


def parse_review_output(
    text: str,
    *,
    review_kind: ReviewKind = ReviewKind.IMPLEMENTATION_REVIEW,
    process_status: ProcessStatus = ProcessStatus.EXIT_OK,
) -> tuple[AuditVerdict, str, list[str], str]:
    """Parse raw reviewer CLI output into `(AuditVerdict, severity, findings, repro_python)`."""
    if process_status != ProcessStatus.EXIT_OK:
        return (
            AuditVerdict.PARSE_ERROR,
            "MAJOR",
            [f"{review_kind.value} process ended with {process_status.value}"],
            "",
        )
    parsed = parse_structured_verdict(text, mode=review_kind.value.lower())
    if parsed.verdict == "PASS":
        return AuditVerdict.PASS, parsed.severity, list(parsed.findings), parsed.repro_python
    if parsed.verdict in ("FAIL", "BROKEN"):
        return AuditVerdict.FAIL, parsed.severity, list(parsed.findings), parsed.repro_python
    return (
        AuditVerdict.PARSE_ERROR,
        parsed.severity,
        list(parsed.findings) or ["Unparseable review verdict"],
        parsed.repro_python,
    )


# ------------------------------------ Restart-Durable Review Quorum Reducer


@dataclass(frozen=True)
class QuorumDecision:
    """Deterministic quorum reduction outcome across 1-slot or 2+1 red-team reviews."""

    status: str
    updated_state: dict[str, Any]
    slots_to_spawn: list[int] = field(default_factory=list)
    avoid_families: list[str] = field(default_factory=list)
    prefer_avoid_families: list[str] = field(default_factory=list)
    escalate_owner: bool = False
    findings: list[str] = field(default_factory=list)
    reason: str = ""


def classify_review_slot_vote(
    receipt: ReviewReceipt,
    *,
    contributor_families: Sequence[str] = (),
) -> tuple[str, str, list[str]]:
    """Return `(vote, reviewer_family, findings)` where vote is PASS|FAIL|NON_VERDICT|CANCELLED."""
    if receipt.process_status == ProcessStatus.CANCELLED:
        return "CANCELLED", "", ["Review task cancelled"]

    reviewer_family = str(receipt.reviewer_route.get("family") or "").strip().lower()
    if not reviewer_family:
        reviewer_family = infer_model_family(
            str(
                receipt.reviewer_route.get("canonical_model")
                or receipt.reviewer_route.get("pool")
                or ""
            )
        )

    banned_families = {str(f).strip().lower() for f in contributor_families if str(f).strip()}
    if reviewer_family and reviewer_family in banned_families:
        return (
            "NON_VERDICT",
            reviewer_family,
            [
                f"Reviewer family {reviewer_family!r} overlaps upstream contributor_families {sorted(banned_families)}"
            ],
        )

    if receipt.process_status in NON_VERDICT_PROCESS_STATUSES:
        return (
            "NON_VERDICT",
            reviewer_family,
            list(receipt.findings)
            or [f"Reviewer slot {receipt.slot_index} process_status={receipt.process_status.value}"],
        )

    if receipt.process_status != ProcessStatus.EXIT_OK:
        return (
            "NON_VERDICT",
            reviewer_family,
            list(receipt.findings)
            or [f"Reviewer slot {receipt.slot_index} non-OK status={receipt.process_status.value}"],
        )

    if receipt.verdict == AuditVerdict.PASS:
        return "PASS", reviewer_family, list(receipt.findings)
    if receipt.verdict == AuditVerdict.FAIL:
        return "FAIL", reviewer_family, list(receipt.findings)
    return (
        "NON_VERDICT",
        reviewer_family,
        list(receipt.findings)
        or [f"Reviewer slot {receipt.slot_index} verdict={receipt.verdict.value}"],
    )


def _slot_watermark(rec: ReviewReceipt) -> dict[str, Any]:
    return {
        "task_id": str(rec.task_id or ""),
        "created_ts": float(rec.created_ts),
        "reviewer_route": dict(rec.reviewer_route or {}),
    }


def _supersedes_watermark(rec: ReviewReceipt, wm: Mapping[str, Any]) -> bool:
    wm_ts = float(wm.get("created_ts", 0.0))
    rec_ts = float(rec.created_ts)
    if rec_ts < wm_ts:
        return False
    wm_task = str(wm.get("task_id") or "")
    rec_task = str(rec.task_id or "")
    if rec_task and wm_task and rec_task != wm_task:
        return True
    if rec_ts > wm_ts:
        return True
    if dict(rec.reviewer_route or {}) != dict(wm.get("reviewer_route") or {}):
        return True
    return False


class ReviewQuorumReducer:
    """Restart-durable reducer for FAST_LANE, STANDARD_CROSS_FAMILY, and RED_TEAM_TIEBREAKER."""

    @staticmethod
    def init_quorum_state(
        *,
        review_kind: ReviewKind,
        review_tier: ReviewTier,
        target_version: int = 1,
        plan_epoch: int = 0,
        human_epoch: int = 0,
        target_kind: str = "hypothesis",
        synthesized_hypothesis_ids: Sequence[str] = (),
        head_sha: str = "",
        contributor_families: Sequence[str] = (),
        quorum_deadline_s: float = INITIAL_QUORUM_DEADLINE_S,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Initialize persisted quorum state for `WorkflowNode.state['review_quorum']`."""
        ts = time.time() if now is None else float(now)
        return {
            "review_kind": review_kind.value,
            "review_tier": review_tier.value,
            "target_version": int(target_version),
            "plan_epoch": int(plan_epoch),
            "human_epoch": int(human_epoch),
            "target_kind": str(target_kind),
            "synthesized_hypothesis_ids": [str(x) for x in synthesized_hypothesis_ids if str(x)],
            "head_sha": str(head_sha),
            "phase": "PRIMARY",
            "quorum_deadline_ts": ts + float(quorum_deadline_s),
            "slot_retries": {"0": 0, "1": 0},
            "invalidated_slots": {},
            "tiebreaker_used": False,
            "contributor_families": sorted(
                {str(f).strip().lower() for f in contributor_families if str(f).strip()}
            ),
            "created_ts": ts,
        }

    @classmethod
    def evaluate_quorum(
        cls,
        quorum_state: Mapping[str, Any],
        receipts: Sequence[ReviewReceipt],
        *,
        now: float | None = None,
        tiebreaker_schedulable: bool = True,
        retry_schedulable: bool = True,
        require_fast_lane_slot: bool = False,
    ) -> QuorumDecision:
        """Reduce current slot receipts and persisted deadline into a deterministic QuorumDecision."""
        ts = time.time() if now is None else float(now)
        state = json.loads(json.dumps(dict(quorum_state)))
        tier = ReviewTier(str(state.get("review_tier", ReviewTier.STANDARD_CROSS_FAMILY.value)))
        target_version = int(state.get("target_version", 1))
        expected_plan_epoch = int(state.get("plan_epoch", 0))
        expected_human_epoch = state.get("human_epoch")
        head_sha = str(state.get("head_sha", ""))
        deadline_ts = float(state.get("quorum_deadline_ts", ts + INITIAL_QUORUM_DEADLINE_S))
        contrib_families = [
            str(f).strip().lower()
            for f in state.get("contributor_families", [])
            if str(f).strip()
        ]
        slot_retries: dict[str, int] = {
            "0": int((state.get("slot_retries") or {}).get("0", 0)),
            "1": int((state.get("slot_retries") or {}).get("1", 0)),
        }
        state["slot_retries"] = slot_retries
        raw_inv = state.get("invalidated_slots")
        invalidated_slots: dict[str, dict[str, Any]] = {
            str(k): dict(v)
            for k, v in (dict(raw_inv) if isinstance(raw_inv, Mapping) else {}).items()
            if isinstance(v, Mapping)
        }
        state["invalidated_slots"] = invalidated_slots

        by_slot: dict[int, ReviewReceipt] = {}
        for rec in receipts:
            if int(rec.target_version) != target_version:
                continue
            if int(getattr(rec, "plan_epoch", 0)) != expected_plan_epoch:
                continue
            if expected_human_epoch is not None and int(getattr(rec, "human_epoch", 0)) != int(expected_human_epoch):
                continue
            if head_sha and rec.head_sha and rec.head_sha != head_sha:
                continue
            s_idx = int(rec.slot_index)
            wm = invalidated_slots.get(str(s_idx))
            if wm is not None and not _supersedes_watermark(rec, wm):
                continue
            prev = by_slot.get(s_idx)
            if prev is None or float(rec.created_ts) >= float(prev.created_ts):
                by_slot[s_idx] = rec

        votes: dict[int, str] = {}
        families: dict[int, str] = {}
        findings_by_slot: dict[int, list[str]] = {}
        for slot_idx, rec in sorted(by_slot.items()):
            vote, fam, fnds = classify_review_slot_vote(
                rec, contributor_families=contrib_families
            )
            if vote == "CANCELLED":
                state["phase"] = "CANCELLED"
                return QuorumDecision(
                    status="CANCELLED",
                    updated_state=state,
                    avoid_families=contrib_families,
                    findings=fnds,
                    reason="Reviewer cancelled due to goal pause or version/epoch bump",
                )
            votes[slot_idx] = vote
            families[slot_idx] = fam
            findings_by_slot[slot_idx] = fnds

        all_findings: list[str] = []
        for idx in sorted(findings_by_slot):
            all_findings.extend(findings_by_slot[idx])

        # ---------------- FAST_LANE
        if tier == ReviewTier.FAST_LANE:
            if 0 not in votes:
                if not require_fast_lane_slot:
                    state["phase"] = "RESOLVED_PASS"
                    return QuorumDecision(
                        status="PASS",
                        updated_state=state,
                        avoid_families=contrib_families,
                        reason="FAST_LANE deterministic gate passed",
                    )
                if ts >= deadline_ts:
                    state["phase"] = "BLOCKED"
                    return QuorumDecision(
                        status="BLOCKED_REVIEW_QUORUM",
                        updated_state=state,
                        avoid_families=contrib_families,
                        escalate_owner=True,
                        findings=["FAST_LANE prescreen quorum deadline expired"],
                        reason="quorum_deadline_expired",
                    )
                return QuorumDecision(
                    status="WAIT_SLOTS",
                    updated_state=state,
                    avoid_families=contrib_families,
                    reason="Waiting for FAST_LANE slot 0",
                )
            if votes[0] == "PASS":
                state["phase"] = "RESOLVED_PASS"
                return QuorumDecision(
                    status="PASS",
                    updated_state=state,
                    avoid_families=contrib_families,
                    findings=all_findings,
                    reason="FAST_LANE slot 0 passed",
                )
            if votes[0] == "FAIL":
                state["phase"] = "RESOLVED_FAIL"
                return QuorumDecision(
                    status="FAIL",
                    updated_state=state,
                    avoid_families=contrib_families,
                    findings=all_findings,
                    reason="FAST_LANE slot 0 failed",
                )
            state["phase"] = "BLOCKED"
            return QuorumDecision(
                status="BLOCKED_REVIEW_QUORUM",
                updated_state=state,
                avoid_families=contrib_families,
                escalate_owner=True,
                findings=all_findings,
                reason="FAST_LANE slot 0 returned NON_VERDICT",
            )

        # ---------------- STANDARD_CROSS_FAMILY (1 distinct-family slot)
        if tier == ReviewTier.STANDARD_CROSS_FAMILY:
            if 0 not in votes:
                if ts >= deadline_ts:
                    state["phase"] = "BLOCKED"
                    return QuorumDecision(
                        status="BLOCKED_REVIEW_QUORUM",
                        updated_state=state,
                        avoid_families=contrib_families,
                        escalate_owner=True,
                        findings=["STANDARD_CROSS_FAMILY quorum deadline expired"],
                        reason="quorum_deadline_expired",
                    )
                return QuorumDecision(
                    status="WAIT_SLOTS",
                    updated_state=state,
                    avoid_families=contrib_families,
                    reason="Waiting for cross-family reviewer slot 0",
                )
            v0 = votes[0]
            if v0 == "PASS":
                state["phase"] = "RESOLVED_PASS"
                return QuorumDecision(
                    status="PASS",
                    updated_state=state,
                    avoid_families=contrib_families,
                    findings=all_findings,
                    reason="STANDARD_CROSS_FAMILY slot 0 passed",
                )
            if v0 == "FAIL":
                state["phase"] = "RESOLVED_FAIL"
                return QuorumDecision(
                    status="FAIL",
                    updated_state=state,
                    avoid_families=contrib_families,
                    findings=all_findings,
                    reason="STANDARD_CROSS_FAMILY slot 0 failed",
                )
            if slot_retries["0"] < 1 and retry_schedulable and ts < deadline_ts:
                slot_retries["0"] += 1
                invalidated_slots["0"] = _slot_watermark(by_slot[0])
                prefer = sorted(set(contrib_families) | ({families[0]} if families.get(0) else set()))
                return QuorumDecision(
                    status="NEED_RETRY_SLOTS",
                    updated_state=state,
                    slots_to_spawn=[0],
                    avoid_families=contrib_families,
                    prefer_avoid_families=prefer,
                    findings=all_findings,
                    reason="Retrying NON_VERDICT slot 0 once on alternate healthy family",
                )
            state["phase"] = "BLOCKED"
            return QuorumDecision(
                status="BLOCKED_REVIEW_QUORUM",
                updated_state=state,
                avoid_families=contrib_families,
                escalate_owner=True,
                findings=all_findings,
                reason="STANDARD_CROSS_FAMILY slot 0 unschedulable or repeated NON_VERDICT",
            )

        # ---------------- RED_TEAM_TIEBREAKER (2 distinct-family slots + 1-shot Director tie-breaker)
        if 0 in families and 1 in families and families[0] and families[0] == families[1]:
            votes[1] = "NON_VERDICT"
            dup_msg = f"Primary red-team slots 0 and 1 used identical family {families[0]!r}"
            findings_by_slot.setdefault(1, []).append(dup_msg)
            all_findings.append(dup_msg)

        phase = str(state.get("phase", "PRIMARY"))
        if phase == "TIEBREAKER" or 2 in votes:
            if 2 in votes:
                v2 = votes[2]
                if v2 == "PASS":
                    state["phase"] = "RESOLVED_PASS"
                    state["tiebreaker_used"] = True
                    return QuorumDecision(
                        status="PASS",
                        updated_state=state,
                        avoid_families=contrib_families,
                        findings=all_findings,
                        reason="Director tie-breaker slot 2 resolved PASS",
                    )
                if v2 == "FAIL":
                    state["phase"] = "RESOLVED_FAIL"
                    state["tiebreaker_used"] = True
                    return QuorumDecision(
                        status="FAIL",
                        updated_state=state,
                        avoid_families=contrib_families,
                        findings=all_findings,
                        reason="Director tie-breaker slot 2 resolved FAIL",
                    )
                state["phase"] = "BLOCKED"
                state["tiebreaker_used"] = True
                return QuorumDecision(
                    status="BLOCKED_REVIEW_QUORUM",
                    updated_state=state,
                    avoid_families=contrib_families,
                    escalate_owner=True,
                    findings=all_findings,
                    reason="Director tie-breaker slot 2 returned NON_VERDICT",
                )

            if not tiebreaker_schedulable or ts >= deadline_ts:
                state["phase"] = "BLOCKED"
                return QuorumDecision(
                    status="BLOCKED_REVIEW_QUORUM",
                    updated_state=state,
                    avoid_families=contrib_families,
                    escalate_owner=True,
                    findings=all_findings + ["Tie-breaker unschedulable or deadline expired"],
                    reason="tiebreaker_unschedulable_or_expired",
                )
            return QuorumDecision(
                status="WAIT_SLOTS",
                updated_state=state,
                avoid_families=contrib_families,
                reason="Waiting for Director tie-breaker slot 2",
            )

        # Primary 2-slot evaluation
        if 0 not in votes or 1 not in votes:
            if ts >= deadline_ts:
                state["phase"] = "BLOCKED"
                return QuorumDecision(
                    status="BLOCKED_REVIEW_QUORUM",
                    updated_state=state,
                    avoid_families=contrib_families,
                    escalate_owner=True,
                    findings=all_findings + ["Primary 2-slot red-team quorum deadline expired"],
                    reason="quorum_deadline_expired",
                )
            return QuorumDecision(
                status="WAIT_SLOTS",
                updated_state=state,
                avoid_families=contrib_families,
                reason="Waiting for primary red-team slots 0 and 1",
            )

        v0, v1 = votes[0], votes[1]

        if v0 == "PASS" and v1 == "PASS":
            state["phase"] = "RESOLVED_PASS"
            return QuorumDecision(
                status="PASS",
                updated_state=state,
                avoid_families=contrib_families,
                findings=all_findings,
                reason="Both cross-family red-team reviewers passed",
            )

        if (v0 == "FAIL" and v1 in ("FAIL", "NON_VERDICT")) or (
            v1 == "FAIL" and v0 in ("FAIL", "NON_VERDICT")
        ):
            state["phase"] = "RESOLVED_FAIL"
            return QuorumDecision(
                status="FAIL",
                updated_state=state,
                avoid_families=contrib_families,
                findings=all_findings,
                reason=f"Fail-closed red-team verdict ({v0}, {v1})",
            )

        if (v0 == "PASS" and v1 in ("FAIL", "NON_VERDICT")) or (
            v1 == "PASS" and v0 in ("FAIL", "NON_VERDICT")
        ):
            if not tiebreaker_schedulable or ts >= deadline_ts:
                state["phase"] = "BLOCKED"
                return QuorumDecision(
                    status="BLOCKED_REVIEW_QUORUM",
                    updated_state=state,
                    avoid_families=contrib_families,
                    escalate_owner=True,
                    findings=all_findings
                    + ["Split quorum requires tie-breaker, but tie-breaker cannot be scheduled"],
                    reason="tiebreaker_unschedulable_or_expired",
                )
            state["phase"] = "TIEBREAKER"
            state["tiebreaker_used"] = True
            state["quorum_deadline_ts"] = ts + TIEBREAKER_DEADLINE_S
            prefer_avoid = sorted(
                set(contrib_families)
                | {f for f in (families.get(0, ""), families.get(1, "")) if f}
            )
            return QuorumDecision(
                status="NEED_TIEBREAKER",
                updated_state=state,
                slots_to_spawn=[2],
                avoid_families=contrib_families,
                prefer_avoid_families=prefer_avoid,
                findings=all_findings,
                reason=f"Split red-team quorum ({v0}, {v1}) escalated to Director slot 2",
            )

        # Remaining case: (NON_VERDICT, NON_VERDICT)
        if (
            slot_retries["0"] < 1
            and slot_retries["1"] < 1
            and retry_schedulable
            and ts < deadline_ts
        ):
            slot_retries["0"] += 1
            slot_retries["1"] += 1
            invalidated_slots["0"] = _slot_watermark(by_slot[0])
            invalidated_slots["1"] = _slot_watermark(by_slot[1])
            prefer_avoid = sorted(
                set(contrib_families)
                | {f for f in (families.get(0, ""), families.get(1, "")) if f}
            )
            return QuorumDecision(
                status="NEED_RETRY_SLOTS",
                updated_state=state,
                slots_to_spawn=[0, 1],
                avoid_families=contrib_families,
                prefer_avoid_families=prefer_avoid,
                findings=all_findings,
                reason="Both primary red-team slots returned NON_VERDICT; retrying once",
            )

        state["phase"] = "BLOCKED"
        return QuorumDecision(
            status="BLOCKED_REVIEW_QUORUM",
            updated_state=state,
            avoid_families=contrib_families,
            escalate_owner=True,
            findings=all_findings,
            reason="Repeated or unschedulable (NON_VERDICT, NON_VERDICT) in red-team quorum",
        )
