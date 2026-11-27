"""Unified durable 4-level Goal -> Hypothesis -> WorkflowNode -> StageTask runtime with 3-way reviews."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
import importlib
from pathlib import Path
import time
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable
import uuid

from conjectra import gates
from conjectra.adapters import get_adapter
from conjectra.backends import DEFAULT_BACKEND_FOR_POOL
from conjectra.arbiter import POOL_NORMALIZATION, ResourceArbiter
from conjectra.gates import classify_failure_domain, is_valid_empirical_outcome
from conjectra.multiagent import (
    MAX_MILESTONE_ANCHOR_BYTES,
    build_milestone_anchor,
    parse_provider_spec,
)
from conjectra.planning import (
    HIGH_RISK_REVIEW_TAGS,
    PlanLifecycleManager,
    ReviewQuorumReducer,
    classify_review_tier,
    collect_contributor_families,
    extract_hard_design_blockers,
    infer_model_family,
    is_reversible_f1_exploratory,
    validate_r1_fast_lane_contract,
)
from conjectra.state_db import StateDB, StateTransitionError
from conjectra.types import (
    AuditVerdict,
    ConstraintStrength,
    EngineeringVerdict,
    EvidenceReceipt,
    FailureDomain,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ResolvedModelRoute,
    ResourceSpec,
    ReviewKind,
    ReviewReceipt,
    ReviewTier,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
    SubGoalKind,
    SubGoalNode,
    WorkflowNode,
)
from conjectra.workload_matrix import ensure_eval_table, extract_workload_features

SUPPORTED_PRIMITIVES: tuple[str, ...] = (
    "direct_hq",
    "x-boost",
    "x-swarm",
    "x-teamwork",
    "readonly_fanout",
)
TERMINAL_WORKFLOW_STATUSES = frozenset(
    {
        "COMPLETED",
        "COMPLETED_VALID_NEGATIVE",
        "FAILED",
        "CANCELLED",
        "STALE",
        "DESIGN_REJECTED",
        "RESULT_REJECTED",
        "BLOCKED_REVIEW_QUORUM",
        "SYNTHESIS_SUPERSEDED",
    }
)

_POOL_TO_HQ_RESOURCE: dict[str, str] = {
    "claude": "llmclaude",
    "codex": "llmcodexaudit",
    "gemini": "llmgemini",
    "deepseek": "llmdeepseek",
    "grok": "llmdeepseek",
}

_FALLBACK_DISTINCT_CANDIDATES: tuple[tuple[str, str, str, str], ...] = (
    ("codex", "gpt-5-codex", "high", "openai"),
    ("gemini", "gemini-pro", "high", "google"),
    ("claude", "claude-opus-5-5", "high", "anthropic"),
    ("deepseek", "deepseek-chat", "high", "deepseek"),
    ("grok", "grok-code-fast", "high", "xai"),
)


@runtime_checkable
class RouteResolverProtocol(Protocol):
    """Protocol implemented by Slice 2's `ModelRouter` for zero-import-coupling DI."""

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
    ) -> ResolvedModelRoute: ...


def _try_build_default_router(
    *,
    arbiter: ResourceArbiter | None,
    db: StateDB | None,
) -> RouteResolverProtocol | None:
    try:
        mod = importlib.import_module("conjectra.model_router")
        cls = getattr(mod, "ModelRouter", None)
        if cls is not None:
            return cls(arbiter=arbiter, db=db)
    except Exception:
        return None
    return None


def _resolve_subgoal_for_hyp(db: Any, hyp: HypothesisNode) -> SubGoalNode | None:
    """Resolve the SubGoalNode for `hyp` if `subgoal_id` is present and DB supports lookup."""
    params = hyp.params if isinstance(hyp.params, Mapping) else {}
    sg_id = getattr(hyp, "subgoal_id", None) or params.get("subgoal_id")
    if not sg_id or db is None or not hasattr(db, "get_subgoal"):
        return None
    try:
        return db.get_subgoal(str(sg_id))
    except Exception:
        return None


def _is_foundation_build_hyp(hyp: HypothesisNode, subgoal: SubGoalNode | None) -> bool:
    """Return True when a hypothesis belongs to a `FOUNDATION_BUILD` SubGoal."""
    if subgoal is not None:
        kind_val = subgoal.kind.value if hasattr(subgoal.kind, "value") else str(subgoal.kind)
        if str(kind_val).strip().upper() == SubGoalKind.FOUNDATION_BUILD.value:
            return True
    params = hyp.params if isinstance(hyp.params, Mapping) else {}
    return str(params.get("subgoal_kind") or "").strip().upper() == SubGoalKind.FOUNDATION_BUILD.value


@dataclass
class WorkflowSelection:
    """Outcome of dynamic outcome-driven primitive selection."""

    primitive: str
    rationale: str
    workload_class: str
    provider_pairing: str = ""
    switched_from: str | None = None
    coordination_tax_pct: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "primitive": self.primitive,
            "rationale": self.rationale,
            "workload_class": self.workload_class,
            "provider_pairing": self.provider_pairing,
            "switched_from": self.switched_from,
            "coordination_tax_pct": self.coordination_tax_pct,
            "details": dict(self.details),
        }

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_dict().get(key, default)

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]


class WorkflowSelector:
    """Selects and dynamically switches collaboration primitives using outcomes & workload_eval_ledger."""

    def __init__(self, max_coordination_tax_pct: float = 35.0) -> None:
        self.max_coordination_tax_pct = float(max_coordination_tax_pct)

    @staticmethod
    def _normalize_primitive(raw: str | None) -> str:
        val = (raw or "").strip().lower()
        if val == "run_workflow":
            return "readonly_fanout"
        if val in SUPPORTED_PRIMITIVES:
            return val
        return "direct_hq"

    @staticmethod
    def _query_ledger_stats(
        db: StateDB | None,
        workload_class: str,
    ) -> dict[str, dict[str, float]]:
        if db is None:
            return {}
        try:
            if int(getattr(db, "_tx_depth", 0) or 0) == 0:
                ensure_eval_table(db)
            with db._lock:
                tbl = db._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'workload_eval_ledger'"
                ).fetchone()
                if tbl is None:
                    return {}
                rows = db._conn.execute(
                    """
                    SELECT mode,
                           COUNT(*) AS n,
                           AVG(holdout_passed) AS pass_rate,
                           AVG(defect_escaped) AS defect_escape_rate,
                           AVG(coordination_tax_pct) AS avg_tax_pct,
                           AVG(effective_yield_per_100k) AS avg_yield
                    FROM workload_eval_ledger
                    WHERE workload_class = ? AND is_real_llm = 1
                    GROUP BY mode
                    """,
                    (workload_class,),
                ).fetchall()
            return {
                str(r["mode"]): {
                    "n": float(r["n"] or 0),
                    "pass_rate": float(r["pass_rate"] or 0.0),
                    "defect_escape_rate": float(r["defect_escape_rate"] or 0.0),
                    "avg_tax_pct": float(r["avg_tax_pct"] or 0.0),
                    "avg_yield": float(r["avg_yield"] or 0.0),
                }
                for r in rows
            }
        except Exception:
            return {}

    def select_primitive(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        db: StateDB | None = None,
        arbiter: ResourceArbiter | None = None,
        prior_receipts: Sequence[EvidenceReceipt] | None = None,
    ) -> WorkflowSelection:
        """Select optimal primitive with outcome-driven escalation/downsizing."""
        params = dict(hyp.params or {})
        readonly = bool(params.get("readonly", False))
        phases = params.get("phases")
        workers = params.get("workers") or params.get("slices")

        scopes: list[list[str]] = []
        if isinstance(workers, Sequence) and not isinstance(workers, (str, bytes)):
            for w in workers:
                if isinstance(w, Mapping) and w.get("writable_paths"):
                    scopes.append([str(p) for p in w["writable_paths"]])
        if not scopes and hyp.writable_paths:
            scopes = (
                [[str(p)] for p in hyp.writable_paths]
                if len(hyp.writable_paths) > 1
                else [list(hyp.writable_paths)]
            )

        milestone_phases = (
            len(phases)
            if isinstance(phases, Sequence) and not isinstance(phases, (str, bytes))
            else int(params.get("milestone_phases", 1))
        )
        interface_frozen = params.get("interface_frozen")
        desc = f"{hyp.title}\n{hyp.statement}\n{hyp.mechanism}"
        feats = extract_workload_features(
            desc or goal.title,
            writable_scopes=scopes or None,
            readonly=readonly,
            interface_frozen=bool(interface_frozen) if interface_frozen is not None else None,
            milestone_phases=max(1, milestone_phases),
        )
        w_class = feats.classify()
        subgoal = _resolve_subgoal_for_hyp(db, hyp)
        if _is_foundation_build_hyp(hyp, subgoal):
            explicit_raw = (
                params.get("workflow_primitive")
                or params.get("collab_mode")
                or params.get("routed_mode")
            )
            prev_prim = self._normalize_primitive(str(explicit_raw)) if explicit_raw else None
            return WorkflowSelection(
                primitive="direct_hq",
                rationale="FOUNDATION_BUILD_DIRECT_HQ_COLLAPSE",
                workload_class=w_class,
                switched_from=prev_prim if prev_prim and prev_prim != "direct_hq" else None,
            )

        ledger_stats = self._query_ledger_stats(db, w_class)

        receipts = list(prior_receipts or ())
        if not receipts and db is not None:
            receipts = db.list_receipts(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id)
            if hyp.parent_id:
                receipts.extend(
                    db.list_receipts(goal_id=goal.goal_id, hypothesis_id=hyp.parent_id)
                )

        explicit_raw = (
            params.get("workflow_primitive")
            or params.get("collab_mode")
            or params.get("routed_mode")
        )
        prev_prim = self._normalize_primitive(str(explicit_raw)) if explicit_raw else None

        self_receipts = [
            r for r in receipts if r.hypothesis_id == hyp.hypothesis_id
        ]
        has_audit_or_defect_escape = any(
            r.audit_verdict in (AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR)
            or any(
                k in (r.failure_reason or "").lower()
                for k in ("defect_escape", "blind_audit", "audit_fail", "counterexample", "subtle")
            )
            for r in self_receipts
        )
        has_interface_or_merge_mismatch = any(
            any(
                k in (r.failure_reason or "").lower()
                for k in ("merge_conflict", "interface_mismatch", "schema_mismatch", "wire_mismatch", "merge")
            )
            for r in self_receipts
        )
        measured_tax: float | None = None
        for r in reversed(receipts):
            if isinstance(r.metrics, Mapping) and "coordination_tax_pct" in r.metrics:
                ent = r.metrics["coordination_tax_pct"]
                raw_tax = ent.get("value") if isinstance(ent, Mapping) else ent
                try:
                    measured_tax = float(raw_tax)
                    break
                except (TypeError, ValueError):
                    pass

        if has_interface_or_merge_mismatch and (prev_prim in (None, "direct_hq")):
            return WorkflowSelection(
                primitive="x-swarm",
                rationale="SWITCH_DIRECT_TO_SWARM_ON_INTERFACE_OR_MERGE_MISMATCH",
                workload_class=w_class,
                switched_from=prev_prim or "direct_hq",
                coordination_tax_pct=measured_tax,
            )

        if has_audit_or_defect_escape and (prev_prim in (None, "direct_hq")):
            return WorkflowSelection(
                primitive="x-boost",
                rationale="SWITCH_DIRECT_TO_BOOST_ON_AUDIT_OR_DEFECT_ESCAPE",
                workload_class=w_class,
                switched_from=prev_prim or "direct_hq",
                coordination_tax_pct=measured_tax,
            )

        boost_stats = ledger_stats.get("x-boost")
        direct_stats = ledger_stats.get("direct_hq")
        tax_from_ledger = boost_stats["avg_tax_pct"] if boost_stats else None
        effective_tax = measured_tax if measured_tax is not None else tax_from_ledger
        low_risk = not feats.adversarial_risk and not bool(params.get("high_risk", False))
        direct_viable = direct_stats is None or direct_stats["pass_rate"] >= (
            boost_stats["pass_rate"] if boost_stats else 0.0
        )
        if (
            prev_prim == "x-boost"
            and low_risk
            and effective_tax is not None
            and effective_tax > self.max_coordination_tax_pct
            and direct_viable
        ):
            return WorkflowSelection(
                primitive="direct_hq",
                rationale="DOWNSIZE_BOOST_TO_DIRECT_HIGH_COORDINATION_TAX",
                workload_class=w_class,
                switched_from="x-boost",
                coordination_tax_pct=effective_tax,
                details={"ledger_boost": boost_stats, "ledger_direct": direct_stats},
            )

        if explicit_raw and prev_prim in SUPPORTED_PRIMITIVES:
            return WorkflowSelection(
                primitive=prev_prim,
                rationale="EXPLICIT_HYPOTHESIS_PRIMITIVE",
                workload_class=w_class,
                coordination_tax_pct=effective_tax,
            )

        if readonly:
            return WorkflowSelection(
                primitive="readonly_fanout",
                rationale="READONLY_FANOUT_WORKLOAD",
                workload_class="readonly_fanout",
            )

        if hyp.kind in (
            HypothesisKind.PARAMETER_RACE,
            HypothesisKind.ABLATION,
            HypothesisKind.FALSIFICATION,
        ) and not params.get("requires_llm_code"):
            return WorkflowSelection(
                primitive="direct_hq",
                rationale="DETERMINISTIC_DIRECT_HQ_DEFAULT",
                workload_class=w_class,
            )

        if isinstance(phases, Sequence) and not isinstance(phases, (str, bytes)) and len(phases) >= 1:
            return WorkflowSelection(
                primitive="x-teamwork",
                rationale="MULTI_PHASE_TEAMWORK_CONFIG",
                workload_class="multi_milestone",
            )

        if isinstance(workers, Sequence) and not isinstance(workers, (str, bytes)) and len(workers) >= 2:
            prim = "x-swarm" if not feats.interface_frozen else "direct_hq"
            return WorkflowSelection(
                primitive=prim,
                rationale="MULTI_WORKER_CONFIG",
                workload_class=w_class,
            )

        return WorkflowSelection(
            primitive="direct_hq",
            rationale="DEFAULT_LINEAR_DIRECT_HQ",
            workload_class=w_class,
            coordination_tax_pct=effective_tax,
        )


class WorkflowRuntime:
    """Unified durable reducer for `direct_hq`, `x-boost`, `x-swarm`, `x-teamwork`, and `readonly_fanout`."""

    def __init__(
        self,
        db: StateDB,
        *,
        selector: WorkflowSelector | None = None,
        arbiter: ResourceArbiter | None = None,
        router: RouteResolverProtocol | None = None,
    ) -> None:
        self.db = db
        self.selector = selector or WorkflowSelector()
        self.arbiter = arbiter or ResourceArbiter()
        self.router = router or _try_build_default_router(arbiter=self.arbiter, db=self.db)
        self.plan_manager = PlanLifecycleManager(self.db)

    @staticmethod
    def _has_mandatory_design_review_trigger(
        goal: ResearchGoal,
        hyp: HypothesisNode,
    ) -> bool:
        """Return True when fidelity, risk, holdout, paid GPU, or irreversibility mandates R1 regardless of overrides."""
        params = dict(hyp.params or {})
        if hyp.fidelity in (FidelityRung.F2_FULL_CONFIRM, FidelityRung.F3_MILESTONE_AUDIT):
            return True
        if goal.risk_profile.lower() in ("high", "critical", "red_team", "high_stakes"):
            return True
        if any(
            bool(params.get(k))
            for k in (
                "touches_holdout",
                "requires_paid_gpu",
                "paid_cloud_gpu",
                "unapproved_paid_gpu",
                "unapproved_paid_cloud_hardware",
                "irreversible",
                "require_design_review_strict",
            )
        ):
            return True
        if str(params.get("required_hardware") or "").strip().lower() in ("cloud_gpu", "paid_cloud_gpu", "slurm_cluster"):
            return True
        norm_tags = {str(t).strip().lower() for t in hyp.tags if str(t).strip()}
        for extra_tag in params.get("tags", []) if isinstance(params.get("tags"), list) else []:
            if str(extra_tag).strip():
                norm_tags.add(str(extra_tag).strip().lower())
        hard_r1_tags = {
            "holdout",
            "banned_path",
            "paid_gpu",
            "cloud_gpu",
            "destructive",
            "irreversible",
            "security",
            "milestone",
        }
        return bool(norm_tags & hard_r1_tags)

    def _requires_design_review(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        cfg: Mapping[str, Any],
        review_tier: ReviewTier,
    ) -> bool:
        """Return True when mandatory risk/fidelity triggers or policy requires LLM pre-code R1 `DESIGN_REVIEW`."""
        params = dict(hyp.params or {})
        # Hard risk, fidelity (F2/F3), holdout, paid-GPU, and irreversible checks ALWAYS dominate overrides
        if self._has_mandatory_design_review_trigger(goal, hyp):
            return True
        subgoal = _resolve_subgoal_for_hyp(self.db, hyp)
        if subgoal is not None:
            compute_target = str(getattr(subgoal, "compute_target", "") or "").strip().lower()
            if compute_target in ("cloud_gpu", "colab_gpu", "paid_cloud_gpu", "paid_gpu", "slurm_cluster") and not bool(
                params.get("paid_gpu_approved") or params.get("cloud_gpu_approved") or params.get("slurm_owner_approved")
            ):
                return True
            if bool(subgoal.wave_gated):
                return False
        elif params.get("wave_gated") is True:
            return False
        if "require_design_review" in cfg:
            return bool(cfg["require_design_review"])
        if "require_design_review" in params:
            return bool(params["require_design_review"])
        if is_reversible_f1_exploratory(goal, hyp, cfg):
            return False
        norm_tags = {str(t).strip().lower() for t in hyp.tags if str(t).strip()}
        if norm_tags & HIGH_RISK_REVIEW_TAGS:
            return True
        if "跨家族审查" in (goal.routing_policy_nl or ""):
            return review_tier == ReviewTier.RED_TEAM_TIEBREAKER
        return False

    @staticmethod
    def _is_policy_blocked_error(exc: BaseException) -> bool:
        return exc.__class__.__name__ == "PolicyBlockedError" or hasattr(exc, "waitable")

    @staticmethod
    def _builder_role_for_hypothesis(hyp: HypothesisNode) -> ScientificRole:
        norm_tags = {str(t).strip().lower() for t in hyp.tags if str(t).strip()}
        if norm_tags & {"theory", "derivation", "theorem"} or "理论" in (hyp.statement or ""):
            return ScientificRole.THEORIST
        return ScientificRole.ENGINEER_BUILDER

    def _resolve_role_route(
        self,
        role: ScientificRole,
        *,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        explicit_spec: str | None = None,
        avoid_families: Sequence[str] = (),
        exclude_specs: Sequence[str] = (),
        propagate_policy_error: bool = False,
        workflow_id: str | None = None,
        wf_config: Mapping[str, Any] | None = None,
        task: StageTask | None = None,
    ) -> ResolvedModelRoute | None:
        """Resolve a capability-verified route for `role` outside `avoid_families`."""
        norm_avoid = {str(f).strip().lower() for f in avoid_families if str(f).strip()}
        eff_spec = None if (not explicit_spec or str(explicit_spec).strip().lower() == "auto") else str(explicit_spec)
        if self.router is not None:
            eval_goal = goal
            if norm_avoid and role == ScientificRole.RESEARCH_DIRECTOR and (
                goal.routing_policy or goal.routing_policy_nl
            ):
                # Tie-breaker Director reviewing an artifact (`norm_avoid` non-empty) must not be
                # trapped by `author_only=True` authoring rules (e.g. theory -> Opus xhigh).
                raw_pol = goal.routing_policy
                if not (isinstance(raw_pol, Mapping) and raw_pol.get("rules")) and goal.routing_policy_nl:
                    compile_fn = getattr(self.router, "compile_routing_policy", None)
                    if callable(compile_fn):
                        try:
                            raw_pol = compile_fn(goal.routing_policy_nl, project=goal.project)
                        except Exception:
                            raw_pol = {}
                if isinstance(raw_pol, Mapping) and raw_pol.get("rules"):
                    kept_rules = [
                        dict(r)
                        for r in raw_pol.get("rules", [])
                        if isinstance(r, Mapping)
                        and (
                            not bool(r.get("author_only", False))
                            or ScientificRole.RESEARCH_DIRECTOR.value
                            in {str(x).strip().lower() for x in (r.get("roles") or [])}
                        )
                    ]
                    eval_goal = ResearchGoal.from_dict(
                        {
                            **goal.to_dict(),
                            "routing_policy_nl": "",
                            "routing_policy": (
                                {
                                    **dict(raw_pol),
                                    "source_nl": "",
                                    "routing_policy_nl": "",
                                    "rules": kept_rules,
                                }
                                if kept_rules
                                else {}
                            ),
                        }
                    )
            try:
                import inspect as _inspect_rr

                rr_sig = _inspect_rr.signature(self.router.resolve_route).parameters
                rr_extra: dict[str, Any] = {}
                if "db" in rr_sig and self.db is not None:
                    rr_extra["db"] = self.db
                if "workflow_id" in rr_sig and workflow_id is not None:
                    rr_extra["workflow_id"] = workflow_id
                if "wf_config" in rr_sig and wf_config is not None:
                    rr_extra["wf_config"] = wf_config
                if "task" in rr_sig and task is not None:
                    rr_extra["task"] = task
                return self.router.resolve_route(
                    role,
                    goal=eval_goal,
                    hyp=hyp,
                    explicit_spec=eff_spec,
                    avoid_families=sorted(norm_avoid),
                    exclude_specs=exclude_specs,
                    **rr_extra,
                )
            except Exception as exc:
                if propagate_policy_error and self._is_policy_blocked_error(exc):
                    raise
                return None

        if eff_spec:
            ps = parse_provider_spec(
                eff_spec,
                role="blind_auditor" if "review" in role.value or "auditor" in role.value else "builder",
                avoid_family=next(iter(norm_avoid), None),
                arbiter=self.arbiter,
            )
            fam = ps.family or infer_model_family(f"{ps.pool}:{ps.model}")
            if fam not in norm_avoid:
                pool_l = ps.pool.lower()
                effort = ps.effort or ("medium" if "sonnet" in ps.model.lower() else "high")
                return ResolvedModelRoute(
                    backend_mode=DEFAULT_BACKEND_FOR_POOL.get(pool_l, "dispatch_script"),
                    execution_host="local",
                    multipool_host_arg="local",
                    pool=ps.pool,
                    canonical_model=ps.model,
                    requested_effort=effort,
                    wire_effort=effort,
                    family=fam,
                    logical_pool=POOL_NORMALIZATION.get(pool_l, pool_l),
                    hq_resource=_POOL_TO_HQ_RESOURCE.get(pool_l),
                    strength=ConstraintStrength.SOFT_FALLBACK,
                    matched_rule_id="explicit_spec",
                )

        excluded_l = {str(s).strip().lower() for s in exclude_specs if str(s).strip()}
        for pool, model, effort, fam in _FALLBACK_DISTINCT_CANDIDATES:
            spec_str = f"{pool}:{model}:{effort}"
            if fam in norm_avoid or spec_str.lower() in excluded_l or f"{pool}:{model}".lower() in excluded_l:
                continue
            return ResolvedModelRoute(
                backend_mode=DEFAULT_BACKEND_FOR_POOL.get(pool, "dispatch_script"),
                execution_host="local",
                multipool_host_arg="local",
                pool=pool,
                canonical_model=model,
                requested_effort=effort,
                wire_effort=effort,
                family=fam,
                logical_pool=POOL_NORMALIZATION.get(pool, pool),
                hq_resource=_POOL_TO_HQ_RESOURCE.get(pool),
                strength=ConstraintStrength.SOFT_FALLBACK,
                matched_rule_id="fallback_distinct_family",
            )
        return None

    def _resolve_builder_route(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        explicit_spec: str | None = None,
    ) -> ResolvedModelRoute:
        role = self._builder_role_for_hypothesis(hyp)
        eff_spec = None if (not explicit_spec or str(explicit_spec).strip().lower() == "auto") else str(explicit_spec)
        route = self._resolve_role_route(
            role,
            goal=goal,
            hyp=hyp,
            explicit_spec=eff_spec,
            propagate_policy_error=True,
        )
        if route is not None:
            return route
        ps = parse_provider_spec(eff_spec or "claude:claude-sonnet-5-5", role="builder", arbiter=self.arbiter)
        pool_l = ps.pool.lower()
        effort = ps.effort or ("medium" if "sonnet" in ps.model.lower() else "high")
        return ResolvedModelRoute(
            backend_mode=DEFAULT_BACKEND_FOR_POOL.get(pool_l, "dispatch_script"),
            execution_host="local",
            multipool_host_arg="local",
            pool=ps.pool,
            canonical_model=ps.model,
            requested_effort=effort,
            wire_effort=effort,
            family=ps.family or infer_model_family(f"{ps.pool}:{ps.model}"),
            logical_pool=POOL_NORMALIZATION.get(pool_l, pool_l),
            hq_resource=_POOL_TO_HQ_RESOURCE.get(pool_l),
            strength=ConstraintStrength.SOFT_FALLBACK,
            matched_rule_id="fallback_builder",
        )

    def _resolve_pool_spec(
        self,
        raw_spec: str | None,
        *,
        role: str,
        avoid_family: str | None = None,
    ) -> tuple[str, str]:
        ps = parse_provider_spec(
            raw_spec,
            role=role,
            avoid_family=avoid_family,
            arbiter=self.arbiter,
        )
        return f"{ps.pool}:{ps.model}", ps.family

    def _make_task(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        workflow: WorkflowNode,
        stage: StageKind,
        *,
        step_key: str,
        deps: Sequence[str] = (),
        attempt: int = 1,
        worktree_path: str | None = None,
        resources_override: ResourceSpec | None = None,
        pinned_route: ResolvedModelRoute | Mapping[str, Any] | None = None,
    ) -> StageTask:
        adapter = get_adapter(goal.adapter_name or goal.project)
        if resources_override is not None:
            res = resources_override
        else:
            try:
                res = adapter.default_resource_spec(stage, hyp, goal)
            except Exception:
                res = ResourceSpec(cpus=1, ram_mb=1024, wall_timeout_s=300)

        if stage == StageKind.CPU_GATE:
            exp_plan = self.db.get_latest_experiment_plan(hyp.hypothesis_id)
            if resources_override is not None and res.wall_timeout_s:
                gate_wt = int(res.wall_timeout_s)
            elif exp_plan is not None:
                gate_wt = int(exp_plan.gate_timeout_s)
            elif "gate_timeout_s" in hyp.params:
                gate_wt = int(hyp.params["gate_timeout_s"])
            else:
                gate_wt = int(res.wall_timeout_s or 90)
            gate_cap = min(max(30, gate_wt), 180) if resources_override is None else min(max(1, gate_wt), 180)
            res = ResourceSpec(
                cpus=res.cpus,
                ram_mb=res.ram_mb,
                gpu=0,
                llm_pool=None,
                eda_sim=res.eda_sim,
                cloud_slot=0,
                wall_timeout_s=gate_cap,
            )
        elif stage == StageKind.EXP_SCREEN and resources_override is None:
            exp_plan = self.db.get_latest_experiment_plan(hyp.hypothesis_id)
            if exp_plan is not None or "screen_timeout_s" in hyp.params:
                scr_wt = int(
                    exp_plan.screen_timeout_s
                    if exp_plan is not None
                    else hyp.params.get("screen_timeout_s", res.wall_timeout_s or 300)
                )
                res = ResourceSpec(
                    cpus=res.cpus,
                    ram_mb=res.ram_mb,
                    gpu=res.gpu,
                    llm_pool=res.llm_pool,
                    eda_sim=res.eda_sim,
                    cloud_slot=res.cloud_slot,
                    wall_timeout_s=max(30, scr_wt),
                )
        elif stage == StageKind.EXP_CONFIRM and resources_override is None:
            exp_plan = self.db.get_latest_experiment_plan(hyp.hypothesis_id)
            if exp_plan is not None or "confirm_timeout_s" in hyp.params:
                cnf_wt = int(
                    exp_plan.confirm_timeout_s
                    if exp_plan is not None
                    else hyp.params.get("confirm_timeout_s", res.wall_timeout_s or 1200)
                )
                res = ResourceSpec(
                    cpus=res.cpus,
                    ram_mb=res.ram_mb,
                    gpu=res.gpu,
                    llm_pool=res.llm_pool,
                    eda_sim=res.eda_sim,
                    cloud_slot=res.cloud_slot,
                    wall_timeout_s=max(30, cnf_wt),
                )

        is_deterministic_zero_llm = (
            stage == StageKind.CODE
            and hyp.kind
            in (
                HypothesisKind.PARAMETER_RACE,
                HypothesisKind.ABLATION,
                HypothesisKind.FALSIFICATION,
            )
            and not hyp.params.get("requires_llm_code", False)
            and workflow.primitive == "direct_hq"
            and workflow.parent_workflow_id is None
        )

        if is_deterministic_zero_llm:
            res = ResourceSpec(
                cpus=res.cpus,
                ram_mb=res.ram_mb,
                gpu=res.gpu,
                llm_pool=None,
                eda_sim=res.eda_sim,
                cloud_slot=res.cloud_slot,
                wall_timeout_s=res.wall_timeout_s,
            )

        route_dict: dict[str, Any] | None = None
        if isinstance(pinned_route, ResolvedModelRoute):
            route_dict = pinned_route.to_dict()
        elif isinstance(pinned_route, Mapping):
            route_dict = dict(pinned_route)
        elif stage == StageKind.CODE and not is_deterministic_zero_llm:
            exp_builder_spec = str(
                workflow.config.get("builder_spec") or hyp.params.get("builder_spec") or ""
            ).strip() or None
            resolved_b = self._resolve_builder_route(
                goal, hyp, explicit_spec=exp_builder_spec
            )
            route_dict = resolved_b.to_dict()

        goal_plan_epoch = int(getattr(goal, "plan_epoch", 0) or 0)
        if route_dict is not None:
            route_dict = {**dict(route_dict), "plan_epoch": goal_plan_epoch}
            res = ResourceSpec(
                cpus=res.cpus,
                ram_mb=res.ram_mb,
                gpu=res.gpu,
                llm_pool=str(route_dict.get("pool") or res.llm_pool or ""),
                eda_sim=res.eda_sim,
                cloud_slot=res.cloud_slot,
                wall_timeout_s=res.wall_timeout_s,
                hq_llm_resource=route_dict.get("hq_resource"),
                pinned_route=route_dict,
            )
            if stage == StageKind.CODE and route_dict.get("family"):
                exec_plan = self.db.get_execution_plan_by_workflow(workflow.workflow_id)
                if exec_plan is not None:
                    fam_l = str(route_dict["family"]).strip().lower()
                    if fam_l and fam_l not in exec_plan.contributor_families:
                        exec_plan.contributor_families = sorted(
                            set(exec_plan.contributor_families) | {fam_l}
                        )
                        self.db.save_execution_plan(exec_plan)

        wf_suffix = workflow.workflow_id[-6:]
        step_slug = step_key or stage.value.lower()
        task = StageTask(
            task_id=f"tsk-{hyp.hypothesis_id}-{wf_suffix}-{step_slug}-a{attempt}-{uuid.uuid4().hex[:5]}",
            idempotency_key=f"{goal.goal_id}:{hyp.hypothesis_id}:{workflow.workflow_id}:{step_slug}:pe{goal_plan_epoch}:a{attempt}",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=stage,
            deps=list(deps),
            resources=res,
            attempt=attempt,
            max_attempts=3,
            status="READY",
            worktree_path=worktree_path,
            enqueued_ts=time.time(),
            workflow_id=workflow.workflow_id,
            goal_version=int(goal.version),
            human_epoch=int(goal.human_epoch),
            plan_epoch=goal_plan_epoch,
            workflow_primitive=workflow.primitive,
            step_key=step_slug,
            pinned_route=route_dict,
        )
        return self.db.enqueue_task(task)

    def _contributor_families_for_workflow(
        self,
        workflow: WorkflowNode,
        hyp: HypothesisNode,
        *,
        review_kind: ReviewKind | None = None,
    ) -> list[str]:
        exec_plan = self.db.get_execution_plan_by_workflow(workflow.workflow_id)
        wf_tasks = [
            t
            for t in self.db.list_tasks(goal_id=workflow.goal_id, hypothesis_id=hyp.hypothesis_id)
            if t.workflow_id == workflow.workflow_id or workflow.primitive in ("x-teamwork", "x-swarm")
        ]
        rcpts = self.db.list_receipts(goal_id=workflow.goal_id, hypothesis_id=hyp.hypothesis_id)
        extra: list[str] = []
        proposer_fam = str(hyp.params.get("proposer_family") or "").strip().lower()
        if proposer_fam:
            extra.append(proposer_fam)
        designer_fam = str(hyp.params.get("designer_family") or "").strip().lower()
        if designer_fam:
            extra.append(designer_fam)

        syn_hids = list(workflow.config.get("synthesized_hypothesis_ids") or [])
        for shid in syn_hids:
            sh = self.db.get_goal_hypothesis(workflow.goal_id, str(shid))
            if sh is None:
                continue
            for fk in ("proposer_family", "designer_family", "builder_family"):
                fv = str(sh.params.get(fk) or "").strip().lower()
                if fv:
                    extra.append(fv)
            wf_tasks.extend(self.db.list_tasks(goal_id=workflow.goal_id, hypothesis_id=sh.hypothesis_id))
            rcpts.extend(self.db.list_receipts(goal_id=workflow.goal_id, hypothesis_id=sh.hypothesis_id))

        # Pre-code R1 DESIGN_REVIEW excludes only upstream proposer/designer families (+ any executed tasks).
        # Once CODE builds run, R2/R3 also exclude builder families!
        if review_kind != ReviewKind.DESIGN_REVIEW:
            bf = str(workflow.config.get("builder_family") or "").strip().lower()
            if bf:
                extra.append(bf)
            bp = str(workflow.config.get("builder_pool") or "").strip()
            if bp:
                fam = infer_model_family(bp)
                if fam:
                    extra.append(fam)
            return collect_contributor_families(
                exec_plan=exec_plan,
                tasks=wf_tasks,
                receipts=rcpts,
                extra_families=extra,
            )

        return collect_contributor_families(
            exec_plan=None,
            tasks=wf_tasks,
            receipts=rcpts,
            extra_families=extra,
        )

    def _spawn_review_quorum_tasks(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        workflow: WorkflowNode,
        *,
        review_kind: ReviewKind,
        review_tier: ReviewTier,
        step_prefix: str,
        next_step_on_pass: str,
        head_sha: str,
        deps: Sequence[str] = (),
        worktree_path: str | None = None,
        explicit_reviewer_spec: str | None = None,
        target_version: int = 1,
        completed_steps: Sequence[str] = (),
    ) -> list[StageTask]:
        """Initialize persisted `review_quorum` state and spawn 1-slot or 2-slot RO review tasks."""
        contrib_families = self._contributor_families_for_workflow(
            workflow, hyp, review_kind=review_kind
        )
        stage_map = {
            ReviewKind.DESIGN_REVIEW: StageKind.DESIGN_REVIEW,
            ReviewKind.IMPLEMENTATION_REVIEW: StageKind.IMPL_REVIEW,
            ReviewKind.RESULT_REVIEW: StageKind.RESULT_REVIEW,
            ReviewKind.SYNTHESIS_REVIEW: StageKind.RESULT_REVIEW,
        }
        role_map = {
            ReviewKind.DESIGN_REVIEW: ScientificRole.DESIGN_REVIEWER,
            ReviewKind.IMPLEMENTATION_REVIEW: ScientificRole.IMPL_REVIEWER,
            ReviewKind.RESULT_REVIEW: ScientificRole.RESULT_AUDITOR,
            ReviewKind.SYNTHESIS_REVIEW: ScientificRole.RESULT_AUDITOR,
        }
        stage = stage_map[review_kind]
        role = role_map[review_kind]
        goal_plan_epoch = int(getattr(goal, "plan_epoch", 0) or 0)
        syn_hids = [
            str(x)
            for x in (workflow.config.get("synthesized_hypothesis_ids") or [])
            if str(x).strip()
        ]
        if review_kind == ReviewKind.SYNTHESIS_REVIEW and syn_hids:
            all_syn_fams: set[str] = set(contrib_families)
            for shid in syn_hids:
                sh = self.db.get_hypothesis(shid)
                if sh is not None:
                    for w_other in self.db.list_workflows(goal_id=goal.goal_id, hypothesis_id=shid):
                        all_syn_fams.update(
                            self._contributor_families_for_workflow(w_other, sh, review_kind=review_kind)
                        )
            contrib_families = sorted(all_syn_fams)

        qstate = ReviewQuorumReducer.init_quorum_state(
            review_kind=review_kind,
            review_tier=review_tier,
            target_version=target_version,
            plan_epoch=goal_plan_epoch,
            human_epoch=int(goal.human_epoch),
            target_kind=(
                "goal_synthesis"
                if review_kind == ReviewKind.SYNTHESIS_REVIEW
                else "hypothesis"
            ),
            synthesized_hypothesis_ids=syn_hids,
            head_sha=head_sha,
            contributor_families=contrib_families,
        )
        qstate["step_prefix"] = step_prefix
        qstate["next_step_on_pass"] = next_step_on_pass

        slots = [0, 1] if review_tier == ReviewTier.RED_TEAM_TIEBREAKER else [0]
        chosen_routes: list[tuple[int, ResolvedModelRoute]] = []
        used_families: set[str] = set(contrib_families)
        used_specs: list[str] = []

        for slot_idx in slots:
            route = self._resolve_role_route(
                role,
                goal=goal,
                hyp=hyp,
                explicit_spec=explicit_reviewer_spec if slot_idx == 0 else None,
                avoid_families=sorted(used_families),
                exclude_specs=used_specs,
            )
            if route is None:
                self._block_workflow_on_quorum(
                    goal,
                    hyp,
                    workflow,
                    qstate,
                    reason=f"Unable to schedule distinct non-contributor family for {review_kind.value} slot {slot_idx}",
                    completed_steps=completed_steps,
                )
                return []
            chosen_routes.append((slot_idx, route))
            used_families.add(route.family.lower())
            used_specs.append(route.canonical_spec())

        self.db.update_workflow_state(
            workflow.workflow_id,
            state_patch={
                "step": step_prefix,
                "review_quorum": qstate,
                "completed_steps": list(completed_steps),
            },
        )
        refreshed_wf = self.db.get_workflow(workflow.workflow_id) or workflow
        emitted: list[StageTask] = []
        for slot_idx, route in chosen_routes:
            step_key = (
                step_prefix
                if len(chosen_routes) == 1 and slot_idx == 0
                else f"{step_prefix}_s{slot_idx}"
            )
            peer_fams = [
                r2.family.lower() for s2, r2 in chosen_routes if s2 != slot_idx
            ]
            peer_specs = [
                r2.canonical_spec() for s2, r2 in chosen_routes if s2 != slot_idx
            ]
            route_meta = {
                **route.to_dict(),
                "role": role.value,
                "avoid_families": sorted(set(contrib_families) | set(peer_fams)),
                "exclude_specs": sorted(set(peer_specs)),
                "review_kind": review_kind.value,
                "review_slot": slot_idx,
                "target_version": int(target_version),
                "head_sha": head_sha,
                "ro_snapshot": True,
            }
            emitted.append(
                self._make_task(
                    goal,
                    hyp,
                    refreshed_wf,
                    stage,
                    step_key=step_key,
                    deps=deps,
                    attempt=1,
                    worktree_path=worktree_path,
                    resources_override=ResourceSpec(
                        cpus=1,
                        ram_mb=1024,
                        llm_pool=route.pool,
                        wall_timeout_s=300,
                        hq_llm_resource=route.hq_resource,
                        pinned_route=route_meta,
                    ),
                    pinned_route=route_meta,
                )
            )
        return emitted

    def _block_workflow_on_quorum(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        workflow: WorkflowNode,
        qstate: Mapping[str, Any],
        *,
        reason: str,
        completed_steps: Sequence[str] = (),
    ) -> None:
        self.db.update_workflow_state(
            workflow.workflow_id,
            status="BLOCKED_REVIEW_QUORUM",
            state_patch={
                "step": "blocked_review_quorum",
                "review_quorum": {**dict(qstate), "phase": "BLOCKED", "block_reason": reason},
                "completed_steps": list(completed_steps),
            },
        )
        self.db.transition_hypothesis_status(
            hyp.hypothesis_id,
            hyp.status,
            "BLOCKED_REVIEW_QUORUM",
            failure_domain=FailureDomain.INFRA_TRANSIENT,
        )
        self.db.upsert_knowledge_item(
            goal.goal_id,
            "owner_escalation",
            f"quorum:{workflow.workflow_id}",
            {
                "workflow_id": workflow.workflow_id,
                "hypothesis_id": hyp.hypothesis_id,
                "reason": reason,
                "review_kind": qstate.get("review_kind"),
                "review_tier": qstate.get("review_tier"),
            },
            hypothesis_id=hyp.hypothesis_id,
            goal_version=int(goal.version),
        )

    def sweep_expired_review_quorums(
        self,
        goal_id: str | None = None,
        *,
        now: float | None = None,
    ) -> list[WorkflowNode]:
        """Sweep RUNNING workflows across restarts and transition expired `quorum_deadline_ts` to BLOCKED_REVIEW_QUORUM."""
        ts = time.time() if now is None else float(now)
        blocked: list[WorkflowNode] = []
        for wf in self.db.list_workflows(goal_id=goal_id, status="RUNNING"):
            qstate = wf.state.get("review_quorum")
            if not isinstance(qstate, Mapping):
                continue
            phase = str(qstate.get("phase") or "")
            if phase not in ("PRIMARY", "TIEBREAKER"):
                continue
            deadline = float(qstate.get("quorum_deadline_ts") or 0.0)
            if deadline <= 0.0 or ts < deadline:
                continue
            goal = self.db.get_goal(wf.goal_id)
            hyp = self.db.get_hypothesis(wf.hypothesis_id)
            if goal is None or hyp is None:
                continue
            self._block_workflow_on_quorum(
                goal,
                hyp,
                wf,
                qstate,
                reason=f"Review quorum deadline expired ({ts:.1f} >= {deadline:.1f})",
                completed_steps=list(wf.state.get("completed_steps") or []),
            )
            for t in self.db.list_tasks(goal_id=wf.goal_id, hypothesis_id=wf.hypothesis_id):
                if t.workflow_id == wf.workflow_id and t.status == "READY":
                    self.db.transition_task_status(t.task_id, "READY", "CANCELLED", finished_ts=ts, now=ts)
            updated = self.db.get_workflow(wf.workflow_id)
            if updated is not None:
                blocked.append(updated)
        return blocked

    def _emit_initial_build_tasks(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        wf: WorkflowNode,
        *,
        deps: Sequence[str] = (),
        worktree_path: str | None = None,
    ) -> tuple[WorkflowNode, list[StageTask]]:
        eff_prim = wf.primitive
        builder_pool = str(wf.config.get("builder_pool") or "claude:claude-sonnet-5-5")

        if eff_prim == "x-boost":
            wf = self.db.update_workflow_state(
                wf.workflow_id,
                state_patch={"step": "boost_build", "repairs_used": 0},
            )
            t = self._make_task(
                goal,
                hyp,
                wf,
                StageKind.CODE,
                step_key="boost_build",
                deps=deps,
                worktree_path=worktree_path,
                resources_override=ResourceSpec(
                    cpus=1, ram_mb=2048, llm_pool=builder_pool, wall_timeout_s=900
                ),
            )
            return wf, [t]

        if eff_prim == "x-swarm":
            workers = list(wf.config.get("workers") or [])
            wf = self.db.update_workflow_state(
                wf.workflow_id,
                state_patch={
                    "step": "swarm_workers",
                    "tick": 1,
                    "expected_workers": [w["name"] for w in workers],
                    "completed_workers": [],
                },
            )
            tasks = [
                self._make_task(
                    goal,
                    hyp,
                    wf,
                    StageKind.CODE,
                    step_key=f"swarm_worker:{w['name']}:t1",
                    deps=deps,
                    worktree_path=worktree_path,
                    resources_override=ResourceSpec(
                        cpus=1, ram_mb=2048, llm_pool=w["llm_pool"], wall_timeout_s=600
                    ),
                )
                for w in workers
            ]
            return wf, tasks

        if eff_prim == "x-teamwork":
            return self._spawn_teamwork_phase(
                wf, goal, hyp, phase_idx=1, worktree_path=worktree_path
            )

        wf = self.db.update_workflow_state(wf.workflow_id, state_patch={"step": "code"})
        t = self._make_task(
            goal,
            hyp,
            wf,
            StageKind.CODE,
            step_key="code",
            deps=deps,
            worktree_path=worktree_path,
        )
        return wf, [t]

    def create_workflow_for_hypothesis(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        primitive: str | None = None,
        parent_workflow_id: str | None = None,
        config_override: Mapping[str, Any] | None = None,
        worktree_path: str | None = None,
    ) -> tuple[WorkflowNode, list[StageTask]]:
        """Create a persisted WorkflowNode (+ ExecutionPlan) and emit initial StageTask(s)."""
        now = time.time()
        selection = None
        if primitive is None:
            selection = self.selector.select_primitive(
                goal, hyp, db=self.db, arbiter=self.arbiter
            )
            eff_prim = selection.primitive
        else:
            eff_prim = WorkflowSelector._normalize_primitive(primitive)

        params = dict(hyp.params or {})
        cfg: dict[str, Any] = dict(config_override or {})
        if selection is not None:
            cfg.setdefault("selection", selection.to_dict())

        raw_builder_spec = str(
            cfg.get("builder_spec") or params.get("builder_spec") or ""
        ).strip() or None
        if eff_prim == "readonly_fanout":
            builder_pool, builder_family = self._resolve_pool_spec(
                raw_builder_spec or "auto",
                role="builder",
            )
            builder_route: ResolvedModelRoute | None = None
        else:
            builder_route = self._resolve_builder_route(
                goal, hyp, explicit_spec=raw_builder_spec
            )
            builder_pool = f"{builder_route.pool}:{builder_route.canonical_model}"
            builder_family = builder_route.family
        cfg.setdefault("builder_pool", builder_pool)
        cfg.setdefault("builder_family", builder_family)
        if builder_route is not None:
            cfg.setdefault("builder_route", builder_route.to_dict())

        wf_id = f"wf_{uuid.uuid4().hex[:12]}"
        wf_budget = int(
            cfg.get("budget_tokens")
            or params.get("workflow_budget_tokens")
            or min(goal.max_tokens_budget, 50000)
        )

        if eff_prim == "readonly_fanout":
            wf = WorkflowNode(
                workflow_id=wf_id,
                goal_id=goal.goal_id,
                hypothesis_id=hyp.hypothesis_id,
                parent_workflow_id=parent_workflow_id,
                primitive=eff_prim,
                status="RUNNING",
                goal_version=int(goal.version),
                human_epoch=int(goal.human_epoch),
                plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                budget_tokens=wf_budget,
                config=cfg,
                state={"step": "fanout_inspect", "completed_steps": []},
                created_ts=now,
                updated_ts=now,
            )
            wf = self.db.save_workflow(wf, now=now)
            t = self._make_task(
                goal,
                hyp,
                wf,
                StageKind.PROPOSE,
                step_key="fanout_inspect",
                worktree_path=worktree_path,
                resources_override=ResourceSpec(
                    cpus=1, ram_mb=1024, llm_pool=builder_pool, wall_timeout_s=300
                ),
            )
            return wf, [t]

        review_tier = (
            ReviewTier(str(cfg["review_tier"]).upper())
            if cfg.get("review_tier")
            else classify_review_tier(
                goal,
                hyp,
                diff_lines=int(params.get("estimated_diff_lines", 0)),
                workflow_primitive=eff_prim,
            )
        )
        cfg["review_tier"] = review_tier.value
        cfg.setdefault("max_repairs", int(params.get("max_repairs", 2)))

        slices_meta: list[dict[str, Any]] = []
        collapsed_info: dict[str, Any] | None = None

        def _collapse_to_direct(from_prim: str, reason: str, details: Mapping[str, Any] | None = None) -> None:
            nonlocal eff_prim, slices_meta, collapsed_info
            collapsed_info = {
                "collapsed_from": from_prim,
                "reason": reason,
                **dict(details or {}),
            }
            eff_prim = "direct_hq"
            cfg["collapsed_from"] = from_prim
            cfg["collapse_reason"] = reason
            if details:
                cfg["collapse_details"] = dict(details)
            cfg.pop("workers", None)
            cfg.pop("verifier_pool", None)
            cfg.pop("phases", None)
            cfg.pop("auditor_pool", None)
            slices_meta = []

        def _find_invalid_slice_edges(slice_list: Sequence[Mapping[str, Any]]) -> str | None:
            seen_names: set[str] = set()
            for idx, s in enumerate(slice_list, start=1):
                if not isinstance(s, Mapping):
                    return f"invalid_slice_type_at_{idx}"
                s_name = str(s.get("name") or "").strip()
                if not s_name:
                    return f"missing_slice_name_at_{idx}"
                if s_name in seen_names:
                    return f"duplicate_slice_name:{s_name}"
                seen_names.add(s_name)
                if "writable_paths" in s and (
                    isinstance(s.get("writable_paths"), (str, bytes))
                    or not isinstance(s.get("writable_paths"), (Sequence, set))
                ):
                    return f"invalid_writable_paths:{s_name}"
            for s in slice_list:
                s_name = str(s.get("name") or "")
                deps = s.get("depends_on") or s.get("after") or []
                if isinstance(deps, (str, bytes)):
                    deps = [deps]
                if not isinstance(deps, Sequence):
                    return f"invalid_slice_deps:{s_name}"
                for dep in deps:
                    dep_s = str(dep).strip()
                    if dep_s and (dep_s == s_name or dep_s not in seen_names):
                        return f"invalid_slice_edge:{s_name}->{dep_s}"
            return None

        subgoal = _resolve_subgoal_for_hyp(self.db, hyp)
        if _is_foundation_build_hyp(hyp, subgoal):
            cfg["foundation_build"] = True
            if subgoal is not None:
                cfg.setdefault("subgoal_id", subgoal.subgoal_id)
            if eff_prim != "direct_hq":
                _collapse_to_direct(
                    eff_prim,
                    "FOUNDATION_BUILD_DIRECT_HQ",
                    {"subgoal_id": getattr(subgoal, "subgoal_id", None) or params.get("subgoal_id")},
                )
                if not (config_override and "review_tier" in config_override):
                    review_tier = classify_review_tier(
                        goal,
                        hyp,
                        diff_lines=int(params.get("estimated_diff_lines", 0)),
                        workflow_primitive="direct_hq",
                    )
                    cfg["review_tier"] = review_tier.value

        if eff_prim == "x-boost":
            adv_pool, adv_family = self._resolve_pool_spec(
                str(cfg.get("adversary_spec") or params.get("adversary_spec") or "auto"),
                role="adversary",
                avoid_family=builder_family,
            )
            cfg["adversary_pool"] = adv_pool
            cfg["adversary_family"] = adv_family

        elif eff_prim == "x-swarm":
            raw_workers = cfg.get("workers") or params.get("workers") or params.get("slices")
            workers: list[dict[str, Any]] = []
            if (
                isinstance(raw_workers, Sequence)
                and not isinstance(raw_workers, (str, bytes))
                and len(raw_workers) >= 1
            ):
                invalid_edge = _find_invalid_slice_edges(
                    [w for w in raw_workers if isinstance(w, Mapping)]
                ) if all(isinstance(w, Mapping) for w in raw_workers) else "non_mapping_worker_slice"
                if invalid_edge:
                    _collapse_to_direct("x-swarm", invalid_edge)
                else:
                    for idx, w in enumerate(raw_workers, start=1):
                        w_name = str(w.get("name") or f"w{idx}")
                        w_paths = list(w.get("writable_paths") or hyp.writable_paths)
                        w_pool, _ = self._resolve_pool_spec(w.get("provider"), role="builder")
                        workers.append(
                            {"name": w_name, "writable_paths": w_paths, "llm_pool": w_pool}
                        )
            else:
                paths = list(hyp.writable_paths) or ["."]
                for idx, wp in enumerate(paths, start=1):
                    workers.append(
                        {
                            "name": f"slice{idx}",
                            "writable_paths": [wp],
                            "llm_pool": builder_pool,
                        }
                    )

            if eff_prim == "x-swarm":
                scope_map = {w["name"]: w["writable_paths"] for w in workers}
                if len(scope_map) >= 2:
                    overlaps = gates.detect_sibling_scope_overlaps(scope_map)
                    if overlaps:
                        _collapse_to_direct(
                            "x-swarm",
                            "x-swarm sibling write-scope overlap",
                            {"overlaps": overlaps},
                        )

            if eff_prim == "x-swarm":
                ver_pool, _ = self._resolve_pool_spec(
                    str(cfg.get("verifier_spec") or params.get("verifier_spec") or "auto"),
                    role="adversary",
                    avoid_family=builder_family,
                )
                cfg["workers"] = workers
                cfg["verifier_pool"] = ver_pool
                cfg.setdefault("max_ticks", int(params.get("max_ticks", 4)))
                slices_meta = list(workers)

        elif eff_prim == "x-teamwork":
            raw_phases = cfg.get("phases") or params.get("phases")
            phases: list[dict[str, Any]] = []
            if (
                isinstance(raw_phases, Sequence)
                and not isinstance(raw_phases, (str, bytes))
                and len(raw_phases) >= 1
            ):
                for p_idx, p in enumerate(raw_phases, start=1):
                    if not isinstance(p, Mapping):
                        _collapse_to_direct("x-teamwork", f"invalid_phase_{p_idx}")
                        break
                    p_slices = list(p.get("slices") or [])
                    if not p_slices:
                        _collapse_to_direct("x-teamwork", f"empty_phase_slices_{p_idx}")
                        break
                    invalid_edge = _find_invalid_slice_edges(p_slices)
                    if invalid_edge:
                        _collapse_to_direct("x-teamwork", f"phase_{p_idx}:{invalid_edge}")
                        break
                    scope_map = {
                        str(s["name"]): list(s.get("writable_paths") or hyp.writable_paths)
                        for s in p_slices
                    }
                    if len(scope_map) >= 2:
                        overlaps = gates.detect_sibling_scope_overlaps(scope_map)
                        if overlaps:
                            _collapse_to_direct(
                                "x-teamwork",
                                f"x-teamwork phase {p_idx} write-scope overlap",
                                {"overlaps": overlaps, "phase": p_idx},
                            )
                            break
                    phases.append(
                        {
                            "name": str(p.get("name") or f"phase_{p_idx}"),
                            "slices": p_slices,
                        }
                    )
                    slices_meta.extend(p_slices)
            else:
                paths = list(hyp.writable_paths) or ["."]
                slices = [
                    {
                        "name": f"slice_{idx}",
                        "writable_paths": [wp],
                        "primitive": "direct_hq",
                    }
                    for idx, wp in enumerate(paths, start=1)
                ]
                phases = [{"name": "phase_1", "slices": slices}]
                slices_meta.extend(slices)
                scope_map = {s["name"]: s["writable_paths"] for s in slices}
                if len(scope_map) >= 2:
                    overlaps = gates.detect_sibling_scope_overlaps(scope_map)
                    if overlaps:
                        _collapse_to_direct(
                            "x-teamwork",
                            "x-teamwork write-scope overlap",
                            {"overlaps": overlaps},
                        )

            if eff_prim == "x-teamwork":
                aud_pool, _ = self._resolve_pool_spec(
                    str(cfg.get("auditor_spec") or params.get("auditor_spec") or "auto"),
                    role="blind_auditor",
                    avoid_family=builder_family,
                )
                cfg["phases"] = phases
                cfg["auditor_pool"] = aud_pool

        init_wf_state: dict[str, Any] = {
            "step": "init",
            "repairs_used": 0,
            "completed_steps": [],
        }
        if collapsed_info is not None:
            init_wf_state["collapsed_from"] = collapsed_info["collapsed_from"]
            init_wf_state["collapse_reason"] = collapsed_info["reason"]

        wf = WorkflowNode(
            workflow_id=wf_id,
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            parent_workflow_id=parent_workflow_id,
            primitive=eff_prim,
            status="RUNNING",
            goal_version=int(goal.version),
            human_epoch=int(goal.human_epoch),
            plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
            budget_tokens=wf_budget,
            config=cfg,
            state=init_wf_state,
            created_ts=now,
            updated_ts=now,
        )
        wf = self.db.save_workflow(wf, now=now)
        if collapsed_info is not None:
            with self.db._tx() as (conn, pending):
                self.db._record_event(
                    conn,
                    pending,
                    "WORKFLOW_COLLAPSED_TO_DIRECT_HQ",
                    goal_id=goal.goal_id,
                    hypothesis_id=hyp.hypothesis_id,
                    payload={"workflow_id": wf.workflow_id, **collapsed_info},
                )
            self.db.upsert_knowledge_item(
                goal.goal_id,
                "workflow_collapse",
                f"collapse:{hyp.hypothesis_id}:{wf.workflow_id}",
                {"workflow_id": wf.workflow_id, **collapsed_info},
                hypothesis_id=hyp.hypothesis_id,
                goal_version=int(goal.version),
            )

        if parent_workflow_id is None:
            proposer_fam = str(params.get("proposer_family") or "").strip().lower()
            designer_fam = str(params.get("designer_family") or "").strip().lower()
            init_contribs = [f for f in (proposer_fam, designer_fam) if f]
            self.plan_manager.create_execution_plan(
                goal,
                hyp,
                workflow_id=wf.workflow_id,
                primitive=eff_prim,
                review_tier=review_tier,
                slices=slices_meta,
                role_routes={},
                contributor_families=init_contribs,
                budget_tokens=wf_budget,
            )
            mandatory_r1 = self._has_mandatory_design_review_trigger(goal, hyp)
            if self._requires_design_review(goal, hyp, cfg, review_tier) and (
                mandatory_r1 or review_tier != ReviewTier.FAST_LANE
            ):
                eff_r1_tier = (
                    ReviewTier.STANDARD_CROSS_FAMILY
                    if (mandatory_r1 and review_tier == ReviewTier.FAST_LANE)
                    else review_tier
                )
                exp_plan = self.db.get_latest_experiment_plan(hyp.hypothesis_id)
                if exp_plan is None:
                    exp_plan = self.plan_manager.build_default_experiment_plan(goal, hyp)
                    exp_plan = self.db.save_experiment_plan(exp_plan)
                dr_tasks = self._spawn_review_quorum_tasks(
                    goal,
                    hyp,
                    wf,
                    review_kind=ReviewKind.DESIGN_REVIEW,
                    review_tier=eff_r1_tier,
                    step_prefix="design_review",
                    next_step_on_pass="build_entry",
                    head_sha=hyp.head_sha or hyp.base_sha,
                    worktree_path=worktree_path,
                    target_version=int(exp_plan.plan_version),
                )
                refreshed = self.db.get_workflow(wf.workflow_id) or wf
                return refreshed, dr_tasks

            # 0-LLM deterministic R1 FAST_LANE contract check when LLM DESIGN_REVIEW is skipped
            exp_plan = self.db.get_latest_experiment_plan(hyp.hypothesis_id)
            if exp_plan is None:
                exp_plan = self.plan_manager.build_default_experiment_plan(goal, hyp)
            if (
                subgoal is not None
                and subgoal.local_metric
                and isinstance(exp_plan.control_baseline, Mapping)
                and not str(exp_plan.control_baseline.get("primary_metric") or "").strip()
            ):
                exp_plan.control_baseline = {
                    **dict(exp_plan.control_baseline),
                    "primary_metric": subgoal.local_metric,
                    "baseline_value": float(subgoal.baseline_value),
                }
            det_gate = validate_r1_fast_lane_contract(
                exp_plan,
                goal,
                hyp,
                repo_path=goal.repo_path or None,
            )
            if not det_gate.passed:
                self.db.update_workflow_state(
                    wf.workflow_id,
                    status="DESIGN_REJECTED",
                    state_patch={
                        "step": "design_rejected",
                        "design_review_mode": "FAST_LANE_0_LLM",
                        "hard_design_blockers": list(det_gate.failures),
                    },
                )
                self.db.transition_hypothesis_status(
                    hyp.hypothesis_id,
                    hyp.status,
                    "NEEDS_PLAN_REVISION",
                    failure_domain=FailureDomain.ENGINEERING_DEFECT,
                )
                refreshed = self.db.get_workflow(wf.workflow_id) or wf
                return refreshed, []
            cfg["design_review_mode"] = "FAST_LANE_0_LLM"
            wf = self.db.update_workflow_state(
                wf.workflow_id,
                config_patch={"design_review_mode": "FAST_LANE_0_LLM"},
                state_patch={"design_review_tier": ReviewTier.FAST_LANE.value},
            ) or wf
            if exp_plan.plan_version > 0 and exp_plan.review_status != "APPROVED":
                exp_plan.review_status = "APPROVED"
                self.db.save_experiment_plan(exp_plan)

        return self._emit_initial_build_tasks(goal, hyp, wf, worktree_path=worktree_path)

    def switch_hypothesis_workflow(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        new_primitive: str,
        reason: str = "",
        config_override: Mapping[str, Any] | None = None,
        worktree_path: str | None = None,
    ) -> tuple[WorkflowNode, list[StageTask]]:
        """Switch an active/stalled hypothesis onto a new workflow primitive cleanly."""
        now = time.time()
        norm_prim = WorkflowSelector._normalize_primitive(new_primitive)
        for old_wf in self.db.list_workflows(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id):
            if old_wf.status not in TERMINAL_WORKFLOW_STATUSES:
                self.db.update_workflow_state(
                    old_wf.workflow_id,
                    status="CANCELLED",
                    state_patch={"cancel_reason": f"switched_to_{norm_prim}: {reason}"},
                    now=now,
                )
        for task in self.db.list_tasks(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id):
            if task.status == "READY":
                self.db.transition_task_status(
                    task.task_id, "READY", "CANCELLED", finished_ts=now, now=now
                )
        hyp = self.db.update_hypothesis_params(
            hyp.hypothesis_id,
            {"workflow_primitive": norm_prim, "switched_reason": reason},
            now=now,
        )
        return self.create_workflow_for_hypothesis(
            goal,
            hyp,
            primitive=norm_prim,
            config_override=config_override,
            worktree_path=worktree_path,
        )

    def enqueue_bounded_repair(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        repair_notes: str = "",
        worktree_path: str | None = None,
    ) -> list[StageTask]:
        """Attach a bounded `StageKind.CODE` repair task onto a hypothesis's running `WorkflowNode`."""
        now = time.time()
        if repair_notes:
            hyp = self.db.update_hypothesis_params(
                hyp.hypothesis_id,
                {"last_repair_notes": repair_notes, "requires_llm_code": True},
                now=now,
            )
        elif not hyp.params.get("requires_llm_code"):
            hyp = self.db.update_hypothesis_params(
                hyp.hypothesis_id,
                {"requires_llm_code": True},
                now=now,
            )

        top_wfs = [
            w
            for w in self.db.list_workflows(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id)
            if w.parent_workflow_id is None and w.primitive != "readonly_fanout"
        ]
        target_wf = next(
            (w for w in top_wfs if w.status == "RUNNING"),
            top_wfs[-1] if top_wfs else None,
        )
        if target_wf is None:
            _wf, tasks = self.create_workflow_for_hypothesis(
                goal,
                hyp,
                config_override={"require_design_review": False, "repair_notes": repair_notes},
                worktree_path=worktree_path,
            )
            return tasks

        builder_route = self._resolve_builder_route(
            goal,
            hyp,
            explicit_spec=str(
                hyp.params.get("builder_spec") or target_wf.config.get("builder_spec") or ""
            ).strip()
            or None,
        )
        builder_role = (
            ScientificRole.THEORIST
            if "theory" in {str(t).lower() for t in hyp.tags}
            else ScientificRole.ENGINEER_BUILDER
        )
        builder_route_meta = {
            **builder_route.to_dict(),
            "role": builder_role.value,
        }
        repairs_used = int(target_wf.state.get("repairs_used", 0)) + 1
        eff_wt = worktree_path or target_wf.state.get("staging_worktree_path")

        if target_wf.primitive == "x-boost":
            step_key = f"boost_repair_{repairs_used}"
        elif target_wf.primitive == "x-swarm":
            next_tick = int(target_wf.state.get("tick", 1)) + 1
            workers_cfg = list(target_wf.config.get("workers") or [])
            target_wf = self.db.update_workflow_state(
                target_wf.workflow_id,
                status="RUNNING",
                state_patch={
                    "step": "swarm_workers",
                    "tick": next_tick,
                    "repairs_used": repairs_used,
                    "last_repair_notes": repair_notes,
                    "completed_workers": [],
                },
                config_patch={
                    "builder_pool": f"{builder_route.pool}:{builder_route.canonical_model}",
                    "builder_family": builder_route.family,
                    "builder_route": builder_route_meta,
                },
                now=now,
            )
            return [
                self._make_task(
                    goal,
                    hyp,
                    target_wf,
                    StageKind.CODE,
                    step_key=f"swarm_worker:{w['name']}:t{next_tick}",
                    attempt=next_tick,
                    worktree_path=eff_wt,
                    resources_override=ResourceSpec(
                        cpus=1,
                        ram_mb=2048,
                        llm_pool=builder_route.pool,
                        wall_timeout_s=600,
                    ),
                    pinned_route=builder_route_meta,
                )
                for w in workers_cfg
            ]
        else:
            step_key = f"code_repair_{repairs_used}"

        target_wf = self.db.update_workflow_state(
            target_wf.workflow_id,
            status="RUNNING",
            state_patch={
                "step": step_key,
                "repairs_used": repairs_used,
                "last_repair_notes": repair_notes,
            },
            config_patch={
                "builder_pool": f"{builder_route.pool}:{builder_route.canonical_model}",
                "builder_family": builder_route.family,
                "builder_route": builder_route_meta,
            },
            now=now,
        )
        existing_step_attempts = [
            int(t.attempt)
            for t in self.db.list_tasks(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id)
            if t.workflow_id == target_wf.workflow_id and t.step_key == step_key
        ]
        eff_attempt = max([repairs_used + 1, *(a + 1 for a in existing_step_attempts)])
        repair_task = self._make_task(
            goal,
            hyp,
            target_wf,
            StageKind.CODE,
            step_key=step_key,
            attempt=eff_attempt,
            worktree_path=eff_wt,
            resources_override=ResourceSpec(
                cpus=1,
                ram_mb=2048,
                llm_pool=builder_route.pool,
                wall_timeout_s=900,
            ),
            pinned_route=builder_route_meta,
        )
        return [repair_task]

    def _spawn_teamwork_phase(
        self,
        parent_wf: WorkflowNode,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        phase_idx: int,
        worktree_path: str | None = None,
    ) -> tuple[WorkflowNode, list[StageTask]]:
        phases = list(parent_wf.config.get("phases") or [])
        phase = phases[phase_idx - 1]
        slices = list(phase.get("slices") or [])
        anchor = str(parent_wf.state.get("milestone_anchor") or "")

        child_ids: list[str] = []
        emitted: list[StageTask] = []
        for s in slices:
            s_name = str(s.get("name") or f"s{len(child_ids) + 1}")
            s_prim = WorkflowSelector._normalize_primitive(str(s.get("primitive") or "direct_hq"))
            if s_prim not in ("direct_hq", "x-boost"):
                s_prim = "direct_hq"
            child_cfg = {
                "slice_name": s_name,
                "writable_paths": list(s.get("writable_paths") or hyp.writable_paths),
                "spec": str(s.get("spec") or hyp.statement),
                "builder_spec": s.get("provider"),
                "adversary_spec": s.get("adversary_spec"),
                "milestone_anchor": anchor[:MAX_MILESTONE_ANCHOR_BYTES],
                "require_design_review": False,
            }
            child_wf, child_tasks = self.create_workflow_for_hypothesis(
                goal,
                hyp,
                primitive=s_prim,
                parent_workflow_id=parent_wf.workflow_id,
                config_override=child_cfg,
                worktree_path=worktree_path,
            )
            child_ids.append(child_wf.workflow_id)
            emitted.extend(child_tasks)

        updated_wf = self.db.update_workflow_state(
            parent_wf.workflow_id,
            status="RUNNING",
            state_patch={
                "step": f"teamwork_phase_{phase_idx}",
                "phase_idx": phase_idx,
                "phase_children": child_ids,
                "completed_children": [],
                "failed_children": [],
                "slice_results": [],
            },
        )
        return updated_wf, emitted

    def _reduce_review_step(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        fresh_wf: WorkflowNode,
        completed_task: StageTask,
        receipt: EvidenceReceipt,
        *,
        review_kind: ReviewKind,
        completed_steps: list[str],
    ) -> tuple[str, list[StageTask]]:
        """Record `ReviewReceipt` and run `ReviewQuorumReducer` (`PASS|FAIL|WAIT|RETRY|TIEBREAK|BLOCKED`)."""
        qstate = dict(fresh_wf.state.get("review_quorum") or {})
        tier = ReviewTier(
            str(
                qstate.get("review_tier")
                or fresh_wf.config.get("review_tier")
                or ReviewTier.STANDARD_CROSS_FAMILY.value
            )
        )
        route_dict = dict(completed_task.pinned_route or {})
        if not route_dict.get("family"):
            route_dict["family"] = infer_model_family(
                receipt.model_used or completed_task.model_spec or "codex:gpt-5-codex"
            )
        slot_idx = int(route_dict.get("review_slot", 0))
        target_ver = int(qstate.get("target_version", route_dict.get("target_version", 1)))
        eff_plan_epoch = int(
            qstate.get(
                "plan_epoch",
                route_dict.get("plan_epoch", getattr(goal, "plan_epoch", 0)),
            )
            or 0
        )
        eff_human_epoch = int(
            qstate.get("human_epoch", getattr(goal, "human_epoch", 0)) or 0
        )
        eff_target_kind = str(
            qstate.get(
                "target_kind",
                "goal_synthesis" if review_kind == ReviewKind.SYNTHESIS_REVIEW else "hypothesis",
            )
        )
        syn_hids = [
            str(x)
            for x in (
                qstate.get("synthesized_hypothesis_ids")
                or fresh_wf.config.get("synthesized_hypothesis_ids")
                or []
            )
            if str(x).strip()
        ]
        head_sha = str(qstate.get("head_sha") or receipt.head_sha or hyp.head_sha or hyp.base_sha)

        sev = str((receipt.metrics or {}).get("severity", "")).upper()
        repro_rc = (receipt.metrics or {}).get("repro_rc")
        eff_verdict = receipt.audit_verdict
        if (
            review_kind == ReviewKind.IMPLEMENTATION_REVIEW
            and fresh_wf.primitive == "x-boost"
            and receipt.process_status == ProcessStatus.EXIT_OK
            and (sev == "MINOR" or repro_rc == 0)
        ):
            eff_verdict = AuditVerdict.PASS

        contrib_fams = list(qstate.get("contributor_families") or []) or self._contributor_families_for_workflow(
            fresh_wf, hyp, review_kind=review_kind
        )
        rev_rcpt = ReviewReceipt(
            review_id=f"rev_{completed_task.task_id}",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            workflow_id=fresh_wf.workflow_id,
            task_id=completed_task.task_id,
            review_kind=review_kind,
            review_tier=tier,
            target_version=target_ver,
            plan_epoch=eff_plan_epoch,
            human_epoch=eff_human_epoch,
            target_kind=eff_target_kind,
            synthesized_hypothesis_ids=syn_hids,
            head_sha=head_sha,
            slot_index=slot_idx,
            author_families=contrib_fams,
            reviewer_route=route_dict,
            verdict=eff_verdict,
            process_status=receipt.process_status,
            severity=sev or ("NONE" if eff_verdict == AuditVerdict.PASS else "MAJOR"),
            findings=[receipt.failure_reason] if receipt.failure_reason else [],
            tiebreaker_used=bool(slot_idx == 2),
            raw_artifact_paths=list(receipt.raw_artifact_paths),
            in_tokens=int(receipt.in_tokens),
            out_tokens=int(receipt.out_tokens),
            created_ts=receipt.created_ts or time.time(),
        )
        self.db.save_review_receipt(rev_rcpt)

        if not qstate:
            # Direct single-slot evaluation when quorum_state was not pre-initialized
            qstate = ReviewQuorumReducer.init_quorum_state(
                review_kind=review_kind,
                review_tier=tier,
                target_version=target_ver,
                plan_epoch=eff_plan_epoch,
                human_epoch=eff_human_epoch,
                target_kind=eff_target_kind,
                synthesized_hypothesis_ids=syn_hids,
                head_sha=head_sha,
                contributor_families=rev_rcpt.author_families,
            )

        all_revs = self.db.list_review_receipts(
            workflow_id=fresh_wf.workflow_id,
            review_kind=review_kind,
            target_version=target_ver,
            plan_epoch=eff_plan_epoch,
            human_epoch=eff_human_epoch,
            head_sha=head_sha,
        )

        # Check whether a Director tie-breaker route is schedulable outside contributor_families
        tb_route = self._resolve_role_route(
            ScientificRole.RESEARCH_DIRECTOR,
            goal=goal,
            hyp=hyp,
            avoid_families=qstate.get("contributor_families", []),
        )
        qdec = ReviewQuorumReducer.evaluate_quorum(
            qstate,
            all_revs,
            tiebreaker_schedulable=(tb_route is not None),
        )

        step_prefix = str(qstate.get("step_prefix") or completed_task.step_key or review_kind.value.lower())

        if qdec.status == "WAIT_SLOTS":
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                state_patch={"review_quorum": qdec.updated_state, "completed_steps": completed_steps},
            )
            return "WAIT", []

        if qdec.status == "CANCELLED":
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                status="CANCELLED",
                state_patch={"review_quorum": qdec.updated_state, "completed_steps": completed_steps},
            )
            return "CANCELLED", []

        review_dep_ids = (
            []
            if (self.db.get_task(completed_task.task_id) or completed_task).status == "FAILED"
            else [completed_task.task_id]
        )

        if qdec.status == "NEED_TIEBREAKER":
            assert tb_route is not None
            pref_route = (
                self._resolve_role_route(
                    ScientificRole.RESEARCH_DIRECTOR,
                    goal=goal,
                    hyp=hyp,
                    avoid_families=qdec.prefer_avoid_families,
                )
                or tb_route
            )
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                state_patch={
                    "step": f"{step_prefix}_tiebreak",
                    "review_quorum": qdec.updated_state,
                    "completed_steps": completed_steps,
                },
            )
            tb_meta = {
                **pref_route.to_dict(),
                "role": ScientificRole.RESEARCH_DIRECTOR.value,
                "avoid_families": sorted(set(qdec.avoid_families)),
                "prefer_avoid_families": sorted(set(qdec.prefer_avoid_families)),
                "exclude_specs": [],
                "review_kind": review_kind.value,
                "review_slot": 2,
                "target_version": target_ver,
                "head_sha": head_sha,
                "ro_snapshot": True,
                "tiebreaker": True,
            }
            tb_task = self._make_task(
                goal,
                hyp,
                fresh_wf,
                completed_task.stage,
                step_key=f"{step_prefix}_s2",
                deps=review_dep_ids,
                attempt=1,
                worktree_path=completed_task.worktree_path,
                resources_override=ResourceSpec(
                    cpus=1,
                    ram_mb=1024,
                    llm_pool=pref_route.pool,
                    wall_timeout_s=300,
                    hq_llm_resource=pref_route.hq_resource,
                    pinned_route=tb_meta,
                ),
                pinned_route=tb_meta,
            )
            return "SPAWNED", [tb_task]

        if qdec.status == "NEED_RETRY_SLOTS":
            retry_role_map = {
                ReviewKind.DESIGN_REVIEW: ScientificRole.DESIGN_REVIEWER,
                ReviewKind.IMPLEMENTATION_REVIEW: ScientificRole.IMPL_REVIEWER,
                ReviewKind.RESULT_REVIEW: ScientificRole.RESULT_AUDITOR,
            }
            retry_role = retry_role_map[review_kind]
            retry_tasks: list[StageTask] = []
            used_fams = set(qdec.prefer_avoid_families)
            chosen_retries: list[tuple[int, ResolvedModelRoute]] = []
            for s_idx in qdec.slots_to_spawn:
                r_route = self._resolve_role_route(
                    retry_role,
                    goal=goal,
                    hyp=hyp,
                    avoid_families=sorted(used_fams),
                ) or self._resolve_role_route(
                    retry_role,
                    goal=goal,
                    hyp=hyp,
                    avoid_families=qdec.avoid_families,
                )
                if r_route is None:
                    self._block_workflow_on_quorum(
                        goal,
                        hyp,
                        fresh_wf,
                        qdec.updated_state,
                        reason="Retry reviewer route unschedulable outside contributor_families",
                        completed_steps=completed_steps,
                    )
                    return "BLOCKED", []
                used_fams.add(r_route.family.lower())
                chosen_retries.append((s_idx, r_route))
            for s_idx, r_route in chosen_retries:
                peer_fams = [r2.family.lower() for idx2, r2 in chosen_retries if idx2 != s_idx]
                peer_specs = [r2.canonical_spec() for idx2, r2 in chosen_retries if idx2 != s_idx]
                r_meta = {
                    **r_route.to_dict(),
                    "role": retry_role.value,
                    "avoid_families": sorted(set(qdec.avoid_families) | set(peer_fams)),
                    "exclude_specs": sorted(set(peer_specs)),
                    "review_kind": review_kind.value,
                    "review_slot": s_idx,
                    "target_version": target_ver,
                    "head_sha": head_sha,
                    "ro_snapshot": True,
                }
                retry_tasks.append(
                    self._make_task(
                        goal,
                        hyp,
                        fresh_wf,
                        completed_task.stage,
                        step_key=f"{step_prefix}_s{s_idx}",
                        deps=review_dep_ids,
                        attempt=2,
                        worktree_path=completed_task.worktree_path,
                        resources_override=ResourceSpec(
                            cpus=1,
                            ram_mb=1024,
                            llm_pool=r_route.pool,
                            wall_timeout_s=300,
                            hq_llm_resource=r_route.hq_resource,
                            pinned_route=r_meta,
                        ),
                        pinned_route=r_meta,
                    )
                )
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                state_patch={"review_quorum": qdec.updated_state, "completed_steps": completed_steps},
            )
            return "SPAWNED", retry_tasks

        if qdec.status == "BLOCKED_REVIEW_QUORUM":
            self._block_workflow_on_quorum(
                goal,
                hyp,
                fresh_wf,
                qdec.updated_state,
                reason=qdec.reason or "Review quorum blocked",
                completed_steps=completed_steps,
            )
            return "BLOCKED", []

        self.db.update_workflow_state(
            fresh_wf.workflow_id,
            state_patch={"review_quorum": qdec.updated_state, "completed_steps": completed_steps},
        )
        return qdec.status, []

    def _handle_empirical_screen_and_confirm(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        fresh_wf: WorkflowNode,
        completed_task: StageTask,
        receipt: EvidenceReceipt,
        *,
        screen_key: str,
        confirm_key: str,
        result_review_prefix: str,
        reviewer_spec: str | None,
        completed_steps: list[str],
    ) -> list[StageTask]:
        """Unified empirical `EXP_SCREEN -> EXP_CONFIRM -> RESULT_REVIEW -> COMPLETED` tail."""
        step_id = completed_task.step_key or completed_task.stage.value.lower()
        review_tier = ReviewTier(
            str(fresh_wf.config.get("review_tier") or ReviewTier.STANDARD_CROSS_FAMILY.value)
        )
        eff_domain = (
            receipt.failure_domain
            if receipt.failure_domain != FailureDomain.NONE
            else classify_failure_domain(receipt, goal=goal)
        )

        # Non-scientific env / infra failures pause cleanly without graveyarding!
        if eff_domain in (FailureDomain.ENV_CONFIG_MISMATCH, FailureDomain.INFRA_TRANSIENT):
            wf_state_name = (
                "BLOCKED_ENV"
                if eff_domain == FailureDomain.ENV_CONFIG_MISMATCH
                else "BLOCKED_INFRA"
            )
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                status="FAILED",
                state_patch={
                    "step": wf_state_name.lower(),
                    "failure_domain": eff_domain.value,
                    "completed_steps": completed_steps,
                },
            )
            return []

        exec_failed = (
            receipt.process_status != ProcessStatus.EXIT_OK
            or receipt.engineering_verdict in (EngineeringVerdict.FAIL_GATE, EngineeringVerdict.FAIL_TEST)
        )

        if step_id == screen_key or completed_task.stage == StageKind.EXP_SCREEN:
            if is_valid_empirical_outcome(receipt, goal) and receipt.scientific_verdict == ScientificVerdict.VALID_NEGATIVE:
                self.db.update_workflow_state(
                    fresh_wf.workflow_id,
                    status="COMPLETED_VALID_NEGATIVE",
                    state_patch={"step": "completed_valid_negative", "completed_steps": completed_steps},
                )
                return []
            if exec_failed or receipt.scientific_verdict in (
                ScientificVerdict.REFUTED,
                ScientificVerdict.VALID_NEGATIVE,
            ):
                self.db.update_workflow_state(
                    fresh_wf.workflow_id,
                    status="FAILED",
                    state_patch={"step": "exp_screen_failed", "completed_steps": completed_steps},
                )
                return []
            if receipt.scientific_verdict == ScientificVerdict.CONFIRMED and hyp.kind not in (
                HypothesisKind.ABLATION,
                HypothesisKind.FALSIFICATION,
            ):
                if (
                    bool(fresh_wf.config.get("foundation_build"))
                    and hyp.fidelity in (FidelityRung.F0_SANITY, FidelityRung.F1_FAST_SCREEN)
                    and not bool((hyp.params or {}).get("require_confirm"))
                ):
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="COMPLETED",
                        state_patch={"step": "completed", "completed_steps": completed_steps},
                    )
                    return []
                self.db.update_workflow_state(
                    fresh_wf.workflow_id,
                    state_patch={"step": confirm_key, "completed_steps": completed_steps},
                )
                return [
                    self._make_task(
                        goal,
                        hyp,
                        fresh_wf,
                        StageKind.EXP_CONFIRM,
                        step_key=confirm_key,
                        deps=[completed_task.task_id],
                        worktree_path=completed_task.worktree_path,
                    )
                ]
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                status="COMPLETED",
                state_patch={"step": "completed", "completed_steps": completed_steps},
            )
            return []

        # EXP_CONFIRM completed
        if is_valid_empirical_outcome(receipt, goal) and receipt.scientific_verdict == ScientificVerdict.VALID_NEGATIVE:
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                status="COMPLETED_VALID_NEGATIVE",
                state_patch={"step": "completed_valid_negative", "completed_steps": completed_steps},
            )
            return []
        if exec_failed or receipt.scientific_verdict != ScientificVerdict.CONFIRMED:
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                status="FAILED",
                state_patch={"step": "exp_confirm_failed", "completed_steps": completed_steps},
            )
            return []

        if review_tier == ReviewTier.FAST_LANE:
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                status="COMPLETED",
                state_patch={"step": "completed", "completed_steps": completed_steps},
            )
            return []

        return self._spawn_review_quorum_tasks(
            goal,
            hyp,
            fresh_wf,
            review_kind=ReviewKind.RESULT_REVIEW,
            review_tier=review_tier,
            step_prefix=result_review_prefix,
            next_step_on_pass="completed",
            head_sha=receipt.head_sha or hyp.head_sha or hyp.base_sha,
            deps=[completed_task.task_id],
            worktree_path=completed_task.worktree_path,
            explicit_reviewer_spec=reviewer_spec,
            target_version=int(goal.version),
            completed_steps=completed_steps,
        )

    def advance_workflow(
        self,
        workflow: WorkflowNode,
        completed_task: StageTask,
        receipt: EvidenceReceipt,
        goal: ResearchGoal,
    ) -> list[StageTask]:
        """Single durable state-transition reducer across all 5 primitives and 3 review gates."""
        fresh_wf = self.db.get_workflow(workflow.workflow_id) or workflow
        if fresh_wf.status in TERMINAL_WORKFLOW_STATUSES:
            return []

        if int(fresh_wf.goal_version) != int(goal.version):
            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                status="STALE",
                state_patch={
                    "stale_reason": f"goal_version {fresh_wf.goal_version} != {goal.version}"
                },
            )
            return []

        hyp = self.db.get_hypothesis(fresh_wf.hypothesis_id)
        if hyp is None:
            self.db.update_workflow_state(fresh_wf.workflow_id, status="FAILED")
            return []

        completed_steps = list(fresh_wf.state.get("completed_steps") or [])
        step_id = completed_task.step_key or completed_task.stage.value.lower()
        if step_id not in completed_steps:
            completed_steps.append(step_id)

        exec_failed = (
            receipt.process_status != ProcessStatus.EXIT_OK
            or receipt.engineering_verdict in (EngineeringVerdict.FAIL_GATE, EngineeringVerdict.FAIL_TEST)
        )
        review_tier = ReviewTier(
            str(fresh_wf.config.get("review_tier") or ReviewTier.STANDARD_CROSS_FAMILY.value)
        )

        emitted: list[StageTask] = []
        prim = fresh_wf.primitive

        # -------- Cross-Experiment S3 Gate: SYNTHESIS_REVIEW (`target_kind == "goal_synthesis"`)
        if (
            step_id.startswith("synthesis_review")
            or str((completed_task.pinned_route or {}).get("review_kind") or "") == ReviewKind.SYNTHESIS_REVIEW.value
            or str(fresh_wf.config.get("target_kind") or "") == "goal_synthesis"
        ):
            outcome, q_tasks = self._reduce_review_step(
                goal,
                hyp,
                fresh_wf,
                completed_task,
                receipt,
                review_kind=ReviewKind.SYNTHESIS_REVIEW,
                completed_steps=completed_steps,
            )
            if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                return q_tasks
            syn_key = str(fresh_wf.config.get("synthesis_key") or f"syn:{fresh_wf.workflow_id}")
            if outcome == "PASS":
                syn_hids = [
                    str(x).strip()
                    for x in (fresh_wf.config.get("synthesized_hypothesis_ids") or [hyp.hypothesis_id])
                    if str(x).strip()
                ]
                expected_snap = fresh_wf.config.get("source_lineage_snapshot")
                missing_snapshot = not (isinstance(expected_snap, Mapping) and bool(expected_snap))
                stale_sources: list[str] = []
                if missing_snapshot:
                    stale_sources.append("missing_source_lineage_snapshot")
                else:
                    assert isinstance(expected_snap, Mapping)
                    try:
                        current_snap = self._compute_synthesis_source_snapshot(goal, syn_hids)
                    except ValueError as exc:
                        stale_sources.append(str(exc))
                        current_snap = {}
                    required_snap_keys = (
                        "status",
                        "head_sha",
                        "lineage_plan_epoch",
                        "lineage_human_epoch",
                        "exp_plan_version",
                        "authoritative_receipt_ids",
                    )
                    for shid in syn_hids:
                        exp_entry = expected_snap.get(shid)
                        cur_entry = current_snap.get(shid)
                        if not isinstance(exp_entry, Mapping) or not isinstance(cur_entry, Mapping):
                            stale_sources.append(f"{shid}:missing")
                            continue
                        if any(rk not in exp_entry for rk in required_snap_keys):
                            stale_sources.append(f"{shid}:incomplete_snapshot")
                            continue
                        if cur_entry.get("status") in (
                            "PRUNED",
                            "HUMAN_INVALIDATED",
                            "REFUTED",
                            "GRAVEYARD",
                            "REJECTED",
                        ):
                            stale_sources.append(f"{shid}:status={cur_entry.get('status')}")
                            continue
                        for k_cmp in required_snap_keys:
                            if cur_entry.get(k_cmp) != exp_entry.get(k_cmp):
                                stale_sources.append(f"{shid}:{k_cmp}_changed")
                                break
                        else:
                            if (
                                "lineage_sci_revision" in exp_entry
                                and cur_entry.get("lineage_sci_revision") != exp_entry.get("lineage_sci_revision")
                            ):
                                stale_sources.append(f"{shid}:lineage_sci_revision_changed")
                if stale_sources:
                    self.db.upsert_knowledge_item(
                        goal.goal_id,
                        "synthesis_pending",
                        syn_key,
                        {
                            "status": (
                                "SUPERSEDED_MISSING_SNAPSHOT"
                                if missing_snapshot
                                else "SUPERSEDED_BY_SOURCE_UPDATE"
                            ),
                            "workflow_id": fresh_wf.workflow_id,
                            "synthesized_hypothesis_ids": syn_hids,
                            "stale_sources": stale_sources,
                        },
                        hypothesis_id=hyp.hypothesis_id,
                        goal_version=int(goal.version),
                    )
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="SYNTHESIS_SUPERSEDED",
                        state_patch={
                            "step": "synthesis_superseded",
                            "stale_sources": stale_sources,
                            "completed_steps": completed_steps,
                        },
                    )
                    if missing_snapshot and not fresh_wf.state.get("respawned_for_missing_snapshot"):
                        try:
                            valid_sources = True
                            for shid in syn_hids:
                                sh_obj = (
                                    self.db.get_goal_hypothesis(goal.goal_id, shid)
                                    if hasattr(self.db, "get_goal_hypothesis")
                                    else self.db.get_hypothesis(shid)
                                )
                                if sh_obj is None or sh_obj.status in (
                                    "PRUNED",
                                    "HUMAN_INVALIDATED",
                                    "REFUTED",
                                    "GRAVEYARD",
                                    "REJECTED",
                                ):
                                    valid_sources = False
                                    break
                            if valid_sources:
                                new_syn_wf, respawned_tasks = self.spawn_synthesis_review(
                                    goal,
                                    hyp,
                                    synthesized_hypothesis_ids=syn_hids,
                                    synthesis_key=syn_key,
                                    thesis=str(fresh_wf.config.get("thesis") or goal.title),
                                    rationale=str(fresh_wf.config.get("rationale") or ""),
                                    findings=dict(fresh_wf.config.get("findings") or {}),
                                )
                                self.db.update_workflow_state(
                                    fresh_wf.workflow_id,
                                    state_patch={
                                        "respawned_for_missing_snapshot": True,
                                        "replacement_workflow_id": new_syn_wf.workflow_id,
                                    },
                                )
                                return respawned_tasks
                        except Exception:
                            pass
                    return []

                findings = dict(fresh_wf.config.get("findings") or {})
                thesis = str(fresh_wf.config.get("thesis") or goal.title)
                self.db.upsert_knowledge_item(
                    goal.goal_id,
                    "synthesis",
                    syn_key,
                    {
                        "status": "VERIFIED_SYNTHESIS",
                        "workflow_id": fresh_wf.workflow_id,
                        "synthesized_hypothesis_ids": syn_hids,
                        "source_lineage_snapshot": expected_snap or {},
                        "thesis": thesis,
                        "rationale": str(fresh_wf.config.get("rationale") or ""),
                        "findings": findings,
                    },
                    hypothesis_id=hyp.hypothesis_id,
                    goal_version=int(goal.version),
                )
                if thesis:
                    PlanLifecycleManager(self.db).create_research_plan(
                        goal,
                        thesis=thesis,
                        uncertainties=findings.get("uncertainties") or (),
                        competing_routes=findings.get("competing_routes") or (),
                        milestone_criteria=findings.get("milestone_criteria") or (),
                        status="APPROVED",
                    )
                self.db.update_workflow_state(
                    fresh_wf.workflow_id,
                    status="COMPLETED",
                    state_patch={"step": "completed", "completed_steps": completed_steps},
                )
            else:
                self.db.update_workflow_state(
                    fresh_wf.workflow_id,
                    status="SYNTHESIS_REJECTED",
                    state_patch={"step": "synthesis_rejected", "completed_steps": completed_steps},
                )
            return []

        # -------- Pre-Code Gate R1: DESIGN_REVIEW (shared across all write primitives)
        if completed_task.stage == StageKind.DESIGN_REVIEW or step_id.startswith("design_review"):
            outcome, q_tasks = self._reduce_review_step(
                goal,
                hyp,
                fresh_wf,
                completed_task,
                receipt,
                review_kind=ReviewKind.DESIGN_REVIEW,
                completed_steps=completed_steps,
            )
            if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                return q_tasks
            if outcome == "PASS":
                exp_plan = self.db.get_latest_experiment_plan(hyp.hypothesis_id)
                if exp_plan is not None and exp_plan.review_status != "APPROVED":
                    exp_plan.review_status = "APPROVED"
                    self.db.save_experiment_plan(exp_plan)
                _, b_tasks = self._emit_initial_build_tasks(
                    goal,
                    hyp,
                    fresh_wf,
                    deps=[completed_task.task_id],
                    worktree_path=completed_task.worktree_path,
                )
                return b_tasks

            # Check whether FAIL contains ONLY theoretical/methodological critiques vs hard safety/holdout/banned/metric/paid-compute violations
            r1_receipts = self.db.list_review_receipts(
                workflow_id=fresh_wf.workflow_id,
                review_kind=ReviewKind.DESIGN_REVIEW.value,
            )
            all_findings: list[str] = []
            for rr in r1_receipts:
                for f in getattr(rr, "findings", ()) or ():
                    fs = str(f).strip()
                    if fs and fs not in all_findings:
                        all_findings.append(fs)
            if receipt.failure_reason.strip() and receipt.failure_reason.strip() not in all_findings:
                all_findings.append(receipt.failure_reason.strip())
            hard_blockers = extract_hard_design_blockers(
                all_findings,
                failure_reason=receipt.failure_reason,
                severity=str((receipt.metrics or {}).get("severity", "")),
            )
            reversible_f1_eligible = is_reversible_f1_exploratory(
                goal,
                hyp,
                fresh_wf.config,
                ignore_explicit_design_review=True,
            )
            valid_structured_fail = (
                receipt.process_status == ProcessStatus.EXIT_OK
                and receipt.audit_verdict == AuditVerdict.FAIL
                and bool(r1_receipts)
                and all(
                    rr.process_status == ProcessStatus.EXIT_OK
                    and rr.verdict in (AuditVerdict.PASS, AuditVerdict.FAIL)
                    for rr in r1_receipts
                )
            )
            exp_plan_chk = self.db.get_latest_experiment_plan(hyp.hypothesis_id)
            if exp_plan_chk is None:
                exp_plan_chk = self.plan_manager.build_default_experiment_plan(goal, hyp)
            contract_gate = validate_r1_fast_lane_contract(
                exp_plan_chk,
                goal,
                hyp,
                repo_path=goal.repo_path or None,
            )
            if not contract_gate.passed:
                hard_blockers = [*hard_blockers, *contract_gate.failures]
            if reversible_f1_eligible and valid_structured_fail and contract_gate.passed and not hard_blockers:
                advisory_checks = list(all_findings) or [
                    f"Advisory design critique ({completed_task.step_key}): check theoretical confounds during F1_FAST_SCREEN"
                ]
                exp_plan = self.db.get_latest_experiment_plan(hyp.hypothesis_id)
                if exp_plan is None:
                    exp_plan = self.plan_manager.build_default_experiment_plan(goal, hyp)
                merged_checks = list(exp_plan.confound_checks)
                for chk in advisory_checks:
                    if chk not in merged_checks:
                        merged_checks.append(chk)
                exp_plan.confound_checks = merged_checks
                exp_plan.review_status = "APPROVED"
                self.db.save_experiment_plan(exp_plan)

                self.db.update_workflow_state(
                    fresh_wf.workflow_id,
                    status="RUNNING",
                    state_patch={
                        "step": "build_entry",
                        "completed_steps": completed_steps,
                        "design_review_advisory": True,
                        "advisory_confound_checks": advisory_checks,
                    },
                )
                hyp = self.db.update_hypothesis_params(
                    hyp.hypothesis_id,
                    {"advisory_confound_checks": advisory_checks},
                )
                with self.db._tx() as (conn, pending):
                    self.db._record_event(
                        conn,
                        pending,
                        "DESIGN_REVIEW_ADVISORY_CONFOUND",
                        goal_id=goal.goal_id,
                        hypothesis_id=hyp.hypothesis_id,
                        task_id=completed_task.task_id,
                        payload={
                            "workflow_id": fresh_wf.workflow_id,
                            "advisory_confound_checks": advisory_checks,
                        },
                    )
                self.db.upsert_knowledge_item(
                    goal.goal_id,
                    "design_advisory",
                    f"r1_advisory:{hyp.hypothesis_id}:{fresh_wf.workflow_id}",
                    {
                        "workflow_id": fresh_wf.workflow_id,
                        "advisory_confound_checks": advisory_checks,
                    },
                    hypothesis_id=hyp.hypothesis_id,
                    goal_version=int(goal.version),
                )
                _, b_tasks = self._emit_initial_build_tasks(
                    goal,
                    hyp,
                    self.db.get_workflow(fresh_wf.workflow_id) or fresh_wf,
                    deps=[completed_task.task_id],
                    worktree_path=completed_task.worktree_path,
                )
                return b_tasks

            self.db.update_workflow_state(
                fresh_wf.workflow_id,
                status="DESIGN_REJECTED",
                state_patch={
                    "step": "design_rejected",
                    "completed_steps": completed_steps,
                    "hard_design_blockers": hard_blockers,
                },
            )
            self.db.transition_hypothesis_status(
                hyp.hypothesis_id,
                hyp.status,
                "NEEDS_PLAN_REVISION",
                failure_domain=FailureDomain.ENGINEERING_DEFECT,
            )
            return []

        # -------- Primitive 0: readonly_fanout
        if prim == "readonly_fanout":
            if step_id == "fanout_inspect" and not exec_failed:
                self.db.update_workflow_state(
                    fresh_wf.workflow_id,
                    state_patch={"step": "fanout_reconcile", "completed_steps": completed_steps},
                )
                emitted.append(
                    self._make_task(
                        goal,
                        hyp,
                        fresh_wf,
                        StageKind.RECONCILE,
                        step_key="fanout_reconcile",
                        deps=[completed_task.task_id],
                        worktree_path=completed_task.worktree_path,
                    )
                )
            else:
                self.db.update_workflow_state(
                    fresh_wf.workflow_id,
                    status="FAILED" if exec_failed else "COMPLETED",
                    state_patch={"step": "done", "completed_steps": completed_steps},
                )

        # -------- Primitive 1: direct_hq
        elif prim == "direct_hq":
            repairs_used = int(fresh_wf.state.get("repairs_used", 0))
            max_repairs = int(fresh_wf.config.get("max_repairs", 2))

            if completed_task.stage == StageKind.CODE:
                if exec_failed:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="FAILED",
                        state_patch={"step": "failed", "completed_steps": completed_steps},
                    )
                else:
                    gate_key = "cpu_gate" if repairs_used == 0 else f"cpu_gate_r{repairs_used}"
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        state_patch={"step": gate_key, "completed_steps": completed_steps},
                    )
                    emitted.append(
                        self._make_task(
                            goal,
                            hyp,
                            fresh_wf,
                            StageKind.CPU_GATE,
                            step_key=gate_key,
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                        )
                    )
            elif completed_task.stage == StageKind.CPU_GATE:
                if exec_failed:
                    if fresh_wf.parent_workflow_id is None and repairs_used < max_repairs:
                        next_r = repairs_used + 1
                        r_key = f"code_repair_{next_r}"
                        self.db.update_workflow_state(
                            fresh_wf.workflow_id,
                            state_patch={
                                "step": r_key,
                                "repairs_used": next_r,
                                "completed_steps": completed_steps,
                            },
                        )
                        emitted.append(
                            self._make_task(
                                goal,
                                hyp,
                                fresh_wf,
                                StageKind.CODE,
                                step_key=r_key,
                                deps=[],
                                attempt=next_r + 1,
                                worktree_path=completed_task.worktree_path,
                            )
                        )
                    else:
                        self.db.update_workflow_state(
                            fresh_wf.workflow_id,
                            status="FAILED",
                            state_patch={"step": "gate_failed", "completed_steps": completed_steps},
                        )
                elif fresh_wf.parent_workflow_id is not None:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="COMPLETED",
                        state_patch={
                            "step": "completed",
                            "completed_steps": completed_steps,
                            "head_sha": receipt.head_sha,
                            "tests_passed": int(receipt.tests_passed),
                        },
                    )
                elif review_tier == ReviewTier.FAST_LANE or (
                    hyp.kind in (HypothesisKind.PARAMETER_RACE, HypothesisKind.ABLATION, HypothesisKind.FALSIFICATION)
                    and not hyp.params.get("requires_llm_code", False)
                ):
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        state_patch={"step": "exp_screen", "completed_steps": completed_steps},
                    )
                    emitted.append(
                        self._make_task(
                            goal,
                            hyp,
                            fresh_wf,
                            StageKind.EXP_SCREEN,
                            step_key="exp_screen",
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                        )
                    )
                else:
                    emitted.extend(
                        self._spawn_review_quorum_tasks(
                            goal,
                            hyp,
                            fresh_wf,
                            review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                            review_tier=review_tier,
                            step_prefix=f"impl_review_r{repairs_used}",
                            next_step_on_pass="exp_screen",
                            head_sha=receipt.head_sha or hyp.head_sha or hyp.base_sha,
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                            target_version=repairs_used + 1,
                            completed_steps=completed_steps,
                        )
                    )
            elif step_id.startswith("impl_review") or completed_task.stage == StageKind.IMPL_REVIEW:
                next_emp_step = str(
                    (fresh_wf.state.get("review_quorum") or {}).get("next_step_on_pass")
                    or "exp_screen"
                )
                outcome, q_tasks = self._reduce_review_step(
                    goal,
                    hyp,
                    fresh_wf,
                    completed_task,
                    receipt,
                    review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                    completed_steps=completed_steps,
                )
                if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                    emitted.extend(q_tasks)
                elif outcome == "PASS":
                    emp_stage = (
                        StageKind.EXP_CONFIRM
                        if next_emp_step == "exp_confirm"
                        else StageKind.EXP_SCREEN
                    )
                    emp_key = "exp_confirm" if emp_stage == StageKind.EXP_CONFIRM else "exp_screen"
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        state_patch={"step": emp_key, "completed_steps": completed_steps},
                    )
                    emitted.append(
                        self._make_task(
                            goal,
                            hyp,
                            fresh_wf,
                            emp_stage,
                            step_key=emp_key,
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                        )
                    )
                elif repairs_used < max_repairs:
                    next_r = repairs_used + 1
                    r_key = f"code_repair_{next_r}"
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        state_patch={
                            "step": r_key,
                            "repairs_used": next_r,
                            "completed_steps": completed_steps,
                        },
                    )
                    emitted.append(
                        self._make_task(
                            goal,
                            hyp,
                            fresh_wf,
                            StageKind.CODE,
                            step_key=r_key,
                            deps=[completed_task.task_id],
                            attempt=next_r + 1,
                            worktree_path=completed_task.worktree_path,
                        )
                    )
                else:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="FAILED",
                        state_patch={"step": "impl_review_failed", "completed_steps": completed_steps},
                    )
            elif completed_task.stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
                emitted.extend(
                    self._handle_empirical_screen_and_confirm(
                        goal,
                        hyp,
                        fresh_wf,
                        completed_task,
                        receipt,
                        screen_key="exp_screen",
                        confirm_key="exp_confirm",
                        result_review_prefix="result_review",
                        reviewer_spec=None,
                        completed_steps=completed_steps,
                    )
                )
            elif completed_task.stage in (StageKind.RESULT_REVIEW, StageKind.BLIND_AUDIT) or step_id.startswith("result_review"):
                outcome, q_tasks = self._reduce_review_step(
                    goal,
                    hyp,
                    fresh_wf,
                    completed_task,
                    receipt,
                    review_kind=ReviewKind.RESULT_REVIEW,
                    completed_steps=completed_steps,
                )
                if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                    emitted.extend(q_tasks)
                elif outcome == "PASS":
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="COMPLETED",
                        state_patch={"step": "completed", "completed_steps": completed_steps},
                    )
                else:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="RESULT_REJECTED",
                        state_patch={"step": "result_rejected", "completed_steps": completed_steps},
                    )

        # -------- Primitive 2: x-boost (with mandatory post-repair re-audit & full EXP_CONFIRM -> RESULT_REVIEW)
        elif prim == "x-boost":
            repairs_used = int(fresh_wf.state.get("repairs_used", 0))
            max_repairs = int(fresh_wf.config.get("max_repairs", 2))
            builder_pool = str(fresh_wf.config.get("builder_pool") or "claude:claude-sonnet-5-5")
            adv_pool = str(fresh_wf.config.get("adversary_pool") or "codex:gpt-5-codex")

            if step_id == "boost_build" or step_id.startswith("boost_repair_"):
                if exec_failed:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="FAILED",
                        state_patch={"step": "failed", "completed_steps": completed_steps},
                    )
                else:
                    gate_key = "boost_gate" if step_id == "boost_build" else f"boost_gate_r{repairs_used}"
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        state_patch={"step": gate_key, "completed_steps": completed_steps},
                    )
                    emitted.append(
                        self._make_task(
                            goal,
                            hyp,
                            fresh_wf,
                            StageKind.CPU_GATE,
                            step_key=gate_key,
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                        )
                    )
            elif step_id == "boost_gate" or step_id.startswith("boost_gate_r"):
                if exec_failed:
                    if repairs_used < max_repairs:
                        next_r = repairs_used + 1
                        r_key = f"boost_repair_{next_r}"
                        self.db.update_workflow_state(
                            fresh_wf.workflow_id,
                            state_patch={
                                "step": r_key,
                                "repairs_used": next_r,
                                "completed_steps": completed_steps,
                            },
                        )
                        emitted.append(
                            self._make_task(
                                goal,
                                hyp,
                                fresh_wf,
                                StageKind.CODE,
                                step_key=r_key,
                                deps=[],
                                attempt=next_r + 1,
                                worktree_path=completed_task.worktree_path,
                                resources_override=ResourceSpec(
                                    cpus=1, ram_mb=2048, llm_pool=builder_pool, wall_timeout_s=900
                                ),
                            )
                        )
                    else:
                        self.db.update_workflow_state(
                            fresh_wf.workflow_id,
                            status="FAILED",
                            state_patch={"step": "gate_failed", "completed_steps": completed_steps},
                        )
                else:
                    # MANDATORY R2 IMPL_REVIEW on EVERY gate pass — including at `repairs_used == max_repairs`!
                    adv_key = f"boost_adversary_{repairs_used}"
                    emitted.extend(
                        self._spawn_review_quorum_tasks(
                            goal,
                            hyp,
                            fresh_wf,
                            review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                            review_tier=(
                                ReviewTier.RED_TEAM_TIEBREAKER
                                if review_tier == ReviewTier.RED_TEAM_TIEBREAKER
                                else ReviewTier.STANDARD_CROSS_FAMILY
                            ),
                            step_prefix=adv_key,
                            next_step_on_pass="exp_screen",
                            head_sha=receipt.head_sha or hyp.head_sha or hyp.base_sha,
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                            explicit_reviewer_spec=adv_pool,
                            target_version=repairs_used + 1,
                            completed_steps=completed_steps,
                        )
                    )
            elif step_id.startswith("boost_adversary_"):
                next_emp_step = str(
                    (fresh_wf.state.get("review_quorum") or {}).get("next_step_on_pass")
                    or "exp_screen"
                )
                outcome, q_tasks = self._reduce_review_step(
                    goal,
                    hyp,
                    fresh_wf,
                    completed_task,
                    receipt,
                    review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                    completed_steps=completed_steps,
                )
                if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                    emitted.extend(q_tasks)
                elif outcome == "PASS":
                    if fresh_wf.parent_workflow_id is not None:
                        self.db.update_workflow_state(
                            fresh_wf.workflow_id,
                            status="COMPLETED",
                            state_patch={
                                "step": "completed",
                                "completed_steps": completed_steps,
                                "head_sha": receipt.head_sha,
                                "tests_passed": int(receipt.tests_passed),
                            },
                        )
                    else:
                        emp_stage = (
                            StageKind.EXP_CONFIRM
                            if next_emp_step == "exp_confirm"
                            else StageKind.EXP_SCREEN
                        )
                        emp_key = "exp_confirm" if emp_stage == StageKind.EXP_CONFIRM else "exp_screen"
                        self.db.update_workflow_state(
                            fresh_wf.workflow_id,
                            state_patch={"step": emp_key, "completed_steps": completed_steps},
                        )
                        emitted.append(
                            self._make_task(
                                goal,
                                hyp,
                                fresh_wf,
                                emp_stage,
                                step_key=emp_key,
                                deps=[completed_task.task_id],
                                worktree_path=completed_task.worktree_path,
                            )
                        )
                elif repairs_used < max_repairs:
                    next_r = repairs_used + 1
                    r_key = f"boost_repair_{next_r}"
                    bstate = dict(hyp.budget_state or {})
                    bstate["code_repairs_used"] = max(int(bstate.get("code_repairs_used", 0)), next_r)
                    self.db.update_hypothesis_v3_state(hyp.hypothesis_id, budget_state_patch=bstate)
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        state_patch={
                            "step": r_key,
                            "repairs_used": next_r,
                            "completed_steps": completed_steps,
                        },
                    )
                    emitted.append(
                        self._make_task(
                            goal,
                            hyp,
                            fresh_wf,
                            StageKind.CODE,
                            step_key=r_key,
                            deps=[completed_task.task_id],
                            attempt=next_r + 1,
                            worktree_path=completed_task.worktree_path,
                            resources_override=ResourceSpec(
                                cpus=1, ram_mb=2048, llm_pool=builder_pool, wall_timeout_s=900
                            ),
                        )
                    )
                else:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="FAILED",
                        state_patch={"step": "repairs_exhausted", "completed_steps": completed_steps},
                    )
            elif completed_task.stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
                emitted.extend(
                    self._handle_empirical_screen_and_confirm(
                        goal,
                        hyp,
                        fresh_wf,
                        completed_task,
                        receipt,
                        screen_key="exp_screen",
                        confirm_key="exp_confirm",
                        result_review_prefix="boost_result_review",
                        reviewer_spec=adv_pool,
                        completed_steps=completed_steps,
                    )
                )
            elif step_id.startswith("boost_result_review") or completed_task.stage == StageKind.RESULT_REVIEW:
                outcome, q_tasks = self._reduce_review_step(
                    goal,
                    hyp,
                    fresh_wf,
                    completed_task,
                    receipt,
                    review_kind=ReviewKind.RESULT_REVIEW,
                    completed_steps=completed_steps,
                )
                if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                    emitted.extend(q_tasks)
                elif outcome == "PASS":
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="COMPLETED",
                        state_patch={"step": "completed", "completed_steps": completed_steps},
                    )
                else:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="RESULT_REJECTED",
                        state_patch={"step": "result_rejected", "completed_steps": completed_steps},
                    )

        # -------- Primitive 3: x-swarm (`workers -> swarm_cpu_gate -> swarm_impl_review -> swarm_exp_screen -> swarm_exp_confirm -> swarm_result_review`)
        elif prim == "x-swarm":
            tick = int(fresh_wf.state.get("tick", 1))
            max_ticks = int(fresh_wf.config.get("max_ticks", 4))
            expected_workers = list(fresh_wf.state.get("expected_workers") or [])
            completed_workers = list(fresh_wf.state.get("completed_workers") or [])
            ver_pool = str(fresh_wf.config.get("verifier_pool") or "codex:gpt-5-codex")

            if step_id.startswith("swarm_worker:"):
                w_name = step_id.split(":", 2)[1]
                if exec_failed:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="FAILED",
                        state_patch={"step": f"worker_failed:{w_name}", "completed_steps": completed_steps},
                    )
                else:
                    if w_name not in completed_workers:
                        completed_workers.append(w_name)
                    if set(expected_workers).issubset(set(completed_workers)):
                        self.db.update_workflow_state(
                            fresh_wf.workflow_id,
                            state_patch={
                                "step": "swarm_cpu_gate",
                                "completed_workers": completed_workers,
                                "completed_steps": completed_steps,
                            },
                        )
                        emitted.append(
                            self._make_task(
                                goal,
                                hyp,
                                fresh_wf,
                                StageKind.CPU_GATE,
                                step_key="swarm_cpu_gate",
                                deps=[completed_task.task_id],
                                attempt=tick,
                                worktree_path=completed_task.worktree_path,
                            )
                        )
                    else:
                        self.db.update_workflow_state(
                            fresh_wf.workflow_id,
                            state_patch={
                                "completed_workers": completed_workers,
                                "completed_steps": completed_steps,
                            },
                        )
            elif step_id == "swarm_cpu_gate":
                if exec_failed:
                    if tick < max_ticks:
                        emitted.extend(
                            self._respawn_swarm_tick(
                                goal,
                                hyp,
                                fresh_wf,
                                tick + 1,
                                completed_task,
                                completed_steps,
                                deps=[],
                            )
                        )
                    else:
                        self.db.update_workflow_state(
                            fresh_wf.workflow_id,
                            status="FAILED",
                            state_patch={"step": "swarm_gate_failed", "completed_steps": completed_steps},
                        )
                else:
                    emitted.extend(
                        self._spawn_review_quorum_tasks(
                            goal,
                            hyp,
                            fresh_wf,
                            review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                            review_tier=(
                                ReviewTier.RED_TEAM_TIEBREAKER
                                if review_tier == ReviewTier.RED_TEAM_TIEBREAKER
                                else ReviewTier.STANDARD_CROSS_FAMILY
                            ),
                            step_prefix=f"swarm_impl_review_t{tick}",
                            next_step_on_pass="swarm_exp_screen",
                            head_sha=receipt.head_sha or hyp.head_sha or hyp.base_sha,
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                            explicit_reviewer_spec=ver_pool,
                            target_version=tick,
                            completed_steps=completed_steps,
                        )
                    )
            elif step_id.startswith("swarm_impl_review"):
                outcome, q_tasks = self._reduce_review_step(
                    goal,
                    hyp,
                    fresh_wf,
                    completed_task,
                    receipt,
                    review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                    completed_steps=completed_steps,
                )
                if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                    emitted.extend(q_tasks)
                elif outcome == "PASS":
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        state_patch={"step": "swarm_exp_screen", "completed_steps": completed_steps},
                    )
                    emitted.append(
                        self._make_task(
                            goal,
                            hyp,
                            fresh_wf,
                            StageKind.EXP_SCREEN,
                            step_key="swarm_exp_screen",
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                        )
                    )
                elif tick < max_ticks:
                    emitted.extend(
                        self._respawn_swarm_tick(
                            goal, hyp, fresh_wf, tick + 1, completed_task, completed_steps
                        )
                    )
                else:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="FAILED",
                        state_patch={"step": "swarm_max_ticks", "completed_steps": completed_steps},
                    )
            elif step_id in ("swarm_exp_screen", "swarm_exp_confirm"):
                emitted.extend(
                    self._handle_empirical_screen_and_confirm(
                        goal,
                        hyp,
                        fresh_wf,
                        completed_task,
                        receipt,
                        screen_key="swarm_exp_screen",
                        confirm_key="swarm_exp_confirm",
                        result_review_prefix=f"swarm_result_review_t{tick}",
                        reviewer_spec=ver_pool,
                        completed_steps=completed_steps,
                    )
                )
            elif step_id.startswith("swarm_result_review") or step_id.startswith("swarm_verify_"):
                outcome, q_tasks = self._reduce_review_step(
                    goal,
                    hyp,
                    fresh_wf,
                    completed_task,
                    receipt,
                    review_kind=ReviewKind.RESULT_REVIEW,
                    completed_steps=completed_steps,
                )
                if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                    emitted.extend(q_tasks)
                elif outcome == "PASS":
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="COMPLETED",
                        state_patch={"step": "completed", "completed_steps": completed_steps},
                    )
                else:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="RESULT_REJECTED",
                        state_patch={"step": "result_rejected", "completed_steps": completed_steps},
                    )

        # -------- Primitive 4: x-teamwork (`phases -> teamwork_cpu_gate -> teamwork_impl_review -> teamwork_exp_screen -> teamwork_exp_confirm -> teamwork_result_review`)
        elif prim == "x-teamwork":
            aud_pool = str(fresh_wf.config.get("auditor_pool") or "codex:gpt-5-codex")
            if step_id == "teamwork_cpu_gate":
                if exec_failed:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="FAILED",
                        state_patch={"step": "teamwork_gate_failed", "completed_steps": completed_steps},
                    )
                else:
                    staging_sha = str(
                        fresh_wf.state.get("staging_head")
                        or receipt.head_sha
                        or hyp.head_sha
                        or hyp.base_sha
                    )
                    emitted.extend(
                        self._spawn_review_quorum_tasks(
                            goal,
                            hyp,
                            fresh_wf,
                            review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                            review_tier=review_tier,
                            step_prefix="teamwork_impl_review",
                            next_step_on_pass="teamwork_exp_screen",
                            head_sha=staging_sha,
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                            explicit_reviewer_spec=aud_pool,
                            target_version=1,
                            completed_steps=completed_steps,
                        )
                    )
            elif step_id.startswith("teamwork_impl_review"):
                outcome, q_tasks = self._reduce_review_step(
                    goal,
                    hyp,
                    fresh_wf,
                    completed_task,
                    receipt,
                    review_kind=ReviewKind.IMPLEMENTATION_REVIEW,
                    completed_steps=completed_steps,
                )
                if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                    emitted.extend(q_tasks)
                elif outcome == "PASS":
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        state_patch={"step": "teamwork_exp_screen", "completed_steps": completed_steps},
                    )
                    emitted.append(
                        self._make_task(
                            goal,
                            hyp,
                            fresh_wf,
                            StageKind.EXP_SCREEN,
                            step_key="teamwork_exp_screen",
                            deps=[completed_task.task_id],
                            worktree_path=completed_task.worktree_path,
                        )
                    )
                else:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="FAILED",
                        state_patch={"step": "teamwork_impl_review_failed", "completed_steps": completed_steps},
                    )
            elif step_id in ("teamwork_exp_screen", "teamwork_exp_confirm"):
                emitted.extend(
                    self._handle_empirical_screen_and_confirm(
                        goal,
                        hyp,
                        fresh_wf,
                        completed_task,
                        receipt,
                        screen_key="teamwork_exp_screen",
                        confirm_key="teamwork_exp_confirm",
                        result_review_prefix="teamwork_result_review",
                        reviewer_spec=aud_pool,
                        completed_steps=completed_steps,
                    )
                )
            elif step_id.startswith("teamwork_result_review") or step_id == "teamwork_blind_audit":
                outcome, q_tasks = self._reduce_review_step(
                    goal,
                    hyp,
                    fresh_wf,
                    completed_task,
                    receipt,
                    review_kind=ReviewKind.RESULT_REVIEW,
                    completed_steps=completed_steps,
                )
                if outcome in ("WAIT", "SPAWNED", "BLOCKED", "CANCELLED"):
                    emitted.extend(q_tasks)
                elif outcome == "PASS":
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="COMPLETED",
                        state_patch={"step": "completed", "completed_steps": completed_steps},
                    )
                else:
                    self.db.update_workflow_state(
                        fresh_wf.workflow_id,
                        status="RESULT_REJECTED",
                        state_patch={"step": "audit_failed", "completed_steps": completed_steps},
                    )

        # Bubble child workflow completion/failure up to parent_workflow_id
        updated_self = self.db.get_workflow(fresh_wf.workflow_id) or fresh_wf
        if updated_self.status in TERMINAL_WORKFLOW_STATUSES:
            self.cancel_terminal_workflow_tasks(updated_self.goal_id, updated_self.workflow_id)
        if updated_self.parent_workflow_id and updated_self.status in ("COMPLETED", "FAILED"):
            parent_wf = self.db.get_workflow(updated_self.parent_workflow_id)
            if parent_wf is not None and parent_wf.status not in TERMINAL_WORKFLOW_STATUSES:
                emitted.extend(
                    self._handle_child_completion(
                        parent_wf,
                        updated_self,
                        completed_task,
                        receipt,
                        goal,
                        hyp,
                    )
                )

        return emitted

    def cancel_terminal_workflow_tasks(
        self,
        goal_id: str,
        workflow_id: str,
        *,
        now: float | None = None,
    ) -> int:
        """Cancel all remaining non-terminal READY/PENDING tasks belonging to a terminal workflow."""
        ts = time.time() if now is None else float(now)
        cancelled = 0
        for rt in self.db.list_tasks(goal_id=goal_id):
            if rt.workflow_id == workflow_id and rt.status in ("READY", "PENDING"):
                with contextlib.suppress(StateTransitionError):
                    self.db.transition_task_status(
                        rt.task_id, rt.status, "CANCELLED", finished_ts=ts, now=ts
                    )
                    cancelled += 1
        return cancelled

    def _respawn_swarm_tick(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        fresh_wf: WorkflowNode,
        next_tick: int,
        completed_task: StageTask,
        completed_steps: list[str],
        *,
        deps: Sequence[str] | None = None,
    ) -> list[StageTask]:
        workers_cfg = list(fresh_wf.config.get("workers") or [])
        self.db.update_workflow_state(
            fresh_wf.workflow_id,
            state_patch={
                "step": "swarm_workers",
                "tick": next_tick,
                "completed_workers": [],
                "completed_steps": completed_steps,
            },
        )
        # Pass deps=[] when triggered by a FAILED swarm_cpu_gate; swarm_impl_review FAIL ends DONE.
        eff_deps = [completed_task.task_id] if deps is None else list(deps)
        return [
            self._make_task(
                goal,
                hyp,
                fresh_wf,
                StageKind.CODE,
                step_key=f"swarm_worker:{w['name']}:t{next_tick}",
                deps=eff_deps,
                attempt=next_tick,
                worktree_path=completed_task.worktree_path,
                resources_override=ResourceSpec(
                    cpus=1, ram_mb=2048, llm_pool=w["llm_pool"], wall_timeout_s=600
                ),
            )
            for w in workers_cfg
        ]

    def _handle_child_completion(
        self,
        parent_wf: WorkflowNode,
        child_wf: WorkflowNode,
        last_task: StageTask,
        last_receipt: EvidenceReceipt,
        goal: ResearchGoal,
        hyp: HypothesisNode,
    ) -> list[StageTask]:
        phase_idx = int(parent_wf.state.get("phase_idx", 1))
        phase_children = list(parent_wf.state.get("phase_children") or [])
        completed_children = list(parent_wf.state.get("completed_children") or [])
        failed_children = list(parent_wf.state.get("failed_children") or [])
        slice_results = list(parent_wf.state.get("slice_results") or [])

        if child_wf.status == "FAILED":
            if child_wf.workflow_id not in failed_children:
                failed_children.append(child_wf.workflow_id)
            self.db.update_workflow_state(
                parent_wf.workflow_id,
                status="FAILED",
                state_patch={
                    "step": f"phase_{phase_idx}_child_failed",
                    "failed_children": failed_children,
                },
            )
            return []

        if child_wf.workflow_id not in completed_children:
            completed_children.append(child_wf.workflow_id)
            slice_results.append(
                {
                    "name": str(child_wf.config.get("slice_name") or child_wf.workflow_id),
                    "provider": str(child_wf.config.get("builder_pool") or child_wf.primitive),
                    "head_sha": str(
                        child_wf.state.get("head_sha") or last_receipt.head_sha or hyp.base_sha
                    ),
                    "writable_paths": list(
                        child_wf.config.get("writable_paths") or hyp.writable_paths
                    ),
                    "tests_passed": int(
                        child_wf.state.get("tests_passed", last_receipt.tests_passed)
                    ),
                }
            )

        if not set(phase_children).issubset(set(completed_children)):
            self.db.update_workflow_state(
                parent_wf.workflow_id,
                state_patch={
                    "completed_children": completed_children,
                    "slice_results": slice_results,
                },
            )
            return []

        staging_wt, staging_head = self._merge_child_results_onto_staging(
            goal,
            hyp,
            parent_wf,
            slice_results,
            last_task.worktree_path,
        )
        phases = list(parent_wf.config.get("phases") or [])
        p_name = (
            str(phases[phase_idx - 1].get("name") or f"phase_{phase_idx}")
            if phase_idx <= len(phases)
            else f"phase_{phase_idx}"
        )
        anchor = build_milestone_anchor(
            phase_idx=phase_idx,
            phase_name=p_name,
            staging_head=staging_head,
            slice_results=slice_results,
            prescreen_note=f"phase={phase_idx} children={len(completed_children)} ok",
        )
        self.db.upsert_knowledge_item(
            goal.goal_id,
            "milestone_anchor",
            f"wf:{parent_wf.workflow_id}:p{phase_idx}",
            {"phase_idx": phase_idx, "anchor": anchor, "bytes": len(anchor.encode("utf-8"))},
            hypothesis_id=hyp.hypothesis_id,
            goal_version=int(goal.version),
        )
        parent_wf = self.db.update_workflow_state(
            parent_wf.workflow_id,
            state_patch={
                "completed_children": completed_children,
                "slice_results": slice_results,
                "milestone_anchor": anchor,
                "staging_worktree_path": staging_wt,
                "staging_head": staging_head,
            },
        )

        if phase_idx < len(phases):
            _, next_tasks = self._spawn_teamwork_phase(
                parent_wf,
                goal,
                hyp,
                phase_idx=phase_idx + 1,
                worktree_path=staging_wt,
            )
            return next_tasks

        # All phases complete -> emit pre-screen `teamwork_cpu_gate` -> `teamwork_impl_review` -> `teamwork_exp_screen`!
        self.db.update_workflow_state(
            parent_wf.workflow_id,
            state_patch={"step": "teamwork_cpu_gate"},
        )
        gate_task = self._make_task(
            goal,
            hyp,
            parent_wf,
            StageKind.CPU_GATE,
            step_key="teamwork_cpu_gate",
            deps=[last_task.task_id],
            worktree_path=staging_wt,
        )
        return [gate_task]

    def _merge_child_results_onto_staging(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        parent_wf: WorkflowNode,
        slice_results: Sequence[Mapping[str, Any]],
        base_wt: str | None,
    ) -> tuple[str | None, str]:
        """Create/reuse parent staging worktree and merge completed child slice commits cleanly."""
        fallback_head = str(
            parent_wf.state.get("staging_head")
            or (slice_results[-1].get("head_sha") if slice_results else "")
            or hyp.head_sha
            or hyp.base_sha
        )
        repo_raw = str(goal.repo_path or "").strip()
        repo_path = Path(repo_raw) if repo_raw else None
        if repo_path is None or not (repo_path.exists() and (repo_path / ".git").exists()):
            return base_wt, fallback_head

        from conjectra.stage_runner import SandboxManager, _git

        sb = SandboxManager()
        mirror = sb.ensure_mirror(repo_path, goal.project)
        existing_wt = parent_wf.state.get("staging_worktree_path")
        if existing_wt and (Path(str(existing_wt)) / ".git").exists():
            staging_wt = Path(str(existing_wt))
        else:
            staging_wt = sb.create_worktree(
                mirror,
                goal.project,
                hyp.branch_name,
                hyp.base_sha,
                origin_repo=repo_path,
                goal_id=goal.goal_id,
                workflow_id=parent_wf.workflow_id,
            )

        seen_shas: set[str] = set()
        for sr in slice_results:
            sha = str(sr.get("head_sha") or "").strip()
            if not sha or sha == hyp.base_sha or sha in seen_shas:
                continue
            seen_shas.add(sha)
            sb._ensure_commit_in_mirror(mirror, repo_path, sha)
            if _git(staging_wt, "merge-base", "--is-ancestor", sha, "HEAD").returncode == 0:
                continue
            m = _git(staging_wt, "merge", "--no-edit", sha)
            if m.returncode != 0:
                _git(staging_wt, "merge", "--abort")
                m2 = _git(staging_wt, "merge", "--no-edit", "-X", "theirs", sha)
                if m2.returncode != 0:
                    _git(staging_wt, "merge", "--abort")

        rev = _git(staging_wt, "rev-parse", "HEAD")
        staging_head = (
            rev.stdout.strip() if rev.returncode == 0 and rev.stdout.strip() else fallback_head
        )
        return str(staging_wt), staging_head

    def _compute_synthesis_source_snapshot(
        self,
        goal: ResearchGoal,
        synthesized_hypothesis_ids: Sequence[str],
    ) -> dict[str, dict[str, Any]]:
        """Capture exact lineage snapshot across ALL synthesized hypotheses for S3 dependency fencing."""
        snapshot: dict[str, dict[str, Any]] = {}
        for raw_hid in synthesized_hypothesis_ids:
            hid = str(raw_hid or "").strip()
            if not hid:
                continue
            sh = (
                self.db.get_goal_hypothesis(goal.goal_id, hid)
                if hasattr(self.db, "get_goal_hypothesis")
                else self.db.get_hypothesis(hid)
            )
            if sh is None or sh.goal_id != goal.goal_id:
                raise ValueError(
                    f"Synthesis hypothesis {hid!r} is missing or does not belong to goal {goal.goal_id!r}"
                )
            ep = self.db.get_latest_experiment_plan(hid) if hasattr(self.db, "get_latest_experiment_plan") else None
            auth_rcpts = (
                self.db.get_authoritative_empirical_receipts(goal, sh)
                if hasattr(self.db, "get_authoritative_empirical_receipts")
                else []
            )
            hparams = dict(sh.params or {})
            snapshot[hid] = {
                "status": sh.status,
                "head_sha": str(sh.head_sha or sh.base_sha or goal.baseline_sha or ""),
                "lineage_plan_epoch": int(hparams.get("lineage_plan_epoch", 0) or 0),
                "lineage_human_epoch": int(hparams.get("lineage_human_epoch", 0) or 0),
                "lineage_sci_revision": int(hparams.get("lineage_sci_revision", 0) or 0),
                "exp_plan_id": str(ep.exp_plan_id if ep is not None else ""),
                "exp_plan_version": int(ep.plan_version if ep is not None else 0),
                "authoritative_receipt_ids": sorted(r.receipt_id for r in auth_rcpts),
            }
        return snapshot

    def spawn_synthesis_review(
        self,
        goal: ResearchGoal,
        anchor_hyp: HypothesisNode,
        *,
        synthesized_hypothesis_ids: Sequence[str],
        synthesis_key: str,
        thesis: str,
        rationale: str = "",
        findings: Mapping[str, Any] | None = None,
        review_tier: ReviewTier | None = None,
    ) -> tuple[WorkflowNode, list[StageTask]]:
        """Spawn cross-experiment S3 `ReviewKind.SYNTHESIS_REVIEW` gating multi-hypothesis synthesis."""
        now = time.time()
        syn_hids = [str(x).strip() for x in synthesized_hypothesis_ids if str(x).strip()]
        if not syn_hids:
            syn_hids = [anchor_hyp.hypothesis_id]
        source_snapshot = self._compute_synthesis_source_snapshot(goal, syn_hids)
        eff_tier = review_tier or (
            ReviewTier.RED_TEAM_TIEBREAKER
            if str(goal.risk_profile).lower() in ("high", "critical") or len(syn_hids) >= 3
            else ReviewTier.STANDARD_CROSS_FAMILY
        )
        wf_id = f"wf_syn_{uuid.uuid4().hex[:10]}"
        wf = WorkflowNode(
            workflow_id=wf_id,
            goal_id=goal.goal_id,
            hypothesis_id=anchor_hyp.hypothesis_id,
            primitive="direct_hq",
            status="RUNNING",
            goal_version=int(goal.version),
            human_epoch=int(goal.human_epoch),
            plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
            budget_tokens=20000,
            config={
                "review_tier": eff_tier.value,
                "target_kind": "goal_synthesis",
                "synthesized_hypothesis_ids": syn_hids,
                "source_lineage_snapshot": source_snapshot,
                "synthesis_key": synthesis_key,
                "thesis": thesis,
                "rationale": rationale,
                "findings": dict(findings or {}),
            },
            state={"step": "synthesis_review", "completed_steps": []},
            created_ts=now,
            updated_ts=now,
        )
        wf = self.db.save_workflow(wf, now=now)
        self.db.upsert_knowledge_item(
            goal.goal_id,
            "synthesis_pending",
            synthesis_key,
            {
                "status": "PENDING_SYNTHESIS_REVIEW",
                "workflow_id": wf.workflow_id,
                "synthesized_hypothesis_ids": syn_hids,
                "source_lineage_snapshot": source_snapshot,
                "thesis": thesis,
                "rationale": rationale,
                "findings": dict(findings or {}),
            },
            hypothesis_id=anchor_hyp.hypothesis_id,
            goal_version=int(goal.version),
            now=now,
        )
        tasks = self._spawn_review_quorum_tasks(
            goal,
            anchor_hyp,
            wf,
            review_kind=ReviewKind.SYNTHESIS_REVIEW,
            review_tier=eff_tier,
            step_prefix="synthesis_review",
            next_step_on_pass="completed",
            head_sha=anchor_hyp.head_sha or anchor_hyp.base_sha or goal.baseline_sha,
            target_version=int(goal.version),
            completed_steps=[],
        )
        return self.db.get_workflow(wf.workflow_id) or wf, tasks

    def respawn_superseded_legacy_synthesis_workflows(
        self,
        goal_id: str | None = None,
    ) -> list[WorkflowNode]:
        """Enqueue snapshot-bound replacement `SYNTHESIS_REVIEW` workflows for migrated snapshot-less S3 workflows on `ACTIVE` goals."""
        spawned: list[WorkflowNode] = []
        active_goal_ids = {
            g.goal_id
            for g in (
                [self.db.get_goal(goal_id)]
                if goal_id
                else self.db.list_goals(status="ACTIVE")
            )
            if g is not None and (not goal_id or self.db.get_goal_status(g.goal_id) == "ACTIVE")
        }
        if not active_goal_ids:
            return spawned

        for gid in sorted(active_goal_ids):
            goal = self.db.get_goal(gid)
            if goal is None or goal.autonomy_mode == "paused":
                continue
            for wf in self.db.list_workflows(goal_id=gid):
                if wf.status != "SYNTHESIS_SUPERSEDED":
                    continue
                st = dict(wf.state or {})
                if st.get("replacement_workflow_id") or st.get("migration_respawn_attempted"):
                    continue
                stale_sources = list(st.get("stale_sources") or [])
                if not any("missing_source_lineage_snapshot" in str(s) for s in stale_sources):
                    continue
                cfg = dict(wf.config or {})
                syn_hids = [str(x) for x in (cfg.get("synthesized_hypothesis_ids") or []) if str(x)]
                anchor = (
                    self.db.get_goal_hypothesis(gid, wf.hypothesis_id)
                    if hasattr(self.db, "get_goal_hypothesis")
                    else self.db.get_hypothesis(wf.hypothesis_id)
                )
                if anchor is None and syn_hids:
                    anchor = self.db.get_hypothesis(syn_hids[0])
                if anchor is None:
                    self.db.update_workflow_state(
                        wf.workflow_id,
                        state_patch={"migration_respawn_attempted": True},
                    )
                    continue
                try:
                    new_wf, _ = self.spawn_synthesis_review(
                        goal,
                        anchor,
                        synthesized_hypothesis_ids=syn_hids or [anchor.hypothesis_id],
                        synthesis_key=str(cfg.get("synthesis_key") or f"syn_re_{wf.workflow_id[-6:]}"),
                        thesis=str(cfg.get("thesis") or goal.title),
                        rationale=str(cfg.get("rationale") or "Re-review migrated S3 synthesis with source lineage snapshot"),
                        findings=dict(cfg.get("findings") or {}),
                        review_tier=cfg.get("review_tier"),
                    )
                    self.db.update_workflow_state(
                        wf.workflow_id,
                        state_patch={
                            "replacement_workflow_id": new_wf.workflow_id,
                            "migration_respawn_attempted": True,
                        },
                    )
                    spawned.append(new_wf)
                except Exception as exc:
                    tries = int(st.get("respawn_transient_attempts", 0) or 0) + 1
                    self.db.update_workflow_state(
                        wf.workflow_id,
                        state_patch={
                            "respawn_transient_attempts": tries,
                            "last_respawn_error": str(exc),
                        },
                    )
        return spawned
