"""Core typed data contracts for conjectra."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, TypeVar

E = TypeVar("E", bound=Enum)


class StageKind(str, Enum):
    """Execution stage within a hypothesis pipeline."""

    PROPOSE = "PROPOSE"
    DESIGN_REVIEW = "DESIGN_REVIEW"
    CODE = "CODE"
    CPU_GATE = "CPU_GATE"
    IMPL_REVIEW = "IMPL_REVIEW"
    EXP_SCREEN = "EXP_SCREEN"
    EXP_CONFIRM = "EXP_CONFIRM"
    RESULT_REVIEW = "RESULT_REVIEW"
    BLIND_AUDIT = "BLIND_AUDIT"
    RECONCILE = "RECONCILE"


class ScientificRole(str, Enum):
    """First-class scientific or engineering role decoupled from model pool."""

    RESEARCH_DIRECTOR = "research_director"
    RESEARCH_LEAD = "research_lead"
    THEORIST = "theorist"
    EXPERIMENT_DESIGNER = "experiment_designer"
    ENGINEER_BUILDER = "engineer_builder"
    DESIGN_REVIEWER = "design_reviewer"
    IMPL_REVIEWER = "impl_reviewer"
    RESULT_AUDITOR = "result_auditor"


class FailureDomain(str, Enum):
    """Orthogonal failure domain isolating infra/env/code defects from empirical outcomes."""

    NONE = "NONE"
    INFRA_TRANSIENT = "INFRA_TRANSIENT"
    ENV_CONFIG_MISMATCH = "ENV_CONFIG_MISMATCH"
    ENGINEERING_DEFECT = "ENGINEERING_DEFECT"
    SCIENTIFIC_OUTCOME = "SCIENTIFIC_OUTCOME"


class ReviewKind(str, Enum):
    """Independent review stage category."""

    DESIGN_REVIEW = "DESIGN_REVIEW"
    IMPLEMENTATION_REVIEW = "IMPLEMENTATION_REVIEW"
    RESULT_REVIEW = "RESULT_REVIEW"
    SYNTHESIS_REVIEW = "SYNTHESIS_REVIEW"
    # Stage-semantics aliases (S1 pre-exec, S2 post-exec, S3 cross-experiment synthesis)
    S1_PRE_EXEC = "DESIGN_REVIEW"
    S2_POST_EXEC = "IMPLEMENTATION_REVIEW"
    S3_CROSS_EXPERIMENT = "SYNTHESIS_REVIEW"


class ReviewTier(str, Enum):
    """Risk/cost-adaptive review depth tier."""

    FAST_LANE = "FAST_LANE"
    STANDARD_CROSS_FAMILY = "STANDARD_CROSS_FAMILY"
    RED_TEAM_TIEBREAKER = "RED_TEAM_TIEBREAKER"


class ConstraintStrength(str, Enum):
    """Routing policy constraint enforcement mode."""

    HARD = "HARD"
    SOFT_FALLBACK = "SOFT_FALLBACK"


class ActionKind(str, Enum):
    """Typed closed-loop action emitted by ResearchLead or ResearchDirector."""

    CONTINUE = "CONTINUE"
    REVISE_HYPOTHESIS = "REVISE_HYPOTHESIS"
    PROPOSE_COMPETING_ROUTE = "PROPOSE_COMPETING_ROUTE"
    ADD_THEORETICAL_DERIVATION = "ADD_THEORETICAL_DERIVATION"
    ADD_DISCRIMINATING_EXPERIMENT = "ADD_DISCRIMINATING_EXPERIMENT"
    REPAIR_IMPLEMENTATION = "REPAIR_IMPLEMENTATION"
    PRUNE_BRANCH = "PRUNE_BRANCH"
    SYNTHESIZE_RESULTS = "SYNTHESIZE_RESULTS"
    SWITCH_WORKFLOW = "SWITCH_WORKFLOW"
    REQUEST_DIRECTOR_ADJUDICATION = "REQUEST_DIRECTOR_ADJUDICATION"
    TERMINATE_REFUTED_ROUTE = "TERMINATE_REFUTED_ROUTE"
    COMPLETE_MILESTONE = "COMPLETE_MILESTONE"
    REQUEST_OWNER_DECISION = "REQUEST_OWNER_DECISION"


class FidelityRung(str, Enum):
    """Multi-fidelity verification rung."""

    F0_SANITY = "F0_SANITY"
    F1_FAST_SCREEN = "F1_FAST_SCREEN"
    F2_FULL_CONFIRM = "F2_FULL_CONFIRM"
    F3_MILESTONE_AUDIT = "F3_MILESTONE_AUDIT"


class HypothesisKind(str, Enum):
    """Category of scientific or engineering hypothesis."""

    COMPETING_ROUTE = "COMPETING_ROUTE"
    REFINEMENT = "REFINEMENT"
    ABLATION = "ABLATION"
    FALSIFICATION = "FALSIFICATION"
    PARAMETER_RACE = "PARAMETER_RACE"


class ProcessStatus(str, Enum):
    """Low-level process execution outcome."""

    EXIT_OK = "EXIT_OK"
    EXIT_NONZERO = "EXIT_NONZERO"
    TIMEOUT = "TIMEOUT"
    OOM = "OOM"
    LEASE_LOST = "LEASE_LOST"
    ENV_ERROR = "ENV_ERROR"
    CANCELLED = "CANCELLED"


class EngineeringVerdict(str, Enum):
    """Deterministic engineering gate/test outcome."""

    PASS = "PASS"
    FAIL_GATE = "FAIL_GATE"
    FAIL_TEST = "FAIL_TEST"
    SKIPPED = "SKIPPED"


class ScientificVerdict(str, Enum):
    """Empirical scientific outcome against hypothesis criteria."""

    CONFIRMED = "CONFIRMED"
    REFUTED = "REFUTED"
    VALID_NEGATIVE = "VALID_NEGATIVE"
    UNKNOWN = "UNKNOWN"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AuditVerdict(str, Enum):
    """Independent blind-audit verdict."""

    NOT_REQUIRED = "NOT_REQUIRED"
    PASS = "PASS"
    FAIL = "FAIL"
    PARSE_ERROR = "PARSE_ERROR"


class EvidenceValidity(str, Enum):
    """Validity classification for an EvidenceReceipt."""

    VALID_EMPIRICAL = "VALID_EMPIRICAL"
    VERIFIED = "VALID_EMPIRICAL"
    VALID_ENGINEERING_ONLY = "VALID_ENGINEERING_ONLY"
    MISSING_PRIMARY_METRIC = "MISSING_PRIMARY_METRIC"
    MISSING_INPUT = "MISSING_INPUT"
    INVALID_SYNTHETIC = "INVALID_SYNTHETIC"
    STALE_GOAL_VERSION = "STALE_GOAL_VERSION"
    HUMAN_INVALIDATED = "HUMAN_INVALIDATED"
    SUPERSEDED_EPOCH = "SUPERSEDED_EPOCH"


def _coerce_enum(enum_cls: type[E], value: E | str) -> E:
    if isinstance(value, enum_cls):
        return value
    return enum_cls(value)


def _serialize_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, list):
        return [_serialize_value(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    return value


def default_hypothesis_budget_state() -> dict[str, Any]:
    """Return default bounded repair, environment recheck, and triage budgets."""
    return {
        "code_repairs_used": 0,
        "max_code_repairs": 2,
        "env_rechecks_used": 0,
        "max_env_rechecks": 1,
        "last_env_signature": "",
        "triage_decisions_used": 0,
        "max_triage_decisions": 2,
    }


@dataclass
class ResourceSpec:
    """Compute and quota resource specification for a stage task."""

    cpus: int = 1
    ram_mb: int = 1024
    gpu: int = 0
    llm_pool: str | None = None
    eda_sim: int = 0
    cloud_slot: int = 0
    wall_timeout_s: int = 900
    hq_llm_resource: str | None = None
    pinned_route: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        out: dict[str, Any] = {
            "cpus": int(self.cpus),
            "ram_mb": int(self.ram_mb),
            "gpu": int(self.gpu),
            "llm_pool": self.llm_pool,
            "eda_sim": int(self.eda_sim),
            "cloud_slot": int(self.cloud_slot),
            "wall_timeout_s": int(self.wall_timeout_s),
        }
        if self.hq_llm_resource is not None:
            out["hq_llm_resource"] = str(self.hq_llm_resource)
        if self.pinned_route is not None:
            out["pinned_route"] = _serialize_value(dict(self.pinned_route))
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResourceSpec:
        """Deserialize from a dictionary."""
        raw_hq_res = data.get("hq_llm_resource")
        raw_route = data.get("pinned_route")
        return cls(
            cpus=int(data.get("cpus", 1)),
            ram_mb=int(data.get("ram_mb", 1024)),
            gpu=int(data.get("gpu", 0)),
            llm_pool=data.get("llm_pool"),
            eda_sim=int(data.get("eda_sim", 0)),
            cloud_slot=int(data.get("cloud_slot", 0)),
            wall_timeout_s=int(data.get("wall_timeout_s", 900)),
            hq_llm_resource=None if raw_hq_res is None else str(raw_hq_res),
            pinned_route=dict(raw_route) if isinstance(raw_route, Mapping) else None,
        )


@dataclass
class ResearchGoal:
    """Top-level research goal, baseline, constraints, and budgets."""

    goal_id: str
    project: str
    title: str
    direction: str
    primary_metric: str
    baseline_value: float
    baseline_sha: str
    repo_path: str
    kill_criteria: list[str]
    banned_paths: list[str]
    writable_prefixes: list[str]
    max_tokens_budget: int
    max_concurrent_branches: int = 4
    adapter_name: str = "generic"
    version: int = 1
    human_epoch: int = 0
    plan_epoch: int = 0
    autonomy_mode: str = "unattended"
    priority_weight: float = 1.0
    min_guaranteed_slots: int = 1
    target_value: float | None = None
    wall_budget_sec: int = 21600
    routing_policy_nl: str = ""
    routing_policy: dict[str, Any] = field(default_factory=dict)
    risk_profile: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "goal_id": self.goal_id,
            "project": self.project,
            "title": self.title,
            "direction": self.direction,
            "primary_metric": self.primary_metric,
            "baseline_value": float(self.baseline_value),
            "baseline_sha": self.baseline_sha,
            "repo_path": self.repo_path,
            "kill_criteria": list(self.kill_criteria),
            "banned_paths": list(self.banned_paths),
            "writable_prefixes": list(self.writable_prefixes),
            "max_tokens_budget": int(self.max_tokens_budget),
            "max_concurrent_branches": int(self.max_concurrent_branches),
            "adapter_name": self.adapter_name,
            "version": int(self.version),
            "human_epoch": int(self.human_epoch),
            "plan_epoch": int(self.plan_epoch),
            "autonomy_mode": self.autonomy_mode,
            "priority_weight": float(self.priority_weight),
            "min_guaranteed_slots": int(self.min_guaranteed_slots),
            "target_value": None if self.target_value is None else float(self.target_value),
            "wall_budget_sec": int(self.wall_budget_sec),
            "routing_policy_nl": self.routing_policy_nl,
            "routing_policy": _serialize_value(dict(self.routing_policy)),
            "risk_profile": self.risk_profile,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResearchGoal:
        """Deserialize from a dictionary."""
        raw_target = data.get("target_value")
        raw_policy = data.get("routing_policy")
        return cls(
            goal_id=str(data["goal_id"]),
            project=str(data["project"]),
            title=str(data["title"]),
            direction=str(data["direction"]),
            primary_metric=str(data["primary_metric"]),
            baseline_value=float(data["baseline_value"]),
            baseline_sha=str(data["baseline_sha"]),
            repo_path=str(data["repo_path"]),
            kill_criteria=[str(x) for x in data.get("kill_criteria", [])],
            banned_paths=[str(x) for x in data.get("banned_paths", [])],
            writable_prefixes=[str(x) for x in data.get("writable_prefixes", [])],
            max_tokens_budget=int(data["max_tokens_budget"]),
            max_concurrent_branches=int(data.get("max_concurrent_branches", 4)),
            adapter_name=str(data.get("adapter_name", "generic")),
            version=int(data.get("version", 1)),
            human_epoch=int(data.get("human_epoch", 0)),
            plan_epoch=int(data.get("plan_epoch", 0)),
            autonomy_mode=str(data.get("autonomy_mode", "unattended")),
            priority_weight=float(data.get("priority_weight", 1.0)),
            min_guaranteed_slots=int(data.get("min_guaranteed_slots", 1)),
            target_value=None if raw_target is None else float(raw_target),
            wall_budget_sec=int(data.get("wall_budget_sec", 21600)),
            routing_policy_nl=str(data.get("routing_policy_nl", "")),
            routing_policy=dict(raw_policy) if isinstance(raw_policy, Mapping) else {},
            risk_profile=str(data.get("risk_profile", "standard")),
        )


@dataclass
class WorkflowNode:
    """Workflow execution node in the 4-level Goal -> Hypothesis -> Workflow -> Task tree."""

    workflow_id: str
    goal_id: str
    hypothesis_id: str
    parent_workflow_id: str | None = None
    primitive: str = "direct_hq"
    status: str = "READY"
    goal_version: int = 1
    human_epoch: int = 0
    plan_epoch: int = 0
    budget_tokens: int = 50000
    config: dict[str, Any] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    created_ts: float = 0.0
    updated_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "parent_workflow_id": self.parent_workflow_id,
            "primitive": self.primitive,
            "status": self.status,
            "goal_version": int(self.goal_version),
            "human_epoch": int(self.human_epoch),
            "plan_epoch": int(self.plan_epoch),
            "budget_tokens": int(self.budget_tokens),
            "config": _serialize_value(dict(self.config)),
            "state": _serialize_value(dict(self.state)),
            "created_ts": float(self.created_ts),
            "updated_ts": float(self.updated_ts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> WorkflowNode:
        """Deserialize from a dictionary."""
        return cls(
            workflow_id=str(data["workflow_id"]),
            goal_id=str(data["goal_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            parent_workflow_id=data.get("parent_workflow_id"),
            primitive=str(data.get("primitive", "direct_hq")),
            status=str(data.get("status", "READY")),
            goal_version=int(data.get("goal_version", 1)),
            human_epoch=int(data.get("human_epoch", 0)),
            plan_epoch=int(data.get("plan_epoch", 0)),
            budget_tokens=int(data.get("budget_tokens", 50000)),
            config=_serialize_value(dict(data.get("config", {}))),
            state=_serialize_value(dict(data.get("state", {}))),
            created_ts=float(data.get("created_ts", 0.0)),
            updated_ts=float(data.get("updated_ts", 0.0)),
        )


@dataclass
class HypothesisNode:
    """Node in the hypothesis exploration tree."""

    hypothesis_id: str
    goal_id: str
    parent_id: str | None
    kind: HypothesisKind
    title: str
    statement: str
    mechanism: str
    fidelity: FidelityRung
    priority_evoi: float
    writable_paths: list[str]
    params: dict[str, Any]
    kill_criteria: list[str]
    acceptance_cmd: str
    screen_cmd: str
    confirm_cmd: str | None
    branch_name: str
    base_sha: str
    head_sha: str | None = None
    status: str = "PENDING"
    scientific_verdict: ScientificVerdict = ScientificVerdict.UNKNOWN
    fingerprint: str = ""
    tags: list[str] = field(default_factory=list)
    uncertainty_id: str | None = None
    exp_plan_id: str | None = None
    failure_domain: FailureDomain = FailureDomain.NONE
    budget_state: dict[str, Any] = field(default_factory=default_hypothesis_budget_state)

    @property
    def routing_policy(self) -> dict[str, Any]:
        """Branch-scoped routing policy dictionary mirrored in `params['routing_policy']`."""
        raw = self.params.get("routing_policy") if isinstance(self.params, Mapping) else None
        return dict(raw) if isinstance(raw, Mapping) else {}

    @routing_policy.setter
    def routing_policy(self, value: Mapping[str, Any] | None) -> None:
        if value is None:
            self.params.pop("routing_policy", None)
        else:
            self.params["routing_policy"] = dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        merged_budget = {
            **default_hypothesis_budget_state(),
            **dict(self.budget_state or {}),
        }
        return {
            "hypothesis_id": self.hypothesis_id,
            "goal_id": self.goal_id,
            "parent_id": self.parent_id,
            "kind": _coerce_enum(HypothesisKind, self.kind).value,
            "title": self.title,
            "statement": self.statement,
            "mechanism": self.mechanism,
            "fidelity": _coerce_enum(FidelityRung, self.fidelity).value,
            "priority_evoi": float(self.priority_evoi),
            "writable_paths": list(self.writable_paths),
            "params": _serialize_value(dict(self.params)),
            "kill_criteria": list(self.kill_criteria),
            "acceptance_cmd": self.acceptance_cmd,
            "screen_cmd": self.screen_cmd,
            "confirm_cmd": self.confirm_cmd,
            "branch_name": self.branch_name,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "status": self.status,
            "scientific_verdict": _coerce_enum(
                ScientificVerdict, self.scientific_verdict
            ).value,
            "fingerprint": self.fingerprint,
            "tags": [str(t) for t in self.tags],
            "uncertainty_id": self.uncertainty_id,
            "exp_plan_id": self.exp_plan_id,
            "failure_domain": _coerce_enum(
                FailureDomain, self.failure_domain
            ).value,
            "budget_state": _serialize_value(merged_budget),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HypothesisNode:
        """Deserialize from a dictionary."""
        raw_budget = data.get("budget_state")
        merged_budget = {
            **default_hypothesis_budget_state(),
            **(dict(raw_budget) if isinstance(raw_budget, Mapping) else {}),
        }
        parsed_params = dict(data.get("params", {}))
        if isinstance(data.get("routing_policy"), Mapping) and "routing_policy" not in parsed_params:
            parsed_params["routing_policy"] = dict(data["routing_policy"])
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            goal_id=str(data["goal_id"]),
            parent_id=data.get("parent_id"),
            kind=_coerce_enum(HypothesisKind, data["kind"]),
            title=str(data["title"]),
            statement=str(data["statement"]),
            mechanism=str(data["mechanism"]),
            fidelity=_coerce_enum(FidelityRung, data["fidelity"]),
            priority_evoi=float(data["priority_evoi"]),
            writable_paths=[str(x) for x in data.get("writable_paths", [])],
            params=parsed_params,
            kill_criteria=[str(x) for x in data.get("kill_criteria", [])],
            acceptance_cmd=str(data["acceptance_cmd"]),
            screen_cmd=str(data["screen_cmd"]),
            confirm_cmd=data.get("confirm_cmd"),
            branch_name=str(data["branch_name"]),
            base_sha=str(data["base_sha"]),
            head_sha=data.get("head_sha"),
            status=str(data.get("status", "PENDING")),
            scientific_verdict=_coerce_enum(
                ScientificVerdict,
                data.get("scientific_verdict", ScientificVerdict.UNKNOWN),
            ),
            fingerprint=str(data.get("fingerprint", "")),
            tags=[str(x) for x in data.get("tags", [])],
            uncertainty_id=data.get("uncertainty_id"),
            exp_plan_id=data.get("exp_plan_id"),
            failure_domain=_coerce_enum(
                FailureDomain, data.get("failure_domain", FailureDomain.NONE)
            ),
            budget_state=merged_budget,
        )


@dataclass
class StageTask:
    """Schedulable stage unit for a hypothesis."""

    task_id: str
    idempotency_key: str
    goal_id: str
    hypothesis_id: str
    stage: StageKind
    deps: list[str]
    resources: ResourceSpec = field(default_factory=ResourceSpec)
    hq_job_id: int | None = None
    attempt: int = 1
    max_attempts: int = 2
    status: str = "READY"
    worktree_path: str | None = None
    enqueued_ts: float = 0.0
    started_ts: float | None = None
    finished_ts: float | None = None
    workflow_id: str | None = None
    goal_version: int = 1
    human_epoch: int = 0
    plan_epoch: int = 0
    workflow_primitive: str = "direct_hq"
    step_key: str = ""
    pinned_route: dict[str, Any] | None = None
    model_spec: str = ""
    hq_llm_resource: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        res = (
            self.resources
            if isinstance(self.resources, ResourceSpec)
            else ResourceSpec.from_dict(self.resources)
        )
        route_dict = (
            _serialize_value(dict(self.pinned_route))
            if isinstance(self.pinned_route, Mapping)
            else (
                _serialize_value(dict(res.pinned_route))
                if isinstance(res.pinned_route, Mapping)
                else None
            )
        )
        eff_model_spec = self.model_spec
        if not eff_model_spec and isinstance(route_dict, Mapping):
            pool = str(route_dict.get("pool") or "").strip()
            model = str(route_dict.get("canonical_model") or "").strip()
            effort = str(route_dict.get("requested_effort") or "").strip()
            if pool and model and effort:
                eff_model_spec = f"{pool}:{model}:{effort}"
            elif pool and model:
                eff_model_spec = f"{pool}:{model}"
        eff_hq_res = self.hq_llm_resource
        if not eff_hq_res:
            if res.hq_llm_resource:
                eff_hq_res = str(res.hq_llm_resource)
            elif isinstance(route_dict, Mapping) and route_dict.get("hq_resource"):
                eff_hq_res = str(route_dict["hq_resource"])
        return {
            "task_id": self.task_id,
            "idempotency_key": self.idempotency_key,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "stage": _coerce_enum(StageKind, self.stage).value,
            "deps": list(self.deps),
            "resources": res.to_dict(),
            "hq_job_id": self.hq_job_id,
            "attempt": int(self.attempt),
            "max_attempts": int(self.max_attempts),
            "status": self.status,
            "worktree_path": self.worktree_path,
            "enqueued_ts": float(self.enqueued_ts),
            "started_ts": None if self.started_ts is None else float(self.started_ts),
            "finished_ts": None
            if self.finished_ts is None
            else float(self.finished_ts),
            "workflow_id": self.workflow_id,
            "goal_version": int(self.goal_version),
            "human_epoch": int(self.human_epoch),
            "plan_epoch": int(self.plan_epoch),
            "workflow_primitive": self.workflow_primitive,
            "step_key": self.step_key,
            "pinned_route": route_dict,
            "model_spec": eff_model_spec,
            "hq_llm_resource": eff_hq_res,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> StageTask:
        """Deserialize from a dictionary."""
        raw_res = data.get("resources")
        if raw_res is None:
            resources = ResourceSpec()
        elif isinstance(raw_res, ResourceSpec):
            resources = raw_res
        else:
            resources = ResourceSpec.from_dict(raw_res)
        raw_hq_job = data.get("hq_job_id")
        raw_started = data.get("started_ts")
        raw_finished = data.get("finished_ts")
        raw_route = data.get("pinned_route")
        pinned_route = (
            dict(raw_route)
            if isinstance(raw_route, Mapping)
            else (dict(resources.pinned_route) if isinstance(resources.pinned_route, Mapping) else None)
        )
        return cls(
            task_id=str(data["task_id"]),
            idempotency_key=str(data["idempotency_key"]),
            goal_id=str(data["goal_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            stage=_coerce_enum(StageKind, data["stage"]),
            deps=[str(x) for x in data.get("deps", [])],
            resources=resources,
            hq_job_id=None if raw_hq_job is None else int(raw_hq_job),
            attempt=int(data.get("attempt", 1)),
            max_attempts=int(data.get("max_attempts", 2)),
            status=str(data.get("status", "READY")),
            worktree_path=data.get("worktree_path"),
            enqueued_ts=float(data.get("enqueued_ts", 0.0)),
            started_ts=None if raw_started is None else float(raw_started),
            finished_ts=None if raw_finished is None else float(raw_finished),
            workflow_id=data.get("workflow_id"),
            goal_version=int(data.get("goal_version", 1)),
            human_epoch=int(data.get("human_epoch", 0)),
            plan_epoch=int(
                data.get("plan_epoch", (pinned_route or {}).get("plan_epoch", 0)) or 0
            ),
            workflow_primitive=str(data.get("workflow_primitive", "direct_hq")),
            step_key=str(data.get("step_key", "")),
            pinned_route=pinned_route,
            model_spec=str(data.get("model_spec", "")),
            hq_llm_resource=str(data.get("hq_llm_resource", "")),
        )


@dataclass
class EvidenceReceipt:
    """Immutable execution evidence and verdict receipt."""

    receipt_id: str
    task_id: str
    goal_id: str
    hypothesis_id: str
    stage: StageKind
    base_sha: str
    head_sha: str
    command: str
    exit_code: int
    wall_sec: float
    max_rss_mb: float
    in_tokens: int
    out_tokens: int
    model_used: str
    process_status: ProcessStatus
    engineering_verdict: EngineeringVerdict
    scientific_verdict: ScientificVerdict
    audit_verdict: AuditVerdict
    tests_passed: int
    tests_total: int
    metrics: dict[str, Any]
    raw_artifact_paths: list[str]
    failure_reason: str = ""
    created_ts: float = 0.0
    evidence_validity: EvidenceValidity = EvidenceValidity.VALID_ENGINEERING_ONLY
    evidence_origin: str = "real_research_goal"
    goal_version: int = 1
    human_epoch: int = 0
    plan_epoch: int = 0
    workflow_id: str | None = None
    failure_domain: FailureDomain = FailureDomain.NONE
    effort_used: str = ""
    review_kind: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "receipt_id": self.receipt_id,
            "task_id": self.task_id,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "stage": _coerce_enum(StageKind, self.stage).value,
            "base_sha": self.base_sha,
            "head_sha": self.head_sha,
            "command": self.command,
            "exit_code": int(self.exit_code),
            "wall_sec": float(self.wall_sec),
            "max_rss_mb": float(self.max_rss_mb),
            "in_tokens": int(self.in_tokens),
            "out_tokens": int(self.out_tokens),
            "model_used": self.model_used,
            "process_status": _coerce_enum(ProcessStatus, self.process_status).value,
            "engineering_verdict": _coerce_enum(
                EngineeringVerdict, self.engineering_verdict
            ).value,
            "scientific_verdict": _coerce_enum(
                ScientificVerdict, self.scientific_verdict
            ).value,
            "audit_verdict": _coerce_enum(AuditVerdict, self.audit_verdict).value,
            "tests_passed": int(self.tests_passed),
            "tests_total": int(self.tests_total),
            "metrics": _serialize_value(dict(self.metrics)),
            "raw_artifact_paths": list(self.raw_artifact_paths),
            "failure_reason": self.failure_reason,
            "created_ts": float(self.created_ts),
            "evidence_validity": _coerce_enum(
                EvidenceValidity, self.evidence_validity
            ).value,
            "evidence_origin": self.evidence_origin,
            "goal_version": int(self.goal_version),
            "human_epoch": int(self.human_epoch),
            "plan_epoch": int(self.plan_epoch),
            "workflow_id": self.workflow_id,
            "failure_domain": _coerce_enum(
                FailureDomain, self.failure_domain
            ).value,
            "effort_used": self.effort_used,
            "review_kind": self.review_kind,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvidenceReceipt:
        """Deserialize from a dictionary."""
        return cls(
            receipt_id=str(data["receipt_id"]),
            task_id=str(data["task_id"]),
            goal_id=str(data["goal_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            stage=_coerce_enum(StageKind, data["stage"]),
            base_sha=str(data["base_sha"]),
            head_sha=str(data["head_sha"]),
            command=str(data["command"]),
            exit_code=int(data["exit_code"]),
            wall_sec=float(data["wall_sec"]),
            max_rss_mb=float(data["max_rss_mb"]),
            in_tokens=int(data["in_tokens"]),
            out_tokens=int(data["out_tokens"]),
            model_used=str(data["model_used"]),
            process_status=_coerce_enum(ProcessStatus, data["process_status"]),
            engineering_verdict=_coerce_enum(
                EngineeringVerdict, data["engineering_verdict"]
            ),
            scientific_verdict=_coerce_enum(
                ScientificVerdict, data["scientific_verdict"]
            ),
            audit_verdict=_coerce_enum(AuditVerdict, data["audit_verdict"]),
            tests_passed=int(data["tests_passed"]),
            tests_total=int(data["tests_total"]),
            metrics=_serialize_value(dict(data.get("metrics", {}))),
            raw_artifact_paths=[str(x) for x in data.get("raw_artifact_paths", [])],
            failure_reason=str(data.get("failure_reason", "")),
            created_ts=float(data.get("created_ts", 0.0)),
            evidence_validity=_coerce_enum(
                EvidenceValidity,
                data.get("evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY),
            ),
            evidence_origin=str(data.get("evidence_origin", "real_research_goal")),
            goal_version=int(data.get("goal_version", 1)),
            human_epoch=int(data.get("human_epoch", 0)),
            plan_epoch=int(data.get("plan_epoch", 0)),
            workflow_id=data.get("workflow_id"),
            failure_domain=_coerce_enum(
                FailureDomain, data.get("failure_domain", FailureDomain.NONE)
            ),
            effort_used=str(data.get("effort_used", "")),
            review_kind=str(data.get("review_kind", "")),
        )


# ------------------------------------------------ V3 Routing, Plans & Reviews


@dataclass
class ModelPolicy:
    """Compiled structured model & effort routing policy for a ResearchGoal."""

    source_nl: str = ""
    project: str = ""
    rules: list[dict[str, Any]] = field(default_factory=list)
    review_overrides: dict[str, str] = field(default_factory=dict)
    default_strength: ConstraintStrength = ConstraintStrength.SOFT_FALLBACK
    allow_effort_alias: bool = False
    foreground_reserve_quota_pct: float = 15.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "source_nl": self.source_nl,
            "project": self.project,
            "rules": _serialize_value(list(self.rules)),
            "review_overrides": {
                str(k): str(v) for k, v in self.review_overrides.items()
            },
            "default_strength": _coerce_enum(
                ConstraintStrength, self.default_strength
            ).value,
            "allow_effort_alias": bool(self.allow_effort_alias),
            "foreground_reserve_quota_pct": float(self.foreground_reserve_quota_pct),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ModelPolicy:
        """Deserialize from a dictionary."""
        return cls(
            source_nl=str(data.get("source_nl", "")),
            project=str(data.get("project", "")),
            rules=[dict(r) for r in data.get("rules", []) if isinstance(r, Mapping)],
            review_overrides={
                str(k): str(v)
                for k, v in dict(data.get("review_overrides") or {}).items()
            },
            default_strength=_coerce_enum(
                ConstraintStrength,
                data.get("default_strength", ConstraintStrength.SOFT_FALLBACK),
            ),
            allow_effort_alias=bool(data.get("allow_effort_alias", False)),
            foreground_reserve_quota_pct=float(
                data.get("foreground_reserve_quota_pct", 15.0)
            ),
        )


@dataclass(frozen=True)
class ResolvedModelRoute:
    """Immutable capability-verified model & effort route pinned once per attempt."""

    backend_mode: str
    execution_host: str
    multipool_host_arg: str
    pool: str
    canonical_model: str
    requested_effort: str
    wire_effort: str
    family: str
    logical_pool: str
    hq_resource: str | None = None
    strength: ConstraintStrength = ConstraintStrength.SOFT_FALLBACK
    matched_rule_id: str = "default"
    allow_effort_alias: bool = False

    def canonical_spec(self) -> str:
        """Return canonical `pool:canonical_model:requested_effort` identifier."""
        return f"{self.pool}:{self.canonical_model}:{self.requested_effort}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "backend_mode": self.backend_mode,
            "execution_host": self.execution_host,
            "multipool_host_arg": self.multipool_host_arg,
            "pool": self.pool,
            "canonical_model": self.canonical_model,
            "requested_effort": self.requested_effort,
            "wire_effort": self.wire_effort,
            "family": self.family,
            "logical_pool": self.logical_pool,
            "hq_resource": self.hq_resource,
            "strength": _coerce_enum(ConstraintStrength, self.strength).value,
            "matched_rule_id": self.matched_rule_id,
            "allow_effort_alias": bool(self.allow_effort_alias),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResolvedModelRoute:
        """Deserialize from a dictionary."""
        raw_hq = data.get("hq_resource")
        return cls(
            backend_mode=str(data["backend_mode"]),
            execution_host=str(data.get("execution_host", "local")),
            multipool_host_arg=str(data.get("multipool_host_arg", "local")),
            pool=str(data["pool"]),
            canonical_model=str(data["canonical_model"]),
            requested_effort=str(data["requested_effort"]),
            wire_effort=str(data.get("wire_effort", data["requested_effort"])),
            family=str(data["family"]),
            logical_pool=str(data.get("logical_pool", "")),
            hq_resource=None if raw_hq is None else str(raw_hq),
            strength=_coerce_enum(
                ConstraintStrength,
                data.get("strength", ConstraintStrength.SOFT_FALLBACK),
            ),
            matched_rule_id=str(data.get("matched_rule_id", "default")),
            allow_effort_alias=bool(data.get("allow_effort_alias", False)),
        )


@dataclass
class ResearchPlan:
    """Versioned goal-level research thesis, explicit uncertainties, and routes."""

    plan_id: str
    goal_id: str
    plan_version: int = 1
    goal_version: int = 1
    human_epoch: int = 0
    plan_epoch: int = 0
    thesis: str = ""
    uncertainties: list[dict[str, Any]] = field(default_factory=list)
    competing_routes: list[dict[str, Any]] = field(default_factory=list)
    milestone_criteria: list[str] = field(default_factory=list)
    status: str = "DRAFT"
    created_ts: float = 0.0
    updated_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "plan_version": int(self.plan_version),
            "goal_version": int(self.goal_version),
            "human_epoch": int(self.human_epoch),
            "plan_epoch": int(self.plan_epoch),
            "thesis": self.thesis,
            "uncertainties": _serialize_value(list(self.uncertainties)),
            "competing_routes": _serialize_value(list(self.competing_routes)),
            "milestone_criteria": [str(x) for x in self.milestone_criteria],
            "status": self.status,
            "created_ts": float(self.created_ts),
            "updated_ts": float(self.updated_ts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ResearchPlan:
        """Deserialize from a dictionary."""
        return cls(
            plan_id=str(data["plan_id"]),
            goal_id=str(data["goal_id"]),
            plan_version=int(data.get("plan_version", 1)),
            goal_version=int(data.get("goal_version", 1)),
            human_epoch=int(data.get("human_epoch", 0)),
            plan_epoch=int(data.get("plan_epoch", 0)),
            thesis=str(data.get("thesis", "")),
            uncertainties=[
                dict(u) for u in data.get("uncertainties", []) if isinstance(u, Mapping)
            ],
            competing_routes=[
                dict(r)
                for r in data.get("competing_routes", [])
                if isinstance(r, Mapping)
            ],
            milestone_criteria=[str(x) for x in data.get("milestone_criteria", [])],
            status=str(data.get("status", "DRAFT")),
            created_ts=float(data.get("created_ts", 0.0)),
            updated_ts=float(data.get("updated_ts", 0.0)),
        )


@dataclass
class ExperimentPlan:
    """Versioned hypothesis-level empirical design, environment, and timeout contract."""

    exp_plan_id: str
    goal_id: str
    hypothesis_id: str
    plan_version: int = 1
    goal_version: int = 1
    human_epoch: int = 0
    plan_epoch: int = 0
    falsifiable_claim: str = ""
    control_baseline: dict[str, Any] = field(default_factory=dict)
    venv_python: str = ""
    env_vars: dict[str, str] = field(default_factory=dict)
    required_hardware: str = "cpu_only"
    min_vram_mb: int = 0
    gate_timeout_s: int = 90
    screen_timeout_s: int = 300
    confirm_timeout_s: int = 1200
    confound_checks: list[str] = field(default_factory=list)
    review_status: str = "PENDING"
    created_ts: float = 0.0
    updated_ts: float = 0.0

    def __post_init__(self) -> None:
        self.gate_timeout_s = min(max(30, int(self.gate_timeout_s or 90)), 180)
        self.screen_timeout_s = max(30, int(self.screen_timeout_s or 300))
        self.confirm_timeout_s = max(30, int(self.confirm_timeout_s or 1200))
        self.min_vram_mb = max(0, int(self.min_vram_mb or 0))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "exp_plan_id": self.exp_plan_id,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "plan_version": int(self.plan_version),
            "goal_version": int(self.goal_version),
            "human_epoch": int(self.human_epoch),
            "plan_epoch": int(self.plan_epoch),
            "falsifiable_claim": self.falsifiable_claim,
            "control_baseline": _serialize_value(dict(self.control_baseline)),
            "venv_python": self.venv_python,
            "env_vars": {str(k): str(v) for k, v in self.env_vars.items()},
            "required_hardware": self.required_hardware,
            "min_vram_mb": int(self.min_vram_mb),
            "gate_timeout_s": int(self.gate_timeout_s),
            "screen_timeout_s": int(self.screen_timeout_s),
            "confirm_timeout_s": int(self.confirm_timeout_s),
            "confound_checks": [str(x) for x in self.confound_checks],
            "review_status": self.review_status,
            "created_ts": float(self.created_ts),
            "updated_ts": float(self.updated_ts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExperimentPlan:
        """Deserialize from a dictionary."""
        return cls(
            exp_plan_id=str(data["exp_plan_id"]),
            goal_id=str(data["goal_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            plan_version=int(data.get("plan_version", 1)),
            goal_version=int(data.get("goal_version", 1)),
            human_epoch=int(data.get("human_epoch", 0)),
            plan_epoch=int(data.get("plan_epoch", 0)),
            falsifiable_claim=str(data.get("falsifiable_claim", "")),
            control_baseline=dict(data.get("control_baseline", {})),
            venv_python=str(data.get("venv_python", "")),
            env_vars={
                str(k): str(v) for k, v in dict(data.get("env_vars") or {}).items()
            },
            required_hardware=str(data.get("required_hardware", "cpu_only")),
            min_vram_mb=int(data.get("min_vram_mb", 0)),
            gate_timeout_s=int(data.get("gate_timeout_s", 90)),
            screen_timeout_s=int(data.get("screen_timeout_s", 300)),
            confirm_timeout_s=int(data.get("confirm_timeout_s", 1200)),
            confound_checks=[str(x) for x in data.get("confound_checks", [])],
            review_status=str(data.get("review_status", "PENDING")),
            created_ts=float(data.get("created_ts", 0.0)),
            updated_ts=float(data.get("updated_ts", 0.0)),
        )


@dataclass
class ExecutionPlan:
    """Workflow execution plan binding slices, role routes, review tier, and lineage."""

    exec_plan_id: str
    workflow_id: str
    goal_id: str
    hypothesis_id: str
    goal_version: int = 1
    human_epoch: int = 0
    plan_epoch: int = 0
    primitive: str = "direct_hq"
    review_tier: ReviewTier = ReviewTier.STANDARD_CROSS_FAMILY
    slices: list[dict[str, Any]] = field(default_factory=list)
    role_routes: dict[str, dict[str, Any]] = field(default_factory=dict)
    contributor_families: list[str] = field(default_factory=list)
    budget_tokens: int = 50000
    created_ts: float = 0.0
    updated_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "exec_plan_id": self.exec_plan_id,
            "workflow_id": self.workflow_id,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "goal_version": int(self.goal_version),
            "human_epoch": int(self.human_epoch),
            "plan_epoch": int(self.plan_epoch),
            "primitive": self.primitive,
            "review_tier": _coerce_enum(ReviewTier, self.review_tier).value,
            "slices": _serialize_value(list(self.slices)),
            "role_routes": _serialize_value(dict(self.role_routes)),
            "contributor_families": sorted(
                {str(f).strip().lower() for f in self.contributor_families if str(f).strip()}
            ),
            "budget_tokens": int(self.budget_tokens),
            "created_ts": float(self.created_ts),
            "updated_ts": float(self.updated_ts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ExecutionPlan:
        """Deserialize from a dictionary."""
        raw_routes = data.get("role_routes")
        return cls(
            exec_plan_id=str(data["exec_plan_id"]),
            workflow_id=str(data["workflow_id"]),
            goal_id=str(data["goal_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            goal_version=int(data.get("goal_version", 1)),
            human_epoch=int(data.get("human_epoch", 0)),
            plan_epoch=int(data.get("plan_epoch", 0)),
            primitive=str(data.get("primitive", "direct_hq")),
            review_tier=_coerce_enum(
                ReviewTier,
                data.get("review_tier", ReviewTier.STANDARD_CROSS_FAMILY),
            ),
            slices=[
                dict(s) for s in data.get("slices", []) if isinstance(s, Mapping)
            ],
            role_routes={
                str(k): dict(v)
                for k, v in (dict(raw_routes) if isinstance(raw_routes, Mapping) else {}).items()
                if isinstance(v, Mapping)
            },
            contributor_families=[
                str(f) for f in data.get("contributor_families", [])
            ],
            budget_tokens=int(data.get("budget_tokens", 50000)),
            created_ts=float(data.get("created_ts", 0.0)),
            updated_ts=float(data.get("updated_ts", 0.0)),
        )


@dataclass
class ReviewReceipt:
    """Typed receipt for Design, Implementation, Result, or Cross-Experiment Synthesis review."""

    review_id: str
    goal_id: str
    hypothesis_id: str
    workflow_id: str = ""
    task_id: str = ""
    review_kind: ReviewKind = ReviewKind.IMPLEMENTATION_REVIEW
    review_tier: ReviewTier = ReviewTier.STANDARD_CROSS_FAMILY
    target_version: int = 1
    plan_epoch: int = 0
    human_epoch: int = 0
    target_kind: str = "hypothesis"
    synthesized_hypothesis_ids: list[str] = field(default_factory=list)
    head_sha: str = ""
    slot_index: int = 0
    author_families: list[str] = field(default_factory=list)
    reviewer_route: dict[str, Any] = field(default_factory=dict)
    verdict: AuditVerdict = AuditVerdict.PARSE_ERROR
    process_status: ProcessStatus = ProcessStatus.EXIT_OK
    severity: str = "NONE"
    findings: list[str] = field(default_factory=list)
    repro_python: str = ""
    tiebreaker_used: bool = False
    raw_artifact_paths: list[str] = field(default_factory=list)
    in_tokens: int = 0
    out_tokens: int = 0
    created_ts: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "review_id": self.review_id,
            "goal_id": self.goal_id,
            "hypothesis_id": self.hypothesis_id,
            "workflow_id": self.workflow_id,
            "task_id": self.task_id,
            "review_kind": _coerce_enum(ReviewKind, self.review_kind).value,
            "review_tier": _coerce_enum(ReviewTier, self.review_tier).value,
            "target_version": int(self.target_version),
            "plan_epoch": int(self.plan_epoch),
            "human_epoch": int(self.human_epoch),
            "target_kind": self.target_kind,
            "synthesized_hypothesis_ids": [str(x) for x in self.synthesized_hypothesis_ids],
            "head_sha": self.head_sha,
            "slot_index": int(self.slot_index),
            "author_families": [str(f) for f in self.author_families],
            "reviewer_route": _serialize_value(dict(self.reviewer_route)),
            "verdict": _coerce_enum(AuditVerdict, self.verdict).value,
            "process_status": _coerce_enum(ProcessStatus, self.process_status).value,
            "severity": self.severity,
            "findings": [str(f) for f in self.findings],
            "repro_python": self.repro_python,
            "tiebreaker_used": bool(self.tiebreaker_used),
            "raw_artifact_paths": [str(p) for p in self.raw_artifact_paths],
            "in_tokens": int(self.in_tokens),
            "out_tokens": int(self.out_tokens),
            "created_ts": float(self.created_ts),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ReviewReceipt:
        """Deserialize from a dictionary."""
        raw_route = data.get("reviewer_route")
        return cls(
            review_id=str(data["review_id"]),
            goal_id=str(data["goal_id"]),
            hypothesis_id=str(data["hypothesis_id"]),
            workflow_id=str(data.get("workflow_id", "")),
            task_id=str(data.get("task_id", "")),
            review_kind=_coerce_enum(
                ReviewKind,
                data.get("review_kind", ReviewKind.IMPLEMENTATION_REVIEW),
            ),
            review_tier=_coerce_enum(
                ReviewTier,
                data.get("review_tier", ReviewTier.STANDARD_CROSS_FAMILY),
            ),
            target_version=int(data.get("target_version", 1)),
            plan_epoch=int(data.get("plan_epoch", 0)),
            human_epoch=int(data.get("human_epoch", 0)),
            target_kind=str(data.get("target_kind", "hypothesis")),
            synthesized_hypothesis_ids=[
                str(x) for x in data.get("synthesized_hypothesis_ids", [])
            ],
            head_sha=str(data.get("head_sha", "")),
            slot_index=int(data.get("slot_index", 0)),
            author_families=[str(f) for f in data.get("author_families", [])],
            reviewer_route=dict(raw_route) if isinstance(raw_route, Mapping) else {},
            verdict=_coerce_enum(
                AuditVerdict, data.get("verdict", AuditVerdict.PARSE_ERROR)
            ),
            process_status=_coerce_enum(
                ProcessStatus, data.get("process_status", ProcessStatus.EXIT_OK)
            ),
            severity=str(data.get("severity", "NONE")),
            findings=[str(f) for f in data.get("findings", [])],
            repro_python=str(data.get("repro_python", "")),
            tiebreaker_used=bool(data.get("tiebreaker_used", False)),
            raw_artifact_paths=[str(p) for p in data.get("raw_artifact_paths", [])],
            in_tokens=int(data.get("in_tokens", 0)),
            out_tokens=int(data.get("out_tokens", 0)),
            created_ts=float(data.get("created_ts", 0.0)),
        )


@dataclass
class LeadAction:
    """Single closed-loop scientific or engineering action emitted by a Lead/Director."""

    action: ActionKind
    target_hypothesis_id: str | None = None
    uncertainty_id: str | None = None
    rationale: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    workflow_primitive: str | None = None
    review_tier: ReviewTier | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "action": _coerce_enum(ActionKind, self.action).value,
            "target_hypothesis_id": self.target_hypothesis_id,
            "uncertainty_id": self.uncertainty_id,
            "rationale": self.rationale,
            "parameters": _serialize_value(dict(self.parameters)),
            "workflow_primitive": self.workflow_primitive,
            "review_tier": None
            if self.review_tier is None
            else _coerce_enum(ReviewTier, self.review_tier).value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LeadAction:
        """Deserialize from a dictionary."""
        raw_tier = data.get("review_tier")
        return cls(
            action=_coerce_enum(ActionKind, data["action"]),
            target_hypothesis_id=data.get("target_hypothesis_id"),
            uncertainty_id=data.get("uncertainty_id"),
            rationale=str(data.get("rationale", "")),
            parameters=dict(data.get("parameters", {})),
            workflow_primitive=data.get("workflow_primitive"),
            review_tier=None
            if raw_tier is None
            else _coerce_enum(ReviewTier, raw_tier),
        )


@dataclass
class LeadDecisionEnvelope:
    """Role-typed decision envelope returned by ResearchLead or ResearchDirector."""

    role: ScientificRole = ScientificRole.RESEARCH_LEAD
    goal_id: str = ""
    goal_version: int = 1
    human_epoch: int = 0
    uncertainty_summary: str = ""
    actions: list[LeadAction] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "role": _coerce_enum(ScientificRole, self.role).value,
            "goal_id": self.goal_id,
            "goal_version": int(self.goal_version),
            "human_epoch": int(self.human_epoch),
            "uncertainty_summary": self.uncertainty_summary,
            "actions": [a.to_dict() for a in self.actions],
            "metadata": _serialize_value(dict(self.metadata)),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> LeadDecisionEnvelope:
        """Deserialize from a dictionary."""
        raw_actions = data.get("actions", [])
        actions = [
            a if isinstance(a, LeadAction) else LeadAction.from_dict(a)
            for a in raw_actions
            if isinstance(a, (LeadAction, Mapping))
        ]
        return cls(
            role=_coerce_enum(
                ScientificRole, data.get("role", ScientificRole.RESEARCH_LEAD)
            ),
            goal_id=str(data.get("goal_id", "")),
            goal_version=int(data.get("goal_version", 1)),
            human_epoch=int(data.get("human_epoch", 0)),
            uncertainty_summary=str(data.get("uncertainty_summary", "")),
            actions=actions,
            metadata=dict(data.get("metadata", {})),
        )


LeadDecision = LeadDecisionEnvelope
