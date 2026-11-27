"""Hypothesis tree lifecycle, EVOI prioritization, anti-filler guard, and proposal ingestion."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any
import uuid

from conjectra.gates import (
    check_no_banned_access,
    classify_failure_domain,
    is_authenticated_negative_empirical_outcome,
    is_valid_empirical_outcome,
    map_timeout_verdict,
    norm_path,
    validate_writable_paths,
)
from conjectra.planning import PlanLifecycleManager
from conjectra.state_db import GraveyardDuplicateError
from conjectra.types import (
    ActionKind,
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    ExperimentPlan,
    FailureDomain,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    LeadDecisionEnvelope,
    ProcessStatus,
    ResearchGoal,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
)

READY_QUEUE_WATERMARK_FACTOR = 1.5
DEFAULT_MIN_EVOI = 0.02
DEFAULT_MAX_UNKNOWN_REFINEMENTS = 1
DEFAULT_DIMINISHING_RETURNS_STREAK = 4
DEFAULT_AI_MODEL = "gemini-pro"
MULTI_POOL_DISPATCH_PATH = os.path.expanduser(
    os.environ.get("CONJECTRA_DISPATCH_SCRIPT") or "~/.conjectra/dispatch.py"
)

# Extra path fragments proposals may never touch (besides goal.banned_paths). Extend with
# CONJECTRA_PROTECTED_PATHS="frag1,frag2" (read at call time).
def protected_repo_fragments() -> tuple[str, ...]:
    return tuple(
        x.strip().lower() for x in os.environ.get("CONJECTRA_PROTECTED_PATHS", "").split(",") if x.strip()
    )


EXTRA_BANNED_FRAGMENTS: tuple[str, ...] = ("prereg_r1.json",)

_KILL_EXPR_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(<=|>=|==|!=|<|>)\s*(-?\d+(?:\.\d+)?)\s*$"
)
_TOKEN_SPLIT_RE = re.compile(r"[A-Za-z0-9_]+")
_TERMINAL_TASK_STATES = frozenset({"DONE", "SUCCEEDED", "PASSED", "FAILED", "CANCELLED"})


@dataclass
class ValidationResult:
    """Outcome of validating a candidate hypothesis against ResearchGoal constraints."""

    valid: bool
    reasons: list[str] = field(default_factory=list)
    normalized_node: HypothesisNode | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation outcome."""
        return {
            "valid": self.valid,
            "reasons": list(self.reasons),
            "normalized_node": (
                None if self.normalized_node is None else self.normalized_node.to_dict()
            ),
        }


@dataclass
class ProposalBatchResult:
    """Result returned by propose_hypotheses / import_proposals_json."""

    accepted: list[HypothesisNode] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    stopped_reason: str | None = None
    in_tokens: int = 0
    out_tokens: int = 0
    model_used: str = ""
    source: str = "seeds"

    def to_dict(self) -> dict[str, Any]:
        """Serialize proposal batch result."""
        return {
            "accepted": [node.to_dict() for node in self.accepted],
            "rejected": list(self.rejected),
            "stopped_reason": self.stopped_reason,
            "in_tokens": int(self.in_tokens),
            "out_tokens": int(self.out_tokens),
            "model_used": self.model_used,
            "source": self.source,
        }


@dataclass
class ReconcileDecision:
    """Deterministic lifecycle transition produced when reconciling an EvidenceReceipt."""

    hypothesis_id: str
    previous_fidelity: FidelityRung
    new_fidelity: FidelityRung
    new_status: str
    scientific_verdict: ScientificVerdict
    action: str
    promoted_to: FidelityRung | None = None
    spawned_hypotheses: list[HypothesisNode] = field(default_factory=list)
    pruned_hypothesis_ids: list[str] = field(default_factory=list)
    cancelled_task_ids: list[str] = field(default_factory=list)
    graveyard_recorded: bool = False
    kill_criterion_hit: str | None = None
    escalate: bool = False
    escalation_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize reconciliation decision."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "previous_fidelity": self.previous_fidelity.value,
            "new_fidelity": self.new_fidelity.value,
            "new_status": self.new_status,
            "scientific_verdict": self.scientific_verdict.value,
            "action": self.action,
            "promoted_to": None if self.promoted_to is None else self.promoted_to.value,
            "spawned_hypotheses": [h.to_dict() for h in self.spawned_hypotheses],
            "pruned_hypothesis_ids": list(self.pruned_hypothesis_ids),
            "cancelled_task_ids": list(self.cancelled_task_ids),
            "graveyard_recorded": self.graveyard_recorded,
            "kill_criterion_hit": self.kill_criterion_hit,
            "escalate": self.escalate,
            "escalation_reason": self.escalation_reason,
        }


def normalize_path_str(path: str, repo_path: str = "") -> str:
    """Normalize a repository-relative path cleanly using gates.norm_path."""
    raw = str(path or "").strip().replace("\\", "/")
    if not raw:
        return ""
    if repo_path:
        repo_norm = repo_path.rstrip("/").replace("\\", "/")
        if raw == repo_norm:
            return "."
        if raw.startswith(repo_norm + "/"):
            raw = raw[len(repo_norm) + 1 :]
    normed = norm_path(raw)
    return "" if normed is None else normed


def normalize_params(value: Any) -> Any:
    """Canonicalize nested parameters for stable fingerprinting."""
    if isinstance(value, Mapping):
        return {str(k): normalize_params(value[k]) for k in sorted(value.keys(), key=str)}
    if isinstance(value, (list, tuple)):
        return [normalize_params(v) for v in value]
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        rounded = round(value, 8)
        return int(rounded) if rounded.is_integer() else rounded
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return str(value)


def compute_fingerprint(
    kind: HypothesisKind | str,
    mechanism: str,
    params: Mapping[str, Any],
    writable_paths: Sequence[str],
    project: str = "",
) -> str:
    """Compute a deterministic normalized SHA-256 fingerprint for deduplication and graveyard lookup."""
    kind_val = kind.value if isinstance(kind, HypothesisKind) else HypothesisKind(kind).value
    norm_paths = sorted(
        {normalize_path_str(p) for p in writable_paths if normalize_path_str(p)}
    )
    payload = {
        "project": project.strip().lower(),
        "kind": kind_val,
        "mechanism": " ".join(str(mechanism or "").strip().lower().split()),
        "params": normalize_params(dict(params or {})),
        "writable_paths": norm_paths,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _db_list_hypotheses(db: Any, goal_id: str) -> list[HypothesisNode]:
    if db is None:
        return []
    fn = getattr(db, "list_hypotheses", None)
    if callable(fn):
        rows = fn(goal_id=goal_id)
        return [r if isinstance(r, HypothesisNode) else HypothesisNode.from_dict(r) for r in rows]
    return []


def _db_register_or_update_hypothesis(
    db: Any,
    hyp: HypothesisNode,
    *,
    prev_status: str | None = None,
) -> None:
    if db is None:
        return
    get_fn = getattr(db, "get_hypothesis", None)
    existing = get_fn(hyp.hypothesis_id) if callable(get_fn) else None
    if existing is None:
        reg_fn = getattr(db, "register_hypothesis", None)
        if callable(reg_fn):
            try:
                reg_fn(hyp)
            except GraveyardDuplicateError:
                return
            return
    trans_fn = getattr(db, "transition_hypothesis_status", None)
    if callable(trans_fn) and existing is not None:
        expected = list({existing.status, prev_status or existing.status, hyp.status})
        trans_fn(
            hyp.hypothesis_id,
            expected,
            hyp.status,
            head_sha=hyp.head_sha,
            scientific_verdict=hyp.scientific_verdict,
            fidelity=hyp.fidelity,
            failure_domain=hyp.failure_domain,
            budget_state_patch=hyp.budget_state,
            exp_plan_id=hyp.exp_plan_id,
        )


def _db_is_in_graveyard(db: Any, fingerprint: str, goal_id: str) -> bool:
    if db is None or not fingerprint:
        return False
    chk_fn = getattr(db, "check_duplicate_fingerprint", None)
    if callable(chk_fn):
        hit = chk_fn(goal_id, fingerprint)
        return bool(hit and hit.get("source") == "graveyard")
    return fingerprint in _db_graveyard_fingerprints(db, goal_id)


def _db_graveyard_fingerprints(db: Any, goal_id: str) -> set[str]:
    if db is None:
        return set()
    fn_list = getattr(db, "list_graveyard", None)
    if not callable(fn_list):
        return set()
    entries = fn_list(goal_id=goal_id)
    return {
        str(e["fingerprint"]).strip().lower()
        for e in entries
        if isinstance(e, Mapping) and e.get("fingerprint")
    }


def _db_record_graveyard(
    db: Any,
    hypothesis: HypothesisNode,
    reason: str,
    evidence: Mapping[str, Any],
    *,
    scientific_verdict: ScientificVerdict = ScientificVerdict.REFUTED,
    new_status: str = "REFUTED",
) -> bool:
    if db is None:
        return False
    fn = getattr(db, "record_graveyard", None)
    if not callable(fn):
        return False
    get_fn = getattr(db, "get_hypothesis", None)
    if callable(get_fn) and get_fn(hypothesis.hypothesis_id) is None:
        reg_fn = getattr(db, "register_hypothesis", None)
        if callable(reg_fn):
            try:
                reg_fn(hypothesis, allow_graveyard=True)
            except Exception:
                pass
    fn(
        hypothesis.hypothesis_id,
        reason,
        scientific_verdict=scientific_verdict,
        details=dict(evidence),
        new_status=new_status,
    )
    return True


def _db_record_valid_negative_knowledge(
    db: Any,
    goal: ResearchGoal,
    hypothesis: HypothesisNode,
    receipt: EvidenceReceipt,
    reason: str,
) -> None:
    if db is None:
        return
    fn = getattr(db, "upsert_knowledge_item", None)
    if not callable(fn):
        return
    metric_val = None
    if isinstance(receipt.metrics, Mapping):
        raw_m = receipt.metrics.get(goal.primary_metric)
        if isinstance(raw_m, Mapping) and "value" in raw_m:
            metric_val = raw_m["value"]
        elif isinstance(raw_m, (int, float)):
            metric_val = raw_m
    content = {
        "hypothesis_id": hypothesis.hypothesis_id,
        "parent_id": hypothesis.parent_id,
        "generation": int(hypothesis.params.get("generation", 1)) if isinstance(hypothesis.params, Mapping) else 1,
        "mechanism": hypothesis.mechanism,
        "primary_metric": goal.primary_metric,
        "measured_value": metric_val,
        "baseline_value": goal.baseline_value,
        "root_cause": reason,
        "receipt_id": receipt.receipt_id,
        "stage": receipt.stage.value,
    }
    try:
        fn(
            goal.goal_id,
            "valid_negative",
            f"hyp:{hypothesis.hypothesis_id}",
            content,
            hypothesis_id=hypothesis.hypothesis_id,
            goal_version=int(getattr(goal, "version", 1)),
            confidence=1.0,
        )
    except Exception:
        pass


def _db_cancel_hypothesis_tasks(db: Any, hypothesis_id: str, goal_id: str) -> list[str]:
    if db is None:
        return []
    list_fn = getattr(db, "list_tasks", None)
    trans_fn = getattr(db, "transition_task_status", None)
    if not callable(list_fn) or not callable(trans_fn):
        return []
    cancelled: list[str] = []
    now = time.time()
    for task in list_fn(goal_id=goal_id, hypothesis_id=hypothesis_id):
        if task.status not in _TERMINAL_TASK_STATES:
            trans_fn(task.task_id, task.status, "CANCELLED", finished_ts=now, now=now)
            cancelled.append(task.task_id)
    return cancelled


def _db_used_tokens(db: Any, goal_id: str) -> int:
    if db is None:
        return 0
    fn = getattr(db, "get_goal_token_usage", None)
    if callable(fn):
        res = fn(goal_id)
        if isinstance(res, Mapping):
            return int(res.get("total_tokens", 0))
    return 0


def _db_list_receipts(
    db: Any,
    goal_id: str,
    hypothesis_id: str | None = None,
) -> list[EvidenceReceipt]:
    if db is None:
        return []
    fn = getattr(db, "list_receipts", None)
    if not callable(fn):
        return []
    rows = fn(goal_id=goal_id, hypothesis_id=hypothesis_id)
    return [r if isinstance(r, EvidenceReceipt) else EvidenceReceipt.from_dict(r) for r in rows]


def validate_hypothesis_candidate(
    candidate: HypothesisNode | Mapping[str, Any],
    goal: ResearchGoal,
    existing_fingerprints: set[str] | None = None,
    graveyard_fingerprints: set[str] | None = None,
) -> ValidationResult:
    """Strictly validate a hypothesis candidate against ResearchGoal constraints and redlines."""
    reasons: list[str] = []
    raw = candidate.to_dict() if isinstance(candidate, HypothesisNode) else dict(candidate)

    try:
        kind = (
            raw["kind"]
            if isinstance(raw.get("kind"), HypothesisKind)
            else HypothesisKind(raw["kind"])
        )
    except Exception:
        return ValidationResult(
            valid=False, reasons=[f"invalid hypothesis kind: {raw.get('kind')!r}"]
        )

    title = str(raw.get("title") or "").strip()
    statement = str(raw.get("statement") or "").strip()
    mechanism = str(raw.get("mechanism") or "").strip()
    if not title:
        reasons.append("missing non-empty title")
    if not statement:
        reasons.append("missing non-empty statement")
    if not mechanism:
        reasons.append("missing non-empty mechanism")

    raw_kc = raw.get("kill_criteria")
    kill_criteria = (
        [str(k).strip() for k in raw_kc if str(k).strip()]
        if isinstance(raw_kc, (list, tuple))
        else []
    )
    if not kill_criteria:
        reasons.append("kill_criteria must be a non-empty list")

    acceptance_cmd = str(raw.get("acceptance_cmd") or "").strip()
    screen_cmd = str(raw.get("screen_cmd") or "").strip()
    confirm_cmd_raw = raw.get("confirm_cmd")
    confirm_cmd = None if confirm_cmd_raw is None else (str(confirm_cmd_raw).strip() or None)
    if not acceptance_cmd:
        reasons.append("acceptance_cmd must be non-empty")
    if not screen_cmd:
        reasons.append("screen_cmd must be non-empty")

    raw_wpaths = raw.get("writable_paths")
    writable_paths = (
        [str(p).strip() for p in raw_wpaths if str(p).strip()]
        if isinstance(raw_wpaths, (list, tuple))
        else []
    )
    if not writable_paths:
        reasons.append("writable_paths must be non-empty")

    # Relativize paths inside goal.repo_path if absolute, then validate via gates.py.
    rel_wpaths: list[str] = []
    for wp in writable_paths:
        normed = normalize_path_str(wp, repo_path=goal.repo_path)
        if not normed:
            reasons.append(f"invalid writable_path: {wp!r}")
        else:
            rel_wpaths.append(normed)

    if rel_wpaths:
        wp_gate = validate_writable_paths(
            changed_paths=(),
            writable_paths=rel_wpaths,
            goal_prefixes=goal.writable_prefixes,
        )
        if not wp_gate.passed:
            reasons.extend(wp_gate.failures)

    hyp_id = str(raw.get("hypothesis_id") or f"hyp_{uuid.uuid4().hex[:10]}").strip()
    branch_name = str(raw.get("branch_name") or f"arr/{goal.project}/{hyp_id}").strip()
    if branch_name in ("main", "master", "origin/main", "origin/master"):
        reasons.append(f"forbidden branch_name '{branch_name}' (cannot target main/master)")

    # Reject goals or explicit paths/commands targeting a protected workspace.
    repo_real = os.path.realpath(os.path.expanduser(goal.repo_path or ""))
    for frag in protected_repo_fragments():
        if frag in (goal.repo_path or "").replace("\\", "/").lower() or frag in repo_real.replace("\\", "/").lower():
            reasons.append(f"goal.repo_path targets protected workspace {frag!r}: {goal.repo_path!r}")
            break

    extra_bans = [*goal.banned_paths, *EXTRA_BANNED_FRAGMENTS, *protected_repo_fragments()]
    for label, cmd_str in (
        ("acceptance_cmd", acceptance_cmd),
        ("screen_cmd", screen_cmd),
        ("confirm_cmd", confirm_cmd or ""),
    ):
        ban_gate = check_no_banned_access(
            paths=rel_wpaths if label == "acceptance_cmd" else (),
            command=cmd_str,
            branch_name=branch_name if label == "acceptance_cmd" else "",
            extra_bans=extra_bans,
        )
        if not ban_gate.passed:
            for fail in ban_gate.failures:
                reasons.append(f"{label}: {fail}")

    if raw.get("params") is not None and not isinstance(raw.get("params"), Mapping):
        reasons.append("params must be a mapping")
    params = dict(raw.get("params") or {}) if isinstance(raw.get("params"), Mapping) else {}

    fidelity_raw = raw.get("fidelity", FidelityRung.F1_FAST_SCREEN)
    try:
        fidelity = (
            fidelity_raw
            if isinstance(fidelity_raw, FidelityRung)
            else FidelityRung(str(fidelity_raw))
        )
    except Exception:
        reasons.append(f"invalid fidelity rung: {fidelity_raw!r}")
        fidelity = FidelityRung.F1_FAST_SCREEN

    fp = str(raw.get("fingerprint") or "").strip().lower()
    if not fp:
        fp = compute_fingerprint(
            kind=kind,
            mechanism=mechanism,
            params=params,
            writable_paths=rel_wpaths or writable_paths,
            project=goal.project,
        )
    if not fp:
        reasons.append("fingerprint must be non-empty")

    if graveyard_fingerprints and fp in graveyard_fingerprints:
        reasons.append(f"fingerprint matches refuted entry in graveyard: {fp[:12]}")
    elif existing_fingerprints and fp in existing_fingerprints:
        reasons.append(f"duplicate fingerprint already in existing tree: {fp[:12]}")

    if reasons:
        return ValidationResult(valid=False, reasons=reasons, normalized_node=None)

    try:
        sci_verdict = (
            raw["scientific_verdict"]
            if isinstance(raw.get("scientific_verdict"), ScientificVerdict)
            else ScientificVerdict(raw.get("scientific_verdict", ScientificVerdict.UNKNOWN.value))
        )
    except Exception:
        sci_verdict = ScientificVerdict.UNKNOWN

    node = HypothesisNode(
        hypothesis_id=hyp_id,
        goal_id=goal.goal_id,
        parent_id=raw.get("parent_id"),
        kind=kind,
        title=title,
        statement=statement,
        mechanism=mechanism,
        fidelity=fidelity,
        priority_evoi=float(raw.get("priority_evoi", 0.0)),
        writable_paths=rel_wpaths,
        params=params,
        kill_criteria=kill_criteria,
        acceptance_cmd=acceptance_cmd,
        screen_cmd=screen_cmd,
        confirm_cmd=confirm_cmd,
        branch_name=branch_name,
        base_sha=str(raw.get("base_sha") or goal.baseline_sha),
        head_sha=raw.get("head_sha"),
        status=str(raw.get("status") or "PENDING"),
        scientific_verdict=sci_verdict,
        fingerprint=fp,
    )
    return ValidationResult(valid=True, reasons=[], normalized_node=node)


def _node_feature_tokens(node: HypothesisNode) -> set[str]:
    tokens: set[str] = set()
    for text in (node.title, node.mechanism, node.kind.value):
        for tok in _TOKEN_SPLIT_RE.findall(text.lower()):
            if len(tok) > 1:
                tokens.add(tok)
    for wp in node.writable_paths:
        tokens.add("path:" + normalize_path_str(wp).lower())
    for k, v in normalize_params(node.params).items():
        tokens.add(f"param:{k}={v}")
    return tokens


def compute_evoi(
    goal: ResearchGoal,
    candidate: HypothesisNode,
    existing_hypotheses: Sequence[HypothesisNode] = (),
    *,
    expected_gain: float | None = None,
    estimated_tokens: int | None = None,
    estimated_wall_s: float | None = None,
) -> float:
    """Compute Expected Value of Information from baseline distance, novelty, and cost."""
    gain_score = 0.0
    if expected_gain is None:
        raw_exp = candidate.params.get("expected_delta") or candidate.params.get("expected_gain")
        if raw_exp is not None:
            try:
                expected_gain = float(raw_exp)
            except (TypeError, ValueError):
                expected_gain = None
    if expected_gain is not None and math.isfinite(expected_gain):
        denom = max(abs(goal.baseline_value), 1.0)
        gain_score = min(2.0, max(0.05, abs(expected_gain) / denom + 0.35))
    elif "expected_value" in candidate.params:
        try:
            exp_val = float(candidate.params["expected_value"])
            direction = goal.direction.strip().lower()
            delta = (
                goal.baseline_value - exp_val
                if direction in ("min", "minimize", "lower")
                else exp_val - goal.baseline_value
            )
            denom = max(abs(goal.baseline_value), 1.0)
            gain_score = min(2.0, max(0.05, 0.4 + delta / denom))
        except (TypeError, ValueError):
            gain_score = 0.0

    if gain_score <= 0.0:
        kind_prior = {
            HypothesisKind.FALSIFICATION: 0.92,
            HypothesisKind.ABLATION: 0.86,
            HypothesisKind.COMPETING_ROUTE: 0.80,
            HypothesisKind.PARAMETER_RACE: 0.65,
            HypothesisKind.REFINEMENT: 0.55,
        }
        gain_score = kind_prior.get(candidate.kind, 0.5)

    if candidate.fidelity == FidelityRung.F1_FAST_SCREEN:
        gain_score *= 1.1
    elif candidate.fidelity == FidelityRung.F2_FULL_CONFIRM:
        gain_score *= 1.05

    cand_tokens = _node_feature_tokens(candidate)
    max_sim = 0.0
    for other in existing_hypotheses:
        if other.hypothesis_id == candidate.hypothesis_id:
            continue
        other_tokens = _node_feature_tokens(other)
        union = cand_tokens | other_tokens
        if union:
            sim = len(cand_tokens & other_tokens) / len(union)
            if sim > max_sim:
                max_sim = sim
    novelty = max(0.15, 1.0 - 0.85 * max_sim)

    toks = estimated_tokens
    if toks is None:
        default_toks = 1000 if candidate.kind == HypothesisKind.PARAMETER_RACE else 8000
        toks = int(candidate.params.get("estimated_tokens", default_toks))
    wall_s = estimated_wall_s
    if wall_s is None:
        wall_s = float(candidate.params.get("estimated_wall_s", 120.0))

    toks = max(0, int(toks))
    wall_s = max(1.0, float(wall_s))
    budget = max(1, int(goal.max_tokens_budget))
    token_frac = min(5.0, toks / budget)
    wall_factor = wall_s / 120.0
    cost_denom = 1.0 + 4.0 * token_frac + 0.25 * math.log1p(wall_factor)

    score = (gain_score * novelty) / max(0.1, cost_denom)
    if not math.isfinite(score):
        return 0.0
    return round(max(0.0, score), 6)


def ready_queue_watermark(dispatchable_slots: int) -> float:
    """Return the ready-queue cap = 1.5 * dispatchable_slots."""
    return READY_QUEUE_WATERMARK_FACTOR * max(1, int(dispatchable_slots))


def check_anti_filler_guard(
    goal: ResearchGoal,
    ready_queue_count: int,
    dispatchable_slots: int,
    used_tokens: int,
    *,
    consecutive_non_improving: int = 0,
    diminishing_threshold: int = DEFAULT_DIMINISHING_RETURNS_STREAK,
    min_reserve_tokens: int = 500,
) -> tuple[bool, str | None]:
    """Return (allowed, stop_reason) enforcing watermark, budget, and diminishing-returns fuses."""
    watermark = ready_queue_watermark(dispatchable_slots)
    if ready_queue_count >= watermark:
        return (
            False,
            f"READY_QUEUE_WATERMARK_REACHED ({ready_queue_count} >= {watermark:.1f})",
        )
    if goal.max_tokens_budget > 0:
        remaining = int(goal.max_tokens_budget) - int(used_tokens)
        if used_tokens >= goal.max_tokens_budget or remaining < min_reserve_tokens:
            return (
                False,
                f"TOKEN_BUDGET_FUSE ({used_tokens}/{goal.max_tokens_budget})",
            )
    if consecutive_non_improving >= diminishing_threshold:
        return (
            False,
            f"DIMINISHING_RETURNS_FUSE (streak={consecutive_non_improving} >= {diminishing_threshold})",
        )
    return (True, None)


def rank_candidates_by_evoi(
    candidates: Sequence[HypothesisNode],
    *,
    min_evoi: float = DEFAULT_MIN_EVOI,
) -> tuple[list[HypothesisNode], list[dict[str, Any]]]:
    """Filter out sub-threshold low-EVOI filler candidates and sort descending by priority_evoi."""
    kept: list[HypothesisNode] = []
    rejected: list[dict[str, Any]] = []
    for node in candidates:
        if node.priority_evoi < min_evoi:
            rejected.append(
                {
                    "hypothesis_id": node.hypothesis_id,
                    "title": node.title,
                    "reasons": [f"low-EVOI filler rejected ({node.priority_evoi:.4f} < {min_evoi:.4f})"],
                }
            )
            continue
        kept.append(node)
    kept.sort(key=lambda h: (-float(h.priority_evoi), h.hypothesis_id))
    return kept, rejected


def _resolve_prefix_path(
    default_path: str,
    writable_prefixes: Sequence[str],
    repo_path: str,
) -> str:
    normed = normalize_path_str(default_path, repo_path=repo_path)
    if not writable_prefixes:
        return normed
    for pref in writable_prefixes:
        p = normalize_path_str(pref, repo_path=repo_path).rstrip("/")
        if not p or normed == p or normed.startswith(p + "/"):
            return normed
    return normalize_path_str(writable_prefixes[0], repo_path=repo_path).rstrip("/")


def build_seed_proposals(goal: ResearchGoal) -> list[dict[str, Any]]:
    """Generate deterministic seed proposals from a registered project adapter, or generic seeds."""
    from conjectra.adapters import available_adapters, get_adapter

    adapter = (goal.adapter_name or "").strip().lower()
    proj = (goal.project or "").strip().lower()
    registered = set(available_adapters()) - {"generic"}
    adapter_key = adapter if adapter in registered else (proj if (adapter in ("", "generic") and proj in registered) else "")
    if adapter_key:

        adapter_nodes = get_adapter(adapter_key).seed_hypotheses(goal)
        out: list[dict[str, Any]] = []
        for node in adapter_nodes:
            d = node.to_dict()
            mapped_paths: list[str] = []
            seen_paths: set[str] = set()
            for wp in node.writable_paths:
                resolved = _resolve_prefix_path(wp, goal.writable_prefixes, goal.repo_path)
                if resolved and resolved not in seen_paths:
                    seen_paths.add(resolved)
                    mapped_paths.append(resolved)
            if mapped_paths != node.writable_paths:
                d["writable_paths"] = mapped_paths
                d["fingerprint"] = ""
            out.append(d)
        return out

    base_wp = _resolve_prefix_path("src", goal.writable_prefixes, goal.repo_path)
    return [
        {
            "kind": HypothesisKind.COMPETING_ROUTE.value,
            "title": f"{goal.project} primary algorithmic improvement route",
            "statement": f"Optimize {goal.primary_metric} ({goal.direction}) via targeted algorithmic improvement.",
            "mechanism": "primary_algorithmic_route",
            "writable_paths": [base_wp],
            "params": {"route": "primary", "estimated_tokens": 4000, "estimated_wall_s": 90.0},
            "kill_criteria": list(goal.kill_criteria) or ["regression_vs_baseline"],
            "screen_cmd": "pytest -q",
            "confirm_cmd": "pytest -q",
            "acceptance_cmd": "pytest -q",
        },
        {
            "kind": HypothesisKind.PARAMETER_RACE.value,
            "title": f"{goal.project} parameter calibration race",
            "statement": f"Tune key parameters to improve {goal.primary_metric} without code risk.",
            "mechanism": "baseline_parameter_race",
            "writable_paths": [base_wp],
            "params": {"route": "param_race", "step_scale": 1.0, "estimated_tokens": 1000, "estimated_wall_s": 45.0},
            "kill_criteria": list(goal.kill_criteria) or ["regression_vs_baseline"],
            "screen_cmd": "pytest -q",
            "confirm_cmd": "pytest -q",
            "acceptance_cmd": "pytest -q",
        },
    ]


def _extract_json_block(text: str) -> Any:
    cleaned = (text or "").strip()
    if not cleaned:
        raise ValueError("empty AI proposer output")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", cleaned, flags=re.IGNORECASE)
    for block in fenced:
        try:
            return json.loads(block.strip())
        except json.JSONDecodeError:
            continue
    for open_ch, close_ch in (("[", "]"), ("{", "}")):
        start = cleaned.find(open_ch)
        end = cleaned.rfind(close_ch)
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                pass
    raise ValueError("no valid JSON payload found in AI output")


def parse_proposals_payload(
    payload: str | Sequence[Mapping[str, Any]] | Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Parse external/AI JSON proposals into a list of candidate dicts."""
    parsed = _extract_json_block(payload) if isinstance(payload, str) else payload
    if isinstance(parsed, Mapping):
        for key in ("proposals", "hypotheses", "candidates", "items"):
            if (
                key in parsed
                and isinstance(parsed[key], Sequence)
                and not isinstance(parsed[key], (str, bytes))
            ):
                parsed = parsed[key]
                break
        else:
            if "kind" in parsed or "title" in parsed or "mechanism" in parsed:
                parsed = [parsed]
            else:
                raise ValueError("JSON object does not contain proposals/hypotheses list")
    if not isinstance(parsed, Sequence) or isinstance(parsed, (str, bytes)):
        raise ValueError("proposals payload must be a JSON array or object with proposals list")
    out: list[dict[str, Any]] = []
    for idx, item in enumerate(parsed):
        if not isinstance(item, Mapping):
            raise ValueError(f"proposal[{idx}] must be a JSON object, got {type(item).__name__}")
        out.append(dict(item))
    return out


def _extract_token_usage(usage_doc: Mapping[str, Any] | None, default_model: str) -> tuple[int, int, str]:
    if not usage_doc:
        return 0, 0, default_model
    model_used = str(usage_doc.get("model") or default_model)
    usage = usage_doc.get("usage")
    if not isinstance(usage, Mapping):
        usage = usage_doc

    def _pick(keys: tuple[str, ...]) -> int:
        for k in keys:
            if k in usage and usage[k] is not None:
                try:
                    return max(0, int(usage[k]))
                except (TypeError, ValueError):
                    continue
        return 0

    in_tok = _pick(("in_tokens", "input_tokens", "prompt_tokens", "inputTokens", "promptTokens"))
    out_tok = _pick(("out_tokens", "output_tokens", "completion_tokens", "outputTokens", "completionTokens"))
    if in_tok == 0 and out_tok == 0 and isinstance(usage.get("models"), Mapping):
        for m_info in usage["models"].values():
            if isinstance(m_info, Mapping):
                toks = m_info.get("tokens") if isinstance(m_info.get("tokens"), Mapping) else m_info
                if isinstance(toks, Mapping):
                    in_tok += int(toks.get("input", toks.get("prompt", 0)) or 0)
                    out_tok += int(toks.get("output", toks.get("candidates", 0)) or 0)
    return in_tok, out_tok, model_used


def _hypothesis_generation(
    hyp: HypothesisNode,
    by_id: Mapping[str, HypothesisNode],
) -> int:
    raw_gen = hyp.params.get("generation") if isinstance(hyp.params, Mapping) else None
    if raw_gen is not None:
        try:
            g = int(raw_gen)
            if g >= 1:
                return g
        except (TypeError, ValueError):
            pass
    gen = 1
    seen = {hyp.hypothesis_id}
    cur = hyp
    while cur.parent_id and cur.parent_id not in seen:
        seen.add(cur.parent_id)
        gen += 1
        parent = by_id.get(cur.parent_id)
        if parent is None:
            break
        cur = parent
    return gen


def _extract_primary_metric_value(
    metrics: Mapping[str, Any] | None,
    primary_metric: str,
) -> float | None:
    if not isinstance(metrics, Mapping) or not primary_metric:
        return None
    raw = metrics.get(primary_metric)
    if isinstance(raw, Mapping) and "value" in raw:
        raw = raw["value"]
    try:
        val = float(raw)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def _build_ai_propose_prompt(
    goal: ResearchGoal,
    existing_hypotheses: Sequence[HypothesisNode],
    graveyard_fps: set[str],
    max_proposals: int,
    *,
    receipts: Sequence[EvidenceReceipt] | None = None,
    graveyard_entries: Sequence[Mapping[str, Any]] | None = None,
    knowledge_items: Sequence[Mapping[str, Any]] | None = None,
) -> str:
    by_id = {h.hypothesis_id: h for h in existing_hypotheses}
    latest_rcpt_by_hyp: dict[str, EvidenceReceipt] = {}
    for rcpt in receipts or ():
        latest_rcpt_by_hyp[rcpt.hypothesis_id] = rcpt

    max_gen = 1
    existing_summary: list[dict[str, Any]] = []
    valid_negative_lessons: list[dict[str, Any]] = []
    for h in existing_hypotheses[-16:]:
        gen = _hypothesis_generation(h, by_id)
        max_gen = max(max_gen, gen)
        rcpt = latest_rcpt_by_hyp.get(h.hypothesis_id)
        measured = (
            _extract_primary_metric_value(rcpt.metrics, goal.primary_metric)
            if rcpt is not None
            else None
        )
        delta_vs_base = (
            round(measured - float(goal.baseline_value), 6)
            if measured is not None
            else None
        )
        item = {
            "hypothesis_id": h.hypothesis_id,
            "parent_id": h.parent_id,
            "generation": gen,
            "kind": h.kind.value,
            "title": h.title,
            "mechanism": h.mechanism,
            "status": h.status,
            "verdict": h.scientific_verdict.value,
            "fingerprint": h.fingerprint[:12],
            "measured_primary_metric": measured,
            "baseline_value": float(goal.baseline_value),
            "delta_vs_baseline": delta_vs_base,
        }
        if rcpt is not None:
            item["evidence_stage"] = rcpt.stage.value
            item["evidence_validity"] = rcpt.evidence_validity.value
            if rcpt.failure_reason:
                item["failure_reason"] = rcpt.failure_reason[:240]
        existing_summary.append(item)
        if h.scientific_verdict in (ScientificVerdict.VALID_NEGATIVE, ScientificVerdict.REFUTED):
            valid_negative_lessons.append(
                {
                    "hypothesis_id": h.hypothesis_id,
                    "generation": gen,
                    "mechanism": h.mechanism,
                    "verdict": h.scientific_verdict.value,
                    "measured_value": measured,
                    "root_cause": (rcpt.failure_reason[:240] if rcpt and rcpt.failure_reason else h.status),
                }
            )

    for gy in (graveyard_entries or ())[-8:]:
        hid = str(gy.get("hypothesis_id") or "")
        if hid and not any(x["hypothesis_id"] == hid for x in valid_negative_lessons):
            valid_negative_lessons.append(
                {
                    "hypothesis_id": hid,
                    "verdict": str(gy.get("scientific_verdict") or "REFUTED"),
                    "root_cause": str(gy.get("reason") or "")[:240],
                }
            )
    for ki in (knowledge_items or ())[-8:]:
        if str(ki.get("category") or "") == "valid_negative":
            content = ki.get("content") if isinstance(ki.get("content"), Mapping) else {}
            hid = str(content.get("hypothesis_id") or ki.get("hypothesis_id") or "")
            if hid and not any(x.get("hypothesis_id") == hid for x in valid_negative_lessons):
                valid_negative_lessons.append(dict(content))

    next_gen = (max_gen + 1) if existing_hypotheses else 1
    return (
        "You are a readonly scientific hypothesis proposer for conjectra.\n"
        f"Project: {goal.project}\n"
        f"Goal: {goal.title} (goal_version={getattr(goal, 'version', 1)})\n"
        f"Primary metric: {goal.primary_metric} ({goal.direction}), baseline={goal.baseline_value}\n"
        f"Writable prefixes (STRICT): {json.dumps(goal.writable_prefixes)}\n"
        f"Banned paths (NEVER TOUCH): {json.dumps(goal.banned_paths)}\n"
        f"Goal kill criteria: {json.dumps(goal.kill_criteria)}\n"
        f"Existing hypothesis lineage & measured F1/F2 metrics: {json.dumps(existing_summary)}\n"
        f"VALID_NEGATIVE / Refuted root-cause lessons: {json.dumps(valid_negative_lessons)}\n"
        f"Refuted graveyard count: {len(graveyard_fps)}\n"
        f"Target evolution generation: {next_gen} (set params.generation={next_gen} and parent_id when building on an existing hypothesis).\n\n"
        f"Return STRICT JSON ONLY (no prose) as a JSON array of up to {max_proposals} hypotheses.\n"
        "Each object MUST have keys: kind (COMPETING_ROUTE|REFINEMENT|ABLATION|FALSIFICATION|PARAMETER_RACE), "
        "parent_id (string or null), title, statement, mechanism, writable_paths (non-empty list within writable_prefixes), "
        "params (object including generation), kill_criteria (non-empty list of strings), screen_cmd, confirm_cmd, acceptance_cmd.\n"
    )


DEFAULT_PROPOSER_POOL_ORDER: tuple[str, ...] = (
    "deepseek",
    "claude",
    "gemini",
    "codex",
    "grok",
)


def _resolve_proposer_candidates(
    *,
    model: str,
    provider_spec: str | None,
    arbiter: Any,
    prefer_gemini_when_default: bool,
) -> list[Any]:
    from conjectra.arbiter import POOL_NORMALIZATION, ResourceArbiter
    from conjectra.multiagent import parse_provider_spec

    live_arbiter = arbiter or ResourceArbiter()
    headroom = live_arbiter.read_pool_headroom()

    ordered_specs: list[str] = []
    if provider_spec and provider_spec.strip().lower() != "auto":
        ordered_specs.append(provider_spec.strip())
    elif model and model != DEFAULT_AI_MODEL:
        ordered_specs.append(model.strip())
    elif prefer_gemini_when_default:
        ordered_specs.append(f"gemini:{DEFAULT_AI_MODEL}")

    for pool in DEFAULT_PROPOSER_POOL_ORDER:
        if pool not in ordered_specs:
            ordered_specs.append(pool)

    specs: list[Any] = []
    seen_labels: set[str] = set()
    for raw in ordered_specs:
        try:
            ps = parse_provider_spec(raw, role="builder", arbiter=live_arbiter)
        except Exception:
            continue
        lbl = ps.label()
        if lbl in seen_labels:
            continue
        seen_labels.add(lbl)
        specs.append(ps)

    def _is_healthy(ps: Any) -> bool:
        norm = POOL_NORMALIZATION.get(ps.pool.lower(), ps.pool.lower())
        info = headroom.get(norm)
        return True if not info else bool(info.get("available", True))

    healthy = [ps for ps in specs if _is_healthy(ps)]
    unhealthy = [ps for ps in specs if not _is_healthy(ps)]
    return healthy + unhealthy


def run_one_shot_ai_proposer(
    goal: ResearchGoal,
    existing_hypotheses: Sequence[HypothesisNode],
    graveyard_fps: set[str],
    *,
    max_proposals: int = 3,
    model: str = DEFAULT_AI_MODEL,
    provider_spec: str | None = None,
    timeout_s: int = 300,
    dispatch_script: str = MULTI_POOL_DISPATCH_PATH,
    subprocess_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    receipts: Sequence[EvidenceReceipt] | None = None,
    graveyard_entries: Sequence[Mapping[str, Any]] | None = None,
    knowledge_items: Sequence[Mapping[str, Any]] | None = None,
    arr_home: str | Path | None = None,
    arbiter: Any = None,
) -> tuple[list[dict[str, Any]], int, int, str, str | None]:
    """Run provider-agnostic multi-pool one-shot AI proposer across deepseek/grok/claude/codex/gemini."""
    prompt = _build_ai_propose_prompt(
        goal,
        existing_hypotheses,
        graveyard_fps,
        max_proposals,
        receipts=receipts,
        graveyard_entries=graveyard_entries,
        knowledge_items=knowledge_items,
    )
    candidates = _resolve_proposer_candidates(
        model=model,
        provider_spec=provider_spec,
        arbiter=arbiter,
        prefer_gemini_when_default=(subprocess_runner is not None),
    )
    if not candidates:
        return [], 0, 0, model or DEFAULT_AI_MODEL, "AI proposer failed: no provider candidates available"

    if subprocess_runner is None:
        from conjectra.multiagent import invoke_provider
        from conjectra.state_db import resolve_arr_home

        home = resolve_arr_home(arr_home)
        art_dir = home / "artifacts" / "propose" / f"{goal.goal_id}_{uuid.uuid4().hex[:8]}"
        cwd_path = Path(goal.repo_path) if os.path.isdir(goal.repo_path) else home
        errors: list[str] = []
        tot_in, tot_out = 0, 0
        last_model = candidates[0].label()
        for idx, prov in enumerate(candidates):
            res = invoke_provider(
                prov,
                prompt,
                cwd=cwd_path,
                art_dir=art_dir,
                tag=f"propose_{idx}_{prov.pool}",
                readonly=True,
                timeout_s=timeout_s,
                multipool_script=Path(dispatch_script) if dispatch_script else None,
            )
            tot_in += int(res.get("in_tokens", 0))
            tot_out += int(res.get("out_tokens", 0))
            last_model = str(res.get("model_used") or prov.label())
            if int(res.get("exit_code", 1)) != 0:
                errors.append(f"{prov.label()}: rc={res.get('exit_code')}")
                continue
            try:
                raw_list = parse_proposals_payload(str(res.get("response") or ""))
                return raw_list[:max_proposals], tot_in, tot_out, last_model, None
            except ValueError as exc:
                errors.append(f"{prov.label()} parse error: {exc}")
                if len(candidates) == 1:
                    return [], tot_in, tot_out, last_model, f"AI proposer JSON parse error: {exc}"
        return [], tot_in, tot_out, last_model, f"AI proposer failed across pools: {'; '.join(errors)}"

    errors_sub: list[str] = []
    tot_in, tot_out = 0, 0
    last_model = model
    for prov in candidates:
        fd, usage_path = tempfile.mkstemp(prefix="arr_propose_usage_", suffix=".json")
        os.close(fd)
        try:
            host = "local"
            cmd = [
                sys.executable,
                dispatch_script,
                prov.pool,
                "--host",
                host,
                "--model",
                prov.model,
                "--readonly",
                "--cwd",
                goal.repo_path,
                "--usage-out",
                usage_path,
                "--timeout",
                str(timeout_s),
                "--tag",
                f"propose-{goal.goal_id}",
            ]
            proc = subprocess_runner(
                cmd, input=prompt, text=True, capture_output=True, timeout=timeout_s
            )
            usage_doc: dict[str, Any] | None = None
            if os.path.exists(usage_path) and os.path.getsize(usage_path) > 0:
                try:
                    with open(usage_path, encoding="utf-8") as fh:
                        loaded = json.load(fh)
                        if isinstance(loaded, dict):
                            usage_doc = loaded
                except Exception:
                    usage_doc = None
            in_tok, out_tok, model_used = _extract_token_usage(usage_doc, prov.model)
            tot_in += in_tok
            tot_out += out_tok
            last_model = model_used
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or f"returncode={proc.returncode}").strip()
                errors_sub.append(f"{prov.label()}: {err}")
                continue
            try:
                raw_list = parse_proposals_payload(proc.stdout or "")
            except ValueError as exc:
                return [], tot_in, tot_out, last_model, f"AI proposer JSON parse error: {exc}"
            return raw_list[:max_proposals], tot_in, tot_out, last_model, None
        finally:
            try:
                if os.path.exists(usage_path):
                    os.unlink(usage_path)
            except OSError:
                pass
    return [], tot_in, tot_out, last_model, f"AI proposer failed: {'; '.join(errors_sub)}"


def _ingest_raw_candidates(
    goal: ResearchGoal,
    raw_candidates: Sequence[Mapping[str, Any]],
    *,
    db: Any = None,
    existing_hypotheses: Sequence[HypothesisNode] | None = None,
    graveyard_fps: set[str] | None = None,
    ready_queue_count: int = 0,
    dispatchable_slots: int | None = None,
    used_tokens: int | None = None,
    min_evoi: float = DEFAULT_MIN_EVOI,
    persist: bool = True,
    source: str = "seeds",
    in_tokens: int = 0,
    out_tokens: int = 0,
    model_used: str = "",
) -> ProposalBatchResult:
    existing = (
        list(existing_hypotheses)
        if existing_hypotheses is not None
        else _db_list_hypotheses(db, goal.goal_id)
    )
    graveyard = (
        set(graveyard_fps)
        if graveyard_fps is not None
        else _db_graveyard_fingerprints(db, goal.goal_id)
    )
    slots = (
        dispatchable_slots
        if dispatchable_slots is not None
        else max(1, goal.max_concurrent_branches)
    )
    tokens_spent = used_tokens if used_tokens is not None else _db_used_tokens(db, goal.goal_id)

    non_improving = 0
    for h in reversed(existing):
        if h.scientific_verdict == ScientificVerdict.CONFIRMED:
            break
        if h.scientific_verdict in (
            ScientificVerdict.REFUTED,
            ScientificVerdict.VALID_NEGATIVE,
            ScientificVerdict.UNKNOWN,
        ) and h.status in (
            "REFUTED",
            "VALID_NEGATIVE",
            "ESCALATED",
            "PRUNED",
            "GRAVEYARD",
        ):
            non_improving += 1

    allowed, stop_reason = check_anti_filler_guard(
        goal,
        ready_queue_count=ready_queue_count,
        dispatchable_slots=slots,
        used_tokens=tokens_spent,
        consecutive_non_improving=non_improving,
    )
    if not allowed:
        return ProposalBatchResult(
            accepted=[],
            rejected=[{"title": "<batch>", "reasons": [stop_reason or "anti-filler guard"]}],
            stopped_reason=stop_reason,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            model_used=model_used,
            source=source,
        )

    existing_fps = {h.fingerprint.strip().lower() for h in existing if h.fingerprint}
    seen_fps: set[str] = set()
    valid_nodes: list[HypothesisNode] = []
    rejected: list[dict[str, Any]] = []

    for raw in raw_candidates:
        vr = validate_hypothesis_candidate(
            raw,
            goal,
            existing_fingerprints=existing_fps | seen_fps,
            graveyard_fingerprints=graveyard,
        )
        if not vr.valid or vr.normalized_node is None:
            rejected.append(
                {
                    "hypothesis_id": str(raw.get("hypothesis_id") or ""),
                    "title": str(raw.get("title") or "<untitled>"),
                    "reasons": vr.reasons,
                }
            )
            continue
        node = vr.normalized_node
        if _db_is_in_graveyard(db, node.fingerprint, goal_id=goal.goal_id):
            rejected.append(
                {
                    "hypothesis_id": node.hypothesis_id,
                    "title": node.title,
                    "reasons": [f"fingerprint in state_db graveyard: {node.fingerprint[:12]}"],
                }
            )
            continue
        node.priority_evoi = compute_evoi(goal, node, existing + valid_nodes)
        seen_fps.add(node.fingerprint)
        valid_nodes.append(node)

    ranked, low_evoi_rejected = rank_candidates_by_evoi(valid_nodes, min_evoi=min_evoi)
    rejected.extend(low_evoi_rejected)

    watermark = ready_queue_watermark(slots)
    headroom = max(0, int(math.floor(watermark - ready_queue_count + 1e-9)))
    if len(ranked) > headroom:
        overflow = ranked[headroom:]
        ranked = ranked[:headroom]
        for ov in overflow:
            rejected.append(
                {
                    "hypothesis_id": ov.hypothesis_id,
                    "title": ov.title,
                    "reasons": [f"trimmed by ready-queue watermark ({watermark:.1f})"],
                }
            )

    if persist and db is not None:
        for node in ranked:
            _db_register_or_update_hypothesis(db, node)

    return ProposalBatchResult(
        accepted=ranked,
        rejected=rejected,
        stopped_reason=None,
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        model_used=model_used,
        source=source,
    )


def import_proposals_json(
    goal: ResearchGoal,
    payload: str | Sequence[Mapping[str, Any]] | Mapping[str, Any],
    *,
    db: Any = None,
    existing_hypotheses: Sequence[HypothesisNode] | None = None,
    graveyard_fps: set[str] | None = None,
    ready_queue_count: int = 0,
    dispatchable_slots: int | None = None,
    used_tokens: int | None = None,
    min_evoi: float = DEFAULT_MIN_EVOI,
    persist: bool = True,
    source: str = "external_json",
) -> ProposalBatchResult:
    """Validate, deduplicate, EVOI-rank, and optionally persist proposals from /teamwork or run_workflow."""
    try:
        raw_list = parse_proposals_payload(payload)
    except ValueError as exc:
        return ProposalBatchResult(
            accepted=[],
            rejected=[{"title": "<payload>", "reasons": [str(exc)]}],
            stopped_reason=f"INVALID_PROPOSAL_JSON: {exc}",
            source=source,
        )
    return _ingest_raw_candidates(
        goal,
        raw_list,
        db=db,
        existing_hypotheses=existing_hypotheses,
        graveyard_fps=graveyard_fps,
        ready_queue_count=ready_queue_count,
        dispatchable_slots=dispatchable_slots,
        used_tokens=used_tokens,
        min_evoi=min_evoi,
        persist=persist,
        source=source,
    )


def propose_hypotheses(
    goal: ResearchGoal,
    *,
    db: Any = None,
    mode: str = "auto",
    external_proposals: str | Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    existing_hypotheses: Sequence[HypothesisNode] | None = None,
    graveyard_fps: set[str] | None = None,
    ready_queue_count: int = 0,
    dispatchable_slots: int | None = None,
    used_tokens: int | None = None,
    min_evoi: float = DEFAULT_MIN_EVOI,
    max_ai_proposals: int = 3,
    ai_model: str = DEFAULT_AI_MODEL,
    ai_provider_spec: str | None = None,
    ai_timeout_s: int = 300,
    subprocess_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    persist: bool = True,
) -> ProposalBatchResult:
    """Propose validated hypotheses via deterministic seeds, one-shot AI, or external JSON import."""
    if external_proposals is not None or mode == "import":
        if external_proposals is None:
            return ProposalBatchResult(
                accepted=[],
                rejected=[{"title": "<payload>", "reasons": ["external_proposals required for mode='import'"]}],
                stopped_reason="MISSING_EXTERNAL_PROPOSALS",
                source="import",
            )
        return import_proposals_json(
            goal,
            external_proposals,
            db=db,
            existing_hypotheses=existing_hypotheses,
            graveyard_fps=graveyard_fps,
            ready_queue_count=ready_queue_count,
            dispatchable_slots=dispatchable_slots,
            used_tokens=used_tokens,
            min_evoi=min_evoi,
            persist=persist,
            source="external_json",
        )

    existing = (
        list(existing_hypotheses)
        if existing_hypotheses is not None
        else _db_list_hypotheses(db, goal.goal_id)
    )
    graveyard = (
        set(graveyard_fps)
        if graveyard_fps is not None
        else _db_graveyard_fingerprints(db, goal.goal_id)
    )
    slots = (
        dispatchable_slots
        if dispatchable_slots is not None
        else max(1, goal.max_concurrent_branches)
    )
    tokens_spent = used_tokens if used_tokens is not None else _db_used_tokens(db, goal.goal_id)

    allowed, stop_reason = check_anti_filler_guard(
        goal,
        ready_queue_count=ready_queue_count,
        dispatchable_slots=slots,
        used_tokens=tokens_spent,
    )
    if not allowed:
        return ProposalBatchResult(
            accepted=[],
            rejected=[{"title": "<batch>", "reasons": [stop_reason or "anti-filler guard"]}],
            stopped_reason=stop_reason,
            source=mode,
        )

    raw_candidates: list[dict[str, Any]] = []
    extra_rejected: list[dict[str, Any]] = []
    in_tok, out_tok, model_used = 0, 0, ""

    if mode in ("seeds", "auto"):
        raw_candidates.extend(build_seed_proposals(goal))

    if mode in ("ai", "auto") and (mode == "ai" or not existing):
        gy_entries = db.list_graveyard(goal_id=goal.goal_id) if (db is not None and hasattr(db, "list_graveyard")) else ()
        know_entries = (
            db.list_knowledge_items(goal_id=goal.goal_id, category="valid_negative")
            if (db is not None and hasattr(db, "list_knowledge_items"))
            else ()
        )
        ai_raw, in_tok, out_tok, model_used, ai_err = run_one_shot_ai_proposer(
            goal,
            existing,
            graveyard,
            max_proposals=max_ai_proposals,
            model=ai_model,
            provider_spec=ai_provider_spec,
            timeout_s=ai_timeout_s,
            subprocess_runner=subprocess_runner,
            receipts=_db_list_receipts(db, goal.goal_id),
            graveyard_entries=gy_entries,
            knowledge_items=know_entries,
            arr_home=getattr(db, "arr_home", None),
        )
        if ai_err:
            extra_rejected.append({"title": "<ai_proposer>", "reasons": [ai_err]})
            if mode == "ai":
                return ProposalBatchResult(
                    accepted=[],
                    rejected=extra_rejected,
                    stopped_reason=ai_err,
                    in_tokens=in_tok,
                    out_tokens=out_tok,
                    model_used=model_used,
                    source="ai",
                )
        else:
            raw_candidates.extend(ai_raw)

    batch = _ingest_raw_candidates(
        goal,
        raw_candidates,
        db=db,
        existing_hypotheses=existing,
        graveyard_fps=graveyard,
        ready_queue_count=ready_queue_count,
        dispatchable_slots=slots,
        used_tokens=tokens_spent,
        min_evoi=min_evoi,
        persist=persist,
        source=mode,
        in_tokens=in_tok,
        out_tokens=out_tok,
        model_used=model_used,
    )
    if extra_rejected:
        batch.rejected = extra_rejected + batch.rejected
    return batch


def _eval_kill_expr(expr: str, metrics: Mapping[str, Any]) -> bool:
    m = _KILL_EXPR_RE.match(expr or "")
    if not m:
        return False
    key, op, rhs_s = m.group(1), m.group(2), m.group(3)
    if key not in metrics:
        return False
    raw_val = metrics[key]
    if isinstance(raw_val, Mapping) and "value" in raw_val:
        raw_val = raw_val["value"]
    try:
        lhs = float(raw_val)
        rhs = float(rhs_s)
    except (TypeError, ValueError):
        return False
    if not math.isfinite(lhs):
        return True
    ops = {
        "<": lhs < rhs,
        "<=": lhs <= rhs,
        ">": lhs > rhs,
        ">=": lhs >= rhs,
        "==": lhs == rhs,
        "!=": lhs != rhs,
    }
    return ops[op]


def _is_empirical_receipt_for_verdict(
    goal: ResearchGoal,
    receipt: EvidenceReceipt,
) -> bool:
    """Return True iff `receipt` is an authentic empirical stage outcome eligible for scientific verdicts.

    Blocks all non-empirical stages (`CODE`, `CPU_GATE`, `BLIND_AUDIT`, `DESIGN_REVIEW`,
    `IMPL_REVIEW`, `RESULT_REVIEW`), stale epochs/versions, missing inputs/metrics,
    `TIMEOUT`/`OOM`/`LEASE_LOST`/`ENV_ERROR`, `cuda_unavailable`, CLI arg errors, and audit fails.
    """
    if receipt.stage not in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
        return False
    if int(receipt.goal_version) != int(goal.version):
        return False
    if int(receipt.human_epoch) != int(goal.human_epoch):
        return False
    if receipt.audit_verdict == AuditVerdict.FAIL:
        return False
    if receipt.evidence_validity in (
        EvidenceValidity.MISSING_INPUT,
        EvidenceValidity.MISSING_PRIMARY_METRIC,
        EvidenceValidity.INVALID_SYNTHETIC,
        EvidenceValidity.STALE_GOAL_VERSION,
        EvidenceValidity.HUMAN_INVALIDATED,
    ):
        return False
    if receipt.failure_domain in (
        FailureDomain.INFRA_TRANSIENT,
        FailureDomain.ENV_CONFIG_MISMATCH,
        FailureDomain.ENGINEERING_DEFECT,
    ):
        return False

    inferred_domain = classify_failure_domain(receipt, goal=goal)
    if inferred_domain in (
        FailureDomain.INFRA_TRANSIENT,
        FailureDomain.ENV_CONFIG_MISMATCH,
    ):
        return False

    if is_valid_empirical_outcome(receipt, goal):
        return True

    # Support minimal unit-test EvidenceReceipts on EXP_SCREEN/EXP_CONFIRM where
    # `evidence_validity` defaulted to VALID_ENGINEERING_ONLY, provided process/gate
    # succeeded cleanly with non-empty numeric metrics and zero failure-domain markers.
    if (
        receipt.evidence_validity == EvidenceValidity.VALID_ENGINEERING_ONLY
        and receipt.failure_domain == FailureDomain.NONE
        and receipt.process_status == ProcessStatus.EXIT_OK
        and receipt.engineering_verdict == EngineeringVerdict.PASS
        and bool(receipt.metrics)
    ):
        return True

    return False


def check_kill_criteria(
    goal: ResearchGoal,
    hypothesis: HypothesisNode,
    receipt: EvidenceReceipt,
) -> str | None:
    """Return the violated empirical kill-criterion string if triggered by valid empirical metrics.

    Never triggers on `BLIND_AUDIT` / `IMPL_REVIEW` failures, `CPU_GATE` failures,
    `TIMEOUT`, `ENV_CONFIG_MISMATCH`, or `"tests_passed == 0"` when `goal.primary_metric != "tests_passed"`.
    """
    if not _is_empirical_receipt_for_verdict(goal, receipt):
        return None
    metrics = receipt.metrics or {}
    if metrics.get("kill_criterion_hit"):
        hit = str(metrics["kill_criterion_hit"])
        if goal.primary_metric != "tests_passed" and "tests_passed" in hit:
            return None
        return hit
    if "kill_criterion" in (receipt.failure_reason or "").lower():
        return receipt.failure_reason
    for crit in list(hypothesis.kill_criteria) + list(goal.kill_criteria):
        m = _KILL_EXPR_RE.match(crit or "")
        if m and m.group(1) == "tests_passed" and goal.primary_metric != "tests_passed":
            continue
        if _eval_kill_expr(crit, metrics):
            return crit
    return None


def _collect_descendants(
    root_id: str,
    all_hypotheses: Sequence[HypothesisNode],
) -> list[HypothesisNode]:
    by_parent: dict[str, list[HypothesisNode]] = {}
    for h in all_hypotheses:
        if h.parent_id:
            by_parent.setdefault(h.parent_id, []).append(h)
    descendants: list[HypothesisNode] = []
    queue = list(by_parent.get(root_id, []))
    seen = {root_id}
    while queue:
        cur = queue.pop(0)
        if cur.hypothesis_id in seen:
            continue
        seen.add(cur.hypothesis_id)
        descendants.append(cur)
        queue.extend(by_parent.get(cur.hypothesis_id, []))
    return descendants


def _spawn_ablation_and_falsification(
    goal: ResearchGoal,
    parent: HypothesisNode,
    existing: Sequence[HypothesisNode],
    graveyard_fps: set[str],
    db: Any = None,
) -> list[HypothesisNode]:
    existing_fps = {h.fingerprint.strip().lower() for h in existing if h.fingerprint}
    parent_gen = _hypothesis_generation(parent, {h.hypothesis_id: h for h in existing})
    specs = [
        (
            HypothesisKind.ABLATION,
            f"Ablation of {parent.title}",
            f"Verify primary mechanism '{parent.mechanism}' causally drives the F1 gain by disabling/ablating it.",
            f"ablation::{parent.mechanism}",
            {
                **dict(parent.params),
                "generation": parent_gen + 1,
                "ablation_of": parent.hypothesis_id,
                "ablation_disable_primary": True,
                "estimated_tokens": 2500,
                "estimated_wall_s": 90.0,
            },
            list(parent.kill_criteria) + ["ablation_shows_no_effect"],
        ),
        (
            HypothesisKind.FALSIFICATION,
            f"Falsification stress on {parent.title}",
            f"Probe adversarial/counterexample edge inputs to falsify '{parent.mechanism}' before F2 confirmation.",
            f"falsification::{parent.mechanism}",
            {
                **dict(parent.params),
                "generation": parent_gen + 1,
                "falsification_of": parent.hypothesis_id,
                "adversarial_stress": True,
                "estimated_tokens": 2500,
                "estimated_wall_s": 90.0,
            },
            list(parent.kill_criteria) + ["counterexample_found"],
        ),
    ]
    spawned: list[HypothesisNode] = []
    for kind, title, statement, mechanism, params, kill_crit in specs:
        hid = f"hyp_{uuid.uuid4().hex[:10]}"
        raw = {
            "hypothesis_id": hid,
            "goal_id": goal.goal_id,
            "parent_id": parent.hypothesis_id,
            "kind": kind.value,
            "title": title,
            "statement": statement,
            "mechanism": mechanism,
            "fidelity": FidelityRung.F1_FAST_SCREEN.value,
            "writable_paths": list(parent.writable_paths),
            "params": params,
            "kill_criteria": kill_crit,
            "acceptance_cmd": parent.acceptance_cmd,
            "screen_cmd": parent.screen_cmd,
            "confirm_cmd": parent.confirm_cmd,
            "branch_name": f"arr/{goal.project}/{hid}",
            "base_sha": parent.head_sha or parent.base_sha,
            "status": "PENDING",
        }
        vr = validate_hypothesis_candidate(
            raw,
            goal,
            existing_fingerprints=existing_fps | {s.fingerprint for s in spawned},
            graveyard_fingerprints=graveyard_fps,
        )
        if not vr.valid or vr.normalized_node is None:
            continue
        node = vr.normalized_node
        if _db_is_in_graveyard(db, node.fingerprint, goal_id=goal.goal_id):
            continue
        node.priority_evoi = compute_evoi(goal, node, list(existing) + spawned)
        spawned.append(node)
        _db_register_or_update_hypothesis(db, node)
    return spawned


def _spawn_diagnostic_refinement(
    goal: ResearchGoal,
    parent: HypothesisNode,
    attempt_num: int,
    existing: Sequence[HypothesisNode],
    graveyard_fps: set[str],
    db: Any = None,
) -> HypothesisNode | None:
    existing_fps = {h.fingerprint.strip().lower() for h in existing if h.fingerprint}
    parent_gen = _hypothesis_generation(parent, {h.hypothesis_id: h for h in existing})
    hid = f"hyp_{uuid.uuid4().hex[:10]}"
    raw = {
        "hypothesis_id": hid,
        "goal_id": goal.goal_id,
        "parent_id": parent.hypothesis_id,
        "kind": HypothesisKind.REFINEMENT.value,
        "title": f"Diagnostic refinement #{attempt_num} for {parent.title}",
        "statement": f"Bounded diagnostic run to resolve UNKNOWN verdict on {parent.hypothesis_id}.",
        "mechanism": f"diagnostic_refinement::{parent.mechanism}",
        "fidelity": FidelityRung.F1_FAST_SCREEN.value,
        "writable_paths": list(parent.writable_paths),
        "params": {
            **dict(parent.params),
            "generation": parent_gen + 1,
            "diagnostic_of": parent.hypothesis_id,
            "diagnostic_attempt": attempt_num,
            "estimated_tokens": 2000,
            "estimated_wall_s": 60.0,
        },
        "kill_criteria": list(parent.kill_criteria),
        "acceptance_cmd": parent.acceptance_cmd,
        "screen_cmd": parent.screen_cmd,
        "confirm_cmd": parent.confirm_cmd,
        "branch_name": f"arr/{goal.project}/{hid}",
        "base_sha": parent.head_sha or parent.base_sha,
        "status": "PENDING",
    }
    vr = validate_hypothesis_candidate(
        raw,
        goal,
        existing_fingerprints=existing_fps,
        graveyard_fingerprints=graveyard_fps,
    )
    if not vr.valid or vr.normalized_node is None:
        return None
    node = vr.normalized_node
    if _db_is_in_graveyard(db, node.fingerprint, goal_id=goal.goal_id):
        return None
    node.priority_evoi = compute_evoi(goal, node, existing)
    _db_register_or_update_hypothesis(db, node)
    return node


def _count_unknown_streak(
    hypothesis: HypothesisNode,
    existing: Sequence[HypothesisNode],
    receipts: Sequence[EvidenceReceipt],
) -> int:
    streak = sum(1 for r in receipts if r.scientific_verdict == ScientificVerdict.UNKNOWN)
    by_id = {h.hypothesis_id: h for h in existing}
    cur = hypothesis
    while cur.parent_id and cur.kind == HypothesisKind.REFINEMENT:
        streak += 1
        parent = by_id.get(cur.parent_id)
        if parent is None:
            break
        cur = parent
    return streak


def reconcile_receipt(
    db: Any,
    goal: ResearchGoal,
    hypothesis: HypothesisNode,
    receipt: EvidenceReceipt,
    *,
    existing_hypotheses: Sequence[HypothesisNode] | None = None,
    graveyard_fps: set[str] | None = None,
    used_tokens: int | None = None,
    max_unknown_refinements: int = DEFAULT_MAX_UNKNOWN_REFINEMENTS,
) -> ReconcileDecision:
    """Apply deterministic multi-fidelity promotion, causal spawning, pruning, or escalation."""
    existing = (
        list(existing_hypotheses)
        if existing_hypotheses is not None
        else _db_list_hypotheses(db, goal.goal_id)
    )
    if all(h.hypothesis_id != hypothesis.hypothesis_id for h in existing):
        existing.append(hypothesis)
    graveyard = (
        set(graveyard_fps)
        if graveyard_fps is not None
        else _db_graveyard_fingerprints(db, goal.goal_id)
    )

    if not hypothesis.fingerprint:
        hypothesis.fingerprint = compute_fingerprint(
            kind=hypothesis.kind,
            mechanism=hypothesis.mechanism,
            params=hypothesis.params,
            writable_paths=hypothesis.writable_paths,
            project=goal.project,
        )

    prev_status = hypothesis.status
    prev_fidelity = hypothesis.fidelity
    empirical_ok = _is_empirical_receipt_for_verdict(goal, receipt)
    eff_domain = (
        receipt.failure_domain
        if receipt.failure_domain != FailureDomain.NONE
        else classify_failure_domain(receipt, goal=goal)
    )
    kill_hit = check_kill_criteria(goal, hypothesis, receipt) if empirical_ok else None

    def _prune_descendants_and_tasks() -> tuple[list[str], list[str]]:
        cancelled_ids = _db_cancel_hypothesis_tasks(db, hypothesis.hypothesis_id, goal.goal_id)
        pruned: list[str] = []
        for child in _collect_descendants(hypothesis.hypothesis_id, existing):
            if child.status not in (
                "REFUTED",
                "VALID_NEGATIVE",
                "PRUNED",
                "COMPLETED",
                "CANCELLED",
                "GRAVEYARD",
            ):
                child_prev = child.status
                child.status = "PRUNED"
                child.scientific_verdict = ScientificVerdict.NOT_APPLICABLE
                _db_register_or_update_hypothesis(db, child, prev_status=child_prev)
                pruned.append(child.hypothesis_id)
                cancelled_ids.extend(_db_cancel_hypothesis_tasks(db, child.hypothesis_id, goal.goal_id))
        return pruned, cancelled_ids

    # 0. Post-confirm milestone audit / RESULT_REVIEW PASS -> COMPLETED
    if (
        receipt.stage in (StageKind.BLIND_AUDIT, StageKind.RESULT_REVIEW)
        and receipt.process_status == ProcessStatus.EXIT_OK
        and receipt.audit_verdict == AuditVerdict.PASS
        and hypothesis.fidelity == FidelityRung.F3_MILESTONE_AUDIT
    ):
        hypothesis.failure_domain = FailureDomain.NONE
        hypothesis.status = "COMPLETED"
        hypothesis.scientific_verdict = ScientificVerdict.CONFIRMED
        _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
        return ReconcileDecision(
            hypothesis_id=hypothesis.hypothesis_id,
            previous_fidelity=prev_fidelity,
            new_fidelity=FidelityRung.F3_MILESTONE_AUDIT,
            new_status="COMPLETED",
            scientific_verdict=ScientificVerdict.CONFIRMED,
            action="COMPLETE",
        )

    # 1a. VALID_NEGATIVE (strictly gated on authentic empirical outcome!)
    if empirical_ok and receipt.scientific_verdict == ScientificVerdict.VALID_NEGATIVE:
        reason = kill_hit or receipt.failure_reason or "scientific_verdict=VALID_NEGATIVE"
        hypothesis.failure_domain = FailureDomain.SCIENTIFIC_OUTCOME
        hypothesis.scientific_verdict = ScientificVerdict.VALID_NEGATIVE
        hypothesis.status = "VALID_NEGATIVE"
        recorded = _db_record_graveyard(
            db,
            hypothesis,
            reason,
            {
                "receipt_id": receipt.receipt_id,
                "stage": receipt.stage.value,
                "metrics": receipt.metrics,
                "kill_criterion_hit": kill_hit,
                "created_ts": receipt.created_ts or time.time(),
            },
            scientific_verdict=ScientificVerdict.VALID_NEGATIVE,
            new_status="VALID_NEGATIVE",
        )
        if not recorded:
            _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
        _db_record_valid_negative_knowledge(db, goal, hypothesis, receipt, reason)
        pruned_ids, cancelled = _prune_descendants_and_tasks()
        return ReconcileDecision(
            hypothesis_id=hypothesis.hypothesis_id,
            previous_fidelity=prev_fidelity,
            new_fidelity=hypothesis.fidelity,
            new_status="VALID_NEGATIVE",
            scientific_verdict=ScientificVerdict.VALID_NEGATIVE,
            action="RECORD_VALID_NEGATIVE_AND_PRUNE",
            pruned_hypothesis_ids=pruned_ids,
            cancelled_task_ids=cancelled,
            graveyard_recorded=recorded,
            kill_criterion_hit=kill_hit,
        )

    # 1b. REFUTED or empirical kill-criterion hit -> record graveyard, mark REFUTED, prune descendants.
    if empirical_ok and (
        receipt.scientific_verdict == ScientificVerdict.REFUTED or kill_hit is not None
    ):
        reason = kill_hit or receipt.failure_reason or "scientific_verdict=REFUTED"
        hypothesis.failure_domain = FailureDomain.SCIENTIFIC_OUTCOME
        hypothesis.scientific_verdict = ScientificVerdict.REFUTED
        hypothesis.status = "REFUTED"

        recorded = _db_record_graveyard(
            db,
            hypothesis,
            reason,
            {
                "receipt_id": receipt.receipt_id,
                "stage": receipt.stage.value,
                "metrics": receipt.metrics,
                "kill_criterion_hit": kill_hit,
                "created_ts": receipt.created_ts or time.time(),
            },
        )
        if not recorded:
            _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)

        pruned_ids, cancelled = _prune_descendants_and_tasks()

        return ReconcileDecision(
            hypothesis_id=hypothesis.hypothesis_id,
            previous_fidelity=prev_fidelity,
            new_fidelity=hypothesis.fidelity,
            new_status="REFUTED",
            scientific_verdict=ScientificVerdict.REFUTED,
            action="REFUTE_AND_PRUNE",
            pruned_hypothesis_ids=pruned_ids,
            cancelled_task_ids=cancelled,
            graveyard_recorded=recorded,
            kill_criterion_hit=kill_hit,
        )

    # 2. CONFIRMED (strictly gated on empirical_ok + EXIT_OK + PASS)
    if (
        empirical_ok
        and receipt.scientific_verdict == ScientificVerdict.CONFIRMED
        and receipt.process_status == ProcessStatus.EXIT_OK
        and receipt.engineering_verdict == EngineeringVerdict.PASS
    ):
        hypothesis.failure_domain = FailureDomain.SCIENTIFIC_OUTCOME
        if hypothesis.fidelity == FidelityRung.F0_SANITY:
            hypothesis.fidelity = FidelityRung.F1_FAST_SCREEN
            hypothesis.status = "PROMOTED_F1"
            hypothesis.scientific_verdict = ScientificVerdict.CONFIRMED
            _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
            return ReconcileDecision(
                hypothesis_id=hypothesis.hypothesis_id,
                previous_fidelity=prev_fidelity,
                new_fidelity=FidelityRung.F1_FAST_SCREEN,
                new_status="PROMOTED_F1",
                scientific_verdict=ScientificVerdict.CONFIRMED,
                action="PROMOTE_F1",
                promoted_to=FidelityRung.F1_FAST_SCREEN,
            )

        if hypothesis.fidelity == FidelityRung.F1_FAST_SCREEN:
            hypothesis.fidelity = FidelityRung.F2_FULL_CONFIRM
            hypothesis.status = "PROMOTED_F2"
            hypothesis.scientific_verdict = ScientificVerdict.CONFIRMED
            _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)

            spawned = _spawn_ablation_and_falsification(
                goal, hypothesis, existing, graveyard, db=db
            )
            return ReconcileDecision(
                hypothesis_id=hypothesis.hypothesis_id,
                previous_fidelity=prev_fidelity,
                new_fidelity=FidelityRung.F2_FULL_CONFIRM,
                new_status="PROMOTED_F2",
                scientific_verdict=ScientificVerdict.CONFIRMED,
                action="PROMOTE_F2",
                promoted_to=FidelityRung.F2_FULL_CONFIRM,
                spawned_hypotheses=spawned,
            )

        if hypothesis.fidelity == FidelityRung.F2_FULL_CONFIRM:
            hypothesis.fidelity = FidelityRung.F3_MILESTONE_AUDIT
            hypothesis.status = "PROMOTED_F3"
            hypothesis.scientific_verdict = ScientificVerdict.CONFIRMED
            _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
            return ReconcileDecision(
                hypothesis_id=hypothesis.hypothesis_id,
                previous_fidelity=prev_fidelity,
                new_fidelity=FidelityRung.F3_MILESTONE_AUDIT,
                new_status="PROMOTED_F3",
                scientific_verdict=ScientificVerdict.CONFIRMED,
                action="PROMOTE_F3",
                promoted_to=FidelityRung.F3_MILESTONE_AUDIT,
            )

        hypothesis.status = "COMPLETED"
        hypothesis.scientific_verdict = ScientificVerdict.CONFIRMED
        _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
        return ReconcileDecision(
            hypothesis_id=hypothesis.hypothesis_id,
            previous_fidelity=prev_fidelity,
            new_fidelity=FidelityRung.F3_MILESTONE_AUDIT,
            new_status="COMPLETED",
            scientific_verdict=ScientificVerdict.CONFIRMED,
            action="COMPLETE",
        )

    # 3a. Failure isolation: INFRA_TRANSIENT (`TIMEOUT`, `OOM`, `LEASE_LOST`, HQ `SIGINT`)
    # NEVER refutes, NEVER writes graveyard, NEVER spawns blind child hypotheses!
    if eff_domain == FailureDomain.INFRA_TRANSIENT:
        sv, block_status = map_timeout_verdict(receipt.process_status)
        if receipt.process_status != ProcessStatus.TIMEOUT:
            block_status = "BLOCKED_INFRA"
        hypothesis.failure_domain = FailureDomain.INFRA_TRANSIENT
        hypothesis.scientific_verdict = sv
        hypothesis.status = block_status
        _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
        return ReconcileDecision(
            hypothesis_id=hypothesis.hypothesis_id,
            previous_fidelity=prev_fidelity,
            new_fidelity=hypothesis.fidelity,
            new_status=block_status,
            scientific_verdict=sv,
            action="HOLD_INFRA_TRANSIENT",
            graveyard_recorded=False,
        )

    # 3b. Failure isolation: ENV_CONFIG_MISMATCH (`cuda_unavailable`, missing venv/module, CLI flag mismatch)
    if eff_domain == FailureDomain.ENV_CONFIG_MISMATCH:
        env_sig = f"{receipt.stage.value}:{(receipt.failure_reason or '')[:120]}:{receipt.metrics.get('skip_reason', '')}"
        bstate = dict(hypothesis.budget_state or {})
        prev_sig = str(bstate.get("last_env_signature") or "")
        rechecks = int(bstate.get("env_rechecks_used", 0))
        max_rechecks = int(bstate.get("max_env_rechecks", 1))
        if prev_sig and prev_sig == env_sig:
            rechecks += 1
        else:
            bstate["last_env_signature"] = env_sig
        bstate["env_rechecks_used"] = rechecks
        hypothesis.budget_state = bstate
        hypothesis.failure_domain = FailureDomain.ENV_CONFIG_MISMATCH
        hypothesis.scientific_verdict = ScientificVerdict.UNKNOWN
        hypothesis.status = "BLOCKED_ENV"
        _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
        return ReconcileDecision(
            hypothesis_id=hypothesis.hypothesis_id,
            previous_fidelity=prev_fidelity,
            new_fidelity=hypothesis.fidelity,
            new_status="BLOCKED_ENV",
            scientific_verdict=ScientificVerdict.UNKNOWN,
            action="BLOCK_ENV_MISMATCH",
            graveyard_recorded=False,
            escalate=rechecks > max_rechecks,
            escalation_reason=(
                f"ENV_MISMATCH_PERSISTS ({env_sig})" if rechecks > max_rechecks else None
            ),
        )

    # 3c. Failure isolation: ENGINEERING_DEFECT on non-empirical stages (`CODE`, `CPU_GATE`, `IMPL_REVIEW`, `BLIND_AUDIT`, `DESIGN_REVIEW`, `RESULT_REVIEW`)
    # Route to bounded local repair (`code_repairs_used <= max_code_repairs`) instead of hypothesis tree explosion!
    if receipt.stage not in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM) or receipt.failure_domain == FailureDomain.ENGINEERING_DEFECT:
        bstate = dict(hypothesis.budget_state or {})
        repairs_used = int(bstate.get("code_repairs_used", 0))
        max_repairs = int(bstate.get("max_code_repairs", 2))
        hypothesis.failure_domain = FailureDomain.ENGINEERING_DEFECT
        hypothesis.scientific_verdict = ScientificVerdict.UNKNOWN
        if repairs_used < max_repairs:
            hypothesis.status = "NEEDS_REPAIR"
            _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
            return ReconcileDecision(
                hypothesis_id=hypothesis.hypothesis_id,
                previous_fidelity=prev_fidelity,
                new_fidelity=hypothesis.fidelity,
                new_status="NEEDS_REPAIR",
                scientific_verdict=ScientificVerdict.UNKNOWN,
                action="REQUEST_BOUNDED_REPAIR",
                graveyard_recorded=False,
            )
        hypothesis.status = "BLOCKED_REPAIR_BUDGET"
        _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
        return ReconcileDecision(
            hypothesis_id=hypothesis.hypothesis_id,
            previous_fidelity=prev_fidelity,
            new_fidelity=hypothesis.fidelity,
            new_status="BLOCKED_REPAIR_BUDGET",
            scientific_verdict=ScientificVerdict.UNKNOWN,
            action="ESCALATE_REPAIR_BUDGET_EXHAUSTED",
            graveyard_recorded=False,
            escalate=True,
            escalation_reason=f"CODE_REPAIR_BUDGET_EXHAUSTED ({repairs_used}/{max_repairs})",
        )

    # 4. Empirical UNKNOWN -> never fabricate pass; bounded diagnostic refinement or escalate.
    hypothesis.scientific_verdict = ScientificVerdict.UNKNOWN
    tokens_spent = used_tokens if used_tokens is not None else _db_used_tokens(db, goal.goal_id)
    budget_ok = goal.max_tokens_budget <= 0 or tokens_spent < goal.max_tokens_budget
    hyp_receipts = _db_list_receipts(db, goal.goal_id, hypothesis.hypothesis_id)
    if all(r.receipt_id != receipt.receipt_id for r in hyp_receipts):
        hyp_receipts.append(receipt)
    streak = _count_unknown_streak(hypothesis, existing, hyp_receipts)

    if budget_ok and streak <= max_unknown_refinements:
        diag = _spawn_diagnostic_refinement(
            goal, hypothesis, streak, existing, graveyard, db=db
        )
        if diag is not None:
            hypothesis.status = "DIAGNOSTIC_PENDING"
            _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
            return ReconcileDecision(
                hypothesis_id=hypothesis.hypothesis_id,
                previous_fidelity=prev_fidelity,
                new_fidelity=hypothesis.fidelity,
                new_status="DIAGNOSTIC_PENDING",
                scientific_verdict=ScientificVerdict.UNKNOWN,
                action="DIAGNOSTIC_REFINEMENT",
                spawned_hypotheses=[diag],
            )

    hypothesis.status = "ESCALATED"
    _db_register_or_update_hypothesis(db, hypothesis, prev_status=prev_status)
    return ReconcileDecision(
        hypothesis_id=hypothesis.hypothesis_id,
        previous_fidelity=prev_fidelity,
        new_fidelity=hypothesis.fidelity,
        new_status="ESCALATED",
        scientific_verdict=ScientificVerdict.UNKNOWN,
        action="ESCALATE_UNKNOWN",
        escalate=True,
        escalation_reason=(
            "TOKEN_BUDGET_EXHAUSTED"
            if not budget_ok
            else f"UNKNOWN_STREAK_EXHAUSTED (streak={streak} > {max_unknown_refinements})"
        ),
    )


def _validate_envelope_goal_ownership(
    db: Any,
    goal: ResearchGoal,
    envelope: LeadDecisionEnvelope,
) -> list[dict[str, Any]]:
    """Pre-flight validate envelope and EVERY referenced entity ID against `goal.goal_id`.

    Returns a list of violation dicts; if non-empty, caller MUST abort before any DB write!
    """
    violations: list[dict[str, Any]] = []
    if envelope.goal_id and str(envelope.goal_id) != str(goal.goal_id):
        violations.append(
            {
                "action": "ENVELOPE",
                "reason": (
                    f"CROSS_GOAL_OWNERSHIP_VIOLATION: envelope.goal_id={envelope.goal_id!r} "
                    f"does not match goal.goal_id={goal.goal_id!r}"
                ),
            }
        )
        return violations

    def _check_hyp_id(act_name: str, raw_hid: Any, field_name: str) -> None:
        if not raw_hid:
            return
        hid_s = str(raw_hid).strip()
        if not hid_s:
            return
        if hasattr(db, "get_goal_hypothesis"):
            try:
                db.get_goal_hypothesis(goal.goal_id, hid_s)
            except ValueError as exc:
                violations.append(
                    {
                        "action": act_name,
                        "field": field_name,
                        "hypothesis_id": hid_s,
                        "reason": f"CROSS_GOAL_OWNERSHIP_VIOLATION: {exc}",
                    }
                )
        elif hasattr(db, "get_hypothesis"):
            h_obj = db.get_hypothesis(hid_s)
            if h_obj is not None and h_obj.goal_id != goal.goal_id:
                violations.append(
                    {
                        "action": act_name,
                        "field": field_name,
                        "hypothesis_id": hid_s,
                        "reason": (
                            f"CROSS_GOAL_OWNERSHIP_VIOLATION: {field_name}={hid_s!r} belongs to "
                            f"goal {h_obj.goal_id!r}, not {goal.goal_id!r}"
                        ),
                    }
                )

    def _check_wf_id(act_name: str, raw_wid: Any, field_name: str) -> None:
        if not raw_wid:
            return
        wid_s = str(raw_wid).strip()
        if not wid_s:
            return
        if hasattr(db, "get_goal_workflow"):
            try:
                db.get_goal_workflow(goal.goal_id, wid_s)
            except ValueError as exc:
                violations.append(
                    {
                        "action": act_name,
                        "field": field_name,
                        "workflow_id": wid_s,
                        "reason": f"CROSS_GOAL_OWNERSHIP_VIOLATION: {exc}",
                    }
                )
        elif hasattr(db, "get_workflow"):
            w_obj = db.get_workflow(wid_s)
            if w_obj is not None and w_obj.goal_id != goal.goal_id:
                violations.append(
                    {
                        "action": act_name,
                        "field": field_name,
                        "workflow_id": wid_s,
                        "reason": (
                            f"CROSS_GOAL_OWNERSHIP_VIOLATION: {field_name}={wid_s!r} belongs to "
                            f"goal {w_obj.goal_id!r}, not {goal.goal_id!r}"
                        ),
                    }
                )

    def _check_task_id(act_name: str, raw_tid: Any, field_name: str) -> None:
        if not raw_tid:
            return
        tid_s = str(raw_tid).strip()
        if not tid_s:
            return
        if hasattr(db, "get_goal_task"):
            try:
                db.get_goal_task(goal.goal_id, tid_s)
            except ValueError as exc:
                violations.append(
                    {
                        "action": act_name,
                        "field": field_name,
                        "task_id": tid_s,
                        "reason": f"CROSS_GOAL_OWNERSHIP_VIOLATION: {exc}",
                    }
                )
        elif hasattr(db, "get_task"):
            t_obj = db.get_task(tid_s)
            if t_obj is not None and t_obj.goal_id != goal.goal_id:
                violations.append(
                    {
                        "action": act_name,
                        "field": field_name,
                        "task_id": tid_s,
                        "reason": (
                            f"CROSS_GOAL_OWNERSHIP_VIOLATION: {field_name}={tid_s!r} belongs to "
                            f"goal {t_obj.goal_id!r}, not {goal.goal_id!r}"
                        ),
                    }
                )

    for act in envelope.actions:
        act_name = act.action.value
        params = dict(act.parameters or {})
        prop = params.get("proposal") if isinstance(params.get("proposal"), Mapping) else {}
        for g_field, g_val in (("parameters.goal_id", params.get("goal_id")), ("proposal.goal_id", prop.get("goal_id"))):
            if g_val and str(g_val) != str(goal.goal_id):
                violations.append(
                    {
                        "action": act_name,
                        "field": g_field,
                        "reason": f"CROSS_GOAL_OWNERSHIP_VIOLATION: {g_field}={g_val!r} != {goal.goal_id!r}",
                    }
                )
        _check_hyp_id(act_name, act.target_hypothesis_id, "target_hypothesis_id")
        _check_hyp_id(act_name, params.get("hypothesis_id"), "parameters.hypothesis_id")
        _check_hyp_id(act_name, params.get("target_hypothesis_id"), "parameters.target_hypothesis_id")
        _check_hyp_id(act_name, params.get("parent_id"), "parameters.parent_id")
        _check_hyp_id(act_name, params.get("parent_hypothesis_id"), "parameters.parent_hypothesis_id")
        _check_hyp_id(act_name, prop.get("hypothesis_id"), "proposal.hypothesis_id")
        _check_hyp_id(act_name, prop.get("parent_id"), "proposal.parent_id")
        _check_hyp_id(act_name, prop.get("parent_hypothesis_id"), "proposal.parent_hypothesis_id")
        for seq_key in ("synthesized_hypothesis_ids", "hypothesis_ids"):
            raw_seq = params.get(seq_key)
            if isinstance(raw_seq, Sequence) and not isinstance(raw_seq, (str, bytes)):
                for item_hid in raw_seq:
                    _check_hyp_id(act_name, item_hid, f"parameters.{seq_key}")
        _check_wf_id(act_name, params.get("workflow_id"), "parameters.workflow_id")
        _check_wf_id(act_name, params.get("parent_workflow_id"), "parameters.parent_workflow_id")
        _check_task_id(act_name, params.get("task_id"), "parameters.task_id")

        eff_target_hid = str(act.target_hypothesis_id or params.get("hypothesis_id") or "").strip()
        for nested_label, nested_map in (
            ("parameters.experiment_plan", params.get("experiment_plan")),
            ("proposal.experiment_plan", prop.get("experiment_plan")),
            ("parameters.research_plan", params.get("research_plan")),
            ("parameters.execution_plan", params.get("execution_plan")),
        ):
            if not isinstance(nested_map, Mapping):
                continue
            ngid = nested_map.get("goal_id")
            if ngid and str(ngid) != str(goal.goal_id):
                violations.append(
                    {
                        "action": act_name,
                        "field": f"{nested_label}.goal_id",
                        "reason": f"CROSS_GOAL_OWNERSHIP_VIOLATION: {nested_label}.goal_id={ngid!r} != {goal.goal_id!r}",
                    }
                )
            nhid = nested_map.get("hypothesis_id")
            if nhid:
                _check_hyp_id(act_name, nhid, f"{nested_label}.hypothesis_id")
                if eff_target_hid and str(nhid).strip() != eff_target_hid:
                    violations.append(
                        {
                            "action": act_name,
                            "field": f"{nested_label}.hypothesis_id",
                            "reason": (
                                f"CROSS_GOAL_OWNERSHIP_VIOLATION: {nested_label}.hypothesis_id={nhid!r} "
                                f"!= target_hypothesis_id={eff_target_hid!r}"
                            ),
                        }
                    )
            nwfid = nested_map.get("workflow_id")
            if nwfid:
                _check_wf_id(act_name, nwfid, f"{nested_label}.workflow_id")
            exp_pid = str(nested_map.get("exp_plan_id") or "").strip()
            if exp_pid and hasattr(db, "get_experiment_plan"):
                ex_ep = db.get_experiment_plan(exp_pid)
                if ex_ep is not None and (
                    ex_ep.goal_id != goal.goal_id
                    or (eff_target_hid and ex_ep.hypothesis_id != eff_target_hid)
                ):
                    violations.append(
                        {
                            "action": act_name,
                            "field": f"{nested_label}.exp_plan_id",
                            "reason": (
                                f"CROSS_GOAL_OWNERSHIP_VIOLATION: exp_plan_id={exp_pid!r} belongs to "
                                f"({ex_ep.goal_id!r}, {ex_ep.hypothesis_id!r})"
                            ),
                        }
                    )
            r_pid = str(nested_map.get("plan_id") or "").strip()
            if r_pid and hasattr(db, "get_research_plan"):
                ex_rp = db.get_research_plan(r_pid)
                if ex_rp is not None and ex_rp.goal_id != goal.goal_id:
                    violations.append(
                        {
                            "action": act_name,
                            "field": f"{nested_label}.plan_id",
                            "reason": f"CROSS_GOAL_OWNERSHIP_VIOLATION: plan_id={r_pid!r} belongs to {ex_rp.goal_id!r}",
                        }
                    )
            x_pid = str(nested_map.get("exec_plan_id") or "").strip()
            if x_pid and hasattr(db, "get_execution_plan"):
                ex_xp = db.get_execution_plan(x_pid)
                if ex_xp is not None and (
                    ex_xp.goal_id != goal.goal_id
                    or (eff_target_hid and ex_xp.hypothesis_id != eff_target_hid)
                ):
                    violations.append(
                        {
                            "action": act_name,
                            "field": f"{nested_label}.exec_plan_id",
                            "reason": (
                                f"CROSS_GOAL_OWNERSHIP_VIOLATION: exec_plan_id={x_pid!r} belongs to "
                                f"({ex_xp.goal_id!r}, {ex_xp.hypothesis_id!r})"
                            ),
                        }
                    )

    return violations


ALLOWED_EXPERIMENT_PLAN_PATCH_KEYS: frozenset[str] = frozenset(
    {
        "falsifiable_claim",
        "control_baseline",
        "venv_python",
        "env_vars",
        "required_hardware",
        "min_vram_mb",
        "gate_timeout_s",
        "screen_timeout_s",
        "confirm_timeout_s",
        "confound_checks",
        "review_status",
    }
)


def _collect_descendant_hids_in_db(db: Any, goal_id: str, root_hid: str) -> set[str]:
    if not hasattr(db, "list_hypotheses"):
        return {root_hid}
    all_hyps = db.list_hypotheses(goal_id=goal_id)
    by_parent: dict[str, list[str]] = {}
    for h in all_hyps:
        if h.parent_id:
            by_parent.setdefault(str(h.parent_id), []).append(h.hypothesis_id)
    seen = {root_hid}
    queue = list(by_parent.get(root_hid, []))
    while queue:
        cur = queue.pop(0)
        if cur in seen:
            continue
        seen.add(cur)
        queue.extend(by_parent.get(cur, []))
    return seen


def _invalidate_synthesis_workflows_for_hids(
    db: Any,
    goal: ResearchGoal,
    affected_hids: set[str],
    *,
    reason: str,
    next_plan_epoch: int,
) -> tuple[set[str], list[str]]:
    """Cancel any active S3 synthesis workflows & tasks whose `synthesized_hypothesis_ids` intersect `affected_hids`."""
    cancelled_wf_ids: set[str] = set()
    cancelled_task_ids: list[str] = []
    if not hasattr(db, "list_workflows"):
        return cancelled_wf_ids, cancelled_task_ids
    now = time.time()
    for wf_obj in db.list_workflows(goal_id=goal.goal_id):
        if wf_obj.status not in ("READY", "RUNNING", "WAITING"):
            continue
        syn_hids = {
            str(x).strip()
            for x in (wf_obj.config.get("synthesized_hypothesis_ids") or [])
            if str(x).strip()
        }
        is_syn = str(wf_obj.config.get("target_kind") or "") == "goal_synthesis" or bool(syn_hids)
        if is_syn and (syn_hids & affected_hids or wf_obj.hypothesis_id in affected_hids):
            cancelled_wf_ids.add(wf_obj.workflow_id)
            db.update_workflow_state(
                wf_obj.workflow_id,
                status="CANCELLED",
                state_patch={
                    "superseded_by_plan_epoch": next_plan_epoch,
                    "cancel_reason": reason,
                },
                now=now,
            )
            syn_key = str(wf_obj.config.get("synthesis_key") or f"syn:{wf_obj.workflow_id}")
            if hasattr(db, "upsert_knowledge_item"):
                db.upsert_knowledge_item(
                    goal.goal_id,
                    "synthesis_pending",
                    syn_key,
                    {
                        "status": "SUPERSEDED_BY_SOURCE_UPDATE",
                        "workflow_id": wf_obj.workflow_id,
                        "synthesized_hypothesis_ids": sorted(syn_hids),
                        "reason": reason,
                    },
                    hypothesis_id=wf_obj.hypothesis_id,
                    goal_version=int(goal.version),
                    now=now,
                )
    if cancelled_wf_ids and hasattr(db, "list_tasks"):
        for t in db.list_tasks(goal_id=goal.goal_id):
            if t.workflow_id in cancelled_wf_ids and t.status in ("READY", "DISPATCHED", "RUNNING"):
                try:
                    db.transition_task_status(
                        t.task_id,
                        ("READY", "DISPATCHED", "RUNNING"),
                        "CANCELLED",
                        finished_ts=now,
                        now=now,
                    )
                    cancelled_task_ids.append(t.task_id)
                    if t.hq_job_id is not None and hasattr(db, "record_pending_hq_cancellation"):
                        db.record_pending_hq_cancellation(
                            hq_job_id=int(t.hq_job_id),
                            entity_kind="task",
                            entity_id=t.task_id,
                            goal_id=goal.goal_id,
                            hypothesis_id=t.hypothesis_id,
                            workflow_id=t.workflow_id or "",
                            epoch=int(goal.human_epoch),
                            plan_epoch=next_plan_epoch,
                            reason=reason,
                            state="CANCEL_PENDING",
                            now=now,
                        )
                except Exception:
                    pass
    return cancelled_wf_ids, cancelled_task_ids


def apply_scientific_actions(
    db: Any,
    goal: ResearchGoal,
    envelope: LeadDecisionEnvelope,
    *,
    workflow_runtime: Any = None,
    source: str = "research_lead",
    raise_on_cross_goal: bool = False,
) -> dict[str, Any]:
    """Execute all 13 typed closed-loop `ActionKind`s emitted by ResearchLead / ResearchDirector."""
    summary: dict[str, Any] = {
        "role": envelope.role.value,
        "uncertainty_summary": envelope.uncertainty_summary,
        "accepted_hypothesis_ids": [],
        "revised_hypothesis_ids": [],
        "repaired_hypothesis_ids": [],
        "pruned_hypothesis_ids": [],
        "terminated_hypothesis_ids": [],
        "blocked_unverified_refutations": [],
        "switched_workflows": [],
        "emitted_task_ids": [],
        "director_requests": [],
        "owner_escalations": [],
        "completed_milestones": [],
        "syntheses_recorded": 0,
        "synthesis_review_workflows": [],
        "continued": 0,
        "cross_goal_violation": False,
        "rejected": [],
    }
    tx_cm = db._tx() if hasattr(db, "_tx") else None

    def _run_in_tx() -> dict[str, Any]:
        violations = _validate_envelope_goal_ownership(db, goal, envelope)
        if violations:
            summary["cross_goal_violation"] = True
            summary["rejected"].extend(violations)
            if raise_on_cross_goal:
                raise ValueError(violations[0]["reason"])
            return summary

        plan_mgr = PlanLifecycleManager(db) if hasattr(db, "save_research_plan") else None

        for act in envelope.actions:
            kind = act.action
            params = dict(act.parameters or {})
            target_hid = act.target_hypothesis_id or params.get("hypothesis_id")
            if target_hid and hasattr(db, "get_goal_hypothesis"):
                target_hyp = db.get_goal_hypothesis(goal.goal_id, str(target_hid))
            elif target_hid and hasattr(db, "get_hypothesis"):
                cand_h = db.get_hypothesis(str(target_hid))
                target_hyp = cand_h if (cand_h is not None and cand_h.goal_id == goal.goal_id) else None
            else:
                target_hyp = None

            if kind == ActionKind.CONTINUE:
                summary["continued"] += 1
                if hasattr(db, "upsert_knowledge_item") and act.rationale:
                    db.upsert_knowledge_item(
                        goal.goal_id,
                        "lead_continue",
                        f"cont:{uuid.uuid4().hex[:8]}",
                        {"rationale": act.rationale, "target": target_hid},
                        hypothesis_id=str(target_hid) if target_hid else None,
                        goal_version=int(goal.version),
                    )

            elif kind == ActionKind.REVISE_HYPOTHESIS:
                if target_hyp is None:
                    summary["rejected"].append({"action": kind.value, "reason": "missing target_hypothesis_id"})
                    continue
                now_rev = time.time()
                affected_hids = _collect_descendant_hids_in_db(db, goal.goal_id, target_hyp.hypothesis_id)
                next_plan_epoch = int(getattr(goal, "plan_epoch", 0) or 0) + 1
                goal.plan_epoch = next_plan_epoch
                if hasattr(db, "save_goal"):
                    db.save_goal(goal)

                syn_wf_ids, _syn_tids = _invalidate_synthesis_workflows_for_hids(
                    db,
                    goal,
                    affected_hids,
                    reason=f"revise_hypothesis:{target_hyp.hypothesis_id}:pe{next_plan_epoch}",
                    next_plan_epoch=next_plan_epoch,
                )
                if hasattr(db, "list_workflows"):
                    for wf_obj in db.list_workflows(goal_id=goal.goal_id):
                        if wf_obj.status in ("READY", "RUNNING", "WAITING") and wf_obj.hypothesis_id in affected_hids:
                            db.update_workflow_state(
                                wf_obj.workflow_id,
                                status="CANCELLED",
                                state_patch={"superseded_by_plan_epoch": next_plan_epoch},
                                now=now_rev,
                            )
                if hasattr(db, "list_tasks"):
                    for t_obj in db.list_tasks(goal_id=goal.goal_id):
                        if t_obj.status not in ("READY", "DISPATCHED", "RUNNING"):
                            continue
                        if t_obj.hypothesis_id in affected_hids or (t_obj.workflow_id and t_obj.workflow_id in syn_wf_ids):
                            try:
                                db.transition_task_status(
                                    t_obj.task_id,
                                    ("READY", "DISPATCHED", "RUNNING"),
                                    "CANCELLED",
                                    finished_ts=now_rev,
                                    now=now_rev,
                                )
                                if t_obj.hq_job_id is not None and hasattr(db, "record_pending_hq_cancellation"):
                                    db.record_pending_hq_cancellation(
                                        hq_job_id=int(t_obj.hq_job_id),
                                        entity_kind="task",
                                        entity_id=t_obj.task_id,
                                        attempt=int(t_obj.attempt),
                                        goal_id=goal.goal_id,
                                        hypothesis_id=t_obj.hypothesis_id,
                                        workflow_id=t_obj.workflow_id or "",
                                        epoch=int(goal.human_epoch),
                                        plan_epoch=next_plan_epoch,
                                        reason=f"revise_hypothesis:{target_hyp.hypothesis_id}:pe{next_plan_epoch}",
                                        state="CANCEL_PENDING",
                                        now=now_rev,
                                    )
                                if hasattr(db, "abort_cancelled_task_promotions"):
                                    db.abort_cancelled_task_promotions(
                                        goal.goal_id,
                                        t_obj.hypothesis_id,
                                        t_obj.task_id,
                                        reason=f"revise_hypothesis:{target_hyp.hypothesis_id}:pe{next_plan_epoch}",
                                        now=now_rev,
                                    )
                            except Exception:
                                pass
                        else:
                            try:
                                synced_route = (
                                    {
                                        **dict(t_obj.pinned_route),
                                        "plan_epoch": next_plan_epoch,
                                        "exec_fence_plan_epoch": next_plan_epoch,
                                        "exec_fence_human_epoch": int(goal.human_epoch),
                                    }
                                    if isinstance(t_obj.pinned_route, Mapping)
                                    else {
                                        "plan_epoch": next_plan_epoch,
                                        "exec_fence_plan_epoch": next_plan_epoch,
                                        "exec_fence_human_epoch": int(goal.human_epoch),
                                    }
                                )
                                db.transition_task_status(
                                    t_obj.task_id,
                                    t_obj.status,
                                    t_obj.status,
                                    plan_epoch=next_plan_epoch,
                                    human_epoch=int(goal.human_epoch),
                                    pinned_route=synced_route,
                                    now=now_rev,
                                )
                            except Exception:
                                pass

                preserved_by_hid: dict[str, list[str]] = {}
                next_sci_rev_by_hid: dict[str, int] = {}
                for aff_hid in sorted(affected_hids):
                    aff_node = (
                        db.get_goal_hypothesis(goal.goal_id, aff_hid)
                        if hasattr(db, "get_goal_hypothesis")
                        else (db.get_hypothesis(aff_hid) if hasattr(db, "get_hypothesis") else None)
                    )
                    cur_sci_rev = int(((aff_node.params if aff_node is not None else {}) or {}).get("lineage_sci_revision", 0) or 0)
                    next_sci_rev_by_hid[aff_hid] = cur_sci_rev + 1
                    preserved_by_hid[aff_hid] = [
                        r.receipt_id
                        for r in _db_list_receipts(db, goal.goal_id, aff_hid)
                        if r.stage in (StageKind.CODE, StageKind.CPU_GATE)
                        and r.process_status == ProcessStatus.EXIT_OK
                        and r.engineering_verdict == EngineeringVerdict.PASS
                        and r.audit_verdict != AuditVerdict.FAIL
                        and r.evidence_validity
                        not in (
                            EvidenceValidity.SUPERSEDED_EPOCH,
                            EvidenceValidity.HUMAN_INVALIDATED,
                            EvidenceValidity.STALE_GOAL_VERSION,
                        )
                    ]
                preserved_rcpts = preserved_by_hid.get(target_hyp.hypothesis_id, [])
                if hasattr(db, "upsert_knowledge_item"):
                    for aff_hid, node_pres in preserved_by_hid.items():
                        db.upsert_knowledge_item(
                            goal.goal_id,
                            "plan_lineage",
                            f"pe:{next_plan_epoch}:{aff_hid}",
                            {
                                "hypothesis_id": aff_hid,
                                "plan_epoch": next_plan_epoch,
                                "human_epoch": int(goal.human_epoch),
                                "sci_revision": next_sci_rev_by_hid[aff_hid],
                                "preserved_upstream_receipt_ids": node_pres,
                                "source": "autonomous_revise_hypothesis",
                            },
                            hypothesis_id=aff_hid,
                            goal_version=int(goal.version),
                            now=now_rev,
                        )

                patch = dict(params.get("params_patch") or {})
                patch["lineage_plan_epoch"] = next_plan_epoch
                patch["lineage_human_epoch"] = int(goal.human_epoch)
                patch["lineage_sci_revision"] = next_sci_rev_by_hid[target_hyp.hypothesis_id]
                patch["preserved_receipt_ids"] = preserved_rcpts
                if act.workflow_primitive:
                    patch["workflow_primitive"] = act.workflow_primitive
                if act.review_tier is not None:
                    patch["review_tier"] = act.review_tier.value
                if hasattr(db, "update_hypothesis_params"):
                    target_hyp = db.update_hypothesis_params(target_hyp.hypothesis_id, patch)
                    for aff_hid, node_pres in preserved_by_hid.items():
                        if aff_hid != target_hyp.hypothesis_id:
                            db.update_hypothesis_params(
                                aff_hid,
                                {
                                    "lineage_plan_epoch": next_plan_epoch,
                                    "lineage_human_epoch": int(goal.human_epoch),
                                    "lineage_sci_revision": next_sci_rev_by_hid[aff_hid],
                                    "preserved_receipt_ids": node_pres,
                                },
                            )
                if plan_mgr is not None and isinstance(params.get("experiment_plan"), Mapping):
                    raw_ep_map = dict(params["experiment_plan"])
                    whitelisted_ep = {
                        k: v for k, v in raw_ep_map.items() if k in ALLOWED_EXPERIMENT_PLAN_PATCH_KEYS
                    }
                    sealed_ep = {
                        "falsifiable_claim": target_hyp.statement,
                        **whitelisted_ep,
                        "exp_plan_id": f"eplan_{uuid.uuid4().hex[:12]}",
                        "goal_id": goal.goal_id,
                        "hypothesis_id": target_hyp.hypothesis_id,
                        "goal_version": int(goal.version),
                        "human_epoch": int(goal.human_epoch),
                        "plan_epoch": next_plan_epoch,
                    }
                    plan_mgr.create_or_revise_experiment_plan(
                        goal,
                        target_hyp,
                        plan=ExperimentPlan.from_dict(sealed_ep),
                    )
                summary["revised_hypothesis_ids"].append(target_hyp.hypothesis_id)

            elif kind in (
                ActionKind.PROPOSE_COMPETING_ROUTE,
                ActionKind.ADD_THEORETICAL_DERIVATION,
                ActionKind.ADD_DISCRIMINATING_EXPERIMENT,
            ):
                raw_prop = dict(params.get("proposal") or params)
                raw_prop.pop("proposal", None)
                if kind == ActionKind.ADD_THEORETICAL_DERIVATION:
                    raw_prop.setdefault("kind", HypothesisKind.COMPETING_ROUTE.value)
                    tags = list(raw_prop.get("tags") or [])
                    for t in ("theory", "derivation"):
                        if t not in tags:
                            tags.append(t)
                    raw_prop["tags"] = tags
                elif kind == ActionKind.ADD_DISCRIMINATING_EXPERIMENT:
                    raw_prop.setdefault("kind", HypothesisKind.FALSIFICATION.value)
                    if target_hid and not raw_prop.get("parent_id"):
                        raw_prop["parent_id"] = str(target_hid)
                else:
                    raw_prop.setdefault("kind", HypothesisKind.COMPETING_ROUTE.value)

                cparams = dict(raw_prop.get("params") or {})
                if not cparams.get("deterministic_no_llm"):
                    cparams.setdefault("requires_llm_code", True)
                if act.workflow_primitive:
                    cparams["workflow_primitive"] = act.workflow_primitive
                if act.review_tier is not None:
                    cparams["review_tier"] = act.review_tier.value
                raw_prop["params"] = cparams
                if act.uncertainty_id:
                    raw_prop["uncertainty_id"] = act.uncertainty_id

                batch = import_proposals_json(
                    goal,
                    [raw_prop],
                    db=db,
                    persist=True,
                    source=source,
                )
                for h in batch.accepted:
                    if hasattr(db, "update_hypothesis_v3_state") and (raw_prop.get("tags") or act.uncertainty_id):
                        db.update_hypothesis_v3_state(
                            h.hypothesis_id,
                            tags=raw_prop.get("tags"),
                            uncertainty_id=act.uncertainty_id,
                        )
                    summary["accepted_hypothesis_ids"].append(h.hypothesis_id)
                summary["rejected"].extend(batch.rejected)

            elif kind == ActionKind.REPAIR_IMPLEMENTATION:
                if target_hyp is None:
                    summary["rejected"].append({"action": kind.value, "reason": "missing target_hypothesis_id"})
                    continue
                bstate = dict(target_hyp.budget_state or {})
                used = int(bstate.get("code_repairs_used", 0))
                max_r = int(bstate.get("max_code_repairs", 2))
                if used >= max_r:
                    summary["rejected"].append(
                        {
                            "action": kind.value,
                            "hypothesis_id": target_hyp.hypothesis_id,
                            "reason": f"code_repairs_used ({used}) >= max_code_repairs ({max_r})",
                        }
                    )
                    continue
                bstate["code_repairs_used"] = used + 1
                if hasattr(db, "transition_hypothesis_status"):
                    target_hyp = db.transition_hypothesis_status(
                        target_hyp.hypothesis_id,
                        target_hyp.status,
                        "PENDING",
                        failure_domain=FailureDomain.ENGINEERING_DEFECT,
                        budget_state_patch=bstate,
                    )
                if workflow_runtime is not None and hasattr(workflow_runtime, "enqueue_bounded_repair"):
                    rep_tasks = workflow_runtime.enqueue_bounded_repair(
                        goal,
                        target_hyp,
                        repair_notes=act.rationale or str(params.get("repair_notes") or ""),
                    )
                    summary["emitted_task_ids"].extend(t.task_id for t in rep_tasks)
                elif hasattr(db, "enqueue_task"):
                    attempt_n = used + 2
                    rtask = db.enqueue_task(
                        StageTask(
                            task_id=f"tsk-{target_hyp.hypothesis_id}-lead-repair-a{attempt_n}-{uuid.uuid4().hex[:5]}",
                            idempotency_key=f"{goal.goal_id}:{target_hyp.hypothesis_id}:lead_repair:a{attempt_n}",
                            goal_id=goal.goal_id,
                            hypothesis_id=target_hyp.hypothesis_id,
                            stage=StageKind.CODE,
                            deps=[],
                            attempt=attempt_n,
                            goal_version=int(goal.version),
                            human_epoch=int(goal.human_epoch),
                            plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                            step_key=f"code_repair_{used + 1}",
                        )
                    )
                    summary["emitted_task_ids"].append(rtask.task_id)
                summary["repaired_hypothesis_ids"].append(target_hyp.hypothesis_id)

            elif kind == ActionKind.PRUNE_BRANCH:
                if target_hyp is None:
                    summary["rejected"].append({"action": kind.value, "reason": "missing target_hypothesis_id"})
                    continue
                prev_s = target_hyp.status
                target_hyp.status = "PRUNED"
                target_hyp.scientific_verdict = ScientificVerdict.NOT_APPLICABLE
                _db_register_or_update_hypothesis(db, target_hyp, prev_status=prev_s)
                _db_cancel_hypothesis_tasks(db, target_hyp.hypothesis_id, goal.goal_id)
                _invalidate_synthesis_workflows_for_hids(
                    db,
                    goal,
                    {target_hyp.hypothesis_id},
                    reason=f"prune_branch:{target_hyp.hypothesis_id}",
                    next_plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                )
                summary["pruned_hypothesis_ids"].append(target_hyp.hypothesis_id)

            elif kind == ActionKind.SYNTHESIZE_RESULTS:
                syn_key = f"syn:{act.uncertainty_id or uuid.uuid4().hex[:8]}"
                raw_syn_hids = params.get("synthesized_hypothesis_ids") or params.get("hypothesis_ids") or []
                syn_hids = [
                    str(x).strip()
                    for x in (raw_syn_hids if isinstance(raw_syn_hids, Sequence) and not isinstance(raw_syn_hids, (str, bytes)) else [])
                    if str(x).strip()
                ]
                needs_s3_review = bool(
                    params.get("require_synthesis_review", False)
                    or len(syn_hids) >= 2
                )
                if (
                    needs_s3_review
                    and workflow_runtime is not None
                    and hasattr(workflow_runtime, "spawn_synthesis_review")
                ):
                    anchor_hyp = target_hyp
                    if anchor_hyp is None and syn_hids and hasattr(db, "get_goal_hypothesis"):
                        anchor_hyp = db.get_goal_hypothesis(goal.goal_id, syn_hids[0])
                    if anchor_hyp is None and hasattr(db, "list_hypotheses"):
                        all_gh = db.list_hypotheses(goal_id=goal.goal_id)
                        anchor_hyp = all_gh[0] if all_gh else None
                    if anchor_hyp is not None:
                        if plan_mgr is not None and params.get("thesis"):
                            plan_mgr.create_research_plan(
                                goal,
                                thesis=str(params["thesis"]),
                                uncertainties=params.get("uncertainties") or (),
                                competing_routes=params.get("competing_routes") or (),
                                milestone_criteria=params.get("milestone_criteria") or (),
                                status="DRAFT",
                            )
                        syn_wf, syn_tasks = workflow_runtime.spawn_synthesis_review(
                            goal,
                            anchor_hyp,
                            synthesized_hypothesis_ids=syn_hids or [anchor_hyp.hypothesis_id],
                            synthesis_key=syn_key,
                            thesis=str(params.get("thesis") or act.rationale or goal.title),
                            rationale=act.rationale,
                            findings=params,
                            review_tier=act.review_tier,
                        )
                        summary["synthesis_review_workflows"].append(syn_wf.workflow_id)
                        summary["emitted_task_ids"].extend(t.task_id for t in syn_tasks)
                        summary["syntheses_recorded"] += 1
                        continue

                summary["syntheses_recorded"] += 1
                if hasattr(db, "upsert_knowledge_item"):
                    db.upsert_knowledge_item(
                        goal.goal_id,
                        "synthesis",
                        syn_key,
                        {
                            "uncertainty_id": act.uncertainty_id,
                            "rationale": act.rationale,
                            "findings": params,
                        },
                        hypothesis_id=str(target_hid) if target_hid else None,
                        goal_version=int(goal.version),
                    )
                if plan_mgr is not None and params.get("thesis"):
                    plan_mgr.create_research_plan(
                        goal,
                        thesis=str(params["thesis"]),
                        uncertainties=params.get("uncertainties") or (),
                        competing_routes=params.get("competing_routes") or (),
                        milestone_criteria=params.get("milestone_criteria") or (),
                        status="APPROVED",
                    )

            elif kind == ActionKind.SWITCH_WORKFLOW:
                if target_hyp is None:
                    summary["rejected"].append({"action": kind.value, "reason": "missing target_hypothesis_id"})
                    continue
                new_prim = str(act.workflow_primitive or params.get("primitive") or "x-boost")
                if workflow_runtime is not None and hasattr(workflow_runtime, "switch_hypothesis_workflow"):
                    new_wf, new_tasks = workflow_runtime.switch_hypothesis_workflow(
                        goal,
                        target_hyp,
                        new_primitive=new_prim,
                        reason=act.rationale,
                        config_override=params.get("config") if isinstance(params.get("config"), Mapping) else None,
                    )
                    summary["switched_workflows"].append(
                        {"hypothesis_id": target_hyp.hypothesis_id, "workflow_id": new_wf.workflow_id, "primitive": new_wf.primitive}
                    )
                    summary["emitted_task_ids"].extend(t.task_id for t in new_tasks)
                else:
                    if hasattr(db, "update_hypothesis_params"):
                        db.update_hypothesis_params(target_hyp.hypothesis_id, {"workflow_primitive": new_prim})
                    summary["switched_workflows"].append(
                        {"hypothesis_id": target_hyp.hypothesis_id, "primitive": new_prim}
                    )

            elif kind == ActionKind.REQUEST_DIRECTOR_ADJUDICATION:
                if hasattr(db, "enqueue_decision_request"):
                    dreq = db.enqueue_decision_request(
                        goal.goal_id,
                        tier="L2",
                        role=ScientificRole.RESEARCH_DIRECTOR,
                        trigger_kind="DIRECTOR_ADJUDICATION",
                        goal_version=int(goal.version),
                        human_epoch=int(goal.human_epoch),
                        payload={
                            "rationale": act.rationale,
                            "target_hypothesis_id": target_hid,
                            "uncertainty_id": act.uncertainty_id,
                            "plan_epoch": int(getattr(goal, "plan_epoch", 0) or 0),
                            **params,
                        },
                    )
                    summary["director_requests"].append(dreq["req_id"])

            elif kind == ActionKind.TERMINATE_REFUTED_ROUTE:
                if target_hyp is None:
                    summary["rejected"].append({"action": kind.value, "reason": "missing target_hypothesis_id"})
                    continue
                if hasattr(db, "get_authoritative_empirical_receipts"):
                    auth_rcpts = db.get_authoritative_empirical_receipts(goal, target_hyp)
                else:
                    raw_rcpts = _db_list_receipts(db, goal.goal_id, target_hyp.hypothesis_id)
                    auth_rcpts = [
                        r for r in raw_rcpts if is_valid_empirical_outcome(r, goal, hyp=target_hyp)
                    ]
                latest_emp_rcpt = auth_rcpts[-1] if auth_rcpts else None
                neg_rcpt = (
                    latest_emp_rcpt
                    if (
                        latest_emp_rcpt is not None
                        and is_authenticated_negative_empirical_outcome(
                            latest_emp_rcpt, target_hyp, goal=goal
                        )
                    )
                    else None
                )
                if neg_rcpt is None:
                    summary["blocked_unverified_refutations"].append(target_hyp.hypothesis_id)
                    summary["rejected"].append(
                        {
                            "action": kind.value,
                            "hypothesis_id": target_hyp.hypothesis_id,
                            "reason": (
                                "TERMINATE_REFUTED_ROUTE rejected: requires authoritative negative empirical "
                                "outcome in current hypothesis plan lineage (REFUTED, VALID_NEGATIVE, or verified kill hit); "
                                f"latest={latest_emp_rcpt.scientific_verdict.value if latest_emp_rcpt else 'NONE'}"
                            ),
                        }
                    )
                    continue

                verdict = (
                    ScientificVerdict.VALID_NEGATIVE
                    if neg_rcpt.scientific_verdict == ScientificVerdict.VALID_NEGATIVE
                    else ScientificVerdict.REFUTED
                )
                _db_record_graveyard(
                    db,
                    target_hyp,
                    act.rationale or neg_rcpt.failure_reason or "TERMINATE_REFUTED_ROUTE",
                    {
                        "receipt_id": neg_rcpt.receipt_id,
                        "metrics": neg_rcpt.metrics,
                        "rationale": act.rationale,
                    },
                    scientific_verdict=verdict,
                    new_status=verdict.value,
                )
                if verdict == ScientificVerdict.VALID_NEGATIVE:
                    _db_record_valid_negative_knowledge(
                        db, goal, target_hyp, neg_rcpt, act.rationale or "VALID_NEGATIVE"
                    )
                _db_cancel_hypothesis_tasks(db, target_hyp.hypothesis_id, goal.goal_id)
                _invalidate_synthesis_workflows_for_hids(
                    db,
                    goal,
                    {target_hyp.hypothesis_id},
                    reason=f"terminate_refuted:{target_hyp.hypothesis_id}",
                    next_plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                )
                summary["terminated_hypothesis_ids"].append(target_hyp.hypothesis_id)

            elif kind == ActionKind.COMPLETE_MILESTONE:
                milestone_key = str(params.get("milestone_id") or target_hid or "milestone")
                if target_hyp is not None:
                    if hasattr(db, "get_authoritative_empirical_receipts"):
                        auth_rcpts = db.get_authoritative_empirical_receipts(goal, target_hyp)
                    else:
                        raw_rcpts = _db_list_receipts(db, goal.goal_id, target_hyp.hypothesis_id)
                        auth_rcpts = [
                            r for r in raw_rcpts if is_valid_empirical_outcome(r, goal, hyp=target_hyp)
                        ]
                    has_confirmed = any(
                        r.scientific_verdict == ScientificVerdict.CONFIRMED
                        for r in auth_rcpts
                    )
                    if not has_confirmed:
                        summary["rejected"].append(
                            {
                                "action": kind.value,
                                "hypothesis_id": target_hyp.hypothesis_id,
                                "reason": "COMPLETE_MILESTONE requires authoritative VALID_EMPIRICAL CONFIRMED receipt in current plan lineage",
                            }
                        )
                        continue
                    prev_s = target_hyp.status
                    target_hyp.status = "COMPLETED"
                    target_hyp.scientific_verdict = ScientificVerdict.CONFIRMED
                    _db_register_or_update_hypothesis(db, target_hyp, prev_status=prev_s)
                if hasattr(db, "upsert_knowledge_item"):
                    db.upsert_knowledge_item(
                        goal.goal_id,
                        "milestone_completed",
                        f"ms:{milestone_key}",
                        {"rationale": act.rationale, "target_hypothesis_id": target_hid, **params},
                        hypothesis_id=str(target_hid) if target_hid else None,
                        goal_version=int(goal.version),
                    )
                summary["completed_milestones"].append(milestone_key)

            elif kind == ActionKind.REQUEST_OWNER_DECISION:
                esc_key = f"esc:{target_hid or uuid.uuid4().hex[:8]}"
                if hasattr(db, "upsert_knowledge_item"):
                    db.upsert_knowledge_item(
                        goal.goal_id,
                        "owner_escalation",
                        esc_key,
                        {
                            "rationale": act.rationale,
                            "target_hypothesis_id": target_hid,
                            "uncertainty_id": act.uncertainty_id,
                            **params,
                        },
                        hypothesis_id=str(target_hid) if target_hid else None,
                        goal_version=int(goal.version),
                    )
                summary["owner_escalations"].append(esc_key)

        return summary

    if tx_cm is not None:
        with tx_cm:
            return _run_in_tx()
    return _run_in_tx()


class ExploreEngine:
    """Stateful helper binding a StateDB instance with hypothesis proposal and reconciliation."""

    def __init__(
        self,
        db: Any = None,
        *,
        min_evoi: float = DEFAULT_MIN_EVOI,
        max_unknown_refinements: int = DEFAULT_MAX_UNKNOWN_REFINEMENTS,
        ai_model: str = DEFAULT_AI_MODEL,
        subprocess_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.db = db
        self.min_evoi = min_evoi
        self.max_unknown_refinements = max_unknown_refinements
        self.ai_model = ai_model
        self.subprocess_runner = subprocess_runner

    def propose(
        self,
        goal: ResearchGoal,
        *,
        mode: str = "auto",
        external_proposals: str | Sequence[Mapping[str, Any]] | Mapping[str, Any] | None = None,
        ready_queue_count: int = 0,
        dispatchable_slots: int | None = None,
        persist: bool = True,
    ) -> ProposalBatchResult:
        """Propose, validate, deduplicate, and EVOI-rank hypotheses for a goal."""
        return propose_hypotheses(
            goal,
            db=self.db,
            mode=mode,
            external_proposals=external_proposals,
            ready_queue_count=ready_queue_count,
            dispatchable_slots=dispatchable_slots,
            min_evoi=self.min_evoi,
            ai_model=self.ai_model,
            subprocess_runner=self.subprocess_runner,
            persist=persist,
        )

    def reconcile(
        self,
        goal: ResearchGoal,
        hypothesis: HypothesisNode,
        receipt: EvidenceReceipt,
    ) -> ReconcileDecision:
        """Reconcile a completed EvidenceReceipt against the hypothesis tree."""
        return reconcile_receipt(
            self.db,
            goal,
            hypothesis,
            receipt,
            max_unknown_refinements=self.max_unknown_refinements,
        )
