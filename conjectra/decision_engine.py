"""Role-typed asynchronous per-Goal DecisionEngine (`ResearchLead` & `ResearchDirector`).

Separates scientific role (`ScientificRole`) from model pool/tier (`ModelRouter`),
executes control-plane LLM calls as single-attempt (`180s` inner / `225s` outer) HQ jobs,
traps `SIGINT`/`SIGTERM` cleanly, forbids live dummy `"pytest -q"` fallbacks, and fences
commits atomically on `(goal_version, human_epoch, attempt, lease_id)`.

Runnable standalone via HQ:
  python3 -m conjectra.decision_engine --req-id <id> --arr-home <home>
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import json
import os
import re
import signal
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from conjectra.backends import DEFAULT_BACKEND_FOR_POOL
from conjectra.arbiter import POOL_NORMALIZATION, ResourceArbiter
from conjectra.explore import (
    _build_ai_propose_prompt,
    _extract_primary_metric_value,
    _hypothesis_generation,
    apply_scientific_actions,
    parse_proposals_payload,
    reconcile_receipt,
)
from conjectra.hq_runtime import HQRuntime
from conjectra.multiagent import ProviderSpec, invoke_provider, parse_provider_spec
from conjectra.planning import infer_model_family
from conjectra.state_db import FencedStaleWriteError, StateDB, resolve_arr_home
from conjectra.types import (
    ActionKind,
    ConstraintStrength,
    EvidenceReceipt,
    FailureDomain,
    HypothesisKind,
    HypothesisNode,
    LeadAction,
    LeadDecisionEnvelope,
    ResearchGoal,
    ResolvedModelRoute,
    ResourceSpec,
    ScientificRole,
    ScientificVerdict,
    StageKind,
)
from conjectra.workload_matrix import ensure_eval_table

DEFAULT_COALESCE_WINDOW_S = 12.0
MAX_DIGEST_BYTES = 4096
DECISION_INNER_TIMEOUT_S = 180
DECISION_OUTER_WALL_TIMEOUT_S = 225

L1_PROVIDER_CANDIDATES: tuple[str, ...] = (
    "codex:gpt-5-codex",
    "gemini:gemini-pro",
    "gemini:gemini-flash",
    "grok:grok-code-fast",
)

L2_PROVIDER_CANDIDATES: tuple[str, ...] = (
    "claude:claude-opus-5-5",
    "codex:gpt-5-codex",
    "gemini:gemini-pro",
    "grok:grok-code-fast",
)

_POOL_TO_HQ_RESOURCE: dict[str, str] = {
    "claude": "llmclaude",
    "codex": "llmcodexaudit",
    "gemini": "llmgemini",
    "deepseek": "llmdeepseek",
    "grok": "llmdeepseek",
}

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


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
    foreground_reserve_quota_pct: float = 15.0,
) -> RouteResolverProtocol | None:
    try:
        mod = importlib.import_module("conjectra.model_router")
        cls = getattr(mod, "ModelRouter", None)
        if cls is not None:
            return cls(
                arbiter=arbiter,
                db=db,
                foreground_reserve_quota_pct=foreground_reserve_quota_pct,
            )
    except Exception:
        return None
    return None


def build_goal_digest(
    db: StateDB,
    goal: ResearchGoal,
    *,
    max_bytes: int = MAX_DIGEST_BYTES,
) -> dict[str, Any]:
    """Build a compact <4KB goal evidence, uncertainty & lineage digest for Lead/Director."""
    hyps = db.list_hypotheses(goal_id=goal.goal_id)
    receipts = db.list_receipts(goal_id=goal.goal_id)
    knowledge = db.list_knowledge_items(goal_id=goal.goal_id, limit=20)
    latest_rplan = db.get_latest_research_plan(goal.goal_id)
    by_id = {h.hypothesis_id: h for h in hyps}
    latest_rcpt: dict[str, EvidenceReceipt] = {}
    for r in receipts:
        latest_rcpt[r.hypothesis_id] = r

    hyp_lines: list[dict[str, Any]] = []
    for h in hyps[-10:]:
        rcpt = latest_rcpt.get(h.hypothesis_id)
        val = (
            _extract_primary_metric_value(rcpt.metrics, goal.primary_metric)
            if rcpt is not None
            else None
        )
        hyp_lines.append(
            {
                "id": h.hypothesis_id,
                "parent": h.parent_id,
                "gen": _hypothesis_generation(h, by_id),
                "kind": h.kind.value,
                "status": h.status,
                "failure_domain": h.failure_domain.value,
                "verdict": h.scientific_verdict.value,
                "metric": val,
                "repairs_used": int(h.budget_state.get("code_repairs_used", 0)),
                "reason": (rcpt.failure_reason[:120] if rcpt and rcpt.failure_reason else ""),
            }
        )

    lessons = [
        {
            "cat": item["category"],
            "key": item["key"],
            "summary": item["content"] if isinstance(item["content"], str) else str(item["content"])[:180],
        }
        for item in knowledge[:8]
    ]

    digest: dict[str, Any] = {
        "goal_id": goal.goal_id,
        "version": int(goal.version),
        "human_epoch": int(goal.human_epoch),
        "primary_metric": goal.primary_metric,
        "direction": goal.direction,
        "baseline_value": float(goal.baseline_value),
        "thesis": latest_rplan.thesis[:200] if latest_rplan else "",
        "uncertainties": (latest_rplan.uncertainties[:4] if latest_rplan else []),
        "hypotheses": hyp_lines,
        "lessons": lessons,
    }

    encoded = json.dumps(digest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    while len(encoded) > max_bytes and (
        digest["hypotheses"] or digest["lessons"] or digest["uncertainties"]
    ):
        if len(digest["lessons"]) > 1:
            digest["lessons"].pop()
        elif digest["uncertainties"]:
            digest["uncertainties"].pop()
        elif len(digest["hypotheses"]) > 1:
            digest["hypotheses"].pop(0)
        else:
            digest["lessons"] = []
            break
        encoded = json.dumps(digest, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return digest


def parse_lead_decision_payload(
    raw_text: str,
    *,
    goal: ResearchGoal,
    default_role: ScientificRole = ScientificRole.RESEARCH_LEAD,
) -> LeadDecisionEnvelope:
    """Parse structured `LeadDecisionEnvelope` JSON (or legacy proposal list) from LLM output."""
    text = (raw_text or "").strip()
    if not text:
        raise ValueError("Empty response from decision provider")

    candidates_text: list[str] = [m.strip() for m in _FENCED_JSON_RE.findall(text) if m.strip()]
    candidates_text.append(text)

    parsed_obj: Any = None
    decoder = json.JSONDecoder()
    for chunk in candidates_text:
        try:
            parsed_obj = json.loads(chunk)
            break
        except json.JSONDecodeError:
            for idx, ch in enumerate(chunk):
                if ch in ("{", "["):
                    try:
                        parsed_obj, _ = decoder.raw_decode(chunk, idx)
                        break
                    except json.JSONDecodeError:
                        continue
            if parsed_obj is not None:
                break

    if isinstance(parsed_obj, Mapping) and "actions" in parsed_obj and isinstance(parsed_obj["actions"], list):
        env_dict = {
            "role": parsed_obj.get("role", default_role.value),
            "goal_id": parsed_obj.get("goal_id", goal.goal_id),
            "goal_version": int(parsed_obj.get("goal_version", goal.version)),
            "human_epoch": int(parsed_obj.get("human_epoch", goal.human_epoch)),
            "uncertainty_summary": str(parsed_obj.get("uncertainty_summary", "")),
            "actions": parsed_obj["actions"],
            "metadata": dict(parsed_obj.get("metadata") or {}),
        }
        return LeadDecisionEnvelope.from_dict(env_dict)

    # Backward-compatible support for JSON proposal array / `{"proposals": [...]}`
    prop_list = parse_proposals_payload(text)
    actions: list[LeadAction] = []
    for cand in prop_list:
        actions.append(
            LeadAction(
                action=ActionKind.PROPOSE_COMPETING_ROUTE,
                target_hypothesis_id=cand.get("parent_id"),
                uncertainty_id=cand.get("uncertainty_id"),
                rationale=str(cand.get("statement") or cand.get("title") or ""),
                parameters={"proposal": dict(cand)},
            )
        )
    return LeadDecisionEnvelope(
        role=default_role,
        goal_id=goal.goal_id,
        goal_version=int(goal.version),
        human_epoch=int(goal.human_epoch),
        uncertainty_summary="Imported structured hypothesis proposals",
        actions=actions,
    )


class DecisionInterruptedError(RuntimeError):
    """Raised when an async HQ DecisionEngine process receives SIGINT/SIGTERM."""


_ACTIVE_SUBMITTING_NONCES_LOCK = threading.Lock()
_ACTIVE_SUBMITTING_NONCES: set[str] = set()


def get_active_submitting_nonces() -> set[str]:
    """Return a thread-safe snapshot of in-flight decision `submit_nonce`s in this process."""
    with _ACTIVE_SUBMITTING_NONCES_LOCK:
        return set(_ACTIVE_SUBMITTING_NONCES)


class DecisionEngine:
    """Role-typed async HQ decision engine (`ResearchLead` & `ResearchDirector`)."""

    def __init__(
        self,
        db: StateDB,
        *,
        hq: HQRuntime | None = None,
        arbiter: ResourceArbiter | None = None,
        router: RouteResolverProtocol | None = None,
        workflow_runtime: Any = None,
        foreground_reserve_quota_pct: float | None = None,
    ) -> None:
        self.db = db
        self.hq = hq
        self.arbiter = arbiter or ResourceArbiter()
        if foreground_reserve_quota_pct is not None:
            self.foreground_reserve_quota_pct = float(foreground_reserve_quota_pct)
        else:
            self.foreground_reserve_quota_pct = float(
                self.db.get_global_config("foreground_reserve_quota_pct", 15.0)
            )
        self.router = router or _try_build_default_router(
            arbiter=self.arbiter,
            db=self.db,
            foreground_reserve_quota_pct=self.foreground_reserve_quota_pct,
        )
        self._active_submitting_nonces: set[str] = set()
        if workflow_runtime is not None:
            self.workflow_runtime = workflow_runtime
        else:
            from conjectra.workflow_runtime import WorkflowRuntime

            self.workflow_runtime = WorkflowRuntime(
                self.db,
                arbiter=self.arbiter,
                router=self.router,
            )

    def _ledger_pool_scores(self) -> dict[str, float]:
        """Compute empirical pass/yield score per pool prefix from workload_eval_ledger (is_real_llm=1)."""
        try:
            if int(getattr(self.db, "_tx_depth", 0) or 0) == 0:
                ensure_eval_table(self.db)
            with self.db._lock:
                tbl = self.db._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'workload_eval_ledger'"
                ).fetchone()
                if tbl is None:
                    return {}
                rows = self.db._conn.execute(
                    """
                    SELECT provider_pairing,
                           AVG(holdout_passed) AS pass_rate,
                           AVG(effective_yield_per_100k) AS avg_yield
                    FROM workload_eval_ledger
                    WHERE is_real_llm = 1
                    GROUP BY provider_pairing
                    """
                ).fetchall()
            scores: dict[str, float] = {}
            for r in rows:
                pairing = str(r["provider_pairing"] or "")
                score = float(r["pass_rate"] or 0.0) * 100.0 + float(r["avg_yield"] or 0.0) * 0.1
                for pool_key in ("deepseek", "grok", "claude", "codex", "gemini"):
                    if pool_key in pairing:
                        scores[pool_key] = max(scores.get(pool_key, 0.0), score)
            return scores
        except Exception:
            return {}

    def select_provider_for_tier(
        self,
        tier: str,
        *,
        exclude_pools: Sequence[str] | set[str] = (),
    ) -> ProviderSpec:
        """Select a provider respecting live quota, ledger yield, and foreground_reserve."""
        t_norm = tier.strip().upper()
        candidates_raw = (
            L2_PROVIDER_CANDIDATES if t_norm == "L2" else (
                "gemini:gemini-pro",
                "gemini:gemini-flash",
                "codex:gpt-5-codex",
                "grok:grok-code-fast",
            )
        )
        headroom = self.arbiter.read_pool_headroom()
        ledger_scores = self._ledger_pool_scores()
        excluded = {str(x).strip().lower() for x in exclude_pools}

        reserve_pct = max(0.0, min(90.0, float(self.foreground_reserve_quota_pct)))
        max_bg_used_pct = 100.0 - reserve_pct

        healthy: list[tuple[float, int, ProviderSpec]] = []
        fallback: list[tuple[int, ProviderSpec]] = []

        for idx, raw in enumerate(candidates_raw):
            spec = parse_provider_spec(
                raw,
                role="blind_auditor" if t_norm == "L2" else "builder",
                arbiter=self.arbiter,
            )
            if spec.pool.lower() in excluded or spec.label().lower() in excluded:
                continue
            logical_pool = POOL_NORMALIZATION.get(spec.pool.lower(), spec.pool.lower())
            info = headroom.get(logical_pool) or {}
            avail = bool(info.get("available", True))
            used_pct = info.get("used_pct")
            respects_reserve = (
                True if used_pct is None else float(used_pct) < max_bg_used_pct
            )
            if avail and (t_norm != "L2" or respects_reserve):
                score = ledger_scores.get(spec.pool.lower(), 0.0)
                if used_pct is not None:
                    score += max(0.0, 100.0 - float(used_pct)) * 0.05
                healthy.append((score, idx, spec))
            else:
                fallback.append((idx, spec))

        if healthy:
            healthy.sort(key=lambda item: (-item[0], item[1]))
            return healthy[0][2]
        if fallback:
            return fallback[0][1]
        return parse_provider_spec(candidates_raw[0], role="builder", arbiter=self.arbiter)

    def resolve_route_for_role(
        self,
        role: ScientificRole | str,
        *,
        goal: ResearchGoal | None = None,
        hyp: HypothesisNode | None = None,
        workflow_id: str | None = None,
        explicit_spec: str | None = None,
        exclude_specs: Sequence[str] = (),
    ) -> ResolvedModelRoute:
        """Resolve an immutable `ResolvedModelRoute` via `ModelRouter` (or capability-safe fallback)."""
        role_enum = (
            role if isinstance(role, ScientificRole) else ScientificRole(str(role))
        )
        if self.router is not None:
            import inspect as _inspect_dr

            dr_sig = _inspect_dr.signature(self.router.resolve_route).parameters
            dr_varkw = any(
                p.kind == _inspect_dr.Parameter.VAR_KEYWORD for p in dr_sig.values()
            )
            dr_extra: dict[str, Any] = {}
            if ("db" in dr_sig or dr_varkw) and self.db is not None:
                dr_extra["db"] = self.db
            if ("workflow_id" in dr_sig or dr_varkw) and workflow_id is not None:
                dr_extra["workflow_id"] = workflow_id
            return self.router.resolve_route(
                role_enum,
                goal=goal,
                hyp=hyp,
                explicit_spec=explicit_spec,
                exclude_specs=exclude_specs,
                **dr_extra,
            )

        tier_equiv = (
            "L2"
            if role_enum in (ScientificRole.RESEARCH_DIRECTOR, ScientificRole.THEORIST)
            else "L1"
        )
        prov = (
            parse_provider_spec(explicit_spec, arbiter=self.arbiter)
            if explicit_spec
            else self.select_provider_for_tier(tier_equiv, exclude_pools=exclude_specs)
        )
        default_effort = "high"
        if "sonnet" in prov.model.lower():
            default_effort = "medium"
        elif role_enum == ScientificRole.RESEARCH_DIRECTOR and "opus" in prov.model.lower():
            default_effort = "xhigh"
        pool_l = prov.pool.lower()
        logical_pool = POOL_NORMALIZATION.get(pool_l, pool_l)
        return ResolvedModelRoute(
            backend_mode=DEFAULT_BACKEND_FOR_POOL.get(pool_l, "dispatch_script"),
            execution_host="local",
            multipool_host_arg="local",
            pool=prov.pool,
            canonical_model=prov.model,
            requested_effort=default_effort,
            wire_effort=default_effort,
            family=prov.family or infer_model_family(f"{prov.pool}:{prov.model}"),
            logical_pool=logical_pool,
            hq_resource=_POOL_TO_HQ_RESOURCE.get(pool_l),
            strength=ConstraintStrength.SOFT_FALLBACK,
            matched_rule_id="fallback_role_route",
        )

    def evaluate_l0(
        self,
        goal: ResearchGoal,
        hypothesis: HypothesisNode,
        receipt: EvidenceReceipt,
    ) -> dict[str, Any]:
        """Execute 0-token deterministic L0 lifecycle transition."""
        dec = reconcile_receipt(self.db, goal, hypothesis, receipt)
        return {
            "tier": "L0",
            "tokens_used": 0,
            "decision": dec.to_dict(),
        }

    def enqueue_decision(
        self,
        goal_id: str,
        *,
        tier: str = "L1",
        role: ScientificRole | str | None = None,
        trigger_kind: str = "EVIDENCE_UPDATE",
        urgency: str = "COALESCED",
        coalesce_window_s: float = DEFAULT_COALESCE_WINDOW_S,
        hypothesis_id: str | None = None,
        head_sha: str = "",
        failure_domain: FailureDomain | str | None = None,
        payload: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Enqueue a role-typed per-Goal or per-Hypothesis decision request with budget & coalesce guards."""
        goal = self.db.get_goal(goal_id)
        if goal is None:
            raise KeyError(f"Unknown goal_id: {goal_id}")
        urg = urgency.strip().upper()
        if urg not in ("IMMEDIATE", "COALESCED"):
            raise ValueError(f"Invalid urgency {urgency!r}; expected IMMEDIATE or COALESCED")

        t_norm = tier.strip().upper()
        if role is None and t_norm not in ("L0", "L1", "L2"):
            raise ValueError(f"Invalid decision tier {tier!r}; expected L0|L1|L2")

        routine_trigger = str(trigger_kind or "").strip().upper() in (
            "EVIDENCE_UPDATE",
            "INITIAL_PLANNING",
            "BRANCH_CONVERGENCE",
        )
        req_payload_map = dict(payload or {})
        is_director_exception = bool(
            req_payload_map.get("milestone_signoff")
            or req_payload_map.get("director_adjudication")
            or req_payload_map.get("tiebreaker")
        )

        if routine_trigger and not is_director_exception:
            eff_role = ScientificRole.RESEARCH_LEAD
            eff_tier = "L0" if t_norm == "L0" else "L1"
        elif role is not None:
            eff_role = (
                role if isinstance(role, ScientificRole) else ScientificRole(str(role).strip())
            )
            eff_tier = (
                "L2"
                if eff_role == ScientificRole.RESEARCH_DIRECTOR
                else ("L0" if t_norm == "L0" else "L1")
            )
        else:
            eff_tier = t_norm
            eff_role = (
                ScientificRole.RESEARCH_DIRECTOR
                if eff_tier == "L2"
                else ScientificRole.RESEARCH_LEAD
            )

        hyp: HypothesisNode | None = (
            self.db.get_hypothesis(hypothesis_id) if hypothesis_id else None
        )
        eff_window = 0.0 if urg == "IMMEDIATE" else max(0.0, float(coalesce_window_s))
        fdom_str = (
            failure_domain.value
            if isinstance(failure_domain, FailureDomain)
            else str(failure_domain or (hyp.failure_domain.value if hyp else "NONE"))
        )
        if hypothesis_id:
            eff_sha = head_sha or (
                hyp.head_sha if hyp and hyp.head_sha else (hyp.base_sha if hyp else "")
            )
            coalesce_key = f"{goal_id}:{hypothesis_id}:{eff_sha}:{fdom_str}"
        elif urg == "IMMEDIATE":
            coalesce_key = ""
        else:
            coalesce_key = f"{goal_id}:{eff_role.value}:{trigger_kind}"

        digest = build_goal_digest(self.db, goal)
        req_payload: dict[str, Any] = {
            "role": eff_role.value,
            "urgency": urg,
            "digest": digest,
            "attempt": 1,
            **dict(payload or {}),
        }
        if hypothesis_id:
            req_payload["hypothesis_id"] = hypothesis_id

        with self.db._tx() as (conn, _pending):
            existing_pending = None
            if coalesce_key:
                existing_pending = conn.execute(
                    """
                    SELECT req_id FROM decision_requests
                    WHERE goal_id = ? AND coalesce_key = ? AND status = 'PENDING'
                    ORDER BY created_ts DESC LIMIT 1
                    """,
                    (goal_id, coalesce_key),
                ).fetchone()

            # Only charge `triage_decisions_used` when creating a new DecisionRequest
            # (never double-charge when coalescing into an already-PENDING request!).
            if (
                existing_pending is None
                and hyp is not None
                and trigger_kind not in ("INITIAL_PLANNING", "HUMAN_GUIDANCE")
            ):
                fresh_hyp = self.db.get_hypothesis(hyp.hypothesis_id) or hyp
                bstate = dict(fresh_hyp.budget_state or {})
                triage_used = int(bstate.get("triage_decisions_used", 0))
                max_triage = int(bstate.get("max_triage_decisions", 2))
                if triage_used >= max_triage and eff_role != ScientificRole.RESEARCH_DIRECTOR:
                    return {
                        "req_id": "",
                        "goal_id": goal_id,
                        "role": eff_role.value,
                        "tier": eff_tier,
                        "status": "SUPPRESSED_BY_TRIAGE_BUDGET",
                        "triage_decisions_used": triage_used,
                        "max_triage_decisions": max_triage,
                    }
                bstate["triage_decisions_used"] = triage_used + 1
                self.db.update_hypothesis_v3_state(
                    fresh_hyp.hypothesis_id, budget_state_patch=bstate, now=now
                )

        req_payload["plan_epoch"] = int(getattr(goal, "plan_epoch", 0) or 0)
        return self.db.enqueue_decision_request(
            goal_id,
            tier=eff_tier,
            role=eff_role,
            attempt=1,
            trigger_kind=trigger_kind,
            goal_version=int(goal.version),
            human_epoch=int(goal.human_epoch),
            coalesce_key=coalesce_key,
            coalesce_window_s=eff_window,
            payload=req_payload,
            now=now,
        )

    def submit_decision_to_hq(
        self,
        req_id: str,
        *,
        arr_home: str | Path | None = None,
        pre_hq_submit_hook: Callable[[dict[str, Any]], None] | None = None,
        post_submit_hook: Callable[[int], None] | None = None,
    ) -> int:
        """Submit a single-attempt async HQ job (`225s` outer wall) under exclusive lease + `PENDING -> SUBMITTING -> RUNNING` CAS."""
        from conjectra.arbiter import read_proc_start_time
        from conjectra.goal_manager import GlobalGoalManager

        req = self.db.get_decision_request(req_id)
        if req is None:
            raise KeyError(f"Unknown decision request: {req_id}")
        goal = self.db.get_goal(str(req["goal_id"]))
        if goal is None:
            raise KeyError(f"Unknown goal for decision request {req_id}: {req['goal_id']}")
        req_payload = dict(req.get("payload") or {})
        hyp_id = req_payload.get("hypothesis_id") or req_payload.get("target_hypothesis_id")
        hyp = (
            self.db.get_goal_hypothesis(goal.goal_id, str(hyp_id))
            if (hyp_id and hasattr(self.db, "get_goal_hypothesis"))
            else (self.db.get_hypothesis(str(hyp_id)) if hyp_id else None)
        )

        hq = self.hq or HQRuntime(resolve_arr_home(arr_home or self.db.arr_home))
        hq.ensure_running()
        home = resolve_arr_home(arr_home or self.db.arr_home)

        role_str = str(req.get("role") or ScientificRole.RESEARCH_LEAD.value)
        wf_id_submit = str(req_payload.get("workflow_id") or "").strip() or None
        route = self.resolve_route_for_role(
            role_str, goal=goal, hyp=hyp, workflow_id=wf_id_submit
        )

        submit_nonce = uuid.uuid4().hex[:12]
        lease = self.db.acquire_lease(
            f"decision:{req['goal_id']}:{req_id}",
            f"dec-{req_id}",
            holder=f"hq-dec-{req_id}-{submit_nonce}",
            ttl_s=float(DECISION_OUTER_WALL_TIMEOUT_S + 30),
            goal_id=str(req["goal_id"]),
            allow_same_holder_renew=False,
        )
        if lease is None or not str(lease.get("lease_id") or "").strip():
            raise FencedStaleWriteError(
                f"DecisionRequest {req_id} already holds an active exclusive lease",
                reason_code="LEASE_CONFLICT",
            )
        lease_id = str(lease["lease_id"])

        req_att = max(1, int(req.get("attempt", 1)))
        req_goal_ver = int(req["goal_version"])
        req_human_ep = int(req["human_epoch"])
        req_plan_ep = req_payload.get("plan_epoch")
        dec_ukey = f"decision:{req_id}:a{req_att}:{submit_nonce}"
        submit_job_name = f"dec-{req_id}-a{req_att}-{submit_nonce[:8]}"
        reserved_here = False
        self._active_submitting_nonces.add(submit_nonce)
        with _ACTIVE_SUBMITTING_NONCES_LOCK:
            _ACTIVE_SUBMITTING_NONCES.add(submit_nonce)
        try:
            # Pre-submit atomic CAS (`PENDING -> SUBMITTING`) + epoch/ACTIVE check + budget reservation
            try:
                with self.db._tx() as (conn, _pending):
                    now_pre = time.time()
                    d_row = conn.execute(
                        "SELECT status, attempt, goal_version, human_epoch, payload_json FROM decision_requests WHERE req_id = ?",
                        (req_id,),
                    ).fetchone()
                    g_row = conn.execute(
                        "SELECT status, payload_json FROM goals WHERE goal_id = ?",
                        (goal.goal_id,),
                    ).fetchone()
                    if d_row is None or g_row is None:
                        raise FencedStaleWriteError(
                            f"DecisionRequest {req_id} or goal {goal.goal_id} missing before submit",
                            reason_code="UNKNOWN_GOAL",
                        )
                    if str(d_row["status"]) != "PENDING":
                        raise FencedStaleWriteError(
                            f"DecisionRequest {req_id} status is {d_row['status']!r}, expected 'PENDING'",
                            reason_code="STATUS_MISMATCH",
                        )
                    cur_pl_pre = json.loads(d_row["payload_json"])
                    if int(cur_pl_pre.get("attempt", d_row["attempt"] or 1)) != req_att:
                        raise FencedStaleWriteError(
                            f"DecisionRequest {req_id} attempt changed before submit",
                            reason_code="STALE_ATTEMPT",
                        )
                    fresh_goal = ResearchGoal.from_dict(json.loads(g_row["payload_json"]))
                    if str(g_row["status"]).upper() != "ACTIVE" or fresh_goal.autonomy_mode == "paused":
                        raise FencedStaleWriteError(
                            f"Goal {goal.goal_id} is not ACTIVE before decision submit",
                            reason_code="GOAL_NOT_ACTIVE",
                        )
                    if (
                        int(fresh_goal.version) != req_goal_ver
                        or int(fresh_goal.human_epoch) != req_human_ep
                        or (
                            req_plan_ep is not None
                            and int(getattr(fresh_goal, "plan_epoch", 0) or 0) != int(req_plan_ep)
                        )
                    ):
                        raise FencedStaleWriteError(
                            f"DecisionRequest {req_id} stale before submit",
                            reason_code="HUMAN_INVALIDATED",
                        )
                    if not self.db.lease_matches_in_tx(
                        conn, f"dec-{req_id}", lease_id, now=now_pre, allow_none=False
                    ):
                        raise FencedStaleWriteError(
                            f"DecisionRequest {req_id} lease {lease_id!r} lost before submit",
                            reason_code="LEASE_MISMATCH",
                        )

                    if hasattr(self.db, "reserve_goal_budget"):
                        res_tok = (
                            min(4000, max(1, int(fresh_goal.max_tokens_budget) // 2))
                            if int(fresh_goal.max_tokens_budget) > 0
                            else 4000
                        )
                        ok_b, committed_tok, max_tok = self.db.reserve_goal_budget(
                            fresh_goal.goal_id,
                            dec_ukey,
                            entity_kind="decision_request",
                            entity_id=req_id,
                            attempt=req_att,
                            model=route.canonical_spec(),
                            reserve_tokens=res_tok,
                            exclude_usage_key=dec_ukey,
                        )
                        if not ok_b:
                            from conjectra.model_router import PolicyBlockedError

                            raise PolicyBlockedError(
                                f"BUDGET_EXHAUSTED: goal {fresh_goal.goal_id} committed={committed_tok} / budget={max_tok}",
                                waitable=False,
                                rule_id="goal_budget",
                                code="BUDGET_EXHAUSTED",
                            )
                        reserved_here = True

                    merged_payload = {
                        **cur_pl_pre,
                        "attempt": req_att,
                        "pinned_route": route.to_dict(),
                        "lease_id": lease_id,
                        "submit_nonce": submit_nonce,
                        "usage_key": dec_ukey,
                        "submit_job_name": submit_job_name,
                        "submit_owner_pid": os.getpid(),
                        "submit_proc_start_time": read_proc_start_time(os.getpid()),
                        "submitting_started_ts": now_pre,
                    }
                    cur_cas = conn.execute(
                        """
                        UPDATE decision_requests
                        SET status = 'SUBMITTING', model_spec = ?, payload_json = ?, updated_ts = ?
                        WHERE req_id = ? AND status = 'PENDING'
                        """,
                        (
                            route.canonical_spec(),
                            json.dumps(merged_payload, sort_keys=True, ensure_ascii=False),
                            now_pre,
                            req_id,
                        ),
                    )
                    if int(cur_cas.rowcount or 0) != 1:
                        raise FencedStaleWriteError(
                            f"CAS failed claiming PENDING DecisionRequest {req_id}",
                            reason_code="STATUS_MISMATCH",
                        )
            except Exception:
                if reserved_here and hasattr(self.db, "release_goal_budget_reservation"):
                    self.db.release_goal_budget_reservation(goal.goal_id, dec_ukey, reason="pre_submit_aborted")
                self.db.release_lease(lease_id)
                raise

            if callable(pre_hq_submit_hook):
                pre_hq_submit_hook(dict(merged_payload))

            # Recheck submit lease & row authority immediately before external hq.submit RPC
            try:
                with self.db._tx() as (conn, _pending):
                    now_imm = time.time()
                    d_imm = conn.execute(
                        "SELECT status, payload_json FROM decision_requests WHERE req_id = ?",
                        (req_id,),
                    ).fetchone()
                    g_imm = conn.execute(
                        "SELECT status, payload_json FROM goals WHERE goal_id = ?",
                        (goal.goal_id,),
                    ).fetchone()
                    if d_imm is None or g_imm is None or str(d_imm["status"]) != "SUBMITTING":
                        raise FencedStaleWriteError(
                            f"DecisionRequest {req_id} changed state before HQ submit",
                            reason_code="STATUS_MISMATCH",
                        )
                    pl_imm = json.loads(d_imm["payload_json"])
                    if str(pl_imm.get("submit_nonce") or "") != submit_nonce:
                        raise FencedStaleWriteError(
                            f"DecisionRequest {req_id} nonce superseded before HQ submit",
                            reason_code="NONCE_MISMATCH",
                        )
                    fg_imm = ResearchGoal.from_dict(json.loads(g_imm["payload_json"]))
                    if (
                        str(g_imm["status"]).upper() != "ACTIVE"
                        or fg_imm.autonomy_mode == "paused"
                        or int(fg_imm.version) != req_goal_ver
                        or int(fg_imm.human_epoch) != req_human_ep
                        or (
                            req_plan_ep is not None
                            and int(getattr(fg_imm, "plan_epoch", 0) or 0) != int(req_plan_ep)
                        )
                    ):
                        raise FencedStaleWriteError(
                            f"Goal {goal.goal_id} epoch/status invalidated before HQ submit",
                            reason_code="HUMAN_INVALIDATED",
                        )
                    if not self.db.lease_matches_in_tx(
                        conn, f"dec-{req_id}", lease_id, now=now_imm, allow_none=False
                    ):
                        raise FencedStaleWriteError(
                            f"DecisionRequest {req_id} lease {lease_id!r} lost before HQ submit",
                            reason_code="LEASE_MISMATCH",
                        )
            except Exception:
                with contextlib.suppress(Exception):
                    with self.db._tx() as (conn, _p):
                        d_ab = conn.execute(
                            "SELECT status, payload_json FROM decision_requests WHERE req_id = ?",
                            (req_id,),
                        ).fetchone()
                        if d_ab is not None and str(d_ab["status"]) == "SUBMITTING":
                            pl_ab = json.loads(d_ab["payload_json"])
                            if str(pl_ab.get("submit_nonce") or "") == submit_nonce:
                                conn.execute(
                                    "UPDATE decision_requests SET status = 'PENDING', updated_ts = ? WHERE req_id = ? AND status = 'SUBMITTING'",
                                    (time.time(), req_id),
                                )
                if reserved_here and hasattr(self.db, "release_goal_budget_reservation"):
                    self.db.release_goal_budget_reservation(goal.goal_id, dec_ukey, reason="pre_hq_rpc_aborted")
                self.db.release_lease(lease_id)
                raise

            cmd = [
                sys.executable,
                "-m",
                "conjectra.decision_engine",
                "--req-id",
                req_id,
                "--arr-home",
                str(home),
                "--lease-id",
                lease_id,
            ]

            try:
                jid = hq.submit_task(
                    cmd,
                    task_id=submit_job_name,
                    cwd=home,
                    stage=StageKind.PROPOSE,
                    resources=ResourceSpec(
                        cpus=1,
                        ram_mb=1024,
                        llm_pool=route.pool,
                        wall_timeout_s=DECISION_OUTER_WALL_TIMEOUT_S,
                        hq_llm_resource=route.hq_resource,
                        pinned_route=route.to_dict(),
                    ),
                    wrap_exit=False,
                )
            except Exception:
                # Only release reservation/requeue immediately if HQ authoritatively proves `submit_job_name`
                # was NOT accepted (or stub HQ lacks job_list/query_job_list). Otherwise leave durable
                # `SUBMITTING` + `RESERVED` intact for `_reconcile_submitting_decisions`.
                confirmed_absent = False
                if not hasattr(hq, "query_job_list") and not hasattr(hq, "job_list"):
                    confirmed_absent = True
                else:
                    q_list = GlobalGoalManager._query_hq_job_list(hq, all_jobs=True)
                    if q_list.get("lookup_status") == "OK":
                        jobs_l = q_list.get("jobs") or []
                        confirmed_absent = not any(
                            str(j.get("name") or "") in {submit_job_name, f"dec-{req_id}"}
                            for j in jobs_l
                            if isinstance(j, Mapping)
                        )
                if confirmed_absent:
                    with contextlib.suppress(Exception):
                        with self.db._tx() as (conn, _p):
                            d_rev = conn.execute(
                                "SELECT status, payload_json FROM decision_requests WHERE req_id = ?",
                                (req_id,),
                            ).fetchone()
                            if d_rev is not None and str(d_rev["status"]) == "SUBMITTING":
                                pl_rev = json.loads(d_rev["payload_json"])
                                if pl_rev.get("submit_nonce") == submit_nonce:
                                    conn.execute(
                                        "UPDATE decision_requests SET status = 'PENDING', updated_ts = ? WHERE req_id = ? AND status = 'SUBMITTING'",
                                        (time.time(), req_id),
                                    )
                    if reserved_here and hasattr(self.db, "release_goal_budget_reservation"):
                        self.db.release_goal_budget_reservation(goal.goal_id, dec_ukey, reason="hq_submit_failed_confirmed_absent")
                    self.db.release_lease(lease_id)
                raise

            if callable(post_submit_hook):
                post_submit_hook(int(jid))

            must_cancel_orphan_hq = False
            cancel_reason = "cancelled_during_submit"
            try:
                with self.db._tx() as (conn, _pending):
                    now_post = time.time()
                    d_post = conn.execute(
                        "SELECT status, payload_json FROM decision_requests WHERE req_id = ?",
                        (req_id,),
                    ).fetchone()
                    g_post = conn.execute(
                        "SELECT status, payload_json FROM goals WHERE goal_id = ?",
                        (goal.goal_id,),
                    ).fetchone()
                    cur_st_post = str(d_post["status"]) if d_post is not None else "MISSING"
                    cur_pl_post = json.loads(d_post["payload_json"]) if d_post is not None else {}
                    g_active_post = False
                    g_epoch_ok = False
                    if g_post is not None:
                        fg_post = ResearchGoal.from_dict(json.loads(g_post["payload_json"]))
                        g_active_post = str(g_post["status"]).upper() == "ACTIVE" and fg_post.autonomy_mode != "paused"
                        g_epoch_ok = (
                            int(fg_post.version) == req_goal_ver
                            and int(fg_post.human_epoch) == req_human_ep
                            and (
                                req_plan_ep is None
                                or int(getattr(fg_post, "plan_epoch", 0) or 0) == int(req_plan_ep)
                            )
                        )

                    if cur_st_post == "RESOLVED":
                        # Fast worker already completed before submit returned — preserve RESOLVED!
                        if str(cur_pl_post.get("submit_nonce") or "") == submit_nonce:
                            conn.execute(
                                "UPDATE decision_requests SET hq_job_id = COALESCE(hq_job_id, ?), updated_ts = ? WHERE req_id = ? AND status = 'RESOLVED'",
                                (int(jid), now_post, req_id),
                            )
                    elif (
                        cur_st_post == "SUBMITTING"
                        and cur_pl_post.get("submit_nonce") == submit_nonce
                        and g_active_post
                        and g_epoch_ok
                    ):
                        conn.execute(
                            """
                            UPDATE decision_requests
                            SET status = 'RUNNING', hq_job_id = ?, updated_ts = ?
                            WHERE req_id = ? AND status = 'SUBMITTING'
                            """,
                            (int(jid), now_post, req_id),
                        )
                    else:
                        must_cancel_orphan_hq = True
                        same_nonce = bool(
                            d_post is not None
                            and str(cur_pl_post.get("submit_nonce") or "") == submit_nonce
                        )
                        cancel_reason = (
                            f"stale_or_cancelled_during_submit:{cur_st_post}"
                            if same_nonce
                            else f"nonce_superseded_during_submit:{submit_nonce}"
                        )
                        # NEVER mutate decision_requests row if a newer submit_nonce owns it!
                        if same_nonce:
                            conn.execute(
                                "UPDATE decision_requests SET hq_job_id = COALESCE(hq_job_id, ?), updated_ts = ? WHERE req_id = ?",
                                (int(jid), now_post, req_id),
                            )
                            if cur_st_post == "SUBMITTING":
                                conn.execute(
                                    "UPDATE decision_requests SET status = 'CANCELLED', resolved_ts = ?, updated_ts = ? WHERE req_id = ? AND status = 'SUBMITTING'",
                                    (now_post, now_post, req_id),
                                )
                        if hasattr(self.db, "record_pending_hq_cancellation"):
                            self.db.record_pending_hq_cancellation(
                                hq_job_id=int(jid),
                                entity_kind="decision",
                                entity_id=req_id,
                                attempt=req_att,
                                goal_id=goal.goal_id,
                                epoch=req_human_ep,
                                plan_epoch=int(req_plan_ep or 0),
                                reason=cancel_reason,
                                state="CANCEL_PENDING",
                                usage_key=dec_ukey,
                                now=now_post,
                            )
                        # Keep `dec_ukey` in `'RESERVED'` until `confirm_hq_cancellation` positively confirms HQ termination!
                        if same_nonce:
                            self.db.release_lease(lease_id, now=now_post)
            except Exception as attach_exc:
                with contextlib.suppress(Exception):
                    hq.cancel_jobs([int(jid)], reason=f"sql_attach_failed:{attach_exc}")
                with contextlib.suppress(Exception):
                    if hasattr(self.db, "record_pending_hq_cancellation"):
                        self.db.record_pending_hq_cancellation(
                            hq_job_id=int(jid),
                            entity_kind="decision",
                            entity_id=req_id,
                            attempt=req_att,
                            goal_id=goal.goal_id,
                            epoch=req_human_ep,
                            plan_epoch=int(req_plan_ep or 0),
                            reason="sql_attach_failed",
                            state="CANCEL_PENDING",
                            usage_key=dec_ukey,
                        )
                # Keep `dec_ukey` in `'RESERVED'` until authoritative HQ termination confirmation!
                with contextlib.suppress(Exception):
                    with self.db._tx() as (conn, _p):
                        d_err = conn.execute(
                            "SELECT payload_json FROM decision_requests WHERE req_id = ?", (req_id,)
                        ).fetchone()
                        if d_err is not None and str(json.loads(d_err["payload_json"]).get("submit_nonce") or "") == submit_nonce:
                            self.db.release_lease(lease_id)
                raise

            if must_cancel_orphan_hq:
                with contextlib.suppress(Exception):
                    hq.cancel_jobs([int(jid)], reason=cancel_reason)
                if (
                    hasattr(self.db, "confirm_hq_cancellation")
                    and (hasattr(hq, "query_job_state") or hasattr(hq, "job_info"))
                    and GlobalGoalManager._hq_positively_terminated(hq, int(jid))
                ):
                    with contextlib.suppress(Exception):
                        self.db.confirm_hq_cancellation(
                            int(jid),
                            entity_kind="decision",
                            entity_id=req_id,
                            attempt=req_att,
                            usage_key=dec_ukey,
                        )
            return int(jid)
        finally:
            self._active_submitting_nonces.discard(submit_nonce)
            with _ACTIVE_SUBMITTING_NONCES_LOCK:
                _ACTIVE_SUBMITTING_NONCES.discard(submit_nonce)

    def _build_role_prompt(
        self,
        req: Mapping[str, Any],
        goal: ResearchGoal,
        hyps: Sequence[HypothesisNode],
        receipts: Sequence[EvidenceReceipt],
    ) -> str:
        role_val = str(req.get("role") or ScientificRole.RESEARCH_LEAD.value)
        gy_fps = {
            str(g["fingerprint"])
            for g in self.db.list_graveyard(goal_id=goal.goal_id)
            if g.get("fingerprint")
        }
        gy_entries = self.db.list_graveyard(goal_id=goal.goal_id)
        knowledge = self.db.list_knowledge_items(goal_id=goal.goal_id, limit=25)
        base_prompt = _build_ai_propose_prompt(
            goal,
            hyps,
            gy_fps,
            max_proposals=3,
            receipts=receipts,
            graveyard_entries=gy_entries,
            knowledge_items=knowledge,
        )
        digest = (req.get("payload") or {}).get("digest") or build_goal_digest(self.db, goal)
        allowed_actions = [a.value for a in ActionKind]
        grounding_note = (
            "IMPORTANT: Ground every action in measured F1/F2 metrics, failure_domain, and existing repository paths.\n"
            "Do NOT spawn new hypotheses to work around ENGINEERING_DEFECT or ENV_CONFIG_MISMATCH — emit "
            "`REPAIR_IMPLEMENTATION` or `REVISE_HYPOTHESIS` instead!\n"
            f"Preferred JSON schema: {{\"role\": {role_val!r}, \"uncertainty_summary\": \"...\", "
            f"\"actions\": [{{\"action\": one of {allowed_actions}, \"target_hypothesis_id\": \"...\", "
            "\"uncertainty_id\": \"...\", \"rationale\": \"...\", \"parameters\": {...}}}]}.\n"
        )
        if role_val == ScientificRole.RESEARCH_DIRECTOR.value:
            return (
                "ROLE: Research Director (Strategic Adjudicator & Tie-Breaker)\n"
                f"TRIGGER: {req.get('trigger_kind')} | Compact Digest (<4KB): {json.dumps(digest)}\n"
                "Task: Adjudicate contradictory evidence or split red-team reviews, prune dead lines, "
                "or authorize a high-leverage structural pivot.\n"
                + grounding_note
                + "\n"
                + base_prompt
            )
        return (
            f"ROLE: {role_val} (Closed-Loop Uncertainty-Reducing Research Lead)\n"
            f"TRIGGER: {req.get('trigger_kind')} | Compact Digest (<4KB): {json.dumps(digest)}\n"
            "Task: Reduce explicit scientific uncertainty via discriminating experiments, bounded code repair, "
            "workflow switching, result synthesis, or grounded competing routes.\n"
            + grounding_note
            + "\n"
            + base_prompt
        )

    def _offline_dry_run_envelope(
        self,
        goal: ResearchGoal,
        hyps: Sequence[HypothesisNode],
        role: ScientificRole,
    ) -> LeadDecisionEnvelope:
        """Offline test-only envelope (`--dry-run-llm` ONLY; never called in live execution)."""
        by_id = {h.hypothesis_id: h for h in hyps}
        max_gen = max((_hypothesis_generation(h, by_id) for h in hyps), default=1)
        next_gen = max(3 if role == ScientificRole.RESEARCH_DIRECTOR else 2, max_gen + 1)
        parent = next(
            (h for h in hyps if h.scientific_verdict == ScientificVerdict.CONFIRMED),
            hyps[0] if hyps else None,
        )
        wp = list(parent.writable_paths) if parent else (list(goal.writable_prefixes) or ["src/"])
        prop = {
            "kind": (
                HypothesisKind.FALSIFICATION.value
                if role == ScientificRole.RESEARCH_DIRECTOR
                else HypothesisKind.REFINEMENT.value
            ),
            "parent_id": parent.hypothesis_id if parent else None,
            "title": f"{role.value} dry-run route (gen {next_gen})",
            "statement": f"Offline dry-run probe on {goal.primary_metric}.",
            "mechanism": f"{role.value}_dryrun_gen{next_gen}",
            "writable_paths": wp,
            "params": {
                "generation": next_gen,
                "deterministic_no_llm": True,
                "estimated_tokens": 2500,
            },
            "kill_criteria": list(goal.kill_criteria) or [f"{goal.primary_metric} < 0"],
            "screen_cmd": parent.screen_cmd if parent else "python3 -c 'print(1)'",
            "confirm_cmd": parent.confirm_cmd if parent else "python3 -c 'print(1)'",
            "acceptance_cmd": parent.acceptance_cmd if parent else "python3 -c 'print(1)'",
        }
        return LeadDecisionEnvelope(
            role=role,
            goal_id=goal.goal_id,
            goal_version=int(goal.version),
            human_epoch=int(goal.human_epoch),
            uncertainty_summary="dry_run_llm synthetic decision",
            actions=[
                LeadAction(
                    action=ActionKind.PROPOSE_COMPETING_ROUTE,
                    target_hypothesis_id=parent.hypothesis_id if parent else None,
                    rationale="dry_run_llm",
                    parameters={"proposal": prop},
                )
            ],
        )

    def process_request(
        self,
        req_id: str,
        *,
        llm_caller: Callable[[ProviderSpec, str], dict[str, Any]] | None = None,
        dry_run_llm: bool = False,
        expected_lease_id: str | None = None,
    ) -> dict[str, Any]:
        """Execute one DecisionRequest (single-attempt `180s`, zero live dummy fallback, atomic tx fence)."""
        req = self.db.get_decision_request(req_id)
        if req is None:
            raise KeyError(f"Unknown decision request: {req_id}")
        goal = self.db.get_goal(str(req["goal_id"]))
        if goal is None:
            return self.db.resolve_decision_request(
                req_id,
                status="FAILED",
                resolution={"reason": f"Goal {req['goal_id']} not found"},
            )

        req_version = int(req["goal_version"])
        req_epoch = int(req["human_epoch"])
        req_attempt = int(req.get("attempt", 1))
        payload_dict = dict(req.get("payload") or {})
        req_plan_epoch = payload_dict.get("plan_epoch")
        cur_req_status = str(req.get("status") or "")
        eff_lease_id = (
            str(expected_lease_id).strip()
            if expected_lease_id and str(expected_lease_id).strip()
            else (
                str(payload_dict.get("lease_id")).strip()
                if payload_dict.get("lease_id") and str(payload_dict.get("lease_id")).strip()
                else None
            )
        )
        dec_ukey = str(
            payload_dict.get("usage_key")
            or (
                f"decision:{req_id}:a{req_attempt}:{payload_dict['submit_nonce']}"
                if payload_dict.get("submit_nonce")
                else f"decision:{req_id}:a{req_attempt}"
            )
        )

        if (
            int(goal.human_epoch) != req_epoch
            or int(goal.version) != req_version
            or (
                req_plan_epoch is not None
                and int(getattr(goal, "plan_epoch", 0) or 0) != int(req_plan_epoch)
            )
        ):
            if eff_lease_id:
                self.db.release_lease(eff_lease_id)
            return self.db.resolve_decision_request(
                req_id,
                status="DISCARDED_BY_NEWER_HUMAN_INTERVENTION",
                worker_self_exit=True,
                resolution={
                    "reason": "stale_before_execution",
                    "req_human_epoch": req_epoch,
                    "goal_human_epoch": int(goal.human_epoch),
                    "req_plan_epoch": req_plan_epoch,
                    "goal_plan_epoch": int(getattr(goal, "plan_epoch", 0) or 0),
                    "req_goal_version": req_version,
                    "goal_version": int(goal.version),
                },
            )

        tier = str(req["tier"]).upper()
        if tier == "L0":
            if eff_lease_id:
                self.db.release_lease(eff_lease_id)
            return self.db.resolve_decision_request(
                req_id,
                status="RESOLVED",
                resolution={"tier": "L0", "tokens_used": 0, "accepted_hypothesis_ids": []},
            )

        # Acquire exclusive inline lease ONLY when invoked synchronously on a `PENDING` request;
        # HQ-submitted (`SUBMITTING`/`RUNNING`) workers MUST present an active `eff_lease_id`!
        if eff_lease_id is None:
            if cur_req_status != "PENDING":
                raise FencedStaleWriteError(
                    f"DecisionRequest {req_id} in status={cur_req_status!r} cannot execute without active lease_id",
                    reason_code="LEASE_MISMATCH",
                )
            inline_nonce = uuid.uuid4().hex[:10]
            inline_lease = self.db.acquire_lease(
                f"decision:{goal.goal_id}:{req_id}",
                f"dec-{req_id}",
                holder=f"inline-dec-{req_id}-{inline_nonce}",
                ttl_s=float(DECISION_OUTER_WALL_TIMEOUT_S + 30),
                goal_id=goal.goal_id,
                allow_same_holder_renew=False,
            )
            if inline_lease is None or not str(inline_lease.get("lease_id") or "").strip():
                raise FencedStaleWriteError(
                    f"DecisionRequest {req_id} already holds a conflicting active lease",
                    reason_code="LEASE_CONFLICT",
                )
            eff_lease_id = str(inline_lease["lease_id"])

        role_enum = ScientificRole(
            str(req.get("role") or (ScientificRole.RESEARCH_DIRECTOR.value if tier == "L2" else ScientificRole.RESEARCH_LEAD.value))
        )

        # Atomic pre-execution check inside `db._tx()` before any LLM call!
        with self.db._tx() as (conn, _p):
            now_exec = time.time()
            row_exec = conn.execute(
                "SELECT status FROM decision_requests WHERE req_id = ?", (req_id,)
            ).fetchone()
            st_exec = str(row_exec["status"]) if row_exec is not None else "MISSING"
            if st_exec not in ("PENDING", "SUBMITTING", "RUNNING"):
                self.db.release_lease(eff_lease_id, now=now_exec)
                raise FencedStaleWriteError(
                    f"DecisionRequest {req_id} status is {st_exec!r} before execution",
                    reason_code="STATUS_MISMATCH",
                )
            if not self.db.lease_matches_in_tx(
                conn, f"dec-{req_id}", eff_lease_id, now=now_exec, allow_none=False
            ):
                raise FencedStaleWriteError(
                    f"DecisionRequest {req_id} lease {eff_lease_id!r} is missing or expired before execution",
                    reason_code="LEASE_MISMATCH",
                )
            if st_exec == "PENDING":
                self.db.resolve_decision_request(req_id, status="RUNNING", now=now_exec)

        hyps = self.db.list_hypotheses(goal_id=goal.goal_id)
        receipts = self.db.list_receipts(goal_id=goal.goal_id)
        prompt = self._build_role_prompt(req, goal, hyps, receipts)

        in_tok = 0
        out_tok = 0
        model_used = ""
        envelope: LeadDecisionEnvelope | None = None

        try:
            if dry_run_llm:
                model_used = "dry_run_llm"
                envelope = self._offline_dry_run_envelope(goal, hyps, role_enum)
            else:
                raw_pinned = payload_dict.get("pinned_route")
                fresh_goal_exec = self.db.get_goal(goal.goal_id) or goal
                hyp_id_exec = payload_dict.get("hypothesis_id") or payload_dict.get("target_hypothesis_id")
                fresh_hyp_exec = (
                    self.db.get_goal_hypothesis(fresh_goal_exec.goal_id, str(hyp_id_exec))
                    if (hyp_id_exec and hasattr(self.db, "get_goal_hypothesis"))
                    else (self.db.get_hypothesis(str(hyp_id_exec)) if hyp_id_exec else None)
                )
                wf_id_exec = str(payload_dict.get("workflow_id") or "").strip() or None
                eff_router = self.router or _try_build_default_router(
                    arbiter=self.arbiter,
                    db=self.db,
                )
                if isinstance(raw_pinned, Mapping):
                    route = ResolvedModelRoute.from_dict(raw_pinned)
                    if eff_router is not None and hasattr(eff_router, "revalidate_pinned_route"):
                        import inspect as _inspect_rv

                        rv_sig = _inspect_rv.signature(eff_router.revalidate_pinned_route).parameters
                        rv_varkw = any(
                            p.kind == _inspect_rv.Parameter.VAR_KEYWORD for p in rv_sig.values()
                        )
                        rv_extra: dict[str, Any] = {}
                        if "hyp" in rv_sig or rv_varkw:
                            rv_extra["hyp"] = fresh_hyp_exec
                        if "workflow_id" in rv_sig or rv_varkw:
                            rv_extra["workflow_id"] = wf_id_exec
                        if "db" in rv_sig or rv_varkw:
                            rv_extra["db"] = self.db
                        reval_reason = ""
                        reval_blocked = False
                        try:
                            reval_route = eff_router.revalidate_pinned_route(
                                role_enum,
                                route,
                                goal=fresh_goal_exec,
                                **rv_extra,
                            )
                            if (
                                reval_route.pool != route.pool
                                or reval_route.canonical_model != route.canonical_model
                                or reval_route.wire_effort != route.wire_effort
                            ):
                                reval_blocked = True
                                reval_reason = (
                                    f"pinned_route_policy_changed:{route.canonical_spec()}->{reval_route.canonical_spec()}"
                                )
                            else:
                                route = reval_route
                        except Exception as pol_exc:
                            if pol_exc.__class__.__name__ == "PolicyBlockedError" or hasattr(pol_exc, "waitable"):
                                reval_blocked = True
                                reval_reason = str(pol_exc)
                            else:
                                raise
                        if reval_blocked:
                            # Abort LLM invocation (0 tokens); check whether fresh policy can re-admit cleanly
                            can_requeue = True
                            try:
                                rr_sig = _inspect_rv.signature(eff_router.resolve_route).parameters
                                rr_varkw = any(
                                    p.kind == _inspect_rv.Parameter.VAR_KEYWORD for p in rr_sig.values()
                                )
                                rr_extra: dict[str, Any] = {}
                                if "hyp" in rr_sig or rr_varkw:
                                    rr_extra["hyp"] = fresh_hyp_exec
                                if "workflow_id" in rr_sig or rr_varkw:
                                    rr_extra["workflow_id"] = wf_id_exec
                                if "db" in rr_sig or rr_varkw:
                                    rr_extra["db"] = self.db
                                eff_router.resolve_route(role_enum, goal=fresh_goal_exec, **rr_extra)
                            except Exception as fresh_exc:
                                if (fresh_exc.__class__.__name__ == "PolicyBlockedError" or hasattr(fresh_exc, "waitable")) and not bool(
                                    getattr(fresh_exc, "waitable", False)
                                ):
                                    can_requeue = False
                            now_rv = time.time()
                            expected_nonce_rv = str(payload_dict.get("submit_nonce") or "")
                            discarded_by_goal_rv = False
                            with self.db._tx() as (conn_rv, _p_rv):
                                if not self.db.lease_matches_in_tx(
                                    conn_rv, f"dec-{req_id}", eff_lease_id, now=now_rv, allow_none=False
                                ):
                                    raise FencedStaleWriteError(
                                        f"DecisionRequest {req_id} lease {eff_lease_id!r} lost before policy revalidation update",
                                        reason_code="LEASE_MISMATCH",
                                    )
                                row_rv = conn_rv.execute(
                                    "SELECT status, attempt, goal_version, human_epoch, payload_json FROM decision_requests WHERE req_id = ?",
                                    (req_id,),
                                ).fetchone()
                                g_rv_row = conn_rv.execute(
                                    "SELECT status, payload_json FROM goals WHERE goal_id = ?",
                                    (goal.goal_id,),
                                ).fetchone()
                                if row_rv is None or str(row_rv["status"]) not in ("PENDING", "SUBMITTING", "RUNNING"):
                                    raise FencedStaleWriteError(
                                        f"DecisionRequest {req_id} status changed before policy revalidation update",
                                        reason_code="STATUS_MISMATCH",
                                    )
                                cur_pl_rv = json.loads(row_rv["payload_json"])
                                cur_att_rv = int(cur_pl_rv.get("attempt", row_rv["attempt"] or 1) or 1)
                                if cur_att_rv != req_attempt:
                                    raise FencedStaleWriteError(
                                        f"DecisionRequest {req_id} attempt mismatch ({cur_att_rv} != {req_attempt})",
                                        reason_code="ATTEMPT_MISMATCH",
                                    )
                                if expected_nonce_rv and str(cur_pl_rv.get("submit_nonce") or "") != expected_nonce_rv:
                                    raise FencedStaleWriteError(
                                        f"DecisionRequest {req_id} nonce superseded before policy revalidation update",
                                        reason_code="NONCE_MISMATCH",
                                    )
                                g_rv_obj = (
                                    ResearchGoal.from_dict(json.loads(g_rv_row["payload_json"]))
                                    if g_rv_row is not None
                                    else None
                                )
                                g_rv_stale = (
                                    g_rv_row is None
                                    or g_rv_obj is None
                                    or str(g_rv_row["status"]).upper() != "ACTIVE"
                                    or g_rv_obj.autonomy_mode == "paused"
                                    or int(g_rv_obj.version) != req_version
                                    or int(g_rv_obj.human_epoch) != req_epoch
                                    or (
                                        req_plan_epoch is not None
                                        and int(getattr(g_rv_obj, "plan_epoch", 0) or 0) != int(req_plan_epoch)
                                    )
                                )
                                if g_rv_stale:
                                    discarded_by_goal_rv = True
                                elif can_requeue:
                                    cleaned_rv = {
                                        k: v
                                        for k, v in payload_dict.items()
                                        if k
                                        not in {
                                            "pinned_route",
                                            "lease_id",
                                            "submit_nonce",
                                            "usage_key",
                                            "submit_job_name",
                                            "submit_owner_pid",
                                            "submit_proc_start_time",
                                            "submitting_started_ts",
                                        }
                                    }
                                    cleaned_rv["attempt"] = req_attempt + 1
                                    cleaned_rv["requeued_reason"] = f"POLICY_REVALIDATED_REQUEUE:{reval_reason}"
                                    cur_upd_rv = conn_rv.execute(
                                        """
                                        UPDATE decision_requests
                                        SET status = 'PENDING', attempt = ?, model_spec = '', hq_job_id = NULL, payload_json = ?, updated_ts = ?
                                        WHERE req_id = ? AND status IN ('PENDING', 'SUBMITTING', 'RUNNING')
                                        """,
                                        (
                                            req_attempt + 1,
                                            json.dumps(cleaned_rv, sort_keys=True, ensure_ascii=False),
                                            now_rv,
                                            req_id,
                                        ),
                                    )
                                    if int(cur_upd_rv.rowcount or 0) != 1:
                                        raise FencedStaleWriteError(
                                            f"Fenced requeue CAS failed on DecisionRequest {req_id}",
                                            reason_code="CAS_FAILED",
                                        )
                                else:
                                    blocked_res_rv = self.db.fenced_resolve_decision_request(
                                        req_id,
                                        expected_goal_version=req_version,
                                        expected_human_epoch=req_epoch,
                                        expected_attempt=req_attempt,
                                        expected_lease_id=eff_lease_id,
                                        expected_statuses=("PENDING", "SUBMITTING", "RUNNING"),
                                        status="BLOCKED_POLICY",
                                        resolution={
                                            "role": role_enum.value,
                                            "tier": tier,
                                            "reason": reval_reason,
                                            "in_tokens": 0,
                                            "out_tokens": 0,
                                            "accepted_hypothesis_ids": [],
                                        },
                                        now=now_rv,
                                    )

                            if discarded_by_goal_rv:
                                if eff_lease_id:
                                    self.db.release_lease(eff_lease_id, now=now_rv)
                                return self.db.resolve_decision_request(
                                    req_id,
                                    status="DISCARDED_BY_NEWER_HUMAN_INTERVENTION",
                                    worker_self_exit=True,
                                    resolution={
                                        "reason": f"stale_during_policy_revalidation:{reval_reason}",
                                        "in_tokens": 0,
                                        "out_tokens": 0,
                                        "accepted_hypothesis_ids": [],
                                    },
                                )
                            if can_requeue:
                                if hasattr(self.db, "release_goal_budget_reservation"):
                                    self.db.release_goal_budget_reservation(
                                        goal.goal_id, dec_ukey, reason="pinned_route_policy_requeued", now=now_rv
                                    )
                                if eff_lease_id:
                                    self.db.release_lease(eff_lease_id, now=now_rv)
                                return self.db.get_decision_request(req_id) or {
                                    "req_id": req_id,
                                    "status": "PENDING",
                                    "requeued_reason": f"POLICY_REVALIDATED_REQUEUE:{reval_reason}",
                                }
                            if hasattr(self.db, "release_goal_budget_reservation"):
                                self.db.release_goal_budget_reservation(
                                    goal.goal_id, dec_ukey, reason="pinned_route_blocked_policy", now=now_rv
                                )
                            if eff_lease_id:
                                self.db.release_lease(eff_lease_id, now=now_rv)
                            return blocked_res_rv
                else:
                    route = self.resolve_route_for_role(
                        role_enum,
                        goal=fresh_goal_exec,
                        hyp=fresh_hyp_exec,
                        workflow_id=wf_id_exec,
                    )
                if hasattr(self.db, "reserve_goal_budget"):
                    res_tok = (
                        min(4000, max(1, int(goal.max_tokens_budget) // 2))
                        if int(goal.max_tokens_budget) > 0
                        else 4000
                    )
                    ok_b, committed_tok, max_tok = self.db.reserve_goal_budget(
                        goal.goal_id,
                        dec_ukey,
                        entity_kind="decision_request",
                        entity_id=req_id,
                        attempt=req_attempt,
                        model=route.canonical_spec(),
                        reserve_tokens=res_tok,
                        exclude_usage_key=dec_ukey,
                    )
                    if not ok_b:
                        if eff_lease_id:
                            self.db.release_lease(eff_lease_id)
                        return self.db.resolve_decision_request(
                            req_id,
                            status="BUDGET_EXHAUSTED",
                            resolution={
                                "role": role_enum.value,
                                "tier": tier,
                                "reason": f"BUDGET_EXHAUSTED: committed={committed_tok} / budget={max_tok}",
                                "in_tokens": 0,
                                "out_tokens": 0,
                                "accepted_hypothesis_ids": [],
                            },
                        )
                prov = ProviderSpec(
                    pool=route.pool,
                    model=route.canonical_model,
                    effort=route.wire_effort,
                    family=route.family,
                )
                art_dir = self.db.arr_home / "artifacts" / "decisions" / req_id
                if llm_caller is not None:
                    res = llm_caller(prov, prompt)
                else:
                    cwd_path = (
                        Path(goal.repo_path)
                        if Path(goal.repo_path).is_dir()
                        else self.db.arr_home
                    )
                    # Single-attempt (`<=1` CLI process invocation per HQ job, 180s cap)
                    res = invoke_provider(
                        prov,
                        prompt,
                        cwd=cwd_path,
                        art_dir=art_dir,
                        tag=f"{req_id}_a{req_attempt}",
                        readonly=True,
                        timeout_s=DECISION_INNER_TIMEOUT_S,
                    )
                in_tok = int(res.get("in_tokens", 0))
                out_tok = int(res.get("out_tokens", 0))
                model_used = str(res.get("model_used") or route.canonical_spec())
                if int(res.get("exit_code", 1)) != 0:
                    return self.db.resolve_decision_request(
                        req_id,
                        status="FAILED",
                        resolution={
                            "role": role_enum.value,
                            "tier": tier,
                            "failure_domain": FailureDomain.INFRA_TRANSIENT.value,
                            "reason": f"Provider exited with code {res.get('exit_code')}",
                            "in_tokens": in_tok,
                            "out_tokens": out_tok,
                            "model_used": model_used,
                            "used_fallback": False,
                            "accepted_hypothesis_ids": [],
                        },
                    )
                try:
                    envelope = parse_lead_decision_payload(
                        str(res.get("response") or ""),
                        goal=goal,
                        default_role=role_enum,
                    )
                except ValueError as exc:
                    return self.db.resolve_decision_request(
                        req_id,
                        status="FAILED",
                        resolution={
                            "role": role_enum.value,
                            "tier": tier,
                            "failure_domain": FailureDomain.INFRA_TRANSIENT.value,
                            "reason": f"Decision output parse failed: {exc}",
                            "in_tokens": in_tok,
                            "out_tokens": out_tok,
                            "model_used": model_used,
                            "used_fallback": False,
                            "accepted_hypothesis_ids": [],
                        },
                    )
        except DecisionInterruptedError as exc:
            if eff_lease_id:
                self.db.release_lease(eff_lease_id)
            return self.db.resolve_decision_request(
                req_id,
                status="CANCELLED",
                worker_self_exit=True,
                resolution={
                    "role": role_enum.value,
                    "reason": str(exc),
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                },
            )

        assert envelope is not None

        # Single atomic `BEGIN IMMEDIATE` transaction wrapping fence check + action mutations + RESOLVED commit!
        try:
            with self.db._tx():
                self.db.fenced_resolve_decision_request(
                    req_id,
                    expected_goal_version=req_version,
                    expected_human_epoch=req_epoch,
                    expected_attempt=req_attempt,
                    expected_lease_id=eff_lease_id,
                    expected_statuses=("RUNNING", "SUBMITTING", "PENDING"),
                    status="RUNNING",
                )
                latest_goal = self.db.get_goal(goal.goal_id) or goal
                action_summary = apply_scientific_actions(
                    self.db,
                    latest_goal,
                    envelope,
                    workflow_runtime=self.workflow_runtime,
                    source=f"decision_{role_enum.value}",
                )
                self.db.upsert_knowledge_item(
                    latest_goal.goal_id,
                    f"decision_{role_enum.value}",
                    f"req:{req_id}",
                    {
                        "req_id": req_id,
                        "role": role_enum.value,
                        "tier": tier,
                        "trigger_kind": req.get("trigger_kind"),
                        "model_used": model_used,
                        "used_fallback": bool(dry_run_llm),
                        "uncertainty_summary": envelope.uncertainty_summary,
                        "accepted_ids": action_summary["accepted_hypothesis_ids"],
                        "action_summary": action_summary,
                        "in_tokens": in_tok,
                        "out_tokens": out_tok,
                    },
                    goal_version=int(latest_goal.version),
                )
                res_payload = {
                    "role": role_enum.value,
                    "tier": tier,
                    "model_used": model_used,
                    "used_fallback": bool(dry_run_llm),
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                    "accepted_hypothesis_ids": action_summary["accepted_hypothesis_ids"],
                    "rejected_count": len(action_summary["rejected"]),
                    "action_summary": action_summary,
                }
                out = self.db.fenced_resolve_decision_request(
                    req_id,
                    expected_goal_version=req_version,
                    expected_human_epoch=req_epoch,
                    expected_attempt=req_attempt,
                    expected_lease_id=eff_lease_id,
                    expected_statuses=("RUNNING", "SUBMITTING", "PENDING"),
                    status="RESOLVED",
                    resolution=res_payload,
                )
        except FencedStaleWriteError as fence_err:
            latest_goal = self.db.get_goal(goal.goal_id)
            if eff_lease_id:
                self.db.release_lease(eff_lease_id)
            if fence_err.reason_code in ("HUMAN_INVALIDATED", "STALE_GOAL_VERSION"):
                return self.db.resolve_decision_request(
                    req_id,
                    status="DISCARDED_BY_NEWER_HUMAN_INTERVENTION",
                    worker_self_exit=True,
                    resolution={
                        "reason": fence_err.reason_code,
                        "req_human_epoch": req_epoch,
                        "current_human_epoch": None if latest_goal is None else int(latest_goal.human_epoch),
                        "req_goal_version": req_version,
                        "current_goal_version": None if latest_goal is None else int(latest_goal.version),
                        "in_tokens": in_tok,
                        "out_tokens": out_tok,
                        "model_used": model_used,
                    },
                )
            return self.db.resolve_decision_request(
                req_id,
                status="FAILED",
                resolution={
                    "reason": fence_err.reason_code,
                    "detail": str(fence_err),
                    "in_tokens": in_tok,
                    "out_tokens": out_tok,
                    "model_used": model_used,
                },
            )
        finally:
            if eff_lease_id:
                self.db.release_lease(eff_lease_id)
        return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Execute one Conjectra DecisionRequest asynchronously")
    ap.add_argument("--req-id", required=True)
    ap.add_argument("--arr-home", default=None)
    ap.add_argument("--lease-id", default=None)
    ap.add_argument("--dry-run-llm", action="store_true")
    args = ap.parse_args(argv)

    def _handle_sig(signum: int, _frame: Any) -> None:
        raise DecisionInterruptedError(f"Interrupted by signal {signum}")

    prev_int = signal.signal(signal.SIGINT, _handle_sig)
    prev_term = signal.signal(signal.SIGTERM, _handle_sig)

    db = StateDB(resolve_arr_home(args.arr_home))
    try:
        engine = DecisionEngine(db)
        out = engine.process_request(
            args.req_id,
            dry_run_llm=bool(args.dry_run_llm),
            expected_lease_id=args.lease_id,
        )
        print(json.dumps(out, indent=2, sort_keys=True, ensure_ascii=False))
        return 0
    finally:
        signal.signal(signal.SIGINT, prev_int)
        signal.signal(signal.SIGTERM, prev_term)
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
