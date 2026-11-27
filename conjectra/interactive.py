"""Sub-50ms WAL-backed interactive control, overview, token-audit, and <3KB takeover layer."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from conjectra.arbiter import (
    GlobalResourceScheduler,
    read_active_foreground_pings,
    record_foreground_ping,
)
from conjectra.metrics import compute_per_goal_empirical_efficiency
from conjectra.state_db import StateDB, resolve_arr_home
from conjectra.types import (
    EvidenceValidity,
    FailureDomain,
    ResearchGoal,
    ReviewKind,
    ScientificVerdict,
    SubGoalNode,
)

ACTIVE_TASK_STATES = frozenset({"DISPATCHED", "RUNNING"})
DEFAULT_TAKEOVER_MAX_BYTES = 2950


def _record_latency_sample(
    home: Path,
    command: str,
    latency_ms: float,
    *,
    provider: str = "interactive",
    record_ping: bool = True,
    ttl_s: float = 120.0,
) -> dict[str, Any] | None:
    """Record foreground ping and lightweight interactive latency telemetry (zero SQLite lock)."""
    ping_info = None
    if record_ping:
        ping_info = record_foreground_ping(home, provider=provider, ttl_s=ttl_s)
    metrics_path = home / "interactive_latency.json"
    try:
        snapshot = {
            "command": command,
            "interactive_latency_ms": round(float(latency_ms), 3),
            "ts": time.time(),
            "provider": provider,
        }
        metrics_path.write_text(
            json.dumps(snapshot, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except OSError:
        pass
    return ping_info


def _get_owner_escalations(db: StateDB, *, goal_id: str | None = None) -> list[dict[str, Any]]:
    conn = getattr(db, "_conn", None)
    db_lock = getattr(db, "_lock", None)
    if conn is not None and db_lock is not None:
        with db_lock:
            if goal_id is not None:
                rows = conn.execute(
                    "SELECT seq, ts, goal_id, hypothesis_id, payload_json FROM events "
                    "WHERE event_type = 'OWNER_ESCALATION' AND goal_id = ? ORDER BY seq ASC LIMIT 500",
                    (goal_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT seq, ts, goal_id, hypothesis_id, payload_json FROM events "
                    "WHERE event_type = 'OWNER_ESCALATION' ORDER BY seq ASC LIMIT 500"
                ).fetchall()
        events = [
            {
                "seq": r["seq"],
                "ts": r["ts"],
                "goal_id": r["goal_id"],
                "hypothesis_id": r["hypothesis_id"],
                "event_type": "OWNER_ESCALATION",
                "payload": json.loads(r["payload_json"] or "{}"),
            }
            for r in rows
        ]
    else:
        events = db.list_events(goal_id=goal_id, since_seq=0, limit=500)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for ev in events:
        if ev.get("event_type") != "OWNER_ESCALATION":
            continue
        p = ev.get("payload") or {}
        gid = str(ev.get("goal_id") or p.get("goal_id") or "")
        gate = str(p.get("gate") or "")
        reason = str(p.get("reason") or "")
        key = (gid, gate, reason)
        if key in seen:
            continue
        seen.add(key)
        out.append(
            {
                "seq": ev.get("seq"),
                "ts": ev.get("ts"),
                "goal_id": gid,
                "hypothesis_id": ev.get("hypothesis_id"),
                "gate": gate,
                "reason": reason,
            }
        )
    return out


def _open_db(arr_home: str | Path | None, db: StateDB | None) -> tuple[Path, StateDB, bool]:
    home = resolve_arr_home(arr_home)
    if db is not None:
        return db.arr_home, db, False
    return home, StateDB(home), True


def _build_subgoals_summary(subgoals: Sequence[SubGoalNode]) -> dict[str, Any]:
    """Build a compact WAL-backed SubGoal summary dict (counts + per-subgoal essentials)."""
    counts: dict[str, int] = {
        "total": len(subgoals),
        "active": 0,
        "locked_dep": 0,
        "paused": 0,
        "awaiting_apex": 0,
        "sealed": 0,
        "merged": 0,
        "killed": 0,
    }
    items: list[dict[str, Any]] = []
    for sg in subgoals:
        st_str = str(sg.status.value if hasattr(sg.status, "value") else sg.status)
        st_key = st_str.strip().lower()
        counts[st_key] = counts.get(st_key, 0) + 1
        items.append(
            {
                "subgoal_id": sg.subgoal_id,
                "goal_id": sg.goal_id,
                "title": sg.title,
                "kind": sg.kind.value if hasattr(sg.kind, "value") else str(sg.kind),
                "status": st_str,
                "local_metric": sg.local_metric,
                "target_value": sg.target_value,
                "sealed_sha": sg.sealed_sha,
                "priority_weight": float(sg.priority_weight),
            }
        )
    return {"counts": counts, "subgoals": items}


def get_overview(
    arr_home: str | Path | None = None,
    *,
    db: StateDB | None = None,
    provider: str = "interactive",
    record_ping: bool = True,
    total_slots: int | None = None,
    scheduler: GlobalResourceScheduler | None = None,
) -> dict[str, Any]:
    """Return sub-50ms multi-Goal overview: priorities, active/borrowed slots, budgets, bottlenecks."""
    t0 = time.perf_counter()
    home, active_db, own_db = _open_db(arr_home, db)
    try:
        goal_records = active_db.list_goal_records()
        active_goal_ids = [
            rec["goal"].goal_id for rec in goal_records if str(rec["status"]) == "ACTIVE"
        ]
        all_subgoals = (
            active_db.list_subgoals() if hasattr(active_db, "list_subgoals") else []
        )
        subgoals_by_goal: dict[str, list[SubGoalNode]] = {}
        for sg in all_subgoals:
            subgoals_by_goal.setdefault(sg.goal_id, []).append(sg)
        subgoals_summary = _build_subgoals_summary(all_subgoals)
        pending_decisions = active_db.list_decision_requests(status="PENDING")
        running_decisions = active_db.list_decision_requests(status="RUNNING")
        active_leases = active_db.list_active_leases()
        global_cfg = active_db.get_global_config()
        escalations = _get_owner_escalations(active_db)

        active_by_goal: dict[str, int] = {}
        ready_by_goal: dict[str, int] = {}
        failed_by_goal: dict[str, int] = {}
        blocked_tasks_by_goal: dict[str, set[str]] = {}
        blocked_hyps_by_goal: dict[str, set[str]] = {}
        failure_domains_by_goal: dict[str, dict[str, int]] = {}
        tokens_by_goal: dict[str, dict[str, int]] = {}

        conn = getattr(active_db, "_conn", None)
        db_lock = getattr(active_db, "_lock", None)
        if conn is not None and db_lock is not None:
            with db_lock:
                task_rows = conn.execute(
                    "SELECT goal_id, status, COUNT(*) AS cnt FROM tasks GROUP BY goal_id, status"
                ).fetchall()
                workflows_total = int(conn.execute("SELECT COUNT(*) FROM workflows").fetchone()[0])
                rcpt_rows = conn.execute(
                    "SELECT goal_id, COALESCE(SUM(in_tokens), 0) AS in_tok, "
                    "COALESCE(SUM(out_tokens), 0) AS out_tok, COUNT(*) AS cnt "
                    "FROM evidence_receipts GROUP BY goal_id"
                ).fetchall()
                hyp_rows = conn.execute(
                    "SELECT goal_id, status, json_extract(payload_json, '$.failure_domain') AS fd "
                    "FROM hypotheses "
                    "WHERE SUBSTR(status, 1, 8) = 'BLOCKED_' "
                    "OR (json_extract(payload_json, '$.failure_domain') IS NOT NULL "
                    "AND UPPER(CAST(json_extract(payload_json, '$.failure_domain') AS TEXT)) != 'NONE')"
                ).fetchall()
            for r_t in task_rows:
                gid = str(r_t["goal_id"] or "")
                st_str = str(r_t["status"] or "")
                cnt = int(r_t["cnt"] or 0)
                if st_str in ACTIVE_TASK_STATES:
                    active_by_goal[gid] = active_by_goal.get(gid, 0) + cnt
                elif st_str == "READY":
                    ready_by_goal[gid] = ready_by_goal.get(gid, 0) + cnt
                elif st_str == "FAILED":
                    failed_by_goal[gid] = failed_by_goal.get(gid, 0) + cnt
                elif st_str.startswith("BLOCKED_"):
                    blocked_tasks_by_goal.setdefault(gid, set()).add(st_str)
            for r_rc in rcpt_rows:
                gid = str(r_rc["goal_id"] or "")
                tokens_by_goal[gid] = {
                    "in_tokens": int(r_rc["in_tok"] or 0),
                    "out_tokens": int(r_rc["out_tok"] or 0),
                    "receipts": int(r_rc["cnt"] or 0),
                }
            for r_h in hyp_rows:
                gid = str(r_h["goal_id"] or "")
                hst = str(r_h["status"] or "")
                if hst.startswith("BLOCKED_"):
                    blocked_hyps_by_goal.setdefault(gid, set()).add(hst)
                fd_raw = r_h["fd"]
                if fd_raw is not None:
                    fd_norm = str(fd_raw).strip()
                    try:
                        fd_val = FailureDomain(fd_norm.upper()).value
                    except ValueError:
                        fd_val = fd_norm
                    if fd_val and fd_val != FailureDomain.NONE.value:
                        gb_fd = failure_domains_by_goal.setdefault(gid, {})
                        gb_fd[fd_val] = gb_fd.get(fd_val, 0) + 1
            eff_receipts_arg = None
            eff_reviews_arg = None
            if active_goal_ids:
                eff_receipts_arg = [
                    r
                    for gid in active_goal_ids
                    for r in active_db.list_receipts(goal_id=gid)
                ]
                if hasattr(active_db, "list_review_receipts"):
                    eff_reviews_arg = [
                        rv
                        for gid in active_goal_ids
                        for rv in active_db.list_review_receipts(goal_id=gid)
                    ]
        else:
            all_tasks = active_db.list_tasks()
            all_receipts = active_db.list_receipts()
            all_workflows = active_db.list_workflows()
            workflows_total = len(all_workflows)
            for t in all_tasks:
                if t.status in ACTIVE_TASK_STATES:
                    active_by_goal[t.goal_id] = active_by_goal.get(t.goal_id, 0) + 1
                elif t.status == "READY":
                    ready_by_goal[t.goal_id] = ready_by_goal.get(t.goal_id, 0) + 1
                elif t.status == "FAILED":
                    failed_by_goal[t.goal_id] = failed_by_goal.get(t.goal_id, 0) + 1
                elif t.status.startswith("BLOCKED_"):
                    blocked_tasks_by_goal.setdefault(t.goal_id, set()).add(t.status)

            for h in active_db.list_hypotheses():
                if h.status.startswith("BLOCKED_"):
                    blocked_hyps_by_goal.setdefault(h.goal_id, set()).add(h.status)
                fd = (
                    h.failure_domain.value
                    if isinstance(h.failure_domain, FailureDomain)
                    else str(h.failure_domain)
                )
                if fd and fd != FailureDomain.NONE.value:
                    gb_fd = failure_domains_by_goal.setdefault(h.goal_id, {})
                    gb_fd[fd] = gb_fd.get(fd, 0) + 1

            for r in all_receipts:
                bucket = tokens_by_goal.setdefault(
                    r.goal_id, {"in_tokens": 0, "out_tokens": 0, "receipts": 0}
                )
                bucket["in_tokens"] += int(r.in_tokens)
                bucket["out_tokens"] += int(r.out_tokens)
                bucket["receipts"] += 1
            eff_receipts_arg = all_receipts
            eff_reviews_arg = None

        esc_by_goal: dict[str, list[dict[str, Any]]] = {}
        for esc in escalations:
            esc_by_goal.setdefault(esc["goal_id"], []).append(esc)

        pending_dec_by_goal: dict[str, int] = {}
        for d in [*pending_decisions, *running_decisions]:
            gid = str(d.get("goal_id") or "")
            pending_dec_by_goal[gid] = pending_dec_by_goal.get(gid, 0) + 1

        class _GoalView:
            def __init__(self, g: ResearchGoal, status_str: str) -> None:
                self.goal_id = g.goal_id
                self.status = status_str
                self.priority_weight = g.priority_weight
                self.min_guaranteed_slots = g.min_guaranteed_slots
                self.max_concurrent_branches = g.max_concurrent_branches

        blocked_goals: set[str] = set()
        goal_views: list[_GoalView] = []
        for rec in goal_records:
            g: ResearchGoal = rec["goal"]
            st = str(rec["status"])
            goal_views.append(_GoalView(g, st))
            if (
                st == "ACTIVE"
                and ready_by_goal.get(g.goal_id, 0) == 0
                and active_by_goal.get(g.goal_id, 0) == 0
                and pending_dec_by_goal.get(g.goal_id, 0) > 0
            ):
                blocked_goals.add(g.goal_id)

        eff_total_slots = (
            int(total_slots)
            if total_slots is not None
            else int(global_cfg.get("total_slots") or max(4, sum(g.max_concurrent_branches for g in goal_views) or 4))
        )
        fg_reserve_cfg = int(global_cfg.get("foreground_reserve_slots", 1))
        sched = scheduler or GlobalResourceScheduler(
            arr_home=home,
            foreground_reserve_slots=fg_reserve_cfg,
        )
        if record_ping:
            record_foreground_ping(home, provider=provider, ttl_s=120.0)
        allocs = sched.compute_goal_slot_allocations(
            goal_views,
            active_by_goal=active_by_goal,
            ready_by_goal=ready_by_goal,
            blocked_goals=blocked_goals,
            total_slots=eff_total_slots,
            arr_home=home,
        )

        goal_rows: list[dict[str, Any]] = []
        bottlenecks: list[dict[str, Any]] = []
        total_tokens_used = 0
        for rec in goal_records:
            g = rec["goal"]
            st = str(rec["status"])
            tok = tokens_by_goal.get(g.goal_id, {"in_tokens": 0, "out_tokens": 0, "receipts": 0})
            used = tok["in_tokens"] + tok["out_tokens"]
            total_tokens_used += used
            alloc = allocs.get(g.goal_id, {})
            g_escs = esc_by_goal.get(g.goal_id, [])
            g_dec_cnt = pending_dec_by_goal.get(g.goal_id, 0)
            goal_bottlenecks: list[str] = []
            if st != "ACTIVE":
                goal_bottlenecks.append(f"STATUS_{st}")
            if g.autonomy_mode in ("paused", "supervised"):
                goal_bottlenecks.append(f"AUTONOMY_{g.autonomy_mode.upper()}")
            if g_dec_cnt > 0:
                goal_bottlenecks.append(f"PENDING_DECISIONS:{g_dec_cnt}")
            if used >= g.max_tokens_budget > 0:
                goal_bottlenecks.append("TOKEN_BUDGET_EXHAUSTED")
            if g_escs:
                goal_bottlenecks.append(f"OWNER_ESCALATION:{g_escs[0]['gate']}")
            if failed_by_goal.get(g.goal_id, 0) > 0:
                goal_bottlenecks.append(f"FAILED_TASKS:{failed_by_goal[g.goal_id]}")

            rplan = active_db.get_latest_research_plan(g.goal_id)
            wait_items = active_db.list_knowledge_items(
                goal_id=g.goal_id, category="dispatch_bottleneck", limit=10
            )
            waiting_model_cnt = sum(
                1
                for item in wait_items
                if isinstance(item.get("content"), Mapping)
                and item["content"].get("status")
                in ("WAITING_FOR_PINNED_MODEL", "WAIT_RESOURCE", "WAIT_FOREGROUND_RESERVE")
            )
            if waiting_model_cnt > 0 and ready_by_goal.get(g.goal_id, 0) > 0:
                goal_bottlenecks.append(f"WAITING_FOR_PINNED_MODEL:{waiting_model_cnt}")

            blocked_statuses = set(blocked_tasks_by_goal.get(g.goal_id, ())) | set(
                blocked_hyps_by_goal.get(g.goal_id, ())
            )
            for bst in sorted(blocked_statuses):
                goal_bottlenecks.append(bst)

            open_uncertainties = 0
            resolved_uncertainties = 0
            if rplan is not None:
                for u in rplan.uncertainties:
                    if str(u.get("status") or "OPEN").upper() in ("RESOLVED", "CLOSED", "CONFIRMED", "REFUTED"):
                        resolved_uncertainties += 1
                    else:
                        open_uncertainties += 1

            for b in goal_bottlenecks:
                bottlenecks.append({"goal_id": g.goal_id, "bottleneck": b})

            goal_rows.append(
                {
                    "goal_id": g.goal_id,
                    "project": g.project,
                    "title": g.title,
                    "status": st,
                    "version": g.version,
                    "human_epoch": g.human_epoch,
                    "autonomy_mode": g.autonomy_mode,
                    "priority_weight": g.priority_weight,
                    "primary_metric": g.primary_metric,
                    "direction": g.direction,
                    "baseline_value": g.baseline_value,
                    "target_value": g.target_value,
                    "min_guaranteed_slots": g.min_guaranteed_slots,
                    "max_concurrent_branches": g.max_concurrent_branches,
                    "active_slots": active_by_goal.get(g.goal_id, 0),
                    "ready_tasks": ready_by_goal.get(g.goal_id, 0),
                    "guaranteed_slots": alloc.get("guaranteed_slots", 0),
                    "allocated_slots": alloc.get("allocated_slots", 0),
                    "borrowed_slots": alloc.get("borrowed_slots", 0),
                    "blocked": g.goal_id in blocked_goals or bool(blocked_statuses),
                    "tokens_used": used,
                    "max_tokens_budget": g.max_tokens_budget,
                    "receipts_count": tok["receipts"],
                    "pending_decisions": g_dec_cnt,
                    "escalations": len(g_escs),
                    "routing_policy_nl": g.routing_policy_nl,
                    "routing_policy_rules": len((g.routing_policy or {}).get("rules") or []),
                    "research_plan_version": rplan.plan_version if rplan is not None else None,
                    "open_uncertainties": open_uncertainties,
                    "resolved_uncertainties": resolved_uncertainties,
                    "waiting_for_pinned_model": waiting_model_cnt,
                    "failure_domains": failure_domains_by_goal.get(g.goal_id, {}),
                    "bottlenecks": goal_bottlenecks,
                    "subgoals_summary": _build_subgoals_summary(subgoals_by_goal.get(g.goal_id, [])),
                }
            )

        if not active_goal_ids:
            per_goal_efficiency: dict[str, dict[str, Any]] = {}
        elif len(active_goal_ids) == 1 and conn is not None and db_lock is not None:
            per_goal_efficiency = compute_per_goal_empirical_efficiency(
                active_db,
                goal_id=active_goal_ids[0],
                goal_records=goal_records,
                receipts=eff_receipts_arg,
                reviews=eff_reviews_arg,
            )
        else:
            per_goal_efficiency = compute_per_goal_empirical_efficiency(
                active_db,
                goal_records=goal_records,
                receipts=eff_receipts_arg,
                reviews=eff_reviews_arg,
            )
        pings = read_active_foreground_pings(home)
        global_budget = int(global_cfg.get("global_max_tokens_budget", 0) or 0)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_latency_sample(
            home, "overview", elapsed_ms, provider=provider, record_ping=False
        )
        return {
            "arr_home": str(home),
            "reactor_sock_exists": (home / "reactor.sock").exists(),
            "goals": goal_rows,
            "subgoals_summary": subgoals_summary,
            "per_goal_empirical_efficiency": per_goal_efficiency,
            "counts": {
                "goals_total": len(goal_rows),
                "goals_active": sum(1 for r in goal_rows if r["status"] == "ACTIVE"),
                "goals_paused": sum(1 for r in goal_rows if r["status"] == "PAUSED"),
                "subgoals_total": subgoals_summary["counts"]["total"],
                "subgoals_active": subgoals_summary["counts"]["active"],
                "tasks_active": sum(active_by_goal.values()),
                "tasks_ready": sum(ready_by_goal.values()),
                "workflows_total": workflows_total,
                "active_leases": len(active_leases),
                "pending_decisions": len(pending_decisions) + len(running_decisions),
            },
            "global_budget": {
                "global_max_tokens_budget": global_budget,
                "total_tokens_used": total_tokens_used,
            },
            "foreground_reserve": {
                "reserve_slots": fg_reserve_cfg,
                "active_pings": pings,
                "foreground_active": bool(pings),
                "total_slots": eff_total_slots,
            },
            "bottlenecks": bottlenecks,
            "interactive_latency_ms": round(elapsed_ms, 3),
        }
    finally:
        if own_db:
            active_db.close()


def inspect_state(
    arr_home: str | Path | None = None,
    *,
    goal_id: str | None = None,
    subgoal_id: str | None = None,
    hypothesis_id: str | None = None,
    workflow_id: str | None = None,
    db: StateDB | None = None,
    provider: str = "interactive",
    record_ping: bool = True,
) -> dict[str, Any]:
    """Inspect Goal, SubGoal, Hypothesis, or Workflow details in <50ms from SQLite WAL."""
    t0 = time.perf_counter()
    home, active_db, own_db = _open_db(arr_home, db)
    try:
        if record_ping:
            record_foreground_ping(home, provider=provider, ttl_s=120.0)

        out: dict[str, Any] = {"arr_home": str(home)}
        if workflow_id is not None:
            wf = active_db.get_workflow(workflow_id)
            if wf is None:
                raise KeyError(f"Unknown workflow_id: {workflow_id}")
            tasks = [t for t in active_db.list_tasks(goal_id=wf.goal_id, hypothesis_id=wf.hypothesis_id) if t.workflow_id == workflow_id]
            receipts = [r for r in active_db.list_receipts(goal_id=wf.goal_id, hypothesis_id=wf.hypothesis_id) if r.workflow_id == workflow_id]
            children = active_db.list_workflows(parent_workflow_id=workflow_id)
            exec_plan = active_db.get_execution_plan_by_workflow(workflow_id)
            rev_receipts = active_db.list_review_receipts(workflow_id=workflow_id)
            out["workflow"] = wf.to_dict()
            out["execution_plan"] = exec_plan.to_dict() if exec_plan is not None else None
            out["review_receipts"] = [rv.to_dict() for rv in rev_receipts]
            out["child_workflows"] = [c.to_dict() for c in children]
            out["tasks"] = [t.to_dict() for t in tasks]
            out["receipts"] = [r.to_dict() for r in receipts[-20:]]
        elif hypothesis_id is not None:
            hyp = active_db.get_hypothesis(hypothesis_id)
            if hyp is None:
                raise KeyError(f"Unknown hypothesis_id: {hypothesis_id}")
            wfs = active_db.list_workflows(hypothesis_id=hypothesis_id)
            tasks = active_db.list_tasks(hypothesis_id=hypothesis_id)
            receipts = active_db.list_receipts(hypothesis_id=hypothesis_id)
            knowledge = active_db.list_knowledge_items(hypothesis_id=hypothesis_id)
            exp_plans = active_db.list_experiment_plans(hypothesis_id=hypothesis_id)
            exec_plans = active_db.list_execution_plans(hypothesis_id=hypothesis_id)
            rev_receipts = active_db.list_review_receipts(hypothesis_id=hypothesis_id)
            out["hypothesis"] = hyp.to_dict()
            if hyp.subgoal_id and hasattr(active_db, "get_subgoal"):
                sg_parent = active_db.get_subgoal(hyp.subgoal_id)
                if sg_parent is not None:
                    out["subgoals_summary"] = _build_subgoals_summary([sg_parent])
            out["experiment_plans"] = [ep.to_dict() for ep in exp_plans]
            out["execution_plans"] = [xp.to_dict() for xp in exec_plans]
            out["review_receipts"] = [rv.to_dict() for rv in rev_receipts]
            out["workflows"] = [w.to_dict() for w in wfs]
            out["tasks"] = [t.to_dict() for t in tasks]
            out["receipts"] = [r.to_dict() for r in receipts[-20:]]
            out["knowledge"] = knowledge
        elif subgoal_id is not None:
            sg = (
                active_db.get_goal_subgoal(goal_id, subgoal_id)
                if goal_id and hasattr(active_db, "get_goal_subgoal")
                else (active_db.get_subgoal(subgoal_id) if hasattr(active_db, "get_subgoal") else None)
            )
            if sg is None:
                raise KeyError(f"Unknown subgoal_id: {subgoal_id}")
            sg_hyps = active_db.list_hypotheses(goal_id=sg.goal_id, subgoal_id=sg.subgoal_id)
            sg_hids = {h.hypothesis_id for h in sg_hyps}
            sg_wfs = [w for w in active_db.list_workflows(goal_id=sg.goal_id) if w.hypothesis_id in sg_hids]
            sg_tasks = [t for t in active_db.list_tasks(goal_id=sg.goal_id) if t.hypothesis_id in sg_hids]
            sg_rcpts = [r for r in active_db.list_receipts(goal_id=sg.goal_id) if r.hypothesis_id in sg_hids]
            sg_decs = active_db.list_decision_requests(
                goal_id=sg.goal_id, subgoal_id=sg.subgoal_id, limit=30
            )
            out["subgoal"] = sg.to_dict()
            out["subgoals_summary"] = _build_subgoals_summary([sg])
            out["hypotheses"] = [h.to_dict() for h in sg_hyps]
            out["workflows"] = [w.to_dict() for w in sg_wfs]
            out["tasks"] = [t.to_dict() for t in sg_tasks[-50:]]
            out["recent_receipts"] = [r.to_dict() for r in sg_rcpts[-20:]]
            out["decision_requests"] = sg_decs
        elif goal_id is not None:
            goal = active_db.get_goal(goal_id)
            if goal is None:
                raise KeyError(f"Unknown goal_id: {goal_id}")
            goal_subgoals = (
                active_db.list_subgoals(goal_id=goal_id)
                if hasattr(active_db, "list_subgoals")
                else []
            )
            out["goal"] = goal.to_dict()
            out["status"] = active_db.get_goal_status(goal_id)
            out["subgoals"] = [sg.to_dict() for sg in goal_subgoals]
            out["subgoals_summary"] = _build_subgoals_summary(goal_subgoals)
            out["routing_policy"] = goal.routing_policy
            out["routing_policy_nl"] = goal.routing_policy_nl
            out["versions"] = active_db.list_goal_versions(goal_id)
            out["research_plans"] = [rp.to_dict() for rp in active_db.list_research_plans(goal_id)]
            out["experiment_plans"] = [ep.to_dict() for ep in active_db.list_experiment_plans(goal_id=goal_id)]
            out["execution_plans"] = [xp.to_dict() for xp in active_db.list_execution_plans(goal_id=goal_id)]
            out["review_receipts"] = [rv.to_dict() for rv in active_db.list_review_receipts(goal_id=goal_id)[-30:]]
            out["hypotheses"] = [h.to_dict() for h in active_db.list_hypotheses(goal_id=goal_id)]
            out["workflows"] = [w.to_dict() for w in active_db.list_workflows(goal_id=goal_id)]
            out["tasks"] = [t.to_dict() for t in active_db.list_tasks(goal_id=goal_id)[-50:]]
            out["recent_receipts"] = [r.to_dict() for r in active_db.list_receipts(goal_id=goal_id)[-20:]]
            out["knowledge"] = active_db.list_knowledge_items(goal_id=goal_id, limit=50)
            out["decision_requests"] = active_db.list_decision_requests(goal_id=goal_id, limit=30)
            out["interventions"] = active_db.list_interventions(goal_id=goal_id, limit=30)
            out["budget"] = active_db.get_goal_token_usage(goal_id)
        else:
            all_subgoals = (
                active_db.list_subgoals() if hasattr(active_db, "list_subgoals") else []
            )
            out["goals"] = [
                {"status": rec["status"], "goal": rec["goal"].to_dict()}
                for rec in active_db.list_goal_records()
            ]
            out["subgoals_summary"] = _build_subgoals_summary(all_subgoals)
            out["global_routing_policy"] = active_db.get_global_config("routing_policy")
            out["workflows"] = [w.to_dict() for w in active_db.list_workflows()[:50]]
            out["review_receipts"] = [rv.to_dict() for rv in active_db.list_review_receipts()[-30:]]
            out["decision_requests"] = active_db.list_decision_requests(limit=30)
            out["interventions"] = active_db.list_interventions(limit=30)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_latency_sample(home, "inspect", elapsed_ms, provider=provider, record_ping=False)
        out["interactive_latency_ms"] = round(elapsed_ms, 3)
        return out
    finally:
        if own_db:
            active_db.close()


def get_token_audit(
    arr_home: str | Path | None = None,
    *,
    goal_id: str | None = None,
    db: StateDB | None = None,
    provider: str = "interactive",
    record_ping: bool = True,
) -> dict[str, Any]:
    """Audit real uncached vs cache token consumption per Goal, stage, model, and validity."""
    t0 = time.perf_counter()
    home, active_db, own_db = _open_db(arr_home, db)
    try:
        if record_ping:
            record_foreground_ping(home, provider=provider, ttl_s=120.0)

        receipts = active_db.list_receipts(goal_id=goal_id)
        goals = (
            [g for g in [active_db.get_goal(goal_id)] if g is not None]
            if goal_id
            else active_db.list_goals()
        )
        by_goal: dict[str, dict[str, Any]] = {
            g.goal_id: {
                "goal_id": g.goal_id,
                "max_tokens_budget": g.max_tokens_budget,
                "uncached_in_tokens": 0,
                "uncached_out_tokens": 0,
                "uncached_total_tokens": 0,
                "cache_read_tokens": 0,
                "real_llm_calls": 0,
            }
            for g in goals
        }
        by_stage: dict[str, dict[str, int]] = {}
        by_model: dict[str, dict[str, int]] = {}
        by_validity: dict[str, int] = {}

        total_in = 0
        total_out = 0
        total_cache = 0
        real_calls = 0

        for r in receipts:
            in_t = int(r.in_tokens)
            out_t = int(r.out_tokens)
            cache_entry = (r.metrics or {}).get("cache_read_tokens")
            cache_t = int(cache_entry.get("value", 0)) if isinstance(cache_entry, Mapping) else 0
            is_real = r.model_used not in ("deterministic", "none", "unknown", "") and (
                in_t > 0 or out_t > 0
            )
            total_in += in_t
            total_out += out_t
            total_cache += cache_t
            if is_real:
                real_calls += 1

            gb = by_goal.setdefault(
                r.goal_id,
                {
                    "goal_id": r.goal_id,
                    "max_tokens_budget": 0,
                    "uncached_in_tokens": 0,
                    "uncached_out_tokens": 0,
                    "uncached_total_tokens": 0,
                    "cache_read_tokens": 0,
                    "real_llm_calls": 0,
                },
            )
            gb["uncached_in_tokens"] += in_t
            gb["uncached_out_tokens"] += out_t
            gb["uncached_total_tokens"] += in_t + out_t
            gb["cache_read_tokens"] += cache_t
            if is_real:
                gb["real_llm_calls"] += 1

            st_b = by_stage.setdefault(r.stage.value, {"in": 0, "out": 0, "cache": 0, "count": 0})
            st_b["in"] += in_t
            st_b["out"] += out_t
            st_b["cache"] += cache_t
            st_b["count"] += 1

            m_b = by_model.setdefault(r.model_used, {"in": 0, "out": 0, "cache": 0, "count": 0})
            m_b["in"] += in_t
            m_b["out"] += out_t
            m_b["cache"] += cache_t
            m_b["count"] += 1

            val_key = (
                r.evidence_validity.value
                if isinstance(r.evidence_validity, EvidenceValidity)
                else str(r.evidence_validity)
            )
            by_validity[val_key] = by_validity.get(val_key, 0) + 1

        for gb in by_goal.values():
            budget = int(gb["max_tokens_budget"])
            gb["remaining_tokens"] = max(0, budget - gb["uncached_total_tokens"]) if budget > 0 else None

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_latency_sample(home, "token-audit", elapsed_ms, provider=provider, record_ping=False)
        return {
            "arr_home": str(home),
            "goal_id": goal_id,
            "uncached_in_tokens": total_in,
            "uncached_out_tokens": total_out,
            "uncached_total_tokens": total_in + total_out,
            "cache_read_tokens": total_cache,
            "gross_tokens": total_in + total_out + total_cache,
            "real_llm_receipts": real_calls,
            "receipts_total": len(receipts),
            "by_goal": list(by_goal.values()),
            "by_stage": by_stage,
            "by_model": by_model,
            "by_validity": by_validity,
            "interactive_latency_ms": round(elapsed_ms, 3),
        }
    finally:
        if own_db:
            active_db.close()


def _truncate_utf8_lines(lines: list[str], max_bytes: int) -> str:
    """Trim lines cleanly from bottom until UTF-8 byte length <= max_bytes."""
    text = "\n".join(lines).rstrip() + "\n"
    if len(text.encode("utf-8")) <= max_bytes:
        return text
    kept: list[str] = []
    suffix = "\n... [truncated to stay <3KB]\n"
    budget = max(128, max_bytes - len(suffix.encode("utf-8")))
    used = 0
    for ln in lines:
        ln_bytes = len((ln + "\n").encode("utf-8"))
        if used + ln_bytes > budget:
            break
        kept.append(ln)
        used += ln_bytes
    if not kept:
        raw = text.encode("utf-8")[:budget].decode("utf-8", errors="ignore")
        return raw + suffix
    return "\n".join(kept).rstrip() + suffix


def build_takeover_brief(
    arr_home: str | Path | None = None,
    *,
    goal_id: str | None = None,
    max_bytes: int = DEFAULT_TAKEOVER_MAX_BYTES,
    db: StateDB | None = None,
    provider: str = "interactive",
    record_ping: bool = True,
) -> dict[str, Any]:
    """Generate a session-independent <3KB takeover brief from SQLite WAL for any fresh agent."""
    t0 = time.perf_counter()
    home, active_db, own_db = _open_db(arr_home, db)
    try:
        if record_ping:
            record_foreground_ping(home, provider=provider, ttl_s=120.0)

        goal_records = active_db.list_goal_records()
        if goal_id is not None:
            goal_records = [r for r in goal_records if r["goal"].goal_id == goal_id]

        lines: list[str] = [
            "# Conjectra Takeover Brief",
            f"CONJECTRA_HOME: `{home}` | Generated: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
            "",
            "## Goals",
        ]
        if not goal_records:
            lines.append("- (no matching goals)")
        for rec in goal_records[:6]:
            g: ResearchGoal = rec["goal"]
            st = rec["status"]
            usage = active_db.get_goal_token_usage(g.goal_id)
            pol_tag = (
                f" | pol_rules={len((g.routing_policy or {}).get('rules') or [])}"
                if g.routing_policy
                else ""
            )
            rplan = active_db.get_latest_research_plan(g.goal_id)
            plan_tag = f" | rplan=v{rplan.plan_version}" if rplan is not None else ""
            lines.append(
                f"- **{g.goal_id}** [{st}] v{g.version}/e{g.human_epoch} "
                f"({g.autonomy_mode}, w={g.priority_weight:.1f}): "
                f"`{g.primary_metric}` {g.direction} base={g.baseline_value}"
                + (f" target={g.target_value}" if g.target_value is not None else "")
                + f" | tok={usage['total_tokens']}/{g.max_tokens_budget}{pol_tag}{plan_tag}"
            )
            if rplan is not None and rplan.thesis:
                lines.append(f"  - Thesis: {rplan.thesis[:90]}")

        subgoals = (
            active_db.list_subgoals(goal_id=goal_id)
            if hasattr(active_db, "list_subgoals")
            else []
        )
        subgoals_summary = _build_subgoals_summary(subgoals)
        if subgoals:
            c_sg = subgoals_summary["counts"]
            lines.extend(
                [
                    "",
                    f"## SubGoals (total={c_sg['total']} active={c_sg['active']} sealed={c_sg['sealed']} locked={c_sg['locked_dep']} awaiting={c_sg['awaiting_apex']})",
                ]
            )
            for sg_item in subgoals_summary["subgoals"][:6]:
                t_tag = (
                    f"->{sg_item['target_value']}"
                    if sg_item.get("target_value") is not None
                    else ""
                )
                sha_tag = f" sha={str(sg_item['sealed_sha'])[:8]}" if sg_item.get("sealed_sha") else ""
                lines.append(
                    f"- [{sg_item['status']}] `{sg_item['subgoal_id']}` ({sg_item['kind']}, w={sg_item['priority_weight']:.1f}) "
                    f"`{sg_item['local_metric']}{t_tag}`{sha_tag}: {str(sg_item['title'])[:48]}"
                )

        hyps = active_db.list_hypotheses(goal_id=goal_id)
        confirmed = [h for h in hyps if h.scientific_verdict == ScientificVerdict.CONFIRMED]
        active_h = [
            h
            for h in hyps
            if h.status not in ("REFUTED", "GRAVEYARD", "PRUNED", "COMPLETED", "VALID_NEGATIVE")
        ]
        refuted = [h for h in hyps if h.status in ("REFUTED", "GRAVEYARD", "VALID_NEGATIVE")]

        lines.extend(["", "## Branches & Findings"])
        for h in sorted(confirmed, key=lambda x: -x.priority_evoi)[:3]:
            lines.append(
                f"- [CONFIRMED] `{h.goal_id}/{h.hypothesis_id}` ({h.fidelity.value}): {h.title[:60]}"
            )
        for h in sorted(active_h, key=lambda x: -x.priority_evoi)[:4]:
            fd_note = (
                f" ({h.failure_domain.value})"
                if h.failure_domain != FailureDomain.NONE
                else ""
            )
            lines.append(
                f"- [{h.status}{fd_note}] `{h.goal_id}/{h.hypothesis_id}` evoi={h.priority_evoi:.2f}: {h.title[:54]}"
            )
        if refuted:
            lines.append(
                "- Refuted/Graveyard: "
                + ", ".join(f"{h.hypothesis_id}({h.mechanism[:24]})" for h in refuted[:4])
            )

        wfs = active_db.list_workflows(goal_id=goal_id)
        active_wfs = [w for w in wfs if w.status in ("READY", "RUNNING", "PAUSED")]
        if active_wfs:
            lines.extend(["", "## Workflows"])
            for w in active_wfs[:4]:
                lines.append(f"- `{w.workflow_id}` [{w.status}] {w.primitive} ({w.goal_id}/{w.hypothesis_id})")

        knowledge = active_db.list_knowledge_items(goal_id=goal_id, limit=6)
        if knowledge:
            lines.extend(["", "## Knowledge"])
            for k in knowledge[:4]:
                lines.append(f"- [{k['category']}] {k['key']} (v{k['goal_version']}): {str(k['content'])[:90]}")

        interventions = active_db.list_interventions(goal_id=goal_id, limit=4)
        pending_decs = active_db.list_decision_requests(goal_id=goal_id, status="PENDING", limit=4)
        escs = _get_owner_escalations(active_db, goal_id=goal_id)
        if interventions or pending_decs or escs:
            lines.extend(["", "## Control State"])
            for d in pending_decs[:3]:
                lines.append(f"- Pending decision `{d['req_id']}` ({d['tier']} on {d['goal_id']}): {d['trigger_kind']}")
            for e in escs[:3]:
                lines.append(f"- Owner gate `{e['gate']}` ({e['goal_id']}): {e['reason'][:70]}")
            for iv in interventions[:2]:
                lines.append(f"- Intervention `{iv['scope']}/{iv['action']}` on {iv.get('goal_id') or 'global'}")

        lines.extend(
            [
                "",
                "## Next Commands",
                f"- `conjectra --home {home} overview`",
                f"- `conjectra --home {home} inspect" + (f" --goal-id {goal_id}`" if goal_id else "`"),
                f"- `conjectra --home {home} intervene --scope goal --action ...`",
            ]
        )

        capped_max = min(int(max_bytes), 3071)
        md = _truncate_utf8_lines(lines, capped_max)
        byte_len = len(md.encode("utf-8"))
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_latency_sample(home, "takeover", elapsed_ms, provider=provider, record_ping=False)
        return {
            "arr_home": str(home),
            "goal_id": goal_id,
            "subgoals_summary": subgoals_summary,
            "brief_markdown": md,
            "brief_bytes": byte_len,
            "under_3kb": byte_len < 3072,
            "interactive_latency_ms": round(elapsed_ms, 3),
        }
    finally:
        if own_db:
            active_db.close()


def get_plans(
    arr_home: str | Path | None = None,
    *,
    goal_id: str | None = None,
    hypothesis_id: str | None = None,
    db: StateDB | None = None,
    provider: str = "interactive",
    record_ping: bool = True,
) -> dict[str, Any]:
    """Return sub-50ms versioned ResearchPlans, ExperimentPlans, and ExecutionPlans from SQLite WAL."""
    t0 = time.perf_counter()
    home, active_db, own_db = _open_db(arr_home, db)
    try:
        if record_ping:
            record_foreground_ping(home, provider=provider, ttl_s=120.0)
        goal_ids = (
            [goal_id]
            if goal_id
            else [g.goal_id for g in active_db.list_goals()]
        )
        rplans = []
        for gid in goal_ids:
            rplans.extend([rp.to_dict() for rp in active_db.list_research_plans(gid)])
        eplans = [
            ep.to_dict()
            for ep in active_db.list_experiment_plans(
                hypothesis_id=hypothesis_id, goal_id=goal_id
            )
        ]
        xplans = [
            xp.to_dict()
            for xp in active_db.list_execution_plans(
                goal_id=goal_id, hypothesis_id=hypothesis_id
            )
        ]
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_latency_sample(home, "plans", elapsed_ms, provider=provider, record_ping=False)
        return {
            "arr_home": str(home),
            "goal_id": goal_id,
            "hypothesis_id": hypothesis_id,
            "research_plans": rplans,
            "experiment_plans": eplans,
            "execution_plans": xplans,
            "interactive_latency_ms": round(elapsed_ms, 3),
        }
    finally:
        if own_db:
            active_db.close()


def get_reviews(
    arr_home: str | Path | None = None,
    *,
    goal_id: str | None = None,
    hypothesis_id: str | None = None,
    workflow_id: str | None = None,
    review_kind: ReviewKind | str | None = None,
    db: StateDB | None = None,
    provider: str = "interactive",
    record_ping: bool = True,
) -> dict[str, Any]:
    """Return sub-50ms 3-way Design/Impl/Result ReviewReceipts & active review quorums from SQLite WAL."""
    t0 = time.perf_counter()
    home, active_db, own_db = _open_db(arr_home, db)
    try:
        if record_ping:
            record_foreground_ping(home, provider=provider, ttl_s=120.0)
        receipts = active_db.list_review_receipts(
            goal_id=goal_id,
            hypothesis_id=hypothesis_id,
            workflow_id=workflow_id,
            review_kind=review_kind,
        )
        wfs = (
            [w for w in [active_db.get_workflow(workflow_id)] if w is not None]
            if workflow_id
            else active_db.list_workflows(goal_id=goal_id, hypothesis_id=hypothesis_id)
        )
        quorums = [
            {
                "workflow_id": w.workflow_id,
                "goal_id": w.goal_id,
                "hypothesis_id": w.hypothesis_id,
                "status": w.status,
                "review_quorum": w.state.get("review_quorum"),
            }
            for w in wfs
            if isinstance(w.state.get("review_quorum"), Mapping)
        ]
        by_kind: dict[str, int] = {}
        for rv in receipts:
            k = rv.review_kind.value
            by_kind[k] = by_kind.get(k, 0) + 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_latency_sample(home, "reviews", elapsed_ms, provider=provider, record_ping=False)
        return {
            "arr_home": str(home),
            "goal_id": goal_id,
            "hypothesis_id": hypothesis_id,
            "workflow_id": workflow_id,
            "counts_by_kind": by_kind,
            "review_receipts": [rv.to_dict() for rv in receipts],
            "active_quorums": quorums,
            "interactive_latency_ms": round(elapsed_ms, 3),
        }
    finally:
        if own_db:
            active_db.close()


def intervene(
    arr_home: str | Path | None = None,
    *,
    scope: str,
    action: str,
    goal_id: str | None = None,
    target_id: str | None = None,
    payload: Mapping[str, Any] | None = None,
    actor: str = "user",
    reason: str = "",
    wake_reactor: bool = True,
    db: StateDB | None = None,
    hq: Any = None,
    provider: str = "interactive",
    record_ping: bool = True,
) -> dict[str, Any]:
    """Apply an atomic human/agent intervention via GlobalGoalManager and wake reactor.sock."""
    from conjectra.controller import send_reactor_wake
    from conjectra.goal_manager import GlobalGoalManager

    t0 = time.perf_counter()
    home, active_db, own_db = _open_db(arr_home, db)
    try:
        if record_ping:
            record_foreground_ping(home, provider=provider, ttl_s=120.0)
        eff_hq = hq
        if eff_hq is None and (home / "hq-server").exists():
            try:
                from conjectra.hq_runtime import HQRuntime

                eff_hq = HQRuntime(home)
            except Exception:
                eff_hq = None
        mgr = GlobalGoalManager(active_db, hq=eff_hq)
        result = mgr.apply_intervention(
            action,
            scope=scope,
            goal_id=goal_id,
            target_id=target_id,
            payload=dict(payload or {}),
            actor=actor,
            reason=reason,
        )
        woke = False
        if wake_reactor:
            woke = send_reactor_wake(
                home,
                {
                    "event": "HUMAN_INTERVENTION",
                    "scope": scope,
                    "action": action,
                    "goal_id": goal_id,
                    "target_id": target_id,
                    "ts": time.time(),
                },
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_latency_sample(home, "intervene", elapsed_ms, provider=provider, record_ping=False)
        return {
            "arr_home": str(home),
            "scope": scope,
            "action": action,
            "goal_id": goal_id,
            "target_id": target_id,
            "result": result,
            "reactor_woke": woke,
            "interactive_latency_ms": round(elapsed_ms, 3),
        }
    finally:
        if own_db:
            active_db.close()


def get_subgoals(
    arr_home: str | Path | None = None,
    *,
    goal_id: str | None = None,
    status: str | None = None,
    db: StateDB | None = None,
    provider: str = "interactive",
    record_ping: bool = True,
) -> dict[str, Any]:
    """Return sub-50ms SubGoal portfolio list and summary counts from SQLite WAL."""
    t0 = time.perf_counter()
    home, active_db, own_db = _open_db(arr_home, db)
    try:
        if record_ping:
            record_foreground_ping(home, provider=provider, ttl_s=120.0)
        eff_status = status.strip().upper() if isinstance(status, str) and status.strip() else None
        subgoals = (
            active_db.list_subgoals(goal_id=goal_id, status=eff_status)
            if hasattr(active_db, "list_subgoals")
            else []
        )
        summary = _build_subgoals_summary(subgoals)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        _record_latency_sample(home, "subgoals", elapsed_ms, provider=provider, record_ping=False)
        return {
            "arr_home": str(home),
            "goal_id": goal_id,
            "status": status,
            "counts": summary["counts"],
            "subgoals": [sg.to_dict() for sg in subgoals],
            "subgoals_summary": summary,
            "interactive_latency_ms": round(elapsed_ms, 3),
        }
    finally:
        if own_db:
            active_db.close()


def get_status_snapshot(
    arr_home: str | Path | None = None,
    *,
    db: StateDB | None = None,
    write_file: bool = True,
) -> dict[str, Any]:
    """Return dashboard-compatible snapshot dict (optionally atomically persisting `<arr_home>/status_snapshot.json`)."""
    from conjectra.status_snapshot import build_status_snapshot, export_status_snapshot

    home, active_db, own_db = _open_db(arr_home, db)
    try:
        if write_file:
            return export_status_snapshot(active_db, arr_home=home)
        return build_status_snapshot(active_db, arr_home=home)
    finally:
        if own_db:
            active_db.close()
