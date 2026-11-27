"""Read-only live status snapshot (`<home>/status_snapshot.json`) for dashboards and status bars."""

from __future__ import annotations

import json
import os
import time
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from conjectra.metrics import compute_per_goal_empirical_efficiency
from conjectra.state_db import TERMINAL_TASK_SUCCESS, StateDB, resolve_arr_home
from conjectra.types import ResearchGoal

RECENT_RUN_WINDOW_S = 1800.0
_TERMINAL_RECENT_STATUSES = TERMINAL_TASK_SUCCESS | frozenset(
    {"FAILED", "CANCELLED", "DEAD", "TIMED_OUT", "LEASE_LOST"}
)


def _is_canary_goal(goal_id: str, title: str = "") -> bool:
    return "canary" in str(goal_id or "").lower() or "canary" in str(title or "").lower()


def _proc_pid_alive(pid: int, expected_start: Any = None) -> bool:
    if pid <= 1:
        return False
    stat_path = Path(f"/proc/{pid}/stat")
    if stat_path.is_file():
        try:
            raw = stat_path.read_text(encoding="utf-8", errors="ignore")
            rp = raw.rfind(")")
            fields = raw[rp + 2 :].split() if rp >= 0 else []
            if fields and fields[0] == "Z":
                return False
            if isinstance(expected_start, (list, tuple)) and len(expected_start) >= 2 and len(fields) > 19:
                return str(fields[19]) == str(expected_start[1])
            return True
        except OSError:
            return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _check_hub_alive(home: Path) -> bool:
    """Verify multi-goal controller PID, reactor.sock, and local HQ server metadata without subprocesses."""
    pid_file = home / "controller.pid"
    hq_dir = home / "hq-server"
    if not ((home / "reactor.sock").exists() and pid_file.is_file() and hq_dir.is_dir() and any(hq_dir.iterdir())):
        return False
    try:
        pid_doc = json.loads(pid_file.read_text(encoding="utf-8"))
        return _proc_pid_alive(int(pid_doc.get("pid") or 0), pid_doc.get("proc_start_time"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def _route_fields(pinned: Mapping[str, Any] | None, model_spec: str, llm_pool: str) -> tuple[str, str, str]:
    p = pinned if isinstance(pinned, Mapping) else {}
    pool = str(p.get("pool") or p.get("family") or "").strip()
    model = str(p.get("canonical_model") or model_spec or llm_pool or "").strip()
    effort = str(p.get("wire_effort") or p.get("requested_effort") or "").strip()
    if not pool and ":" in model:
        pool = model.split(":", 1)[0].strip()
    if not pool:
        lm = model.lower()
        for kws, pname in (
            (("opus", "sonnet", "claude"), "claude"),
            (("gpt-", "codex"), "codex"),
            (("gemini",), "gemini"),
            (("deepseek",), "deepseek"),
            (("grok",), "grok"),
        ):
            if any(k in lm for k in kws):
                pool = pname
                break
    return pool or "conjectra", model, effort


def _task_item(
    item_id: str, agent: str, model: str, effort: str, project: str, title: str,
    cwd: str, status: str, started: float, last_out: float, step: str, cmd: str,
    *, tail: str = "", exit_code: int | None = None,
) -> dict[str, Any]:
    return {
        "id": item_id, "kind": "conjectra", "agent": agent, "model": model, "effort": effort,
        "project": project, "title": title, "cwd": cwd, "status": status,
        "started_at": started, "last_output_at": last_out, "out_bytes": 0,
        "child_step": step, "log_path": None, "log_tail": tail, "command": cmd, "exit_code": exit_code,
    }


def _list_latest_events(db: StateDB, *, limit: int = 500) -> list[dict[str, Any]]:
    """Fetch the latest `limit` events ordered oldest-to-newest within the tail window."""
    conn = getattr(db, "_conn", None)
    lock = getattr(db, "_lock", None)
    if conn is not None and lock is not None:
        with lock:
            rows = conn.execute(
                "SELECT seq, event_id, event_type, goal_id, hypothesis_id, task_id, ts, payload_json "
                "FROM events ORDER BY seq DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in reversed(rows):
            item = dict(r)
            item["payload"] = json.loads(item.pop("payload_json"))
            out.append(item)
        return out

    cursor = 0
    collected: list[dict[str, Any]] = []
    while True:
        page = db.list_events(since_seq=cursor, limit=limit)
        if not page:
            break
        collected.extend(page)
        last_seq = int(page[-1].get("seq") or 0)
        if len(page) < limit or last_seq <= cursor:
            break
        cursor = last_seq
    return collected[-limit:]


def build_status_snapshot(
    db: StateDB, *, arr_home: str | Path | None = None, now: float | None = None,
) -> dict[str, Any]:
    """Build a dashboard `RemoteSchedSummary`-compatible payload from SQLite WAL."""
    ts_now = time.time() if now is None else float(now)
    home = resolve_arr_home(arr_home if arr_home is not None else db.arr_home)
    goal_records = db.list_goal_records()
    active_goals: dict[str, ResearchGoal] = {
        rec["goal"].goal_id: rec["goal"]
        for rec in goal_records
        if str(rec.get("status") or "") == "ACTIVE" and not _is_canary_goal(rec["goal"].goal_id, rec["goal"].title)
    }
    tasks = db.list_tasks()
    receipts = db.list_receipts()
    reviews = db.list_review_receipts()
    decisions = db.list_decision_requests(limit=200)
    hyps = db.list_hypotheses()
    events = _list_latest_events(db, limit=500)
    eff = compute_per_goal_empirical_efficiency(
        db,
        goal_records=[rec for rec in goal_records if rec["goal"].goal_id in active_goals],
        receipts=receipts,
        reviews=reviews,
    )
    hub_alive = _check_hub_alive(home)

    active_runs: list[dict[str, Any]] = []
    for t in tasks:
        if t.goal_id in active_goals and t.status in ("DISPATCHED", "RUNNING"):
            g = active_goals[t.goal_id]
            agent, model, effort = _route_fields(t.pinned_route, t.model_spec, t.resources.llm_pool)
            started = float(t.started_ts or t.enqueued_ts or ts_now)
            active_runs.append(_task_item(
                f"conjectra:{t.task_id}", agent, model, effort, g.project,
                f"{t.stage.value} · {t.step_key or t.hypothesis_id[:12]}",
                str(t.worktree_path or g.repo_path or ""), "running", started, started,
                f"{t.stage.value} a{int(t.attempt)}", f"conjectra {t.stage.value} {t.hypothesis_id}",
            ))
    for d in decisions:
        gid, st = str(d.get("goal_id") or ""), str(d.get("status") or "").upper()
        if gid in active_goals and st in ("SUBMITTING", "RUNNING"):
            g = active_goals[gid]
            payload = d.get("payload") if isinstance(d.get("payload"), Mapping) else {}
            agent, model, effort = _route_fields(payload.get("pinned_route"), str(d.get("model_spec") or ""), "")
            started = float(d.get("updated_ts") or d.get("created_ts") or ts_now)
            active_runs.append(_task_item(
                f"conjectra:{d['req_id']}", agent, model, effort, g.project,
                f"DEC:{d.get('trigger_kind', 'LEAD')} ({d.get('tier', 'L1')})",
                str(g.repo_path or ""), "running", started, started,
                f"{d.get('role') or 'research_lead'} {st}", f"conjectra decision {d.get('trigger_kind', '')}".strip(),
            ))

    rcpt_by_task = {r.task_id: r for r in receipts}
    recent_runs: list[dict[str, Any]] = []
    for t in tasks:
        fin_ts = float(t.finished_ts or 0.0)
        if t.goal_id not in active_goals or t.status not in _TERMINAL_RECENT_STATUSES or fin_ts <= 0 or (ts_now - fin_ts) > RECENT_RUN_WINDOW_S:
            continue
        g, r = active_goals[t.goal_id], rcpt_by_task.get(t.task_id)
        agent, model, effort = _route_fields(t.pinned_route, (r.model_used if r else "") or t.model_spec, t.resources.llm_pool)
        ok = t.status in ("DONE", "SUCCEEDED", "PASSED")
        tail = (r.failure_reason if (r and r.failure_reason) else (r.engineering_verdict.value if r else t.status))[:200]
        recent_runs.append(_task_item(
            f"conjectra:{t.task_id}", agent, model, effort, g.project,
            f"{t.stage.value} · {t.step_key or t.hypothesis_id[:12]}",
            str(t.worktree_path or g.repo_path or ""), "done" if ok else "failed",
            float(t.started_ts or t.enqueued_ts or fin_ts), fin_ts, f"{t.stage.value} {t.status}",
            f"conjectra {t.stage.value} {t.hypothesis_id}",
            tail=tail, exit_code=int(r.exit_code) if r is not None else (0 if ok else 1),
        ))
    recent_runs.sort(key=lambda item: float(item.get("last_output_at") or 0.0), reverse=True)

    hyp_by_id = {h.hypothesis_id: h for h in hyps}
    usage_by_goal = {gid: db.get_goal_token_usage(gid) for gid in active_goals}
    open_asks: list[dict[str, str]] = []
    seen_asks: set[str] = set()
    resolved_goals: set[str] = set()
    resolved_hyps: set[str] = set()
    for ev in reversed(events):
        etype = str(ev.get("event_type") or "")
        gid = str(ev.get("goal_id") or "")
        hid = str(ev.get("hypothesis_id") or "")
        p = ev.get("payload") if isinstance(ev.get("payload"), Mapping) else {}
        if etype in ("HUMAN_INTERVENTION", "MAINTENANCE_STATE_HEAL") and gid:
            resolved_goals.add(gid)
            continue
        if etype == "GOAL_STATUS" and gid and str(p.get("status") or "").upper() == "ACTIVE":
            resolved_goals.add(gid)
            continue
        if etype in ("UNBLOCK_STALLED_EXPLORATION", "HYPOTHESIS_GRAVEYARDED") and hid:
            resolved_hyps.add(hid)
            continue
        if etype == "CONTROLLER_DECISION_APPLIED":
            act = str(p.get("action") or "").upper()
            if act in ("COMPLETE_MILESTONE", "REBALANCE", "STOP_GOAL") and gid:
                resolved_goals.add(gid)
            elif act in ("PIVOT_ROUTE", "TERMINATE_REFUTED_ROUTE") and hid:
                resolved_hyps.add(hid)
            continue
        if etype != "OWNER_ESCALATION" or gid not in active_goals:
            continue
        if gid in resolved_goals:
            continue
        if hid and (
            hid in resolved_hyps
            or (
                hid in hyp_by_id
                and str(hyp_by_id[hid].status) in ("CONFIRMED", "COMPLETED", "GRAVEYARD", "REFUTED", "ARCHIVED")
            )
        ):
            continue
        gate = str(p.get("gate") or "OWNER_GATE")
        reason_str = str(p.get("reason") or "Requires owner input")
        if ("TOKEN_BUDGET" in reason_str or "token budget" in reason_str.lower() or gate == "WAIT_BUDGET"):
            g_obj = active_goals[gid]
            if g_obj.max_tokens_budget > 0 and int(usage_by_goal[gid].get("total_tokens", 0)) < int(g_obj.max_tokens_budget):
                continue
        key = f"esc:{gid}:{gate}"
        if key not in seen_asks:
            seen_asks.add(key)
            reason = reason_str[:140]
            open_asks.append({"id": key, "project": active_goals[gid].project, "line": gid, "category": gate.lower(),
                              "question": f"{gate}: {reason} | A (recommended): approve via intervene / raise budget; B: pause this branch"})
    for d in decisions:
        gid, trig = str(d.get("goal_id") or ""), str(d.get("trigger_kind") or "")
        if gid in active_goals and str(d.get("status") or "").upper() == "PENDING" and trig in ("REQUEST_DIRECTOR_ADJUDICATION", "OWNER_ESCALATION", "WAIT_BUDGET"):
            aid = f"dec:{d['req_id']}"
            if aid not in seen_asks:
                seen_asks.add(aid)
                open_asks.append({"id": aid, "project": active_goals[gid].project, "line": gid,
                                  "category": "director_adjudication" if "DIRECTOR" in trig else "budget",
                                  "question": f"{trig} needs confirmation ({gid}) | A (recommended): continue on the default route; B: pause this goal"})
    for gid, g in active_goals.items():
        usage = db.get_goal_token_usage(gid)
        if g.max_tokens_budget > 0 and int(usage.get("total_tokens", 0)) >= int(g.max_tokens_budget) and f"budget:{gid}" not in seen_asks:
            seen_asks.add(f"budget:{gid}")
            open_asks.append({"id": f"budget:{gid}", "project": g.project, "line": gid, "category": "budget",
                              "question": f"{gid} hit its token budget ({usage.get('total_tokens')}/{g.max_tokens_budget}) | A (recommended): rebalance_goals to raise it; B: keep the current cap"})

    warnings: list[str] = [] if hub_alive else ["conjectra daemon offline or reactor.sock missing"]
    seen_warn: set[tuple[str, str]] = set()
    for h in hyps:
        if h.goal_id in active_goals and h.status in ("BLOCKED_ENV", "BLOCKED_INFRA", "BLOCKED_QUOTA", "BLOCKED_POLICY", "BLOCKED_REVIEW_QUORUM"):
            if (h.goal_id, h.status) not in seen_warn:
                seen_warn.add((h.goal_id, h.status))
                warnings.append(f"{active_goals[h.goal_id].project} {h.goal_id}: {h.status}")
    for gid, row in eff.items():
        oh = row.get("framework_overhead_token_share_pct")
        if isinstance(oh, (int, float)) and float(oh) > 75.0:
            warnings.append(f"{row['project']} ({gid}) overhead_share={oh}% > 75%")

    tot_concl = sum(int(row.get("verified_empirical_conclusions") or 0) for row in eff.values())
    tot_tok = sum(int(row.get("total_uncached_tokens") or 0) for row in eff.values())
    tot_opus = sum(int(row.get("opus_tokens") or 0) for row in eff.values())
    opus_pct = f"{round(100.0 * tot_opus / tot_tok, 1)}%" if tot_tok > 0 else "0.0%"
    return {
        "present": True, "hub_alive": hub_alive, "sampled_at": ts_now, "state_age_s": 0, "state_age_sec": 0,
        "budget_text": f"Conjectra: {len(active_goals)} goals running · {tot_concl} verified conclusions · Opus {opus_pct}",
        "active_runs": active_runs[:16], "recent_runs": recent_runs[:8],
        "open_asks": open_asks[:8], "warnings": warnings[:8], "goals_efficiency": eff,
    }


def export_status_snapshot(db: StateDB, *, arr_home: str | Path | None = None, now: float | None = None) -> dict[str, Any]:
    """Build and atomically persist `<arr_home>/status_snapshot.json` (0 LLM tokens, tmp -> os.replace)."""
    home = resolve_arr_home(arr_home if arr_home is not None else db.arr_home)
    home.mkdir(parents=True, exist_ok=True)
    snap = build_status_snapshot(db, arr_home=home, now=now)
    out_path = home / "status_snapshot.json"
    tmp_path = home / f"status_snapshot.json.tmp.{os.getpid()}.{uuid.uuid4().hex[:8]}"
    tmp_path.write_text(json.dumps(snap, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_path, out_path)
    return snap
