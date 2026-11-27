"""Global multi-goal lifecycle, versioning, human-in-the-loop interventions, and epoch isolation."""

from __future__ import annotations

import json
import math
import re
import socket
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from conjectra import gates
from conjectra.gates import (
    check_no_banned_access,
    classify_failure_domain,
    is_valid_empirical_outcome,
    validate_writable_paths,
)
from conjectra.hq_runtime import HQRuntime
from conjectra.model_router import ModelRouter, PolicyBlockedError
from conjectra.state_db import StateDB, StateTransitionError, compute_hypothesis_fingerprint
from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    ExperimentPlan,
    FailureDomain,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ResearchPlan,
    ResourceSpec,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
    SubGoalKind,
    SubGoalNode,
    SubGoalStatus,
    WorkflowNode,
)

DEFAULT_GLOBAL_MAX_TOKENS_BUDGET = 2_000_000
DEFAULT_FOREGROUND_RESERVE_SLOTS = 1

VALID_AUTONOMY_MODES = frozenset({"unattended", "supervised", "paused", "manual"})
MINIMIZE_DIRECTIONS = frozenset({"min", "minimize", "lower", "down"})
MAXIMIZE_DIRECTIONS = frozenset({"max", "maximize", "higher", "up"})
VALID_DIRECTIONS = MINIMIZE_DIRECTIONS | MAXIMIZE_DIRECTIONS

ACTIVE_TASK_STATUSES = ("READY", "DISPATCHED", "RUNNING")
TERMINAL_HYP_STATUSES = frozenset(
    {"REFUTED", "PRUNED", "CANCELLED", "GRAVEYARD", "HUMAN_INVALIDATED"}
)
ACTIVE_WORKFLOW_STATUSES = frozenset({"READY", "RUNNING", "WAITING", "PAUSED"})

OBJECTIVE_FIELDS = frozenset(
    {
        "primary_metric",
        "direction",
        "baseline_value",
        "target_value",
        "kill_criteria",
        "baseline_sha",
    }
)
SCOPE_FIELDS = frozenset({"writable_prefixes", "banned_paths"})
RESOURCE_FIELDS = frozenset(
    {
        "max_tokens_budget",
        "max_concurrent_branches",
        "priority_weight",
        "min_guaranteed_slots",
        "wall_budget_sec",
        "autonomy_mode",
    }
)

_KILL_EXPR_RE = re.compile(
    r"^\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(<=|>=|==|!=|<|>)\s*([A-Za-z0-9_.-]+)\s*$"
)


def _is_finite_number(val: Any) -> bool:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def _normalize_direction(direction: str) -> str:
    d = str(direction or "").strip().lower()
    if d not in VALID_DIRECTIONS:
        raise ValueError(
            f"Invalid direction {direction!r}; expected one of {sorted(VALID_DIRECTIONS)}"
        )
    return "minimize" if d in MINIMIZE_DIRECTIONS else "maximize"


def _validate_goal_fields(goal: ResearchGoal) -> None:
    if not str(goal.goal_id).strip():
        raise ValueError("goal_id must be non-empty")
    if not str(goal.primary_metric).strip():
        raise ValueError("primary_metric must be non-empty")
    _normalize_direction(goal.direction)
    if not _is_finite_number(goal.baseline_value):
        raise ValueError(f"baseline_value must be finite, got {goal.baseline_value!r}")
    if goal.target_value is not None and not _is_finite_number(goal.target_value):
        raise ValueError(f"target_value must be finite, got {goal.target_value!r}")
    if int(goal.max_tokens_budget) < 0:
        raise ValueError("max_tokens_budget must be >= 0")
    if int(goal.max_concurrent_branches) < 1:
        raise ValueError("max_concurrent_branches must be >= 1")
    if not _is_finite_number(goal.priority_weight) or float(goal.priority_weight) <= 0:
        raise ValueError("priority_weight must be finite and > 0")
    if int(goal.min_guaranteed_slots) < 0:
        raise ValueError("min_guaranteed_slots must be >= 0")
    if int(goal.wall_budget_sec) <= 0:
        raise ValueError("wall_budget_sec must be > 0")
    if goal.autonomy_mode not in VALID_AUTONOMY_MODES:
        raise ValueError(
            f"Invalid autonomy_mode {goal.autonomy_mode!r}; expected one of {sorted(VALID_AUTONOMY_MODES)}"
        )


def _flatten_metrics(metrics: Mapping[str, Any]) -> dict[str, float | int | bool]:
    flat: dict[str, float | int | bool] = {}
    for k, v in (metrics or {}).items():
        raw = v.get("value") if isinstance(v, Mapping) and "value" in v else v
        if isinstance(raw, bool):
            flat[str(k)] = raw
        elif isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            flat[str(k)] = float(raw)
    return flat


def _evaluate_kill_criteria_local(
    kill_criteria: Sequence[str],
    flat_values: Mapping[str, float | int | bool],
) -> list[str]:
    triggered: list[str] = []
    for crit in kill_criteria:
        m = _KILL_EXPR_RE.match(str(crit or ""))
        if not m:
            continue
        key, op, raw_rhs = m.groups()
        if key not in flat_values:
            continue
        lhs = flat_values[key]
        if raw_rhs.lower() in ("true", "false"):
            rhs: Any = raw_rhs.lower() == "true"
        else:
            try:
                rhs = float(raw_rhs)
            except ValueError:
                rhs = raw_rhs
        try:
            hit = False
            if op == ">":
                hit = lhs > rhs
            elif op == ">=":
                hit = lhs >= rhs
            elif op == "<":
                hit = lhs < rhs
            elif op == "<=":
                hit = lhs <= rhs
            elif op == "==":
                hit = lhs == rhs
            elif op == "!=":
                hit = lhs != rhs
            if hit:
                triggered.append(str(crit))
        except TypeError:
            continue
    return triggered


def _notify_reactor(arr_home: Path, payload: Mapping[str, Any]) -> bool:
    sock_path = arr_home / "reactor.sock"
    if not sock_path.exists():
        return False
    raw = (json.dumps(dict(payload), sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    for sock_type in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        try:
            with socket.socket(socket.AF_UNIX, sock_type) as sock:
                sock.settimeout(1.0)
                sock.connect(str(sock_path))
                sock.sendall(raw)
            return True
        except OSError:
            continue
    return False


class GlobalGoalManager:
    """Manages global configuration, multi-goal lifecycles, versioning, HITL interventions, and epoch isolation."""

    def __init__(
        self,
        db: StateDB,
        hq: HQRuntime | None = None,
        *,
        notify_reactor: bool = True,
    ) -> None:
        self.db = db
        if hq is None and (db.arr_home / "hq-server").exists():
            try:
                self.hq: HQRuntime | None = HQRuntime(db.arr_home)
            except Exception:
                self.hq = None
        else:
            self.hq = hq
        self.notify_reactor = notify_reactor

    @staticmethod
    def _query_hq_job_state(hq_client: Any, jid: int | None) -> dict[str, Any]:
        """Return typed `{"lookup_status": "FOUND"|"NOT_FOUND"|"QUERY_ERROR", "job": dict|None, "error": str}`."""
        if jid is None:
            return {"lookup_status": "NOT_FOUND", "job": None, "error": ""}
        if hq_client is None:
            return {"lookup_status": "QUERY_ERROR", "job": None, "error": "hq_client_unavailable"}
        if hasattr(hq_client, "query_job_state"):
            try:
                res = hq_client.query_job_state(int(jid))
                if isinstance(res, Mapping) and str(res.get("lookup_status") or "").upper() in (
                    "FOUND",
                    "NOT_FOUND",
                    "QUERY_ERROR",
                ):
                    return {
                        "lookup_status": str(res["lookup_status"]).upper(),
                        "job": dict(res["job"]) if isinstance(res.get("job"), Mapping) else None,
                        "error": str(res.get("error") or ""),
                    }
                return {"lookup_status": "QUERY_ERROR", "job": None, "error": "invalid_query_job_state_payload"}
            except Exception as exc:
                return {"lookup_status": "QUERY_ERROR", "job": None, "error": str(exc)}
        if bool(getattr(hq_client, "simulate_query_error", False)):
            return {"lookup_status": "QUERY_ERROR", "job": None, "error": "simulated_query_error"}
        if not hasattr(hq_client, "job_info"):
            return {"lookup_status": "QUERY_ERROR", "job": None, "error": "job_info_unsupported"}
        try:
            info = hq_client.job_info(int(jid))
        except Exception as exc:
            return {"lookup_status": "QUERY_ERROR", "job": None, "error": str(exc)}
        if info is None:
            return {"lookup_status": "NOT_FOUND", "job": None, "error": ""}
        if isinstance(info, Mapping):
            return {"lookup_status": "FOUND", "job": dict(info), "error": ""}
        return {"lookup_status": "QUERY_ERROR", "job": None, "error": f"unexpected_job_info:{type(info).__name__}"}

    @staticmethod
    def _query_hq_job_list(hq_client: Any, *, all_jobs: bool = True) -> dict[str, Any]:
        """Return typed `{"lookup_status": "OK"|"QUERY_ERROR", "jobs": list[dict], "error": str}`."""
        if hq_client is None:
            return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": "hq_client_unavailable"}
        if bool(getattr(hq_client, "simulate_job_list_error", False)):
            return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": "simulated_job_list_error"}
        if hasattr(hq_client, "query_job_list"):
            try:
                res = hq_client.query_job_list(all_jobs=all_jobs)
                if (
                    isinstance(res, Mapping)
                    and str(res.get("lookup_status") or "").upper() in ("OK", "QUERY_ERROR")
                    and isinstance(res.get("jobs"), list)
                ):
                    return {
                        "lookup_status": str(res["lookup_status"]).upper(),
                        "jobs": [dict(j) for j in res["jobs"] if isinstance(j, Mapping)],
                        "error": str(res.get("error") or ""),
                    }
                return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": "invalid_query_job_list_payload"}
            except Exception as exc:
                return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": str(exc)}
        if type(hq_client).__name__ == "HQRuntime":
            import shutil as _shutil_hq
            import subprocess as _subp_hq

            hq_bin = getattr(hq_client, "hq_bin", None) or _shutil_hq.which("hq")
            if not hq_bin:
                return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": "hq_binary_missing"}
            server_dir = getattr(hq_client, "server_dir", None)
            cmd = [str(hq_bin)]
            if server_dir:
                cmd.extend(["--server-dir", str(server_dir)])
            cmd.extend(["--output-mode", "json", "job", "list"])
            if all_jobs:
                cmd.append("--all")
            try:
                proc = _subp_hq.run(cmd, capture_output=True, text=True, timeout=15, check=False)
            except Exception as exc:
                return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": str(exc)}
            out_s = (proc.stdout or "").strip()
            if proc.returncode != 0 or not out_s:
                return {
                    "lookup_status": "QUERY_ERROR",
                    "jobs": [],
                    "error": (proc.stderr or "").strip() or f"hq_job_list_rc_{proc.returncode}",
                }
            try:
                parsed = json.loads(out_s)
            except json.JSONDecodeError as exc:
                return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": f"malformed_json:{exc}"}
            if not isinstance(parsed, list):
                return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": "non_list_json"}
            return {
                "lookup_status": "OK",
                "jobs": [dict(j) for j in parsed if isinstance(j, Mapping)],
                "error": "",
            }
        if hasattr(hq_client, "job_list"):
            try:
                try:
                    jobs = hq_client.job_list(all_jobs=all_jobs)
                except TypeError:
                    jobs = hq_client.job_list()
                if isinstance(jobs, list):
                    return {
                        "lookup_status": "OK",
                        "jobs": [dict(j) for j in jobs if isinstance(j, Mapping)],
                        "error": "",
                    }
                return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": "stub_job_list_non_list"}
            except Exception as exc:
                return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": str(exc)}
        submitted = getattr(hq_client, "submitted", None)
        if isinstance(submitted, list):
            synth: list[dict[str, Any]] = []
            running_ids = getattr(hq_client, "running_job_ids", None)
            for item in submitted:
                if isinstance(item, Mapping) and item.get("job_id") is not None:
                    jid_i = int(item["job_id"])
                    is_run = isinstance(running_ids, set) and jid_i in running_ids
                    synth.append(
                        {
                            "id": jid_i,
                            "name": str(item.get("name") or item.get("task_id") or ""),
                            "state": "running" if is_run else "finished",
                            "info": {"task_stats": {"running": 1 if is_run else 0, "waiting": 0, "finished": 0 if is_run else 1}},
                        }
                    )
            return {"lookup_status": "OK", "jobs": synth, "error": ""}
        return {"lookup_status": "QUERY_ERROR", "jobs": [], "error": "job_list_unsupported"}

    @staticmethod
    def _hq_positively_terminated(
        hq_client: Any,
        jid: int | None,
        *,
        cancel_map: Mapping[int, bool] | None = None,
        prefetched_info: Mapping[str, Any] | None = None,
    ) -> bool:
        """Return True ONLY when HQ positively confirms terminal job state.

        Never treats `hq_client is None` or lookup errors on real HQRuntime as terminated.
        """
        if jid is None:
            return True
        if hq_client is None and prefetched_info is None:
            return False
        if hq_client is not None and hasattr(hq_client, "is_pending_canceled_or_dep_failed"):
            try:
                c_loc, d_fail = hq_client.is_pending_canceled_or_dep_failed(int(jid))
                if c_loc or d_fail:
                    return True
            except Exception:
                pass
        info: Mapping[str, Any] | None = prefetched_info if isinstance(prefetched_info, Mapping) else None
        if info is None:
            if hq_client is None or (not hasattr(hq_client, "job_info") and not hasattr(hq_client, "query_job_state")):
                return bool(isinstance(cancel_map, Mapping) and cancel_map.get(int(jid)) is True)
            q_state = GlobalGoalManager._query_hq_job_state(hq_client, int(jid))
            if q_state["lookup_status"] != "FOUND" or not isinstance(q_state.get("job"), Mapping):
                return False
            info = q_state["job"]
        jinfo = info.get("info")
        stats = jinfo.get("task_stats") if isinstance(jinfo, Mapping) else None
        if isinstance(stats, Mapping):
            if int(stats.get("running", 0) or 0) > 0 or int(stats.get("waiting", 0) or 0) > 0:
                return False
            term_cnt = (
                int(stats.get("canceled", 0) or 0)
                + int(stats.get("aborted", 0) or 0)
                + int(stats.get("failed", 0) or 0)
                + int(stats.get("finished", 0) or 0)
            )
            if term_cnt > 0 or info.get("finished_at") is not None:
                return True
        st = str(info.get("state") or info.get("status") or "").strip().lower()
        return st in ("canceled", "cancelled", "aborted", "failed", "finished", "completed", "done")

    # ------------------------------------------------------------ Global Config

    def get_global_config(self) -> dict[str, Any]:
        """Return effective global runtime config merged with defaults."""
        persisted = self.db.get_global_config() or {}
        cfg: dict[str, Any] = {
            "global_max_tokens_budget": DEFAULT_GLOBAL_MAX_TOKENS_BUDGET,
            "foreground_reserve_slots": DEFAULT_FOREGROUND_RESERVE_SLOTS,
        }
        cfg.update(persisted)
        return cfg

    def set_global_config(self, key: str, value: Any) -> dict[str, Any]:
        """Validate and persist a single global configuration key."""
        if key == "global_max_tokens_budget":
            if isinstance(value, bool) or int(value) <= 0:
                raise ValueError("global_max_tokens_budget must be a positive integer")
            value = int(value)
        elif key == "foreground_reserve_slots":
            if isinstance(value, bool) or int(value) < 0:
                raise ValueError("foreground_reserve_slots must be >= 0")
            value = int(value)
        return self.db.set_global_config(key, value)

    # ---------------------------------------------------------- Goal Lifecycle

    def create_or_update_goal(
        self,
        goal: ResearchGoal | Mapping[str, Any],
        *,
        status: str | None = None,
        actor: str = "user",
        reason: str = "",
        change_summary: str = "",
    ) -> ResearchGoal:
        """Create a new ResearchGoal or update an existing one with version tracking."""
        cand = (
            goal
            if isinstance(goal, ResearchGoal)
            else ResearchGoal.from_dict(dict(goal))
        )
        _validate_goal_fields(cand)

        existing = self.db.get_goal(cand.goal_id)
        if existing is None:
            stored = ResearchGoal.from_dict(
                {
                    **cand.to_dict(),
                    "version": max(1, int(cand.version)),
                    "human_epoch": max(0, int(cand.human_epoch)),
                }
            )
            self.db.save_goal(stored, status=status or "ACTIVE")
            self.db.save_goal_version(
                stored,
                change_summary=change_summary or "initial goal creation",
                actor=actor,
                reason=reason,
            )
            self._wake_reactor("GOAL_CREATED", goal_id=stored.goal_id)
            return stored

        next_version = max(int(cand.version), int(existing.version) + 1)
        next_epoch = max(int(cand.human_epoch), int(existing.human_epoch) + 1)
        stored = ResearchGoal.from_dict(
            {
                **cand.to_dict(),
                "version": next_version,
                "human_epoch": next_epoch,
            }
        )
        self.db.save_goal(stored, status=status)
        self.db.save_goal_version(
            stored,
            change_summary=change_summary or f"updated goal to v{next_version}",
            actor=actor,
            reason=reason,
        )
        self._wake_reactor("GOAL_UPDATED", goal_id=stored.goal_id)
        return stored

    # ---------------------------------------------- Versioning & Impact Recheck

    def analyze_version_impact(
        self, goal_id: str, patch: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Analyze what hypotheses, tasks, workflows, and conclusions a goal patch would affect."""
        if not isinstance(patch, Mapping) or not patch:
            raise ValueError("patch must be a non-empty mapping")
        goal = self.db.get_goal(goal_id)
        if goal is None:
            raise KeyError(f"Unknown goal_id: {goal_id}")

        base_dict = goal.to_dict()
        unknown_keys = sorted(set(patch.keys()) - set(base_dict.keys()))
        if unknown_keys:
            raise ValueError(f"Unknown ResearchGoal fields in patch: {unknown_keys}")

        merged = dict(base_dict)
        for k, v in patch.items():
            merged[k] = v
        merged["version"] = int(goal.version) + 1
        merged["human_epoch"] = int(goal.human_epoch) + 1
        candidate = ResearchGoal.from_dict(merged)
        _validate_goal_fields(candidate)

        changed_fields = [
            k for k in sorted(patch.keys()) if base_dict.get(k) != merged.get(k)
        ]
        objective_changed = any(f in OBJECTIVE_FIELDS for f in changed_fields)
        scope_changed = any(f in SCOPE_FIELDS for f in changed_fields)
        resource_changed = any(f in RESOURCE_FIELDS for f in changed_fields)

        tasks = self.db.list_tasks(goal_id=goal_id)
        affected_task_ids: list[str] = []
        if objective_changed or scope_changed:
            for t in tasks:
                if t.status in ACTIVE_TASK_STATUSES and int(t.goal_version) < int(candidate.version):
                    affected_task_ids.append(t.task_id)

        workflows = self.db.list_workflows(goal_id=goal_id)
        affected_workflow_ids: list[str] = []
        if objective_changed or scope_changed:
            for wf in workflows:
                if wf.status in ACTIVE_WORKFLOW_STATUSES and int(wf.goal_version) < int(candidate.version):
                    affected_workflow_ids.append(wf.workflow_id)

        recheck_preview = self.recheck_conclusions_against_version(
            candidate, persist=False
        )
        verdict_changes = [
            r for r in recheck_preview["results"] if r["changed"]
        ]

        if objective_changed or scope_changed:
            recommendation = (
                "CANCEL_INFLIGHT_AND_RECHECK"
                if (affected_task_ids or verdict_changes)
                else "RECHECK_AND_CONTINUE"
            )
        else:
            recommendation = "CONFIG_ONLY_UPDATE"

        return {
            "goal_id": goal_id,
            "current_version": int(goal.version),
            "next_version": int(candidate.version),
            "changed_fields": changed_fields,
            "objective_changed": objective_changed,
            "scope_changed": scope_changed,
            "resource_changed": resource_changed,
            "affected_task_ids": affected_task_ids,
            "affected_workflow_ids": affected_workflow_ids,
            "recheck_preview": recheck_preview,
            "verdict_changes": verdict_changes,
            "recommendation": recommendation,
        }

    def recheck_conclusions_against_version(
        self,
        goal: ResearchGoal | str,
        *,
        persist: bool = True,
    ) -> dict[str, Any]:
        """Re-evaluate stored EvidenceReceipts against updated objective/baseline/direction/kill_criteria without re-running raw sims."""
        resolved_goal = self.db.get_goal(goal) if isinstance(goal, str) else goal
        if resolved_goal is None:
            raise KeyError(f"Unknown goal_id: {goal}")

        direction = _normalize_direction(resolved_goal.direction)
        receipts = self.db.list_receipts(goal_id=resolved_goal.goal_id)
        hyps = {
            h.hypothesis_id: h
            for h in self.db.list_hypotheses(goal_id=resolved_goal.goal_id)
        }
        tasks = {
            t.task_id: t
            for t in self.db.list_tasks(goal_id=resolved_goal.goal_id)
        }

        evaluated: list[dict[str, Any]] = []
        latest_by_hyp: dict[str, dict[str, Any]] = {}

        for rcpt in receipts:
            hyp = hyps.get(rcpt.hypothesis_id)
            task = tasks.get(rcpt.task_id)
            old_verdict = rcpt.scientific_verdict

            if rcpt.stage not in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM) and old_verdict == ScientificVerdict.NOT_APPLICABLE:
                continue

            if rcpt.evidence_validity == EvidenceValidity.HUMAN_INVALIDATED or (
                hyp is not None and hyp.status in ("PRUNED", "HUMAN_INVALIDATED")
            ):
                new_validity = EvidenceValidity.HUMAN_INVALIDATED
                new_verdict = ScientificVerdict.NOT_APPLICABLE
                reason = "branch pruned or human-invalidated"
                metric_val = None
            elif rcpt.evidence_validity == EvidenceValidity.INVALID_SYNTHETIC:
                new_validity = EvidenceValidity.INVALID_SYNTHETIC
                new_verdict = ScientificVerdict.UNKNOWN
                reason = "invalid synthetic evidence cannot be confirmed"
                metric_val = None
            elif rcpt.evidence_validity == EvidenceValidity.MISSING_INPUT:
                new_validity = EvidenceValidity.MISSING_INPUT
                new_verdict = ScientificVerdict.UNKNOWN
                reason = "missing required input artifact"
                metric_val = None
            elif (
                rcpt.process_status != ProcessStatus.EXIT_OK
                or rcpt.engineering_verdict != EngineeringVerdict.PASS
            ):
                new_validity = rcpt.evidence_validity
                new_verdict = (
                    ScientificVerdict.UNKNOWN
                    if old_verdict in (ScientificVerdict.CONFIRMED, ScientificVerdict.REFUTED)
                    and not is_valid_empirical_outcome(
                        rcpt,
                        ResearchGoal.from_dict(
                            {**resolved_goal.to_dict(), "version": int(rcpt.goal_version)}
                        ),
                    )
                    else old_verdict
                )
                reason = rcpt.failure_reason or "non-PASS process/engineering receipt"
                metric_val = None
            else:
                aligned_goal = ResearchGoal.from_dict(
                    {**resolved_goal.to_dict(), "version": int(rcpt.goal_version)}
                )
                eff_domain = (
                    rcpt.failure_domain
                    if rcpt.failure_domain != FailureDomain.NONE
                    else classify_failure_domain(rcpt, goal=aligned_goal)
                )
                flat = _flatten_metrics(rcpt.metrics)
                if resolved_goal.primary_metric not in flat:
                    if resolved_goal.primary_metric == "tests_passed" and rcpt.tests_total > 0:
                        flat["tests_passed"] = float(rcpt.tests_passed)
                if eff_domain in (
                    FailureDomain.INFRA_TRANSIENT,
                    FailureDomain.ENV_CONFIG_MISMATCH,
                    FailureDomain.ENGINEERING_DEFECT,
                ) or rcpt.evidence_validity in (
                    EvidenceValidity.VALID_ENGINEERING_ONLY,
                    EvidenceValidity.MISSING_PRIMARY_METRIC,
                ):
                    if (
                        rcpt.evidence_validity == EvidenceValidity.VALID_EMPIRICAL
                        and resolved_goal.primary_metric in flat
                        and is_valid_empirical_outcome(rcpt, aligned_goal)
                    ):
                        pass
                    else:
                        new_validity = (
                            EvidenceValidity.MISSING_PRIMARY_METRIC
                            if resolved_goal.primary_metric not in flat
                            else rcpt.evidence_validity
                        )
                        new_verdict = ScientificVerdict.UNKNOWN
                        reason = (
                            rcpt.failure_reason
                            or f"Receipt lacks authentic VALID_EMPIRICAL outcome ({eff_domain.value})"
                        )
                        metric_val = None
                        flat = {}
                if flat and resolved_goal.primary_metric not in flat:
                    new_validity = EvidenceValidity.MISSING_PRIMARY_METRIC
                    new_verdict = ScientificVerdict.UNKNOWN
                    reason = (
                        f"Missing primary_metric {resolved_goal.primary_metric!r} "
                        "in stored receipt metrics"
                    )
                    metric_val = None
                elif flat:
                    metric_val = float(flat[resolved_goal.primary_metric])
                    all_kills = [
                        k
                        for k in list(resolved_goal.kill_criteria)
                        + (list(hyp.kill_criteria) if hyp is not None else [])
                        if not (
                            resolved_goal.primary_metric != "tests_passed"
                            and "tests_passed" in str(k)
                        )
                    ]
                    triggered = _evaluate_kill_criteria_local(all_kills, flat)
                    if triggered:
                        new_validity = EvidenceValidity.VALID_EMPIRICAL
                        new_verdict = ScientificVerdict.VALID_NEGATIVE
                        reason = f"kill_criterion triggered: {triggered[0]}"
                    else:
                        improved = (
                            metric_val < float(resolved_goal.baseline_value)
                            if direction == "minimize"
                            else metric_val > float(resolved_goal.baseline_value)
                        )
                        new_validity = EvidenceValidity.VALID_EMPIRICAL
                        if improved:
                            new_verdict = ScientificVerdict.CONFIRMED
                            reason = ""
                        else:
                            new_verdict = ScientificVerdict.VALID_NEGATIVE
                            reason = (
                                f"{resolved_goal.primary_metric}={metric_val} did not beat "
                                f"baseline {resolved_goal.baseline_value} ({direction})"
                            )

            meets_target: bool | None = None
            if metric_val is not None and resolved_goal.target_value is not None:
                meets_target = (
                    metric_val <= float(resolved_goal.target_value)
                    if direction == "minimize"
                    else metric_val >= float(resolved_goal.target_value)
                )

            entry = {
                "receipt_id": rcpt.receipt_id,
                "task_id": rcpt.task_id,
                "hypothesis_id": rcpt.hypothesis_id,
                "stage": rcpt.stage.value,
                "old_verdict": old_verdict.value,
                "new_verdict": new_verdict.value,
                "new_validity": new_validity.value,
                "metric_value": metric_val,
                "meets_target": meets_target,
                "changed": old_verdict != new_verdict,
                "reason": reason,
                "task_goal_version": task.goal_version if task is not None else rcpt.goal_version,
            }
            evaluated.append(entry)
            latest_by_hyp[rcpt.hypothesis_id] = entry

        hyp_updates: list[dict[str, Any]] = []
        if persist:
            for hid, latest in latest_by_hyp.items():
                hyp = hyps.get(hid)
                if hyp is None:
                    continue
                new_v = ScientificVerdict(latest["new_verdict"])
                if hyp.scientific_verdict != new_v:
                    old_status = hyp.status
                    if new_v == ScientificVerdict.VALID_NEGATIVE:
                        new_status = "VALID_NEGATIVE"
                    elif new_v == ScientificVerdict.UNKNOWN:
                        new_status = "RECHECK_UNKNOWN"
                    elif new_v == ScientificVerdict.CONFIRMED:
                        new_status = (
                            "COMPLETED"
                            if hyp.status in ("VALID_NEGATIVE", "RECHECK_UNKNOWN", "REFUTED")
                            else hyp.status
                        )
                    else:
                        new_status = hyp.status
                    self.db.transition_hypothesis_status(
                        hid,
                        old_status,
                        new_status,
                        scientific_verdict=new_v,
                    )
                    hyp_updates.append(
                        {
                            "hypothesis_id": hid,
                            "old_verdict": hyp.scientific_verdict.value,
                            "new_verdict": new_v.value,
                            "old_status": old_status,
                            "new_status": new_status,
                        }
                    )
                self.db.upsert_knowledge_item(
                    resolved_goal.goal_id,
                    "rechecked_conclusion",
                    hid,
                    latest,
                    hypothesis_id=hid,
                    goal_version=resolved_goal.version,
                )
            self.db.upsert_knowledge_item(
                resolved_goal.goal_id,
                "version_recheck",
                f"v{resolved_goal.version}",
                {
                    "goal_version": resolved_goal.version,
                    "primary_metric": resolved_goal.primary_metric,
                    "baseline_value": resolved_goal.baseline_value,
                    "direction": direction,
                    "evaluated_receipts": len(evaluated),
                    "changed_count": sum(1 for r in evaluated if r["changed"]),
                    "hypothesis_updates": hyp_updates,
                },
                goal_version=resolved_goal.version,
            )

        return {
            "goal_id": resolved_goal.goal_id,
            "goal_version": int(resolved_goal.version),
            "total_receipts": len(receipts),
            "evaluated_receipts": len(evaluated),
            "results": evaluated,
            "hypothesis_updates": hyp_updates,
            "changed_count": sum(1 for r in evaluated if r["changed"]),
        }

    # ------------------------------------------------------- Epoch Isolation

    def validate_receipt_epoch(
        self,
        goal: ResearchGoal | None,
        task: StageTask | None,
        receipt: EvidenceReceipt | None,
    ) -> EvidenceValidity:
        """Check whether a completed task/receipt is still valid for the current goal version and human epoch."""
        if receipt is not None and receipt.evidence_validity in (
            EvidenceValidity.HUMAN_INVALIDATED,
            EvidenceValidity.SUPERSEDED_EPOCH,
            EvidenceValidity.STALE_GOAL_VERSION,
        ):
            return receipt.evidence_validity

        hyp_id = (
            task.hypothesis_id
            if task is not None
            else (receipt.hypothesis_id if receipt is not None else None)
        )
        wf_id = (
            task.workflow_id
            if task is not None and task.workflow_id
            else (receipt.workflow_id if receipt is not None else None)
        )
        goal_id = (
            goal.goal_id
            if goal is not None
            else (
                task.goal_id
                if task is not None
                else (receipt.goal_id if receipt is not None else None)
            )
        )

        if goal_id:
            goal_status = self.db.get_goal_status(goal_id)
            if goal_status == "STOPPED":
                return EvidenceValidity.HUMAN_INVALIDATED

        if hyp_id:
            hyp = self.db.get_hypothesis(hyp_id)
            if hyp is not None and hyp.status in ("PRUNED", "HUMAN_INVALIDATED"):
                return EvidenceValidity.HUMAN_INVALIDATED

        if wf_id:
            wf = self.db.get_workflow(wf_id)
            if wf is not None and wf.status in ("PRUNED", "HUMAN_INVALIDATED", "CANCELLED"):
                return EvidenceValidity.HUMAN_INVALIDATED

        if task is not None and task.status in ("PRUNED", "HUMAN_INVALIDATED", "CANCELLED"):
            return EvidenceValidity.HUMAN_INVALIDATED

        if goal is not None:
            preserved_ids = self.get_preserved_upstream_receipt_ids(
                goal.goal_id, hypothesis_id=hyp_id
            )
            return gates.validate_receipt_epoch(
                receipt,
                goal,
                task=task,
                preserved_receipt_ids=preserved_ids,
            )

        if receipt is not None:
            return receipt.evidence_validity
        return EvidenceValidity.VALID_ENGINEERING_ONLY

    def get_preserved_upstream_receipt_ids(
        self,
        goal_id: str,
        *,
        hypothesis_id: str | None = None,
    ) -> set[str]:
        """Return receipt IDs explicitly preserved across `update_plan` / `replan_from_node` lineage."""
        out: set[str] = set()
        for item in self.db.list_knowledge_items(goal_id=goal_id, category="plan_lineage", limit=50):
            content = item.get("content")
            if not isinstance(content, Mapping):
                continue
            item_hid = item.get("hypothesis_id") or content.get("hypothesis_id")
            if hypothesis_id is not None and item_hid and str(item_hid) != str(hypothesis_id):
                continue
            for rid in content.get("preserved_upstream_receipt_ids") or []:
                if str(rid).strip():
                    out.add(str(rid).strip())
        return out

    # ---------------------------------------------------- HITL Interventions

    def _cancel_tasks_and_hq(
        self,
        tasks: Sequence[StageTask],
        *,
        reason: str,
        epoch: int = 0,
        plan_epoch: int = 0,
    ) -> tuple[list[str], list[int]]:
        cancelled_task_ids: list[str] = []
        hq_job_ids: list[int] = []
        hq_job_meta: list[tuple[int, str, int]] = []
        now = time.time()
        for t in tasks:
            if t.status not in ACTIVE_TASK_STATUSES:
                continue
            try:
                self.db.transition_task_status(
                    t.task_id,
                    ACTIVE_TASK_STATUSES,
                    "CANCELLED",
                    finished_ts=now,
                    now=now,
                )
                cancelled_task_ids.append(t.task_id)
                if t.hq_job_id is not None:
                    jid = int(t.hq_job_id)
                    hq_job_ids.append(jid)
                    hq_job_meta.append((jid, t.task_id, int(t.attempt)))
                    if hasattr(self.db, "record_pending_hq_cancellation"):
                        self.db.record_pending_hq_cancellation(
                            hq_job_id=jid,
                            entity_kind="task",
                            entity_id=t.task_id,
                            attempt=int(t.attempt),
                            goal_id=t.goal_id,
                            hypothesis_id=t.hypothesis_id,
                            workflow_id=t.workflow_id or "",
                            epoch=int(epoch or t.human_epoch),
                            plan_epoch=int(plan_epoch or getattr(t, "plan_epoch", 0) or 0),
                            reason=reason,
                            state="CANCEL_PENDING",
                            now=now,
                        )
                if hasattr(self.db, "abort_cancelled_task_promotions"):
                    self.db.abort_cancelled_task_promotions(
                        t.goal_id,
                        t.hypothesis_id,
                        t.task_id,
                        reason=reason,
                        now=now,
                    )
            except StateTransitionError:
                continue
        if hq_job_ids and self.hq is not None:
            cancel_map: Mapping[int, bool] | None = None
            try:
                res_c = self.hq.cancel_jobs(hq_job_ids, reason=reason)
                if isinstance(res_c, Mapping):
                    cancel_map = res_c
            except Exception:
                pass
            if hasattr(self.db, "confirm_hq_cancellation"):
                for jid, ent_id_c, att_c in hq_job_meta:
                    if self._hq_positively_terminated(self.hq, jid, cancel_map=cancel_map):
                        self.db.confirm_hq_cancellation(
                            jid, entity_kind="task", entity_id=ent_id_c, attempt=att_c, now=now
                        )
        return cancelled_task_ids, hq_job_ids

    def _cancel_decisions_and_hq(
        self,
        goal_id: str,
        *,
        reason: str,
        hypothesis_ids: set[str] | None = None,
        subgoal_id: str | None = None,
        epoch: int = 0,
        plan_epoch: int = 0,
    ) -> tuple[list[str], list[int]]:
        """Cancel active (`PENDING`/`RUNNING`) `dec-*` DecisionRequests and persist durable HQ cancellations."""
        cancelled_reqs: list[str] = []
        hq_job_ids: list[int] = []
        hq_dec_meta: list[tuple[int, str, int, str]] = []
        now = time.time()
        for st in ("PENDING", "SUBMITTING", "RUNNING"):
            for req in self.db.list_decision_requests(goal_id=goal_id, status=st):
                pbody = req.get("payload") or {}
                rhid = pbody.get("hypothesis_id") or pbody.get("target_hypothesis_id")
                rsg_id = str(req.get("subgoal_id") or pbody.get("subgoal_id") or "").strip() or None
                if subgoal_id is not None:
                    matches_sg = rsg_id == subgoal_id or (
                        hypothesis_ids is not None and rhid and str(rhid) in hypothesis_ids
                    )
                    if not matches_sg:
                        continue
                elif hypothesis_ids is not None:
                    if rhid and str(rhid) not in hypothesis_ids:
                        continue
                    if not rhid:
                        continue
                rid = str(req["req_id"])
                req_att_c = int(pbody.get("attempt", req.get("attempt", 1)) or 1)
                self.db.resolve_decision_request(
                    rid,
                    status="CANCELLED",
                    resolution={"reason": reason},
                    now=now,
                )
                cancelled_reqs.append(rid)
                raw_jid = req.get("hq_job_id")
                if raw_jid is None and st == "SUBMITTING" and self.hq is not None:
                    q_list = self._query_hq_job_list(self.hq, all_jobs=True)
                    if str(q_list.get("lookup_status") or "").upper() == "OK":
                        s_name_c = str(pbody.get("submit_job_name") or f"dec-{rid}")
                        for j_item in q_list.get("jobs") or []:
                            if (
                                isinstance(j_item, Mapping)
                                and j_item.get("id") is not None
                                and str(j_item.get("name") or "") in {s_name_c, f"dec-{rid}"}
                            ):
                                raw_jid = int(j_item["id"])
                                self.db.resolve_decision_request(
                                    rid,
                                    status="CANCELLED",
                                    hq_job_id=raw_jid,
                                    resolution={"reason": reason},
                                    now=now,
                                )
                                break
                if raw_jid is not None:
                    jid = int(raw_jid)
                    hq_job_ids.append(jid)
                    dec_ukey_c = str(
                        pbody.get("usage_key")
                        or (
                            f"decision:{rid}:a{req_att_c}:{pbody['submit_nonce']}"
                            if pbody.get("submit_nonce")
                            else f"decision:{rid}:a{req_att_c}"
                        )
                    )
                    hq_dec_meta.append((jid, rid, req_att_c, dec_ukey_c))
                    if hasattr(self.db, "record_pending_hq_cancellation"):
                        self.db.record_pending_hq_cancellation(
                            hq_job_id=jid,
                            entity_kind="decision",
                            entity_id=rid,
                            attempt=req_att_c,
                            goal_id=goal_id,
                            hypothesis_id=str(rhid or ""),
                            epoch=int(epoch),
                            plan_epoch=int(plan_epoch),
                            reason=reason,
                            state="CANCEL_PENDING",
                            usage_key=dec_ukey_c,
                            now=now,
                        )
        if hq_job_ids and self.hq is not None:
            cancel_map: Mapping[int, bool] | None = None
            try:
                res_c = self.hq.cancel_jobs(hq_job_ids, reason=reason)
                if isinstance(res_c, Mapping):
                    cancel_map = res_c
            except Exception:
                pass
            if hasattr(self.db, "confirm_hq_cancellation"):
                for jid, rid_c, r_att_c, ukey_c in hq_dec_meta:
                    if self._hq_positively_terminated(self.hq, jid, cancel_map=cancel_map):
                        self.db.confirm_hq_cancellation(
                            jid,
                            entity_kind="decision",
                            entity_id=rid_c,
                            attempt=r_att_c,
                            usage_key=ukey_c,
                            now=now,
                        )
        return cancelled_reqs, hq_job_ids

    def _collect_affected_synthesis_workflows(
        self,
        goal_id: str,
        affected_hids: set[str],
    ) -> list[WorkflowNode]:
        """Return active S3 `SYNTHESIS_REVIEW` workflows whose `synthesized_hypothesis_ids` intersect `affected_hids`."""
        out: list[WorkflowNode] = []
        for wf_obj in self.db.list_workflows(goal_id=goal_id):
            if wf_obj.status not in ACTIVE_WORKFLOW_STATUSES:
                continue
            syn_hids = {
                str(x).strip()
                for x in (wf_obj.config.get("synthesized_hypothesis_ids") or [])
                if str(x).strip()
            }
            is_syn = str(wf_obj.config.get("target_kind") or "") == "goal_synthesis" or bool(syn_hids)
            if is_syn and (bool(syn_hids & affected_hids) or wf_obj.hypothesis_id in affected_hids):
                out.append(wf_obj)
        return out

    def _mark_synthesis_workflows_superseded(
        self,
        goal: ResearchGoal,
        syn_workflows: Sequence[WorkflowNode],
        *,
        reason: str,
        plan_epoch: int,
    ) -> list[str]:
        cancelled_ids: list[str] = []
        for wf_obj in syn_workflows:
            self.db.update_workflow_state(
                wf_obj.workflow_id,
                status="CANCELLED",
                state_patch={"superseded_by_plan_epoch": plan_epoch, "cancel_reason": reason},
            )
            cancelled_ids.append(wf_obj.workflow_id)
            syn_key = str(wf_obj.config.get("synthesis_key") or f"syn:{wf_obj.workflow_id}")
            syn_hids = [
                str(x).strip()
                for x in (wf_obj.config.get("synthesized_hypothesis_ids") or [])
                if str(x).strip()
            ]
            self.db.upsert_knowledge_item(
                goal.goal_id,
                "synthesis_pending",
                syn_key,
                {
                    "status": "SUPERSEDED_BY_SOURCE_UPDATE",
                    "workflow_id": wf_obj.workflow_id,
                    "synthesized_hypothesis_ids": syn_hids,
                    "reason": reason,
                },
                hypothesis_id=wf_obj.hypothesis_id,
                goal_version=int(goal.version),
            )
        return cancelled_ids

    def _collect_hyp_descendants(
        self, goal_id: str, root_hyp_id: str
    ) -> list[HypothesisNode]:
        all_hyps = self.db.list_hypotheses(goal_id=goal_id)
        by_parent: dict[str, list[HypothesisNode]] = {}
        for h in all_hyps:
            if h.parent_id:
                by_parent.setdefault(h.parent_id, []).append(h)
        out: list[HypothesisNode] = []
        queue = list(by_parent.get(root_hyp_id, []))
        seen = {root_hyp_id}
        while queue:
            cur = queue.pop(0)
            if cur.hypothesis_id in seen:
                continue
            seen.add(cur.hypothesis_id)
            out.append(cur)
            queue.extend(by_parent.get(cur.hypothesis_id, []))
        return out

    def _collect_wf_descendants(
        self, goal_id: str, root_wf_id: str
    ) -> list[WorkflowNode]:
        all_wfs = self.db.list_workflows(goal_id=goal_id)
        by_parent: dict[str, list[WorkflowNode]] = {}
        for wf in all_wfs:
            if wf.parent_workflow_id:
                by_parent.setdefault(wf.parent_workflow_id, []).append(wf)
        out: list[WorkflowNode] = []
        queue = list(by_parent.get(root_wf_id, []))
        seen = {root_wf_id}
        while queue:
            cur = queue.pop(0)
            if cur.workflow_id in seen:
                continue
            seen.add(cur.workflow_id)
            out.append(cur)
            queue.extend(by_parent.get(cur.workflow_id, []))
        return out

    def _require_goal(self, goal_id: str | None) -> ResearchGoal:
        if not goal_id or not str(goal_id).strip():
            raise ValueError("goal_id is required")
        goal = self.db.get_goal(str(goal_id))
        if goal is None:
            raise KeyError(f"Unknown goal_id: {goal_id}")
        return goal

    def _bump_goal_epoch(
        self,
        goal: ResearchGoal,
        *,
        status: str | None = None,
        bump_plan_epoch: bool = False,
        sync_unaffected_task_ids: set[str] | None = None,
    ) -> tuple[ResearchGoal, int, int]:
        before_epoch = int(goal.human_epoch)
        after_epoch = before_epoch + 1
        cur_plan_epoch = int(getattr(goal, "plan_epoch", 0) or 0)
        next_plan_epoch = cur_plan_epoch + 1 if bump_plan_epoch else cur_plan_epoch
        updated = ResearchGoal.from_dict(
            {
                **goal.to_dict(),
                "human_epoch": after_epoch,
                "plan_epoch": next_plan_epoch,
            }
        )
        self.db.save_goal(updated, status=status)
        if sync_unaffected_task_ids is not None:
            now = time.time()
            for t in self.db.list_tasks(goal_id=goal.goal_id):
                if t.status in ACTIVE_TASK_STATUSES and t.task_id in sync_unaffected_task_ids:
                    new_route = (
                        {
                            **dict(t.pinned_route),
                            "plan_epoch": next_plan_epoch,
                            "exec_fence_plan_epoch": next_plan_epoch,
                            "exec_fence_human_epoch": after_epoch,
                        }
                        if isinstance(t.pinned_route, Mapping)
                        else {
                            "plan_epoch": next_plan_epoch,
                            "exec_fence_plan_epoch": next_plan_epoch,
                            "exec_fence_human_epoch": after_epoch,
                        }
                    )
                    try:
                        self.db.transition_task_status(
                            t.task_id,
                            ACTIVE_TASK_STATUSES,
                            t.status,
                            human_epoch=after_epoch,
                            plan_epoch=next_plan_epoch,
                            pinned_route=new_route,
                            now=now,
                        )
                    except StateTransitionError:
                        continue
        return updated, before_epoch, after_epoch

    def _pause_single_goal(
        self,
        goal: ResearchGoal,
        *,
        actor: str = "user",
        reason: str = "",
        cancel_running: bool = True,
    ) -> tuple[dict[str, Any], int, int]:
        """Pause a Goal cleanly: cancel DISPATCHED/RUNNING HQ tasks to CANCELLED (never mutating to READY)."""
        eff_goal_id = goal.goal_id
        now = time.time()
        all_tasks = self.db.list_tasks(goal_id=eff_goal_id)
        running_tasks = [t for t in all_tasks if t.status in ("DISPATCHED", "RUNNING")]
        cancelled_tasks: list[str] = []
        cancelled_hq: list[int] = []
        paused_records: list[dict[str, Any]] = []

        if cancel_running and running_tasks:
            for t in running_tasks:
                paused_records.append(t.to_dict())
            cancelled_tasks, cancelled_hq = self._cancel_tasks_and_hq(
                running_tasks,
                reason=reason or f"goal_paused:{eff_goal_id}:{actor}",
                epoch=int(goal.human_epoch) + 1,
                plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
            )
        cancelled_decs, dec_hq = self._cancel_decisions_and_hq(
            eff_goal_id,
            reason=reason or f"goal_paused:{eff_goal_id}:{actor}",
            epoch=int(goal.human_epoch) + 1,
            plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
        )
        cancelled_hq.extend(dec_hq)

        paused_wfs: list[str] = []
        by_wf: dict[str, list[dict[str, Any]]] = {}
        standalone_paused: list[dict[str, Any]] = []
        for snap in paused_records:
            wf_id = snap.get("workflow_id")
            if wf_id:
                by_wf.setdefault(str(wf_id), []).append(snap)
            else:
                standalone_paused.append(snap)

        for wf in self.db.list_workflows(goal_id=eff_goal_id):
            if wf.status in ("READY", "RUNNING", "WAITING"):
                patch: dict[str, Any] = {"paused_prev_status": wf.status}
                if wf.workflow_id in by_wf:
                    existing_paused = list(wf.state.get("paused_tasks") or [])
                    patch["paused_tasks"] = existing_paused + by_wf[wf.workflow_id]
                self.db.update_workflow_state(
                    wf.workflow_id,
                    status="PAUSED",
                    state_patch=patch,
                    now=now,
                )
                paused_wfs.append(wf.workflow_id)

        if standalone_paused:
            self.db.upsert_knowledge_item(
                eff_goal_id,
                "paused_standalone_tasks",
                "active_pause",
                {"tasks": standalone_paused, "paused_ts": now},
                goal_version=int(goal.version),
                now=now,
            )

        _, before_epoch, after_epoch = self._bump_goal_epoch(goal, status="PAUSED")
        return (
            {
                "goal_id": eff_goal_id,
                "status": "PAUSED",
                "cancelled_task_ids": cancelled_tasks,
                "cancelled_hq_job_ids": cancelled_hq,
                "paused_workflow_ids": paused_wfs,
            },
            before_epoch,
            after_epoch,
        )

    def _resume_single_goal(
        self,
        goal: ResearchGoal,
        *,
        spawn_immediately_if_safe: bool = True,
    ) -> tuple[dict[str, Any], int, int]:
        """Resume a PAUSED Goal: advance READY tasks' epoch and spawn attempt+1 replacements once HQ terminated."""
        eff_goal_id = goal.goal_id
        now = time.time()
        ready_ids = {
            t.task_id
            for t in self.db.list_tasks(goal_id=eff_goal_id)
            if t.status == "READY"
        }
        updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
            goal,
            status="ACTIVE",
            sync_unaffected_task_ids=ready_ids,
        )

        resumed_wfs: list[str] = []
        spawned_task_ids: list[str] = []
        deferred_hq_jobs: list[int] = []

        def _hq_terminated(
            jid: int | None,
            ent_id: str = "",
            *,
            entity_kind: str | None = None,
            attempt: int | None = None,
        ) -> bool:
            if jid is None:
                return True
            if hasattr(self.db, "list_pending_hq_cancellations"):
                confirmed_rows = self.db.list_pending_hq_cancellations(
                    goal_id=eff_goal_id, state="CANCEL_CONFIRMED"
                )
                if any(
                    int(r.get("hq_job_id", -1)) == int(jid)
                    and (
                        not entity_kind
                        or not r.get("entity_kind")
                        or str(r.get("entity_kind")) == entity_kind
                    )
                    and (not ent_id or not r.get("entity_id") or str(r.get("entity_id")) == ent_id)
                    and (
                        attempt is None
                        or not r.get("attempt")
                        or int(r.get("attempt") or 0) == int(attempt)
                    )
                    for r in confirmed_rows
                ):
                    return True
            q_state = self._query_hq_job_state(self.hq, int(jid))
            lookup_status = str(q_state.get("lookup_status") or "QUERY_ERROR").upper()
            if lookup_status == "QUERY_ERROR":
                return False
            if lookup_status == "FOUND":
                if self._hq_positively_terminated(
                    self.hq, int(jid), prefetched_info=q_state.get("job")
                ):
                    if hasattr(self.db, "confirm_hq_cancellation"):
                        self.db.confirm_hq_cancellation(
                            int(jid),
                            entity_kind=entity_kind,
                            entity_id=ent_id or None,
                            attempt=attempt,
                            now=now,
                        )
                    return True
                return False
            if lookup_status == "NOT_FOUND":
                has_live_lease = bool(
                    ent_id
                    and any(
                        str(ls.get("task_id") or "") in {ent_id, f"dec-{ent_id}"}
                        for ls in self.db.list_active_leases(now=now)
                    )
                )
                if not has_live_lease:
                    if hasattr(self.db, "confirm_hq_cancellation"):
                        self.db.confirm_hq_cancellation(
                            int(jid),
                            entity_kind=entity_kind,
                            entity_id=ent_id or None,
                            attempt=attempt,
                            now=now,
                        )
                    return True
            return False

        waiting_cancel_confirm_ids: list[str] = []
        for wf in self.db.list_workflows(goal_id=eff_goal_id):
            if wf.status == "PAUSED":
                prev_st = str(wf.state.get("paused_prev_status") or "RUNNING")
                paused_snaps = list(wf.state.get("paused_tasks") or [])
                remaining_snaps: list[dict[str, Any]] = []
                if spawn_immediately_if_safe and paused_snaps:
                    for snap in paused_snaps:
                        jid = snap.get("hq_job_id")
                        tid_s = str(snap.get("task_id") or "")
                        att_s = int(snap["attempt"]) if snap.get("attempt") is not None else None
                        if not _hq_terminated(
                            int(jid) if jid is not None else None,
                            tid_s,
                            entity_kind="task",
                            attempt=att_s,
                        ):
                            snap_wait = {**dict(snap), "wait_reason": "WAIT_CANCEL_CONFIRM"}
                            remaining_snaps.append(snap_wait)
                            if jid is not None:
                                deferred_hq_jobs.append(int(jid))
                            if tid_s:
                                waiting_cancel_confirm_ids.append(tid_s)
                            continue
                        repl = self._spawn_replacement_task_from_snapshot(
                            updated_goal, snap, now=now
                        )
                        spawned_task_ids.append(repl.task_id)
                else:
                    remaining_snaps = paused_snaps
                self.db.update_workflow_state(
                    wf.workflow_id,
                    status=prev_st if prev_st in ("READY", "RUNNING", "WAITING") else "RUNNING",
                    state_patch={"paused_tasks": remaining_snaps},
                    now=now,
                )
                resumed_wfs.append(wf.workflow_id)

        standalone_items = self.db.list_knowledge_items(
            goal_id=eff_goal_id, category="paused_standalone_tasks", limit=5
        )
        for item in standalone_items:
            if item.get("key") != "active_pause":
                continue
            content = item.get("content") or {}
            snaps = list(content.get("tasks") or [])
            rem: list[dict[str, Any]] = []
            for snap in snaps:
                jid = snap.get("hq_job_id")
                tid_s = str(snap.get("task_id") or "")
                att_s = int(snap["attempt"]) if snap.get("attempt") is not None else None
                if spawn_immediately_if_safe and _hq_terminated(
                    int(jid) if jid is not None else None,
                    tid_s,
                    entity_kind="task",
                    attempt=att_s,
                ):
                    repl = self._spawn_replacement_task_from_snapshot(
                        updated_goal, snap, now=now
                    )
                    spawned_task_ids.append(repl.task_id)
                else:
                    rem.append({**dict(snap), "wait_reason": "WAIT_CANCEL_CONFIRM"})
                    if jid is not None:
                        deferred_hq_jobs.append(int(jid))
                    if tid_s:
                        waiting_cancel_confirm_ids.append(tid_s)
            self.db.upsert_knowledge_item(
                eff_goal_id,
                "paused_standalone_tasks",
                "active_pause",
                {"tasks": rem, "resumed_ts": now},
                goal_version=int(updated_goal.version),
                now=now,
            )

        return (
            {
                "goal_id": eff_goal_id,
                "status": "ACTIVE",
                "resumed_workflow_ids": resumed_wfs,
                "spawned_replacement_task_ids": spawned_task_ids,
                "deferred_hq_job_ids": deferred_hq_jobs,
                "waiting_cancel_confirm_task_ids": waiting_cancel_confirm_ids,
                "resume_gate": "WAIT_CANCEL_CONFIRM" if waiting_cancel_confirm_ids else "READY",
            },
            before_epoch,
            after_epoch,
        )

    def _spawn_replacement_task_from_snapshot(
        self,
        goal: ResearchGoal,
        snap: Mapping[str, Any],
        *,
        pinned_route: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> StageTask:
        ts = time.time() if now is None else float(now)
        old_task = StageTask.from_dict(snap)
        next_attempt = int(old_task.attempt) + 1
        stage = old_task.stage
        wf_part = f":{old_task.workflow_id}" if old_task.workflow_id else ""
        step_part = old_task.step_key or stage.value
        goal_plan_epoch = int(getattr(goal, "plan_epoch", 0) or 0)
        new_idem = (
            f"{goal.goal_id}:{old_task.hypothesis_id}{wf_part}:{step_part}:"
            f"pe{goal_plan_epoch}:a{next_attempt}:e{int(goal.human_epoch)}"
        )
        res = ResourceSpec.from_dict(old_task.resources.to_dict())
        eff_route = (
            dict(pinned_route)
            if isinstance(pinned_route, Mapping)
            else (dict(old_task.pinned_route) if isinstance(old_task.pinned_route, Mapping) else None)
        )
        if eff_route is not None:
            eff_route["plan_epoch"] = goal_plan_epoch
            res.llm_pool = str(eff_route.get("pool") or res.llm_pool or "") or None
            res.hq_llm_resource = eff_route.get("hq_resource")
            res.pinned_route = dict(eff_route)
        new_task = StageTask(
            task_id=f"tsk-{old_task.hypothesis_id}-{stage.value.lower()}-a{next_attempt}-{uuid.uuid4().hex[:6]}",
            idempotency_key=new_idem,
            goal_id=goal.goal_id,
            hypothesis_id=old_task.hypothesis_id,
            stage=stage,
            deps=list(old_task.deps),
            resources=res,
            attempt=next_attempt,
            max_attempts=max(int(old_task.max_attempts), next_attempt),
            status="READY",
            worktree_path=old_task.worktree_path,
            enqueued_ts=ts,
            workflow_id=old_task.workflow_id,
            goal_version=int(goal.version),
            human_epoch=int(goal.human_epoch),
            plan_epoch=goal_plan_epoch,
            workflow_primitive=old_task.workflow_primitive,
            step_key=old_task.step_key,
            pinned_route=eff_route,
        )
        return self.db.enqueue_task(new_task)

    @staticmethod
    def _stage_to_role(stage: StageKind, hyp: HypothesisNode | None = None) -> ScientificRole | None:
        if stage == StageKind.DESIGN_REVIEW:
            return ScientificRole.DESIGN_REVIEWER
        if stage == StageKind.IMPL_REVIEW:
            return ScientificRole.IMPL_REVIEWER
        if stage in (StageKind.RESULT_REVIEW, StageKind.BLIND_AUDIT):
            return ScientificRole.RESULT_AUDITOR
        if stage == StageKind.PROPOSE:
            return ScientificRole.EXPERIMENT_DESIGNER
        if stage == StageKind.CODE:
            if hyp is not None and hyp.kind in (
                HypothesisKind.PARAMETER_RACE,
                HypothesisKind.ABLATION,
                HypothesisKind.FALSIFICATION,
            ) and not hyp.params.get("requires_llm_code", False):
                return None
            if hyp is not None and "theory" in {str(t).lower() for t in hyp.tags}:
                return ScientificRole.THEORIST
            return ScientificRole.ENGINEER_BUILDER
        return None

    @staticmethod
    def _policy_targets_role(
        goal: ResearchGoal,
        hyp: HypothesisNode | None,
        role_val: str,
        *,
        is_review_stage: bool = False,
        workflow_id: str | None = None,
        wf_config: Mapping[str, Any] | None = None,
        global_policy: Any = None,
        global_policy_nl: str | None = None,
    ) -> bool:
        """Return True when any active scoped policy (workflow -> branch -> goal -> global) targets `role_val` or sets disallows."""
        reviewer_roles = {
            ScientificRole.DESIGN_REVIEWER.value,
            ScientificRole.IMPL_REVIEWER.value,
            ScientificRole.RESULT_AUDITOR.value,
        }
        scope_docs: list[Mapping[str, Any]] = []
        if isinstance(wf_config, Mapping) and isinstance(wf_config.get("routing_policy"), Mapping):
            scope_docs.append(dict(wf_config["routing_policy"]))
        if hyp is not None and workflow_id and isinstance(hyp.params.get("workflow_routing_policies"), Mapping):
            wf_map_pol = hyp.params["workflow_routing_policies"].get(workflow_id)
            if isinstance(wf_map_pol, Mapping):
                scope_docs.append(dict(wf_map_pol))
        if hyp is not None and isinstance(hyp.params.get("routing_policy"), Mapping):
            scope_docs.append(dict(hyp.params["routing_policy"]))
        if goal.routing_policy and isinstance(goal.routing_policy, Mapping) and (
            goal.routing_policy.get("rules")
            or goal.routing_policy.get("disallowed_models")
            or goal.routing_policy.get("disallowed_pools")
            or goal.routing_policy.get("disallowed_families")
        ):
            scope_docs.append(dict(goal.routing_policy))
        elif goal.routing_policy_nl:
            try:
                scope_docs.append(ModelRouter.compile_routing_policy(goal.routing_policy_nl, project=goal.project))
            except Exception:
                pass
        elif goal.routing_policy:
            try:
                scope_docs.append(ModelRouter.compile_routing_policy(goal.routing_policy, project=goal.project))
            except Exception:
                pass
        if isinstance(global_policy, Mapping) and (
            global_policy.get("rules")
            or global_policy.get("disallowed_models")
            or global_policy.get("disallowed_pools")
            or global_policy.get("disallowed_families")
            or global_policy.get("disallowed")
        ):
            scope_docs.append(dict(global_policy))
        elif global_policy:
            try:
                scope_docs.append(ModelRouter.compile_routing_policy(global_policy, project=goal.project))
            except Exception:
                pass
        if global_policy_nl:
            try:
                scope_docs.append(ModelRouter.compile_routing_policy(global_policy_nl, project=goal.project))
            except Exception:
                pass

        rules: list[Mapping[str, Any]] = []
        has_global_disallow = False
        for doc in scope_docs:
            if (
                doc.get("disallowed_models")
                or doc.get("disallowed_pools")
                or doc.get("disallowed_families")
                or doc.get("disallowed")
            ):
                has_global_disallow = True
            for r_item in doc.get("rules") or []:
                if isinstance(r_item, Mapping):
                    rules.append(r_item)
        if not rules:
            return has_global_disallow

        hyp_tags: set[str] = set()
        if hyp is not None:
            for tg in hyp.tags:
                if str(tg).strip():
                    hyp_tags.add(str(tg).strip().lower())
            extra_t = hyp.params.get("tags")
            if isinstance(extra_t, list):
                for tg in extra_t:
                    if str(tg).strip():
                        hyp_tags.add(str(tg).strip().lower())

        goal_proj = (goal.project or "").strip().lower()
        norm_role = str(role_val or "").strip().lower()

        for rule in rules:
            if not isinstance(rule, Mapping):
                continue
            proj_scope = str(rule.get("project_scope") or "").strip().lower()
            if proj_scope and goal_proj and proj_scope != goal_proj:
                continue
            r_roles = {str(x).strip().lower() for x in (rule.get("roles") or []) if str(x).strip()}
            if (
                bool(rule.get("author_only", False))
                and (norm_role in reviewer_roles or is_review_stage)
                and norm_role not in r_roles
            ):
                continue
            if r_roles and norm_role not in r_roles:
                continue
            r_tags = {str(x).strip().lower() for x in (rule.get("tags") or []) if str(x).strip()}
            if r_tags and not (r_tags & hyp_tags):
                continue
            return True
        return False

    @staticmethod
    def _goal_for_role_resolution(
        goal: ResearchGoal,
        role: ScientificRole,
        *,
        is_review_stage: bool = False,
        has_avoid_families: bool = False,
    ) -> ResearchGoal:
        """Strip untargeted `author_only` rules when resolving a tie-breaker `RESEARCH_DIRECTOR` on a review stage."""
        if role != ScientificRole.RESEARCH_DIRECTOR or not (is_review_stage or has_avoid_families):
            return goal
        if not (goal.routing_policy or goal.routing_policy_nl):
            return goal
        raw_pol = goal.routing_policy
        if not (isinstance(raw_pol, Mapping) and raw_pol.get("rules")) and goal.routing_policy_nl:
            try:
                raw_pol = ModelRouter.compile_routing_policy(goal.routing_policy_nl, project=goal.project)
            except Exception:
                return goal
        if not (isinstance(raw_pol, Mapping) and raw_pol.get("rules")):
            return goal
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
        return ResearchGoal.from_dict(
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

    def _surgically_rebind_ready_tasks_for_policy(
        self,
        goal: ResearchGoal,
        tasks: Sequence[StageTask],
        *,
        router: ModelRouter,
        actor: str,
        reason: str,
    ) -> tuple[list[str], list[str], list[str]]:
        """Cancel ONLY READY tasks whose route changes/blocks under the updated policy; preserve review quorum diversity."""
        cancelled_ids: list[str] = []
        replaced_ids: list[str] = []
        blocked_ids: list[str] = []
        now = time.time()
        hyps_by_id = {
            h.hypothesis_id: h for h in self.db.list_hypotheses(goal_id=goal.goal_id)
        }
        wfs_by_id = {
            w.workflow_id: w for w in self.db.list_workflows(goal_id=goal.goal_id)
        }
        # Track chosen families/specs across sibling review slots in the same `(workflow_id, step_prefix)`
        sibling_families: dict[tuple[str, str], set[str]] = {}
        sibling_specs: dict[tuple[str, str], list[str]] = {}
        for other_t in self.db.list_tasks(goal_id=goal.goal_id):
            if other_t.status not in ("DISPATCHED", "RUNNING") or not other_t.workflow_id:
                continue
            oroute = other_t.pinned_route or {}
            if isinstance(oroute, Mapping) and "review_slot" in oroute:
                spref = re.sub(r"_s\d+$", "", other_t.step_key or other_t.stage.value.lower())
                key = (other_t.workflow_id, spref)
                fam = str(oroute.get("family") or "").strip().lower()
                if fam:
                    sibling_families.setdefault(key, set()).add(fam)
                if other_t.model_spec:
                    sibling_specs.setdefault(key, []).append(other_t.model_spec)

        for t in tasks:
            if t.status != "READY":
                continue
            hyp = hyps_by_id.get(t.hypothesis_id)
            cur_route = t.pinned_route if isinstance(t.pinned_route, Mapping) else {}
            role_override = str(cur_route.get("role") or "").strip()
            if role_override:
                try:
                    eff_role: ScientificRole | None = ScientificRole(role_override)
                except ValueError:
                    eff_role = self._stage_to_role(t.stage, hyp)
            elif bool(cur_route.get("tiebreaker")) or str(t.step_key or "").endswith("_s2"):
                eff_role = ScientificRole.RESEARCH_DIRECTOR
            else:
                eff_role = self._stage_to_role(t.stage, hyp)

            if eff_role is None and t.resources.llm_pool is None and not cur_route:
                continue
            resolved_role = eff_role or ScientificRole.ENGINEER_BUILDER

            avoid_fams: set[str] = {
                str(f).strip().lower()
                for f in (cur_route.get("avoid_families") or [])
                if str(f).strip()
            }
            excl_specs: list[str] = [
                str(s) for s in (cur_route.get("exclude_specs") or []) if str(s).strip()
            ]
            slot_group_key: tuple[str, str] | None = None
            if t.workflow_id and ("review_slot" in cur_route or t.stage in (
                StageKind.DESIGN_REVIEW,
                StageKind.IMPL_REVIEW,
                StageKind.RESULT_REVIEW,
                StageKind.BLIND_AUDIT,
            )):
                spref = re.sub(r"_s\d+$", "", t.step_key or t.stage.value.lower())
                slot_group_key = (t.workflow_id, spref)
                wf_obj = wfs_by_id.get(t.workflow_id)
                if wf_obj is not None:
                    qstate = wf_obj.state.get("review_quorum")
                    if isinstance(qstate, Mapping):
                        # Reset base avoid_fams to contributor_families + already-rebound sibling slots
                        # so rebinding slot 0 and slot 1 sequentially never deadlocks on old peer family!
                        avoid_fams = {
                            str(f).strip().lower()
                            for f in (qstate.get("contributor_families") or [])
                            if str(f).strip()
                        }
                        excl_specs = []
                avoid_fams |= sibling_families.get(slot_group_key, set())
                excl_specs.extend(sibling_specs.get(slot_group_key, []))

            is_rev_stage = t.stage in (
                StageKind.DESIGN_REVIEW,
                StageKind.IMPL_REVIEW,
                StageKind.RESULT_REVIEW,
                StageKind.BLIND_AUDIT,
            )
            wf_rebind = self.db.get_workflow(t.workflow_id) if t.workflow_id else None
            wf_cfg_rebind = dict(wf_rebind.config or {}) if wf_rebind is not None else None
            glob_pol_reb = self.db.get_global_config("routing_policy")
            glob_pol_nl_reb = self.db.get_global_config("routing_policy_nl", "")
            explicit_spec: str | None = None
            if cur_route.get("canonical_model") and not self._policy_targets_role(
                goal,
                hyp,
                resolved_role.value,
                is_review_stage=is_rev_stage,
                workflow_id=t.workflow_id,
                wf_config=wf_cfg_rebind,
                global_policy=glob_pol_reb,
                global_policy_nl=str(glob_pol_nl_reb or ""),
            ):
                pool_p = str(cur_route.get("pool") or "").strip()
                mod_p = str(cur_route.get("canonical_model") or "").strip()
                eff_p = str(cur_route.get("requested_effort") or "").strip()
                if pool_p and mod_p:
                    explicit_spec = f"{pool_p}:{mod_p}:{eff_p}" if eff_p else f"{pool_p}:{mod_p}"

            eval_goal = self._goal_for_role_resolution(
                goal,
                resolved_role,
                is_review_stage=is_rev_stage,
                has_avoid_families=bool(avoid_fams),
            )
            try:
                import inspect as _inspect_reb

                rr_sig = _inspect_reb.signature(router.resolve_route).parameters
                rr_extra: dict[str, Any] = {}
                if "db" in rr_sig:
                    rr_extra["db"] = self.db
                if "workflow_id" in rr_sig:
                    rr_extra["workflow_id"] = t.workflow_id
                if "task" in rr_sig:
                    rr_extra["task"] = t
                if "wf_config" in rr_sig and wf_cfg_rebind is not None:
                    rr_extra["wf_config"] = wf_cfg_rebind
                new_route = router.resolve_route(
                    resolved_role,
                    goal=eval_goal,
                    hyp=hyp,
                    explicit_spec=explicit_spec,
                    avoid_families=sorted(avoid_fams),
                    prefer_avoid_families=list(cur_route.get("prefer_avoid_families") or []),
                    exclude_specs=excl_specs,
                    **rr_extra,
                )
            except PolicyBlockedError as exc:
                if exc.waitable:
                    continue
                self.db.transition_task_status(
                    t.task_id,
                    "READY",
                    exc.code or "BLOCKED_POLICY",
                    finished_ts=now,
                    now=now,
                )
                blocked_ids.append(t.task_id)
                continue

            if slot_group_key is not None:
                sibling_families.setdefault(slot_group_key, set()).add(new_route.family.lower())
                sibling_specs.setdefault(slot_group_key, []).append(new_route.canonical_spec())

            same_route = (
                isinstance(cur_route, Mapping)
                and cur_route.get("pool") == new_route.pool
                and cur_route.get("canonical_model") == new_route.canonical_model
                and cur_route.get("requested_effort") == new_route.requested_effort
                and cur_route.get("wire_effort") == new_route.wire_effort
            )
            if same_route:
                continue

            self.db.transition_task_status(
                t.task_id,
                "READY",
                "CANCELLED",
                finished_ts=now,
                now=now,
            )
            cancelled_ids.append(t.task_id)
            merged_route = {
                **(dict(cur_route) if isinstance(cur_route, Mapping) else {}),
                **new_route.to_dict(),
                "role": resolved_role.value,
                "avoid_families": sorted(avoid_fams),
                "exclude_specs": list(excl_specs),
            }
            repl = self._spawn_replacement_task_from_snapshot(
                goal,
                t.to_dict(),
                pinned_route=merged_route,
                now=now,
            )
            replaced_ids.append(repl.task_id)
        return cancelled_ids, replaced_ids, blocked_ids

    def _wake_reactor(self, event: str, **fields: Any) -> None:
        if not self.notify_reactor:
            return
        _notify_reactor(
            self.db.arr_home,
            {"event": event, "ts": time.time(), **fields},
        )

    def apply_intervention(
        self,
        action: str,
        *,
        scope: str = "goal",
        goal_id: str | None = None,
        target_id: str | None = None,
        payload: Mapping[str, Any] | None = None,
        actor: str = "user",
        reason: str = "",
    ) -> dict[str, Any]:
        """Apply a human/operator intervention atomically across global, goal, branch, workflow, or task scope."""
        act = str(action or "").strip().lower()
        scp = str(scope or "").strip().lower()
        body = dict(payload or {})
        before_epoch: int | None = None
        after_epoch: int | None = None
        result: dict[str, Any]

        if scp == "global":
            if act == "pause_all":
                paused: list[str] = []
                cancelled_tasks_all: list[str] = []
                cancelled_hq_all: list[int] = []
                for rec in self.db.list_goal_records(status="ACTIVE"):
                    g = rec["goal"]
                    p_res, _, _ = self._pause_single_goal(
                        g,
                        actor=actor,
                        reason=reason,
                        cancel_running=bool(body.get("cancel_running", True)),
                    )
                    paused.append(g.goal_id)
                    cancelled_tasks_all.extend(p_res["cancelled_task_ids"])
                    cancelled_hq_all.extend(p_res["cancelled_hq_job_ids"])
                result = {
                    "paused_goal_ids": paused,
                    "cancelled_task_ids": cancelled_tasks_all,
                    "cancelled_hq_job_ids": cancelled_hq_all,
                }

            elif act == "resume_all":
                resumed: list[str] = []
                spawned_all: list[str] = []
                for rec in self.db.list_goal_records(status="PAUSED"):
                    g = rec["goal"]
                    r_res, _, _ = self._resume_single_goal(g)
                    resumed.append(g.goal_id)
                    spawned_all.extend(r_res["spawned_replacement_task_ids"])
                result = {
                    "resumed_goal_ids": resumed,
                    "spawned_replacement_task_ids": spawned_all,
                }

            elif act in ("set_routing_policy", "set_model_policy"):
                raw_pol = (
                    body.get("policy")
                    or body.get("routing_policy")
                    or body.get("policy_nl")
                    or body.get("routing_policy_nl")
                    or body
                )
                compiled = ModelRouter.compile_routing_policy(
                    raw_pol,
                    project=str(body.get("project") or ""),
                )
                self.db.set_global_config("routing_policy", compiled)
                router_glob = ModelRouter(db=self.db)
                c_all: list[str] = []
                r_all: list[str] = []
                b_all: list[str] = []
                for g_rec in self.db.list_goal_records(status="ACTIVE"):
                    g_act = g_rec["goal"]
                    g_tasks = self.db.list_tasks(goal_id=g_act.goal_id)
                    c_ids, r_ids, b_ids = self._surgically_rebind_ready_tasks_for_policy(
                        g_act, g_tasks, router=router_glob, actor=actor, reason=reason
                    )
                    c_all.extend(c_ids)
                    r_all.extend(r_ids)
                    b_all.extend(b_ids)
                result = {
                    "scope": "global",
                    "routing_policy": compiled,
                    "rules_count": len(compiled.get("rules", [])),
                    "cancelled_task_ids": c_all,
                    "spawned_replacement_task_ids": r_all,
                    "blocked_task_ids": b_all,
                }

            elif act == "rebalance_goals":
                updates_by_goal: dict[str, dict[str, Any]] = {}
                for key_alias, field_name in (
                    ("weights", "priority_weight"),
                    ("priorities", "priority_weight"),
                    ("min_slots", "min_guaranteed_slots"),
                    ("min_guaranteed_slots", "min_guaranteed_slots"),
                    ("budgets", "max_tokens_budget"),
                    ("max_tokens_budget", "max_tokens_budget"),
                ):
                    mapping = body.get(key_alias)
                    if isinstance(mapping, Mapping):
                        for gid, val in mapping.items():
                            updates_by_goal.setdefault(str(gid), {})[field_name] = val
                if isinstance(body.get("goals"), Mapping):
                    for gid, patch_map in body["goals"].items():
                        if not isinstance(patch_map, Mapping):
                            raise ValueError(f"Invalid rebalance patch for {gid}")
                        updates_by_goal.setdefault(str(gid), {}).update(patch_map)
                if not updates_by_goal:
                    raise ValueError("rebalance_goals requires goal weight/slot/budget updates in payload")

                updated_goals: list[dict[str, Any]] = []
                for gid, gpatch in updates_by_goal.items():
                    g = self._require_goal(gid)
                    merged = g.to_dict()
                    merged.update(gpatch)
                    merged["human_epoch"] = int(g.human_epoch) + 1
                    new_g = ResearchGoal.from_dict(merged)
                    _validate_goal_fields(new_g)
                    self.db.save_goal(new_g)
                    now_sync = time.time()
                    for t in self.db.list_tasks(goal_id=gid):
                        if t.status in ACTIVE_TASK_STATUSES:
                            try:
                                self.db.transition_task_status(
                                    t.task_id,
                                    ACTIVE_TASK_STATUSES,
                                    t.status,
                                    human_epoch=new_g.human_epoch,
                                    now=now_sync,
                                )
                            except StateTransitionError:
                                continue
                    updated_goals.append(
                        {
                            "goal_id": gid,
                            "priority_weight": new_g.priority_weight,
                            "min_guaranteed_slots": new_g.min_guaranteed_slots,
                            "max_tokens_budget": new_g.max_tokens_budget,
                            "human_epoch": new_g.human_epoch,
                        }
                    )
                result = {"updated_goals": updated_goals}

            elif act == "set_global_budget":
                updated_keys: dict[str, Any] = {}
                tok_val = body.get(
                    "global_max_tokens_budget", body.get("max_tokens_budget")
                )
                if tok_val is not None:
                    res = self.set_global_config("global_max_tokens_budget", tok_val)
                    updated_keys["global_max_tokens_budget"] = res["value"]
                res_slots = body.get("foreground_reserve_slots")
                if res_slots is not None:
                    res = self.set_global_config("foreground_reserve_slots", res_slots)
                    updated_keys["foreground_reserve_slots"] = res["value"]
                if not updated_keys:
                    raise ValueError(
                        "set_global_budget requires global_max_tokens_budget or foreground_reserve_slots"
                    )
                result = {"global_config": updated_keys}

            else:
                raise ValueError(f"Unsupported global intervention action: {action!r}")

        elif scp == "goal":
            goal = self._require_goal(goal_id or target_id)
            eff_goal_id = goal.goal_id
            cur_status = self.db.get_goal_status(eff_goal_id)

            if act == "pause":
                result, before_epoch, after_epoch = self._pause_single_goal(
                    goal,
                    actor=actor,
                    reason=reason,
                    cancel_running=bool(body.get("cancel_running", True)),
                )

            elif act == "resume":
                if cur_status == "STOPPED" and not body.get("force", False):
                    raise StateTransitionError(
                        f"Cannot resume STOPPED goal {eff_goal_id} without force=True"
                    )
                result, before_epoch, after_epoch = self._resume_single_goal(goal)

            elif act == "stop":
                _, before_epoch, after_epoch = self._bump_goal_epoch(goal, status="STOPPED")
                tasks = self.db.list_tasks(goal_id=eff_goal_id)
                cancelled_tasks, cancelled_hq = self._cancel_tasks_and_hq(
                    tasks, reason=reason or f"goal {eff_goal_id} stopped by {actor}"
                )
                cancelled_wfs: list[str] = []
                for wf in self.db.list_workflows(goal_id=eff_goal_id):
                    if wf.status in ACTIVE_WORKFLOW_STATUSES:
                        self.db.update_workflow_state(wf.workflow_id, status="CANCELLED")
                        cancelled_wfs.append(wf.workflow_id)
                cancelled_dreqs, dec_hq_ids = self._cancel_decisions_and_hq(
                    eff_goal_id,
                    reason=reason or f"goal {eff_goal_id} stopped by {actor}",
                    epoch=after_epoch,
                    plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                )
                cancelled_hq.extend(dec_hq_ids)
                result = {
                    "goal_id": eff_goal_id,
                    "status": "STOPPED",
                    "cancelled_task_ids": cancelled_tasks,
                    "cancelled_hq_job_ids": cancelled_hq,
                    "cancelled_workflow_ids": cancelled_wfs,
                    "cancelled_decision_requests": cancelled_dreqs,
                }

            elif act == "update_objective":
                patch = dict(body.get("patch")) if isinstance(body.get("patch"), Mapping) else body
                cancel_stale = bool(patch.pop("cancel_stale_tasks", True))
                impact = self.analyze_version_impact(eff_goal_id, patch)
                before_epoch = int(goal.human_epoch)
                after_epoch = before_epoch + 1
                next_version = int(goal.version) + 1
                merged = goal.to_dict()
                merged.update(patch)
                merged["version"] = next_version
                merged["human_epoch"] = after_epoch
                updated_goal = ResearchGoal.from_dict(merged)
                _validate_goal_fields(updated_goal)

                self.db.save_goal(updated_goal)
                self.db.save_goal_version(
                    updated_goal,
                    change_summary=f"update_objective: {', '.join(impact['changed_fields'])}",
                    actor=actor,
                    reason=reason,
                )

                cancelled_tasks: list[str] = []
                cancelled_hq: list[int] = []
                if cancel_stale and (impact["objective_changed"] or impact["scope_changed"]):
                    stale_tasks = [
                        t
                        for t in self.db.list_tasks(goal_id=eff_goal_id)
                        if t.task_id in set(impact["affected_task_ids"])
                    ]
                    cancelled_tasks, cancelled_hq = self._cancel_tasks_and_hq(
                        stale_tasks,
                        reason=reason or f"stale under goal version v{next_version}",
                    )

                recheck = self.recheck_conclusions_against_version(
                    updated_goal, persist=True
                )
                result = {
                    "goal_id": eff_goal_id,
                    "version": next_version,
                    "impact": impact,
                    "recheck": recheck,
                    "cancelled_task_ids": cancelled_tasks,
                    "cancelled_hq_job_ids": cancelled_hq,
                }

            elif act == "update_budget":
                merged = goal.to_dict()
                changed = False
                if "max_tokens_budget" in body:
                    val = int(body["max_tokens_budget"])
                    if val <= 0:
                        raise ValueError("max_tokens_budget must be > 0")
                    merged["max_tokens_budget"] = val
                    changed = True
                if "wall_budget_sec" in body:
                    merged["wall_budget_sec"] = int(body["wall_budget_sec"])
                    changed = True
                if "max_concurrent_branches" in body:
                    merged["max_concurrent_branches"] = int(body["max_concurrent_branches"])
                    changed = True
                if not changed:
                    raise ValueError(
                        "update_budget requires max_tokens_budget, wall_budget_sec, or max_concurrent_branches"
                    )
                before_epoch = int(goal.human_epoch)
                after_epoch = before_epoch + 1
                merged["human_epoch"] = after_epoch
                updated_goal = ResearchGoal.from_dict(merged)
                _validate_goal_fields(updated_goal)
                self.db.save_goal(updated_goal)
                unaffected_ids = {
                    t.task_id
                    for t in self.db.list_tasks(goal_id=eff_goal_id)
                    if t.status in ACTIVE_TASK_STATUSES
                }
                for tid in unaffected_ids:
                    try:
                        cur_t = self.db.get_task(tid)
                        if cur_t is not None and cur_t.status in ACTIVE_TASK_STATUSES:
                            self.db.transition_task_status(
                                tid, ACTIVE_TASK_STATUSES, cur_t.status, human_epoch=after_epoch
                            )
                    except StateTransitionError:
                        continue
                result = {
                    "goal_id": eff_goal_id,
                    "max_tokens_budget": updated_goal.max_tokens_budget,
                    "wall_budget_sec": updated_goal.wall_budget_sec,
                    "max_concurrent_branches": updated_goal.max_concurrent_branches,
                }

            elif act in ("set_priority", "focus"):
                merged = goal.to_dict()
                changed = False
                if "priority_weight" in body:
                    merged["priority_weight"] = float(body["priority_weight"])
                    changed = True
                elif act == "focus":
                    merged["priority_weight"] = max(float(goal.priority_weight) + 1.0, 2.0)
                    changed = True
                if "min_guaranteed_slots" in body:
                    merged["min_guaranteed_slots"] = int(body["min_guaranteed_slots"])
                    changed = True
                if not changed:
                    raise ValueError("set_priority requires priority_weight or min_guaranteed_slots")
                before_epoch = int(goal.human_epoch)
                after_epoch = before_epoch + 1
                merged["human_epoch"] = after_epoch
                updated_goal = ResearchGoal.from_dict(merged)
                _validate_goal_fields(updated_goal)
                self.db.save_goal(updated_goal)
                for cur_t in self.db.list_tasks(goal_id=eff_goal_id):
                    if cur_t.status in ACTIVE_TASK_STATUSES:
                        try:
                            self.db.transition_task_status(
                                cur_t.task_id, ACTIVE_TASK_STATUSES, cur_t.status, human_epoch=after_epoch
                            )
                        except StateTransitionError:
                            continue
                result = {
                    "goal_id": eff_goal_id,
                    "priority_weight": updated_goal.priority_weight,
                    "min_guaranteed_slots": updated_goal.min_guaranteed_slots,
                }

            elif act == "set_autonomy":
                mode = str(body.get("autonomy_mode") or "").strip()
                if mode not in VALID_AUTONOMY_MODES:
                    raise ValueError(
                        f"Invalid autonomy_mode {mode!r}; expected one of {sorted(VALID_AUTONOMY_MODES)}"
                    )
                before_epoch = int(goal.human_epoch)
                after_epoch = before_epoch + 1
                updated_goal = ResearchGoal.from_dict(
                    {
                        **goal.to_dict(),
                        "autonomy_mode": mode,
                        "human_epoch": after_epoch,
                    }
                )
                self.db.save_goal(updated_goal)
                if mode != "paused":
                    for cur_t in self.db.list_tasks(goal_id=eff_goal_id):
                        if cur_t.status in ACTIVE_TASK_STATUSES:
                            try:
                                self.db.transition_task_status(
                                    cur_t.task_id, ACTIVE_TASK_STATUSES, cur_t.status, human_epoch=after_epoch
                                )
                            except StateTransitionError:
                                continue
                result = {"goal_id": eff_goal_id, "autonomy_mode": mode}

            elif act in ("set_routing_policy", "set_model_policy"):
                raw_pol = (
                    body.get("policy")
                    or body.get("routing_policy")
                    or body.get("policy_nl")
                    or body.get("routing_policy_nl")
                    or body
                )
                compiled = ModelRouter.compile_routing_policy(
                    raw_pol,
                    project=str(body.get("project") or goal.project),
                )
                before_epoch = int(goal.human_epoch)
                after_epoch = before_epoch + 1
                nl_str = (
                    raw_pol
                    if isinstance(raw_pol, str)
                    else str(compiled.get("source_nl") or goal.routing_policy_nl)
                )
                updated_goal = ResearchGoal.from_dict(
                    {
                        **goal.to_dict(),
                        "human_epoch": after_epoch,
                        "routing_policy_nl": nl_str,
                        "routing_policy": compiled,
                    }
                )
                self.db.save_goal(updated_goal)
                all_goal_tasks = self.db.list_tasks(goal_id=eff_goal_id)
                for cur_t in all_goal_tasks:
                    if cur_t.status in ACTIVE_TASK_STATUSES:
                        try:
                            self.db.transition_task_status(
                                cur_t.task_id, ACTIVE_TASK_STATUSES, cur_t.status, human_epoch=after_epoch
                            )
                        except StateTransitionError:
                            continue
                router = ModelRouter(db=self.db)
                refreshed_tasks = self.db.list_tasks(goal_id=eff_goal_id)
                cancelled_ready, replaced_ready, blocked_ready = (
                    self._surgically_rebind_ready_tasks_for_policy(
                        updated_goal,
                        refreshed_tasks,
                        router=router,
                        actor=actor,
                        reason=reason or "set_routing_policy",
                    )
                )
                result = {
                    "goal_id": eff_goal_id,
                    "routing_policy_nl": nl_str,
                    "routing_policy": compiled,
                    "rules_count": len(compiled.get("rules", [])),
                    "cancelled_task_ids": cancelled_ready,
                    "spawned_replacement_task_ids": replaced_ready,
                    "blocked_task_ids": blocked_ready,
                }

            elif act == "update_plan":
                with self.db._tx():
                    unaffected_hids = {
                        str(x).strip()
                        for x in (body.get("unaffected_hypothesis_ids") or [])
                        if str(x).strip()
                    }
                    all_goal_hyps = self.db.list_hypotheses(goal_id=eff_goal_id)
                    affected_goal_hids = {
                        h.hypothesis_id for h in all_goal_hyps if h.hypothesis_id not in unaffected_hids
                    }
                    affected_syn_wfs = self._collect_affected_synthesis_workflows(
                        eff_goal_id, affected_goal_hids
                    )
                    affected_syn_wf_ids = {w.workflow_id for w in affected_syn_wfs}
                    all_goal_tasks = self.db.list_tasks(goal_id=eff_goal_id)
                    superseded_tasks = [
                        t
                        for t in all_goal_tasks
                        if t.status in ACTIVE_TASK_STATUSES
                        and (
                            t.hypothesis_id not in unaffected_hids
                            or (t.workflow_id is not None and t.workflow_id in affected_syn_wf_ids)
                        )
                    ]
                    unaffected_task_ids = {
                        t.task_id
                        for t in all_goal_tasks
                        if t.status in ACTIVE_TASK_STATUSES
                        and t.hypothesis_id in unaffected_hids
                        and (t.workflow_id is None or t.workflow_id not in affected_syn_wf_ids)
                    }
                    updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                        goal,
                        bump_plan_epoch=True,
                        sync_unaffected_task_ids=unaffected_task_ids,
                    )
                    next_plan_epoch = int(updated_goal.plan_epoch)
                    c_tasks, c_hq = self._cancel_tasks_and_hq(
                        superseded_tasks,
                        reason=reason or f"update_plan:pe{next_plan_epoch}:{actor}",
                        epoch=after_epoch,
                        plan_epoch=next_plan_epoch,
                    )
                    c_decs, dec_hq = self._cancel_decisions_and_hq(
                        eff_goal_id,
                        reason=reason or f"update_plan:pe{next_plan_epoch}:{actor}",
                        epoch=after_epoch,
                        plan_epoch=next_plan_epoch,
                    )
                    c_hq.extend(dec_hq)
                    cancelled_wfs = self._mark_synthesis_workflows_superseded(
                        updated_goal,
                        affected_syn_wfs,
                        reason=reason or f"update_plan:pe{next_plan_epoch}:{actor}",
                        plan_epoch=next_plan_epoch,
                    )
                    for wf_obj in self.db.list_workflows(goal_id=eff_goal_id):
                        if (
                            wf_obj.status in ACTIVE_WORKFLOW_STATUSES
                            and wf_obj.hypothesis_id not in unaffected_hids
                            and wf_obj.workflow_id not in affected_syn_wf_ids
                        ):
                            self.db.update_workflow_state(
                                wf_obj.workflow_id,
                                status="CANCELLED",
                                state_patch={"superseded_by_plan_epoch": next_plan_epoch},
                            )
                            cancelled_wfs.append(wf_obj.workflow_id)

                    preserved_rcpts = [
                        r.receipt_id
                        for r in self.db.list_receipts(goal_id=eff_goal_id)
                        if r.process_status == ProcessStatus.EXIT_OK
                        and r.engineering_verdict == EngineeringVerdict.PASS
                        and r.audit_verdict != AuditVerdict.FAIL
                        and r.evidence_validity
                        not in (
                            EvidenceValidity.SUPERSEDED_EPOCH,
                            EvidenceValidity.HUMAN_INVALIDATED,
                            EvidenceValidity.STALE_GOAL_VERSION,
                        )
                        and (
                            r.stage in (StageKind.CODE, StageKind.CPU_GATE)
                            or r.hypothesis_id in unaffected_hids
                        )
                    ]
                    for h_obj in all_goal_hyps:
                        if h_obj.hypothesis_id in affected_goal_hids:
                            h_pres = [
                                r.receipt_id
                                for r in self.db.list_receipts(goal_id=eff_goal_id, hypothesis_id=h_obj.hypothesis_id)
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
                            cur_sci_rev_g = int((h_obj.params or {}).get("lineage_sci_revision", 0) or 0)
                            self.db.update_hypothesis_params(
                                h_obj.hypothesis_id,
                                {
                                    "lineage_plan_epoch": next_plan_epoch,
                                    "lineage_human_epoch": after_epoch,
                                    "lineage_sci_revision": cur_sci_rev_g + 1,
                                    "preserved_receipt_ids": h_pres,
                                },
                            )
                    self.db.upsert_knowledge_item(
                        eff_goal_id,
                        "plan_lineage",
                        f"pe:{next_plan_epoch}",
                        {
                            "plan_epoch": next_plan_epoch,
                            "human_epoch": after_epoch,
                            "preserved_upstream_receipt_ids": preserved_rcpts,
                            "superseded_task_ids": c_tasks,
                            "superseded_workflow_ids": cancelled_wfs,
                            "superseded_decision_ids": c_decs,
                        },
                        goal_version=int(updated_goal.version),
                    )
                    rplan = ResearchPlan(
                        plan_id=str(body.get("plan_id") or f"rplan_{uuid.uuid4().hex[:10]}"),
                        goal_id=eff_goal_id,
                        goal_version=int(updated_goal.version),
                        human_epoch=after_epoch,
                        plan_epoch=next_plan_epoch,
                        thesis=str(body.get("thesis") or updated_goal.title),
                        uncertainties=list(body.get("uncertainties") or []),
                        competing_routes=list(body.get("competing_routes") or []),
                        milestone_criteria=[str(x) for x in (body.get("milestone_criteria") or [])],
                        status=str(body.get("status") or "APPROVED"),
                    )
                    saved_plan = self.db.save_research_plan(rplan, supersede_previous=True)
                    result = {
                        "goal_id": eff_goal_id,
                        "plan_id": saved_plan.plan_id,
                        "plan_version": saved_plan.plan_version,
                        "plan_epoch": next_plan_epoch,
                        "cancelled_task_ids": c_tasks,
                        "cancelled_hq_job_ids": c_hq,
                        "cancelled_workflow_ids": cancelled_wfs,
                        "cancelled_decision_ids": c_decs,
                        "preserved_upstream_receipt_ids": preserved_rcpts,
                    }

            elif act == "replan_from_node":
                hid = str(target_id or body.get("hypothesis_id") or body.get("node_id") or "").strip()
                if not hid:
                    raise ValueError("replan_from_node requires target_id or hypothesis_id")
                result, before_epoch, after_epoch = self._replan_from_node_impl(
                    eff_goal_id, hid, body, actor=actor, reason=reason
                )

            elif act == "override_verdict":
                hid = str(target_id or body.get("hypothesis_id") or "").strip()
                if not hid:
                    raise ValueError("override_verdict requires target_id or hypothesis_id")
                result, before_epoch, after_epoch = self._override_verdict_impl(
                    eff_goal_id, hid, body, actor=actor, reason=reason
                )

            elif act in ("inject_guidance", "inject"):
                guidance_text = str(
                    body.get("guidance") or body.get("note") or body.get("message") or reason
                ).strip()
                if not guidance_text:
                    raise ValueError("inject_guidance requires non-empty guidance text")
                all_ids = {
                    t.task_id
                    for t in self.db.list_tasks(goal_id=eff_goal_id)
                    if t.status in ACTIVE_TASK_STATUSES
                }
                updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=all_ids
                )
                k_item = self.db.upsert_knowledge_item(
                    eff_goal_id,
                    "human_guidance",
                    f"guidance:e{after_epoch}",
                    {"guidance": guidance_text, "actor": actor, "epoch": after_epoch},
                    goal_version=int(updated_goal.version),
                )
                dreq_id: str | None = None
                if bool(body.get("enqueue_replan", False)):
                    dreq = self.db.enqueue_decision_request(
                        eff_goal_id,
                        role=ScientificRole.RESEARCH_LEAD,
                        tier="L1",
                        trigger_kind="HUMAN_GUIDANCE",
                        goal_version=int(updated_goal.version),
                        human_epoch=after_epoch,
                        coalesce_key=f"{eff_goal_id}:guidance:e{after_epoch}",
                        coalesce_window_s=0.0,
                        payload={"guidance": guidance_text, "actor": actor},
                    )
                    dreq_id = str(dreq["req_id"])
                result = {
                    "goal_id": eff_goal_id,
                    "knowledge_item_id": k_item["item_id"],
                    "decision_req_id": dreq_id,
                }

            else:
                raise ValueError(f"Unsupported goal intervention action: {action!r}")

        elif scp in ("branch", "workflow", "task"):
            if act == "prune_branch":
                hid = str(target_id or body.get("hypothesis_id") or "").strip()
                if not hid:
                    raise ValueError("prune_branch requires target_id (hypothesis_id)")
                hyp = self.db.get_hypothesis(hid)
                if hyp is None:
                    raise KeyError(f"Unknown hypothesis_id: {hid}")
                goal = self._require_goal(goal_id or hyp.goal_id)
                if hyp.goal_id != goal.goal_id:
                    raise ValueError(f"Hypothesis {hid} does not belong to goal {goal.goal_id}")

                to_prune = [hyp] + self._collect_hyp_descendants(goal.goal_id, hid)
                pruned_ids: list[str] = []
                cancelled_tasks: list[str] = []
                cancelled_hq: list[int] = []
                cancelled_wfs: list[str] = []

                for node in to_prune:
                    if node.status not in TERMINAL_HYP_STATUSES:
                        verdict = (
                            node.scientific_verdict
                            if node.scientific_verdict in (ScientificVerdict.REFUTED, ScientificVerdict.VALID_NEGATIVE)
                            else ScientificVerdict.NOT_APPLICABLE
                        )
                        self.db.transition_hypothesis_status(
                            node.hypothesis_id,
                            node.status,
                            "PRUNED",
                            scientific_verdict=verdict,
                        )
                        pruned_ids.append(node.hypothesis_id)
                    t_list = self.db.list_tasks(
                        goal_id=goal.goal_id, hypothesis_id=node.hypothesis_id
                    )
                    ct, chq = self._cancel_tasks_and_hq(
                        t_list, reason=reason or f"branch {hid} pruned by {actor}"
                    )
                    cancelled_tasks.extend(ct)
                    cancelled_hq.extend(chq)
                    for wf in self.db.list_workflows(
                        goal_id=goal.goal_id, hypothesis_id=node.hypothesis_id
                    ):
                        if wf.status in ACTIVE_WORKFLOW_STATUSES:
                            self.db.update_workflow_state(wf.workflow_id, status="PRUNED")
                            cancelled_wfs.append(wf.workflow_id)

                pruned_set = {n.hypothesis_id for n in to_prune}
                affected_syn_wfs = self._collect_affected_synthesis_workflows(goal.goal_id, pruned_set)
                affected_syn_wf_ids = {w.workflow_id for w in affected_syn_wfs}
                if affected_syn_wf_ids:
                    syn_tasks = [
                        t
                        for t in self.db.list_tasks(goal_id=goal.goal_id)
                        if t.workflow_id in affected_syn_wf_ids and t.status in ACTIVE_TASK_STATUSES
                    ]
                    ct_s, chq_s = self._cancel_tasks_and_hq(
                        syn_tasks, reason=reason or f"branch {hid} pruned invalidating synthesis"
                    )
                    cancelled_tasks.extend(ct_s)
                    cancelled_hq.extend(chq_s)
                    cancelled_wfs.extend(
                        self._mark_synthesis_workflows_superseded(
                            goal,
                            affected_syn_wfs,
                            reason=reason or f"branch {hid} pruned",
                            plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                        )
                    )

                _, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal,
                    sync_unaffected_task_ids={
                        t.task_id
                        for t in self.db.list_tasks(goal_id=goal.goal_id)
                        if t.status in ACTIVE_TASK_STATUSES
                        and t.hypothesis_id not in pruned_set
                        and (t.workflow_id is None or t.workflow_id not in affected_syn_wf_ids)
                    },
                )
                result = {
                    "goal_id": goal.goal_id,
                    "pruned_hypothesis_ids": pruned_ids,
                    "cancelled_task_ids": cancelled_tasks,
                    "cancelled_hq_job_ids": cancelled_hq,
                    "cancelled_workflow_ids": cancelled_wfs,
                }

            elif act in ("focus_branch", "focus"):
                hid = str(target_id or body.get("hypothesis_id") or "").strip()
                if not hid:
                    raise ValueError("focus_branch requires target_id (hypothesis_id)")
                hyp = self.db.get_hypothesis(hid)
                if hyp is None:
                    raise KeyError(f"Unknown hypothesis_id: {hid}")
                if hyp.status in TERMINAL_HYP_STATUSES:
                    raise StateTransitionError(
                        f"Cannot focus terminal hypothesis {hid} ({hyp.status})"
                    )
                goal = self._require_goal(goal_id or hyp.goal_id)
                new_evoi = float(
                    body.get("priority_evoi", max(float(hyp.priority_evoi) + 1.0, 2.0))
                )
                if not math.isfinite(new_evoi) or new_evoi <= 0:
                    raise ValueError("priority_evoi must be finite and > 0")
                self.db.update_hypothesis_priority(hid, new_evoi)
                self.db.update_hypothesis_params(
                    hid, {"human_focused": True, "focus_reason": reason}
                )
                all_active = {
                    t.task_id
                    for t in self.db.list_tasks(goal_id=goal.goal_id)
                    if t.status in ACTIVE_TASK_STATUSES
                }
                _, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=all_active
                )
                result = {
                    "goal_id": goal.goal_id,
                    "hypothesis_id": hid,
                    "priority_evoi": new_evoi,
                }

            elif act == "inject_hypothesis":
                goal = self._require_goal(goal_id)
                if not body:
                    raise ValueError("inject_hypothesis requires hypothesis spec in payload")
                hid = str(body.get("hypothesis_id") or f"hyp_{uuid.uuid4().hex[:10]}")
                kind = HypothesisKind(body.get("kind", HypothesisKind.COMPETING_ROUTE.value))
                fidelity = FidelityRung(body.get("fidelity", FidelityRung.F1_FAST_SCREEN.value))
                writable_paths = [
                    str(p) for p in (body.get("writable_paths") or goal.writable_prefixes)
                ]
                banned_chk = check_no_banned_access(
                    paths=writable_paths,
                    extra_bans=goal.banned_paths,
                )
                if not banned_chk.passed:
                    raise ValueError(f"Banned writable_paths: {banned_chk.failures}")
                scope_chk = validate_writable_paths(
                    writable_paths,
                    writable_paths,
                    goal_prefixes=goal.writable_prefixes,
                )
                if not scope_chk.passed:
                    raise ValueError(f"Invalid writable_paths: {scope_chk.failures}")

                title = str(body.get("title") or "").strip()
                mechanism = str(body.get("mechanism") or title).strip()
                if not title or not mechanism:
                    raise ValueError("inject_hypothesis requires non-empty title and mechanism")
                params = {
                    **dict(body.get("params") or {}),
                    "human_injected": True,
                    "goal_version": int(goal.version),
                }
                screen_cmd = str(body.get("screen_cmd") or "pytest -q")
                fp = str(body.get("fingerprint") or "").strip() or compute_hypothesis_fingerprint(
                    goal_id=goal.goal_id,
                    kind=kind,
                    mechanism=mechanism,
                    params=params,
                    writable_paths=writable_paths,
                    screen_cmd=screen_cmd,
                )
                node = HypothesisNode(
                    hypothesis_id=hid,
                    goal_id=goal.goal_id,
                    parent_id=body.get("parent_id"),
                    kind=kind,
                    title=title,
                    statement=str(body.get("statement") or title),
                    mechanism=mechanism,
                    fidelity=fidelity,
                    priority_evoi=float(body.get("priority_evoi", 1.5)),
                    writable_paths=writable_paths,
                    params=params,
                    kill_criteria=[
                        str(x)
                        for x in (body.get("kill_criteria") or goal.kill_criteria or ["tests_passed == 0"])
                    ],
                    acceptance_cmd=str(body.get("acceptance_cmd") or screen_cmd),
                    screen_cmd=screen_cmd,
                    confirm_cmd=body.get("confirm_cmd", screen_cmd),
                    branch_name=str(body.get("branch_name") or f"arr/{goal.project}/{hid}"),
                    base_sha=str(body.get("base_sha") or goal.baseline_sha),
                    status="PENDING",
                    fingerprint=fp,
                )
                stored_node, created = self.db.register_hypothesis(
                    node, allow_graveyard=bool(body.get("allow_graveyard", False))
                )
                _, before_epoch, after_epoch = self._bump_goal_epoch(goal)
                result = {
                    "goal_id": goal.goal_id,
                    "hypothesis_id": stored_node.hypothesis_id,
                    "created": created,
                }

            elif act == "redesign_experiment":
                raw_target = str(target_id or body.get("hypothesis_id") or "").strip()
                if not raw_target:
                    raise ValueError("redesign_experiment requires target_id (hypothesis_id or workflow_id)")
                old_hyp = self.db.get_hypothesis(raw_target)
                if old_hyp is None and scp == "workflow":
                    wf = self.db.get_workflow(raw_target)
                    if wf is not None:
                        old_hyp = self.db.get_hypothesis(wf.hypothesis_id)
                if old_hyp is None:
                    raise KeyError(f"Unknown hypothesis_id: {raw_target}")
                hid = old_hyp.hypothesis_id
                goal = self._require_goal(goal_id or old_hyp.goal_id)

                old_tasks = self.db.list_tasks(goal_id=goal.goal_id, hypothesis_id=hid)
                cancelled_tasks, cancelled_hq = self._cancel_tasks_and_hq(
                    old_tasks,
                    reason=reason or f"hypothesis {hid} redesigned by {actor}",
                )
                cancelled_wfs: list[str] = []
                for wf in self.db.list_workflows(goal_id=goal.goal_id, hypothesis_id=hid):
                    if wf.status in ACTIVE_WORKFLOW_STATUSES:
                        self.db.update_workflow_state(wf.workflow_id, status="CANCELLED")
                        cancelled_wfs.append(wf.workflow_id)
                if old_hyp.status not in TERMINAL_HYP_STATUSES:
                    self.db.transition_hypothesis_status(
                        old_hyp.hypothesis_id,
                        old_hyp.status,
                        "PRUNED",
                        scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
                    )

                new_hid = str(body.get("new_hypothesis_id") or f"hyp_{uuid.uuid4().hex[:10]}")
                new_mech = str(body.get("mechanism") or f"redesigned::{old_hyp.mechanism}")
                new_params = {
                    **dict(old_hyp.params),
                    **dict(body.get("params") or {}),
                    "redesigned_from": old_hyp.hypothesis_id,
                    "human_redesigned": True,
                    "goal_version": int(goal.version),
                }
                new_wp = [str(p) for p in (body.get("writable_paths") or old_hyp.writable_paths)]
                new_screen = str(body.get("screen_cmd") or old_hyp.screen_cmd)
                new_fp = compute_hypothesis_fingerprint(
                    goal_id=goal.goal_id,
                    kind=HypothesisKind(body.get("kind", old_hyp.kind.value)),
                    mechanism=new_mech,
                    params=new_params,
                    writable_paths=new_wp,
                    screen_cmd=new_screen,
                )
                new_node = HypothesisNode(
                    hypothesis_id=new_hid,
                    goal_id=goal.goal_id,
                    parent_id=old_hyp.hypothesis_id,
                    kind=HypothesisKind(body.get("kind", old_hyp.kind.value)),
                    title=str(body.get("title") or f"Redesigned: {old_hyp.title}"),
                    statement=str(body.get("statement") or old_hyp.statement),
                    mechanism=new_mech,
                    fidelity=FidelityRung(body.get("fidelity", old_hyp.fidelity.value)),
                    priority_evoi=float(body.get("priority_evoi", max(float(old_hyp.priority_evoi), 1.5))),
                    writable_paths=new_wp,
                    params=new_params,
                    kill_criteria=[str(x) for x in (body.get("kill_criteria") or old_hyp.kill_criteria)],
                    acceptance_cmd=str(body.get("acceptance_cmd") or old_hyp.acceptance_cmd),
                    screen_cmd=new_screen,
                    confirm_cmd=body.get("confirm_cmd", old_hyp.confirm_cmd),
                    branch_name=str(body.get("branch_name") or f"arr/{goal.project}/{new_hid}"),
                    base_sha=str(body.get("base_sha") or old_hyp.head_sha or old_hyp.base_sha),
                    status="PENDING",
                    fingerprint=new_fp,
                )
                stored_new, _ = self.db.register_hypothesis(
                    new_node, allow_graveyard=bool(body.get("allow_graveyard", False))
                )
                _, before_epoch, after_epoch = self._bump_goal_epoch(goal)
                result = {
                    "goal_id": goal.goal_id,
                    "replaced_hypothesis_id": old_hyp.hypothesis_id,
                    "new_hypothesis_id": stored_new.hypothesis_id,
                    "cancelled_task_ids": cancelled_tasks,
                    "cancelled_hq_job_ids": cancelled_hq,
                    "cancelled_workflow_ids": cancelled_wfs,
                }

            elif act == "cancel_workflow":
                wf_id = str(target_id or body.get("workflow_id") or "").strip()
                if not wf_id:
                    raise ValueError("cancel_workflow requires target_id (workflow_id)")
                wf = self.db.get_workflow(wf_id)
                if wf is None:
                    raise KeyError(f"Unknown workflow_id: {wf_id}")
                goal = self._require_goal(goal_id or wf.goal_id)

                all_wfs = [wf] + self._collect_wf_descendants(goal.goal_id, wf_id)
                wf_ids = {w.workflow_id for w in all_wfs}
                cancelled_wfs: list[str] = []
                for w in all_wfs:
                    if w.status in ACTIVE_WORKFLOW_STATUSES:
                        self.db.update_workflow_state(w.workflow_id, status="CANCELLED")
                        cancelled_wfs.append(w.workflow_id)
                wf_tasks = [
                    t
                    for t in self.db.list_tasks(goal_id=goal.goal_id)
                    if t.workflow_id in wf_ids
                ]
                cancelled_tasks, cancelled_hq = self._cancel_tasks_and_hq(
                    wf_tasks, reason=reason or f"workflow {wf_id} cancelled by {actor}"
                )
                _, before_epoch, after_epoch = self._bump_goal_epoch(goal)
                result = {
                    "goal_id": goal.goal_id,
                    "cancelled_workflow_ids": cancelled_wfs,
                    "cancelled_task_ids": cancelled_tasks,
                    "cancelled_hq_job_ids": cancelled_hq,
                }

            elif act == "cancel_task":
                tid = str(target_id or body.get("task_id") or "").strip()
                if not tid:
                    raise ValueError("cancel_task requires target_id (task_id)")
                task = self.db.get_task(tid)
                if task is None:
                    raise KeyError(f"Unknown task_id: {tid}")
                goal = self._require_goal(goal_id or task.goal_id)
                cancelled_tasks, cancelled_hq = self._cancel_tasks_and_hq(
                    [task], reason=reason or f"task {tid} cancelled by {actor}"
                )
                unaffected_ids = {
                    t.task_id
                    for t in self.db.list_tasks(goal_id=goal.goal_id)
                    if t.status in ACTIVE_TASK_STATUSES and t.task_id != tid
                }
                _, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=unaffected_ids
                )
                result = {
                    "goal_id": goal.goal_id,
                    "cancelled_task_ids": cancelled_tasks,
                    "cancelled_hq_job_ids": cancelled_hq,
                }

            elif act in ("set_routing_policy", "set_model_policy"):
                raw_pol = (
                    body.get("policy")
                    or body.get("routing_policy")
                    or body.get("policy_nl")
                    or body.get("routing_policy_nl")
                    or body
                )
                if scp == "branch":
                    hid = str(target_id or body.get("hypothesis_id") or "").strip()
                    if not hid:
                        raise ValueError("branch set_routing_policy requires target_id (hypothesis_id)")
                    hyp = self.db.get_hypothesis(hid)
                    if hyp is None:
                        raise KeyError(f"Unknown hypothesis_id: {hid}")
                    goal = self._require_goal(goal_id or hyp.goal_id)
                    compiled = ModelRouter.compile_routing_policy(raw_pol, project=goal.project)
                    self.db.update_hypothesis_params(hid, {"routing_policy": compiled})
                    all_active = {
                        t.task_id
                        for t in self.db.list_tasks(goal_id=goal.goal_id)
                        if t.status in ACTIVE_TASK_STATUSES
                    }
                    updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                        goal, sync_unaffected_task_ids=all_active
                    )
                    router = ModelRouter(db=self.db)
                    branch_tasks = self.db.list_tasks(goal_id=goal.goal_id, hypothesis_id=hid)
                    c_ids, r_ids, b_ids = self._surgically_rebind_ready_tasks_for_policy(
                        updated_goal, branch_tasks, router=router, actor=actor, reason=reason
                    )
                    result = {
                        "goal_id": goal.goal_id,
                        "hypothesis_id": hid,
                        "routing_policy": compiled,
                        "cancelled_task_ids": c_ids,
                        "spawned_replacement_task_ids": r_ids,
                        "blocked_task_ids": b_ids,
                    }
                elif scp == "workflow":
                    wf_id = str(target_id or body.get("workflow_id") or "").strip()
                    wf = self.db.get_workflow(wf_id) if wf_id else None
                    if wf is None:
                        raise KeyError(f"Unknown workflow_id: {wf_id}")
                    goal = self._require_goal(goal_id or wf.goal_id)
                    compiled = ModelRouter.compile_routing_policy(raw_pol, project=goal.project)
                    self.db.update_workflow_state(
                        wf.workflow_id, config_patch={"routing_policy": compiled}
                    )
                    cur_wf_hyp = self.db.get_hypothesis(wf.hypothesis_id)
                    wf_pol_map = (
                        dict((cur_wf_hyp.params or {}).get("workflow_routing_policies") or {})
                        if cur_wf_hyp is not None
                        and isinstance((cur_wf_hyp.params or {}).get("workflow_routing_policies"), Mapping)
                        else {}
                    )
                    wf_pol_map[wf.workflow_id] = compiled
                    self.db.update_hypothesis_params(
                        wf.hypothesis_id, {"workflow_routing_policies": wf_pol_map}
                    )
                    all_active = {
                        t.task_id
                        for t in self.db.list_tasks(goal_id=goal.goal_id)
                        if t.status in ACTIVE_TASK_STATUSES
                    }
                    updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                        goal, sync_unaffected_task_ids=all_active
                    )
                    router = ModelRouter(db=self.db)
                    wf_tasks = [
                        t
                        for t in self.db.list_tasks(goal_id=goal.goal_id, hypothesis_id=wf.hypothesis_id)
                        if t.workflow_id == wf.workflow_id
                    ]
                    c_ids, r_ids, b_ids = self._surgically_rebind_ready_tasks_for_policy(
                        updated_goal, wf_tasks, router=router, actor=actor, reason=reason
                    )
                    result = {
                        "goal_id": goal.goal_id,
                        "workflow_id": wf.workflow_id,
                        "routing_policy": compiled,
                        "cancelled_task_ids": c_ids,
                        "spawned_replacement_task_ids": r_ids,
                        "blocked_task_ids": b_ids,
                    }
                else:
                    tid = str(target_id or body.get("task_id") or "").strip()
                    task = self.db.get_task(tid) if tid else None
                    if task is None:
                        raise KeyError(f"Unknown task_id: {tid}")
                    goal = self._require_goal(goal_id or task.goal_id)
                    hyp = self.db.get_hypothesis(task.hypothesis_id)
                    explicit_spec = str(body.get("explicit_spec") or (raw_pol if isinstance(raw_pol, str) else "")).strip()
                    router = ModelRouter(db=self.db)
                    role = self._stage_to_role(task.stage, hyp) or ScientificRole.ENGINEER_BUILDER
                    import inspect as _inspect_tp

                    tp_sig = _inspect_tp.signature(router.resolve_route).parameters
                    tp_extra: dict[str, Any] = {}
                    if "db" in tp_sig:
                        tp_extra["db"] = self.db
                    if "workflow_id" in tp_sig:
                        tp_extra["workflow_id"] = task.workflow_id
                    if "task" in tp_sig:
                        tp_extra["task"] = task
                    new_route = router.resolve_route(
                        role, goal=goal, hyp=hyp, explicit_spec=explicit_spec or None, **tp_extra
                    )
                    unaffected = {
                        t.task_id
                        for t in self.db.list_tasks(goal_id=goal.goal_id)
                        if t.status in ACTIVE_TASK_STATUSES and t.task_id != tid
                    }
                    updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                        goal, sync_unaffected_task_ids=unaffected
                    )
                    c_ids, hq_ids = self._cancel_tasks_and_hq([task], reason=reason or "task_rebound")
                    repl = self._spawn_replacement_task_from_snapshot(
                        updated_goal, task.to_dict(), pinned_route=new_route.to_dict()
                    )
                    result = {
                        "goal_id": goal.goal_id,
                        "cancelled_task_ids": c_ids,
                        "cancelled_hq_job_ids": hq_ids,
                        "spawned_replacement_task_ids": [repl.task_id],
                        "pinned_route": new_route.to_dict(),
                    }

            elif act == "update_plan":
                with self.db._tx():
                    hid = str(target_id or body.get("hypothesis_id") or "").strip()
                    hyp = self.db.get_hypothesis(hid) if hid else None
                    if hyp is None:
                        raise KeyError(f"Unknown hypothesis_id: {hid}")
                    goal = self._require_goal(goal_id or hyp.goal_id)
                    if hyp.goal_id != goal.goal_id:
                        raise ValueError(f"Hypothesis {hid} does not belong to goal {goal.goal_id}")
                    affected_nodes = [hyp] + self._collect_hyp_descendants(goal.goal_id, hid)
                    affected_hids = {n.hypothesis_id for n in affected_nodes}
                    affected_syn_wfs = self._collect_affected_synthesis_workflows(goal.goal_id, affected_hids)
                    affected_syn_wf_ids = {w.workflow_id for w in affected_syn_wfs}
                    all_goal_tasks = self.db.list_tasks(goal_id=goal.goal_id)
                    superseded_tasks = [
                        t
                        for t in all_goal_tasks
                        if t.status in ACTIVE_TASK_STATUSES
                        and (
                            t.hypothesis_id in affected_hids
                            or (t.workflow_id is not None and t.workflow_id in affected_syn_wf_ids)
                        )
                    ]
                    unaffected_sibling_ids = {
                        t.task_id
                        for t in all_goal_tasks
                        if t.status in ACTIVE_TASK_STATUSES
                        and t.hypothesis_id not in affected_hids
                        and (t.workflow_id is None or t.workflow_id not in affected_syn_wf_ids)
                    }
                    updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                        goal,
                        bump_plan_epoch=True,
                        sync_unaffected_task_ids=unaffected_sibling_ids,
                    )
                    next_plan_epoch = int(updated_goal.plan_epoch)
                    c_tasks, c_hq = self._cancel_tasks_and_hq(
                        superseded_tasks,
                        reason=reason or f"branch_update_plan:{hid}:pe{next_plan_epoch}",
                        epoch=after_epoch,
                        plan_epoch=next_plan_epoch,
                    )
                    c_decs, dec_hq = self._cancel_decisions_and_hq(
                        goal.goal_id,
                        reason=reason or f"branch_update_plan:{hid}:pe{next_plan_epoch}",
                        hypothesis_ids=affected_hids,
                        epoch=after_epoch,
                        plan_epoch=next_plan_epoch,
                    )
                    c_hq.extend(dec_hq)
                    cancelled_wfs = self._mark_synthesis_workflows_superseded(
                        updated_goal,
                        affected_syn_wfs,
                        reason=reason or f"branch_update_plan:{hid}:pe{next_plan_epoch}",
                        plan_epoch=next_plan_epoch,
                    )
                    for wf_obj in self.db.list_workflows(goal_id=goal.goal_id):
                        if (
                            wf_obj.status in ACTIVE_WORKFLOW_STATUSES
                            and wf_obj.hypothesis_id in affected_hids
                            and wf_obj.workflow_id not in affected_syn_wf_ids
                        ):
                            self.db.update_workflow_state(
                                wf_obj.workflow_id,
                                status="CANCELLED",
                                state_patch={"superseded_by_plan_epoch": next_plan_epoch},
                            )
                            cancelled_wfs.append(wf_obj.workflow_id)
                    for desc in affected_nodes[1:]:
                        if desc.status not in TERMINAL_HYP_STATUSES:
                            self.db.transition_hypothesis_status(
                                desc.hypothesis_id,
                                desc.status,
                                "PRUNED",
                                scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
                            )

                    preserved_rcpts: list[str] = []
                    for aff_hid in sorted(affected_hids):
                        node_preserved = [
                            r.receipt_id
                            for r in self.db.list_receipts(goal_id=goal.goal_id, hypothesis_id=aff_hid)
                            if r.process_status == ProcessStatus.EXIT_OK
                            and r.engineering_verdict == EngineeringVerdict.PASS
                            and r.audit_verdict != AuditVerdict.FAIL
                            and r.evidence_validity
                            not in (
                                EvidenceValidity.SUPERSEDED_EPOCH,
                                EvidenceValidity.HUMAN_INVALIDATED,
                                EvidenceValidity.STALE_GOAL_VERSION,
                            )
                            and r.stage in (StageKind.CODE, StageKind.CPU_GATE)
                        ]
                        if aff_hid == hid:
                            preserved_rcpts = node_preserved
                        aff_node_b = self.db.get_hypothesis(aff_hid)
                        cur_sci_rev_b = int(((aff_node_b.params if aff_node_b is not None else {}) or {}).get("lineage_sci_revision", 0) or 0)
                        self.db.update_hypothesis_params(
                            aff_hid,
                            {
                                "lineage_plan_epoch": next_plan_epoch,
                                "lineage_human_epoch": after_epoch,
                                "lineage_sci_revision": cur_sci_rev_b + 1,
                                "preserved_receipt_ids": node_preserved,
                            },
                        )
                    self.db.upsert_knowledge_item(
                        goal.goal_id,
                        "plan_lineage",
                        f"pe:{next_plan_epoch}:{hid}",
                        {
                            "hypothesis_id": hid,
                            "plan_epoch": next_plan_epoch,
                            "human_epoch": after_epoch,
                            "preserved_upstream_receipt_ids": preserved_rcpts,
                            "superseded_task_ids": c_tasks,
                            "superseded_workflow_ids": cancelled_wfs,
                        },
                        hypothesis_id=hid,
                        goal_version=int(updated_goal.version),
                    )
                    exp_plan = ExperimentPlan(
                        exp_plan_id=str(body.get("exp_plan_id") or f"eplan_{uuid.uuid4().hex[:10]}"),
                        goal_id=goal.goal_id,
                        hypothesis_id=hid,
                        goal_version=int(updated_goal.version),
                        human_epoch=after_epoch,
                        plan_epoch=next_plan_epoch,
                        falsifiable_claim=str(body.get("falsifiable_claim") or hyp.statement),
                        control_baseline=dict(body.get("control_baseline") or {}),
                        venv_python=str(body.get("venv_python") or ""),
                        env_vars={str(k): str(v) for k, v in dict(body.get("env_vars") or {}).items()},
                        required_hardware=str(body.get("required_hardware") or "cpu_only"),
                        min_vram_mb=int(body.get("min_vram_mb", 0)),
                        gate_timeout_s=int(body.get("gate_timeout_s", 90)),
                        screen_timeout_s=int(body.get("screen_timeout_s", 300)),
                        confirm_timeout_s=int(body.get("confirm_timeout_s", 1200)),
                        confound_checks=[str(x) for x in (body.get("confound_checks") or [])],
                        review_status=str(body.get("review_status") or "APPROVED"),
                    )
                    saved_exp = self.db.save_experiment_plan(exp_plan, link_hypothesis=True)
                    if hyp.status in ("BLOCKED_ENV", "BLOCKED_INFRA"):
                        self.db.transition_hypothesis_status(
                            hid,
                            hyp.status,
                            "PENDING",
                            failure_domain=FailureDomain.NONE,
                            budget_state_patch={"env_rechecks_used": 0},
                            exp_plan_id=saved_exp.exp_plan_id,
                        )
                    result = {
                        "goal_id": goal.goal_id,
                        "hypothesis_id": hid,
                        "exp_plan_id": saved_exp.exp_plan_id,
                        "plan_version": saved_exp.plan_version,
                        "plan_epoch": next_plan_epoch,
                        "cancelled_task_ids": c_tasks,
                        "cancelled_hq_job_ids": c_hq,
                        "cancelled_workflow_ids": cancelled_wfs,
                        "cancelled_decision_ids": c_decs,
                        "preserved_upstream_receipt_ids": preserved_rcpts,
                    }

            elif act == "replan_from_node":
                raw_hid = str(target_id or body.get("hypothesis_id") or body.get("node_id") or "").strip()
                if not raw_hid:
                    raise ValueError("replan_from_node requires target_id or hypothesis_id")
                h_lookup = self.db.get_hypothesis(raw_hid)
                if h_lookup is None:
                    w_lookup = self.db.get_workflow(raw_hid)
                    if w_lookup is not None:
                        raw_hid = w_lookup.hypothesis_id
                result, before_epoch, after_epoch = self._replan_from_node_impl(
                    goal_id, raw_hid, body, actor=actor, reason=reason
                )

            elif act == "override_verdict":
                raw_hid = str(target_id or body.get("hypothesis_id") or "").strip()
                if not raw_hid:
                    raise ValueError("override_verdict requires target_id or hypothesis_id")
                h_lookup = self.db.get_hypothesis(raw_hid)
                if h_lookup is None:
                    t_lookup = self.db.get_task(raw_hid)
                    if t_lookup is not None:
                        raw_hid = t_lookup.hypothesis_id
                result, before_epoch, after_epoch = self._override_verdict_impl(
                    goal_id, raw_hid, body, actor=actor, reason=reason
                )

            elif act == "switch_workflow":
                from conjectra.workflow_runtime import WorkflowRuntime

                raw_target = str(target_id or body.get("hypothesis_id") or body.get("workflow_id") or "").strip()
                hyp = self.db.get_hypothesis(raw_target)
                if hyp is None and raw_target:
                    wf_obj = self.db.get_workflow(raw_target)
                    if wf_obj is not None:
                        hyp = self.db.get_hypothesis(wf_obj.hypothesis_id)
                if hyp is None:
                    raise KeyError(f"Unknown hypothesis/workflow target: {raw_target}")
                goal = self._require_goal(goal_id or hyp.goal_id)
                new_prim = str(body.get("workflow_primitive") or body.get("primitive") or "").strip()
                if not new_prim:
                    raise ValueError("switch_workflow requires workflow_primitive")
                branch_running = [
                    t
                    for t in self.db.list_tasks(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id)
                    if t.status in ("DISPATCHED", "RUNNING")
                ]
                c_running, c_hq = self._cancel_tasks_and_hq(
                    branch_running, reason=reason or f"switch_workflow:{new_prim}"
                )
                unaffected = {
                    t.task_id
                    for t in self.db.list_tasks(goal_id=goal.goal_id)
                    if t.status in ACTIVE_TASK_STATUSES and t.hypothesis_id != hyp.hypothesis_id
                }
                updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=unaffected
                )
                wfr = WorkflowRuntime(self.db, router=ModelRouter(db=self.db))
                new_wf, new_tasks = wfr.switch_hypothesis_workflow(
                    updated_goal,
                    hyp,
                    new_primitive=new_prim,
                    reason=reason or f"human_switch:{actor}",
                    config_override=body.get("config"),
                )
                result = {
                    "goal_id": goal.goal_id,
                    "hypothesis_id": hyp.hypothesis_id,
                    "workflow_id": new_wf.workflow_id,
                    "primitive": new_wf.primitive,
                    "cancelled_task_ids": c_running,
                    "cancelled_hq_job_ids": c_hq,
                    "enqueued_task_ids": [t.task_id for t in new_tasks],
                }

            elif act in ("inject_guidance", "inject"):
                guidance_text = str(
                    body.get("guidance") or body.get("note") or body.get("message") or reason
                ).strip()
                if not guidance_text:
                    raise ValueError("inject_guidance requires non-empty guidance text")
                hid = str(target_id or body.get("hypothesis_id") or "").strip()
                hyp = self.db.get_hypothesis(hid) if hid else None
                goal = self._require_goal(goal_id or (hyp.goal_id if hyp else None))
                all_active = {
                    t.task_id
                    for t in self.db.list_tasks(goal_id=goal.goal_id)
                    if t.status in ACTIVE_TASK_STATUSES
                }
                updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=all_active
                )
                if hyp is not None:
                    self.db.update_hypothesis_params(
                        hyp.hypothesis_id, {"human_guidance": guidance_text}
                    )
                k_item = self.db.upsert_knowledge_item(
                    goal.goal_id,
                    "human_guidance",
                    f"guidance:{hid or target_id or 'branch'}:e{after_epoch}",
                    {"guidance": guidance_text, "actor": actor, "epoch": after_epoch},
                    hypothesis_id=hyp.hypothesis_id if hyp else None,
                    goal_version=int(updated_goal.version),
                )
                result = {
                    "goal_id": goal.goal_id,
                    "hypothesis_id": hyp.hypothesis_id if hyp else None,
                    "knowledge_item_id": k_item["item_id"],
                }

            else:
                raise ValueError(f"Unsupported {scp} intervention action: {action!r}")

        elif scp == "subgoal":
            result, before_epoch, after_epoch = self._apply_subgoal_intervention(
                act,
                goal_id=goal_id,
                target_id=target_id,
                body=body,
                actor=actor,
                reason=reason,
            )

        else:
            raise ValueError(
                f"Invalid scope {scope!r}; expected one of 'global', 'goal', 'subgoal', 'branch', 'workflow', 'task'"
            )

        intervention_rec = self.db.record_intervention(
            scp,
            act,
            goal_id=goal_id or result.get("goal_id"),
            target_id=target_id,
            actor=actor,
            reason=reason,
            before_epoch=before_epoch,
            after_epoch=after_epoch,
            payload=body,
            result=result,
        )
        self._wake_reactor(
            "HUMAN_INTERVENTION",
            scope=scp,
            action=act,
            goal_id=goal_id or result.get("goal_id"),
            target_id=target_id,
            intervention_id=intervention_rec["intervention_id"],
        )
        return {
            **result,
            "intervention_id": intervention_rec["intervention_id"],
            "scope": scp,
            "action": act,
            "before_epoch": before_epoch,
            "after_epoch": after_epoch,
        }

    def _replan_from_node_impl(
        self,
        goal_id: str | None,
        hid: str,
        body: Mapping[str, Any],
        *,
        actor: str = "user",
        reason: str = "",
    ) -> tuple[dict[str, Any], int, int]:
        """Surgically replan from `hid` at `from_stage`, invalidating downstream descendants while preserving verified upstream receipts."""
        from conjectra.workflow_runtime import WorkflowRuntime

        stage_order = [
            StageKind.PROPOSE,
            StageKind.DESIGN_REVIEW,
            StageKind.CODE,
            StageKind.CPU_GATE,
            StageKind.IMPL_REVIEW,
            StageKind.EXP_SCREEN,
            StageKind.EXP_CONFIRM,
            StageKind.RESULT_REVIEW,
            StageKind.BLIND_AUDIT,
        ]
        with self.db._tx():
            hyp = self.db.get_hypothesis(hid)
            if hyp is None:
                raise KeyError(f"Unknown hypothesis_id: {hid}")
            goal = self._require_goal(goal_id or hyp.goal_id)
            if hyp.goal_id != goal.goal_id:
                raise ValueError(f"Hypothesis {hid} does not belong to goal {goal.goal_id}")

            raw_from_stage = str(body.get("from_stage") or "CODE").strip().upper()
            from_stage = StageKind(raw_from_stage) if raw_from_stage in {s.value for s in StageKind} else StageKind.CODE
            cutoff_idx = stage_order.index(from_stage) if from_stage in stage_order else 2
            upstream_stages = set(stage_order[:cutoff_idx])

            descendants = self._collect_hyp_descendants(goal.goal_id, hid)
            affected_hids = {hid} | {d.hypothesis_id for d in descendants}
            affected_syn_wfs = self._collect_affected_synthesis_workflows(goal.goal_id, affected_hids)
            affected_syn_wf_ids = {w.workflow_id for w in affected_syn_wfs}

            all_tasks = self.db.list_tasks(goal_id=goal.goal_id)
            superseded_tasks = [
                t
                for t in all_tasks
                if t.status in ACTIVE_TASK_STATUSES
                and (
                    t.hypothesis_id in affected_hids
                    or (t.workflow_id is not None and t.workflow_id in affected_syn_wf_ids)
                )
            ]
            unaffected_sibling_ids = {
                t.task_id
                for t in all_tasks
                if t.status in ACTIVE_TASK_STATUSES
                and t.hypothesis_id not in affected_hids
                and (t.workflow_id is None or t.workflow_id not in affected_syn_wf_ids)
            }

            updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                goal,
                bump_plan_epoch=True,
                sync_unaffected_task_ids=unaffected_sibling_ids,
            )
            next_plan_epoch = int(updated_goal.plan_epoch)

            c_tasks, c_hq = self._cancel_tasks_and_hq(
                superseded_tasks,
                reason=reason or f"replan_from_node:{hid}:{from_stage.value}:pe{next_plan_epoch}",
                epoch=after_epoch,
                plan_epoch=next_plan_epoch,
            )
            c_decs, dec_hq = self._cancel_decisions_and_hq(
                goal.goal_id,
                reason=reason or f"replan_from_node:{hid}:pe{next_plan_epoch}",
                hypothesis_ids=affected_hids,
                epoch=after_epoch,
                plan_epoch=next_plan_epoch,
            )
            c_hq.extend(dec_hq)

            cancelled_wfs = self._mark_synthesis_workflows_superseded(
                updated_goal,
                affected_syn_wfs,
                reason=reason or f"replan_from_node:{hid}:pe{next_plan_epoch}",
                plan_epoch=next_plan_epoch,
            )
            for wf_obj in self.db.list_workflows(goal_id=goal.goal_id):
                if (
                    wf_obj.status in ACTIVE_WORKFLOW_STATUSES
                    and wf_obj.hypothesis_id in affected_hids
                    and wf_obj.workflow_id not in affected_syn_wf_ids
                ):
                    self.db.update_workflow_state(
                        wf_obj.workflow_id,
                        status="CANCELLED",
                        state_patch={"superseded_by_plan_epoch": next_plan_epoch},
                    )
                    cancelled_wfs.append(wf_obj.workflow_id)

            pruned_descendants: list[str] = []
            for desc in descendants:
                if desc.status not in TERMINAL_HYP_STATUSES:
                    self.db.transition_hypothesis_status(
                        desc.hypothesis_id,
                        desc.status,
                        "PRUNED",
                        scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
                    )
                    pruned_descendants.append(desc.hypothesis_id)

            preserved_rcpts: list[str] = []
            for aff_hid in sorted(affected_hids):
                node_preserved = [
                    r.receipt_id
                    for r in self.db.list_receipts(goal_id=goal.goal_id, hypothesis_id=aff_hid)
                    if r.stage in upstream_stages
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
                aff_node_r = self.db.get_hypothesis(aff_hid)
                cur_sci_rev_r = int(((aff_node_r.params if aff_node_r is not None else {}) or {}).get("lineage_sci_revision", 0) or 0)
                updated_node = self.db.update_hypothesis_params(
                    aff_hid,
                    {
                        "lineage_plan_epoch": next_plan_epoch,
                        "lineage_human_epoch": after_epoch,
                        "lineage_sci_revision": cur_sci_rev_r + 1,
                        "preserved_receipt_ids": node_preserved,
                    },
                )
                if aff_hid == hid:
                    preserved_rcpts = node_preserved
                    hyp = updated_node
            self.db.upsert_knowledge_item(
                goal.goal_id,
                "plan_lineage",
                f"replan:{hid}:pe{next_plan_epoch}",
                {
                    "hypothesis_id": hid,
                    "from_stage": from_stage.value,
                    "plan_epoch": next_plan_epoch,
                    "human_epoch": after_epoch,
                    "preserved_upstream_receipt_ids": preserved_rcpts,
                    "pruned_descendant_hypothesis_ids": pruned_descendants,
                    "superseded_task_ids": c_tasks,
                    "superseded_workflow_ids": cancelled_wfs,
                },
                hypothesis_id=hid,
                goal_version=int(updated_goal.version),
            )

            if isinstance(body.get("params_patch"), Mapping):
                hyp = self.db.update_hypothesis_params(hid, dict(body["params_patch"]))

            if isinstance(body.get("experiment_plan"), Mapping):
                ep_dict = {
                    "exp_plan_id": f"eplan_{uuid.uuid4().hex[:10]}",
                    "goal_id": goal.goal_id,
                    "hypothesis_id": hid,
                    "goal_version": int(updated_goal.version),
                    "human_epoch": after_epoch,
                    "plan_epoch": next_plan_epoch,
                    "falsifiable_claim": hyp.statement,
                    **dict(body["experiment_plan"]),
                }
                self.db.save_experiment_plan(ExperimentPlan.from_dict(ep_dict), link_hypothesis=True)

            if hyp.status != "PENDING":
                hyp = self.db.transition_hypothesis_status(
                    hid,
                    hyp.status,
                    "PENDING",
                    scientific_verdict=ScientificVerdict.UNKNOWN,
                    failure_domain=FailureDomain.NONE,
                )

            enqueued_ids: list[str] = []
            new_wf_id: str | None = None
            has_unconfirmed = hasattr(self.db, "has_unconfirmed_hq_cancellations") and self.db.has_unconfirmed_hq_cancellations(
                goal.goal_id, hypothesis_id=hid
            )
            if bool(body.get("spawn_workflow", True)) and not has_unconfirmed:
                wfr = WorkflowRuntime(self.db, router=ModelRouter(db=self.db))
                if from_stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
                    wf_re = WorkflowNode(
                        workflow_id=f"wf_{uuid.uuid4().hex[:12]}",
                        goal_id=goal.goal_id,
                        hypothesis_id=hid,
                        primitive="direct_hq",
                        status="RUNNING",
                        goal_version=int(updated_goal.version),
                        human_epoch=after_epoch,
                        plan_epoch=next_plan_epoch,
                        state={"step": from_stage.value.lower(), "completed_steps": [s.value.lower() for s in upstream_stages]},
                    )
                    wf_re = self.db.save_workflow(wf_re)
                    new_wf_id = wf_re.workflow_id
                    t_emp = wfr._make_task(
                        updated_goal,
                        hyp,
                        wf_re,
                        from_stage,
                        step_key=from_stage.value.lower(),
                    )
                    enqueued_ids.append(t_emp.task_id)
                else:
                    wf_re, wf_tasks = wfr.create_workflow_for_hypothesis(
                        updated_goal,
                        hyp,
                        primitive=str(body.get("workflow_primitive") or hyp.params.get("workflow_primitive") or "direct_hq"),
                        config_override={"require_design_review": from_stage == StageKind.DESIGN_REVIEW},
                    )
                    new_wf_id = wf_re.workflow_id
                    enqueued_ids.extend(t.task_id for t in wf_tasks)

            return (
                {
                    "goal_id": goal.goal_id,
                    "hypothesis_id": hid,
                    "from_stage": from_stage.value,
                    "plan_epoch": next_plan_epoch,
                    "workflow_id": new_wf_id,
                    "enqueued_task_ids": enqueued_ids,
                    "waiting_cancel_confirm": bool(has_unconfirmed),
                    "pruned_descendant_hypothesis_ids": pruned_descendants,
                    "cancelled_task_ids": c_tasks,
                    "cancelled_hq_job_ids": c_hq,
                    "cancelled_workflow_ids": cancelled_wfs,
                    "cancelled_decision_ids": c_decs,
                    "preserved_upstream_receipt_ids": preserved_rcpts,
                },
                before_epoch,
                after_epoch,
            )

    def _override_verdict_impl(
        self,
        goal_id: str | None,
        hid: str,
        body: Mapping[str, Any],
        *,
        actor: str = "user",
        reason: str = "",
    ) -> tuple[dict[str, Any], int, int]:
        """Atomically record an audited human verdict override on a hypothesis/branch."""
        with self.db._tx():
            hyp = self.db.get_hypothesis(hid)
            if hyp is None:
                raise KeyError(f"Unknown hypothesis_id: {hid}")
            goal = self._require_goal(goal_id or hyp.goal_id)
            if hyp.goal_id != goal.goal_id:
                raise ValueError(f"Hypothesis {hid} does not belong to goal {goal.goal_id}")

            raw_sv = str(body.get("scientific_verdict") or body.get("verdict") or "").strip().upper()
            if not raw_sv:
                raise ValueError("override_verdict requires scientific_verdict (e.g. CONFIRMED, REFUTED, VALID_NEGATIVE, UNKNOWN)")
            new_sv = ScientificVerdict(raw_sv)
            new_status = str(
                body.get("status")
                or (
                    "COMPLETED"
                    if new_sv == ScientificVerdict.CONFIRMED
                    else (new_sv.value if new_sv in (ScientificVerdict.REFUTED, ScientificVerdict.VALID_NEGATIVE) else "PENDING")
                )
            )

            cancel_branch = new_sv in (ScientificVerdict.REFUTED, ScientificVerdict.VALID_NEGATIVE) or bool(
                body.get("cancel_inflight", new_status in ("PRUNED", "REFUTED", "VALID_NEGATIVE"))
            )
            all_tasks = self.db.list_tasks(goal_id=goal.goal_id)
            unaffected_ids = {
                t.task_id
                for t in all_tasks
                if t.status in ACTIVE_TASK_STATUSES and (not cancel_branch or t.hypothesis_id != hid)
            }
            updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                goal, sync_unaffected_task_ids=unaffected_ids
            )

            c_tasks: list[str] = []
            c_hq: list[int] = []
            if cancel_branch:
                branch_tasks = [t for t in all_tasks if t.status in ACTIVE_TASK_STATUSES and t.hypothesis_id == hid]
                c_tasks, c_hq = self._cancel_tasks_and_hq(
                    branch_tasks,
                    reason=reason or f"override_verdict:{new_sv.value}:{actor}",
                    epoch=after_epoch,
                    plan_epoch=int(updated_goal.plan_epoch),
                )
                for wf_obj in self.db.list_workflows(goal_id=goal.goal_id, hypothesis_id=hid):
                    if wf_obj.status in ACTIVE_WORKFLOW_STATUSES:
                        self.db.update_workflow_state(wf_obj.workflow_id, status="CANCELLED")

            prev_sv = hyp.scientific_verdict.value
            prev_status = hyp.status
            self.db.transition_hypothesis_status(
                hid,
                hyp.status,
                new_status,
                scientific_verdict=new_sv,
            )
            self.db.upsert_knowledge_item(
                goal.goal_id,
                "human_verdict_override",
                f"override:{hid}:e{after_epoch}",
                {
                    "hypothesis_id": hid,
                    "previous_verdict": prev_sv,
                    "new_verdict": new_sv.value,
                    "previous_status": prev_status,
                    "new_status": new_status,
                    "actor": actor,
                    "reason": reason or str(body.get("rationale") or "human_override"),
                    "epoch": after_epoch,
                },
                hypothesis_id=hid,
                goal_version=int(updated_goal.version),
            )
            return (
                {
                    "goal_id": goal.goal_id,
                    "hypothesis_id": hid,
                    "previous_verdict": prev_sv,
                    "scientific_verdict": new_sv.value,
                    "status": new_status,
                    "cancelled_task_ids": c_tasks,
                    "cancelled_hq_job_ids": c_hq,
                },
                before_epoch,
                after_epoch,
            )

    def _require_subgoal(
        self,
        goal_id: str | None,
        target_id: str | None,
        body: Mapping[str, Any],
    ) -> tuple[ResearchGoal, SubGoalNode]:
        sg_id = str(target_id or body.get("subgoal_id") or "").strip()
        if not sg_id:
            raise ValueError("subgoal intervention requires target_id (subgoal_id)")
        sg = (
            self.db.get_goal_subgoal(goal_id, sg_id)
            if goal_id
            else self.db.get_subgoal(sg_id)
        )
        if sg is None:
            raise KeyError(f"Unknown subgoal_id: {sg_id}")
        goal = self._require_goal(goal_id or sg.goal_id)
        return goal, sg

    def _subgoal_hypotheses(self, goal_id: str, subgoal_id: str) -> list[HypothesisNode]:
        hyps = self.db.list_hypotheses(goal_id=goal_id)
        out: list[HypothesisNode] = []
        for h in hyps:
            h_sg = getattr(h, "subgoal_id", None) or (
                (h.params or {}).get("subgoal_id") if isinstance(h.params, Mapping) else None
            )
            if h_sg == subgoal_id:
                out.append(h)
        return out

    def _is_hq_job_terminated(
        self,
        goal_id: str,
        jid: int | None,
        ent_id: str = "",
        *,
        entity_kind: str | None = None,
        attempt: int | None = None,
        now: float | None = None,
    ) -> bool:
        if jid is None:
            return True
        ts = time.time() if now is None else float(now)
        if hasattr(self.db, "list_pending_hq_cancellations"):
            confirmed_rows = self.db.list_pending_hq_cancellations(
                goal_id=goal_id, state="CANCEL_CONFIRMED"
            )
            if any(
                int(r.get("hq_job_id", -1)) == int(jid)
                and (not entity_kind or not r.get("entity_kind") or str(r.get("entity_kind")) == entity_kind)
                and (not ent_id or not r.get("entity_id") or str(r.get("entity_id")) == ent_id)
                and (attempt is None or not r.get("attempt") or int(r.get("attempt") or 0) == int(attempt))
                for r in confirmed_rows
            ):
                return True
        q_state = self._query_hq_job_state(self.hq, int(jid))
        lookup_status = str(q_state.get("lookup_status") or "QUERY_ERROR").upper()
        if lookup_status == "FOUND":
            if self._hq_positively_terminated(self.hq, int(jid), prefetched_info=q_state.get("job")):
                if hasattr(self.db, "confirm_hq_cancellation"):
                    self.db.confirm_hq_cancellation(
                        int(jid), entity_kind=entity_kind, entity_id=ent_id or None, attempt=attempt, now=ts
                    )
                return True
            return False
        if lookup_status == "NOT_FOUND":
            has_live_lease = bool(
                ent_id
                and any(
                    str(ls.get("task_id") or "") in {ent_id, f"dec-{ent_id}"}
                    for ls in self.db.list_active_leases(now=ts)
                )
            )
            if not has_live_lease:
                if hasattr(self.db, "confirm_hq_cancellation"):
                    self.db.confirm_hq_cancellation(
                        int(jid), entity_kind=entity_kind, entity_id=ent_id or None, attempt=attempt, now=ts
                    )
                return True
        return False

    def _apply_subgoal_intervention(
        self,
        act: str,
        *,
        goal_id: str | None,
        target_id: str | None,
        body: Mapping[str, Any],
        actor: str,
        reason: str,
    ) -> tuple[dict[str, Any], int | None, int | None]:
        """Atomically execute SubGoal interventions (`pause`, `resume`, `kill`, `retune`, `merge`, `spawn`)."""
        now = time.time()
        with self.db._tx():
            if act == "pause":
                goal, sg = self._require_subgoal(goal_id, target_id, body)
                if str(sg.status).upper() in (SubGoalStatus.KILLED.value, SubGoalStatus.MERGED.value):
                    raise StateTransitionError(
                        f"Cannot pause subgoal {sg.subgoal_id} in terminal status {sg.status!r}"
                    )
                sg_hyps = self._subgoal_hypotheses(goal.goal_id, sg.subgoal_id)
                sg_hids = {h.hypothesis_id for h in sg_hyps}
                all_tasks = self.db.list_tasks(goal_id=goal.goal_id)
                running_tasks = [
                    t
                    for t in all_tasks
                    if t.hypothesis_id in sg_hids and t.status in ("DISPATCHED", "RUNNING")
                ]
                paused_records = [t.to_dict() for t in running_tasks]
                cancelled_tasks: list[str] = []
                cancelled_hq: list[int] = []
                if bool(body.get("cancel_running", True)) and running_tasks:
                    cancelled_tasks, cancelled_hq = self._cancel_tasks_and_hq(
                        running_tasks,
                        reason=reason or f"subgoal_paused:{sg.subgoal_id}:{actor}",
                        epoch=int(goal.human_epoch) + 1,
                        plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                    )
                cancelled_decs, dec_hq = self._cancel_decisions_and_hq(
                    goal.goal_id,
                    reason=reason or f"subgoal_paused:{sg.subgoal_id}:{actor}",
                    hypothesis_ids=sg_hids,
                    subgoal_id=sg.subgoal_id,
                    epoch=int(goal.human_epoch) + 1,
                    plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                )
                cancelled_hq.extend(dec_hq)

                by_wf: dict[str, list[dict[str, Any]]] = {}
                standalone_paused: list[dict[str, Any]] = []
                for snap in paused_records:
                    wf_id = snap.get("workflow_id")
                    if wf_id:
                        by_wf.setdefault(str(wf_id), []).append(snap)
                    else:
                        standalone_paused.append(snap)

                paused_wfs: list[str] = []
                for wf in self.db.list_workflows(goal_id=goal.goal_id):
                    if wf.hypothesis_id in sg_hids and wf.status in ("READY", "RUNNING", "WAITING"):
                        patch: dict[str, Any] = {
                            "paused_prev_status": wf.status,
                            "paused_by_subgoal": sg.subgoal_id,
                        }
                        if wf.workflow_id in by_wf:
                            patch["paused_tasks"] = list(wf.state.get("paused_tasks") or []) + by_wf[wf.workflow_id]
                        self.db.update_workflow_state(
                            wf.workflow_id, status="PAUSED", state_patch=patch, now=now
                        )
                        paused_wfs.append(wf.workflow_id)

                if standalone_paused:
                    self.db.upsert_knowledge_item(
                        goal.goal_id,
                        "paused_subgoal_tasks",
                        sg.subgoal_id,
                        {"tasks": standalone_paused, "paused_ts": now},
                        goal_version=int(goal.version),
                        now=now,
                    )

                prev_st = str(sg.status)
                self.db.transition_subgoal_status(
                    sg.subgoal_id,
                    None,
                    SubGoalStatus.PAUSED.value,
                    params_patch={
                        "paused_prev_status": prev_st if prev_st != SubGoalStatus.PAUSED.value else (sg.params or {}).get("paused_prev_status", SubGoalStatus.ACTIVE.value),
                        "paused_reason": reason,
                        "paused_by": actor,
                    },
                    now=now,
                )
                unaffected_ids = {
                    t.task_id
                    for t in all_tasks
                    if t.status in ACTIVE_TASK_STATUSES and t.hypothesis_id not in sg_hids
                }
                _, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=unaffected_ids
                )
                return (
                    {
                        "goal_id": goal.goal_id,
                        "subgoal_id": sg.subgoal_id,
                        "status": SubGoalStatus.PAUSED.value,
                        "previous_status": prev_st,
                        "paused_workflow_ids": paused_wfs,
                        "cancelled_task_ids": cancelled_tasks,
                        "cancelled_hq_job_ids": cancelled_hq,
                        "cancelled_decision_ids": cancelled_decs,
                    },
                    before_epoch,
                    after_epoch,
                )

            if act == "resume":
                goal, sg = self._require_subgoal(goal_id, target_id, body)
                if str(sg.status).upper() in (SubGoalStatus.KILLED.value, SubGoalStatus.MERGED.value):
                    raise StateTransitionError(
                        f"Cannot resume subgoal {sg.subgoal_id} in terminal status {sg.status!r}"
                    )
                sg_hyps = self._subgoal_hypotheses(goal.goal_id, sg.subgoal_id)
                sg_hids = {h.hypothesis_id for h in sg_hyps}
                all_tasks = self.db.list_tasks(goal_id=goal.goal_id)
                sync_ids = {
                    t.task_id
                    for t in all_tasks
                    if t.status in ACTIVE_TASK_STATUSES
                    and (t.hypothesis_id not in sg_hids or t.status == "READY")
                }
                updated_goal, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=sync_ids
                )

                resumed_wfs: list[str] = []
                spawned_task_ids: list[str] = []
                waiting_confirm_ids: list[str] = []
                for wf in self.db.list_workflows(goal_id=goal.goal_id):
                    if wf.hypothesis_id in sg_hids and wf.status == "PAUSED":
                        prev_wf_st = str(wf.state.get("paused_prev_status") or "RUNNING")
                        paused_snaps = list(wf.state.get("paused_tasks") or [])
                        rem_snaps: list[dict[str, Any]] = []
                        for snap in paused_snaps:
                            jid = snap.get("hq_job_id")
                            tid_s = str(snap.get("task_id") or "")
                            att_s = int(snap["attempt"]) if snap.get("attempt") is not None else None
                            if not self._is_hq_job_terminated(
                                goal.goal_id,
                                int(jid) if jid is not None else None,
                                tid_s,
                                entity_kind="task",
                                attempt=att_s,
                                now=now,
                            ):
                                rem_snaps.append({**dict(snap), "wait_reason": "WAIT_CANCEL_CONFIRM"})
                                if tid_s:
                                    waiting_confirm_ids.append(tid_s)
                                continue
                            repl = self._spawn_replacement_task_from_snapshot(
                                updated_goal, snap, now=now
                            )
                            spawned_task_ids.append(repl.task_id)
                        self.db.update_workflow_state(
                            wf.workflow_id,
                            status=prev_wf_st if prev_wf_st in ("READY", "RUNNING", "WAITING") else "RUNNING",
                            state_patch={"paused_tasks": rem_snaps},
                            now=now,
                        )
                        resumed_wfs.append(wf.workflow_id)

                for item in self.db.list_knowledge_items(
                    goal_id=goal.goal_id, category="paused_subgoal_tasks", limit=20
                ):
                    if item.get("key") != sg.subgoal_id:
                        continue
                    snaps = list((item.get("content") or {}).get("tasks") or [])
                    if not snaps:
                        continue
                    rem_s: list[dict[str, Any]] = []
                    for snap in snaps:
                        jid = snap.get("hq_job_id")
                        tid_s = str(snap.get("task_id") or "")
                        att_s = int(snap["attempt"]) if snap.get("attempt") is not None else None
                        if self._is_hq_job_terminated(
                            goal.goal_id,
                            int(jid) if jid is not None else None,
                            tid_s,
                            entity_kind="task",
                            attempt=att_s,
                            now=now,
                        ):
                            repl = self._spawn_replacement_task_from_snapshot(
                                updated_goal, snap, now=now
                            )
                            spawned_task_ids.append(repl.task_id)
                        else:
                            rem_s.append({**dict(snap), "wait_reason": "WAIT_CANCEL_CONFIRM"})
                            if tid_s:
                                waiting_confirm_ids.append(tid_s)
                    self.db.upsert_knowledge_item(
                        goal.goal_id,
                        "paused_subgoal_tasks",
                        sg.subgoal_id,
                        {"tasks": rem_s, "resumed_ts": now},
                        goal_version=int(updated_goal.version),
                        now=now,
                    )

                prev_sg_st = str((sg.params or {}).get("paused_prev_status") or SubGoalStatus.ACTIVE.value)
                if prev_sg_st == SubGoalStatus.LOCKED_DEP.value:
                    deps = [self.db.get_subgoal(d) for d in (sg.depends_on_subgoals or [])]
                    all_sealed = all(
                        dn is not None and str(dn.status).upper() in (SubGoalStatus.SEALED.value, SubGoalStatus.MERGED.value)
                        for dn in deps
                    )
                    target_sg_st = SubGoalStatus.ACTIVE.value if (all_sealed or bool(body.get("force"))) else SubGoalStatus.LOCKED_DEP.value
                elif prev_sg_st in (SubGoalStatus.ACTIVE.value, SubGoalStatus.AWAITING_APEX.value):
                    target_sg_st = prev_sg_st
                else:
                    target_sg_st = SubGoalStatus.ACTIVE.value

                self.db.transition_subgoal_status(
                    sg.subgoal_id,
                    None,
                    target_sg_st,
                    params_patch={"resumed_by": actor, "resumed_ts": now},
                    now=now,
                )
                return (
                    {
                        "goal_id": goal.goal_id,
                        "subgoal_id": sg.subgoal_id,
                        "status": target_sg_st,
                        "resumed_workflow_ids": resumed_wfs,
                        "spawned_replacement_task_ids": spawned_task_ids,
                        "waiting_cancel_confirm_task_ids": waiting_confirm_ids,
                    },
                    before_epoch,
                    after_epoch,
                )

            if act == "kill":
                goal, sg = self._require_subgoal(goal_id, target_id, body)
                sg_hyps = self._subgoal_hypotheses(goal.goal_id, sg.subgoal_id)
                sg_hids = {h.hypothesis_id for h in sg_hyps}
                all_tasks = self.db.list_tasks(goal_id=goal.goal_id)
                sg_tasks = [
                    t
                    for t in all_tasks
                    if t.hypothesis_id in sg_hids and t.status in ACTIVE_TASK_STATUSES
                ]
                cancelled_tasks, cancelled_hq = self._cancel_tasks_and_hq(
                    sg_tasks,
                    reason=reason or f"subgoal_killed:{sg.subgoal_id}:{actor}",
                    epoch=int(goal.human_epoch) + 1,
                    plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                )
                cancelled_decs, dec_hq = self._cancel_decisions_and_hq(
                    goal.goal_id,
                    reason=reason or f"subgoal_killed:{sg.subgoal_id}:{actor}",
                    hypothesis_ids=sg_hids,
                    subgoal_id=sg.subgoal_id,
                    epoch=int(goal.human_epoch) + 1,
                    plan_epoch=int(getattr(goal, "plan_epoch", 0) or 0),
                )
                cancelled_hq.extend(dec_hq)
                cancelled_wfs: list[str] = []
                for wf in self.db.list_workflows(goal_id=goal.goal_id):
                    if wf.hypothesis_id in sg_hids and wf.status in ACTIVE_WORKFLOW_STATUSES:
                        self.db.update_workflow_state(
                            wf.workflow_id,
                            status="CANCELLED",
                            state_patch={"cancel_reason": reason or f"subgoal_killed:{sg.subgoal_id}"},
                            now=now,
                        )
                        cancelled_wfs.append(wf.workflow_id)
                pruned_hyps: list[str] = []
                for h in sg_hyps:
                    if h.status not in TERMINAL_HYP_STATUSES and h.status not in ("COMPLETED", "VALID_NEGATIVE"):
                        self.db.transition_hypothesis_status(
                            h.hypothesis_id,
                            h.status,
                            "PRUNED",
                            scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
                            now=now,
                        )
                        pruned_hyps.append(h.hypothesis_id)
                self.db.transition_subgoal_status(
                    sg.subgoal_id,
                    None,
                    SubGoalStatus.KILLED.value,
                    params_patch={"killed_reason": reason, "killed_by": actor, "killed_ts": now},
                    now=now,
                )
                reallocated_to: list[str] = []
                raw_realloc = body.get("reallocate_to")
                if raw_realloc or bool(body.get("reallocate")):
                    siblings = [
                        s
                        for s in self.db.list_subgoals(goal_id=goal.goal_id, status=SubGoalStatus.ACTIVE.value)
                        if s.subgoal_id != sg.subgoal_id
                    ]
                    if raw_realloc:
                        target_ids = (
                            [str(raw_realloc)]
                            if isinstance(raw_realloc, str)
                            else [str(x) for x in raw_realloc]
                        )
                        targets = [s for s in siblings if s.subgoal_id in set(target_ids)]
                    else:
                        targets = siblings
                    if targets:
                        w_share = float(sg.priority_weight or 0.0) / len(targets)
                        b_share = int(sg.max_tokens_budget or 0) // len(targets)
                        for ts_node in targets:
                            patch_r: dict[str, Any] = {
                                "priority_weight": float(ts_node.priority_weight) + w_share
                            }
                            if b_share > 0:
                                patch_r["max_tokens_budget"] = int(ts_node.max_tokens_budget or 0) + b_share
                            self.db.update_subgoal_fields(ts_node.subgoal_id, patch_r, now=now)
                            reallocated_to.append(ts_node.subgoal_id)

                unaffected_ids = {
                    t.task_id
                    for t in all_tasks
                    if t.status in ACTIVE_TASK_STATUSES and t.hypothesis_id not in sg_hids
                }
                _, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=unaffected_ids
                )
                return (
                    {
                        "goal_id": goal.goal_id,
                        "subgoal_id": sg.subgoal_id,
                        "status": SubGoalStatus.KILLED.value,
                        "cancelled_task_ids": cancelled_tasks,
                        "cancelled_hq_job_ids": cancelled_hq,
                        "cancelled_workflow_ids": cancelled_wfs,
                        "cancelled_decision_ids": cancelled_decs,
                        "pruned_hypothesis_ids": pruned_hyps,
                        "reallocated_to": reallocated_to,
                    },
                    before_epoch,
                    after_epoch,
                )

            if act == "retune":
                goal, sg = self._require_subgoal(goal_id, target_id, body)
                allowed_retune = {
                    "title",
                    "direction_thesis",
                    "direct_contribution",
                    "local_metric",
                    "target_value",
                    "baseline_value",
                    "direction",
                    "kill_criteria",
                    "compute_target",
                    "spec_paths",
                    "writable_prefixes",
                    "priority_weight",
                    "max_tokens_budget",
                    "max_auto_generations",
                    "wave_gated",
                    "base_sha",
                }
                patch = {k: v for k, v in body.items() if k in allowed_retune}
                if "priority_weight" in patch:
                    pw = float(patch["priority_weight"])
                    if not math.isfinite(pw) or pw <= 0.0:
                        raise ValueError("priority_weight must be finite and > 0")
                    patch["priority_weight"] = pw
                if "max_tokens_budget" in patch and int(patch["max_tokens_budget"]) < 0:
                    raise ValueError("max_tokens_budget must be >= 0")
                if "max_auto_generations" in patch and int(patch["max_auto_generations"]) < 1:
                    raise ValueError("max_auto_generations must be >= 1")
                if "direction" in patch and str(patch["direction"]).strip().lower() not in VALID_DIRECTIONS:
                    raise ValueError(f"Invalid direction: {patch['direction']!r}")
                updated_sg = self.db.update_subgoal_fields(sg.subgoal_id, patch, now=now) if patch else sg
                desired_st = body.get("status")
                if desired_st is not None:
                    updated_sg = self.db.transition_subgoal_status(
                        sg.subgoal_id, None, str(desired_st), now=now
                    )
                elif (
                    str(updated_sg.status).upper() == SubGoalStatus.AWAITING_APEX.value
                    and bool(body.get("reactivate", True))
                ):
                    updated_sg = self.db.transition_subgoal_status(
                        sg.subgoal_id,
                        SubGoalStatus.AWAITING_APEX.value,
                        SubGoalStatus.ACTIVE.value,
                        now=now,
                    )
                all_active = {
                    t.task_id
                    for t in self.db.list_tasks(goal_id=goal.goal_id)
                    if t.status in ACTIVE_TASK_STATUSES
                }
                _, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal,
                    bump_plan_epoch=bool(body.get("bump_plan_epoch", False)),
                    sync_unaffected_task_ids=all_active,
                )
                return (
                    {
                        "goal_id": goal.goal_id,
                        "subgoal_id": updated_sg.subgoal_id,
                        "status": updated_sg.status,
                        "subgoal": updated_sg.to_dict(),
                    },
                    before_epoch,
                    after_epoch,
                )

            if act == "spawn":
                goal = self._require_goal(goal_id or body.get("goal_id"))
                sg_id = str(body.get("subgoal_id") or target_id or f"sg_{uuid.uuid4().hex[:10]}").strip()
                title = str(body.get("title") or "").strip()
                thesis = str(body.get("direction_thesis") or body.get("thesis") or title).strip()
                contrib = str(body.get("direct_contribution") or body.get("contribution") or "").strip()
                local_metric = str(body.get("local_metric") or goal.primary_metric or "").strip()
                if not title or not thesis or not contrib or not local_metric:
                    raise ValueError(
                        "spawn subgoal requires non-empty title, direction_thesis, direct_contribution, and local_metric"
                    )
                deps = [str(x).strip() for x in (body.get("depends_on_subgoals") or []) if str(x).strip()]
                has_unsealed_dep = False
                dep_sha = ""
                for d_id in deps:
                    d_sg = self.db.get_goal_subgoal(goal.goal_id, d_id)
                    if d_sg is None or str(d_sg.status).upper() not in (
                        SubGoalStatus.SEALED.value,
                        SubGoalStatus.MERGED.value,
                    ):
                        has_unsealed_dep = True
                    elif d_sg.sealed_sha or d_sg.base_sha:
                        dep_sha = str(d_sg.sealed_sha or d_sg.base_sha)
                eff_status = str(
                    body.get("status")
                    or (SubGoalStatus.LOCKED_DEP.value if has_unsealed_dep else SubGoalStatus.ACTIVE.value)
                )
                raw_kind = body.get("kind", SubGoalKind.DIRECTION_TRACK.value)
                sg_node = SubGoalNode(
                    subgoal_id=sg_id,
                    goal_id=goal.goal_id,
                    title=title,
                    direction_thesis=thesis,
                    direct_contribution=contrib,
                    local_metric=local_metric,
                    kind=raw_kind if isinstance(raw_kind, SubGoalKind) else SubGoalKind(str(raw_kind)),
                    status=eff_status,
                    baseline_value=float(body.get("baseline_value", goal.baseline_value)),
                    target_value=None if body.get("target_value") is None else float(body["target_value"]),
                    direction=str(body.get("direction") or goal.direction),
                    kill_criteria=[str(x) for x in (body.get("kill_criteria") or goal.kill_criteria)],
                    depends_on_subgoals=deps,
                    track_branch=str(body.get("track_branch") or f"track/{goal.project}/{sg_id}"),
                    base_sha=str(body.get("base_sha") or dep_sha or goal.baseline_sha),
                    compute_target=str(body.get("compute_target") or "cpu_only"),
                    spec_paths=[str(x) for x in (body.get("spec_paths") or [])],
                    writable_prefixes=[str(x) for x in (body.get("writable_prefixes") or goal.writable_prefixes)],
                    priority_weight=float(body.get("priority_weight", 1.0)),
                    max_tokens_budget=int(body.get("max_tokens_budget", 0)),
                    max_auto_generations=int(body.get("max_auto_generations", 1)),
                    wave_gated=bool(body.get("wave_gated", True)),
                    params=dict(body.get("params") or {}),
                )
                stored_sg = self.db.save_subgoal(sg_node, now=now)
                created_hids: list[str] = []
                for idx, raw_s in enumerate(body.get("initial_slices") or body.get("proposals") or [], start=1):
                    if not isinstance(raw_s, Mapping):
                        continue
                    s_hid = str(raw_s.get("hypothesis_id") or f"hyp_{uuid.uuid4().hex[:10]}")
                    s_kind = HypothesisKind(raw_s.get("kind", HypothesisKind.COMPETING_ROUTE.value))
                    s_wp = [str(p) for p in (raw_s.get("writable_paths") or stored_sg.writable_prefixes or goal.writable_prefixes)]
                    s_title = str(raw_s.get("title") or f"{stored_sg.title} slice {idx}").strip()
                    s_mech = str(raw_s.get("mechanism") or s_title).strip()
                    s_screen = str(raw_s.get("screen_cmd") or "pytest -q")
                    s_params = {
                        **dict(raw_s.get("params") or {}),
                        "subgoal_id": stored_sg.subgoal_id,
                        "subgoal_local_metric": stored_sg.local_metric,
                        "subgoal_baseline_value": float(stored_sg.baseline_value),
                        "subgoal_direction": stored_sg.direction,
                        "wave_gated": stored_sg.wave_gated,
                    }
                    if stored_sg.spec_paths:
                        s_params.setdefault("spec_paths", list(stored_sg.spec_paths))
                    s_fp = str(raw_s.get("fingerprint") or "").strip() or compute_hypothesis_fingerprint(
                        goal_id=goal.goal_id,
                        kind=s_kind,
                        mechanism=s_mech,
                        params=s_params,
                        writable_paths=s_wp,
                        screen_cmd=s_screen,
                    )
                    h_node = HypothesisNode(
                        hypothesis_id=s_hid,
                        goal_id=goal.goal_id,
                        parent_id=raw_s.get("parent_id"),
                        kind=s_kind,
                        title=s_title,
                        statement=str(raw_s.get("statement") or s_title),
                        mechanism=s_mech,
                        fidelity=FidelityRung(raw_s.get("fidelity", FidelityRung.F1_FAST_SCREEN.value)),
                        priority_evoi=float(raw_s.get("priority_evoi", 1.2)),
                        writable_paths=s_wp,
                        params=s_params,
                        kill_criteria=[str(x) for x in (raw_s.get("kill_criteria") or stored_sg.kill_criteria)],
                        acceptance_cmd=str(raw_s.get("acceptance_cmd") or s_screen),
                        screen_cmd=s_screen,
                        confirm_cmd=raw_s.get("confirm_cmd", s_screen),
                        branch_name=str(raw_s.get("branch_name") or f"arr/{goal.project}/{s_hid}"),
                        base_sha=str(raw_s.get("base_sha") or stored_sg.base_sha or goal.baseline_sha),
                        status="PENDING",
                        fingerprint=s_fp,
                        subgoal_id=stored_sg.subgoal_id,
                    )
                    reg_h, _ = self.db.register_hypothesis(h_node)
                    created_hids.append(reg_h.hypothesis_id)
                all_active = {
                    t.task_id
                    for t in self.db.list_tasks(goal_id=goal.goal_id)
                    if t.status in ACTIVE_TASK_STATUSES
                }
                _, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=all_active
                )
                return (
                    {
                        "goal_id": goal.goal_id,
                        "subgoal_id": stored_sg.subgoal_id,
                        "status": stored_sg.status,
                        "subgoal": stored_sg.to_dict(),
                        "created_hypothesis_ids": created_hids,
                    },
                    before_epoch,
                    after_epoch,
                )

            if act == "merge":
                raw_sources = body.get("source_subgoal_ids") or body.get("sources") or []
                if isinstance(raw_sources, str):
                    source_ids = [s.strip() for s in raw_sources.split(",") if s.strip()]
                else:
                    source_ids = [str(s).strip() for s in raw_sources if str(s).strip()]
                if len(source_ids) < 2:
                    raise ValueError("merge subgoal requires at least 2 source_subgoal_ids")
                first_sg = self.db.get_subgoal(source_ids[0])
                eff_gid = goal_id or (first_sg.goal_id if first_sg is not None else None)
                goal = self._require_goal(eff_gid)
                src_nodes: list[SubGoalNode] = []
                merge_shas: list[str] = []
                for sid in source_ids:
                    sn = self.db.get_goal_subgoal(goal.goal_id, sid)
                    if sn is None:
                        raise KeyError(f"Unknown subgoal_id: {sid}")
                    src_nodes.append(sn)
                    sha = str(sn.sealed_sha or "")
                    if not sha:
                        for sh in self._subgoal_hypotheses(goal.goal_id, sid):
                            if sh.head_sha:
                                sha = str(sh.head_sha)
                                break
                    if sha:
                        merge_shas.append(sha)
                if not merge_shas:
                    raise ValueError(
                        f"merge subgoals {source_ids} requires at least one sealed_sha or head_sha"
                    )
                merge_sg_id = str(body.get("subgoal_id") or target_id or f"sg_merge_{uuid.uuid4().hex[:8]}").strip()
                primary_sha = merge_shas[0]
                merge_sg = SubGoalNode(
                    subgoal_id=merge_sg_id,
                    goal_id=goal.goal_id,
                    title=str(body.get("title") or f"Integration merge: {', '.join(source_ids)}"),
                    direction_thesis=str(
                        body.get("direction_thesis")
                        or f"Integrate sealed tracks {', '.join(source_ids)} and verify end-to-end metric"
                    ),
                    direct_contribution=str(
                        body.get("direct_contribution")
                        or f"Unified integration of {', '.join(source_ids)}"
                    ),
                    local_metric=str(body.get("local_metric") or goal.primary_metric),
                    kind=SubGoalKind.INTEGRATION_MERGE,
                    status=SubGoalStatus.ACTIVE.value,
                    baseline_value=float(body.get("baseline_value", goal.baseline_value)),
                    target_value=(
                        goal.target_value
                        if body.get("target_value") is None
                        else float(body["target_value"])
                    ),
                    direction=str(body.get("direction") or goal.direction),
                    kill_criteria=[str(x) for x in (body.get("kill_criteria") or goal.kill_criteria)],
                    depends_on_subgoals=[],
                    source_subgoal_ids=source_ids,
                    track_branch=f"track/{goal.project}/{merge_sg_id}",
                    base_sha=primary_sha,
                    writable_prefixes=[str(x) for x in (body.get("writable_prefixes") or goal.writable_prefixes)],
                    priority_weight=float(body.get("priority_weight", 1.5)),
                    wave_gated=bool(body.get("wave_gated", True)),
                    params={"merge_source_shas": merge_shas},
                )
                stored_merge = self.db.save_subgoal(merge_sg, now=now)
                integ_hid: str | None = None
                if bool(body.get("create_hypothesis", True)):
                    ihid = str(body.get("hypothesis_id") or f"hyp_merge_{uuid.uuid4().hex[:8]}")
                    i_wp = list(stored_merge.writable_prefixes or goal.writable_prefixes or ["src/"])
                    i_cmd = str(body.get("screen_cmd") or "pytest -q")
                    i_params = {
                        "merge_source_shas": merge_shas,
                        "merge_subgoal_ids": source_ids,
                        "requires_llm_code": True,
                        "subgoal_id": stored_merge.subgoal_id,
                    }
                    i_fp = compute_hypothesis_fingerprint(
                        goal_id=goal.goal_id,
                        kind=HypothesisKind.COMPETING_ROUTE,
                        mechanism=f"integration_merge::{'+'.join(source_ids)}",
                        params=i_params,
                        writable_paths=i_wp,
                        screen_cmd=i_cmd,
                    )
                    i_node = HypothesisNode(
                        hypothesis_id=ihid,
                        goal_id=goal.goal_id,
                        parent_id=None,
                        kind=HypothesisKind.COMPETING_ROUTE,
                        title=f"Integrate {', '.join(source_ids)}",
                        statement=stored_merge.direction_thesis,
                        mechanism=f"integration_merge::{'+'.join(source_ids)}",
                        fidelity=FidelityRung.F1_FAST_SCREEN,
                        priority_evoi=1.8,
                        writable_paths=i_wp,
                        params=i_params,
                        kill_criteria=list(stored_merge.kill_criteria),
                        acceptance_cmd=str(body.get("acceptance_cmd") or i_cmd),
                        screen_cmd=i_cmd,
                        confirm_cmd=body.get("confirm_cmd", i_cmd),
                        branch_name=f"arr/{goal.project}/{ihid}",
                        base_sha=primary_sha,
                        status="PENDING",
                        fingerprint=i_fp,
                        subgoal_id=stored_merge.subgoal_id,
                    )
                    reg_i, _ = self.db.register_hypothesis(i_node)
                    integ_hid = reg_i.hypothesis_id
                all_active = {
                    t.task_id
                    for t in self.db.list_tasks(goal_id=goal.goal_id)
                    if t.status in ACTIVE_TASK_STATUSES
                }
                _, before_epoch, after_epoch = self._bump_goal_epoch(
                    goal, sync_unaffected_task_ids=all_active
                )
                return (
                    {
                        "goal_id": goal.goal_id,
                        "subgoal_id": stored_merge.subgoal_id,
                        "source_subgoal_ids": source_ids,
                        "merge_source_shas": merge_shas,
                        "status": stored_merge.status,
                        "integration_hypothesis_id": integ_hid,
                    },
                    before_epoch,
                    after_epoch,
                )

            raise ValueError(f"Unsupported subgoal intervention action: {act!r}")


__all__ = [
    "DEFAULT_FOREGROUND_RESERVE_SLOTS",
    "DEFAULT_GLOBAL_MAX_TOKENS_BUDGET",
    "GlobalGoalManager",
    "VALID_AUTONOMY_MODES",
]
