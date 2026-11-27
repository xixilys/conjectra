"""Lightweight deterministic event-driven control core (0 resident LLM) for conjectra."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import errno
import json
import os
import re
import selectors
import signal
import socket
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from conjectra import gates
from conjectra.adapters import get_adapter
from conjectra.arbiter import (
    ConflictManager,
    GlobalResourceScheduler,
    ResourceArbiter,
    read_proc_start_time,
    verify_process_identity,
)
from conjectra.decision_engine import DEFAULT_COALESCE_WINDOW_S, DecisionEngine
from conjectra.explore import ExploreEngine, check_kill_criteria
from conjectra.goal_manager import GlobalGoalManager
from conjectra.hq_runtime import HQRuntime
from conjectra.model_router import ModelRouter, PolicyBlockedError
from conjectra.stage_runner import StageRunner
from conjectra.state_db import (
    FencedStaleWriteError,
    StateDB,
    StateTransitionError,
    evaluate_promotion_git_state,
    resolve_arr_home,
)
from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    FailureDomain,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ResourceSpec,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
)
from conjectra.workflow_runtime import TERMINAL_WORKFLOW_STATUSES, WorkflowRuntime
from conjectra.workload_matrix import record_workload_eval

TERMINAL_TASK_STATES = frozenset(
    {
        "DONE",
        "SUCCEEDED",
        "PASSED",
        "FAILED",
        "CANCELLED",
        "DEAD",
        "TIMED_OUT",
        "BLOCKED_POLICY",
        "BLOCKED_QUOTA",
    }
)
ACTIVE_TASK_STATES = frozenset({"DISPATCHED", "RUNNING"})
TEST_SIM_STAGES = frozenset({StageKind.CPU_GATE, StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM})
READONLY_REVIEW_STAGES = frozenset(
    {
        StageKind.DESIGN_REVIEW,
        StageKind.IMPL_REVIEW,
        StageKind.RESULT_REVIEW,
        StageKind.BLIND_AUDIT,
    }
)

OWNER_GATE_GOAL_BAR = "GOAL_OR_BAR_CHANGE"
OWNER_GATE_MAIN_HOLDOUT = "MAIN_OR_HOLDOUT_TOUCH"
OWNER_GATE_PAID_COMPUTE = "NEW_PAID_COMPUTE"
OWNER_GATE_ROUTE_AUDIT = "ROUTE_REFUTATION_OR_AUDIT_FAIL"

ALLOWED_OWNER_GATES = frozenset(
    {
        OWNER_GATE_GOAL_BAR,
        OWNER_GATE_MAIN_HOLDOUT,
        OWNER_GATE_PAID_COMPUTE,
        OWNER_GATE_ROUTE_AUDIT,
    }
)


def send_reactor_wake(
    arr_home: str | Path | None,
    payload: Mapping[str, Any] | None = None,
    *,
    timeout_s: float = 1.5,
) -> bool:
    """Send an instant wakeup event to <CONJECTRA_HOME>/reactor.sock."""
    home = resolve_arr_home(arr_home)
    sock_path = home / "reactor.sock"
    if not sock_path.exists():
        return False
    msg_dict = dict(payload or {"event": "WAKE", "ts": time.time()})
    raw = (json.dumps(msg_dict, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")
    for sock_type in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        try:
            with socket.socket(socket.AF_UNIX, sock_type) as sock:
                sock.settimeout(timeout_s)
                sock.connect(str(sock_path))
                sock.sendall(raw)
            return True
        except OSError:
            continue
    return False


def _task_idempotency_key(
    goal_id: str,
    hypothesis_id: str,
    stage: StageKind,
    attempt: int,
    *,
    workflow_id: str | None = None,
    step_key: str = "",
) -> str:
    if workflow_id:
        return f"{goal_id}:{hypothesis_id}:{workflow_id}:{step_key or stage.value}:a{attempt}"
    return f"{goal_id}:{hypothesis_id}:{step_key or stage.value}:a{attempt}"


def _make_stage_task(
    goal: ResearchGoal,
    hyp: HypothesisNode,
    stage: StageKind,
    *,
    deps: Sequence[str] = (),
    attempt: int = 1,
    worktree_path: str | None = None,
    resources_override: ResourceSpec | None = None,
    workflow_id: str | None = None,
    workflow_primitive: str = "direct_hq",
    step_key: str = "",
    pinned_route: Mapping[str, Any] | None = None,
) -> StageTask:
    adapter = get_adapter(goal.adapter_name or goal.project)
    res = resources_override or adapter.default_resource_spec(stage, hyp, goal)
    if stage == StageKind.CODE and hyp.kind in (
        HypothesisKind.PARAMETER_RACE,
        HypothesisKind.ABLATION,
        HypothesisKind.FALSIFICATION,
    ) and not hyp.params.get("requires_llm_code", False):
        res = ResourceSpec(
            cpus=res.cpus,
            ram_mb=res.ram_mb,
            gpu=res.gpu,
            llm_pool=None,
            eda_sim=res.eda_sim,
            cloud_slot=res.cloud_slot,
            wall_timeout_s=res.wall_timeout_s,
        )
    route_dict = dict(pinned_route) if isinstance(pinned_route, Mapping) else None
    if route_dict is not None:
        res = ResourceSpec.from_dict(res.to_dict())
        res.llm_pool = str(route_dict.get("pool") or res.llm_pool or "") or None
        res.hq_llm_resource = route_dict.get("hq_resource")
        res.pinned_route = dict(route_dict)
    return StageTask(
        task_id=f"tsk-{hyp.hypothesis_id}-{stage.value.lower()}-a{attempt}-{uuid.uuid4().hex[:6]}",
        idempotency_key=_task_idempotency_key(
            goal.goal_id,
            hyp.hypothesis_id,
            stage,
            attempt,
            workflow_id=workflow_id,
            step_key=step_key,
        ),
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
        workflow_id=workflow_id,
        goal_version=int(getattr(goal, "version", 1)),
        human_epoch=int(getattr(goal, "human_epoch", 0)),
        workflow_primitive=workflow_primitive,
        step_key=step_key,
        pinned_route=route_dict,
    )


def get_owner_escalations(db: StateDB, *, goal_id: str | None = None) -> list[dict[str, Any]]:
    """Return recorded Owner-Only Gate escalations from the ledger."""
    events = db.list_events(goal_id=goal_id, since_seq=0, limit=1000)
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for ev in events:
        if ev.get("event_type") != "OWNER_ESCALATION":
            continue
        p = ev.get("payload") or {}
        gate = str(p.get("gate") or "")
        if gate not in ALLOWED_OWNER_GATES:
            continue
        gid = str(ev.get("goal_id") or p.get("goal_id") or "")
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
                "details": p.get("details") or {},
            }
        )
    return out


@dataclass
class GoalReactorContext:
    """Independent per-Goal state inside the single 0-token Multi-Goal daemon."""

    goal_id: str
    last_propose_receipt_count: int = -1
    last_decision_receipt_count: int = -1
    failed_blind_audits: int = 0
    decision_blocked: bool = False
    last_step_ts: float = 0.0


class ResearchController:
    """Deterministic event-driven Multi-Goal reactor and pipeline coordinator (0 resident LLM)."""

    def __init__(
        self,
        arr_home: str | Path | None = None,
        *,
        db: StateDB | None = None,
        hq: HQRuntime | None = None,
        arbiter: ResourceArbiter | None = None,
        scheduler: GlobalResourceScheduler | None = None,
        explore: ExploreEngine | None = None,
        stage_runner: StageRunner | None = None,
        goal_manager: GlobalGoalManager | None = None,
        workflow_runtime: WorkflowRuntime | None = None,
        decision_engine: DecisionEngine | None = None,
        router: ModelRouter | None = None,
        execution_mode: str = "hq",
        propose_mode: str = "auto",
        dry_run_llm: bool = False,
        sweep_interval_s: float = 15.0,
        max_concurrent_test_sim: int = 2,
        decision_coalesce_window_s: float = DEFAULT_COALESCE_WINDOW_S,
        use_workflows: bool = True,
    ) -> None:
        self.arr_home = resolve_arr_home(arr_home)
        self.arr_home.mkdir(parents=True, exist_ok=True)
        self._own_db = db is None
        self.db = db or StateDB(self.arr_home)
        self.sock_path = self.arr_home / "reactor.sock"
        self.pid_path = self.arr_home / "controller.pid"
        self.execution_mode = execution_mode
        self.propose_mode = propose_mode
        self.dry_run_llm = dry_run_llm
        self.sweep_interval_s = float(sweep_interval_s)
        self.max_concurrent_test_sim = max(1, int(max_concurrent_test_sim))
        self.decision_coalesce_window_s = max(0.0, float(decision_coalesce_window_s))
        self.use_workflows = bool(use_workflows)
        self._last_status_export_ts: float = 0.0

        self.hq = hq or HQRuntime(
            self.arr_home,
            reactor_socket=self.sock_path,
        )
        self.arbiter = arbiter or ResourceArbiter()
        self.router = router or ModelRouter(arbiter=self.arbiter, db=self.db)
        self.scheduler = scheduler or GlobalResourceScheduler(
            self.arbiter,
            arr_home=self.arr_home,
            global_test_sim_cap=self.max_concurrent_test_sim,
        )
        self.explore = explore or ExploreEngine(self.db)
        self.stage_runner = stage_runner or StageRunner(arr_home=self.arr_home)
        self.goal_manager = goal_manager or GlobalGoalManager(self.db, hq=self.hq)
        self.workflow_runtime = workflow_runtime or WorkflowRuntime(
            self.db, arbiter=self.arbiter, router=self.router
        )
        self.decision_engine = decision_engine or DecisionEngine(
            self.db, hq=self.hq, arbiter=self.arbiter, router=self.router
        )

        self._handled_tasks: set[str] = set()
        self._recorded_escalations: set[tuple[str, str, str]] = set()
        self._handled_decision_ids: set[str] = set()
        self._goal_contexts: dict[str, GoalReactorContext] = {}
        self._last_propose_receipt_count: dict[str, int] = {}
        self._failed_blind_audits: dict[str, int] = {}
        self._stop_requested = False
        self._wake_write_fd: int | None = None
        self._last_status_export_ts: float = 0.0
        self._consecutive_barrier_lock_failures: int = 0
        self._max_consecutive_barrier_lock_failures: int = 5
        self._hydrate_from_events()
        self._reconcile_maintenance_barrier({})
        self._reconcile_promotion_intents({})

    def _goal_ctx(self, goal_id: str) -> GoalReactorContext:
        ctx = self._goal_contexts.get(goal_id)
        if ctx is None:
            ctx = GoalReactorContext(
                goal_id=goal_id,
                last_propose_receipt_count=self._last_propose_receipt_count.get(goal_id, -1),
                failed_blind_audits=self._failed_blind_audits.get(goal_id, 0),
            )
            self._goal_contexts[goal_id] = ctx
        return ctx

    def close(self) -> None:
        """Close owned resources."""
        if self._own_db:
            self.db.close()

    def _hydrate_from_events(self) -> None:
        events = self.db.list_events(since_seq=0, limit=5000)
        for ev in events:
            etype = ev.get("event_type")
            p = ev.get("payload") or {}
            if etype == "CONTROLLER_TASK_HANDLED":
                tid = ev.get("task_id") or p.get("task_id")
                if tid:
                    self._handled_tasks.add(str(tid))
            elif etype == "OWNER_ESCALATION":
                gid = str(ev.get("goal_id") or p.get("goal_id") or "")
                gate = str(p.get("gate") or "")
                reason = str(p.get("reason") or "")
                self._recorded_escalations.add((gid, gate, reason))
            elif etype == "BLIND_AUDIT_FAILED":
                gid = str(ev.get("goal_id") or "")
                if gid:
                    self._failed_blind_audits[gid] = self._failed_blind_audits.get(gid, 0) + 1
                    self._goal_ctx(gid).failed_blind_audits = self._failed_blind_audits[gid]
            elif etype == "CONTROLLER_PROPOSE_ROUND":
                gid = str(ev.get("goal_id") or "")
                cnt = int(p.get("receipts_count", 0))
                if gid:
                    prev = self._last_propose_receipt_count.get(gid, -1)
                    self._last_propose_receipt_count[gid] = max(prev, cnt)
                    self._goal_ctx(gid).last_propose_receipt_count = max(
                        self._goal_ctx(gid).last_propose_receipt_count, cnt
                    )
            elif etype == "CONTROLLER_DECISION_APPLIED":
                rid = str(p.get("req_id") or "")
                if rid:
                    self._handled_decision_ids.add(rid)

    def _record_ledger_event(
        self,
        event_type: str,
        *,
        goal_id: str | None = None,
        hypothesis_id: str | None = None,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        with self.db._tx() as (conn, pending):
            self.db._record_event(
                conn,
                pending,
                event_type,
                goal_id=goal_id,
                hypothesis_id=hypothesis_id,
                task_id=task_id,
                payload=payload,
            )

    @staticmethod
    def _is_fatal_authority_or_db_error(exc: BaseException) -> bool:
        """Return True for fatal SQLite/authority/I/O failures that must fail closed & propagate."""
        if isinstance(exc, sqlite3.DatabaseError):
            msg = str(exc).lower()
            if isinstance(exc, sqlite3.OperationalError) and "locked" in msg and "disk" not in msg:
                return False
            return True
        if exc.__class__.__name__ == "StaleRecoveryLeaseError":
            return True
        if isinstance(exc, OSError) and getattr(exc, "errno", None) in (
            errno.EIO,
            errno.ENOSPC,
            errno.EROFS,
            errno.EBADF,
        ):
            return True
        msg = str(exc).lower()
        return any(
            tok in msg
            for tok in (
                "database disk image is malformed",
                "file is not a database",
                "disk i/o error",
                "maintenance_barrier",
                "authority failure",
            )
        )

    def _emit_db_independent_fatal_marker(
        self,
        exc: BaseException,
        *,
        phase: str,
        goal_id: str | None = None,
    ) -> None:
        """Write fatal authority/DB failure marker directly to filesystem & stderr without touching SQLite."""
        rec = {
            "ts": time.time(),
            "pid": os.getpid(),
            "phase": phase,
            "goal_id": goal_id,
            "exc_type": type(exc).__name__,
            "error": str(exc),
        }
        line = json.dumps(rec, sort_keys=True, ensure_ascii=False)
        with contextlib.suppress(OSError):
            log_dir = self.arr_home / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "daemon_fatal.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        with contextlib.suppress(OSError):
            sys.stderr.write(f"CONJECTRA_DAEMON_FATAL: {line}\n")
            sys.stderr.flush()

    def _isolate_step_exception(
        self,
        exc: BaseException,
        *,
        phase: str,
        goal_id: str | None = None,
        hypothesis_id: str | None = None,
        task_id: str | None = None,
        summary: dict[str, Any] | None = None,
    ) -> None:
        """Isolate task/hypothesis-local failures while propagating fatal DB/authority errors fail-closed."""
        if self._is_fatal_authority_or_db_error(exc):
            self._emit_db_independent_fatal_marker(exc, phase=phase, goal_id=goal_id)
            raise exc
        err_entry = {
            "phase": phase,
            "goal_id": goal_id,
            "hypothesis_id": hypothesis_id,
            "task_id": task_id,
            "exc_type": type(exc).__name__,
            "error": str(exc),
        }
        if summary is not None:
            summary["errors"] = int(summary.get("errors", 0)) + 1
            summary.setdefault("step_errors", []).append(err_entry)
        with contextlib.suppress(Exception):
            self._record_ledger_event(
                "DAEMON_STEP_ERROR",
                goal_id=goal_id,
                hypothesis_id=hypothesis_id,
                task_id=task_id,
                payload=err_entry,
            )
        if hypothesis_id:
            with contextlib.suppress(Exception):
                h_obj = self.db.get_hypothesis(hypothesis_id)
                if h_obj is not None and h_obj.status not in (
                    "COMPLETED",
                    "REFUTED",
                    "PRUNED",
                    "NEEDS_REPAIR",
                    "HUMAN_INVALIDATED",
                ):
                    try:
                        self.db.transition_hypothesis_status(
                            h_obj.hypothesis_id,
                            h_obj.status,
                            "NEEDS_REPAIR",
                            head_sha=h_obj.head_sha,
                            failure_domain=FailureDomain.ENGINEERING_DEFECT,
                        )
                    except Exception:
                        self.db.update_hypothesis_status(
                            h_obj.hypothesis_id,
                            status="NEEDS_REPAIR",
                            failure_domain=FailureDomain.ENGINEERING_DEFECT,
                        )

    def flag_owner_escalation(
        self,
        goal_id: str,
        gate: str,
        reason: str,
        *,
        hypothesis_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> bool:
        """Record one of the 4 Owner-Only Gates if not already flagged."""
        if gate not in ALLOWED_OWNER_GATES:
            raise ValueError(f"Unsupported owner escalation gate: {gate!r}")
        key = (goal_id, gate, reason)
        if key in self._recorded_escalations:
            return False
        self._recorded_escalations.add(key)
        self._record_ledger_event(
            "OWNER_ESCALATION",
            goal_id=goal_id,
            hypothesis_id=hypothesis_id,
            payload={
                "goal_id": goal_id,
                "gate": gate,
                "reason": reason,
                "details": details or {},
            },
        )
        return True

    def _enqueue_stage(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        stage: StageKind,
        *,
        deps: Sequence[str] = (),
        attempt: int = 1,
        worktree_path: str | None = None,
        resources_override: ResourceSpec | None = None,
        workflow_id: str | None = None,
        workflow_primitive: str = "direct_hq",
        step_key: str = "",
    ) -> StageTask:
        task = _make_stage_task(
            goal,
            hyp,
            stage,
            deps=deps,
            attempt=attempt,
            worktree_path=worktree_path,
            resources_override=resources_override,
            workflow_id=workflow_id,
            workflow_primitive=workflow_primitive,
            step_key=step_key,
        )
        return self.db.enqueue_task(task)

    def _enqueue_initial_hypothesis_work(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
    ) -> list[StageTask]:
        """Start execution for a newly accepted hypothesis via WorkflowRuntime or direct stage."""
        existing = self.db.list_tasks(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id)
        if existing:
            return []
        if self.use_workflows:
            existing_wfs = self.db.list_workflows(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id)
            if not existing_wfs:
                try:
                    _wf, tasks = self.workflow_runtime.create_workflow_for_hypothesis(goal, hyp)
                    return tasks
                except ValueError as exc:
                    self.db.update_hypothesis_status(
                        hyp.hypothesis_id, status="BLOCKED_INVALID_WORKFLOW"
                    )
                    self._record_ledger_event(
                        "WORKFLOW_SPEC_REJECTED",
                        goal_id=goal.goal_id,
                        hypothesis_id=hyp.hypothesis_id,
                        payload={"error": str(exc)},
                    )
                    return []
                except Exception as exc:
                    self._isolate_step_exception(
                        exc,
                        phase="enqueue_initial_hypothesis_work",
                        goal_id=goal.goal_id,
                        hypothesis_id=hyp.hypothesis_id,
                    )
                    return []
            return []
        return [self._enqueue_stage(goal, hyp, StageKind.CODE)]

    def _record_knowledge_and_eval_ledger(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        task: StageTask,
        receipt: EvidenceReceipt,
    ) -> None:
        """Persist empirical knowledge items and real-LLM workload_eval_ledger rows."""
        if receipt.scientific_verdict in (
            ScientificVerdict.CONFIRMED,
            ScientificVerdict.REFUTED,
            ScientificVerdict.VALID_NEGATIVE,
        ) or receipt.audit_verdict in (AuditVerdict.PASS, AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR):
            cat = (
                "audit"
                if task.stage == StageKind.BLIND_AUDIT
                else ("negative_lesson" if receipt.scientific_verdict in (ScientificVerdict.REFUTED, ScientificVerdict.VALID_NEGATIVE) else "confirmed_finding")
            )
            self.db.upsert_knowledge_item(
                goal.goal_id,
                cat,
                f"hyp:{hyp.hypothesis_id}:{task.stage.value}",
                {
                    "hypothesis_id": hyp.hypothesis_id,
                    "mechanism": hyp.mechanism,
                    "stage": task.stage.value,
                    "scientific_verdict": receipt.scientific_verdict.value,
                    "audit_verdict": receipt.audit_verdict.value,
                    "evidence_validity": receipt.evidence_validity.value,
                    "metrics": receipt.metrics,
                    "failure_reason": receipt.failure_reason,
                },
                hypothesis_id=hyp.hypothesis_id,
                goal_version=int(goal.version),
            )

        if self.dry_run_llm:
            return
        usage_paths = [
            p
            for p in receipt.raw_artifact_paths
            if p.endswith("usage.json") and Path(p).is_file()
        ]
        uncached_total = int(receipt.in_tokens) + int(receipt.out_tokens)
        if not usage_paths or uncached_total <= 0:
            return
        valid_usage: list[str] = []
        for up in usage_paths:
            try:
                udoc = json.loads(Path(up).read_text(encoding="utf-8"))
                if udoc.get("estimated") is True or udoc.get("synthetic") is True:
                    continue
                valid_usage.append(up)
            except (OSError, json.JSONDecodeError):
                continue
        if not valid_usage:
            return
        cache_entry = (receipt.metrics or {}).get("cache_read_tokens")
        cache_tok = int(cache_entry.get("value", 0)) if isinstance(cache_entry, Mapping) else 0
        passed = (
            receipt.process_status == ProcessStatus.EXIT_OK
            and receipt.engineering_verdict == EngineeringVerdict.PASS
            and receipt.audit_verdict not in (AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR)
        )
        try:
            record_workload_eval(
                self.db,
                workload_class="tight_adversarial" if task.workflow_primitive == "x-boost" else "disjoint_parallel",
                mode=task.workflow_primitive or "direct_hq",
                provider_pairing=receipt.model_used or "unknown",
                is_real_llm=True,
                hq_used=(self.execution_mode == "hq"),
                holdout_cmd=receipt.command,
                holdout_rc=int(receipt.exit_code),
                holdout_passed=passed,
                defect_escaped=(receipt.audit_verdict in (AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR)),
                gate_intercepted=(receipt.engineering_verdict == EngineeringVerdict.FAIL_GATE),
                uncached_tokens=uncached_total,
                cache_read_tokens=cache_tok,
                wall_sec=float(receipt.wall_sec),
                raw_usage_paths=valid_usage,
                head_sha=receipt.head_sha,
                source="controller_runtime",
            )
        except Exception:
            pass

    def _synthesize_missing_receipt(
        self,
        task: StageTask,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        process_status: ProcessStatus,
        reason: str,
    ) -> EvidenceReceipt:
        log_candidates = [
            str(self.arr_home / "logs" / "jobs" / f"{task.task_id}.stderr"),
            str(self.arr_home / "logs" / "jobs" / f"{task.task_id}.stdout"),
        ]
        receipt = EvidenceReceipt(
            receipt_id=f"rcpt-synth-{task.task_id}-a{int(task.attempt)}-{uuid.uuid4().hex[:6]}",
            task_id=task.task_id,
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=task.stage,
            base_sha=hyp.base_sha,
            head_sha=hyp.head_sha or hyp.base_sha,
            command=f"stage:{task.stage.value}",
            exit_code=-1,
            wall_sec=(
                max(0.0, (task.finished_ts or time.time()) - (task.started_ts or task.enqueued_ts))
                if task.started_ts
                else 0.0
            ),
            max_rss_mb=0.0,
            in_tokens=0,
            out_tokens=0,
            model_used="none",
            process_status=process_status,
            engineering_verdict=EngineeringVerdict.FAIL_TEST,
            scientific_verdict=ScientificVerdict.UNKNOWN,
            audit_verdict=AuditVerdict.NOT_REQUIRED,
            tests_passed=0,
            tests_total=0,
            metrics={"attempt": int(task.attempt)},
            raw_artifact_paths=log_candidates,
            failure_reason=reason,
            created_ts=time.time(),
        )
        return self.db.record_receipt(receipt)

    def _reconcile_promotion_intents(self, summary: dict[str, Any]) -> None:
        """Reconcile leftover `PREPARED` two-phase Git promotion intents under shared `promotion_branch_lock`."""
        if not hasattr(self.db, "list_promotion_intents"):
            return
        now = time.time()
        active_leases = self.db.list_active_leases(now=now)
        live_lease_tids = {str(ls.get("task_id") or "") for ls in active_leases}
        for row in self.db.list_promotion_intents(state="PREPARED"):
            intent_id = str(row["intent_id"])
            gid = str(row["goal_id"])
            hid = str(row["hypothesis_id"])
            hyp_obj = self.db.get_hypothesis(hid)
            rec_branch = str(row.get("branch_name") or "").strip()
            lock_branch = rec_branch or (hyp_obj.branch_name if hyp_obj is not None else "")
            lock_ctx = (
                self.db.promotion_branch_lock(gid, lock_branch, nonblocking=True)
                if hasattr(self.db, "promotion_branch_lock")
                else None
            )
            if lock_ctx is not None:
                with lock_ctx as got_prom_lock:
                    if not got_prom_lock:
                        # Active StageRunner holds the promotion critical-section lock; never abort mid-FF!
                        continue
                    self._reconcile_single_promotion_intent_locked(
                        intent_id,
                        gid=gid,
                        exp_branch=rec_branch,
                        live_lease_tids=live_lease_tids,
                        summary=summary,
                        now=now,
                    )
            else:
                self._reconcile_single_promotion_intent_locked(
                    intent_id,
                    gid=gid,
                    exp_branch=rec_branch,
                    live_lease_tids=live_lease_tids,
                    summary=summary,
                    now=now,
                )

    def _reconcile_single_promotion_intent_locked(
        self,
        intent_id: str,
        *,
        gid: str,
        exp_branch: str,
        live_lease_tids: set[str],
        summary: dict[str, Any],
        now: float,
    ) -> None:
        fresh_rows = [
            r
            for r in self.db.list_promotion_intents(goal_id=gid, state="PREPARED")
            if str(r["intent_id"]) == intent_id
        ]
        if not fresh_rows:
            return
        row = fresh_rows[0]
        tid = str(row["task_id"])
        att = int(row["attempt"])
        t_obj = self.db.get_task(tid)
        if (
            t_obj is not None
            and t_obj.status in ("DISPATCHED", "RUNNING")
            and int(t_obj.attempt) == att
            and tid in live_lease_tids
        ):
            return

        wt_raw = str(row.get("wt_path") or "")
        cand_sha = str(row.get("candidate_sha") or "")
        base_sha = str(row.get("expected_base_sha") or "")
        sanitized_tid = re.sub(r"[^A-Za-z0-9._-]+", "_", tid)
        ref_names = list(
            dict.fromkeys(
                [
                    f"refs/arr-promote/{tid}-a{att}",
                    f"refs/arr-promote/{sanitized_tid}-a{att}",
                ]
            )
        )
        wt_p = Path(wt_raw) if wt_raw else None
        git_st = evaluate_promotion_git_state(wt_p, exp_branch, base_sha, cand_sha)
        actual_head = str(git_st["actual_head"])
        actual_branch = str(git_st["actual_branch"])
        branch_ok = bool(git_st["branch_ok"])
        branch_confirmed_unpromoted = bool(git_st["branch_confirmed_unpromoted"])
        goal_obj = self.db.get_goal(gid)
        version_ok = goal_obj is None or int(goal_obj.version) == int(row["goal_version"])
        delete_refs = False
        if cand_sha and actual_head == cand_sha and branch_ok:
            self.db.commit_promotion_intent(
                intent_id,
                candidate_sha=cand_sha,
                update_hypothesis_head=True,
                reason=(
                    "crash_recovered_roll_forward"
                    if version_ok
                    else "crash_recovered_roll_forward_post_version_bump"
                ),
                now=now,
            )
            summary["promotions_recovered_committed"] = (
                summary.get("promotions_recovered_committed", 0) + 1
            )
            delete_refs = True
        elif (bool(base_sha) and actual_head == base_sha and branch_ok) or branch_confirmed_unpromoted:
            self.db.abort_promotion_intent(
                intent_id,
                reason=(
                    f"crash_recovered_abort:head={actual_head or 'unavailable'},"
                    f"branch={actual_branch or 'detached'},exp_branch={exp_branch or '*'},base={base_sha}"
                ),
                now=now,
            )
            summary["promotions_recovered_aborted"] = (
                summary.get("promotions_recovered_aborted", 0) + 1
            )
            delete_refs = True
        else:
            # Ambiguous/unverifiable git state: keep durable `PREPARED` (`UNRESOLVED_FF_AMBIGUOUS`) and preserve `refs/arr-promote/*`!
            unres_reason = (
                f"UNRESOLVED_FF_AMBIGUOUS:head={actual_head or 'unavailable'},"
                f"branch={actual_branch or 'detached'},base={base_sha}"
            )[:500]
            if hasattr(self.db, "mark_promotion_intent_unresolved"):
                self.db.mark_promotion_intent_unresolved(
                    intent_id, reason=unres_reason, now=now
                )
            summary["promotions_unresolved_ambiguous"] = (
                summary.get("promotions_unresolved_ambiguous", 0) + 1
            )

        if delete_refs and wt_p is not None and wt_p.is_dir():
            for ref_n in ref_names:
                try:
                    subprocess.run(
                        ["git", "-C", str(wt_p), "update-ref", "-d", ref_n],
                        capture_output=True,
                        text=True,
                        timeout=10,
                        check=False,
                    )
                except Exception:
                    pass

    def _is_hq_cancellation_confirmed(
        self,
        jid: int | None,
        *,
        goal_id: str | None = None,
        entity_kind: str | None = None,
        entity_id: str = "",
        attempt: int | None = None,
        usage_key: str | None = None,
        now: float | None = None,
    ) -> bool:
        """Return True ONLY when HQ job cancellation is durably confirmed or positively terminated (`QUERY_ERROR` -> False)."""
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
                and (not entity_id or not r.get("entity_id") or str(r.get("entity_id")) == entity_id)
                and (attempt is None or not r.get("attempt") or int(r.get("attempt") or 0) == int(attempt))
                and (not usage_key or not r.get("usage_key") or str(r.get("usage_key")) == str(usage_key))
                for r in confirmed_rows
            ):
                return True

        q_state = self.goal_manager._query_hq_job_state(self.hq, int(jid))
        lookup_status = str(q_state.get("lookup_status") or "QUERY_ERROR").upper()
        if lookup_status == "QUERY_ERROR":
            return False
        if lookup_status == "FOUND":
            if self.goal_manager._hq_positively_terminated(
                self.hq, int(jid), prefetched_info=q_state.get("job")
            ):
                if hasattr(self.db, "confirm_hq_cancellation"):
                    self.db.confirm_hq_cancellation(
                        int(jid),
                        entity_kind=entity_kind,
                        entity_id=entity_id or None,
                        attempt=attempt,
                        usage_key=usage_key,
                        now=ts,
                    )
                return True
            return False
        if lookup_status == "NOT_FOUND":
            has_live_lease = bool(
                entity_id
                and any(
                    str(ls.get("task_id") or "") in {entity_id, f"dec-{entity_id}"}
                    for ls in self.db.list_active_leases(now=ts)
                )
            )
            if not has_live_lease:
                if hasattr(self.db, "confirm_hq_cancellation"):
                    self.db.confirm_hq_cancellation(
                        int(jid),
                        entity_kind=entity_kind,
                        entity_id=entity_id or None,
                        attempt=attempt,
                        usage_key=usage_key,
                        now=ts,
                    )
                return True
        return False

    def _sweep_leases_and_hq_jobs(self, summary: dict[str, Any]) -> None:
        now = time.time()
        reaped = self.db.reap_expired_leases(now=now)
        summary["leases_reaped"] += len(reaped)
        active_leases = self.db.list_active_leases(now=now)
        if active_leases:
            rec = self.arbiter.recover_stale_leases(active_leases, now_ts=now)
            for stale in rec["stale_to_requeue"]:
                self.db.release_lease(stale["lease_id"], now=now)
                summary["leases_reaped"] += 1

        self._reconcile_promotion_intents(summary)

        if self.execution_mode != "hq":
            return

        if hasattr(self.db, "list_pending_hq_cancellations"):
            pending_cancels = self.db.list_pending_hq_cancellations(state="CANCEL_PENDING")
            if pending_cancels:
                jids_to_cancel = sorted({int(r["hq_job_id"]) for r in pending_cancels})
                try:
                    self.hq.cancel_jobs(jids_to_cancel, reason="daemon_drain_pending_hq_cancellations")
                except Exception:
                    pass
                for row_c in pending_cancels:
                    jid_c = int(row_c["hq_job_id"])
                    ent_c = str(row_c.get("entity_id") or "")
                    ek_c = str(row_c.get("entity_kind") or "") or None
                    att_c = int(row_c.get("attempt") or 0) or None
                    gid_c = str(row_c.get("goal_id") or "") or None
                    uk_c = str(row_c.get("usage_key") or "") or None
                    if self._is_hq_cancellation_confirmed(
                        jid_c,
                        goal_id=gid_c,
                        entity_kind=ek_c,
                        entity_id=ent_c,
                        attempt=att_c,
                        usage_key=uk_c,
                        now=now,
                    ):
                        summary["hq_cancellations_confirmed"] = summary.get("hq_cancellations_confirmed", 0) + 1

        # Reconcile any DISPATCHED/RUNNING tasks whose HQ job crashed/exited without calling notify-exit
        saw_waiting_hq = False
        for task in self.db.list_tasks():
            if task.status not in ACTIVE_TASK_STATES or task.hq_job_id is None:
                continue
            canceled_local, dep_failed = self.hq.is_pending_canceled_or_dep_failed(task.hq_job_id)
            if canceled_local or dep_failed:
                fresh_t = self.db.get_task(task.task_id)
                if fresh_t is None or fresh_t.status not in ACTIVE_TASK_STATES:
                    continue
                try:
                    self.db.transition_task_status(
                        task.task_id,
                        tuple(ACTIVE_TASK_STATES),
                        "CANCELLED" if canceled_local else "FAILED",
                        finished_ts=now,
                        now=now,
                    )
                except StateTransitionError:
                    continue
                if canceled_local and task.hq_job_id is not None:
                    if hasattr(self.db, "confirm_hq_cancellation"):
                        self.db.confirm_hq_cancellation(
                            int(task.hq_job_id),
                            entity_kind="task",
                            entity_id=task.task_id,
                            attempt=int(fresh_t.attempt),
                            now=now,
                        )
                    if hasattr(self.db, "release_goal_budget_reservation"):
                        self.db.release_goal_budget_reservation(
                            task.goal_id,
                            f"task:{task.task_id}:a{int(fresh_t.attempt)}",
                            reason="hq_pending_canceled_local",
                            now=now,
                        )
                summary["recovered_tasks"] += 1
                continue

            info = self.hq.job_info(task.hq_job_id)
            if info is None:
                continue
            state = str(info.get("state") or info.get("status") or "").lower()
            if not state:
                jinfo = info.get("info") or {}
                stats = jinfo.get("task_stats") or {}
                canceled_cnt = int(stats.get("canceled", 0) or 0)
                aborted_cnt = int(stats.get("aborted", 0) or 0)
                failed_cnt = int(stats.get("failed", 0) or 0)
                waiting_cnt = int(stats.get("waiting", 0) or 0)
                running_cnt = int(stats.get("running", 0) or 0)
                finished_cnt = int(stats.get("finished", 0) or 0)
                is_open = bool(jinfo.get("is_open", False))
                finished_at = info.get("finished_at")
                if canceled_cnt > 0 or aborted_cnt > 0:
                    state = "canceled"
                elif failed_cnt > 0:
                    state = "failed"
                elif (
                    not is_open
                    and waiting_cnt == 0
                    and running_cnt == 0
                    and (finished_cnt > 0 or finished_at is not None)
                ):
                    state = "finished"
                elif waiting_cnt > 0:
                    state = "waiting"
            if state == "waiting":
                saw_waiting_hq = True
            if state in ("finished", "completed", "done"):
                # Re-read fresh task status after HQ RPC to avoid TOCTOU when worker already finalized
                fresh_t = self.db.get_task(task.task_id)
                if fresh_t is None or fresh_t.status not in ACTIVE_TASK_STATES:
                    continue
                receipts = self.db.list_receipts(task_id=task.task_id)
                if receipts:
                    final = "DONE" if receipts[-1].process_status == ProcessStatus.EXIT_OK else "FAILED"
                    try:
                        self.db.transition_task_status(
                            task.task_id,
                            tuple(ACTIVE_TASK_STATES),
                            final,
                            finished_ts=now,
                            now=now,
                        )
                        summary["recovered_tasks"] += 1
                    except StateTransitionError:
                        continue
            elif state in ("failed", "canceled", "cancelled", "lost", "error"):
                fresh_t = self.db.get_task(task.task_id)
                if fresh_t is None or fresh_t.status not in ACTIVE_TASK_STATES:
                    continue
                receipts = self.db.list_receipts(task_id=task.task_id)
                if not receipts:
                    goal = self.db.get_goal(task.goal_id)
                    hyp = self.db.get_hypothesis(task.hypothesis_id)
                    if goal and hyp:
                        self._synthesize_missing_receipt(
                            fresh_t,
                            goal,
                            hyp,
                            process_status=ProcessStatus.LEASE_LOST,
                            reason=f"HQ job {task.hq_job_id} terminated in state={state} without receipt",
                        )
                try:
                    self.db.transition_task_status(
                        task.task_id,
                        tuple(ACTIVE_TASK_STATES),
                        "CANCELLED" if "cancel" in state else "FAILED",
                        finished_ts=now,
                        now=now,
                    )
                except StateTransitionError:
                    continue
                if "cancel" in state and task.hq_job_id is not None:
                    if hasattr(self.db, "confirm_hq_cancellation"):
                        self.db.confirm_hq_cancellation(
                            int(task.hq_job_id),
                            entity_kind="task",
                            entity_id=task.task_id,
                            attempt=int(fresh_t.attempt),
                            now=now,
                        )
                    if hasattr(self.db, "release_goal_budget_reservation"):
                        self.db.release_goal_budget_reservation(
                            task.goal_id,
                            f"task:{task.task_id}:a{int(fresh_t.attempt)}",
                            reason=f"hq_terminal_{state}",
                            now=now,
                        )
                summary["recovered_tasks"] += 1

        if saw_waiting_hq:
            live_workers = [w for w in self.hq.worker_list() if w.get("ended") is None]
            if not live_workers:
                self.hq.ensure_running()
                summary["recovered_tasks"] += 1

        self._reconcile_terminal_reserved_ledger_rows(summary, now=now)

    def _handle_completed_task(
        self,
        task: StageTask,
        goal: ResearchGoal,
        summary: dict[str, Any],
    ) -> None:
        if task.task_id in self._handled_tasks:
            return
        hyp = self.db.get_hypothesis(task.hypothesis_id)
        if hyp is None:
            self._handled_tasks.add(task.task_id)
            return

        # Release any active leases (branch + repo_scope) held by this task
        ConflictManager.release_task_leases(self.db, task.task_id)

        receipts = self.db.list_receipts(task_id=task.task_id)
        if not receipts:
            if task.status == "CANCELLED":
                self._handled_tasks.add(task.task_id)
                return
            receipt = self._synthesize_missing_receipt(
                task,
                goal,
                hyp,
                process_status=ProcessStatus.ENV_ERROR,
                reason=f"Task {task.task_id} ended status={task.status} without EvidenceReceipt",
            )
        else:
            receipt = receipts[-1]

        # (a) Epoch / version / plan_epoch isolation via GlobalGoalManager
        epoch_validity = self.goal_manager.validate_receipt_epoch(goal, task, receipt)
        if epoch_validity in (
            EvidenceValidity.STALE_GOAL_VERSION,
            EvidenceValidity.HUMAN_INVALIDATED,
            EvidenceValidity.SUPERSEDED_EPOCH,
        ):
            event_name = (
                "SUPERSEDED_EPOCH"
                if epoch_validity == EvidenceValidity.SUPERSEDED_EPOCH
                else "STALE_OR_INVALIDATED_RECEIPT_ISOLATED"
            )
            if receipt.evidence_validity != epoch_validity:
                updated_rcpt = EvidenceReceipt.from_dict(
                    {**receipt.to_dict(), "evidence_validity": epoch_validity.value}
                )
                with self.db._tx() as (conn, _p):
                    conn.execute(
                        "UPDATE evidence_receipts SET payload_json = ? WHERE receipt_id = ?",
                        (
                            json.dumps(updated_rcpt.to_dict(), sort_keys=True, ensure_ascii=False),
                            updated_rcpt.receipt_id,
                        ),
                    )
                receipt = updated_rcpt
            self._record_ledger_event(
                event_name,
                goal_id=goal.goal_id,
                hypothesis_id=hyp.hypothesis_id,
                task_id=task.task_id,
                payload={
                    "validity": epoch_validity.value,
                    "task_goal_version": int(getattr(task, "goal_version", 1)),
                    "current_goal_version": int(goal.version),
                    "task_human_epoch": int(getattr(task, "human_epoch", 0)),
                    "receipt_human_epoch": int(getattr(receipt, "human_epoch", 0)),
                    "current_human_epoch": int(goal.human_epoch),
                    "task_plan_epoch": int(getattr(task, "plan_epoch", 0) or 0),
                    "receipt_plan_epoch": int(getattr(receipt, "plan_epoch", 0) or 0),
                    "current_plan_epoch": int(getattr(goal, "plan_epoch", 0) or 0),
                    "receipt_id": receipt.receipt_id,
                },
            )
            summary["stale_isolated"] = summary.get("stale_isolated", 0) + 1
            self._mark_task_handled(task)
            return

        # Keep head_sha up to date if CODE committed on the branch
        if receipt.head_sha and receipt.head_sha != hyp.head_sha:
            try:
                hyp = self.db.transition_hypothesis_status(
                    hyp.hypothesis_id,
                    hyp.status,
                    hyp.status,
                    head_sha=receipt.head_sha,
                )
            except Exception:
                hyp.head_sha = receipt.head_sha

        # Detect banned/main touch in receipt
        if "BANNED" in receipt.failure_reason.upper() or "READ_BAN" in receipt.failure_reason.upper():
            self.flag_owner_escalation(
                goal.goal_id,
                OWNER_GATE_MAIN_HOLDOUT,
                receipt.failure_reason,
                hypothesis_id=hyp.hypothesis_id,
                details={"task_id": task.task_id, "stage": task.stage.value},
            )

        # (b) Record structured knowledge & real-LLM eval ledger
        self._record_knowledge_and_eval_ledger(goal, hyp, task, receipt)

        eff_domain = (
            receipt.failure_domain
            if receipt.failure_domain != FailureDomain.NONE
            else gates.classify_failure_domain(receipt, goal=goal)
        )
        failed_exec = (
            receipt.process_status != ProcessStatus.EXIT_OK
            or receipt.engineering_verdict in (EngineeringVerdict.FAIL_GATE, EngineeringVerdict.FAIL_TEST)
        )
        ctx = self._goal_ctx(goal.goal_id)

        # Unified WorkflowNode-driven progression when task belongs to a WorkflowNode
        if task.workflow_id:
            wf = self.db.get_workflow(task.workflow_id)
            if wf is not None and wf.status not in TERMINAL_WORKFLOW_STATUSES:
                # Transient infra retry (`LEASE_LOST`, `ENV_ERROR` transient) before consuming workflow repairs
                if (
                    failed_exec
                    and eff_domain == FailureDomain.INFRA_TRANSIENT
                    and receipt.process_status in (ProcessStatus.LEASE_LOST, ProcessStatus.ENV_ERROR)
                    and task.attempt < task.max_attempts
                ):
                    retry_dec = self.arbiter.record_failure(
                        hyp.hypothesis_id,
                        failure_kind=receipt.process_status,
                        current_pool=task.resources.llm_pool,
                    )
                    if retry_dec.should_retry:
                        next_route = dict(task.pinned_route) if isinstance(task.pinned_route, Mapping) else None
                        role = self.goal_manager._stage_to_role(task.stage, hyp)
                        if role is not None and not self.dry_run_llm:
                            failed_spec = task.model_spec
                            try:
                                resolved = self.router.resolve_route(
                                    role,
                                    goal=goal,
                                    hyp=hyp,
                                    exclude_specs=[failed_spec] if failed_spec else (),
                                )
                                next_route = {
                                    **(next_route or {}),
                                    **resolved.to_dict(),
                                }
                            except PolicyBlockedError as exc:
                                if not exc.waitable:
                                    b_code = exc.code or "BLOCKED_POLICY"
                                    self.db.transition_hypothesis_status(
                                        hyp.hypothesis_id,
                                        hyp.status,
                                        b_code,
                                        failure_domain=FailureDomain.ENV_CONFIG_MISMATCH,
                                    )
                                    self.flag_owner_escalation(
                                        goal.goal_id,
                                        OWNER_GATE_PAID_COMPUTE if "QUOTA" in b_code else OWNER_GATE_GOAL_BAR,
                                        str(exc),
                                        hypothesis_id=hyp.hypothesis_id,
                                    )
                                    self._mark_task_handled(task)
                                    return
                        self.goal_manager._spawn_replacement_task_from_snapshot(
                            goal, task.to_dict(), pinned_route=next_route
                        )
                        summary["tasks_enqueued"] += 1
                        self._mark_task_handled(task)
                        return

                if not failed_exec:
                    self.arbiter.record_success(hyp.hypothesis_id)

                emitted = self.workflow_runtime.advance_workflow(wf, task, receipt, goal)
                summary["tasks_enqueued"] += len(emitted)
                updated_wf = self.db.get_workflow(wf.workflow_id)
                cur_h = self.db.get_hypothesis(hyp.hypothesis_id) or hyp

                if updated_wf is not None and updated_wf.status == "BLOCKED_REVIEW_QUORUM":
                    self.flag_owner_escalation(
                        goal.goal_id,
                        OWNER_GATE_ROUTE_AUDIT,
                        str((updated_wf.state.get("review_quorum") or {}).get("block_reason") or "Review quorum blocked"),
                        hypothesis_id=cur_h.hypothesis_id,
                    )

                if task.stage in READONLY_REVIEW_STAGES and receipt.audit_verdict in (
                    AuditVerdict.FAIL,
                    AuditVerdict.PARSE_ERROR,
                ):
                    ctx.failed_blind_audits += 1
                    self._failed_blind_audits[goal.goal_id] = ctx.failed_blind_audits
                    self._record_ledger_event(
                        "BLIND_AUDIT_FAILED",
                        goal_id=goal.goal_id,
                        hypothesis_id=cur_h.hypothesis_id,
                        task_id=task.task_id,
                        payload={"count": ctx.failed_blind_audits, "stage": task.stage.value},
                    )
                    if ctx.failed_blind_audits > 2:
                        self.flag_owner_escalation(
                            goal.goal_id,
                            OWNER_GATE_ROUTE_AUDIT,
                            f"Review/audit failures exceeded threshold ({ctx.failed_blind_audits} > 2)",
                            hypothesis_id=cur_h.hypothesis_id,
                        )
                    if (
                        updated_wf is not None
                        and updated_wf.status in ("FAILED", "DESIGN_REJECTED", "RESULT_REJECTED")
                        and not emitted
                    ):
                        target_st = (
                            "NEEDS_PLAN_REVISION"
                            if updated_wf.status == "DESIGN_REJECTED"
                            else ("RESULT_REJECTED" if updated_wf.status == "RESULT_REJECTED" else "NEEDS_REPAIR")
                        )
                        preserve_sv = (
                            cur_h.scientific_verdict
                            if cur_h.scientific_verdict == ScientificVerdict.CONFIRMED
                            else ScientificVerdict.UNKNOWN
                        )
                        try:
                            self.db.transition_hypothesis_status(
                                cur_h.hypothesis_id,
                                cur_h.status,
                                target_st,
                                head_sha=cur_h.head_sha,
                                scientific_verdict=preserve_sv,
                                failure_domain=FailureDomain.ENGINEERING_DEFECT,
                            )
                        except Exception:
                            pass
                        summary["reconciled"] += 1
                    self._mark_task_handled(task)
                    return

                if task.stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
                    if eff_domain in (FailureDomain.ENV_CONFIG_MISMATCH, FailureDomain.INFRA_TRANSIENT):
                        target_blk = (
                            "BLOCKED_ENV"
                            if eff_domain == FailureDomain.ENV_CONFIG_MISMATCH
                            else "BLOCKED_INFRA"
                        )
                        try:
                            self.db.transition_hypothesis_status(
                                cur_h.hypothesis_id,
                                cur_h.status,
                                target_blk,
                                head_sha=cur_h.head_sha,
                                scientific_verdict=ScientificVerdict.UNKNOWN,
                                failure_domain=eff_domain,
                            )
                        except Exception:
                            pass
                        summary["reconciled"] += 1
                        self._mark_task_handled(task)
                        return

                    empirical_ok = gates.is_valid_empirical_outcome(receipt, goal) or (
                        receipt.process_status == ProcessStatus.EXIT_OK
                        and receipt.engineering_verdict == EngineeringVerdict.PASS
                        and eff_domain in (FailureDomain.NONE, FailureDomain.SCIENTIFIC_OUTCOME)
                        and bool(receipt.metrics)
                    )
                    if not empirical_ok:
                        try:
                            self.db.transition_hypothesis_status(
                                cur_h.hypothesis_id,
                                cur_h.status,
                                "NEEDS_REPAIR",
                                head_sha=cur_h.head_sha,
                                scientific_verdict=ScientificVerdict.UNKNOWN,
                                failure_domain=FailureDomain.ENGINEERING_DEFECT,
                            )
                        except Exception:
                            pass
                        summary["reconciled"] += 1
                        self._mark_task_handled(task)
                        return

                    if (
                        task.stage == StageKind.EXP_SCREEN
                        and cur_h.kind in (HypothesisKind.ABLATION, HypothesisKind.FALSIFICATION)
                    ):
                        kill_hit = check_kill_criteria(goal, cur_h, receipt)
                        if (
                            receipt.scientific_verdict
                            in (ScientificVerdict.REFUTED, ScientificVerdict.VALID_NEGATIVE)
                            or kill_hit is not None
                        ):
                            dec = self.explore.reconcile(goal, cur_h, receipt)
                            summary["reconciled"] += 1
                            if dec.cancelled_task_ids:
                                self._cancel_hq_for_tasks(dec.cancelled_task_ids)
                                summary["pruned"] += 1
                        else:
                            try:
                                self.db.transition_hypothesis_status(
                                    cur_h.hypothesis_id,
                                    cur_h.status,
                                    "COMPLETED",
                                    head_sha=cur_h.head_sha,
                                    scientific_verdict=receipt.scientific_verdict,
                                    failure_domain=FailureDomain.NONE,
                                )
                            except Exception:
                                pass
                            summary["reconciled"] += 1
                        self._mark_task_handled(task)
                        return

                    dec = self.explore.reconcile(goal, cur_h, receipt)
                    summary["reconciled"] += 1
                    if dec.cancelled_task_ids:
                        self._cancel_hq_for_tasks(dec.cancelled_task_ids)
                        summary["pruned"] += 1
                    for child in dec.spawned_hypotheses:
                        created = self._enqueue_initial_hypothesis_work(goal, child)
                        summary["tasks_enqueued"] += len(created)
                    if (
                        updated_wf is not None
                        and updated_wf.status == "COMPLETED"
                        and updated_wf.parent_workflow_id is None
                    ):
                        refreshed_h = self.db.get_hypothesis(cur_h.hypothesis_id) or cur_h
                        if refreshed_h.status not in ("REFUTED", "VALID_NEGATIVE", "GRAVEYARD"):
                            try:
                                self.db.transition_hypothesis_status(
                                    refreshed_h.hypothesis_id,
                                    refreshed_h.status,
                                    "COMPLETED",
                                    head_sha=refreshed_h.head_sha,
                                    scientific_verdict=(
                                        receipt.scientific_verdict
                                        if receipt.scientific_verdict != ScientificVerdict.NOT_APPLICABLE
                                        else ScientificVerdict.CONFIRMED
                                    ),
                                    failure_domain=FailureDomain.NONE,
                                )
                            except Exception:
                                pass
                    if dec.escalate:
                        self._check_goal_escalations(goal, extra_reason=dec.escalation_reason)
                    self._mark_task_handled(task)
                    return

                if (
                    updated_wf is not None
                    and updated_wf.status == "COMPLETED"
                    and updated_wf.parent_workflow_id is None
                ):
                    refreshed_h = self.db.get_hypothesis(cur_h.hypothesis_id) or cur_h
                    try:
                        self.db.transition_hypothesis_status(
                            refreshed_h.hypothesis_id,
                            refreshed_h.status,
                            "COMPLETED",
                            head_sha=refreshed_h.head_sha,
                            scientific_verdict=ScientificVerdict.CONFIRMED,
                            failure_domain=FailureDomain.NONE,
                        )
                    except Exception:
                        pass
                    summary["reconciled"] += 1
                elif (
                    updated_wf is not None
                    and updated_wf.status == "FAILED"
                    and updated_wf.parent_workflow_id is None
                    and not emitted
                ):
                    refreshed_h = self.db.get_hypothesis(cur_h.hypothesis_id) or cur_h
                    try:
                        self.db.transition_hypothesis_status(
                            refreshed_h.hypothesis_id,
                            refreshed_h.status,
                            "NEEDS_REPAIR",
                            head_sha=refreshed_h.head_sha,
                            scientific_verdict=ScientificVerdict.UNKNOWN,
                            failure_domain=FailureDomain.ENGINEERING_DEFECT,
                        )
                    except Exception:
                        pass
                    summary["reconciled"] += 1

                self._mark_task_handled(task)
                return

        # Standalone non-workflow task handling
        if failed_exec:
            fail_kind: str | ProcessStatus = (
                receipt.engineering_verdict.value
                if receipt.engineering_verdict in (EngineeringVerdict.FAIL_GATE, EngineeringVerdict.FAIL_TEST)
                and receipt.process_status in (ProcessStatus.EXIT_OK, ProcessStatus.EXIT_NONZERO)
                else receipt.process_status
            )
            retry_dec = self.arbiter.record_failure(
                hyp.hypothesis_id,
                failure_kind=fail_kind,
                current_pool=task.resources.llm_pool,
            )
            if (
                retry_dec.should_retry
                and task.attempt < task.max_attempts
                and eff_domain != FailureDomain.SCIENTIFIC_OUTCOME
            ):
                next_res = ResourceSpec.from_dict(task.resources.to_dict())
                if retry_dec.next_pool:
                    next_res.llm_pool = retry_dec.next_pool
                next_stage = task.stage
                if task.stage == StageKind.CPU_GATE and retry_dec.reason.startswith("GATE_REPAIR"):
                    prev_code = [
                        t
                        for t in self.db.list_tasks(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id, stage=StageKind.CODE)
                        if t.resources.llm_pool
                    ]
                    if prev_code:
                        next_stage = StageKind.CODE
                        next_res = ResourceSpec.from_dict(prev_code[-1].resources.to_dict())
                self._enqueue_stage(
                    goal,
                    hyp,
                    next_stage,
                    attempt=task.attempt + 1,
                    worktree_path=task.worktree_path,
                    resources_override=next_res,
                    workflow_id=task.workflow_id,
                    workflow_primitive=task.workflow_primitive,
                    step_key=task.step_key,
                )
                summary["tasks_enqueued"] += 1
                self._mark_task_handled(task)
                return

            dec = self.explore.reconcile(goal, hyp, receipt)
            summary["reconciled"] += 1
            if dec.cancelled_task_ids:
                self._cancel_hq_for_tasks(dec.cancelled_task_ids)
            if dec.escalate:
                self._check_goal_escalations(goal, extra_reason=dec.escalation_reason)
            self._mark_task_handled(task)
            return

        # Standalone execution succeeded
        self.arbiter.record_success(hyp.hypothesis_id)

        if task.stage == StageKind.CODE:
            self._enqueue_stage(
                goal,
                hyp,
                StageKind.CPU_GATE,
                deps=[task.task_id],
                worktree_path=task.worktree_path,
            )
            summary["tasks_enqueued"] += 1

        elif task.stage == StageKind.CPU_GATE:
            self._enqueue_stage(
                goal,
                hyp,
                StageKind.EXP_SCREEN,
                deps=[task.task_id],
                worktree_path=task.worktree_path,
            )
            summary["tasks_enqueued"] += 1

        elif task.stage == StageKind.EXP_SCREEN:
            if hyp.kind in (HypothesisKind.ABLATION, HypothesisKind.FALSIFICATION):
                kill_hit = check_kill_criteria(goal, hyp, receipt)
                if receipt.scientific_verdict == ScientificVerdict.REFUTED or kill_hit is not None:
                    dec = self.explore.reconcile(goal, hyp, receipt)
                    summary["reconciled"] += 1
                    if dec.cancelled_task_ids:
                        self._cancel_hq_for_tasks(dec.cancelled_task_ids)
                    summary["pruned"] += 1
                else:
                    try:
                        self.db.transition_hypothesis_status(
                            hyp.hypothesis_id,
                            hyp.status,
                            "COMPLETED",
                            head_sha=hyp.head_sha,
                            scientific_verdict=receipt.scientific_verdict,
                        )
                    except Exception:
                        pass
                    self._enqueue_stage(
                        goal,
                        hyp,
                        StageKind.RECONCILE,
                        deps=[task.task_id],
                        worktree_path=task.worktree_path,
                    )
                    summary["tasks_enqueued"] += 1
            else:
                dec = self.explore.reconcile(goal, hyp, receipt)
                summary["reconciled"] += 1
                if dec.action == "PROMOTE_F2":
                    self._enqueue_stage(
                        goal,
                        hyp,
                        StageKind.EXP_CONFIRM,
                        deps=[task.task_id],
                        worktree_path=task.worktree_path,
                    )
                    summary["tasks_enqueued"] += 1
                    for child in dec.spawned_hypotheses:
                        self._enqueue_stage(goal, child, StageKind.CODE)
                        summary["tasks_enqueued"] += 1
                elif dec.action == "PROMOTE_F1":
                    self._enqueue_stage(
                        goal,
                        hyp,
                        StageKind.EXP_SCREEN,
                        deps=[task.task_id],
                        attempt=task.attempt + 1,
                        worktree_path=task.worktree_path,
                    )
                    summary["tasks_enqueued"] += 1
                elif dec.action == "REFUTE_AND_PRUNE":
                    self._cancel_hq_for_tasks(dec.cancelled_task_ids)
                    summary["pruned"] += 1
                elif dec.action == "DIAGNOSTIC_REFINEMENT":
                    for child in dec.spawned_hypotheses:
                        self._enqueue_stage(goal, child, StageKind.CODE)
                        summary["tasks_enqueued"] += 1
                elif dec.escalate:
                    self._check_goal_escalations(goal, extra_reason=dec.escalation_reason)

        elif task.stage == StageKind.EXP_CONFIRM:
            dec = self.explore.reconcile(goal, hyp, receipt)
            summary["reconciled"] += 1
            if dec.action == "PROMOTE_F3":
                self._enqueue_stage(
                    goal,
                    hyp,
                    StageKind.BLIND_AUDIT,
                    deps=[task.task_id],
                    worktree_path=task.worktree_path,
                )
                summary["tasks_enqueued"] += 1
            elif dec.action == "REFUTE_AND_PRUNE":
                self._cancel_hq_for_tasks(dec.cancelled_task_ids)
                summary["pruned"] += 1
            elif dec.action == "DIAGNOSTIC_REFINEMENT":
                for child in dec.spawned_hypotheses:
                    self._enqueue_stage(goal, child, StageKind.CODE)
                    summary["tasks_enqueued"] += 1

        elif task.stage in READONLY_REVIEW_STAGES:
            if receipt.audit_verdict in (AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR):
                ctx = self._goal_ctx(goal.goal_id)
                ctx.failed_blind_audits += 1
                self._failed_blind_audits[goal.goal_id] = ctx.failed_blind_audits
                self._record_ledger_event(
                    "BLIND_AUDIT_FAILED",
                    goal_id=goal.goal_id,
                    hypothesis_id=hyp.hypothesis_id,
                    task_id=task.task_id,
                    payload={"count": ctx.failed_blind_audits, "stage": task.stage.value},
                )
                repairs_used = int(hyp.budget_state.get("code_repairs_used", 0))
                max_repairs = int(hyp.budget_state.get("max_code_repairs", 2))
                preserve_sv = (
                    hyp.scientific_verdict
                    if hyp.scientific_verdict == ScientificVerdict.CONFIRMED
                    else ScientificVerdict.UNKNOWN
                )
                if repairs_used < max_repairs:
                    self.db.transition_hypothesis_status(
                        hyp.hypothesis_id,
                        hyp.status,
                        "NEEDS_REPAIR",
                        head_sha=hyp.head_sha,
                        scientific_verdict=preserve_sv,
                        failure_domain=FailureDomain.ENGINEERING_DEFECT,
                        budget_state_patch={"code_repairs_used": repairs_used + 1},
                    )
                    self.db.update_hypothesis_params(
                        hyp.hypothesis_id,
                        {
                            "requires_llm_code": True,
                            "last_repair_notes": receipt.failure_reason or "Audit defect repair",
                        },
                    )
                    refreshed_h = self.db.get_hypothesis(hyp.hypothesis_id) or hyp
                    self._enqueue_stage(
                        goal,
                        refreshed_h,
                        StageKind.CODE,
                        deps=[task.task_id],
                        attempt=repairs_used + 2,
                        worktree_path=task.worktree_path,
                    )
                    summary["tasks_enqueued"] += 1
                else:
                    self.db.transition_hypothesis_status(
                        hyp.hypothesis_id,
                        hyp.status,
                        "NEEDS_REPAIR",
                        head_sha=hyp.head_sha,
                        scientific_verdict=preserve_sv,
                        failure_domain=FailureDomain.ENGINEERING_DEFECT,
                    )
                summary["reconciled"] += 1
                if ctx.failed_blind_audits > 2:
                    self.flag_owner_escalation(
                        goal.goal_id,
                        OWNER_GATE_ROUTE_AUDIT,
                        f"Blind audit failures exceeded threshold ({ctx.failed_blind_audits} > 2)",
                        hypothesis_id=hyp.hypothesis_id,
                    )
            else:
                self._enqueue_stage(
                    goal,
                    hyp,
                    StageKind.RECONCILE,
                    deps=[task.task_id],
                    worktree_path=task.worktree_path,
                )
                summary["tasks_enqueued"] += 1

        elif task.stage == StageKind.RECONCILE:
            if hyp.fidelity == FidelityRung.F3_MILESTONE_AUDIT or hyp.status in ("PROMOTED_F3", "COMPLETED"):
                try:
                    self.db.transition_hypothesis_status(
                        hyp.hypothesis_id,
                        hyp.status,
                        "COMPLETED",
                        head_sha=hyp.head_sha,
                        scientific_verdict=ScientificVerdict.CONFIRMED,
                    )
                except Exception:
                    pass
                summary["reconciled"] += 1
            else:
                dec = self.explore.reconcile(goal, hyp, receipt)
                summary["reconciled"] += 1

        self._mark_task_handled(task)

    def _mark_task_handled(self, task: StageTask) -> None:
        self._handled_tasks.add(task.task_id)
        self._record_ledger_event(
            "CONTROLLER_TASK_HANDLED",
            goal_id=task.goal_id,
            hypothesis_id=task.hypothesis_id,
            task_id=task.task_id,
            payload={"stage": task.stage.value, "status": task.status},
        )

    def _cancel_hq_for_tasks(self, task_ids: Sequence[str]) -> None:
        if self.execution_mode != "hq" or not task_ids:
            return
        job_ids = []
        for tid in task_ids:
            t = self.db.get_task(tid)
            if t is not None and t.hq_job_id is not None:
                job_ids.append(t.hq_job_id)
        if job_ids:
            self.hq.cancel_jobs(job_ids, reason="pruned_by_reconcile")

    def _maybe_propose_round(self, goal: ResearchGoal, summary: dict[str, Any]) -> None:
        ctx = self._goal_ctx(goal.goal_id)
        ctx.last_step_ts = time.time()

        # Respect autonomy_mode ONLY for this Goal (never stalls sibling Goals!)
        if goal.autonomy_mode == "paused":
            ctx.decision_blocked = True
            return

        hyps = self.db.list_hypotheses(goal_id=goal.goal_id)
        ready_tasks = self.db.list_ready_tasks(goal_id=goal.goal_id)
        receipts_count = len(self.db.list_receipts(goal_id=goal.goal_id))

        # No batch barrier: trigger when initial round OR ready queue < max_concurrent_branches & new evidence arrived
        is_initial = len(hyps) == 0
        has_new_evidence = (
            receipts_count > 0
            and receipts_count > ctx.last_propose_receipt_count
            and len(ready_tasks) < max(1, goal.max_concurrent_branches)
        )
        if not is_initial and not has_new_evidence:
            return

        if goal.autonomy_mode == "supervised":
            ctx.decision_blocked = True
            return

        if self.db.is_budget_exhausted(goal.goal_id):
            self.flag_owner_escalation(
                goal.goal_id,
                OWNER_GATE_PAID_COMPUTE,
                "Token budget exhausted; owner approval required to raise budget",
            )
            return

        # Non-blocking per-Goal ResearchLead/ResearchDirector DecisionRequest when running real AI/auto mode on HQ
        if (
            not self.dry_run_llm
            and self.propose_mode in ("auto", "ai")
            and self.execution_mode == "hq"
        ):
            pending_or_running = [
                d
                for d in self.db.list_decision_requests(goal_id=goal.goal_id)
                if d.get("status") in ("PENDING", "SUBMITTING", "RUNNING")
            ]
            if is_initial:
                if not pending_or_running:
                    self.decision_engine.enqueue_decision(
                        goal.goal_id,
                        role=ScientificRole.RESEARCH_LEAD,
                        tier="L1",
                        trigger_kind="INITIAL_PLANNING",
                        urgency="IMMEDIATE",
                        coalesce_window_s=0.0,
                        payload={"initial_planning": True},
                    )
                    summary["decisions_enqueued"] = summary.get("decisions_enqueued", 0) + 1
                return
            if (
                not pending_or_running
                and receipts_count > 0
                and receipts_count > ctx.last_decision_receipt_count
            ):
                role = ScientificRole.RESEARCH_LEAD
                tier = "L1"
                self.decision_engine.enqueue_decision(
                    goal.goal_id,
                    role=role,
                    tier=tier,
                    trigger_kind="EVIDENCE_UPDATE",
                    urgency="COALESCED",
                    coalesce_window_s=self.decision_coalesce_window_s,
                    payload={"receipts_count": receipts_count},
                )
                ctx.last_decision_receipt_count = receipts_count
                ctx.last_propose_receipt_count = receipts_count
                self._last_propose_receipt_count[goal.goal_id] = receipts_count
                summary["decisions_enqueued"] = summary.get("decisions_enqueued", 0) + 1
            return

        if is_initial:
            eff_mode = "seeds" if (self.dry_run_llm and self.propose_mode == "auto") else self.propose_mode
            batch = self.explore.propose(
                goal,
                mode=eff_mode,
                ready_queue_count=len(ready_tasks),
                dispatchable_slots=goal.max_concurrent_branches,
                persist=True,
            )
        else:
            eff_mode = "seeds" if self.dry_run_llm else ("ai" if self.propose_mode == "auto" else self.propose_mode)
            batch = self.explore.propose(
                goal,
                mode=eff_mode,
                ready_queue_count=len(ready_tasks),
                dispatchable_slots=goal.max_concurrent_branches,
                persist=True,
            )
            is_guard_fuse = bool(
                batch.stopped_reason
                and batch.stopped_reason.startswith(
                    (
                        "READY_QUEUE_WATERMARK_REACHED",
                        "TOKEN_BUDGET_FUSE",
                        "DIMINISHING_RETURNS_FUSE",
                    )
                )
            )
            if not batch.accepted and not is_guard_fuse:
                confirmed_parents = [
                    h
                    for h in hyps
                    if h.kind in (HypothesisKind.COMPETING_ROUTE, HypothesisKind.PARAMETER_RACE)
                    and h.scientific_verdict == ScientificVerdict.CONFIRMED
                    and h.status in ("COMPLETED", "PROMOTED_F2", "PROMOTED_F3")
                ]
                if not confirmed_parents:
                    failed_audit_hyp_ids = {
                        r.hypothesis_id
                        for r in self.db.list_receipts(goal_id=goal.goal_id, stage=StageKind.BLIND_AUDIT)
                        if r.audit_verdict in (AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR)
                    }
                    confirmed_parents = [
                        h
                        for h in hyps
                        if h.fidelity == FidelityRung.F3_MILESTONE_AUDIT
                        or h.hypothesis_id in failed_audit_hyp_ids
                    ]
                if not confirmed_parents:
                    confirmed_parents = [
                        h
                        for h in hyps
                        if h.scientific_verdict == ScientificVerdict.CONFIRMED
                    ]
                round2_exists = any(
                    h.kind == HypothesisKind.REFINEMENT and h.params.get("round") == 2
                    for h in hyps
                )
                if confirmed_parents and not round2_exists:
                    parent = sorted(confirmed_parents, key=lambda h: -h.priority_evoi)[0]
                    is_audit_refine = (
                        parent.scientific_verdict != ScientificVerdict.CONFIRMED
                        or parent.status in ("REFUTED", "GRAVEYARD")
                    )
                    r2_statement = (
                        f"Address blind-audit findings on route '{parent.mechanism}' to further improve {goal.primary_metric}."
                        if is_audit_refine
                        else f"Refine validated route '{parent.mechanism}' to further improve {goal.primary_metric}."
                    )
                    r2_candidate = {
                        "parent_id": parent.hypothesis_id,
                        "kind": HypothesisKind.REFINEMENT.value,
                        "title": f"Round 2 refinement of {parent.title}",
                        "statement": r2_statement,
                        "mechanism": f"round2_refinement::{parent.mechanism}",
                        "fidelity": FidelityRung.F1_FAST_SCREEN.value,
                        "writable_paths": list(parent.writable_paths),
                        "params": {
                            **dict(parent.params),
                            "round": 2,
                            "refined_from": parent.hypothesis_id,
                        },
                        "kill_criteria": list(parent.kill_criteria),
                        "acceptance_cmd": parent.acceptance_cmd,
                        "screen_cmd": parent.screen_cmd,
                        "confirm_cmd": parent.confirm_cmd,
                        "base_sha": parent.head_sha or parent.base_sha,
                    }
                    batch = self.explore.propose(
                        goal,
                        mode="import",
                        external_proposals=[r2_candidate],
                        ready_queue_count=len(ready_tasks),
                        dispatchable_slots=goal.max_concurrent_branches,
                        persist=True,
                    )
        ctx.last_propose_receipt_count = receipts_count
        self._last_propose_receipt_count[goal.goal_id] = receipts_count
        self._record_ledger_event(
            "CONTROLLER_PROPOSE_ROUND",
            goal_id=goal.goal_id,
            payload={
                "mode": eff_mode,
                "accepted": len(batch.accepted),
                "rejected": len(batch.rejected),
                "stopped_reason": batch.stopped_reason,
                "receipts_count": receipts_count,
            },
        )
        for rej in batch.rejected:
            reasons = " ".join(str(x) for x in rej.get("reasons", []))
            if "banned" in reasons.lower() or "main" in reasons.lower():
                self.flag_owner_escalation(
                    goal.goal_id,
                    OWNER_GATE_MAIN_HOLDOUT,
                    f"Proposal rejected for banned/main touch: {reasons}",
                )

        if batch.accepted:
            summary["proposed"] += len(batch.accepted)
            for hyp in batch.accepted:
                created = self._enqueue_initial_hypothesis_work(goal, hyp)
                summary["tasks_enqueued"] += len(created)

    def _reconcile_terminal_reserved_ledger_rows(
        self,
        summary: dict[str, Any],
        *,
        now: float | None = None,
    ) -> None:
        """Reconcile `'RESERVED'` `token_usage_ledger` rows owned by terminal tasks/decisions without active `CANCEL_PENDING`."""
        from conjectra.decision_engine import get_active_submitting_nonces

        ts = time.time() if now is None else float(now)
        with self.db._lock:
            res_rows = [
                dict(r)
                for r in self.db._conn.execute(
                    "SELECT usage_key, goal_id, entity_kind, entity_id, attempt FROM token_usage_ledger WHERE state = 'RESERVED'"
                ).fetchall()
            ]
        if not res_rows:
            return

        active_nonces = get_active_submitting_nonces() | set(
            getattr(self.decision_engine, "_active_submitting_nonces", set()) or set()
        )
        active_leases = self.db.list_active_leases(now=ts)
        active_leases_by_id = {str(ls.get("lease_id") or ""): ls for ls in active_leases}
        active_lease_tids = {str(ls.get("task_id") or "") for ls in active_leases}
        cached_job_list: dict[str, Any] | None = None

        for r_item in res_rows:
            ukey = str(r_item.get("usage_key") or "")
            gid = str(r_item.get("goal_id") or "")
            ek = str(r_item.get("entity_kind") or "")
            ent_id = str(r_item.get("entity_id") or "")
            att = max(1, int(r_item.get("attempt", 1) or 1))

            if ek in ("stage_task", "task"):
                t_obj = self.db.get_task(ent_id)
                if (
                    t_obj is not None
                    and t_obj.status not in TERMINAL_TASK_STATES
                    and int(t_obj.attempt) <= att
                ):
                    continue
                with self.db._lock:
                    has_unconf_t = self.db._has_unconfirmed_hq_cancel_in_conn(
                        self.db._conn, "task", ent_id, att
                    )
                if has_unconf_t:
                    continue
                if (
                    self.execution_mode == "hq"
                    and t_obj is not None
                    and t_obj.hq_job_id is not None
                    and int(t_obj.attempt) == att
                ):
                    t_jid = int(t_obj.hq_job_id)
                    q_t = self.goal_manager._query_hq_job_state(self.hq, t_jid)
                    st_t = str(q_t.get("lookup_status") or "QUERY_ERROR").upper()
                    if st_t == "QUERY_ERROR":
                        continue
                    if st_t == "FOUND":
                        if (
                            self.goal_manager._hq_positively_terminated(
                                self.hq, t_jid, prefetched_info=q_t.get("job")
                            )
                            and ent_id not in active_lease_tids
                        ):
                            self.db.release_goal_budget_reservation(
                                gid, ukey, reason="terminal_task_hq_terminated", now=ts
                            )
                        else:
                            self.db.record_pending_hq_cancellation(
                                hq_job_id=t_jid,
                                entity_kind="task",
                                entity_id=ent_id,
                                attempt=att,
                                goal_id=gid,
                                hypothesis_id=t_obj.hypothesis_id,
                                workflow_id=t_obj.workflow_id or "",
                                reason="terminal_task_still_live_on_hq",
                                state="CANCEL_PENDING",
                                now=ts,
                            )
                            with contextlib.suppress(Exception):
                                self.hq.cancel_jobs([t_jid], reason="terminal_task_still_live_on_hq")
                        continue
                    if st_t == "NOT_FOUND" and ent_id not in active_lease_tids:
                        self.db.release_goal_budget_reservation(
                            gid, ukey, reason="terminal_task_hq_not_found", now=ts
                        )
                    continue
                if ent_id not in active_lease_tids:
                    self.db.release_goal_budget_reservation(
                        gid, ukey, reason="terminal_task_no_hq_job", now=ts
                    )
                continue

            if ek in ("decision_request", "decision"):
                dreq = self.db.get_decision_request(ent_id)
                pl = dict((dreq or {}).get("payload") or {})
                dreq_att = max(1, int((dreq or {}).get("attempt", 1) or 1))
                dreq_nonce = str(pl.get("submit_nonce") or "").strip()
                s_nonce = ukey.split(":", 3)[3] if ukey.count(":") >= 3 else ""
                same_gen = (
                    dreq is not None
                    and dreq_att == att
                    and (not s_nonce or dreq_nonce == s_nonce)
                )
                if same_gen and str(dreq.get("status") or "") in (
                    "PENDING",
                    "SUBMITTING",
                    "RUNNING",
                ):
                    continue
                if s_nonce and s_nonce in active_nonces:
                    continue
                with self.db._lock:
                    has_unconf_d = self.db._has_unconfirmed_hq_cancel_in_conn(
                        self.db._conn, "decision", ent_id, att, usage_key=ukey
                    )
                    row_phq = (
                        self.db._conn.execute(
                            "SELECT hq_job_id FROM pending_hq_cancellations WHERE usage_key = ? LIMIT 1",
                            (ukey,),
                        ).fetchone()
                        if (not has_unconf_d and not same_gen and s_nonce)
                        else None
                    )
                if has_unconf_d:
                    continue

                d_jid: int | None = None
                if same_gen:
                    if dreq is not None and dreq.get("hq_job_id") is not None:
                        d_jid = int(dreq["hq_job_id"])
                elif row_phq is not None and row_phq["hq_job_id"] is not None:
                    d_jid = int(row_phq["hq_job_id"])
                elif self.execution_mode == "hq" and (
                    hasattr(self.hq, "query_job_list") or hasattr(self.hq, "job_list")
                ):
                    if cached_job_list is None:
                        cached_job_list = self.goal_manager._query_hq_job_list(self.hq, all_jobs=True)
                    if str(cached_job_list.get("lookup_status") or "QUERY_ERROR").upper() != "OK":
                        continue
                    exp_name = (
                        f"dec-{ent_id}-a{att}-{s_nonce[:8]}"
                        if s_nonce
                        else f"dec-{ent_id}"
                    )
                    for j_it in cached_job_list.get("jobs") or ():
                        if (
                            isinstance(j_it, Mapping)
                            and str(j_it.get("name") or "") == exp_name
                            and j_it.get("id") is not None
                        ):
                            d_jid = int(j_it["id"])
                            break
                    if d_jid is None:
                        if f"dec-{ent_id}" not in active_lease_tids:
                            self.db.release_goal_budget_reservation(
                                gid, ukey, reason="stale_generation_decision_absent_on_hq", now=ts
                            )
                        continue

                if d_jid is not None:
                    if self.execution_mode != "hq":
                        if f"dec-{ent_id}" not in active_lease_tids:
                            self.db.release_goal_budget_reservation(
                                gid, ukey, reason="terminal_decision_inline", now=ts
                            )
                        continue
                    jid_i = int(d_jid)
                    q_d = self.goal_manager._query_hq_job_state(self.hq, jid_i)
                    st_d = str(q_d.get("lookup_status") or "QUERY_ERROR").upper()
                    if st_d == "QUERY_ERROR":
                        continue
                    if st_d == "FOUND":
                        if (
                            self.goal_manager._hq_positively_terminated(
                                self.hq, jid_i, prefetched_info=q_d.get("job")
                            )
                            and f"dec-{ent_id}" not in active_lease_tids
                        ):
                            self.db.release_goal_budget_reservation(
                                gid, ukey, reason="terminal_decision_hq_terminated", now=ts
                            )
                        else:
                            self.db.record_pending_hq_cancellation(
                                hq_job_id=jid_i,
                                entity_kind="decision",
                                entity_id=ent_id,
                                attempt=att,
                                goal_id=gid,
                                reason="terminal_decision_still_live_on_hq",
                                state="CANCEL_PENDING",
                                usage_key=ukey,
                                now=ts,
                            )
                            with contextlib.suppress(Exception):
                                self.hq.cancel_jobs([jid_i], reason="terminal_decision_still_live_on_hq")
                        continue
                    if st_d == "NOT_FOUND" and f"dec-{ent_id}" not in active_lease_tids:
                        self.db.release_goal_budget_reservation(
                            gid, ukey, reason="terminal_decision_hq_not_found", now=ts
                        )
                    continue

                # Unattached terminal decision (`hq_job_id is None`) with leftover `'RESERVED'` row:
                lid = str(pl.get("lease_id") or "")
                owner_pid = int(pl.get("submit_owner_pid", 0) or 0)
                owner_start = pl.get("submit_proc_start_time")
                if lid and lid in active_leases_by_id and owner_pid > 0 and owner_pid != os.getpid():
                    if verify_process_identity(owner_pid, owner_start).alive:
                        continue
                if self.execution_mode != "hq":
                    if lid:
                        self.db.release_lease(lid, now=ts)
                    self.db.release_goal_budget_reservation(
                        gid, ukey, reason="cancelled_unattached_decision_inline", now=ts
                    )
                    continue
                if cached_job_list is None:
                    cached_job_list = self.goal_manager._query_hq_job_list(self.hq, all_jobs=True)
                if str(cached_job_list.get("lookup_status") or "QUERY_ERROR").upper() != "OK":
                    continue
                s_name = str(pl.get("submit_job_name") or f"dec-{ent_id}")
                matched_j: dict[str, Any] | None = None
                for j_entry in cached_job_list.get("jobs") or []:
                    if not isinstance(j_entry, Mapping):
                        continue
                    jn = str(j_entry.get("name") or "")
                    if jn == s_name or (s_name == f"dec-{ent_id}" and jn == f"dec-{ent_id}"):
                        matched_j = dict(j_entry)
                        break
                if matched_j is not None and matched_j.get("id") is not None:
                    m_jid = int(matched_j["id"])
                    with self.db._tx() as (conn, _p):
                        conn.execute(
                            "UPDATE decision_requests SET hq_job_id = COALESCE(hq_job_id, ?), updated_ts = ? WHERE req_id = ?",
                            (m_jid, ts, ent_id),
                        )
                        if lid:
                            self.db.release_lease(lid, now=ts)
                        self.db.record_pending_hq_cancellation(
                            hq_job_id=m_jid,
                            entity_kind="decision",
                            entity_id=ent_id,
                            attempt=att,
                            goal_id=gid,
                            reason="reconcile_cancelled_unattached_decision",
                            state="CANCEL_PENDING",
                            usage_key=ukey,
                            now=ts,
                        )
                    with contextlib.suppress(Exception):
                        self.hq.cancel_jobs([m_jid], reason="reconcile_cancelled_unattached_decision")
                    if self.goal_manager._hq_positively_terminated(self.hq, m_jid):
                        self.db.confirm_hq_cancellation(
                            m_jid, entity_kind="decision", entity_id=ent_id, attempt=att, usage_key=ukey, now=ts
                        )
                else:
                    if lid:
                        self.db.release_lease(lid, now=ts)
                    self.db.release_goal_budget_reservation(
                        gid, ukey, reason="cancelled_unattached_decision_absent_on_hq", now=ts
                    )

    def _reconcile_submitting_decisions(self, summary: dict[str, Any]) -> None:
        """Durable `SUBMITTING` crash-recovery reducer.

        Reconciles interrupted `SUBMITTING` DecisionRequests by `(req_id, attempt, submit_nonce)` and
        authoritative HQ `query_job_list`:
          - `QUERY_ERROR`: fails closed (retains `SUBMITTING` and `RESERVED` budget).
          - Matching HQ job `submit_job_name` FOUND: atomically attaches `hq_job_id` and transitions to
            `RUNNING` (renewing/preserving lease) if Goal/epochs remain valid, else transitions to
            `CANCELLED` + records `CANCEL_PENDING` + cancels HQ job while keeping `RESERVED` until confirmed.
          - Authoritatively ABSENT (`lookup_status == "OK"`): retires old lease, releases `RESERVED`
            admission row, and requeues `SUBMITTING -> PENDING` (or `CANCELLED` if Goal inactive/stale).
        """
        from conjectra.decision_engine import get_active_submitting_nonces

        now = time.time()
        self._reconcile_terminal_reserved_ledger_rows(summary, now=now)
        submitting_rows = self.db.list_decision_requests(status="SUBMITTING")
        if not submitting_rows:
            return
        active_nonces = get_active_submitting_nonces() | set(
            getattr(self.decision_engine, "_active_submitting_nonces", set()) or set()
        )
        active_leases_by_id = {
            str(ls.get("lease_id") or ""): ls for ls in self.db.list_active_leases(now=now)
        }
        candidates: list[dict[str, Any]] = []
        for req in submitting_rows:
            pl = dict(req.get("payload") or {})
            s_nonce = str(pl.get("submit_nonce") or "")
            if s_nonce and s_nonce in active_nonces:
                continue
            lid = str(pl.get("lease_id") or "")
            owner_pid = int(pl.get("submit_owner_pid", 0) or 0)
            owner_start = pl.get("submit_proc_start_time")
            lease_alive = bool(lid and lid in active_leases_by_id)
            if lease_alive and owner_pid > 0 and owner_pid != os.getpid():
                if verify_process_identity(owner_pid, owner_start).alive:
                    continue
            candidates.append(req)

        if not candidates:
            return

        hq_jobs: list[dict[str, Any]] = []
        if self.execution_mode == "hq":
            q_list = self.goal_manager._query_hq_job_list(self.hq, all_jobs=True)
            if str(q_list.get("lookup_status") or "QUERY_ERROR").upper() != "OK":
                return
            hq_jobs = list(q_list.get("jobs") or [])

        for req in candidates:
            req_id = str(req["req_id"])
            gid = str(req["goal_id"])
            pl = dict(req.get("payload") or {})
            s_nonce = str(pl.get("submit_nonce") or "")
            req_att = max(1, int(pl.get("attempt", req.get("attempt", 1)) or 1))
            lid = str(pl.get("lease_id") or "")
            s_name = str(pl.get("submit_job_name") or f"dec-{req_id}")
            dec_ukey = str(
                pl.get("usage_key")
                or (f"decision:{req_id}:a{req_att}:{s_nonce}" if s_nonce else f"decision:{req_id}:a{req_att}")
            )
            req_goal_ver = int(req["goal_version"])
            req_human_ep = int(req["human_epoch"])
            req_plan_ep = pl.get("plan_epoch")

            matched_job: dict[str, Any] | None = None
            for j in hq_jobs:
                if not isinstance(j, Mapping):
                    continue
                jn = str(j.get("name") or "")
                if jn == s_name or (s_name == f"dec-{req_id}" and jn == f"dec-{req_id}"):
                    matched_job = dict(j)
                    break

            if matched_job is not None and matched_job.get("id") is not None:
                m_jid = int(matched_job["id"])
                must_cancel_hq = False
                hq_pos_done = self.goal_manager._hq_positively_terminated(
                    self.hq, m_jid, prefetched_info=matched_job
                )
                with self.db._tx() as (conn, _p):
                    now_rec = time.time()
                    d_row = conn.execute(
                        "SELECT status, attempt, payload_json FROM decision_requests WHERE req_id = ?",
                        (req_id,),
                    ).fetchone()
                    if d_row is None:
                        continue
                    cur_st = str(d_row["status"])
                    cur_att = max(1, int(d_row["attempt"] or 1))
                    cur_pl = json.loads(d_row["payload_json"])
                    same_nonce_att = (
                        cur_att == req_att
                        and str(cur_pl.get("submit_nonce") or "") == s_nonce
                    )
                    if cur_st == "RESOLVED":
                        if same_nonce_att:
                            conn.execute(
                                "UPDATE decision_requests SET hq_job_id = COALESCE(hq_job_id, ?), updated_ts = ? WHERE req_id = ? AND status = 'RESOLVED'",
                                (m_jid, now_rec, req_id),
                            )
                        continue
                    if cur_st != "SUBMITTING" or not same_nonce_att:
                        continue
                    g_row = conn.execute(
                        "SELECT status, payload_json FROM goals WHERE goal_id = ?",
                        (gid,),
                    ).fetchone()
                    g_ok = False
                    if g_row is not None:
                        fg = ResearchGoal.from_dict(json.loads(g_row["payload_json"]))
                        g_ok = (
                            str(g_row["status"]).upper() == "ACTIVE"
                            and fg.autonomy_mode != "paused"
                            and int(fg.version) == req_goal_ver
                            and int(fg.human_epoch) == req_human_ep
                            and (
                                req_plan_ep is None
                                or int(getattr(fg, "plan_epoch", 0) or 0) == int(req_plan_ep)
                            )
                        )
                    has_unconf = self.db._has_unconfirmed_hq_cancel_in_conn(
                        conn, "decision", req_id, req_att, usage_key=dec_ukey
                    )
                    if g_ok and not has_unconf and not hq_pos_done:
                        new_exp = now_rec + float(225 + 30)
                        if lid:
                            conn.execute(
                                """
                                INSERT INTO leases (
                                    lease_id, resource_key, task_id, goal_id, holder, capacity,
                                    acquired_ts, renewed_ts, expires_ts, released_ts, status
                                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, NULL, 'ACTIVE')
                                ON CONFLICT(lease_id) DO UPDATE SET
                                    renewed_ts = excluded.renewed_ts,
                                    expires_ts = MAX(leases.expires_ts, excluded.expires_ts),
                                    released_ts = NULL,
                                    status = 'ACTIVE'
                                """,
                                (
                                    lid,
                                    f"decision:{gid}:{req_id}",
                                    f"dec-{req_id}",
                                    gid,
                                    f"hq-dec-{req_id}-{s_nonce}",
                                    now_rec,
                                    now_rec,
                                    new_exp,
                                ),
                            )
                        conn.execute(
                            """
                            UPDATE decision_requests
                            SET status = 'RUNNING', hq_job_id = ?, updated_ts = ?
                            WHERE req_id = ? AND status = 'SUBMITTING'
                            """,
                            (m_jid, now_rec, req_id),
                        )
                        summary["recovered_submitting_decisions"] = (
                            summary.get("recovered_submitting_decisions", 0) + 1
                        )
                    elif hq_pos_done:
                        conn.execute(
                            """
                            UPDATE decision_requests
                            SET status = 'FAILED', hq_job_id = ?, resolved_ts = ?, updated_ts = ?
                            WHERE req_id = ? AND status = 'SUBMITTING'
                            """,
                            (m_jid, now_rec, now_rec, req_id),
                        )
                        if lid:
                            self.db.release_lease(lid, now=now_rec)
                        self.db.release_goal_budget_reservation(
                            gid, dec_ukey, reason="submitting_hq_already_terminated", now=now_rec
                        )
                        summary["recovered_submitting_decisions"] = (
                            summary.get("recovered_submitting_decisions", 0) + 1
                        )
                    else:
                        must_cancel_hq = True
                        conn.execute(
                            """
                            UPDATE decision_requests
                            SET status = 'CANCELLED', hq_job_id = ?, resolved_ts = ?, updated_ts = ?
                            WHERE req_id = ? AND status = 'SUBMITTING'
                            """,
                            (m_jid, now_rec, now_rec, req_id),
                        )
                        self.db.record_pending_hq_cancellation(
                            hq_job_id=m_jid,
                            entity_kind="decision",
                            entity_id=req_id,
                            attempt=req_att,
                            goal_id=gid,
                            epoch=req_human_ep,
                            plan_epoch=int(req_plan_ep or 0),
                            reason="reconcile_submitting_stale_or_paused",
                            state="CANCEL_PENDING",
                            usage_key=dec_ukey,
                            now=now_rec,
                        )
                        if lid:
                            self.db.release_lease(lid, now=now_rec)
                        summary["recovered_submitting_decisions"] = (
                            summary.get("recovered_submitting_decisions", 0) + 1
                        )
                if must_cancel_hq and self.execution_mode == "hq":
                    try:
                        self.hq.cancel_jobs([m_jid], reason="reconcile_submitting_stale_or_paused")
                    except Exception:
                        pass
            else:
                # Authoritatively ABSENT on HQ (`lookup_status == "OK"`): safe to retire lease,
                # release `RESERVED` admission row, and requeue `SUBMITTING -> PENDING`!
                with self.db._tx() as (conn, _p):
                    now_rec = time.time()
                    d_row = conn.execute(
                        "SELECT status, payload_json FROM decision_requests WHERE req_id = ?",
                        (req_id,),
                    ).fetchone()
                    if d_row is None or str(d_row["status"]) != "SUBMITTING":
                        continue
                    cur_pl = json.loads(d_row["payload_json"])
                    if str(cur_pl.get("submit_nonce") or "") != s_nonce:
                        continue
                    if lid:
                        self.db.release_lease(lid, now=now_rec)
                    self.db.release_goal_budget_reservation(
                        gid, dec_ukey, reason="submitting_crash_before_hq_accept", now=now_rec
                    )
                    g_row = conn.execute(
                        "SELECT status, payload_json FROM goals WHERE goal_id = ?",
                        (gid,),
                    ).fetchone()
                    g_ok = False
                    if g_row is not None:
                        fg = ResearchGoal.from_dict(json.loads(g_row["payload_json"]))
                        g_ok = (
                            str(g_row["status"]).upper() == "ACTIVE"
                            and fg.autonomy_mode != "paused"
                            and int(fg.version) == req_goal_ver
                            and int(fg.human_epoch) == req_human_ep
                            and (
                                req_plan_ep is None
                                or int(getattr(fg, "plan_epoch", 0) or 0) == int(req_plan_ep)
                            )
                        )
                    cleaned_pl = {
                        k: v
                        for k, v in cur_pl.items()
                        if k
                        not in {
                            "lease_id",
                            "submit_nonce",
                            "usage_key",
                            "submit_job_name",
                            "submit_owner_pid",
                            "submit_proc_start_time",
                            "submitting_started_ts",
                        }
                    }
                    if g_ok:
                        conn.execute(
                            """
                            UPDATE decision_requests
                            SET status = 'PENDING', payload_json = ?, updated_ts = ?
                            WHERE req_id = ? AND status = 'SUBMITTING'
                            """,
                            (
                                json.dumps(cleaned_pl, sort_keys=True, ensure_ascii=False),
                                now_rec,
                                req_id,
                            ),
                        )
                    else:
                        conn.execute(
                            """
                            UPDATE decision_requests
                            SET status = 'CANCELLED', payload_json = ?, resolved_ts = ?, updated_ts = ?
                            WHERE req_id = ? AND status = 'SUBMITTING'
                            """,
                            (
                                json.dumps(cleaned_pl, sort_keys=True, ensure_ascii=False),
                                now_rec,
                                now_rec,
                                req_id,
                            ),
                        )
                    summary["recovered_submitting_decisions"] = (
                        summary.get("recovered_submitting_decisions", 0) + 1
                    )

    def _sweep_decision_requests(
        self,
        goals_by_id: Mapping[str, ResearchGoal],
        summary: dict[str, Any],
    ) -> None:
        """Submit ready DecisionRequests whose coalesce window elapsed and reconcile finished ones."""
        now = time.time()

        # 0. Reconcile interrupted durable SUBMITTING decision intents before admission
        self._reconcile_submitting_decisions(summary)

        # 1. Reconcile RUNNING HQ decision jobs
        if self.execution_mode == "hq":
            for req in self.db.list_decision_requests(status="RUNNING"):
                try:
                    jid = req.get("hq_job_id")
                    if jid is None:
                        continue
                    q_dec = self.goal_manager._query_hq_job_state(self.hq, int(jid))
                    if str(q_dec.get("lookup_status") or "").upper() != "FOUND":
                        continue
                    info = q_dec.get("job")
                    if not isinstance(info, Mapping):
                        continue
                    jinfo = info.get("info") or {}
                    stats = jinfo.get("task_stats") or {}
                    if (
                        int(stats.get("running", 0) or 0) == 0
                        and int(stats.get("waiting", 0) or 0) == 0
                        and (
                            int(stats.get("failed", 0) or 0) > 0
                            or int(stats.get("canceled", 0) or 0) > 0
                            or int(stats.get("aborted", 0) or 0) > 0
                        )
                    ):
                        fresh = self.db.get_decision_request(str(req["req_id"]))
                        if fresh and fresh.get("status") == "RUNNING":
                            if hasattr(self.db, "confirm_hq_cancellation"):
                                self.db.confirm_hq_cancellation(
                                    int(jid),
                                    entity_kind="decision",
                                    entity_id=str(req["req_id"]),
                                    attempt=max(1, int(req.get("attempt", 1) or 1)),
                                    now=now,
                                )
                            self.db.resolve_decision_request(
                                str(req["req_id"]),
                                status="FAILED",
                                resolution={"reason": f"HQ decision job {jid} failed/canceled"},
                            )
                except Exception as exc:
                    self._isolate_step_exception(
                        exc,
                        phase="reconcile_running_decision",
                        goal_id=str(req.get("goal_id") or "") or None,
                        summary=summary,
                    )

        # 2. Enqueue initial work for RESOLVED decisions
        for req in self.db.list_decision_requests(status="RESOLVED", limit=100):
            rid = str(req["req_id"])
            if rid in self._handled_decision_ids:
                continue
            gid = str(req["goal_id"])
            goal = goals_by_id.get(gid)
            if goal is None:
                continue
            accepted_ids = list((req.get("resolution") or {}).get("accepted_hypothesis_ids") or [])
            for hid in accepted_ids:
                try:
                    hyp = self.db.get_hypothesis(str(hid))
                    if hyp is not None:
                        created = self._enqueue_initial_hypothesis_work(goal, hyp)
                        summary["tasks_enqueued"] += len(created)
                except Exception as exc:
                    self._isolate_step_exception(
                        exc,
                        phase="apply_resolved_decision_hypothesis",
                        goal_id=gid,
                        hypothesis_id=str(hid),
                        summary=summary,
                    )
            self._handled_decision_ids.add(rid)
            with contextlib.suppress(Exception):
                self._record_ledger_event(
                    "CONTROLLER_DECISION_APPLIED",
                    goal_id=gid,
                    payload={"req_id": rid, "accepted_count": len(accepted_ids)},
                )

        # 3. Sweep PENDING requests whose coalesce window has elapsed
        pending_ready = self.db.list_decision_requests(status="PENDING", ready_before_ts=now)
        for req in pending_ready:
            gid = str(req["goal_id"])
            try:
                goal = goals_by_id.get(gid)
                if goal is None or goal.autonomy_mode != "unattended":
                    continue
                req_plan_epoch = (req.get("payload") or {}).get("plan_epoch")
                if (
                    int(goal.version) != int(req["goal_version"])
                    or int(goal.human_epoch) != int(req["human_epoch"])
                    or (
                        req_plan_epoch is not None
                        and int(getattr(goal, "plan_epoch", 0) or 0) != int(req_plan_epoch)
                    )
                ):
                    self.db.resolve_decision_request(
                        str(req["req_id"]),
                        status="DISCARDED_BY_NEWER_HUMAN_INTERVENTION",
                        resolution={"reason": "stale_before_submit"},
                    )
                    continue

                tier_u = str(req.get("tier") or "L1").upper()
                if tier_u != "L0":
                    dec_att = max(1, int(req.get("attempt", 1) or 1))
                    dec_ukey = str(
                        (req.get("payload") or {}).get("usage_key")
                        or f"decision:{req['req_id']}:a{dec_att}"
                    )
                    if self.db.is_budget_exhausted(gid, exclude_usage_key=dec_ukey):
                        summary.setdefault("blocked_by_budget", []).append(str(req["req_id"]))
                        self.db.upsert_knowledge_item(
                            gid,
                            "dispatch_bottleneck",
                            f"budget:dec:{req['req_id']}",
                            {
                                "status": "WAIT_BUDGET",
                                "reason": f"Token budget exhausted for goal {gid}",
                                "req_id": str(req["req_id"]),
                            },
                            goal_version=int(goal.version),
                        )
                        self.flag_owner_escalation(
                            gid,
                            OWNER_GATE_PAID_COMPUTE,
                            f"Token budget exhausted for goal {gid} (decision {req['req_id']} gated)",
                        )
                        continue
            except Exception as exc:
                self._isolate_step_exception(
                    exc,
                    phase="sweep_pending_decision_precheck",
                    goal_id=gid,
                    hypothesis_id=str((req.get("payload") or {}).get("hypothesis_id") or "") or None,
                    summary=summary,
                )
                continue

            try:
                if self.execution_mode == "hq" and not self.dry_run_llm:
                    self.decision_engine.submit_decision_to_hq(
                        str(req["req_id"]), arr_home=self.arr_home
                    )
                    summary["decisions_submitted"] = summary.get("decisions_submitted", 0) + 1
                else:
                    out = self.decision_engine.process_request(
                        str(req["req_id"]), dry_run_llm=self.dry_run_llm
                    )
                    if out.get("status") == "BUDGET_EXHAUSTED":
                        summary.setdefault("blocked_by_budget", []).append(str(req["req_id"]))
                        self.flag_owner_escalation(
                            gid,
                            OWNER_GATE_PAID_COMPUTE,
                            f"Token budget exhausted for goal {gid}",
                        )
                        continue
                    summary["decisions_submitted"] = summary.get("decisions_submitted", 0) + 1
                    if out.get("status") == "RESOLVED":
                        accepted_ids = list((out.get("resolution") or {}).get("accepted_hypothesis_ids") or [])
                        for hid in accepted_ids:
                            hyp = self.db.get_hypothesis(str(hid))
                            if hyp is not None:
                                created = self._enqueue_initial_hypothesis_work(goal, hyp)
                                summary["tasks_enqueued"] += len(created)
                        self._handled_decision_ids.add(str(req["req_id"]))
                        self._record_ledger_event(
                            "CONTROLLER_DECISION_APPLIED",
                            goal_id=gid,
                            payload={"req_id": str(req["req_id"]), "accepted_count": len(accepted_ids)},
                        )
            except FencedStaleWriteError:
                summary["stale_isolated"] = summary.get("stale_isolated", 0) + 1
                continue
            except PolicyBlockedError as exc:
                b_code = exc.code or ("WAIT_RESOURCE" if exc.waitable else "BLOCKED_POLICY")
                if exc.waitable or b_code in ("WAIT_RESOURCE", "WAIT_FOREGROUND_RESERVE"):
                    summary.setdefault("waiting_for_model", []).append(str(req["req_id"]))
                    cur_p = dict(req.get("payload") or {})
                    exc_pin = getattr(exc, "pinned_route", None)
                    if isinstance(exc_pin, Mapping) and not cur_p.get("pinned_route"):
                        cur_p["pinned_route"] = dict(exc_pin)
                    cur_p["wait_reason"] = b_code
                    with self.db._tx() as (conn, _pending):
                        conn.execute(
                            "UPDATE decision_requests SET payload_json = ?, updated_ts = ? WHERE req_id = ? AND status = 'PENDING'",
                            (json.dumps(cur_p, sort_keys=True, ensure_ascii=False), now, str(req["req_id"])),
                        )
                    self.db.upsert_knowledge_item(
                        gid,
                        "dispatch_bottleneck",
                        f"wait:dec:{req['req_id']}",
                        {
                            "status": "WAITING_FOR_PINNED_MODEL",
                            "wait_reason": b_code,
                            "reason": str(exc),
                            "rule_id": exc.rule_id,
                            "req_id": str(req["req_id"]),
                        },
                        goal_version=int(goal.version),
                    )
                elif b_code == "BUDGET_EXHAUSTED":
                    summary.setdefault("blocked_by_budget", []).append(str(req["req_id"]))
                    cur_p = dict(req.get("payload") or {})
                    cur_p["wait_reason"] = "WAIT_BUDGET"
                    with self.db._tx() as (conn, _pending):
                        conn.execute(
                            "UPDATE decision_requests SET payload_json = ?, updated_ts = ? WHERE req_id = ? AND status = 'PENDING'",
                            (json.dumps(cur_p, sort_keys=True, ensure_ascii=False), now, str(req["req_id"])),
                        )
                    self.flag_owner_escalation(
                        gid,
                        OWNER_GATE_PAID_COMPUTE,
                        str(exc),
                    )
                else:
                    self.db.resolve_decision_request(
                        str(req["req_id"]),
                        status=b_code,
                        resolution={"reason": str(exc), "rule_id": exc.rule_id},
                    )
                    self.flag_owner_escalation(
                        gid,
                        OWNER_GATE_PAID_COMPUTE if "QUOTA" in b_code else OWNER_GATE_GOAL_BAR,
                        str(exc),
                    )
            except Exception as exc:
                self._isolate_step_exception(
                    exc,
                    phase="sweep_pending_decision_submit",
                    goal_id=gid,
                    hypothesis_id=str((req.get("payload") or {}).get("hypothesis_id") or "") or None,
                    summary=summary,
                )

    def _check_goal_escalations(self, goal: ResearchGoal, *, extra_reason: str = "") -> None:
        if self.db.is_budget_exhausted(goal.goal_id):
            self.flag_owner_escalation(
                goal.goal_id,
                OWNER_GATE_PAID_COMPUTE,
                f"Token budget exhausted for goal {goal.goal_id}",
            )
        hyps = self.db.list_hypotheses(goal_id=goal.goal_id)
        if hyps:
            routes = [h for h in hyps if h.kind == HypothesisKind.COMPETING_ROUTE]
            if routes and all(
                h.status in ("REFUTED", "GRAVEYARD", "ESCALATED")
                or h.scientific_verdict == ScientificVerdict.REFUTED
                for h in routes
            ):
                active_other = [
                    h
                    for h in hyps
                    if h.status not in ("REFUTED", "GRAVEYARD", "PRUNED", "COMPLETED", "ESCALATED")
                ]
                if not active_other:
                    self.flag_owner_escalation(
                        goal.goal_id,
                        OWNER_GATE_ROUTE_AUDIT,
                        extra_reason or "All competing routes refuted/exhausted; route pivot decision needed",
                    )

        if self._failed_blind_audits.get(goal.goal_id, 0) > 2:
            self.flag_owner_escalation(
                goal.goal_id,
                OWNER_GATE_ROUTE_AUDIT,
                f"Failed blind audits > 2 ({self._failed_blind_audits[goal.goal_id]})",
            )

    def _dispatch_ready_tasks(
        self,
        summary: dict[str, Any],
        *,
        goal_id: str | None = None,
    ) -> None:
        all_goal_records = self.db.list_goal_records()
        goals_by_id: dict[str, ResearchGoal] = {}
        goal_status_by_id: dict[str, str] = {}
        for rec in all_goal_records:
            g = rec["goal"]
            if goal_id is not None and g.goal_id != goal_id:
                continue
            goal_status_by_id[g.goal_id] = str(rec["status"])
            if rec["status"] == "ACTIVE" and g.autonomy_mode != "paused":
                goals_by_id[g.goal_id] = g
        if not goals_by_id:
            return

        all_tasks = self.db.list_tasks()
        running_tasks = [t for t in all_tasks if t.status in ACTIVE_TASK_STATES]
        active_by_project: dict[str, int] = {}
        active_by_goal: dict[str, int] = {}
        for rt in running_tasks:
            g = goals_by_id.get(rt.goal_id)
            if g:
                active_by_project[g.project] = active_by_project.get(g.project, 0) + 1
                active_by_goal[g.goal_id] = active_by_goal.get(g.goal_id, 0) + 1

        ready_all = [t for t in self.db.list_ready_tasks() if t.goal_id in goals_by_id]
        if not ready_all:
            return

        hyps_by_id: dict[str, HypothesisNode] = {}
        recent_receipts_by_goal: dict[str, int] = {}
        for gid in goals_by_id:
            for h in self.db.list_hypotheses(goal_id=gid):
                hyps_by_id[h.hypothesis_id] = h
            recent_receipts_by_goal[gid] = len(self.db.list_receipts(goal_id=gid))

        blocked_goals: set[str] = set()
        pending_decs = {
            str(d.get("goal_id") or "")
            for d in (
                self.db.list_decision_requests(status="PENDING")
                + self.db.list_decision_requests(status="RUNNING")
            )
        }
        ready_by_goal: dict[str, int] = {}
        for t in ready_all:
            ready_by_goal[t.goal_id] = ready_by_goal.get(t.goal_id, 0) + 1
        for rec in all_goal_records:
            g = rec["goal"]
            if rec["status"] != "ACTIVE" or g.autonomy_mode in ("paused", "supervised"):
                blocked_goals.add(g.goal_id)
            elif g.goal_id in pending_decs and ready_by_goal.get(g.goal_id, 0) == 0:
                blocked_goals.add(g.goal_id)

        global_cfg = self.db.get_global_config()
        total_slots = int(
            global_cfg.get("total_slots")
            or max(1, sum(g.max_concurrent_branches for g in goals_by_id.values()))
        )

        # Build lightweight view objects including paused/blocked goals so elastic borrowing computes cleanly
        class _GoalSlotView:
            def __init__(self, g: ResearchGoal, status_str: str) -> None:
                self.goal_id = g.goal_id
                self.status = status_str
                self.priority_weight = g.priority_weight
                self.min_guaranteed_slots = g.min_guaranteed_slots
                self.max_concurrent_branches = g.max_concurrent_branches

        slot_views = {
            rec["goal"].goal_id: _GoalSlotView(rec["goal"], rec["status"])
            for rec in all_goal_records
            if goal_id is None or rec["goal"].goal_id == goal_id
        }
        allocs = self.scheduler.compute_goal_slot_allocations(
            list(slot_views.values()),
            active_by_goal=active_by_goal,
            ready_by_goal=ready_by_goal,
            blocked_goals=blocked_goals,
            total_slots=total_slots,
            arr_home=self.arr_home,
        )

        ranked_items = self.scheduler.rank_ready_tasks_globally(
            ready_all,
            goals_by_id,
            hyps_by_id,
            active_by_goal=active_by_goal,
            recent_receipts_by_goal=recent_receipts_by_goal,
            blocked_goals=blocked_goals,
            total_slots=total_slots,
            arr_home=self.arr_home,
        )
        queued_projects = [goals_by_id[item["task"].goal_id].project for item in ranked_items]

        if self.execution_mode == "hq":
            self.hq.ensure_running()

        active_leases = self.db.list_active_leases()
        bg_cap, _fg_reserved = self.scheduler.effective_background_slot_cap(
            total_slots, arr_home=self.arr_home
        )

        for item in ranked_items:
            if len(running_tasks) >= bg_cap and bg_cap > 0:
                break
            if bg_cap == 0:
                break
            task: StageTask = item["task"]
            goal: ResearchGoal = item["goal"]
            hyp: HypothesisNode = item["hyp"]

            if (
                int(getattr(task, "goal_version", 1)) != int(goal.version)
                or int(getattr(task, "human_epoch", 0)) != int(goal.human_epoch)
                or int(getattr(task, "plan_epoch", 0) or 0) != int(getattr(goal, "plan_epoch", 0) or 0)
            ):
                with contextlib.suppress(StateTransitionError):
                    self.db.transition_task_status(
                        task.task_id, "READY", "CANCELLED", finished_ts=time.time()
                    )
                continue

            goal_cap = int((allocs.get(goal.goal_id) or {}).get("allocated_slots", goal.max_concurrent_branches))
            eff_goal_cap = max(1, min(goal.max_concurrent_branches, goal_cap))
            if task.stage in READONLY_REVIEW_STAGES:
                eff_goal_cap = max(eff_goal_cap, 2)
            if active_by_goal.get(goal.goal_id, 0) >= eff_goal_cap:
                continue

            # Capability & policy route check via ModelRouter (preserving review-quorum diversity & role!)
            pinned_route_dict = (
                dict(task.pinned_route) if isinstance(task.pinned_route, Mapping) else None
            )
            role_override = (
                str(pinned_route_dict.get("role") or "").strip()
                if isinstance(pinned_route_dict, Mapping)
                else ""
            )
            if role_override:
                try:
                    role: ScientificRole | None = ScientificRole(role_override)
                except ValueError:
                    role = self.goal_manager._stage_to_role(task.stage, hyp)
            elif (
                isinstance(pinned_route_dict, Mapping) and bool(pinned_route_dict.get("tiebreaker"))
            ) or str(task.step_key or "").endswith("_s2"):
                role = ScientificRole.RESEARCH_DIRECTOR
            else:
                role = self.goal_manager._stage_to_role(task.stage, hyp)

            wf_obj_pol = self.db.get_workflow(task.workflow_id) if task.workflow_id else None
            wf_cfg_pol = dict(wf_obj_pol.config or {}) if wf_obj_pol is not None else None
            wf_pol_map = (
                dict(hyp.params.get("workflow_routing_policies") or {})
                if isinstance(hyp.params.get("workflow_routing_policies"), Mapping)
                else {}
            )
            has_policy = bool(
                global_cfg.get("routing_policy")
                or global_cfg.get("routing_policy_nl")
                or goal.routing_policy
                or goal.routing_policy_nl
                or getattr(hyp, "routing_policy", None)
                or hyp.params.get("routing_policy")
                or (task.workflow_id and wf_pol_map.get(task.workflow_id))
                or (wf_cfg_pol and wf_cfg_pol.get("routing_policy"))
                or (isinstance(pinned_route_dict, Mapping) and pinned_route_dict.get("routing_policy"))
            )
            if not has_policy and hasattr(self.router, "has_effective_policy"):
                try:
                    import inspect as _inspect_hep

                    hep_fn = getattr(self.router, "has_effective_policy")
                    hep_sig = _inspect_hep.signature(hep_fn).parameters
                    hep_kw: dict[str, Any] = {"goal": goal, "hyp": hyp}
                    if "db" in hep_sig:
                        hep_kw["db"] = self.db
                    if "workflow_id" in hep_sig:
                        hep_kw["workflow_id"] = task.workflow_id
                    if "task" in hep_sig:
                        hep_kw["task"] = task
                    if "wf_config" in hep_sig:
                        hep_kw["wf_config"] = wf_cfg_pol
                    has_policy = bool(hep_fn(**hep_kw))
                except Exception:
                    pass
            if role is not None and (not self.dry_run_llm or has_policy):
                avoid_fams: set[str] = set()
                excl_specs: set[str] = set()
                pref_avoid_fams: list[str] = []
                if isinstance(pinned_route_dict, Mapping):
                    avoid_fams.update(
                        str(f).strip().lower()
                        for f in (pinned_route_dict.get("avoid_families") or [])
                        if str(f).strip()
                    )
                    excl_specs.update(
                        str(s).strip()
                        for s in (pinned_route_dict.get("exclude_specs") or [])
                        if str(s).strip()
                    )
                    pref_avoid_fams = [
                        str(f).strip().lower()
                        for f in (pinned_route_dict.get("prefer_avoid_families") or [])
                        if str(f).strip()
                    ]

                # Enforce distinct families across concurrent sibling review slots in the same workflow step!
                if task.workflow_id and task.stage in READONLY_REVIEW_STAGES:
                    if wf_obj_pol is not None:
                        qstate = wf_obj_pol.state.get("review_quorum")
                        if isinstance(qstate, Mapping):
                            avoid_fams.update(
                                str(f).strip().lower()
                                for f in (qstate.get("contributor_families") or [])
                                if str(f).strip()
                            )
                    my_prefix = (task.step_key or "").rsplit("_s", 1)[0]
                    for rt in running_tasks:
                        if (
                            rt.task_id != task.task_id
                            and rt.workflow_id == task.workflow_id
                            and (rt.step_key or "").rsplit("_s", 1)[0] == my_prefix
                        ):
                            rt_route = rt.pinned_route or {}
                            if isinstance(rt_route, Mapping) and rt_route.get("family"):
                                avoid_fams.add(str(rt_route["family"]).strip().lower())
                            if rt.model_spec:
                                excl_specs.add(rt.model_spec)

                policy_matches_role = (
                    self.goal_manager._policy_targets_role(
                        goal,
                        hyp,
                        role.value,
                        is_review_stage=(task.stage in READONLY_REVIEW_STAGES),
                        workflow_id=task.workflow_id,
                        wf_config=wf_cfg_pol,
                        global_policy=global_cfg.get("routing_policy"),
                        global_policy_nl=str(global_cfg.get("routing_policy_nl") or ""),
                    )
                    if has_policy
                    else False
                )
                explicit_spec = None
                if isinstance(pinned_route_dict, Mapping) and pinned_route_dict.get("canonical_model"):
                    pool_p = str(pinned_route_dict.get("pool") or "").strip()
                    mod_p = str(pinned_route_dict.get("canonical_model") or "").strip()
                    eff_p = str(pinned_route_dict.get("requested_effort") or "").strip()
                    if pool_p and mod_p and not policy_matches_role:
                        explicit_spec = f"{pool_p}:{mod_p}:{eff_p}" if eff_p else f"{pool_p}:{mod_p}"
                elif hyp.params.get("builder_spec") and task.stage == StageKind.CODE and not policy_matches_role:
                    explicit_spec = str(hyp.params["builder_spec"]).strip() or None

                eval_goal = self.goal_manager._goal_for_role_resolution(
                    goal,
                    role,
                    is_review_stage=(task.stage in READONLY_REVIEW_STAGES),
                    has_avoid_families=bool(avoid_fams),
                )
                try:
                    import inspect as _inspect_rr

                    rr_sig = _inspect_rr.signature(self.router.resolve_route).parameters
                    rr_extra: dict[str, Any] = {}
                    if "db" in rr_sig:
                        rr_extra["db"] = self.db
                    if "workflow_id" in rr_sig:
                        rr_extra["workflow_id"] = task.workflow_id
                    if "task" in rr_sig:
                        rr_extra["task"] = task
                    if "wf_config" in rr_sig and wf_cfg_pol is not None:
                        rr_extra["wf_config"] = wf_cfg_pol
                    resolved_route = self.router.resolve_route(
                        role,
                        goal=eval_goal,
                        hyp=hyp,
                        explicit_spec=explicit_spec,
                        avoid_families=sorted(avoid_fams),
                        prefer_avoid_families=pref_avoid_fams,
                        exclude_specs=sorted(excl_specs),
                        **rr_extra,
                    )
                    pinned_route_dict = {
                        **(pinned_route_dict or {}),
                        **resolved_route.to_dict(),
                        "role": role.value,
                        "plan_epoch": int(getattr(goal, "plan_epoch", 0) or 0),
                        "avoid_families": sorted(avoid_fams),
                        "exclude_specs": sorted(excl_specs),
                    }
                    pinned_route_dict.pop("wait_reason", None)
                except PolicyBlockedError as exc:
                    b_code = exc.code or ("WAIT_RESOURCE" if exc.waitable else "BLOCKED_POLICY")
                    if exc.waitable or b_code in ("WAIT_RESOURCE", "WAIT_FOREGROUND_RESERVE"):
                        summary.setdefault("waiting_for_model", []).append(task.task_id)
                        exc_pin = getattr(exc, "pinned_route", None)
                        wait_route = {
                            **(dict(exc_pin) if isinstance(exc_pin, Mapping) else {}),
                            **(pinned_route_dict or {}),
                            "wait_reason": b_code,
                        }
                        try:
                            self.db.transition_task_status(
                                task.task_id,
                                "READY",
                                "READY",
                                pinned_route=wait_route,
                            )
                        except Exception:
                            pass
                        self.db.upsert_knowledge_item(
                            goal.goal_id,
                            "dispatch_bottleneck",
                            f"wait:{task.task_id}",
                            {
                                "status": "WAITING_FOR_PINNED_MODEL",
                                "wait_reason": b_code,
                                "reason": str(exc),
                                "rule_id": exc.rule_id,
                                "task_id": task.task_id,
                            },
                            hypothesis_id=hyp.hypothesis_id,
                            goal_version=int(goal.version),
                        )
                        continue
                    with contextlib.suppress(StateTransitionError):
                        self.db.transition_task_status(
                            task.task_id,
                            "READY",
                            b_code,
                            finished_ts=time.time(),
                        )
                    try:
                        self.db.transition_hypothesis_status(
                            hyp.hypothesis_id,
                            hyp.status,
                            b_code,
                            failure_domain=FailureDomain.ENV_CONFIG_MISMATCH,
                        )
                    except Exception:
                        pass
                    self.flag_owner_escalation(
                        goal.goal_id,
                        OWNER_GATE_PAID_COMPUTE if "QUOTA" in b_code else OWNER_GATE_GOAL_BAR,
                        str(exc),
                        hypothesis_id=hyp.hypothesis_id,
                    )
                    continue

            conflict_ok, _conflict_reason = ConflictManager.check_task_conflicts(
                task,
                goal,
                hyp,
                running_tasks,
                active_leases,
                goals_by_id=goals_by_id,
                hyps_by_id=hyps_by_id,
                max_concurrent_test_sim=self.max_concurrent_test_sim,
            )
            if not conflict_ok:
                continue

            eff_pool = (
                str(pinned_route_dict.get("pool"))
                if isinstance(pinned_route_dict, Mapping) and pinned_route_dict.get("pool")
                else task.resources.llm_pool
            )
            fg_quota_ok, _fg_reason = self.scheduler.check_foreground_quota_reserve(
                eff_pool, arr_home=self.arr_home
            )
            if not fg_quota_ok:
                continue

            task_ukey = f"task:{task.task_id}:a{task.attempt}"
            usage = self.db.get_goal_token_usage(goal.goal_id, exclude_usage_key=task_ukey)
            admit_extra_kw: dict[str, Any] = {}
            try:
                import inspect as _inspect_admit

                admit_sig = _inspect_admit.signature(self.arbiter.can_admit).parameters
                if "db" in admit_sig:
                    admit_extra_kw["db"] = self.db
                if "exclude_usage_key" in admit_sig:
                    admit_extra_kw["exclude_usage_key"] = task_ukey
            except Exception:
                pass
            admit = self.arbiter.can_admit(
                task,
                project=goal.project,
                lane_id=hyp.hypothesis_id,
                tokens_used_for_goal=usage.get("committed_tokens", usage["total_tokens"]),
                max_tokens_budget=goal.max_tokens_budget,
                active_by_project=active_by_project,
                queued_projects=queued_projects,
                total_slots=total_slots,
                **admit_extra_kw,
            )
            if not admit.allowed:
                if admit.reason.startswith("MEM_BELOW_10GB_WAIT"):
                    continue
                if admit.reason.startswith(("TOKEN_BUDGET_EXCEEDED", "VALLEY_FUSE_BLOCKED", "POOL_QUOTA_EXHAUSTED")):
                    self.flag_owner_escalation(
                        goal.goal_id,
                        OWNER_GATE_PAID_COMPUTE,
                        admit.reason,
                        hypothesis_id=hyp.hypothesis_id,
                    )
                continue

            needs_llm_budget = bool(
                role is not None
                or task.resources.llm_pool
                or (isinstance(pinned_route_dict, Mapping) and pinned_route_dict.get("pool"))
            )
            if needs_llm_budget and hasattr(self.db, "is_budget_exhausted"):
                if self.db.is_budget_exhausted(goal.goal_id, exclude_usage_key=task_ukey):
                    summary.setdefault("blocked_by_budget", []).append(task.task_id)
                    self.flag_owner_escalation(
                        goal.goal_id,
                        OWNER_GATE_PAID_COMPUTE,
                        f"Token budget exhausted for goal {goal.goal_id}",
                        hypothesis_id=hyp.hypothesis_id,
                    )
                    continue

            now = time.time()
            ttl_s = float(max(60, task.resources.wall_timeout_s + 60))
            lease_owner_id = f"task:{task.task_id}:a{task.attempt}"
            if task.stage in READONLY_REVIEW_STAGES:
                ro_sha = str(
                    (pinned_route_dict or {}).get("head_sha")
                    or hyp.head_sha
                    or hyp.base_sha
                    or "HEAD"
                )
                lease = self.db.acquire_branch_ro_lease(
                    goal.goal_id,
                    hyp.hypothesis_id,
                    ro_sha,
                    task.task_id,
                    holder=lease_owner_id,
                    ttl_s=ttl_s,
                    max_concurrent=4,
                    now=now,
                )
            else:
                lease = self.db.acquire_branch_writer_lease(
                    goal.goal_id,
                    hyp.hypothesis_id,
                    task.task_id,
                    holder=lease_owner_id,
                    ttl_s=ttl_s,
                    now=now,
                )
            if lease is None:
                continue

            scope_leases: list[dict[str, Any]] = []
            if task.stage == StageKind.CODE:
                acquired_scopes = ConflictManager.acquire_repo_scope_leases(
                    self.db,
                    task_id=task.task_id,
                    goal_id=goal.goal_id,
                    repo_path=goal.repo_path,
                    writable_paths=hyp.writable_paths,
                    holder=lease_owner_id,
                    ttl_s=ttl_s,
                    now=now,
                )
                if acquired_scopes is None:
                    self.db.release_lease(lease["lease_id"], now=now)
                    continue
                scope_leases = acquired_scopes

            # Reserve concurrency admission budget ONLY AFTER all leases succeed!
            reserved_budget_here = False
            if needs_llm_budget and hasattr(self.db, "reserve_goal_budget") and not self.dry_run_llm:
                res_tok = (
                    min(4000, max(1, int(goal.max_tokens_budget) // 2))
                    if int(goal.max_tokens_budget) > 0
                    else 4000
                )
                ok_b, _comm_t, _max_t = self.db.reserve_goal_budget(
                    goal.goal_id,
                    task_ukey,
                    entity_kind="stage_task",
                    entity_id=task.task_id,
                    attempt=int(task.attempt),
                    model=str(
                        (pinned_route_dict or {}).get("canonical_model")
                        or task.model_spec
                        or ""
                    ),
                    reserve_tokens=res_tok,
                    exclude_usage_key=task_ukey,
                )
                if not ok_b:
                    ConflictManager.release_task_leases(self.db, task.task_id, now=now)
                    summary.setdefault("blocked_by_budget", []).append(task.task_id)
                    self.flag_owner_escalation(
                        goal.goal_id,
                        OWNER_GATE_PAID_COMPUTE,
                        f"Token budget reservation exhausted for goal {goal.goal_id}",
                        hypothesis_id=hyp.hypothesis_id,
                    )
                    continue
                reserved_budget_here = True

            acquired_lease_keys = [
                str(lease["resource_key"]),
                *(str(sl["resource_key"]) for sl in scope_leases if sl.get("resource_key")),
            ]
            acquired_lease_ids = [
                str(lease["lease_id"]),
                *(str(sl["lease_id"]) for sl in scope_leases if sl.get("lease_id")),
            ]
            pinned_route_dict = {
                **(pinned_route_dict or {}),
                "expected_goal_version": int(goal.version),
                "lease_owner_id": lease_owner_id,
                "acquired_lease_keys": acquired_lease_keys,
                "acquired_lease_ids": acquired_lease_ids,
            }

            if self.execution_mode == "inline":
                try:
                    running_task = self.db.fenced_transition_task(
                        task.task_id,
                        "READY",
                        "RUNNING",
                        expected_goal_version=int(task.goal_version),
                        expected_human_epoch=int(task.human_epoch),
                        expected_attempt=int(task.attempt),
                        expected_lease_id=str(lease["lease_id"]),
                        started_ts=now,
                        pinned_route=pinned_route_dict,
                        now=now,
                    )
                except FencedStaleWriteError:
                    if reserved_budget_here and hasattr(self.db, "release_goal_budget_reservation"):
                        self.db.release_goal_budget_reservation(
                            goal.goal_id, task_ukey, reason="fenced_stale_inline", now=now
                        )
                    ConflictManager.release_task_leases(self.db, task.task_id, now=now)
                    continue
                receipt = self.stage_runner.run_stage(
                    running_task,
                    hyp,
                    goal,
                    db=self.db,
                    persist=True,
                    notify=False,
                    dry_run_llm=self.dry_run_llm,
                )
                end_ts = time.time()
                final_status = "DONE" if receipt.process_status == ProcessStatus.EXIT_OK else "FAILED"
                self.db.transition_task_status(
                    task.task_id,
                    ("READY", "RUNNING"),
                    final_status,
                    finished_ts=end_ts,
                    now=end_ts,
                )
                ConflictManager.release_task_leases(self.db, task.task_id, now=end_ts)
                active_leases = self.db.list_active_leases(now=end_ts)
                summary["tasks_dispatched"] += 1
            else:
                cmd = [
                    sys.executable,
                    "-m",
                    "conjectra.stage_runner",
                    "--task-id",
                    task.task_id,
                    "--arr-home",
                    str(self.arr_home),
                ]
                if self.dry_run_llm:
                    cmd.append("--dry-run-llm")
                try:
                    import inspect as _inspect_sr

                    sr_mod = __import__("conjectra.stage_runner", fromlist=["main", "build_parser"])
                    sr_parser_fn = getattr(self.stage_runner, "build_parser", None) or getattr(
                        sr_mod, "build_parser", None
                    )
                    if callable(sr_parser_fn):
                        sr_opts = set(sr_parser_fn()._option_string_actions.keys())
                    else:
                        sr_src = _inspect_sr.getsource(getattr(sr_mod, "main"))
                        sr_opts = {
                            flag
                            for flag in ("--attempt", "--goal-version", "--lease-owner")
                            if f'"{flag}"' in sr_src or f"'{flag}'" in sr_src
                        }
                    if "--attempt" in sr_opts:
                        cmd.extend(["--attempt", str(int(task.attempt))])
                    if "--goal-version" in sr_opts:
                        cmd.extend(["--goal-version", str(int(goal.version))])
                    if "--lease-owner" in sr_opts:
                        cmd.extend(["--lease-owner", lease_owner_id])
                except Exception:
                    pass

                hq_res = ResourceSpec.from_dict(task.resources.to_dict())
                if isinstance(pinned_route_dict, Mapping):
                    hq_res.pinned_route = dict(pinned_route_dict)
                    if pinned_route_dict.get("hq_resource"):
                        hq_res.hq_llm_resource = str(pinned_route_dict["hq_resource"])
                    if pinned_route_dict.get("pool"):
                        hq_res.llm_pool = str(pinned_route_dict["pool"])
                elif admit.resolved_pool:
                    hq_res.llm_pool = admit.resolved_pool
                elif hq_res.llm_pool and ":" in hq_res.llm_pool:
                    hq_res.llm_pool = hq_res.llm_pool.split(":", 1)[0].strip()
                if pinned_route_dict is not None:
                    try:
                        self.db.transition_task_status(
                            task.task_id,
                            "READY",
                            "READY",
                            pinned_route=pinned_route_dict,
                            now=now,
                        )
                    except Exception:
                        pass
                try:
                    hq_jid = self.hq.submit_task(
                        task,
                        command=cmd,
                        resources=hq_res,
                        stage=task.stage,
                        cwd=str(self.arr_home),
                        wrap_exit=True,
                    )
                except Exception:
                    if reserved_budget_here and hasattr(self.db, "release_goal_budget_reservation"):
                        self.db.release_goal_budget_reservation(
                            goal.goal_id, task_ukey, reason="hq_submit_failed", now=now
                        )
                    ConflictManager.release_task_leases(self.db, task.task_id, now=now)
                    raise
                try:
                    dispatched_task = self.db.fenced_transition_task(
                        task.task_id,
                        "READY",
                        "DISPATCHED",
                        expected_goal_version=int(task.goal_version),
                        expected_human_epoch=int(task.human_epoch),
                        expected_attempt=int(task.attempt),
                        expected_lease_id=str(lease["lease_id"]),
                        hq_job_id=hq_jid,
                        started_ts=now,
                        pinned_route=pinned_route_dict,
                        now=now,
                    )
                except FencedStaleWriteError:
                    self.hq.cancel_jobs([hq_jid], reason="fenced_stale_dispatch")
                    if reserved_budget_here and hasattr(self.db, "release_goal_budget_reservation"):
                        self.db.release_goal_budget_reservation(
                            goal.goal_id, task_ukey, reason="fenced_stale_dispatch", now=now
                        )
                    ConflictManager.release_task_leases(self.db, task.task_id, now=now)
                    continue
                running_tasks.append(dispatched_task)
                active_leases.append(lease)
                active_leases.extend(scope_leases)
                active_by_project[goal.project] = active_by_project.get(goal.project, 0) + 1
                active_by_goal[goal.goal_id] = active_by_goal.get(goal.goal_id, 0) + 1
                summary["tasks_dispatched"] += 1

    def _recover_resumed_goal_tasks(
        self,
        goal: ResearchGoal,
        summary: dict[str, Any],
    ) -> None:
        """Spawn fresh attempt+1 READY tasks ONLY after cancelled HQ jobs are positively confirmed."""
        now = time.time()

        def _hq_confirmed_done(jid: int | None, ent_id: str = "") -> bool:
            if jid is None:
                return True
            if self.execution_mode != "hq":
                if hasattr(self.db, "confirm_hq_cancellation"):
                    self.db.confirm_hq_cancellation(int(jid), now=now)
                return True
            return self._is_hq_cancellation_confirmed(
                int(jid), goal_id=goal.goal_id, entity_id=ent_id, now=now
            )

        for wf in self.db.list_workflows(goal_id=goal.goal_id):
            if wf.status not in ("READY", "RUNNING", "WAITING"):
                continue
            paused_snaps = list(wf.state.get("paused_tasks") or [])
            if not paused_snaps:
                continue
            rem: list[dict[str, Any]] = []
            for snap in paused_snaps:
                jid = snap.get("hq_job_id")
                tid_s = str(snap.get("task_id") or "")
                if not _hq_confirmed_done(int(jid) if jid is not None else None, tid_s):
                    rem.append({**dict(snap), "wait_reason": "WAIT_CANCEL_CONFIRM"})
                    if tid_s:
                        summary.setdefault("waiting_cancel_confirm", []).append(tid_s)
                    continue
                self.goal_manager._spawn_replacement_task_from_snapshot(goal, snap, now=now)
                summary["tasks_enqueued"] += 1
            if len(rem) != len(paused_snaps):
                self.db.update_workflow_state(
                    wf.workflow_id,
                    state_patch={"paused_tasks": rem},
                    now=now,
                )

        for item in self.db.list_knowledge_items(
            goal_id=goal.goal_id, category="paused_standalone_tasks", limit=5
        ):
            if item.get("key") != "active_pause":
                continue
            content = item.get("content") or {}
            snaps = list(content.get("tasks") or [])
            if not snaps:
                continue
            rem_s: list[dict[str, Any]] = []
            for snap in snaps:
                jid = snap.get("hq_job_id")
                tid_s = str(snap.get("task_id") or "")
                if not _hq_confirmed_done(int(jid) if jid is not None else None, tid_s):
                    rem_s.append({**dict(snap), "wait_reason": "WAIT_CANCEL_CONFIRM"})
                    if tid_s:
                        summary.setdefault("waiting_cancel_confirm", []).append(tid_s)
                    continue
                self.goal_manager._spawn_replacement_task_from_snapshot(goal, snap, now=now)
                summary["tasks_enqueued"] += 1
            if len(rem_s) != len(snaps):
                self.db.upsert_knowledge_item(
                    goal.goal_id,
                    "paused_standalone_tasks",
                    "active_pause",
                    {"tasks": rem_s, "resumed_ts": now},
                    goal_version=int(goal.version),
                    now=now,
                )

    @staticmethod
    def _read_boot_id() -> str:
        p = Path("/proc/sys/kernel/random/boot_id")
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8").strip()
        except OSError:
            pass
        return "local_boot"

    @staticmethod
    def _barrier_pid_is_live(pid: int) -> bool:
        if pid <= 0:
            return False
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            pass
        stat_p = Path(f"/proc/{pid}/stat")
        if stat_p.is_file():
            try:
                raw_stat = stat_p.read_text(encoding="utf-8", errors="ignore")
                rparen = raw_stat.rfind(")")
                if rparen != -1:
                    fields = raw_stat[rparen + 1 :].split()
                    if fields and fields[0] in ("Z", "X", "x"):
                        return False
            except OSError:
                pass
        return True

    def _reconcile_maintenance_barrier(
        self, summary: dict[str, Any]
    ) -> tuple[bool, set[str]]:
        """Evaluate `maintenance_barrier` liveness and safely recover abandoned barriers.

        Returns `(barrier_all, barriered_goals)`.
        """
        barrier = self.db.get_global_config("maintenance_barrier")
        barriered_goals: set[str] = set()
        if not (isinstance(barrier, Mapping) and bool(barrier.get("active"))):
            return False, barriered_goals

        now = time.time()
        barrier_id = str(barrier.get("barrier_id") or "")
        lease_id = str(barrier.get("lease_id") or "")
        owner_pid = int(barrier.get("owner_pid", 0) or 0)
        owner_host = str(barrier.get("owner_host") or socket.gethostname())
        cur_host = socket.gethostname()
        boot_id = str(barrier.get("boot_id") or "")
        cur_boot = self._read_boot_id()
        hb_ts = float(
            barrier.get("heartbeat_ts")
            or barrier.get("started_ts")
            or barrier.get("ts")
            or 0.0
        )
        ttl_sec = float(barrier.get("ttl_sec", 90.0) or 90.0)

        boot_mismatch = bool(boot_id and cur_boot and boot_id != cur_boot)
        ttl_expired = bool(hb_ts > 0.0 and (now - hb_ts) > ttl_sec)
        dead_pid = bool(
            owner_host == cur_host
            and owner_pid > 0
            and not self._barrier_pid_is_live(owner_pid)
        ) or (owner_pid <= 0 and "owner_pid" in barrier)

        if boot_mismatch or ttl_expired or dead_pid:
            auth_lock_ctx = (
                self.db.barrier_authority_lock(nonblocking=True)
                if hasattr(self.db, "barrier_authority_lock")
                else None
            )
            if auth_lock_ctx is not None:
                with auth_lock_ctx as got_auth_lock:
                    if not got_auth_lock:
                        # Recovery owner currently holds `barrier_authority.flock` inside HQ drain;
                        # keep barrier active until it finishes its serialized step!
                        raw_b_goals_busy = [str(g) for g in (barrier.get("goals") or []) if str(g).strip()]
                        return (not raw_b_goals_busy or "*" in raw_b_goals_busy), set(raw_b_goals_busy)
                    return self._takeover_abandoned_barrier_locked(
                        barrier_id=barrier_id,
                        lease_id=lease_id,
                        cur_host=cur_host,
                        cur_boot=cur_boot,
                        summary=summary,
                    )
            return self._takeover_abandoned_barrier_locked(
                barrier_id=barrier_id,
                lease_id=lease_id,
                cur_host=cur_host,
                cur_boot=cur_boot,
                summary=summary,
            )

        raw_b_goals = [str(g) for g in (barrier.get("goals") or []) if str(g).strip()]
        barrier_all = False
        if not raw_b_goals or "*" in raw_b_goals:
            barrier_all = True
        else:
            barriered_goals.update(raw_b_goals)
        self.db.set_global_config(
            "maintenance_ack",
            {
                "barrier_id": barrier_id,
                "lease_id": lease_id,
                "pid": os.getpid(),
                "ack_ts": now,
            },
        )
        summary["maintenance_barrier_active"] = True
        summary["barrier_id"] = barrier_id
        return barrier_all, barriered_goals

    def _takeover_abandoned_barrier_locked(
        self,
        *,
        barrier_id: str,
        lease_id: str,
        cur_host: str,
        cur_boot: str,
        summary: dict[str, Any],
    ) -> tuple[bool, set[str]]:
        reconciled_tasks: list[str] = []
        with self.db._tx() as (conn, _p):
            now_tx = time.time()
            cur_bar = self.db.get_global_config("maintenance_barrier")
            if not (
                isinstance(cur_bar, Mapping)
                and bool(cur_bar.get("active"))
                and str(cur_bar.get("barrier_id") or "") == barrier_id
                and str(cur_bar.get("lease_id") or "") == lease_id
            ):
                return False, set()

            cur_owner_pid = int(cur_bar.get("owner_pid", 0) or 0)
            cur_owner_host = str(cur_bar.get("owner_host") or cur_host)
            cur_boot_val = str(cur_bar.get("boot_id") or "")
            cur_hb_ts = float(
                cur_bar.get("heartbeat_ts")
                or cur_bar.get("started_ts")
                or cur_bar.get("ts")
                or 0.0
            )
            cur_ttl = float(cur_bar.get("ttl_sec", 90.0) or 90.0)
            tx_boot_mismatch = bool(cur_boot_val and cur_boot and cur_boot_val != cur_boot)
            tx_ttl_expired = bool(cur_hb_ts > 0.0 and (now_tx - cur_hb_ts) > cur_ttl)
            tx_dead_pid = bool(
                cur_owner_host == cur_host
                and cur_owner_pid > 0
                and not self._barrier_pid_is_live(cur_owner_pid)
            ) or (cur_owner_pid <= 0 and "owner_pid" in cur_bar)
            if not (tx_boot_mismatch or tx_ttl_expired or tx_dead_pid):
                raw_b_goals_tx = [str(g) for g in (cur_bar.get("goals") or []) if str(g).strip()]
                return (not raw_b_goals_tx or "*" in raw_b_goals_tx), set(raw_b_goals_tx)

            abandon_reason = (
                "boot_id_mismatch"
                if tx_boot_mismatch
                else ("ttl_expired" if tx_ttl_expired else f"dead_owner_pid:{cur_owner_pid}")
            )
            fences = self.db.get_global_config("maintenance_barrier_fences") or {}
            prev_gen = max(
                int(fences.get("barrier_generation", 0) or 0),
                int(cur_bar.get("generation", 0) or 0),
            )
            new_gen = prev_gen + 1
            revoked_list = [str(x) for x in (fences.get("revoked_lease_ids") or []) if str(x)]
            if lease_id and lease_id not in revoked_list:
                revoked_list.append(lease_id)
            if lease_id and hasattr(self.db, "revoke_barrier_lease_in_conn"):
                self.db.revoke_barrier_lease_in_conn(
                    conn,
                    lease_id,
                    barrier_id=barrier_id,
                    generation=new_gen,
                    reason=abandon_reason,
                    now=now_tx,
                )
            self.db.set_global_config(
                "maintenance_barrier_fences",
                {
                    "barrier_generation": new_gen,
                    "active_lease_id": "",
                    "revoked_lease_ids": revoked_list,
                    "updated_ts": now_tx,
                },
                now=now_tx,
            )

            recovery_stamp = self.db.get_global_config("LIVE_V3_RECOVERY_APPLIED")
            if recovery_stamp is None:
                for snap in cur_bar.get("interrupted_task_snapshots") or []:
                    if not isinstance(snap, Mapping):
                        continue
                    gid_s = str(snap.get("goal_id") or "")
                    hid_s = str(snap.get("hypothesis_id") or "")
                    tid_s = str(snap.get("task_id") or "")
                    g_obj = self.db.get_goal(gid_s) if gid_s else None
                    g_st = self.db.get_goal_status(gid_s) if gid_s else None
                    cur_t = self.db.get_task(tid_s) if tid_s else None
                    if (
                        g_obj is not None
                        and g_st == "ACTIVE"
                        and cur_t is not None
                        and cur_t.status == "CANCELLED"
                    ):
                        has_active_sibling = any(
                            t2.status in ("READY", "DISPATCHED", "RUNNING")
                            for t2 in self.db.list_tasks(goal_id=gid_s, hypothesis_id=hid_s)
                        )
                        if not has_active_sibling:
                            jid_s = snap.get("hq_job_id")
                            if jid_s is not None and hasattr(self.db, "record_pending_hq_cancellation"):
                                self.db.record_pending_hq_cancellation(
                                    hq_job_id=int(jid_s),
                                    entity_kind="task",
                                    entity_id=tid_s,
                                    attempt=int(snap.get("attempt", 1) or 1),
                                    goal_id=gid_s,
                                    hypothesis_id=hid_s,
                                    workflow_id=str(snap.get("workflow_id") or ""),
                                    epoch=int(g_obj.human_epoch),
                                    plan_epoch=int(getattr(g_obj, "plan_epoch", 0) or 0),
                                    reason=f"abandoned_barrier_reconcile:{barrier_id}",
                                    state="CANCEL_PENDING",
                                    now=now_tx,
                                )
                            existing_paused = self.db.list_knowledge_items(
                                goal_id=gid_s, category="paused_standalone_tasks", limit=1
                            )
                            prev_snaps = (
                                list((existing_paused[0].get("content") or {}).get("tasks") or [])
                                if existing_paused
                                else []
                            )
                            if not any(s.get("task_id") == tid_s for s in prev_snaps):
                                prev_snaps.append(dict(snap))
                                self.db.upsert_knowledge_item(
                                    gid_s,
                                    "paused_standalone_tasks",
                                    "active_pause",
                                    {"tasks": prev_snaps, "resumed_ts": now_tx},
                                    goal_version=int(g_obj.version),
                                    now=now_tx,
                                )
                                reconciled_tasks.append(tid_s)

            self.db.set_global_config(
                "maintenance_barrier",
                {
                    "active": False,
                    "generation": new_gen,
                    "abandoned_barrier_id": barrier_id,
                    "abandoned_lease_id": lease_id,
                    "abandon_reason": abandon_reason,
                    "recovery_stamp_present": recovery_stamp is not None,
                    "reconciled_task_ids": reconciled_tasks,
                    "recovered_by_pid": os.getpid(),
                    "ts": now_tx,
                },
                now=now_tx,
            )
            self._record_ledger_event(
                "ABANDONED_BARRIER_RECOVERED",
                payload={
                    "barrier_id": barrier_id,
                    "lease_id": lease_id,
                    "generation": new_gen,
                    "owner_pid": cur_owner_pid,
                    "reason": abandon_reason,
                    "recovery_stamp_present": recovery_stamp is not None,
                    "reconciled_task_ids": reconciled_tasks,
                },
            )
        summary["abandoned_barrier_recovered"] = True
        summary["abandoned_barrier_id"] = barrier_id
        return False, set()

    def _maybe_export_status_snapshot(self, *, min_interval_s: float = 5.0) -> bool:
        """Best-effort throttled dashboard status snapshot at end of step (fault-isolated)."""
        now = time.monotonic()
        last_ts = float(getattr(self, "_last_status_export_ts", 0.0))
        if last_ts > 0.0 and (now - last_ts) < max(0.0, float(min_interval_s)):
            return False
        try:
            import importlib

            bridge = importlib.import_module("conjectra.status_snapshot")
            export_fn = getattr(bridge, "export_status_snapshot", None)
            if not callable(export_fn):
                return False
            self._last_status_export_ts = now
            export_fn(self.db, arr_home=self.arr_home)
            return True
        except Exception:
            return False

    def step(self, *, goal_id: str | None = None) -> dict[str, Any]:
        """Run one deterministic multi-Goal reconcile + decision-sweep + streaming dispatch pass."""
        summary: dict[str, Any] = {
            "ts": time.time(),
            "leases_reaped": 0,
            "recovered_tasks": 0,
            "reconciled": 0,
            "stale_isolated": 0,
            "proposed": 0,
            "decisions_enqueued": 0,
            "decisions_submitted": 0,
            "tasks_enqueued": 0,
            "tasks_dispatched": 0,
            "pruned": 0,
        }

        try:
            self._sweep_leases_and_hq_jobs(summary)
        except Exception as exc:
            self._isolate_step_exception(exc, phase="sweep_leases_and_hq_jobs", goal_id=goal_id, summary=summary)
        try:
            self._reconcile_submitting_decisions(summary)
        except Exception as exc:
            self._isolate_step_exception(exc, phase="reconcile_submitting_decisions", goal_id=goal_id, summary=summary)
        if hasattr(self.workflow_runtime, "respawn_superseded_legacy_synthesis_workflows"):
            try:
                respawned_s3 = self.workflow_runtime.respawn_superseded_legacy_synthesis_workflows(goal_id=goal_id)
                if respawned_s3:
                    summary["respawned_synthesis_workflows"] = len(respawned_s3)
            except Exception as exc:
                self._isolate_step_exception(exc, phase="respawn_legacy_synthesis", goal_id=goal_id, summary=summary)

        try:
            expired_wfs = self.workflow_runtime.sweep_expired_review_quorums(goal_id=goal_id)
            if expired_wfs:
                summary["expired_review_quorums"] = len(expired_wfs)
                for b_wf in expired_wfs:
                    self.flag_owner_escalation(
                        b_wf.goal_id,
                        OWNER_GATE_ROUTE_AUDIT,
                        str((b_wf.state.get("review_quorum") or {}).get("block_reason") or "Review quorum deadline expired"),
                        hypothesis_id=b_wf.hypothesis_id,
                    )
        except Exception as exc:
            self._isolate_step_exception(exc, phase="sweep_expired_review_quorums", goal_id=goal_id, summary=summary)

        try:
            barrier_all, barriered_goals = self._reconcile_maintenance_barrier(summary)
            self._consecutive_barrier_lock_failures = 0
        except sqlite3.OperationalError as exc:
            if "locked" in str(exc).lower() and "disk" not in str(exc).lower():
                self._consecutive_barrier_lock_failures += 1
                summary["authority_lock_blocked"] = True
                summary["barrier_lock_failures"] = self._consecutive_barrier_lock_failures
                if self._consecutive_barrier_lock_failures < self._max_consecutive_barrier_lock_failures:
                    self._isolate_step_exception(
                        exc,
                        phase="reconcile_maintenance_barrier",
                        goal_id=goal_id,
                        summary=summary,
                    )
                    self._maybe_export_status_snapshot()
                    return summary
                fatal_exc = RuntimeError(
                    f"Persistent barrier authority failure after {self._consecutive_barrier_lock_failures} consecutive database lock failures: {exc}"
                )
                self._emit_db_independent_fatal_marker(
                    fatal_exc, phase="reconcile_maintenance_barrier", goal_id=goal_id
                )
                self._stop_requested = True
                raise fatal_exc from exc
            raise

        all_goals = (
            [g for g in [self.db.get_goal(goal_id)] if g is not None]
            if goal_id
            else self.db.list_goals(status="ACTIVE")
        )
        goals = [
            g
            for g in all_goals
            if not barrier_all and g.goal_id not in barriered_goals
        ]
        if not goals:
            self._maybe_export_status_snapshot()
            return summary

        goals_by_id = {g.goal_id: g for g in goals}

        # 0. Spawn replacement attempt+1 tasks for any resumed Goals once cancelled HQ jobs finished
        for goal in goals:
            try:
                self._recover_resumed_goal_tasks(goal, summary)
            except Exception as exc:
                self._isolate_step_exception(
                    exc, phase="recover_resumed_goal_tasks", goal_id=goal.goal_id, summary=summary
                )

        # 1. Process all newly finished/failed tasks across ALL active non-barriered Goals
        for goal in goals:
            try:
                goal_tasks = self.db.list_tasks(goal_id=goal.goal_id)
            except Exception as exc:
                self._isolate_step_exception(
                    exc, phase="list_tasks", goal_id=goal.goal_id, summary=summary
                )
                continue
            for task in goal_tasks:
                if task.status in TERMINAL_TASK_STATES and task.task_id not in self._handled_tasks:
                    try:
                        self._handle_completed_task(task, goal, summary)
                    except Exception as exc:
                        self._isolate_step_exception(
                            exc,
                            phase="handle_completed_task",
                            goal_id=goal.goal_id,
                            hypothesis_id=task.hypothesis_id,
                            task_id=task.task_id,
                            summary=summary,
                        )
                        with contextlib.suppress(Exception):
                            self._mark_task_handled(task)

        # 2. Per-Goal non-blocking propose/decision & escalation checks
        for goal in goals:
            try:
                self._maybe_propose_round(goal, summary)
                self._check_goal_escalations(goal)
            except Exception as exc:
                self._isolate_step_exception(
                    exc, phase="maybe_propose_round", goal_id=goal.goal_id, summary=summary
                )

        # 3. Sweep ready DecisionRequests whose coalesce window elapsed
        try:
            self._sweep_decision_requests(goals_by_id, summary)
        except Exception as exc:
            self._isolate_step_exception(
                exc, phase="sweep_decision_requests", goal_id=goal_id, summary=summary
            )

        # 4. Immediately refill freed worker slots across ALL active non-barriered Goals (streaming, 0 barrier)
        if goal_id is not None:
            try:
                self._dispatch_ready_tasks(summary, goal_id=goal_id)
            except Exception as exc:
                self._isolate_step_exception(
                    exc, phase="dispatch_ready_tasks", goal_id=goal_id, summary=summary
                )
        elif barriered_goals:
            for g in goals:
                try:
                    self._dispatch_ready_tasks(summary, goal_id=g.goal_id)
                except Exception as exc:
                    self._isolate_step_exception(
                        exc, phase="dispatch_ready_tasks", goal_id=g.goal_id, summary=summary
                    )
        else:
            try:
                self._dispatch_ready_tasks(summary, goal_id=None)
            except Exception as exc:
                self._isolate_step_exception(
                    exc, phase="dispatch_ready_tasks_global", goal_id=None, summary=summary
                )
                for g in goals:
                    try:
                        self._dispatch_ready_tasks(summary, goal_id=g.goal_id)
                    except Exception as g_exc:
                        self._isolate_step_exception(
                            g_exc, phase="dispatch_ready_tasks", goal_id=g.goal_id, summary=summary
                        )
        self._maybe_export_status_snapshot()
        return summary

    def _acquire_pid_lock(self) -> None:
        if self.pid_path.exists():
            try:
                doc = json.loads(self.pid_path.read_text(encoding="utf-8"))
                old_pid = int(doc.get("pid", 0))
                old_start = doc.get("proc_start_time")
                if old_pid > 0 and old_pid != os.getpid():
                    liveness = verify_process_identity(old_pid, old_start)
                    if liveness.alive:
                        raise RuntimeError(f"Controller daemon already running with PID {old_pid}")
            except (OSError, ValueError, json.JSONDecodeError):
                pass

        pid = os.getpid()
        payload = {
            "pid": pid,
            "proc_start_time": read_proc_start_time(pid),
            "started_ts": time.time(),
        }
        self.pid_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    def _release_pid_lock(self) -> None:
        try:
            if self.pid_path.exists():
                doc = json.loads(self.pid_path.read_text(encoding="utf-8"))
                if int(doc.get("pid", 0)) == os.getpid():
                    self.pid_path.unlink(missing_ok=True)
        except Exception:
            pass

    def stop(self) -> None:
        """Signal the reactor loop to exit cleanly and wake select() immediately."""
        self._stop_requested = True
        wfd = self._wake_write_fd
        if wfd is not None:
            try:
                os.write(wfd, b"\x01")
            except OSError:
                pass
        send_reactor_wake(self.arr_home, {"event": "STOP"})

    def serve_forever(
        self,
        *,
        goal_id: str | None = None,
        max_seconds: float | None = None,
        on_step: Any = None,
    ) -> dict[str, Any]:
        """Run the Unix-socket event reactor with deterministic ExitStack cleanup (`R5 P2 #3`)."""
        self._stop_requested = False
        steps = 0
        wake_events = 0
        deadline = (time.monotonic() + max_seconds) if max_seconds is not None else None

        with contextlib.ExitStack() as stack:
            self._acquire_pid_lock()
            stack.callback(self._release_pid_lock)

            if self.sock_path.exists():
                self.sock_path.unlink(missing_ok=True)
            stack.callback(lambda: self.sock_path.unlink(missing_ok=True))

            srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            stack.callback(srv.close)
            srv.bind(str(self.sock_path))
            srv.listen(64)
            srv.setblocking(False)

            wake_r, wake_w = os.pipe()

            def _close_fd(fd: int) -> None:
                with contextlib.suppress(OSError):
                    os.close(fd)

            stack.callback(_close_fd, wake_w)
            stack.callback(_close_fd, wake_r)
            os.set_blocking(wake_r, False)
            os.set_blocking(wake_w, False)
            self._wake_write_fd = wake_w
            stack.callback(lambda: setattr(self, "_wake_write_fd", None))

            sel = selectors.DefaultSelector()
            stack.callback(sel.close)
            sel.register(srv, selectors.EVENT_READ)
            sel.register(wake_r, selectors.EVENT_READ)

            prev_sigterm = signal.getsignal(signal.SIGTERM)
            prev_sigint = signal.getsignal(signal.SIGINT)

            def _handle_sig(_signum: int, _frame: Any) -> None:
                self.stop()

            def _restore_signals() -> None:
                with contextlib.suppress(ValueError):
                    signal.signal(signal.SIGTERM, prev_sigterm)
                    signal.signal(signal.SIGINT, prev_sigint)

            try:
                signal.signal(signal.SIGTERM, _handle_sig)
                signal.signal(signal.SIGINT, _handle_sig)
                stack.callback(_restore_signals)
            except ValueError:
                pass  # non-main thread in tests

            def _safe_step() -> dict[str, Any]:
                try:
                    return self.step(goal_id=goal_id)
                except Exception as exc:
                    if self._is_fatal_authority_or_db_error(exc):
                        self._emit_db_independent_fatal_marker(
                            exc, phase="serve_forever_step", goal_id=goal_id
                        )
                        self._stop_requested = True
                        raise
                    err_summary: dict[str, Any] = {
                        "ts": time.time(),
                        "errors": 0,
                        "step_errors": [],
                    }
                    self._isolate_step_exception(
                        exc,
                        phase="serve_forever_step",
                        goal_id=goal_id,
                        summary=err_summary,
                    )
                    return err_summary

            res = _safe_step()
            steps += 1
            if callable(on_step):
                on_step(res)

            while not self._stop_requested:
                timeout = self.sweep_interval_s
                if deadline is not None:
                    rem = deadline - time.monotonic()
                    if rem <= 0:
                        break
                    timeout = min(timeout, rem)

                events = sel.select(timeout=timeout)
                if self._stop_requested:
                    break
                triggered = False
                for key, _ in events:
                    if key.fileobj == wake_r:
                        try:
                            os.read(wake_r, 1024)
                        except OSError:
                            pass
                    elif key.fileobj is srv:
                        while True:
                            try:
                                conn, _ = srv.accept()
                            except BlockingIOError:
                                break
                            with conn:
                                conn.settimeout(1.0)
                                chunks: list[bytes] = []
                                while True:
                                    try:
                                        data = conn.recv(4096)
                                    except OSError:
                                        break
                                    if not data:
                                        break
                                    chunks.append(data)
                                if chunks:
                                    wake_events += 1
                            triggered = True
                if self._stop_requested:
                    break
                res = _safe_step()
                steps += 1
                if callable(on_step):
                    on_step(res)
                if triggered:
                    continue

        return {"steps": steps, "wake_events": wake_events}
