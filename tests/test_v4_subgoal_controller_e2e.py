"""V4 SubGoal Wave-Gated Controller, Workflow Collapse, HITL Scope, and Sub-50ms CLI E2E Tests."""

from __future__ import annotations

import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Sequence

import pytest

from conjectra import cli
from conjectra.arbiter import ResourceArbiter
from conjectra.controller import ResearchController, _make_stage_task
from conjectra.interactive import (
    build_takeover_brief,
    get_overview,
    inspect_state,
    intervene,
)
from conjectra.state_db import StateDB
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
    ResolvedModelRoute,
    ResourceSpec,
    ReviewTier,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
    SubGoalKind,
    SubGoalNode,
    SubGoalStatus,
    WorkflowNode,
)
from conjectra.workflow_runtime import WorkflowRuntime, WorkflowSelector


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_repo(repo_dir: Path) -> str:
    repo_dir.mkdir(parents=True, exist_ok=True)
    _git(repo_dir, "init", "-b", "main")
    _git(repo_dir, "config", "user.name", "tester")
    _git(repo_dir, "config", "user.email", "tester@conjectra.local")
    (repo_dir / "src").mkdir(exist_ok=True)
    (repo_dir / "tests").mkdir(exist_ok=True)
    (repo_dir / "src" / "mod.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (repo_dir / "tests" / "test_mod.py").write_text(
        "from src.mod import run\ndef test_run():\n    assert run() == 1\n",
        encoding="utf-8",
    )
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-m", "init")
    return _git(repo_dir, "rev-parse", "HEAD")


def _make_goal(
    goal_id: str,
    repo: Path,
    base_sha: str,
    *,
    routing_policy: dict[str, Any] | None = None,
    routing_policy_nl: str = "",
    max_concurrent_branches: int = 4,
) -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project="subgoal-e2e",
        title=f"Goal {goal_id}",
        direction="maximize",
        primary_metric="score",
        baseline_value=0.5,
        baseline_sha=base_sha,
        repo_path=str(repo),
        kill_criteria=["score < 0.1"],
        banned_paths=["lockbox/"],
        writable_prefixes=["src/", "tests/"],
        max_tokens_budget=200_000,
        max_concurrent_branches=max_concurrent_branches,
        adapter_name="generic",
        routing_policy=dict(routing_policy or {}),
        routing_policy_nl=routing_policy_nl,
        target_value=0.95,
    )


def _make_subgoal(
    goal: ResearchGoal,
    sg_id: str,
    *,
    kind: SubGoalKind = SubGoalKind.DIRECTION_TRACK,
    status: str = "ACTIVE",
    local_metric: str = "local_score",
    target_value: float | None = 0.85,
    depends_on_subgoals: list[str] | None = None,
    priority_weight: float = 1.0,
    wave_gated: bool = True,
    max_auto_generations: int = 1,
    compute_target: str = "cpu_only",
    base_sha: str | None = None,
    sealed_sha: str | None = None,
    params: dict[str, Any] | None = None,
) -> SubGoalNode:
    return SubGoalNode(
        subgoal_id=sg_id,
        goal_id=goal.goal_id,
        title=f"SubGoal {sg_id}",
        direction_thesis=f"Advance {sg_id} on {local_metric}",
        direct_contribution=f"Contributes {sg_id} module to {goal.goal_id}",
        local_metric=local_metric,
        kind=kind,
        status=status,
        baseline_value=0.5,
        target_value=target_value,
        direction="maximize",
        kill_criteria=[f"{local_metric} < 0.1"],
        depends_on_subgoals=list(depends_on_subgoals or []),
        track_branch=f"track/{goal.project}/{sg_id}",
        base_sha=base_sha if base_sha is not None else goal.baseline_sha,
        sealed_sha=sealed_sha,
        compute_target=compute_target,
        writable_prefixes=["src/"],
        priority_weight=priority_weight,
        max_auto_generations=max_auto_generations,
        wave_gated=wave_gated,
        params=dict(params or {}),
    )


def _make_hyp(
    goal: ResearchGoal,
    hid: str,
    *,
    subgoal_id: str | None = None,
    kind: HypothesisKind = HypothesisKind.COMPETING_ROUTE,
    fidelity: FidelityRung = FidelityRung.F1_FAST_SCREEN,
    tags: list[str] | None = None,
    writable_paths: list[str] | None = None,
    params: dict[str, Any] | None = None,
    status: str = "PENDING",
    scientific_verdict: ScientificVerdict = ScientificVerdict.UNKNOWN,
    head_sha: str | None = None,
    priority_evoi: float = 1.5,
) -> HypothesisNode:
    merged_params = {"requires_llm_code": True, **dict(params or {})}
    if subgoal_id is not None:
        merged_params.setdefault("subgoal_id", subgoal_id)
    return HypothesisNode(
        hypothesis_id=hid,
        goal_id=goal.goal_id,
        parent_id=None,
        kind=kind,
        title=f"Hyp {hid}",
        statement=f"Verify {hid} improves target metric",
        mechanism=f"mech_{hid}",
        fidelity=fidelity,
        priority_evoi=priority_evoi,
        writable_paths=list(writable_paths or ["src/mod.py"]),
        params=merged_params,
        kill_criteria=["score < 0.1"],
        acceptance_cmd="pytest -q",
        screen_cmd="pytest -q",
        confirm_cmd="pytest -q",
        branch_name=f"conjectra/{goal.project}/{hid}",
        base_sha=goal.baseline_sha,
        head_sha=head_sha or goal.baseline_sha,
        status=status,
        scientific_verdict=scientific_verdict,
        fingerprint=f"fp_{goal.goal_id}_{hid}",
        tags=list(tags or []),
        subgoal_id=subgoal_id,
    )


class _FakeModelRouter:
    def resolve_route(
        self,
        role: Any,
        *,
        avoid_families: Sequence[str] = (),
        exclude_specs: Sequence[str] = (),
        **_kw: Any,
    ) -> ResolvedModelRoute:
        role_val = role.value if hasattr(role, "value") else str(role)
        avoid = {str(f).strip().lower() for f in (avoid_families or ()) if str(f).strip()}
        excluded = {str(s).strip().lower() for s in (exclude_specs or ()) if str(s).strip()}
        if "director" in role_val or "auditor" in role_val or "reviewer" in role_val:
            candidates = [
                ("codex", "gpt-5-codex", "openai", "high", "llmcodexaudit", "local_codex_exec_direct", "llm_codex"),
                ("google", "gemini-pro", "google", "high", "llmgoogle", "local_gemini_direct", "llm_google"),
                ("claude", "claude-opus-5-5", "anthropic", "high", "llmclaude", "local_claude_direct", "llm_claude"),
            ]
        else:
            candidates = [
                ("claude", "claude-opus-5-5", "anthropic", "medium", "llmclaude", "local_claude_direct", "llm_claude"),
                ("codex", "gpt-5-codex", "openai", "medium", "llmcodex", "local_codex_exec_direct", "llm_codex"),
                ("google", "gemini-pro", "google", "medium", "llmgoogle", "local_gemini_direct", "llm_google"),
            ]
        for pool, model, fam, effort, hq_res, backend, log_pool in candidates:
            if fam in avoid:
                continue
            spec = f"{pool}:{model}:{effort}".lower()
            if spec in excluded or f"{pool}:{model}".lower() in excluded:
                continue
            return ResolvedModelRoute(
                backend_mode=backend,
                execution_host="local",
                multipool_host_arg="local",
                pool=pool,
                canonical_model=model,
                requested_effort=effort,
                wire_effort=effort,
                family=fam,
                logical_pool=log_pool,
                hq_resource=hq_res,
                matched_rule_id="fake_router",
            )
        return ResolvedModelRoute(
            backend_mode="local_codex_exec_direct",
            execution_host="local",
            multipool_host_arg="local",
            pool="codex",
            canonical_model="gpt-5-codex",
            requested_effort="high",
            wire_effort="high",
            family="openai",
            logical_pool="llm_codex",
            hq_resource="llmcodexaudit",
            matched_rule_id="fake_router_fallback",
        )


class _StubHQ:
    def __init__(self) -> None:
        self.submitted: list[dict[str, Any]] = []
        self.cancelled: list[tuple[list[int], str | None]] = []
        self.running_job_ids: set[int] = set()
        self._next_jid = 700

    def ensure_running(self) -> dict[str, Any]:
        return {"started_server": False, "started_worker": False}

    def submit_task(self, task: Any, *, resources: ResourceSpec, stage: StageKind, **_kw: Any) -> int:
        self._next_jid += 1
        jid = self._next_jid
        self.running_job_ids.add(jid)
        tid = getattr(task, "task_id", None) or _kw.get("task_id") or str(task)
        self.submitted.append(
            {
                "job_id": jid,
                "task_id": tid,
                "stage": stage,
                "llm_pool": resources.llm_pool,
            }
        )
        return jid

    def cancel_jobs(self, job_ids: list[int], reason: str | None = None) -> dict[int, bool]:
        ids = [int(j) for j in job_ids]
        self.cancelled.append((ids, reason))
        for jid in ids:
            self.running_job_ids.discard(jid)
        return {jid: True for jid in ids}

    def is_pending_canceled_or_dep_failed(self, _jid: int) -> tuple[bool, bool]:
        return False, False

    def worker_list(self) -> list[dict[str, Any]]:
        return [{"id": 1, "ended": None}]

    def job_info(self, jid: int) -> dict[str, Any]:
        is_run = int(jid) in self.running_job_ids
        return {
            "finished_at": None if is_run else "2026-11-28T00:00:00Z",
            "info": {
                "is_open": False,
                "task_stats": {
                    "running": 1 if is_run else 0,
                    "waiting": 0,
                    "finished": 0,
                    "failed": 0,
                    "canceled": 0 if is_run else 1,
                    "aborted": 0,
                },
            },
        }


def _make_receipt(
    goal: ResearchGoal,
    hyp: HypothesisNode,
    task: StageTask,
    *,
    stage: StageKind | None = None,
    sverdict: ScientificVerdict = ScientificVerdict.CONFIRMED,
    averdict: AuditVerdict = AuditVerdict.NOT_REQUIRED,
    pstatus: ProcessStatus = ProcessStatus.EXIT_OK,
    everdict: EngineeringVerdict = EngineeringVerdict.PASS,
    metrics: dict[str, Any] | None = None,
    head_sha: str | None = None,
    in_tokens: int = 0,
    out_tokens: int = 0,
) -> EvidenceReceipt:
    return EvidenceReceipt(
        receipt_id=f"rcpt_{task.task_id}_{stage.value if stage else task.stage.value}",
        task_id=task.task_id,
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=stage or task.stage,
        base_sha=hyp.base_sha,
        head_sha=head_sha or hyp.head_sha or hyp.base_sha,
        command="pytest -q",
        exit_code=0 if pstatus == ProcessStatus.EXIT_OK else 1,
        wall_sec=0.2,
        max_rss_mb=32.0,
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        model_used="test-model",
        process_status=pstatus,
        engineering_verdict=everdict,
        scientific_verdict=sverdict,
        audit_verdict=averdict,
        tests_passed=1 if pstatus == ProcessStatus.EXIT_OK else 0,
        tests_total=1,
        metrics=dict(metrics if metrics is not None else {"local_score": {"value": 0.90}}),
        raw_artifact_paths=[],
        evidence_validity=EvidenceValidity.VALID_EMPIRICAL,
        goal_version=int(goal.version),
        human_epoch=int(goal.human_epoch),
        plan_epoch=int(goal.plan_epoch),
    )


def test_wave_gated_pre_f1_bypass_and_foundation_build_collapse(tmp_path: Path) -> None:
    arr_home = tmp_path / "state_home"
    repo = tmp_path / "repo"
    base_sha = _init_repo(repo)
    with StateDB(arr_home) as db:
        goal = _make_goal("g-r1", repo, base_sha, routing_policy_nl="理论需要跨家族审查")
        db.save_goal(goal, status="ACTIVE")

        sg_wave = _make_subgoal(goal, "sg-wave", wave_gated=True)
        db.save_subgoal(sg_wave)

        rt = WorkflowRuntime(db, router=_FakeModelRouter())

        # 1. Wave-gated F1 hypothesis skips Pre-F1 DESIGN_REVIEW even with theory/backbone tags + cross-family NL policy
        h_wave = _make_hyp(
            goal,
            "h-wave-f1",
            subgoal_id=sg_wave.subgoal_id,
            tags=["theory", "backbone"],
            params={"requires_llm_code": True, "require_design_review": True},
        )
        db.register_hypothesis(h_wave)
        assert rt._requires_design_review(goal, h_wave, {}, ReviewTier.RED_TEAM_TIEBREAKER) is False
        wf_w, tasks_w = rt.create_workflow_for_hypothesis(goal, h_wave, primitive="direct_hq")
        assert len(tasks_w) == 1 and tasks_w[0].stage == StageKind.CODE
        assert wf_w.config.get("design_review_mode") == "FAST_LANE_0_LLM"

        # Fallback via hyp.params["wave_gated"] = True when subgoal_id is unset
        h_param_wg = _make_hyp(
            goal,
            "h-param-wg",
            subgoal_id=None,
            tags=["backbone"],
            params={"requires_llm_code": True, "wave_gated": True},
        )
        assert rt._requires_design_review(goal, h_param_wg, {}, ReviewTier.RED_TEAM_TIEBREAKER) is False

        # 2. Mandatory triggers STILL enforce DESIGN_REVIEW on wave_gated=True SubGoals
        mandatory_cases = [
            _make_hyp(goal, "h-f2", subgoal_id=sg_wave.subgoal_id, fidelity=FidelityRung.F2_FULL_CONFIRM),
            _make_hyp(goal, "h-f3", subgoal_id=sg_wave.subgoal_id, fidelity=FidelityRung.F3_MILESTONE_AUDIT),
            _make_hyp(goal, "h-holdout", subgoal_id=sg_wave.subgoal_id, tags=["holdout"]),
            _make_hyp(goal, "h-destr", subgoal_id=sg_wave.subgoal_id, tags=["destructive"]),
            _make_hyp(
                goal,
                "h-paid",
                subgoal_id=sg_wave.subgoal_id,
                params={"requires_paid_gpu": True},
            ),
            _make_hyp(
                goal,
                "h-cloud",
                subgoal_id=sg_wave.subgoal_id,
                params={"required_hardware": "cloud_gpu"},
            ),
        ]
        for h_m in mandatory_cases:
            db.register_hypothesis(h_m)
            assert rt._requires_design_review(goal, h_m, {}, ReviewTier.STANDARD_CROSS_FAMILY) is True
            _wf_m, tasks_m = rt.create_workflow_for_hypothesis(goal, h_m, primitive="direct_hq")
            assert tasks_m and tasks_m[0].stage == StageKind.DESIGN_REVIEW

        sg_cloud = _make_subgoal(goal, "sg-cloud", wave_gated=True, compute_target="cloud_gpu")
        db.save_subgoal(sg_cloud)
        h_sg_cloud = _make_hyp(goal, "h-sg-cloud", subgoal_id=sg_cloud.subgoal_id)
        db.register_hypothesis(h_sg_cloud)
        assert rt._requires_design_review(goal, h_sg_cloud, {}, ReviewTier.FAST_LANE) is True

        # 3. FOUNDATION_BUILD collapses multi-phase/multi-worker specs directly to direct_hq slices
        goal_f = _make_goal("g-found", repo, base_sha)
        db.save_goal(goal_f, status="ACTIVE")
        sg_found = _make_subgoal(goal_f, "sg-found", kind=SubGoalKind.FOUNDATION_BUILD, wave_gated=True)
        db.save_subgoal(sg_found)
        h_found = _make_hyp(
            goal_f,
            "h-found",
            subgoal_id=sg_found.subgoal_id,
            kind=HypothesisKind.PARAMETER_RACE,
            params={
                "requires_llm_code": False,
                "workflow_primitive": "x-teamwork",
                "phases": [
                    {"name": "p1", "slices": [{"name": "s1", "writable_paths": ["src/mod.py"]}]},
                ],
                "workers": [
                    {"name": "w1", "writable_paths": ["src/a.py"]},
                    {"name": "w2", "writable_paths": ["src/b.py"]},
                ],
            },
        )
        db.register_hypothesis(h_found)
        sel = WorkflowSelector().select_primitive(goal_f, h_found, db=db)
        assert sel.primitive == "direct_hq"
        assert sel.rationale == "FOUNDATION_BUILD_DIRECT_HQ_COLLAPSE"

        wf_f, tasks_f = rt.create_workflow_for_hypothesis(goal_f, h_found, primitive="x-teamwork")
        assert wf_f.primitive == "direct_hq"
        assert wf_f.config.get("foundation_build") is True
        assert wf_f.config.get("collapsed_from") == "x-teamwork"
        assert len(tasks_f) == 1 and tasks_f[0].stage == StageKind.CODE

        # Advance FOUNDATION_BUILD directly through CODE -> CPU_GATE -> EXP_SCREEN -> COMPLETED (no EXP_CONFIRM)
        rc_code = _make_receipt(goal_f, h_found, tasks_f[0], stage=StageKind.CODE)
        t_gate = rt.advance_workflow(wf_f, tasks_f[0], rc_code, goal_f)
        assert len(t_gate) == 1 and t_gate[0].stage == StageKind.CPU_GATE

        rc_gate = _make_receipt(goal_f, h_found, t_gate[0], stage=StageKind.CPU_GATE)
        t_scr = rt.advance_workflow(wf_f, t_gate[0], rc_gate, goal_f)
        assert len(t_scr) == 1 and t_scr[0].stage == StageKind.EXP_SCREEN

        rc_scr = _make_receipt(
            goal_f, h_found, t_scr[0], stage=StageKind.EXP_SCREEN, sverdict=ScientificVerdict.CONFIRMED
        )
        t_after = rt.advance_workflow(wf_f, t_scr[0], rc_scr, goal_f)
        assert t_after == []
        wf_done = db.get_workflow(wf_f.workflow_id)
        assert wf_done is not None and wf_done.status == "COMPLETED"


def test_parallel_subgoal_non_starvation_and_wave_gated_suppression(tmp_path: Path) -> None:
    arr_home = tmp_path / "state_home"
    repo = tmp_path / "repo"
    base_sha = _init_repo(repo)
    with StateDB(arr_home) as db:
        goal = _make_goal("g-sched", repo, base_sha, max_concurrent_branches=4)
        db.save_goal(goal, status="ACTIVE")
        db.set_global_config("total_slots", 2)

        sg_paused = _make_subgoal(goal, "sg-paused", status="PAUSED", priority_weight=10.0)
        sg_locked = _make_subgoal(
            goal,
            "sg-locked",
            status="LOCKED_DEP",
            depends_on_subgoals=["sg-high"],
            priority_weight=8.0,
        )
        sg_high = _make_subgoal(goal, "sg-high", status="ACTIVE", priority_weight=3.0, wave_gated=True)
        sg_low = _make_subgoal(goal, "sg-low", status="ACTIVE", priority_weight=1.0, wave_gated=True)
        for sg in (sg_paused, sg_locked, sg_high, sg_low):
            db.save_subgoal(sg)

        h_paused = _make_hyp(
            goal, "h-paused", subgoal_id="sg-paused", kind=HypothesisKind.PARAMETER_RACE,
            writable_paths=["src/p.py"], params={"requires_llm_code": False},
        )
        h_locked = _make_hyp(
            goal, "h-locked", subgoal_id="sg-locked", kind=HypothesisKind.PARAMETER_RACE,
            writable_paths=["src/l.py"], params={"requires_llm_code": False},
        )
        h_high_1 = _make_hyp(
            goal, "h-high-1", subgoal_id="sg-high", kind=HypothesisKind.PARAMETER_RACE,
            writable_paths=["src/h1.py"], params={"requires_llm_code": False},
        )
        h_high_2 = _make_hyp(
            goal, "h-high-2", subgoal_id="sg-high", kind=HypothesisKind.PARAMETER_RACE,
            writable_paths=["src/h2.py"], params={"requires_llm_code": False},
        )
        h_low = _make_hyp(
            goal, "h-low", subgoal_id="sg-low", kind=HypothesisKind.PARAMETER_RACE,
            writable_paths=["src/low.py"], params={"requires_llm_code": False},
        )
        for h in (h_paused, h_locked, h_high_1, h_high_2, h_low):
            db.register_hypothesis(h)

        t_paused = db.enqueue_task(_make_stage_task(goal, h_paused, StageKind.CPU_GATE))
        t_locked = db.enqueue_task(_make_stage_task(goal, h_locked, StageKind.CPU_GATE))
        t_high_1 = db.enqueue_task(_make_stage_task(goal, h_high_1, StageKind.CPU_GATE))
        t_high_2 = db.enqueue_task(_make_stage_task(goal, h_high_2, StageKind.CPU_GATE))
        t_low = db.enqueue_task(_make_stage_task(goal, h_low, StageKind.CPU_GATE))

        # Add a completed receipt inside sg-high while t_high_1 & t_high_2 are still READY/active in the wave!
        t_high_done = _make_stage_task(goal, h_high_1, StageKind.CODE, attempt=1)
        t_high_done.status = "DONE"
        db.enqueue_task(t_high_done)
        db.record_receipt(_make_receipt(goal, h_high_1, t_high_done, stage=StageKind.CODE))

        hq = _StubHQ()
        arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
        ctrl = ResearchController(
            arr_home,
            db=db,
            hq=hq,  # type: ignore[arg-type]
            arbiter=arbiter,
            execution_mode="hq",
            propose_mode="auto",
            dry_run_llm=False,
        )
        ctrl._handled_tasks.add(t_high_done.task_id)
        summary = ctrl.step(goal_id=goal.goal_id)

        # 1. PAUSED and LOCKED_DEP SubGoals are NEVER dispatched
        assert db.get_task(t_paused.task_id).status == "READY"  # type: ignore[union-attr]
        assert db.get_task(t_locked.task_id).status == "READY"  # type: ignore[union-attr]

        # 2. With 2 slots, sg-high gets slot 1 (higher priority_weight=3.0) and sg-low gets slot 2 (non-starvation!)
        assert summary["tasks_dispatched"] == 2
        assert db.get_task(t_high_1.task_id).status == "DISPATCHED"  # type: ignore[union-attr]
        assert db.get_task(t_low.task_id).status == "DISPATCHED"  # type: ignore[union-attr]
        assert db.get_task(t_high_2.task_id).status == "READY"  # type: ignore[union-attr]

        # 3. Wave-gated decision suppression: active wave has remaining tasks -> 0 per-receipt EVIDENCE_UPDATE calls!
        assert summary["decisions_enqueued"] == 0
        assert db.list_decision_requests(goal_id=goal.goal_id) == []


def test_auto_unlock_seal_awaiting_apex_and_opt_in_split(tmp_path: Path) -> None:
    arr_home = tmp_path / "state_home"
    repo = tmp_path / "repo"
    base_sha = _init_repo(repo)
    with StateDB(arr_home) as db:
        hq = _StubHQ()
        arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)

        # Goal 1: sg-1 seals on CONFIRMED wave drain -> unlocks sg-2 propagating sealed_sha -> triggers 1 APEX + 1 TrackLead
        g1 = _make_goal("g-seal", repo, base_sha)
        db.save_goal(g1, status="ACTIVE")
        sg1 = _make_subgoal(g1, "sg-1", local_metric="cache_hit", target_value=0.80)
        sg2 = _make_subgoal(
            g1,
            "sg-2",
            status="LOCKED_DEP",
            depends_on_subgoals=["sg-1"],
            local_metric="latency_ms",
            target_value=12.0,
            base_sha=base_sha,
        )
        sg_stalled = _make_subgoal(g1, "sg-stall", local_metric="acc", target_value=0.9)
        db.save_subgoal(sg1)
        db.save_subgoal(sg2)
        db.save_subgoal(sg_stalled)

        winning_sha = "deadbeefcafebabe1234567890abcdef12345678"
        h1 = _make_hyp(
            g1,
            "h-win",
            subgoal_id="sg-1",
            status="COMPLETED",
            scientific_verdict=ScientificVerdict.CONFIRMED,
            head_sha=winning_sha,
        )
        db.register_hypothesis(h1)
        t1 = _make_stage_task(g1, h1, StageKind.EXP_SCREEN)
        t1.status = "DONE"
        db.enqueue_task(t1)
        db.record_receipt(
            _make_receipt(
                g1,
                h1,
                t1,
                stage=StageKind.EXP_SCREEN,
                sverdict=ScientificVerdict.CONFIRMED,
                metrics={"cache_hit": {"value": 0.88}},
                head_sha=winning_sha,
            )
        )

        # sg-stall has only terminal/stalled hypotheses -> transitions to AWAITING_APEX
        h_stall = _make_hyp(
            g1,
            "h-stall",
            subgoal_id="sg-stall",
            status="REFUTED",
            scientific_verdict=ScientificVerdict.REFUTED,
        )
        db.register_hypothesis(h_stall)

        ctrl = ResearchController(
            arr_home,
            db=db,
            hq=hq,  # type: ignore[arg-type]
            arbiter=arbiter,
            execution_mode="hq",
            propose_mode="auto",
            dry_run_llm=False,
        )
        ctrl._handled_tasks.add(t1.task_id)
        ctrl.step(goal_id=g1.goal_id)

        sg1_after = db.get_subgoal("sg-1")
        sg2_after = db.get_subgoal("sg-2")
        sg_stall_after = db.get_subgoal("sg-stall")
        assert sg1_after is not None and sg1_after.status == SubGoalStatus.SEALED.value
        assert sg1_after.sealed_sha == winning_sha
        assert sg2_after is not None and sg2_after.status == SubGoalStatus.ACTIVE.value
        assert sg2_after.base_sha == winning_sha
        assert sg_stall_after is not None and sg_stall_after.status == SubGoalStatus.AWAITING_APEX.value

        decs = db.list_decision_requests(goal_id=g1.goal_id)
        apex_decs = [d for d in decs if d["trigger_kind"] == "APEX_PORTFOLIO_REVIEW"]
        lead_decs = [d for d in decs if d.get("subgoal_id") == "sg-2" and d["role"] == ScientificRole.RESEARCH_LEAD.value]
        assert len(apex_decs) == 1
        assert apex_decs[0]["role"] == ScientificRole.RESEARCH_DIRECTOR.value
        assert apex_decs[0]["tier"] == "L2"
        assert len(lead_decs) == 1

        # Second step must NOT enqueue duplicate APEX_PORTFOLIO_REVIEW or TrackLead decisions
        s2 = ctrl.step(goal_id=g1.goal_id)
        assert s2["decisions_enqueued"] == 0

        # Goal 2: requires BLIND_AUDIT -> does NOT seal when BLIND_AUDIT failed/missing
        g2 = _make_goal("g-audit", repo, base_sha)
        db.save_goal(g2, status="ACTIVE")
        sg_aud = _make_subgoal(g2, "sg-aud", params={"requires_blind_audit": True})
        db.save_subgoal(sg_aud)
        h_aud = _make_hyp(
            g2,
            "h-aud",
            subgoal_id="sg-aud",
            fidelity=FidelityRung.F3_MILESTONE_AUDIT,
            status="NEEDS_REPAIR",
            scientific_verdict=ScientificVerdict.CONFIRMED,
        )
        h_aud.budget_state["code_repairs_used"] = 2
        db.register_hypothesis(h_aud)
        t_aud = _make_stage_task(g2, h_aud, StageKind.BLIND_AUDIT)
        t_aud.status = "DONE"
        db.enqueue_task(t_aud)
        db.record_receipt(
            _make_receipt(
                g2,
                h_aud,
                t_aud,
                stage=StageKind.BLIND_AUDIT,
                sverdict=ScientificVerdict.NOT_APPLICABLE,
                averdict=AuditVerdict.FAIL,
            )
        )
        ctrl._handled_tasks.add(t_aud.task_id)
        ctrl.step(goal_id=g2.goal_id)
        assert db.get_subgoal("sg-aud").status == SubGoalStatus.AWAITING_APEX.value  # type: ignore[union-attr]

        # Goal 3: Opt-in INITIAL_SUBGOAL_SPLIT when routing_policy["enable_subgoals"] is truthy
        g3 = _make_goal("g-split", repo, base_sha, routing_policy={"enable_subgoals": True})
        db.save_goal(g3, status="ACTIVE")
        ctrl.step(goal_id=g3.goal_id)
        decs_g3 = db.list_decision_requests(goal_id=g3.goal_id)
        assert len(decs_g3) == 1
        assert decs_g3[0]["trigger_kind"] == "INITIAL_SUBGOAL_SPLIT"
        assert decs_g3[0]["role"] == ScientificRole.RESEARCH_DIRECTOR.value
        assert decs_g3[0]["tier"] == "L2"

        # Goal 4: Legacy Goal without enable_subgoals and 0 subgoals keeps v3 INITIAL_PLANNING behavior
        g4 = _make_goal("g-legacy", repo, base_sha)
        db.save_goal(g4, status="ACTIVE")
        ctrl.step(goal_id=g4.goal_id)
        decs_g4 = db.list_decision_requests(goal_id=g4.goal_id)
        assert len(decs_g4) == 1
        assert decs_g4[0]["trigger_kind"] == "INITIAL_PLANNING"
        assert decs_g4[0]["role"] == ScientificRole.RESEARCH_LEAD.value
        assert decs_g4[0]["tier"] == "L1"


def test_subgoal_interventions_cli_overview_inspect_takeover_and_latency(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arr_home = tmp_path / "state_home"
    arr_home.mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    base_sha = _init_repo(repo)

    # Start a real AF_UNIX reactor.sock listener to verify wakeup on scope="subgoal" intervene
    sock_path = arr_home / "reactor.sock"
    received_wakes: list[dict[str, Any]] = []
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(8)
    srv.settimeout(0.5)
    stop_srv = threading.Event()

    def _listen() -> None:
        while not stop_srv.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                continue
            with conn:
                data = conn.recv(4096)
                if data:
                    received_wakes.append(json.loads(data.decode("utf-8").splitlines()[0]))

    t_srv = threading.Thread(target=_listen, daemon=True)
    t_srv.start()

    try:
        db = StateDB(arr_home)
        hq = _StubHQ()
        goal = _make_goal("g-hitl", repo, base_sha)
        db.save_goal(goal, status="ACTIVE")

        # 1. Spawn SubGoal via interactive.intervene(scope="subgoal", action="spawn")
        sp1 = intervene(
            arr_home,
            scope="subgoal",
            action="spawn",
            goal_id=goal.goal_id,
            target_id="sg-track-a",
            payload={
                "title": "Track A sparse cache",
                "direction_thesis": "Sparsify cache lines",
                "direct_contribution": "Reduces memory traffic",
                "local_metric": "hit_rate",
                "target_value": 0.9,
                "priority_weight": 2.0,
                "initial_slices": [
                    {"hypothesis_id": "h-track-a", "title": "Slice A1", "mechanism": "sparse_a1"}
                ],
            },
            db=db,
            hq=hq,
        )
        assert sp1["reactor_woke"] is True
        assert sp1["result"]["subgoal_id"] == "sg-track-a"
        assert sp1["result"]["created_hypothesis_ids"] == ["h-track-a"]

        # Create a sibling SubGoal + running task/workflow for pause/resume/retune/kill/merge tests
        sg_b = _make_subgoal(goal, "sg-track-b", local_metric="throughput", target_value=100.0, priority_weight=1.0)
        db.save_subgoal(sg_b)
        h_a = db.get_hypothesis("h-track-a")
        assert h_a is not None
        rt = WorkflowRuntime(db, router=_FakeModelRouter())
        wf_a, t_a_list = rt.create_workflow_for_hypothesis(goal, h_a, primitive="direct_hq")
        t_a = t_a_list[0]
        # Simulate DISPATCHED task on HQ
        db.transition_task_status(t_a.task_id, "READY", "DISPATCHED", hq_job_id=901)
        hq.running_job_ids.add(901)

        # 2. Pause sg-track-a -> cancels running task, pauses workflow, keeps goal & sg-track-b ACTIVE
        p_res = intervene(
            arr_home,
            scope="subgoal",
            action="pause",
            goal_id=goal.goal_id,
            target_id="sg-track-a",
            reason="operator pause track A",
            db=db,
            hq=hq,
        )
        assert p_res["result"]["status"] == "PAUSED"
        assert t_a.task_id in p_res["result"]["cancelled_task_ids"]
        assert wf_a.workflow_id in p_res["result"]["paused_workflow_ids"]
        assert db.get_subgoal("sg-track-a").status == "PAUSED"  # type: ignore[union-attr]
        assert db.get_subgoal("sg-track-b").status == "ACTIVE"  # type: ignore[union-attr]
        assert db.get_goal_status(goal.goal_id) == "ACTIVE"

        # 3. Resume sg-track-a -> restores ACTIVE, resumes workflow, spawns attempt+1 replacement task
        r_res = intervene(
            arr_home,
            scope="subgoal",
            action="resume",
            goal_id=goal.goal_id,
            target_id="sg-track-a",
            db=db,
            hq=hq,
        )
        assert r_res["result"]["status"] == "ACTIVE"
        assert wf_a.workflow_id in r_res["result"]["resumed_workflow_ids"]
        assert len(r_res["result"]["spawned_replacement_task_ids"]) == 1
        assert db.get_workflow(wf_a.workflow_id).status == "RUNNING"  # type: ignore[union-attr]

        # 4. Retune sg-track-b from AWAITING_APEX back to ACTIVE with new target/weight via CLI intervene
        db.transition_subgoal_status("sg-track-b", "ACTIVE", "AWAITING_APEX")
        db.close()

        rc_iv = cli.main(
            [
                "--arr-home",
                str(arr_home),
                "intervene",
                "--scope",
                "subgoal",
                "--action",
                "retune",
                "--goal-id",
                goal.goal_id,
                "--target-id",
                "sg-track-b",
                "--payload",
                json.dumps({"target_value": 120.0, "priority_weight": 3.5}),
            ]
        )
        assert rc_iv == 0
        iv_json = json.loads(capsys.readouterr().out)
        assert iv_json["result"]["status"] == "ACTIVE"
        assert iv_json["result"]["subgoal"]["target_value"] == 120.0
        assert iv_json["result"]["subgoal"]["priority_weight"] == 3.5

        # 5. CLI `conjectra subgoals` and `conjectra inspect --subgoal-id`
        assert cli.main(["--arr-home", str(arr_home), "subgoals", "--goal-id", goal.goal_id, "--status", "ACTIVE"]) == 0
        sgs_out = json.loads(capsys.readouterr().out)
        assert sgs_out["counts"]["total"] == 2
        assert sgs_out["interactive_latency_ms"] < 50.0

        assert cli.main(["--arr-home", str(arr_home), "inspect", "--subgoal-id", "sg-track-b"]) == 0
        insp_sg = json.loads(capsys.readouterr().out)
        assert insp_sg["subgoal"]["subgoal_id"] == "sg-track-b"
        assert insp_sg["interactive_latency_ms"] < 50.0

        # 6. Merge two sealed SubGoals + Kill a third SubGoal with reallocation
        db2 = StateDB(arr_home)
        try:
            db2.transition_subgoal_status("sg-track-a", "ACTIVE", "SEALED", sealed_sha="aaa1111")
            db2.transition_subgoal_status("sg-track-b", "ACTIVE", "SEALED", sealed_sha="bbb2222")

            m_res = intervene(
                arr_home,
                scope="subgoal",
                action="merge",
                goal_id=goal.goal_id,
                payload={"source_subgoal_ids": ["sg-track-a", "sg-track-b"], "subgoal_id": "sg-merged"},
                db=db2,
            )
            assert m_res["result"]["subgoal_id"] == "sg-merged"
            assert db2.get_subgoal("sg-merged").kind == SubGoalKind.INTEGRATION_MERGE  # type: ignore[union-attr]

            sg_kill_target = _make_subgoal(goal, "sg-doomed", priority_weight=2.0)
            db2.save_subgoal(sg_kill_target)
            k_res = intervene(
                arr_home,
                scope="subgoal",
                action="kill",
                goal_id=goal.goal_id,
                target_id="sg-doomed",
                payload={"reallocate_to": "sg-merged"},
                db=db2,
            )
            assert k_res["result"]["status"] == "KILLED"
            assert "sg-merged" in k_res["result"]["reallocated_to"]
            assert db2.get_subgoal("sg-merged").priority_weight == pytest.approx(3.5)  # type: ignore[union-attr]

            # 7. Overview (<50ms), inspect_state(goal_id), and build_takeover_brief (<3KB) include subgoals_summary
            ov = get_overview(arr_home, db=db2, record_ping=False)
            assert ov["interactive_latency_ms"] < 50.0
            assert "subgoals_summary" in ov
            assert ov["subgoals_summary"]["counts"]["total"] == 4
            for item in ov["subgoals_summary"]["subgoals"]:
                for req_k in (
                    "subgoal_id",
                    "title",
                    "kind",
                    "status",
                    "local_metric",
                    "target_value",
                    "sealed_sha",
                    "priority_weight",
                ):
                    assert req_k in item

            insp_g = inspect_state(arr_home, goal_id=goal.goal_id, db=db2, record_ping=False)
            assert insp_g["interactive_latency_ms"] < 50.0
            assert insp_g["subgoals_summary"]["counts"]["sealed"] == 2

            brief = build_takeover_brief(arr_home, goal_id=goal.goal_id, db=db2, record_ping=False)
            assert brief["under_3kb"] is True
            assert brief["brief_bytes"] < 3072
            assert brief["subgoals_summary"]["counts"]["total"] == 4
            assert "SubGoals" in brief["brief_markdown"]
        finally:
            db2.close()

        assert any(w.get("scope") == "subgoal" for w in received_wakes)
    finally:
        stop_srv.set()
        t_srv.join(timeout=2.0)
        srv.close()


def test_overview_sql_fast_path_multi_goal_scaling_and_latency(tmp_path: Path) -> None:
    arr_home = tmp_path / "state_home"
    repo = tmp_path / "repo"
    base_sha = _init_repo(repo)
    now = time.time()
    with StateDB(arr_home) as db:
        g_arch = _make_goal("g-arch", repo, base_sha)
        g_pause = _make_goal("g-pause", repo, base_sha)
        g_act = _make_goal("g-act", repo, base_sha)
        db.save_goal(g_arch, status="ARCHIVED")
        db.save_goal(g_pause, status="PAUSED")
        db.save_goal(g_act, status="ACTIVE")

        sg_act = _make_subgoal(g_act, "sg-act", local_metric="score", target_value=0.9)
        db.save_subgoal(sg_act)

        h_arch_0 = _make_hyp(
            g_arch,
            "h-arch-0",
            status="REFUTED",
            scientific_verdict=ScientificVerdict.REFUTED,
        )
        h_arch_def = HypothesisNode(
            hypothesis_id="h-arch-def",
            goal_id=g_arch.goal_id,
            parent_id=None,
            kind=HypothesisKind.COMPETING_ROUTE,
            title="Defective archived hyp",
            statement="test",
            mechanism="m_def",
            fidelity=FidelityRung.F1_FAST_SCREEN,
            priority_evoi=1.0,
            writable_paths=["src/mod.py"],
            params={},
            kill_criteria=["score < 0.1"],
            acceptance_cmd="pytest -q",
            screen_cmd="pytest -q",
            confirm_cmd="pytest -q",
            branch_name="conjectra/p/h-arch-def",
            base_sha=base_sha,
            head_sha=base_sha,
            status="NEEDS_REPAIR",
            failure_domain=FailureDomain.ENGINEERING_DEFECT,
            fingerprint="fp_arch_def",
        )
        db.register_hypothesis(h_arch_0)
        db.register_hypothesis(h_arch_def)

        h_pause_blk = _make_hyp(g_pause, "h-pause-blk", status="BLOCKED_BUDGET")
        db.register_hypothesis(h_pause_blk)

        h_act = _make_hyp(g_act, "h-act", subgoal_id="sg-act", status="RUNNING")
        db.register_hypothesis(h_act)

        for i in range(120):
            t_a = _make_stage_task(g_arch, h_arch_0, StageKind.EXP_SCREEN, attempt=i + 1)
            t_a.status = "DONE"
            db.enqueue_task(t_a)
            db.record_receipt(
                _make_receipt(
                    g_arch,
                    h_arch_0,
                    t_a,
                    stage=StageKind.EXP_SCREEN,
                    in_tokens=10,
                    out_tokens=5,
                )
            )
            if i < 15:
                db.save_workflow(
                    WorkflowNode(
                        workflow_id=f"wf_arch_{i}",
                        goal_id=g_arch.goal_id,
                        hypothesis_id=h_arch_0.hypothesis_id,
                        parent_workflow_id=None,
                        primitive="direct_hq",
                        status="COMPLETED",
                        goal_version=1,
                        human_epoch=0,
                        plan_epoch=0,
                        budget_tokens=10000,
                        config={},
                        state={"step": "completed"},
                        created_ts=now,
                        updated_ts=now,
                    )
                )

        for i in range(60):
            t_p = _make_stage_task(g_pause, h_pause_blk, StageKind.CPU_GATE, attempt=i + 1)
            t_p.status = "FAILED" if i < 2 else "READY"
            db.enqueue_task(t_p)
            db.record_receipt(
                _make_receipt(
                    g_pause,
                    h_pause_blk,
                    t_p,
                    stage=StageKind.CPU_GATE,
                    in_tokens=4,
                    out_tokens=2,
                )
            )

        t_act_run = _make_stage_task(g_act, h_act, StageKind.CODE, attempt=1)
        t_act_run.status = "DISPATCHED"
        db.enqueue_task(t_act_run)
        t_act_rdy = _make_stage_task(g_act, h_act, StageKind.CPU_GATE, attempt=2)
        t_act_rdy.status = "READY"
        db.enqueue_task(t_act_rdy)
        t_act_blk = _make_stage_task(g_act, h_act, StageKind.DESIGN_REVIEW, attempt=3)
        t_act_blk.status = "BLOCKED_QUORUM"
        db.enqueue_task(t_act_blk)
        for i in range(20):
            t_done = _make_stage_task(g_act, h_act, StageKind.EXP_SCREEN, attempt=10 + i)
            t_done.status = "DONE"
            db.enqueue_task(t_done)
            db.record_receipt(
                _make_receipt(
                    g_act,
                    h_act,
                    t_done,
                    stage=StageKind.EXP_SCREEN,
                    in_tokens=50,
                    out_tokens=25,
                )
            )

        with db._tx() as (conn, pending):
            db._record_event(
                conn,
                pending,
                "OWNER_ESCALATION",
                goal_id=g_act.goal_id,
                payload={"gate": "PAID_COMPUTE", "reason": "budget_check"},
            )

        ov = get_overview(arr_home, db=db, record_ping=False)
        assert ov["interactive_latency_ms"] < 50.0
        assert ov["counts"]["goals_total"] == 3
        assert ov["counts"]["goals_active"] == 1
        assert ov["counts"]["goals_paused"] == 1
        assert ov["counts"]["workflows_total"] == 15
        assert ov["counts"]["tasks_active"] == 1
        assert ov["counts"]["tasks_ready"] == 59  # 58 on g-pause + 1 on g-act
        assert ov["global_budget"]["total_tokens_used"] == (120 * 15 + 60 * 6 + 20 * 75)
        assert list(ov["per_goal_empirical_efficiency"].keys()) == [g_act.goal_id]

        by_gid = {row["goal_id"]: row for row in ov["goals"]}
        assert by_gid["g-arch"]["failure_domains"] == {FailureDomain.ENGINEERING_DEFECT.value: 1}
        assert "BLOCKED_BUDGET" in by_gid["g-pause"]["bottlenecks"]
        assert "FAILED_TASKS:2" in by_gid["g-pause"]["bottlenecks"]
        assert "BLOCKED_QUORUM" in by_gid["g-act"]["bottlenecks"]
        assert "OWNER_ESCALATION:PAID_COMPUTE" in by_gid["g-act"]["bottlenecks"]
