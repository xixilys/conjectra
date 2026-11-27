"""Tests for Multi-Goal elastic scheduler, streaming reactor, HITL intervene, <3KB takeover, and <50ms CLI."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import pytest

from conjectra import cli
from conjectra.arbiter import (
    ConflictManager,
    GlobalResourceScheduler,
    ResourceArbiter,
    clear_foreground_pings,
    read_active_foreground_pings,
    record_foreground_ping,
)
from conjectra.controller import ResearchController, _make_stage_task
from conjectra.interactive import (
    build_takeover_brief,
    get_overview,
    get_token_audit,
    inspect_state,
    intervene,
)
from conjectra.stage_runner import SandboxManager, StageRunner
from conjectra.state_db import StateDB
from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ResourceSpec,
    ScientificVerdict,
    StageKind,
    StageTask,
)


def _git(cwd: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _create_repo(repo_dir: Path) -> str:
    repo_dir.mkdir(parents=True, exist_ok=True)
    _git(repo_dir, "init", "-b", "main")
    _git(repo_dir, "config", "user.name", "tester")
    _git(repo_dir, "config", "user.email", "tester@conjectra.local")
    (repo_dir / "src").mkdir(exist_ok=True)
    (repo_dir / "tests").mkdir(exist_ok=True)
    (repo_dir / "pyproject.toml").write_text('[tool.pytest.ini_options]\npythonpath = ["."]\n', encoding="utf-8")
    (repo_dir / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repo_dir / "src" / "a.py").write_text("def val_a() -> int:\n    return 1\n", encoding="utf-8")
    (repo_dir / "src" / "b.py").write_text("def val_b() -> int:\n    return 2\n", encoding="utf-8")
    (repo_dir / "tests" / "test_all.py").write_text(
        "from src.a import val_a\nfrom src.b import val_b\n\n"
        "def test_vals():\n"
        "    assert val_a() == 1 and val_b() == 2\n"
        "    print('METRIC score=0.90 denom=1')\n",
        encoding="utf-8",
    )
    _git(repo_dir, "add", "-A")
    _git(repo_dir, "commit", "-m", "init")
    return _git(repo_dir, "rev-parse", "HEAD")


def _make_goal(
    goal_id: str,
    repo_path: Path,
    base_sha: str,
    *,
    project: str = "mproj",
    priority_weight: float = 1.0,
    min_guaranteed_slots: int = 1,
    max_concurrent_branches: int = 3,
    autonomy_mode: str = "unattended",
    writable_prefixes: list[str] | None = None,
) -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project=project,
        title=f"Goal {goal_id}",
        direction="maximize",
        primary_metric="score",
        baseline_value=0.5,
        baseline_sha=base_sha,
        repo_path=str(repo_path),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["lockbox/"],
        writable_prefixes=writable_prefixes or ["src"],
        max_tokens_budget=100_000,
        max_concurrent_branches=max_concurrent_branches,
        adapter_name="generic",
        priority_weight=priority_weight,
        min_guaranteed_slots=min_guaranteed_slots,
        autonomy_mode=autonomy_mode,
        target_value=0.95,
    )


def _make_hyp(
    goal: ResearchGoal,
    hid: str,
    *,
    writable_paths: list[str] | None = None,
    priority_evoi: float = 1.0,
    kind: HypothesisKind = HypothesisKind.PARAMETER_RACE,
    screen_cmd: str = "python3 -m pytest -s -q",
) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id=hid,
        goal_id=goal.goal_id,
        parent_id=None,
        kind=kind,
        title=f"Hyp {hid}",
        statement=f"Improve score for {goal.goal_id}",
        mechanism=f"mech_{hid}",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=priority_evoi,
        writable_paths=writable_paths or ["src/a.py"],
        params={"seed": hid},
        kill_criteria=["tests_passed == 0"],
        acceptance_cmd="python3 -m pytest -s -q",
        screen_cmd=screen_cmd,
        confirm_cmd=screen_cmd,
        branch_name=f"arr/{goal.project}/{goal.goal_id}-{hid}",
        base_sha=goal.baseline_sha,
        fingerprint=f"fp-{goal.goal_id}-{hid}",
    )


def test_elastic_borrowing_aging_decay_foreground_reserve_and_repo_scope_conflicts(
    tmp_path: Path,
) -> None:
    arr_home = tmp_path / "arr_home"
    quotas = tmp_path / "quotas.json"
    quotas.write_text(
        json.dumps(
            {
                "claude/acc1": {
                    "quota": {"plan": "max", "windows": [{"name": "Claude 7 day", "used": 88.0}]}
                }
            }
        ),
        encoding="utf-8",
    )
    arbiter = ResourceArbiter(
        quotas_path=quotas,
        ledger_path=tmp_path / "missing_l.json",
        gemini_quota_path=tmp_path / "missing_v.json",
        disk_mounts=[str(tmp_path)],
        min_ram_floor_mb=256,
        max_load1=128.0,
    )
    sched = GlobalResourceScheduler(
        arbiter,
        arr_home=arr_home,
        foreground_reserve_slots=1,
        foreground_reserve_quota_pct=15.0,
    )

    repo = tmp_path / "shared_repo"
    base_sha = _create_repo(repo)
    g_a = _make_goal("g-a", repo, base_sha, priority_weight=2.0, min_guaranteed_slots=1, max_concurrent_branches=3)
    g_b = _make_goal("g-b", repo, base_sha, priority_weight=1.0, min_guaranteed_slots=1, max_concurrent_branches=2)
    g_c = _make_goal("g-c", repo, base_sha, priority_weight=1.0, min_guaranteed_slots=1, max_concurrent_branches=2)

    class _View:
        def __init__(self, g: ResearchGoal, status: str) -> None:
            self.goal_id = g.goal_id
            self.status = status
            self.priority_weight = g.priority_weight
            self.min_guaranteed_slots = g.min_guaranteed_slots
            self.max_concurrent_branches = g.max_concurrent_branches

    # g-b is PAUSED -> g-a elastically borrows g-b's unused slots up to max_concurrent_branches
    allocs = sched.compute_goal_slot_allocations(
        [_View(g_a, "ACTIVE"), _View(g_b, "PAUSED"), _View(g_c, "ACTIVE")],
        active_by_goal={"g-a": 2, "g-b": 0, "g-c": 1},
        ready_by_goal={"g-a": 3, "g-b": 2, "g-c": 0},
        total_slots=4,
    )
    assert allocs["g-b"]["allocated_slots"] == 0
    assert allocs["g-a"]["allocated_slots"] == 3  # 1 guaranteed + 2 borrowed
    assert allocs["g-a"]["borrowed_slots"] == 1
    assert allocs["g-c"]["allocated_slots"] == 1

    # Aging bonus prevents starvation of older ready tasks even with lower EVOI
    h_fast = _make_hyp(g_a, "h-fast", priority_evoi=1.2)
    h_starved = _make_hyp(g_c, "h-starved", priority_evoi=0.8)
    now_ts = time.time()
    t_new = StageTask(
        task_id="t-new",
        idempotency_key="k-new",
        goal_id="g-a",
        hypothesis_id=h_fast.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        enqueued_ts=now_ts,
    )
    t_old = StageTask(
        task_id="t-old",
        idempotency_key="k-old",
        goal_id="g-c",
        hypothesis_id=h_starved.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        enqueued_ts=now_ts - 900.0,  # waited 15 minutes
    )
    ranked = sched.rank_ready_tasks_globally(
        [t_new, t_old],
        {"g-a": g_a, "g-c": g_c},
        {h_fast.hypothesis_id: h_fast, h_starved.hypothesis_id: h_starved},
        active_by_goal={"g-a": 1, "g-c": 1},
        recent_receipts_by_goal={"g-a": 6, "g-c": 0},  # g-a also has diminishing-returns decay
        total_slots=4,
    )
    assert ranked[0]["task"].task_id == "t-old"
    assert ranked[0]["aging_bonus"] > 0.0
    assert ranked[1]["decay"] < ranked[0]["decay"]

    # Foreground interactive reserve: ping reserves 1 slot + 15% quota headroom
    clear_foreground_pings(arr_home)
    assert sched.effective_background_slot_cap(4, arr_home=arr_home) == (4, 0)
    record_foreground_ping(arr_home, "claude", ttl_s=120.0)
    assert len(read_active_foreground_pings(arr_home)) == 1
    assert sched.effective_background_slot_cap(4, arr_home=arr_home) == (3, 1)
    q_ok, q_reason = sched.check_foreground_quota_reserve("claude", arr_home=arr_home)
    assert q_ok is False
    assert "FOREGROUND_QUOTA_RESERVE" in q_reason

    # Unified ConflictManager: cross-Goal shared-repo mutex leases detect prefix overlap
    db = StateDB(arr_home)
    try:
        db.save_goal(g_a)
        db.save_goal(g_b)
        acq = ConflictManager.acquire_repo_scope_leases(
            db,
            task_id="tsk-g1",
            goal_id="g-a",
            repo_path=repo,
            writable_paths=["src/"],
            holder="test",
            ttl_s=60.0,
        )
        assert acq is not None and acq[0]["resource_key"].startswith("repo_scope:")
        # Overlapping child path on g-b sharing same origin repo must be blocked
        acq_overlap = ConflictManager.acquire_repo_scope_leases(
            db,
            task_id="tsk-g2",
            goal_id="g-b",
            repo_path=repo,
            writable_paths=["src/a.py"],
            holder="test",
            ttl_s=60.0,
        )
        assert acq_overlap is None
        ConflictManager.release_task_leases(db, "tsk-g1")
        acq_after = ConflictManager.acquire_repo_scope_leases(
            db,
            task_id="tsk-g2",
            goal_id="g-b",
            repo_path=repo,
            writable_paths=["src/a.py"],
            holder="test",
            ttl_s=60.0,
        )
        assert acq_after is not None
    finally:
        db.close()


def test_multigoal_independent_progression_pause_borrowing_and_streaming_refill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arr_home = tmp_path / "arr_home"
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(tmp_path / "sb"))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(tmp_path / "wt"))
    repo_shared = tmp_path / "shared_repo"
    base_sha = _create_repo(repo_shared)

    db = StateDB(arr_home)
    g1 = _make_goal("g-1", repo_shared, base_sha, project="shared", priority_weight=2.0, max_concurrent_branches=2)
    g2 = _make_goal("g-2", repo_shared, base_sha, project="shared", priority_weight=1.0, autonomy_mode="paused")
    g3 = _make_goal("g-3", repo_shared, base_sha, project="shared", priority_weight=1.5, max_concurrent_branches=2)
    for g in (g1, g2, g3):
        db.save_goal(g, status="ACTIVE")
    db.set_global_config("total_slots", 4)

    # Disjoint paths on the SAME shared origin repo run concurrently without .git lock collisions
    h1 = _make_hyp(g1, "h1", writable_paths=["src/a.py"])
    h2 = _make_hyp(g2, "h2", writable_paths=["src/b.py"])
    h3 = _make_hyp(g3, "h3", writable_paths=["src/b.py"])
    for h in (h1, h2, h3):
        db.register_hypothesis(h)
        db.enqueue_task(_make_stage_task(g1 if h is h1 else (g2 if h is h2 else g3), h, StageKind.CODE))

    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(
        arr_home,
        db=db,
        arbiter=arbiter,
        execution_mode="hq",
        propose_mode="seeds",
        dry_run_llm=True,
        sweep_interval_s=15.0,
        use_workflows=True,
    )

    t_daemon = threading.Thread(
        target=lambda: ctrl.serve_forever(max_seconds=40.0),
        daemon=True,
    )
    t_daemon.start()
    try:
        deadline = time.monotonic() + 35.0
        g1_g3_progressed = False
        while time.monotonic() < deadline:
            r1 = db.list_receipts(goal_id="g-1")
            r2 = db.list_receipts(goal_id="g-2")
            r3 = db.list_receipts(goal_id="g-3")
            # g-2 (autonomy_mode='paused') must stay idle (0 receipts), while g-1 and g-3 advance through EXP_SCREEN!
            assert len(r2) == 0
            if (
                any(r.stage == StageKind.EXP_SCREEN for r in r1)
                and any(r.stage == StageKind.EXP_SCREEN for r in r3)
            ):
                g1_g3_progressed = True
                break
            time.sleep(0.2)
        assert g1_g3_progressed, "Sibling Goals g-1 and g-3 did not stream independently while g-2 was paused"
    finally:
        ctrl.stop()
        t_daemon.join(timeout=5.0)
        try:
            ctrl.hq.stop()
        except Exception:
            pass
        ctrl.close()


def test_rebalance_update_objective_epoch_isolation_takeover_and_sub50ms_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arr_home = tmp_path / "arr_home"
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(tmp_path / "sb"))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(tmp_path / "wt"))
    repo = tmp_path / "repo"
    base_sha = _create_repo(repo)

    db = StateDB(arr_home)
    g1 = _make_goal("g-alpha", repo, base_sha, project="alpha", priority_weight=1.0)
    g2 = _make_goal("g-beta", repo, base_sha, project="beta", priority_weight=1.0)
    db.save_goal(g1, status="ACTIVE")
    db.save_goal(g2, status="ACTIVE")

    h1 = _make_hyp(g1, "h-a1")
    h2 = _make_hyp(g2, "h-b1")
    db.register_hypothesis(h1)
    db.register_hypothesis(h2)
    db.upsert_knowledge_item(
        "g-alpha",
        "confirmed_finding",
        "k1",
        {"metric": "score", "value": 0.90},
        hypothesis_id=h1.hypothesis_id,
        goal_version=1,
    )

    # Enqueue a task at v1, then bump objective to v2 via intervene -> v1 receipt must be STALE_GOAL_VERSION
    old_task = _make_stage_task(g1, h1, StageKind.EXP_SCREEN)
    old_task.status = "DONE"
    db.enqueue_task(old_task)
    db.record_receipt(
        EvidenceReceipt(
            receipt_id="rcpt-stale-v1",
            task_id=old_task.task_id,
            goal_id=g1.goal_id,
            hypothesis_id=h1.hypothesis_id,
            stage=StageKind.EXP_SCREEN,
            base_sha=base_sha,
            head_sha=base_sha,
            command="pytest -q",
            exit_code=0,
            wall_sec=0.5,
            max_rss_mb=32.0,
            in_tokens=100,
            out_tokens=50,
            model_used="gemini-flash",
            process_status=ProcessStatus.EXIT_OK,
            engineering_verdict=EngineeringVerdict.PASS,
            scientific_verdict=ScientificVerdict.CONFIRMED,
            audit_verdict=AuditVerdict.NOT_REQUIRED,
            tests_passed=1,
            tests_total=1,
            metrics={"score": {"value": 0.85, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"}},
            raw_artifact_paths=["/tmp/log.txt"],
            goal_version=1,
            evidence_validity=EvidenceValidity.VALID_EMPIRICAL,
        )
    )

    # 1. Human dynamic rebalance_goals + update_objective via interactive.intervene & CLI
    rebal = intervene(
        arr_home,
        scope="global",
        action="rebalance_goals",
        payload={"weights": {"g-alpha": 4.0, "g-beta": 0.5}, "min_slots": {"g-alpha": 2}},
        db=db,
        wake_reactor=False,
    )
    assert rebal["interactive_latency_ms"] < 100.0
    assert db.get_goal("g-alpha").priority_weight == 4.0
    assert db.get_goal("g-alpha").min_guaranteed_slots == 2

    upd = intervene(
        arr_home,
        scope="goal",
        action="update_objective",
        goal_id="g-alpha",
        payload={"baseline_value": 0.92, "target_value": 0.98},
        db=db,
        wake_reactor=False,
    )
    assert upd["result"]["version"] == 2

    # Controller step isolates stale v1 receipt cleanly without promoting h1 under v2!
    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(arr_home, db=db, arbiter=arbiter, execution_mode="inline", dry_run_llm=True)
    step_sum = ctrl.step(goal_id="g-alpha")
    assert step_sum["stale_isolated"] >= 1
    assert db.get_hypothesis("h-a1").status != "COMPLETED"
    db.close()

    # 2. Sub-50ms WAL read queries + <3KB session-independent takeover brief
    clear_foreground_pings(arr_home)
    ov = get_overview(arr_home)
    assert ov["interactive_latency_ms"] < 50.0
    assert ov["counts"]["goals_total"] == 2
    assert len(read_active_foreground_pings(arr_home)) >= 1

    insp = inspect_state(arr_home, goal_id="g-alpha")
    assert insp["interactive_latency_ms"] < 50.0
    assert insp["goal"]["version"] == 2

    ta = get_token_audit(arr_home, goal_id="g-alpha")
    assert ta["interactive_latency_ms"] < 50.0
    assert ta["uncached_total_tokens"] == 150

    to = build_takeover_brief(arr_home)
    assert to["interactive_latency_ms"] < 50.0
    assert to["under_3kb"] is True
    assert to["brief_bytes"] < 3072
    assert "g-alpha" in to["brief_markdown"] and "g-beta" in to["brief_markdown"]

    # 3. CLI commands including --detach --notify-cmd
    assert cli.main(["--arr-home", str(arr_home), "overview"]) == 0
    assert json.loads(capsys.readouterr().out)["counts"]["goals_total"] == 2

    assert cli.main(["--arr-home", str(arr_home), "takeover", "--goal-id", "g-alpha", "--format", "markdown"]) == 0
    md_out = capsys.readouterr().out
    assert len(md_out.encode("utf-8")) < 3072

    notify_log = tmp_path / "notify.jsonl"
    assert (
        cli.main(
            [
                "--arr-home",
                str(arr_home),
                "daemon",
                "--mode",
                "inline",
                "--dry-run-llm",
                "--max-seconds",
                "0.1",
                "--detach",
                "--notify-cmd",
                str(notify_log),
            ]
        )
        == 0
    )
    det = json.loads(capsys.readouterr().out)
    assert det["detached"] is True
    deadline = time.monotonic() + 5.0
    while not notify_log.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert notify_log.is_file()
    assert "conjectra daemon finished" in notify_log.read_text(encoding="utf-8")


def test_stage_runner_blind_audit_uses_structured_verdict_and_propagates_metadata(
    tmp_path: Path,
) -> None:
    arr_home = tmp_path / "arr_home"
    repo = tmp_path / "repo"
    base_sha = _create_repo(repo)
    fake_mp = tmp_path / "fake_mp.py"
    fake_mp.write_text(
        """#!/usr/bin/env python3
import argparse, json
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("pool")
ap.add_argument("--host", default="local")
ap.add_argument("--model", default="m")
ap.add_argument("--cwd", default=".")
ap.add_argument("--out-md", default=None)
ap.add_argument("--usage-out", default=None)
ap.add_argument("--timeout", default="60")
ap.add_argument("--tag", default="t")
ap.add_argument("--readonly", action="store_true")
args = ap.parse_args()
Path(args.out_md).write_text("AUDIT_VERDICT: PASS; AUDIT_VERDICT: FAIL\\n", encoding="utf-8")
Path(args.usage_out).write_text(json.dumps({"model": "codex-test", "usage": {"input_tokens": 10, "output_tokens": 5}}), encoding="utf-8")
""",
        encoding="utf-8",
    )
    fake_mp.chmod(0o755)

    goal = _make_goal("g-aud", repo, base_sha)
    goal.version = 3
    hyp = _make_hyp(goal, "h-aud")
    task = StageTask(
        task_id="tsk-aud",
        idempotency_key="k-aud",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.BLIND_AUDIT,
        deps=[],
        resources=ResourceSpec(llm_pool="codex:gpt-5-codex"),
        workflow_id="wf_aud_1",
        goal_version=3,
        workflow_primitive="direct_hq",
        step_key="blind_audit",
    )
    sb = SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt")
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb, multipool_script=fake_mp)
    rcpt = runner.run_stage(task, hyp, goal, persist=False, notify=False, dry_run_llm=False)
    # Contradictory verdict string must NEVER yield AuditVerdict.PASS
    assert rcpt.audit_verdict in (AuditVerdict.PARSE_ERROR, AuditVerdict.FAIL)
    assert rcpt.goal_version == 3
    assert rcpt.workflow_id == "wf_aud_1"
    assert rcpt.evidence_origin == "real_research_goal"


def test_no_batch_barrier_and_zero_filler_idle(tmp_path: Path) -> None:
    arr_home = tmp_path / "arr_home"
    repo = tmp_path / "repo"
    base_sha = _create_repo(repo)
    db = StateDB(arr_home)
    goal = _make_goal("g-nobarrier", repo, base_sha, max_concurrent_branches=3)
    db.save_goal(goal, status="ACTIVE")

    h_running = _make_hyp(goal, "h-run")
    db.register_hypothesis(h_running)
    # One task is actively RUNNING (batch barrier would have blocked here!),
    # while one receipt already landed and ready queue < max_concurrent_branches.
    t_run = _make_stage_task(goal, h_running, StageKind.EXP_SCREEN)
    t_run.status = "RUNNING"
    db.enqueue_task(t_run)

    t_done = _make_stage_task(goal, h_running, StageKind.CPU_GATE)
    t_done.status = "DONE"
    db.enqueue_task(t_done)
    db.record_receipt(
        EvidenceReceipt(
            receipt_id="rcpt-new-ev",
            task_id=t_done.task_id,
            goal_id=goal.goal_id,
            hypothesis_id=h_running.hypothesis_id,
            stage=StageKind.CPU_GATE,
            base_sha=base_sha,
            head_sha=base_sha,
            command="pytest -q",
            exit_code=0,
            wall_sec=0.1,
            max_rss_mb=16.0,
            in_tokens=0,
            out_tokens=0,
            model_used="deterministic",
            process_status=ProcessStatus.EXIT_OK,
            engineering_verdict=EngineeringVerdict.PASS,
            scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
            audit_verdict=AuditVerdict.NOT_REQUIRED,
            tests_passed=1,
            tests_total=1,
            metrics={},
            raw_artifact_paths=["/tmp/gate.log"],
        )
    )

    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(
        arr_home,
        db=db,
        arbiter=arbiter,
        execution_mode="hq",
        propose_mode="auto",
        dry_run_llm=False,
        decision_coalesce_window_s=60.0,
    )
    ctrl._handled_tasks.add(t_done.task_id)
    summary: dict[str, Any] = {
        "decisions_enqueued": 0,
        "proposed": 0,
        "tasks_enqueued": 0,
    }
    # Even with an active RUNNING task, non-blocking DecisionRequest is enqueued (0 batch barrier!)
    ctrl._maybe_propose_round(goal, summary)
    assert summary["decisions_enqueued"] == 1
    dreqs = db.list_decision_requests(goal_id=goal.goal_id, status="PENDING")
    assert len(dreqs) == 1

    # Idle gracefully at 0 LLM tokens when no positive-EVOI work remains (0 filler tasks)
    goal_idle = _make_goal("g-idle", repo, base_sha)
    db.save_goal(goal_idle, status="ACTIVE")
    h_done = _make_hyp(goal_idle, "h-done")
    h_done.status = "COMPLETED"
    h_done.scientific_verdict = ScientificVerdict.CONFIRMED
    db.register_hypothesis(h_done)
    ctrl._goal_ctx(goal_idle.goal_id).last_propose_receipt_count = 0
    s_idle = ctrl.step(goal_id=goal_idle.goal_id)
    assert s_idle["proposed"] == 0
    assert s_idle["decisions_enqueued"] == 0
    assert s_idle["tasks_enqueued"] == 0
    db.close()
