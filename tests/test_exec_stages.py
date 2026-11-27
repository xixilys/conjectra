"""Tests for stage_runner.py and PROJALPHA / ProjBeta project adapters."""

from __future__ import annotations

import json
import os
import socket
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from conjectra.stage_runner import (
    DEFAULT_MULTIPOOL_SCRIPT,
    FALLBACK_SANDBOX_BASE,
    PRIMARY_SANDBOX_BASE,
    SCRATCH_ROOT,
    SandboxError,
    SandboxManager,
    StageRunner,
    _build_multipool_cmd,
    extract_cache_tokens,
    extract_token_breakdown,
    extract_token_usage,
    wait_for_mem_available,
)
from conjectra.state_db import StateDB
from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
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


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_sample_repo(repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "tester")
    _git(repo, "config", "user.email", "tester@example.com")
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    (repo / "heavy_data").mkdir()
    (repo / "src" / "algo.py").write_text("def score(x: int) -> int:\n    return x + 1\n", encoding="utf-8")
    (repo / "src" / "test_algo.py").write_text(
        "from src.algo import score\n\ndef test_score():\n    assert score(2) >= 3\n",
        encoding="utf-8",
    )
    (repo / "docs" / "notes.md").write_text("# Notes\n", encoding="utf-8")
    (repo / "heavy_data" / "blob.bin").write_text("big payload\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")
    return _git(repo, "rev-parse", "HEAD")


def _snapshot_git_dir(repo: Path) -> dict[str, object]:
    git_dir = repo / ".git"
    refs = _git(repo, "show-ref")
    objects = sorted(str(p.relative_to(git_dir)) for p in (git_dir / "objects").rglob("*") if p.is_file())
    head = (git_dir / "HEAD").read_text(encoding="utf-8")
    return {"refs": refs, "objects": objects, "head": head}


def test_shared_clone_isolation_and_sparse_worktree_lifecycle(tmp_path: Path):
    origin = tmp_path / "origin_repo"
    base_sha = _init_sample_repo(origin)
    before = _snapshot_git_dir(origin)

    sb = SandboxManager(
        sandbox_base=tmp_path / "sandboxes",
        worktree_base=tmp_path / "worktrees",
    )
    mirror = sb.ensure_mirror(origin, "demo")
    assert (mirror / ".git" / "objects" / "info" / "alternates").is_file()

    # Protected branch must be rejected.
    with pytest.raises(SandboxError):
        sb.create_worktree(mirror, "demo", "main", base_sha)

    wt = sb.create_worktree(
        mirror,
        "demo",
        "arr/demo/h1",
        base_sha,
        origin_repo=origin,
        sparse_paths=["src/"],
    )
    assert (wt / "src" / "algo.py").is_file()
    assert not (wt / "heavy_data" / "blob.bin").exists()

    # Commit inside isolated worktree and verify origin .git is untouched.
    (wt / "src" / "algo.py").write_text("def score(x: int) -> int:\n    return x + 2\n", encoding="utf-8")
    _git(wt, "config", "user.name", "arr")
    _git(wt, "config", "user.email", "arr@local")
    _git(wt, "add", "src/algo.py")
    _git(wt, "commit", "-m", "isolated branch change")

    after = _snapshot_git_dir(origin)
    assert after == before
    assert _git(origin, "branch", "--list").strip() == "* main"

    # Cleanup removes worktree and mirror branch without touching origin.
    assert sb.cleanup_worktree(mirror, wt, branch_name="arr/demo/h1", delete_branch=True)
    assert not wt.exists()
    assert "arr/demo/h1" not in _git(mirror, "branch", "--list")
    assert _snapshot_git_dir(origin) == before


def test_sandbox_disk_high_water_fallback(monkeypatch: pytest.MonkeyPatch):
    # Exercise the built-in default roots, not the per-session overrides set in conftest.
    monkeypatch.delenv("CONJECTRA_SANDBOX_ROOT", raising=False)
    monkeypatch.delenv("CONJECTRA_WORKTREE_ROOT", raising=False)
    sb_full = SandboxManager(disk_used_pct_fn=lambda _p: 86.0)
    sb_root, wt_root = sb_full.resolve_roots()
    assert sb_root == FALLBACK_SANDBOX_BASE
    assert "arr-wt" in str(wt_root)

    if SCRATCH_ROOT.exists():
        sb_ok = SandboxManager(disk_used_pct_fn=lambda _p: 40.0)
        sb_root2, _ = sb_ok.resolve_roots()
        assert sb_root2 == PRIMARY_SANDBOX_BASE


def test_extract_token_usage_all_pool_formats():
    gemini_doc = {
        "route": "gemini_cli",
        "model": "gemini-flash",
        "usage": {
            "input_tokens": 165406,
            "output_tokens": 62966,
            "cache_read_tokens": 1768306,
            "total_tokens": 228372,
        },
    }
    in_t, out_t, m = extract_token_usage(gemini_doc)
    assert (in_t, out_t, m) == (165406, 62966, "gemini-flash")
    assert extract_cache_tokens(gemini_doc) == 1768306

    claude_doc = {
        "route": "claude_cli_messages",
        "model": "claude-opus-5-5",
        "usage": {
            "input_tokens": 500,
            "cache_read_input_tokens": 200,
            "cache_creation_input_tokens": 50,
            "output_tokens": 110,
        },
    }
    in_c, out_c, mc = extract_token_usage(claude_doc)
    assert (in_c, out_c, mc) == (500, 110, "claude-opus-5-5")
    assert extract_cache_tokens(claude_doc) == 200
    assert extract_token_breakdown(claude_doc) == (500, 110, 200, 50, "claude-opus-5-5")

    codex_doc = {
        "route": "codex_cli",
        "model": "gpt-5-codex",
        "usage": {"input_tokens": 1200, "cached_input_tokens": 900, "output_tokens": 75},
    }
    assert extract_token_usage(codex_doc) == (300, 75, "gpt-5-codex")
    assert extract_cache_tokens(codex_doc) == 900
    assert extract_token_breakdown(codex_doc) == (300, 75, 900, 0, "gpt-5-codex")

    dsh_doc = {
        "route": "deepseek",
        "model": "deepseek-chat",
        "usage": {"input_tokens": 300, "cache_read_tokens": 50, "output_tokens": 90},
    }
    in_d, out_d, md = extract_token_usage(dsh_doc)
    assert (in_d, out_d, md) == (300, 90, "deepseek-chat")
    assert extract_cache_tokens(dsh_doc) == 50


def test_stage_split_pipeline_mocked_and_socket_wakeup(tmp_path: Path):
    origin = tmp_path / "repo"
    base_sha = _init_sample_repo(origin)
    arr_home = tmp_path / "arr_home"
    arr_home.mkdir()

    # Fake dispatch.py handling both CODE edits and --readonly BLIND_AUDIT.
    fake_mp = tmp_path / "fake_mp.py"
    audit_capture = tmp_path / "audit_capture.json"
    fake_mp.write_text(
        f"""#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path
ap = argparse.ArgumentParser()
ap.add_argument("pool")
ap.add_argument("--host", default="local")
ap.add_argument("--model", default="m")
ap.add_argument("--cwd", default=".")
ap.add_argument("--out-md", default=None)
ap.add_argument("--usage-out", default=None)
ap.add_argument("--timeout", type=int, default=60)
ap.add_argument("--tag", default="")
ap.add_argument("--readonly", action="store_true")
args, _ = ap.parse_known_args()
prompt = sys.stdin.read()
if args.readonly:
    Path({str(audit_capture)!r}).write_text(json.dumps({{"pool": args.pool, "model": args.model, "prompt": prompt}}))
    resp = "Checked diff and raw metrics.\\nAUDIT_VERDICT: PASS\\n"
    usage = {{"route": args.pool, "model": args.model, "usage": {{"input_tokens": 410, "output_tokens": 45}}}}
else:
    algo = Path(args.cwd) / "src" / "algo.py"
    algo.write_text("def score(x: int) -> int:\\n    return x + 10\\n", encoding="utf-8")
    resp = "Updated score()."
    usage = {{"route": args.pool, "model": args.model, "usage": {{"input_tokens": 900, "cache_read_tokens": 1500, "output_tokens": 120}}}}
if args.out_md:
    Path(args.out_md).write_text(resp, encoding="utf-8")
if args.usage_out:
    Path(args.usage_out).write_text(json.dumps(usage), encoding="utf-8")
print(resp)
""",
        encoding="utf-8",
    )
    os.chmod(fake_mp, os.stat(fake_mp).st_mode | stat.S_IXUSR)

    # Fake slot_lock.py recording invocation only for gpu>0 / eda_sim>0 confirm runs.
    lock_marker = tmp_path / "slot_lock_invoked.txt"
    fake_lock = tmp_path / "fake_slot_lock.py"
    fake_lock.write_text(
        f"""#!/usr/bin/env python3
import subprocess, sys
from pathlib import Path
cmd = sys.argv[1:]
if cmd and cmd[0] == "--":
    cmd = cmd[1:]
Path({str(lock_marker)!r}).write_text(" ".join(cmd), encoding="utf-8")
raise SystemExit(subprocess.call(cmd))
""",
        encoding="utf-8",
    )
    os.chmod(fake_lock, os.stat(fake_lock).st_mode | stat.S_IXUSR)

    # Real Unix domain socket listener simulating controller reactor.sock.
    sock_path = arr_home / "reactor.sock"
    received_events: list[dict] = []
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(8)
    srv.settimeout(5.0)
    stop_srv = threading.Event()

    def _listen() -> None:
        while not stop_srv.is_set():
            try:
                conn, _ = srv.accept()
            except OSError:
                break
            with conn:
                data = conn.recv(65536).decode("utf-8", errors="replace").strip()
                if data:
                    received_events.append(json.loads(data.splitlines()[0]))

    th = threading.Thread(target=_listen, daemon=True)
    th.start()

    try:
        db = StateDB(arr_home)
        goal = ResearchGoal(
            goal_id="g-demo",
            project="demo",
            title="Improve demo score",
            direction="max",
            primary_metric="score",
            baseline_value=3.0,
            baseline_sha=base_sha,
            repo_path=str(origin),
            kill_criteria=["tests_passed == 0"],
            banned_paths=["sealed_eval/", "lockbox/"],
            writable_prefixes=["src/"],
            max_tokens_budget=50000,
            adapter_name="generic",
        )
        db.save_goal(goal)
        hyp = HypothesisNode(
            hypothesis_id="h-demo-1",
            goal_id=goal.goal_id,
            parent_id=None,
            kind=HypothesisKind.COMPETING_ROUTE,
            title="Increase score offset",
            statement="Increase offset in src/algo.py",
            mechanism="offset +10",
            fidelity=FidelityRung.F1_FAST_SCREEN,
            priority_evoi=1.5,
            writable_paths=["src/algo.py"],
            params={"offset": 10},
            kill_criteria=["tests_passed == 0"],
            acceptance_cmd=f"{sys.executable} -m pytest -q -p no:cacheprovider src/test_algo.py",
            screen_cmd=(
                f"{sys.executable} -m pytest -q -p no:cacheprovider src/test_algo.py && "
                "echo 'METRIC score=12.0 denom=1'"
            ),
            confirm_cmd=(
                f"{sys.executable} -m pytest -q -p no:cacheprovider src/test_algo.py && "
                "echo 'METRIC score=12.0 denom=5'"
            ),
            branch_name="arr/demo/h-demo-1",
            base_sha=base_sha,
        )
        db.register_hypothesis(hyp)

        sb = SandboxManager(
            sandbox_base=tmp_path / "sandboxes",
            worktree_base=tmp_path / "worktrees",
        )
        runner = StageRunner(
            arr_home=arr_home,
            sandbox_manager=sb,
            multipool_script=fake_mp,
            slot_lock_script=fake_lock,
        )

        t_code = StageTask(
            task_id="t-code",
            idempotency_key="k-code",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.CODE,
            deps=[],
            resources=ResourceSpec(llm_pool="gemini:gemini-flash", wall_timeout_s=60),
        )
        db.enqueue_task(t_code)
        r_code = runner.run_stage(t_code, hyp, goal, db=db)
        assert r_code.process_status == ProcessStatus.EXIT_OK
        assert r_code.in_tokens == 900 and r_code.out_tokens == 120
        assert r_code.metrics["cache_read_tokens"] == {
            "value": 1500,
            "denominator": 1,
            "stage_rung": "CODE",
        }
        assert r_code.head_sha != base_sha
        hyp = db.get_hypothesis(hyp.hypothesis_id)
        assert hyp.head_sha == r_code.head_sha

        t_gate = StageTask(
            task_id="t-gate",
            idempotency_key="k-gate",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.CPU_GATE,
            deps=["t-code"],
            resources=ResourceSpec(wall_timeout_s=60),
        )
        db.enqueue_task(t_gate)
        r_gate = runner.run_stage(t_gate, hyp, goal, db=db)
        assert r_gate.engineering_verdict == EngineeringVerdict.PASS
        assert r_gate.in_tokens == 0 and r_gate.out_tokens == 0
        assert r_gate.tests_passed == 1

        t_screen = StageTask(
            task_id="t-screen",
            idempotency_key="k-screen",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.EXP_SCREEN,
            deps=["t-gate"],
            resources=ResourceSpec(wall_timeout_s=60),
        )
        db.enqueue_task(t_screen)
        r_screen = runner.run_stage(t_screen, hyp, goal, db=db)
        assert r_screen.scientific_verdict == ScientificVerdict.CONFIRMED
        assert r_screen.metrics["score"]["value"] == 12.0
        assert r_screen.metrics["score"]["stage_rung"] == "F1_FAST_SCREEN"
        assert not lock_marker.exists()  # CPU screen must NOT wrap with slot_lock

        # Confirm with eda_sim=1 MUST invoke slot_lock wrapper.
        t_confirm = StageTask(
            task_id="t-confirm",
            idempotency_key="k-confirm",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.EXP_CONFIRM,
            deps=["t-screen"],
            resources=ResourceSpec(eda_sim=1, wall_timeout_s=60),
        )
        db.enqueue_task(t_confirm)
        r_confirm = runner.run_stage(t_confirm, hyp, goal, db=db)
        assert r_confirm.scientific_verdict == ScientificVerdict.CONFIRMED
        assert r_confirm.metrics["score"]["stage_rung"] == "F2_FULL_CONFIRM"
        assert lock_marker.is_file()

        # Cross-family BLIND_AUDIT: author used Gemini -> audit uses Codex.
        t_audit = StageTask(
            task_id="t-audit",
            idempotency_key="k-audit",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.BLIND_AUDIT,
            deps=["t-confirm"],
            resources=ResourceSpec(wall_timeout_s=60),
        )
        db.enqueue_task(t_audit)
        r_audit = runner.run_stage(t_audit, hyp, goal, db=db)
        assert r_audit.audit_verdict == AuditVerdict.PASS
        assert r_audit.in_tokens == 410 and r_audit.out_tokens == 45
        cap = json.loads(audit_capture.read_text(encoding="utf-8"))
        assert cap["pool"] == "codex"
        assert "Updated score()." not in cap["prompt"]  # Never author narrative

        t_rec = StageTask(
            task_id="t-rec",
            idempotency_key="k-rec",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.RECONCILE,
            deps=["t-audit"],
        )
        db.enqueue_task(t_rec)
        r_rec = runner.run_stage(t_rec, hyp, goal, db=db)
        assert r_rec.process_status == ProcessStatus.EXIT_OK
        import time as _t
        deadline = _t.monotonic() + 2.0
        while len(received_events) < 6 and _t.monotonic() < deadline:
            _t.sleep(0.02)
        assert len(received_events) >= 6
        assert received_events[-1]["stage"] == "RECONCILE"
        db.close()
    finally:
        stop_srv.set()
        srv.close()


def test_gpu_vram_guard_triggers_env_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    origin = tmp_path / "repo"
    base_sha = _init_sample_repo(origin)
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-vram",
        project="demo",
        title="VRAM check",
        direction="max",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(origin),
        kill_criteria=["tests_passed == 0"],
        banned_paths=[],
        writable_prefixes=["src/"],
        max_tokens_budget=1000,
    )
    hyp = HypothesisNode(
        hypothesis_id="h-vram",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.PARAMETER_RACE,
        title="GPU test",
        statement="GPU test",
        mechanism="gpu",
        fidelity=FidelityRung.F2_FULL_CONFIRM,
        priority_evoi=1.0,
        writable_paths=["src/"],
        params={},
        kill_criteria=["tests_passed == 0"],
        acceptance_cmd="true",
        screen_cmd="true",
        confirm_cmd="true",
        branch_name="arr/demo/h-vram",
        base_sha=base_sha,
    )
    task = StageTask(
        task_id="t-gpu",
        idempotency_key="k-gpu",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.EXP_CONFIRM,
        deps=[],
        resources=ResourceSpec(gpu=1, wall_timeout_s=30),
    )
    monkeypatch.setattr(
        "conjectra.stage_runner.check_live_gpu_vram",
        lambda min_free_mb=1500: (False, {"max_free_mb": 200, "min_required_mb": min_free_mb}),
    )
    runner = StageRunner(
        arr_home=arr_home,
        sandbox_manager=SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt"),
    )
    receipt = runner.run_stage(task, hyp, goal, db=db, notify=False)
    assert receipt.process_status == ProcessStatus.ENV_ERROR
    assert receipt.scientific_verdict == ScientificVerdict.UNKNOWN
    assert "VRAM" in receipt.failure_reason
    db.close()




def test_deterministic_parameter_race_code_skip_and_gate_rejection(tmp_path: Path):
    origin = tmp_path / "repo"
    base_sha = _init_sample_repo(origin)
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-gate",
        project="demo",
        title="Gate checks",
        direction="max",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(origin),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["sealed_eval/"],
        writable_prefixes=["src/"],
        max_tokens_budget=5000,
    )
    hyp = HypothesisNode(
        hypothesis_id="h-gate",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.PARAMETER_RACE,
        title="Param race",
        statement="Param race",
        mechanism="param",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.0,
        writable_paths=["src/"],
        params={"alpha": 0.5},
        kill_criteria=["tests_passed == 0"],
        acceptance_cmd=f"{sys.executable} -m pytest -q -p no:cacheprovider src/test_algo.py",
        screen_cmd="true",
        confirm_cmd="true",
        branch_name="arr/demo/h-gate",
        base_sha=base_sha,
    )
    sb = SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt")
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb)

    # PARAMETER_RACE with llm_pool=None skips LLM cleanly at 0 tokens.
    t_code = StageTask(
        task_id="t-pr-code",
        idempotency_key="k-pr-code",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        resources=ResourceSpec(llm_pool=None),
    )
    r_code = runner.run_stage(t_code, hyp, goal, db=db, notify=False)
    assert r_code.in_tokens == 0 and r_code.out_tokens == 0
    assert r_code.model_used == "deterministic"

    # Weakening an assertion in the worktree must be caught by CPU_GATE.
    mirror = sb.ensure_mirror(origin, "demo")
    wt = sb.create_worktree(mirror, "demo", hyp.branch_name, base_sha, origin_repo=origin)
    (wt / "src" / "test_algo.py").write_text(
        "def test_score():\n    pass\n",
        encoding="utf-8",
    )
    _git(wt, "config", "user.name", "arr")
    _git(wt, "config", "user.email", "arr@local")
    _git(wt, "add", "src/test_algo.py")
    _git(wt, "commit", "-m", "weaken assertion")

    t_gate = StageTask(
        task_id="t-pr-gate",
        idempotency_key="k-pr-gate",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CPU_GATE,
        deps=[],
    )
    r_gate = runner.run_stage(t_gate, hyp, goal, db=db, notify=False)
    assert r_gate.engineering_verdict == EngineeringVerdict.FAIL_GATE
    db.close()


def test_mem_available_wait_and_nice_n10_prefix(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # 1. wait_for_mem_available succeeds when above floor and times out when below floor.
    monkeypatch.setattr("conjectra.stage_runner.read_mem_available_mb", lambda: 12000)
    ok, avail = wait_for_mem_available(min_mb=10240, max_wait_s=0.2, poll_s=0.05)
    assert ok is True and avail == 12000

    monkeypatch.setattr("conjectra.stage_runner.read_mem_available_mb", lambda: 4096)
    ok_low, avail_low = wait_for_mem_available(min_mb=10240, max_wait_s=0.08, poll_s=0.02)
    assert ok_low is False and avail_low == 4096

    origin = tmp_path / "repo"
    base_sha = _init_sample_repo(origin)
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-mem",
        project="demo",
        title="Memory & nice guard",
        direction="max",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(origin),
        kill_criteria=["tests_passed == 0"],
        banned_paths=[],
        writable_prefixes=["src/"],
        max_tokens_budget=5000,
    )
    hyp = HypothesisNode(
        hypothesis_id="h-mem",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.PARAMETER_RACE,
        title="Mem guard hyp",
        statement="Mem guard hyp",
        mechanism="guard",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.0,
        writable_paths=["src/"],
        params={},
        kill_criteria=["tests_passed == 0"],
        acceptance_cmd=f"{sys.executable} -m pytest -q -p no:cacheprovider src/test_algo.py",
        screen_cmd="echo METRIC score=5.0 denom=1",
        confirm_cmd="echo METRIC score=5.0 denom=1",
        branch_name="arr/demo/h-mem",
        base_sha=base_sha,
    )
    runner = StageRunner(
        arr_home=arr_home,
        sandbox_manager=SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt"),
    )

    # 2. Timeout when MemAvailable < 10 GiB across CPU_GATE, EXP_SCREEN, EXP_CONFIRM.
    monkeypatch.setattr("conjectra.stage_runner.wait_for_mem_available", lambda *a, **kw: (False, 4096))
    for idx, stg in enumerate((StageKind.CPU_GATE, StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM)):
        t = StageTask(
            task_id=f"t-mem-fail-{idx}",
            idempotency_key=f"k-mem-fail-{idx}",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=stg,
            deps=[],
            resources=ResourceSpec(wall_timeout_s=30),
        )
        rcpt = runner.run_stage(t, hyp, goal, db=db, notify=False)
        assert rcpt.process_status == ProcessStatus.ENV_ERROR
        assert rcpt.engineering_verdict == EngineeringVerdict.SKIPPED
        assert rcpt.scientific_verdict == ScientificVerdict.UNKNOWN
        assert rcpt.failure_reason == "MEM_BELOW_10GB_TIMEOUT"
        assert rcpt.command.startswith("nice -n 10 ")

    # 3. When MemAvailable >= 10 GiB, commands run under nice -n 10.
    monkeypatch.setattr("conjectra.stage_runner.wait_for_mem_available", lambda *a, **kw: (True, 16384))
    t_ok = StageTask(
        task_id="t-nice-ok",
        idempotency_key="k-nice-ok",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.EXP_SCREEN,
        deps=[],
        resources=ResourceSpec(wall_timeout_s=30),
    )
    rcpt_ok = runner.run_stage(t_ok, hyp, goal, db=db, notify=False)
    assert rcpt_ok.process_status == ProcessStatus.EXIT_OK
    assert rcpt_ok.command.startswith("nice -n 10 ")
    assert rcpt_ok.scientific_verdict == ScientificVerdict.CONFIRMED
    db.close()


def test_codex_cli_direct_audit_and_dispatch_host_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):

    cmd_remote = _build_multipool_cmd(
        DEFAULT_MULTIPOOL_SCRIPT,
        "deepseek",
        "deepseek-chat",
        tmp_path,
        tmp_path / "out.md",
        tmp_path / "usage.json",
        60,
        "tag1",
        readonly=False,
        multipool_host_arg="remote",
    )
    assert cmd_remote[cmd_remote.index("--host") + 1] == "remote"

    cmd_gemini = _build_multipool_cmd(
        DEFAULT_MULTIPOOL_SCRIPT,
        "gemini",
        "gemini-flash",
        tmp_path,
        tmp_path / "out.md",
        tmp_path / "usage.json",
        60,
        "tag2",
        readonly=False,
    )
    assert cmd_gemini[cmd_gemini.index("--host") + 1] == "local"

    # Fake codex binary verifying direct `codex exec ... --json -` when no dispatcher script is configured.
    fake_codex = tmp_path / "codex"
    args_capture = tmp_path / "codex_args.json"
    fake_codex.write_text(
        f"""#!/usr/bin/env python3
import json, sys
from pathlib import Path
Path({str(args_capture)!r}).write_text(json.dumps(sys.argv[1:]))
_ = sys.stdin.read()
print(json.dumps({{"type": "thread.started", "thread_id": "t1"}}))
print(json.dumps({{"type": "item.completed", "item": {{"id": "i1", "type": "agent_message", "text": "Verified diff.\\nAUDIT_VERDICT: PASS"}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 321, "output_tokens": 27}}}}))
""",
        encoding="utf-8",
    )
    os.chmod(fake_codex, os.stat(fake_codex).st_mode | stat.S_IXUSR)
    monkeypatch.setattr("conjectra.stage_runner.shutil.which", lambda name: str(fake_codex) if name == "codex" else None)

    origin = tmp_path / "repo"
    base_sha = _init_sample_repo(origin)
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-codex",
        project="demo",
        title="Codex direct audit",
        direction="max",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(origin),
        kill_criteria=["tests_passed == 0"],
        banned_paths=[],
        writable_prefixes=["src/"],
        max_tokens_budget=5000,
    )
    hyp = HypothesisNode(
        hypothesis_id="h-codex",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Codex audit test",
        statement="Codex audit test",
        mechanism="codex",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.0,
        writable_paths=["src/"],
        params={},
        kill_criteria=["tests_passed == 0"],
        acceptance_cmd="true",
        screen_cmd="true",
        confirm_cmd="true",
        branch_name="arr/demo/h-codex",
        base_sha=base_sha,
    )
    runner = StageRunner(
        arr_home=arr_home,
        sandbox_manager=SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt"),
    )
    t_audit = StageTask(
        task_id="t-codex-audit",
        idempotency_key="k-codex-audit",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.BLIND_AUDIT,
        deps=[],
        resources=ResourceSpec(llm_pool="codex:gpt-5-codex", wall_timeout_s=60),
    )
    rcpt = runner.run_stage(t_audit, hyp, goal, db=db, notify=False)
    assert rcpt.process_status == ProcessStatus.EXIT_OK
    assert rcpt.audit_verdict == AuditVerdict.PASS
    assert rcpt.in_tokens == 321 and rcpt.out_tokens == 27
    assert rcpt.model_used == "gpt-5-codex"
    argv_used = json.loads(args_capture.read_text(encoding="utf-8"))
    assert argv_used[:3] == ["exec", "--skip-git-repo-check", "-m"]
    assert "-s" in argv_used and argv_used[argv_used.index("-s") + 1] == "workspace-write"
    assert "-C" in argv_used and "--add-dir" in argv_used
    assert "--json" in argv_used and argv_used[-1] == "-"
    db.close()


def test_stage_runner_main_does_not_overwrite_superseded_or_paused_task(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conjectra.stage_runner import main as stage_runner_main

    origin = tmp_path / "repo_main_fence"
    base_sha = _init_sample_repo(origin)
    arr_home = tmp_path / "arr_home_main_fence"
    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-main-fence",
        project="demo",
        title="Main CLI fence",
        direction="max",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(origin),
        kill_criteria=[],
        banned_paths=[],
        writable_prefixes=["src/"],
        max_tokens_budget=50000,
    )
    db.save_goal(goal)
    hyp = HypothesisNode(
        hypothesis_id="h-main-fence",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Main CLI fence hyp",
        statement="stmt",
        mechanism="mech",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.0,
        writable_paths=["src/"],
        params={},
        kill_criteria=[],
        acceptance_cmd="true",
        screen_cmd="true",
        confirm_cmd="true",
        branch_name="arr/demo/h-main-fence",
        base_sha=base_sha,
    )
    db.register_hypothesis(hyp)
    t_run = StageTask(
        task_id="t-main-fence",
        idempotency_key="k-main-fence",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.RECONCILE,
        status="RUNNING",
        attempt=1,
        deps=[],
    )
    db.enqueue_task(t_run)

    # Simulate Goal becoming PAUSED right before worker finishes
    db.update_goal_status(goal.goal_id, "PAUSED")
    db.transition_task_status(t_run.task_id, "RUNNING", "CANCELLED")
    db.close()

    monkeypatch.setattr("conjectra.stage_runner.notify_controller", lambda *_a, **_kw: True)
    rc = stage_runner_main(["--task-id", t_run.task_id, "--arr-home", str(arr_home), "--dry-run-llm"])
    assert rc == 0

    db_check = StateDB(arr_home)
    try:
        after_t = db_check.get_task(t_run.task_id)
        assert after_t is not None
        # Cancelled/paused task MUST remain CANCELLED (never overwritten to DONE by stale worker!)
        assert after_t.status == "CANCELLED"
    finally:
        db_check.close()


def test_p1_03_r3_exec_fence_metrics_schema_and_promotion_lock_path(tmp_path: Path) -> None:
    from conjectra import gates

    origin = tmp_path / "repo_r3_metrics"
    base_sha = _init_sample_repo(origin)
    arr_home = tmp_path / "arr_home_r3_metrics"
    db = StateDB(arr_home)
    try:
        goal = ResearchGoal(
            goal_id="g-r3-m",
            project="demo",
            title="Metric schema & lock path",
            direction="max",
            primary_metric="score",
            baseline_value=1.0,
            baseline_sha=base_sha,
            repo_path=str(origin),
            kill_criteria=[],
            banned_paths=[],
            writable_prefixes=["src/"],
            max_tokens_budget=50_000,
            human_epoch=2,
            plan_epoch=3,
        )
        db.save_goal(goal)
        hyp = HypothesisNode(
            hypothesis_id="h-r3-m",
            goal_id=goal.goal_id,
            parent_id=None,
            kind=HypothesisKind.COMPETING_ROUTE,
            title="Hyp",
            statement="stmt",
            mechanism="mech",
            fidelity=FidelityRung.F1_FAST_SCREEN,
            priority_evoi=1.0,
            writable_paths=["src/"],
            params={},
            kill_criteria=[],
            acceptance_cmd="true",
            screen_cmd="echo METRIC score=2.5 denom=10",
            confirm_cmd="true",
            branch_name="arr/demo/h-r3-m",
            base_sha=base_sha,
        )
        db.register_hypothesis(hyp)
        # Register revised peer branch A so `hyp` is an unaffected sibling even without an ExperimentPlan
        db.register_hypothesis(
            HypothesisNode(
                hypothesis_id="h-r3-peer-a",
                goal_id=goal.goal_id,
                parent_id=None,
                kind=HypothesisKind.COMPETING_ROUTE,
                title="Peer A",
                statement="stmt a",
                mechanism="mech a",
                fidelity=FidelityRung.F1_FAST_SCREEN,
                priority_evoi=1.0,
                writable_paths=["src/"],
                params={"lineage_plan_epoch": 3, "lineage_human_epoch": 2, "lineage_sci_revision": 1},
                kill_criteria=[],
                acceptance_cmd="true",
                screen_cmd="true",
                confirm_cmd="true",
                branch_name="arr/demo/h-r3-peer-a",
                base_sha=base_sha,
            )
        )
        t_scr = StageTask(
            task_id="t-r3-m",
            idempotency_key="k-r3-m",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.EXP_SCREEN,
            status="RUNNING",
            human_epoch=2,
            plan_epoch=3,
            deps=[],
            worktree_path=str(origin),
        )
        db.enqueue_task(t_scr)
        runner = StageRunner(arr_home=arr_home)
        rcpt = runner.run_stage(t_scr, hyp, goal, db=db, notify=False)
        assert rcpt.plan_epoch == 0
        assert rcpt.metrics["exec_fence_plan_epoch"]["value"] == 3
        assert rcpt.metrics["exec_fence_human_epoch"]["value"] == 2
        assert rcpt.metrics["sci_revision"]["value"] == 0
        m_gate = gates.check_receipt_metrics(rcpt, require_metrics=True)
        assert m_gate.passed is True, m_gate.failures

        lp = StageRunner._promotion_lock_path(arr_home, db, "goal/projgamma:v1", "arr/projgamma/h-1")
        assert lp == arr_home / "locks" / "promote_goal_projgamma_v1_arr_projgamma_h-1.flock"
    finally:
        db.close()


def test_vm_stat_parsing_for_macos_mem_available():
    from conjectra.stage_runner import parse_vm_stat_available_mb

    text = (
        "Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
        "Pages free:                                    12800.\n"
        "Pages active:                                 500000.\n"
        "Pages inactive:                               51200.\n"
        "Pages speculative:                              6400.\n"
        "Pages wired down:                             248854.\n"
        "Pages purgeable:                                1600.\n"
    )
    # (12800 + 51200 + 6400 + 1600) pages * 16 KiB = 1125 MiB; active/wired pages are not available.
    assert parse_vm_stat_available_mb(text) == 1125


def test_hq_runtime_without_hq_binary_fails_only_on_use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from conjectra import hq_runtime

    monkeypatch.delenv("CONJECTRA_HQ_BIN", raising=False)
    monkeypatch.setattr(hq_runtime, "DEFAULT_HQ_BIN_CANDIDATES", ())
    monkeypatch.setattr(hq_runtime.shutil, "which", lambda _name: None)
    rt = hq_runtime.HQRuntime(tmp_path / "home")  # inline mode constructs this; it must not raise
    assert rt.hq_bin is None
    with pytest.raises(FileNotFoundError, match="HyperQueue binary"):
        rt._base_cmd()
