"""Tests for provider-agnostic x-boost (Defect 5), x-swarm, x-teamwork, notification/detach helpers, and workload router."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import pytest

from conjectra.cli import main as cli_main
from conjectra.multiagent import (
    ProviderSpec,
    XSwarmBoard,
    build_milestone_anchor,
    notify_hook,
    parse_provider_spec,
    run_boost,
    run_xswarm,
    run_xteamwork,
    spawn_detached_conjectra,
)
from conjectra.stage_runner import SandboxManager
from conjectra.state_db import StateDB
from conjectra.workload_matrix import (
    extract_workload_features,
    record_workload_eval,
    route_workload,
    run_layer0_gate_checks,
)


def _make_git_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "init", "-b", "main"], check=True, capture_output=True)
    for rel, txt in files.items():
        fp = repo / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(txt, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@l", "commit", "-m", "init"],
        check=True,
        capture_output=True,
    )
    return repo


def test_provider_spec_model_agnostic_and_cross_family_enforcement() -> None:
    b = parse_provider_spec("claude:claude-sonnet-5-5:high", role="builder")
    # Sonnet effort capped at medium per WORKFLOW.md
    assert b.pool == "claude"
    assert b.effort == "medium"
    assert b.family == "anthropic"

    # Auto adversary must pick a distinct model family and default effort to medium
    adv = parse_provider_spec("auto", role="adversary", avoid_family=b.family)
    assert adv.family != b.family
    assert adv.effort == "medium"

    # Arbitrary custom CLI provider works without tying to fixed models
    custom = parse_provider_spec("cmd:opencode run --quiet", role="builder")
    assert custom.pool == "cmd"
    assert custom.custom_cmd == "opencode run --quiet"
    assert custom.family.startswith("custom_")

    # Same-family explicit adversary is rejected
    with pytest.raises(ValueError, match="Cross-family violation"):
        parse_provider_spec("claude:claude-opus-5-5:high", role="adversary", avoid_family="anthropic")


def test_xswarm_board_anti_broadcast_and_zero_token_stall(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "arr_home")
    try:
        board = XSwarmBoard(db)
        board.create_swarm(
            swarm_id="sw1",
            project="demo",
            title="Demo Swarm",
            repo_path="/tmp/demo",
            base_sha="deadbeef",
            members=[
                {
                    "name": "w1",
                    "role": "worker",
                    "provider": {"pool": "claude", "model": "claude-sonnet-5-5", "effort": "medium", "family": "anthropic"},
                    "writable_paths": ["a.py"],
                    "allowed_targets": ["verifier", "coordinator"],
                },
                {
                    "name": "w2",
                    "role": "worker",
                    "provider": {"pool": "gemini", "model": "gemini-flash", "effort": "high", "family": "google"},
                    "writable_paths": ["b.py"],
                    "allowed_targets": ["verifier", "coordinator"],
                },
            ],
            stall_timeout_s=30.0,
        )
        # Worker @all broadcast blocked
        with pytest.raises(ValueError, match="Broadcast @all is forbidden"):
            board.post("sw1", "w1", "@all hey everyone", now=100.0)

        # Lateral worker-to-worker chat blocked unless explicitly declared
        with pytest.raises(ValueError, match="forbidden lateral chat"):
            board.post("sw1", "w1", "@w2 let us chat", now=100.0)

        # Valid coordinator -> worker wakeup & cursor advance
        p = board.post("sw1", "coordinator", "@w1 implement a.py", now=100.0)
        assert len(board.pending_wakeups("sw1")) == 1
        board.advance_cursor("sw1", "w1", p["seq"])
        assert len(board.pending_wakeups("sw1")) == 0

        # 0-token stall check triggers after stall_timeout_s with 0 LLM tokens
        stall = board.check_stall_zero_token("sw1", now=135.0)
        assert stall["stalled"] is True
        assert stall["monitor_tokens_used"] == 0
    finally:
        db.close()


def test_custom_cmd_provider_in_xboost(tmp_path: Path) -> None:
    repo = _make_git_repo(
        tmp_path,
        {
            "calc.py": "def clamp_pos(x: int) -> int:\n    return x\n",
            "test_calc.py": "from calc import clamp_pos\n\ndef test_pos():\n    assert clamp_pos(5) == 5\n    assert clamp_pos(-3) == 0\n",
        },
    )
    builder_script = tmp_path / "fake_builder.py"
    builder_script.write_text(
        "import sys, pathlib\n"
        "_ = sys.stdin.read()\n"
        "pathlib.Path('calc.py').write_text('def clamp_pos(x: int) -> int:\\n    return max(0, int(x))\\n')\n"
        "print('done')\n",
        encoding="utf-8",
    )
    adv_script = tmp_path / "fake_adv.py"
    adv_script.write_text(
        "import sys\n_ = sys.stdin.read()\nprint('BOOST_VERDICT: PASS')\n",
        encoding="utf-8",
    )
    sb = SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt")
    summary = run_boost(
        repo_path=repo,
        project="custom_demo",
        task_spec="Clamp negative integers to 0.",
        acceptance_cmd="pytest -q",
        writable_paths=["calc.py"],
        builder_spec=f"cmd:python3 {builder_script}",
        adversary_spec=f"cmd:python3 {adv_script}",
        use_hq=False,
        arr_home=tmp_path / "arr_home",
        sandbox_manager=sb,
    )
    assert summary["status"] == "PASSED"
    assert summary["cross_family_verified"] is True
    assert summary["rounds_used"] == 1


def test_run_boost_defect5_minor_false_alarm_and_repro_repair_loop(tmp_path: Path) -> None:
    repo = _make_git_repo(
        tmp_path,
        {
            "calc.py": "def safe_div(a: float, b: float) -> float:\n    return a / b\n",
            "test_calc.py": "from calc import safe_div\n\ndef test_div():\n    assert safe_div(6.0, 2.0) == 3.0\n",
        },
    )
    sb = SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt")

    # Case 1: Adversary returns BROKEN with severity=MINOR -> passes at 0 repair tokens!
    builder_calls_minor = 0

    def _hook_minor(prompt: str, prov: ProviderSpec, cwd: Path, readonly: bool, tag: str) -> tuple[int, str, int, int, int]:
        nonlocal builder_calls_minor
        if not readonly:
            builder_calls_minor += 1
            (cwd / "calc.py").write_text(
                "def safe_div(a: float, b: float) -> float:\n    return 0.0 if b == 0 else a / b\n",
                encoding="utf-8",
            )
            return 0, "built", 500, 200, 0
        return (
            0,
            json.dumps(
                {
                    "verdict": "BROKEN",
                    "severity": "MINOR",
                    "findings": ["Missing docstring"],
                    "repro_python": "",
                }
            ),
            200,
            50,
            0,
        )

    s_minor = run_boost(
        repo_path=repo,
        project="minor_demo",
        task_spec="Safe divide returning 0.0 when b==0.",
        acceptance_cmd="pytest -q",
        writable_paths=["calc.py"],
        builder_spec="claude:claude-sonnet-5-5:medium",
        adversary_spec="codex:gpt-5-codex",
        max_repairs=2,
        use_hq=False,
        arr_home=tmp_path / "arr_home_minor",
        sandbox_manager=sb,
        llm_hook=_hook_minor,
    )
    assert s_minor["status"] == "PASSED"
    assert s_minor["repairs_used"] == 0
    assert builder_calls_minor == 1
    assert s_minor["adversary"]["effort"] == "medium"

    # Case 2: Adversary claims BLOCKER + repro_python that exits rc == 0 -> discarded at 0 repair tokens!
    builder_calls_fa = 0

    def _hook_false_alarm(prompt: str, prov: ProviderSpec, cwd: Path, readonly: bool, tag: str) -> tuple[int, str, int, int, int]:
        nonlocal builder_calls_fa
        if not readonly:
            builder_calls_fa += 1
            (cwd / "calc.py").write_text(
                "def safe_div(a: float, b: float) -> float:\n    return 0.0 if b == 0 else a / b\n",
                encoding="utf-8",
            )
            return 0, "built", 500, 200, 0
        return (
            0,
            json.dumps(
                {
                    "verdict": "BROKEN",
                    "severity": "BLOCKER",
                    "findings": ["Claimed ZeroDivisionError on b=0"],
                    "repro_python": "from calc import safe_div\nassert safe_div(5.0, 0.0) == 0.0",
                }
            ),
            200,
            50,
            0,
        )

    s_fa = run_boost(
        repo_path=repo,
        project="fa_demo",
        task_spec="Safe divide returning 0.0 when b==0.",
        acceptance_cmd="pytest -q",
        writable_paths=["calc.py"],
        builder_spec="claude:claude-sonnet-5-5:medium",
        adversary_spec="codex:gpt-5-codex",
        max_repairs=2,
        use_hq=False,
        arr_home=tmp_path / "arr_home_fa",
        sandbox_manager=sb,
        llm_hook=_hook_false_alarm,
    )
    assert s_fa["status"] == "PASSED"
    assert s_fa["false_alarms_discarded"] == 1
    assert s_fa["repairs_used"] == 0
    assert builder_calls_fa == 1

    # Case 3: Real BLOCKER + reproducing repro_python (rc != 0) triggers Repair -> Gate cleanly with max_repairs=1!
    builder_calls_real = 0

    def _hook_real_repro(prompt: str, prov: ProviderSpec, cwd: Path, readonly: bool, tag: str) -> tuple[int, str, int, int, int]:
        nonlocal builder_calls_real
        if not readonly:
            builder_calls_real += 1
            if builder_calls_real == 1:
                (cwd / "calc.py").write_text(
                    "def safe_div(a: float, b: float) -> float:\n    return float(a) / float(b)\n",
                    encoding="utf-8",
                )
            else:
                (cwd / "calc.py").write_text(
                    "def safe_div(a: float, b: float) -> float:\n    return 0.0 if b == 0 else float(a) / float(b)\n",
                    encoding="utf-8",
                )
                (cwd / "test_calc.py").write_text(
                    "from calc import safe_div\n\ndef test_div():\n    assert safe_div(6.0, 2.0) == 3.0\n    assert safe_div(6.0, 0.0) == 0.0\n",
                    encoding="utf-8",
                )
            return 0, "built", 700, 300, 0
        return (
            0,
            json.dumps(
                {
                    "verdict": "BROKEN",
                    "severity": "BLOCKER",
                    "findings": ["ZeroDivisionError on safe_div(6.0, 0.0)"],
                    "repro_python": "from calc import safe_div\nassert safe_div(6.0, 0.0) == 0.0",
                }
            ),
            200,
            80,
            0,
        )

    s_real = run_boost(
        repo_path=repo,
        project="real_repro_demo",
        task_spec="Safe divide returning 0.0 when b==0.",
        acceptance_cmd="pytest -q",
        writable_paths=["calc.py", "test_calc.py"],
        builder_spec="claude:claude-sonnet-5-5:medium",
        adversary_spec="codex:gpt-5-codex",
        max_repairs=1,
        use_hq=False,
        arr_home=tmp_path / "arr_home_real",
        sandbox_manager=sb,
        llm_hook=_hook_real_repro,
    )
    assert s_real["status"] == "PASSED"
    assert s_real["repairs_used"] == 1
    assert builder_calls_real == 2
    assert s_real["repro_checks"][0]["reproduced"] is True
    assert s_real["coordination_tax_pct"] < 50.0


def test_notify_hook_and_spawn_detached_conjectra(tmp_path: Path) -> None:
    captured: list[tuple[list[str], str]] = []

    def _fake_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured.append((cmd, str(kwargs.get("input") or "")))
        return subprocess.CompletedProcess(cmd, 0, stdout='{"ok": true}', stderr="")

    res_cmd = notify_hook(
        "notify-send --app=conjectra {subject}",
        "Goal Update",
        "Goal finished cleanly",
        runner=_fake_run,
    )
    assert res_cmd["sent"] is True
    assert res_cmd["transport"] == "command"
    assert res_cmd["cmd"] == ["notify-send", "--app=conjectra", "Goal Update"]
    assert captured[0][1] == "Goal finished cleanly"

    events = tmp_path / "events.jsonl"
    res_file = notify_hook(f"file:{events}", "Ping", "Check status", runner=_fake_run)
    assert res_file["sent"] is True
    assert res_file["transport"] == "file"
    assert json.loads(events.read_text(encoding="utf-8").splitlines()[0])["subject"] == "Ping"
    assert len(captured) == 1

    with pytest.raises(ValueError):
        notify_hook("", "subj", "body")

    spawned_calls: list[list[str]] = []

    class _FakeProc:
        pid = 42424

    def _fake_popen(cmd: list[str], **kwargs: Any) -> _FakeProc:
        spawned_calls.append(cmd)
        assert kwargs.get("start_new_session") is True
        return _FakeProc()

    det = spawn_detached_conjectra(
        ["--arr-home", "/tmp/old", "daemon", "--detach"],
        arr_home=tmp_path / "arr_home",
        notify_cmd="notify-send {subject}",
        popen_fn=_fake_popen,
    )
    assert det["detached"] is True
    assert det["pid"] == 42424
    assert "--detach" not in spawned_calls[0]
    assert "--notify-cmd" in spawned_calls[0]


def test_real_ledger_validation_and_calibrated_workload_router_cli(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    try:
        # 1. Layer 0 deterministic gate checks work with 0 mocks
        l0 = run_layer0_gate_checks(tmp_path)
        assert len(l0) == 3
        assert all(item["intercepted"] is True for item in l0)

        # 2. Boundary validation on extract_workload_features
        with pytest.raises(ValueError):
            extract_workload_features("")
        with pytest.raises(ValueError):
            extract_workload_features("valid", readonly=True, writable_scopes=[["a.py"]], interface_frozen=False)

        # 3. Synthetic / missing raw_usage_paths MUST be rejected for is_real_llm=True
        fake_usage = tmp_path / "fake_usage.json"
        fake_usage.write_text(json.dumps({"synthetic": True, "usage": {"input_tokens": 100, "output_tokens": 50}}), encoding="utf-8")
        with pytest.raises(ValueError, match="synthetic"):
            record_workload_eval(
                db,
                workload_class="tight_adversarial",
                mode="x-boost",
                provider_pairing="claude <-> codex",
                is_real_llm=True,
                hq_used=True,
                holdout_cmd="pytest -q",
                holdout_rc=0,
                holdout_passed=True,
                defect_escaped=False,
                gate_intercepted=False,
                uncached_tokens=150,
                raw_usage_paths=[str(fake_usage)],
            )

        # 4. Record real empirical usage observations on disk
        u_boost = tmp_path / "real_boost_usage.json"
        u_boost.write_text(
            json.dumps({"model": "claude-sonnet-5-5", "usage": {"input_tokens": 1200, "output_tokens": 400}}),
            encoding="utf-8",
        )
        u_direct = tmp_path / "real_direct_usage.json"
        u_direct.write_text(
            json.dumps({"model": "deepseek-v4.1", "usage": {"input_tokens": 900, "output_tokens": 300}}),
            encoding="utf-8",
        )
        record_workload_eval(
            db,
            workload_class="tight_adversarial",
            mode="x-boost",
            provider_pairing="claude:claude-sonnet-5-5 <-> codex:gpt-5-codex",
            is_real_llm=True,
            hq_used=True,
            holdout_cmd="pytest -q",
            holdout_rc=0,
            holdout_passed=True,
            defect_escaped=False,
            gate_intercepted=False,
            uncached_tokens=1600,
            coordination_tax_pct=25.0,
            raw_usage_paths=[str(u_boost)],
        )
        record_workload_eval(
            db,
            workload_class="tight_adversarial",
            mode="direct_hq",
            provider_pairing="deepseek:deepseek-v4.1",
            is_real_llm=True,
            hq_used=True,
            holdout_cmd="pytest -q",
            holdout_rc=1,
            holdout_passed=False,
            defect_escaped=True,
            gate_intercepted=False,
            uncached_tokens=1200,
            coordination_tax_pct=0.0,
            raw_usage_paths=[str(u_direct)],
        )
    finally:
        db.close()

    # Now verify `conjectra route-workload` uses the empirical calibration from SQLite!
    rc2 = cli_main(
        [
            "--arr-home",
            str(arr_home),
            "route-workload",
            "--description",
            "Fix subtle unhashable evaluation_id deadlock boundary in single pruning module",
            "--scope",
            "harness/core/design/pruning.py",
        ]
    )
    assert rc2 == 0
    routed = json.loads(capsys.readouterr().out)
    assert routed["workload_class"] == "tight_adversarial"
    assert routed["recommended_mode"] == "x-boost"
    assert routed["calibration_status"].startswith("EMPIRICALLY_CALIBRATED")

    # Untested workload class stays UNKNOWN empirically
    untested = route_workload(
        "Read-only survey of module interfaces",
        readonly=True,
        arr_home=arr_home,
    )
    assert untested["empirical_winner"] == "UNKNOWN"
    assert untested["calibration_status"].startswith("UNKNOWN_EMPIRICAL")

    # Verify milestone anchor <4KB hard cap
    huge_anchor = build_milestone_anchor(
        phase_idx=1,
        phase_name="Big Phase",
        staging_head="a" * 40,
        slice_results=[
            {"name": f"s{i}", "provider": "claude", "head_sha": "b" * 40, "writable_paths": ["x.py"], "tests_passed": 5}
            for i in range(120)
        ],
        prescreen_note="z" * 5000,
    )
    assert len(huge_anchor.encode("utf-8")) <= 4096


def test_run_boost_enforces_goal_hard_routing_policy_and_scoped_disallows(
    tmp_path: Path,
) -> None:
    import time

    from conjectra.arbiter import ResourceArbiter
    from conjectra.model_router import ModelRouter, PolicyBlockedError
    from conjectra.multiagent import run_boost
    from conjectra.types import ResearchGoal

    origin = tmp_path / "boost_pol_repo"
    origin.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(origin), "init", "-q", "-b", "main"], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "user.name", "t"], check=True)
    subprocess.run(["git", "-C", str(origin), "config", "user.email", "t@local"], check=True)
    (origin / "solver.py").write_text("def solve(): return 1\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(origin), "add", "-A"], check=True)
    subprocess.run(["git", "-C", str(origin), "commit", "-q", "-m", "init"], check=True)
    base_sha = subprocess.run(
        ["git", "-C", str(origin), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    q_snap = tmp_path / "q_boost.json"
    q_snap.write_text(
        json.dumps(
            [
                {
                    "provider": "claude",
                    "plan": "MAX",
                    "windows": [{"name": "Claude 7 days", "used": 100.0, "resetsAt": None}],
                    "at": now_iso,
                },
                {
                    "provider": "codex",
                    "plan": "pro",
                    "windows": [{"name": "7 days", "used": 10.0, "resetsAt": None}],
                    "at": now_iso,
                },
            ]
        ),
        encoding="utf-8",
    )
    arb = ResourceArbiter(
        quotas_path=tmp_path / "nq.json",
        quota_snapshot_path=q_snap,
        ledger_path=tmp_path / "nl.json",
        gemini_quota_path=tmp_path / "nv.json",
    )
    router = ModelRouter(arbiter=arb, arr_home=tmp_path / "arr_home")
    goal = ResearchGoal(
        goal_id="g-boost-pol",
        project="demo",
        title="Boost policy check",
        direction="max",
        primary_metric="acc",
        baseline_value=0.5,
        baseline_sha=base_sha,
        repo_path=str(origin),
        kill_criteria=[],
        banned_paths=[],
        writable_prefixes=["solver.py"],
        max_tokens_budget=50000,
        routing_policy={
            "rules": [
                {
                    "rule_id": "hard-opus",
                    "roles": ["engineer_builder"],
                    "strength": "HARD",
                    "candidates": [{"provider_key": "claude-opus-5-5", "effort": "xhigh"}],
                    "disallowed_models": ["gemini-flash"],
                }
            ]
        },
    )
    # Legacy builder_spec="gemini:gemini-flash:high" MUST NOT bypass Goal HARD Opus rule -> raises WAIT_RESOURCE!
    with pytest.raises(PolicyBlockedError) as exc_info:
        run_boost(
            repo_path=origin,
            project="demo",
            task_spec="Fix solver",
            acceptance_cmd="pytest -q",
            writable_paths=["solver.py"],
            builder_spec="gemini:gemini-flash:high",
            use_hq=False,
            arr_home=tmp_path / "arr_home",
            goal=goal,
            model_router=router,
        )
    assert exc_info.value.code == "WAIT_RESOURCE"
    assert exc_info.value.waitable is True


def test_p1_05_r3_mixed_custom_cmd_never_disables_router_and_scoped_policies(
    tmp_path: Path,
) -> None:
    from conjectra.arbiter import ResourceArbiter
    from conjectra.model_router import ModelRouter, PolicyBlockedError
    from conjectra.types import FidelityRung, HypothesisKind, HypothesisNode, ResearchGoal

    repo = _make_git_repo(
        tmp_path / "r3_mixed",
        {
            "solver.py": "def f(x: int) -> int:\n    return x\n",
            "test_solver.py": "from solver import f\n\ndef test_f():\n    assert f(2) == 4\n",
        },
    )
    base_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    arb = ResourceArbiter(
        quotas_path=tmp_path / "nq.json",
        ledger_path=tmp_path / "nl.json",
        gemini_quota_path=tmp_path / "nv.json",
    )
    router = ModelRouter(arbiter=arb, arr_home=tmp_path / "arr_home")

    # Goal has NO restriction, but branch `hyp` disallows gemini-flash!
    goal_empty = ResearchGoal(
        goal_id="g-mixed-cmd",
        project="demo",
        title="Mixed cmd & scoped policy",
        direction="max",
        primary_metric="acc",
        baseline_value=0.5,
        baseline_sha=base_sha,
        repo_path=str(repo),
        kill_criteria=[],
        banned_paths=[],
        writable_prefixes=["solver.py"],
        max_tokens_budget=50_000,
    )
    hyp_branch_disallow = HypothesisNode(
        hypothesis_id="h-branch-disallow",
        goal_id=goal_empty.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Branch disallow",
        statement="stmt",
        mechanism="mech",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.0,
        writable_paths=["solver.py"],
        params={
            "routing_policy": {
                "disallowed_models": ["gemini-flash"],
            }
        },
        kill_criteria=[],
        acceptance_cmd="pytest -q",
        screen_cmd="true",
        confirm_cmd="true",
        branch_name="arr/demo/h-branch-disallow",
        base_sha=base_sha,
    )

    marker = tmp_path / "must_never_run.txt"
    adv_cmd = f"cmd:python3 -c \"import pathlib; pathlib.Path({str(marker)!r}).write_text('ran'); print('BOOST_VERDICT: PASS')\""

    # 1. Trigger B counterexample: prohibited ordinary builder + `cmd:` adversary MUST fail BEFORE execution!
    with pytest.raises(PolicyBlockedError) as exc_prohib:
        run_boost(
            repo_path=repo,
            project="demo",
            task_spec="Double x",
            acceptance_cmd="pytest -q",
            writable_paths=["solver.py"],
            builder_spec="gemini:gemini-flash:high",
            adversary_spec=adv_cmd,
            use_hq=False,
            arr_home=tmp_path / "arr_home",
            goal=goal_empty,
            hyp=hyp_branch_disallow,
            model_router=router,
        )
    assert exc_prohib.value.code == "BLOCKED_DISALLOWED"
    assert not marker.exists()

    # 2. Allowed ordinary builder + UNREGISTERED `cmd:` adversary in policy-bound run MUST fail closed (`BLOCKED_UNREGISTERED_CUSTOM_CMD`)!
    with pytest.raises(PolicyBlockedError) as exc_unreg:
        run_boost(
            repo_path=repo,
            project="demo",
            task_spec="Double x",
            acceptance_cmd="pytest -q",
            writable_paths=["solver.py"],
            builder_spec="claude:claude-sonnet-5-5:medium",
            adversary_spec=adv_cmd,
            use_hq=False,
            arr_home=tmp_path / "arr_home",
            goal=goal_empty,
            hyp=hyp_branch_disallow,
            model_router=router,
        )
    assert exc_unreg.value.code == "BLOCKED_UNREGISTERED_CUSTOM_CMD"
    assert exc_unreg.value.waitable is False
    assert not marker.exists()

    # 2b. Even when `hyp=None` and `goal.routing_policy == {}`, ANY goal-associated run rejects unregistered `cmd:`!
    with pytest.raises(PolicyBlockedError) as exc_goal_empty_unreg:
        run_boost(
            repo_path=repo,
            project="demo",
            task_spec="Double x",
            acceptance_cmd="pytest -q",
            writable_paths=["solver.py"],
            builder_spec="claude:claude-sonnet-5-5:medium",
            adversary_spec=adv_cmd,
            use_hq=False,
            arr_home=tmp_path / "arr_home",
            goal=goal_empty,
            hyp=None,
            model_router=router,
        )
    assert exc_goal_empty_unreg.value.code == "BLOCKED_UNREGISTERED_CUSTOM_CMD"
    assert not marker.exists()

    # 3. Same fail-closed enforcement in `run_xswarm` and `run_xteamwork` before creating worktrees!
    with pytest.raises(PolicyBlockedError) as exc_sw:
        run_xswarm(
            repo_path=repo,
            project="demo",
            title="Swarm blocked",
            spec_workers=[{"name": "w1", "provider": "gemini:gemini-flash:high", "writable_paths": ["solver.py"], "brief": "fix"}],
            verifier_spec=adv_cmd,
            use_hq=False,
            arr_home=tmp_path / "arr_home",
            goal=goal_empty,
            hyp=hyp_branch_disallow,
            model_router=router,
        )
    assert exc_sw.value.code == "BLOCKED_DISALLOWED"

    with pytest.raises(PolicyBlockedError) as exc_tw:
        run_xteamwork(
            repo_path=repo,
            project="demo",
            title="Teamwork blocked",
            phases=[{"name": "p1", "slices": [{"name": "s1", "provider": "claude:claude-sonnet-5-5:medium", "writable_paths": ["solver.py"], "spec": "fix"}]}],
            final_acceptance_cmd="pytest -q",
            auditor_spec=adv_cmd,
            use_hq=False,
            arr_home=tmp_path / "arr_home",
            goal=goal_empty,
            hyp=hyp_branch_disallow,
            model_router=router,
        )
    assert exc_tw.value.code == "BLOCKED_UNREGISTERED_CUSTOM_CMD"

    # 4. Explicitly registered `allowed_custom_commands` with declared `family` enforces cross-family & succeeds when distinct!
    pass_adv_script = tmp_path / "reg_adv.py"
    pass_adv_script.write_text("import sys\n_ = sys.stdin.read()\nprint('BOOST_VERDICT: PASS')\n", encoding="utf-8")
    reg_cmd = f"cmd:python3 {pass_adv_script}"

    hyp_same_fam_cmd = HypothesisNode.from_dict(
        {
            **hyp_branch_disallow.to_dict(),
            "params": {
                "routing_policy": {
                    "allowed_custom_commands": {reg_cmd: {"family": "anthropic"}},
                }
            },
        }
    )
    with pytest.raises(PolicyBlockedError) as exc_cf:
        run_boost(
            repo_path=repo,
            project="demo",
            task_spec="Double x",
            acceptance_cmd="pytest -q",
            writable_paths=["solver.py"],
            builder_spec="claude:claude-sonnet-5-5:medium",
            adversary_spec=reg_cmd,
            use_hq=False,
            arr_home=tmp_path / "arr_home",
            goal=goal_empty,
            hyp=hyp_same_fam_cmd,
            model_router=router,
        )
    assert exc_cf.value.code == "BLOCKED_CROSS_FAMILY"

    hyp_distinct_fam_cmd = HypothesisNode.from_dict(
        {
            **hyp_branch_disallow.to_dict(),
            "params": {
                "routing_policy": {
                    "allowed_custom_commands": {reg_cmd: {"family": "openai"}},
                }
            },
        }
    )
    sb = SandboxManager(sandbox_base=tmp_path / "sb_reg", worktree_base=tmp_path / "wt_reg")

    def _hook_reg(prompt: str, prov: ProviderSpec, cwd: Path, readonly: bool, tag: str) -> tuple[int, str, int, int, int]:
        if not readonly:
            (cwd / "solver.py").write_text("def f(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
            return 0, "ok", 100, 40, 0
        return 0, "BOOST_VERDICT: PASS", 80, 20, 0

    ok_sum = run_boost(
        repo_path=repo,
        project="demo",
        task_spec="Double x",
        acceptance_cmd="pytest -q",
        writable_paths=["solver.py"],
        builder_spec="claude:claude-sonnet-5-5:medium",
        adversary_spec=reg_cmd,
        use_hq=False,
        arr_home=tmp_path / "arr_home",
        sandbox_manager=sb,
        llm_hook=_hook_reg,
        goal=goal_empty,
        hyp=hyp_distinct_fam_cmd,
        model_router=router,
    )
    assert ok_sum["status"] == "PASSED"
    assert ok_sum["adversary"]["family"] == "openai"


def test_p1_05_r4_persisted_global_hitl_policy_enforced_across_goal_standalone_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conjectra.arbiter import ResourceArbiter
    from conjectra.goal_manager import GlobalGoalManager
    from conjectra.model_router import ModelRouter, PolicyBlockedError
    from conjectra.types import FidelityRung, HypothesisKind, HypothesisNode, ResearchGoal, ScientificRole

    repo = _make_git_repo(
        tmp_path / "r4_global_pol",
        {
            "solver.py": "def f(x: int) -> int:\n    return x\n",
            "test_solver.py": "from solver import f\n\ndef test_f():\n    assert f(2) == 4\n",
        },
    )
    base_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()

    arr_home = tmp_path / "arr_home_glob_r4"
    db = StateDB(arr_home)
    gm = GlobalGoalManager(db)

    # Apply supported `scope="global"` HITL `set_routing_policy` disallowing healthy `claude-sonnet-5-5` & `gemini-flash`!
    hitl_res = gm.apply_intervention(
        "set_routing_policy",
        scope="global",
        payload={
            "policy": {
                "disallowed_models": ["claude-sonnet-5-5", "gemini-flash"],
            },
        },
    )
    assert hitl_res["scope"] == "global"

    arb = ResourceArbiter(
        quotas_path=tmp_path / "nq.json",
        ledger_path=tmp_path / "nl.json",
        gemini_quota_path=tmp_path / "nv.json",
    )
    router = ModelRouter(arbiter=arb, db=db, arr_home=arr_home)
    goal = ResearchGoal(
        goal_id="g-glob-pol",
        project="demo",
        title="Global policy check",
        direction="max",
        primary_metric="acc",
        baseline_value=0.5,
        baseline_sha=base_sha,
        repo_path=str(repo),
        kill_criteria=[],
        banned_paths=[],
        writable_prefixes=["solver.py"],
        max_tokens_budget=50_000,
    )
    db.save_goal(goal)
    hyp = HypothesisNode(
        hypothesis_id="h-glob-pol",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Hyp",
        statement="stmt",
        mechanism="mech",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.0,
        writable_paths=["solver.py"],
        params={},
        kill_criteria=[],
        acceptance_cmd="pytest -q",
        screen_cmd="true",
        confirm_cmd="true",
        branch_name="arr/demo/h-glob-pol",
        base_sha=base_sha,
    )
    db.register_hypothesis(hyp)

    # 1. Normal goal-bound `resolve_route` skips globally-disallowed `claude-sonnet-5-5` default & blocks explicit sonnet!
    r_auto = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER, goal=goal, hyp=hyp, execution_host="local"
    )
    assert r_auto.canonical_model != "claude-sonnet-5-5"
    assert "gemini-flash" not in r_auto.canonical_model

    with pytest.raises(PolicyBlockedError) as exc_glob_goal:
        router.resolve_route(
            ScientificRole.ENGINEER_BUILDER,
            goal=goal,
            hyp=hyp,
            explicit_spec="claude:claude-sonnet-5-5:medium",
            execution_host="local",
        )
    assert exc_glob_goal.value.code == "BLOCKED_DISALLOWED"

    # Also auto-resolves persisted SQLite global policy via `CONJECTRA_HQ_HOME` even when `db=None`!
    monkeypatch.setenv("CONJECTRA_HQ_HOME", str(arr_home))
    router_env_only = ModelRouter(arbiter=arb)
    with pytest.raises(PolicyBlockedError) as exc_env:
        router_env_only.resolve_route(
            ScientificRole.ENGINEER_BUILDER,
            explicit_spec="claude:claude-sonnet-5-5:medium",
            execution_host="local",
        )
    assert exc_env.value.code == "BLOCKED_DISALLOWED"
    db.close()

    # 2. Context-free (`goal=None, model_router=None`) standalone `run_boost`, `run_xswarm`, `run_xteamwork` enforce global policy!
    with pytest.raises(PolicyBlockedError) as exc_b_cf:
        run_boost(
            repo_path=repo,
            project="demo",
            task_spec="Double x",
            acceptance_cmd="pytest -q",
            writable_paths=["solver.py"],
            builder_spec="claude:claude-sonnet-5-5:medium",
            use_hq=False,
            arr_home=arr_home,
        )
    assert exc_b_cf.value.code == "BLOCKED_DISALLOWED"

    with pytest.raises(PolicyBlockedError) as exc_cmd_cf:
        run_boost(
            repo_path=repo,
            project="demo",
            task_spec="Double x",
            acceptance_cmd="pytest -q",
            writable_paths=["solver.py"],
            builder_spec="codex:gpt-5-codex:high",
            adversary_spec="cmd:/bin/echo PASS",
            use_hq=False,
            arr_home=arr_home,
        )
    assert exc_cmd_cf.value.code == "BLOCKED_UNREGISTERED_CUSTOM_CMD"

    with pytest.raises(PolicyBlockedError) as exc_sw_cf:
        run_xswarm(
            repo_path=repo,
            project="demo",
            title="Swarm global block",
            spec_workers=[{"name": "w1", "provider": "gemini:gemini-flash:high", "writable_paths": ["solver.py"], "brief": "fix"}],
            use_hq=False,
            arr_home=arr_home,
        )
    assert exc_sw_cf.value.code == "BLOCKED_DISALLOWED"

    with pytest.raises(PolicyBlockedError) as exc_tw_cf:
        run_xteamwork(
            repo_path=repo,
            project="demo",
            title="Teamwork global block",
            phases=[{"name": "p1", "slices": [{"name": "s1", "provider": "claude:claude-sonnet-5-5:medium", "writable_paths": ["solver.py"], "spec": "fix"}]}],
            final_acceptance_cmd="pytest -q",
            use_hq=False,
            arr_home=arr_home,
        )
    assert exc_tw_cf.value.code == "BLOCKED_DISALLOWED"

    # 3. Public `cli.py` entrypoints (`boost`, `xswarm`, `xteamwork`) enforce persisted global policy!
    with pytest.raises(PolicyBlockedError) as exc_cli_b:
        cli_main(
            [
                "--arr-home",
                str(arr_home),
                "boost",
                "--repo-path",
                str(repo),
                "--project",
                "demo",
                "--spec",
                "Double x",
                "--acceptance-cmd",
                "pytest -q",
                "--writable-path",
                "solver.py",
                "--builder",
                "claude:claude-sonnet-5-5:medium",
            ]
        )
    assert exc_cli_b.value.code == "BLOCKED_DISALLOWED"

    sw_spec_file = tmp_path / "sw_cli_spec.json"
    sw_spec_file.write_text(
        json.dumps(
            {
                "repo_path": str(repo),
                "project": "demo",
                "workers": [{"name": "w1", "provider": "gemini:gemini-flash:high", "writable_paths": ["solver.py"], "brief": "fix"}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PolicyBlockedError) as exc_cli_sw:
        cli_main(["--arr-home", str(arr_home), "xswarm", "--spec-file", str(sw_spec_file), "--no-hq"])
    assert exc_cli_sw.value.code == "BLOCKED_DISALLOWED"

    tw_spec_file = tmp_path / "tw_cli_spec.json"
    tw_spec_file.write_text(
        json.dumps(
            {
                "repo_path": str(repo),
                "project": "demo",
                "final_acceptance_cmd": "pytest -q",
                "phases": [{"name": "p1", "slices": [{"name": "s1", "provider": "claude:claude-sonnet-5-5:medium", "writable_paths": ["solver.py"], "spec": "fix"}]}],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PolicyBlockedError) as exc_cli_tw:
        cli_main(["--arr-home", str(arr_home), "xteamwork", "--spec-file", str(tw_spec_file)])
    assert exc_cli_tw.value.code == "BLOCKED_DISALLOWED"
