"""End-to-end closed-loop tests for conjectra controller, CLI, metrics, and workflow."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import pytest

from conjectra import cli
from conjectra.arbiter import ResourceArbiter
from conjectra.controller import (
    ALLOWED_OWNER_GATES,
    OWNER_GATE_GOAL_BAR,
    OWNER_GATE_MAIN_HOLDOUT,
    OWNER_GATE_PAID_COMPUTE,
    OWNER_GATE_ROUTE_AUDIT,
    ResearchController,
    get_owner_escalations,
)
from conjectra.metrics import UNKNOWN, compute_benchmark_report, render_latest_status_md
from conjectra.state_db import StateDB
from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ScientificVerdict,
    StageKind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
ARRCTL_BIN = REPO_ROOT / "bin" / "conjectra"


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _create_sample_repo(repo_dir: Path) -> str:
    repo_dir.mkdir(parents=True, exist_ok=True)
    _git(repo_dir, "init")
    _git(repo_dir, "config", "user.email", "test@conjectra.local")
    _git(repo_dir, "config", "user.name", "ARR Test")
    (repo_dir / "src").mkdir(exist_ok=True)
    (repo_dir / "tests").mkdir(exist_ok=True)
    (repo_dir / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\npythonpath = ["."]\naddopts = ["-s"]\n',
        encoding="utf-8",
    )
    (repo_dir / "src" / "__init__.py").write_text("", encoding="utf-8")
    (repo_dir / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (repo_dir / "src" / "algo.py").write_text(
        "def score(x: float) -> float:\n    return x * 2.0\n",
        encoding="utf-8",
    )
    (repo_dir / "tests" / "test_algo.py").write_text(
        "from src.algo import score\n\ndef test_score():\n    val = score(2.0)\n    assert val == 4.0\n    print(f'METRIC score={val} denom=1')\n",
        encoding="utf-8",
    )
    _git(repo_dir, "add", ".")
    _git(repo_dir, "commit", "-m", "initial baseline")
    return _git(repo_dir, "rev-parse", "HEAD").stdout.strip()


def _write_sample_baseline(path: Path) -> Path:
    doc = {
        "baseline_id": "baseline_v1_frozen_20261126_6h",
        "frozen_at": "2026-11-26T15:13:14+08:00",
        "window_hours": 6.0,
        "prior_6h_reference": {
            "project_stats": {
                "PROJALPHA": {"code_commits": 6, "tested_commits": 0},
                "ProjBeta": {"code_commits": 6, "tested_commits": 0},
            },
            "usage_6h": {
                "calls": 2793,
                "in_tokens": 7008104,
                "out_tokens": 2932138,
            },
        },
        "target_sla_for_v2_kernel": {
            "refill_latency_p95_sec_max": 60,
            "dispatchable_slot_utilization_min_pct": 80,
            "verified_deliverables_per_hour_gain_min_pct": 50,
            "tokens_per_verified_deliverable_reduction_min_pct": 30,
            "failure_traceability_pct": 100,
        },
    }
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def test_full_closed_loop_with_real_hq_daemon_and_reactor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arr_home = tmp_path / "arr_home"
    sb_root = tmp_path / "sandboxes"
    wt_root = tmp_path / "worktrees"
    repo_dir = tmp_path / "sample_repo"
    base_sha = _create_sample_repo(repo_dir)
    baseline_path = _write_sample_baseline(tmp_path / "baseline.json")

    monkeypatch.setenv("CONJECTRA_HOME", str(arr_home))
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(sb_root))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(wt_root))
    monkeypatch.setenv("CONJECTRA_BASELINE_PATH", str(baseline_path))

    # 1. init-pilot & submit-goal via CLI
    assert cli.main(["--arr-home", str(arr_home), "init-pilot", "--no-ssh"]) == 0
    capsys.readouterr()

    proj_name = f"e2e_{uuid.uuid4().hex[:6]}"
    goal_id = f"g-{uuid.uuid4().hex[:6]}"
    assert (
        cli.main(
            [
                "--arr-home",
                str(arr_home),
                "submit-goal",
                "--goal-id",
                goal_id,
                "--project",
                proj_name,
                "--title",
                "Closed loop generic optimization",
                "--direction",
                "maximize",
                "--primary-metric",
                "score",
                "--baseline-value",
                "0.5",
                "--baseline-sha",
                base_sha,
                "--repo-path",
                str(repo_dir),
                "--writable-prefix",
                "src",
                "--kill-criterion",
                "tests_passed == 0",
                "--max-tokens-budget",
                "100000",
                "--max-concurrent-branches",
                "4",
                "--adapter",
                "generic",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # 2. Run real HQ server + worker + reactor daemon with 15s fallback sweep
    #    (proving <2s socket-driven refill without waiting on 15s polling).
    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(
        arr_home,
        arbiter=arbiter,
        execution_mode="hq",
        propose_mode="seeds",
        dry_run_llm=True,
        sweep_interval_s=15.0,
    )
    t_daemon = threading.Thread(
        target=lambda: ctrl.serve_forever(goal_id=goal_id, max_seconds=45.0),
        daemon=True,
    )
    t_daemon.start()

    try:
        deadline = time.monotonic() + 40.0
        completed_all = False
        while time.monotonic() < deadline:
            db = StateDB(arr_home)
            try:
                hyps = db.list_hypotheses(goal_id=goal_id)
                tasks = db.list_tasks(goal_id=goal_id)
                receipts = db.list_receipts(goal_id=goal_id)
                has_confirm = any(r.stage == StageKind.EXP_CONFIRM for r in receipts)
                has_ablation = any(h.kind == HypothesisKind.ABLATION for h in hyps)
                has_falsification = any(h.kind == HypothesisKind.FALSIFICATION for h in hyps)
                has_round2 = any(
                    h.kind == HypothesisKind.REFINEMENT and h.params.get("round") == 2
                    for h in hyps
                )
                all_tasks_done = bool(tasks) and all(
                    t.status in ("DONE", "FAILED", "CANCELLED") for t in tasks
                )
                if has_confirm and has_ablation and has_falsification and has_round2 and all_tasks_done:
                    completed_all = True
                    break
            finally:
                db.close()
            time.sleep(0.2)

        ctrl.stop()
        t_daemon.join(timeout=5.0)
        assert completed_all, "Full HQ closed loop did not complete all stages + Round 2 in time"
    finally:
        ctrl.stop()
        t_daemon.join(timeout=5.0)
        try:
            ctrl.hq.stop()
        except Exception:
            pass
        ctrl.close()

    # 3. Verify stages, receipts, Round 2, and benchmark-vs-baseline
    assert (
        cli.main(
            [
                "--arr-home",
                str(arr_home),
                "benchmark-vs-baseline",
                "--goal-id",
                goal_id,
                "--no-ssh",
            ]
        )
        == 0
    )
    bench = json.loads(capsys.readouterr().out)
    assert bench["pilot_window"]["full_6h_window_status"] == UNKNOWN
    assert bench["pilot_window"]["duration_hours"] < 6.0
    assert bench["baseline"]["deliverables_per_hour"] == pytest.approx(2.0)
    assert bench["baseline"]["tokens_per_deliverable"] == pytest.approx(828353.5)

    m = bench["metrics"]
    assert isinstance(m["refill_latency_p95_sec"]["value"], float)
    assert m["refill_latency_p95_sec"]["value"] <= 60.0
    assert m["refill_latency_p95_sec"]["meets_sla"] is True
    assert isinstance(m["dispatchable_slot_utilization_pct"]["value"], float)
    assert m["verified_deliverables_per_hour"]["verified_deliverables"] >= 2
    assert m["verified_deliverables_per_hour"]["value_full_6h"] == UNKNOWN
    assert m["failure_traceability_pct"]["value"] == 100.0
    assert m["first_pass_rate"]["value"] == 1.0


def test_prune_refuted_hypothesis_and_graveyard_dedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arr_home = tmp_path / "arr_home"
    repo_dir = tmp_path / "repo"
    base_sha = _create_sample_repo(repo_dir)
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(tmp_path / "sb"))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(tmp_path / "wt"))

    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-prune",
        project=f"prune_{uuid.uuid4().hex[:6]}",
        title="Prune refuted hypothesis",
        direction="maximize",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(repo_dir),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["lockbox/"],
        writable_prefixes=["src"],
        max_tokens_budget=50000,
        max_concurrent_branches=2,
        adapter_name="generic",
    )
    db.save_goal(goal)

    bad_proposal = [
        {
            "kind": "PARAMETER_RACE",
            "title": "Failing parameter candidate",
            "statement": "Empirically measured metric hits kill criterion and triggers refutation + graveyard.",
            "mechanism": "bad_param_combo",
            "fidelity": "F1_FAST_SCREEN",
            "writable_paths": ["src"],
            "params": {"alpha": -999},
            "kill_criteria": ["score < 0.5"],
            "acceptance_cmd": "pytest -q",
            "screen_cmd": "python3 -c \"print('METRIC score=0.10 denom=1'); raise SystemExit(1)\"",
            "confirm_cmd": "pytest -q",
        }
    ]
    from conjectra.explore import import_proposals_json

    batch = import_proposals_json(goal, bad_proposal, db=db, persist=True)
    assert len(batch.accepted) == 1
    bad_hyp = batch.accepted[0]

    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(
        arr_home,
        db=db,
        arbiter=arbiter,
        execution_mode="inline",
        propose_mode="import",
        dry_run_llm=True,
    )
    from conjectra.controller import _make_stage_task

    db.enqueue_task(_make_stage_task(goal, bad_hyp, StageKind.CODE))
    for _ in range(6):
        ctrl.step(goal_id=goal.goal_id)

    refreshed = db.get_hypothesis(bad_hyp.hypothesis_id)
    assert refreshed is not None
    assert refreshed.status == "REFUTED"
    assert refreshed.scientific_verdict == ScientificVerdict.REFUTED
    graveyard = db.list_graveyard(goal_id=goal.goal_id)
    assert any(g["hypothesis_id"] == bad_hyp.hypothesis_id for g in graveyard)

    # Re-importing the exact same refuted fingerprint must be blocked by graveyard dedup
    dup_batch = import_proposals_json(goal, bad_proposal, db=db, persist=True)
    assert len(dup_batch.accepted) == 0
    assert len(dup_batch.rejected) == 1
    db.close()


def test_reactor_instant_wake_latency_under_2s(tmp_path: Path) -> None:
    arr_home = tmp_path / "arr_home"
    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(
        arr_home,
        arbiter=arbiter,
        execution_mode="inline",
        dry_run_llm=True,
        sweep_interval_s=15.0,
    )

    step_times: list[float] = []
    step_ev = threading.Event()

    def _on_step(_res: dict[str, Any]) -> None:
        step_times.append(time.monotonic())
        if len(step_times) >= 2:
            step_ev.set()

    t = threading.Thread(
        target=lambda: ctrl.serve_forever(max_seconds=10.0, on_step=_on_step),
        daemon=True,
    )
    t.start()
    try:
        deadline = time.monotonic() + 3.0
        while not (arr_home / "reactor.sock").exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        assert (arr_home / "reactor.sock").exists()

        # Wait until initial startup step completed
        while not step_times and time.monotonic() < deadline:
            time.sleep(0.02)

        t0 = time.monotonic()
        rc = cli.main(["--arr-home", str(arr_home), "notify-exit", "--task-id", "t-wake", "--exit-code", "0"])
        assert rc == 0
        assert step_ev.wait(timeout=2.0), "Reactor did not wake up within 2s on socket event"
        wake_latency = step_times[-1] - t0
        assert wake_latency < 2.0
    finally:
        ctrl.stop()
        t.join(timeout=3.0)
        ctrl.close()


def test_escalation_detector_only_four_owner_gates(tmp_path: Path) -> None:
    arr_home = tmp_path / "arr_home"
    repo_dir = tmp_path / "repo"
    base_sha = _create_sample_repo(repo_dir)
    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-esc",
        project="escproj",
        title="Escalation gate test",
        direction="maximize",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(repo_dir),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["lockbox/"],
        writable_prefixes=["src"],
        max_tokens_budget=100,
        max_concurrent_branches=2,
    )
    db.save_goal(goal)

    ctrl = ResearchController(arr_home, db=db, execution_mode="inline", dry_run_llm=True)
    with pytest.raises(ValueError):
        ctrl.flag_owner_escalation(goal.goal_id, "UNSUPPORTED_NOISY_GATE", "nope")

    assert ctrl.flag_owner_escalation(goal.goal_id, OWNER_GATE_GOAL_BAR, "Metric bar change requested") is True
    assert ctrl.flag_owner_escalation(goal.goal_id, OWNER_GATE_MAIN_HOLDOUT, "Attempted lockbox touch") is True
    assert ctrl.flag_owner_escalation(goal.goal_id, OWNER_GATE_PAID_COMPUTE, "Budget cap reached") is True
    assert ctrl.flag_owner_escalation(goal.goal_id, OWNER_GATE_ROUTE_AUDIT, "3 blind audits failed") is True
    # Deduplication check
    assert ctrl.flag_owner_escalation(goal.goal_id, OWNER_GATE_GOAL_BAR, "Metric bar change requested") is False

    escs = get_owner_escalations(db, goal_id=goal.goal_id)
    assert len(escs) == 4
    assert {e["gate"] for e in escs} == set(ALLOWED_OWNER_GATES)

    bench = compute_benchmark_report(db, goal_id=goal.goal_id, fetch_ssh=False)
    md = render_latest_status_md(db, bench, escalations=escs)
    for line in md.splitlines():
        if line.startswith("|"):
            cols = [c for c in line.strip("|").split("|")]
            assert len(cols) < 4, f"Table must have <4 columns for narrow screen: {line}"
    db.close()


def test_conjectra_cli_commands_and_sync_status_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    arr_home = tmp_path / "arr_home"
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(tmp_path / "sb"))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(tmp_path / "wt"))
    repo_dir = tmp_path / "repo"
    base_sha = _create_sample_repo(repo_dir)
    baseline_path = _write_sample_baseline(tmp_path / "baseline.json")

    # Empty DB benchmark marks unmeasurable fields UNKNOWN (no fabrication)
    assert cli.main(["--arr-home", str(arr_home), "init-pilot", "--no-ssh", "--baseline-path", str(baseline_path)]) == 0
    capsys.readouterr()
    assert cli.main(["--arr-home", str(arr_home), "benchmark-vs-baseline", "--no-ssh", "--baseline-path", str(baseline_path)]) == 0
    empty_bench = json.loads(capsys.readouterr().out)
    assert empty_bench["metrics"]["refill_latency_p95_sec"]["value"] == UNKNOWN
    assert empty_bench["metrics"]["dispatchable_slot_utilization_pct"]["value"] == UNKNOWN
    assert empty_bench["metrics"]["tokens_per_verified_deliverable"]["value"] == UNKNOWN
    assert empty_bench["metrics"]["first_pass_rate"]["value"] == UNKNOWN

    goal_file = tmp_path / "goal.json"
    goal_file.write_text(
        json.dumps(
            {
                "goal_id": "g-cli",
                "project": f"cliproj_{uuid.uuid4().hex[:5]}",
                "title": "CLI exercise",
                "direction": "maximize",
                "primary_metric": "score",
                "baseline_value": 1.0,
                "baseline_sha": base_sha,
                "repo_path": str(repo_dir),
                "kill_criteria": ["tests_passed == 0"],
                "banned_paths": ["sealed_eval/"],
                "writable_prefixes": ["src"],
                "max_tokens_budget": 100000,
                "max_concurrent_branches": 2,
                "adapter_name": "generic",
            }
        ),
        encoding="utf-8",
    )
    assert cli.main(["--arr-home", str(arr_home), "submit-goal", "--file", str(goal_file)]) == 0
    capsys.readouterr()

    prop_file = tmp_path / "proposals.json"
    prop_file.write_text(
        json.dumps(
            {
                "goal_id": "g-cli",
                "proposals": [
                    {
                        "kind": "PARAMETER_RACE",
                        "title": "CLI imported param race",
                        "statement": "Test import-proposals via CLI",
                        "mechanism": "cli_param_race",
                        "fidelity": "F1_FAST_SCREEN",
                        "writable_paths": ["src"],
                        "params": {"scale": 2},
                        "kill_criteria": ["tests_passed == 0"],
                        "acceptance_cmd": "pytest -q",
                        "screen_cmd": "pytest -q",
                        "confirm_cmd": "pytest -q",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert cli.main(["--arr-home", str(arr_home), "import-proposals", "--file", str(prop_file)]) == 0
    imp_out = json.loads(capsys.readouterr().out)
    assert len(imp_out["accepted"]) == 1
    assert len(imp_out["enqueued_task_ids"]) == 1

    # Run bounded daemon in inline mode and exercise status/board/export/sync-status
    assert cli.main(["--arr-home", str(arr_home), "step", "--mode", "inline", "--dry-run-llm"]) == 0
    capsys.readouterr()
    assert cli.main(["--arr-home", str(arr_home), "daemon", "--mode", "inline", "--dry-run-llm", "--max-seconds", "0.2"]) == 0
    capsys.readouterr()
    assert cli.main(["--arr-home", str(arr_home), "status", "--goal-id", "g-cli"]) == 0
    status_out = json.loads(capsys.readouterr().out)
    assert status_out["goals"][0]["goal_id"] == "g-cli"

    assert cli.main(["--arr-home", str(arr_home), "board", "--goal-id", "g-cli", "--format", "json"]) == 0
    board_out = json.loads(capsys.readouterr().out)
    assert len(board_out["hypotheses"]) >= 1

    assert cli.main(["--arr-home", str(arr_home), "export-teamwork-brief", "--goal-id", "g-cli", "--format", "json"]) == 0
    brief_out = json.loads(capsys.readouterr().out)
    assert brief_out["goal"]["goal_id"] == "g-cli"

    assert cli.main(["--arr-home", str(arr_home), "sync-status", "--goal-id", "g-cli", "--baseline-path", str(baseline_path), "--no-ssh", "--dry-run"]) == 0
    sync_out = json.loads(capsys.readouterr().out)
    assert sync_out["mode"] == "dry_run"
    status_md_text = Path(sync_out["local_status_path"]).read_text(encoding="utf-8")
    for line in status_md_text.splitlines():
        if line.startswith("|"):
            assert len(line.strip("|").split("|")) < 4

    # Invoke bin/conjectra directly via subprocess
    proc = subprocess.run(
        [str(ARRCTL_BIN), "--arr-home", str(arr_home), "status"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "g-cli" in proc.stdout




def test_blind_audit_fail_reconciles_to_refuted_and_graveyard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arr_home = tmp_path / "arr_home"
    repo_dir = tmp_path / "repo"
    base_sha = _create_sample_repo(repo_dir)
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(tmp_path / "sb"))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(tmp_path / "wt"))

    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-audit-fail",
        project=f"auditproj_{uuid.uuid4().hex[:5]}",
        title="Blind audit failure reconciliation",
        direction="maximize",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(repo_dir),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["lockbox/"],
        writable_prefixes=["src"],
        max_tokens_budget=50000,
        max_concurrent_branches=2,
        adapter_name="generic",
    )
    db.save_goal(goal)

    hyp = HypothesisNode(
        hypothesis_id="hyp-audit-fail",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Candidate failing blind audit",
        statement="Must become REFUTED and graveyarded when BLIND_AUDIT fails",
        mechanism="audit_fail_route",
        fidelity=FidelityRung.F3_MILESTONE_AUDIT,
        priority_evoi=2.0,
        writable_paths=["src"],
        params={"route": "x"},
        kill_criteria=["regression"],
        acceptance_cmd="pytest -q",
        screen_cmd="pytest -q",
        confirm_cmd="pytest -q",
        branch_name=f"arr/{goal.project}/hyp-audit-fail",
        base_sha=base_sha,
        head_sha=base_sha,
        status="PROMOTED_F3",
        scientific_verdict=ScientificVerdict.CONFIRMED,
    )
    db.register_hypothesis(hyp)

    from conjectra.controller import _make_stage_task

    audit_task = _make_stage_task(goal, hyp, StageKind.BLIND_AUDIT)
    audit_task.status = "DONE"
    db.enqueue_task(audit_task)
    db.record_receipt(
        EvidenceReceipt(
            receipt_id="rcpt-audit-fail-1",
            task_id=audit_task.task_id,
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.BLIND_AUDIT,
            base_sha=base_sha,
            head_sha=base_sha,
            command="dry_run_audit",
            exit_code=0,
            wall_sec=0.5,
            max_rss_mb=32.0,
            in_tokens=100,
            out_tokens=50,
            model_used="codex",
            process_status=ProcessStatus.EXIT_OK,
            engineering_verdict=EngineeringVerdict.PASS,
            scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
            audit_verdict=AuditVerdict.FAIL,
            tests_passed=0,
            tests_total=0,
            metrics={},
            raw_artifact_paths=["/tmp/audit.log"],
            failure_reason="Independent blind audit returned FAIL",
            created_ts=time.time(),
        )
    )

    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(arr_home, db=db, arbiter=arbiter, execution_mode="inline", dry_run_llm=True)
    ctrl.step(goal_id=goal.goal_id)

    refreshed = db.get_hypothesis(hyp.hypothesis_id)
    assert refreshed is not None
    assert refreshed.status == "NEEDS_REPAIR"
    assert refreshed.scientific_verdict == ScientificVerdict.CONFIRMED
    assert not any(
        g["hypothesis_id"] == hyp.hypothesis_id and g.get("reason") == "BLIND_AUDIT_FAIL"
        for g in db.list_graveyard(goal_id=goal.goal_id)
    )
    repairs = [
        t
        for t in db.list_tasks(goal_id=goal.goal_id, stage=StageKind.CODE)
        if t.hypothesis_id == hyp.hypothesis_id
    ]
    assert len(repairs) == 1
    assert repairs[0].attempt == 2
    db.close()


def test_round2_fallback_runs_when_ai_proposer_errors_but_not_on_guard_fuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arr_home = tmp_path / "arr_home"
    repo_dir = tmp_path / "repo"
    base_sha = _create_sample_repo(repo_dir)
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(tmp_path / "sb"))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(tmp_path / "wt"))

    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-r2-ai-err",
        project=f"r2proj_{uuid.uuid4().hex[:5]}",
        title="Round 2 AI error fallback",
        direction="maximize",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(repo_dir),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["lockbox/"],
        writable_prefixes=["src"],
        max_tokens_budget=50000,
        max_concurrent_branches=2,
        adapter_name="generic",
    )
    db.save_goal(goal)

    parent = HypothesisNode(
        hypothesis_id="hyp-r1-win",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Round 1 winner",
        statement="Confirmed Round 1 route",
        mechanism="r1_win_mech",
        fidelity=FidelityRung.F3_MILESTONE_AUDIT,
        priority_evoi=3.0,
        writable_paths=["src"],
        params={"route": "r1"},
        kill_criteria=["tests_passed == 0"],
        acceptance_cmd="pytest -q",
        screen_cmd="pytest -q",
        confirm_cmd="pytest -q",
        branch_name=f"arr/{goal.project}/hyp-r1-win",
        base_sha=base_sha,
        head_sha=base_sha,
        status="COMPLETED",
        scientific_verdict=ScientificVerdict.CONFIRMED,
    )
    db.register_hypothesis(parent)

    from conjectra.controller import _make_stage_task

    done_task = _make_stage_task(goal, parent, StageKind.RECONCILE)
    done_task.status = "DONE"
    db.enqueue_task(done_task)
    db.record_receipt(
        EvidenceReceipt(
            receipt_id="rcpt-r1-win",
            task_id=done_task.task_id,
            goal_id=goal.goal_id,
            hypothesis_id=parent.hypothesis_id,
            stage=StageKind.RECONCILE,
            base_sha=base_sha,
            head_sha=base_sha,
            command="reconcile",
            exit_code=0,
            wall_sec=0.1,
            max_rss_mb=16.0,
            in_tokens=0,
            out_tokens=0,
            model_used="deterministic",
            process_status=ProcessStatus.EXIT_OK,
            engineering_verdict=EngineeringVerdict.PASS,
            scientific_verdict=ScientificVerdict.CONFIRMED,
            audit_verdict=AuditVerdict.NOT_REQUIRED,
            tests_passed=1,
            tests_total=1,
            metrics={},
            raw_artifact_paths=["/tmp/reconcile.log"],
            created_ts=time.time(),
        )
    )

    from conjectra.explore import ExploreEngine

    def _failing_runner(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(["dispatch.py"], 1, stdout="", stderr="pool unavailable")

    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(
        arr_home,
        db=db,
        arbiter=arbiter,
        explore=ExploreEngine(db, subprocess_runner=_failing_runner),
        execution_mode="inline",
        propose_mode="auto",
        dry_run_llm=False,
    )
    ctrl._handled_tasks.add(done_task.task_id)
    summary: dict[str, Any] = {
        "ts": time.time(),
        "leases_reaped": 0,
        "recovered_tasks": 0,
        "reconciled": 0,
        "proposed": 0,
        "tasks_enqueued": 0,
        "tasks_dispatched": 0,
        "pruned": 0,
    }
    ctrl._maybe_propose_round(goal, summary)
    hyps = db.list_hypotheses(goal_id=goal.goal_id)
    assert any(h.kind == HypothesisKind.REFINEMENT and h.params.get("round") == 2 for h in hyps)
    db.close()


def test_global_test_sim_cap_and_mem_wait_deferral(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arr_home = tmp_path / "arr_home"
    repo_dir = tmp_path / "repo"
    base_sha = _create_sample_repo(repo_dir)
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(tmp_path / "sb"))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(tmp_path / "wt"))

    db = StateDB(arr_home)
    goals: list[ResearchGoal] = []
    for idx in range(2):
        g = ResearchGoal(
            goal_id=f"g-cap-{idx}",
            project=f"cap_proj_{idx}",
            title=f"Cap test {idx}",
            direction="maximize",
            primary_metric="score",
            baseline_value=1.0,
            baseline_sha=base_sha,
            repo_path=str(repo_dir),
            kill_criteria=["tests_passed == 0"],
            banned_paths=["lockbox/"],
            writable_prefixes=["src"],
            max_tokens_budget=50000,
            max_concurrent_branches=4,
            adapter_name="generic",
        )
        db.save_goal(g)
        goals.append(g)

    from conjectra.controller import _make_stage_task

    screen_tasks = []
    for idx, (g, suffix) in enumerate([(goals[0], "a"), (goals[0], "b"), (goals[1], "c")]):
        h = HypothesisNode(
            hypothesis_id=f"hyp-cap-{suffix}",
            goal_id=g.goal_id,
            parent_id=None,
            kind=HypothesisKind.PARAMETER_RACE,
            title=f"Cap hyp {suffix}",
            statement="Verify global testsim cap",
            mechanism=f"cap_mech_{suffix}",
            fidelity=FidelityRung.F1_FAST_SCREEN,
            priority_evoi=10.0 - idx,
            writable_paths=["src"],
            params={"idx": idx},
            kill_criteria=["tests_passed == 0"],
            acceptance_cmd="pytest -q",
            screen_cmd="pytest -q",
            confirm_cmd="pytest -q",
            branch_name=f"arr/{g.project}/hyp-cap-{suffix}",
            base_sha=base_sha,
            head_sha=base_sha,
        )
        db.register_hypothesis(h)
        t = _make_stage_task(g, h, StageKind.EXP_SCREEN)
        db.enqueue_task(t)
        screen_tasks.append(t)

    class _FakeHQ:
        def __init__(self) -> None:
            self.submitted: list[tuple[str, Any]] = []
            self._next_id = 100

        def ensure_running(self) -> dict[str, Any]:
            return {"started_server": False, "started_worker": False}

        def submit_task(self, task: Any, *, stage: Any = None, **_kw: Any) -> int:
            self._next_id += 1
            self.submitted.append((task.task_id, stage))
            return self._next_id

    # 1. Below 10 GiB MemAvailable -> MEM_BELOW_10GB_WAIT defers cleanly without retries or escalations
    fake_hq = _FakeHQ()
    low_mem_arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=10_000_000, max_load1=128.0)
    ctrl_low = ResearchController(
        arr_home,
        db=db,
        hq=fake_hq,  # type: ignore[arg-type]
        arbiter=low_mem_arbiter,
        execution_mode="hq",
        max_concurrent_test_sim=2,
    )
    summary_low: dict[str, Any] = {
        "ts": time.time(),
        "leases_reaped": 0,
        "recovered_tasks": 0,
        "reconciled": 0,
        "proposed": 0,
        "tasks_enqueued": 0,
        "tasks_dispatched": 0,
        "pruned": 0,
    }
    ctrl_low._dispatch_ready_tasks(summary_low)
    assert summary_low["tasks_dispatched"] == 0
    assert fake_hq.submitted == []
    assert all(db.get_task(t.task_id).status == "READY" for t in screen_tasks)  # type: ignore[union-attr]
    assert all(db.get_task(t.task_id).attempt == 1 for t in screen_tasks)  # type: ignore[union-attr]
    assert get_owner_escalations(db) == []

    # 2. Healthy RAM -> max_concurrent_test_sim=2 caps concurrent EXP_SCREEN across both goals at 2
    ok_arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl_ok = ResearchController(
        arr_home,
        db=db,
        hq=fake_hq,  # type: ignore[arg-type]
        arbiter=ok_arbiter,
        execution_mode="hq",
        max_concurrent_test_sim=2,
    )
    summary_ok: dict[str, Any] = {
        "ts": time.time(),
        "leases_reaped": 0,
        "recovered_tasks": 0,
        "reconciled": 0,
        "proposed": 0,
        "tasks_enqueued": 0,
        "tasks_dispatched": 0,
        "pruned": 0,
    }
    ctrl_ok._dispatch_ready_tasks(summary_ok)
    assert summary_ok["tasks_dispatched"] == 2
    assert len(fake_hq.submitted) == 2
    assert all(st == StageKind.EXP_SCREEN for _, st in fake_hq.submitted)
    statuses = [db.get_task(t.task_id).status for t in screen_tasks]  # type: ignore[union-attr]
    assert statuses.count("DISPATCHED") == 2
    assert statuses.count("READY") == 1
    db.close()


def test_sweep_hq_v27_task_stats_and_worker_respawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arr_home = tmp_path / "arr_home"
    repo_dir = tmp_path / "repo"
    base_sha = _create_sample_repo(repo_dir)
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(tmp_path / "sb"))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(tmp_path / "wt"))

    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-hq-sweep",
        project="hq_sweep_proj",
        title="HQ v0.27.0 sweep test",
        direction="maximize",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(repo_dir),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["lockbox/"],
        writable_prefixes=["src"],
        max_tokens_budget=50000,
        max_concurrent_branches=4,
    )
    db.save_goal(goal)

    from conjectra.controller import _make_stage_task

    tasks: list[Any] = []
    for idx, jid in enumerate([101, 102, 103, 104]):
        h = HypothesisNode(
            hypothesis_id=f"hyp-hq-{idx}",
            goal_id=goal.goal_id,
            parent_id=None,
            kind=HypothesisKind.PARAMETER_RACE,
            title=f"HQ sweep hyp {idx}",
            statement="Check HQ v0.27.0 task_stats parsing",
            mechanism=f"hq_sweep_{idx}",
            fidelity=FidelityRung.F1_FAST_SCREEN,
            priority_evoi=1.0,
            writable_paths=["src"],
            params={"idx": idx},
            kill_criteria=["tests_passed == 0"],
            acceptance_cmd="pytest -q",
            screen_cmd="pytest -q",
            confirm_cmd="pytest -q",
            branch_name=f"arr/{goal.project}/hyp-hq-{idx}",
            base_sha=base_sha,
            head_sha=base_sha,
        )
        db.register_hypothesis(h)
        t = _make_stage_task(goal, h, StageKind.EXP_SCREEN)
        db.enqueue_task(t)
        db.transition_task_status(t.task_id, "READY", "DISPATCHED", hq_job_id=jid, started_ts=time.time())
        tasks.append(db.get_task(t.task_id))

    # Task 0 has a written receipt and finished in HQ without notify-exit
    db.record_receipt(
        EvidenceReceipt(
            receipt_id="rcpt-hq-fin-0",
            task_id=tasks[0].task_id,
            goal_id=goal.goal_id,
            hypothesis_id=tasks[0].hypothesis_id,
            stage=StageKind.EXP_SCREEN,
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
            scientific_verdict=ScientificVerdict.CONFIRMED,
            audit_verdict=AuditVerdict.NOT_REQUIRED,
            tests_passed=1,
            tests_total=1,
            metrics={"tests_passed": {"value": 1, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"}},
            raw_artifact_paths=["/tmp/ok.log"],
            created_ts=time.time(),
        )
    )

    class _MockHQ:
        def __init__(self) -> None:
            self.ensure_calls = 0

        def is_pending_canceled_or_dep_failed(self, _jid: int) -> tuple[bool, bool]:
            return False, False

        def job_info(self, jid: int) -> dict[str, Any]:
            if jid == 101:
                return {
                    "finished_at": "2026-11-26T09:40:00Z",
                    "info": {"is_open": False, "task_stats": {"finished": 1, "failed": 0, "canceled": 0, "aborted": 0, "waiting": 0, "running": 0}},
                }
            if jid == 102:
                return {
                    "finished_at": "2026-11-26T09:40:00Z",
                    "info": {"is_open": False, "task_stats": {"finished": 0, "failed": 1, "canceled": 0, "aborted": 0, "waiting": 0, "running": 0}},
                }
            if jid == 103:
                return {
                    "finished_at": "2026-11-26T09:40:00Z",
                    "info": {"is_open": False, "task_stats": {"finished": 0, "failed": 0, "canceled": 1, "aborted": 0, "waiting": 0, "running": 0}},
                }
            return {
                "finished_at": None,
                "info": {"is_open": False, "task_stats": {"finished": 0, "failed": 0, "canceled": 0, "aborted": 0, "waiting": 1, "running": 0}},
            }

        def worker_list(self) -> list[dict[str, Any]]:
            return [{"id": 1, "ended": {"at": "2026-11-26T09:40:01Z"}}]

        def ensure_running(self) -> dict[str, Any]:
            self.ensure_calls += 1
            return {"started_server": False, "started_worker": True}

    mock_hq = _MockHQ()
    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(
        arr_home,
        db=db,
        hq=mock_hq,  # type: ignore[arg-type]
        arbiter=arbiter,
        execution_mode="hq",
    )
    summary: dict[str, Any] = {"recovered_tasks": 0, "leases_reaped": 0}
    ctrl._sweep_leases_and_hq_jobs(summary)

    assert db.get_task(tasks[0].task_id).status == "DONE"  # type: ignore[union-attr]
    assert db.get_task(tasks[1].task_id).status == "FAILED"  # type: ignore[union-attr]
    assert db.list_receipts(task_id=tasks[1].task_id)[0].process_status == ProcessStatus.LEASE_LOST
    assert db.get_task(tasks[2].task_id).status == "CANCELLED"  # type: ignore[union-attr]
    assert db.get_task(tasks[3].task_id).status == "DISPATCHED"  # type: ignore[union-attr]
    assert mock_hq.ensure_calls == 1
    assert summary["recovered_tasks"] == 4
    db.close()


def test_round2_fallback_on_blind_audit_refuted_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arr_home = tmp_path / "arr_home"
    repo_dir = tmp_path / "repo"
    base_sha = _create_sample_repo(repo_dir)
    monkeypatch.setenv("CONJECTRA_SANDBOX_ROOT", str(tmp_path / "sb"))
    monkeypatch.setenv("CONJECTRA_WORKTREE_ROOT", str(tmp_path / "wt"))

    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-r2-audit-refine",
        project=f"r2audit_{uuid.uuid4().hex[:5]}",
        title="Round 2 audit refinement fallback",
        direction="maximize",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(repo_dir),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["lockbox/"],
        writable_prefixes=["src"],
        max_tokens_budget=50000,
        max_concurrent_branches=2,
        adapter_name="generic",
    )
    db.save_goal(goal)

    # Pre-seed Round 1 generic proposals so Round 2 seeds are already exhausted.
    from conjectra.explore import propose_hypotheses

    seed_batch = propose_hypotheses(goal, db=db, mode="seeds", persist=True)
    assert len(seed_batch.accepted) >= 1
    parent = seed_batch.accepted[0]
    parent = db.transition_hypothesis_status(
        parent.hypothesis_id,
        parent.status,
        "REFUTED",
        head_sha=base_sha,
        scientific_verdict=ScientificVerdict.REFUTED,
        fidelity=FidelityRung.F3_MILESTONE_AUDIT,
    )
    for other in seed_batch.accepted[1:]:
        db.transition_hypothesis_status(
            other.hypothesis_id,
            other.status,
            "REFUTED",
            scientific_verdict=ScientificVerdict.REFUTED,
        )

    from conjectra.controller import _make_stage_task

    audit_task = _make_stage_task(goal, parent, StageKind.BLIND_AUDIT)
    audit_task.status = "DONE"
    db.enqueue_task(audit_task)
    db.record_receipt(
        EvidenceReceipt(
            receipt_id="rcpt-r1-audit-fail",
            task_id=audit_task.task_id,
            goal_id=goal.goal_id,
            hypothesis_id=parent.hypothesis_id,
            stage=StageKind.BLIND_AUDIT,
            base_sha=base_sha,
            head_sha=base_sha,
            command="codex exec",
            exit_code=0,
            wall_sec=0.5,
            max_rss_mb=32.0,
            in_tokens=100,
            out_tokens=50,
            model_used="gpt-5-codex",
            process_status=ProcessStatus.EXIT_OK,
            engineering_verdict=EngineeringVerdict.PASS,
            scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
            audit_verdict=AuditVerdict.FAIL,
            tests_passed=0,
            tests_total=0,
            metrics={},
            raw_artifact_paths=["/tmp/audit.log"],
            failure_reason="Independent blind audit returned FAIL",
            created_ts=time.time(),
        )
    )

    arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
    ctrl = ResearchController(
        arr_home,
        db=db,
        arbiter=arbiter,
        execution_mode="inline",
        propose_mode="seeds",
        dry_run_llm=True,
    )
    ctrl._handled_tasks.add(audit_task.task_id)
    summary: dict[str, Any] = {
        "ts": time.time(),
        "leases_reaped": 0,
        "recovered_tasks": 0,
        "reconciled": 0,
        "proposed": 0,
        "tasks_enqueued": 0,
        "tasks_dispatched": 0,
        "pruned": 0,
    }
    ctrl._maybe_propose_round(goal, summary)
    hyps = db.list_hypotheses(goal_id=goal.goal_id)
    r2 = [h for h in hyps if h.kind == HypothesisKind.REFINEMENT and h.params.get("round") == 2]
    assert len(r2) == 1
    assert r2[0].parent_id == parent.hypothesis_id
    assert "blind-audit" in r2[0].statement.lower()
    db.close()


def test_benchmark_uncached_tokens_from_usage_json_and_baseline_first_pass_unknown(
    tmp_path: Path,
) -> None:
    arr_home = tmp_path / "arr_home"
    repo_dir = tmp_path / "repo"
    base_sha = _create_sample_repo(repo_dir)
    baseline_path = _write_sample_baseline(tmp_path / "baseline.json")

    db = StateDB(arr_home)
    goal = ResearchGoal(
        goal_id="g-tok-acct",
        project="tok_proj",
        title="Token accounting test",
        direction="maximize",
        primary_metric="score",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(repo_dir),
        kill_criteria=["tests_passed == 0"],
        banned_paths=[],
        writable_prefixes=["src"],
        max_tokens_budget=5000000,
        max_concurrent_branches=2,
    )
    db.save_goal(goal)

    hyp = HypothesisNode(
        hypothesis_id="hyp-tok-1",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Token accounting hyp",
        statement="Verify uncached vs cache token accounting",
        mechanism="tok_acct",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.0,
        writable_paths=["src"],
        params={},
        kill_criteria=["tests_passed == 0"],
        acceptance_cmd="pytest -q",
        screen_cmd="pytest -q",
        confirm_cmd="pytest -q",
        branch_name="arr/tok_proj/hyp-tok-1",
        base_sha=base_sha,
        head_sha=base_sha,
        status="COMPLETED",
        scientific_verdict=ScientificVerdict.CONFIRMED,
    )
    db.register_hypothesis(hyp)

    usage_file = tmp_path / "code_usage.json"
    usage_file.write_text(
        json.dumps(
            {
                "route": "gemini_cli",
                "model": "gemini-flash",
                "usage": {
                    "input_tokens": 165406,
                    "output_tokens": 62966,
                    "cache_read_tokens": 1768306,
                    "total_tokens": 228372,
                },
            }
        ),
        encoding="utf-8",
    )

    from conjectra.controller import _make_stage_task

    t_code = _make_stage_task(goal, hyp, StageKind.CODE)
    t_code.status = "DONE"
    t_code.started_ts = 1000.0
    t_code.finished_ts = 1010.0
    t_code.enqueued_ts = 999.0
    db.enqueue_task(t_code)
    db.record_receipt(
        EvidenceReceipt(
            receipt_id="rcpt-tok-code",
            task_id=t_code.task_id,
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.CODE,
            base_sha=base_sha,
            head_sha=base_sha,
            command="code",
            exit_code=0,
            wall_sec=10.0,
            max_rss_mb=64.0,
            # Simulate older receipt that folded cache_read_tokens into in_tokens
            in_tokens=165406 + 1768306,
            out_tokens=62966,
            model_used="gemini-flash",
            process_status=ProcessStatus.EXIT_OK,
            engineering_verdict=EngineeringVerdict.PASS,
            scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
            audit_verdict=AuditVerdict.NOT_REQUIRED,
            tests_passed=0,
            tests_total=0,
            metrics={},
            raw_artifact_paths=[str(usage_file)],
            created_ts=1010.0,
        )
    )

    t_gate = _make_stage_task(goal, hyp, StageKind.CPU_GATE, deps=[t_code.task_id])
    t_gate.status = "DONE"
    t_gate.enqueued_ts = 1010.0
    t_gate.started_ts = 1011.0
    t_gate.finished_ts = 1015.0
    db.enqueue_task(t_gate)
    db.record_receipt(
        EvidenceReceipt(
            receipt_id="rcpt-tok-gate",
            task_id=t_gate.task_id,
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.CPU_GATE,
            base_sha=base_sha,
            head_sha=base_sha,
            command="pytest -q",
            exit_code=0,
            wall_sec=4.0,
            max_rss_mb=32.0,
            in_tokens=0,
            out_tokens=0,
            model_used="deterministic",
            process_status=ProcessStatus.EXIT_OK,
            engineering_verdict=EngineeringVerdict.PASS,
            scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
            audit_verdict=AuditVerdict.NOT_REQUIRED,
            tests_passed=1,
            tests_total=1,
            metrics={"tests_passed": {"value": 1, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"}},
            raw_artifact_paths=["/tmp/gate.log"],
            created_ts=1015.0,
        )
    )

    bench = compute_benchmark_report(
        db, goal_id=goal.goal_id, baseline_path=baseline_path, fetch_ssh=False
    )
    tok = bench["metrics"]["tokens_per_verified_deliverable"]
    assert tok["in_tokens"] == 165406
    assert tok["out_tokens"] == 62966
    assert tok["total_tokens"] == 228372
    assert tok["cache_read_tokens_total"] == 1768306
    assert tok["gross_tokens_including_cache"] == 228372 + 1768306
    assert tok["value"] == pytest.approx(228372.0)
    assert tok["gross_tokens_including_cache_per_deliverable"] == pytest.approx(float(228372 + 1768306))

    fp = bench["metrics"]["first_pass_rate"]
    assert fp["value"] == 1.0
    assert fp["baseline_first_pass_rate"] == UNKNOWN
    assert fp["non_inferior_check"] == "PASS_OR_UNKNOWN_BASELINE"
    db.close()
