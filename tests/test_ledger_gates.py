"""Comprehensive unit tests for conjectra.types, conjectra.state_db, and conjectra.gates."""

from __future__ import annotations

import re

import json
import math
import subprocess
import threading

import pytest

from conjectra import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
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
from conjectra.gates import (
    banned_path,
    check_assertion_non_weakening,
    check_clean_tree,
    check_no_banned_access,
    check_no_chip_specialization,
    check_no_internal_ids,
    check_non_doc_code_diff,
    check_raw_artifacts_and_evidence,
    check_real_path_commits,
    check_receipt_metrics,
    detect_sibling_scope_overlaps,
    finite_number,
    load_read_bans,
    norm_path,
    run_cpu_gate,
    scope_overlaps,
    validate_writable_paths,
)
from conjectra.state_db import (
    GraveyardDuplicateError,
    StateDB,
    StateTransitionError,
    compute_hypothesis_fingerprint,
)


def _sample_goal(goal_id: str = "goal-1") -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project="conjectra-test",
        title="Reduce tail latency",
        direction="minimize",
        primary_metric="p99_ms",
        baseline_value=100.0,
        baseline_sha="1111111",
        repo_path="/tmp/repo",
        kill_criteria=["regression > 5%"],
        banned_paths=["secret_eval/"],
        writable_prefixes=["src/", "tests/"],
        max_tokens_budget=10_000,
        max_concurrent_branches=3,
        adapter_name="generic",
    )


def _sample_hypothesis(
    hypothesis_id: str = "hyp-1",
    goal_id: str = "goal-1",
    *,
    mechanism: str = "Vectorize inner loop",
    params: dict | None = None,
    writable_paths: list[str] | None = None,
    fingerprint: str = "",
) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id=hypothesis_id,
        goal_id=goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Vectorize kernel",
        statement="Vectorizing inner loop cuts p99 by 20%",
        mechanism=mechanism,
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.85,
        writable_paths=writable_paths or ["src/kernel.py", "tests/test_kernel.py"],
        params=params if params is not None else {"tile": 64, "unroll": 4},
        kill_criteria=["p99_ms > 105"],
        acceptance_cmd="pytest -q tests/test_kernel.py",
        screen_cmd="python3 bench.py --fast",
        confirm_cmd="python3 bench.py --full",
        branch_name=f"hyp/{hypothesis_id}",
        base_sha="1111111",
        fingerprint=fingerprint,
    )


def _sample_receipt(
    receipt_id: str = "rcpt-1",
    task_id: str = "task-1",
    goal_id: str = "goal-1",
    hypothesis_id: str = "hyp-1",
    *,
    in_tokens: int = 100,
    out_tokens: int = 50,
    artifact_paths: list[str] | None = None,
    metrics: dict | None = None,
) -> EvidenceReceipt:
    return EvidenceReceipt(
        receipt_id=receipt_id,
        task_id=task_id,
        goal_id=goal_id,
        hypothesis_id=hypothesis_id,
        stage=StageKind.EXP_SCREEN,
        base_sha="1111111",
        head_sha="2222222",
        command="pytest -q tests/test_kernel.py",
        exit_code=0,
        wall_sec=1.25,
        max_rss_mb=128.0,
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        model_used="test-model",
        process_status=ProcessStatus.EXIT_OK,
        engineering_verdict=EngineeringVerdict.PASS,
        scientific_verdict=ScientificVerdict.CONFIRMED,
        audit_verdict=AuditVerdict.NOT_REQUIRED,
        tests_passed=4,
        tests_total=4,
        metrics=metrics
        if metrics is not None
        else {
            "p99_ms": {
                "value": 78.5,
                "denominator": 1000,
                "stage_rung": FidelityRung.F1_FAST_SCREEN.value,
            }
        },
        raw_artifact_paths=["/tmp/raw/screen.json"]
        if artifact_paths is None
        else artifact_paths,
        failure_reason="",
        created_ts=1700000000.0,
    )


# ----------------------------------------------------------------- Types tests


def test_enums_and_dataclass_json_roundtrip():
    assert issubclass(StageKind, str)
    assert [s.value for s in StageKind] == [
        "PROPOSE",
        "DESIGN_REVIEW",
        "CODE",
        "CPU_GATE",
        "IMPL_REVIEW",
        "EXP_SCREEN",
        "EXP_CONFIRM",
        "RESULT_REVIEW",
        "BLIND_AUDIT",
        "RECONCILE",
    ]
    assert [f.value for f in FidelityRung] == [
        "F0_SANITY",
        "F1_FAST_SCREEN",
        "F2_FULL_CONFIRM",
        "F3_MILESTONE_AUDIT",
    ]
    assert [k.value for k in HypothesisKind] == [
        "COMPETING_ROUTE",
        "REFINEMENT",
        "ABLATION",
        "FALSIFICATION",
        "PARAMETER_RACE",
    ]
    assert [p.value for p in ProcessStatus] == [
        "EXIT_OK",
        "EXIT_NONZERO",
        "TIMEOUT",
        "OOM",
        "LEASE_LOST",
        "ENV_ERROR",
        "CANCELLED",
    ]
    assert [e.value for e in EngineeringVerdict] == [
        "PASS",
        "FAIL_GATE",
        "FAIL_TEST",
        "SKIPPED",
    ]
    assert [v.value for v in ScientificVerdict] == [
        "CONFIRMED",
        "REFUTED",
        "VALID_NEGATIVE",
        "UNKNOWN",
        "NOT_APPLICABLE",
    ]
    assert [a.value for a in AuditVerdict] == [
        "NOT_REQUIRED",
        "PASS",
        "FAIL",
        "PARSE_ERROR",
    ]

    goal = _sample_goal()
    hyp = _sample_hypothesis()
    task = StageTask(
        task_id="task-1",
        idempotency_key="goal-1:hyp-1:CODE:1",
        goal_id="goal-1",
        hypothesis_id="hyp-1",
        stage=StageKind.CODE,
        deps=["task-0"],
        resources=ResourceSpec(cpus=4, ram_mb=4096, gpu=1, llm_pool="pool-a"),
        hq_job_id=42,
        worktree_path="/tmp/wt/hyp-1",
        enqueued_ts=10.0,
    )
    receipt = _sample_receipt()

    for cls, obj in (
        (ResourceSpec, task.resources),
        (ResearchGoal, goal),
        (HypothesisNode, hyp),
        (StageTask, task),
        (EvidenceReceipt, receipt),
    ):
        encoded = json.dumps(obj.to_dict())
        decoded = cls.from_dict(json.loads(encoded))
        assert decoded == obj


# -------------------------------------------------------------- StateDB tests


def test_state_db_wal_events_and_idempotency(tmp_path, monkeypatch):
    monkeypatch.setenv("CONJECTRA_HOME", str(tmp_path / "arr_home"))
    with StateDB() as db:
        assert db.db_path == (tmp_path / "arr_home" / "state.db").resolve()
        mode = db._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal"

        goal = _sample_goal()
        db.save_goal(goal)
        assert db.get_goal("goal-1") == goal
        assert len(db.list_goals(status="ACTIVE")) == 1

        hyp = _sample_hypothesis()
        stored_hyp, created = db.register_hypothesis(hyp)
        assert created is True
        assert stored_hyp.fingerprint != ""

        t1 = StageTask(
            task_id="task-1",
            idempotency_key="idem-key-1",
            goal_id="goal-1",
            hypothesis_id=stored_hyp.hypothesis_id,
            stage=StageKind.CODE,
            deps=[],
            resources=ResourceSpec(cpus=2),
        )
        enq1 = db.enqueue_task(t1)
        t1_dup = StageTask(
            task_id="task-1-different-id",
            idempotency_key="idem-key-1",
            goal_id="goal-1",
            hypothesis_id=stored_hyp.hypothesis_id,
            stage=StageKind.CODE,
            deps=[],
        )
        enq2 = db.enqueue_task(t1_dup)
        assert enq2.task_id == enq1.task_id
        assert len(db.list_tasks(goal_id="goal-1")) == 1

        # Verify events.jsonl append-only log matches DB events
        lines = db.events_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        events = [json.loads(line) for line in lines]
        assert [e["event_type"] for e in events] == [
            "GOAL_SAVED",
            "HYPOTHESIS_REGISTERED",
            "TASK_ENQUEUED",
        ]


def test_hypothesis_fingerprint_dedup_and_graveyard(tmp_path):
    with StateDB(tmp_path) as db:
        db.save_goal(_sample_goal())
        h1 = _sample_hypothesis(
            "hyp-1",
            mechanism="  Vectorize   INNER loop ",
            params={"unroll": 4, "tile": 64},
            writable_paths=["./tests/test_kernel.py", "src/kernel.py/"],
        )
        h2 = _sample_hypothesis(
            "hyp-2",
            mechanism="vectorize inner loop",
            params={"tile": 64, "unroll": 4},
            writable_paths=["src/kernel.py", "tests/test_kernel.py"],
        )
        assert compute_hypothesis_fingerprint(
            h1.goal_id, h1.kind, h1.mechanism, h1.params, h1.writable_paths, h1.screen_cmd
        ) == compute_hypothesis_fingerprint(
            h2.goal_id, h2.kind, h2.mechanism, h2.params, h2.writable_paths, h2.screen_cmd
        )

        stored1, c1 = db.register_hypothesis(h1)
        stored2, c2 = db.register_hypothesis(h2)
        assert c1 is True
        assert c2 is False
        assert stored2.hypothesis_id == stored1.hypothesis_id

        db.record_graveyard(stored1.hypothesis_id, "Refuted on screen rung")
        h3 = _sample_hypothesis("hyp-3", mechanism="vectorize inner loop")
        with pytest.raises(GraveyardDuplicateError):
            db.register_hypothesis(h3)


def test_atomic_transitions_ready_deps_and_budget(tmp_path):
    with StateDB(tmp_path) as db:
        db.save_goal(_sample_goal())
        h = _sample_hypothesis()
        db.register_hypothesis(h)

        t_code = db.enqueue_task(
            StageTask(
                task_id="t-code",
                idempotency_key="k-code",
                goal_id="goal-1",
                hypothesis_id="hyp-1",
                stage=StageKind.CODE,
                deps=[],
            )
        )
        db.enqueue_task(
            StageTask(
                task_id="t-gate",
                idempotency_key="k-gate",
                goal_id="goal-1",
                hypothesis_id="hyp-1",
                stage=StageKind.CPU_GATE,
                deps=["t-code"],
            )
        )
        assert [t.task_id for t in db.list_ready_tasks("goal-1")] == ["t-code"]

        running = db.transition_task_status(
            t_code.task_id, "READY", "RUNNING", started_ts=100.0
        )
        assert running.status == "RUNNING"
        with pytest.raises(StateTransitionError):
            db.transition_task_status(t_code.task_id, "READY", "RUNNING")

        db.transition_task_status("t-code", "RUNNING", "DONE", finished_ts=120.0)
        assert [t.task_id for t in db.list_ready_tasks("goal-1")] == ["t-gate"]

        db.transition_hypothesis_status(
            "hyp-1", "PENDING", "ACTIVE", head_sha="2222222"
        )
        assert db.get_hypothesis("hyp-1").head_sha == "2222222"

        db.record_receipt(_sample_receipt("r1", "t-code", in_tokens=6000, out_tokens=1000))
        usage = db.get_goal_token_usage("goal-1")
        assert usage["total_tokens"] == 7000
        assert usage["remaining_tokens"] == 3000
        assert usage["budget_exhausted"] is False

        db.record_receipt(_sample_receipt("r2", "t-gate", in_tokens=2500, out_tokens=1000))
        assert db.is_budget_exhausted("goal-1") is True
        summary = db.get_goal_summary("goal-1")
        assert summary["tasks_total"] == 2
        assert summary["receipts_total"] == 2


def test_ttl_leases_and_concurrent_contention(tmp_path):
    with StateDB(tmp_path) as db:
        t0 = 1000.0
        l1 = db.acquire_lease(
            "gpu:0", "task-1", "worker-a", ttl_s=60.0, max_concurrent=1, now=t0
        )
        assert l1 is not None
        assert db.acquire_lease(
            "gpu:0", "task-2", "worker-b", ttl_s=60.0, max_concurrent=1, now=t0 + 10
        ) is None

        renewed = db.renew_lease(l1["lease_id"], ttl_s=120.0, holder="worker-a", now=t0 + 20)
        assert renewed is not None
        assert renewed["expires_ts"] == t0 + 140.0

        # Cannot renew with wrong holder
        assert db.renew_lease(l1["lease_id"], ttl_s=60.0, holder="worker-b", now=t0 + 25) is None

        # Reap after expiry frees slot
        reaped = db.reap_expired_leases(now=t0 + 200.0)
        assert len(reaped) == 1
        assert reaped[0]["lease_id"] == l1["lease_id"]

        l2 = db.acquire_lease(
            "gpu:0", "task-2", "worker-b", ttl_s=60.0, max_concurrent=1, now=t0 + 201.0
        )
        assert l2 is not None
        assert db.release_lease(l2["lease_id"], holder="worker-b", now=t0 + 205.0) is True
        assert db.list_active_leases("gpu:0", now=t0 + 206.0) == []

        # Concurrent contention for 2 slots across 8 threads
        acquired: list[dict] = []
        lock = threading.Lock()

        def _worker(idx: int) -> None:
            lease = db.acquire_lease(
                "eda_sim", f"task-{idx}", f"w-{idx}", ttl_s=60.0, max_concurrent=2, now=t0 + 300.0
            )
            if lease is not None:
                with lock:
                    acquired.append(lease)

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(8)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        assert len(acquired) == 2


# --------------------------------------------------------------- Gates tests


def test_scope_helpers_and_sibling_overlaps():
    assert norm_path("./src/foo/**") == "src/foo"
    assert norm_path("../escape") is None
    assert norm_path("/abs/path") is None
    assert scope_overlaps(["src/a"], ["src/a/sub.py"]) == [("src/a", "src/a/sub.py")]

    conflicts = detect_sibling_scope_overlaps(
        {
            "hyp-a": ["src/a/"],
            "hyp-b": ["src/b/"],
            "hyp-c": ["src/a/sub.py"],
        }
    )
    assert len(conflicts) == 1
    assert conflicts[0]["left"] == "hyp-a"
    assert conflicts[0]["right"] == "hyp-c"

    ok = validate_writable_paths(
        ["src/a/mod.py", "docs/notes.md"],
        ["src/a/"],
        goal_prefixes=["src/"],
        bookkeeping_paths=["docs/"],
    )
    assert ok.passed is True

    bad = validate_writable_paths(
        ["src/b/other.py"],
        ["src/a/"],
        goal_prefixes=["src/"],
    )
    assert bad.passed is False
    assert any("outside writable_paths" in f for f in bad.failures)


def test_banned_paths_read_bans_symlinks_and_commands(tmp_path, monkeypatch):
    read_bans = tmp_path / "READ_BANS.md"
    read_bans.write_text(
        "| Path | Reason |\n|---|---|\n"
        "| `directions/eval/units/PILOT-*` | sealed pilot units |\n"
        "| `directions/eval/units/TRUTH-*` | ground-truth points |\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONJECTRA_READ_BANS", str(read_bans))
    bans = load_read_bans()
    assert len(bans) > 0
    for safe in ("conjectra/types.py", "src/model.py", "tests/test_foo.py", "docs/REPORT.md"):
        assert banned_path(safe) is False

    for bad in (
        "holdout_out/metrics.json",
        "frozen_baseline/config.yaml",
        "/home/user/sealed_eval/batch.json",
        "sub/lockbox/oracle.json",
        "prereg_r1.json",
        "~/.codex/memories/note.md",
        "directions/eval/units/PILOT-SCALE-01/run.log",
        "directions/eval/units/TRUTH-20260101-01/points/p0.json",
    ):
        assert banned_path(bad) is True

    # Symlink parent into a banned directory must still be caught (abs and relative)
    banned_target = tmp_path / "lockbox"
    banned_target.mkdir()
    secret_file = banned_target / "oracle.txt"
    secret_file.write_text("42", encoding="utf-8")
    link_dir = tmp_path / "innocent_link"
    link_dir.symlink_to(banned_target)
    assert banned_path(str(link_dir / "oracle.txt")) is True

    # Regression: CWD inside a banned directory (e.g. .../protected) must not pollute
    # CLI flags (-m, -q, &&) or repo-relative paths (src/m0.py, tests/).
    protected_cwd = tmp_path / "harness" / "protected"
    (protected_cwd / "tests").mkdir(parents=True)
    monkeypatch.chdir(protected_cwd)
    rel_ok = check_no_banned_access(
        paths=["src/m0.py", "tests/"],
        command="python3 -m pytest -q tests/ && echo ok",
        extra_bans=["/protected", "protected"],
    )
    assert rel_ok.passed is True, rel_ok.failures

    # Direct reference to the banned directory itself is still rejected
    abs_bad = check_no_banned_access(
        command=f"cat {protected_cwd}/foo.py",
        extra_bans=["/protected", "protected"],
    )
    assert abs_bad.passed is False

    cmd_res = check_no_banned_access(
        command="cat holdout_out/scores.json",
        branch_name="main",
    )
    assert cmd_res.passed is False
    assert len(cmd_res.failures) >= 2


def test_assertion_non_weakening_and_ids_and_chip_checks():
    weak_diff = "\n".join(
        [
            "diff --git a/tests/test_a.py b/tests/test_a.py",
            "--- a/tests/test_a.py",
            "+++ b/tests/test_a.py",
            "@@ -1,3 +1,3 @@",
            "-    assert result == 42",
            "+    # assert result == 42",
            "+    pytest.skip('temporarily disabled')",
        ]
    )
    weak_res = check_assertion_non_weakening(weak_diff)
    assert weak_res.passed is False
    assert len(weak_res.failures) == 3

    ok_diff = "\n".join(
        [
            "diff --git a/tests/test_a.py b/tests/test_a.py",
            "--- a/tests/test_a.py",
            "+++ b/tests/test_a.py",
            "@@ -1,2 +1,2 @@",
            "-    assert result == 42",
            "+    assert result == 43",
        ]
    )
    assert check_assertion_non_weakening(ok_diff).passed is True

    assert check_no_internal_ids("x = 1  # ref TICKET-1234").passed is False
    assert check_no_internal_ids("class HypothesisNode:\n    pass").passed is True
    assert check_no_chip_specialization(
        "target = 'device_a7'", extra_patterns=(re.compile(r"\bdevice_a7\b"),)
    ).passed is False
    assert check_no_chip_specialization("target = 'generic'").passed is True


def test_raw_artifacts_and_metrics_validation(tmp_path):
    assert finite_number(1.5) is True
    assert finite_number(True) is False
    assert finite_number(math.nan) is False
    assert finite_number(math.inf) is False

    bad_prose_receipt = _sample_receipt(artifact_paths=[], metrics={})
    res_prose = check_raw_artifacts_and_evidence(bad_prose_receipt)
    assert res_prose.passed is False

    bad_metric_receipt = _sample_receipt(
        metrics={"p99_ms": {"value": math.nan, "denominator": 0, "stage_rung": "BAD"}}
    )
    res_metric = check_receipt_metrics(bad_metric_receipt)
    assert res_metric.passed is False
    assert len(res_metric.failures) == 3

    raw_art = tmp_path / "screen.json"
    raw_art.write_text('{"p99_ms": 78.5}', encoding="utf-8")
    good_receipt = _sample_receipt(artifact_paths=[str(raw_art)])
    assert (
        check_raw_artifacts_and_evidence(
            good_receipt, require_existing_artifacts=True, require_tests=True
        ).passed
        is True
    )


def test_git_real_path_and_composite_cpu_gate(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args: str) -> str:
        proc = subprocess.run(
            ["git", "-C", str(repo), *args],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    git("init")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Tester")
    (repo / "src").mkdir()
    (repo / "docs").mkdir()
    (repo / "src" / "app.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "base")
    base_sha = git("rev-parse", "HEAD")

    # Doc-only commit should fail check_non_doc_code_diff
    (repo / "docs" / "notes.md").write_text("# notes\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "doc only")
    doc_sha = git("rev-parse", "HEAD")
    assert check_non_doc_code_diff(repo, base_sha, doc_sha).passed is False

    # Valid code commit on feature branch should pass run_cpu_gate
    git("checkout", "-b", "hyp/opt-1")
    (repo / "src" / "app.py").write_text("def run():\n    return 2\n", encoding="utf-8")
    git("add", ".")
    git("commit", "-m", "code change")
    head_sha = git("rev-parse", "HEAD")

    assert check_clean_tree(repo).passed is True
    assert check_real_path_commits(repo, base_sha, head_sha).passed is True
    assert check_real_path_commits(repo, head_sha, base_sha).passed is False

    art = tmp_path / "artifact.log"
    art.write_text("PASS\n", encoding="utf-8")
    receipt = _sample_receipt(artifact_paths=[str(art)])
    receipt.base_sha = base_sha
    receipt.head_sha = head_sha

    gate_ok = run_cpu_gate(
        repo,
        base_sha,
        head_sha,
        writable_paths=["src/"],
        bookkeeping_paths=["docs/"],
        branch_name="hyp/opt-1",
        receipt=receipt,
    )
    assert gate_ok.passed is True

    # Out-of-scope write fails composite gate
    gate_oob = run_cpu_gate(
        repo,
        base_sha,
        head_sha,
        writable_paths=["tests/"],
        branch_name="hyp/opt-1",
        receipt=receipt,
    )
    assert gate_oob.passed is False
    assert any("outside writable_paths" in f for f in gate_oob.failures)

    # Dirty tree fails clean check
    (repo / "src" / "untracked.py").write_text("x = 1\n", encoding="utf-8")
    assert check_clean_tree(repo).passed is False
