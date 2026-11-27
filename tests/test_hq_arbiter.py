"""Tests for HyperQueue runtime and cross-project resource arbiter."""

from __future__ import annotations

import datetime as dt
import fcntl
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from conjectra.arbiter import (
    ResourceArbiter,
    inspect_heavy_slots,
    read_cloud_slot_state,
    read_proc_start_time,
    validate_pool_policy,
    verify_process_identity,
)
from conjectra.hq_runtime import (
    DAGTaskSpec,
    HQRuntime,
    detect_host_resources,
    resource_spec_to_requests,
    to_hq_resource_name,
)
from conjectra.types import ProcessStatus, ResourceSpec, StageKind, StageTask


def _make_task(
    task_id: str,
    *,
    stage: StageKind = StageKind.CODE,
    ram_mb: int = 256,
    gpu: int = 0,
    llm_pool: str | None = None,
    cloud_slot: int = 0,
    hypothesis_id: str = "hyp-1",
) -> StageTask:
    return StageTask(
        task_id=task_id,
        idempotency_key=f"idem-{task_id}",
        goal_id="goal-1",
        hypothesis_id=hypothesis_id,
        stage=stage,
        deps=[],
        resources=ResourceSpec(
            cpus=1,
            ram_mb=ram_mb,
            gpu=gpu,
            llm_pool=llm_pool,
            cloud_slot=cloud_slot,
            wall_timeout_s=30,
        ),
    )


def test_hq_ephemeral_dag_hot_insert_callback_and_cancel(tmp_path: Path) -> None:
    arr_home = tmp_path / "arr_home"
    sock_path = arr_home / "reactor.sock"
    arr_home.mkdir(parents=True, exist_ok=True)

    events: list[dict] = []
    stop_srv = threading.Event()
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(str(sock_path))
    srv.listen(8)
    srv.settimeout(0.2)

    def _listen() -> None:
        while not stop_srv.is_set():
            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            with conn:
                data = conn.recv(4096).decode("utf-8", errors="ignore")
                for line in data.splitlines():
                    if line.strip():
                        events.append(json.loads(line))

    listener = threading.Thread(target=_listen, daemon=True)
    listener.start()

    rt = HQRuntime(
        arr_home=arr_home,
        worker_resources={
            "cpus": 4,
            "ram_mb": 4096,
            "gpu": 0,
            "llm_gemini": 2,
            "llm_claude": 1,
            "eda_sim": 1,
            "cloud_slot": 0,
        },
        reactor_socket=sock_path,
    )
    try:
        boot = rt.ensure_running()
        assert boot["server"] is not None
        assert len(boot["workers"]) >= 1
        assert rt.server_info()["version"].startswith("v")

        # 1. Native DAG submission via submit_dag (task B depends on task A)
        marker_file = tmp_path / "dag_order.log"
        dag = rt.submit_dag(
            [
                DAGTaskSpec(
                    task_id="dag-a",
                    command=["/bin/bash", "-c", f"sleep 0.15 && echo A >> {marker_file}"],
                    resources=ResourceSpec(cpus=1, ram_mb=128, llm_pool="llm_gemini", wall_timeout_s=30),
                    priority=10,
                ),
                DAGTaskSpec(
                    task_id="dag-b",
                    deps=["dag-a"],
                    command=["/bin/bash", "-c", f"echo B >> {marker_file}"],
                    resources=ResourceSpec(cpus=1, ram_mb=128, wall_timeout_s=30),
                    priority=5,
                ),
            ],
            name="test-dag",
        )
        dag_info = rt.wait_job(dag.hq_job_id, timeout_s=30.0)
        assert dag_info is not None
        assert dag_info["info"]["task_stats"]["finished"] == 2
        assert marker_file.read_text(encoding="utf-8").splitlines() == ["A", "B"]

        # 2. Hot-insertion: parent -> child (via deps=[parent_jid]) + immediate callback
        hot_log = tmp_path / "hot.log"
        parent_jid = rt.submit_task(
            ["/bin/bash", "-c", f"sleep 0.3 && echo PARENT >> {hot_log}"],
            task_id="hot-parent",
            resources=ResourceSpec(cpus=1, ram_mb=128, wall_timeout_s=30),
        )
        child_jid = rt.submit_task(
            ["/bin/bash", "-c", f"echo CHILD >> {hot_log}"],
            task_id="hot-child",
            deps=[parent_jid],
            resources=ResourceSpec(cpus=1, ram_mb=128, wall_timeout_s=30),
        )
        child_info = rt.wait_job(child_jid, timeout_s=30.0)
        assert child_info is not None
        assert child_info["info"]["task_stats"]["finished"] == 1
        assert hot_log.read_text(encoding="utf-8").splitlines() == ["PARENT", "CHILD"]

        # Verify reactor socket received immediate exit notifications
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and len(events) < 4:
            time.sleep(0.05)
        seen_tids = {e["task_id"] for e in events}
        assert {"dag-a", "dag-b", "hot-parent", "hot-child"}.issubset(seen_tids)

        # 3. Job cancellation (running task + pending hot-inserted dependent)
        slow_jid = rt.submit_task(
            ["/bin/sleep", "30"],
            task_id="slow-parent",
            resources=ResourceSpec(cpus=1, ram_mb=128, wall_timeout_s=60),
        )
        dep_canceled_jid = rt.submit_task(
            ["/bin/echo", "should_never_run"],
            task_id="dep-cancel",
            deps=[slow_jid],
            resources=ResourceSpec(cpus=1, ram_mb=128, wall_timeout_s=30),
        )
        cancel_res = rt.cancel_jobs([slow_jid, dep_canceled_jid], reason="pruned_branch")
        assert cancel_res[slow_jid] is True
        assert cancel_res[dep_canceled_jid] is True
        canceled_flag, _ = rt.is_pending_canceled_or_dep_failed(dep_canceled_jid)
        assert canceled_flag is True
        assert any(j["id"] == slow_jid for j in rt.job_list(all_jobs=True))
    finally:
        stop_srv.set()
        srv.close()
        rt.stop()


def test_resource_alias_and_host_detection() -> None:
    assert to_hq_resource_name("ram_mb") == "rammb"
    assert to_hq_resource_name("llm_gemini") == "llmgemini"
    assert to_hq_resource_name("test_sim") == "testsim"
    caps = detect_host_resources(cpus=8, ram_mb=16384, gpu=0)
    assert caps["cpus"] == 8
    assert caps["ram_mb"] == 16384
    assert caps["gpu"] == 0
    assert caps["test_sim"] == 2

    default_caps = detect_host_resources()
    assert 1 <= default_caps["cpus"] <= 8
    assert default_caps["test_sim"] == 2

    _, code_reqs = resource_spec_to_requests(ResourceSpec(cpus=1, ram_mb=128), stage=StageKind.CODE)
    assert "testsim" not in code_reqs
    _, gate_reqs = resource_spec_to_requests(ResourceSpec(cpus=1, ram_mb=128), stage=StageKind.CPU_GATE)
    assert gate_reqs["testsim"] == 1
    _, exp_reqs = resource_spec_to_requests(ResourceSpec(cpus=1, ram_mb=128), stage=StageKind.EXP_CONFIRM)
    assert exp_reqs["testsim"] == 1
    _, explicit_reqs = resource_spec_to_requests({"cpus": 1, "test_sim": 1})
    assert explicit_reqs["testsim"] == 1


def test_default_mem_floor_and_testsim_hq_guard(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Default ResourceArbiter enforces 10 GiB (10240 MB) floor on heavy/test/sim stages
    monkeypatch.delenv("CONJECTRA_MIN_AVAIL_RAM_MB", raising=False)
    arbiter = ResourceArbiter(
        quotas_path=tmp_path / "missing_q.json",
        ledger_path=tmp_path / "missing_l.json",
        gemini_quota_path=tmp_path / "missing_v.json",
        disk_mounts=[str(tmp_path)],
    )
    monkeypatch.setattr("conjectra.arbiter.read_mem_available_mb", lambda: 8192)
    monkeypatch.setattr("conjectra.arbiter.read_load1", lambda: 1.0)
    gate_task = _make_task("t-gate-low-mem", stage=StageKind.CPU_GATE, ram_mb=256)
    dec_gate = arbiter.can_admit(gate_task, project="projalpha")
    assert dec_gate.allowed is False
    assert dec_gate.reason == "MEM_BELOW_10GB_WAIT: MemAvailable=8192MB < 10240MB"

    code_task = _make_task("t-code-ok", stage=StageKind.CODE, ram_mb=256)
    dec_code = arbiter.can_admit(code_task, project="projalpha")
    assert dec_code.allowed is True
    monkeypatch.undo()

    # HQ worker started with default detect_host_resources() registers testsim=sum(2) and cpus<=8
    arr_home = tmp_path / "arr_home_guard"
    rt = HQRuntime(arr_home=arr_home)
    try:
        boot = rt.ensure_running()
        wrk_res = {r["name"]: r for r in boot["workers"][0]["configuration"]["resources"]["resources"]}
        assert "testsim" in wrk_res
        assert wrk_res["testsim"]["size"] == 20000
        cpus_res = wrk_res["cpus"]
        if cpus_res.get("kind") == "range":
            cpu_cnt = int(cpus_res["end"]) - int(cpus_res["start"]) + 1
        else:
            cpu_cnt = len(cpus_res.get("values") or [])
        assert 1 <= cpu_cnt <= 8

        jid = rt.submit_task(
            ["/bin/echo", "testsim_ok"],
            task_id="gate-testsim",
            stage=StageKind.CPU_GATE,
            resources=ResourceSpec(cpus=1, ram_mb=128, wall_timeout_s=30),
        )
        info = rt.wait_job(jid, timeout_s=30.0)
        assert info is not None
        assert info["info"]["task_stats"]["finished"] == 1
    finally:
        rt.stop()


def test_arbiter_disk_ram_and_policy_fuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("conjectra.arbiter.read_load1", lambda: 1.0)
    ledger = tmp_path / "ledger.json"
    valley_cst = dt.datetime(2026, 11, 26, 2, 0, tzinfo=dt.timezone(dt.timedelta(hours=8)))
    valley_day = valley_cst.strftime("%Y-%m-%d")
    ledger.write_text(json.dumps({"days": {valley_day: {"costCny": 48.5}}}), encoding="utf-8")

    quotas = tmp_path / "quotas.json"
    quotas.write_text(
        json.dumps(
            {
                "claude/acc1": {
                    "quota": {"plan": "max", "windows": [{"name": "Claude 7 day", "used": 96.0}]}
                },
                "deepseek/acc1": {
                    "quota": {"plan": "pro", "windows": [{"name": "7 day", "used": 20.0}]}
                },
            }
        ),
        encoding="utf-8",
    )
    vg = tmp_path / "gemini-quota.json"
    vg.write_text(json.dumps({"pools": {"gemini": {"weekly": {"remaining_fraction": 0.7}}}}), encoding="utf-8")

    arbiter = ResourceArbiter(
        quotas_path=quotas,
        ledger_path=ledger,
        gemini_quota_path=vg,
        disk_mounts=[str(tmp_path)],
        disk_max_pct=85.0,
        min_ram_floor_mb=1024,
    )

    # Policy checks
    assert validate_pool_policy(pool="gemini:claude")[0] is False
    assert validate_pool_policy(pool="claude", model="claude-sonnet-5-5", effort="high")[0] is False
    assert validate_pool_policy(pool="claude", model="claude-sonnet-5-5", effort="medium")[0] is True
    monkeypatch.setenv("CONJECTRA_HOST_POOL_BLOCKS", "remote=llm_gemini")
    assert validate_pool_policy(pool="gemini", model="gemini-pro", host="remote")[0] is False
    assert validate_pool_policy(pool="gemini", model="gemini-pro", host="local")[0] is True

    # Valley CNY >= 48 fuse
    valley_task = _make_task("t-valley", llm_pool="deepseek-local")
    dec = arbiter.can_admit(valley_task, project="projalpha", now_cst=valley_cst)
    assert dec.allowed is False
    assert "VALLEY_FUSE_BLOCKED" in dec.reason

    # 7d Claude >= 95% falls back to healthy pool
    claude_task = _make_task("t-claude", llm_pool="llm_claude")
    dec_fb = arbiter.can_admit(claude_task, project="projalpha")
    assert dec_fb.allowed is True
    assert dec_fb.fallback_pool == "llm_gemini"

    # Disk >= 85% fuse
    from types import SimpleNamespace

    with monkeypatch.context() as mp_disk:
        mp_disk.setattr(
            "conjectra.arbiter.shutil.disk_usage",
            lambda _: SimpleNamespace(total=100 * 1024**3, used=90 * 1024**3, free=10 * 1024**3),
        )
        heavy_task = _make_task("t-heavy", stage=StageKind.EXP_CONFIRM, ram_mb=512)
        dec_disk = arbiter.can_admit(heavy_task, project="projbeta")
        assert dec_disk.allowed is False
        assert "DISK_FULL" in dec_disk.reason

    # RAM floor fuse
    huge_ram = _make_task("t-oom", ram_mb=10_000_000)
    dec_ram = arbiter.can_admit(huge_ram, project="projalpha")
    assert dec_ram.allowed is False
    assert "LOW_RAM" in dec_ram.reason

    # Fair-share yield when another project is waiting with 0 active slots
    cpu_task = _make_task("t-fair", stage=StageKind.CPU_GATE, ram_mb=128)
    dec_fair = arbiter.can_admit(
        cpu_task,
        project="projalpha",
        active_by_project={"projalpha": 3, "projbeta": 0},
        queued_projects=["projalpha", "projbeta"],
        total_slots=4,
    )
    assert dec_fair.allowed is False
    assert "FAIR_SHARE_YIELD" in dec_fair.reason


def test_arbiter_retry_bounds_circuit_breaker_and_interop(tmp_path: Path) -> None:
    arbiter = ResourceArbiter(
        quotas_path=tmp_path / "missing_q.json",
        ledger_path=tmp_path / "missing_l.json",
        gemini_quota_path=tmp_path / "missing_v.json",
        disk_mounts=[str(tmp_path)],
        max_infra_retries=2,
        max_gate_repairs=1,
        circuit_breaker_threshold=3,
    )

    # Bounded infra retries (max 2) + lane-local circuit breaker (>=3 consecutive)
    r1 = arbiter.record_failure("lane-a", failure_kind=ProcessStatus.ENV_ERROR, current_pool="llm_gemini")
    assert r1.should_retry is True and r1.next_pool == "llm_claude"
    r2 = arbiter.record_failure("lane-a", failure_kind=ProcessStatus.TIMEOUT, current_pool="llm_claude")
    assert r2.should_retry is True and r2.next_pool == "llm_deepseek"
    r3 = arbiter.record_failure("lane-a", failure_kind=ProcessStatus.ENV_ERROR, current_pool="llm_deepseek")
    assert r3.should_retry is False and r3.lane_tripped is True

    # Lane-local: lane-a tripped, lane-b unaffected
    assert arbiter.is_lane_open("lane-a") is False
    assert arbiter.is_lane_open("lane-b") is True

    # Gate repair bounded at max 1
    g1 = arbiter.record_failure("lane-b", failure_kind="FAIL_GATE")
    assert g1.should_retry is True
    g2 = arbiter.record_failure("lane-b", failure_kind="FAIL_GATE")
    assert g2.should_retry is False

    # Read-only heavy slot lock inspection without disturbing live flock
    lock_dir = tmp_path / "locks"
    lock_dir.mkdir()
    lock_file = lock_dir / "heavy_slot_0.lock"
    fd = os.open(str(lock_file), os.O_CREAT | os.O_RDWR, 0o664)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        os.write(fd, json.dumps({"slot": 0, "pid": os.getpid(), "cmd": "sim"}).encode("utf-8"))
        snap = inspect_heavy_slots(lock_dir, max_slots=4)
        assert snap.busy_slots == 1
        assert snap.holders[0]["pid"] == os.getpid()
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)

    # Read-only external cloud-GPU slot state adapter
    cloud_file = tmp_path / "cloud_slot_state_live.json"
    cloud_file.write_text(json.dumps({"status": "ready", "available_slots": 2, "active_jobs": 1}), encoding="utf-8")
    cloud = read_cloud_slot_state([tmp_path])
    assert cloud.available_slots == 2
    assert cloud.stale is False

    # PID + /proc/<pid>/stat starttime crash recovery
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)"])
    try:
        parsed = read_proc_start_time(proc.pid)
        assert parsed is not None
        _, start_t = parsed
        assert verify_process_identity(proc.pid, start_t).alive is True
        recycled = verify_process_identity(proc.pid, start_t + 999)
        assert recycled.alive is False and recycled.recycled is True
        plan = arbiter.recover_stale_leases(
            [
                {"lease_id": "live", "pid": proc.pid, "proc_start_time": start_t, "expires_ts": time.time() + 60},
                {"lease_id": "recycled", "pid": proc.pid, "proc_start_time": start_t + 1, "expires_ts": time.time() + 60},
                {"lease_id": "dead", "pid": 99999999, "proc_start_time": 123, "expires_ts": time.time() + 60},
            ]
        )
        assert [x["lease_id"] for x in plan["still_alive"]] == ["live"]
        assert {x["lease_id"] for x in plan["stale_to_requeue"]} == {"recycled", "dead"}
    finally:
        proc.terminate()
        proc.wait(timeout=5)


def test_p1_07_r2_arbiter_excludes_task_own_budget_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conjectra.state_db import StateDB
    from conjectra.types import ResearchGoal

    monkeypatch.setattr("conjectra.arbiter.read_mem_available_mb", lambda *_a, **_kw: 32768)
    monkeypatch.setattr("conjectra.arbiter.read_load1", lambda *_a, **_kw: 1.0)
    db = StateDB(tmp_path / "arr_home_budget")
    db.save_goal(
        ResearchGoal(
            goal_id="g-self-res",
            project="demo",
            title="Self reservation check",
            direction="max",
            primary_metric="score",
            baseline_value=1.0,
            baseline_sha="HEAD",
            repo_path=str(tmp_path),
            kill_criteria=[],
            banned_paths=[],
            writable_prefixes=["src/"],
            max_tokens_budget=7000,
        )
    )
    arb = ResourceArbiter(
        quotas_path=tmp_path / "q.json",
        ledger_path=tmp_path / "l.json",
        gemini_quota_path=tmp_path / "v.json",
        disk_mounts=[],
    )
    # Seed 2,000 settled tokens from an earlier task + 4,000 active RESERVED tokens on t-self:a1 (total committed = 6,000)
    db.settle_goal_budget(
        "g-self-res",
        "task:t-earlier:a1",
        entity_kind="stage_task",
        entity_id="t-earlier",
        attempt=1,
        model="claude-sonnet-5-5",
        input_tokens=1500,
        output_tokens=500,
        state="SETTLED",
    )
    ok_r, comm, _ = db.reserve_goal_budget(
        "g-self-res",
        "task:t-self:a1",
        entity_kind="stage_task",
        entity_id="t-self",
        attempt=1,
        model="claude-sonnet-5-5",
        reserve_tokens=4000,
    )
    assert ok_r is True and comm == 6000

    t_self = StageTask(
        task_id="t-self",
        idempotency_key="k-self",
        goal_id="g-self-res",
        hypothesis_id="h1",
        stage=StageKind.CODE,
        attempt=1,
        deps=[],
        resources=ResourceSpec(llm_pool="llm_claude", ram_mb=1024),
    )
    t_other = StageTask(
        task_id="t-other",
        idempotency_key="k-other",
        goal_id="g-self-res",
        hypothesis_id="h2",
        stage=StageKind.CODE,
        attempt=1,
        deps=[],
        resources=ResourceSpec(llm_pool="llm_claude", ram_mb=1024),
    )

    assert arb.resolve_goal_tokens_for_task(db, t_self) == 2000
    exhausted_self, used_self = arb.is_goal_budget_exhausted_for_task(db, t_self, max_tokens_budget=5000)
    assert exhausted_self is False and used_self == 2000

    admit_self = arb.can_admit(
        t_self,
        project="demo",
        db=db,
        tokens_used_for_goal=6000,
        max_tokens_budget=5000,
    )
    assert admit_self.allowed is True

    # A distinct task t_other MUST count t-self's 4,000 reservation (2,000 + 4,000 = 6,000 >= 5,000) and block!
    admit_other = arb.can_admit(
        t_other,
        project="demo",
        db=db,
        tokens_used_for_goal=6000,
        max_tokens_budget=5000,
    )
    assert admit_other.allowed is False
    assert admit_other.reason.startswith("TOKEN_BUDGET_EXCEEDED:6000>=5000")
    db.close()


def test_p1_04_r3_hq_query_job_state_three_way_disambiguation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hq = HQRuntime(tmp_path / "arr_home_qjs", hq_bin="/bin/true")

    # 1. FOUND: live HQ server returns valid JSON array or dict matching job_id
    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps([{"info": {"id": 77, "name": "tsk-77", "task_stats": {"running": 1}}}]),
            stderr="",
        ),
    )
    found = hq.query_job_state(77)
    assert found["lookup_status"] == "FOUND"
    assert found["job"] is not None and found["job"]["info"]["id"] == 77
    assert found["error"] == ""
    assert hq.job_info(77) == found["job"]

    # 2. NOT_FOUND: live reachable HQ server returns rc==0 + empty JSON list '[]\n' (real HQ v0.27.0 behavior)
    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(
            args,
            0,
            stdout="[]\n",
            stderr="2026-11-27T01:01:17Z ERROR Job 99999 not found\n",
        ),
    )
    not_found = hq.query_job_state(99999)
    assert not_found["lookup_status"] == "NOT_FOUND"
    assert not_found["job"] is None
    assert hq.job_info(99999) is None

    # 3. QUERY_ERROR cases MUST NEVER masquerade as NOT_FOUND!
    # 3a. Nonzero exit code (server unreachable / RPC error)
    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(
            args, 1, stdout="", stderr="ERROR Cannot connect to server"
        ),
    )
    err_rc = hq.query_job_state(77)
    assert err_rc["lookup_status"] == "QUERY_ERROR"
    assert err_rc["job"] is None
    assert "Cannot connect" in err_rc["error"]
    assert hq.job_info(77) is None

    # 3b. Subprocess TimeoutExpired / OSError
    def _raise_timeout(args: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args, timeout=10.0)

    monkeypatch.setattr(hq, "_run_hq", _raise_timeout)
    err_to = hq.query_job_state(77)
    assert err_to["lookup_status"] == "QUERY_ERROR"
    assert err_to["job"] is None
    assert "TimeoutExpired" in err_to["error"]

    # 3c. Empty stdout or malformed JSON despite rc == 0
    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(args, 0, stdout="", stderr=""),
    )
    assert hq.query_job_state(77)["lookup_status"] == "QUERY_ERROR"

    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(args, 0, stdout="{truncated-json", stderr=""),
    )
    assert hq.query_job_state(77)["lookup_status"] == "QUERY_ERROR"

    # 3d. Mismatched job ID in payload or invalid input ID
    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(
            args, 0, stdout=json.dumps([{"info": {"id": 88}}]), stderr=""
        ),
    )
    assert hq.query_job_state(77)["lookup_status"] == "QUERY_ERROR"
    assert hq.query_job_state("not-an-int")["lookup_status"] == "QUERY_ERROR"  # type: ignore[arg-type]


def test_p1_04_r4_hq_query_job_list_fail_closed_disambiguation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hq = HQRuntime(tmp_path / "arr_home_qjl", hq_bin="/bin/true")

    # 1. OK (non-empty list and authoritative empty list `[]` with rc == 0)
    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(
            args,
            0,
            stdout=json.dumps([{"id": 12, "name": "dec-r1-a1-deadbeef"}]),
            stderr="",
        ),
    )
    ok_list = hq.query_job_list(all_jobs=True)
    assert ok_list["lookup_status"] == "OK"
    assert len(ok_list["jobs"]) == 1 and ok_list["jobs"][0]["name"] == "dec-r1-a1-deadbeef"
    assert hq.job_list(all_jobs=True) == ok_list["jobs"]

    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(args, 0, stdout="[]\n", stderr=""),
    )
    ok_empty = hq.query_job_list(all_jobs=True)
    assert ok_empty["lookup_status"] == "OK"
    assert ok_empty["jobs"] == []

    # 2. QUERY_ERROR on nonzero exit, TimeoutExpired, empty stdout, malformed JSON, non-list JSON, or server-down stderr
    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(
            args, 1, stdout="", stderr="ERROR Cannot connect to server"
        ),
    )
    err_rc = hq.query_job_list(all_jobs=True)
    assert err_rc["lookup_status"] == "QUERY_ERROR"
    assert err_rc["jobs"] == []
    assert "Cannot connect" in err_rc["error"]

    def _raise_to(args: list[str], **_kw: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=args, timeout=10.0)

    monkeypatch.setattr(hq, "_run_hq", _raise_to)
    assert hq.query_job_list(all_jobs=True)["lookup_status"] == "QUERY_ERROR"

    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(args, 0, stdout="", stderr=""),
    )
    assert hq.query_job_list(all_jobs=True)["lookup_status"] == "QUERY_ERROR"

    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(args, 0, stdout="{bad-json", stderr=""),
    )
    assert hq.query_job_list(all_jobs=True)["lookup_status"] == "QUERY_ERROR"

    monkeypatch.setattr(
        hq,
        "_run_hq",
        lambda args, **_kw: subprocess.CompletedProcess(args, 0, stdout='{"not": "a_list"}', stderr=""),
    )
    assert hq.query_job_list(all_jobs=True)["lookup_status"] == "QUERY_ERROR"
