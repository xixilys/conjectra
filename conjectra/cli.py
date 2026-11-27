"""CLI entrypoint for conjectra (`conjectra`)."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from conjectra.controller import (
    ResearchController,
    _make_stage_task,
    get_owner_escalations,
    send_reactor_wake,
)
from conjectra.explore import import_proposals_json
from conjectra.goal_manager import GlobalGoalManager
from conjectra.hq_runtime import HQRuntime
from conjectra.interactive import (
    build_takeover_brief,
    get_overview,
    get_plans,
    get_reviews,
    get_subgoals,
    get_token_audit,
    inspect_state,
    intervene,
)
from conjectra.metrics import (
    compute_benchmark_report,
    load_baseline,
    render_latest_status_md,
    sync_status_mirror,
)
from conjectra.multiagent import run_boost, run_xswarm, run_xteamwork
from conjectra.state_db import StateDB, resolve_arr_home
from conjectra.types import ResearchGoal, StageKind
from conjectra.workload_matrix import route_workload, run_eval_matrix


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False))


def _resolve_git_head(repo_path: str) -> str:
    try:
        proc = subprocess.run(
            ["git", "-C", repo_path, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout.strip()
    except OSError:
        pass
    return "HEAD"


def _cmd_init_pilot(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    home.mkdir(parents=True, exist_ok=True)
    db = StateDB(home)
    try:
        hq_status: dict[str, Any] = {"started": False}
        if args.start_hq:
            hq = HQRuntime(home)
            hq_status = hq.ensure_running()
        _, baseline_src = load_baseline(
            args.baseline_path, arr_home=home, fetch_ssh=not args.no_ssh
        )
        _print_json(
            {
                "arr_home": str(home),
                "db_path": str(db.db_path),
                "events_path": str(db.events_path),
                "baseline_source": baseline_src,
                "hq": hq_status,
            }
        )
        return 0
    finally:
        db.close()


def _cmd_submit_goal(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    if args.file:
        raw = json.loads(Path(args.file).read_text(encoding="utf-8"))
        if "baseline_sha" not in raw or not raw["baseline_sha"]:
            raw["baseline_sha"] = _resolve_git_head(str(raw["repo_path"]))
        goal = ResearchGoal.from_dict(raw)
    else:
        missing = [
            f"--{k.replace('_', '-')}"
            for k in ("goal_id", "project", "title", "direction", "primary_metric", "repo_path")
            if getattr(args, k) is None
        ]
        if missing or args.baseline_value is None:
            if args.baseline_value is None:
                missing.append("--baseline-value")
            raise SystemExit(f"Missing required submit-goal options (or pass --file): {', '.join(missing)}")
        base_sha = args.baseline_sha or _resolve_git_head(args.repo_path)
        goal = ResearchGoal(
            goal_id=args.goal_id,
            project=args.project,
            title=args.title,
            direction=args.direction,
            primary_metric=args.primary_metric,
            baseline_value=float(args.baseline_value),
            baseline_sha=base_sha,
            repo_path=str(Path(args.repo_path).expanduser().resolve()),
            kill_criteria=list(args.kill_criterion or ["tests_passed == 0"]),
            banned_paths=list(args.banned_path or []),
            writable_prefixes=list(args.writable_prefix or ["."]),
            max_tokens_budget=int(args.max_tokens_budget),
            max_concurrent_branches=int(args.max_concurrent_branches),
            adapter_name=args.adapter,
            autonomy_mode=args.autonomy_mode,
            priority_weight=float(args.priority_weight),
            min_guaranteed_slots=int(args.min_guaranteed_slots),
            target_value=args.target_value,
            wall_budget_sec=int(args.wall_budget_sec),
        )

    db = StateDB(home)
    try:
        mgr = GlobalGoalManager(db)
        stored = mgr.create_or_update_goal(goal, status="ACTIVE")
    finally:
        db.close()
    woke = send_reactor_wake(home, {"event": "GOAL_SUBMITTED", "goal_id": stored.goal_id})
    _print_json({"goal": stored.to_dict(), "reactor_woke": woke})
    return 0


def _cmd_step(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    ctrl = ResearchController(
        home,
        execution_mode=args.mode,
        propose_mode=args.propose_mode,
        dry_run_llm=args.dry_run_llm,
    )
    try:
        summary = ctrl.step(goal_id=args.goal_id)
        _print_json(summary)
        return 0
    finally:
        ctrl.close()


def _cmd_daemon(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    pid_path = home / "controller.pid"
    if args.stop:
        if not pid_path.is_file():
            _print_json({"stopped": False, "reason": "no controller.pid"})
            return 0
        doc = json.loads(pid_path.read_text(encoding="utf-8"))
        pid = int(doc.get("pid", 0))
        if pid > 0:
            try:
                os.kill(pid, signal.SIGTERM)
                _print_json({"stopped": True, "pid": pid})
                return 0
            except ProcessLookupError:
                pid_path.unlink(missing_ok=True)
                _print_json({"stopped": False, "reason": "stale pid cleaned", "pid": pid})
                return 0
        return 0

    if args.background or getattr(args, "detach", False):
        cmd = [
            sys.executable,
            "-m",
            "conjectra.cli",
            "--arr-home",
            str(home),
            "daemon",
            "--mode",
            args.mode,
            "--propose-mode",
            args.propose_mode,
            "--sweep-interval",
            str(args.sweep_interval),
        ]
        if args.goal_id:
            cmd.extend(["--goal-id", args.goal_id])
        if args.dry_run_llm:
            cmd.append("--dry-run-llm")
        if args.max_seconds is not None:
            cmd.extend(["--max-seconds", str(args.max_seconds)])
        if getattr(args, "notify_cmd", None):
            cmd.extend(["--notify-cmd", args.notify_cmd])
        log_dir = home / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_f = (log_dir / "controller-daemon.log").open("ab")
        d_env = os.environ.copy()
        d_root = str(Path(__file__).resolve().parents[1])
        d_env["PYTHONPATH"] = d_root + (os.pathsep + d_env["PYTHONPATH"] if d_env.get("PYTHONPATH") else "")
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=str(home),
                start_new_session=True,
                env=d_env,
            )
        finally:
            log_f.close()
        deadline = time.monotonic() + 5.0
        sock_ready = False
        while time.monotonic() < deadline:
            if (home / "reactor.sock").exists():
                sock_ready = True
                break
            time.sleep(0.05)
        _print_json(
            {
                "started": True,
                "detached": bool(getattr(args, "detach", False) or args.background),
                "pid": proc.pid,
                "reactor_sock_ready": sock_ready,
                "notify_cmd": getattr(args, "notify_cmd", None),
            }
        )
        return 0

    try:
        ctrl = ResearchController(
            home,
            execution_mode=args.mode,
            propose_mode=args.propose_mode,
            dry_run_llm=args.dry_run_llm,
            sweep_interval_s=args.sweep_interval,
        )
    except Exception as exc:
        fatal_rec = {
            "ts": time.time(),
            "pid": os.getpid(),
            "phase": "daemon_init",
            "goal_id": args.goal_id,
            "exc_type": type(exc).__name__,
            "error": str(exc),
        }
        with contextlib.suppress(OSError):
            log_dir = home / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            with (log_dir / "daemon_fatal.jsonl").open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(fatal_rec, sort_keys=True, ensure_ascii=False) + "\n")
        if ResearchController._is_fatal_authority_or_db_error(exc):
            _print_json(
                {
                    "daemon_failed": True,
                    "fatal": True,
                    "restarts": 0,
                    "exc_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            return 1
        raise
    max_rapid_restarts = 5
    rapid_window_s = 60.0
    rapid_crash_timestamps: list[float] = []
    restarts = 0
    try:
        while True:
            started_ts = time.monotonic()
            try:
                res = ctrl.serve_forever(goal_id=args.goal_id, max_seconds=args.max_seconds)
                res["supervisor_restarts"] = restarts
                if getattr(args, "notify_cmd", None):
                    res["notification"] = _notify_cmd(
                        args.notify_cmd, subject="conjectra daemon finished", payload=res
                    )
                _print_json(res)
                return 0
            except Exception as exc:
                fatal_rec = {
                    "ts": time.time(),
                    "pid": os.getpid(),
                    "phase": "daemon_supervisor",
                    "goal_id": args.goal_id,
                    "exc_type": type(exc).__name__,
                    "error": str(exc),
                }
                with contextlib.suppress(OSError):
                    log_dir = home / "logs"
                    log_dir.mkdir(parents=True, exist_ok=True)
                    with (log_dir / "daemon_fatal.jsonl").open("a", encoding="utf-8") as fh:
                        fh.write(json.dumps(fatal_rec, sort_keys=True, ensure_ascii=False) + "\n")
                is_fatal_db_err = hasattr(ctrl, "_is_fatal_authority_or_db_error") and ctrl._is_fatal_authority_or_db_error(exc)
                if is_fatal_db_err or args.max_seconds is not None or bool(getattr(ctrl, "_stop_requested", False)):
                    if is_fatal_db_err:
                        _print_json(
                            {
                                "daemon_failed": True,
                                "fatal": True,
                                "restarts": restarts,
                                "exc_type": type(exc).__name__,
                                "error": str(exc),
                            }
                        )
                        return 1
                    raise
                if "already running" in str(exc).lower():
                    raise
                now_m = time.monotonic()
                if now_m - started_ts >= rapid_window_s:
                    rapid_crash_timestamps = [now_m]
                else:
                    rapid_crash_timestamps = [
                        t for t in rapid_crash_timestamps if (now_m - t) <= rapid_window_s
                    ]
                    rapid_crash_timestamps.append(now_m)
                restarts += 1
                with contextlib.suppress(Exception):
                    ctrl._record_ledger_event(
                        "DAEMON_SUPERVISOR_RESTART",
                        goal_id=args.goal_id,
                        payload={
                            "restarts": restarts,
                            "rapid_count": len(rapid_crash_timestamps),
                            "exc_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
                if len(rapid_crash_timestamps) > max_rapid_restarts:
                    _print_json(
                        {
                            "daemon_failed": True,
                            "restarts": restarts,
                            "exc_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    )
                    return 1
                backoff_s = min(8.0, 1.0 * (2 ** min(len(rapid_crash_timestamps) - 1, 3)))
                time.sleep(backoff_s)
    finally:
        ctrl.close()


def _cmd_notify_exit(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    payload: dict[str, Any] = {
        "event": "task_exit",
        "task_id": args.task_id,
        "exit_code": int(args.exit_code),
        "hq_job_id": args.hq_job_id,
        "ts": time.time(),
    }
    if args.status:
        db = StateDB(home)
        try:
            task = db.get_task(args.task_id)
            if task is not None and task.status in ("READY", "DISPATCHED", "RUNNING"):
                db.transition_task_status(
                    task.task_id,
                    task.status,
                    args.status,
                    finished_ts=time.time(),
                )
        finally:
            db.close()
    woke = send_reactor_wake(home, payload)
    _print_json({"notified": woke, **payload})
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    db = StateDB(home)
    try:
        goals = [g for g in ([db.get_goal(args.goal_id)] if args.goal_id else db.list_goals()) if g is not None]
        summaries = [db.get_goal_summary(g.goal_id) for g in goals]
        escalations = get_owner_escalations(db, goal_id=args.goal_id)
        hq = HQRuntime(home)
        sinfo = hq.server_info()
        workers = hq.worker_list() if sinfo is not None else []
        payload = {
            "arr_home": str(home),
            "reactor_sock_exists": (home / "reactor.sock").exists(),
            "hq_server_running": sinfo is not None,
            "hq_workers": len([w for w in workers if w.get("ended") is None]),
            "goals": summaries,
            "escalations": escalations,
        }
        _print_json(payload)
        return 0
    finally:
        db.close()


def _cmd_board(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    db = StateDB(home)
    try:
        hyps = db.list_hypotheses(goal_id=args.goal_id)
        events = db.list_events(goal_id=args.goal_id, since_seq=args.since_seq, limit=args.limit)
        graveyard = db.list_graveyard(goal_id=args.goal_id)
        board_doc = {
            "hypotheses": [
                {
                    "hypothesis_id": h.hypothesis_id,
                    "goal_id": h.goal_id,
                    "kind": h.kind.value,
                    "fidelity": h.fidelity.value,
                    "status": h.status,
                    "verdict": h.scientific_verdict.value,
                    "evoi": h.priority_evoi,
                    "title": h.title,
                    "branch": h.branch_name,
                    "head_sha": h.head_sha,
                }
                for h in hyps
            ],
            "graveyard_count": len(graveyard),
            "events": events,
        }
        if args.format == "json":
            _print_json(board_doc)
        else:
            print(f"# Conjectra Board ({len(hyps)} hypotheses, {len(graveyard)} graveyard)")
            for h in board_doc["hypotheses"]:
                print(
                    f"- [{h['status']}/{h['verdict']}] {h['hypothesis_id']} "
                    f"({h['kind']}, {h['fidelity']}, evoi={h['evoi']:.2f}): {h['title']}"
                )
            print(f"\nRecent events ({len(events)}):")
            for ev in events[-15:]:
                print(f"  #{ev['seq']} {ev['event_type']} goal={ev.get('goal_id')} hyp={ev.get('hypothesis_id')}")
        return 0
    finally:
        db.close()


def _cmd_export_teamwork_brief(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    db = StateDB(home)
    try:
        goals = [g for g in ([db.get_goal(args.goal_id)] if args.goal_id else db.list_goals(status="ACTIVE")) if g is not None]
        if not goals:
            goals = db.list_goals()
        if not goals:
            raise SystemExit("No goals found in StateDB")
        goal = goals[0]
        hyps = db.list_hypotheses(goal_id=goal.goal_id)
        receipts = db.list_receipts(goal_id=goal.goal_id)
        graveyard = db.list_graveyard(goal_id=goal.goal_id)
        usage = db.get_goal_token_usage(goal.goal_id)
        escalations = get_owner_escalations(db, goal_id=goal.goal_id)
        brief = {
            "goal": goal.to_dict(),
            "budget": usage,
            "hypotheses": [h.to_dict() for h in hyps],
            "recent_receipts": [r.to_dict() for r in receipts[-20:]],
            "graveyard": graveyard,
            "escalations": escalations,
        }
        if args.format == "json":
            _print_json(brief)
        else:
            lines = [
                f"# Teamwork Brief: {goal.title} ({goal.goal_id})",
                f"- Project: `{goal.project}` | Metric: `{goal.primary_metric}` ({goal.direction}, baseline={goal.baseline_value})",
                f"- Tokens: {usage['total_tokens']} / {goal.max_tokens_budget}",
                f"- Writable prefixes: {', '.join(goal.writable_prefixes)}",
                f"- Hypotheses: {len(hyps)} | Receipts: {len(receipts)} | Graveyard: {len(graveyard)}",
            ]
            print("\n".join(lines))
        return 0
    finally:
        db.close()


def _cmd_import_proposals(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    raw_text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    inferred_goal_id = args.goal_id or (parsed.get("goal_id") if isinstance(parsed, dict) else None)
    db = StateDB(home)
    try:
        goal = db.get_goal(inferred_goal_id) if inferred_goal_id else None
        if goal is None and not inferred_goal_id:
            active = db.list_goals(status="ACTIVE")
            goal = active[0] if active else None
        if goal is None:
            raise SystemExit(f"Unknown or missing goal_id: {inferred_goal_id}")
        ready_cnt = len(db.list_ready_tasks(goal_id=goal.goal_id))
        batch = import_proposals_json(
            goal,
            parsed,
            db=db,
            ready_queue_count=ready_cnt,
            dispatchable_slots=goal.max_concurrent_branches,
            min_evoi=float(args.min_evoi),
            persist=True,
        )
        enqueued_ids: list[str] = []
        if not args.no_enqueue:
            for hyp in batch.accepted:
                if not db.list_tasks(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id):
                    t = _make_stage_task(goal, hyp, StageKind.CODE)
                    stored = db.enqueue_task(t)
                    enqueued_ids.append(stored.task_id)
        if enqueued_ids:
            send_reactor_wake(home, {"event": "PROPOSALS_IMPORTED", "goal_id": goal.goal_id})
        out = batch.to_dict()
        out["enqueued_task_ids"] = enqueued_ids
        _print_json(out)
        return 0
    finally:
        db.close()


def _cmd_benchmark_vs_baseline(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    db = StateDB(home)
    try:
        report = compute_benchmark_report(
            db,
            goal_id=args.goal_id,
            baseline_path=args.baseline_path,
            slots=args.slots,
            fetch_ssh=not args.no_ssh,
        )
        _print_json(report)
        return 0
    finally:
        db.close()


def _cmd_sync_status(args: argparse.Namespace) -> int:
    home = resolve_arr_home(args.arr_home)
    db = StateDB(home)
    try:
        report = compute_benchmark_report(
            db,
            goal_id=args.goal_id,
            baseline_path=args.baseline_path,
            fetch_ssh=not args.no_ssh,
        )
        escalations = get_owner_escalations(db, goal_id=args.goal_id)
        status_md = render_latest_status_md(db, report, escalations=escalations)
        (home / "LATEST_STATUS.md").write_text(status_md, encoding="utf-8")
        (home / "benchmark_latest.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if args.dry_run:
            _print_json(
                {
                    "mode": "dry_run",
                    "local_status_path": str(home / "LATEST_STATUS.md"),
                    "local_benchmark_path": str(home / "benchmark_latest.json"),
                }
            )
            return 0
        res = sync_status_mirror(status_md, report, target=args.target)
        _print_json(res)
        return 0
    finally:
        db.close()


def _read_spec_arg(spec_arg: str) -> str:
    if spec_arg.startswith("@"):
        return Path(spec_arg[1:]).expanduser().read_text(encoding="utf-8")
    return spec_arg


def _notify_cmd(
    target: str | None,
    *,
    subject: str,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Send a completion/status notification to a local JSONL file or a user-supplied command hook."""
    if not target:
        return {"notified": False}
    compact = json.dumps(dict(payload), sort_keys=True, ensure_ascii=False)
    if len(compact) > 2500:
        compact = compact[:2490] + "...}"

    if target.startswith(("file:", "/")) or target.startswith("./"):
        p = Path(target[5:] if target.startswith("file:") else target).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"subject": subject, "payload": dict(payload), "ts": time.time()}) + "\n")
        return {"notified": True, "transport": "file", "target": str(p)}

    from conjectra.multiagent import notify_hook

    res = notify_hook(target, subject, compact, msg_type="status", timeout_s=10)
    return {
        "notified": bool(res.get("sent")),
        "transport": res.get("transport"),
        "target": target,
        "rc": res.get("returncode"),
        "stderr": str(res.get("stderr") or "")[:240] if not res.get("sent") else "",
    }


def _launch_detached(args: argparse.Namespace, command_name: str) -> dict[str, Any]:
    """Spawn the current conjectra subcommand detached without `--detach` and return receipt JSON."""
    home = resolve_arr_home(getattr(args, "arr_home", None))
    log_dir = home / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{command_name}_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    result_path = log_dir / f"detach_{tag}.json"
    log_path = log_dir / f"detach_{tag}.log"

    raw_argv = list(getattr(args, "_raw_argv", []) or [])
    child_args = [tok for tok in raw_argv if tok != "--detach"]
    if "--arr-home" not in child_args and "--home" not in child_args:
        child_args = ["--arr-home", str(home), *child_args]
    cmd = [sys.executable, "-m", "conjectra.cli", *child_args]

    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    pkg_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    out_f = result_path.open("wb")
    err_f = log_path.open("wb")
    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=out_f,
            stderr=err_f,
            cwd=str(home),
            env=env,
            start_new_session=True,
        )
    finally:
        out_f.close()
        err_f.close()
    return {
        "detached": True,
        "command": command_name,
        "pid": proc.pid,
        "arr_home": str(home),
        "result_path": str(result_path),
        "log_path": str(log_path),
        "notify_cmd": getattr(args, "notify_cmd", None),
    }


def _open_cli_router_context(arr_home_arg: str | None) -> tuple[Path, StateDB | None, Any]:
    from conjectra.arbiter import ResourceArbiter
    from conjectra.model_router import ModelRouter

    raw_home = arr_home_arg or os.environ.get("CONJECTRA_HQ_HOME") or os.environ.get("CONJECTRA_HOME")
    home = resolve_arr_home(raw_home)
    db: StateDB | None = None
    if raw_home is not None or (home / "state.db").is_file():
        db = StateDB(home)
    router = ModelRouter(arbiter=ResourceArbiter(), db=db, arr_home=home)
    return home, db, router


def _cmd_boost(args: argparse.Namespace) -> int:
    if getattr(args, "detach", False):
        _print_json(_launch_detached(args, "boost"))
        return 0
    home, db, router = _open_cli_router_context(getattr(args, "arr_home", None))
    try:
        summary = run_boost(
            repo_path=args.repo_path,
            project=args.project,
            task_spec=_read_spec_arg(args.spec),
            acceptance_cmd=args.acceptance_cmd,
            writable_paths=list(args.writable_path),
            banned_paths=list(args.banned_path or []),
            builder_spec=args.builder,
            adversary_spec=args.adversary,
            max_rounds=args.max_rounds,
            timeout_s=args.timeout,
            in_place=args.in_place,
            arr_home=home,
            model_router=router,
        )
    finally:
        if db is not None:
            db.close()
    if getattr(args, "notify_cmd", None):
        summary["notification"] = _notify_cmd(
            args.notify_cmd, subject=f"conjectra boost {summary.get('status')}", payload=summary
        )
    _print_json(summary)
    return 0 if summary["status"] == "PASSED" else 1


def _cmd_xswarm(args: argparse.Namespace) -> int:
    if getattr(args, "detach", False):
        _print_json(_launch_detached(args, "xswarm"))
        return 0
    cfg = json.loads(Path(args.spec_file).expanduser().read_text(encoding="utf-8"))
    home, db, router = _open_cli_router_context(getattr(args, "arr_home", None))
    try:
        summary = run_xswarm(
            repo_path=cfg["repo_path"],
            project=cfg.get("project", "xswarm"),
            title=cfg.get("title", "Cross-provider xswarm"),
            spec_workers=cfg["workers"],
            verifier_spec=cfg.get("verifier"),
            verifier_cmd=cfg.get("verifier_cmd", "pytest -q"),
            max_ticks=int(cfg.get("max_ticks", 8)),
            stall_timeout_s=float(cfg.get("stall_timeout_s", 180.0)),
            timeout_s=int(cfg.get("timeout_s", 600)),
            use_hq=not args.no_hq,
            arr_home=home,
            model_router=router,
        )
    finally:
        if db is not None:
            db.close()
    if getattr(args, "notify_cmd", None):
        summary["notification"] = _notify_cmd(
            args.notify_cmd, subject=f"conjectra xswarm {summary.get('status')}", payload=summary
        )
    _print_json(summary)
    return 0 if summary["status"] == "PASSED" else 1


def _cmd_xteamwork(args: argparse.Namespace) -> int:
    if getattr(args, "detach", False):
        _print_json(_launch_detached(args, "xteamwork"))
        return 0
    cfg = json.loads(Path(args.spec_file).expanduser().read_text(encoding="utf-8"))
    home, db, router = _open_cli_router_context(getattr(args, "arr_home", None))
    try:
        summary = run_xteamwork(
            repo_path=cfg["repo_path"],
            project=cfg.get("project", "xteamwork"),
            title=cfg.get("title", "Cross-provider xteamwork"),
            phases=cfg["phases"],
            final_acceptance_cmd=cfg["final_acceptance_cmd"],
            banned_paths=list(cfg.get("banned_paths") or []),
            prescreener_spec=cfg.get("prescreener"),
            auditor_spec=cfg.get("auditor"),
            timeout_s=int(cfg.get("timeout_s", 600)),
            arr_home=home,
            model_router=router,
        )
    finally:
        if db is not None:
            db.close()
    if getattr(args, "notify_cmd", None):
        summary["notification"] = _notify_cmd(
            args.notify_cmd, subject=f"conjectra xteamwork {summary.get('status')}", payload=summary
        )
    _print_json(summary)
    return 0 if summary["status"] == "PASSED" else 1


def _cmd_route_workload(args: argparse.Namespace) -> int:
    scopes = [
        [p.strip() for p in item.split(",") if p.strip()]
        for item in (args.scope or [])
    ]
    res = route_workload(
        args.description,
        writable_scopes=scopes or None,
        readonly=args.readonly,
        interface_frozen=False if args.coupled_interface else None,
        milestone_phases=args.phases,
        estimated_minutes=args.minutes,
        arr_home=args.arr_home,
    )
    _print_json(res)
    return 0


def _cmd_eval_matrix(args: argparse.Namespace) -> int:
    if getattr(args, "detach", False):
        _print_json(_launch_detached(args, "eval-matrix"))
        return 0
    res = run_eval_matrix(arr_home=args.arr_home)
    if getattr(args, "notify_cmd", None):
        res["notification"] = _notify_cmd(
            args.notify_cmd, subject="conjectra eval-matrix completed", payload=res
        )
    _print_json(res)
    return 0


def _cmd_overview(args: argparse.Namespace) -> int:
    res = get_overview(
        args.arr_home,
        provider=args.provider,
        total_slots=args.total_slots,
    )
    _print_json(res)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    res = inspect_state(
        args.arr_home,
        goal_id=args.goal_id,
        subgoal_id=args.subgoal_id,
        hypothesis_id=args.hypothesis_id,
        workflow_id=args.workflow_id,
        provider=args.provider,
    )
    _print_json(res)
    return 0


def _cmd_subgoals(args: argparse.Namespace) -> int:
    res = get_subgoals(
        args.arr_home,
        goal_id=args.goal_id,
        status=args.status,
        provider=args.provider,
    )
    _print_json(res)
    return 0


def _cmd_token_audit(args: argparse.Namespace) -> int:
    res = get_token_audit(
        args.arr_home,
        goal_id=args.goal_id,
        provider=args.provider,
    )
    _print_json(res)
    return 0


def _cmd_takeover(args: argparse.Namespace) -> int:
    res = build_takeover_brief(
        args.arr_home,
        goal_id=args.goal_id,
        max_bytes=args.max_bytes,
        provider=args.provider,
    )
    if args.format == "markdown":
        print(res["brief_markdown"], end="")
    else:
        _print_json(res)
    return 0


def _cmd_set_policy(args: argparse.Namespace) -> int:
    raw_spec = _read_spec_arg(args.policy).strip()
    try:
        parsed_pol: Any = json.loads(raw_spec)
    except json.JSONDecodeError:
        parsed_pol = raw_spec
    payload: dict[str, Any] = {"policy": parsed_pol}
    if args.project:
        payload["project"] = args.project
    res = intervene(
        args.arr_home,
        scope=args.scope,
        action="set_routing_policy",
        goal_id=args.goal_id,
        target_id=args.target_id,
        payload=payload,
        actor=args.actor,
        reason=args.reason,
        wake_reactor=not args.no_wake,
        provider=args.provider,
    )
    _print_json(res)
    return 0


def _cmd_plans(args: argparse.Namespace) -> int:
    res = get_plans(
        args.arr_home,
        goal_id=args.goal_id,
        hypothesis_id=args.hypothesis_id,
        provider=args.provider,
    )
    _print_json(res)
    return 0


def _cmd_reviews(args: argparse.Namespace) -> int:
    res = get_reviews(
        args.arr_home,
        goal_id=args.goal_id,
        hypothesis_id=args.hypothesis_id,
        workflow_id=args.workflow_id,
        review_kind=args.review_kind,
        provider=args.provider,
    )
    _print_json(res)
    return 0


def _cmd_intervene(args: argparse.Namespace) -> int:
    payload: dict[str, Any] = {}
    if args.payload:
        loaded = json.loads(
            Path(args.payload[1:]).expanduser().read_text(encoding="utf-8")
            if args.payload.startswith("@")
            else args.payload
        )
        if not isinstance(loaded, dict):
            raise SystemExit("--payload must decode to a JSON object")
        payload = loaded
    res = intervene(
        args.arr_home,
        scope=args.scope,
        action=args.action,
        goal_id=args.goal_id,
        target_id=args.target_id,
        payload=payload,
        actor=args.actor,
        reason=args.reason,
        wake_reactor=not args.no_wake,
        provider=args.provider,
    )
    _print_json(res)
    return 0


def _cmd_status_snapshot(args: argparse.Namespace) -> int:
    import importlib

    home = resolve_arr_home(getattr(args, "arr_home", None))
    try:
        bridge = importlib.import_module("conjectra.status_snapshot")
    except ModuleNotFoundError as exc:
        _print_json({"ok": False, "error": f"status_snapshot_unavailable: {exc}"})
        return 1
    export_fn = getattr(bridge, "export_status_snapshot", None)
    if not callable(export_fn):
        _print_json({"ok": False, "error": "export_status_snapshot missing from conjectra.status_snapshot"})
        return 1
    with StateDB(home) as db:
        kwargs: dict[str, Any] = {"arr_home": home}
        res = export_fn(db, **kwargs)
    _print_json(res if isinstance(res, dict) else {"ok": True, "result": res})
    return 0


def build_parser() -> argparse.ArgumentParser:
    root_common = argparse.ArgumentParser(add_help=False)
    root_common.add_argument(
        "--home", "--arr-home", dest="arr_home", metavar="DIR", default=None, help="State directory (default: $CONJECTRA_HOME or ~/.conjectra)"
    )

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--home", "--arr-home", dest="arr_home", metavar="DIR", default=argparse.SUPPRESS, help="State directory (default: $CONJECTRA_HOME or ~/.conjectra)"
    )

    parser = argparse.ArgumentParser(
        prog="conjectra",
        description="conjectra: a zero-token control plane for multi-goal autonomous research",
        parents=[root_common],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_init = sub.add_parser("init-pilot", parents=[common], help="Initialize CONJECTRA_HOME ledger and optionally HQ")
    p_init.add_argument("--start-hq", action="store_true", help="Ensure HQ server and worker are running")
    p_init.add_argument("--baseline-path", default=None)
    p_init.add_argument("--no-ssh", action="store_true")
    p_init.set_defaults(func=_cmd_init_pilot)

    p_sub = sub.add_parser("submit-goal", parents=[common], help="Register an active ResearchGoal")
    p_sub.add_argument("--file", default=None, help="Path to ResearchGoal JSON")
    p_sub.add_argument("--goal-id", default=None)
    p_sub.add_argument("--project", default=None)
    p_sub.add_argument("--title", default=None)
    p_sub.add_argument("--direction", default="minimize")
    p_sub.add_argument("--primary-metric", default=None)
    p_sub.add_argument("--baseline-value", type=float, default=None)
    p_sub.add_argument("--baseline-sha", default=None)
    p_sub.add_argument("--repo-path", default=None)
    p_sub.add_argument("--writable-prefix", action="append", default=None)
    p_sub.add_argument("--banned-path", action="append", default=None)
    p_sub.add_argument("--kill-criterion", action="append", default=None)
    p_sub.add_argument("--max-tokens-budget", type=int, default=200000)
    p_sub.add_argument("--max-concurrent-branches", type=int, default=4)
    p_sub.add_argument("--adapter", default="generic")
    p_sub.add_argument("--autonomy-mode", default="unattended")
    p_sub.add_argument("--priority-weight", type=float, default=1.0)
    p_sub.add_argument("--min-guaranteed-slots", type=int, default=1)
    p_sub.add_argument("--target-value", type=float, default=None)
    p_sub.add_argument("--wall-budget-sec", type=int, default=21600)
    p_sub.set_defaults(func=_cmd_submit_goal)

    p_step = sub.add_parser("step", parents=[common], help="Run one deterministic controller step")
    p_step.add_argument("--goal-id", default=None)
    p_step.add_argument("--mode", choices=["hq", "inline"], default="hq")
    p_step.add_argument("--propose-mode", choices=["auto", "seeds", "ai"], default="auto")
    p_step.add_argument("--dry-run-llm", action="store_true")
    p_step.set_defaults(func=_cmd_step)

    p_daemon = sub.add_parser("daemon", parents=[common], help="Run event-driven reactor daemon")
    p_daemon.add_argument("--goal-id", default=None)
    p_daemon.add_argument("--mode", choices=["hq", "inline"], default="hq")
    p_daemon.add_argument("--propose-mode", choices=["auto", "seeds", "ai"], default="auto")
    p_daemon.add_argument("--dry-run-llm", action="store_true")
    p_daemon.add_argument("--sweep-interval", type=float, default=15.0)
    p_daemon.add_argument("--max-seconds", type=float, default=None)
    p_daemon.add_argument("--background", action="store_true")
    p_daemon.add_argument("--detach", action="store_true")
    p_daemon.add_argument("--notify-cmd", default=None, help="file:/path.jsonl or a command template ({subject} substituted; body on stdin)")
    p_daemon.add_argument("--stop", action="store_true")
    p_daemon.set_defaults(func=_cmd_daemon)

    p_notify = sub.add_parser("notify-exit", parents=[common], help="Wake reactor on task completion")
    p_notify.add_argument("--task-id", required=True)
    p_notify.add_argument("--exit-code", type=int, default=0)
    p_notify.add_argument("--hq-job-id", type=int, default=None)
    p_notify.add_argument("--status", default=None)
    p_notify.set_defaults(func=_cmd_notify_exit)

    p_status = sub.add_parser("status", parents=[common], help="Show ledger, HQ, and escalation status")
    p_status.add_argument("--goal-id", default=None)
    p_status.set_defaults(func=_cmd_status)

    p_board = sub.add_parser("board", parents=[common], help="Show hypothesis leaderboard and events")
    p_board.add_argument("--goal-id", default=None)
    p_board.add_argument("--since-seq", type=int, default=0)
    p_board.add_argument("--limit", type=int, default=50)
    p_board.add_argument("--format", choices=["json", "text"], default="json")
    p_board.set_defaults(func=_cmd_board)

    p_brief = sub.add_parser("export-teamwork-brief", parents=[common], help="Export brief for workflow/teamwork")
    p_brief.add_argument("--goal-id", default=None)
    p_brief.add_argument("--format", choices=["json", "markdown"], default="json")
    p_brief.set_defaults(func=_cmd_export_teamwork_brief)

    p_import = sub.add_parser("import-proposals", parents=[common], help="Import validated proposal JSON")
    p_import.add_argument("--goal-id", default=None)
    p_import.add_argument("--file", required=True, help="JSON file path or '-' for stdin")
    p_import.add_argument("--min-evoi", type=float, default=0.05)
    p_import.add_argument("--no-enqueue", action="store_true")
    p_import.set_defaults(func=_cmd_import_proposals)

    p_bench = sub.add_parser("benchmark-vs-baseline", parents=[common], help="Compute machine-verified KPIs")
    p_bench.add_argument("--goal-id", default=None)
    p_bench.add_argument("--baseline-path", default=None)
    p_bench.add_argument("--slots", type=int, default=None)
    p_bench.add_argument("--no-ssh", action="store_true")
    p_bench.set_defaults(func=_cmd_benchmark_vs_baseline)

    p_sync = sub.add_parser(
        "sync-status",
        parents=[common],
        help="Write LATEST_STATUS.md & benchmark_latest.json and mirror them to $CONJECTRA_STATUS_MIRROR",
    )
    p_sync.add_argument("--goal-id", default=None)
    p_sync.add_argument("--baseline-path", default=None)
    p_sync.add_argument("--target", default=None, help="Local dir or user@host:/dir (default: $CONJECTRA_STATUS_MIRROR)")
    p_sync.add_argument("--no-ssh", action="store_true")
    p_sync.add_argument("--dry-run", action="store_true")
    p_sync.set_defaults(func=_cmd_sync_status)

    p_boost = sub.add_parser("boost", parents=[common], help="Clean-room cross-provider Builder <-> Blind Adversary ping-pong")
    p_boost.add_argument("--repo-path", required=True)
    p_boost.add_argument("--project", default="boost")
    p_boost.add_argument("--spec", required=True, help="Task specification string or @file path")
    p_boost.add_argument("--acceptance-cmd", required=True)
    p_boost.add_argument("--writable-path", action="append", required=True)
    p_boost.add_argument("--banned-path", action="append", default=None)
    p_boost.add_argument("--builder", default="auto", help="Provider spec e.g. claude:claude-sonnet-5-5:medium or cmd:...")
    p_boost.add_argument("--adversary", default="auto", help="Distinct-family provider, e.g. codex:gpt-5-codex:high")
    p_boost.add_argument("--max-rounds", type=int, default=3, help="1 initial build + up to 2 repairs (default: 3)")
    p_boost.add_argument("--timeout", type=int, default=600)
    p_boost.add_argument("--in-place", action="store_true")
    p_boost.add_argument("--detach", action="store_true")
    p_boost.add_argument("--notify-cmd", default=None, help="file:/path.jsonl or a command template ({subject} substituted; body on stdin)")
    p_boost.set_defaults(func=_cmd_boost)

    p_xswarm = sub.add_parser("xswarm", parents=[common], help="Clean-room multi-provider @name cursor swarm on HQ")
    p_xswarm.add_argument("--spec-file", required=True, help="JSON config with repo_path, project, title, workers, verifier_cmd")
    p_xswarm.add_argument("--no-hq", action="store_true")
    p_xswarm.add_argument("--detach", action="store_true")
    p_xswarm.add_argument("--notify-cmd", default=None, help="file:/path.jsonl or a command template ({subject} substituted; body on stdin)")
    p_xswarm.set_defaults(func=_cmd_xswarm)

    p_xtw = sub.add_parser("xteamwork", parents=[common], help="Clean-room multi-phase milestone teamwork with <4KB flush")
    p_xtw.add_argument("--spec-file", required=True, help="JSON config with repo_path, project, title, phases, final_acceptance_cmd")
    p_xtw.add_argument("--detach", action="store_true")
    p_xtw.add_argument("--notify-cmd", default=None, help="file:/path.jsonl or a command template ({subject} substituted; body on stdin)")
    p_xtw.set_defaults(func=_cmd_xteamwork)

    p_route = sub.add_parser("route-workload", parents=[common], help="Classify task features and pick optimal mode from empirical ledger")
    p_route.add_argument("--description", required=True)
    p_route.add_argument("--scope", action="append", default=None, help="Comma-separated writable paths per slice (repeatable)")
    p_route.add_argument("--readonly", action="store_true")
    p_route.add_argument("--coupled-interface", action="store_true")
    p_route.add_argument("--phases", type=int, default=1)
    p_route.add_argument("--minutes", type=int, default=20)
    p_route.set_defaults(func=_cmd_route_workload)

    p_matrix = sub.add_parser("eval-matrix", parents=[common], help="Run 4-class fault-seeded empirical mode ablation benchmark")
    p_matrix.add_argument("--detach", action="store_true")
    p_matrix.add_argument("--notify-cmd", default=None, help="file:/path.jsonl or a command template ({subject} substituted; body on stdin)")
    p_matrix.set_defaults(func=_cmd_eval_matrix)

    p_ov = sub.add_parser("overview", parents=[common], help="Sub-50ms multi-Goal overview from SQLite WAL")
    p_ov.add_argument("--total-slots", type=int, default=None)
    p_ov.add_argument("--provider", default="interactive")
    p_ov.set_defaults(func=_cmd_overview)

    p_insp = sub.add_parser("inspect", parents=[common], help="Sub-50ms Goal/SubGoal/Hypothesis/Workflow inspection")
    p_insp.add_argument("--goal-id", default=None)
    p_insp.add_argument("--subgoal-id", default=None)
    p_insp.add_argument("--hypothesis-id", default=None)
    p_insp.add_argument("--workflow-id", default=None)
    p_insp.add_argument("--provider", default="interactive")
    p_insp.set_defaults(func=_cmd_inspect)

    p_sgs = sub.add_parser("subgoals", parents=[common], help="Sub-50ms SubGoal portfolio inspection from SQLite WAL")
    p_sgs.add_argument("--goal-id", default=None)
    p_sgs.add_argument("--status", default=None)
    p_sgs.add_argument("--provider", default="interactive")
    p_sgs.set_defaults(func=_cmd_subgoals)

    p_ta = sub.add_parser("token-audit", parents=[common], help="Audit real uncached vs cache tokens per Goal")
    p_ta.add_argument("--goal-id", default=None)
    p_ta.add_argument("--provider", default="interactive")
    p_ta.set_defaults(func=_cmd_token_audit)

    p_to = sub.add_parser("takeover", parents=[common], help="Emit <3KB session-independent takeover brief")
    p_to.add_argument("--goal-id", default=None)
    p_to.add_argument("--format", choices=["json", "markdown"], default="json")
    p_to.add_argument("--max-bytes", type=int, default=2950)
    p_to.add_argument("--provider", default="interactive")
    p_to.set_defaults(func=_cmd_takeover)

    p_iv = sub.add_parser("intervene", parents=[common], help="Atomic HITL intervention + instant reactor.sock wake")
    p_iv.add_argument("--scope", required=True, choices=["global", "goal", "subgoal", "branch", "workflow", "task"])
    p_iv.add_argument("--action", required=True)
    p_iv.add_argument("--goal-id", default=None)
    p_iv.add_argument("--target-id", default=None)
    p_iv.add_argument("--payload", default=None, help="JSON object string or @path")
    p_iv.add_argument("--actor", default="user")
    p_iv.add_argument("--reason", default="")
    p_iv.add_argument("--no-wake", action="store_true")
    p_iv.add_argument("--provider", default="interactive")
    p_iv.set_defaults(func=_cmd_intervene)

    p_pol = sub.add_parser("set-policy", parents=[common], help="Compile & apply NL or structured V3 model routing policy")
    p_pol.add_argument("--scope", default="goal", choices=["global", "goal", "branch", "workflow", "task"])
    p_pol.add_argument("--goal-id", default=None)
    p_pol.add_argument("--target-id", default=None)
    p_pol.add_argument("--policy", required=True, help="Chinese/English NL policy string, JSON object, or @file")
    p_pol.add_argument("--project", default=None)
    p_pol.add_argument("--actor", default="user")
    p_pol.add_argument("--reason", default="")
    p_pol.add_argument("--no-wake", action="store_true")
    p_pol.add_argument("--provider", default="interactive")
    p_pol.set_defaults(func=_cmd_set_policy)

    p_plans = sub.add_parser("plans", parents=[common], help="Sub-50ms inspection of versioned Research/Experiment/Execution plans")
    p_plans.add_argument("--goal-id", default=None)
    p_plans.add_argument("--hypothesis-id", default=None)
    p_plans.add_argument("--provider", default="interactive")
    p_plans.set_defaults(func=_cmd_plans)

    p_revs = sub.add_parser("reviews", parents=[common], help="Sub-50ms inspection of 3-way Design/Impl/Result/Synthesis ReviewReceipts")
    p_revs.add_argument("--goal-id", default=None)
    p_revs.add_argument("--hypothesis-id", default=None)
    p_revs.add_argument("--workflow-id", default=None)
    p_revs.add_argument(
        "--review-kind",
        default=None,
        choices=["DESIGN_REVIEW", "IMPLEMENTATION_REVIEW", "RESULT_REVIEW", "SYNTHESIS_REVIEW"],
    )
    p_revs.add_argument("--provider", default="interactive")
    p_revs.set_defaults(func=_cmd_reviews)

    p_vd = sub.add_parser("status-snapshot", parents=[common], help="Export dashboard live status snapshot atomically")
    p_vd.set_defaults(func=_cmd_status_snapshot)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(raw_argv)
    args._raw_argv = raw_argv
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
