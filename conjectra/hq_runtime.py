"""HyperQueue server/worker lifecycle and dynamic DAG runtime for conjectra."""

from __future__ import annotations

import json
import os
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from conjectra.types import ResourceSpec, StageKind, StageTask

DEFAULT_HQ_BIN_CANDIDATES = (str(Path.home() / ".local" / "bin" / "hq"),)

# HyperQueue v0.27.0 accepts only alphanumeric resource names ([a-zA-Z0-9]+).
RESOURCE_ALIASES: dict[str, str] = {
    "cpus": "cpus",
    "ram_mb": "rammb",
    "rammb": "rammb",
    "gpu": "gpu",
    "llm_gemini": "llmgemini",
    "llmgemini": "llmgemini",
    "llm_claude": "llmclaude",
    "llmclaude": "llmclaude",
    "llm_deepseek": "llmdeepseek",
    "llmdeepseek": "llmdeepseek",
    "llm_codex_audit": "llmcodexaudit",
    "llmcodexaudit": "llmcodexaudit",
    "eda_sim": "edasim",
    "edasim": "edasim",
    "test_sim": "testsim",
    "testsim": "testsim",
    "cloud_slot": "cloudslot",
    "cloudslot": "cloudslot",
}

POOL_TO_RESOURCE: dict[str, str] = {
    "gemini": "llm_gemini",
    "gemini-cli": "llm_gemini",
    "llm_gemini": "llm_gemini",
    "claude": "llm_claude",
    "llm_claude": "llm_claude",
    "deepseek": "llm_deepseek",
    "llm_deepseek": "llm_deepseek",
    "codex": "llm_codex_audit",
    "llm_codex_audit": "llm_codex_audit",
    "grok": "llm_deepseek",
}


def resolve_arr_home(arr_home: str | Path | None = None) -> Path:
    """Resolve CONJECTRA_HOME from explicit argument, environment, or default ~/.conjectra."""
    raw = arr_home or os.environ.get("CONJECTRA_HOME") or (Path.home() / ".conjectra")
    return Path(raw).expanduser().resolve()


def resolve_hq_bin(hq_bin: str | Path | None = None) -> str:
    """Resolve the HyperQueue binary path."""
    if hq_bin:
        p = Path(hq_bin).expanduser()
        if p.exists():
            return str(p)
    env_bin = os.environ.get("CONJECTRA_HQ_BIN")
    if env_bin and Path(env_bin).expanduser().exists():
        return str(Path(env_bin).expanduser())
    for cand in DEFAULT_HQ_BIN_CANDIDATES:
        if Path(cand).exists():
            return cand
    found = shutil.which("hq")
    if found:
        return found
    raise FileNotFoundError(
        "HyperQueue binary `hq` not found (set CONJECTRA_HQ_BIN, install to ~/.local/bin/hq, or add it to PATH)"
    )


def to_hq_resource_name(name: str) -> str:
    """Normalize a logical resource name to an HQ-compatible alphanumeric key."""
    key = name.strip()
    if key in RESOURCE_ALIASES:
        return RESOURCE_ALIASES[key]
    cleaned = "".join(ch for ch in key if ch.isalnum())
    if not cleaned:
        raise ValueError(f"Invalid resource name: {name!r}")
    return cleaned


def _toml_str(val: str) -> str:
    return json.dumps(str(val), ensure_ascii=False)


def _detect_mem_available_mb() -> int:
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return max(512, kb // 1024)
    except Exception:
        pass
    return 8192


def _detect_gpu_count() -> int:
    smi = shutil.which("nvidia-smi")
    if not smi:
        return 0
    try:
        proc = subprocess.run(
            [smi, "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return 0
        lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
        return len(lines)
    except Exception:
        return 0


def detect_host_resources(
    *,
    llm_gemini: int = 3,
    llm_claude: int = 2,
    llm_deepseek: int = 2,
    llm_codex_audit: int = 2,
    eda_sim: int = 2,
    test_sim: int = 2,
    cloud_slot: int = 0,
    cpus: int | None = None,
    ram_mb: int | None = None,
    gpu: int | None = None,
) -> dict[str, int]:
    """Detect live host capacity and combine with logical pool capacities."""
    detected_cpus = cpus if cpus is not None else min(8, max(1, os.cpu_count() or 1))
    detected_ram = ram_mb if ram_mb is not None else _detect_mem_available_mb()
    detected_gpu = gpu if gpu is not None else _detect_gpu_count()
    return {
        "cpus": int(detected_cpus),
        "ram_mb": int(detected_ram),
        "gpu": int(detected_gpu),
        "llm_gemini": int(llm_gemini),
        "llm_claude": int(llm_claude),
        "llm_deepseek": int(llm_deepseek),
        "llm_codex_audit": int(llm_codex_audit),
        "eda_sim": int(eda_sim),
        "test_sim": int(test_sim),
        "cloud_slot": int(cloud_slot),
    }


_TEST_SIM_STAGES = frozenset(
    {StageKind.CPU_GATE.value, StageKind.EXP_SCREEN.value, StageKind.EXP_CONFIRM.value}
)


_SKIP_RESOURCE_METADATA_KEYS = frozenset(
    {
        "cpus",
        "ram_mb",
        "gpu",
        "llm_pool",
        "eda_sim",
        "test_sim",
        "cloud_slot",
        "wall_timeout_s",
        "hq_llm_resource",
        "pinned_route",
        "model_spec",
        "effort",
    }
)


def resource_spec_to_requests(
    resources: ResourceSpec | Mapping[str, Any] | None,
    *,
    stage: StageKind | str | None = None,
) -> tuple[int, dict[str, int]]:
    """Convert ResourceSpec or mapping into (cpus, {hq_resource_name: amount})."""
    if resources is None:
        data: dict[str, Any] = {}
    elif isinstance(resources, ResourceSpec):
        data = resources.to_dict()
    else:
        data = dict(resources)

    cpus = max(1, int(data.get("cpus", 1)))
    reqs: dict[str, int] = {}

    for logical_key in ("ram_mb", "gpu", "eda_sim", "test_sim", "cloud_slot"):
        val = int(data.get(logical_key, 0) or 0)
        if val > 0:
            reqs[to_hq_resource_name(logical_key)] = val

    stage_val = stage.value if isinstance(stage, StageKind) else (str(stage) if stage is not None else None)
    if stage_val in _TEST_SIM_STAGES:
        reqs["testsim"] = max(1, reqs.get("testsim", 0))

    hq_llm_res = data.get("hq_llm_resource")
    pinned_route = data.get("pinned_route")
    pool = data.get("llm_pool")
    if isinstance(hq_llm_res, str) and hq_llm_res.strip():
        reqs[to_hq_resource_name(hq_llm_res.strip())] = 1
    elif isinstance(pinned_route, Mapping) and isinstance(pinned_route.get("hq_resource"), str) and pinned_route["hq_resource"].strip():
        reqs[to_hq_resource_name(str(pinned_route["hq_resource"]).strip())] = 1
    elif pool:
        pool_head = str(pool).strip().split(":", 1)[0].strip().lower()
        if pool_head and pool_head not in ("cmd", "custom", "webchat", "none"):
            mapped = POOL_TO_RESOURCE.get(pool_head, pool_head)
            reqs[to_hq_resource_name(mapped)] = 1

    for k, v in data.items():
        if k in _SKIP_RESOURCE_METADATA_KEYS:
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        amt = int(v)
        if amt > 0:
            reqs[to_hq_resource_name(str(k))] = amt

    return cpus, reqs


def _notify_reactor_socket(
    sock_path: Path,
    payload: dict[str, Any],
    timeout_s: float = 1.5,
) -> bool:
    """Best-effort Unix socket notification to controller reactor."""
    if not sock_path.exists():
        return False
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
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


@dataclass
class DAGTaskSpec:
    """Single task definition inside a multi-task HyperQueue DAG submission."""

    task_id: str
    command: list[str]
    deps: list[str] = field(default_factory=list)
    resources: ResourceSpec | dict[str, Any] = field(default_factory=ResourceSpec)
    stage: StageKind | str | None = None
    priority: int = 0
    time_limit_s: int | None = None
    cwd: str | Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    stdout: str | Path | None = None
    stderr: str | Path | None = None
    on_exit_cmd: list[str] | None = None


@dataclass
class DAGSubmission:
    """Result of submitting a multi-task DAG to HyperQueue."""

    hq_job_id: int
    task_id_map: dict[str, int]
    jobfile_path: Path


class HQRuntime:
    """Lifecycle and dynamic DAG submission manager for HyperQueue."""

    def __init__(
        self,
        arr_home: str | Path | None = None,
        *,
        hq_bin: str | Path | None = None,
        worker_resources: Mapping[str, int] | None = None,
        reactor_socket: str | Path | None = None,
        notify_cli: Sequence[str] | None = None,
    ) -> None:
        self.arr_home = resolve_arr_home(arr_home)
        self.server_dir = self.arr_home / "hq-server"
        self.logs_dir = self.arr_home / "logs"
        self.jobs_log_dir = self.logs_dir / "jobs"
        self.wrappers_dir = self.logs_dir / "wrappers"
        self.work_dir = self.arr_home / "hq-work"
        self.reactor_socket = (
            Path(reactor_socket).expanduser()
            if reactor_socket
            else self.arr_home / "reactor.sock"
        )
        self.notify_cli = list(notify_cli) if notify_cli else None
        self.hq_bin = resolve_hq_bin(hq_bin)
        self.worker_resources = dict(worker_resources) if worker_resources is not None else None

        self._lock = threading.Lock()
        self._pending_deps: dict[int, dict[str, Any]] = {}

    def _base_cmd(self, *, json_mode: bool = True) -> list[str]:
        cmd = [self.hq_bin, "--server-dir", str(self.server_dir)]
        if json_mode:
            cmd.extend(["--output-mode", "json"])
        return cmd

    def _run_hq(
        self,
        args: Sequence[str],
        *,
        json_mode: bool = True,
        timeout_s: float = 15.0,
        check: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        cmd = self._base_cmd(json_mode=json_mode) + list(args)
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if check and proc.returncode != 0:
            raise RuntimeError(
                f"HQ command failed (rc={proc.returncode}): {' '.join(cmd)}\n"
                f"stderr={proc.stderr.strip()}"
            )
        return proc

    @staticmethod
    def _parse_json_stdout(stdout: str) -> Any:
        text = (stdout or "").strip()
        if not text:
            return None
        return json.loads(text)

    def server_info(self) -> dict[str, Any] | None:
        """Return parsed server info if the HQ server is reachable, else None."""
        if not self.server_dir.exists():
            return None
        proc = self._run_hq(["server", "info"], json_mode=True, timeout_s=5.0)
        if proc.returncode != 0:
            return None
        try:
            data = self._parse_json_stdout(proc.stdout)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def worker_list(self, *, all_workers: bool = False) -> list[dict[str, Any]]:
        """Return list of workers from the HQ server."""
        args = ["worker", "list"]
        if all_workers:
            args.append("--all")
        proc = self._run_hq(args, json_mode=True, timeout_s=5.0)
        if proc.returncode != 0:
            return []
        try:
            data = self._parse_json_stdout(proc.stdout)
            return list(data) if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def query_job_list(
        self,
        all_jobs: bool = True,
        *,
        states: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Query live HQ server job list with explicit fail-closed error disambiguation.

        Returns `{"lookup_status": "OK" | "QUERY_ERROR", "jobs": list[dict], "error": str}`.
        Only returns `"OK"` when a live reachable `hq-server` returns `rc == 0` and a valid JSON list
        (including `[]` when zero jobs exist); any non-zero exit, timeout, `OSError`, empty stdout,
        malformed JSON, or server-unreachable stderr returns `"QUERY_ERROR"`.
        """
        args = ["job", "list"]
        if states:
            args.extend(["--filter", ",".join(states)])
        elif all_jobs:
            args.append("--all")

        try:
            proc = self._run_hq(args, json_mode=True, timeout_s=10.0)
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "lookup_status": "QUERY_ERROR",
                "jobs": [],
                "error": f"hq_rpc_exception:{type(exc).__name__}:{exc}",
            }

        stderr_txt = (proc.stderr or "").strip()
        stdout_txt = (proc.stdout or "").strip()
        if proc.returncode != 0:
            return {
                "lookup_status": "QUERY_ERROR",
                "jobs": [],
                "error": stderr_txt or stdout_txt or f"hq_nonzero_rc:{proc.returncode}",
            }
        if not stdout_txt:
            return {
                "lookup_status": "QUERY_ERROR",
                "jobs": [],
                "error": stderr_txt or "empty_json_stdout",
            }

        low_err = stderr_txt.lower()
        if any(
            bad in low_err
            for bad in (
                "cannot connect",
                "connection refused",
                "cannot find hq access file",
                "no running server",
                "timed out",
            )
        ):
            return {
                "lookup_status": "QUERY_ERROR",
                "jobs": [],
                "error": stderr_txt,
            }

        try:
            data = json.loads(stdout_txt)
        except json.JSONDecodeError as exc:
            return {
                "lookup_status": "QUERY_ERROR",
                "jobs": [],
                "error": f"invalid_json:{exc}",
            }

        if not isinstance(data, list) or any(not isinstance(item, Mapping) for item in data):
            return {
                "lookup_status": "QUERY_ERROR",
                "jobs": [],
                "error": f"unexpected_job_list_json:{type(data).__name__}",
            }

        return {
            "lookup_status": "OK",
            "jobs": [dict(item) for item in data],
            "error": "",
        }

    def job_list(
        self,
        *,
        all_jobs: bool = True,
        states: Sequence[str] | None = None,
    ) -> list[dict[str, Any]]:
        """List jobs tracked by the HQ server."""
        res = self.query_job_list(all_jobs=all_jobs, states=states)
        return res["jobs"] if res["lookup_status"] == "OK" else []

    def query_job_state(self, hq_job_id: int) -> dict[str, Any]:
        """Query live HQ server for `hq_job_id` with explicit 3-state disambiguation.

        Returns `{"lookup_status": "FOUND" | "NOT_FOUND" | "QUERY_ERROR", "job": dict | None, "error": str}`.
        `NOT_FOUND` is returned ONLY when a live reachable `hq-server` returns valid JSON (`rc == 0`)
        authoritatively confirming `hq_job_id` is absent; any RPC, nonzero exit, timeout, or JSON error
        returns `QUERY_ERROR` so callers never mistake query failure for positive termination.
        """
        try:
            jid = int(hq_job_id)
        except (TypeError, ValueError) as exc:
            return {
                "lookup_status": "QUERY_ERROR",
                "job": None,
                "error": f"invalid_hq_job_id:{hq_job_id!r}:{exc}",
            }

        try:
            proc = self._run_hq(["job", "info", str(jid)], json_mode=True, timeout_s=10.0)
        except (OSError, subprocess.SubprocessError) as exc:
            return {
                "lookup_status": "QUERY_ERROR",
                "job": None,
                "error": f"hq_rpc_exception:{type(exc).__name__}:{exc}",
            }

        stderr_txt = (proc.stderr or "").strip()
        stdout_txt = (proc.stdout or "").strip()
        if proc.returncode != 0:
            return {
                "lookup_status": "QUERY_ERROR",
                "job": None,
                "error": stderr_txt or stdout_txt or f"hq_nonzero_rc:{proc.returncode}",
            }
        if not stdout_txt:
            return {
                "lookup_status": "QUERY_ERROR",
                "job": None,
                "error": stderr_txt or "empty_json_stdout",
            }

        try:
            data = json.loads(stdout_txt)
        except json.JSONDecodeError as exc:
            return {
                "lookup_status": "QUERY_ERROR",
                "job": None,
                "error": f"invalid_json:{exc}",
            }

        low_err = stderr_txt.lower()
        if any(
            bad in low_err
            for bad in (
                "cannot connect",
                "connection refused",
                "cannot find hq access file",
                "no running server",
                "timed out",
            )
        ):
            return {
                "lookup_status": "QUERY_ERROR",
                "job": None,
                "error": stderr_txt,
            }

        def _extract_id(rec: Mapping[str, Any]) -> int | None:
            info = rec.get("info")
            raw_id = info.get("id") if isinstance(info, Mapping) and "id" in info else rec.get("id")
            if raw_id is None or isinstance(raw_id, bool):
                return None
            try:
                return int(raw_id)
            except (TypeError, ValueError):
                return -1

        if isinstance(data, list):
            if not data:
                return {
                    "lookup_status": "NOT_FOUND",
                    "job": None,
                    "error": stderr_txt,
                }
            first = data[0]
            if not isinstance(first, Mapping):
                return {
                    "lookup_status": "QUERY_ERROR",
                    "job": None,
                    "error": f"unexpected_list_element:{type(first).__name__}",
                }
            rec_id = _extract_id(first)
            if rec_id is not None and rec_id != jid:
                return {
                    "lookup_status": "QUERY_ERROR",
                    "job": None,
                    "error": f"mismatched_job_id:{rec_id}!={jid}",
                }
            return {
                "lookup_status": "FOUND",
                "job": dict(first),
                "error": "",
            }

        if isinstance(data, Mapping):
            if any(k in data for k in ("info", "tasks", "id", "state")):
                rec_id = _extract_id(data)
                if rec_id is not None and rec_id != jid:
                    return {
                        "lookup_status": "QUERY_ERROR",
                        "job": None,
                        "error": f"mismatched_job_id:{rec_id}!={jid}",
                    }
                return {
                    "lookup_status": "FOUND",
                    "job": dict(data),
                    "error": "",
                }
            if data.get("found") is False or str(
                data.get("lookup_status") or data.get("status") or ""
            ).strip().upper() == "NOT_FOUND":
                return {
                    "lookup_status": "NOT_FOUND",
                    "job": None,
                    "error": str(data.get("error") or stderr_txt),
                }
            return {
                "lookup_status": "QUERY_ERROR",
                "job": None,
                "error": str(data.get("error") or stderr_txt or "unexpected_json_object"),
            }

        return {
            "lookup_status": "QUERY_ERROR",
            "job": None,
            "error": f"unexpected_json_type:{type(data).__name__}",
        }

    def job_info(self, job_id: int) -> dict[str, Any] | None:
        """Return detailed HQ job info dictionary when authoritatively FOUND, else None."""
        res = self.query_job_state(job_id)
        return res["job"] if res["lookup_status"] == "FOUND" else None

    def task_list(self, job_id: int) -> list[dict[str, Any]]:
        """Return task states for a given HQ job ID."""
        proc = self._run_hq(["task", "list", str(int(job_id))], json_mode=True, timeout_s=10.0)
        if proc.returncode != 0:
            return []
        try:
            data = self._parse_json_stdout(proc.stdout)
            if isinstance(data, dict):
                tasks = data.get(str(int(job_id))) or []
                return list(tasks)
            return list(data) if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    def ensure_running(
        self,
        *,
        resources: Mapping[str, int] | None = None,
        timeout_s: float = 20.0,
    ) -> dict[str, Any]:
        """Start detached HQ server & worker under CONJECTRA_HOME if not already healthy."""
        self.arr_home.mkdir(parents=True, exist_ok=True)
        self.server_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_log_dir.mkdir(parents=True, exist_ok=True)
        self.wrappers_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        started_server = False
        started_worker = False

        sinfo = self.server_info()
        if sinfo is None:
            srv_log = (self.logs_dir / "hq-server.log").open("ab")
            try:
                subprocess.Popen(
                    [self.hq_bin, "--server-dir", str(self.server_dir), "server", "start"],
                    stdin=subprocess.DEVNULL,
                    stdout=srv_log,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.arr_home),
                    start_new_session=True,
                )
            finally:
                srv_log.close()
            started_server = True
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                sinfo = self.server_info()
                if sinfo is not None:
                    break
                time.sleep(0.1)
            if sinfo is None:
                raise RuntimeError(f"Timed out waiting for HQ server in {self.server_dir}")

        workers = [w for w in self.worker_list() if w.get("ended") is None]
        if not workers:
            res_map = dict(
                resources
                if resources is not None
                else (self.worker_resources if self.worker_resources is not None else detect_host_resources())
            )
            cpus = max(1, int(res_map.get("cpus", os.cpu_count() or 1)))
            worker_cmd = [
                self.hq_bin,
                "--server-dir",
                str(self.server_dir),
                "worker",
                "start",
                f"--cpus={cpus}",
                "--detect-resources=mem",
                "--on-server-lost=stop",
                "--work-dir",
                str(self.work_dir),
            ]
            for raw_name, raw_val in sorted(res_map.items()):
                if raw_name == "cpus":
                    continue
                val = int(raw_val or 0)
                # HyperQueue rejects `--resource <name>=sum(0)` ("Resource <name> is empty").
                if val <= 0:
                    continue
                hq_name = to_hq_resource_name(raw_name)
                worker_cmd.extend(["--resource", f"{hq_name}=sum({val})"])

            wrk_log = (self.logs_dir / "hq-worker.log").open("ab")
            try:
                subprocess.Popen(
                    worker_cmd,
                    stdin=subprocess.DEVNULL,
                    stdout=wrk_log,
                    stderr=subprocess.STDOUT,
                    cwd=str(self.arr_home),
                    start_new_session=True,
                )
            finally:
                wrk_log.close()
            started_worker = True
            deadline = time.monotonic() + timeout_s
            while time.monotonic() < deadline:
                workers = [w for w in self.worker_list() if w.get("ended") is None]
                if workers:
                    break
                time.sleep(0.1)
            if not workers:
                raise RuntimeError(f"Timed out waiting for HQ worker in {self.server_dir}")

        return {
            "server": sinfo,
            "workers": workers,
            "started_server": started_server,
            "started_worker": started_worker,
        }

    def stop(self, *, timeout_s: float = 10.0) -> None:
        """Stop all HQ workers and the HQ server cleanly."""
        with self._lock:
            pending = list(self._pending_deps.values())
        for entry in pending:
            entry["cancel_event"].set()

        if self.server_info() is None:
            return
        self._run_hq(["worker", "stop", "all"], json_mode=False, timeout_s=timeout_s)
        self._run_hq(["server", "stop"], json_mode=False, timeout_s=timeout_s)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.server_info() is None:
                return
            time.sleep(0.1)

    def _build_exit_wrapper(
        self,
        *,
        task_id: str,
        command: Sequence[str],
        on_exit_cmd: Sequence[str] | None = None,
        reactor_socket: Path | None = None,
        notify_cli: Sequence[str] | None = None,
    ) -> list[str]:
        """Build a small wrapper script that runs command and emits immediate exit callbacks."""
        self.wrappers_dir.mkdir(parents=True, exist_ok=True)
        safe_id = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in task_id)[:48]
        script_path = self.wrappers_dir / f"wrap_{safe_id}_{uuid.uuid4().hex[:8]}.sh"
        sock = reactor_socket if reactor_socket is not None else self.reactor_socket
        cli_cmd = list(notify_cli) if notify_cli is not None else self.notify_cli

        cmd_str = " ".join(shlex.quote(str(x)) for x in command)
        # Make `python -m conjectra.*` importable on workers even when running from a source checkout.
        pkg_root = shlex.quote(str(Path(__file__).resolve().parents[1]))
        lines = [
            "#!/usr/bin/env bash",
            "set +e",
            f"export PYTHONPATH={pkg_root}${{PYTHONPATH:+:$PYTHONPATH}}",
            cmd_str,
            "rc=$?",
        ]
        if sock is not None:
            py_notify = (
                "import json, os, socket, sys, time; "
                "sock_p = sys.argv[1]; tid = sys.argv[2]; rc = int(sys.argv[3]); "
                "jid = os.environ.get('HQ_JOB_ID'); "
                "payload = json.dumps({'event': 'task_exit', 'task_id': tid, "
                "'hq_job_id': int(jid) if jid and jid.isdigit() else None, "
                "'exit_code': rc, 'ts': time.time()}) + '\\n'; "
                "\nfor st in (socket.SOCK_STREAM, socket.SOCK_DGRAM):\n"
                "  try:\n"
                "    s = socket.socket(socket.AF_UNIX, st)\n"
                "    s.settimeout(1.5)\n"
                "    s.connect(sock_p)\n"
                "    s.sendall(payload.encode('utf-8'))\n"
                "    s.close()\n"
                "    break\n"
                "  except OSError:\n"
                "    pass\n"
            )
            lines.append(
                f"if [ -S {shlex.quote(str(sock))} ]; then "
                f"{shlex.quote(sys.executable)} -c {shlex.quote(py_notify)} "
                f"{shlex.quote(str(sock))} {shlex.quote(task_id)} \"$rc\" >/dev/null 2>&1 || true; fi"
            )
        if cli_cmd:
            cli_str = " ".join(shlex.quote(str(x)) for x in cli_cmd)
            lines.append(
                f"{cli_str} --task-id {shlex.quote(task_id)} "
                f"--hq-job-id \"${{HQ_JOB_ID:-0}}\" --exit-code \"$rc\" >/dev/null 2>&1 || true"
            )
        if on_exit_cmd:
            exit_str = " ".join(shlex.quote(str(x)) for x in on_exit_cmd)
            lines.append(f"CONJECTRA_EXIT_CODE=\"$rc\" {exit_str} >/dev/null 2>&1 || true")
        lines.append("exit \"$rc\"")
        script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        script_path.chmod(0o755)
        return ["/bin/bash", str(script_path)]

    def _resolve_stdio(self, task_id: str, stdout: str | Path | None, stderr: str | Path | None) -> tuple[str, str]:
        self.jobs_log_dir.mkdir(parents=True, exist_ok=True)
        safe = "".join(ch if (ch.isalnum() or ch in "-_") else "_" for ch in task_id)[:64]
        out_p = str(stdout) if stdout is not None else str(self.jobs_log_dir / f"{safe}.stdout")
        err_p = str(stderr) if stderr is not None else str(self.jobs_log_dir / f"{safe}.stderr")
        return out_p, err_p

    def _normalize_task_inputs(
        self,
        task_or_cmd: StageTask | Sequence[str] | str,
        *,
        task_id: str | None = None,
        command: Sequence[str] | str | None = None,
        resources: ResourceSpec | Mapping[str, Any] | None = None,
        deps: Sequence[int | str] | None = None,
        priority: int = 0,
        time_limit_s: int | None = None,
        cwd: str | Path | None = None,
    ) -> tuple[str, list[str], ResourceSpec | Mapping[str, Any] | None, list[int], int, int | None, str]:
        if isinstance(task_or_cmd, StageTask):
            tid = task_id or task_or_cmd.task_id
            raw_cmd = command
            if raw_cmd is None:
                raise ValueError("command must be provided when submitting a StageTask")
            res = resources if resources is not None else task_or_cmd.resources
            if time_limit_s is not None:
                tlimit: int | None = int(time_limit_s)
            else:
                res_obj = res if isinstance(res, ResourceSpec) else ResourceSpec.from_dict(res or {})
                wall_s = int(res_obj.wall_timeout_s or 0)
                if wall_s <= 0:
                    tlimit = None
                elif task_or_cmd.stage == StageKind.CPU_GATE:
                    tlimit = max(min(wall_s, 180) + 45, 225)
                else:
                    tlimit = wall_s + 45
            work_cwd = str(cwd or task_or_cmd.worktree_path or self.arr_home)
            dep_list = [int(d) for d in (deps or [])]
        else:
            tid = task_id or f"task_{uuid.uuid4().hex[:10]}"
            raw_cmd = command if command is not None else task_or_cmd
            res = resources
            tlimit = time_limit_s
            if tlimit is None and isinstance(resources, ResourceSpec):
                tlimit = resources.wall_timeout_s
            work_cwd = str(cwd or self.arr_home)
            dep_list = [int(d) for d in (deps or [])]

        if isinstance(raw_cmd, str):
            cmd_list = ["/bin/bash", "-lc", raw_cmd]
        else:
            cmd_list = [str(x) for x in raw_cmd]
        if not cmd_list:
            raise ValueError("command cannot be empty")
        return tid, cmd_list, res, dep_list, int(priority), tlimit, work_cwd

    def _build_submit_args(
        self,
        *,
        name: str,
        cmd_list: list[str],
        resources: ResourceSpec | Mapping[str, Any] | None,
        stage: StageKind | str | None = None,
        priority: int,
        time_limit_s: int | None,
        cwd: str,
        env: Mapping[str, str] | None,
        stdout: str,
        stderr: str,
        job_id: int | None = None,
    ) -> list[str]:
        cpus, reqs = resource_spec_to_requests(resources, stage=stage)
        args = ["submit"]
        if job_id is not None:
            args.append(f"--job={int(job_id)}")
        else:
            args.append(f"--name={name}")
        args.extend(
            [
                f"--cpus={cpus}",
                f"--priority={int(priority)}",
                "--cwd",
                cwd,
                "--stdout",
                stdout,
                "--stderr",
                stderr,
            ]
        )
        if time_limit_s is not None and int(time_limit_s) > 0:
            args.append(f"--time-limit={int(time_limit_s)}s")
        for rname, ramt in sorted(reqs.items()):
            args.extend(["--resource", f"{rname}={ramt}"])
        if env:
            for k, v in sorted(env.items()):
                args.append(f"--env={k}={v}")
        args.append("--")
        args.extend(cmd_list)
        return args

    def open_job(self, name: str, *, max_fails: int | None = None) -> int:
        """Open a HyperQueue job container to support hot-insertion."""
        args = ["job", "open", f"--name={name}"]
        if max_fails is not None:
            args.append(f"--max-fails={int(max_fails)}")
        proc = self._run_hq(args, json_mode=True, check=True)
        return int(self._parse_json_stdout(proc.stdout)["id"])

    def close_job(self, job_id: int) -> None:
        """Close an open HyperQueue job."""
        self._run_hq(["job", "close", str(int(job_id))], json_mode=True, check=True)

    def _deps_terminal_state(self, dep_ids: Sequence[int]) -> tuple[bool, bool, str]:
        """Return (all_done, all_ok, detail) for HQ job dependencies."""
        all_done = True
        for dep_id in dep_ids:
            info = self.job_info(dep_id)
            if info is None:
                return True, False, f"missing dependency job {dep_id}"
            jinfo = info.get("info") or {}
            stats = jinfo.get("task_stats") or {}
            if jinfo.get("is_open"):
                all_done = False
                continue
            waiting = int(stats.get("waiting", 0))
            running = int(stats.get("running", 0))
            failed = int(stats.get("failed", 0))
            canceled = int(stats.get("canceled", 0))
            aborted = int(stats.get("aborted", 0))
            if failed > 0 or canceled > 0 or aborted > 0:
                return True, False, f"dependency job {dep_id} terminated abnormally ({stats})"
            if waiting > 0 or running > 0:
                all_done = False
        return all_done, True, ""

    def submit_task(
        self,
        task_or_cmd: StageTask | Sequence[str] | str,
        *,
        task_id: str | None = None,
        name: str | None = None,
        command: Sequence[str] | str | None = None,
        resources: ResourceSpec | Mapping[str, Any] | None = None,
        stage: StageKind | str | None = None,
        deps: Sequence[int | str] | None = None,
        priority: int = 0,
        time_limit_s: int | None = None,
        cwd: str | Path | None = None,
        env: Mapping[str, str] | None = None,
        stdout: str | Path | None = None,
        stderr: str | Path | None = None,
        on_exit_cmd: Sequence[str] | None = None,
        reactor_socket: str | Path | None = None,
        notify_cli: Sequence[str] | None = None,
        wrap_exit: bool = True,
        on_submitted: Callable[[int], None] | None = None,
    ) -> int:
        """Submit a task to HyperQueue, supporting dynamic dependency hot-insertion."""
        effective_stage = stage if stage is not None else (task_or_cmd.stage if isinstance(task_or_cmd, StageTask) else None)
        tid, cmd_list, res, dep_ids, prio, tlimit, work_cwd = self._normalize_task_inputs(
            task_or_cmd,
            task_id=task_id,
            command=command,
            resources=resources,
            deps=deps,
            priority=priority,
            time_limit_s=time_limit_s,
            cwd=cwd,
        )
        out_p, err_p = self._resolve_stdio(tid, stdout, stderr)
        job_name = name or tid
        sock = Path(reactor_socket).expanduser() if reactor_socket else self.reactor_socket
        wrapped_cmd = (
            self._build_exit_wrapper(
                task_id=tid,
                command=cmd_list,
                on_exit_cmd=on_exit_cmd,
                reactor_socket=sock,
                notify_cli=notify_cli,
            )
            if wrap_exit
            else cmd_list
        )
        merged_env = {"CONJECTRA_TASK_ID": tid}
        if env:
            merged_env.update({str(k): str(v) for k, v in env.items()})

        if not dep_ids:
            submit_args = self._build_submit_args(
                name=job_name,
                cmd_list=wrapped_cmd,
                resources=res,
                stage=effective_stage,
                priority=prio,
                time_limit_s=tlimit,
                cwd=work_cwd,
                env=merged_env,
                stdout=out_p,
                stderr=err_p,
            )
            proc = self._run_hq(submit_args, json_mode=True, check=True)
            jid = int(self._parse_json_stdout(proc.stdout)["id"])
            if on_submitted:
                on_submitted(jid)
            return jid

        all_done, all_ok, detail = self._deps_terminal_state(dep_ids)
        if all_done and not all_ok:
            raise RuntimeError(f"Cannot submit {tid}: {detail}")
        if all_done and all_ok:
            submit_args = self._build_submit_args(
                name=job_name,
                cmd_list=wrapped_cmd,
                resources=res,
                stage=effective_stage,
                priority=prio,
                time_limit_s=tlimit,
                cwd=work_cwd,
                env=merged_env,
                stdout=out_p,
                stderr=err_p,
            )
            proc = self._run_hq(submit_args, json_mode=True, check=True)
            jid = int(self._parse_json_stdout(proc.stdout)["id"])
            if on_submitted:
                on_submitted(jid)
            return jid

        # Hot-insert via an open HQ job ID immediately, releasing it once deps complete.
        jid = self.open_job(job_name)
        done_event = threading.Event()
        cancel_event = threading.Event()
        with self._lock:
            self._pending_deps[jid] = {
                "task_id": tid,
                "deps": list(dep_ids),
                "done_event": done_event,
                "cancel_event": cancel_event,
                "canceled": False,
                "dep_failed": False,
            }

        def _wait_and_attach() -> None:
            try:
                for dep_id in dep_ids:
                    while not cancel_event.is_set():
                        # Wait on upstream pending open job if applicable first
                        dep_entry = None
                        with self._lock:
                            dep_entry = self._pending_deps.get(dep_id)
                        if dep_entry is not None:
                            dep_entry["done_event"].wait(timeout=0.5)
                            if cancel_event.is_set():
                                break
                        # `hq job wait` blocks until the dependency job completes
                        self._run_hq(["job", "wait", str(dep_id)], json_mode=False, timeout_s=3600.0)
                        done, ok, _ = self._deps_terminal_state([dep_id])
                        if done:
                            if not ok:
                                with self._lock:
                                    if jid in self._pending_deps:
                                        self._pending_deps[jid]["dep_failed"] = True
                                cancel_event.set()
                            break
                        time.sleep(0.05)

                if cancel_event.is_set():
                    try:
                        self.close_job(jid)
                    except Exception:
                        pass
                    return

                submit_args = self._build_submit_args(
                    name=job_name,
                    cmd_list=wrapped_cmd,
                    resources=res,
                    stage=effective_stage,
                    priority=prio,
                    time_limit_s=tlimit,
                    cwd=work_cwd,
                    env=merged_env,
                    stdout=out_p,
                    stderr=err_p,
                    job_id=jid,
                )
                self._run_hq(submit_args, json_mode=True, check=True)
                self.close_job(jid)
                if on_submitted:
                    on_submitted(jid)
            finally:
                done_event.set()

        t = threading.Thread(target=_wait_and_attach, name=f"hq-dep-{jid}", daemon=True)
        t.start()
        return jid

    def submit_dag(
        self,
        tasks: Sequence[DAGTaskSpec],
        *,
        name: str = "arr-dag",
        max_fails: int | None = None,
        job_id: int | None = None,
        wrap_exit: bool = True,
    ) -> DAGSubmission:
        """Submit a multi-task DAG natively using `hq job submit-file`."""
        if not tasks:
            raise ValueError("DAG task list cannot be empty")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_log_dir.mkdir(parents=True, exist_ok=True)

        id_map: dict[str, int] = {t.task_id: idx + 1 for idx, t in enumerate(tasks)}
        lines: list[str] = [f"name = {_toml_str(name)}"]
        if max_fails is not None:
            lines.append(f"max_fails = {int(max_fails)}")

        for t in tasks:
            hq_tid = id_map[t.task_id]
            dep_ints = [id_map[d] for d in t.deps]
            out_p, err_p = self._resolve_stdio(t.task_id, t.stdout, t.stderr)
            cwd_str = str(t.cwd or self.arr_home)
            cmd_list = (
                self._build_exit_wrapper(
                    task_id=t.task_id,
                    command=t.command,
                    on_exit_cmd=t.on_exit_cmd,
                )
                if wrap_exit
                else list(t.command)
            )
            cpus, reqs = resource_spec_to_requests(t.resources, stage=t.stage)
            res_entries = {"cpus": cpus, **reqs}
            res_toml = ", ".join(f"{k} = {v}" for k, v in res_entries.items())

            lines.append("")
            lines.append("[[task]]")
            lines.append(f"id = {hq_tid}")
            lines.append("command = [" + ", ".join(_toml_str(c) for c in cmd_list) + "]")
            lines.append("deps = [" + ", ".join(str(d) for d in dep_ints) + "]")
            lines.append(f"priority = {int(t.priority)}")
            lines.append(f"cwd = {_toml_str(cwd_str)}")
            lines.append(f"stdout = {_toml_str(out_p)}")
            lines.append(f"stderr = {_toml_str(err_p)}")
            tlimit = t.time_limit_s
            if tlimit is None and isinstance(t.resources, ResourceSpec):
                tlimit = t.resources.wall_timeout_s
            if tlimit is not None and int(tlimit) > 0:
                lines.append(f"time_limit = {_toml_str(f'{int(tlimit)}s')}")
            merged_env = {"CONJECTRA_TASK_ID": t.task_id, **t.env}
            lines.append("[task.env]")
            for ek, ev in sorted(merged_env.items()):
                lines.append(f"{ek} = {_toml_str(ev)}")
            lines.append("[[task.request]]")
            lines.append(f"resources = {{ {res_toml} }}")

        jobfile_path = self.wrappers_dir / f"dag_{uuid.uuid4().hex[:10]}.toml"
        self.wrappers_dir.mkdir(parents=True, exist_ok=True)
        jobfile_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        args = ["job", "submit-file"]
        if job_id is not None:
            args.extend(["--job", str(int(job_id))])
        args.append(str(jobfile_path))
        proc = self._run_hq(args, json_mode=True, check=True)
        parsed = self._parse_json_stdout(proc.stdout)
        returned_jid = int(parsed["id"]) if isinstance(parsed, dict) and "id" in parsed else int(job_id or 0)
        return DAGSubmission(hq_job_id=returned_jid, task_id_map=id_map, jobfile_path=jobfile_path)

    def cancel_jobs(self, job_ids: Sequence[int], reason: str | None = None) -> dict[int, bool]:
        """Cancel one or more HQ jobs (including pending dependency-gated open jobs)."""
        results: dict[int, bool] = {}
        for raw_id in job_ids:
            jid = int(raw_id)
            with self._lock:
                entry = self._pending_deps.get(jid)
            if entry is not None:
                entry["canceled"] = True
                entry["cancel_event"].set()
                entry["done_event"].set()

            args = ["job", "cancel", str(jid)]
            if reason:
                args.extend(["--reason", reason])
            proc = self._run_hq(args, json_mode=True, timeout_s=10.0)
            if proc.returncode == 0:
                results[jid] = True
            else:
                results[jid] = entry is not None
        return results

    def is_pending_canceled_or_dep_failed(self, job_id: int) -> tuple[bool, bool]:
        """Return (canceled, dep_failed) for hot-inserted jobs tracked in memory."""
        with self._lock:
            entry = self._pending_deps.get(int(job_id))
        if not entry:
            return False, False
        return bool(entry["canceled"]), bool(entry["dep_failed"])

    def wait_job(self, job_id: int, timeout_s: float = 120.0) -> dict[str, Any] | None:
        """Wait for an HQ job (including pending hot-inserted dependencies) to reach terminal state."""
        jid = int(job_id)
        deadline = time.monotonic() + timeout_s
        with self._lock:
            entry = self._pending_deps.get(jid)
        if entry is not None:
            rem = max(0.1, deadline - time.monotonic())
            entry["done_event"].wait(timeout=rem)
            if entry["canceled"] or entry["dep_failed"]:
                return self.job_info(jid)

        rem = max(0.5, deadline - time.monotonic())
        self._run_hq(["job", "wait", str(jid)], json_mode=False, timeout_s=rem)
        return self.job_info(jid)
