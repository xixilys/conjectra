"""Process-group-safe subprocess runner shared by stage execution and CLI backends."""

from __future__ import annotations

import os
import resource
import signal
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

from conjectra.types import ProcessStatus


def run_subprocess(
    cmd: list[str] | str,
    *,
    cwd: Path,
    timeout_s: int | float,
    stdin_text: str | None = None,
    shell: bool = False,
    extra_env: Mapping[str, str] | None = None,
    stage_deadline: float | None = None,
) -> tuple[int, str, str, float, float, ProcessStatus]:
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if extra_env:
        env.update(extra_env)

    eff_timeout = max(0.05, float(timeout_s))
    t0 = time.perf_counter()
    before_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdin=subprocess.PIPE if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=shell,
            env=env,
            start_new_session=True,
        )
    except OSError as exc:
        wall = time.perf_counter() - t0
        return 127, "", str(exc), wall, 0.0, ProcessStatus.ENV_ERROR

    try:
        out, err = proc.communicate(input=stdin_text, timeout=eff_timeout)
        wall = time.perf_counter() - t0
        after_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        max_rss_mb = max(before_rss, after_rss) / 1024.0
        status = ProcessStatus.EXIT_OK if proc.returncode == 0 else ProcessStatus.EXIT_NONZERO
        if proc.returncode in (-9, 137) or "MemoryError" in (err or ""):
            status = ProcessStatus.OOM
        return proc.returncode, out or "", err or "", wall, max_rss_mb, status
    except subprocess.TimeoutExpired:
        pgid = proc.pid
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except OSError:
            proc.terminate()

        if stage_deadline is not None:
            term_wait = min(8.0, max(0.05, (stage_deadline - time.monotonic()) - 1.5))
        else:
            term_wait = 2.0

        try:
            out, err = proc.communicate(timeout=term_wait)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except OSError:
                proc.kill()
            try:
                out, err = proc.communicate(timeout=1.2)
            except subprocess.TimeoutExpired:
                out, err = "", ""
        wall = time.perf_counter() - t0
        return (
            124,
            out or "",
            (err or "") + f"\n[BLOCKED_TIMEOUT after {eff_timeout:.2f}s]",
            wall,
            0.0,
            ProcessStatus.TIMEOUT,
        )
