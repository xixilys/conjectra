#!/usr/bin/env python3
"""Minimal heavy-job slot lock: ``python -m conjectra.slot_lock [--slots N] -- <cmd...>``.

Heavy stages (GPU / long simulation) are wrapped with this helper so that at most ``N``
of them run at once on a host, regardless of how many HyperQueue workers are alive.
Slots are plain ``flock`` files under ``$CONJECTRA_LOCK_DIR`` (default: ``<tmp>/conjectra-locks``).
"""

from __future__ import annotations

import argparse
import fcntl
import os
import subprocess
import tempfile
import time
from pathlib import Path


def default_lock_dir() -> Path:
    return Path(os.environ.get("CONJECTRA_LOCK_DIR") or (Path(tempfile.gettempdir()) / "conjectra-locks"))


def read_mem_available_kb(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    try:
        for line in meminfo_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return 0


def read_load1() -> float:
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):
        return 0.0


def held_slots(lock_dir: Path | None = None) -> list[str]:
    """Names of slot files currently held by some process (non-blocking probe)."""
    d = lock_dir or default_lock_dir()
    held: list[str] = []
    if not d.is_dir():
        return held
    for p in sorted(d.glob("heavy-*.lock")):
        try:
            with p.open("a+", encoding="utf-8") as fh:
                try:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except BlockingIOError:
                    held.append(p.name)
        except OSError:
            continue
    return held


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--slots", type=int, default=int(os.environ.get("CONJECTRA_HEAVY_SLOTS", "1")))
    ap.add_argument("--poll", type=float, default=1.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("missing command after --")
    lock_dir = default_lock_dir()
    lock_dir.mkdir(parents=True, exist_ok=True)
    slots = max(1, int(args.slots))
    while True:
        for i in range(slots):
            fh = (lock_dir / f"heavy-{i}.lock").open("a+", encoding="utf-8")
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                fh.close()
                continue
            try:
                return subprocess.call(cmd)
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                fh.close()
        time.sleep(max(0.05, float(args.poll)))


if __name__ == "__main__":
    raise SystemExit(main())
