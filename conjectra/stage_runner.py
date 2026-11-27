"""Stage-split zero-residency execution runner and shared-clone sandbox manager."""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import inspect
import json
import os
import re
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Sequence

from conjectra import gates, planning
from conjectra.adapters import detect_env_config_mismatch, get_adapter, parse_test_counts
from conjectra.backends import DEFAULT_BACKEND_FOR_POOL, BackendRequest, get_backend
from conjectra.backends.builtin import build_dispatch_cmd
from conjectra.backends.proc import run_subprocess
from conjectra.model_router import ModelRouter, PolicyBlockedError, default_execution_host
from conjectra.state_db import StateDB, resolve_arr_home
from conjectra.types import (
    AuditVerdict,
    ConstraintStrength,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    FailureDomain,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ResolvedModelRoute,
    ReviewKind,
    ReviewReceipt,
    ReviewTier,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
)

# Optional fast scratch disk (e.g. a local NVMe mount). Used while it stays below the
# high-water mark; otherwise sandboxes fall back to the state home.
SCRATCH_ROOT = Path(os.environ.get("CONJECTRA_SCRATCH_ROOT") or "/nonexistent-conjectra-scratch")
PRIMARY_SANDBOX_BASE = SCRATCH_ROOT / "sandboxes"
PRIMARY_WT_BASE = SCRATCH_ROOT / "arr-wt"
FALLBACK_SANDBOX_BASE = Path(os.path.expanduser("~/.conjectra/sandboxes"))
FALLBACK_WT_BASE = Path(os.path.expanduser("~/.conjectra/arr-wt"))
DISK_HIGH_WATER_PCT = 85.0

# Optional external dispatcher (see docs/backends.md). When this default path is in effect,
# calls go through the backend registry instead of the script.
DEFAULT_MULTIPOOL_SCRIPT = Path(
    os.path.expanduser(os.environ.get("CONJECTRA_DISPATCH_SCRIPT") or "~/.conjectra/dispatch.py")
)
DEFAULT_SLOT_LOCK_SCRIPT = Path(__file__).resolve().parent / "slot_lock.py"
PROTECTED_BRANCHES = frozenset({"main", "master", "refs/heads/main", "refs/heads/master"})
_MEM_AVAIL_RE = re.compile(r"^MemAvailable:\s+(\d+)\s+kB$", re.MULTILINE)


_VM_STAT_PAGE_RE = re.compile(r"page size of (\d+) bytes")
_VM_STAT_LINE_RE = re.compile(r"^Pages (free|inactive|speculative|purgeable):\s+(\d+)\.?$", re.MULTILINE)


def parse_vm_stat_available_mb(text: str) -> int:
    """Approximate available memory (MiB) from macOS `vm_stat` output: free + inactive + speculative + purgeable."""
    page_m = _VM_STAT_PAGE_RE.search(text)
    page_size = int(page_m.group(1)) if page_m else 4096
    pages = sum(int(n) for _kind, n in _VM_STAT_LINE_RE.findall(text))
    return (pages * page_size) // (1024 * 1024)


def read_mem_available_mb(meminfo_path: Path = Path("/proc/meminfo")) -> int:
    """Read MemAvailable in MiB from /proc/meminfo, or `vm_stat` on macOS (returns 0 if unreadable)."""
    try:
        text = meminfo_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        if sys.platform == "darwin" and meminfo_path == Path("/proc/meminfo"):
            try:
                proc = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5, check=False)
            except (OSError, subprocess.SubprocessError):
                return 0
            return parse_vm_stat_available_mb(proc.stdout) if proc.returncode == 0 else 0
        return 0
    m = _MEM_AVAIL_RE.search(text)
    if not m:
        return 0
    return int(m.group(1)) // 1024


def wait_for_mem_available(
    min_mb: int | None = None,
    max_wait_s: float = 180.0,
    poll_s: float = 2.0,
) -> tuple[bool, int]:
    """Wait until /proc/meminfo MemAvailable >= min_mb or max_wait_s elapses."""
    target_mb = int(os.environ.get("CONJECTRA_MIN_AVAIL_RAM_MB", "10240")) if min_mb is None else int(min_mb)
    avail_mb = read_mem_available_mb()
    if target_mb <= 0 or avail_mb >= target_mb:
        return True, avail_mb

    deadline = time.monotonic() + max(0.0, float(max_wait_s))
    step = max(0.02, float(poll_s))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(step, remaining))
        avail_mb = read_mem_available_mb()
        if avail_mb >= target_mb:
            return True, avail_mb
    avail_mb = read_mem_available_mb()
    return avail_mb >= target_mb, avail_mb


def _prefix_nice(cmd: str, niceness: int = 10) -> str:
    """Wrap a shell command with `nice -n <niceness>` preserving env vars and chains."""
    stripped = cmd.strip()
    prefix = f"nice -n {int(niceness)} "
    if stripped.startswith(prefix):
        return cmd
    return f"{prefix}bash -lc {shlex.quote(cmd)}"


def _default_disk_used_pct(path: Path) -> float:
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    usage = shutil.disk_usage(probe)
    if usage.total <= 0:
        return 100.0
    return (usage.used / usage.total) * 100.0


def _sanitize_rel(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._/-]+", "-", value.strip()).strip("/")
    parts = [p for p in cleaned.split("/") if p and p not in (".", "..")]
    return "/".join(parts) or "branch"


def _git(
    repo: Path,
    *args: str,
    check: bool = False,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=check,
        timeout=timeout,
    )


class SandboxError(RuntimeError):
    """Raised when sandbox or worktree isolation cannot be safely established."""


class SandboxManager:
    """Non-intrusive shared-clone mirror and sparse isolated worktree manager."""

    def __init__(
        self,
        *,
        sandbox_base: Path | None = None,
        worktree_base: Path | None = None,
        disk_used_pct_fn: Callable[[Path], float] | None = None,
        disk_threshold_pct: float = DISK_HIGH_WATER_PCT,
    ) -> None:
        self._explicit_sandbox_base = sandbox_base
        self._explicit_wt_base = worktree_base
        self._disk_used_pct_fn = disk_used_pct_fn or _default_disk_used_pct
        self.disk_threshold_pct = float(disk_threshold_pct)

    def resolve_roots(self) -> tuple[Path, Path]:
        """Select the scratch-disk roots, or the state-home fallback when the scratch disk is >= 85% full."""
        if self._explicit_sandbox_base is not None or self._explicit_wt_base is not None:
            sb = self._explicit_sandbox_base or FALLBACK_SANDBOX_BASE
            wt = self._explicit_wt_base or FALLBACK_WT_BASE
            return sb, wt

        env_sb = os.environ.get("CONJECTRA_SANDBOX_ROOT")
        env_wt = os.environ.get("CONJECTRA_WORKTREE_ROOT")
        if env_sb or env_wt:
            return (
                Path(env_sb).expanduser() if env_sb else FALLBACK_SANDBOX_BASE,
                Path(env_wt).expanduser() if env_wt else FALLBACK_WT_BASE,
            )

        primary_mount = SCRATCH_ROOT
        try:
            if primary_mount.exists():
                used_pct = self._disk_used_pct_fn(primary_mount)
                if used_pct < self.disk_threshold_pct:
                    return PRIMARY_SANDBOX_BASE, PRIMARY_WT_BASE
        except OSError:
            pass
        return FALLBACK_SANDBOX_BASE, FALLBACK_WT_BASE

    @contextlib.contextmanager
    def _mirror_flock(self, lock_path: Path) -> Iterator[None]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lfh:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lfh.fileno(), fcntl.LOCK_UN)

    def ensure_mirror(self, origin_repo: str | Path, project: str) -> Path:
        """Create or verify a shared-clone mirror without ever writing to origin .git."""
        origin = Path(origin_repo).expanduser().resolve()
        if not origin.exists():
            raise SandboxError(f"Origin repository does not exist: {origin}")

        sb_base, _ = self.resolve_roots()
        safe_proj = _sanitize_rel(project)
        mirror = sb_base / safe_proj
        lock_path = sb_base / f".arr_mirror_{safe_proj.replace('/', '_')}.lock"
        with self._mirror_flock(lock_path):
            if (mirror / ".git").exists():
                return mirror

            sb_base.mkdir(parents=True, exist_ok=True)
            proc = subprocess.run(
                ["git", "clone", "--shared", "--no-checkout", str(origin), str(mirror)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if proc.returncode != 0:
                raise SandboxError(
                    f"git clone --shared --no-checkout failed for {origin} -> {mirror}: {proc.stderr.strip()}"
                )
        return mirror

    def _ensure_commit_in_mirror_unlocked(
        self, mirror: Path, origin_repo: Path | None, commit_sha: str
    ) -> str:
        rev = _git(mirror, "rev-parse", "--verify", f"{commit_sha}^{{commit}}")
        if rev.returncode == 0:
            return rev.stdout.strip()
        if origin_repo is not None:
            _git(mirror, "fetch", "--no-tags", str(origin_repo), commit_sha)
            rev2 = _git(mirror, "rev-parse", "--verify", f"{commit_sha}^{{commit}}")
            if rev2.returncode == 0:
                return rev2.stdout.strip()
        head_rev = _git(mirror, "rev-parse", "--verify", "HEAD^{commit}")
        if commit_sha in ("", "HEAD") and head_rev.returncode == 0:
            return head_rev.stdout.strip()
        raise SandboxError(f"Commit {commit_sha!r} not found in shared mirror {mirror}")

    def _ensure_commit_in_mirror(self, mirror: Path, origin_repo: Path | None, commit_sha: str) -> str:
        lock_path = mirror.parent / f".arr_mirror_{mirror.name}.lock"
        with self._mirror_flock(lock_path):
            return self._ensure_commit_in_mirror_unlocked(mirror, origin_repo, commit_sha)

    def create_worktree(
        self,
        mirror_dir: str | Path,
        project: str,
        branch_name: str,
        base_sha: str,
        *,
        origin_repo: str | Path | None = None,
        sparse_paths: Sequence[str] | None = None,
        goal_id: str | None = None,
        workflow_id: str | None = None,
    ) -> Path:
        """Create or reuse an isolated worktree on a competitive research branch."""
        if branch_name.strip() in PROTECTED_BRANCHES:
            raise SandboxError(f"Refusing to create worktree on protected branch: {branch_name}")

        mirror = Path(mirror_dir)
        origin = Path(origin_repo).expanduser().resolve() if origin_repo else None

        _, wt_base = self.resolve_roots()
        safe_proj = _sanitize_rel(project)
        safe_branch = _sanitize_rel(branch_name)
        eff_branch = branch_name.strip()
        lock_path = mirror.parent / f".arr_mirror_{mirror.name}.lock"

        with self._mirror_flock(lock_path):
            owners_path = mirror.parent / f".arr_branch_owners_{mirror.name}.json"
            owners: dict[str, str] = {}
            if owners_path.is_file():
                try:
                    loaded = json.loads(owners_path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        owners = {str(k): str(v) for k, v in loaded.items()}
                except (OSError, json.JSONDecodeError):
                    owners = {}

            needs_namespace = False
            if workflow_id:
                needs_namespace = True
            elif goal_id:
                prev_owner = owners.get(eff_branch)
                if prev_owner is not None and prev_owner != goal_id:
                    needs_namespace = True

            if needs_namespace and (goal_id or workflow_id):
                rel_branch = safe_branch[4:] if safe_branch.startswith("arr/") else safe_branch
                parts = ["arr"]
                if goal_id:
                    parts.append(_sanitize_rel(goal_id))
                if workflow_id:
                    parts.append(_sanitize_rel(workflow_id))
                parts.append(rel_branch)
                eff_branch = "/".join(parts)
                wt_rel = Path(*parts[1:])
                wt_path = wt_base / safe_proj / wt_rel
            else:
                wt_path = wt_base / safe_proj / safe_branch
                if goal_id and eff_branch not in owners:
                    owners[eff_branch] = goal_id
                    owners_path.write_text(
                        json.dumps(owners, sort_keys=True) + "\n", encoding="utf-8"
                    )

            if eff_branch in PROTECTED_BRANCHES:
                raise SandboxError(f"Refusing to create worktree on protected branch: {eff_branch}")
            wt_path.parent.mkdir(parents=True, exist_ok=True)

            resolved_base = self._ensure_commit_in_mirror_unlocked(mirror, origin, base_sha)

            if wt_path.exists() and (wt_path / ".git").exists():
                if sparse_paths:
                    self._apply_sparse_checkout(wt_path, sparse_paths)
                return wt_path

            branch_exists = _git(mirror, "rev-parse", "--verify", f"refs/heads/{eff_branch}").returncode == 0
            if branch_exists:
                cmd = ["worktree", "add", "--force", str(wt_path), eff_branch]
            else:
                cmd = ["worktree", "add", "--force", "-b", eff_branch, str(wt_path), resolved_base]

            proc = _git(mirror, *cmd)
            if proc.returncode != 0:
                raise SandboxError(f"git worktree add failed for {wt_path}: {proc.stderr.strip()}")

        if sparse_paths:
            self._apply_sparse_checkout(wt_path, sparse_paths)
        return wt_path

    def _apply_sparse_checkout(self, wt_path: Path, sparse_paths: Sequence[str]) -> None:
        cleaned = [p.strip().lstrip("./") for p in sparse_paths if p and p.strip()]
        if not cleaned:
            return
        init_proc = _git(wt_path, "sparse-checkout", "init", "--no-cone")
        if init_proc.returncode != 0:
            raise SandboxError(f"sparse-checkout init failed: {init_proc.stderr.strip()}")
        set_proc = _git(wt_path, "sparse-checkout", "set", "--no-cone", *cleaned)
        if set_proc.returncode != 0:
            raise SandboxError(f"sparse-checkout set failed: {set_proc.stderr.strip()}")

    def cleanup_worktree(
        self,
        mirror_dir: str | Path,
        wt_path: str | Path,
        *,
        branch_name: str | None = None,
        delete_branch: bool = False,
    ) -> bool:
        """Safely retire an isolated worktree without touching origin repo or main."""
        mirror = Path(mirror_dir)
        wt = Path(wt_path)
        removed = False
        if wt.exists():
            proc = _git(mirror, "worktree", "remove", "--force", str(wt))
            if proc.returncode == 0:
                removed = True
            else:
                shutil.rmtree(wt, ignore_errors=True)
                removed = not wt.exists()
        _git(mirror, "worktree", "prune")
        if delete_branch and branch_name and branch_name.strip() not in PROTECTED_BRANCHES:
            _git(mirror, "branch", "-D", branch_name)
        return removed

    @staticmethod
    def _strip_write_permissions(root: Path) -> None:
        """Recursively strip write bits (`chmod -R a-w`) on a detached review snapshot."""
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            dp = Path(dirpath)
            for fn in filenames:
                fp = dp / fn
                if fp.is_symlink():
                    continue
                try:
                    mode = stat.S_IMODE(fp.lstat().st_mode)
                    fp.chmod(mode & ~0o222)
                except OSError:
                    pass
            if not dp.is_symlink():
                try:
                    dmode = stat.S_IMODE(dp.lstat().st_mode)
                    dp.chmod(dmode & ~0o222)
                except OSError:
                    pass

    @staticmethod
    def _restore_owner_write_permissions(root: Path) -> None:
        """Restore owner write bits (`u+w`) prior to removing a read-only snapshot worktree."""
        if not root.exists():
            return
        for dirpath, _dirnames, filenames in os.walk(root, topdown=True):
            dp = Path(dirpath)
            if not dp.is_symlink():
                try:
                    dmode = stat.S_IMODE(dp.lstat().st_mode)
                    dp.chmod(dmode | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRUSR)
                except OSError:
                    pass
            for fn in filenames:
                fp = dp / fn
                if fp.is_symlink():
                    continue
                try:
                    mode = stat.S_IMODE(fp.lstat().st_mode)
                    fp.chmod(mode | stat.S_IWUSR | stat.S_IRUSR)
                except OSError:
                    pass

    def create_readonly_snapshot(
        self,
        mirror_dir: str | Path,
        project: str,
        task_id: str,
        attempt: int,
        head_sha: str,
        *,
        origin_repo: str | Path | None = None,
    ) -> Path:
        """Create a per-attempt detached worktree under `arr-wt-ro/<proj>/<task_id>_a<attempt>` and `chmod -R a-w`."""
        mirror = Path(mirror_dir)
        origin = Path(origin_repo).expanduser().resolve() if origin_repo else None
        _, wt_base = self.resolve_roots()
        safe_proj = _sanitize_rel(project)
        safe_tid = _sanitize_rel(task_id).replace("/", "_")
        ro_wt = wt_base.parent / "arr-wt-ro" / safe_proj / f"{safe_tid}_a{int(attempt)}"
        lock_path = mirror.parent / f".arr_mirror_{mirror.name}.lock"

        with self._mirror_flock(lock_path):
            resolved_sha = self._ensure_commit_in_mirror_unlocked(mirror, origin, head_sha)
            if ro_wt.exists():
                self._restore_owner_write_permissions(ro_wt)
                _git(mirror, "worktree", "remove", "--force", str(ro_wt))
                if ro_wt.exists():
                    shutil.rmtree(ro_wt, ignore_errors=True)
            ro_wt.parent.mkdir(parents=True, exist_ok=True)
            proc = _git(mirror, "worktree", "add", "--force", "--detach", str(ro_wt), resolved_sha)
            if proc.returncode != 0:
                raise SandboxError(
                    f"git worktree add --detach failed for RO snapshot {ro_wt}: {proc.stderr.strip()}"
                )

        self._strip_write_permissions(ro_wt)
        return ro_wt

    def cleanup_readonly_snapshot(self, mirror_dir: str | Path, ro_wt: str | Path) -> bool:
        """Restore `u+w` and remove a detached read-only review snapshot worktree."""
        wt = Path(ro_wt)
        self._restore_owner_write_permissions(wt)
        return self.cleanup_worktree(mirror_dir, wt, delete_branch=False)

    def create_attempt_scratch(
        self,
        mirror_dir: str | Path,
        project: str,
        task_id: str,
        attempt: int,
        base_head_sha: str,
        *,
        origin_repo: str | Path | None = None,
        sparse_paths: Sequence[str] | None = None,
    ) -> Path:
        """Create an attempt-specific detached git scratch worktree under `arr-wt-scratch/<proj>/`."""
        mirror = Path(mirror_dir)
        origin = Path(origin_repo).expanduser().resolve() if origin_repo else None
        _, wt_base = self.resolve_roots()
        safe_proj = _sanitize_rel(project)
        safe_tid = _sanitize_rel(task_id).replace("/", "_")
        scratch_wt = (
            wt_base.parent
            / "arr-wt-scratch"
            / safe_proj
            / f"{safe_tid}_a{int(attempt)}_{uuid.uuid4().hex[:6]}"
        )
        lock_path = mirror.parent / f".arr_mirror_{mirror.name}.lock"

        with self._mirror_flock(lock_path):
            resolved_sha = self._ensure_commit_in_mirror_unlocked(mirror, origin, base_head_sha)
            if scratch_wt.exists():
                _git(mirror, "worktree", "remove", "--force", str(scratch_wt))
                if scratch_wt.exists():
                    shutil.rmtree(scratch_wt, ignore_errors=True)
            scratch_wt.parent.mkdir(parents=True, exist_ok=True)
            proc = _git(mirror, "worktree", "add", "--force", "--detach", str(scratch_wt), resolved_sha)
            if proc.returncode != 0:
                raise SandboxError(
                    f"git worktree add --detach failed for attempt scratch {scratch_wt}: {proc.stderr.strip()}"
                )

        if sparse_paths:
            self._apply_sparse_checkout(scratch_wt, sparse_paths)
        return scratch_wt

    def cleanup_attempt_scratch(self, mirror_dir: str | Path, scratch_wt: str | Path) -> bool:
        """Force-remove a detached attempt scratch worktree and prune mirror worktree metadata."""
        mirror = Path(mirror_dir)
        wt = Path(scratch_wt)
        lock_path = mirror.parent / f".arr_mirror_{mirror.name}.lock"
        with self._mirror_flock(lock_path):
            return self.cleanup_worktree(mirror, wt, delete_branch=False)


def _int_from_mapping(mapping: Mapping[str, Any], *keys: str) -> int:
    for k in keys:
        v = mapping.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return int(v)
    return 0


def extract_token_breakdown(usage_doc: Mapping[str, Any] | None) -> tuple[int, int, int, int, str]:
    """Extract normalized `(uncached_in, out_tok, cache_read, cache_write, model_used)` from `--usage-out` JSON."""
    if not isinstance(usage_doc, Mapping):
        return 0, 0, 0, 0, "unknown"

    model = str(usage_doc.get("model") or usage_doc.get("route") or "unknown")
    route_low = str(usage_doc.get("route") or "").lower()
    model_low = model.lower()
    raw_in = 0
    out_tok = 0
    cache_read = 0
    cache_write = 0

    u = usage_doc.get("usage")
    is_anthropic = route_low == "claude_cli" or any(
        k in model_low for k in ("claude", "opus", "sonnet", "haiku", "anthropic")
    )
    if isinstance(u, Mapping):
        if any(
            k in u
            for k in (
                "cache_read_input_tokens",
                "cacheReadInputTokens",
                "cache_creation_input_tokens",
                "cacheCreationInputTokens",
            )
        ):
            is_anthropic = True

        raw_in = _int_from_mapping(
            u, "input_tokens", "inputTokens", "prompt_tokens", "promptTokens", "input"
        )
        out_tok = _int_from_mapping(
            u,
            "output_tokens",
            "outputTokens",
            "completion_tokens",
            "candidates_tokens",
            "candidatesTokens",
            "output",
        )
        pt_details = u.get("prompt_tokens_details") if isinstance(u.get("prompt_tokens_details"), Mapping) else {}
        in_details = u.get("input_tokens_details") if isinstance(u.get("input_tokens_details"), Mapping) else {}
        cache_read = (
            _int_from_mapping(
                u,
                "cache_read_tokens",
                "cacheReadTokens",
                "cache_read_input_tokens",
                "cacheReadInputTokens",
                "cached_input_tokens",
                "cachedContentTokenCount",
            )
            or _int_from_mapping(pt_details, "cached_tokens")
            or _int_from_mapping(in_details, "cached_tokens")
        )
        cache_write = _int_from_mapping(
            u,
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
            "cache_creation_tokens",
            "cacheCreationTokens",
            "cache_write_tokens",
            "cacheWriteTokens",
        )

    mu = usage_doc.get("modelUsage")
    if (raw_in == 0 and out_tok == 0) and isinstance(mu, Mapping):
        is_anthropic = True
        for _m_name, stats in mu.items():
            if isinstance(stats, Mapping):
                raw_in += _int_from_mapping(stats, "inputTokens", "input_tokens")
                out_tok += _int_from_mapping(stats, "outputTokens", "output_tokens")
                cache_read += _int_from_mapping(
                    stats,
                    "cacheReadInputTokens",
                    "cache_read_input_tokens",
                    "cacheReadTokens",
                    "cache_read_tokens",
                )
                cache_write += _int_from_mapping(
                    stats,
                    "cacheCreationInputTokens",
                    "cache_creation_input_tokens",
                    "cacheCreationTokens",
                    "cache_creation_tokens",
                )

    is_openai = not is_anthropic and (
        route_low == "codex_cli"
        or any(k in model_low for k in ("gpt-", "codex", "openai"))
        or (isinstance(u, Mapping) and ("cached_input_tokens" in u or "prompt_tokens_details" in u or "input_tokens_details" in u))
    )
    if is_openai and cache_read > 0 and raw_in >= cache_read:
        uncached_in = raw_in - cache_read
    elif not is_anthropic and isinstance(u, Mapping) and "cachedContentTokenCount" in u and raw_in >= cache_read > 0:
        uncached_in = raw_in - cache_read
    else:
        uncached_in = raw_in

    return max(0, uncached_in), max(0, out_tok), max(0, cache_read), max(0, cache_write), model


def extract_token_usage(usage_doc: Mapping[str, Any] | None) -> tuple[int, int, str]:
    """Extract uncached `(in_tokens, out_tokens, model_used)` from `--usage-out` JSON."""
    uncached_in, out_tok, _cr, _cw, model = extract_token_breakdown(usage_doc)
    return uncached_in, out_tok, model


def extract_cache_tokens(usage_doc: Mapping[str, Any] | None) -> int:
    """Extract `cache_read` tokens from `--usage-out` JSON."""
    _in, _out, cache_read, _cw, _mod = extract_token_breakdown(usage_doc)
    return cache_read


def check_live_gpu_vram(min_free_mb: int = 1500) -> tuple[bool, dict[str, Any]]:
    """Query nvidia-smi for available GPU VRAM before launching GPU confirm runs."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, {"error": f"nvidia-smi unavailable: {exc}"}

    if proc.returncode != 0:
        return False, {"error": proc.stderr.strip() or f"nvidia-smi rc={proc.returncode}"}

    gpus: list[dict[str, Any]] = []
    max_free = -1
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            free_mb = int(float(parts[2]))
            total_mb = int(float(parts[3]))
            util_pct = int(float(parts[4]))
        except ValueError:
            continue
        gpus.append(
            {
                "index": parts[0],
                "name": parts[1],
                "free_mb": free_mb,
                "total_mb": total_mb,
                "util_pct": util_pct,
            }
        )
        max_free = max(max_free, free_mb)

    if not gpus:
        return False, {"error": "No GPUs parsed from nvidia-smi"}
    return max_free >= min_free_mb, {"gpus": gpus, "max_free_mb": max_free, "min_required_mb": min_free_mb}


def notify_controller(arr_home: Path, payload: Mapping[str, Any]) -> bool:
    """Send instant wakeup notification to <CONJECTRA_HOME>/reactor.sock or conjectra notify-exit."""
    sock_path = arr_home / "reactor.sock"
    msg = (json.dumps(dict(payload), sort_keys=True) + "\n").encode("utf-8")
    if sock_path.exists():
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(2.0)
                sock.connect(str(sock_path))
                sock.sendall(msg)
            return True
        except OSError:
            pass

    conjectra_bin = shutil.which("conjectra") or str(Path(__file__).resolve().parents[1] / "bin" / "conjectra")
    if Path(conjectra_bin).is_file():
        task_id = str(payload.get("task_id") or "")
        rc = str(payload.get("exit_code", 0))
        try:
            proc = subprocess.run(
                [conjectra_bin, "notify-exit", "--task-id", task_id, "--exit-code", rc],
                env={**os.environ, "CONJECTRA_HOME": str(arr_home)},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=5,
                check=False,
            )
            return proc.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False
    return False




_run_subprocess = run_subprocess


def _resolve_pool_and_model(
    llm_pool: str | None,
    *,
    for_audit: bool = False,
    author_model: str = "",
) -> tuple[str, str]:
    """Map a loose pool/model token to `(pool, model)` (legacy path used when no route is pinned)."""
    raw = (llm_pool or "").strip()
    if raw:
        low = raw.lower()
        if ":" in raw:
            parts = [p.strip() for p in raw.split(":")]
            return parts[0], parts[1]
        if low == "fake" or low == "llm_fake":
            return "fake", "fake-model"
        if low.startswith("gemini-"):
            return "gemini", raw
        if "gpt-" in low or low in ("codex", "llm_codex_audit", "llmcodexaudit"):
            return "codex", "gpt-5-codex" if low in ("codex", "llm_codex_audit", "llmcodexaudit") else raw
        if "deepseek" in low or low in ("llm_deepseek", "llmdeepseek"):
            return "deepseek", (
                "deepseek-chat" if low in ("deepseek", "llm_deepseek", "llmdeepseek") else raw
            )
        if "grok" in low:
            return "grok", "grok-code-fast"
        if "claude" in low or "opus" in low or "sonnet" in low or low == "llmclaude":
            if "sonnet" in low:
                return "claude", "claude-sonnet-5-5"
            return "claude", "claude-opus-5-5" if low in ("claude", "llm_claude", "llmclaude", "opus") else raw
        if low in ("gemini", "gemini-cli", "llm_gemini", "llmgemini"):
            return "gemini", "gemini-pro" if for_audit else "gemini-flash"

    if for_audit:
        # Cross-family default: never let the author's model family audit its own work.
        a_low = (author_model or "").lower()
        if "gemini" in a_low:
            return "codex", "gpt-5-codex"
        return "gemini", "gemini-pro"
    return "gemini", "gemini-flash"


def _build_multipool_cmd(
    script_path: Path,
    pool: str,
    model: str,
    cwd: Path,
    out_md: Path,
    usage_out: Path,
    timeout_s: int,
    tag: str,
    *,
    readonly: bool,
    effort: str | None = None,
    multipool_host_arg: str | None = None,
) -> list[str]:
    """Back-compat wrapper around :func:`conjectra.backends.builtin.build_dispatch_cmd`."""
    return build_dispatch_cmd(
        script_path,
        pool,
        model,
        cwd,
        out_md,
        usage_out,
        timeout_s,
        tag,
        readonly=readonly,
        effort=effort,
        host_arg=multipool_host_arg,
    )


def _dispatch_effort(pool: str, wire_effort: str) -> str:
    """Clamp efforts to the subset accepted by the reference dispatcher contract."""
    if pool in ("claude", "codex") and wire_effort == "xhigh":
        return "max"
    if pool == "deepseek" and wire_effort == "medium":
        return "high"
    if pool == "deepseek" and wire_effort == "xhigh":
        return "max"
    if pool == "grok":
        return "high"
    return wire_effort


def _run_llm_pool(
    *,
    multipool: Path,
    pool: str,
    model: str,
    cwd: Path,
    out_md: Path,
    usage_out: Path,
    timeout_s: int,
    tag: str,
    prompt: str,
    readonly: bool,
    effort: str | None = None,
    pinned_route: Mapping[str, Any] | ResolvedModelRoute | None = None,
) -> tuple[int, str, str, float, float, ProcessStatus, str]:
    """Execute a single-attempt LLM invocation through the backend plugin registry.

    When ``multipool`` is the default dispatcher path, the route's ``backend_mode`` picks a
    registered backend (``claude_cli``, ``codex_cli``, ``fake``, ...). An explicitly supplied
    dispatcher script always wins and is invoked through the ``dispatch_script`` backend.
    ``CONJECTRA_BACKEND_OVERRIDE`` forces every call onto one backend (e.g. ``fake`` in CI).
    """
    is_default_mp = multipool.expanduser().resolve() == DEFAULT_MULTIPOOL_SCRIPT.expanduser().resolve()

    eff_pool = pool
    eff_model = model
    wire_effort = (effort or "high").strip()
    backend_mode = "dispatch_script"
    mp_host_arg: str | None = None

    if isinstance(pinned_route, ResolvedModelRoute):
        route_obj: ResolvedModelRoute | None = pinned_route
    elif isinstance(pinned_route, Mapping):
        route_obj = ResolvedModelRoute.from_dict(pinned_route)
    else:
        route_obj = None
        if is_default_mp:
            try:
                spec_str = f"{pool}:{model}:{effort}" if effort else f"{pool}:{model}"
                role_hint = "impl_reviewer" if readonly else "engineer_builder"
                route_obj = ModelRouter().resolve_route(
                    role_hint,
                    explicit_spec=spec_str,
                    execution_host=default_execution_host(),
                    strength=ConstraintStrength.SOFT_FALLBACK,
                )
            except Exception:
                route_obj = None

    if route_obj is not None:
        eff_pool = route_obj.pool
        eff_model = route_obj.canonical_model
        wire_effort = route_obj.wire_effort
        backend_mode = route_obj.backend_mode if is_default_mp else "dispatch_script"
        mp_host_arg = route_obj.multipool_host_arg
    elif is_default_mp:
        backend_mode = DEFAULT_BACKEND_FOR_POOL.get(eff_pool, "dispatch_script")

    forced = str(os.environ.get("CONJECTRA_BACKEND_OVERRIDE") or "").strip()
    if forced and is_default_mp:
        backend_mode = forced

    if backend_mode == "dispatch_script":
        wire_effort = _dispatch_effort(eff_pool, wire_effort)

    backend = get_backend(backend_mode)
    if backend is None:
        msg = f"BLOCKED_ENV: no backend registered for backend_mode={backend_mode!r} (pool={eff_pool!r})"
        return 127, "", msg, 0.0, 0.0, ProcessStatus.ENV_ERROR, f"backend:{backend_mode}"

    request = BackendRequest(
        pool=eff_pool,
        model=eff_model,
        effort=wire_effort,
        prompt=prompt,
        cwd=cwd,
        out_md=out_md,
        usage_out=usage_out,
        timeout_s=int(timeout_s),
        tag=tag,
        readonly=readonly,
        dispatch_script=multipool,
        dispatch_host_arg=mp_host_arg,
        route=route_obj,
    )
    return backend.invoke(request).as_tuple()


def _commit_worktree_changes(wt_path: Path, hyp: HypothesisNode) -> tuple[str, bool]:
    """Commit isolated worktree changes on the competitive branch without touching main."""
    head_proc = _git(wt_path, "rev-parse", "HEAD")
    cur_head = head_proc.stdout.strip() if head_proc.returncode == 0 else hyp.base_sha
    status = _git(wt_path, "status", "--porcelain")
    if status.returncode != 0 or not status.stdout.strip():
        return cur_head, False

    branch_proc = _git(wt_path, "rev-parse", "--abbrev-ref", "HEAD")
    cur_branch = branch_proc.stdout.strip()
    if cur_branch in PROTECTED_BRANCHES:
        raise SandboxError(f"Refusing commit on protected branch {cur_branch}")

    _git(wt_path, "add", "-A")
    msg = f"arr: {hyp.hypothesis_id} {hyp.title[:60]}"
    cproc = _git(
        wt_path,
        "-c",
        "user.name=conjectra",
        "-c",
        "user.email=conjectra@local",
        "commit",
        "-m",
        msg,
    )
    if cproc.returncode != 0:
        raise SandboxError(f"git commit failed in {wt_path}: {cproc.stderr.strip()}")
    new_head = _git(wt_path, "rev-parse", "HEAD").stdout.strip()
    return new_head, True


def _parse_audit_verdict(text: str, *, mode: str = "audit") -> AuditVerdict:
    parsed = gates.parse_structured_verdict(text, mode=mode)
    if parsed.verdict == "PASS":
        return AuditVerdict.PASS
    if parsed.verdict == "PARSE_ERROR":
        return AuditVerdict.PARSE_ERROR
    return AuditVerdict.FAIL


def _infer_review_slot_index(
    step_key: str, raw_pinned: Mapping[str, Any] | None = None
) -> int:
    if isinstance(raw_pinned, Mapping) and raw_pinned.get("review_slot") is not None:
        return int(raw_pinned["review_slot"])
    key = (step_key or "").strip().lower()
    if "tiebreaker" in key:
        return 2
    m_end = re.search(r"(?:_slot_|_s|:)(\d+)$", key)
    if m_end:
        return int(m_end.group(1))
    m = re.search(r"slot[_-]?(\d+)", key)
    if m:
        return int(m.group(1))
    return 0


@dataclass
class StageRunner:
    """Executes single stage tasks in isolation and emits EvidenceReceipts."""

    arr_home: Path | None = None
    sandbox_manager: SandboxManager | None = None
    multipool_script: Path | None = None
    slot_lock_script: Path | None = None
    model_router: ModelRouter | None = None

    def _resolve_worktree(
        self,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
    ) -> tuple[Path, str]:
        if task.worktree_path and Path(task.worktree_path).exists():
            wt = Path(task.worktree_path)
            rev = _git(wt, "rev-parse", "HEAD")
            head_sha = rev.stdout.strip() if rev.returncode == 0 else (hyp.head_sha or hyp.base_sha)
            return wt, head_sha

        repo_path = Path(goal.repo_path)
        if repo_path.exists() and (repo_path / ".git").exists():
            sb = self.sandbox_manager or SandboxManager()
            mirror = sb.ensure_mirror(repo_path, goal.project)
            wt = sb.create_worktree(
                mirror,
                goal.project,
                hyp.branch_name,
                hyp.head_sha or hyp.base_sha,
                origin_repo=repo_path,
                goal_id=goal.goal_id,
                workflow_id=task.workflow_id,
            )
            rev = _git(wt, "rev-parse", "HEAD")
            head_sha = rev.stdout.strip() if rev.returncode == 0 else (hyp.head_sha or hyp.base_sha)
            return wt, head_sha

        raise SandboxError(f"No valid worktree or git repository available for {goal.repo_path}")

    @staticmethod
    def _raw_pinned_dict(task: StageTask) -> dict[str, Any]:
        if isinstance(task.pinned_route, Mapping):
            return dict(task.pinned_route)
        if isinstance(task.resources.pinned_route, Mapping):
            return dict(task.resources.pinned_route)
        return {}

    @contextlib.contextmanager
    def _readonly_review_worktree(
        self,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        fallback_wt: Path,
        fallback_head: str,
    ) -> Iterator[tuple[Path, str]]:
        """Yield a per-attempt `chmod -R a-w` detached snapshot worktree and clean it up in `finally`."""
        raw_pinned = self._raw_pinned_dict(task)
        pinned_head = str(raw_pinned.get("head_sha") or "").strip()
        repo_path = Path(goal.repo_path)
        source_repo = fallback_wt if (fallback_wt / ".git").exists() else repo_path
        if repo_path.exists() and (repo_path / ".git").exists():
            sb = self.sandbox_manager or SandboxManager()
            mirror = sb.ensure_mirror(repo_path, goal.project)
            target_sha = pinned_head or fallback_head or hyp.head_sha or hyp.base_sha
            ro_wt = sb.create_readonly_snapshot(
                mirror,
                goal.project,
                task.task_id,
                task.attempt,
                target_sha,
                origin_repo=source_repo,
            )
            try:
                rev = _git(ro_wt, "rev-parse", "HEAD")
                ro_head = rev.stdout.strip() if rev.returncode == 0 else target_sha
                yield ro_wt, ro_head
            finally:
                sb.cleanup_readonly_snapshot(mirror, ro_wt)
            return

        yield fallback_wt, (pinned_head or fallback_head)

    @contextlib.contextmanager
    def _writable_attempt_scratch(
        self,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        wt_path: Path,
        base_head_sha: str,
    ) -> Iterator[Path]:
        """Yield an attempt-specific detached git scratch worktree and remove/prune it in `finally`."""
        repo_path = Path(goal.repo_path)
        source_repo = wt_path if (wt_path / ".git").exists() else repo_path
        if repo_path.exists() and (repo_path / ".git").exists():
            sb = self.sandbox_manager or SandboxManager()
            mirror = sb.ensure_mirror(repo_path, goal.project)
            scratch_wt = sb.create_attempt_scratch(
                mirror,
                goal.project,
                task.task_id,
                task.attempt,
                base_head_sha,
                origin_repo=source_repo,
            )
            try:
                yield scratch_wt
            finally:
                sb.cleanup_attempt_scratch(mirror, scratch_wt)
            return

        # Fallback for synthetic non-mirror repos: temporary detached git worktree off wt_path directly
        tmp_parent = Path(tempfile.mkdtemp(prefix=f"arr_scratch_{task.attempt}_"))
        scratch_wt = tmp_parent / "wt"
        proc = _git(wt_path, "worktree", "add", "--force", "--detach", str(scratch_wt), base_head_sha)
        if proc.returncode != 0:
            shutil.rmtree(tmp_parent, ignore_errors=True)
            raise SandboxError(f"Failed creating detached attempt scratch from {wt_path}: {proc.stderr.strip()}")
        try:
            yield scratch_wt
        finally:
            _git(wt_path, "worktree", "remove", "--force", str(scratch_wt))
            _git(wt_path, "worktree", "prune")
            shutil.rmtree(tmp_parent, ignore_errors=True)

    @staticmethod
    def _resolve_review_kind_for_task(
        task: StageTask, raw_pinned: Mapping[str, Any]
    ) -> tuple[ReviewKind, ScientificRole]:
        stage_to_kind = {
            StageKind.DESIGN_REVIEW: (ReviewKind.DESIGN_REVIEW, ScientificRole.DESIGN_REVIEWER),
            StageKind.IMPL_REVIEW: (ReviewKind.IMPLEMENTATION_REVIEW, ScientificRole.IMPL_REVIEWER),
            StageKind.RESULT_REVIEW: (ReviewKind.RESULT_REVIEW, ScientificRole.RESULT_AUDITOR),
        }
        default_kind, role = stage_to_kind[task.stage]
        raw_rk = str(raw_pinned.get("review_kind") or "").strip()
        if raw_rk:
            for cand in ReviewKind:
                if cand.value == raw_rk or cand.name == raw_rk:
                    return cand, role
        return default_kind, role

    @staticmethod
    def _refresh_synced_task_epochs(
        db: StateDB | None,
        task: StageTask,
        default_human_epoch: int,
        default_plan_epoch: int,
    ) -> tuple[int, int]:
        """Refresh `(human_epoch, plan_epoch)` from SQLite `tasks.payload_json` when an unaffected sibling task's epochs were synced."""
        if db is None:
            return default_human_epoch, default_plan_epoch
        with db._lock:
            row = db._conn.execute(
                "SELECT status, attempt, payload_json FROM tasks WHERE task_id = ?",
                (task.task_id,),
            ).fetchone()
        if row is None:
            return default_human_epoch, default_plan_epoch
        if int(row["attempt"]) != int(task.attempt):
            return default_human_epoch, default_plan_epoch
        if str(row["status"] or "").upper() not in ("DISPATCHED", "RUNNING", "READY"):
            return default_human_epoch, default_plan_epoch
        t_payload = json.loads(row["payload_json"])
        eff_human = int(t_payload.get("human_epoch", default_human_epoch) or 0)
        raw_pinned = t_payload.get("pinned_route") if isinstance(t_payload.get("pinned_route"), Mapping) else {}
        if ("plan_epoch" in t_payload and t_payload.get("plan_epoch") is not None) or (
            "plan_epoch" in raw_pinned and raw_pinned.get("plan_epoch") is not None
        ):
            eff_plan = max(
                int(t_payload.get("plan_epoch") or 0),
                int(raw_pinned.get("plan_epoch") or 0),
            )
        else:
            eff_plan = default_plan_epoch
        return eff_human, eff_plan

    @staticmethod
    def _resolve_branch_scientific_lineage(
        db: StateDB | None,
        hyp: HypothesisNode,
        *,
        eff_human_epoch: int,
        eff_plan_epoch: int,
    ) -> tuple[int, int, int]:
        """Return `(branch_human_epoch, branch_plan_epoch, sci_revision)` for `hyp`'s own scientific revision.

        Separates global execution-fence epochs (`eff_human_epoch`, `eff_plan_epoch` synced across
        unaffected sibling tasks) from per-branch scientific revision identity so revising branch A
        never stamps unaffected sibling B's receipts with an unmatching `plan_epoch`!
        """
        cur_hyp = db.get_hypothesis(hyp.hypothesis_id) if db is not None else None
        active_hyp = cur_hyp if cur_hyp is not None else hyp
        hparams = dict(active_hyp.params or {})
        latest_ep = (
            db.get_latest_experiment_plan(active_hyp.hypothesis_id)
            if (db is not None and hasattr(db, "get_latest_experiment_plan"))
            else None
        )

        sci_revision = int(hparams.get("lineage_sci_revision", 0) or 0)
        ep_plan_epoch = int(getattr(latest_ep, "plan_epoch", 0) or 0) if latest_ep is not None else 0
        ep_human_epoch = int(getattr(latest_ep, "human_epoch", 0) or 0) if latest_ep is not None else 0
        ep_plan_ver = int(getattr(latest_ep, "plan_version", 1) or 1) if latest_ep is not None else 1

        has_explicit_branch_revision = (
            sci_revision > 0
            or "lineage_plan_epoch" in hparams
            or "lineage_human_epoch" in hparams
            or ep_plan_epoch > 0
            or ep_human_epoch > 0
            or ep_plan_ver > 1
        )
        if has_explicit_branch_revision:
            branch_plan_epoch = max(int(hparams.get("lineage_plan_epoch", 0) or 0), ep_plan_epoch)
            if "lineage_human_epoch" in hparams or ep_human_epoch > 0:
                branch_human_epoch = max(
                    int(hparams.get("lineage_human_epoch", 0) or 0), ep_human_epoch
                )
            else:
                branch_human_epoch = 0
        else:
            peer_has_branch_lineage = False
            if db is not None and latest_ep is None and hasattr(db, "list_hypotheses"):
                for peer_h in db.list_hypotheses(goal_id=active_hyp.goal_id):
                    p_params = peer_h.params or {}
                    if (
                        "lineage_plan_epoch" in p_params
                        or "lineage_human_epoch" in p_params
                        or "lineage_sci_revision" in p_params
                    ):
                        peer_has_branch_lineage = True
                        break
            if latest_ep is not None or peer_has_branch_lineage:
                branch_plan_epoch = 0
                branch_human_epoch = 0
            else:
                branch_plan_epoch = int(eff_plan_epoch)
                branch_human_epoch = int(eff_human_epoch)

        return int(branch_human_epoch), int(branch_plan_epoch), int(sci_revision)

    @staticmethod
    def _precheck_wt_for_promotion(
        wt_path: Path,
        *,
        expected_wt_branch: str,
        base_head_sha: str,
    ) -> str | None:
        sym_proc = _git(wt_path, "symbolic-ref", "-q", "--short", "HEAD", timeout=10.0)
        cur_branch = sym_proc.stdout.strip() if sym_proc.returncode == 0 else ""
        if not cur_branch:
            return "Refusing promotion onto detached HEAD worktree"
        if cur_branch in PROTECTED_BRANCHES:
            return f"Refusing promotion onto protected branch {cur_branch}"
        if expected_wt_branch and cur_branch != expected_wt_branch:
            return f"wt_path branch changed ({cur_branch} != {expected_wt_branch})"
        cur_wt_head = _git(wt_path, "rev-parse", "HEAD", timeout=10.0).stdout.strip()
        if cur_wt_head != base_head_sha:
            return f"wt_path HEAD moved ({cur_wt_head[:12]} != {base_head_sha[:12]})"
        cur_wt_dirty = _git(wt_path, "status", "--porcelain", timeout=10.0).stdout.strip()
        if cur_wt_dirty:
            return f"wt_path is dirty before promotion: {cur_wt_dirty.splitlines()[0]}"
        return None

    @staticmethod
    def _extract_task_lease_fence_meta(
        task: StageTask,
        raw_pinned: Mapping[str, Any],
        default_goal_version: int,
        *,
        cli_goal_version: int | None = None,
        cli_lease_owner: str | None = None,
    ) -> tuple[int, str, list[str]]:
        if cli_goal_version is not None:
            exp_gv = int(cli_goal_version)
        elif raw_pinned.get("expected_goal_version") is not None:
            exp_gv = int(raw_pinned["expected_goal_version"])
        else:
            exp_gv = int(getattr(task, "goal_version", 0) or default_goal_version or 1)
        lease_owner = str(
            cli_lease_owner
            or raw_pinned.get("lease_owner_id")
            or getattr(task, "lease_owner_id", "")
            or ""
        ).strip()
        raw_keys = raw_pinned.get("acquired_lease_keys") or getattr(task, "acquired_lease_keys", None) or []
        req_keys = [str(k).strip() for k in raw_keys if str(k).strip()] if isinstance(raw_keys, Sequence) and not isinstance(raw_keys, (str, bytes)) else []
        return exp_gv, lease_owner, req_keys

    @staticmethod
    def _verify_and_renew_leases_in_conn(
        conn: Any,
        db: StateDB,
        *,
        task: StageTask,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        lease_owner_id: str,
        acquired_lease_keys: Sequence[str],
        ttl_sec: float = 300.0,
    ) -> str | None:
        """Verify required or task-held leases are active & unexpired inside `conn`, renewing valid TTLs."""
        req_keys = [str(k).strip() for k in acquired_lease_keys if str(k).strip()]
        if hasattr(db, "verify_and_renew_task_leases") and (req_keys or lease_owner_id):
            fn = getattr(db, "verify_and_renew_task_leases")
            sig = inspect.signature(fn).parameters
            eff_owner = lease_owner_id or f"task:{task.task_id}:a{int(task.attempt)}"
            call_kwargs: dict[str, Any] = {}
            if "ttl_sec" in sig:
                call_kwargs["ttl_sec"] = float(ttl_sec)
            elif "ttl_s" in sig:
                call_kwargs["ttl_s"] = float(ttl_sec)
            res = fn(task.task_id, int(task.attempt), eff_owner, req_keys, **call_kwargs)
            if isinstance(res, tuple):
                ok_l, reason_l = bool(res[0]), str(res[1] or "LEASE_EXPIRED_OR_LOST")
            else:
                ok_l, reason_l = bool(res), "LEASE_EXPIRED_OR_LOST"
            if not ok_l:
                return f"FENCED_LEASE_EXPIRED: {reason_l}"
            return None

        has_leases_tbl = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'leases'"
        ).fetchone()
        if has_leases_tbl is None:
            return None

        now_ts = time.time()
        new_exp = now_ts + max(60.0, float(ttl_sec))
        writer_branch_key = f"branch:{goal.goal_id}:{hyp.hypothesis_id}"

        if req_keys:
            for rk in req_keys:
                row = conn.execute(
                    """
                    SELECT lease_id, holder, status, expires_ts
                    FROM leases
                    WHERE resource_key = ? AND task_id = ?
                    ORDER BY expires_ts DESC
                    LIMIT 1
                    """,
                    (rk, task.task_id),
                ).fetchone()
                if row is None:
                    return f"FENCED_LEASE_EXPIRED: missing required lease {rk}"
                if str(row["status"] or "").upper() != "ACTIVE" or float(row["expires_ts"] or 0.0) <= now_ts:
                    return f"FENCED_LEASE_EXPIRED: lease {rk} status={row['status']} expired"
                if lease_owner_id and str(row["holder"] or "") != lease_owner_id:
                    return f"FENCED_LEASE_EXPIRED: lease {rk} holder={row['holder']} != {lease_owner_id}"
                if rk.startswith("branch_ro:"):
                    conflict_w = conn.execute(
                        "SELECT 1 FROM leases WHERE resource_key = ? AND status = 'ACTIVE' AND expires_ts > ? AND task_id != ? LIMIT 1",
                        (writer_branch_key, now_ts, task.task_id),
                    ).fetchone()
                    if conflict_w is not None:
                        return f"FENCED_LEASE_EXPIRED: writer active on {writer_branch_key}"
                conn.execute(
                    "UPDATE leases SET renewed_ts = ?, expires_ts = ? WHERE lease_id = ?",
                    (now_ts, max(float(row["expires_ts"]), new_exp), row["lease_id"]),
                )
            return None

        rows = conn.execute(
            "SELECT lease_id, resource_key, holder, status, expires_ts FROM leases WHERE task_id = ?",
            (task.task_id,),
        ).fetchall()
        if not rows:
            return None

        by_rk: dict[str, list[Any]] = {}
        for r in rows:
            by_rk.setdefault(str(r["resource_key"]), []).append(r)

        for rk, rk_rows in by_rk.items():
            valid_rows = [
                r
                for r in rk_rows
                if str(r["status"] or "").upper() == "ACTIVE" and float(r["expires_ts"] or 0.0) > now_ts
            ]
            if not valid_rows:
                return f"FENCED_LEASE_EXPIRED: task lease {rk} expired or released"
            best = max(valid_rows, key=lambda r: float(r["expires_ts"]))
            if lease_owner_id and str(best["holder"] or "") != lease_owner_id:
                return f"FENCED_LEASE_EXPIRED: lease {rk} holder={best['holder']} != {lease_owner_id}"
            conn.execute(
                "UPDATE leases SET renewed_ts = ?, expires_ts = ? WHERE lease_id = ?",
                (now_ts, max(float(best["expires_ts"]), new_exp), best["lease_id"]),
            )
        return None

    @contextlib.contextmanager
    def _task_lease_heartbeat(
        self,
        db: StateDB | None,
        *,
        task: StageTask,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        lease_owner_id: str,
        acquired_lease_keys: Sequence[str],
        ttl_sec: float,
    ) -> Iterator[None]:
        """Periodically heartbeat-renew unexpired task leases while a stage executes."""
        if db is None:
            yield
            return
        stop_evt = threading.Event()
        eff_ttl = max(60.0, float(ttl_sec))
        override_iv = getattr(self, "_lease_heartbeat_interval_s", None)
        interval_s = (
            max(0.02, float(override_iv))
            if override_iv is not None
            else min(20.0, max(1.0, eff_ttl / 3.0))
        )

        def _beat() -> None:
            while not stop_evt.wait(interval_s):
                try:
                    with db._tx() as (conn, _pending):
                        err = self._verify_and_renew_leases_in_conn(
                            conn,
                            db,
                            task=task,
                            goal=goal,
                            hyp=hyp,
                            lease_owner_id=lease_owner_id,
                            acquired_lease_keys=acquired_lease_keys,
                            ttl_sec=eff_ttl,
                        )
                        if err is not None:
                            return
                except Exception:
                    return

        hb_thread = threading.Thread(
            target=_beat,
            name=f"arr-lease-hb-{task.task_id[:24]}",
            daemon=True,
        )
        hb_thread.start()
        try:
            yield
        finally:
            stop_evt.set()
            hb_thread.join(timeout=1.0)

    @staticmethod
    def _verify_worker_fence_in_conn(
        conn: Any,
        db: StateDB,
        *,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        init_task_status: str,
        expected_goal_version: int,
        expected_human_epoch: int,
        expected_plan_epoch: int,
        base_head_sha: str,
        lease_owner_id: str = "",
        acquired_lease_keys: Sequence[str] = (),
        lease_ttl_sec: float = 300.0,
        check_branch_head: bool = True,
    ) -> str | None:
        """Return `None` when active goal/version/epoch/task/lease/branch fence holds, else failure reason."""
        t_row = conn.execute(
            "SELECT status, attempt, payload_json FROM tasks WHERE task_id = ?",
            (task.task_id,),
        ).fetchone()
        req_goal_ver = int(expected_goal_version or 1)
        req_human_ep = expected_human_epoch
        req_plan_ep = expected_plan_epoch
        eff_lease_owner = lease_owner_id
        eff_lease_keys = [str(k).strip() for k in acquired_lease_keys if str(k).strip()]
        if t_row is not None:
            cur_t_status = str(t_row["status"] or "").upper()
            allowed_statuses = {"DISPATCHED", "RUNNING"}
            if init_task_status == "READY":
                allowed_statuses.add("READY")
            if cur_t_status not in allowed_statuses:
                return f"task_status={cur_t_status} not in {sorted(allowed_statuses)}"
            if int(t_row["attempt"]) != int(task.attempt):
                return f"task_attempt={t_row['attempt']} != {task.attempt}"
            t_payload = json.loads(t_row["payload_json"])
            req_human_ep = int(t_payload.get("human_epoch", expected_human_epoch) or 0)
            raw_t_pinned = (
                t_payload.get("pinned_route")
                if isinstance(t_payload.get("pinned_route"), Mapping)
                else {}
            )
            if not expected_goal_version:
                req_goal_ver = int(
                    raw_t_pinned.get("expected_goal_version")
                    or t_payload.get("goal_version")
                    or getattr(goal, "version", 1)
                    or 1
                )
            if not eff_lease_owner:
                eff_lease_owner = str(
                    raw_t_pinned.get("lease_owner_id")
                    or t_payload.get("lease_owner_id")
                    or ""
                ).strip()
            if not eff_lease_keys:
                db_keys = raw_t_pinned.get("acquired_lease_keys") or t_payload.get("acquired_lease_keys") or []
                if isinstance(db_keys, Sequence) and not isinstance(db_keys, (str, bytes)):
                    eff_lease_keys = [str(k).strip() for k in db_keys if str(k).strip()]
            if ("plan_epoch" in t_payload and t_payload.get("plan_epoch") is not None) or (
                "plan_epoch" in raw_t_pinned and raw_t_pinned.get("plan_epoch") is not None
            ):
                req_plan_ep = max(
                    int(t_payload.get("plan_epoch") or 0),
                    int(raw_t_pinned.get("plan_epoch") or 0),
                )

        g_row = conn.execute(
            "SELECT status, payload_json FROM goals WHERE goal_id = ?",
            (goal.goal_id,),
        ).fetchone()
        if g_row is not None:
            g_status = str(g_row["status"] or "").upper()
            if g_status != "ACTIVE":
                return f"goal_status={g_status} (expected ACTIVE)"
            g_payload = json.loads(g_row["payload_json"])
            cur_goal_ver = int(g_payload.get("version", 1) or 1)
            if cur_goal_ver != req_goal_ver:
                return f"goal_version mismatch ({cur_goal_ver} != {req_goal_ver})"
            cur_human_ep = int(g_payload.get("human_epoch", 0) or 0)
            if cur_human_ep != req_human_ep:
                return f"human_epoch mismatch ({cur_human_ep} != {req_human_ep})"
            cur_plan_ep = int(g_payload.get("plan_epoch", 0) or 0)
            if cur_plan_ep != req_plan_ep:
                return f"plan_epoch mismatch ({cur_plan_ep} != {req_plan_ep})"

        has_cancel_tbl = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pending_hq_cancellations'"
        ).fetchone()
        if has_cancel_tbl is not None:
            c_row = conn.execute(
                "SELECT state FROM pending_hq_cancellations WHERE entity_id = ? AND state IN ('CANCEL_PENDING', 'CANCEL_CONFIRMED') LIMIT 1",
                (task.task_id,),
            ).fetchone()
            if c_row is not None:
                return f"pending_hq_cancellation={c_row['state']}"

        lease_err = StageRunner._verify_and_renew_leases_in_conn(
            conn,
            db,
            task=task,
            goal=goal,
            hyp=hyp,
            lease_owner_id=eff_lease_owner,
            acquired_lease_keys=eff_lease_keys,
            ttl_sec=lease_ttl_sec,
        )
        if lease_err is not None:
            return lease_err

        if check_branch_head:
            fresh_hyp = db.get_hypothesis(hyp.hypothesis_id)
            if fresh_hyp is not None:
                cur_branch_head = (fresh_hyp.head_sha or fresh_hyp.base_sha or "").strip()
                if cur_branch_head and cur_branch_head != base_head_sha:
                    return f"hypothesis_head_moved ({cur_branch_head[:12]} != {base_head_sha[:12]})"

        has_prom = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'promotion_intents'"
        ).fetchone()
        if has_prom is not None:
            unresolved_row = conn.execute(
                """
                SELECT intent_id, reason
                FROM promotion_intents
                WHERE goal_id = ? AND hypothesis_id = ? AND state = 'PREPARED'
                  AND NOT (
                      task_id = ? AND attempt = ?
                      AND COALESCE(reason, '') NOT LIKE 'UNRESOLVED_FF_AMBIGUOUS%'
                  )
                ORDER BY updated_ts DESC
                LIMIT 1
                """,
                (goal.goal_id, hyp.hypothesis_id, task.task_id, int(task.attempt)),
            ).fetchone()
            if unresolved_row is not None:
                return (
                    f"FENCED_UNRESOLVED_PROMOTION: unresolved PREPARED promotion_intent "
                    f"{unresolved_row['intent_id']} on hypothesis ({unresolved_row['reason'] or 'pending'})"
                )

        return None

    @staticmethod
    def _ensure_promotion_intents_table_in_conn(conn: Any) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS promotion_intents (
                intent_id TEXT PRIMARY KEY,
                goal_id TEXT,
                hypothesis_id TEXT,
                task_id TEXT,
                attempt INTEGER,
                branch_name TEXT,
                wt_path TEXT,
                expected_base_sha TEXT,
                candidate_sha TEXT,
                goal_version INTEGER,
                human_epoch INTEGER,
                plan_epoch INTEGER,
                state TEXT,
                reason TEXT,
                created_ts REAL,
                updated_ts REAL
            )
            """
        )

    @classmethod
    def _prepare_promotion_intent_in_conn(
        cls,
        conn: Any,
        db: StateDB,
        *,
        intent_id: str,
        goal_id: str,
        hypothesis_id: str,
        task_id: str,
        attempt: int,
        branch_name: str,
        wt_path: Path,
        expected_base_sha: str,
        candidate_sha: str,
        goal_version: int,
        human_epoch: int,
        plan_epoch: int,
    ) -> None:
        if hasattr(db, "prepare_promotion_intent"):
            fn = getattr(db, "prepare_promotion_intent")
            sig = inspect.signature(fn).parameters
            all_kw = {
                "intent_id": intent_id,
                "goal_id": goal_id,
                "hypothesis_id": hypothesis_id,
                "task_id": task_id,
                "attempt": int(attempt),
                "branch_name": branch_name,
                "wt_path": str(wt_path),
                "expected_base_sha": expected_base_sha,
                "candidate_sha": candidate_sha,
                "goal_version": int(goal_version),
                "human_epoch": int(human_epoch),
                "plan_epoch": int(plan_epoch),
            }
            filtered = {k: v for k, v in all_kw.items() if k in sig}
            fn(**filtered)
            return

        cls._ensure_promotion_intents_table_in_conn(conn)
        ts = time.time()
        conn.execute(
            """
            INSERT OR REPLACE INTO promotion_intents (
                intent_id, goal_id, hypothesis_id, task_id, attempt,
                branch_name, wt_path, expected_base_sha, candidate_sha,
                goal_version, human_epoch, plan_epoch, state, reason,
                created_ts, updated_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PREPARED', '', ?, ?)
            """,
            (
                intent_id,
                goal_id,
                hypothesis_id,
                task_id,
                int(attempt),
                branch_name,
                str(wt_path),
                expected_base_sha,
                candidate_sha,
                int(goal_version),
                int(human_epoch),
                int(plan_epoch),
                ts,
                ts,
            ),
        )

    @classmethod
    def _commit_promotion_intent_in_conn(cls, conn: Any, db: StateDB, intent_id: str) -> None:
        if hasattr(db, "commit_promotion_intent"):
            fn = getattr(db, "commit_promotion_intent")
            sig = inspect.signature(fn).parameters
            if "intent_id" in sig and len(sig) == 1:
                fn(intent_id=intent_id)
            else:
                fn(intent_id)
            return
        cls._ensure_promotion_intents_table_in_conn(conn)
        conn.execute(
            "UPDATE promotion_intents SET state = 'COMMITTED', updated_ts = ? WHERE intent_id = ?",
            (time.time(), intent_id),
        )

    @classmethod
    def _abort_promotion_intent(cls, db: StateDB, intent_id: str, reason: str) -> None:
        if hasattr(db, "abort_promotion_intent"):
            fn = getattr(db, "abort_promotion_intent")
            sig = inspect.signature(fn).parameters
            if "reason" in sig:
                fn(intent_id, reason=reason)
            else:
                fn(intent_id)
            return
        with db._tx() as (conn, _pending):
            cls._ensure_promotion_intents_table_in_conn(conn)
            conn.execute(
                "UPDATE promotion_intents SET state = 'ABORTED', reason = ?, updated_ts = ? WHERE intent_id = ?",
                (str(reason)[:500], time.time(), intent_id),
            )

    @staticmethod
    def _promotion_lock_path(
        home: Path,
        db: StateDB | None,
        goal_id: str,
        branch_name: str,
    ) -> Path:
        """Return shared promotion `fcntl.flock` path (`<db_dir>/locks/promote_<safe_goal>_<safe_branch>.flock`)."""
        base_dir = Path(db.db_path).parent if (db is not None and hasattr(db, "db_path")) else home
        safe_goal = re.sub(r"[^A-Za-z0-9._-]+", "_", str(goal_id or "default"))
        safe_branch = re.sub(r"[^A-Za-z0-9._-]+", "_", str(branch_name or "default"))
        lock_dir = base_dir / "locks"
        lock_dir.mkdir(parents=True, exist_ok=True)
        return lock_dir / f"promote_{safe_goal}_{safe_branch}.flock"

    @staticmethod
    @contextlib.contextmanager
    def _acquire_promotion_flock(lock_path: Path) -> Iterator[None]:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        fh = lock_path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        finally:
            fh.close()

    def _execute_ff_merge_with_recovery(
        self,
        conn: Any,
        *,
        wt_path: Path,
        candidate_sha: str,
        expected_base_sha: str,
        expected_wt_branch: str,
        intent_id: str,
    ) -> tuple[str, str | None]:
        """Run bounded hookless `git merge --ff-only` inside `BEGIN IMMEDIATE` and reconcile any ambiguous timeout.

        Returns `(outcome, error_reason)` where `outcome` is `"PROMOTED"`, `"ABORTED"`, or `"UNRESOLVED"`:
        - `"PROMOTED"`: branch `HEAD == candidate_sha` on `expected_wt_branch` (even after timeout/non-zero rc).
        - `"ABORTED"`: `rev-parse HEAD == expected_base_sha` positively confirmed; marks `promotion_intents.state = 'ABORTED'`.
        - `"UNRESOLVED"`: ambiguous or unreadable `HEAD`; preserves durable `'PREPARED'` intent so recovery can reconcile safely.
        """
        ff_timed_out = False
        ff_rc = 0
        ff_stderr = ""
        try:
            ff = _git(
                wt_path,
                "-c",
                "core.hooksPath=/dev/null",
                "merge",
                "--ff-only",
                candidate_sha,
                timeout=15.0,
            )
            ff_rc = ff.returncode
            ff_stderr = ff.stderr.strip()
        except subprocess.TimeoutExpired as exc:
            ff_timed_out = True
            ff_stderr = f"timed out: {exc}"

        actual_head = ""
        actual_branch = ""
        head_ok = False
        try:
            head_proc = _git(wt_path, "rev-parse", "HEAD", timeout=10.0)
            head_ok = head_proc.returncode == 0
            actual_head = head_proc.stdout.strip() if head_ok else ""
            sym_proc = _git(wt_path, "symbolic-ref", "-q", "--short", "HEAD", timeout=10.0)
            actual_branch = sym_proc.stdout.strip() if sym_proc.returncode == 0 else ""
        except (subprocess.TimeoutExpired, OSError):
            head_ok = False
            actual_head = ""
            actual_branch = ""
        branch_ok = not expected_wt_branch or actual_branch == expected_wt_branch

        if head_ok and actual_head == candidate_sha and branch_ok:
            return "PROMOTED", None

        if ff_timed_out:
            detail = f"FF_PROMOTE_REJECTED: git merge --ff-only {candidate_sha[:12]} {ff_stderr}"
        else:
            detail = (
                f"FF_PROMOTE_REJECTED: git merge --ff-only {candidate_sha[:12]} failed "
                f"(rc={ff_rc}, head={actual_head[:12]}, branch={actual_branch!r}): {ff_stderr}"
            )

        if head_ok and actual_head == expected_base_sha and branch_ok:
            conn.execute(
                "UPDATE promotion_intents SET state = 'ABORTED', reason = ?, updated_ts = ? WHERE intent_id = ?",
                (detail[:500], time.time(), intent_id),
            )
            return "ABORTED", detail

        unresolved_msg = f"UNRESOLVED_FF_AMBIGUOUS: {detail}"
        conn.execute(
            "UPDATE promotion_intents SET reason = ?, updated_ts = ? WHERE intent_id = ? AND state = 'PREPARED'",
            (unresolved_msg[:500], time.time(), intent_id),
        )
        return "UNRESOLVED", unresolved_msg

    def _validate_and_promote_scratch(
        self,
        *,
        scratch_dir: Path,
        wt_path: Path,
        base_head_sha: str,
        expected_wt_branch: str,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        db: StateDB | None,
        persist: bool,
        init_task_status: str,
        expected_goal_version: int,
        expected_human_epoch: int,
        expected_plan_epoch: int,
        lease_owner_id: str = "",
        acquired_lease_keys: Sequence[str] = (),
        lease_ttl_sec: float = 300.0,
        home: Path | None = None,
        prom_lock_stack: contextlib.ExitStack | None = None,
    ) -> tuple[str, bool, str | None, bool, dict[str, Any] | None]:
        """Validate diff in `scratch_dir` and promote `wt_path` under shared `promote_*.flock`.

        Returns `(new_head_sha, committed, error_reason, was_fenced, pending_promotion_commit)`.
        Never touches `wt_path` if validation, ancestry, or fencing fails, and NEVER calls `git reset --hard`!
        """
        # Remove transient Python cache dirs from scratch before staging
        for pyc in scratch_dir.rglob("__pycache__"):
            if ".git" not in pyc.parts:
                shutil.rmtree(pyc, ignore_errors=True)
        for ptc in scratch_dir.rglob(".pytest_cache"):
            if ".git" not in ptc.parts:
                shutil.rmtree(ptc, ignore_errors=True)

        _git(scratch_dir, "add", "-A")
        names_proc = _git(scratch_dir, "diff", "--cached", "--no-renames", "--name-only", base_head_sha)
        if names_proc.returncode != 0:
            return base_head_sha, False, f"scratch git diff failed: {names_proc.stderr.strip()}", False, None

        changed_files = [ln.strip() for ln in names_proc.stdout.splitlines() if ln.strip()]
        if not changed_files:
            return base_head_sha, False, None, False, None

        diff_proc = _git(scratch_dir, "diff", "--cached", "--no-renames", base_head_sha)
        diff_text = diff_proc.stdout if diff_proc.returncode == 0 else ""

        gate_failures: list[str] = []
        eff_writable = list(hyp.writable_paths or goal.writable_prefixes or [])
        if eff_writable:
            wp_res = gates.validate_writable_paths(
                changed_files,
                eff_writable,
                goal_prefixes=goal.writable_prefixes,
            )
            if not wp_res.passed:
                gate_failures.extend(wp_res.failures)

        ban_res = gates.check_no_banned_access(
            paths=changed_files,
            diff_text=diff_text,
            branch_name=hyp.branch_name,
            extra_bans=goal.banned_paths,
            repo_path=scratch_dir,
        )
        if not ban_res.passed:
            gate_failures.extend(ban_res.failures)

        id_res = gates.check_no_internal_ids(gates._added_diff_lines(diff_text))
        if not id_res.passed:
            gate_failures.extend(id_res.failures)

        if gate_failures:
            return base_head_sha, False, f"SCRATCH_GATE_REJECTED: {'; '.join(gate_failures)}", False, None

        # Commit on detached scratch if uncommitted index changes exist
        status_scratch = _git(scratch_dir, "status", "--porcelain")
        if status_scratch.returncode == 0 and status_scratch.stdout.strip():
            msg = f"arr: {hyp.hypothesis_id} {hyp.title[:60]}"
            cproc = _git(
                scratch_dir,
                "-c",
                "user.name=conjectra",
                "-c",
                "user.email=conjectra@local",
                "commit",
                "-m",
                msg,
            )
            if cproc.returncode != 0:
                return base_head_sha, False, f"scratch git commit failed: {cproc.stderr.strip()}", False, None

        candidate_sha = _git(scratch_dir, "rev-parse", "HEAD").stdout.strip()
        if not candidate_sha or candidate_sha == base_head_sha:
            return base_head_sha, False, None, False, None

        # Ensure candidate_sha git object is visible in wt_path (instant no-op when sharing SandboxManager mirror)
        if _git(wt_path, "cat-file", "-e", f"{candidate_sha}^{{commit}}").returncode != 0:
            _git(wt_path, "fetch", "--no-tags", str(scratch_dir), candidate_sha)

        # Strict proper-descendant check (`candidate_sha != base_head_sha` AND `base_head_sha` is ancestor of `candidate_sha`)
        anc_proc = _git(wt_path, "merge-base", "--is-ancestor", base_head_sha, candidate_sha)
        if anc_proc.returncode != 0:
            return (
                base_head_sha,
                False,
                f"FF_PROMOTE_REJECTED: candidate {candidate_sha[:12]} is not a fast-forward descendant of {base_head_sha[:12]}",
                False,
                None,
            )

        def _precheck_wt_state() -> str | None:
            return self._precheck_wt_for_promotion(
                wt_path,
                expected_wt_branch=expected_wt_branch,
                base_head_sha=base_head_sha,
            )

        promote_ref = f"refs/arr-promote/{_sanitize_rel(task.task_id).replace('/', '_')}-a{int(task.attempt)}"
        ref_proc = _git(wt_path, "update-ref", promote_ref, candidate_sha, timeout=10.0)
        if ref_proc.returncode != 0:
            return (
                base_head_sha,
                False,
                f"FF_PROMOTE_REJECTED: failed writing pin ref {promote_ref}: {ref_proc.stderr.strip()}",
                False,
                None,
            )

        intent_id = f"prom-{task.task_id}-a{int(task.attempt)}-{uuid.uuid4().hex[:8]}"
        eff_branch_for_intent = expected_wt_branch or hyp.branch_name
        eff_home = home or resolve_arr_home(self.arr_home)
        lock_path = self._promotion_lock_path(eff_home, db, goal.goal_id, eff_branch_for_intent)
        local_lock_stack = prom_lock_stack if prom_lock_stack is not None else contextlib.ExitStack()
        owns_local_stack = prom_lock_stack is None

        try:
            # Lock hierarchy: ALWAYS acquire `promote_<goal>_<branch>.flock` OUTSIDE `db._tx()`!
            local_lock_stack.enter_context(self._acquire_promotion_flock(lock_path))

            # Phase 1 (committed SQLite tx inside `promote_*.flock`): verify fences + wt_path + record durable PREPARED
            if db is not None:
                eff_h_ep, eff_p_ep = self._refresh_synced_task_epochs(
                    db, task, expected_human_epoch, expected_plan_epoch
                )
                with db._tx() as (conn, _pending):
                    fence_err = self._verify_worker_fence_in_conn(
                        conn,
                        db,
                        task=task,
                        hyp=hyp,
                        goal=goal,
                        init_task_status=init_task_status,
                        expected_goal_version=expected_goal_version,
                        expected_human_epoch=expected_human_epoch,
                        expected_plan_epoch=expected_plan_epoch,
                        base_head_sha=base_head_sha,
                        lease_owner_id=lease_owner_id,
                        acquired_lease_keys=acquired_lease_keys,
                        lease_ttl_sec=max(300.0, float(lease_ttl_sec)),
                        check_branch_head=True,
                    )
                    if fence_err is not None:
                        _git(wt_path, "update-ref", "-d", promote_ref, timeout=10.0)
                        local_lock_stack.close()
                        return base_head_sha, False, f"FENCED_PRE_PROMOTE: {fence_err}", True, None

                    wt_pre_err = _precheck_wt_state()
                    if wt_pre_err is not None:
                        _git(wt_path, "update-ref", "-d", promote_ref, timeout=10.0)
                        local_lock_stack.close()
                        return base_head_sha, False, f"FENCED_PRE_PROMOTE: {wt_pre_err}", True, None

                    self._prepare_promotion_intent_in_conn(
                        conn,
                        db,
                        intent_id=intent_id,
                        goal_id=goal.goal_id,
                        hypothesis_id=hyp.hypothesis_id,
                        task_id=task.task_id,
                        attempt=int(task.attempt),
                        branch_name=eff_branch_for_intent,
                        wt_path=wt_path,
                        expected_base_sha=base_head_sha,
                        candidate_sha=candidate_sha,
                        goal_version=expected_goal_version,
                        human_epoch=eff_h_ep,
                        plan_epoch=eff_p_ep,
                    )
            else:
                wt_pre_err = _precheck_wt_state()
                if wt_pre_err is not None:
                    _git(wt_path, "update-ref", "-d", promote_ref, timeout=10.0)
                    local_lock_stack.close()
                    return base_head_sha, False, f"FENCED_PRE_PROMOTE: {wt_pre_err}", True, None

            after_prep_hook = getattr(self, "_after_promotion_prepared_hook", None)
            if callable(after_prep_hook):
                after_prep_hook(
                    intent_id=intent_id,
                    task=task,
                    hyp=hyp,
                    goal=goal,
                    wt_path=wt_path,
                    candidate_sha=candidate_sha,
                    expected_base_sha=base_head_sha,
                )

            # When `persist=True` with an outer `prom_lock_stack`, defer BOTH `git merge --ff-only` (Phase 2)
            # AND `[hypotheses.head_sha + COMMITTED + record_receipt]` (Phase 3) into ONE UNBROKEN `db._tx()`
            # (`BEGIN IMMEDIATE`) while keeping `promote_*.flock` held!
            if db is not None and persist and not owns_local_stack:
                return (
                    candidate_sha,
                    True,
                    None,
                    False,
                    {
                        "intent_id": intent_id,
                        "candidate_sha": candidate_sha,
                        "expected_base_sha": base_head_sha,
                        "expected_wt_branch": expected_wt_branch,
                        "promote_ref": promote_ref,
                        "wt_path": wt_path,
                    },
                )

            # Standalone / `persist=False` path: execute [fresh check -> FF -> COMMITTED] inside one unbroken `db._tx()`
            if db is not None:
                with db._tx() as (conn, _pending):
                    self._ensure_promotion_intents_table_in_conn(conn)
                    i_row = conn.execute(
                        "SELECT state, expected_base_sha, candidate_sha, reason FROM promotion_intents WHERE intent_id = ?",
                        (intent_id,),
                    ).fetchone()
                    fenced_in_tx: str | None = None
                    if i_row is None:
                        fenced_in_tx = f"promotion_intent {intent_id} missing before FF"
                    elif str(i_row["state"] or "") != "PREPARED":
                        fenced_in_tx = f"promotion_intent {intent_id} state={i_row['state']} (reason={i_row['reason'] or ''})"
                    elif (
                        str(i_row["candidate_sha"] or "") != candidate_sha
                        or str(i_row["expected_base_sha"] or "") != base_head_sha
                    ):
                        fenced_in_tx = f"promotion_intent {intent_id} sha mismatch"
                    else:
                        fenced_in_tx = self._verify_worker_fence_in_conn(
                            conn,
                            db,
                            task=task,
                            hyp=hyp,
                            goal=goal,
                            init_task_status=init_task_status,
                            expected_goal_version=expected_goal_version,
                            expected_human_epoch=expected_human_epoch,
                            expected_plan_epoch=expected_plan_epoch,
                            base_head_sha=base_head_sha,
                            lease_owner_id=lease_owner_id,
                            acquired_lease_keys=acquired_lease_keys,
                            lease_ttl_sec=max(300.0, float(lease_ttl_sec)),
                            check_branch_head=True,
                        )
                    if fenced_in_tx is None:
                        fenced_in_tx = _precheck_wt_state()
                    if fenced_in_tx is not None:
                        if i_row is not None and str(i_row["state"] or "") == "PREPARED":
                            conn.execute(
                                "UPDATE promotion_intents SET state = 'ABORTED', reason = ?, updated_ts = ? WHERE intent_id = ?",
                                (f"FENCED_PRE_PROMOTE: {fenced_in_tx}"[:500], time.time(), intent_id),
                            )
                        _git(wt_path, "update-ref", "-d", promote_ref, timeout=10.0)
                        local_lock_stack.close()
                        return base_head_sha, False, f"FENCED_PRE_PROMOTE: {fenced_in_tx}", True, None

                    ff_outcome, ff_reason = self._execute_ff_merge_with_recovery(
                        conn,
                        wt_path=wt_path,
                        candidate_sha=candidate_sha,
                        expected_base_sha=base_head_sha,
                        expected_wt_branch=expected_wt_branch,
                        intent_id=intent_id,
                    )
                    if ff_outcome != "PROMOTED":
                        if ff_outcome == "ABORTED":
                            _git(wt_path, "update-ref", "-d", promote_ref, timeout=10.0)
                        local_lock_stack.close()
                        return base_head_sha, False, ff_reason, False, None

                    after_ff_hook = getattr(self, "_after_promotion_ff_hook", None)
                    if callable(after_ff_hook):
                        after_ff_hook(
                            intent_id=intent_id,
                            task=task,
                            hyp=hyp,
                            goal=goal,
                            wt_path=wt_path,
                            candidate_sha=candidate_sha,
                            expected_base_sha=base_head_sha,
                        )
                    self._commit_promotion_intent_in_conn(conn, db, intent_id)
            else:
                wt_pre_err2 = _precheck_wt_state()
                if wt_pre_err2 is not None:
                    _git(wt_path, "update-ref", "-d", promote_ref, timeout=10.0)
                    local_lock_stack.close()
                    return base_head_sha, False, f"FENCED_PRE_PROMOTE: {wt_pre_err2}", True, None
                ff = _git(wt_path, "merge", "--ff-only", candidate_sha, timeout=15.0)
                after_head = _git(wt_path, "rev-parse", "HEAD", timeout=10.0).stdout.strip()
                after_sym = _git(wt_path, "symbolic-ref", "-q", "--short", "HEAD", timeout=10.0)
                after_branch = after_sym.stdout.strip() if after_sym.returncode == 0 else ""
                if (
                    ff.returncode != 0
                    or after_head != candidate_sha
                    or (expected_wt_branch and after_branch != expected_wt_branch)
                ):
                    ff_reason = (
                        f"FF_PROMOTE_REJECTED: git merge --ff-only {candidate_sha[:12]} failed "
                        f"(rc={ff.returncode}, head={after_head[:12]}, branch={after_branch!r}): {ff.stderr.strip()}"
                    )
                    _git(wt_path, "update-ref", "-d", promote_ref, timeout=10.0)
                    local_lock_stack.close()
                    return base_head_sha, False, ff_reason, False, None

            _git(wt_path, "update-ref", "-d", promote_ref, timeout=10.0)
            local_lock_stack.close()
            return candidate_sha, True, None, False, None
        except Exception:
            local_lock_stack.close()
            raise

    def _resolve_pinned_or_route(
        self,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        *,
        role: ScientificRole | str,
        avoid_families: Sequence[str] = (),
        fallback_explicit: str | None = None,
        db: StateDB | None = None,
        wf_config: Mapping[str, Any] | None = None,
    ) -> tuple[str, str, str, ResolvedModelRoute | None]:
        """Return `(pool, model, wire_effort, route_or_none)` strictly honoring effective scoped policy."""
        raw_pinned = task.pinned_route or task.resources.pinned_route
        wf_id = getattr(task, "workflow_id", None)
        router = self.model_router or ModelRouter(arr_home=self.arr_home, db=db)
        has_eff_policy = router.has_effective_policy(
            goal=goal,
            hyp=hyp,
            workflow_id=wf_id,
            task=task,
            wf_config=wf_config,
            db=db,
        )
        host = default_execution_host()
        if isinstance(raw_pinned, Mapping) and raw_pinned.get("pool") and raw_pinned.get("canonical_model"):
            route = ResolvedModelRoute.from_dict(raw_pinned)
            if has_eff_policy:
                checked_route = router.revalidate_pinned_route(
                    role,
                    route,
                    goal=goal,
                    hyp=hyp,
                    workflow_id=wf_id,
                    task=task,
                    wf_config=wf_config,
                    db=db,
                    avoid_families=avoid_families,
                    execution_host=host,
                )
                return (
                    checked_route.pool,
                    checked_route.canonical_model,
                    checked_route.wire_effort,
                    checked_route,
                )
            return route.pool, route.canonical_model, route.wire_effort, route

        if has_eff_policy and router.has_matching_hard_rule(
            role,
            goal=goal,
            hyp=hyp,
            workflow_id=wf_id,
            task=task,
            wf_config=wf_config,
            db=db,
        ):
            explicit = None
        else:
            raw_model_spec = (task.model_spec or "").strip()
            raw_pool_spec = (task.resources.llm_pool or "").strip()
            generic_pool_tokens = {
                "",
                "auto",
                "llm_claude",
                "llmclaude",
                "llm_gemini",
                "llmgemini",
                "llm_codex_audit",
                "llmcodexaudit",
                "llm_deepseek",
                "llmdeepseek",
            }
            if (
                has_eff_policy
                and not raw_model_spec
                and raw_pool_spec.lower() in generic_pool_tokens
                and router.has_matching_rule(
                    role,
                    goal=goal,
                    hyp=hyp,
                    workflow_id=wf_id,
                    task=task,
                    wf_config=wf_config,
                    db=db,
                )
            ):
                explicit = (fallback_explicit or "").strip() or None
            else:
                explicit = (raw_model_spec or raw_pool_spec or fallback_explicit or "").strip() or None
            if explicit and explicit.lower() == "auto":
                explicit = None

        try:
            route = router.resolve_route(
                role,
                goal=goal,
                hyp=hyp,
                workflow_id=wf_id,
                task=task,
                wf_config=wf_config,
                db=db,
                explicit_spec=explicit,
                avoid_families=avoid_families,
                execution_host=host,
                strength=ConstraintStrength.SOFT_FALLBACK if explicit is None else None,
            )
            return route.pool, route.canonical_model, route.wire_effort, route
        except PolicyBlockedError:
            raise
        except Exception as exc:
            if hasattr(exc, "waitable"):
                raise PolicyBlockedError(
                    str(exc),
                    waitable=bool(getattr(exc, "waitable", False)),
                    rule_id=str(getattr(exc, "rule_id", "")),
                    code=str(getattr(exc, "code", "BLOCKED_POLICY")),
                ) from exc
            if has_eff_policy:
                raise
            for_aud = str(role) in (
                ScientificRole.DESIGN_REVIEWER.value,
                ScientificRole.IMPL_REVIEWER.value,
                ScientificRole.RESULT_AUDITOR.value,
            )
            p_fb, m_fb = _resolve_pool_and_model(explicit, for_audit=for_aud)
            eff_fb = "medium" if "sonnet" in m_fb.lower() else "high"
            return p_fb, m_fb, eff_fb, None

    def run_stage(
        self,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        *,
        db: StateDB | None = None,
        persist: bool = True,
        notify: bool = True,
        dry_run_llm: bool = False,
        expected_goal_version: int | None = None,
        lease_owner_id: str | None = None,
    ) -> EvidenceReceipt:
        """Execute a single stage task in isolation and return its structured EvidenceReceipt."""
        home = resolve_arr_home(self.arr_home)
        own_db = bool(persist and db is None)
        active_db: StateDB | None = db if db is not None else (StateDB(home) if persist else None)
        hb_stack = contextlib.ExitStack()
        prom_lock_stack = contextlib.ExitStack()
        try:
            return self._run_stage_impl(
                task,
                hyp,
                goal,
                home=home,
                db=active_db,
                persist=persist,
                notify=notify,
                dry_run_llm=dry_run_llm,
                expected_goal_version=expected_goal_version,
                lease_owner_id=lease_owner_id,
                hb_stack=hb_stack,
                prom_lock_stack=prom_lock_stack,
            )
        finally:
            prom_lock_stack.close()
            hb_stack.close()
            if own_db and active_db is not None:
                active_db.close()

    def _run_stage_impl(
        self,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        *,
        home: Path,
        db: StateDB | None,
        persist: bool,
        notify: bool,
        dry_run_llm: bool,
        expected_goal_version: int | None,
        lease_owner_id: str | None,
        hb_stack: contextlib.ExitStack,
        prom_lock_stack: contextlib.ExitStack,
    ) -> EvidenceReceipt:
        art_dir = home / "artifacts" / goal.goal_id / hyp.hypothesis_id / f"{task.task_id}_a{task.attempt}"
        art_dir.mkdir(parents=True, exist_ok=True)
        log_path = art_dir / "stage.log"

        is_review_only_stage = task.stage in (
            StageKind.DESIGN_REVIEW,
            StageKind.IMPL_REVIEW,
            StageKind.RESULT_REVIEW,
        )
        raw_pinned_init = self._raw_pinned_dict(task)
        if is_review_only_stage and not task.worktree_path and Path(goal.repo_path).joinpath(".git").exists():
            wt_path = Path(goal.repo_path)
            head_sha = str(raw_pinned_init.get("head_sha") or hyp.head_sha or hyp.base_sha)
        else:
            wt_path, head_sha = self._resolve_worktree(task, hyp, goal)

        adapter = get_adapter(goal.adapter_name or goal.project)
        multipool = self.multipool_script or DEFAULT_MULTIPOOL_SCRIPT
        slot_lock = self.slot_lock_script or DEFAULT_SLOT_LOCK_SCRIPT

        receipt_id = f"rcpt-{task.task_id}-a{task.attempt}-{uuid.uuid4().hex[:8]}"
        in_tokens = 0
        out_tokens = 0
        model_used = "deterministic"
        effort_used = ""
        review_kind_str = ""
        exit_code = 0
        wall_sec = 0.0
        max_rss_mb = 0.0
        process_status = ProcessStatus.EXIT_OK
        eng_verdict = EngineeringVerdict.PASS
        sci_verdict = ScientificVerdict.NOT_APPLICABLE
        audit_verdict = AuditVerdict.NOT_REQUIRED
        evidence_validity = EvidenceValidity.VALID_ENGINEERING_ONLY
        evidence_origin = str((hyp.params or {}).get("evidence_origin") or "real_research_goal")
        default_gv = int(getattr(task, "goal_version", 0) or getattr(goal, "version", 1))
        goal_version, eff_lease_owner, eff_lease_keys = self._extract_task_lease_fence_meta(
            task,
            raw_pinned_init,
            default_gv,
            cli_goal_version=expected_goal_version,
            cli_lease_owner=lease_owner_id,
        )
        lease_ttl_s = float(max(60, int(task.resources.wall_timeout_s or 60) + 60))
        human_epoch = int(getattr(task, "human_epoch", 0) or getattr(goal, "human_epoch", 0))
        plan_epoch = int(
            getattr(task, "plan_epoch", 0)
            or raw_pinned_init.get("plan_epoch", 0)
            or getattr(goal, "plan_epoch", 0)
            or 0
        )
        init_task_status = str(getattr(task.status, "value", task.status)).upper()
        cached_in_tokens_total = 0
        workflow_id = getattr(task, "workflow_id", None)
        tests_passed = 0
        tests_total = 0
        metrics: dict[str, Any] = {}
        artifacts: list[str] = [str(log_path)]
        failure_reason = ""
        cmd_str = ""
        diag_tail = ""
        pending_review_receipt: ReviewReceipt | None = None
        pending_promotion_commit: dict[str, Any] | None = None
        policy_blocked_domain: FailureDomain | None = None

        wf_node = None
        if workflow_id and db is not None:
            wf_node = db.get_workflow(workflow_id)
        wf_cfg = wf_node.config if wf_node is not None else None

        policy_blocked = False
        pre_resolved: tuple[str, str, str, ResolvedModelRoute | None] | None = None
        pre_resolved_boost_adv: ResolvedModelRoute | None = None
        try:
            _router_pre = self.model_router or ModelRouter(arr_home=self.arr_home, db=db)
            has_eff_policy = _router_pre.has_effective_policy(
                goal=goal,
                hyp=hyp,
                workflow_id=workflow_id,
                task=task,
                wf_config=wf_cfg,
                db=db,
            )
            if task.stage == StageKind.PROPOSE:
                pre_resolved = self._resolve_pinned_or_route(
                    task,
                    hyp,
                    goal,
                    role=ScientificRole.RESEARCH_LEAD,
                    db=db,
                    wf_config=wf_cfg,
                )
            elif task.stage == StageKind.CODE:
                collab_mode_pre = str(
                    hyp.params.get("collab_mode") or hyp.params.get("routed_mode") or "direct_hq"
                )
                legacy_b_fallback = (
                    str(hyp.params.get("builder_spec") or "").strip() or None
                    if (collab_mode_pre == "x-boost" and not has_eff_policy)
                    else None
                )
                if not (
                    not task.resources.llm_pool
                    and not task.pinned_route
                    and not task.resources.pinned_route
                    and not task.model_spec
                    and not legacy_b_fallback
                    and hyp.kind.value in ("PARAMETER_RACE", "ABLATION", "FALSIFICATION")
                ):
                    pre_resolved = self._resolve_pinned_or_route(
                        task,
                        hyp,
                        goal,
                        role=ScientificRole.ENGINEER_BUILDER,
                        fallback_explicit=legacy_b_fallback,
                        db=db,
                        wf_config=wf_cfg,
                    )
                    if collab_mode_pre == "x-boost":
                        _bp, _bm, _be, _b_route = pre_resolved
                        _b_fam = (
                            _b_route.family
                            if _b_route is not None
                            else planning.infer_model_family(f"{_bp}:{_bm}")
                        )
                        _raw_adv = (
                            str(hyp.params.get("adversary_spec") or "").strip()
                            if not has_eff_policy
                            else ""
                        )
                        if _router_pre.has_matching_hard_rule(
                            ScientificRole.IMPL_REVIEWER,
                            goal=goal,
                            hyp=hyp,
                            workflow_id=workflow_id,
                            task=task,
                            wf_config=wf_cfg,
                            db=db,
                        ):
                            _adv_exp = None
                        else:
                            _adv_exp = None if (not _raw_adv or _raw_adv.lower() == "auto") else _raw_adv
                        pre_resolved_boost_adv = _router_pre.resolve_route(
                            ScientificRole.IMPL_REVIEWER,
                            goal=goal,
                            hyp=hyp,
                            workflow_id=workflow_id,
                            task=task,
                            wf_config=wf_cfg,
                            db=db,
                            explicit_spec=_adv_exp,
                            avoid_families=[_b_fam] if _b_fam else (),
                            execution_host=default_execution_host(),
                            strength=ConstraintStrength.SOFT_FALLBACK if _adv_exp is None else None,
                        )
            elif task.stage in (StageKind.DESIGN_REVIEW, StageKind.IMPL_REVIEW, StageKind.RESULT_REVIEW):
                _rk, _rr = self._resolve_review_kind_for_task(task, raw_pinned_init)
                review_kind_str = _rk.value
                _prev = list(db.list_receipts(hypothesis_id=hyp.hypothesis_id)) if db is not None else []
                if _rk.value == "SYNTHESIS_REVIEW" and wf_node is not None and db is not None:
                    for _shid in wf_node.config.get("synthesized_hypothesis_ids") or []:
                        for _sr in db.list_receipts(hypothesis_id=str(_shid)):
                            if all(_sr.receipt_id != _er.receipt_id for _er in _prev):
                                _prev.append(_sr)
                _xp = (
                    db.get_execution_plan_by_workflow(workflow_id)
                    if (db is not None and workflow_id)
                    else None
                )
                _af = planning.collect_contributor_families(exec_plan=_xp, receipts=_prev)
                pre_resolved = self._resolve_pinned_or_route(
                    task,
                    hyp,
                    goal,
                    role=_rr,
                    avoid_families=_af,
                    db=db,
                    wf_config=wf_cfg,
                )
            elif task.stage == StageKind.BLIND_AUDIT:
                _is_boost_adv = (task.step_key or "").startswith("boost_adversary_")
                review_kind_str = (
                    ReviewKind.IMPLEMENTATION_REVIEW.value
                    if _is_boost_adv
                    else ReviewKind.RESULT_REVIEW.value
                )
                _blind_role = (
                    ScientificRole.IMPL_REVIEWER
                    if _is_boost_adv
                    else ScientificRole.RESULT_AUDITOR
                )
                _prev_b = db.list_receipts(hypothesis_id=hyp.hypothesis_id) if db is not None else []
                _am = ""
                for _r in reversed(_prev_b):
                    if _r.stage == StageKind.CODE and _r.model_used != "deterministic":
                        _am = _r.model_used
                        break
                _af_b = [planning.infer_model_family(_am)] if _am else []
                pre_resolved = self._resolve_pinned_or_route(
                    task,
                    hyp,
                    goal,
                    role=_blind_role,
                    avoid_families=_af_b,
                    db=db,
                    wf_config=wf_cfg,
                )
        except PolicyBlockedError as exc:
            policy_blocked = True
            exit_code = 78
            process_status = ProcessStatus.ENV_ERROR
            eng_verdict = EngineeringVerdict.FAIL_GATE
            sci_verdict = ScientificVerdict.UNKNOWN
            if task.stage in (
                StageKind.BLIND_AUDIT,
                StageKind.DESIGN_REVIEW,
                StageKind.IMPL_REVIEW,
                StageKind.RESULT_REVIEW,
            ):
                audit_verdict = AuditVerdict.PARSE_ERROR
            failure_reason = str(exc)
            policy_blocked_domain = (
                FailureDomain.INFRA_TRANSIENT
                if exc.waitable
                else FailureDomain.ENV_CONFIG_MISMATCH
            )
            log_path.write_text(f"POLICY_BLOCKED ({exc.code}): {exc}\n", encoding="utf-8")

        pre_exec_fence_err: str | None = None
        if not policy_blocked and task.stage != StageKind.RECONCILE and db is not None:
            with db._tx() as (conn, _pending):
                pre_exec_fence_err = self._verify_worker_fence_in_conn(
                    conn,
                    db,
                    task=task,
                    hyp=hyp,
                    goal=goal,
                    init_task_status=init_task_status,
                    expected_goal_version=goal_version,
                    expected_human_epoch=human_epoch,
                    expected_plan_epoch=plan_epoch,
                    base_head_sha=head_sha,
                    lease_owner_id=eff_lease_owner,
                    acquired_lease_keys=eff_lease_keys,
                    lease_ttl_sec=lease_ttl_s,
                    check_branch_head=(task.stage == StageKind.CODE),
                )
            if pre_exec_fence_err is not None:
                cmd_str = "fenced_pre_exec"
                exit_code = 75
                process_status = ProcessStatus.LEASE_LOST
                eng_verdict = EngineeringVerdict.SKIPPED
                sci_verdict = ScientificVerdict.UNKNOWN
                if task.stage in (
                    StageKind.BLIND_AUDIT,
                    StageKind.DESIGN_REVIEW,
                    StageKind.IMPL_REVIEW,
                    StageKind.RESULT_REVIEW,
                ):
                    audit_verdict = AuditVerdict.PARSE_ERROR
                pending_review_receipt = None
                evidence_validity = (
                    EvidenceValidity.STALE_GOAL_VERSION
                    if "goal_version mismatch" in pre_exec_fence_err
                    else EvidenceValidity.HUMAN_INVALIDATED
                )
                failure_reason = f"FENCED_PRE_EXEC: {pre_exec_fence_err}"
                log_path.write_text(f"{failure_reason}\n", encoding="utf-8")

        if not policy_blocked and pre_exec_fence_err is None and task.stage != StageKind.RECONCILE:
            hb_stack.enter_context(
                self._task_lease_heartbeat(
                    db,
                    task=task,
                    goal=goal,
                    hyp=hyp,
                    lease_owner_id=eff_lease_owner,
                    acquired_lease_keys=eff_lease_keys,
                    ttl_sec=lease_ttl_s,
                )
            )

        if policy_blocked or pre_exec_fence_err is not None:
            if policy_blocked:
                cmd_str = "policy_blocked"
        elif task.stage == StageKind.PROPOSE:
            assert pre_resolved is not None
            pool, model, effort_used, route_obj = pre_resolved
            prompt = adapter.build_code_prompt(hyp, goal, wt_path)
            if wf_node is not None:
                anchor = str(wf_node.config.get("milestone_anchor") or wf_node.state.get("milestone_anchor") or "")
                prompt = (
                    f"READ-ONLY WORKFLOW STEP ({wf_node.primitive} / {task.step_key or 'propose'}):\n"
                    + (f"Milestone anchor:\n{anchor[:4096]}\n\n" if anchor else "")
                    + prompt
                )
            if dry_run_llm:
                cmd_str = f"dry_run_propose:{pool}:{model}:{effort_used}"
                model_used = model
                log_path.write_text(
                    f"dry_run_llm=True stage=PROPOSE step={task.step_key or 'propose'}\n",
                    encoding="utf-8",
                )
            else:
                out_md = art_dir / "propose_out.md"
                usage_out = art_dir / "propose_usage.json"
                exit_code, out, err, wall_sec, max_rss_mb, process_status, cmd_str = _run_llm_pool(
                    multipool=multipool,
                    pool=pool,
                    model=model,
                    effort=effort_used,
                    pinned_route=route_obj,
                    cwd=wt_path,
                    out_md=out_md,
                    usage_out=usage_out,
                    timeout_s=task.resources.wall_timeout_s,
                    tag=f"{task.task_id}-propose",
                    prompt=prompt,
                    readonly=True,
                )
                diag_tail = f"{out[-2000:]}\n{err[-2000:]}"
                log_path.write_text(f"STDOUT:\n{out}\n\nSTDERR:\n{err}\n", encoding="utf-8")
                if out_md.is_file():
                    artifacts.append(str(out_md))
                if usage_out.is_file():
                    artifacts.append(str(usage_out))
                    u_doc = json.loads(usage_out.read_text(encoding="utf-8"))
                    in_tokens, out_tokens, cache_tok, cache_write_tok, model_used = extract_token_breakdown(u_doc)
                    cached_in_tokens_total = cache_tok
                    if cache_tok > 0:
                        metrics["cache_read_tokens"] = {
                            "value": cache_tok,
                            "denominator": 1,
                            "stage_rung": task.stage.value,
                        }
                    if cache_write_tok > 0:
                        metrics["cache_write_tokens"] = {
                            "value": cache_write_tok,
                            "denominator": 1,
                            "stage_rung": task.stage.value,
                        }
                else:
                    model_used = model
                if exit_code != 0:
                    eng_verdict = EngineeringVerdict.FAIL_GATE
                    failure_reason = f"PROPOSE pool {pool}:{model} exited {exit_code}"

        elif task.stage == StageKind.CODE:
            collab_mode = str(
                hyp.params.get("collab_mode") or hyp.params.get("routed_mode") or "direct_hq"
            )
            base_head_sha = head_sha
            sym_init = _git(wt_path, "symbolic-ref", "-q", "--short", "HEAD")
            expected_wt_branch = sym_init.stdout.strip() if sym_init.returncode == 0 else ""
            if (
                not task.resources.llm_pool
                and not task.pinned_route
                and not task.resources.pinned_route
                and not task.model_spec
                and pre_resolved is None
                and hyp.kind.value in ("PARAMETER_RACE", "ABLATION", "FALSIFICATION")
            ):
                cmd_str = "deterministic_parameterized_no_llm"
                model_used = "deterministic"
                effort_used = ""
                log_path.write_text(f"Deterministic {hyp.kind.value} skip CODE LLM\n", encoding="utf-8")
            else:
                assert pre_resolved is not None
                pool, model, effort_used, route_obj = pre_resolved
                if dry_run_llm:
                    cmd_str = f"dry_run_llm:{pool}:{model}:{effort_used}"
                    model_used = model
                    log_path.write_text(f"dry_run_llm=True (routed_mode={collab_mode})\n", encoding="utf-8")
                elif collab_mode == "x-boost":
                    from conjectra.multiagent import run_boost

                    with self._writable_attempt_scratch(task, hyp, goal, wt_path, base_head_sha) as scratch_wt:
                        prompt = adapter.build_code_prompt(hyp, goal, scratch_wt)
                        b_spec = f"{pool}:{model}:{effort_used}"
                        adv_spec = (
                            pre_resolved_boost_adv.canonical_spec()
                            if pre_resolved_boost_adv is not None
                            else "auto"
                        )
                        b_sum = run_boost(
                            repo_path=scratch_wt,
                            project=goal.project,
                            task_spec=prompt,
                            acceptance_cmd=hyp.acceptance_cmd,
                            writable_paths=hyp.writable_paths,
                            banned_paths=goal.banned_paths,
                            builder_spec=b_spec,
                            adversary_spec=adv_spec,
                            max_rounds=int(hyp.params.get("boost_max_rounds", 2)),
                            timeout_s=task.resources.wall_timeout_s,
                            in_place=True,
                            use_hq=False,
                            multipool_script=multipool,
                            arr_home=home,
                            goal=goal,
                            hyp=hyp,
                            workflow_id=workflow_id,
                            wf_config=wf_cfg,
                            task=task,
                            model_router=self.model_router or ModelRouter(arr_home=home, db=db),
                            pinned_builder_route=route_obj.to_dict() if route_obj is not None else None,
                            pinned_adversary_route=(
                                pre_resolved_boost_adv.to_dict()
                                if pre_resolved_boost_adv is not None
                                else None
                            ),
                        )
                        cmd_str = f"x-boost:{b_sum['provider_pairing']}"
                        in_tokens = int(b_sum["uncached_in_tokens"])
                        out_tokens = int(b_sum["uncached_out_tokens"])
                        cache_tok = int(b_sum.get("cache_read_tokens", 0))
                        cached_in_tokens_total = cache_tok
                        model_used = b_sum["provider_pairing"]
                        log_path.write_text(json.dumps(b_sum, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                        for up in b_sum.get("raw_usage_paths", []):
                            if up not in artifacts:
                                artifacts.append(up)
                        metrics["coordination_tax_pct"] = {
                            "value": float(b_sum.get("coordination_tax_pct", 0.0)),
                            "denominator": 100,
                            "stage_rung": task.stage.value,
                        }
                        if cache_tok > 0:
                            metrics["cache_read_tokens"] = {
                                "value": cache_tok,
                                "denominator": 1,
                                "stage_rung": task.stage.value,
                            }
                        if b_sum["status"] == "PASSED":
                            head_sha, _committed, prom_err, was_fenced, pending_promotion_commit = (
                                self._validate_and_promote_scratch(
                                    scratch_dir=scratch_wt,
                                    wt_path=wt_path,
                                    base_head_sha=base_head_sha,
                                    expected_wt_branch=expected_wt_branch,
                                    task=task,
                                    hyp=hyp,
                                    goal=goal,
                                    db=db,
                                    persist=persist,
                                    init_task_status=init_task_status,
                                    expected_goal_version=goal_version,
                                    expected_human_epoch=human_epoch,
                                    expected_plan_epoch=plan_epoch,
                                    lease_owner_id=eff_lease_owner,
                                    acquired_lease_keys=eff_lease_keys,
                                    lease_ttl_sec=lease_ttl_s,
                                    home=home,
                                    prom_lock_stack=prom_lock_stack,
                                )
                            )
                            if prom_err is not None:
                                exit_code = 75 if was_fenced else 1
                                process_status = ProcessStatus.LEASE_LOST if was_fenced else ProcessStatus.EXIT_NONZERO
                                eng_verdict = EngineeringVerdict.SKIPPED if was_fenced else EngineeringVerdict.FAIL_GATE
                                sci_verdict = ScientificVerdict.UNKNOWN
                                if was_fenced:
                                    evidence_validity = (
                                        EvidenceValidity.STALE_GOAL_VERSION
                                        if "goal_version mismatch" in prom_err
                                        else EvidenceValidity.HUMAN_INVALIDATED
                                    )
                                failure_reason = prom_err
                        else:
                            exit_code = 1
                            process_status = ProcessStatus.EXIT_NONZERO
                            eng_verdict = EngineeringVerdict.FAIL_GATE
                            failure_reason = f"x-boost ended status={b_sum['status']}"
                else:
                    with self._writable_attempt_scratch(task, hyp, goal, wt_path, base_head_sha) as scratch_wt:
                        prompt = adapter.build_code_prompt(hyp, goal, scratch_wt)
                        if wf_node is not None or task.step_key:
                            step_hdr = f"WORKFLOW STEP: {task.workflow_primitive} / {task.step_key or 'code'} (goal v{goal_version})"
                            anchor = str(
                                (wf_node.config.get("milestone_anchor") if wf_node else "")
                                or (wf_node.state.get("milestone_anchor") if wf_node else "")
                                or ""
                            )
                            slice_spec = str((wf_node.config.get("spec") if wf_node else "") or "")
                            prefix_lines = [step_hdr]
                            if slice_spec:
                                prefix_lines.append(f"Slice spec: {slice_spec[:1200]}")
                            if anchor:
                                prefix_lines.append(f"Milestone anchor (<4KB):\n{anchor[:4096]}")
                            if db is not None and (
                                (task.step_key or "").startswith("boost_repair_") or task.attempt > 1
                            ):
                                prev_rcpts = db.list_receipts(hypothesis_id=hyp.hypothesis_id)
                                finding_chunks: list[str] = []
                                for prev_r in reversed(prev_rcpts):
                                    is_bad_audit = prev_r.stage in (
                                        StageKind.BLIND_AUDIT,
                                        StageKind.IMPL_REVIEW,
                                        StageKind.DESIGN_REVIEW,
                                        StageKind.RESULT_REVIEW,
                                    ) and prev_r.audit_verdict in (
                                        AuditVerdict.FAIL,
                                        AuditVerdict.PARSE_ERROR,
                                    )
                                    is_bad_gate = prev_r.stage == StageKind.CPU_GATE and prev_r.exit_code != 0
                                    if not (is_bad_audit or is_bad_gate):
                                        continue
                                    for art_p in reversed(prev_r.raw_artifact_paths):
                                        if (
                                            art_p.endswith("audit_out.md")
                                            or art_p.endswith("review_out.md")
                                            or art_p.endswith("stage.log")
                                        ):
                                            try:
                                                txt = Path(art_p).read_text(encoding="utf-8", errors="replace").strip()
                                                if txt:
                                                    finding_chunks.append(txt[-3500:])
                                                    break
                                            except OSError:
                                                continue
                                    if not finding_chunks and prev_r.failure_reason:
                                        finding_chunks.append(prev_r.failure_reason)
                                    if finding_chunks:
                                        break
                                if finding_chunks:
                                    prefix_lines.append(
                                        "PREVIOUS AUDIT / GATE FINDINGS YOU MUST FIX COMPLETELY:\n"
                                        + "\n---\n".join(finding_chunks)[:4000]
                                    )
                            prompt = "\n".join(prefix_lines) + "\n\n" + prompt
                        out_md = art_dir / "code_out.md"
                        usage_out = art_dir / "code_usage.json"
                        exit_code, out, err, wall_sec, max_rss_mb, process_status, cmd_str = _run_llm_pool(
                            multipool=multipool,
                            pool=pool,
                            model=model,
                            effort=effort_used,
                            pinned_route=route_obj,
                            cwd=scratch_wt,
                            out_md=out_md,
                            usage_out=usage_out,
                            timeout_s=task.resources.wall_timeout_s,
                            tag=task.task_id,
                            prompt=prompt,
                            readonly=False,
                        )
                        diag_tail = f"{out[-2000:]}\n{err[-2000:]}"
                        log_path.write_text(f"STDOUT:\n{out}\n\nSTDERR:\n{err}\n", encoding="utf-8")
                        if out_md.is_file():
                            artifacts.append(str(out_md))
                        if usage_out.is_file():
                            artifacts.append(str(usage_out))
                            u_doc = json.loads(usage_out.read_text(encoding="utf-8"))
                            in_tokens, out_tokens, cache_tok, cache_write_tok, model_used = extract_token_breakdown(u_doc)
                            cached_in_tokens_total = cache_tok
                            if cache_tok > 0:
                                metrics["cache_read_tokens"] = {
                                    "value": cache_tok,
                                    "denominator": 1,
                                    "stage_rung": task.stage.value,
                                }
                            if cache_write_tok > 0:
                                metrics["cache_write_tokens"] = {
                                    "value": cache_write_tok,
                                    "denominator": 1,
                                    "stage_rung": task.stage.value,
                                }
                        else:
                            model_used = model
                        if exit_code == 0 and process_status == ProcessStatus.EXIT_OK:
                            head_sha, _committed, prom_err, was_fenced, pending_promotion_commit = (
                                self._validate_and_promote_scratch(
                                    scratch_dir=scratch_wt,
                                    wt_path=wt_path,
                                    base_head_sha=base_head_sha,
                                    expected_wt_branch=expected_wt_branch,
                                    task=task,
                                    hyp=hyp,
                                    goal=goal,
                                    db=db,
                                    persist=persist,
                                    init_task_status=init_task_status,
                                    expected_goal_version=goal_version,
                                    expected_human_epoch=human_epoch,
                                    expected_plan_epoch=plan_epoch,
                                    lease_owner_id=eff_lease_owner,
                                    acquired_lease_keys=eff_lease_keys,
                                    lease_ttl_sec=lease_ttl_s,
                                    home=home,
                                    prom_lock_stack=prom_lock_stack,
                                )
                            )
                            if prom_err is not None:
                                exit_code = 75 if was_fenced else 1
                                process_status = ProcessStatus.LEASE_LOST if was_fenced else ProcessStatus.EXIT_NONZERO
                                eng_verdict = EngineeringVerdict.SKIPPED if was_fenced else EngineeringVerdict.FAIL_GATE
                                sci_verdict = ScientificVerdict.UNKNOWN
                                if was_fenced:
                                    evidence_validity = (
                                        EvidenceValidity.STALE_GOAL_VERSION
                                        if "goal_version mismatch" in prom_err
                                        else EvidenceValidity.HUMAN_INVALIDATED
                                    )
                                failure_reason = prom_err
                        else:
                            eng_verdict = EngineeringVerdict.FAIL_GATE
                            failure_reason = f"CODE pool {pool}:{model} exited {exit_code}"

        elif task.stage == StageKind.CPU_GATE:
            raw_wt = int(task.resources.wall_timeout_s or 90)
            if raw_wt == 900:
                eplan_gate = (
                    db.get_latest_experiment_plan(hyp.hypothesis_id) if db is not None else None
                )
                raw_wt = int(eplan_gate.gate_timeout_s) if eplan_gate is not None else 90
            wall_cap_s = float(min(max(1, raw_wt), 180))
            stage_t0 = time.monotonic()
            stage_deadline = stage_t0 + wall_cap_s
            cleanup_grace_s = 10.0 if wall_cap_s >= 15.0 else min(0.4, wall_cap_s * 0.25)

            raw_cmd = adapter.build_stage_command(task, hyp, goal, wt_path, art_dir)
            cmd_str = _prefix_nice(raw_cmd)
            ban_check = gates.check_no_banned_access(
                command=raw_cmd,
                branch_name=hyp.branch_name,
                extra_bans=goal.banned_paths,
            )
            if not ban_check.passed:
                exit_code = 2
                process_status = ProcessStatus.EXIT_NONZERO
                eng_verdict = EngineeringVerdict.FAIL_GATE
                failure_reason = "; ".join(ban_check.failures)
                log_path.write_text(f"GATE_BLOCKED:\n{failure_reason}\n", encoding="utf-8")
            else:
                gate_res = gates.run_cpu_gate(
                    wt_path,
                    hyp.base_sha,
                    head_sha,
                    hyp.writable_paths,
                    goal_prefixes=goal.writable_prefixes,
                    extra_bans=goal.banned_paths,
                    command=raw_cmd,
                    branch_name=hyp.branch_name,
                    require_clean=True,
                    require_non_doc=(head_sha != hyp.base_sha),
                    require_distinct_commits=(head_sha != hyp.base_sha),
                )
                rem_for_mem = stage_deadline - time.monotonic()
                mem_wait_budget = max(0.0, min(30.0, rem_for_mem - (cleanup_grace_s + 5.0)))
                mem_ok, avail_mb = wait_for_mem_available(max_wait_s=mem_wait_budget)
                if not mem_ok:
                    exit_code = 75
                    process_status = ProcessStatus.ENV_ERROR
                    eng_verdict = EngineeringVerdict.SKIPPED
                    sci_verdict = ScientificVerdict.UNKNOWN
                    evidence_validity = EvidenceValidity.MISSING_INPUT
                    failure_reason = "MEM_BELOW_10GB_TIMEOUT"
                    wall_sec = time.monotonic() - stage_t0
                    log_path.write_text(
                        f"MEM_BELOW_10GB_TIMEOUT: MemAvailable={avail_mb}MB\n",
                        encoding="utf-8",
                    )
                else:
                    rem = stage_deadline - time.monotonic()
                    if rem <= cleanup_grace_s:
                        exit_code = 124
                        process_status = ProcessStatus.TIMEOUT
                        eng_verdict = EngineeringVerdict.FAIL_TEST
                        sci_verdict = ScientificVerdict.UNKNOWN
                        wall_sec = time.monotonic() - stage_t0
                        failure_reason = (
                            f"BLOCKED_TIMEOUT: CPU_GATE remaining budget {rem:.2f}s <= "
                            f"{cleanup_grace_s:.1f}s cleanup reserve (wall_cap={wall_cap_s:.1f}s)"
                        )
                        log_path.write_text(f"{failure_reason}\n", encoding="utf-8")
                    else:
                        run_timeout_s = rem - cleanup_grace_s
                        exit_code, out, err, _sub_wall, max_rss_mb, process_status = _run_subprocess(
                            cmd_str,
                            cwd=wt_path,
                            timeout_s=run_timeout_s,
                            shell=True,
                            stage_deadline=stage_deadline,
                        )
                        wall_sec = time.monotonic() - stage_t0
                        diag_tail = f"{out[-2000:]}\n{err[-2000:]}"
                        log_path.write_text(
                            f"GATES: {gate_res.to_dict()}\n\nSTDOUT:\n{out}\n\nSTDERR:\n{err}\n",
                            encoding="utf-8",
                        )
                        tests_passed, tests_total = parse_test_counts(f"{out}\n{err}", exit_code)
                        parsed_m, _, extra_arts, _ = adapter.parse_metrics(
                            task.stage, out, err, wt_path, art_dir, exit_code, hyp, goal
                        )
                        metrics.update(parsed_m)
                        artifacts.extend(a for a in extra_arts if a not in artifacts)
                        if process_status == ProcessStatus.TIMEOUT:
                            eng_verdict = EngineeringVerdict.FAIL_TEST
                            sci_verdict = ScientificVerdict.UNKNOWN
                            failure_reason = (
                                f"BLOCKED_TIMEOUT: CPU_GATE exceeded budget ({wall_sec:.2f}s / {wall_cap_s:.1f}s)"
                            )
                        elif not gate_res.passed:
                            eng_verdict = EngineeringVerdict.FAIL_GATE
                            failure_reason = "; ".join(gate_res.failures)
                            if exit_code == 0:
                                exit_code = 1
                        elif exit_code != 0 or tests_passed <= 0:
                            eng_verdict = EngineeringVerdict.FAIL_TEST
                            failure_reason = (
                                f"CPU_GATE test failure (rc={exit_code}, passed={tests_passed}/{tests_total})"
                            )

        elif task.stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
            raw_cmd = adapter.build_stage_command(task, hyp, goal, wt_path, art_dir)
            ban_check = gates.check_no_banned_access(
                command=raw_cmd,
                branch_name=hyp.branch_name,
                extra_bans=goal.banned_paths,
            )
            if not ban_check.passed:
                exit_code = 2
                process_status = ProcessStatus.EXIT_NONZERO
                eng_verdict = EngineeringVerdict.FAIL_GATE
                sci_verdict = ScientificVerdict.UNKNOWN
                failure_reason = "; ".join(ban_check.failures)
                log_path.write_text(f"BANNED_ACCESS:\n{failure_reason}\n", encoding="utf-8")
                cmd_str = raw_cmd
            else:
                needs_lock = task.stage == StageKind.EXP_CONFIRM and (
                    task.resources.gpu > 0 or task.resources.eda_sim > 0
                )
                vram_ok = True
                vram_info: dict[str, Any] = {}
                if task.stage == StageKind.EXP_CONFIRM and task.resources.gpu > 0:
                    min_vram = int(os.environ.get("CONJECTRA_MIN_GPU_VRAM_MB", "1500"))
                    vram_ok, vram_info = check_live_gpu_vram(min_free_mb=min_vram)

                if not vram_ok:
                    cmd_str = raw_cmd
                    exit_code = 75
                    process_status = ProcessStatus.ENV_ERROR
                    eng_verdict = EngineeringVerdict.SKIPPED
                    sci_verdict = ScientificVerdict.UNKNOWN
                    evidence_validity = EvidenceValidity.MISSING_INPUT
                    failure_reason = f"Insufficient live GPU VRAM: {vram_info}"
                    log_path.write_text(json.dumps(vram_info, indent=2) + "\n", encoding="utf-8")
                else:
                    if needs_lock:
                        base_cmd = (
                            f"{shlex.quote(sys.executable or 'python3')} "
                            f"{shlex.quote(str(slot_lock))} -- bash -lc {shlex.quote(raw_cmd)}"
                        )
                    else:
                        base_cmd = raw_cmd
                    cmd_str = _prefix_nice(base_cmd)
                    mem_ok, avail_mb = wait_for_mem_available()
                    if not mem_ok:
                        exit_code = 75
                        process_status = ProcessStatus.ENV_ERROR
                        eng_verdict = EngineeringVerdict.SKIPPED
                        sci_verdict = ScientificVerdict.UNKNOWN
                        evidence_validity = EvidenceValidity.MISSING_INPUT
                        failure_reason = "MEM_BELOW_10GB_TIMEOUT"
                        log_path.write_text(
                            f"MEM_BELOW_10GB_TIMEOUT: MemAvailable={avail_mb}MB\n",
                            encoding="utf-8",
                        )
                    else:
                        exit_code, out, err, wall_sec, max_rss_mb, process_status = _run_subprocess(
                            cmd_str,
                            cwd=wt_path,
                            timeout_s=task.resources.wall_timeout_s,
                            shell=True,
                        )
                        diag_tail = f"{out[-2000:]}\n{err[-2000:]}"
                        log_path.write_text(f"STDOUT:\n{out}\n\nSTDERR:\n{err}\n", encoding="utf-8")
                        tests_passed, tests_total = parse_test_counts(f"{out}\n{err}", exit_code)
                        metrics, evidence_validity, sci_verdict, extra_arts, parsed_reason = adapter.parse_metrics_with_validity(
                            task.stage, out, err, wt_path, art_dir, exit_code, hyp, goal
                        )
                        artifacts.extend(a for a in extra_arts if a not in artifacts)
                        if process_status == ProcessStatus.TIMEOUT:
                            eng_verdict = EngineeringVerdict.FAIL_TEST
                            sci_verdict = ScientificVerdict.UNKNOWN
                            failure_reason = f"BLOCKED_TIMEOUT: {task.stage.value} timed out after {wall_sec:.2f}s"
                        elif process_status == ProcessStatus.OOM:
                            eng_verdict = EngineeringVerdict.FAIL_TEST
                            sci_verdict = ScientificVerdict.UNKNOWN
                            failure_reason = parsed_reason or f"OOM during {task.stage.value}"
                        elif exit_code != 0:
                            eng_verdict = EngineeringVerdict.FAIL_TEST
                            failure_reason = parsed_reason or f"Command exited with code {exit_code}"
                        elif sci_verdict in (
                            ScientificVerdict.REFUTED,
                            ScientificVerdict.UNKNOWN,
                            ScientificVerdict.VALID_NEGATIVE,
                        ):
                            failure_reason = parsed_reason

        elif task.stage in (StageKind.DESIGN_REVIEW, StageKind.IMPL_REVIEW, StageKind.RESULT_REVIEW):
            r_kind, r_role = self._resolve_review_kind_for_task(task, raw_pinned_init)
            review_kind_str = r_kind.value
            syn_hids = [
                str(x).strip()
                for x in ((wf_node.config.get("synthesized_hypothesis_ids") if wf_node is not None else None) or [])
                if str(x).strip()
            ]

            prev_receipts = list(db.list_receipts(hypothesis_id=hyp.hypothesis_id)) if db is not None else []
            if r_kind.value == "SYNTHESIS_REVIEW" and db is not None and syn_hids:
                for shid in syn_hids:
                    for sr in db.list_receipts(hypothesis_id=shid):
                        if all(sr.receipt_id != er.receipt_id for er in prev_receipts):
                            prev_receipts.append(sr)

            xplan = (
                db.get_execution_plan_by_workflow(workflow_id)
                if (db is not None and workflow_id)
                else None
            )
            rplan = db.get_latest_research_plan(goal.goal_id) if db is not None else None
            eplan = db.get_latest_experiment_plan(hyp.hypothesis_id) if db is not None else None
            author_families = planning.collect_contributor_families(
                exec_plan=xplan,
                receipts=prev_receipts,
            )

            assert pre_resolved is not None
            pool, model, effort_used, route_obj = pre_resolved

            with self._readonly_review_worktree(task, hyp, goal, wt_path, head_sha) as (ro_wt, ro_head):
                head_sha = ro_head
                if r_kind == ReviewKind.DESIGN_REVIEW:
                    rev_prompt = planning.build_design_review_prompt(
                        goal,
                        hyp,
                        research_plan=rplan,
                        experiment_plan=eplan,
                    )
                elif r_kind == ReviewKind.IMPLEMENTATION_REVIEW:
                    diff_text = _git(ro_wt, "diff", "--no-renames", f"{hyp.base_sha}...{ro_head}").stdout
                    gate_rcpt = next(
                        (r for r in reversed(prev_receipts) if r.stage == StageKind.CPU_GATE),
                        None,
                    )
                    gate_summary = (
                        f"rc={gate_rcpt.exit_code}, passed={gate_rcpt.tests_passed}/{gate_rcpt.tests_total}, "
                        f"verdict={gate_rcpt.engineering_verdict.value}"
                        if gate_rcpt is not None
                        else "CPU_GATE passed"
                    )
                    rev_prompt = planning.build_impl_review_prompt(
                        goal,
                        hyp,
                        diff_text=diff_text[:32000],
                        gate_summary=gate_summary,
                        experiment_plan=eplan,
                    )
                elif r_kind.value == "SYNTHESIS_REVIEW" and hasattr(planning, "build_synthesis_review_prompt"):
                    syn_hyps: list[HypothesisNode] = []
                    if db is not None and syn_hids:
                        for shid in syn_hids:
                            sh_obj = db.get_hypothesis(shid)
                            if sh_obj is not None:
                                syn_hyps.append(sh_obj)
                    if not syn_hyps:
                        syn_hyps = [hyp]
                    syn_emp_receipts = [
                        r
                        for r in prev_receipts
                        if r.stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM)
                    ]
                    thesis_claim = str(
                        (wf_node.config.get("thesis") if wf_node is not None else "") or hyp.statement
                    )
                    rev_prompt = planning.build_synthesis_review_prompt(
                        goal,
                        syn_hyps,
                        syn_emp_receipts,
                        thesis_claim,
                    )
                else:
                    emp_receipts = [
                        r
                        for r in prev_receipts
                        if r.stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM)
                    ]
                    diff_stat = _git(ro_wt, "diff", "--stat", f"{hyp.base_sha}...{ro_head}").stdout
                    rev_prompt = planning.build_result_review_prompt(
                        goal,
                        hyp,
                        receipts=emp_receipts,
                        experiment_plan=eplan,
                        diff_stat=diff_stat,
                    )

                if dry_run_llm:
                    cmd_str = f"dry_run_review:{r_kind.value}:{pool}:{model}:{effort_used}"
                    model_used = model
                    audit_verdict = AuditVerdict.PASS
                    rev_sev = "NONE"
                    rev_findings: list[str] = []
                    rev_repro = ""
                    log_path.write_text(f"{r_kind.value}: PASS (dry_run_llm)\n", encoding="utf-8")
                else:
                    out_md = art_dir / "review_out.md"
                    usage_out = art_dir / "review_usage.json"
                    exit_code, out, err, wall_sec, max_rss_mb, process_status, cmd_str = _run_llm_pool(
                        multipool=multipool,
                        pool=pool,
                        model=model,
                        effort=effort_used,
                        pinned_route=route_obj,
                        cwd=ro_wt,
                        out_md=out_md,
                        usage_out=usage_out,
                        timeout_s=task.resources.wall_timeout_s,
                        tag=f"{task.task_id}-{r_kind.value.lower()}",
                        prompt=rev_prompt,
                        readonly=True,
                    )
                    diag_tail = f"{out[-2000:]}\n{err[-2000:]}"
                    log_path.write_text(f"STDOUT:\n{out}\n\nSTDERR:\n{err}\n", encoding="utf-8")
                    if out_md.is_file():
                        artifacts.append(str(out_md))
                        rev_txt = out_md.read_text(encoding="utf-8", errors="replace")
                    else:
                        rev_txt = out
                    if usage_out.is_file():
                        artifacts.append(str(usage_out))
                        u_doc = json.loads(usage_out.read_text(encoding="utf-8"))
                        in_tokens, out_tokens, cache_tok, cache_write_tok, model_used = extract_token_breakdown(u_doc)
                        cached_in_tokens_total = cache_tok
                        if cache_tok > 0:
                            metrics["cache_read_tokens"] = {
                                "value": cache_tok,
                                "denominator": 1,
                                "stage_rung": task.stage.value,
                            }
                        if cache_write_tok > 0:
                            metrics["cache_write_tokens"] = {
                                "value": cache_write_tok,
                                "denominator": 1,
                                "stage_rung": task.stage.value,
                            }
                    else:
                        model_used = model

                    audit_verdict, rev_sev, rev_findings, rev_repro = planning.parse_review_output(
                        rev_txt,
                        review_kind=r_kind,
                        process_status=process_status,
                    )
                    if process_status != ProcessStatus.EXIT_OK or audit_verdict != AuditVerdict.PASS:
                        eng_verdict = EngineeringVerdict.FAIL_GATE
                        failure_reason = "; ".join(rev_findings) or f"{r_kind.value} verdict={audit_verdict.value}"

            slot_idx = _infer_review_slot_index(task.step_key, raw_pinned_init)
            if xplan is not None:
                eff_tier = xplan.review_tier
            elif wf_node is not None and wf_node.config.get("review_tier"):
                try:
                    eff_tier = ReviewTier(str(wf_node.config["review_tier"]))
                except ValueError:
                    eff_tier = planning.classify_review_tier(goal, hyp)
            else:
                eff_tier = planning.classify_review_tier(goal, hyp)
            if raw_pinned_init.get("target_version") is not None:
                target_ver = int(raw_pinned_init["target_version"])
            elif eplan is not None and r_kind == ReviewKind.DESIGN_REVIEW:
                target_ver = int(eplan.plan_version)
            else:
                target_ver = goal_version
            route_payload = (
                route_obj.to_dict()
                if route_obj is not None
                else {
                    "pool": pool,
                    "canonical_model": model,
                    "requested_effort": effort_used,
                    "wire_effort": effort_used,
                    "family": planning.infer_model_family(f"{pool}:{model}"),
                }
            )
            eff_human_rev, eff_plan_rev = self._refresh_synced_task_epochs(
                db, task, human_epoch, plan_epoch
            )
            br_human_rev, br_plan_rev, _ = self._resolve_branch_scientific_lineage(
                db,
                hyp,
                eff_human_epoch=eff_human_rev,
                eff_plan_epoch=eff_plan_rev,
            )
            eff_target_kind = (
                "goal_synthesis" if r_kind.value == "SYNTHESIS_REVIEW" else "hypothesis"
            )
            stamp_rev_plan = eff_plan_rev if eff_target_kind == "goal_synthesis" else br_plan_rev
            stamp_rev_human = eff_human_rev if eff_target_kind == "goal_synthesis" else br_human_rev
            rev_sig_params = inspect.signature(ReviewReceipt).parameters
            rev_kwargs: dict[str, Any] = {
                "review_id": f"rev-{task.task_id}-a{task.attempt}-{uuid.uuid4().hex[:8]}",
                "goal_id": goal.goal_id,
                "hypothesis_id": hyp.hypothesis_id,
                "workflow_id": workflow_id or "",
                "task_id": task.task_id,
                "review_kind": r_kind,
                "review_tier": eff_tier,
                "target_version": target_ver,
                "head_sha": head_sha,
                "slot_index": slot_idx,
                "author_families": author_families,
                "reviewer_route": route_payload,
                "verdict": audit_verdict,
                "process_status": process_status,
                "severity": rev_sev,
                "findings": rev_findings,
                "repro_python": rev_repro,
                "tiebreaker_used": (slot_idx == 2),
                "raw_artifact_paths": list(artifacts),
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "created_ts": time.time(),
            }
            if "plan_epoch" in rev_sig_params:
                rev_kwargs["plan_epoch"] = stamp_rev_plan
            if "human_epoch" in rev_sig_params:
                rev_kwargs["human_epoch"] = stamp_rev_human
            if "target_kind" in rev_sig_params:
                rev_kwargs["target_kind"] = eff_target_kind
            if "synthesized_hypothesis_ids" in rev_sig_params:
                rev_kwargs["synthesized_hypothesis_ids"] = list(syn_hids)
            pending_review_receipt = ReviewReceipt(**rev_kwargs)
            setattr(pending_review_receipt, "plan_epoch", stamp_rev_plan)
            setattr(pending_review_receipt, "human_epoch", stamp_rev_human)
            setattr(pending_review_receipt, "target_kind", eff_target_kind)
            setattr(pending_review_receipt, "synthesized_hypothesis_ids", list(syn_hids))

        elif task.stage == StageKind.BLIND_AUDIT:
            review_kind_str = ReviewKind.IMPLEMENTATION_REVIEW.value if (task.step_key or "").startswith("boost_adversary_") else ReviewKind.RESULT_REVIEW.value
            prev_receipts = db.list_receipts(hypothesis_id=hyp.hypothesis_id) if db is not None else []
            assert pre_resolved is not None
            pool, model, effort_used, route_obj = pre_resolved

            with self._readonly_review_worktree(task, hyp, goal, wt_path, head_sha) as (ro_wt, ro_head):
                head_sha = ro_head
                diff_text = _git(ro_wt, "diff", "--no-renames", f"{hyp.base_sha}...{ro_head}").stdout
                raw_evidence = ""
                if db is not None:
                    raw_rows = [
                        {
                            "stage": r.stage.value,
                            "exit_code": r.exit_code,
                            "eng_verdict": r.engineering_verdict.value,
                            "sci_verdict": r.scientific_verdict.value,
                            "metrics": r.metrics,
                            "artifacts": r.raw_artifact_paths,
                        }
                        for r in prev_receipts
                    ]
                    raw_evidence = json.dumps(raw_rows, indent=2)
                stage_ctx = ""
                if (task.step_key or "").startswith("boost_adversary_"):
                    stage_ctx = (
                        "STAGE CONTEXT: This is a pre-experiment code & unit-test adversarial audit "
                        "(CPU_GATE unit tests already ran; suite-wide EXP_SCREEN/EXP_CONFIRM benchmark metrics run AFTER this audit). "
                        "Audit code correctness, invariants, edge cases, and unit/regression tests against the Hypothesis statement; "
                        "do NOT fail merely because post-stage EXP_SCREEN benchmark metrics have not run yet. "
                        "You have read-only access to the worktree to inspect full files if the diff below is truncated.\n"
                    )
                audit_prompt = (
                    "INDEPENDENT READ-ONLY BLIND AUDIT.\n"
                    "Evaluate ONLY against the spec, git diff, and raw receipts/logs below.\n"
                    + stage_ctx
                    + f"Goal: {goal.title} ({goal.primary_metric}, dir={goal.direction}, baseline={goal.baseline_value})\n"
                    f"Hypothesis: {hyp.title}\n"
                    f"Statement: {hyp.statement}\n"
                    f"Acceptance command: {hyp.acceptance_cmd}\n"
                    f"Kill criteria: {json.dumps(hyp.kill_criteria)}\n"
                    f"Git diff ({hyp.base_sha[:12]}..{ro_head[:12]}):\n```diff\n{diff_text[:32000]}\n```\n"
                    f"Raw receipts:\n```json\n{raw_evidence[:8000]}\n```\n"
                    "End your response with exactly one line: `AUDIT_VERDICT: PASS` or `AUDIT_VERDICT: FAIL`.\n"
                )
                if dry_run_llm:
                    cmd_str = f"dry_run_audit:{pool}:{model}"
                    model_used = model
                    audit_verdict = AuditVerdict.PASS
                    log_path.write_text("AUDIT_VERDICT: PASS\n", encoding="utf-8")
                else:
                    out_md = art_dir / "audit_out.md"
                    usage_out = art_dir / "audit_usage.json"
                    exit_code, out, err, wall_sec, max_rss_mb, process_status, cmd_str = _run_llm_pool(
                        multipool=multipool,
                        pool=pool,
                        model=model,
                        effort=effort_used,
                        pinned_route=route_obj,
                        cwd=ro_wt,
                        out_md=out_md,
                        usage_out=usage_out,
                        timeout_s=task.resources.wall_timeout_s,
                        tag=f"{task.task_id}-audit",
                        prompt=audit_prompt,
                        readonly=True,
                    )
                    diag_tail = f"{out[-2000:]}\n{err[-2000:]}"
                    log_path.write_text(f"STDOUT:\n{out}\n\nSTDERR:\n{err}\n", encoding="utf-8")
                    if out_md.is_file():
                        artifacts.append(str(out_md))
                        audit_txt = out_md.read_text(encoding="utf-8", errors="replace")
                    else:
                        audit_txt = out
                    if usage_out.is_file():
                        artifacts.append(str(usage_out))
                        u_doc = json.loads(usage_out.read_text(encoding="utf-8"))
                        in_tokens, out_tokens, cache_tok, cache_write_tok, model_used = extract_token_breakdown(u_doc)
                        cached_in_tokens_total = cache_tok
                        if cache_tok > 0:
                            metrics["cache_read_tokens"] = {
                                "value": cache_tok,
                                "denominator": 1,
                                "stage_rung": task.stage.value,
                            }
                        if cache_write_tok > 0:
                            metrics["cache_write_tokens"] = {
                                "value": cache_write_tok,
                                "denominator": 1,
                                "stage_rung": task.stage.value,
                            }
                    else:
                        model_used = model
                    if exit_code != 0:
                        audit_verdict = AuditVerdict.PARSE_ERROR if process_status == ProcessStatus.TIMEOUT else AuditVerdict.FAIL
                        eng_verdict = EngineeringVerdict.FAIL_GATE
                        failure_reason = f"BLIND_AUDIT exited {exit_code}"
                    else:
                        audit_mode = (
                            "boost"
                            if (task.step_key or "").startswith("boost_adversary_")
                            else "audit"
                        )
                        parsed_v = gates.parse_structured_verdict(audit_txt, mode=audit_mode)
                        if parsed_v.verdict == "PASS":
                            audit_verdict = AuditVerdict.PASS
                        elif parsed_v.verdict == "PARSE_ERROR":
                            audit_verdict = AuditVerdict.PARSE_ERROR
                            failure_reason = (
                                "; ".join(parsed_v.findings)
                                or "Structured audit verdict parse error"
                            )
                        else:
                            audit_verdict = AuditVerdict.FAIL
                            failure_reason = (
                                "; ".join(parsed_v.findings)
                                or "Independent blind audit returned FAIL"
                            )

        elif task.stage == StageKind.RECONCILE:
            cmd_str = "reconcile"
            log_path.write_text(f"Reconciled {hyp.hypothesis_id} on {head_sha}\n", encoding="utf-8")

        hb_stack.close()

        if process_status == ProcessStatus.TIMEOUT:
            sci_verdict = ScientificVerdict.UNKNOWN
            if "BLOCKED_TIMEOUT" not in failure_reason:
                failure_reason = (
                    f"BLOCKED_TIMEOUT: {failure_reason}"
                    if failure_reason
                    else f"BLOCKED_TIMEOUT: {task.stage.value} timed out"
                )

        eff_human_epoch, eff_plan_epoch = self._refresh_synced_task_epochs(
            db, task, human_epoch, plan_epoch
        )
        branch_human_epoch, branch_plan_epoch, sci_revision = self._resolve_branch_scientific_lineage(
            db,
            hyp,
            eff_human_epoch=eff_human_epoch,
            eff_plan_epoch=eff_plan_epoch,
        )
        rung_val = hyp.fidelity.value if hasattr(hyp.fidelity, "value") else str(hyp.fidelity or "F1")
        metrics["exec_fence_plan_epoch"] = {
            "value": int(eff_plan_epoch),
            "denominator": 1,
            "stage_rung": rung_val,
        }
        metrics["exec_fence_human_epoch"] = {
            "value": int(eff_human_epoch),
            "denominator": 1,
            "stage_rung": rung_val,
        }
        metrics["sci_revision"] = {
            "value": int(sci_revision),
            "denominator": 1,
            "stage_rung": rung_val,
        }

        rcpt_kwargs: dict[str, Any] = {
            "receipt_id": receipt_id,
            "task_id": task.task_id,
            "goal_id": goal.goal_id,
            "hypothesis_id": hyp.hypothesis_id,
            "stage": task.stage,
            "base_sha": hyp.base_sha,
            "head_sha": head_sha,
            "command": cmd_str,
            "exit_code": exit_code,
            "wall_sec": round(wall_sec, 4),
            "max_rss_mb": round(max_rss_mb, 3),
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "model_used": model_used,
            "process_status": process_status,
            "engineering_verdict": eng_verdict,
            "scientific_verdict": sci_verdict,
            "audit_verdict": audit_verdict,
            "tests_passed": tests_passed,
            "tests_total": tests_total,
            "metrics": metrics,
            "raw_artifact_paths": artifacts,
            "failure_reason": failure_reason,
            "created_ts": time.time(),
            "evidence_validity": evidence_validity,
            "evidence_origin": evidence_origin,
            "goal_version": goal_version,
            "human_epoch": branch_human_epoch,
            "workflow_id": workflow_id,
            "effort_used": effort_used,
            "review_kind": review_kind_str,
        }
        if "plan_epoch" in inspect.signature(EvidenceReceipt).parameters:
            rcpt_kwargs["plan_epoch"] = branch_plan_epoch
        provisional_receipt = EvidenceReceipt(**rcpt_kwargs)
        setattr(provisional_receipt, "plan_epoch", branch_plan_epoch)

        if policy_blocked_domain is not None:
            computed_domain = policy_blocked_domain
        elif process_status in (ProcessStatus.TIMEOUT, ProcessStatus.OOM, ProcessStatus.LEASE_LOST):
            computed_domain = FailureDomain.INFRA_TRANSIENT
        elif task.stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM) and process_status != ProcessStatus.ENV_ERROR:
            computed_domain = adapter.last_failure_domain
            if detect_env_config_mismatch(diag_tail, "", failure_reason):
                computed_domain = FailureDomain.ENV_CONFIG_MISMATCH
        else:
            computed_domain = gates.classify_failure_domain(
                provisional_receipt,
                goal=goal,
                stderr_or_log_tail=diag_tail,
            )

        receipt = EvidenceReceipt.from_dict(
            {
                **provisional_receipt.to_dict(),
                "failure_domain": computed_domain.value,
                "human_epoch": branch_human_epoch,
                "plan_epoch": branch_plan_epoch,
            }
        )
        setattr(receipt, "plan_epoch", branch_plan_epoch)

        if persist:
            assert db is not None
            with db._tx() as (conn, _pending):
                if pending_promotion_commit is not None:
                    # Phase 2+3 Combined Atomic Authority Transaction:
                    # Holding `promote_<goal>_<branch>.flock` (OUTER) + ONE UNBROKEN `db._tx()` (`BEGIN IMMEDIATE`, INNER),
                    # re-verify all authority fences, execute bounded local `git merge --ff-only`, and commit
                    # `hypotheses.head_sha + COMMITTED + record_receipt` atomically so concurrent `REVISE_HYPOTHESIS` /
                    # `update_plan` / `pause` writers wait cleanly on SQLite's `busy_timeout` and NEVER race FF!
                    p_intent_id = str(pending_promotion_commit["intent_id"])
                    p_cand_sha = str(pending_promotion_commit["candidate_sha"])
                    p_base_sha = str(pending_promotion_commit["expected_base_sha"])
                    p_exp_branch = str(pending_promotion_commit.get("expected_wt_branch") or "")
                    p_wt_path = Path(pending_promotion_commit["wt_path"])

                    self._ensure_promotion_intents_table_in_conn(conn)
                    i_row = conn.execute(
                        "SELECT state, expected_base_sha, candidate_sha, reason FROM promotion_intents WHERE intent_id = ?",
                        (p_intent_id,),
                    ).fetchone()
                    prom_fence_err: str | None = None
                    if i_row is None:
                        prom_fence_err = f"promotion_intent {p_intent_id} missing before FF"
                    elif str(i_row["state"] or "") != "PREPARED":
                        prom_fence_err = (
                            f"promotion_intent {p_intent_id} state={i_row['state']} "
                            f"(reason={i_row['reason'] or ''})"
                        )
                    elif (
                        str(i_row["candidate_sha"] or "") != p_cand_sha
                        or str(i_row["expected_base_sha"] or "") != p_base_sha
                    ):
                        prom_fence_err = f"promotion_intent {p_intent_id} sha mismatch"
                    else:
                        prom_fence_err = self._verify_worker_fence_in_conn(
                            conn,
                            db,
                            task=task,
                            hyp=hyp,
                            goal=goal,
                            init_task_status=init_task_status,
                            expected_goal_version=goal_version,
                            expected_human_epoch=human_epoch,
                            expected_plan_epoch=plan_epoch,
                            base_head_sha=p_base_sha,
                            lease_owner_id=eff_lease_owner,
                            acquired_lease_keys=eff_lease_keys,
                            lease_ttl_sec=max(300.0, float(lease_ttl_s)),
                            check_branch_head=True,
                        )
                    if prom_fence_err is None:
                        prom_fence_err = self._precheck_wt_for_promotion(
                            p_wt_path,
                            expected_wt_branch=p_exp_branch,
                            base_head_sha=p_base_sha,
                        )

                    if prom_fence_err is not None:
                        if i_row is not None and str(i_row["state"] or "") == "PREPARED":
                            conn.execute(
                                "UPDATE promotion_intents SET state = 'ABORTED', reason = ?, updated_ts = ? WHERE intent_id = ?",
                                (f"FENCED_PRE_PROMOTE: {prom_fence_err}"[:500], time.time(), p_intent_id),
                            )
                        pending_review_receipt = None
                        exit_code = 75
                        head_sha = p_base_sha
                        process_status = ProcessStatus.LEASE_LOST
                        eng_verdict = EngineeringVerdict.SKIPPED
                        sci_verdict = ScientificVerdict.UNKNOWN
                        evidence_validity = (
                            EvidenceValidity.STALE_GOAL_VERSION
                            if "goal_version mismatch" in prom_fence_err
                            else EvidenceValidity.HUMAN_INVALIDATED
                        )
                        failure_reason = f"FENCED_PRE_PROMOTE: {prom_fence_err}"
                        receipt = EvidenceReceipt.from_dict(
                            {
                                **receipt.to_dict(),
                                "head_sha": p_base_sha,
                                "exit_code": exit_code,
                                "process_status": process_status.value,
                                "engineering_verdict": eng_verdict.value,
                                "scientific_verdict": sci_verdict.value,
                                "evidence_validity": evidence_validity.value,
                                "failure_domain": FailureDomain.INFRA_TRANSIENT.value,
                                "failure_reason": failure_reason,
                                "human_epoch": branch_human_epoch,
                                "plan_epoch": branch_plan_epoch,
                            }
                        )
                        setattr(receipt, "plan_epoch", branch_plan_epoch)
                    else:
                        in_tx_before_ff = getattr(self, "_in_promotion_authority_tx_before_ff_hook", None)
                        if callable(in_tx_before_ff):
                            in_tx_before_ff(
                                intent_id=p_intent_id,
                                task=task,
                                hyp=hyp,
                                goal=goal,
                                wt_path=p_wt_path,
                                candidate_sha=p_cand_sha,
                                expected_base_sha=p_base_sha,
                            )

                        ff_outcome, ff_reason = self._execute_ff_merge_with_recovery(
                            conn,
                            wt_path=p_wt_path,
                            candidate_sha=p_cand_sha,
                            expected_base_sha=p_base_sha,
                            expected_wt_branch=p_exp_branch,
                            intent_id=p_intent_id,
                        )
                        if ff_outcome != "PROMOTED":
                            pending_review_receipt = None
                            exit_code = 1
                            head_sha = p_base_sha
                            process_status = ProcessStatus.EXIT_NONZERO
                            eng_verdict = EngineeringVerdict.FAIL_GATE
                            sci_verdict = ScientificVerdict.UNKNOWN
                            failure_reason = str(ff_reason or "FF_PROMOTE_REJECTED")
                            if ff_outcome == "UNRESOLVED":
                                # Preserve refs/arr-promote pin + PREPARED intent so controller can reconcile
                                pending_promotion_commit = None
                            receipt = EvidenceReceipt.from_dict(
                                {
                                    **receipt.to_dict(),
                                    "head_sha": p_base_sha,
                                    "exit_code": exit_code,
                                    "process_status": process_status.value,
                                    "engineering_verdict": eng_verdict.value,
                                    "scientific_verdict": sci_verdict.value,
                                    "failure_domain": (
                                        FailureDomain.INFRA_TRANSIENT.value
                                        if ff_outcome == "UNRESOLVED"
                                        else FailureDomain.ENGINEERING_DEFECT.value
                                    ),
                                    "failure_reason": failure_reason,
                                    "human_epoch": branch_human_epoch,
                                    "plan_epoch": branch_plan_epoch,
                                }
                            )
                            setattr(receipt, "plan_epoch", branch_plan_epoch)
                        else:
                            after_ff_hook = getattr(self, "_after_promotion_ff_hook", None)
                            if callable(after_ff_hook):
                                after_ff_hook(
                                    intent_id=p_intent_id,
                                    task=task,
                                    hyp=hyp,
                                    goal=goal,
                                    wt_path=p_wt_path,
                                    candidate_sha=p_cand_sha,
                                    expected_base_sha=p_base_sha,
                                )

                            cur_hyp = db.get_hypothesis(hyp.hypothesis_id)
                            if cur_hyp is not None:
                                db.transition_hypothesis_status(
                                    hyp.hypothesis_id,
                                    cur_hyp.status,
                                    cur_hyp.status,
                                    head_sha=p_cand_sha,
                                )
                            self._commit_promotion_intent_in_conn(conn, db, p_intent_id)
                elif (
                    not policy_blocked
                    and process_status != ProcessStatus.LEASE_LOST
                    and task.stage != StageKind.RECONCILE
                ):
                    persist_fence_err = self._verify_worker_fence_in_conn(
                        conn,
                        db,
                        task=task,
                        hyp=hyp,
                        goal=goal,
                        init_task_status=init_task_status,
                        expected_goal_version=goal_version,
                        expected_human_epoch=human_epoch,
                        expected_plan_epoch=plan_epoch,
                        base_head_sha=head_sha,
                        lease_owner_id=eff_lease_owner,
                        acquired_lease_keys=eff_lease_keys,
                        lease_ttl_sec=lease_ttl_s,
                        check_branch_head=False,
                    )
                    if persist_fence_err is not None:
                        pending_review_receipt = None
                        exit_code = 75
                        process_status = ProcessStatus.LEASE_LOST
                        eng_verdict = EngineeringVerdict.SKIPPED
                        sci_verdict = ScientificVerdict.UNKNOWN
                        evidence_validity = (
                            EvidenceValidity.STALE_GOAL_VERSION
                            if "goal_version mismatch" in persist_fence_err
                            else EvidenceValidity.HUMAN_INVALIDATED
                        )
                        failure_reason = f"FENCED_PRE_PERSIST: {persist_fence_err}"
                        fenced_audit_v = (
                            AuditVerdict.PARSE_ERROR.value
                            if task.stage
                            in (
                                StageKind.BLIND_AUDIT,
                                StageKind.DESIGN_REVIEW,
                                StageKind.IMPL_REVIEW,
                                StageKind.RESULT_REVIEW,
                            )
                            else receipt.audit_verdict.value
                        )
                        receipt = EvidenceReceipt.from_dict(
                            {
                                **receipt.to_dict(),
                                "exit_code": exit_code,
                                "process_status": process_status.value,
                                "engineering_verdict": eng_verdict.value,
                                "scientific_verdict": sci_verdict.value,
                                "audit_verdict": fenced_audit_v,
                                "evidence_validity": evidence_validity.value,
                                "failure_domain": FailureDomain.INFRA_TRANSIENT.value,
                                "failure_reason": failure_reason,
                                "human_epoch": branch_human_epoch,
                                "plan_epoch": branch_plan_epoch,
                            }
                        )
                        setattr(receipt, "plan_epoch", branch_plan_epoch)

                if pending_review_receipt is not None:
                    db.save_review_receipt(pending_review_receipt)
                db.record_receipt(receipt)
                # When StateDB.record_receipt natively owns settle_goal_budget (v4+), avoid double-settlement;
                # still invoke an instance-level settle_goal_budget hook when StateDB lacks the class method.
                if not hasattr(type(db), "settle_goal_budget") and hasattr(db, "settle_goal_budget"):
                    db.settle_goal_budget(
                        task.goal_id,
                        f"task:{task.task_id}:a{task.attempt}",
                        entity_kind="stage_task",
                        entity_id=task.task_id,
                        attempt=int(task.attempt),
                        model=str(model_used or ""),
                        input_tokens=int(in_tokens),
                        output_tokens=int(out_tokens),
                        cached_input_tokens=int(cached_in_tokens_total),
                        state="SETTLED",
                    )
            if pending_promotion_commit is not None:
                _git(
                    Path(pending_promotion_commit["wt_path"]),
                    "update-ref",
                    "-d",
                    str(pending_promotion_commit["promote_ref"]),
                    timeout=10.0,
                )
            prom_lock_stack.close()
        elif db is not None and hasattr(db, "settle_goal_budget"):
            db.settle_goal_budget(
                task.goal_id,
                f"task:{task.task_id}:a{task.attempt}",
                entity_kind="stage_task",
                entity_id=task.task_id,
                attempt=int(task.attempt),
                model=str(model_used or ""),
                input_tokens=int(in_tokens),
                output_tokens=int(out_tokens),
                cached_input_tokens=int(cached_in_tokens_total),
                state="SETTLED",
            )

        if notify:
            notify_controller(
                home,
                {
                    "event": "STAGE_EXITED",
                    "task_id": task.task_id,
                    "goal_id": goal.goal_id,
                    "hypothesis_id": hyp.hypothesis_id,
                    "workflow_id": workflow_id,
                    "step_key": task.step_key,
                    "goal_version": goal_version,
                    "human_epoch": eff_human_epoch,
                    "plan_epoch": eff_plan_epoch,
                    "stage": task.stage.value,
                    "receipt_id": receipt.receipt_id,
                    "exit_code": exit_code,
                    "head_sha": head_sha,
                },
            )

        return receipt


def main(argv: Sequence[str] | None = None) -> int:
    """Standalone CLI entrypoint for HQ stage-split worker invocations."""
    ap = argparse.ArgumentParser(description="Execute one conjectra stage task and exit immediately.")
    ap.add_argument("--task-id", required=True)
    ap.add_argument("--arr-home", default=None)
    ap.add_argument("--dry-run-llm", action="store_true")
    ap.add_argument("--attempt", type=int, default=None)
    ap.add_argument("--goal-version", type=int, default=None)
    ap.add_argument("--lease-owner", default=None)
    args = ap.parse_args(argv)

    home = resolve_arr_home(args.arr_home)
    db = StateDB(home)
    try:
        task = db.get_task(args.task_id)
        if task is None:
            raise KeyError(f"Task not found: {args.task_id}")
        hyp = db.get_hypothesis(task.hypothesis_id)
        goal = db.get_goal(task.goal_id)
        if hyp is None or goal is None:
            raise KeyError(f"Missing hypothesis/goal for task {args.task_id}")
        expected_attempt = int(args.attempt) if args.attempt is not None else int(task.attempt)
        if int(task.attempt) != expected_attempt:
            return 75
        raw_pinned = StageRunner._raw_pinned_dict(task)
        expected_gv, eff_lease_owner, eff_lease_keys = StageRunner._extract_task_lease_fence_meta(
            task,
            raw_pinned,
            int(getattr(goal, "version", 1) or 1),
            cli_goal_version=args.goal_version,
            cli_lease_owner=args.lease_owner,
        )
        if int(getattr(goal, "version", 1) or 1) != expected_gv:
            return 75
        if task.status == "READY":
            db.transition_task_status(
                task.task_id, "READY", "RUNNING", started_ts=time.time()
            )
            task = db.get_task(args.task_id) or task
        runner = StageRunner(arr_home=home)
        receipt = runner.run_stage(
            task,
            hyp,
            goal,
            db=db,
            persist=True,
            notify=True,
            dry_run_llm=args.dry_run_llm,
            expected_goal_version=expected_gv,
            lease_owner_id=eff_lease_owner or None,
        )
        final_status = "DONE" if receipt.process_status == ProcessStatus.EXIT_OK else "FAILED"
        transitioned = False
        with db._tx() as (conn, _pending):
            t_row = conn.execute(
                "SELECT status, attempt, payload_json FROM tasks WHERE task_id = ?",
                (args.task_id,),
            ).fetchone()
            g_row = conn.execute(
                "SELECT status, payload_json FROM goals WHERE goal_id = ?",
                (task.goal_id,),
            ).fetchone()
            if t_row is not None and g_row is not None:
                t_status = str(t_row["status"] or "").upper()
                g_status = str(g_row["status"] or "").upper()
                t_payload = json.loads(t_row["payload_json"])
                g_payload = json.loads(g_row["payload_json"])
                raw_t_pinned = (
                    t_payload.get("pinned_route")
                    if isinstance(t_payload.get("pinned_route"), Mapping)
                    else {}
                )
                t_plan_ep = max(
                    int(t_payload.get("plan_epoch") or 0),
                    int(raw_t_pinned.get("plan_epoch") or 0),
                )
                lease_err_end = StageRunner._verify_and_renew_leases_in_conn(
                    conn,
                    db,
                    task=task,
                    goal=goal,
                    hyp=hyp,
                    lease_owner_id=eff_lease_owner,
                    acquired_lease_keys=eff_lease_keys,
                    ttl_sec=60.0,
                )
                if (
                    int(t_row["attempt"]) == expected_attempt
                    and t_status in ("RUNNING", "DISPATCHED")
                    and g_status == "ACTIVE"
                    and int(g_payload.get("version", 1) or 1) == expected_gv
                    and int(g_payload.get("human_epoch", 0) or 0) == int(t_payload.get("human_epoch", 0) or 0)
                    and int(g_payload.get("plan_epoch", 0) or 0) == t_plan_ep
                    and lease_err_end is None
                ):
                    db.transition_task_status(
                        task.task_id,
                        ("RUNNING", "DISPATCHED"),
                        final_status,
                        finished_ts=time.time(),
                    )
                    transitioned = True
        print(json.dumps(receipt.to_dict(), sort_keys=True))
        if not transitioned and task.stage != StageKind.RECONCILE:
            return 75
        return 0 if receipt.process_status == ProcessStatus.EXIT_OK else (receipt.exit_code or 1)
    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
