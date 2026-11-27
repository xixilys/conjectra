"""Cross-project resource, quota, disk, fair-share, and crash-recovery arbiter."""

from __future__ import annotations

import datetime as dt
import fcntl
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from conjectra import gates
from conjectra.types import ProcessStatus, ResourceSpec, StageKind, StageTask

from conjectra import slot_lock as _SLOT_LOCK_MODULE

_CONFIG_DIR = Path(os.environ.get("CONJECTRA_HOME") or (Path.home() / ".conjectra")).expanduser()

_MEM_AVAIL_RE = re.compile(r"^MemAvailable:\s+(\d+)\s+kB$", re.MULTILINE)

# Substrings that must never appear in a pool/model spec (e.g. untrusted relays). Extend via
# CONJECTRA_BANNED_POOLS="a,b,c".
BANNED_POOL_SUBSTRINGS: tuple[str, ...] = tuple(
    x.strip().lower() for x in os.environ.get("CONJECTRA_BANNED_POOLS", "gemini:claude").split(",") if x.strip()
)
SONNET_BLOCKED_EFFORTS = ("high", "xhigh", "max")
DEFAULT_DISK_MOUNTS: tuple[str, ...] = ("/", *(p for p in os.environ.get("CONJECTRA_DISK_MOUNTS", "").split(":") if p))
DEFAULT_POOL_FALLBACK_ORDER = (
    "llm_gemini",
    "llm_claude",
    "llm_deepseek",
    "llm_codex_audit",
)

POOL_NORMALIZATION = {
    "gemini": "llm_gemini",
    "gemini-cli": "llm_gemini",
    "llm_gemini": "llm_gemini",
    "claude": "llm_claude",
    "llm_claude": "llm_claude",
    "deepseek": "llm_deepseek",
    "llm_deepseek": "llm_deepseek",
    "grok": "llm_deepseek",
    "codex": "llm_codex_audit",
    "llm_codex_audit": "llm_codex_audit",
    "deepseek-local": "deepseek_local",
    "deepseek_local": "deepseek_local",
}


@dataclass(frozen=True)
class GpuStatus:
    """Live NVIDIA GPU telemetry snapshot."""

    index: int
    name: str
    total_mb: int
    free_mb: int
    utilization_pct: float


@dataclass(frozen=True)
class DiskStatus:
    """Filesystem usage snapshot."""

    mount: str
    total_gb: float
    free_gb: float
    used_pct: float


@dataclass(frozen=True)
class HeavySlotSnapshot:
    """Read-only snapshot of live heavy-slot locks (see `conjectra.slot_lock`)."""

    lock_dir: Path
    max_slots: int
    busy_slots: int
    holders: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CloudSlotState:
    """Read-only snapshot of external cloud-GPU slot state."""

    available_slots: int
    status: str
    active_jobs: int
    raw_path: str | None = None
    stale: bool = True


@dataclass(frozen=True)
class AdmissionDecision:
    """Admission outcome from the cross-project arbiter."""

    allowed: bool
    reason: str
    resolved_pool: str | None = None
    fallback_pool: str | None = None
    disk_statuses: list[DiskStatus] = field(default_factory=list)
    mem_available_mb: int | None = None
    load1: float | None = None


@dataclass(frozen=True)
class RetryDecision:
    """Bounded retry and lane circuit-breaker recommendation."""

    should_retry: bool
    reason: str
    next_pool: str | None = None
    lane_tripped: bool = False


@dataclass(frozen=True)
class ProcessLiveness:
    """PID + /proc/<pid>/stat starttime identity check result."""

    pid: int
    alive: bool
    state: str
    expected_start_time: int | None
    actual_start_time: int | None
    recycled: bool = False


def read_mem_available_mb() -> int | None:
    """Read live Linux MemAvailable in MiB."""
    p = Path("/proc/meminfo")
    if not p.exists():
        return None
    try:
        m = _MEM_AVAIL_RE.search(p.read_text(encoding="utf-8", errors="ignore"))
        return (int(m.group(1)) // 1024) if m else None
    except OSError:
        return None


def read_load1() -> float | None:
    """Read live 1-minute Linux load average."""
    p = Path("/proc/loadavg")
    if not p.exists():
        return None
    try:
        parts = p.read_text(encoding="utf-8", errors="ignore").strip().split()
        return float(parts[0]) if parts else None
    except (OSError, ValueError):
        return None


def read_gpu_status() -> list[GpuStatus]:
    """Query live GPU free VRAM and utilization via nvidia-smi."""
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    try:
        proc = subprocess.run(
            [
                smi,
                "--query-gpu=index,name,memory.total,memory.free,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return []
        gpus: list[GpuStatus] = []
        for raw in proc.stdout.splitlines():
            parts = [p.strip() for p in raw.split(",")]
            if len(parts) < 5:
                continue
            gpus.append(
                GpuStatus(
                    index=int(parts[0]),
                    name=parts[1],
                    total_mb=int(float(parts[2])),
                    free_mb=int(float(parts[3])),
                    utilization_pct=float(parts[4]),
                )
            )
        return gpus
    except Exception:
        return []


def check_disk_usage(mounts: Sequence[str | Path] = DEFAULT_DISK_MOUNTS) -> list[DiskStatus]:
    """Inspect disk usage percentage on configured mounts without failing on absent paths."""
    statuses: list[DiskStatus] = []
    for raw in mounts:
        p = Path(raw)
        if not p.exists():
            continue
        try:
            usage = shutil.disk_usage(p)
        except OSError:
            continue
        used = max(0, usage.total - usage.free)
        used_pct = round(100.0 * used / usage.total, 2) if usage.total > 0 else 100.0
        statuses.append(
            DiskStatus(
                mount=str(p),
                total_gb=round(usage.total / (1024**3), 2),
                free_gb=round(usage.free / (1024**3), 2),
                used_pct=used_pct,
            )
        )
    return statuses


def inspect_heavy_slots(
    lock_dir: Path | str | None = None,
    *,
    max_slots: int = 4,
) -> HeavySlotSnapshot:
    """Inspect existing `heavy_slot_*.lock` files read-only without disturbing live locks."""
    if lock_dir is None:
        resolved_dir = _SLOT_LOCK_MODULE.default_lock_dir()
    else:
        resolved_dir = Path(lock_dir)

    holders: list[dict[str, Any]] = []
    busy = 0
    if not resolved_dir.exists():
        return HeavySlotSnapshot(lock_dir=resolved_dir, max_slots=max_slots, busy_slots=0, holders=[])

    for lock_path in sorted(resolved_dir.glob("heavy_slot_*.lock")):
        is_locked = False
        fd: int | None = None
        try:
            fd = os.open(str(lock_path), os.O_RDONLY)
            try:
                fcntl.flock(fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
                fcntl.flock(fd, fcntl.LOCK_UN)
            except BlockingIOError:
                is_locked = True
        except OSError:
            continue
        finally:
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass

        meta: dict[str, Any] = {}
        try:
            raw = lock_path.read_text(encoding="utf-8", errors="ignore").strip()
            if raw:
                parsed = json.loads(raw)
                if isinstance(parsed, dict):
                    meta = parsed
        except Exception:
            meta = {}

        pid = meta.get("pid")
        pid_alive = isinstance(pid, int) and Path(f"/proc/{pid}/stat").exists()
        if is_locked or pid_alive:
            busy += 1
            holders.append(
                {
                    "lock_file": lock_path.name,
                    "locked": is_locked,
                    "pid": pid,
                    "cmd": meta.get("cmd"),
                    "acquired_at": meta.get("acquired_at"),
                }
            )
    return HeavySlotSnapshot(
        lock_dir=resolved_dir,
        max_slots=max_slots,
        busy_slots=busy,
        holders=holders,
    )


def read_cloud_slot_state(
    search_paths: Sequence[str | Path] | None = None,
    *,
    stale_after_s: float = 3600.0,
) -> CloudSlotState:
    """Read external cloud-GPU slot state JSON read-only without interfering with any runner."""
    candidates: list[Path] = []
    env_p = os.environ.get("CONJECTRA_CLOUD_SLOT_STATE_PATH")
    if env_p:
        candidates.append(Path(env_p).expanduser())
    if search_paths:
        for sp in search_paths:
            p = Path(sp).expanduser()
            if p.is_dir():
                candidates.extend(sorted(p.glob("cloud_slot_state*.json")))
                candidates.extend(sorted(p.glob("cloud_runtime*.json")))
            else:
                candidates.extend(Path(m) for m in sorted(glob.glob(str(p))))
    else:
        default_globs = (
            str(Path.home() / ".cache" / "cloud_slot_state*.json"),
            str(Path.home() / ".cache" / "cloud_runtime*.json"),
            "/tmp/cloud_slot_state*.json",
            "/tmp/cloud_runtime*.json",
        )
        for pat in default_globs:
            candidates.extend(Path(m) for m in sorted(glob.glob(pat)))

    seen_keys: set[str] = set()
    fresh_slots = 0
    fresh_active = 0
    fresh_paths: list[str] = []
    fresh_status: str | None = None
    fallback_stale: CloudSlotState | None = None

    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            resolved_str = str(cand.resolve())
        except OSError:
            resolved_str = str(cand)
        if resolved_str in seen_keys:
            continue
        try:
            doc = json.loads(cand.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue
        rt_id = str(doc.get("runtime_id") or doc.get("runtime") or "").strip()
        dedup_k = f"rt:{rt_id}" if rt_id else f"path:{resolved_str}"
        if dedup_k in seen_keys:
            continue
        seen_keys.add(resolved_str)
        seen_keys.add(dedup_k)
        try:
            age_s = time.time() - cand.stat().st_mtime
        except OSError:
            age_s = stale_after_s + 1.0
        ts = doc.get("updated_at") or doc.get("ts")
        if isinstance(ts, (int, float)):
            age_s = max(0.0, time.time() - float(ts))
        stale = age_s > stale_after_s
        status = str(doc.get("status") or ("stale" if stale else "ready")).lower()
        slots = int(
            doc.get(
                "available_slots",
                1 if (not stale and status in ("ready", "connected", "active", "idle")) else 0,
            )
        )
        active_jobs = max(0, int(doc.get("active_jobs", 0)))
        if not stale:
            fresh_slots += max(0, slots)
            fresh_active += active_jobs
            fresh_paths.append(str(cand))
            if fresh_status is None or status in ("ready", "connected", "active"):
                fresh_status = status
        elif fallback_stale is None:
            fallback_stale = CloudSlotState(
                available_slots=0,
                status=status,
                active_jobs=active_jobs,
                raw_path=str(cand),
                stale=True,
            )

    if fresh_paths:
        return CloudSlotState(
            available_slots=fresh_slots,
            status=fresh_status or "ready",
            active_jobs=fresh_active,
            raw_path=",".join(fresh_paths),
            stale=False,
        )
    if fallback_stale is not None:
        return fallback_stale
    return CloudSlotState(available_slots=0, status="unavailable", active_jobs=0, raw_path=None, stale=True)


def read_proc_start_time(pid: int) -> tuple[str, int] | None:
    """Return (proc_state, starttime_ticks) from /proc/<pid>/stat, or None if missing."""
    stat_path = Path(f"/proc/{int(pid)}/stat")
    if not stat_path.exists():
        return None
    try:
        raw = stat_path.read_text(encoding="utf-8", errors="ignore").strip()
    except OSError:
        return None
    rparen = raw.rfind(")")
    if rparen < 0:
        return None
    rest = raw[rparen + 2 :].split()
    # After `(comm)`, field index 0 is `state` (proc field 3), index 19 is `starttime` (proc field 22).
    if len(rest) <= 19:
        return None
    try:
        return rest[0], int(rest[19])
    except ValueError:
        return None


def _normalize_expected_start_time(expected_start_time: Any) -> int | None:
    """Normalize an int or `(state, starttime_ticks)` tuple/list into an int start time."""
    if expected_start_time is None or isinstance(expected_start_time, bool):
        return None
    if isinstance(expected_start_time, (list, tuple)):
        if len(expected_start_time) >= 2:
            try:
                return int(expected_start_time[1])
            except (TypeError, ValueError):
                return None
        return None
    try:
        return int(expected_start_time)
    except (TypeError, ValueError):
        return None


def verify_process_identity(
    pid: int,
    expected_start_time: int | Sequence[Any] | None,
) -> ProcessLiveness:
    """Verify PID existence and /proc/<pid>/stat start time to avoid PID recycling bugs."""
    norm_expected = _normalize_expected_start_time(expected_start_time)
    parsed = read_proc_start_time(pid)
    if parsed is None:
        return ProcessLiveness(
            pid=pid,
            alive=False,
            state="MISSING",
            expected_start_time=norm_expected,
            actual_start_time=None,
            recycled=False,
        )
    state, actual_start = parsed
    if state == "Z":
        return ProcessLiveness(
            pid=pid,
            alive=False,
            state=state,
            expected_start_time=norm_expected,
            actual_start_time=actual_start,
            recycled=False,
        )
    if norm_expected is not None and actual_start != norm_expected:
        return ProcessLiveness(
            pid=pid,
            alive=False,
            state=state,
            expected_start_time=norm_expected,
            actual_start_time=actual_start,
            recycled=True,
        )
    return ProcessLiveness(
        pid=pid,
        alive=True,
        state=state,
        expected_start_time=norm_expected,
        actual_start_time=actual_start,
        recycled=False,
    )


def _host_pool_blocks() -> dict[str, set[str]]:
    """Parse ``CONJECTRA_HOST_POOL_BLOCKS="remote=llm_gemini,llm_x;edge=llm_claude"``."""
    out: dict[str, set[str]] = {}
    for part in os.environ.get("CONJECTRA_HOST_POOL_BLOCKS", "").split(";"):
        host, sep, pools = part.partition("=")
        if sep and host.strip():
            out[host.strip().lower()] = {p.strip().lower() for p in pools.split(",") if p.strip()}
    return out


def validate_pool_policy(
    *,
    pool: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    host: str | None = None,
) -> tuple[bool, str]:
    """Enforce hard routing guards: banned relay substrings, the Sonnet effort cap, and optional
    per-host pool blocks (``CONJECTRA_HOST_POOL_BLOCKS``)."""
    combined = f"{pool or ''} {model or ''}".lower()
    for banned in BANNED_POOL_SUBSTRINGS:
        if banned in combined:
            return False, f"BLOCKED_BANNED_CHANNEL:{banned}"

    norm_pool = POOL_NORMALIZATION.get((pool or "").strip().lower(), (pool or "").strip().lower())
    low_model = (model or "").strip().lower()
    low_effort = (effort or "").strip().lower()

    if "sonnet" in low_model and low_effort in SONNET_BLOCKED_EFFORTS:
        return False, f"BLOCKED_SONNET_CAP:effort={low_effort}"

    blocked = _host_pool_blocks().get((host or "").strip().lower(), set())
    if norm_pool in blocked:
        return False, f"BLOCKED_HOST_POOL:{norm_pool}@{host}"

    return True, "OK"


class ResourceArbiter:
    """Cross-project resource, quota, disk, fair-share, retry, and recovery arbiter."""

    def __init__(
        self,
        *,
        quotas_path: Path | str | None = None,
        quota_snapshot_path: Path | str | None = None,
        ledger_path: Path | str | None = None,
        gemini_quota_path: Path | str | None = None,
        telemetry_script: Path | str | None = None,
        disk_mounts: Sequence[str | Path] = DEFAULT_DISK_MOUNTS,
        disk_max_pct: float = 85.0,
        min_ram_floor_mb: int | None = None,
        max_load1: float = 32.0,
        min_gpu_free_mb: int = 2048,
        cny_valley_cutoff: float = 48.0,
        quota_high_watermark_pct: float = 95.0,
        max_infra_retries: int = 2,
        max_gate_repairs: int = 1,
        circuit_breaker_threshold: int = 3,
        lock_dir: Path | str | None = None,
        cloud_slot_search_paths: Sequence[str | Path] | None = None,
        use_live_telemetry_cli: bool = False,
    ) -> None:
        self.quotas_path = Path(quotas_path or (_CONFIG_DIR / "quotas.json"))
        if quota_snapshot_path is not None:
            self.quota_snapshot_path: Path | None = Path(quota_snapshot_path)
        elif quotas_path is None and gemini_quota_path is None:
            self.quota_snapshot_path = _CONFIG_DIR / "quota-snapshot.json"
        else:
            self.quota_snapshot_path = None
        self.ledger_path = Path(ledger_path or (_CONFIG_DIR / "spend-ledger.json"))
        self.gemini_quota_path = Path(gemini_quota_path or (_CONFIG_DIR / "gemini-quota.json"))
        self.telemetry_script = Path(telemetry_script) if telemetry_script else _CONFIG_DIR / "quota_telemetry.py"
        self.disk_mounts = tuple(disk_mounts)
        self.disk_max_pct = float(disk_max_pct)
        self._explicit_min_ram_floor_mb = None if min_ram_floor_mb is None else int(min_ram_floor_mb)
        self.min_ram_floor_mb = 2048 if min_ram_floor_mb is None else int(min_ram_floor_mb)
        self.max_load1 = float(max_load1)
        self.min_gpu_free_mb = int(min_gpu_free_mb)
        self.cny_valley_cutoff = float(cny_valley_cutoff)
        self.quota_high_watermark_pct = float(quota_high_watermark_pct)
        self.max_infra_retries = int(max_infra_retries)
        self.max_gate_repairs = int(max_gate_repairs)
        self.circuit_breaker_threshold = int(circuit_breaker_threshold)
        self.lock_dir = lock_dir
        self.cloud_slot_search_paths = cloud_slot_search_paths
        self.use_live_telemetry_cli = use_live_telemetry_cli

        self._lane_failures: dict[str, int] = {}
        self._lane_failed_pools: dict[str, set[str]] = {}
        self._tripped_lanes: set[str] = set()
        self._infra_retries: dict[str, int] = {}
        self._gate_repairs: dict[str, int] = {}

    def read_cny_spend_today(self, *, now_cst: dt.datetime | None = None) -> dict[str, Any]:
        """Read today's pay-as-you-go spend from the local spend ledger and the off-peak window status.

        Used as a fuse for the (disabled-by-default) `deepseek-local` pay-per-token pool: calls are
        only allowed inside the provider's off-peak window (00:30-08:30 UTC+8) and below a daily cutoff.
        """
        cst = now_cst or dt.datetime.now(dt.timezone(dt.timedelta(hours=8)))
        mins = cst.hour * 60 + cst.minute
        in_valley = 30 <= mins < (8 * 60 + 30)
        today = cst.strftime("%Y-%m-%d")
        spent = 0.0
        if self.ledger_path.exists():
            try:
                doc = json.loads(self.ledger_path.read_text(encoding="utf-8"))
                days = doc.get("days") if isinstance(doc, dict) else None
                if isinstance(days, dict) and today in days:
                    spent = float((days[today] or {}).get("costCny") or 0.0)
                else:
                    entries = doc if isinstance(doc, list) else doc.get("entries") or doc.get("records") or []
                    spent = sum(
                        float(e.get("costCny") or e.get("cny") or 0.0)
                        for e in entries
                        if isinstance(e, dict) and today in str(e.get("date") or e.get("timestamp") or "")
                    )
            except Exception:
                spent = 0.0
        return {
            "date_cst": today,
            "in_valley_0030_0830": in_valley,
            "spent_cny": round(spent, 4),
            "cutoff_cny": self.cny_valley_cutoff,
            "valley_allowed": bool(in_valley and spent < self.cny_valley_cutoff),
        }

    def read_pool_headroom(self) -> dict[str, dict[str, Any]]:
        """Read 7-day quota headroom per logical pool from local JSON files (or an optional telemetry script).

        All sources are optional; with none present every pool is reported available.
        See docs/configuration.md for the accepted file formats.
        """
        pools: dict[str, dict[str, Any]] = {
            "llm_gemini": {"used_pct": None, "available": True, "source": "default"},
            "llm_claude": {"used_pct": None, "available": True, "source": "default"},
            "llm_deepseek": {"used_pct": None, "available": True, "source": "default"},
            "llm_codex_audit": {"used_pct": None, "available": True, "source": "default"},
        }

        if self.use_live_telemetry_cli and self.telemetry_script.exists():
            try:
                proc = subprocess.run(
                    [sys.executable, "-B", str(self.telemetry_script), "--minutes", "15"],
                    capture_output=True,
                    text=True,
                    timeout=20,
                    check=False,
                )
                if proc.returncode == 0 and proc.stdout.strip().startswith("{"):
                    tel = json.loads(proc.stdout)
                    rec = tel.get("recommendation") or {}
                    if rec.get("freeze_gemini_subagent_spawns"):
                        pools["llm_gemini"]["available"] = False
                        pools["llm_gemini"]["source"] = "telemetry_freeze"
                    q7 = tel.get("owned_7d_subscription_quotas") or {}
                    self._merge_quota_snapshot(pools, q7)
                    return pools
            except Exception:
                pass

        if self.gemini_quota_path.exists():
            try:
                vg = json.loads(self.gemini_quota_path.read_text(encoding="utf-8"))
                vg_pools = vg.get("pools") if isinstance(vg, dict) else None
                if isinstance(vg_pools, dict):
                    gem = (vg_pools.get("gemini") or {}).get("weekly") or {}
                    rf = gem.get("remaining_fraction")
                    if rf is not None:
                        used_pct = round((1.0 - max(0.0, min(1.0, float(rf)))) * 100.0, 2)
                        pools["llm_gemini"] = {
                            "used_pct": used_pct,
                            "available": used_pct < self.quota_high_watermark_pct,
                            "source": "gemini_quota",
                        }
            except Exception:
                pass

        if self.quotas_path.exists():
            try:
                qdoc = json.loads(self.quotas_path.read_text(encoding="utf-8"))
                self._merge_quotas_json(pools, qdoc)
            except Exception:
                pass

        if self.quota_snapshot_path is not None and self.quota_snapshot_path.exists():
            try:
                snap_doc = json.loads(self.quota_snapshot_path.read_text(encoding="utf-8"))
                if isinstance(snap_doc, list):
                    self._merge_quota_snapshot_array(pools, snap_doc)
                elif isinstance(snap_doc, Mapping):
                    self._merge_quotas_json(pools, snap_doc)
            except Exception:
                pass

        return pools

    def _merge_quota_snapshot_array(
        self,
        pools: dict[str, dict[str, Any]],
        snap_list: Sequence[Any],
        *,
        max_age_s: float = 21600.0,
        now_utc: dt.datetime | None = None,
    ) -> None:
        """Parse a `[{provider, at, windows:[{name, used}]}]` quota snapshot array with a 6h freshness check."""
        now_dt = now_utc or dt.datetime.now(dt.timezone.utc)
        claude_used: list[float] = []
        codex_used: list[float] = []
        od_used: list[float] = []

        for item in snap_list:
            if not isinstance(item, Mapping):
                continue
            provider = str(item.get("provider") or "").strip().lower()
            if provider not in ("claude", "anthropic", "codex", "openai", "deepseek"):
                continue

            at_raw = str(item.get("at") or "").strip()
            if at_raw:
                try:
                    iso_str = at_raw[:-1] + "+00:00" if at_raw.endswith("Z") else at_raw
                    at_dt = dt.datetime.fromisoformat(iso_str)
                    if at_dt.tzinfo is None:
                        at_dt = at_dt.replace(tzinfo=dt.timezone.utc)
                    age_s = (now_dt - at_dt).total_seconds()
                    if age_s > max_age_s:
                        continue
                except ValueError:
                    continue

            for w in item.get("windows") or []:
                if not isinstance(w, Mapping) or w.get("used") is None:
                    continue
                wname = str(w.get("name") or "").lower()
                if "7" not in wname and "week" not in wname:
                    continue
                used_val = float(w["used"])
                if provider in ("claude", "anthropic"):
                    claude_used.append(used_val)
                elif provider in ("codex", "openai"):
                    codex_used.append(used_val)
                elif provider == "deepseek":
                    od_used.append(used_val)

        for target_pool, vals in (
            ("llm_claude", claude_used),
            ("llm_codex_audit", codex_used),
            ("llm_deepseek", od_used),
        ):
            if vals and pools[target_pool].get("source") == "default":
                min_used = min(vals)
                pools[target_pool] = {
                    "used_pct": min_used,
                    "available": min_used < self.quota_high_watermark_pct,
                    "source": "quota_snapshot",
                }

    def _merge_quota_snapshot(self, pools: dict[str, dict[str, Any]], q7: Mapping[str, Any]) -> None:
        for src_key, target_pool in (
            ("codex_pro", "llm_codex_audit"),
            ("claude", "llm_claude"),
            ("deepseek", "llm_deepseek"),
        ):
            entries = q7.get(src_key) or []
            used_vals = []
            for ent in entries:
                for w in ent.get("windows") or []:
                    if "7" in str(w.get("name") or "") and w.get("used_pct") is not None:
                        used_vals.append(float(w["used_pct"]))
            if used_vals:
                min_used = min(used_vals)
                pools[target_pool] = {
                    "used_pct": min_used,
                    "available": min_used < self.quota_high_watermark_pct,
                    "source": "telemetry",
                }

    def _merge_quotas_json(self, pools: dict[str, dict[str, Any]], qdoc: Mapping[str, Any]) -> None:
        claude_used: list[float] = []
        codex_used: list[float] = []
        od_used: list[float] = []
        for key, entry in (qdoc or {}).items():
            q = (entry or {}).get("quota") or {}
            wins = q.get("windows") or []
            for w in wins:
                wname = str(w.get("name") or "").lower()
                if "7" not in wname and "week" not in wname:
                    continue
                if w.get("used") is None:
                    continue
                used = float(w["used"])
                if key.startswith(("claude/", "anthropic/")):
                    claude_used.append(used)
                elif "codex" in key.lower() or "openai" in key.lower():
                    codex_used.append(used)
                elif "deepseek" in key.lower():
                    od_used.append(used)
        for target_pool, vals in (
            ("llm_claude", claude_used),
            ("llm_codex_audit", codex_used),
            ("llm_deepseek", od_used),
        ):
            if vals:
                min_used = min(vals)
                pools[target_pool] = {
                    "used_pct": min_used,
                    "available": min_used < self.quota_high_watermark_pct,
                    "source": "quotas_json",
                }

    def select_fallback_pool(
        self,
        failed_pool: str | None,
        *,
        exclude_pools: Sequence[str] | set[str] | None = None,
        pool_headroom: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> str | None:
        """Pick the next healthy subscription pool when the current pool fails or hits quota."""
        headroom = pool_headroom if pool_headroom is not None else self.read_pool_headroom()
        norm_failed = POOL_NORMALIZATION.get((failed_pool or "").strip().lower(), failed_pool)
        excluded = {
            POOL_NORMALIZATION.get(str(p).strip().lower(), str(p))
            for p in (exclude_pools or ())
        }
        if norm_failed:
            excluded.add(str(norm_failed))
        for cand in DEFAULT_POOL_FALLBACK_ORDER:
            if cand in excluded:
                continue
            info = headroom.get(cand) or {}
            if info.get("available", True):
                return cand
        return None

    def select_fallback_model_spec(
        self,
        failed_pool_or_spec: str | None = None,
        *,
        role: str = "engineer_builder",
        avoid_families: Sequence[str] = (),
        exclude_specs: Sequence[str] = (),
    ) -> str | None:
        """Select a capability-verified fallback `pool:canonical_model:effort` spec via ModelRouter."""
        from conjectra.model_router import ModelRouter, PolicyBlockedError

        excluded = list(exclude_specs)
        if failed_pool_or_spec and failed_pool_or_spec.strip():
            excluded.append(failed_pool_or_spec.strip())
        router = ModelRouter(arbiter=self)
        try:
            route = router.resolve_route(
                role,
                avoid_families=avoid_families,
                exclude_specs=excluded,
            )
            return route.canonical_spec()
        except PolicyBlockedError:
            return None

    def is_lane_open(self, lane_id: str) -> bool:
        """Return True unless the lane-local circuit breaker has tripped."""
        return lane_id not in self._tripped_lanes

    def record_success(self, lane_id: str) -> None:
        """Reset failure counters and close the lane-local circuit breaker on success."""
        self._lane_failures.pop(lane_id, None)
        self._lane_failed_pools.pop(lane_id, None)
        self._tripped_lanes.discard(lane_id)
        self._infra_retries.pop(lane_id, None)
        self._gate_repairs.pop(lane_id, None)

    def record_failure(
        self,
        lane_id: str,
        *,
        failure_kind: str | ProcessStatus,
        current_pool: str | None = None,
    ) -> RetryDecision:
        """Record a task failure and compute bounded retry / lane-local breaker decision."""
        kind_str = failure_kind.value if isinstance(failure_kind, ProcessStatus) else str(failure_kind).upper()
        self._lane_failures[lane_id] = self._lane_failures.get(lane_id, 0) + 1
        streak = self._lane_failures[lane_id]
        if current_pool:
            norm_curr = POOL_NORMALIZATION.get(current_pool.strip().lower(), current_pool.strip().lower())
            self._lane_failed_pools.setdefault(lane_id, set()).add(norm_curr)

        if streak >= self.circuit_breaker_threshold:
            self._tripped_lanes.add(lane_id)
            return RetryDecision(
                should_retry=False,
                reason=f"CIRCUIT_BREAKER_OPEN:streak={streak}",
                next_pool=None,
                lane_tripped=True,
            )

        is_gate_or_test = kind_str in ("FAIL_GATE", "FAIL_TEST", "GATE_FAIL", "TEST_FAIL")
        if is_gate_or_test:
            used = self._gate_repairs.get(lane_id, 0)
            if used < self.max_gate_repairs:
                self._gate_repairs[lane_id] = used + 1
                return RetryDecision(
                    should_retry=True,
                    reason=f"GATE_REPAIR:{used + 1}/{self.max_gate_repairs}",
                    next_pool=current_pool,
                    lane_tripped=False,
                )
            return RetryDecision(
                should_retry=False,
                reason=f"GATE_REPAIR_EXHAUSTED:{used}/{self.max_gate_repairs}",
                next_pool=None,
                lane_tripped=False,
            )

        is_infra = kind_str in (
            ProcessStatus.ENV_ERROR.value,
            ProcessStatus.TIMEOUT.value,
            ProcessStatus.LEASE_LOST.value,
            ProcessStatus.OOM.value,
            "INFRA_ERROR",
            "TRANSIENT",
        )
        if is_infra:
            used = self._infra_retries.get(lane_id, 0)
            if used < self.max_infra_retries:
                self._infra_retries[lane_id] = used + 1
                failed_set = self._lane_failed_pools.get(lane_id, set())
                fallback = (
                    self.select_fallback_pool(current_pool, exclude_pools=failed_set)
                    if current_pool
                    else None
                )
                return RetryDecision(
                    should_retry=True,
                    reason=f"INFRA_RETRY:{used + 1}/{self.max_infra_retries}",
                    next_pool=fallback or current_pool,
                    lane_tripped=False,
                )
            return RetryDecision(
                should_retry=False,
                reason=f"INFRA_RETRIES_EXHAUSTED:{used}/{self.max_infra_retries}",
                next_pool=None,
                lane_tripped=False,
            )

        return RetryDecision(
            should_retry=False,
            reason=f"NON_RETRYABLE:{kind_str}",
            next_pool=None,
            lane_tripped=False,
        )

    @staticmethod
    def task_usage_key(task: StageTask) -> str:
        """Canonical `token_usage_ledger` reservation key for a StageTask attempt."""
        return f"task:{task.task_id}:a{int(task.attempt)}"

    @classmethod
    def resolve_goal_tokens_for_task(
        cls,
        db: Any,
        task: StageTask,
        *,
        fallback_tokens: int = 0,
        exclude_usage_key: str | None = None,
    ) -> int:
        """Return committed goal tokens excluding `task`'s own active reservation."""
        if db is None or not hasattr(db, "get_goal_token_usage"):
            return int(fallback_tokens)
        own_key = exclude_usage_key or cls.task_usage_key(task)
        import inspect as _inspect

        params = _inspect.signature(db.get_goal_token_usage).parameters
        if "exclude_usage_key" in params:
            usage = db.get_goal_token_usage(task.goal_id, exclude_usage_key=own_key)
            return int(usage.get("committed_tokens", usage.get("total_tokens", 0)))

        usage = db.get_goal_token_usage(task.goal_id)
        committed = int(usage.get("committed_tokens", usage.get("total_tokens", 0)))
        if hasattr(db, "_lock") and hasattr(db, "_conn"):
            try:
                with db._lock:
                    has_tbl = db._conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'token_usage_ledger'"
                    ).fetchone()
                    if has_tbl is not None:
                        row = db._conn.execute(
                            "SELECT COALESCE(SUM(reserved_tokens), 0) AS own_res FROM token_usage_ledger WHERE goal_id = ? AND usage_key = ? AND state = 'RESERVED'",
                            (task.goal_id, own_key),
                        ).fetchone()
                        if row is not None:
                            committed = max(0, committed - int(row["own_res"] or 0))
            except Exception:
                pass
        return committed

    @classmethod
    def is_goal_budget_exhausted_for_task(
        cls,
        db: Any,
        task: StageTask,
        *,
        max_tokens_budget: int | None = None,
        exclude_usage_key: str | None = None,
    ) -> tuple[bool, int]:
        """Check whether a Goal's token budget is exhausted excluding `task`'s own reservation."""
        own_key = exclude_usage_key or cls.task_usage_key(task)
        if db is not None and hasattr(db, "is_budget_exhausted"):
            import inspect as _inspect

            params = _inspect.signature(db.is_budget_exhausted).parameters
            if "exclude_usage_key" in params:
                exhausted = bool(db.is_budget_exhausted(task.goal_id, exclude_usage_key=own_key))
                used = cls.resolve_goal_tokens_for_task(db, task, exclude_usage_key=own_key)
                return exhausted, used
        used = cls.resolve_goal_tokens_for_task(db, task, exclude_usage_key=own_key)
        if max_tokens_budget is not None and int(max_tokens_budget) > 0:
            return used >= int(max_tokens_budget), used
        return False, used

    def can_admit(
        self,
        task: StageTask,
        *,
        project: str,
        lane_id: str | None = None,
        model: str | None = None,
        effort: str | None = None,
        host: str | None = None,
        tokens_used_for_goal: int = 0,
        max_tokens_budget: int | None = None,
        db: Any = None,
        exclude_usage_key: str | None = None,
        active_by_project: Mapping[str, int] | None = None,
        queued_projects: Sequence[str] | None = None,
        total_slots: int = 4,
        now_cst: dt.datetime | None = None,
    ) -> AdmissionDecision:
        """Evaluate all disk, RAM, load, GPU, quota, policy, budget, and fair-share guards."""
        effective_lane = lane_id or task.hypothesis_id
        if not self.is_lane_open(effective_lane):
            return AdmissionDecision(allowed=False, reason=f"CIRCUIT_BREAKER_OPEN:{effective_lane}")

        res = task.resources if isinstance(task.resources, ResourceSpec) else ResourceSpec.from_dict(task.resources)
        heavy_stage = task.stage in (
            StageKind.CODE,
            StageKind.DESIGN_REVIEW,
            StageKind.IMPL_REVIEW,
            StageKind.RESULT_REVIEW,
            StageKind.EXP_SCREEN,
            StageKind.EXP_CONFIRM,
            StageKind.BLIND_AUDIT,
        ) or res.gpu > 0 or res.eda_sim > 0 or res.ram_mb >= 4096

        # 1. Disk guard (<85%) for heavy / sandbox / code starts
        disks = check_disk_usage(self.disk_mounts)
        if heavy_stage:
            for d in disks:
                if d.used_pct >= self.disk_max_pct:
                    return AdmissionDecision(
                        allowed=False,
                        reason=f"DISK_FULL:{d.mount}:{d.used_pct}%>={self.disk_max_pct}%",
                        disk_statuses=disks,
                    )

        # 2. Token budget fuse (excluding task's own active reservation when db is available)
        eff_tokens_used = (
            self.resolve_goal_tokens_for_task(
                db,
                task,
                fallback_tokens=tokens_used_for_goal,
                exclude_usage_key=exclude_usage_key,
            )
            if db is not None
            else int(tokens_used_for_goal)
        )
        if max_tokens_budget is not None and max_tokens_budget > 0 and eff_tokens_used >= max_tokens_budget:
            if res.llm_pool or (isinstance(task.pinned_route, Mapping) and task.pinned_route.get("pool")):
                return AdmissionDecision(
                    allowed=False,
                    reason=f"TOKEN_BUDGET_EXCEEDED:{eff_tokens_used}>={max_tokens_budget}",
                    disk_statuses=disks,
                )

        # 3. Live RAM & loadavg check
        mem_mb = read_mem_available_mb()
        load1 = read_load1()
        if self._explicit_min_ram_floor_mb is not None:
            ram_floor_mb = self._explicit_min_ram_floor_mb
        elif task.stage in (
            StageKind.CPU_GATE,
            StageKind.EXP_SCREEN,
            StageKind.EXP_CONFIRM,
            StageKind.BLIND_AUDIT,
        ):
            ram_floor_mb = int(os.environ.get("CONJECTRA_MIN_AVAIL_RAM_MB", "10240"))
        else:
            ram_floor_mb = 2048

        if mem_mb is not None and mem_mb < ram_floor_mb:
            return AdmissionDecision(
                allowed=False,
                reason=f"MEM_BELOW_10GB_WAIT: MemAvailable={mem_mb}MB < {ram_floor_mb}MB",
                disk_statuses=disks,
                mem_available_mb=mem_mb,
                load1=load1,
            )
        if mem_mb is not None and (mem_mb - res.ram_mb) < ram_floor_mb:
            return AdmissionDecision(
                allowed=False,
                reason=f"LOW_RAM:avail={mem_mb}MB,req={res.ram_mb}MB,floor={ram_floor_mb}MB",
                disk_statuses=disks,
                mem_available_mb=mem_mb,
                load1=load1,
            )
        if load1 is not None and load1 >= self.max_load1 and heavy_stage:
            return AdmissionDecision(
                allowed=False,
                reason=f"HIGH_LOAD:load1={load1:.2f}>={self.max_load1:.1f}",
                disk_statuses=disks,
                mem_available_mb=mem_mb,
                load1=load1,
            )

        # 4. GPU & heavy slot lock check
        if res.gpu > 0:
            gpus = read_gpu_status()
            if not any(g.free_mb >= self.min_gpu_free_mb for g in gpus):
                return AdmissionDecision(
                    allowed=False,
                    reason=f"GPU_VRAM_LOW:need>={self.min_gpu_free_mb}MB",
                    disk_statuses=disks,
                    mem_available_mb=mem_mb,
                    load1=load1,
                )
            slot_snap = inspect_heavy_slots(self.lock_dir)
            if slot_snap.busy_slots >= slot_snap.max_slots:
                return AdmissionDecision(
                    allowed=False,
                    reason=f"HEAVY_SLOTS_FULL:{slot_snap.busy_slots}/{slot_snap.max_slots}",
                    disk_statuses=disks,
                    mem_available_mb=mem_mb,
                    load1=load1,
                )

        # 5. Cloud slot check (read-only)
        if res.cloud_slot > 0:
            cloud = read_cloud_slot_state(self.cloud_slot_search_paths)
            if cloud.available_slots < res.cloud_slot:
                return AdmissionDecision(
                    allowed=False,
                    reason=f"CLOUD_SLOT_UNAVAILABLE:avail={cloud.available_slots},status={cloud.status}",
                    disk_statuses=disks,
                    mem_available_mb=mem_mb,
                    load1=load1,
                )

        # 6. Pool policy + 7d quota + CNY valley fuse
        resolved_pool = None
        fallback_pool = None
        if res.llm_pool:
            ok_pol, pol_reason = validate_pool_policy(
                pool=res.llm_pool,
                model=model,
                effort=effort,
                host=host,
            )
            if not ok_pol:
                return AdmissionDecision(
                    allowed=False,
                    reason=pol_reason,
                    disk_statuses=disks,
                    mem_available_mb=mem_mb,
                    load1=load1,
                )
            norm_pool = _normalize_pool_name(res.llm_pool)
            if norm_pool == "deepseek_local":
                cny = self.read_cny_spend_today(now_cst=now_cst)
                if not cny["valley_allowed"]:
                    return AdmissionDecision(
                        allowed=False,
                        reason=f"VALLEY_FUSE_BLOCKED:in_valley={cny['in_valley_0030_0830']},spent={cny['spent_cny']}",
                        disk_statuses=disks,
                        mem_available_mb=mem_mb,
                        load1=load1,
                    )
            headroom = self.read_pool_headroom()
            pool_info = headroom.get(norm_pool)
            if pool_info and not pool_info.get("available", True):
                fallback_pool = self.select_fallback_pool(norm_pool, pool_headroom=headroom)
                if not fallback_pool:
                    return AdmissionDecision(
                        allowed=False,
                        reason=f"POOL_QUOTA_EXHAUSTED:{norm_pool}",
                        disk_statuses=disks,
                        mem_available_mb=mem_mb,
                        load1=load1,
                    )
                resolved_pool = fallback_pool
            else:
                resolved_pool = norm_pool

        # 7. Cross-project fair-share guard
        if active_by_project is not None and queued_projects:
            other_queued = [p for p in queued_projects if p != project]
            if other_queued and total_slots >= 2:
                proj_active = int(active_by_project.get(project, 0))
                other_active = sum(int(active_by_project.get(p, 0)) for p in other_queued)
                fair_cap = max(1, total_slots - len(set(other_queued)))
                if proj_active >= fair_cap and other_active == 0:
                    return AdmissionDecision(
                        allowed=False,
                        reason=f"FAIR_SHARE_YIELD:project={project},active={proj_active},waiting={sorted(set(other_queued))}",
                        resolved_pool=resolved_pool,
                        fallback_pool=fallback_pool,
                        disk_statuses=disks,
                        mem_available_mb=mem_mb,
                        load1=load1,
                    )

        return AdmissionDecision(
            allowed=True,
            reason="ADMITTED",
            resolved_pool=resolved_pool,
            fallback_pool=fallback_pool,
            disk_statuses=disks,
            mem_available_mb=mem_mb,
            load1=load1,
        )

    @staticmethod
    def recover_stale_leases(
        leases: Sequence[Mapping[str, Any]],
        *,
        now_ts: float | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        """Classify persisted leases into alive vs stale-to-requeue using PID + /proc/<pid>/stat."""
        now = time.time() if now_ts is None else float(now_ts)
        still_alive: list[dict[str, Any]] = []
        stale_to_requeue: list[dict[str, Any]] = []

        for lease in leases:
            item = dict(lease)
            pid = item.get("pid")
            expected_start = item.get("proc_start_time")
            expires_ts = item.get("expires_ts")

            if pid is not None:
                liveness = verify_process_identity(
                    int(pid),
                    expected_start,
                )
                item["liveness"] = {
                    "alive": liveness.alive,
                    "state": liveness.state,
                    "recycled": liveness.recycled,
                    "actual_start_time": liveness.actual_start_time,
                }
                if liveness.alive:
                    still_alive.append(item)
                else:
                    stale_to_requeue.append(item)
                continue

            if expires_ts is not None and float(expires_ts) <= now:
                stale_to_requeue.append(item)
            else:
                still_alive.append(item)

        return {"still_alive": still_alive, "stale_to_requeue": stale_to_requeue}

    def check_llm_pool(
        self,
        pool: str,
        *,
        model: str | None = None,
        effort: str | None = None,
        host: str | None = None,
        now_cst: dt.datetime | None = None,
    ) -> AdmissionDecision:
        """Check whether an LLM pool satisfies routing policy, valley fuse, and 7d quota headroom."""
        ok_pol, pol_reason = validate_pool_policy(
            pool=pool,
            model=model,
            effort=effort,
            host=host,
        )
        if not ok_pol:
            return AdmissionDecision(allowed=False, reason=pol_reason)
        norm_pool = _normalize_pool_name(pool)
        if norm_pool == "deepseek_local":
            cny = self.read_cny_spend_today(now_cst=now_cst)
            if not cny["valley_allowed"]:
                return AdmissionDecision(
                    allowed=False,
                    reason=f"VALLEY_FUSE_BLOCKED:in_valley={cny['in_valley_0030_0830']},spent={cny['spent_cny']}",
                )
        headroom = self.read_pool_headroom()
        info = headroom.get(norm_pool)
        if info and not info.get("available", True):
            fallback = self.select_fallback_pool(norm_pool, pool_headroom=headroom)
            return AdmissionDecision(
                allowed=False,
                reason=f"POOL_QUOTA_EXHAUSTED:{norm_pool}",
                fallback_pool=fallback,
            )
        return AdmissionDecision(allowed=True, reason="OK", resolved_pool=norm_pool)


# ------------------------------------------------ Foreground Interactive Reserve


def _foreground_pings_path(arr_home: Path | str | None) -> Path:
    if arr_home is not None:
        base = Path(arr_home).expanduser().resolve()
    else:
        env_home = os.environ.get("CONJECTRA_HOME")
        base = Path(env_home).expanduser().resolve() if env_home else (Path.home() / ".conjectra").resolve()
    base.mkdir(parents=True, exist_ok=True)
    return base / "foreground_pings.json"


def _normalize_pool_name(provider: str | None) -> str:
    raw = (provider or "").strip().lower()
    if not raw:
        return "interactive"
    head = raw.split(":", 1)[0].strip()
    return POOL_NORMALIZATION.get(raw, POOL_NORMALIZATION.get(head, head or raw))


def record_foreground_ping(
    arr_home: Path | str | None,
    provider: str = "interactive",
    ttl_s: float = 120.0,
    *,
    now_ts: float | None = None,
) -> dict[str, Any]:
    """Record a foreground interactive session heartbeat with TTL for slot/quota reserve."""
    path = _foreground_pings_path(arr_home)
    now = time.time() if now_ts is None else float(now_ts)
    ttl = max(1.0, float(ttl_s))
    norm_pool = _normalize_pool_name(provider)
    lock_path = path.with_suffix(".lock")
    entry = {
        "provider": str(provider or "interactive"),
        "pool": norm_pool,
        "ts": now,
        "expires_ts": now + ttl,
        "pid": os.getpid(),
    }
    with lock_path.open("a+", encoding="utf-8") as lfh:
        fcntl.flock(lfh.fileno(), fcntl.LOCK_EX)
        try:
            current: dict[str, Any] = {}
            if path.is_file():
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(loaded, dict):
                        current = loaded
                except (OSError, json.JSONDecodeError):
                    current = {}
            cleaned: dict[str, Any] = {
                k: v
                for k, v in current.items()
                if isinstance(v, dict) and float(v.get("expires_ts", 0.0)) > now
            }
            cleaned[norm_pool] = entry
            tmp_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp_path.write_text(json.dumps(cleaned, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lfh.fileno(), fcntl.LOCK_UN)
    return entry


def read_active_foreground_pings(
    arr_home: Path | str | None,
    *,
    now_ts: float | None = None,
) -> dict[str, dict[str, Any]]:
    """Return active non-expired foreground interactive pings keyed by normalized pool."""
    path = _foreground_pings_path(arr_home)
    if not path.is_file():
        return {}
    now = time.time() if now_ts is None else float(now_ts)
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(loaded, dict):
        return {}
    return {
        str(k): dict(v)
        for k, v in loaded.items()
        if isinstance(v, dict) and float(v.get("expires_ts", 0.0)) > now
    }


def clear_foreground_pings(arr_home: Path | str | None) -> None:
    """Remove all foreground ping records under CONJECTRA_HOME (primarily for tests)."""
    path = _foreground_pings_path(arr_home)
    path.unlink(missing_ok=True)


# ------------------------------------------------ Unified Conflict Management


class ConflictManager:
    """Cross-Goal repo-scope mutex leases and device/slot/pool conflict checks."""

    @staticmethod
    def repo_hash(repo_path: str | Path) -> str:
        """Return a stable short SHA-1 hash for a canonical origin repository path."""
        resolved = str(Path(repo_path).expanduser().resolve())
        return hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:12]

    @classmethod
    def normalize_repo_scope_key(cls, repo_path: str | Path, path_prefix: str) -> str:
        """Build canonical `repo_scope:<repo_hash>:<norm_path>` mutex key."""
        rhash = cls.repo_hash(repo_path)
        np = gates.norm_path(path_prefix)
        scope = "__root__" if np in (None, "") else np
        return f"repo_scope:{rhash}:{scope}"

    @classmethod
    def extract_repo_scopes(
        cls,
        repo_path: str | Path,
        writable_paths: Sequence[str],
    ) -> list[tuple[str, str, str]]:
        """Return `(lease_key, repo_hash, norm_path)` tuples for all writable paths."""
        rhash = cls.repo_hash(repo_path)
        seen: set[str] = set()
        out: list[tuple[str, str, str]] = []
        raw_paths = list(writable_paths) or [""]
        for raw in raw_paths:
            np = gates.norm_path(raw)
            norm = "" if np is None else np
            if norm in seen:
                continue
            seen.add(norm)
            scope_label = "__root__" if norm == "" else norm
            out.append((f"repo_scope:{rhash}:{scope_label}", rhash, norm))
        return out

    @classmethod
    def has_repo_scope_conflict(
        cls,
        repo_path: str | Path,
        writable_paths: Sequence[str],
        active_leases: Sequence[Mapping[str, Any]],
        *,
        ignore_task_id: str | None = None,
    ) -> tuple[bool, str]:
        """Return (True, reason) when any active lease overlaps on the same origin repo."""
        rhash = cls.repo_hash(repo_path)
        prefix = f"repo_scope:{rhash}:"
        cand_norms = [
            "" if gates.norm_path(p) is None else str(gates.norm_path(p))
            for p in (list(writable_paths) or [""])
        ]
        for lease in active_leases:
            if ignore_task_id and str(lease.get("task_id") or "") == ignore_task_id:
                continue
            rk = str(lease.get("resource_key") or "")
            if not rk.startswith(prefix):
                continue
            held_raw = rk[len(prefix) :]
            held_norm = "" if held_raw == "__root__" else held_raw
            for cand in cand_norms:
                if gates.paths_overlap(cand, held_norm):
                    return (
                        True,
                        f"REPO_SCOPE_CONFLICT:{rk}:held_by={lease.get('task_id')}",
                    )
        return False, "OK"

    @classmethod
    def acquire_repo_scope_leases(
        cls,
        db: Any,
        *,
        task_id: str,
        goal_id: str,
        repo_path: str | Path,
        writable_paths: Sequence[str],
        holder: str,
        ttl_s: float = 300.0,
        now: float | None = None,
    ) -> list[dict[str, Any]] | None:
        """Atomically acquire `repo_scope:<repo_hash>:<norm_path>` leases or roll back on conflict."""
        ts = time.time() if now is None else float(now)
        active = db.list_active_leases(now=ts)
        conflict, _reason = cls.has_repo_scope_conflict(
            repo_path,
            writable_paths,
            active,
            ignore_task_id=task_id,
        )
        if conflict:
            return None

        scopes = cls.extract_repo_scopes(repo_path, writable_paths)
        acquired: list[dict[str, Any]] = []
        for lease_key, _rhash, _norm in scopes:
            lease = db.acquire_lease(
                lease_key,
                task_id,
                holder=holder,
                ttl_s=ttl_s,
                goal_id=goal_id,
                max_concurrent=1,
                now=ts,
            )
            if lease is None:
                for prev in acquired:
                    db.release_lease(prev["lease_id"], now=ts)
                return None
            acquired.append(lease)
        return acquired

    @staticmethod
    def release_task_leases(db: Any, task_id: str, *, now: float | None = None) -> int:
        """Release all active leases held by `task_id`."""
        ts = time.time() if now is None else float(now)
        released = 0
        for lease in db.list_active_leases(now=ts):
            if str(lease.get("task_id") or "") == task_id:
                if db.release_lease(lease["lease_id"], now=ts):
                    released += 1
        return released

    @classmethod
    def check_task_conflicts(
        cls,
        task: StageTask,
        goal: Any,
        hyp: Any,
        running_tasks: Sequence[StageTask],
        active_leases: Sequence[Mapping[str, Any]],
        *,
        goals_by_id: Mapping[str, Any] | None = None,
        hyps_by_id: Mapping[str, Any] | None = None,
        max_concurrent_test_sim: int = 2,
    ) -> tuple[bool, str]:
        """Check branch, cross-Goal shared-repo scope, and testsim/GPU/pool conflicts."""
        test_sim_stages = {StageKind.CPU_GATE, StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM}
        if task.stage in test_sim_stages:
            active_ts = sum(1 for rt in running_tasks if rt.stage in test_sim_stages)
            if active_ts >= max(1, int(max_concurrent_test_sim)):
                return False, f"GLOBAL_TESTSIM_CAP:{active_ts}/{max_concurrent_test_sim}"

        # Branch mutex across running tasks/leases
        branch_key = f"branch:{goal.goal_id}:{hyp.hypothesis_id}"
        for lease in active_leases:
            if lease.get("resource_key") == branch_key and lease.get("task_id") != task.task_id:
                return False, f"BRANCH_LEASE_HELD:{branch_key}"

        # Shared-repo scope conflict across ALL Goals & nested Workflows for CODE tasks
        if task.stage == StageKind.CODE:
            writable = list(getattr(hyp, "writable_paths", ()) or ())
            conflict, reason = cls.has_repo_scope_conflict(
                goal.repo_path,
                writable,
                active_leases,
                ignore_task_id=task.task_id,
            )
            if conflict:
                return False, reason

            # Also check against in-flight running CODE tasks on the same origin repo
            my_rhash = cls.repo_hash(goal.repo_path)
            for rt in running_tasks:
                if rt.task_id == task.task_id or rt.stage != StageKind.CODE:
                    continue
                other_goal = goals_by_id.get(rt.goal_id) if goals_by_id else None
                other_hyp = hyps_by_id.get(rt.hypothesis_id) if hyps_by_id else None
                if other_goal is None or other_hyp is None:
                    continue
                if cls.repo_hash(other_goal.repo_path) != my_rhash:
                    continue
                other_paths = list(getattr(other_hyp, "writable_paths", ()) or ()) or [""]
                my_paths = writable or [""]
                if gates.scope_overlaps(my_paths, other_paths):
                    return (
                        False,
                        f"CROSS_GOAL_SCOPE_OVERLAP:repo={my_rhash}:other_task={rt.task_id}",
                    )

        return True, "OK"


# -------------------------------------------- Multi-Goal Elastic Global Scheduler


_STAGE_COMPLETION_BONUS: dict[StageKind, float] = {
    StageKind.RECONCILE: 0.45,
    StageKind.BLIND_AUDIT: 0.35,
    StageKind.EXP_CONFIRM: 0.30,
    StageKind.EXP_SCREEN: 0.20,
    StageKind.CPU_GATE: 0.15,
    StageKind.CODE: 0.0,
    StageKind.PROPOSE: 0.05,
}


class GlobalResourceScheduler:
    """Multi-Goal elastic slot borrowing, aging, diminishing-returns, and foreground reserve scheduler."""

    def __init__(
        self,
        arbiter: ResourceArbiter | None = None,
        *,
        arr_home: Path | str | None = None,
        global_test_sim_cap: int = 2,
        foreground_reserve_slots: int = 1,
        foreground_reserve_quota_pct: float = 15.0,
        aging_rate_per_min: float = 0.25,
        aging_max_bonus: float = 3.0,
        diminishing_returns_decay: float = 0.85,
        always_reserve_foreground: bool = False,
    ) -> None:
        self.arbiter = arbiter or ResourceArbiter()
        self.arr_home = arr_home
        self.global_test_sim_cap = max(1, int(global_test_sim_cap))
        self.foreground_reserve_slots = max(0, int(foreground_reserve_slots))
        self.foreground_reserve_quota_pct = max(0.0, min(100.0, float(foreground_reserve_quota_pct)))
        self.aging_rate_per_min = max(0.0, float(aging_rate_per_min))
        self.aging_max_bonus = max(0.0, float(aging_max_bonus))
        self.diminishing_returns_decay = max(0.1, min(1.0, float(diminishing_returns_decay)))
        self.always_reserve_foreground = bool(always_reserve_foreground)

    def effective_background_slot_cap(
        self,
        total_slots: int,
        *,
        arr_home: Path | str | None = None,
        now_ts: float | None = None,
    ) -> tuple[int, int]:
        """Return `(background_slot_cap, reserved_foreground_slots)` given live foreground pings."""
        home = arr_home if arr_home is not None else self.arr_home
        pings = read_active_foreground_pings(home, now_ts=now_ts) if home is not None else {}
        reserve = self.foreground_reserve_slots if (self.always_reserve_foreground or pings) else 0
        reserve = min(max(0, int(total_slots)), reserve)
        return max(0, int(total_slots) - reserve), reserve

    def check_foreground_quota_reserve(
        self,
        pool: str | None,
        *,
        arr_home: Path | str | None = None,
        now_ts: float | None = None,
    ) -> tuple[bool, str]:
        """Ensure background LLM tasks keep >= `foreground_reserve_quota_pct` headroom when foreground is active."""
        if not pool or self.foreground_reserve_quota_pct <= 0:
            return True, "OK"
        home = arr_home if arr_home is not None else self.arr_home
        if home is None:
            return True, "OK"
        pings = read_active_foreground_pings(home, now_ts=now_ts)
        if not pings:
            return True, "OK"
        norm_pool = _normalize_pool_name(pool)
        if norm_pool not in pings and "interactive" not in pings:
            # Protect any pinged provider pool specifically, plus general reserve when headroom is tight.
            pass
        headroom = self.arbiter.read_pool_headroom()
        info = headroom.get(norm_pool) or {}
        used_pct = info.get("used_pct")
        if used_pct is not None:
            remaining_pct = 100.0 - float(used_pct)
            if remaining_pct <= self.foreground_reserve_quota_pct:
                return (
                    False,
                    f"FOREGROUND_QUOTA_RESERVE:{norm_pool}:remaining={remaining_pct:.1f}%<={self.foreground_reserve_quota_pct:.1f}%",
                )
        return True, "OK"

    def compute_task_score(
        self,
        task: StageTask,
        hyp: Any,
        goal: Any,
        *,
        recent_goal_receipts: int = 0,
        now_ts: float | None = None,
    ) -> dict[str, Any]:
        """Compute marginal scientific value/cost score with aging and diminishing returns."""
        now = time.time() if now_ts is None else float(now_ts)
        base_evoi = max(0.01, float(getattr(hyp, "priority_evoi", 1.0)))
        priority_weight = max(0.1, float(getattr(goal, "priority_weight", 1.0)))
        decay = self.diminishing_returns_decay ** max(0, int(recent_goal_receipts))
        wait_sec = max(0.0, now - float(task.enqueued_ts or now))
        aging_bonus = min(self.aging_max_bonus, (wait_sec / 60.0) * self.aging_rate_per_min)
        stage_bonus = _STAGE_COMPLETION_BONUS.get(task.stage, 0.0)
        res = task.resources if isinstance(task.resources, ResourceSpec) else ResourceSpec.from_dict(task.resources)
        est_cost = (
            1.0
            + (float(res.ram_mb) / 4096.0)
            + (0.5 if res.llm_pool else 0.0)
            + (0.5 if res.gpu > 0 else 0.0)
        )
        score = ((base_evoi * priority_weight * decay) + stage_bonus + aging_bonus) / est_cost
        return {
            "score": round(score, 6),
            "base_evoi": base_evoi,
            "priority_weight": priority_weight,
            "decay": round(decay, 4),
            "aging_bonus": round(aging_bonus, 4),
            "stage_bonus": stage_bonus,
            "est_cost": round(est_cost, 4),
        }

    def compute_goal_slot_allocations(
        self,
        goals: Sequence[Any],
        *,
        active_by_goal: Mapping[str, int],
        ready_by_goal: Mapping[str, int],
        blocked_goals: set[str] | frozenset[str] | Sequence[str] = (),
        total_slots: int = 4,
        arr_home: Path | str | None = None,
        now_ts: float | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Allocate guaranteed slots per runnable Goal and elastically redistribute idle/paused/blocked slots."""
        blocked_set = set(blocked_goals)
        bg_cap, fg_reserved = self.effective_background_slot_cap(
            total_slots, arr_home=arr_home, now_ts=now_ts
        )

        allocs: dict[str, dict[str, Any]] = {}
        runnable: list[Any] = []
        for g in goals:
            gid = g.goal_id
            max_b = max(1, int(getattr(g, "max_concurrent_branches", 4)))
            min_g = max(0, min(max_b, int(getattr(g, "min_guaranteed_slots", 1))))
            active = int(active_by_goal.get(gid, 0))
            ready = int(ready_by_goal.get(gid, 0))
            demand = active + ready
            is_paused = str(getattr(g, "status", "ACTIVE")).upper() != "ACTIVE"
            is_blocked = gid in blocked_set
            is_runnable = (not is_paused) and (not is_blocked) and demand > 0
            allocs[gid] = {
                "goal_id": gid,
                "priority_weight": float(getattr(g, "priority_weight", 1.0)),
                "min_guaranteed_slots": min_g,
                "max_concurrent_branches": max_b,
                "active_slots": active,
                "ready_tasks": ready,
                "demand": demand,
                "guaranteed_slots": 0,
                "allocated_slots": 0,
                "borrowed_slots": 0,
                "runnable": is_runnable,
                "blocked": is_blocked,
                "paused": is_paused,
                "foreground_reserved_slots": fg_reserved,
                "background_slot_cap": bg_cap,
            }
            if is_runnable:
                runnable.append(g)

        remaining = bg_cap
        # Pass 1: Assign min_guaranteed_slots (weighted order) for runnable Goals
        runnable_sorted = sorted(
            runnable,
            key=lambda g: (-float(getattr(g, "priority_weight", 1.0)), g.goal_id),
        )
        for g in runnable_sorted:
            info = allocs[g.goal_id]
            want_floor = min(info["min_guaranteed_slots"], info["demand"], remaining)
            info["guaranteed_slots"] = want_floor
            info["allocated_slots"] = want_floor
            remaining -= want_floor

        # Pass 2: Elastic borrowing of idle/paused/decision-blocked Goal slots up to max_concurrent_branches
        while remaining > 0:
            progressed = False
            for g in runnable_sorted:
                if remaining <= 0:
                    break
                info = allocs[g.goal_id]
                cap = min(info["max_concurrent_branches"], info["demand"])
                if info["allocated_slots"] < cap:
                    info["allocated_slots"] += 1
                    remaining -= 1
                    progressed = True
            if not progressed:
                break

        for info in allocs.values():
            info["borrowed_slots"] = max(0, info["active_slots"] - info["min_guaranteed_slots"])
            info["elastic_headroom"] = max(0, info["allocated_slots"] - info["active_slots"])

        return allocs

    def rank_ready_tasks_globally(
        self,
        ready_tasks: Sequence[StageTask],
        goals_by_id: Mapping[str, Any],
        hyps_by_id: Mapping[str, Any],
        *,
        active_by_goal: Mapping[str, int] | None = None,
        recent_receipts_by_goal: Mapping[str, int] | None = None,
        blocked_goals: set[str] | frozenset[str] | Sequence[str] = (),
        total_slots: int = 4,
        arr_home: Path | str | None = None,
        now_ts: float | None = None,
    ) -> list[dict[str, Any]]:
        """Rank ready tasks across ALL active Goals by guarantee deficit, marginal EVOI/cost, aging, and decay."""
        now = time.time() if now_ts is None else float(now_ts)
        active_map = dict(active_by_goal or {})
        rec_map = dict(recent_receipts_by_goal or {})
        ready_by_goal: dict[str, int] = {}
        for t in ready_tasks:
            ready_by_goal[t.goal_id] = ready_by_goal.get(t.goal_id, 0) + 1

        allocs = self.compute_goal_slot_allocations(
            list(goals_by_id.values()),
            active_by_goal=active_map,
            ready_by_goal=ready_by_goal,
            blocked_goals=blocked_goals,
            total_slots=total_slots,
            arr_home=arr_home,
            now_ts=now,
        )

        ranked: list[dict[str, Any]] = []
        for task in ready_tasks:
            goal = goals_by_id.get(task.goal_id)
            hyp = hyps_by_id.get(task.hypothesis_id)
            if goal is None or hyp is None:
                continue
            g_alloc = allocs.get(task.goal_id) or {}
            if not g_alloc.get("runnable", True):
                continue
            below_floor = 1 if active_map.get(task.goal_id, 0) < g_alloc.get("guaranteed_slots", 0) else 0
            breakdown = self.compute_task_score(
                task,
                hyp,
                goal,
                recent_goal_receipts=rec_map.get(task.goal_id, 0),
                now_ts=now,
            )
            ranked.append(
                {
                    "task": task,
                    "goal": goal,
                    "hyp": hyp,
                    "below_floor": below_floor,
                    "allocated_slots": g_alloc.get("allocated_slots", goal.max_concurrent_branches),
                    **breakdown,
                }
            )

        ranked.sort(
            key=lambda item: (
                -item["below_floor"],
                -item["score"],
                float(item["task"].enqueued_ts or 0.0),
                item["task"].task_id,
            )
        )
        return ranked
