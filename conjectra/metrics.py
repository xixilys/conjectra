"""Machine-verified metrics, baseline comparison, and status mirroring for conjectra."""

from __future__ import annotations

import json
import math
import os
import pathlib
import sqlite3
import shlex
import subprocess
import time
from typing import Any

from conjectra.gates import classify_failure_domain, is_valid_empirical_outcome
from conjectra.state_db import TERMINAL_TASK_SUCCESS, StateDB, resolve_arr_home
from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    FailureDomain,
    ProcessStatus,
    ResearchGoal,
    ReviewKind,
    ScientificVerdict,
    StageKind,
    StageTask,
)

# Optional remote mirror for status files, e.g. "user@dashboard-host:/srv/conjectra" (ssh) or a
# local directory path. Unset = no mirroring.
DEFAULT_MIRROR_TARGET = os.environ.get("CONJECTRA_STATUS_MIRROR", "")
# Optional remote baseline location "user@host:/path/baseline.json" fetched over ssh.
DEFAULT_REMOTE_BASELINE = os.environ.get("CONJECTRA_BASELINE_REMOTE", "")
BASELINE_CACHE_NAME = "baseline.json"

UNKNOWN = "UNKNOWN"

REAL_EVIDENCE_ORIGINS = frozenset(
    {"real_research_goal", "blind_audited", "real_llm_ablation"}
)
NON_REAL_EVIDENCE_ORIGINS = frozenset({"unit_test", "synthetic_sim", "mock", "dry_run"})
_INVALID_VALIDITIES = frozenset(
    {
        EvidenceValidity.INVALID_SYNTHETIC,
        EvidenceValidity.STALE_GOAL_VERSION,
        EvidenceValidity.HUMAN_INVALIDATED,
    }
)


def _is_real_evidence_receipt(receipt: EvidenceReceipt) -> bool:
    origin = (getattr(receipt, "evidence_origin", "real_research_goal") or "").strip()
    if not origin:
        return True
    return origin not in NON_REAL_EVIDENCE_ORIGINS


def _percentile(sorted_vals: list[float], pct: float) -> float:
    if not sorted_vals:
        raise ValueError("Cannot compute percentile of empty list")
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    rank = (pct / 100.0) * (len(sorted_vals) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return float(sorted_vals[lo])
    frac = rank - lo
    return float(sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo]))


def load_baseline(
    baseline_path: str | os.PathLike[str] | None = None,
    *,
    arr_home: str | os.PathLike[str] | None = None,
    remote: str | None = None,
    fetch_ssh: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    """Load the frozen baseline JSON from an explicit path, $CONJECTRA_BASELINE_PATH, the local
    cache, or (optionally) a remote ``user@host:/path`` over ssh."""
    home = resolve_arr_home(arr_home)
    cached_path = home / BASELINE_CACHE_NAME
    candidates: list[pathlib.Path] = []
    if baseline_path is not None:
        candidates.append(pathlib.Path(baseline_path).expanduser())
    env_baseline = os.environ.get("CONJECTRA_BASELINE_PATH")
    if env_baseline:
        candidates.append(pathlib.Path(env_baseline).expanduser())
    candidates.append(cached_path)

    for cand in candidates:
        if cand.is_file():
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
                if cand != cached_path:
                    home.mkdir(parents=True, exist_ok=True)
                    cached_path.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    )
                return data, str(cand)
            except (OSError, json.JSONDecodeError):
                continue

    target = remote if remote is not None else DEFAULT_REMOTE_BASELINE
    if fetch_ssh and target and ":" in target:
        host, _, rpath = target.partition(":")
        try:
            proc = subprocess.run(
                ["ssh", "-n", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", host, f"cat {shlex.quote(rpath)}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                data = json.loads(proc.stdout)
                home.mkdir(parents=True, exist_ok=True)
                cached_path.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8",
                )
                return data, f"ssh://{host}{rpath}"
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
            pass

    return None, "UNAVAILABLE"


def extract_baseline_metrics(baseline_raw: dict[str, Any] | None) -> dict[str, Any]:
    """Extract normalized baseline KPIs from the frozen baseline JSON."""
    if not baseline_raw:
        return {
            "baseline_id": UNKNOWN,
            "window_hours": UNKNOWN,
            "deliverables_total": UNKNOWN,
            "tested_commits": UNKNOWN,
            "in_tokens": UNKNOWN,
            "out_tokens": UNKNOWN,
            "total_tokens": UNKNOWN,
            "deliverables_per_hour": UNKNOWN,
            "tokens_per_deliverable": UNKNOWN,
            "target_sla": {},
        }

    window_hours = float(baseline_raw["window_hours"])
    prior = baseline_raw.get("prior_6h_reference", {})
    proj_stats = prior.get("project_stats", {})
    deliverables_total = sum(
        int(stats.get("code_commits", 0)) for stats in proj_stats.values()
    )
    tested_commits = sum(
        int(stats.get("tested_commits", 0)) for stats in proj_stats.values()
    )
    usage = prior.get("usage_6h", {})
    in_tokens = int(usage.get("in_tokens", 0))
    out_tokens = int(usage.get("out_tokens", 0))
    total_tokens = in_tokens + out_tokens

    deliv_per_hr: float | str = (
        round(deliverables_total / window_hours, 6) if window_hours > 0 else UNKNOWN
    )
    tok_per_deliv: float | str = (
        round(total_tokens / deliverables_total, 2) if deliverables_total > 0 else UNKNOWN
    )

    return {
        "baseline_id": baseline_raw.get("baseline_id", UNKNOWN),
        "frozen_at": baseline_raw.get("frozen_at", UNKNOWN),
        "window_hours": window_hours,
        "deliverables_total": deliverables_total,
        "tested_commits": tested_commits,
        "in_tokens": in_tokens,
        "out_tokens": out_tokens,
        "total_tokens": total_tokens,
        "deliverables_per_hour": deliv_per_hr,
        "tokens_per_deliverable": tok_per_deliv,
        "target_sla": dict(baseline_raw.get("target_sla_for_v2_kernel", {})),
    }


def _compute_refill_latencies(tasks: list[StageTask]) -> list[float]:
    by_id = {t.task_id: t for t in tasks}
    latencies: list[float] = []
    for task in tasks:
        if task.started_ts is None or task.enqueued_ts <= 0:
            continue
        ready_ts = float(task.enqueued_ts)
        for dep_id in task.deps:
            dep = by_id.get(dep_id)
            if dep is not None and dep.finished_ts is not None:
                ready_ts = max(ready_ts, float(dep.finished_ts))
        latencies.append(max(0.0, float(task.started_ts) - ready_ts))
    return sorted(latencies)


def _compute_slot_utilization(
    tasks: list[StageTask],
    *,
    slots: int,
    now_ts: float,
) -> dict[str, Any]:
    if slots <= 0:
        return {
            "pct": UNKNOWN,
            "busy_slot_sec": 0.0,
            "dispatchable_wall_sec": 0.0,
            "slots": slots,
        }

    by_id = {t.task_id: t for t in tasks}
    windows: list[tuple[float, float, float | None]] = []
    points: set[float] = set()

    for task in tasks:
        if task.enqueued_ts <= 0 or task.started_ts is None:
            continue
        ready_ts = float(task.enqueued_ts)
        for dep_id in task.deps:
            dep = by_id.get(dep_id)
            if dep is not None and dep.finished_ts is not None:
                ready_ts = max(ready_ts, float(dep.finished_ts))
        start_ts = max(ready_ts, float(task.started_ts))
        end_ts = float(task.finished_ts) if task.finished_ts is not None else now_ts
        if end_ts < start_ts:
            end_ts = start_ts
        windows.append((ready_ts, start_ts, end_ts))
        points.add(ready_ts)
        points.add(start_ts)
        points.add(end_ts)

    if len(points) < 2:
        return {
            "pct": UNKNOWN,
            "busy_slot_sec": 0.0,
            "dispatchable_wall_sec": 0.0,
            "slots": slots,
        }

    sorted_pts = sorted(points)
    busy_slot_sec = 0.0
    dispatchable_wall_sec = 0.0

    for idx in range(len(sorted_pts) - 1):
        a = sorted_pts[idx]
        b = sorted_pts[idx + 1]
        dt = b - a
        if dt <= 0:
            continue
        mid = 0.5 * (a + b)
        ready_or_active = 0
        running = 0
        for ready_ts, start_ts, end_ts in windows:
            assert end_ts is not None
            if ready_ts <= mid <= end_ts:
                ready_or_active += 1
            if start_ts <= mid <= end_ts:
                running += 1
        if ready_or_active > 0:
            dispatchable_wall_sec += dt
            busy_slot_sec += min(running, slots) * dt

    if dispatchable_wall_sec <= 0:
        return {
            "pct": UNKNOWN,
            "busy_slot_sec": 0.0,
            "dispatchable_wall_sec": 0.0,
            "slots": slots,
        }

    pct = min(100.0, 100.0 * busy_slot_sec / (dispatchable_wall_sec * slots))
    return {
        "pct": round(pct, 2),
        "busy_slot_sec": round(busy_slot_sec, 4),
        "dispatchable_wall_sec": round(dispatchable_wall_sec, 4),
        "slots": slots,
    }


def _is_verified_deliverable(receipt: EvidenceReceipt) -> bool:
    if not _is_real_evidence_receipt(receipt):
        return False
    validity = getattr(receipt, "evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY)
    if validity in _INVALID_VALIDITIES or validity in (
        EvidenceValidity.MISSING_INPUT,
        EvidenceValidity.MISSING_PRIMARY_METRIC,
    ):
        return False
    if receipt.process_status != ProcessStatus.EXIT_OK:
        return False
    if receipt.engineering_verdict != EngineeringVerdict.PASS:
        return False
    if receipt.audit_verdict in (AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR):
        return False
    if receipt.scientific_verdict == ScientificVerdict.REFUTED:
        return False
    return receipt.stage in {
        StageKind.CPU_GATE,
        StageKind.EXP_SCREEN,
        StageKind.EXP_CONFIRM,
        StageKind.BLIND_AUDIT,
    }


def _is_failure_receipt(receipt: EvidenceReceipt) -> bool:
    return (
        receipt.process_status != ProcessStatus.EXIT_OK
        or receipt.engineering_verdict in {EngineeringVerdict.FAIL_GATE, EngineeringVerdict.FAIL_TEST}
        or receipt.scientific_verdict == ScientificVerdict.REFUTED
        or receipt.audit_verdict == AuditVerdict.FAIL
    )


def _is_traceable_failure(receipt: EvidenceReceipt) -> bool:
    has_reason = bool(receipt.failure_reason and receipt.failure_reason.strip())
    has_artifacts = len(receipt.raw_artifact_paths) > 0
    return has_reason and has_artifacts


def _pick_int(mapping: dict[str, Any], *keys: str) -> int | None:
    for k in keys:
        val = mapping.get(k)
        if isinstance(val, (int, float)) and not isinstance(val, bool):
            return max(0, int(val))
    return None


def normalize_provider_token_usage(
    doc: dict[str, Any] | None,
    *,
    model_hint: str = "",
) -> dict[str, Any]:
    """Normalize provider-specific token payloads into clean uncached vs cache_read vs cache_write counts.

    Conventions handled explicitly:
    - Anthropic (`claude-*`, `claude_cli`, or `cache_read_input_tokens` / `cache_creation_input_tokens`):
      `input_tokens` is ALREADY uncached; `cache_read_input_tokens` is cache read;
      `cache_creation_input_tokens` is cache write (NEVER folded into `cache_read_tokens`).
    - OpenAI / Codex (`gpt-*`, `codex`, `codex_cli`, `cached_input_tokens`,
      or `prompt_tokens_details.cached_tokens`):
      `input_tokens` / `prompt_tokens` is INCLUSIVE of cached tokens, so
      `uncached_input_tokens = max(0, raw_input - cached_tokens)`.
    """
    if not isinstance(doc, dict):
        return {
            "provider_convention": "generic",
            "uncached_input_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
            "output_tokens": 0,
            "gross_input_tokens": 0,
        }

    usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else doc
    route_low = str(doc.get("route") or "").lower()
    model_low = f"{doc.get('model') or ''} {model_hint}".lower()

    raw_in = _pick_int(
        usage, "input_tokens", "inputTokens", "prompt_tokens", "promptTokens", "in_tokens", "input"
    )
    if raw_in is None:
        raw_in = _pick_int(
            doc, "input_tokens", "inputTokens", "prompt_tokens", "promptTokens", "in_tokens"
        )
    out_tok = _pick_int(
        usage,
        "output_tokens",
        "outputTokens",
        "completion_tokens",
        "candidates_tokens",
        "candidatesTokens",
        "out_tokens",
        "output",
    )
    if out_tok is None:
        out_tok = _pick_int(doc, "output_tokens", "outputTokens", "completion_tokens", "out_tokens")

    has_anthropic_keys = any(
        k in usage or k in doc
        for k in (
            "cache_read_input_tokens",
            "cacheReadInputTokens",
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
        )
    )
    is_anthropic = (
        has_anthropic_keys
        or route_low == "claude_cli"
        or any(tok in model_low for tok in ("claude", "opus", "sonnet", "haiku", "anthropic"))
    )

    pt_details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
    in_details = usage.get("input_tokens_details") if isinstance(usage.get("input_tokens_details"), dict) else {}
    has_openai_cache_keys = (
        "cached_input_tokens" in usage
        or "cached_input_tokens" in doc
        or "cached_tokens" in pt_details
        or "cached_tokens" in in_details
    )
    is_openai = (
        not is_anthropic
        and (
            has_openai_cache_keys
            or route_low == "codex_cli"
            or any(tok in model_low for tok in ("gpt-", "codex", "openai"))
        )
    )

    if is_anthropic:
        convention = "anthropic"
        cache_read = (
            _pick_int(usage, "cache_read_input_tokens", "cacheReadInputTokens", "cache_read_tokens")
            or _pick_int(doc, "cache_read_input_tokens", "cacheReadInputTokens", "cache_read_tokens")
            or 0
        )
        cache_write = (
            _pick_int(
                usage,
                "cache_creation_input_tokens",
                "cacheCreationInputTokens",
                "cache_creation_tokens",
                "cache_write_tokens",
            )
            or _pick_int(
                doc,
                "cache_creation_input_tokens",
                "cacheCreationInputTokens",
                "cache_creation_tokens",
                "cache_write_tokens",
            )
            or 0
        )
        # Fallback to modelUsage if top-level usage was zero
        mu = doc.get("modelUsage")
        if raw_in is None and out_tok is None and isinstance(mu, dict):
            m_in = m_out = m_cr = m_cw = 0
            for stats in mu.values():
                if isinstance(stats, dict):
                    m_in += int(_pick_int(stats, "inputTokens", "input_tokens") or 0)
                    m_out += int(_pick_int(stats, "outputTokens", "output_tokens") or 0)
                    m_cr += int(_pick_int(stats, "cacheReadInputTokens", "cache_read_input_tokens") or 0)
                    m_cw += int(_pick_int(stats, "cacheCreationInputTokens", "cache_creation_input_tokens") or 0)
            raw_in, out_tok, cache_read, cache_write = m_in, m_out, m_cr, m_cw
        uncached_in = max(0, int(raw_in or 0))
    elif is_openai:
        convention = "openai"
        cached_tok = (
            _pick_int(usage, "cached_input_tokens", "cache_read_tokens")
            or _pick_int(pt_details, "cached_tokens")
            or _pick_int(in_details, "cached_tokens")
            or _pick_int(doc, "cached_input_tokens", "cache_read_tokens")
            or 0
        )
        cache_write = (
            _pick_int(usage, "cache_write_tokens", "cache_creation_tokens")
            or _pick_int(doc, "cache_write_tokens", "cache_creation_tokens")
            or 0
        )
        in_val = max(0, int(raw_in or 0))
        # OpenAI reports input_tokens / prompt_tokens inclusive of cached_input_tokens
        if cached_tok > 0 and in_val >= cached_tok:
            uncached_in = in_val - cached_tok
            cache_read = cached_tok
        else:
            uncached_in = in_val
            cache_read = cached_tok
    else:
        convention = "generic"
        cache_read = (
            _pick_int(usage, "cache_read_tokens", "cacheReadTokens", "cachedContentTokenCount")
            or _pick_int(doc, "cache_read_tokens", "cacheReadTokens")
            or 0
        )
        cache_write = (
            _pick_int(
                usage,
                "cache_write_tokens",
                "cacheWriteTokens",
                "cache_creation_tokens",
                "cacheCreationTokens",
            )
            or _pick_int(doc, "cache_write_tokens", "cache_creation_tokens")
            or 0
        )
        in_val = max(0, int(raw_in or 0))
        if "cachedContentTokenCount" in usage and in_val >= cache_read > 0:
            uncached_in = in_val - cache_read
        else:
            uncached_in = in_val

    out_final = max(0, int(out_tok or 0))
    return {
        "provider_convention": convention,
        "uncached_input_tokens": uncached_in,
        "cache_read_tokens": max(0, int(cache_read)),
        "cache_write_tokens": max(0, int(cache_write)),
        "output_tokens": out_final,
        "gross_input_tokens": uncached_in + max(0, int(cache_read)) + max(0, int(cache_write)),
    }


def _extract_receipt_token_breakdown(receipt: EvidenceReceipt) -> dict[str, Any]:
    """Extract normalized `{uncached_input_tokens, cache_read_tokens, cache_write_tokens, output_tokens}`."""
    for raw_path in receipt.raw_artifact_paths:
        p = pathlib.Path(raw_path)
        if not p.name.endswith("usage.json") or not p.is_file():
            continue
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(doc, dict):
            continue
        usage = doc.get("usage") if isinstance(doc.get("usage"), dict) else doc
        has_any = any(
            _pick_int(src, "input_tokens", "inputTokens", "prompt_tokens", "in_tokens", "output_tokens", "outputTokens", "completion_tokens", "out_tokens") is not None
            for src in (usage, doc)
        ) or isinstance(doc.get("modelUsage"), dict)
        if has_any:
            return normalize_provider_token_usage(doc, model_hint=receipt.model_used or "")

    def _metric_int(mkey: str) -> int:
        raw_m = receipt.metrics.get(mkey)
        if isinstance(raw_m, dict):
            v = raw_m.get("value")
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                return max(0, int(v))
        elif isinstance(raw_m, (int, float)) and not isinstance(raw_m, bool):
            return max(0, int(raw_m))
        return 0

    cr = _metric_int("cache_read_tokens")
    cw = _metric_int("cache_write_tokens")
    uncached_in = max(0, int(receipt.in_tokens))
    out_tok = max(0, int(receipt.out_tokens))
    return {
        "provider_convention": "receipt_fallback",
        "uncached_input_tokens": uncached_in,
        "cache_read_tokens": cr,
        "cache_write_tokens": cw,
        "output_tokens": out_tok,
        "gross_input_tokens": uncached_in + cr + cw,
    }


def _extract_receipt_tokens(receipt: EvidenceReceipt) -> tuple[int, int, int]:
    """Extract `(uncached_in_tokens, out_tokens, cache_read_tokens)` from raw `*usage.json` or receipt."""
    bd = _extract_receipt_token_breakdown(receipt)
    return (
        int(bd["uncached_input_tokens"]),
        int(bd["output_tokens"]),
        int(bd["cache_read_tokens"]),
    )


def compute_benchmark_report(
    db: StateDB,
    *,
    goal_id: str | None = None,
    baseline_path: str | os.PathLike[str] | None = None,
    slots: int | None = None,
    fetch_ssh: bool = True,
    now: float | None = None,
) -> dict[str, Any]:
    """Compute machine-verified pilot KPIs and baseline comparisons from StateDB."""
    now_ts = time.time() if now is None else float(now)
    goals = [g for g in ([db.get_goal(goal_id)] if goal_id else db.list_goals()) if g is not None]
    tasks = db.list_tasks(goal_id=goal_id)
    receipts = db.list_receipts(goal_id=goal_id)
    hypotheses = db.list_hypotheses(goal_id=goal_id)

    baseline_raw, baseline_source = load_baseline(
        baseline_path, arr_home=db.arr_home, fetch_ssh=fetch_ssh
    )
    baseline = extract_baseline_metrics(baseline_raw)

    if slots is None:
        slots = max(1, sum(max(1, g.max_concurrent_branches) for g in goals)) if goals else 4

    timestamps: list[float] = []
    for t in tasks:
        if t.enqueued_ts > 0:
            timestamps.append(float(t.enqueued_ts))
        if t.started_ts is not None:
            timestamps.append(float(t.started_ts))
        if t.finished_ts is not None:
            timestamps.append(float(t.finished_ts))
        elif t.started_ts is not None:
            timestamps.append(now_ts)
    for r in receipts:
        if r.created_ts > 0:
            timestamps.append(float(r.created_ts))

    if len(timestamps) >= 2:
        pilot_window_sec = max(0.0, max(timestamps) - min(timestamps))
    elif receipts:
        pilot_window_sec = sum(max(0.0, r.wall_sec) for r in receipts)
    else:
        pilot_window_sec = 0.0
    pilot_window_hours = pilot_window_sec / 3600.0

    # Refill latency P95
    latencies = _compute_refill_latencies(tasks)
    if latencies:
        refill_p95: float | str = round(_percentile(latencies, 95.0), 4)
        refill_p50: float | str = round(_percentile(latencies, 50.0), 4)
        refill_max: float | str = round(latencies[-1], 4)
    else:
        refill_p95 = UNKNOWN
        refill_p50 = UNKNOWN
        refill_max = UNKNOWN

    # Slot utilization
    slot_util = _compute_slot_utilization(tasks, slots=slots, now_ts=now_ts)

    # Separate real research receipts from unit_test / synthetic_sim receipts
    origin_real_receipts = [r for r in receipts if _is_real_evidence_receipt(r)]
    real_receipts = [
        r
        for r in origin_real_receipts
        if getattr(r, "evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY)
        not in _INVALID_VALIDITIES
    ]
    non_real_receipts_count = len(receipts) - len(origin_real_receipts)
    hyp_ids_with_non_real_only = {
        r.hypothesis_id for r in receipts
    } - {r.hypothesis_id for r in origin_real_receipts}

    # Verified deliverables (hypotheses with confirmed or passing gate/exp evidence, plus stage receipts)
    verified_hyps: set[str] = set()
    confirmed_hyps: set[str] = set()
    passed_receipts = 0
    for r in real_receipts:
        if _is_verified_deliverable(r):
            passed_receipts += 1
            verified_hyps.add(r.hypothesis_id)
        if (
            r.engineering_verdict == EngineeringVerdict.PASS
            and r.scientific_verdict == ScientificVerdict.CONFIRMED
            and r.audit_verdict not in (AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR)
            and getattr(r, "evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY)
            not in (EvidenceValidity.MISSING_INPUT, EvidenceValidity.MISSING_PRIMARY_METRIC)
        ):
            confirmed_hyps.add(r.hypothesis_id)
    for h in hypotheses:
        if (
            h.scientific_verdict == ScientificVerdict.CONFIRMED
            and h.hypothesis_id not in hyp_ids_with_non_real_only
        ):
            confirmed_hyps.add(h.hypothesis_id)
            verified_hyps.add(h.hypothesis_id)

    verified_count = len(verified_hyps)
    deliv_per_hour_pilot: float | str = (
        round(verified_count / pilot_window_hours, 4) if pilot_window_hours > 0 else UNKNOWN
    )

    baseline_window_hours = baseline.get("window_hours")
    if (
        isinstance(baseline_window_hours, (int, float))
        and pilot_window_hours >= float(baseline_window_hours)
        and pilot_window_hours > 0
    ):
        deliv_per_hour_6h: float | str = deliv_per_hour_pilot
        full_6h_status = "COMPLETE"
    else:
        deliv_per_hour_6h = UNKNOWN
        full_6h_status = UNKNOWN

    # Token efficiency (uncached apples-to-apples vs baseline + explicit cache read/write breakdown)
    in_tokens = 0
    out_tokens = 0
    cache_read_tokens_total = 0
    cache_write_tokens_total = 0
    for r in real_receipts:
        bd = _extract_receipt_token_breakdown(r)
        in_tokens += int(bd["uncached_input_tokens"])
        out_tokens += int(bd["output_tokens"])
        cache_read_tokens_total += int(bd["cache_read_tokens"])
        cache_write_tokens_total += int(bd["cache_write_tokens"])
    total_tokens = in_tokens + out_tokens
    gross_tokens_including_cache = total_tokens + cache_read_tokens_total + cache_write_tokens_total
    tokens_per_deliv: float | str = (
        round(total_tokens / verified_count, 2) if verified_count > 0 else UNKNOWN
    )
    gross_tokens_per_deliv: float | str = (
        round(gross_tokens_including_cache / verified_count, 2)
        if verified_count > 0
        else UNKNOWN
    )

    # First-pass rate (attempt==1 in gate/screen/confirm stages)
    first_pass_stages = {StageKind.CPU_GATE, StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM}
    task_by_id = {t.task_id: t for t in tasks}
    first_attempts = [
        r
        for r in receipts
        if r.stage in first_pass_stages
        and (task_by_id.get(r.task_id) is None or task_by_id[r.task_id].attempt == 1)
    ]
    if first_attempts:
        first_passed = sum(
            1
            for r in first_attempts
            if r.process_status == ProcessStatus.EXIT_OK
            and r.engineering_verdict == EngineeringVerdict.PASS
        )
        first_pass_rate: float | str = round(first_passed / len(first_attempts), 4)
    else:
        first_passed = 0
        first_pass_rate = UNKNOWN

    baseline_tested_commits = baseline.get("tested_commits")
    baseline_first_pass_rate: str = UNKNOWN
    if (
        isinstance(first_pass_rate, float)
        and first_pass_rate > 0
        and baseline_tested_commits in (0, UNKNOWN, None)
    ):
        non_inferior_check: str = "PASS_OR_UNKNOWN_BASELINE"
    else:
        non_inferior_check = UNKNOWN

    # Failure traceability
    failure_receipts = [r for r in receipts if _is_failure_receipt(r)]
    failed_tasks_no_receipt = [
        t
        for t in tasks
        if t.status in {"FAILED", "DEAD", "TIMED_OUT", "LEASE_LOST"}
        and not any(r.task_id == t.task_id for r in receipts)
    ]
    total_failures = len(failure_receipts) + len(failed_tasks_no_receipt)
    traceable_failures = sum(1 for r in failure_receipts if _is_traceable_failure(r))
    if not receipts and not tasks:
        traceability_pct: float | str = UNKNOWN
    elif total_failures == 0:
        traceability_pct = 100.0
    else:
        traceability_pct = round(100.0 * traceable_failures / total_failures, 2)

    # Baseline deltas
    base_deliv_hr = baseline.get("deliverables_per_hour")
    if isinstance(deliv_per_hour_pilot, float) and isinstance(base_deliv_hr, (int, float)) and base_deliv_hr > 0:
        deliv_gain_pct: float | str = round(
            100.0 * (deliv_per_hour_pilot - float(base_deliv_hr)) / float(base_deliv_hr),
            2,
        )
    else:
        deliv_gain_pct = UNKNOWN

    base_tok_per_deliv = baseline.get("tokens_per_deliverable")
    if (
        isinstance(tokens_per_deliv, float)
        and isinstance(base_tok_per_deliv, (int, float))
        and base_tok_per_deliv > 0
    ):
        token_reduction_pct: float | str = round(
            100.0 * (float(base_tok_per_deliv) - tokens_per_deliv) / float(base_tok_per_deliv),
            2,
        )
    else:
        token_reduction_pct = UNKNOWN

    sla = baseline.get("target_sla", {})
    p95_max = float(sla.get("refill_latency_p95_sec_max", 60.0))
    util_min = float(sla.get("dispatchable_slot_utilization_min_pct", 80.0))
    trace_target = float(sla.get("failure_traceability_pct", 100.0))

    return {
        "generated_ts": now_ts,
        "goal_id": goal_id or "ALL",
        "baseline_source": baseline_source,
        "baseline": baseline,
        "pilot_window": {
            "duration_sec": round(pilot_window_sec, 4),
            "duration_hours": round(pilot_window_hours, 6),
            "full_6h_window_status": full_6h_status,
        },
        "metrics": {
            "refill_latency_p95_sec": {
                "value": refill_p95,
                "p50_sec": refill_p50,
                "max_sec": refill_max,
                "denominator": len(latencies),
                "target_max_sec": p95_max,
                "meets_sla": (
                    bool(refill_p95 <= p95_max) if isinstance(refill_p95, float) else UNKNOWN
                ),
            },
            "dispatchable_slot_utilization_pct": {
                "value": slot_util["pct"],
                "busy_slot_sec": slot_util["busy_slot_sec"],
                "dispatchable_wall_sec": slot_util["dispatchable_wall_sec"],
                "slots": slot_util["slots"],
                "target_min_pct": util_min,
                "meets_sla": (
                    bool(slot_util["pct"] >= util_min)
                    if isinstance(slot_util["pct"], float)
                    else UNKNOWN
                ),
            },
            "verified_deliverables_per_hour": {
                "value_pilot_window": deliv_per_hour_pilot,
                "value_full_6h": deliv_per_hour_6h,
                "verified_deliverables": verified_count,
                "confirmed_hypotheses": len(confirmed_hyps),
                "passed_stage_receipts": passed_receipts,
                "pilot_window_hours": round(pilot_window_hours, 6),
                "baseline_per_hour": base_deliv_hr,
                "gain_vs_baseline_pct_pilot_window": deliv_gain_pct,
            },
            "tokens_per_verified_deliverable": {
                "value": tokens_per_deliv,
                "total_tokens": total_tokens,
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "uncached_input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "cache_read_tokens": cache_read_tokens_total,
                "cache_write_tokens": cache_write_tokens_total,
                "cache_read_tokens_total": cache_read_tokens_total,
                "cache_write_tokens_total": cache_write_tokens_total,
                "gross_tokens_including_cache": gross_tokens_including_cache,
                "gross_tokens_including_cache_per_deliverable": gross_tokens_per_deliv,
                "denominator_verified_deliverables": verified_count,
                "baseline_tokens_per_deliverable": base_tok_per_deliv,
                "reduction_vs_baseline_pct": token_reduction_pct,
            },
            "first_pass_rate": {
                "value": first_pass_rate,
                "numerator_passed": first_passed,
                "denominator_first_attempts": len(first_attempts),
                "baseline_first_pass_rate": baseline_first_pass_rate,
                "non_inferior_check": non_inferior_check,
            },
            "failure_traceability_pct": {
                "value": traceability_pct,
                "numerator_traceable": traceable_failures,
                "denominator_failures": total_failures,
                "target_pct": trace_target,
                "meets_sla": (
                    bool(traceability_pct >= trace_target)
                    if isinstance(traceability_pct, float)
                    else UNKNOWN
                ),
            },
        },
        "counts": {
            "goals": len(goals),
            "hypotheses": len(hypotheses),
            "tasks": len(tasks),
            "tasks_done": sum(1 for t in tasks if t.status in TERMINAL_TASK_SUCCESS),
            "receipts": len(receipts),
            "real_receipts": len(real_receipts),
            "non_real_receipts_excluded": non_real_receipts_count,
        },
        "v3_metrics": compute_v3_research_metrics(
            db, goal_id=goal_id, slots=slots, now=now_ts
        ),
    }


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _compute_idle_with_ready_ratio(
    tasks: list[StageTask],
    *,
    slots: int,
    now_ts: float,
) -> float | str:
    if slots <= 0:
        return UNKNOWN
    by_id = {t.task_id: t for t in tasks}
    windows: list[tuple[float, float | None, float | None]] = []
    points: set[float] = set()
    for task in tasks:
        if task.enqueued_ts <= 0:
            continue
        ready_ts = float(task.enqueued_ts)
        for dep_id in task.deps:
            dep = by_id.get(dep_id)
            if dep is not None and dep.finished_ts is not None:
                ready_ts = max(ready_ts, float(dep.finished_ts))
            elif dep is not None and dep.finished_ts is None:
                ready_ts = max(ready_ts, now_ts)
        start_ts = float(task.started_ts) if task.started_ts is not None else None
        if start_ts is None and task.status not in ("READY", "DISPATCHED", "RUNNING"):
            continue
        if start_ts is not None:
            start_ts = max(ready_ts, start_ts)
            end_ts = float(task.finished_ts) if task.finished_ts is not None else now_ts
            if end_ts < start_ts:
                end_ts = start_ts
        else:
            end_ts = now_ts
        if end_ts <= ready_ts:
            continue
        windows.append((ready_ts, start_ts, end_ts))
        points.add(ready_ts)
        if start_ts is not None:
            points.add(start_ts)
        points.add(end_ts)

    if len(points) < 2:
        return UNKNOWN
    sorted_pts = sorted(points)
    active_or_ready_sec = 0.0
    idle_with_ready_sec = 0.0
    for idx in range(len(sorted_pts) - 1):
        a = sorted_pts[idx]
        b = sorted_pts[idx + 1]
        dt = b - a
        if dt <= 0:
            continue
        mid = 0.5 * (a + b)
        has_ready_waiting = False
        running = 0
        ready_or_active = 0
        for ready_ts, start_ts, end_ts in windows:
            assert end_ts is not None
            if ready_ts <= mid <= end_ts:
                ready_or_active += 1
            if start_ts is None:
                if ready_ts <= mid <= end_ts:
                    has_ready_waiting = True
            else:
                if ready_ts <= mid < start_ts:
                    has_ready_waiting = True
                if start_ts <= mid <= end_ts:
                    running += 1
        if ready_or_active > 0:
            active_or_ready_sec += dt
            if has_ready_waiting and running < slots:
                idle_with_ready_sec += dt
    if active_or_ready_sec <= 0:
        return UNKNOWN
    return round(min(100.0, 100.0 * idle_with_ready_sec / active_or_ready_sec), 2)


def compute_v2_research_metrics(
    db: StateDB,
    *,
    goal_id: str | None = None,
    slots: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Compute Section 5 & Multi-Goal v2 research telemetry directly from raw SQLite rows."""
    now_ts = time.time() if now is None else float(now)
    with db._lock:
        conn = db._conn
        goal_rows = (
            conn.execute(
                "SELECT goal_id, status, payload_json FROM goals WHERE goal_id = ?",
                (goal_id,),
            ).fetchall()
            if goal_id is not None
            else conn.execute("SELECT goal_id, status, payload_json FROM goals").fetchall()
        )
        task_rows = (
            conn.execute(
                "SELECT payload_json FROM tasks WHERE goal_id = ? ORDER BY enqueued_ts ASC, task_id ASC",
                (goal_id,),
            ).fetchall()
            if goal_id is not None
            else conn.execute(
                "SELECT payload_json FROM tasks ORDER BY enqueued_ts ASC, task_id ASC"
            ).fetchall()
        )
        receipt_rows = (
            conn.execute(
                "SELECT payload_json FROM evidence_receipts WHERE goal_id = ? ORDER BY created_ts ASC, receipt_id ASC",
                (goal_id,),
            ).fetchall()
            if goal_id is not None
            else conn.execute(
                "SELECT payload_json FROM evidence_receipts ORDER BY created_ts ASC, receipt_id ASC"
            ).fetchall()
        )
        event_rows = (
            conn.execute(
                "SELECT event_type, goal_id, hypothesis_id, task_id, ts, payload_json FROM events WHERE goal_id = ? ORDER BY seq ASC",
                (goal_id,),
            ).fetchall()
            if goal_id is not None
            else conn.execute(
                "SELECT event_type, goal_id, hypothesis_id, task_id, ts, payload_json FROM events ORDER BY seq ASC"
            ).fetchall()
        )

        intervention_rows: list[sqlite3.Row] = []
        if _table_exists(conn, "interventions"):
            intervention_rows = (
                conn.execute(
                    "SELECT scope, action, goal_id, payload_json, result_json, created_ts FROM interventions WHERE goal_id = ? OR goal_id IS NULL",
                    (goal_id,),
                ).fetchall()
                if goal_id is not None
                else conn.execute(
                    "SELECT scope, action, goal_id, payload_json, result_json, created_ts FROM interventions"
                ).fetchall()
            )

        decision_rows: list[sqlite3.Row] = []
        if _table_exists(conn, "decision_requests"):
            decision_rows = (
                conn.execute(
                    "SELECT req_id, goal_id, tier, status FROM decision_requests WHERE goal_id = ?",
                    (goal_id,),
                ).fetchall()
                if goal_id is not None
                else conn.execute(
                    "SELECT req_id, goal_id, tier, status FROM decision_requests"
                ).fetchall()
            )

        workflow_rows: list[sqlite3.Row] = []
        if _table_exists(conn, "workflows"):
            workflow_rows = (
                conn.execute(
                    "SELECT workflow_id, goal_id, primitive, status FROM workflows WHERE goal_id = ?",
                    (goal_id,),
                ).fetchall()
                if goal_id is not None
                else conn.execute(
                    "SELECT workflow_id, goal_id, primitive, status FROM workflows"
                ).fetchall()
            )

        eval_ledger_rows: list[sqlite3.Row] = []
        if _table_exists(conn, "workload_eval_ledger"):
            eval_ledger_rows = conn.execute(
                """
                SELECT workload_class, mode, provider_pairing, holdout_passed,
                       defect_escaped, uncached_tokens, cache_read_tokens, wall_sec
                FROM workload_eval_ledger
                WHERE is_real_llm = 1
                """
            ).fetchall()

    tasks = [StageTask.from_dict(json.loads(r["payload_json"])) for r in task_rows]
    receipts = [EvidenceReceipt.from_dict(json.loads(r["payload_json"])) for r in receipt_rows]
    task_by_id = {t.task_id: t for t in tasks}

    goals_by_status: dict[str, int] = {}
    total_goal_slots = 0
    for gr in goal_rows:
        st = str(gr["status"])
        goals_by_status[st] = goals_by_status.get(st, 0) + 1
        g_payload = json.loads(gr["payload_json"])
        if st == "ACTIVE":
            total_goal_slots += max(1, int(g_payload.get("max_concurrent_branches", 4)))
    if slots is None:
        slots = max(1, total_goal_slots) if total_goal_slots > 0 else 4

    # 1. Separate receipts by evidence_origin (never count unit_test / synthetic_sim as real research yield)
    by_origin: dict[str, int] = {}
    real_receipts: list[EvidenceReceipt] = []
    non_real_receipts: list[EvidenceReceipt] = []
    for r in receipts:
        origin = (getattr(r, "evidence_origin", "real_research_goal") or "unknown").strip()
        by_origin[origin] = by_origin.get(origin, 0) + 1
        if _is_real_evidence_receipt(r):
            real_receipts.append(r)
        else:
            non_real_receipts.append(r)

    # 2. Scientific conclusions (strictly from real_receipts with VALID_EMPIRICAL)
    positive_hyp_ids: set[str] = set()
    negative_hyp_ids: set[str] = set()
    valid_negative_only_hyps: set[str] = set()
    refuted_empirical_hyps: set[str] = set()

    for r in real_receipts:
        validity = getattr(r, "evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY)
        if validity != EvidenceValidity.VALID_EMPIRICAL:
            continue
        if r.stage not in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
            continue
        if r.process_status != ProcessStatus.EXIT_OK or r.engineering_verdict != EngineeringVerdict.PASS:
            continue
        if r.audit_verdict in (AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR):
            continue
        if r.scientific_verdict == ScientificVerdict.CONFIRMED:
            positive_hyp_ids.add(r.hypothesis_id)
        elif r.scientific_verdict == ScientificVerdict.VALID_NEGATIVE:
            negative_hyp_ids.add(r.hypothesis_id)
            valid_negative_only_hyps.add(r.hypothesis_id)
        elif r.scientific_verdict == ScientificVerdict.REFUTED:
            negative_hyp_ids.add(r.hypothesis_id)
            refuted_empirical_hyps.add(r.hypothesis_id)

    negative_hyp_ids -= positive_hyp_ids
    verified_positive_conclusions = len(positive_hyp_ids)
    valid_negative_conclusions = len(negative_hyp_ids)
    total_conclusions = verified_positive_conclusions + valid_negative_conclusions

    # 3. Window and token accounting over real receipts + tasks
    timestamps: list[float] = []
    for t in tasks:
        if t.enqueued_ts > 0:
            timestamps.append(float(t.enqueued_ts))
        if t.started_ts is not None:
            timestamps.append(float(t.started_ts))
        if t.finished_ts is not None:
            timestamps.append(float(t.finished_ts))
    for r in real_receipts:
        if r.created_ts > 0:
            timestamps.append(float(r.created_ts))
    if len(timestamps) >= 2:
        window_sec = max(0.0, max(timestamps) - min(timestamps))
    elif real_receipts:
        window_sec = sum(max(0.0, float(r.wall_sec)) for r in real_receipts)
    else:
        window_sec = 0.0
    window_hours = window_sec / 3600.0

    real_uncached_in = 0
    real_out = 0
    real_cache = 0
    by_model_and_primitive: dict[str, dict[str, Any]] = {}
    wasted_tokens = 0
    stale_gv = 0
    human_inv = 0
    missing_metric_or_input = 0
    invalid_synth = 0
    parse_err_audits = 0

    real_cache_write = 0
    for r in real_receipts:
        bd = _extract_receipt_token_breakdown(r)
        r_in = int(bd["uncached_input_tokens"])
        r_out = int(bd["output_tokens"])
        r_cache = int(bd["cache_read_tokens"])
        r_cache_write = int(bd["cache_write_tokens"])
        real_uncached_in += r_in
        real_out += r_out
        real_cache += r_cache
        real_cache_write += r_cache_write
        task = task_by_id.get(r.task_id)
        primitive = getattr(task, "workflow_primitive", "direct_hq") if task is not None else "direct_hq"
        key = f"{r.model_used}::{primitive}"
        bucket = by_model_and_primitive.setdefault(
            key,
            {
                "model_used": r.model_used,
                "primitive": primitive,
                "receipts": 0,
                "uncached_in_tokens": 0,
                "uncached_input_tokens": 0,
                "out_tokens": 0,
                "output_tokens": 0,
                "uncached_total_tokens": 0,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
                "wall_sec": 0.0,
                "positive_conclusions": 0,
                "valid_negative_conclusions": 0,
                "failures": 0,
            },
        )
        bucket["receipts"] += 1
        bucket["uncached_in_tokens"] += r_in
        bucket["uncached_input_tokens"] += r_in
        bucket["out_tokens"] += r_out
        bucket["output_tokens"] += r_out
        bucket["uncached_total_tokens"] += r_in + r_out
        bucket["cache_read_tokens"] += r_cache
        bucket["cache_write_tokens"] += r_cache_write
        bucket["wall_sec"] = round(float(bucket["wall_sec"]) + float(r.wall_sec), 4)

        validity = getattr(r, "evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY)
        if validity == EvidenceValidity.STALE_GOAL_VERSION:
            stale_gv += 1
        elif validity == EvidenceValidity.HUMAN_INVALIDATED:
            human_inv += 1
        elif validity in (EvidenceValidity.MISSING_INPUT, EvidenceValidity.MISSING_PRIMARY_METRIC):
            missing_metric_or_input += 1
        elif validity == EvidenceValidity.INVALID_SYNTHETIC:
            invalid_synth += 1
        if r.audit_verdict == AuditVerdict.PARSE_ERROR:
            parse_err_audits += 1

        if (
            validity == EvidenceValidity.VALID_EMPIRICAL
            and r.stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM)
            and r.process_status == ProcessStatus.EXIT_OK
            and r.engineering_verdict == EngineeringVerdict.PASS
        ):
            if r.scientific_verdict == ScientificVerdict.CONFIRMED:
                bucket["positive_conclusions"] += 1
            elif r.scientific_verdict in (ScientificVerdict.VALID_NEGATIVE, ScientificVerdict.REFUTED):
                bucket["valid_negative_conclusions"] += 1

        is_waste = (
            r.process_status != ProcessStatus.EXIT_OK
            or r.engineering_verdict in (EngineeringVerdict.FAIL_GATE, EngineeringVerdict.FAIL_TEST)
            or r.audit_verdict in (AuditVerdict.FAIL, AuditVerdict.PARSE_ERROR)
            or validity in _INVALID_VALIDITIES
        )
        if is_waste:
            bucket["failures"] += 1
            wasted_tokens += r_in + r_out

    for r in non_real_receipts:
        validity = getattr(r, "evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY)
        if validity == EvidenceValidity.INVALID_SYNTHETIC:
            invalid_synth += 1

    real_uncached_total = real_uncached_in + real_out
    scientific_yield_per_hour: float | str = (
        round(total_conclusions / window_hours, 4)
        if window_hours > 0 and (real_receipts or tasks)
        else UNKNOWN
    )
    scientific_yield_per_1m_tokens: float | str = (
        round(total_conclusions / (real_uncached_total / 1_000_000.0), 4)
        if real_uncached_total > 0
        else UNKNOWN
    )

    # 4. Successor dispatch latency & cross-goal refill latency
    succ_lats = _compute_refill_latencies(tasks)
    successor_dispatch_latency_sec = (
        {
            "p50": round(_percentile(succ_lats, 50.0), 4),
            "p95": round(_percentile(succ_lats, 95.0), 4),
            "max": round(succ_lats[-1], 4),
            "count": len(succ_lats),
        }
        if succ_lats
        else {"p50": UNKNOWN, "p95": UNKNOWN, "max": UNKNOWN, "count": 0}
    )

    cross_goal_lats: list[float] = []
    interactive_ms: list[float] = []
    scheduler_llm_tokens = 0
    leases_expired = 0
    tasks_recovered = 0
    owner_escalations = 0

    for er in event_rows:
        etype = str(er["event_type"])
        epayload = json.loads(er["payload_json"])
        if etype == "CROSS_GOAL_REFILL" and isinstance(epayload.get("latency_sec"), (int, float)):
            cross_goal_lats.append(max(0.0, float(epayload["latency_sec"])))
        if etype in ("INTERACTIVE_QUERY", "INTERACTIVE_COMMAND", "FOREGROUND_PING") or (
            "interactive_latency_ms" in epayload
        ):
            raw_ms = epayload.get("interactive_latency_ms", epayload.get("latency_ms"))
            if isinstance(raw_ms, (int, float)) and not isinstance(raw_ms, bool):
                interactive_ms.append(max(0.0, float(raw_ms)))
        if etype == "SCHEDULER_LLM_CALL":
            scheduler_llm_tokens += int(epayload.get("in_tokens", 0)) + int(epayload.get("out_tokens", 0))
        elif etype == "LEASE_EXPIRED":
            leases_expired += 1
        elif etype in ("TASK_RECOVERED", "TASK_REQUEUED"):
            tasks_recovered += 1
        elif etype == "OWNER_ESCALATION":
            owner_escalations += 1

    # Also derive cross-goal slot handoffs from task timestamps if no explicit CROSS_GOAL_REFILL events logged
    if not cross_goal_lats and goal_id is None:
        finished_tasks = [t for t in tasks if t.finished_ts is not None]
        started_tasks = [t for t in tasks if t.started_ts is not None and t.enqueued_ts > 0]
        ready_ts_map: dict[str, float] = {}
        for t in started_tasks:
            r_ts = float(t.enqueued_ts)
            for dep_id in t.deps:
                dep = task_by_id.get(dep_id)
                if dep is not None and dep.finished_ts is not None:
                    r_ts = max(r_ts, float(dep.finished_ts))
            ready_ts_map[t.task_id] = r_ts
        for t_next in started_tasks:
            assert t_next.started_ts is not None
            r_ts = ready_ts_map[t_next.task_id]
            best_gap: float | None = None
            for t_prev in finished_tasks:
                assert t_prev.finished_ts is not None
                if t_prev.goal_id == t_next.goal_id:
                    continue
                if r_ts <= float(t_prev.finished_ts) <= float(t_next.started_ts):
                    gap = float(t_next.started_ts) - float(t_prev.finished_ts)
                    if gap <= 60.0 and (best_gap is None or gap < best_gap):
                        best_gap = gap
            if best_gap is not None:
                cross_goal_lats.append(best_gap)

    cross_goal_lats.sort()
    cross_goal_refill_events = len(cross_goal_lats)
    cross_goal_refill_p95_sec: float | str = (
        round(_percentile(cross_goal_lats, 95.0), 4) if cross_goal_lats else UNKNOWN
    )

    idle_with_ready_ratio_pct = _compute_idle_with_ready_ratio(
        tasks, slots=slots, now_ts=now_ts
    )

    # Human interventions + discarded decisions
    int_by_scope: dict[str, int] = {}
    int_by_action: dict[str, int] = {}
    for ir in intervention_rows:
        sc = str(ir["scope"])
        act = str(ir["action"])
        int_by_scope[sc] = int_by_scope.get(sc, 0) + 1
        int_by_action[act] = int_by_action.get(act, 0) + 1
        res_doc = json.loads(ir["result_json"])
        if isinstance(res_doc.get("latency_ms"), (int, float)) and not isinstance(res_doc.get("latency_ms"), bool):
            interactive_ms.append(max(0.0, float(res_doc["latency_ms"])))

    interactive_ms.sort()
    interactive_response_latency_ms = (
        {
            "p50": round(_percentile(interactive_ms, 50.0), 3),
            "p95": round(_percentile(interactive_ms, 95.0), 3),
            "max": round(interactive_ms[-1], 3),
            "count": len(interactive_ms),
        }
        if interactive_ms
        else {"p50": UNKNOWN, "p95": UNKNOWN, "max": UNKNOWN, "count": 0}
    )

    dec_total = len(decision_rows)
    dec_resolved = sum(1 for d in decision_rows if str(d["status"]) == "RESOLVED")
    dec_discarded = sum(
        1 for d in decision_rows if str(d["status"]) == "DISCARDED_BY_NEWER_HUMAN_INTERVENTION"
    )

    wf_by_status: dict[str, int] = {}
    for wr in workflow_rows:
        ws = str(wr["status"])
        wf_by_status[ws] = wf_by_status.get(ws, 0) + 1

    eval_ledger_summary: dict[str, dict[str, Any]] = {}
    for er in eval_ledger_rows:
        k = f"{er['provider_pairing']}::{er['mode']}"
        eb = eval_ledger_summary.setdefault(
            k,
            {"provider_pairing": er["provider_pairing"], "mode": er["mode"], "runs": 0, "passed": 0, "uncached_tokens": 0},
        )
        eb["runs"] += 1
        eb["passed"] += int(bool(er["holdout_passed"]))
        eb["uncached_tokens"] += int(er["uncached_tokens"])

    failed_tasks = sum(1 for t in tasks if t.status in ("FAILED", "TIMED_OUT", "LEASE_LOST"))
    cancelled_tasks = sum(1 for t in tasks if t.status == "CANCELLED")
    rework_retries = sum(1 for t in tasks if int(t.attempt) > 1)

    return {
        "generated_ts": now_ts,
        "goal_id": goal_id or "ALL",
        "window_hours": round(window_hours, 6),
        "evidence_origin_breakdown": {
            "real_receipts": len(real_receipts),
            "non_real_receipts": len(non_real_receipts),
            "by_origin": by_origin,
        },
        "verified_positive_conclusions": verified_positive_conclusions,
        "valid_negative_conclusions": valid_negative_conclusions,
        "valid_negative_breakdown": {
            "valid_negative": len(valid_negative_only_hyps - positive_hyp_ids),
            "refuted_empirical": len(refuted_empirical_hyps - positive_hyp_ids),
        },
        "scientific_yield_total": total_conclusions,
        "scientific_yield_per_hour": scientific_yield_per_hour,
        "scientific_yield_per_1m_tokens": scientific_yield_per_1m_tokens,
        "tokens_real": {
            "uncached_in_tokens": real_uncached_in,
            "uncached_input_tokens": real_uncached_in,
            "out_tokens": real_out,
            "output_tokens": real_out,
            "uncached_total_tokens": real_uncached_total,
            "cache_read_tokens": real_cache,
            "cache_write_tokens": real_cache_write,
        },
        "by_model_and_primitive": by_model_and_primitive,
        "eval_ledger_by_model_and_mode": eval_ledger_summary,
        "waste_and_rework_breakdown": {
            "failed_tasks": failed_tasks,
            "cancelled_tasks": cancelled_tasks,
            "rework_retries": rework_retries,
            "stale_goal_version_receipts": stale_gv,
            "human_invalidated_receipts": human_inv,
            "missing_metric_or_input_receipts": missing_metric_or_input,
            "invalid_synthetic_receipts": invalid_synth,
            "parse_error_audits": parse_err_audits,
            "wasted_uncached_tokens": wasted_tokens,
            "discarded_decisions": dec_discarded,
        },
        "successor_dispatch_latency_sec": successor_dispatch_latency_sec,
        "cross_goal_refill_events": cross_goal_refill_events,
        "cross_goal_refill_p95_sec": cross_goal_refill_p95_sec,
        "idle_with_ready_ratio_pct": idle_with_ready_ratio_pct,
        "interactive_response_latency_ms": interactive_response_latency_ms,
        "human_interventions_summary": {
            "total": len(intervention_rows),
            "by_scope": int_by_scope,
            "by_action": int_by_action,
            "discarded_stale_decisions": dec_discarded,
        },
        "scheduler_llm_tokens": scheduler_llm_tokens,
        "autonomy_and_recovery_summary": {
            "leases_expired": leases_expired,
            "tasks_recovered": tasks_recovered,
            "owner_escalations": owner_escalations,
            "decision_requests_total": dec_total,
            "decision_requests_resolved": dec_resolved,
            "decision_requests_discarded": dec_discarded,
            "workflows_total": len(workflow_rows),
            "workflows_by_status": wf_by_status,
            "goals_by_status": goals_by_status,
        },
    }


def compute_per_goal_empirical_efficiency(
    db: StateDB,
    *,
    goal_id: str | None = None,
    goal_records: list[dict[str, Any]] | None = None,
    receipts: list[EvidenceReceipt] | None = None,
    reviews: list[Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-active-goal experiment velocity, pre-F1 rejections, Opus share, and framework overhead token share."""
    records = goal_records if goal_records is not None else db.list_goal_records()
    selected: list[tuple[ResearchGoal, str]] = []
    for rec in records:
        g_obj: ResearchGoal = rec["goal"]
        g_status = str(rec.get("status") or "ACTIVE")
        if goal_id is not None:
            if g_obj.goal_id == goal_id:
                selected.append((g_obj, g_status))
        elif g_status == "ACTIVE":
            selected.append((g_obj, g_status))
    if not selected:
        return {}

    target_gids = {g_obj.goal_id for g_obj, _ in selected}
    eff_receipts = receipts if receipts is not None else db.list_receipts(goal_id=goal_id)
    eff_reviews = reviews if reviews is not None else db.list_review_receipts(goal_id=goal_id)

    dec_rows: list[sqlite3.Row] = []
    advisory_r1_wfs: set[str] = set()
    advisory_r1_tasks: set[str] = set()
    with db._lock:
        if _table_exists(db._conn, "decision_requests"):
            dec_rows = (
                db._conn.execute(
                    "SELECT goal_id, model_spec, status, resolution_json FROM decision_requests WHERE goal_id = ?",
                    (goal_id,),
                ).fetchall()
                if goal_id is not None
                else db._conn.execute(
                    "SELECT goal_id, model_spec, status, resolution_json FROM decision_requests"
                ).fetchall()
            )
        if _table_exists(db._conn, "events"):
            ev_rows = (
                db._conn.execute(
                    "SELECT task_id, payload_json FROM events WHERE event_type = 'DESIGN_REVIEW_ADVISORY_CONFOUND' AND goal_id = ?",
                    (goal_id,),
                ).fetchall()
                if goal_id is not None
                else db._conn.execute(
                    "SELECT task_id, payload_json FROM events WHERE event_type = 'DESIGN_REVIEW_ADVISORY_CONFOUND'"
                ).fetchall()
            )
            for er in ev_rows:
                if er["task_id"]:
                    advisory_r1_tasks.add(str(er["task_id"]))
                try:
                    ep = json.loads(er["payload_json"] or "{}")
                    if isinstance(ep, dict) and ep.get("workflow_id"):
                        advisory_r1_wfs.add(str(ep["workflow_id"]))
                except (TypeError, json.JSONDecodeError):
                    pass
        if _table_exists(db._conn, "workflows"):
            wf_rows = (
                db._conn.execute(
                    "SELECT workflow_id, payload_json FROM workflows WHERE goal_id = ?",
                    (goal_id,),
                ).fetchall()
                if goal_id is not None
                else db._conn.execute("SELECT workflow_id, payload_json FROM workflows").fetchall()
            )
            for wr in wf_rows:
                try:
                    wp = json.loads(wr["payload_json"] or "{}")
                    if isinstance(wp, dict) and isinstance(wp.get("state"), dict) and wp["state"].get("design_review_advisory"):
                        advisory_r1_wfs.add(str(wr["workflow_id"]))
                except (TypeError, json.JSONDecodeError):
                    pass

    goals_by_id = {g_obj.goal_id: g_obj for g_obj, _ in selected}
    acc: dict[str, dict[str, Any]] = {
        g_obj.goal_id: {
            "goal_id": g_obj.goal_id,
            "project": g_obj.project,
            "status": g_status,
            "real_experiments_run": 0,
            "real_experiment_attempts": 0,
            "exp_screen_runs": 0,
            "exp_confirm_runs": 0,
            "pos_hyps": set(),
            "neg_hyps": set(),
            "pre_experiment_rejections": 0,
            "r1_advisory_critiques": 0,
            "opus_tokens": 0,
            "decision_tokens": 0,
            "pre_f1_review_tokens": 0,
            "framework_overhead_tokens": 0,
            "execution_tokens": 0,
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
        for g_obj, g_status in selected
    }

    seen_receipt_task_ids: set[str] = set()
    _pre_f1_stages = frozenset({StageKind.PROPOSE, StageKind.DESIGN_REVIEW, StageKind.IMPL_REVIEW})
    _other_fw_stages = frozenset({StageKind.RESULT_REVIEW, StageKind.RECONCILE})

    for r in eff_receipts:
        if r.goal_id not in target_gids or not _is_real_evidence_receipt(r):
            continue
        if r.task_id:
            seen_receipt_task_ids.add(r.task_id)
        b = acc[r.goal_id]
        validity = getattr(r, "evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY)
        if r.stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM) and validity != EvidenceValidity.INVALID_SYNTHETIC:
            b["real_experiment_attempts"] += 1
            if r.process_status == ProcessStatus.EXIT_OK:
                b["real_experiments_run"] += 1
                if r.stage == StageKind.EXP_SCREEN:
                    b["exp_screen_runs"] += 1
                else:
                    b["exp_confirm_runs"] += 1

        g_obj = goals_by_id[r.goal_id]
        aligned_g = ResearchGoal.from_dict({**g_obj.to_dict(), "version": int(r.goal_version)})
        if is_valid_empirical_outcome(r, aligned_g):
            if r.scientific_verdict == ScientificVerdict.CONFIRMED:
                b["pos_hyps"].add(r.hypothesis_id)
            elif r.scientific_verdict in (ScientificVerdict.VALID_NEGATIVE, ScientificVerdict.REFUTED):
                b["neg_hyps"].add(r.hypothesis_id)

        bd = _extract_receipt_token_breakdown(r)
        r_in = int(bd["uncached_input_tokens"])
        r_out = int(bd["output_tokens"])
        b["cache_read_tokens"] += int(bd["cache_read_tokens"])
        b["cache_write_tokens"] += int(bd["cache_write_tokens"])
        tok = r_in + r_out
        if "opus" in (r.model_used or "").lower():
            b["opus_tokens"] += tok
        if r.stage in _pre_f1_stages:
            b["pre_f1_review_tokens"] += tok
            b["framework_overhead_tokens"] += tok
        elif r.stage in _other_fw_stages:
            b["framework_overhead_tokens"] += tok
        else:
            b["execution_tokens"] += tok

    for rv in eff_reviews:
        if rv.goal_id not in target_gids:
            continue
        b = acc[rv.goal_id]
        if rv.review_kind == ReviewKind.DESIGN_REVIEW and rv.verdict == AuditVerdict.FAIL:
            is_advisory = bool(
                (rv.workflow_id and rv.workflow_id in advisory_r1_wfs)
                or (rv.task_id and rv.task_id in advisory_r1_tasks)
            )
            if is_advisory:
                b["r1_advisory_critiques"] += 1
            else:
                b["pre_experiment_rejections"] += 1
        # Deduplicate review_records that already correspond to an evidence_receipt task_id
        if rv.task_id and rv.task_id in seen_receipt_task_ids:
            continue
        tok = max(0, int(rv.in_tokens)) + max(0, int(rv.out_tokens))
        rmod = str((rv.reviewer_route or {}).get("canonical_model") or "").lower()
        if "opus" in rmod:
            b["opus_tokens"] += tok
        if rv.review_kind in (ReviewKind.DESIGN_REVIEW, ReviewKind.IMPLEMENTATION_REVIEW):
            b["pre_f1_review_tokens"] += tok
        b["framework_overhead_tokens"] += tok

    for dr in dec_rows:
        gid = str(dr["goal_id"] or "")
        if gid not in target_gids:
            continue
        b = acc[gid]
        try:
            res_doc = json.loads(dr["resolution_json"] or "{}")
            if not isinstance(res_doc, dict):
                res_doc = {}
        except (TypeError, json.JSONDecodeError):
            res_doc = {}
        d_in = res_doc.get("in_tokens", 0)
        d_out = res_doc.get("out_tokens", 0)
        tok = (
            max(0, int(d_in)) if isinstance(d_in, (int, float)) and not isinstance(d_in, bool) else 0
        ) + (
            max(0, int(d_out)) if isinstance(d_out, (int, float)) and not isinstance(d_out, bool) else 0
        )
        d_mod = str(res_doc.get("model_used") or dr["model_spec"] or "").lower()
        if "opus" in d_mod:
            b["opus_tokens"] += tok
        b["decision_tokens"] += tok
        b["framework_overhead_tokens"] += tok

    out: dict[str, dict[str, Any]] = {}
    for g_obj, _ in selected:
        b = acc[g_obj.goal_id]
        pos_cnt = len(b["pos_hyps"])
        neg_cnt = len(b["neg_hyps"] - b["pos_hyps"])
        total_tok = int(b["framework_overhead_tokens"]) + int(b["execution_tokens"])
        opus_share: float | str = (
            round(100.0 * int(b["opus_tokens"]) / total_tok, 2) if total_tok > 0 else UNKNOWN
        )
        overhead_share: float | str = (
            round(100.0 * int(b["framework_overhead_tokens"]) / total_tok, 2)
            if total_tok > 0
            else UNKNOWN
        )
        out[g_obj.goal_id] = {
            "goal_id": g_obj.goal_id,
            "project": b["project"],
            "status": b["status"],
            "real_experiments_run": int(b["real_experiments_run"]),
            "real_experiment_attempts": int(b["real_experiment_attempts"]),
            "exp_screen_runs": int(b["exp_screen_runs"]),
            "exp_confirm_runs": int(b["exp_confirm_runs"]),
            "verified_empirical_conclusions": pos_cnt + neg_cnt,
            "verified_positive_conclusions": pos_cnt,
            "valid_negative_conclusions": neg_cnt,
            "pre_experiment_rejections": int(b["pre_experiment_rejections"]),
            "r1_advisory_critiques": int(b["r1_advisory_critiques"]),
            "opus_tokens": int(b["opus_tokens"]),
            "opus_token_share_pct": opus_share,
            "decision_tokens": int(b["decision_tokens"]),
            "pre_f1_review_tokens": int(b["pre_f1_review_tokens"]),
            "framework_overhead_tokens": int(b["framework_overhead_tokens"]),
            "execution_tokens": int(b["execution_tokens"]),
            "cache_read_tokens": int(b["cache_read_tokens"]),
            "cache_write_tokens": int(b["cache_write_tokens"]),
            "total_uncached_tokens": total_tok,
            "framework_overhead_token_share_pct": overhead_share,
        }
    return out


def compute_v3_research_metrics(
    db: StateDB,
    *,
    goal_id: str | None = None,
    slots: int | None = None,
    now: float | None = None,
) -> dict[str, Any]:
    """Compute V3 scientific & efficiency KPIs: positive/negative yield, uncertainty reduction, high-cost model utilization, code/plan rework, waste ratio, multi-Goal efficiency, and HITL latency."""
    now_ts = time.time() if now is None else float(now)
    v2 = compute_v2_research_metrics(db, goal_id=goal_id, slots=slots, now=now_ts)
    goals = (
        [g for g in [db.get_goal(goal_id)] if g is not None]
        if goal_id
        else db.list_goals()
    )
    goals_by_id: dict[str, ResearchGoal] = {g.goal_id: g for g in goals}
    hyps = db.list_hypotheses(goal_id=goal_id)
    tasks = db.list_tasks(goal_id=goal_id)
    receipts = db.list_receipts(goal_id=goal_id)
    reviews = db.list_review_receipts(goal_id=goal_id)

    # 1. Strict V3 empirical conclusions (`is_valid_empirical_outcome`)
    v3_pos_hyps: set[str] = set()
    v3_neg_hyps: set[str] = set()
    failure_domains: dict[str, int] = {}
    high_cost_models = ("claude-opus-5-5", "gpt-5-pro", "webchat-gpt-xhigh")
    high_cost_calls = 0
    high_cost_effective_calls = 0
    high_cost_tokens = 0

    for r in receipts:
        if not _is_real_evidence_receipt(r):
            continue
        g = goals_by_id.get(r.goal_id)
        eff_dom = (
            r.failure_domain
            if r.failure_domain != FailureDomain.NONE
            else classify_failure_domain(r, goal=g)
        )
        if eff_dom != FailureDomain.NONE:
            failure_domains[eff_dom.value] = failure_domains.get(eff_dom.value, 0) + 1

        if g is not None:
            aligned_g = ResearchGoal.from_dict({**g.to_dict(), "version": int(r.goal_version)})
            if is_valid_empirical_outcome(r, aligned_g):
                if r.scientific_verdict == ScientificVerdict.CONFIRMED:
                    v3_pos_hyps.add(r.hypothesis_id)
                elif r.scientific_verdict in (
                    ScientificVerdict.VALID_NEGATIVE,
                    ScientificVerdict.REFUTED,
                ):
                    v3_neg_hyps.add(r.hypothesis_id)

        mod_low = (r.model_used or "").lower()
        if any(hm in mod_low for hm in high_cost_models):
            r_in, r_out, _ = _extract_receipt_tokens(r)
            high_cost_calls += 1
            high_cost_tokens += r_in + r_out
            if (
                r.process_status == ProcessStatus.EXIT_OK
                and r.engineering_verdict == EngineeringVerdict.PASS
                and r.audit_verdict != AuditVerdict.PARSE_ERROR
            ):
                high_cost_effective_calls += 1

    for rv in reviews:
        rmod = str((rv.reviewer_route or {}).get("canonical_model") or "").lower()
        if any(hm in rmod for hm in high_cost_models):
            high_cost_calls += 1
            high_cost_tokens += int(rv.in_tokens) + int(rv.out_tokens)
            if rv.process_status == ProcessStatus.EXIT_OK and rv.verdict in (
                AuditVerdict.PASS,
                AuditVerdict.FAIL,
            ):
                high_cost_effective_calls += 1

    v3_neg_hyps -= v3_pos_hyps
    v3_total_conclusions = len(v3_pos_hyps) + len(v3_neg_hyps)
    win_hr = float(v2["window_hours"])
    real_tok = int(v2["tokens_real"]["uncached_total_tokens"])
    v3_yield_hr: float | str = (
        round(v3_total_conclusions / win_hr, 4) if win_hr > 0 and (receipts or tasks) else UNKNOWN
    )
    v3_yield_1m: float | str = (
        round(v3_total_conclusions / (real_tok / 1_000_000.0), 4)
        if real_tok > 0
        else UNKNOWN
    )

    # 2. Uncertainty reduction across ResearchPlans + uncertainty-linked hypotheses
    total_uncertainties = 0
    resolved_uncertainties = 0
    resolved_uncertainty_ids = {
        str(h.uncertainty_id)
        for h in hyps
        if h.uncertainty_id
        and h.scientific_verdict
        in (
            ScientificVerdict.CONFIRMED,
            ScientificVerdict.VALID_NEGATIVE,
            ScientificVerdict.REFUTED,
        )
    }
    plan_reworks = 0
    total_plans = 0
    for g in goals:
        rplans = db.list_research_plans(g.goal_id)
        total_plans += len(rplans)
        if len(rplans) > 1:
            plan_reworks += len(rplans) - 1
        latest_rp = rplans[-1] if rplans else None
        if latest_rp is not None:
            for idx, u in enumerate(latest_rp.uncertainties):
                total_uncertainties += 1
                uid = str(u.get("id") or u.get("uncertainty_id") or f"u_{idx}")
                ust = str(u.get("status") or "OPEN").upper()
                if ust in ("RESOLVED", "CLOSED", "CONFIRMED", "REFUTED") or uid in resolved_uncertainty_ids:
                    resolved_uncertainties += 1
        eplans_by_hyp: dict[str, int] = {}
        for ep in db.list_experiment_plans(goal_id=g.goal_id):
            total_plans += 1
            eplans_by_hyp[ep.hypothesis_id] = eplans_by_hyp.get(ep.hypothesis_id, 0) + 1
        for cnt in eplans_by_hyp.values():
            if cnt > 1:
                plan_reworks += cnt - 1

    # Also count uncertainty-linked hypotheses when no explicit ResearchPlan uncertainties exist
    if total_uncertainties == 0:
        linked_uids = {str(h.uncertainty_id) for h in hyps if h.uncertainty_id}
        total_uncertainties = len(linked_uids)
        resolved_uncertainties = len(linked_uids & resolved_uncertainty_ids)

    uncertainty_reduction_rate: float | str = (
        round(resolved_uncertainties / total_uncertainties, 4)
        if total_uncertainties > 0
        else UNKNOWN
    )

    # 3. High-cost model utilization
    high_cost_effective_rate: float | str = (
        round(high_cost_effective_calls / high_cost_calls, 4)
        if high_cost_calls > 0
        else UNKNOWN
    )

    # 4. Code rework vs Plan rework rates
    code_tasks = [t for t in tasks if t.stage == StageKind.CODE]
    code_rework_tasks = [
        t
        for t in code_tasks
        if int(t.attempt) > 1 or "repair" in (t.step_key or "").lower()
    ]
    code_rework_rate: float | str = (
        round(len(code_rework_tasks) / len(code_tasks), 4) if code_tasks else UNKNOWN
    )
    plan_rework_rate: float | str = (
        round(plan_reworks / total_plans, 4) if total_plans > 0 else UNKNOWN
    )

    # 5. Invalid / wasted task ratio
    terminal_tasks = [
        t
        for t in tasks
        if t.status
        in (
            "DONE",
            "SUCCEEDED",
            "PASSED",
            "FAILED",
            "CANCELLED",
            "DEAD",
            "TIMED_OUT",
            "LEASE_LOST",
            "BLOCKED_POLICY",
            "BLOCKED_QUOTA",
        )
    ]
    wasted_tasks = [
        t
        for t in terminal_tasks
        if t.status not in TERMINAL_TASK_SUCCESS
    ]
    invalid_or_wasted_task_ratio: float | str = (
        round(len(wasted_tasks) / len(terminal_tasks), 4)
        if terminal_tasks
        else UNKNOWN
    )

    # 6. Multi-Goal parallel efficiency
    eff_slots = slots or max(1, sum(max(1, g.max_concurrent_branches) for g in goals) or 4)
    slot_util = _compute_slot_utilization(tasks, slots=eff_slots, now_ts=now_ts)

    per_goal_efficiency = compute_per_goal_empirical_efficiency(
        db,
        goal_id=goal_id,
        receipts=receipts,
        reviews=reviews,
    )

    tr = v2["tokens_real"]
    return {
        "verified_positive_conclusions": len(v3_pos_hyps),
        "valid_negative_conclusions": len(v3_neg_hyps),
        "scientific_conclusions_total": v3_total_conclusions,
        "scientific_yield_per_hour": v3_yield_hr,
        "scientific_yield_per_1m_tokens": v3_yield_1m,
        "per_goal_empirical_efficiency": per_goal_efficiency,
        "token_telemetry": {
            "uncached_input_tokens": int(tr.get("uncached_input_tokens", tr.get("uncached_in_tokens", 0))),
            "cache_read_tokens": int(tr.get("cache_read_tokens", 0)),
            "cache_write_tokens": int(tr.get("cache_write_tokens", 0)),
            "output_tokens": int(tr.get("output_tokens", tr.get("out_tokens", 0))),
            "uncached_total_tokens": int(tr.get("uncached_total_tokens", 0)),
        },
        "uncertainty_reduction": {
            "resolved": resolved_uncertainties,
            "total": total_uncertainties,
            "reduction_rate": uncertainty_reduction_rate,
        },
        "high_cost_model_utilization": {
            "calls": high_cost_calls,
            "effective_calls": high_cost_effective_calls,
            "effective_rate": high_cost_effective_rate,
            "uncached_tokens": high_cost_tokens,
        },
        "rework_rates": {
            "code_rework_tasks": len(code_rework_tasks),
            "code_tasks_total": len(code_tasks),
            "code_rework_rate": code_rework_rate,
            "plan_reworks": plan_reworks,
            "plans_total": total_plans,
            "plan_rework_rate": plan_rework_rate,
        },
        "invalid_or_wasted_task_ratio": {
            "wasted_tasks": len(wasted_tasks),
            "terminal_tasks": len(terminal_tasks),
            "ratio": invalid_or_wasted_task_ratio,
            "wasted_uncached_tokens": v2["waste_and_rework_breakdown"]["wasted_uncached_tokens"],
        },
        "failure_domain_breakdown": failure_domains,
        "review_receipts_total": len(reviews),
        "multigoal_parallel_efficiency": {
            "slot_utilization_pct": slot_util["pct"],
            "cross_goal_refill_events": v2["cross_goal_refill_events"],
            "cross_goal_refill_p95_sec": v2["cross_goal_refill_p95_sec"],
            "idle_with_ready_ratio_pct": v2["idle_with_ready_ratio_pct"],
        },
        "human_intervention_latency_ms": v2["interactive_response_latency_ms"],
    }


def render_latest_status_md(
    db: StateDB,
    benchmark: dict[str, Any],
    *,
    escalations: list[dict[str, Any]] | None = None,
    v2_metrics: dict[str, Any] | None = None,
) -> str:
    """Render narrow-screen (<4 columns) markdown status for dashboards."""
    goals = db.list_goals()
    hyps = db.list_hypotheses()
    tasks = db.list_tasks()
    m = benchmark["metrics"]
    pw = benchmark["pilot_window"]
    ts_str = time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(benchmark["generated_ts"]))

    lines: list[str] = [
        "# Conjectra Status",
        "",
        f"Updated: `{ts_str}`",
        f"Window: `{pw['duration_hours']}h` (6h: `{pw['full_6h_window_status']}`)",
        "",
        "## KPIs vs Baseline",
        "",
        "| Metric | Pilot | Target/Base |",
        "| --- | --- | --- |",
        f"| Refill P95 (s) | {m['refill_latency_p95_sec']['value']} | <= {m['refill_latency_p95_sec']['target_max_sec']} |",
        f"| Slot Util (%) | {m['dispatchable_slot_utilization_pct']['value']} | >= {m['dispatchable_slot_utilization_pct']['target_min_pct']}% |",
        f"| Deliv/hr (pilot) | {m['verified_deliverables_per_hour']['value_pilot_window']} | Base {m['verified_deliverables_per_hour']['baseline_per_hour']} |",
        f"| Deliv/hr (6h) | {m['verified_deliverables_per_hour']['value_full_6h']} | 6h window |",
        f"| Tokens/Deliv | {m['tokens_per_verified_deliverable']['value']} | Base {m['tokens_per_verified_deliverable']['baseline_tokens_per_deliverable']} |",
        f"| Gross Tok/Deliv | {m['tokens_per_verified_deliverable']['gross_tokens_including_cache_per_deliverable']} | Cache {m['tokens_per_verified_deliverable']['cache_read_tokens_total']} |",
        f"| First Pass | {m['first_pass_rate']['value']} ({m['first_pass_rate']['numerator_passed']}/{m['first_pass_rate']['denominator_first_attempts']}) | Base {m['first_pass_rate']['baseline_first_pass_rate']} |",
        f"| Fail Trace (%) | {m['failure_traceability_pct']['value']} | {m['failure_traceability_pct']['target_pct']}% |",
        "",
        "## Goals & Budgets",
        "",
        "| Goal | Tokens | Hyp / Tasks |",
        "| --- | --- | --- |",
    ]

    for g in goals:
        usage = db.get_goal_token_usage(g.goal_id)
        g_hyps = [h for h in hyps if h.goal_id == g.goal_id]
        g_tasks = [t for t in tasks if t.goal_id == g.goal_id]
        lines.append(
            f"| {g.project}:{g.goal_id[:10]} | {usage['total_tokens']}/{g.max_tokens_budget} | {len(g_hyps)} / {len(g_tasks)} |"
        )
    if not goals:
        lines.append("| (none) | 0 | 0 / 0 |")

    lines.extend(
        [
            "",
            "## Owner Gates",
            "",
            "| Gate | Goal | Detail |",
            "| --- | --- | --- |",
        ]
    )
    esc_list = escalations or []
    if esc_list:
        for esc in esc_list[:20]:
            lines.append(
                f"| {esc.get('gate', 'GATE')} | {esc.get('goal_id', '-')} | {esc.get('reason', '')} |"
            )
    else:
        lines.append("| NONE | - | No owner-only escalations |")

    if v2_metrics is not None:
        lines.extend(
            [
                "",
                "## V2 Research Telemetry",
                "",
                "| Signal | Value | Detail |",
                "| --- | --- | --- |",
                f"| Pos / Neg | {v2_metrics['verified_positive_conclusions']} / {v2_metrics['valid_negative_conclusions']} | Real yield {v2_metrics['scientific_yield_total']} |",
                f"| Yield / hr | {v2_metrics['scientific_yield_per_hour']} | Per 1M tok: {v2_metrics['scientific_yield_per_1m_tokens']} |",
                f"| Idle/Ready (%) | {v2_metrics['idle_with_ready_ratio_pct']} | X-Goal refills: {v2_metrics['cross_goal_refill_events']} |",
                f"| Scheduler Tok | {v2_metrics['scheduler_llm_tokens']} | Interventions: {v2_metrics['human_interventions_summary']['total']} |",
            ]
        )

    v3 = benchmark.get("v3_metrics")
    if isinstance(v3, dict):
        unc = v3.get("uncertainty_reduction") or {}
        hc = v3.get("high_cost_model_utilization") or {}
        rw = v3.get("rework_rates") or {}
        wst = v3.get("invalid_or_wasted_task_ratio") or {}
        lines.extend(
            [
                "",
                "## V3 Scientific KPIs",
                "",
                "| Signal | Value | Detail |",
                "| --- | --- | --- |",
                f"| Pos / Neg | {v3.get('verified_positive_conclusions', 0)} / {v3.get('valid_negative_conclusions', 0)} | {v3.get('scientific_yield_per_1m_tokens', UNKNOWN)} / 1M tok |",
                f"| Uncertainty | {unc.get('resolved', 0)}/{unc.get('total', 0)} | Rate: {unc.get('reduction_rate', UNKNOWN)} |",
                f"| Frontier Util | {hc.get('effective_rate', UNKNOWN)} | {hc.get('effective_calls', 0)}/{hc.get('calls', 0)} calls |",
                f"| Rework (C/P) | {rw.get('code_rework_rate', UNKNOWN)} / {rw.get('plan_rework_rate', UNKNOWN)} | Waste: {wst.get('ratio', UNKNOWN)} |",
            ]
        )

    lines.append("")
    return "\n".join(lines)


def sync_status_mirror(
    status_md: str,
    benchmark: dict[str, Any],
    *,
    target: str | None = None,
    status_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mirror LATEST_STATUS.md, benchmark_latest.json and optional status_snapshot.json to a dashboard.

    ``target`` is a local directory or ``user@host:/dir`` (ssh). Defaults to $CONJECTRA_STATUS_MIRROR.
    """
    tgt = (target if target is not None else DEFAULT_MIRROR_TARGET).strip()
    if not tgt:
        return {"mode": "disabled", "reason": "no mirror target (set CONJECTRA_STATUS_MIRROR)"}
    files: list[tuple[str, str]] = [
        ("LATEST_STATUS.md", status_md),
        ("benchmark_latest.json", json.dumps(benchmark, indent=2, ensure_ascii=False) + "\n"),
    ]
    if status_snapshot is not None:
        files.append(
            ("status_snapshot.json", json.dumps(status_snapshot, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
        )
    is_ssh = ":" in tgt and not pathlib.Path(tgt).exists() and not tgt.startswith(("/", "."))
    if not is_ssh:
        out_dir = pathlib.Path(tgt).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        written = {}
        for name, payload in files:
            (out_dir / name).write_text(payload, encoding="utf-8")
            written[name] = str(out_dir / name)
        return {"mode": "local", "dir": str(out_dir), "files": written}
    host, _, rdir = tgt.partition(":")
    for name, payload in files:
        dest = f"{rdir.rstrip('/')}/{name}"
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=5",
                host,
                f"mkdir -p {shlex.quote(rdir)} && cat > {shlex.quote(dest)}",
            ],
            input=payload,
            text=True,
            check=True,
            timeout=15,
        )
    return {"mode": "ssh", "host": host, "dir": rdir, "files": [n for n, _ in files]}
