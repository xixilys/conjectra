"""Provider-agnostic clean-room multi-agent modes built on Conjectra:
1. `boost` (`/x-boost`): Heterogeneous Builder <-> Blind Adversary ping-pong loop.
2. `xswarm` (`/x-swarm`): SQLite append-only board + `@name` cursor wakeup + 0-token stall watchdog.
3. `xteamwork` (`/x-teamwork`): Pre-flight contract + disjoint scopes + <4KB milestone flush + 2-tier audit.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Callable, Mapping, Sequence
import uuid

from conjectra import gates
from conjectra.arbiter import ResourceArbiter
from conjectra.hq_runtime import HQRuntime
from conjectra.stage_runner import (
    DEFAULT_MULTIPOOL_SCRIPT,
    SandboxManager,
    _git,
    _run_llm_pool,
    _run_subprocess,
    extract_cache_tokens,
    extract_token_usage,
    parse_test_counts,
    wait_for_mem_available,
)
from conjectra.state_db import StateDB, resolve_arr_home
from conjectra.types import (
    ProcessStatus,
    ResourceSpec,
    StageKind,
)

MAX_BOARD_MSG_CHARS = 4096
MAX_MILESTONE_ANCHOR_BYTES = 4096
MENTION_RE = re.compile(r"@([a-zA-Z0-9][a-zA-Z0-9_-]{0,31})")
BOOST_VERDICT_RE = re.compile(r"BOOST_VERDICT\s*:\s*(PASS|BROKEN)", re.IGNORECASE)
AUDIT_VERDICT_RE = re.compile(r"AUDIT_VERDICT\s*:\s*(PASS|FAIL)", re.IGNORECASE)

LLMHook = Callable[[str, "ProviderSpec", Path, bool, str], tuple[int, str, int, int, int]]


@dataclass(frozen=True)
class ProviderSpec:
    """Model-agnostic CLI/pool specification (`<pool>:<model>[:<effort>]` or `cmd:<shell>`)."""

    pool: str
    model: str
    effort: str = "medium"
    family: str = "unknown"
    custom_cmd: str | None = None
    pinned_route: dict[str, Any] | None = None

    def label(self) -> str:
        if self.custom_cmd:
            return f"cmd:{self.model}"
        return f"{self.pool}:{self.model}"

    def canonical_spec(self) -> str:
        if self.custom_cmd:
            return f"cmd:{self.custom_cmd}"
        return f"{self.pool}:{self.model}:{self.effort}"

    def to_dict(self) -> dict[str, Any]:
        out = {
            "pool": self.pool,
            "model": self.model,
            "effort": self.effort,
            "family": self.family,
            "custom_cmd": self.custom_cmd,
        }
        if self.pinned_route is not None:
            out["pinned_route"] = dict(self.pinned_route)
        return out


def infer_model_family(pool: str, model: str) -> str:
    """Infer provider family so cross-family blind checks never pair same-origin models."""
    combined = f"{pool}:{model}".lower()
    if pool == "cmd":
        for fam, keys in (
            ("anthropic", ("claude", "sonnet", "opus", "haiku")),
            ("openai", ("gpt", "codex", "o1", "o3", "o4")),
            ("google", ("gemini",)),
            ("deepseek", ("deepseek",)),
            ("xai", ("grok",)),
        ):
            if any(k in combined for k in keys):
                return fam
        return f"custom_{hashlib.sha1(combined.encode('utf-8')).hexdigest()[:6]}"
    if any(k in combined for k in ("claude", "sonnet", "opus", "haiku")):
        return "anthropic"
    if any(k in combined for k in ("codex", "gpt-")):
        return "openai"
    if "gemini" in combined:
        return "google"
    if "grok" in combined:
        return "xai"
    if "deepseek" in combined:
        return "deepseek"
    return "unknown"


def parse_provider_spec(
    spec: str | None,
    *,
    role: str = "builder",
    avoid_family: str | None = None,
    arbiter: ResourceArbiter | None = None,
) -> ProviderSpec:
    """Parse an explicit provider spec or select a healthy cross-family default."""
    raw = (spec or "auto").strip()
    if raw and raw.lower() != "auto":
        if raw.startswith("cmd:"):
            shell_cmd = raw[4:].strip()
            if not shell_cmd:
                raise ValueError("Empty custom command in cmd:<shell_cmd>")
            short_name = shlex.split(shell_cmd)[0] if shell_cmd else "custom"
            fam = infer_model_family("cmd", shell_cmd)
            return ProviderSpec(
                pool="cmd",
                model=Path(short_name).name,
                effort="custom",
                family=fam,
                custom_cmd=shell_cmd,
            )
        parts = raw.split(":")
        if len(parts) == 1:
            token = parts[0].strip()
            low = token.lower()
            default_eff = "medium" if role in ("adversary", "blind_auditor") else "high"
            if low in ("claude", "sonnet"):
                pool, model, effort = "claude", "claude-sonnet-5-5", "medium"
            elif low == "opus":
                pool, model, effort = "claude", "claude-opus-5-5", default_eff
            elif low == "codex":
                pool, model, effort = "codex", "gpt-5-codex", default_eff
            elif low == "gpt-pro":
                pool, model, effort = "codex", "gpt-5-pro", default_eff
            elif low in ("gemini", "flash"):
                pool, model, effort = "gemini", "gemini-flash", default_eff
            elif low == "gemini-pro":
                pool, model, effort = "gemini", "gemini-pro", default_eff
            elif low == "deepseek":
                pool, model, effort = "deepseek", "deepseek-chat", "high"
            elif low == "grok":
                pool, model, effort = "grok", "grok-code-fast", "high"
            else:
                pool, model, effort = "gemini", token, default_eff
        elif len(parts) == 2:
            pool, model = parts[0].strip(), parts[1].strip()
            if "sonnet" in model.lower():
                effort = "medium"
            elif pool in ("deepseek", "grok") or "deepseek" in model.lower() or "grok" in model.lower():
                effort = "high"
            elif role in ("adversary", "blind_auditor"):
                effort = "medium"
            else:
                effort = "high"
        else:
            pool, model, effort = parts[0].strip(), parts[1].strip(), parts[2].strip()
        if "sonnet" in model.lower() and effort.lower() in ("high", "xhigh", "max"):
            effort = "medium"
        if (pool == "deepseek" or "deepseek-chat" in model.lower()) and effort.lower() == "medium":
            effort = "high"
        if (pool == "grok" or "grok-code-fast" in model.lower()) and effort.lower() != "high":
            effort = "high"
        fam = infer_model_family(pool, model)
        if avoid_family and fam == avoid_family:
            raise ValueError(
                f"Cross-family violation: role={role} spec '{raw}' shares family '{fam}' with '{avoid_family}'"
            )
        return ProviderSpec(pool=pool, model=model, effort=effort, family=fam)

    headroom = arbiter.read_pool_headroom() if arbiter is not None else {}

    def _pool_ok(logical_pool: str) -> bool:
        info = headroom.get(logical_pool)
        return True if not info else bool(info.get("available", True))

    if role in ("prescreener", "l1_batch"):
        candidates = [
            ("gemini", "gemini-flash", "high", "llm_gemini"),
            ("deepseek", "deepseek-chat", "high", "llm_deepseek"),
            ("claude", "claude-sonnet-5-5", "low", "llm_claude"),
        ]
    elif role in ("adversary", "blind_auditor"):
        candidates = [
            ("codex", "gpt-5-codex", "medium", "llm_codex_audit"),
            ("claude", "claude-opus-5-5", "medium", "llm_claude"),
            ("gemini", "gemini-pro", "medium", "llm_gemini"),
            ("grok", "grok-code-fast", "high", "llm_deepseek"),
        ]
    else:
        candidates = [
            ("claude", "claude-sonnet-5-5", "medium", "llm_claude"),
            ("gemini", "gemini-pro", "high", "llm_gemini"),
            ("deepseek", "deepseek-chat", "high", "llm_deepseek"),
            ("codex", "gpt-5-codex", "high", "llm_codex_audit"),
        ]

    for pool, model, effort, logical_pool in candidates:
        fam = infer_model_family(pool, model)
        if avoid_family and fam == avoid_family:
            continue
        if _pool_ok(logical_pool):
            return ProviderSpec(pool=pool, model=model, effort=effort, family=fam)

    for pool, model, effort, _ in candidates:
        fam = infer_model_family(pool, model)
        if not avoid_family or fam != avoid_family:
            return ProviderSpec(pool=pool, model=model, effort=effort, family=fam)

    raise RuntimeError(f"Unable to select cross-family provider avoiding {avoid_family!r}")


def invoke_provider(
    provider: ProviderSpec,
    prompt: str,
    *,
    cwd: Path,
    art_dir: Path,
    tag: str,
    readonly: bool = False,
    timeout_s: int = 600,
    multipool_script: Path | None = None,
    llm_hook: LLMHook | None = None,
) -> dict[str, Any]:
    """Execute one-shot prompt against any configured ProviderSpec and exit immediately."""
    art_dir.mkdir(parents=True, exist_ok=True)
    out_md = art_dir / f"{tag}_out.md"
    usage_out = art_dir / f"{tag}_usage.json"

    if llm_hook is not None:
        t0 = time.perf_counter()
        rc, text, in_tok, out_tok, cache_tok = llm_hook(prompt, provider, cwd, readonly, tag)
        wall = time.perf_counter() - t0
        out_md.write_text(text, encoding="utf-8")
        usage_out.write_text(
            json.dumps(
                {
                    "model": provider.label(),
                    "synthetic": True,
                    "usage": {
                        "input_tokens": in_tok,
                        "output_tokens": out_tok,
                        "cache_read_tokens": cache_tok,
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "exit_code": rc,
            "response": text,
            "in_tokens": in_tok,
            "out_tokens": out_tok,
            "cache_tokens": cache_tok,
            "wall_sec": wall,
            "max_rss_mb": 0.0,
            "model_used": provider.label(),
            "out_md": str(out_md),
            "usage_out": str(usage_out),
            "is_real_llm": False,
            "process_status": ProcessStatus.EXIT_OK if rc == 0 else ProcessStatus.EXIT_NONZERO,
        }

    if provider.custom_cmd:
        rc, out, err, wall, rss, pstat = _run_subprocess(
            provider.custom_cmd,
            cwd=cwd,
            timeout_s=timeout_s,
            stdin_text=prompt,
            shell=True,
        )
        text = out if out.strip() else err
        out_md.write_text(text, encoding="utf-8")
        est_in = max(1, len(prompt) // 4)
        est_out = max(1, len(text) // 4)
        usage_out.write_text(
            json.dumps(
                {
                    "model": provider.label(),
                    "estimated": True,
                    "usage": {"input_tokens": est_in, "output_tokens": est_out, "cache_read_tokens": 0},
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "exit_code": rc,
            "response": text,
            "in_tokens": est_in,
            "out_tokens": est_out,
            "cache_tokens": 0,
            "wall_sec": wall,
            "max_rss_mb": rss,
            "model_used": provider.label(),
            "out_md": str(out_md),
            "usage_out": str(usage_out),
            "is_real_llm": False,
            "process_status": pstat,
        }

    mp_script = multipool_script or DEFAULT_MULTIPOOL_SCRIPT
    from conjectra.types import ResolvedModelRoute

    route_obj = (
        ResolvedModelRoute.from_dict(provider.pinned_route)
        if isinstance(provider.pinned_route, Mapping)
        and provider.pinned_route.get("pool")
        and provider.pinned_route.get("canonical_model")
        else None
    )
    rc, out, err, wall, rss, pstat, _cmd_str = _run_llm_pool(
        multipool=mp_script,
        pool=provider.pool,
        model=provider.model,
        effort=provider.effort,
        pinned_route=route_obj,
        cwd=cwd,
        out_md=out_md,
        usage_out=usage_out,
        timeout_s=timeout_s,
        tag=tag,
        prompt=prompt,
        readonly=readonly,
    )
    text = out_md.read_text(encoding="utf-8", errors="replace") if out_md.is_file() else (out or err)
    in_tok, out_tok, model_used = 0, 0, provider.label()
    cache_tok = 0
    is_real = False
    if usage_out.is_file():
        try:
            udoc = json.loads(usage_out.read_text(encoding="utf-8"))
            in_tok, out_tok, parsed_m = extract_token_usage(udoc)
            cache_tok = extract_cache_tokens(udoc)
            if parsed_m and parsed_m != "unknown":
                model_used = f"{provider.pool}:{parsed_m}" if ":" not in parsed_m else parsed_m
            is_real = bool(
                not udoc.get("synthetic")
                and not udoc.get("estimated")
                and (in_tok + out_tok) > 0
            )
        except Exception:
            pass
    return {
        "exit_code": rc,
        "response": text,
        "in_tokens": in_tok,
        "out_tokens": out_tok,
        "cache_tokens": cache_tok,
        "wall_sec": wall,
        "max_rss_mb": rss,
        "model_used": model_used,
        "out_md": str(out_md),
        "usage_out": str(usage_out) if usage_out.is_file() else "",
        "is_real_llm": is_real,
        "process_status": pstat,
    }


def _run_nice_test(
    acceptance_cmd: str,
    *,
    cwd: Path,
    timeout_s: int = 300,
) -> tuple[int, str, str, float, float, ProcessStatus, int, int]:
    """Wait for MemAvailable >= 10GiB (or env override) and execute acceptance_cmd under nice -n 10."""
    mem_ok, avail_mb = wait_for_mem_available(max_wait_s=min(180.0, max(5.0, timeout_s / 2.0)))
    if not mem_ok:
        return (
            75,
            "",
            f"MEM_BELOW_10GB_TIMEOUT (avail={avail_mb}MB)",
            0.0,
            0.0,
            ProcessStatus.ENV_ERROR,
            0,
            0,
        )
    wrapped = f"nice -n 10 bash -lc {shlex.quote(acceptance_cmd)}"
    rc, out, err, wall, rss, pstat = _run_subprocess(
        wrapped,
        cwd=cwd,
        timeout_s=timeout_s,
        shell=True,
    )
    passed, total = parse_test_counts(f"{out}\n{err}", rc)
    return rc, out, err, wall, rss, pstat, passed, total


def _run_gate_test(
    acceptance_cmd: str,
    *,
    cwd: Path,
    timeout_s: int = 300,
    hq: HQRuntime | None = None,
    task_id: str | None = None,
    art_dir: Path | None = None,
) -> tuple[int, str, str, float, float, ProcessStatus, int, int, int | None]:
    """Run acceptance/holdout test via HyperQueue (`testsim=1`, `nice -n 10`) when `hq` is provided."""
    mem_ok, avail_mb = wait_for_mem_available(max_wait_s=min(180.0, max(5.0, timeout_s / 2.0)))
    if not mem_ok:
        return (
            75,
            "",
            f"MEM_BELOW_10GB_TIMEOUT (avail={avail_mb}MB)",
            0.0,
            0.0,
            ProcessStatus.ENV_ERROR,
            0,
            0,
            None,
        )
    if hq is None:
        rc, out, err, wall, rss, pstat, passed, total = _run_nice_test(
            acceptance_cmd, cwd=cwd, timeout_s=timeout_s
        )
        return rc, out, err, wall, rss, pstat, passed, total, None

    tid = task_id or f"gate-{uuid.uuid4().hex[:8]}"
    log_dir = art_dir or hq.jobs_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    out_file = log_dir / f"{tid}.out"
    err_file = log_dir / f"{tid}.err"

    t0 = time.perf_counter()
    jid = hq.submit_task(
        ["/usr/bin/nice", "-n", "10", "/bin/bash", "-lc", acceptance_cmd],
        task_id=tid,
        cwd=cwd,
        stage=StageKind.CPU_GATE,
        resources=ResourceSpec(cpus=2, ram_mb=2048, wall_timeout_s=timeout_s),
        stdout=out_file,
        stderr=err_file,
        wrap_exit=False,
    )
    info = hq.wait_job(jid, timeout_s=float(timeout_s) + 20.0)
    wall = time.perf_counter() - t0
    stats = (info or {}).get("info", {}).get("task_stats", {})
    ok = (
        int(stats.get("finished", 0)) > 0
        and int(stats.get("failed", 0)) == 0
        and int(stats.get("canceled", 0)) == 0
        and int(stats.get("aborted", 0)) == 0
    )
    rc = 0 if ok else 1
    out = out_file.read_text(encoding="utf-8", errors="replace") if out_file.is_file() else ""
    err = err_file.read_text(encoding="utf-8", errors="replace") if err_file.is_file() else ""
    passed, total = parse_test_counts(f"{out}\n{err}", rc)
    pstat = ProcessStatus.EXIT_OK if rc == 0 else ProcessStatus.EXIT_NONZERO
    return rc, out, err, wall, 0.0, pstat, passed, total, jid


def _parse_boost_verdict(text: str) -> tuple[str, str]:
    """Extract (PASS|BROKEN, counterexample_summary) from adversary output."""
    verdict = "BROKEN"
    for line in reversed((text or "").splitlines()):
        m = BOOST_VERDICT_RE.search(line)
        if m:
            verdict = m.group(1).upper()
            break
    counterexample = ""
    m_ce = re.search(r"COUNTEREXAMPLE\s*:\s*(.+)", text or "", re.IGNORECASE | re.DOTALL)
    if m_ce:
        counterexample = m_ce.group(1).strip()[:2000]
    elif verdict == "BROKEN":
        counterexample = (text or "").strip()[-1500:]
    return verdict, counterexample


def _run_repro_python(
    repro_python: str,
    *,
    cwd: Path,
    timeout_s: int = 10,
) -> tuple[int, str, str]:
    """Run adversary-supplied repro_python deterministically under `nice -n 10` (<=10s)."""
    snippet = (repro_python or "").strip()
    if not snippet:
        return 0, "", ""
    nice_bin = shutil.which("nice") or "/usr/bin/nice"
    cmd = [nice_bin, "-n", "10", sys.executable, "-c", snippet]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cur_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{cwd}{os.pathsep}{cur_pp}" if cur_pp else str(cwd)
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=min(10, max(1, int(timeout_s))),
            env=env,
            check=False,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout.decode("utf-8", errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
        err = exc.stderr.decode("utf-8", errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
        return 124, out, f"{err}\n[REPRO_TIMEOUT after {timeout_s}s]".strip()
    except OSError as exc:
        return 127, "", str(exc)


# ---------------------------------------------------------------------------
# 1. Clean-Room Cross-Provider /x-boost
# ---------------------------------------------------------------------------


def _router_has_matching_hard_rule(
    router: Any,
    role: Any,
    goal: Any,
    hyp: Any,
) -> bool:
    """Return True when `goal` defines a matching `ConstraintStrength.HARD` routing rule for `(role, hyp)`."""
def _router_has_matching_hard_rule(
    router: Any,
    role: Any,
    goal: Any,
    hyp: Any,
    *,
    workflow_id: str | None = None,
    task: Any = None,
    wf_config: Mapping[str, Any] | None = None,
    db: StateDB | None = None,
) -> bool:
    if hasattr(router, "has_matching_hard_rule"):
        return bool(
            router.has_matching_hard_rule(
                role,
                goal=goal,
                hyp=hyp,
                workflow_id=workflow_id,
                task=task,
                wf_config=wf_config,
                db=db,
            )
        )
    return False


def _provider_spec_from_route(route: Any) -> ProviderSpec:
    if str(route.pool) == "cmd":
        shell_cmd = str(route.canonical_model).strip()
        short_name = Path(shlex.split(shell_cmd)[0]).name if shell_cmd else "custom"
        return ProviderSpec(
            pool="cmd",
            model=short_name,
            effort="custom",
            family=route.family,
            custom_cmd=shell_cmd,
            pinned_route=route.to_dict(),
        )
    return ProviderSpec(
        pool=route.pool,
        model=route.canonical_model,
        effort=route.wire_effort,
        family=route.family,
        pinned_route=route.to_dict(),
    )


def _resolve_multiagent_provider(
    *,
    spec: str | None,
    sci_role: Any,
    legacy_role: str,
    arbiter: ResourceArbiter,
    home: Path,
    db: StateDB | None = None,
    goal: Any = None,
    hyp: Any = None,
    workflow_id: str | None = None,
    wf_config: Mapping[str, Any] | None = None,
    task: Any = None,
    model_router: Any = None,
    pinned_route: Mapping[str, Any] | None = None,
    avoid_families: Sequence[str] = (),
    prefer_avoid_families: Sequence[str] = (),
) -> ProviderSpec:
    """Resolve one multi-agent participant slot ALWAYS enforcing `ModelRouter` & SQLite/capabilities global policy."""
    from conjectra.model_router import ModelRouter, PolicyBlockedError
    from conjectra.types import ConstraintStrength, ResolvedModelRoute

    raw_spec = (spec or "").strip()
    router = model_router or ModelRouter(arbiter=arbiter, db=db, arr_home=home)
    has_eff_policy = router.has_effective_policy(
        goal=goal,
        hyp=hyp,
        workflow_id=workflow_id,
        task=task,
        wf_config=wf_config,
        db=db,
    )
    is_policy_bound = bool(
        has_eff_policy
        or goal is not None
        or hyp is not None
        or bool(workflow_id)
        or bool(wf_config)
        or task is not None
    )

    if (
        isinstance(pinned_route, Mapping)
        and pinned_route.get("pool")
        and pinned_route.get("canonical_model")
    ):
        if is_policy_bound:
            checked = router.revalidate_pinned_route(
                sci_role,
                pinned_route,
                goal=goal,
                hyp=hyp,
                workflow_id=workflow_id,
                task=task,
                wf_config=wf_config,
                db=db,
                avoid_families=avoid_families,
            )
            return _provider_spec_from_route(checked)
        r_obj = ResolvedModelRoute.from_dict(pinned_route)
        if avoid_families and r_obj.family.lower() in {str(f).lower() for f in avoid_families}:
            raise PolicyBlockedError(
                f"BLOCKED_CROSS_FAMILY: pinned route family {r_obj.family!r} overlaps {list(avoid_families)}",
                waitable=False,
                code="BLOCKED_CROSS_FAMILY",
            )
        return _provider_spec_from_route(r_obj)

    if raw_spec.startswith("cmd:"):
        if is_policy_bound:
            cmd_route = router.resolve_route(
                sci_role,
                goal=goal,
                hyp=hyp,
                workflow_id=workflow_id,
                task=task,
                wf_config=wf_config,
                db=db,
                explicit_spec=raw_spec,
                avoid_families=avoid_families,
                prefer_avoid_families=prefer_avoid_families,
            )
            checked_cmd = router.revalidate_pinned_route(
                sci_role,
                cmd_route,
                goal=goal,
                hyp=hyp,
                workflow_id=workflow_id,
                task=task,
                wf_config=wf_config,
                db=db,
                avoid_families=avoid_families,
            )
            return _provider_spec_from_route(checked_cmd)
        low_cmd = raw_spec.lower()
        for banned in router.banned_channels:
            if banned and banned in low_cmd:
                raise PolicyBlockedError(
                    f"BLOCKED_BANNED_CHANNEL:{banned}",
                    waitable=False,
                    rule_id="explicit_spec",
                    code="BLOCKED_BANNED_CHANNEL",
                )
        return parse_provider_spec(
            raw_spec,
            role=legacy_role,
            avoid_family=avoid_families[0] if avoid_families else None,
            arbiter=arbiter,
        )

    if _router_has_matching_hard_rule(
        router,
        sci_role,
        goal,
        hyp,
        workflow_id=workflow_id,
        task=task,
        wf_config=wf_config,
        db=db,
    ):
        route = router.resolve_route(
            sci_role,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            task=task,
            wf_config=wf_config,
            db=db,
            explicit_spec=None,
            avoid_families=avoid_families,
            prefer_avoid_families=prefer_avoid_families,
        )
        return _provider_spec_from_route(route)

    if not is_policy_bound:
        eff_arbiter = getattr(router, "arbiter", None) or arbiter
        # Preserve legacy reviewer default effort ("medium" on 2-part spec) while STILL validating through ModelRouter!
        parsed_fb = parse_provider_spec(
            raw_spec or "auto",
            role=legacy_role,
            avoid_family=avoid_families[0] if avoid_families else None,
            arbiter=eff_arbiter,
        )
        route = router.resolve_route(
            sci_role,
            db=db,
            explicit_spec=f"{parsed_fb.pool}:{parsed_fb.model}:{parsed_fb.effort}",
            avoid_families=avoid_families,
            prefer_avoid_families=prefer_avoid_families,
            strength=ConstraintStrength.SOFT_FALLBACK,
        )
        return _provider_spec_from_route(route)

    eff_opt = None if (not raw_spec or raw_spec.lower() == "auto") else raw_spec
    route = router.resolve_route(
        sci_role,
        goal=goal,
        hyp=hyp,
        workflow_id=workflow_id,
        task=task,
        wf_config=wf_config,
        db=db,
        explicit_spec=eff_opt,
        avoid_families=avoid_families,
        prefer_avoid_families=prefer_avoid_families,
    )
    return _provider_spec_from_route(route)


def run_boost(
    *,
    repo_path: str | Path,
    project: str,
    task_spec: str,
    acceptance_cmd: str,
    writable_paths: Sequence[str],
    banned_paths: Sequence[str] = (),
    builder_spec: str | None = None,
    adversary_spec: str | None = None,
    max_rounds: int = 3,
    max_repairs: int | None = None,
    timeout_s: int = 600,
    in_place: bool = False,
    use_hq: bool = True,
    multipool_script: Path | None = None,
    arr_home: str | Path | None = None,
    sandbox_manager: SandboxManager | None = None,
    llm_hook: LLMHook | None = None,
    goal: Any = None,
    hyp: Any = None,
    workflow_id: str | None = None,
    wf_config: Mapping[str, Any] | None = None,
    task: Any = None,
    model_router: Any = None,
    pinned_builder_route: Mapping[str, Any] | None = None,
    pinned_adversary_route: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run clean-room Build -> Gate -> <=max_repairs(Adversary -> ReproCheck -> Repair -> Gate)."""
    home = resolve_arr_home(arr_home)
    db = StateDB(home)
    arbiter = ResourceArbiter()
    run_id = f"boost-{uuid.uuid4().hex[:10]}"
    art_root = home / "artifacts" / "x_boost" / run_id
    art_root.mkdir(parents=True, exist_ok=True)

    try:
        from conjectra.types import ScientificRole

        builder = _resolve_multiagent_provider(
            spec=builder_spec,
            sci_role=ScientificRole.ENGINEER_BUILDER,
            legacy_role="builder",
            arbiter=arbiter,
            home=home,
            db=db,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            wf_config=wf_config,
            task=task,
            model_router=model_router,
            pinned_route=pinned_builder_route,
        )
        adversary = _resolve_multiagent_provider(
            spec=adversary_spec,
            sci_role=ScientificRole.IMPL_REVIEWER,
            legacy_role="adversary",
            arbiter=arbiter,
            home=home,
            db=db,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            wf_config=wf_config,
            task=task,
            model_router=model_router,
            pinned_route=pinned_adversary_route,
            avoid_families=[builder.family],
        )
        if builder.family == adversary.family:
            raise ValueError(f"Builder and adversary must be distinct model families, got {builder.family}")

        repo = Path(repo_path).expanduser().resolve()
        ban_pre = gates.check_no_banned_access(
            paths=writable_paths,
            command=acceptance_cmd,
            extra_bans=banned_paths,
            repo_path=repo,
        )
        if not ban_pre.passed:
            raise ValueError(f"Pre-flight banned access blocked: {ban_pre.failures}")

        base_rev = _git(repo, "rev-parse", "HEAD")
        if base_rev.returncode != 0:
            raise RuntimeError(f"Invalid git repo {repo}: {base_rev.stderr}")
        base_sha = base_rev.stdout.strip()
        branch_name = f"arr/{project}/{run_id}"

        if in_place:
            cur_branch = _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
            if cur_branch in ("main", "master"):
                raise ValueError(f"Refusing --in-place on protected workspace/branch ({cur_branch} at {repo})")
            wt_path = repo
        else:
            sb = sandbox_manager or SandboxManager()
            mirror = sb.ensure_mirror(repo, project)
            wt_path = sb.create_worktree(mirror, project, branch_name, base_sha, origin_repo=repo)

        hq: HQRuntime | None = None
        if use_hq and llm_hook is None:
            hq = HQRuntime(home)
            hq.ensure_running()

        rounds_trace: list[dict[str, Any]] = []
        raw_usage_paths: list[str] = []
        hq_jobs_submitted: list[int] = []
        llm_is_real_flags: list[bool] = []
        repro_checks: list[dict[str, Any]] = []
        total_in = 0
        total_out = 0
        total_cache = 0
        adversary_tokens = 0
        false_alarms_discarded = 0
        final_status = "FAILED"
        head_sha = base_sha

        max_iters = max(1, min(4, int(max_rounds)))
        repairs_allowed = max(0, int(max_repairs)) if max_repairs is not None else max(0, max_iters - 1)
        repairs_used = 0

        def _commit_and_gate(round_tag: str) -> tuple[bool, int, str, str, float, int, int, str]:
            nonlocal head_sha
            if not in_place:
                _git(wt_path, "add", "--", *writable_paths)
                diff_cached = _git(wt_path, "diff", "--cached", "--name-only")
                if diff_cached.stdout.strip():
                    _git(
                        wt_path,
                        "-c",
                        "user.name=conjectra",
                        "-c",
                        "user.email=conjectra@local",
                        "commit",
                        "-m",
                        f"boost({run_id}): {round_tag} implementation",
                    )
            head_sha = _git(wt_path, "rev-parse", "HEAD").stdout.strip() or base_sha
            gate_res = gates.run_cpu_gate(
                wt_path,
                base_sha,
                head_sha,
                writable_paths,
                goal_prefixes=writable_paths,
                extra_bans=banned_paths,
                command=acceptance_cmd,
                branch_name=branch_name,
                require_clean=(not in_place),
                require_non_doc=(not in_place),
                require_distinct_commits=(not in_place),
            )
            t_rc, t_out, t_err, t_wall, _t_rss, _t_pstat, t_passed, t_total, gjid = _run_gate_test(
                acceptance_cmd,
                cwd=wt_path,
                timeout_s=min(300, timeout_s),
                hq=hq,
                task_id=f"{run_id}-{round_tag}-gate",
                art_dir=art_root,
            )
            if gjid is not None:
                hq_jobs_submitted.append(int(gjid))
            ok = bool(gate_res.passed and t_rc == 0 and t_passed > 0)
            gate_err = "; ".join(gate_res.failures) if not gate_res.passed else ""
            return ok, t_rc, t_out, t_err, t_wall, t_passed, t_total, gate_err

        def _record_usage(res: Mapping[str, Any]) -> None:
            nonlocal total_in, total_out, total_cache
            total_in += int(res.get("in_tokens", 0))
            total_out += int(res.get("out_tokens", 0))
            total_cache += int(res.get("cache_tokens", 0))
            up = str(res.get("usage_out") or "")
            if up and up not in raw_usage_paths:
                raw_usage_paths.append(up)
            llm_is_real_flags.append(bool(res.get("is_real_llm", False)))

        # Step 1: Initial Build -> Gate
        rnd_num = 1
        rnd_dir = art_root / f"round_{rnd_num}"
        rnd_dir.mkdir(parents=True, exist_ok=True)
        b_prompt = (
            f"ROLE: Clean-Room Boost Builder (Initial Build)\n"
            f"TASK REQUIREMENT:\n{task_spec}\n\n"
            f"STRICT WRITABLE PATHS: {', '.join(writable_paths)}\n"
            f"ACCEPTANCE COMMAND: {acceptance_cmd}\n"
            "RULES:\n"
            "- Modify ONLY the allowed writable paths; preserve all existing assertions.\n"
            "- Handle boundary inputs (empty, unhashable, zero/negative, overflow, type mismatch) explicitly.\n"
            "- Run the acceptance command locally before finishing.\n"
        )
        b_res = invoke_provider(
            builder,
            b_prompt,
            cwd=wt_path,
            art_dir=rnd_dir,
            tag=f"r{rnd_num}_builder",
            readonly=False,
            timeout_s=timeout_s,
            multipool_script=multipool_script,
            llm_hook=llm_hook,
        )
        _record_usage(b_res)
        gate_ok, t_rc, t_out, t_err, t_wall, t_passed, t_total, gate_err = _commit_and_gate(f"r{rnd_num}")
        builder_tok_cur = int(b_res["in_tokens"]) + int(b_res["out_tokens"])

        while not gate_ok and repairs_used < repairs_allowed:
            feedback = (
                f"Machine gate / test failed (rc={t_rc}, passed={t_passed}/{t_total}).\n"
                f"Gate failures: {gate_err}\n"
                f"Raw stdout/stderr:\n{(t_out + chr(10) + t_err)[-2500:]}"
            )
            rounds_trace.append(
                {
                    "round": rnd_num,
                    "head_sha": head_sha,
                    "gate_passed": False,
                    "test_rc": t_rc,
                    "tests_passed": t_passed,
                    "adversary_verdict": "SKIPPED_ON_GATE_FAIL",
                    "builder_tokens": builder_tok_cur,
                    "adversary_tokens": 0,
                    "failure": feedback[:500],
                }
            )
            repairs_used += 1
            rnd_num += 1
            rnd_dir = art_root / f"round_{rnd_num}"
            rnd_dir.mkdir(parents=True, exist_ok=True)
            rep_prompt = (
                f"ROLE: Clean-Room Boost Builder (Gate Repair {repairs_used}/{repairs_allowed})\n"
                f"TASK REQUIREMENT:\n{task_spec}\n\n"
                f"GATE/TEST FAILURE TO FIX:\n{feedback}\n\n"
                f"STRICT WRITABLE PATHS: {', '.join(writable_paths)}\n"
                f"ACCEPTANCE COMMAND: {acceptance_cmd}\n"
            )
            b_res = invoke_provider(
                builder,
                rep_prompt,
                cwd=wt_path,
                art_dir=rnd_dir,
                tag=f"r{rnd_num}_builder",
                readonly=False,
                timeout_s=timeout_s,
                multipool_script=multipool_script,
                llm_hook=llm_hook,
            )
            _record_usage(b_res)
            gate_ok, t_rc, t_out, t_err, t_wall, t_passed, t_total, gate_err = _commit_and_gate(f"r{rnd_num}")
            builder_tok_cur = int(b_res["in_tokens"]) + int(b_res["out_tokens"])

        if not gate_ok:
            rounds_trace.append(
                {
                    "round": rnd_num,
                    "head_sha": head_sha,
                    "gate_passed": False,
                    "test_rc": t_rc,
                    "tests_passed": t_passed,
                    "adversary_verdict": "SKIPPED_ON_GATE_FAIL",
                    "builder_tokens": builder_tok_cur,
                    "adversary_tokens": 0,
                    "failure": f"Gate failed (rc={t_rc}): {gate_err}"[:500],
                }
            )
        else:
            # Step 2: <=max_repairs loop: Adversary -> ReproCheck -> Repair -> Gate
            while True:
                diff_text = _git(wt_path, "diff", "--no-renames", f"{base_sha}..{head_sha}").stdout
                if in_place and not diff_text.strip():
                    diff_text = _git(wt_path, "diff", "--no-renames").stdout
                raw_test_log = f"EXIT_CODE={t_rc} PASSED={t_passed}/{t_total}\n{t_out}\n{t_err}"[-6000:]

                adv_prompt = (
                    "ROLE: Cross-Family Blind Adversary (Zero Author Narrative Provided)\n"
                    "You receive ONLY (1) requirement, (2) raw git diff, and (3) raw test output.\n"
                    "Probe boundary cases (empty, bool/int, unhashable, zero/negative, NaN/Inf, type mismatch, concurrency).\n\n"
                    f"1. REQUIREMENT:\n{task_spec}\nAcceptance command: {acceptance_cmd}\n\n"
                    f"2. RAW GIT DIFF:\n```diff\n{diff_text[:12000]}\n```\n\n"
                    f"3. RAW TEST LOG:\n```text\n{raw_test_log}\n```\n\n"
                    "Return STRICT JSON ONLY:\n"
                    '{"verdict": "PASS"|"BROKEN", "severity": "BLOCKER"|"MAJOR"|"MINOR"|"NONE", '
                    '"findings": ["concise issue"], "repro_python": "optional python snippet that exits non-zero on defect, else empty string"}'
                )
                a_res = invoke_provider(
                    adversary,
                    adv_prompt,
                    cwd=wt_path,
                    art_dir=rnd_dir,
                    tag=f"r{rnd_num}_adversary",
                    readonly=True,
                    timeout_s=timeout_s,
                    multipool_script=multipool_script,
                    llm_hook=llm_hook,
                )
                _record_usage(a_res)
                adv_tok = int(a_res["in_tokens"]) + int(a_res["out_tokens"])
                adversary_tokens += adv_tok

                parsed = gates.parse_structured_verdict(a_res["response"], mode="boost")
                findings_txt = "; ".join(parsed.findings) if parsed.findings else a_res["response"].strip()[-1200:]
                repro_rc: int | None = None
                repro_out = ""

                # Pass on explicit PASS or MINOR severity
                if a_res["exit_code"] == 0 and (parsed.verdict == "PASS" or parsed.severity == "MINOR"):
                    rounds_trace.append(
                        {
                            "round": rnd_num,
                            "head_sha": head_sha,
                            "gate_passed": True,
                            "test_rc": t_rc,
                            "tests_passed": t_passed,
                            "test_wall_sec": round(t_wall, 3),
                            "adversary_verdict": "PASS_MINOR" if (parsed.severity == "MINOR" and parsed.verdict != "PASS") else "PASS",
                            "severity": parsed.severity,
                            "counterexample": findings_txt[:600] if parsed.severity == "MINOR" else "",
                            "builder_tokens": builder_tok_cur,
                            "adversary_tokens": adv_tok,
                        }
                    )
                    final_status = "PASSED"
                    break

                # ReproCheck: run repro_python deterministically (nice -n 10, <=10s)
                if parsed.repro_python.strip() and parsed.verdict in ("BROKEN", "FAIL"):
                    repro_rc, r_out, r_err = _run_repro_python(
                        parsed.repro_python, cwd=wt_path, timeout_s=10
                    )
                    repro_out = (r_out + "\n" + r_err).strip()
                    repro_checks.append(
                        {
                            "round": rnd_num,
                            "repro_rc": repro_rc,
                            "reproduced": repro_rc != 0,
                            "output": repro_out[:500],
                        }
                    )
                    if repro_rc == 0:
                        # Non-reproducing false alarm: discard at 0 repair tokens!
                        false_alarms_discarded += 1
                        rounds_trace.append(
                            {
                                "round": rnd_num,
                                "head_sha": head_sha,
                                "gate_passed": True,
                                "test_rc": t_rc,
                                "tests_passed": t_passed,
                                "test_wall_sec": round(t_wall, 3),
                                "adversary_verdict": "FALSE_ALARM_DISCARDED",
                                "severity": parsed.severity,
                                "repro_rc": repro_rc,
                                "counterexample": findings_txt[:600],
                                "builder_tokens": builder_tok_cur,
                                "adversary_tokens": adv_tok,
                            }
                        )
                        final_status = "PASSED"
                        break

                rounds_trace.append(
                    {
                        "round": rnd_num,
                        "head_sha": head_sha,
                        "gate_passed": True,
                        "test_rc": t_rc,
                        "tests_passed": t_passed,
                        "test_wall_sec": round(t_wall, 3),
                        "adversary_verdict": parsed.verdict,
                        "severity": parsed.severity,
                        "repro_rc": repro_rc,
                        "counterexample": findings_txt[:600],
                        "builder_tokens": builder_tok_cur,
                        "adversary_tokens": adv_tok,
                    }
                )

                if repairs_used >= repairs_allowed:
                    final_status = "FAILED"
                    break

                # Repair -> Gate
                repairs_used += 1
                rnd_num += 1
                rnd_dir = art_root / f"round_{rnd_num}"
                rnd_dir.mkdir(parents=True, exist_ok=True)
                repro_note = f"\nRepro output (rc={repro_rc}):\n{repro_out[:1200]}\n" if repro_rc is not None else ""
                rep_prompt = (
                    f"ROLE: Clean-Room Boost Repair ({repairs_used}/{repairs_allowed})\n"
                    f"TASK REQUIREMENT:\n{task_spec}\n\n"
                    f"CONFIRMED ADVERSARIAL FINDING ({adversary.label()}, severity={parsed.severity}):\n"
                    f"{findings_txt}\n{repro_note}\n"
                    f"STRICT WRITABLE PATHS: {', '.join(writable_paths)}\n"
                    f"ACCEPTANCE COMMAND: {acceptance_cmd}\n"
                    "Fix the defect cleanly, add a regression test for this counterexample, and keep acceptance green."
                )
                b_res = invoke_provider(
                    builder,
                    rep_prompt,
                    cwd=wt_path,
                    art_dir=rnd_dir,
                    tag=f"r{rnd_num}_builder",
                    readonly=False,
                    timeout_s=timeout_s,
                    multipool_script=multipool_script,
                    llm_hook=llm_hook,
                )
                _record_usage(b_res)
                builder_tok_cur = int(b_res["in_tokens"]) + int(b_res["out_tokens"])
                gate_ok, t_rc, t_out, t_err, t_wall, t_passed, t_total, gate_err = _commit_and_gate(f"r{rnd_num}")
                if not gate_ok:
                    rounds_trace.append(
                        {
                            "round": rnd_num,
                            "head_sha": head_sha,
                            "gate_passed": False,
                            "test_rc": t_rc,
                            "tests_passed": t_passed,
                            "adversary_verdict": "REPAIR_GATE_FAILED",
                            "builder_tokens": builder_tok_cur,
                            "adversary_tokens": 0,
                            "failure": f"Repair gate failed (rc={t_rc}): {gate_err}"[:500],
                        }
                    )
                    if repairs_used >= repairs_allowed:
                        final_status = "FAILED"
                        break
                    continue

                if repairs_used >= repairs_allowed:
                    rounds_trace.append(
                        {
                            "round": rnd_num,
                            "head_sha": head_sha,
                            "gate_passed": True,
                            "test_rc": t_rc,
                            "tests_passed": t_passed,
                            "test_wall_sec": round(t_wall, 3),
                            "adversary_verdict": "REPAIRED_AND_GATED",
                            "builder_tokens": builder_tok_cur,
                            "adversary_tokens": 0,
                        }
                    )
                    final_status = "PASSED"
                    break

        with db._tx() as (conn, pending):
            db._record_event(
                conn,
                pending,
                "X_BOOST_COMPLETED",
                payload={
                    "run_id": run_id,
                    "project": project,
                    "status": final_status,
                    "builder": builder.to_dict(),
                    "adversary": adversary.to_dict(),
                    "rounds_used": len(rounds_trace),
                    "head_sha": head_sha,
                    "uncached_tokens": total_in + total_out,
                    "cache_read_tokens": total_cache,
                },
            )
        summary = {
            "mode": "x-boost",
            "run_id": run_id,
            "status": final_status,
            "builder": builder.to_dict(),
            "adversary": adversary.to_dict(),
            "provider_pairing": f"{builder.label()} <-> {adversary.label()}",
            "is_real_llm": bool(llm_is_real_flags and all(llm_is_real_flags)),
            "hq_jobs_submitted": hq_jobs_submitted,
            "raw_usage_paths": raw_usage_paths,
            "cross_family_verified": builder.family != adversary.family,
            "worktree_path": str(wt_path),
            "branch_name": branch_name if not in_place else None,
            "base_sha": base_sha,
            "head_sha": head_sha,
            "rounds_used": len(rounds_trace),
            "max_rounds": max_iters,
            "max_repairs": repairs_allowed,
            "repairs_used": repairs_used,
            "false_alarms_discarded": false_alarms_discarded,
            "repro_checks": repro_checks,
            "uncached_in_tokens": total_in,
            "uncached_out_tokens": total_out,
            "uncached_total_tokens": total_in + total_out,
            "cache_read_tokens": total_cache,
            "coordination_tax_pct": round(
                (adversary_tokens / max(1, total_in + total_out)) * 100.0, 2
            ),
            "rounds": rounds_trace,
        }
        (art_root / "boost_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 2. Clean-Room Cross-Provider Restricted /x-swarm
# ---------------------------------------------------------------------------


class XSwarmBoard:
    """Append-only SQLite message board + @name cursor wakeup + 0-token stall watchdog."""

    def __init__(self, db: StateDB) -> None:
        self.db = db
        self._init_tables()

    def _init_tables(self) -> None:
        with self.db._lock, self.db._conn:
            self.db._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS xswarm_runs (
                    swarm_id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    title TEXT NOT NULL,
                    repo_path TEXT NOT NULL,
                    base_sha TEXT NOT NULL,
                    status TEXT NOT NULL,
                    stall_timeout_s REAL NOT NULL,
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS xswarm_members (
                    swarm_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    role TEXT NOT NULL,
                    provider_json TEXT NOT NULL,
                    writable_paths_json TEXT NOT NULL,
                    allowed_targets_json TEXT NOT NULL,
                    acceptance_cmd TEXT NOT NULL,
                    worktree_path TEXT NOT NULL,
                    branch_name TEXT NOT NULL,
                    cursor_seq INTEGER NOT NULL DEFAULT 0,
                    turns_run INTEGER NOT NULL DEFAULT 0,
                    status TEXT NOT NULL DEFAULT 'DORMANT',
                    PRIMARY KEY (swarm_id, name)
                );
                CREATE TABLE IF NOT EXISTS xswarm_messages (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    swarm_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    mentions_json TEXT NOT NULL,
                    body TEXT NOT NULL,
                    created_ts REAL NOT NULL
                );
                """
            )

    def create_swarm(
        self,
        *,
        swarm_id: str,
        project: str,
        title: str,
        repo_path: str,
        base_sha: str,
        members: Sequence[Mapping[str, Any]],
        stall_timeout_s: float = 180.0,
    ) -> dict[str, Any]:
        """Register a multi-provider swarm after enforcing disjoint writable scopes."""
        writer_scopes: dict[str, Sequence[str]] = {}
        for m in members:
            wpaths = list(m.get("writable_paths") or [])
            if wpaths and m.get("role") != "verifier":
                writer_scopes[str(m["name"])] = wpaths
        overlaps = gates.detect_sibling_scope_overlaps(writer_scopes)
        if overlaps:
            raise ValueError(f"Disjoint write-scope violation in xswarm members: {overlaps}")

        now = time.time()
        with self.db._lock, self.db._conn:
            self.db._conn.execute(
                """
                INSERT INTO xswarm_runs(swarm_id, project, title, repo_path, base_sha, status, stall_timeout_s, created_ts, updated_ts)
                VALUES (?, ?, ?, ?, ?, 'ACTIVE', ?, ?, ?)
                """,
                (swarm_id, project, title, repo_path, base_sha, float(stall_timeout_s), now, now),
            )
            for m in members:
                self.db._conn.execute(
                    """
                    INSERT INTO xswarm_members(
                        swarm_id, name, role, provider_json, writable_paths_json,
                        allowed_targets_json, acceptance_cmd, worktree_path, branch_name, cursor_seq, turns_run, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, 0, 'DORMANT')
                    """,
                    (
                        swarm_id,
                        str(m["name"]),
                        str(m.get("role") or "worker"),
                        json.dumps(m["provider"]),
                        json.dumps(list(m.get("writable_paths") or [])),
                        json.dumps(list(m.get("allowed_targets") or ["verifier", "coordinator"])),
                        str(m.get("acceptance_cmd") or "true"),
                        str(m.get("worktree_path") or ""),
                        str(m.get("branch_name") or ""),
                    ),
                )
        return {"swarm_id": swarm_id, "members": len(members)}

    def post(
        self,
        swarm_id: str,
        sender: str,
        body: str,
        *,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Append a message enforcing <=4KB cap, no worker @all broadcast, and allowed peer edges."""
        ts = float(now if now is not None else time.time())
        trimmed = (body or "").strip()
        if not trimmed:
            raise ValueError("Empty board message")
        if len(trimmed) > MAX_BOARD_MSG_CHARS:
            trimmed = trimmed[: MAX_BOARD_MSG_CHARS - 20] + "\n...[truncated]"

        mentions = list(dict.fromkeys(m.lower() for m in MENTION_RE.findall(trimmed)))
        with self.db._lock, self.db._conn:
            sender_row = self.db._conn.execute(
                "SELECT role, allowed_targets_json FROM xswarm_members WHERE swarm_id = ? AND name = ?",
                (swarm_id, sender),
            ).fetchone()
            if sender != "coordinator":
                if "all" in mentions:
                    raise ValueError(f"Broadcast @all is forbidden for worker '{sender}' (anti-token-storm rule)")
                if sender_row is not None and sender_row["role"] == "worker":
                    allowed = set(json.loads(sender_row["allowed_targets_json"]))
                    illegal = [t for t in mentions if t not in allowed and t != sender]
                    if illegal:
                        raise ValueError(
                            f"Worker '{sender}' attempted forbidden lateral chat @{illegal}; allowed={sorted(allowed)}"
                        )

            cur = self.db._conn.execute(
                """
                INSERT INTO xswarm_messages(swarm_id, sender, mentions_json, body, created_ts)
                VALUES (?, ?, ?, ?, ?)
                """,
                (swarm_id, sender, json.dumps(mentions), trimmed, ts),
            )
            seq = int(cur.lastrowid)
            self.db._conn.execute(
                "UPDATE xswarm_runs SET updated_ts = ? WHERE swarm_id = ?",
                (ts, swarm_id),
            )
        return {"seq": seq, "sender": sender, "mentions": mentions, "created_ts": ts}

    def pending_wakeups(self, swarm_id: str) -> list[dict[str, Any]]:
        """Return dormant members who have unread @mentions after their cursor_seq."""
        with self.db._lock:
            members = self.db._conn.execute(
                "SELECT * FROM xswarm_members WHERE swarm_id = ? ORDER BY name ASC",
                (swarm_id,),
            ).fetchall()
            msgs = self.db._conn.execute(
                "SELECT * FROM xswarm_messages WHERE swarm_id = ? ORDER BY seq ASC",
                (swarm_id,),
            ).fetchall()

        wakeups: list[dict[str, Any]] = []
        for m in members:
            name = m["name"].lower()
            cursor = int(m["cursor_seq"])
            unread_mentions = [
                dict(row)
                for row in msgs
                if int(row["seq"]) > cursor
                and (name in json.loads(row["mentions_json"]) or "all" in json.loads(row["mentions_json"]))
                and row["sender"].lower() != name
            ]
            if unread_mentions:
                all_since_cursor = [dict(row) for row in msgs if int(row["seq"]) > cursor]
                wakeups.append({"member": dict(m), "unread": all_since_cursor})
        return wakeups

    def advance_cursor(self, swarm_id: str, name: str, seq: int) -> None:
        with self.db._lock, self.db._conn:
            self.db._conn.execute(
                """
                UPDATE xswarm_members
                SET cursor_seq = MAX(cursor_seq, ?), turns_run = turns_run + 1, status = 'DORMANT'
                WHERE swarm_id = ? AND name = ?
                """,
                (int(seq), swarm_id, name),
            )

    def check_stall_zero_token(self, swarm_id: str, *, now: float | None = None) -> dict[str, Any]:
        """Pure-Python 0-LLM-token stall detector."""
        ts = float(now if now is not None else time.time())
        with self.db._lock:
            run = self.db._conn.execute(
                "SELECT * FROM xswarm_runs WHERE swarm_id = ?",
                (swarm_id,),
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown swarm_id: {swarm_id}")
            last_msg = self.db._conn.execute(
                "SELECT * FROM xswarm_messages WHERE swarm_id = ? ORDER BY seq DESC LIMIT 1",
                (swarm_id,),
            ).fetchone()
        pending = self.pending_wakeups(swarm_id)
        last_ts = float(last_msg["created_ts"]) if last_msg else float(run["created_ts"])
        idle_s = max(0.0, ts - last_ts)
        stalled = run["status"] == "ACTIVE" and len(pending) == 0 and idle_s >= float(run["stall_timeout_s"])
        return {
            "swarm_id": swarm_id,
            "status": run["status"],
            "stalled": stalled,
            "idle_sec": round(idle_s, 3),
            "pending_wakeups": [w["member"]["name"] for w in pending],
            "monitor_tokens_used": 0,
        }

    def mark_status(self, swarm_id: str, status: str) -> None:
        with self.db._lock, self.db._conn:
            self.db._conn.execute(
                "UPDATE xswarm_runs SET status = ?, updated_ts = ? WHERE swarm_id = ?",
                (status, time.time(), swarm_id),
            )

    def snapshot(self, swarm_id: str) -> dict[str, Any]:
        with self.db._lock:
            run = self.db._conn.execute(
                "SELECT * FROM xswarm_runs WHERE swarm_id = ?", (swarm_id,)
            ).fetchone()
            if run is None:
                raise KeyError(f"Unknown swarm_id: {swarm_id}")
            members = [
                dict(r)
                for r in self.db._conn.execute(
                    "SELECT * FROM xswarm_members WHERE swarm_id = ? ORDER BY name ASC",
                    (swarm_id,),
                ).fetchall()
            ]
            messages = [
                dict(r)
                for r in self.db._conn.execute(
                    "SELECT * FROM xswarm_messages WHERE swarm_id = ? ORDER BY seq ASC",
                    (swarm_id,),
                ).fetchall()
            ]
        return {"run": dict(run), "members": members, "messages": messages}


def run_xswarm(
    *,
    repo_path: str | Path,
    project: str,
    title: str,
    spec_workers: Sequence[Mapping[str, Any]],
    verifier_spec: str | None = None,
    verifier_cmd: str = "pytest -q",
    max_ticks: int = 8,
    stall_timeout_s: float = 180.0,
    timeout_s: int = 600,
    use_hq: bool = True,
    multipool_script: Path | None = None,
    arr_home: str | Path | None = None,
    sandbox_manager: SandboxManager | None = None,
    llm_hook: LLMHook | None = None,
    goal: Any = None,
    hyp: Any = None,
    workflow_id: str | None = None,
    wf_config: Mapping[str, Any] | None = None,
    task: Any = None,
    model_router: Any = None,
) -> dict[str, Any]:
    """Run a bounded multi-provider x-swarm with isolated worktrees and @verifier handoff."""
    home = resolve_arr_home(arr_home)
    db = StateDB(home)
    arbiter = ResourceArbiter()
    board = XSwarmBoard(db)
    swarm_id = f"xswarm-{uuid.uuid4().hex[:8]}"
    art_root = home / "artifacts" / "x_swarm" / swarm_id
    art_root.mkdir(parents=True, exist_ok=True)

    try:
        from conjectra.types import ScientificRole

        # Resolve all worker & verifier routes BEFORE creating worktrees so policy blocks fail closed with 0 side effects!
        resolved_worker_provs: list[ProviderSpec] = []
        worker_families: list[str] = []
        for w in spec_workers:
            prov = _resolve_multiagent_provider(
                spec=w.get("provider"),
                sci_role=ScientificRole.ENGINEER_BUILDER,
                legacy_role="builder",
                arbiter=arbiter,
                home=home,
                db=db,
                goal=goal,
                hyp=hyp,
                workflow_id=workflow_id,
                wf_config=wf_config,
                task=task,
                model_router=model_router,
                pinned_route=w.get("pinned_route") if isinstance(w.get("pinned_route"), Mapping) else None,
            )
            resolved_worker_provs.append(prov)
            if prov.family not in worker_families:
                worker_families.append(prov.family)

        v_prov = _resolve_multiagent_provider(
            spec=verifier_spec,
            sci_role=ScientificRole.IMPL_REVIEWER,
            legacy_role="adversary",
            arbiter=arbiter,
            home=home,
            db=db,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            wf_config=wf_config,
            task=task,
            model_router=model_router,
            avoid_families=worker_families[:1] if len(worker_families) > 1 else worker_families,
            prefer_avoid_families=worker_families,
        )

        repo = Path(repo_path).expanduser().resolve()
        base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
        sb = sandbox_manager or SandboxManager()
        mirror = sb.ensure_mirror(repo, project)
        integration_wt = sb.create_worktree(
            mirror, project, f"arr/{project}/{swarm_id}-Verifier", base_sha, origin_repo=repo
        )

        members_cfg: list[dict[str, Any]] = []
        for idx, w in enumerate(spec_workers):
            w_name = str(w["name"]).lower()
            prov = resolved_worker_provs[idx]
            br = f"arr/{project}/{swarm_id}-{w_name}"
            wt = sb.create_worktree(mirror, project, br, base_sha, origin_repo=repo)
            allowed = list(w.get("allowed_targets") or ["verifier", "coordinator"])
            members_cfg.append(
                {
                    "name": w_name,
                    "role": "worker",
                    "provider": prov.to_dict(),
                    "writable_paths": list(w["writable_paths"]),
                    "allowed_targets": allowed,
                    "acceptance_cmd": str(w.get("acceptance_cmd") or verifier_cmd),
                    "worktree_path": str(wt),
                    "branch_name": br,
                    "brief": str(w["brief"]),
                }
            )
        all_worker_names = [m["name"] for m in members_cfg]
        members_cfg.append(
            {
                "name": "verifier",
                "role": "verifier",
                "provider": v_prov.to_dict(),
                "writable_paths": [],
                "allowed_targets": [*all_worker_names, "coordinator"],
                "acceptance_cmd": verifier_cmd,
                "worktree_path": str(integration_wt),
                "branch_name": f"arr/{project}/{swarm_id}-Verifier",
                "brief": "Merge worker branches onto integration worktree, run verifier_cmd, and audit interface alignment.",
            }
        )

        board.create_swarm(
            swarm_id=swarm_id,
            project=project,
            title=title,
            repo_path=str(repo),
            base_sha=base_sha,
            members=members_cfg,
            stall_timeout_s=stall_timeout_s,
        )

        hq: HQRuntime | None = None
        if use_hq and llm_hook is None:
            hq = HQRuntime(home)
            hq.ensure_running()

        # Seed initial @mentions from coordinator (no @all needed; targeted briefs)
        for m in members_cfg:
            if m["role"] == "worker":
                board.post(
                    swarm_id,
                    "coordinator",
                    f"@{m['name']} TASK: {m['brief']}\n"
                    f"Writable paths: {', '.join(m['writable_paths'])}\n"
                    f"Allowed @mentions ONLY: {', '.join('@' + t for t in m['allowed_targets'])}. "
                    "When ready for integration/interface check, @mention your allowed downstream or @verifier.",
                )

        total_in = 0
        total_out = 0
        total_cache = 0
        board_overhead_tokens = 0
        hq_jobs_submitted: list[int] = []
        raw_usage_paths: list[str] = []
        llm_is_real_flags: list[bool] = []
        final_status = "INCOMPLETE"

        for tick in range(1, max_ticks + 1):
            wakeups = board.pending_wakeups(swarm_id)
            if not wakeups:
                stall = board.check_stall_zero_token(swarm_id)
                if stall["stalled"]:
                    final_status = "STALLED"
                break

            for item in wakeups:
                mem_row = item["member"]
                unread = item["unread"]
                m_name = mem_row["name"]
                prov = ProviderSpec(**json.loads(mem_row["provider_json"]))
                wt = Path(mem_row["worktree_path"])
                wpaths = json.loads(mem_row["writable_paths_json"])
                allowed = json.loads(mem_row["allowed_targets_json"])
                max_seq = max(int(u["seq"]) for u in unread)

                board_context = "\n".join(
                    f"[#{u['seq']} {u['sender']}] {u['body']}" for u in unread
                )
                board_overhead_tokens += max(1, len(board_context) // 4)

                if mem_row["role"] == "verifier":
                    # Deterministically merge all worker branches into integration_wt and run verifier_cmd
                    _git(wt, "reset", "--hard", base_sha)
                    merge_errors: list[str] = []
                    for wm in members_cfg:
                        if wm["role"] != "worker":
                            continue
                        w_head = _git(Path(wm["worktree_path"]), "rev-parse", "HEAD").stdout.strip()
                        if w_head and w_head != base_sha:
                            _git(mirror, "fetch", str(wm["worktree_path"]), f"{w_head}:refs/heads/{wm['branch_name']}-sync")
                            m_proc = _git(
                                wt,
                                "-c",
                                "user.name=conjectra",
                                "-c",
                                "user.email=conjectra@local",
                                "merge",
                                "--no-edit",
                                w_head,
                            )
                            if m_proc.returncode != 0:
                                merge_errors.append(f"{wm['name']}: {m_proc.stderr.strip()}")
                                _git(wt, "merge", "--abort")

                    if hq is not None:
                        jid = hq.submit_task(
                            task_id=f"{swarm_id}-verify-t{tick}",
                            command=["/usr/bin/nice", "-n", "10", "/bin/bash", "-lc", verifier_cmd],
                            cwd=wt,
                            stage=StageKind.CPU_GATE,
                            resources=ResourceSpec(cpus=2, ram_mb=2048, wall_timeout_s=min(300, timeout_s)),
                        )
                        hq_jobs_submitted.append(jid)
                        hq.wait_job(jid, timeout_s=min(300.0, float(timeout_s)))

                    v_rc, v_out, v_err, _vw, _vr, _vp, v_passed, v_total = _run_nice_test(
                        verifier_cmd, cwd=wt, timeout_s=min(300, timeout_s)
                    )
                    diff_txt = _git(wt, "diff", f"{base_sha}..HEAD").stdout
                    v_prompt = (
                        f"ROLE: x-swarm Cross-Family Integration Verifier (Tick {tick})\n"
                        f"Unread board messages:\n{board_context}\n\n"
                        f"Merge errors: {merge_errors}\n"
                        f"Integration test ({verifier_cmd}): rc={v_rc}, passed={v_passed}/{v_total}\n"
                        f"Raw test output:\n{(v_out + chr(10) + v_err)[-3500:]}\n"
                        f"Combined git diff:\n```diff\n{diff_txt[:10000]}\n```\n\n"
                        f"If merge_errors is empty, rc==0, passed>0, and interfaces match, reply `@coordinator [done] AUDIT_VERDICT: PASS`.\n"
                        f"Otherwise @mention ONLY the specific responsible worker(s) from {all_worker_names} with exact fix needed and `AUDIT_VERDICT: FAIL`."
                    )
                    res = invoke_provider(
                        prov,
                        v_prompt,
                        cwd=wt,
                        art_dir=art_root,
                        tag=f"t{tick}_verifier",
                        readonly=True,
                        timeout_s=timeout_s,
                        multipool_script=multipool_script,
                        llm_hook=llm_hook,
                    )
                    total_in += res["in_tokens"]
                    total_out += res["out_tokens"]
                    total_cache += int(res.get("cache_tokens", 0))
                    if res.get("usage_out") and res["usage_out"] not in raw_usage_paths:
                        raw_usage_paths.append(res["usage_out"])
                    llm_is_real_flags.append(bool(res.get("is_real_llm", False)))
                    board.advance_cursor(swarm_id, m_name, max_seq)
                    reply_txt = res["response"].strip() or "@coordinator [done] AUDIT_VERDICT: FAIL"
                    board.post(swarm_id, "verifier", reply_txt)

                    parsed_aud = gates.parse_structured_verdict(reply_txt, mode="audit")
                    if not merge_errors and v_rc == 0 and v_passed > 0 and parsed_aud.verdict == "PASS":
                        final_status = "PASSED"
                        board.mark_status(swarm_id, "PASSED")
                        break
                else:
                    w_prompt = (
                        f"ROLE: x-swarm Worker `{m_name}` (Tick {tick})\n"
                        f"Unread cursor messages:\n{board_context}\n\n"
                        f"STRICT WRITABLE PATHS: {', '.join(wpaths)}\n"
                        f"ALLOWED @MENTIONS ONLY: {', '.join('@' + t for t in allowed)} (NEVER @all!)\n"
                        "Implement your slice in your isolated worktree, verify locally, and end your response "
                        "with a concise handoff naming the next allowed @target (e.g. `@verifier [done] interface summary`)."
                    )
                    res = invoke_provider(
                        prov,
                        w_prompt,
                        cwd=wt,
                        art_dir=art_root,
                        tag=f"t{tick}_{m_name}",
                        readonly=False,
                        timeout_s=timeout_s,
                        multipool_script=multipool_script,
                        llm_hook=llm_hook,
                    )
                    total_in += res["in_tokens"]
                    total_out += res["out_tokens"]
                    total_cache += int(res.get("cache_tokens", 0))
                    if res.get("usage_out") and res["usage_out"] not in raw_usage_paths:
                        raw_usage_paths.append(res["usage_out"])
                    llm_is_real_flags.append(bool(res.get("is_real_llm", False)))
                    _git(wt, "add", "--", *wpaths)
                    if _git(wt, "diff", "--cached", "--name-only").stdout.strip():
                        _git(
                            wt,
                            "-c",
                            "user.name=conjectra",
                            "-c",
                            "user.email=conjectra@local",
                            "commit",
                            "-m",
                            f"xswarm({swarm_id}): {m_name} tick {tick}",
                        )
                    board.advance_cursor(swarm_id, m_name, max_seq)

                    # Sanitize worker post so a sloppy model forgetting @target still routes to allowed downstream/verifier
                    raw_reply = res["response"].strip()[-1200:]
                    found_mentions = [x.lower() for x in MENTION_RE.findall(raw_reply) if x.lower() != "all"]
                    valid_mentions = [x for x in found_mentions if x in allowed]
                    if not valid_mentions:
                        default_next = next((a for a in allowed if a != "coordinator"), "verifier")
                        post_body = f"@{default_next} [done] {raw_reply}"
                    else:
                        cleaned_reply = re.sub(r"@all\b", "", raw_reply, flags=re.IGNORECASE)
                        post_body = f"@{valid_mentions[0]} {cleaned_reply}"
                    board.post(swarm_id, m_name, post_body)

            if final_status == "PASSED":
                break

        snap = board.snapshot(swarm_id)
        stall_check = board.check_stall_zero_token(swarm_id)
        integ_head = _git(integration_wt, "rev-parse", "HEAD").stdout.strip() or base_sha
        worker_labels = sorted({m["provider"]["pool"] + ":" + m["provider"]["model"] for m in members_cfg if m["role"] == "worker"})
        summary = {
            "mode": "x-swarm",
            "swarm_id": swarm_id,
            "status": final_status,
            "integration_worktree": str(integration_wt),
            "head_sha": integ_head,
            "provider_pairing": f"{'+'.join(worker_labels)} -> {v_prov.label()}",
            "is_real_llm": bool(llm_is_real_flags and all(llm_is_real_flags)),
            "raw_usage_paths": raw_usage_paths,
            "hq_jobs_submitted": hq_jobs_submitted,
            "messages_count": len(snap["messages"]),
            "uncached_in_tokens": total_in,
            "uncached_out_tokens": total_out,
            "uncached_total_tokens": total_in + total_out,
            "cache_read_tokens": total_cache,
            "board_overhead_tokens": board_overhead_tokens,
            "coordination_tax_pct": round(
                (board_overhead_tokens / max(1, total_in + total_out)) * 100.0, 2
            ),
            "zero_token_stall_watchdog": stall_check,
            "members": [
                {
                    "name": r["name"],
                    "role": r["role"],
                    "provider": json.loads(r["provider_json"]),
                    "turns_run": r["turns_run"],
                    "cursor_seq": r["cursor_seq"],
                }
                for r in snap["members"]
            ],
        }
        (art_root / "xswarm_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 3. Clean-Room Cross-Provider Milestone /x-teamwork
# ---------------------------------------------------------------------------


def build_milestone_anchor(
    *,
    phase_idx: int,
    phase_name: str,
    staging_head: str,
    slice_results: Sequence[Mapping[str, Any]],
    prescreen_note: str,
) -> str:
    """Build deterministic <4KB milestone handoff anchor after destroying phase worker contexts."""
    lines = [
        f"# Milestone {phase_idx}: {phase_name}",
        f"- Staging HEAD: `{staging_head[:12]}`",
        f"- Slices merged: `{len(slice_results)}`",
    ]
    for sr in slice_results:
        lines.append(
            f"  - `{sr['name']}` ({sr['provider']}): `{sr['head_sha'][:10]}` "
            f"paths={sr['writable_paths']} passed={sr['tests_passed']}"
        )
    if prescreen_note:
        lines.append(f"- L1 Pre-screen: {prescreen_note[:600]}")
    anchor = "\n".join(lines) + "\n"
    encoded = anchor.encode("utf-8")
    if len(encoded) > MAX_MILESTONE_ANCHOR_BYTES:
        anchor = encoded[: MAX_MILESTONE_ANCHOR_BYTES - 24].decode("utf-8", errors="ignore") + "\n...[capped<4KB]\n"
    return anchor


def run_xteamwork(
    *,
    repo_path: str | Path,
    project: str,
    title: str,
    phases: Sequence[Mapping[str, Any]],
    final_acceptance_cmd: str,
    banned_paths: Sequence[str] = (),
    prescreener_spec: str | None = None,
    auditor_spec: str | None = None,
    timeout_s: int = 600,
    use_hq: bool = True,
    multipool_script: Path | None = None,
    arr_home: str | Path | None = None,
    sandbox_manager: SandboxManager | None = None,
    llm_hook: LLMHook | None = None,
    goal: Any = None,
    hyp: Any = None,
    workflow_id: str | None = None,
    wf_config: Mapping[str, Any] | None = None,
    task: Any = None,
    model_router: Any = None,
) -> dict[str, Any]:
    """Run clean-room multi-phase teamwork with contract lock, disjoint scopes, <4KB context flush, and 2-tier audit."""
    home = resolve_arr_home(arr_home)
    db = StateDB(home)
    arbiter = ResourceArbiter()
    tw_id = f"xtw-{uuid.uuid4().hex[:8]}"
    art_root = home / "artifacts" / "x_teamwork" / tw_id
    art_root.mkdir(parents=True, exist_ok=True)

    try:
        from conjectra.types import ScientificRole

        repo = Path(repo_path).expanduser().resolve()
        base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

        # Phase 0: Pre-flight Contract, Disjoint Write-Scope & Scoped Policy Gate BEFORE creating worktrees
        pre_slice_provs: dict[tuple[int, str], ProviderSpec] = {}
        pre_builder_families: list[str] = []
        for p_idx, phase in enumerate(phases, start=1):
            slices = list(phase.get("slices") or [])
            scope_map = {str(s["name"]): list(s["writable_paths"]) for s in slices}
            overlaps = gates.detect_sibling_scope_overlaps(scope_map)
            if overlaps:
                raise ValueError(f"Phase {p_idx} sibling write-scope overlap rejected before dispatch: {overlaps}")
            for s in slices:
                ban_chk = gates.check_no_banned_access(
                    paths=s["writable_paths"],
                    command=str(s.get("acceptance_cmd") or final_acceptance_cmd),
                    extra_bans=banned_paths,
                    repo_path=repo,
                )
                if not ban_chk.passed:
                    raise ValueError(f"Phase {p_idx} slice {s['name']} violated banned paths: {ban_chk.failures}")
                s_prov_pre = _resolve_multiagent_provider(
                    spec=s.get("provider"),
                    sci_role=ScientificRole.ENGINEER_BUILDER,
                    legacy_role="builder",
                    arbiter=arbiter,
                    home=home,
                    db=db,
                    goal=goal,
                    hyp=hyp,
                    workflow_id=workflow_id,
                    wf_config=wf_config,
                    task=task,
                    model_router=model_router,
                    pinned_route=s.get("pinned_route") if isinstance(s.get("pinned_route"), Mapping) else None,
                )
                pre_slice_provs[(p_idx, str(s["name"]))] = s_prov_pre
                if s_prov_pre.family not in pre_builder_families:
                    pre_builder_families.append(s_prov_pre.family)

        prescreener = _resolve_multiagent_provider(
            spec=prescreener_spec,
            sci_role=ScientificRole.IMPL_REVIEWER,
            legacy_role="prescreener",
            arbiter=arbiter,
            home=home,
            db=db,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            wf_config=wf_config,
            task=task,
            model_router=model_router,
        )
        pre_auditor = _resolve_multiagent_provider(
            spec=auditor_spec,
            sci_role=ScientificRole.RESULT_AUDITOR,
            legacy_role="blind_auditor",
            arbiter=arbiter,
            home=home,
            db=db,
            goal=goal,
            hyp=hyp,
            workflow_id=workflow_id,
            wf_config=wf_config,
            task=task,
            model_router=model_router,
            avoid_families=pre_builder_families[:1] if len(pre_builder_families) > 1 else pre_builder_families,
            prefer_avoid_families=pre_builder_families,
        )

        sb = sandbox_manager or SandboxManager()
        mirror = sb.ensure_mirror(repo, project)
        staging_branch = f"arr/{project}/{tw_id}-staging"
        staging_wt = sb.create_worktree(mirror, project, staging_branch, base_sha, origin_repo=repo)
        staging_head = base_sha

        hq: HQRuntime | None = None
        if use_hq and llm_hook is None:
            hq = HQRuntime(home)
            hq.ensure_running()

        milestone_anchor = ""
        raw_transcript_chars_discarded = 0
        phase_reports: list[dict[str, Any]] = []
        total_in = 0
        total_out = 0
        total_cache = 0
        gate_tokens = 0
        builder_families: set[str] = set()
        builder_labels: set[str] = set()
        raw_usage_paths: list[str] = []
        llm_is_real_flags: list[bool] = []
        hq_jobs_submitted: list[int] = []

        def _track_xtw_res(r: Mapping[str, Any]) -> None:
            nonlocal total_in, total_out, total_cache
            total_in += int(r.get("in_tokens", 0))
            total_out += int(r.get("out_tokens", 0))
            total_cache += int(r.get("cache_tokens", 0))
            up = str(r.get("usage_out") or "")
            if up and up not in raw_usage_paths:
                raw_usage_paths.append(up)
            llm_is_real_flags.append(bool(r.get("is_real_llm", False)))

        for p_idx, phase in enumerate(phases, start=1):
            p_name = str(phase.get("name") or f"phase_{p_idx}")
            slices = list(phase.get("slices") or [])
            slice_results: list[dict[str, Any]] = []

            for s in slices:
                s_name = str(s["name"])
                s_paths = list(s["writable_paths"])
                s_acc = str(s.get("acceptance_cmd") or final_acceptance_cmd)
                s_prov = pre_slice_provs[(p_idx, s_name)]
                builder_families.add(s_prov.family)
                builder_labels.add(s_prov.label())

                s_branch = f"arr/{project}/{tw_id}-p{p_idx}-{s_name}"
                s_wt = sb.create_worktree(mirror, project, s_branch, staging_head, origin_repo=staging_wt)

                # Fresh worker instance receives ONLY the <4KB milestone anchor + its narrow slice spec!
                s_prompt = (
                    f"ROLE: x-teamwork Slice Builder `{s_name}` (Phase {p_idx}: {p_name})\n"
                    f"LOCKED MILESTONE ANCHOR (<4KB, prior raw sessions destroyed):\n{milestone_anchor or '(Initial phase)'}\n\n"
                    f"SLICE SPEC:\n{s['spec']}\n"
                    f"STRICT WRITABLE PATHS: {', '.join(s_paths)}\n"
                    f"SLICE ACCEPTANCE CMD: {s_acc}\n"
                    "RULES: Never delete/weaken existing assertions; no stubs/hardcoded returns; stay strictly inside writable paths."
                )
                res = invoke_provider(
                    s_prov,
                    s_prompt,
                    cwd=s_wt,
                    art_dir=art_root / f"p{p_idx}_{s_name}",
                    tag=f"p{p_idx}_{s_name}",
                    readonly=False,
                    timeout_s=timeout_s,
                    multipool_script=multipool_script,
                    llm_hook=llm_hook,
                )
                _track_xtw_res(res)
                raw_transcript_chars_discarded += len(s_prompt) + len(res["response"])

                _git(s_wt, "add", "--", *s_paths)
                if _git(s_wt, "diff", "--cached", "--name-only").stdout.strip():
                    _git(
                        s_wt,
                        "-c",
                        "user.name=conjectra",
                        "-c",
                        "user.email=conjectra@local",
                        "commit",
                        "-m",
                        f"xteamwork({tw_id}): p{p_idx} {s_name}",
                    )
                s_head = _git(s_wt, "rev-parse", "HEAD").stdout.strip()

                g_res = gates.run_cpu_gate(
                    s_wt,
                    staging_head,
                    s_head,
                    s_paths,
                    goal_prefixes=s_paths,
                    extra_bans=banned_paths,
                    command=s_acc,
                    branch_name=s_branch,
                    require_clean=True,
                    require_non_doc=True,
                    require_distinct_commits=True,
                )
                rc, out, err, _w, _r, _ps, passed, total, gjid = _run_gate_test(
                    s_acc,
                    cwd=s_wt,
                    timeout_s=min(300, timeout_s),
                    hq=hq,
                    task_id=f"{tw_id}-p{p_idx}-{s_name}-gate",
                    art_dir=art_root,
                )
                if gjid is not None:
                    hq_jobs_submitted.append(int(gjid))
                if not g_res.passed or rc != 0 or passed <= 0:
                    sb.cleanup_worktree(mirror, s_wt, branch_name=s_branch, delete_branch=False)
                    return {
                        "mode": "x-teamwork",
                        "teamwork_id": tw_id,
                        "status": "FAILED_SLICE_GATE",
                        "failed_phase": p_idx,
                        "failed_slice": s_name,
                        "gate_failures": g_res.failures,
                        "test_rc": rc,
                        "tests_passed": passed,
                        "staging_worktree": str(staging_wt),
                        "staging_head": staging_head,
                        "provider_pairing": f"{'+'.join(sorted(builder_labels))} -> {prescreener.label()}",
                        "is_real_llm": bool(llm_is_real_flags and all(llm_is_real_flags)),
                        "raw_usage_paths": raw_usage_paths,
                        "hq_jobs_submitted": hq_jobs_submitted,
                        "uncached_total_tokens": total_in + total_out,
                        "cache_read_tokens": total_cache,
                        "coordination_tax_pct": 0.0,
                        "context_compression_ratio": 1.0,
                    }

                # Merge non-overlapping slice onto staging branch & immediately destroy slice worktree/context
                _git(mirror, "fetch", str(s_wt), f"{s_head}:refs/heads/{s_branch}-merged")
                m_proc = _git(
                    staging_wt,
                    "-c",
                    "user.name=conjectra",
                    "-c",
                    "user.email=conjectra@local",
                    "merge",
                    "--no-edit",
                    s_head,
                )
                sb.cleanup_worktree(mirror, s_wt, branch_name=s_branch, delete_branch=False)
                if m_proc.returncode != 0:
                    return {
                        "mode": "x-teamwork",
                        "teamwork_id": tw_id,
                        "status": "MERGE_CONFLICT",
                        "failed_phase": p_idx,
                        "failed_slice": s_name,
                        "stderr": m_proc.stderr.strip(),
                        "staging_worktree": str(staging_wt),
                        "staging_head": staging_head,
                        "provider_pairing": f"{'+'.join(sorted(builder_labels))} -> {prescreener.label()}",
                        "is_real_llm": bool(llm_is_real_flags and all(llm_is_real_flags)),
                        "raw_usage_paths": raw_usage_paths,
                        "hq_jobs_submitted": hq_jobs_submitted,
                        "uncached_total_tokens": total_in + total_out,
                        "cache_read_tokens": total_cache,
                        "coordination_tax_pct": 0.0,
                        "context_compression_ratio": 1.0,
                    }
                staging_head = _git(staging_wt, "rev-parse", "HEAD").stdout.strip()
                slice_results.append(
                    {
                        "name": s_name,
                        "provider": s_prov.label(),
                        "head_sha": s_head,
                        "writable_paths": s_paths,
                        "tests_passed": passed,
                    }
                )

            # Tier 1 Cheap Read-Only Pre-Screen at end of Phase
            p_diff = _git(staging_wt, "diff", f"{base_sha}..{staging_head}").stdout
            ps_prompt = (
                f"ROLE: L1 Cheap Read-Only Pre-Screen (Phase {p_idx}: {p_name})\n"
                "Scan diff for obvious missing edge-case tests, stubs, or contract drift in <=5 lines.\n"
                f"```diff\n{p_diff[:8000]}\n```"
            )
            ps_res = invoke_provider(
                prescreener,
                ps_prompt,
                cwd=staging_wt,
                art_dir=art_root / f"p{p_idx}_prescreen",
                tag=f"p{p_idx}_prescreen",
                readonly=True,
                timeout_s=min(240, timeout_s),
                multipool_script=multipool_script,
                llm_hook=llm_hook,
            )
            _track_xtw_res(ps_res)
            ps_tok = ps_res["in_tokens"] + ps_res["out_tokens"]
            gate_tokens += ps_tok

            milestone_anchor = build_milestone_anchor(
                phase_idx=p_idx,
                phase_name=p_name,
                staging_head=staging_head,
                slice_results=slice_results,
                prescreen_note=ps_res["response"].strip(),
            )
            (art_root / f"milestone_{p_idx}_anchor.md").write_text(milestone_anchor, encoding="utf-8")
            phase_reports.append(
                {
                    "phase": p_idx,
                    "name": p_name,
                    "staging_head": staging_head,
                    "slices": slice_results,
                    "anchor_bytes": len(milestone_anchor.encode("utf-8")),
                }
            )

        # Final Consolidated Regression + Tier 2 Cross-Family Blind Audit
        f_rc, f_out, f_err, _fw, _fr, _fp, f_passed, f_total, f_jid = _run_gate_test(
            final_acceptance_cmd,
            cwd=staging_wt,
            timeout_s=min(300, timeout_s),
            hq=hq,
            task_id=f"{tw_id}-final-gate",
            art_dir=art_root,
        )
        if f_jid is not None:
            hq_jobs_submitted.append(int(f_jid))
        auditor = pre_auditor
        full_diff = _git(staging_wt, "diff", "--no-renames", f"{base_sha}..{staging_head}").stdout
        audit_prompt = (
            "ROLE: Tier-2 Cross-Family Milestone Blind Auditor (Zero Author Summary Fed)\n"
            f"PROJECT GOAL: {title}\n"
            f"FINAL ACCEPTANCE CMD: {final_acceptance_cmd} -> rc={f_rc}, passed={f_passed}/{f_total}\n"
            f"RAW TEST LOG:\n{(f_out + chr(10) + f_err)[-4000:]}\n"
            f"CONSOLIDATED GIT DIFF:\n```diff\n{full_diff[:14000]}\n```\n"
            'Return JSON `{"verdict": "PASS"|"FAIL", "severity": "BLOCKER"|"MAJOR"|"MINOR"|"NONE", "findings": [...]}` '
            "or end with `AUDIT_VERDICT: PASS` or `AUDIT_VERDICT: FAIL`."
        )
        a_res = invoke_provider(
            auditor,
            audit_prompt,
            cwd=staging_wt,
            art_dir=art_root / "tier2_blind_audit",
            tag="tier2_blind_audit",
            readonly=True,
            timeout_s=timeout_s,
            multipool_script=multipool_script,
            llm_hook=llm_hook,
        )
        _track_xtw_res(a_res)
        a_tok = a_res["in_tokens"] + a_res["out_tokens"]
        gate_tokens += a_tok

        parsed_aud = gates.parse_structured_verdict(a_res["response"] or "", mode="audit")
        audit_passed = bool(f_rc == 0 and f_passed > 0 and parsed_aud.verdict == "PASS")
        final_status = "PASSED" if audit_passed else "FAILED_BLIND_AUDIT"
        anchor_bytes = len(milestone_anchor.encode("utf-8"))

        summary = {
            "mode": "x-teamwork",
            "teamwork_id": tw_id,
            "status": final_status,
            "staging_worktree": str(staging_wt),
            "staging_branch": staging_branch,
            "base_sha": base_sha,
            "staging_head": staging_head,
            "phases_completed": len(phase_reports),
            "prescreener": prescreener.to_dict(),
            "blind_auditor": auditor.to_dict(),
            "provider_pairing": f"{'+'.join(sorted(builder_labels))} -> {prescreener.label()} -> {auditor.label()}",
            "is_real_llm": bool(llm_is_real_flags and all(llm_is_real_flags)),
            "raw_usage_paths": raw_usage_paths,
            "hq_jobs_submitted": hq_jobs_submitted,
            "cross_family_audit": auditor.family not in builder_families,
            "final_tests_passed": f_passed,
            "final_anchor_bytes": anchor_bytes,
            "raw_session_chars_flushed": raw_transcript_chars_discarded,
            "context_compression_ratio": round(
                raw_transcript_chars_discarded / max(1, anchor_bytes), 2
            ),
            "uncached_in_tokens": total_in,
            "uncached_out_tokens": total_out,
            "uncached_total_tokens": total_in + total_out,
            "cache_read_tokens": total_cache,
            "coordination_tax_pct": round((gate_tokens / max(1, total_in + total_out)) * 100.0, 2),
            "phases": phase_reports,
        }
        (art_root / "xteamwork_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return summary
    finally:
        db.close()


# ---------------------------------------------------------------------------
# 4. Notification hook & detached CLI spawn helpers
# ---------------------------------------------------------------------------


def notify_hook(
    target: str,
    subject: str,
    body: str,
    *,
    msg_type: str = "status",
    timeout_s: int = 20,
    runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
) -> dict[str, Any]:
    """Deliver a completion/status notification through a user-supplied hook.

    ``target`` is either ``file:/path/to/events.jsonl`` (append one JSON line) or a shell-quoted
    command template; ``{subject}`` and ``{type}`` are substituted and the body is sent on stdin.
    Examples: ``"notify-send {subject}"``, ``"./scripts/post_to_chat.sh --title {subject}"``.
    """
    tgt = str(target or "").strip()
    subj = str(subject or "").strip()
    if not tgt:
        raise ValueError("notify_hook target must be non-empty")
    if not subj:
        raise ValueError("notify_hook subject must be non-empty")
    text = str(body or "").strip() or subj
    if len(text) > MAX_BOARD_MSG_CHARS:
        text = text[: MAX_BOARD_MSG_CHARS - 20] + "\n...[truncated]"

    if tgt.startswith("file:"):
        path = Path(tgt[5:]).expanduser()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps({"ts": time.time(), "type": msg_type, "subject": subj, "body": text}) + "\n")
            return {"sent": True, "target": tgt, "subject": subj, "transport": "file", "returncode": 0}
        except OSError as exc:
            return {"sent": False, "target": tgt, "subject": subj, "transport": "file", "returncode": 1, "stderr": str(exc)}

    exec_fn = runner or subprocess.run
    cmd = [tok.replace("{subject}", subj).replace("{type}", msg_type) for tok in shlex.split(tgt)]
    transport = "command"
    try:
        proc = exec_fn(
            cmd,
            input=text,
            text=True,
            capture_output=True,
            timeout=max(1, int(timeout_s)),
            check=False,
        )
        return {
            "sent": proc.returncode == 0,
            "target": tgt,
            "subject": subj,
            "transport": transport,
            "returncode": proc.returncode,
            "stdout": (proc.stdout or "").strip(),
            "stderr": (proc.stderr or "").strip(),
            "cmd": cmd,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "sent": False,
            "target": tgt,
            "subject": subj,
            "transport": transport,
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc),
            "cmd": cmd,
        }


def spawn_detached_conjectra(
    argv: Sequence[str],
    *,
    arr_home: str | Path | None = None,
    log_name: str | None = None,
    notify_cmd: str | None = None,
    cwd: str | Path | None = None,
    popen_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Spawn a detached `conjectra` subprocess (`start_new_session=True`) and return immediately."""
    home = resolve_arr_home(arr_home)
    log_dir = home / "logs" / "detached"
    log_dir.mkdir(parents=True, exist_ok=True)
    fname = log_name or f"conjectra-{int(time.time())}-{uuid.uuid4().hex[:6]}.log"
    log_path = log_dir / fname

    cleaned_args: list[str] = []
    skip_next = False
    for idx, tok in enumerate(argv):
        if skip_next:
            skip_next = False
            continue
        if tok == "--detach":
            continue
        if tok.startswith(("--arr-home=", "--home=")):
            continue
        if tok in ("--arr-home", "--home"):
            skip_next = True
            continue
        cleaned_args.append(str(tok))

    cmd = [sys.executable, "-m", "conjectra.cli", "--arr-home", str(home), *cleaned_args]
    if notify_cmd and "--notify-cmd" not in cleaned_args:
        cmd.extend(["--notify-cmd", str(notify_cmd)])

    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CONJECTRA_HOME"] = str(home)
    pkg_root = str(Path(__file__).resolve().parents[1])
    env["PYTHONPATH"] = pkg_root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    launcher = popen_fn or subprocess.Popen
    with log_path.open("ab") as log_f:
        proc = launcher(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            cwd=str(cwd or home),
            start_new_session=True,
            env=env,
        )
    return {
        "detached": True,
        "pid": int(getattr(proc, "pid", 0)),
        "arr_home": str(home),
        "log_path": str(log_path),
        "cmd": cmd,
    }
