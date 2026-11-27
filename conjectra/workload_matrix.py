"""Workload-to-mode deterministic router and real-LLM empirical ablation matrix for Conjectra.

Strict anti-hallucination guarantees:
1. Zero scripted/mocked `llm_hook` callbacks or hardcoded token/pass literals in this module.
2. `workload_eval_ledger` enforces `is_real_llm IN (0, 1)`, `hq_used IN (0, 1)`, `provider_pairing`,
   and on-disk `raw_usage_paths_json` verification before any `is_real_llm=1` row can be inserted.
3. `route_workload()` queries ONLY `is_real_llm = 1` rows, ranks `(mode, provider_pairing)` pairs
   by measured hidden-holdout pass rate + defect escape rate + token yield (filtered by live pool
   quota health), and marks any unmeasured workload/mode as `UNKNOWN`.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence
import uuid

from conjectra import gates
from conjectra.arbiter import ResourceArbiter
from conjectra.hq_runtime import HQRuntime
from conjectra.model_router import ModelRouter
from conjectra.multiagent import (
    _run_gate_test,
    parse_provider_spec,
    run_boost,
    run_xswarm,
    run_xteamwork,
)
from conjectra.stage_runner import SandboxManager, _git
from conjectra.state_db import StateDB, resolve_arr_home
from conjectra.types import (
    ConstraintStrength,
    EngineeringVerdict,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ResourceSpec,
    StageKind,
    StageTask,
)

UNKNOWN = "UNKNOWN"

WORKLOAD_CLASSES = (
    "readonly_fanout",
    "disjoint_parallel",
    "tight_adversarial",
    "coupled_interface",
    "multi_milestone",
)

CANDIDATE_MODES_BY_WORKLOAD: dict[str, tuple[str, ...]] = {
    "readonly_fanout": ("run_workflow", "direct_hq"),
    "disjoint_parallel": ("direct_hq", "x-swarm"),
    "tight_adversarial": ("direct_hq", "x-boost"),
    "coupled_interface": ("direct_hq", "x-swarm"),
    "multi_milestone": ("direct_hq", "x-teamwork"),
}

ADVERSARIAL_KEYWORDS = (
    "deadlock",
    "race",
    "invariant",
    "subtle",
    "edge case",
    "overflow",
    "unhashable",
    "numerical",
    "boundary",
    "bit-exact",
    "pareto",
    "nan",
    "lock",
    "死锁",
    "并发",
    "边界",
    "反例",
    "精度",
    "对账",
)

COUPLED_KEYWORDS = (
    "negotiate",
    "co-design",
    "producer",
    "consumer",
    "protocol",
    "wire format",
    "handshake",
    "schema align",
    "联调",
    "接口对齐",
    "握手",
    "上下游",
)

REQUIRED_LEDGER_COLUMNS = {
    "eval_id",
    "workload_class",
    "mode",
    "provider_pairing",
    "is_real_llm",
    "hq_used",
    "holdout_cmd",
    "holdout_rc",
    "holdout_passed",
    "defect_escaped",
    "gate_intercepted",
    "uncached_tokens",
    "cache_read_tokens",
    "coordination_tax_pct",
    "context_compression_ratio",
    "wall_sec",
    "effective_yield_per_100k",
    "raw_usage_paths_json",
    "head_sha",
    "source",
    "created_ts",
}


@dataclass(frozen=True)
class WorkloadFeatures:
    """Observable structural coordinates of a research or engineering task."""

    readonly: bool = False
    slice_count: int = 1
    file_count: int = 1
    disjoint_scopes: bool = True
    interface_frozen: bool = True
    adversarial_risk: bool = False
    milestone_phases: int = 1
    estimated_minutes: int = 20

    def classify(self) -> str:
        if self.readonly:
            return "readonly_fanout"
        if self.milestone_phases >= 2 or self.slice_count >= 5 or self.estimated_minutes > 60:
            return "multi_milestone"
        if self.slice_count >= 2 and (not self.interface_frozen or not self.disjoint_scopes):
            return "coupled_interface"
        if self.slice_count == 1 and self.adversarial_risk:
            return "tight_adversarial"
        if self.slice_count >= 2 and self.disjoint_scopes and self.interface_frozen:
            return "disjoint_parallel"
        return "tight_adversarial" if self.adversarial_risk else "disjoint_parallel"


def extract_workload_features(
    description: str,
    *,
    writable_scopes: Sequence[Sequence[str]] | None = None,
    readonly: bool = False,
    interface_frozen: bool | None = None,
    milestone_phases: int = 1,
    estimated_minutes: int = 20,
) -> WorkloadFeatures:
    """Extract structural WorkloadFeatures deterministically with boundary validation."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    if int(milestone_phases) < 1 or int(milestone_phases) > 64:
        raise ValueError(f"milestone_phases out of bounds [1, 64]: {milestone_phases}")
    if int(estimated_minutes) < 1 or int(estimated_minutes) > 10080:
        raise ValueError(f"estimated_minutes out of bounds [1, 10080]: {estimated_minutes}")

    scopes = [list(s) for s in (writable_scopes or []) if list(s)]
    if len(scopes) > 128:
        raise ValueError(f"writable_scopes slice count out of bounds (max 128, got {len(scopes)})")

    low = description.lower()
    is_ro = readonly or any(
        k in low for k in ("read-only", "readonly", "survey", "audit only", "只读", "扫描", "调研")
    )
    if is_ro and scopes and any(len(s) > 0 for s in scopes) and interface_frozen is False:
        raise ValueError("Invalid combination: readonly=True cannot combine with coupled writable scopes")

    slice_cnt = max(1, len(scopes))
    flat_files = [p for sc in scopes for p in sc if str(p).strip()]
    file_cnt = max(1, len(set(flat_files))) if flat_files else 1

    disjoint = True
    if len(scopes) >= 2:
        scope_map = {f"s{i}": sc for i, sc in enumerate(scopes)}
        disjoint = len(gates.detect_sibling_scope_overlaps(scope_map)) == 0

    has_coupled_kw = any(k in low for k in COUPLED_KEYWORDS)
    frozen = (not has_coupled_kw) if interface_frozen is None else bool(interface_frozen)
    adv_risk = any(k in low for k in ADVERSARIAL_KEYWORDS)
    phases = int(milestone_phases)
    if any(k in low for k in ("multi-phase", "milestone", "跨阶段", "里程碑", "多轮演进")) and phases == 1:
        phases = 2

    return WorkloadFeatures(
        readonly=is_ro,
        slice_count=slice_cnt,
        file_count=file_cnt,
        disjoint_scopes=disjoint,
        interface_frozen=frozen,
        adversarial_risk=adv_risk,
        milestone_phases=phases,
        estimated_minutes=int(estimated_minutes),
    )


def ensure_eval_table(db: StateDB) -> dict[str, Any]:
    """Ensure strict v2 `workload_eval_ledger` exists, purging legacy unverified v1 rows."""
    migrated_legacy_rows = 0
    with db._lock, db._conn:
        existing_cols = {
            str(r[1])
            for r in db._conn.execute("PRAGMA table_info(workload_eval_ledger)").fetchall()
        }
        if existing_cols and not REQUIRED_LEDGER_COLUMNS.issubset(existing_cols):
            cnt_row = db._conn.execute("SELECT COUNT(*) FROM workload_eval_ledger").fetchone()
            migrated_legacy_rows = int(cnt_row[0]) if cnt_row else 0
            db._conn.execute("DROP TABLE workload_eval_ledger")

        db._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS workload_eval_ledger (
                eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workload_class TEXT NOT NULL,
                mode TEXT NOT NULL,
                provider_pairing TEXT NOT NULL,
                is_real_llm INTEGER NOT NULL CHECK (is_real_llm IN (0, 1)),
                hq_used INTEGER NOT NULL CHECK (hq_used IN (0, 1)),
                holdout_cmd TEXT NOT NULL,
                holdout_rc INTEGER NOT NULL,
                holdout_passed INTEGER NOT NULL,
                defect_escaped INTEGER NOT NULL,
                gate_intercepted INTEGER NOT NULL,
                uncached_tokens INTEGER NOT NULL,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                coordination_tax_pct REAL NOT NULL,
                context_compression_ratio REAL NOT NULL,
                wall_sec REAL NOT NULL,
                effective_yield_per_100k REAL NOT NULL,
                raw_usage_paths_json TEXT NOT NULL,
                head_sha TEXT NOT NULL DEFAULT '',
                source TEXT NOT NULL,
                created_ts REAL NOT NULL
            )
            """
        )
    return {"purged_legacy_unverified_rows": migrated_legacy_rows}


def purge_non_real_eval_rows(db: StateDB) -> int:
    """Delete any `is_real_llm = 0` rows from `workload_eval_ledger` and return deleted count."""
    ensure_eval_table(db)
    with db._lock, db._conn:
        cur = db._conn.execute("DELETE FROM workload_eval_ledger WHERE is_real_llm = 0")
        return int(cur.rowcount or 0)


DEFAULT_STRUCTURAL_PRIORS: dict[str, dict[str, Any]] = {
    "readonly_fanout": {
        "mode": "run_workflow",
        "skill": "run-workflow",
        "pairing": "gemini:gemini-flash",
        "rationale": "Read-only short fan-out (<10m): Python orchestration uses 0 coordinator tokens and enforces JSON schema.",
    },
    "disjoint_parallel": {
        "mode": "direct_hq",
        "skill": "conjectra",
        "pairing": "deepseek:deepseek-reasoner",
        "rationale": "Frozen contract + disjoint paths: 1-way HQ isolated worktree streaming achieves 0% board tax.",
    },
    "tight_adversarial": {
        "mode": "x-boost",
        "skill": "/x-boost",
        "pairing": "claude:claude-sonnet-5-5 <-> codex:gpt-5-codex",
        "rationale": "Single-point tight coupling with high subtle-bug risk: cross-family context-stripped adversary catches blind spots.",
    },
    "coupled_interface": {
        "mode": "x-swarm",
        "skill": "/x-swarm",
        "pairing": "claude:claude-sonnet-5-5+deepseek:deepseek-reasoner -> codex:gpt-5-codex",
        "rationale": "Co-evolving producer/consumer interfaces: @name cursor wakeup aligns schemas on HQ without @all storms.",
    },
    "multi_milestone": {
        "mode": "x-teamwork",
        "skill": "/x-teamwork",
        "pairing": "claude:claude-sonnet-5-5+deepseek:deepseek-reasoner -> grok:grok-code-fast -> codex:gpt-5-codex",
        "rationale": "Multi-phase evolution: pre-flight scope lock prevents merge conflicts and <4KB milestone flush prevents context rot.",
    },
}


def _extract_pools_from_pairing(pairing: str) -> list[str]:
    """Extract pool prefixes (`claude`, `codex`, `gemini`, `deepseek`, `grok`) from a pairing label."""
    pools: list[str] = []
    for token in (
        pairing.replace("<->", " ")
        .replace("->", " ")
        .replace("+", " ")
        .replace(",", " ")
        .split()
    ):
        if ":" in token:
            cand = token.split(":", 1)[0].strip()
            if cand in ("claude", "codex", "gemini", "deepseek", "grok", "deepseek-local"):
                pools.append(cand)
        elif token in ("claude", "codex", "gemini", "deepseek", "grok", "deepseek-local"):
            pools.append(token)
    return pools


def _is_pairing_healthy(pairing: str, arbiter: ResourceArbiter | None) -> bool:
    if arbiter is None:
        return True
    for pool in _extract_pools_from_pairing(pairing):
        chk = arbiter.check_llm_pool(pool)
        if not chk.allowed:
            return False
    return True


def _skill_for_mode(mode: str) -> str:
    if mode in ("x-boost", "x-swarm", "x-teamwork"):
        return f"/{mode}"
    if mode == "run_workflow":
        return "run-workflow"
    return "conjectra"


def record_workload_eval(
    db: StateDB,
    *,
    workload_class: str,
    mode: str,
    provider_pairing: str,
    is_real_llm: bool,
    hq_used: bool,
    holdout_cmd: str,
    holdout_rc: int,
    holdout_passed: bool,
    defect_escaped: bool,
    gate_intercepted: bool,
    uncached_tokens: int,
    cache_read_tokens: int = 0,
    coordination_tax_pct: float = 0.0,
    context_compression_ratio: float = 1.0,
    wall_sec: float = 0.0,
    raw_usage_paths: Sequence[str] = (),
    head_sha: str = "",
    source: str = "eval_matrix",
) -> dict[str, Any]:
    """Validate raw usage provenance and insert one evaluation observation into SQLite."""
    if workload_class not in WORKLOAD_CLASSES:
        raise ValueError(f"Unsupported workload_class: {workload_class!r}")
    if not mode or not provider_pairing:
        raise ValueError("mode and provider_pairing must be non-empty")

    verified_paths: list[str] = []
    if is_real_llm:
        if not raw_usage_paths:
            raise ValueError(
                "Refusing to record is_real_llm=1 row without raw_usage_paths on disk"
            )
        for raw_p in raw_usage_paths:
            p = Path(raw_p)
            if not p.is_file():
                raise ValueError(f"Missing raw LLM usage file for is_real_llm=1: {raw_p}")
            doc = json.loads(p.read_text(encoding="utf-8"))
            if doc.get("estimated") is True or doc.get("synthetic") is True:
                raise ValueError(
                    f"Refusing to record estimated/synthetic usage file as is_real_llm=1: {raw_p}"
                )
            verified_paths.append(str(p.resolve()))
        if int(uncached_tokens) <= 0:
            raise ValueError(
                f"Refusing to record is_real_llm=1 with uncached_tokens={uncached_tokens} <= 0"
            )
    else:
        verified_paths = [str(x) for x in raw_usage_paths]

    ensure_eval_table(db)
    yield_per_100k = (100000.0 / max(1, int(uncached_tokens))) if holdout_passed else 0.0
    row = {
        "workload_class": workload_class,
        "mode": mode,
        "provider_pairing": provider_pairing,
        "is_real_llm": int(bool(is_real_llm)),
        "hq_used": int(bool(hq_used)),
        "holdout_cmd": holdout_cmd,
        "holdout_rc": int(holdout_rc),
        "holdout_passed": int(bool(holdout_passed)),
        "defect_escaped": int(bool(defect_escaped)),
        "gate_intercepted": int(bool(gate_intercepted)),
        "uncached_tokens": int(uncached_tokens),
        "cache_read_tokens": int(cache_read_tokens),
        "coordination_tax_pct": round(float(coordination_tax_pct), 2),
        "context_compression_ratio": round(float(context_compression_ratio), 2),
        "wall_sec": round(float(wall_sec), 4),
        "effective_yield_per_100k": round(yield_per_100k, 3),
        "raw_usage_paths": verified_paths,
        "head_sha": head_sha,
        "source": source,
    }
    with db._lock, db._conn:
        db._conn.execute(
            """
            INSERT INTO workload_eval_ledger (
                workload_class, mode, provider_pairing, is_real_llm, hq_used,
                holdout_cmd, holdout_rc, holdout_passed, defect_escaped, gate_intercepted,
                uncached_tokens, cache_read_tokens, coordination_tax_pct,
                context_compression_ratio, wall_sec, effective_yield_per_100k,
                raw_usage_paths_json, head_sha, source, created_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row["workload_class"],
                row["mode"],
                row["provider_pairing"],
                row["is_real_llm"],
                row["hq_used"],
                row["holdout_cmd"],
                row["holdout_rc"],
                row["holdout_passed"],
                row["defect_escaped"],
                row["gate_intercepted"],
                row["uncached_tokens"],
                row["cache_read_tokens"],
                row["coordination_tax_pct"],
                row["context_compression_ratio"],
                row["wall_sec"],
                row["effective_yield_per_100k"],
                json.dumps(verified_paths, ensure_ascii=False),
                row["head_sha"],
                row["source"],
                time.time(),
            ),
        )
    return row


def route_workload(
    description: str,
    *,
    writable_scopes: Sequence[Sequence[str]] | None = None,
    readonly: bool = False,
    interface_frozen: bool | None = None,
    milestone_phases: int = 1,
    estimated_minutes: int = 20,
    arr_home: str | Path | None = None,
    arbiter: ResourceArbiter | None = None,
) -> dict[str, Any]:
    """Select optimal mode AND `(mode, provider_pairing)` from verified `is_real_llm=1` history."""
    feats = extract_workload_features(
        description,
        writable_scopes=writable_scopes,
        readonly=readonly,
        interface_frozen=interface_frozen,
        milestone_phases=milestone_phases,
        estimated_minutes=estimated_minutes,
    )
    w_class = feats.classify()
    prior = DEFAULT_STRUCTURAL_PRIORS[w_class]
    expected_modes = CANDIDATE_MODES_BY_WORKLOAD.get(w_class, (prior["mode"],))

    home = resolve_arr_home(arr_home)
    db = StateDB(home)
    try:
        ensure_eval_table(db)
        with db._lock:
            mode_rows = db._conn.execute(
                """
                SELECT mode,
                       COUNT(*) AS n,
                       SUM(hq_used) AS hq_runs,
                       AVG(holdout_passed) AS holdout_pass_rate,
                       AVG(defect_escaped) AS defect_escape_rate,
                       AVG(coordination_tax_pct) AS avg_tax_pct,
                       AVG(uncached_tokens) AS avg_tokens,
                       AVG(cache_read_tokens) AS avg_cache_tokens,
                       AVG(wall_sec) AS avg_wall_sec,
                       AVG(effective_yield_per_100k) AS avg_yield
                FROM workload_eval_ledger
                WHERE workload_class = ? AND is_real_llm = 1
                GROUP BY mode
                ORDER BY AVG(holdout_passed) DESC,
                         AVG(defect_escaped) ASC,
                         AVG(effective_yield_per_100k) DESC,
                         AVG(wall_sec) ASC
                """,
                (w_class,),
            ).fetchall()
            pairing_rows = db._conn.execute(
                """
                SELECT mode,
                       provider_pairing,
                       COUNT(*) AS n,
                       SUM(hq_used) AS hq_runs,
                       AVG(holdout_passed) AS holdout_pass_rate,
                       AVG(defect_escaped) AS defect_escape_rate,
                       AVG(coordination_tax_pct) AS avg_tax_pct,
                       AVG(uncached_tokens) AS avg_tokens,
                       AVG(wall_sec) AS avg_wall_sec,
                       AVG(effective_yield_per_100k) AS avg_yield
                FROM workload_eval_ledger
                WHERE workload_class = ? AND is_real_llm = 1
                GROUP BY mode, provider_pairing
                ORDER BY AVG(holdout_passed) DESC,
                         AVG(defect_escaped) ASC,
                         AVG(effective_yield_per_100k) DESC,
                         AVG(wall_sec) ASC
                """,
                (w_class,),
            ).fetchall()
    finally:
        db.close()

    empirical_modes = [dict(r) for r in mode_rows]
    empirical_pairings = [dict(r) for r in pairing_rows]
    tested_modes = {str(r["mode"]) for r in empirical_modes}
    untested_modes = [m for m in expected_modes if m not in tested_modes]
    total_real_runs = sum(int(r["n"]) for r in empirical_modes)

    chosen_mode = prior["mode"]
    chosen_pairing = prior["pairing"]
    empirical_winner: str = UNKNOWN
    quota_fallback_note = ""

    live_arbiter = arbiter or ResourceArbiter()
    passing_pairings = [
        p for p in empirical_pairings if float(p.get("holdout_pass_rate") or 0.0) > 0.0
    ]

    if passing_pairings:
        selected_row = None
        for cand in passing_pairings:
            if _is_pairing_healthy(str(cand["provider_pairing"]), live_arbiter):
                selected_row = cand
                break
        if selected_row is None:
            selected_row = passing_pairings[0]
            quota_fallback_note = "ALL_EMPIRICAL_POOLS_NEAR_LIMIT"
        elif selected_row is not passing_pairings[0]:
            quota_fallback_note = (
                f"SWITCHED_FROM_{passing_pairings[0]['provider_pairing']}_DUE_TO_QUOTA"
            )

        chosen_mode = str(selected_row["mode"])
        chosen_pairing = str(selected_row["provider_pairing"])
        empirical_winner = f"{chosen_mode} [{chosen_pairing}]"
        if len(tested_modes) >= 2:
            calibration_status = (
                f"EMPIRICALLY_CALIBRATED_ABLATION "
                f"(real_runs={total_real_runs}, modes_tested={len(tested_modes)})"
            )
        else:
            calibration_status = (
                f"PARTIAL_REAL_EMPIRICAL "
                f"(real_runs={total_real_runs}, untested_modes={untested_modes})"
            )
    else:
        calibration_status = (
            f"UNKNOWN_EMPIRICAL_USING_STRUCTURAL_PRIOR "
            f"(real_runs={total_real_runs}, untested_modes={list(expected_modes)})"
        )

    mode_status_map: dict[str, Any] = {}
    for m in expected_modes:
        found = next((r for r in empirical_modes if r["mode"] == m), None)
        mode_status_map[m] = found if found is not None else UNKNOWN

    return {
        "workload_class": w_class,
        "features": {
            "readonly": feats.readonly,
            "slice_count": feats.slice_count,
            "file_count": feats.file_count,
            "disjoint_scopes": feats.disjoint_scopes,
            "interface_frozen": feats.interface_frozen,
            "adversarial_risk": feats.adversarial_risk,
            "milestone_phases": feats.milestone_phases,
            "estimated_minutes": feats.estimated_minutes,
        },
        "recommended_mode": chosen_mode,
        "recommended_skill": _skill_for_mode(chosen_mode),
        "provider_pairing": chosen_pairing,
        "empirical_winner": empirical_winner,
        "calibration_status": calibration_status,
        "quota_fallback_note": quota_fallback_note,
        "untested_modes": untested_modes,
        "mode_status": mode_status_map,
        "empirical_leaderboard": empirical_modes,
        "empirical_pairing_leaderboard": empirical_pairings,
        "rationale": prior["rationale"],
    }


# ---------------------------------------------------------------------------
# Real HQ + Real Multi-Provider Empirical Ablation Harness (Zero Mocks)
# ---------------------------------------------------------------------------


def _init_temp_git_repo(root: Path, files: Mapping[str, str]) -> Path:
    repo = root / "origin_repo"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "-C", str(repo), "init", "-b", "main"], check=True, capture_output=True)
    for rel, content in files.items():
        fp = repo / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.name=arr-bench",
            "-c",
            "user.email=arr-bench@local",
            "commit",
            "-m",
            "seed workload repo",
        ],
        check=True,
        capture_output=True,
    )
    return repo


def _run_external_holdout_on_hq(
    *,
    hq: HQRuntime,
    worktree_path: Path,
    holdout_file: Path,
    art_dir: Path,
    tag: str,
    timeout_s: int = 180,
) -> tuple[int, str, str, int]:
    """Run unseen external holdout pytest file (outside worktree) via HyperQueue."""
    cmd = (
        f"PYTHONPATH={shlex.quote(str(worktree_path))} "
        f"{shlex.quote(sys.executable)} -m pytest -q {shlex.quote(str(holdout_file))}"
    )
    rc, out, err, _wall, _rss, _pstat, _passed, _total, jid = _run_gate_test(
        cmd,
        cwd=worktree_path.parent,
        timeout_s=timeout_s,
        hq=hq,
        task_id=f"holdout-{tag}-{uuid.uuid4().hex[:6]}",
        art_dir=art_dir,
    )
    return rc, out, err, int(jid or 0)


def _run_direct_hq_slices(
    *,
    repo_path: Path,
    project: str,
    slices: Sequence[Mapping[str, Any]],
    hq: HQRuntime,
    sb: SandboxManager,
    arr_home: Path,
    timeout_s: int = 420,
) -> dict[str, Any]:
    """Execute 1..N slices via real HQ `StageRunner` (`CODE` -> `CPU_GATE`) and merge onto integration worktree."""
    db = StateDB(arr_home)
    hq.ensure_running()
    base_sha = _git(repo_path, "rev-parse", "HEAD").stdout.strip()
    goal_id = f"eval-direct-{project}-{uuid.uuid4().hex[:8]}"
    all_prefixes = sorted({p.split("/", 1)[0] for s in slices for p in s["writable_paths"]})

    goal = ResearchGoal(
        goal_id=goal_id,
        project=project,
        title=f"Direct HQ empirical eval ({project})",
        direction="maximize",
        primary_metric="tests_passed",
        baseline_value=0.0,
        baseline_sha=base_sha,
        repo_path=str(repo_path),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["lockbox/", "sealed_eval/"],
        writable_prefixes=all_prefixes,
        max_tokens_budget=500000,
        max_concurrent_branches=max(2, len(slices)),
        adapter_name="generic",
    )
    db.save_goal(goal)

    mirror = sb.ensure_mirror(repo_path, project)
    slice_entries: list[dict[str, Any]] = []
    hq_jobs: list[int] = []
    env_overrides = {
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent),
        "CONJECTRA_SANDBOX_ROOT": str(sb.sandbox_base),
        "CONJECTRA_WORKTREE_ROOT": str(sb.worktree_base),
    }

    try:
        # 1. Submit all slice CODE tasks concurrently to HyperQueue
        router = ModelRouter()
        for idx, s in enumerate(slices, start=1):
            prov = parse_provider_spec(str(s["provider"]), role="builder")
            pinned_dict: dict[str, Any] | None = None
            hq_llm_res: str | None = None
            llm_pool_spec = prov.canonical_spec()
            if not prov.custom_cmd:
                route = router.resolve_route(
                    "engineer_builder",
                    explicit_spec=prov.canonical_spec(),
                    strength=ConstraintStrength.SOFT_FALLBACK,
                )
                pinned_dict = route.to_dict()
                hq_llm_res = route.hq_resource
                llm_pool_spec = route.canonical_spec()
            hid = f"hyp-{goal_id}-s{idx}"
            br = f"arr/{project}/{hid}"
            wt = sb.create_worktree(mirror, project, br, base_sha, origin_repo=repo_path)
            hyp = HypothesisNode(
                hypothesis_id=hid,
                goal_id=goal_id,
                parent_id=None,
                kind=HypothesisKind.COMPETING_ROUTE,
                title=f"Slice {s['name']}",
                statement=str(s["spec"]),
                mechanism=f"direct_hq::{s['name']}::{uuid.uuid4().hex[:6]}",
                fidelity=FidelityRung.F1_FAST_SCREEN,
                priority_evoi=1.0,
                writable_paths=list(s["writable_paths"]),
                params={"provider_model": prov.model, "collab_mode": "direct_hq"},
                kill_criteria=["tests_passed == 0"],
                acceptance_cmd=str(s["acceptance_cmd"]),
                screen_cmd=str(s["acceptance_cmd"]),
                confirm_cmd=None,
                branch_name=br,
                base_sha=base_sha,
            )
            hyp, _ = db.register_hypothesis(hyp)
            code_task = StageTask(
                task_id=f"tsk-{hid}-code",
                idempotency_key=f"{goal_id}:{hid}:CODE:a1",
                goal_id=goal_id,
                hypothesis_id=hid,
                stage=StageKind.CODE,
                deps=[],
                resources=ResourceSpec(
                    cpus=1,
                    ram_mb=2048,
                    llm_pool=llm_pool_spec,
                    wall_timeout_s=timeout_s,
                    hq_llm_resource=hq_llm_res,
                    pinned_route=pinned_dict,
                ),
                attempt=1,
                max_attempts=1,
                status="DISPATCHED",
                worktree_path=str(wt),
                enqueued_ts=time.time(),
                pinned_route=pinned_dict,
                model_spec=llm_pool_spec,
                hq_llm_resource=hq_llm_res or "",
            )
            db.enqueue_task(code_task)
            jid = hq.submit_task(
                code_task,
                command=[
                    sys.executable,
                    "-m",
                    "conjectra.stage_runner",
                    "--task-id",
                    code_task.task_id,
                    "--arr-home",
                    str(arr_home),
                ],
                stage=StageKind.CODE,
                cwd=str(arr_home),
                env=env_overrides,
                wrap_exit=False,
            )
            hq_jobs.append(jid)
            slice_entries.append(
                {
                    "slice": s,
                    "provider": prov,
                    "hyp_id": hid,
                    "branch": br,
                    "wt": wt,
                    "code_task_id": code_task.task_id,
                    "code_jid": jid,
                }
            )

        for entry in slice_entries:
            hq.wait_job(entry["code_jid"], timeout_s=float(timeout_s) + 30.0)

        # 2. Submit all slice CPU_GATE tasks concurrently to HyperQueue
        for entry in slice_entries:
            hid = entry["hyp_id"]
            gate_task = StageTask(
                task_id=f"tsk-{hid}-gate",
                idempotency_key=f"{goal_id}:{hid}:CPU_GATE:a1",
                goal_id=goal_id,
                hypothesis_id=hid,
                stage=StageKind.CPU_GATE,
                deps=[entry["code_task_id"]],
                resources=ResourceSpec(cpus=2, ram_mb=2048, wall_timeout_s=min(90, timeout_s)),
                attempt=1,
                max_attempts=1,
                status="DISPATCHED",
                worktree_path=str(entry["wt"]),
                enqueued_ts=time.time(),
            )
            db.enqueue_task(gate_task)
            gjid = hq.submit_task(
                gate_task,
                command=[
                    sys.executable,
                    "-m",
                    "conjectra.stage_runner",
                    "--task-id",
                    gate_task.task_id,
                    "--arr-home",
                    str(arr_home),
                ],
                stage=StageKind.CPU_GATE,
                cwd=str(arr_home),
                env=env_overrides,
                wrap_exit=False,
            )
            hq_jobs.append(gjid)
            entry["gate_task_id"] = gate_task.task_id
            entry["gate_jid"] = gjid

        for entry in slice_entries:
            hq.wait_job(entry["gate_jid"], timeout_s=float(min(240, timeout_s)) + 30.0)

        # 3. Collect real receipts, usage artifacts, and merge onto integration worktree
        total_in = 0
        total_out = 0
        total_cache = 0
        usage_paths: list[str] = []
        gate_intercepted = False
        all_gates_ok = True

        integ_wt = (
            slice_entries[0]["wt"]
            if len(slice_entries) == 1
            else sb.create_worktree(
                mirror, project, f"arr/{project}/{goal_id}-integ", base_sha, origin_repo=repo_path
            )
        )
        merge_errors: list[str] = []

        for entry in slice_entries:
            rcpts = db.list_receipts(hypothesis_id=entry["hyp_id"])
            for r in rcpts:
                total_in += r.in_tokens
                total_out += r.out_tokens
                c_entry = r.metrics.get("cache_read_tokens")
                if isinstance(c_entry, Mapping):
                    total_cache += int(c_entry.get("value", 0))
                for ap in r.raw_artifact_paths:
                    if ap.endswith("usage.json") and ap not in usage_paths:
                        usage_paths.append(ap)
                if r.stage == StageKind.CPU_GATE:
                    if r.engineering_verdict != EngineeringVerdict.PASS or r.process_status != ProcessStatus.EXIT_OK:
                        all_gates_ok = False
                        gate_intercepted = True

            if len(slice_entries) > 1:
                s_head = _git(entry["wt"], "rev-parse", "HEAD").stdout.strip()
                if s_head and s_head != base_sha:
                    _git(mirror, "fetch", str(entry["wt"]), f"{s_head}:refs/heads/{entry['branch']}-sync")
                    mp = _git(
                        integ_wt,
                        "-c",
                        "user.name=conjectra",
                        "-c",
                        "user.email=conjectra@local",
                        "merge",
                        "--no-edit",
                        s_head,
                    )
                    if mp.returncode != 0:
                        merge_errors.append(mp.stderr.strip())
                        _git(integ_wt, "merge", "--abort")
                        all_gates_ok = False
                        gate_intercepted = True

        integ_head = _git(integ_wt, "rev-parse", "HEAD").stdout.strip()
        pairing_label = "+".join(sorted({e["provider"].label() for e in slice_entries}))
        db.update_goal_status(goal_id, "ARCHIVED")
        return {
            "worktree_path": integ_wt,
            "head_sha": integ_head,
            "provider_pairing": pairing_label,
            "all_gates_ok": all_gates_ok,
            "gate_intercepted": gate_intercepted,
            "merge_errors": merge_errors,
            "uncached_total_tokens": total_in + total_out,
            "cache_read_tokens": total_cache,
            "raw_usage_paths": usage_paths,
            "hq_jobs_submitted": hq_jobs,
        }
    finally:
        db.close()


def run_layer0_gate_checks(tmp: Path) -> list[dict[str, Any]]:
    """Run deterministic 0-token kernel gate fault checks (never recorded as LLM runs)."""
    gate_injections: list[dict[str, Any]] = []
    g_repo = _init_temp_git_repo(
        tmp / "l0_faults",
        {
            "pkg/core.py": "def score(x: int) -> int:\n    return x * 2\n",
            "tests/test_core.py": "from pkg.core import score\n\ndef test_score():\n    assert score(3) == 6\n",
        },
    )
    g_base = subprocess.check_output(["git", "-C", str(g_repo), "rev-parse", "HEAD"], text=True).strip()
    (g_repo / "tests/test_core.py").write_text("def test_score():\n    pass\n", encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(g_repo), "commit", "-am", "tamper assert"],
        check=True,
        capture_output=True,
        env={
            **dict(os.environ),
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@l",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@l",
        },
    )
    g_tamper = subprocess.check_output(["git", "-C", str(g_repo), "rev-parse", "HEAD"], text=True).strip()
    f1 = gates.run_cpu_gate(g_repo, g_base, g_tamper, ["tests/test_core.py"], goal_prefixes=["pkg", "tests"])
    gate_injections.append({"fault": "assertion_deletion", "intercepted": not f1.passed, "detail": f1.failures})

    f2 = gates.validate_writable_paths(
        changed_paths=["pkg/core.py", "secret/other.py"], writable_paths=["pkg/core.py"]
    )
    gate_injections.append({"fault": "out_of_scope_write", "intercepted": not f2.passed, "detail": f2.failures})

    f3 = gates.check_no_banned_access(command="cat lockbox/oracle.json")
    gate_injections.append({"fault": "banned_lockbox_access", "intercepted": not f3.passed, "detail": f3.failures})
    return gate_injections


def run_eval_matrix(
    *,
    arr_home: str | Path | None = None,
    workloads: Sequence[str] | None = None,
    modes: Sequence[str] | None = None,
    builder_spec: str = "deepseek",
    secondary_builder_spec: str = "claude:claude-sonnet-5-5:medium",
    adversary_spec: str = "codex:gpt-5-codex:high",
    prescreener_spec: str = "grok",
    timeout_s: int = 420,
) -> dict[str, Any]:
    """Run REAL HyperQueue + REAL multi-provider LLM empirical ablation across selected cells.

    Unselected or unexecuted `(workload_class, mode)` cells remain `UNKNOWN` in both the ledger
    and the returned summary. Zero `llm_hook` mocks or hardcoded token numbers exist here.
    """
    selected_workloads = list(
        workloads or ("tight_adversarial", "disjoint_parallel", "coupled_interface", "multi_milestone")
    )
    for wc in selected_workloads:
        if wc not in WORKLOAD_CLASSES:
            raise ValueError(f"Unknown workload_class {wc!r}; allowed: {WORKLOAD_CLASSES}")

    selected_modes = set(modes) if modes else None
    home = resolve_arr_home(arr_home)
    db = StateDB(home)
    migration_info = ensure_eval_table(db)
    hq = HQRuntime(home)
    hq.ensure_running()

    # Keep ablation artifacts under CONJECTRA_HOME so raw_usage_paths remain permanently auditable!
    run_tag = f"matrix-{int(time.time())}-{uuid.uuid4().hex[:6]}"
    eval_art_root = home / "artifacts" / "eval_matrix" / run_tag
    eval_art_root.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []

    def _want_mode(m: str) -> bool:
        return selected_modes is None or m in selected_modes

    try:
        with tempfile.TemporaryDirectory(prefix="arr_real_matrix_") as tmp_str:
            tmp = Path(tmp_str)
            holdout_dir = tmp / "external_holdouts"
            holdout_dir.mkdir(parents=True, exist_ok=True)
            sb = SandboxManager(sandbox_base=tmp / "sandboxes", worktree_base=tmp / "worktrees")
            gate_injections = run_layer0_gate_checks(tmp)

            # ------------------------------------------------------------------
            # W1: tight_adversarial — Scalar ID Deduplication + 2D Pareto Frontier
            # External unseen holdout checks bool/int collision (`True == 1`),
            # nested unhashables, duplicate Pareto ties, NaN/Inf/negative rejection.
            # ------------------------------------------------------------------
            if "tight_adversarial" in selected_workloads:
                w1_repo = _init_temp_git_repo(
                    tmp / "w1",
                    {
                        "src/__init__.py": "",
                        "src/metrics_guard.py": (
                            "from typing import Any, Iterable\n\n"
                            "def dedup_eval_ids(items: Iterable[Any]) -> list[str | int]:\n"
                            "    raise NotImplementedError\n\n"
                            "def pareto_frontier_2d(points: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:\n"
                            "    raise NotImplementedError\n"
                        ),
                        "tests/test_metrics_guard.py": (
                            "from src.metrics_guard import dedup_eval_ids, pareto_frontier_2d\n\n"
                            "def test_author_smoke():\n"
                            "    assert dedup_eval_ids(['r1', 'r1', 'r2']) == ['r1', 'r2']\n"
                            "    assert pareto_frontier_2d([(2.0, 5.0), (3.0, 6.0), (4.0, 2.0)]) == [(2.0, 5.0), (4.0, 2.0)]\n"
                        ),
                    },
                )
                holdout_w1 = holdout_dir / "test_holdout_w1.py"
                holdout_w1.write_text(
                    "import math\nimport pytest\nfrom src.metrics_guard import dedup_eval_ids, pareto_frontier_2d\n\n"
                    "def test_bool_int_str_and_nested_unhashable():\n"
                    "    raw = [True, 1, '1', False, 0, None, 3.14, float('nan'), ['bad'], {'k': 1}, {9}, 1, '1', -7]\n"
                    "    assert dedup_eval_ids(raw) == [1, '1', 0, -7]\n"
                    "    assert [type(x) for x in dedup_eval_ids([False, 0, True, 1])] == [int, int]\n\n"
                    "def test_pareto_ties_nan_inf_negative_and_scale():\n"
                    "    assert pareto_frontier_2d([]) == []\n"
                    "    pts = [(2.0, 4.0), (2.0, 4.0), (2.0, 5.0), (3.0, 4.0), (1.5, 6.0), (5.0, 1.0)]\n"
                    "    assert pareto_frontier_2d(pts) == [(1.5, 6.0), (2.0, 4.0), (5.0, 1.0)]\n"
                    "    for bad in [[(-0.1, 2.0)], [(1.0, float('nan'))], [(float('inf'), 1.0)]]:\n"
                    "        with pytest.raises(ValueError):\n"
                    "            pareto_frontier_2d(bad)\n",
                    encoding="utf-8",
                )
                w1_spec = (
                    "Implement `dedup_eval_ids(items)` and `pareto_frontier_2d(points)` in `src/metrics_guard.py`:\n"
                    "1. `dedup_eval_ids(items)` returns deduplicated valid scalar IDs (`str` or exact `int`, NEVER `bool` or `float`) "
                    "preserving first-seen order; silently skip `None`, `bool`, `float`, and any unhashable objects without raising `TypeError`; "
                    "`1` and `'1'` must remain distinct.\n"
                    "2. `pareto_frontier_2d(points)` minimizes BOTH `(latency_ms, area_um2)` coordinates: raise `ValueError` on negative, NaN, or Inf "
                    "coordinates; discard dominated points (`p <= q` on both and `<` on at least one); deduplicate identical frontier coordinates; "
                    "and return sorted ascending by `(latency_ms, area_um2)`."
                )

                if _want_mode("direct_hq"):
                    t0 = time.perf_counter()
                    d_res = _run_direct_hq_slices(
                        repo_path=w1_repo,
                        project="w1_direct",
                        slices=[
                            {
                                "name": "guard",
                                "provider": builder_spec,
                                "writable_paths": ["src/metrics_guard.py", "tests/test_metrics_guard.py"],
                                "spec": w1_spec,
                                "acceptance_cmd": "python3 -m pytest -q tests/test_metrics_guard.py",
                            }
                        ],
                        hq=hq,
                        sb=sb,
                        arr_home=home,
                        timeout_s=timeout_s,
                    )
                    wall = time.perf_counter() - t0
                    h_rc, _ho, _he, _hjid = _run_external_holdout_on_hq(
                        hq=hq,
                        worktree_path=d_res["worktree_path"],
                        holdout_file=holdout_w1,
                        art_dir=eval_art_root,
                        tag="w1_direct",
                    )
                    passed = bool(d_res["all_gates_ok"] and h_rc == 0)
                    results.append(
                        record_workload_eval(
                            db,
                            workload_class="tight_adversarial",
                            mode="direct_hq",
                            provider_pairing=d_res["provider_pairing"],
                            is_real_llm=True,
                            hq_used=True,
                            holdout_cmd=str(holdout_w1.name),
                            holdout_rc=h_rc,
                            holdout_passed=passed,
                            defect_escaped=bool(d_res["all_gates_ok"] and h_rc != 0),
                            gate_intercepted=bool(d_res["gate_intercepted"]),
                            uncached_tokens=d_res["uncached_total_tokens"],
                            cache_read_tokens=d_res["cache_read_tokens"],
                            coordination_tax_pct=0.0,
                            context_compression_ratio=1.0,
                            wall_sec=wall,
                            raw_usage_paths=d_res["raw_usage_paths"],
                            head_sha=d_res["head_sha"],
                            source=f"{run_tag}:w1_direct",
                        )
                    )

                if _want_mode("x-boost"):
                    for b_cand in dict.fromkeys([builder_spec, secondary_builder_spec]):
                        t0 = time.perf_counter()
                        b_sum = run_boost(
                            repo_path=w1_repo,
                            project="w1_boost",
                            task_spec=w1_spec,
                            acceptance_cmd="python3 -m pytest -q tests/test_metrics_guard.py",
                            writable_paths=["src/metrics_guard.py", "tests/test_metrics_guard.py"],
                            builder_spec=b_cand,
                            adversary_spec=adversary_spec,
                            max_rounds=2,
                            timeout_s=timeout_s,
                            use_hq=True,
                            arr_home=home,
                            sandbox_manager=sb,
                        )
                        wall = time.perf_counter() - t0
                        h_rc, _ho, _he, _hjid = _run_external_holdout_on_hq(
                            hq=hq,
                            worktree_path=Path(b_sum["worktree_path"]),
                            holdout_file=holdout_w1,
                            art_dir=eval_art_root,
                            tag=f"w1_boost_{b_cand.split(':')[0]}",
                        )
                        passed = bool(b_sum["status"] == "PASSED" and h_rc == 0)
                        results.append(
                            record_workload_eval(
                                db,
                                workload_class="tight_adversarial",
                                mode="x-boost",
                                provider_pairing=b_sum["provider_pairing"],
                                is_real_llm=bool(b_sum["is_real_llm"]),
                                hq_used=bool(b_sum["hq_jobs_submitted"]),
                                holdout_cmd=str(holdout_w1.name),
                                holdout_rc=h_rc,
                                holdout_passed=passed,
                                defect_escaped=bool(b_sum["status"] == "PASSED" and h_rc != 0),
                                gate_intercepted=bool(b_sum["rounds_used"] > 1 or b_sum["status"] != "PASSED"),
                                uncached_tokens=b_sum["uncached_total_tokens"],
                                cache_read_tokens=b_sum["cache_read_tokens"],
                                coordination_tax_pct=b_sum["coordination_tax_pct"],
                                context_compression_ratio=1.0,
                                wall_sec=wall,
                                raw_usage_paths=b_sum["raw_usage_paths"],
                                head_sha=b_sum["head_sha"],
                                source=f"{run_tag}:w1_boost",
                            )
                        )

            # ------------------------------------------------------------------
            # W2: disjoint_parallel — Frozen Contract across 2 Disjoint Modules
            # ------------------------------------------------------------------
            if "disjoint_parallel" in selected_workloads:
                w2_repo = _init_temp_git_repo(
                    tmp / "w2",
                    {
                        "src/__init__.py": "",
                        "src/contracts.py": (
                            "from dataclasses import dataclass\n\n"
                            "@dataclass(frozen=True)\n"
                            "class WindowSummary:\n"
                            "    count: int\n"
                            "    p95: float\n"
                        ),
                        "src/clipper.py": "def clip_iqr(values: list[float], k: float = 1.5) -> list[float]:\n    raise NotImplementedError\n",
                        "src/summarizer.py": (
                            "from src.contracts import WindowSummary\n\n"
                            "def summarize_window(values: list[float]) -> WindowSummary:\n"
                            "    raise NotImplementedError\n"
                        ),
                        "tests/test_smoke.py": (
                            "from src.clipper import clip_iqr\nfrom src.summarizer import summarize_window\n\n"
                            "def test_smoke():\n"
                            "    assert isinstance(clip_iqr([1.0, 2.0, 3.0, 100.0]), list)\n"
                        ),
                    },
                )
                holdout_w2 = holdout_dir / "test_holdout_w2.py"
                holdout_w2.write_text(
                    "import pytest\nfrom src.clipper import clip_iqr\nfrom src.summarizer import summarize_window\n\n"
                    "def test_disjoint_pipeline():\n"
                    "    assert clip_iqr([]) == []\n"
                    "    clipped = clip_iqr([10.0, 12.0, 11.0, 13.0, 12.5, 500.0], k=1.5)\n"
                    "    assert 500.0 not in clipped and len(clipped) == 5\n"
                    "    s = summarize_window(clipped)\n"
                    "    assert s.count == 5 and 12.5 <= s.p95 <= 13.0\n"
                    "    s0 = summarize_window([])\n"
                    "    assert s0.count == 0 and s0.p95 == 0.0\n",
                    encoding="utf-8",
                )
                s1_spec = (
                    "Implement `clip_iqr(values: list[float], k: float = 1.5) -> list[float]` in `src/clipper.py`: "
                    "return `[]` on empty input; for `<4` items return `list(values)`; otherwise compute Q1 (25th pct) and Q3 (75th pct) "
                    "using linear interpolation on sorted values, and filter `values` keeping `[Q1 - k*(Q3-Q1), Q3 + k*(Q3-Q1)]` in original order."
                )
                s2_spec = (
                    "Implement `summarize_window(values: list[float]) -> WindowSummary` in `src/summarizer.py` using `WindowSummary` from `src/contracts.py`: "
                    "for empty `values` return `WindowSummary(count=0, p95=0.0)`; otherwise compute 95th percentile via linear interpolation on sorted floats."
                )

                if _want_mode("direct_hq"):
                    t0 = time.perf_counter()
                    d2 = _run_direct_hq_slices(
                        repo_path=w2_repo,
                        project="w2_direct",
                        slices=[
                            {
                                "name": "clipper",
                                "provider": builder_spec,
                                "writable_paths": ["src/clipper.py"],
                                "spec": s1_spec,
                                "acceptance_cmd": "python3 -c 'from src.clipper import clip_iqr; assert clip_iqr([1,2,3,4,100]) == [1,2,3,4]'",
                            },
                            {
                                "name": "summarizer",
                                "provider": secondary_builder_spec,
                                "writable_paths": ["src/summarizer.py"],
                                "spec": s2_spec,
                                "acceptance_cmd": "python3 -c 'from src.summarizer import summarize_window; assert summarize_window([10.0, 20.0]).count == 2'",
                            },
                        ],
                        hq=hq,
                        sb=sb,
                        arr_home=home,
                        timeout_s=timeout_s,
                    )
                    wall = time.perf_counter() - t0
                    h_rc, _ho, _he, _hjid = _run_external_holdout_on_hq(
                        hq=hq,
                        worktree_path=d2["worktree_path"],
                        holdout_file=holdout_w2,
                        art_dir=eval_art_root,
                        tag="w2_direct",
                    )
                    results.append(
                        record_workload_eval(
                            db,
                            workload_class="disjoint_parallel",
                            mode="direct_hq",
                            provider_pairing=d2["provider_pairing"],
                            is_real_llm=True,
                            hq_used=True,
                            holdout_cmd=str(holdout_w2.name),
                            holdout_rc=h_rc,
                            holdout_passed=bool(d2["all_gates_ok"] and h_rc == 0),
                            defect_escaped=bool(d2["all_gates_ok"] and h_rc != 0),
                            gate_intercepted=bool(d2["gate_intercepted"]),
                            uncached_tokens=d2["uncached_total_tokens"],
                            cache_read_tokens=d2["cache_read_tokens"],
                            coordination_tax_pct=0.0,
                            context_compression_ratio=1.0,
                            wall_sec=wall,
                            raw_usage_paths=d2["raw_usage_paths"],
                            head_sha=d2["head_sha"],
                            source=f"{run_tag}:w2_direct",
                        )
                    )

                if _want_mode("x-swarm"):
                    t0 = time.perf_counter()
                    sw2 = run_xswarm(
                        repo_path=w2_repo,
                        project="w2_swarm",
                        title="Disjoint window clipper and summarizer",
                        spec_workers=[
                            {"name": "clip", "provider": builder_spec, "writable_paths": ["src/clipper.py"], "brief": s1_spec},
                            {"name": "summ", "provider": secondary_builder_spec, "writable_paths": ["src/summarizer.py"], "brief": s2_spec},
                        ],
                        verifier_spec=adversary_spec,
                        verifier_cmd="python3 -m pytest -q tests/test_smoke.py",
                        max_ticks=4,
                        timeout_s=timeout_s,
                        use_hq=True,
                        arr_home=home,
                        sandbox_manager=sb,
                    )
                    wall = time.perf_counter() - t0
                    h_rc, _ho, _he, _hjid = _run_external_holdout_on_hq(
                        hq=hq,
                        worktree_path=Path(sw2["integration_worktree"]),
                        holdout_file=holdout_w2,
                        art_dir=eval_art_root,
                        tag="w2_swarm",
                    )
                    results.append(
                        record_workload_eval(
                            db,
                            workload_class="disjoint_parallel",
                            mode="x-swarm",
                            provider_pairing=sw2["provider_pairing"],
                            is_real_llm=bool(sw2["is_real_llm"]),
                            hq_used=bool(sw2["hq_jobs_submitted"]),
                            holdout_cmd=str(holdout_w2.name),
                            holdout_rc=h_rc,
                            holdout_passed=bool(sw2["status"] == "PASSED" and h_rc == 0),
                            defect_escaped=bool(sw2["status"] == "PASSED" and h_rc != 0),
                            gate_intercepted=bool(sw2["status"] != "PASSED"),
                            uncached_tokens=sw2["uncached_total_tokens"],
                            cache_read_tokens=sw2["cache_read_tokens"],
                            coordination_tax_pct=sw2["coordination_tax_pct"],
                            context_compression_ratio=1.0,
                            wall_sec=wall,
                            raw_usage_paths=sw2["raw_usage_paths"],
                            head_sha=sw2["head_sha"],
                            source=f"{run_tag}:w2_swarm",
                        )
                    )

            # ------------------------------------------------------------------
            # W3: coupled_interface — Unfrozen Producer/Consumer Wire Packet Schema
            # Producer (`src/emitter.py`) & Consumer (`src/ingestor.py`) have NO pre-shared
            # wire key names in code; `direct_hq` runs blind in isolated worktrees while
            # `x-swarm` (`use_hq=True`) aligns internal dict keys via `@verifier` on HQ!
            # ------------------------------------------------------------------
            if "coupled_interface" in selected_workloads:
                w3_repo = _init_temp_git_repo(
                    tmp / "w3",
                    {
                        "src/__init__.py": "",
                        "src/emitter.py": (
                            "def encode_telemetry(deltas_cycles: list[int], clock_mhz: int) -> dict:\n"
                            "    \"\"\"Encode cycle deltas (filtering negatives as dropped) + clock_mhz into a wire dict.\"\"\"\n"
                            "    raise NotImplementedError\n"
                        ),
                        "src/ingestor.py": (
                            "def decode_telemetry(packet: dict) -> dict[str, float]:\n"
                            "    \"\"\"Consume packet from encode_telemetry and return {'total_ns': float, 'drop_rate': float}.\"\"\"\n"
                            "    raise NotImplementedError\n"
                        ),
                        "tests/test_coupled.py": (
                            "from src.emitter import encode_telemetry\n"
                            "from src.ingestor import decode_telemetry\n\n"
                            "def test_roundtrip_500mhz():\n"
                            "    out = decode_telemetry(encode_telemetry([500, -1, 1000, -2], clock_mhz=500))\n"
                            "    assert abs(out['total_ns'] - 3000.0) < 1e-6\n"
                            "    assert abs(out['drop_rate'] - 0.5) < 1e-6\n"
                        ),
                    },
                )
                holdout_w3 = holdout_dir / "test_holdout_w3.py"
                holdout_w3.write_text(
                    "import pytest\nfrom src.emitter import encode_telemetry\nfrom src.ingestor import decode_telemetry\n\n"
                    "def test_coupled_edge_cases():\n"
                    "    out = decode_telemetry(encode_telemetry([250, 750, -5, 0], clock_mhz=1000))\n"
                    "    assert abs(out['total_ns'] - 1000.0) < 1e-6\n"
                    "    assert abs(out['drop_rate'] - 0.25) < 1e-6\n"
                    "    empty = decode_telemetry(encode_telemetry([], clock_mhz=500))\n"
                    "    assert empty['total_ns'] == 0.0 and empty['drop_rate'] == 0.0\n",
                    encoding="utf-8",
                )
                emit_brief = (
                    "Implement `encode_telemetry(deltas_cycles: list[int], clock_mhz: int) -> dict` in `src/emitter.py` ONLY: "
                    "raise `ValueError` if `clock_mhz <= 0`; separate non-negative cycle deltas (`>= 0`) from negative dropped entries (`< 0`), "
                    "and return a JSON-serializable dict wire packet carrying valid cycle deltas, dropped count, total count, and clock_mhz."
                )
                ingest_brief = (
                    "Implement `decode_telemetry(packet: dict) -> dict[str, float]` in `src/ingestor.py` ONLY: "
                    "parse the wire dict produced by `encode_telemetry`, compute `total_ns = sum(valid_cycles) * 1000.0 / clock_mhz` "
                    "(0.0 when empty) and `drop_rate = dropped_count / total_count` (0.0 when empty), returning `{'total_ns': ..., 'drop_rate': ...}`."
                )

                if _want_mode("direct_hq"):
                    t0 = time.perf_counter()
                    d3 = _run_direct_hq_slices(
                        repo_path=w3_repo,
                        project="w3_direct",
                        slices=[
                            {
                                "name": "emitter",
                                "provider": builder_spec,
                                "writable_paths": ["src/emitter.py"],
                                "spec": emit_brief,
                                "acceptance_cmd": "python3 -c 'from src.emitter import encode_telemetry; assert isinstance(encode_telemetry([10, -1], 500), dict)'",
                            },
                            {
                                "name": "ingestor",
                                "provider": secondary_builder_spec,
                                "writable_paths": ["src/ingestor.py"],
                                "spec": ingest_brief,
                                "acceptance_cmd": "python3 -c 'import src.ingestor; assert callable(src.ingestor.decode_telemetry)'",
                            },
                        ],
                        hq=hq,
                        sb=sb,
                        arr_home=home,
                        timeout_s=timeout_s,
                    )
                    wall = time.perf_counter() - t0
                    h_rc, _ho, _he, _hjid = _run_external_holdout_on_hq(
                        hq=hq,
                        worktree_path=d3["worktree_path"],
                        holdout_file=holdout_w3,
                        art_dir=eval_art_root,
                        tag="w3_direct",
                    )
                    results.append(
                        record_workload_eval(
                            db,
                            workload_class="coupled_interface",
                            mode="direct_hq",
                            provider_pairing=d3["provider_pairing"],
                            is_real_llm=True,
                            hq_used=True,
                            holdout_cmd=str(holdout_w3.name),
                            holdout_rc=h_rc,
                            holdout_passed=bool(d3["all_gates_ok"] and h_rc == 0),
                            defect_escaped=bool(d3["all_gates_ok"] and h_rc != 0),
                            gate_intercepted=bool(d3["gate_intercepted"]),
                            uncached_tokens=d3["uncached_total_tokens"],
                            cache_read_tokens=d3["cache_read_tokens"],
                            coordination_tax_pct=0.0,
                            context_compression_ratio=1.0,
                            wall_sec=wall,
                            raw_usage_paths=d3["raw_usage_paths"],
                            head_sha=d3["head_sha"],
                            source=f"{run_tag}:w3_direct",
                        )
                    )

                if _want_mode("x-swarm"):
                    t0 = time.perf_counter()
                    sw3 = run_xswarm(
                        repo_path=w3_repo,
                        project="w3_swarm",
                        title="Align telemetry emitter and ingestor wire packet keys",
                        spec_workers=[
                            {
                                "name": "emitter",
                                "provider": builder_spec,
                                "writable_paths": ["src/emitter.py"],
                                "allowed_targets": ["ingestor", "verifier", "coordinator"],
                                "brief": emit_brief + " Handoff exact dict keys to @ingestor or @verifier.",
                            },
                            {
                                "name": "ingestor",
                                "provider": secondary_builder_spec,
                                "writable_paths": ["src/ingestor.py"],
                                "allowed_targets": ["emitter", "verifier", "coordinator"],
                                "brief": ingest_brief,
                            },
                        ],
                        verifier_spec=adversary_spec,
                        verifier_cmd="python3 -m pytest -q tests/test_coupled.py",
                        max_ticks=6,
                        timeout_s=timeout_s,
                        use_hq=True,
                        arr_home=home,
                        sandbox_manager=sb,
                    )
                    wall = time.perf_counter() - t0
                    h_rc, _ho, _he, _hjid = _run_external_holdout_on_hq(
                        hq=hq,
                        worktree_path=Path(sw3["integration_worktree"]),
                        holdout_file=holdout_w3,
                        art_dir=eval_art_root,
                        tag="w3_swarm",
                    )
                    results.append(
                        record_workload_eval(
                            db,
                            workload_class="coupled_interface",
                            mode="x-swarm",
                            provider_pairing=sw3["provider_pairing"],
                            is_real_llm=bool(sw3["is_real_llm"]),
                            hq_used=bool(sw3["hq_jobs_submitted"]),
                            holdout_cmd=str(holdout_w3.name),
                            holdout_rc=h_rc,
                            holdout_passed=bool(sw3["status"] == "PASSED" and h_rc == 0),
                            defect_escaped=bool(sw3["status"] == "PASSED" and h_rc != 0),
                            gate_intercepted=bool(sw3["status"] != "PASSED"),
                            uncached_tokens=sw3["uncached_total_tokens"],
                            cache_read_tokens=sw3["cache_read_tokens"],
                            coordination_tax_pct=sw3["coordination_tax_pct"],
                            context_compression_ratio=1.0,
                            wall_sec=wall,
                            raw_usage_paths=sw3["raw_usage_paths"],
                            head_sha=sw3["head_sha"],
                            source=f"{run_tag}:w3_swarm",
                        )
                    )

            # ------------------------------------------------------------------
            # W4: multi_milestone — 2-Phase Disjoint Build + <4KB Flush + Tier-2 Audit
            # ------------------------------------------------------------------
            if "multi_milestone" in selected_workloads and _want_mode("x-teamwork"):
                w4_repo = _init_temp_git_repo(
                    tmp / "w4",
                    {
                        "fe/__init__.py": "",
                        "fe/schedule.py": "def pack_stages(ops: list[tuple[str, float]], budget: float) -> list[list[str]]:\n    raise NotImplementedError\n",
                        "be/__init__.py": "",
                        "be/slack.py": "def worst_negative_slack(stage_delays: list[float], clock_period: float) -> float:\n    raise NotImplementedError\n",
                        "integ/__init__.py": "",
                        "integ/closure.py": "def timing_Report(ops: list[tuple[str, float]], clock_period: float) -> dict:\n    raise NotImplementedError\n",
                        "tests/test_milestone.py": (
                            "from fe.schedule import pack_stages\nfrom be.slack import worst_negative_slack\n\n"
                            "def test_p1_contracts():\n"
                            "    assert callable(pack_stages) and callable(worst_negative_slack)\n"
                        ),
                    },
                )
                holdout_w4 = holdout_dir / "test_holdout_w4.py"
                holdout_w4.write_text(
                    "from integ.closure import evaluate_closure\n\n"
                    "def test_full_closure():\n"
                    "    ops = [('mul', 1.4), ('add', 0.5), ('mux', 1.2), ('reg', 0.4)]\n"
                    "    rep = evaluate_closure(ops, clock_period=2.0)\n"
                    "    assert rep['stage_count'] == 2\n"
                    "    assert abs(rep['wns'] - 0.1) < 1e-6\n"
                    "    assert rep['timing_met'] is True\n",
                    encoding="utf-8",
                )
                t0 = time.perf_counter()
                tw4 = run_xteamwork(
                    repo_path=w4_repo,
                    project="w4_teamwork",
                    title="Two-phase RTL stage packing and WNS closure report",
                    phases=[
                        {
                            "name": "Phase 1 Disjoint FE Packing & BE Slack",
                            "slices": [
                                {
                                    "name": "fe_pack",
                                    "provider": builder_spec,
                                    "writable_paths": ["fe/schedule.py"],
                                    "spec": (
                                        "Implement `pack_stages(ops: list[tuple[str, float]], budget: float) -> list[list[tuple[str, float]]]` "
                                        "in `fe/schedule.py`: greedily pack consecutive `(op_name, delay)` items into a stage while sum of delays `<= budget` "
                                        "(raise `ValueError` if `budget <= 0` or any `delay < 0`; if a single op exceeds `budget`, place it alone in its own stage)."
                                    ),
                                    "acceptance_cmd": "python3 -c 'from fe.schedule import pack_stages; assert len(pack_stages([(\"a\",1.4),(\"b\",0.5),(\"c\",1.2)], 2.0)) == 2'",
                                },
                                {
                                    "name": "be_slack",
                                    "provider": secondary_builder_spec,
                                    "writable_paths": ["be/slack.py"],
                                    "spec": (
                                        "Implement `worst_negative_slack(stage_delays: list[float], clock_period: float) -> float` in `be/slack.py`: "
                                        "return `clock_period` if `stage_delays` is empty, else `min(clock_period - d for d in stage_delays)`."
                                    ),
                                    "acceptance_cmd": "python3 -c 'from be.slack import worst_negative_slack; assert abs(worst_negative_slack([1.9, 1.6], 2.0) - 0.1) < 1e-6'",
                                },
                            ],
                        },
                        {
                            "name": "Phase 2 Closure Integration on <4KB Milestone Anchor",
                            "slices": [
                                {
                                    "name": "integ_closure",
                                    "provider": builder_spec,
                                    "writable_paths": ["integ/closure.py", "tests/test_milestone.py"],
                                    "spec": (
                                        "Implement `evaluate_closure(ops: list[tuple[str, float]], clock_period: float) -> dict` in `integ/closure.py` "
                                        "calling `pack_stages(ops, clock_period)` and `worst_negative_slack`, returning "
                                        "`{'stage_count': int, 'wns': float, 'timing_met': bool(wns >= 0.0)}`, and add unit tests in `tests/test_milestone.py`."
                                    ),
                                    "acceptance_cmd": "python3 -m pytest -q tests/test_milestone.py",
                                }
                            ],
                        },
                    ],
                    final_acceptance_cmd="python3 -m pytest -q tests/test_milestone.py",
                    prescreener_spec=prescreener_spec,
                    auditor_spec=adversary_spec,
                    timeout_s=timeout_s,
                    use_hq=True,
                    arr_home=home,
                    sandbox_manager=sb,
                )
                wall = time.perf_counter() - t0
                wt4_str = tw4.get("staging_worktree")
                if wt4_str and Path(wt4_str).exists():
                    h_rc, _ho, _he, _hjid = _run_external_holdout_on_hq(
                        hq=hq,
                        worktree_path=Path(wt4_str),
                        holdout_file=holdout_w4,
                        art_dir=eval_art_root,
                        tag="w4_teamwork",
                    )
                else:
                    h_rc = 1
                if tw4.get("raw_usage_paths"):
                    results.append(
                        record_workload_eval(
                            db,
                            workload_class="multi_milestone",
                            mode="x-teamwork",
                            provider_pairing=tw4["provider_pairing"],
                            is_real_llm=bool(tw4["is_real_llm"]),
                            hq_used=bool(tw4.get("hq_jobs_submitted")),
                            holdout_cmd=str(holdout_w4.name),
                            holdout_rc=h_rc,
                            holdout_passed=bool(tw4["status"] == "PASSED" and h_rc == 0),
                            defect_escaped=bool(tw4["status"] == "PASSED" and h_rc != 0),
                            gate_intercepted=bool(tw4["status"] != "PASSED"),
                            uncached_tokens=tw4["uncached_total_tokens"],
                            cache_read_tokens=tw4.get("cache_read_tokens", 0),
                            coordination_tax_pct=tw4["coordination_tax_pct"],
                            context_compression_ratio=tw4["context_compression_ratio"],
                            wall_sec=wall,
                            raw_usage_paths=tw4["raw_usage_paths"],
                            head_sha=tw4.get("staging_head", ""),
                            source=f"{run_tag}:w4_teamwork",
                        )
                    )

        # Build honest summary across ALL 5 workload classes (marking untested cells UNKNOWN)
        winners_by_class: dict[str, Any] = {}
        matrix_cells: dict[str, dict[str, Any]] = {}
        with db._lock:
            for wc in WORKLOAD_CLASSES:
                matrix_cells[wc] = {}
                for cand_m in CANDIDATE_MODES_BY_WORKLOAD[wc]:
                    cell_row = db._conn.execute(
                        """
                        SELECT COUNT(*) AS n,
                               AVG(holdout_passed) AS pass_rate,
                               AVG(uncached_tokens) AS avg_tokens,
                               AVG(coordination_tax_pct) AS avg_tax_pct,
                               AVG(effective_yield_per_100k) AS avg_yield
                        FROM workload_eval_ledger
                        WHERE workload_class = ? AND mode = ? AND is_real_llm = 1
                        """,
                        (wc, cand_m),
                    ).fetchone()
                    if cell_row and int(cell_row["n"] or 0) > 0:
                        matrix_cells[wc][cand_m] = dict(cell_row)
                    else:
                        matrix_cells[wc][cand_m] = UNKNOWN

                top = db._conn.execute(
                    """
                    SELECT mode, provider_pairing, COUNT(*) AS n, AVG(holdout_passed) AS pass_rate
                    FROM workload_eval_ledger
                    WHERE workload_class = ? AND is_real_llm = 1
                    GROUP BY mode, provider_pairing
                    HAVING AVG(holdout_passed) > 0
                    ORDER BY AVG(holdout_passed) DESC, AVG(defect_escaped) ASC, AVG(effective_yield_per_100k) DESC
                    LIMIT 1
                    """,
                    (wc,),
                ).fetchone()
                winners_by_class[wc] = (
                    f"{top['mode']} [{top['provider_pairing']}] (n={top['n']}, pass_rate={round(float(top['pass_rate']), 2)})"
                    if top
                    else UNKNOWN
                )

        return {
            "run_tag": run_tag,
            "migration": migration_info,
            "layer0_deterministic_gate_checks": gate_injections,
            "layer0_interception_rate_pct": round(
                (sum(1 for g in gate_injections if g["intercepted"]) / max(1, len(gate_injections))) * 100.0,
                2,
            ),
            "ablation_runs_this_invocation": results,
            "matrix_status_by_cell": matrix_cells,
            "empirical_winners_by_workload": winners_by_class,
        }
    finally:
        db.close()
