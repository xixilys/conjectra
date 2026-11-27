"""Project adapters for conjectra stage command construction and metric parsing."""

from __future__ import annotations

import json
import math
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from conjectra.types import (
    EvidenceValidity,
    FailureDomain,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ResearchGoal,
    ResourceSpec,
    ScientificVerdict,
    StageKind,
    StageTask,
)

ENV_MISMATCH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\[cuda_unavailable\]|\bcuda_unavailable\b", re.IGNORECASE),
    re.compile(r"torch\.cuda\.is_available\(\)\s*(?:is|==)\s*False", re.IGNORECASE),
    re.compile(r"\bcuda\s+is\s+not\s+available\b", re.IGNORECASE),
    re.compile(r"\bfound\s+no\s+nvidia\s+driver\b", re.IGNORECASE),
    re.compile(r"\b\w*(?:eval|data|dataset)inputerror\b", re.IGNORECASE),
    re.compile(r"\b(?:modulenotfounderror|importerror|no\s+module\s+named)\b", re.IGNORECASE),
    re.compile(r"\bunrecognized\s+arguments?\s*:", re.IGNORECASE),
    re.compile(r"\bno\s+such\s+option\s*:", re.IGNORECASE),
    re.compile(r"\berror:\s+invalid\s+choice\s*:", re.IGNORECASE),
    re.compile(r"\b(?i:keyerror):\s*['\"][A-Z][A-Z0-9]*_[A-Z0-9_]+['\"]"),  # missing env var, e.g. KeyError: 'DATA_DIR'
    re.compile(r"\bmissing[_\s-]?(?:dataset|input|venv|checkpoint|episodes?_jsonl)\b", re.IGNORECASE),
)


def detect_env_config_mismatch(
    stdout: str = "", stderr: str = "", reason: str = ""
) -> str:
    """Return concise diagnostic message when stdout/stderr matches an env/config mismatch pattern."""
    blob = "\n".join(x for x in (stdout, stderr, reason) if x)
    if not blob:
        return ""
    for pat in ENV_MISMATCH_PATTERNS:
        m = pat.search(blob)
        if m:
            for line in blob.splitlines():
                if pat.search(line):
                    return line.strip()[:240]
            return m.group(0)
    return ""

PYTEST_SUMMARY_RE = re.compile(
    r"(?:(?P<passed>\d+)\s+passed)?(?:.*?(?P<failed>\d+)\s+failed)?(?:.*?(?P<errors>\d+)\s+errors?)?(?:.*?(?P<skipped>\d+)\s+skipped)?",
    re.IGNORECASE,
)
UNITTEST_RAN_RE = re.compile(r"Ran\s+(?P<total>\d+)\s+tests?\s+in\s+(?P<sec>[0-9.]+)s")


def metric_entry(
    value: float | int,
    denominator: int | float | str,
    stage_rung: FidelityRung | str,
    *,
    unit: str = "",
    **extra: Any,
) -> dict[str, Any]:
    """Build a structured metric entry with explicit denominator and stage rung."""
    rung_val = stage_rung.value if isinstance(stage_rung, FidelityRung) else str(stage_rung)
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite metric value: {value}")
    entry: dict[str, Any] = {
        "value": value,
        "denominator": denominator,
        "stage_rung": rung_val,
    }
    if unit:
        entry["unit"] = unit
    entry.update(extra)
    return entry


def compute_hypothesis_fingerprint(
    goal_id: str,
    kind: HypothesisKind | str,
    writable_paths: list[str],
    params: dict[str, Any],
    mechanism: str,
    screen_cmd: str = "",
) -> str:
    """Compute a deterministic canonical SHA-256 fingerprint matching state_db."""
    from conjectra.state_db import compute_hypothesis_fingerprint as _db_fingerprint

    return _db_fingerprint(
        goal_id=goal_id,
        kind=kind,
        mechanism=mechanism,
        params=params,
        writable_paths=writable_paths,
        screen_cmd=screen_cmd,
    )


def parse_test_counts(output: str, exit_code: int) -> tuple[int, int]:
    """Extract (tests_passed, tests_total) from pytest or unittest output."""
    passed = 0
    failed = 0
    errors = 0
    total = 0

    for line in reversed((output or "").splitlines()):
        s = line.strip()
        if "passed" in s or "failed" in s or "error" in s:
            m_p = re.search(r"(\d+)\s+passed", s)
            m_f = re.search(r"(\d+)\s+failed", s)
            m_e = re.search(r"(\d+)\s+error", s)
            if m_p or m_f or m_e:
                passed = int(m_p.group(1)) if m_p else 0
                failed = int(m_f.group(1)) if m_f else 0
                errors = int(m_e.group(1)) if m_e else 0
                total = passed + failed + errors
                return passed, total

    m_uni = UNITTEST_RAN_RE.search(output or "")
    if m_uni:
        total = int(m_uni.group("total"))
        if exit_code == 0 and "FAILED" not in (output or ""):
            return total, total
        m_fail = re.search(r"failures=(\d+)", output or "")
        m_err = re.search(r"errors=(\d+)", output or "")
        n_bad = (int(m_fail.group(1)) if m_fail else 0) + (int(m_err.group(1)) if m_err else 0)
        if n_bad == 0 and exit_code != 0:
            n_bad = 1
        return max(0, total - n_bad), total

    return 0, 0


def evaluate_kill_criteria(
    kill_criteria: list[str],
    flat_values: dict[str, float | int | bool],
) -> list[str]:
    """Evaluate simple threshold kill-criteria expressions against observed values."""
    triggered: list[str] = []
    expr_re = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(<=|>=|==|!=|<|>)\s*([A-Za-z0-9_.-]+)\s*$"
    )
    for crit in kill_criteria:
        m = expr_re.match(crit)
        if not m:
            continue
        key, op, raw_rhs = m.groups()
        if key not in flat_values:
            continue
        lhs = flat_values[key]
        if raw_rhs.lower() in ("true", "false"):
            rhs: Any = raw_rhs.lower() == "true"
        else:
            try:
                rhs = float(raw_rhs)
            except ValueError:
                rhs = raw_rhs
        try:
            hit = False
            if op == ">":
                hit = lhs > rhs
            elif op == ">=":
                hit = lhs >= rhs
            elif op == "<":
                hit = lhs < rhs
            elif op == "<=":
                hit = lhs <= rhs
            elif op == "==":
                hit = lhs == rhs
            elif op == "!=":
                hit = lhs != rhs
            if hit:
                triggered.append(crit)
        except TypeError:
            continue
    return triggered


_MINIMIZE_DIRECTIONS = frozenset({"min", "minimize", "lower", "down", "reduce"})


def evaluate_scientific_verdict(
    metrics: dict[str, Any],
    flat_vals: dict[str, float | int | bool],
    goal: ResearchGoal,
    hyp: HypothesisNode,
    stage: StageKind,
    exit_code: int,
    *,
    missing_input_reason: str = "",
    combined_output: str = "",
) -> tuple[EvidenceValidity, ScientificVerdict, str]:
    """Evaluate strict scientific verdict and evidence validity from measured metrics."""
    if stage == StageKind.CPU_GATE:
        reason = f"Gate command exited {exit_code}" if exit_code != 0 else ""
        return EvidenceValidity.VALID_ENGINEERING_ONLY, ScientificVerdict.NOT_APPLICABLE, reason

    if stage not in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
        return EvidenceValidity.VALID_ENGINEERING_ONLY, ScientificVerdict.NOT_APPLICABLE, ""

    if missing_input_reason:
        return EvidenceValidity.MISSING_INPUT, ScientificVerdict.UNKNOWN, missing_input_reason

    merged_flat: dict[str, float | int | bool] = dict(flat_vals)
    for k, v in metrics.items():
        if k not in merged_flat and isinstance(v, dict) and "value" in v:
            val = v["value"]
            if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
                merged_flat[k] = float(val)

    primary = (goal.primary_metric or "").strip()
    has_primary = bool(primary) and (primary in metrics or primary in merged_flat)

    env_mismatch = detect_env_config_mismatch(combined_output, "", "")
    if env_mismatch and (exit_code != 0 or not has_primary):
        return (
            EvidenceValidity.MISSING_INPUT,
            ScientificVerdict.UNKNOWN,
            f"ENV_CONFIG_MISMATCH: {env_mismatch}",
        )

    raw_criteria = list(dict.fromkeys([*(hyp.kill_criteria or []), *(goal.kill_criteria or [])]))
    if primary != "tests_passed" and not has_primary:
        criteria = [
            c
            for c in raw_criteria
            if not re.match(r"^\s*tests_passed\s*(?:<=|>=|==|!=|<|>)", c)
        ]
    else:
        criteria = raw_criteria

    hits = evaluate_kill_criteria(criteria, merged_flat)
    if hits:
        validity = EvidenceValidity.VALID_EMPIRICAL if has_primary else EvidenceValidity.VALID_ENGINEERING_ONLY
        return validity, ScientificVerdict.REFUTED, f"Kill criteria triggered: {hits}"

    if exit_code != 0:
        if env_mismatch or not metrics:
            return (
                EvidenceValidity.MISSING_INPUT,
                ScientificVerdict.UNKNOWN,
                f"ENV_CONFIG_MISMATCH: {env_mismatch}"
                if env_mismatch
                else f"Command exited with code {exit_code} without structured metrics",
            )
        return (
            EvidenceValidity.VALID_ENGINEERING_ONLY,
            ScientificVerdict.UNKNOWN,
            f"Command exited with code {exit_code} without triggering empirical kill criteria",
        )

    if not metrics:
        return EvidenceValidity.MISSING_INPUT, ScientificVerdict.UNKNOWN, "No structured metrics or test counts reported"

    tp_entry = metrics.get("tests_passed")
    if isinstance(tp_entry, dict):
        denom_raw = tp_entry.get("denominator", 0)
        val_raw = tp_entry.get("value", 0)
        if (
            isinstance(denom_raw, (int, float))
            and not isinstance(denom_raw, bool)
            and denom_raw > 0
            and isinstance(val_raw, (int, float))
            and not isinstance(val_raw, bool)
            and float(val_raw) <= 0
        ):
            if primary == "tests_passed":
                return EvidenceValidity.VALID_EMPIRICAL, ScientificVerdict.REFUTED, "Zero tests passed"
            if not has_primary:
                return EvidenceValidity.VALID_ENGINEERING_ONLY, ScientificVerdict.UNKNOWN, "Zero tests passed"

    if not primary:
        return (
            EvidenceValidity.MISSING_PRIMARY_METRIC,
            ScientificVerdict.UNKNOWN,
            "Goal primary_metric is empty",
        )

    if not has_primary:
        return (
            EvidenceValidity.MISSING_PRIMARY_METRIC,
            ScientificVerdict.UNKNOWN,
            f"Primary metric {primary!r} missing from {stage.value} metrics (tests_passed alone is engineering-only)",
        )

    raw_val = merged_flat[primary] if primary in merged_flat else metrics[primary]["value"]
    try:
        val = float(raw_val)
    except (TypeError, ValueError):
        return (
            EvidenceValidity.MISSING_PRIMARY_METRIC,
            ScientificVerdict.UNKNOWN,
            f"Non-numeric primary metric {primary!r}: {raw_val!r}",
        )
    if not math.isfinite(val):
        return (
            EvidenceValidity.MISSING_PRIMARY_METRIC,
            ScientificVerdict.UNKNOWN,
            f"Non-finite primary metric {primary!r}: {raw_val!r}",
        )

    baseline = float(goal.baseline_value)
    direction = (goal.direction or "maximize").strip().lower()
    if direction in _MINIMIZE_DIRECTIONS:
        if not (val < baseline):
            return (
                EvidenceValidity.VALID_EMPIRICAL,
                ScientificVerdict.VALID_NEGATIVE,
                f"{primary}={val} did not improve below baseline={baseline}",
            )
    else:
        if not (val > baseline):
            return (
                EvidenceValidity.VALID_EMPIRICAL,
                ScientificVerdict.VALID_NEGATIVE,
                f"{primary}={val} did not improve above baseline={baseline}",
            )

    return EvidenceValidity.VALID_EMPIRICAL, ScientificVerdict.CONFIRMED, ""


class ProjectAdapter(ABC):
    """Base interface for project-specific execution and evidence parsing."""

    name: str = "generic"
    default_repo_path: str = ""
    default_writable_prefixes: list[str] = []
    default_banned_paths: list[str] = []
    default_kill_criteria: list[str] = []
    last_evidence_validity: EvidenceValidity = EvidenceValidity.VALID_ENGINEERING_ONLY
    last_failure_domain: FailureDomain = FailureDomain.NONE

    @abstractmethod
    def seed_hypotheses(self, goal: ResearchGoal) -> list[HypothesisNode]:
        """Generate initial deterministic competing/falsification hypotheses."""

    @abstractmethod
    def build_stage_command(
        self,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        worktree_path: Path,
        artifact_dir: Path,
    ) -> str:
        """Return shell command to execute in worktree_path for the given stage."""

    def build_code_prompt(
        self,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        worktree_path: Path,
    ) -> str:
        """Build a concise, bounded prompt for a one-shot AI CODE worker."""
        writable = ", ".join(hyp.writable_paths) if hyp.writable_paths else "(none)"
        banned = ", ".join(goal.banned_paths) if goal.banned_paths else "(none)"
        kills = "; ".join(hyp.kill_criteria or goal.kill_criteria)
        return (
            f"Goal: {goal.title}\n"
            f"Primary metric: {goal.primary_metric} (direction={goal.direction}, baseline={goal.baseline_value})\n"
            f"Hypothesis [{hyp.hypothesis_id}] ({hyp.kind.value}): {hyp.title}\n"
            f"Statement: {hyp.statement}\n"
            f"Mechanism: {hyp.mechanism}\n"
            f"Params: {json.dumps(hyp.params, sort_keys=True)}\n"
            f"Worktree: {worktree_path}\n"
            f"ALLOWED writable paths ONLY: {writable}\n"
            f"FORBIDDEN paths (NEVER read or modify): {banned}\n"
            f"Kill criteria: {kills}\n"
            f"Acceptance command to keep green: {hyp.acceptance_cmd}\n\n"
            "Instructions:\n"
            "1. Inspect only files relevant to the allowed writable paths.\n"
            "2. Implement the minimal clean change on this branch; do NOT weaken or delete assertions.\n"
            "3. Run the acceptance command locally before finishing.\n"
            "4. Do NOT touch any forbidden paths or git branches outside HEAD.\n"
        )

    def default_resource_spec(
        self,
        stage: StageKind,
        hyp: HypothesisNode,
        goal: ResearchGoal,
    ) -> ResourceSpec:
        """Return default resource hints for a stage."""
        if stage == StageKind.CODE:
            return ResourceSpec(cpus=2, ram_mb=2048, llm_pool="gemini", wall_timeout_s=900)
        if stage == StageKind.CPU_GATE:
            return ResourceSpec(cpus=2, ram_mb=2048, wall_timeout_s=90)
        if stage in (
            StageKind.BLIND_AUDIT,
            StageKind.DESIGN_REVIEW,
            StageKind.IMPL_REVIEW,
            StageKind.RESULT_REVIEW,
        ):
            return ResourceSpec(cpus=1, ram_mb=1024, llm_pool="codex", wall_timeout_s=600)
        needs_gpu = str(hyp.params.get("required_hardware", "")).lower() in ("local_gpu", "cloud_gpu") or bool(
            int(hyp.params.get("gpu", 0) or 0)
        )
        if stage == StageKind.EXP_CONFIRM:
            return ResourceSpec(cpus=2, ram_mb=4096, gpu=1 if needs_gpu else 0, wall_timeout_s=1200)
        if stage == StageKind.EXP_SCREEN:
            return ResourceSpec(cpus=2, ram_mb=2048, gpu=1 if needs_gpu else 0, wall_timeout_s=300)
        return ResourceSpec(cpus=1, ram_mb=1024, wall_timeout_s=300)

    @abstractmethod
    def parse_metrics(
        self,
        stage: StageKind,
        stdout: str,
        stderr: str,
        worktree_path: Path,
        artifact_dir: Path,
        exit_code: int,
        hyp: HypothesisNode,
        goal: ResearchGoal,
    ) -> tuple[dict[str, Any], ScientificVerdict, list[str], str]:
        """Parse stdout/artifacts into structured metrics, verdict, artifacts, and failure reason."""

    def parse_metrics_with_validity(
        self,
        stage: StageKind,
        stdout: str,
        stderr: str,
        worktree_path: Path,
        artifact_dir: Path,
        exit_code: int,
        hyp: HypothesisNode,
        goal: ResearchGoal,
    ) -> tuple[dict[str, Any], EvidenceValidity, ScientificVerdict, list[str], str]:
        """Parse metrics and return `(metrics, evidence_validity, scientific_verdict, artifacts, reason)`."""
        self.last_evidence_validity = (
            EvidenceValidity.VALID_ENGINEERING_ONLY
            if stage not in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM)
            else EvidenceValidity.MISSING_INPUT
        )
        metrics, sci_verdict, artifacts, reason = self.parse_metrics(
            stage, stdout, stderr, worktree_path, artifact_dir, exit_code, hyp, goal
        )
        validity = getattr(self, "last_evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY)
        if (
            stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM)
            and sci_verdict == ScientificVerdict.CONFIRMED
        ):
            primary = (goal.primary_metric or "").strip()
            if not primary or primary not in metrics:
                validity = EvidenceValidity.MISSING_PRIMARY_METRIC
                sci_verdict = ScientificVerdict.UNKNOWN
                reason = f"Primary metric {primary!r} missing from {stage.value} metrics"
                self.last_evidence_validity = validity

        env_diag = detect_env_config_mismatch(stdout, stderr, reason)
        if env_diag or validity == EvidenceValidity.MISSING_INPUT:
            self.last_failure_domain = FailureDomain.ENV_CONFIG_MISMATCH
            if env_diag and stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM) and validity != EvidenceValidity.VALID_EMPIRICAL:
                validity = EvidenceValidity.MISSING_INPUT
                sci_verdict = ScientificVerdict.UNKNOWN
                if not reason:
                    reason = f"ENV_CONFIG_MISMATCH: {env_diag}"
                self.last_evidence_validity = validity
        elif stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
            if validity == EvidenceValidity.VALID_EMPIRICAL and sci_verdict in (
                ScientificVerdict.CONFIRMED,
                ScientificVerdict.VALID_NEGATIVE,
                ScientificVerdict.REFUTED,
            ):
                self.last_failure_domain = FailureDomain.SCIENTIFIC_OUTCOME
            elif exit_code == 0 and validity == EvidenceValidity.MISSING_PRIMARY_METRIC:
                self.last_failure_domain = FailureDomain.ENV_CONFIG_MISMATCH
            else:
                self.last_failure_domain = FailureDomain.ENGINEERING_DEFECT
        else:
            self.last_failure_domain = (
                FailureDomain.NONE if exit_code == 0 else FailureDomain.ENGINEERING_DEFECT
            )

        return metrics, validity, sci_verdict, artifacts, reason


class GenericAdapter(ProjectAdapter):
    """Generic fallback adapter executing hypothesis commands directly."""

    name = "generic"

    def seed_hypotheses(self, goal: ResearchGoal) -> list[HypothesisNode]:
        params = {"route": "baseline_refinement"}
        mech = "Generic baseline verification and bounded refinement"
        fp = compute_hypothesis_fingerprint(
            goal.goal_id,
            HypothesisKind.COMPETING_ROUTE,
            list(goal.writable_prefixes),
            params,
            mech,
        )
        return [
            HypothesisNode(
                hypothesis_id=f"{goal.goal_id}-h1",
                goal_id=goal.goal_id,
                parent_id=None,
                kind=HypothesisKind.COMPETING_ROUTE,
                title=f"Baseline verification for {goal.title}",
                statement=f"Verify baseline and test improvement on {goal.primary_metric}.",
                mechanism=mech,
                fidelity=FidelityRung.F1_FAST_SCREEN,
                priority_evoi=1.0,
                writable_paths=list(goal.writable_prefixes),
                params=params,
                kill_criteria=list(goal.kill_criteria) or ["tests_passed == 0"],
                acceptance_cmd="pytest -q",
                screen_cmd="pytest -q",
                confirm_cmd="pytest -q",
                branch_name=f"arr/{goal.project}/{goal.goal_id}-h1",
                base_sha=goal.baseline_sha,
                fingerprint=fp,
            )
        ]

    def build_stage_command(
        self,
        task: StageTask,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        worktree_path: Path,
        artifact_dir: Path,
    ) -> str:
        if task.stage == StageKind.CPU_GATE:
            return hyp.acceptance_cmd or "pytest -q"
        if task.stage == StageKind.EXP_SCREEN:
            return hyp.screen_cmd or hyp.acceptance_cmd or "pytest -q"
        if task.stage == StageKind.EXP_CONFIRM:
            return hyp.confirm_cmd or hyp.screen_cmd or hyp.acceptance_cmd or "pytest -q"
        return hyp.acceptance_cmd or "true"

    def parse_metrics(
        self,
        stage: StageKind,
        stdout: str,
        stderr: str,
        worktree_path: Path,
        artifact_dir: Path,
        exit_code: int,
        hyp: HypothesisNode,
        goal: ResearchGoal,
    ) -> tuple[dict[str, Any], ScientificVerdict, list[str], str]:
        combined = f"{stdout}\n{stderr}"
        passed, total = parse_test_counts(combined, exit_code)
        rung = FidelityRung.F2_FULL_CONFIRM if stage == StageKind.EXP_CONFIRM else FidelityRung.F1_FAST_SCREEN
        metrics: dict[str, Any] = {}
        flat_vals: dict[str, float | int | bool] = {}
        if total > 0:
            metrics["tests_passed"] = metric_entry(passed, total, rung, unit="tests")
            flat_vals["tests_passed"] = passed
        for m in re.finditer(r"METRIC\s+([A-Za-z0-9_.-]+)=([-+]?[0-9]*\.?[0-9]+)\s*(?:denom=(\S+))?", combined):
            name, raw_val, denom = m.groups()
            val = float(raw_val)
            if math.isfinite(val):
                metrics[name] = metric_entry(val, denom or "1", rung)
                flat_vals[name] = val

        validity, sci_verdict, reason = evaluate_scientific_verdict(
            metrics, flat_vals, goal, hyp, stage, exit_code, combined_output=combined
        )
        self.last_evidence_validity = validity
        return metrics, sci_verdict, [], reason


ADAPTER_ENTRY_POINT_GROUP = "conjectra.adapters"
_ADAPTERS: dict[str, Any] = {"generic": GenericAdapter}
_ADAPTERS_DISCOVERED = False


def register_adapter(name: str, factory: Any) -> None:
    """Register a ProjectAdapter class/factory under ``name`` (case-insensitive)."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("adapter name must be non-empty")
    _ADAPTERS[key] = factory


def _discover_adapters() -> None:
    global _ADAPTERS_DISCOVERED
    if _ADAPTERS_DISCOVERED:
        return
    _ADAPTERS_DISCOVERED = True
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group=ADAPTER_ENTRY_POINT_GROUP):
            try:
                _ADAPTERS.setdefault(ep.name.lower(), ep.load())
            except Exception:  # pragma: no cover - broken plugin must not kill the runtime
                continue
    except Exception:  # pragma: no cover
        pass
    import importlib
    import os

    for item in os.environ.get("CONJECTRA_ADAPTERS", "").split(","):
        name, sep, spec = item.partition("=")
        if not sep or not name.strip() or not spec.strip():
            continue
        mod_name, _, attr = spec.strip().partition(":")
        try:
            mod = importlib.import_module(mod_name)
            _ADAPTERS[name.strip().lower()] = getattr(mod, attr) if attr else mod
        except Exception:
            continue


def available_adapters() -> list[str]:
    _discover_adapters()
    return sorted(_ADAPTERS)


def get_adapter(name: str) -> ProjectAdapter:
    """Resolve a ProjectAdapter instance by adapter_name (unknown names fall back to ``generic``)."""
    _discover_adapters()
    key = (name or "generic").strip().lower()
    factory = _ADAPTERS.get(key) or GenericAdapter
    obj = factory() if callable(factory) and not isinstance(factory, ProjectAdapter) else factory
    return obj if isinstance(obj, ProjectAdapter) else GenericAdapter()


__all__ = [
    "ADAPTER_ENTRY_POINT_GROUP",
    "GenericAdapter",
    "ProjectAdapter",
    "compute_hypothesis_fingerprint",
    "evaluate_kill_criteria",
    "evaluate_scientific_verdict",
    "available_adapters",
    "get_adapter",
    "register_adapter",
    "metric_entry",
    "parse_test_counts",
]
