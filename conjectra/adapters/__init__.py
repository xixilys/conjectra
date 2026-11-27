"""Project adapters for conjectra stage command construction and metric parsing."""

from __future__ import annotations

import inspect
import json
import math
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping
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
    SubGoalNode,
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


def _resolve_verdict_target(
    goal: ResearchGoal,
    hyp: HypothesisNode,
    subgoal: SubGoalNode | None = None,
) -> tuple[str, float, str, list[str], bool]:
    """Resolve effective `(primary_metric, baseline, direction, kill_criteria, is_subgoal)`."""
    hparams = dict(hyp.params or {}) if isinstance(getattr(hyp, "params", None), Mapping) else {}
    if subgoal is not None and str(getattr(subgoal, "local_metric", "") or "").strip():
        primary = str(subgoal.local_metric).strip()
        baseline = float(getattr(subgoal, "baseline_value", 0.0) or 0.0)
        direction = str(getattr(subgoal, "direction", "") or "maximize").strip().lower()
        sg_kc = [str(k).strip() for k in (getattr(subgoal, "kill_criteria", None) or []) if str(k).strip()]
        criteria = list(dict.fromkeys([*(hyp.kill_criteria or []), *sg_kc]))
        return primary, baseline, direction, criteria, True

    raw_local = str(hparams.get("subgoal_local_metric") or "").strip()
    if raw_local:
        raw_base = hparams.get("subgoal_baseline_value")
        baseline = float(raw_base) if raw_base is not None else 0.0
        direction = str(hparams.get("subgoal_direction") or "maximize").strip().lower()
        raw_sg_kc = hparams.get("subgoal_kill_criteria")
        sg_kc = (
            [str(k).strip() for k in raw_sg_kc if str(k).strip()]
            if isinstance(raw_sg_kc, (list, tuple))
            else []
        )
        criteria = list(dict.fromkeys([*(hyp.kill_criteria or []), *sg_kc]))
        return raw_local, baseline, direction, criteria, True

    primary = (goal.primary_metric or "").strip()
    baseline = float(goal.baseline_value)
    direction = (goal.direction or "maximize").strip().lower()
    criteria = list(dict.fromkeys([*(hyp.kill_criteria or []), *(goal.kill_criteria or [])]))
    return primary, baseline, direction, criteria, False


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
    subgoal: SubGoalNode | None = None,
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

    primary, baseline, direction, raw_criteria, using_subgoal = _resolve_verdict_target(
        goal, hyp, subgoal=subgoal
    )
    has_primary = bool(primary) and (primary in metrics or primary in merged_flat)

    env_mismatch = detect_env_config_mismatch(combined_output, "", "")
    if env_mismatch and (exit_code != 0 or not has_primary):
        return (
            EvidenceValidity.MISSING_INPUT,
            ScientificVerdict.UNKNOWN,
            f"ENV_CONFIG_MISMATCH: {env_mismatch}",
        )

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
            "SubGoal local_metric is empty" if using_subgoal else "Goal primary_metric is empty",
        )

    if not has_primary:
        label = "SubGoal local metric" if using_subgoal else "Primary metric"
        return (
            EvidenceValidity.MISSING_PRIMARY_METRIC,
            ScientificVerdict.UNKNOWN,
            f"{label} {primary!r} missing from {stage.value} metrics (tests_passed alone is engineering-only)",
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

    def _collect_bounded_spec_excerpts(
        self,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        worktree_path: Path,
        *,
        max_bytes: int = 3600,
    ) -> str:
        """Read safe, repo-contained `spec_paths` bounded strictly under `<4KB`."""
        raw_specs = (hyp.params or {}).get("spec_paths") if isinstance(hyp.params, Mapping) else None
        if not raw_specs:
            return ""
        spec_list = (
            [str(raw_specs).strip()]
            if isinstance(raw_specs, str)
            else [str(s).strip() for s in raw_specs if str(s).strip()]
        )
        if not spec_list:
            return ""

        from conjectra.explore import protected_repo_fragments
        from conjectra.gates import check_no_banned_access

        wt_root = worktree_path.expanduser().resolve() if worktree_path else None
        repo_root = Path(goal.repo_path).expanduser().resolve() if goal.repo_path else None
        roots = [r for r in (wt_root, repo_root) if r is not None and r.exists()]
        if not roots:
            return ""

        extra_bans = [*list(goal.banned_paths or ()), *protected_repo_fragments()]
        chunks: list[tuple[str, str]] = []
        seen_resolved: set[Path] = set()
        for sp in spec_list:
            pre_gate = check_no_banned_access(
                paths=[sp],
                extra_bans=extra_bans,
                repo_path=roots[0],
            )
            if not pre_gate.passed:
                continue
            cand_path: Path | None = None
            matched_root: Path | None = None
            sp_path = Path(sp).expanduser()
            for root in roots:
                try:
                    resolved = (sp_path if sp_path.is_absolute() else (root / sp_path)).resolve()
                except OSError:
                    continue
                if not resolved.is_relative_to(root):
                    continue
                if resolved.is_file():
                    cand_path = resolved
                    matched_root = root
                    break
            if cand_path is None or matched_root is None or cand_path in seen_resolved:
                continue
            post_gate = check_no_banned_access(
                paths=[sp, str(cand_path)],
                extra_bans=extra_bans,
                repo_path=matched_root,
            )
            if not post_gate.passed:
                continue
            try:
                text = cand_path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                continue
            if not text:
                continue
            seen_resolved.add(cand_path)
            rel_label = str(cand_path.relative_to(matched_root))
            chunks.append((rel_label, text))

        if not chunks:
            return ""

        header = "Spec excerpts (<4KB):\n"
        budget = max(256, min(3900, int(max_bytes))) - len(header.encode("utf-8"))
        per_file_budget = max(128, budget // len(chunks))
        parts: list[str] = []
        used = 0
        for label, body in chunks:
            prefix = f"--- {label} ---\n"
            prefix_b = len(prefix.encode("utf-8"))
            if used + prefix_b >= budget:
                break
            rem = min(per_file_budget, budget - used) - prefix_b
            if rem <= 32:
                break
            body_bytes = body.encode("utf-8")
            if len(body_bytes) > rem:
                marker = "\n...[truncated]"
                take = max(1, rem - len(marker.encode("utf-8")))
                clipped = body_bytes[:take].decode("utf-8", errors="ignore") + marker
            else:
                clipped = body
            block = prefix + clipped + "\n"
            block_b = len(block.encode("utf-8"))
            if used + block_b > budget:
                break
            parts.append(block)
            used += block_b

        if not parts:
            return ""
        out = header + "".join(parts)
        while len(out.encode("utf-8")) >= 4096 and len(parts) > 0:
            parts.pop()
            out = (header + "".join(parts)) if parts else ""
        return out

    def build_code_prompt(
        self,
        hyp: HypothesisNode,
        goal: ResearchGoal,
        worktree_path: Path,
    ) -> str:
        """Build a concise, bounded prompt for a one-shot AI CODE worker."""
        writable = ", ".join(hyp.writable_paths) if hyp.writable_paths else "(none)"
        banned = ", ".join(goal.banned_paths) if goal.banned_paths else "(none)"
        eff_primary, eff_base, eff_dir, eff_kills, is_sg = _resolve_verdict_target(goal, hyp)
        kills = "; ".join(eff_kills)
        metric_line = (
            f"SubGoal local metric: {eff_primary} (direction={eff_dir}, baseline={eff_base}) | "
            f"Master metric: {goal.primary_metric}\n"
            if is_sg
            else f"Primary metric: {goal.primary_metric} (direction={goal.direction}, baseline={goal.baseline_value})\n"
        )
        spec_section = self._collect_bounded_spec_excerpts(hyp, goal, worktree_path)
        spec_block = f"\n{spec_section}" if spec_section else ""
        acc_cmd = hyp.acceptance_cmd or "(none)"
        return (
            f"Goal: {goal.title}\n"
            + metric_line
            + f"Hypothesis [{hyp.hypothesis_id}] ({hyp.kind.value}): {hyp.title}\n"
            + f"Statement: {hyp.statement}\n"
            + f"Mechanism: {hyp.mechanism}\n"
            + f"Params: {json.dumps(hyp.params, sort_keys=True)}\n"
            + f"Worktree: {worktree_path}\n"
            + f"ALLOWED writable paths ONLY: {writable}\n"
            + f"FORBIDDEN paths (NEVER read or modify): {banned}\n"
            + f"Kill criteria: {kills}\n"
            + f"Acceptance command to keep green: {acc_cmd}\n"
            + spec_block
            + "\nInstructions:\n"
            + "1. Inspect only files relevant to the allowed writable paths.\n"
            + "2. Implement the minimal clean change on this branch; do NOT weaken or delete assertions.\n"
            + f"3. Run acceptance_cmd (`{acc_cmd}`) inside worktree `{worktree_path}` and verify it passes before committing.\n"
            + "4. Do NOT touch any forbidden paths or git branches outside HEAD.\n"
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
        *,
        subgoal: SubGoalNode | None = None,
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
        *,
        subgoal: SubGoalNode | None = None,
    ) -> tuple[dict[str, Any], EvidenceValidity, ScientificVerdict, list[str], str]:
        """Parse metrics and return `(metrics, evidence_validity, scientific_verdict, artifacts, reason)`."""
        self.last_evidence_validity = (
            EvidenceValidity.VALID_ENGINEERING_ONLY
            if stage not in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM)
            else EvidenceValidity.MISSING_INPUT
        )
        pm_sig = inspect.signature(self.parse_metrics).parameters
        accepts_subgoal = "subgoal" in pm_sig or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in pm_sig.values()
        )
        if accepts_subgoal:
            metrics, sci_verdict, artifacts, reason = self.parse_metrics(
                stage,
                stdout,
                stderr,
                worktree_path,
                artifact_dir,
                exit_code,
                hyp,
                goal,
                subgoal=subgoal,
            )
        else:
            hyp_for_parse = hyp
            if (
                subgoal is not None
                and str(getattr(subgoal, "local_metric", "") or "").strip()
                and not str((hyp.params or {}).get("subgoal_local_metric") or "").strip()
            ):
                patched_params = {
                    **dict(hyp.params or {}),
                    "subgoal_local_metric": str(subgoal.local_metric).strip(),
                    "subgoal_baseline_value": float(subgoal.baseline_value),
                    "subgoal_direction": str(subgoal.direction or "maximize"),
                    "subgoal_kill_criteria": list(subgoal.kill_criteria or ()),
                }
                hyp_for_parse = HypothesisNode.from_dict({**hyp.to_dict(), "params": patched_params})
            metrics, sci_verdict, artifacts, reason = self.parse_metrics(
                stage, stdout, stderr, worktree_path, artifact_dir, exit_code, hyp_for_parse, goal
            )
        validity = getattr(self, "last_evidence_validity", EvidenceValidity.VALID_ENGINEERING_ONLY)
        if (
            stage in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM)
            and sci_verdict == ScientificVerdict.CONFIRMED
        ):
            primary, _, _, _, using_subgoal = _resolve_verdict_target(goal, hyp, subgoal=subgoal)
            if not primary or primary not in metrics:
                validity = EvidenceValidity.MISSING_PRIMARY_METRIC
                sci_verdict = ScientificVerdict.UNKNOWN
                label = "SubGoal local metric" if using_subgoal else "Primary metric"
                reason = f"{label} {primary!r} missing from {stage.value} metrics"
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
        *,
        subgoal: SubGoalNode | None = None,
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
            metrics,
            flat_vals,
            goal,
            hyp,
            stage,
            exit_code,
            combined_output=combined,
            subgoal=subgoal,
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
