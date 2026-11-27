"""Deterministic engineering, safety, scope, and evidence gates for conjectra."""

from __future__ import annotations

import json
import math
import os
import pathlib
import re
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    ExperimentPlan,
    FailureDomain,
    FidelityRung,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ScientificVerdict,
    StageKind,
    StageTask,
)

def default_read_bans_path() -> pathlib.Path:
    """Optional Markdown table of extra banned paths (first column), e.g. sealed eval sets.

    Resolved at call time from ``CONJECTRA_READ_BANS`` (default ``~/.conjectra/READ_BANS.md``).
    """
    return pathlib.Path(
        os.environ.get("CONJECTRA_READ_BANS") or "~/.conjectra/READ_BANS.md"
    ).expanduser()


DEFAULT_READ_BANS_PATH = None  # sentinel: resolve via default_read_bans_path() at call time

_HEAD_RE = re.compile(r"^[0-9a-f]{7,64}$")
_DOC_EXTENSIONS = frozenset({".md", ".rst", ".txt"})

# Always-on redlines (merged with READ_BANS.md and per-goal banned_paths)
CORE_REDLINE_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)(^|[/\\])\.holdout([/\\]|$)"),
    re.compile(r"(?i)(^|[/\\])\.codex([/\\]|$)"),
    re.compile(r"(?i)(^|[/\\])\.(ssh|gnupg|aws|config/gcloud)([/\\]|$)"),
    re.compile(r"(?i)(^|[/\\])(auth|credentials|secrets)[^/\\]*\.(json|toml|yaml|yml)$"),
    re.compile(r"(?i)(^|[/\\])(\.netrc|\.env|id_rsa|id_ed25519|human_signing_key)$"),
    re.compile(r"(?i)(^|[/\\])holdout_out([/\\]|$)"),
    re.compile(r"(?i)(^|[/\\])frozen_baseline([/\\]|$)"),
    re.compile(r"(?i)(^|[/\\])sealed_eval([/\\]|$)"),
    re.compile(r"(?i)(^|[/\\])lockbox([/\\]|$)"),
    re.compile(r"(?i)(^|[/\\])prereg_r1\.json$"),
)

def _env_regexes(var: str, *, ignore_case: bool = False) -> tuple[re.Pattern[str], ...]:
    flags = re.IGNORECASE if ignore_case else 0
    out: list[re.Pattern[str]] = []
    for raw in os.environ.get(var, "").split(","):
        if raw.strip():
            try:
                out.append(re.compile(raw.strip(), flags))
            except re.error:
                continue
    return tuple(out)


# Internal wave/node/ticket identifiers prohibited in deliverable code/comments.
# Add project-specific ticket formats with CONJECTRA_INTERNAL_ID_PATTERNS="ABC-\d+,XYZ-\d+".
_INTERNAL_ID_REGEXES: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:TICKET|TASK|JIRA)-\d+\b"),
    re.compile(r"\bwave[-_]?\d+(?:\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bnode[-_]\d+\b", re.IGNORECASE),
    *_env_regexes("CONJECTRA_INTERNAL_ID_PATTERNS"),
)

# Target-specific specialization tokens (benchmark-/device-specific hacks) that must not leak
# into generic code. Configure with CONJECTRA_SPECIALIZATION_PATTERNS="\bdevice_a\b,...".
_CHIP_SPECIALIZATION_REGEXES: tuple[re.Pattern[str], ...] = (
    *_env_regexes("CONJECTRA_SPECIALIZATION_PATTERNS", ignore_case=True),
)

_ASSERT_CALL_RE = re.compile(
    r"(?:\bassert\b|self\.assert\w+\(|pytest\.raises\(|EXPECT_\w+\(|ASSERT_\w+\()"
)
_SKIP_MARK_RE = re.compile(
    r"(?:pytest\.skip\(|@pytest\.mark\.skip\b|@pytest\.mark\.xfail\b|@unittest\.skip\b|\.skipTest\()"
)
_MAIN_WRITE_CMD_RE = re.compile(
    r"\bgit\s+(?:checkout|switch|push|merge|rebase|reset|commit)\b[^\n;|&]*\bmain\b"
)


@dataclass
class GateResult:
    """Structured outcome of one or more deterministic gates."""

    passed: bool
    failures: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    checked: list[str] = field(default_factory=list)

    def merge(self, other: GateResult) -> GateResult:
        """Merge another GateResult into this one."""
        self.failures.extend(other.failures)
        self.warnings.extend(other.warnings)
        self.checked.extend(other.checked)
        self.passed = len(self.failures) == 0
        return self

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dictionary."""
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "warnings": list(self.warnings),
            "checked": list(self.checked),
        }


# ------------------------------------------------------ Path & scope helpers
# Adapted from autotaskgen/graph.py (norm_path, path_contains, paths_overlap, scope_overlaps)


def norm_path(p: Any) -> str | None:
    """Repo-relative prefix; '' = whole repo; None = invalid. Trailing /** or /* dropped."""
    if not isinstance(p, str):
        return None
    s = p.strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    for suf in ("/**", "/*"):
        if s.endswith(suf):
            s = s[: -len(suf)]
    s = re.sub(r"/+", "/", s)
    s = s.rstrip("/") if s != "/" else s
    if s in ("", ".", "*", "**"):
        return ""
    if s.startswith("/") or ".." in s.split("/"):
        return None
    return s


def path_contains(outer: str, inner: str) -> bool:
    """Return True when normalized prefix `outer` contains normalized path `inner`."""
    return outer == "" or inner == outer or inner.startswith(outer.rstrip("/") + "/")


def paths_overlap(a: str, b: str) -> bool:
    """Return True when two normalized repo-relative prefixes overlap."""
    return path_contains(a, b) or path_contains(b, a)


def scope_overlaps(a: Sequence[str], b: Sequence[str]) -> list[tuple[str, str]]:
    """Return overlapping pairs between two normalized write-scope sequences."""
    norm_a = [x for x in (norm_path(p) for p in a) if x is not None]
    norm_b = [y for y in (norm_path(p) for p in b) if y is not None]
    return [(x, y) for x in norm_a for y in norm_b if paths_overlap(x, y)]


def detect_sibling_scope_overlaps(
    sibling_scopes: Mapping[str, Sequence[str]],
) -> list[dict[str, Any]]:
    """Detect write-scope overlaps across sibling branches/hypotheses."""
    names = sorted(sibling_scopes.keys())
    conflicts: list[dict[str, Any]] = []
    for idx, left in enumerate(names):
        for right in names[idx + 1 :]:
            overlaps = scope_overlaps(sibling_scopes[left], sibling_scopes[right])
            if overlaps:
                conflicts.append(
                    {
                        "left": left,
                        "right": right,
                        "overlaps": overlaps,
                    }
                )
    return conflicts


def validate_writable_paths(
    changed_paths: Sequence[str],
    writable_paths: Sequence[str],
    *,
    goal_prefixes: Sequence[str] = (),
    bookkeeping_paths: Sequence[str] = (),
) -> GateResult:
    """Verify writable_paths validity, containment in goal_prefixes, and diff containment."""
    failures: list[str] = []
    norm_writable: list[str] = []
    for raw in writable_paths:
        np = norm_path(raw)
        if np is None:
            failures.append(f"invalid writable_path: {raw!r}")
        else:
            norm_writable.append(np)

    if not norm_writable:
        failures.append("writable_paths must contain at least one valid relative prefix")

    norm_goal = [x for x in (norm_path(g) for g in goal_prefixes) if x is not None]
    if norm_goal and norm_writable:
        for wp in norm_writable:
            if not any(path_contains(gp, wp) for gp in norm_goal):
                failures.append(
                    f"writable_path {wp!r} is outside goal writable_prefixes {norm_goal}"
                )

    norm_bookkeeping = [x for x in (norm_path(b) for b in bookkeeping_paths) if x is not None]
    for changed in changed_paths:
        nc = norm_path(changed)
        if nc is None:
            failures.append(f"invalid changed path: {changed!r}")
            continue
        in_writable = any(path_contains(wp, nc) for wp in norm_writable)
        in_bookkeeping = any(path_contains(bp, nc) for bp in norm_bookkeeping)
        if not (in_writable or in_bookkeeping):
            failures.append(f"outside writable_paths: {changed}")

    return GateResult(passed=not failures, failures=failures, checked=["writable_paths"])


# ----------------------------------------------------------- Banned-path gate


_ASCII_PATH_TOK_RE = re.compile(r"^[A-Za-z0-9_./*~?{}\-]+$")
_READ_BANS_CACHE: dict[tuple[str, int], list[str]] = {}


def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    clean = pattern.strip().replace("\\", "/").rstrip("/")
    escaped = re.escape(clean)
    escaped = escaped.replace(r"\*\*", ".*").replace(r"\*", r"[^/]*")
    return re.compile(rf"(?i)(?:^|[/\\]){escaped}(?:[/\\]|$)")


def _extract_ban_patterns_from_cell(cell: str) -> list[str]:
    """Extract genuine path/glob tokens from a READ_BANS.md table cell."""
    out: list[str] = []
    # Strip parenthetical notes and split on Chinese/whitespace separators
    cleaned = re.sub(r"（[^）]*）|\([^)]*\)", " ", cell)
    last_dir = ""
    for raw_tok in re.split(r"[\s、，；`]+", cleaned):
        tok = raw_tok.strip(".,;: ")
        if not tok or "**" in tok and "/**/" not in tok and not tok.endswith("/**"):
            continue
        if not _ASCII_PATH_TOK_RE.fullmatch(tok):
            continue
        if tok in ("--check", "-", "—"):
            continue
        if "/" in tok:
            out.append(tok)
            if "/" in tok.rstrip("/"):
                last_dir = tok.rsplit("/", 1)[0] + "/"
        elif tok.endswith("*") and last_dir:
            out.append(last_dir + tok)
        elif tok.startswith(("TRUTH", "lockbox")):
            out.append(tok)
    return out


def load_read_bans(
    read_bans_path: pathlib.Path | str | None = DEFAULT_READ_BANS_PATH,
) -> list[str]:
    """Parse banned path patterns from READ_BANS.md if readable."""
    if read_bans_path is None:
        read_bans_path = default_read_bans_path()
    if not str(read_bans_path):
        return []
    path = pathlib.Path(read_bans_path)
    try:
        st = path.stat()
    except OSError:
        return []
    cache_key = (str(path), st.st_mtime_ns)
    if cache_key in _READ_BANS_CACHE:
        return list(_READ_BANS_CACHE[cache_key])

    patterns: list[str] = []
    seen: set[str] = set()
    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "|" not in line:
            continue
        cols = [c.strip() for c in line.split("|")]
        first = cols[1] if line.startswith("|") and len(cols) > 1 else cols[0]
        if not first or set(first) <= {"-", ":"} or first in ("路径", "单元目录"):
            continue
        for pat in _extract_ban_patterns_from_cell(first):
            if pat not in seen:
                seen.add(pat)
                patterns.append(pat)

    _READ_BANS_CACHE[cache_key] = patterns
    return list(patterns)


def _compile_ban_regexes(
    extra_bans: Sequence[str] = (),
    read_bans_path: pathlib.Path | str | None = DEFAULT_READ_BANS_PATH,
) -> tuple[re.Pattern[str], ...]:
    compiled = list(CORE_REDLINE_REGEXES)
    for pat in [*load_read_bans(read_bans_path), *extra_bans]:
        if isinstance(pat, str) and pat.strip():
            compiled.append(_glob_to_regex(pat))
    return tuple(compiled)


def _path_candidates(
    raw_path: str,
    *,
    repo_path: str | pathlib.Path | None = None,
) -> set[str]:
    """Return literal, expanded, and safely resolved path candidates for ban checks.

    Never resolves arbitrary CLI flags or repo-relative tokens against an unrelated CWD;
    only calls ``os.path.realpath`` for absolute/``~`` paths, ``repo_path``-anchored
    existing paths, or existing relative symlinks whose realpath differs from abspath.
    """
    c = raw_path.strip()
    if not c:
        return set()
    expanded = os.path.expanduser(c)
    candidates = {c, expanded}
    try:
        if os.path.isabs(expanded) or c.startswith("~"):
            candidates.add(os.path.realpath(expanded))
        elif repo_path is not None and not c.startswith("-") and c not in ("&&", "||", ";", "|"):
            anchored = os.path.join(os.fspath(repo_path), expanded)
            if os.path.lexists(anchored):
                candidates.add(os.path.realpath(anchored))
        elif os.path.lexists(expanded):
            real = os.path.realpath(expanded)
            if real != os.path.abspath(expanded):
                candidates.add(real)
    except (OSError, ValueError):
        pass
    return candidates


def banned_path(
    path: Any,
    *,
    extra_bans: Sequence[str] = (),
    read_bans_path: pathlib.Path | str | None = DEFAULT_READ_BANS_PATH,
    repo_path: str | pathlib.Path | None = None,
) -> bool:
    """Return True if literal, expanded, or realpath resolves into a banned location."""
    if not isinstance(path, str) or not path.strip():
        return False
    regexes = _compile_ban_regexes(extra_bans=extra_bans, read_bans_path=read_bans_path)
    candidates = _path_candidates(path, repo_path=repo_path)
    return any(rx.search(cand) for cand in candidates for rx in regexes)


def _extract_command_tokens(command: str) -> list[str]:
    if not command.strip():
        return []
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()
    out: list[str] = []
    for tok in tokens:
        out.append(tok)
        if "=" in tok:
            out.append(tok.split("=", 1)[1])
    return out


def check_no_banned_access(
    *,
    paths: Sequence[str] = (),
    command: str = "",
    diff_text: str = "",
    receipt: EvidenceReceipt | None = None,
    branch_name: str = "",
    extra_bans: Sequence[str] = (),
    read_bans_path: pathlib.Path | str | None = DEFAULT_READ_BANS_PATH,
    repo_path: str | pathlib.Path | None = None,
) -> GateResult:
    """Enforce zero banned path reads/writes across paths, commands, diffs, and receipts."""
    failures: list[str] = []
    regexes = _compile_ban_regexes(extra_bans=extra_bans, read_bans_path=read_bans_path)

    def _is_banned(candidate: str) -> bool:
        checks = _path_candidates(candidate, repo_path=repo_path)
        return bool(checks) and any(rx.search(item) for item in checks for rx in regexes)

    for p in paths:
        if _is_banned(p):
            failures.append(f"banned path access: {p}")

    if branch_name.strip() == "main":
        failures.append("branch_name cannot target protected 'main' branch")

    if command:
        if _MAIN_WRITE_CMD_RE.search(command):
            failures.append(f"command attempts protected main branch mutation: {command}")
        for tok in _extract_command_tokens(command):
            if _is_banned(tok):
                failures.append(f"banned path in command: {tok}")

    if diff_text:
        for line in diff_text.splitlines():
            if line.startswith(("+++ ", "--- ", "diff --git ")):
                for part in line.split():
                    cleaned = part[2:] if part.startswith(("a/", "b/")) else part
                    if _is_banned(cleaned):
                        failures.append(f"banned path in diff header: {cleaned}")
            elif line.startswith("+") and not line.startswith("+++"):
                for tok in re.findall(r"[^\s\"'`<>(),;]+", line[1:]):
                    if ("/" in tok or tok.endswith(".json")) and _is_banned(tok):
                        failures.append(f"banned path reference in diff content: {tok}")

    if receipt is not None:
        if receipt.command:
            for tok in _extract_command_tokens(receipt.command):
                if _is_banned(tok):
                    failures.append(f"banned path in receipt command: {tok}")
        for art in receipt.raw_artifact_paths:
            if _is_banned(art):
                failures.append(f"banned path in receipt artifact: {art}")

    return GateResult(passed=not failures, failures=failures, checked=["banned_paths"])


# ---------------------------------------------- Git tree, diff & real_path gates


def _git(repo_path: str | pathlib.Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_path), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def check_clean_tree(repo_path: str | pathlib.Path) -> GateResult:
    """Verify git working tree has no uncommitted or untracked files."""
    proc = _git(repo_path, "status", "--porcelain")
    if proc.returncode != 0:
        return GateResult(
            passed=False,
            failures=[f"git status failed: {proc.stderr.strip()}"],
            checked=["clean_tree"],
        )
    dirty = [line for line in proc.stdout.splitlines() if line.strip()]
    if dirty:
        return GateResult(
            passed=False,
            failures=[f"dirty worktree ({len(dirty)} entries): {dirty[0]}"],
            checked=["clean_tree"],
        )
    return GateResult(passed=True, checked=["clean_tree"])


def check_real_path_commits(
    repo_path: str | pathlib.Path,
    base_sha: str,
    head_sha: str,
    *,
    require_distinct: bool = True,
) -> GateResult:
    """Verify base_sha and head_sha are real ancestor->descendant commits in repo_path."""
    failures: list[str] = []
    if not _HEAD_RE.fullmatch(base_sha or ""):
        failures.append(f"invalid base_sha format: {base_sha!r}")
    if not _HEAD_RE.fullmatch(head_sha or ""):
        failures.append(f"invalid head_sha format: {head_sha!r}")
    if failures:
        return GateResult(passed=False, failures=failures, checked=["real_path_commits"])

    base_rev = _git(repo_path, "rev-parse", "--verify", f"{base_sha}^{{commit}}")
    if base_rev.returncode != 0:
        failures.append(f"base_sha not found in repo: {base_sha}")
    head_rev = _git(repo_path, "rev-parse", "--verify", f"{head_sha}^{{commit}}")
    if head_rev.returncode != 0:
        failures.append(f"head_sha not found in repo: {head_sha}")
    if failures:
        return GateResult(passed=False, failures=failures, checked=["real_path_commits"])

    full_base = base_rev.stdout.strip()
    full_head = head_rev.stdout.strip()
    if require_distinct and full_base == full_head:
        failures.append(f"base_sha and head_sha must differ (both {full_head})")

    anc = _git(repo_path, "merge-base", "--is-ancestor", full_base, full_head)
    if anc.returncode != 0:
        failures.append(f"base_sha {base_sha} is not an ancestor of head_sha {head_sha}")

    return GateResult(passed=not failures, failures=failures, checked=["real_path_commits"])


def _is_doc_path(path: str) -> bool:
    p = pathlib.PurePosixPath(path.replace("\\", "/"))
    if p.parts and p.parts[0] == "docs":
        return True
    return p.suffix.lower() in _DOC_EXTENSIONS


def check_non_doc_code_diff(
    repo_path: str | pathlib.Path,
    base_sha: str,
    head_sha: str,
) -> GateResult:
    """Ensure the commit range changes at least one non-doc code/config/test file."""
    proc = _git(repo_path, "diff", "--no-renames", "--name-only", f"{base_sha}...{head_sha}")
    if proc.returncode != 0:
        return GateResult(
            passed=False,
            failures=[f"git diff failed: {proc.stderr.strip()}"],
            checked=["non_doc_code_diff"],
        )
    changed = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    code_files = [p for p in changed if not _is_doc_path(p)]
    if not code_files:
        return GateResult(
            passed=False,
            failures=["diff contains zero non-doc code files"],
            checked=["non_doc_code_diff"],
        )
    return GateResult(passed=True, checked=["non_doc_code_diff"])


# ------------------------------------ Anti-shortcut, chip & internal-ID gates


def check_no_internal_ids(
    text: str,
    *,
    extra_patterns: Sequence[re.Pattern[str]] = (),
) -> GateResult:
    """Reject internal wave/node/ticket IDs in added code or comments."""
    failures: list[str] = []
    patterns = (*_INTERNAL_ID_REGEXES, *extra_patterns)
    for line_no, line in enumerate(text.splitlines(), start=1):
        for rx in patterns:
            match = rx.search(line)
            if match:
                failures.append(f"internal ID {match.group(0)!r} found on line {line_no}")
    return GateResult(passed=not failures, failures=failures, checked=["no_internal_ids"])


def check_no_chip_specialization(
    text: str,
    *,
    extra_patterns: Sequence[re.Pattern[str]] = (),
) -> GateResult:
    """Reject hardcoded chip/target specialization tokens in generic code/diffs."""
    failures: list[str] = []
    patterns = (*_CHIP_SPECIALIZATION_REGEXES, *extra_patterns)
    for line_no, line in enumerate(text.splitlines(), start=1):
        for rx in patterns:
            match = rx.search(line)
            if match:
                failures.append(
                    f"chip specialization token {match.group(0)!r} found on line {line_no}"
                )
    return GateResult(passed=not failures, failures=failures, checked=["no_chip_specialization"])


def _added_diff_lines(diff_text: str) -> str:
    added: list[str] = []
    for line in diff_text.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            added.append(line[1:])
    return "\n".join(added)


def check_assertion_non_weakening(diff_text: str) -> GateResult:
    """Reject unified diffs that delete, comment out, or skip assertions without replacement."""
    failures: list[str] = []
    current_file = "<diff>"
    removed_asserts = 0
    added_asserts = 0
    skip_hits: list[str] = []
    commented_asserts: list[str] = []

    def _flush_file() -> None:
        if removed_asserts > added_asserts:
            failures.append(
                f"{current_file}: assertions reduced ({removed_asserts} removed vs {added_asserts} added)"
            )
        for hit in commented_asserts:
            failures.append(f"{current_file}: commented-out assertion: {hit}")
        for hit in skip_hits:
            failures.append(f"{current_file}: newly added skip/xfail directive: {hit}")

    for line in diff_text.splitlines():
        if line.startswith("+++ "):
            _flush_file()
            target = line[4:].strip()
            current_file = target[2:] if target.startswith("b/") else target
            removed_asserts = 0
            added_asserts = 0
            skip_hits = []
            commented_asserts = []
            continue
        if line.startswith("--- ") or line.startswith("diff --git ") or line.startswith("@@ "):
            continue
        if line.startswith("-"):
            body = line[1:].strip()
            if body and not body.startswith(("#", "//")) and _ASSERT_CALL_RE.search(body):
                removed_asserts += 1
        elif line.startswith("+"):
            body = line[1:].strip()
            if not body:
                continue
            if body.startswith(("#", "//")) and _ASSERT_CALL_RE.search(body):
                commented_asserts.append(body[:80])
                continue
            if not body.startswith(("#", "//")) and _ASSERT_CALL_RE.search(body):
                added_asserts += 1
            if _SKIP_MARK_RE.search(body):
                skip_hits.append(body[:80])

    _flush_file()
    return GateResult(passed=not failures, failures=failures, checked=["assertion_non_weakening"])


# ---------------------------------------------- Metric & raw artifact gates


def finite_number(value: Any) -> bool:
    """Return True only for finite int/float values (excluding bool, NaN, Inf)."""
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def check_receipt_metrics(
    receipt: EvidenceReceipt,
    *,
    require_metrics: bool = True,
) -> GateResult:
    """Validate that receipt metrics use explicit finite numeric values, denominator, and stage_rung."""
    failures: list[str] = []
    valid_rungs = {r.value for r in FidelityRung}
    if require_metrics and not receipt.metrics:
        failures.append("receipt.metrics must be non-empty")

    for name, entry in receipt.metrics.items():
        if not isinstance(entry, Mapping):
            failures.append(
                f"metric {name!r} must be a dict with value, denominator, stage_rung"
            )
            continue
        if not finite_number(entry.get("value")):
            failures.append(f"metric {name!r} requires finite numeric 'value'")
        denom = entry.get("denominator")
        if isinstance(denom, bool) or denom is None:
            failures.append(f"metric {name!r} missing valid 'denominator'")
        elif type(denom) in (int, float):
            if not finite_number(denom) or denom <= 0:
                failures.append(f"metric {name!r} numeric 'denominator' must be > 0")
        elif isinstance(denom, str):
            if not denom.strip():
                failures.append(f"metric {name!r} string 'denominator' must be non-empty")
        else:
            failures.append(f"metric {name!r} has invalid 'denominator' type")

        rung = entry.get("stage_rung")
        if not isinstance(rung, str) or rung not in valid_rungs:
            failures.append(
                f"metric {name!r} requires valid 'stage_rung' in {sorted(valid_rungs)}"
            )

    return GateResult(passed=not failures, failures=failures, checked=["receipt_metrics"])


def check_raw_artifacts_and_evidence(
    receipt: EvidenceReceipt,
    *,
    require_existing_artifacts: bool = False,
    require_metrics: bool = True,
    require_tests: bool = False,
    extra_bans: Sequence[str] = (),
) -> GateResult:
    """Reject LLM-prose-only passes; require raw artifact paths, exit status, and typed metrics."""
    result = GateResult(passed=True, checked=["raw_artifacts_and_evidence"])

    if not receipt.raw_artifact_paths:
        result.failures.append("raw_artifact_paths must be non-empty (cannot pass on prose)")
    else:
        for art in receipt.raw_artifact_paths:
            if not isinstance(art, str) or not art.strip():
                result.failures.append("raw_artifact_paths contains empty entry")
                continue
            if banned_path(art, extra_bans=extra_bans):
                result.failures.append(f"raw artifact points to banned path: {art}")
                continue
            if require_existing_artifacts:
                p = pathlib.Path(art)
                if not p.is_file() or p.stat().st_size == 0:
                    result.failures.append(f"missing or empty raw artifact: {art}")

    if receipt.engineering_verdict == EngineeringVerdict.PASS:
        if receipt.process_status != ProcessStatus.EXIT_OK or receipt.exit_code != 0:
            result.failures.append(
                f"engineering PASS inconsistent with status={receipt.process_status.value} exit_code={receipt.exit_code}"
            )
        if receipt.tests_total < 0 or receipt.tests_passed < 0 or receipt.tests_passed > receipt.tests_total:
            result.failures.append("invalid test counts on receipt")
        if require_tests and (receipt.tests_total == 0 or receipt.tests_passed < receipt.tests_total):
            result.failures.append(
                f"tests not fully passing ({receipt.tests_passed}/{receipt.tests_total})"
            )

    result.merge(check_receipt_metrics(receipt, require_metrics=require_metrics))
    result.passed = len(result.failures) == 0
    return result


# ------------------------------------------------------ Structured verdict gate

_VERDICT_VALUES = frozenset({"PASS", "FAIL", "BROKEN"})
_SEVERITY_VALUES = frozenset({"BLOCKER", "MAJOR", "MINOR", "NONE"})
_FENCED_BLOCK_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)
_ANCHORED_VERDICT_LINE_RE = re.compile(
    r"^\s*(?:AUDIT_VERDICT|BOOST_VERDICT|DESIGN_VERDICT|IMPL_VERDICT|RESULT_VERDICT)\s*:\s*(PASS|FAIL|BROKEN)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedVerdict:
    """Structured outcome parsed from an audit or adversary verdict response."""

    verdict: str
    severity: str = "NONE"
    findings: list[str] = field(default_factory=list)
    repro_python: str = ""
    raw_source: str = ""


def _coerce_parsed_json_verdict(obj: Mapping[str, Any], *, raw_source: str) -> ParsedVerdict | None:
    if "verdict" not in obj:
        return None
    verdict = str(obj.get("verdict") or "").strip().upper()
    if verdict not in _VERDICT_VALUES:
        return ParsedVerdict(
            verdict="PARSE_ERROR",
            severity="MAJOR",
            findings=[f"Invalid verdict value in JSON: {obj.get('verdict')!r}"],
            repro_python=str(obj.get("repro_python") or "").strip(),
            raw_source=raw_source,
        )
    sev_raw = str(obj.get("severity") or "").strip().upper()
    if sev_raw in _SEVERITY_VALUES:
        severity = sev_raw
    elif verdict == "PASS":
        severity = "NONE"
    else:
        severity = "MAJOR"

    raw_findings = obj.get("findings")
    if isinstance(raw_findings, list):
        findings = [str(item).strip() for item in raw_findings if str(item).strip()]
    elif isinstance(raw_findings, str) and raw_findings.strip():
        findings = [raw_findings.strip()]
    else:
        findings = []

    repro_python = str(obj.get("repro_python") or "").strip()
    return ParsedVerdict(
        verdict=verdict,
        severity=severity,
        findings=findings,
        repro_python=repro_python,
        raw_source=raw_source,
    )


def _extract_json_verdicts(text: str) -> list[tuple[dict[str, Any], str]]:
    """Extract JSON dict objects containing a 'verdict' key (fenced blocks first, then standalone)."""
    found: list[tuple[dict[str, Any], str]] = []
    for block in _FENCED_BLOCK_RE.findall(text):
        candidate = block.strip()
        if not candidate:
            continue
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            found.append((obj, "json_fenced"))
    if found:
        return found

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict) and "verdict" in obj:
                return [(obj, "json_standalone")]
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        brace = text.find("{", idx)
        if brace == -1:
            break
        try:
            obj, end = decoder.raw_decode(text, brace)
        except json.JSONDecodeError:
            idx = brace + 1
            continue
        if isinstance(obj, dict) and "verdict" in obj:
            found.append((obj, "json_standalone"))
        idx = max(brace + 1, end)
    return found


def parse_structured_verdict(text: str, *, mode: str = "audit") -> ParsedVerdict:
    """Parse an audit/boost verdict from JSON first, else an anchored standalone final verdict line.

    Contradictory standalone verdict lines, prose-embedded verdict strings, or missing verdicts
    return ``verdict="PARSE_ERROR"`` and NEVER yield a false ``PASS``.
    """
    _ = mode
    body = text or ""
    if not body.strip():
        return ParsedVerdict(
            verdict="PARSE_ERROR",
            severity="NONE",
            findings=[],
            repro_python="",
            raw_source="empty",
        )

    json_candidates = _extract_json_verdicts(body)
    if json_candidates:
        parsed_items: list[ParsedVerdict] = []
        for obj, src in json_candidates:
            pv = _coerce_parsed_json_verdict(obj, raw_source=src)
            if pv is not None:
                parsed_items.append(pv)
        if parsed_items:
            distinct_verdicts = {p.verdict for p in parsed_items}
            if len(distinct_verdicts) > 1:
                return ParsedVerdict(
                    verdict="PARSE_ERROR",
                    severity="MAJOR",
                    findings=["Contradictory JSON verdict objects in response"],
                    repro_python="",
                    raw_source="json_contradictory",
                )
            return parsed_items[-1]

    non_empty_lines = [ln for ln in body.splitlines() if ln.strip()]
    if not non_empty_lines:
        return ParsedVerdict(
            verdict="PARSE_ERROR",
            severity="NONE",
            findings=[],
            repro_python="",
            raw_source="empty",
        )

    matched_verdicts: list[str] = []
    for ln in non_empty_lines:
        m = _ANCHORED_VERDICT_LINE_RE.match(ln)
        if m:
            matched_verdicts.append(m.group(1).upper())

    final_match = _ANCHORED_VERDICT_LINE_RE.match(non_empty_lines[-1])
    if not matched_verdicts or final_match is None:
        return ParsedVerdict(
            verdict="PARSE_ERROR",
            severity="MAJOR" if matched_verdicts else "NONE",
            findings=["Missing anchored standalone final verdict line"],
            repro_python="",
            raw_source="prose_only",
        )

    if len(set(matched_verdicts)) > 1:
        return ParsedVerdict(
            verdict="PARSE_ERROR",
            severity="MAJOR",
            findings=[f"Contradictory standalone verdict lines: {sorted(set(matched_verdicts))}"],
            repro_python="",
            raw_source="contradictory_lines",
        )

    verdict = final_match.group(1).upper()
    severity = "NONE" if verdict == "PASS" else "MAJOR"
    return ParsedVerdict(
        verdict=verdict,
        severity=severity,
        findings=[],
        repro_python="",
        raw_source="anchored_line",
    )


# ------------------------------------------------------------ Composite gate


def run_cpu_gate(
    repo_path: str | pathlib.Path,
    base_sha: str,
    head_sha: str,
    writable_paths: Sequence[str],
    *,
    goal_prefixes: Sequence[str] = (),
    bookkeeping_paths: Sequence[str] = (),
    extra_bans: Sequence[str] = (),
    command: str = "",
    branch_name: str = "",
    receipt: EvidenceReceipt | None = None,
    require_clean: bool = True,
    require_non_doc: bool = True,
    require_distinct_commits: bool = True,
    require_chip_neutral: bool = True,
    read_bans_path: pathlib.Path | str | None = DEFAULT_READ_BANS_PATH,
) -> GateResult:
    """Run all deterministic CPU gates against a candidate repo commit."""
    res = GateResult(passed=True)
    if require_clean:
        res.merge(check_clean_tree(repo_path))

    res.merge(
        check_real_path_commits(
            repo_path, base_sha, head_sha, require_distinct=require_distinct_commits
        )
    )
    if not res.passed:
        return res

    names_proc = _git(repo_path, "diff", "--no-renames", "--name-only", f"{base_sha}...{head_sha}")
    diff_proc = _git(repo_path, "diff", "--no-renames", f"{base_sha}...{head_sha}")
    if names_proc.returncode != 0 or diff_proc.returncode != 0:
        res.failures.append(f"git diff failed: {names_proc.stderr.strip() or diff_proc.stderr.strip()}")
        res.passed = False
        return res

    changed_paths = [p.strip() for p in names_proc.stdout.splitlines() if p.strip()]
    diff_text = diff_proc.stdout
    added_text = _added_diff_lines(diff_text)

    if require_non_doc:
        res.merge(check_non_doc_code_diff(repo_path, base_sha, head_sha))

    res.merge(
        validate_writable_paths(
            changed_paths,
            writable_paths,
            goal_prefixes=goal_prefixes,
            bookkeeping_paths=bookkeeping_paths,
        )
    )
    res.merge(
        check_no_banned_access(
            paths=changed_paths,
            command=command,
            diff_text=diff_text,
            receipt=receipt,
            branch_name=branch_name,
            extra_bans=extra_bans,
            read_bans_path=read_bans_path,
            repo_path=repo_path,
        )
    )
    res.merge(check_assertion_non_weakening(diff_text))
    res.merge(check_no_internal_ids(added_text))
    if require_chip_neutral:
        res.merge(check_no_chip_specialization(added_text))
    if receipt is not None:
        res.merge(check_raw_artifacts_and_evidence(receipt, extra_bans=extra_bans))

    res.passed = len(res.failures) == 0
    return res


# ------------------------------- Failure domain & empirical validity gates

_INFRA_TRANSIENT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bhq\s+sigint\b", re.IGNORECASE),
    re.compile(r"\b(?:429|502|503|504)\b.*\b(?:rate\s*limit|bad\s*gateway|service\s*unavailable|gateway\s*timeout)\b", re.IGNORECASE),
    re.compile(r"\brate[_\s-]?limit(?:ed)?\b", re.IGNORECASE),
    re.compile(r"\bconnection\s+reset\s+by\s+peer\b", re.IGNORECASE),
    re.compile(r"\bno\s+space\s+left\s+on\s+device\b", re.IGNORECASE),
    re.compile(r"\bdisk\s+quota\s+exceeded\b", re.IGNORECASE),
    re.compile(r"\bdatabase\s+is\s+locked\b", re.IGNORECASE),
)

_ENV_MISMATCH_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcuda[_\s-]?unavailable\b", re.IGNORECASE),
    re.compile(r"\btorch\.cuda\.is_available\(\)\s+is\s+false\b", re.IGNORECASE),
    re.compile(r"\bno\s+cuda\s+gpus?\s+are\s+available\b", re.IGNORECASE),
    re.compile(r"\bnvidia-smi:\s*(?:command\s+)?not\s+found\b", re.IGNORECASE),
    re.compile(r"\b\w*(?:eval|data|dataset)inputerror\b", re.IGNORECASE),
    re.compile(r"\bmodulenotfounderror\b", re.IGNORECASE),
    re.compile(r"\bimporterror\b", re.IGNORECASE),
    re.compile(r"\bunrecognized\s+arguments?\s*:", re.IGNORECASE),
    re.compile(r"\bno\s+such\s+option\s*:", re.IGNORECASE),
    re.compile(r"\berror:\s+invalid\s+choice\s*:", re.IGNORECASE),
    re.compile(r"\bmissing[_\s-]?(?:dataset|input|venv|checkpoint)\b", re.IGNORECASE),
)

_REVIEW_STAGES = frozenset({
    StageKind.BLIND_AUDIT,
    StageKind.DESIGN_REVIEW,
    StageKind.IMPL_REVIEW,
    StageKind.RESULT_REVIEW,
})

_EMPIRICAL_STAGES = frozenset({
    StageKind.EXP_SCREEN,
    StageKind.EXP_CONFIRM,
})

_EXTERNAL_CD_RE = re.compile(r"(?:^|[;&|\n])\s*cd\s+(['\"]?)(/[^\s;|&'\"]+)")


def _has_finite_primary_metric(
    receipt: EvidenceReceipt, goal: ResearchGoal | None
) -> bool:
    if goal is None or not goal.primary_metric:
        return bool(receipt.metrics)
    if goal.primary_metric not in receipt.metrics:
        return False
    val = receipt.metrics[goal.primary_metric]
    if isinstance(val, Mapping) and "value" in val:
        val = val["value"]
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        return False
    return math.isfinite(float(val))


def classify_failure_domain(
    receipt: EvidenceReceipt,
    *,
    goal: ResearchGoal | None = None,
    stderr_or_log_tail: str = "",
) -> FailureDomain:
    """Classify an EvidenceReceipt into an orthogonal FailureDomain.

    Guarantees that timeouts, lease loss, OOM, CUDA/venv/CLI mismatches, and
    engineering/review failures NEVER masquerade as empirical scientific outcomes.
    """
    if receipt.process_status == ProcessStatus.CANCELLED:
        return FailureDomain.NONE

    if receipt.process_status in (
        ProcessStatus.TIMEOUT,
        ProcessStatus.OOM,
        ProcessStatus.LEASE_LOST,
    ):
        return FailureDomain.INFRA_TRANSIENT

    diag_parts = [
        receipt.failure_reason or "",
        stderr_or_log_tail or "",
        str(receipt.metrics.get("error") or ""),
        str(receipt.metrics.get("skip_reason") or ""),
    ]
    diag_text = "\n".join(p for p in diag_parts if p)

    if receipt.process_status != ProcessStatus.EXIT_OK:
        for pat in _INFRA_TRANSIENT_PATTERNS:
            if pat.search(diag_text):
                return FailureDomain.INFRA_TRANSIENT

    if (
        receipt.process_status == ProcessStatus.ENV_ERROR
        or receipt.evidence_validity == EvidenceValidity.MISSING_INPUT
    ):
        return FailureDomain.ENV_CONFIG_MISMATCH

    for pat in _ENV_MISMATCH_PATTERNS:
        if pat.search(diag_text):
            return FailureDomain.ENV_CONFIG_MISMATCH

    if receipt.stage in _REVIEW_STAGES:
        if receipt.audit_verdict == AuditVerdict.FAIL:
            return FailureDomain.ENGINEERING_DEFECT
        if receipt.audit_verdict == AuditVerdict.PARSE_ERROR:
            return FailureDomain.INFRA_TRANSIENT
        if (
            receipt.process_status == ProcessStatus.EXIT_OK
            and receipt.audit_verdict == AuditVerdict.PASS
        ):
            return FailureDomain.NONE
        return FailureDomain.ENGINEERING_DEFECT

    if receipt.stage in (StageKind.PROPOSE, StageKind.CODE, StageKind.CPU_GATE):
        if (
            receipt.process_status == ProcessStatus.EXIT_OK
            and receipt.engineering_verdict
            in (EngineeringVerdict.PASS, EngineeringVerdict.SKIPPED)
            and receipt.audit_verdict != AuditVerdict.FAIL
        ):
            return FailureDomain.NONE
        return FailureDomain.ENGINEERING_DEFECT

    if receipt.stage in _EMPIRICAL_STAGES:
        if receipt.audit_verdict == AuditVerdict.FAIL:
            return FailureDomain.ENGINEERING_DEFECT
        has_primary = _has_finite_primary_metric(receipt, goal)
        if (
            receipt.evidence_validity == EvidenceValidity.VALID_EMPIRICAL
            and has_primary
            and receipt.engineering_verdict == EngineeringVerdict.PASS
            and receipt.process_status == ProcessStatus.EXIT_OK
        ):
            return FailureDomain.SCIENTIFIC_OUTCOME
        if (
            receipt.process_status == ProcessStatus.EXIT_NONZERO
            and receipt.evidence_validity == EvidenceValidity.VALID_EMPIRICAL
            and has_primary
            and receipt.scientific_verdict
            in (ScientificVerdict.REFUTED, ScientificVerdict.VALID_NEGATIVE)
            and (goal is None or goal.primary_metric != "tests_passed")
            and receipt.engineering_verdict != EngineeringVerdict.FAIL_GATE
        ):
            return FailureDomain.SCIENTIFIC_OUTCOME
        if receipt.evidence_validity in (
            EvidenceValidity.MISSING_PRIMARY_METRIC,
            EvidenceValidity.INVALID_SYNTHETIC,
        ):
            if receipt.process_status == ProcessStatus.EXIT_OK:
                return FailureDomain.ENV_CONFIG_MISMATCH
            return FailureDomain.ENGINEERING_DEFECT
        if receipt.process_status != ProcessStatus.EXIT_OK or receipt.engineering_verdict in (
            EngineeringVerdict.FAIL_GATE,
            EngineeringVerdict.FAIL_TEST,
        ):
            return FailureDomain.ENGINEERING_DEFECT

    if (
        receipt.process_status == ProcessStatus.EXIT_OK
        and receipt.engineering_verdict
        in (EngineeringVerdict.PASS, EngineeringVerdict.SKIPPED)
    ):
        return FailureDomain.NONE
    return FailureDomain.ENGINEERING_DEFECT


def evaluate_kill_criteria(
    receipt: EvidenceReceipt,
    hypothesis: HypothesisNode | None = None,
    goal: ResearchGoal | None = None,
) -> str | None:
    """Evaluate empirical kill-criteria expressions against finite receipt metrics."""
    criteria: list[str] = []
    if hypothesis is not None:
        criteria.extend(str(c) for c in (hypothesis.kill_criteria or []) if str(c).strip())
    if goal is not None:
        for gc in goal.kill_criteria or []:
            if str(gc).strip() and str(gc) not in criteria:
                criteria.append(str(gc))
    if not criteria:
        return None

    numeric_metrics: dict[str, float] = {}
    for k, raw in (receipt.metrics or {}).items():
        val = raw.get("value") if isinstance(raw, Mapping) else raw
        if isinstance(val, (int, float)) and not isinstance(val, bool) and math.isfinite(float(val)):
            numeric_metrics[str(k)] = float(val)
    if goal is not None and goal.primary_metric == "tests_passed":
        numeric_metrics.setdefault("tests_passed", float(receipt.tests_passed))
        numeric_metrics.setdefault("tests_total", float(receipt.tests_total))

    op_re = re.compile(
        r"^\s*([A-Za-z_][A-Za-z0-9_.-]*)\s*(<=|>=|==|!=|<|>)\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)\s*$"
    )
    for crit in criteria:
        m = op_re.match(crit)
        if not m:
            continue
        key, op, rhs_s = m.group(1), m.group(2), m.group(3)
        if key in ("tests_passed", "tests_total") and (goal is None or goal.primary_metric != "tests_passed"):
            continue
        if key not in numeric_metrics:
            continue
        lhs = numeric_metrics[key]
        rhs = float(rhs_s)
        hit = (
            (op == "<" and lhs < rhs)
            or (op == "<=" and lhs <= rhs)
            or (op == ">" and lhs > rhs)
            or (op == ">=" and lhs >= rhs)
            or (op == "==" and lhs == rhs)
            or (op == "!=" and lhs != rhs)
        )
        if hit:
            return crit
    return None


def _extract_metric_int(metrics: Mapping[str, Any] | None, key: str) -> int | None:
    if not isinstance(metrics, Mapping) or key not in metrics:
        return None
    raw = metrics.get(key)
    if isinstance(raw, Mapping):
        raw = raw.get("value")
    if raw is None or isinstance(raw, bool):
        return None
    try:
        return int(raw)
    except (TypeError, ValueError):
        return None


def validate_receipt_epoch(
    receipt: EvidenceReceipt | None,
    goal: ResearchGoal | None,
    *,
    task: StageTask | None = None,
    preserved_receipt_ids: Sequence[str] | set[str] = (),
) -> EvidenceValidity:
    """Unconditionally fence `goal_version` and global execution-fence epochs on BOTH task and receipt.

    Unaffected sibling tasks whose global execution-fence epochs were synced across a peer branch
    revision carry `metrics['exec_fence_plan_epoch']` / `metrics['exec_fence_human_epoch']` while
    preserving the branch's own scientific revision in `receipt.plan_epoch` / `receipt.human_epoch`.
    """
    if receipt is not None and receipt.evidence_validity in (
        EvidenceValidity.HUMAN_INVALIDATED,
        EvidenceValidity.SUPERSEDED_EPOCH,
        EvidenceValidity.STALE_GOAL_VERSION,
    ):
        return receipt.evidence_validity

    if goal is not None:
        if task is not None and int(task.goal_version) != int(goal.version):
            return EvidenceValidity.STALE_GOAL_VERSION
        if receipt is not None and int(receipt.goal_version) != int(goal.version):
            return EvidenceValidity.STALE_GOAL_VERSION

        goal_plan_epoch = int(getattr(goal, "plan_epoch", 0) or 0)
        preserved_set = {str(x) for x in preserved_receipt_ids if str(x)}
        is_preserved = receipt is not None and receipt.receipt_id in preserved_set
        if not is_preserved:
            if task is not None and int(getattr(task, "plan_epoch", 0) or 0) != goal_plan_epoch:
                return EvidenceValidity.SUPERSEDED_EPOCH
            if receipt is not None:
                rcpt_fence_pe = _extract_metric_int(receipt.metrics, "exec_fence_plan_epoch")
                eff_rcpt_pe = (
                    rcpt_fence_pe
                    if rcpt_fence_pe is not None
                    else int(getattr(receipt, "plan_epoch", 0) or 0)
                )
                if eff_rcpt_pe != goal_plan_epoch:
                    return EvidenceValidity.SUPERSEDED_EPOCH

        if not is_preserved:
            if task is not None and int(task.human_epoch) != int(goal.human_epoch):
                return EvidenceValidity.HUMAN_INVALIDATED
            if receipt is not None:
                rcpt_fence_he = _extract_metric_int(receipt.metrics, "exec_fence_human_epoch")
                eff_rcpt_he = (
                    rcpt_fence_he
                    if rcpt_fence_he is not None
                    else int(receipt.human_epoch)
                )
                if eff_rcpt_he != int(goal.human_epoch):
                    return EvidenceValidity.HUMAN_INVALIDATED

    if receipt is not None:
        return receipt.evidence_validity
    return EvidenceValidity.VALID_ENGINEERING_ONLY


def is_receipt_authoritative_for_hypothesis(
    receipt: EvidenceReceipt,
    goal: ResearchGoal,
    hyp: HypothesisNode,
    *,
    preserved_receipt_ids: Sequence[str] | set[str] = (),
    latest_exp_plan: ExperimentPlan | None = None,
) -> bool:
    """Return True ONLY if `receipt` belongs to `(goal, hyp)` and matches `hyp`'s current branch scientific revision.

    Separates per-branch scientific revision identity (`lineage_sci_revision`, `lineage_plan_epoch`,
    `lineage_human_epoch`) from global execution-fence epochs synced onto unaffected sibling tasks:
    revising branch A never invalidates unaffected sibling B's new evidence or revives stale evidence on A.
    """
    if receipt.goal_id != goal.goal_id or receipt.hypothesis_id != hyp.hypothesis_id:
        return False
    if int(receipt.goal_version) != int(goal.version):
        return False
    if receipt.evidence_validity in (
        EvidenceValidity.SUPERSEDED_EPOCH,
        EvidenceValidity.HUMAN_INVALIDATED,
        EvidenceValidity.STALE_GOAL_VERSION,
    ):
        return False

    hparams = dict(hyp.params or {})
    preserved_set: set[str] = {str(x) for x in preserved_receipt_ids if str(x)}
    for pid in hparams.get("preserved_receipt_ids") or []:
        if str(pid).strip():
            preserved_set.add(str(pid).strip())

    if receipt.receipt_id not in preserved_set:
        hyp_sci_rev = int(hparams.get("lineage_sci_revision", 0) or 0)
        rcpt_sci_rev = _extract_metric_int(receipt.metrics, "sci_revision")
        if rcpt_sci_rev is not None and rcpt_sci_rev != hyp_sci_rev:
            return False

        ep_plan_epoch = (
            int(getattr(latest_exp_plan, "plan_epoch", 0) or 0)
            if latest_exp_plan is not None
            else 0
        )
        ep_human_epoch = (
            int(getattr(latest_exp_plan, "human_epoch", 0) or 0)
            if latest_exp_plan is not None
            else 0
        )
        ep_plan_ver = (
            int(getattr(latest_exp_plan, "plan_version", 1) or 1)
            if latest_exp_plan is not None
            else 1
        )
        has_explicit_branch_revision = (
            hyp_sci_rev > 0
            or "lineage_plan_epoch" in hparams
            or "lineage_human_epoch" in hparams
            or ep_plan_epoch > 0
            or ep_human_epoch > 0
            or ep_plan_ver > 1
        )
        if has_explicit_branch_revision:
            req_plan_epoch = max(int(hparams.get("lineage_plan_epoch", 0) or 0), ep_plan_epoch)
            if int(getattr(receipt, "plan_epoch", 0) or 0) != req_plan_epoch:
                return False
            if "lineage_human_epoch" in hparams or ep_human_epoch > 0:
                req_human_epoch = max(int(hparams.get("lineage_human_epoch", 0) or 0), ep_human_epoch)
                if int(getattr(receipt, "human_epoch", 0) or 0) != req_human_epoch:
                    return False
        else:
            # Untouched initial revision (`sci_revision == 0`, `req_plan_epoch == 0`):
            # when worker explicitly stamped `exec_fence_plan_epoch` separately, `receipt.plan_epoch` must be 0.
            if (
                rcpt_sci_rev is None
                and isinstance(receipt.metrics, Mapping)
                and "exec_fence_plan_epoch" in receipt.metrics
                and int(getattr(receipt, "plan_epoch", 0) or 0) != 0
            ):
                return False

    return is_valid_empirical_outcome(receipt, goal)


def is_valid_empirical_outcome(
    receipt: EvidenceReceipt,
    goal: ResearchGoal,
    *,
    hyp: HypothesisNode | None = None,
    preserved_receipt_ids: Sequence[str] | set[str] = (),
    latest_exp_plan: ExperimentPlan | None = None,
) -> bool:
    """Universal predicate gating CONFIRMED, VALID_NEGATIVE, REFUTED, and graveyard writes."""
    if receipt.goal_id != goal.goal_id:
        return False
    if receipt.stage not in _EMPIRICAL_STAGES:
        return False
    if int(receipt.goal_version) != int(goal.version):
        return False
    if receipt.evidence_validity != EvidenceValidity.VALID_EMPIRICAL:
        return False
    if hyp is not None:
        if receipt.hypothesis_id != hyp.hypothesis_id:
            return False
        hparams = dict(hyp.params or {})
        preserved_set: set[str] = {str(x) for x in preserved_receipt_ids if str(x)}
        for pid in hparams.get("preserved_receipt_ids") or []:
            if str(pid).strip():
                preserved_set.add(str(pid).strip())
        has_hyp_lineage = (
            "lineage_plan_epoch" in hparams
            or "lineage_human_epoch" in hparams
            or latest_exp_plan is not None
        )
        if has_hyp_lineage and receipt.receipt_id not in preserved_set:
            req_plan_epoch = int(hparams.get("lineage_plan_epoch", 0) or 0)
            if latest_exp_plan is not None:
                req_plan_epoch = max(req_plan_epoch, int(getattr(latest_exp_plan, "plan_epoch", 0) or 0))
            if int(getattr(receipt, "plan_epoch", 0) or 0) != req_plan_epoch:
                return False
            if "lineage_human_epoch" in hparams:
                req_human_epoch = int(hparams.get("lineage_human_epoch", 0) or 0)
                if int(getattr(receipt, "human_epoch", 0) or 0) != req_human_epoch:
                    return False
    if receipt.audit_verdict == AuditVerdict.FAIL:
        return False
    if not _has_finite_primary_metric(receipt, goal):
        return False

    if receipt.failure_domain not in (
        FailureDomain.NONE,
        FailureDomain.SCIENTIFIC_OUTCOME,
    ):
        return False
    if classify_failure_domain(receipt, goal=goal) != FailureDomain.SCIENTIFIC_OUTCOME:
        return False

    if (
        receipt.process_status == ProcessStatus.EXIT_OK
        and receipt.engineering_verdict == EngineeringVerdict.PASS
    ):
        return True

    if (
        receipt.process_status == ProcessStatus.EXIT_NONZERO
        and receipt.failure_domain == FailureDomain.SCIENTIFIC_OUTCOME
        and receipt.scientific_verdict
        in (ScientificVerdict.REFUTED, ScientificVerdict.VALID_NEGATIVE)
        and goal.primary_metric != "tests_passed"
        and receipt.engineering_verdict != EngineeringVerdict.FAIL_GATE
    ):
        return True

    return False


def is_authenticated_negative_empirical_outcome(
    receipt: EvidenceReceipt,
    hypothesis: HypothesisNode | None = None,
    *,
    goal: ResearchGoal | None = None,
    preserved_receipt_ids: Sequence[str] | set[str] = (),
) -> bool:
    """Strict guard: require VERIFIED empirical SCIENTIFIC_OUTCOME + EXIT_OK + negative verdict or kill hit.

    Positive (`CONFIRMED`) and `UNKNOWN` empirical receipts NEVER authorize refutation or graveyarding.
    """
    if receipt.stage not in _EMPIRICAL_STAGES:
        return False
    if receipt.evidence_validity != EvidenceValidity.VERIFIED:
        return False
    if receipt.process_status != ProcessStatus.EXIT_OK:
        return False
    if receipt.engineering_verdict != EngineeringVerdict.PASS:
        return False
    if receipt.audit_verdict == AuditVerdict.FAIL:
        return False
    if goal is not None:
        if not is_valid_empirical_outcome(
            receipt,
            goal,
            hyp=hypothesis,
            preserved_receipt_ids=preserved_receipt_ids,
        ):
            return False
    else:
        if not _has_finite_primary_metric(receipt, None):
            return False
        if receipt.failure_domain != FailureDomain.SCIENTIFIC_OUTCOME:
            if not (
                receipt.failure_domain == FailureDomain.NONE
                and classify_failure_domain(receipt) == FailureDomain.SCIENTIFIC_OUTCOME
            ):
                return False

    eff_dom = (
        receipt.failure_domain
        if receipt.failure_domain != FailureDomain.NONE
        else classify_failure_domain(receipt, goal=goal)
    )
    if eff_dom != FailureDomain.SCIENTIFIC_OUTCOME:
        return False

    kill_hit = evaluate_kill_criteria(receipt, hypothesis, goal)
    if receipt.scientific_verdict == ScientificVerdict.CONFIRMED and kill_hit is None:
        return False
    if receipt.scientific_verdict in (ScientificVerdict.REFUTED, ScientificVerdict.VALID_NEGATIVE):
        return True
    if kill_hit is not None and receipt.scientific_verdict != ScientificVerdict.CONFIRMED:
        return True
    return False


def map_timeout_verdict(process_status: ProcessStatus) -> tuple[ScientificVerdict, str]:
    """Ensure TIMEOUT maps strictly to UNKNOWN / BLOCKED_TIMEOUT, never REFUTED."""
    if process_status == ProcessStatus.TIMEOUT:
        return ScientificVerdict.UNKNOWN, "BLOCKED_TIMEOUT"
    return ScientificVerdict.UNKNOWN, process_status.value


def validate_experiment_plan_gate(
    plan: ExperimentPlan,
    goal: ResearchGoal | None = None,
    hyp: HypothesisNode | None = None,
    *,
    repo_path: str | pathlib.Path | None = None,
    require_existing_venv: bool = False,
    extra_bans: Sequence[str] = (),
) -> GateResult:
    """Validate an ExperimentPlan before empirical or gate execution."""
    res = GateResult(passed=True, checked=["experiment_plan_gate"])

    if goal is not None and plan.goal_id != goal.goal_id:
        res.failures.append(
            f"Cross-goal ExperimentPlan ownership violation: plan.goal_id={plan.goal_id!r} != goal.goal_id={goal.goal_id!r}"
        )
    if hyp is not None:
        if plan.hypothesis_id != hyp.hypothesis_id:
            res.failures.append(
                f"ExperimentPlan hypothesis mismatch: plan.hypothesis_id={plan.hypothesis_id!r} != hyp.hypothesis_id={hyp.hypothesis_id!r}"
            )
        if goal is not None and hyp.goal_id != goal.goal_id:
            res.failures.append(
                f"Cross-goal Hypothesis ownership violation: hyp.goal_id={hyp.goal_id!r} != goal.goal_id={goal.goal_id!r}"
            )

    if not plan.falsifiable_claim.strip():
        res.failures.append("ExperimentPlan.falsifiable_claim must be non-empty")

    cb = plan.control_baseline if isinstance(plan.control_baseline, Mapping) else {}
    if not cb:
        res.failures.append("ExperimentPlan.control_baseline must be a non-empty baseline specification")
    else:
        pm = str(cb.get("primary_metric") or (goal.primary_metric if goal is not None else "")).strip()
        if not pm:
            res.failures.append("ExperimentPlan.control_baseline must declare a non-empty primary_metric")
        has_base_ref = (
            "baseline_value" in cb
            or bool(str(cb.get("baseline_sha") or "").strip())
            or bool(str(cb.get("baseline_name") or "").strip())
        )
        if not has_base_ref:
            res.failures.append("ExperimentPlan.control_baseline must specify baseline_value, baseline_sha, or baseline_name")

    if not (30 <= int(plan.gate_timeout_s) <= 180):
        res.failures.append(
            f"gate_timeout_s={plan.gate_timeout_s} outside mandatory [30, 180] range"
        )
    if int(plan.screen_timeout_s) < 30 or int(plan.confirm_timeout_s) < 30:
        res.failures.append("screen_timeout_s and confirm_timeout_s must be >= 30s")

    allowed_hw = {"cpu_only", "local_gpu", "cloud_gpu"}
    if plan.required_hardware not in allowed_hw:
        res.failures.append(
            f"Invalid required_hardware={plan.required_hardware!r}; expected one of {sorted(allowed_hw)}"
        )
    if int(plan.min_vram_mb) > 0 and plan.required_hardware == "cpu_only":
        res.failures.append(
            "min_vram_mb > 0 requires required_hardware in {'local_gpu', 'cloud_gpu'}"
        )

    eff_bans = list(extra_bans)
    if goal is not None:
        eff_bans.extend(goal.banned_paths)

    if plan.venv_python.strip():
        if banned_path(plan.venv_python, extra_bans=eff_bans):
            res.failures.append(f"venv_python points to banned path: {plan.venv_python}")
        elif require_existing_venv:
            vp = pathlib.Path(plan.venv_python).expanduser()
            if not vp.is_file() or not os.access(vp, os.X_OK):
                res.failures.append(f"venv_python is missing or not executable: {plan.venv_python}")

    eff_repo = repo_path or (goal.repo_path if goal is not None else None)
    repo_resolved = (
        pathlib.Path(eff_repo).expanduser().resolve() if eff_repo else None
    )

    if hyp is not None:
        for label, cmd in (
            ("acceptance_cmd", hyp.acceptance_cmd),
            ("screen_cmd", hyp.screen_cmd),
            ("confirm_cmd", hyp.confirm_cmd or ""),
        ):
            if not cmd.strip():
                continue
            res.merge(
                check_no_banned_access(
                    command=cmd,
                    extra_bans=eff_bans,
                    repo_path=eff_repo,
                )
            )
            if repo_resolved is not None:
                for m in _EXTERNAL_CD_RE.finditer(cmd):
                    target_dir = pathlib.Path(m.group(2)).expanduser().resolve()
                    try:
                        target_dir.relative_to(repo_resolved)
                    except ValueError:
                        res.failures.append(
                            f"{label} escapes worktree via external cd to {m.group(2)!r}"
                        )

    res.passed = len(res.failures) == 0
    return res
