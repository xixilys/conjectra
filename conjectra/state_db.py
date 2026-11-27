"""SQLite WAL state ledger and append-only event log for conjectra."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import pathlib
from pathlib import Path
import sqlite3
import threading
import time
import uuid
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from conjectra.types import (
    EvidenceReceipt,
    ExecutionPlan,
    ExperimentPlan,
    FailureDomain,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ResearchGoal,
    ResearchPlan,
    ReviewKind,
    ReviewReceipt,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
    WorkflowNode,
)

TERMINAL_TASK_SUCCESS = frozenset({"DONE", "SUCCEEDED", "PASSED"})


class StateDBError(Exception):
    """Base error for StateDB ledger operations."""


class DuplicateHypothesisError(StateDBError, ValueError):
    """Raised when a hypothesis matches an existing or graveyarded fingerprint."""


class GraveyardDuplicateError(DuplicateHypothesisError):
    """Raised when a hypothesis repeats a previously killed/graveyarded experiment."""


class StateTransitionError(StateDBError, ValueError):
    """Raised when an atomic state transition fails its expected precondition."""


class FencedStaleWriteError(StateTransitionError):
    """Raised when an atomic `(goal_version, human_epoch, attempt, lease_id)` fence fails."""

    def __init__(self, message: str, *, reason_code: str = "FENCE_FAILED") -> None:
        super().__init__(message)
        self.reason_code = reason_code


def resolve_arr_home(arr_home: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """Resolve CONJECTRA_HOME from explicit argument, environment variable, or ~/.conjectra."""
    raw_val: str | os.PathLike[str] | None = arr_home
    if raw_val is None:
        env_val = os.environ.get("CONJECTRA_HOME")
        if env_val:
            raw_val = env_val
    if raw_val is not None:
        s = os.fspath(raw_val)
        return pathlib.Path(s).expanduser().resolve()
    return (pathlib.Path.home() / ".conjectra").resolve()


def _normalize_rel_path(path: str) -> str:
    s = path.strip().replace("\\", "/")
    while s.startswith("./"):
        s = s[2:]
    while "//" in s:
        s = s.replace("//", "/")
    return s.rstrip("/")


def compute_hypothesis_fingerprint(
    goal_id: str,
    kind: HypothesisKind | str,
    mechanism: str,
    params: dict[str, Any] | None = None,
    writable_paths: Sequence[str] = (),
    screen_cmd: str = "",
) -> str:
    """Compute a deterministic normalized SHA-256 fingerprint for deduplication."""
    kind_val = kind.value if isinstance(kind, HypothesisKind) else HypothesisKind(kind).value
    norm_mech = " ".join(str(mechanism).split()).lower()
    norm_cmd = " ".join(str(screen_cmd).split())
    norm_paths = sorted(
        {_normalize_rel_path(p) for p in writable_paths if isinstance(p, str) and p.strip()}
    )
    canonical = {
        "goal_id": str(goal_id).strip(),
        "kind": kind_val,
        "mechanism": norm_mech,
        "params": params or {},
        "writable_paths": norm_paths,
        "screen_cmd": norm_cmd,
    }
    blob = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def fingerprint_for_node(node: HypothesisNode) -> str:
    """Return normalized fingerprint for a HypothesisNode, computing one if empty."""
    if node.fingerprint and node.fingerprint.strip():
        return node.fingerprint.strip().lower()
    return compute_hypothesis_fingerprint(
        goal_id=node.goal_id,
        kind=node.kind,
        mechanism=node.mechanism,
        params=node.params,
        writable_paths=node.writable_paths,
        screen_cmd=node.screen_cmd,
    )


def evaluate_promotion_git_state(
    wt_path: Path | None,
    exp_branch: str,
    base_sha: str,
    cand_sha: str,
) -> dict[str, Any]:
    """Read promotion worktree Git HEAD/branch/expected-ref state requiring positive proof of unchanged base."""
    import subprocess as _subp_prom

    actual_head = ""
    actual_branch = ""
    if wt_path is not None and wt_path.is_dir():
        try:
            p_head = _subp_prom.run(
                ["git", "-C", str(wt_path), "rev-parse", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if p_head.returncode == 0:
                actual_head = p_head.stdout.strip()
            p_br = _subp_prom.run(
                ["git", "-C", str(wt_path), "symbolic-ref", "-q", "--short", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if p_br.returncode == 0:
                actual_branch = p_br.stdout.strip()
        except Exception:
            actual_head = ""

    branch_ok = bool(actual_branch) and (not exp_branch or actual_branch == exp_branch)
    branch_confirmed_unpromoted = False
    if (
        not branch_ok
        and bool(actual_branch)
        and bool(exp_branch)
        and wt_path is not None
        and wt_path.is_dir()
    ):
        try:
            ref_chk = _subp_prom.run(
                ["git", "-C", str(wt_path), "rev-parse", "--verify", f"refs/heads/{exp_branch}"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            # Only confirm unpromoted when ref-read succeeds AND proves exact base_sha.
            if ref_chk.returncode == 0:
                exp_br_sha = ref_chk.stdout.strip()
                if bool(base_sha) and exp_br_sha == base_sha and exp_br_sha != cand_sha:
                    branch_confirmed_unpromoted = True
        except Exception:
            branch_confirmed_unpromoted = False

    return {
        "actual_head": actual_head,
        "actual_branch": actual_branch,
        "branch_ok": branch_ok,
        "branch_confirmed_unpromoted": branch_confirmed_unpromoted,
    }


class StateDB:
    """SQLite WAL ledger backed by <CONJECTRA_HOME>/state.db and <CONJECTRA_HOME>/events.jsonl."""

    def __init__(self, arr_home: str | os.PathLike[str] | None = None) -> None:
        self.arr_home = resolve_arr_home(arr_home)
        self.arr_home.mkdir(parents=True, exist_ok=True)
        self.db_path = self.arr_home / "state.db"
        self.events_path = self.arr_home / "events.jsonl"
        self._lock = threading.RLock()
        self._tx_depth = 0
        self._tx_pending_events: list[dict[str, Any]] = []
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure_connection()
        self._init_schema()

    def _configure_connection(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA busy_timeout=30000")
            for attempt in range(5):
                try:
                    mode_row = self._conn.execute("PRAGMA journal_mode").fetchone()
                    if mode_row is None or str(mode_row[0]).lower() != "wal":
                        self._conn.execute("PRAGMA journal_mode=WAL")
                    break
                except sqlite3.OperationalError as exc:
                    if "locked" not in str(exc).lower() or attempt == 4:
                        raise
                    time.sleep(0.05 * (attempt + 1))
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> StateDB:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()

    @contextlib.contextmanager
    def _tx(self) -> Iterator[tuple[sqlite3.Connection, list[dict[str, Any]]]]:
        """Execute an immediate SQLite transaction (reentrant per thread) and flush events after outer commit."""
        with self._lock:
            if self._tx_depth > 0:
                self._tx_depth += 1
                try:
                    yield self._conn, self._tx_pending_events
                finally:
                    self._tx_depth -= 1
                return

            self._tx_pending_events = []
            self._conn.execute("BEGIN IMMEDIATE")
            self._tx_depth = 1
            events_to_flush: list[dict[str, Any]] = []
            try:
                yield self._conn, self._tx_pending_events
                self._conn.execute("COMMIT")
                events_to_flush = list(self._tx_pending_events)
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise
            finally:
                self._tx_depth = 0
                self._tx_pending_events = []
            if events_to_flush:
                self._append_events_jsonl(events_to_flush)

    def _append_events_jsonl(self, events: Sequence[dict[str, Any]]) -> None:
        with self.events_path.open("a", encoding="utf-8") as fh:
            for ev in events:
                fh.write(json.dumps(ev, sort_keys=True, ensure_ascii=False) + "\n")
            fh.flush()

    def _record_event(
        self,
        conn: sqlite3.Connection,
        pending_events: list[dict[str, Any]],
        event_type: str,
        *,
        goal_id: str | None = None,
        hypothesis_id: str | None = None,
        task_id: str | None = None,
        payload: dict[str, Any] | None = None,
        ts: float | None = None,
    ) -> dict[str, Any]:
        now = time.time() if ts is None else float(ts)
        event_id = f"ev_{uuid.uuid4().hex}"
        body = payload or {}
        payload_json = json.dumps(body, sort_keys=True, ensure_ascii=False)
        cur = conn.execute(
            """
            INSERT INTO events (event_id, event_type, goal_id, hypothesis_id, task_id, ts, payload_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, event_type, goal_id, hypothesis_id, task_id, now, payload_json),
        )
        record = {
            "seq": cur.lastrowid,
            "event_id": event_id,
            "event_type": event_type,
            "goal_id": goal_id,
            "hypothesis_id": hypothesis_id,
            "task_id": task_id,
            "ts": now,
            "payload": body,
        }
        pending_events.append(record)
        return record

    def _init_schema(self) -> None:
        with self._lock:
            ver_row = self._conn.execute("PRAGMA user_version").fetchone()
            if ver_row is not None and int(ver_row[0]) >= 8:
                has_goals = self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'goals'"
                ).fetchone()
                if has_goals is not None:
                    return
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS goals (
                    goal_id TEXT PRIMARY KEY,
                    project TEXT NOT NULL,
                    title TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    primary_metric TEXT NOT NULL,
                    baseline_value REAL NOT NULL,
                    baseline_sha TEXT NOT NULL,
                    repo_path TEXT NOT NULL,
                    max_tokens_budget INTEGER NOT NULL,
                    max_concurrent_branches INTEGER NOT NULL,
                    adapter_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE',
                    payload_json TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS hypotheses (
                    hypothesis_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    parent_id TEXT,
                    kind TEXT NOT NULL,
                    fidelity TEXT NOT NULL,
                    priority_evoi REAL NOT NULL,
                    branch_name TEXT NOT NULL,
                    base_sha TEXT NOT NULL,
                    head_sha TEXT,
                    status TEXT NOT NULL,
                    scientific_verdict TEXT NOT NULL,
                    fingerprint TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL,
                    UNIQUE (goal_id, fingerprint)
                );
                CREATE INDEX IF NOT EXISTS idx_hypotheses_goal_status
                    ON hypotheses (goal_id, status);

                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    goal_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    status TEXT NOT NULL,
                    hq_job_id INTEGER,
                    attempt INTEGER NOT NULL,
                    max_attempts INTEGER NOT NULL,
                    worktree_path TEXT,
                    enqueued_ts REAL NOT NULL,
                    started_ts REAL,
                    finished_ts REAL,
                    payload_json TEXT NOT NULL,
                    updated_ts REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_tasks_goal_status
                    ON tasks (goal_id, status);
                CREATE INDEX IF NOT EXISTS idx_tasks_hyp_stage
                    ON tasks (hypothesis_id, stage);

                CREATE TABLE IF NOT EXISTS leases (
                    lease_id TEXT PRIMARY KEY,
                    resource_key TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    goal_id TEXT NOT NULL,
                    holder TEXT NOT NULL,
                    capacity INTEGER NOT NULL DEFAULT 1,
                    acquired_ts REAL NOT NULL,
                    renewed_ts REAL NOT NULL,
                    expires_ts REAL NOT NULL,
                    released_ts REAL,
                    status TEXT NOT NULL DEFAULT 'ACTIVE'
                );
                CREATE INDEX IF NOT EXISTS idx_leases_res_status_exp
                    ON leases (resource_key, status, expires_ts);
                CREATE INDEX IF NOT EXISTS idx_leases_task_status
                    ON leases (task_id, status);

                CREATE TABLE IF NOT EXISTS evidence_receipts (
                    receipt_id TEXT PRIMARY KEY,
                    task_id TEXT NOT NULL,
                    goal_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    process_status TEXT NOT NULL,
                    engineering_verdict TEXT NOT NULL,
                    scientific_verdict TEXT NOT NULL,
                    audit_verdict TEXT NOT NULL,
                    in_tokens INTEGER NOT NULL,
                    out_tokens INTEGER NOT NULL,
                    wall_sec REAL NOT NULL,
                    max_rss_mb REAL NOT NULL,
                    tests_passed INTEGER NOT NULL,
                    tests_total INTEGER NOT NULL,
                    created_ts REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_receipts_goal_ts
                    ON evidence_receipts (goal_id, created_ts);
                CREATE INDEX IF NOT EXISTS idx_receipts_hyp_stage
                    ON evidence_receipts (hypothesis_id, stage);
                CREATE INDEX IF NOT EXISTS idx_receipts_task
                    ON evidence_receipts (task_id);

                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT NOT NULL UNIQUE,
                    event_type TEXT NOT NULL,
                    goal_id TEXT,
                    hypothesis_id TEXT,
                    task_id TEXT,
                    ts REAL NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_events_goal_seq
                    ON events (goal_id, seq);

                CREATE TABLE IF NOT EXISTS graveyard (
                    fingerprint TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    scientific_verdict TEXT NOT NULL,
                    details_json TEXT NOT NULL,
                    created_ts REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_graveyard_goal
                    ON graveyard (goal_id);

                CREATE TABLE IF NOT EXISTS global_config (
                    config_key TEXT PRIMARY KEY,
                    value_json TEXT NOT NULL,
                    updated_ts REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS goal_versions (
                    version_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    human_epoch INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    change_summary TEXT NOT NULL DEFAULT '',
                    actor TEXT NOT NULL DEFAULT 'user',
                    reason TEXT NOT NULL DEFAULT '',
                    created_ts REAL NOT NULL,
                    UNIQUE (goal_id, version)
                );
                CREATE INDEX IF NOT EXISTS idx_goal_versions_goal_ver
                    ON goal_versions (goal_id, version);

                CREATE TABLE IF NOT EXISTS goal_knowledge (
                    item_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    hypothesis_id TEXT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    content_json TEXT NOT NULL,
                    goal_version INTEGER NOT NULL DEFAULT 1,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL,
                    UNIQUE (goal_id, category, key)
                );
                CREATE INDEX IF NOT EXISTS idx_goal_knowledge_goal_cat
                    ON goal_knowledge (goal_id, category);

                CREATE TABLE IF NOT EXISTS workflows (
                    workflow_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    hypothesis_id TEXT NOT NULL,
                    parent_workflow_id TEXT,
                    primitive TEXT NOT NULL,
                    status TEXT NOT NULL,
                    goal_version INTEGER NOT NULL DEFAULT 1,
                    budget_tokens INTEGER NOT NULL DEFAULT 50000,
                    payload_json TEXT NOT NULL,
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_workflows_goal_status
                    ON workflows (goal_id, status);
                CREATE INDEX IF NOT EXISTS idx_workflows_hypothesis
                    ON workflows (hypothesis_id);

                CREATE TABLE IF NOT EXISTS interventions (
                    intervention_id TEXT PRIMARY KEY,
                    scope TEXT NOT NULL,
                    action TEXT NOT NULL,
                    goal_id TEXT,
                    target_id TEXT,
                    actor TEXT NOT NULL DEFAULT 'user',
                    reason TEXT NOT NULL DEFAULT '',
                    before_epoch INTEGER,
                    after_epoch INTEGER,
                    payload_json TEXT NOT NULL,
                    result_json TEXT NOT NULL,
                    created_ts REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_interventions_goal_ts
                    ON interventions (goal_id, created_ts);

                CREATE TABLE IF NOT EXISTS decision_requests (
                    req_id TEXT PRIMARY KEY,
                    goal_id TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    trigger_kind TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'PENDING',
                    goal_version INTEGER NOT NULL DEFAULT 1,
                    human_epoch INTEGER NOT NULL DEFAULT 0,
                    coalesce_key TEXT NOT NULL DEFAULT '',
                    available_ts REAL NOT NULL,
                    payload_json TEXT NOT NULL,
                    resolution_json TEXT NOT NULL DEFAULT '{}',
                    hq_job_id INTEGER,
                    created_ts REAL NOT NULL,
                    updated_ts REAL NOT NULL,
                    resolved_ts REAL
                );
                CREATE INDEX IF NOT EXISTS idx_decision_requests_goal_status
                    ON decision_requests (goal_id, status);
                """
            )
            self._migrate_v3_schema()
            self._migrate_v4_schema()
            self._migrate_v5_schema()
            self._migrate_v6_schema()
            self._migrate_v7_schema()
            self._migrate_v8_schema()
        self._respawn_migrated_s3_workflows()

    def _migrate_v3_schema(self) -> None:
        """Cross-process atomic `BEGIN IMMEDIATE` WAL schema upgrade gated on `user_version < 3`."""
        with self._lock:
            fast_ver = self._conn.execute("PRAGMA user_version").fetchone()
            if fast_ver is not None and int(fast_ver[0]) >= 3:
                return
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                ver_row = self._conn.execute("PRAGMA user_version").fetchone()
                user_version = int(ver_row[0]) if ver_row is not None else 0
                if user_version >= 3:
                    self._conn.execute("COMMIT")
                    return

                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS research_plans (
                        plan_id TEXT PRIMARY KEY,
                        goal_id TEXT NOT NULL,
                        plan_version INTEGER NOT NULL,
                        goal_version INTEGER NOT NULL,
                        human_epoch INTEGER NOT NULL DEFAULT 0,
                        status TEXT NOT NULL DEFAULT 'DRAFT',
                        payload_json TEXT NOT NULL,
                        created_ts REAL NOT NULL,
                        updated_ts REAL NOT NULL,
                        UNIQUE (goal_id, plan_version)
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_research_plans_goal
                        ON research_plans (goal_id, plan_version DESC)
                    """
                )

                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS experiment_plans (
                        exp_plan_id TEXT PRIMARY KEY,
                        goal_id TEXT NOT NULL,
                        hypothesis_id TEXT NOT NULL,
                        plan_version INTEGER NOT NULL,
                        goal_version INTEGER NOT NULL,
                        human_epoch INTEGER NOT NULL DEFAULT 0,
                        review_status TEXT NOT NULL DEFAULT 'PENDING',
                        payload_json TEXT NOT NULL,
                        created_ts REAL NOT NULL,
                        updated_ts REAL NOT NULL,
                        UNIQUE (hypothesis_id, plan_version)
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_experiment_plans_hyp
                        ON experiment_plans (hypothesis_id, plan_version DESC)
                    """
                )

                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS execution_plans (
                        exec_plan_id TEXT PRIMARY KEY,
                        workflow_id TEXT NOT NULL UNIQUE,
                        goal_id TEXT NOT NULL,
                        hypothesis_id TEXT NOT NULL,
                        goal_version INTEGER NOT NULL,
                        human_epoch INTEGER NOT NULL DEFAULT 0,
                        primitive TEXT NOT NULL,
                        review_tier TEXT NOT NULL,
                        payload_json TEXT NOT NULL,
                        created_ts REAL NOT NULL,
                        updated_ts REAL NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_execution_plans_hyp
                        ON execution_plans (hypothesis_id, created_ts DESC)
                    """
                )

                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS review_records (
                        review_id TEXT PRIMARY KEY,
                        goal_id TEXT NOT NULL,
                        hypothesis_id TEXT NOT NULL,
                        workflow_id TEXT NOT NULL DEFAULT '',
                        task_id TEXT NOT NULL DEFAULT '',
                        review_kind TEXT NOT NULL,
                        review_tier TEXT NOT NULL,
                        target_version INTEGER NOT NULL DEFAULT 1,
                        head_sha TEXT NOT NULL DEFAULT '',
                        slot_index INTEGER NOT NULL DEFAULT 0,
                        verdict TEXT NOT NULL,
                        process_status TEXT NOT NULL DEFAULT 'EXIT_OK',
                        severity TEXT NOT NULL DEFAULT 'NONE',
                        tiebreaker_used INTEGER NOT NULL DEFAULT 0,
                        in_tokens INTEGER NOT NULL DEFAULT 0,
                        out_tokens INTEGER NOT NULL DEFAULT 0,
                        payload_json TEXT NOT NULL,
                        created_ts REAL NOT NULL,
                        UNIQUE (workflow_id, review_kind, target_version, head_sha, slot_index)
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_review_records_quorum
                        ON review_records (hypothesis_id, review_kind, target_version, head_sha)
                    """
                )
                self._conn.execute(
                    """
                    CREATE VIEW IF NOT EXISTS review_receipts AS
                        SELECT * FROM review_records
                    """
                )

                alter_specs: tuple[tuple[str, str, str], ...] = (
                    ("hypotheses", "failure_domain", "TEXT NOT NULL DEFAULT 'NONE'"),
                    ("tasks", "model_spec", "TEXT NOT NULL DEFAULT ''"),
                    ("tasks", "hq_llm_resource", "TEXT NOT NULL DEFAULT ''"),
                    ("evidence_receipts", "failure_domain", "TEXT NOT NULL DEFAULT 'NONE'"),
                    ("evidence_receipts", "effort_used", "TEXT NOT NULL DEFAULT ''"),
                    ("decision_requests", "role", "TEXT NOT NULL DEFAULT ''"),
                    ("decision_requests", "model_spec", "TEXT NOT NULL DEFAULT ''"),
                    ("decision_requests", "attempt", "INTEGER NOT NULL DEFAULT 1"),
                )
                cols_cache: dict[str, set[str]] = {}
                for table_name, col_name, col_ddl in alter_specs:
                    if table_name not in cols_cache:
                        info_rows = self._conn.execute(
                            f"PRAGMA table_info({table_name})"
                        ).fetchall()
                        cols_cache[table_name] = {str(r["name"]) for r in info_rows}
                    if col_name not in cols_cache[table_name]:
                        self._conn.execute(
                            f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_ddl}"
                        )
                        cols_cache[table_name].add(col_name)

                self._conn.execute("PRAGMA user_version = 3")
                self._conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise

    def _migrate_v4_schema(self) -> None:
        """Cross-process atomic `BEGIN IMMEDIATE` WAL schema upgrade to `user_version = 4`.

        Adds `plan_epoch` columns, epoch-bound `review_records` uniqueness, unified idempotent
        `token_usage_ledger` (backfilling historical receipts + decision requests), and durable
        `pending_hq_cancellations`.
        """
        with self._lock:
            fast_ver = self._conn.execute("PRAGMA user_version").fetchone()
            if fast_ver is not None and int(fast_ver[0]) >= 4:
                return
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                ver_row = self._conn.execute("PRAGMA user_version").fetchone()
                user_version = int(ver_row[0]) if ver_row is not None else 0
                if user_version >= 4:
                    self._conn.execute("COMMIT")
                    return

                alter_specs_v4: tuple[tuple[str, str, str], ...] = (
                    ("goals", "plan_epoch", "INTEGER NOT NULL DEFAULT 0"),
                    ("research_plans", "plan_epoch", "INTEGER NOT NULL DEFAULT 0"),
                    ("experiment_plans", "plan_epoch", "INTEGER NOT NULL DEFAULT 0"),
                    ("execution_plans", "plan_epoch", "INTEGER NOT NULL DEFAULT 0"),
                    ("workflows", "plan_epoch", "INTEGER NOT NULL DEFAULT 0"),
                    ("tasks", "plan_epoch", "INTEGER NOT NULL DEFAULT 0"),
                    ("evidence_receipts", "plan_epoch", "INTEGER NOT NULL DEFAULT 0"),
                    ("decision_requests", "plan_epoch", "INTEGER NOT NULL DEFAULT 0"),
                )
                cols_cache: dict[str, set[str]] = {}
                for table_name, col_name, col_ddl in alter_specs_v4:
                    if table_name not in cols_cache:
                        info_rows = self._conn.execute(
                            f"PRAGMA table_info({table_name})"
                        ).fetchall()
                        cols_cache[table_name] = {str(r["name"]) for r in info_rows}
                    if cols_cache[table_name] and col_name not in cols_cache[table_name]:
                        self._conn.execute(
                            f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_ddl}"
                        )
                        cols_cache[table_name].add(col_name)

                # Rebuild `review_records` so UNIQUE binds `(workflow_id, review_kind, plan_epoch, human_epoch, target_version, head_sha, slot_index)`
                rev_cols = {
                    str(r["name"])
                    for r in self._conn.execute("PRAGMA table_info(review_records)").fetchall()
                }
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS review_records_v4 (
                        review_id TEXT PRIMARY KEY,
                        goal_id TEXT NOT NULL,
                        hypothesis_id TEXT NOT NULL,
                        workflow_id TEXT NOT NULL DEFAULT '',
                        task_id TEXT NOT NULL DEFAULT '',
                        review_kind TEXT NOT NULL,
                        review_tier TEXT NOT NULL,
                        target_version INTEGER NOT NULL DEFAULT 1,
                        plan_epoch INTEGER NOT NULL DEFAULT 0,
                        human_epoch INTEGER NOT NULL DEFAULT 0,
                        target_kind TEXT NOT NULL DEFAULT 'hypothesis',
                        head_sha TEXT NOT NULL DEFAULT '',
                        slot_index INTEGER NOT NULL DEFAULT 0,
                        verdict TEXT NOT NULL,
                        process_status TEXT NOT NULL DEFAULT 'EXIT_OK',
                        severity TEXT NOT NULL DEFAULT 'NONE',
                        tiebreaker_used INTEGER NOT NULL DEFAULT 0,
                        in_tokens INTEGER NOT NULL DEFAULT 0,
                        out_tokens INTEGER NOT NULL DEFAULT 0,
                        payload_json TEXT NOT NULL,
                        created_ts REAL NOT NULL,
                        UNIQUE (workflow_id, review_kind, plan_epoch, human_epoch, target_version, head_sha, slot_index)
                    )
                    """
                )
                if rev_cols:
                    pe_expr = "plan_epoch" if "plan_epoch" in rev_cols else "0"
                    he_expr = "human_epoch" if "human_epoch" in rev_cols else "0"
                    tk_expr = "target_kind" if "target_kind" in rev_cols else "'hypothesis'"
                    self._conn.execute(
                        f"""
                        INSERT OR IGNORE INTO review_records_v4 (
                            review_id, goal_id, hypothesis_id, workflow_id, task_id,
                            review_kind, review_tier, target_version, plan_epoch, human_epoch, target_kind,
                            head_sha, slot_index, verdict, process_status, severity, tiebreaker_used,
                            in_tokens, out_tokens, payload_json, created_ts
                        )
                        SELECT
                            review_id, goal_id, hypothesis_id, workflow_id, task_id,
                            review_kind, review_tier, target_version, {pe_expr}, {he_expr}, {tk_expr},
                            head_sha, slot_index, verdict, process_status, severity, tiebreaker_used,
                            in_tokens, out_tokens, payload_json, created_ts
                        FROM review_records
                        """
                    )
                    self._conn.execute("DROP VIEW IF EXISTS review_receipts")
                    self._conn.execute("DROP TABLE review_records")
                self._conn.execute("ALTER TABLE review_records_v4 RENAME TO review_records")
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_review_records_quorum
                        ON review_records (hypothesis_id, review_kind, plan_epoch, target_version, head_sha)
                    """
                )
                self._conn.execute(
                    """
                    CREATE VIEW IF NOT EXISTS review_receipts AS
                        SELECT * FROM review_records
                    """
                )

                # Unified idempotent token budget & spend ledger
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS token_usage_ledger (
                        usage_key TEXT PRIMARY KEY,
                        goal_id TEXT NOT NULL,
                        entity_kind TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 1,
                        model TEXT NOT NULL DEFAULT '',
                        reserved_tokens INTEGER NOT NULL DEFAULT 0,
                        input_tokens INTEGER NOT NULL DEFAULT 0,
                        output_tokens INTEGER NOT NULL DEFAULT 0,
                        cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                        state TEXT NOT NULL DEFAULT 'SETTLED',
                        created_ts REAL NOT NULL,
                        updated_ts REAL NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_token_usage_goal
                        ON token_usage_ledger (goal_id, state)
                    """
                )

                # Backfill historical `evidence_receipts` idempotently
                rcpt_rows = self._conn.execute(
                    """
                    SELECT r.receipt_id, r.task_id, r.goal_id, r.in_tokens, r.out_tokens,
                           r.created_ts, r.payload_json, COALESCE(t.attempt, 1) AS task_attempt
                    FROM evidence_receipts r
                    LEFT JOIN tasks t ON t.task_id = r.task_id
                    """
                ).fetchall()
                for rr in rcpt_rows:
                    pdoc = {}
                    with contextlib.suppress(Exception):
                        pdoc = json.loads(rr["payload_json"])
                    tid = str(rr["task_id"] or "")
                    att = int(rr["task_attempt"] or 1)
                    ukey = f"task:{tid}:a{att}" if tid else f"rcpt:{rr['receipt_id']}"
                    mod_str = str(pdoc.get("model_used") or "")
                    cache_raw = (pdoc.get("metrics") or {}).get("cache_read_tokens", 0)
                    cache_tok = int(cache_raw.get("value", 0)) if isinstance(cache_raw, Mapping) else int(cache_raw or 0)
                    cts = float(rr["created_ts"] or time.time())
                    self._conn.execute(
                        """
                        INSERT INTO token_usage_ledger (
                            usage_key, goal_id, entity_kind, entity_id, attempt, model,
                            reserved_tokens, input_tokens, output_tokens, cached_input_tokens,
                            state, created_ts, updated_ts
                        ) VALUES (?, ?, 'stage_task', ?, ?, ?, 0, ?, ?, ?, 'SETTLED', ?, ?)
                        ON CONFLICT(usage_key) DO UPDATE SET
                            input_tokens = MAX(token_usage_ledger.input_tokens, excluded.input_tokens),
                            output_tokens = MAX(token_usage_ledger.output_tokens, excluded.output_tokens),
                            cached_input_tokens = MAX(token_usage_ledger.cached_input_tokens, excluded.cached_input_tokens)
                        """,
                        (
                            ukey,
                            str(rr["goal_id"]),
                            tid or str(rr["receipt_id"]),
                            att,
                            mod_str,
                            max(0, int(rr["in_tokens"] or 0)),
                            max(0, int(rr["out_tokens"] or 0)),
                            max(0, cache_tok),
                            cts,
                            cts,
                        ),
                    )

                # Backfill historical `decision_requests` idempotently
                dreq_rows = self._conn.execute(
                    """
                    SELECT req_id, goal_id, status, model_spec, attempt,
                           payload_json, resolution_json, created_ts, updated_ts
                    FROM decision_requests
                    """
                ).fetchall()
                for dr in dreq_rows:
                    rdoc = {}
                    with contextlib.suppress(Exception):
                        rdoc = json.loads(dr["resolution_json"] or "{}")
                    in_t = max(0, int(rdoc.get("in_tokens", 0) or 0))
                    out_t = max(0, int(rdoc.get("out_tokens", 0) or 0))
                    cache_t = max(0, int(rdoc.get("cached_input_tokens", 0) or 0))
                    st = str(dr["status"] or "")
                    if (in_t + out_t) <= 0 and st in ("PENDING", "RUNNING"):
                        continue
                    att_d = max(1, int(dr["attempt"] or 1))
                    ukey_d = f"decision:{dr['req_id']}:a{att_d}"
                    mod_d = str(rdoc.get("model_used") or dr["model_spec"] or "")
                    cts_d = float(dr["created_ts"] or time.time())
                    uts_d = float(dr["updated_ts"] or cts_d)
                    self._conn.execute(
                        """
                        INSERT OR IGNORE INTO token_usage_ledger (
                            usage_key, goal_id, entity_kind, entity_id, attempt, model,
                            reserved_tokens, input_tokens, output_tokens, cached_input_tokens,
                            state, created_ts, updated_ts
                        ) VALUES (?, ?, 'decision_request', ?, ?, ?, 0, ?, ?, ?, 'SETTLED', ?, ?)
                        """,
                        (
                            ukey_d,
                            str(dr["goal_id"]),
                            str(dr["req_id"]),
                            att_d,
                            mod_d,
                            in_t,
                            out_t,
                            cache_t,
                            cts_d,
                            uts_d,
                        ),
                    )

                # Durable HQ cancellation ledger
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pending_hq_cancellations (
                        cancel_id TEXT PRIMARY KEY,
                        hq_job_id INTEGER NOT NULL,
                        entity_kind TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        goal_id TEXT NOT NULL,
                        hypothesis_id TEXT NOT NULL DEFAULT '',
                        workflow_id TEXT NOT NULL DEFAULT '',
                        epoch INTEGER NOT NULL DEFAULT 0,
                        plan_epoch INTEGER NOT NULL DEFAULT 0,
                        state TEXT NOT NULL DEFAULT 'CANCEL_PENDING',
                        reason TEXT NOT NULL DEFAULT '',
                        created_ts REAL NOT NULL,
                        updated_ts REAL NOT NULL,
                        confirmed_ts REAL,
                        UNIQUE (hq_job_id, entity_kind, entity_id)
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_pending_hq_cancel_goal_state
                        ON pending_hq_cancellations (goal_id, state)
                    """
                )

                self._conn.execute("PRAGMA user_version = 4")
                self._conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise

    def _migrate_v5_schema(self) -> None:
        """Cross-process atomic `BEGIN IMMEDIATE` WAL schema upgrade to `user_version = 5`.

        Adds durable two-phase `promotion_intents` ledger for crash-recoverable strict fast-forward Git promotion.
        """
        with self._lock:
            fast_ver = self._conn.execute("PRAGMA user_version").fetchone()
            if fast_ver is not None and int(fast_ver[0]) >= 5:
                return
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                ver_row = self._conn.execute("PRAGMA user_version").fetchone()
                user_version = int(ver_row[0]) if ver_row is not None else 0
                if user_version >= 5:
                    self._conn.execute("COMMIT")
                    return

                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS promotion_intents (
                        intent_id TEXT PRIMARY KEY,
                        goal_id TEXT NOT NULL,
                        hypothesis_id TEXT NOT NULL,
                        task_id TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 1,
                        branch_name TEXT NOT NULL DEFAULT '',
                        wt_path TEXT NOT NULL DEFAULT '',
                        expected_base_sha TEXT NOT NULL DEFAULT '',
                        candidate_sha TEXT NOT NULL DEFAULT '',
                        goal_version INTEGER NOT NULL DEFAULT 1,
                        human_epoch INTEGER NOT NULL DEFAULT 0,
                        plan_epoch INTEGER NOT NULL DEFAULT 0,
                        state TEXT NOT NULL DEFAULT 'PREPARED',
                        reason TEXT NOT NULL DEFAULT '',
                        created_ts REAL NOT NULL,
                        updated_ts REAL NOT NULL
                    )
                    """
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_promotion_intents_state
                        ON promotion_intents (state, goal_id)
                    """
                )
                self._conn.execute("PRAGMA user_version = 5")
                self._conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise

    def _migrate_v6_schema(self) -> None:
        """Cross-process atomic `BEGIN IMMEDIATE` WAL schema upgrade to `user_version = 6`.

        1. Adds permanent `revoked_barrier_leases` table (no truncation) and backfills existing revoked IDs.
        2. Adds `attempt` column to `pending_hq_cancellations` for attempt-scoped reservation release.
        3. Fails closed on any active legacy S3 synthesis workflows lacking `source_lineage_snapshot`.
        """
        with self._lock:
            fast_ver = self._conn.execute("PRAGMA user_version").fetchone()
            if fast_ver is not None and int(fast_ver[0]) >= 6:
                return
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                ver_row = self._conn.execute("PRAGMA user_version").fetchone()
                user_version = int(ver_row[0]) if ver_row is not None else 0
                if user_version >= 6:
                    self._conn.execute("COMMIT")
                    return

                now_m = time.time()
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS revoked_barrier_leases (
                        lease_id TEXT PRIMARY KEY,
                        barrier_id TEXT NOT NULL DEFAULT '',
                        generation INTEGER NOT NULL DEFAULT 0,
                        reason TEXT NOT NULL DEFAULT '',
                        revoked_ts REAL NOT NULL
                    )
                    """
                )
                fence_row = self._conn.execute(
                    "SELECT value_json FROM global_config WHERE config_key = 'maintenance_barrier_fences'"
                ).fetchone()
                if fence_row is not None:
                    try:
                        fence_doc = json.loads(fence_row["value_json"])
                        if isinstance(fence_doc, Mapping):
                            gen_m = int(fence_doc.get("barrier_generation", 0) or 0)
                            for rl_id in fence_doc.get("revoked_lease_ids") or []:
                                clean_rl = str(rl_id or "").strip()
                                if clean_rl:
                                    self._conn.execute(
                                        """
                                        INSERT OR IGNORE INTO revoked_barrier_leases (
                                            lease_id, barrier_id, generation, reason, revoked_ts
                                        ) VALUES (?, '', ?, 'migrated_v6_backfill', ?)
                                        """,
                                        (clean_rl, gen_m, now_m),
                                    )
                    except (TypeError, ValueError, json.JSONDecodeError):
                        pass

                phq_cols = {
                    str(r[1])
                    for r in self._conn.execute("PRAGMA table_info(pending_hq_cancellations)").fetchall()
                }
                if phq_cols and "attempt" not in phq_cols:
                    self._conn.execute(
                        "ALTER TABLE pending_hq_cancellations ADD COLUMN attempt INTEGER NOT NULL DEFAULT 1"
                    )

                wf_rows = self._conn.execute(
                    "SELECT workflow_id, payload_json FROM workflows WHERE status IN ('READY', 'RUNNING', 'WAITING', 'PAUSED')"
                ).fetchall()
                for wr in wf_rows:
                    try:
                        wf_doc = json.loads(wr["payload_json"])
                    except (TypeError, json.JSONDecodeError):
                        continue
                    cfg = wf_doc.get("config") if isinstance(wf_doc.get("config"), Mapping) else {}
                    syn_hids = cfg.get("synthesized_hypothesis_ids") or []
                    is_syn = str(cfg.get("target_kind") or "") == "goal_synthesis" or bool(syn_hids)
                    snap = cfg.get("source_lineage_snapshot")
                    if is_syn and not (isinstance(snap, Mapping) and bool(snap)):
                        wf_id_m = str(wr["workflow_id"])
                        st_doc = dict(wf_doc.get("state") or {}) if isinstance(wf_doc.get("state"), Mapping) else {}
                        st_doc["step"] = "synthesis_superseded"
                        st_doc["stale_sources"] = ["migrated_v6_missing_source_lineage_snapshot"]
                        wf_doc["status"] = "SYNTHESIS_SUPERSEDED"
                        wf_doc["state"] = st_doc
                        wf_doc["updated_ts"] = now_m
                        self._conn.execute(
                            "UPDATE workflows SET status = 'SYNTHESIS_SUPERSEDED', payload_json = ?, updated_ts = ? WHERE workflow_id = ?",
                            (json.dumps(wf_doc, sort_keys=True, ensure_ascii=False), now_m, wf_id_m),
                        )
                        t_rows = self._conn.execute(
                            "SELECT task_id, attempt, payload_json FROM tasks WHERE status = 'READY'"
                        ).fetchall()
                        for tr in t_rows:
                            tid_m = str(tr["task_id"])
                            tatt_m = int(tr["attempt"] or 1)
                            try:
                                t_doc = json.loads(tr["payload_json"])
                            except (TypeError, json.JSONDecodeError):
                                continue
                            if str(t_doc.get("workflow_id") or "") != wf_id_m:
                                continue
                            t_doc["status"] = "CANCELLED"
                            t_doc["finished_ts"] = now_m
                            self._conn.execute(
                                "UPDATE tasks SET status = 'CANCELLED', finished_ts = ?, payload_json = ?, updated_ts = ? WHERE task_id = ?",
                                (now_m, json.dumps(t_doc, sort_keys=True, ensure_ascii=False), now_m, tid_m),
                            )
                            self._conn.execute(
                                "UPDATE token_usage_ledger SET reserved_tokens = 0, state = 'RELEASED', updated_ts = ? WHERE usage_key = ? AND state = 'RESERVED'",
                                (now_m, f"task:{tid_m}:a{tatt_m}"),
                            )

                self._conn.execute("PRAGMA user_version = 6")
                self._conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise

    def _migrate_v7_schema(self) -> None:
        """Cross-process atomic `BEGIN IMMEDIATE` WAL schema upgrade to `user_version = 7`.

        1. Rebuilds `pending_hq_cancellations` with a 4-column conflict key
           `UNIQUE (hq_job_id, entity_kind, entity_id, attempt)` so attempt-specific rows
           never overwrite `attempt` or clobber another attempt's `CANCEL_CONFIRMED`.
        2. Marks any remaining active snapshot-less S3 workflows `SYNTHESIS_SUPERSEDED` for
           automatic snapshot-bound replacement on `ACTIVE` goals.
        """
        with self._lock:
            fast_ver = self._conn.execute("PRAGMA user_version").fetchone()
            if fast_ver is not None and int(fast_ver[0]) >= 7:
                return
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                ver_row = self._conn.execute("PRAGMA user_version").fetchone()
                user_version = int(ver_row[0]) if ver_row is not None else 0
                if user_version >= 7:
                    self._conn.execute("COMMIT")
                    return

                now_m = time.time()
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS pending_hq_cancellations_v7 (
                        cancel_id TEXT PRIMARY KEY,
                        hq_job_id INTEGER NOT NULL,
                        entity_kind TEXT NOT NULL,
                        entity_id TEXT NOT NULL,
                        attempt INTEGER NOT NULL DEFAULT 1,
                        goal_id TEXT NOT NULL,
                        hypothesis_id TEXT NOT NULL DEFAULT '',
                        workflow_id TEXT NOT NULL DEFAULT '',
                        epoch INTEGER NOT NULL DEFAULT 0,
                        plan_epoch INTEGER NOT NULL DEFAULT 0,
                        state TEXT NOT NULL DEFAULT 'CANCEL_PENDING',
                        reason TEXT NOT NULL DEFAULT '',
                        created_ts REAL NOT NULL,
                        updated_ts REAL NOT NULL,
                        confirmed_ts REAL,
                        UNIQUE (hq_job_id, entity_kind, entity_id, attempt)
                    )
                    """
                )
                has_old_phq = (
                    self._conn.execute(
                        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'pending_hq_cancellations'"
                    ).fetchone()
                    is not None
                )
                if has_old_phq:
                    old_cols = {
                        str(r[1])
                        for r in self._conn.execute("PRAGMA table_info(pending_hq_cancellations)").fetchall()
                    }
                    att_expr = "MAX(1, COALESCE(attempt, 1))" if "attempt" in old_cols else "1"
                    old_rows = self._conn.execute(
                        f"""
                        SELECT cancel_id, hq_job_id, entity_kind, entity_id, {att_expr} AS eff_att,
                               goal_id, hypothesis_id, workflow_id, epoch, plan_epoch,
                               state, reason, created_ts, updated_ts, confirmed_ts
                        FROM pending_hq_cancellations
                        """
                    ).fetchall()
                    for r in old_rows:
                        eff_att = int(r["eff_att"] or 1)
                        cid_v7 = f"hqc_{int(r['hq_job_id'])}_{r['entity_kind']}_{r['entity_id']}_a{eff_att}"
                        self._conn.execute(
                            """
                            INSERT OR REPLACE INTO pending_hq_cancellations_v7 (
                                cancel_id, hq_job_id, entity_kind, entity_id, attempt, goal_id,
                                hypothesis_id, workflow_id, epoch, plan_epoch,
                                state, reason, created_ts, updated_ts, confirmed_ts
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                cid_v7,
                                int(r["hq_job_id"]),
                                str(r["entity_kind"]),
                                str(r["entity_id"]),
                                eff_att,
                                str(r["goal_id"]),
                                str(r["hypothesis_id"] or ""),
                                str(r["workflow_id"] or ""),
                                int(r["epoch"] or 0),
                                int(r["plan_epoch"] or 0),
                                str(r["state"] or "CANCEL_PENDING"),
                                str(r["reason"] or ""),
                                float(r["created_ts"] or now_m),
                                float(r["updated_ts"] or now_m),
                                r["confirmed_ts"],
                            ),
                        )
                    self._conn.execute("DROP TABLE pending_hq_cancellations")
                self._conn.execute(
                    "ALTER TABLE pending_hq_cancellations_v7 RENAME TO pending_hq_cancellations"
                )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_pending_hq_cancel_state
                        ON pending_hq_cancellations (state, goal_id)
                    """
                )

                self._conn.execute("PRAGMA user_version = 7")
                self._conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise

    def _migrate_v8_schema(self) -> None:
        """Cross-process atomic `BEGIN IMMEDIATE` WAL schema upgrade to `user_version = 8`.

        Adds immutable `usage_key TEXT NOT NULL DEFAULT ''` to `pending_hq_cancellations`
        so nonce-scoped decision cancellations release ONLY their exact reservation row
        without touching live sibling attempts/nonces.
        """
        with self._lock:
            fast_ver = self._conn.execute("PRAGMA user_version").fetchone()
            if fast_ver is not None and int(fast_ver[0]) >= 8:
                return
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                ver_row = self._conn.execute("PRAGMA user_version").fetchone()
                user_version = int(ver_row[0]) if ver_row is not None else 0
                if user_version >= 8:
                    self._conn.execute("COMMIT")
                    return

                phq_cols = {
                    str(r[1])
                    for r in self._conn.execute("PRAGMA table_info(pending_hq_cancellations)").fetchall()
                }
                if phq_cols and "usage_key" not in phq_cols:
                    self._conn.execute(
                        "ALTER TABLE pending_hq_cancellations ADD COLUMN usage_key TEXT NOT NULL DEFAULT ''"
                    )
                # Backfill unambiguous legacy v7 empty-usage_key rows; quarantine ambiguous ones.
                legacy_rows = self._conn.execute(
                    "SELECT cancel_id, hq_job_id, entity_kind, entity_id, attempt FROM pending_hq_cancellations WHERE usage_key = ''"
                ).fetchall()
                for lrow in legacy_rows:
                    cid = str(lrow["cancel_id"])
                    ek = str(lrow["entity_kind"] or "")
                    ent_id = str(lrow["entity_id"] or "")
                    att = max(1, int(lrow["attempt"] if lrow["attempt"] is not None else 1))
                    hq_jid = lrow["hq_job_id"]
                    resolved_ukey = ""
                    if ek in ("decision", "decision_request"):
                        d_row = self._conn.execute(
                            "SELECT hq_job_id, attempt, payload_json FROM decision_requests WHERE req_id = ?",
                            (ent_id,),
                        ).fetchone()
                        if (
                            d_row is not None
                            and d_row["hq_job_id"] is not None
                            and hq_jid is not None
                            and int(d_row["hq_job_id"]) == int(hq_jid)
                            and max(1, int(d_row["attempt"] or 1)) == att
                        ):
                            try:
                                d_pl = json.loads(d_row["payload_json"] or "{}")
                            except (ValueError, TypeError):
                                d_pl = {}
                            d_nonce = str(d_pl.get("submit_nonce") or "").strip()
                            if d_nonce:
                                resolved_ukey = f"decision:{ent_id}:a{att}:{d_nonce}"
                        if not resolved_ukey:
                            prefix = f"decision:{ent_id}:a{att}"
                            glob_p = self._escape_glob(prefix) + ":*"
                            led_rows = self._conn.execute(
                                """
                                SELECT usage_key FROM token_usage_ledger
                                WHERE (usage_key = ? OR usage_key GLOB ?) AND state = 'RESERVED'
                                """,
                                (prefix, glob_p),
                            ).fetchall()
                            if len(led_rows) == 1:
                                resolved_ukey = str(led_rows[0]["usage_key"])
                            elif len(led_rows) > 1:
                                resolved_ukey = f"legacy_ambiguous:{cid}"
                    elif ek in ("task", "stage_task"):
                        prefix = f"task:{ent_id}:a{att}"
                        glob_p = self._escape_glob(prefix) + ":*"
                        led_rows = self._conn.execute(
                            """
                            SELECT usage_key FROM token_usage_ledger
                            WHERE (usage_key = ? OR usage_key GLOB ?) AND state = 'RESERVED'
                            """,
                            (prefix, glob_p),
                        ).fetchall()
                        if len(led_rows) == 1:
                            resolved_ukey = str(led_rows[0]["usage_key"])
                        elif len(led_rows) > 1:
                            resolved_ukey = f"legacy_ambiguous:{cid}"
                    if resolved_ukey:
                        self._conn.execute(
                            "UPDATE pending_hq_cancellations SET usage_key = ? WHERE cancel_id = ? AND usage_key = ''",
                            (resolved_ukey, cid),
                        )
                self._conn.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_pending_hq_cancel_usage_key
                        ON pending_hq_cancellations (usage_key, state)
                    """
                )
                self._conn.execute("PRAGMA user_version = 8")
                self._conn.execute("COMMIT")
            except BaseException:
                with contextlib.suppress(sqlite3.Error):
                    self._conn.execute("ROLLBACK")
                raise

    def _respawn_migrated_s3_workflows(self) -> None:
        """Best-effort enqueue of replacement snapshot-bound S3 workflows for migrated snapshot-less workflows."""
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM workflows WHERE status = 'SYNTHESIS_SUPERSEDED' LIMIT 1"
            ).fetchone()
        if row is None:
            return
        with contextlib.suppress(Exception):
            from conjectra.workflow_runtime import WorkflowRuntime

            WorkflowRuntime(self).respawn_superseded_legacy_synthesis_workflows()

    @staticmethod
    def _has_unconfirmed_hq_cancel_in_conn(
        conn: sqlite3.Connection,
        entity_kind: str,
        entity_id: str,
        attempt: int,
        *,
        usage_key: str | None = None,
    ) -> bool:
        if usage_key:
            row_p = conn.execute(
                """
                SELECT 1 FROM pending_hq_cancellations
                WHERE entity_kind IN (?, ?) AND entity_id = ? AND attempt = ?
                  AND (usage_key = ? OR usage_key = '' OR usage_key LIKE 'legacy_ambiguous:%')
                  AND state != 'CANCEL_CONFIRMED'
                LIMIT 1
                """,
                (entity_kind, f"{entity_kind}_request", str(entity_id), int(attempt), str(usage_key)),
            ).fetchone()
        else:
            row_p = conn.execute(
                """
                SELECT 1 FROM pending_hq_cancellations
                WHERE entity_kind IN (?, ?) AND entity_id = ? AND attempt = ? AND state != 'CANCEL_CONFIRMED'
                LIMIT 1
                """,
                (entity_kind, f"{entity_kind}_request", str(entity_id), int(attempt)),
            ).fetchone()
        return row_p is not None

    # ------------------------------------------------------------------ Goals

    def save_goal(self, goal: ResearchGoal, *, status: str | None = None) -> ResearchGoal:
        """Create or update a ResearchGoal in the ledger."""
        now = time.time()
        payload_json = json.dumps(goal.to_dict(), sort_keys=True, ensure_ascii=False)
        with self._tx() as (conn, pending):
            existing = conn.execute(
                "SELECT created_ts, status FROM goals WHERE goal_id = ?", (goal.goal_id,)
            ).fetchone()
            created_ts = float(existing["created_ts"]) if existing else now
            effective_status = (
                status
                if status is not None
                else (str(existing["status"]) if existing is not None else "ACTIVE")
            )
            conn.execute(
                """
                INSERT INTO goals (
                    goal_id, project, title, direction, primary_metric, baseline_value,
                    baseline_sha, repo_path, max_tokens_budget, max_concurrent_branches,
                    adapter_name, status, plan_epoch, payload_json, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(goal_id) DO UPDATE SET
                    project=excluded.project,
                    title=excluded.title,
                    direction=excluded.direction,
                    primary_metric=excluded.primary_metric,
                    baseline_value=excluded.baseline_value,
                    baseline_sha=excluded.baseline_sha,
                    repo_path=excluded.repo_path,
                    max_tokens_budget=excluded.max_tokens_budget,
                    max_concurrent_branches=excluded.max_concurrent_branches,
                    adapter_name=excluded.adapter_name,
                    status=excluded.status,
                    plan_epoch=excluded.plan_epoch,
                    payload_json=excluded.payload_json,
                    updated_ts=excluded.updated_ts
                """,
                (
                    goal.goal_id,
                    goal.project,
                    goal.title,
                    goal.direction,
                    goal.primary_metric,
                    float(goal.baseline_value),
                    goal.baseline_sha,
                    goal.repo_path,
                    int(goal.max_tokens_budget),
                    int(goal.max_concurrent_branches),
                    goal.adapter_name,
                    effective_status,
                    int(getattr(goal, "plan_epoch", 0) or 0),
                    payload_json,
                    created_ts,
                    now,
                ),
            )
            self._record_event(
                conn,
                pending,
                "GOAL_SAVED",
                goal_id=goal.goal_id,
                payload={
                    "status": effective_status,
                    "project": goal.project,
                    "version": int(goal.version),
                    "human_epoch": int(goal.human_epoch),
                },
                ts=now,
            )
        return goal

    def get_goal_status(self, goal_id: str) -> str | None:
        """Return lifecycle status string for a goal_id, or None if not found."""
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
        return None if row is None else str(row["status"])

    def list_goal_records(self, *, status: str | None = None) -> list[dict[str, Any]]:
        """List goal records with status, timestamps, and deserialized ResearchGoal."""
        query = "SELECT status, created_ts, updated_ts, payload_json FROM goals"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_ts ASC, goal_id ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [
            {
                "status": str(r["status"]),
                "created_ts": float(r["created_ts"]),
                "updated_ts": float(r["updated_ts"]),
                "goal": ResearchGoal.from_dict(json.loads(r["payload_json"])),
            }
            for r in rows
        ]

    def get_goal(self, goal_id: str) -> ResearchGoal | None:
        """Fetch a ResearchGoal by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
        if row is None:
            return None
        return ResearchGoal.from_dict(json.loads(row["payload_json"]))

    def list_goals(self, *, status: str | None = None) -> list[ResearchGoal]:
        """List registered ResearchGoals, optionally filtered by status."""
        query = "SELECT payload_json FROM goals"
        params: list[Any] = []
        if status is not None:
            query += " WHERE status = ?"
            params.append(status)
        query += " ORDER BY created_ts ASC, goal_id ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [ResearchGoal.from_dict(json.loads(r["payload_json"])) for r in rows]

    def update_goal_status(self, goal_id: str, status: str) -> None:
        """Update goal lifecycle status."""
        now = time.time()
        with self._tx() as (conn, pending):
            cur = conn.execute(
                "UPDATE goals SET status = ?, updated_ts = ? WHERE goal_id = ?",
                (status, now, goal_id),
            )
            if cur.rowcount == 0:
                raise KeyError(f"Unknown goal_id: {goal_id}")
            self._record_event(
                conn, pending, "GOAL_STATUS", goal_id=goal_id, payload={"status": status}, ts=now
            )

    # ------------------------------------------------------------- Hypotheses

    def check_duplicate_fingerprint(
        self, goal_id: str, fingerprint: str
    ) -> dict[str, Any] | None:
        """Check whether a normalized fingerprint exists in graveyard or hypotheses."""
        fp = fingerprint.strip().lower()
        with self._lock:
            gy = self._conn.execute(
                "SELECT * FROM graveyard WHERE fingerprint = ? AND goal_id = ?",
                (fp, goal_id),
            ).fetchone()
            if gy is not None:
                return {"source": "graveyard", **dict(gy)}
            hyp = self._conn.execute(
                "SELECT hypothesis_id, status, fingerprint FROM hypotheses WHERE goal_id = ? AND fingerprint = ?",
                (goal_id, fp),
            ).fetchone()
            if hyp is not None:
                return {"source": "hypotheses", **dict(hyp)}
        return None

    def register_hypothesis(
        self,
        node: HypothesisNode,
        *,
        allow_graveyard: bool = False,
    ) -> tuple[HypothesisNode, bool]:
        """Register a hypothesis with normalized fingerprint deduplication.

        Returns `(stored_node, created)`. If an identical fingerprint already exists
        for the same goal, returns `(existing_node, False)`. If the fingerprint was
        previously placed in the graveyard and `allow_graveyard` is False, raises
        `GraveyardDuplicateError`.
        """
        fp = fingerprint_for_node(node)
        now = time.time()
        normalized = HypothesisNode.from_dict({**node.to_dict(), "fingerprint": fp})
        payload_json = json.dumps(normalized.to_dict(), sort_keys=True, ensure_ascii=False)

        with self._tx() as (conn, pending):
            if normalized.parent_id:
                parent_row = conn.execute(
                    "SELECT goal_id FROM hypotheses WHERE hypothesis_id = ?",
                    (normalized.parent_id,),
                ).fetchone()
                if parent_row is not None and str(parent_row["goal_id"]) != normalized.goal_id:
                    raise ValueError(
                        f"Cross-goal ownership violation: parent_id {normalized.parent_id!r} belongs to "
                        f"goal {parent_row['goal_id']!r}, not {normalized.goal_id!r}"
                    )
            gy = conn.execute(
                "SELECT hypothesis_id, reason FROM graveyard WHERE fingerprint = ? AND goal_id = ?",
                (fp, normalized.goal_id),
            ).fetchone()
            if gy is not None and not allow_graveyard:
                raise GraveyardDuplicateError(
                    f"Hypothesis fingerprint {fp[:12]} repeats graveyard entry "
                    f"{gy['hypothesis_id']} ({gy['reason']})"
                )

            existing = conn.execute(
                "SELECT payload_json FROM hypotheses WHERE goal_id = ? AND fingerprint = ?",
                (normalized.goal_id, fp),
            ).fetchone()
            if existing is not None:
                return HypothesisNode.from_dict(json.loads(existing["payload_json"])), False

            by_id = conn.execute(
                "SELECT goal_id, payload_json FROM hypotheses WHERE hypothesis_id = ?",
                (normalized.hypothesis_id,),
            ).fetchone()
            if by_id is not None:
                if str(by_id["goal_id"]) != normalized.goal_id:
                    raise ValueError(
                        f"Cross-goal ownership violation: hypothesis_id {normalized.hypothesis_id!r} "
                        f"already belongs to goal {by_id['goal_id']!r}, not {normalized.goal_id!r}"
                    )
                return HypothesisNode.from_dict(json.loads(by_id["payload_json"])), False

            conn.execute(
                """
                INSERT INTO hypotheses (
                    hypothesis_id, goal_id, parent_id, kind, fidelity, priority_evoi,
                    branch_name, base_sha, head_sha, status, scientific_verdict,
                    failure_domain, fingerprint, payload_json, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    normalized.hypothesis_id,
                    normalized.goal_id,
                    normalized.parent_id,
                    normalized.kind.value,
                    normalized.fidelity.value,
                    float(normalized.priority_evoi),
                    normalized.branch_name,
                    normalized.base_sha,
                    normalized.head_sha,
                    normalized.status,
                    normalized.scientific_verdict.value,
                    normalized.failure_domain.value,
                    fp,
                    payload_json,
                    now,
                    now,
                ),
            )
            self._record_event(
                conn,
                pending,
                "HYPOTHESIS_REGISTERED",
                goal_id=normalized.goal_id,
                hypothesis_id=normalized.hypothesis_id,
                payload={"fingerprint": fp, "kind": normalized.kind.value},
                ts=now,
            )
        return normalized, True

    def get_hypothesis(self, hypothesis_id: str) -> HypothesisNode | None:
        """Fetch a HypothesisNode by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM hypotheses WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone()
        if row is None:
            return None
        return HypothesisNode.from_dict(json.loads(row["payload_json"]))

    def get_goal_hypothesis(self, goal_id: str, hypothesis_id: str) -> HypothesisNode | None:
        """Fetch a HypothesisNode by ID and strictly verify `hyp.goal_id == goal_id`."""
        hyp = self.get_hypothesis(hypothesis_id)
        if hyp is None:
            return None
        if hyp.goal_id != goal_id:
            raise ValueError(
                f"Cross-goal ownership violation: hypothesis {hypothesis_id!r} belongs to "
                f"goal {hyp.goal_id!r}, not {goal_id!r}"
            )
        return hyp

    def list_hypotheses(
        self,
        goal_id: str | None = None,
        *,
        status: str | None = None,
        kind: HypothesisKind | str | None = None,
    ) -> list[HypothesisNode]:
        """Query hypotheses ordered by descending priority_evoi."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if kind is not None:
            kind_val = kind.value if isinstance(kind, HypothesisKind) else HypothesisKind(kind).value
            clauses.append("kind = ?")
            params.append(kind_val)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT payload_json FROM hypotheses{where} "
            "ORDER BY priority_evoi DESC, created_ts ASC"
        )
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [HypothesisNode.from_dict(json.loads(r["payload_json"])) for r in rows]

    def get_hypothesis_children(self, parent_id: str) -> list[HypothesisNode]:
        """Return direct child hypotheses of a parent node."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload_json FROM hypotheses WHERE parent_id = ? ORDER BY priority_evoi DESC, created_ts ASC",
                (parent_id,),
            ).fetchall()
        return [HypothesisNode.from_dict(json.loads(r["payload_json"])) for r in rows]

    def update_hypothesis_params(
        self,
        hypothesis_id: str,
        params_patch: Mapping[str, Any],
        *,
        now: float | None = None,
    ) -> HypothesisNode:
        """Merge `params_patch` into a hypothesis's `params` in SQLite without altering its fingerprint."""
        ts = float(now if now is not None else time.time())
        with self._tx() as (conn, _pending):
            row = conn.execute(
                "SELECT payload_json FROM hypotheses WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown hypothesis_id: {hypothesis_id}")
            node = HypothesisNode.from_dict(json.loads(row["payload_json"]))
            node.params = {**dict(node.params), **dict(params_patch)}
            payload_json = json.dumps(node.to_dict(), sort_keys=True, ensure_ascii=False)
            conn.execute(
                "UPDATE hypotheses SET payload_json = ?, updated_ts = ? WHERE hypothesis_id = ?",
                (payload_json, ts, hypothesis_id),
            )
        return node

    def update_hypothesis_priority(
        self,
        hypothesis_id: str,
        priority_evoi: float,
        *,
        now: float | None = None,
    ) -> HypothesisNode:
        """Update a hypothesis's priority_evoi in SQLite."""
        ts = float(now if now is not None else time.time())
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT payload_json FROM hypotheses WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown hypothesis_id: {hypothesis_id}")
            node = HypothesisNode.from_dict(json.loads(row["payload_json"]))
            node.priority_evoi = float(priority_evoi)
            payload_json = json.dumps(node.to_dict(), sort_keys=True, ensure_ascii=False)
            conn.execute(
                "UPDATE hypotheses SET priority_evoi = ?, payload_json = ?, updated_ts = ? WHERE hypothesis_id = ?",
                (float(priority_evoi), payload_json, ts, hypothesis_id),
            )
            self._record_event(
                conn,
                pending,
                "HYPOTHESIS_PRIORITY_UPDATED",
                goal_id=node.goal_id,
                hypothesis_id=node.hypothesis_id,
                payload={"priority_evoi": float(priority_evoi)},
                ts=ts,
            )
        return node

    def transition_hypothesis_status(
        self,
        hypothesis_id: str,
        expected_statuses: Sequence[str] | str,
        new_status: str,
        *,
        head_sha: str | None = None,
        scientific_verdict: ScientificVerdict | str | None = None,
        fidelity: FidelityRung | str | None = None,
        failure_domain: FailureDomain | str | None = None,
        budget_state_patch: Mapping[str, Any] | None = None,
        exp_plan_id: str | None = None,
        now: float | None = None,
    ) -> HypothesisNode:
        """Atomically transition a hypothesis status if current status matches expectation."""
        expected = (
            {expected_statuses}
            if isinstance(expected_statuses, str)
            else set(expected_statuses)
        )
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT status, payload_json FROM hypotheses WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown hypothesis_id: {hypothesis_id}")
            cur_status = row["status"]
            if cur_status not in expected:
                raise StateTransitionError(
                    f"Hypothesis {hypothesis_id} status is {cur_status!r}, expected one of {sorted(expected)}"
                )
            node = HypothesisNode.from_dict(json.loads(row["payload_json"]))
            node.status = new_status
            if head_sha is not None:
                node.head_sha = head_sha
            if scientific_verdict is not None:
                node.scientific_verdict = (
                    scientific_verdict
                    if isinstance(scientific_verdict, ScientificVerdict)
                    else ScientificVerdict(scientific_verdict)
                )
            if fidelity is not None:
                node.fidelity = (
                    fidelity
                    if isinstance(fidelity, FidelityRung)
                    else FidelityRung(fidelity)
                )
            if failure_domain is not None:
                node.failure_domain = (
                    failure_domain
                    if isinstance(failure_domain, FailureDomain)
                    else FailureDomain(failure_domain)
                )
            if budget_state_patch is not None:
                node.budget_state = {**dict(node.budget_state), **dict(budget_state_patch)}
            if exp_plan_id is not None:
                node.exp_plan_id = exp_plan_id
            payload_json = json.dumps(node.to_dict(), sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                UPDATE hypotheses
                SET status = ?, head_sha = ?, scientific_verdict = ?, fidelity = ?,
                    failure_domain = ?, payload_json = ?, updated_ts = ?
                WHERE hypothesis_id = ? AND status = ?
                """,
                (
                    node.status,
                    node.head_sha,
                    node.scientific_verdict.value,
                    node.fidelity.value,
                    node.failure_domain.value,
                    payload_json,
                    ts,
                    hypothesis_id,
                    cur_status,
                ),
            )
            self._record_event(
                conn,
                pending,
                "HYPOTHESIS_TRANSITION",
                goal_id=node.goal_id,
                hypothesis_id=node.hypothesis_id,
                payload={
                    "from": cur_status,
                    "to": node.status,
                    "scientific_verdict": node.scientific_verdict.value,
                    "failure_domain": node.failure_domain.value,
                    "head_sha": node.head_sha,
                },
                ts=ts,
            )
        return node

    def update_hypothesis_v3_state(
        self,
        hypothesis_id: str,
        *,
        failure_domain: FailureDomain | str | None = None,
        budget_state_patch: Mapping[str, Any] | None = None,
        exp_plan_id: str | None = None,
        uncertainty_id: str | None = None,
        tags: Sequence[str] | None = None,
        now: float | None = None,
    ) -> HypothesisNode:
        """Atomically update V3 failure_domain, budget_state, plan link, or tags on a hypothesis."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, _pending):
            row = conn.execute(
                "SELECT payload_json FROM hypotheses WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown hypothesis_id: {hypothesis_id}")
            node = HypothesisNode.from_dict(json.loads(row["payload_json"]))
            if failure_domain is not None:
                node.failure_domain = (
                    failure_domain
                    if isinstance(failure_domain, FailureDomain)
                    else FailureDomain(failure_domain)
                )
            if budget_state_patch is not None:
                node.budget_state = {**dict(node.budget_state), **dict(budget_state_patch)}
            if exp_plan_id is not None:
                node.exp_plan_id = exp_plan_id
            if uncertainty_id is not None:
                node.uncertainty_id = uncertainty_id
            if tags is not None:
                node.tags = [str(t) for t in tags]
            payload_json = json.dumps(node.to_dict(), sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                UPDATE hypotheses
                SET failure_domain = ?, payload_json = ?, updated_ts = ?
                WHERE hypothesis_id = ?
                """,
                (node.failure_domain.value, payload_json, ts, hypothesis_id),
            )
        return node

    def record_graveyard(
        self,
        hypothesis_id: str,
        reason: str,
        *,
        scientific_verdict: ScientificVerdict | str = ScientificVerdict.REFUTED,
        details: dict[str, Any] | None = None,
        new_status: str = "GRAVEYARD",
    ) -> dict[str, Any]:
        """Record a hypothesis fingerprint in the graveyard and update its status."""
        verdict = (
            scientific_verdict
            if isinstance(scientific_verdict, ScientificVerdict)
            else ScientificVerdict(scientific_verdict)
        )
        now = time.time()
        details_json = json.dumps(details or {}, sort_keys=True, ensure_ascii=False)
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT fingerprint, goal_id, payload_json FROM hypotheses WHERE hypothesis_id = ?",
                (hypothesis_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown hypothesis_id: {hypothesis_id}")
            node = HypothesisNode.from_dict(json.loads(row["payload_json"]))
            fp = row["fingerprint"] or fingerprint_for_node(node)
            node.status = new_status
            node.scientific_verdict = verdict
            node.fingerprint = fp
            conn.execute(
                """
                INSERT INTO graveyard (
                    fingerprint, goal_id, hypothesis_id, reason, scientific_verdict,
                    details_json, created_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint) DO UPDATE SET
                    reason=excluded.reason,
                    scientific_verdict=excluded.scientific_verdict,
                    details_json=excluded.details_json
                """,
                (fp, node.goal_id, node.hypothesis_id, reason, verdict.value, details_json, now),
            )
            conn.execute(
                """
                UPDATE hypotheses
                SET status = ?, scientific_verdict = ?, payload_json = ?, updated_ts = ?
                WHERE hypothesis_id = ?
                """,
                (
                    node.status,
                    node.scientific_verdict.value,
                    json.dumps(node.to_dict(), sort_keys=True, ensure_ascii=False),
                    now,
                    hypothesis_id,
                ),
            )
            self._record_event(
                conn,
                pending,
                "HYPOTHESIS_GRAVEYARDED",
                goal_id=node.goal_id,
                hypothesis_id=node.hypothesis_id,
                payload={"fingerprint": fp, "reason": reason, "verdict": verdict.value},
                ts=now,
            )
        return {
            "fingerprint": fp,
            "goal_id": node.goal_id,
            "hypothesis_id": node.hypothesis_id,
            "reason": reason,
            "scientific_verdict": verdict.value,
            "created_ts": now,
        }

    def list_graveyard(self, goal_id: str | None = None) -> list[dict[str, Any]]:
        """List graveyard entries, optionally filtered by goal_id."""
        query = "SELECT * FROM graveyard"
        params: list[Any] = []
        if goal_id is not None:
            query += " WHERE goal_id = ?"
            params.append(goal_id)
        query += " ORDER BY created_ts DESC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["details"] = json.loads(item.pop("details_json"))
            out.append(item)
        return out

    # ------------------------------------------------------------------ Tasks

    def enqueue_task(self, task: StageTask) -> StageTask:
        """Idempotently enqueue a StageTask keyed by `idempotency_key`."""
        now = time.time()
        enqueued_ts = task.enqueued_ts if task.enqueued_ts > 0 else now
        stored = StageTask.from_dict({**task.to_dict(), "enqueued_ts": enqueued_ts})
        stored_dict = stored.to_dict()
        payload_json = json.dumps(stored_dict, sort_keys=True, ensure_ascii=False)

        with self._tx() as (conn, pending):
            hyp_owner = conn.execute(
                "SELECT goal_id FROM hypotheses WHERE hypothesis_id = ?",
                (stored.hypothesis_id,),
            ).fetchone()
            if hyp_owner is not None and str(hyp_owner["goal_id"]) != stored.goal_id:
                raise ValueError(
                    f"Cross-goal ownership violation: task {stored.task_id!r} has goal_id={stored.goal_id!r} "
                    f"but hypothesis {stored.hypothesis_id!r} belongs to {hyp_owner['goal_id']!r}"
                )
            if stored.workflow_id:
                wf_owner = conn.execute(
                    "SELECT goal_id, hypothesis_id FROM workflows WHERE workflow_id = ?",
                    (stored.workflow_id,),
                ).fetchone()
                if wf_owner is not None and (
                    str(wf_owner["goal_id"]) != stored.goal_id
                    or str(wf_owner["hypothesis_id"]) != stored.hypothesis_id
                ):
                    raise ValueError(
                        f"Cross-goal/hypothesis workflow mismatch on task {stored.task_id!r}: "
                        f"workflow {stored.workflow_id!r} belongs to ({wf_owner['goal_id']!r}, {wf_owner['hypothesis_id']!r})"
                    )
            existing_by_id = conn.execute(
                "SELECT goal_id, hypothesis_id, idempotency_key FROM tasks WHERE task_id = ?",
                (stored.task_id,),
            ).fetchone()
            if existing_by_id is not None and (
                str(existing_by_id["goal_id"]) != stored.goal_id
                or str(existing_by_id["hypothesis_id"]) != stored.hypothesis_id
            ):
                raise ValueError(
                    f"Cross-goal/hypothesis task_id collision: {stored.task_id!r} already belongs to "
                    f"({existing_by_id['goal_id']!r}, {existing_by_id['hypothesis_id']!r})"
                )
            existing = conn.execute(
                "SELECT goal_id, hypothesis_id, payload_json FROM tasks WHERE idempotency_key = ?",
                (stored.idempotency_key,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["goal_id"]) != stored.goal_id
                    or str(existing["hypothesis_id"]) != stored.hypothesis_id
                ):
                    raise ValueError(
                        f"Cross-goal/hypothesis idempotency_key collision: {stored.idempotency_key!r} belongs to "
                        f"({existing['goal_id']!r}, {existing['hypothesis_id']!r})"
                    )
                return StageTask.from_dict(json.loads(existing["payload_json"]))

            conn.execute(
                """
                INSERT INTO tasks (
                    task_id, idempotency_key, goal_id, hypothesis_id, stage, status,
                    hq_job_id, attempt, max_attempts, worktree_path,
                    model_spec, hq_llm_resource, plan_epoch,
                    enqueued_ts, started_ts, finished_ts, payload_json, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored.task_id,
                    stored.idempotency_key,
                    stored.goal_id,
                    stored.hypothesis_id,
                    stored.stage.value,
                    stored.status,
                    stored.hq_job_id,
                    int(stored.attempt),
                    int(stored.max_attempts),
                    stored.worktree_path,
                    str(stored_dict.get("model_spec", "")),
                    str(stored_dict.get("hq_llm_resource", "")),
                    int(getattr(stored, "plan_epoch", 0) or 0),
                    float(stored.enqueued_ts),
                    stored.started_ts,
                    stored.finished_ts,
                    payload_json,
                    now,
                ),
            )
            self._record_event(
                conn,
                pending,
                "TASK_ENQUEUED",
                goal_id=stored.goal_id,
                hypothesis_id=stored.hypothesis_id,
                task_id=stored.task_id,
                payload={
                    "idempotency_key": stored.idempotency_key,
                    "stage": stored.stage.value,
                    "status": stored.status,
                },
                ts=now,
            )
        return stored

    save_stage_task = enqueue_task

    def get_task(self, task_id: str) -> StageTask | None:
        """Fetch a StageTask by task_id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        return StageTask.from_dict(json.loads(row["payload_json"]))

    def get_goal_task(self, goal_id: str, task_id: str) -> StageTask | None:
        """Fetch a StageTask by ID and strictly verify `task.goal_id == goal_id`."""
        task = self.get_task(task_id)
        if task is None:
            return None
        if task.goal_id != goal_id:
            raise ValueError(
                f"Cross-goal ownership violation: task {task_id!r} belongs to "
                f"goal {task.goal_id!r}, not {goal_id!r}"
            )
        return task

    def get_task_by_idempotency_key(self, idempotency_key: str) -> StageTask | None:
        """Fetch a StageTask by idempotency_key."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM tasks WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return StageTask.from_dict(json.loads(row["payload_json"]))

    def list_tasks(
        self,
        goal_id: str | None = None,
        *,
        hypothesis_id: str | None = None,
        status: str | None = None,
        stage: StageKind | str | None = None,
    ) -> list[StageTask]:
        """Query tasks with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if stage is not None:
            stage_val = stage.value if isinstance(stage, StageKind) else StageKind(stage).value
            clauses.append("stage = ?")
            params.append(stage_val)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = f"SELECT payload_json FROM tasks{where} ORDER BY enqueued_ts ASC, task_id ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [StageTask.from_dict(json.loads(r["payload_json"])) for r in rows]

    def list_ready_tasks(self, goal_id: str | None = None) -> list[StageTask]:
        """Return tasks in READY status whose dependency tasks have all succeeded."""
        all_tasks = self.list_tasks(goal_id=goal_id)
        status_by_id = {t.task_id: t.status for t in all_tasks}
        ready: list[StageTask] = []
        for task in all_tasks:
            if task.status != "READY":
                continue
            if all(status_by_id.get(dep) in TERMINAL_TASK_SUCCESS for dep in task.deps):
                ready.append(task)
        return ready

    def transition_task_status(
        self,
        task_id: str,
        expected_statuses: Sequence[str] | str,
        new_status: str,
        *,
        hq_job_id: int | None = None,
        worktree_path: str | None = None,
        started_ts: float | None = None,
        finished_ts: float | None = None,
        attempt: int | None = None,
        pinned_route: Mapping[str, Any] | None = None,
        human_epoch: int | None = None,
        plan_epoch: int | None = None,
        now: float | None = None,
    ) -> StageTask:
        """Atomically transition a task's status if its current status matches."""
        expected = (
            {expected_statuses}
            if isinstance(expected_statuses, str)
            else set(expected_statuses)
        )
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT status, payload_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown task_id: {task_id}")
            cur_status = row["status"]
            if cur_status not in expected:
                raise StateTransitionError(
                    f"Task {task_id} status is {cur_status!r}, expected one of {sorted(expected)}"
                )
            task = StageTask.from_dict(json.loads(row["payload_json"]))
            prev_hq_job_id = task.hq_job_id
            task.status = new_status
            if hq_job_id is not None:
                task.hq_job_id = int(hq_job_id)
            if worktree_path is not None:
                task.worktree_path = worktree_path
            if started_ts is not None:
                task.started_ts = float(started_ts)
            if finished_ts is not None:
                task.finished_ts = float(finished_ts)
            prev_attempt = int(task.attempt)
            if attempt is not None:
                task.attempt = int(attempt)
            if pinned_route is not None:
                task.pinned_route = dict(pinned_route)
            if human_epoch is not None:
                task.human_epoch = int(human_epoch)
            if plan_epoch is not None:
                task.plan_epoch = int(plan_epoch)
            task_dict = task.to_dict()
            payload_json = json.dumps(task_dict, sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, hq_job_id = ?, attempt = ?, worktree_path = ?,
                    model_spec = ?, hq_llm_resource = ?, plan_epoch = ?,
                    started_ts = ?, finished_ts = ?, payload_json = ?, updated_ts = ?
                WHERE task_id = ? AND status = ?
                """,
                (
                    task.status,
                    task.hq_job_id,
                    task.attempt,
                    task.worktree_path,
                    str(task_dict.get("model_spec", "")),
                    str(task_dict.get("hq_llm_resource", "")),
                    int(getattr(task, "plan_epoch", 0) or 0),
                    task.started_ts,
                    task.finished_ts,
                    payload_json,
                    ts,
                    task_id,
                    cur_status,
                ),
            )
            if new_status in ("CANCELLED", "PRUNED", "FAILED", "HUMAN_INVALIDATED", "SUPERSEDED") or new_status.startswith("BLOCKED_"):
                is_logical_cancel = new_status in ("CANCELLED", "PRUNED", "HUMAN_INVALIDATED", "SUPERSEDED")
                was_hq_submitted = (prev_hq_job_id is not None) or (task.hq_job_id is not None)
                has_unconf = self._has_unconfirmed_hq_cancel_in_conn(conn, "task", task_id, prev_attempt)
                if not has_unconf and not (is_logical_cancel and was_hq_submitted):
                    conn.execute(
                        """
                        UPDATE token_usage_ledger
                        SET reserved_tokens = 0, state = 'RELEASED', updated_ts = ?
                        WHERE usage_key = ? AND state = 'RESERVED'
                        """,
                        (ts, f"task:{task_id}:a{prev_attempt}"),
                    )
            self._record_event(
                conn,
                pending,
                "TASK_TRANSITION",
                goal_id=task.goal_id,
                hypothesis_id=task.hypothesis_id,
                task_id=task.task_id,
                payload={
                    "from": cur_status,
                    "to": task.status,
                    "attempt": task.attempt,
                    "hq_job_id": task.hq_job_id,
                },
                ts=ts,
            )
        return task

    # ----------------------------------------------------------------- Leases

    def _reap_expired_in_tx(
        self,
        conn: sqlite3.Connection,
        pending: list[dict[str, Any]],
        *,
        now: float,
        resource_key: str | None = None,
    ) -> list[dict[str, Any]]:
        query = "SELECT * FROM leases WHERE status = 'ACTIVE' AND expires_ts <= ?"
        params: list[Any] = [now]
        if resource_key is not None:
            query += " AND resource_key = ?"
            params.append(resource_key)
        rows = conn.execute(query, params).fetchall()
        reaped: list[dict[str, Any]] = []
        for r in rows:
            conn.execute(
                "UPDATE leases SET status = 'EXPIRED', released_ts = ? WHERE lease_id = ? AND status = 'ACTIVE'",
                (now, r["lease_id"]),
            )
            item = dict(r)
            item["status"] = "EXPIRED"
            item["released_ts"] = now
            reaped.append(item)
            self._record_event(
                conn,
                pending,
                "LEASE_EXPIRED",
                goal_id=r["goal_id"] or None,
                task_id=r["task_id"],
                payload={"lease_id": r["lease_id"], "resource_key": r["resource_key"]},
                ts=now,
            )
        return reaped

    def acquire_lease(
        self,
        resource_key: str,
        task_id: str,
        holder: str,
        ttl_s: float = 300.0,
        *,
        goal_id: str = "",
        capacity: int = 1,
        max_concurrent: int = 1,
        allow_same_holder_renew: bool = True,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Acquire a TTL resource lease if capacity permits; returns lease dict or None."""
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        if capacity <= 0 or max_concurrent <= 0:
            raise ValueError("capacity and max_concurrent must be positive")
        ts = time.time() if now is None else float(now)
        expires = ts + float(ttl_s)

        with self._tx() as (conn, pending):
            self._reap_expired_in_tx(conn, pending, now=ts, resource_key=resource_key)

            if allow_same_holder_renew:
                existing = conn.execute(
                    """
                    SELECT * FROM leases
                    WHERE resource_key = ? AND task_id = ? AND holder = ? AND status = 'ACTIVE'
                    """,
                    (resource_key, task_id, holder),
                ).fetchone()
                if existing is not None:
                    conn.execute(
                        "UPDATE leases SET renewed_ts = ?, expires_ts = ? WHERE lease_id = ?",
                        (ts, expires, existing["lease_id"]),
                    )
                    out = dict(existing)
                    out["renewed_ts"] = ts
                    out["expires_ts"] = expires
                    return out

            row = conn.execute(
                """
                SELECT COALESCE(SUM(capacity), 0) AS used
                FROM leases
                WHERE resource_key = ? AND status = 'ACTIVE' AND expires_ts > ?
                """,
                (resource_key, ts),
            ).fetchone()
            used = int(row["used"])
            if used + capacity > max_concurrent:
                return None

            lease_id = f"lease_{uuid.uuid4().hex[:16]}"
            conn.execute(
                """
                INSERT INTO leases (
                    lease_id, resource_key, task_id, goal_id, holder, capacity,
                    acquired_ts, renewed_ts, expires_ts, released_ts, status
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, 'ACTIVE')
                """,
                (
                    lease_id,
                    resource_key,
                    task_id,
                    goal_id,
                    holder,
                    int(capacity),
                    ts,
                    ts,
                    expires,
                ),
            )
            self._record_event(
                conn,
                pending,
                "LEASE_ACQUIRED",
                goal_id=goal_id or None,
                task_id=task_id,
                payload={
                    "lease_id": lease_id,
                    "resource_key": resource_key,
                    "holder": holder,
                    "capacity": int(capacity),
                    "expires_ts": expires,
                },
                ts=ts,
            )
            return {
                "lease_id": lease_id,
                "resource_key": resource_key,
                "task_id": task_id,
                "goal_id": goal_id,
                "holder": holder,
                "capacity": int(capacity),
                "acquired_ts": ts,
                "renewed_ts": ts,
                "expires_ts": expires,
                "released_ts": None,
                "status": "ACTIVE",
            }

    def renew_lease(
        self,
        lease_id: str,
        ttl_s: float = 300.0,
        *,
        holder: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Renew an active, non-expired TTL lease."""
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        ts = time.time() if now is None else float(now)
        expires = ts + float(ttl_s)
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT * FROM leases WHERE lease_id = ?", (lease_id,)
            ).fetchone()
            if row is None or row["status"] != "ACTIVE":
                return None
            if float(row["expires_ts"]) <= ts:
                self._reap_expired_in_tx(conn, pending, now=ts, resource_key=row["resource_key"])
                return None
            if holder is not None and row["holder"] != holder:
                return None
            conn.execute(
                "UPDATE leases SET renewed_ts = ?, expires_ts = ? WHERE lease_id = ? AND status = 'ACTIVE'",
                (ts, expires, lease_id),
            )
            out = dict(row)
            out["renewed_ts"] = ts
            out["expires_ts"] = expires
            return out

    def release_lease(
        self,
        lease_id: str,
        *,
        holder: str | None = None,
        now: float | None = None,
    ) -> bool:
        """Release an active lease."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT * FROM leases WHERE lease_id = ? AND status = 'ACTIVE'",
                (lease_id,),
            ).fetchone()
            if row is None:
                return False
            if holder is not None and row["holder"] != holder:
                return False
            cur = conn.execute(
                "UPDATE leases SET status = 'RELEASED', released_ts = ? WHERE lease_id = ? AND status = 'ACTIVE'",
                (ts, lease_id),
            )
            if cur.rowcount == 0:
                return False
            self._record_event(
                conn,
                pending,
                "LEASE_RELEASED",
                goal_id=row["goal_id"] or None,
                task_id=row["task_id"],
                payload={"lease_id": lease_id, "resource_key": row["resource_key"]},
                ts=ts,
            )
            return True

    def reap_expired_leases(self, *, now: float | None = None) -> list[dict[str, Any]]:
        """Mark all expired ACTIVE leases as EXPIRED and return them."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            return self._reap_expired_in_tx(conn, pending, now=ts)

    def list_active_leases(
        self, resource_key: str | None = None, *, now: float | None = None
    ) -> list[dict[str, Any]]:
        """List currently active, non-expired leases."""
        ts = time.time() if now is None else float(now)
        query = "SELECT * FROM leases WHERE status = 'ACTIVE' AND expires_ts > ?"
        params: list[Any] = [ts]
        if resource_key is not None:
            query += " AND resource_key = ?"
            params.append(resource_key)
        query += " ORDER BY acquired_ts ASC"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------ Evidence & Budget

    def record_receipt(self, receipt: EvidenceReceipt) -> EvidenceReceipt:
        """Persist an EvidenceReceipt idempotently by receipt_id."""
        now = time.time()
        created_ts = receipt.created_ts if receipt.created_ts > 0 else now
        stored = EvidenceReceipt.from_dict({**receipt.to_dict(), "created_ts": created_ts})
        stored_dict = stored.to_dict()
        payload_json = json.dumps(stored_dict, sort_keys=True, ensure_ascii=False)

        with self._tx() as (conn, pending):
            existing = conn.execute(
                "SELECT payload_json FROM evidence_receipts WHERE receipt_id = ?",
                (stored.receipt_id,),
            ).fetchone()
            if existing is not None:
                return EvidenceReceipt.from_dict(json.loads(existing["payload_json"]))

            conn.execute(
                """
                INSERT INTO evidence_receipts (
                    receipt_id, task_id, goal_id, hypothesis_id, stage,
                    process_status, engineering_verdict, scientific_verdict, audit_verdict,
                    failure_domain, effort_used, plan_epoch,
                    in_tokens, out_tokens, wall_sec, max_rss_mb,
                    tests_passed, tests_total, created_ts, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    stored.receipt_id,
                    stored.task_id,
                    stored.goal_id,
                    stored.hypothesis_id,
                    stored.stage.value,
                    stored.process_status.value,
                    stored.engineering_verdict.value,
                    stored.scientific_verdict.value,
                    stored.audit_verdict.value,
                    stored.failure_domain.value,
                    stored.effort_used,
                    int(getattr(stored, "plan_epoch", 0) or 0),
                    int(stored.in_tokens),
                    int(stored.out_tokens),
                    float(stored.wall_sec),
                    float(stored.max_rss_mb),
                    int(stored.tests_passed),
                    int(stored.tests_total),
                    float(stored.created_ts),
                    payload_json,
                ),
            )
            import re as _re_att

            m_att_raw = (stored.metrics or {}).get("attempt")
            if isinstance(m_att_raw, Mapping):
                m_att_raw = m_att_raw.get("value")
            rid_att_match = _re_att.search(r"-a(\d+)-[0-9a-fA-F]+$", str(stored.receipt_id or ""))
            if m_att_raw is not None and int(m_att_raw or 0) > 0:
                t_att = int(m_att_raw)
            elif rid_att_match is not None and int(rid_att_match.group(1)) > 0:
                t_att = int(rid_att_match.group(1))
            else:
                t_row = (
                    conn.execute(
                        "SELECT attempt FROM tasks WHERE task_id = ?", (stored.task_id,)
                    ).fetchone()
                    if stored.task_id
                    else None
                )
                t_att = int(t_row["attempt"]) if t_row is not None else 1
            ukey = f"task:{stored.task_id}:a{t_att}" if stored.task_id else f"rcpt:{stored.receipt_id}"
            cache_raw = (stored.metrics or {}).get("cache_read_tokens", 0)
            cache_tok = (
                int(cache_raw.get("value", 0))
                if isinstance(cache_raw, Mapping)
                else int(cache_raw or 0)
            )
            self.settle_goal_budget(
                stored.goal_id,
                ukey,
                entity_kind="stage_task",
                entity_id=stored.task_id or stored.receipt_id,
                attempt=t_att,
                model=stored.model_used,
                input_tokens=int(stored.in_tokens),
                output_tokens=int(stored.out_tokens),
                cached_input_tokens=max(0, cache_tok),
                state="SETTLED",
            )
            self._record_event(
                conn,
                pending,
                "EVIDENCE_RECORDED",
                goal_id=stored.goal_id,
                hypothesis_id=stored.hypothesis_id,
                task_id=stored.task_id,
                payload={
                    "receipt_id": stored.receipt_id,
                    "stage": stored.stage.value,
                    "process_status": stored.process_status.value,
                    "engineering_verdict": stored.engineering_verdict.value,
                    "scientific_verdict": stored.scientific_verdict.value,
                    "audit_verdict": stored.audit_verdict.value,
                    "in_tokens": int(stored.in_tokens),
                    "out_tokens": int(stored.out_tokens),
                },
                ts=now,
            )
        return stored

    def get_receipt(self, receipt_id: str) -> EvidenceReceipt | None:
        """Fetch an EvidenceReceipt by ID."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM evidence_receipts WHERE receipt_id = ?",
                (receipt_id,),
            ).fetchone()
        if row is None:
            return None
        return EvidenceReceipt.from_dict(json.loads(row["payload_json"]))

    def list_receipts(
        self,
        goal_id: str | None = None,
        *,
        hypothesis_id: str | None = None,
        task_id: str | None = None,
        stage: StageKind | str | None = None,
    ) -> list[EvidenceReceipt]:
        """List evidence receipts with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if task_id is not None:
            clauses.append("task_id = ?")
            params.append(task_id)
        if stage is not None:
            stage_val = stage.value if isinstance(stage, StageKind) else StageKind(stage).value
            clauses.append("stage = ?")
            params.append(stage_val)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT payload_json FROM evidence_receipts{where} "
            "ORDER BY created_ts ASC, receipt_id ASC"
        )
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [EvidenceReceipt.from_dict(json.loads(r["payload_json"])) for r in rows]

    def get_latest_receipt(
        self,
        hypothesis_id: str,
        stage: StageKind | str | None = None,
    ) -> EvidenceReceipt | None:
        """Fetch the most recent receipt for a hypothesis (and optional stage)."""
        receipts = self.list_receipts(hypothesis_id=hypothesis_id, stage=stage)
        return receipts[-1] if receipts else None

    def get_authoritative_empirical_receipts(
        self,
        goal: ResearchGoal,
        hyp: HypothesisNode,
        *,
        preserved_receipt_ids: Sequence[str] | set[str] = (),
    ) -> list[EvidenceReceipt]:
        """Return empirical receipts for `(goal, hyp)` that remain authoritative for `hyp`'s current plan lineage."""
        from conjectra import gates

        latest_ep = self.get_latest_experiment_plan(hyp.hypothesis_id)
        extra_preserved: set[str] = {str(x) for x in preserved_receipt_ids if str(x)}
        for item in self.list_knowledge_items(goal_id=goal.goal_id, category="plan_lineage", limit=50):
            content = item.get("content")
            if not isinstance(content, Mapping):
                continue
            item_hid = item.get("hypothesis_id") or content.get("hypothesis_id")
            if item_hid and str(item_hid) != hyp.hypothesis_id:
                continue
            if int(content.get("plan_epoch", -1)) == int(hyp.params.get("lineage_plan_epoch", getattr(goal, "plan_epoch", 0)) or 0):
                for rid in content.get("preserved_upstream_receipt_ids") or []:
                    if str(rid).strip():
                        extra_preserved.add(str(rid).strip())

        rcpts = self.list_receipts(goal_id=goal.goal_id, hypothesis_id=hyp.hypothesis_id)
        return [
            r
            for r in rcpts
            if gates.is_receipt_authoritative_for_hypothesis(
                r,
                goal,
                hyp,
                preserved_receipt_ids=extra_preserved,
                latest_exp_plan=latest_ep,
            )
        ]

    def reserve_goal_budget(
        self,
        goal_id: str,
        usage_key: str,
        *,
        entity_kind: str,
        entity_id: str,
        attempt: int = 1,
        model: str = "",
        reserve_tokens: int = 4000,
        exclude_usage_key: str | None = None,
    ) -> tuple[bool, int, int]:
        """Atomically check and reserve concurrency admission budget in `token_usage_ledger`.

        Note: `reserve_tokens` is a concurrency admission reservation (not a hard provider cap);
        settled ledger budget spend is `uncached_input_tokens + output_tokens` (`cached_input_tokens`
        is tracked separately as telemetry). Returns `(allowed, committed_tokens, max_tokens_budget)`.
        """
        now = time.time()
        eff_reserve = max(0, int(reserve_tokens))
        eff_excl = str(exclude_usage_key or usage_key)
        eff_excl_glob = self._escape_glob(eff_excl) + ":*" if eff_excl else ""
        with self._tx() as (conn, _pending):
            goal_row = conn.execute(
                "SELECT max_tokens_budget FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
            if goal_row is None:
                return True, 0, 0
            budget = int(goal_row["max_tokens_budget"])

            agg = conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN state = 'SETTLED' THEN (input_tokens + output_tokens) ELSE 0 END), 0) AS settled,
                    COALESCE(SUM(CASE WHEN state = 'RESERVED' AND usage_key != ? AND (? = '' OR NOT (usage_key GLOB ?)) THEN reserved_tokens ELSE 0 END), 0) AS other_reserved
                FROM token_usage_ledger
                WHERE goal_id = ?
                """,
                (eff_excl, eff_excl_glob, eff_excl_glob, goal_id),
            ).fetchone()
            settled = int(agg["settled"])
            other_reserved = int(agg["other_reserved"])
            committed_other = settled + other_reserved

            existing = conn.execute(
                "SELECT state, reserved_tokens FROM token_usage_ledger WHERE usage_key = ?",
                (usage_key,),
            ).fetchone()
            if existing is not None and str(existing["state"]) in ("RESERVED", "SETTLED"):
                own_res = int(existing["reserved_tokens"]) if str(existing["state"]) == "RESERVED" else 0
                return True, committed_other + own_res, budget

            if eff_reserve <= 0:
                return True, committed_other, budget

            if budget <= 0 or settled >= budget or committed_other >= budget or (committed_other + eff_reserve) > budget:
                return False, committed_other, budget

            conn.execute(
                """
                INSERT INTO token_usage_ledger (
                    usage_key, goal_id, entity_kind, entity_id, attempt, model,
                    reserved_tokens, input_tokens, output_tokens, cached_input_tokens,
                    state, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, 0, 'RESERVED', ?, ?)
                ON CONFLICT(usage_key) DO UPDATE SET
                    reserved_tokens = excluded.reserved_tokens,
                    model = CASE WHEN excluded.model != '' THEN excluded.model ELSE token_usage_ledger.model END,
                    state = 'RESERVED',
                    updated_ts = excluded.updated_ts
                WHERE token_usage_ledger.state = 'RELEASED'
                """,
                (
                    usage_key,
                    goal_id,
                    str(entity_kind),
                    str(entity_id),
                    int(attempt),
                    str(model or ""),
                    eff_reserve,
                    now,
                    now,
                ),
            )
            return True, committed_other + eff_reserve, budget

    def release_goal_budget_reservation(
        self,
        goal_id: str,
        usage_key: str,
        *,
        reason: str = "",
        now: float | None = None,
    ) -> bool:
        """Release an unsettled `'RESERVED'` admission row without touching `'SETTLED'` actual spend."""
        _ = (goal_id, reason)
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, _pending):
            cur = conn.execute(
                """
                UPDATE token_usage_ledger
                SET reserved_tokens = 0, state = 'RELEASED', updated_ts = ?
                WHERE usage_key = ? AND state = 'RESERVED'
                """,
                (ts, usage_key),
            )
            return int(cur.rowcount or 0) > 0

    def settle_goal_budget(
        self,
        goal_id: str,
        usage_key: str,
        *,
        entity_kind: str = "stage_task",
        entity_id: str = "",
        attempt: int = 1,
        model: str = "",
        input_tokens: int = 0,
        output_tokens: int = 0,
        cached_input_tokens: int = 0,
        state: str = "SETTLED",
    ) -> int:
        """Idempotently settle actual uncached `input_tokens + output_tokens` for a task or decision attempt."""
        now = time.time()
        in_tok = max(0, int(input_tokens))
        out_tok = max(0, int(output_tokens))
        cache_tok = max(0, int(cached_input_tokens))
        with self._tx() as (conn, _pending):
            conn.execute(
                """
                INSERT INTO token_usage_ledger (
                    usage_key, goal_id, entity_kind, entity_id, attempt, model,
                    reserved_tokens, input_tokens, output_tokens, cached_input_tokens,
                    state, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(usage_key) DO UPDATE SET
                    reserved_tokens = 0,
                    input_tokens = MAX(token_usage_ledger.input_tokens, excluded.input_tokens),
                    output_tokens = MAX(token_usage_ledger.output_tokens, excluded.output_tokens),
                    cached_input_tokens = MAX(token_usage_ledger.cached_input_tokens, excluded.cached_input_tokens),
                    model = CASE WHEN excluded.model != '' THEN excluded.model ELSE token_usage_ledger.model END,
                    state = CASE
                        WHEN token_usage_ledger.state = 'SETTLED' THEN 'SETTLED'
                        ELSE excluded.state
                    END,
                    updated_ts = excluded.updated_ts
                """,
                (
                    usage_key,
                    goal_id,
                    str(entity_kind),
                    str(entity_id),
                    int(attempt),
                    str(model or ""),
                    in_tok,
                    out_tok,
                    cache_tok,
                    str(state or "SETTLED"),
                    now,
                    now,
                ),
            )
            tot_row = conn.execute(
                """
                SELECT COALESCE(SUM(input_tokens + output_tokens), 0) AS settled
                FROM token_usage_ledger
                WHERE goal_id = ? AND state = 'SETTLED'
                """,
                (goal_id,),
            ).fetchone()
            return int(tot_row["settled"])

    def get_goal_token_usage(
        self,
        goal_id: str,
        *,
        exclude_usage_key: str | None = None,
    ) -> dict[str, Any]:
        """Aggregate unified task + decision uncached input/output token spend and active admission reservations."""
        excl = str(exclude_usage_key or "")
        excl_glob = self._escape_glob(excl) + ":*" if excl else ""
        with self._lock:
            goal_row = self._conn.execute(
                "SELECT max_tokens_budget FROM goals WHERE goal_id = ?", (goal_id,)
            ).fetchone()
            rcpt_agg = self._conn.execute(
                """
                SELECT
                    COALESCE(SUM(in_tokens), 0) AS rcpt_in,
                    COALESCE(SUM(out_tokens), 0) AS rcpt_out,
                    COALESCE(SUM(wall_sec), 0.0) AS wall_sec,
                    COALESCE(MAX(max_rss_mb), 0.0) AS peak_rss_mb,
                    COUNT(*) AS receipts_count
                FROM evidence_receipts
                WHERE goal_id = ?
                """,
                (goal_id,),
            ).fetchone()
            ledger_agg = self._conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN state = 'SETTLED' THEN input_tokens ELSE 0 END), 0) AS in_tokens,
                    COALESCE(SUM(CASE WHEN state = 'SETTLED' THEN output_tokens ELSE 0 END), 0) AS out_tokens,
                    COALESCE(SUM(CASE WHEN state = 'SETTLED' THEN cached_input_tokens ELSE 0 END), 0) AS cached_input_tokens,
                    COALESCE(SUM(CASE WHEN state = 'RESERVED' AND (? = '' OR (usage_key != ? AND NOT (usage_key GLOB ?))) THEN reserved_tokens ELSE 0 END), 0) AS reserved_tokens,
                    COALESCE(SUM(CASE WHEN state = 'SETTLED' AND entity_kind = 'decision_request' THEN (input_tokens + output_tokens) ELSE 0 END), 0) AS decision_tokens
                FROM token_usage_ledger
                WHERE goal_id = ?
                """,
                (excl, excl, excl_glob, goal_id),
            ).fetchone()
        in_tokens = max(int(rcpt_agg["rcpt_in"]), int(ledger_agg["in_tokens"]))
        out_tokens = max(int(rcpt_agg["rcpt_out"]), int(ledger_agg["out_tokens"]))
        reserved_tokens = int(ledger_agg["reserved_tokens"])
        total_tokens = in_tokens + out_tokens
        committed_tokens = total_tokens + reserved_tokens
        budget = int(goal_row["max_tokens_budget"]) if goal_row is not None else 0
        remaining = max(0, budget - committed_tokens) if goal_row is not None else 0
        exhausted = bool(goal_row is not None and committed_tokens >= budget)
        return {
            "goal_id": goal_id,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "cached_input_tokens": int(ledger_agg["cached_input_tokens"]),
            "decision_tokens": int(ledger_agg["decision_tokens"]),
            "reserved_tokens": reserved_tokens,
            "committed_tokens": committed_tokens,
            "total_tokens": total_tokens,
            "max_tokens_budget": budget,
            "remaining_tokens": remaining,
            "budget_exhausted": exhausted,
            "wall_sec": float(rcpt_agg["wall_sec"]),
            "peak_rss_mb": float(rcpt_agg["peak_rss_mb"]),
            "receipts_count": int(rcpt_agg["receipts_count"]),
        }

    def is_budget_exhausted(
        self,
        goal_id: str,
        *,
        exclude_usage_key: str | None = None,
    ) -> bool:
        """Return True if a goal's settled spend + other active reservations reach or exceed `max_tokens_budget`."""
        return bool(
            self.get_goal_token_usage(goal_id, exclude_usage_key=exclude_usage_key)[
                "budget_exhausted"
            ]
        )

    def list_events(
        self,
        goal_id: str | None = None,
        *,
        since_seq: int = 0,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Query ordered ledger events after `since_seq`."""
        clauses = ["seq > ?"]
        params: list[Any] = [int(since_seq)]
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        params.append(int(limit))
        query = (
            f"SELECT seq, event_id, event_type, goal_id, hypothesis_id, task_id, ts, payload_json "
            f"FROM events WHERE {' AND '.join(clauses)} ORDER BY seq ASC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item.pop("payload_json"))
            out.append(item)
        return out

    def get_goal_summary(self, goal_id: str) -> dict[str, Any]:
        """Return aggregated state summary for controller.py and metrics.py."""
        hypotheses = self.list_hypotheses(goal_id=goal_id)
        tasks = self.list_tasks(goal_id=goal_id)
        receipts = self.list_receipts(goal_id=goal_id)
        graveyard = self.list_graveyard(goal_id=goal_id)
        leases = [l for l in self.list_active_leases() if l["goal_id"] == goal_id]
        budget = self.get_goal_token_usage(goal_id)

        hyp_by_status: dict[str, int] = {}
        hyp_by_verdict: dict[str, int] = {}
        for h in hypotheses:
            hyp_by_status[h.status] = hyp_by_status.get(h.status, 0) + 1
            v = h.scientific_verdict.value
            hyp_by_verdict[v] = hyp_by_verdict.get(v, 0) + 1

        task_by_status: dict[str, int] = {}
        task_by_stage: dict[str, int] = {}
        for t in tasks:
            task_by_status[t.status] = task_by_status.get(t.status, 0) + 1
            s = t.stage.value
            task_by_stage[s] = task_by_stage.get(s, 0) + 1

        return {
            "goal_id": goal_id,
            "hypotheses_total": len(hypotheses),
            "hypotheses_by_status": hyp_by_status,
            "hypotheses_by_verdict": hyp_by_verdict,
            "tasks_total": len(tasks),
            "tasks_by_status": task_by_status,
            "tasks_by_stage": task_by_stage,
            "receipts_total": len(receipts),
            "graveyard_total": len(graveyard),
            "active_leases": len(leases),
            "budget": budget,
        }

    # ----------------------------------------------------------- Global Config

    def set_global_config(
        self, key: str, value: Any, *, now: float | None = None
    ) -> dict[str, Any]:
        """Set a global runtime configuration entry in SQLite."""
        ts = time.time() if now is None else float(now)
        value_json = json.dumps(value, sort_keys=True, ensure_ascii=False)
        with self._tx() as (conn, pending):
            conn.execute(
                """
                INSERT INTO global_config (config_key, value_json, updated_ts)
                VALUES (?, ?, ?)
                ON CONFLICT(config_key) DO UPDATE SET
                    value_json=excluded.value_json,
                    updated_ts=excluded.updated_ts
                """,
                (str(key), value_json, ts),
            )
            self._record_event(
                conn,
                pending,
                "GLOBAL_CONFIG_UPDATED",
                payload={"key": str(key), "value": value},
                ts=ts,
            )
        return {"config_key": str(key), "value": value, "updated_ts": ts}

    def get_global_config(
        self, key: str | None = None, default: Any = None
    ) -> Any:
        """Fetch one global config value by key, or all config values when key is None."""
        with self._lock:
            if key is None:
                rows = self._conn.execute(
                    "SELECT config_key, value_json FROM global_config ORDER BY config_key ASC"
                ).fetchall()
                return {str(r["config_key"]): json.loads(r["value_json"]) for r in rows}
            row = self._conn.execute(
                "SELECT value_json FROM global_config WHERE config_key = ?",
                (str(key),),
            ).fetchone()
        if row is None:
            return default
        return json.loads(row["value_json"])

    # -------------------------------------------------------------- Workflows

    def save_workflow(
        self, workflow: WorkflowNode, *, now: float | None = None
    ) -> WorkflowNode:
        """Create or update a WorkflowNode in the ledger (ownership-checked)."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            hyp_owner = conn.execute(
                "SELECT goal_id FROM hypotheses WHERE hypothesis_id = ?",
                (workflow.hypothesis_id,),
            ).fetchone()
            if hyp_owner is not None and str(hyp_owner["goal_id"]) != workflow.goal_id:
                raise ValueError(
                    f"Cross-goal ownership violation: workflow {workflow.workflow_id!r} has goal_id={workflow.goal_id!r} "
                    f"but hypothesis {workflow.hypothesis_id!r} belongs to {hyp_owner['goal_id']!r}"
                )
            if workflow.parent_workflow_id:
                parent_wf = conn.execute(
                    "SELECT goal_id FROM workflows WHERE workflow_id = ?",
                    (workflow.parent_workflow_id,),
                ).fetchone()
                if parent_wf is not None and str(parent_wf["goal_id"]) != workflow.goal_id:
                    raise ValueError(
                        f"Cross-goal ownership violation: parent_workflow_id {workflow.parent_workflow_id!r} "
                        f"belongs to {parent_wf['goal_id']!r}, not {workflow.goal_id!r}"
                    )
            existing = conn.execute(
                "SELECT goal_id, hypothesis_id, created_ts FROM workflows WHERE workflow_id = ?",
                (workflow.workflow_id,),
            ).fetchone()
            if existing is not None and (
                str(existing["goal_id"]) != workflow.goal_id
                or str(existing["hypothesis_id"]) != workflow.hypothesis_id
            ):
                raise ValueError(
                    f"Cross-goal/hypothesis workflow_id collision: {workflow.workflow_id!r} belongs to "
                    f"({existing['goal_id']!r}, {existing['hypothesis_id']!r}), not ({workflow.goal_id!r}, {workflow.hypothesis_id!r})"
                )
            created_ts = (
                float(existing["created_ts"])
                if existing is not None
                else (workflow.created_ts if workflow.created_ts > 0 else ts)
            )
            updated_ts = ts
            stored = WorkflowNode.from_dict(
                {
                    **workflow.to_dict(),
                    "created_ts": created_ts,
                    "updated_ts": updated_ts,
                }
            )
            payload_json = json.dumps(
                stored.to_dict(), sort_keys=True, ensure_ascii=False
            )
            conn.execute(
                """
                INSERT INTO workflows (
                    workflow_id, goal_id, hypothesis_id, parent_workflow_id,
                    primitive, status, goal_version, budget_tokens, plan_epoch,
                    payload_json, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(workflow_id) DO UPDATE SET
                    goal_id=excluded.goal_id,
                    hypothesis_id=excluded.hypothesis_id,
                    parent_workflow_id=excluded.parent_workflow_id,
                    primitive=excluded.primitive,
                    status=excluded.status,
                    goal_version=excluded.goal_version,
                    budget_tokens=excluded.budget_tokens,
                    plan_epoch=excluded.plan_epoch,
                    payload_json=excluded.payload_json,
                    updated_ts=excluded.updated_ts
                """,
                (
                    stored.workflow_id,
                    stored.goal_id,
                    stored.hypothesis_id,
                    stored.parent_workflow_id,
                    stored.primitive,
                    stored.status,
                    int(stored.goal_version),
                    int(stored.budget_tokens),
                    int(getattr(stored, "plan_epoch", 0) or 0),
                    payload_json,
                    created_ts,
                    updated_ts,
                ),
            )
            self._record_event(
                conn,
                pending,
                "WORKFLOW_SAVED",
                goal_id=stored.goal_id,
                hypothesis_id=stored.hypothesis_id,
                payload={
                    "workflow_id": stored.workflow_id,
                    "primitive": stored.primitive,
                    "status": stored.status,
                    "parent_workflow_id": stored.parent_workflow_id,
                },
                ts=updated_ts,
            )
        return stored

    def get_workflow(self, workflow_id: str) -> WorkflowNode | None:
        """Fetch a WorkflowNode by workflow_id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
        if row is None:
            return None
        return WorkflowNode.from_dict(json.loads(row["payload_json"]))

    save_workflow_instance = save_workflow

    def get_goal_workflow(self, goal_id: str, workflow_id: str) -> WorkflowNode | None:
        """Fetch a WorkflowNode by ID and strictly verify `wf.goal_id == goal_id`."""
        wf = self.get_workflow(workflow_id)
        if wf is None:
            return None
        if wf.goal_id != goal_id:
            raise ValueError(
                f"Cross-goal ownership violation: workflow {workflow_id!r} belongs to "
                f"goal {wf.goal_id!r}, not {goal_id!r}"
            )
        return wf

    def list_workflows(
        self,
        goal_id: str | None = None,
        *,
        hypothesis_id: str | None = None,
        parent_workflow_id: str | None = None,
        status: str | None = None,
    ) -> list[WorkflowNode]:
        """List WorkflowNodes with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if parent_workflow_id is not None:
            clauses.append("parent_workflow_id = ?")
            params.append(parent_workflow_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        query = (
            f"SELECT payload_json FROM workflows{where} "
            "ORDER BY created_ts ASC, workflow_id ASC"
        )
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [WorkflowNode.from_dict(json.loads(r["payload_json"])) for r in rows]

    def update_workflow_state(
        self,
        workflow_id: str,
        *,
        status: str | None = None,
        state_patch: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        config_patch: Mapping[str, Any] | None = None,
        expected_statuses: Sequence[str] | str | None = None,
        budget_tokens: int | None = None,
        now: float | None = None,
    ) -> WorkflowNode:
        """Atomically update a WorkflowNode's status, state, config, or budget."""
        expected = (
            None
            if expected_statuses is None
            else (
                {expected_statuses}
                if isinstance(expected_statuses, str)
                else set(expected_statuses)
            )
        )
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT status, payload_json FROM workflows WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown workflow_id: {workflow_id}")
            cur_status = str(row["status"])
            if expected is not None and cur_status not in expected:
                raise StateTransitionError(
                    f"Workflow {workflow_id} status is {cur_status!r}, expected one of {sorted(expected)}"
                )
            wf = WorkflowNode.from_dict(json.loads(row["payload_json"]))
            if status is not None:
                wf.status = status
            if state is not None:
                wf.state = dict(state)
            elif state_patch is not None:
                wf.state = {**dict(wf.state), **dict(state_patch)}
            if config_patch is not None:
                wf.config = {**dict(wf.config), **dict(config_patch)}
            if budget_tokens is not None:
                wf.budget_tokens = int(budget_tokens)
            wf.updated_ts = ts
            payload_json = json.dumps(wf.to_dict(), sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                UPDATE workflows
                SET status = ?, budget_tokens = ?, payload_json = ?, updated_ts = ?
                WHERE workflow_id = ? AND status = ?
                """,
                (
                    wf.status,
                    int(wf.budget_tokens),
                    payload_json,
                    ts,
                    workflow_id,
                    cur_status,
                ),
            )
            self._record_event(
                conn,
                pending,
                "WORKFLOW_UPDATED",
                goal_id=wf.goal_id,
                hypothesis_id=wf.hypothesis_id,
                payload={
                    "workflow_id": wf.workflow_id,
                    "from": cur_status,
                    "to": wf.status,
                },
                ts=ts,
            )
        return wf

    # ---------------------------------------------------------- Goal Versions

    def save_goal_version(
        self,
        goal: ResearchGoal,
        *,
        change_summary: str = "",
        actor: str = "user",
        reason: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        """Persist an immutable/upserted snapshot for a ResearchGoal version."""
        ts = time.time() if now is None else float(now)
        version_id = f"gv_{goal.goal_id}_v{int(goal.version)}"
        payload_json = json.dumps(goal.to_dict(), sort_keys=True, ensure_ascii=False)
        with self._tx() as (conn, pending):
            existing = conn.execute(
                "SELECT version_id, created_ts FROM goal_versions WHERE goal_id = ? AND version = ?",
                (goal.goal_id, int(goal.version)),
            ).fetchone()
            if existing is not None:
                version_id = str(existing["version_id"])
                created_ts = float(existing["created_ts"])
            else:
                created_ts = ts
            conn.execute(
                """
                INSERT INTO goal_versions (
                    version_id, goal_id, version, human_epoch, payload_json,
                    change_summary, actor, reason, created_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(goal_id, version) DO UPDATE SET
                    human_epoch=excluded.human_epoch,
                    payload_json=excluded.payload_json,
                    change_summary=excluded.change_summary,
                    actor=excluded.actor,
                    reason=excluded.reason
                """,
                (
                    version_id,
                    goal.goal_id,
                    int(goal.version),
                    int(goal.human_epoch),
                    payload_json,
                    change_summary,
                    actor,
                    reason,
                    created_ts,
                ),
            )
            self._record_event(
                conn,
                pending,
                "GOAL_VERSION_SAVED",
                goal_id=goal.goal_id,
                payload={
                    "version_id": version_id,
                    "version": int(goal.version),
                    "human_epoch": int(goal.human_epoch),
                    "actor": actor,
                    "reason": reason,
                },
                ts=ts,
            )
        return {
            "version_id": version_id,
            "goal_id": goal.goal_id,
            "version": int(goal.version),
            "human_epoch": int(goal.human_epoch),
            "goal": goal.to_dict(),
            "change_summary": change_summary,
            "actor": actor,
            "reason": reason,
            "created_ts": created_ts,
        }

    def list_goal_versions(self, goal_id: str) -> list[dict[str, Any]]:
        """List saved versions for a goal ordered by ascending version."""
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT version_id, goal_id, version, human_epoch, payload_json,
                       change_summary, actor, reason, created_ts
                FROM goal_versions
                WHERE goal_id = ?
                ORDER BY version ASC, created_ts ASC
                """,
                (goal_id,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "version_id": str(r["version_id"]),
                    "goal_id": str(r["goal_id"]),
                    "version": int(r["version"]),
                    "human_epoch": int(r["human_epoch"]),
                    "goal": json.loads(r["payload_json"]),
                    "change_summary": str(r["change_summary"]),
                    "actor": str(r["actor"]),
                    "reason": str(r["reason"]),
                    "created_ts": float(r["created_ts"]),
                }
            )
        return out

    # ---------------------------------------------------------- Interventions

    def record_intervention(
        self,
        scope: str,
        action: str,
        *,
        goal_id: str | None = None,
        target_id: str | None = None,
        actor: str = "user",
        reason: str = "",
        before_epoch: int | None = None,
        after_epoch: int | None = None,
        payload: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        intervention_id: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Record a human/operator intervention in the audit table and event log."""
        ts = time.time() if now is None else float(now)
        iid = intervention_id or f"int_{uuid.uuid4().hex[:14]}"
        payload_dict = dict(payload or {})
        result_dict = dict(result or {})
        payload_json = json.dumps(payload_dict, sort_keys=True, ensure_ascii=False)
        result_json = json.dumps(result_dict, sort_keys=True, ensure_ascii=False)
        with self._tx() as (conn, pending):
            conn.execute(
                """
                INSERT INTO interventions (
                    intervention_id, scope, action, goal_id, target_id,
                    actor, reason, before_epoch, after_epoch,
                    payload_json, result_json, created_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    iid,
                    scope,
                    action,
                    goal_id,
                    target_id,
                    actor,
                    reason,
                    before_epoch,
                    after_epoch,
                    payload_json,
                    result_json,
                    ts,
                ),
            )
            self._record_event(
                conn,
                pending,
                "HUMAN_INTERVENTION",
                goal_id=goal_id,
                payload={
                    "intervention_id": iid,
                    "scope": scope,
                    "action": action,
                    "target_id": target_id,
                    "actor": actor,
                    "reason": reason,
                    "before_epoch": before_epoch,
                    "after_epoch": after_epoch,
                },
                ts=ts,
            )
        return {
            "intervention_id": iid,
            "scope": scope,
            "action": action,
            "goal_id": goal_id,
            "target_id": target_id,
            "actor": actor,
            "reason": reason,
            "before_epoch": before_epoch,
            "after_epoch": after_epoch,
            "payload": payload_dict,
            "result": result_dict,
            "created_ts": ts,
        }

    def list_interventions(
        self,
        goal_id: str | None = None,
        *,
        scope: str | None = None,
        action: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """List recorded interventions with optional filters (newest first)."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if scope is not None:
            clauses.append("scope = ?")
            params.append(scope)
        if action is not None:
            clauses.append("action = ?")
            params.append(action)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        query = (
            f"SELECT * FROM interventions{where} "
            "ORDER BY created_ts DESC, rowid DESC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["payload"] = json.loads(item.pop("payload_json"))
            item["result"] = json.loads(item.pop("result_json"))
            out.append(item)
        return out

    # --------------------------------------------------------- Goal Knowledge

    def upsert_knowledge_item(
        self,
        goal_id: str,
        category: str,
        key: str,
        content: Any,
        *,
        hypothesis_id: str | None = None,
        goal_version: int = 1,
        confidence: float = 1.0,
        item_id: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Insert or update a structured knowledge item keyed by (goal_id, category, key)."""
        ts = time.time() if now is None else float(now)
        content_json = json.dumps(content, sort_keys=True, ensure_ascii=False)
        with self._tx() as (conn, pending):
            existing = conn.execute(
                "SELECT item_id, created_ts FROM goal_knowledge WHERE goal_id = ? AND category = ? AND key = ?",
                (goal_id, category, key),
            ).fetchone()
            eff_id = (
                str(existing["item_id"])
                if existing is not None
                else (item_id or f"know_{uuid.uuid4().hex[:14]}")
            )
            created_ts = float(existing["created_ts"]) if existing is not None else ts
            conn.execute(
                """
                INSERT INTO goal_knowledge (
                    item_id, goal_id, hypothesis_id, category, key,
                    content_json, goal_version, confidence, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(goal_id, category, key) DO UPDATE SET
                    hypothesis_id=excluded.hypothesis_id,
                    content_json=excluded.content_json,
                    goal_version=excluded.goal_version,
                    confidence=excluded.confidence,
                    updated_ts=excluded.updated_ts
                """,
                (
                    eff_id,
                    goal_id,
                    hypothesis_id,
                    category,
                    key,
                    content_json,
                    int(goal_version),
                    float(confidence),
                    created_ts,
                    ts,
                ),
            )
            self._record_event(
                conn,
                pending,
                "KNOWLEDGE_UPSERTED",
                goal_id=goal_id,
                hypothesis_id=hypothesis_id,
                payload={"item_id": eff_id, "category": category, "key": key},
                ts=ts,
            )
        return {
            "item_id": eff_id,
            "goal_id": goal_id,
            "hypothesis_id": hypothesis_id,
            "category": category,
            "key": key,
            "content": content,
            "goal_version": int(goal_version),
            "confidence": float(confidence),
            "created_ts": created_ts,
            "updated_ts": ts,
        }

    def list_knowledge_items(
        self,
        goal_id: str | None = None,
        *,
        category: str | None = None,
        hypothesis_id: str | None = None,
        goal_version: int | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Query goal_knowledge entries."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if category is not None:
            clauses.append("category = ?")
            params.append(category)
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if goal_version is not None:
            clauses.append("goal_version = ?")
            params.append(int(goal_version))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        query = (
            f"SELECT * FROM goal_knowledge{where} "
            "ORDER BY updated_ts DESC, item_id ASC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            item = dict(r)
            item["content"] = json.loads(item.pop("content_json"))
            out.append(item)
        return out

    # ------------------------------------------------------ Decision Requests

    @staticmethod
    def _row_to_decision_request(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        payload = json.loads(item.pop("payload_json"))
        resolution = json.loads(item.pop("resolution_json"))
        raw_role = payload.get("role")
        if isinstance(raw_role, str) and raw_role.strip():
            derived_role = raw_role.strip()
        else:
            derived_role = (
                ScientificRole.RESEARCH_DIRECTOR.value
                if str(item.get("tier") or "").upper() == "L2"
                else ScientificRole.RESEARCH_LEAD.value
            )
        item["role"] = derived_role
        item["attempt"] = int(payload.get("attempt", 1))
        item["model_spec"] = str(payload.get("model_spec", ""))
        item["payload"] = payload
        item["resolution"] = resolution
        return item

    def enqueue_decision_request(
        self,
        goal_id: str,
        *,
        tier: str = "L1",
        role: ScientificRole | str | None = None,
        model_spec: str = "",
        attempt: int = 1,
        trigger_kind: str = "EVIDENCE_UPDATE",
        goal_version: int = 1,
        human_epoch: int = 0,
        coalesce_key: str = "",
        coalesce_window_s: float = 0.0,
        payload: Mapping[str, Any] | None = None,
        req_id: str | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Enqueue a per-Goal decision request, optionally coalescing pending requests."""
        ts = time.time() if now is None else float(now)
        avail_ts = ts + max(0.0, float(coalesce_window_s))
        body = dict(payload or {})
        if role is not None:
            eff_role = role.value if isinstance(role, ScientificRole) else str(role).strip()
            body["role"] = eff_role
        elif "role" in body and isinstance(body["role"], ScientificRole):
            body["role"] = body["role"].value
        eff_role_col = str(body.get("role") or "")
        if model_spec:
            body["model_spec"] = str(model_spec)
        eff_model_col = str(body.get("model_spec") or "")
        body["attempt"] = int(body.get("attempt", attempt))
        eff_attempt = int(body["attempt"])

        with self._tx() as (conn, pending):
            if coalesce_key:
                existing = conn.execute(
                    """
                    SELECT * FROM decision_requests
                    WHERE goal_id = ? AND coalesce_key = ? AND status = 'PENDING'
                    ORDER BY created_ts DESC LIMIT 1
                    """,
                    (goal_id, coalesce_key),
                ).fetchone()
                if existing is not None:
                    merged_payload = {
                        **json.loads(existing["payload_json"]),
                        **body,
                    }
                    payload_json = json.dumps(
                        merged_payload, sort_keys=True, ensure_ascii=False
                    )
                    conn.execute(
                        """
                        UPDATE decision_requests
                        SET tier = ?, role = ?, model_spec = ?, attempt = ?,
                            goal_version = ?, human_epoch = ?,
                            payload_json = ?, updated_ts = ?
                        WHERE req_id = ?
                        """,
                        (
                            tier,
                            str(merged_payload.get("role") or ""),
                            str(merged_payload.get("model_spec") or ""),
                            int(merged_payload.get("attempt", 1)),
                            int(goal_version),
                            int(human_epoch),
                            payload_json,
                            ts,
                            existing["req_id"],
                        ),
                    )
                    row = conn.execute(
                        "SELECT * FROM decision_requests WHERE req_id = ?",
                        (existing["req_id"],),
                    ).fetchone()
                    self._record_event(
                        conn,
                        pending,
                        "DECISION_REQUEST_COALESCED",
                        goal_id=goal_id,
                        payload={
                            "req_id": str(existing["req_id"]),
                            "tier": tier,
                            "role": str(merged_payload.get("role") or ""),
                            "coalesce_key": coalesce_key,
                        },
                        ts=ts,
                    )
                    return self._row_to_decision_request(row)

            eff_id = req_id or f"dreq_{uuid.uuid4().hex[:12]}"
            payload_json = json.dumps(body, sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO decision_requests (
                    req_id, goal_id, tier, role, model_spec, attempt,
                    trigger_kind, status,
                    goal_version, human_epoch, coalesce_key, available_ts,
                    payload_json, resolution_json, hq_job_id,
                    created_ts, updated_ts, resolved_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?, ?, '{}', NULL, ?, ?, NULL)
                """,
                (
                    eff_id,
                    goal_id,
                    tier,
                    eff_role_col,
                    eff_model_col,
                    eff_attempt,
                    trigger_kind,
                    int(goal_version),
                    int(human_epoch),
                    coalesce_key,
                    avail_ts,
                    payload_json,
                    ts,
                    ts,
                ),
            )
            self._record_event(
                conn,
                pending,
                "DECISION_REQUEST_ENQUEUED",
                goal_id=goal_id,
                payload={
                    "req_id": eff_id,
                    "tier": tier,
                    "role": eff_role_col,
                    "attempt": eff_attempt,
                    "trigger_kind": trigger_kind,
                    "goal_version": int(goal_version),
                    "human_epoch": int(human_epoch),
                    "available_ts": avail_ts,
                },
                ts=ts,
            )
            row = conn.execute(
                "SELECT * FROM decision_requests WHERE req_id = ?", (eff_id,)
            ).fetchone()
            return self._row_to_decision_request(row)

    def get_decision_request(self, req_id: str) -> dict[str, Any] | None:
        """Fetch a decision request by req_id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM decision_requests WHERE req_id = ?", (req_id,)
            ).fetchone()
        return None if row is None else self._row_to_decision_request(row)

    def list_decision_requests(
        self,
        goal_id: str | None = None,
        *,
        status: str | None = None,
        tier: str | None = None,
        role: ScientificRole | str | None = None,
        ready_before_ts: float | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Query decision requests with optional filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if tier is not None:
            clauses.append("tier = ?")
            params.append(tier)
        if ready_before_ts is not None:
            clauses.append("available_ts <= ?")
            params.append(float(ready_before_ts))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(int(limit))
        query = (
            f"SELECT * FROM decision_requests{where} "
            "ORDER BY available_ts ASC, created_ts ASC, req_id ASC LIMIT ?"
        )
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        items = [self._row_to_decision_request(r) for r in rows]
        if role is not None:
            target_role = (
                role.value if isinstance(role, ScientificRole) else str(role).strip()
            )
            items = [it for it in items if it.get("role") == target_role]
        return items

    def resolve_decision_request(
        self,
        req_id: str,
        *,
        status: str = "RESOLVED",
        resolution: Mapping[str, Any] | None = None,
        hq_job_id: int | None = None,
        attempt: int | None = None,
        model_spec: str | None = None,
        role: ScientificRole | str | None = None,
        expected_statuses: Sequence[str] | str | None = None,
        worker_self_exit: bool = False,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Transition a decision request (e.g. RUNNING, RESOLVED, DISCARDED_BY_NEWER_HUMAN_INTERVENTION)."""
        ts = time.time() if now is None else float(now)
        expected = (
            None
            if expected_statuses is None
            else (
                {expected_statuses}
                if isinstance(expected_statuses, str)
                else set(expected_statuses)
            )
        )
        terminal = status not in ("PENDING", "SUBMITTING", "RUNNING")
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT * FROM decision_requests WHERE req_id = ?", (req_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown req_id: {req_id}")
            cur_status = str(row["status"])
            if expected is not None and cur_status not in expected:
                raise StateTransitionError(
                    f"DecisionRequest {req_id} status is {cur_status!r}, expected one of {sorted(expected)}"
                )
            cur_payload = json.loads(row["payload_json"])
            if attempt is not None:
                cur_payload["attempt"] = int(attempt)
            if model_spec is not None:
                cur_payload["model_spec"] = str(model_spec)
            if role is not None:
                cur_payload["role"] = (
                    role.value if isinstance(role, ScientificRole) else str(role).strip()
                )
            payload_json = json.dumps(cur_payload, sort_keys=True, ensure_ascii=False)
            res_body = (
                dict(resolution)
                if resolution is not None
                else json.loads(row["resolution_json"])
            )
            res_json = json.dumps(res_body, sort_keys=True, ensure_ascii=False)
            eff_hq = hq_job_id if hq_job_id is not None else row["hq_job_id"]
            resolved_ts = ts if terminal else row["resolved_ts"]
            conn.execute(
                """
                UPDATE decision_requests
                SET status = ?, role = ?, model_spec = ?, attempt = ?,
                    payload_json = ?, resolution_json = ?, hq_job_id = ?,
                    updated_ts = ?, resolved_ts = ?
                WHERE req_id = ?
                """,
                (
                    status,
                    str(cur_payload.get("role") or ""),
                    str(cur_payload.get("model_spec") or ""),
                    int(cur_payload.get("attempt", 1)),
                    payload_json,
                    res_json,
                    eff_hq,
                    ts,
                    resolved_ts,
                    req_id,
                ),
            )
            if terminal:
                d_att = int(cur_payload.get("attempt", 1))
                eff_ukey = str(
                    cur_payload.get("usage_key")
                    or (
                        f"decision:{req_id}:a{d_att}:{cur_payload['submit_nonce']}"
                        if cur_payload.get("submit_nonce")
                        else f"decision:{req_id}:a{d_att}"
                    )
                )
                in_t = int(res_body.get("in_tokens", 0) or 0)
                out_t = int(res_body.get("out_tokens", 0) or 0)
                c_t = int(res_body.get("cached_input_tokens", res_body.get("cached_in_tokens", 0)) or 0)
                if in_t + out_t > 0 or status == "RESOLVED":
                    self.settle_goal_budget(
                        str(row["goal_id"]),
                        eff_ukey,
                        entity_kind="decision_request",
                        entity_id=req_id,
                        attempt=d_att,
                        model=str(res_body.get("model_used") or cur_payload.get("model_spec") or row["model_spec"] or ""),
                        input_tokens=in_t,
                        output_tokens=out_t,
                        cached_input_tokens=c_t,
                        state="SETTLED",
                    )
                else:
                    is_logical_cancel = status in (
                        "CANCELLED",
                        "SUPERSEDED",
                        "DISCARDED_BY_NEWER_HUMAN_INTERVENTION",
                    )
                    was_hq_submitted_or_submitting = (
                        (eff_hq is not None)
                        or (cur_status in ("SUBMITTING", "RUNNING"))
                        or bool(cur_payload.get("submit_nonce"))
                    )
                    has_unconf = self._has_unconfirmed_hq_cancel_in_conn(
                        conn, "decision", req_id, d_att, usage_key=eff_ukey
                    )
                    if not has_unconf and (
                        bool(worker_self_exit)
                        or not (is_logical_cancel and was_hq_submitted_or_submitting)
                    ):
                        self.release_goal_budget_reservation(
                            str(row["goal_id"]),
                            eff_ukey,
                            reason=f"decision_terminal:{status}",
                            now=ts,
                        )
                        if eff_ukey != f"decision:{req_id}:a{d_att}":
                            self.release_goal_budget_reservation(
                                str(row["goal_id"]),
                                f"decision:{req_id}:a{d_att}",
                                reason=f"decision_terminal:{status}",
                                now=ts,
                            )
            self._record_event(
                conn,
                pending,
                "DECISION_REQUEST_RESOLVED",
                goal_id=str(row["goal_id"]),
                payload={
                    "req_id": req_id,
                    "from": cur_status,
                    "to": status,
                    "attempt": int(cur_payload.get("attempt", 1)),
                    "hq_job_id": eff_hq,
                },
                ts=ts,
            )
            updated = conn.execute(
                "SELECT * FROM decision_requests WHERE req_id = ?", (req_id,)
            ).fetchone()
            return self._row_to_decision_request(updated)

    # ------------------------- V3 Plans, Reviews, RW Leases & Tx Fencing

    def save_research_plan(
        self,
        plan: ResearchPlan,
        *,
        supersede_previous: bool = True,
        now: float | None = None,
    ) -> ResearchPlan:
        """Persist a ResearchPlan, allocating `plan_version = MAX + 1` atomically on insert."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            existing = conn.execute(
                "SELECT goal_id, plan_version, created_ts FROM research_plans WHERE plan_id = ?",
                (plan.plan_id,),
            ).fetchone()
            if existing is not None:
                if str(existing["goal_id"]) != plan.goal_id:
                    raise ValueError(
                        f"Cross-goal ResearchPlan collision: plan_id {plan.plan_id!r} belongs to "
                        f"{existing['goal_id']!r}, not {plan.goal_id!r}"
                    )
                eff_version = int(existing["plan_version"])
                created_ts = float(existing["created_ts"])
            else:
                ver_row = conn.execute(
                    "SELECT COALESCE(MAX(plan_version), 0) + 1 AS next_v FROM research_plans WHERE goal_id = ?",
                    (plan.goal_id,),
                ).fetchone()
                eff_version = int(ver_row["next_v"])
                created_ts = plan.created_ts if plan.created_ts > 0 else ts

            if supersede_previous and plan.status == "APPROVED":
                prev_rows = conn.execute(
                    "SELECT plan_id, payload_json FROM research_plans WHERE goal_id = ? AND status = 'APPROVED' AND plan_id != ?",
                    (plan.goal_id, plan.plan_id),
                ).fetchall()
                for pr in prev_rows:
                    prev_obj = ResearchPlan.from_dict(json.loads(pr["payload_json"]))
                    prev_obj.status = "SUPERSEDED"
                    prev_obj.updated_ts = ts
                    conn.execute(
                        "UPDATE research_plans SET status = 'SUPERSEDED', payload_json = ?, updated_ts = ? WHERE plan_id = ?",
                        (
                            json.dumps(prev_obj.to_dict(), sort_keys=True, ensure_ascii=False),
                            ts,
                            pr["plan_id"],
                        ),
                    )

            stored = ResearchPlan.from_dict(
                {
                    **plan.to_dict(),
                    "plan_version": eff_version,
                    "created_ts": created_ts,
                    "updated_ts": ts,
                }
            )
            payload_json = json.dumps(stored.to_dict(), sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO research_plans (
                    plan_id, goal_id, plan_version, goal_version, human_epoch,
                    status, payload_json, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(plan_id) DO UPDATE SET
                    goal_version=excluded.goal_version,
                    human_epoch=excluded.human_epoch,
                    status=excluded.status,
                    payload_json=excluded.payload_json,
                    updated_ts=excluded.updated_ts
                """,
                (
                    stored.plan_id,
                    stored.goal_id,
                    int(stored.plan_version),
                    int(stored.goal_version),
                    int(stored.human_epoch),
                    stored.status,
                    payload_json,
                    created_ts,
                    ts,
                ),
            )
            self._record_event(
                conn,
                pending,
                "RESEARCH_PLAN_SAVED",
                goal_id=stored.goal_id,
                payload={
                    "plan_id": stored.plan_id,
                    "plan_version": int(stored.plan_version),
                    "status": stored.status,
                },
                ts=ts,
            )
        return stored

    def get_research_plan(self, plan_id: str) -> ResearchPlan | None:
        """Fetch a ResearchPlan by plan_id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM research_plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
        return None if row is None else ResearchPlan.from_dict(json.loads(row["payload_json"]))

    def get_latest_research_plan(
        self, goal_id: str, *, status: str | None = None
    ) -> ResearchPlan | None:
        """Fetch the highest-version ResearchPlan for a Goal."""
        query = "SELECT payload_json FROM research_plans WHERE goal_id = ?"
        params: list[Any] = [goal_id]
        if status is not None:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY plan_version DESC LIMIT 1"
        with self._lock:
            row = self._conn.execute(query, params).fetchone()
        return None if row is None else ResearchPlan.from_dict(json.loads(row["payload_json"]))

    def list_research_plans(self, goal_id: str) -> list[ResearchPlan]:
        """List all ResearchPlans for a Goal ordered by descending plan_version."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT payload_json FROM research_plans WHERE goal_id = ? ORDER BY plan_version DESC",
                (goal_id,),
            ).fetchall()
        return [ResearchPlan.from_dict(json.loads(r["payload_json"])) for r in rows]

    def save_experiment_plan(
        self,
        plan: ExperimentPlan,
        *,
        link_hypothesis: bool = True,
        now: float | None = None,
    ) -> ExperimentPlan:
        """Persist an ExperimentPlan, allocating `plan_version = MAX + 1` atomically on insert."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            hyp_owner = conn.execute(
                "SELECT goal_id FROM hypotheses WHERE hypothesis_id = ?",
                (plan.hypothesis_id,),
            ).fetchone()
            if hyp_owner is not None and str(hyp_owner["goal_id"]) != plan.goal_id:
                raise ValueError(
                    f"Cross-goal ExperimentPlan ownership violation: hypothesis {plan.hypothesis_id!r} belongs to "
                    f"{hyp_owner['goal_id']!r}, not {plan.goal_id!r}"
                )
            existing = conn.execute(
                "SELECT goal_id, hypothesis_id, plan_version, created_ts FROM experiment_plans WHERE exp_plan_id = ?",
                (plan.exp_plan_id,),
            ).fetchone()
            if existing is not None:
                if (
                    str(existing["goal_id"]) != plan.goal_id
                    or str(existing["hypothesis_id"]) != plan.hypothesis_id
                ):
                    raise ValueError(
                        f"Cross-goal/hypothesis ExperimentPlan collision: exp_plan_id {plan.exp_plan_id!r} belongs to "
                        f"({existing['goal_id']!r}, {existing['hypothesis_id']!r}), not ({plan.goal_id!r}, {plan.hypothesis_id!r})"
                    )
                eff_version = int(existing["plan_version"])
                created_ts = float(existing["created_ts"])
            else:
                ver_row = conn.execute(
                    "SELECT COALESCE(MAX(plan_version), 0) + 1 AS next_v FROM experiment_plans WHERE hypothesis_id = ?",
                    (plan.hypothesis_id,),
                ).fetchone()
                eff_version = int(ver_row["next_v"])
                created_ts = plan.created_ts if plan.created_ts > 0 else ts

            stored = ExperimentPlan.from_dict(
                {
                    **plan.to_dict(),
                    "plan_version": eff_version,
                    "created_ts": created_ts,
                    "updated_ts": ts,
                }
            )
            payload_json = json.dumps(stored.to_dict(), sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO experiment_plans (
                    exp_plan_id, goal_id, hypothesis_id, plan_version, goal_version,
                    human_epoch, plan_epoch, review_status, payload_json, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(exp_plan_id) DO UPDATE SET
                    goal_version=excluded.goal_version,
                    human_epoch=excluded.human_epoch,
                    plan_epoch=excluded.plan_epoch,
                    review_status=excluded.review_status,
                    payload_json=excluded.payload_json,
                    updated_ts=excluded.updated_ts
                """,
                (
                    stored.exp_plan_id,
                    stored.goal_id,
                    stored.hypothesis_id,
                    int(stored.plan_version),
                    int(stored.goal_version),
                    int(stored.human_epoch),
                    int(getattr(stored, "plan_epoch", 0) or 0),
                    stored.review_status,
                    payload_json,
                    created_ts,
                    ts,
                ),
            )
            if link_hypothesis:
                hyp_row = conn.execute(
                    "SELECT payload_json FROM hypotheses WHERE hypothesis_id = ?",
                    (stored.hypothesis_id,),
                ).fetchone()
                if hyp_row is not None:
                    hyp_node = HypothesisNode.from_dict(json.loads(hyp_row["payload_json"]))
                    hyp_node.exp_plan_id = stored.exp_plan_id
                    conn.execute(
                        "UPDATE hypotheses SET payload_json = ?, updated_ts = ? WHERE hypothesis_id = ?",
                        (
                            json.dumps(hyp_node.to_dict(), sort_keys=True, ensure_ascii=False),
                            ts,
                            stored.hypothesis_id,
                        ),
                    )
            self._record_event(
                conn,
                pending,
                "EXPERIMENT_PLAN_SAVED",
                goal_id=stored.goal_id,
                hypothesis_id=stored.hypothesis_id,
                payload={
                    "exp_plan_id": stored.exp_plan_id,
                    "plan_version": int(stored.plan_version),
                    "review_status": stored.review_status,
                },
                ts=ts,
            )
        return stored

    def get_experiment_plan(self, exp_plan_id: str) -> ExperimentPlan | None:
        """Fetch an ExperimentPlan by exp_plan_id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM experiment_plans WHERE exp_plan_id = ?",
                (exp_plan_id,),
            ).fetchone()
        return None if row is None else ExperimentPlan.from_dict(json.loads(row["payload_json"]))

    def get_latest_experiment_plan(self, hypothesis_id: str) -> ExperimentPlan | None:
        """Fetch the highest-version ExperimentPlan for a Hypothesis."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM experiment_plans WHERE hypothesis_id = ? ORDER BY plan_version DESC LIMIT 1",
                (hypothesis_id,),
            ).fetchone()
        return None if row is None else ExperimentPlan.from_dict(json.loads(row["payload_json"]))

    def list_experiment_plans(
        self,
        hypothesis_id: str | None = None,
        *,
        goal_id: str | None = None,
    ) -> list[ExperimentPlan]:
        """List ExperimentPlans filtered by hypothesis_id and/or goal_id."""
        clauses: list[str] = []
        params: list[Any] = []
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT payload_json FROM experiment_plans{where} ORDER BY plan_version DESC, created_ts DESC",
                params,
            ).fetchall()
        return [ExperimentPlan.from_dict(json.loads(r["payload_json"])) for r in rows]

    def save_execution_plan(
        self, plan: ExecutionPlan, *, now: float | None = None
    ) -> ExecutionPlan:
        """Create or update an ExecutionPlan keyed by workflow_id / exec_plan_id."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            hyp_owner = conn.execute(
                "SELECT goal_id FROM hypotheses WHERE hypothesis_id = ?",
                (plan.hypothesis_id,),
            ).fetchone()
            if hyp_owner is not None and str(hyp_owner["goal_id"]) != plan.goal_id:
                raise ValueError(
                    f"Cross-goal ExecutionPlan ownership violation: hypothesis {plan.hypothesis_id!r} belongs to "
                    f"{hyp_owner['goal_id']!r}, not {plan.goal_id!r}"
                )
            if plan.workflow_id:
                wf_owner = conn.execute(
                    "SELECT goal_id, hypothesis_id FROM workflows WHERE workflow_id = ?",
                    (plan.workflow_id,),
                ).fetchone()
                if wf_owner is not None and (
                    str(wf_owner["goal_id"]) != plan.goal_id
                    or str(wf_owner["hypothesis_id"]) != plan.hypothesis_id
                ):
                    raise ValueError(
                        f"Cross-goal/hypothesis ExecutionPlan workflow mismatch: workflow {plan.workflow_id!r} belongs to "
                        f"({wf_owner['goal_id']!r}, {wf_owner['hypothesis_id']!r}), not ({plan.goal_id!r}, {plan.hypothesis_id!r})"
                    )
            by_id = conn.execute(
                "SELECT exec_plan_id, workflow_id, goal_id, hypothesis_id, created_ts FROM execution_plans WHERE exec_plan_id = ?",
                (plan.exec_plan_id,),
            ).fetchone()
            by_wf = (
                conn.execute(
                    "SELECT exec_plan_id, workflow_id, goal_id, hypothesis_id, created_ts FROM execution_plans WHERE workflow_id = ?",
                    (plan.workflow_id,),
                ).fetchone()
                if plan.workflow_id
                else None
            )
            if by_id is not None:
                if (
                    str(by_id["goal_id"]) != plan.goal_id
                    or str(by_id["hypothesis_id"]) != plan.hypothesis_id
                ):
                    raise ValueError(
                        f"Cross-goal/hypothesis ExecutionPlan collision: {by_id['exec_plan_id']!r} belongs to "
                        f"({by_id['goal_id']!r}, {by_id['hypothesis_id']!r}), not ({plan.goal_id!r}, {plan.hypothesis_id!r})"
                    )
            if by_wf is not None:
                if (
                    str(by_wf["goal_id"]) != plan.goal_id
                    or str(by_wf["hypothesis_id"]) != plan.hypothesis_id
                ):
                    raise ValueError(
                        f"Cross-goal/hypothesis ExecutionPlan workflow collision: {by_wf['exec_plan_id']!r} belongs to "
                        f"({by_wf['goal_id']!r}, {by_wf['hypothesis_id']!r}), not ({plan.goal_id!r}, {plan.hypothesis_id!r})"
                    )
            if (
                by_id is not None
                and by_wf is not None
                and str(by_id["exec_plan_id"]) != str(by_wf["exec_plan_id"])
            ):
                raise ValueError(
                    f"Ambiguous dual-key ExecutionPlan collision: exec_plan_id={plan.exec_plan_id!r} matches "
                    f"{by_id['exec_plan_id']!r} while workflow_id={plan.workflow_id!r} matches {by_wf['exec_plan_id']!r}"
                )
            if by_id is not None and str(by_id["workflow_id"] or "") != str(plan.workflow_id or ""):
                raise ValueError(
                    f"ExecutionPlan workflow_id mismatch for {by_id['exec_plan_id']!r}: existing workflow_id is "
                    f"{by_id['workflow_id']!r}, got {plan.workflow_id!r}"
                )
            existing = by_id if by_id is not None else by_wf
            eff_id = str(existing["exec_plan_id"]) if existing is not None else plan.exec_plan_id
            created_ts = (
                float(existing["created_ts"])
                if existing is not None
                else (plan.created_ts if plan.created_ts > 0 else ts)
            )
            stored = ExecutionPlan.from_dict(
                {
                    **plan.to_dict(),
                    "exec_plan_id": eff_id,
                    "created_ts": created_ts,
                    "updated_ts": ts,
                }
            )
            payload_json = json.dumps(stored.to_dict(), sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO execution_plans (
                    exec_plan_id, workflow_id, goal_id, hypothesis_id,
                    goal_version, human_epoch, primitive, review_tier,
                    payload_json, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(exec_plan_id) DO UPDATE SET
                    goal_version=excluded.goal_version,
                    human_epoch=excluded.human_epoch,
                    primitive=excluded.primitive,
                    review_tier=excluded.review_tier,
                    payload_json=excluded.payload_json,
                    updated_ts=excluded.updated_ts
                """,
                (
                    stored.exec_plan_id,
                    stored.workflow_id,
                    stored.goal_id,
                    stored.hypothesis_id,
                    int(stored.goal_version),
                    int(stored.human_epoch),
                    stored.primitive,
                    stored.review_tier.value,
                    payload_json,
                    created_ts,
                    ts,
                ),
            )
            self._record_event(
                conn,
                pending,
                "EXECUTION_PLAN_SAVED",
                goal_id=stored.goal_id,
                hypothesis_id=stored.hypothesis_id,
                payload={
                    "exec_plan_id": stored.exec_plan_id,
                    "workflow_id": stored.workflow_id,
                    "primitive": stored.primitive,
                    "review_tier": stored.review_tier.value,
                },
                ts=ts,
            )
        return stored

    def get_execution_plan(self, exec_plan_id: str) -> ExecutionPlan | None:
        """Fetch an ExecutionPlan by exec_plan_id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM execution_plans WHERE exec_plan_id = ?",
                (exec_plan_id,),
            ).fetchone()
        return None if row is None else ExecutionPlan.from_dict(json.loads(row["payload_json"]))

    def get_execution_plan_by_workflow(self, workflow_id: str) -> ExecutionPlan | None:
        """Fetch an ExecutionPlan by workflow_id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM execution_plans WHERE workflow_id = ?",
                (workflow_id,),
            ).fetchone()
        return None if row is None else ExecutionPlan.from_dict(json.loads(row["payload_json"]))

    def list_execution_plans(
        self,
        goal_id: str | None = None,
        *,
        hypothesis_id: str | None = None,
    ) -> list[ExecutionPlan]:
        """List ExecutionPlans filtered by goal_id and/or hypothesis_id."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT payload_json FROM execution_plans{where} ORDER BY created_ts DESC",
                params,
            ).fetchall()
        return [ExecutionPlan.from_dict(json.loads(r["payload_json"])) for r in rows]

    def save_review_receipt(
        self, receipt: ReviewReceipt, *, now: float | None = None
    ) -> ReviewReceipt:
        """Idempotently persist or update an epoch-bound ReviewReceipt in `review_records`."""
        ts = time.time() if now is None else float(now)
        created_ts = receipt.created_ts if receipt.created_ts > 0 else ts
        stored = ReviewReceipt.from_dict({**receipt.to_dict(), "created_ts": created_ts})
        with self._tx() as (conn, pending):
            existing = conn.execute(
                """
                SELECT review_id FROM review_records
                WHERE workflow_id = ? AND review_kind = ? AND plan_epoch = ?
                  AND human_epoch = ? AND target_version = ?
                  AND head_sha = ? AND slot_index = ?
                """,
                (
                    stored.workflow_id,
                    stored.review_kind.value,
                    int(stored.plan_epoch),
                    int(stored.human_epoch),
                    int(stored.target_version),
                    stored.head_sha,
                    int(stored.slot_index),
                ),
            ).fetchone()
            if existing is not None and str(existing["review_id"]) != stored.review_id:
                stored = ReviewReceipt.from_dict(
                    {**stored.to_dict(), "review_id": str(existing["review_id"])}
                )
            payload_json = json.dumps(stored.to_dict(), sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                INSERT INTO review_records (
                    review_id, goal_id, hypothesis_id, workflow_id, task_id,
                    review_kind, review_tier, target_version, plan_epoch, human_epoch, target_kind,
                    head_sha, slot_index,
                    verdict, process_status, severity, tiebreaker_used,
                    in_tokens, out_tokens, payload_json, created_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(review_id) DO UPDATE SET
                    task_id=excluded.task_id,
                    plan_epoch=excluded.plan_epoch,
                    human_epoch=excluded.human_epoch,
                    target_kind=excluded.target_kind,
                    verdict=excluded.verdict,
                    process_status=excluded.process_status,
                    severity=excluded.severity,
                    tiebreaker_used=excluded.tiebreaker_used,
                    in_tokens=excluded.in_tokens,
                    out_tokens=excluded.out_tokens,
                    payload_json=excluded.payload_json,
                    created_ts=excluded.created_ts
                """,
                (
                    stored.review_id,
                    stored.goal_id,
                    stored.hypothesis_id,
                    stored.workflow_id,
                    stored.task_id,
                    stored.review_kind.value,
                    stored.review_tier.value,
                    int(stored.target_version),
                    int(stored.plan_epoch),
                    int(stored.human_epoch),
                    str(stored.target_kind or "hypothesis"),
                    stored.head_sha,
                    int(stored.slot_index),
                    stored.verdict.value,
                    stored.process_status.value,
                    stored.severity,
                    1 if stored.tiebreaker_used else 0,
                    int(stored.in_tokens),
                    int(stored.out_tokens),
                    payload_json,
                    float(stored.created_ts),
                ),
            )
            self._record_event(
                conn,
                pending,
                "REVIEW_RECEIPT_RECORDED",
                goal_id=stored.goal_id,
                hypothesis_id=stored.hypothesis_id,
                task_id=stored.task_id or None,
                payload={
                    "review_id": stored.review_id,
                    "review_kind": stored.review_kind.value,
                    "review_tier": stored.review_tier.value,
                    "plan_epoch": int(stored.plan_epoch),
                    "human_epoch": int(stored.human_epoch),
                    "target_kind": str(stored.target_kind or "hypothesis"),
                    "slot_index": int(stored.slot_index),
                    "verdict": stored.verdict.value,
                    "process_status": stored.process_status.value,
                },
                ts=ts,
            )
        return stored

    record_review_receipt = save_review_receipt

    def get_review_receipt(self, review_id: str) -> ReviewReceipt | None:
        """Fetch a ReviewReceipt by review_id."""
        with self._lock:
            row = self._conn.execute(
                "SELECT payload_json FROM review_records WHERE review_id = ?",
                (review_id,),
            ).fetchone()
        return None if row is None else ReviewReceipt.from_dict(json.loads(row["payload_json"]))

    def list_review_receipts(
        self,
        goal_id: str | None = None,
        *,
        hypothesis_id: str | None = None,
        workflow_id: str | None = None,
        review_kind: ReviewKind | str | None = None,
        target_version: int | None = None,
        plan_epoch: int | None = None,
        human_epoch: int | None = None,
        head_sha: str | None = None,
    ) -> list[ReviewReceipt]:
        """List ReviewReceipts matching optional quorum/hypothesis/epoch filters."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if workflow_id is not None:
            clauses.append("workflow_id = ?")
            params.append(workflow_id)
        if review_kind is not None:
            rk_val = (
                review_kind.value
                if isinstance(review_kind, ReviewKind)
                else ReviewKind(review_kind).value
            )
            clauses.append("review_kind = ?")
            params.append(rk_val)
        if target_version is not None:
            clauses.append("target_version = ?")
            params.append(int(target_version))
        if plan_epoch is not None:
            clauses.append("plan_epoch = ?")
            params.append(int(plan_epoch))
        if human_epoch is not None:
            clauses.append("human_epoch = ?")
            params.append(int(human_epoch))
        if head_sha is not None:
            clauses.append("head_sha = ?")
            params.append(head_sha)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT payload_json FROM review_records{where} ORDER BY slot_index ASC, created_ts ASC",
                params,
            ).fetchall()
        return [ReviewReceipt.from_dict(json.loads(r["payload_json"])) for r in rows]

    # --------------------- Shared RO vs Exclusive Writer Branch Leases

    def acquire_branch_ro_lease(
        self,
        goal_id: str,
        hypothesis_id: str,
        head_sha: str,
        task_id: str,
        holder: str,
        *,
        ttl_s: float = 300.0,
        max_concurrent: int = 4,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Acquire shared `branch_ro:{goal_id}:{hyp_id}:{head_sha}` lease iff no writer holds `branch:`."""
        if ttl_s <= 0 or max_concurrent <= 0:
            raise ValueError("ttl_s and max_concurrent must be positive")
        ts = time.time() if now is None else float(now)
        expires = ts + float(ttl_s)
        writer_key = f"branch:{goal_id}:{hypothesis_id}"
        ro_key = f"branch_ro:{goal_id}:{hypothesis_id}:{head_sha}"

        with self._tx() as (conn, pending):
            self._reap_expired_in_tx(conn, pending, now=ts, resource_key=writer_key)
            self._reap_expired_in_tx(conn, pending, now=ts, resource_key=ro_key)

            active_writer = conn.execute(
                "SELECT 1 FROM leases WHERE resource_key = ? AND status = 'ACTIVE' AND expires_ts > ? LIMIT 1",
                (writer_key, ts),
            ).fetchone()
            if active_writer is not None:
                return None

            existing = conn.execute(
                """
                SELECT * FROM leases
                WHERE resource_key = ? AND task_id = ? AND holder = ? AND status = 'ACTIVE'
                """,
                (ro_key, task_id, holder),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    "UPDATE leases SET renewed_ts = ?, expires_ts = ? WHERE lease_id = ?",
                    (ts, expires, existing["lease_id"]),
                )
                out = dict(existing)
                out["renewed_ts"] = ts
                out["expires_ts"] = expires
                return out

            used_row = conn.execute(
                """
                SELECT COALESCE(SUM(capacity), 0) AS used
                FROM leases
                WHERE resource_key = ? AND status = 'ACTIVE' AND expires_ts > ?
                """,
                (ro_key, ts),
            ).fetchone()
            if int(used_row["used"]) + 1 > int(max_concurrent):
                return None

            lease_id = f"lease_{uuid.uuid4().hex[:16]}"
            conn.execute(
                """
                INSERT INTO leases (
                    lease_id, resource_key, task_id, goal_id, holder, capacity,
                    acquired_ts, renewed_ts, expires_ts, released_ts, status
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, NULL, 'ACTIVE')
                """,
                (lease_id, ro_key, task_id, goal_id, holder, ts, ts, expires),
            )
            self._record_event(
                conn,
                pending,
                "LEASE_ACQUIRED",
                goal_id=goal_id or None,
                task_id=task_id,
                payload={
                    "lease_id": lease_id,
                    "resource_key": ro_key,
                    "holder": holder,
                    "capacity": 1,
                    "expires_ts": expires,
                },
                ts=ts,
            )
            return {
                "lease_id": lease_id,
                "resource_key": ro_key,
                "task_id": task_id,
                "goal_id": goal_id,
                "holder": holder,
                "capacity": 1,
                "acquired_ts": ts,
                "renewed_ts": ts,
                "expires_ts": expires,
                "released_ts": None,
                "status": "ACTIVE",
            }

    def acquire_branch_writer_lease(
        self,
        goal_id: str,
        hypothesis_id: str,
        task_id: str,
        holder: str,
        *,
        ttl_s: float = 300.0,
        now: float | None = None,
    ) -> dict[str, Any] | None:
        """Acquire exclusive `branch:{goal_id}:{hyp_id}` lease iff zero `branch_ro:` leases are active."""
        if ttl_s <= 0:
            raise ValueError("ttl_s must be positive")
        ts = time.time() if now is None else float(now)
        expires = ts + float(ttl_s)
        writer_key = f"branch:{goal_id}:{hypothesis_id}"
        ro_glob = (
            "branch_ro:"
            + self._escape_glob(goal_id)
            + ":"
            + self._escape_glob(hypothesis_id)
            + ":*"
        )

        with self._tx() as (conn, pending):
            self._reap_expired_in_tx(conn, pending, now=ts)

            active_ro = conn.execute(
                "SELECT 1 FROM leases WHERE resource_key GLOB ? AND status = 'ACTIVE' AND expires_ts > ? LIMIT 1",
                (ro_glob, ts),
            ).fetchone()
            if active_ro is not None:
                return None

            existing = conn.execute(
                """
                SELECT * FROM leases
                WHERE resource_key = ? AND task_id = ? AND holder = ? AND status = 'ACTIVE'
                """,
                (writer_key, task_id, holder),
            ).fetchone()
            if existing is not None:
                conn.execute(
                    "UPDATE leases SET renewed_ts = ?, expires_ts = ? WHERE lease_id = ?",
                    (ts, expires, existing["lease_id"]),
                )
                out = dict(existing)
                out["renewed_ts"] = ts
                out["expires_ts"] = expires
                return out

            active_writer = conn.execute(
                "SELECT 1 FROM leases WHERE resource_key = ? AND status = 'ACTIVE' AND expires_ts > ? LIMIT 1",
                (writer_key, ts),
            ).fetchone()
            if active_writer is not None:
                return None

            lease_id = f"lease_{uuid.uuid4().hex[:16]}"
            conn.execute(
                """
                INSERT INTO leases (
                    lease_id, resource_key, task_id, goal_id, holder, capacity,
                    acquired_ts, renewed_ts, expires_ts, released_ts, status
                ) VALUES (?, ?, ?, ?, ?, 1, ?, ?, ?, NULL, 'ACTIVE')
                """,
                (lease_id, writer_key, task_id, goal_id, holder, ts, ts, expires),
            )
            self._record_event(
                conn,
                pending,
                "LEASE_ACQUIRED",
                goal_id=goal_id or None,
                task_id=task_id,
                payload={
                    "lease_id": lease_id,
                    "resource_key": writer_key,
                    "holder": holder,
                    "capacity": 1,
                    "expires_ts": expires,
                },
                ts=ts,
            )
            return {
                "lease_id": lease_id,
                "resource_key": writer_key,
                "task_id": task_id,
                "goal_id": goal_id,
                "holder": holder,
                "capacity": 1,
                "acquired_ts": ts,
                "renewed_ts": ts,
                "expires_ts": expires,
                "released_ts": None,
                "status": "ACTIVE",
            }

    @staticmethod
    def _escape_glob(token: str) -> str:
        """Escape SQLite GLOB metacharacters (`[`, `*`, `?`) so `_` and all ID chars match literally."""
        out: list[str] = []
        for ch in str(token):
            if ch in ("[", "*", "?"):
                out.append(f"[{ch}]")
            else:
                out.append(ch)
        return "".join(out)

    def has_active_branch_ro_leases(
        self, goal_id: str, hypothesis_id: str, *, now: float | None = None
    ) -> bool:
        """Return True if any shared `branch_ro:{goal_id}:{hyp_id}:*` lease is active."""
        ts = time.time() if now is None else float(now)
        ro_glob = (
            "branch_ro:"
            + self._escape_glob(goal_id)
            + ":"
            + self._escape_glob(hypothesis_id)
            + ":*"
        )
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM leases WHERE resource_key GLOB ? AND status = 'ACTIVE' AND expires_ts > ? LIMIT 1",
                (ro_glob, ts),
            ).fetchone()
        return row is not None

    # ---------------- Atomic `(goal_version, human_epoch, attempt, lease_id)` Fencing

    def promotion_lock_path(self, goal_id: str, branch_name: str) -> Path:
        """Canonical cross-process `fcntl.flock` path serializing branch promotion vs reconciler/HITL cancel."""
        import re as _re_lock

        safe_g = _re_lock.sub(r"[^A-Za-z0-9._-]+", "_", str(goal_id or "default"))
        safe_b = _re_lock.sub(r"[^A-Za-z0-9._-]+", "_", str(branch_name or "default"))
        locks_dir = self.db_path.parent / "locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        return locks_dir / f"promote_{safe_g}_{safe_b}.flock"

    @contextlib.contextmanager
    def promotion_branch_lock(
        self, goal_id: str, branch_name: str, *, nonblocking: bool = False
    ) -> Iterator[bool]:
        """Acquire exclusive `promote_<safe_goal>_<safe_branch>.flock` (`LOCK_NB` when `nonblocking=True`)."""
        import fcntl as _fcntl

        lock_path = self.promotion_lock_path(goal_id, branch_name)
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        try:
            flags = _fcntl.LOCK_EX | (_fcntl.LOCK_NB if nonblocking else 0)
            try:
                _fcntl.flock(fd, flags)
                acquired = True
            except BlockingIOError:
                acquired = False
            yield acquired
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)

    @contextlib.contextmanager
    def barrier_authority_lock(self, *, nonblocking: bool = False) -> Iterator[bool]:
        """Acquire exclusive `<arr_home>/locks/barrier_authority.flock` serializing HQ drain vs TTL takeover."""
        import fcntl as _fcntl

        locks_dir = self.db_path.parent / "locks"
        locks_dir.mkdir(parents=True, exist_ok=True)
        lock_path = locks_dir / "barrier_authority.flock"
        fd = os.open(str(lock_path), os.O_CREAT | os.O_RDWR, 0o600)
        acquired = False
        try:
            flags = _fcntl.LOCK_EX | (_fcntl.LOCK_NB if nonblocking else 0)
            try:
                _fcntl.flock(fd, flags)
                acquired = True
            except BlockingIOError:
                acquired = False
            yield acquired
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    _fcntl.flock(fd, _fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                os.close(fd)

    def revoke_barrier_lease_in_conn(
        self,
        conn: sqlite3.Connection,
        lease_id: str,
        *,
        barrier_id: str = "",
        generation: int = 0,
        reason: str = "",
        now: float | None = None,
    ) -> None:
        """Persist permanent barrier lease revocation without list truncation."""
        clean_id = str(lease_id or "").strip()
        if not clean_id:
            return
        ts = time.time() if now is None else float(now)
        conn.execute(
            """
            INSERT OR REPLACE INTO revoked_barrier_leases (
                lease_id, barrier_id, generation, reason, revoked_ts
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (clean_id, str(barrier_id or ""), int(generation), str(reason or ""), ts),
        )

    def is_barrier_lease_revoked(self, lease_id: str) -> bool:
        """Return True if `lease_id` has been permanently revoked in `revoked_barrier_leases` or fences config."""
        clean_id = str(lease_id or "").strip()
        if not clean_id:
            return False
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM revoked_barrier_leases WHERE lease_id = ? LIMIT 1",
                (clean_id,),
            ).fetchone()
            if row is not None:
                return True
        fences = self.get_global_config("maintenance_barrier_fences") or {}
        if isinstance(fences, Mapping):
            return clean_id in {str(x) for x in (fences.get("revoked_lease_ids") or []) if str(x)}
        return False

    def list_revoked_barrier_leases(self) -> list[str]:
        """Return all permanently revoked barrier lease IDs in chronological order."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT lease_id FROM revoked_barrier_leases ORDER BY revoked_ts ASC, lease_id ASC"
            ).fetchall()
        return [str(r["lease_id"]) for r in rows]

    def abort_cancelled_task_promotions(
        self,
        goal_id: str,
        hypothesis_id: str,
        task_id: str,
        *,
        reason: str = "",
        now: float | None = None,
    ) -> int:
        """Serialize with any in-flight promotion lock and reconcile/abort leftover `PREPARED` intents for `task_id`."""
        import re as _re_prom
        import subprocess as _subp_prom

        hyp_obj = self.get_hypothesis(hypothesis_id) if hypothesis_id else None
        branch_name = hyp_obj.branch_name if hyp_obj is not None else ""
        ts = time.time() if now is None else float(now)
        handled = 0
        with self.promotion_branch_lock(
            goal_id, branch_name, nonblocking=bool(self._tx_depth > 0)
        ) as got_lock:
            if not got_lock:
                return 0
            rows = [
                r
                for r in self.list_promotion_intents(goal_id=goal_id, state="PREPARED")
                if str(r.get("task_id") or "") == str(task_id)
            ]
            for row in rows:
                intent_id = str(row["intent_id"])
                wt_raw = str(row.get("wt_path") or "")
                cand_sha = str(row.get("candidate_sha") or "")
                base_sha = str(row.get("expected_base_sha") or "")
                exp_branch = str(row.get("branch_name") or branch_name or "")
                att = int(row.get("attempt", 1) or 1)
                wt_p = pathlib.Path(wt_raw) if wt_raw else None
                git_st = evaluate_promotion_git_state(wt_p, exp_branch, base_sha, cand_sha)
                actual_head = str(git_st["actual_head"])
                actual_branch = str(git_st["actual_branch"])
                branch_ok = bool(git_st["branch_ok"])
                branch_confirmed_unpromoted = bool(git_st["branch_confirmed_unpromoted"])
                delete_refs = False
                if cand_sha and actual_head == cand_sha and branch_ok:
                    self.commit_promotion_intent(
                        intent_id,
                        candidate_sha=cand_sha,
                        update_hypothesis_head=True,
                        reason=f"roll_forward_before_cancel:{reason}",
                        now=ts,
                    )
                    delete_refs = True
                elif (bool(base_sha) and actual_head == base_sha and branch_ok) or branch_confirmed_unpromoted:
                    self.abort_promotion_intent(
                        intent_id,
                        reason=reason or "task_cancelled_before_ff",
                        now=ts,
                    )
                    delete_refs = True
                else:
                    self.mark_promotion_intent_unresolved(
                        intent_id,
                        reason=(
                            f"UNRESOLVED_FF_AMBIGUOUS:head={actual_head or 'unavailable'},"
                            f"branch={actual_branch or 'detached'},base={base_sha}"
                        )[:500],
                        now=ts,
                    )
                handled += 1
                if delete_refs and wt_p is not None and wt_p.is_dir():
                    sanitized_tid = _re_prom.sub(r"[^A-Za-z0-9._-]+", "_", str(task_id))
                    for ref_n in dict.fromkeys(
                        [
                            f"refs/arr-promote/{task_id}-a{att}",
                            f"refs/arr-promote/{sanitized_tid}-a{att}",
                        ]
                    ):
                        with contextlib.suppress(Exception):
                            _subp_prom.run(
                                ["git", "-C", str(wt_p), "update-ref", "-d", ref_n],
                                capture_output=True,
                                text=True,
                                timeout=10,
                                check=False,
                            )
        return handled

    def record_pending_hq_cancellation(
        self,
        *,
        hq_job_id: int,
        entity_kind: str,
        entity_id: str,
        goal_id: str,
        attempt: int = 1,
        hypothesis_id: str = "",
        workflow_id: str = "",
        epoch: int = 0,
        plan_epoch: int = 0,
        reason: str = "",
        state: str = "CANCEL_PENDING",
        usage_key: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        """Persist a durable `CANCEL_PENDING` HQ cancellation row."""
        ts = time.time() if now is None else float(now)
        eff_att = max(1, int(attempt or 1))
        eff_ukey = str(usage_key or "").strip()
        cid = f"hqc_{hq_job_id}_{entity_kind}_{entity_id}_a{eff_att}"
        with self._tx() as (conn, _pending):
            conn.execute(
                """
                INSERT INTO pending_hq_cancellations (
                    cancel_id, hq_job_id, entity_kind, entity_id, attempt, goal_id,
                    hypothesis_id, workflow_id, epoch, plan_epoch,
                    state, reason, created_ts, updated_ts, confirmed_ts, usage_key
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(hq_job_id, entity_kind, entity_id, attempt) DO UPDATE SET
                    epoch = excluded.epoch,
                    plan_epoch = excluded.plan_epoch,
                    usage_key = CASE
                        WHEN pending_hq_cancellations.usage_key != '' THEN pending_hq_cancellations.usage_key
                        ELSE excluded.usage_key
                    END,
                    state = CASE
                        WHEN pending_hq_cancellations.state = 'CANCEL_CONFIRMED' THEN 'CANCEL_CONFIRMED'
                        ELSE excluded.state
                    END,
                    reason = excluded.reason,
                    updated_ts = excluded.updated_ts
                """,
                (
                    cid,
                    int(hq_job_id),
                    str(entity_kind),
                    str(entity_id),
                    eff_att,
                    str(goal_id),
                    str(hypothesis_id or ""),
                    str(workflow_id or ""),
                    int(epoch),
                    int(plan_epoch),
                    str(state or "CANCEL_PENDING"),
                    str(reason or ""),
                    ts,
                    ts,
                    ts if state == "CANCEL_CONFIRMED" else None,
                    eff_ukey,
                ),
            )
            row = conn.execute(
                "SELECT * FROM pending_hq_cancellations WHERE hq_job_id = ? AND entity_kind = ? AND entity_id = ? AND attempt = ?",
                (int(hq_job_id), str(entity_kind), str(entity_id), eff_att),
            ).fetchone()
            return dict(row) if row is not None else {}

    def confirm_hq_cancellation(
        self,
        hq_job_id: int,
        *,
        entity_kind: str | None = None,
        entity_id: str | None = None,
        attempt: int | None = None,
        usage_key: str | None = None,
        now: float | None = None,
    ) -> int:
        """Mark matching `pending_hq_cancellations` rows `CANCEL_CONFIRMED` and release exact nonce/attempt reservations."""
        ts = time.time() if now is None else float(now)
        clauses = ["hq_job_id = ?", "state != 'CANCEL_CONFIRMED'"]
        filter_params: list[Any] = [int(hq_job_id)]
        if entity_kind is not None:
            clauses.append("entity_kind = ?")
            filter_params.append(str(entity_kind))
        if entity_id is not None:
            clauses.append("entity_id = ?")
            filter_params.append(str(entity_id))
        if attempt is not None:
            clauses.append("attempt = ?")
            filter_params.append(int(attempt))
        eff_arg_ukey = str(usage_key or "").strip()
        if eff_arg_ukey:
            clauses.append("(usage_key = ? OR usage_key = '' OR usage_key LIKE 'legacy_ambiguous:%')")
            filter_params.append(eff_arg_ukey)
        with self._tx() as (conn, _pending):
            rows_to_confirm = conn.execute(
                f"SELECT entity_kind, entity_id, attempt, usage_key FROM pending_hq_cancellations WHERE {' AND '.join(clauses)}",
                filter_params,
            ).fetchall()
            cur = conn.execute(
                f"""
                UPDATE pending_hq_cancellations
                SET state = 'CANCEL_CONFIRMED', updated_ts = ?, confirmed_ts = ?
                WHERE {' AND '.join(clauses)}
                """,
                [ts, ts, *filter_params],
            )
            for rc in rows_to_confirm:
                rc_ukey = str(rc["usage_key"] or "").strip()
                if rc_ukey and not rc_ukey.startswith("legacy_ambiguous:"):
                    conn.execute(
                        """
                        UPDATE token_usage_ledger
                        SET reserved_tokens = 0, state = 'RELEASED', updated_ts = ?
                        WHERE usage_key = ? AND state = 'RESERVED'
                        """,
                        (ts, rc_ukey),
                    )
                    continue
                if eff_arg_ukey:
                    # Exact caller-supplied usage_key only; never wildcard-release sibling nonces.
                    conn.execute(
                        """
                        UPDATE token_usage_ledger
                        SET reserved_tokens = 0, state = 'RELEASED', updated_ts = ?
                        WHERE usage_key = ? AND state = 'RESERVED'
                        """,
                        (ts, eff_arg_ukey),
                    )
                    continue
                rc_att = max(1, int(rc["attempt"] if rc["attempt"] is not None else 1))
                prefix = (
                    f"task:{rc['entity_id']}:a{rc_att}"
                    if str(rc["entity_kind"]) == "task"
                    else f"decision:{rc['entity_id']}:a{rc_att}"
                )
                # Legacy un-nonced key only (`exact =`), never `GLOB 'prefix:*'` onto nonce-bearing rows.
                conn.execute(
                    """
                    UPDATE token_usage_ledger
                    SET reserved_tokens = 0, state = 'RELEASED', updated_ts = ?
                    WHERE usage_key = ? AND state = 'RESERVED'
                    """,
                    (ts, prefix),
                )
            return int(cur.rowcount or 0)

    def verify_and_renew_task_leases(
        self,
        task_id: str,
        attempt: int,
        lease_owner_id: str = "",
        required_keys: Sequence[str] = (),
        *,
        ttl_sec: float | None = None,
        now: float | None = None,
    ) -> tuple[bool, str]:
        """Verify inside `db._tx()` that all required task leases remain ACTIVE and unexpired, optionally renewing TTL."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, _pending):
            t_row = conn.execute(
                "SELECT attempt, status FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if t_row is not None and int(t_row["attempt"]) != int(attempt):
                return False, f"STALE_ATTEMPT:expected={attempt},actual={t_row['attempt']}"

            matched_ids: list[str] = []
            for rkey in required_keys:
                clean_k = str(rkey or "").strip()
                if not clean_k:
                    continue
                l_rows = conn.execute(
                    """
                    SELECT lease_id, holder, expires_ts
                    FROM leases
                    WHERE resource_key = ? AND task_id = ? AND status = 'ACTIVE'
                    """,
                    (clean_k, task_id),
                ).fetchall()
                valid_row = None
                for lr in l_rows:
                    if float(lr["expires_ts"]) <= ts:
                        continue
                    if (
                        not lease_owner_id
                        or str(lr["holder"]) == str(lease_owner_id)
                        or clean_k.startswith("branch_ro:")
                    ):
                        valid_row = lr
                        break
                if valid_row is None:
                    return False, f"LEASE_EXPIRED_OR_LOST:{clean_k}"
                matched_ids.append(str(valid_row["lease_id"]))

            if not required_keys:
                all_rows = conn.execute(
                    "SELECT lease_id, resource_key, holder, status, expires_ts FROM leases WHERE task_id = ?",
                    (task_id,),
                ).fetchall()
                by_rk: dict[str, list[sqlite3.Row]] = {}
                for lr in all_rows:
                    by_rk.setdefault(str(lr["resource_key"]), []).append(lr)
                for rk, rk_rows in by_rk.items():
                    valid_rk = [
                        lr
                        for lr in rk_rows
                        if str(lr["status"]).upper() == "ACTIVE"
                        and float(lr["expires_ts"]) > ts
                        and (
                            not lease_owner_id
                            or str(lr["holder"]) == str(lease_owner_id)
                            or rk.startswith("branch_ro:")
                        )
                    ]
                    if not valid_rk:
                        return False, f"LEASE_EXPIRED_OR_LOST:{rk}"
                    best_lr = max(valid_rk, key=lambda r: float(r["expires_ts"]))
                    matched_ids.append(str(best_lr["lease_id"]))

            if ttl_sec is not None and float(ttl_sec) > 0 and matched_ids:
                new_exp = ts + float(ttl_sec)
                for lid in matched_ids:
                    conn.execute(
                        "UPDATE leases SET renewed_ts = ?, expires_ts = ? WHERE lease_id = ? AND status = 'ACTIVE'",
                        (ts, new_exp, lid),
                    )
            return True, "OK"

    def prepare_promotion_intent(
        self,
        intent_id: str | None = None,
        *,
        goal_id: str,
        hypothesis_id: str,
        task_id: str,
        attempt: int = 1,
        branch_name: str = "",
        wt_path: str = "",
        expected_base_sha: str = "",
        candidate_sha: str = "",
        goal_version: int = 1,
        human_epoch: int = 0,
        plan_epoch: int = 0,
        reason: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        """Atomically persist a `PREPARED` two-phase Git promotion intent before mutating `wt_path`."""
        ts = time.time() if now is None else float(now)
        eff_id = str(intent_id or f"prom:{task_id}:a{int(attempt)}")
        with self._tx() as (conn, _pending):
            existing = conn.execute(
                "SELECT * FROM promotion_intents WHERE intent_id = ?", (eff_id,)
            ).fetchone()
            if existing is not None and str(existing["state"]) in ("PREPARED", "COMMITTED"):
                if str(existing["candidate_sha"]) != str(candidate_sha):
                    raise StateTransitionError(
                        f"Promotion intent {eff_id!r} already exists in state={existing['state']!r} with candidate_sha={existing['candidate_sha']!r}"
                    )
                return dict(existing)
            conn.execute(
                """
                INSERT INTO promotion_intents (
                    intent_id, goal_id, hypothesis_id, task_id, attempt,
                    branch_name, wt_path, expected_base_sha, candidate_sha,
                    goal_version, human_epoch, plan_epoch,
                    state, reason, created_ts, updated_ts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'PREPARED', ?, ?, ?)
                ON CONFLICT(intent_id) DO UPDATE SET
                    expected_base_sha = excluded.expected_base_sha,
                    candidate_sha = excluded.candidate_sha,
                    goal_version = excluded.goal_version,
                    human_epoch = excluded.human_epoch,
                    plan_epoch = excluded.plan_epoch,
                    state = 'PREPARED',
                    reason = excluded.reason,
                    updated_ts = excluded.updated_ts
                """,
                (
                    eff_id,
                    str(goal_id),
                    str(hypothesis_id),
                    str(task_id),
                    int(attempt),
                    str(branch_name or ""),
                    str(wt_path or ""),
                    str(expected_base_sha or ""),
                    str(candidate_sha or ""),
                    int(goal_version),
                    int(human_epoch),
                    int(plan_epoch),
                    str(reason or ""),
                    ts,
                    ts,
                ),
            )
            row = conn.execute(
                "SELECT * FROM promotion_intents WHERE intent_id = ?", (eff_id,)
            ).fetchone()
            return dict(row) if row is not None else {}

    def commit_promotion_intent(
        self,
        intent_id: str,
        *,
        candidate_sha: str | None = None,
        update_hypothesis_head: bool = True,
        reason: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        """CAS-transition a `PREPARED` promotion intent to `COMMITTED` and advance `hypothesis.head_sha`."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, _pending):
            row = conn.execute(
                "SELECT * FROM promotion_intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown promotion intent_id: {intent_id}")
            if str(row["state"]) == "COMMITTED":
                return dict(row)
            if str(row["state"]) != "PREPARED":
                raise StateTransitionError(
                    f"Cannot commit promotion intent {intent_id!r} from state={row['state']!r}"
                )
            eff_sha = str(candidate_sha or row["candidate_sha"] or "")
            if update_hypothesis_head and eff_sha:
                h_row = conn.execute(
                    "SELECT payload_json FROM hypotheses WHERE hypothesis_id = ?",
                    (str(row["hypothesis_id"]),),
                ).fetchone()
                if h_row is not None:
                    h_node = HypothesisNode.from_dict(json.loads(h_row["payload_json"]))
                    h_node.head_sha = eff_sha
                    conn.execute(
                        "UPDATE hypotheses SET head_sha = ?, payload_json = ?, updated_ts = ? WHERE hypothesis_id = ?",
                        (
                            eff_sha,
                            json.dumps(h_node.to_dict(), sort_keys=True, ensure_ascii=False),
                            ts,
                            h_node.hypothesis_id,
                        ),
                    )
            cur = conn.execute(
                """
                UPDATE promotion_intents
                SET state = 'COMMITTED', candidate_sha = ?, reason = ?, updated_ts = ?
                WHERE intent_id = ? AND state = 'PREPARED'
                """,
                (eff_sha, str(reason or row["reason"] or ""), ts, intent_id),
            )
            if int(cur.rowcount or 0) == 0:
                raise StateTransitionError(f"CAS failed committing promotion intent {intent_id!r}")
            if eff_sha:
                row_task_id = str(row["task_id"] or "")
                row_att = max(1, int(row["attempt"] or 1))
                rcpt_rows = conn.execute(
                    "SELECT receipt_id, payload_json FROM evidence_receipts WHERE task_id = ?",
                    (row_task_id,),
                ).fetchall()
                for rr in rcpt_rows:
                    r_pl = json.loads(rr["payload_json"])
                    r_att_raw = r_pl.get("attempt")
                    if r_att_raw is not None and int(r_att_raw or 1) != row_att:
                        continue
                    if str(r_pl.get("head_sha") or "") != eff_sha:
                        if r_pl.get("head_sha"):
                            r_pl.setdefault("pre_reconcile_head_sha", str(r_pl["head_sha"]))
                        r_pl["head_sha"] = eff_sha
                        r_pl["reconciled_promotion_sha"] = eff_sha
                        conn.execute(
                            "UPDATE evidence_receipts SET payload_json = ? WHERE receipt_id = ?",
                            (json.dumps(r_pl, sort_keys=True, ensure_ascii=False), str(rr["receipt_id"])),
                        )
            updated = conn.execute(
                "SELECT * FROM promotion_intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            return dict(updated) if updated is not None else {}

    def mark_promotion_intent_unresolved(
        self,
        intent_id: str,
        *,
        reason: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        """Keep a `PREPARED` promotion intent durable with updated `UNRESOLVED_FF_AMBIGUOUS` reason."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, _pending):
            row = conn.execute(
                "SELECT * FROM promotion_intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown promotion intent_id: {intent_id}")
            if str(row["state"]) == "PREPARED":
                conn.execute(
                    "UPDATE promotion_intents SET reason = ?, updated_ts = ? WHERE intent_id = ? AND state = 'PREPARED'",
                    (str(reason or row["reason"] or ""), ts, intent_id),
                )
            updated = conn.execute(
                "SELECT * FROM promotion_intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            return dict(updated) if updated is not None else {}

    def abort_promotion_intent(
        self,
        intent_id: str,
        *,
        reason: str = "",
        now: float | None = None,
    ) -> dict[str, Any]:
        """CAS-transition a `PREPARED` promotion intent to `ABORTED`."""
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, _pending):
            row = conn.execute(
                "SELECT * FROM promotion_intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown promotion intent_id: {intent_id}")
            if str(row["state"]) == "ABORTED":
                return dict(row)
            if str(row["state"]) != "PREPARED":
                raise StateTransitionError(
                    f"Cannot abort promotion intent {intent_id!r} from state={row['state']!r}"
                )
            conn.execute(
                """
                UPDATE promotion_intents
                SET state = 'ABORTED', reason = ?, updated_ts = ?
                WHERE intent_id = ? AND state = 'PREPARED'
                """,
                (str(reason or ""), ts, intent_id),
            )
            updated = conn.execute(
                "SELECT * FROM promotion_intents WHERE intent_id = ?", (intent_id,)
            ).fetchone()
            return dict(updated) if updated is not None else {}

    def list_promotion_intents(
        self,
        *,
        state: str | None = None,
        goal_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Query `promotion_intents` rows filtered by state and/or goal_id."""
        clauses: list[str] = []
        params: list[Any] = []
        if state is not None:
            clauses.append("state = ?")
            params.append(str(state))
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(str(goal_id))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM promotion_intents{where} ORDER BY created_ts ASC",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def list_pending_hq_cancellations(
        self,
        goal_id: str | None = None,
        *,
        state: str | None = "CANCEL_PENDING",
        hypothesis_id: str | None = None,
        workflow_id: str | None = None,
        entity_id: str | None = None,
        entity_kind: str | None = None,
        attempt: int | None = None,
    ) -> list[dict[str, Any]]:
        """Query durable HQ cancellation rows."""
        clauses: list[str] = []
        params: list[Any] = []
        if goal_id is not None:
            clauses.append("goal_id = ?")
            params.append(goal_id)
        if state is not None:
            clauses.append("state = ?")
            params.append(state)
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(hypothesis_id)
        if workflow_id is not None:
            clauses.append("workflow_id = ?")
            params.append(workflow_id)
        if entity_id is not None:
            clauses.append("entity_id = ?")
            params.append(entity_id)
        if entity_kind is not None:
            clauses.append("entity_kind = ?")
            params.append(entity_kind)
        if attempt is not None:
            clauses.append("attempt = ?")
            params.append(int(attempt))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM pending_hq_cancellations{where} ORDER BY created_ts ASC",
                params,
            ).fetchall()
        return [dict(r) for r in rows]

    def has_unconfirmed_hq_cancellations(
        self,
        goal_id: str,
        *,
        hypothesis_id: str | None = None,
        workflow_id: str | None = None,
        entity_id: str | None = None,
    ) -> bool:
        """Return True if any `CANCEL_PENDING` HQ cancellation row remains unconfirmed."""
        return bool(
            self.list_pending_hq_cancellations(
                goal_id=goal_id,
                state="CANCEL_PENDING",
                hypothesis_id=hypothesis_id,
                workflow_id=workflow_id,
                entity_id=entity_id,
            )
        )

    @staticmethod
    def verify_goal_epoch_in_tx(
        conn: sqlite3.Connection,
        goal_id: str,
        *,
        expected_goal_version: int,
        expected_human_epoch: int,
        expected_plan_epoch: int | None = None,
    ) -> tuple[bool, str, ResearchGoal | None]:
        """Verify strict `(goal_version, human_epoch, plan_epoch)` equality inside an active `BEGIN IMMEDIATE` tx."""
        row = conn.execute(
            "SELECT payload_json FROM goals WHERE goal_id = ?", (goal_id,)
        ).fetchone()
        if row is None:
            return False, "UNKNOWN_GOAL", None
        fresh_goal = ResearchGoal.from_dict(json.loads(row["payload_json"]))
        if int(fresh_goal.version) != int(expected_goal_version):
            return False, "STALE_GOAL_VERSION", fresh_goal
        if int(fresh_goal.human_epoch) != int(expected_human_epoch):
            return False, "HUMAN_INVALIDATED", fresh_goal
        if expected_plan_epoch is not None and int(getattr(fresh_goal, "plan_epoch", 0) or 0) != int(expected_plan_epoch):
            return False, "SUPERSEDED_EPOCH", fresh_goal
        return True, "OK", fresh_goal

    @staticmethod
    def lease_matches_in_tx(
        conn: sqlite3.Connection,
        task_id: str,
        expected_lease_id: str | None,
        *,
        now: float,
        allow_none: bool = True,
    ) -> bool:
        """Verify inside an active `BEGIN IMMEDIATE` tx that `expected_lease_id` is still ACTIVE."""
        if expected_lease_id is None or not str(expected_lease_id).strip():
            return bool(allow_none)
        row = conn.execute(
            """
            SELECT 1 FROM leases
            WHERE lease_id = ? AND task_id = ? AND status = 'ACTIVE' AND expires_ts > ?
            LIMIT 1
            """,
            (str(expected_lease_id), task_id, now),
        ).fetchone()
        return row is not None

    def fenced_transition_task(
        self,
        task_id: str,
        expected_statuses: Sequence[str] | str,
        new_status: str,
        *,
        expected_goal_version: int,
        expected_human_epoch: int,
        expected_attempt: int,
        expected_lease_id: str | None = None,
        hq_job_id: int | None = None,
        worktree_path: str | None = None,
        started_ts: float | None = None,
        finished_ts: float | None = None,
        pinned_route: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> StageTask:
        """Atomically fence `(goal_version, human_epoch, attempt, lease_id)` and transition a task."""
        expected = (
            {expected_statuses}
            if isinstance(expected_statuses, str)
            else set(expected_statuses)
        )
        ts = time.time() if now is None else float(now)
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT status, payload_json FROM tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown task_id: {task_id}")
            task = StageTask.from_dict(json.loads(row["payload_json"]))
            ok, reason, _fresh_goal = self.verify_goal_epoch_in_tx(
                conn,
                task.goal_id,
                expected_goal_version=expected_goal_version,
                expected_human_epoch=expected_human_epoch,
            )
            if not ok:
                raise FencedStaleWriteError(
                    f"Task {task_id} failed goal epoch fence: {reason}",
                    reason_code=reason,
                )
            cur_status = str(row["status"])
            if cur_status not in expected:
                raise FencedStaleWriteError(
                    f"Task {task_id} status is {cur_status!r}, expected {sorted(expected)}",
                    reason_code="STATUS_MISMATCH",
                )
            if int(task.attempt) != int(expected_attempt):
                raise FencedStaleWriteError(
                    f"Task {task_id} attempt is {task.attempt}, expected {expected_attempt}",
                    reason_code="STALE_ATTEMPT",
                )
            if not self.lease_matches_in_tx(conn, task_id, expected_lease_id, now=ts):
                raise FencedStaleWriteError(
                    f"Task {task_id} lease {expected_lease_id!r} is missing or expired",
                    reason_code="LEASE_MISMATCH",
                )

            prev_hq_job_id = task.hq_job_id
            task.status = new_status
            if hq_job_id is not None:
                task.hq_job_id = int(hq_job_id)
            if worktree_path is not None:
                task.worktree_path = worktree_path
            if started_ts is not None:
                task.started_ts = float(started_ts)
            if finished_ts is not None:
                task.finished_ts = float(finished_ts)
            if pinned_route is not None:
                task.pinned_route = dict(pinned_route)
            task_dict = task.to_dict()
            payload_json = json.dumps(task_dict, sort_keys=True, ensure_ascii=False)
            conn.execute(
                """
                UPDATE tasks
                SET status = ?, hq_job_id = ?, attempt = ?, worktree_path = ?,
                    model_spec = ?, hq_llm_resource = ?,
                    started_ts = ?, finished_ts = ?, payload_json = ?, updated_ts = ?
                WHERE task_id = ? AND status = ?
                """,
                (
                    task.status,
                    task.hq_job_id,
                    task.attempt,
                    task.worktree_path,
                    str(task_dict.get("model_spec", "")),
                    str(task_dict.get("hq_llm_resource", "")),
                    task.started_ts,
                    task.finished_ts,
                    payload_json,
                    ts,
                    task_id,
                    cur_status,
                ),
            )
            if new_status in ("CANCELLED", "PRUNED", "FAILED", "HUMAN_INVALIDATED", "SUPERSEDED") or new_status.startswith("BLOCKED_"):
                is_logical_cancel = new_status in ("CANCELLED", "PRUNED", "HUMAN_INVALIDATED", "SUPERSEDED")
                was_hq_submitted = (prev_hq_job_id is not None) or (task.hq_job_id is not None)
                has_unconf = self._has_unconfirmed_hq_cancel_in_conn(
                    conn, "task", task_id, int(expected_attempt)
                )
                if not has_unconf and not (is_logical_cancel and was_hq_submitted):
                    conn.execute(
                        """
                        UPDATE token_usage_ledger
                        SET reserved_tokens = 0, state = 'RELEASED', updated_ts = ?
                        WHERE usage_key = ? AND state = 'RESERVED'
                        """,
                        (ts, f"task:{task_id}:a{int(expected_attempt)}"),
                    )
            self._record_event(
                conn,
                pending,
                "TASK_TRANSITION_FENCED",
                goal_id=task.goal_id,
                hypothesis_id=task.hypothesis_id,
                task_id=task.task_id,
                payload={
                    "from": cur_status,
                    "to": task.status,
                    "attempt": int(task.attempt),
                    "goal_version": int(expected_goal_version),
                    "human_epoch": int(expected_human_epoch),
                },
                ts=ts,
            )
            return task

    def fenced_resolve_decision_request(
        self,
        req_id: str,
        *,
        expected_goal_version: int,
        expected_human_epoch: int,
        expected_attempt: int,
        expected_lease_id: str | None = None,
        expected_statuses: Sequence[str] | str = ("RUNNING",),
        status: str = "RESOLVED",
        resolution: Mapping[str, Any] | None = None,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Atomically fence `(goal_version, human_epoch, attempt, lease_id)` and resolve a DecisionRequest."""
        expected = (
            {expected_statuses}
            if isinstance(expected_statuses, str)
            else set(expected_statuses)
        )
        ts = time.time() if now is None else float(now)
        terminal = status not in ("PENDING", "SUBMITTING", "RUNNING")
        with self._tx() as (conn, pending):
            row = conn.execute(
                "SELECT * FROM decision_requests WHERE req_id = ?", (req_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown req_id: {req_id}")
            goal_id = str(row["goal_id"])
            ok, reason, _fresh_goal = self.verify_goal_epoch_in_tx(
                conn,
                goal_id,
                expected_goal_version=expected_goal_version,
                expected_human_epoch=expected_human_epoch,
            )
            if not ok:
                raise FencedStaleWriteError(
                    f"DecisionRequest {req_id} failed goal epoch fence: {reason}",
                    reason_code=reason,
                )
            cur_status = str(row["status"])
            if cur_status not in expected:
                raise FencedStaleWriteError(
                    f"DecisionRequest {req_id} status is {cur_status!r}, expected {sorted(expected)}",
                    reason_code="STATUS_MISMATCH",
                )
            cur_payload = json.loads(row["payload_json"])
            cur_attempt = int(cur_payload.get("attempt", 1))
            if cur_attempt != int(expected_attempt):
                raise FencedStaleWriteError(
                    f"DecisionRequest {req_id} attempt is {cur_attempt}, expected {expected_attempt}",
                    reason_code="STALE_ATTEMPT",
                )
            if not self.lease_matches_in_tx(
                conn, f"dec-{req_id}", expected_lease_id, now=ts, allow_none=False
            ):
                raise FencedStaleWriteError(
                    f"DecisionRequest {req_id} lease {expected_lease_id!r} is missing or expired",
                    reason_code="LEASE_MISMATCH",
                )

            res_body = (
                dict(resolution)
                if resolution is not None
                else json.loads(row["resolution_json"])
            )
            res_json = json.dumps(res_body, sort_keys=True, ensure_ascii=False)
            resolved_ts = ts if terminal else row["resolved_ts"]
            conn.execute(
                """
                UPDATE decision_requests
                SET status = ?, resolution_json = ?, updated_ts = ?, resolved_ts = ?
                WHERE req_id = ?
                """,
                (status, res_json, ts, resolved_ts, req_id),
            )
            if terminal:
                eff_ukey = str(
                    cur_payload.get("usage_key")
                    or (
                        f"decision:{req_id}:a{cur_attempt}:{cur_payload['submit_nonce']}"
                        if cur_payload.get("submit_nonce")
                        else f"decision:{req_id}:a{cur_attempt}"
                    )
                )
                in_t = int(res_body.get("in_tokens", 0) or 0)
                out_t = int(res_body.get("out_tokens", 0) or 0)
                c_t = int(res_body.get("cached_input_tokens", res_body.get("cached_in_tokens", 0)) or 0)
                if in_t + out_t > 0 or status == "RESOLVED":
                    self.settle_goal_budget(
                        goal_id,
                        eff_ukey,
                        entity_kind="decision_request",
                        entity_id=req_id,
                        attempt=cur_attempt,
                        model=str(res_body.get("model_used") or cur_payload.get("model_spec") or row["model_spec"] or ""),
                        input_tokens=in_t,
                        output_tokens=out_t,
                        cached_input_tokens=c_t,
                        state="SETTLED",
                    )
                else:
                    is_logical_cancel = status in (
                        "CANCELLED",
                        "SUPERSEDED",
                        "DISCARDED_BY_NEWER_HUMAN_INTERVENTION",
                    )
                    was_hq_submitted_or_submitting = (
                        (row["hq_job_id"] is not None)
                        or (cur_status in ("SUBMITTING", "RUNNING"))
                        or bool(cur_payload.get("submit_nonce"))
                    )
                    has_unconf = self._has_unconfirmed_hq_cancel_in_conn(
                        conn, "decision", req_id, cur_attempt, usage_key=eff_ukey
                    )
                    if not has_unconf and not (is_logical_cancel and was_hq_submitted_or_submitting):
                        self.release_goal_budget_reservation(
                            goal_id,
                            eff_ukey,
                            reason=f"decision_terminal_fenced:{status}",
                            now=ts,
                        )
                        if eff_ukey != f"decision:{req_id}:a{cur_attempt}":
                            self.release_goal_budget_reservation(
                                goal_id,
                                f"decision:{req_id}:a{cur_attempt}",
                                reason=f"decision_terminal_fenced:{status}",
                                now=ts,
                            )
            self._record_event(
                conn,
                pending,
                "DECISION_REQUEST_RESOLVED_FENCED",
                goal_id=goal_id,
                payload={
                    "req_id": req_id,
                    "from": cur_status,
                    "to": status,
                    "attempt": cur_attempt,
                    "goal_version": int(expected_goal_version),
                    "human_epoch": int(expected_human_epoch),
                },
                ts=ts,
            )
            updated = conn.execute(
                "SELECT * FROM decision_requests WHERE req_id = ?", (req_id,)
            ).fetchone()
            return self._row_to_decision_request(updated)
