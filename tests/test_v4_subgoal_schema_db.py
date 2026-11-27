"""Unit and migration tests for V4 SubGoal schema, types, and StateDB operations."""

from __future__ import annotations

import concurrent.futures
import pathlib
import sqlite3

import pytest

from conjectra.state_db import StateDB, StateTransitionError
from conjectra.types import (
    ActionKind,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    LeadAction,
    ResearchGoal,
    ResourceSpec,
    SubGoalKind,
    SubGoalNode,
    SubGoalStatus,
)


def _make_goal(goal_id: str = "goal-v4-1") -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project="projgamma",
        title="V4 SubGoal Portfolio",
        direction="Improve end-to-end throughput via modular subgoals",
        primary_metric="throughput",
        baseline_value=1.0,
        baseline_sha="deadbeef0001",
        repo_path="/tmp/v4-repo",
        kill_criteria=["throughput < 0.8"],
        banned_paths=[],
        writable_prefixes=["src/", "tests/"],
        max_tokens_budget=500_000,
    )


def _make_hyp(
    hypothesis_id: str = "hyp-v4-1",
    goal_id: str = "goal-v4-1",
    *,
    subgoal_id: str | None = None,
    params: dict | None = None,
    mechanism: str = "kernel-fusion",
) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id=hypothesis_id,
        goal_id=goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title=f"Hyp {hypothesis_id}",
        statement="Fused kernel lowers memory traffic",
        mechanism=mechanism,
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.75,
        writable_paths=["src/kernel.py"],
        params=dict(params or {}),
        kill_criteria=["latency_ms > 50"],
        acceptance_cmd="pytest -q tests/test_kernel.py",
        screen_cmd="python -m src.bench",
        confirm_cmd=None,
        branch_name=f"arr/projgamma/{hypothesis_id}",
        base_sha="deadbeef0001",
        subgoal_id=subgoal_id,
    )


def _make_subgoal(
    subgoal_id: str = "sg-1",
    goal_id: str = "goal-v4-1",
    *,
    kind: SubGoalKind = SubGoalKind.DIRECTION_TRACK,
    status: str = "ACTIVE",
    priority_weight: float = 1.0,
) -> SubGoalNode:
    return SubGoalNode(
        subgoal_id=subgoal_id,
        goal_id=goal_id,
        title=f"SubGoal {subgoal_id}",
        direction_thesis="Replace inner dense matmul with block-sparse tile kernel",
        direct_contribution="2x kernel speedup feeding integration track",
        local_metric="kernel_gflops",
        kind=kind,
        status=status,
        baseline_value=120.0,
        target_value=240.0,
        direction="maximize",
        kill_criteria=["kernel_gflops < 100.0"],
        depends_on_subgoals=[],
        source_subgoal_ids=[],
        track_branch=f"track/projgamma/{subgoal_id}",
        base_sha="deadbeef0001",
        sealed_sha=None,
        compute_target="cpu_only",
        spec_paths=["docs/kernel_spec.md"],
        writable_prefixes=["src/kernel/"],
        priority_weight=priority_weight,
        max_tokens_budget=80_000,
        max_auto_generations=2,
        wave_gated=True,
        params={"tile_size": 32},
    )


def test_subgoal_enums_actions_and_node_roundtrip() -> None:
    assert SubGoalKind.DIRECTION_TRACK.value == "DIRECTION_TRACK"
    assert SubGoalKind.FOUNDATION_BUILD.value == "FOUNDATION_BUILD"
    assert SubGoalKind.DIAGNOSTIC_PROBE.value == "DIAGNOSTIC_PROBE"
    assert SubGoalKind.COMPETING_RACE.value == "COMPETING_RACE"
    assert SubGoalKind.INTEGRATION_MERGE.value == "INTEGRATION_MERGE"

    assert {s.value for s in SubGoalStatus} == {
        "ACTIVE",
        "LOCKED_DEP",
        "PAUSED",
        "AWAITING_APEX",
        "SEALED",
        "MERGED",
        "KILLED",
    }

    for action_name in (
        "SPAWN_SUBGOAL",
        "RETUNE_SUBGOAL",
        "MERGE_SUBGOALS",
        "KILL_AND_REALLOCATE",
        "SYNC_SUBGOAL_BASE",
    ):
        assert ActionKind(action_name).value == action_name

    sg = _make_subgoal("sg-rt", priority_weight=1.75)
    sg.depends_on_subgoals = ["sg-0"]
    sg.source_subgoal_ids = ["sg-a", "sg-b"]
    sg.sealed_sha = "cafebabe9999"
    sg.created_ts = 100.5
    sg.updated_ts = 200.5

    raw = sg.to_dict()
    restored = SubGoalNode.from_dict(raw)
    assert restored == sg
    assert restored.kind is SubGoalKind.DIRECTION_TRACK
    assert restored.target_value == 240.0
    assert restored.sealed_sha == "cafebabe9999"
    assert restored.max_auto_generations == 2
    assert restored.wave_gated is True

    minimal = SubGoalNode.from_dict(
        {
            "subgoal_id": "sg-min",
            "goal_id": "g-min",
            "title": "Minimal",
            "direction_thesis": "Thesis",
            "direct_contribution": "Contrib",
            "local_metric": "loss",
        }
    )
    assert minimal.kind is SubGoalKind.DIRECTION_TRACK
    assert minimal.status == "ACTIVE"
    assert minimal.target_value is None
    assert minimal.sealed_sha is None
    assert minimal.priority_weight == 1.0
    assert minimal.max_auto_generations == 1
    assert minimal.wave_gated is True


def test_hypothesis_and_lead_action_subgoal_id_backward_compat() -> None:
    # ResourceSpec.to_dict() keys remain untouched
    assert set(ResourceSpec().to_dict().keys()) == {
        "cpus",
        "ram_mb",
        "gpu",
        "llm_pool",
        "eda_sim",
        "cloud_slot",
        "wall_timeout_s",
    }

    hyp_explicit = _make_hyp("h-exp", subgoal_id="sg-10")
    hyp_dict = hyp_explicit.to_dict()
    assert hyp_dict["subgoal_id"] == "sg-10"
    assert HypothesisNode.from_dict(hyp_dict).subgoal_id == "sg-10"

    # Legacy HypothesisNode dict with subgoal_id only in params
    legacy_hyp_dict = hyp_explicit.to_dict()
    legacy_hyp_dict.pop("subgoal_id", None)
    legacy_hyp_dict["params"] = {"subgoal_id": "sg-from-params"}
    assert HypothesisNode.from_dict(legacy_hyp_dict).subgoal_id == "sg-from-params"

    # Neither set -> None
    plain_hyp_dict = hyp_explicit.to_dict()
    plain_hyp_dict.pop("subgoal_id", None)
    plain_hyp_dict["params"] = {}
    assert HypothesisNode.from_dict(plain_hyp_dict).subgoal_id is None

    # LeadAction explicit target_subgoal_id
    act_explicit = LeadAction(
        action=ActionKind.RETUNE_SUBGOAL,
        rationale="Lower target slightly",
        parameters={"target_value": 200.0},
        target_subgoal_id="sg-10",
    )
    act_dict = act_explicit.to_dict()
    assert act_dict["target_subgoal_id"] == "sg-10"
    assert LeadAction.from_dict(act_dict).target_subgoal_id == "sg-10"

    # Legacy LeadAction dict with subgoal_id inside parameters
    act_legacy = LeadAction.from_dict(
        {
            "action": "SYNC_SUBGOAL_BASE",
            "parameters": {"subgoal_id": "sg-param-1", "sha": "abc123"},
        }
    )
    assert act_legacy.target_subgoal_id == "sg-param-1"

    act_plain = LeadAction.from_dict({"action": "CONTINUE"})
    assert act_plain.target_subgoal_id is None


def test_v8_to_v9_idempotent_migration_and_backfill(tmp_path: pathlib.Path) -> None:
    arr_home = tmp_path / "v8_home"
    with StateDB(arr_home) as db:
        assert int(db._conn.execute("PRAGMA user_version").fetchone()[0]) == 9
        db.save_goal(_make_goal("g-mig"))
        hyp = _make_hyp("h-mig", "g-mig", params={"subgoal_id": "sg-backfill-h"})
        db.register_hypothesis(hyp)
        dreq = db.enqueue_decision_request(
            "g-mig",
            tier="L1",
            payload={"subgoal_id": "sg-backfill-d", "note": "legacy"},
        )

    # Downgrade DB on disk to simulate an exact pre-v9 (v8) schema state
    db_path = arr_home / "state.db"
    raw = sqlite3.connect(str(db_path))
    raw.executescript("DROP TABLE IF EXISTS subgoals;")
    raw.execute("DROP INDEX IF EXISTS idx_hypotheses_goal_subgoal")
    raw.execute("DROP INDEX IF EXISTS idx_decision_requests_goal_subgoal")
    raw.execute("ALTER TABLE hypotheses DROP COLUMN subgoal_id")
    raw.execute("ALTER TABLE decision_requests DROP COLUMN subgoal_id")
    raw.execute("PRAGMA user_version = 8")
    raw.commit()
    raw.close()

    # Concurrent opens migrate v8 -> v9 atomically
    def _open_ver(_idx: int) -> int:
        with StateDB(arr_home) as mig_db:
            return int(mig_db._conn.execute("PRAGMA user_version").fetchone()[0])

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        vers = list(pool.map(_open_ver, range(4)))
    assert vers == [9, 9, 9, 9]

    with StateDB(arr_home) as db:
        with db._lock:
            tables = {
                str(r[0])
                for r in db._conn.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                ).fetchall()
            }
            assert "subgoals" in tables

            sg_idxs = {
                str(r[1])
                for r in db._conn.execute("PRAGMA index_list(subgoals)").fetchall()
            }
            assert "idx_subgoals_goal_status" in sg_idxs

            hyp_cols = {
                str(r[1])
                for r in db._conn.execute("PRAGMA table_info(hypotheses)").fetchall()
            }
            assert "subgoal_id" in hyp_cols
            hyp_idxs = {
                str(r[1])
                for r in db._conn.execute("PRAGMA index_list(hypotheses)").fetchall()
            }
            assert "idx_hypotheses_goal_subgoal" in hyp_idxs

            dreq_cols = {
                str(r[1])
                for r in db._conn.execute("PRAGMA table_info(decision_requests)").fetchall()
            }
            assert "subgoal_id" in dreq_cols
            dreq_idxs = {
                str(r[1])
                for r in db._conn.execute("PRAGMA index_list(decision_requests)").fetchall()
            }
            assert "idx_decision_requests_goal_subgoal" in dreq_idxs

            h_row = db._conn.execute(
                "SELECT subgoal_id FROM hypotheses WHERE hypothesis_id = 'h-mig'"
            ).fetchone()
            assert h_row is not None and h_row["subgoal_id"] == "sg-backfill-h"

            d_row = db._conn.execute(
                "SELECT subgoal_id FROM decision_requests WHERE req_id = ?",
                (dreq["req_id"],),
            ).fetchone()
            assert d_row is not None and d_row["subgoal_id"] == "sg-backfill-d"

    # Re-opening is idempotent
    with StateDB(arr_home) as db2:
        assert int(db2._conn.execute("PRAGMA user_version").fetchone()[0]) == 9


def test_subgoal_crud_ownership_transitions_and_fields(tmp_path: pathlib.Path) -> None:
    with StateDB(tmp_path / "crud_home") as db:
        db.save_goal(_make_goal("g1"))
        db.save_goal(_make_goal("g2"))

        sg1 = _make_subgoal("sg-1", "g1", priority_weight=1.0)
        sg2 = _make_subgoal("sg-2", "g1", status="LOCKED_DEP", priority_weight=2.5)
        sg3 = _make_subgoal("sg-3", "g2", status="PAUSED", priority_weight=3.0)

        saved1 = db.save_subgoal(sg1, now=1000.0)
        assert saved1.created_ts == 1000.0
        assert saved1.updated_ts == 1000.0

        # Upsert preserves created_ts and updates updated_ts
        saved1_u = db.save_subgoal(
            SubGoalNode.from_dict({**saved1.to_dict(), "title": "Updated Title"}),
            now=1100.0,
        )
        assert saved1_u.created_ts == 1000.0
        assert saved1_u.updated_ts == 1100.0
        assert saved1_u.title == "Updated Title"

        # Cross-goal collision rejected
        with pytest.raises(ValueError, match="Cross-goal ownership violation"):
            db.save_subgoal(_make_subgoal("sg-1", "g2"))

        db.save_subgoal(sg2, now=1001.0)
        db.save_subgoal(sg3, now=1002.0)

        assert db.get_subgoal("missing") is None
        assert db.get_goal_subgoal("g1", "missing") is None
        assert db.get_goal_subgoal("g1", "sg-1") is not None
        with pytest.raises(ValueError, match="Cross-goal ownership violation"):
            db.get_goal_subgoal("g1", "sg-3")

        # list_subgoals ordered by descending priority_weight
        g1_all = db.list_subgoals(goal_id="g1")
        assert [s.subgoal_id for s in g1_all] == ["sg-2", "sg-1"]
        g1_locked = db.list_subgoals(goal_id="g1", status="LOCKED_DEP")
        assert [s.subgoal_id for s in g1_locked] == ["sg-2"]
        all_paused = db.list_subgoals(status="PAUSED")
        assert [s.subgoal_id for s in all_paused] == ["sg-3"]

        # transition_subgoal_status
        with pytest.raises(KeyError):
            db.transition_subgoal_status("sg-nope", "ACTIVE", "SEALED")
        with pytest.raises(StateTransitionError):
            db.transition_subgoal_status("sg-2", "ACTIVE", "SEALED")

        unlocked = db.transition_subgoal_status(
            "sg-2",
            ["LOCKED_DEP", "PAUSED"],
            "ACTIVE",
            params_patch={"unlocked_reason": "upstream_sealed"},
            now=1200.0,
        )
        assert unlocked.status == "ACTIVE"
        assert unlocked.params["unlocked_reason"] == "upstream_sealed"

        sealed = db.transition_subgoal_status(
            "sg-1",
            None,
            "SEALED",
            sealed_sha="feedface1234",
            now=1300.0,
        )
        assert sealed.status == "SEALED"
        assert sealed.sealed_sha == "feedface1234"

        # update_subgoal_fields updates payload and indexed columns
        with pytest.raises(KeyError):
            db.update_subgoal_fields("sg-nope", {"local_metric": "x"})

        retuned = db.update_subgoal_fields(
            "sg-2",
            {
                "local_metric": "tile_latency_us",
                "target_value": 15.0,
                "direction": "minimize",
                "priority_weight": 5.0,
                "base_sha": "feedface1234",
                "max_tokens_budget": 120_000,
                "goal_id": "hijack-ignored",
            },
            now=1400.0,
        )
        assert retuned.goal_id == "g1"
        assert retuned.local_metric == "tile_latency_us"
        assert retuned.target_value == 15.0
        assert retuned.direction == "minimize"
        assert retuned.priority_weight == 5.0
        assert retuned.base_sha == "feedface1234"
        assert retuned.max_tokens_budget == 120_000

        with db._lock:
            row = db._conn.execute(
                "SELECT priority_weight, base_sha, sealed_sha, status FROM subgoals WHERE subgoal_id = 'sg-2'"
            ).fetchone()
            assert row["priority_weight"] == 5.0
            assert row["base_sha"] == "feedface1234"
            assert row["status"] == "ACTIVE"


def test_hypothesis_and_decision_request_subgoal_filters_and_sync(
    tmp_path: pathlib.Path,
) -> None:
    with StateDB(tmp_path / "filter_home") as db:
        db.save_goal(_make_goal("g1"))
        db.save_subgoal(_make_subgoal("sg-a", "g1"))
        db.save_subgoal(_make_subgoal("sg-b", "g1"))

        h1, created1 = db.register_hypothesis(
            _make_hyp("h1", "g1", subgoal_id="sg-a", mechanism="m1")
        )
        h2, created2 = db.register_hypothesis(
            _make_hyp("h2", "g1", params={"subgoal_id": "sg-b"}, mechanism="m2")
        )
        h3, created3 = db.register_hypothesis(
            _make_hyp("h3", "g1", subgoal_id=None, mechanism="m3")
        )
        assert created1 and created2 and created3
        assert h2.subgoal_id == "sg-b"

        assert [h.hypothesis_id for h in db.list_hypotheses(goal_id="g1", subgoal_id="sg-a")] == ["h1"]
        assert [h.hypothesis_id for h in db.list_hypotheses(subgoal_id="sg-b")] == ["h2"]
        assert len(db.list_hypotheses(goal_id="g1")) == 3

        # Updating subgoal_id via update_hypothesis_fields / update_hypothesis_params keeps column synced
        db.update_hypothesis_fields("h3", {"subgoal_id": "sg-a"})
        assert {
            h.hypothesis_id for h in db.list_hypotheses(goal_id="g1", subgoal_id="sg-a")
        } == {"h1", "h3"}

        db.update_hypothesis_params("h3", {"subgoal_id": "sg-b"})
        assert [
            h.hypothesis_id for h in db.list_hypotheses(goal_id="g1", subgoal_id="sg-a")
        ] == ["h1"]
        assert {
            h.hypothesis_id for h in db.list_hypotheses(goal_id="g1", subgoal_id="sg-b")
        } == {"h2", "h3"}

        # Transitioning & graveyarding preserve subgoal_id column
        db.transition_hypothesis_status("h1", "PENDING", "RUNNING")
        assert [
            h.hypothesis_id
            for h in db.list_hypotheses(goal_id="g1", status="RUNNING", subgoal_id="sg-a")
        ] == ["h1"]
        db.record_graveyard("h2", "refuted")
        assert [
            h.hypothesis_id
            for h in db.list_hypotheses(goal_id="g1", status="GRAVEYARD", subgoal_id="sg-b")
        ] == ["h2"]

        # DecisionRequest enqueue + coalesce + subgoal_id filter
        d1 = db.enqueue_decision_request(
            "g1",
            tier="L1",
            subgoal_id="sg-a",
            coalesce_key="track:sg-a",
            payload={"reason": "first"},
            now=100.0,
        )
        assert d1["subgoal_id"] == "sg-a"
        assert d1["payload"]["subgoal_id"] == "sg-a"

        d1_coalesced = db.enqueue_decision_request(
            "g1",
            tier="L1",
            coalesce_key="track:sg-a",
            payload={"reason": "second"},
            now=105.0,
        )
        assert d1_coalesced["req_id"] == d1["req_id"]
        assert d1_coalesced["subgoal_id"] == "sg-a"

        d2 = db.enqueue_decision_request(
            "g1",
            tier="L1",
            payload={"subgoal_id": "sg-b"},
            now=110.0,
        )
        d3 = db.enqueue_decision_request("g1", tier="L2", now=115.0)
        assert d2["subgoal_id"] == "sg-b"
        assert d3["subgoal_id"] is None

        assert [r["req_id"] for r in db.list_decision_requests("g1", subgoal_id="sg-a")] == [d1["req_id"]]
        assert [r["req_id"] for r in db.list_decision_requests(subgoal_id="sg-b")] == [d2["req_id"]]
        assert len(db.list_decision_requests("g1")) == 3
