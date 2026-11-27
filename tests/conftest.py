"""Shared pytest fixtures: registers a small three-seed demo adapter used by several tests."""

from __future__ import annotations

import os

# The production default keeps 10 GiB of MemAvailable free before heavy stages start. Hosted CI
# runners and laptops often have less, which would make every end-to-end test wait for memory.
# Tests that check the default itself clear this variable explicitly. Subprocesses inherit it.
os.environ.setdefault("CONJECTRA_MIN_AVAIL_RAM_MB", "256")

from conjectra.adapters import GenericAdapter, compute_hypothesis_fingerprint, register_adapter
from conjectra.types import FidelityRung, HypothesisKind, HypothesisNode, ResearchGoal


class ThreeSeedAdapter(GenericAdapter):
    """Deterministic adapter emitting a parameter race, a competing route and a falsification seed."""

    name = "demo3"

    def seed_hypotheses(self, goal: ResearchGoal) -> list[HypothesisNode]:
        prefix = (goal.writable_prefixes or ["src/"])[0].rstrip("/")
        specs = [
            ("h1-param-race", HypothesisKind.PARAMETER_RACE, "Learning-rate / warmup parameter race",
             {"lr": 3e-4, "warmup": 200}, 1.5),
            ("h2-route-alt", HypothesisKind.COMPETING_ROUTE, "Alternative optimizer route",
             {"route": "alt_optimizer"}, 1.3),
            ("h3-falsify-ub95", HypothesisKind.FALSIFICATION, "Falsify the UB95 <= 0.0148 regression bound",
             {"n_samples": 201, "max_ub95": 0.0148}, 1.2),
        ]
        out: list[HypothesisNode] = []
        for slug, kind, title, params, evoi in specs:
            writable = [f"{prefix}/{slug.replace('-', '_')}.py", f"{prefix}/test_{slug.replace('-', '_')}.py"]
            mech = f"{title} (deterministic demo seed)"
            out.append(
                HypothesisNode(
                    hypothesis_id=f"{goal.goal_id}-{slug}",
                    goal_id=goal.goal_id,
                    parent_id=None,
                    kind=kind,
                    title=title,
                    statement=f"{title} improves {goal.primary_metric}.",
                    mechanism=mech,
                    fidelity=FidelityRung.F1_FAST_SCREEN,
                    priority_evoi=evoi,
                    writable_paths=writable,
                    params=params,
                    kill_criteria=list(goal.kill_criteria) or ["tests_passed == 0"],
                    acceptance_cmd="pytest -q",
                    screen_cmd="pytest -q",
                    confirm_cmd="pytest -q",
                    branch_name=f"arr/{goal.project}/{goal.goal_id}-{slug}",
                    base_sha=goal.baseline_sha,
                    fingerprint=compute_hypothesis_fingerprint(goal.goal_id, kind, writable, params, mech),
                )
            )
        return out


register_adapter("demo3", ThreeSeedAdapter)
