"""Example project adapter: experiments write ``results.csv``; the adapter turns it into evidence.

An adapter tells Conjectra three project-specific things:

1. ``seed_hypotheses`` — deterministic starting hypotheses (used by ``--propose-mode seeds``);
2. ``build_stage_command`` — the shell command each stage runs inside the worktree;
3. ``parse_metrics`` — how to read metrics from stdout / artifacts, and what they mean.

Verdict logic (baseline comparison, kill criteria, missing-metric handling) is shared: call
``evaluate_scientific_verdict`` instead of re-implementing it.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from conjectra.adapters import (
    GenericAdapter,
    compute_hypothesis_fingerprint,
    evaluate_scientific_verdict,
    metric_entry,
    parse_test_counts,
)
from conjectra.types import (
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ResearchGoal,
    ScientificVerdict,
    StageKind,
    StageTask,
)

__all__ = ["CsvMetricsAdapter"]


class CsvMetricsAdapter(GenericAdapter):
    name = "csv_metrics"

    def seed_hypotheses(self, goal: ResearchGoal) -> list[HypothesisNode]:
        seeds = [
            ("lr-race", HypothesisKind.PARAMETER_RACE, "Learning-rate race", {"lr": [1e-4, 3e-4, 1e-3]}),
            ("alt-loss", HypothesisKind.COMPETING_ROUTE, "Alternative loss function", {"loss": "huber"}),
        ]
        out = []
        for slug, kind, title, params in seeds:
            writable = list(goal.writable_prefixes)
            mech = f"csv_metrics::{slug}"
            out.append(
                HypothesisNode(
                    hypothesis_id=f"{goal.goal_id}-{slug}",
                    goal_id=goal.goal_id,
                    parent_id=None,
                    kind=kind,
                    title=title,
                    statement=f"{title} improves {goal.primary_metric}",
                    mechanism=mech,
                    fidelity=FidelityRung.F1_FAST_SCREEN,
                    priority_evoi=1.0,
                    writable_paths=writable,
                    params=params,
                    kill_criteria=list(goal.kill_criteria) or ["tests_passed == 0"],
                    acceptance_cmd="python3 -m pytest -q",
                    screen_cmd="python3 experiment.py --quick --out {artifacts}/results.csv",
                    confirm_cmd="python3 experiment.py --out {artifacts}/results.csv",
                    branch_name=f"arr/{goal.project}/{goal.goal_id}-{slug}",
                    base_sha=goal.baseline_sha,
                    fingerprint=compute_hypothesis_fingerprint(goal.goal_id, kind, writable, params, mech),
                )
            )
        return out

    def build_stage_command(
        self, task: StageTask, hyp: HypothesisNode, goal: ResearchGoal, worktree_path: Path, artifact_dir: Path
    ) -> str:
        cmd = super().build_stage_command(task, hyp, goal, worktree_path, artifact_dir)
        return cmd.replace("{artifacts}", str(artifact_dir))

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
        if stage not in (StageKind.EXP_SCREEN, StageKind.EXP_CONFIRM):
            return super().parse_metrics(stage, stdout, stderr, worktree_path, artifact_dir, exit_code, hyp, goal)
        rung = FidelityRung.F2_FULL_CONFIRM if stage == StageKind.EXP_CONFIRM else FidelityRung.F1_FAST_SCREEN
        metrics: dict[str, Any] = {}
        flat: dict[str, float | int | bool] = {}
        passed, total = parse_test_counts(f"{stdout}\n{stderr}", exit_code)
        if total:
            metrics["tests_passed"] = metric_entry(passed, total, rung, unit="tests")
            flat["tests_passed"] = passed
        results = artifact_dir / "results.csv"
        artifacts: list[str] = []
        if results.is_file():
            artifacts.append(str(results))
            with results.open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))  # columns: metric,value,n
            for row in rows:
                try:
                    val = float(row["value"])
                except (KeyError, ValueError):
                    continue
                if math.isfinite(val):
                    metrics[row["metric"]] = metric_entry(val, row.get("n") or "1", rung)
                    flat[row["metric"]] = val
        validity, verdict, reason = evaluate_scientific_verdict(
            metrics, flat, goal, hyp, stage, exit_code, combined_output=stdout
        )
        self.last_evidence_validity = validity
        return metrics, verdict, artifacts, reason
