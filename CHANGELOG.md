# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break).

## [Unreleased]

### Added
* **SubGoal portfolio & two-tier decision architecture** (`SubGoalNode`, SQLite schema `user_version = 9`):
  * Hierarchical track decomposition (`DIRECTION_TRACK`, `FOUNDATION_BUILD`, `INTEGRATION_MERGE`) with local metric verdicts, dependency DAGs (`LOCKED_DEP` auto-unlock with `sealed_sha` propagation), weighted non-starvation scheduling, and `max_auto_generations` caps.
  * Two-tier decisions: track-isolated `TrackLead` (`RESEARCH_LEAD` on `<3KB` per-SubGoal digests) and portfolio-level `ApexDirector` (`RESEARCH_DIRECTOR` on live portfolio sandtable + `SPAWN_SUBGOAL`, `RETUNE_SUBGOAL`, `MERGE_SUBGOALS`, `KILL_AND_REALLOCATE`, `SYNC_SUBGOAL_BASE`).
  * Wave-gated fast-lane execution (Pre-F1 `DESIGN_REVIEW` bypass when non-mandatory; `FOUNDATION_BUILD` collapse to `direct_hq` F0/F1; bounded `<4KB` safe `spec_paths` prompt excerpts).
  * SubGoal CLI & interactive controls (`conjectra subgoals`, `inspect --subgoal-id`, `intervene --scope subgoal`), plus SQL fast-path aggregation in `overview`.
* **Generic `slurm_cluster` governance & multi-slot cloud state**: `slurm_queue_checked` / `slurm_owner_approved` policy gates, cached `slurm_sinfo.json` / `slurm_queue_state.json` sandtable snapshots, and multi-account `cloud_slot_state*.json` aggregation.
* **Controller & workflow resilience**: paginated event hydration (`>5000` events), bounded single infra retry for `FAILED` decisions, orphan/terminal workflow isolation, `proc_start_time` tuple/list liveness verification, and stage-specific gate/screen/confirm timeouts.

## [0.1.0] — first public release

### Added
* Zero-token multi-goal `ResearchController` with unix-socket wake-ups and fallback sweeps.
* SQLite WAL ledger: goals, hypotheses, single-attempt stage tasks, evidence receipts, versioned
  research / experiment / execution plans, review receipts, interventions, leases, events.
* Stage pipeline `PROPOSE → DESIGN_REVIEW → CODE → CPU_GATE → IMPL_REVIEW → EXP_SCREEN →
  EXP_CONFIRM → RESULT_REVIEW → RECONCILE` with separate engineering / scientific / audit verdicts and
  failure-domain classification.
* Automatic ablation and falsification branches, graveyard de-duplication, EVOI-ranked proposals.
* `ModelRouter` with provider catalog, structured and natural-language routing policies, `HARD` /
  `SOFT_FALLBACK` strengths, cross-family reviewer selection and quota awareness.
* Backend plugin interface (`conjectra.backends`) with `claude_cli`, `codex_cli`, `gemini_cli`,
  `openai_compat`, `command`, `dispatch_script` and an offline `fake` backend; registration via code,
  `CONJECTRA_BACKENDS` or the `conjectra.backends` entry point.
* Project adapter registry (`conjectra.adapters`) with a generic `METRIC name=value` adapter.
* Collaboration primitives: `direct_hq`, `x-boost`, `x-swarm`, `x-teamwork`, `readonly_fanout`, and a
  workload router.
* Human-in-the-loop interventions with `human_epoch` / `goal_version` fencing; < 50 ms `overview` /
  `inspect`; < 3 KB `takeover` brief; status snapshot and mirror export.
* HyperQueue execution with resource-aware placement, plus an `inline` mode.
* Offline quickstart, backend and adapter plugin examples, GitHub Actions CI.
