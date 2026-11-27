# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break).

## [Unreleased]

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
