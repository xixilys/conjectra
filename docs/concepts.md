# Concepts

This page is the vocabulary you need to read Conjectra's ledger, CLI output and code.

## Goal (`ResearchGoal`)

A research question bound to one git repository:

| Field | Meaning |
|---|---|
| `goal_id`, `project`, `title` | Identity. `project` also names branches (`arr/<project>/<hypothesis>`). |
| `primary_metric`, `direction`, `baseline_value`, `target_value` | What "better" means. The metric is parsed from stage output by the project adapter. |
| `repo_path`, `baseline_sha` | The repository and commit every hypothesis branches from. The original checkout is never written to. |
| `writable_prefixes` | The only paths a hypothesis may modify (e.g. `["src"]`). |
| `banned_paths` | Paths no proposal, command or diff may touch (sealed eval sets, holdouts, credentials). |
| `kill_criteria` | Expressions such as `tests_passed == 0` or `ub95 > 0.0148`; hitting one is a *refutation*. |
| `max_tokens_budget`, `max_concurrent_branches`, `priority_weight`, `min_guaranteed_slots` | Budget and fair-share knobs. |
| `routing_policy` / `routing_policy_nl` | Which models may do what (structured JSON or natural language). |
| `version`, `human_epoch`, `plan_epoch` | Fencing counters, see *Epochs* below. |

## Hypothesis (`HypothesisNode`)

A falsifiable claim plus the commands that test it. Kinds:

* `COMPETING_ROUTE` — an alternative approach to the goal.
* `PARAMETER_RACE` — the same approach with different parameters.
* `REFINEMENT` — a follow-up on a promising branch.
* `ABLATION` — removes a component of a confirmed result to see whether it mattered.
* `FALSIFICATION` — a stress test designed to break a confirmed result.

Confirmed hypotheses automatically spawn ablation and falsification children. Refuted ones go to
the **graveyard**; new proposals whose fingerprint (kind + writable paths + params + mechanism)
matches a graveyard entry are rejected.

Each hypothesis carries `acceptance_cmd` (deterministic tests), `screen_cmd` (fast experiment) and
`confirm_cmd` (full experiment), plus a fidelity rung:

`F0_SANITY → F1_FAST_SCREEN → F2_FULL_CONFIRM → F3_MILESTONE_AUDIT`

## Stages (`StageKind`) and tasks (`StageTask`)

Every step is a separate, single-attempt task with its own resources and timeout:

| Stage | Who works | What it checks |
|---|---|---|
| `PROPOSE` | research lead (model) or adapter seeds | proposals are validated before entering the ledger |
| `DESIGN_REVIEW` | reviewer model, read-only | is the experiment design sound? |
| `CODE` | builder model in an isolated worktree | produces a commit on the hypothesis branch |
| `CPU_GATE` | deterministic | `acceptance_cmd` must pass; dirty worktrees and banned-path diffs fail |
| `IMPL_REVIEW` | reviewer model of a different family, read-only snapshot | is the implementation correct? |
| `EXP_SCREEN` / `EXP_CONFIRM` | deterministic | run the experiment, parse metrics, compare to baseline |
| `RESULT_REVIEW` | reviewer model of a different family | are the conclusions supported by the raw artifacts? |
| `BLIND_AUDIT` | reviewer model | legacy combined audit, still used by some workflows |
| `RECONCILE` | deterministic | merges findings into goal knowledge |

Retries are new tasks with `attempt + 1`, never in-place re-runs.

## Evidence receipts (`EvidenceReceipt`)

Every finished task writes exactly one receipt. A receipt keeps three verdicts apart:

* `engineering_verdict`: `PASS | FAIL_GATE | FAIL_TEST | SKIPPED` — did the code/process work?
* `scientific_verdict`: `CONFIRMED | REFUTED | VALID_NEGATIVE | UNKNOWN | NOT_APPLICABLE` — what did
  the experiment say about the hypothesis?
* `audit_verdict`: `NOT_REQUIRED | PASS | FAIL | PARSE_ERROR` — what did the reviewer say?

and classifies any failure into a `FailureDomain`:

| Failure domain | Examples | Can it refute a hypothesis? |
|---|---|---|
| `INFRA_TRANSIENT` | lease lost, worker killed, 429/502 | no — retried |
| `ENV_CONFIG_MISMATCH` | CUDA missing, bad CLI flag, missing dataset | no — routed to a fix or to you |
| `ENGINEERING_DEFECT` | failing tests, dirty worktree, review FAIL | no — routed back to `CODE` |
| `SCIENTIFIC_OUTCOME` | a valid measurement that beat or missed the baseline | yes |

`evidence_validity` (`VALID_EMPIRICAL`, `MISSING_PRIMARY_METRIC`, `STALE_GOAL_VERSION`,
`HUMAN_INVALIDATED`, …) records whether the numbers can be trusted at all. Only
`SCIENTIFIC_OUTCOME` + `VALID_EMPIRICAL` evidence may move a hypothesis to the graveyard, and
"all tests passed" on its own never counts as a confirmation.

## Roles and model routes

Model work is described by a `ScientificRole` — `research_director`, `research_lead`, `theorist`,
`experiment_designer`, `engineer_builder`, `design_reviewer`, `impl_reviewer`, `result_auditor` —
not by a model name. The `ModelRouter` turns *(role, goal, hypothesis, policy, quotas)* into a
`ResolvedModelRoute` (pool, canonical model, effort, family, backend, host) that is **pinned** to the
task. Reviewer routes avoid the author's model family.

Policies have two strengths: `HARD` (wait for the resource rather than downgrade) and
`SOFT_FALLBACK` (fall back down an ordered candidate list).

## Workflows

A hypothesis is executed by one of several collaboration primitives:

* `direct_hq` — the plain stage pipeline above.
* `x-boost` — builder ↔ blind adversary ping-pong with a bounded number of repairs.
* `x-swarm` — several providers working on coupled interfaces through an append-only board.
* `x-teamwork` — multi-phase milestones with disjoint write scopes and < 4 KB context flushes.
* `readonly_fanout` — parallel read-only analyses.

`conjectra route-workload` classifies a task description and recommends a primitive.

## Epochs and fencing

Interventions (`conjectra intervene ...`) run in one `BEGIN IMMEDIATE` transaction and bump the
goal's `human_epoch` (and, for objective changes, `version`). Every task carries the epoch it was
created under; a result that arrives after an intervention is recorded as `SUPERSEDED_EPOCH` /
`STALE_GOAL_VERSION` evidence instead of changing state. `pause` cancels running jobs and resuming
creates fresh `attempt + 1` tasks.

## Leases

Branches, repository scopes and resource slots are guarded by leases with TTLs. Writers hold
exclusive `branch:` leases; reviewers share read-only `branch_ro:` leases on immutable snapshots, so
several reviews can run in parallel without blocking or seeing a moving target.

## Escalations

Some decisions are reserved for the owner: changing the goal or its success bar, touching `main`
or a holdout, starting new paid compute, and repeated route refutations or audit failures. The controller records an owner escalation, keeps working on
everything else, and shows the pending question in `overview`, `status` and the takeover brief.
