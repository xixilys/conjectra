# Architecture

Conjectra is three cooperating pieces around one SQLite database:

```
┌────────────────────────── control plane (0 LLM tokens) ──────────────────────────┐
│                                                                                   │
│  CLI / API ──tx──▶ SQLite WAL ledger ◀──tx── ResearchController (reactor daemon)   │
│  (submit, intervene,   goals, hypotheses,      ├─ GlobalGoalManager  (multi-goal  │
│   overview, takeover)  tasks, receipts,        │    fair share, interventions)    │
│        │               reviews, plans,         ├─ ResourceArbiter    (RAM, disk,   │
│        └── unix socket wake ──────────────▶    │    load, GPU, quotas, retries)    │
│                        leases, events          ├─ ModelRouter        (policy →     │
│                                                │    pinned route)                 │
│                                                ├─ ExploreEngine      (proposals,  │
│                                                │    graveyard, EVOI)              │
│                                                └─ WorkflowRuntime    (x-boost,    │
│                                                     x-swarm, x-teamwork, …)       │
└───────────────────────────────────────────┬───────────────────────────────────────┘
                                            │ single-attempt jobs (cmd + resources + time limit)
                                            ▼
┌──────────────────────────── execution plane ─────────────────────────────────────┐
│  HyperQueue server + workers   (or: inline, in-process)                           │
│     └─ python -m conjectra.stage_runner --task-id …                               │
│          ├─ SandboxManager: shared bare mirror + sparse git worktree per branch   │
│          ├─ ProjectAdapter: stage command + metric parsing                        │
│          ├─ backend plugin: model call for CODE / reviews / planning              │
│          └─ writes EvidenceReceipt, then wakes the controller via reactor.sock    │
└───────────────────────────────────────────────────────────────────────────────────┘
```

## The controller loop

`ResearchController.step()` is a pure function of the ledger (plus HyperQueue job state). One step:

1. **Reap & recover** — expire leases, reconcile HyperQueue jobs that died, recover tasks stuck in
   `SUBMITTING`, honour a maintenance barrier if one is set.
2. **Reconcile receipts** — for every finished task, apply the receipt: advance the hypothesis,
   route failures by `FailureDomain`, spawn ablation/falsification children, prune refuted routes,
   raise owner escalations.
3. **Decide** — enqueue planning / research-lead / director decision requests where needed (these
   are model calls, so they also run as jobs).
4. **Propose** — top up each goal's ready queue from adapter seeds or model proposals, validated
   against writable/banned paths, kill criteria and the graveyard.
5. **Dispatch** — admit ready tasks by fair share (guaranteed slots, borrowing, aging, foreground
   reserve), acquire leases, pin a model route, and submit.

`conjectra daemon` runs steps on socket wake-ups (a task exit or an intervention) and on a slow
fallback sweep, so refill latency is seconds while idle cost is zero.

## Why single-attempt jobs

* A job that hangs is killed by HyperQueue's time limit; the controller sees `LEASE_LOST` /
  `TIMEOUT`, classifies it as `INFRA_TRANSIENT`, and creates a fresh `attempt + 1` task with a new
  route if the policy allows. No hidden in-process retry loops.
* The route is pinned before submission, so a receipt always says which model, effort and backend
  produced it.
* Everything a job needs is in the ledger; the controller can restart at any time.

## Fencing

Every state-changing write checks a fence inside `BEGIN IMMEDIATE`:
`(goal_version, human_epoch, attempt, lease_id)`. Interventions bump `human_epoch`; objective
changes bump `goal_version`. A stale writer gets a `FencedStaleWriteError` and its evidence is kept
but marked `SUPERSEDED_EPOCH` / `STALE_GOAL_VERSION`.

## Sandboxing

* The goal's repository is cloned once into a shared bare mirror; each hypothesis gets its own
  sparse worktree on branch `arr/<project>/<hypothesis_id>`. The original checkout is never
  modified and `main`/`master` are never targets.
* Reviews run on detached, read-only snapshots of an exact `head_sha` under shared `branch_ro:`
  leases.
* Diffs, commands and proposals are checked against banned paths (goal `banned_paths`, built-in
  redlines such as credentials and sealed eval folders, and an optional `READ_BANS.md`), and against
  test-weakening (`assert` removal, new `skip`/`xfail`).

## Promotion

A hypothesis that passes `EXP_CONFIRM` and its result review is promoted by a fast-forward of a
promotion ref under a file lock, recorded as a durable promotion intent so a crash between "git ref
updated" and "ledger updated" is repaired on the next step.

## Observability

* `overview`, `inspect`, `plans`, `reviews`, `token-audit` read the WAL directly (< 50 ms).
* `takeover` renders a < 3 KB Markdown brief for hand-off.
* `status-snapshot` writes `<home>/status_snapshot.json` for dashboards; `sync-status` writes
  `LATEST_STATUS.md` / `benchmark_latest.json` and can mirror them elsewhere.
* `events.jsonl` in the state home is an append-only event log.
