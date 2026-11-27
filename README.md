# Conjectra

**A zero-token control plane for multi-goal autonomous research.**
You submit research goals against git repositories; Conjectra turns them into falsifiable
hypotheses, runs every step as a short, isolated, single-attempt job, and keeps an append-only
evidence ledger that says — with machine-checked receipts — what was confirmed, what was refuted,
and what is merely unknown.

The name comes from Karl Popper's *Conjectures and Refutations*: a hypothesis only counts when it
survived a real attempt to kill it.

> Status: **alpha**. The runtime was extracted from a private deployment that runs several
> research goals unattended in parallel; it is young, and APIs and the on-disk schema may still
> change. Linux and macOS (POSIX) only.

---

## Why

LLM agents are good at writing code and bad at being honest about results. Long-running agent
loops drift, forget what was already tried, quietly mark flaky runs as wins, review their own work,
and burn tokens while doing nothing. Conjectra separates the two concerns:

* **A deterministic controller decides.** It is a single daemon that never calls a model. It
  schedules, fences, retries, prunes and records — in milliseconds, for zero tokens.
* **Models only work.** Every model call is one bounded job (write code, review a diff, plan an
  experiment) executed in an isolated git worktree, with a pinned model route and a hard timeout.
  Its output is accepted only through evidence gates.

## Features

* **Multi-goal scheduling** — many goals share one worker pool with guaranteed slots, elastic
  borrowing, aging and a foreground reserve for interactive use.
* **Falsification-first evidence ledger** — every stage emits an `EvidenceReceipt` with separate
  *engineering*, *scientific* and *audit* verdicts plus a `FailureDomain`. Infrastructure hiccups,
  environment mismatches and engineering bugs can never be recorded as "hypothesis refuted";
  "tests passed" alone never counts as a scientific confirmation.
* **Automatic ablation & falsification** — a confirmed result spawns its own ablation and stress
  test branches; refuted routes go to a graveyard that de-duplicates future proposals.
* **Cross-family blind review** — design, implementation and result reviews run on read-only
  snapshots and are routed to a *different model family* than the author.
* **Pluggable model backends** — Claude Code, Codex CLI, Gemini CLI, any OpenAI-compatible endpoint,
  any command-line tool, or your own Python plugin. A deterministic `fake` backend makes the whole
  pipeline testable offline.
* **Policy-driven model routing** — one router validates every route against a provider catalog;
  routing policies can be structured JSON or natural language
  (`"theory work uses Opus xhigh, routine analysis uses gpt-5-codex high"`), with `HARD` vs
  `SOFT_FALLBACK` semantics and quota awareness.
* **Human-in-the-loop with fencing** — `pause`, `resume`, `rebalance_goals`, `update_objective`,
  `prune_branch`, `focus`, `inject_hypothesis`, `override_verdict`, … are atomic SQLite
  transactions that bump a `human_epoch`; late results from before the intervention are fenced off
  instead of silently overwriting your decision.
* **SubGoal portfolios & two-tier decisions** — complex goals can decompose into independent
  `SubGoalNode` tracks with local metrics, dependency locks (`LOCKED_DEP` auto-unlock with
  `sealed_sha` propagation), weighted non-starvation scheduling, wave-gated fast lanes, track-isolated
  `TrackLead` reviews (`< 3 KB` digests), and portfolio-level `ApexDirector` actions (`spawn`,
  `retune`, `merge`, `kill`, `sync`).
* **Session-independent hand-off** — `overview` / `subgoals` / `inspect` answer in < 50 ms straight
  from the WAL; `takeover` emits a < 3 KB brief that any human or agent can pick up from.
* **Real job execution** — production runs go through [HyperQueue](https://github.com/It4innovations/hyperqueue)
  with resource-aware placement (CPU, RAM, GPU, LLM pool slots); an `inline` mode runs everything
  in-process for development.

## Architecture

```
            you / an agent
   submit-goal · intervene · overview · takeover
                   │   (atomic SQLite tx + unix-socket wake)
                   ▼
┌──────────────────────────────────────────────────────────────┐
│  ResearchController  — deterministic reactor, 0 LLM tokens   │
│  propose ▸ fence (goal_version, human_epoch) ▸ lease ▸        │
│  dispatch ▸ reconcile receipts ▸ prune / promote / escalate  │
│        ResourceArbiter · ModelRouter · GoalManager           │
└────────┬──────────────────────┬──────────────────────────────┘
         │ single-attempt jobs  │ append-only evidence
         ▼                      ▼
┌──────────────────────┐  ┌───────────────────────────────────┐
│ HyperQueue / inline  │  │ SQLite WAL ledger                 │
│  StageRunner         │  │ goals · hypotheses · tasks ·      │
│  ├ git worktree      │  │ receipts · reviews · plans ·      │
│  ├ adapter command   │  │ interventions · leases · events   │
│  └ backend plugin ───┼──┼─► claude_cli · codex_cli ·        │
│     (model call)     │  │   gemini_cli · openai_compat ·    │
└──────────────────────┘  │   command · fake · <yours>        │
                          └───────────────────────────────────┘
```

A hypothesis moves through stages, each one a separate job:

```
PROPOSE → DESIGN_REVIEW → CODE → CPU_GATE → IMPL_REVIEW → EXP_SCREEN → EXP_CONFIRM → RESULT_REVIEW → RECONCILE
                                    │                           │
                          deterministic tests            F1 fast screen → F2 full confirm
```

More detail: [docs/architecture.md](docs/architecture.md) · [docs/concepts.md](docs/concepts.md)

## Five-minute quickstart

Requirements: Python ≥ 3.10, git. No API keys, no network, no HyperQueue needed for the demo.

```bash
git clone https://github.com/xixilys/conjectra && cd conjectra
pip install -e .            # optional; the demo also runs straight from the checkout
./examples/quickstart/run.sh
```

The script builds a toy repository whose test prints `METRIC score=<value>`, registers the goal
*"maximize score"*, runs the controller inline for ~20 s with a tiny offline plugin backend playing
the engineer and the reviewers, then shows the ledger and pauses the goal:

```
== 4. what happened
goal demo-score: status=ACTIVE receipts=49 tokens=0/100000 human_epoch=0
12 hypotheses; verdicts: {'CONFIRMED': 12}
  [     CONFIRMED] COMPETING_ROUTE  toy primary algorithmic improvement route
  [     CONFIRMED] PARAMETER_RACE   More Newton iterations
  [     CONFIRMED] ABLATION         Ablation of More Newton iterations
  [     CONFIRMED] FALSIFICATION    Falsification stress on More Newton iterations
  ...
recent evidence receipts by stage: {'CODE': 3, 'CPU_GATE': 5, 'IMPL_REVIEW': 1, 'EXP_SCREEN': 7, 'EXP_CONFIRM': 1, 'RESULT_REVIEW': 2, 'RECONCILE': 1}
takeover brief: 1453 bytes
== 5. human intervention: pause the goal
goal demo-score: status=PAUSED receipts=49 tokens=0/100000 human_epoch=1
```

### Using it on your own repository

```bash
conjectra init-pilot --no-ssh
conjectra submit-goal --file my_goal.json          # see examples/quickstart/goal.json
conjectra import-proposals --file proposals.json   # optional: your own hypotheses
conjectra daemon --mode inline --propose-mode seeds --max-seconds 600
conjectra overview
conjectra subgoals --goal-id my-goal
conjectra inspect --goal-id my-goal
conjectra intervene --scope goal --goal-id my-goal --action pause --reason "need to look"
conjectra takeover --goal-id my-goal --format markdown
```

For real model calls, install the CLIs you want to use (`claude`, `codex`, `gemini`) or point the
`openai_compat` backend at an endpoint, and drop `CONJECTRA_BACKEND_OVERRIDE`. For parallel
production runs install HyperQueue (`hq` on `PATH`) and use `--mode hq` (the default).

## Examples

| Example | What it shows |
|---|---|
| [`examples/quickstart`](examples/quickstart) | Full offline closed loop on a toy repo, HITL pause, takeover brief |
| [`examples/custom_backend`](examples/custom_backend) | Writing and registering a model backend plugin (code, env var, entry point) |
| [`examples/custom_adapter`](examples/custom_adapter) | A project adapter that seeds hypotheses and parses domain metrics |

## Documentation

* [Concepts](docs/concepts.md) — goals, hypotheses, stages, receipts, verdicts, failure domains, epochs
* [Architecture](docs/architecture.md) — components, data flow, fencing and recovery
* [Backends](docs/backends.md) — the model/execution backend plugin interface
* [Adapters](docs/adapters.md) — project adapters: seeds, commands, metric parsing
* [Configuration](docs/configuration.md) — environment variables, provider catalog, quota files
* [CLI reference](docs/cli.md)

## How it compares

Conjectra is a *control plane*, not an agent. It is meant to sit above the coding agents you
already use and below the human who owns the research question.

| | Conjectra | Agent frameworks (LangGraph, AutoGen, CrewAI, …) | Autonomous-research agents (e.g. AI-Scientist-style pipelines) | Workflow engines (Airflow, Prefect, Ray, HyperQueue) | Experiment trackers (MLflow, W&B) |
|---|---|---|---|---|---|
| Who decides what runs next | Deterministic controller, 0 tokens | An LLM in the loop | An LLM in the loop | Your DAG | You |
| Unit of work | Single-attempt, sandboxed job with pinned model route | Agent turn / tool call | Agent session | Task | Run |
| Scientific verdict semantics (confirmed / refuted / unknown, failure domains) | Built in | — | Partly, usually LLM-judged | — | — |
| Cross-family blind review of code and results | Built in | DIY | Rare | — | — |
| Many goals sharing one budget & worker pool | Built in | DIY | Usually one goal per run | Yes (generic) | — |
| Human interventions fenced against stale results | Built in (`human_epoch`) | DIY | — | Pause/kill | — |
| Model backends | Pluggable (CLI, HTTP, command, Python) | Pluggable | Usually fixed | n/a | n/a |

It is complementary to these tools: HyperQueue is the execution layer, any coding agent can be a
backend, and metrics can still be logged to your tracker.

## Project status & limitations

* Alpha; expect schema migrations between minor versions (they are additive and automatic so far).
* POSIX only (uses `fcntl`, process groups, unix sockets).
* The default provider catalog (`conjectra/data/providers.json`) lists example model ids; adjust
  them to what your CLIs and gateways actually accept.
* The natural-language policy compiler understands English and Chinese keywords.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please report security issues privately as described in
[SECURITY.md](SECURITY.md).

## License

[Apache License 2.0](LICENSE)
