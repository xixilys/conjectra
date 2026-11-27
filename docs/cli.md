# CLI reference

Generated from `conjectra --help`. Every command accepts `--home DIR` (alias `--arr-home`).
Commands print JSON unless noted.

| Command | Purpose |
|---|---|
| [`init-pilot`](#init-pilot) | Initialize CONJECTRA_HOME ledger and optionally HQ |
| [`submit-goal`](#submit-goal) | Register an active ResearchGoal |
| [`step`](#step) | Run one deterministic controller step |
| [`daemon`](#daemon) | Run event-driven reactor daemon |
| [`notify-exit`](#notify-exit) | Wake reactor on task completion |
| [`status`](#status) | Show ledger, HQ, and escalation status |
| [`board`](#board) | Show hypothesis leaderboard and events |
| [`export-teamwork-brief`](#export-teamwork-brief) | Export brief for workflow/teamwork |
| [`import-proposals`](#import-proposals) | Import validated proposal JSON |
| [`benchmark-vs-baseline`](#benchmark-vs-baseline) | Compute machine-verified KPIs |
| [`sync-status`](#sync-status) | Write LATEST_STATUS.md & benchmark_latest.json and mirror them to $CONJECTRA_STATUS_MIRROR |
| [`boost`](#boost) | Clean-room cross-provider Builder <-> Blind Adversary ping-pong |
| [`xswarm`](#xswarm) | Clean-room multi-provider @name cursor swarm on HQ |
| [`xteamwork`](#xteamwork) | Clean-room multi-phase milestone teamwork with <4KB flush |
| [`route-workload`](#route-workload) | Classify task features and pick optimal mode from empirical ledger |
| [`eval-matrix`](#eval-matrix) | Run 4-class fault-seeded empirical mode ablation benchmark |
| [`overview`](#overview) | Sub-50ms multi-Goal overview from SQLite WAL |
| [`inspect`](#inspect) | Sub-50ms Goal/SubGoal/Hypothesis/Workflow inspection |
| [`subgoals`](#subgoals) | Sub-50ms SubGoal portfolio inspection from SQLite WAL |
| [`token-audit`](#token-audit) | Audit real uncached vs cache tokens per Goal |
| [`takeover`](#takeover) | Emit <3KB session-independent takeover brief |
| [`intervene`](#intervene) | Atomic HITL intervention + instant reactor.sock wake |
| [`set-policy`](#set-policy) | Compile & apply NL or structured V3 model routing policy |
| [`plans`](#plans) | Sub-50ms inspection of versioned Research/Experiment/Execution plans |
| [`reviews`](#reviews) | Sub-50ms inspection of 3-way Design/Impl/Result/Synthesis ReviewReceipts |
| [`status-snapshot`](#status-snapshot) | Export dashboard live status snapshot atomically |

## init-pilot

Initialize CONJECTRA_HOME ledger and optionally HQ

```
usage: conjectra init-pilot [-h] [--home DIR] [--start-hq]
                            [--baseline-path BASELINE_PATH] [--no-ssh]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --start-hq            Ensure HQ server and worker are running
  --baseline-path BASELINE_PATH
  --no-ssh
```

## submit-goal

Register an active ResearchGoal

```
usage: conjectra submit-goal [-h] [--home DIR] [--file FILE]
                             [--goal-id GOAL_ID] [--project PROJECT]
                             [--title TITLE] [--direction DIRECTION]
                             [--primary-metric PRIMARY_METRIC]
                             [--baseline-value BASELINE_VALUE]
                             [--baseline-sha BASELINE_SHA]
                             [--repo-path REPO_PATH]
                             [--writable-prefix WRITABLE_PREFIX]
                             [--banned-path BANNED_PATH]
                             [--kill-criterion KILL_CRITERION]
                             [--max-tokens-budget MAX_TOKENS_BUDGET]
                             [--max-concurrent-branches MAX_CONCURRENT_BRANCHES]
                             [--adapter ADAPTER]
                             [--autonomy-mode AUTONOMY_MODE]
                             [--priority-weight PRIORITY_WEIGHT]
                             [--min-guaranteed-slots MIN_GUARANTEED_SLOTS]
                             [--target-value TARGET_VALUE]
                             [--wall-budget-sec WALL_BUDGET_SEC]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --file FILE           Path to ResearchGoal JSON
  --goal-id GOAL_ID
  --project PROJECT
  --title TITLE
  --direction DIRECTION
  --primary-metric PRIMARY_METRIC
  --baseline-value BASELINE_VALUE
  --baseline-sha BASELINE_SHA
  --repo-path REPO_PATH
  --writable-prefix WRITABLE_PREFIX
  --banned-path BANNED_PATH
  --kill-criterion KILL_CRITERION
  --max-tokens-budget MAX_TOKENS_BUDGET
  --max-concurrent-branches MAX_CONCURRENT_BRANCHES
  --adapter ADAPTER
  --autonomy-mode AUTONOMY_MODE
  --priority-weight PRIORITY_WEIGHT
  --min-guaranteed-slots MIN_GUARANTEED_SLOTS
  --target-value TARGET_VALUE
  --wall-budget-sec WALL_BUDGET_SEC
```

## step

Run one deterministic controller step

```
usage: conjectra step [-h] [--home DIR] [--goal-id GOAL_ID]
                      [--mode {hq,inline}] [--propose-mode {auto,seeds,ai}]
                      [--dry-run-llm]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --mode {hq,inline}
  --propose-mode {auto,seeds,ai}
  --dry-run-llm
```

## daemon

Run event-driven reactor daemon

```
usage: conjectra daemon [-h] [--home DIR] [--goal-id GOAL_ID]
                        [--mode {hq,inline}] [--propose-mode {auto,seeds,ai}]
                        [--dry-run-llm] [--sweep-interval SWEEP_INTERVAL]
                        [--max-seconds MAX_SECONDS] [--background] [--detach]
                        [--notify-cmd NOTIFY_CMD] [--stop]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --mode {hq,inline}
  --propose-mode {auto,seeds,ai}
  --dry-run-llm
  --sweep-interval SWEEP_INTERVAL
  --max-seconds MAX_SECONDS
  --background
  --detach
  --notify-cmd NOTIFY_CMD
                        file:/path.jsonl or a command template ({subject}
                        substituted; body on stdin)
  --stop
```

## notify-exit

Wake reactor on task completion

```
usage: conjectra notify-exit [-h] [--home DIR] --task-id TASK_ID
                             [--exit-code EXIT_CODE] [--hq-job-id HQ_JOB_ID]
                             [--status STATUS]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --task-id TASK_ID
  --exit-code EXIT_CODE
  --hq-job-id HQ_JOB_ID
  --status STATUS
```

## status

Show ledger, HQ, and escalation status

```
usage: conjectra status [-h] [--home DIR] [--goal-id GOAL_ID]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
```

## board

Show hypothesis leaderboard and events

```
usage: conjectra board [-h] [--home DIR] [--goal-id GOAL_ID]
                       [--since-seq SINCE_SEQ] [--limit LIMIT]
                       [--format {json,text}]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --since-seq SINCE_SEQ
  --limit LIMIT
  --format {json,text}
```

## export-teamwork-brief

Export brief for workflow/teamwork

```
usage: conjectra export-teamwork-brief [-h] [--home DIR] [--goal-id GOAL_ID]
                                       [--format {json,markdown}]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --format {json,markdown}
```

## import-proposals

Import validated proposal JSON

```
usage: conjectra import-proposals [-h] [--home DIR] [--goal-id GOAL_ID] --file
                                  FILE [--min-evoi MIN_EVOI] [--no-enqueue]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --file FILE           JSON file path or '-' for stdin
  --min-evoi MIN_EVOI
  --no-enqueue
```

## benchmark-vs-baseline

Compute machine-verified KPIs

```
usage: conjectra benchmark-vs-baseline [-h] [--home DIR] [--goal-id GOAL_ID]
                                       [--baseline-path BASELINE_PATH]
                                       [--slots SLOTS] [--no-ssh]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --baseline-path BASELINE_PATH
  --slots SLOTS
  --no-ssh
```

## sync-status

Write LATEST_STATUS.md & benchmark_latest.json and mirror them to $CONJECTRA_STATUS_MIRROR

```
usage: conjectra sync-status [-h] [--home DIR] [--goal-id GOAL_ID]
                             [--baseline-path BASELINE_PATH] [--target TARGET]
                             [--no-ssh] [--dry-run]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --baseline-path BASELINE_PATH
  --target TARGET       Local dir or user@host:/dir (default:
                        $CONJECTRA_STATUS_MIRROR)
  --no-ssh
  --dry-run
```

## boost

Clean-room cross-provider Builder <-> Blind Adversary ping-pong

```
usage: conjectra boost [-h] [--home DIR] --repo-path REPO_PATH
                       [--project PROJECT] --spec SPEC --acceptance-cmd
                       ACCEPTANCE_CMD --writable-path WRITABLE_PATH
                       [--banned-path BANNED_PATH] [--builder BUILDER]
                       [--adversary ADVERSARY] [--max-rounds MAX_ROUNDS]
                       [--timeout TIMEOUT] [--in-place] [--detach]
                       [--notify-cmd NOTIFY_CMD]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --repo-path REPO_PATH
  --project PROJECT
  --spec SPEC           Task specification string or @file path
  --acceptance-cmd ACCEPTANCE_CMD
  --writable-path WRITABLE_PATH
  --banned-path BANNED_PATH
  --builder BUILDER     Provider spec e.g. claude:claude-sonnet-5-5:medium or
                        cmd:...
  --adversary ADVERSARY
                        Distinct-family provider, e.g. codex:gpt-5-codex:high
  --max-rounds MAX_ROUNDS
                        1 initial build + up to 2 repairs (default: 3)
  --timeout TIMEOUT
  --in-place
  --detach
  --notify-cmd NOTIFY_CMD
                        file:/path.jsonl or a command template ({subject}
                        substituted; body on stdin)
```

## xswarm

Clean-room multi-provider @name cursor swarm on HQ

```
usage: conjectra xswarm [-h] [--home DIR] --spec-file SPEC_FILE [--no-hq]
                        [--detach] [--notify-cmd NOTIFY_CMD]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --spec-file SPEC_FILE
                        JSON config with repo_path, project, title, workers,
                        verifier_cmd
  --no-hq
  --detach
  --notify-cmd NOTIFY_CMD
                        file:/path.jsonl or a command template ({subject}
                        substituted; body on stdin)
```

## xteamwork

Clean-room multi-phase milestone teamwork with <4KB flush

```
usage: conjectra xteamwork [-h] [--home DIR] --spec-file SPEC_FILE [--detach]
                           [--notify-cmd NOTIFY_CMD]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --spec-file SPEC_FILE
                        JSON config with repo_path, project, title, phases,
                        final_acceptance_cmd
  --detach
  --notify-cmd NOTIFY_CMD
                        file:/path.jsonl or a command template ({subject}
                        substituted; body on stdin)
```

## route-workload

Classify task features and pick optimal mode from empirical ledger

```
usage: conjectra route-workload [-h] [--home DIR] --description DESCRIPTION
                                [--scope SCOPE] [--readonly]
                                [--coupled-interface] [--phases PHASES]
                                [--minutes MINUTES]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --description DESCRIPTION
  --scope SCOPE         Comma-separated writable paths per slice (repeatable)
  --readonly
  --coupled-interface
  --phases PHASES
  --minutes MINUTES
```

## eval-matrix

Run 4-class fault-seeded empirical mode ablation benchmark

```
usage: conjectra eval-matrix [-h] [--home DIR] [--detach]
                             [--notify-cmd NOTIFY_CMD]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --detach
  --notify-cmd NOTIFY_CMD
                        file:/path.jsonl or a command template ({subject}
                        substituted; body on stdin)
```

## overview

Sub-50ms multi-Goal overview from SQLite WAL

```
usage: conjectra overview [-h] [--home DIR] [--total-slots TOTAL_SLOTS]
                          [--provider PROVIDER]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --total-slots TOTAL_SLOTS
  --provider PROVIDER
```

## inspect

Sub-50ms Goal/SubGoal/Hypothesis/Workflow inspection

```
usage: conjectra inspect [-h] [--home DIR] [--goal-id GOAL_ID]
                         [--subgoal-id SUBGOAL_ID]
                         [--hypothesis-id HYPOTHESIS_ID]
                         [--workflow-id WORKFLOW_ID] [--provider PROVIDER]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --subgoal-id SUBGOAL_ID
  --hypothesis-id HYPOTHESIS_ID
  --workflow-id WORKFLOW_ID
  --provider PROVIDER
```

## subgoals

Sub-50ms SubGoal portfolio inspection from SQLite WAL

```
usage: conjectra subgoals [-h] [--home DIR] [--goal-id GOAL_ID]
                          [--status STATUS] [--provider PROVIDER]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --status STATUS
  --provider PROVIDER
```

## token-audit

Audit real uncached vs cache tokens per Goal

```
usage: conjectra token-audit [-h] [--home DIR] [--goal-id GOAL_ID]
                             [--provider PROVIDER]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --provider PROVIDER
```

## takeover

Emit <3KB session-independent takeover brief

```
usage: conjectra takeover [-h] [--home DIR] [--goal-id GOAL_ID]
                          [--format {json,markdown}] [--max-bytes MAX_BYTES]
                          [--provider PROVIDER]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --format {json,markdown}
  --max-bytes MAX_BYTES
  --provider PROVIDER
```

## intervene

Atomic HITL intervention + instant reactor.sock wake

```
usage: conjectra intervene [-h] [--home DIR] --scope
                           {global,goal,subgoal,branch,workflow,task} --action
                           ACTION [--goal-id GOAL_ID] [--target-id TARGET_ID]
                           [--payload PAYLOAD] [--actor ACTOR]
                           [--reason REASON] [--no-wake] [--provider PROVIDER]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --scope {global,goal,subgoal,branch,workflow,task}
  --action ACTION
  --goal-id GOAL_ID
  --target-id TARGET_ID
  --payload PAYLOAD     JSON object string or @path
  --actor ACTOR
  --reason REASON
  --no-wake
  --provider PROVIDER
```

## set-policy

Compile & apply NL or structured V3 model routing policy

```
usage: conjectra set-policy [-h] [--home DIR]
                            [--scope {global,goal,branch,workflow,task}]
                            [--goal-id GOAL_ID] [--target-id TARGET_ID]
                            --policy POLICY [--project PROJECT]
                            [--actor ACTOR] [--reason REASON] [--no-wake]
                            [--provider PROVIDER]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --scope {global,goal,branch,workflow,task}
  --goal-id GOAL_ID
  --target-id TARGET_ID
  --policy POLICY       Chinese/English NL policy string, JSON object, or
                        @file
  --project PROJECT
  --actor ACTOR
  --reason REASON
  --no-wake
  --provider PROVIDER
```

## plans

Sub-50ms inspection of versioned Research/Experiment/Execution plans

```
usage: conjectra plans [-h] [--home DIR] [--goal-id GOAL_ID]
                       [--hypothesis-id HYPOTHESIS_ID] [--provider PROVIDER]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --hypothesis-id HYPOTHESIS_ID
  --provider PROVIDER
```

## reviews

Sub-50ms inspection of 3-way Design/Impl/Result/Synthesis ReviewReceipts

```
usage: conjectra reviews [-h] [--home DIR] [--goal-id GOAL_ID]
                         [--hypothesis-id HYPOTHESIS_ID]
                         [--workflow-id WORKFLOW_ID]
                         [--review-kind {DESIGN_REVIEW,IMPLEMENTATION_REVIEW,RESULT_REVIEW,SYNTHESIS_REVIEW}]
                         [--provider PROVIDER]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
  --goal-id GOAL_ID
  --hypothesis-id HYPOTHESIS_ID
  --workflow-id WORKFLOW_ID
  --review-kind {DESIGN_REVIEW,IMPLEMENTATION_REVIEW,RESULT_REVIEW,SYNTHESIS_REVIEW}
  --provider PROVIDER
```

## status-snapshot

Export dashboard live status snapshot atomically

```
usage: conjectra status-snapshot [-h] [--home DIR]

options:
  -h, --help            show this help message and exit
  --home DIR, --arr-home DIR
                        State directory (default: $CONJECTRA_HOME or
                        ~/.conjectra)
```

