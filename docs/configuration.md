# Configuration

Conjectra has no config file of its own: state lives in one directory, and everything else is an
environment variable or a JSON file you point it at.

## State home

`--home DIR` (alias `--arr-home`) or `CONJECTRA_HOME` (default `~/.conjectra`) holds:

| Path | Content |
|---|---|
| `state.db` (+ `-wal`, `-shm`) | the SQLite WAL ledger |
| `events.jsonl` | append-only event log |
| `artifacts/` | per-task logs, model outputs, usage JSON, experiment artifacts |
| `logs/` | daemon, HyperQueue server/worker and job logs |
| `hq-server/` | HyperQueue server directory (mode `hq`) |
| `reactor.sock`, `controller.pid` | daemon wake socket and pid file |
| `status_snapshot.json`, `LATEST_STATUS.md`, `benchmark_latest.json` | status exports |

## Environment variables

### Paths and execution

| Variable | Default | Purpose |
|---|---|---|
| `CONJECTRA_HOME` | `~/.conjectra` | state home |
| `CONJECTRA_SANDBOX_ROOT` / `CONJECTRA_WORKTREE_ROOT` | `~/.conjectra/sandboxes`, `~/.conjectra/arr-wt` | git mirrors and per-branch worktrees |
| `CONJECTRA_SCRATCH_ROOT` | unset | optional fast disk; used for sandboxes while below 85 % full |
| `CONJECTRA_HQ_BIN` | `hq` on `PATH` or `~/.local/bin/hq` | HyperQueue binary |
| `CONJECTRA_EXECUTION_HOST` | `local` | logical host label used by provider host restrictions |
| `CONJECTRA_LOCK_DIR`, `CONJECTRA_HEAVY_SLOTS` | `<tmp>/conjectra-locks`, `1` | heavy-job slot lock (`python -m conjectra.slot_lock`) |
| `CONJECTRA_DISK_MOUNTS` | `/` | extra `:`-separated mounts the arbiter keeps below the disk high-water mark |

### Host guards

| Variable | Default | Purpose |
|---|---|---|
| `CONJECTRA_MIN_AVAIL_RAM_MB` | `10240` | test/experiment stages wait for this much `MemAvailable`. **Lower it on laptops / small CI runners.** |
| `CONJECTRA_DISK_MAX_PCT` | `85` | no new stage is admitted while a watched mount (see `CONJECTRA_DISK_MOUNTS`) is at least this full |
| `CONJECTRA_MIN_GPU_VRAM_MB` | `1500` | free VRAM required before GPU stages start |

### Models and backends

| Variable | Purpose |
|---|---|
| `CONJECTRA_PROVIDERS` | path to your provider catalog (default: the packaged `conjectra/data/providers.json`) |
| `CONJECTRA_BACKEND_OVERRIDE` | send every model call to one backend (`fake` in CI) |
| `CONJECTRA_BACKENDS` | register plugins: `name=module:attr,name2=module:attr` |
| `CONJECTRA_CLAUDE_BIN`, `CONJECTRA_CODEX_BIN`, `CONJECTRA_GEMINI_BIN` | override CLI executables |
| `CONJECTRA_<POOL>_BASE_URL`, `CONJECTRA_<POOL>_API_KEY` | endpoint for `openai_compat` pools (`DEEPSEEK`, `GROK`, …); fallback `CONJECTRA_OPENAI_BASE_URL` / `_API_KEY` |
| `CONJECTRA_<POOL>_CMD` | argv template for `command` pools, e.g. `CONJECTRA_WEBCHAT_CMD` |
| `CONJECTRA_DISPATCH_SCRIPT` | external dispatcher for the `dispatch_script` backend |
| `CONJECTRA_FAKE_RESPONSES` | JSON `{substring: reply}` table for the `fake` backend |
| `CONJECTRA_BANNED_POOLS` | comma-separated substrings never allowed in a pool/model spec (default `gemini:claude`) |
| `CONJECTRA_HOST_POOL_BLOCKS` | `host=pool1,pool2;host2=pool3` — logical pools a host may not use |

### Guardrails

| Variable | Purpose |
|---|---|
| `CONJECTRA_READ_BANS` | Markdown table whose first column lists extra banned paths/globs (default `~/.conjectra/READ_BANS.md`) |
| `CONJECTRA_PROTECTED_PATHS` | comma-separated path fragments no goal or proposal may target |
| `CONJECTRA_INTERNAL_ID_PATTERNS` | comma-separated regexes for ticket ids that must not leak into deliverable code |
| `CONJECTRA_SPECIALIZATION_PATTERNS` | comma-separated regexes for benchmark/device-specific hacks to reject |
| `CONJECTRA_ADAPTERS` | register project adapters: `name=module:attr` |

### Reporting

| Variable | Purpose |
|---|---|
| `CONJECTRA_BASELINE_PATH` | frozen baseline JSON used by `benchmark-vs-baseline` |
| `CONJECTRA_BASELINE_REMOTE` | `user@host:/path` to fetch the baseline over ssh |
| `CONJECTRA_STATUS_MIRROR` | directory or `user@host:/dir` that `sync-status` mirrors status files to |

## Provider catalog

`conjectra/data/providers.json` maps provider keys to pools, families and the efforts each backend
accepts. Copy it, edit it, and set `CONJECTRA_PROVIDERS`:

```jsonc
{
  "providers": {
    "my-model": {
      "canonical_model": "my-model-2025-01",   // what the backend receives
      "pool": "mypool",                         // logical pool (used by policies and quotas)
      "logical_pool": "llm_mypool",
      "hq_resource": "llmmypool",               // HyperQueue resource consumed per call (or null)
      "family": "acme",                         // reviewers avoid the author's family
      "backend": "mybackend",                   // optional: pin a registered backend
      "allowed_execution_hosts": ["local"],
      "native_cli_efforts": ["low", "medium", "high"],
      "multipool_argparse_efforts": ["low", "medium", "high"],
      "policy_allowed_efforts": ["low", "medium", "high"],
      "multipool_effort_map": {"xhigh": "high", "max": "high"},
      "default_effort": "high",
      "aliases": ["my", "acme-model"]
    }
  },
  "hq_resource_mapping": {"llm_mypool": "llmmypool"},
  "default_role_routing": {"impl_reviewer": ["my-model:high", "gpt-5-codex:high"]},
  "banned_channels": []
}
```

Optional `effort_model_ids` (`{"low": "<id>", "high": "<id>"}`) covers CLIs that encode the
reasoning effort in the model id.

The default catalog applies one opinionated cost guard: `claude-sonnet-*` is capped at `medium`
effort (`BLOCKED_SONNET_CAP` under `HARD` policies, silently clamped otherwise).

## Routing policies

Per goal (`submit-goal`, `intervene --action set_routing_policy`) or globally (`set-policy`).
Structured form:

```json
{"rules": [
  {"rule_id": "theory", "tags": ["theory"], "spec": "claude:claude-opus-5-5:xhigh", "strength": "HARD"},
  {"rule_id": "default", "spec": "gpt-5-codex:high", "strength": "SOFT_FALLBACK"}
]}
```

Natural-language form (English or Chinese keywords), compiled to the same rules:

```
theory work uses Opus xhigh; routine analysis uses gpt-5-codex high;
implementation uses Sonnet medium or a low-cost model; major theory requires cross-family review
```

Specificity order: tag + role > tag > role > default. `HARD` rules wait for the resource
(`WAIT_RESOURCE`) instead of downgrading.

## Quota files (optional)

The arbiter keeps pools above a high-water mark of their 7-day quota and reserves headroom for
interactive use. Without quota files every pool is treated as available. Accepted formats (all under
the state home by default):

* `quotas.json` — `{"<provider>/<account>": {"quota": {"windows": [{"name": "7 day", "used": 42.0}]}}}`
  where `<provider>` is `claude`, `codex`/`openai`, `deepseek`, …
* `quota-snapshot.json` — `[{"provider": "claude", "at": "2026-01-01T00:00:00Z", "windows": [...]}]`
  (entries older than 6 h are ignored)
* `gemini-quota.json` — `{"pools": {"gemini": {"weekly": {"remaining_fraction": 0.7}}}}`
* `spend-ledger.json` — daily pay-as-you-go spend for the disabled-by-default `deepseek-local` pool
