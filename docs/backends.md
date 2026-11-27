# Model / execution backends

A **backend** executes one routed model call. The controller never calls a model itself: it pins
a `ResolvedModelRoute` to a task, the stage runner builds the prompt, and the backend named by the
route's `backend_mode` runs it inside the task's sandboxed worktree.

Backends are deliberately small: one method, process-style result, files on disk. That makes it
easy to wrap a CLI agent, an HTTP API, a browser-automation tool, or a local model runner.

## The interface

```python
from conjectra.backends import BackendRequest, BackendResult


class MyBackend:
    def invoke(self, request: BackendRequest) -> BackendResult:
        ...
```

`BackendRequest` (all fields are plain data):

| Field | Meaning |
|---|---|
| `pool`, `model`, `effort` | The pinned route: logical pool (`claude`, `codex`, `gemini`, `deepseek`, …), canonical model id, wire effort (`low` … `max`). |
| `prompt` | The full prompt text. |
| `cwd` | Working directory: an isolated git worktree (write calls) or a read-only snapshot (reviews). |
| `readonly` | `True` for reviews/audits. Your backend **must not** modify files when this is set. |
| `timeout_s` | Hard wall-clock budget. Return `ProcessStatus.TIMEOUT` (exit code 124) if exceeded. |
| `out_md` | Write the model's final answer here (Markdown/plain text). Reviews are parsed from it. |
| `usage_out` | Write a JSON usage record here, e.g. `{"usage": {"input_tokens": 1200, "output_tokens": 300}}`. |
| `tag` | Stable identifier of the call (task id + step), handy for logs and scripted fakes. |
| `route` | The full `ResolvedModelRoute` (family, host, rule id, …) if you need more context. |
| `dispatch_script`, `dispatch_host_arg` | Only used by the `dispatch_script` backend. |

`BackendResult`: `exit_code`, `stdout`, `stderr`, `wall_sec`, `max_rss_mb`, `status`
(`conjectra.types.ProcessStatus`), `command` (a printable description for the receipt).

What the runner does with your result:

* **Write calls (`CODE`)** — whatever you changed in `cwd` is committed on the hypothesis branch and
  then has to pass `CPU_GATE` and review. Exit code ≠ 0 is an engineering failure, never a
  scientific refutation.
* **Read-only calls (reviews, audits)** — `out_md` must contain a verdict. The most robust format
  is a JSON object: `{"verdict": "PASS" | "FAIL" | "BROKEN", "severity": "...", "findings": [...],
  "repro_python": ""}`. A standalone final line such as `AUDIT_VERDICT: PASS` (or `DESIGN_VERDICT`,
  `IMPL_VERDICT`, `RESULT_VERDICT`) also works.
  Ambiguous or contradictory output becomes `PARSE_ERROR`, never a false `PASS`.
* **Token accounting** — numbers in `usage_out` feed per-goal budgets and the token audit.

## Built-in backends

| `backend_mode` | Default for pools | What it runs | Configuration |
|---|---|---|---|
| `claude_cli` | `claude` | `claude -p --model M --effort E --output-format json` (read-only: `--tools Read,Grep,Glob`) | `CONJECTRA_CLAUDE_BIN` |
| `codex_cli` | `codex` | `codex exec -m M -c model_reasoning_effort=E --json -` (read-only: empty scratch dir as writable root) | `CONJECTRA_CODEX_BIN` |
| `gemini_cli` | `gemini` | `gemini --model M --output-format json -p …` (read-only: `--approval-mode plan`) | `CONJECTRA_GEMINI_BIN` |
| `openai_compat` | `deepseek`, `grok`, `deepseek-local` | `POST {base}/chat/completions` — text only, so route it to review/planning roles | `CONJECTRA_<POOL>_BASE_URL`, `CONJECTRA_<POOL>_API_KEY` (fallback `CONJECTRA_OPENAI_*`) |
| `command` / `webchat` | `cmd`, `webchat` | any CLI: prompt on stdin, answer on stdout | `CONJECTRA_<POOL>_CMD="tool --model {model} --effort {effort}"` |
| `dispatch_script` | anything else | an external dispatcher script (contract below) | `CONJECTRA_DISPATCH_SCRIPT` |
| `fake` | `fake` | deterministic offline replies (PASS verdict for reviews, no-op for writes) | `CONJECTRA_FAKE_RESPONSES=path.json` (`{substring: reply}`) |

The CLI flags above match the tools at the time of writing; if your installed version differs,
subclass the backend or use `command`.

`CONJECTRA_BACKEND_OVERRIDE=<name>` sends **every** call to one backend — useful for CI
(`fake`) or for a gateway that fronts all providers.

A provider entry in the catalog can also pin a backend explicitly with `"backend": "<name>"`
(see [configuration.md](configuration.md)).

## Registering your own backend

Pick whichever fits your deployment; later registrations win.

**1. In code** (tests, notebooks, embedding Conjectra in another service):

```python
from conjectra.backends import register_backend
register_backend("mybackend", MyBackend)        # class, factory, or instance
```

**2. Environment variable** (no packaging needed):

```bash
export CONJECTRA_BACKENDS="mybackend=my_pkg.my_module:MyBackend"
```

**3. Entry point** (recommended for distributable plugins), in the plugin's `pyproject.toml`:

```toml
[project.entry-points."conjectra.backends"]
mybackend = "my_pkg.my_module:MyBackend"
```

Then route to it, either for everything (`CONJECTRA_BACKEND_OVERRIDE=mybackend`), or per provider
by adding a catalog entry with `"pool": "mypool", "backend": "mybackend"` and referring to it in a
routing policy (`"spec": "mypool:my-model:high"`).

`conjectra.backends.available_backends()` lists what is registered.

### Example: wrapping a web-chat automation tool

Tools that drive a chat web UI usually expose a CLI that reads a prompt and prints the answer.
That is exactly the `command` contract, so no Python is required:

```bash
export CONJECTRA_WEBCHAT_CMD="my-webchat-cli run --site chatgpt --model {model}"
```

and the `webchat` pool (catalog entry `webchat-gpt-xhigh`) will use it. For richer integration —
streaming, session reuse, attachments — write a backend class and register it under `webchat`.

A complete, runnable plugin lives in [`examples/custom_backend`](../examples/custom_backend).

## The `dispatch_script` contract

For setups that already have a multi-provider dispatcher, Conjectra can call it instead of a
backend class:

```
python3 <script> <pool> --host <host> --model <model> --cwd <dir> \
        --out-md <file> --usage-out <file> --timeout <sec> --tag <tag> [--effort <e>] [--readonly]
```

The prompt arrives on stdin. The script writes the final text to `--out-md`, a usage JSON to
`--usage-out`, and exits 0 on success.

## Testing a backend

```python
from pathlib import Path
from conjectra.backends import BackendRequest

req = BackendRequest(pool="mypool", model="m", effort="high", prompt="Say PASS",
                     cwd=Path("."), out_md=Path("out.md"), usage_out=Path("usage.json"),
                     timeout_s=30, tag="smoke", readonly=True)
print(MyBackend().invoke(req))
```

For end-to-end checks run the quickstart with your backend:
`CONJECTRA_BACKENDS=... CONJECTRA_BACKEND_OVERRIDE=mybackend ./examples/quickstart/run.sh`.
