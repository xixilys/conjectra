"""Built-in backends: ``fake``, ``claude_cli``, ``codex_cli``, ``gemini_cli``, ``openai_compat``,
``command`` and ``dispatch_script``.

Every backend writes the model's final text to ``request.out_md`` and a JSON usage record to
``request.usage_out``; the stage runner turns both into an ``EvidenceReceipt``.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable

from conjectra.backends.base import BackendRequest, BackendResult
from conjectra.backends.proc import run_subprocess
from conjectra.types import ProcessStatus

READONLY_PROMPT_NOTE = (
    "READ-ONLY AUDIT MODE: Inspect files and git history only. "
    "NEVER modify, create, delete, or commit any repository files."
)


def _write_usage(req: BackendRequest, payload: dict[str, Any]) -> None:
    req.usage_out.parent.mkdir(parents=True, exist_ok=True)
    req.usage_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_text(req: BackendRequest, text: str) -> None:
    if text:
        req.out_md.parent.mkdir(parents=True, exist_ok=True)
        req.out_md.write_text(text, encoding="utf-8")


def _exe(name: str, env_var: str, *fallbacks: str) -> str:
    override = os.environ.get(env_var)
    if override:
        return override
    found = shutil.which(name)
    if found:
        return found
    for fb in fallbacks:
        cand = os.path.expanduser(fb)
        if os.path.exists(cand):
            return cand
    return name


# ---------------------------------------------------------------------------
# fake
# ---------------------------------------------------------------------------

FAKE_PASS_VERDICT = json.dumps(
    {"verdict": "PASS", "severity": "NONE", "findings": [], "repro_python": ""}
)


class FakeBackend:
    """Deterministic, offline backend for tests, demos and CI.

    * read-only calls (reviews / audits) return a structured PASS verdict;
    * write calls return a short note and change nothing;
    * ``responder`` (or ``CONJECTRA_FAKE_RESPONSES`` = path to a JSON ``{substring: reply}`` map)
      lets you script replies by matching on the tag or the prompt.
    """

    def __init__(self, responder: Callable[[BackendRequest], str] | None = None) -> None:
        self.responder = responder
        self.calls: list[BackendRequest] = []

    def _scripted(self, req: BackendRequest) -> str | None:
        path = os.environ.get("CONJECTRA_FAKE_RESPONSES")
        if not path:
            return None
        try:
            table = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not isinstance(table, dict):
            return None
        for key, reply in table.items():
            if key and (key in req.tag or key in req.prompt):
                return str(reply)
        return None

    def invoke(self, request: BackendRequest) -> BackendResult:
        self.calls.append(request)
        t0 = time.perf_counter()
        if self.responder is not None:
            text = self.responder(request)
        else:
            text = self._scripted(request)
            if text is None:
                text = FAKE_PASS_VERDICT if request.readonly else "fake backend: no changes made"
        _write_text(request, text)
        _write_usage(
            request,
            {
                "route": "fake",
                "host": "local",
                "pool": request.pool,
                "model": request.model,
                "effort": request.effort,
                "returncode": 0,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        )
        return BackendResult(
            exit_code=0,
            stdout=text,
            wall_sec=time.perf_counter() - t0,
            status=ProcessStatus.EXIT_OK,
            command=f"fake:{request.pool}:{request.model}",
        )


# ---------------------------------------------------------------------------
# claude_cli  (Claude Code: `claude -p`)
# ---------------------------------------------------------------------------


def parse_claude_json(stdout: str) -> tuple[str, dict[str, Any] | None, bool]:
    """Parse `claude -p --output-format json` payload into `(result_text, raw_json, is_error)`."""
    for line in reversed((stdout or "").splitlines()):
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            data = json.loads(s)
        except Exception:
            continue
        if isinstance(data, dict):
            return str(data.get("result") or ""), data, bool(data.get("is_error"))
    return "", None, True


class ClaudeCLIBackend:
    def invoke(self, request: BackendRequest) -> BackendResult:
        exe = _exe("claude", "CONJECTRA_CLAUDE_BIN", "~/.local/bin/claude")
        if request.readonly:
            ro_args = [
                "--dangerously-skip-permissions",
                "--tools",
                "Read,Grep,Glob",
                "--append-system-prompt",
                READONLY_PROMPT_NOTE,
            ]
        else:
            ro_args = ["--dangerously-skip-permissions"]
        cmd = [
            exe,
            "-p",
            "--model",
            request.model,
            "--effort",
            request.effort,
            "--strict-mcp-config",
            "--output-format",
            "json",
            *ro_args,
        ]
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        rc, out, err, wall, rss, status = run_subprocess(
            cmd, cwd=request.cwd, timeout_s=request.timeout_s, stdin_text=request.prompt
        )
        text, data, is_err = parse_claude_json(out)
        if rc == 0 and is_err:
            rc, status = 1, ProcessStatus.EXIT_NONZERO
        _write_text(request, text)
        _write_usage(
            request,
            {
                "route": "claude_cli",
                "host": "local",
                "model": request.model,
                "effort": request.effort,
                "returncode": rc,
                "usage": (data or {}).get("usage"),
                "modelUsage": (data or {}).get("modelUsage"),
            },
        )
        return BackendResult(rc, out, err, wall, rss, status, cmd_str)


# ---------------------------------------------------------------------------
# codex_cli  (OpenAI Codex CLI: `codex exec --json`)
# ---------------------------------------------------------------------------


def parse_codex_jsonl(stdout: str) -> tuple[str, dict[str, int]]:
    """Extract final agent_message text and token usage from `codex exec --json` JSONL."""
    messages: list[str] = []
    usage: dict[str, int] = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0}
    for line in (stdout or "").splitlines():
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            ev = json.loads(s)
        except Exception:
            continue
        if not isinstance(ev, dict):
            continue
        ev_type = ev.get("type")
        if ev_type == "item.completed":
            item = ev.get("item")
            if isinstance(item, dict) and item.get("type") == "agent_message":
                txt = item.get("text")
                if isinstance(txt, str) and txt.strip():
                    messages.append(txt)
        elif ev_type == "turn.completed":
            u = ev.get("usage")
            if isinstance(u, dict):
                usage["input_tokens"] += int(u.get("input_tokens") or 0)
                in_det = u.get("input_tokens_details") if isinstance(u.get("input_tokens_details"), dict) else {}
                usage["cached_input_tokens"] += int(
                    u.get("cached_input_tokens") or in_det.get("cached_tokens") or 0
                )
                usage["output_tokens"] += int(u.get("output_tokens") or 0)
    return "\n\n".join(messages), usage


class CodexCLIBackend:
    def invoke(self, request: BackendRequest) -> BackendResult:
        exe = _exe("codex", "CONJECTRA_CODEX_BIN", "~/.npm-global/bin/codex")
        ro_scratch: Path | None = None
        extra_env: dict[str, str] | None = None
        try:
            base = [
                exe,
                "exec",
                "--skip-git-repo-check",
                "-m",
                request.model,
                "-c",
                f'model_reasoning_effort="{request.effort}"',
                "-s",
                "workspace-write",
            ]
            if request.readonly:
                # Reviews run against an empty scratch dir as the writable root so the
                # reviewer physically cannot edit the repository under review.
                ro_scratch = Path(tempfile.gettempdir()) / f"conjectra_codex_ro_{int(time.time())}_{uuid.uuid4().hex[:6]}"
                ro_scratch.mkdir(parents=True, exist_ok=True)
                extra_env = {"TMPDIR": str(ro_scratch)}
                cmd = [*base, "-C", str(ro_scratch), "--add-dir", str(ro_scratch), "--json", "-"]
            else:
                cmd = [*base, "-C", str(request.cwd), "--json", "-"]
            cmd_str = " ".join(shlex.quote(x) for x in cmd)
            rc, out, err, wall, rss, status = run_subprocess(
                cmd,
                cwd=request.cwd,
                timeout_s=request.timeout_s,
                stdin_text=request.prompt,
                extra_env=extra_env,
            )
            text, usage = parse_codex_jsonl(out)
            _write_text(request, text)
            _write_usage(
                request,
                {
                    "route": "codex_cli",
                    "host": "local",
                    "model": request.model,
                    "effort": request.effort,
                    "returncode": rc,
                    "usage": usage,
                },
            )
            return BackendResult(rc, out, err, wall, rss, status, cmd_str)
        finally:
            if ro_scratch is not None:
                shutil.rmtree(ro_scratch, ignore_errors=True)


# ---------------------------------------------------------------------------
# gemini_cli  (Gemini CLI: `gemini -p ... --output-format json`)
# ---------------------------------------------------------------------------


def parse_gemini_json(stdout: str) -> tuple[str, dict[str, Any] | None, str | None]:
    """Parse a Gemini-CLI style JSON payload into `(response_text, raw_json, error_msg)`."""
    data: dict[str, Any] | None = None
    for line in reversed((stdout or "").splitlines()):
        s = line.strip()
        if not s.startswith("{"):
            continue
        try:
            obj = json.loads(s)
        except Exception:
            continue
        if isinstance(obj, dict):
            data = obj
            break
    if data is None:
        try:
            obj = json.loads(stdout or "")
            data = obj if isinstance(obj, dict) else None
        except ValueError:
            data = None
    if data is None:
        return "", None, "No JSON payload returned by gemini"
    status = str(data.get("status") or "SUCCESS").upper()
    if status != "SUCCESS" or data.get("error"):
        return "", data, str(data.get("error") or f"gemini status={status}")
    return str(data.get("response") or ""), data, None


class GeminiCLIBackend:
    def invoke(self, request: BackendRequest) -> BackendResult:
        exe = _exe("gemini", "CONJECTRA_GEMINI_BIN", "~/.local/bin/gemini")
        prompt = request.prompt + (f"\n\n[{READONLY_PROMPT_NOTE}]\n" if request.readonly else "")
        mode_flags = ["--approval-mode", "plan"] if request.readonly else ["--yolo"]
        cmd = [exe, "--model", request.model, "--output-format", "json", *mode_flags]
        cmd_str = " ".join(shlex.quote(x) for x in [*cmd, "-p", "<prompt>"])
        rc, out, err, wall, rss, status = run_subprocess(
            [*cmd, "-p", prompt], cwd=request.cwd, timeout_s=request.timeout_s
        )
        text, data, g_err = parse_gemini_json(out)
        if rc == 0 and g_err is not None:
            rc, status = 1, ProcessStatus.EXIT_NONZERO
            err = f"{err}\n{g_err}".strip()
        _write_text(request, text)
        _write_usage(
            request,
            {
                "route": "gemini_cli",
                "host": "local",
                "model": request.model,
                "effort": request.effort,
                "returncode": rc,
                "usage": (data or {}).get("usage") or (data or {}).get("stats"),
            },
        )
        return BackendResult(rc, out, err, wall, rss, status, cmd_str)


# ---------------------------------------------------------------------------
# openai_compat  (any OpenAI-compatible /chat/completions endpoint; text-only)
# ---------------------------------------------------------------------------


class OpenAICompatBackend:
    """Text-only chat-completions backend (DeepSeek, xAI, vLLM, Ollama, LiteLLM, ...).

    Configure per pool with ``CONJECTRA_<POOL>_BASE_URL`` / ``CONJECTRA_<POOL>_API_KEY``
    (pool upper-cased, ``-`` -> ``_``), falling back to ``CONJECTRA_OPENAI_BASE_URL`` /
    ``CONJECTRA_OPENAI_API_KEY``. It cannot edit files, so route it to review / planning roles.
    """

    def _cfg(self, pool: str) -> tuple[str, str]:
        key = pool.upper().replace("-", "_")
        base = os.environ.get(f"CONJECTRA_{key}_BASE_URL") or os.environ.get("CONJECTRA_OPENAI_BASE_URL") or ""
        api_key = os.environ.get(f"CONJECTRA_{key}_API_KEY") or os.environ.get("CONJECTRA_OPENAI_API_KEY") or ""
        return base.rstrip("/"), api_key

    def invoke(self, request: BackendRequest) -> BackendResult:
        base, api_key = self._cfg(request.pool)
        cmd_str = f"POST {base or '<unset>'}/chat/completions model={request.model}"
        if not base:
            msg = f"openai_compat: no base URL configured for pool {request.pool!r}"
            _write_usage(request, {"route": "openai_compat", "returncode": 127, "error": msg})
            return BackendResult(127, "", msg, 0.0, 0.0, ProcessStatus.ENV_ERROR, cmd_str)
        prompt = request.prompt + (f"\n\n[{READONLY_PROMPT_NOTE}]\n" if request.readonly else "")
        body: dict[str, Any] = {"model": request.model, "messages": [{"role": "user", "content": prompt}]}
        if request.effort and request.effort not in ("default", "custom", "off"):
            body["reasoning_effort"] = request.effort
        http_req = urllib.request.Request(
            f"{base}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})},
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(http_req, timeout=max(1, int(request.timeout_s))) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
        except TimeoutError as exc:
            return BackendResult(124, "", str(exc), time.perf_counter() - t0, 0.0, ProcessStatus.TIMEOUT, cmd_str)
        except (urllib.error.URLError, OSError) as exc:
            _write_usage(request, {"route": "openai_compat", "returncode": 1, "error": str(exc)})
            return BackendResult(1, "", str(exc), time.perf_counter() - t0, 0.0, ProcessStatus.EXIT_NONZERO, cmd_str)
        wall = time.perf_counter() - t0
        try:
            doc = json.loads(raw)
            text = str(doc["choices"][0]["message"]["content"] or "")
            usage = doc.get("usage") or {}
        except (ValueError, KeyError, IndexError, TypeError):
            return BackendResult(1, raw, "unparseable chat-completions response", wall, 0.0, ProcessStatus.EXIT_NONZERO, cmd_str)
        _write_text(request, text)
        _write_usage(
            request,
            {
                "route": "openai_compat",
                "host": "local",
                "pool": request.pool,
                "model": request.model,
                "effort": request.effort,
                "returncode": 0,
                "usage": {
                    "input_tokens": int(usage.get("prompt_tokens") or 0),
                    "output_tokens": int(usage.get("completion_tokens") or 0),
                },
            },
        )
        return BackendResult(0, raw, "", wall, 0.0, ProcessStatus.EXIT_OK, cmd_str)


# ---------------------------------------------------------------------------
# command  (generic argv template; prompt on stdin, final text on stdout)
# ---------------------------------------------------------------------------


class CommandBackend:
    """Run any CLI that reads the prompt on stdin and prints the answer on stdout.

    Configure with ``CONJECTRA_<POOL>_CMD`` (shell-quoted argv; ``{model}``, ``{effort}``,
    ``{cwd}``, ``{readonly}`` are substituted). Useful for wrapping web-UI automation tools,
    local model runners or in-house gateways without writing Python.
    """

    def invoke(self, request: BackendRequest) -> BackendResult:
        key = request.pool.upper().replace("-", "_")
        template = os.environ.get(f"CONJECTRA_{key}_CMD") or os.environ.get("CONJECTRA_COMMAND_CMD") or ""
        if not template:
            msg = f"command backend: set CONJECTRA_{key}_CMD for pool {request.pool!r}"
            return BackendResult(127, "", msg, 0.0, 0.0, ProcessStatus.ENV_ERROR, "")
        subst = {
            "model": request.model,
            "effort": request.effort,
            "cwd": str(request.cwd),
            "readonly": "1" if request.readonly else "0",
        }
        argv = [tok.format(**subst) for tok in shlex.split(template)]
        cmd_str = " ".join(shlex.quote(x) for x in argv)
        prompt = request.prompt + (f"\n\n[{READONLY_PROMPT_NOTE}]\n" if request.readonly else "")
        rc, out, err, wall, rss, status = run_subprocess(
            argv, cwd=request.cwd, timeout_s=request.timeout_s, stdin_text=prompt
        )
        _write_text(request, out.strip())
        _write_usage(
            request,
            {"route": "command", "pool": request.pool, "model": request.model, "returncode": rc, "usage": {}},
        )
        return BackendResult(rc, out, err, wall, rss, status, cmd_str)


# ---------------------------------------------------------------------------
# dispatch_script  (external dispatcher with a fixed CLI contract)
# ---------------------------------------------------------------------------


def build_dispatch_cmd(
    script_path: Path,
    pool: str,
    model: str,
    cwd: Path,
    out_md: Path,
    usage_out: Path,
    timeout_s: int,
    tag: str,
    *,
    readonly: bool,
    effort: str | None = None,
    host_arg: str | None = None,
) -> list[str]:
    """argv for ``<script> <pool> --host H --model M --cwd D --out-md F --usage-out U --timeout S --tag T``."""
    cmd = [
        sys.executable or "python3",
        str(script_path),
        pool,
        "--host",
        host_arg or "local",
        "--model",
        model,
        "--cwd",
        str(cwd),
        "--out-md",
        str(out_md),
        "--usage-out",
        str(usage_out),
        "--timeout",
        str(max(30, int(timeout_s))),
        "--tag",
        tag,
    ]
    if effort:
        cmd.extend(["--effort", str(effort)])
    if readonly:
        cmd.append("--readonly")
    return cmd


class DispatchScriptBackend:
    """Delegate to an external dispatcher script (see ``docs/backends.md`` for the contract)."""

    def invoke(self, request: BackendRequest) -> BackendResult:
        script = request.dispatch_script
        if script is None or not Path(script).exists():
            msg = (
                "dispatch_script backend: no dispatcher script found "
                f"({script}); set CONJECTRA_DISPATCH_SCRIPT or pick another backend"
            )
            return BackendResult(127, "", msg, 0.0, 0.0, ProcessStatus.ENV_ERROR, "")
        cmd = build_dispatch_cmd(
            Path(script),
            request.pool,
            request.model,
            request.cwd,
            request.out_md,
            request.usage_out,
            request.timeout_s,
            request.tag,
            readonly=request.readonly,
            effort=request.effort,
            host_arg=request.dispatch_host_arg,
        )
        cmd_str = " ".join(shlex.quote(x) for x in cmd)
        rc, out, err, wall, rss, status = run_subprocess(
            cmd, cwd=request.cwd, timeout_s=request.timeout_s, stdin_text=request.prompt
        )
        return BackendResult(rc, out, err, wall, rss, status, cmd_str)


BUILTIN_BACKENDS: dict[str, Callable[[], Any]] = {
    "fake": FakeBackend,
    "claude_cli": ClaudeCLIBackend,
    "codex_cli": CodexCLIBackend,
    "gemini_cli": GeminiCLIBackend,
    "openai_compat": OpenAICompatBackend,
    "command": CommandBackend,
    "webchat": CommandBackend,
    "dispatch_script": DispatchScriptBackend,
}

DEFAULT_BACKEND_FOR_POOL: dict[str, str] = {
    "claude": "claude_cli",
    "codex": "codex_cli",
    "gemini": "gemini_cli",
    "deepseek": "openai_compat",
    "deepseek-local": "openai_compat",
    "grok": "openai_compat",
    "webchat": "webchat",
    "fake": "fake",
    "cmd": "command",
}
