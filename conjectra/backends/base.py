"""Model / execution backend plugin interface.

A *backend* turns one routed LLM call (pool + model + effort + prompt, executed inside a
sandboxed worktree) into a process-style result. The controller never talks to a model
directly: every call is a single attempt, executed by a stage runner, and the backend is
looked up by the route's ``backend_mode`` string.

Third-party backends can be registered three ways:

1. In code: ``register_backend("mybackend", MyBackend())``.
2. Via the ``CONJECTRA_BACKENDS`` env var: ``"mybackend=my_pkg.mod:MyBackend,other=pkg:factory"``.
3. Via the ``conjectra.backends`` entry-point group in your package's ``pyproject.toml``::

       [project.entry-points."conjectra.backends"]
       mybackend = "my_pkg.mod:MyBackend"

The entry-point target may be a class (instantiated with no arguments), a zero-argument
factory, or an already-constructed instance. It must expose ``invoke(request) -> BackendResult``.
"""

from __future__ import annotations

import importlib
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from conjectra.types import ProcessStatus

ENTRY_POINT_GROUP = "conjectra.backends"


@dataclass
class BackendRequest:
    """One single-attempt model invocation."""

    pool: str
    model: str
    effort: str
    prompt: str
    cwd: Path
    out_md: Path
    usage_out: Path
    timeout_s: int
    tag: str
    readonly: bool
    dispatch_script: Path | None = None
    dispatch_host_arg: str | None = None
    route: Any = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendResult:
    """Process-style outcome. ``out_md`` / ``usage_out`` should be written by the backend."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    wall_sec: float = 0.0
    max_rss_mb: float = 0.0
    status: ProcessStatus = ProcessStatus.EXIT_OK
    command: str = ""

    def as_tuple(self) -> tuple[int, str, str, float, float, ProcessStatus, str]:
        return (
            self.exit_code,
            self.stdout,
            self.stderr,
            self.wall_sec,
            self.max_rss_mb,
            self.status,
            self.command,
        )


@runtime_checkable
class Backend(Protocol):
    """Anything with an ``invoke(request) -> BackendResult`` method."""

    def invoke(self, request: BackendRequest) -> BackendResult: ...


_REGISTRY: dict[str, Backend] = {}
_LOCK = threading.Lock()
_DISCOVERED = False


def _instantiate(target: Any) -> Backend:
    obj = target
    if isinstance(target, type):
        obj = target()
    elif callable(target) and not hasattr(target, "invoke"):
        obj = target()
    if not hasattr(obj, "invoke"):
        raise TypeError(f"backend {target!r} does not expose invoke(request)")
    return obj  # type: ignore[return-value]


def _load_target(spec: str) -> Any:
    mod_name, _, attr = spec.partition(":")
    mod = importlib.import_module(mod_name.strip())
    return getattr(mod, attr.strip()) if attr else mod


def register_backend(name: str, backend: Backend | type | Callable[[], Backend]) -> None:
    """Register (or replace) a backend under ``name``."""
    key = str(name).strip().lower()
    if not key:
        raise ValueError("backend name must be non-empty")
    inst = _instantiate(backend)
    with _LOCK:
        _REGISTRY[key] = inst


def unregister_backend(name: str) -> None:
    with _LOCK:
        _REGISTRY.pop(str(name).strip().lower(), None)


def _discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return
    _DISCOVERED = True
    from conjectra.backends import builtin

    for key, factory in builtin.BUILTIN_BACKENDS.items():
        _REGISTRY.setdefault(key, factory())
    try:
        from importlib.metadata import entry_points

        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                _REGISTRY[ep.name.lower()] = _instantiate(ep.load())
            except Exception:  # pragma: no cover - broken third-party plugin must not kill the runtime
                continue
    except Exception:  # pragma: no cover
        pass
    raw = os.environ.get("CONJECTRA_BACKENDS", "")
    for item in raw.split(","):
        name, sep, spec = item.partition("=")
        if not sep or not name.strip() or not spec.strip():
            continue
        try:
            _REGISTRY[name.strip().lower()] = _instantiate(_load_target(spec))
        except Exception:
            continue


def get_backend(name: str) -> Backend | None:
    with _LOCK:
        _discover()
        return _REGISTRY.get(str(name or "").strip().lower())


def available_backends() -> list[str]:
    with _LOCK:
        _discover()
        return sorted(_REGISTRY)
