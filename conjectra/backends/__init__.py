"""Pluggable model / execution backends. See ``conjectra.backends.base`` and ``docs/backends.md``."""

from conjectra.backends.base import (
    ENTRY_POINT_GROUP,
    Backend,
    BackendRequest,
    BackendResult,
    available_backends,
    get_backend,
    register_backend,
    unregister_backend,
)
from conjectra.backends.builtin import DEFAULT_BACKEND_FOR_POOL, FakeBackend

__all__ = [
    "DEFAULT_BACKEND_FOR_POOL",
    "ENTRY_POINT_GROUP",
    "Backend",
    "BackendRequest",
    "BackendResult",
    "FakeBackend",
    "available_backends",
    "get_backend",
    "register_backend",
    "unregister_backend",
]
