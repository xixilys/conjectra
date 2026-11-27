"""A 12-line custom backend: plays the 'engineer' by editing the toy repo, and PASSes reviews.

Registered for the demo via:
    CONJECTRA_BACKENDS="toy=toy_backend:ToyEditBackend"  CONJECTRA_BACKEND_OVERRIDE=toy
Real deployments register Claude Code / Codex / Gemini CLIs, an OpenAI-compatible endpoint,
or your own plugin the same way (see docs/backends.md).
"""

from __future__ import annotations

import re

from conjectra.backends import BackendRequest, BackendResult, FakeBackend


class ToyEditBackend(FakeBackend):
    def invoke(self, request: BackendRequest) -> BackendResult:
        if not request.readonly:
            target = request.cwd / "src" / "score.py"
            if target.is_file():
                src = target.read_text(encoding="utf-8")
                target.write_text(re.sub(r"iterations: int = \d+", "iterations: int = 4", src), encoding="utf-8")
        return super().invoke(request)
