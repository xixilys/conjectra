"""Example backend plugin: a rule-based reviewer + scripted builder, no model required.

* Read-only calls (reviews / audits): byte-compiles every ``*.py`` file under the reviewed
  snapshot and returns a structured verdict (FAIL with findings if anything does not compile).
* Write calls: appends a marker comment to the files listed in the prompt's
  ``ALLOWED writable paths ONLY:`` line (stand-in for a real coding agent).

Swap the body of ``invoke`` for an HTTP call or a subprocess to wrap a real model.
"""

from __future__ import annotations

import json
import py_compile
import re
import time
from pathlib import Path

from conjectra.backends import BackendRequest, BackendResult
from conjectra.types import ProcessStatus

__all__ = ["CompileCheckBackend"]


class CompileCheckBackend:
    name = "compile_check"

    def _review(self, root: Path) -> str:
        findings: list[str] = []
        for path in sorted(root.rglob("*.py")):
            if ".git" in path.parts:
                continue
            try:
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                findings.append(f"{path.relative_to(root)}: {exc.msg.strip().splitlines()[-1]}")
        verdict = "FAIL" if findings else "PASS"
        severity = "MAJOR" if findings else "NONE"
        return json.dumps({"verdict": verdict, "severity": severity, "findings": findings, "repro_python": ""})

    def _build(self, request: BackendRequest) -> str:
        m = re.search(r"ALLOWED writable paths ONLY:\s*(.+)", request.prompt)
        touched: list[str] = []
        for raw in (m.group(1).split(",") if m else []):
            rel = raw.strip()
            target = request.cwd / rel
            if rel and target.is_file() and target.suffix == ".py":
                target.write_text(target.read_text(encoding="utf-8") + "\n# touched by compile_check backend\n", encoding="utf-8")
                touched.append(rel)
        return f"compile_check backend touched: {touched or 'nothing'}"

    def invoke(self, request: BackendRequest) -> BackendResult:
        t0 = time.perf_counter()
        text = self._review(request.cwd) if request.readonly else self._build(request)
        request.out_md.parent.mkdir(parents=True, exist_ok=True)
        request.out_md.write_text(text, encoding="utf-8")
        request.usage_out.write_text(
            json.dumps({"route": self.name, "model": request.model, "usage": {"input_tokens": 0, "output_tokens": 0}}),
            encoding="utf-8",
        )
        return BackendResult(
            exit_code=0,
            stdout=text,
            wall_sec=time.perf_counter() - t0,
            status=ProcessStatus.EXIT_OK,
            command=f"{self.name}:{request.pool}:{request.model}",
        )
