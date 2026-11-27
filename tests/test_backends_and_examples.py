"""Tests for the backend plugin registry, built-in offline backends, and the shipped examples."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from conjectra import backends as be
from conjectra.backends import BackendRequest, FakeBackend, get_backend, register_backend
from conjectra.backends import base as be_base
from conjectra.types import (
    EvidenceValidity,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ScientificVerdict,
    StageKind,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _req(tmp_path: Path, *, readonly: bool, prompt: str = "hello", pool: str = "fake", tag: str = "t1") -> BackendRequest:
    return BackendRequest(
        pool=pool,
        model="m",
        effort="high",
        prompt=prompt,
        cwd=tmp_path,
        out_md=tmp_path / "out" / "answer.md",
        usage_out=tmp_path / "out" / "usage.json",
        timeout_s=30,
        tag=tag,
        readonly=readonly,
    )


def test_builtin_backends_are_registered() -> None:
    names = be.available_backends()
    for expected in ("fake", "claude_cli", "codex_cli", "gemini_cli", "openai_compat", "command", "dispatch_script"):
        assert expected in names


def test_fake_backend_review_pass_and_scripted_responses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fb = FakeBackend()
    res = fb.invoke(_req(tmp_path, readonly=True))
    assert res.exit_code == 0 and res.status == ProcessStatus.EXIT_OK
    assert json.loads((tmp_path / "out" / "answer.md").read_text())["verdict"] == "PASS"
    assert json.loads((tmp_path / "out" / "usage.json").read_text())["route"] == "fake"

    table = tmp_path / "fake.json"
    table.write_text(json.dumps({"design": "DESIGN_VERDICT: FAIL"}), encoding="utf-8")
    monkeypatch.setenv("CONJECTRA_FAKE_RESPONSES", str(table))
    FakeBackend().invoke(_req(tmp_path, readonly=True, tag="design-review-1"))
    assert (tmp_path / "out" / "answer.md").read_text() == "DESIGN_VERDICT: FAIL"


def test_register_backend_in_code_and_via_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class Recorder:
        def invoke(self, request: BackendRequest):  # noqa: ANN201
            calls.append(request.tag)
            return be.BackendResult(exit_code=0, command="recorder")

    register_backend("recorder", Recorder)
    try:
        assert get_backend("recorder").invoke(_req(tmp_path, readonly=False)).command == "recorder"
        assert calls == ["t1"]
    finally:
        be.unregister_backend("recorder")

    monkeypatch.syspath_prepend(str(REPO_ROOT / "examples" / "custom_backend"))
    monkeypatch.setenv("CONJECTRA_BACKENDS", "compile_check=conjectra_echo_backend:CompileCheckBackend")
    monkeypatch.setattr(be_base, "_DISCOVERED", False)
    assert get_backend("compile_check") is not None
    be.unregister_backend("compile_check")


def test_command_backend_and_missing_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cmd_be = get_backend("command")
    res_missing = cmd_be.invoke(_req(tmp_path, readonly=True, pool="mytool"))
    assert res_missing.exit_code == 127 and res_missing.status == ProcessStatus.ENV_ERROR

    monkeypatch.setenv("CONJECTRA_MYTOOL_CMD", f"{sys.executable} -c \"import sys; print('echo:' + sys.stdin.read()[:5])\"")
    res = cmd_be.invoke(_req(tmp_path, readonly=False, pool="mytool", prompt="abcdefgh"))
    assert res.exit_code == 0
    assert (tmp_path / "out" / "answer.md").read_text() == "echo:abcde"


def test_openai_compat_without_endpoint_is_env_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONJECTRA_OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("CONJECTRA_NOPE_BASE_URL", raising=False)
    res = get_backend("openai_compat").invoke(_req(tmp_path, readonly=True, pool="nope"))
    assert res.exit_code == 127 and res.status == ProcessStatus.ENV_ERROR


def test_example_compile_check_backend_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "examples" / "custom_backend"))
    from conjectra_echo_backend import CompileCheckBackend

    (tmp_path / "ok.py").write_text("x = 1\n", encoding="utf-8")
    res = CompileCheckBackend().invoke(_req(tmp_path, readonly=True))
    assert json.loads(res.stdout)["verdict"] == "PASS"

    (tmp_path / "bad.py").write_text("def broken(:\n", encoding="utf-8")
    res_bad = CompileCheckBackend().invoke(_req(tmp_path, readonly=True))
    doc = json.loads(res_bad.stdout)
    assert doc["verdict"] == "FAIL" and any("bad.py" in f for f in doc["findings"])


def test_example_csv_adapter_turns_results_into_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.syspath_prepend(str(REPO_ROOT / "examples" / "custom_adapter"))
    from conjectra_csv_adapter import CsvMetricsAdapter

    goal = ResearchGoal(
        goal_id="g-csv",
        project="csvproj",
        title="Lower validation loss",
        direction="minimize",
        primary_metric="val_loss",
        baseline_value=0.50,
        baseline_sha="0" * 40,
        repo_path=str(tmp_path),
        kill_criteria=["tests_passed == 0"],
        banned_paths=["sealed_eval/"],
        writable_prefixes=["src"],
        max_tokens_budget=10_000,
        adapter_name="csv_metrics",
    )
    adapter = CsvMetricsAdapter()
    seeds = adapter.seed_hypotheses(goal)
    assert {h.kind for h in seeds} == {HypothesisKind.PARAMETER_RACE, HypothesisKind.COMPETING_ROUTE}

    hyp = seeds[1]
    _unused = HypothesisNode(
        hypothesis_id="h1",
        goal_id=goal.goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="t",
        statement="s",
        mechanism="m",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.0,
        writable_paths=["src"],
        params={},
        kill_criteria=["tests_passed == 0"],
        acceptance_cmd="pytest -q",
        screen_cmd="python3 experiment.py",
        confirm_cmd="python3 experiment.py",
        branch_name="arr/csvproj/h1",
        base_sha="0" * 40,
    )
    assert _unused.kind == hyp.kind
    art = tmp_path / "art"
    art.mkdir()
    (art / "results.csv").write_text("metric,value,n\nval_loss,0.41,200\n", encoding="utf-8")
    metrics, validity, verdict, artifacts, _ = adapter.parse_metrics_with_validity(
        StageKind.EXP_SCREEN, "", "", tmp_path, art, 0, hyp, goal
    )
    assert metrics["val_loss"]["value"] == pytest.approx(0.41)
    assert validity == EvidenceValidity.VALID_EMPIRICAL
    assert verdict == ScientificVerdict.CONFIRMED
    assert artifacts == [str(art / "results.csv")]

    (art / "results.csv").write_text("metric,value,n\nval_loss,0.66,200\n", encoding="utf-8")
    _, validity2, verdict2, _, _ = adapter.parse_metrics_with_validity(
        StageKind.EXP_SCREEN, "", "", tmp_path, art, 0, hyp, goal
    )
    assert validity2 == EvidenceValidity.VALID_EMPIRICAL
    assert verdict2 == ScientificVerdict.VALID_NEGATIVE
