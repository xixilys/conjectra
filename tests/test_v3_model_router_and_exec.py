"""Tests for V3 ModelRouter, NL policy compiler, quota snapshot, direct CLI backends, RO snapshots, CPU_GATE wall, and FailureDomain isolation."""

from __future__ import annotations

import datetime as dt
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

from typing import Any

import pytest

from conjectra.adapters import GenericAdapter, get_adapter
from conjectra.arbiter import ResourceArbiter, record_foreground_ping
from conjectra.hq_runtime import HQRuntime, resource_spec_to_requests
from conjectra.model_router import ModelRouter, PolicyBlockedError
from conjectra.multiagent import invoke_provider, parse_provider_spec
from conjectra.planning import PlanLifecycleManager, classify_review_tier
from conjectra.stage_runner import SandboxManager, StageRunner
from conjectra.state_db import StateDB
from conjectra.types import (
    AuditVerdict,
    ConstraintStrength,
    EngineeringVerdict,
    EvidenceValidity,
    FailureDomain,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ResourceSpec,
    ReviewKind,
    ReviewTier,
    ScientificRole,
    ScientificVerdict,
    StageKind,
    StageTask,
)


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def _init_git_repo(repo: Path) -> str:
    repo.mkdir(parents=True, exist_ok=True)
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "tester")
    _git(repo, "config", "user.email", "tester@local")
    (repo / "src").mkdir()
    (repo / "src" / "core.py").write_text("def compute(x: int) -> int:\n    return x * 2\n", encoding="utf-8")
    (repo / "src" / "test_core.py").write_text(
        "from src.core import compute\n\ndef test_compute():\n    assert compute(3) == 6\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial")
    return _git(repo, "rev-parse", "HEAD")


def _make_goal(
    repo: Path,
    base_sha: str,
    *,
    goal_id: str = "g-v3",
    project: str = "projgamma",
    primary_metric: str = "val_loss",
    direction: str = "minimize",
    baseline_value: float = 1.0,
    routing_policy_nl: str = "",
    routing_policy: dict | None = None,
) -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project=project,
        title="V3 router & execution test goal",
        direction=direction,
        primary_metric=primary_metric,
        baseline_value=baseline_value,
        baseline_sha=base_sha,
        repo_path=str(repo),
        kill_criteria=["tests_passed == 0", "val_loss > 5.0"],
        banned_paths=["lockbox/", "sealed_eval/"],
        writable_prefixes=["src/"],
        max_tokens_budget=200_000,
        routing_policy_nl=routing_policy_nl,
        routing_policy=routing_policy or {},
    )


def _make_hyp(
    goal_id: str,
    base_sha: str,
    *,
    hyp_id: str = "h-v3",
    tags: list[str] | None = None,
    acceptance_cmd: str = "pytest -q src/test_core.py",
    screen_cmd: str = "echo METRIC val_loss=0.8 denom=100",
) -> HypothesisNode:
    return HypothesisNode(
        hypothesis_id=hyp_id,
        goal_id=goal_id,
        parent_id=None,
        kind=HypothesisKind.COMPETING_ROUTE,
        title="Test hypothesis",
        statement="Improve val_loss via bounded change",
        mechanism="core scaling",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=1.2,
        writable_paths=["src/core.py"],
        params={},
        kill_criteria=["tests_passed == 0"],
        acceptance_cmd=acceptance_cmd,
        screen_cmd=screen_cmd,
        confirm_cmd=screen_cmd,
        branch_name=f"arr/projgamma/{hyp_id}",
        base_sha=base_sha,
        tags=tags or [],
    )


def _local_only_gemini_caps(tmp_path: Path) -> Path:
    """Copy the default provider catalog with Gemini restricted to the `local` execution host."""
    from conjectra.model_router import DEFAULT_CAPABILITIES_PATH

    doc = json.loads(Path(DEFAULT_CAPABILITIES_PATH).read_text(encoding="utf-8"))
    for rec in doc["providers"].values():
        if rec.get("pool") == "gemini":
            rec["allowed_execution_hosts"] = ["local"]
            rec["blocked_execution_hosts"] = ["remote"]
    out = tmp_path / "providers_local_gemini.json"
    out.write_text(json.dumps(doc), encoding="utf-8")
    return out


# ------------------------------------ 1. NL Policy Compiler & Specificity Rank


def test_nl_policy_compilation_projgamma_benchmark_and_cross_family_review(tmp_path: Path) -> None:
    nl = (
        "PROJGAMMA 的理论创新方向使用 Opus xhigh，"
        "Backbone 创新允许使用 gpt-5-pro xhigh，"
        "常规科研分析使用 gpt-5-codex high，"
        "普通实现使用 Sonnet medium 或低成本模型，"
        "重大理论方案必须经过另一模型家族审查"
    )
    compiled = ModelRouter.compile_routing_policy(nl, project="projgamma")
    assert compiled["review_overrides"]["theory"] == ReviewTier.RED_TEAM_TIEBREAKER.value
    assert len(compiled["rules"]) == 4

    arb = ResourceArbiter(
        quotas_path=tmp_path / "no_q.json",
        ledger_path=tmp_path / "no_l.json",
        gemini_quota_path=tmp_path / "no_v.json",
    )
    router = ModelRouter(arbiter=arb)
    goal = _make_goal(tmp_path, "deadbeef", routing_policy_nl=nl, routing_policy=compiled)

    # 1. Theory tag -> Opus xhigh HARD on native Claude CLI
    hyp_theory = _make_hyp(goal.goal_id, "deadbeef", hyp_id="h-theory", tags=["theory"])
    assert classify_review_tier(goal, hyp_theory, compiled_policy=compiled) == ReviewTier.RED_TEAM_TIEBREAKER

    r_theory = router.resolve_route(
        ScientificRole.THEORIST, goal=goal, hyp=hyp_theory, execution_host="local"
    )
    assert r_theory.canonical_model == "claude-opus-5-5"
    assert r_theory.requested_effort == "xhigh"
    assert r_theory.wire_effort == "xhigh"
    assert r_theory.backend_mode == "claude_cli"
    assert r_theory.strength == ConstraintStrength.HARD
    assert r_theory.hq_resource == "llmclaude"

    # Cross-family reviewer on `theory` hypothesis MUST NOT deadlock on author's Opus HARD rule!
    r_rev1 = router.resolve_route(
        ScientificRole.DESIGN_REVIEWER,
        goal=goal,
        hyp=hyp_theory,
        avoid_families=[r_theory.family],
        execution_host="local",
    )
    assert r_rev1.family != "anthropic"
    assert r_rev1.canonical_model == "gpt-5-codex"

    r_rev2 = router.resolve_route(
        ScientificRole.DESIGN_REVIEWER,
        goal=goal,
        hyp=hyp_theory,
        avoid_families=[r_theory.family, r_rev1.family],
        execution_host="local",
    )
    assert r_rev2.family not in {"anthropic", r_rev1.family}
    assert r_rev2.canonical_model == "gemini-pro"

    # 2. Backbone tag -> gpt-5-pro xhigh via codex exec
    hyp_bb = _make_hyp(goal.goal_id, "deadbeef", hyp_id="h-bb", tags=["backbone"])
    r_bb = router.resolve_route(
        ScientificRole.THEORIST, goal=goal, hyp=hyp_bb, execution_host="local"
    )
    assert r_bb.canonical_model == "gpt-5-pro"
    assert r_bb.requested_effort == "xhigh"
    assert r_bb.wire_effort == "xhigh"
    assert r_bb.backend_mode == "codex_cli"
    assert r_bb.strength == ConstraintStrength.SOFT_FALLBACK

    # 3. Routine ResearchLead analysis -> gpt-5-codex high
    hyp_routine = _make_hyp(goal.goal_id, "deadbeef", hyp_id="h-routine", tags=[])
    r_lead = router.resolve_route(
        ScientificRole.RESEARCH_LEAD, goal=goal, hyp=hyp_routine, execution_host="local"
    )
    assert r_lead.canonical_model == "gpt-5-codex"
    assert r_lead.wire_effort == "high"

    # 4. Routine EngineerBuilder -> Sonnet medium; falls back to low-cost DeepSeek when Claude 7d exhausted
    r_bld = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER, goal=goal, hyp=hyp_routine, execution_host="local"
    )
    assert r_bld.canonical_model == "claude-sonnet-5-5"
    assert r_bld.wire_effort == "medium"

    q_exhausted = tmp_path / "quotas_claude_full.json"
    q_exhausted.write_text(
        json.dumps(
            {
                "claude/acc1": {
                    "quota": {"windows": [{"name": "Claude 7 days", "used": 98.0}]}
                }
            }
        ),
        encoding="utf-8",
    )
    arb_full = ResourceArbiter(
        quotas_path=q_exhausted,
        ledger_path=tmp_path / "no_l.json",
        gemini_quota_path=tmp_path / "no_v.json",
    )
    router_full = ModelRouter(arbiter=arb_full)
    # Routine builder SOFT_FALLBACK falls back to low-cost candidate
    r_bld_fb = router_full.resolve_route(
        ScientificRole.ENGINEER_BUILDER, goal=goal, hyp=hyp_routine, execution_host="local"
    )
    assert r_bld_fb.canonical_model == "deepseek-chat"
    assert r_bld_fb.wire_effort == "high"

    # Theory HARD rule MUST pause with WAIT_RESOURCE (waitable=True, never silently downgrade Opus xhigh!)
    with pytest.raises(PolicyBlockedError) as exc_quota:
        router_full.resolve_route(
            ScientificRole.THEORIST, goal=goal, hyp=hyp_theory, execution_host="local"
        )
    assert exc_quota.value.code == "WAIT_RESOURCE"
    assert exc_quota.value.waitable is True
    assert exc_quota.value.pinned_route is not None
    assert exc_quota.value.pinned_route["canonical_model"] == "claude-opus-5-5"
    assert exc_quota.value.pinned_route["wire_effort"] == "xhigh"


def test_specificity_precedence_and_policy_compile_boundary_errors(tmp_path: Path) -> None:
    # Empty / None boundary inputs & natural English prose clauses without false-positive effort errors
    assert ModelRouter.compile_routing_policy(None)["rules"] == []
    assert ModelRouter.compile_routing_policy("")["rules"] == []
    assert ModelRouter.compile_routing_policy({})["rules"] == []
    eng_prose = ModelRouter.compile_routing_policy(
        "use Sonnet as fallback, use gpt-5-codex by default, use gpt-5-pro unless busy"
    )
    assert len(eng_prose["rules"]) == 3

    # Invalid / unknown HARD model, unknown effort modifier, banned channel, conflicting HARD rules
    with pytest.raises(ValueError, match="No known provider|Unparseable"):
        ModelRouter.compile_routing_policy("核心方向必须使用 ImaginaryGPT high")

    with pytest.raises(ValueError, match="effort|turbo"):
        ModelRouter.compile_routing_policy("理论方向必须使用 Opus turbo")

    with pytest.raises(ValueError, match="Banned channel"):
        ModelRouter.compile_routing_policy("使用 untrusted-relay opus high")

    with pytest.raises(ValueError, match="Conflicting HARD"):
        ModelRouter.compile_routing_policy(
            {
                "rules": [
                    {"rule_id": "r1", "roles": ["theorist"], "tags": ["theory"], "spec": "opus:xhigh", "strength": "HARD"},
                    {"rule_id": "r2", "roles": ["theorist"], "tags": ["theory"], "spec": "gpt-5-pro:xhigh", "strength": "HARD"},
                ]
            }
        )

    # Specificity rank: (1) explicit_spec > (2) role+tag > (3) tag_only > (4) role_only > (5) goal_default
    policy = ModelRouter.compile_routing_policy(
        {
            "rules": [
                {"rule_id": "rank5_default", "spec": "gemini-flash:high", "strength": "SOFT_FALLBACK"},
                {"rule_id": "rank4_role", "roles": ["theorist"], "spec": "gpt-5-codex:high", "strength": "SOFT_FALLBACK"},
                {"rule_id": "rank3_tag", "tags": ["theory"], "spec": "gpt-5-pro:xhigh", "strength": "SOFT_FALLBACK"},
                {
                    "rule_id": "rank2_role_and_tag",
                    "roles": ["theorist"],
                    "tags": ["theory"],
                    "spec": "opus:xhigh",
                    "strength": "HARD",
                },
            ]
        }
    )
    router = ModelRouter(
        arbiter=ResourceArbiter(
            quotas_path=tmp_path / "q.json",
            ledger_path=tmp_path / "l.json",
            gemini_quota_path=tmp_path / "v.json",
        )
    )
    goal = _make_goal(tmp_path, "deadbeef", routing_policy=policy)
    hyp_th = _make_hyp(goal.goal_id, "deadbeef", hyp_id="h-th", tags=["theory"])
    hyp_plain = _make_hyp(goal.goal_id, "deadbeef", hyp_id="h-pl", tags=[])

    # Rank 1: explicit_spec beats Rank 2
    r1 = router.resolve_route(
        ScientificRole.THEORIST,
        goal=goal,
        hyp=hyp_th,
        explicit_spec="gemini:gemini-pro:xhigh",
        execution_host="local",
    )
    assert r1.matched_rule_id == "explicit_spec"
    assert r1.canonical_model == "gemini-pro-xhigh"
    assert r1.wire_effort == "xhigh"

    # Rank 2: (role AND tag) beats Rank 3 & Rank 4
    r2 = router.resolve_route(ScientificRole.THEORIST, goal=goal, hyp=hyp_th, execution_host="local")
    assert r2.matched_rule_id == "rank2_role_and_tag"
    assert r2.canonical_model == "claude-opus-5-5"

    # Rank 3: tag_only beats Rank 5 for research_lead on theory tag
    r3 = router.resolve_route(ScientificRole.RESEARCH_LEAD, goal=goal, hyp=hyp_th, execution_host="local")
    assert r3.matched_rule_id == "rank3_tag"
    assert r3.canonical_model == "gpt-5-pro"

    # Rank 4: role_only beats Rank 5 on untagged hypothesis
    r4 = router.resolve_route(ScientificRole.THEORIST, goal=goal, hyp=hyp_plain, execution_host="local")
    assert r4.matched_rule_id == "rank4_role"
    assert r4.canonical_model == "gpt-5-codex"

    # Rank 5: goal_default matches untagged engineer_builder
    r5 = router.resolve_route(ScientificRole.ENGINEER_BUILDER, goal=goal, hyp=hyp_plain, execution_host="local")
    assert r5.matched_rule_id == "rank5_default"
    assert r5.canonical_model == "gemini-flash"


def test_capability_caps_foreground_reserve_and_zero_raw_resource_model_bug(tmp_path: Path) -> None:
    arr_home = tmp_path / "arr_home"
    q_snap = tmp_path / "quota-snapshot.json"
    now_iso = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    stale_iso = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=5)).strftime("%Y-%m-%dT%H:%M:%SZ")
    q_snap.write_text(
        json.dumps(
            [
                {
                    "provider": "claude",
                    "plan": "MAX 5X",
                    "windows": [
                        {"name": "Claude 7 days", "used": 91.0, "resetsAt": None},
                        {"name": "Codex 7 days", "used": 99.0, "resetsAt": None},
                    ],
                    "at": now_iso,
                },
                {
                    "provider": "claude",
                    "plan": "MAX 5X",
                    "windows": [{"name": "Claude 7 days", "used": 87.0, "resetsAt": None}],
                    "at": now_iso,
                },
                {
                    "provider": "codex",
                    "plan": "pro",
                    "windows": [{"name": "7 days", "used": 22.5, "resetsAt": None}],
                    "at": now_iso,
                },
                {
                    "provider": "deepseek",
                    "plan": "Design Plan Go",
                    "windows": [{"name": "7 days", "used": 14.0, "resetsAt": None}],
                    "at": now_iso,
                },
                {
                    "provider": "stale-bridge",
                    "plan": "ultra",
                    "windows": [{"name": "7 days", "used": 100.0, "resetsAt": None}],
                    "at": stale_iso,
                },
            ]
        ),
        encoding="utf-8",
    )
    arb = ResourceArbiter(
        quotas_path=tmp_path / "missing_q.json",
        quota_snapshot_path=q_snap,
        ledger_path=tmp_path / "missing_l.json",
        gemini_quota_path=tmp_path / "missing_v.json",
    )
    headroom = arb.read_pool_headroom()
    assert headroom["llm_claude"]["used_pct"] == 87.0
    assert headroom["llm_claude"]["source"] == "quota_snapshot"
    assert headroom["llm_codex_audit"]["used_pct"] == 22.5
    assert headroom["llm_deepseek"]["used_pct"] == 14.0
    # Stale stale-bridge must NEVER mark independent Gemini unavailable
    assert headroom["llm_gemini"]["available"] is True

    router = ModelRouter(arbiter=arb, arr_home=arr_home, foreground_reserve_quota_pct=15.0)

    # Sonnet > medium strictly blocked under HARD
    with pytest.raises(PolicyBlockedError) as exc_sonnet:
        router.resolve_route("engineer_builder", explicit_spec="claude:claude-sonnet-5-5:high", execution_host="local")
    assert exc_sonnet.value.code == "BLOCKED_SONNET_CAP"

    # gemini-ultra xhigh blocked under HARD (gemini-ultra supports low|medium|high only)
    with pytest.raises(PolicyBlockedError):
        router.resolve_route("theorist", explicit_spec="gemini:gemini-ultra:xhigh", execution_host="local")

    # deepseek medium blocked under HARD, mapped to wire_effort="high" under SOFT_FALLBACK
    with pytest.raises(PolicyBlockedError):
        router.resolve_route(
            "engineer_builder",
            explicit_spec="deepseek:deepseek-chat:medium",
            execution_host="local",
            strength=ConstraintStrength.HARD,
        )
    r_od_soft = router.resolve_route(
        "engineer_builder",
        explicit_spec="deepseek:deepseek-chat:medium",
        execution_host="local",
        strength=ConstraintStrength.SOFT_FALLBACK,
    )
    assert r_od_soft.requested_effort == "medium"
    assert r_od_soft.wire_effort == "high"

    # Provider host restrictions declared in the catalog are enforced
    router_local = ModelRouter(capabilities_path=_local_only_gemini_caps(tmp_path))
    with pytest.raises(PolicyBlockedError) as exc_host:
        router_local.resolve_route("theorist", explicit_spec="gemini:gemini-pro:high", execution_host="remote")
    assert exc_host.value.code == "BLOCKED_HOST"

    # Raw HQ resource keys NEVER leak as canonical_model
    for raw_key in ("llm_claude", "llm_gemini", "llm_codex_audit", "llm_deepseek"):
        rt = router.resolve_route("engineer_builder", explicit_spec=raw_key, execution_host="local")
        assert rt.canonical_model not in ("llm_claude", "llm_gemini", "llm_codex_audit", "llm_deepseek")

    # Foreground reserve waitable block when active ping + used_pct=87% > 85%
    record_foreground_ping(arr_home, provider="claude", ttl_s=120.0)
    with pytest.raises(PolicyBlockedError) as exc_fg:
        router.resolve_route("theorist", explicit_spec="claude:claude-opus-5-5:high", execution_host="local")
    assert exc_fg.value.code == "WAIT_FOREGROUND_RESERVE"
    assert exc_fg.value.waitable is True


# ----------------------- 2. HQRuntime Resource Skip List & Outer Timeout Guard


def test_hq_runtime_resource_metadata_skip_and_outer_time_limit(tmp_path: Path) -> None:
    router = ModelRouter()
    route = router.resolve_route(
        ScientificRole.THEORIST, explicit_spec="claude:claude-opus-5-5:xhigh", execution_host="local"
    )
    spec = ResourceSpec(
        cpus=2,
        ram_mb=2048,
        llm_pool=route.canonical_spec(),
        wall_timeout_s=90,
        hq_llm_resource=route.hq_resource,
        pinned_route=route.to_dict(),
    )
    cpus, reqs = resource_spec_to_requests(
        {
            **spec.to_dict(),
            "model_spec": route.canonical_spec(),
            "effort": "xhigh",
        },
        stage=StageKind.CODE,
    )
    assert cpus == 2
    assert reqs["llmclaude"] == 1
    assert reqs["rammb"] == 2048
    assert "pinnedroute" not in reqs
    assert "modelspec" not in reqs

    hq = HQRuntime(arr_home=tmp_path / "arr_home")
    gate_task = StageTask(
        task_id="t-gate-lim",
        idempotency_key="k-gate-lim",
        goal_id="g1",
        hypothesis_id="h1",
        stage=StageKind.CPU_GATE,
        deps=[],
        resources=ResourceSpec(cpus=2, ram_mb=1024, wall_timeout_s=90),
    )
    _, _, _, _, _, gate_tlimit, _ = hq._normalize_task_inputs(gate_task, command=["echo", "ok"])
    assert gate_tlimit == 225

    code_task = StageTask(
        task_id="t-code-lim",
        idempotency_key="k-code-lim",
        goal_id="g1",
        hypothesis_id="h1",
        stage=StageKind.CODE,
        deps=[],
        resources=spec,
    )
    _, _, _, _, _, code_tlimit, _ = hq._normalize_task_inputs(code_task, command=["echo", "ok"])
    assert code_tlimit == 135


# ------------ 3. Single-Invocation CLI Backends, Effort Fidelity & RO Snapshot


def test_single_invocation_backends_and_enforced_readonly_review_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "fake_bins"
    bin_dir.mkdir()
    inv_log = tmp_path / "invocations.jsonl"

    def _make_fake_cli(name: str, body: str) -> Path:
        p = bin_dir / name
        p.write_text("#!/usr/bin/env python3\n" + body, encoding="utf-8")
        p.chmod(p.stat().st_mode | stat.S_IXUSR)
        return p

    _make_fake_cli(
        "claude",
        f"""import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
_ = sys.stdin.read()
with open({str(inv_log)!r}, "a", encoding="utf-8") as f:
    f.write(json.dumps({{"cli": "claude", "argv": argv, "cwd": os.getcwd()}}) + "\\n")
if "--tools" in argv:
    # Verify RO snapshot actually blocks file creation!
    ro_blocked = False
    try:
        Path(os.getcwd(), "src", "illegal_mutation.py").write_text("boom", encoding="utf-8")
    except PermissionError:
        ro_blocked = True
    res = json.dumps({{"verdict": "PASS" if ro_blocked else "FAIL", "severity": "NONE", "findings": []}}) + "\\nIMPL_VERDICT: " + ("PASS" if ro_blocked else "FAIL")
else:
    Path(os.getcwd(), "src", "core.py").write_text("def compute(x: int) -> int:\\n    return x * 2\\n# refined\\n", encoding="utf-8")
    res = "Implemented refinement."
print(json.dumps({{"is_error": False, "result": res, "usage": {{"input_tokens": 210, "output_tokens": 55}}}}))
""",
    )
    _make_fake_cli(
        "gemini",
        f"""import json, os, sys
argv = sys.argv[1:]
with open({str(inv_log)!r}, "a", encoding="utf-8") as f:
    f.write(json.dumps({{"cli": "gemini", "argv": [a if not a.startswith("-p=") else "-p=<redacted>" for a in argv]}}) + "\\n")
print(json.dumps({{"status": "SUCCESS", "response": '{{"verdict": "PASS", "severity": "NONE", "findings": []}}\\nDESIGN_VERDICT: PASS', "usage": {{"input_tokens": 180, "output_tokens": 40}}}}))
""",
    )
    # Minimal OpenAI-compatible /chat/completions endpoint for the `openai_compat` backend.
    import http.server
    import threading as _threading

    class _ChatHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)) or b"{}")
            with open(inv_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"cli": "openai_compat", "path": self.path, "body": body}) + "\n")
            payload = json.dumps(
                {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 120, "completion_tokens": 30}}
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args: Any) -> None:
            return

    chat_srv = http.server.HTTPServer(("127.0.0.1", 0), _ChatHandler)
    _threading.Thread(target=chat_srv.serve_forever, daemon=True).start()
    monkeypatch.setenv("CONJECTRA_DEEPSEEK_BASE_URL", f"http://127.0.0.1:{chat_srv.server_address[1]}/v1")

    monkeypatch.setattr(
        "conjectra.stage_runner.shutil.which",
        lambda name: str(bin_dir / name) if (bin_dir / name).is_file() else shutil.which(name),
    )
    import shutil

    origin = tmp_path / "origin"
    base_sha = _init_git_repo(origin)
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    sb = SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt")
    router = ModelRouter(
        arbiter=ResourceArbiter(
            quotas_path=tmp_path / "q.json",
            ledger_path=tmp_path / "l.json",
            gemini_quota_path=tmp_path / "v.json",
        )
    )
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb, model_router=router)

    goal = _make_goal(origin, base_sha)
    db.save_goal(goal)
    hyp = _make_hyp(goal.goal_id, base_sha, tags=["theory"])
    hyp, _ = db.register_hypothesis(hyp)
    plm = PlanLifecycleManager(db)
    plm.create_research_plan(goal, thesis="Verify scaling")
    plm.create_or_revise_experiment_plan(goal, hyp, repo_path=origin)

    # 1. DESIGN_REVIEW on Gemini gemini-pro xhigh (`--approval-mode plan`, no `--yolo`, RO snapshot)
    r_gpro = router.resolve_route(
        ScientificRole.DESIGN_REVIEWER,
        explicit_spec="gemini:gemini-pro:xhigh",
        execution_host="local",
    )
    t_drev = StageTask(
        task_id="t-design-rev",
        idempotency_key="k-design-rev",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.DESIGN_REVIEW,
        deps=[],
        workflow_id="wf-v3-1",
        pinned_route=r_gpro.to_dict(),
        resources=ResourceSpec(llm_pool=r_gpro.canonical_spec(), pinned_route=r_gpro.to_dict(), wall_timeout_s=60),
    )
    rcpt_drev = runner.run_stage(t_drev, hyp, goal, db=db, notify=False)
    assert rcpt_drev.audit_verdict == AuditVerdict.PASS
    assert rcpt_drev.effort_used == "xhigh"
    assert rcpt_drev.review_kind == ReviewKind.DESIGN_REVIEW.value
    assert rcpt_drev.failure_domain == FailureDomain.NONE

    # 2. CODE on Claude Opus xhigh (`--effort xhigh`, 1 invocation)
    r_opus = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER,
        explicit_spec="claude:claude-opus-5-5:xhigh",
        execution_host="local",
    )
    t_code = StageTask(
        task_id="t-code-opus",
        idempotency_key="k-code-opus",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        workflow_id="wf-v3-1",
        pinned_route=r_opus.to_dict(),
        resources=ResourceSpec(llm_pool=r_opus.canonical_spec(), pinned_route=r_opus.to_dict(), wall_timeout_s=60),
    )
    rcpt_code = runner.run_stage(t_code, hyp, goal, db=db, notify=False)
    assert rcpt_code.process_status == ProcessStatus.EXIT_OK
    assert rcpt_code.effort_used == "xhigh"
    assert rcpt_code.head_sha != base_sha
    hyp = db.get_hypothesis(hyp.hypothesis_id)

    # 3. IMPL_REVIEW in `chmod -R a-w` detached RO snapshot (`arr-wt-ro/...`, cleaned up in finally)
    # Verify WorkflowRuntime seam: `step_key="impl_review_r1_s1"` + `review_slot` / `target_version` / `head_sha`
    r_impl_rev = router.resolve_route(
        ScientificRole.IMPL_REVIEWER,
        explicit_spec="claude:claude-opus-5-5:high",
        execution_host="local",
    )
    pinned_with_wf_meta = {
        **r_impl_rev.to_dict(),
        "review_slot": 1,
        "target_version": 2,
        "head_sha": rcpt_code.head_sha,
    }
    t_irev = StageTask(
        task_id="t-impl-rev",
        idempotency_key="k-impl-rev",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.IMPL_REVIEW,
        deps=["t-code-opus"],
        workflow_id="wf-v3-1",
        step_key="impl_review_r1_s1",
        pinned_route=pinned_with_wf_meta,
        resources=ResourceSpec(
            llm_pool=r_impl_rev.canonical_spec(),
            pinned_route=pinned_with_wf_meta,
            wall_timeout_s=60,
        ),
    )
    rcpt_irev = runner.run_stage(t_irev, hyp, goal, db=db, notify=False)
    assert rcpt_irev.audit_verdict == AuditVerdict.PASS
    assert rcpt_irev.review_kind == ReviewKind.IMPLEMENTATION_REVIEW.value

    # Verify RO snapshot directory was cleaned up after review AND review_records persisted slot_index=1 & target_version=2
    ro_parent = tmp_path / "arr-wt-ro" / goal.project
    assert not (ro_parent / "t-impl-rev_a1").exists()
    rev_rows = db.list_review_receipts(
        hypothesis_id=hyp.hypothesis_id, review_kind=ReviewKind.IMPLEMENTATION_REVIEW
    )
    assert len(rev_rows) == 1
    assert rev_rows[0].slot_index == 1
    assert rev_rows[0].target_version == 2
    assert rev_rows[0].head_sha == rcpt_code.head_sha
    assert rev_rows[0].verdict == AuditVerdict.PASS

    # 3b. Unpinned PolicyBlockedError (`sonnet:high`) fails closed in StageRunner without launching any CLI!
    t_blocked = StageTask(
        task_id="t-code-blocked",
        idempotency_key="k-code-blocked",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        model_spec="claude:claude-sonnet-5-5:high",
        resources=ResourceSpec(llm_pool="claude:claude-sonnet-5-5:high", wall_timeout_s=60),
    )
    rcpt_blocked = runner.run_stage(t_blocked, hyp, goal, db=db, notify=False)
    assert rcpt_blocked.exit_code == 78
    assert rcpt_blocked.process_status == ProcessStatus.ENV_ERROR
    assert rcpt_blocked.engineering_verdict == EngineeringVerdict.FAIL_GATE
    assert rcpt_blocked.scientific_verdict == ScientificVerdict.UNKNOWN
    assert rcpt_blocked.failure_domain == FailureDomain.ENV_CONFIG_MISMATCH
    assert "BLOCKED_SONNET_CAP" in rcpt_blocked.failure_reason

    # 4. `multiagent.invoke_provider` preserves effort on deepseek (`medium -> high`)
    ps_od = parse_provider_spec("deepseek:deepseek-chat:medium")
    assert ps_od.effort == "high"
    od_res = invoke_provider(
        ps_od,
        "Reply ok",
        cwd=origin,
        art_dir=tmp_path / "od_art",
        tag="od1",
        readonly=True,
        timeout_s=30,
    )
    assert od_res["exit_code"] == 0

    calls = [json.loads(line) for line in inv_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(calls) == 4  # Exactly 1 process invocation per task!
    assert "--approval-mode" in calls[0]["argv"] and "plan" in calls[0]["argv"]
    assert "--yolo" not in calls[0]["argv"]
    assert "gemini-pro-xhigh" in calls[0]["argv"]
    assert calls[1]["argv"][calls[1]["argv"].index("--effort") + 1] == "xhigh"
    assert "arr-wt-ro" in calls[2]["cwd"]
    assert calls[3]["cli"] == "openai_compat"
    assert calls[3]["path"] == "/v1/chat/completions"
    assert calls[3]["body"]["reasoning_effort"] == "high"
    chat_srv.shutdown()
    db.close()


# --------- 4. CPU_GATE <=180s Strict Wall & FailureDomain Isolation ----------


def test_cpu_gate_strict_wall_deadline_and_prelaunch_reserve_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = tmp_path / "origin"
    base_sha = _init_git_repo(origin)
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    sb = SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt")
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb)

    goal = _make_goal(origin, base_sha, goal_id="g-wall")
    db.save_goal(goal)

    # Default CPU_GATE resource spec is 90s (in 60-120s band, <=180s)
    for ad_name in ("generic", "demo3"):
        ad = get_adapter(ad_name)
        h_tmp = _make_hyp(goal.goal_id, base_sha)
        assert ad.default_resource_spec(StageKind.CPU_GATE, h_tmp, goal).wall_timeout_s == 90

    monkeypatch.setattr("conjectra.stage_runner.wait_for_mem_available", lambda *a, **kw: (True, 16384))

    # Case A: Hanging child process ignoring SIGTERM is killed via pgid SIGKILL within budget -> BLOCKED_TIMEOUT + UNKNOWN
    hyp_hang = _make_hyp(
        goal.goal_id,
        base_sha,
        hyp_id="h-hang",
        acceptance_cmd=f"{sys.executable} -c 'import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)'",
    )
    db.register_hypothesis(hyp_hang)
    t_hang = StageTask(
        task_id="t-gate-hang",
        idempotency_key="k-gate-hang",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_hang.hypothesis_id,
        stage=StageKind.CPU_GATE,
        deps=[],
        resources=ResourceSpec(wall_timeout_s=2),
    )
    t0 = time.monotonic()
    r_hang = runner.run_stage(t_hang, hyp_hang, goal, db=db, notify=False)
    elapsed = time.monotonic() - t0
    assert elapsed < 4.5
    assert r_hang.process_status == ProcessStatus.TIMEOUT
    assert r_hang.scientific_verdict == ScientificVerdict.UNKNOWN
    assert r_hang.failure_domain == FailureDomain.INFRA_TRANSIENT
    assert "BLOCKED_TIMEOUT" in r_hang.failure_reason

    # Case B: When mem_wait consumes budget leaving rem <= 10.0s cleanup reserve on >=15s gate, skip subprocess launch!
    marker = tmp_path / "should_not_run.txt"
    hyp_skip = _make_hyp(
        goal.goal_id,
        base_sha,
        hyp_id="h-skip",
        acceptance_cmd=f"touch {marker}",
    )
    db.register_hypothesis(hyp_skip)

    real_mono = time.monotonic
    fake_offset = 0.0

    def _advancing_wait(*_args, **_kwargs):
        nonlocal fake_offset
        fake_offset += 12.0  # Leaves 8.0s <= 10.0s reserve on a 20s wall_cap_s!
        return True, 16384

    monkeypatch.setattr("conjectra.stage_runner.time.monotonic", lambda: real_mono() + fake_offset)
    monkeypatch.setattr("conjectra.stage_runner.wait_for_mem_available", _advancing_wait)

    t_skip = StageTask(
        task_id="t-gate-skip",
        idempotency_key="k-gate-skip",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_skip.hypothesis_id,
        stage=StageKind.CPU_GATE,
        deps=[],
        resources=ResourceSpec(wall_timeout_s=20),
    )
    r_skip = runner.run_stage(t_skip, hyp_skip, goal, db=db, notify=False)
    assert not marker.exists()
    assert r_skip.process_status == ProcessStatus.TIMEOUT
    assert r_skip.scientific_verdict == ScientificVerdict.UNKNOWN
    assert r_skip.failure_domain == FailureDomain.INFRA_TRANSIENT
    assert "BLOCKED_TIMEOUT" in r_skip.failure_reason
    db.close()


def test_env_config_mismatch_never_marked_refuted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("conjectra.stage_runner.wait_for_mem_available", lambda *a, **kw: (True, 16384))
    origin = tmp_path / "origin"
    base_sha = _init_git_repo(origin)
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    sb = SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt")
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb)

    goal = _make_goal(origin, base_sha, goal_id="g-env", primary_metric="val_loss")
    db.save_goal(goal)

    env_cases = [
        ("cuda_unavail", "echo '[cuda_unavailable] torch.cuda.is_available() is False' >&2; exit 1"),
        ("eval_input_err", "echo 'EvalInputError: missing dataset shard' >&2; exit 2"),
        ("cli_arg_err", "echo 'run_eval.py: error: unrecognized arguments: --foo-bar' >&2; exit 2"),
        (
            "twostep_has_primary_then_cli_err",
            "echo 'METRIC val_loss=0.8 denom=100'; echo 'error: unrecognized arguments: --targeted-count' >&2; exit 2",
        ),
        ("mod_missing", "echo 'ModuleNotFoundError: No module named torch_scatter' >&2; exit 1"),
        ("tests_zero_no_metric", "echo '0 passed, 2 failed in 0.03s'; exit 1"),
        (
            "twostep_has_primary_then_crash",
            "echo 'METRIC val_loss=0.8 denom=100'; echo 'RuntimeError: boom' >&2; exit 1",
        ),
    ]
    for slug, bad_cmd in env_cases:
        hyp = _make_hyp(goal.goal_id, base_sha, hyp_id=f"h-{slug}", screen_cmd=bad_cmd)
        db.register_hypothesis(hyp)
        t_scr = StageTask(
            task_id=f"t-{slug}",
            idempotency_key=f"k-{slug}",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            stage=StageKind.EXP_SCREEN,
            deps=[],
            resources=ResourceSpec(wall_timeout_s=30),
        )
        rcpt = runner.run_stage(t_scr, hyp, goal, db=db, notify=False)
        assert rcpt.scientific_verdict == ScientificVerdict.UNKNOWN, f"{slug} wrongly got {rcpt.scientific_verdict}"
        if slug not in ("tests_zero_no_metric", "twostep_has_primary_then_crash"):
            assert rcpt.failure_domain == FailureDomain.ENV_CONFIG_MISMATCH
            assert rcpt.evidence_validity == EvidenceValidity.MISSING_INPUT
        else:
            assert rcpt.failure_domain == FailureDomain.ENGINEERING_DEFECT
            assert rcpt.evidence_validity == EvidenceValidity.VALID_ENGINEERING_ONLY

    # Conversely, a genuine empirical run measuring primary_metric worse than baseline is SCIENTIFIC_OUTCOME + VALID_NEGATIVE
    hyp_emp = _make_hyp(
        goal.goal_id,
        base_sha,
        hyp_id="h-emp-neg",
        screen_cmd="echo 'METRIC val_loss=1.25 denom=100'",
    )
    db.register_hypothesis(hyp_emp)
    t_emp = StageTask(
        task_id="t-emp-neg",
        idempotency_key="k-emp-neg",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_emp.hypothesis_id,
        stage=StageKind.EXP_SCREEN,
        deps=[],
        resources=ResourceSpec(wall_timeout_s=30),
    )
    rcpt_emp = runner.run_stage(t_emp, hyp_emp, goal, db=db, notify=False)
    assert rcpt_emp.scientific_verdict == ScientificVerdict.VALID_NEGATIVE
    assert rcpt_emp.evidence_validity == EvidenceValidity.VALID_EMPIRICAL
    assert rcpt_emp.failure_domain == FailureDomain.SCIENTIFIC_OUTCOME

    # Verify the default adapter honors gpu / required_hardware=="local_gpu" on both EXP_SCREEN & EXP_CONFIRM
    pb = get_adapter("generic")
    hyp_gpu = _make_hyp(goal.goal_id, base_sha, hyp_id="h-gpu")
    hyp_gpu.params = {"required_hardware": "local_gpu"}
    assert pb.default_resource_spec(StageKind.EXP_SCREEN, hyp_gpu, goal).gpu == 1
    assert pb.default_resource_spec(StageKind.EXP_CONFIRM, hyp_gpu, goal).gpu == 1
    assert pb.default_resource_spec(StageKind.CPU_GATE, hyp_gpu, goal).gpu == 0

    _ = GenericAdapter
    db.close()


def test_p1_05_hard_exhausted_wait_resource_structured_constraints_and_mac_gemini_block(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    origin = tmp_path / "origin"
    base_sha = _init_git_repo(origin)

    # 1. Catalog-declared host block is enforced by the router
    router_base = ModelRouter(
        capabilities_path=_local_only_gemini_caps(tmp_path),
        arbiter=ResourceArbiter(
            quotas_path=tmp_path / "no_q.json",
            ledger_path=tmp_path / "no_l.json",
            gemini_quota_path=tmp_path / "no_v.json",
        ),
    )
    with pytest.raises(PolicyBlockedError) as exc_host:
        router_base.resolve_route(
            ScientificRole.THEORIST,
            explicit_spec="gemini:gemini-pro:high",
            execution_host="remote",
        )
    assert exc_host.value.code == "BLOCKED_HOST"
    assert exc_host.value.waitable is False

    # 2. Structured forced / hard / allow_fallback=False without strength -> HARD; contradictions & unknown keys -> BLOCKED_INVALID_POLICY
    pol_forced = ModelRouter.compile_routing_policy(
        {
            "rules": [
                {"rule_id": "r_forced", "role": "theorist", "model": "claude-opus-5-5", "effort": "xhigh", "forced": True},
                {"rule_id": "r_hard", "role": "research_lead", "model": "gpt-5-pro", "effort": "xhigh", "hard": True},
                {"rule_id": "r_nofb", "role": "engineer_builder", "model": "gpt-5-codex", "effort": "high", "allow_fallback": False},
            ]
        }
    )
    by_id = {r["rule_id"]: r for r in pol_forced["rules"]}
    assert by_id["r_forced"]["strength"] == ConstraintStrength.HARD.value
    assert by_id["r_hard"]["strength"] == ConstraintStrength.HARD.value
    assert by_id["r_nofb"]["strength"] == ConstraintStrength.HARD.value

    for bad_policy in (
        {"rules": [{"role": "theorist", "model": "claude-opus-5-5", "forced": True, "strength": "SOFT_FALLBACK"}]},
        {"rules": [{"role": "theorist", "model": "claude-opus-5-5", "forced": True, "allow_fallback": True}]},
        {"unknown_top_knob": True, "rules": [{"role": "theorist", "model": "claude-opus-5-5"}]},
        {"rules": [{"role": "theorist", "model": "claude-opus-5-5", "unsupported_rule_flag": 1}]},
        {"rules": [{"role": "theorist", "candidates": [{"spec": "claude-opus-5-5:high", "bad_cand_key": "x"}]}]},
        {"disallowed_families": ["martian_ai"]},
        {"disallowed_models": ["nonexistent-gpt-99"]},
    ):
        with pytest.raises(PolicyBlockedError) as exc_bad:
            ModelRouter.compile_routing_policy(bad_policy)
        assert exc_bad.value.code == "BLOCKED_INVALID_POLICY"
        assert exc_bad.value.waitable is False

    # 3. Actual 0% remaining / 100% used quota on uncompiled structured forced:True rule -> WAIT_RESOURCE (waitable=True) -> replenish -> exact pinned dispatch
    q_file = tmp_path / "quotas_live.json"
    q_file.write_text(
        json.dumps(
            {
                "claude/acc1": {
                    "quota": {"windows": [{"name": "Claude 7 days", "used": 100.0}]}
                }
            }
        ),
        encoding="utf-8",
    )
    arb = ResourceArbiter(
        quotas_path=q_file,
        ledger_path=tmp_path / "no_l2.json",
        gemini_quota_path=tmp_path / "no_v2.json",
    )
    router = ModelRouter(arbiter=arb)
    goal_raw_forced = _make_goal(origin, base_sha, goal_id="g-p1-05")
    # Pass raw uncompiled dict with `forced: True` directly on goal.routing_policy!
    goal_raw_forced.routing_policy = {
        "rules": [
            {
                "rule_id": "pin_opus_xhigh",
                "role": "theorist",
                "tag": "theory",
                "model": "claude-opus-5-5",
                "effort": "xhigh",
                "forced": True,
            }
        ]
    }
    hyp_th = _make_hyp(goal_raw_forced.goal_id, base_sha, tags=["theory"])

    with pytest.raises(PolicyBlockedError) as exc_wait:
        router.resolve_route(ScientificRole.THEORIST, goal=goal_raw_forced, hyp=hyp_th, execution_host="local")
    assert exc_wait.value.code == "WAIT_RESOURCE"
    assert exc_wait.value.waitable is True
    assert exc_wait.value.pinned_route is not None
    assert exc_wait.value.pinned_route["canonical_model"] == "claude-opus-5-5"
    assert exc_wait.value.pinned_route["wire_effort"] == "xhigh"

    # Replenish Claude quota -> exact same pinned route succeeds!
    q_file.write_text(
        json.dumps(
            {
                "claude/acc1": {
                    "quota": {"windows": [{"name": "Claude 7 days", "used": 22.0}]}
                }
            }
        ),
        encoding="utf-8",
    )
    resumed_route = router.resolve_route(
        ScientificRole.THEORIST, goal=goal_raw_forced, hyp=hyp_th, execution_host="local"
    )
    assert resumed_route.canonical_model == "claude-opus-5-5"
    assert resumed_route.wire_effort == "xhigh"
    assert resumed_route.strength == ConstraintStrength.HARD

    # 4. Enforce disallowed / disallowed_models / disallowed_families / disallowed_pools across explicit, fallback, role_default & quorum
    goal_dis = _make_goal(origin, base_sha, goal_id="g-disallowed")
    goal_dis.routing_policy = {
        "disallowed_models": ["claude-sonnet-5-5"],
        "disallowed_families": ["openai"],
        "disallowed_pools": ["grok"],
        "disallowed": ["claude:claude-opus-5-5:max"],
        "rules": [
            {
                "rule_id": "soft_builder",
                "role": "engineer_builder",
                "candidates": ["claude:claude-sonnet-5-5:medium", "gpt-5-codex:high", "deepseek-chat:high"],
                "strength": "SOFT_FALLBACK",
            }
        ],
    }
    hyp_b = _make_hyp(goal_dis.goal_id, base_sha)

    # Explicit disallowed spec fails closed with BLOCKED_DISALLOWED (waitable=False)
    with pytest.raises(PolicyBlockedError) as exc_dis_exp:
        router.resolve_route(
            ScientificRole.ENGINEER_BUILDER,
            goal=goal_dis,
            explicit_spec="codex:gpt-5-codex:high",
            execution_host="local",
        )
    assert exc_dis_exp.value.code == "BLOCKED_DISALLOWED"
    assert exc_dis_exp.value.waitable is False

    with pytest.raises(PolicyBlockedError) as exc_dis_max:
        router.resolve_route(
            ScientificRole.THEORIST,
            goal=goal_dis,
            explicit_spec="claude:claude-opus-5-5:max",
            execution_host="local",
        )
    assert exc_dis_max.value.code == "BLOCKED_DISALLOWED"
    # Whereas Opus high is allowed!
    r_opus_ok = router.resolve_route(
        ScientificRole.THEORIST,
        goal=goal_dis,
        explicit_spec="claude:claude-opus-5-5:high",
        execution_host="local",
    )
    assert r_opus_ok.canonical_model == "claude-opus-5-5"

    # SOFT_FALLBACK skips disallowed Sonnet & OpenAI gpt-5-codex and picks DeepSeek
    r_soft = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER, goal=goal_dis, hyp=hyp_b, execution_host="local"
    )
    assert r_soft.canonical_model == "deepseek-chat"

    # Quorum reviewer routing (`role_default` with avoid_families=["anthropic"]) skips disallowed OpenAI & grok -> picks Google gemini-pro!
    r_quorum_rev = router.resolve_route(
        ScientificRole.IMPL_REVIEWER,
        goal=goal_dis,
        hyp=hyp_b,
        avoid_families=["anthropic"],
        execution_host="local",
    )
    assert r_quorum_rev.family == "google"
    assert r_quorum_rev.canonical_model == "gemini-pro"


def test_p1_06_writable_codex_attempt_scratch_isolation_fences_and_budget_settle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "fake_codex_bin"
    bin_dir.mkdir()
    inv_log = tmp_path / "codex_invocations.jsonl"
    mode_file = tmp_path / "codex_mode.txt"

    origin = tmp_path / "origin"
    base_sha = _init_git_repo(origin)
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    sb = SandboxManager(sandbox_base=tmp_path / "sb", worktree_base=tmp_path / "wt")

    fake_codex = bin_dir / "codex"
    fake_codex.write_text(
        f"""#!/usr/bin/env python3
import json, os, sqlite3, sys, time
from pathlib import Path

argv = sys.argv[1:]
_ = sys.stdin.read()
cwd = Path(os.getcwd())
mode = Path({str(mode_file)!r}).read_text(encoding="utf-8").strip()
with open({str(inv_log)!r}, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({{"mode": mode, "cwd": str(cwd), "argv": argv}}) + "\\n")

if mode == "dirty_fail":
    (cwd / "src" / "core.py").write_text("BROKEN_UNVALIDATED_EDIT\\n", encoding="utf-8")
    (cwd / "src" / "untracked_dirty.py").write_text("junk = 1\\n", encoding="utf-8")
    sys.exit(1)
elif mode == "dirty_timeout":
    (cwd / "src" / "core.py").write_text("TIMEOUT_PARTIAL_EDIT\\n", encoding="utf-8")
    time.sleep(10.0)
elif mode == "outside_writable":
    (cwd / "README.md").write_text("Unauthorized outside src/ edit\\n", encoding="utf-8")
elif mode == "mid_attempt_pause":
    (cwd / "src" / "core.py").write_text("def compute(x: int) -> int:\\n    return x * 3\\n", encoding="utf-8")
    conn = sqlite3.connect({str(db.db_path)!r})
    conn.execute("UPDATE goals SET status = 'PAUSED' WHERE goal_id = 'g-scratch'")
    conn.commit()
    conn.close()
elif mode == "mid_attempt_plan_epoch":
    (cwd / "src" / "core.py").write_text("def compute(x: int) -> int:\\n    return x * 5\\n", encoding="utf-8")
    conn = sqlite3.connect({str(db.db_path)!r})
    row = conn.execute("SELECT payload_json FROM goals WHERE goal_id = 'g-scratch'").fetchone()
    doc = json.loads(row[0])
    doc["plan_epoch"] = 7
    conn.execute("UPDATE goals SET payload_json = ? WHERE goal_id = 'g-scratch'", (json.dumps(doc),))
    conn.commit()
    conn.close()
elif mode == "clean_promote":
    (cwd / "src" / "core.py").write_text("def compute(x: int) -> int:\\n    return x * 10\\n", encoding="utf-8")
elif mode == "sibling_epoch_sync":
    (cwd / "src" / "core.py").write_text("def compute(x: int) -> int:\\n    return x * 20\\n", encoding="utf-8")
    conn = sqlite3.connect({str(db.db_path)!r})
    g_row = conn.execute("SELECT payload_json FROM goals WHERE goal_id = 'g-scratch'").fetchone()
    g_doc = json.loads(g_row[0])
    g_doc["human_epoch"] = 2
    g_doc["plan_epoch"] = 8
    conn.execute("UPDATE goals SET payload_json = ? WHERE goal_id = 'g-scratch'", (json.dumps(g_doc),))
    t_row = conn.execute("SELECT payload_json FROM tasks WHERE task_id = 't-codex-sibling-sync'").fetchone()
    t_doc = json.loads(t_row[0])
    t_doc["human_epoch"] = 2
    t_doc["plan_epoch"] = 8
    if isinstance(t_doc.get("pinned_route"), dict):
        t_doc["pinned_route"]["plan_epoch"] = 8
    conn.execute("UPDATE tasks SET payload_json = ? WHERE task_id = 't-codex-sibling-sync'", (json.dumps(t_doc),))
    conn.commit()
    conn.close()

print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "done"}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 1000, "cached_input_tokens": 700, "output_tokens": 80}}}}))
""",
        encoding="utf-8",
    )
    fake_codex.chmod(fake_codex.stat().st_mode | stat.S_IXUSR)

    import shutil as _shutil

    monkeypatch.setattr(
        "conjectra.stage_runner.shutil.which",
        lambda name: str(fake_codex) if name == "codex" else _shutil.which(name),
    )

    settled_calls: list[dict[str, Any]] = []

    def _spy_settle(goal_id: str, usage_key: str, **kwargs: Any) -> int:
        settled_calls.append({"goal_id": goal_id, "usage_key": usage_key, **kwargs})
        return int(kwargs.get("input_tokens", 0)) + int(kwargs.get("output_tokens", 0))

    monkeypatch.setattr(db, "settle_goal_budget", _spy_settle, raising=False)

    router = ModelRouter(
        arbiter=ResourceArbiter(
            quotas_path=tmp_path / "q.json",
            ledger_path=tmp_path / "l.json",
            gemini_quota_path=tmp_path / "v.json",
        )
    )
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb, model_router=router)
    goal = _make_goal(origin, base_sha, goal_id="g-scratch")
    db.save_goal(goal)
    hyp = _make_hyp(goal.goal_id, base_sha, hyp_id="h-scratch")
    hyp, _ = db.register_hypothesis(hyp)

    mirror = sb.ensure_mirror(origin, goal.project)
    wt_path = sb.create_worktree(
        mirror, goal.project, hyp.branch_name, base_sha, origin_repo=origin, goal_id=goal.goal_id
    )
    r_codex = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER, explicit_spec="codex:gpt-5-codex:high", execution_host="local"
    )

    def _assert_wt_untouched() -> None:
        head_now = subprocess.run(
            ["git", "-C", str(wt_path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty_now = subprocess.run(
            ["git", "-C", str(wt_path), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        assert head_now == base_sha
        assert dirty_now == ""
        scratch_root = tmp_path / "arr-wt-scratch" / goal.project
        if scratch_root.exists():
            assert list(scratch_root.iterdir()) == []

    # Case 1: Dirty failed writable Codex attempt -> scratch discarded, wt_path untouched!
    mode_file.write_text("dirty_fail", encoding="utf-8")
    t1 = StageTask(
        task_id="t-codex-fail",
        idempotency_key="k-codex-fail",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route={**r_codex.to_dict(), "plan_epoch": 0},
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=r_codex.to_dict(), wall_timeout_s=30),
    )
    rcpt1 = runner.run_stage(t1, hyp, goal, db=db, notify=False)
    assert rcpt1.engineering_verdict == EngineeringVerdict.FAIL_GATE
    assert rcpt1.head_sha == base_sha
    _assert_wt_untouched()

    # Case 2: Timed-out writable Codex attempt -> scratch discarded, wt_path untouched!
    mode_file.write_text("dirty_timeout", encoding="utf-8")
    t2 = StageTask(
        task_id="t-codex-to",
        idempotency_key="k-codex-to",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=r_codex.to_dict(),
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=r_codex.to_dict(), wall_timeout_s=1),
    )
    rcpt2 = runner.run_stage(t2, hyp, goal, db=db, notify=False)
    assert rcpt2.process_status == ProcessStatus.TIMEOUT
    assert rcpt2.scientific_verdict == ScientificVerdict.UNKNOWN
    _assert_wt_untouched()

    # Case 3: Edit outside hyp.writable_paths ("src/") -> rejected in scratch before touching wt_path!
    mode_file.write_text("outside_writable", encoding="utf-8")
    t3 = StageTask(
        task_id="t-codex-outside",
        idempotency_key="k-codex-outside",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=r_codex.to_dict(),
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=r_codex.to_dict(), wall_timeout_s=30),
    )
    rcpt3 = runner.run_stage(t3, hyp, goal, db=db, notify=False)
    assert rcpt3.engineering_verdict == EngineeringVerdict.FAIL_GATE
    assert "SCRATCH_GATE_REJECTED" in rcpt3.failure_reason
    _assert_wt_untouched()

    # Case 4: Mid-attempt pause & mid-attempt plan_epoch bump -> pre-promotion fence discards scratch!
    mode_file.write_text("mid_attempt_pause", encoding="utf-8")
    t4 = StageTask(
        task_id="t-codex-paused",
        idempotency_key="k-codex-paused",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=r_codex.to_dict(),
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=r_codex.to_dict(), wall_timeout_s=30),
    )
    rcpt4 = runner.run_stage(t4, hyp, goal, db=db, notify=False)
    assert rcpt4.process_status == ProcessStatus.LEASE_LOST
    assert rcpt4.evidence_validity == EvidenceValidity.HUMAN_INVALIDATED
    assert "FENCED_PRE_PROMOTE" in rcpt4.failure_reason
    _assert_wt_untouched()
    db.update_goal_status(goal.goal_id, "ACTIVE")

    mode_file.write_text("mid_attempt_plan_epoch", encoding="utf-8")
    t5 = StageTask(
        task_id="t-codex-epoch",
        idempotency_key="k-codex-epoch",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route={**r_codex.to_dict(), "plan_epoch": 0},
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=r_codex.to_dict(), wall_timeout_s=30),
    )
    rcpt5 = runner.run_stage(t5, hyp, goal, db=db, notify=False)
    assert rcpt5.process_status == ProcessStatus.LEASE_LOST
    assert "plan_epoch mismatch" in rcpt5.failure_reason
    _assert_wt_untouched()

    # Case 5: Clean validated promotion at current plan_epoch=7 -> promotes candidate_sha, stamps plan_epoch, normalizes OpenAI cached tokens & settles budget!
    mode_file.write_text("clean_promote", encoding="utf-8")
    t6 = StageTask(
        task_id="t-codex-ok",
        idempotency_key="k-codex-ok",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route={**r_codex.to_dict(), "plan_epoch": 7},
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=r_codex.to_dict(), wall_timeout_s=30),
    )
    rcpt6 = runner.run_stage(t6, hyp, goal, db=db, notify=False)
    assert rcpt6.process_status == ProcessStatus.EXIT_OK
    assert rcpt6.engineering_verdict == EngineeringVerdict.PASS
    assert rcpt6.head_sha != base_sha
    assert getattr(rcpt6, "plan_epoch", None) == 7
    assert rcpt6.in_tokens == 300  # 1000 inclusive - 700 cached_input_tokens!
    assert rcpt6.out_tokens == 80
    assert rcpt6.metrics["cache_read_tokens"]["value"] == 700
    assert "return x * 10" in (wt_path / "src" / "core.py").read_text(encoding="utf-8")
    assert subprocess.run(
        ["git", "-C", str(wt_path), "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip() == ""
    assert db.get_hypothesis(hyp.hypothesis_id).head_sha == rcpt6.head_sha

    # Case 6: Surgical sibling intervention syncs both goals & unaffected task epochs mid-attempt (`human_epoch=2, plan_epoch=8`) -> promotes cleanly and stamps synced epochs!
    hyp_after_t6 = db.get_hypothesis(hyp.hypothesis_id)
    assert hyp_after_t6 is not None
    mode_file.write_text("sibling_epoch_sync", encoding="utf-8")
    t7 = StageTask(
        task_id="t-codex-sibling-sync",
        idempotency_key="k-codex-sibling-sync",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        attempt=1,
        deps=[],
        human_epoch=0,
        worktree_path=str(wt_path),
        pinned_route={**r_codex.to_dict(), "plan_epoch": 7},
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=r_codex.to_dict(), wall_timeout_s=30),
    )
    db.enqueue_task(t7)
    rcpt7 = runner.run_stage(t7, hyp_after_t6, goal, db=db, notify=False)
    assert rcpt7.process_status == ProcessStatus.EXIT_OK
    assert rcpt7.engineering_verdict == EngineeringVerdict.PASS
    assert rcpt7.head_sha != rcpt6.head_sha
    assert rcpt7.human_epoch == 2
    assert getattr(rcpt7, "plan_epoch", None) == 8
    assert "return x * 20" in (wt_path / "src" / "core.py").read_text(encoding="utf-8")

    # Verify every writable Codex invocation ran inside arr-wt-scratch (NEVER wt_path!)
    logged_runs = [json.loads(line) for line in inv_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(logged_runs) == 7
    for entry in logged_runs:
        assert "arr-wt-scratch" in entry["cwd"]
        assert entry["cwd"] != str(wt_path)

    # Verify settle_goal_budget was invoked ONCE per persisted task with normalized uncached input_tokens=300 & cached_input_tokens=700
    ok_settle = [s for s in settled_calls if s["usage_key"] == "task:t-codex-ok:a1"]
    assert len(ok_settle) == 1
    assert ok_settle[0]["input_tokens"] == 300
    assert ok_settle[0]["output_tokens"] == 80
    assert ok_settle[0]["cached_input_tokens"] == 700
    assert ok_settle[0]["state"] == "SETTLED"
    db.close()


def test_p1_05_r2_rule_local_disallows_survive_fallback_and_xboost_enforces_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("conjectra.stage_runner.wait_for_mem_available", lambda *a, **kw: (True, 16384))
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # 1. SOFT_FALLBACK rule prefers claude-opus-5-5 (quota 100% exhausted) and rule-locally disallows gemini-pro.
    #    Role defaults for engineer_builder MUST NOT select disallowed gemini-pro, and explicit gemini-pro MUST block!
    q_snap = tmp_path / "quota_r2.json"
    q_snap.write_text(
        json.dumps(
            [
                {
                    "provider": "claude",
                    "plan": "MAX",
                    "windows": [{"name": "Claude 7 days", "used": 100.0, "resetsAt": None}],
                    "at": now_iso,
                },
                {
                    "provider": "codex",
                    "plan": "pro",
                    "windows": [{"name": "7 days", "used": 15.0, "resetsAt": None}],
                    "at": now_iso,
                },
                {
                    "provider": "deepseek",
                    "plan": "go",
                    "windows": [{"name": "7 days", "used": 10.0, "resetsAt": None}],
                    "at": now_iso,
                },
            ]
        ),
        encoding="utf-8",
    )
    arb = ResourceArbiter(
        quotas_path=tmp_path / "none_q.json",
        quota_snapshot_path=q_snap,
        ledger_path=tmp_path / "none_l.json",
        gemini_quota_path=tmp_path / "none_v.json",
    )
    router = ModelRouter(arbiter=arb, arr_home=tmp_path / "arr_home_r2")
    origin = tmp_path / "origin_r2"
    base_sha = _init_git_repo(origin)

    goal_soft = _make_goal(
        origin,
        base_sha,
        goal_id="g-r2-soft",
        routing_policy={
            "rules": [
                {
                    "rule_id": "kernel-soft-no-gemini-pro",
                    "roles": ["engineer_builder"],
                    "tags": ["kernel"],
                    "strength": "SOFT_FALLBACK",
                    "candidates": [{"provider_key": "claude-opus-5-5", "effort": "high"}],
                    "disallowed_models": ["gemini-pro", "gemini-flash"],
                }
            ]
        },
    )
    hyp_kernel = _make_hyp(goal_soft.goal_id, base_sha, hyp_id="h-kernel", tags=["kernel"])

    r_fallback = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER,
        goal=goal_soft,
        hyp=hyp_kernel,
        execution_host="local",
    )
    assert "gemini-pro" not in r_fallback.canonical_model
    assert "gemini-flash" not in r_fallback.canonical_model
    assert r_fallback.family in ("deepseek", "openai")

    with pytest.raises(PolicyBlockedError) as exc_exp_disallowed:
        router.resolve_route(
            ScientificRole.ENGINEER_BUILDER,
            goal=goal_soft,
            hyp=hyp_kernel,
            explicit_spec="gemini:gemini-pro:high",
            execution_host="local",
        )
    assert exc_exp_disallowed.value.code == "BLOCKED_DISALLOWED"

    # 2. Legacy `collab_mode == "x-boost"` with `hyp.params["builder_spec"]` MUST obey Goal HARD policy & WAIT_RESOURCE!
    arr_home = tmp_path / "arr_home_boost"
    db = StateDB(arr_home)
    sb = SandboxManager(sandbox_base=tmp_path / "sb_b", worktree_base=tmp_path / "wt_b")
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb, model_router=router)

    goal_hard = _make_goal(
        origin,
        base_sha,
        goal_id="g-r2-boost",
        routing_policy={
            "rules": [
                {
                    "rule_id": "hard-opus-builder",
                    "roles": ["engineer_builder"],
                    "strength": "HARD",
                    "candidates": [{"provider_key": "claude-opus-5-5", "effort": "xhigh"}],
                    "disallowed_models": ["gemini-flash"],
                }
            ]
        },
    )
    db.save_goal(goal_hard)
    hyp_boost = HypothesisNode.from_dict(
        {
            **_make_hyp(
                goal_hard.goal_id,
                base_sha,
                hyp_id="h-boost-legacy",
                acceptance_cmd="python3 -c 'import src.core as c; assert c.compute(3) == 9; print(\"1 passed\")'",
            ).to_dict(),
            "params": {
                "collab_mode": "x-boost",
                "builder_spec": "gemini:gemini-flash:high",
                "boost_max_rounds": 1,
            },
        }
    )
    hyp_boost, _ = db.register_hypothesis(hyp_boost)

    t_boost_wait = StageTask(
        task_id="t-boost-wait",
        idempotency_key="k-boost-wait",
        goal_id=goal_hard.goal_id,
        hypothesis_id=hyp_boost.hypothesis_id,
        stage=StageKind.CODE,
        deps=[],
        resources=ResourceSpec(llm_pool="auto", wall_timeout_s=30),
    )
    # While Claude quota is 100% used, legacy builder_spec MUST NOT bypass HARD Opus pin -> WAIT_RESOURCE!
    rcpt_wait = runner.run_stage(t_boost_wait, hyp_boost, goal_hard, db=db, notify=False)
    assert rcpt_wait.exit_code == 78
    assert rcpt_wait.failure_domain == FailureDomain.INFRA_TRANSIENT
    assert "WAIT_RESOURCE" in rcpt_wait.failure_reason

    # Restore Claude quota -> x-boost runs Opus xhigh builder + cross-family Codex adversary inside scratch and promotes!
    q_snap.write_text(
        json.dumps(
            [
                {
                    "provider": "claude",
                    "plan": "MAX",
                    "windows": [{"name": "Claude 7 days", "used": 20.0, "resetsAt": None}],
                    "at": now_iso,
                },
                {
                    "provider": "codex",
                    "plan": "pro",
                    "windows": [{"name": "7 days", "used": 15.0, "resetsAt": None}],
                    "at": now_iso,
                },
            ]
        ),
        encoding="utf-8",
    )
    bin_dir = tmp_path / "boost_bins"
    bin_dir.mkdir()
    boost_log = tmp_path / "boost_cli.jsonl"

    fake_claude = bin_dir / "claude"
    fake_claude.write_text(
        f"""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
argv = sys.argv[1:]
_ = sys.stdin.read()
with open({str(boost_log)!r}, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({{"cli": "claude", "argv": argv, "cwd": os.getcwd()}}) + "\\n")
Path(os.getcwd(), "src", "core.py").write_text("def compute(x: int) -> int:\\n    return x * 3\\n", encoding="utf-8")
print(json.dumps({{"is_error": False, "result": "updated", "usage": {{"input_tokens": 150, "output_tokens": 40}}}}))
""",
        encoding="utf-8",
    )
    fake_claude.chmod(0o755)

    fake_codex = bin_dir / "codex"
    fake_codex.write_text(
        f"""#!/usr/bin/env python3
import json, os, sys
argv = sys.argv[1:]
_ = sys.stdin.read()
with open({str(boost_log)!r}, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({{"cli": "codex", "argv": argv, "cwd": os.getcwd()}}) + "\\n")
print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "Clean.\\nBOOST_VERDICT: PASS"}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 120, "output_tokens": 25}}}}))
""",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)

    import shutil as _shutil

    monkeypatch.setattr(
        "conjectra.stage_runner.shutil.which",
        lambda name: str(bin_dir / name) if (bin_dir / name).is_file() else _shutil.which(name),
    )

    t_boost_ok = StageTask(
        task_id="t-boost-ok",
        idempotency_key="k-boost-ok",
        goal_id=goal_hard.goal_id,
        hypothesis_id=hyp_boost.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        resources=ResourceSpec(llm_pool="auto", wall_timeout_s=60),
    )
    db.enqueue_task(t_boost_ok)
    rcpt_ok = runner.run_stage(t_boost_ok, hyp_boost, goal_hard, db=db, notify=False)
    assert rcpt_ok.process_status == ProcessStatus.EXIT_OK
    assert rcpt_ok.engineering_verdict == EngineeringVerdict.PASS
    assert rcpt_ok.head_sha != base_sha
    assert "claude-opus-5-5" in rcpt_ok.model_used
    assert "gemini-flash" not in rcpt_ok.model_used
    cli_calls = [json.loads(x) for x in boost_log.read_text(encoding="utf-8").splitlines() if x.strip()]
    assert [c["cli"] for c in cli_calls] == ["claude", "codex"]
    assert cli_calls[0]["argv"][cli_calls[0]["argv"].index("--effort") + 1] == "xhigh"
    db.close()


def test_p1_04_r2_and_p1_06_r2_worker_lease_ttl_goal_version_and_strict_ff_promotion_intents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conjectra.arbiter import ConflictManager
    from conjectra.stage_runner import main as stage_runner_main

    origin = tmp_path / "origin_fences"
    base_sha = _init_git_repo(origin)
    arr_home = tmp_path / "arr_home_fences"
    db = StateDB(arr_home)
    sb = SandboxManager(sandbox_base=tmp_path / "sb_f", worktree_base=tmp_path / "wt_f")
    router = ModelRouter(
        arbiter=ResourceArbiter(
            quotas_path=tmp_path / "q.json",
            ledger_path=tmp_path / "l.json",
            gemini_quota_path=tmp_path / "v.json",
        )
    )
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb, model_router=router)

    goal = _make_goal(origin, base_sha, goal_id="g-fences")
    db.save_goal(goal)
    hyp = _make_hyp(goal.goal_id, base_sha, hyp_id="h-fences")
    hyp, _ = db.register_hypothesis(hyp)

    mirror = sb.ensure_mirror(origin, goal.project)
    wt_path = sb.create_worktree(
        mirror, goal.project, hyp.branch_name, base_sha, origin_repo=origin, goal_id=goal.goal_id
    )
    r_codex = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER, explicit_spec="codex:gpt-5-codex:high", execution_host="local"
    )

    mode_file = tmp_path / "fence_mode.txt"
    inv_count_file = tmp_path / "fence_inv_count.txt"
    inv_count_file.write_text("0", encoding="utf-8")

    fake_codex = tmp_path / "codex"
    fake_codex.write_text(
        f"""#!/usr/bin/env python3
import json, os, sqlite3, subprocess, sys
from pathlib import Path

_ = sys.stdin.read()
cwd = Path(os.getcwd())
cnt = int(Path({str(inv_count_file)!r}).read_text(encoding="utf-8")) + 1
Path({str(inv_count_file)!r}).write_text(str(cnt), encoding="utf-8")
mode = Path({str(mode_file)!r}).read_text(encoding="utf-8").strip()

if mode == "expire_lease_mid_attempt":
    (cwd / "src" / "core.py").write_text("def compute(x: int) -> int:\\n    return x * 11\\n", encoding="utf-8")
    conn = sqlite3.connect({str(db.db_path)!r})
    conn.execute("UPDATE leases SET expires_ts = 1.0, status = 'EXPIRED' WHERE task_id = 't-lease-mid'")
    conn.commit()
    conn.close()
elif mode == "bump_goal_version_mid_attempt":
    (cwd / "src" / "core.py").write_text("def compute(x: int) -> int:\\n    return x * 12\\n", encoding="utf-8")
    conn = sqlite3.connect({str(db.db_path)!r})
    row = conn.execute("SELECT payload_json FROM goals WHERE goal_id = 'g-fences'").fetchone()
    doc = json.loads(row[0])
    doc["version"] = 2
    conn.execute("UPDATE goals SET payload_json = ? WHERE goal_id = 'g-fences'", (json.dumps(doc),))
    conn.commit()
    conn.close()
elif mode == "non_ff_orphan_commit":
    subprocess.run(["git", "-C", str(cwd), "checkout", "--orphan", "rogue_root"], check=True, capture_output=True)
    (cwd / "src" / "core.py").write_text("def compute(x: int) -> int:\\n    return x * 13\\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(cwd), "add", "-A"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=r", "-c", "user.email=r@l", "commit", "-m", "non-ff"],
        check=True,
        capture_output=True,
    )
elif mode == "clean_ff_promote":
    (cwd / "src" / "core.py").write_text("def compute(x: int) -> int:\\n    return x * 42\\n", encoding="utf-8")

print(json.dumps({{"type": "item.completed", "item": {{"type": "agent_message", "text": "ok"}}}}))
print(json.dumps({{"type": "turn.completed", "usage": {{"input_tokens": 200, "output_tokens": 30}}}}))
""",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    import shutil as _shutil

    monkeypatch.setattr(
        "conjectra.stage_runner.shutil.which",
        lambda name: str(fake_codex) if name == "codex" else _shutil.which(name),
    )

    # 1. Pre-exec expired worker lease -> rejected BEFORE launching codex!
    owner_pre = "task:t-lease-pre:a1"
    b_lease = db.acquire_branch_writer_lease(goal.goal_id, hyp.hypothesis_id, "t-lease-pre", holder=owner_pre, ttl_s=60.0)
    assert b_lease is not None
    with db._tx() as (conn, _p):
        conn.execute("UPDATE leases SET expires_ts = ? WHERE lease_id = ?", (time.time() - 5.0, b_lease["lease_id"]))
    pin_pre = {
        **r_codex.to_dict(),
        "expected_goal_version": 1,
        "lease_owner_id": owner_pre,
        "acquired_lease_keys": [b_lease["resource_key"]],
    }
    t_pre = StageTask(
        task_id="t-lease-pre",
        idempotency_key="k-lease-pre",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=pin_pre,
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=pin_pre, wall_timeout_s=30),
    )
    db.enqueue_task(t_pre)
    rcpt_pre = runner.run_stage(t_pre, hyp, goal, db=db, notify=False)
    assert rcpt_pre.process_status == ProcessStatus.LEASE_LOST
    assert "FENCED_LEASE_EXPIRED" in rcpt_pre.failure_reason
    assert int(inv_count_file.read_text(encoding="utf-8")) == 0
    ConflictManager.release_task_leases(db, "t-lease-pre")

    # 2. Mid-attempt worker lease expiration -> pre-promote fence rejects without touching wt_path!
    mode_file.write_text("expire_lease_mid_attempt", encoding="utf-8")
    owner_mid = "task:t-lease-mid:a1"
    b_mid = db.acquire_branch_writer_lease(goal.goal_id, hyp.hypothesis_id, "t-lease-mid", holder=owner_mid, ttl_s=120.0)
    s_mid = ConflictManager.acquire_repo_scope_leases(
        db, task_id="t-lease-mid", goal_id=goal.goal_id, repo_path=origin, writable_paths=["src/"], holder=owner_mid, ttl_s=120.0
    )
    assert b_mid is not None and s_mid is not None
    pin_mid = {
        **r_codex.to_dict(),
        "expected_goal_version": 1,
        "lease_owner_id": owner_mid,
        "acquired_lease_keys": [b_mid["resource_key"], *[x["resource_key"] for x in s_mid]],
    }
    t_mid = StageTask(
        task_id="t-lease-mid",
        idempotency_key="k-lease-mid",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=pin_mid,
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=pin_mid, wall_timeout_s=30),
    )
    db.enqueue_task(t_mid)
    rcpt_mid = runner.run_stage(t_mid, hyp, goal, db=db, notify=False)
    assert rcpt_mid.process_status == ProcessStatus.LEASE_LOST
    assert "FENCED_LEASE_EXPIRED" in rcpt_mid.failure_reason
    assert subprocess.run(["git", "-C", str(wt_path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() == base_sha
    ConflictManager.release_task_leases(db, "t-lease-mid")

    # 3. Mid-attempt goal_version bump (1 -> 2) -> pre-promote fence marks STALE_GOAL_VERSION without advancing wt_path!
    mode_file.write_text("bump_goal_version_mid_attempt", encoding="utf-8")
    pin_gv = {**r_codex.to_dict(), "expected_goal_version": 1}
    t_gv = StageTask(
        task_id="t-gv-bump",
        idempotency_key="k-gv-bump",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        goal_version=1,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=pin_gv,
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=pin_gv, wall_timeout_s=30),
    )
    db.enqueue_task(t_gv)
    rcpt_gv = runner.run_stage(t_gv, hyp, goal, db=db, notify=False)
    assert rcpt_gv.process_status == ProcessStatus.LEASE_LOST
    assert rcpt_gv.evidence_validity == EvidenceValidity.STALE_GOAL_VERSION
    assert "goal_version mismatch" in rcpt_gv.failure_reason
    assert subprocess.run(["git", "-C", str(wt_path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() == base_sha

    # CLI `--goal-version 1` exits 75 immediately when DB goal is at version 2 without mutating task status!
    monkeypatch.setattr("conjectra.stage_runner.notify_controller", lambda *_a, **_kw: True)
    db.close()
    rc_cli_gv = stage_runner_main(
        ["--task-id", "t-gv-bump", "--arr-home", str(arr_home), "--attempt", "1", "--goal-version", "1", "--dry-run-llm"]
    )
    assert rc_cli_gv == 75
    db = StateDB(arr_home)
    assert db.get_task("t-gv-bump").status == "RUNNING"

    # Reset goal version to 2 on our local goal object for subsequent checks
    goal = db.get_goal(goal.goal_id)
    assert goal is not None and goal.version == 2

    # 4. Non-fast-forward orphan commit changing only allowed `src/core.py` -> rejected by strict ancestry check (NO reset --hard)!
    mode_file.write_text("non_ff_orphan_commit", encoding="utf-8")
    pin_v2 = {**r_codex.to_dict(), "expected_goal_version": 2}
    t_nonff = StageTask(
        task_id="t-nonff",
        idempotency_key="k-nonff",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        goal_version=2,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=pin_v2,
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=pin_v2, wall_timeout_s=30),
    )
    db.enqueue_task(t_nonff)
    rcpt_nonff = runner.run_stage(t_nonff, hyp, goal, db=db, notify=False)
    assert rcpt_nonff.engineering_verdict == EngineeringVerdict.FAIL_GATE
    assert "FF_PROMOTE_REJECTED" in rcpt_nonff.failure_reason
    assert subprocess.run(["git", "-C", str(wt_path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip() == base_sha

    # 5. Clean proper-descendant promotion with valid leases -> advances wt_path, records COMMITTED promotion_intent, and deletes refs/arr-promote/*
    mode_file.write_text("clean_ff_promote", encoding="utf-8")
    owner_ok = "task:t-ff-ok:a1"
    b_ok = db.acquire_branch_writer_lease(goal.goal_id, hyp.hypothesis_id, "t-ff-ok", holder=owner_ok, ttl_s=120.0)
    assert b_ok is not None
    pin_ok = {
        **r_codex.to_dict(),
        "expected_goal_version": 2,
        "lease_owner_id": owner_ok,
        "acquired_lease_keys": [b_ok["resource_key"]],
    }
    t_ok = StageTask(
        task_id="t-ff-ok",
        idempotency_key="k-ff-ok",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        goal_version=2,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=pin_ok,
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=pin_ok, wall_timeout_s=30),
    )
    db.enqueue_task(t_ok)
    rcpt_ff = runner.run_stage(t_ok, hyp, goal, db=db, notify=False)
    assert rcpt_ff.process_status == ProcessStatus.EXIT_OK
    assert rcpt_ff.head_sha != base_sha
    assert db.get_hypothesis(hyp.hypothesis_id).head_sha == rcpt_ff.head_sha
    prom_refs = subprocess.run(
        ["git", "-C", str(wt_path), "for-each-ref", "refs/arr-promote/"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert prom_refs == ""
    with db._lock:
        intent_row = db._conn.execute(
            "SELECT state, expected_base_sha, candidate_sha FROM promotion_intents WHERE task_id = 't-ff-ok'"
        ).fetchone()
    assert intent_row is not None
    assert intent_row["state"] == "COMMITTED"
    assert intent_row["expected_base_sha"] == base_sha
    assert intent_row["candidate_sha"] == rcpt_ff.head_sha
    ConflictManager.release_task_leases(db, "t-ff-ok")

    # 6. `run_stage(..., db=None, persist=True)` opens StateDB(home) up front -> records promotion_intents AND updates hyp.head_sha!
    hyp_after_ff = db.get_hypothesis(hyp.hypothesis_id)
    assert hyp_after_ff is not None
    db.close()

    mode_file.write_text("clean_ff_promote", encoding="utf-8")
    # Write a fresh change in scratch via another task with db=None, persist=True
    fake_codex.write_text(
        fake_codex.read_text(encoding="utf-8").replace("return x * 42", "return x * 99"),
        encoding="utf-8",
    )
    t_db_none = StageTask(
        task_id="t-db-none-persist",
        idempotency_key="k-db-none-persist",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        goal_version=2,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=pin_v2,
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=pin_v2, wall_timeout_s=30),
    )
    rcpt_db_none = runner.run_stage(t_db_none, hyp_after_ff, goal, db=None, persist=True, notify=False)
    assert rcpt_db_none.process_status == ProcessStatus.EXIT_OK
    assert rcpt_db_none.head_sha != hyp_after_ff.head_sha

    db = StateDB(arr_home)
    hyp_reloaded = db.get_hypothesis(hyp.hypothesis_id)
    assert hyp_reloaded is not None and hyp_reloaded.head_sha == rcpt_db_none.head_sha
    with db._lock:
        intent_none_row = db._conn.execute(
            "SELECT state, candidate_sha FROM promotion_intents WHERE task_id = 't-db-none-persist'"
        ).fetchone()
    assert intent_none_row is not None
    assert intent_none_row["state"] == "COMMITTED"
    assert intent_none_row["candidate_sha"] == rcpt_db_none.head_sha

    # 7. Non-CODE stages (`IMPL_REVIEW` & `EXP_SCREEN`): pre-exec expired lease fence, heartbeat renewal, pre-persist fence, and main() rc=75!
    monkeypatch.setattr("conjectra.stage_runner.wait_for_mem_available", lambda *a, **kw: (True, 16384))
    ro_owner = "task:t-rev-lease-exp:a1"
    ro_lease = db.acquire_branch_ro_lease(
        goal.goal_id, hyp.hypothesis_id, hyp_reloaded.head_sha, "t-rev-lease-exp", holder=ro_owner, ttl_s=60.0
    )
    assert ro_lease is not None
    with db._tx() as (conn, _p):
        conn.execute("UPDATE leases SET expires_ts = ? WHERE lease_id = ?", (time.time() - 5.0, ro_lease["lease_id"]))

    pin_rev_exp = {
        **r_codex.to_dict(),
        "expected_goal_version": 2,
        "head_sha": hyp_reloaded.head_sha,
        "lease_owner_id": ro_owner,
        "acquired_lease_keys": [ro_lease["resource_key"]],
    }
    t_rev_exp = StageTask(
        task_id="t-rev-lease-exp",
        idempotency_key="k-rev-lease-exp",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.IMPL_REVIEW,
        status="RUNNING",
        goal_version=2,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=pin_rev_exp,
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=pin_rev_exp, wall_timeout_s=30),
    )
    db.enqueue_task(t_rev_exp)
    inv_before_rev = int(inv_count_file.read_text(encoding="utf-8"))
    rcpt_rev_exp = runner.run_stage(t_rev_exp, hyp_reloaded, goal, db=db, notify=False)
    assert rcpt_rev_exp.process_status == ProcessStatus.LEASE_LOST
    assert rcpt_rev_exp.exit_code == 75
    assert "FENCED_PRE_EXEC" in rcpt_rev_exp.failure_reason and "FENCED_LEASE_EXPIRED" in rcpt_rev_exp.failure_reason
    assert int(inv_count_file.read_text(encoding="utf-8")) == inv_before_rev
    assert db.list_review_receipts(hypothesis_id=hyp.hypothesis_id) == []
    ConflictManager.release_task_leases(db, "t-rev-lease-exp")

    # Background lease heartbeat renews active EXP_SCREEN lease during execution
    runner._lease_heartbeat_interval_s = 0.05
    scr_owner = "task:t-scr-hb:a1"
    scr_lease = db.acquire_branch_ro_lease(
        goal.goal_id, hyp.hypothesis_id, hyp_reloaded.head_sha, "t-scr-hb", holder=scr_owner, ttl_s=1.0
    )
    assert scr_lease is not None
    initial_exp = float(scr_lease["expires_ts"])
    hyp_scr = HypothesisNode.from_dict(
        {
            **hyp_reloaded.to_dict(),
            "screen_cmd": f"{sys.executable} -c 'import time; time.sleep(0.18); print(\"METRIC score=9.0 denom=1\")'",
        }
    )
    pin_scr = {
        "expected_goal_version": 2,
        "lease_owner_id": scr_owner,
        "acquired_lease_keys": [scr_lease["resource_key"]],
    }
    t_scr_hb = StageTask(
        task_id="t-scr-hb",
        idempotency_key="k-scr-hb",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.EXP_SCREEN,
        status="RUNNING",
        goal_version=2,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=pin_scr,
        resources=ResourceSpec(wall_timeout_s=30),
    )
    db.enqueue_task(t_scr_hb)
    rcpt_scr_hb = runner.run_stage(t_scr_hb, hyp_scr, goal, db=db, notify=False)
    assert rcpt_scr_hb.process_status == ProcessStatus.EXIT_OK
    with db._lock:
        renewed_row = db._conn.execute(
            "SELECT expires_ts FROM leases WHERE lease_id = ?", (scr_lease["lease_id"],)
        ).fetchone()
    assert renewed_row is not None and float(renewed_row["expires_ts"]) > initial_exp + 30.0
    ConflictManager.release_task_leases(db, "t-scr-hb")

    # `main()` on non-RECONCILE stage with expired lease returns 75 and does NOT overwrite task status to DONE!
    main_exp_owner = "task:t-main-nonrec-exp:a1"
    m_lease = db.acquire_branch_ro_lease(
        goal.goal_id, hyp.hypothesis_id, hyp_reloaded.head_sha, "t-main-nonrec-exp", holder=main_exp_owner, ttl_s=60.0
    )
    assert m_lease is not None
    with db._tx() as (conn, _p):
        conn.execute("UPDATE leases SET expires_ts = ? WHERE lease_id = ?", (time.time() - 5.0, m_lease["lease_id"]))
    pin_main_exp = {
        **r_codex.to_dict(),
        "expected_goal_version": 2,
        "head_sha": hyp_reloaded.head_sha,
        "lease_owner_id": main_exp_owner,
        "acquired_lease_keys": [m_lease["resource_key"]],
    }
    t_main_exp = StageTask(
        task_id="t-main-nonrec-exp",
        idempotency_key="k-main-nonrec-exp",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.IMPL_REVIEW,
        status="DISPATCHED",
        goal_version=2,
        deps=[],
        worktree_path=str(wt_path),
        pinned_route=pin_main_exp,
        resources=ResourceSpec(llm_pool=r_codex.canonical_spec(), pinned_route=pin_main_exp, wall_timeout_s=30),
    )
    db.enqueue_task(t_main_exp)
    db.close()

    rc_main_exp = stage_runner_main(
        ["--task-id", "t-main-nonrec-exp", "--arr-home", str(arr_home), "--dry-run-llm"]
    )
    assert rc_main_exp == 75
    db = StateDB(arr_home)
    assert db.get_task("t-main-nonrec-exp").status == "DISPATCHED"
    db.close()


def test_p1_05_r3_scoped_branch_workflow_task_policies_and_stale_pinned_route_revalidation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conjectra.types import WorkflowNode

    origin = tmp_path / "origin_p105_r3"
    base_sha = _init_git_repo(origin)
    arr_home = tmp_path / "arr_home_p105_r3"
    db = StateDB(arr_home)
    arb = ResourceArbiter(
        quotas_path=tmp_path / "nq.json",
        ledger_path=tmp_path / "nl.json",
        gemini_quota_path=tmp_path / "nv.json",
    )
    router = ModelRouter(arbiter=arb, db=db, arr_home=arr_home)
    sb = SandboxManager(sandbox_base=tmp_path / "sb_p105", worktree_base=tmp_path / "wt_p105")
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb, model_router=router)

    # Goal has NO engineer restriction (only disallows grok globally on goal)
    goal = _make_goal(
        origin,
        base_sha,
        goal_id="g-scoped-r3",
        routing_policy={"disallowed_models": ["grok-code-fast"]},
    )
    db.save_goal(goal)

    # Branch sets HARD engineer_builder -> gpt-5-codex:high and disallows claude-sonnet-5-5!
    # Workflow wf-1 overrides engineer_builder -> gemini-pro:high WITHOUT leaking to sibling wf-2!
    hyp = HypothesisNode.from_dict(
        {
            **_make_hyp(goal.goal_id, base_sha, hyp_id="h-scoped-r3").to_dict(),
            "params": {
                "routing_policy": {
                    "rules": [
                        {
                            "rule_id": "branch-hard-codex",
                            "roles": ["engineer_builder"],
                            "strength": "HARD",
                            "candidates": [{"provider_key": "gpt-5-codex", "effort": "high"}],
                        }
                    ],
                    "disallowed_models": ["claude-sonnet-5-5"],
                },
                "workflow_routing_policies": {
                    "wf-1": {
                        "rules": [
                            {
                                "rule_id": "wf1-hard-gemini",
                                "roles": ["engineer_builder"],
                                "strength": "HARD",
                                "candidates": [{"provider_key": "gemini-pro", "effort": "high"}],
                            }
                        ]
                    }
                },
            },
        }
    )
    db.register_hypothesis(hyp)
    db.save_workflow(
        WorkflowNode(
            workflow_id="wf-1",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            primitive="direct_hq",
            config={
                "routing_policy": hyp.params["workflow_routing_policies"]["wf-1"],
            },
        )
    )
    db.save_workflow(
        WorkflowNode(
            workflow_id="wf-2",
            goal_id=goal.goal_id,
            hypothesis_id=hyp.hypothesis_id,
            primitive="direct_hq",
            config={},
        )
    )

    # 1. Branch-level resolution (or sibling wf-2) selects branch HARD gpt-5-codex:high (NOT default sonnet!)
    r_branch = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER,
        goal=goal,
        hyp=hyp,
        workflow_id="wf-2",
        execution_host="local",
    )
    assert r_branch.canonical_model == "gpt-5-codex"
    assert r_branch.matched_rule_id == "branch-hard-codex"

    # 2. Workflow wf-1 narrowest-scope HARD rule selects gemini-pro while still inheriting Goal+Branch disallows!
    r_wf1 = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER,
        goal=goal,
        hyp=hyp,
        workflow_id="wf-1",
        execution_host="local",
    )
    assert r_wf1.canonical_model == "gemini-pro"
    assert r_wf1.matched_rule_id == "wf1-hard-gemini"

    eff_wf1 = router.resolve_effective_policy(goal=goal, hyp=hyp, workflow_id="wf-1")
    assert "grok-code-fast" in eff_wf1["disallowed_bundle"]["models"]
    assert "claude-sonnet-5-5" in eff_wf1["disallowed_bundle"]["models"]

    # 3. Stale pinned_route (claude-sonnet-5-5 or gemini-flash) on branch with HARD gpt-5-codex MUST fail revalidation + dry_run!
    stale_pin = {
        "backend_mode": "gemini_cli",
        "execution_host": "local",
        "multipool_host_arg": "local",
        "pool": "gemini",
        "canonical_model": "gemini-flash",
        "requested_effort": "high",
        "wire_effort": "high",
        "family": "google",
        "logical_pool": "llm_gemini",
        "hq_resource": "llmgemini",
        "strength": "SOFT_FALLBACK",
        "matched_rule_id": "role_default",
    }
    t_stale_pin = StageTask(
        task_id="t-stale-pin",
        idempotency_key="k-stale-pin",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        workflow_id="wf-2",
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        pinned_route=stale_pin,
        resources=ResourceSpec(llm_pool="llm_gemini", pinned_route=stale_pin, wall_timeout_s=30),
    )
    db.enqueue_task(t_stale_pin)
    rcpt_stale = runner.run_stage(t_stale_pin, hyp, goal, db=db, dry_run_llm=True, notify=False)
    assert rcpt_stale.exit_code == 78
    assert rcpt_stale.process_status == ProcessStatus.ENV_ERROR
    assert "violates active HARD rule" in rcpt_stale.failure_reason

    # 4. Unpinned task with generic `llm_claude` resource bucket on branch with HARD gpt-5-codex uses `gpt-5-codex` in dry-run!
    t_dry_sol = StageTask(
        task_id="t-dry-codex",
        idempotency_key="k-dry-codex",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        workflow_id="wf-2",
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        resources=ResourceSpec(llm_pool="llm_claude", wall_timeout_s=30),
    )
    db.enqueue_task(t_dry_sol)
    rcpt_dry = runner.run_stage(t_dry_sol, hyp, goal, db=db, dry_run_llm=True, notify=False)
    assert rcpt_dry.process_status == ProcessStatus.EXIT_OK
    assert rcpt_dry.model_used == "gpt-5-codex"

    # 5. Unpinned BLIND_AUDIT (`llm_pool=""`) unconditionally routes via ModelRouter (`IMPL_REVIEWER` for boost_adversary_*)!
    hyp_blind_pol = HypothesisNode.from_dict(
        {
            **_make_hyp(goal.goal_id, base_sha, hyp_id="h-blind-pol").to_dict(),
            "params": {
                "routing_policy": {
                    "rules": [
                        {
                            "rule_id": "branch-hard-impl-rev",
                            "roles": ["impl_reviewer"],
                            "strength": "HARD",
                            "candidates": [{"provider_key": "claude-opus-5-5", "effort": "high"}],
                        }
                    ]
                }
            },
        }
    )
    db.register_hypothesis(hyp_blind_pol)
    t_blind = StageTask(
        task_id="t-blind-unpinned",
        idempotency_key="k-blind-unpinned",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_blind_pol.hypothesis_id,
        stage=StageKind.BLIND_AUDIT,
        step_key="boost_adversary_1",
        status="RUNNING",
        deps=[],
        resources=ResourceSpec(llm_pool="", wall_timeout_s=30),
    )
    db.enqueue_task(t_blind)
    rcpt_blind = runner.run_stage(t_blind, hyp_blind_pol, goal, db=db, dry_run_llm=True, notify=False)
    assert rcpt_blind.process_status == ProcessStatus.EXIT_OK
    assert rcpt_blind.model_used == "claude-opus-5-5"
    db.close()


def test_p1_03_r3_and_p1_06_r3_branch_lineage_stamp_and_serialized_promotion_flock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import fcntl
    from conjectra.types import ExperimentPlan

    monkeypatch.setattr("conjectra.stage_runner.wait_for_mem_available", lambda *a, **kw: (True, 16384))

    origin = tmp_path / "origin_r3_exec"
    base_sha = _init_git_repo(origin)
    arr_home = tmp_path / "arr_home_r3_exec"
    db = StateDB(arr_home)
    sb = SandboxManager(sandbox_base=tmp_path / "sb_r3", worktree_base=tmp_path / "wt_r3")
    arb = ResourceArbiter(
        quotas_path=tmp_path / "nq.json",
        ledger_path=tmp_path / "nl.json",
        gemini_quota_path=tmp_path / "nv.json",
    )
    router = ModelRouter(arbiter=arb, db=db, arr_home=arr_home)
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb, model_router=router)

    # --- Part A: P1-03-R3 unaffected sibling B stamps branch plan_epoch=0 + synced exec_fence_plan_epoch=1 ---
    goal = ResearchGoal(
        goal_id="g-r3-lin",
        project="projgamma",
        title="R3 lineage & promotion",
        direction="minimize",
        primary_metric="val_loss",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(origin),
        kill_criteria=["val_loss > 5.0"],
        banned_paths=[],
        writable_prefixes=["src/"],
        max_tokens_budget=200_000,
        version=1,
        human_epoch=1,
        plan_epoch=1,
    )
    db.save_goal(goal)

    # Branch B has an unchanged epoch-0 ExperimentPlan while Goal bumped to plan_epoch=1, human_epoch=1 (revising A)
    hyp_b = _make_hyp(
        goal.goal_id,
        base_sha,
        hyp_id="h-sib-b",
        screen_cmd="echo METRIC val_loss=0.42 denom=100",
    )
    hyp_b, _ = db.register_hypothesis(hyp_b)
    ep_b = ExperimentPlan(
        exp_plan_id="ep-sib-b-v1",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_b.hypothesis_id,
        plan_version=1,
        goal_version=1,
        human_epoch=0,
        plan_epoch=0,
        falsifiable_claim="B reduces val_loss below 0.5",
    )
    db.save_experiment_plan(ep_b)

    # B's active EXP_SCREEN task had its execution-fence epochs synced to (human_epoch=1, plan_epoch=1)
    t_b_exp = StageTask(
        task_id="t-sib-b-screen",
        idempotency_key="k-sib-b-screen",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_b.hypothesis_id,
        stage=StageKind.EXP_SCREEN,
        status="RUNNING",
        goal_version=1,
        human_epoch=1,
        plan_epoch=1,
        deps=[],
        worktree_path=str(origin),
        resources=ResourceSpec(wall_timeout_s=30),
    )
    db.enqueue_task(t_b_exp)
    rcpt_b = runner.run_stage(t_b_exp, hyp_b, goal, db=db, notify=False)
    assert rcpt_b.process_status == ProcessStatus.EXIT_OK
    # B's receipt keeps B's unchanged scientific plan lineage (0, 0, rev 0) while carrying synced global fence (1, 1)!
    assert rcpt_b.plan_epoch == 0
    assert rcpt_b.human_epoch == 0
    assert rcpt_b.metrics["sci_revision"]["value"] == 0
    assert rcpt_b.metrics["exec_fence_plan_epoch"]["value"] == 1
    assert rcpt_b.metrics["exec_fence_human_epoch"]["value"] == 1

    # Revised branch A (`lineage_plan_epoch=1, lineage_human_epoch=1, lineage_sci_revision=1`) stamps (1, 1, rev 1)
    hyp_a = HypothesisNode.from_dict(
        {
            **_make_hyp(
                goal.goal_id,
                base_sha,
                hyp_id="h-rev-a",
                screen_cmd="echo METRIC val_loss=0.35 denom=100",
            ).to_dict(),
            "params": {
                "lineage_plan_epoch": 1,
                "lineage_human_epoch": 1,
                "lineage_sci_revision": 1,
            },
        }
    )
    hyp_a, _ = db.register_hypothesis(hyp_a)
    t_a_exp = StageTask(
        task_id="t-rev-a-screen",
        idempotency_key="k-rev-a-screen",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_a.hypothesis_id,
        stage=StageKind.EXP_SCREEN,
        status="RUNNING",
        goal_version=1,
        human_epoch=1,
        plan_epoch=1,
        deps=[],
        worktree_path=str(origin),
        resources=ResourceSpec(wall_timeout_s=30),
    )
    db.enqueue_task(t_a_exp)
    rcpt_a = runner.run_stage(t_a_exp, hyp_a, goal, db=db, notify=False)
    assert rcpt_a.plan_epoch == 1
    assert rcpt_a.human_epoch == 1
    assert rcpt_a.metrics["sci_revision"]["value"] == 1

    # --- Part B: P1-06-R3 interleaved cancellation/replan right after PREPARED (before lock/FF) -> ZERO Git FF ---
    mirror = sb.ensure_mirror(origin, goal.project)
    wt_prom = sb.create_worktree(mirror, goal.project, hyp_b.branch_name, base_sha, origin_repo=origin)

    bin_dir = tmp_path / "prom_bins"
    bin_dir.mkdir()
    fake_codex = bin_dir / "codex"
    fake_codex.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
_ = sys.stdin.read()
Path(os.getcwd(), "src", "core.py").write_text("def compute(x: int) -> int:\\n    return x * 11\\n", encoding="utf-8")
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "edited"}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 80, "output_tokens": 20}}))
""",
        encoding="utf-8",
    )
    fake_codex.chmod(0o755)
    import shutil as _shutil

    monkeypatch.setattr(
        "conjectra.stage_runner.shutil.which",
        lambda name: str(bin_dir / name) if (bin_dir / name).is_file() else _shutil.which(name),
    )

    r_sol = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER,
        explicit_spec="codex:gpt-5-codex:high",
        execution_host="local",
    )
    pin_c1 = {
        **r_sol.to_dict(),
        "expected_goal_version": 1,
        "plan_epoch": 1,
    }
    t_cancel_mid = StageTask(
        task_id="t-prom-cancel-after-prep",
        idempotency_key="k-prom-cancel-after-prep",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_b.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        goal_version=1,
        human_epoch=1,
        plan_epoch=1,
        deps=[],
        worktree_path=str(wt_prom),
        pinned_route=pin_c1,
        resources=ResourceSpec(llm_pool=r_sol.canonical_spec(), pinned_route=pin_c1, wall_timeout_s=30),
    )
    db.enqueue_task(t_cancel_mid)

    def _cancel_between_prepared_and_flock(**kw: object) -> None:
        # At this exact point, PREPARED is committed in SQLite and wt_prom has NOT moved yet!
        assert _git(wt_prom, "rev-parse", "HEAD") == base_sha
        db.transition_task_status("t-prom-cancel-after-prep", "RUNNING", "CANCELLED")

    runner._after_promotion_prepared_hook = _cancel_between_prepared_and_flock  # type: ignore[attr-defined]
    try:
        rcpt_fenced = runner.run_stage(t_cancel_mid, hyp_b, goal, db=db, notify=False)
    finally:
        runner._after_promotion_prepared_hook = None  # type: ignore[attr-defined]

    assert rcpt_fenced.exit_code == 75
    assert rcpt_fenced.process_status == ProcessStatus.LEASE_LOST
    assert "FENCED_PRE_PROMOTE" in rcpt_fenced.failure_reason
    # ZERO Git fast-forward occurred on shared branch worktree!
    assert _git(wt_prom, "rev-parse", "HEAD") == base_sha
    assert db.get_hypothesis(hyp_b.hypothesis_id).head_sha in (None, base_sha)
    with db._lock:
        aborted_row = db._conn.execute(
            "SELECT state FROM promotion_intents WHERE task_id = 't-prom-cancel-after-prep'"
        ).fetchone()
    assert aborted_row is not None and aborted_row["state"] == "ABORTED"

    # --- Part C: Verify `promote_<safe_goal>_<safe_branch>.flock` is held across FF + crash right after FF leaves PREPARED ---
    expected_lock = StageRunner._promotion_lock_path(arr_home, db, goal.goal_id, hyp_b.branch_name)
    t_crash_ff = StageTask(
        task_id="t-prom-crash-after-ff",
        idempotency_key="k-prom-crash-after-ff",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_b.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        goal_version=1,
        human_epoch=1,
        plan_epoch=1,
        deps=[],
        worktree_path=str(wt_prom),
        pinned_route=pin_c1,
        resources=ResourceSpec(llm_pool=r_sol.canonical_spec(), pinned_route=pin_c1, wall_timeout_s=30),
    )
    db.enqueue_task(t_crash_ff)

    class _SimulatedCrashAfterFF(RuntimeError):
        pass

    captured_cand_sha: list[str] = []

    def _crash_right_after_ff(**kw: object) -> None:
        # Probe LOCK_NB on a fresh file description to prove the worker holds `promote_*.flock` right now!
        probe_fd = os.open(str(expected_lock), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe_fd)
        captured_cand_sha.append(str(kw["candidate_sha"]))
        raise _SimulatedCrashAfterFF("worker died after FF before Phase 3 SQLite commit")

    runner._after_promotion_ff_hook = _crash_right_after_ff  # type: ignore[attr-defined]
    try:
        with pytest.raises(_SimulatedCrashAfterFF):
            runner.run_stage(t_crash_ff, hyp_b, goal, db=db, notify=False)
    finally:
        runner._after_promotion_ff_hook = None  # type: ignore[attr-defined]

    # Lock MUST be released upon crash cleanup, wt_prom is at candidate_sha, and intent remains PREPARED for reconciler roll-forward!
    probe_after = os.open(str(expected_lock), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(probe_after, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(probe_after, fcntl.LOCK_UN)
    finally:
        os.close(probe_after)

    assert len(captured_cand_sha) == 1
    cand_sha = captured_cand_sha[0]
    assert _git(wt_prom, "rev-parse", "HEAD") == cand_sha
    prep_intents = db.list_promotion_intents(goal_id=goal.goal_id, state="PREPARED")
    matching_prep = [r for r in prep_intents if r["task_id"] == "t-prom-crash-after-ff"]
    assert len(matching_prep) == 1
    assert matching_prep[0]["candidate_sha"] == cand_sha
    # Reconciler / commit_promotion_intent rolls forward cleanly to COMMITTED + updates hypothesis.head_sha!
    db.commit_promotion_intent(
        matching_prep[0]["intent_id"],
        candidate_sha=cand_sha,
        update_hypothesis_head=True,
        reason="crash_recovered_roll_forward",
    )
    assert db.get_hypothesis(hyp_b.hypothesis_id).head_sha == cand_sha

    # --- Part D (`P1-06-R4`): Unbroken `BEGIN IMMEDIATE` across `[Fresh Check -> FF -> Head+Receipt+COMMITTED]` ---
    # D1. Worker holds `promote_*.flock` and commits Phase-1 `PREPARED`, then `REVISE_HYPOTHESIS` / `update_plan`
    #     cancels the task inside `db._tx()` (where `abort_cancelled_task_promotions` returns 0 via `LOCK_NB`!)
    #     BEFORE Phase-2+3 `BEGIN IMMEDIATE`: worker MUST abort, NEVER run `git merge --ff-only`, and leave BOTH
    #     Git `HEAD` and `hypotheses.head_sha` at `cand_sha`!
    import threading as _threading

    fake_codex.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
_ = sys.stdin.read()
Path(os.getcwd(), "src", "core.py").write_text("def compute(x: int) -> int:\\n    return x * 99\\n", encoding="utf-8")
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "edited2"}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 50, "output_tokens": 15}}))
""",
        encoding="utf-8",
    )
    hyp_b_cur = db.get_hypothesis(hyp_b.hypothesis_id)
    assert hyp_b_cur is not None
    t_rev_race = StageTask(
        task_id="t-prom-r4-revise-race",
        idempotency_key="k-prom-r4-revise-race",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_b.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        goal_version=1,
        human_epoch=1,
        plan_epoch=1,
        deps=[],
        worktree_path=str(wt_prom),
        pinned_route=pin_c1,
        resources=ResourceSpec(llm_pool=r_sol.canonical_spec(), pinned_route=pin_c1, wall_timeout_s=30),
    )
    db.enqueue_task(t_rev_race)

    def _simulate_revise_hypothesis_while_flock_held(**kw: object) -> None:
        # Prove `promote_*.flock` IS already held by worker right now (`abort_cancelled_task_promotions` returns 0 inside `db._tx()`)!
        probe_fd2 = os.open(str(expected_lock), os.O_RDWR | os.O_CREAT, 0o644)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(probe_fd2, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(probe_fd2)
        with db._tx():
            db.transition_task_status("t-prom-r4-revise-race", "RUNNING", "CANCELLED")
            aborted_n = db.abort_cancelled_task_promotions(
                goal.goal_id,
                hyp_b.hypothesis_id,
                "t-prom-r4-revise-race",
                reason="revised_mid_promote",
            )
            assert aborted_n == 0

    runner._after_promotion_prepared_hook = _simulate_revise_hypothesis_while_flock_held  # type: ignore[attr-defined]
    try:
        rcpt_r4_race = runner.run_stage(t_rev_race, hyp_b_cur, goal, db=db, notify=False)
    finally:
        runner._after_promotion_prepared_hook = None  # type: ignore[attr-defined]

    assert rcpt_r4_race.exit_code == 75
    assert rcpt_r4_race.process_status == ProcessStatus.LEASE_LOST
    assert "FENCED_PRE_PROMOTE" in rcpt_r4_race.failure_reason
    # ZERO Git FF and ZERO `hypotheses.head_sha` advancement!
    assert _git(wt_prom, "rev-parse", "HEAD") == cand_sha
    assert db.get_hypothesis(hyp_b.hypothesis_id).head_sha == cand_sha
    assert rcpt_r4_race.head_sha == cand_sha
    with db._lock:
        r4_aborted = db._conn.execute(
            "SELECT state FROM promotion_intents WHERE task_id = 't-prom-r4-revise-race'"
        ).fetchone()
    assert r4_aborted is not None and r4_aborted["state"] == "ABORTED"

    # D2. Concurrent SQLite writer attempting `BEGIN IMMEDIATE` WHILE worker is inside Phase-2+3 `BEGIN IMMEDIATE`
    #     waits cleanly on `busy_timeout` until `[FF -> head_sha + COMMITTED + receipt]` commits atomically!
    t_concurrent_wait = StageTask(
        task_id="t-prom-r4-busy-wait",
        idempotency_key="k-prom-r4-busy-wait",
        goal_id=goal.goal_id,
        hypothesis_id=hyp_b.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        goal_version=1,
        human_epoch=1,
        plan_epoch=1,
        deps=[],
        worktree_path=str(wt_prom),
        pinned_route=pin_c1,
        resources=ResourceSpec(llm_pool=r_sol.canonical_spec(), pinned_route=pin_c1, wall_timeout_s=30),
    )
    db.enqueue_task(t_concurrent_wait)

    writer_started = _threading.Event()
    writer_events: list[tuple[str, str]] = []
    bg_threads: list[_threading.Thread] = []

    def _spawn_competing_writer_inside_authority_tx(**kw: object) -> None:
        def _bg_writer() -> None:
            db2 = StateDB(arr_home)
            try:
                writer_started.set()
                with db2._tx():
                    cur_h = db2.get_hypothesis(hyp_b.hypothesis_id)
                    writer_events.append(("committed_after_prom", str(cur_h.head_sha if cur_h else "")))
            finally:
                db2.close()

        th = _threading.Thread(target=_bg_writer, daemon=True)
        bg_threads.append(th)
        th.start()
        assert writer_started.wait(timeout=2.0)
        time.sleep(0.06)
        # Competing writer MUST still be blocked waiting on our `BEGIN IMMEDIATE` lock before FF runs!
        assert writer_events == []

    runner._in_promotion_authority_tx_before_ff_hook = _spawn_competing_writer_inside_authority_tx  # type: ignore[attr-defined]
    try:
        rcpt_ok_r4 = runner.run_stage(t_concurrent_wait, hyp_b_cur, goal, db=db, notify=False)
    finally:
        runner._in_promotion_authority_tx_before_ff_hook = None  # type: ignore[attr-defined]
        for th in bg_threads:
            th.join(timeout=5.0)

    assert rcpt_ok_r4.process_status == ProcessStatus.EXIT_OK
    final_wt_head = _git(wt_prom, "rev-parse", "HEAD")
    assert final_wt_head != cand_sha
    assert db.get_hypothesis(hyp_b.hypothesis_id).head_sha == final_wt_head
    # Competing writer unblocked cleanly AFTER Phase-2+3 committed and observed the new `final_wt_head`!
    assert writer_events == [("committed_after_prom", final_wt_head)]
    db.close()


def test_stage4_p3_slash_opus_default_routing_and_policy_preservation(tmp_path: Path) -> None:
    arb = ResourceArbiter(
        quotas_path=tmp_path / "no_q.json",
        ledger_path=tmp_path / "no_l.json",
        gemini_quota_path=tmp_path / "no_v.json",
    )
    router = ModelRouter(arbiter=arb)

    # 1. RESEARCH_LEAD & EXPERIMENT_DESIGNER default to gpt-5-codex:high (never Opus)
    for role in (ScientificRole.RESEARCH_LEAD, ScientificRole.EXPERIMENT_DESIGNER):
        r = router.resolve_route(role, execution_host="local")
        assert r.canonical_model == "gpt-5-codex"
        assert r.wire_effort == "high"
        # Avoiding openai falls back to gemini-flash
        r_no_oai = router.resolve_route(role, avoid_families=["openai"], execution_host="local")
        assert r_no_oai.canonical_model == "gemini-flash"
        assert r_no_oai.wire_effort == "high"
        # Avoiding openai + google falls back to claude-sonnet-5-5:medium (never Opus)
        r_anth = router.resolve_route(role, avoid_families=["openai", "google"], execution_host="local")
        assert r_anth.canonical_model == "claude-sonnet-5-5"
        assert r_anth.wire_effort == "medium"

    # 2. DESIGN_REVIEWER, IMPL_REVIEWER, RESULT_AUDITOR never default to Opus across fallback chain
    for rev_role in (
        ScientificRole.DESIGN_REVIEWER,
        ScientificRole.IMPL_REVIEWER,
        ScientificRole.RESULT_AUDITOR,
    ):
        r1 = router.resolve_route(rev_role, execution_host="local")
        assert r1.canonical_model == "gpt-5-codex"
        r2 = router.resolve_route(rev_role, avoid_families=["openai"], execution_host="local")
        assert r2.canonical_model == "gemini-pro"
        r3 = router.resolve_route(rev_role, avoid_families=["openai", "google"], execution_host="local")
        assert r3.canonical_model == "grok-code-fast"
        r4 = router.resolve_route(
            rev_role, avoid_families=["openai", "google", "xai"], execution_host="local"
        )
        assert r4.canonical_model == "claude-sonnet-5-5"
        assert r4.wire_effort == "medium"

    # 3. Generic `claude` / `llm_claude` pool tokens map to sonnet for lead/reviewers, Opus strictly for director/theorist
    assert router.resolve_route(ScientificRole.RESEARCH_LEAD, explicit_spec="llm_claude", execution_host="local").canonical_model == "claude-sonnet-5-5"
    assert router.resolve_route(ScientificRole.DESIGN_REVIEWER, explicit_spec="claude", execution_host="local").canonical_model == "claude-sonnet-5-5"
    assert router.resolve_route(ScientificRole.RESEARCH_DIRECTOR, execution_host="local").canonical_model == "claude-opus-5-5"
    assert router.resolve_route(ScientificRole.THEORIST, execution_host="local").canonical_model == "claude-opus-5-5"


def test_stage5_p1_04_r5_ff_timeout_after_ref_update_and_ambiguous_head(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from conjectra import stage_runner as sr_mod

    monkeypatch.setattr("conjectra.stage_runner.wait_for_mem_available", lambda *a, **kw: (True, 16384))

    origin = tmp_path / "origin_r5_ff"
    base_sha = _init_git_repo(origin)
    arr_home = tmp_path / "arr_home_r5_ff"
    db = StateDB(arr_home)
    sb = SandboxManager(sandbox_base=tmp_path / "sb_r5", worktree_base=tmp_path / "wt_r5")
    arb = ResourceArbiter(
        quotas_path=tmp_path / "nq.json",
        ledger_path=tmp_path / "nl.json",
        gemini_quota_path=tmp_path / "nv.json",
    )
    router = ModelRouter(arbiter=arb, db=db, arr_home=arr_home)
    runner = StageRunner(arr_home=arr_home, sandbox_manager=sb, model_router=router)

    goal = ResearchGoal(
        goal_id="g-r5-ff",
        project="projgamma",
        title="R5 FF timeout safety",
        direction="minimize",
        primary_metric="val_loss",
        baseline_value=1.0,
        baseline_sha=base_sha,
        repo_path=str(origin),
        kill_criteria=[],
        banned_paths=[],
        writable_prefixes=["src/"],
        max_tokens_budget=200_000,
        version=1,
        human_epoch=0,
        plan_epoch=0,
    )
    db.save_goal(goal)
    hyp = _make_hyp(goal.goal_id, base_sha, hyp_id="h-r5-ff")
    hyp, _ = db.register_hypothesis(hyp)
    mirror = sb.ensure_mirror(origin, goal.project)
    wt = sb.create_worktree(mirror, goal.project, hyp.branch_name, base_sha, origin_repo=origin, goal_id=goal.goal_id)

    bin_dir = tmp_path / "fake_codex_r5"
    bin_dir.mkdir()
    fake_codex = bin_dir / "codex"
    fake_codex.write_text(
        """#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
_ = sys.stdin.read()
cwd = Path(os.getcwd())
cur = (cwd / "src" / "core.py").read_text(encoding="utf-8")
(cwd / "src" / "core.py").write_text(cur + "# step\\n", encoding="utf-8")
print(json.dumps({"type": "item.completed", "item": {"type": "agent_message", "text": "done"}}))
print(json.dumps({"type": "turn.completed", "usage": {"input_tokens": 60, "output_tokens": 20}}))
""",
        encoding="utf-8",
    )
    fake_codex.chmod(fake_codex.stat().st_mode | stat.S_IXUSR)
    import shutil as _shutil

    monkeypatch.setattr(
        "conjectra.stage_runner.shutil.which",
        lambda name: str(fake_codex) if name == "codex" else _shutil.which(name),
    )

    pin_sol = router.resolve_route(
        ScientificRole.ENGINEER_BUILDER,
        explicit_spec="codex:gpt-5-codex:high",
        execution_host="local",
    ).to_dict()

    real_git = sr_mod._git
    hook_disabled_calls: list[tuple[str, ...]] = []
    mode = {"val": "timeout_after_advance"}

    def _spy_git(repo: Path, *args: str, check: bool = False, timeout: float | None = None):
        if "merge" in args and "--ff-only" in args:
            hook_disabled_calls.append(tuple(args))
            assert "core.hooksPath=/dev/null" in args
            if mode["val"] == "timeout_after_advance":
                real_git(repo, *args, check=True, timeout=timeout)
                raise subprocess.TimeoutExpired(cmd=["git", "merge"], timeout=15.0)
            if mode["val"] in ("timeout_unchanged", "timeout_unreadable_head", "timeout_revparse_timeout"):
                raise subprocess.TimeoutExpired(cmd=["git", "merge"], timeout=15.0)
        if mode["val"] == "timeout_unreadable_head" and args == ("rev-parse", "HEAD") and hook_disabled_calls:
            return subprocess.CompletedProcess(args=["git", "rev-parse", "HEAD"], returncode=128, stdout="", stderr="ambiguous read fail")
        if mode["val"] == "timeout_revparse_timeout" and args == ("rev-parse", "HEAD") and hook_disabled_calls:
            raise subprocess.TimeoutExpired(cmd=["git", "rev-parse"], timeout=10.0)
        return real_git(repo, *args, check=check, timeout=timeout)

    monkeypatch.setattr(sr_mod, "_git", _spy_git)

    # Case 1: FF advances branch ref and THEN times out -> re-reading HEAD discovers candidate_sha -> rolls forward COMMITTED!
    t1 = StageTask(
        task_id="t-r5-ff-adv",
        idempotency_key="k-r5-ff-adv",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        worktree_path=str(wt),
        pinned_route=pin_sol,
        resources=ResourceSpec(llm_pool="codex:gpt-5-codex:high", pinned_route=pin_sol, wall_timeout_s=30),
    )
    db.enqueue_task(t1)
    rcpt1 = runner.run_stage(t1, hyp, goal, db=db, notify=False)
    head_after_1 = _git(wt, "rev-parse", "HEAD")
    assert len(hook_disabled_calls) == 1
    assert head_after_1 != base_sha
    assert rcpt1.process_status == ProcessStatus.EXIT_OK
    assert rcpt1.head_sha == head_after_1
    assert db.get_hypothesis(hyp.hypothesis_id).head_sha == head_after_1
    with db._lock:
        r1_row = db._conn.execute(
            "SELECT state FROM promotion_intents WHERE task_id = 't-r5-ff-adv'"
        ).fetchone()
    assert r1_row is not None and r1_row["state"] == "COMMITTED"
    assert _git(wt, "for-each-ref", "refs/arr-promote/") == ""

    # Case 2: FF times out BEFORE changing HEAD (HEAD == expected_base_sha confirmed) -> cleanly marks ABORTED!
    mode["val"] = "timeout_unchanged"
    hyp2 = db.get_hypothesis(hyp.hypothesis_id)
    assert hyp2 is not None
    t2 = StageTask(
        task_id="t-r5-ff-unchanged",
        idempotency_key="k-r5-ff-unchanged",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        worktree_path=str(wt),
        pinned_route=pin_sol,
        resources=ResourceSpec(llm_pool="codex:gpt-5-codex:high", pinned_route=pin_sol, wall_timeout_s=30),
    )
    db.enqueue_task(t2)
    rcpt2 = runner.run_stage(t2, hyp2, goal, db=db, notify=False)
    assert rcpt2.process_status == ProcessStatus.EXIT_NONZERO
    assert rcpt2.head_sha == head_after_1
    assert db.get_hypothesis(hyp.hypothesis_id).head_sha == head_after_1
    with db._lock:
        r2_row = db._conn.execute(
            "SELECT state FROM promotion_intents WHERE task_id = 't-r5-ff-unchanged'"
        ).fetchone()
    assert r2_row is not None and r2_row["state"] == "ABORTED"

    # Case 3: FF times out and post-timeout HEAD cannot be read -> keeps durable PREPARED intent & promote_ref pin!
    mode["val"] = "timeout_unreadable_head"
    hook_disabled_calls.clear()
    hyp3 = db.get_hypothesis(hyp.hypothesis_id)
    assert hyp3 is not None
    t3 = StageTask(
        task_id="t-r5-ff-ambig",
        idempotency_key="k-r5-ff-ambig",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        worktree_path=str(wt),
        pinned_route=pin_sol,
        resources=ResourceSpec(llm_pool="codex:gpt-5-codex:high", pinned_route=pin_sol, wall_timeout_s=30),
    )
    db.enqueue_task(t3)
    rcpt3 = runner.run_stage(t3, hyp3, goal, db=db, notify=False)
    assert rcpt3.process_status == ProcessStatus.EXIT_NONZERO
    assert "UNRESOLVED_FF_AMBIGUOUS" in rcpt3.failure_reason
    with db._lock:
        r3_row = db._conn.execute(
            "SELECT state, reason FROM promotion_intents WHERE task_id = 't-r5-ff-ambig'"
        ).fetchone()
    assert r3_row is not None and r3_row["state"] == "PREPARED"
    assert "UNRESOLVED_FF_AMBIGUOUS" in str(r3_row["reason"])
    assert "refs/arr-promote/" in _git(wt, "for-each-ref", "refs/arr-promote/")

    # Case 4: Subsequent task t4 on hyp3 is fenced while unresolved PREPARED intent persists!
    hyp4 = db.get_hypothesis(hyp.hypothesis_id)
    assert hyp4 is not None
    t4 = StageTask(
        task_id="t-r5-ff-fenced-after-unresolved",
        idempotency_key="k-r5-ff-fenced-after-unresolved",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        worktree_path=str(wt),
        pinned_route=pin_sol,
        resources=ResourceSpec(llm_pool="codex:gpt-5-codex:high", pinned_route=pin_sol, wall_timeout_s=30),
    )
    db.enqueue_task(t4)
    rcpt4 = runner.run_stage(t4, hyp4, goal, db=db, notify=False)
    assert rcpt4.exit_code == 75
    assert rcpt4.process_status == ProcessStatus.LEASE_LOST
    assert "FENCED_UNRESOLVED_PROMOTION" in rcpt4.failure_reason

    # Case 5: Post-FF rev-parse TimeoutExpired falls through cleanly to UNRESOLVED instead of rolling back tx!
    with db._tx() as (conn, _p):
        conn.execute("DELETE FROM promotion_intents WHERE task_id = 't-r5-ff-ambig'")
    mode["val"] = "timeout_revparse_timeout"
    hook_disabled_calls.clear()
    t5 = StageTask(
        task_id="t-r5-ff-revparse-timeout",
        idempotency_key="k-r5-ff-revparse-timeout",
        goal_id=goal.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=StageKind.CODE,
        status="RUNNING",
        deps=[],
        worktree_path=str(wt),
        pinned_route=pin_sol,
        resources=ResourceSpec(llm_pool="codex:gpt-5-codex:high", pinned_route=pin_sol, wall_timeout_s=30),
    )
    db.enqueue_task(t5)
    rcpt5 = runner.run_stage(t5, hyp4, goal, db=db, notify=False)
    assert rcpt5.process_status == ProcessStatus.EXIT_NONZERO
    assert "UNRESOLVED_FF_AMBIGUOUS" in rcpt5.failure_reason
    with db._lock:
        r5_row = db._conn.execute(
            "SELECT state, reason FROM promotion_intents WHERE task_id = 't-r5-ff-revparse-timeout'"
        ).fetchone()
    assert r5_row is not None and r5_row["state"] == "PREPARED"
    assert "UNRESOLVED_FF_AMBIGUOUS" in str(r5_row["reason"])
    db.close()
