"""Tests for DecisionEngine (L0/L1/L2), WorkflowRuntime/WorkflowSelector, and multi-gen Explore."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import pytest

from conjectra.arbiter import ResourceArbiter
from conjectra.decision_engine import (
    MAX_DIGEST_BYTES,
    DecisionEngine,
    build_goal_digest,
)
from conjectra.explore import (
    _build_ai_propose_prompt,
    propose_hypotheses,
    reconcile_receipt,
)
from conjectra.state_db import StateDB
from conjectra.types import (
    AuditVerdict,
    EngineeringVerdict,
    EvidenceReceipt,
    EvidenceValidity,
    FidelityRung,
    HypothesisKind,
    HypothesisNode,
    ProcessStatus,
    ResearchGoal,
    ScientificVerdict,
    StageKind,
)
from conjectra.workload_matrix import record_workload_eval
from conjectra.workflow_runtime import WorkflowRuntime, WorkflowSelector


def _make_goal(
    goal_id: str = "g_dec_1",
    *,
    version: int = 1,
    human_epoch: int = 0,
    autonomy_mode: str = "unattended",
    primary_metric: str = "score",
    direction: str = "maximize",
    baseline_value: float = 0.5,
) -> ResearchGoal:
    return ResearchGoal(
        goal_id=goal_id,
        project="demo_proj",
        title="Improve score cleanly",
        direction=direction,
        primary_metric=primary_metric,
        baseline_value=baseline_value,
        baseline_sha="deadbeef1234",
        repo_path="/tmp/demo_repo",
        kill_criteria=["tests_passed == 0"],
        banned_paths=["lockbox/", "sealed_eval/"],
        writable_prefixes=["src/", "tests/"],
        max_tokens_budget=200_000,
        max_concurrent_branches=4,
        adapter_name="generic",
        version=version,
        human_epoch=human_epoch,
        autonomy_mode=autonomy_mode,
    )


def _make_hyp(
    goal: ResearchGoal,
    hid: str = "h_gen1",
    *,
    parent_id: str | None = None,
    generation: int = 1,
    kind: HypothesisKind = HypothesisKind.COMPETING_ROUTE,
    writable_paths: list[str] | None = None,
    params: dict[str, Any] | None = None,
) -> HypothesisNode:
    merged_params = {"generation": generation, **(params or {})}
    return HypothesisNode(
        hypothesis_id=hid,
        goal_id=goal.goal_id,
        parent_id=parent_id,
        kind=kind,
        title=f"Hyp {hid}",
        statement=f"Optimize {goal.primary_metric} via {hid}",
        mechanism=f"mech_{hid}",
        fidelity=FidelityRung.F1_FAST_SCREEN,
        priority_evoi=0.8,
        writable_paths=writable_paths or ["src/algo.py"],
        params=merged_params,
        kill_criteria=list(goal.kill_criteria),
        acceptance_cmd="pytest -q",
        screen_cmd="pytest -q",
        confirm_cmd="pytest -q",
        branch_name=f"arr/{goal.project}/{hid}",
        base_sha=goal.baseline_sha,
        fingerprint=f"fp_{hid}",
    )


def _make_receipt(
    hyp: HypothesisNode,
    *,
    receipt_id: str = "r_1",
    task_id: str = "t_1",
    stage: StageKind = StageKind.EXP_SCREEN,
    process_status: ProcessStatus = ProcessStatus.EXIT_OK,
    engineering_verdict: EngineeringVerdict = EngineeringVerdict.PASS,
    scientific_verdict: ScientificVerdict = ScientificVerdict.CONFIRMED,
    audit_verdict: AuditVerdict = AuditVerdict.NOT_REQUIRED,
    metrics: dict[str, Any] | None = None,
    failure_reason: str = "",
) -> EvidenceReceipt:
    return EvidenceReceipt(
        receipt_id=receipt_id,
        task_id=task_id,
        goal_id=hyp.goal_id,
        hypothesis_id=hyp.hypothesis_id,
        stage=stage,
        base_sha=hyp.base_sha,
        head_sha="cafebabe1234",
        command="pytest -q",
        exit_code=0 if process_status == ProcessStatus.EXIT_OK else 1,
        wall_sec=3.0,
        max_rss_mb=128.0,
        in_tokens=0,
        out_tokens=0,
        model_used="test",
        process_status=process_status,
        engineering_verdict=engineering_verdict,
        scientific_verdict=scientific_verdict,
        audit_verdict=audit_verdict,
        tests_passed=3 if engineering_verdict == EngineeringVerdict.PASS else 0,
        tests_total=3,
        metrics=metrics or {"score": {"value": 0.75, "denominator": 1, "stage_rung": "F1_FAST_SCREEN"}},
        raw_artifact_paths=["/tmp/art.json"],
        failure_reason=failure_reason,
        evidence_validity=EvidenceValidity.VALID_EMPIRICAL,
        goal_version=1,
    )


def test_decision_engine_tiered_l0_l1_l2_and_foreground_reserve(tmp_path: Path) -> None:
    arr_home = tmp_path / "arr_home"
    quotas_file = tmp_path / "quotas.json"
    gemini_quota_file = tmp_path / "gemini-quota.json"

    # Claude is at 90% used (exceeds 100 - 15 = 85% reserve limit for L2),
    # DeepSeek is at 98% used (unavailable), Codex is at 20% used (healthy).
    quotas_file.write_text(
        json.dumps(
            {
                "claude/acct1": {"quota": {"windows": [{"name": "Claude 7 day", "used": 90.0}]}},
                "deepseek/acct1": {"quota": {"windows": [{"name": "7d", "used": 98.0}]}},
                "codex/acct1": {"quota": {"windows": [{"name": "7d", "used": 20.0}]}},
            }
        ),
        encoding="utf-8",
    )
    gemini_quota_file.write_text(
        json.dumps({"pools": {"gemini": {"weekly": {"remaining_fraction": 0.6}}}}),
        encoding="utf-8",
    )
    arbiter = ResourceArbiter(quotas_path=quotas_file, gemini_quota_path=gemini_quota_file)

    db = StateDB(arr_home)
    try:
        goal = _make_goal("g_tiers")
        db.save_goal(goal)
        hyp = _make_hyp(goal, "h_root")
        db.register_hypothesis(hyp)

        # Record real LLM eval ledger favoring gemini for L1
        usage_file = tmp_path / "real_usage.json"
        usage_file.write_text(json.dumps({"model": "gemini:gemini-flash", "usage": {"input_tokens": 500, "output_tokens": 200}}), encoding="utf-8")
        record_workload_eval(
            db,
            workload_class="disjoint_parallel",
            mode="direct_hq",
            provider_pairing="gemini:gemini-flash",
            is_real_llm=True,
            hq_used=True,
            holdout_cmd="pytest -q",
            holdout_rc=0,
            holdout_passed=True,
            defect_escaped=False,
            gate_intercepted=False,
            uncached_tokens=700,
            raw_usage_paths=[str(usage_file)],
            source="test_real",
        )

        engine = DecisionEngine(db, arbiter=arbiter, foreground_reserve_quota_pct=15.0)

        # L0: 0 tokens deterministic
        rcpt = _make_receipt(hyp, scientific_verdict=ScientificVerdict.CONFIRMED)
        l0_out = engine.evaluate_l0(goal, hyp, rcpt)
        assert l0_out["tier"] == "L0"
        assert l0_out["tokens_used"] == 0
        assert l0_out["decision"]["action"] == "PROMOTE_F2"

        # L1 skips unavailable deepseek (98% used) and picks gemini (40% used + ledger winner)
        l1_prov = engine.select_provider_for_tier("L1")
        assert l1_prov.pool == "gemini"

        # L2 skips claude (90% >= 85% foreground reserve) and gemini once gemini hits 90% used, picking codex (20%)
        gemini_quota_file.write_text(
            json.dumps({"pools": {"gemini": {"weekly": {"remaining_fraction": 0.1}}}}),
            encoding="utf-8",
        )
        l2_prov = engine.select_provider_for_tier("L2")
        assert l2_prov.pool == "codex"
    finally:
        db.close()


def test_decision_engine_coalescing_nonblocking_and_human_epoch_discard(tmp_path: Path) -> None:
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    try:
        g1 = _make_goal("g1", human_epoch=1)
        g2 = _make_goal("g2", human_epoch=1)
        db.save_goal(g1)
        db.save_goal(g2)
        h1 = _make_hyp(g1, "h1")
        h2 = _make_hyp(g2, "h2")
        db.register_hypothesis(h1)
        db.register_hypothesis(h2)

        # Compact digest strictly <4KB even with lessons
        for i in range(15):
            db.upsert_knowledge_item(g1.goal_id, "valid_negative", f"k{i}", {"detail": "x" * 300})
        digest = build_goal_digest(db, g1)
        assert len(json.dumps(digest).encode("utf-8")) <= MAX_DIGEST_BYTES

        engine = DecisionEngine(db)

        # Coalesced requests share coalesce_key and wait coalesce_window_s=12.0
        t0 = 1000.0
        r1a = engine.enqueue_decision(g1.goal_id, tier="L1", urgency="COALESCED", coalesce_window_s=12.0, now=t0)
        r1b = engine.enqueue_decision(g1.goal_id, tier="L1", urgency="COALESCED", coalesce_window_s=12.0, payload={"extra": 2}, now=t0 + 2.0)
        assert r1a["req_id"] == r1b["req_id"]
        assert db.list_decision_requests(g1.goal_id, status="PENDING", ready_before_ts=t0 + 5.0) == []
        assert len(db.list_decision_requests(g1.goal_id, status="PENDING", ready_before_ts=t0 + 13.0)) == 1

        # Independent Goal g2 IMMEDIATE request executes cleanly while g1 is pending
        r2 = engine.enqueue_decision(g2.goal_id, tier="L1", urgency="IMMEDIATE", now=t0)
        assert r2["available_ts"] <= t0
        res2 = engine.process_request(r2["req_id"], dry_run_llm=True)
        assert res2["status"] == "RESOLVED"
        assert len(res2["resolution"]["accepted_hypothesis_ids"]) == 1

        # Optimistic human-priority check: simulate human intervention while L2 LLM is thinking
        r1_imm = engine.enqueue_decision(g1.goal_id, tier="L2", trigger_kind="STALLED_PIVOT", urgency="IMMEDIATE", now=t0)

        def _human_intervenes_midflight(prov: Any, prompt: str) -> dict[str, Any]:
            g_updated = db.get_goal(g1.goal_id)
            assert g_updated is not None
            g_updated.human_epoch += 1
            db.save_goal(g_updated)
            return {
                "exit_code": 0,
                "response": json.dumps(
                    [
                        {
                            "kind": "FALSIFICATION",
                            "title": "Stale pivot",
                            "statement": "Should be discarded",
                            "mechanism": "stale_pivot",
                            "writable_paths": ["src/algo.py"],
                            "params": {"generation": 3},
                            "kill_criteria": ["tests_passed == 0"],
                            "screen_cmd": "pytest -q",
                            "acceptance_cmd": "pytest -q",
                        }
                    ]
                ),
                "in_tokens": 100,
                "out_tokens": 50,
                "model_used": prov.label(),
            }

        res1 = engine.process_request(r1_imm["req_id"], llm_caller=_human_intervenes_midflight)
        assert res1["status"] == "DISCARDED_BY_NEWER_HUMAN_INTERVENTION"
        # Ensure stale proposal was NOT committed
        assert all(h.mechanism != "stale_pivot" for h in db.list_hypotheses(goal_id=g1.goal_id))

        # CLI standalone execution (`python3 -m conjectra.decision_engine --req-id ... --arr-home ...`)
        r_cli = engine.enqueue_decision(g2.goal_id, tier="L2", trigger_kind="CONTRADICTION", urgency="IMMEDIATE", now=t0 + 20.0)
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "conjectra.decision_engine",
                "--req-id",
                r_cli["req_id"],
                "--arr-home",
                str(arr_home),
                "--dry-run-llm",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        cli_out = json.loads(proc.stdout)
        assert cli_out["status"] == "RESOLVED"

        # Boundary / error checks
        with pytest.raises(KeyError):
            engine.enqueue_decision("no_such_goal")
        with pytest.raises(ValueError):
            engine.enqueue_decision(g2.goal_id, urgency="BOGUS")
        with pytest.raises(ValueError):
            engine.enqueue_decision(g2.goal_id, tier="L9")
        with pytest.raises(KeyError):
            engine.process_request("no_such_req")
    finally:
        db.close()


def test_explore_valid_negative_lineage_and_multipool_fallback(tmp_path: Path) -> None:
    db = StateDB(tmp_path / "arr_home")
    try:
        goal = _make_goal("g_exp", primary_metric="loss_delta", direction="minimize", baseline_value=0.05)
        db.save_goal(goal)

        h1 = _make_hyp(goal, "h_parent", generation=1, writable_paths=["src/a.py"])
        db.register_hypothesis(h1)
        h1_child = _make_hyp(goal, "h_child", parent_id=h1.hypothesis_id, generation=2, writable_paths=["src/b.py"])
        db.register_hypothesis(h1_child)

        # VALID_NEGATIVE receipt records into graveyard + goal_knowledge and prunes child
        vn_rcpt = _make_receipt(
            h1,
            receipt_id="rcpt_vn",
            scientific_verdict=ScientificVerdict.VALID_NEGATIVE,
            metrics={"loss_delta": {"value": 0.08, "denominator": 20, "stage_rung": "F1_FAST_SCREEN"}},
            failure_reason="Eviction threshold 0.08 worse than baseline 0.05",
        )
        dec = reconcile_receipt(db, goal, h1, vn_rcpt)
        assert dec.scientific_verdict == ScientificVerdict.VALID_NEGATIVE
        assert dec.new_status == "VALID_NEGATIVE"
        assert dec.action == "RECORD_VALID_NEGATIVE_AND_PRUNE"
        assert dec.graveyard_recorded is True
        assert h1_child.hypothesis_id in dec.pruned_hypothesis_ids

        know = db.list_knowledge_items(goal_id=goal.goal_id, category="valid_negative")
        assert len(know) == 1
        assert know[0]["content"]["measured_value"] == pytest.approx(0.08)
        assert "worse than baseline" in know[0]["content"]["root_cause"]

        # Prompt includes multi-generation lineage, measured primary_metric vs baseline, and VALID_NEGATIVE lessons
        prompt = _build_ai_propose_prompt(
            goal,
            db.list_hypotheses(goal_id=goal.goal_id),
            set(),
            3,
            receipts=[vn_rcpt],
            graveyard_entries=db.list_graveyard(goal_id=goal.goal_id),
            knowledge_items=know,
        )
        assert "generation" in prompt
        assert "Target evolution generation: 3" in prompt
        assert "0.08" in prompt
        assert "worse than baseline" in prompt

        # Multi-pool fallback in run_one_shot_ai_proposer: first pool fails rc=1, second succeeds
        calls: list[list[str]] = []

        def _fallback_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(cmd)
            if len(calls) == 1:
                return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="429 quota exceeded")
            payload = [
                {
                    "kind": "COMPETING_ROUTE",
                    "parent_id": h1.hypothesis_id,
                    "title": "Gen 3 adaptive route",
                    "statement": "Use adaptive margin after gen1 negative",
                    "mechanism": "adaptive_margin_gen3",
                    "writable_paths": ["src/c.py"],
                    "params": {"generation": 3},
                    "kill_criteria": ["tests_passed == 0"],
                    "screen_cmd": "pytest -q",
                    "acceptance_cmd": "pytest -q",
                }
            ]
            return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps(payload), stderr="")

        batch = propose_hypotheses(
            goal,
            db=db,
            mode="ai",
            ai_provider_spec="deepseek",
            subprocess_runner=_fallback_runner,
        )
        assert len(calls) == 2
        assert calls[0][2] == "deepseek"
        assert calls[1][2] != "deepseek"
        assert len(batch.accepted) == 1
        assert batch.accepted[0].params.get("generation") == 3
    finally:
        db.close()


def test_workflow_selector_dynamic_switching_and_nested_xteamwork(tmp_path: Path) -> None:
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    try:
        goal = _make_goal("g_wf")
        db.save_goal(goal)
        selector = WorkflowSelector(max_coordination_tax_pct=35.0)
        runtime = WorkflowRuntime(db, selector=selector)

        # 1. Switch direct_hq -> x-boost on audit fail / defect escape
        h_esc = _make_hyp(goal, "h_esc", params={"workflow_primitive": "direct_hq"})
        db.register_hypothesis(h_esc)
        fail_audit = _make_receipt(
            h_esc,
            receipt_id="r_esc",
            stage=StageKind.BLIND_AUDIT,
            audit_verdict=AuditVerdict.FAIL,
            failure_reason="defect_escape on boundary input",
        )
        sel_boost = selector.select_primitive(goal, h_esc, db=db, prior_receipts=[fail_audit])
        assert sel_boost.primitive == "x-boost"
        assert sel_boost.switched_from == "direct_hq"

        # 2. Switch direct_hq -> x-swarm on interface/merge mismatch
        h_merge = _make_hyp(goal, "h_merge", params={"workflow_primitive": "direct_hq"})
        db.register_hypothesis(h_merge)
        fail_merge = _make_receipt(
            h_merge,
            receipt_id="r_merge",
            stage=StageKind.CPU_GATE,
            engineering_verdict=EngineeringVerdict.FAIL_GATE,
            failure_reason="interface_mismatch between producer and consumer",
        )
        sel_swarm = selector.select_primitive(goal, h_merge, db=db, prior_receipts=[fail_merge])
        assert sel_swarm.primitive == "x-swarm"

        # 3. Downsize x-boost -> direct_hq when workload_eval_ledger shows excessive coordination tax on low-risk task
        usage_p = tmp_path / "u_tax.json"
        usage_p.write_text(json.dumps({"model": "claude:claude-sonnet-5-5", "usage": {"input_tokens": 1000, "output_tokens": 500}}), encoding="utf-8")
        record_workload_eval(
            db,
            workload_class="disjoint_parallel",
            mode="x-boost",
            provider_pairing="claude:claude-sonnet-5-5 <-> codex:gpt-5-codex",
            is_real_llm=True,
            hq_used=True,
            holdout_cmd="pytest -q",
            holdout_rc=0,
            holdout_passed=True,
            defect_escaped=False,
            gate_intercepted=False,
            uncached_tokens=1500,
            coordination_tax_pct=52.0,
            raw_usage_paths=[str(usage_p)],
            source="test_tax",
        )
        h_tax = _make_hyp(goal, "h_tax", params={"workflow_primitive": "x-boost"})
        db.register_hypothesis(h_tax)
        sel_down = selector.select_primitive(goal, h_tax, db=db)
        assert sel_down.primitive == "direct_hq"
        assert sel_down.switched_from == "x-boost"

        # 4. True nesting in WorkflowRuntime: x-teamwork spawns child WorkflowNodes (direct_hq & x-boost),
        #    flushes <4KB milestone anchor, then emits cross-family blind_audit -> COMPLETED!
        h_tw = _make_hyp(
            goal,
            "h_tw",
            writable_paths=["src/fe.py", "src/be.py", "src/integ.py"],
            params={
                "workflow_primitive": "x-teamwork",
                "phases": [
                    {
                        "name": "Phase 1",
                        "slices": [
                            {"name": "fe", "writable_paths": ["src/fe.py"], "primitive": "direct_hq"},
                            {"name": "be", "writable_paths": ["src/be.py"], "primitive": "x-boost"},
                        ],
                    },
                    {
                        "name": "Phase 2",
                        "slices": [
                            {"name": "integ", "writable_paths": ["src/integ.py"], "primitive": "direct_hq"},
                        ],
                    },
                ],
            },
        )
        db.register_hypothesis(h_tw)
        parent_wf, p1_tasks = runtime.create_workflow_for_hypothesis(goal, h_tw)
        assert parent_wf.primitive == "x-teamwork"
        assert len(p1_tasks) == 2

        child_wfs = db.list_workflows(goal_id=goal.goal_id, parent_workflow_id=parent_wf.workflow_id)
        assert {w.primitive for w in child_wfs} == {"direct_hq", "x-boost"}

        # Advance Phase 1 child 1 (direct_hq: CODE -> CPU_GATE -> COMPLETED)
        t_fe_code = next(t for t in p1_tasks if t.workflow_primitive == "direct_hq")
        wf_fe = db.get_workflow(t_fe_code.workflow_id)
        assert wf_fe is not None
        fe_next = runtime.advance_workflow(
            wf_fe,
            t_fe_code,
            _make_receipt(h_tw, receipt_id="rc_fe1", task_id=t_fe_code.task_id, stage=StageKind.CODE),
            goal,
        )
        assert len(fe_next) == 1 and fe_next[0].stage == StageKind.CPU_GATE
        p1_after_fe = runtime.advance_workflow(
            wf_fe,
            fe_next[0],
            _make_receipt(h_tw, receipt_id="rc_fe2", task_id=fe_next[0].task_id, stage=StageKind.CPU_GATE),
            goal,
        )
        # Still waiting for x-boost sibling in Phase 1
        assert p1_after_fe == []

        # Advance Phase 1 child 2 (x-boost: boost_build -> boost_gate -> boost_adversary_0(PASS) -> Phase 2!)
        t_be_build = next(t for t in p1_tasks if t.workflow_primitive == "x-boost")
        wf_be = db.get_workflow(t_be_build.workflow_id)
        assert wf_be is not None
        be_gate = runtime.advance_workflow(
            wf_be,
            t_be_build,
            _make_receipt(h_tw, receipt_id="rc_be1", task_id=t_be_build.task_id, stage=StageKind.CODE),
            goal,
        )
        assert len(be_gate) == 1 and be_gate[0].step_key == "boost_gate"
        be_adv = runtime.advance_workflow(
            wf_be,
            be_gate[0],
            _make_receipt(h_tw, receipt_id="rc_be2", task_id=be_gate[0].task_id, stage=StageKind.CPU_GATE),
            goal,
        )
        assert len(be_adv) == 1 and be_adv[0].step_key == "boost_adversary_0"
        p2_tasks = runtime.advance_workflow(
            wf_be,
            be_adv[0],
            _make_receipt(
                h_tw,
                receipt_id="rc_be3",
                task_id=be_adv[0].task_id,
                stage=StageKind.BLIND_AUDIT,
                audit_verdict=AuditVerdict.PASS,
            ),
            goal,
        )
        # Milestone 1 anchor flushed & Phase 2 child spawned!
        anchors = db.list_knowledge_items(goal_id=goal.goal_id, category="milestone_anchor")
        assert len(anchors) == 1
        assert anchors[0]["content"]["bytes"] <= 4096
        assert len(p2_tasks) == 1

        # Complete Phase 2 child -> parent emits teamwork_cpu_gate -> 2-slot teamwork_impl_review (RED_TEAM_TIEBREAKER)
        # -> teamwork_exp_screen -> teamwork_exp_confirm -> 2-slot teamwork_result_review -> COMPLETED
        t_p2_code = p2_tasks[0]
        wf_p2 = db.get_workflow(t_p2_code.workflow_id)
        assert wf_p2 is not None
        t_p2_gate = runtime.advance_workflow(
            wf_p2,
            t_p2_code,
            _make_receipt(h_tw, receipt_id="rc_p2_1", task_id=t_p2_code.task_id, stage=StageKind.CODE),
            goal,
        )[0]
        tw_gate_tasks = runtime.advance_workflow(
            wf_p2,
            t_p2_gate,
            _make_receipt(h_tw, receipt_id="rc_p2_2", task_id=t_p2_gate.task_id, stage=StageKind.CPU_GATE),
            goal,
        )
        assert len(tw_gate_tasks) == 1 and tw_gate_tasks[0].step_key == "teamwork_cpu_gate"
        tw_r2_tasks = runtime.advance_workflow(
            parent_wf,
            tw_gate_tasks[0],
            _make_receipt(h_tw, receipt_id="rc_tw_gate", task_id=tw_gate_tasks[0].task_id, stage=StageKind.CPU_GATE),
            goal,
        )
        assert len(tw_r2_tasks) == 2
        assert runtime.advance_workflow(
            parent_wf,
            tw_r2_tasks[0],
            _make_receipt(
                h_tw,
                receipt_id="rc_tw_r2_0",
                task_id=tw_r2_tasks[0].task_id,
                stage=StageKind.IMPL_REVIEW,
                audit_verdict=AuditVerdict.PASS,
            ),
            goal,
        ) == []
        screen_tasks = runtime.advance_workflow(
            parent_wf,
            tw_r2_tasks[1],
            _make_receipt(
                h_tw,
                receipt_id="rc_tw_r2_1",
                task_id=tw_r2_tasks[1].task_id,
                stage=StageKind.IMPL_REVIEW,
                audit_verdict=AuditVerdict.PASS,
            ),
            goal,
        )
        assert len(screen_tasks) == 1
        assert screen_tasks[0].step_key == "teamwork_exp_screen"
        confirm_tasks = runtime.advance_workflow(
            parent_wf,
            screen_tasks[0],
            _make_receipt(
                h_tw,
                receipt_id="rc_tw_scr",
                task_id=screen_tasks[0].task_id,
                stage=StageKind.EXP_SCREEN,
                scientific_verdict=ScientificVerdict.CONFIRMED,
            ),
            goal,
        )
        assert len(confirm_tasks) == 1
        assert confirm_tasks[0].step_key == "teamwork_exp_confirm"
        audit_tasks = runtime.advance_workflow(
            parent_wf,
            confirm_tasks[0],
            _make_receipt(
                h_tw,
                receipt_id="rc_tw_cnf",
                task_id=confirm_tasks[0].task_id,
                stage=StageKind.EXP_CONFIRM,
                scientific_verdict=ScientificVerdict.CONFIRMED,
            ),
            goal,
        )
        assert len(audit_tasks) == 2
        assert runtime.advance_workflow(
            parent_wf,
            audit_tasks[0],
            _make_receipt(
                h_tw,
                receipt_id="rc_tw_aud0",
                task_id=audit_tasks[0].task_id,
                stage=StageKind.RESULT_REVIEW,
                audit_verdict=AuditVerdict.PASS,
            ),
            goal,
        ) == []
        final_emitted = runtime.advance_workflow(
            parent_wf,
            audit_tasks[1],
            _make_receipt(
                h_tw,
                receipt_id="rc_tw_aud1",
                task_id=audit_tasks[1].task_id,
                stage=StageKind.RESULT_REVIEW,
                audit_verdict=AuditVerdict.PASS,
            ),
            goal,
        )
        assert final_emitted == []
        final_parent = db.get_workflow(parent_wf.workflow_id)
        assert final_parent is not None and final_parent.status == "COMPLETED"

        # x-swarm: workers -> swarm_cpu_gate -> swarm_impl_review -> swarm_exp_screen -> swarm_exp_confirm -> swarm_result_review -> COMPLETED
        h_sw = _make_hyp(
            goal,
            "h_sw",
            writable_paths=["src/enc.py", "src/dec.py"],
            params={"workflow_primitive": "x-swarm"},
        )
        db.register_hypothesis(h_sw)
        sw_wf, sw_tasks = runtime.create_workflow_for_hypothesis(goal, h_sw)
        assert len(sw_tasks) == 2
        assert runtime.advance_workflow(
            sw_wf,
            sw_tasks[0],
            _make_receipt(h_sw, receipt_id="rc_sw_w1", task_id=sw_tasks[0].task_id, stage=StageKind.CODE),
            goal,
        ) == []
        sw_gate = runtime.advance_workflow(
            sw_wf,
            sw_tasks[1],
            _make_receipt(h_sw, receipt_id="rc_sw_w2", task_id=sw_tasks[1].task_id, stage=StageKind.CODE),
            goal,
        )
        assert len(sw_gate) == 1 and sw_gate[0].step_key == "swarm_cpu_gate"
        sw_r2 = runtime.advance_workflow(
            sw_wf,
            sw_gate[0],
            _make_receipt(h_sw, receipt_id="rc_sw_gate", task_id=sw_gate[0].task_id, stage=StageKind.CPU_GATE),
            goal,
        )
        assert len(sw_r2) == 1 and sw_r2[0].step_key == "swarm_impl_review_t1"
        sw_screen = runtime.advance_workflow(
            sw_wf,
            sw_r2[0],
            _make_receipt(
                h_sw,
                receipt_id="rc_sw_r2",
                task_id=sw_r2[0].task_id,
                stage=StageKind.IMPL_REVIEW,
                audit_verdict=AuditVerdict.PASS,
            ),
            goal,
        )
        assert len(sw_screen) == 1 and sw_screen[0].step_key == "swarm_exp_screen"
        sw_confirm = runtime.advance_workflow(
            sw_wf,
            sw_screen[0],
            _make_receipt(
                h_sw,
                receipt_id="rc_sw_scr",
                task_id=sw_screen[0].task_id,
                stage=StageKind.EXP_SCREEN,
                scientific_verdict=ScientificVerdict.CONFIRMED,
            ),
            goal,
        )
        assert len(sw_confirm) == 1 and sw_confirm[0].step_key == "swarm_exp_confirm"
        sw_verify = runtime.advance_workflow(
            sw_wf,
            sw_confirm[0],
            _make_receipt(
                h_sw,
                receipt_id="rc_sw_cnf",
                task_id=sw_confirm[0].task_id,
                stage=StageKind.EXP_CONFIRM,
                scientific_verdict=ScientificVerdict.CONFIRMED,
            ),
            goal,
        )
        assert len(sw_verify) == 1 and sw_verify[0].step_key == "swarm_result_review_t1"
        assert runtime.advance_workflow(
            sw_wf,
            sw_verify[0],
            _make_receipt(
                h_sw,
                receipt_id="rc_sw_aud",
                task_id=sw_verify[0].task_id,
                stage=StageKind.RESULT_REVIEW,
                audit_verdict=AuditVerdict.PASS,
            ),
            goal,
        ) == []
        sw_final = db.get_workflow(sw_wf.workflow_id)
        assert sw_final is not None and sw_final.status == "COMPLETED"

        h_sw_neg = _make_hyp(
            goal,
            "h_sw_neg",
            writable_paths=["src/enc2.py"],
            params={"workflow_primitive": "x-swarm"},
        )
        db.register_hypothesis(h_sw_neg)
        sw_neg_wf, sw_neg_tasks = runtime.create_workflow_for_hypothesis(goal, h_sw_neg)
        sw_neg_gate = runtime.advance_workflow(
            sw_neg_wf,
            sw_neg_tasks[0],
            _make_receipt(h_sw_neg, receipt_id="rc_swn_1", task_id=sw_neg_tasks[0].task_id, stage=StageKind.CODE),
            goal,
        )[0]
        sw_neg_r2 = runtime.advance_workflow(
            sw_neg_wf,
            sw_neg_gate,
            _make_receipt(h_sw_neg, receipt_id="rc_swn_g", task_id=sw_neg_gate.task_id, stage=StageKind.CPU_GATE),
            goal,
        )[0]
        sw_neg_scr = runtime.advance_workflow(
            sw_neg_wf,
            sw_neg_r2,
            _make_receipt(
                h_sw_neg,
                receipt_id="rc_swn_r2",
                task_id=sw_neg_r2.task_id,
                stage=StageKind.IMPL_REVIEW,
                audit_verdict=AuditVerdict.PASS,
            ),
            goal,
        )[0]
        sw_neg_cnf = runtime.advance_workflow(
            sw_neg_wf,
            sw_neg_scr,
            _make_receipt(
                h_sw_neg,
                receipt_id="rc_swn_2",
                task_id=sw_neg_scr.task_id,
                stage=StageKind.EXP_SCREEN,
                scientific_verdict=ScientificVerdict.CONFIRMED,
            ),
            goal,
        )[0]
        assert runtime.advance_workflow(
            sw_neg_wf,
            sw_neg_cnf,
            _make_receipt(
                h_sw_neg,
                receipt_id="rc_swn_3",
                task_id=sw_neg_cnf.task_id,
                stage=StageKind.EXP_CONFIRM,
                scientific_verdict=ScientificVerdict.VALID_NEGATIVE,
            ),
            goal,
        ) == []
        sw_neg_final = db.get_workflow(sw_neg_wf.workflow_id)
        assert sw_neg_final is not None and sw_neg_final.status == "COMPLETED_VALID_NEGATIVE"

        # Stale goal_version marks workflow STALE
        goal_v2 = _make_goal("g_wf", version=2)
        h_ro = _make_hyp(goal, "h_ro", params={"readonly": True})
        db.register_hypothesis(h_ro)
        wf_ro, ro_tasks = runtime.create_workflow_for_hypothesis(goal, h_ro)
        assert wf_ro.primitive == "readonly_fanout"
        assert runtime.advance_workflow(wf_ro, ro_tasks[0], _make_receipt(h_ro), goal_v2) == []
        assert db.get_workflow(wf_ro.workflow_id).status == "STALE"
    finally:
        db.close()


def test_xboost_repair_not_prematurely_refuted_and_zero_receipt_guard(tmp_path: Path) -> None:
    from conjectra.controller import ResearchController
    from conjectra.metrics import _is_real_evidence_receipt

    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    try:
        goal = _make_goal("g_xboost_ctrl")
        db.save_goal(goal)
        hyp = _make_hyp(goal, "h_xb_live")
        db.register_hypothesis(hyp)
        rt = WorkflowRuntime(db)
        wf, init_tasks = rt.create_workflow_for_hypothesis(
            goal,
            hyp,
            primitive="x-boost",
            config_override={"builder_spec": "claude:claude-sonnet-5-5", "adversary_spec": "codex:gpt-5-codex"},
        )
        t_code = init_tasks[0]
        db.transition_task_status(t_code.task_id, "READY", "DONE", finished_ts=time.time())
        t_gate = rt.advance_workflow(
            wf,
            t_code,
            _make_receipt(hyp, receipt_id="r_code", task_id=t_code.task_id, stage=StageKind.CODE),
            goal,
        )[0]
        db.transition_task_status(t_gate.task_id, "READY", "DONE", finished_ts=time.time())
        t_adv = rt.advance_workflow(
            wf,
            t_gate,
            _make_receipt(hyp, receipt_id="r_gate", task_id=t_gate.task_id, stage=StageKind.CPU_GATE),
            goal,
        )[0]
        db.transition_task_status(t_adv.task_id, "READY", "DONE", finished_ts=time.time())
        adv_fail = _make_receipt(
            hyp,
            receipt_id="r_adv_fail",
            task_id=t_adv.task_id,
            stage=StageKind.BLIND_AUDIT,
            scientific_verdict=ScientificVerdict.NOT_APPLICABLE,
            audit_verdict=AuditVerdict.FAIL,
            failure_reason="Independent blind audit returned FAIL",
        )
        db.record_receipt(adv_fail)

        arbiter = ResourceArbiter(disk_mounts=(tmp_path,), min_ram_floor_mb=256, max_load1=128.0)
        ctrl = ResearchController(arr_home, db=db, arbiter=arbiter, execution_mode="inline", propose_mode="ai", dry_run_llm=False)
        summary = {
            "ts": time.time(),
            "leases_reaped": 0,
            "recovered_tasks": 0,
            "reconciled": 0,
            "proposed": 0,
            "tasks_enqueued": 0,
            "tasks_dispatched": 0,
            "pruned": 0,
        }
        ctrl._handle_completed_task(t_adv, goal, summary)
        reloaded_hyp = db.get_hypothesis(hyp.hypothesis_id)
        reloaded_wf = db.get_workflow(wf.workflow_id)
        repair_tasks = [t for t in db.list_tasks(goal_id=goal.goal_id) if t.step_key == "boost_repair_1"]
        assert reloaded_hyp is not None and reloaded_hyp.status == "PENDING"
        assert reloaded_wf is not None and reloaded_wf.status == "RUNNING"
        assert len(repair_tasks) == 1 and repair_tasks[0].status == "READY"

        # Zero-receipt guard: pre-seeded Goal with 0 receipts must NOT enqueue L1 DecisionRequest
        g0 = _make_goal("g_zero_rcpt")
        db.save_goal(g0)
        h0 = _make_hyp(g0, "h_seed0")
        db.register_hypothesis(h0)
        ctrl._maybe_propose_round(g0, summary)
        assert db.list_decision_requests(goal_id=g0.goal_id) == []

        # Custom real evidence origins must be treated as real_receipts
        real_r = EvidenceReceipt.from_dict({**_make_receipt(hyp).to_dict(), "evidence_origin": "projgamma_real_goal"})
        synth_r = EvidenceReceipt.from_dict({**_make_receipt(hyp).to_dict(), "evidence_origin": "synthetic_sim"})
        assert _is_real_evidence_receipt(real_r) is True
        assert _is_real_evidence_receipt(synth_r) is False
    finally:
        db.close()


def test_decision_engine_marks_ai_falsification_requires_llm_code(tmp_path: Path) -> None:
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    try:
        goal = _make_goal("g_ai_req_llm")
        db.save_goal(goal)
        parent = _make_hyp(goal, "h_seed")
        db.register_hypothesis(parent)
        engine = DecisionEngine(db)
        req = engine.enqueue_decision(goal.goal_id, tier="L1", urgency="IMMEDIATE")

        def _fake_llm(_prov: Any, _prompt: str) -> dict[str, Any]:
            return {
                "exit_code": 0,
                "in_tokens": 100,
                "out_tokens": 50,
                "model_used": "gemini:gemini-pro",
                "response": json.dumps(
                    [
                        {
                            "kind": "FALSIFICATION",
                            "parent_id": parent.hypothesis_id,
                            "title": "Falsify boundary edge case",
                            "statement": "Write new stress test and falsify boundary bug.",
                            "mechanism": "boundary_stress_test",
                            "writable_paths": ["tests/test_boundary.py"],
                            "params": {"seed": 1},
                            "kill_criteria": ["tests_passed == 0"],
                            "acceptance_cmd": "pytest -q tests/test_boundary.py",
                            "screen_cmd": "pytest -q tests/test_boundary.py",
                            "confirm_cmd": "pytest -q tests/test_boundary.py",
                        }
                    ]
                ),
            }

        res = engine.process_request(req["req_id"], llm_caller=_fake_llm, dry_run_llm=False)
        assert res["status"] == "RESOLVED"
        new_hids = res["resolution"]["accepted_hypothesis_ids"]
        assert len(new_hids) == 1
        new_hyp = db.get_hypothesis(new_hids[0])
        assert new_hyp is not None and new_hyp.params.get("requires_llm_code") is True
        rt = WorkflowRuntime(db)
        _, tasks = rt.create_workflow_for_hypothesis(goal, new_hyp)
        assert tasks[0].resources.llm_pool is not None
    finally:
        db.close()


def test_stage4_default_direct_hq_and_collapse_overlapping_specs(tmp_path: Path) -> None:
    """Priority 6: default to linear direct_hq and auto-collapse overlapping x-swarm/x-teamwork specs."""
    arr_home = tmp_path / "arr_home"
    db = StateDB(arr_home)
    try:
        goal = _make_goal("g_collapse")
        db.save_goal(goal)
        selector = WorkflowSelector()
        rt = WorkflowRuntime(db, selector=selector)

        # 1. Parent BLIND_AUDIT failure must NOT cascade new child hypothesis into x-boost; default is direct_hq
        parent = _make_hyp(goal, "h_parent")
        db.register_hypothesis(parent)
        db.record_receipt(
            _make_receipt(
                parent,
                receipt_id="rc_parent_audit",
                stage=StageKind.BLIND_AUDIT,
                audit_verdict=AuditVerdict.FAIL,
                failure_reason="defect_escape on parent",
            )
        )
        child = HypothesisNode(
            hypothesis_id="h_child_explore",
            goal_id=goal.goal_id,
            parent_id=parent.hypothesis_id,
            kind=HypothesisKind.COMPETING_ROUTE,
            title="Parallel refactor across subsystem",
            statement="Explore parallel kernel cleanup",
            mechanism="linear_direct_test",
            fidelity=FidelityRung.F1_FAST_SCREEN,
            priority_evoi=0.7,
            writable_paths=["a.py", "b.py"],
            params={},
            kill_criteria=["tests_passed == 0"],
            acceptance_cmd="pytest -q",
            screen_cmd="pytest -q",
            confirm_cmd="pytest -q",
            branch_name="arr/h_child_explore",
            base_sha="deadbeef",
        )
        db.register_hypothesis(child)
        sel = selector.select_primitive(goal, child, db=db)
        assert sel.primitive == "direct_hq"

        # 2. Overlapping x-swarm workers auto-collapse to sequential direct_hq + WORKFLOW_COLLAPSED_TO_DIRECT_HQ
        h_ov_sw = _make_hyp(
            goal,
            "h_ov_sw",
            writable_paths=["src/demo", "src/demo/test_vcm.py"],
            params={
                "workflow_primitive": "x-swarm",
                "workers": [
                    {"name": "w1", "writable_paths": ["src/demo"]},
                    {"name": "w2", "writable_paths": ["src/demo/test_vcm.py"]},
                ],
            },
        )
        db.register_hypothesis(h_ov_sw)
        wf_sw, tasks_sw = rt.create_workflow_for_hypothesis(goal, h_ov_sw)
        assert wf_sw.primitive == "direct_hq"
        assert wf_sw.config.get("collapsed_from") == "x-swarm"
        assert len(tasks_sw) == 1 and tasks_sw[0].stage == StageKind.CODE

        # 3. Overlapping x-teamwork phases AND invalid slice edges auto-collapse on switch_hypothesis_workflow
        h_ov_tw = _make_hyp(
            goal,
            "h_ov_tw",
            writable_paths=["src/mod.py"],
            params={"workflow_primitive": "direct_hq"},
        )
        db.register_hypothesis(h_ov_tw)
        wf_tw, tasks_tw = rt.switch_hypothesis_workflow(
            goal,
            h_ov_tw,
            new_primitive="x-teamwork",
            reason="test_collapse",
            config_override={
                "phases": [
                    {
                        "name": "p1",
                        "slices": [
                            {"name": "s1", "writable_paths": ["src/mod.py"], "depends_on": ["missing_edge"]},
                        ],
                    }
                ]
            },
        )
        assert wf_tw.primitive == "direct_hq"
        assert wf_tw.config.get("collapsed_from") == "x-teamwork"
        assert len(tasks_tw) == 1 and tasks_tw[0].stage == StageKind.CODE

        with db._tx() as (conn, _):
            evs = conn.execute(
                "SELECT hypothesis_id, payload_json FROM events WHERE event_type = 'WORKFLOW_COLLAPSED_TO_DIRECT_HQ'"
            ).fetchall()
        assert {r["hypothesis_id"] for r in evs} == {"h_ov_sw", "h_ov_tw"}
    finally:
        db.close()
