"""Pretty-print a few fields from `conjectra overview` / `conjectra inspect` JSON (stdin)."""

from __future__ import annotations

import collections
import json
import sys

mode = sys.argv[1] if len(sys.argv) > 1 else "overview"
doc = json.load(sys.stdin)

if mode in ("overview", "status"):
    for g in doc.get("goals", []):
        print(
            f"goal {g['goal_id']}: status={g['status']} receipts={g['receipts_count']} "
            f"tokens={g['tokens_used']}/{g['max_tokens_budget']} human_epoch={g['human_epoch']}"
        )
    print(f"overview latency: {doc.get('interactive_latency_ms')} ms")
elif mode == "inspect":
    hyps = doc.get("hypotheses", [])
    verdicts = collections.Counter(h.get("scientific_verdict") for h in hyps)
    print(f"{len(hyps)} hypotheses; verdicts: {dict(verdicts)}")
    for h in hyps[:6]:
        print(f"  [{h['scientific_verdict']:>14}] {h['kind']:<16} {h['title']}")
    stages = collections.Counter(r.get("stage") for r in doc.get("recent_receipts", []))
    print(f"recent evidence receipts by stage: {dict(stages)}")
