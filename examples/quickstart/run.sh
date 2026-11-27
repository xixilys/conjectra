#!/usr/bin/env bash
# Five-minute local demo: no HyperQueue, no API keys, no network.
#
#   ./examples/quickstart/run.sh            # uses ./.conjectra-demo as scratch space
#   KEEP=1 ./examples/quickstart/run.sh     # keep the scratch dir afterwards
#
# What it does:
#   1. creates a toy git repo whose test prints `METRIC score=<value>`
#   2. registers a research goal ("maximize score") against it
#   3. runs the zero-token controller inline; model calls go to the offline `fake` backend
#   4. shows the ledger: overview, per-goal inspect, a <3KB takeover brief
#   5. applies a human intervention (pause) and shows it took effect
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
WORK="${WORK:-$PWD/.conjectra-demo}"
CJ="python3 $ROOT/bin/conjectra --home $WORK/home"
command -v conjectra >/dev/null 2>&1 && CJ="conjectra --home $WORK/home"

rm -rf "$WORK"
mkdir -p "$WORK"
export CONJECTRA_SANDBOX_ROOT="$WORK/sandboxes"
export CONJECTRA_WORKTREE_ROOT="$WORK/worktrees"
# The default host guard waits for 10 GiB of free RAM before test/experiment stages; the toy
# repo needs almost nothing, so relax it for laptops and small CI runners.
export CONJECTRA_MIN_AVAIL_RAM_MB="${CONJECTRA_MIN_AVAIL_RAM_MB:-512}"
# Likewise the disk guard refuses new stages once `/` is 85 % full; the demo writes a few MB.
export CONJECTRA_DISK_MAX_PCT="${CONJECTRA_DISK_MAX_PCT:-99}"
# Every model call goes to a tiny offline plugin backend (examples/quickstart/toy_backend.py):
# write calls bump the Newton iteration count, review calls return a structured PASS.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
export CONJECTRA_BACKENDS="toy=toy_backend:ToyEditBackend"
export CONJECTRA_BACKEND_OVERRIDE="${CONJECTRA_BACKEND_OVERRIDE:-toy}"

echo "== 1. toy repo"
REPO="$WORK/toy-repo"
mkdir -p "$REPO/src" "$REPO/tests"
cp "$HERE/toy_repo/src/score.py" "$REPO/src/score.py"
cp "$HERE/toy_repo/tests/test_score.py" "$REPO/tests/test_score.py"
touch "$REPO/src/__init__.py" "$REPO/tests/__init__.py"
printf '[tool.pytest.ini_options]\npythonpath = ["."]\naddopts = ["-s"]\n' > "$REPO/pyproject.toml"
git -C "$REPO" init -q
git -C "$REPO" -c user.email=demo@example.invalid -c user.name=demo add .
git -C "$REPO" -c user.email=demo@example.invalid -c user.name=demo commit -qm "baseline"
BASE_SHA="$(git -C "$REPO" rev-parse HEAD)"

echo "== 2. init ledger + submit goal"
$CJ init-pilot --no-ssh >/dev/null
sed -e "s#__REPO__#$REPO#" -e "s#__SHA__#$BASE_SHA#" "$HERE/goal.json" > "$WORK/goal.json"
$CJ submit-goal --file "$WORK/goal.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("submitted goal:", (d.get("goal") or d).get("goal_id"))'
$CJ import-proposals --file "$HERE/proposals.json" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("accepted proposals:", len(d["accepted"]), "| enqueued tasks:", len(d["enqueued_task_ids"]))'

echo "== 3. run the controller inline for a few seconds"
$CJ daemon --mode inline --goal-id demo-score --propose-mode seeds --sweep-interval 0.2 --max-seconds "${SECONDS_BUDGET:-20}" >/dev/null

echo "== 4. what happened"
$CJ overview | python3 "$HERE/summarize.py" overview
$CJ inspect --goal-id demo-score | python3 "$HERE/summarize.py" inspect
$CJ takeover --goal-id demo-score --format markdown > "$WORK/takeover.md"
echo "takeover brief: $(wc -c < "$WORK/takeover.md") bytes -> $WORK/takeover.md"

echo "== 5. human intervention: pause the goal"
$CJ intervene --scope goal --goal-id demo-score --action pause --reason "demo: owner review" >/dev/null
$CJ overview | python3 "$HERE/summarize.py" status

[ "${KEEP:-0}" = "1" ] || rm -rf "$WORK"
echo "done."
