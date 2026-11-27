# Contributing to Conjectra

Thanks for helping. Conjectra is small enough that one person can understand the whole control
plane, and we want to keep it that way: deterministic, inspectable, honest about evidence.

## Development setup

```bash
git clone https://github.com/xixilys/conjectra && cd conjectra
python3 -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
# HyperQueue is needed for the tests that start a real hq server/worker:
#   https://github.com/It4innovations/hyperqueue/releases  (put `hq` on PATH)
git config --global user.email >/dev/null || git config --global user.email "you@example.com"
```

Run the checks CI runs:

```bash
ruff check .
pytest -q
./examples/quickstart/run.sh
```

On machines with less than ~10 GiB free RAM set `CONJECTRA_MIN_AVAIL_RAM_MB=1024` (the default
host guard waits for 10 GiB before test/experiment stages).

## Ground rules

* **Never weaken a test to make it pass.** Do not delete or loosen assertions, add `skip`/`xfail`,
  or special-case a benchmark. If a test is wrong, explain why in the PR and fix the test
  deliberately.
* **The controller stays zero-token.** Anything that calls a model must be a job (a stage or a
  decision request) with a pinned route — never an inline call inside `ResearchController.step()`.
* **Failure domains matter.** New failure modes must be classified (`INFRA_TRANSIENT`,
  `ENV_CONFIG_MISMATCH`, `ENGINEERING_DEFECT`, `SCIENTIFIC_OUTCOME`). Only valid empirical evidence
  may refute a hypothesis.
* **Schema changes are additive** and come with a migration in `state_db.py` plus a test.
* **No secrets, personal paths or private endpoints** in code, tests or fixtures. CI-style scans:
  `gitleaks detect --no-git -s .`

## Adding a backend or adapter

Prefer a plugin package (entry points `conjectra.backends` / `conjectra.adapters`) over changes to
core; see [docs/backends.md](docs/backends.md), [docs/adapters.md](docs/adapters.md) and the
`examples/` folder. Built-in backends are welcome for widely used public CLIs or APIs.

## Pull requests

* Keep PRs focused; describe the behaviour change and how you tested it (paste the `pytest`
  summary line).
* Update `CHANGELOG.md` under *Unreleased* for user-visible changes, and regenerate
  `docs/cli.md` with `python3 scripts/gen_cli_docs.py > docs/cli.md` if you touched the CLI.
* By contributing you agree that your contribution is licensed under the Apache License 2.0.
