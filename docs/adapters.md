# Project adapters

An adapter is the only project-specific code Conjectra needs. It answers three questions:

| Method | Question |
|---|---|
| `seed_hypotheses(goal)` | What are good, deterministic starting hypotheses for this project? |
| `build_stage_command(task, hyp, goal, worktree, artifact_dir)` | Which shell command does each stage run? |
| `parse_metrics(stage, stdout, stderr, worktree, artifact_dir, exit_code, hyp, goal)` | Which metrics did the run produce, and what verdict do they imply? |

Optional overrides: `build_code_prompt` (the prompt for `CODE` jobs) and `default_resource_spec`
(CPU / RAM / GPU / timeout per stage; the default asks for one GPU on experiment stages when
`hyp.params["required_hardware"]` is `local_gpu` / `cloud_gpu` or `hyp.params["gpu"]` is set).

## The generic adapter

`adapter_name: "generic"` works for any repository where:

* `acceptance_cmd`, `screen_cmd`, `confirm_cmd` are shell commands run in the worktree;
* metrics are printed as `METRIC <name>=<value> denom=<n>` lines (pytest summaries are also
  parsed into `tests_passed`).

## Writing your own

Subclass `GenericAdapter` (or `ProjectAdapter`) and keep verdict logic shared by calling
`evaluate_scientific_verdict(metrics, flat_values, goal, hyp, stage, exit_code)` — it implements
baseline comparison for both directions, kill criteria, missing-metric handling, and the rule that
"tests passed" alone never confirms a hypothesis.

```python
from conjectra.adapters import GenericAdapter, evaluate_scientific_verdict, metric_entry

class MyAdapter(GenericAdapter):
    name = "myproject"

    def parse_metrics(self, stage, stdout, stderr, worktree, artifact_dir, exit_code, hyp, goal):
        value = read_my_metric(artifact_dir / "summary.json")
        metrics = {"accuracy": metric_entry(value, 1000, "F1_FAST_SCREEN")}
        validity, verdict, reason = evaluate_scientific_verdict(
            metrics, {"accuracy": value}, goal, hyp, stage, exit_code
        )
        self.last_evidence_validity = validity
        return metrics, verdict, [str(artifact_dir / "summary.json")], reason
```

A complete example that reads `results.csv` lives in
[`examples/custom_adapter`](../examples/custom_adapter).

## Registering

```python
from conjectra.adapters import register_adapter
register_adapter("myproject", MyAdapter)
```

or `CONJECTRA_ADAPTERS="myproject=my_pkg.adapter:MyAdapter"`, or an entry point:

```toml
[project.entry-points."conjectra.adapters"]
myproject = "my_pkg.adapter:MyAdapter"
```

Select it with `"adapter_name": "myproject"` in the goal. Unknown names fall back to `generic`.
If the goal's `adapter_name` is `generic`/empty but its `project` matches a registered adapter,
that adapter's seeds are used.

## Rules adapters must keep

* Never write outside `worktree` / `artifact_dir`; never read `goal.banned_paths`.
* Report *missing* metrics as missing (`MISSING_PRIMARY_METRIC` / `MISSING_INPUT`), never as a
  negative result — an environment problem must not refute a hypothesis.
* Keep commands deterministic and bounded by the stage's `wall_timeout_s`.
