# Custom adapter example

`conjectra_csv_adapter.CsvMetricsAdapter` is a project adapter for repositories whose experiments
write a `results.csv` (`metric,value,n`). It seeds two hypotheses, substitutes the per-task
artifact directory into stage commands, and turns the CSV into metrics — reusing Conjectra's shared
verdict logic so baseline comparison, kill criteria and missing-metric handling stay consistent.

Register it (see [docs/adapters.md](../../docs/adapters.md)):

```bash
pip install -e examples/custom_adapter        # entry point "csv_metrics"
# or, without installing:
export PYTHONPATH=examples/custom_adapter CONJECTRA_ADAPTERS="csv_metrics=conjectra_csv_adapter:CsvMetricsAdapter"
```

and set `"adapter_name": "csv_metrics"` in your goal JSON.
