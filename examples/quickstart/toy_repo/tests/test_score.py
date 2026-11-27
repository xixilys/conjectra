from src.score import score


def test_score_reports_metric():
    val = score()
    assert 0.0 < val <= 1.0
    # The generic adapter parses `METRIC <name>=<value> denom=<n>` lines from stdout.
    print(f"METRIC score={val} denom=1")
