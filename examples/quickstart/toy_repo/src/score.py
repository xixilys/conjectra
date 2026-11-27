"""Toy 'research target': approximate sqrt(2) with a few Newton steps."""


def approx_sqrt2(iterations: int = 2) -> float:
    x = 1.0
    for _ in range(iterations):
        x = 0.5 * (x + 2.0 / x)
    return x


def score(iterations: int = 2) -> float:
    """Higher is better: negative absolute error, scaled to [0, 1]."""
    err = abs(approx_sqrt2(iterations) - 2.0**0.5)
    return round(1.0 / (1.0 + 1e3 * err), 6)
