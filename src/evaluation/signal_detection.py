from __future__ import annotations

from math import erf, sqrt


def _approx_norm_ppf(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    low, high = -8.0, 8.0
    for _ in range(80):
        mid = (low + high) / 2
        cdf = 0.5 * (1 + erf(mid / sqrt(2)))
        if cdf < p:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def compute_d_prime(hit_rate: float, false_alarm_rate: float) -> float:
    return _approx_norm_ppf(hit_rate) - _approx_norm_ppf(false_alarm_rate)
