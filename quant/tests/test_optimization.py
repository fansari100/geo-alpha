"""Sensor tasking optimization."""

from __future__ import annotations

from geoalpha_quant.optimization import (
    SensorTaskingProblem,
    TargetRequest,
    solve_sensor_tasking,
)


def test_greedy_respects_budget():
    targets = [
        TargetRequest("alpha", value=10.0, dwell_max=30, priority="ROUTINE"),
        TargetRequest("bravo", value=8.0, dwell_max=30, priority="PRIORITY"),
        TargetRequest("charlie", value=5.0, dwell_max=30, priority="ROUTINE"),
        TargetRequest("flash", value=4.0, dwell_max=30, priority="FLASH"),
    ]
    problem = SensorTaskingProblem(targets=targets, total_budget_s=60.0)
    result = solve_sensor_tasking(problem)
    assert result.dwell_seconds.sum() <= 60.0 + 1e-6
    # Highest-priority should always be allocated when it fits.
    assignment = result.as_assignment(problem)
    assert "flash" in assignment


def test_zero_targets_returns_empty():
    problem = SensorTaskingProblem(targets=[], total_budget_s=60.0)
    result = solve_sensor_tasking(problem)
    assert result.dwell_seconds.size == 0
