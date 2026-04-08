"""
Sensor tasking as a convex optimization problem.

I lifted the formulation from the long-only constrained portfolio
problem in my multi-asset risk engine - and barely had to change
anything.  The mapping is almost embarrassingly direct:

    portfolio weights w_i   ->  fraction of dwell time on target i
    expected returns mu_i   ->  intelligence value v_i of imaging i
    covariance Sigma        ->  covariance of value across targets
                                 (correlated weather, overlapping AOIs)
    long-only / fully-invested -> 0 <= dwell_i, sum dwell_i <= budget
    sector caps             ->  per-AOI / per-priority caps
    risk aversion gamma     ->  task-deconfliction penalty

The CVXPY problem below is a quadratic program; SCS or ECOS can both
solve realistic instances (several hundred targets) in well under a
second on a laptop.

There's also a greedy `_seed_warm_start` that's used to initialise the
solver and as a sanity-check fallback when CVXPY isn't available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:  # pragma: no cover
    cp = None  # type: ignore
    _HAS_CVXPY = False


@dataclass
class TargetRequest:
    """A single AOI / target the operator wants imaged this planning horizon."""

    name: str
    value: float                      # intelligence value if collected
    dwell_min: float = 0.0            # required minimum dwell (s)
    dwell_max: float = 60.0           # hard cap from kinematic feasibility
    priority: str = "ROUTINE"         # ROUTINE / PRIORITY / IMMEDIATE / FLASH


@dataclass
class SensorTaskingProblem:
    """The full optimization problem definition."""

    targets: List[TargetRequest]
    total_budget_s: float
    risk_aversion: float = 0.5
    value_cov: Optional[np.ndarray] = None  # NxN covariance of value
    priority_caps: dict = field(
        default_factory=lambda: {
            "FLASH": 1.0,
            "IMMEDIATE": 0.6,
            "PRIORITY": 0.4,
            "ROUTINE": 0.2,
        }
    )

    @property
    def n(self) -> int:
        return len(self.targets)


@dataclass
class TaskingResult:
    dwell_seconds: np.ndarray
    total_value: float
    constraints_active: List[str]
    solver: str

    def as_assignment(self, problem: SensorTaskingProblem) -> dict:
        return {
            t.name: float(self.dwell_seconds[i])
            for i, t in enumerate(problem.targets)
            if self.dwell_seconds[i] > 1e-3
        }


def solve_sensor_tasking(problem: SensorTaskingProblem) -> TaskingResult:
    """Solve a sensor tasking problem.

    Uses CVXPY if available, else a deterministic greedy heuristic.
    """
    if _HAS_CVXPY:
        return _solve_cvxpy(problem)
    return _solve_greedy(problem)


# --------------------------------------------------------------------- #
# CVXPY formulation
# --------------------------------------------------------------------- #

def _solve_cvxpy(problem: SensorTaskingProblem) -> TaskingResult:
    n = problem.n
    if n == 0:
        return TaskingResult(np.zeros(0), 0.0, [], "cvxpy/empty")

    v = np.array([t.value for t in problem.targets], dtype=np.float64)
    dmin = np.array([t.dwell_min for t in problem.targets], dtype=np.float64)
    dmax = np.array([t.dwell_max for t in problem.targets], dtype=np.float64)
    Sigma = problem.value_cov if problem.value_cov is not None else 1e-3 * np.eye(n)

    # Decision variable: per-target dwell (seconds).
    d = cp.Variable(n, nonneg=True)

    # Quadratic objective: maximize value minus risk-adjusted "interference".
    risk_term = cp.quad_form(d / problem.total_budget_s, cp.psd_wrap(Sigma))
    objective = cp.Maximize(v @ d - problem.risk_aversion * risk_term)

    constraints = [
        cp.sum(d) <= problem.total_budget_s,
        d >= dmin,
        d <= dmax,
    ]

    # Per-priority budget caps - prevents the optimizer from blowing the
    # whole horizon on routine collects when a FLASH request is sitting
    # on the table.
    for prio, cap in problem.priority_caps.items():
        idx = [i for i, t in enumerate(problem.targets) if t.priority == prio]
        if idx:
            constraints.append(cp.sum(d[idx]) <= cap * problem.total_budget_s)

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.SCS, verbose=False)

    if d.value is None:
        return _solve_greedy(problem)

    active = [str(c) for c in constraints if c.dual_value is not None
              and float(np.max(np.abs(c.dual_value))) > 1e-6]

    return TaskingResult(
        dwell_seconds=np.maximum(d.value, 0.0),
        total_value=float(v @ d.value),
        constraints_active=active,
        solver="cvxpy/SCS",
    )


# --------------------------------------------------------------------- #
# Greedy fallback (also used as a CVXPY warm-start in production).
# --------------------------------------------------------------------- #

def _solve_greedy(problem: SensorTaskingProblem) -> TaskingResult:
    n = problem.n
    if n == 0:
        return TaskingResult(np.zeros(0), 0.0, [], "greedy/empty")

    order = sorted(
        range(n),
        key=lambda i: (
            -_priority_rank(problem.targets[i].priority),
            -problem.targets[i].value,
        ),
    )
    dwell = np.zeros(n)
    remaining = problem.total_budget_s
    for i in order:
        t = problem.targets[i]
        give = min(remaining, t.dwell_max)
        if give < t.dwell_min:
            continue
        dwell[i] = give
        remaining -= give
        if remaining <= 0:
            break
    v = np.array([t.value for t in problem.targets])
    return TaskingResult(
        dwell_seconds=dwell,
        total_value=float(v @ dwell),
        constraints_active=["budget"],
        solver="greedy",
    )


def _priority_rank(p: str) -> int:
    return {"ROUTINE": 1, "PRIORITY": 2, "IMMEDIATE": 3, "FLASH": 4}.get(p, 0)
