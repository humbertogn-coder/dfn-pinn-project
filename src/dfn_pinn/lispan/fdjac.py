"""Forward-difference Jacobian with ABSOLUTE steps for scipy.optimize.least_squares on finite-volume residuals.

scipy's '2-point' Jacobian uses relative steps (diff_step * |x|); for a parameter vector made of log-multipliers and
offsets, x is (close to) zero at the nominal values, where scipy falls back to a step of ~1.5e-8.  Such a step is
below the integration tolerance of the BDF solver, so the Jacobian columns are solver noise and the trust-region
iterations stall (seen on 2026-10-08: SPAN500 fit stuck at 290 mV, measured-data fits converging only slowly).
Here every parameter gets a fixed step (default 0.01: 1 % for log-multipliers, 1 mV for offsets stored in 0.1 V
units), flipped to a backward difference at an upper bound, and the residual at x is reused from the last call.
"""

from __future__ import annotations

import numpy as np


def least_squares_fd(resid, steps, lo=None, hi=None):
    """Return (fun, jac) callables for least_squares(fun, x0, jac=jac, ...)."""
    steps = np.asarray(steps, dtype=float)
    cache = {}

    def fun(x):
        r = resid(x)
        cache["x"], cache["r"] = np.array(x, dtype=float), r
        return r

    def jac(x, *args):
        x = np.asarray(x, dtype=float)
        r0 = cache["r"] if ("x" in cache and np.array_equal(cache["x"], x)) else resid(x)
        J = np.empty((r0.size, x.size))
        for i, h in enumerate(steps):
            if hi is not None and x[i] + h > hi[i]:
                h = -h
            xp = x.copy()
            xp[i] += h
            J[:, i] = (resid(xp) - r0) / h
        return J

    return fun, jac


class FitCheckpoint:
    """Best point of a running least-squares fit, written whenever the misfit improves, so that a fit killed by a
    machine restart continues from there (--resume) instead of from the start.  Keeps the number of model solves and
    the wall time already spent, so that cost figures cover the whole fit."""

    def __init__(self, path):
        import json
        from pathlib import Path
        self.path, self._json = Path(path), json
        self.state = json.loads(self.path.read_text()) if self.path.exists() else None
        self.best = self.state["rms"] if self.state else float("inf")

    @property
    def prior_solves(self):
        return int(self.state["solves"]) if self.state else 0

    @property
    def prior_wall_s(self):
        return float(self.state["wall_s"]) if self.state else 0.0

    def x(self):
        return np.array(self.state["x"], dtype=float) if self.state else None

    def record(self, x, rms, solves, wall_s, extra=None):
        if rms < self.best:
            self.best = rms
            self.state = {"x": [float(v) for v in x], "rms": float(rms), "solves": int(solves),
                          "wall_s": float(wall_s), **(extra or {})}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(self._json.dumps(self.state, indent=1))
            tmp.replace(self.path)
