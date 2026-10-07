"""Aged-cell discharge data -> inverse-problem inputs (step C: commercial-cell data).

Data layout (one npz per cycle, written by scripts/v2_make_aging_dataset.py for the synthetic
stand-in and by a reader of the real LG M50 data set): t [s] from the start of the discharge,
I [A] (> 0 discharge), V [V], and optional ground-truth scalars (theta_n0, theta_p0, eps_am_n,
eps_am_p, SEI thickness, LLI/LAM summary variables).

The inverse PINN sees the CC part of the discharge as a Protocol(current_A, ramp_s, t_end_s) and a
voltage data file restricted to t >= t_min (the real current is a step, the PINN current a tanh ramp,
so the first ~100 s are not compared).

Charge-equivalent alignment (default): the PINN's ramp I tanh(t/tau) passes the charge I (t - tau ln 2),
i.e. it lags the data's current step by tau ln 2 (20.8 s for tau = 30 s, worth 18 mV rms / 128 mV max at
1C on the LG M50 cell, V2_RESULTS.md section 8).  The data are therefore placed at t_pinn = t_data +
tau ln 2 (equal charge passed) and the protocol is lengthened by the same amount.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .params import CellParams, Protocol

TRUTH_KEYS = ("theta_n0", "theta_p0", "eps_am_n", "eps_am_p", "SEI_thickness_m", "cycle")


def load_cycle(path) -> dict:
    raw = np.load(path, allow_pickle=False)
    out = {"t": np.asarray(raw["t"], float), "I": np.asarray(raw["I"], float), "V": np.asarray(raw["V"], float)}
    out["truth"] = {k: float(raw[k]) for k in raw.files if k not in ("t", "I", "V") and np.asarray(raw[k]).ndim == 0}
    return out


def cc_segment(t, I, V, rel_tol=0.05):
    """Constant-current part of a discharge: from the first sample to the last one whose current is within
    rel_tol of the median current (the CV tail, where |I| decays, is cut)."""
    I_med = float(np.median(I[: max(3, len(I) // 2)]))
    ok = np.abs(I - I_med) <= rel_tol * abs(I_med)
    last = int(np.max(np.nonzero(ok)[0]))
    t_cc = t[: last + 1] - t[0]
    return t_cc, I_med, V[: last + 1]


def charge_shift(ramp_s: float, align: str = "charge") -> float:
    """Time offset that makes the ramped PINN current and the stepped data current carry equal charge."""
    return float(ramp_s) * np.log(2.0) if (align == "charge" and ramp_s > 0) else 0.0


def protocol_for_cycle(cycle: dict, ramp_s=30.0, align="charge") -> Protocol:
    t_cc, I_med, _ = cc_segment(cycle["t"], cycle["I"], cycle["V"])
    return Protocol(current_A=abs(I_med), ramp_s=ramp_s, t_end_s=float(t_cc[-1]) + charge_shift(ramp_s, align))


def write_inverse_data(cycle: dict, out_path, t_min=100.0, ramp_s=30.0, align="charge"):
    """Voltage data file for TrainConfig.data_path: CC segment, t_data >= t_min, placed at t_data + charge_shift."""
    t_cc, I_med, V_cc = cc_segment(cycle["t"], cycle["I"], cycle["V"])
    keep = t_cc >= t_min
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, t=t_cc[keep] + charge_shift(ramp_s, align), V=V_cc[keep], I=np.full(int(keep.sum()), abs(I_med)))
    return out_path, int(keep.sum())


def truth_multipliers(cycle: dict, fresh: CellParams | None = None) -> dict:
    """Ground truth expressed as the multipliers the PINN learns (fresh value = 1)."""
    fresh = fresh or CellParams()
    tr = cycle.get("truth", {})
    out = {}
    if "theta_n0" in tr:
        out["theta_n0"] = tr["theta_n0"] / fresh.theta_n0
    if "theta_p0" in tr:
        out["theta_p0"] = tr["theta_p0"] / fresh.theta_p0
    if "eps_am_n" in tr:
        out["eps_am_n"] = tr["eps_am_n"] / fresh.eps_am_n
    if "eps_am_p" in tr:
        out["eps_am_p"] = tr["eps_am_p"] / fresh.eps_am_p
    return out
