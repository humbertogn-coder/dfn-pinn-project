"""Compare a trained v2 model with the PyBaMM reference on the native grid."""

from __future__ import annotations

import numpy as np
import torch


def _t(a, dtype):
    return torch.as_tensor(np.asarray(a, dtype=np.float64), dtype=dtype).reshape(-1, 1)


def _chunked(fn, *args, chunk=4096, grad=False):
    """Evaluate fn on row chunks (keeps memory bounded on the full reference grid)."""
    outs = []
    for i in range(0, args[0].shape[0], chunk):
        part = [a[i:i + chunk] for a in args]
        with torch.set_grad_enabled(grad):
            outs.append(fn(*part).detach())
    return torch.cat(outs)


def predict_on_reference(model, ref, max_times=None, kinetics="inverse"):
    """Return predicted fields on the reference grid (numpy, physical units).

    kinetics="hard": phi_e inside the electrodes is the Butler-Volmer-defined field.
    """
    dtype = next(model.parameters()).dtype
    c = model.sc.cell
    t_s = ref["t"]
    if max_times is not None and len(t_s) > max_times:
        idx = np.unique(np.linspace(0, len(t_s) - 1, max_times).astype(int))
    else:
        idx = np.arange(len(t_s))
    t_s = t_s[idx]
    th = t_s / model.sc.t_end
    out = {"t": t_s, "idx": idx}
    with torch.no_grad():
        tt = _t(th, dtype)
        out["V"] = model.voltage(tt).numpy().ravel()
        # electrolyte on the whole-cell grid
        X = ref["x"] / c.L
        XX, TT = np.meshgrid(X, th, indexing="ij")
        Xt, Tt = _t(XX.ravel(), dtype), _t(TT.ravel(), dtype)
        s1 = (Xt > c.X1).to(dtype)
        s2 = (Xt > c.X2).to(dtype)
        out["c_e"] = (_chunked(model.ce, Xt, Tt, s1, s2) * c.c_e0).numpy().reshape(XX.shape)
        phie = _chunked(model.phie, Xt, Tt, s1, s2)
        if kinetics == "hard":
            from .residuals import Residuals
            res = Residuals(model, "hard")
            for k, lo, hi in (("n", -1.0, c.X1), ("p", c.X2, 2.0)):
                mask = ((Xt > lo) & (Xt < hi)).view(-1)
                if mask.any():
                    def fn(X, T, k=k):
                        X = X.clone().requires_grad_(True)
                        return res.electrode_fields(k, X, T)["phie"]
                    phie[mask] = _chunked(fn, Xt[mask], Tt[mask], chunk=2048, grad=True)
        out["phi_e"] = phie.numpy().reshape(XX.shape)
        for k, xs, L0, Lk in (("n", ref["x_n"], 0.0, c.L_n), ("p", ref["x_p"], c.L_n + c.L_s, c.L_p)):
            xk = (xs - L0) / Lk
            XK, TK = np.meshgrid(xk, th, indexing="ij")
            xkt, tkt = _t(XK.ravel(), dtype), _t(TK.ravel(), dtype)
            out[f"j_{k}"] = _chunked(lambda a, b: model.j(k, a, b), xkt, tkt).numpy().reshape(XK.shape)
            out[f"phi_s_{k}"] = _chunked(lambda a, b: model.phis(k, a, b), xkt, tkt).numpy().reshape(XK.shape)
            # theta() differentiates h at the surface internally -> needs grad mode
            th_surf = _chunked(lambda a, b: model.theta(k, torch.ones_like(a), a, b), xkt, tkt, chunk=2048, grad=True)
            out[f"theta_surf_{k}"] = th_surf.numpy().reshape(XK.shape)
    return out


def compare(model, ref, max_times=None, kinetics="inverse"):
    """Field errors with the same kinds of limits as the v1 acceptance policy."""
    pred = predict_on_reference(model, ref, max_times, kinetics=kinetics)
    idx = pred["idx"]
    c = model.sc.cell
    r = {}

    def err(key, ref_key=None):
        a = pred[key]
        b = ref[ref_key or key][..., idx]
        e = a - b
        return float(np.sqrt(np.mean(e ** 2))), float(np.max(np.abs(e)))

    r["V_rmse_mV"], r["V_max_mV"] = (1e3 * v for v in err("V"))
    r["ce_rmse"], r["ce_max"] = err("c_e")
    r["phie_rmse_mV"], r["phie_max_mV"] = (1e3 * v for v in err("phi_e"))
    r["phisp_rmse_mV"], r["phisp_max_mV"] = (1e3 * v for v in err("phi_s_p"))
    for k, cmax, jref in (("n", c.cmax_n, model.sc.j_ref_n), ("p", c.cmax_p, model.sc.j_ref_p)):
        rm, mx = err(f"j_{k}")
        r[f"j_{k}_rmse_rel"], r[f"j_{k}_max_rel"] = rm / jref, mx / jref
        rm, mx = err(f"theta_surf_{k}")
        r[f"thsurf_{k}_rmse"], r[f"thsurf_{k}_max"] = rm, mx
    return r, pred


# Screening limits used in the report (same spirit as configs/dfn_baseline_v1.json).
LIMITS = {
    "V_max_mV": 5.0, "phie_max_mV": 5.0, "phisp_max_mV": 5.0, "ce_max": 10.0,
    "j_n_max_rel": 0.01, "j_p_max_rel": 0.01, "thsurf_n_max": 1e-3, "thsurf_p_max": 1e-3,
}


def screen(metrics):
    return {k: (metrics[k], lim, metrics[k] <= lim) for k, lim in LIMITS.items() if k in metrics}
