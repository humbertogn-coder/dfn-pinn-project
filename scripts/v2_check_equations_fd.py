"""Finite-difference check of the v2 equation conventions on PyBaMM fields.

Independent of any network: evaluates the constitutive laws and sign
conventions used in src/dfn_pinn/v2/residuals.py directly on the reference
arrays (finite-volume cell centres/edges).

    python scripts/v2_check_equations_fd.py results/v2_reference/<reference>.npz

Expected (x80/r120 reference): OCP and j0 identical to PyBaMM, inverse BV
within 1e-3 mV, electrolyte current within ~1e-4 of i_app away from the two
interface faces, charge balance ~1e-4, solid first-order form ~1e-6, salt
balance closed in the electrode interiors (finite differences in time and at
the interfaces dominate the remaining residual).
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dfn_pinn.v2.reference import load  # noqa: E402
from dfn_pinn.v2.params import CellParams, Protocol, Scales, FARADAY  # noqa: E402
from dfn_pinn.v2 import constitutive as cv  # noqa: E402


def T(a):
    return torch.as_tensor(np.asarray(a), dtype=torch.float64)


def main(path):
    ref = load(path)
    cell = CellParams(**ref["meta"]["cell"])
    prot = Protocol(**ref["meta"]["protocol"])
    sc = Scales(cell, prot)
    mesh = ref["meta"]["mesh"]
    x, t = ref["x"], ref["t"]
    ce, phie = ref["c_e"], ref["phi_e"]
    sel = t > 100.0                     # skip the ramp for derivative-based checks
    edges = np.concatenate([np.linspace(0, cell.L_n, mesh["x_n"] + 1),
                            np.linspace(cell.L_n, cell.L_n + cell.L_s, mesh["x_s"] + 1)[1:],
                            np.linspace(cell.L_n + cell.L_s, cell.L, mesh["x_p"] + 1)[1:]])
    width = np.diff(edges)[:, None]
    assert np.allclose(0.5 * (edges[1:] + edges[:-1]), x), "unexpected reference grid"
    n_idx, p_idx = x < cell.L_n, x > cell.L_n + cell.L_s
    interface_faces = [np.argmin(abs(edges[1:-1] - cell.L_n)), np.argmin(abs(edges[1:-1] - cell.L_n - cell.L_s))]
    interface_cells = np.zeros(len(x), bool)
    for xi in (cell.L_n, cell.L_n + cell.L_s):
        interface_cells[np.argsort(abs(x - xi))[:2]] = True

    eps = np.where(n_idx, cell.eps_n, np.where(p_idx, cell.eps_p, cell.eps_s))[:, None]
    kappa = eps ** cell.brug * cv.electrolyte_conductivity(T(ce)).numpy()
    D = eps ** cell.brug * cv.electrolyte_diffusivity(T(ce)).numpy()
    dx = np.diff(x)[:, None]
    harm = lambda a: 2 * a[1:] * a[:-1] / (a[1:] + a[:-1])  # noqa: E731

    # 1) electrolyte current on interior faces
    i_e = -harm(kappa) * (np.diff(phie, axis=0) / dx
                          - 2 * (1 - cell.t_plus) * cell.tdf * cell.V_T * np.diff(np.log(ce), axis=0) / dx)
    err = np.abs(i_e - ref["i_e"][1:-1])[:, sel] / sc.i_ref
    mask = np.ones(err.shape[0], bool)
    mask[interface_faces] = False
    print(f"[i_e]      max |formula - PyBaMM| / i_app, interior faces: {err[mask].max():.2e}")

    # 2) charge balance d i_e/dx = a j
    aj = np.zeros_like(ce)
    aj[n_idx], aj[p_idx] = cell.a_n * ref["j_n"], cell.a_p * ref["j_p"]
    div = np.diff(ref["i_e"], axis=0) / width
    print(f"[charge]   max |d i_e/dx - a j| / (i_app/L_n): {np.abs(div - aj)[:, sel].max() / (sc.i_ref / cell.L_n):.2e}")

    # 3) salt balance eps dc/dt + dN/dx - (1-t+) a j / F
    N = np.concatenate([np.zeros((1, len(t))), -harm(D) * np.diff(ce, axis=0) / dx, np.zeros((1, len(t)))])
    mass = (eps * np.gradient(ce, t, axis=1) + np.diff(N, axis=0) / width
            - (1 - cell.t_plus) * aj / FARADAY) / ((1 - cell.t_plus) * cell.a_n * sc.j_ref_n / FARADAY)
    m = np.abs(mass[~interface_cells][:, sel])
    print(f"[salt]     residual / source scale, electrode interiors: median {np.median(m):.2e}, 95th pct {np.percentile(m, 95):.2e}")

    # 4) kinetics with the v2 constitutive functions
    for k, idx in (("n", n_idx), ("p", p_idx)):
        th, j = ref[f"theta_surf_{k}"], ref[f"j_{k}"]
        U = cv.ocp(T(th), k).numpy()
        j0 = cv.exchange_current(T(ce[idx]), T(th), k, cell).numpy()
        eta = ref[f"phi_s_{k}"] - phie[idx] - U
        eta_bv = cv.bv_overpotential(T(j), T(j0), cell.T).numpy()
        print(f"[kinetics] {k}: |U - U_PyBaMM| {np.abs(U - ref[f'U_{k}']).max():.1e} V, "
              f"|j0/j0_PyBaMM - 1| {np.abs(j0 / ref[f'j0_{k}'] - 1)[:, 1:].max():.1e}, "
              f"|eta - asinh form| {1e3 * np.abs(eta - eta_bv).max():.3f} mV")

    # 5) solid phase, first-order form sigma dphi_s/dx = -(i_app - i_e)
    iapp = prot.current(t) / cell.area
    for k, sig, idx in (("n", cell.sigma_n, n_idx), ("p", cell.sigma_p, p_idx)):
        ph, xs = ref[f"phi_s_{k}"], ref[f"x_{k}"]
        inner = np.concatenate([[False], idx[:-1] & idx[1:], [False]])
        resid = sig * np.diff(ph, axis=0) / np.diff(xs)[:, None] + iapp[None, :] - ref["i_e"][inner]
        print(f"[solid]    {k}: max |sigma dphi/dx + i_app - i_e| / i_app: {np.abs(resid[:, sel]).max() / sc.i_ref:.2e}")

    # 6) particle surface-gradient sign versus reaction sign
    for k, R in (("n", cell.R_n), ("p", cell.R_p)):
        th, r = ref[f"theta_{k}"], ref[f"r_{k}"]
        g = (th[-1] - th[-2]) / ((r[-1] - r[-2]) / R)
        agree = np.mean(np.sign(g[:, sel]) == -np.sign(ref[f"j_{k}"][:, sel]))
        print(f"[particle] {k}: fraction with sign(d theta/d rho) = -sign(j): {agree:.3f}")


if __name__ == "__main__":
    main(sys.argv[1])
