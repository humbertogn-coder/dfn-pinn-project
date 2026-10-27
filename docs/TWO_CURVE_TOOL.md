# Two-curve tool: fitting measured Li-SPAN discharge curves with the chemistry network

Status 2026-10-27: built and validated against finite-volume (FV) fits on six published datasets.
Network: `results/lispan_chem_runs/chem_v2_norar_s0/final.pt` (Phase 3b v2, docs/PHASE3B_CHEM_PINN.md).

## What it does

Given discharge curves of one cell measured at two (or more) constant currents, the tool returns the effective
parameters of the Li-SPAN continuum model with uncertainties and validity flags, in about a minute on one CPU core.
An FV fit of the same curves takes 12-50 min from a good starting point, and hours with multiple starts.

| parameter | meaning | how the tool handles it |
|---|---|---|
| w_S | effective active-sulfur factor, Q_data = w_S Q_model | sets the model C-rate of each curve: r = 0.767 i / w_S (i in A/g of the data basis) |
| dU0_2 | step-2 potential relative to step 1 [V] | network input, box -0.15...+0.28 V |
| dU0_3rel2 | step-3 potential relative to step 2 [V] | network input, box -0.26...+0.20 V (the steps stay in order) |
| b_12, b_3 | OCV-slope multipliers of steps 1-2 and of step 3 | network inputs, x0.5-1.6 and x0.4-3.5 |
| k0_1 | rate-constant multiplier of the reversible step 1 | network input, x0.1-100 |
| d | common shift of U0_1..3 [V] | analytic (V + d), +-0.2 V |
| Z_CC | contact resistance [Ohm m2] | analytic (V - Z_CC I), 0-0.05 Ohm m2 (0-500 Ohm cm2) |

The capacity of a data point fixes the model time (t = 3.6 q / i, plus the inverse of the model's 30 s current ramp);
this time does not depend on w_S, so w_S acts only through the rate input.  For the irreversible steps 2-3 only
k0 exp(F U0 / 2RT) matters, so their k0 are carried by the potentials (docs/PHASE1_IDENTIFIABILITY.md).

Fit: bounded least squares (trust region) on all curves, each curve weighted equally; 8 starts (box centre plus a
Latin hypercube); central-difference Jacobian with all perturbed parameter sets in one network call.

## The rate range sets which data can be used

The network was trained for model rates 0.1-1 C.  Every curve must map inside that range at the fitted w_S:

    w_S in [0.767 i_max / 1.0,  0.767 i_min / 0.1]

So the fastest-to-slowest current ratio can be at most 10.  At exactly 10, w_S is fixed by the range; Simanjuntak
0.1 + 1 C is that case, with w_S = 1, which is the paper's own normalization.  The tool refuses a wider span.

The fast curves of the literature datasets map above 1 C, so the pairs used in Phase 2 do not fit in the range:

| dataset | Phase 2 pair | model rates at the Phase 2 w_S | pair used here |
|---|---|---|---|
| Hao 2025 SPAN (w_S ~0.7) | 0.1 + 2 C | 0.08 / 1.6 C | 0.2 + 1 C (prediction 0.5 C) |
| Chen 2019 ether / carbonate (w_S ~1.2) | 0.4 + 2 A/g | 0.26 / 1.3 C | 0.4 + 1 A/g |
| Ma 2026 SPAN / SPAN-A (w_S ~0.7-1.1) | 0.2 + 1 C | 0.2-0.36 / 1.2-1.8 C | 0.2 + 0.5 C (prediction 0.3 C) |
| Simanjuntak 2024 Fig. 4b | 0.1 + 1 C | 0.1 / 1 C | 0.1 + 1 C (prediction 0.2 C) |

Fitting the literature's fast rates (2 C and above) would need a network trained on a wider rate range.

## Usage

    python scripts/lispan_two_curve_fit.py --dataset hao2025_SPAN --rates 0.2 1 --predict 0.5 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --experiment 0.1 1 --predict 0.2 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --csv cellA_0.2C.csv:0.335 cellA_1C.csv:1.675 --name cellA --plot
    python scripts/lispan_two_curve_fit.py --dataset chen2019_ether --rates 0.4 1 --fix k0_1=1

Data:
- `--csv` takes two columns, capacity [mAh/g] and voltage [V], with one header line, at the specific current after
  the colon [A/g] on the same capacity basis.
- `--dataset` takes the curves of scripts/lispan_lit_rates.py.
- `--experiment` takes the Simanjuntak Fig. 4b points.
- The fit uses n = 80 points, uniform in capacity, from the dataset's minimum capacity up to `--frac` (default 0.90)
  of the measured end capacity.

Other options:
- `--fix` holds parameters at given values, in natural units.
- `--predict` gives the network curve at other rates and its misfit to data at those rates.
- `--fv-check` runs the FV solver at the solution and refits those FV curves with the network.  The parameter change
  is the error the network itself causes at this solution (the "network bias").

Output: `results/lispan/two_curve/<name>.json`, plus `<name>.png` with `--plot`.  The JSON holds:
- the parameters, with linearized sd (w_S, b, k0 in % of the value; dU0 and d in mV; Z_CC in Ohm cm2);
- the correlation matrix;
- the rms misfit per curve, the model rates, and T_max (the data end as a fraction of the network window);
- the per-step U0 and b;
- the flags;
- every start's solution, the equivalent alternatives, and their spread;
- the predictions and the FV check.

### Validity flags

| flag | level | meaning |
|---|---|---|
| misfit | warn > 5 mV, bad > 15 mV | The rms misfit is well above the network's own error (1-2.5 mV vs FV).  The continuum model does not reproduce the data, so the parameters are effective values. |
| box_edge | warn | A chemistry input is within 2 % of the box width of a bound.  The optimum may lie outside, and the network is least accurate at the box corners. |
| rate_edge | warn | w_S is at a rate-range bound: a curve sits at 0.1 or 1 C. |
| beyond_window | warn | Data reach past the network's time window (T > 1): the network extrapolates there.  Lower `--frac`. |
| poorly_identified | warn | sd > 25 % of the box width. |
| non_unique | warn | Another start fits as well (cost difference inside the 95 % chi-square region) but sits > 3 sd away.  The linearized sd understates the uncertainty; use `spread`. |
| network_bias | warn | The FV self-check moves a parameter by > 1 sd. |
| correlated | info | Correlation above 0.95 between two parameters. |
| Z_CC_zero, w_S_fixed_by_range | info | The parameter is at its physical bound, or is fixed by the rate span. |

The linearized sd uses sigma from the residuals.  When the residuals are model error (smooth and correlated) rather
than noise, it is a lower bound.

## Validation against FV fits (2026-10-27)

Method:
1. For each dataset, the network fit is run on the points described above.
2. Then the same points, parameters, bounds and curve weights are fitted with the FV solver, starting from the
   network's solution (`scripts/lispan_lit_v2form.py`).  The grid is 10 / 5 volumes with rtol 1e-5 for the fit, and
   20 / 10 volumes for the reported curves.
3. The FV optimum is what an exact surrogate would return.  The difference between the two fits is the error of the
   tool.

Figure: `report/phase3/figures/p3b_two_curve.png`, made by `report/phase3/make_two_curve_figure.py`, which also prints
this table.

| data | free | network rms [mV] | FV at network [mV] | FV optimum [mV] | net - FV, max over parameters [sd] | prediction (network / FV optimum) [mV] | flags |
|---|---|---|---|---|---|---|---|
| Simanjuntak 0.1+1 C | 7 | 17.0 | 17.1 | 17.0 | 0.31 (b_12) | 0.2 C: 24.4 / 23.8 | misfit, non_unique |
| Chen ether 0.4+1 A/g | 8 | 15.5 | 17.5 | 15.9 | 1.79 (b_3) | - | box_edge b_3, misfit, network_bias, non_unique |
| Chen carb. 0.4+1 A/g | 8 | 14.1 | 14.3 | 14.2 | 0.52 (b_3) | - | misfit |
| Hao 0.2+1 C | 8 | 12.4 | 12.7 | 12.5 | 0.26 (w_S) | 0.5 C: 15.1 / 15.1 | misfit, poorly_identified b_3 |
| Ma SPAN 0.2+0.5 C | 8 | 45.7 | 47.5 | 45.7 | 1.47 (b_3) | 0.3 C: 51.6 / 52.0 | box_edge b_3, box_edge k0_1, misfit, non_unique, poorly_identified dU0_3rel2, poorly_identified k0_1 |
| Ma SPAN-A 0.2+0.5 C | 8 | 47.4 | 47.3 | 46.8 | 0.38 (w_S) | 0.3 C: 36.7 / 36.7 | beyond_window, box_edge k0_1, misfit, non_unique, poorly_identified k0_1 |

Network fits (sd in brackets; w_S, b, k0 in %, potentials in mV, Z_CC in Ohm cm2):

| data | w_S | dU0_2 | dU0_3rel2 | b_12 | b_3 | k0_1 | d | Z_CC |
|---|---|---|---|---|---|---|---|---|
| Simanjuntak 0.1+1 C | 1 (fixed) | +4 (30) | +58 (28) | 0.85 (10) | 0.92 (12) | 0.67 (73) | +7 (29) | 314 (13) |
| Chen ether 0.4+1 A/g | 1.257 (0.7) | +50 (8) | -203 (29) | 1.06 (3) | 3.50 (7), edge | 2.7 (27) | +25 (8) | 10 (9) |
| Chen carb. 0.4+1 A/g | 1.143 (0.5) | +85 (6) | -119 (35) | 0.83 (3) | 2.40 (19) | 4.4 (26) | -30 (6) | 0 (7) |
| Hao 0.2+1 C | 0.733 (1.0) | +63 (5) | -173 (23) | 0.86 (2) | 0.72 (110) | 2.5 (16) | -25 (4) | 28 (5) |
| Ma SPAN 0.2+0.5 C | 0.828 (6) | -41 (21) | -225 (150) | 1.04 (6) | 3.50 (42), edge | 100 (470), edge | +39 (16) | 420 (31) |
| Ma SPAN-A 0.2+0.5 C | 0.906 (3) | +183 (17) | -42 (30) | 0.67 (14) | 0.53 (12) | 100 (880), edge | +12 (24) | 330 (22) |

Findings:
1. **The network reaches the FV optimum.**
   - Over the 47 free parameters of the six fits, the network and FV estimates differ by less than 1 sd in 45 cases,
     and by 0.5 sd or less in 41.
   - The two exceptions are both b_3 at the upper box edge (x3.5): Chen ether (1.8 sd) and Ma SPAN (1.5 sd).  The
     tool flags both (box_edge; for Chen ether also network_bias from the self-check).
   - The FV optimum's misfit is within 0.6 mV of the network fit's in every case.
   - The four predictions of a third rate agree with FV within 0.6 mV (Hao 0.5 C, Ma SPAN and SPAN-A 0.3 C,
     Simanjuntak 0.2 C).
2. **The misfit is the continuum model's, not the network's.**
   - The misfits are 12-17 mV for Hao, Chen and Simanjuntak, and 46-47 mV for Ma.  The FV solver gives the same
     values.
   - The Phase 2 conclusion stands: the effective Li-SPAN model leaves tens of mV on two-rate fits of real SPAN cells
     (Phase 2 free fits: 11-65 mV), so the parameters are effective values.
   - For the Ma cells (46-47 mV), the data are not described by the model.  The fits sit at box edges (k0_1 = 100,
     b_3 = 3.5), the alternatives span most of the box (non_unique), and those parameters should not be interpreted.
3. **What is identified.**
   - On the cells the model describes best (Chen, Hao): w_S to 0.5-1 %, dU0_2 to 5-9 mV, d to 4-8 mV, and b_12 to
     2-3 %.
   - dU0_3rel2 is identified to 23-35 mV, k0_1 to 16-27 %, and Z_CC to 5-9 Ohm cm2.
   - b_3 is identified only when the data reach step 3 deeply: 7-19 % for Chen, but 110 % for Hao, whose curves end
     at 73 % of the network window.
4. **Network bias** (FV self-check):
   - The bias is up to 2 mV in dU0_2 and d, up to 8 mV in dU0_3rel2, below 1 % in w_S and b_12, and up to 3 % in
     k0_1.  At the box edge b_3 moves by 10-24 %.
   - It is below 0.6 sd everywhere except b_3 of Chen ether (1.5 sd), the same parameter that differs most from the
     FV fit.
   - At the box edge its sign does not predict the direction of the error (Chen ether b_3: self-check -10 %, network
     minus FV +12 %).  Use the bias as a flag, not as a correction.
5. **Cost.**
   - Network fit with 8 starts: 35-70 s on one core.
   - FV check: 25-35 s.
   - FV fit from the network's solution: 12-50 min with 24-86 residual evaluations, i.e. 12-60x the network fit,
     before counting multiple starts.

Limitations:
- **Validation covers agreement, not the global optimum.**  The FV fits started at the network's solution, so the
  validation shows that the network and FV agree on the optimum the network finds.  It does not show that this is
  the global FV optimum.  The network's 8 starts and the non_unique flag address that part.
- **Rate range 0.1-1 C** (above).
- **Time window.**
  - The network covers 0.95 of the FV cut-off time at Z_CC = 0, so the last ~10 % of the measured capacity is not
    used by default (frac 0.90).
  - When the fitted curve's cut-off comes early (small Z_CC, Ma SPAN-A 0.2 C), the data can still pass the window.
    The beyond_window flag catches this; on that curve the network differs from FV by 3-11 mV.
- **Box edges.**
  - Three of the six fits sit at a box edge: b_3 = 3.5, or k0_1 = 100.
  - Chen ether is a cell the model describes at 15 mV, yet b_3 still hits the upper edge.  A future network could
    extend b_3 to ~5.
  - For the Ma cells the edges reflect the model misfit, not the box.

## Synthetic test with w_S free (known truth, 2026-10-27)

The literature validation compares the network with FV but has no true answer.  This test has one
(`scripts/lispan_two_curve_synthetic.py`, results `results/lispan/two_curve/synthetic_0.2+0.6.json`):
1. FV discharges are generated for the 13 evaluation chemistries of the v2 design.  They include the box-corner test
   chemistries p5-p7 and the three literature-like ones.
2. The rates are 0.2 + 0.6 C.  Interior rates are needed so the rate range does not pin w_S; the evaluation
   references at 0.1 / 1 C would put the true w_S exactly on a bound.
3. The truth has w_S = 0.8 or 1.25 (alternating), d = +20 mV and Z_CC = 20 Ohm cm2.
4. The curves are fitted exactly like measured data (capacity basis, 80 points to 0.90 of the end capacity, 8
   starts), noise-free and with 5 mV noise (2 draws per chemistry).

Noise-free errors come from the network alone.  The 10 chemistries without flags give:

| parameter | median | max | note |
|---|---|---|---|
| w_S | 0.12 % | 0.26 % | |
| dU0_2 | 0.4 mV | 1.4 mV | |
| dU0_3rel2 | 1.5 mV | 4.1 mV | |
| b_12 | 0.2 % | 0.3 % | |
| b_3 | 0.6 % | 1.1 % | |
| k0_1 | 2.2 % | 35 % | max at p1 (k0_1 x45: step 1 near equilibrium, weak sensitivity; sd at 5 mV ~40 %) |
| d | 0.5 mV | 1.2 mV | |
| Z_CC | 1.3 Ohm cm2 | 2.0 Ohm cm2 | |

The three flagged chemistries (box_edge) are the worst cases:
- **p6** (dU0_2 +0.26 V, dU0_3rel2 +0.18 V, b_3 x0.45): the fit runs to the dU0_3rel2 edge.  The errors are +20 mV in
  dU0_3rel2, +9 % in b_3 and 0.8 % in w_S.
- **p7** (k0_1 x93): k0_1 comes out -59 %, flagged through b_12 at its edge.
- **p5**: small errors despite the flag.

With 5 mV noise (26 fits), the error divided by the tool's linearized sd has a median of 0.55-0.98 for every
parameter except w_S.  92-100 % of those values lie within 2 sd (95 % expected).  w_S is the exception: median 1.23,
only 81 % within 2 sd.  Its noise sd is small (median 0.22 %), so the network's own error of ~0.15 % is not
negligible against it.  In absolute terms all w_S errors of unflagged fits stay below 0.6 %.  Practical reading:
treat the w_S uncertainty as at least ~0.3 %.

The misfit flag fires in 14 of the 26 noisy fits, because rms 4.5-5.6 mV is just above its 5 mV threshold.  The
threshold assumes measurement noise well below 5 mV.

## Files

- `src/dfn_pinn/lispan/two_curve.py`: Curve, TwoCurveFit (mapping, fit, report with flags, prediction), FV
  counterpart (fv_params, fv_curve, fv_check).
- `scripts/lispan_two_curve_fit.py`: CLI (data loaders, JSON and figure).
- `scripts/lispan_lit_v2form.py`: FV fit of the same points for the validation (checkpointed, `--resume`).
- `report/phase3/make_two_curve_figure.py`: validation figure and table.
- `scripts/lispan_two_curve_synthetic.py`: known-truth test (FV curves of the design's evaluation chemistries at
  interior rates, w_S / d / Z_CC away from nominal, noise-free and noisy).  `--workers 2 --draws 2` took 52 min in
  the sandbox.
- `tests/test_lispan_two_curve.py`: 5 tests:
  - ramp inversion;
  - rate mapping and w_S bounds;
  - exact recovery of network-generated curves;
  - prediction and rate limits;
  - FV parameter mapping.
- Results (git-ignored): `results/lispan/two_curve/<name>.json|.png`, `<name>_fv.json`.

Validation commands (sandbox, 2026-10-27):

    python scripts/lispan_two_curve_fit.py --dataset chen2019_ether --rates 0.4 1 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --dataset chen2019_carbonate --rates 0.4 1 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --dataset hao2025_SPAN --rates 0.2 1 --predict 0.5 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --experiment 0.1 1 --predict 0.2 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --dataset ma2026_SPAN --rates 0.2 0.5 --predict 0.3 --fv-check --plot
    python scripts/lispan_two_curve_fit.py --dataset ma2026_SPAN-A --rates 0.2 0.5 --predict 0.3 --fv-check --plot
    python scripts/lispan_lit_v2form.py --dataset chen2019_ether --rates 0.4 1          # and the same for the others
    python report/phase3/make_two_curve_figure.py
