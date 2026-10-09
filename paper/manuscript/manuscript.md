---
title: "A physics-informed neural network for a lithium / sulfurized-polyacrylonitrile cell model: accuracy, identifiability and parameter estimation"
target: Journal of The Electrochemical Society
status: DRAFT 1 (all sections drafted, 2026-10-09; references incomplete; for review by C. Guerrero and P. Balbuena)
numbers: every double-brace placeholder is filled from paper/numbers.json by scripts/paper_build.py (manuscript_filled.md)
---

# Abstract

We present a physics-informed neural network (PINN) for the continuum model of a lithium / sulfurized-polyacrylonitrile
(Li-SPAN) cell of Simanjuntak et al., together with a documented finite-volume reference that closes the places where
the published model is ambiguous. Building the reaction-extent ordering, the applied current and the Butler-Volmer
relations into the network, and adding a global charge-inventory residual, the PINN trained from random
initialization reproduces the reference cell voltage to {{forward_V_rmse_mV_0.1C_mean}} mV rms at 0.1 C and
{{forward_V_rmse_mV_1C_mean}} mV rms at 1 C over three seeds. A Fisher-information analysis shows that one discharge
rate cannot separate the reaction kinetics from the contact resistance, while 0.1 C and 1 C together determine eight
parameters to about 1 % or better. With a Levenberg-Marquardt step on the parameters, the multi-rate inverse PINN
recovers them from noisy synthetic curves to within {{inverse_lm_vs_ideal_max_crlb}} Cramer-Rao bounds of an ideal
least-squares estimator, and the remaining difference is predicted quantitatively by the PINN's own forward error.
Applied to measured curves of two SPAN cells, the model is calibrated to 10-20 mV, limited by model-form error and,
for single-rate data, by the confounding of kinetics and open-circuit parameters. A finite-volume discharge costs
seconds and PINN training hours; we state where the PINN is and is not worth this cost.

# 1. Introduction

Sulfurized polyacrylonitrile (SPAN) cathodes store sulfur covalently bound to a carbonized polymer backbone, so
that in carbonate electrolytes the lithium-sulfur conversion proceeds without dissolved polysulfides. Simanjuntak
et al. [1] formulated the first continuum model of a Li | SPAN cell: three sequential one-electron reduction steps
of a bound S4 chain, Li2S precipitation from the S2- they release, and Nernst-Planck transport in the electrolyte,
validated against discharge curves at 0.05-1 C. Physics-informed neural networks (PINNs) [2, 3] solve such models
by training a network to satisfy the governing equations and have been applied to the Doyle-Fuller-Newman (DFN)
model of Li-ion cells [4-6]; their attraction is that the same network can absorb measured data and return
parameters [7].

The Li-SPAN model is a harder target than the DFN for a PINN in three respects: the cathode carries several local
state variables coupled through multi-reaction Butler-Volmer kinetics in the Tafel regime, a trace species varies
over nine decades, and the reaction fronts move through the electrode. In this work we (i) close the places where
the published model is ambiguous and document a finite-volume (FV) reference of it, (ii) build a PINN that
reproduces that reference to a fraction of a millivolt, with the structure that made this possible, (iii) quantify
what discharge curves can identify, (iv) estimate parameters from synthetic and measured curves and compare with an
ideal least-squares estimator, and (v) state the computational cost honestly.

# 2. Model

We use the model of Ref. [1] (main text and supplementary information) in one spatial dimension: cathode
$0 \le y \le L_\mathrm{cat}$ (100 um) and glass-fibre separator $L_\mathrm{cat} \le y \le L_\mathrm{tot}$ (618 um),
the Li foil a planar electrode at $y = L_\mathrm{tot}$. The SPAN species obey local balances (no solid transport)

$$\frac{\partial c_{S4}}{\partial t} = -\tfrac12 a r_1,\quad
\frac{\partial c_{S3}}{\partial t} = \tfrac12 a (r_1 - r_2),\quad
\frac{\partial c_{S2}}{\partial t} = \tfrac12 a (r_2 - r_3),\quad
\frac{\partial c_{S1}}{\partial t} = \tfrac12 a (r_1 + r_3),$$

with mass-action Butler-Volmer rates

$$r_m = k_{0,m}\left[a^\mathrm{ed}_m e^{-x_m} - \delta_m\, a^\mathrm{prod}_m e^{x_m}\right],\qquad
x_m = \frac{F}{2RT}\left(\Delta\phi - U_{0,m} + b_m \zeta_m\right),\quad \zeta_m = 1 - \frac{c_m}{c_\mathrm{ref}},$$

where $\Delta\phi = \phi_s - \phi_e$, the open-circuit potential of each step is linear in its reaction extent
(Table 3 of [1]) and $\delta_m \in \{0, 1\}$ marks reversible steps. Li2S precipitates from the released S2- with a
solubility product; the electrolyte (1 M LiPF6 in EC:DEC) follows Nernst-Planck transport with the measured salt
diffusivity, transference number and conductivity; charge conservation couples the faradaic current
$F a \sum_m r_m$ to the ionic and electronic currents; the cell voltage is
$E = \phi_s(0) - \phi_s(L_\mathrm{tot}) - Z_\mathrm{CC} I$. The complete equation set, parameter values and their
sources are given in the Supplementary Information.

**Closures.** Reproducing the published curves required choices where the source is silent or inconsistent
(Table 1): the faradaic current includes reaction 1 (the printed definition counts only S2- production, although
reaction 1 carries one third of the capacity); the rate constants of Table 3 put every reaction at equilibrium,
whereas the published rate dependence (0.17 V between 0.1 and 1 C) requires the Tafel regime, so three effective
constants were fitted to the published curves ({{fig6a_rms_mV_0.1C}} / {{fig6a_rms_mV_1C}} mV rms at 0.1 / 1 C);
reaction 3 is irreversible during discharge (its literal reverse term changes the voltage by
{{assumption[all three reactions reversible (literal mass action, TTT)]_0.1C_rms_mV}} mV rms and contradicts the
published species evolution); and the S2- reference concentrations follow from the published open-circuit break
points and saturation plateau. Each closure is quantified in Table 1.

**Benchmark for the PINN.** The PINN is asked to reproduce the FV solution of the same equations with three
numerical simplifications, each small compared with the model-experiment misfit and quantified in Table S1:
reaction 2 also irreversible, no double-layer charging, and a 30 s current ramp.

# 3. Finite-volume reference

The equations are discretized with a cell-centred finite-volume method (40 cathode and 20 separator volumes; 20 /
10 changes the voltage by less than {{assumption[coarser grid N_c = 20, N_s = 10]_1C_max_mV}} mV), with logarithmic
species states, and integrated with a stiff BDF method (relative tolerance 1e-6, sparse finite-difference
Jacobian); one discharge takes {{cost_fv_0.1C_40x20_cpu_s}} s (0.1 C) on one CPU core. Figure 1 compares the
reference with the published curves.

# 4. Physics-informed neural network

**Hard structure.** Everything the physics fixes exactly is built into the parametrization; the rest is penalized.
Inputs are $(Y, T) = (y / L, t / t_\mathrm{end})$ with Fourier and short-time features in $T$.

* *Reaction extents.* Network outputs $P_m > 0$ define the remaining educt fractions $e_m = \exp(-T P_m)$ and the
  extents $\xi_1 = 1 - e_1$, $\xi_2 = \xi_1 (1 - e_2)$, $\xi_3 = \xi_2 (1 - e_3)$, so that
  $0 \le \xi_3 \le \xi_2 \le \xi_1 < 1$ and $\xi(0) = 0$ hold exactly; the species follow as
  $c_{S4} = c_0 e_1$, $c_{S3} = c_0 \xi_1 e_2$, $c_{S2} = c_0 \xi_2 e_3$, $c_{S1} = c_0 (\xi_1 + \xi_3)$, and the
  Li2S volume fraction from $\xi_2 + \xi_3$. Sulfur conservation, positivity and the initial state are exact.
  Species are computed from $e_m$ directly: forming $1 - \xi_1$ in single precision loses the educt once
  $T P_1 > 17$.
* *Current.* A positive network field $\rho$ distributes the total reduction rate,
  $R = (I/F)\, \rho / \int_0^{L_\mathrm{cat}} a \rho\, \mathrm{d}y$, so that the faradaic current integrates to
  the applied current at every time (Gauss-Legendre quadrature); the ionic current follows by integration.
* *Kinetics.* $\Delta\phi(y, t)$ is the root of $\sum_m r_m(\Delta\phi) = R$ (bracketed bisection plus Newton,
  gradients by the implicit-function theorem with a floored slope), so the Butler-Volmer relations hold exactly
  and the individual rates split $R$ among the three steps.
* *Electrolyte.* The salt concentration is a bounded deviation from its initial value that vanishes at $t = 0$;
  the electrolyte potential satisfies the anode kinetics by construction.

**Residuals.** (1) The three extent equations
$\mathrm{d}\xi_m / \mathrm{d}T = t_\mathrm{end} a r_m / (2 c_0)$; (2) Ohm's law in the solid and (3) in the
electrolyte (Huber loss: the Tafel relation turns early spatial roughness of the species into very large potential
gradients); (4) the salt balance in cathode and separator with its flux conditions; (5) the global salt inventory;
and (6) the global charge inventory

$$2 F c_0 L_\mathrm{cat} \int_0^1 (\xi_1 + \xi_2 + \xi_3)\, \mathrm{d}Y = \int_0^t I\, \mathrm{d}t,$$

which follows from the extent equations but is not enforced by them: their small mean residual integrates over a
nine-hour discharge into an extent excess that, in the last plateau, appears directly as a voltage error through
the open-circuit slope $b_3$. The separator salt residual is scaled with its own natural source
$(1 - t_+) I / (F L_\mathrm{sep})$: on the cathode scale it looked converged while the 1 C separator transient was
25 % off. Residual groups are balanced by adaptive weights that equalize their gradient norms.

**Training.** Four fully connected networks (4 x 96, tanh), 30 000 Adam steps with the learning rate decaying
from 1e-3 to 1e-5, 768 / 512 / 96 collocation points (cathode / separator / boundary) resampled every step, the
charge-inventory residual switched on after 10 000 steps (active from the start it competes with the extent
equations before they are learned), single precision, one CPU thread; three seeds per rate.

# 5. Identifiability and parameter estimation

**Local identifiability.** Sensitivities of the FV voltage to 17 parameters (central differences) at 0.05-1 C give
the Fisher information for 1 mV white noise and the Cramer-Rao lower bounds (CRLB).

**Inverse PINN.** One network per rate, sharing the unknown parameters (log-multipliers of $k_{0,m}$, $b_m$,
$Z_\mathrm{CC}$ and an offset of $U_{0,1}$). Stage 1 re-adapts the forward networks at the initial guess; stage 2
adds a data misfit. Plain gradient descent on all unknowns crawls along the correlated parameter directions;
instead, every 250 steps a Levenberg-Marquardt step is taken on the parameters with the fields frozen
(finite-difference Jacobian of the data residuals, about 10 s), after which the fields re-equilibrate. For
measured data, where the model cannot reproduce the data exactly, the fields are trained by the physics only and a
parameter step is accepted only if the re-equilibrated misfit decreases (trust region).

**Reference estimator.** The same parameters are fitted with the FV model by nonlinear least squares
(trust-region reflective, forward-difference Jacobian with absolute steps of 1 % / 1 mV) on exactly the same data
and noise realization; with synthetic data this is the ideal estimator against which the PINN is judged.

# 6. Results

## 6.1 Forward accuracy and reproducibility

Trained from random initialization with the recipe of Section 4, the PINN reproduces the FV cell voltage to
{{forward_V_rmse_mV_0.1C_mean}} mV rms at 0.1 C and {{forward_V_rmse_mV_1C_mean}} mV rms at 1 C (means over three
seeds; spread {{forward_V_rmse_mV_0.1C_spread}} / {{forward_V_rmse_mV_1C_spread}} mV; Fig. 2, Table 2), evaluated from
100 s after the start of the current ramp to 98 % of the discharge. The largest deviations sit at the hand-overs between
the three reactions, where the open-circuit potential changes slope (Fig. 2, bottom); for one 0.1 C seed this is a
narrow spike of {{forward_V_max_mV_0.1C_seed1}} mV at the start of the third plateau, the other runs stay below
{{forward_V_max_mV_0.1C_seed2}} mV. The internal states are
reproduced as well (Fig. 3): the SPAN species to below 1 mol/m3 of 598, the Li2S volume fraction and, at 1 C, the
electrolyte concentration (Table 2). Two ingredients were decisive and are worth stating as negative results. Without
the global charge-inventory residual the 0.1 C error wanders between about 0.7 and 1.5 mV during training although
every local residual keeps falling: the extent equations are satisfied on average only to their residual level, and
the integrated excess appears in the last plateau through the open-circuit slope $b_3$ (1 mV per mol/m3). With the
separator salt residual scaled like the cathode one, the 1 C separator transient converged to a 25 % error while
the residual looked small.

## 6.2 What a discharge curve identifies

Figure 4 shows the Cramer-Rao bounds of the eight parameters used below for 1 mV noise and 100 points per curve.
Three properties of the model decide what can be estimated. (i) For a step treated as irreversible (Tafel), the rate
depends on $k_{0,m}\exp(F U_{0,m}/2RT)$ only, so $U_{0,2}$ and $U_{0,3}$ are exactly confounded with $k_{0,2}$ and
$k_{0,3}$ (correlation -1.000); only the products are identifiable, and $U_{0,2}$, $U_{0,3}$ are fixed at their
open-circuit values. (ii) At a single rate, every $k_{0,m}$, $U_{0,1}$ and $Z_\mathrm{CC}$ produce nearly the same
uniform voltage offset: at 0.1 C the bounds are {{crlb_0.1C_k0_2}} % for $k_{0,2}$ and {{crlb_0.1C_Z_CC}} % for
$Z_\mathrm{CC}$, i.e. these parameters are not identifiable, while the open-circuit slopes $b_m$ are (below
{{crlb_0.1C_b_1}} %). Two rates separate the ohmic drop, which scales with $I$, from the Tafel shift, which scales
with $\ln I$: with 0.1 C and 1 C all eight parameters are bounded to {{crlb_0.1C+1C_k0_1}} % or better
($Z_\mathrm{CC}$ {{crlb_0.1C+1C_Z_CC}} %, $U_{0,1}$ {{crlb_0.1C+1C_U0_1}} mV). (iii) The electrolyte conductivity,
transference number and solid conductivity act mainly through an ohmic-like drop collinear with $Z_\mathrm{CC}$
(correlation {{corr_4rates_kappa0_Z_CC}} for the conductivity; even with four rates their bounds are
{{crlb_4rates_kappa0}} %, {{crlb_4rates_t_plus}} % and {{crlb_4rates_kappa_SPAN}} %), and the Li2S kinetics,
solubility product and S2- diffusivity move the voltage by at most {{sens_max_Li2S_mV_per_efold}} mV per e-fold
change; all of them are fixed at their literature values.

## 6.3 Parameter estimation from synthetic data

Synthetic discharges at 0.1 C and 1 C (FV benchmark, 1 mV white noise, one point every 300 s and 30 s) are
inverted for the eight identifiable parameters from a deliberately poor initial guess ($k_0$ x 2 / 0.5 / 1.5, $b$
x 1.1 / 0.9 / 1.1, $U_{0,1}$ + 20 mV, $Z_\mathrm{CC}$ x 1.5; initial misfit 84 mV rms). With Adam on network weights and
parameters together, the parameters crawl along the correlated directions and are still {{inverse_adam_error_k0_1}} %
off in $k_{0,1}$ after 7000 steps (Fig. 5a). With the Levenberg-Marquardt step on the parameters (Fig. 5b), the first
step reduces the misfit to a few millivolts and all parameters are within 1-2 % after 750 steps. Table 3 compares the
final estimates with the ideal estimator - nonlinear FV least squares on exactly the same data and noise
({{inverse_fv_solves_per_rate}} model solves per rate) - and with the Cramer-Rao bounds: the PINN differs from the
ideal estimator by at most {{inverse_lm_vs_ideal_max_crlb}} bounds (median {{inverse_lm_vs_ideal_median_crlb}}).
This difference is not random: the PINN voltage at its final parameters deviates from the FV voltage at the same
parameters by {{inverse_pinn_forward_error_rms_mV_0.1C}} / {{inverse_pinn_forward_error_rms_mV_1C}} mV rms
(0.1 / 1 C), and propagating this forward error through the FV Jacobian, $-(J^\top J)^{-1} J^\top e$, predicts the
PINN-minus-ideal differences parameter by parameter (e.g. $k_{0,2}$: predicted
{{inverse_error_model_predicted_k0_2}} %, actual {{inverse_error_model_actual_k0_2}} %; $U_{0,1}$: predicted
{{inverse_error_model_predicted_U0_1}} mV, actual {{inverse_error_model_actual_U0_1}} mV). The PINN inverse is thus
the ideal estimator plus the bias of its own forward error, which is dominated by the 1 C electrolyte fields.

## 6.4 Measured data

**Simanjuntak et al. cell.** The experimental points of Fig. 4b of Ref. [1] (12-22 per rate, digitized) are
fitted at 0.1 C and 1 C with the eight parameters of Section 6.3, and the 0.05 C and 0.2 C curves are predicted
(Fig. 6a, Table 5). The FV least-squares fit reaches {{measured_fig4b_FV_rms_mV_0.1C}} / {{measured_fig4b_FV_rms_mV_1C}}
mV rms at 0.1 / 1 C and predicts the other two rates to {{measured_fig4b_FV_rms_mV_0.05C}} /
{{measured_fig4b_FV_rms_mV_0.2C}} mV (nominal model: 31 / 61 mV). The residual is a model-form error, not noise: it
is flat in several parameter directions (different starts end at different parameters with the same misfit), so only
the contact resistance is pinned ({{measured_fig4b_Z_CC_fit_Ohm_m2}} Ohm m2; the authors' own best value is 0.035).
The PINN inverse - fields trained by the physics only, parameter steps accepted only when the re-equilibrated misfit
decreases - lands on the same valley (FV model at the PINN's parameters: {{measured_fig4b_PINN_rms_mV_0.1C}} /
{{measured_fig4b_PINN_rms_mV_1C}} mV; {{measured_fig4b_Z_CC_pinn_Ohm_m2}} Ohm m2). With model-form error, letting
the data act on the fields (as for synthetic data) bends them away from the physics: the PINN then reports a smaller
misfit that the FV model at its parameters does not reproduce.

**SPAN500.** The C/10 discharges of a SPAN cathode synthesized at 500 C [8] are a single-rate data set of a different
cell (ether electrolyte, coin cell, unknown sulfur content). Keeping the geometry and transport of Ref. [1] (they
contribute < 1 mV at this current, Table S1), taking the series resistance from the high-frequency intercept of the
reported impedance spectrum and fitting the effective sulfur fraction with the open-circuit parameters, the third
discharge is reproduced to {{span500_paper_rms_mV}} mV rms (Fig. 6b). This is well below the change between
consecutive cycles of the same cell ({{span500_cycle_to_cycle_rms_mV}} mV rms between cycles 2 and 3), so the
calibrated parameters describe one cycle, not the material. The effective sulfur fraction,
{{span500_paper_w_S}} g S per g SPAN at the model's 1.5 electrons per sulfur, is close to the 40-50 wt% reported for
typical SPAN. A single rate does not determine the kinetics: with the rate constants of all three steps raised a
thousandfold the fit is as good ({{span500_fast_rms_mV}} mV), but the two calibrations predict 1 C curves that differ
by up to {{span500_1C_prediction_max_diff_mV}} mV in the first plateau (for the irreversible steps 2 and 3 a change of $k_0$ is exactly a shift of
$U_0$, so only step 1 distinguishes them). Determining the kinetics needs at least a second rate, consistent with
Section 6.2. The PINN inverse on the same data (fields by the physics only, w_S and current from the FV
fit, data up to {{span500_pinn_window_q_max}} mAh/g so that the window excludes the cut-off collapse) stops at
parameters where the FV model misses the data by {{span500_pinn_fv_rms_mV}} mV, against
{{span500_fv_window_fit_rms_mV}} mV for the FV optimum on the same window - again the same flat valley.

## 6.5 Cost

Table 4 lists the costs on one core of the same CPU. An FV discharge takes {{cost_fv_0.1C_40x20_cpu_s}} s (0.1 C)
to {{cost_fv_1C_40x20_cpu_s}} s (1 C); training the PINN with the recipe of Section 4 takes
{{cost_pinn_training_0.1C_cpu_h}} / {{cost_pinn_training_1C_cpu_h}} h of CPU time, about three orders of magnitude
more, after which the trained network returns a discharge curve in {{cost_pinn_eval_voltage_0.1C_ms}} ms - but only
for the parameters and protocol it was trained on. For the eight-parameter inverse the FV least-squares fit needs
{{cost_inverse_fv_cpu_h}} h of CPU time, the PINN route {{cost_inverse_pinn_wall_h}} h on top of
{{cost_inverse_pinn_pretraining_wall_h}} h of forward training. For a one-dimensional cell model that a stiff
integrator solves in seconds, the PINN is therefore not a faster solver, and we do not claim it is. Its use is
methodological: the same differentiable object represents the fields, satisfies the physics and absorbs data, its
inverse is as accurate as the ideal estimator up to the bias of its forward error (Section 6.3), and the
formulation can be extended - not tested here - to settings where mesh-based solvers become expensive
(multi-dimensional electrodes, parametric networks that take the parameters as inputs).

# 7. Conclusions

1. The Li-SPAN model of Ref. [1] reproduces its published curves only with three closures (faradaic current of
   reaction 1, effective rate constants in the Tafel regime, irreversible reaction 3 during discharge); with them a
   finite-volume reference matches the published curves to {{fig6a_rms_mV_0.1C}} / {{fig6a_rms_mV_1C}} mV rms
   (0.1 / 1 C).
2. A PINN with the reaction-extent ordering, the current balance and the kinetics built in, and a global
   charge-inventory residual, reproduces this reference to a fraction of a millivolt from random initialization;
   without the inventory residual, the error of the extents integrates into millivolt errors in the last plateau.
3. One discharge rate does not identify the kinetics separately from the contact resistance and the open-circuit
   parameters; two rates (0.1 C and 1 C) do, for eight parameters.
4. The multi-rate inverse PINN, with a Levenberg-Marquardt step on the physical parameters, reaches the accuracy of
   an ideal least-squares estimator up to a bias that its own forward error predicts; improving the forward model
   improves the inverse directly.
5. On measured data the misfit (10-20 mV) is a model-form error, larger than any numerical error discussed here and,
   for SPAN500, smaller than the variation between consecutive cycles; multi-rate data are needed to calibrate the
   kinetics.
6. For this one-dimensional model the PINN is about three orders of magnitude more expensive than a stiff
   finite-volume solver; its value lies in the uniform treatment of forward and inverse problems, not in speed.

# References

1. E. K. Simanjuntak, T. Danner, P. Wang, M. R. Buchmeiser, A. Latz, Electrochim. Acta 497, 144571 (2024).
2. M. Raissi, P. Perdikaris, G. E. Karniadakis, J. Comput. Phys. 378, 686 (2019).
3. *(PINN review - to be chosen)*
4. *(PINN-DFN references - to be completed from the project's literature list)*
8. *(Wang et al., Nat. Mater. 25, 791 (2026) - SPAN500 data)*
