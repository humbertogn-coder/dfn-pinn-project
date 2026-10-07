# Li-SPAN continuum model: formulation for a numerical reference and a PINN

Status: equations complete (2026-10-05); numerical reference implemented and validated against Figs 5a, 6a, 6b (2026-10-06, section 5).
Source: E.K. Simanjuntak, T. Danner, P. Wang, M.R. Buchmeiser, A. Latz, "A
novel modeling approach for sulfurized polyacrylonitrile (SPAN) electrodes in
Li metal batteries", Electrochimica Acta 497 (2024) 144571,
https://doi.org/10.1016/j.electacta.2024.144571 (open access, CC BY 4.0), main
text + Supplementary Material (SI, 17 pp., equations S1-S42, Table S1, Figs
S1-S5). Equation numbers below: (n) = main text, (Sn) = SI. Items still
marked **[open]** are the only ones not fixed by paper + SI.

## 1. What changes with respect to the DFN

| Aspect | Li-ion DFN (Chen2020) | Li-SPAN (Simanjuntak 2024) |
| --- | --- | --- |
| Negative electrode | porous graphite, particle diffusion | Li metal foil, not resolved: planar plating/stripping reaction at y = L_tot (S7-S10), U_eq,0 = 0 V |
| Positive electrode | NMC particles with radial diffusion | SPAN: sulfur covalently bound to PAN; **no solid-state diffusion**, 4 local species c_S4, c_S3, c_S2, c_S1 (ODEs in time at each y) |
| Reactions | 1 intercalation reaction per electrode | 3 sequential electrochemical SPAN reactions (1)-(3) + Li2S precipitation (9), chemical, on cathode surfaces |
| Electrolyte species | binary salt LiPF6 (concentrated-solution theory) | Li+, PF6-, S2- with Nernst-Planck fluxes (S17-S19) and CST-equivalent D_Li+, D_PF6- (S20-S21); S2- trace species |
| Solid phases | fixed | Li2S volume fraction grows (S39); porosity (S42) and active areas (S40-S41) change |
| Extra physics | none in the benchmark | double layer current i_DL = a_SPAN c_DL d(Delta phi)/dt (S30); contact resistance Z_CC (S38) |
| Coordinates | (x, r, t) | (y, t) only |

The PINN therefore loses the radial dimension (cheaper) but gains several
local state variables, multi-reaction kinetics, a trace species spanning nine
orders of magnitude and a precipitation threshold.

## 2. Geometry and parameters (main text Tables 1-3, SI Table S1)

Two cells appear in the paper. The **validated cell** (main text, Figs 4-6,
compared with experiments) is the one to reproduce first:

* y = 0 cathode current collector, y = L_cat = 100 um cathode/separator
  interface, y = L_tot = L_cat + L_sep, L_sep = 618 um (two Whatman GF
  separators, 2 x 309 um), Li foil surface at y = L_tot (anode not resolved).
* Cathode: carbon+binder eps_CB = 0.032 (rho 1810 kg/m3), SPAN eps_SPAN =
  0.0945 (rho 1440 kg/m3), Li2S eps_Li2S,0 = 1e-5 (rho 1659 kg/m3), porosity
  0.87; a_SPAN,0 = 1e7 1/m; surface exponent xi = 1.5; Bruggeman beta_cat = 1.5.
* Separator: glass fibre eps_sep = 0.11, porosity 0.89, Bruggeman beta_sep = 1.
* Electrolyte 1 M LiPF6 in EC:DEC: D_LiPF6 = 2.52e-10 m2/s, D_S2- = 4.7e-10
  m2/s (constant), t+ = 0.1625, kappa = 0.796 S/m, thermodynamic factor
  (1 + dln f/dln c) = 1.6. The paper uses the concentration-dependent Lundgren
  2014 correlations for D_LiPF6(c), kappa(c) **[open: functional forms not in
  the SI; use the 1 M values as constants first, the salt gradients at <= 1C
  (1 mA/cm2) are small]**.
* Initial: c_Li+ = 1000.02, c_PF6- = 1000, c_S2- = 0.01 mol/m3; SPAN species
  c_S4 = 598, c_S3 = c_S2 = c_S1 = 1e-5 mol/m3 (Table 2); reference
  concentrations 598, 598, 598, 1196 mol/m3 for S4, S3Li, S2Li, S1Li
  (volume basis: cathode volume, consistent with eps_SPAN rho_SPAN and the
  capacity 47.45 Ah/m2 of Table S1 for the designed cell).
* SPAN conductivity kappa_SPAN = 1 S/m (kappa_eff = kappa_SPAN eps_SPAN^beta,
  S34); double layer c_DL = 0.1 F/m2; Z_CC = 0.025 Ohm m2 (Table 2; main
  text quotes best agreement at 0.035 Ohm m2, Fig. 6).
* 1/10 C = 0.1 mA/cm2 (1 C = 10 A/m2); voltage window 1.0-3.0 V. Temperature
  **[open: not stated; assume 298.15 K]**.

The **designed cell** of SI Table S1 (eps_SPAN 0.6, L_sep 20 um, L_an 25 um,
526 Wh/kg) is only used for the energy-density projections (S43-S62) and is
not needed for the PINN.

## 3. Reaction network (discharge direction)

    (1) 1/2 PAN-S4-PAN + e- + Li+  ->  1/2 PAN-S3Li + 1/2 PAN-S1Li
    (2) 1/2 PAN-S3Li  + e-         ->  1/2 PAN-S2Li + 1/2 S2-
    (3) 1/2 PAN-S2Li  + e-         ->  1/2 PAN-S1Li + 1/2 S2-
    (9) 2 Li+ + S2-               <=>  Li2S(s)            (chemical, cathode surfaces only)
    (S7) Li+ + e-                 <=>  Li(s)              (anode surface, planar)

n = 1 electron per reaction step (1)-(3). Six electrons per PAN-S4-PAN chain
(1.5 e- per S, 75 % of the 1672 mAh/gS of elemental sulfur), consistent with
the ~1250 mAh/gS capacity in the paper and with the SI statement that chains
of length 8 to 2 are reducible (C_SPAN,theo = 6/8 C_S8 w_SPAN, S47).

### Rate laws (S1-S6, main text (4)-(8), (10)-(14))

Generalized Butler-Volmer form for every reaction m (S1):

    r_m = k0_m a_ed^(1-alpha_m) a_prod^alpha_m [exp(-alpha_m dmu_m/RT) - exp((1-alpha_m) dmu_m/RT)]
    a_ed = prod_i a_i^|nu_i|  (educts),  a_prod = prod_i a_i^|nu_i|  (products)   (S2-S3)
    a_i = c_i / c_i^0  (ideal solution; solids 1; SPAN species a = gamma c~)      (S4)
    dmu_m = sum_i nu_i mu_i,  mu_i = mu_i^0 + RT ln a_i + z_i F phi_k             (S5-S6)

SPAN reactions (main text):

    dmu_Sx = F (phi_elode - phi_elyte - U_Sx^eq)
    U_Sx^eq = (1/F) [dmu0_Sx + RT ln gamma_Sx(c~_Sx) + RT ln(c~_ed / c~_prod)]      (7)
    U_Sx^eq,ref = U_Sx^eq,0 - b_Sx zeta_Sx,   zeta_Sx = 1 - c~_Sx                    (8)

i.e. the standard + activity-coefficient part is a straight line in the local
reaction coordinate zeta (fitted to the measured OCV, Table 3), and the
ln(c~_ed/c~_prod) term of (7) remains. Parameters (Table 3): k0 = 1e-2,
1e-2, 1e-4 mol/m2/s; U^eq,0 = 2.2, 1.9, 1.66 V; b = 0.3, 0.28, 0.62 V;
alpha = 0.5.

Li2S (S11-S16, (9)-(10)): r_Li2S = k0_Li2S (a_Li+ a_S2-)^alpha [exp(-alpha dmu/RT) - exp((1-alpha) dmu/RT)],
dmu_Li2S = RT ln K_sp - RT ln(a_Li+ a_S2-), K_sp = exp(dmu0/RT) = 10, k0 = 2e2
mol/m2/s. (Both main text (10) and SI (12), (16) use a_Li+ to the first
power although the stoichiometry is 2 Li+; a_Li+ ~ 1 so the difference is
numerically small. Keep the paper's form to reproduce its figures.)

Anode (S7-S10): r_Li = k0_Li a_Li+^(1-alpha) [...], dmu_Li = F (phi_elode -
phi_elyte - U_Li^eq), U_Li^eq = 0 + (RT/F) ln a_Li+, k0 = 3.94 mol/m2/s,
alpha = 0.5, phi_elode(anode) = 0 is the potential reference (S37). No
polysulfide reactions at the anode (carbonate electrolyte, S1.2).

## 4. Governing equations (SI S1.3-S1.4, now confirmed)

Unknowns on the cathode y in [0, L_cat]: c_S4, c_S3, c_S2, c_S1, eps_Li2S,
c_Li+, c_PF6-, c_S2-, phi_elode, phi_elyte. On the separator [L_cat, L_tot]:
c_Li+, c_PF6-, c_S2-, phi_elyte.

Local SPAN balances (no transport; a = a_SPAN(y, t), rates r_m per unit SPAN area):

    d c_S4/dt = -1/2 a r1
    d c_S3/dt = +1/2 a r1 - 1/2 a r2
    d c_S2/dt = +1/2 a r2 - 1/2 a r3
    d c_S1/dt = +1/2 a r1 + 1/2 a r3

Two invariants follow (sulfur atoms, chain ends): e.g. 4 c_S4 + 3 c_S3 + 2 c_S2
+ c_S1 + (S2- released) = const; useful as hard constraints in the PINN.

Li2S and volume fractions (S39-S42):

    d eps_Li2S/dt = (MW_Li2S / rho_Li2S) a_Li2S r_Li2S
    a_Li2S = a_SPAN eps_Li2S                                   (empirical, S40)
    a_SPAN = a_SPAN,0 (eps_elyte / eps_elyte,0)^xi              (S41)
    eps_elyte = 1 - eps_SPAN - eps_Li2S - eps_CB                (S42)

Electrolyte mass balances, dilute-solution Nernst-Planck form (S17-S19):

    d(eps_elyte c_i)/dt = -dN_i/dy + s_i^chem + s_i^echem,   i = Li+, PF6-, S2-
    N_i = -D_i^eff dc_i/dy - D_i^eff c_i (z_i F / RT) dphi_elyte/dy
    D_i^eff = D_i^0 eps_elyte^beta   (beta_cat = 1.5, beta_sep = 1)

with the CST-equivalent coefficients for the salt ions (S20-S21):

    D_Li+^0  = D_LiPF6 + kappa0 RT (t+ - 1) t+ / (F^2 c_Li+) (1 + dln f/dln c)
    D_PF6-^0 = D_LiPF6 + kappa0 RT (t+ - 1)(1 - t+) / (F^2 c_Li+) (1 + dln f/dln c)

(as printed in the SI; check the dimensional consistency when implementing:
kappa0 RT/(F^2 c) has units of m2/s). D_S2- constant. Sources (S22-S24):

    s_S2-^echem = +1/2 a (r2 + r3)      (S22 in the SI's chain-length notation)
    s_S2-^chem  = -a_Li2S r_Li2S
    s_Li+^chem  = -2 a_Li2S r_Li2S
    s_Li+^echem = -a r1  (reaction 1 consumes one Li+ per electron)

Boundary conditions (S25-S26): y = 0, N_i = 0 for all species; y = L_tot,
N = 0 for S2- and PF6-, N_Li+ = -r_Li (plating/stripping).

Electrolyte charge (S27-S30):

    0 = -di_elyte/dy + i_F + i_DL
    i_elyte = sum_i z_i F N_i
    i_F  = -2 F s_S2-^echem                                     (S29)
    i_DL = a_SPAN c_DL d(phi_elode - phi_elyte)/dt              (S30)

**Settled (section 5.1):** S29 ties the faradaic current to S2- production
only (reactions 2 and 3), but reaction (1) also transfers one electron and
consumes one Li+ per step while releasing no S2-; phase 1 carries one third
of the capacity in Fig. 5a, so the stoichiometrically consistent form
i_F = F a (r1 + r2 + r3) (with the Li+ sink -a r1 in the cathode) is the one
implemented.

Solid charge (S31-S37):

    di_elode/dy + di_elyte/dy = 0,   i_elode = -kappa_eff dphi_elode/dy  (sign per S33/S36)
    kappa_eff = kappa_SPAN eps_SPAN^beta_cat
    y = L_cat:  dphi_elode/dy = 0;   y = 0:  -kappa_eff dphi_elode/dy = I;   phi_elode(L_tot) = 0

Cell voltage (S38): E_cell = phi_elode(0) - phi_elode(L_tot) + Z_CC I, with the
SI's sign convention for I (discharge current negative in their convention
so that Z_CC lowers the voltage; confirm against Fig. 6, where larger Z_CC
gives lower voltage at high rate).

Initial conditions: SPAN species at Table 2 values, c_Li+ = 1000.02, c_PF6- =
1000, c_S2- = 0.01, eps_Li2S = 1e-5, potentials at equilibrium (consistent
with the double layer; the double layer removes the algebraic t = 0 corner).

## 5. Numerical reference (implemented, 2026-10-06)

`src/dfn_pinn/lispan/` (params.py, model.py), `scripts/lispan_discharge.py`,
`scripts/lispan_digitize_paper.py`, `scripts/lispan_fit_paper.py`,
`tests/test_lispan_reference.py` (7 tests). PyBaMM has no Li-SPAN model, so
the reference is our own 1-D finite-volume method-of-lines code (scipy BDF
with a sparse finite-difference Jacobian; 20 cathode + 10 separator volumes
as in the paper; a 0.1 C discharge takes 4 s, 1 C 6 s).

### 5.1 Formulation as implemented (conventions of model.py)

* y = 0 collector, y = L_cat = 100 um cathode/separator interface, y = L_tot
  = 718 um Li surface (phi_s(Li) = 0). Discharge current I > 0 [A/m2]; the
  ionic current i_e(y) (in +y) is 0 at y = 0 and -I in the separator,
  i_s + i_e = -I in the cathode.
* States per cathode volume: ln c_S4, ln c_S3, ln c_S2, ln c_S1 (SPAN
  species), ln eps_Li2S, Delta phi = phi_s - phi_e (double layer); per volume
  of the whole cell: c_PF6- and ln c_S2-. c_Li+ = c_PF6- + 2 c_S2-
  (electroneutrality). Log states because the species span many decades and
  the sqrt(activity) factors of (S1) make the plain form singularly stiff near
  zero; floors (1e-6 mol/m3 SPAN species, 1e-14 mol/m3 S2-) soften the log
  dynamics below physically meaningful levels.
* Rates (mass-action form of (S1)/(4)-(8), reduction positive):
  r_m = k0_m [a_ed,m e^(-x_m) - a_prod,m e^(+x_m)], x_m = F(Delta phi -
  U0_m + b_m zeta_m)/(2RT), zeta_1 = 1 - c_S4/598, zeta_2 = 1 - c_S3/598,
  zeta_3 = 1 - c_S2/598; a_ed = (a_S4^1/2 a_Li, a_S3^1/2, a_S2^1/2),
  a_prod = ((a_S3 a_S1)^1/2, (a_S2 a_S)^1/2, (a_S1 a_S)^1/2), a_S =
  c_S2-/c_S,ref. Species: dc_S4/dt = -a r1/2, dc_S3/dt = a(r1 - r2)/2,
  dc_S2/dt = a(r2 - r3)/2, dc_S1/dt = a(r1 + r3)/2 (1 S4 chain -> 1 S3Li +
  1 SLi, so c_S1 reaches 598 after phase 1 and 1196 at the end, as in Fig.
  5a). Faradaic current i_F = F a (r1 + r2 + r3): phase 1 carries one third
  of the capacity in Fig. 5a, so S29 (S2- production only) cannot be the
  current definition - open item 2 settled.
* Li2S: r_L = (k0_L / K_sp^1/2)(a_Li a_S^L - K_sp) (= (S12),(16) with alpha =
  1/2), a_S^L = c_S2- K_sp / c_sat, a_L = a_SPAN (eps_Li2S + eps_seed) for
  precipitation and a_SPAN eps_Li2S for dissolution (a permanent nucleation
  seed eps_seed = 1e-5, the paper's initial value; without it the seed
  dissolves within microseconds at the start of discharge and Li2S could
  never nucleate), d eps_Li2S/dt = (M/rho) a_L r_L, a_SPAN = a0
  (eps_e/eps_e0)^xi, eps_e = 1 - eps_SPAN - eps_CB - eps_Li2S.
* Electrolyte: Nernst-Planck fluxes with the dilute ion diffusivities D+ =
  D_salt/(2(1 - t+)), D- = D_salt/(2 t+) (salt diffusion coefficient and t+
  exact) and the *measured* conductivity kappa0 (c/c0) eps^beta for the
  migration current (the diffusion-potential coefficient then agrees with
  concentrated-solution theory within 10 %; the SI's (S20)-(S21) as printed
  give t+ = 0.94 and were not used). Zero fluxes at y = 0; at the Li
  surface N_PF6- = N_S2- = 0 and N_Li+ = -I/F.
* Charge: a c_DL d(Delta phi)/dt = di_e/dy + F a sum r_m; i_e at interior
  cathode faces follows algebraically from Delta phi' = (I + i_e)/kappa_s +
  (i_e + B)/kappa_e (B = F sum z_i D_i dc_i/dy), so no linear solve is needed.
  phi_e(L_tot) = -(RT/F) ln a_Li - (2RT/F) asinh(I/(2 F k0_Li a_Li^1/2)),
  E = Delta phi(0) + phi_e(0) - I dy/(2 kappa_s) - Z_CC I (Z_CC lowers the
  voltage, as Fig. 6b requires).
* Initial state: Table 2 concentrations; Delta phi from r1 = 0 (2.67 V, the
  initial spike of Figs 4a/6); S2- equilibrated with the reverse of (2) (the
  paper's 0.01 mol/m3 would be oxidised within microseconds); current ramp
  I tanh(t/1 s). Stop at E = 1.0 V. Specific capacity in mAh/g_S uses the
  sulfur implied by the SPAN concentrations (4 x 598 mol/m3 x 32.07 g/mol x
  L_cat = 0.767 mg/cm2, theoretical 1254 mAh/g_S = 6 e- per chain), which is
  what the paper's axes use (1 C = 1 mA/cm2 = 0.96 mAh/cm2), not the 0.6
  mg/cm2 quoted in its text.

### 5.2 What paper + SI do not fix, and how it was settled

Reproducing Figs 5a, 6a and 6b exposed three places where the literal
parameter values cannot have produced the published curves:

1. **Kinetic regime.** With Table 3 (k0 = 1e-2, 1e-2, 1e-4 mol/m2/s) and
   a_SPAN = 1e7 1/m the exchange rates exceed the demand (1e-8 mol/m2/s at
   0.1 C) by six orders of magnitude and the SPAN reactions sit at
   equilibrium: no rate dependence beyond ohmic/concentration effects, and
   an electrode-mediated comproportionation S3 + S1 -> 2 S2 that puts
   PAN-S2Li in the cathode from the first mAh/g on. The paper's Fig. 6a shows
   instead a Tafel slope of 0.118 V per decade of k0 and 0.17 V between 0.1 C
   and 1 C at Z_CC = 0: its simulations run in the Tafel regime, i.e. with
   exchange rates comparable to the demand. Only the products a_SPAN k0 enter,
   so we keep a_SPAN and fit three *effective* k0 to the digitized Fig. 6a
   curves (0.1 C and 1 C, Z_CC = 0): k0_eff = (2.95e-8, 2.39e-8, 2.34e-8)
   mol/m2/s (`results/lispan/fit_k0.json`, rms 18 mV), i.e. about 1e-6 times
   Table 3 for reactions 1-2 and 1e-4 times for reaction 3 - the three
   effective prefactors are nearly equal, unlike Table 3's 100:100:1.
2. **Reversibility of reaction (3).** With the mass-action reverse term the
   oxidation S2- + PAN-SLi -> PAN-S2Li + e- runs strongly during phases 1-2
   (Delta phi is 0.5-0.9 V above U_3), consuming PAN-SLi and keeping S2- at
   1e-16 mol/m3; Fig. 5a shows c_S1 flat at 598 mol/m3 until reaction 3
   starts and Fig. 7e shows S2- rising exponentially in equilibrium with
   reaction (2) to the solubility limit at ~350 mAh/g. Reaction (3) is
   therefore treated as irreversible in discharge (parameter `reversible =
   (True, True, False)`); its reverse is the charging reaction, outside the
   paper's scope.
3. **S2- references.** Fig. 7f shows the saturation concentration scaling with
   K_sp and equal to 1e-5 mol/m3 for K_sp = 10 (also the plateau of Figs 5b
   and 7e), which is not K_sp x 0.01 mol/m3; the model OCV break points of
   Fig. 3 (1.97 V, 1.73 V) on the other hand need the Nernst shift of a S2-
   activity referred to 0.01 mol/m3 (+0.09 V on U_2, U_3). We use c_S,ref =
   0.01 mol/m3 (Table 2, "initial condition" as the text says) in the SPAN
   kinetics and c_sat = 1e-5 mol/m3 at K_sp = 10 in the Li2S driving force.

Also found: Fig. 4b is consistent with Z_CC = 0.025 Ohm m2 (Table 2), not
with the 0.035 quoted as best fit in the text; and the SI's (S20)-(S21)
swap the roles of t+ and 1 - t+.

### 5.3 Validation against the paper (results/lispan/)

| Quantity | this reference | paper (digitized) |
| --- | --- | --- |
| 0.1 C, Z_CC = 0: voltage vs Fig. 6a | rms 16 mV, max 32 mV | - |
| 1 C, Z_CC = 0: voltage vs Fig. 6a | rms 18 mV, max 39 mV | - |
| 0.1 C / 1 C capacity to 1.0 V, Z_CC = 0 | 1248 / 1177 mAh/g_S | ~1230 / 1236 |
| 0.05 / 0.1 / 0.2 / 1 C capacity, Z_CC = 0.025 | 1254 / 1240 / 1212 / 1025 | 1252 / 1214 / 1229 / 994 (Fig. 4b) |
| Species vs Fig. 5a (0.1 C) | S3 peak 490 at 400, S2 peak 540 at 830, c_S1 598 -> 1196, Li2S onset ~350, eps_Li2S end 0.032 | S3 520 at 400, S2 533 at 800, same plateau, onset ~350, 0.028-0.030 |
| 1 C - 0.1 C at 300 mAh/g, Z_CC = 0 | 0.17 V | 0.17 V |
| Z_CC effect at 1 C | 0.25 V per 0.025 Ohm m2 | same (Fig. 6b) |
| k0 sensitivity, phase 1 | 0.09 V/decade | 0.12 V/decade |

Figures: `results/lispan/ref_Zcc0/discharge_vs_paper.png` (Fig. 6a overlay +
species), `results/lispan/ref_Zcc0.025/discharge_vs_paper.png` (Fig. 4b
overlay; the digitized 0.05-0.2 C points of Fig. 4b overlap and are noisy),
`fields_0.1C.png` (profiles of c_Li+, c_S2-, phi_e, Delta phi, c_S2Li,
eps_Li2S). Grid convergence 10/5 vs 30/15 volumes: < 5 mV rms. Sulfur and
electron balances close to 2e-3 and 5e-3 (tests).

The reference is good enough for its purpose (ground truth for the forward
Li-SPAN PINN and synthetic data for the inverse problem): the equations are
the paper's, the parameters that the paper does not pin down are documented
above and exposed as explicit parameters, and the owner can ask T. Danner
(DLR) for the MATLAB implementation to replace the three effective k0 and the
two S2- references by the authors' values if an exact reproduction is ever
needed.

## 6. Li-SPAN forward PINN (implemented 2026-10-06, `src/dfn_pinn/lispan/pinn.py`)

Scripts: `scripts/lispan_train.py` (config `configs/lispan_forward_01C.json`,
`--resume`, `--init`), `scripts/lispan_plot_run.py`. Same philosophy as the
v2 DFN PINN: everything the physics fixes exactly is built into the
parametrization, the rest is a residual with adaptive group weights.

### 6.1 Fields and hard structure

* Inputs (y, t) -> (Y, T) with the v2 time features (2T-1, 2g-1, Fourier
  modes, short-time exponentials), g = tanh(t/tau_ramp).
* Reaction extents. Three networks outputs P_m = exp(net_m) > 0 define the
  remaining educt fractions e_m = exp(-T P_m) and the extents xi_1 = 1 - e_1,
  xi_2 = xi_1 (1 - e_2), xi_3 = xi_2 (1 - e_3), so that 0 <= xi_3 <= xi_2 <=
  xi_1 < 1 and xi(0) = 0 are exact. Species: c_S4 = c_S4,0 e_1, c_S3 = c_S4,0
  xi_1 e_2, c_S2 = c_S4,0 xi_2 e_3, c_S1 = c_S4,0 (xi_1 + xi_3) (one PAN-S4-PAN
  chain gives one S3Li and one SLi), eps_Li2S = eps_L0 + V_m c_S4,0 (xi_2 +
  xi_3): sulfur conservation, positivity, initial conditions and the Li2S
  inventory are exact. The dissolved S2- (< 1e-8 of the sulfide) is
  neglected and its activity in the kinetics is fixed at saturation; the
  finite-volume reference run the same way differs from the full model by
  0.03 mV in voltage and 0.04 mol/m3 in species (section 5).
* Total reduction rate R(y,t) = (I/F) rho / int_0^L a rho dy with rho =
  exp(net) > 0: the electrode integral of the faradaic current equals the
  applied current exactly (hard projection, Gauss-Legendre in y at every
  sample), and the ionic current i_e(y,t) = -I int_0^y a rho / int_0^L a rho
  follows.
* Kinetics exact: Delta phi(y,t) is the root of sum_m f_m(Delta phi) = R
  (bracketed bisection + Newton, gradients by the implicit function theorem
  with the slope floored at -F R/(2RT) so that the sensitivity stays at the
  Tafel scale where the educts are exhausted during training). The
  individual rates f_m(Delta phi) then split R among the three reactions.
* Electrolyte: salt c_e = c_0 (1 + s tanh(net) (1 - e^{-t/tau_ic})) (initial
  condition exact, bounded deviation so that kappa_e never collapses), phi_e
  = phi_e,anode(t) + (1 - Y) net (anode kinetics built in).
* Voltage E = Delta phi(0,t) + phi_e(0,t) - Z_CC I(t).

### 6.2 Residuals (dimensionless, adaptive group weights as in v2)

1. extents: d xi_m/dT - t_end a f_m/(2 c_S4,0), m = 1, 2, 3 (the ODEs tie the
   extents to the kinetic split; the sum is exact by the projection);
2. solid Ohm / kinetics consistency in the cathode: d(Delta phi)/dy = (I +
   i_e)/kappa_s + (i_e + B)/kappa_e with B = F (D+ - D-) eps^beta dc/dy;
3. electrolyte Ohm in both regions: d phi_e/dy = -(i_e + B)/kappa_e;
4. salt balance in both regions (binary electrolyte, transference form):
   d(eps c)/dt = d/dy(D_eff dc/dy) - (1 - t+) a R (cathode; the Li+ sink is
   one Li+ per electron once the sulfide precipitates) and the three flux
   conditions (zero at the collector, (1 - t+) I/F at the Li surface,
   continuity at the interface).

Residuals 2 and 3 use a Huber loss (quadratic below 3, linear above): the
Tafel relation turns any spatial roughness of the species into very large
d(Delta phi)/dy early in training (residuals of 1e4 were seen), which with a
plain square dominated the gradient and stalled the run. No double layer
(time constant < 1 s); the reference for the evaluation is run with c_DL ->
0 and the same current ramp.

### 6.3 What had to be changed to make it trainable (negative results)

* **Reaction (2) reversible = stiff.** With the mass-action reverse term the
  S2Li concentration at high Delta phi is slaved to an equilibrium c_S2 ~
  exp(-2x_2) that spans ten decades; a representation error of 1e-5 in
  c_S2/c_ref produces a reverse rate 20x the demand. The finite-volume
  reference handles this with log states; a PINN cannot. Reaction (2) is
  therefore treated as irreversible like reaction (3) in the PINN benchmark
  (`reversible = (True, False, False)`): the reference changes by 2.6 mV rms
  / 5.6 mV max and 10 mol/m3 at 0.1 C, 0.1 mV at 1 C. Reaction (1) keeps its
  reverse term (its operating point is within 1-2 kT of equilibrium in phase
  1, no stiffness).
* **float32 cancellation.** Computing c_S4 = c_S4,0 (1 - xi_1) with xi_1 =
  1 - e^{-T P} loses the educt entirely once T P > 17 (1 - e^{-17} rounds to
  1) and makes the forward Tafel term of reaction (1) jump; the species are
  computed from the remaining fractions e_m directly. Before this fix the ODE
  residuals plateaued at 0.1-0.6 and the voltage error at 150 mV; after it
  the residuals fell by two orders of magnitude within 500 steps.
* **Implicit-root sensitivity.** Where the extents overshoot during training
  (educts exhausted before the end), sum f_m < R for every Delta phi, the
  root sits at the bracket edge and the implicit derivative diverges; the
  slope floor and a clamped (straight-through) Newton correction fix it.

### 6.4 Status

Run A (`runA_floor1e-12_lispan_f01C_20261006T184521Z`, 0.1 C, width 64,
fourier_t 4, activity floor 1e-12 mol/m3): plateau at 24 mV rms / 79 mV
max from step 1000 on. Cause: the floor let the exhausted educt of reaction
1 keep 1.4 % of the current after phase 1 through the e^{-x} = 130-1e5 Tafel
amplification at low Delta phi, which the extents (xi_1 <= 1) could not
follow; the ODE residual sat at a uniform -0.04.

Run `lispan_f01C_20261006T190806Z` (same architecture, floor 1e-30 in
activity units, exponent cap 60; 10 000 Adam steps, 45 min on one core):
voltage **1.18 mV rms / 3.9 mV max** against the finite-volume reference
(points t > 100 s, first 98 % of the discharge), Delta phi(0) 1.2 mV rms,
species 0.09 / 0.61 / 1.03 / 1.31 mol/m3 rms (S4 / S3 / S2 / S1, 0.1-0.2 %
of 598), eps_Li2S 5e-5, phi_e 0.025 mV, c_e 1.17 mol/m3 rms (a 0.1 %
inventory offset: the salt balance is local, so the global inventory
drifts; run B adds the global inventory residual of v2). The
S3 -> S2 -> S1 hand-overs at 450 and 900 mAh/g are resolved. Figures:
`results/lispan_runs/<run>/plot_final.png`.

Run B (`lispan_f01C_w96_20261006T194300Z`, `configs/lispan_forward_01C_w96.json`:
width 96, fourier_t 8, 20 000 steps, 768 cathode points, global salt
residual; 1 h 45 min on one core): final **1.0 mV rms / 3.4 mV max**,
species 0.1 / 0.5 / 0.8 / 1.0 mol/m3, eps_Li2S 4e-5, c_e 0.16 mol/m3 (the
global inventory residual removed the offset), phi_e 0.03 mV. The
checkpoints along the run evaluate between 0.39 / 1.4 mV (step 5000) and
1.4 / 4.5 mV (step 10 000) although every residual keeps falling (ODE
terms 2e-5, Ohm 5e-6 at the end): the solution wanders along a direction
the residuals constrain only weakly. Tracing the error (same root solve
applied to the finite-volume species and rates reproduces the reference
Delta phi to 1e-4 V): in phase 3 a 0.5 % deficit of c_S2 (1.7 mol/m3)
costs 1.9 mV through the OCV slope b_3 = 0.62 V (1 mV per mol/m3 of S2Li),
so the late-time voltage is an integrated-extent accuracy problem, not a
kinetics one. Remedies for the next run, in order: a fixed extra weight on
the extent ODEs (the adaptive scheme equalizes gradient norms, not
importance for the voltage), parameter averaging over the last checkpoints
(EMA), and the v2 L-BFGS polish. Figures:
`results/lispan_runs/<run>/plot_final.png`.

### 6.5 Inverse problem (planned)

Identify a small, identifiable subset of {U0_m, b_m, k0_m, Z_CC, D_salt,
kappa0} (all exposed as log-multipliers / offsets in `LiSPANPINN.PARAMETERS`)
from discharge curves at several C-rates; sensitivities first (as in
V2_RESULTS.md 3.0). The paper itself reports low sensitivity to k0 in the
Tafel regime for the voltage *shape* but 0.118 V per decade in level, and
strong sensitivity to beta and eps_SPAN at 1 C (Figs S3-S4).

## 7. Data for the Li-SPAN work

* Forward PINN: no data; physics only. The numerical reference of section 5
  is used for evaluation.
* Inverse PINN, stage 1: synthetic voltage curves from the numerical
  reference (known parameters, controlled noise), as done for the DFN.
* Inverse PINN, stage 2: the paper's experimental discharge curves at 0.05,
  0.1, 0.2 and 1 C (Fig. 4 / Fig. S1; digitize the circles, or request the
  data from the corresponding author, T. Danner, DLR), then any in-house
  Li-SPAN cells of the Balbuena group.

## 8. Remaining open items

1. Lundgren 2014 correlations D_LiPF6(c), kappa(c): constants at 1 M with
   kappa scaled linearly in c; the salt gradient at 1 C is only 3 % (Fig.
   `fields_0.1C.png`: 25 mol/m3 at 0.1 C), so the correlations matter little.
2. ~~i_F definition~~ settled: stoichiometric sum (section 5.1).
3. Temperature assumed 25 C; sign of I fixed by Fig. 6b (Z_CC lowers E).
4. Experimental data points: figures only, unless the authors share them.
5. Effective kinetics and S2- references (section 5.2): fitted/inferred from
   the figures; ask the authors for the implementation to confirm.
