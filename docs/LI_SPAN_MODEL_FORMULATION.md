# Li-SPAN continuum model: formulation for a numerical reference and a PINN

Status: working draft (2026-10-03). Source: E.K. Simanjuntak, T. Danner, P. Wang,
M.R. Buchmeiser, A. Latz, "A novel modeling approach for sulfurized
polyacrylonitrile (SPAN) electrodes in Li metal batteries", Electrochimica Acta
497 (2024) 144571, https://doi.org/10.1016/j.electacta.2024.144571 (open access,
CC BY 4.0).

The main text gives the reaction network, rate laws and all parameter values,
but the complete transport equations are in the Supporting Information (SI),
which is NOT yet in this repository. Items marked **[SI]** must be checked
against it before any result is reported. Items marked **[assumption]** are my
reconstruction from the main text and from the authors' earlier framework
(Danner et al., Electrochim. Acta 184 (2015) 124; Danner & Latz, Electrochim.
Acta 322 (2019) 134719).

## 1. What changes with respect to the DFN

| Aspect | Li-ion DFN (Chen2020) | Li-SPAN (Simanjuntak 2024) |
| --- | --- | --- |
| Negative electrode | porous graphite, particle diffusion | Li metal foil = planar interface at y = L_tot (BV, U = 0) |
| Positive electrode | NMC particles with radial diffusion | SPAN: sulfur covalently bound to PAN; **no solid diffusion**, 4 local species |
| Reactions | 1 intercalation reaction per electrode | 3 sequential electrochemical reactions + Li2S precipitation (chemical) |
| Electrolyte species | binary salt LiPF6 | Li+, PF6-, S2- (S2- trace, 1e-12 to 1e-3 mol/m3) |
| Solid phases | fixed | Li2S volume fraction grows; porosity decreases |
| Extra physics | none in benchmark | double layer (0.1 F/m2), contact resistance Z_CC |
| Coordinates | (x, r, t) | (y, t) only |

The PINN therefore loses the radial dimension (cheaper) but gains several
local state variables, multi-reaction kinetics, a trace species spanning nine
orders of magnitude and a nucleation-type threshold.

## 2. Geometry and parameters (paper Tables 1-3)

* y = 0 cathode current collector, y = L_cat = 100 um cathode/separator,
  y = L_tot = L_cat + L_sep, L_sep = 618 um (two Whatman GF separators,
  2 x 309 um), Li foil at y = L_tot.
* Cathode volume fractions: carbon+binder 0.032, SPAN 0.0945, Li2S 1e-5
  (initial), porosity 0.87; specific SPAN area a = 1e7 1/m; Li2S surface
  exponent xi = 1.5 **[SI: definition of the Li2S surface area]**.
* Separator solid fraction 0.11, porosity 0.89; Bruggeman 1.5 (cathode),
  1 (separator).
* Electrolyte 1 M LiPF6 in EC:DEC: D_LiPF6 2.52e-10 m2/s (concentration
  dependent, Lundgren 2014 **[SI: functional form]**), D_S2- 4.7e-10 m2/s,
  t+ 0.1625, kappa 0.796 S/m **[SI: kappa(c)]**, thermodynamic factor 1.6.
* Initial: c_Li+ = 1000.02, c_PF6- = 1000, c_S2- = 0.01 mol/m3.
* SPAN conductivity 1 S/m; double layer 0.1 F/m2; Z_CC = 0.025 Ohm m2
  (Table 2; the text quotes best agreement at 0.035 Ohm m2, to be clarified).
* 1/10 C = 0.1 mA/cm2 (1 C = 10 A/m2); voltage window 1.0-3.0 V. Temperature
  not stated **[assumption: 298.15 K]**.

## 3. Reaction network (discharge direction)

    (1) 1/2 PAN-S4-PAN + e- + Li+  ->  1/2 PAN-S3Li + 1/2 PAN-S1Li
    (2) 1/2 PAN-S3Li  + e-         ->  1/2 PAN-S2Li + 1/2 S2-
    (3) 1/2 PAN-S2Li  + e-         ->  1/2 PAN-S1Li + 1/2 S2-
    (4) 2 Li+ + S2-               <=>  Li2S(s)            (chemical, on cathode surfaces)
    (A) Li+ + e-                  <=>  Li(s)              (anode, planar)

Six electrons per PAN-S4-PAN chain (1.5 e- per S, i.e. 75 % of 1672 mAh/gS),
consistent with the ~1250 mAh/gS capacity in the paper. Initial c_S4 = 598
mol/m3 (electrode volume **[SI: confirm volume basis]**); the reference
concentrations are 598, 598, 598 and 1196 mol/m3 for S4, S3Li, S2Li, S1Li.

### Rate laws (paper Eqs. 4-8, 10-14)

For electrochemical reaction i (n = 1), with reduction counted positive:

    R_i = k0_i a_ed^(1-alpha) a_prod^alpha [exp(-alpha dmu_i/RT) - exp((1-alpha) dmu_i/RT)]
    dmu_i = F (phi_s - phi_e - U_i)
    U_i  = U0_i - b_i zeta_i + (RT/F) ln(a_ed / a_prod),   zeta_i = 1 - c~_reactant

a_ed and a_prod are products of activities raised to |stoichiometric
coefficient| (ideal electrolyte, c~ = c/c_ref; SPAN species a = gamma c~).
The paper's Eq. 7-8 combine Delta mu0 and RT ln gamma into the linear
U0 - b zeta; whether the ln(a_ed/a_prod) term is kept separately in U_i
**[SI]**. Parameters: k0 = 1e-2, 1e-2, 1e-4 mol/m2/s; U0 = 2.2, 1.9, 1.66 V;
b = 0.3, 0.28, 0.62 V; alpha = 0.5.

Li2S: R_Li2S = k0_Li2S (a_Li+^2 a_S2-)^alpha [exp(-alpha dmu/RT) - exp((1-alpha) dmu/RT)],
dmu = RT ln K_sp - RT ln(a_Li+^2 a_S2-), K_sp = 10, k0_Li2S = 2e2 mol/m2/s.
Anode: BV with k0 = 3.94 mol/m2/s, alpha = 0.5, U = 0.

## 4. Governing equations [assumption, to be verified against SI]

Unknowns on the cathode y in [0, L_cat]: c_S4, c_S3, c_S2, c_S1, eps_Li2S,
c_e (salt), c_S (S2-), phi_s, phi_e. On the separator: c_e, c_S, phi_e.

Local balances (cathode, a = a_SPAN):

    d c_S4/dt = -1/2 a R1
    d c_S3/dt = +1/2 a R1 - 1/2 a R2
    d c_S2/dt = +1/2 a R2 - 1/2 a R3
    d c_S1/dt = +1/2 a R1 + 1/2 a R3
    d eps_Li2S/dt = V_Li2S a_Li2S R_Li2S,   V_Li2S = M/rho = 2.77e-5 m3/mol
    eps_e = eps_e0 - eps_Li2S

Charge (faradaic + double layer), i_F = F (R1 + R2 + R3) per unit SPAN area:

    d i_e/dy = -a i_F - a C_DL d(phi_s - phi_e)/dt      (sign: i_e > 0 toward cathode in discharge)
    i_s = -sigma_eff d phi_s/dy,  d i_s/dy = -d i_e/dy

Electrolyte. Because c_S2- <= 1e-3 mol/m3 << c_e ~ 1000 mol/m3, a defensible
simplification is a binary LiPF6 electrolyte (DFN-type salt and current
equations with eps_e(t)) plus a trace S2- Nernst-Planck equation in the
electric field of that electrolyte. The paper uses an extension of
concentrated solution theory with all three ions **[SI]**; the simplification
must be shown to reproduce the full model (Fig. 5b) before it is used.

    d(eps_e c_e)/dt = d/dy(eps_e^b D(c_e) d c_e/dy) - (1 - t+) a i_F / F   [cathode]
    d(eps_e c_S)/dt = d/dy(eps_e^b D_S d c_S/dy + migration) + 1/2 a (R2 + R3) - a_Li2S R_Li2S

Boundary conditions: y = 0: zero salt and S2- flux, i_e = 0, i_s = i_app.
y = L_cat: continuity of c, phi_e, fluxes; i_s = 0. y = L_tot (Li foil):
i_e = i_app, Li+ flux = i_app/F (salt flux (1 - t+) i_app/F convention **[SI]**),
S2- flux 0 (no shuttle; the paper neglects polysulfide reactions at the anode),
BV: phi_s,anode - phi_e(L_tot) = eta_A with phi_s,anode = 0 (gauge).

Cell voltage: V = phi_s(0) - Z_CC i_app.

Initial conditions: SPAN at the initial concentrations, c_e = 1000, c_S = 0.01,
eps_Li2S = 1e-5, potentials at equilibrium (consistent with the double layer).

## 5. Numerical reference (needed before any PINN work)

PyBaMM has no Li-SPAN model. Plan:

1. Implement the model as a custom `pybamm.BaseModel` (rhs for the local
   species, eps_Li2S, c_e, c_S, the double-layer potential; algebraic for
   phi_e, phi_s), 1D finite volumes, Casadi solver. A small scipy method-of-
   lines solver is an independent cross-check.
2. Reproduce the paper's Figs. 4b (rate curves), 5a/5b (species and Li2S)
   and 6b (Z_CC) within plotting accuracy. Only then use it as ground truth.

## 6. PINN design notes for Li-SPAN

* Inputs (y, t); no radial coordinate. Reuse the v2 ingredients: physical
  output scales, hard initial conditions, tanh current ramp, resampling,
  adaptive group weights, PyBaMM-style reference only for evaluation.
* Local species: hard IC and hard element conservation. Sulfur and chain-end
  balances give two algebraic invariants (e.g. c_S4 + c_S3/2 + c_S2/2 + ...),
  which can be built into the parametrization so only 2 of 4 species are
  learned.
* S2- must be learned as log(c_S2-) (range 1e-12 to 1e-3 mol/m3).
* Learned reaction current: with three parallel reactions sharing one
  potential difference, the v1/v2 "learned j + inverse BV" idea does not
  invert reaction by reaction. Two options to compare:
  (a) learn the interfacial potential difference dphi = phi_s - phi_e and
      evaluate R1-R3 directly; (b) learn the total faradaic current i_F with
      the hard electrode-integral projection (as in v2) and recover dphi by a
      differentiable monotone root solve of sum_i R_i(dphi) = i_F/F
      (implicit-function gradients). Option (b) is the natural extension of
      the current research question to multi-reaction chemistry.
* Li2S nucleation is a threshold phenomenon (S2- pinned near K_sp after
  ~400 mAh/g). Expect a sharp time feature: use a time-marching/causal
  schedule or a feature built from the precipitation driving force.
* The double layer removes the algebraic t = 0 corner and is helpful for the
  PINN; keep it.
* Inverse problem (the scientifically interesting part): identify a small,
  identifiable subset of {U0_i, b_i, k0_i, K_sp, Z_CC, Bruggeman} from
  discharge curves at several C-rates (paper data on request from the authors,
  or the group's own Li-SPAN cells). Check identifiability with sensitivities
  before fitting; the paper itself reports low sensitivity to k0 (Fig. S1).

## 7. Open items for the owner

1. Obtain the Supporting Information PDF (full equations, Lundgren D(c) and
   kappa(c), Li2S surface-area law, volume basis of c_S4).
2. Decide the target data: paper curves (digitized), authors' data on
   request, or in-house Li-SPAN cells from the Balbuena group.
3. Confirm Z_CC (0.025 vs 0.035 Ohm m2) and temperature with the authors or
   by refitting.
