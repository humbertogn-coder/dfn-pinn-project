# PINN-DFN project: state for future sessions (Claude)

Owner: Carlos Guerrero (TAMU, Balbuena group). Speak Spanish with the owner;
write code/docs in English (repo convention).

SCOPE DECISION (owner, 2026-10-07): Li-SPAN only - Li-S will NOT be done. Next: real Li-SPAN measurements
(raw discharge curves at >= 2 C-rates + cell metadata + slow/GITT OCV; list sent to the owner 2026-10-07).
PUBLICATION PLAN (owner + advisor meeting, 2026-10-07): short, correct paper for J. Electrochem. Soc. (Newman-type
modeling audience; not exhaustive). Owner will later request a long, simple-language English report (theory, formulas,
analogies) for an undergraduate's progress presentation. Candidate real data: Wang et al., Nat. Mater. 25, 791 (2026)
(Balbuena group co-authors): source data in repo data/span_natmater2026/ - Fig. 4a = SPAN500 cycles 1-3 at C/10 only
(1C = 600 mA/g_SPAN, 1-3 V, ~1400 pts/discharge; cycle-1 627, cycle-2 458, cycle-3 449 mAh/g_SPAN), Fig. 4c EIS
(~10-15 Ohm cm2 total -> ohmic drop ~2 mV at C/10), coin cells, ~2 mg/cm2, ether electrolytes (2 M LiFSI DME:BTFE or
1 M LiTFSI DOL/DME + LiNO3 - which one for Fig. 4a not stated); missing S content of SPAN500, thickness, other rates.
Overlay results/lispan/span500_vs_lispan_model.png: cycles 2-3 similar shape to Simanjuntak, 50-100 mV lower mid-way.
2026-10-07/08: owner approved the 5 paper steps; real-data case LIMITED to the Nat. Mater. 2026 data (owner may get more
later, do not wait for it). Student guide (English, simple language) DONE: report/PINN_DFN_LiSPAN_student_guide.docx
(48 pp; built by report/build/build.js from ch_*.js, figures by report/make_figures.py, equations by
report/build/render_equations.py). Step 1 (reproducibility) RUNNING: configs/lispan_forward_final_{0.1,1}C.json (from
scratch, 30k steps, charge_total_start 10000 - active from step 1 the inventory term fought the extent ODEs, first
attempt STOPPED), seed 0 launched; seeds 1, 2 to follow. Steps 2-5 started 2026-10-08 (see the dated section).

## Locations
- Owner's connected folder (Windows): `C:\Users\humbe\OneDrive\Documents\Claude\PINN-DFN-Project`
  - `DFN_PINN_Project/` = scientific Git repo (remote github.com/humbertogn-coder/dfn-pinn-project,
    HEAD b08dd50 + Codex files + all v2 files added by Claude; owner pushed on 2026-10-05). Nothing pushed by Claude.
  - `Practica/analysis/potential_level_shape/` = Codex's last isolated analysis (not in Git).
  - `_claude_transfer/code_snapshot.tar.gz` = tarball Claude made to copy code to the cloud (owner may delete).
- Owner env: Anaconda Prompt, `conda activate dfn-pinn` (Python 3.11, torch 2.14, PyBaMM 26.8).
- Grace HPRC available (`ssh humbertogn@grace.hprc.tamu.edu`, scratch `/scratch/user/humbertogn`).
- Cloud sandbox limits seen: 2 CPU cores, ~4 GB memory cgroup for Claude's processes (an
  8x-batch float64 L-BFGS + two trainings + a plot got OOM-killed). WebSearch blocked by org policy,
  WebFetch needs owner permission per URL -> ask the owner for PDFs.
- Owner folder `references/`: Simanjuntak2024_LiSPAN_main.pdf + _SI.pdf (SI received 2026-10-05; equations now
  complete, see docs/LI_SPAN_MODEL_FORMULATION.md). `data_LG_M50/`: metadata of the synthetic LG M50 degradation
  dataset (12 cells p1c1..p4c3, PyBaMM DFN Chen2020 + SEI/cracking/plating/LAM, 1C/1C 25 C, 648-1200 cycles,
  retention 69-99 %); full per-cell time series + "degradation pathways" folders and the dataset citation still to
  be provided by the owner. `Siguientes_pasos_PINN_LiSPAN.pdf`: plain-language plan (Spanish) for the owner.

## History (Codex, Sep 11 - Oct 3 2026)
Full DFN PINN never converged (best 23 mV on a 1 s benchmark). Root causes found by Claude:
fields = base + 0.01*NN (overpotentials needed NN ~ 350), 1 s step-from-rest benchmark,
2x12 nets / 128 fixed points / 2000 steps / 34 unit-weight terms.

## v2 (Claude, 2026-10-03/04) - docs/V2_DESIGN.md, docs/V2_RESULTS.md, docs/V2_QUICKSTART.md
Benchmark: Chen2020 DFN, SOC 1, I = 5 A tanh(t/30 s), 0-3000 s; PyBaMM Casadi reference x80/r120,
evaluation only. Physical output scales; hard IC; kink features; learned j with hard electrode-current
projection; inverse BV; first-order solid charge; particle = theta_bar + parabola(j) + zero-mean
correction (flux BC/inventory exact); resampling; gradient-norm group weights; global salt term.

### Step 1 (forward refinement) outcome, 2026-10-04
- Baseline (no Fourier): 3 seeds, V 2.56/2.68/3.19 mV rmse, max 9.1/8.6/8.9 mV; ERROR FIELDS IDENTICAL
  across seeds (corr 0.94-0.99) -> systematic representational bias, not optimizer noise.
- Tried without gain at equal cost: separator-side sampling, kinetics reweighting (trade-off),
  derived-current formulation, hard kinetics, causal time weights. Fixed-batch L-BFGS OVERFITS the
  collocation points (loss -43 %, c_e error x5). Mini-batch L-BFGS (resample every 25 it): 10-20 % gain.
- Direct regression test (fit networks to PyBaMM fields, no physics) showed the plain (2t-1, 2g-1) time
  inputs cannot represent V(t) better than 8 mV: spectral bias in TIME. Fourier time features
  (`fourier_t=4`, sin/cos 2 pi k t) fix it: PINN reaches V 0.54 mV rmse / 2.46 mV max, phi_e max 2.6 mV,
  c_e rmse 7 / max 62, j_n rms 0.7 %, j_p rms 2.7 % (max 23 % in the 30-150 s transient at the
  positive separator side), theta_surf rms 5e-4/1e-3 in 20k steps (run f4_seed0_20261004T120413Z).
  f4 seed study complete: V 0.51 +/- 0.04 mV rmse, 2.48 +/- 0.39 mV max; c_e rmse 8.5 / max 77;
  j_n 0.7 %, j_p 2.4 %; theta_surf rms 6e-4/8e-4 (runs f4_seed0/1/2). Mini-batch L-BFGS on f4 adds
  nothing (optimizer no longer the limit). Seed error correlation fell to 0.72-0.94.
- `configs/v2_forward_1C.json` now has fourier_t=4. Trainer options added: inventory=derived,
  kinetics=hard, causal, fixed_group_weights, x_edge_fraction, soft_current (projection=false),
  mini-batch L-BFGS, Adam-state save/restore (--init, --fresh-optimizer), parameter multipliers
  (inverse problem: D_n, D_p, k_n, k_p, sigma_p).
- Synthetic inverse (old architecture, from scratch, 1 mV noise): D_p +3.8 %, k_n -8.2 % from 2.5x/0.4x
  starts (inverse_scratch_20261004T033045Z). Should be repeated with fourier_t=4 (config updated).

### Step 2 (inverse + identifiability) outcome, 2026-10-04 evening
- scripts/v2_identifiability.py (PyBaMM FD sensitivities, FIM, CRLB, correlations). At 1C, 1 mV noise:
  D_p 0.55 %, D_e 2.9 %, k_n 3.1 %, kappa_e 4.0 %, D_n 4.5 %, k_p 19 %, sigma_p 37 % (CRLB rel. std);
  (k_p, sigma_p) is the weak, correlated pair (-0.81 at 1C, -0.97 with 3 rates). Three rates improve all.
- PINN multipliers now cover D_n, D_p, k_n, k_p, sigma_p, D_e, kappa_e.
- Synthetic inverse with f4 (D_p, k_n, D_n from 2.5x, 0.4x, 0.6x; 1 mV noise): from scratch 15k + 10k cont:
  D_p +0.8 %, k_n -0.2 %, D_n -8.8 % (D_n is the slow direction, still drifting to 1; misfit at the noise
  floor). Warm start from the forward model: +1.5 %, -1.0 %, -10.1 % after 5k steps.
- Early transient attempt (short_t=[50,200] + early2 sampling, run f4s_seed0): regression fit better but PINN
  worse at 20k (V 0.81/5.09 mV) except j_p/theta_p; not adopted. Regression evidence: width 96 fits best
  (V 2.42 mV vs 3.36 for f4 at 3000 steps) -> recommended next forward run: width 96, 40k steps.

### Step 3 (multi-rate inverse + error model) outcome, 2026-10-05 (V2_RESULTS.md 3.3)
- New: src/dfn_pinn/v2/multirate.py + scripts/v2_inverse_multirate.py + configs/v2_inverse_multirate.json:
  one DFNPINN per protocol (C/2 2.5 A/6500 s, 1C, 2C 10 A/1400 s), shared log-multipliers, ~1 s/step on 2 cores.
  References ref_I2.5A_t6500s_* and ref_I10A_t1400s_* (x80/r120, ~60 MB each, NOT copied to the owner; regenerate
  with v2_make_reference.py --current 2.5 --t-end 6500 / --current 10 --t-end 1400). Data npz copied.
- 5-parameter attempt (D_p,k_n,D_n,D_e,kappa_e) diverged twice (electrolyte pair compensates c_e errors) ->
  added param_warmup_steps, param_log_bound, param_release_steps (per-parameter release), --init (warm start
  releasing more parameters), --resume (rolling latest.pt every latest_every=250 steps, also in v2_train.py;
  the cloud container restarted 3 times during these runs).
- 3-parameter multi-rate run (f4_multirate3_20261004T224451Z, 10k steps): D_p -3.7 %, k_n +1.1 %, D_n -28 %;
  misfits at the 1 mV noise floor from step 5000, fields NOT converged (c_e max err 176 at 1C, 374 at 2C).
- KEY FINDING: parameter error = systematic forward voltage error / rms voltage sensitivity, not the CRLB.
  rms dV/dln p at 1C [mV]: D_n 5.0, D_p 73, k_n 45, k_p 16, sigma_p 6.8, D_e 41, kappa_e 22 -> a 0.5 mV forward
  error = 10 % D_n, 0.7 % D_p, 1.1 % k_n (matches the single-rate result -8.8/+0.8/-0.2 %). Bias test
  (f4_biastest_true0: true parameters, converged forward model, noise-free data) settles at D_n -4.1 %,
  D_p +0.5 %, k_n -0.2 % -> biased minimum confirmed. Multi-rate data only help once the forward error at the
  highest rate is below the noise; hierarchical release needed.
- scripts/v2_bias_prediction.py PREDICTS the bias from a forward run alone (least squares of the forward
  voltage error on the FD sensitivities): f4_seed0 -> D_p +0.25 %, k_n -0.11 %, D_n -3.7 % (observed +0.5/-0.2/
  -4.1 %); seeds 1/2 -> D_n -5.2 / -6.2 %. Use it to report every inverse estimate with its predicted bias.
- C-F factorial DONE (V2_RESULTS.md section 5; fourier_t=4, seed 0, 20k steps): C (direct BV + soft current)
  and F (direct BV + hard projection) are STUCK at V rmse ~460 mV through 10k steps (stopped; same with unit
  weights, probe_C_unitw) -> the direct sinh residual is the v1 failure mode. D (inverse BV + soft current)
  converges to 0.70 mV rmse / 4.8 max; E (inverse + hard projection, the default) 0.57 / 2.46, ~2x ahead of D
  at every step. Kinetics residual = decisive factor; hard projection = second-order gain.
- f4_w96_40k DONE (V2_RESULTS.md section 6; width 96, 40k steps, 4.7 h): V 0.29 mV rmse / 1.34 max (baseline
  0.54 / 2.46), phi_e max 1.0 mV, j_p 1.2 % rms / 12.5 % max, thsurf max 0.0020/0.0032 -> every error 1.5-2.5x
  better EXCEPT the c_e boundary layer at the negative collector (max 62 -> 52 mol/m3).
- The D_n bias did NOT shrink: predicted -3.6 %, bias test from the w96 model -3.9 % (w64: -4.1 %). Error
  decomposition (3.3): persistent late-time offsets at the negative collector, U_n -0.39 mV (thsurf,n too high by
  8e-4), eta_n +0.73 mV, dphi_e +0.61 mV, identical for both models; origin = c_e(x=0) under-predicted by 26-36
  mol/m3 at t ~ 1200 and ~ 2400 s (the bumps from the graphite-staging current redistribution), a boundary layer
  at the zero-flux collector. It enters eta_n via j0 ~ sqrt(c_e) and dphi_e via the concentration overpotential
  and projects on the D_n direction. Reference mesh error excluded (x40/r60 vs x80/r120 = +0.4 % D_n equiv.).
- Remedies tested (V2_RESULTS.md 3.3, 7): collector sampling (f4_coll_seed0) WORSE (c_e(0) -49/-66, V 0.81);
  fourier_t=8 (f8_seed0) WORSE (V 0.65, D_n pred -9 %). Mechanism found: the SOFT zero-flux BC at the collector
  is violated (wall slope +9e5 mol/m4 at 2400 s, bc residual only 5e-3 in its normalization). Implemented
  collector_bc="hard" (model._xe: cos(pi X) + sin(pi/2 kink) features -> d c_e/dx = d phi_e/dx = 0 exact at both
  collectors; unit test added). Run f4_hardbc_seed0: c_e rmse 7.0 -> 3.4, max 62 -> 32, V 0.54 -> 0.45 mV,
  eta_n late offset gone; BUT predicted D_n bias still -4.7 %: the late-time V offset (+0.27 mV, t > 2400 s)
  persists and moved to other components. fourier_period=2 (f4_hardbc_p2_seed0) REJECTED the aliasing
  hypothesis (V 0.60, offset +0.27 unchanged, D_n pred -4.4 %). The late-time error is the same in all four
  forward variants: +0.3..0.8 mV at 2300-2700 s, -0.4..-1.2 at 2900-2950 s, +1.3..3.2 mV jump at t = t_end
  (= every run's "V max"); reference smooth there; window = graphite surface stoichiometry 0.21 at the
  collector (steep end of the graphite OCP, D_n sensitivity largest). Remedies to try on the owner's
  machine: end-of-discharge time feature, seed study of collector_bc=hard, hard collectors + width 96 x 40k
  (expected ~0.2 mV). Extended domain TESTED (f4_hardbc_t3300_seed0, configs/v2_forward_1C_t3300.json, new
  reference ref_I5A_t3300s_*): removes the end-point jump (max 2.7 -> 1.3 mV on [0,3000]) but the 2100-2700 s
  offset (+0.4-0.6 mV) and the D_n bias (-6 %) stay. Diagnosis (V2_RESULTS.md 7): the PINN under-resolves the
  current redistribution of the graphite plateau transition (j_n(0) rises to 1.32x the mean at 2500 s,
  PINN 2-3 % low in every variant) -> solid-phase/negative-electrode representation at the transition is the
  next target (time sampling around the transition, OCP-slope time feature, more capacity in the n-nets).
- DEFAULT CHANGED: configs/v2_forward_1C.json, v2_inverse_synthetic.json, v2_inverse_multirate.json now have
  collector_bc="hard". --init checkpoints must match (v2_train.py warns on architecture mismatch; all runs
  before f4_hardbc_* are collector_bc=soft).
- BUG FIXED 2026-10-05: first --resume re-stepped the scheduler on the stored lr (double decay). Affected:
  f4_multirate3 (lr x0.4 from step 1000) and 250 steps of f8_seed0. Now closed-form lr on resume.

### Step C groundwork (aged cells / commercial-cell data), 2026-10-05 evening (V2_RESULTS.md section 8)
- DFNPINN.PARAMETERS now also has theta_n0, theta_p0 (LLI), eps_am_n, eps_am_p (LAM; scale a = 3 eps/R incl. the hard
  projection) and R0 (lumped resistance, LINEAR: R0 = 5e-3 Ohm * log_mult["R0"], only in the terminal-voltage data
  term, model.terminal_voltage). V0/phie0 are now properties of the current theta0 multipliers. Unit test added (13 pass).
- New: src/dfn_pinn/v2/aging.py (cycle npz -> CC segment -> Protocol + data file), scripts/v2_make_aging_dataset.py
  (PyBaMM DFN + OKane2022 degradation, 1C CC-CV cycling, writes cycles/cycle_XXXX.npz with truth + summary.csv),
  scripts/v2_inverse_aging.py (one aged cycle -> inverse with --init fresh model -> aging_result.json),
  scripts/v2_import_lgm50.py (generic reader for the real LG M50 files; adjust COLUMN_ALIASES once seen),
  configs/v2_inverse_aging.json. Smoke-tested end to end.
- Synthetic cell results/aging_synthetic/cellA (200 cycles, coarse mesh, every 25th discharge saved) generated in the
  sandbox (NOT in Git; ~1 MB; copied to the owner's folder when ready).
- Unattended chain armed (results/run_stepB_chain.sh): after f4_hardbc_w96_40k finishes -> bias prediction ->
  B_hier_w96 (hierarchical inverse D_p,k_n then D_n at step 2000, warm start from A, 6000 steps) -> C_aging_c200
  (aging inverse on cycle 200). Logs results/stepB_chain.log, results/B_hier_w96.log, results/C_aging_c200.log.
  If the container restarts, re-arm the chain (nohup bash results/run_stepB_chain.sh &) and resume the forward run.

- First aging inverse (C_aging_cellB_c200_w64, cellB cycle 200, LLI 6.7 % / LAM_n 6.4 %): negative-electrode
  lithium content (theta_n0 x eps_am_n) recovered to -1.2 %, but the LLI/LAM split is wrong (theta_n0 +9.8 %,
  eps_am_n -10 %; correlation -0.992 in the FIM), theta_p0 -7.8 %, R0 at its 1.5 mOhm bound; misfit 1.48 mV still
  falling. A run WITHOUT the forward pre-adaptation stage ran away (stopped). Identifiability of the 5 aging
  parameters (results/v2_identifiability_aging): CRLB 0.15-0.4 %, sensitivities 300-2000 mV/e-fold, condition 890.
  Remedies listed in V2_RESULTS.md section 8 (longer budgets, 10 s sampling, staged release, CV tail / low-rate data).

### Steps A-B-C outcome, 2026-10-06 morning (V2_RESULTS.md 8, 9)
- A DONE: f4_hardbc_w96_40k_20261005T194145Z (hard collectors, width 96, 40k steps): V 0.25 mV rms / 1.15 max,
  c_e max 6.6 mol/m3 (first run passing 10), phi_e 0.75 mV, j_n 0.4 %, j_p 0.8 %, thsurf max 0.002. Predicted
  D_n bias -0.93 % (D_p -0.01, k_n +0.47). THE model to start every inverse from (width 96!). Seed 1 of it running
  at hand-over (f4_hardbc_w96_40k_seed1, log results/f4_hardbc_w96_40k_seed1.log, ~5 h; resume if restarted).
- B DONE: hierarchical inverse (B_hier_w96: D_p,k_n released at 0, D_n at 2000, 6000 steps -> +1.7/-1.1/-12.8 %,
  D_n budget-limited) + continuation B2_hier_w96_cont (param_lr 0.02, lr 1e-4, 6000 steps) ->
  D_p +0.1 %, k_n +0.4 %, D_n -0.6 % (CRLB 0.25/0.41/1.5 %). Solid-phase inverse at 1C solved.
- C: three aging inverses on cellB cycle 200 (w64 two-stage; w96 two-stage; w96 no warm-up, data scale 5 mV) all
  land in the degenerate valley: theta_n0 +10-12 %, eps_am_n -10..-12 %, theta_p0 -8..-14 %, eps_am_p -1..-3 %,
  R0 at bound; negative-electrode lithium content (theta_n0*eps_am_n) -1.1..-1.2 %. ROOT CAUSE (PyBaMM check):
  the plain DFN with the TRUE aged (theta0, eps_am[, porosity]) misses the aged discharge by 31 mV rms (cellB)
  and 45 mV (cellC = no cracking); aged cell ABOVE the DFN (best R0 negative). The 5-parameter aging DFN cannot
  represent the degradation model's 1C discharge (suspects: reversible plated-Li stripping, SEI film resistance
  distribution, porosity). Next for C: diagnose with PyBaMM which mechanism (switch them off one by one), add the
  needed physics as residuals/parameters, or use the C/100 CV tail / rests of the real data set.
- Generator now exports a_n_ratio, a_p_ratio, roughness_n, porosity_n/p truth; --no-cracking option (cellC).

### Step C root cause + Step D (Li-SPAN reference), 2026-10-06 afternoon
- C ROOT CAUSE FOUND (scripts/v2_aging_mismatch_diagnosis.py, results/aging_mismatch_diagnosis*.log, V2_RESULTS 8 end):
  NOT aging. Every O'Kane variant (no LAM, no plating, ...) gives the same 56-75 mV rms with best R0 = -10..-14
  mOhm, already at cycle 2 (fresh cell); all variants WITHOUT particle mechanics give 6.5-8.6 mV; the full model
  with `"stress-induced diffusion": "false"` gives 7.4 mV. PyBaMM enables stress-induced diffusion (Ai 2019 eq 12)
  by default whenever particle mechanics is on: D_k(c) = D_k0 (1 + theta_M,k c), theta_M = Omega/(RT) 2 Omega E/
  (9(1-nu)), c_0 = 0; OKane2022 -> NMC factor 110-370 (D_p effectively 100-400x Chen2020), graphite 1.2-1.5. The
  data have no NMC diffusion polarization -> discharge 60-70 mV above the PINN's DFN, current-proportional ->
  "negative R0". The real LG M50 data set (cracking -> mechanics) has the same feature. Also: the 30 s tanh ramp
  lags the data's current step by 20.8 s of charge (18.5 mV rms / 128 max vs fresh cell at 1C; 7 mV with 1 s).
  FIX next: (a) regenerate synthetic cells with stress-induced diffusion off (5-parameter aging DFN then adequate
  to ~7 mV) and redo the LLI/LAM inverse with ramp_s <= 1 s; (b) for the real data: add D_k(c) = D_k0(1+theta_M c)
  to residuals.py (particle term) with OKane2022 mechanical parameters, or release D_p in a first stage.
- D DONE (Li-SPAN numerical reference): src/dfn_pinn/lispan/{params,model}.py, scripts/lispan_discharge.py,
  lispan_digitize_paper.py, lispan_fit_paper.py, tests/test_lispan_reference.py (7 pass). 1-D FV MOL, scipy BDF,
  log states (SPAN species, S2-, eps_Li2S), double layer, NP electrolyte with measured kappa, Z_CC. 0.1 C in 4 s.
  Reproduces paper Fig 6a (0.1 C, 1 C, Zcc=0) to 16/18 mV rms, Fig 5a species (S3 peak 490 vs 520, S2 540 vs 533,
  c_S1 598->1196, Li2S onset ~350) and Fig 4b capacities (1254/1240/1212/1025 vs 1252/1214/1229/994 at
  0.05/0.1/0.2/1 C, Zcc 0.025). Paper + SI do NOT fix (documented in LI_SPAN_MODEL_FORMULATION.md 5.2): (1) the
  kinetic regime - Table 3 k0 with a_SPAN 1e7 would be at equilibrium; Fig 6a shows Tafel (0.118 V/decade k0) ->
  effective k0 fitted (2.95e-8, 2.39e-8, 2.34e-8 mol/m2/s, results/lispan/fit_k0.json); (2) reaction 3 must be
  irreversible in discharge (else comproportionation S3+S1->2S2 consumes PAN-SLi, absent in Fig 5a); (3) S2-
  references: c_ref 0.01 in kinetics, saturation 1e-5 mol/m3 at K_sp 10 (Figs 5b/7). i_F = F a (r1+r2+r3)
  settled (phase 1 carries capacity). Fig 4b consistent with Zcc 0.025, not 0.035. Digitized paper curves in
  results/lispan/paper_digitized/. Reference npz: results/lispan/ref_Zcc0*/discharge_*C.npz (+ figures).

### Evening 2026-10-06: Li-SPAN forward PINN + step C redo (both DONE; results below)
- Li-SPAN forward PINN implemented: src/dfn_pinn/lispan/pinn.py, scripts/lispan_train.py (+ --resume), lispan_plot_run.py,
  configs/lispan_forward_01C*.json. Design (LI_SPAN_MODEL_FORMULATION.md 6): hard extents xi1>=xi2>=xi3 (log-type
  remaining fractions e^{-T P}), species/Li2S/sulfur exact; R = (I/F) rho/int(a rho) hard projection; Delta phi = root
  of sum f_m = R (bisection+Newton, implicit grads, slope floored at -F R/2RT); phi_e with anode value built in; salt
  with tanh-bounded deviation; residuals: 3 extent ODEs, solid/electrolyte Ohm (Huber), salt PDE + 3 flux BCs +
  global inventory. Benchmark choice: reactions 2 AND 3 irreversible (reaction 2's mass-action reverse term is a
  10-decade stiff constraint a PINN cannot resolve; reference differs 2.6 mV at 0.1 C). Reference for evaluation:
  results/lispan/pinn_reference_0.1C_rii_Zcc0.025.npz (FV, c_DL 1e-6, ramp 30 s).
  Pitfalls fixed (documented 6.3): float32 cancellation of 1 - xi (compute species from e_m); activity floor must be
  ~1e-30 (1e-12 mol/m3 left a 1.4 % phantom rate of exhausted reaction 1 -> 24 mV plateau, run A kept as
  results/lispan_runs/runA_*); Huber on potential residuals (Tafel amplifies species roughness into 1e4 residuals).
  RESULT run lispan_f01C_20261006T190806Z (w64, fourier 4, 10k steps, 45 min): V 1.18 mV rms / 3.9 max vs FV,
  species 0.1-1.3 mol/m3, eps_L 5e-5, phi_e 0.025 mV, c_e 1.2 mol/m3 (inventory offset). Run B
  (lispan_f01C_w96_20261006T194300Z: w96, fourier 8, 20k steps, global salt inventory, 1.6 h): V 1.00 mV rms /
  3.4 max (checkpoints 0.39-1.5 mV: the error is dominated by the integration bias of c_S2 x b_3, ~1 mV per
  mol/m3), species 0.1-1.0 mol/m3, c_e 0.15 mol/m3 (inventory term fixed the offset), phi_e 0.027 mV. Documented in
  LI_SPAN_MODEL_FORMULATION.md 6.4-6.5 with the remedies (ODE weight / EMA of weights / L-BFGS tail).
- Step C redo: PINN now has stress-enhanced diffusion (CellParams theta_M_n/p; model.D_factor; residual D lap + D'
  |grad|^2; reference functions; test added, 14 pass) and the OKane solid Bruggeman sigma_eff; aging data aligned by
  charge (aging.charge_shift). PyBaMM check: plain DFN + D(theta) + sigma_eff at the true aged state matches the aged
  discharge to 12.7 mV rms, best R0 +2.35 mOhm (SEI/plating), 4.9 mV residual. Turning stress-diffusion OFF in the
  generator is not viable (stress-driven LAM collapses the NMC in one cycle; cellD attempt killed after 1.5 h).
  RESULT (V2_RESULTS.md end of section 8): stage 1 = results/v2_runs/C3_aging_cellC_c200_Dstress_stage1_20261006T200401Z
  (4000 forward steps, fresh params, D(theta)+sigma_eff; w_scale had to be divided by the D factor at theta_0 - the
  first attempt *_stopped_wscale stalled at particle 0.1); stage 2 = C3b_aging_cellC_c200_Dstress_R0free_20261006T211155Z
  (8000 steps, R0 bound raised to 10 mOhm after a first stage 2 pinned R0 at the old 2.5 mOhm bound; launched with
  --init <stage1>/pretrain/final.pt --pretrain-steps 0 --set 'param_bounds={"R0": 2.0}'). cellC cycle 200 (LLI 4.7 %,
  LAM_n 6.3 %, LAM_p 2.8 %, SEI 77 nm): theta_n0 +1.4 %, theta_p0 -0.5 %, eps_am_n -1.6 %, eps_am_p +0.2 %, R0 5.91
  mOhm (PyBaMM lumped at the true state 5.47), neg.-electrode lithium -0.3 %, LLI 5.1 vs 4.8 %, LAM_n 7.8 vs 6.3 %,
  misfit 0.21 mV rms; parameters still creeping to the truth at the end (longer stage 2 / lower lr would help).
  Conclusion: with the model form fixed the 5-parameter aging DFN is adequate at 1C; the remaining (theta_n0,
  eps_am_n) valley is what 1C data leave - low-rate or CV-tail data (the real set has them) split LLI from LAM_n.
  Seed-1 forward run: DONE 2026-10-07 (see below).

### 2026-10-07 (night, Opus 5.5 session): Li-SPAN identifiability, forward fixes, first inverses (LI_SPAN 6.4-6.7)
- Identifiability (scripts/lispan_identifiability.py, results/lispan/identifiability/): U0_2/U0_3 exactly confounded
  with k0_2/k0_3 (Tafel; only k0 exp(F U0/2RT) identifiable); single rate cannot separate kinetics from Z_CC once OCV
  params are free; 0.1 C + 1 C make {k0 x3, b x3, U0_1, Z_CC} identifiable to < 1 % (1 mV noise); kappa0, t+,
  kappa_SPAN, D_salt, K_sp, k0_L, D_S practically invisible in discharge voltage.
- Forward 0.1 C fix: global charge-inventory residual `charge_total` (2 F c40 L int sum xi = Q(t); own group
  "inventory"; cfg.charge_total_scale): run C lispan_f01C_w96_inv_20261007T032802Z 0.21 mV rms / 0.95 max (control
  without it 0.7-1.5 mV). EMA option added (no measurable gain).
- Forward 1 C: lispan_f1C_w96_sepscale_20261007T061233Z 0.12 mV rms / 0.66 max, c_e 0.7 mol/m3, after the fix
  cfg.salt_sep_natural_scale (separator salt residual on (1-t+)I/(F L_sep); the cathode scale hid a 25 % error of the
  1 C separator transient). Earlier 1C runs kept (lispan_f1C_w96_inv_* 1.97 mV; *_cont_* STOPPED, see STOPPED.txt).
- Inverse 0.1 C (two stages: lispan_train.py with cfg.fixed_multipliers = guess, then inverse_params + data):
  k0 x(2,0.5,1.5) -> -1.65/-0.51/-0.06 % (ideal FV estimator with the same noise -1.61/+0.06/-0.09); with Z_CC
  +1.25/+1.01/+1.27/+2.73 % (ideal +1.2/+1.4/+1.2/+2.6). Data: scripts/lispan_make_inverse_data.py ->
  results/lispan/inverse_data/V_{0.1,1}C_nominal.npz. Plot: scripts/lispan_plot_inverse.py.
- Multi-rate inverse (src/dfn_pinn/lispan/multirate.py, scripts/lispan_inverse_multirate.py; one PINN per rate,
  shared log_mult): 8 params from 0.1 C + 1 C, guess k0 x(2,0.5,1.5), b x(1.1,0.9,1.1), U0_1 +20 mV, Z_CC x1.5.
  First attempt STOPPED (the 125 mV initial 1C misfit wrecked the 1C electrolyte fields) -> cfg.net_freeze_steps
  (parameters first, fields frozen): lispan_inv8p_multirate_freeze_20261007T073258Z, Adam crawls along the
  correlated directions: k0 +9.8/-2.1/+0.6 %, b -1.0/-0.6/+0.1 %, U0_1 -3.7 mV, Z_CC +0.4 % after 7000 steps.
  FIX = Levenberg-Marquardt on the parameters with fields frozen every 250 steps (cfg.gn_every/gn_iters/gn_until,
  multirate.gauss_newton_parameters; FD Jacobian of data residuals, ~10 s): lispan_inv8p_multirate_lm_20261007T083347Z
  k0 +1.96/-0.31/+0.31 %, b -0.14/+0.05/+0.06 %, U0_1 -0.99 mV, Z_CC +0.09 % (ideal estimator for the same noise
  k0 +1.56/+1.08/+0.12 %, b +0.05/+0.30/+0.02 %, U0_1 -0.67 mV, Z_CC +0.04 %; CRLB 1.0/0.7/0.75 %, 0.17/0.20/0.12 %,
  0.44 mV, 0.09 %). First LM call: 84 -> 2.6 mV misfit. Figure results/lispan/inverse_8p_multirate_adam_vs_lm.png.
  AdaptiveWeights now skips non-finite gradient norms (a NaN weight had made every later loss NaN).
  DFN note: the DFN PINN's V is a network output (tied to eps_am etc. only via residuals), so a frozen-field LM needs
  a kinetic voltage evaluation there first.
- MEASURED data (LI_SPAN 6.8): Fig. 4b circles digitized (lispan_digitize_paper.py --experiment; 21/18/22/12 points),
  converted with the paper's capacity convention (make_inverse_data --experiment). Nominal model 31/50/61/46 mV rms.
  FV least squares (scripts/lispan_fit_experiment.py, 8 params, 0.1+1 C): 18.4 mV, flat valley (two starts, different
  params, same misfit); robust: Z_CC = 0.033 Ohm m2 (paper 0.035), k0_3 x3-4, b_2 x0.6; predictions 0.05/0.2 C 29-30/31-32.
  PINN on real data: joint training bends physics (reports 14 mV, FV at its params 20.5); split modes drift (frozen-field
  gradient lacks d u*/d theta); trust-region mode (cfg.gn_mode="tr", data_to_fields=false) is monotone and
  physics-consistent (PINN 20.24 = FV 20.20 mV at its params) but stalls 1.8 mV above the FV optimum.
  Lessons for Li-S: physics-only fields + total parameter sensitivities (parametric PINN / implicit diff / Broyden).
- DFN seed-1 of f4_hardbc_w96_40k DONE (V2_RESULTS 9): V 0.23 mV rms / 1.18 max (seed 0: 0.25/1.15), c_e max 11 vs
  6.6 mol/m3, predicted bias D_p/k_n/D_n +0.07/+0.27/-1.76 % (seed 0 -0.01/+0.47/-0.93): voltage reproducible, D_n bias
  has ~1 % seed spread.
- DFN seed-1 run (f4_hardbc_w96_40k_seed1) paused/resumed around these jobs (SIGSTOP/SIGCONT pid in sandbox).
- Tests: tests/test_lispan_pinn.py (4 tests: charge_passed, inventory identity, charge_total residual, parameter
  roundtrip).

### 2026-10-08: paper steps 1-5 in progress (docs/PAPER_PLAN.md, docs/PAPER_ASSUMPTIONS.md)
- Step 1 (reproducibility, final recipe from scratch): lispan_final_0.1C_seed0_20261008T005207Z running (resumed
  after container restarts with --resume; time_s excludes pauses): 2.59 / 1.66 / 1.24 mV @2500/5000/7500 (charge
  inventory switches on at 10000; run C needed it for 0.21 mV).
  BUG FOUND + FIXED 2026-10-08: `lispan_train.py --resume` rebuilt the protocol from config.json, whose key is
  'current_A_m2', but only accepted 'current' -> silently the default 1.0 A/m2 (0.1 C). The 1 C seed-0 run
  lispan_final_1C_seed0_20261008T005207Z trained a 0.1 C model on the 1 C window from step 1751 (its "0.012 mV" evals
  were 0.1 C on 2781 s - INVALID; STOPPED.txt written; config.json there was overwritten with the wrong protocol).
  Fix: LiSPANProtocol.from_dict (both keys, raises otherwise) in lispan_train.py and dfn_pinn.lispan.runs; test
  test_protocol_dict_roundtrip. The 0.1 C run is unaffected (default = its current). Only resumed lispan_train runs
  were exposed (checked all train.log: only these two; multirate resumes use spec.json 'current'). New 1 C seed-0 run:
  lispan_final_1C_seed0_20261008T215609Z (from scratch, by the queue). paper_build's checkpoint re-evaluation caught
  the same bug in runs.load_model (1 C sepscale run 257 mV vs logged 0.12) - after the fix it reproduces 0.119 mV.
  BUG 2 FOUND + FIXED: FV least squares (lispan_fit_experiment.py, lispan_span500.py) used scipy's relative FD steps,
  which collapse to 1.5e-8 at x = 0 (nominal log-multipliers / offsets) - below the BDF tolerance, Jacobian = noise.
  New src/dfn_pinn/lispan/fdjac.py (absolute steps 0.01, backward at bounds; test). SPAN500 fit 12.2 -> 10.0 mV; the
  Fig. 4b fit is rerun as fit_0.1+1C_fdabs.json (the old 18.4 mV "flat valley" result must be re-checked).
- 2026-10-08 ~22:55 UTC container restart (second of the day): queue relaunched (resumes 0.1 C seed 0 at 12001,
  1 C seed 0 new run at 3001). 0.1 C seed 0 after the inventory switched on: 2.72 mV @10000 -> 0.33 mV rms / 0.90 max
  @12500 (from scratch, so the recipe works). FV fits now checkpoint their best point (fdjac.FitCheckpoint; --resume
  in lispan_fit_experiment.py and lispan_span500.py; ckpt_*.json next to the results; the two interrupted fits were
  resumed from checkpoints reconstructed from their logs, noted inside the ckpt files) and use ftol = xtol = 1e-4
  (BDF solutions are ~1e-6 accurate; scipy's 1e-8 made them run to max_nfev).
- SPAN500 C/10 cycle 3 DONE (LI_SPAN 6.9): paper kinetics 10.0 mV rms (w_S 0.376, U0 2.088/1.831/1.436 V, b
  0.222/0.262/0.545 V), k0 x1000 11.3 mV (w_S 0.396); cycle 2 with the same parameters 55 mV; cycle 2 vs 3 of the data
  54 mV rms (cycle 2 47 mV lower) -> calibration is cycle-specific; 1 C predictions differ up to 135 mV in plateau 1 (95 at 50 mAh/g)
  only (k0_2,3 x1000 = exact U0 shift for irreversible steps); both 393 mAh/g_SPAN at 1 C.
- Manuscript draft started: paper/manuscript/manuscript.md (methods written; numbers as {{key}} placeholders filled by
  paper_build from numbers.json -> manuscript_filled.md, unresolved keys reported). paper_build --forward-runs
  (stand-in runs) now writes *_TEST outputs and no numbers.
- 2026-10-09 ~01:00 UTC: STEP 1 SEED 0 DONE (from scratch, final recipe, 30 000 steps): 0.1 C 0.164 mV rms / 0.74 max
  (lispan_final_0.1C_seed0_20261008T005207Z; 0.33 mV at 12 500 right after the inventory switched on), 1 C 0.225 /
  0.77 mV, c_e 1.5 mol/m3 rms (lispan_final_1C_seed0_20261008T215609Z; 9.1 @5000, 2.3 @12500, 0.83 @17500). Seeds 1
  started by the queue (0.1 C seed1 ..._20261009T011304Z, 1 C seed1 ..._20261009T014339Z).
  NOTE container reclaimed whenever the session idles a few minutes (restarts at ~20:55, 22:55, ~23:15, ~00:00 UTC):
  background jobs only progress while a turn is active -> keep polling with sleeps (<= 10 min per call), relaunch the
  queue after any restart.
- STEP 2 (cost) DONE: results/paper/cost_table.{json,md}: FV discharge 3.9-6.9 s CPU; PINN training 1.66 / 1.72 h CPU
  projected from measured 187-211 ms/step (2.2-2.8 h wall with the machine shared), < 1 mV from step 12 500 (0.1 C) /
  17 500 (1 C); trained PINN voltage at 1000 times ~50 ms; 8p inverse FV least squares 68 solves/rate = 0.22 h CPU vs
  PINN 1.94 h (+ 5.0 h forward pre-training). `--md-only` rewrites the .md from the JSON.
- Ideal estimator (nonlinear FV LS, fit_0.1+1C_nominal.json): k0 +1.58/+1.10/+0.07 %, b +0.04/+0.30/+0.02 %, U0_1
  -0.69 mV, Z_CC +0.04 % (= linearized values). PINN-LM vs ideal: max 2.0 CRLB (k0_2), median 0.7.
  scripts/paper_inverse_error_model.py: PINN forward error at its estimate 0.15 / 0.71 mV (0.1 / 1 C) predicts the
  PINN-ideal differences (k0_2 -1.08 vs -1.41 %, U0_1 -0.27 vs -0.31 mV, ...) -> the PINN inverse = ideal + bias of
  its forward error. runs.load_multirate added.
- Fig. 4b measured fit rerun with absolute FD steps (fit_0.1+1C_fdabs.json): 18.38 mV, same valley/params as before
  (k0 1.06/0.74/3.51, b_2 0.60, U0_1 +27 mV, Z_CC 0.033) -> old conclusions confirmed.
- LI_SPAN 6.5 corrected: four-rate CRLB kappa0 72 %, t_plus 48 %, kappa_SPAN 15-17 % (not "70-190 %").
- paper_build: all manuscript placeholders resolve (forward means over the finished seeds).
- 2026-10-09 ~06:30 UTC: ALL FIVE PAPER STEPS DONE. Step 1 (3 seeds, from scratch): 0.1 C 0.164 / 0.435 / 0.177 mV
  rms (mean 0.26), 1 C 0.225 / 0.137 / 0.179 (mean 0.18) (LI_SPAN 6.4 table). Full paper_build: 190 numbers, 65 input
  files, manuscript draft 1 with 61 placeholders all resolved. Cost table re-measured on the idle machine (FV 4.4-7.2 s,
  PINN 1.87 / 1.93 h CPU, FV 8p inverse 0.24 h; ~10-15 % run-to-run timing variation on this cloud CPU). Stand-in
  test outputs moved to paper/_test_standin/.
- Seeds 1 DONE: 0.1 C 0.435 mV rms / 3.54 max (the max is a spike at the phase 2/3 hand-over, Q ~ 915; 0.35 mV rms
  without it), 1 C 0.137 / 0.63 mV. Seeds 2 running (queue).
- SPAN500 PINN inverse DONE (LI_SPAN 6.9): needed (i) a data/PINN window that excludes the fitted model's cut-off
  collapse (rate-spec "t_end" option added to lispan_inverse_multirate.py), (ii) a stage-1 re-adaptation at the
  measured current (configs/lispan_forward_span500_stage1.json), (iii) param_log_bound 5 (U0_3 had hit -0.2 V). Final
  lispan_inv_span500_c3_paper_wS_b5_20261009T044900Z: FV at its parameters 8.9 mV vs FV optimum on the same window /
  w_S 8.2 mV (scripts/lispan_span500_window_fit.py). Earlier attempts kept with NOTE/STOPPED files.
- Python 3.10 parse check of all scripts/src on the owner's device VM: OK (avoid PEP 701 nested f-strings).
- Job queue: scripts/run_paper_queue.sh (idempotent; seeds 0-2 x 2 rates + SPAN500 PINN inverse; after a container
  restart just relaunch `nohup bash scripts/run_paper_queue.sh >> results/paper_queue.log 2>&1 &`). NOTE run C (lispan_f01C_w96_inv) was warm-started (its first loss is the new
  inventory term only) - the final-recipe runs are the honest from-scratch numbers. Seeds 1, 2 next
  (`--set seed=1 --name lispan_final_{cr}C_seed1`).
- Step 4 DONE: scripts/paper_assumption_checks.py -> results/paper/assumption_checks.{json,md,png} (V compared at
  equal delivered charge). Benchmark vs: reaction 2 reversible 2.66/0.12 mV rms (0.1/1 C); all reversible (TTT)
  53 mV rms, 190 max at 0.1 C (1 C solve does not finish -> skipped); c_DL 0.1 F/m2 4.6/3.4 mV rms, <= 12 mV, +6
  mAh/g (100 F/m2 of cell with a = 1e7 1/m); 1 s ramp <= 0.06 mV rms; grid 20/10 <= 0.08; Z_CC 0.035 = 10/100 mV;
  c_sat, c_S,ref x10 <= 0.01 mV; kappa -10 % 0.3/3.0; D_salt -20 % 0.1/1.0; T +10 K (RT/F only) 0.8/2.9 mV rms;
  salt excursion 1.8/13.5 %. Written up in docs/PAPER_ASSUMPTIONS.md (inherited A1-A7, closures B1-B8, benchmark
  C1-C5, inverse/data D1-D5, not claimed).
- Step 2 (cost): scripts/paper_cost_table.py (FV CPU timings, PINN training from histories + measured CPU per
  step -> projected CPU h, evaluation ms, inverse costs) + src/dfn_pinn/lispan/runs.py (load_model, history,
  time_to). Needs the FV synthetic 8p fit: scripts/lispan_fit_experiment.py now has --data nominal --noise-mV
  --noise-seed --start --grid (same noise draw/window as the PINN multirate run; reports errors + CRLB on the actual
  points) -> results/lispan/fit_experiment/fit_0.1+1C_nominal.json (running). Trained PINN voltage at 1000 times:
  ~0.1 s.
- Step 5 (SPAN500, Nat. Mater. 2026 Fig. 4a C/10 cycle 3): scripts/lispan_span500.py (Simanjuntak geometry and
  transport kept; Z_CC 2.2e-4 Ohm m2 from the EIS HF intercept; w_S (effective S fraction) fitted, sets the model
  current ~1.28 A/m2; U0_1..3, b_1..3 fitted; two kinetic hypotheses k0 x1 ("paper") and x1000 ("fast"), each
  predicts 1 C). Nominal 80 mV rms; "paper" fit ~12 mV rms (w_S 0.344, U0 -107/-60/-102 mV, b x0.68/0.68/0.83).
  Writes results/lispan/span500/fit_cycle3.{json,png,log} and V_span500_c3_<hyp>.npz for a PINN inverse. Methods
  facts: 12 mm coin cells, Celgard 2320, ~2.0 mg/cm2, SPAN:SP:SA 8:1:1, C/10 = 60 mA/g_SPAN, ether electrolytes;
  SPAN500 = shortest chains (model's S4 chain is phenomenological there).
- Step 3: scripts/paper_build.py -> paper/figures (fig1..fig6, figS1), paper/tables (T1-T5), paper/numbers.json,
  paper/MANIFEST.json (sha256 of inputs + code); recomputes PINN metrics from final.pt; --recompute reruns the FV
  scripts; --only <sections>; --forward-runs JSON override. Tested: reference, identifiability, inverse, tables.
- Owner's home has a folder "SPAN" (not connected) - ask whether it holds measured data.

### 2026-10-09: results summary for the advisor review (DONE)
- report/results_summary/PINN_LiSPAN_results_summary.docx (24 pages incl. title + TOC, English, student-guide
  language): executive summary, context, methodology, phases + problems solved, results 4.1-4.7, analysis (error
  hierarchy), conclusions, paper status + decisions for the advisor, limitations, readiness checklist (Table
  "tabready"), 12-slide outline, key numbers. Fact-checked against paper/numbers.json and docs (22 corrections
  applied, e.g. Tafel equivalence k0 x1000 = U0 + 0.355 V; TTT 53 mV = steps 2+3 reversible; CRLB 1.02 vs 1.04 %
  = actual points vs 100-point design). Build: `cd report/results_summary/build && NODE_PATH=$(npm root -g) node
  build.js` (equations: render_equations.py from equations.json; figures A/B: ../make_figures.py). TOC must be
  updated in Word (F9). Delivered in chat; NOT yet copied to the owner's folder (device offline at delivery).
- Readiness answer given to the owner: enough to approve writing; before submission need literature + references +
  novelty, agreed framing (cost result, SPAN500 placement), SI assembly; optional: Danner confirmation, multi-rate
  SPAN data.

### 2026-10-10: short advisor summary + chemical-information study (DONE)
- Owner asked for a SHORT, simple, visual summary (the 24-page results summary was too long), narrative: PINN-DFN on
  synthetic data -> LG M50 -> Li-SPAN model -> validated vs Simanjuntak -> independent Nat. Mater. data -> use the
  tool for chemistry. Deliverable: report/advisor_summary/PINN_LiSPAN_advisor_summary.docx (13 pages, English).
  Build: `python report/advisor_summary/make_figures.py` (drawings a*, results r*, chemistry c*) ->
  `python report/advisor_summary/collect_numbers.py` (numbers.json; no hand-typed numbers) ->
  `cd report/advisor_summary/build && NODE_PATH=$(npm root -g) node build.js`. Fact-checked by a subagent (18 items
  fixed, e.g. step 1 releases no S2-; Z_CC: paper text 0.035, Table 2 0.025; the CRLB column is a 1-sigma bound).
- CLARIFIED TO OWNER: no MEASURED LG M50 data were ever used (LG M50 = Chen2020 parameters, curves from PyBaMM,
  aged cells synthetic). Questions answered (Spanish): why DFN first (validate where the answer is known; lessons;
  credibility); inverse PINN is NOT faster for 1-D (FV fit 14 min vs PINN ~2 h + pretraining; a trained inverse PINN
  is tied to its data set; speed comes with parametric PINNs / expensive 2-D-3-D solvers).
- New data: Nat. Mater. SI Fig. 21 (SPAN300/320/500, cycles 1-2, C/10) digitized by
  scripts/digitize_natmater_si_fig21.py -> data/span_natmater2026/digitized/ (1.65 mAh/g x 5.7 mV per pixel; cycle-2
  monotonic filter; SPAN500 vs source data ~10 mV rms). EIS HF intercepts (Im = 0): SPAN300 8.33, SPAN320 5.20,
  SPAN500 2.05 Ohm (12 mm disc). Source data Fig. 4e: CE cycle 1 SPAN500 73.55 %, SPAN300 81.6, SPAN320 76.1 %.
- scripts/lispan_span_chem.py (FV fit w_S, U0_1..3, b_1..3, PINN window data + stage-1/inverse configs, window FV
  fit) -> results/lispan/span_chem/fit_<label>.json; scripts/run_chem_queue.sh (stage 1 + TR inverse per label;
  gn_max_step 1.0 = 0.1 V per TR step, needed for shifts of 0.3-0.6 V; SI runs adam_steps 5000). Runs
  lispan_chem_{s500_c1,s500_c2,si300_c2,si320_c2,si500_c2}_{stage1,inv}_*; first s500_c2_inv (gn_max_step 0.3) has
  STOPPED.txt; s500_c1_stage1 has NOTE.txt (a duplicate resume was killed after 8 s). Superseded first SI fits in
  results/lispan/span_chem/superseded_monotonic_filter/.
  Results (PINN, misfit vs best FV on the same window; U0 in V; w_S from FV):
  SPAN500 c1 11.6 vs 10.6 mV, U0 1.561/1.521/1.213, w_S 0.543; c2 8.9 vs 8.3, 2.031/1.811/1.306, 0.393;
  c3 (paper case) 8.9 vs 8.2, 2.094/1.851/1.445, 0.376; SI SPAN300 c2 8.6 vs 7.7, 2.129/1.872/1.545, 0.519;
  SPAN320 10.9 vs 10.0, 2.100/1.881/1.452, 0.446; SPAN500 8.75 vs 8.2, 2.011/1.803/1.311, 0.395.
  Reading: first discharge 470/291/92 mV lower than cycle 2 (step 1 most); w_S drop 28 % ~ 26 % first-cycle loss;
  SPAN300 -> SPAN500 steps down 118/68/234 mV (step 3 most), w_S 52 -> 40 wt% (effective; follows capacity).
- scripts/lispan_rate_analysis.py -> results/lispan/span_chem/rate_analysis.{json,npz}: Simanjuntak cell calibrated by
  the Fig. 4b PINN inverse (Z_CC 0.0344, k0 x2.2/1.7/4.5): 1 C loss vs 0.05 C 470 mV = Z_CC 344 + kinetics 136 +
  electrolyte 32 (one-at-a-time removal, overlapping); capacity 1253 -> 1003 mAh/g_S, 60 % of chains stop at PAN-S2Li
  at 1.0 V (no Z_CC: 1201; fast kinetics: 1143); Li2S uniform (2 %), salt +-13 %; PINN fields vs FV 0.15-0.31 mV.
  Caveat (LI_SPAN 6.8): only Z_CC is firmly determined by the 30 digitized points; other shares are trends.

### 2026-10-11: Joshua's deck + email to Dr. Balbuena
- Joshua Tran made "Li-SPAN PINN Results Summary.pptx" (11 slides, TAMU maroon style). Claude added slides 10
  ("Nat. Mater. SPAN Results": figure d1 = report/advisor_summary/make_figures.py d1_span_chem_deck) and 11
  ("SPAN Chemistry: Conclusions": 4 stat cards + next-step bar), fixed slide 3 (table frame too short), pointed
  slide 9 to Figure 8, updated key point 4 on the last slide. Script: report/advisor_summary/deck/build_new_slides.py.
  Output: PINN-DFN-Project/Li-SPAN_PINN_Results_Summary_v2.pptx (owner's folder). Short English email to
  Dr. Perla Balbuena drafted in chat.

## Next steps (paper)
1. Owner + advisor review of paper/manuscript/manuscript_filled.md and paper/figures; decide JES format (Word/LaTeX).
2. Complete the reference list (PINN / PINN-DFN literature from the owner's list; do not invent citations).
3. Supplementary information: full equation set (LI_SPAN 4-5), Table S1 = T1, assumptions doc.
4. Optional: ask T. Danner (DLR) for the MATLAB implementation to confirm the closures B1-B4; multi-rate SPAN data
   from the owner (the folder "SPAN" in his home may hold some) would turn the SPAN500 case into a kinetic calibration.

## Older next steps (DFN line, paused by the Li-SPAN-only scope decision)
1. Forward: seeds 1-2 of f4_hardbc_w96_40k (reproducibility of 0.25 mV); then 2C / C/2 forward models of the same
   quality for the multi-rate inverse (k_p, sigma_p, electrolyte pair).
2. Inverse, hierarchical: forward fields per rate first, then D_p/k_n, then D_n/k_p/sigma_p/electrolyte, each
   released when the forward error at the relevant rate is below the noise (param_release_steps / --init).
   Report estimates with the error model (forward error / sensitivity) next to the CRLB.
3. Finish the C-F factorial table (V2_RESULTS.md section 5) and copy run folders to the owner.
4. Other C-rates, pulse protocol, LG M50 experimental data (Chen 2020) - data source to be decided by the owner.
5. Li-SPAN: reference and forward PINN DONE (1.0 mV rms at 0.1 C, above). Next: (a) push the forward error below
   0.5 mV (ODE-group weight / EMA of adaptive weights / L-BFGS tail / 1 C run with configs/lispan_forward_01C_w96.json
   as template); (b) sensitivities/identifiability of {U0_m, b_m, k0_m, K_sp, Z_CC, beta} with the FV reference
   (finite differences) and the error model of V2_RESULTS 3; (c) inverse on synthetic 0.1 C + 1 C curves
   (LiSPANTrainConfig has the inverse fields: inverse_params, data_path). Ask T. Danner (DLR) for the MATLAB code to
   confirm the 5.2 items.
6. Step C: synthetic aging DONE with the consistent model (above). Next: (a) longer stage 2 / lower lr and the
   hierarchical release (R0, eps_am_p first, then theta_p0, then the (theta_n0, eps_am_n) pair) to trim the 1.5 %;
   (b) cellB cycle 200 (cracking: a_n changes with the crack area, so eps_am and the area no longer coincide -
   decide what the PINN's eps_am_n means there); (c) add a low-rate discharge or the CV tail of the same cycle as a
   second data set to split LLI from LAM_n; (d) the real LG M50 data set (reader ready: v2_import_lgm50.py) once
   the owner shares one cell's files and the citation - the OKane-consistent config (configs/v2_inverse_aging_okane.json)
   is the one to use, with a fresh forward model trained with it first (the section-9 model has Chen2020 sigma and
   no D(theta); stage 1 re-adapts it, 4000 steps were enough here).

## 2026-10-12: advisor meeting feedback (Dr. Balbuena)
- Feedback: she finds it hard to see the application of the paper / what the project would be used for.
- Assessment: valid. As framed, the work answers "can a PINN solve and fit the Li-SPAN model?" (yes, ~0.2 mV) but not "what problem does it solve?"; for a 1-D cell the FV solver is faster, and the chemistry study confirms/quantifies the Nature Materials conclusions.
- Proposed reframings (NOT decided, awaiting owner/advisor input):
  1. DFT <-> cell bridge (preferred): group computes reduction potentials of PAN-bound S chains (S8/S6/S4/S2, S=C) with DFT; model with a chain-length mix uses them WITHOUT fitting and is tested against SPAN300/320/500 (Nat. Mater.); the inverse tool supplies per-step potentials from real curves. Compare differences (robust on both sides), not absolute values.
  2. Diagnosis of collaborators' cells from two discharge rates (contact resistance vs kinetics vs active sulfur).
  3. Parametric PINN for many curves (long cycling/aging); 2-D/3-D electrodes where FV is expensive.
- Suggested question for the advisor: which number would she like to extract from a discharge curve, and does the group already have DFT potentials for SPAN.
- Checked (2026-10-12): the Nat. Mater. paper's DFT (Gaussian 16, B3LYP/6-311G(d,p), SPAN320 models; calculations by Y.Z., D.K., P.B.) gives only (i) dG = -29.26 eV for the long- to short-chain transformation during synthesis (SI Fig. 11) and (ii) DOS with/without pi-pi stacking, 0.09 -> 0.20 states/eV near E_F (SI Fig. 20). NO reduction potentials of SPAN-bound chains. The "~2.4 V (S8/S6 2-) vs ~2.1 V (S4 2-)" values are generic Li-S literature for dissolved polysulfides, not computed for SPAN. Our fitted U0 (SI Fig. 21, cycle 2): SPAN300 2.129/1.872/1.545, SPAN320 2.100/1.881/1.452, SPAN500 2.011/1.803/1.311 V. So the DFT bridge needs new calculations, but can reuse the group's existing SPAN320 models.
- Owner decisions (2026-10-13) on application options: (5) electrode-thickness design for Battery500 DISCARDED; (2) "what is worth computing" map kept IN RESERVE; (1)+(4) SPAN curve reader as an INTERNAL repository tool (not a public web app: anything Battery500 is confidential); (3) DFN-PINN aging/SEI with public data is the owner's favourite (most computationally justified). Plan: present options to Dr. Balbuena and let her decide; JES methods paper possibly in parallel.
- Reliability plan for the SPAN reader (proposed): fit-quality gate, per-parameter uncertainty gate, synthetic "trap" tests (simulate effects the model lacks, check bias of w_S and dU0), compare only within electrolyte/rate groups, blind test against independently measured S content; must beat the naive reading (capacity + mean voltage) or it adds nothing.
- Nat. Mater. source data Fig. 4e = capacity vs cycle only (S8/300/320/500, up to ~1000 cycles); no per-cycle voltage curves, so not usable for SPAN aging diagnosis.
- Owner draft proposal (2026-10-13): "Physics-Informed Learning for Reliable Parameter Identification and Predictive Modeling of Li-SPAN Batteries" (JES): identifiability vs number/choice of C-rates, FV vs PINN vs parametric surrogate benchmark, validation on Wang 2026 data, experimental recommendations. My review: good reframing; suggested (1) quantitative falsifiable hypothesis (Tafel degeneracy: k0 x f == U0 + (2RT/F) ln f; C/10-only fits differ ~135 mV at 1 C), (2) add published SPAN rate-capability curves (>=3 rates, same electrode) for a real predictive test, plus fit-0.1C-predict-1C on Simanjuntak Fig. 4b, (3) experiment-ranking figure (CRLB per parameter vs experiment set), (4) bounded parametric surrogate with a break-even number of fits, (5) position vs prior identifiability work (e.g. Forman 2012 DFN/LFP Fisher; Bizeray 2019 SPM), (6) lead with the application question in title/abstract. Estimated remaining work ~3 months incl. writing.
- Revised JES proposal written as a Claude Doc (2026-10-13): "How Many Discharge Curves Are Needed to Characterize a Li-SPAN Cathode? Identifiability and Prediction with Physics-Informed Learning" - https://claude.ai/code/artifact/e3aeee29-7cf8-447f-879a-02540639bc44 (summary, testable hypothesis with Tafel degeneracy, objectives, data table, tasks 1-5, status tracker, risks, 12-week roadmap with gates at weeks 2 and 5, aging/SEI follow-on, references; Forman 2012 and Bizeray 2019 cited from memory, to verify).

## 2026-10-14: Phase 1 of the identifiability proposal (done)
- Scripts: scripts/lispan_phase1_fisher.py (linearized Bayesian identifiability, 14 params, rates 0.05-1 C; 2 C left out because the FV solver stalls at 2 C for b_2 +-5 %), scripts/lispan_phase1_fit.py (FV calibrate-then-predict, optional fixed params), queues scripts/run_phase1_fits.sh and run_phase1_fits_2.sh. Results results/lispan/phase1/; figures report/phase1/make_figures.py -> p1_measured.png, p2_ranking.png (+ _sigma15mV), p3_synthetic.png; note docs/PHASE1_IDENTIFIABILITY.md. Verified by an independent subagent check (fixed: synthetic 2 C range 236-306 mV, not 198).
- Findings: (1) measured Simanjuntak cell: five C/10-only calibrations fit 0.1 C at 15.2-15.3 mV but miss 1 C by 32-318 mV (free Z_CC -> 0.0107 Ohm m2, 217 mV; only Z_CC = 0.035 assumed gives 32 mV); 0.1+0.2 C also misses 1 C by 217 mV (Z_CC 0.0079); 0.1+1 C: Z_CC 0.0331, predicts 0.05/0.2 C within 30/31 mV. (2) synthetic (5 mV noise): C/10-only fits within 0.7 mV at 0.1 C miss 1 C by 146-278 mV and 2 C by 236-306 mV; C/10+C/5 predicts 1 C to 6.2 mV; C/10+1C recovers all multipliers within 1.5 % (U0_1 within 0.6 mV) and predicts 2 C to 1.3 mV. (3) Fisher (5 mV): 1 C prediction sd 170 mV (C/10), 7.9 (C/10+C/5), 2.5 (C/10+C/2); Z_CC post/prior 0.82 (C/10) -> 0.05 (C/10+1C); k0 exp(F U0/2RT) of steps 2-3 identified (0.31 -> 0.026-0.034 with two rates) but k0, U0 separately never (>= 0.43, corr <= -0.996); electrolyte transport never identified (>= 0.46).
- Hypothesis in the proposal doc REVISED: k0-U0 degeneracy of irreversible steps holds at every current (no rate set separates them; only the combination matters); a fast second rate fixes contact resistance and predictions. Proposal doc updated (hypothesis, status table, Phase 1 results subsection with 2 figures).
- Next (Phase 2): literature SPAN rate-capability datasets (owner to collect PDFs); Gate 1 = advisor review.

## 2026-10-15: Phase 2 - assessment of five owner-supplied files + literature pilot
- Files: Moschner et al. Adv. Sci. 12, 2415436 (2025) + SI docx; Ma et al. Small 22, e74169 (2026); Hao et al. ACS Nano
  19, 25385 (2025); Chen et al. Nat. Commun. 10, 1021 (2019). Note: docs/PHASE2_LITERATURE_DATA.md (fact-checked by a
  subagent; corrections applied). Data: data/literature_span/ (README): chen2019 (Fig. 3b/3d, Se0.06SPAN, ether and
  carbonate, 0.4-10 A/g, read EXACTLY from PDF vector paths, scripts/extract_chen2019_fig3.py) and ma2026 (Fig. 6c/6d,
  SPAN and SPAN aerogel, carbonate, 0.1-2 C, digitized, scripts/digitize_ma2026_fig6.py; 0.1 C = first cycle).
- Verdicts: Chen 2019 and Ma 2026 USE; Moschner 2025 (same group as the model: Danner/Simanjuntak/Latz) not a rate
  dataset (C/50, C/10 only; rate tests as capacities in SI Fig. S8) but best for interpretation (full model parameter
  tables SI S4-S6; EIS; resistive-layer simulations = Z_CC-like signature) -> ask authors for raw C-rate curves;
  Hao 2025 low priority (FeS2/SPAN only in main text; FeS2 not in model). Moschner cites Perez Beltran & Balbuena
  (JPCC 125, 13185 (2021); ACS AMI 13, 491 (2021)) - check whether they contain lithiation voltages.
- Pilot (scripts/lispan_lit_rates.py; effective model = Simanjuntak geometry, w_S basis mapping; fitted w_S, U0_1..3,
  b_1..3, k0_1, Z_CC; coarse 10/5 grid): one slow rate fitted to 7-10 mV misses faster rates by 58-433 mV (all 4
  datasets). Chen: slowest+fastest (0.4+10 A/g) predicts intermediates within 33-50 mV; all-six fits 34/25 mV rms
  (carbonate/ether; model limit); slow+medium (0.4+2) interpolates 1 A/g (25-29 mV) but extrapolates 4-10 A/g badly
  (37-147 mV): two rates cannot split log (kinetic) vs linear (ohmic) loss (l3_split: 36/31 mV per e-fold + 40/56 mV
  per A/g). Ma: 0.2 C alone fits to 8 mV; 0.2+1 C cannot be fitted (49-73 mV) - cells lose capacity with rate; forward
  check (transport_check_ma2026_SPAN-A.json): salt diffusivity x0.1/x0.03 cuts 1 C capacity 1037 -> 899/486 at
  unchanged voltage (measured 686) -> transport time scale ~10-30x slower than the model default (electrolyte in a
  thicker electrode, or solid-state in SPAN particles). Fits with D_salt free STOPPED (salt-depleted solves too stiff;
  NOTE.txt). Two all-rates fits were interrupted by a container restart and rerun (NOTE.txt).
- Figures report/phase2/figures/l1_curves.png, l2_errors.png, l3_split.png. Proposal doc updated (data table, hypothesis:
  "fastest rate of interest", status table, Phase 2 pilot subsection with l1 and l3).
- Protocol consequence: calibrate on the slowest + the fastest rate of interest (no extrapolation); a third rate
  splits kinetic/ohmic; the second rate also tests the model's transport. Solver needs robustness for salt depletion.
