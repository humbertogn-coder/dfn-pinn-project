# Two-curve fits of the SPAN-DB cells (2026-10-30)

First run of the two-curve tool (docs/TWO_CURVE_TOOL.md) over every SPAN-DB cell with discharge curves at two or more
currents.  Network: `chem_v2_norar_s0` (0.1-1 C).  SPAN-DB snapshot: after batch 10 (38 articles, 415 cells, 1670
curves).  Refit with the wide-range network v4 on 2026-11-02: section "Network v4" below.

    python scripts/lispan_spandb_fit.py --spandb <span_db folder> --workers 2 --fv-check
    python scripts/lispan_spandb_fit.py --spandb <span_db> --network results/lispan_chem_runs/chem_v3_s1/final.pt \
        --out results/lispan/spandb_v3s1 --workers 2 --fv-check
    python scripts/lispan_spandb_fit.py --spandb <span_db> --summary-only          # rebuild summary.csv/json
    python report/spandb/make_spandb_figures.py --spandb <span_db>                 # figures + numbers.json

## Curve and pair choice

Rules in the script header.  The ones that came from failures of the first run (`results/lispan/spandb_run1/`, kept):

| rule | why (run 1 case) |
|---|---|
| cycle 1 is never used, unless every curve of the test is numbered 1 (per-step numbering) | SPAN's first discharge is an activation with a low plateau; Park 2023 C1/C2/C4 paired a first cycle with a 10th cycle: 83-156 mV |
| rate order: over the common capacity window the mean voltage must fall with current (25 mV tolerance); the worst violator (ties: the slowest) is dropped | the slowest step of a rate test often contains the activation (Li 2023 C1 0.1 C: 157 mV -> 10.6 mV without it; Chen 2024 C1/C2 0.1 C) |
| partial curves (points start beyond 15 % of the end capacity) are dropped when two complete curves remain | Qin 2024 0.05 A/g was digitized only in its tail (the rest hidden under other curves) |
| unreported capacity basis: taken equal to the current basis; both unreported: one unknown basis (w_S not comparable with the S content) | 7 cells were skipped in run 1 |
| fallback pair: the pair whose feasible w_S range is closest to the estimate (end capacity / 1200 mAh/g), widest first | run 1's widest pair pinned w_S far from the estimate (Wang 2024 C1: 72 mV -> 36 mV) |
| predictions scored only inside the network window (T <= 1) | beyond the window the network extrapolates in time and produces spurious dips |

## Results (`results/lispan/spandb/`, figures `report/spandb/figures/`)

- 47 cells -> 35 fitted, 12 skipped: 8 without two usable currents (Li 2023b: areal capacity; Park 2023: only a first
  cycle plus one other; Kakiage 2024: cell capacity), 4 with no pair inside 0.1-1 C (Wu 2023b 2-3 C, Shao 2024,
  Chen 2024 C4 at 0.01-0.05 C).
- **22 of 35 fitted cells within 25 mV** (median 13.9 mV), the level of the model on its own paper's cell
  (Simanjuntak 19.9 mV here, 17 mV in the tool validation).
- **Network vs FV at the fitted parameters** (`--fv-check`): median 1.9 mV, max 4.9 mV over those 22 cells.  Larger
  values (6-28 mV) only for cells the model does not describe (beyond the time window or at box edges).  The misfit
  is the model's, as in the six-dataset validation.
- By electrolyte family (`sdb_misfit.png`):

| family | fitted | <= 25 mV | cells above 25 mV |
|---|---|---|---|
| carbonate | 12 | 7 | Shi 2024 C1/C2 (34, 39), Wang 2024 C1 (36), Ma 2026 C1/C2 (26, 47) |
| dilute ether | 6 | 2 (Chen 2019 C1 DOL/DME 15, Qin 2024 C1 DEE 11) | Wang 2023 (DOL/DME + LiNO3, 82), Jia 2024 C1/C2 (DOL/DME, 48, 38), Wu 2024 C3 (DME, 52) |
| LHCE | 5 | 5 | - |
| HCE | 2 | 1 (Chen 2024 C2, 2,5-DMTHF, 10) | Chen 2024 C3 (THF, 60) |
| fluorinated (commercial) | 3 | 2 | Zuo 2024 C4 (64) |
| other liquid | 1 | 1 (Li 2023, DMMS silane, 11) | - |
| polymer / solid | 6 | 4 (gel, poly-DOL x2, PISE) | Shi 2023b (LLZO, 72), Jin 2024 (hydride, 57) |

  Dilute solvating ethers mostly fail, weakly solvating DEE and the LHCEs fit: consistent with polysulfide
  dissolution, which the model does not contain.  The two solid inorganic electrolytes need polarizations far above
  the model's series-resistance box.

- **Blind predictions** (`sdb_predictions.png`): for the 22 described cells, currents between the two calibration
  currents are predicted at 10-25 mV (0.8-2.2x the fit misfit, 8 predictions); extrapolated currents at 16-55 mV
  (1.3-4.5x, 7 predictions).
- **Same electrode, different electrolyte** (`sdb_controlled.png`; sd = sd_total):
  - Chen 2024: Z = 95 +- 5 Ohm cm2 for the HCE cell vs 0 +- 7 for the LHCE cell.
  - Qin 2024: DEE, DEE/ETFE (LHCE) and LHCE + 10 % Py13TFSI share dU0_2 = 35-36 +- 6 mV; Z = 17, 38, 7 Ohm cm2
    (+- 6-7); w_S 0.68-0.71.
  - Chen 2019 (Se0.06SPAN): d = +8 +- 7 mV (ether) vs -31 +- 6 mV (carbonate), a shift of -39 +- 9 mV.
  - Wu 2024: DBE/TTE vs DME/TTE not interpreted (k0_1 of the DME/TTE cell at the box edge, wide sd).
- w_S is an effective factor: with the network's fixed cell (100 um cathode, 618 um glass fibre) it also absorbs
  transport differences, so w_S > 1 on an S basis (Hu 2023, Huang 2023, You 2024, Simanjuntak with its current
  converted from the electrode basis: 1.25-1.49) does not mean more than the theoretical capacity.  Parameters are
  effective; comparisons within a paper are more reliable than absolute values across papers.

## w_S and the sulfur content

The tool does not measure the S content; w_S is the capacity (and current) scale between the data basis and the model's
sulfur.  Check on the described cells (misfit <= 25 mV):
- capacity per g SPAN (8 cells with a reported S content 37.8-47.6 wt%): w_S follows the S content (r = 0.89) with
  w_S / S fraction = 1.02-1.48 (median ~1.35);
- capacity per g S (9 cells): w_S = 1.14-1.43 in eight, the same factor (Shi 2023a C3: 0.96, at the 25 mV limit with box-edge
  parameters).
So real coin cells deliver ~1.2-1.4x the model cell's capacity at the same normalized rate: a geometry/transport factor
(the network's cell has a 618 um glass-fibre separator), not chemistry.  w_S / S fraction is a relative utilization,
meaningful within a paper (Qin 2024: 1.42/1.44/1.48; Chen 2024 LHCE 1.27 vs HCE 1.02); as an S-content estimator it
would be good to ~+-15 % only.  The S content must come from elemental analysis; a blind test of w_S against it needs
geometry inputs in the network.

## Network v3 (0.08-3 C)

Run started 2026-10-30 into `results/lispan/spandb_v3s1/` (same rules, ~3-4 min per cell).  Interrupted twice by
container restarts; superseded by v4 (below) before it completed, so it is kept as a partial run and not summarized.

## Network v4 (0.08-3 C), 2026-11-02

Same rules and SPAN-DB snapshot, network `chem_v4_warm_salt_s0` (docs/PHASE3B_CHEM_PINN.md, "v4 results"; floor in
`NETWORK_SD`).  45 min on 2 workers (2-3.5 min per cell; v2 46 min, v3 3-4 min per cell).

    python scripts/lispan_spandb_fit.py --spandb <span_db> --network results/lispan_chem_runs/chem_v4_warm_salt_s0/final.pt \
        --out results/lispan/spandb_v4ws --workers 2 --fv-check
    python report/spandb/compare_networks.py            # v2 vs v4 -> report/spandb/figures/sdb_v2_v4.png, compare_v2_v4.json

With the wider range the pair rule picks the slowest and fastest currents inside 0.08-3 C, so 32 of the 35 cells
fitted by both networks are calibrated on a wider pair (only Shi 2023a C3, Shi 2023b C4 and Wang 2023 keep theirs).

| | v2 (0.1-1 C) | v4 (0.08-3 C) |
|---|---|---|
| fitted cells | 35 | 38 (+ Wu 2023b C1/C2 at 2 + 3 C, + Shao 2024 C2) |
| misfit <= 25 mV | 22 | 26 (+ Wu 2023b C1/C2 10.0/10.1, Chen 2024 C3 16.7, Shi 2024 C2 24.4; none lost) |
| median misfit, all fitted | 17.9 mV | 18.4 mV |
| network vs FV at the fitted parameters (cells <= 25 mV) | median 1.9, max 4.9 mV | median 0.74, max 3.4 mV |
| blind predictions scored (cells <= 25 mV) | 15 | 51 |
| between the calibration currents | 8, median 13.9 mV, 8 within 25 | 39, median 19.3 mV, 30 within 25 |
| extrapolated (up to 2.5x) | 7, median 27.5 mV, 3 within 25 | 12, median 25.4 mV, 6 within 25 |

What changes with v4:
1. **More cells and the fast currents.**  Wu 2023b (2-3 C only) is now reachable; most cells are calibrated on their
   slowest and fastest currents (e.g. Chen 2019 0.4 + 2 A/g, Chen 2024 0.5 + 2-2.5 C, Wu 2024 0.25-0.5 + 2.5 A/g), as the
   Phase 2 protocol recommends, and the other currents become blind interpolation tests.
2. **The misfit is the model's, even more clearly.**  The network is 2.5x closer to FV than v2 at the fitted
   parameters.  Cells whose misfit grows with the wider pair (Li 2023 10.6 -> 16.0, Shi 2023a C2 11.7 -> 18.1, You
   2024 11.0 -> 16.3, Ma 2026 C1 25.6 -> 53.2, Jia 2024 C2 38 -> 59 mV) are cells whose rate dependence the model does
   not follow over the wider span (the Phase 2 finding: real cells lose capacity with rate faster than the model).
3. **A low misfit needs a blind prediction to back it.**  Six of the 26 cells within 25 mV miss an in-between current
   by more than 25 mV: Chen 2024 C3 (THF HCE; the 1.5 C curve covers only 32 % of the window, the 1.0 C prediction misses
   by 99 mV), Shi 2024 C2 (42 and 74 mV; 102 mV extrapolated), Wu 2024 C2 (34-52), Hu 2023 (27-42), Wu 2023 C1 (34) and Simanjuntak (26.9,
   marginal).  Proposed reading rule: a cell is described when the misfit is <= 25 mV AND every in-between prediction is
   <= 25 mV: 20 cells (17 with at least one such prediction, 3 without one).
4. **Parameters depend on the rate span.**  For the 22 cells described by both networks (21 on a wider pair with v4),
   the v4 - v2 shift in units of the combined sd_total has a median of 0.4-1.5; within 2 sd: k0_1 88 %, dU0_2 82 %,
   dU0_3rel2 79 %, w_S 73 %, b_12 70 %, b_3 68 %, d 64 %, Z_CC 61 %.  With the network error small (item 2), these
   shifts are model-form error: d, Z_CC and the OCV slopes absorb the rate dependence the model lacks, so they are
   effective values for the rate span used.  Compare cells only on comparable pairs.
5. **The same-electrode comparisons hold** (v4 values, sd_total):
   - Chen 2019 (Se0.06SPAN): d = +24 +- 7 mV (ether) vs -18 +- 7 mV (carbonate), a shift of -42 +- 9 mV (v2: -39 +- 9).
   - Chen 2024: Z = 93 +- 3 Ohm cm2 for the HCE cell vs 27 +- 2 for the LHCE cell (v2: 95 +- 5 vs 0 +- 7): the
     difference holds, the LHCE value moves with the 2.5 C curve.
   - Qin 2024: dU0_2 = 34 / 37 / 35 +- 6-7 mV (v2 35-36); Z = 39 / 47 / 9 Ohm cm2 (+-3; v2 17 / 38 / 7): LHCE + IL
     lowest in both.
6. **Data issue:** Shao 2024 C2 (now inside the range) pairs a curve labelled "CIHP-SPAN" with a "2nd" cycle and
   fits at 227 mV (both curves beyond the window).  The two curves are probably not one cell's rate test; to check in
   SPAN-DB.

## Files

- `scripts/lispan_spandb_fit.py` - the fits (rules above), `--summary-only` rebuilds the summary.
- `results/lispan/spandb/` - run 2 (this document): per-cell JSON, summary.csv/json, run.log.
- `results/lispan/spandb_run1/` - run 1 (before the rules above), kept for reference.
- `results/lispan/spandb_v3s1/` - network v3 (partial).
- `results/lispan/spandb_v4ws/` - network v4 (chem_v4_warm_salt_s0); `report/spandb/compare_networks.py` -> 
  `report/spandb/figures/sdb_v2_v4.png`, `report/spandb/compare_v2_v4.json`.
- `report/spandb/make_spandb_figures.py` -> `report/spandb/figures/sdb_{funnel,misfit,examples,predictions,controlled}.png`,
  `report/spandb/numbers.json`.
- `report/seminar/make_seminar_figures.py` -> `report/seminar/figures/` (network accuracy v2/v3, two-curve crops) for
  the group-seminar deck (Claude Slides artifact "PINN for Li-SPAN: group seminar").
