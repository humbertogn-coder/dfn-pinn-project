# Two-curve fits of the SPAN-DB cells (2026-10-30)

First run of the two-curve tool (docs/TWO_CURVE_TOOL.md) over every SPAN-DB cell with discharge curves at two or more
currents.  Network: `chem_v2_norar_s0` (0.1-1 C).  SPAN-DB snapshot: after batch 10 (38 articles, 415 cells, 1670
curves).

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

## Network v3 (0.08-3 C)

Run started 2026-10-30 into `results/lispan/spandb_v3s1/` (same rules, ~3-4 min per cell).  It adds the fast
currents (Chen 2019 up to 3 C-equivalent, Wu 2023b 2-3 C) and more blind predictions, at the larger v3 error floor
(NETWORK_SD).  Results to be summarized when complete.

## Files

- `scripts/lispan_spandb_fit.py` - the fits (rules above), `--summary-only` rebuilds the summary.
- `results/lispan/spandb/` - run 2 (this document): per-cell JSON, summary.csv/json, run.log.
- `results/lispan/spandb_run1/` - run 1 (before the rules above), kept for reference.
- `results/lispan/spandb_v3s1/` - network v3.
- `report/spandb/make_spandb_figures.py` -> `report/spandb/figures/sdb_{funnel,misfit,examples,predictions,controlled}.png`,
  `report/spandb/numbers.json`.
- `report/seminar/make_seminar_figures.py` -> `report/seminar/figures/` (network accuracy v2/v3, two-curve crops) for
  the group-seminar deck (Claude Slides artifact "PINN for Li-SPAN: group seminar").
