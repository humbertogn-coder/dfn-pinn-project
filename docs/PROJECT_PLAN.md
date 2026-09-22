# DFN PINN Project Plan

## Goal

Implement a reproducible research codebase for the full isothermal Doyle-Fuller-Newman battery model and evaluate whether a mixed-variable inverse Butler-Volmer PINN improves conditioning, conservation, and inverse parameter estimation.

## Phase 0: Repository and Environment

Acceptance criteria:

- package imports locally
- `python scripts/check_project.py` runs
- `pytest -q` runs
- repository is pushed to GitHub
- same smoke check runs on HPRC Grace

Estimated time: 1-3 days.

## Phase 1: PyBaMM Reference Exporter

Acceptance criteria:

- run PyBaMM DFN with Chen2020 parameters
- export voltage, time, protocol metadata, and selected internal states
- save reproducible HDF5 output
- include a conservation/sign audit

Estimated time: 1-2 weeks.

## Phase 2: Baseline PINN

Acceptance criteria:

- simple modular MLPs run on a small controlled problem
- residual scaling utilities are tested
- direct Butler-Volmer baseline trains without code crashes
- metrics compare PINN output against PyBaMM reference data

Estimated time: 2-4 weeks.

## Phase 3: Inverse Butler-Volmer and Projection Ablations

Acceptance criteria:

- implement direct BV, learned-j direct BV, learned-j inverse BV, and projected inverse BV variants
- report voltage/state error, conservation error, failure rate, and gradient diagnostics
- run multiple random seeds

Estimated time: 4-8 weeks.

## Phase 4: Identifiability-Aware Inverse Problem

Acceptance criteria:

- sensitivity screening selects a small parameter subset
- local Fisher/SVD analysis flags non-identifiable combinations
- inverse recovery uses synthetic data with noise and held-out protocols
- compare with numerical solver or least-squares baseline

Estimated time: 4-8 weeks.

## Phase 5: Experimental Validation and Paper Figures

Acceptance criteria:

- evaluate open LG M50 data
- report mV-scale voltage residuals
- generate paper-ready figures
- document limitations and pivot criteria

Estimated time: 4-8 weeks.

## Practical Estimate

- MVP functional: 3-6 weeks
- strong research result: 3-5 months
- serious publishable result: 5-8 months

## Immediate Next Ticket

Status reviewed 2026-09-20: Phase 2 is in progress. A working PyBaMM reference,
HDF5 pipeline, constitutive/kinetic/projection primitives and spherical residual
tests exist. A no-flux particle PINN was tested with five seeds. Switched-flux
particle training still has startup accuracy tradeoffs. No coupled DFN PINN has
been trained; implemented projection/kinetic primitives do not complete Phase 3.

The next controlled experiment targets thin early surface layers while keeping
512 collocation points and preserving early-core coverage. Do not indefinitely
tune one synthetic case: establish an explicit accuracy budget and a go/no-go
review before coupling electrolyte and charge residuals.

Remaining effort estimates from this status, assuming regular researcher work:
roughly 4-8 weeks for a first coupled DFN PINN prototype, 3-5 months for a strong
comparative/inverse study, and 5-8 months for a manuscript-ready study including
experimental evaluation. These are planning ranges, not calendar promises or
computed completion percentages. Convergence, identifiability and data access
may extend them. HPRC research deployment remains to be verified separately.
