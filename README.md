# DFN PINN Project

Research project to implement and evaluate physics-informed neural networks (PINNs) for the full Doyle-Fuller-Newman model, focusing on:

- reproducible reference solutions with PyBaMM
- a mixed-variable formulation with learned interfacial current
- an inverse Butler-Volmer residual to investigate conditioning
- hard current conservation through projection
- identifiability analysis before parameter estimation

## Current Status

The repository currently contains a minimal project skeleton. The first goal is to establish a reproducible workflow:

1. Local environment
2. Git and GitHub
3. Codex-assisted development
4. HPRC Grace for larger computational runs

## Create the Local Environment

From the project directory in Anaconda Prompt:

```bat
conda create -n dfn-pinn python=3.11 -y
conda activate dfn-pinn
pip install -r requirements.txt
pip install -e .
```

## Verify the Installation

```bat
python scripts/check_project.py
pytest -q
```

## Git Workflow

```bat
git init
git add .
git commit -m "Initialize DFN PINN project"
```

For a new repository, create an empty repository on GitHub and connect the remote. Replace `REPOSITORY_URL` with its clone URL:

```bat
git branch -M main
git remote add origin REPOSITORY_URL
git push -u origin main
```

## HPRC Grace

The project owner's existing Grace environment can be activated as follows. Collaborators should substitute their own environment and checkout paths:

```bash
module purge
module load GCCcore/13.2.0
module load Python/3.11.5
source /scratch/user/humbertogn/ase_env/bin/activate
cd /scratch/user/humbertogn/proyectos/dfn-pinn-project
pip install -r requirements.txt
pip install -e .
python scripts/check_project.py
pytest -q
```

## First Technical Goal

Run the initial Chen2020 isothermal DFN discharge at 1C:

```bash
python scripts/export_pybamm_reference.py --protocol 1C
```

The script exports terminal signals to CSV and HDF5, a voltage plot, and JSON
metadata to a timestamped directory under `results/`. It checks finite outputs,
increasing time, the applied current, and the final voltage cutoff.

This is an initial solver smoke run. Internal-state exports, mesh convergence,
and a conservation audit are still required before using it as a validated PINN
reference. Generated results are excluded from Git.

The setup follows the [official PyBaMM example](https://github.com/pybamm-team/PyBaMM/blob/main/examples/scripts/experimental_protocols/cccv.py).

## Mesh Refinement Check

```bat
python scripts/check_mesh_convergence.py
```

This repeats the same discharge with 20, 40, and 80 points in each spatial
domain, using one-second outputs. It compares voltages at shared sample times
and reports cutoff-time differences relative to the finest tested mesh.
CSV signals, metrics, JSON settings, and a figure are saved under `results/`.
The finest mesh is not an exact solution. This check does not yet establish
internal-state accuracy, solver-tolerance independence, or conservation.

## Solver Tolerance Check

```bat
python scripts/check_solver_tolerances.py
```

This holds the mesh at 80 points per domain and compares `(rtol, atol)` pairs
`(1e-6, 1e-8)`, `(1e-7, 1e-9)`, and `(1e-8, 1e-10)` for the same discharge.
It saves CSV signals and metrics, JSON settings, and a plot under `results/`.
Voltage differences use shared one-second output times; cutoff times are
compared separately. The tightest run is a provisional reference, not an exact
solution. These sampled voltage checks do not certify internal-state accuracy,
conservation, or behavior between output samples.

## Current and Lithium Conservation

```bat
python scripts/check_conservation.py
```

This audits the baseline discharge at 80 points/domain with `rtol=1e-8` and
`atol=1e-10`. Reaction-derived electrode currents are compared with the applied
current, using oxidation-positive signs (+I negative, -I positive). Lithium
inventories are summed across both electrodes and the electrolyte. Individual
electrode changes are also compared with the applied charge divided by Faraday's
constant. The CSV, JSON report, and plot are saved under `results/`.

The audit uses PyBaMM's spatial averages and inventories, not an independent
integration of exported fields. It reports sampled global balance residuals
without imposing a pass threshold. Small balances alone do not establish local
state accuracy or mesh convergence. These equations apply to the current model
without side reactions and must be revisited if degradation is introduced.

## Internal Variable Inventory

```bat
python scripts/inspect_internal_variables.py
```

This inventories 23 primary and auxiliary fields at 80 points/domain using
10-second output samples and tight solver tolerances. It records exact PyBaMM
names, units, domains, native axis order, node/edge coordinates, shapes, and
sampled ranges in JSON and Markdown under `results/`. It checks finite entries
and coordinate lengths. Full field arrays are not exported at this stage.
Particle concentrations use native `(r, x, t)` order. Surface values must be
read separately rather than taken from the last radial cell center.

## Export Full Internal Fields

```bat
python scripts/export_internal_fields.py
```

Exports all 23 inventoried arrays with lossless compression, per-field native
coordinates, units, dimension scales, terminal signals, and simulation metadata.
The exporter reopens the file and checks exact equality against the simulation
before naming it `reference.h5`. Outputs live in a new directory under `results/`.
See [HDF5 schema](docs/HDF5_SCHEMA.md). Successful export verifies data fidelity,
not physical accuracy or mesh convergence.

## Visualize Stored Internal Fields

```bat
python scripts/plot_internal_fields.py
```

Reads the latest `results/internal_fields_*/reference.h5`, prints the chosen
path, and generates three figures without importing or running PyBaMM. Use
`--input PATH_TO_REFERENCE_H5` to select a specific file. New timestamped plot
directories are created beside the input file. A JSON manifest records the
source, actual sampled times, and electrode positions used for radial profiles.
Figures show concentrations and currents at the start, near mid-discharge, and
at cutoff. Radial curves use native centers, not exact particle boundaries.
This visualization is not an accuracy or convergence certificate.

## Internal Concentration Mesh Study

```bat
python scripts/check_concentration_convergence.py
```

Compares 20, 40, and 80 points/domain at fixed tight solver tolerances. It checks
both full particle concentrations, both surface concentrations, and electrolyte
concentration separately in each region. Finer solutions are interpolated
spatially onto the coarse interior nodes, at shared 10-second output times.
No extrapolation or interpolation across electrolyte interfaces is performed.
Outputs include pairwise maximum and unweighted RMS differences, scale-based
percentages, sampled arrays, settings, coordinates, and a figure under `results/`.
The report documents sampling and interpolation limits. No accuracy threshold
or convergence certificate is implied by a successful run.

## Initial Transient Investigation

```bat
python scripts/check_initial_transient.py
```

Runs six short 1C simulations over 120 seconds with half-second output. It
separates radial and through-cell refinement for positive particle and surface
concentrations, including 80-to-160 comparisons. Two interior query grids expose
sampling sensitivity. Reports locate each maximum in time, x and r, and retain
settings and sampled arrays. This diagnostic does not certify the full discharge
or distinguish interpolation error from discretization error exactly.
