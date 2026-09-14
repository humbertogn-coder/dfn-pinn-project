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
