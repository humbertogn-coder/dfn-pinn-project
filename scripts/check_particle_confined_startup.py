"""Run bounded analytic startup checks; no training."""

from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    print("Checking confined startup: linear-time bulk, surface flux signs and neural derivatives.", flush=True)
    print("No training; no diffusion or inventory acceptance implied.", flush=True)
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest",
        "tests/test_particle_confined_startup.py", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).resolve().parents[1]))
