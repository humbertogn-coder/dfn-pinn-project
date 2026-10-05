"""Run only the startup representation checks; no training."""

from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    print("Checking particle startup: SI scales, initial value, center, signed flux and neural derivatives.", flush=True)
    print("Positive-time physics only. No training or full DFN validation.", flush=True)
    raise SystemExit(subprocess.call([sys.executable, "-m", "pytest",
        "tests/test_particle_startup_candidate.py", "-q", "-p", "no:cacheprovider"],
        cwd=Path(__file__).resolve().parents[1]))
