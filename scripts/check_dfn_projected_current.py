"""Audit shared hard-current projection; no optimization or DFN validation."""

from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    print('Auditing DFN current projection: physical integrals, query independence, derivatives and shared wiring.', flush=True)
    print('Direct Butler-Volmer unchanged. No optimizer step or physical acceptance.', flush=True)
    raise SystemExit(subprocess.call([sys.executable, '-m', 'pytest',
        str(root/'tests/test_dfn_projected_current.py'), '-q', '-p', 'no:cacheprovider'], cwd=root))
