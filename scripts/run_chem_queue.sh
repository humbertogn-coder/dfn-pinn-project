#!/usr/bin/env bash
# Chemical-information study (advisor summary, 2026-10-10): PINN inverse on more Nat. Mater. SPAN data sets.
# Per label: wait for the FV fit (scripts/lispan_span_chem.py writes fit_<label>.json with a "window" entry and the
# two configs), stage 1 (forward re-adaptation at the measured current), then the trust-region PINN inverse.
# Idempotent: done (final.pt) / interrupted (latest.pt -> --resume) / new.  Logs: results/<run name>.log
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
RUNS=results/lispan_runs
INIT0=results/lispan_runs/lispan_final_0.1C_seed0_20261008T005207Z/final.pt

run_dir() { for d in $(ls -d "$RUNS/${1}_2"* 2>/dev/null | sort -r); do [ -f "$d/STOPPED.txt" ] || { echo "$d"; return; }; done; }

current_dir() {  # run folder of a name once no process uses it any more (after a queue restart a job may still run)
  local d; d=$(run_dir "$1")
  while [ -n "$d" ] && { pgrep -f -- "python3 -u scripts/.*$(basename "$d")" > /dev/null || pgrep -f -- "python3 -u scripts/.*$1" > /dev/null; }; do sleep 60; d=$(run_dir "$1"); done
  while [ -z "$d" ] && pgrep -f -- "python3 -u scripts/.*--name $1" > /dev/null; do sleep 60; d=$(run_dir "$1"); done
  echo "$d"
}

job() {  # job <name> <script> <args...>   (script = train | inverse)
  local name=$1 kind=$2; shift 2
  local d; d=$(current_dir "$name")
  if [ -n "$d" ] && [ -f "$d/final.pt" ]; then echo "done  $name"; return 0; fi
  local py=scripts/lispan_train.py; [ "$kind" = inverse ] && py=scripts/lispan_inverse_multirate.py
  if [ -n "$d" ] && [ -f "$d/latest.pt" ]; then
    echo "resume $name $(date -u +%H:%M)"; python3 -u $py --resume "$d" >> "results/$name.log" 2>&1
  else
    echo "start $name $(date -u +%H:%M)"; python3 -u $py "$@" >> "results/$name.log" 2>&1
  fi
}

chain() {
  for L in "$@"; do
    until python3 -c "import json,sys; sys.exit(0 if 'window' in json.load(open('results/lispan/span_chem/fit_$L.json')) else 1)" 2>/dev/null; do sleep 60; done
    job "lispan_chem_${L}_stage1" train --config "configs/lispan_chem_${L}_stage1.json" --init "$INIT0" --name "lispan_chem_${L}_stage1"
    local s1; s1=$(run_dir "lispan_chem_${L}_stage1")
    python3 - "$L" "$s1/final.pt" <<'PY'
import json, sys
p = f"configs/lispan_chem_{sys.argv[1]}_inv.json"
c = json.load(open(p)); c["rates"][0]["init"] = sys.argv[2]
# 0.1 V per trust-region step on U0 (default 0.03 V cannot reach shifts of 0.3-0.6 V in 8000 steps; see the STOPPED
# note of the first lispan_chem_s500_c2_inv run)
c["train"]["gn_max_step"] = 1.0
json.dump(c, open(p, "w"), indent=2)
PY
    job "lispan_chem_${L}_inv" inverse --config "configs/lispan_chem_${L}_inv.json"
  done
}

if [ $# -gt 0 ]; then chain "$@"; else
  ( chain s500_c2 si320_c2 si500_c2 ) &
  ( chain s500_c1 si300_c2 ) &
  wait
fi
echo "chem queue finished $(date -u)"
