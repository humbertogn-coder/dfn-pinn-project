#!/usr/bin/env bash
# Paper job queue (idempotent): rerun after a container restart and it resumes what was interrupted.
#   step 1  seeds 0, 1, 2 of the final forward recipe at 0.1 C and 1 C (two jobs at a time, one per core)
#   step 5  PINN inverse on the SPAN500 C/10 data (after seed 0 of 0.1 C, whose network it starts from)
# A job is: done (final.pt), interrupted (latest.pt -> --resume) or new.  Logs: results/<name>.log
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
RUNS=results/lispan_runs

run_dir() {  # newest non-stopped run folder of a name, or empty
  for d in $(ls -d "$RUNS/${1}_2"* 2>/dev/null | sort -r); do
    [ -f "$d/STOPPED.txt" ] || { echo "$d"; return; }
  done
}

current_dir() {  # run folder of a name after any process still using it has ended (re-read: it may get STOPPED)
  local d
  d=$(run_dir "$1")
  while [ -n "$d" ] && pgrep -f -- "$d" > /dev/null; do sleep 60; d=$(run_dir "$1"); done
  echo "$d"
}

forward() {  # forward <rate> <seed>
  local name="lispan_final_${1}C_seed${2}" d
  d=$(current_dir "$name")
  if [ -n "$d" ] && [ -f "$d/final.pt" ]; then echo "done  $name"; return; fi
  if [ -n "$d" ] && [ -f "$d/latest.pt" ]; then
    echo "resume $name"; python3 -u scripts/lispan_train.py --resume "$d" >> "results/$name.log" 2>&1
  else
    echo "start $name"; python3 -u scripts/lispan_train.py --config "configs/lispan_forward_final_${1}C.json" \
      --name "$name" --set seed=${2} >> "results/$name.log" 2>&1
  fi
}

inverse_span500() {
  local name="lispan_inv_span500_c3_paper" d
  d=$(current_dir "$name")
  if [ -n "$d" ] && [ -f "$d/final.pt" ]; then echo "done  $name"; return; fi
  if [ -n "$d" ] && [ -f "$d/latest.pt" ]; then
    echo "resume $name"; python3 -u scripts/lispan_inverse_multirate.py --resume "$d" >> "results/$name.log" 2>&1
  else
    echo "start $name"; python3 -u scripts/lispan_inverse_multirate.py --config configs/lispan_inverse_span500_paper.json \
      >> "results/$name.log" 2>&1
  fi
}

( forward 0.1 0; forward 0.1 1; inverse_span500; forward 0.1 2 ) &
( forward 1 0; forward 1 1; forward 1 2 ) &
wait
echo "queue finished $(date -u)"
