#!/usr/bin/env bash
# Restart of the two all-rates fits interrupted by a container restart (see results/lispan/literature_rates/NOTE.txt).
set -u
cd "$(dirname "$0")/.."
OUTD=results/lispan/literature_rates
LOG=$OUTD/logs
run() { local tag=$1; shift; python3 -u scripts/lispan_lit_rates.py "$@" > "$LOG/$tag.log" 2>&1; echo "$(date -u +%H:%M:%S) done $tag"; }
run chen_c_all_r2 --dataset chen2019_carbonate --fit 0.4 1 2 4 6 10 --max-nfev 8 --start "$(cat $LOG/chen_c_all_laststate.json)" &
run chen_e_all_r2 --dataset chen2019_ether --fit 0.4 1 2 4 6 10 --max-nfev 8 --start "$(cat $LOG/chen_e_all_laststate.json)" &
wait
echo "$(date -u +%H:%M:%S) queue 4 done"
