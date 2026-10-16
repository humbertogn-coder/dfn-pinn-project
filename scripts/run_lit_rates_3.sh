#!/usr/bin/env bash
# Phase 2 pilot, revised queue (replaces the remainder of run_lit_rates.sh / run_lit_rates_2.sh, stopped after the
# Ma 2026 slow + fast fits; see results/lispan/literature_rates/NOTE.txt). Lane A waits for PID $1, lane B for $2.
set -u
cd "$(dirname "$0")/.."
OUTD=results/lispan/literature_rates
LOG=$OUTD/logs
run() { local tag=$1; shift; python3 -u scripts/lispan_lit_rates.py "$@" > "$LOG/$tag.log" 2>&1; echo "$(date -u +%H:%M:%S) done $tag"; }
(
  while kill -0 "$1" 2>/dev/null; do sleep 20; done
  run chen_c_slow    --dataset chen2019_carbonate --fit 0.4
  run chen_c_bracket --dataset chen2019_carbonate --fit 0.4 10 --start-json $OUTD/chen2019_carbonate_fit0.4+2.json
  run ma_A_slow      --dataset ma2026_SPAN-A --fit 0.2
  run chen_c_all     --dataset chen2019_carbonate --fit 0.4 1 2 4 6 10 --max-nfev 15 --start-json $OUTD/chen2019_carbonate_fit0.4+10.json
) &
(
  while kill -0 "$2" 2>/dev/null; do sleep 20; done
  run chen_e_slow    --dataset chen2019_ether --fit 0.4
  run chen_e_bracket --dataset chen2019_ether --fit 0.4 10 --start-json $OUTD/chen2019_ether_fit0.4+2.json
  run ma_S_slow      --dataset ma2026_SPAN --fit 0.2
  run chen_e_all     --dataset chen2019_ether --fit 0.4 1 2 4 6 10 --max-nfev 15 --start-json $OUTD/chen2019_ether_fit0.4+10.json
) &
wait
echo "$(date -u +%H:%M:%S) queue 3 done"
