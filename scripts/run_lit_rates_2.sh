#!/usr/bin/env bash
# Phase 2 pilot, second batch: "bracketing" calibrations (slowest + fastest rate, intermediate rates predicted)
# and the remaining all-rates fits. Starts when the first batch (PID $1) has finished.
set -u
cd "$(dirname "$0")/.."
OUTD=results/lispan/literature_rates
LOG=$OUTD/logs
while kill -0 "$1" 2>/dev/null; do sleep 30; done
run() { local tag=$1; shift; python3 -u scripts/lispan_lit_rates.py "$@" > "$LOG/$tag.log" 2>&1; echo "$(date -u +%H:%M:%S) done $tag"; }
(
  run chen_c_bracket --dataset chen2019_carbonate --fit 0.4 10 --start-json $OUTD/chen2019_carbonate_fit0.4+2.json
  run ma_A_bracket   --dataset ma2026_SPAN-A --fit 0.2 2 --start-json $OUTD/ma2026_SPAN-A_fit0.2+1.json
  run chen_e_all     --dataset chen2019_ether --fit 0.4 1 2 4 6 10 --max-nfev 20 --start-json $OUTD/chen2019_ether_fit0.4+2.json
) &
(
  run chen_e_bracket --dataset chen2019_ether --fit 0.4 10 --start-json $OUTD/chen2019_ether_fit0.4+2.json
  run ma_S_bracket   --dataset ma2026_SPAN --fit 0.2 2 --start-json $OUTD/ma2026_SPAN_fit0.2+1.json
  run ma_S_all       --dataset ma2026_SPAN --fit 0.1 0.2 0.3 0.5 1 2 --max-nfev 20 --start-json $OUTD/ma2026_SPAN_fit0.2+1.json
) &
wait
echo "$(date -u +%H:%M:%S) batch 2 done"
