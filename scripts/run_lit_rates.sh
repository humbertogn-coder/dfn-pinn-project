#!/usr/bin/env bash
# Phase 2 pilot queue (scripts/lispan_lit_rates.py), two lanes on two cores.
# Per dataset: slow + fast calibration, slow only, then all rates (warm start from slow + fast).
set -u
cd "$(dirname "$0")/.."
OUTD=results/lispan/literature_rates
LOG=$OUTD/logs
mkdir -p "$LOG"
run() { local tag=$1; shift; python3 -u scripts/lispan_lit_rates.py "$@" > "$LOG/$tag.log" 2>&1; echo "$(date -u +%H:%M:%S) done $tag"; }
(
  run chen_c_slowfast --dataset chen2019_carbonate --fit 0.4 2
  run ma_A_slowfast   --dataset ma2026_SPAN-A --fit 0.2 1
  run chen_c_slow     --dataset chen2019_carbonate --fit 0.4
  run ma_A_slow       --dataset ma2026_SPAN-A --fit 0.2
  run chen_c_all      --dataset chen2019_carbonate --fit 0.4 1 2 4 6 10 --max-nfev 20 --start-json $OUTD/chen2019_carbonate_fit0.4+2.json
) &
(
  run chen_e_slowfast --dataset chen2019_ether --fit 0.4 2
  run ma_S_slowfast   --dataset ma2026_SPAN --fit 0.2 1
  run chen_e_slow     --dataset chen2019_ether --fit 0.4
  run ma_S_slow       --dataset ma2026_SPAN --fit 0.2
  run ma_A_all        --dataset ma2026_SPAN-A --fit 0.1 0.2 0.3 0.5 1 2 --max-nfev 20 --start-json $OUTD/ma2026_SPAN-A_fit0.2+1.json
) &
wait
echo "$(date -u +%H:%M:%S) lanes done"
