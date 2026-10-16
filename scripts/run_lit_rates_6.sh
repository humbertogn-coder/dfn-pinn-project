#!/usr/bin/env bash
# Restricted transport test on the Ma 2026 cells: w_S, Z_CC and the salt diffusivity free, all equilibrium and
# kinetic parameters fixed at the 0.2 + 1 C calibration (see results/lispan/literature_rates/NOTE.txt).
set -u
cd "$(dirname "$0")/.."
OUTD=results/lispan/literature_rates
LOG=$OUTD/logs
run() { local tag=$1; shift; python3 -u scripts/lispan_lit_rates.py "$@" > "$LOG/$tag.log" 2>&1; echo "$(date -u +%H:%M:%S) done $tag"; }
run ma_A_D3 --dataset ma2026_SPAN-A --fit 0.2 1 --extra D_salt --tag Dfree3 --timeout 45 --max-nfev 12 \
    --fix $(cat $LOG/ma2026_SPAN-A_fixargs.txt) --start-json $OUTD/ma2026_SPAN-A_fit0.2+1.json --start '{"D_salt": 0.05}' &
run ma_S_D3 --dataset ma2026_SPAN --fit 0.2 1 --extra D_salt --tag Dfree3 --timeout 45 --max-nfev 12 \
    --fix $(cat $LOG/ma2026_SPAN_fixargs.txt) --start-json $OUTD/ma2026_SPAN_fit0.2+1.json --start '{"D_salt": 0.05}' &
wait
echo "$(date -u +%H:%M:%S) queue 6 done"
