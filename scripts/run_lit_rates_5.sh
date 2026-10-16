#!/usr/bin/env bash
# Phase 2 pilot, follow-up: Ma 2026 two-rate calibrations with the salt diffusivity free (the check in
# results/lispan/literature_rates/transport_check_ma2026_SPAN-A.json showed that a ~30x slower salt diffusion
# reproduces a rate-dependent capacity loss). Starts when queue 4 (PID $1) has finished.
set -u
cd "$(dirname "$0")/.."
OUTD=results/lispan/literature_rates
LOG=$OUTD/logs
while kill -0 "$1" 2>/dev/null; do sleep 20; done
run() { local tag=$1; shift; python3 -u scripts/lispan_lit_rates.py "$@" > "$LOG/$tag.log" 2>&1; echo "$(date -u +%H:%M:%S) done $tag"; }
run ma_A_slowfast_D --dataset ma2026_SPAN-A --fit 0.2 1 --extra D_salt --tag Dfree --start-json $OUTD/ma2026_SPAN-A_fit0.2+1.json --start '{"D_salt": 0.05}' &
run ma_S_slowfast_D --dataset ma2026_SPAN --fit 0.2 1 --extra D_salt --tag Dfree --start-json $OUTD/ma2026_SPAN_fit0.2+1.json --start '{"D_salt": 0.05}' &
wait
echo "$(date -u +%H:%M:%S) queue 5 done"
