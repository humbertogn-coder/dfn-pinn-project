#!/usr/bin/env bash
# Phase 2 pilot on the plain-SPAN rate test of Hao et al. (2025), SI Fig. S9 (data/literature_span/hao2025):
# slowest rate only, slowest + fastest, slow + medium, and all five rates (model limit; started from the
# slowest + fastest estimates). Two lanes (two CPUs); a finished fit (its JSON exists) is skipped, so the queue can be
# relaunched after a container restart.
set -u
cd "$(dirname "$0")/.."
OUTD=results/lispan/literature_rates
LOG=$OUTD/logs
mkdir -p "$LOG"
run() {
    local tag=$1 out=$2; shift 2
    if [ -f "$OUTD/$out.json" ]; then echo "$(date -u +%H:%M:%S) skip $tag (done)"; return; fi
    python3 -u scripts/lispan_lit_rates.py "$@" > "$LOG/$tag.log" 2>&1
    echo "$(date -u +%H:%M:%S) done $tag"
}
(
    run hao_0.1 hao2025_SPAN_fit0.1 --dataset hao2025_SPAN --fit 0.1
    run hao_0.1+0.5 hao2025_SPAN_fit0.1+0.5 --dataset hao2025_SPAN --fit 0.1 0.5
) &
(
    run hao_0.1+2 hao2025_SPAN_fit0.1+2 --dataset hao2025_SPAN --fit 0.1 2
    run hao_all hao2025_SPAN_fit0.1+0.2+0.5+1+2 --dataset hao2025_SPAN --fit 0.1 0.2 0.5 1 2 --max-nfev 12 \
        --start-json $OUTD/hao2025_SPAN_fit0.1+2.json
) &
wait
echo "$(date -u +%H:%M:%S) hao queue done"
