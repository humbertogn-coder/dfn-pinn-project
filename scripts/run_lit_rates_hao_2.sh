#!/usr/bin/env bash
# Second Hao (2025) queue, started next to run_lit_rates_hao.sh: each lane waits for a CPU freed by the first queue.
#   lane 1: 0.2 + 2 C calibration (the 0.1 C step was measured on a still-activating cell, see
#           docs/PHASE2_LITERATURE_DATA.md), after the 0.1 + 0.5 C fit of the first queue;
#   lane 2: one-rate aging check on SI Fig. S12b (scripts/lispan_lit_aging_check.py), after the all-rates fit.
set -u
cd "$(dirname "$0")/.."
OUTD=results/lispan/literature_rates
LOG=$OUTD/logs
waitfor() { while [ ! -f "$1" ]; do sleep 30; done; }
(
    waitfor $OUTD/hao2025_SPAN_fit0.1+0.5.json
    if [ ! -f $OUTD/hao2025_SPAN_fit0.2+2.json ]; then
        python3 -u scripts/lispan_lit_rates.py --dataset hao2025_SPAN --fit 0.2 2 > $LOG/hao_0.2+2.log 2>&1
    fi
    echo "$(date -u +%H:%M:%S) done hao_0.2+2"
) &
(
    waitfor "$OUTD/hao2025_SPAN_fit0.1+0.2+0.5+1+2.json"
    if [ ! -f $OUTD/aging_check_hao2025.json ]; then
        python3 -u scripts/lispan_lit_aging_check.py > $LOG/hao_aging.log 2>&1
    fi
    echo "$(date -u +%H:%M:%S) done hao_aging"
) &
wait
echo "$(date -u +%H:%M:%S) hao queue 2 done"
