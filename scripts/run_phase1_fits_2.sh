#!/usr/bin/env bash
# Phase 1, second batch: two LOW rates (0.1 + 0.2 C) as calibration, measured and synthetic.
# Starts when the first batch (scripts/run_phase1_fits.sh, PID given as $1) has finished.
set -u
cd "$(dirname "$0")/.."
LOG=results/lispan/phase1/logs
while kill -0 "$1" 2>/dev/null; do sleep 30; done
run() { local tag=$1; shift; python3 -u scripts/lispan_phase1_fit.py "$@" > "$LOG/$tag.log" 2>&1; echo "$(date -u +%H:%M:%S) done $tag"; }
run exp_01_02 --data experiment --fit 0.1 0.2 --predict 0.05 1 &
run syn_01_02 --data synthetic --fit 0.1 0.2 --predict 0.05 1 &
wait
echo "$(date -u +%H:%M:%S) batch 2 done"
