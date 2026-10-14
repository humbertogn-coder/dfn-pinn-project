#!/usr/bin/env bash
# Phase 1 calibrate-then-predict fits (scripts/lispan_phase1_fit.py), two lanes on two cores.
# Truth for the synthetic runs = calibrated Simanjuntak cell (Z_CC multiplier 1.3235 -> 0.033 Ohm m2).
set -u
cd "$(dirname "$0")/.."
LOG=results/lispan/phase1/logs
mkdir -p "$LOG"
run() { local tag=$1; shift; python3 -u scripts/lispan_phase1_fit.py "$@" > "$LOG/$tag.log" 2>&1; echo "$(date -u +%H:%M:%S) done $tag"; }

(   # lane A: measured data, calibrate on 0.1 C only, contact resistance free or assumed
    run exp_01_free  --data experiment --fit 0.1 --predict 0.05 0.2 1
    run exp_01_Z0    --data experiment --fit 0.1 --predict 0.05 0.2 1 --fix Z_CC=0   --tag Z0
    run exp_01_Z0.5  --data experiment --fit 0.1 --predict 0.05 0.2 1 --fix Z_CC=0.5 --tag Z0.5
    run exp_01_Z1.4  --data experiment --fit 0.1 --predict 0.05 0.2 1 --fix Z_CC=1.4 --tag Z1.4
    run exp_01_Z2    --data experiment --fit 0.1 --predict 0.05 0.2 1 --fix Z_CC=2   --tag Z2
) &
(   # lane B: synthetic data (5 mV noise), then the measured 0.1 + 1 C calibration
    run syn_01_free  --data synthetic --fit 0.1 --predict 0.05 1 2
    run syn_01_Zhalf --data synthetic --fit 0.1 --predict 0.05 1 2 --fix Z_CC=0.6617 --tag Zhalf
    run syn_01_Z2x   --data synthetic --fit 0.1 --predict 0.05 1 2 --fix Z_CC=2.647  --tag Z2x
    run syn_01_1     --data synthetic --fit 0.1 1 --predict 0.05 2
    run exp_01_1     --data experiment --fit 0.1 1 --predict 0.05 0.2
) &
wait
echo "$(date -u +%H:%M:%S) all done"
