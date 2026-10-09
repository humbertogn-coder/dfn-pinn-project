Machine: Intel(R) Xeon(R) Processor @ 2.10GHz, one thread per process.

| task | cost | accuracy |
| --- | --- | --- |
| FV discharge 0.1 C, 20/10 volumes | 4.4 s CPU | reference |
| FV discharge 0.1 C, 40/20 volumes | 5.4 s CPU | reference |
| FV discharge 1 C, 20/10 volumes | 5.3 s CPU | reference |
| FV discharge 1 C, 40/20 volumes | 7.2 s CPU | reference |
| PINN training 0.1 C, seed 0 | 2.79 h wall, machine shared (evaluations every 2500 steps: < 1 mV from step 12500, < 0.5 mV from step 12500) | V rms 0.16 mV, max 0.74 mV |
| PINN training 0.1 C, seed 1 | 1.78 h wall, machine shared (evaluations every 2500 steps: < 1 mV from step 12500, < 0.5 mV from step 17500) | V rms 0.43 mV, max 3.54 mV |
| PINN training 0.1 C, seed 2 | 2.92 h wall, machine shared (evaluations every 2500 steps: < 1 mV from step 12500, < 0.5 mV from step 17500) | V rms 0.18 mV, max 0.81 mV |
| PINN training 1 C, seed 0 | 2.20 h wall, machine shared (evaluations every 2500 steps: < 1 mV from step 17500, < 0.5 mV from step 22500) | V rms 0.23 mV, max 0.77 mV |
| PINN training 1 C, seed 1 | 1.78 h wall, machine shared (evaluations every 2500 steps: < 1 mV from step 15000, < 0.5 mV from step 17500) | V rms 0.14 mV, max 0.63 mV |
| PINN training 1 C, seed 2 | 2.90 h wall, machine shared (evaluations every 2500 steps: < 1 mV from step 15000, < 0.5 mV from step 20000) | V rms 0.18 mV, max 0.70 mV |
| PINN training 0.1 C, CPU projection from measured step time (30000 steps) | 1.87 h CPU (215 / 229 ms per step) | - |
| PINN training 1 C, CPU projection from measured step time (30000 steps) | 1.93 h CPU (228 / 233 ms per step) | - |
| trained PINN 0.1 C: voltage at 1000 times / all fields 20 x 1000 | 60 ms / 36 ms | as trained |
| trained PINN 1 C: voltage at 1000 times / all fields 20 x 1000 | 59 ms / 35 ms | as trained |
| 8-parameter inverse, FV least squares (68 solves per rate) | 0.24 h CPU | k0_1 +1.58, k0_2 +1.10, k0_3 +0.07, b_1 +0.03, b_2 +0.30, b_3 +0.02, U0_1 -0.69, Z_CC +0.04 |
| 8-parameter inverse, PINN (stage 1 + multi-rate LM) | 1.94 h + 5.0 h forward pre-training | k0_1 +1.96, k0_2 -0.31, k0_3 +0.31, b_1 -0.14, b_2 +0.05, b_3 +0.06, U0_1 -0.99, Z_CC +0.09 |
