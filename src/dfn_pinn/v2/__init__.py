"""DFN PINN, version 2.

A compact, self-contained forward/inverse PINN for the isothermal DFN model
with Chen2020 parameters. Design notes are in docs/V2_DESIGN.md.

Key differences from the v1 ``DFNSmoke`` assembly:

* every network output is multiplied by a physically estimated scale, so the
  raw network outputs are O(1) at the solution (v1 used a fixed 0.01 factor,
  which required raw outputs of O(50-350) for the overpotentials);
* the benchmark is a full constant-current discharge with a smooth tanh
  current ramp instead of a 1 s step from rest;
* collocation points are resampled every step and networks are wider/deeper;
* residuals are nondimensionalized so that each term is O(1).
"""

from .params import CellParams, Protocol, Scales  # noqa: F401
