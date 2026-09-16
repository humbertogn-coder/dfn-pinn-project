# Differentiable Constitutive Functions

`src/dfn_pinn/constitutive.py` provides raw Chen2020 OCPs, exchange-current
densities and the Nyman2008 electrolyte transport fits using PyTorch only.
SI units and active-area current conventions are documented in each function.
No data are fitted and no neural network is trained.

Source audited: installed PyBaMM 26.8.0.0
`input/parameters/lithium_ion/Chen2020.py` and the processed `prim.U` and
`prim.j0` expressions in `parameters/lithium_ion_parameters.py`.
The gas and Faraday constants match that installation. Tests deliberately
require that version; upgrading requires another provenance/compatibility audit.

The tensor functions preserve device, dtype and autograd. Use float64 for
scientific verification. All exchange-current inputs must have matching dtype
and device; ordinary tensor broadcasting applies. Domain checks reject invalid
inputs without clipping. These eager checks may synchronize GPU execution;
performance optimization is deferred until profiling.

## Scope and Endpoint Differences

PyBaMM clips some inputs and regularizes powers near depletion. Its processed
OCP also adds endpoint barriers. OCP functions implement raw parameter fits,
not those endpoint barriers. Exchange current provides explicit `mode="raw"`
and `mode="pybamm_26_8"`. The latter uses the exporting version's RegPower:
`x*(x*x+delta*delta)**(-0.25)*sqrt(scale)`, where x=c/scale and delta=0.001.
Scales are 1000 mol/m3 for electrolyte and c_s_max for both solid factors.
Use this explicit reference mode when matching HDF5 kinetics. The correction
is nonzero even away from endpoints; raw and regularized j0 are not identical.
All inputs remain strictly admissible; clipping and exact endpoints are unsupported.
Tests compare each mode to its corresponding raw or processed expression
on stoichiometry [0.01, 0.99], electrolyte concentration [100, 3000] mol/m3 and
exchange-current temperature [0.95, 1.05]*298.15 K, varying one argument at a
time. These ranges cover the current reference, but are not a full joint-domain
or empirical validity certificate. Do not claim equivalence near endpoints.
Input checks enforce physical domains, not an empirical calibration range.

OCP entropic coefficients are zero in Chen2020. The Nyman transport fits have
no temperature dependence in this parameterization; none is invented here.
Solid diffusivities/conductivities are reference constants, not learned values.
Porosity/Bruggeman multipliers and PDE residuals are not yet implemented.

## Verification

Run `python scripts/check_constitutive.py`.
Tests compare values and first/second derivatives to symbolic PyBaMM expressions,
and use PyTorch gradcheck/gradgradcheck on exchange current, including broadcast
temperature. Very small diffusivities are scaled before error comparison.
Invalid-input checks and dtype preservation are also covered. No DFN simulation,
HDF5 rewrite, training or Git operation is performed by this command.

References: Chen et al., Journal of The Electrochemical Society 167 (2020),
080534; Nyman et al., Electrochimica Acta 53 (2008), 6356-6365, as attributed
by the inspected PyBaMM parameter module.
