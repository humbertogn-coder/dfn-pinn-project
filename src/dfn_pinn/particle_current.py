"""Shared current source for local particle inventory and surface residuals."""

import math
from torch import nn

from .constitutive import _check, ocp, exchange_current, MAX_CONCENTRATION
from .kinetics import direct_bv
from .current_integral import CurrentIntegral
from .particle_balance import particle_mean_from_charge
from .spherical_diffusion import surface_residual


class ParticleCurrent(nn.Module):
    """Connect a pointwise j(x/L,tau) [A/m2] to two particle constraints.

    This supplies an inventory target, not a concentration projection or a
    diffusion solution. The same current module is registered only once.
    Current quadrature must be resolved; discontinuities need separate handling.
    """

    def __init__(self, current_model, scales, order=32):
        super().__init__()
        self.scales = scales
        self.integral = CurrentIntegral(current_model, scales.time_s, order)

    def mean_target(self, points, initial_mean):
        """Target normalized mean at (N,2) [x/L,tau], given initial mean."""
        return particle_mean_from_charge(initial_mean, self.integral(points), self.scales)

    def kinetics_residual(self, points, stoichiometry, electrolyte_concentration,
                          solid_potential, electrolyte_potential, temperature,
                          electrode, current_scale, *, mode):
        """(j - BV(phi_s-phi_e-U, j0))/j_ref at (N,2) [x/L,tau].

        All fields must be (N,1), with SI concentrations/potentials/temperature.
        Stoichiometry is c_s_surface/c_s_max, NOT an arbitrary c_ref ratio.
        The caller supplies surface fields with their graphs intact. No implicit
        solve or concentration projection is performed. Mode is explicit.
        """
        _check(points, "kinetics points")
        if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
            raise ValueError("Expected nonempty (N,2) kinetics points")
        if not bool(((points[:, 0] >= 0) & (points[:, 0] <= 1)).all()) or not bool((points[:, 1] >= 0).all()):
            raise ValueError("Require x/L in [0,1] and tau >= 0")
        if not math.isfinite(current_scale) or current_scale <= 0:
            raise ValueError("Current scale must be positive and finite")
        current = self.integral.current_model(points)
        fields = (current, stoichiometry, electrolyte_concentration,
                  solid_potential, electrolyte_potential, temperature)
        for field in fields:
            _check(field, "kinetics field")
            if field.shape != (len(points), 1) or field.dtype != points.dtype or field.device != points.device:
                raise ValueError("Kinetics fields must be (N,1) with input dtype and device")
        potential = ocp(stoichiometry, electrode)
        j0 = exchange_current(electrolyte_concentration,
                              stoichiometry*MAX_CONCENTRATION[electrode],
                              temperature, electrode, mode=mode)
        eta = solid_potential-electrolyte_potential-potential
        residual = (current-direct_bv(eta, j0, temperature))/current_scale
        _check(residual, "kinetics residual")
        return residual

    def flux_residual(self, concentration_model, surface_points, current_scale):
        """Use the same j at (N,3) [rho=1,x/L,tau>0] surface points."""
        # Validate coordinates before evaluating the current callable.
        _check(surface_points, "surface points")
        if surface_points.ndim != 2 or surface_points.shape[1] != 3 or len(surface_points) == 0:
            raise ValueError("Expected nonempty (N,3) surface points")
        if not bool((surface_points[:, 0] == 1).all()) or not bool((surface_points[:, 2] > 0).all()):
            raise ValueError("Surface flux requires rho=1 and tau>0")
        current = self.integral.current_model(surface_points[:, 1:])
        _check(current, "interfacial current")
        if current.shape != (len(surface_points), 1):
            raise ValueError("Current model must return (N,1)")
        return surface_residual(concentration_model, surface_points, current,
                                self.scales, current_scale)
