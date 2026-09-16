# Spherical Diffusion Residual

Implemented in `src/dfn_pinn/spherical_diffusion.py` for constant diffusivity,
as in the current isothermal Chen2020 particles. No x diffusion is introduced.
The model consumes independent point rows (rho=r/R, global x/L, tau=t/t_ref)
and returns shape (N,1), c_hat=c/c_ref. Inputs require gradients.

    lambda = D*t_ref/R^2
    residual = d_tau c_hat - lambda*(d_rhorho c_hat + 2/rho*d_rho c_hat)

This is the dimensional PDE divided by c_ref/t_ref. At rho=0, use
3*d_rhorho c_hat, assuming smooth radial symmetry. Enforce d_rho c_hat=0
separately: evaluating the regular limit alone does not impose symmetry.
The implementation avoids division by zero even in an unselected torch branch.
Very small nonzero radii can still amplify numerical derivative errors.

At rho=1, the flux residual is

    (-F*D*c_ref/R*d_rho c_hat - j)/j_ref

j is signed interfacial current, not a*j or geometric current. Positive j
removes solid lithium. The flux API excludes tau=0 at the current-step corner.
Initial concentration enforcement and the full PINN loss remain pending.

Volume means are 3*integral(c_hat*rho^2 d_rho). For constant flux they satisfy
d(mean c)/dt=-3*j/(F*R). Caller-supplied radial weights must cover [0,1]
and contain d_rho only, not preweighted volume factors.

Autograd uses a summed-output vector-Jacobian product. This is valid only for
pointwise models: no attention across samples, batch normalization or other
cross-row coupling. Derivatives retain the parameter graph for loss training.
Geometry and diffusivity are fixed constants in this first API; inverse
parameter estimation and concentration-dependent diffusivity are not supported.

Run `python scripts/check_spherical_residual.py`. Tests cover both electrode
scales, an exact quadratic solution, the no-flux spherical eigenmode, center
symmetry, both flux signs, volume conservation, constant/linear fields and
first/second derivatives of a residual loss with respect to a model coefficient.
The quadratic flux solution has a compatible nonuniform initial profile: it
is not a claim to solve the uniform-initial, suddenly applied flux benchmark.
The eigenmode test uses interior points to avoid differentiating a removable
singularity in its explicit sine-over-radius expression.

This verifies equation assembly against analytic fields, not neural training,
full DFN accuracy or an empirical battery model. No HDF5 data are modified.
