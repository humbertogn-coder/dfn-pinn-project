# PINN Reference Reader

Run `python scripts/prepare_pinn_reference.py` to inspect the latest completed
working-reference bundle, or provide `--input <bundle-directory>` explicitly.
No simulation runs and source HDF5 files are never modified.

The reader verifies SHA256, file size, schema, native coordinates, units,
dimension scales and compatible segment settings. It reports differences at
common timestamps for all fields and terminal signals without interpolation.
These diagnostics do not certify all-field agreement or numerical accuracy.

Time ownership is fine_startup through its endpoint, startup strictly after
that endpoint through its endpoint, and full strictly thereafter. At the
current settings this gives [0, 0.2], (0.2, 10], and (10, cutoff] seconds.
`ReferenceBundle.iter_field` yields one native spatial slice at a time with
its segment and dimensional time. Particle axes remain r,x; no resampling occurs.

Concentrations are divided by electrode maximum concentrations or initial
electrolyte concentration. Positions use total cell thickness for global x
and each particle radius for r. Time uses a fixed 3600-second 1C scale.
All transforms have zero offset and are inverted by multiplication.
Coordinate scales are recorded for consumers; native HDF5 coordinates remain
dimensional. Values are not clipped to [0,1]. No dataset extrema are fitted.

Scales are reconstructed from Chen2020 only when installed PyBaMM matches the
export version and layer geometry agrees. A complete frozen parameter snapshot
is still desirable for long-term reproducibility. Potentials and currents are
readable but remain dimensional until their PINN scaling is specified.

The timestamped preparation report records scales, ranges, sample counts and
overlap diagnostics. It is preparation for inspection, not approval for training.
No training split is created. Overlapping simulations are not independent test
data, and dense startup sampling must not disproportionately weight evaluation.
