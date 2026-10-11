# Data requirements

The pipeline consumes one xarray Dataset per month (or any run of
timesteps) with these variables and coordinates.

Required
- reflectivity_1km (time, y, x), dBZ. Any 2-D level works; 1 km AGL is
  the default guidance for continuity with radar climatologies, and the
  column maximum gives essentially the same objects for shallow
  cold-season cells. The lowest model level is not recommended: over
  complex terrain it is divorced from tropospheric dynamics and shrinks
  objects to the precipitation footprint, which would also require
  retuning box_sizes.
- theta_e (time, z, y, x), K. Equivalent potential temperature, Bolton
  (1980). Not dry theta, not theta_v; the preflight rejects ranges that
  look like either.
- heights (time, z, y, x), m AGL, same vertical count as theta_e,
  increasing upward.
- XLAT, XLONG (y, x) coordinates in degrees, and a time coordinate.

Required when the downstream corridor is on (the default)
- u3d, v3d (time, z, y, x), m/s, same vertical count as heights.
  Set Params(use_advection=False) to gate on spatial overlap only.

Optional
- An instantaneous surface precipitation rate (time, y, x), mm/h, for
  attribution. It must be instantaneous and co-located in time with the
  reflectivity; accumulations trigger a warning and undercount
  convective precipitation.
- HWP (time, y, x): object means are added to the table when present.
- ter (y, x): terrain for the quicklook figure.

Grid spacing is estimated from XLAT/XLONG; supply uniform dx_km/dy_km
through the adapters when no 2-D coordinates exist.
