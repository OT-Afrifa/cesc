# The algorithm

Three stages per timestep. Stage numbering follows the paper.

## Stage 1: the supportive environment

From theta_e profiles, per column:
- PI layer: a run of consecutive levels with d(theta_e)/dz < 0 whose
  cumulative drop is at least 1 K, at least 500 m deep, based below
  5000 m AGL. The first qualifying run scanning upward is recorded.
- Stable near-surface layer: scanning upward from the surface,
  cumulative positive theta_e rise of at least 2 K, accumulated to
  min(PI base, 3000 m AGL). Wiggles smaller than 5e-3 K do not reset
  the accumulation, which tolerates well-mixed layers capped by
  higher-theta_e air.

Columns with both are the ECE; PI columns without the stable layer are
the SCE (surface-based convection environment). env_mode "ECE" gates on
the first, "PI_only" on their union. The selected environment is
dilated one pixel and extended downwind: from seeds every 5 pixels,
rays run along the local mass-weighted PI-layer wind for wind speed
times advection_time (20 min), skipped below 2 m/s, then widened by
halfwidth (10 km). Support mask = core OR corridor.

## Stage 2: cell detection

On the reflectivity field, restricted to precipitating pixels (Z at
least -2 dBZ) and not to the environment: for each box width in
box_sizes (10, 20, 40 km), the NaN-aware local mean mu and standard
deviation sigma come from integral images (boxes clamp at the domain
edge), and

    faint:  Z - mu >= max(5 dBZ, 0.30 sigma)
    strong: Z - mu >= max(10 dBZ, 0.30 sigma)

Per-scale detections are OR'd across scales; strong is a subset of
faint. The faint union is labeled (8-connectivity) and split along
watershed lines of the Gaussian-smoothed enhancement Enh = Z - Z_bg,
seeded with local maxima above the 85th percentile separated by at
least max(8 px, min_area/grid). Objects below 4.05 km2 are removed.

Z_bg is the bg_radius (30 km) uniform-filter background. It exists for
the quicklook display and the watershed seeding only. It never enters
the detection thresholds; those use the per-scale box backgrounds.

## Stage 3: the gate and the split

An object is kept when its overlap with the support mask is at least
5 pixels and at least 30% of its area. Kept objects are renumbered
1..N; those with frac_strong at or above 0.5 go to labels_strong, the
rest to labels_faint, sharing the labels_env numbering. The object
table records geometry, overlap fractions, wind-aligned length and
width of the supported footprint, and attribution statistics when a
rate field is supplied.
