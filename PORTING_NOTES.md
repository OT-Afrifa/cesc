# Porting notes

The research modules (id_pro3.py v2.0, the notebook pipeline cells, and
distance_calculator.py) are the reference for numerical behavior. Four
places in this package deliberately differ. Everything else is a
faithful port; the Stage 1 scans, the integral-image thresholds, the
union-find labeling, the watershed seeding, and the gate reproduce the
reference logic line for line.

## 1. ece_strict is gone

The research signature carried `ece_strict` and its docstring described
it, but the body never read it: the support mask is always
core OR corridor, and the corridor is empty when use_advection is off.
The package drops the argument. `Params(use_advection=False)` gives the
core-only behavior the docstring described.

## 2. The two-hour RAINNC increment at the second timestep of a month

In `run_convective_pipeline_over_times`, when the first-hour increment
is computed from the previous month's last file, `prev_rainnc` is left
at that previous-month value. The next iteration then differences the
second hour against it, so the second timestep of every month except
the simulation start carries a two-hour increment and double-counts the
first hour's precipitation. The package sets `prev_rainnc` to the
current file's value after the first increment.

Consequence for existing research output: the seasonal
`CESC_*_grids_*.nc` files written by the reference runner carry this
double count in `rainnc`, `convective_precip`, and
`convective_precip_masked` at the second hour of each month. Monthly
accumulations built from them are high by up to one hour of
domain-mean precipitation. Rerunning with this package, or repairing
that one timestep per month, removes it.

## 3. The downstream-metrics enrichment indexed the wrong objects

`add_downstream_metrics_to_objects` builds slices from `labels_env`
(compact kept-object ids 1..N) and then indexes them with `o['id']`,
which is the watershed id. The two numberings agree only until the
first object is dropped at the gate; after that, every enriched row
reads the footprint of a different object. Affected fields:
area_ece_px, area_ds_px, area_sup_px, frac_support, L_ds_km, W_ds_km,
and the overwritten frac_within_ece and frac_downstream. The gate
decisions themselves are unaffected, because the gate uses the overlap
computed correctly earlier in the loop. The package maps watershed ids
through the kept-id remapping before indexing, so each row is enriched
with its own footprint.

Consequence for existing research output: in the parquet object
tables, those columns are scrambled across objects at any timestep
where at least one watershed object failed the gate. Columns written
before the enrichment (size_km2, frac_strong, intensity_cls,
frac_within_support, kept_env, centroid) are correct.

## 4. Signature defaults replaced by Params

Research signatures carried pre-Table-1 defaults (faint_min_diff 2.0,
strong_min_diff 5.0, min_area 6.0, box_sizes (20, 40, 80),
bg_radius 20). The production run cells always overrode them with the
Table 1 values. In this package thresholds have no function-level
defaults; the only defaults live in `cesc.Params`, which ships the
Table 1 production values (faint_min_diff 5.0, strong_min_diff 10.0,
k_sigma 0.30, box_sizes (10, 20, 40) km, bg_radius 30 km,
min_area 4.05 km2, support gate 30% and 5 px, advection 20 min,
halfwidth 10 km, frac_strong_cutoff 0.5).

## Also worth knowing

The quicklook figure is opt-in (`make_figure=True`); the reference
built and discarded a cartopy figure every timestep. The month runner
verifies that the positionally selected wrfout filename contains the
expected hour tag and falls back to a time-based search, where the
reference assumed a gap-free file list.
