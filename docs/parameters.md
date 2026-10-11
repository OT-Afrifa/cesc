# Parameters

Every parameter lives in cesc.Params with the published Table 1
production defaults. Nothing is hard-coded elsewhere.

| parameter | default | role |
|---|---|---|
| env_mode | "ECE" | "ECE" or "PI_only" (surface-based admitted) |
| unstable_threshold | 1.0 K | minimum cumulative PI drop |
| stable_threshold | 2.0 K | minimum stable-layer rise |
| max_pi_height_m | 5000 | PI search ceiling (AGL) |
| min_pi_depth_m | 500 | minimum PI depth |
| max_stable_height_m | 3000 | stable scan ceiling (AGL) |
| theta_eps | 5e-3 K | scan noise tolerance |
| faint_min_diff | 5.0 dBZ | faint threshold floor |
| strong_min_diff | 10.0 dBZ | strong threshold floor |
| faint_k_sigma, strong_k_sigma | 0.30 | sigma multipliers |
| box_sizes | (10, 20, 40) km | multi-scale box widths |
| bg_radius | 30 km | display and watershed-seeding background only |
| min_area_km2 | 4.05 | area filter (5 px at 900 m) |
| min_support_frac | 0.30 | gate: fraction inside support |
| min_support_px | 5 | gate: pixels inside support |
| use_advection | True | build the corridor |
| advection_time_min | 20 | corridor lookahead |
| halfwidth_km | 10 | corridor half-width |
| seed_stride_px | 5 | corridor seed spacing |
| min_speed_ms | 2 | minimum corridor wind |
| frac_strong_cutoff | 0.5 | strong/faint split |
| precip_floor_dbz | -2 | precipitating-domain floor |

Sensitivity, from the published Table 2 (11-hour case totals):
advection_time 10/30 min moves CESC area about -4/+4% and volume
-3.3/+4.6%; bg_radius 10/50 km moves area +14.2/-4.8% (through
watershed seeding, not thresholds); box_sizes doubled to (20, 40, 80)
km moves area +46% and volume +62%. box_sizes is the sensitive one:
scale it with the size of the convective cells being targeted rather
than treating it as universal.
