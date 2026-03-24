# CESC: Cold-Season Embedded/Emergent Convection Identification

A Python package for detecting and analysing **Convection Embedded within or
Emergent from Stratiform Clouds (CESC)** in cold-season orographic precipitation
using convection-permitting WRF model output.

Developed at the University of Wyoming in support of the SNOWIE (Seeded and
Natural Orographic Wintertime Clouds: the Idaho Experiment) research programme.

---

## Background

In cold-season frontal systems, convective cells can form above a stable
near-surface layer when a potentially unstable (PI) layer aloft is lifted to
saturation by frontal ascent or orographic forcing.  These elevated cells are
decoupled from the underlying terrain, emerge above the surrounding stratiform
cloud deck, and can deposit snowfall in locations that standard orographic
precipitation models do not anticipate.

Afrifa et al. (2025) presented a detailed case study of one such event over
the Snake River Plain and Central Idaho Mountains on 7 February 2017 during
SNOWIE IOP12.  Afrifa et al. (2026, accepted) used 100 m large-eddy simulations
to examine the microphysical structure of those cells.  This package implements
the objective identification algorithm described in Afrifa et al. (Part 1,
submitted 2026), which makes it possible to run a systematic climatological
analysis of CESC frequency, intensity, and precipitation contribution across
full cold seasons.

---

## Algorithm overview

The pipeline has three stages.

**Stage 1 -- ECE classification**

Each model column is tested for the Embedded/Emergent Convection Environment
(ECE): a surface stable layer (cumulative theta_e rise >= 2 K below 3000 m
AGL) overlain by a potentially unstable layer (cumulative theta_e drop >= 1 K
over at least 500 m depth, confined below 5000 m AGL).  Where both conditions
are satisfied, the column is flagged as ECE.  If advection is enabled, a
downstream corridor is appended using the PI-layer mean wind and an assumed
convective lifespan (default 20 minutes).

**Stage 2 -- Object detection**

Locally enhanced reflectivity objects are identified over the full
precipitating domain using a multi-scale adaptive background method adapted
from Yeh and Colle (2025).  For each of three box widths (20, 40, 80 km) a
per-pixel local mean and standard deviation are computed; a pixel is flagged
when its reflectivity exceeds the background by at least
`max(min_diff, k_sigma * local_sigma)`.  This is done separately for faint
and strong thresholds.  The union across all scales captures both small
isolated cells and larger organised clusters.

**Stage 3 -- Environmental gating and intensity classification**

Detected objects are retained as CESC only when at least 30 % of their area
(and at least 5 pixels) falls inside the support mask (ECE core + corridor).
Each kept object is further classified as *faint* or *strong* based on the
fraction of its pixels that exceeded the strong detection threshold, controlled
by the `frac_strong_cutoff` parameter.

---

## Installation

### From PyPI (when published)

```bash
pip install cesc
```

### From source

```bash
git clone https://github.com/OT-Afrifa/cesc.git
cd cesc
pip install -e .
```

### Dependencies

Core runtime dependencies are declared in `pyproject.toml` and installed
automatically by pip.  The main ones are:

| Package | Version | Role |
|---|---|---|
| numpy | >= 1.24 | array operations |
| numba | >= 0.57 | JIT-compiled hot loops |
| xarray | >= 2023.1 | labelled arrays and IO |
| scipy | >= 1.10 | image filtering, distance transforms |
| scikit-image | >= 0.20 | watershed, peak detection |
| wrf-python | >= 1.3.4 | reading wrfout variables |
| cartopy | >= 0.21 | map projections for quicklook figures |
| netCDF4 | >= 1.6 | direct NetCDF4 writing for streaming output |

On NCAR Derecho, a suitable environment can be created with:

```bash
module load conda
conda create -n cesc_env python=3.11
conda activate cesc_env
pip install -e ".[dev]"
```

---

## Input data format

The package currently expects WRF model output that has been pre-extracted
into monthly NetCDF files with the following variables:

| Variable name | Dimensions | Description |
|---|---|---|
| `reflectivity_1km` | (time, y, x) | Simulated radar reflectivity at 1 km AGL (dBZ) |
| `theta_e` | (time, bottom_top, y, x) | Equivalent potential temperature (K) |
| `heights` | (time, bottom_top_stag, y, x) | Full-level heights AGL (m) |
| `u3d` | (time, bottom_top, y, x) | Zonal wind (m/s) |
| `v3d` | (time, bottom_top, y, x) | Meridional wind (m/s) |
| `HWP` | (time, y, x) | 0-5 km AGL total hydrometeor water path (g/m^2) |
| `ter` | (y, x) or (time, y, x) | Terrain height (m) |
| `XLAT` | (y, x) | Latitudes (degrees north) |
| `XLONG` | (y, x) | Longitudes (degrees east) |

Precipitation increments and cloud-top temperature are read directly from the
original wrfout files, one hour at a time, so those files must be accessible
alongside the pre-extracted NC files.

Support for other NWP model formats (e.g. HRRR, RAP) is planned for a future
release.

---

## Quick start

### Single time step (interactive / notebook)

```python
import numpy as np
import xarray as xr
from cesc.pipeline import run_cesc_pipeline, scan_pi_stable

# Load one time step from your pre-extracted file
ds = xr.open_dataset("WRF_extracted_vars_900m_feb2017.nc",
                     decode_times=True).set_coords(["XLAT","XLONG"])
da_t = ds.sel(time="2017-02-07 22:00")

# Compute layer delta-theta_e
dtheta_e = np.diff(da_t["theta_e"].values.astype(np.float64), axis=0)
heights  = da_t["heights"].values.astype(np.float64)

# Stage 1: identify ECE columns
pi_str, pi_dep, st_str, st_dep, pi_h0, pi_h1, st_h0, st_h1 = scan_pi_stable(
    dtheta_e, heights,
    max_pi_height      = 5000.0,
    unstable_threshold = 1.0,
    stable_threshold   = 2.0,
)

# Stages 2 and 3: detect and gate CESC objects
result = run_cesc_pipeline(
    Z_2d_da            = da_t["reflectivity_1km"],
    ter_da             = da_t["ter"],
    HWP_da             = da_t["HWP"],
    pi_strength_da     = pi_str,
    stable_strength_da = st_str,
    pi_start_h_da      = pi_h0,
    pi_end_h_da        = pi_h1,
    u3d                = da_t["u3d"],
    v3d                = da_t["v3d"],
    heights            = da_t["heights"],
    use_advection      = True,
    advection_time     = 20,
    make_plot          = True,
)

# Inspect results
kept = [o for o in result["objects"] if o["kept_env"]]
print(f"Detected {len(kept)} CESC objects")
print(f"  faint:  {sum(1 for o in kept if o['intensity_cls']=='faint')}")
print(f"  strong: {sum(1 for o in kept if o['intensity_cls']=='strong')}")

# The quicklook figure is in result["fig"]
result["fig"].savefig("cesc_2200UTC.png", dpi=150, bbox_inches="tight")
```

### Batch run for a full month (low memory, streaming output)

```python
from cesc.run import run_month_streaming

run_month_streaming(
    nc_path          = "/data/WRF_extracted_vars_900m_feb2017.nc",
    wrfout_path      = "/data/wrfout_d02/",
    out_nc_path      = "/output/CESC_grids_feb2017.nc",
    out_parquet_path = "/output/CESC_objects_feb2017.parquet",
    pipeline_kwargs  = dict(
        frac_strong_cutoff = 0.5,   # tune on the 7 Feb 2017 case first
    ),
)
```

Peak RAM stays near 400 MB regardless of month length because the monthly
file is opened lazily and only one hour is loaded at a time.

### Command-line interface

After installation, a `run-cesc` command is available:

```bash
run-cesc \
    --nc  /data/WRF_extracted_vars_900m_feb2017.nc \
    --wrf /data/wrfout_d02/ \
    --out_nc /output/CESC_grids_feb2017.nc \
    --out_pq /output/CESC_objects_feb2017.parquet \
    --advection_time 20 \
    --frac_strong_cutoff 0.5
```

Run `run-cesc --help` for the full parameter list.

---

## Output variables

### Gridded NetCDF (one record per hour)

| Variable | dtype | Description |
|---|---|---|
| `labels_env` | int16 | All kept CESC objects (compact 1..N labels) |
| `labels_faint` | int16 | Subset of labels_env where frac_strong < cutoff |
| `labels_strong` | int16 | Subset of labels_env where frac_strong >= cutoff |
| `faint_det` | int8 | Pixel-level faint detection mask |
| `strong_det` | int8 | Pixel-level strong detection mask |
| `ece_core` | int8 | ECE core mask (both conditions met) |
| `corridor` | int8 | Downstream advection corridor mask |
| `support` | int8 | ECE core union corridor |
| `ece_dist_km` | float32 | Distance to nearest ECE pixel (km) |
| `Z_bg` | float32 | Display-only smoothed background reflectivity |
| `Enh` | float32 | Z minus display background (drives watershed seeding) |
| `rainnc_inc` | float32 | Hourly precipitation increment (mm) |
| `convective_precip` | float32 | Hourly precip inside CESC objects (mm; 0 outside) |
| `convective_precip_masked` | float32 | Same, NaN outside CESC objects |
| `CTT` | float32 | WRF column cloud-top temperature (degrees C) |

### Object parquet table (one row per object per hour)

Key columns include `time`, `object_uid`, `cls` (cell/band/complex),
`intensity_cls` (faint/strong), `size_km2`, `major_km`, `minor_km`, `aspect`,
`frac_strong`, `frac_within_ece`, `frac_downstream`, `centroid_lat`,
`centroid_lon`, `dist_to_ece_core_km`, `L_ds_km`, `W_ds_km`, and
`hwp_mean_gm2`.

---

## Algorithm parameters (Table 1, Afrifa et al. Part 1)

| Parameter | Default | Description |
|---|---|---|
| `unstable_threshold` | 1.0 K | Minimum cumulative PI theta_e drop |
| `stable_threshold` | 2.0 K | Minimum cumulative stable layer theta_e rise |
| `faint_min_diff` | 2.0 dBZ | Absolute floor for faint detection |
| `faint_k_sigma` | 0.30 | Sigma multiplier for faint threshold |
| `strong_min_diff` | 5.0 dBZ | Absolute floor for strong detection |
| `strong_k_sigma` | 0.30 | Sigma multiplier for strong threshold |
| `min_area` | 6.0 km^2 | Minimum object area |
| `min_support_frac` | 0.30 | Minimum fraction of object inside support mask |
| `min_support_px` | 5 | Minimum pixel count inside support mask |
| `advection_time` | 20 min | Assumed convective lifespan for corridor length |
| `halfwidth` | 10.0 km | Lateral half-width of the downstream corridor |
| `min_speed` | 2.0 m/s | Minimum wind speed to draw a corridor |
| `frac_strong_cutoff` | 0.5 | Threshold separating faint from strong objects |

Note: `box_sizes` (default 20, 40, 80 km) is a structural design choice rather
than a tunable threshold.  It controls which spatial scales are used to compute
the adaptive detection background.  Smaller boxes catch sharp isolated cells;
larger boxes catch broader organised clusters.  The union across scales is taken
to avoid missing either type.  This follows the same multi-scale union design
used by Yeh and Colle (2025).

---

## Tuning `frac_strong_cutoff`

The `frac_strong_cutoff` parameter controls the boundary between faint and
strong objects.  To set this for your dataset:

1. Run the single-time-step example on the 7 February 2017 22 UTC snapshot
   (the best-observed CESC event in the SNOWIE dataset).
2. Examine panel 4 of the quicklook figure, which shows faint objects in blue
   and strong objects in red.
3. Adjust `frac_strong_cutoff` until the split looks physically reasonable for
   that case (roughly half the objects by count falling on each side is a good
   starting point).
4. Use the same value for all months in the seasonal run.

---

## Running the tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

The tests do not require any WRF data.  They verify correct output shapes and
types on synthetic reflectivity fields.

---

## Dataset

The SNOWIE 900 m WRF simulation data used in Afrifa et al. (2025, 2026, Part 1)
is available from the NSF NCAR Research Data Archive (RDA), dataset ds604.0.

---

## Citation

If you use this package in published research, please cite:

> Afrifa, F. O., B. Geerts, and co-authors, 2026: Climatology of cold-season
> emergent convection in frontal systems and its impact on orographic
> precipitation. Part 1: Detection algorithm. *Mon. Wea. Rev.*, in review.

And for the SNOWIE case study that motivated the algorithm:

> Afrifa, F. O., B. Geerts, L. Xue, S. Chen, C. Hohman, C. Grasmick, and
> T. Zaremba, 2025: A case study of cold-season emergent orographic convection
> and its impact on precipitation. Part I: Mesoscale analysis. *Mon. Wea. Rev.*,
> **153**, 2229-2250. https://doi.org/10.1175/MWR-D-24-0241.1

The cell detection method is adapted from:

> Yeh, P., and B. A. Colle, 2025: A Comparison of Approaches to Objectively
> Identify Precipitation Structures within the Comma Head of Mid-Latitude
> Cyclones. *J. Atmos. Oceanic Technol.*, **42**, 463-477.
> https://doi.org/10.1175/JTECH-D-24-0055.1

---

## Acknowledgements

This work was supported by the National Science Foundation under grants
AGS-1546939 and AGS-1546986 (SNOWIE), and by the Wyoming Water Development
Office.  Computing resources were provided by NSF NCAR through the Derecho
supercomputer (Computational and Information Systems Laboratory 2023).

---

## Licence

MIT -- see `LICENSE`.
