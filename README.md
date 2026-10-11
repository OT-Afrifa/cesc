# cesc

`cesc` is the Python implementation of the algorithm of Afrifa et al. (2026a, *Journal of the Atmospheric Sciences*, in revision) for identifying convection embedded within or emergent from cold-season stratiform clouds (CESC). It finds convective cells that are rooted above a stable near-surface layer, using profiles of equivalent potential temperature, the winds in the unstable layer and a two-dimensional reflectivity field. The algorithm was built and tested on 900 m WRF output from the SNOWIE campaign over Idaho, but it runs on any gridded dataset that supplies those fields in the units listed below. You can call it from a Python script, a Jupyter notebook or the `cesc` command line.

## 1. What the algorithm does

Each timestep is processed on its own in three stages.

**Stage 1 - the supportive environment:** The algorithm scans the `theta_e` profile in every grid column for a layer of potential instability (PI), where `theta_e` falls with height, sitting above a stable layer near the surface, where `theta_e` rises. A column that has both is part of the embedded/emergent convection environment (ECE). Because a cell triggered in the ECE keeps moving after it forms, the ECE is then extended downwind along the mean wind in the PI layer. The ECE and this downwind corridor together make up the support mask. Setting `env_mode="PI_only"` drops the stable-layer requirement, so the support mask then also admits surface-based convection.

**Stage 2 - convective cells:** Cells are found in the 2-D reflectivity field by comparing each grid point with a background reflectivity computed over several box sizes. A point counts as convective when it exceeds that background by an adaptive threshold, and touching convective points are grouped into objects with a watershed step. This stage runs over the whole domain and does not look at the thermodynamic environment at all.

**Stage 3 - the environmental gate:** An object from Stage 2 is kept as CESC only if at least 30% of its area and at least 5 of its grid points fall inside the support mask. Objects that fail the gate are dropped from the CESC population. Each kept object is then labeled strong or faint from the share of its area that exceeds the strong threshold.

Surface precipitation plays no part in any of the three stages. Once the objects exist, an instantaneous surface precipitation rate can be masked by the object labels to get the precipitation that CESC produces. Use a rate for this step and not an hourly accumulation such as WRF `RAINNC`. This is because a short-lived cell moving at 15 to 30 m s⁻¹, for example, drops much of its hourly accumulation outside the footprint it has at the output time, so an accumulation assigns too little precipitation to CESC. `cesc check` warns when a precipitation field looks like an accumulation.

The package does not track cells from one output time to the next. With hourly output such as the 900 m WRF runs, cells in these storms last roughly 20 to 40 min, and a cell moving at 20 m s⁻¹ travels about 70 km between output times, so matching cells across hours would mostly match different cells. If your output is a few minutes apart, `cesc.contrib.linker` offers an experimental advection-based linker. The package also leaves units, variable names and precipitation types as you supply them.

## 2. Required inputs

| Field | Dimensions | Units | Used for |
| --- | --- | --- | --- |
| Reflectivity at one level (1 km AGL in Afrifa et al. 2026a) | `(time, y, x)` | dBZ | Stage 2 |
| Equivalent potential temperature, `theta_e` | `(time, z, y, x)` | K | Stage 1 |
| Height of each model level | `(time, z, y, x)` | m AGL | Stage 1 |
| Zonal and meridional wind, `u` and `v` | `(time, z, y, x)` | m s⁻¹ | Downwind corridor |
| Grid spacing, `dx` and `dy` | scalar | km | Areas and distances |
| Instantaneous surface precipitation rate (optional) | `(time, y, x)` | mm h⁻¹ | Precipitation attribution only |

Reflectivity must be in dBZ, not linear units, and heights must be in meters above ground, not kilometers or meters above sea level. If your dataset has temperature, pressure and humidity but no `theta_e`, the WRF adapter computes `theta_e` with the Bolton (1980) formula. For other datasets, compute `theta_e` yourself before building the input. Skip the precipitation rate if you only need the objects and their properties.

The adapters turn your data into the dataset the pipeline expects. `adapters.from_arrays(...)` takes NumPy arrays already in memory, and `adapters.from_extracted_wrf(...)` takes an extracted WRF file opened with xarray.


## 3. Installation

The package needs Python 3.7 or later. Installing it with pip also installs everything the algorithm needs to run: NumPy, SciPy, pandas, xarray, scikit-image, netCDF4 (so xarray can open NetCDF files), PyYAML (for `cesc run config.yaml`) and Numba. Numba changes only the run time, and the results are identical with or without it. The oldest versions tested, and the floors pip enforces, are those of the environment that ran the published production months: Python 3.7.12, NumPy 1.21.6, SciPy 1.7.3, pandas 1.3.5, xarray 0.20.2, scikit-image 0.19.2, netCDF4 1.6.5, PyYAML 6.0 and Numba 0.55.1. Install straight from GitHub with

```bash
python -m pip install "git+https://github.com/OT-Afrifa/cesc.git"
```

or, from the top of a cloned copy of the repository,

```bash
python -m pip install .
```

Two groups of packages stay optional because pip often cannot build them from source. `wrf-python`, which you need only to read raw `wrfout` files, comes in with `python -m pip install ".[wrf]"`, and matplotlib and cartopy, which you need only for the quicklook figure, come in with `".[plot]"`. On HPC systems both install far more reliably from conda-forge. The repository ships an `environment.yml` that builds a complete environment in one step, after which you install `cesc` itself into it:

```bash
conda env create -f environment.yml
conda activate cesc
python -m pip install -e .
```

If you plan to edit the code or run analyses from inside the repository, install in editable mode instead, so that changes under `src/cesc/` take effect without reinstalling. Activate the environment you will run in (on NCAR systems, for example, `conda activate /path/to/your/environment`), confirm with `which python` that it is the interpreter you expect, and run from the repository root

```bash
python setup_env.py
```

`python setup_env.py --check` reports the state of the installation without changing anything. A working editable install prints

```text
Package root : /path/to/cesc
src/ exists  : True
import cesc  : OK, version 0.1.0
resolves to  : /path/to/cesc/src/cesc/__init__.py
```

and `cesc --help` should list the two subcommands, `check` and `run`.

In Jupyter, the kernel has to use the same environment you installed into. Running `import sys; print(sys.executable)` and then `import cesc; print(cesc.__file__)` in a cell should show that environment's interpreter and, for an editable install, a path ending in `src/cesc/__init__.py`. If `import cesc` works in a terminal and fails in a notebook, the kernel is pointing at a different environment. Switch the kernel or restart it.

## 4. Minimal working example

The script below builds a small synthetic domain on a 0.9 km grid with a stable layer under a layer of potential instability in its western half only. It places one convective cell inside that environment and one in the east, about 80 km away. Both cells pass Stage 2, but only the western cell passes the gate in Stage 3.

```python
import numpy as np
import cesc
from cesc import adapters
from cesc.environment import scan_pi_stable

nz, ny, nx = 20, 120, 160
z = np.linspace(0, 6000, nz)                      # m AGL
heights = np.broadcast_to(z[:, None, None], (nz, ny, nx)).copy()

# Background theta_e increases slowly with height everywhere (no PI).
theta_e = np.broadcast_to((280 + 0.0005 * z)[:, None, None], (nz, ny, nx)).copy()

# Western half: theta_e rises 4 K over the lowest 1.5 km (stable layer),
# then falls 3 K between 2 and 3 km (PI layer).
rise = 4.0 * np.clip(z / 1500, 0, 1)
drop = np.where(z > 2000, -3.0 * np.clip((z - 2000) / 1000, 0, 1), 0.0)
theta_e[:, :, : nx // 2] = (280 + rise + drop)[:, None, None]

# Reflectivity: 8 dBZ background and two 30 dBZ cells.
reflectivity = np.full((ny, nx), 8.0)
reflectivity[50:60, 30:40] = 30.0     # inside the ECE
reflectivity[50:60, 130:140] = 30.0   # far outside it

ds = adapters.from_arrays(
    reflectivity=reflectivity,
    theta_e=theta_e,
    heights=heights,
    u3d=np.full((nz, ny, nx), 10.0),  # 10 m/s westerly
    v3d=np.zeros((nz, ny, nx)),
    dx_km=0.9,
    dy_km=0.9,
)

params = cesc.Params(env_mode="ECE")  # Table 1 defaults
da = ds.isel(time=0)

# Stage 1: find the PI and stable layers in every column.
dtheta_e = np.diff(da.theta_e.values, axis=0)
pi, _, stable, _, h0, h1, _, _ = scan_pi_stable(
    dtheta_e,
    da.heights.values,
    params.max_pi_height_m,
    params.unstable_threshold,
    params.stable_threshold,
    params.min_pi_depth_m,
    params.max_stable_height_m,
    params.theta_eps,
)

# Stages 2 and 3: detect cells and apply the environmental gate.
result = cesc.identify_time(
    da.reflectivity_1km, pi, stable, h0, h1,
    u3d=da.u3d, v3d=da.v3d, heights=da.heights,
    params=params,
)

cesc.print_summary(result["summary"])
```

Running the script prints

```text
2 watershed object; 1 kept (1 strong, 0 faint); 1 dropped at gate; mean area 80 km2; CESC area fraction 0.52%; precip fraction n/a
```

Both cells pass Stage 2. The western cell is kept, and its 30 dBZ core is 22 dBZ above the background, so it is labeled strong. The eastern cell is dropped at the gate, because none of it lies in the ECE or in the corridor downwind of it. Each cell covers 100 grid points (81 km² at 0.9 km spacing), or 0.52% of the 15,552 km² domain. The precipitation fraction is `n/a` because the example supplies no precipitation rate.

## 5. Running a month of WRF output

For a month of extracted WRF variables, `run_wrf_month` loops over the timesteps and writes the gridded output and the object table:

```python
import xarray as xr
import cesc
from cesc import adapters

ds = adapters.from_extracted_wrf(xr.open_dataset("WRF_extracted_vars_900m_201702.nc"))

grids, objects = cesc.run_wrf_month(
    ds,
    out_nc_path="out/grids.nc",
    out_csv_path="out/objects.csv",
    params=cesc.Params(env_mode="ECE"),
    all_files_path=None,
)
```

If some fields, such as `RAINNC` or cloud-top temperature, exist only in the original hourly `wrfout` files, pass the directory holding those files as `all_files_path`. The NetCDF file holds the gridded masks and labels, and the CSV holds one row per object. Object tables are written as CSV rather than Parquet, so `pyarrow` is not needed.

Run `cesc check data.nc` on each input file before a long job. It catches the problems that otherwise surface hours into a run: reflectivity in linear units, heights in kilometers, missing grid spacing and accumulated precipitation passed in as a rate. On an HPC system, submit each month as its own batch job, so a failed month can be rerun without repeating the others. The `examples/` directory has the quickstart notebook, a script that runs one month (`run_month.py`), a PBS batch script for a season and a YAML configuration for `cesc run config.yaml`.

## 6. Default parameters

All parameters live in `cesc.Params`, and the defaults are the production values in Table 1 of the algorithm paper. Any of them can be changed by keyword, for example `cesc.Params(box_sizes=(20.0, 40.0, 80.0))`, and `print(cesc.Params())` lists the full set. The ones that set the algorithm's behavior are

| Parameter | Default | Meaning |
| --- | --- | --- |
| `env_mode` | `"ECE"` | `"ECE"` requires the stable layer; `"PI_only"` drops it |
| `stable_threshold` | 2.0 K | Rise in `theta_e` that defines the stable layer |
| `max_stable_height_m` | 3000 m | Stable layer must lie within this height AGL |
| `unstable_threshold` | 1.0 K | Drop in `theta_e` that defines the PI layer |
| `min_pi_depth_m` | 500 m | Minimum depth of the PI layer |
| `max_pi_height_m` | 5000 m | PI layer must lie below this height AGL |
| `box_sizes` | (10, 20, 40) km | Background reflectivity scales |
| `faint_min_diff`, `strong_min_diff` | 5 dBZ, 10 dBZ | Minimum excess over the background |
| `faint_k_sigma`, `strong_k_sigma` | 0.3, 0.3 | Weight on local variability in each adaptive threshold |
| `bg_radius` | 30 km | Radius used in the watershed step |
| `min_area_km2` | 4.05 km² | Smallest object kept (5 grid points at 0.9 km) |
| `advection_time_min`, `halfwidth_km` | 20 min, 10 km | Length and half-width of the downwind corridor |
| `min_support_frac`, `min_support_px` | 0.3, 5 | Overlap an object needs with the support mask |
| `frac_strong_cutoff` | 0.5 | Share of the object above the strong threshold for the strong label |

The remaining fields (`theta_eps`, `use_advection`, `min_speed_ms`, `seed_stride_px`, `precip_floor_dbz`, `length_cut_km` and `aspect_cut`) are documented in `src/cesc/params.py`.

Read the sensitivity tests in the paper before changing any of these for production work. Halving or doubling `advection_time_min` moves CESC area by about 4% and precipitation volume by -3.3% to +4.6%. Setting `bg_radius` to 10 km or 50 km changes CESC area by +14.2% and -4.8%. `box_sizes` matters most: doubling it to (20, 40, 80) km raises CESC area by about 46% and precipitation volume by about 62%, so choose the box sizes from the size of the cells in your own data and grid.

## 7. Comparing with the original research pipeline

If you compare this package with output from the research scripts behind the papers, read `PORTING_NOTES.md` first. The package corrects four things in those scripts, and two of them change existing output. The research runner never updated the previous `RAINNC` value after the first hour of a month, so the second timestep of every month carries two hours of accumulation. It also attached the downstream metrics (`area_ece_px`, `area_ds_px`, `area_sup_px`, `frac_support`, `L_ds_km`, `W_ds_km`) to the wrong objects at any timestep where the gate dropped an object. Columns written before that step, and the gate decisions themselves, are unaffected. For February 2017, the package reproduces the research `labels_env` footprint at all 672 hourly timesteps, with only the object numbering changed.

## 8. Citation

If you use `cesc`, please cite the algorithm paper and the case study it builds on. `CITATION.cff` holds the same information in machine-readable form.

Afrifa, F. O., B. Geerts, L. Xue, and J. Kukulies, 2026a: Cold-season emergent convection in frontal systems and its impact on orographic precipitation. Part 1: Detection algorithm. *J. Atmos. Sci.*, in revision.

Afrifa, F. O., B. Geerts, L. Xue, S. Chen, C. Hohman, C. Grasmick, and T. Zaremba, 2025: A case study of cold-season emergent orographic convection and its impact on precipitation. Part I: Mesoscale analysis. *Mon. Wea. Rev.*, **153**, 2229-2250, https://doi.org/10.1175/MWR-D-24-0241.1.

Afrifa, F. O., and co-authors, 2026: A case study of cold-season emergent orographic convection and its impact on precipitation. Part 2: High-resolution LES analysis of convective cell evolution and precipitation processes. *Mon. Wea. Rev.*, **154**, 1269-1290, https://doi.org/10.1175/MWR-D-25-0157.1.

The cell detection follows Yeh and Colle (2025), the `theta_e` formula follows Bolton (1980) and the definition of potential instability follows Schultz and Schumacher (1999).

Bolton, D., 1980: The computation of equivalent potential temperature. *Mon. Wea. Rev.*, **108**, 1046-1053.

Schultz, D. M., and P. N. Schumacher, 1999: The use and misuse of conditional symmetric instability. *Mon. Wea. Rev.*, **127**, 2709-2732.

Yeh, P., and B. A. Colle, 2025: A comparison of approaches to objectively identify precipitation structures within the comma head of mid-latitude cyclones. *J. Atmos. Oceanic Technol.*, **42**, 463-477, https://doi.org/10.1175/JTECH-D-24-0055.1.

## 9. Repository layout

```text
cesc/
├── README.md
├── setup_env.py
├── pyproject.toml
├── environment.yml
├── CITATION.cff
├── PORTING_NOTES.md
├── src/cesc/      the package
├── examples/      quickstart notebook, run scripts, a YAML configuration and a PBS script
└── tests/         pytest suite
```

The package code is under `src/cesc/`. Questions and bug reports go to the GitHub issue tracker at https://github.com/OT-Afrifa/cesc/issues.