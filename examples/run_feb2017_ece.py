# =====================================================================
# February 2017, env_mode = "ECE" (stable boundary layer required),
# Table 1 defaults, outputs to /path/to/cesc_output/.
#
# This is the first production run of the packaged algorithm. It
# mirrors notebook cell 8 of Cesc_p1_algorithm_.ipynb but runs from the
# installed cesc package, writes to the package output directory, and
# adds the reconstructed instantaneous rate for attribution when the
# PRATE file is present.
#
# SUBMIT THIS AS A BATCH JOB, not on a login node:
#     qsub examples/run_month.pbs
# Derecho login nodes reap long-running CPU-heavy processes; the first
# attempt at this run was killed there after ~100 of 672 timesteps.
#
# The run is resumable. Part files are written every checkpoint_every
# timesteps into <out>.parts/ and the accumulator is cleared, so memory
# stays flat. If the job is interrupted, resubmit: completed parts are
# detected and skipped and the run continues from the first missing
# timestep, giving a result identical to an uninterrupted run.
# =====================================================================
import os
import sys
import time

import xarray as xr

# Make cesc importable whether or not it has been installed. If you ran
# `python setup_env.py` once in this environment, the plain `import cesc`
# below already works and this block does nothing. If not, it locates the
# package from this file's position (examples/ is one level under root)
# and puts src/ on the path for this run.
try:
    import cesc
except ModuleNotFoundError:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import cesc

from cesc import adapters
from cesc.preflight import check_precip_rate

IN_FILE = ("/path/to/extracted/"
           "WRF_extracted_vars_900m_201702.nc")
WRFOUT_DIR = "/path/to/wrfout_d02/"
PRATE_FILE = ("/path/to/extracted/"
              "final_data_cesc/seasonal/PRATE_surface_201702.nc")
OUT_DIR = "/path/to/cesc_output/"

os.makedirs(OUT_DIR, exist_ok=True)

print("cesc %s | numba: %s" % (
    cesc.__version__,
    __import__("cesc._compat", fromlist=["HAVE_NUMBA"]).HAVE_NUMBA))

params = cesc.Params(env_mode="ECE")     # Table 1 defaults, stable_BL = YES
print("params:", params.to_dict())

ds = adapters.from_extracted_wrf(xr.open_dataset(IN_FILE, decode_times=True))
print("input: %s  (%d timesteps)" % (IN_FILE, ds.sizes["time"]))

prate = None
if os.path.exists(PRATE_FILE):
    prate = xr.open_dataset(PRATE_FILE)["prate_surface"]
    for m in check_precip_rate(prate, name="prate_surface"):
        print(" ", m)
    print("attribution: prate_surface from %s" % PRATE_FILE)
else:
    print("attribution: PRATE file not found; identification-only for "
          "the rate fields (RAINNC increment still written)")

t0 = time.time()
grids, objects = cesc.run_wrf_month(
    ds,
    out_nc_path=OUT_DIR + "CESC_ECE_bs124_grids_201702.nc",
    out_csv_path=OUT_DIR + "CESC_ECE_bs124_objects_201702.csv",
    params=params,
    all_files_path=WRFOUT_DIR,
    precip_rate_da=prate,
    checkpoint_every=48,
    resume=True,
    verbose=True,
)
dt = time.time() - t0

print("\n==================== RUN COMPLETE ====================")
print("wall time: %.1f min" % (dt / 60.0))
kept = objects[objects.kept_env] if objects is not None else None
if kept is not None:
    print("object instances: %d kept of %d detected" % (
        len(kept), len(objects)))
    print("strong / faint  : %d / %d" % (
        int((kept.intensity_cls == "strong").sum()),
        int((kept.intensity_cls == "faint").sum())))
    print("median area     : %.0f km2" % kept.size_km2.median())
print("gridded output  : %sCESC_ECE_bs124_grids_201702.nc" % OUT_DIR)
print("object table    : %sCESC_ECE_bs124_objects_201702.csv" % OUT_DIR)
