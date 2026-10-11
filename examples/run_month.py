# =====================================================================
# Run one CESC month, for any month and either environment mode.
#
#     python examples/run_month.py 201702 ECE
#     python examples/run_month.py 201801 PI_only
#
# Generalizes run_feb2017_ece.py, which is kept as the worked example.
# Verified against the original pipeline for February 2017 ECE: same
# environment masks, same detections, same object pixels, same classes
# (only object id numbering differs, which nothing depends on).
#
# Submit via the job array in examples/run_season.pbs rather than
# calling this directly for a whole season.
# =====================================================================
import os
import sys
import time
 
import xarray as xr
 
try:
    import cesc
except ModuleNotFoundError:
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    import cesc
 
from cesc import adapters
from cesc.preflight import check_precip_rate
 
# ---------------------------------------------------------------------
# Paths. The two cold seasons keep their raw wrfout in different places.
# ---------------------------------------------------------------------
IN_TMPL = ("/path/to/extracted/"
           "WRF_extracted_vars_900m_%s.nc")
PRATE_TMPL = ("/path/to/extracted/"
              "final_data_cesc/seasonal/PRATE_surface_%s.nc")
WRFOUT_2017 = "/path/to/wrfout_d02/"
WRFOUT_2018 = ("/path/to/wrfout_backup/"
               "snowie_seasonal_data_900m/")
OUT_DIR = "/path/to/cesc_output/"
 
SEASON_2017 = ("201611", "201612", "201701", "201702", "201703")
SEASON_2018 = ("201710", "201711", "201712", "201801", "201802",
               "201803", "201804")
 
 
def wrfout_dir_for(month):
    if month in SEASON_2017:
        return WRFOUT_2017
    if month in SEASON_2018:
        return WRFOUT_2018
    raise ValueError("%s is not in either cold season" % month)
 
 
def main(month, env_mode):
    if month not in SEASON_2017 + SEASON_2018:
        raise SystemExit(
            "month %s is outside the two cold seasons:\n  %s\n  %s"
            % (month, " ".join(SEASON_2017), " ".join(SEASON_2018)))
 
    in_file = IN_TMPL % month
    prate_file = PRATE_TMPL % month
    wrfout_dir = wrfout_dir_for(month)
    tag = "ECE" if env_mode.upper() == "ECE" else "PI_ONLY"
 
    out_nc = OUT_DIR + "CESC_%s_bs124_grids_%s.nc" % (tag, month)
    out_csv = OUT_DIR + "CESC_%s_bs124_objects_%s.csv" % (tag, month)
 
    # Fail early and loudly rather than partway through a queued job.
    missing = [p for p in (in_file, wrfout_dir) if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing input(s):\n  " + "\n  ".join(missing))
    os.makedirs(OUT_DIR, exist_ok=True)
 
    print("month      : %s" % month)
    print("env_mode   : %s" % env_mode)
    print("input      : %s" % in_file)
    print("wrfout     : %s" % wrfout_dir)
    print("out grids  : %s" % out_nc)
    print("out objects: %s" % out_csv)
    print("cesc %s | numba: %s"
          % (cesc.__version__,
             __import__("cesc._compat", fromlist=["HAVE_NUMBA"]).HAVE_NUMBA))
 
    params = cesc.Params(env_mode=env_mode)
    ds = adapters.from_extracted_wrf(xr.open_dataset(in_file))
    print("timesteps  : %d" % ds.sizes["time"])
 
    prate = None
    if os.path.exists(prate_file):
        prate = xr.open_dataset(prate_file)["prate_surface"]
        for m in check_precip_rate(prate, name="prate_surface"):
            print("  %s" % m)
    else:
        print("NOTE: no PRATE file for %s; running without instantaneous "
              "rate attribution (RAINNC increment still written)." % month)
 
    t0 = time.time()
    grids, objects = cesc.run_wrf_month(
        ds,
        out_nc_path=out_nc,
        out_csv_path=out_csv,
        params=params,
        all_files_path=wrfout_dir,
        precip_rate_da=prate,
        checkpoint_every=48,
        resume=True,
        verbose=True,
    )
    dt = time.time() - t0
 
    print("\n==================== RUN COMPLETE ====================")
    print("month / mode  : %s / %s" % (month, env_mode))
    print("wall time     : %.1f min" % (dt / 60.0))
    print("timesteps     : %d (complete: %s)"
          % (grids.sizes.get("time", 0),
             grids.attrs.get("cesc_complete", "?")))
    if objects is not None:
        kept = objects[objects.kept_env]
        print("objects       : %d kept of %d detected" % (len(kept),
                                                          len(objects)))
        print("strong / faint: %d / %d"
              % (int((kept.intensity_cls == "strong").sum()),
                 int((kept.intensity_cls == "faint").sum())))
        print("median area   : %.0f km2" % kept.size_km2.median())
 
 
if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(
            "usage: python run_month.py YYYYMM {ECE|PI_only}\n"
            "  e.g. python run_month.py 201702 ECE")
    main(sys.argv[1], sys.argv[2])
 