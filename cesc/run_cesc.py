# cesc/run_cesc.py
"""
Seasonal batch runner for the CESC pipeline.

Processes a full month of WRF output one time step at a time, keeping peak
RAM near 400 MB regardless of month length.  Gridded outputs are written
incrementally to a pre-allocated NetCDF4 file with an UNLIMITED time axis.
Object statistics are flushed to a staging parquet every 24 steps and merged
at the end.
"""

import numpy as np
import pandas as pd
import os
import gc
from glob import glob

from cesc.cesc_id import run_cesc_id, scan_pi_stable
from cesc.utils import find_wrfout, read_rainnc, read_ctt, StreamingNCWriter

try:
    import xarray as xr
    _XR = True
except ImportError:
    _XR = False

# Default pipeline settings matching Table 1 of Afrifa et al. Part 1
DEFAULT_KWARGS = dict(
    ece_strict          = False,
    env_mode            = "ECE",
    unstable_threshold  = 1.0,
    stable_threshold    = 2.0,
    bg_radius           = 30.0,
    faint_min_diff      = 2.0,
    faint_k_sigma       = 0.30,
    strong_min_diff     = 5.0,
    strong_k_sigma      = 0.30,
    box_sizes           = (20.0, 40.0, 80.0),
    min_area            = 6.0,
    min_support_frac    = 0.30,
    min_support_px      = 5,
    use_advection       = True,
    advection_time      = 20,
    halfwidth           = 10.0,
    seed_stride_px      = 5,
    min_speed           = 2.0,
    frac_strong_cutoff  = 0.5,
    make_plot           = False,
)

REQUIRED_VARS = ["reflectivity_1km", "theta_e", "heights", "u3d", "v3d", "HWP", "ter"]
COORD_VARS = ["XLAT", "XLONG"]


def open_lazy(nc_path: str):
    """
    Open a pre-extracted monthly NC file lazily.

    Loads static fields (terrain, coordinates) into memory once.  All
    time-varying data remains on disk until requested inside the loop.

    Parameters
    ----------
    nc_path : str

    Returns
    -------
    ds_lazy  : xr.Dataset -- lazy, time-varying variables
    ter_da   : 2D DataArray -- terrain height (already loaded)
    xlat     : (ny, nx) float32 -- latitudes (already loaded)
    xlong    : (ny, nx) float32 -- longitudes (already loaded)
    times    : numpy datetime64 array
    """
    if not _XR:
        raise ImportError("xarray is required.")

    ds = xr.open_dataset(nc_path, decode_times=True)
    avail = [v for v in REQUIRED_VARS+COORD_VARS if v in ds]
    miss = [v for v in REQUIRED_VARS+COORD_VARS if v not in ds]
    if miss:
        print(f"[open_lazy] WARNING -- variables absent from {os.path.basename(nc_path)}: {miss}")

    ds = ds[avail]
    for v in ("XTIME", "Times"):
        ds = ds.drop_vars(v, errors="ignore")
    for c in COORD_VARS:
        if c in ds:
            ds = ds.set_coords(c)

    xlat = ds["XLAT"].values.astype(np.float32)
    xlong = ds["XLONG"].values.astype(np.float32)
    ter_da = (ds["ter"].isel(time=0).load()
              if "time" in ds["ter"].dims else ds["ter"].load())
    times = ds.time.values

    print(f"[open_lazy] {os.path.basename(nc_path)} | {len(times)} steps | "
          f"vars: {list(ds.data_vars)}")
    return ds, ter_da, xlat, xlong, times


def run_month_streaming(
    nc_path: str,
    wrfout_path: str,
    out_nc_path: str,
    out_parquet_path: str,
    cesc_id_kwargs: dict | None = None,
    flush_every: int = 24,
    compress: int = 3,
):
    """
    Process a full month one time step at a time with bounded memory use.

    For each hour:
      1. Load one time slice from the lazy dataset (~400 MB)
      2. Run scan_pi_stable to identify ECE columns
      3. Run run_cesc_pipeline to detect and gate CESC objects
      4. Compute the hourly precipitation increment from wrfout RAINNC
      5. Write 2D gridded fields to a streaming NetCDF4 file
      6. Accumulate object rows; flush to parquet every flush_every steps
      7. Delete the slice and call gc.collect()

    Parameters
    ----------
    nc_path          : path to the pre-extracted monthly NC file (WRF format)
    wrfout_path      : directory containing wrfout_d02_* files
    out_nc_path      : output gridded NetCDF path (created fresh)
    out_parquet_path : output object table parquet path
    cesc_id_kwargs  : overrides for any DEFAULT_KWARGS entries
    flush_every      : steps between parquet chunk flushes
    compress         : NetCDF zlib compression level
    """
    pkw = {**DEFAULT_KWARGS, **(cesc_id_kwargs or {})}

    ds, ter_da, xlat, xlong, times = open_lazy(nc_path)
    ny, nx = xlat.shape

    writer = StreamingNCWriter(out_nc_path, ny, nx, xlat, xlong, compress=compress)

    # First-hour RAINNC baseline
    t0 = pd.Timestamp(times[0])
    f0 = find_wrfout(wrfout_path, t0)
    if f0 is None:
        raise RuntimeError(f"No wrfout file found for {t0}")

    rainnc_curr = read_rainnc(f0)
    is_start = (t0.month==10 and t0.day==1 and t0.hour==0)
    fprev = None if is_start else find_wrfout(wrfout_path, t0-pd.Timedelta(hours=1))
    if fprev:
        rainnc_prev = read_rainnc(fprev)
        first_inc = np.clip(rainnc_curr-rainnc_prev, 0, None)
        print("First-hour diff uses previous file:\n"
              f" prev={os.path.basename(rainnc_prev)}\n"
              f" curr={os.path.basename(rainnc_curr)}")
    else:
        rainnc_prev = rainnc_curr.copy()
        first_inc = np.zeros_like(rainnc_curr)
        print("INFO: No previous-hour wrfout found (simulation start or missing). "
              "First-hour increment set to 0.")

    obj_rows = []
    parts = []

    for i, t in enumerate(times):
        ts = pd.Timestamp(t)
        print(f"  {ts}  ({i+1}/{len(times)})", end="\r", flush=True)

        da_t = ds.isel(time=i).load()

        theta_e_np = np.ascontiguousarray(da_t["theta_e"].values.astype(np.float64))
        heights_np = np.ascontiguousarray(da_t["heights"].values.astype(np.float64))
        dtheta_e = np.diff(theta_e_np, axis=0)

        (pi_str, pi_dep, st_str, st_dep,
         pi_h0, pi_h1, st_h0, st_h1) = scan_pi_stable(
            dtheta_e, heights_np,
            max_pi_height=5000.0,
            unstable_threshold=pkw.get("unstable_threshold", 1.0),
            stable_threshold=pkw.get("stable_threshold", 2.0),
        )

        out = run_cesc_id(
            Z_2d_da=da_t["reflectivity_1km"],
            ter_da=ter_da,
            HWP_da=da_t.get("HWP"),
            pi_strength_da=pi_str,
            stable_strength_da=st_str,
            pi_start_h_da=pi_h0,
            pi_end_h_da=pi_h1,
            u3d=da_t["u3d"],
            v3d=da_t["v3d"],
            heights=da_t["heights"],
            **pkw,
        )

        if out.get("fig") is not None:
            try:
                out["fig"].clf()
            except Exception:
                pass
            out["fig"] = None

        # Hourly precip increment
        fi = find_wrfout(wrfout_path, ts) if i>0 else f0
        if fi is None:
            precip_inc = np.zeros((ny,nx), dtype=np.float32)
        else:
            rn = read_rainnc(fi)
            precip_inc = np.clip(rn-rainnc_prev, 0, None).astype(np.float32)
            rainnc_prev = rn

        ctt_arr = read_ctt(fi) if fi else np.full((ny, nx), np.nan, np.float32)
        cesc_m = out["labels_env_da"].values > 0
        conv_p = np.where(cesc_m, precip_inc, 0.0).astype(np.float32)
        conv_pn = np.where(cesc_m, precip_inc, np.nan).astype(np.float32)

        writer.append(ts, {
            "labels_raw"             : out["labels_raw_da"].values,
            "labels_env"             : out["labels_env_da"].values,
            "labels_faint"           : out["labels_faint_da"].values,
            "labels_strong"          : out["labels_strong_da"].values,
            "pi_mask"                : out["masks"]["pi"].values,
            "stable_mask"            : out["masks"]["stable"].values,
            "ece_core"               : out["masks"]["ece_core"].values,
            "corridor"               : out["masks"]["corridor"].values,
            "support"                : out["masks"]["support"].values,
            "faint_det"              : out["masks"]["faint_det"].values,
            "strong_det"             : out["masks"]["strong_det"].values,
            "ece_dist_km"            : out["masks"]["ece_dist_km"].values,
            "Z_bg"                   : out["masks"]["Z_bg"].values,
            "Enh"                    : out["masks"]["Enh"].values,
            "rainnc_inc"             : precip_inc,
            "convective_precip"      : conv_p,
            "convective_precip_masked": conv_pn,
            "CTT"                    : ctt_arr,
        })

        objs = out.get("objects", [])
        if objs:
            import pandas as _pd
            df = _pd.DataFrame(objs)
            df["time"] = ts
            df["object_uid"] = (ts.strftime("%Y%m%d%H")+"_"
                                + df["id"].astype(str).str.zfill(4))
            obj_rows.append(df)

        if len(obj_rows) >= flush_every or i == len(times)-1:
            if obj_rows:
                import pandas as _pd
                chunk = _pd.concat(obj_rows, ignore_index=True)
                pp = out_parquet_path.replace(".parquet",
                                              f"_part{len(parts):04d}.parquet")
                chunk.to_parquet(pp, index=False)
                parts.append(pp)
                obj_rows = []
                del chunk
                gc.collect()

        del da_t, out, theta_e_np, heights_np, dtheta_e
        del pi_str, pi_dep, st_str, st_dep, pi_h0, pi_h1, st_h0, st_h1
        del precip_inc, ctt_arr, conv_p, conv_pn, cesc_m
        gc.collect()

    writer.close()

    if parts:
        import pandas as _pd
        df_all = _pd.concat([_pd.read_parquet(p) for p in parts], ignore_index=True)
        df_all.to_parquet(out_parquet_path, index=False)
        for p in parts:
            os.remove(p)
        print(f"\n[run_month_streaming] Objects -> {out_parquet_path} "
              f"({len(df_all)} rows)")
    else:
        print("\n[run_month_streaming] No objects detected this month.")

    ds.close()
    print(f"[run_month_streaming] Done -> {out_nc_path}")
