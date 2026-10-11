"""
pipeline.py

    A single timestep pipeline that ports convective_id_pipeline
"""

import os
import shutil
from glob import glob

import numpy as np
import pandas as pd
import xarray as xr

from .params import Params
from .distance import estimate_dx_dy_km

from . import environment as env
from . import detect as det
from . import gate as gt
from . import attribute as attr


def identify_time(Z_2d_da, pi_strength, stable_strength=None, pi_start_h=None,
                  pi_end_h=None, u3d=None, v3d=None, heights=None,
                  precip_rate=None, aux_fields=None, params=None,
                  make_figure=False, ter_da=None, title=None, level_title=None,
                  extent=None,
                  ):
    """
    Full two phase-CESC identification for one timestep.

    Parameters
    ----------
    Z_2d_da: 2-D xr.DataArray
        Reflectivity (dBZ) with XLAT and XLONG coordinates. The default level
        guidance is 1 km AGL. Column maximum gives essentially the same objects
        for shallow cold-season cells.

    pi_strength, stable_strength: (ny, nx) float
        Outputs of environment.
    """
    p = params or Params()

    if pi_strength is None:
        raise ValueError("pi_strength must be provided -"
                         "run 'environment.scan_pi_stable' first.")

    lats = np.asarray(Z_2d_da.XLAT)
    lons = np.asarray(Z_2d_da.XLONG)
    dx_km, dy_km = estimate_dx_dy_km(lats, lons)
    grid_km = 0.5 * (dx_km + dy_km)

    Z = np.asarray(Z_2d_da.values, dtype=np.float64)
    Zbg, Enh = gt.background_and_enhancement(Z, p.bg_radius, grid_km)
    strat_base = np.isfinite(Z) & (Z >= p.precip_floor_dbz)

    def _da(arr, name):
        return xr.DataArray(arr, coords=Z_2d_da.coords,
                            dims=Z_2d_da.dims, name=name)

    # ----- Stage 1 -----
    pi_mask, stable_mask, both_mask, ece_core = env.environment_masks(
        pi_strength, stable_strength, p)

    corridor = np.zeros_like(ece_core, dtype=bool)
    u_bar_ece = np.zeros_like(ece_core, dtype=np.float32)
    v_bar_ece = np.zeros_like(ece_core, dtype=np.float32)

    if p.use_advection:
        if u3d is None or v3d is None or heights is None:
            raise ValueError("use_advection requires u3d, v3d and heights -"
                             "set 'Params(use_advectiob=False)' to gate on"
                             "spatial overlap only.")

        if ece_core.any():
            DEFAULT_LO, DEFAULT_HI = 2000.0, 5000.0
            u_layer = u3d.isel(bottom_top=slice(0, -1)).values.astype(np.float32)
            v_layer = v3d.isel(bottom_top=slice(0, -1)).values.astype(np.float32)
            z_full = np.asarray(heights.values, dtype=np.float32)
            h_start = (np.asarray(pi_start_h, dtype=np.float32)
                       if pi_start_h is not None
                       else np.full(Z.shape, DEFAULT_LO, np.float32))
            h_end = (np.asarray(pi_end_h, dtype=np.float32)
                     if pi_end_h is not None
                     else np.full(Z.shape, DEFAULT_HI, np.float32))
            u_bar_np, v_bar_np = env.layer_mean_wind(
                u_layer, v_layer, z_full, h_start, h_end,
                DEFAULT_LO, DEFAULT_HI,
            )
            u_bar_ece = np.ma.masked_where(~ece_core, u_bar_np)
            v_bar_ece = np.ma.masked_where(~ece_core, v_bar_np)

            # layer_mean_wind returns NaN where a column had no integration
            # layer so the 1-pixel core dilation can sit over such columns.
            # Hypot on the filled values avoids the sqrt-of-NaN warning, and
            # the explicit nan_to_num keeps a NaN lookahead from becoming a
            # garbage integer in downstream_corridor_2d (whose seed loop also
            # skips non-finite winds, so this is belt and braces).

            u_filled = np.nan_to_num(np.ma.filled(u_bar_ece, np.nan), nan=0.0)
            v_filled = np.nan_to_num(np.ma.filled(v_bar_ece, np.nan), nan=0.0)
            spd = np.hypot(u_filled, v_filled)
            lookahead_km = spd * 3.6 * (p.advection_time_min / 60.0)
            corridor = env.downstream_corridor_2d(
                ece_core, u_bar_ece, v_bar_ece, grid_km,
                lookahead_km=lookahead_km, halfwidth=p.halfwidth_km,
                seed_stride_px=p.seed_stride_px, min_speed=p.min_speed_ms,
            )

    support_mask = ece_core | corridor

    # ----- Stage 2 -----
    faint_det, strong_det, labels_cc, _ = det.detect_objects(
        Z_2d_da, strat_base, lats, lons, p.box_sizes,
        p.faint_min_diff, p.faint_k_sigma,
        p.strong_min_diff, p.strong_k_sigma,
        p.min_area_km2, p.length_cut_km, p.aspect_cut,
    )
    n_cc = int(labels_cc.max())

    labels_ws = gt.watershed_split(faint_det, Enh, grid_km, p.min_area_km2)
    labels_raw_da = _da(labels_ws, 'convective_labels_raw')

    # ----- Stage 3 -----
    (labels_env, labels_faint, labels_strong, objs, n_dropped_gate,
     ece_dist_km) = gt.gate_and_classify(
        labels_ws, faint_det, strong_det, ece_core, corridor, support_mask,
        lats, lons, dx_km, dy_km, p, aux_fields=aux_fields,
     )

    objs = gt.add_downstream_metrics(
        labels_env, ece_core, corridor, objs, u_bar_ece, v_bar_ece,
        dx_km, dy_km,
    )

    conv_rate, precip_frac = attr.attribute_precip(labels_env, precip_rate)
    objs = attr.per_object_rates(labels_env, precip_rate, objs, dx_km * dy_km)

    kept = [o for o in objs if o['kept_env']]
    n_strong = sum(1 for o in kept if o['intensity_cls'] == 'strong')
    summary = dict(
        n_watershed_objects=int(labels_ws.max()),
        n_connected_components=n_cc,
        n_kept=len(kept),
        n_dropped_gate=n_dropped_gate,
        n_strong=n_strong,
        n_faint=len(kept) - n_strong,
        mean_area_km2=(float(np.mean([o['size_km2'] for o in kept]))
                       if kept else np.nan),
        cesc_area_frac=float((labels_env > 0).mean()),
        precip_frac=precip_frac,
    )

    masks = {
        'pi': _da(pi_mask, 'pi_mask'),
        'stable': _da(stable_mask, 'stable_mask'),
        'both': _da(both_mask, 'both_mask'),
        'ece_core': _da(ece_core, 'ece_core'),
        'ece_dist_km': _da(ece_dist_km, 'ece_dist_km'),
        'corridor': _da(corridor, 'corridor'),
        'support': _da(support_mask, 'support'),
        'faint_det': _da(faint_det, 'faint_det'),
        'strong_det': _da(strong_det, 'strong_det'),
        'Z_bg': _da(Zbg, 'Z_bg'),
        'Enh': _da(Enh, 'Z_enh'),
    }

    fig = None
    if make_figure:
        from .quicklook import quicklook_figure
        fig = quicklook_figure(Z_2d_da, ter_da, masks, labels_env, labels_faint,
                               labels_strong, p, title=title, level_title=level_title,
                               extent=extent)

    return dict(
        labels_raw_da=labels_raw_da,
        labels_env_da=_da(labels_env, 'convective_labels_env_gated'),
        labels_faint_da=_da(labels_faint, 'convective_labels_faint'),
        labels_strong_da=_da(labels_strong, 'convective_labels_strong'),
        conv_rate=(None if conv_rate is None else _da(conv_rate, 'convective_rate')),
        masks=masks, objects=objs,
        u_bar_np=u_bar_ece, v_bar_np=v_bar_ece,
        summary=summary, fig=fig,
    )


def print_summary(summary, prefix=""):
    """
    One-block sanity summary. Silent success does not guarantee success.
    """
    s = summary
    frac = (
        "n/a" if s.get("precip_frac") is None or not np.isfinite(s.get("precip_frac", 
                                                                       np.nan))
        else "%.1f%%" % (100 * s["precip_frac"])
    )
    print(
        "%s%d watershed object; %d kept (%d strong, %d faint); "
        "%d dropped at gate; mean area %.0f km2; "
        "CESC area fraction %.2f%%; precip fraction %s" % (
            prefix, s["n_watershed_objects"], s["n_kept"], s["n_strong"],
            s["n_faint"], s["n_dropped_gate"], s["mean_area_km2"] if np.isfinite(
                s["mean_area_km2"]
            ) else 0, 100 * s["cesc_area_frac"], frac
        )
    )


# ---------------------------------------------------------------------
# Month runner for the extracted 900 m WRF files
# ---------------------------------------------------------------------

def _is_netcdf_safe_attr(v):
    if isinstance(v, (str, bytes, int, float, np.integer, np.floating)):
        return True

    if isinstance(v, (list, tuple)):
        return all(_is_netcdf_safe_attr(x) for x in v)

    if isinstance(v, np.ndarray):
        return np.issubdtype(v.dtype, np.number) or v.dtype.kind in ("U", "S")

    return False


def scrub_netcdf_unfriendly_attrs(ds):
    """
    Remove attrs (cartopy pr wrf-python objects) NetCDF cannot write.
    """
    ds = ds.copy()
    for k in list(ds.attrs):
        if k == "projection" or not _is_netcdf_safe_attr(ds.attrs[k]):
            del ds.attrs[k]
    for name in list(ds.variables):
        attrs = dict(ds[name].attrs)
        changed = False
        for k, v in list(attrs.items()):
            if k == "projection" or not _is_netcdf_safe_attr(v):
                attrs.pop(k, None)
                changed = True
        if changed:
            ds[name].attrs = attrs
    return ds
 
 
def safe_to_netcdf(ds, out_path, encoding=None, **kw):
    """Write; on attr TypeError/ValueError scrub and retry once."""
    try:
        ds.to_netcdf(out_path, encoding=encoding, **kw)
    except (TypeError, ValueError) as e:
        print("[safe_to_netcdf] first attempt failed (%s): %s"
              % (e.__class__.__name__, e))
        print("[safe_to_netcdf] scrubbing non-serializable attrs, retrying")
        scrub_netcdf_unfriendly_attrs(ds).to_netcdf(out_path,
                                                    encoding=encoding, **kw)
        print("[safe_to_netcdf] success after scrub")
 
 
def _find_wrfout_for_time(base_dir, ts):
    """The wrfout file for the exact hour, else best effort in the day."""
    ts = pd.Timestamp(ts)
    files = sorted(glob("%swrfout_d02_%s*" % (base_dir,
                                              ts.strftime("%Y-%m-%d_%H"))))
    if files:
        return files[0]
    files = sorted(glob("%swrfout_d02_%s_*" % (base_dir,
                                               ts.strftime("%Y-%m-%d"))))
    if files:
        return files[-1] if ts.hour >= 12 else files[0]
    return None
 
 
def run_wrf_month(ds, out_nc_path, out_csv_path=None, params=None,
                  times=None, all_files_path=None, precip_rate_da=None,
                  compress_level=3, checkpoint_every=48, resume=True,
                  keep_parts=False, verbose=True, out_parquet_path=None):
    """Run the pipeline over every timestep of an extracted-WRF month.
 
    Parameters
    ----------
    ds : xr.Dataset
        The extracted 900 m month file. Required variables: theta_e
        (time, bottom_top, y, x), heights (same shape), reflectivity_1km
        (time, y, x), u3d, v3d, ter, and coordinates XLAT, XLONG, time.
        HWP is used for the object table when present.
    out_nc_path : str
        Output path for the gridded NetCDF.
    out_csv_path : str
        Output path for the object table, written as CSV. The table is
        one row per object instance per timestep. CSV rather than
        parquet: it needs no pyarrow, reads anywhere, and these tables
        are small (a month of February 2017 is a few MB).
    out_parquet_path : str, optional
        Deprecated alias for out_csv_path. A .parquet suffix is replaced
        with .csv; the file written is always CSV.
    params : Params
    times : array-like, optional
        Subset of ds.time to run; defaults to all.
    all_files_path : str, optional
        Directory of raw wrfout files for the hourly RAINNC increment
        and CTT. When None those variables are skipped and the run is
        identification-only.
    precip_rate_da : xr.DataArray, optional
        Instantaneous surface rate (time, y, x) in mm/h for
        attribution, for example the reconstructed prate_surface.
    compress_level : int
        zlib level for the NetCDF encoding.
    checkpoint_every : int, optional
        Write a part file every N timesteps and clear the in-memory
        accumulator, so memory stays flat and an interrupted run keeps
        its completed blocks.
    resume : bool
        Skip timesteps already present in the parts directory and
        continue from the first missing one. The result is identical to
        an uninterrupted run.
    keep_parts : bool
        Keep the <out_nc_path>.parts working directory after a
        successful run. The default removes it once the stitched
        outputs are verified, since it is then redundant and roughly
        doubles the disk used per month. Parts of an incomplete run are
        always kept, because they are what makes a resubmission resume.
    verbose : bool
 
    Returns
    -------
    grids_ds : xr.Dataset
    objects_df : pandas.DataFrame or None
    """
    if out_csv_path is None:
        if out_parquet_path is None:
            raise TypeError("out_csv_path is required")
        out_csv_path = out_parquet_path
    if out_csv_path.endswith(".parquet"):
        out_csv_path = out_csv_path[: -len(".parquet")] + ".csv"
 
    p = params or Params()
    from .preflight import check_dataset
    check_dataset(ds, params=p, raise_on_error=True)
 
    have_wrfout = all_files_path is not None
    if have_wrfout:
        try:
            from netCDF4 import Dataset as NCDataset
            from wrf import getvar
        except ImportError:
            raise ImportError(
                "all_files_path was given but netCDF4/wrf-python are not "
                "importable. Install wrf-python or pass "
                "all_files_path=None for an identification-only run.")
 
    def _da_at(da, xlat, xlong, time_val):
        da = da.assign_coords(XLAT=xlat, XLONG=xlong)
        da = da.expand_dims(time=[np.datetime64(pd.Timestamp(time_val))])
        drop = [c for c in da.coords if c not in ("time", "XLAT", "XLONG")]
        if drop:
            da = da.reset_coords(drop, drop=True)
        return da
 
    def _da_now(da, da_t):
        return _da_at(da, da_t.XLAT, da_t.XLONG, da_t.time.values)
 
    for v in ("XTIME", "Times"):
        if v in ds.coords or v in ds.variables:
            ds = ds.drop_vars(v)
 
    if times is None:
        times = ds.time.values
    idx = pd.Index(times)
    if idx.has_duplicates:
        times = idx[~idx.duplicated(keep="first")].values
        if verbose:
            print("Dropped duplicates; using %d unique times." % len(times))
 
    # The full month as requested. `times` is trimmed later by resume, so
    # completeness must be judged against this, not against whatever is
    # left to do in the current invocation.
    n_requested = len(times)
 
    grid_keys = ["labels_raw", "labels_env", "labels_faint", "labels_strong",
                 "pi_mask", "stable_mask", "ece_core", "ece_dist_km",
                 "corridor", "support", "faint_det", "strong_det",
                 "Z_bg", "Enh"]
    if have_wrfout:
        grid_keys += ["CTT", "convective_precip_masked", "rainnc",
                      "convective_precip"]
    if precip_rate_da is not None:
        grid_keys += ["prate_surface", "convective_prate"]
    grids_acc = dict((k, []) for k in grid_keys)
    obj_rows = []
 
    prev_rainnc = None
    month_files = []
    first_inc = None
    if have_wrfout:
        month_tag = pd.Timestamp(times[0]).strftime("%Y-%m")
        month_files = sorted(glob("%swrfout_d02_%s-*"
                                  % (all_files_path, month_tag)))
        if not month_files:
            raise RuntimeError("No wrfout files for %s in %s"
                               % (month_tag, all_files_path))
        t0 = pd.Timestamp(times[0])
        curr0_file = _find_wrfout_for_time(all_files_path, t0) or month_files[0]
        with NCDataset(curr0_file) as d0:
            curr0_rainnc = getvar(d0, "RAINNC", timeidx=0)
        is_sim_start = (t0.month == 10 and t0.day == 1 and t0.hour == 0)
        prev_file = None if is_sim_start else _find_wrfout_for_time(
            all_files_path, t0 - pd.Timedelta(hours=1))
        if prev_file:
            with NCDataset(prev_file) as dp:
                prev_rainnc = getvar(dp, "RAINNC", timeidx=0)
            first_inc = (curr0_rainnc - prev_rainnc).where(
                lambda x: x >= 0, other=np.nan).fillna(0.0)
            # Reference-code fix: the research runner left prev_rainnc at
            # the previous-month value here, so the second timestep's
            # increment spanned two hours (see PORTING_NOTES.md).
            prev_rainnc = curr0_rainnc
            if verbose:
                print("First-hour diff uses previous file:\n prev=%s\n curr=%s"
                      % (os.path.basename(prev_file),
                         os.path.basename(curr0_file)))
        else:
            prev_rainnc = curr0_rainnc.copy()
            first_inc = prev_rainnc * 0.0
            if verbose:
                print("INFO: no previous-hour wrfout (simulation start or "
                      "missing); first-hour increment set to 0.")
 
    # ------------------------------------------------------------------
    # Chunked writer. Each checkpoint writes ONE part file holding only
    # the timesteps since the last checkpoint, then clears the
    # accumulators. Memory therefore stays flat at checkpoint_every
    # timesteps instead of growing with the month, and a run that stops
    # leaves every completed part on disk, so it can resume.
    # ------------------------------------------------------------------
    parts_dir = out_nc_path + ".parts"
 
    def _encoding_for(gd):
        enc = {}
        for v in gd.data_vars:
            if "label" in v:
                dtype = "int16"
            elif v in ("corridor", "support", "ece_core", "pi_mask",
                       "stable_mask", "faint_det", "strong_det"):
                dtype = "int8"
            else:
                dtype = "float32"
            enc[v] = {"zlib": True, "complevel": compress_level,
                      "dtype": dtype,
                      "chunksizes": (1,) + gd[v].shape[1:]}
        enc["time"] = {"dtype": "int64"}
        return enc
 
    def _write_part(part_index):
        """Write the accumulated block as one part file, then free it."""
        if not any(grids_acc.values()):
            return
        gd = xr.Dataset(dict(
            (name, xr.concat(seq, dim="time", coords="minimal"))
            for name, seq in grids_acc.items() if seq))
        gd = gd.assign_coords(XLAT=ds.XLAT, XLONG=ds.XLONG)
        gd = gd.set_coords(["XLAT", "XLONG"])
        gd.attrs.update({
            "title": "CESC outputs (part)",
            "history": "created by cesc %s" % __import__(
                "cesc").__version__,
            "Conventions": "CF-1.8",
            "cesc_params": str(p.to_dict()),
            "cesc_part_index": int(part_index),
        })
        os.makedirs(parts_dir, exist_ok=True)
        tmp = os.path.join(parts_dir, "part_%05d.nc.tmp" % part_index)
        final = os.path.join(parts_dir, "part_%05d.nc" % part_index)
        safe_to_netcdf(gd, tmp, encoding=_encoding_for(gd))
        os.rename(tmp, final)          # atomic: a part file is never half-written
        if obj_rows:
            dfo = pd.concat(obj_rows, ignore_index=True)
            dfo.to_csv(os.path.join(parts_dir, "part_%05d.csv" % part_index),
                       index=False)
        # free the block
        for seq in grids_acc.values():
            del seq[:]
        del obj_rows[:]
        gd.close()
 
    def _finalize():
        """Stitch the parts into the final outputs and mark them complete."""
        part_files = sorted(glob(os.path.join(parts_dir, "part_*.nc")))
        if not part_files:
            return None, None
        try:
            gd = xr.open_mfdataset(part_files, combine="by_coords",
                                   data_vars="minimal", coords="minimal",
                                   compat="override")
            lazy = True
        except ImportError:
            # dask is absent; concatenate eagerly. Safe because the parts
            # are written in blocks and this is the only place the whole
            # record is assembled. If the month is too large for memory,
            # install dask or read the parts directly with open_output.
            gd = xr.concat([xr.open_dataset(f) for f in part_files],
                           dim="time", coords="minimal",
                           data_vars="minimal")
            lazy = False
        n_have = int(gd.sizes.get("time", 0))
        complete = n_have >= n_requested
        gd.attrs.update({
            "title": "CESC outputs",
            "cesc_params": str(p.to_dict()),
            "cesc_timesteps_written": n_have,
            "cesc_timesteps_requested": int(n_requested),
            "cesc_complete": "yes" if complete else "no",
        })
        if not complete:
            gd.attrs["cesc_warning"] = (
                "PARTIAL OUTPUT: %d of %d requested timesteps. The run did "
                "not finish. Re-run with resume=True to continue."
                % (n_have, n_requested))
        os.makedirs(os.path.dirname(out_nc_path), exist_ok=True)
        tmp = out_nc_path + ".tmp"
        if lazy:
            gd = gd.chunk({"time": max(1, checkpoint_every or 24)})
        safe_to_netcdf(gd, tmp, encoding=_encoding_for(gd))
        gd.close()
        os.replace(tmp, out_nc_path)
 
        csvs = sorted(glob(os.path.join(parts_dir, "part_*.csv")))
        dfo = None
        if csvs:
            dfo = pd.concat([pd.read_csv(c) for c in csvs],
                            ignore_index=True)
            os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
            tmp_csv = out_csv_path + ".tmp"
            dfo.to_csv(tmp_csv, index=False)
            os.replace(tmp_csv, out_csv_path)
 
        result = xr.open_dataset(out_nc_path)
 
        # ---- housekeeping: drop the parts, but only once the stitched
        # outputs are verified. The parts are the only copy of the work
        # until the final file exists and reads back correctly, so this
        # re-opens the file and checks it before deleting anything. On a
        # partial or unverifiable run the parts are kept, which is what
        # makes the next submission resumable.
        if not keep_parts:
            ok = (complete
                  and int(result.sizes.get("time", 0)) == n_requested
                  and result.attrs.get("cesc_complete") == "yes")
            if ok and dfo is not None:
                try:
                    n_csv = sum(1 for _ in open(out_csv_path)) - 1
                    ok = n_csv == len(dfo)
                    if not ok:
                        print("parts kept: object CSV has %d rows, expected "
                              "%d" % (n_csv, len(dfo)))
                except OSError as exc:
                    ok = False
                    print("parts kept: could not verify %s (%s)"
                          % (out_csv_path, exc))
            if ok:
                shutil.rmtree(parts_dir, ignore_errors=True)
                if verbose:
                    print("removed %s (run complete and verified)"
                          % parts_dir)
            elif verbose:
                print("kept %s: the run is not complete, so resubmitting "
                      "will resume from these." % parts_dir)
 
        return result, dfo
 
    # ---- resume: drop timesteps already written to part files ----
    part_index = 0
    if resume:
        existing = sorted(glob(os.path.join(parts_dir, "part_*.nc")))
        if existing:
            done_times = []
            for f in existing:
                with xr.open_dataset(f) as _d:
                    done_times.append(_d.time.values)
            done = np.concatenate(done_times)
            part_index = len(existing)
            keep = ~np.isin(times, done)
            n_skip = int((~keep).sum())
            if n_skip:
                if verbose:
                    print("resume: %d timesteps already written in %d part "
                          "file(s); continuing from %s"
                          % (n_skip, part_index,
                             str(times[keep][0])[:16] if keep.any()
                             else "(nothing left)"))
                times = times[keep]
            if len(times) == 0:
                if verbose:
                    print("resume: every requested timestep is already "
                          "written; finalizing.")
                return _finalize()
 
    curr_file_i = month_files[0] if month_files else None
    for i, t in enumerate(times):
        if verbose:
            print(t)
        da_t = ds.sel(time=t)
 
        theta_e_np = np.ascontiguousarray(
            da_t.theta_e.values.astype(np.float64))
        heights_np = np.ascontiguousarray(
            da_t.heights.values.astype(np.float64))
        dtheta_e_np = np.diff(theta_e_np, axis=0).astype(np.float64)
 
        (pi_str, pi_dep, st_str, st_dep,
         pi_h0, pi_h1, st_h0, st_h1) = env.scan_pi_stable(
            dtheta_e_np, heights_np, p.max_pi_height_m,
            p.unstable_threshold, p.stable_threshold,
            p.min_pi_depth_m, p.max_stable_height_m, p.theta_eps)
 
        prate_t = None
        if precip_rate_da is not None:
            try:
                prate_t = precip_rate_da.sel(time=t).values
            except KeyError:
                prate_t = None
 
        aux = {}
        if "HWP" in da_t:
            aux["hwp_gm2"] = da_t.HWP.values
 
        out = identify_time(
            Z_2d_da=da_t.reflectivity_1km,
            pi_strength=pi_str, stable_strength=st_str,
            pi_start_h=pi_h0, pi_end_h=pi_h1,
            u3d=da_t.u3d, v3d=da_t.v3d, heights=da_t.heights,
            precip_rate=prate_t, aux_fields=aux,
            params=p, make_figure=False)
 
        if verbose:
            print_summary(out["summary"], prefix="  ")
 
        cesc_mask = out["labels_env_da"] > 0
 
        if have_wrfout:
            if i == 0:
                inc = first_inc
                curr_file_i = _find_wrfout_for_time(
                    all_files_path, pd.Timestamp(t)) or month_files[0]
            else:
                curr_file_i = (month_files[i] if i < len(month_files)
                               else _find_wrfout_for_time(
                                   all_files_path, pd.Timestamp(t)))
                # positional lookup assumes one file per hour with no
                # gaps; verify the filename hour and fall back to a
                # time-based search when it does not match
                hour_tag = pd.Timestamp(t).strftime("%Y-%m-%d_%H")
                if curr_file_i is None or hour_tag not in curr_file_i:
                    curr_file_i = _find_wrfout_for_time(
                        all_files_path, pd.Timestamp(t))
                if curr_file_i is None:
                    raise IndexError("Missing wrfout for %s" % t)
                with NCDataset(curr_file_i) as dsc:
                    curr_rainnc = getvar(dsc, "RAINNC", timeidx=0)
                inc = (curr_rainnc - prev_rainnc).where(
                    lambda x: x >= 0, other=np.nan).fillna(0.0)
                prev_rainnc = curr_rainnc
 
            grids_acc["rainnc"].append(_da_now(inc, da_t))
            grids_acc["convective_precip"].append(
                _da_now(inc.where(cesc_mask, other=0.0), da_t))
            grids_acc["convective_precip_masked"].append(
                _da_now(inc.where(cesc_mask), da_t))
            with NCDataset(curr_file_i) as dsc:
                ctt_nc = getvar(dsc, "ctt", timeidx=0)
            grids_acc["CTT"].append(_da_now(ctt_nc, da_t))
 
        if prate_t is not None:
            pr_da = xr.DataArray(prate_t, coords=out["labels_env_da"].coords,
                                 dims=out["labels_env_da"].dims)
            grids_acc["prate_surface"].append(_da_now(pr_da, da_t))
            grids_acc["convective_prate"].append(
                _da_now(pr_da.where(cesc_mask, other=0.0), da_t))
 
        to_stack = {
            "labels_raw": out["labels_raw_da"],
            "labels_env": out["labels_env_da"],
            "labels_faint": out["labels_faint_da"],
            "labels_strong": out["labels_strong_da"],
            "pi_mask": out["masks"]["pi"],
            "stable_mask": out["masks"]["stable"],
            "ece_core": out["masks"]["both"],
            "ece_dist_km": out["masks"]["ece_dist_km"],
            "corridor": out["masks"]["corridor"],
            "support": out["masks"]["support"],
            "faint_det": out["masks"]["faint_det"],
            "strong_det": out["masks"]["strong_det"],
            "Z_bg": out["masks"]["Z_bg"],
            "Enh": out["masks"]["Enh"],
        }
        for k, da in to_stack.items():
            grids_acc[k].append(_da_now(da, da_t))
 
        objs = out.get("objects", [])
        if objs:
            df = pd.DataFrame(objs)
            df["time"] = pd.Timestamp(t)
            df["object_id_within_time"] = df["id"].astype(int)
            for col in df.columns:
                df[col] = df[col].apply(
                    lambda v: v.item() if isinstance(v, np.generic) else v)
            df["object_uid"] = (pd.Timestamp(t).strftime("%Y%m%d%H") + "_"
                                + df["object_id_within_time"].astype(str)
                                .str.zfill(4))
            obj_rows.append(df)
 
        if checkpoint_every and (i + 1) % checkpoint_every == 0:
            if verbose:
                print("checkpoint: writing part %d (%d/%d timesteps this "
                      "run)" % (part_index, i + 1, len(times)))
            _write_part(part_index)
            part_index += 1
 
    _write_part(part_index)          # the remaining partial block
    grids_ds, objects_df = _finalize()
    if verbose:
        n = grids_ds.sizes.get("time", 0) if grids_ds is not None else 0
        print("Wrote gridded outputs -> %s" % out_nc_path)
        print("%s: %d timesteps written."
              % ("COMPLETE" if grids_ds is not None
                 and grids_ds.attrs.get("cesc_complete") == "yes"
                 else "PARTIAL", n))
        if objects_df is not None:
            kept = objects_df[objects_df.kept_env]
            print("Object table: %d rows, %d kept (%d strong, %d faint) "
                  "-> %s" % (len(objects_df), len(kept),
                             int((kept.intensity_cls == "strong").sum()),
                             int((kept.intensity_cls == "faint").sum()),
                             out_csv_path))
    return grids_ds, objects_df
 
 
def run(ds, params=None, **kwargs):
    """Alias for run_wrf_month for the documented cesc.run entry point."""
    return run_wrf_month(ds, params=params, **kwargs)
 
 
def open_output(path, require_complete=True):
    """Open a cesc grids file, refusing a partial one by default.
 
    A run that stops between checkpoints leaves a valid, readable NetCDF
    holding only the timesteps written so far. Opening it with plain
    ``xr.open_dataset`` gives no hint that it is short, which is how a
    truncated month reaches an analysis notebook unnoticed. This checks
    the completeness attributes written by ``run_wrf_month``.
 
    Parameters
    ----------
    path : str
    require_complete : bool
        Raise if the file is a partial checkpoint. Set False to inspect
        one deliberately (a warning is issued instead).
 
    Returns
    -------
    xr.Dataset
    """
    import warnings
    ds = xr.open_dataset(path)
    done = ds.attrs.get("cesc_timesteps_written")
    want = ds.attrs.get("cesc_timesteps_requested")
    flag = ds.attrs.get("cesc_complete")
 
    if flag is None:
        warnings.warn(
            "%s carries no cesc completeness attributes; it predates this "
            "check. Verify its time axis covers the period you expect "
            "(a full month of hourly output is 24 x days)." % path)
        return ds
 
    if flag != "yes":
        msg = ("%s is a PARTIAL checkpoint: %s of %s timesteps "
               "(%s to %s). The run did not finish. Re-run it, or pass "
               "require_complete=False to inspect it anyway."
               % (path, done, want,
                  str(ds.time.values[0])[:16], str(ds.time.values[-1])[:16]))
        if require_complete:
            raise RuntimeError(msg)
        warnings.warn(msg)
    return ds