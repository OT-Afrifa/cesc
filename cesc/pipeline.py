# cesc/pipeline.py
"""
Main CESC ID pipeline.

This module implements the full three-stage algorithm for identifying
Convection Embedded within or Emergent from Stratiform Clouds (CESC) in
cold-season orographic environments, as described in Afrifa et al. (Part 1,
submitted 2026).

Stage 1 -- Environmental classification (ECE detection)
    Each atmospheric column is tested for the two thermodynamic conditions
    required for elevated convection: a surface stable layer (cumulative
    theta_e rise >= stable_threshold below 3000 m AGL) overlain by a
    potentially unstable layer (cumulative theta_e drop >= unstable_threshold
    over at least 500 m depth, confined below 5000 m AGL).  Columns meeting
    both conditions form the ECE core.  If use_advection is True, a
    downstream corridor is appended using the PI-layer mean wind and the
    assumed convective lifespan.

Stage 2 -- Reflectivity-based object detection
    Locally enhanced reflectivity objects are identified over the full
    precipitating domain using the multi-scale ID-PRO approach from
    cesc.id_pro.detect_objects_id_pro.  Faint and strong detection masks
    are tracked separately.

Stage 3 -- Environmental gating and intensity classification
    Detected objects are retained as CESC only when a sufficient fraction of
    their area (min_support_frac, min_support_px) overlaps the support mask
    (ECE core plus corridor).  Each retained object is further classified as
    faint or strong based on the fraction of its pixels that exceeded the
    strong detection threshold (frac_strong_cutoff).

References
----------
Afrifa, F. O., and co-authors, 2026: Climatology of cold-season emergent
    convection in frontal systems and its impact on orographic precipitation.
    Part 1: Detection algorithm.  Mon. Wea. Rev., in review.

Afrifa, F. O., B. Geerts, L. Xue, S. Chen, C. Hohman, C. Grasmick, and
    T. Zaremba, 2025: A case study of cold-season emergent orographic
    convection and its impact on precipitation. Part I: Mesoscale analysis.
    Mon. Wea. Rev., 153, 2229-2250.

Yeh, P., and B. A. Colle, 2025: A Comparison of Approaches to Objectively
    Identify Precipitation Structures within the Comma Head of Mid-Latitude
    Cyclones.  J. Atmos. Oceanic Technol., 42, 463-477.
"""

import numpy as np
import xarray as xr
from scipy.ndimage import (binary_dilation, find_objects,
                           uniform_filter, distance_transform_edt)
from skimage import morphology, segmentation, filters, feature
from numba import njit, prange

from cesc import id_pro as id
from cesc.distance_calculator import estimate_dx_dy_km

try:
    import cartopy.crs as crs
    import cartopy.feature as cf
    import matplotlib.pyplot as plt
    from cartopy.mpl.gridliner import LATITUDE_FORMATTER, LONGITUDE_FORMATTER
    from wrf import smooth2d
    _PLOT_AVAILABLE = True
except ImportError:
    _PLOT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Environmental scan helpers
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _surface_stable_scan(dt, heights_full, pi_start_k, max_stable_height_m, eps):
    """
    Scan upward from the surface accumulating positive theta_e increments.

    Starts at k=0 (ground level) and stops at the earlier of pi_start_k or
    max_stable_height_m (3000 m AGL per Afrifa et al. Part 1 Section 2b).
    Small negative wiggles (|v| < eps) are tolerated without resetting the
    accumulator, which handles well-mixed layers capped by higher-theta_e air.

    Parameters
    ----------
    dt                  : 1D float64 -- layer delta-theta_e (K), length nz
    heights_full        : 1D float64 -- full-level heights AGL (m), length nz+1
    pi_start_k          : int -- first unstable level (upper bound of scan)
    max_stable_height_m : float -- height ceiling for the scan (m AGL)
    eps                 : float -- noise tolerance (K)

    Returns
    -------
    cumsum : float -- cumulative positive theta_e rise (K)
    ee     : int   -- exclusive end index of the stable layer
    """
    cumsum = 0.0
    ee = 0
    for k in range(pi_start_k):
        if heights_full[k] > max_stable_height_m:
            break
        v = dt[k]
        if not np.isfinite(v):
            break
        if v > eps:
            cumsum += v
            ee = k + 1
    return cumsum, ee


@njit(parallel=True, fastmath=True, cache=True)
def scan_pi_stable(
    dtheta_e,
    heights_full,
    max_pi_height,
    unstable_threshold,
    stable_threshold,
):
    """
    Identify ECE columns: surface stable layer below a potentially unstable layer.

    PI layer conditions (Afrifa et al. Part 1, Section 2b):
      - Consecutive levels with delta-theta_e < -eps
      - Cumulative theta_e drop >= unstable_threshold (1 K, positive convention)
      - Layer depth >= 500 m
      - Confined below max_pi_height (5000 m AGL)

    Stable layer conditions:
      - Surface-anchored: scan starts at k=0
      - Capped at min(PI base, 3000 m AGL)
      - Cumulative positive theta_e rise >= stable_threshold (2 K)

    Parameters
    ----------
    dtheta_e           : (nz, ny, nx) float64 -- layer delta-theta_e (K)
    heights_full       : (nz+1, ny, nx) float64 -- full-level heights AGL (m)
    max_pi_height      : float -- PI search ceiling (m AGL)
    unstable_threshold : float -- minimum PI theta_e drop (K, positive)
    stable_threshold   : float -- minimum stable layer rise (K)

    Returns
    -------
    Eight (ny, nx) float64 arrays, all NaN where the criterion is not satisfied:
      pi_str, pi_dep, st_str, st_dep,
      pi_h0, pi_h1, st_h0, st_h1
    """
    nz, ny, nx = dtheta_e.shape
    eps = 5e-3
    MAX_STABLE_HEIGHT = 3000.0   # m AGL per paper

    pi_str = np.full((ny, nx), np.nan)
    pi_dep = np.full((ny, nx), np.nan)
    st_str = np.full((ny, nx), np.nan)
    st_dep = np.full((ny, nx), np.nan)
    pi_h0 = np.full((ny, nx), np.nan)
    pi_h1 = np.full((ny, nx), np.nan)
    st_h0 = np.full((ny, nx), np.nan)
    st_h1 = np.full((ny, nx), np.nan)

    for j in prange(ny):
        for i in range(nx):
            hf = heights_full[:, j, i]
            dt = dtheta_e[:,   j, i]

            max_lev = -1
            for k in range(nz):
                if hf[k] <= max_pi_height:
                    max_lev = k
                else:
                    break
            if max_lev < 0:
                continue

            run_start = -1
            found_pi = False

            for k in range(max_lev + 1):
                v = dt[k]
                unstable = np.isfinite(v) and (v < -eps)
                if unstable:
                    if run_start == -1:
                        run_start = k
                else:
                    if run_start != -1:
                        s = run_start; e = k
                        drop = 0.0
                        valid = False
                        for kk in range(s, e):
                            vv = dt[kk]
                            if np.isfinite(vv):
                                drop += vv
                                valid = True
                        depth = hf[e] - hf[s]   # CHANGE 9
                        if valid and drop <= -unstable_threshold and depth >= 500.0:
                            pi_str[j, i] = drop
                            pi_h0[j, i] = hf[s]
                            pi_h1[j, i] = hf[e]
                            pi_dep[j, i] = depth
                            cumsum, ee = _surface_stable_scan(   # CHANGE 10
                                dt, hf, s, MAX_STABLE_HEIGHT, eps)
                            if cumsum >= stable_threshold and ee > 0:
                                st_str[j, i] = cumsum
                                st_h0[j, i] = hf[0]
                                st_h1[j, i] = hf[ee]
                                st_dep[j, i] = hf[ee]-hf[0]
                            found_pi = True
                        run_start = -1
                if found_pi:
                    break

            # trailing PI run reaching max_lev
            if (not found_pi) and (run_start != -1):
                s = run_start
                e = max_lev + 1
                drop = 0.0
                valid = False
                for kk in range(s, e):
                    vv = dt[kk]
                    if np.isfinite(vv):
                        drop += vv
                        valid = True
                depth = hf[e] - hf[s]   # CHANGE 9
                if valid and drop <= -unstable_threshold and depth >= 500.0:
                    pi_str[j, i] = drop
                    pi_h0[j, i] = hf[s]
                    pi_h1[j, i] = hf[e]
                    pi_dep[j, i] = depth
                    cumsum, ee = _surface_stable_scan(   # CHANGE 10
                        dt, hf, s, MAX_STABLE_HEIGHT, eps)
                    if cumsum >= stable_threshold and ee > 0:
                        st_str[j, i] = cumsum
                        st_h0[j, i] = hf[0]
                        st_h1[j, i] = hf[ee]
                        st_dep[j, i] = hf[ee]-hf[0]

    return pi_str, pi_dep, st_str, st_dep, pi_h0, pi_h1, st_h0, st_h1


# ---------------------------------------------------------------------------
# Wind helpers
# ---------------------------------------------------------------------------

@njit(parallel=True, fastmath=True, cache=True)
def layer_mean_wind(u_layer, v_layer, z_full, h_start, h_end,
                    default_lo, default_hi):
    """
    Mass-weighted mean wind within each column's PI layer.

    Parameters
    ----------
    u_layer, v_layer : (nzm, ny, nx) float32 -- mid-level winds (m/s)
    z_full           : (nzm+1, ny, nx) float32 -- full-level heights AGL (m)
    h_start, h_end   : (ny, nx) float32 -- PI layer base and top (m AGL)
    default_lo/hi    : float -- fallback layer bounds when PI info is missing

    Returns
    -------
    ubar, vbar : (ny, nx) float32  (NaN where integration layer is absent)
    """
    nzm, ny, nx = u_layer.shape
    ubar = np.empty((ny, nx), dtype=np.float32)
    vbar = np.empty((ny, nx), dtype=np.float32)
    for j in prange(ny):
        for i in range(nx):
            hs = h_start[j, i]
            he = h_end[j, i]
            if not (hs == hs) or not (he == he) or he <= hs:
                hs = default_lo
                he = default_hi
            numu = 0.0
            numv = 0.0
            den = 0.0
            for k in range(nzm):
                zbot = z_full[k,  j, i]
                ztop = z_full[k+1, j, i]
                lo = max(zbot, hs)
                hi = min(ztop, he)
                th = hi - lo
                if th > 0.0:
                    numu += u_layer[k, j, i]*th
                    numv += v_layer[k, j, i]*th
                    den += th
            if den > 0.0:
                ubar[j, i] = numu/den
                vbar[j, i] = numv/den
            else:
                ubar[j, i] = np.nan
                vbar[j, i] = np.nan
    return ubar, vbar


# ---------------------------------------------------------------------------
# Corridor helpers
# ---------------------------------------------------------------------------
def _disk(radius_px):
    """Circular boolean structuring element of given pixel radius."""
    r = max(0, int(radius_px))
    yy, xx = np.ogrid[-r:r+1, -r:r+1]
    return (yy*yy + xx*xx) <= r*r


def _draw_line_boolean(y0, x0, y1, x1, out):
    """
    Bresenham line rasterisation onto a boolean array.
    Clips silently to array bounds.
    """
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
    dy = abs(y1-y0)
    dx = abs(x1-x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < out.shape[0] and 0 <= x0 < out.shape[1]:
            out[y0, x0] = True
        if y0 == y1 and x0 == x1:
            break
        e2 = 2*err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy


def build_corridor(ece_mask, u_bar, v_bar, grid_km,
                   lookahead_km, halfwidth, seed_stride_px, min_speed):
    """
    Build the downstream advection corridor from ECE seed pixels.

    A line is drawn from each ECE pixel in the direction of the local
    PI-layer mean wind for lookahead_km[j,i] km.  The resulting skeleton is
    dilated laterally by halfwidth km.

    Parameters
    ----------
    ece_mask       : (ny, nx) bool
    u_bar, v_bar   : (ny, nx) float -- PI-layer mean wind (m/s)
    grid_km        : float -- isotropic grid spacing (km)
    lookahead_km   : (ny, nx) float -- per-point downstream reach (km)
    halfwidth      : float -- lateral half-width (km)
    seed_stride_px : int -- seed spacing (pixels)
    min_speed      : float -- minimum wind speed to draw a corridor (m/s)

    Returns
    -------
    (ny, nx) bool
    """
    ny, nx = ece_mask.shape
    out = np.zeros((ny, nx), dtype=bool)
    Lpx = np.round(lookahead_km / grid_km).astype(int)
    halfw_px = int(round(halfwidth / grid_km))
    dil_foot = _disk(halfw_px)
    U = np.hypot(u_bar, v_bar)

    for y in range(0, ny, seed_stride_px):
        xs = np.where(ece_mask[y, :])[0]
        if xs.size == 0:
            continue
        for x in xs[::seed_stride_px]:
            spd = U[y, x]
            if not np.isfinite(spd) or spd < min_speed:
                continue
            vx = u_bar[y, x]
            vy = v_bar[y, x]
            if not np.isfinite(vx) or not np.isfinite(vy):
                continue
            nxu = vx / (spd + 1e-9)
            nyv = -vy / (spd + 1e-9)   # rows increase southward
            _draw_line_boolean(y, x,
                               y + nyv*Lpx[y, x],
                               x + nxu*Lpx[y, x], out)
    if halfw_px > 0:
        out = binary_dilation(out, structure=dil_foot)
    return out


# ---------------------------------------------------------------------------
# Wind-aligned geometry helper
# ---------------------------------------------------------------------------

def _wind_L_W(mask_bool, cy, cx, u_bar, v_bar, dx_km, dy_km, slc):
    yy,xx=np.nonzero(mask_bool)
    if yy.size==0: return np.nan,np.nan
    try:
        if u_bar is None or v_bar is None: raise ValueError
        cf=cy+slc[0].start; cg=cx+slc[1].start
        u0=float(u_bar[cf,cg]); v0=float(v_bar[cf,cg])
        if not np.isfinite(u0) or not np.isfinite(v0) or (u0==0 and v0==0):
            raise ValueError
        sx,sy=u0,-v0; nrm=(sx*sx+sy*sy)**0.5; sx/=nrm; sy/=nrm
        cxw,cyw=-sy,sx
        X=(xx-cx)*dx_km; Y=(yy-cy)*dy_km
        return float((X*sx+Y*sy).ptp()),float((X*cxw+Y*cyw).ptp())
    except Exception:
        x=(xx-cx)*dx_km; y=(yy-cy)*dy_km
        x0=x-x.mean(); y0=y-y.mean()
        sxx=np.mean(x0*x0); syy=np.mean(y0*y0); sxy=np.mean(x0*y0)
        tr=sxx+syy; det=sxx*syy-sxy*sxy
        l1=tr/2+max(tr*tr/4-det,0)**0.5
        l2=tr/2-max(tr*tr/4-det,0)**0.5
        return float(2*max(l1,0)**0.5),float(2*max(l2,0)**0.5)


# ---------------------------------------------------------------------------
# Main pipeline function
# ---------------------------------------------------------------------------

def run_cesc_pipeline(
    Z_2d_da, ter_da,
    HWP_da=None,
    pi_strength_da=None, stable_strength_da=None,
    pi_start_h_da=None,  pi_end_h_da=None,
    u3d=None, v3d=None,  heights=None,
    unstable_threshold=1.0,
    stable_threshold=2.0,
    ece_strict=True,
    bg_radius=20.0,
    faint_min_diff=2.0,
    faint_k_sigma=0.30,
    strong_min_diff=5.0,
    strong_k_sigma=0.30,
    box_sizes=(20.0, 40.0, 80.0),
    min_area=6.0,
    min_support_frac=0.30,
    min_support_px=5,
    use_advection=True,
    advection_time=20,
    halfwidth=10.0,
    seed_stride_px=5,
    min_speed=2.0,
    frac_strong_cutoff=0.5,
    env_mode="ECE",
    extent=[-118, -109.4, 42, 46.2],
    title=None,
    level_title=None,
    make_plot=True,
):
    """
    Identify CESC objects for a single model time step.

    Parameters
    ----------
    Z_2d_da             : 2D xr.DataArray -- reflectivity at 1 km AGL (dBZ);
                          must have XLAT and XLONG as coordinates
    ter_da              : 2D xr.DataArray -- terrain height (m); for plotting
    HWP_da              : 2D xr.DataArray -- total hydrometeor water path
                          (g/m^2); optional, used for per-object statistics
    pi_strength_da      : 2D array -- PI layer cumulative theta_e drop (K,
                          negative); required
    stable_strength_da  : 2D array -- stable layer cumulative theta_e rise (K)
    pi_start_h_da       : 2D array -- PI layer base height (m AGL)
    pi_end_h_da         : 2D array -- PI layer top height (m AGL)
    u3d, v3d            : 3D xr.DataArrays -- zonal and meridional wind (m/s);
                          required when use_advection=True
    heights             : 3D xr.DataArray -- full-level heights AGL (m);
                          required when use_advection=True
    unstable_threshold  : minimum PI strength (K, positive convention)
    stable_threshold    : minimum stable layer theta_e rise (K)
    ece_strict          : if True, use only the ECE core as the support region;
                          if False, add the downstream corridor as well
    bg_radius           : smoothing radius (km) for the display-only background
                          panel and the watershed seeding field.  This does NOT
                          affect the multi-scale detection thresholds.
    faint_min_diff      : dBZ floor for faint detection
    faint_k_sigma       : sigma multiplier for faint threshold
    strong_min_diff     : dBZ floor for strong detection
    strong_k_sigma      : sigma multiplier for strong threshold
    box_sizes           : box widths (km) for multi-scale detection;
                          structural design choice, not a tuning threshold
    min_area            : minimum object area to retain (km^2)
    min_support_frac    : minimum fraction of object area inside support mask
    min_support_px      : minimum pixel count inside support mask
    use_advection       : compute downstream corridor from PI-layer winds
    advection_time      : assumed convective lifespan (minutes)
    halfwidth           : lateral corridor half-width (km)
    seed_stride_px      : seed spacing for corridor drawing (pixels)
    min_speed           : minimum wind speed to draw a corridor (m/s)
    frac_strong_cutoff  : objects with frac_strong >= this value are classified
                          as strong; below it as faint.  Tune this on a well-
                          observed case (e.g. 7 Feb 2017 IOP12 from SNOWIE).
    env_mode            : 'ECE' requires both stable + PI; 'PI_only' skips the
                          stable layer check
    extent              : [lon_min, lon_max, lat_min, lat_max] for map plots
    title, level_title  : optional strings for the quicklook figure
    make_plot           : produce the quicklook figure (set False for batch runs)

    Returns
    -------
    dict with keys:
      strat_da          : precipitating-domain mask
      Enh_da            : reflectivity enhancement (Z minus display background)
      Zbg_da            : display-only smoothed background reflectivity
      labels_raw_da     : watershed labels before environmental gating
      labels_env_da     : all kept CESC objects (compact labels 1..N)
      labels_faint_da   : subset of labels_env where frac_strong < cutoff
      labels_strong_da  : subset of labels_env where frac_strong >= cutoff
      masks             : dict of 2D DataArrays (pi, stable, ece_core,
                          corridor, support, faint_det, strong_det, ...)
      objects           : list of per-object dicts
      u_bar_np          : 2D PI-layer mean zonal wind (m/s)
      v_bar_np          : 2D PI-layer mean meridional wind (m/s)
      fig               : matplotlib figure or None if make_plot=False
    """
    if pi_strength_da is None:
        raise ValueError("pi_strength_da must be provided.")

    dx_km, dy_km = estimate_dx_dy_km(
        np.asarray(Z_2d_da.XLAT), np.asarray(Z_2d_da.XLONG))
    grid_km = 0.5*(dx_km+dy_km)

    # Display + watershed background (NOT used for detection thresholds)
    def _smooth(a, size):
        a = np.asarray(a, dtype=np.float32)
        v = np.isfinite(a).astype(np.float32)
        a0 = np.nan_to_num(a, nan=0.0).astype(np.float32)
        num = uniform_filter(a0, size=size, mode="nearest")
        den = uniform_filter(v, size=size, mode="nearest")
        out = num/np.maximum(den, 1e-9)
        out[den < 1e-6] = np.nan
        return out

    win_px = int(np.round((2 * bg_radius) / grid_km)) | 1   # must be odd
    Z = Z_2d_da.values
    Zbg = _smooth(Z, win_px)
    Enh = Z - Zbg  # enhancement relative to the display-scale background

    strat = np.isfinite(Z) & (Z >= -2.0)

    def _da(arr, name):
        return xr.DataArray(arr, coords=Z_2d_da.coords, dims=Z_2d_da.dims, name=name)
    strat_da = _da(strat, 'stratiform_base')
    Zbg_da = _da(Zbg, 'Z_bg')
    Enh_da = _da(Enh, 'Z_enh')

    # Stage 1: ECE environmental masks
    pi_arr = np.asarray(pi_strength_da)
    st_arr = np.asarray(stable_strength_da) if stable_strength_da is not None else None
    pi_mask = np.isfinite(pi_arr) & (pi_arr <= -abs(unstable_threshold))

    if env_mode == "ECE" and st_arr is not None:
        stable_mask = np.isfinite(st_arr) & (st_arr >= abs(stable_threshold))
        both_mask = pi_mask & stable_mask
    else:
        stable_mask = (np.isfinite(st_arr) & (st_arr >= abs(stable_threshold))
                       if st_arr is not None else np.zeros_like(pi_mask, bool))
        both_mask = pi_mask

    ece_core = binary_dilation(both_mask, structure=morphology.disk(1))

    # ---- downstream corridor ----
    corridor = np.zeros_like(ece_core, dtype=bool)
    u_bar_ece = np.zeros_like(ece_core, dtype=np.float32)
    v_bar_ece = np.zeros_like(ece_core, dtype=np.float32)

    if use_advection:
        if u3d is None or v3d is None or heights is None:
            raise ValueError("u3d, v3d, heights required when use_advection=True.")
        if ece_core.any():
            DEFAULT_LO, DEFAULT_HI = 2000.0, 5000.0
            u_layer = u3d.isel(bottom_top=slice(0, -1)).values.astype(np.float32)
            v_layer = v3d.isel(bottom_top=slice(0, -1)).values.astype(np.float32)
            z_full = heights.values.astype(np.float32)
            h_start = (np.asarray(pi_start_h_da) if pi_start_h_da is not None
                       else np.full(Z.shape, DEFAULT_LO, dtype=np.float32))
            h_end = (np.asarray(pi_end_h_da) if pi_end_h_da is not None
                     else np.full(Z.shape, DEFAULT_HI, dtype=np.float32))
            u_bar_np, v_bar_np = layer_mean_wind(
                u_layer, v_layer, z_full, h_start, h_end, DEFAULT_LO, DEFAULT_HI)

            u_bar_ece = np.ma.masked_where(~ece_core, u_bar_np)
            v_bar_ece = np.ma.masked_where(~ece_core, v_bar_np)
            spd = np.sqrt(u_bar_ece**2 + v_bar_ece**2)
            lookahead_km = spd * 3.6 * (advection_time / 60.0)
            corridor = build_corridor(
                ece_core, u_bar_ece, v_bar_ece, grid_km,
                lookahead_km=lookahead_km, halfwidth=halfwidth,
                seed_stride_px=seed_stride_px, min_speed=min_speed)

    support = ece_core | corridor

    # Stage 2: detection over full precipitating domain
    faint_det, strong_det, _, _ = id.detect_objects_id_pro(
        field_da=Z_2d_da, gate_mask=strat,
        lats2d=np.asarray(Z_2d_da.XLAT),
        lons2d=np.asarray(Z_2d_da.XLONG),
        box_sizes=box_sizes,
        faint_min_diff=faint_min_diff, faint_k_sigma=faint_k_sigma,
        strong_min_diff=strong_min_diff, strong_k_sigma=strong_k_sigma,
        min_area=min_area)

    sm = filters.gaussian(
        np.where(np.isfinite(Enh), Enh, np.nanmin(Enh)-10.0),
        sigma=1.0, preserve_range=True)
    valid_seeds = faint_det & np.isfinite(sm)
    pct85 = np.nanpercentile(sm[valid_seeds], 85) if valid_seeds.any() else 0.0
    seed_mask = (sm > pct85) & faint_det
    min_dist_px = max(8, int(round(min_area / grid_km)))
    coords = feature.peak_local_max(sm, labels=seed_mask.astype(np.uint8),
                                    min_distance=min_dist_px, exclude_border=False)
    markers = np.zeros_like(faint_det, dtype=np.int32)
    for idx, (yy, xx) in enumerate(coords, start=1):
        markers[yy, xx] = idx
    if markers.max() == 0 and faint_det.any():
        markers = morphology.label(faint_det, connectivity=1)
    labels_ws = segmentation.watershed(-sm, markers=markers, mask=faint_det)
    labels_raw_da = _da(labels_ws, 'convective_labels_raw')

    # Stage 3: environmental gating + faint/strong classification
    ece_dist = distance_transform_edt(~ece_core)*grid_km
    pix_km2 = grid_km*grid_km
    slices = find_objects(labels_ws)
    objs, kept_ids = [], []

    for lab, slc in enumerate(slices, start=1):
        if slc is None:
            continue
        obj = (labels_ws[slc] == lab)
        area_px = int(obj.sum())
        if area_px == 0:
            continue

        jj, ii = np.nonzero(obj)
        jj += slc[0].start
        ii += slc[1].start
        ov_e = int((ece_core[slc] & obj).sum())
        ov_c = int((corridor[slc] & obj).sum())
        ov_s = int((support[slc] & obj).sum())
        fs = ov_s/max(area_px, 1)
        keep = (ov_s >= min_support_px) and (fs >= min_support_frac)
        if keep:
            kept_ids.append(lab)
        cy = int(round(jj.mean()))
        cx = int(round(ii.mean()))
        L = W = asp = ang = np.nan
        try:
            _, (L, W, asp, ang) = id._compute_moments_and_axes(jj, ii, dx_km, dy_km)
        except Exception:
            pass
        cls = id._classify(L, W, asp)
        ns = int(strong_det[jj, ii].sum())
        nf = int(faint_det[jj, ii].sum())
        frs = ns/max(area_px, 1)
        objs.append(dict(
            id=lab, kept_env=bool(keep), cls=cls,
            centroid_lat=float(np.nanmean(Z_2d_da.XLAT.values[jj, ii])),
            centroid_lon=float(np.nanmean(Z_2d_da.XLONG.values[jj, ii])),
            centroid_yx=(cy, cx),
            dist_to_ece_core_km=float(ece_dist[cy, cx]),
            major_km=L, minor_km=W, aspect=asp, orientation_deg=ang,
            area_px=area_px, size_km2=area_px*pix_km2,
            frac_within_support=fs, overlap_support_px=ov_s,
            frac_within_ece=ov_e/max(area_px, 1),
            frac_downstream=ov_c/max(area_px, 1),
            n_faint_px=nf, n_strong_px=ns, frac_strong=frs,
            intensity_cls=("strong" if frs >= frac_strong_cutoff else "faint"),
            hwp_mean_gm2=(float(np.nanmean(HWP_da.values[jj, ii]))
                          if HWP_da is not None else np.nan),
        ))

    if kept_ids:
        remap = np.zeros(labels_ws.max()+1, dtype=np.int32)
        for nid, old in enumerate(kept_ids, start=1):
            remap[old] = nid
        labels_env = remap[labels_ws]
    else:
        labels_env = np.zeros_like(labels_ws)

    labels_env_da = _da(labels_env, 'convective_labels_env_gated')

    strong_ids = {o['id'] for o in objs
                  if o['kept_env'] and o['frac_strong'] >= frac_strong_cutoff}
    old2new = {old: nid for nid, old in enumerate(kept_ids, start=1)} if kept_ids else {}
    labels_strong = np.zeros_like(labels_env)
    labels_faint = np.zeros_like(labels_env)
    for o in objs:
        if not o['kept_env']:
            continue
        nid = old2new.get(o['id'], 0)
        if nid == 0:
            continue
        if o['id'] in strong_ids:
            labels_strong[labels_env == nid] = nid
        else:
            labels_faint[labels_env == nid] = nid

    labels_faint_da = _da(labels_faint, 'convective_labels_faint')
    labels_strong_da = _da(labels_strong, 'convective_labels_strong')

    # Add downstream geometry metrics
    masks = {
        'pi'         :_da(pi_mask,    'pi_mask'),
        'stable'     :_da(stable_mask,'stable_mask'),
        'both'       :_da(both_mask,  'both_mask'),
        'ece_core'   :_da(ece_core,   'ece_core'),
        'ece_dist_km':_da(ece_dist,   'ece_dist_km'),
        'corridor'   :_da(corridor,   'corridor'),
        'support'    :_da(support,    'support'),
        'faint_det'  :_da(faint_det,  'faint_det'),
        'strong_det' :_da(strong_det, 'strong_det'),
        'Z_bg'       :Zbg_da,
        'Enh'        :Enh_da,
    }

    result = dict(
        strat_da=strat_da, Enh_da=Enh_da, Zbg_da=Zbg_da,
        labels_raw_da=labels_raw_da,
        labels_env_da=labels_env_da,
        labels_faint_da=labels_faint_da,
        labels_strong_da=labels_strong_da,
        masks=masks, objects=objs,
        u_bar_np=u_bar_ece, v_bar_np=v_bar_ece,
        fig=None,
    )
    result = _enrich_objects(result)

    if make_plot and _PLOT_AVAILABLE:
        result['fig'] = _quicklook(result, Z_2d_da, ter_da, Zbg_da, Enh_da,
                                   support, labels_env_da, labels_faint_da,
                                   labels_strong_da, extent, title,
                                   bg_radius, level_title)
    return result


# ---------------------------------------------------------------------------
# Object enrichment
# ---------------------------------------------------------------------------

def _enrich_objects(result):
    labels = np.asarray(result['labels_env_da'].values, dtype=np.int32)
    ece = np.asarray(result['masks']['ece_core'].values, dtype=bool)
    corridor = np.asarray(result['masks']['corridor'].values, dtype=bool)
    ds = corridor & (~ece)
    sup = corridor | ece
    lats = np.asarray(result['labels_env_da'].XLAT)
    lons = np.asarray(result['labels_env_da'].XLONG)
    dx_km, dy_km = estimate_dx_dy_km(lats, lons)
    u0 = np.asarray(result['u_bar_np']) if result['u_bar_np'] is not None else None
    v0 = np.asarray(result['v_bar_np']) if result['v_bar_np'] is not None else None
    maxlab = int(labels.max())
    if maxlab == 0:
        return result
    slices = []
    for lab in range(1, maxlab+1):
        ys, xs = np.where(labels == lab)
        slices.append(None if ys.size == 0 else
                      (slice(ys.min(), ys.max()+1), slice(xs.min(), xs.max()+1)))
    for o in result['objects']:
        lab = int(o['id'])
        if lab < 1 or lab > maxlab:
            continue
        slc = slices[lab-1]
        if slc is None:
            continue
        obj = (labels[slc] == lab)
        apx = int(obj.sum())
        if apx == 0:
            continue
        ae = int((ece[slc] & obj).sum())
        ad = int((ds[slc] & obj).sum())
        as_ = int((sup[slc] & obj).sum())
        o.update(area_ece_px=ae, area_ds_px=ad, area_sup_px=as_,
                 frac_within_ece=ae/apx, frac_downstream=ad/apx, frac_support=as_/apx)
        sm = sup[slc] & obj
        if 'centroid_yx' in o and o['centroid_yx'] is not None:
            cy_f, cx_f = int(o['centroid_yx'][0]), int(o['centroid_yx'][1])
            cy_l = cy_f-slc[0].start
            cx_l = cx_f-slc[1].start
        else:
            yy, xx = np.nonzero(obj)
            cy_l = int(round(yy.mean()))
            cx_l = int(round(xx.mean()))
        Lk = Wk = np.nan
        if sm.any():
            Lk, Wk = _wind_L_W(sm, cy_l, cx_l, u0, v0, dx_km, dy_km, slc)
        o['L_ds_km'] = float(Lk)
        o['W_ds_km'] = float(Wk)
    return result


# ---------------------------------------------------------------------------
# Quicklook figure
# ---------------------------------------------------------------------------

def _quicklook(result, Z_2d_da, ter_da, Zbg_da, Enh_da, support,
               labels_env_da, labels_faint_da, labels_strong_da,
               extent, title, bg_radius, level_title):
    if not _PLOT_AVAILABLE:
        return None
    BORDERS = cf.BORDERS.with_scale('10m')
    STATES = cf.STATES.with_scale('10m')
    proj = crs.LambertConformal(central_longitude=-117.55, central_latitude=44)
    fig, axes = plt.subplots(2, 2, figsize=(25, 16.5), sharey='row', sharex='col',
                             subplot_kw={'projection': proj})
    ax = axes.flatten()
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(hspace=0.15, wspace=-0.1)
    lv = level_title or "1 km AGL"
    titles = [
        f"Z [{lv}] -- precipitating domain",
        f"Z [{lv}] -- display background ({bg_radius:.0f} km smoothing)",
        "Z enhancement | ECE+corridor (magenta) | all CESC (red)",
        "CESC: faint (blue) | strong (red)",
    ]
    kws = dict(levels=np.arange(-10, 45, 5), cmap="jet",
               transform=crs.PlateCarree(), extend="max", alpha=0.8, rasterized=True)
    try:
        [a.set_extent(extent, crs=crs.PlateCarree()) for a in ax]
        [a.add_feature(BORDERS, linewidth=2, edgecolor="k") for a in ax]
        [a.add_feature(STATES, linewidth=2, edgecolor="k", alpha=0.5) for a in ax]
        [a.set_title(t, fontsize=12) for a, t in zip(ax, titles)]
        tc = [a.pcolormesh(ter_da.XLONG, ter_da.XLAT, ter_da,
                           cmap='Greys', transform=crs.PlateCarree()) for a in ax]
        for a in [ax[0], ax[2]]:
            gl = a.gridlines(x_inline=False, alpha=0.45)
            gl.left_labels = True
            gl.yformatter = LATITUDE_FORMATTER
        for a in [ax[2], ax[3]]:
            gl = a.gridlines(x_inline=False, alpha=0.45)
            gl.bottom_labels = True
            gl.xformatter = LONGITUDE_FORMATTER
    except Exception:
        tc = [None]
    strat = result['strat_da']
    ax[0].contourf(strat.XLONG, strat.XLAT,
                   smooth2d(Z_2d_da.where(strat), 4), **kws)
    ax[1].contourf(Zbg_da.XLONG, Zbg_da.XLAT, smooth2d(Zbg_da, 4), **kws)
    ecf = ax[2].contourf(Enh_da.XLONG, Enh_da.XLAT, smooth2d(Enh_da, 4), **kws)
    ax[3].contourf(Z_2d_da.XLONG, Z_2d_da.XLAT, smooth2d(Z_2d_da, 4), **kws)
    supp_da = xr.DataArray(support.astype(np.int8),
                           coords=Z_2d_da.coords, dims=Z_2d_da.dims)
    ax[2].contour(supp_da.XLONG, supp_da.XLAT, supp_da, levels=[0.5],
                  colors='white', linewidths=3, transform=crs.PlateCarree(), zorder=5)
    ax[2].contour(supp_da.XLONG, supp_da.XLAT, supp_da, levels=[0.5],
                  colors='m', linewidths=1.5, transform=crs.PlateCarree(), zorder=5)
    ax[2].contour(labels_env_da.XLONG, labels_env_da.XLAT, labels_env_da > 0,
                  colors='red', linewidths=1.5, transform=crs.PlateCarree(), zorder=6)
    if labels_faint_da.values.any():
        ax[3].contour(labels_faint_da.XLONG, labels_faint_da.XLAT,
                      labels_faint_da > 0, colors='blue', linewidths=1.5,
                      transform=crs.PlateCarree(),zorder=6)
    if labels_strong_da.values.any():
        ax[3].contour(labels_strong_da.XLONG, labels_strong_da.XLAT,
                      labels_strong_da > 0, colors='red', linewidths=1.5,
                      transform=crs.PlateCarree(), zorder=6)
    fig.suptitle(title or 'CESC identification', y=0.92, fontsize=20, fontweight="bold")
    if tc[0] is not None:
        cax = fig.add_axes([0.35, 0.05, 0.35, 0.025])
        plt.colorbar(tc[0], cax=cax, orientation="horizontal").set_label(
            "Terrain height (m)", fontsize=13, fontweight="bold")
    cax2 = fig.add_axes([0.35, -0.02, 0.35, 0.025])
    cb = plt.colorbar(ecf, cax=cax2, orientation="horizontal", ticks=np.arange(-10, 50, 10))
    cb.set_label("Reflectivity (dBZ)", fontsize=14, fontweight="bold")
    return fig
