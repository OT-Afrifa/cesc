# Stage 1 kernels ported from the research notebook (cells building
# scan_pi_stable, layer_mean_wind_numba, downstream_corridor_2d).
# Numerical behavior preserved; numba optional via cesc._compat.
"""Stage 1: the supportive environment.

From vertical profiles of equivalent potential temperature the scan
identifies, per column, a potential-instability (PI) layer and, beneath
it, a stable near-surface layer. Columns with both form the ECE core
(elevated convection environment); columns with the PI layer but a weak
or absent stable layer are the SCE (surface-based convection
environment). The env_mode switch in Params selects which environments
support an object downstream: "ECE" gates on the stable-plus-PI columns
only, "PI_only" gates on every PI column (ECE plus SCE together).

The core is dilated by one pixel and extended downwind: from seed
pixels, rays run along the local PI-layer mass-weighted wind for
wind speed times advection_time, then widen laterally by halfwidth_km.
The support mask is core OR corridor.
"""
import numpy as np
from scipy.ndimage import binary_dilation

from ._compat import njit, prange


@njit(cache=True, fastmath=True)
def _surface_stable_scan(dt, heights_full, pi_start_k, max_stable_height_m, eps):
    """Scan upward from the surface accumulating positive theta_e rise.

    Stops at min(pi_start_k, max_stable_height_m). Small negative
    wiggles (|d theta_e| < eps) do not reset the accumulation, which
    tolerates well-mixed layers capped by higher-theta_e air.

    Parameters
    ----------
    dt : (nz,) float64
        Layer delta-theta_e (K).
    heights_full : (nz+1,) float64
        Full-level heights AGL (m).
    pi_start_k : int
        First PI level index (upper bound of the scan).
    max_stable_height_m : float
        Absolute height ceiling (m AGL).
    eps : float
        Noise tolerance (K).

    Returns
    -------
    cumsum : float
        Cumulative positive theta_e rise (K).
    ee : int
        Exclusive end index of the identified stable layer.
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
def scan_pi_stable(dtheta_e, heights_full, max_pi_height,
                   unstable_threshold, stable_threshold,
                   min_pi_depth, max_stable_height, eps):
    """Per-column PI layer and surface stable layer from theta_e profiles.

    PI layer: consecutive levels with delta-theta_e < -eps whose
    cumulative drop is >= unstable_threshold, whose depth is
    >= min_pi_depth, and whose base lies below max_pi_height. The first
    qualifying run (scanning upward) is recorded; the stable scan then
    runs from the surface to min(PI base, max_stable_height) and must
    accumulate >= stable_threshold.

    Parameters
    ----------
    dtheta_e : (nz, ny, nx) float64
        Layer delta-theta_e (K), np.diff(theta_e, axis=0).
    heights_full : (nz+1, ny, nx) float64
        Full-level heights AGL (m).
    max_pi_height : float
        PI search ceiling (m AGL); 5000 in the paper.
    unstable_threshold : float
        Minimum PI strength (K), positive convention; 1 in the paper.
    stable_threshold : float
        Minimum stable-layer rise (K); 2 in the paper.
    min_pi_depth : float
        Minimum PI depth (m); 500 in the paper.
    max_stable_height : float
        Stable scan ceiling (m AGL); 3000 in the paper.
    eps : float
        Noise tolerance (K); 5e-3 in the paper.

    Returns
    -------
    Eight (ny, nx) float64 arrays, NaN where the criterion is not met:
    pi_strength, pi_depth, stable_strength, stable_depth,
    pi_height_start, pi_height_end, stable_height_start,
    stable_height_end.
    """
    nz, ny, nx = dtheta_e.shape

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
            dt = dtheta_e[:, j, i]

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
                        s = run_start
                        e = k
                        drop = 0.0
                        valid = False
                        for kk in range(s, e):
                            vv = dt[kk]
                            if np.isfinite(vv):
                                drop += vv
                                valid = True
                        depth = hf[e] - hf[s]
                        if (valid and drop <= -unstable_threshold
                                and depth >= min_pi_depth):
                            pi_str[j, i] = drop
                            pi_h0[j, i] = hf[s]
                            pi_h1[j, i] = hf[e]
                            pi_dep[j, i] = depth
                            cumsum, ee = _surface_stable_scan(
                                dt, hf, s, max_stable_height, eps)
                            if cumsum >= stable_threshold and ee > 0:
                                st_str[j, i] = cumsum
                                st_h0[j, i] = hf[0]
                                st_h1[j, i] = hf[ee]
                                st_dep[j, i] = hf[ee] - hf[0]
                            found_pi = True
                        run_start = -1
                if found_pi:
                    break

            # a run still open at the ceiling
            if not found_pi and run_start != -1:
                s = run_start
                e = max_lev + 1
                drop = 0.0
                valid = False
                for kk in range(s, e):
                    vv = dt[kk]
                    if np.isfinite(vv):
                        drop += vv
                        valid = True
                depth = hf[e] - hf[s]
                if (valid and drop <= -unstable_threshold
                        and depth >= min_pi_depth):
                    pi_str[j, i] = drop
                    pi_h0[j, i] = hf[s]
                    pi_h1[j, i] = hf[e]
                    pi_dep[j, i] = depth
                    cumsum, ee = _surface_stable_scan(
                        dt, hf, s, max_stable_height, eps)
                    if cumsum >= stable_threshold and ee > 0:
                        st_str[j, i] = cumsum
                        st_h0[j, i] = hf[0]
                        st_h1[j, i] = hf[ee]
                        st_dep[j, i] = hf[ee] - hf[0]

    return pi_str, pi_dep, st_str, st_dep, pi_h0, pi_h1, st_h0, st_h1


@njit(parallel=True, fastmath=True, cache=True)
def layer_mean_wind(u_layer, v_layer, z_full, h_start, h_end,
                    default_lo, default_hi):
    """Mass-weighted mean wind within each column's PI layer.

    Integration limits come from h_start / h_end (m AGL) and fall back
    to default_lo / default_hi where those are NaN or degenerate.

    Parameters
    ----------
    u_layer, v_layer : (nzm, ny, nx) float32
        Mid-level winds (m/s).
    z_full : (nzm+1, ny, nx) float32
        Full-level heights AGL (m).
    h_start, h_end : (ny, nx) float32
        PI-layer base and top (m AGL).
    default_lo, default_hi : float
        Fallback layer bounds (m AGL).

    Returns
    -------
    ubar, vbar : (ny, nx) float32, NaN where no integration layer.
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
                zbot = z_full[k, j, i]
                ztop = z_full[k + 1, j, i]
                lo = max(zbot, hs)
                hi = min(ztop, he)
                th = hi - lo
                if th > 0.0:
                    numu += u_layer[k, j, i] * th
                    numv += v_layer[k, j, i] * th
                    den += th
            if den > 0.0:
                ubar[j, i] = numu / den
                vbar[j, i] = numv / den
            else:
                ubar[j, i] = np.nan
                vbar[j, i] = np.nan
    return ubar, vbar


def _disk(radius_px):
    """Circular boolean structuring element of a given pixel radius."""
    r = max(0, int(radius_px))
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    return (yy * yy + xx * xx) <= r * r


def _draw_line_boolean(y0, x0, y1, x1, out):
    """Bresenham line rasterization onto a boolean array; clips silently."""
    y0, x0, y1, x1 = int(y0), int(x0), int(y1), int(x1)
    dy = abs(y1 - y0)
    dx = abs(x1 - x0)
    sy = 1 if y0 < y1 else -1
    sx = 1 if x0 < x1 else -1
    err = dx - dy
    while True:
        if 0 <= y0 < out.shape[0] and 0 <= x0 < out.shape[1]:
            out[y0, x0] = True
        if y0 == y1 and x0 == x1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy


def downstream_corridor_2d(ece_mask, u_bar, v_bar, grid_km, lookahead_km,
                           halfwidth, seed_stride_px, min_speed):
    """Downstream advection corridor from environment seed pixels.

    From each seed (every seed_stride_px pixels of the mask), a line is
    drawn along the local PI-layer wind for lookahead_km[j, i], then the
    skeleton is dilated laterally by halfwidth km.

    Row order convention: rows increase southward in the reference grid,
    so the y-component of the ray is -v.

    Parameters
    ----------
    ece_mask : (ny, nx) bool
    u_bar, v_bar : (ny, nx) float
        PI-layer mean wind (m/s); masked arrays accepted.
    grid_km : float
        Isotropic grid spacing (km).
    lookahead_km : (ny, nx) float
        Per-point downstream reach (km).
    halfwidth : float
        Lateral half-width (km).
    seed_stride_px : int
        Seed spacing (pixels).
    min_speed : float
        Minimum wind speed to draw a ray (m/s).

    Returns
    -------
    out : (ny, nx) bool
    """
    ny, nx = ece_mask.shape
    out = np.zeros((ny, nx), dtype=bool)
    Lpx = np.round(np.asarray(lookahead_km) / grid_km).astype(int)
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
            nyv = -vy / (spd + 1e-9)
            _draw_line_boolean(y, x, y + nyv * Lpx[y, x],
                               x + nxu * Lpx[y, x], out)
    if halfw_px > 0:
        out = binary_dilation(out, structure=dil_foot)
    return out


def environment_masks(pi_strength, stable_strength, params):
    """Environment masks from the per-column scan outputs.

    Parameters
    ----------
    pi_strength : (ny, nx) float
        Cumulative theta_e drop of the PI layer (K, negative values),
        NaN where no qualifying layer.
    stable_strength : (ny, nx) float or None
        Cumulative theta_e rise of the stable layer (K), NaN where none.
    params : Params

    Returns
    -------
    pi_mask, stable_mask, env_mask, core : (ny, nx) bool
        env_mask is the gating environment selected by params.env_mode
        (stable AND PI for "ECE", PI alone for "PI_only"); core is
        env_mask dilated by a one-pixel disk.
    """
    pi_arr = np.asarray(pi_strength)
    pi_mask = np.isfinite(pi_arr) & (pi_arr <= -abs(params.unstable_threshold))

    if stable_strength is not None:
        st_arr = np.asarray(stable_strength)
        stable_mask = np.isfinite(st_arr) & (
            st_arr >= abs(params.stable_threshold))
    else:
        stable_mask = np.zeros_like(pi_mask, dtype=bool)

    if params.env_mode == "ECE" and stable_strength is not None:
        env_mask = pi_mask & stable_mask
    else:
        env_mask = pi_mask

    core = binary_dilation(env_mask, structure=_disk(1))
    return pi_mask, stable_mask, env_mask, core
