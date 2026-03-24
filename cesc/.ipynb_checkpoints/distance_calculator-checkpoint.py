# cesc/distance_calculator.py
"""
Grid-spacing estimator for WRF lat/lon coordinate arrays.

The WRF model uses a Lambert Conformal projection internally but writes
output on a regular lat/lon grid.  Grid spacing in km therefore varies
slightly across the domain.  This module provides a simple domain-averaged
estimate that is accurate enough for the box-size computations in the
detection algorithm.
"""

import numpy as np


def estimate_dx_dy_km(lat2d: np.ndarray, lon2d: np.ndarray) -> tuple[float, float]:
    """
    Estimate the average grid spacing in kilometres from a 2D lat/lon array.

    The east-west spacing is derived from a central row of the domain and
    accounts for the convergence of meridians via cos(lat).  The north-south
    spacing uses a central column.  Both are averaged across the interior of
    the domain (ignoring the outermost 10 rows/columns) to reduce edge effects
    from projection distortion.

    Parameters
    ----------
    lat2d : (ny, nx) float array -- latitudes in degrees
    lon2d : (ny, nx) float array -- longitudes in degrees

    Returns
    -------
    dx_km : float -- mean east-west grid spacing (km)
    dy_km : float -- mean north-south grid spacing (km)

    Notes
    -----
    1 degree of latitude ~ 111.0 km everywhere.
    1 degree of longitude ~ 111.0 * cos(lat) km.
    """
    lat2d = np.asarray(lat2d, dtype=np.float64)
    lon2d = np.asarray(lon2d, dtype=np.float64)

    ny, nx = lat2d.shape
    pad = min(10, ny // 4, nx // 4)

    # East-west spacing: average over interior rows
    mid_rows = slice(pad, ny - pad)
    mean_lat = np.mean(lat2d[mid_rows, :])
    lon_km = 111.0 * np.cos(np.deg2rad(mean_lat))
    dx_km = float(np.mean(np.abs(np.diff(lon2d[mid_rows, :], axis=1))) * lon_km)

    # North-south spacing: average over interior columns
    mid_cols = slice(pad, nx - pad)
    dy_km = float(np.mean(np.abs(np.diff(lat2d[:, mid_cols], axis=0))) * 111.0)

    # Guard against degenerate or single-row/column arrays
    if not np.isfinite(dx_km) or dx_km <= 0:
        dx_km = 0.9
    if not np.isfinite(dy_km) or dy_km <= 0:
        dy_km = 0.9

    return dx_km, dy_km
