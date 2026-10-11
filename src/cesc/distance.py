# Ported from the research module distance_calculator.py
# (Francis O.T. Afrifa, 2026). Numerical behavior preserved.
"""Grid-spacing estimators for 2-D lat/lon coordinate arrays.

WRF runs on a Lambert Conformal projection but writes output on a
near-regular lat/lon grid, so spacing in km varies slightly across the
domain. A domain-average estimate is accurate enough for the box-size
computations in the detection algorithm.
"""
import numpy as np


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) between two points given in degrees."""
    R = 6371.0
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dlat = np.deg2rad(lat2 - lat1)
    dlon = np.deg2rad(lon2 - lon1)
    a = np.sin(dlat / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2.0) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


def estimate_dx_dy_km(lats2d, lons2d):
    """Domain-average grid spacing (km) from 2-D lat/lon arrays.

    This is the estimator the research pipeline uses: finite differences
    at the domain center with a simple spherical scale factor, falling
    back to medians along the center lines when the center differences
    degenerate.

    Parameters
    ----------
    lats2d, lons2d : (ny, nx) float arrays
        Latitude and longitude in degrees.

    Returns
    -------
    dx_km, dy_km : float
    """
    lats2d = np.asarray(lats2d)
    lons2d = np.asarray(lons2d)
    ny, nx = lats2d.shape
    j = ny // 2
    i = nx // 2
    j2 = min(j + 1, ny - 1)
    i2 = min(i + 1, nx - 1)
    lat_km = 111.0
    lon_km = 111.0 * np.cos(np.deg2rad(lats2d[j, i]))
    dy_km = abs(lats2d[j2, i] - lats2d[j, i]) * lat_km
    dx_km = abs(lons2d[j, i2] - lons2d[j, i]) * lon_km
    if dx_km <= 0 or dy_km <= 0:
        dy_km = np.median(np.abs(np.diff(lats2d[:, i]))) * lat_km
        dx_km = np.median(np.abs(np.diff(lons2d[j, :]))) * lon_km
    return float(dx_km), float(dy_km)


def estimate_dx_dy_km_haversine(lat2d, lon2d):
    """Haversine-based spacing estimate (km); robust median over the grid."""
    lat2d = np.asarray(lat2d)
    lon2d = np.asarray(lon2d)
    dx = haversine_km(lat2d[:, :-1], lon2d[:, :-1], lat2d[:, 1:], lon2d[:, 1:])
    dy = haversine_km(lat2d[:-1, :], lon2d[:-1, :], lat2d[1:, :], lon2d[1:, :])
    return float(np.nanmedian(dx)), float(np.nanmedian(dy))
