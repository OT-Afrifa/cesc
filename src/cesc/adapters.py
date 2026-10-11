"""Data adapters.

One internal contract, several ways in. The pipeline consumes an
xarray Dataset with theta_e (time, bottom_top, y, x) in K, heights
(same shape) in m AGL, reflectivity_1km (time, y, x) in dBZ, u3d and
v3d for the corridor wind, optional HWP and ter, and XLAT/XLONG/time
coordinates. Adding an adapter must not touch the algorithm modules.
"""
import numpy as np
import xarray as xr

from .thermo import theta_e_bolton


def from_extracted_wrf(ds):
    """Adapter for the project's extracted 900 m monthly files.

    These already carry the contract variables; this validates presence
    and sets XLAT/XLONG as coordinates.
    """
    for v in ("XLAT", "XLONG"):
        if v in ds and v not in ds.coords:
            ds = ds.set_coords(v)
    return ds


def from_generic(ds, name_map, dx_km=None, dy_km=None):
    """Rename-and-check adapter for CF-ish NetCDF.

    Parameters
    ----------
    ds : xr.Dataset
    name_map : dict
        Maps the caller's names onto the contract names, for example
        {"REFL_1KM": "reflectivity_1km", "THE": "theta_e",
         "Z_AGL": "heights", "U": "u3d", "V": "v3d",
         "lat2d": "XLAT", "lon2d": "XLONG"}.
    dx_km, dy_km : float, optional
        Uniform spacing when no 2-D lat/lon exists; synthetic XLAT and
        XLONG are then built so the spacing estimator returns them.
    """
    ds = ds.rename(dict((k, v) for k, v in name_map.items() if k in ds))
    if "XLAT" not in ds and dx_km is not None and dy_km is not None:
        ref = ds["reflectivity_1km"]
        ny, nx = ref.shape[-2], ref.shape[-1]
        lat0 = 45.0
        lat = lat0 + (np.arange(ny) * dy_km) / 111.0
        lon = (np.arange(nx) * dx_km) / (111.0 * np.cos(np.deg2rad(lat0)))
        lon2d, lat2d = np.meshgrid(lon, lat)
        ds = ds.assign_coords(
            XLAT=(ref.dims[-2:], lat2d), XLONG=(ref.dims[-2:], lon2d))
    for v in ("XLAT", "XLONG"):
        if v in ds and v not in ds.coords:
            ds = ds.set_coords(v)
    return ds


def from_arrays(reflectivity, theta_e=None, heights=None, u3d=None, v3d=None,
                lat2d=None, lon2d=None, dx_km=None, dy_km=None, times=None,
                temperature=None, pressure=None, qv=None, precip_rate=None):
    """Plain-numpy adapter, for radar users and tests.

    Supply either theta_e directly or temperature, pressure, and qv,
    from which theta_e is computed with Bolton (1980). Supply either
    2-D lat/lon or uniform dx_km/dy_km.

    Parameters
    ----------
    reflectivity : (nt, ny, nx) or (ny, nx) float, dBZ.
    theta_e : (nt, nz, ny, nx) or (nz, ny, nx) float, K, optional.
    heights : same shape as theta_e, m AGL.
    u3d, v3d : winds matching theta_e's vertical count.
    lat2d, lon2d : (ny, nx) float, degrees.
    dx_km, dy_km : float, used when lat/lon are absent.
    times : array-like of datetimes, optional.
    temperature, pressure, qv : arrays for the Bolton computation, K,
        Pa, kg/kg.
    precip_rate : (nt, ny, nx) float, mm/h, optional.

    Returns
    -------
    xr.Dataset in the internal contract.
    """
    Z = np.asarray(reflectivity, dtype=np.float64)
    if Z.ndim == 2:
        Z = Z[None, ...]
    nt, ny, nx = Z.shape

    if theta_e is None:
        if temperature is None or pressure is None or qv is None:
            raise ValueError("supply theta_e, or temperature, pressure, "
                             "and qv for the Bolton computation")
        theta_e = theta_e_bolton(temperature, pressure, qv)
    th = np.asarray(theta_e, dtype=np.float64)
    if th.ndim == 3:
        th = th[None, ...]
    if heights is None:
        raise ValueError("heights (m AGL, full levels) are required")
    hz = np.asarray(heights, dtype=np.float64)
    if hz.ndim == 3:
        hz = hz[None, ...]

    if lat2d is None or lon2d is None:
        if dx_km is None or dy_km is None:
            raise ValueError("supply lat2d/lon2d or dx_km/dy_km")
        lat0 = 45.0
        lat = lat0 + (np.arange(ny) * dy_km) / 111.0
        lon = (np.arange(nx) * dx_km) / (111.0 * np.cos(np.deg2rad(lat0)))
        lon2d, lat2d = np.meshgrid(lon, lat)

    if times is None:
        times = np.array([np.datetime64("2000-01-01") +
                          np.timedelta64(k, "h") for k in range(nt)])

    data = dict(
        reflectivity_1km=(("time", "south_north", "west_east"), Z),
        theta_e=(("time", "bottom_top", "south_north", "west_east"), th),
        heights=(("time", "bottom_top", "south_north", "west_east"), hz),
    )
    if u3d is not None:
        u = np.asarray(u3d, dtype=np.float64)
        v = np.asarray(v3d, dtype=np.float64)
        if u.ndim == 3:
            u = u[None, ...]
            v = v[None, ...]
        data["u3d"] = (("time", "bottom_top", "south_north", "west_east"), u)
        data["v3d"] = (("time", "bottom_top", "south_north", "west_east"), v)
    if precip_rate is not None:
        pr = np.asarray(precip_rate, dtype=np.float64)
        if pr.ndim == 2:
            pr = pr[None, ...]
        data["precip_rate"] = (("time", "south_north", "west_east"), pr)

    ds = xr.Dataset(
        data,
        coords=dict(
            time=("time", np.asarray(times)),
            XLAT=(("south_north", "west_east"), np.asarray(lat2d)),
            XLONG=(("south_north", "west_east"), np.asarray(lon2d)),
        ),
    )
    return ds


def from_wrf(path_or_ds, times=None, z_agl_km=1.0):
    """Adapter for raw wrfout files via wrf-python.

    Computes theta_e (Bolton), destaggers winds, interpolates
    reflectivity to z_agl_km above ground, and returns the contract
    Dataset. Requires netCDF4 and wrf-python; import fails otherwise
    with an instruction to install them or extract the fields once and
    use from_extracted_wrf.
    """
    try:
        from netCDF4 import Dataset as NCDataset
        from wrf import getvar, destagger, interplevel, ALL_TIMES
    except ImportError:
        raise ImportError(
            "from_wrf needs netCDF4 and wrf-python (conda install "
            "wrf-python). Alternatively extract theta_e, heights, "
            "reflectivity_1km, u3d, v3d once and use from_extracted_wrf.")

    close = False
    if isinstance(path_or_ds, str):
        nc = NCDataset(path_or_ds)
        close = True
    else:
        nc = path_or_ds
    try:
        tk = getvar(nc, "tk", timeidx=ALL_TIMES if times is None else times)
        p = getvar(nc, "p", timeidx=ALL_TIMES if times is None else times,
                   units="Pa")
        qv = getvar(nc, "QVAPOR",
                    timeidx=ALL_TIMES if times is None else times)
        z = getvar(nc, "height_agl",
                   timeidx=ALL_TIMES if times is None else times)
        dbz = getvar(nc, "dbz", timeidx=ALL_TIMES if times is None else times)
        ua = getvar(nc, "ua", timeidx=ALL_TIMES if times is None else times)
        va = getvar(nc, "va", timeidx=ALL_TIMES if times is None else times)
        lat = getvar(nc, "XLAT", timeidx=0)
        lon = getvar(nc, "XLONG", timeidx=0)

        th_e = theta_e_bolton(np.asarray(tk), np.asarray(p), np.asarray(qv))
        z1km = interplevel(dbz, z, z_agl_km * 1000.0)

        squeeze_time = "Time" not in tk.dims
        def _t(a):
            arr = np.asarray(a)
            return arr[None, ...] if squeeze_time else arr

        nt = 1 if squeeze_time else tk.sizes["Time"]
        tvals = (np.atleast_1d(np.asarray(getvar(nc, "times",
                 timeidx=ALL_TIMES if times is None else times))))
        ds = xr.Dataset(
            dict(
                theta_e=(("time", "bottom_top", "south_north", "west_east"),
                         _t(th_e)),
                heights=(("time", "bottom_top", "south_north", "west_east"),
                         _t(np.asarray(z))),
                reflectivity_1km=(("time", "south_north", "west_east"),
                                  _t(np.asarray(z1km))),
                u3d=(("time", "bottom_top", "south_north", "west_east"),
                     _t(np.asarray(ua))),
                v3d=(("time", "bottom_top", "south_north", "west_east"),
                     _t(np.asarray(va))),
            ),
            coords=dict(
                time=("time", tvals[:nt]),
                XLAT=(("south_north", "west_east"), np.asarray(lat)),
                XLONG=(("south_north", "west_east"), np.asarray(lon)),
            ),
        )
        return ds
    finally:
        if close:
            nc.close()
