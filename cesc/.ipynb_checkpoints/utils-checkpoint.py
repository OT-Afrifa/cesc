# cesc/utils.py
"""
Input/output helpers shared across the CESC pipeline.

Covers:
  - wrfout file discovery by timestamp
  - reading RAINNC and cloud-top temperature from wrfout
  - safe NetCDF writing (strips non-serialisable attributes)
  - a streaming NetCDF writer for memory-efficient seasonal runs
"""

import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime
from glob import glob
from netCDF4 import Dataset as NC4Dataset

try:
    from wrf import getvar
    _WRF_AVAILABLE = True
except ImportError:
    _WRF_AVAILABLE = False


# ---------------------------------------------------------------------------
# wrfout file discovery
# ---------------------------------------------------------------------------

def find_wrfout(base_dir: str, ts: pd.Timestamp) -> str | None:
    """
    Return the wrfout file path for the hour given by ts, or None if not found.

    Tries an exact-hour match first (wrfout_d02_YYYY-MM-DD_HH*), then falls
    back to the nearest file on the same calendar day.

    Parameters
    ----------
    base_dir : str -- directory containing wrfout_d02_* files
    ts       : pd.Timestamp -- target hour

    Returns
    -------
    str or None
    """
    ts = pd.Timestamp(ts)
    pat_hour = f"{base_dir}wrfout_d02_{ts.strftime('%Y-%m-%d_%H')}*"
    files = sorted(glob(pat_hour))
    if files:
        return files[0]

    pat_day = f"{base_dir}wrfout_d02_{ts.strftime('%Y-%m-%d')}_*"
    files = sorted(glob(pat_day))
    if not files:
        return None
    return files[-1] if ts.hour >= 12 else files[0]


def read_rainnc(wrfout_path: str) -> np.ndarray:
    """
    Read the accumulated grid-scale precipitation (RAINNC, mm) from a wrfout file.

    Parameters
    ----------
    wrfout_path : str

    Returns
    -------
    (ny, nx) float32 array
    """
    if not _WRF_AVAILABLE:
        raise ImportError("wrf-python is required to read wrfout files.")
    with NC4Dataset(wrfout_path) as d:
        return getvar(d, "RAINNC", timeidx=0).values.astype(np.float32)


def read_ctt(wrfout_path: str) -> np.ndarray:
    """
    Read the column cloud-top temperature (degrees C) from a wrfout file.

    Parameters
    ----------
    wrfout_path : str

    Returns
    -------
    (ny, nx) float32 array
    """
    if not _WRF_AVAILABLE:
        raise ImportError("wrf-python is required to read wrfout files.")
    with NC4Dataset(wrfout_path) as d:
        return getvar(d, "ctt", timeidx=0).values.astype(np.float32)


# ---------------------------------------------------------------------------
# NetCDF attribute scrubbing
# ---------------------------------------------------------------------------

def _is_nc_safe(v) -> bool:
    """Return True if v can be written as a NetCDF attribute."""
    if isinstance(v, (str, bytes, int, float, np.integer, np.floating)):
        return True
    if isinstance(v, (list, tuple)):
        return all(_is_nc_safe(x) for x in v)
    if isinstance(v, np.ndarray):
        return np.issubdtype(v.dtype, np.number) or v.dtype.kind in ("U", "S")
    return False


def scrub_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Remove attributes that NetCDF4 cannot serialise (e.g. cartopy projection
    objects stored by wrf-python).

    Parameters
    ----------
    ds : xr.Dataset

    Returns
    -------
    xr.Dataset with offending attributes stripped
    """
    ds = ds.copy()
    for k in list(ds.attrs):
        if k == "projection" or not _is_nc_safe(ds.attrs[k]):
            del ds.attrs[k]
    for name in list(ds.variables):
        attrs = {k: v for k, v in ds[name].attrs.items()
                 if k != "projection" and _is_nc_safe(v)}
        ds[name].attrs = attrs
    return ds


def safe_to_netcdf(ds: xr.Dataset, out_path: str,
                   encoding: dict | None = None, **kwargs):
    """
    Write an xr.Dataset to NetCDF, scrubbing non-serialisable attributes if
    the first attempt raises TypeError or ValueError.

    Parameters
    ----------
    ds       : xr.Dataset
    out_path : str
    encoding : dict or None
    **kwargs : passed through to ds.to_netcdf()
    """
    try:
        ds.to_netcdf(out_path, encoding=encoding, **kwargs)
    except (TypeError, ValueError) as exc:
        print(f"[safe_to_netcdf] First attempt failed ({type(exc).__name__}): {exc}")
        print("[safe_to_netcdf] Scrubbing attributes and retrying.")
        scrub_attrs(ds).to_netcdf(out_path, encoding=encoding, **kwargs)
        print("[safe_to_netcdf] Succeeded after scrub.")


# ---------------------------------------------------------------------------
# Streaming NetCDF writer
# ---------------------------------------------------------------------------

class StreamingNCWriter:
    """
    Write 2D gridded fields to a NetCDF4 file one time step at a time.

    The output file is created on construction with an UNLIMITED time axis.
    Variables are created on first use so the caller does not need to
    declare them in advance.  Periodic sync calls protect against data loss
    from job interruptions.

    Usage
    -----
    ::

        writer = StreamingNCWriter(path, ny, nx, xlat, xlong)
        for ts, arrays in ...:
            writer.append(ts, {"labels_env": arr1, "rainnc": arr2, ...})
        writer.close()

    Parameters
    ----------
    out_path  : str -- output file path (created fresh; any existing file is
                overwritten)
    ny, nx    : int -- spatial dimensions
    xlat      : (ny, nx) float -- latitudes (written once as a static variable)
    xlong     : (ny, nx) float -- longitudes
    compress  : int -- zlib compression level (1=fast, 9=smallest)
    """

    def __init__(self, out_path: str, ny: int, nx: int,
                 xlat: np.ndarray, xlong: np.ndarray, compress: int = 3):
        self.out_path = out_path
        self._t_idx = 0
        self._compress = compress

        nc = NC4Dataset(out_path, "w", format="NETCDF4")
        nc.createDimension("time", None)   # UNLIMITED
        nc.createDimension("y", ny)
        nc.createDimension("x", nx)
        nc.title = "CESC identification output"
        nc.history = f"created {datetime.utcnow().isoformat()}Z"

        tv = nc.createVariable("time", "f8", ("time",),
                               zlib=True, complevel=compress)
        tv.units = "hours since 1970-01-01 00:00:00"
        tv.calendar = "standard"

        lv = nc.createVariable("XLAT",  "f4", ("y", "x"), zlib=True, complevel=compress)
        ov = nc.createVariable("XLONG", "f4", ("y", "x"), zlib=True, complevel=compress)
        lv[:] = xlat
        ov[:] = xlong

        self._nc = nc

    def _make_var(self, name: str, dtype: str):
        """Create a time-varying 2D variable if it does not exist yet."""
        if name not in self._nc.variables:
            self._nc.createVariable(
                name, dtype, ("time", "y", "x"),
                zlib=True, complevel=self._compress)

    def append(self, timestamp: pd.Timestamp, arrays: dict):
        """
        Write one time step.

        Parameters
        ----------
        timestamp : pd.Timestamp
        arrays    : dict mapping variable name to (ny, nx) array.
                    Boolean arrays are stored as int8.
                    Arrays whose name contains 'label' are stored as int16.
                    Everything else is stored as float32.
        """
        epoch = pd.Timestamp("1970-01-01")
        self._nc["time"][self._t_idx] = (timestamp - epoch).total_seconds() / 3600.0

        for name, arr in arrays.items():
            arr = np.asarray(arr)
            if arr.dtype == bool or arr.dtype == np.bool_:
                dtype = "i1"
                arr = arr.astype(np.int8)
            elif "label" in name:
                dtype = "i2"
                arr = arr.astype(np.int16)
            else:
                dtype = "f4"
                arr = arr.astype(np.float32)
            self._make_var(name, dtype)
            self._nc[name][self._t_idx, :, :] = arr

        self._t_idx += 1
        if self._t_idx % 12 == 0:
            self._nc.sync()

    def close(self):
        """Flush and close the output file."""
        self._nc.sync()
        self._nc.close()
        print(f"[StreamingNCWriter] Closed {self.out_path} ({self._t_idx} steps)")
