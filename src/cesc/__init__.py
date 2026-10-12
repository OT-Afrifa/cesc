"""
cesc: identification of convection embedded within or emergent from
cold-season stratiform clouds (Afrifa et al., Part 1, J. Atmos. Sci.).

Quick use:

    import xarray as xr
    import cesc

    ds = xr.open_dataset("WRF_extracted_vars_900m_201702.nc")
    ds = cesc.adapters.from_extracted_wrf(ds)
    grids, objects = cesc.run(
        ds, params=cesc.Params(env_mode="ECE"),
        out_nc_path="out/CESC_ECE_grids.nc",
        out_parquet_path="out/CESC_ECE_objects.parquet",
        all_files_path=None)
"""
__version__ = "0.1.0"

from .params import Params
from .pipeline import (identify_time, run_wrf_month, run,
                       print_summary, open_output)
from . import adapters, preflight, environment, detect, gate, attribute
from . import thermo, distance, contrib

__all__ = ["Params", "identify_time", "run_wrf_month", "run",
           "print_summary", "open_output", "adapters", "preflight", "environment",
           "detect", "gate", "attribute", "thermo", "distance", "contrib",
           "__version__"]
