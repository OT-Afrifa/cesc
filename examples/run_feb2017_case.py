"""
examples/run_feb2017_case.py
----------------------------
Single-time-step validation against the well-documented 7 February 2017 CESC
event (SNOWIE IOP12) described in Afrifa et al. (2025, 2026).

This script reproduces the 22 UTC snapshot shown in Figures 2-5 of Afrifa
et al. Part 1 (submitted 2026).  It assumes you have the pre-extracted WRF
900 m output for February 2017 and the corresponding wrfout files.

Usage
-----
::

    python examples/run_feb2017_case.py \
        --nc  /path/to/WRF_extracted_vars_900m_feb2017.nc \
        --wrf /path/to/wrfout_d02_dir/ \
        --out /path/to/output_dir/

Expected output
---------------
- CESC_feb2017_2200UTC_grids.nc   -- 2D gridded fields for this one hour
- CESC_feb2017_2200UTC_objects.parquet  -- per-object statistics
- CESC_feb2017_2200UTC_quicklook.png    -- four-panel figure

The algorithm should detect the large cluster of convective cells advected
from eastern Oregon and the narrow convective bands over the western foothills
of the Idaho Central Mountains, with a domain-wide CESC precipitation fraction
near 25-33 % at this time (consistent with Figure 6b in Afrifa et al. Part 1).
"""

import argparse
import os
import numpy as np
import xarray as xr
import pandas as pd

from cesc.cesc_id import run_cesc_id, scan_pi_stable


def parse_args():
    p = argparse.ArgumentParser(description="Run CESC algorithm for 7 Feb 2017 22 UTC")
    p.add_argument("--nc",  required=True, help="Pre-extracted Feb 2017 NC file")
    p.add_argument("--wrf", required=True, help="Directory with wrfout_d02_* files")
    p.add_argument("--out", required=True, help="Output directory")
    p.add_argument("--time", default="2017-02-07 22:00", help="Target UTC time")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)

    target = pd.Timestamp(args.time)
    print(f"Loading {os.path.basename(args.nc)} ...")
    ds = xr.open_dataset(args.nc, decode_times=True).set_coords(["XLAT","XLONG"])

    print(f"Selecting {target} UTC ...")
    da_t = ds.sel(time=target)

    # Compute delta-theta_e for the vertical stability scan
    theta_e_np  = np.ascontiguousarray(da_t["theta_e"].values.astype(np.float64))
    heights_np  = np.ascontiguousarray(da_t["heights"].values.astype(np.float64))
    dtheta_e    = np.diff(theta_e_np, axis=0)

    print("Running scan_pi_stable ...")
    (pi_str, pi_dep, st_str, st_dep,
     pi_h0, pi_h1, st_h0, st_h1) = scan_pi_stable(
        dtheta_e, heights_np,
        max_pi_height      = 5000.0,
        unstable_threshold = 1.0,
        stable_threshold   = 2.0,
    )

    print("Running CESC ID ...")
    result = run_cesc_id(
        Z_2d_da            = da_t["reflectivity_1km"],
        ter_da             = da_t["ter"],
        HWP_da             = da_t.get("HWP"),
        pi_strength_da     = pi_str,
        stable_strength_da = st_str,
        pi_start_h_da      = pi_h0,
        pi_end_h_da        = pi_h1,
        u3d                = da_t["u3d"],
        v3d                = da_t["v3d"],
        heights            = da_t["heights"],
        # Table 1 defaults
        ece_strict          = False,
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
        frac_strong_cutoff  = 0.5,
        extent              = [-118.17, -109.26, 42.05, 46.198],
        title               = f"CESC identification -- {target.strftime('%Y-%m-%d %H UTC')}",
        make_plot           = True,
    )

    # Summary statistics
    objs_kept = [o for o in result["objects"] if o["kept_env"]]
    n_faint  = sum(1 for o in objs_kept if o["intensity_cls"]=="faint")
    n_strong = sum(1 for o in objs_kept if o["intensity_cls"]=="strong")
    print(f"\nKept CESC objects: {len(objs_kept)} total, "
          f"{n_faint} faint, {n_strong} strong")

    # Save figure
    if result.get("fig") is not None:
        fig_path = os.path.join(args.out,
                                f"CESC_feb2017_"
                                f"{target.strftime('%Y%m%d_%H')}UTC_quicklook.png")
        result["fig"].savefig(fig_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved -> {fig_path}")

    # Save object table
    if objs_kept:
        df = pd.DataFrame(objs_kept)
        df["time"] = target
        pq = os.path.join(args.out,
                          f"CESC_feb2017_{target.strftime('%Y%m%d_%H')}UTC_objects.parquet")
        df.to_parquet(pq, index=False)
        print(f"Objects saved -> {pq}")

    # Save gridded outputs
    import xarray as xr
    from cesc.utils import safe_to_netcdf
    grids = xr.Dataset({
        "labels_env"   : result["labels_env_da"],
        "labels_faint" : result["labels_faint_da"],
        "labels_strong": result["labels_strong_da"],
        "faint_det"    : result["masks"]["faint_det"],
        "strong_det"   : result["masks"]["strong_det"],
        "ece_core"     : result["masks"]["ece_core"],
        "corridor"     : result["masks"]["corridor"],
        "support"      : result["masks"]["support"],
        "Z_bg"         : result["masks"]["Z_bg"],
        "Enh"          : result["masks"]["Enh"],
    })
    nc_out = os.path.join(args.out,
                          f"CESC_feb2017_{target.strftime('%Y%m%d_%H')}UTC_grids.nc")
    safe_to_netcdf(grids, nc_out)
    print(f"Grids saved -> {nc_out}")


if __name__ == "__main__":
    main()
