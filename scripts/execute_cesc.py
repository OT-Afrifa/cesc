#!/usr/bin/env python
"""
scripts/execute_cesc.py
-------------------
Command-line entry point for the CESC batch runner.

Installed as ``execute-cesc`` when you pip-install the package.

Usage
-----
::

    execute-cesc --nc /path/to/WRF_extracted_vars_900m_feb2017.nc \\
             --wrf /path/to/wrfout_d02_dir/ \\
             --out_nc /path/to/CESC_grids_feb2017.nc \\
             --out_pq /path/to/CESC_objects_feb2017.parquet

All pipeline parameters can be overridden with --key value flags.  Boolean
flags use 1/0 (e.g. --use_advection 1).

Run ``execute-cesc --help`` for the full parameter list.
"""

import argparse
import os


def parse_args():
    p = argparse.ArgumentParser(
        description="Run the CESC identification pipeline on a monthly WRF file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Required paths
    p.add_argument("--nc",     required=True, help="Pre-extracted monthly NC file")
    p.add_argument("--wrf",    required=True, help="Directory containing wrfout_d02_* files")
    p.add_argument("--out_nc", required=True, help="Output gridded NetCDF path")
    p.add_argument("--out_pq", required=True, help="Output object parquet path")

    # Optional cesc id parameters (Table 1 defaults)
    p.add_argument("--unstable_threshold",  type=float, default=1.0)
    p.add_argument("--stable_threshold",    type=float, default=2.0)
    p.add_argument("--faint_min_diff",      type=float, default=2.0)
    p.add_argument("--faint_k_sigma",       type=float, default=0.30)
    p.add_argument("--strong_min_diff",     type=float, default=5.0)
    p.add_argument("--strong_k_sigma",      type=float, default=0.30)
    p.add_argument("--bg_radius",           type=float, default=30.0)
    p.add_argument("--min_area",            type=float, default=6.0)
    p.add_argument("--min_support_frac",    type=float, default=0.30)
    p.add_argument("--min_support_px",      type=int,   default=5)
    p.add_argument("--advection_time",      type=int,   default=20)
    p.add_argument("--halfwidth",           type=float, default=10.0)
    p.add_argument("--frac_strong_cutoff",  type=float, default=0.5)
    p.add_argument("--use_advection",       type=int,   default=1,
                   help="1=True 0=False")
    p.add_argument("--ece_strict",          type=int,   default=0,
                   help="1=True 0=False")
    p.add_argument("--env_mode",            type=str,   default="ECE")
    p.add_argument("--compress",            type=int,   default=3)
    p.add_argument("--flush_every",         type=int,   default=24)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out_nc)), exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_pq)), exist_ok=True)

    from cesc.run_cesc import run_month_streaming

    cesc_id_kwargs = dict(
        unstable_threshold = args.unstable_threshold,
        stable_threshold   = args.stable_threshold,
        faint_min_diff     = args.faint_min_diff,
        faint_k_sigma      = args.faint_k_sigma,
        strong_min_diff    = args.strong_min_diff,
        strong_k_sigma     = args.strong_k_sigma,
        bg_radius          = args.bg_radius,
        min_area           = args.min_area,
        min_support_frac   = args.min_support_frac,
        min_support_px     = args.min_support_px,
        advection_time     = args.advection_time,
        halfwidth          = args.halfwidth,
        frac_strong_cutoff = args.frac_strong_cutoff,
        use_advection      = bool(args.use_advection),
        ece_strict         = bool(args.ece_strict),
        env_mode           = args.env_mode,
        make_plot          = False,
    )

    run_month_streaming(
        nc_path          = args.nc,
        wrfout_path      = args.wrf,
        out_nc_path      = args.out_nc,
        out_parquet_path = args.out_pq,
        cesc_id_kwargs  = cesc_id_kwargs,
        flush_every      = args.flush_every,
        compress         = args.compress,
    )


if __name__ == "__main__":
    main()
