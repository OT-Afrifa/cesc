"""Command line: cesc run config.yaml and cesc check data.nc."""
import argparse
import sys


def _cmd_check(args):
    import xarray as xr
    from .preflight import check_dataset, check_precip_rate
    from .params import Params
    ds = xr.open_dataset(args.path)
    print("cesc check: %s" % args.path)
    msgs = check_dataset(ds, params=Params(), raise_on_error=False,
                         verbose=False)
    for m in msgs:
        print(" ", m)
    for cand in ("precip_rate", "prate_surface"):
        if cand in ds:
            for m in check_precip_rate(ds[cand], name=cand):
                print(" ", m)
    bad = [m for m in msgs if m.startswith("FAIL")]
    print("readiness: %s" % ("NOT READY (%d failures)" % len(bad)
                             if bad else "ready"))
    return 1 if bad else 0


def _cmd_run(args):
    import yaml
    import xarray as xr
    from .params import Params
    from .pipeline import run_wrf_month
    from .adapters import from_extracted_wrf

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    params = Params.from_dict(cfg.get("params", {}))
    ds = from_extracted_wrf(xr.open_dataset(cfg["input"]))
    prate = None
    if cfg.get("precip_rate_file"):
        prate = xr.open_dataset(cfg["precip_rate_file"])[
            cfg.get("precip_rate_var", "prate_surface")]
    run_wrf_month(
        ds,
        out_nc_path=cfg["out_nc"],
        out_csv_path=cfg.get("out_csv") or cfg["out_parquet"],
        params=params,
        all_files_path=cfg.get("wrfout_dir"),
        precip_rate_da=prate,
        checkpoint_every=cfg.get("checkpoint_every"),
    )
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(
        prog="cesc",
        description="CESC identification (Afrifa et al., Part 1, JAS).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="preflight a NetCDF file")
    c.add_argument("path")
    c.set_defaults(fn=_cmd_check)
    r = sub.add_parser("run", help="run from a YAML config")
    r.add_argument("config")
    r.set_defaults(fn=_cmd_run)
    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
