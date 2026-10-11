"""
Synthetic end-to-end tests.
 
A small domain with a stable-plus-PI environment on its west side, two
planted reflectivity cells (one inside the supportive environment, one
far outside), and a stratiform background. The algorithm must keep the
supported cell, drop the unsupported one at the gate, and classify the
kept one as strong.

"""
import numpy as np
import pytest
import xarray as xr
 
import cesc
from cesc import adapters, preflight
from cesc.environment import scan_pi_stable
from cesc.detect import multiscale_detect, _cc_label_8
 
 
NY, NX = 120, 160
DX = 0.9  # km
 
 
def _profile_columns():
    """theta_e and heights: west half ECE (stable below PI), east half
    neutral."""
    nz = 20
    z = np.linspace(0.0, 6000.0, nz)
    heights = np.broadcast_to(z[:, None, None], (nz, NY, NX)).copy()
 
    theta = np.zeros((nz, NY, NX))
    # neutral slightly stable everywhere
    base = 280.0 + 0.0005 * z
    theta[:] = base[:, None, None]
    # west half: +4 K rise over the lowest 1.5 km, then a 3 K drop
    # between 2 and 3 km (the PI layer)
    west = slice(0, NX // 2)
    rise = np.clip(z / 1500.0, 0, 1) * 4.0
    drop = np.where((z > 2000.0) & (z <= 3000.0),
                    -3.0 * (z - 2000.0) / 1000.0, 0.0)
    drop = np.where(z > 3000.0, -3.0, drop)
    prof = 280.0 + rise + drop
    theta[:, :, west] = prof[:, None, None]
    return theta, heights
 
 
def _reflectivity():
    """Stratiform 8 dBZ everywhere, a 20-pixel-wide 30 dBZ cell inside
    the west environment, and an identical cell in the far east."""
    Z = np.full((NY, NX), 8.0)
    rng = np.random.default_rng(0)
    Z += rng.normal(0, 0.5, size=Z.shape)
    Z[50:60, 30:40] = 30.0     # supported cell (west)
    Z[50:60, 130:140] = 30.0   # unsupported cell (east)
    return Z
 
 
def _dataset():
    theta, heights = _profile_columns()
    Z = _reflectivity()
    u = np.full(theta.shape, 10.0)   # westerly
    v = np.zeros(theta.shape)
    return adapters.from_arrays(
        reflectivity=Z, theta_e=theta, heights=heights,
        u3d=u, v3d=v, dx_km=DX, dy_km=DX)
 
 
def test_scan_finds_pi_and_stable():
    theta, heights = _profile_columns()
    dth = np.diff(theta, axis=0)
    p = cesc.Params()
    pi_str, pi_dep, st_str, st_dep, h0, h1, s0, s1 = scan_pi_stable(
        dth, heights, p.max_pi_height_m, p.unstable_threshold,
        p.stable_threshold, p.min_pi_depth_m, p.max_stable_height_m,
        p.theta_eps)
    west = np.isfinite(pi_str[:, : NX // 2])
    east = np.isfinite(pi_str[:, NX // 2:])
    assert west.all()
    assert not east.any()
    assert np.isfinite(st_str[:, : NX // 2]).all()
    assert np.nanmin(pi_dep[:, : NX // 2]) >= 500.0
 
 
def test_adaptive_thresholds_not_flat():
    """The sigma term must matter: raise local variability so the
    adaptive branch of max(min_diff, k_sigma * sigma) is exercised."""
    Z = np.full((80, 80), 0.0)
    rng = np.random.default_rng(1)
    noisy = rng.normal(0, 30.0, size=(80, 80))
    Z[:, 40:] = noisy[:, 40:]
    Z[20, 20] = 6.0    # exceeds the 5 dBZ floor over the quiet half
    Z[20, 60] = 6.0    # same exceedance fails where sigma is large
    faint, strong = multiscale_detect(
        Z, np.ones_like(Z, bool), DX, DX, (10.0,),
        faint_min_diff=5.0, faint_k_sigma=0.3,
        strong_min_diff=10.0, strong_k_sigma=0.3)
    assert faint[20, 20]
    assert not faint[20, 60]
 
 
def test_cc_label_8_merges_diagonals():
    m = np.zeros((5, 5), bool)
    m[0, 0] = m[1, 1] = m[2, 2] = True
    labels, n = _cc_label_8(m)
    assert n == 1
 
 
def test_end_to_end_gate_keeps_supported_drops_unsupported():
    ds = _dataset()
    p = cesc.Params(env_mode="ECE")
    da_t = ds.isel(time=0)
    theta = da_t.theta_e.values
    heights = da_t.heights.values
    dth = np.diff(theta, axis=0)
    pi_str, _, st_str, _, h0, h1, _, _ = scan_pi_stable(
        dth, heights, p.max_pi_height_m, p.unstable_threshold,
        p.stable_threshold, p.min_pi_depth_m, p.max_stable_height_m,
        p.theta_eps)
    out = cesc.identify_time(
        Z_2d_da=da_t.reflectivity_1km, pi_strength=pi_str,
        stable_strength=st_str, pi_start_h=h0, pi_end_h=h1,
        u3d=da_t.u3d, v3d=da_t.v3d, heights=da_t.heights,
        params=p)
    s = out["summary"]
    assert s["n_kept"] >= 1
    assert s["n_dropped_gate"] >= 1
    labels = out["labels_env_da"].values
    assert labels[55, 35] > 0        # supported cell kept
    assert labels[55, 135] == 0      # unsupported cell dropped
    kept = [o for o in out["objects"] if o["kept_env"]]
    assert all(o["intensity_cls"] == "strong" for o in kept)
    assert all(o["frac_within_support"] >= p.min_support_frac for o in kept)
 
 
def test_pi_only_mode_admits_the_east_cell_when_pi_exists_there():
    """With env_mode PI_only and the east half given PI but no stable
    layer, the east cell must pass."""
    theta, heights = _profile_columns()
    z = heights[:, 0, 0]
    drop = np.where((z > 2000.0) & (z <= 3000.0),
                    -3.0 * (z - 2000.0) / 1000.0, 0.0)
    drop = np.where(z > 3000.0, -3.0, drop)
    east = slice(NX // 2, NX)
    theta[:, :, east] = (280.0 + drop)[:, None, None]  # PI, no stable rise
    Z = _reflectivity()
    u = np.full(theta.shape, 10.0)
    v = np.zeros(theta.shape)
    ds = adapters.from_arrays(reflectivity=Z, theta_e=theta,
                              heights=heights, u3d=u, v3d=v,
                              dx_km=DX, dy_km=DX)
    da_t = ds.isel(time=0)
    dth = np.diff(da_t.theta_e.values, axis=0)
 
    for mode, expect_east in (("ECE", False), ("PI_only", True)):
        p = cesc.Params(env_mode=mode)
        pi_str, _, st_str, _, h0, h1, _, _ = scan_pi_stable(
            dth, da_t.heights.values, p.max_pi_height_m,
            p.unstable_threshold, p.stable_threshold, p.min_pi_depth_m,
            p.max_stable_height_m, p.theta_eps)
        out = cesc.identify_time(
            Z_2d_da=da_t.reflectivity_1km, pi_strength=pi_str,
            stable_strength=st_str, pi_start_h=h0, pi_end_h=h1,
            u3d=da_t.u3d, v3d=da_t.v3d, heights=da_t.heights, params=p)
        got_east = out["labels_env_da"].values[55, 135] > 0
        assert got_east == expect_east, mode
 
 
def test_attribution_and_rainfree_pixels():
    ds = _dataset()
    p = cesc.Params()
    da_t = ds.isel(time=0)
    dth = np.diff(da_t.theta_e.values, axis=0)
    pi_str, _, st_str, _, h0, h1, _, _ = scan_pi_stable(
        dth, da_t.heights.values, p.max_pi_height_m, p.unstable_threshold,
        p.stable_threshold, p.min_pi_depth_m, p.max_stable_height_m,
        p.theta_eps)
    rate = np.zeros((NY, NX))
    rate[52:58, 32:38] = 2.0     # rain only inside part of the west cell
    out = cesc.identify_time(
        Z_2d_da=da_t.reflectivity_1km, pi_strength=pi_str,
        stable_strength=st_str, pi_start_h=h0, pi_end_h=h1,
        u3d=da_t.u3d, v3d=da_t.v3d, heights=da_t.heights,
        precip_rate=rate, params=p)
    assert out["summary"]["precip_frac"] == pytest.approx(1.0)
    kept = [o for o in out["objects"] if o["kept_env"]]
    assert kept[0]["precip_rate_max_mmhr"] == pytest.approx(2.0)
 
 
def test_preflight_catches_theta_and_km_heights_and_accumulation():
    ds = _dataset()
    bad = ds.copy()
    bad["theta_e"] = bad["theta_e"] - 260.0   # ~20 K: not a theta_e
    with pytest.raises(preflight.PreflightError):
        preflight.check_dataset(bad, params=cesc.Params())
 
    bad2 = ds.copy()
    bad2["heights"] = bad2["heights"] / 1000.0
    with pytest.raises(preflight.PreflightError):
        preflight.check_dataset(bad2, params=cesc.Params())
 
    acc = np.cumsum(np.ones((12, NY, NX)), axis=0)
    da = xr.DataArray(acc, dims=("time", "south_north", "west_east"),
                      name="RAINNC")
    with pytest.warns(UserWarning):
        preflight.check_precip_rate(da)
 
 
def test_params_single_source_of_truth():
    p = cesc.Params()
    assert p.faint_min_diff == 5.0
    assert p.strong_min_diff == 10.0
    assert p.box_sizes == (10.0, 20.0, 40.0)
    assert p.min_area_km2 == 4.05
    assert p.bg_radius == 30.0
    assert cesc.Params(env_mode="PI_ONLY").env_mode == "PI_only"
    with pytest.raises(ValueError):
        cesc.Params(env_mode="nonsense")
 
 
def test_corridor_extends_downwind():
    from cesc.environment import downstream_corridor_2d
    mask = np.zeros((60, 60), bool)
    mask[30, 10] = True
    u = np.full((60, 60), 10.0)
    v = np.zeros((60, 60))
    look = np.full((60, 60), 18.0)   # km
    out = downstream_corridor_2d(mask, u, v, grid_km=DX, lookahead_km=look,
                                 halfwidth=2.0, seed_stride_px=1,
                                 min_speed=2.0)
    assert out[30, 25]          # downwind (east) reached
    assert not out[30, 5]       # upwind untouched
 
 
def test_preflight_theta_e_layer_restricted():
    """The check validates only the scanned layer (<= max_pi_height).
    Stratospheric theta_e above the ceiling does not condemn the field;
    a Celsius-scale bulk does."""
    import xarray as xr
    nz, ny, nx = 80, 40, 40
    # heights 0 to 26 km; theta_e rises to ~310 K by 5 km then explodes
    # into the stratosphere the way real theta_e does
    z = np.linspace(0, 26000, nz)
    te = np.where(z < 5000, 295 + z / 5000 * 15,
                  np.where(z < 10000, 310 + (z - 5000) / 5000 * 15,
                           325 + (z - 10000) / 16000 * 230))
    heights = np.broadcast_to(z[:, None, None], (nz, ny, nx)).copy()
 
    def _ds(field):
        return xr.Dataset(dict(
            theta_e=(("time", "bottom_top", "south_north", "west_east"),
                     field[None]),
            heights=(("time", "bottom_top", "south_north", "west_east"),
                     heights[None]),
            reflectivity_1km=(("time", "south_north", "west_east"),
                              np.full((1, ny, nx), 8.0)),
            u3d=(("time", "bottom_top", "south_north", "west_east"),
                 np.zeros((1, nz, ny, nx))),
            v3d=(("time", "bottom_top", "south_north", "west_east"),
                 np.zeros((1, nz, ny, nx)))),
            coords=dict(time=("time", [np.datetime64("2017-02-01")]),
                        XLAT=(("south_north", "west_east"),
                              np.full((ny, nx), 44.0)),
                        XLONG=(("south_north", "west_east"),
                               np.full((ny, nx), -115.0))))
 
    real = te[:, None, None] * np.ones((nz, ny, nx))
    msgs = preflight.check_dataset(_ds(real), params=cesc.Params(),
                                   raise_on_error=False)
    # passes despite a 555 K stratospheric maximum, because the scanned
    # layer below 5 km is physical
    assert not any(m.startswith("FAIL") and "theta_e" in m for m in msgs)
    assert any("below 5000 m" in m for m in msgs)
 
    celsius = real - 273.15
    with pytest.raises(preflight.PreflightError):
        preflight.check_dataset(_ds(celsius), params=cesc.Params())
 
 
def test_partial_output_is_flagged_and_refused(tmp_path):
    """A checkpoint from an unfinished run must announce itself, and
    open_output must refuse it. A killed run otherwise leaves a valid,
    readable, silently short file."""
    import xarray as xr
    ds = _dataset()
    p = cesc.Params(env_mode="ECE")
    out_nc = str(tmp_path / "grids.nc")
    grids, _ = cesc.run_wrf_month(
        ds, out_nc_path=out_nc,
        out_csv_path=str(tmp_path / "obj.csv"),
        params=p, all_files_path=None, verbose=False)
 
    d = xr.open_dataset(out_nc)
    assert d.attrs["cesc_complete"] == "yes"
    assert d.attrs["cesc_timesteps_written"] == d.sizes["time"]
    # a complete file opens without complaint
    assert cesc.open_output(out_nc).sizes["time"] == d.sizes["time"]
 
    # simulate what a killed run leaves behind
    part = d.isel(time=slice(0, 1)).copy()
    part.attrs["cesc_complete"] = "no"
    part.attrs["cesc_timesteps_written"] = 1
    part.attrs["cesc_timesteps_requested"] = int(d.sizes["time"]) + 5
    part_path = str(tmp_path / "partial.nc")
    part.to_netcdf(part_path)
 
    with pytest.raises(RuntimeError):
        cesc.open_output(part_path)
    with pytest.warns(UserWarning):
        cesc.open_output(part_path, require_complete=False)
 
 
def test_parts_removed_on_success_kept_on_failure(tmp_path):
    """Parts are working files: delete them once the stitched outputs
    are verified, keep them when the run is incomplete (they are what
    makes a resubmission resume)."""
    import os
    from cesc import pipeline
    ds = _dataset()
    p = cesc.Params(env_mode="ECE")
 
    # complete run: parts removed, metadata reports the full request
    out = str(tmp_path / "a" / "grids.nc")
    g, _ = cesc.run_wrf_month(
        ds, out_nc_path=out, out_csv_path=str(tmp_path / "a" / "obj.csv"),
        params=p, all_files_path=None, checkpoint_every=1, verbose=False)
    assert not os.path.isdir(out + ".parts")
    assert g.attrs["cesc_complete"] == "yes"
    assert g.attrs["cesc_timesteps_requested"] == g.sizes["time"]
 
    # keep_parts overrides
    out_c = str(tmp_path / "c" / "grids.nc")
    cesc.run_wrf_month(
        ds, out_nc_path=out_c, out_csv_path=str(tmp_path / "c" / "obj.csv"),
        params=p, all_files_path=None, checkpoint_every=1,
        keep_parts=True, verbose=False)
    assert os.path.isdir(out_c + ".parts")