# tests/test_basic.py
"""
Smoke tests for the CESC package.

These tests do not require WRF data.  They verify that the core functions
return the correct output shapes and types on synthetic inputs.
"""

import numpy as np
import pytest
import xarray as xr

from cesc.distance_calculator import estimate_dx_dy_km
from cesc.id_pro import (
    detect_objects_id_pro,
    _cc_label_8,
    _compute_moments_and_axes,
    _classify,
)


# ---------------------------------------------------------------------------
# Grid spacing
# ---------------------------------------------------------------------------

def test_estimate_dx_dy_simple():
    lat = np.linspace(42.0, 46.0, 50)
    lon = np.linspace(-118.0, -109.0, 60)
    lon2d, lat2d = np.meshgrid(lon, lat)
    dx, dy = estimate_dx_dy_km(lat2d, lon2d)
    assert 0.5 < dx < 20.0, f"dx_km out of range: {dx}"
    assert 0.5 < dy < 20.0, f"dy_km out of range: {dy}"


# ---------------------------------------------------------------------------
# Connected-component labeller
# ---------------------------------------------------------------------------

def test_cc_single_blob():
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    labels, n = _cc_label_8(mask)
    assert n == 1
    assert (labels[mask] == 1).all()
    assert (labels[~mask] == 0).all()


def test_cc_two_blobs():
    mask = np.zeros((10, 20), dtype=bool)
    mask[2:5, 2:5] = True
    mask[2:5, 15:18] = True
    labels, n = _cc_label_8(mask)
    assert n == 2


def test_cc_empty():
    mask = np.zeros((8, 8), dtype=bool)
    labels, n = _cc_label_8(mask)
    assert n == 0
    assert labels.sum() == 0


# ---------------------------------------------------------------------------
# Object geometry
# ---------------------------------------------------------------------------

def test_moments_circle():
    r = 10
    jj, ii = np.where(
        (np.arange(50)[:, None]-25)**2 + (np.arange(50)[None, :]-25)**2 <= r**2
    )
    yx, (L, W, asp, ang) = _compute_moments_and_axes(jj, ii, 0.9, 0.9)
    # A circle should have aspect near 1 and finite axes
    assert np.isfinite(L) and np.isfinite(W)
    assert 0.7 < asp <= 1.0, f"Expected aspect near 1, got {asp}"


def test_classify_band():
    assert _classify(150.0, 30.0, 0.2) == "band"


def test_classify_cell():
    assert _classify(50.0, 40.0, 0.8) == "cell"


def test_classify_complex():
    assert _classify(np.nan, np.nan, np.nan) == "complex"


# ---------------------------------------------------------------------------
# Detection function
# ---------------------------------------------------------------------------

def _make_field(ny=100, nx=120, n_cells=3, cell_strength=15.0, seed=42):
    """Synthetic reflectivity with a stratiform background and a few cells."""
    rng = np.random.default_rng(seed)
    background = rng.normal(5.0, 2.0, (ny, nx)).astype(np.float32)
    for _ in range(n_cells):
        cy = rng.integers(20, ny-20)
        cx = rng.integers(20, nx-20)
        background[cy-3:cy+3, cx-3:cx+3] += cell_strength
    lat = np.linspace(42.0, 46.0, ny)
    lon = np.linspace(-118.0, -109.0, nx)
    lon2d, lat2d = np.meshgrid(lon, lat)
    da = xr.DataArray(
        background,
        dims=["y", "x"],
        attrs={"long_name": "Reflectivity at 1 km AGL"},
    )
    da = da.assign_coords(XLAT=(["y", "x"], lat2d.astype(np.float32)),
                          XLONG=(["y", "x"], lon2d.astype(np.float32)))
    return da, lat2d.astype(np.float32), lon2d.astype(np.float32)


def test_detect_returns_correct_shapes():
    da, lats, lons = _make_field()
    ny, nx = da.shape
    gate = np.ones((ny, nx), dtype=bool)
    faint, strong, labels, props = detect_objects_id_pro(
        da, gate, lats, lons, min_area=1.0)
    assert faint.shape == (ny, nx)
    assert strong.shape == (ny, nx)
    assert labels.shape == (ny, nx)
    assert isinstance(props, list)


def test_detect_finds_cells():
    da, lats, lons = _make_field(n_cells=3, cell_strength=20.0)
    ny, nx = da.shape
    gate = np.ones((ny, nx), dtype=bool)
    faint, strong, labels, props = detect_objects_id_pro(
        da, gate, lats, lons, min_area=1.0)
    # Should find at least 1 object in a field with clear spikes
    assert len(props) >= 1, "Expected at least one detected object"


def test_strong_subset_of_faint():
    da, lats, lons = _make_field()
    ny, nx = da.shape
    gate = np.ones((ny, nx), dtype=bool)
    faint, strong, _, _ = detect_objects_id_pro(da, gate, lats, lons, min_area=1.0)
    # Strong pixels must all be faint pixels (strong is subset)
    assert (strong & ~faint).sum() == 0, "Strong pixels found outside faint union"


def test_frac_strong_between_0_and_1():
    da, lats, lons = _make_field(n_cells=5, cell_strength=25.0)
    ny, nx = da.shape
    gate = np.ones((ny, nx), dtype=bool)
    _, _, _, props = detect_objects_id_pro(da, gate, lats, lons, min_area=1.0)
    for p in props:
        assert 0.0 <= p.frac_strong <= 1.0, f"frac_strong out of [0,1]: {p.frac_strong}"
