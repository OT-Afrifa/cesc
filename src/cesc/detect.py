# Stage 2 ported from the research module id_pro3.py v2.0
# (Francis O.T. Afrifa, 2026). Numerical behavior preserved. The
# research signatures carried pre-Table-1 defaults; here every
# threshold is a required argument and defaults live only in Params.
"""Stage 2: convective-cell detection.

Multi-scale adaptive background thresholding on a 2-D reflectivity
field (the ID-PRO design following Yeh and Colle 2025). For each box
width, a NaN-aware local mean mu and standard deviation sigma are
computed from integral images, and the detection condition is

    Z - mu >= max(min_diff, k_sigma * sigma)

evaluated separately with the faint and strong parameter pairs. A small
box catches sharp isolated cells; a large box catches broader organized
clusters modestly enhanced locally but clearly elevated against the
wider environment. Per-scale detections are OR'd across scales; strong
is a subset of faint.

Detection runs over the full precipitating domain (Z above the
precipitating floor), not the environment region. Environmental gating
happens in Stage 3.

The bg_radius background of the pipeline does not appear here. It
drives the quicklook display and the watershed seeding field only.
"""
from math import sqrt, degrees
from dataclasses import dataclass

import numpy as np

from ._compat import njit, prange
from .distance import estimate_dx_dy_km


@njit(cache=True, fastmath=True)
def _integral_image_nan_sq(a):
    """Prefix sums (value, square, finite count) over a 2-D float64 array.

    NaN pixels contribute nothing. The triplet gives O(1) local mean and
    standard deviation for any rectangular box.

    Parameters
    ----------
    a : (ny, nx) float64

    Returns
    -------
    isum, isq, icnt : (ny+1, nx+1) float64
    """
    ny, nx = a.shape
    isum = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    isq = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    icnt = np.zeros((ny + 1, nx + 1), dtype=np.float64)
    for j in range(ny):
        s_row = 0.0
        q_row = 0.0
        c_row = 0.0
        for i in range(nx):
            v = a[j, i]
            if v == v:
                s_row += v
                q_row += v * v
                c_row += 1.0
            isum[j + 1, i + 1] = isum[j, i + 1] + s_row
            isq[j + 1, i + 1] = isq[j, i + 1] + q_row
            icnt[j + 1, i + 1] = icnt[j, i + 1] + c_row
    return isum, isq, icnt


@njit(cache=True, fastmath=True, parallel=True)
def _box_mean_std_from_integral(isum, isq, icnt, rj, ri):
    """Per-pixel local mean and standard deviation from integral images.

    The box centred on (j, i) has half-sizes rj, ri and clamps at the
    domain boundary, so edge pixels use smaller but valid boxes.

    Returns
    -------
    mu, sig : (ny, nx) float64, NaN where fewer than 2 finite neighbours.
    """
    ny = isum.shape[0] - 1
    nx = isum.shape[1] - 1
    mu = np.empty((ny, nx), dtype=np.float64)
    sig = np.empty((ny, nx), dtype=np.float64)
    for j in prange(ny):
        j0 = max(j - rj, 0)
        j1 = min(j + rj, ny - 1)
        for i in range(nx):
            i0 = max(i - ri, 0)
            i1 = min(i + ri, nx - 1)
            S = isum[j1 + 1, i1 + 1] - isum[j0, i1 + 1] - isum[j1 + 1, i0] + isum[j0, i0]
            Q = isq[j1 + 1, i1 + 1] - isq[j0, i1 + 1] - isq[j1 + 1, i0] + isq[j0, i0]
            C = icnt[j1 + 1, i1 + 1] - icnt[j0, i1 + 1] - icnt[j1 + 1, i0] + icnt[j0, i0]
            if C > 1.5:
                m = S / C
                v = max(Q / C - m * m, 0.0)
                mu[j, i] = m
                sig[j, i] = sqrt(v)
            else:
                mu[j, i] = np.nan
                sig[j, i] = np.nan
    return mu, sig


@njit(cache=True)
def _cc_label_8(mask):
    """8-connectivity component labels via union-find with path halving.

    The second pass performs full root traversal; a two-hop traversal
    leaves deep union-find trees partially unflattened and splits single
    components (a bug fixed in the research code v2.0 and preserved
    fixed here). Labels are compacted to 1..N.

    Parameters
    ----------
    mask : (ny, nx) bool

    Returns
    -------
    labels : (ny, nx) int32, 0 = background.
    n_labels : int
    """
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    parent = np.arange(ny * nx + 1, dtype=np.int32)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    next_label = 1
    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                continue
            pos = []
            for jj, ii in ((j - 1, i - 1), (j - 1, i), (j - 1, i + 1), (j, i - 1)):
                if 0 <= jj < ny and 0 <= ii < nx and mask[jj, ii]:
                    nb = labels[jj, ii]
                    if nb > 0:
                        pos.append(nb)
            if len(pos) == 0:
                labels[j, i] = next_label
                next_label += 1
            else:
                m = pos[0]
                labels[j, i] = m
                for n in pos[1:]:
                    if n != m:
                        union(m, n)

    for j in range(ny):
        for i in range(nx):
            lab = labels[j, i]
            if lab > 0:
                root = lab
                while parent[root] != root:
                    root = parent[root]
                labels[j, i] = root

    seen = {}
    new_id = 0
    for j in range(ny):
        for i in range(nx):
            lab = labels[j, i]
            if lab > 0:
                if lab not in seen:
                    new_id += 1
                    seen[lab] = new_id
                labels[j, i] = seen[lab]

    return labels, new_id


@dataclass
class ObjectProps:
    """Geometric and statistical properties of one detected object.

    frac_strong is the fraction of pixels that exceeded the strong
    threshold at any box scale; classify against
    Params.frac_strong_cutoff.
    """
    id: int
    area: float
    npix: int
    centroid_yx: tuple
    centroid_lonlat: tuple
    major: float
    minor: float
    aspect: float
    orientation_deg: float
    peak_value: float
    median_value: float
    frac_strong: float
    cls: str


def _compute_moments_and_axes(idx_j, idx_i, dx_km, dy_km):
    """Centroid and principal axes of a pixel set via second moments.

    Returns
    -------
    centroid_yx : (y_km, x_km)
    axes : (major_km, minor_km, aspect_ratio, orientation_deg)
    """
    if idx_j.size == 0:
        return (np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan)
    y = idx_j.astype(np.float64) * dy_km
    x = idx_i.astype(np.float64) * dx_km
    xbar = np.mean(x)
    ybar = np.mean(y)
    x0 = x - xbar
    y0 = y - ybar
    sxx = np.mean(x0 * x0)
    syy = np.mean(y0 * y0)
    sxy = np.mean(x0 * y0)
    tr = sxx + syy
    det = sxx * syy - sxy * sxy
    lam1 = tr / 2 + sqrt(max(tr * tr / 4 - det, 0.0))
    lam2 = tr / 2 - sqrt(max(tr * tr / 4 - det, 0.0))
    L = 2.0 * sqrt(max(lam1, 0.0))
    W = 2.0 * sqrt(max(lam2, 0.0))
    ang_deg = (degrees(0.5 * np.arctan2(2 * sxy, sxx - syy + 1e-12)) + 360) % 360
    return (ybar, xbar), (L, W, W / L if L > 1e-6 else np.nan, ang_deg)


def _classify(L, W, aspect, length_cut, aspect_cut):
    """Shape class: 'band', 'cell', or 'complex'.

    band: major axis >= length_cut km and aspect <= aspect_cut.
    cell: 10 km <= major axis < length_cut and aspect > aspect_cut.
    complex: everything else.
    """
    if not np.isfinite(L) or not np.isfinite(aspect):
        return "complex"
    if L >= length_cut and aspect <= aspect_cut:
        return "band"
    if 10.0 <= L < length_cut and aspect > aspect_cut:
        return "cell"
    return "complex"


def multiscale_detect(field, gate_mask, dx_km, dy_km, box_sizes,
                      faint_min_diff, faint_k_sigma,
                      strong_min_diff, strong_k_sigma):
    """Faint and strong detection unions across the box scales.

    Parameters
    ----------
    field : (ny, nx) float
        Reflectivity (dBZ); NaN allowed.
    gate_mask : (ny, nx) bool
        Detection domain, normally the full precipitating area
        (Z >= precip_floor_dbz), not the environment region.
    dx_km, dy_km : float
        Grid spacing (km).
    box_sizes : sequence of float
        Box full widths (km).
    faint_min_diff, faint_k_sigma, strong_min_diff, strong_k_sigma : float
        Threshold parameters.

    Returns
    -------
    faint_union, strong_union : (ny, nx) bool
    """
    data = np.asarray(field, dtype=np.float64)
    ny, nx = data.shape
    gate = np.asarray(gate_mask, dtype=bool)

    isum, isq, icnt = _integral_image_nan_sq(data)

    faint_union = np.zeros((ny, nx), dtype=bool)
    strong_union = np.zeros((ny, nx), dtype=bool)

    for size_km in box_sizes:
        r_pix = max(1, int(0.5 * size_km / ((dx_km + dy_km) * 0.5)))
        mu, sig = _box_mean_std_from_integral(isum, isq, icnt, r_pix, r_pix)
        diff = data - mu

        thr_faint = np.maximum(faint_min_diff, faint_k_sigma * sig)
        thr_strong = np.maximum(strong_min_diff, strong_k_sigma * sig)

        valid = gate & np.isfinite(data)
        faint_union |= (diff >= thr_faint) & valid
        strong_union |= (diff >= thr_strong) & valid

    return faint_union, strong_union


def detect_objects(field_da, gate_mask, lats2d, lons2d, box_sizes,
                   faint_min_diff, faint_k_sigma,
                   strong_min_diff, strong_k_sigma,
                   min_area_km2, length_cut_km, aspect_cut):
    """Detection, labeling, area filter, and object properties.

    This is the ported detect_objects_id_pro. The watershed refinement
    and the environmental gate live in cesc.gate; the labels returned
    here are the plain 8-connected components of the faint union after
    the area filter.

    Returns
    -------
    faint_union, strong_union : (ny, nx) bool
    labels_f : (ny, nx) int32, compact 1..N after area filtering.
    props : list of ObjectProps
    """
    field = np.asarray(field_da.values if hasattr(field_da, "values")
                       else field_da, dtype=np.float64)
    assert field.ndim == 2
    ny, nx = field.shape

    dx_km, dy_km = estimate_dx_dy_km(lats2d, lons2d)
    pix_km2 = dx_km * dy_km

    faint_union, strong_union = multiscale_detect(
        field, gate_mask, dx_km, dy_km, box_sizes,
        faint_min_diff, faint_k_sigma, strong_min_diff, strong_k_sigma)

    labels, nlab = _cc_label_8(faint_union)
    if nlab == 0:
        return faint_union, strong_union, np.zeros((ny, nx), np.int32), []

    keep = np.zeros(nlab + 1, dtype=bool)
    for lab in range(1, nlab + 1):
        if (labels == lab).sum() * pix_km2 >= min_area_km2:
            keep[lab] = True

    labmap = {}
    newlab = 0
    for lab in range(1, nlab + 1):
        if keep[lab]:
            newlab += 1
            labmap[lab] = newlab

    if newlab == 0:
        return faint_union, strong_union, np.zeros((ny, nx), np.int32), []

    labels_f = np.zeros_like(labels)
    for old, new in labmap.items():
        labels_f[labels == old] = new

    lats = np.asarray(lats2d)
    lons = np.asarray(lons2d)
    props = []
    for lab in range(1, newlab + 1):
        obj = labels_f == lab
        jj, ii = np.where(obj)
        area = jj.size * pix_km2
        ckm, (L, W, asp, ang) = _compute_moments_and_axes(jj, ii, dx_km, dy_km)
        peak = float(np.nanmax(field[obj])) if jj.size else np.nan
        med = float(np.nanmedian(field[obj])) if jj.size else np.nan
        clon = float(np.nanmean(lons[jj, ii])) if jj.size else np.nan
        clat = float(np.nanmean(lats[jj, ii])) if jj.size else np.nan
        frac_str = int(strong_union[obj].sum()) / max(jj.size, 1)
        props.append(ObjectProps(
            id=lab, area=area, npix=jj.size,
            centroid_yx=(float(ckm[0]), float(ckm[1])),
            centroid_lonlat=(clon, clat),
            major=L, minor=W, aspect=asp, orientation_deg=ang,
            peak_value=peak, median_value=med,
            frac_strong=frac_str,
            cls=_classify(L, W, asp, length_cut_km, aspect_cut),
        ))

    return faint_union, strong_union, labels_f, props
