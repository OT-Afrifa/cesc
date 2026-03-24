# cesc/id_pro.py
"""
Multi-scale convective object identification module (ID-PRO adaptation).

This module implements the reflectivity-based detection step of the CESC
algorithm described in Afrifa et al. (Part 1, submitted 2026).  The core
approach is adapted from the ID-PRO algorithm of Yeh and Colle (2025):

    Yeh, P., and B. A. Colle, 2025: A Comparison of Approaches to Objectively
    Identify Precipitation Structures within the Comma Head of Mid-Latitude
    Cyclones.  J. Atmos. Oceanic Technol., 42, 463-477.

For each of several box sizes (default 20, 40, 80 km), a per-pixel local
background mean and standard deviation are computed using integral images.
A pixel is flagged as convective if its reflectivity exceeds the local
background by at least max(min_diff, k_sigma * local_sigma), evaluated
separately for "faint" and "strong" thresholds.  The union of detections
across all scales is taken, so both small sharp-peaked cells (caught at 20
km scale) and larger organised clusters (caught at 80 km scale) are
identified.

The faint and strong detection masks are kept separate through the entire
pipeline so downstream code can classify each object by the fraction of its
pixels that exceeded the strong threshold (ObjectProps.frac_strong).

References
----------
Afrifa, F. O., and co-authors, 2026: Climatology of cold-season emergent
    convection in frontal systems and its impact on orographic precipitation.
    Part 1: Detection algorithm.  Mon. Wea. Rev., in review.

Afrifa, F. O., B. Geerts, L. Xue, S. Chen, C. Hohman, C. Grasmick, and
    T. Zaremba, 2025: A case study of cold-season emergent orographic
    convection and its impact on precipitation. Part I: Mesoscale analysis.
    Mon. Wea. Rev., 153, 2229-2250.
"""

import numpy as np
from numba import njit, prange
from math import sqrt, degrees
from dataclasses import dataclass

from cesc.distance_calculator import estimate_dx_dy_km


# ---------------------------------------------------------------------------
# Integral-image helpers
# ---------------------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _integral_image_nan_sq(a):
    """
    Build prefix-sum tables for mean/std computation over any rectangular box.

    NaN values are excluded from all accumulators so local statistics remain
    valid even where reflectivity data are missing.

    Parameters
    ----------
    a : (ny, nx) float64

    Returns
    -------
    isum : (ny+1, nx+1) float64 -- prefix sum of finite values
    isq  : (ny+1, nx+1) float64 -- prefix sum of squared finite values
    icnt : (ny+1, nx+1) float64 -- count of finite values
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
            if v == v:          # NaN != NaN
                s_row += v
                q_row += v * v
                c_row += 1.0
            isum[j+1, i+1] = isum[j, i+1] + s_row
            isq[j+1, i+1] = isq[j, i+1] + q_row
            icnt[j+1, i+1] = icnt[j, i+1] + c_row
    return isum, isq, icnt


@njit(cache=True, fastmath=True, parallel=True)
def _box_mean_std_from_integral(isum, isq, icnt, rj, ri):
    """
    Compute per-pixel local mean and standard deviation from integral images.

    The box centred on (j, i) has half-sizes rj (rows) and ri (columns).
    Edges are clamped so every pixel uses the largest valid box that fits
    within the domain.

    Parameters
    ----------
    isum, isq, icnt : integral image triplet from _integral_image_nan_sq
    rj, ri          : int -- half-sizes in pixels

    Returns
    -------
    mu  : (ny, nx) float64 — local mean   (NaN where < 2 finite neighbours)
    sig : (ny, nx) float64 — local std    (NaN where < 2 finite neighbours)
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
            S = isum[j1+1, i1+1] - isum[j0, i1+1] - isum[j1+1, i0] + isum[j0, i0]
            Q = isq[j1+1, i1+1] - isq[j0, i1+1] - isq[j1+1, i0] + isq[j0, i0]
            C = icnt[j1+1, i1+1] - icnt[j0, i1+1] - icnt[j1+1, i0] + icnt[j0, i0]
            if C > 1.5:
                m = S / C
                v = max(Q / C - m * m, 0.0)
                mu[j, i] = m
                sig[j, i] = sqrt(v)
            else:
                mu[j, i] = np.nan
                sig[j, i] = np.nan
    return mu, sig


# ---------------------------------------------------------------------------
# Connected-component labelling (8-connectivity)
# ---------------------------------------------------------------------------

@njit(cache=True)
def _cc_label_8(mask):
    """
    Label connected components of a 2D boolean mask using 8-connectivity.

    Implements union-find with path halving for efficient root traversal.
    The second pass uses a full while-loop to follow the parent chain to its
    root, avoiding the two-hop truncation that can split a single component
    into multiple labels when the union-find tree is deep.

    Parameters
    ----------
    mask : (ny, nx) bool

    Returns
    -------
    labels   : (ny, nx) int32 -- 1..N labels, 0 = background
    n_labels : int
    """
    ny, nx = mask.shape
    labels = np.zeros((ny, nx), dtype=np.int32)
    parent = np.arange(ny * nx + 1, dtype=np.int32)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]   # path halving
            a = parent[a]
        return a

    def union(a, b):
        ra = find(a)
        rb = find(b)
        if ra != rb:
            parent[rb] = ra

    # First pass: provisional labels + union-find links
    next_label = 1
    for j in range(ny):
        for i in range(nx):
            if not mask[j, i]:
                continue
            # CHANGE 2: no dead placeholder loop here (removed for dj in (-1,0,-1))
            pos = []
            for jj, ii in ((j-1, i-1), (j-1, i), (j-1, i+1), (j, i-1)):
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

    # CHANGE 1: full root traversal (old code did only 2 hops, which left deep
    # union-find trees partially unflattened and split single components)
    for j in range(ny):
        for i in range(nx):
            lab = labels[j, i]
            if lab > 0:
                root = lab
                while parent[root] != root:
                    root = parent[root]
                labels[j, i] = root

    # Compact to 1..N
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


# ---------------------------------------------------------------------------
# Object geometry
# ---------------------------------------------------------------------------

@dataclass
class ObjectProps:
    """
    Geometric and statistical properties of one identified convective object.

    Attributes
    ----------
    id              : compact integer label (1..N)
    area            : planform area (km^2)
    npix            : pixel count
    centroid_yx     : (y_km, x_km) centroid in grid-space kilometres
    centroid_lonlat : (lon_deg, lat_deg)
    major           : major principal-axis length (km)
    minor           : minor principal-axis length (km)
    aspect          : minor/major ratio  (0 = line, 1 = circle)
    orientation_deg : angle of major axis from east, counter-clockwise (degrees)
    peak_value      : maximum reflectivity in the object (dBZ)
    median_value    : median reflectivity (dBZ)
    frac_strong     : fraction of object pixels that exceeded the strong threshold
                      at any box scale.  Use with pipeline parameter
                      frac_strong_cutoff to classify objects as faint or strong.
    cls             : 'cell' | 'band' | 'complex'
    """
    id             : int
    area           : float
    npix           : int
    centroid_yx    : tuple
    centroid_lonlat: tuple
    major          : float
    minor          : float
    aspect         : float
    orientation_deg: float
    peak_value     : float
    median_value   : float
    frac_strong    : float
    cls            : str


def _compute_moments_and_axes(idx_j, idx_i, dx_km, dy_km):
    """
    Compute centroid and principal axes via second central moments.

    Parameters
    ----------
    idx_j, idx_i : 1D int arrays -- row and column pixel indices
    dx_km, dy_km : grid spacings (km)

    Returns
    -------
    centroid_yx : (y_km, x_km)
    axes        : (major_km, minor_km, aspect_ratio, orientation_deg)
    """
    if idx_j.size == 0:
        return (np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan)
    y = idx_j.astype(np.float64) * dy_km
    x = idx_i.astype(np.float64) * dx_km
    xbar = np.mean(x)
    ybar = np.mean(y)
    x0 = x - xbar
    y0 = y - ybar
    sxx = np.mean(x0*x0)
    syy = np.mean(y0*y0)
    sxy = np.mean(x0*y0)
    tr = sxx + syy
    det = sxx*syy - sxy*sxy
    lam1 = tr/2 + sqrt(max(tr*tr/4 - det, 0.0))
    lam2 = tr/2 - sqrt(max(tr*tr/4 - det, 0.0))
    L = 2.0*sqrt(max(lam1, 0.0))
    W = 2.0*sqrt(max(lam2, 0.0))
    ang_deg = (degrees(0.5*np.arctan2(2*sxy, sxx-syy+1e-12)) + 360) % 360
    return (ybar, xbar), (L, W, W/L if L > 1e-6 else np.nan, ang_deg)


def _classify(L, W, aspect, length_cut=100.0, aspect_cut=0.5):
    """
    Classify an object by shape.

    'band'    — major axis >= length_cut km AND aspect <= aspect_cut
    'cell'    — 10 km <= major axis < length_cut AND aspect > aspect_cut
    'complex' — everything else (irregular shape or degenerate geometry)
    """
    if not np.isfinite(L) or not np.isfinite(aspect):
        return "complex"
    if L >= length_cut and aspect <= aspect_cut:
        return "band"
    if 10.0 <= L < length_cut and aspect > aspect_cut:
        return "cell"
    return "complex"


# ---------------------------------------------------------------------------
# Core detection function
# ---------------------------------------------------------------------------

def detect_objects_id_pro(
    field_da,
    gate_mask,
    lats2d,
    lons2d,
    box_sizes=(20.0, 40.0, 80.0),
    faint_min_diff=2.0,
    faint_k_sigma=0.30,
    strong_min_diff=5.0,
    strong_k_sigma=0.30,
    min_area=6.0,
    length_cut=100.0,
    aspect_cut=0.5,
):
    """
    Identify locally enhanced convective objects using multi-scale adaptive
    background thresholding.

    For each box width in box_sizes the function computes a per-pixel local
    background mean (mu) and standard deviation (sig).  Detection fires when:

        Z - mu  >=  max(min_diff, k_sigma * sig)

    evaluated with the faint and strong parameter pairs separately.  The faint
    and strong detections are accumulated into faint_union and strong_union via
    logical OR across all three scales.  Because strong_min_diff > faint_min_diff
    (with equal k_sigma), the strong pixels form a subset of the faint pixels.

    Note on box_sizes
    -----------------
    This parameter is a structural design choice (multi-scale union to reduce
    sensitivity to any single background scale).
    The three default widths mirror those used by Yeh and Colle (2025) and
    are not listed in the algorithm sensitivity table.

    Note on gate_mask
    -----------------
    This should be the full precipitating domain (Z >= -2 dBZ).  ECE/corridor
    gating is applied downstream in Stage 3 of the pipeline via overlap
    fraction tests.  Passing the ECE region here would prevent detection of
    downstream cells before the corridor is computed.

    Parameters
    ----------
    field_da        : 2D xr.DataArray -- reflectivity at 1 km AGL (dBZ)
    gate_mask       : 2D bool -- detection domain (see note above)
    lats2d, lons2d  : 2D float -- geographic coordinates
    box_sizes       : tuple of box full-widths (km)
    faint_min_diff  : dBZ floor for faint detection
    faint_k_sigma   : sigma multiplier for faint threshold
    strong_min_diff : dBZ floor for strong detection
    strong_k_sigma  : sigma multiplier for strong threshold
    min_area        : minimum object area to retain (km^2)
    length_cut      : major-axis length threshold for band classification (km)
    aspect_cut      : aspect-ratio threshold separating bands from cells

    Returns
    -------
    faint_union  : (ny, nx) bool -- faint detection pixels
    strong_union : (ny, nx) bool -- strong detection pixels
    labels_f     : (ny, nx) int32 -- compact object labels 1..N
    props        : list[ObjectProps]
    """
    assert field_da.ndim == 2
    ny, nx = field_da.shape
    data = np.asarray(field_da.values, dtype=np.float64)
    gate = gate_mask.astype(np.bool_)

    dx_km, dy_km = estimate_dx_dy_km(lats2d, lons2d)
    pix_km2 = dx_km * dy_km

    # Build integral images once for all box sizes
    isum, isq, icnt = _integral_image_nan_sq(data)

    faint_union = np.zeros((ny, nx), dtype=np.bool_)
    strong_union = np.zeros((ny, nx), dtype=np.bool_)

    for size_km in box_sizes:
        r_pix = max(1, int(0.5 * size_km / ((dx_km + dy_km) * 0.5)))
        mu, sig = _box_mean_std_from_integral(isum, isq, icnt, r_pix, r_pix)
        diff = data - mu

        thr_faint = np.maximum(faint_min_diff,  faint_k_sigma * sig)
        thr_strong = np.maximum(strong_min_diff, strong_k_sigma * sig)

        valid = gate & np.isfinite(data)
        faint_union |= (diff >= thr_faint) & valid
        strong_union |= (diff >= thr_strong) & valid

    # Label connected components of the faint union
    labels, nlab = _cc_label_8(faint_union)
    if nlab == 0:
        return faint_union, strong_union, np.zeros((ny, nx), dtype=np.int32), []

    # Area filter
    keep = np.zeros(nlab + 1, dtype=np.bool_)
    for lab in range(1, nlab+1):
        if (labels == lab).sum() * pix_km2 >= min_area:
            keep[lab] = True

    labmap = {}
    newlab = 0
    for lab in range(1, nlab+1):
        if keep[lab]:
            newlab += 1
            labmap[lab] = newlab

    if newlab == 0:
        return faint_union, strong_union, np.zeros((ny, nx), dtype=np.int32), []

    labels_f = np.zeros_like(labels)
    for old, new in labmap.items():
        labels_f[labels == old] = new

    props = []
    for lab in range(1, newlab+1):
        obj = (labels_f == lab)
        jj, ii = np.where(obj)
        area = jj.size * pix_km2
        ckm, (L, W, asp, ang) = _compute_moments_and_axes(jj, ii, dx_km, dy_km)
        peak = float(np.nanmax(data[obj])) if jj.size else np.nan
        med = float(np.nanmedian(data[obj])) if jj.size else np.nan
        clon = float(np.nanmean(lons2d[jj, ii])) if jj.size else np.nan
        clat = float(np.nanmean(lats2d[jj, ii])) if jj.size else np.nan
        # CHANGE 6: fraction of pixels exceeding the strong threshold
        frac_str = int(strong_union[obj].sum()) / max(jj.size, 1)
        props.append(ObjectProps(
            id=lab, area=area, npix=jj.size,
            centroid_yx=(float(ckm[0]), float(ckm[1])),
            centroid_lonlat=(clon, clat),
            major=L, minor=W, aspect=asp, orientation_deg=ang,
            peak_value=peak, median_value=med,
            frac_strong=frac_str,
            cls=_classify(L, W, asp, length_cut=length_cut, aspect_cut=aspect_cut),
        ))

    return faint_union, strong_union, labels_f, props


# ---------------------------------------------------------------------------
# Advection-aware object linker
# ---------------------------------------------------------------------------

@dataclass
class TrackLink:
    """Represents a temporal link between an object at time, t1 and one at the next, t2."""
    prev_id  : int
    curr_id  : int
    distance : float


def link_by_advection(prev_props, curr_props, ubar, vbar,
                      dt_seconds, max_link_km=30.0):
    """
    Nearest-neighbour object linker using expected advection displacement.

    For each object in prev_props the expected position is centroid +
    (ubar, vbar) * dt.  The closest object in curr_props within max_link_km
    of that position is taken as the successor.

    Parameters
    ----------
    prev_props, curr_props : lists of ObjectProps
    ubar, vbar             : PI-layer mean wind (m/s); scalars
    dt_seconds             : seconds between frames
    max_link_km            : maximum linking radius (km)

    Returns
    -------
    list[TrackLink]
    """
    links = []
    if not prev_props or not curr_props:
        return links
    for p in prev_props:
        up = float(ubar)
        vp = float(vbar)
        dx_adv = up * dt_seconds / 1000.0
        dy_adv = vp * dt_seconds / 1000.0
        best_id, best_d = None, 1e9
        for c in curr_props:
            dx = (c.centroid_yx[1] - p.centroid_yx[1]) - dx_adv
            dy = (c.centroid_yx[0] - p.centroid_yx[0]) - dy_adv
            d = sqrt(dx*dx + dy*dy)
            if d < best_d:
                best_id, best_d = c.id, d
        if best_id is not None and best_d <= max_link_km:
            links.append(TrackLink(prev_id=p.id, curr_id=best_id, distance=best_d))

    return links
