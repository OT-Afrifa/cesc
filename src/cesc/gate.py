# Stage 3 ported from the watershed and gating blocks of
# convective_id_pipeline in the research notebook. Numerics preserved.
"""Stage 3: watershed refinement, the environmental gate, and the
strong/faint split.

The faint detection union is split into objects along watershed lines
of a Gaussian-smoothed enhancement field Enh = Z - Z_bg, seeded with
local maxima above the field's 85th percentile. Z_bg is the bg_radius
uniform-filter background; this is its only role besides the quicklook
display. It never enters the Stage 2 thresholds.

An object survives the gate when its overlap with the support mask is
at least min_support_px pixels and at least min_support_frac of its
area. Kept objects are relabeled compactly and split into strong
(frac_strong >= frac_strong_cutoff) and faint label grids sharing the
same numbering.
"""
import numpy as np
from scipy.ndimage import (uniform_filter, distance_transform_edt,
                           find_objects)
from skimage import filters, feature, morphology, segmentation

from .detect import _compute_moments_and_axes, _classify


def nan_uniform_filter(a, size):
    """Uniform filter that ignores NaN pixels (normalized convolution)."""
    a = np.asarray(a, dtype=np.float32)
    valid = np.isfinite(a).astype(np.float32)
    a0 = np.nan_to_num(a, nan=0.0).astype(np.float32)
    num = uniform_filter(a0, size=size, mode="nearest")
    den = uniform_filter(valid, size=size, mode="nearest")
    out = num / np.maximum(den, 1e-9)
    out[den < 1e-6] = np.nan
    return out


def background_and_enhancement(Z, bg_radius_km, grid_km):
    """Display background Z_bg and enhancement Enh = Z - Z_bg.

    The window is 2 * bg_radius rounded to an odd pixel count. Used for
    the quicklook display and watershed seeding only.
    """
    win_px = int(np.round((2 * bg_radius_km) / grid_km)) | 1
    Zbg = nan_uniform_filter(Z, size=win_px)
    return Zbg, Z - Zbg


def watershed_split(faint_det, Enh, grid_km, min_area_km2):
    """Split touching detections along watershed lines of smoothed Enh.

    Seeds are local maxima of the Gaussian-smoothed enhancement above
    its 85th percentile within the detection mask, separated by at
    least max(8 px, min_area_km2 / grid_km). If no seeds are found the
    plain connected components are returned.

    Returns
    -------
    labels_ws : (ny, nx) int32
    """
    sm = filters.gaussian(
        np.where(np.isfinite(Enh), Enh, np.nanmin(Enh) - 10.0),
        sigma=1.0, preserve_range=True)
    valid_seeds = faint_det & np.isfinite(sm)
    pct85 = np.nanpercentile(sm[valid_seeds], 85) if valid_seeds.any() else 0.0
    seed_mask = (sm > pct85) & faint_det
    min_dist_px = max(8, int(round(min_area_km2 / grid_km)))
    coords = feature.peak_local_max(sm, labels=seed_mask.astype(np.uint8),
                                    min_distance=min_dist_px,
                                    exclude_border=False)
    markers = np.zeros_like(faint_det, dtype=np.int32)
    for idx2, (yy, xx) in enumerate(coords, start=1):
        markers[yy, xx] = idx2
    if markers.max() == 0 and faint_det.any():
        markers = morphology.label(faint_det, connectivity=1)
    return segmentation.watershed(-sm, markers=markers, mask=faint_det)


def gate_and_classify(labels_ws, faint_det, strong_det, ece_core, corridor,
                      support_mask, lats2d, lons2d, dx_km, dy_km, params,
                      aux_fields=None):
    """The environmental gate and the per-object table.

    Parameters
    ----------
    labels_ws : (ny, nx) int32
        Watershed labels from Stage 2/3.
    faint_det, strong_det : (ny, nx) bool
        Detection unions.
    ece_core, corridor, support_mask : (ny, nx) bool
        Stage 1 masks; support = core OR corridor.
    lats2d, lons2d : (ny, nx) float
    dx_km, dy_km : float
    params : Params
    aux_fields : dict of name -> (ny, nx) array, optional
        Extra fields whose object-mean is added as "<name>_mean".

    Returns
    -------
    labels_env, labels_faint, labels_strong : (ny, nx) int32
        Kept objects, compact 1..N; the faint/strong grids share the
        labels_env numbering.
    objs : list of dict
        One row per watershed object (kept or not; kept_env records the
        gate decision), matching the research object table.
    n_dropped_gate : int
    """
    aux_fields = aux_fields or {}
    ny, nx = labels_ws.shape
    ece_dist_km = distance_transform_edt(~ece_core) * (0.5 * (dx_km + dy_km))
    pix_area_km2 = dx_km * dy_km
    slices_ws = find_objects(labels_ws)
    objs = []
    kept_ids = []
    lats = np.asarray(lats2d)
    lons = np.asarray(lons2d)

    for lab, slc in enumerate(slices_ws, start=1):
        if slc is None:
            continue
        obj = labels_ws[slc] == lab
        area_px = int(obj.sum())
        if area_px == 0:
            continue
        yy, xx = np.nonzero(obj)
        yy = yy + slc[0].start
        xx = xx + slc[1].start

        ov_ece = int((ece_core[slc] & obj).sum())
        ov_corr = int((corridor[slc] & obj).sum())
        ov_sup = int((support_mask[slc] & obj).sum())
        frac_sup = ov_sup / max(area_px, 1)
        keep = (ov_sup >= params.min_support_px) and \
               (frac_sup >= params.min_support_frac)
        if keep:
            kept_ids.append(lab)

        cy = int(round(yy.mean()))
        cx = int(round(xx.mean()))
        L, W, asp, ang = np.nan, np.nan, np.nan, np.nan
        try:
            _, (L, W, asp, ang) = _compute_moments_and_axes(yy, xx, dx_km, dy_km)
        except Exception:
            pass
        cls = _classify(L, W, asp, params.length_cut_km, params.aspect_cut)
        size_km2 = area_px * pix_area_km2
        clat = float(np.nanmean(lats[yy, xx]))
        clon = float(np.nanmean(lons[yy, xx]))
        n_strong = int(strong_det[yy, xx].sum())
        n_faint = int(faint_det[yy, xx].sum())
        frac_str = n_strong / max(area_px, 1)

        row = dict(
            id=lab, kept_env=bool(keep), cls=cls,
            centroid_lat=clat, centroid_lon=clon,
            centroid_yx=(cy, cx),
            dist_to_ece_core_km=float(ece_dist_km[cy, cx]),
            major_km=L, minor_km=W, aspect=asp, orientation_deg=ang,
            area_px=area_px, size_km2=size_km2,
            frac_within_support=frac_sup,
            overlap_support_px=ov_sup,
            frac_within_ece=ov_ece / max(area_px, 1),
            frac_downstream=ov_corr / max(area_px, 1),
            n_faint_px=n_faint, n_strong_px=n_strong,
            frac_strong=frac_str,
            intensity_cls=("strong" if frac_str >= params.frac_strong_cutoff
                           else "faint"),
        )
        for name, arr in aux_fields.items():
            row[name + "_mean"] = (float(np.nanmean(np.asarray(arr)[yy, xx]))
                                   if arr is not None else np.nan)
        objs.append(row)

    if kept_ids:
        remap = np.zeros(int(labels_ws.max()) + 1, dtype=np.int32)
        for nid, old in enumerate(kept_ids, start=1):
            remap[old] = nid
        labels_env = remap[labels_ws]
    else:
        labels_env = np.zeros_like(labels_ws)

    strong_ids = set(o["id"] for o in objs
                     if o["kept_env"] and
                     o["frac_strong"] >= params.frac_strong_cutoff)
    old2new = ({old: nid for nid, old in enumerate(kept_ids, start=1)}
               if kept_ids else {})
    labels_strong = np.zeros_like(labels_env)
    labels_faint = np.zeros_like(labels_env)
    for o in objs:
        if not o["kept_env"]:
            continue
        new_id = old2new.get(o["id"], 0)
        if new_id == 0:
            continue
        if o["id"] in strong_ids:
            labels_strong[labels_env == new_id] = new_id
        else:
            labels_faint[labels_env == new_id] = new_id

    n_dropped_gate = len(objs) - len(kept_ids)
    return (labels_env, labels_faint, labels_strong, objs, n_dropped_gate,
            ece_dist_km)


def wind_aligned_length_width(mask_bool, cy, cx, u_bar, v_bar,
                              dx_km, dy_km, slc):
    """Length and width (km) of a footprint aligned with the wind.

    Falls back to PCA axes when the wind is absent or calm.
    """
    yy, xx = np.nonzero(mask_bool)
    if yy.size == 0:
        return np.nan, np.nan
    try:
        if u_bar is None or v_bar is None:
            raise ValueError
        cy_f = cy + slc[0].start
        cx_f = cx + slc[1].start
        u0 = float(u_bar[cy_f, cx_f])
        v0 = float(v_bar[cy_f, cx_f])
        if not np.isfinite(u0) or not np.isfinite(v0) or (u0 == 0 and v0 == 0):
            raise ValueError
        sx, sy = u0, -v0
        nrm = (sx * sx + sy * sy) ** 0.5
        sx /= nrm
        sy /= nrm
        cxw, cyw = -sy, sx
        X = (xx - cx) * dx_km
        Y = (yy - cy) * dy_km
        return (float(np.ptp(X * sx + Y * sy)), float(np.ptp(X * cxw + Y * cyw)))
    except Exception:
        x = (xx - cx) * dx_km
        y = (yy - cy) * dy_km
        x0 = x - x.mean()
        y0 = y - y.mean()
        sxx = np.mean(x0 * x0)
        syy = np.mean(y0 * y0)
        sxy = np.mean(x0 * y0)
        tr = sxx + syy
        det = sxx * syy - sxy * sxy
        lam1 = tr / 2 + max(tr * tr / 4 - det, 0) ** 0.5
        lam2 = tr / 2 - max(tr * tr / 4 - det, 0) ** 0.5
        return float(2 * max(lam1, 0) ** 0.5), float(2 * max(lam2, 0) ** 0.5)


def add_downstream_metrics(labels_env, ece_core, corridor, objs,
                           u_bar, v_bar, dx_km, dy_km):
    """Enrich kept objects with ECE/corridor overlap and aligned axes.

    Adds area_ece_px, area_ds_px, area_sup_px, frac_within_ece,
    frac_downstream, frac_support, L_ds_km, W_ds_km. Mirrors
    add_downstream_metrics_to_objects of the research code, keyed on
    the kept-object (labels_env) numbering.
    """
    labels = np.asarray(labels_env, dtype=np.int32)
    ece = np.asarray(ece_core, dtype=bool)
    corr = np.asarray(corridor, dtype=bool)
    ds = corr & (~ece)
    sup = corr | ece

    u0 = np.asarray(u_bar) if u_bar is not None else None
    v0 = np.asarray(v_bar) if v_bar is not None else None

    maxlab = int(labels.max())
    if maxlab == 0:
        return objs

    # map old watershed id -> new compact id for kept objects
    kept = [o for o in objs if o["kept_env"]]
    old_ids_in_order = [o["id"] for o in kept]
    old2new = {old: nid for nid, old in enumerate(old_ids_in_order, start=1)}

    slices = []
    for lab in range(1, maxlab + 1):
        ys, xs = np.where(labels == lab)
        slices.append(None if ys.size == 0 else
                      (slice(ys.min(), ys.max() + 1),
                       slice(xs.min(), xs.max() + 1)))

    for o in kept:
        new_id = old2new.get(int(o["id"]), 0)
        if new_id < 1 or new_id > maxlab:
            continue
        slc = slices[new_id - 1]
        if slc is None:
            continue
        obj = labels[slc] == new_id
        area_px = int(obj.sum())
        if area_px == 0:
            continue

        a_ece = int((ece[slc] & obj).sum())
        a_ds = int((ds[slc] & obj).sum())
        a_sup = int((sup[slc] & obj).sum())
        o.update(area_ece_px=a_ece, area_ds_px=a_ds, area_sup_px=a_sup,
                 frac_within_ece=a_ece / area_px,
                 frac_downstream=a_ds / area_px,
                 frac_support=a_sup / area_px)

        sup_mask = sup[slc] & obj
        cy_full, cx_full = int(o["centroid_yx"][0]), int(o["centroid_yx"][1])
        cy_loc = cy_full - slc[0].start
        cx_loc = cx_full - slc[1].start

        L_km, W_km = np.nan, np.nan
        if sup_mask.any():
            L_km, W_km = wind_aligned_length_width(
                sup_mask, cy_loc, cx_loc, u0, v0, dx_km, dy_km, slc)
        o["L_ds_km"] = float(L_km)
        o["W_ds_km"] = float(W_km)

    return objs
