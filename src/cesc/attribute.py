"""Precipitation attribution.

Attribution is separate from identification. Objects are defined from
reflectivity; precipitation is attributed afterwards by masking a
surface precipitation-rate field with the object labels. Two
consequences are by design:

Objects legitimately contain rain-free surface pixels. The object
outlines the convective cell aloft, and over dry valleys part of the
falling ice sublimates or evaporates below cloud base. These pixels add
area but no precipitation, so they move the area fractions and not the
precipitation fractions. Do not "fix" this.

The rate must be instantaneous and co-located in time with the
reflectivity. An hourly accumulation trails moving cells by several
cell diameters and undercounts convective precipitation: in the
reference case the event fraction rose from 10.6% to 12.3% and the peak
hour from 15.8% to 28.0% when the displaced hourly field was replaced
with the instantaneous rate. The preflight warns when the supplied
field looks like an accumulation.
"""
import numpy as np


def attribute_precip(labels_env, precip_rate):
    """Mask the rate field with the object labels.

    Parameters
    ----------
    labels_env : (ny, nx) int
        Kept-object labels, 0 = background.
    precip_rate : (ny, nx) float or None
        Instantaneous surface precipitation rate (mm/h).

    Returns
    -------
    conv_rate : (ny, nx) float or None
        The rate where labels_env > 0, NaN elsewhere.
    fraction : float or None
        Domain precipitation fraction attributed to the objects.
    """
    if precip_rate is None:
        return None, None
    rate = np.asarray(precip_rate, dtype=np.float64)
    mask = np.asarray(labels_env) > 0
    conv = np.where(mask, rate, np.nan)
    tot = np.nansum(rate)
    frac = float(np.nansum(np.where(mask, rate, 0.0)) / tot) if tot > 0 else np.nan
    return conv, frac


def per_object_rates(labels_env, precip_rate, objs, pix_area_km2):
    """Add mean and max rate and rain volume per kept object.

    Adds precip_rate_mean_mmhr, precip_rate_max_mmhr, and
    precip_volume_m3hr (rate times pixel area summed over the object)
    to each kept row. Rows of dropped objects are left unchanged.
    """
    if precip_rate is None:
        return objs
    rate = np.asarray(precip_rate, dtype=np.float64)
    labels = np.asarray(labels_env)
    kept = [o for o in objs if o.get("kept_env")]
    old_ids = [o["id"] for o in kept]
    old2new = {old: nid for nid, old in enumerate(old_ids, start=1)}
    for o in kept:
        nid = old2new[o["id"]]
        sel = labels == nid
        if not sel.any():
            continue
        r = rate[sel]
        o["precip_rate_mean_mmhr"] = float(np.nanmean(r))
        o["precip_rate_max_mmhr"] = float(np.nanmax(r))
        # mm/h over km^2: 1 mm/h * 1 km^2 = 1000 m^3/h
        o["precip_volume_m3hr"] = float(np.nansum(
            np.nan_to_num(r, nan=0.0)) * pix_area_km2 * 1000.0)
    return objs
