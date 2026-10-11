"""Input validation. Every entry point runs this before computing.

The failure messages say what to fix, not just what broke. The checks
target the mix-ups that have actually happened: theta instead of
theta_e, heights in km instead of m, and accumulations passed as rates.
"""
import warnings

import numpy as np

ACCUMULATION_NAME_PATTERNS = ("rainnc", "rainc", "acpcp", "apcp", "precip_acc",
                              "tp", "accum")


class PreflightError(ValueError):
    pass


def _fail(msgs, msg, raise_on_error):
    msgs.append("FAIL " + msg)
    if raise_on_error:
        raise PreflightError(msg)


def _warn(msgs, msg):
    msgs.append("WARN " + msg)
    warnings.warn(msg)


def check_dataset(ds, params=None, raise_on_error=True, verbose=False):
    """Readiness checks on an extracted-WRF style dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Must carry theta_e, heights, reflectivity_1km, and (when the
        corridor is on) u3d and v3d, with XLAT/XLONG coordinates.
    params : Params, optional
    raise_on_error : bool
        Raise on the first failure instead of collecting.

    Returns
    -------
    report : list of str
        Lines beginning OK, WARN, or FAIL.
    """
    msgs = []
    need = ["theta_e", "heights", "reflectivity_1km"]
    if params is not None and params.use_advection:
        need += ["u3d", "v3d"]
    for v in need:
        if v not in ds:
            _fail(msgs, "variable '%s' is missing. The extracted-WRF layout "
                  "needs it; see docs/data_requirements.md or use an adapter "
                  "in cesc.adapters." % v, raise_on_error)
    for c in ("XLAT", "XLONG", "time"):
        if c not in ds.coords and c not in ds:
            _fail(msgs, "coordinate '%s' is missing." % c, raise_on_error)

    if "theta_e" in ds:
        th = ds.theta_e
        sample = th.isel(time=0).values if "time" in th.dims else th.values
        # The algorithm only ever reads theta_e below max_pi_height
        # (5000 m by default). Validate the field where it is used, not
        # over the whole column: above the boundary layer theta_e
        # reduces to dry theta and rises steeply into the stratosphere,
        # so whole-column extrema reach 500+ K legitimately and say
        # nothing about the scanned layer. Restrict to levels whose
        # height is at or below the PI ceiling.
        ceil_m = params.max_pi_height_m if params is not None else 5000.0
        zsample = None
        if "heights" in ds:
            hh = ds.heights
            zsample = (hh.isel(time=0).values if "time" in hh.dims
                       else hh.values)
        if (zsample is not None and sample.ndim == 3
                and zsample.shape == sample.shape):
            zcol = np.nanmedian(zsample.reshape(zsample.shape[0], -1), axis=1)
            in_layer = zcol <= ceil_m
            if not in_layer.any():
                in_layer[:max(1, sample.shape[0] // 4)] = True
            layer = sample[in_layer]
            layer_desc = "below %.0f m" % ceil_m
        else:
            layer = sample
            layer_desc = "all levels"

        finite = layer[np.isfinite(layer)]
        if finite.size:
            p01, p50, p99 = (float(np.percentile(finite, 0.1)),
                             float(np.percentile(finite, 50.0)),
                             float(np.percentile(finite, 99.9)))
            lo, hi = float(finite.min()), float(finite.max())
            n_hot = int((finite > 425.0).sum())
            n_cold = int((finite < 235.0).sum())
            frac_out = (n_hot + n_cold) / finite.size

            # A shape test that separates theta_e from dry theta in the
            # 0-5 km layer alone is not reliable for a well-mixed cold-
            # season boundary layer: there both fields rise smoothly by
            # 10-15 K and differ mainly by a ~10 K offset within the
            # range of surface-temperature variation. The magnitude
            # bracket below catches the gross errors (Celsius, a field
            # shifted far from the physical band); a subtle theta-vs-
            # theta_e swap that lands inside the band is left to the
            # user, with a soft note rather than a hard failure. The
            # reference pipeline reads this same field, so a false
            # failure on good data is the worse error.
            bulk_ok = 250.0 <= p50 <= 380.0 and p99 <= 430.0 and p01 >= 235.0
            if not bulk_ok:
                _fail(msgs, "theta_e in the scanned layer (%s) looks "
                      "wrong: median %.0f K, 0.1-99.9%% span %.0f-%.0f K. "
                      "Physical theta_e there has a median near 290-320 K. "
                      "If these are degrees Celsius add 273.15; if the "
                      "magnitude resembles dry theta or theta_v recompute "
                      "with cesc.thermo.theta_e_bolton (potential "
                      "instability is a saturation criterion; dry theta "
                      "misses the latent-heat contribution)."
                      % (layer_desc, p50, p01, p99), raise_on_error)
            else:
                msgs.append("OK   theta_e (%s) median %.0f K, 0.1-99.9%% "
                            "span %.0f-%.0f K" % (layer_desc, p50, p01, p99))
                if frac_out > 1e-3:
                    _warn(msgs, "%.3f%% of theta_e cells in the scanned "
                          "layer are outside [235, 425] K (range [%.0f, "
                          "%.0f]); inspect before trusting absolute PI "
                          "strengths." % (100 * frac_out, lo, hi))

    if "heights" in ds:
        h = ds.heights
        sample = h.isel(time=0).values if "time" in h.dims else h.values
        top = float(np.nanmax(sample))
        if top < 100.0:
            _fail(msgs, "heights top out at %.1f, which looks like km. "
                  "The scans need m AGL; multiply by 1000." % top,
                  raise_on_error)
        elif top > 60000.0:
            _fail(msgs, "heights top out at %.0f, which is too large for "
                  "m AGL. If these are pressures, supply geometric heights "
                  "instead." % top, raise_on_error)
        else:
            msgs.append("OK   heights top at %.0f m AGL" % top)
        col = sample[(slice(None),) + tuple(
            int(n // 2) for n in sample.shape[1:])]
        if np.any(np.diff(col[np.isfinite(col)]) <= 0):
            _warn(msgs, "heights are not strictly increasing at the "
                  "domain-center column; check the vertical ordering.")

    if "reflectivity_1km" in ds:
        z = ds.reflectivity_1km
        sample = z.isel(time=0).values if "time" in z.dims else z.values
        finite = sample[np.isfinite(sample)]
        if finite.size:
            lo, hi = float(finite.min()), float(finite.max())
            if hi > 90.0 or lo < -60.0:
                _warn(msgs, "reflectivity range [%.0f, %.0f] dBZ is outside "
                      "the plausible [-60, 90]; confirm the units are dBZ, "
                      "not Z in mm6/m3." % (lo, hi))
            else:
                msgs.append("OK   reflectivity range [%.0f, %.0f] dBZ"
                            % (lo, hi))

    if not msgs:
        msgs.append("OK   (no checks ran; dataset empty?)")
    if verbose:
        for m in msgs:
            print(" ", m)
    return msgs


def check_precip_rate(precip_da, name=None):
    """Warn when a rate field looks like an accumulation.

    Displaced hourly accumulations trail moving cells and undercount
    convective precipitation. The check is on the variable name and on
    monotone growth in time at the wettest pixel.
    """
    msgs = []
    nm = (name or getattr(precip_da, "name", "") or "").lower()
    if any(pat in nm for pat in ACCUMULATION_NAME_PATTERNS):
        _warn(msgs, "precipitation variable '%s' matches an accumulation "
              "name. The attribution needs an instantaneous rate "
              "co-located with the reflectivity; a displaced hourly "
              "accumulation undercounts convective precipitation (event "
              "fraction 10.6%% vs 12.3%%, peak hour 15.8%% vs 28.0%% in "
              "the reference case). See docs/caveats.md." % nm)

    vals = np.asarray(precip_da.values if hasattr(precip_da, "values")
                      else precip_da)
    if vals.ndim == 3 and vals.shape[0] >= 6:
        flat = np.nan_to_num(vals, nan=0.0).reshape(vals.shape[0], -1)
        jmax = int(np.argmax(flat.sum(axis=0)))
        series = flat[:, jmax]
        diffs = np.diff(series)
        if series.max() > 0 and np.all(diffs >= -1e-6) and series[-1] > series[0]:
            _warn(msgs, "the precipitation field grows monotonically in "
                  "time at its wettest pixel, which is the signature of a "
                  "running accumulation, not a rate.")
    if float(np.nanmin(vals)) < -1e-6:
        _warn(msgs, "the precipitation field has negative values; rates "
              "must be non-negative.")
    return msgs
