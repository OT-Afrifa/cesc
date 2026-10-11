# Ported from id_pro3.py (link_by_advection). Experimental.
"""Advection-aware nearest-neighbour linking between two timesteps.

This is not tracking. Hourly output cannot follow 20-40 minute cells
advecting at 15-30 m/s; the expected displacement over an hour is
several cell diameters, and objects at the next hour are usually new
cells. The linker estimates, for each object at t1, the position its
centroid would advect to and takes the nearest object at t2 within
max_link_km of that point. There is no conflict resolution; one-to-one
is not enforced. Use it for qualitative continuity only.
"""
from math import sqrt
from dataclasses import dataclass


@dataclass
class TrackLink:
    """A temporal link between an object at t1 and one at t2."""
    prev_id: int
    curr_id: int
    distance: float


def link_by_advection(prev_props, curr_props, ubar, vbar, dt_seconds,
                      max_link_km=30.0):
    """Link objects between two consecutive timesteps.

    Parameters
    ----------
    prev_props, curr_props : lists of ObjectProps
        centroid_yx must be in grid-km coordinates.
    ubar, vbar : float
        Mean PI-layer wind components (m/s).
    dt_seconds : float
        Time between frames (s).
    max_link_km : float
        Maximum linking radius (km).

    Returns
    -------
    links : list of TrackLink
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
            d = sqrt(dx * dx + dy * dy)
            if d < best_d:
                best_id, best_d = c.id, d
        if best_id is not None and best_d <= max_link_km:
            links.append(TrackLink(prev_id=p.id, curr_id=best_id,
                                   distance=best_d))
    return links
