"""The four-panel quicklook figure.

Imported lazily by identify_time when make_figure is True. Needs
matplotlib; uses cartopy when importable and falls back to plain axes
without it. The panel layout follows the research pipeline: full
precipitating field, the bg_radius display background, enhancement
with the support mask and kept objects, and the faint/strong split.
"""
import numpy as np


def quicklook_figure(Z_2d_da, ter_da, masks, labels_env, labels_faint,
                     labels_strong, params, title=None, level_title=None,
                     extent=None):
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as crs
        import cartopy.feature as cf
        proj = crs.LambertConformal(central_longitude=-117.55,
                                    central_latitude=44)
        subplot_kw = {"projection": proj}
        have_cartopy = True
    except ImportError:
        subplot_kw = {}
        have_cartopy = False

    fig, axes = plt.subplots(2, 2, figsize=(22, 14), subplot_kw=subplot_kw)
    ax = axes.flatten()
    lv = level_title or Z_2d_da.attrs.get("long_name", "1 km AGL")
    titles = [
        "Z [%s], precipitating domain" % lv,
        "Z display background (%.0f km smoothing)" % params.bg_radius,
        "Enhancement | support mask (magenta) | kept CESC (red)",
        "CESC: faint (blue) | strong (red)",
    ]
    lons = np.asarray(Z_2d_da.XLONG)
    lats = np.asarray(Z_2d_da.XLAT)
    kws = dict(levels=np.arange(-10, 45, 5), cmap="jet", extend="max",
               alpha=0.85)
    if have_cartopy:
        kws["transform"] = crs.PlateCarree()
        for a in ax:
            if extent:
                a.set_extent(extent, crs=crs.PlateCarree())
            a.add_feature(cf.BORDERS.with_scale("10m"), linewidth=1.5)
            a.add_feature(cf.STATES.with_scale("10m"), linewidth=1.0,
                          alpha=0.5)
    for a, t in zip(ax, titles):
        a.set_title(t, fontsize=12)
        if ter_da is not None:
            a.pcolormesh(lons, lats, np.asarray(ter_da), cmap="Greys",
                         **({"transform": crs.PlateCarree()}
                            if have_cartopy else {}))

    Z = np.asarray(Z_2d_da.values, dtype=float)
    strat = np.isfinite(Z) & (Z >= params.precip_floor_dbz)
    ax[0].contourf(lons, lats, np.where(strat, Z, np.nan), **kws)
    ax[1].contourf(lons, lats, np.asarray(masks["Z_bg"].values), **kws)
    enh_cf = ax[2].contourf(lons, lats, np.asarray(masks["Enh"].values),
                            **kws)
    ax[3].contourf(lons, lats, Z, **kws)

    tkw = {"transform": crs.PlateCarree()} if have_cartopy else {}
    sup = np.asarray(masks["support"].values).astype(int)
    ax[2].contour(lons, lats, sup, levels=[0.5], colors="white",
                  linewidths=3, zorder=5, **tkw)
    ax[2].contour(lons, lats, sup, levels=[0.5], colors="m",
                  linewidths=1.5, zorder=5, **tkw)
    ax[2].contour(lons, lats, np.asarray(labels_env) > 0, colors="red",
                  linewidths=1.5, zorder=6, **tkw)
    if np.any(np.asarray(labels_faint)):
        ax[3].contour(lons, lats, np.asarray(labels_faint) > 0,
                      colors="blue", linewidths=1.5, zorder=6, **tkw)
    if np.any(np.asarray(labels_strong)):
        ax[3].contour(lons, lats, np.asarray(labels_strong) > 0,
                      colors="red", linewidths=1.5, zorder=6, **tkw)

    fig.suptitle(title or "CESC identification", y=0.98, fontsize=16,
                 fontweight="bold")
    cax = fig.add_axes([0.35, 0.03, 0.30, 0.02])
    plt.colorbar(enh_cf, cax=cax, orientation="horizontal").set_label(
        "Reflectivity (dBZ)")
    return fig
