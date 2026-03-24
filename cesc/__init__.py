# cesc/__init__.py
"""
CESC: Identification and analysis of Convection Embedded within or Emergent
from Stratiform Clouds in cold-season orographic precipitation.

This package implements the two-step algorithm described in Afrifa et al.
(Part 1, submitted 2026).  The first step identifies Embedded/Emergent
Convection Environments (ECE) from vertical theta_e profiles in WRF model
output.  The second step detects convective reflectivity objects within or
downstream of ECE regions using a multi-scale adaptive background method
adapted from Yeh and Colle (2025).

Typical usage
-------------
Single time step::

    from cesc.pipeline import run_cesc_pipeline, scan_pi_stable
    import numpy as np

    dtheta_e = np.diff(theta_e_3d, axis=0)
    pi_str, pi_dep, st_str, st_dep, pi_h0, pi_h1, st_h0, st_h1 = scan_pi_stable(
        dtheta_e, heights_3d, max_pi_height=5000.0,
        unstable_threshold=1.0, stable_threshold=2.0,
    )
    result = run_cesc_pipeline(
        Z_2d_da=reflectivity_da, ter_da=terrain_da,
        pi_strength_da=pi_str, stable_strength_da=st_str,
        pi_start_h_da=pi_h0, pi_end_h_da=pi_h1,
        u3d=u_da, v3d=v_da, heights=heights_da,
    )

Seasonal batch run::

    from cesc.run import run_month_streaming
    run_month_streaming(
        nc_path="/path/to/WRF_extracted_vars_900m_feb2017.nc",
        wrfout_path="/path/to/wrfout_d02_dir/",
        out_nc_path="/path/to/CESC_grids_feb2017.nc",
        out_parquet_path="/path/to/CESC_objects_feb2017.parquet",
    )

References
----------
Afrifa, F. O., and co-authors, 2026: Climatology of cold-season emergent
    convection in frontal systems and its impact on orographic precipitation.
    Part 1: Detection algorithm.  Mon. Wea. Rev., in review.

Afrifa, F. O., B. Geerts, L. Xue, S. Chen, C. Hohman, C. Grasmick, and
    T. Zaremba, 2025: A case study of cold-season emergent orographic
    convection and its impact on precipitation. Part I: Mesoscale analysis.
    Mon. Wea. Rev., 153, 2229-2250.

Yeh, P., and B. A. Colle, 2025: A Comparison of Approaches to Objectively
    Identify Precipitation Structures within the Comma Head of Mid-Latitude
    Cyclones.  J. Atmos. Oceanic Technol., 42, 463-477.
"""

__version__ = "1.0.0"
__author__  = "Francis Osei Tutu Afrifa, Bart Geerts"

from cesc.pipeline import run_cesc_pipeline, scan_pi_stable
from cesc.id_pro    import detect_objects_id_pro, ObjectProps
from cesc.utils     import find_wrfout, StreamingNCWriter

__all__ = [
    "run_cesc_pipeline",
    "scan_pi_stable",
    "detect_objects_id_pro",
    "ObjectProps",
    "find_wrfout",
    "StreamingNCWriter",
]
