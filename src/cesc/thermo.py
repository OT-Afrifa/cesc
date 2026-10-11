"""Equivalent potential temperature following Bolton (1980).

The CESC algorithm scans theta_e, not dry theta and not theta_v.
Potential instability is a saturation criterion (Schultz and Schumacher
1999): the latent-heat contribution carried by theta_e is what makes a
moist layer unstable when lifted to saturation. A dry theta or theta_v
profile misses it. The preflight module checks supplied "theta_e"
against a dry theta computed from the same inputs to catch this mix-up.

Reference: Bolton, D., 1980: The computation of equivalent potential
temperature. Mon. Wea. Rev., 108, 1046-1053. Equations 10, 15, 43.
"""
import numpy as np

P0 = 100000.0        # Pa
RD_CP = 0.2854       # Rd/cp for dry air
EPS = 0.622          # Rd/Rv


def dewpoint_from_q(pressure_pa, qv):
    """Dewpoint (K) from pressure (Pa) and water-vapor mixing ratio (kg/kg).

    Inverts Bolton eq. 10 for the vapor pressure e = p q / (eps + q).
    """
    qv = np.maximum(np.asarray(qv, dtype=np.float64), 1e-12)
    e = pressure_pa * qv / (EPS + qv)
    e_hpa = np.maximum(e / 100.0, 1e-6)
    ln_e = np.log(e_hpa / 6.112)
    return 243.5 * ln_e / (17.67 - ln_e) + 273.15


def t_lcl(temperature_k, dewpoint_k):
    """Temperature at the lifting condensation level (K), Bolton eq. 15."""
    td = np.asarray(dewpoint_k, dtype=np.float64)
    t = np.asarray(temperature_k, dtype=np.float64)
    return 1.0 / (1.0 / (td - 56.0) + np.log(t / td) / 800.0) + 56.0


def theta_e_bolton(temperature_k, pressure_pa, qv):
    """Equivalent potential temperature (K), Bolton eq. 43.

    Parameters
    ----------
    temperature_k : array
        Air temperature (K).
    pressure_pa : array
        Pressure (Pa).
    qv : array
        Water-vapor mixing ratio (kg/kg).

    Returns
    -------
    theta_e : array (K)
    """
    t = np.asarray(temperature_k, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)
    r = np.maximum(np.asarray(qv, dtype=np.float64), 1e-12)
    td = dewpoint_from_q(p, r)
    tl = t_lcl(t, td)
    # theta at the LCL with the moisture-corrected exponent (eq. 24 of
    # Bolton as used in eq. 43)
    theta_dl = t * (P0 / p) ** (RD_CP * (1.0 - 0.28 * r))
    theta_l = theta_dl * (t / tl) ** (0.28 * r)
    return theta_l * np.exp((3376.0 / tl - 2.54) * r * (1.0 + 0.81 * r))


def theta_dry(temperature_k, pressure_pa):
    """Dry potential temperature (K). Used by preflight to catch mix-ups."""
    t = np.asarray(temperature_k, dtype=np.float64)
    p = np.asarray(pressure_pa, dtype=np.float64)
    return t * (P0 / p) ** RD_CP
