"""The single source of truth for algorithm parameters.

Every default below is the Table 1 production value from Afrifa et al.
(Part 1, J. Atmos. Sci.). Some research-code function signatures carried
older defaults (faint_min_diff 2.0, min_area 6.0, box_sizes (20, 40, 80));
those signatures are gone in this package. Functions in cesc take their
parameters explicitly from a Params instance, so a call site cannot
silently disagree with the library.
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple


@dataclass
class Params:
    """Algorithm parameters with Table 1 defaults.

    Attributes
    ----------
    env_mode : str
        "ECE" requires the stable near-surface layer beneath the PI layer
        (elevated convection). "PI_only" drops the stability requirement,
        admitting surface-based convection; this is the PI_ONLY
        sensitivity configuration of the paper.
    unstable_threshold : float
        Minimum cumulative theta_e drop of the PI layer (K, positive
        convention).
    stable_threshold : float
        Minimum cumulative positive theta_e rise of the near-surface
        layer (K).
    max_pi_height_m : float
        PI search ceiling (m AGL).
    min_pi_depth_m : float
        Minimum PI layer depth (m).
    max_stable_height_m : float
        Stable-layer scan ceiling (m AGL); the scan also stops at the
        PI base.
    theta_eps : float
        Noise tolerance for the theta_e scans (K).
    faint_min_diff, strong_min_diff : float
        Absolute dBZ floors of the adaptive detection thresholds.
    faint_k_sigma, strong_k_sigma : float
        Sigma multipliers of the adaptive detection thresholds.
    box_sizes : tuple of float
        Full widths (km) of the multi-scale background boxes. Scale
        these with the size of the cells being targeted; doubling to
        (20, 40, 80) km moved CESC area +46% and volume +62% in the
        published sensitivity test.
    bg_radius : float
        Smoothing radius (km) of the display background Z_bg. Drives
        the quicklook panel and the watershed seeding field Enh only.
        It never enters the detection thresholds.
    min_area_km2 : float
        Minimum object area (km^2); 4.05 is 5 grid points at 900 m.
    min_support_frac : float
        Minimum fraction of an object's area inside the support mask.
    min_support_px : int
        Minimum pixel count of an object inside the support mask.
    use_advection : bool
        Build the downstream corridor from the PI-layer wind.
    advection_time_min : float
        Corridor lookahead time (minutes).
    halfwidth_km : float
        Lateral corridor half-width (km).
    seed_stride_px : int
        Corridor seed spacing (pixels).
    min_speed_ms : float
        Minimum wind speed to draw a corridor ray (m/s).
    frac_strong_cutoff : float
        Objects with frac_strong at or above this are strong, below it
        faint.
    precip_floor_dbz : float
        Precipitating-domain floor; detection runs where Z >= this.
    length_cut_km, aspect_cut : float
        Shape-classification thresholds (band / cell / complex).
    """

    env_mode: str = "ECE"
    unstable_threshold: float = 1.0
    stable_threshold: float = 2.0
    max_pi_height_m: float = 5000.0
    min_pi_depth_m: float = 500.0
    max_stable_height_m: float = 3000.0
    theta_eps: float = 5e-3

    faint_min_diff: float = 5.0
    faint_k_sigma: float = 0.30
    strong_min_diff: float = 10.0
    strong_k_sigma: float = 0.30
    box_sizes: Tuple[float, ...] = (10.0, 20.0, 40.0)
    bg_radius: float = 30.0
    min_area_km2: float = 4.05

    min_support_frac: float = 0.30
    min_support_px: int = 5

    use_advection: bool = True
    advection_time_min: float = 20.0
    halfwidth_km: float = 10.0
    seed_stride_px: int = 5
    min_speed_ms: float = 2.0

    frac_strong_cutoff: float = 0.5
    precip_floor_dbz: float = -2.0

    length_cut_km: float = 100.0
    aspect_cut: float = 0.5

    def __post_init__(self):
        mode = str(self.env_mode).upper().replace("-", "_")
        if mode in ("PI_ONLY", "PIONLY", "PI"):
            mode = "PI_only"
        elif mode == "ECE":
            mode = "ECE"
        else:
            raise ValueError(
                "env_mode must be 'ECE' or 'PI_only', got %r" % self.env_mode)
        object.__setattr__(self, "env_mode", mode)
        if self.min_area_km2 <= 0:
            raise ValueError("min_area_km2 must be positive")
        if not (0.0 <= self.min_support_frac <= 1.0):
            raise ValueError("min_support_frac must lie in [0, 1]")
        if len(self.box_sizes) == 0:
            raise ValueError("box_sizes must contain at least one width")

    def to_dict(self):
        d = asdict(self)
        d["box_sizes"] = list(self.box_sizes)
        return d

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        if "box_sizes" in d:
            d["box_sizes"] = tuple(d["box_sizes"])
        return cls(**d)
