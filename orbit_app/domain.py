from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import re
from typing import Any


def is_comet_designation(value: str) -> bool:
    """Recognize provisional and permanently numbered comet designations."""
    # MPC interstellar designations (1I, 2I, 3I/ATLAS, …) behave like comet
    # designations throughout the UI, including comet photometry and optional
    # Marsden A1/A2/A3 parameters and optional DT activity lag.
    return bool(re.match(r"^\s*(?:[ACDPX]/|\d+[PDI](?:/|\b))", value or "", re.I))


@dataclass(slots=True)
class Observation:
    designation: str = ""
    time: str = ""
    ra_deg: float | None = None
    dec_deg: float | None = None
    magnitude: float | None = None
    band: str = ""
    station: str = ""
    catalog: str = ""
    residual_ra_arcsec: float | None = None
    residual_dec_arcsec: float | None = None
    used_in_fit: bool | None = None
    excluded_from_fit: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrbitSolution:
    designation: str
    epoch_jd: float
    semi_major_axis_au: float
    eccentricity: float
    inclination_deg: float
    node_deg: float
    arg_peri_deg: float
    mean_anomaly_deg: float
    absolute_magnitude: float = 18.0
    slope_g: float = 0.15
    photometric_exponent: float = 2.0
    phase_correction: bool = True
    provisional: bool = True
    position_au: tuple[float, float, float] | None = None
    velocity_au_per_day: tuple[float, float, float] | None = None
    state_covariance: tuple[tuple[float, ...], ...] | None = None
    element_uncertainties: dict[str, float] | None = None
    fit_observation_count: int | None = None
    fit_rms_arcsec: float | None = None
    fit_normalized_rms: float | None = None
    display_name: str | None = None
    nongravitational_parameters: dict[str, float] | None = None
    dynamical_1a: dict[str, float | str] | None = None
    apparition_elements: tuple[dict[str, Any], ...] | None = None
    fit_model_note: str | None = None
    data_source: str = "MPC"


@dataclass(slots=True)
class EphemerisRow:
    time: datetime
    ra_deg: float
    dec_deg: float
    distance_earth_au: float
    distance_sun_au: float
    magnitude: float

