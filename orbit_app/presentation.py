"""Pure formatting and projection helpers shared by the Qt user interface.

This module deliberately has no PySide6 or Matplotlib dependency.  Keeping
these deterministic transformations outside ``main_window`` makes them usable
from reports, tests, and future non-GUI front ends.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from .domain import Observation, OrbitSolution
from .i18n import tr
from .orbit_engine import MU_SUN


SKY_TRACK_HORIZONS = (
    (tr("± 30 dní"), 30),
    (tr("± 60 dní"), 60),
    (tr("± 100 dní"), 100),
    (tr("± 150 dní"), 150),
    (tr("± 200 dní"), 200),
    (tr("± 300 dní"), 300),
    (tr("± 600 dní"), 600),
    (tr("± 2,5 roku"), 913),
    (tr("± 5 let"), 1826),
    (tr("± 10 let"), 3652),
)

LINKAGE_STATUS_LABELS = {
    "linked": tr("společně fitován"),
    "seed_only": tr("jen samostatný fit"),
    "failed": tr("společný dynamický model neprošel"),
    "not_attempted": tr("společný model nezkoušen"),
    "integrated_only": tr("jen integrovaná předpověď"),
}


def sky_track_sample_count(horizon_days: int) -> int:
    """Keep short tracks smooth without making multi-year views enormous."""
    return min(2401, max(241, int(horizon_days) * 5 + 1))


def equatorial_ra_ticks(
    center_ra_deg: float, field_deg: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return sky-anchored RA ticks for a target-centred rectangular map."""
    if field_deg <= 15.0:
        step_hours = 0.25
    elif field_deg <= 30.0:
        step_hours = 0.5
    elif field_deg <= 60.0:
        step_hours = 1.0
    elif field_deg <= 120.0:
        step_hours = 2.0
    else:
        step_hours = 3.0
    step_deg = 15.0 * step_hours
    half_field = field_deg / 2.0
    start = np.ceil((center_ra_deg - half_field) / step_deg) * step_deg
    absolute = np.arange(
        start, center_ra_deg + half_field + step_deg * 1e-8, step_deg,
    )
    positions = absolute - center_ra_deg
    labels: list[str] = []
    for degrees in absolute:
        total_minutes = int(round((degrees % 360.0) * 4.0)) % (24 * 60)
        hours, minutes = divmod(total_minutes, 60)
        labels.append(
            f"{hours:02d}h" if minutes == 0 else f"{hours:02d}h{minutes:02d}"
        )
    return positions, tuple(labels)


def datetime_from_jd(jd: float) -> datetime | None:
    """Convert a Julian calendar date without relying on Unix timestamps.

    The calendar representation of TT/TDB epochs is retained for element
    reports.  Extreme values outside Python's datetime range are unavailable
    rather than fatal to the UI.
    """
    if not np.isfinite(jd):
        return None
    try:
        return datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(
            days=float(jd) - 2451545.0
        )
    except (OverflowError, ValueError, OSError):
        return None


def apparition_linkage_summary(solution: OrbitSolution) -> str | None:
    records = solution.apparition_linkage or ()
    if len(records) < 2:
        return None
    linked = sum(record.get("status") == "linked" for record in records)
    years = ", ".join(
        str(record.get("return_year") or "?")
        for record in records if record.get("status") == "linked"
    )
    if linked == len(records):
        return tr("NAVÁZÁNO {0}/{1} návratů ({2})", linked, len(records), years)
    if linked >= 2:
        return tr("ČÁSTEČNÝ SPOLEČNÝ FIT {0}/{1} návratů ({2})", linked, len(records), years)
    return tr("SPOLEČNÝ MODEL NEPOTVRZEN — fit 1/{0} návratů", len(records))


def apparition_observation_count_text(solution: OrbitSolution) -> str:
    used = solution.fit_observation_count
    available = solution.fit_available_observation_count
    if used is None:
        return ""
    if available is not None and available > used:
        return tr("{0} reprezentativních z {1} v propojeném oblouku", used, available)
    return tr("{0} použitých", used)


def perihelion_jd_from_solution(
    solution: OrbitSolution, mu: float = MU_SUN,
) -> float:
    """Return the nearest osculating perihelion epoch represented by a state."""
    mean_motion = np.sqrt(mu / abs(solution.semi_major_axis_au) ** 3)
    mean_anomaly = float(solution.mean_anomaly_deg)
    if solution.eccentricity < 1.0:
        mean_anomaly = (mean_anomaly + 180.0) % 360.0 - 180.0
    return solution.epoch_jd - np.radians(mean_anomaly) / mean_motion


def project_ecliptic_camera(
    points: np.ndarray, azimuth_deg: float, elevation_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orthographically project J2000-ecliptic XYZ through a solar camera."""
    values = np.asarray(points, dtype=float)
    original_shape = values.shape[:-1]
    flat = values.reshape(-1, 3)
    azimuth = np.radians(float(azimuth_deg))
    elevation = np.radians(float(elevation_deg))
    camera = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    right = np.array([-np.sin(azimuth), np.cos(azimuth), 0.0])
    up = np.cross(camera, right)
    return (
        (flat @ right).reshape(original_shape),
        (flat @ up).reshape(original_shape),
        (flat @ camera).reshape(original_shape),
    )


def solar_orbit_clip_radius(
    solution: OrbitSolution, viewport_radius: float,
) -> float | None:
    """Keep closed conics whole; bound only genuinely open trajectories."""
    if solution.eccentricity < 1.0 and solution.semi_major_axis_au > 0.0:
        return None
    return float(viewport_radius) * 1.5


def trajectory_return_masks(
    trajectory_epochs: np.ndarray, return_epochs: np.ndarray, current_jd: float,
    encounter_epochs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Classify trajectory samples into the active arc and its surroundings."""
    epochs = np.asarray(trajectory_epochs, dtype=float)
    perihelia = np.sort(np.asarray(return_epochs, dtype=float))
    if epochs.size == 0:
        empty = np.zeros(0, dtype=bool)
        return empty, empty, empty
    encounters = np.asarray(
        () if encounter_epochs is None else encounter_epochs, dtype=float,
    )
    encounters = encounters[np.isfinite(encounters)]
    if perihelia.size == 0 and encounters.size == 0:
        return (
            np.zeros(epochs.size, dtype=bool),
            np.ones(epochs.size, dtype=bool),
            np.zeros(epochs.size, dtype=bool),
        )
    return_boundaries = (
        (perihelia[:-1] + perihelia[1:]) / 2.0
        if perihelia.size > 1 else np.empty(0, dtype=float)
    )
    boundaries = np.unique(np.concatenate((return_boundaries, encounters)))
    active_index = int(np.searchsorted(boundaries, float(current_jd), side="right"))
    lower = -np.inf if active_index == 0 else float(boundaries[active_index - 1])
    upper = np.inf if active_index == len(boundaries) else float(boundaries[active_index])
    older = epochs < lower
    active = (epochs >= lower) & (epochs < upper)
    newer = epochs >= upper
    return older, active, newer


def format_ra_sexagesimal(ra_deg: float) -> str:
    total_seconds = round((float(ra_deg) % 360.0) * 240.0, 2) % 86400.0
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60.0
    return f"{hours:02d} {minutes:02d} {seconds:05.2f}"


def format_dec_sexagesimal(dec_deg: float) -> str:
    sign = "+" if float(dec_deg) >= 0.0 else "−"
    total_seconds = round(abs(float(dec_deg)) * 3600.0, 1)
    degrees = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60.0
    return f"{sign}{degrees:02d} {minutes:02d} {seconds:04.1f}"


def observation_residual_score(observation: Observation) -> float | None:
    """Largest absolute coordinate residual, matching the table colour scale."""
    values = (observation.residual_ra_arcsec, observation.residual_dec_arcsec)
    if any(value is None or not np.isfinite(float(value)) for value in values):
        return None
    return max(abs(float(value)) for value in values if value is not None)


def parse_output_epoch(value: str) -> datetime:
    text = value.strip()
    for date_format in ("%Y", "%Y-%m-%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, date_format)
        except ValueError:
            pass
    raise ValueError(tr("Neplatná výstupní epocha."))


def observations_to_tsv(observations: list[Observation]) -> str:
    """Create a clipboard-friendly astrometry table including both residuals."""
    lines = [
        tr(
            "Čas (UTC)\tRA (h m s)\tDec (° ′ ″)\tMag / pásmo\tStanice\tPoužít\tO−C RA "
            "[″]\tO−C Dec [″]\tVe fitu",
        )
    ]
    for observation in observations:
        magnitude = (
            "—" if observation.magnitude is None
            else f"{observation.magnitude:.2f}"
        )
        if observation.band:
            magnitude = f"{magnitude} {observation.band}"
        residual_ra = (
            "—" if observation.residual_ra_arcsec is None
            else f"{observation.residual_ra_arcsec:+.3f}"
        )
        residual_dec = (
            "—" if observation.residual_dec_arcsec is None
            else f"{observation.residual_dec_arcsec:+.3f}"
        )
        used = (
            tr("ano") if observation.used_in_fit is True
            else tr("ne") if observation.used_in_fit is False else "—"
        )
        lines.append("\t".join((
            observation.time or "—",
            "—" if observation.ra_deg is None
            else format_ra_sexagesimal(observation.ra_deg),
            "—" if observation.dec_deg is None
            else format_dec_sexagesimal(observation.dec_deg),
            magnitude,
            observation.station or "—",
            tr("ne") if observation.excluded_from_fit else tr("ano"),
            residual_ra,
            residual_dec,
            used,
        )))
    return "\n".join(lines)
