"""Pure formatting and projection helpers shared by the Qt user interface.

This module deliberately has no PySide6 or Matplotlib dependency.  Keeping
these deterministic transformations outside ``main_window`` makes them usable
from reports, tests, and future non-GUI front ends.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from .domain import Observation, OrbitSolution, is_comet_designation
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

LUNAR_SIDEREAL_PERIOD_DAYS = 27.321661


def format_diameter(diameter_km: float) -> str:
    """Diameter with a unit suited to its size (m below 1 km, km above)."""
    value = float(diameter_km)
    if not np.isfinite(value) or value <= 0.0:
        return "—"
    if value < 0.001:
        return f"{value * 1e5:.0f} cm"
    if value < 1.0:
        metres = value * 1000.0
        return f"{metres:.0f} m" if metres >= 10.0 else f"{metres:.1f} m"
    return f"{value:.0f} km" if value >= 100.0 else f"{value:.1f} km"


def asteroid_size_lines(solution: OrbitSolution) -> list[str]:
    """Report lines with the diameter range implied by H for an asteroid.

    Empty for comets (H/M of an active comet measures the coma, not the
    nucleus) and while H is only a placeholder.
    """
    from .photometry import (
        ASTEROID_ALBEDO_EXTREME, ASTEROID_ALBEDO_TYPICAL,
        diameter_from_absolute_magnitude_km,
    )

    if (
        not solution.absolute_magnitude_known
        or not np.isfinite(float(solution.absolute_magnitude))
        or is_comet_designation(solution.designation)
        or is_comet_designation(solution.display_name or "")
    ):
        return []
    h = float(solution.absolute_magnitude)

    def size_range(albedos: tuple[float, float]) -> str:
        bright, dark = albedos
        return (
            f"{format_diameter(diameter_from_absolute_magnitude_km(h, bright))} – "
            f"{format_diameter(diameter_from_absolute_magnitude_km(h, dark))}"
        )

    typical_bright, typical_dark = ASTEROID_ALBEDO_TYPICAL
    extreme_bright, extreme_dark = ASTEROID_ALBEDO_EXTREME
    return [
        tr("Odhad průměru z H (albedo není známo):"),
        tr(
            "  obvyklý rozsah  {0}  (světlé těleso typu S p_V = {1:.2f} až tmavé typu C p_V = {2:.2f})",
            size_range(ASTEROID_ALBEDO_TYPICAL), typical_bright, typical_dark,
        ),
        tr(
            "  krajní meze     {0}  (p_V = {1:.2f} až {2:.2f})",
            size_range(ASTEROID_ALBEDO_EXTREME), extreme_bright, extreme_dark,
        ),
        tr(
            "  D = 1329 km / √p_V · 10^(−H/5); nejistota H ±0,3 mag sama mění průměr "
            "asi o ±15 %.",
        ),
    ]


def sky_track_sample_count(horizon_days: int) -> int:
    """Keep short tracks smooth without making multi-year views enormous."""
    return min(2401, max(241, int(horizon_days) * 5 + 1))


def equatorial_ra_ticks(
    center_ra_deg: float, field_deg: float,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Return sky-anchored RA ticks for a target-centred rectangular map."""
    if field_deg <= 0.5:
        step_hours = 10.0 / 3600.0
    elif field_deg <= 2.0:
        step_hours = 1.0 / 60.0
    elif field_deg <= 6.0:
        step_hours = 5.0 / 60.0
    elif field_deg <= 15.0:
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
        total_seconds = int(round((degrees % 360.0) * 240.0)) % (24 * 3600)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if seconds:
            labels.append(f"{hours:02d}h{minutes:02d}m{seconds:02d}s")
        else:
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


def projected_ecliptic_curtain_quads(
    orbit_xyz_au: np.ndarray, azimuth_deg: float, elevation_deg: float,
) -> np.ndarray:
    """Project an orbit-to-ecliptic curtain as adjacent screen-space quads.

    ``orbit_xyz_au`` is a sampled heliocentric J2000-ecliptic path in AU.  The
    returned array has shape ``(N - 1, 4, 2)``.  Local quadrilaterals avoid the
    false closing chords produced when a concave projected curtain is filled as
    one polygon.
    """
    orbit = np.asarray(orbit_xyz_au, dtype=float)
    if orbit.ndim != 2 or orbit.shape[1] != 3:
        raise ValueError("orbit_xyz_au must have shape (N, 3)")
    if len(orbit) < 2:
        return np.empty((0, 4, 2), dtype=float)
    base = orbit.copy()
    base[:, 2] = 0.0
    orbit_x, orbit_y, _ = project_ecliptic_camera(
        orbit, azimuth_deg, elevation_deg,
    )
    base_x, base_y, _ = project_ecliptic_camera(
        base, azimuth_deg, elevation_deg,
    )
    return np.stack((
        np.column_stack((orbit_x[:-1], orbit_y[:-1])),
        np.column_stack((orbit_x[1:], orbit_y[1:])),
        np.column_stack((base_x[1:], base_y[1:])),
        np.column_stack((base_x[:-1], base_y[:-1])),
    ), axis=1)


def lunar_orbit_path_epochs(current_jd: float) -> tuple[int, np.ndarray]:
    """Return a day-keyed, current-epoch lunar path in dynamical JD.

    The path is centred on the UTC-like integer JD bucket instead of starting
    at the first frame ever drawn.  Consequently an animated Moon always lies
    on the same ephemeris curve while the cache remains reusable for one day.
    """
    cache_day = int(np.floor(float(current_jd)))
    half_period = LUNAR_SIDEREAL_PERIOD_DAYS / 2.0
    centre = cache_day + .5
    return cache_day, np.linspace(
        centre - half_period, centre + half_period, 481,
    )


def trajectory_display_indices(
    trajectory_epochs: np.ndarray,
    capture_ranges: tuple[tuple[float, float], ...] = (),
    maximum_points: int = 6000,
) -> np.ndarray:
    """Thin a long trajectory while retaining dense planet-capture arcs."""
    epochs = np.asarray(trajectory_epochs, dtype=float)
    if epochs.ndim != 1:
        raise ValueError("trajectory_epochs must be one-dimensional")
    if len(epochs) <= maximum_points:
        return np.arange(len(epochs), dtype=int)
    if maximum_points < 2:
        raise ValueError("maximum_points must be at least two")

    capture_mask = np.zeros(len(epochs), dtype=bool)
    for start, end in capture_ranges:
        lower, upper = sorted((float(start), float(end)))
        capture_mask |= (epochs >= lower) & (epochs <= upper)
    capture_indices = np.flatnonzero(capture_mask)

    # Reserve enough context to show the heliocentric branches that lead into
    # and out of a capture.  If the captured intervals alone are enormous,
    # keep a dense representative subset rather than exhausting the plot.
    capture_budget = min(len(capture_indices), max(2, maximum_points * 3 // 4))
    if len(capture_indices) > capture_budget:
        capture_indices = capture_indices[
            np.linspace(0, len(capture_indices) - 1, capture_budget, dtype=int)
        ]
    background_budget = maximum_points - len(capture_indices)
    background = np.linspace(
        0, len(epochs) - 1, max(2, background_budget), dtype=int,
    )
    selected = np.unique(np.concatenate((background, capture_indices)))
    if len(selected) > maximum_points:
        selected = selected[
            np.linspace(0, len(selected) - 1, maximum_points, dtype=int)
        ]
    return selected


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
