from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from .domain import Observation, OrbitSolution, is_comet_designation
from .mpc_orbit import cartesian_to_elements
from .observatories import (
    dynamical_jd, earth_heliocentric_ecliptic, geodetic_offsets_gcrs,
    station_offsets_gcrs,
)
from .orbit_engine import MU_SUN, NBodyIntegrator, StateVector
from .planetary import PLANETS, SplinePlanetEphemeris
from .photometry import estimate_absolute_magnitude, hg_phase_correction

C_AU_PER_DAY = 173.1446326846693
_PLANET_NAMES = (
    "mercury", "venus", "earth", "moon", "mars",
    "jupiter", "saturn", "uranus", "neptune",
)


class OrbitFitError(RuntimeError):
    pass


class OrbitFitCancelled(Exception):
    """Raised cooperatively when the user cancels a numerical fit."""


ProgressCallback = Callable[[str, int, int], None]
CancelCallback = Callable[[], bool]


def _calibrate_fitted_photometry(
    observations: list[Observation], solution: OrbitSolution,
) -> None:
    """Reduce observed magnitudes to H/cometary M using the fitted geometry."""
    measured: list[tuple[datetime, float]] = []
    for observation in observations:
        if observation.magnitude is None or not observation.time:
            continue
        try:
            magnitude = float(observation.magnitude)
            instant = parse_observation_time(observation.time)
        except (TypeError, ValueError):
            continue
        if np.isfinite(magnitude):
            measured.append((instant, magnitude))
    if not measured:
        return
    # Thousands of photometric rows add no robustness to a median but would
    # needlessly repeat an expensive long-period propagation.
    if len(measured) > 240:
        indices = np.linspace(0, len(measured) - 1, 240).round().astype(int)
        measured = [measured[index] for index in np.unique(indices)]
    try:
        # Local import avoids coupling the numerical fitter to preview startup.
        from .preview_orbit import apparent_geometry
        epochs = dynamical_jd([item[0] for item in measured])
        geometry = apparent_geometry(solution, epochs)
        phase = (
            hg_phase_correction(geometry["phase"], solution.slope_g)
            if solution.phase_correction else None
        )
        solution.absolute_magnitude = estimate_absolute_magnitude(
            np.asarray([item[1] for item in measured], dtype=float),
            geometry["delta"], geometry["r"],
            solution.photometric_exponent, phase,
        )
    except (ValueError, FloatingPointError, RuntimeError):
        # Photometry must never invalidate an otherwise sound astrometric fit.
        return


class _InterpolatedPlanetEphemeris:
    """Dense short-arc cache around an underlying planetary ephemeris."""

    def __init__(self, source, start_jd: float, stop_jd: float, step_days: float = 0.125):
        self.source = source
        count = max(5, int(np.ceil((stop_jd - start_jd) / step_days)) + 1)
        self.epochs = np.linspace(start_jd, stop_jd, count)
        self.splines = {
            body: CubicSpline(
                self.epochs,
                np.vstack([source.position(body, float(epoch)) for epoch in self.epochs]),
                axis=0,
            )
            for body in _PLANET_NAMES
        }

    def position(self, body: str, jd: float) -> np.ndarray:
        epoch = float(jd)
        if epoch < self.epochs[0] or epoch > self.epochs[-1]:
            return np.asarray(self.source.position(body, epoch), dtype=float)
        return np.asarray(self.splines[body.lower()](epoch), dtype=float)

    def positions(self, bodies: tuple[str, ...], jd: float) -> tuple[np.ndarray, ...]:
        return tuple(self.position(body, jd) for body in bodies)


def _equatorial_to_ecliptic(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    eps = np.radians(23.4392911)
    return np.column_stack((
        values[..., 0],
        values[..., 1] * np.cos(eps) + values[..., 2] * np.sin(eps),
        -values[..., 1] * np.sin(eps) + values[..., 2] * np.cos(eps),
    ))


def _ecliptic_to_equatorial(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    eps = np.radians(23.4392911)
    return np.column_stack((
        values[..., 0],
        values[..., 1] * np.cos(eps) - values[..., 2] * np.sin(eps),
        values[..., 1] * np.sin(eps) + values[..., 2] * np.cos(eps),
    ))


def _truncate_rows_to_recent_years(
    rows: list[tuple[Observation, datetime, float, float]], years: float | None,
) -> list[tuple[Observation, datetime, float, float]]:
    if years is None or not rows:
        return rows
    newest = max(item[1] for item in rows)
    cutoff = newest - timedelta(days=365.25 * years)
    return [item for item in rows if item[1] >= cutoff]


def fit_selection_count(
    observations: list[Observation], years: float | None,
) -> int:
    """Count valid, enabled observations in the selected recent-time window."""
    parsed: list[tuple[Observation, datetime, float, float]] = []
    for observation in observations:
        if (
            observation.excluded_from_fit or observation.ra_deg is None or
            observation.dec_deg is None or not observation.time
        ):
            continue
        try:
            instant = parse_observation_time(observation.time)
        except ValueError:
            continue
        parsed.append((observation, instant, observation.ra_deg, observation.dec_deg))
    return len(_truncate_rows_to_recent_years(parsed, years))


def _perihelion_offset_days_from_state(
    state_vector: np.ndarray, force_parabolic: bool = False,
) -> float:
    """Time since perihelion, avoiding the singular ``a, M`` form at e=1."""
    position = np.asarray(state_vector[:3], dtype=float)
    velocity = np.asarray(state_vector[3:6], dtype=float)
    radius = float(np.linalg.norm(position))
    angular_momentum = np.cross(position, velocity)
    h_norm = float(np.linalg.norm(angular_momentum))
    speed2 = float(np.dot(velocity, velocity))
    eccentricity_vector = (
        (speed2 - MU_SUN / radius) * position
        - float(np.dot(position, velocity)) * velocity
    ) / MU_SUN
    eccentricity = float(np.linalg.norm(eccentricity_vector))
    if force_parabolic or abs(eccentricity - 1.0) < 1e-4:
        if eccentricity <= 1e-12 or h_norm <= 1e-18:
            raise ValueError("Nelze určit směr perihelu.")
        cosine = float(np.dot(eccentricity_vector, position)) / (eccentricity * radius)
        sine = float(
            np.dot(np.cross(eccentricity_vector, position), angular_momentum)
        ) / (eccentricity * radius * h_norm)
        true_anomaly = float(np.arctan2(sine, cosine))
        barker_d = float(np.tan(true_anomaly / 2.0))
        semilatus_rectum = h_norm * h_norm / MU_SUN
        return 0.5 * np.sqrt(semilatus_rectum ** 3 / MU_SUN) * (
            barker_d + barker_d ** 3 / 3.0
        )
    a, e, _, _, _, mean_anomaly = cartesian_to_elements(position, velocity)
    mean_motion = 0.01720209895 / abs(a) ** 1.5
    if e < 1.0:
        mean_anomaly = (mean_anomaly + 180.0) % 360.0 - 180.0
    return float(np.radians(mean_anomaly) / mean_motion)


def _formal_element_uncertainties(
    state_vector: np.ndarray, elements: tuple[float, float, float, float, float, float],
    astrometric_jacobian: np.ndarray, astrometric_residual: np.ndarray,
    free_parameter_count: int, nongrav_names: tuple[str, ...] = (),
) -> tuple[dict[str, float] | None, np.ndarray | None]:
    """Linearised 1-sigma uncertainties; robust-fit covariance is approximate."""
    try:
        degrees_of_freedom = max(1, astrometric_residual.size - free_parameter_count)
        variance = float(np.dot(astrometric_residual, astrometric_residual) / degrees_of_freedom)
        parameter_covariance = np.linalg.pinv(
            astrometric_jacobian.T @ astrometric_jacobian
        ) * variance
        state_covariance = parameter_covariance[:6, :6]
        element_jacobian = np.empty((6, 6), dtype=float)
        angular_indices = (2, 3, 4, 5)
        for column in range(6):
            step = max(abs(float(state_vector[column])) * 1e-6, 1e-8 if column < 3 else 1e-10)
            plus = state_vector.copy(); plus[column] += step
            minus = state_vector.copy(); minus[column] -= step
            plus_elements = np.asarray(cartesian_to_elements(plus[:3], plus[3:6]), dtype=float)
            minus_elements = np.asarray(cartesian_to_elements(minus[:3], minus[3:6]), dtype=float)
            difference = plus_elements - minus_elements
            for index in angular_indices:
                difference[index] = (difference[index] + 180.0) % 360.0 - 180.0
            element_jacobian[:, column] = difference / (2.0 * step)
        element_covariance = element_jacobian @ state_covariance @ element_jacobian.T
        sigma = np.sqrt(np.maximum(0.0, np.diag(element_covariance)))
        a, e, _, _, _, _ = elements
        q_state_gradient = np.empty(6, dtype=float)
        peri_state_gradient = np.empty(6, dtype=float)
        near_parabolic = abs(e - 1.0) < 1e-3
        for column in range(6):
            step = max(
                abs(float(state_vector[column])) * 1e-6,
                1e-8 if column < 3 else 1e-10,
            )
            plus = state_vector.copy(); plus[column] += step
            minus = state_vector.copy(); minus[column] -= step
            plus_elements = cartesian_to_elements(plus[:3], plus[3:6])
            minus_elements = cartesian_to_elements(minus[:3], minus[3:6])
            plus_q = plus_elements[0] * (1.0 - plus_elements[1])
            minus_q = minus_elements[0] * (1.0 - minus_elements[1])
            q_state_gradient[column] = (plus_q - minus_q) / (2.0 * step)
            plus_peri = _perihelion_offset_days_from_state(plus, near_parabolic)
            minus_peri = _perihelion_offset_days_from_state(minus, near_parabolic)
            # T = epoch - elapsed time, hence the minus sign.  It vanishes
            # after squaring but documents the quantity being propagated.
            peri_state_gradient[column] = -(plus_peri - minus_peri) / (2.0 * step)
        q_sigma = float(np.sqrt(max(
            0.0, q_state_gradient @ state_covariance @ q_state_gradient,
        )))
        mean_motion_rad = 0.01720209895 / abs(a) ** 1.5
        mean_motion_deg = np.degrees(mean_motion_rad)
        mean_motion_sigma = abs(1.5 * mean_motion_deg / a) * sigma[0]
        peri_sigma = float(np.sqrt(max(
            0.0, peri_state_gradient @ state_covariance @ peri_state_gradient,
        )))
        names = ("a", "e", "i", "node", "arg_peri", "mean_anomaly")
        uncertainties = {name: float(value) for name, value in zip(names, sigma)}
        uncertainties.update({
            "q": q_sigma,
            "mean_motion_deg_day": float(mean_motion_sigma),
            "perihelion_jd": peri_sigma,
        })
        for index, name in enumerate(nongrav_names, start=6):
            # A parameters are scaled by 1e-8 inside the optimiser; DT is days.
            scale = 1.0 if name == "DT" else 1e-8
            uncertainties[name] = float(
                np.sqrt(max(0.0, parameter_covariance[index, index])) * scale
            )
        return uncertainties, state_covariance
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError):
        return None, None


def parse_observation_time(value: str) -> datetime:
    normalized = value.strip().replace("Z", "+00:00")
    # Python 3.10 accepts only selected ISO fractional-second widths.  ADES
    # legitimately contains values such as ``...51.17Z`` (station 807 in the
    # C/2024 Y1 data), so normalise every fraction to six digits first.
    match = re.match(r"^(.*:\d{2})\.(\d+)([+-]\d{2}:\d{2})$", normalized)
    if match:
        normalized = f"{match.group(1)}.{(match.group(2) + '000000')[:6]}{match.group(3)}"
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _equatorial_direction(
    states: np.ndarray, epochs: np.ndarray, station_offsets: np.ndarray | None = None,
    earth_positions: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    earth = (
        earth_heliocentric_ecliptic(epochs)
        if earth_positions is None else np.asarray(earth_positions, dtype=float)
    )
    vectors = states[:, :3] - earth
    eps = np.radians(23.4392911)
    equatorial = np.column_stack((
        vectors[:, 0],
        vectors[:, 1] * np.cos(eps) - vectors[:, 2] * np.sin(eps),
        vectors[:, 1] * np.sin(eps) + vectors[:, 2] * np.cos(eps),
    ))
    if station_offsets is not None:
        equatorial -= station_offsets
    x, y, z = equatorial.T
    radius = np.linalg.norm(equatorial, axis=1)
    ra = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
    dec = np.degrees(np.arcsin(np.clip(z / radius, -1.0, 1.0)))
    return ra, dec


def _observation_offsets_gcrs(
    rows: list[tuple[Observation, datetime, float, float]],
) -> np.ndarray:
    """Resolve terrestrial stations and per-observation spacecraft offsets."""
    times = [item[1] for item in rows]
    offsets = station_offsets_gcrs([item[0].station for item in rows], times)
    geodetic_indices: list[int] = []
    geodetic_coordinates: list[tuple[float, float, float]] = []
    for index, (observation, _, _, _) in enumerate(rows):
        wgs84 = observation.raw.get("observer_wgs84")
        if wgs84 is not None:
            try:
                coordinate = tuple(float(value) for value in wgs84)
                if len(coordinate) == 3 and np.all(np.isfinite(coordinate)):
                    geodetic_indices.append(index)
                    geodetic_coordinates.append(coordinate)
            except (TypeError, ValueError):
                pass
        spacecraft = observation.raw.get("observer_position_gcrs_au")
        if spacecraft is None:
            continue
        try:
            vector = np.asarray(spacecraft, dtype=float)
            if vector.shape == (3,) and np.all(np.isfinite(vector)):
                offsets[index] = vector
        except (TypeError, ValueError):
            continue
    if geodetic_indices:
        offsets[np.asarray(geodetic_indices)] = geodetic_offsets_gcrs(
            geodetic_coordinates, [times[index] for index in geodetic_indices]
        )
    return offsets


def _observation_sigma_arcsec(observation: Observation) -> tuple[float, float]:
    """Return ADES 1-sigma values or a conservative epoch-based fallback.

    A large part of the historical archive was converted to ADES from legacy
    records and carries no meaningful per-row uncertainty.  One arcsecond is
    far too optimistic for visual/micrometric nineteenth-century astrometry.
    Conversely, a modern CCD record is not assumed to be good merely because
    it is modern: robust residual handling below can still downweight/reject it.
    """
    lowered = {str(key).casefold(): value for key, value in observation.raw.items()}

    try:
        year = parse_observation_time(observation.time).year
    except (TypeError, ValueError):
        year = 2000
    if year < 1850:
        fallback = 20.0
    elif year < 1900:
        fallback = 12.0
    elif year < 1950:
        fallback = 6.0
    elif year < 1990:
        fallback = 3.0
    elif year < 2010:
        fallback = 1.5
    else:
        fallback = 1.0

    def value_for(key: str) -> float:
        try:
            value = float(lowered.get(key, fallback))
            return value if np.isfinite(value) and value > 0.0 else fallback
        except (TypeError, ValueError):
            return fallback

    return value_for("rmsra"), value_for("rmsdec")


def _observation_sigmas(rows: list[tuple[Observation, datetime, float, float]]) -> np.ndarray:
    return np.asarray([
        value
        for item in rows
        for value in _observation_sigma_arcsec(item[0])
    ], dtype=float)


def _representative_indices(epochs: np.ndarray, maximum: int = 120) -> np.ndarray:
    """Sample both observation density and elapsed time over a large data set."""
    values = np.asarray(epochs, dtype=float)
    if values.size <= maximum:
        return np.arange(values.size, dtype=int)
    # Rank sampling represents dense observing runs.  Time sampling prevents
    # thousands of recent measurements from leaving an older, sparse part of
    # the arc represented by only one point (which can make a fit interpolate
    # the chosen rows while failing catastrophically between them).
    rank_count = maximum // 2
    time_count = maximum - rank_count
    rank_indices = np.linspace(0, values.size - 1, rank_count).astype(int)
    targets = np.linspace(values[0], values[-1], time_count)
    right = np.searchsorted(values, targets, side="left")
    right = np.clip(right, 0, values.size - 1)
    left = np.clip(right - 1, 0, values.size - 1)
    time_indices = np.where(
        np.abs(values[right] - targets) < np.abs(values[left] - targets), right, left,
    )
    selected = np.unique(np.concatenate((rank_indices, time_indices))).astype(int)
    # Clustered multi-apparition data can map most uniformly spaced time
    # targets onto the same few edge observations.  Previously a requested
    # 480-row sample of Halley-like data could silently shrink to roughly 260
    # rows. Fill the unused capacity from the remaining rank distribution.
    if selected.size < maximum:
        remaining = np.setdiff1d(
            np.arange(values.size, dtype=int), selected, assume_unique=True,
        )
        needed = min(maximum - selected.size, remaining.size)
        if needed:
            fill_positions = np.linspace(
                0, remaining.size - 1, needed,
            ).round().astype(int)
            selected = np.unique(np.concatenate((selected, remaining[fill_positions])))
    return selected.astype(int)


def initial_orbit_from_observations(
    observations: list[Observation], method: str = "systematic",
    perihelion_distance_au: float | None = None,
    minimum_observations: int = 6,
    return_candidates: bool = False,
) -> OrbitSolution | list[OrbitSolution]:
    """Estimate a heliocentric state from angles and times without an MPC orbit.

    The observed line-of-sight motion supplies the transverse velocity.  A
    bounded search in geocentric range and radial velocity is then refined
    against a representative subset with the same N-body model used by the
    final six-parameter differential correction.  This is an initial-orbit
    estimate, not the final solution.

    ``method="systematic"`` searches topocentric range and range rate.  The
    optional ``method="vaisala"`` instead constrains the state to be close to
    perihelion at a user-supplied perihelion distance.  That second result is a
    Väisälä hypothesis intended only as a seed for the full differential fit.
    """
    if method not in {"systematic", "vaisala"}:
        raise ValueError(f"Neznámá metoda předběžné dráhy: {method}")
    if method == "vaisala" and (
        perihelion_distance_au is None or not np.isfinite(perihelion_distance_au)
        or perihelion_distance_au <= 0.0
    ):
        raise OrbitFitError("Pro Väisälovu dráhu zadejte kladnou perihelovou vzdálenost q.")
    parsed: list[tuple[Observation, datetime, float, float]] = []
    for observation in observations:
        if (
            observation.excluded_from_fit or observation.ra_deg is None
            or observation.dec_deg is None or not observation.time
        ):
            continue
        try:
            parsed.append((
                observation, parse_observation_time(observation.time),
                float(observation.ra_deg), float(observation.dec_deg),
            ))
        except (TypeError, ValueError):
            continue
    parsed.sort(key=lambda item: item[1])
    if len(parsed) < minimum_observations:
        raise OrbitFitError(
            f"Pro toto počáteční určení dráhy je potřeba alespoň "
            f"{minimum_observations} platných měření."
        )

    all_epochs = dynamical_jd([item[1] for item in parsed])
    arc_days = float(all_epochs[-1] - all_epochs[0])
    if arc_days < 0.01:
        raise OrbitFitError(
            "Pozorovací oblouk je příliš krátký pro samostatné počáteční určení dráhy."
        )
    # Never differentiate a time-ordered mixture of sight lines from widely
    # separated observers.  SWAN/PUNCH-style data can interleave SOHO,
    # STEREO-A, low-Earth spacecraft and geocentric measurements; their
    # apparent directions are discontinuous even though each station's own
    # track is smooth.  Use the best-populated single-station track only for
    # the initial angular position/velocity.  All observers still participate
    # in the range search and the final differential correction below.
    station_indices: dict[str, list[int]] = {}
    for index, item in enumerate(parsed):
        station_indices.setdefault(item[0].station or "500", []).append(index)
    anchor_indices = max(
        station_indices.values(),
        key=lambda indices: (
            len(indices), float(all_epochs[indices[-1]] - all_epochs[indices[0]])
        ),
    )
    if len(anchor_indices) < 3:
        anchor_indices = list(range(len(parsed)))
    median_epoch = float(np.median(all_epochs))
    reference_index = min(anchor_indices, key=lambda index: abs(all_epochs[index] - median_epoch))
    reference_epoch = float(all_epochs[reference_index])
    ra = np.radians(np.asarray([item[2] for item in parsed]))
    dec = np.radians(np.asarray([item[3] for item in parsed]))
    directions = np.column_stack((np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)))
    relative_days = all_epochs - reference_epoch

    # A local quadratic fit is markedly more stable than differencing adjacent
    # high-cadence observations, while retaining curvature on multi-day arcs.
    local_count = min(len(anchor_indices), 60)
    local_indices = np.asarray(sorted(
        anchor_indices, key=lambda index: abs(relative_days[index])
    )[:local_count])
    local_times = relative_days[local_indices]
    degree = 2 if np.ptp(local_times) > 0.2 else 1
    coefficients = [np.polyfit(local_times, directions[local_indices, axis], degree) for axis in range(3)]
    line_of_sight = np.asarray([np.polyval(value, 0.0) for value in coefficients])
    line_of_sight /= np.linalg.norm(line_of_sight)
    angular_velocity = np.asarray([
        np.polyval(np.polyder(value), 0.0) for value in coefficients
    ])
    angular_velocity -= line_of_sight * float(np.dot(line_of_sight, angular_velocity))

    reference_row = parsed[reference_index]
    reference_offset = _observation_offsets_gcrs([reference_row])[0]
    dt = 0.01
    earth_ecliptic = earth_heliocentric_ecliptic(np.array([
        reference_epoch - dt, reference_epoch, reference_epoch + dt,
    ]))
    earth_equatorial = _ecliptic_to_equatorial(earth_ecliptic)
    observer_position = earth_equatorial[1] + reference_offset
    observer_velocity = (earth_equatorial[2] - earth_equatorial[0]) / (2.0 * dt)
    supplied_velocity = reference_row[0].raw.get("observer_velocity_gcrs_au_per_day")
    if supplied_velocity is not None:
        try:
            vector = np.asarray(supplied_velocity, dtype=float)
            if vector.shape == (3,) and np.all(np.isfinite(vector)):
                observer_velocity += vector
        except (TypeError, ValueError):
            pass
    else:
        spacecraft_indices = [
            index for index in local_indices
            if parsed[index][0].raw.get("observer_position_gcrs_au") is not None
        ]
        if len(spacecraft_indices) >= 3:
            spacecraft_times = relative_days[spacecraft_indices]
            spacecraft_positions = np.asarray([
                parsed[index][0].raw["observer_position_gcrs_au"]
                for index in spacecraft_indices
            ], dtype=float)
            offset_degree = 2 if np.ptp(spacecraft_times) > 0.2 else 1
            offset_coefficients = [
                np.polyfit(spacecraft_times, spacecraft_positions[:, axis], offset_degree)
                for axis in range(3)
            ]
            observer_position = earth_equatorial[1] + np.asarray([
                np.polyval(value, 0.0) for value in offset_coefficients
            ])
            observer_velocity += np.asarray([
                np.polyval(np.polyder(value), 0.0) for value in offset_coefficients
            ])

    sample_count = min(36, len(parsed))
    sample_indices = np.unique(np.linspace(0, len(parsed) - 1, sample_count).astype(int))
    sample = [parsed[index] for index in sample_indices]
    sample_epochs = all_epochs[sample_indices]
    sample_ra = np.asarray([item[2] for item in sample])
    sample_dec = np.asarray([item[3] for item in sample])
    sample_offsets = _observation_offsets_gcrs(sample)
    sample_sigmas = _observation_sigmas(sample)
    # Every trial state is compared at the same observation epochs.  Reading
    # the DE kernel for Earth's position inside each range/range-rate trial
    # dominated short-arc solve time, even though those positions never
    # change.  Compute them once and reuse them for both light-time and sky
    # direction calculations.
    sample_earth = earth_heliocentric_ecliptic(sample_epochs)
    # The small sample makes each individual trial cheap, but different range
    # branches can be almost indistinguishable on it.  Score the few optimised
    # finalists against the complete arc so a dense multi-station PCCP data
    # set can select the physically correct basin.
    all_offsets = _observation_offsets_gcrs(parsed)
    all_earth = earth_heliocentric_ecliptic(all_epochs)
    all_ra = np.asarray([item[2] for item in parsed])
    all_dec = np.asarray([item[3] for item in parsed])
    all_sigmas = _observation_sigmas(parsed)
    # Initial range/range-rate search needs only the broad basin of attraction.
    # Over a short discovery arc a solar two-body model is sufficient and is
    # far cheaper than evaluating all eight planetary perturbers hundreds of
    # times.  The final differential correction below still uses the full
    # N-body model.
    engine = NBodyIntegrator(
        PLANETS, perturbers=(), relative_tolerance=2e-9, absolute_tolerance=2e-11
    )

    def state_for(parameters: np.ndarray) -> np.ndarray:
        distance = float(np.exp(parameters[0]))
        radial_velocity = float(parameters[1])
        position_equatorial = observer_position + distance * line_of_sight
        velocity_equatorial = (
            observer_velocity + radial_velocity * line_of_sight + distance * angular_velocity
        )
        return np.concatenate((
            _equatorial_to_ecliptic(position_equatorial.reshape(1, 3))[0],
            _equatorial_to_ecliptic(velocity_equatorial.reshape(1, 3))[0],
        ))

    def angular_residual(parameters: np.ndarray) -> np.ndarray:
        try:
            vector = state_for(parameters)
            state = StateVector(reference_epoch, vector[:3], vector[3:])
            propagated = engine.propagate(state, sample_epochs)
            light_time = np.linalg.norm(
                propagated[:, :3] - sample_earth, axis=1
            ) / C_AU_PER_DAY
            emitted = engine.propagate(state, sample_epochs - light_time)
            computed_ra, computed_dec = _equatorial_direction(
                emitted, sample_epochs, sample_offsets, sample_earth,
            )
            delta_ra = (computed_ra - sample_ra + 180.0) % 360.0 - 180.0
            delta_ra *= np.cos(np.radians(sample_dec))
            physical = np.column_stack((delta_ra, computed_dec - sample_dec)).ravel() * 3600.0
            return physical / sample_sigmas
        except (RuntimeError, ValueError, FloatingPointError):
            return np.full(sample_count * 2, 1e9)

    def full_arc_score(parameters: np.ndarray) -> float:
        """Robust RMS on all rows, used only to rank a few range branches."""
        try:
            vector = state_for(parameters)
            state = StateVector(reference_epoch, vector[:3], vector[3:])
            propagated = engine.propagate(state, all_epochs)
            light_time = np.linalg.norm(
                propagated[:, :3] - all_earth, axis=1
            ) / C_AU_PER_DAY
            emitted = engine.propagate(state, all_epochs - light_time)
            computed_ra, computed_dec = _equatorial_direction(
                emitted, all_epochs, all_offsets, all_earth,
            )
            delta_ra = (computed_ra - all_ra + 180.0) % 360.0 - 180.0
            delta_ra *= np.cos(np.radians(all_dec))
            residual = np.column_stack((
                delta_ra, computed_dec - all_dec,
            )).ravel() * 3600.0 / all_sigmas
            pairs = residual.reshape(-1, 2)
            radial = np.linalg.norm(pairs, axis=1)
            # Ignore at most the worst two percent while selecting a basin;
            # actual rejection remains a separate, visible post-fit step.
            cutoff = float(np.quantile(radial, .98)) if len(radial) >= 20 else np.inf
            retained = pairs[radial <= cutoff]
            return float(np.sqrt(np.mean(np.square(retained))))
        except (RuntimeError, ValueError, FloatingPointError):
            return 1e9

    seeds: list[tuple[float, np.ndarray]] = []
    raw_range_seeds: list[np.ndarray] = []
    if method == "systematic":
        for distance in np.geomspace(0.03, 60.0, 14):
            for radial_velocity in (-0.04, 0.0, 0.04):
                parameters = np.array([np.log(distance), radial_velocity])
                residual = angular_residual(parameters)
                seeds.append((float(np.median(np.abs(residual))), parameters))
        raw_range_seeds = [
            np.array([np.log(distance), 0.0])
            for distance in np.geomspace(0.03, 60.0, 32)
        ]
    else:
        # Intersect the observed ray with a heliocentric sphere r=q.  At
        # perihelion the heliocentric radial velocity is zero; solving that
        # scalar condition supplies the corresponding topocentric range rate.
        q = float(perihelion_distance_au)
        projection = float(np.dot(observer_position, line_of_sight))
        discriminant = projection * projection - (
            float(np.dot(observer_position, observer_position)) - q * q
        )
        distances = []
        if discriminant >= 0.0:
            root = float(np.sqrt(discriminant))
            distances = [value for value in (-projection - root, -projection + root) if value > 0.005]
        if not distances:
            minimum_q = float(np.linalg.norm(
                observer_position - projection * line_of_sight
            ))
            raise OrbitFitError(
                "Zadané q neprotíná pozorovaný směr. "
                f"Pro tuto geometrii musí být q alespoň {minimum_q:.3f} AU."
            )
        for distance in distances:
            position = observer_position + distance * line_of_sight
            base_velocity = observer_velocity + distance * angular_velocity
            denominator = float(np.dot(position, line_of_sight))
            radial_velocity = (
                -float(np.dot(position, base_velocity)) / denominator
                if abs(denominator) > 1e-10 else 0.0
            )
            radial_velocity = float(np.clip(radial_velocity, -0.12, 0.12))
            for distance_scale in (0.995, 1.0, 1.005):
                parameters = np.array([np.log(distance * distance_scale), radial_velocity])
                residual = angular_residual(parameters)
                seeds.append((float(np.median(np.abs(residual))), parameters))
    seeds.sort(key=lambda item: item[0])

    def search_residual(parameters: np.ndarray) -> np.ndarray:
        astrometry = angular_residual(parameters)
        vector = state_for(parameters)
        position = vector[:3]
        velocity = vector[3:]
        radius = float(np.linalg.norm(position))
        if method != "vaisala":
            # The admissible region for ordinary Solar-system discoveries is
            # bound or only mildly hyperbolic.  Without this guard a three-point
            # tracklet can be matched by absurd, very-near states with enormous
            # heliocentric excess velocity.
            specific_energy = float(np.dot(velocity, velocity) / 2.0 - MU_SUN / radius)
            return np.append(astrometry, max(0.0, specific_energy - 5e-6) / 1e-6)
        heliocentric_radial_velocity = float(np.dot(position, velocity) / max(radius, 1e-12))
        q = float(perihelion_distance_au)
        # Strong, but finite, constraints retain numerical stability and make
        # explicit that this is a perihelion hypothesis rather than a free fit.
        return np.append(astrometry, (
            (radius - q) / max(1e-6, q * 1e-7),
            heliocentric_radial_velocity / 1e-7,
        ))

    # Do not let several nearly identical grid points consume every nonlinear
    # attempt.  Distinct topocentric ranges represent genuinely different
    # short-arc orbit families and must each get a chance to converge.
    diverse_seeds: list[np.ndarray] = []
    for _, seed in seeds:
        if all(abs(float(seed[0] - item[0])) >= .45 for item in diverse_seeds):
            diverse_seeds.append(seed)
        if len(diverse_seeds) >= 6:
            break
    best: tuple[float, np.ndarray] | None = None
    finalists: list[tuple[float, np.ndarray]] = []
    for seed in diverse_seeds:
        # Preserve genuinely different range families.  With one global
        # distance bound every trial slid into the same nearest local minimum,
        # so the apparent multi-start search was illusory.
        distance_lower = max(np.log(.005), float(seed[0]) - .32)
        distance_upper = min(np.log(120.0), float(seed[0]) + .32)
        result = least_squares(
            search_residual, seed,
            bounds=([distance_lower, -0.12], [distance_upper, 0.12]),
            loss="soft_l1", f_scale=5.0, max_nfev=24,
        )
        score = full_arc_score(result.x)
        if np.isfinite(score):
            finalists.append((score, result.x.copy()))
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, result.x)
    if best is None or best[0] > 3600.0:
        raise OrbitFitError("Z astrometrie se nepodařilo sestavit použitelný počáteční odhad dráhy.")

    designation = next((item[0].designation for item in parsed if item[0].designation), "Vlastní objekt")
    magnitudes = [item[0].magnitude for item in parsed if item[0].magnitude is not None]
    cometary = is_comet_designation(designation)

    def solution_for(parameters: np.ndarray) -> OrbitSolution:
        vector = state_for(parameters)
        try:
            elements = cartesian_to_elements(vector[:3], vector[3:])
        except Exception as exc:
            raise OrbitFitError(
                "Počáteční astrometrická dráha nevede na použitelnou kuželosečku."
            ) from exc
        return OrbitSolution(
            designation=designation, epoch_jd=reference_epoch,
            semi_major_axis_au=elements[0], eccentricity=elements[1],
            inclination_deg=elements[2], node_deg=elements[3],
            arg_peri_deg=elements[4], mean_anomaly_deg=elements[5],
            absolute_magnitude=float(np.median(magnitudes)) if magnitudes else 18.0,
            photometric_exponent=4.0 if cometary else 2.0,
            phase_correction=not cometary,
            provisional=True,
            position_au=tuple(float(value) for value in vector[:3]),
            velocity_au_per_day=tuple(float(value) for value in vector[3:]),
        )

    if return_candidates:
        candidates: list[OrbitSolution] = []
        # Retain distinct heliocentric energy/range families, not repeated
        # numerical variants of the same minimum.
        for _, parameters in sorted(finalists, key=lambda item: item[0]):
            candidate = solution_for(parameters)
            if all(
                abs(candidate.semi_major_axis_au - existing.semi_major_axis_au)
                > max(.15, .08 * abs(existing.semi_major_axis_au))
                for existing in candidates
            ):
                candidates.append(candidate)
            if len(candidates) >= 6:
                break
        if method == "systematic":
            # Raw systematic-ranging states are deliberately retained as
            # seeds for the later six-dimensional correction.  The two-
            # parameter angular-motion model can otherwise deform or discard
            # the correct distant branch before the real orbit fit sees it.
            candidates.extend(
                solution_for(parameters)
                for parameters in sorted(raw_range_seeds, key=full_arc_score)
            )
        return candidates or [solution_for(best[1])]
    return solution_for(best[1])


def _fit_orbit_once(
    observations: list[Observation], initial: OrbitSolution | None,
    constraints: dict[str, float] | None = None,
    truncate_years: float | None = None,
    nongravitational: dict[str, float] | None = None,
    fixed_nongravitational: dict[str, float] | None = None,
    maximum_observations: int | None = 120,
    max_nfev: int | None = None,
    robust_loss: bool = False,
    robust_scale: float = 3.0,
    screening: bool = False,
    cancel_callback: CancelCallback | None = None,
) -> tuple[OrbitSolution, float, int]:
    def check_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise OrbitFitCancelled("Výpočet dráhy byl zrušen uživatelem.")

    check_cancelled()
    derived_initial = initial is None
    if initial is None:
        initial = initial_orbit_from_observations(observations)
    if initial.position_au is None or initial.velocity_au_per_day is None:
        raise OrbitFitError("Chybí počáteční kartézský stav dráhy.")
    parsed_rows: list[tuple[Observation, datetime, float, float]] = []
    for observation in observations:
        observation.used_in_fit = False
        observation.residual_ra_arcsec = None
        observation.residual_dec_arcsec = None
        if observation.ra_deg is None or observation.dec_deg is None or not observation.time:
            continue
        try:
            item = (observation, parse_observation_time(observation.time), observation.ra_deg, observation.dec_deg)
            parsed_rows.append(item)
        except ValueError:
            continue
    parsed_rows.sort(key=lambda item: item[1])
    reportable = _truncate_rows_to_recent_years(parsed_rows, truncate_years)
    usable = [item for item in reportable if not item[0].excluded_from_fit]
    if len(usable) < 6:
        raise OrbitFitError("Pro vlastní fit je potřeba alespoň šest platných astrometrických pozorování.")
    usable.sort(key=lambda item: item[1])
    available_observation_count = len(usable)
    # Bound interactive run time while retaining the full time baseline.
    if maximum_observations is not None and len(usable) > maximum_observations:
        all_usable_epochs = dynamical_jd([item[1] for item in usable])
        indices = _representative_indices(all_usable_epochs, maximum_observations)
        usable = [usable[index] for index in indices]
    usable_times = [item[1] for item in usable]
    epochs = dynamical_jd(usable_times)
    observer_offsets = _observation_offsets_gcrs(usable)
    observed_ra = np.array([item[2] for item in usable])
    observed_dec = np.array([item[3] for item in usable])
    astrometric_sigmas = _observation_sigmas(usable)
    nongrav_names = tuple(
        name for name in ("A1", "A2", "A3", "DT")
        if name in (nongravitational or {})
    )
    initial_nongrav = {
        name: float((initial.nongravitational_parameters or {}).get(name, 0.0))
        for name in nongrav_names
    }
    initial_nongrav.update(nongravitational or {})
    fixed_nongrav = {
        str(name): float(value)
        for name, value in (fixed_nongravitational or {}).items()
        if name not in nongrav_names
    }

    def nongrav_from_vector(vector: np.ndarray) -> dict[str, float]:
        return {
            name: float(vector[6 + index]) * (1.0 if name == "DT" else 1e-8)
            for index, name in enumerate(nongrav_names)
        }

    def force_parameters(vector: np.ndarray) -> dict[str, float]:
        values = dict(fixed_nongrav)
        values.update(nongrav_from_vector(vector))
        return values

    arc_days = float(np.ptp(epochs))
    # On a discovery arc of a few days, repeatedly looking up all eight
    # planets while scipy estimates a numerical Jacobian dominates the run
    # time.  Differential correction can use the solar force field here: the
    # missing, almost constant planetary acceleration is absorbed into the
    # fitted state at the central epoch.  A final correction below always uses
    # the full DE ephemeris, so this shortcut does not become the reported
    # orbit.
    use_fast_differential_correction = arc_days <= 30.0 and not nongrav_names
    if arc_days <= 730.0:
        full_ephemeris = _InterpolatedPlanetEphemeris(
            PLANETS, float(np.min(epochs)) - 2.0, float(np.max(epochs)) + 2.0,
        )
    else:
        cache_start = min(float(np.min(epochs)), float(initial.epoch_jd)) - 2.0
        cache_end = max(float(np.max(epochs)), float(initial.epoch_jd)) + 2.0
        full_ephemeris = SplinePlanetEphemeris(
            PLANETS, _PLANET_NAMES, cache_start, cache_end,
        )
    exact_full_engine = NBodyIntegrator(
        full_ephemeris, relative_tolerance=3e-11, absolute_tolerance=2e-13,
        nongravitational_parameters=initial_nongrav,
    )
    full_engine = NBodyIntegrator(
        full_ephemeris, relative_tolerance=3e-11, absolute_tolerance=2e-13,
        nongravitational_parameters=initial_nongrav,
    )
    if use_fast_differential_correction:
        optimizer_engine = NBodyIntegrator(
            PLANETS, perturbers=(), relative_tolerance=3e-11,
            absolute_tolerance=2e-13,
        )
    elif arc_days > 730.0:
        # Century-scale numerical Jacobians need dozens of integrations.  A
        # relaxed pass finds the correction cheaply; the strict DE440 model
        # below polishes and reports the final state.
        optimizer_engine = NBodyIntegrator(
            full_ephemeris, relative_tolerance=1e-8, absolute_tolerance=1e-10,
            nongravitational_parameters=initial_nongrav,
        )
    else:
        optimizer_engine = full_engine

    def propagate(
        vector: np.ndarray, state: StateVector, output_epochs: np.ndarray,
        engine: NBodyIntegrator,
    ) -> np.ndarray:
        parameters = force_parameters(vector)
        if parameters != engine.nongravitational_parameters:
            engine = NBodyIntegrator(
                engine.ephemeris, perturbers=engine.perturbers,
                relative_tolerance=engine.rtol, absolute_tolerance=engine.atol,
                nongravitational_parameters=parameters,
            )
        return engine.propagate(state, output_epochs)

    reference_epoch = float(np.median(epochs))
    source_state = StateVector(
        initial.epoch_jd, np.asarray(initial.position_au), np.asarray(initial.velocity_au_per_day)
    )
    # Fit the Cartesian state near the middle of the selected observations.
    # This shortens every integration and improves numerical conditioning.
    source_parameters = np.concatenate((
        source_state.packed(),
        np.array([
            initial_nongrav.get(name, 0.0) / (1.0 if name == "DT" else 1e-8)
            for name in nongrav_names
        ]),
    ))
    try:
        initial_state_at_reference = propagate(
            source_parameters, source_state, np.array([reference_epoch]), exact_full_engine,
        )[0]
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise OrbitFitError(
            "Počáteční kombinaci A1/A2/A3/DT se nepodařilo stabilně integrovat. "
            "Zkuste začít s hodnotami parametrů blízkými nule."
        ) from exc
    initial_vector = np.concatenate((
        initial_state_at_reference, source_parameters[6:],
    ))
    earth_at_observations = earth_heliocentric_ecliptic(epochs)

    def astrometric_residuals(
        vector: np.ndarray, engine: NBodyIntegrator = optimizer_engine,
    ) -> np.ndarray:
        state = StateVector(reference_epoch, vector[:3], vector[3:6])
        propagated = propagate(vector, state, epochs, engine)
        if not np.all(np.isfinite(propagated)):
            raise FloatingPointError("Integrace vrátila nekonečný stav.")
        light_time = np.linalg.norm(
            propagated[:, :3] - earth_at_observations, axis=1
        ) / C_AU_PER_DAY
        emitted = propagate(vector, state, epochs - light_time, engine)
        if not np.all(np.isfinite(emitted)):
            raise FloatingPointError("Integrace po opravě o dobu letu selhala.")
        computed_ra, computed_dec = _equatorial_direction(
            emitted, epochs, observer_offsets, earth_at_observations,
        )
        delta_ra = (computed_ra - observed_ra + 180.0) % 360.0 - 180.0
        delta_ra *= np.cos(np.radians(observed_dec))
        delta_dec = computed_dec - observed_dec
        return np.column_stack((delta_ra, delta_dec)).ravel() * 3600.0

    def residuals_with_engine(
        vector: np.ndarray, engine: NBodyIntegrator,
    ) -> np.ndarray:
        check_cancelled()
        try:
            astrometry = astrometric_residuals(vector, engine) / astrometric_sigmas
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            # Numerical least-squares deliberately probes both sides of every
            # parameter. With free A1/A2/A3/DT one such trial can imply a
            # collision with the Sun or absurd outgassing, forcing solve_ivp
            # below machine step size. That trial is a bad point in parameter
            # space, not a reason to abort the otherwise valid fit.
            astrometry = np.full(observed_ra.size * 2, 1e12, dtype=float)
        if not constraints:
            return astrometry
        try:
            elements = cartesian_to_elements(vector[:3], vector[3:6])
        except Exception:
            return np.append(astrometry, np.full(len(constraints), 1e9))
        names = ("a", "e", "i", "node", "arg_peri", "mean_anomaly")
        penalties = []
        for name, target in constraints.items():
            actual = elements[names.index(name)]
            if name in ("i", "node", "arg_peri", "mean_anomaly"):
                difference = (actual - target + 180.0) % 360.0 - 180.0
                penalties.append(difference * 1e5)
            elif name == "e":
                penalties.append((actual - target) * 1e7)
            else:
                penalties.append((actual - target) * 1e6)
        return np.append(astrometry, penalties)

    def residuals(vector: np.ndarray) -> np.ndarray:
        return residuals_with_engine(vector, optimizer_engine)

    fine_scale = np.concatenate((
        np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5]),
        np.ones(len(nongrav_names)),
    ))
    nongrav_lower = np.array([
        -1000.0 if name == "DT" else -1e4 for name in nongrav_names
    ])
    nongrav_upper = np.array([
        1000.0 if name == "DT" else 1e4 for name in nongrav_names
    ])
    lower_bounds = np.concatenate((np.full(6, -np.inf), nongrav_lower))
    upper_bounds = np.concatenate((np.full(6, np.inf), nongrav_upper))

    if derived_initial:
        # The angles-only initializer deliberately favours a broad basin over
        # sub-arcsecond precision.  First pull its six Cartesian components
        # into the local minimum with a wide robust scale; the normal fine fit
        # below then handles individual outliers.
        coarse = least_squares(
            residuals,
            initial_vector,
            method="trf",
            x_scale="jac",
            bounds=(lower_bounds, upper_bounds),
            loss="soft_l1",
            f_scale=100.0,
            max_nfev=60,
            verbose=0,
        )
        initial_vector = coarse.x

    check_cancelled()
    result = least_squares(
        residuals,
        initial_vector,
        method="trf",
        x_scale=("jac" if derived_initial else fine_scale),
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1" if robust_loss else "linear",
        f_scale=float(robust_scale),
        max_nfev=(
            max_nfev if max_nfev is not None
            else 100 if derived_initial else 24
        ),
        verbose=0,
    )
    if use_fast_differential_correction:
        # Polish the fast short-arc solution with the complete planetary force
        # model.  Starting in the correct basin needs only a few evaluations.
        result = least_squares(
            lambda vector: residuals_with_engine(vector, full_engine),
            result.x,
            method="trf", x_scale=fine_scale,
            bounds=(lower_bounds, upper_bounds), loss="linear",
            max_nfev=4 if screening else 8, verbose=0,
        )
    if not result.success and result.nfev < 2:
        raise OrbitFitError(result.message)
    check_cancelled()
    elements = cartesian_to_elements(result.x[:3], result.x[3:6])
    reporting_engine = optimizer_engine if screening else full_engine
    try:
        final_astrometric_residual = astrometric_residuals(result.x, reporting_engine)
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise OrbitFitError(
            "Fit skončil v numericky nestabilní kombinaci A1/A2/A3/DT. "
            "Zkuste vypnout DT nebo některý slabě určený negravitační parametr."
        ) from exc
    rms = float(np.sqrt(np.mean(np.square(final_astrometric_residual))))
    normalized_rms = float(np.sqrt(np.mean(np.square(
        final_astrometric_residual / astrometric_sigmas
    ))))
    if (
        not screening
        and normalized_rms > 100.0
        and rms > 3600.0
    ):
        raise OrbitFitError(
            "Diferenciální korekce se vzdálila od astrometrie o celé stupně "
            f"(RMS {rms:.0f}″; {normalized_rms:.0f}σ). Takové řešení nelze "
            "považovat za určenou dráhu."
        )
    astrometric_jacobian = np.asarray(result.jac[:final_astrometric_residual.size, :], dtype=float)
    if screening:
        uncertainties, state_covariance = None, None
    else:
        uncertainties, state_covariance = _formal_element_uncertainties(
            result.x, elements, astrometric_jacobian,
            final_astrometric_residual / astrometric_sigmas,
            max(1, 6 + len(nongrav_names) - len(constraints or {})), nongrav_names,
        )
    solution = OrbitSolution(
        designation=initial.designation,
        epoch_jd=reference_epoch,
        semi_major_axis_au=elements[0], eccentricity=elements[1], inclination_deg=elements[2],
        node_deg=elements[3], arg_peri_deg=elements[4], mean_anomaly_deg=elements[5],
        absolute_magnitude=initial.absolute_magnitude, slope_g=initial.slope_g,
        photometric_exponent=initial.photometric_exponent,
        phase_correction=initial.phase_correction,
        provisional=True,
        position_au=tuple(float(value) for value in result.x[:3]),
        velocity_au_per_day=tuple(float(value) for value in result.x[3:6]),
        state_covariance=(
            tuple(tuple(float(value) for value in row) for row in state_covariance)
            if state_covariance is not None else None
        ),
        element_uncertainties=uncertainties,
        fit_observation_count=len(usable),
        fit_available_observation_count=available_observation_count,
        fit_rms_arcsec=rms,
        fit_normalized_rms=normalized_rms,
        display_name=initial.display_name,
        nongravitational_parameters=force_parameters(result.x) or None,
    )
    if screening:
        return solution, rms, len(usable)
    # Calculate reportable O-C values for every valid observation, not only for
    # the (at most 120) points retained by the interactive optimizer.
    report_times = [item[1] for item in reportable]
    report_epochs = dynamical_jd(report_times)
    report_offsets = _observation_offsets_gcrs(reportable)
    report_ra = np.array([item[2] for item in reportable])
    report_dec = np.array([item[3] for item in reportable])
    report_state = StateVector(reference_epoch, result.x[:3], result.x[3:6])
    check_cancelled()
    try:
        report_positions = propagate(result.x, report_state, report_epochs, full_engine)
        report_earth = earth_heliocentric_ecliptic(report_epochs)
        report_light_time = np.linalg.norm(
            report_positions[:, :3] - report_earth, axis=1,
        ) / C_AU_PER_DAY
        report_emitted = propagate(
            result.x, report_state, report_epochs - report_light_time, full_engine
        )
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise OrbitFitError(
            "Kontrolní integrace O−C skončila v nestabilní kombinaci "
            "A1/A2/A3/DT; zkuste vypnout nejslaběji určený parametr."
        ) from exc
    computed_ra, computed_dec = _equatorial_direction(
        report_emitted, report_epochs, report_offsets, report_earth,
    )
    observed_minus_computed_ra = (report_ra - computed_ra + 180.0) % 360.0 - 180.0
    observed_minus_computed_ra *= np.cos(np.radians(report_dec)) * 3600.0
    observed_minus_computed_dec = (report_dec - computed_dec) * 3600.0
    fitted_ids = {id(item[0]) for item in usable}
    for item, residual_ra, residual_dec in zip(
        reportable, observed_minus_computed_ra, observed_minus_computed_dec
    ):
        observation = item[0]
        observation.residual_ra_arcsec = float(residual_ra)
        observation.residual_dec_arcsec = float(residual_dec)
        observation.used_in_fit = id(observation) in fitted_ids

    # The RMS shown to the user must describe the same O-C values displayed in
    # the table.  Refuse a numerically inconsistent result instead of offering
    # thousands-of-arcseconds residuals beside a deceptively small fit RMS.
    fitted_report = np.asarray([
        (item[0].residual_ra_arcsec, item[0].residual_dec_arcsec)
        for item in reportable if id(item[0]) in fitted_ids
    ], dtype=float)
    report_rms = float(np.sqrt(np.mean(np.square(fitted_report))))
    if not np.isfinite(report_rms) or not np.isclose(report_rms, rms, rtol=2e-3, atol=2e-3):
        raise OrbitFitError(
            "Kontrola O−C odhalila numericky nekonzistentní fit; původní dráha zůstala zachována."
        )
    return solution, rms, len(usable)


def _normalized_observation_residual(observation: Observation) -> float:
    if observation.residual_ra_arcsec is None or observation.residual_dec_arcsec is None:
        return 0.0
    sigma_ra, sigma_dec = _observation_sigma_arcsec(observation)
    return float(np.hypot(
        observation.residual_ra_arcsec / sigma_ra,
        observation.residual_dec_arcsec / sigma_dec,
    ))


def _is_automatic_rejection_candidate(
    observation: Observation, fit_rms_arcsec: float,
) -> bool:
    """Require both statistical and visibly significant astrometric error."""
    if observation.residual_ra_arcsec is None or observation.residual_dec_arcsec is None:
        return False
    physical_residual = float(np.hypot(
        observation.residual_ra_arcsec, observation.residual_dec_arcsec,
    ))
    sigma_ra, sigma_dec = _observation_sigma_arcsec(observation)
    return bool(
        _normalized_observation_residual(observation) > 3.0
        and physical_residual > max(2.0, 3.0 * max(sigma_ra, sigma_dec))
    )


def _observing_apparition_groups(
    observations: list[Observation], solution: OrbitSolution | None = None,
    gap_years: float = 30.0,
) -> list[list[Observation]]:
    """Split valid astrometry into chronological observing apparitions."""
    dated: list[tuple[datetime, Observation]] = []
    for observation in observations:
        if (
            observation.excluded_from_fit or not observation.time
            or observation.ra_deg is None or observation.dec_deg is None
        ):
            continue
        try:
            dated.append((parse_observation_time(observation.time), observation))
        except ValueError:
            continue
    dated.sort(key=lambda item: item[0])
    if not dated:
        return []

    periodic_comet = bool(
        solution is not None
        and is_comet_designation(solution.designation)
        and 0.0 <= solution.eccentricity < 1.0
        and solution.semi_major_axis_au > 0.0
    )
    if periodic_comet:
        assert solution is not None
        period_days = (
            2.0 * np.pi * solution.semi_major_axis_au ** 1.5 / np.sqrt(MU_SUN)
        )
        if np.isfinite(period_days) and period_days > 0.0:
            epochs = dynamical_jd([item[0] for item in dated])
            mean_motion = 2.0 * np.pi / period_days
            signed_mean_anomaly = np.radians(
                (solution.mean_anomaly_deg + 180.0) % 360.0 - 180.0
            )
            reference_perihelion = (
                solution.epoch_jd - signed_mean_anomaly / mean_motion
            )
            passage_indices = np.floor(
                (epochs - reference_perihelion) / period_days + 0.5
            ).astype(np.int64)
            groups: dict[int, list[Observation]] = {}
            for passage, (_instant, observation) in zip(passage_indices, dated):
                groups.setdefault(int(passage), []).append(observation)
            return [groups[passage] for passage in sorted(groups)]

    minimum_gap = gap_years * 365.25 * 86400.0
    blocks: list[list[Observation]] = []
    start = 0
    for index in range(1, len(dated)):
        if (dated[index][0] - dated[index - 1][0]).total_seconds() >= minimum_gap:
            blocks.append([item[1] for item in dated[start:index]])
            start = index
    blocks.append([item[1] for item in dated[start:]])
    return blocks


def _latest_observing_apparition(
    observations: list[Observation], solution: OrbitSolution | None = None,
    gap_years: float = 30.0, minimum_size: int = 1,
) -> list[Observation]:
    """Return the latest sufficiently populated observing apparition.

    For a bound periodic comet, observations are assigned to the nearest
    perihelion passage using the seed orbit's period.  This works for both
    Halley-type comets and short-period objects such as Encke; a fixed
    decades-long gap does not.  The old gap split remains a deliberately
    conservative fallback when no usable periodic comet orbit is available.
    """
    blocks = _observing_apparition_groups(observations, solution, gap_years)
    if not blocks:
        return []
    for block in reversed(blocks):
        if len(block) >= max(1, minimum_size):
            return block
    return blocks[-1]


def _apparition_linkage_record(
    observations: list[Observation], solution: OrbitSolution,
    status: str = "not_attempted", reason: str = "",
) -> dict[str, object]:
    """Build an auditable summary for one observed comet apparition."""
    instants: list[datetime] = []
    for observation in observations:
        try:
            instants.append(parse_observation_time(observation.time))
        except (TypeError, ValueError):
            continue
    years = [instant.year for instant in instants]
    year_start = min(years) if years else None
    year_end = max(years) if years else None
    return_year = year_start
    if instants and 0.0 <= solution.eccentricity < 1.0 and solution.semi_major_axis_au > 0.0:
        period_days = 2.0 * np.pi * solution.semi_major_axis_au ** 1.5 / np.sqrt(MU_SUN)
        if np.isfinite(period_days) and period_days > 0.0:
            mean_motion = 2.0 * np.pi / period_days
            signed_mean_anomaly = np.radians(
                (solution.mean_anomaly_deg + 180.0) % 360.0 - 180.0
            )
            reference_perihelion = solution.epoch_jd - signed_mean_anomaly / mean_motion
            middle_jd = float(np.median(dynamical_jd(instants)))
            passage = round((middle_jd - reference_perihelion) / period_days)
            try:
                perihelion = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(
                    days=reference_perihelion + passage * period_days - 2451545.0
                )
                return_year = perihelion.year
            except (OverflowError, ValueError):
                pass
    return {
        "return_year": return_year,
        "year_start": year_start,
        "year_end": year_end,
        "available": len(observations),
        "fitted": 0,
        "status": status,
        "reason": reason,
        "rms_arcsec": None,
        "normalized_rms": None,
    }


def _update_apparition_linkage_metrics(
    record: dict[str, object], observations: list[Observation],
) -> None:
    residuals: list[float] = []
    normalized: list[float] = []
    fitted = 0
    for observation in observations:
        if observation.used_in_fit:
            fitted += 1
        if observation.residual_ra_arcsec is None or observation.residual_dec_arcsec is None:
            continue
        residuals.extend((
            float(observation.residual_ra_arcsec),
            float(observation.residual_dec_arcsec),
        ))
        sigma_ra, sigma_dec = _observation_sigma_arcsec(observation)
        normalized.extend((
            float(observation.residual_ra_arcsec) / sigma_ra,
            float(observation.residual_dec_arcsec) / sigma_dec,
        ))
    record["fitted"] = fitted
    if residuals:
        record["rms_arcsec"] = float(np.sqrt(np.mean(np.square(residuals))))
    if normalized:
        record["normalized_rms"] = float(np.sqrt(np.mean(np.square(normalized))))


def _attach_multi_apparition_elements(
    observations: list[Observation], solution: OrbitSolution,
) -> None:
    """Attach observed and next-return osculating elements to a periodic comet."""
    if not is_comet_designation(solution.designation):
        return
    if solution.eccentricity >= 1.0 or solution.semi_major_axis_au <= 0.0:
        return
    times: list[datetime] = []
    for observation in observations:
        if not observation.time:
            continue
        try:
            times.append(parse_observation_time(observation.time))
        except ValueError:
            continue
    if len(times) < 2:
        return
    epochs = dynamical_jd(times)
    period_days = 2.0 * np.pi * solution.semi_major_axis_au ** 1.5 / np.sqrt(MU_SUN)
    # Only create the automatic table when astrometry reaches across a
    # substantial fraction of an orbit.  The normal Evolution tab remains the
    # explicit tool for a single apparition.
    if float(np.ptp(epochs)) < .45 * period_days:
        return
    try:
        from .orbit_evolution import perihelion_returns_between
        # Include the perihelion associated with the first observations even
        # when a recovery arc begins months or years after it.
        start = float(np.min(epochs) - .65 * period_days)
        end = float(np.max(epochs) + 1.10 * period_days)
        returns = perihelion_returns_between(solution, start, end)
        latest_observation = float(np.max(epochs))
        # A close planetary encounter can lengthen the next revolution enough
        # that 1.1 times the current osculating period is insufficient. Extend
        # only when necessary, avoiding a second costly integration normally.
        if not any(item.epoch_jd > latest_observation for item in returns):
            for multiplier in (2.0, 4.0):
                extended = perihelion_returns_between(
                    solution, start,
                    float(np.max(epochs) + multiplier * period_days),
                )
                if extended:
                    returns = extended
                if any(item.epoch_jd > latest_observation for item in returns):
                    break
    except (ValueError, RuntimeError, FloatingPointError):
        return
    latest_observation = float(np.max(epochs))
    solution.apparition_elements = tuple({
        "index": item.index,
        "epoch_jd": item.epoch_jd,
        "q": item.perihelion_au,
        "e": item.eccentricity,
        "a": item.semi_major_axis_au,
        "period_years": item.period_years,
        "i": item.inclination_deg,
        "node": item.node_deg,
        "arg_peri": item.arg_peri_deg,
        "future": item.epoch_jd > latest_observation,
    } for item in returns)
    # Replace period-extrapolated labels in the linkage audit with the actual
    # perihelion years found by the same perturbed N-body integration.  For
    # Halley, simply subtracting today's osculating period labels the 1910
    # return as 1911 because Jupiter changes the period between apparitions.
    if solution.apparition_linkage:
        return_years: list[int] = []
        for item in returns:
            if item.epoch_jd > latest_observation:
                continue
            try:
                instant = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(
                    days=float(item.epoch_jd) - 2451545.0
                )
                return_years.append(instant.year)
            except (OverflowError, ValueError):
                continue
        unused = list(return_years)
        for record in solution.apparition_linkage:
            if not unused or record.get("return_year") is None:
                continue
            approximate = int(record["return_year"])
            actual = min(unused, key=lambda year: abs(year - approximate))
            record["return_year"] = actual
            unused.remove(actual)


def _automatic_orbit_score(solution: OrbitSolution) -> float:
    """Find_Orb-style short-arc score: fit quality plus orbit plausibility."""
    residual_score = float(
        solution.fit_normalized_rms
        if solution.fit_normalized_rms is not None else solution.fit_rms_arcsec or 1e9
    ) / 2.0
    eccentricity = float(solution.eccentricity)
    score = residual_score + eccentricity / 2.0
    if eccentricity > 1.01:
        score += (eccentricity - 1.01) * 1000.0
    return score


def _prefer_parabolic_short_arc(
    observations: list[Observation], candidate: tuple[OrbitSolution, float, int],
    truncate_years: float | None,
    cancel_callback: CancelCallback | None = None,
) -> tuple[OrbitSolution, float, int]:
    """Try Find_Orb's second, near-parabolic branch for a hyperbolic short arc."""
    solution = candidate[0]
    # A parabolic prior is useful only for an ambiguous, mildly hyperbolic
    # discovery orbit.  It is physically wrong for a securely interstellar
    # object such as 3I/ATLAS and must not be attempted on a long arc.
    if solution.eccentricity <= 1.01 or solution.eccentricity >= 1.2:
        return candidate
    valid_times: list[datetime] = []
    for observation in observations:
        if observation.excluded_from_fit or not observation.time:
            continue
        try:
            valid_times.append(parse_observation_time(observation.time))
        except ValueError:
            pass
    if len(valid_times) >= 2 and (max(valid_times) - min(valid_times)).days > 30:
        return candidate

    previous_report = [
        (
            observation.residual_ra_arcsec,
            observation.residual_dec_arcsec,
            observation.used_in_fit,
        )
        for observation in observations
    ]

    def restore_candidate_report() -> None:
        for observation, values in zip(observations, previous_report):
            (
                observation.residual_ra_arcsec,
                observation.residual_dec_arcsec,
                observation.used_in_fit,
            ) = values

    try:
        parabolic = _fit_orbit_once(
            observations, solution, {"e": 1.0}, truncate_years, None,
            cancel_callback=cancel_callback,
        )
    except OrbitFitCancelled:
        raise
    except Exception:
        restore_candidate_report()
        return candidate
    # A physical prior must never excuse a visibly poorer astrometric fit.
    if parabolic[1] > candidate[1] * 1.10:
        restore_candidate_report()
        return candidate
    if _automatic_orbit_score(parabolic[0]) < _automatic_orbit_score(solution):
        return parabolic
    restore_candidate_report()
    return candidate


def fit_orbit(
    observations: list[Observation], initial: OrbitSolution | None,
    constraints: dict[str, float] | None = None,
    truncate_years: float | None = None,
    nongravitational: dict[str, float] | None = None,
    progress_callback: ProgressCallback | None = None,
    cancel_callback: CancelCallback | None = None,
) -> tuple[OrbitSolution, float, int]:
    """Fit an orbit with conservative automatic rejection of astrometric blunders."""
    progress_step = 0
    progress_total = 0

    def check_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise OrbitFitCancelled("Výpočet dráhy byl zrušen uživatelem.")

    def report_progress(label: str, total: int | None = None) -> None:
        nonlocal progress_step, progress_total
        check_cancelled()
        if total is not None:
            progress_total = max(int(total), progress_step + 1)
        progress_step += 1
        if progress_total > 0 and progress_step > progress_total:
            progress_total = progress_step
        if progress_callback is not None:
            progress_callback(label, progress_step, progress_total)

    def fit_once(*args, **kwargs):
        kwargs.setdefault("cancel_callback", cancel_callback)
        return _fit_orbit_once(*args, **kwargs)

    report_progress("Připravuji astrometrii a počáteční dráhu…")
    active_times: list[datetime] = []
    for observation in observations:
        if (
            observation.excluded_from_fit or not observation.time
            or observation.ra_deg is None or observation.dec_deg is None
        ):
            continue
        try:
            active_times.append(parse_observation_time(observation.time))
        except ValueError:
            pass
    short_arc = bool(
        len(active_times) >= 2
        and (max(active_times) - min(active_times)).total_seconds() <= 30.0 * 86400.0
    )
    arc_days = (
        (max(active_times) - min(active_times)).total_seconds() / 86400.0
        if len(active_times) >= 2 else 0.0
    )
    long_arc = arc_days >= 20.0 * 365.25
    active_count = len(active_times)
    # Range is intrinsically weakly constrained on a discovery arc.  This is
    # true for a 20-observation PCCP object just as much as for a 400-row one,
    # and an existing preliminary/MPC seed may itself sit in the wrong local
    # basin.  Use several range branches whenever a six-parameter solution is
    # possible; the cost naturally stays lower for the smaller data sets.
    multibranch = bool(
        short_arc and active_count >= 6
        and not constraints and not nongravitational
    )
    multi_apparition = False
    multi_apparition_seed: tuple[OrbitSolution, float, int] | None = None
    seed_report: list[tuple[float | None, float | None, bool | None]] | None = None
    apparition_groups: list[list[Observation]] = []
    apparition_linkage: list[dict[str, object]] = []
    if multibranch:
        initial_candidates = initial_orbit_from_observations(
            observations, return_candidates=True,
        )
        assert isinstance(initial_candidates, list)
        if initial is not None:
            # Treat a supplied preliminary/catalogue orbit as a useful extra
            # hypothesis, never as the only branch allowed to converge.
            initial_candidates.insert(0, initial)
        screened: list[tuple[float, OrbitSolution]] = []
        # Six locally optimised hypotheses followed by the twelve raw ranges
        # with the best complete-arc score cover the useful systematic region
        # without making every hypothesis pay for a full DE440 correction.
        for candidate in initial_candidates[:19 if initial is not None else 18]:
            try:
                report_progress("Prověřuji kandidátní větev krátkého oblouku…", 24)
                trial = fit_once(
                    observations, candidate, constraints, truncate_years,
                    nongravitational, maximum_observations=36,
                    max_nfev=14, robust_loss=True, screening=True,
                )
                if np.isfinite(trial[1]):
                    # Screening ranks the basin only.  Preserve its original
                    # range seed for the full correction; the intentionally
                    # abbreviated screening fit can otherwise shift a weak
                    # short-arc solution across a basin boundary.
                    screened.append((trial[1], candidate))
            except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
                continue
        if not screened:
            raise OrbitFitError(
                "Systematické rangování nenašlo použitelnou větev dráhy."
            )
        finalists: list[tuple[
            tuple[OrbitSolution, float, int],
            list[tuple[float | None, float | None, bool | None]],
        ]] = []
        for _, candidate in sorted(screened, key=lambda item: item[0])[:3]:
            try:
                report_progress("Zpřesňuji nejlepší větev na všech měřeních…")
                final_result = fit_once(
                    observations, candidate, constraints, truncate_years,
                    nongravitational, maximum_observations=None,
                    max_nfev=40, robust_loss=True,
                )
                finalists.append((final_result, [
                    (
                        observation.residual_ra_arcsec,
                        observation.residual_dec_arcsec,
                        observation.used_in_fit,
                    )
                    for observation in observations
                ]))
            except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
                continue
        if not finalists:
            raise OrbitFitError(
                "Úplná korekce kandidátních drah selhala."
            )
        result, selected_report = min(
            finalists, key=lambda item: item[0][1]
        )
        # Each finalist computes its own O-C table.  Restore the table that
        # belongs to the winning solution rather than accidentally leaving
        # values from whichever candidate happened to run last.
        for observation, report_values in zip(observations, selected_report):
            (
                observation.residual_ra_arcsec,
                observation.residual_dec_arcsec,
                observation.used_in_fit,
            ) = report_values
    else:
        # Always establish residuals before any automatic rejection.  A
        # chronologically isolated follow-up batch is not itself evidence of
        # bad astrometry; only actual post-fit residuals may exclude it.
        periodic_comet = bool(
            initial is not None
            and is_comet_designation(initial.designation)
            and 0.0 <= initial.eccentricity < 1.0
            and initial.semi_major_axis_au > 0.0
        )
        apparition_groups = (
            [
                group for group in _observing_apparition_groups(observations, initial)
                if len(group) >= 6
            ]
            if periodic_comet else []
        )
        apparition_linkage = [
            _apparition_linkage_record(group, initial)
            for group in apparition_groups
        ]
        recent_apparition = apparition_groups[-1] if apparition_groups else []
        multi_apparition = bool(
            periodic_comet and len(apparition_groups) >= 2
            and 6 <= len(recent_apparition) < active_count
        )
        if multi_apparition and nongravitational:
            # Worst-case plan: recent seed and polish, then three propagated
            # attempts, one local astrometric seed and three local attempts
            # for every older observed apparition, followed by reporting.
            progress_total = max(
                progress_step + 2,
                3 + 7 * (len(apparition_groups) - 1),
            )
        if (
            multi_apparition and not nongravitational
        ):
            # A single gravity-only correction cannot faithfully join active
            # comet apparitions separated by decades: it otherwise invents a
            # compromise state with huge systematic O-C values. Fit the latest
            # apparition and use planetary integration only to predict the
            # other returns. Enabling A1/A2/A3 below opts into a genuinely
            # linked multi-apparition solution.
            for observation in observations:
                observation.used_in_fit = False
                observation.residual_ra_arcsec = None
                observation.residual_dec_arcsec = None
            report_progress("Fituji poslední pozorovaný návrat…", 3)
            result = fit_once(
                recent_apparition, initial, constraints, truncate_years,
                None, maximum_observations=120,
                max_nfev=8, robust_loss=True,
            )
            result[0].fit_model_note = (
                "Bez A1–A3 byl fit určen z posledního pozorovaného návratu; "
                "starší a budoucí návraty jsou čistě gravitační integrace, "
                "nikoli společný fit všech zjevení."
            )
            for record in apparition_linkage:
                record["status"] = "integrated_only"
                record["reason"] = "A1–A3 nebyly zapnuty; návrat nebyl společně fitován."
            if apparition_linkage:
                apparition_linkage[-1]["status"] = "seed_only"
                apparition_linkage[-1]["reason"] = "Samostatný fit posledního návratu."
                _update_apparition_linkage_metrics(
                    apparition_linkage[-1], recent_apparition,
                )
        elif multi_apparition:
            # First lock onto the best-observed modern apparition.  Correcting
            # all six state components across a century immediately forces
            # every numerical Jacobian evaluation to repeat that full span.
            seed_nongrav = (
                {
                    name: value for name, value in (nongravitational or {}).items()
                    if name != "DT"
                }
                if "DT" in (nongravitational or {}) else nongravitational
            )
            report_progress("Určuji lokální dráhu posledního návratu…")
            seed = fit_once(
                recent_apparition, initial, constraints, truncate_years,
                None, fixed_nongravitational=seed_nongrav,
                maximum_observations=120,
                max_nfev=8, robust_loss=True,
            )
            # A four-parameter comet model may need more than the quick seed
            # pass even within the modern apparition.  Only pay for a denser
            # polishing pass when the first result is visibly worse than the
            # expected CCD astrometry; this specifically prevents carrying
            # 8–18 arcsecond 1986 residuals into the historical linkage.
            seed_normalized = seed[0].fit_normalized_rms
            if (
                seed[1] > 3.0
                or seed_normalized is None
                or not np.isfinite(seed_normalized)
                or seed_normalized > 3.0
            ):
                unpolished_report = [
                    (
                        observation.residual_ra_arcsec,
                        observation.residual_dec_arcsec,
                        observation.used_in_fit,
                    )
                    for observation in observations
                ]
                polished_nongrav = dict(seed_nongrav or {})
                polished_nongrav.update({
                    name: value
                    for name, value in (
                        seed[0].nongravitational_parameters or {}
                    ).items()
                    if name in polished_nongrav
                })
                try:
                    report_progress("Zpřesňuji lokální dráhu posledního návratu…")
                    polished = fit_once(
                        recent_apparition, seed[0], constraints, truncate_years,
                        None, fixed_nongravitational=polished_nongrav,
                        maximum_observations=240,
                        max_nfev=12, robust_loss=True,
                    )
                    if polished[1] < seed[1]:
                        seed = polished
                    else:
                        for observation, values in zip(
                            observations, unpolished_report,
                        ):
                            (
                                observation.residual_ra_arcsec,
                                observation.residual_dec_arcsec,
                                observation.used_in_fit,
                            ) = values
                except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
                    for observation, values in zip(
                        observations, unpolished_report,
                    ):
                        (
                            observation.residual_ra_arcsec,
                            observation.residual_dec_arcsec,
                            observation.used_in_fit,
                        ) = values
            # The dedicated A1/A2/A3 pass below performs the one expensive
            # full-arc correction. Avoid doing the same century integration
            # twice merely to manufacture another intermediate seed.
            result = seed
            multi_apparition_seed = seed
            seed_report = [
                (
                    observation.residual_ra_arcsec,
                    observation.residual_dec_arcsec,
                    observation.used_in_fit,
                )
                for observation in observations
            ]
            if apparition_linkage:
                apparition_linkage[-1]["status"] = "seed_only"
                apparition_linkage[-1]["reason"] = (
                    "Výchozí samostatný fit posledního návratu; čeká na navázání starších."
                )
                _update_apparition_linkage_metrics(
                    apparition_linkage[-1], recent_apparition,
                )
        else:
            report_progress("Provádím diferenciální korekci dráhy…", 3)
            result = fit_once(
                observations, initial, constraints, truncate_years,
                nongravitational, maximum_observations=120,
            )
    if not constraints and not nongravitational:
        result = (
            _prefer_parabolic_short_arc(
                observations, result, truncate_years, cancel_callback,
            )
            if cancel_callback is not None else
            _prefer_parabolic_short_arc(observations, result, truncate_years)
        )
    # For an ordinary six-parameter orbit the representative subset keeps the
    # interactive calculation fast. A1/A2/A3 are far more sensitive to the
    # complete time/station distribution, so a nongravitational solution uses
    # the subset only as a seed and is finally corrected on every enabled row.
    if nongravitational:
        # Preserve A1/A2/A3 found by the recent-apparition seed. In particular,
        # DT has no derivative while all acceleration components are zero; a
        # staged A-fit followed by A+DT is much better conditioned than asking
        # the optimiser to discover all four parameters simultaneously.
        def fitted_nongrav(seed_solution: OrbitSolution) -> dict[str, float]:
            values = dict(nongravitational)
            values.update({
                name: value
                for name, value in (
                    seed_solution.nongravitational_parameters or {}
                ).items()
                if name in values
            })
            return values

        if multi_apparition_seed is not None and seed_report is not None:
            # Link one earlier return at a time.  Jumping directly from the
            # 1986 state to a simultaneous 1835+1910+1986 correction leaves a
            # numerical Jacobian thousands of sigma from its local basin and
            # was the source of the spurious degree-sized Halley residuals.
            last_good = multi_apparition_seed
            last_good_report = seed_report
            linked_count = 1
            for first_group in range(len(apparition_groups) - 2, -1, -1):
                stage_observations = [
                    observation
                    for group in apparition_groups[first_group:]
                    for observation in group
                ]
                stage_count = len(apparition_groups) - first_group
                all_parameters = fitted_nongrav(last_good[0])
                # A3 and especially DT are strongly correlated with the state
                # on only two apparitions.  Observatory-grade comet software
                # therefore grows the model as the arc grows instead of
                # releasing every parameter in the first century-spanning
                # correction.  All user-selected parameters are restored once
                # enough returns constrain them.
                progressive_names = (
                    ("A1", "A2") if stage_count <= 2 else
                    ("A1", "A2", "A3") if stage_count == 3 else
                    ("A1", "A2", "A3", "DT")
                )
                progressive = {
                    name: all_parameters[name]
                    for name in progressive_names if name in all_parameters
                }
                nongrav_attempts: list[tuple[dict[str, float], int, float]] = []

                def add_attempt(values: dict[str, float], evaluations: int, scale: float) -> None:
                    key = tuple(sorted(values))
                    if values and all(tuple(sorted(item[0])) != key for item in nongrav_attempts):
                        nongrav_attempts.append((values, evaluations, scale))

                add_attempt(progressive, 28, 25.0)
                add_attempt({
                    name: value for name, value in all_parameters.items()
                    if name != "DT"
                }, 36, 50.0)
                add_attempt(all_parameters, 48, 100.0)
                candidate = None
                candidate_source = "propagated"
                failures: list[str] = []
                try:
                    for parameters, evaluations, scale in nongrav_attempts:
                        try:
                            return_year = apparition_linkage[first_group].get(
                                "return_year", "?"
                            )
                            report_progress(
                                f"Navazuji návrat {return_year}: "
                                f"společný fit {'+'.join(parameters)}…"
                            )
                            trial = fit_once(
                                stage_observations, last_good[0], constraints,
                                truncate_years, parameters,
                                maximum_observations=min(480, 120 * stage_count),
                                max_nfev=evaluations, robust_loss=True,
                                robust_scale=scale,
                            )
                            normalized = trial[0].fit_normalized_rms
                            if (
                                np.isfinite(trial[1])
                                and trial[1] <= 60.0
                                and normalized is not None
                                and np.isfinite(normalized)
                                and normalized <= 8.0
                            ):
                                candidate = trial
                                break
                            normalized_text = (
                                "nedostupné" if normalized is None
                                or not np.isfinite(normalized) else f"{normalized:.2f}σ"
                            )
                            failures.append(
                                f"{'+'.join(parameters)}: RMS {trial[1]:.2f}″; "
                                f"{normalized_text}"
                            )
                        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError) as exc:
                            failures.append(f"{'+'.join(parameters)}: {exc}")
                    # A propagated solution can be only tens of arcseconds
                    # from an older apparition and still lie outside the
                    # narrow basin in which a century-scale numerical
                    # Jacobian converges.  This is common after one or more
                    # unobserved revolutions.  Determine an independent local
                    # state from that apparition's own astrometry, then retry
                    # the same joint correction from that second branch.  The
                    # local state is never accepted as the answer by itself:
                    # every linked return must still pass the common O-C test.
                    if candidate is None:
                        local_force = {
                            name: value for name, value in all_parameters.items()
                            if name != "DT"
                        }
                        try:
                            return_year = apparition_linkage[first_group].get(
                                "return_year", "?"
                            )
                            report_progress(
                                f"Návrat {return_year}: určuji nezávislou "
                                "lokální dráhu z astrometrie…"
                            )
                            local_seed = fit_once(
                                apparition_groups[first_group], last_good[0],
                                constraints, truncate_years, None,
                                fixed_nongravitational=local_force,
                                maximum_observations=120, max_nfev=36,
                                robust_loss=True, robust_scale=25.0,
                            )
                            failures.append(
                                "lokální astrometrická větev: "
                                f"RMS {local_seed[1]:.2f}″"
                            )
                            for parameters, evaluations, scale in nongrav_attempts:
                                try:
                                    report_progress(
                                        f"Návrat {return_year}: společný fit z "
                                        "lokální astrometrické větve "
                                        f"({'+'.join(parameters)})…"
                                    )
                                    trial = fit_once(
                                        stage_observations, local_seed[0], constraints,
                                        truncate_years, parameters,
                                        maximum_observations=min(
                                            480, 120 * stage_count,
                                        ),
                                        max_nfev=max(evaluations, 48),
                                        robust_loss=True, robust_scale=scale,
                                    )
                                    normalized = trial[0].fit_normalized_rms
                                    if (
                                        np.isfinite(trial[1])
                                        and trial[1] <= 60.0
                                        and normalized is not None
                                        and np.isfinite(normalized)
                                        and normalized <= 8.0
                                    ):
                                        candidate = trial
                                        candidate_source = "local_astrometry"
                                        break
                                    normalized_text = (
                                        "nedostupné" if normalized is None
                                        or not np.isfinite(normalized)
                                        else f"{normalized:.2f}σ"
                                    )
                                    failures.append(
                                        "lokální větev "
                                        f"{'+'.join(parameters)}: RMS "
                                        f"{trial[1]:.2f}″; {normalized_text}"
                                    )
                                except (
                                    OrbitFitError, ValueError, RuntimeError,
                                    FloatingPointError,
                                ) as exc:
                                    failures.append(
                                        "lokální větev "
                                        f"{'+'.join(parameters)}: {exc}"
                                    )
                        except (
                            OrbitFitError, ValueError, RuntimeError,
                            FloatingPointError,
                        ) as exc:
                            failures.append(
                                f"lokální astrometrická větev: {exc}"
                            )
                    if candidate is None:
                        detail = "; ".join(failures[-3:]) or "bez použitelného pokusu"
                        raise OrbitFitError(
                            "Vícefázové připojení staršího návratu neprošlo "
                            f"kontrolou kvality ({detail})."
                        )
                except (OrbitFitError, ValueError, RuntimeError, FloatingPointError) as exc:
                    if apparition_linkage:
                        apparition_linkage[first_group]["status"] = "failed"
                        apparition_linkage[first_group]["reason"] = str(exc)
                        _update_apparition_linkage_metrics(
                            apparition_linkage[first_group],
                            apparition_groups[first_group],
                        )
                        for older_index in range(first_group):
                            apparition_linkage[older_index]["status"] = "not_attempted"
                            apparition_linkage[older_index]["reason"] = (
                                "Nezkoušeno, protože navázání následujícího návratu selhalo."
                            )
                    result = last_good
                    for observation, report_values in zip(
                        observations, last_good_report,
                    ):
                        (
                            observation.residual_ra_arcsec,
                            observation.residual_dec_arcsec,
                            observation.used_in_fit,
                        ) = report_values
                    result[0].fit_model_note = (
                        f"Společná korekce pro {linked_count} z "
                        f"{len(apparition_groups)} pozorovaných návratů prošla; "
                        "další starší návrat neprošel kontrolou kvality a je "
                        "zobrazen pouze jako planetárně integrovaná předpověď."
                    )
                    break
                last_good = candidate
                linked_count += 1
                for group_index in range(first_group, len(apparition_groups)):
                    apparition_linkage[group_index]["status"] = "linked"
                    apparition_linkage[group_index]["reason"] = (
                        "Společný N-body fit; počáteční větev připojeného "
                        "návratu byla určena nezávisle z jeho astrometrie."
                        if candidate_source == "local_astrometry" else
                        "Společný N-body fit s ostatními označenými návraty."
                    )
                    _update_apparition_linkage_metrics(
                        apparition_linkage[group_index], apparition_groups[group_index],
                    )
                last_good_report = [
                    (
                        observation.residual_ra_arcsec,
                        observation.residual_dec_arcsec,
                        observation.used_in_fit,
                    )
                    for observation in observations
                ]
                result = candidate
            else:
                result[0].fit_model_note = (
                    f"Společný N-body fit propojil všech "
                    f"{len(apparition_groups)} pozorovaných návratů postupnou "
                    "vícepočáteční diferenciální korekcí z lokálních "
                    "astrometrických řešení."
                )
        else:
            report_progress("Zpřesňuji negravitační řešení na celém oblouku…")
            result = fit_once(
                observations, result[0], constraints, truncate_years,
                fitted_nongrav(result[0]),
                maximum_observations=480 if long_arc else None,
                max_nfev=8 if long_arc else 12, robust_loss=True,
            )
    elif not multibranch and not constraints and short_arc and active_count > 120:
        # A representative subset is a fast seed, not a sufficient final
        # answer for a dense PCCP/NEOCP discovery arc.  Competing range
        # solutions can differ by only hundredths of an arcsecond on a subset
        # yet imply entirely different periods and perihelia.  All rows decide
        # the branch and make the reported observation count honest.
        report_progress("Zpřesňuji řešení na všech měřeních…")
        result = fit_once(
            observations, result[0], constraints, truncate_years, nongravitational,
            maximum_observations=None, max_nfev=36, robust_loss=True,
        )
    # Traditional post-fit rejection remains useful for isolated blunders
    # which do not form a separated final batch.  Remove at most one pass so a
    # weak preliminary orbit cannot cascade into wholesale data rejection.
    if not constraints:
        candidates = [
            item for item in observations
            if not item.excluded_from_fit
            and _is_automatic_rejection_candidate(item, result[1])
        ]
        active_count = sum(
            not item.excluded_from_fit
            and item.residual_ra_arcsec is not None
            and item.residual_dec_arcsec is not None
            for item in observations
        )
        # A bad provisional solution must not erase a large fraction of the
        # arc in one automatic pass. Keep at least 95% and remove the worst
        # normalized residuals first; the user can inspect/reject more later.
        maximum_rejections = max(1, int(np.ceil(active_count * .05)))
        outliers = sorted(
            candidates, key=_normalized_observation_residual, reverse=True,
        )[:maximum_rejections]
        if outliers and active_count - len(outliers) >= 6:
            for observation in outliers:
                observation.excluded_from_fit = True
                observation.raw["automatic_exclusion"] = True
            # A second century-scale differential correction costs another
            # complete numerical Jacobian. Keep the robust long-arc solution
            # and merely flag its clear outliers; the next explicit user fit
            # starts without them. Shorter arcs are cheap enough to refit now.
            if long_arc:
                note = (
                    f"{len(outliers)} zjevně odlehlých měření bylo označeno "
                    "k vyřazení; kvůli délce oblouku se projeví až při dalším "
                    "spuštění fitu."
                )
                result[0].fit_model_note = " ".join(filter(None, (
                    result[0].fit_model_note, note,
                )))
            else:
                report_progress("Přepočítávám dráhu po vyřazení odlehlých měření…")
                result = fit_once(
                    observations, result[0], constraints, truncate_years, nongravitational,
                    maximum_observations=(
                        None if nongravitational or (short_arc and not constraints)
                        else 120
                    ),
                    max_nfev=12,
                )
                if not nongravitational:
                    result = (
                        _prefer_parabolic_short_arc(
                            observations, result, truncate_years,
                            cancel_callback,
                        )
                        if cancel_callback is not None else
                        _prefer_parabolic_short_arc(
                            observations, result, truncate_years,
                        )
                    )
    report_progress("Dokončuji O−C, nejistoty a fotometrii…")
    _calibrate_fitted_photometry(observations, result[0])
    if apparition_linkage:
        result[0].apparition_linkage = tuple(apparition_linkage)
    _attach_multi_apparition_elements(observations, result[0])
    return result
