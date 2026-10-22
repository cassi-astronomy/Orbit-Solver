from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Callable

import numpy as np
from scipy.optimize import least_squares

from .astrometry import (
    is_deprecated, is_fit_candidate, mark_duplicate_measurements, normal_place_observations,
    observation_sigma_arcsec, observation_sigmas, parse_observation_time,
)
from .domain import Observation, OrbitSolution, is_comet_designation
from .i18n import tr
from .mpc_orbit import cartesian_to_elements
from .observatories import (
    dynamical_jd, earth_heliocentric_ecliptic, geodetic_offsets_gcrs,
    station_offsets_gcrs,
)
from .orbit_engine import (
    LONG_ARC_DAYS, MU_SUN, STANDARD_TOLERANCES, NBodyIntegrator, StateVector,
    tolerances_for_span,
)
from .planetary import PLANETS, SplinePlanetEphemeris
from .photometry import estimate_absolute_magnitude, hg_phase_correction

__all__ = [
    "OrbitFitCancelled", "OrbitFitError", "fit_orbit", "fit_selection_count",
    "initial_orbit_from_observations", "parse_observation_time",
]

C_AU_PER_DAY = 173.1446326846693
NONGRAVITATIONAL_NAMES = ("A1", "A2", "A3", tr("DT"))
# Optimizer units: A1–A3 are fitted in units of 1e-8 AU/day², DT in days.
_NONGRAV_SCALE = {"A1": 1e-8, "A2": 1e-8, "A3": 1e-8, tr("DT"): 1.0}
_PLANET_NAMES = (
    "mercury", "venus", "earth", "moon", "mars",
    "jupiter", "saturn", "uranus", "neptune", "pluto",
)

# Backwards-compatible private names used across the application and tests.
_observation_sigma_arcsec = observation_sigma_arcsec
_observation_sigmas = observation_sigmas
_normal_place_observations = normal_place_observations


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



class _InterpolatedPlanetEphemeris(SplinePlanetEphemeris):
    """Dense (3-hour) planetary cache for a single apparition."""

    def __init__(self, source, start_jd: float, stop_jd: float, step_days: float = 0.125):
        super().__init__(source, _PLANET_NAMES, start_jd, stop_jd, step_days)


def _dynamics_engine(
    start_jd: float, end_jd: float,
    nongravitational_parameters: dict[str, float] | None = None,
) -> NBodyIntegrator:
    """Return the force model used for every fit and validation over a span.

    The planetary cache and the integration tolerances both follow the span
    (TDB Julian dates): a dense cache and standard tolerances inside one
    apparition, a daily cache and PRECISE tolerances for multi-year arcs.
    """
    start = float(min(start_jd, end_jd)) - 2.0
    end = float(max(start_jd, end_jd)) + 2.0
    span = end - start
    ephemeris = (
        _InterpolatedPlanetEphemeris(PLANETS, start, end)
        if span <= LONG_ARC_DAYS else
        SplinePlanetEphemeris(PLANETS, _PLANET_NAMES, start, end)
    )
    rtol, atol = tolerances_for_span(span)
    return NBodyIntegrator(
        ephemeris, relative_tolerance=rtol, absolute_tolerance=atol,
        nongravitational_parameters=nongravitational_parameters,
    )


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
        if not is_fit_candidate(observation):
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
            raise ValueError(tr("Nelze určit směr perihelu."))
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
        jacobian = np.asarray(astrometric_jacobian, dtype=float)
        if jacobian.ndim != 2 or jacobian.shape[0] < jacobian.shape[1]:
            return None, None
        # Forming pinv(J.T @ J) squares the condition number and used to turn
        # an effectively unconstrained direction into a deceptively small
        # formal error.  Normalize columns, inspect rank explicitly, and form
        # the covariance from the SVD of J itself.  When a parameter is not
        # identifiable, showing no formal uncertainty is safer than a false
        # one (especially for T and A1--A3 on short arcs).
        column_norms = np.linalg.norm(jacobian, axis=0)
        if np.any(~np.isfinite(column_norms)) or np.any(column_norms <= 0.0):
            return None, None
        normalized_jacobian = jacobian / column_norms
        _u, singular_values, vt = np.linalg.svd(
            normalized_jacobian, full_matrices=False,
        )
        if not singular_values.size or singular_values[0] <= 0.0:
            return None, None
        tolerance = (
            np.finfo(float).eps * max(normalized_jacobian.shape)
            * singular_values[0]
        )
        rank = int(np.count_nonzero(singular_values > tolerance))
        condition = float(singular_values[0] / singular_values[-1])
        if rank < jacobian.shape[1] or not np.isfinite(condition) or condition > 1e12:
            return None, None
        normalized_covariance = (
            vt.T @ np.diag(1.0 / np.square(singular_values)) @ vt
        )
        inverse_column_scale = np.diag(1.0 / column_norms)
        parameter_covariance = (
            inverse_column_scale @ normalized_covariance @ inverse_column_scale
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
            scale = 1.0 if name == tr("DT") else 1e-8
            uncertainties[name] = float(
                np.sqrt(max(0.0, parameter_covariance[index, index])) * scale
            )
        return uncertainties, state_covariance
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError):
        return None, None


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


def _adaptive_finite_difference_jacobian(
    function: Callable[[np.ndarray], np.ndarray], vector: np.ndarray,
    steps: np.ndarray, lower_bounds: np.ndarray, upper_bounds: np.ndarray,
    astrometric_size: int,
) -> np.ndarray:
    """Numerical Jacobian with Find_Orb-style observable-sized perturbations.

    A fixed Cartesian perturbation is a poor derivative step over several
    revolutions: the same velocity change can move a one-week prediction by a
    fraction of a milliarcsecond and a century prediction by degrees.  Adapt
    each persistent step until its largest normalized astrometric effect is of
    order unity.  This keeps the derivative above integrator/round-off noise
    without leaving the local tangent region.
    """
    point = np.asarray(vector, dtype=float)
    base = np.asarray(function(point), dtype=float)
    jacobian = np.empty((base.size, point.size), dtype=float)
    for column in range(point.size):
        step = max(abs(float(steps[column])), np.finfo(float).eps)
        difference = None
        signed_step = step
        for _attempt in range(10):
            signed_step = step
            if point[column] + signed_step > upper_bounds[column]:
                signed_step = -step
            if point[column] + signed_step < lower_bounds[column]:
                signed_step = step * .5
            trial_point = point.copy()
            trial_point[column] += signed_step
            trial = np.asarray(function(trial_point), dtype=float)
            difference = trial - base
            astrometric_change = float(np.max(np.abs(
                difference[:astrometric_size]
            )))
            if np.isfinite(astrometric_change) and .05 <= astrometric_change <= 3.0:
                break
            if not np.isfinite(astrometric_change) or astrometric_change <= 1e-15:
                scale = .2 if not np.isfinite(astrometric_change) else 5.0
            else:
                scale = float(np.clip(1.0 / astrometric_change, .2, 5.0))
            step = max(step * scale, np.finfo(float).eps)
        assert difference is not None
        steps[column] = step
        derivative = difference / signed_step
        jacobian[:, column] = np.where(np.isfinite(derivative), derivative, 0.0)
    return jacobian


def _local_light_time_states(
    propagated: np.ndarray, observer_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Move reception-time states to emission time without a second long integration.

    Light time is at most hours for ordinary Solar-System astrometry.  A local
    second-order expansion using solar acceleration is therefore far below the
    measurement noise, while avoiding another decades-long N-body propagation
    in every numerical-Jacobian evaluation.  The final reported O-C table is
    still recomputed with an independent full integration.
    """
    states = np.asarray(propagated, dtype=float)
    observers = np.asarray(observer_positions, dtype=float)
    light_time = np.linalg.norm(states[:, :3] - observers, axis=1) / C_AU_PER_DAY
    position = states[:, :3]
    radius = np.linalg.norm(position, axis=1)
    acceleration = -MU_SUN * position / np.maximum(radius[:, None] ** 3, 1e-30)
    emitted = states.copy()
    emitted[:, :3] = (
        position - states[:, 3:] * light_time[:, None]
        + .5 * acceleration * np.square(light_time)[:, None]
    )
    return emitted, light_time


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
        raise ValueError(tr("Neznámá metoda předběžné dráhy: {0}", method))
    if method == "vaisala" and (
        perihelion_distance_au is None or not np.isfinite(perihelion_distance_au)
        or perihelion_distance_au <= 0.0
    ):
        raise OrbitFitError(tr("Pro Väisälovu dráhu zadejte kladnou perihelovou vzdálenost q."))
    parsed: list[tuple[Observation, datetime, float, float]] = []
    for observation in observations:
        if not is_fit_candidate(observation):
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
            tr(
                "Pro toto počáteční určení dráhy je potřeba alespoň {0} platných měření.",
                minimum_observations,
            )
        )

    all_epochs = dynamical_jd([item[1] for item in parsed])
    arc_days = float(all_epochs[-1] - all_epochs[0])
    if arc_days < 0.01:
        raise OrbitFitError(
            tr("Pozorovací oblouk je příliš krátký pro samostatné počáteční určení dráhy.")
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
                tr(
                    "Zadané q neprotíná pozorovaný směr. Pro tuto geometrii musí být q "
                    "alespoň {0:.3f} AU.",
                    minimum_q,
                )
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
        raise OrbitFitError(tr(
            "Z astrometrie se nepodařilo sestavit použitelný počáteční odhad dráhy.",
        ))

    designation = next((item[0].designation for item in parsed if item[0].designation), tr(
        "Vlastní objekt",
    ))
    magnitudes = [item[0].magnitude for item in parsed if item[0].magnitude is not None]
    cometary = is_comet_designation(designation)

    def solution_for(parameters: np.ndarray) -> OrbitSolution:
        vector = state_for(parameters)
        try:
            elements = cartesian_to_elements(vector[:3], vector[3:])
        except Exception as exc:
            raise OrbitFitError(
                tr("Počáteční astrometrická dráha nevede na použitelnou kuželosečku.")
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
    """One differential correction of the Cartesian state (and free A1–A3/DT).

    ``nongravitational`` names the force parameters to fit (with their seed
    values); ``fixed_nongravitational`` holds force-law values kept constant.
    The state is fitted at the median epoch of the selected observations and
    every integration uses the tolerances returned by
    :func:`~orbit_app.orbit_engine.tolerances_for_span`, so a multi-decade arc
    is never corrected with a model whose numerical error exceeds the
    astrometric noise.  ``screening`` only skips the formal uncertainty and the
    full O-C report (used to rank short-arc range branches).
    """
    def check_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise OrbitFitCancelled(tr("Výpočet dráhy byl zrušen uživatelem."))

    check_cancelled()
    derived_initial = initial is None
    if initial is None:
        initial = initial_orbit_from_observations(observations)
    if initial.position_au is None or initial.velocity_au_per_day is None:
        raise OrbitFitError(tr("Chybí počáteční kartézský stav dráhy."))
    parsed_rows: list[tuple[Observation, datetime, float, float]] = []
    for observation in observations:
        observation.used_in_fit = False
        observation.residual_ra_arcsec = None
        observation.residual_dec_arcsec = None
        if observation.ra_deg is None or observation.dec_deg is None or not observation.time:
            continue
        try:
            parsed_rows.append((
                observation, parse_observation_time(observation.time),
                observation.ra_deg, observation.dec_deg,
            ))
        except ValueError:
            continue
    parsed_rows.sort(key=lambda item: item[1])
    reportable = _truncate_rows_to_recent_years(parsed_rows, truncate_years)
    usable = [item for item in reportable if is_fit_candidate(item[0])]
    if len(usable) < 6:
        raise OrbitFitError(tr(
            "Pro vlastní fit je potřeba alespoň šest platných astrometrických "
            "pozorování.",
        ))
    available_observation_count = len(usable)
    # Bound interactive run time while retaining the full time baseline.
    if maximum_observations is not None and len(usable) > maximum_observations:
        all_usable_epochs = dynamical_jd([item[1] for item in usable])
        indices = _representative_indices(all_usable_epochs, maximum_observations)
        usable = [usable[index] for index in indices]
    epochs = dynamical_jd([item[1] for item in usable])
    observer_offsets = _observation_offsets_gcrs(usable)
    observed_ra = np.array([item[2] for item in usable])
    observed_dec = np.array([item[3] for item in usable])
    astrometric_sigmas = _observation_sigmas(usable, account_for_batch_correlation=True)
    nongrav_names = tuple(
        name for name in NONGRAVITATIONAL_NAMES if name in (nongravitational or {})
    )
    initial_nongrav = {
        name: float((initial.nongravitational_parameters or {}).get(name, 0.0))
        for name in nongrav_names
    }
    initial_nongrav.update({
        name: float(value) for name, value in (nongravitational or {}).items()
        if name in nongrav_names
    })
    # Distance-law constants are fixed model metadata, not fitted dimensions.
    # Preserve them through every differential-correction pass; otherwise an
    # asteroid selected as r^-2 silently reverts to the Marsden comet law.
    fixed_nongrav = {
        str(name): float(value)
        for source in (nongravitational or {}, fixed_nongravitational or {})
        for name, value in source.items()
        if name not in nongrav_names
    }

    def nongrav_from_vector(vector: np.ndarray) -> dict[str, float]:
        return {
            name: float(vector[6 + index]) * _NONGRAV_SCALE[name]
            for index, name in enumerate(nongrav_names)
        }

    def force_parameters(vector: np.ndarray) -> dict[str, float]:
        values = dict(fixed_nongrav)
        values.update(nongrav_from_vector(vector))
        return values

    arc_days = float(np.ptp(epochs))
    span_start = min(float(np.min(epochs)), float(initial.epoch_jd))
    span_end = max(float(np.max(epochs)), float(initial.epoch_jd))
    full_engine = _dynamics_engine(span_start, span_end, {**fixed_nongrav, **initial_nongrav})
    # On a discovery arc of a few days, repeatedly looking up all planets while
    # scipy estimates a numerical Jacobian dominates the run time.  The almost
    # constant planetary acceleration is absorbed into the fitted state; a
    # polish with the complete model below always follows.
    use_fast_differential_correction = arc_days <= 30.0 and not nongrav_names
    optimizer_engine = (
        NBodyIntegrator(
            PLANETS, perturbers=(), relative_tolerance=STANDARD_TOLERANCES[0],
            absolute_tolerance=STANDARD_TOLERANCES[1],
        )
        if use_fast_differential_correction else full_engine
    )

    def engine_for(vector: np.ndarray, engine: NBodyIntegrator) -> NBodyIntegrator:
        parameters = force_parameters(vector)
        if parameters == engine.nongravitational_parameters:
            return engine
        return engine.with_parameters(parameters)

    reference_epoch = float(np.median(epochs))
    source_state = StateVector(
        initial.epoch_jd, np.asarray(initial.position_au), np.asarray(initial.velocity_au_per_day)
    )
    source_parameters = np.concatenate((
        source_state.packed(),
        np.array([initial_nongrav.get(name, 0.0) / _NONGRAV_SCALE[name] for name in nongrav_names]),
    ))
    # Fit the Cartesian state near the middle of the selected observations.
    # This shortens every integration and improves numerical conditioning.
    try:
        initial_state_at_reference = engine_for(source_parameters, full_engine).propagate(
            source_state, np.array([reference_epoch]),
        )[0]
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise OrbitFitError(
            tr(
                "Počáteční kombinaci A1/A2/A3/DT se nepodařilo stabilně integrovat. "
                "Zkuste začít s hodnotami parametrů blízkými nule.",
            )
        ) from exc
    initial_vector = np.concatenate((initial_state_at_reference, source_parameters[6:]))
    earth_at_observations = earth_heliocentric_ecliptic(epochs)

    def projected(states: np.ndarray) -> np.ndarray:
        emitted, _light_time = _local_light_time_states(states, earth_at_observations)
        computed_ra, computed_dec = _equatorial_direction(
            emitted, epochs, observer_offsets, earth_at_observations,
        )
        delta_ra = (computed_ra - observed_ra + 180.0) % 360.0 - 180.0
        delta_ra *= np.cos(np.radians(observed_dec))
        return np.column_stack((delta_ra, computed_dec - observed_dec)).ravel() * 3600.0

    def astrometric_residuals(vector: np.ndarray, engine: NBodyIntegrator) -> np.ndarray:
        state = StateVector(reference_epoch, vector[:3], vector[3:6])
        propagated = engine_for(vector, engine).propagate(state, epochs)
        if not np.all(np.isfinite(propagated)):
            raise FloatingPointError(tr("Integrace vrátila nekonečný stav."))
        return projected(propagated)

    element_names = ("a", "e", "i", "node", "arg_peri", "mean_anomaly")

    def constraint_values(vector: np.ndarray) -> np.ndarray:
        """Strong quadratic penalties holding user-fixed osculating elements."""
        if not constraints:
            return np.empty(0, dtype=float)
        try:
            elements = cartesian_to_elements(vector[:3], vector[3:6])
        except Exception:
            return np.full(len(constraints), 1e9)
        penalties = []
        for name, target in constraints.items():
            actual = elements[element_names.index(name)]
            if name in ("i", "node", "arg_peri", "mean_anomaly"):
                penalties.append(((actual - target + 180.0) % 360.0 - 180.0) * 1e5)
            elif name == "e":
                penalties.append((actual - target) * 1e7)
            else:
                penalties.append((actual - target) * 1e6)
        return np.asarray(penalties, dtype=float)

    residual_size = observed_ra.size * 2 + len(constraints or {})

    def residuals_with_engine(vector: np.ndarray, engine: NBodyIntegrator) -> np.ndarray:
        check_cancelled()
        try:
            astrometry = astrometric_residuals(vector, engine) / astrometric_sigmas
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            # Numerical least-squares deliberately probes both sides of every
            # parameter. With free A1/A2/A3/DT one such trial can imply a
            # collision with the Sun or absurd outgassing.  That trial is a bad
            # point in parameter space, not a reason to abort the fit.
            astrometry = np.full(observed_ra.size * 2, 1e12, dtype=float)
        return np.append(astrometry, constraint_values(vector))

    use_variational_jacobian = arc_days > 365.25
    variational_cache: dict[str, np.ndarray | None] = {
        "vector": None, "residual": None, "jacobian": None,
    }

    def variational_evaluation(vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Residuals plus one-pass variational derivatives (state and A/DT)."""
        cached = variational_cache["vector"]
        if cached is not None and np.array_equal(np.asarray(cached), vector):
            return np.asarray(variational_cache["residual"]), np.asarray(variational_cache["jacobian"])
        check_cancelled()
        working_engine = engine_for(vector, optimizer_engine)
        state = StateVector(reference_epoch, vector[:3], vector[3:6])
        propagated, state_partials = working_engine.propagate_with_partials(
            state, epochs, nongrav_names,
            np.asarray([_NONGRAV_SCALE[name] for name in nongrav_names]),
        )
        astrometry = projected(propagated)
        column_steps = np.concatenate((
            np.array([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]),
            np.array([.05 if name == tr("DT") else 5e-3 for name in nongrav_names]),
        ))
        jacobian = np.empty((astrometry.size, 6 + len(nongrav_names)), dtype=float)
        # Only the cheap sky projection is differenced; the expensive dynamical
        # derivatives came from the same augmented integration.  Size the
        # perturbation from the propagated partial so it stays local (a few
        # arcseconds at one AU) even across many revolutions.
        for column, nominal_step in enumerate(column_steps):
            sensitivity = float(np.max(np.linalg.norm(state_partials[:, :3, column], axis=1)))
            step = float(nominal_step)
            if np.isfinite(sensitivity) and sensitivity > 0.0:
                step = min(step, 1e-5 / sensitivity)
            step = max(step, 1e-12)
            displacement = state_partials[:, :, column] * step
            jacobian[:, column] = (
                projected(propagated + displacement) - projected(propagated - displacement)
            ) / (2.0 * step)
        residual_values = astrometry / astrometric_sigmas
        jacobian_values = jacobian / astrometric_sigmas[:, None]
        if constraints:
            penalties = constraint_values(vector)
            penalty_jacobian = np.zeros((len(penalties), vector.size), dtype=float)
            for column in range(6):
                plus = vector.copy()
                minus = vector.copy()
                plus[column] += column_steps[column]
                minus[column] -= column_steps[column]
                penalty_jacobian[:, column] = (
                    constraint_values(plus) - constraint_values(minus)
                ) / (2.0 * column_steps[column])
            residual_values = np.append(residual_values, penalties)
            jacobian_values = np.vstack((jacobian_values, penalty_jacobian))
        variational_cache.update(vector=vector.copy(), residual=residual_values, jacobian=jacobian_values)
        return residual_values, jacobian_values

    def optimizer_residuals(vector: np.ndarray) -> np.ndarray:
        if use_variational_jacobian:
            try:
                return variational_evaluation(vector)[0]
            except (RuntimeError, ValueError, FloatingPointError, OverflowError):
                return np.full(residual_size, 1e12, dtype=float)
        return residuals_with_engine(vector, optimizer_engine)

    fine_scale = np.concatenate((
        np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5]), np.ones(len(nongrav_names)),
    ))
    # Keep numerical derivatives away from physically absurd trial forces.
    # The scaled A limit of 100 means 1e-6 AU/day^2, already far above the
    # usual Marsden parameters, while still admitting a larger supplied seed.
    nongrav_limits = np.array([
        max(365.0, abs(initial_nongrav.get(name, 0.0)) * 2.0 + 30.0) if name == tr("DT")
        else max(100.0, abs(initial_nongrav.get(name, 0.0)) / 1e-8 * 10.0 + 10.0)
        for name in nongrav_names
    ])
    lower_bounds = np.concatenate((np.full(6, -np.inf), -nongrav_limits))
    upper_bounds = np.concatenate((np.full(6, np.inf), nongrav_limits))
    derivative_steps = np.concatenate((
        np.array([1e-5, 1e-5, 1e-5, 1e-9, 1e-9, 1e-9]),
        np.array([.1 if name == tr("DT") else 1e-3 for name in nongrav_names]),
    ))

    def optimizer_jacobian(vector: np.ndarray) -> np.ndarray:
        if use_variational_jacobian:
            return variational_evaluation(vector)[1]
        return _adaptive_finite_difference_jacobian(
            optimizer_residuals, vector, derivative_steps,
            lower_bounds, upper_bounds, observed_ra.size * 2,
        )

    # Over a multi-year arc, generic finite-difference steps can either vanish
    # in integration noise or move the prediction far outside the local phase
    # branch; Find_Orb's full_improvement() sizes every perturbation by its
    # astrometric effect, and so does the Jacobian above.
    jacobian_method = optimizer_jacobian if arc_days > 365.25 else "2-point"
    if derived_initial:
        # The angles-only initializer favours a broad basin over precision.
        # Pull it into the local minimum with a wide robust scale first.
        coarse = least_squares(
            optimizer_residuals, initial_vector, jac=jacobian_method, method="trf",
            x_scale="jac", bounds=(lower_bounds, upper_bounds),
            loss="soft_l1", f_scale=100.0, max_nfev=60,
        )
        initial_vector = coarse.x

    check_cancelled()
    result = least_squares(
        optimizer_residuals, initial_vector, jac=jacobian_method, method="trf",
        x_scale=("jac" if derived_initial else fine_scale),
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1" if robust_loss else "linear", f_scale=float(robust_scale),
        max_nfev=max_nfev if max_nfev is not None else 100 if derived_initial else 24,
    )
    if use_fast_differential_correction:
        # Polish the fast short-arc solution with the complete planetary force
        # model.  Starting in the correct basin needs only a few evaluations.
        result = least_squares(
            lambda vector: residuals_with_engine(vector, full_engine), result.x,
            method="trf", x_scale=fine_scale, bounds=(lower_bounds, upper_bounds),
            loss="linear", max_nfev=4 if screening else 8,
        )
    if not result.success and result.nfev < 2:
        raise OrbitFitError(result.message)
    result_vector = result.x
    check_cancelled()
    elements = cartesian_to_elements(result_vector[:3], result_vector[3:6])
    try:
        final_astrometric_residual = astrometric_residuals(result_vector, full_engine)
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise OrbitFitError(
            tr(
                "Fit skončil v numericky nestabilní kombinaci A1/A2/A3/DT. Zkuste vypnout "
                "DT nebo některý slabě určený negravitační parametr.",
            )
        ) from exc
    rms = float(np.sqrt(np.mean(np.square(final_astrometric_residual))))
    normalized_rms = float(np.sqrt(np.mean(np.square(
        final_astrometric_residual / astrometric_sigmas
    ))))
    if not screening and normalized_rms > 100.0 and rms > 3600.0:
        raise OrbitFitError(
            tr(
                "Diferenciální korekce se vzdálila od astrometrie o celé stupně (RMS "
                "{0:.0f}″; {1:.0f}σ). Takové řešení nelze považovat za určenou dráhu.",
                rms,
                normalized_rms,
            )
        )
    if screening:
        uncertainties, state_covariance = None, None
    else:
        if use_variational_jacobian:
            final_jacobian = variational_evaluation(result_vector)[1]
        else:
            final_jacobian = np.asarray(result.jac, dtype=float)
        uncertainties, state_covariance = _formal_element_uncertainties(
            result_vector, elements, final_jacobian[:final_astrometric_residual.size, :],
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
        position_au=tuple(float(value) for value in result_vector[:3]),
        velocity_au_per_day=tuple(float(value) for value in result_vector[3:6]),
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
        nongravitational_parameters=force_parameters(result_vector) or None,
    )
    if screening:
        return solution, rms, len(usable)
    # Report O-C for every valid observation, not only the optimizer subset,
    # with an independent integration and iterated light time.
    report_rows = list(reportable)
    report_epochs = dynamical_jd([item[1] for item in report_rows])
    report_offsets = _observation_offsets_gcrs(report_rows)
    report_ra = np.array([item[2] for item in report_rows])
    report_dec = np.array([item[3] for item in report_rows])
    report_state = StateVector(reference_epoch, result_vector[:3], result_vector[3:6])
    report_engine = engine_for(result_vector, full_engine)
    check_cancelled()
    try:
        report_positions = report_engine.propagate(report_state, report_epochs)
        report_earth = earth_heliocentric_ecliptic(report_epochs)
        report_light_time = np.linalg.norm(
            report_positions[:, :3] - report_earth, axis=1,
        ) / C_AU_PER_DAY
        report_emitted = report_engine.propagate(report_state, report_epochs - report_light_time)
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise OrbitFitError(
            tr(
                "Kontrolní integrace O−C skončila v nestabilní kombinaci A1/A2/A3/DT; "
                "zkuste vypnout nejslaběji určený parametr.",
            )
        ) from exc
    computed_ra, computed_dec = _equatorial_direction(
        report_emitted, report_epochs, report_offsets, report_earth,
    )
    residual_ra = (report_ra - computed_ra + 180.0) % 360.0 - 180.0
    residual_ra *= np.cos(np.radians(report_dec)) * 3600.0
    residual_dec = (report_dec - computed_dec) * 3600.0
    fitted_ids = {id(item[0]) for item in usable}
    for item, delta_ra, delta_dec in zip(report_rows, residual_ra, residual_dec):
        observation = item[0]
        observation.residual_ra_arcsec = float(delta_ra)
        observation.residual_dec_arcsec = float(delta_dec)
        observation.used_in_fit = id(observation) in fitted_ids
    # The RMS shown to the user must describe the same O-C values displayed in
    # the table.  Refuse a numerically inconsistent result instead of offering
    # large residuals beside a deceptively small fit RMS.
    fitted_report = np.asarray([
        (item[0].residual_ra_arcsec, item[0].residual_dec_arcsec)
        for item in report_rows if id(item[0]) in fitted_ids
    ], dtype=float)
    report_rms = float(np.sqrt(np.mean(np.square(fitted_report))))
    if not np.isfinite(report_rms) or not np.isclose(report_rms, rms, rtol=2e-3, atol=2e-3):
        raise OrbitFitError(
            tr(
                "Kontrola O−C odhalila numericky nekonzistentní fit; původní dráha "
                "zůstala zachována.",
            )
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


def _coherent_secondary_track_candidates(
    observations: list[Observation],
) -> list[Observation]:
    """Find a repeated offset track beside a well-fitted nominal track.

    Fragment/secondary-nucleus astrometry is sometimes returned by the MPC
    under the same designation as the primary object. It is not an isolated
    blunder: several measurements from one station/night form a second,
    internally precise O-C cluster. Requiring a simultaneous near-zero cluster
    keeps this distinct from rejecting a whole night because an orbit is poor.
    """
    groups: dict[tuple[str, str], list[Observation]] = {}
    for observation in observations:
        if (
            observation.excluded_from_fit
            or observation.residual_ra_arcsec is None
            or observation.residual_dec_arcsec is None
            or not observation.station or not observation.time
        ):
            continue
        try:
            night = parse_observation_time(observation.time).date().isoformat()
        except ValueError:
            continue
        groups.setdefault((observation.station.strip().upper(), night), []).append(
            observation
        )

    selected: list[Observation] = []
    for group in groups.values():
        if len(group) < 6:
            continue
        vectors = np.asarray([
            (float(item.residual_ra_arcsec), float(item.residual_dec_arcsec))
            for item in group
        ])
        radii = np.linalg.norm(vectors, axis=1)
        near = radii <= 3.0
        far = radii >= 8.0
        if int(np.count_nonzero(near)) < 3 or int(np.count_nonzero(far)) < 3:
            continue
        far_vectors = vectors[far]
        centre = np.median(far_vectors, axis=0)
        centre_distance = float(np.linalg.norm(centre))
        cluster_radius = max(2.5, 0.08 * centre_distance)
        distances = np.linalg.norm(far_vectors - centre, axis=1)
        if centre_distance < 8.0 or float(np.median(distances)) > cluster_radius:
            continue
        far_indices = np.flatnonzero(far)
        members = far_indices[distances <= max(4.0, 0.15 * centre_distance)]
        if len(members) < 3:
            continue
        selected.extend(group[int(index)] for index in members)
    return list({id(item): item for item in selected}.values())


def _is_weighted(observation: Observation) -> bool:
    """Enabled row that carries weight (not a duplicate or deprecated record)."""
    return bool(
        not observation.excluded_from_fit
        and not observation.raw.get("duplicate_measurement")
        and not is_deprecated(observation)
    )


def _orbital_period_days(solution: OrbitSolution | None) -> float | None:
    if solution is None:
        return None
    a = float(solution.semi_major_axis_au)
    if not (0.0 <= solution.eccentricity < 1.0) or not np.isfinite(a) or a <= 0.0:
        return None
    return float(2.0 * np.pi * a ** 1.5 / np.sqrt(MU_SUN))


# Shortest observing gap that separates two apparitions (days): longer than the
# seasonal solar-conjunction gap within one apparition.
MINIMUM_APPARITION_GAP_DAYS = 300.0


def _observing_apparition_groups(
    observations: list[Observation], solution: OrbitSolution | None = None,
) -> list[list[Observation]]:
    """Split valid astrometry into chronological observing apparitions.

    A new apparition starts after an observing gap of at least half the
    orbital period of ``solution`` and never less than
    ``MINIMUM_APPARITION_GAP_DAYS``, so the seasonal conjunction gaps of one
    apparition do not split it.  The rule is deliberately independent of the
    seed's osculating period at other epochs: extrapolating a perihelion time
    through Jupiter encounters can put a boundary in the middle of an
    apparition.  Without a bound orbit the minimum gap alone is used.
    """
    dated: list[tuple[datetime, Observation]] = []
    for observation in observations:
        if not is_fit_candidate(observation):
            continue
        try:
            dated.append((parse_observation_time(observation.time), observation))
        except ValueError:
            continue
    dated.sort(key=lambda item: item[0])
    if not dated:
        return []
    period_days = _orbital_period_days(solution)
    minimum_gap_days = (
        max(MINIMUM_APPARITION_GAP_DAYS, 0.5 * period_days) if period_days is not None
        else MINIMUM_APPARITION_GAP_DAYS
    )
    blocks: list[list[Observation]] = []
    start = 0
    for index in range(1, len(dated)):
        gap = (dated[index][0] - dated[index - 1][0]).total_seconds() / 86400.0
        if gap >= minimum_gap_days:
            blocks.append([item[1] for item in dated[start:index]])
            start = index
    blocks.append([item[1] for item in dated[start:]])
    return blocks


def _apparition_linkage_record(
    observations: list[Observation], solution: OrbitSolution | None,
    status: str = "not_attempted", reason: str = "",
) -> dict[str, object]:
    """Build an auditable summary for one observed apparition."""
    instants: list[datetime] = []
    for observation in observations:
        try:
            instants.append(parse_observation_time(observation.time))
        except (TypeError, ValueError):
            continue
    years = [instant.year for instant in instants]
    year_start = min(years) if years else None
    year_end = max(years) if years else None
    return {
        # Median TDB epoch of the apparition; used to label it with the
        # nearest integrated perihelion.
        "epoch_jd": float(np.median(dynamical_jd(instants))) if instants else None,
        # Replaced by the integrated perihelion year once a solution exists.
        "return_year": year_start,
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
        if not _is_weighted(observation):
            continue
        if observation.used_in_fit:
            fitted += 1
        if observation.residual_ra_arcsec is None or observation.residual_dec_arcsec is None:
            continue
        residuals.extend((
            float(observation.residual_ra_arcsec), float(observation.residual_dec_arcsec),
        ))
        sigma_ra, sigma_dec = _observation_sigma_arcsec(observation)
        normalized.extend((
            float(observation.residual_ra_arcsec) / sigma_ra,
            float(observation.residual_dec_arcsec) / sigma_dec,
        ))
    record["fitted"] = fitted
    record["rms_arcsec"] = float(np.sqrt(np.mean(np.square(residuals)))) if residuals else None
    record["normalized_rms"] = (
        float(np.sqrt(np.mean(np.square(normalized)))) if normalized else None
    )


def _evaluate_solution_observations(
    observations: list[Observation], solution: OrbitSolution,
    nongravitational: dict[str, float] | None = None,
    cancel_callback: CancelCallback | None = None,
) -> tuple[float, float, int]:
    """Propagate an existing solution and populate O-C without fitting it.

    Every row with a valid time and position receives O-C (``used_in_fit`` is
    cleared); the returned physical and normalized RMS describe the fit
    candidates only.
    """
    if solution.position_au is None or solution.velocity_au_per_day is None:
        raise OrbitFitError(tr("Chybí počáteční kartézský stav dráhy."))
    parsed: list[tuple[Observation, datetime, float, float]] = []
    for observation in observations:
        if observation.ra_deg is None or observation.dec_deg is None or not observation.time:
            continue
        try:
            parsed.append((
                observation, parse_observation_time(observation.time),
                float(observation.ra_deg), float(observation.dec_deg),
            ))
        except (TypeError, ValueError):
            continue
    if not parsed:
        raise OrbitFitError(tr("Pro kontrolu dráhy nejsou dostupná platná měření."))
    parsed.sort(key=lambda item: item[1])
    if cancel_callback is not None and cancel_callback():
        raise OrbitFitCancelled(tr("Výpočet dráhy byl zrušen uživatelem."))
    epochs = dynamical_jd([item[1] for item in parsed])
    force_parameters = dict(solution.nongravitational_parameters or {})
    force_parameters.update(nongravitational or {})
    engine = _dynamics_engine(
        min(float(np.min(epochs)), float(solution.epoch_jd)),
        max(float(np.max(epochs)), float(solution.epoch_jd)),
        force_parameters,
    )
    source = StateVector(
        solution.epoch_jd, np.asarray(solution.position_au, dtype=float),
        np.asarray(solution.velocity_au_per_day, dtype=float),
    )
    try:
        received = engine.propagate(source, epochs)
        earth = earth_heliocentric_ecliptic(epochs)
        light_time = np.linalg.norm(received[:, :3] - earth, axis=1) / C_AU_PER_DAY
        emitted = engine.propagate(source, epochs - light_time)
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise OrbitFitError(
            tr("Kontrolní propagace dráhy skončila numerickou chybou.")
        ) from exc
    offsets = _observation_offsets_gcrs(parsed)
    computed_ra, computed_dec = _equatorial_direction(emitted, epochs, offsets, earth)
    observed_ra = np.asarray([item[2] for item in parsed])
    observed_dec = np.asarray([item[3] for item in parsed])
    residual_ra = (observed_ra - computed_ra + 180.0) % 360.0 - 180.0
    residual_ra *= np.cos(np.radians(observed_dec)) * 3600.0
    residual_dec = (observed_dec - computed_dec) * 3600.0
    normalized: list[float] = []
    physical: list[float] = []
    for item, delta_ra, delta_dec in zip(parsed, residual_ra, residual_dec):
        observation = item[0]
        observation.residual_ra_arcsec = float(delta_ra)
        observation.residual_dec_arcsec = float(delta_dec)
        observation.used_in_fit = False
        if not is_fit_candidate(observation):
            continue
        sigma_ra, sigma_dec = _observation_sigma_arcsec(observation)
        physical.extend((float(delta_ra), float(delta_dec)))
        normalized.extend((float(delta_ra) / sigma_ra, float(delta_dec) / sigma_dec))
    if not physical:
        return float("nan"), float("nan"), 0
    return (
        float(np.sqrt(np.mean(np.square(physical)))),
        float(np.sqrt(np.mean(np.square(normalized)))),
        len(physical) // 2,
    )


def _attach_multi_apparition_elements(
    observations: list[Observation], solution: OrbitSolution,
) -> None:
    """Attach the observed-arc evolution and per-return osculating elements.

    One integration from the first observed apparition to the first return
    after today supplies the perihelion table, the planetary encounters, the
    continuous element history and the trajectory.  The Evolution tab shows it
    directly, so the user does not have to integrate the same span again.
    """
    if not (is_comet_designation(solution.designation) or solution.apparition_linkage):
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
    now_jd = float(dynamical_jd([datetime.now(timezone.utc)])[0])
    try:
        from .orbit_evolution import calculate_orbit_evolution_between
        # Include the perihelion associated with the first observations even
        # when a recovery arc begins months or years after it, and integrate
        # far enough to include the first perihelion after today (the first
        # unobserved return of an old recovery may already be in the past).
        start = float(np.min(epochs) - .65 * period_days)
        future_reference = max(float(np.max(epochs)), now_jd)
        evolution = None
        # A close planetary encounter can lengthen the next revolution beyond
        # 1.1 current periods; extend only when necessary.
        for multiplier in (1.10, 2.0, 4.0):
            evolution = calculate_orbit_evolution_between(
                solution, start, float(future_reference + multiplier * period_days),
            )
            if any(item.epoch_jd > future_reference for item in evolution.returns):
                break
    except (ValueError, RuntimeError, FloatingPointError):
        return
    returns = evolution.returns
    solution.observed_evolution = replace(
        evolution,
        model_note=(
            tr(
                "Integrace spočtená při vlastním fitu přes pozorované návraty až po první "
                "návrat po dnešku; planetární perturbace a použitý silový model jsou "
                "stejné jako ve fitu.",
            )
        ),
    )
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
        "state": tuple(float(value) for value in item.state),
        "future": item.epoch_jd > now_jd,
    } for item in returns)
    # Label every apparition with the integrated perihelion nearest to its
    # median epoch.  Each record is matched independently: aphelion-only
    # detections (78P in 2014 and 2015) must not shift the other labels, and a
    # perturbed period cannot mislabel Halley's 1910 return as 1911.
    for record in solution.apparition_linkage or ():
        epoch = record.get("epoch_jd")
        if epoch is None or not returns:
            continue
        nearest = min(returns, key=lambda item: abs(item.epoch_jd - float(epoch)))
        try:
            record["return_year"] = (
                datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
                + timedelta(days=float(nearest.epoch_jd) - 2451545.0)
            ).year
        except (OverflowError, ValueError):
            continue


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
    """Fit an orbit to astrometry.

    Three regimes are distinguished:

    * a short discovery arc (≤ 30 days) is solved from several range branches;
    * one observed apparition (or a selected recent window) is solved by one
      differential correction, followed by conservative blunder rejection;
    * astrometry split into several apparitions by long observing gaps is
      linked by :func:`orbit_app.linkage.link_apparitions` (line-of-variations
      branch search with precise integration, for any object type).

    ``nongravitational`` lists A1/A2/A3/DT to be fitted (with seed values) and
    may carry fixed force-law metadata such as ``R_EXPONENT``.
    """
    progress_step = 0
    progress_total = 0

    def check_cancelled() -> None:
        if cancel_callback is not None and cancel_callback():
            raise OrbitFitCancelled(tr("Výpočet dráhy byl zrušen uživatelem."))

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

    report_progress(tr("Připravuji astrometrii a počáteční dráhu…"))
    # Residuals left by an older run must never be read as the prediction of
    # the current trial.  OrbitFitWorker works on copies, so a failed or
    # cancelled run leaves the user's table untouched.
    for observation in observations:
        observation.used_in_fit = False
        observation.residual_ra_arcsec = None
        observation.residual_dec_arcsec = None
    duplicate_count = mark_duplicate_measurements(observations)
    active_rows: list[tuple[Observation, datetime, float, float]] = []
    for observation in observations:
        if not is_fit_candidate(observation):
            continue
        try:
            active_rows.append((
                observation, parse_observation_time(observation.time),
                observation.ra_deg, observation.dec_deg,
            ))
        except ValueError:
            pass
    active_rows.sort(key=lambda item: item[1])
    active_rows = _truncate_rows_to_recent_years(active_rows, truncate_years)
    active_times = [item[1] for item in active_rows]
    arc_days = (
        (max(active_times) - min(active_times)).total_seconds() / 86400.0
        if len(active_times) >= 2 else 0.0
    )
    short_arc = len(active_times) >= 2 and arc_days <= 30.0
    long_arc = arc_days > LONG_ARC_DAYS
    active_count = len(active_times)
    # Range is intrinsically weakly constrained on a discovery arc, and an
    # existing preliminary/MPC seed may itself sit in the wrong basin.  Use
    # several range branches whenever a six-parameter solution is possible.
    multibranch = bool(
        short_arc and active_count >= 6 and not constraints and not nongravitational
    )
    apparition_groups = (
        [] if short_arc else [
            group for group in _observing_apparition_groups(
                [item[0] for item in active_rows], initial,
            )
            # Two timed positions (four numbers) already constrain the phase
            # of a propagated branch; a single isolated row does not.
            if len(group) >= 2
        ]
    )
    apparition_linkage: list[dict[str, object]] = []
    linkage_note = ""

    if multibranch:
        result = _fit_short_arc_branches(
            observations, initial, constraints, truncate_years, nongravitational,
            fit_once, report_progress,
        )
    elif len(apparition_groups) >= 2:
        from .linkage import link_apparitions

        result, apparition_linkage, linkage_note = link_apparitions(
            observations, apparition_groups, initial,
            constraints=constraints, nongravitational=nongravitational,
            fit_once=fit_once, evaluate=_evaluate_solution_observations,
            report_progress=report_progress, cancel_callback=cancel_callback,
        )
    else:
        report_progress(tr("Provádím diferenciální korekci dráhy…"), 3)
        result = fit_once(
            observations, initial, constraints, truncate_years, nongravitational,
            maximum_observations=120,
        )
        if nongravitational or (short_arc and active_count > 120) or long_arc:
            # A1/A2/A3 are sensitive to the complete time/station distribution,
            # and a dense discovery arc can hide competing range solutions in
            # a representative subset.  Correct the seed on every enabled row.
            report_progress(tr("Zpřesňuji řešení na všech měřeních…"))
            result = fit_once(
                observations, result[0], constraints, truncate_years,
                _fitted_nongravitational(nongravitational, result[0]),
                maximum_observations=None, max_nfev=12 if long_arc else 36,
                robust_loss=True,
            )
    if not constraints and not nongravitational:
        result = _prefer_parabolic_short_arc(
            observations, result, truncate_years, cancel_callback,
        )
    # Traditional post-fit rejection remains useful for isolated blunders.
    # Remove at most one pass so a weak preliminary orbit cannot cascade into
    # wholesale data rejection.  Linked apparitions handle rejection per stage.
    if not constraints and not long_arc and not apparition_linkage:
        result = _reject_isolated_blunders(
            observations, result, constraints, truncate_years, nongravitational,
            short_arc, fit_once, report_progress,
        )
    report_progress(tr("Dokončuji O−C, nejistoty a fotometrii…"))
    notes = [result[0].fit_model_note or "", linkage_note]
    if duplicate_count:
        notes.append(
            tr(
                "{0} duplicitních redukcí téhož měření (jiný hvězdný katalog) bylo "
                "vynecháno z vah fitu.",
                duplicate_count,
            )
        )
    result[0].fit_model_note = " ".join(item for item in notes if item) or None
    _calibrate_fitted_photometry(observations, result[0])
    if apparition_linkage:
        result[0].apparition_linkage = tuple(apparition_linkage)
    _attach_multi_apparition_elements(observations, result[0])
    return result


def _fitted_nongravitational(
    requested: dict[str, float] | None, solution: OrbitSolution,
) -> dict[str, float] | None:
    """Requested A/DT names with the values reached by ``solution``."""
    if not requested:
        return None
    values = dict(requested)
    values.update({
        name: value
        for name, value in (solution.nongravitational_parameters or {}).items()
        if name in values
    })
    return values


def _snapshot_report(observations: list[Observation]) -> list[tuple]:
    return [
        (item.residual_ra_arcsec, item.residual_dec_arcsec, item.used_in_fit)
        for item in observations
    ]


def _restore_report(observations: list[Observation], report: list[tuple]) -> None:
    for observation, values in zip(observations, report):
        (
            observation.residual_ra_arcsec,
            observation.residual_dec_arcsec,
            observation.used_in_fit,
        ) = values


def _fit_short_arc_branches(
    observations, initial, constraints, truncate_years, nongravitational,
    fit_once, report_progress,
) -> tuple[OrbitSolution, float, int]:
    """Screen several range branches of a discovery arc and keep the best."""
    initial_candidates = initial_orbit_from_observations(observations, return_candidates=True)
    assert isinstance(initial_candidates, list)
    if initial is not None:
        # A supplied preliminary/catalogue orbit is one more hypothesis, never
        # the only branch allowed to converge.
        initial_candidates.insert(0, initial)
    screened: list[tuple[float, OrbitSolution]] = []
    # Six locally optimised hypotheses followed by the twelve raw ranges with
    # the best complete-arc score cover the useful systematic region.
    for candidate in initial_candidates[:19 if initial is not None else 18]:
        try:
            report_progress(tr("Prověřuji kandidátní větev krátkého oblouku…"), 24)
            trial = fit_once(
                observations, candidate, constraints, truncate_years, nongravitational,
                maximum_observations=36, max_nfev=14, robust_loss=True, screening=True,
            )
            if np.isfinite(trial[1]):
                # Screening ranks the basin only; keep the original range seed
                # because the abbreviated fit can cross a basin boundary.
                screened.append((trial[1], candidate))
        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
            continue
    if not screened:
        raise OrbitFitError(tr("Systematické rangování nenašlo použitelnou větev dráhy."))
    finalists: list[tuple[tuple[OrbitSolution, float, int], list[tuple]]] = []
    for _, candidate in sorted(screened, key=lambda item: item[0])[:3]:
        try:
            report_progress(tr("Zpřesňuji nejlepší větev na všech měřeních…"))
            final_result = fit_once(
                observations, candidate, constraints, truncate_years, nongravitational,
                maximum_observations=None, max_nfev=40, robust_loss=True,
            )
            finalists.append((final_result, _snapshot_report(observations)))
        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
            continue
    if not finalists:
        raise OrbitFitError(tr("Úplná korekce kandidátních drah selhala."))
    result, selected_report = min(finalists, key=lambda item: item[0][1])
    # Restore the O-C table belonging to the winner, not to the last finalist.
    _restore_report(observations, selected_report)
    return result


def _reject_isolated_blunders(
    observations, result, constraints, truncate_years, nongravitational,
    short_arc, fit_once, report_progress,
) -> tuple[OrbitSolution, float, int]:
    candidates = [
        item for item in observations
        if _is_weighted(item) and _is_automatic_rejection_candidate(item, result[1])
    ]
    active_count = sum(
        _is_weighted(item)
        and item.residual_ra_arcsec is not None and item.residual_dec_arcsec is not None
        for item in observations
    )
    # Keep at least 95% and remove the worst normalized residuals first; the
    # user can inspect and reject more later.
    maximum_rejections = max(1, int(np.ceil(active_count * .05)))
    outliers = sorted(
        candidates, key=_normalized_observation_residual, reverse=True,
    )[:maximum_rejections]
    if not outliers or active_count - len(outliers) < 6:
        return result
    for observation in outliers:
        observation.excluded_from_fit = True
        observation.raw["automatic_exclusion"] = True
        observation.raw["automatic_exclusion_reason"] = (
            tr(
                "Izolované měření s velkým normalizovaným O−C; po automatickém vyřazení "
                "byla dráha znovu spočtena.",
            )
        )
    report_progress(tr("Přepočítávám dráhu po vyřazení odlehlých měření…"))
    result = fit_once(
        observations, result[0], constraints, truncate_years,
        _fitted_nongravitational(nongravitational, result[0]),
        maximum_observations=None if nongravitational or short_arc else 120,
        max_nfev=12,
    )
    if not nongravitational and not constraints:
        result = _prefer_parabolic_short_arc(observations, result, truncate_years)
    return result
