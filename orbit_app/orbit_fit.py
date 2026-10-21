from __future__ import annotations

import re
from dataclasses import replace
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
LINKAGE_ACCEPT_RMS_ARCSEC = 60.0
LINKAGE_ACCEPT_NORMALIZED_RMS = 8.0
LINKAGE_DIVERGENCE_RMS_ARCSEC = 200.0
_PLANET_NAMES = (
    "mercury", "venus", "earth", "moon", "mars",
    "jupiter", "saturn", "uranus", "neptune", "pluto",
)


class OrbitFitError(RuntimeError):
    pass


class OrbitFitCancelled(Exception):
    """Raised cooperatively when the user cancels a numerical fit."""


ProgressCallback = Callable[[str, int, int], None]
CancelCallback = Callable[[], bool]


def _branch_preserving_backtrack(
    initial: np.ndarray,
    proposed: np.ndarray,
    residual_evaluator: Callable[[np.ndarray], np.ndarray],
    component_groups: list[np.ndarray],
    sigmas: np.ndarray,
    minimum_scale: float = 1.0 / 256.0,
) -> tuple[np.ndarray, float]:
    """Shorten a correction step until it remains on the propagated branch.

    Residuals and ``sigmas`` use the same two-components-per-observation
    layout.  The joint normalized objective must improve, while no already
    linked apparition may become materially worse.  Scale zero deliberately
    preserves the verified state when even a short step leaves its basin.
    """
    start = np.asarray(initial, dtype=float)
    target = np.asarray(proposed, dtype=float)
    baseline = np.asarray(residual_evaluator(start), dtype=float)
    sigma_values = np.asarray(sigmas, dtype=float)
    if (
        baseline.shape != sigma_values.shape
        or not np.all(np.isfinite(baseline))
        or not np.all(np.isfinite(sigma_values))
        or np.any(sigma_values <= 0.0)
    ):
        return start.copy(), 0.0
    baseline_normalized = baseline / sigma_values
    baseline_cost = float(np.dot(baseline_normalized, baseline_normalized))
    nonempty_groups = [indices for indices in component_groups if indices.size]
    baseline_group_rms = [
        float(np.sqrt(np.mean(np.square(baseline[indices]))))
        for indices in nonempty_groups
    ]
    scale = 1.0
    while scale >= minimum_scale:
        candidate = start + scale * (target - start)
        try:
            residual = np.asarray(residual_evaluator(candidate), dtype=float)
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            scale *= 0.5
            continue
        if residual.shape != baseline.shape or not np.all(np.isfinite(residual)):
            scale *= 0.5
            continue
        normalized = residual / sigma_values
        cost = float(np.dot(normalized, normalized))
        groups_ok = True
        for indices, starting_rms in zip(nonempty_groups, baseline_group_rms):
            candidate_rms = float(np.sqrt(np.mean(np.square(residual[indices]))))
            # Allow only sampling/numerical-scale movement in a good return;
            # the remote return must supply the real decrease in joint cost.
            permitted = max(starting_rms * 1.02, starting_rms + 0.05)
            if candidate_rms > permitted:
                groups_ok = False
                break
        if groups_ok and cost < baseline_cost * (1.0 - 1e-10):
            return candidate, scale
        scale *= 0.5
    return start.copy(), 0.0


def _linkage_trial_is_acceptable(
    trial: tuple[OrbitSolution, float, int],
) -> bool:
    """Apply the common quality gate used by multi-apparition branch trials."""
    normalized_rms = trial[0].fit_normalized_rms
    return bool(
        np.isfinite(trial[1])
        and trial[1] <= LINKAGE_ACCEPT_RMS_ARCSEC
        and normalized_rms is not None
        and np.isfinite(normalized_rms)
        and normalized_rms <= LINKAGE_ACCEPT_NORMALIZED_RMS
    )


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


def _observation_sigmas(
    rows: list[tuple[Observation, datetime, float, float]],
    account_for_batch_correlation: bool = False,
) -> np.ndarray:
    """Return astrometric sigmas, optionally accounting for same-night batches.

    Measurements from one station during one observing night share catalogue,
    timing and reduction systematics.  Follow the Vereš et al. (2017) rule
    used by JPL/GRSS: up to four measurements retain their nominal weight and
    a batch of N > 4 is deweighted by sqrt(N/4).  This keeps a dense CCD run
    from overwhelming a sparse apparition while retaining its time coverage.
    """
    values = np.asarray([
        value
        for item in rows
        for value in _observation_sigma_arcsec(item[0])
    ], dtype=float)
    if not account_for_batch_correlation or not rows:
        return values
    batch_sizes: dict[tuple[str, object], int] = {}
    batch_keys: list[tuple[str, object]] = []
    for observation, instant, _ra, _dec in rows:
        # Noon boundary keeps one astronomical night together across midnight.
        observing_night = (instant - timedelta(hours=12)).date()
        key = ((observation.station or "").strip().upper(), observing_night)
        batch_keys.append(key)
        batch_sizes[key] = batch_sizes.get(key, 0) + 1
    factors = np.asarray([
        np.sqrt(max(1.0, batch_sizes[key] / 4.0))
        for key in batch_keys
        for _coordinate in range(2)
    ], dtype=float)
    return values * factors


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


def _arc_extension_counts(observation_count: int) -> tuple[int, ...]:
    """Return monotonically growing Find_Orb-style arc-extension sizes."""
    if observation_count <= 0:
        return ()
    counts: list[int] = []
    count = min(6, observation_count)
    while count < observation_count:
        counts.append(count)
        # Each extension performs both a secular-parameter prefit and a full
        # state correction.  Geometric growth by four keeps a many-thousand
        # row apparition interactive while still approaching it gradually.
        count = min(observation_count, count * 4)
    counts.append(observation_count)
    return tuple(counts)


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


def _balanced_linkage_observations(
    apparition_groups: list[list[Observation]], maximum_per_group: int = 120,
) -> list[Observation]:
    """Return a chronological sample in which no apparition dominates.

    This is only a branch-finding sample.  Once a common dynamical branch is
    found, the final correction still uses every enabled observation with its
    normal astrometric uncertainty.
    """
    populated = [group for group in apparition_groups if group]
    if not populated:
        return []
    populated = [
        [observation for observation in group if not observation.excluded_from_fit]
        for group in populated
    ]
    populated = [group for group in populated if group]
    if not populated:
        return []
    populated = [_normal_place_observations(group) for group in populated]
    sizes = [len(group) for group in populated]
    if max(sizes) <= 2 * min(sizes) and max(sizes) <= maximum_per_group:
        return [observation for group in populated for observation in group]
    target = min(maximum_per_group, min(sizes))
    balanced: list[Observation] = []
    for group in populated:
        if len(group) <= target:
            balanced.extend(group)
            continue
        chronological = sorted(group, key=lambda item: parse_observation_time(item.time))
        buckets: dict[tuple[str, object], list[Observation]] = {}
        for observation in chronological:
            instant = parse_observation_time(observation.time)
            key = (
                (observation.station or "").strip().upper(),
                (instant - timedelta(hours=12)).date(),
            )
            buckets.setdefault(key, []).append(observation)
        bucket_rows = list(buckets.values())
        # Give every station-night one representative when possible, then add
        # points round-robin.  Within a night the chosen ranks spread over its
        # full time span instead of always taking the first exposure.
        if len(bucket_rows) > target:
            bucket_epochs = np.asarray([
                np.mean([
                    parse_observation_time(item.time).timestamp()
                    for item in bucket
                ])
                for bucket in bucket_rows
            ])
            keep = _representative_indices(bucket_epochs, target)
            bucket_rows = [bucket_rows[index] for index in keep]
        allocations = [0] * len(bucket_rows)
        while sum(allocations) < target:
            candidates = [
                index for index, bucket in enumerate(bucket_rows)
                if allocations[index] < len(bucket)
            ]
            if not candidates:
                break
            index = min(
                candidates,
                key=lambda value: allocations[value] / len(bucket_rows[value]),
            )
            allocations[index] += 1
        for bucket, count in zip(bucket_rows, allocations):
            if count <= 0:
                continue
            indices = np.linspace(0, len(bucket) - 1, count).round().astype(int)
            balanced.extend(bucket[index] for index in np.unique(indices))
    balanced.sort(key=lambda observation: parse_observation_time(observation.time))
    return balanced


def _normal_place_observations(
    observations: list[Observation], maximum_independent_per_night: int = 4,
) -> list[Observation]:
    """Build deterministic station-night normal places for branch finding.

    A dense CCD batch is not a collection of fully independent directions:
    catalogue and reduction errors are shared by the whole station-night.
    Batches up to ``maximum_independent_per_night`` are therefore retained
    unchanged, while larger batches are represented by a spherical mean with
    an uncertainty that gains at most that many independent measurements.
    The original observations remain untouched and are used by the final fit.
    """
    passthrough: list[Observation] = []
    buckets: dict[tuple[str, object], list[Observation]] = {}
    for observation in observations:
        if observation.excluded_from_fit or not observation.time:
            continue
        try:
            instant = parse_observation_time(observation.time)
        except ValueError:
            continue
        if observation.ra_deg is None or observation.dec_deg is None:
            passthrough.append(observation)
            continue
        key = (
            (observation.station or "").strip().upper(),
            (instant - timedelta(hours=12)).date(),
        )
        buckets.setdefault(key, []).append(observation)

    output: list[Observation] = list(passthrough)
    for bucket in buckets.values():
        chronological = sorted(bucket, key=lambda item: parse_observation_time(item.time))
        if len(chronological) <= maximum_independent_per_night:
            output.extend(chronological)
            continue

        instants = [parse_observation_time(item.time) for item in chronological]
        epochs = np.asarray([instant.timestamp() for instant in instants], dtype=float)
        ras = np.radians([float(item.ra_deg) for item in chronological])
        decs = np.radians([float(item.dec_deg) for item in chronological])
        directions = np.column_stack((
            np.cos(decs) * np.cos(ras),
            np.cos(decs) * np.sin(ras),
            np.sin(decs),
        ))
        mean_direction = np.mean(directions, axis=0)
        norm = float(np.linalg.norm(mean_direction))
        if not np.isfinite(norm) or norm <= 0.0:
            output.extend(chronological)
            continue
        mean_direction /= norm
        mean_ra = float(np.degrees(np.arctan2(
            mean_direction[1], mean_direction[0],
        )) % 360.0)
        mean_dec = float(np.degrees(np.arcsin(np.clip(mean_direction[2], -1.0, 1.0))))
        mean_timestamp = float(np.mean(epochs))
        mean_time = datetime.fromtimestamp(
            mean_timestamp, tz=instants[0].tzinfo,
        ).isoformat().replace("+00:00", "Z")

        sigma_pairs = np.asarray([
            _observation_sigma_arcsec(item) for item in chronological
        ], dtype=float)
        effective_count = min(len(chronological), maximum_independent_per_night)
        # The formal mean may improve as sqrt(N), but never below the actual
        # within-night scatter or an epoch-appropriate systematic floor.
        delta_ra = (
            (np.degrees(ras) - mean_ra + 180.0) % 360.0 - 180.0
        ) * np.cos(np.radians(mean_dec)) * 3600.0
        delta_dec = (np.degrees(decs) - mean_dec) * 3600.0
        nominal = np.sqrt(np.mean(np.square(sigma_pairs), axis=0)) / np.sqrt(
            effective_count
        )
        scatter = np.asarray((
            np.std(delta_ra, ddof=1), np.std(delta_dec, ddof=1),
        )) / np.sqrt(effective_count)
        sigma = np.maximum(nominal, scatter)
        template = chronological[len(chronological) // 2]
        raw = dict(template.raw)
        raw.update({
            "rmsRA": float(sigma[0]),
            "rmsDec": float(sigma[1]),
            "normalPlaceCount": len(chronological),
        })
        output.append(Observation(
            designation=template.designation,
            time=mean_time,
            ra_deg=mean_ra,
            dec_deg=mean_dec,
            magnitude=template.magnitude,
            band=template.band,
            station=template.station,
            catalog=template.catalog,
            raw=raw,
        ))
    output.sort(key=lambda item: parse_observation_time(item.time))
    return output


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
    fit_state: bool = True,
    observation_weights: dict[int, float] | None = None,
    strict_polish: bool = False,
    preserve_linkage_branch: bool = False,
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
    astrometric_sigmas = _observation_sigmas(
        usable, account_for_batch_correlation=True,
    )
    if observation_weights:
        row_weights = np.asarray([
            max(1e-12, float(observation_weights.get(id(item[0]), 1.0)))
            for item in usable
        ])
        # Least squares divides residuals by sigma.  Dividing sigma by the
        # square root gives each apparition the requested total statistical
        # leverage without cloning observations or changing their published
        # per-row uncertainty.
        astrometric_sigmas = astrometric_sigmas / np.repeat(
            np.sqrt(row_weights), 2,
        )
    nongrav_names = tuple(
        name for name in ("A1", "A2", "A3", "DT")
        if name in (nongravitational or {})
    )
    if not fit_state and not nongrav_names:
        raise OrbitFitError(
            "Rychlý apariční předfit potřebuje alespoň jeden volný "
            "negravitační parametr."
        )
    initial_nongrav = {
        name: float((initial.nongravitational_parameters or {}).get(name, 0.0))
        for name in nongrav_names
    }
    initial_nongrav.update(nongravitational or {})
    # Distance-law constants are fixed model metadata, not fitted dimensions.
    # Preserve them through every differential-correction pass; otherwise an
    # asteroid selected as r^-2 silently reverts to the Marsden comet law.
    fixed_nongrav = {
        str(name): float(value)
        for name, value in (nongravitational or {}).items()
        if name not in nongrav_names
    }
    fixed_nongrav.update({
        str(name): float(value)
        for name, value in (fixed_nongravitational or {}).items()
        if name not in nongrav_names
    })

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
            full_ephemeris,
            relative_tolerance=3e-7 if screening else 1e-8,
            absolute_tolerance=3e-9 if screening else 1e-10,
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
        emitted, _light_time = _local_light_time_states(
            propagated, earth_at_observations,
        )
        if not np.all(np.isfinite(emitted)):
            raise FloatingPointError("Lokální oprava o dobu letu světla selhala.")
        computed_ra, computed_dec = _equatorial_direction(
            emitted, epochs, observer_offsets, earth_at_observations,
        )
        delta_ra = (computed_ra - observed_ra + 180.0) % 360.0 - 180.0
        delta_ra *= np.cos(np.radians(observed_dec))
        delta_dec = computed_dec - observed_dec
        return np.column_stack((delta_ra, delta_dec)).ravel() * 3600.0

    def variational_residuals_and_jacobian(
        vector: np.ndarray, engine: NBodyIntegrator,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Evaluate normalized residuals and one-pass variational derivatives."""
        parameters = force_parameters(vector)
        working_engine = engine
        if parameters != engine.nongravitational_parameters:
            working_engine = NBodyIntegrator(
                engine.ephemeris, perturbers=engine.perturbers,
                relative_tolerance=engine.rtol, absolute_tolerance=engine.atol,
                nongravitational_parameters=parameters,
                include_solar_relativity=engine.include_solar_relativity,
            )
        state = StateVector(reference_epoch, vector[:3], vector[3:6])
        parameter_scales = np.asarray([
            1.0 if name == "DT" else 1e-8 for name in nongrav_names
        ])
        propagated, state_partials = working_engine.propagate_with_partials(
            state, epochs, nongrav_names, parameter_scales,
        )

        def projected(states: np.ndarray) -> np.ndarray:
            emitted, _light_time = _local_light_time_states(
                states, earth_at_observations,
            )
            computed_ra, computed_dec = _equatorial_direction(
                emitted, epochs, observer_offsets, earth_at_observations,
            )
            delta_ra = (computed_ra - observed_ra + 180.0) % 360.0 - 180.0
            delta_ra *= np.cos(np.radians(observed_dec))
            return np.column_stack((
                delta_ra, computed_dec - observed_dec,
            )).ravel() * 3600.0

        astrometry = projected(propagated)
        column_steps = np.concatenate((
            np.array([1e-6, 1e-6, 1e-6, 1e-8, 1e-8, 1e-8]),
            np.array([.05 if name == "DT" else 5e-3 for name in nongrav_names]),
        ))
        astrometric_jacobian = np.empty(
            (astrometry.size, 6 + len(nongrav_names)), dtype=float,
        )
        # Only the cheap sky projection is differenced here.  All expensive
        # dynamical derivatives came from the same augmented integration.
        for column, nominal_step in enumerate(column_steps):
            # Over several revolutions a unit velocity or A correction can
            # move the remote apparition by AU.  Size the cheap projection
            # perturbation from the propagated partial so it remains local
            # (roughly a few arcseconds at one AU) without reintegrating.
            position_sensitivity = float(np.max(np.linalg.norm(
                state_partials[:, :3, column], axis=1,
            )))
            step = float(nominal_step)
            if np.isfinite(position_sensitivity) and position_sensitivity > 0.0:
                step = min(step, 1e-5 / position_sensitivity)
            step = max(step, 1e-12)
            displacement = state_partials[:, :, column] * step
            astrometric_jacobian[:, column] = (
                projected(propagated + displacement)
                - projected(propagated - displacement)
            ) / (2.0 * step)

        residual_values = astrometry / astrometric_sigmas
        jacobian_values = astrometric_jacobian / astrometric_sigmas[:, None]
        if constraints:
            names = ("a", "e", "i", "node", "arg_peri", "mean_anomaly")

            def constraint_values(candidate: np.ndarray) -> np.ndarray:
                try:
                    elements = cartesian_to_elements(candidate[:3], candidate[3:6])
                except Exception:
                    return np.full(len(constraints), 1e9)
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
                return np.asarray(penalties, dtype=float)

            penalties = constraint_values(vector)
            penalty_jacobian = np.zeros(
                (len(penalties), 6 + len(nongrav_names)), dtype=float,
            )
            for column in range(6):
                step = column_steps[column]
                plus = vector.copy()
                minus = vector.copy()
                plus[column] += step
                minus[column] -= step
                penalty_jacobian[:, column] = (
                    constraint_values(plus) - constraint_values(minus)
                ) / (2.0 * step)
            residual_values = np.append(residual_values, penalties)
            jacobian_values = np.vstack((jacobian_values, penalty_jacobian))
        return residual_values, jacobian_values

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
    # Keep numerical derivatives away from physically absurd trial forces.
    # The scaled A limit of 100 means 1e-6 AU/day^2, already far above the
    # usual Marsden parameters, while still admitting a larger supplied seed.
    nongrav_limits = np.array([
        (
            max(365.0, abs(initial_nongrav.get(name, 0.0)) * 2.0 + 30.0)
            if name == "DT" else
            max(100.0, abs(initial_nongrav.get(name, 0.0)) / 1e-8 * 10.0 + 10.0)
        )
        for name in nongrav_names
    ])
    nongrav_lower = -nongrav_limits
    nongrav_upper = nongrav_limits
    lower_bounds = np.concatenate((np.full(6, -np.inf), nongrav_lower))
    upper_bounds = np.concatenate((np.full(6, np.inf), nongrav_upper))

    def expand_fit_vector(vector: np.ndarray) -> np.ndarray:
        if fit_state:
            return vector
        return np.concatenate((initial_vector[:6], vector))

    optimizer_initial = initial_vector if fit_state else initial_vector[6:]
    optimizer_scale = fine_scale if fit_state else fine_scale[6:]
    optimizer_lower = lower_bounds if fit_state else lower_bounds[6:]
    optimizer_upper = upper_bounds if fit_state else upper_bounds[6:]
    full_derivative_steps = np.concatenate((
        np.array([1e-5, 1e-5, 1e-5, 1e-9, 1e-9, 1e-9]),
        np.array([
            .1 if name == "DT" else 1e-3 for name in nongrav_names
        ]),
    ))
    derivative_steps = (
        full_derivative_steps if fit_state else full_derivative_steps[6:]
    ).copy()

    use_variational_jacobian = arc_days > 365.25
    variational_cache: dict[str, np.ndarray | None] = {
        "vector": None, "residual": None, "jacobian": None,
    }

    def variational_evaluation(
        vector: np.ndarray, engine: NBodyIntegrator = optimizer_engine,
    ) -> tuple[np.ndarray, np.ndarray]:
        cached_vector = variational_cache["vector"]
        if (
            engine is optimizer_engine and cached_vector is not None
            and np.array_equal(np.asarray(cached_vector), vector)
        ):
            return (
                np.asarray(variational_cache["residual"]),
                np.asarray(variational_cache["jacobian"]),
            )
        full_residual, full_jacobian = variational_residuals_and_jacobian(
            expand_fit_vector(vector), engine,
        )
        selected_jacobian = full_jacobian if fit_state else full_jacobian[:, 6:]
        if engine is optimizer_engine:
            variational_cache["vector"] = vector.copy()
            variational_cache["residual"] = full_residual
            variational_cache["jacobian"] = selected_jacobian
        return full_residual, selected_jacobian

    def optimizer_residuals(vector: np.ndarray) -> np.ndarray:
        if use_variational_jacobian:
            check_cancelled()
            try:
                return variational_evaluation(vector)[0]
            except (RuntimeError, ValueError, FloatingPointError, OverflowError):
                return np.full(
                    observed_ra.size * 2 + len(constraints or {}),
                    1e12, dtype=float,
                )
        return residuals(expand_fit_vector(vector))

    def optimizer_jacobian(vector: np.ndarray) -> np.ndarray:
        if use_variational_jacobian:
            return variational_evaluation(vector)[1]
        return _adaptive_finite_difference_jacobian(
            optimizer_residuals, vector, derivative_steps,
            optimizer_lower, optimizer_upper, observed_ra.size * 2,
        )

    # Over a multi-year arc, generic finite-difference steps can either vanish
    # in integration noise or move the prediction far outside the local phase
    # branch.  Find_Orb's full_improvement() explicitly sizes every perturbation
    # by its astrometric effect; use the same principle here.  SciPy's TRF then
    # obtains the covariance/SVD directions from this well-scaled Jacobian.
    numerical_jacobian = optimizer_jacobian if arc_days > 365.25 else "2-point"

    if derived_initial:
        # The angles-only initializer deliberately favours a broad basin over
        # sub-arcsecond precision.  First pull its six Cartesian components
        # into the local minimum with a wide robust scale; the normal fine fit
        # below then handles individual outliers.
        coarse = least_squares(
            optimizer_residuals,
            initial_vector,
            jac=numerical_jacobian,
            method="trf",
            x_scale="jac",
            bounds=(lower_bounds, upper_bounds),
            loss="soft_l1",
            f_scale=100.0,
            max_nfev=60,
            verbose=0,
        )
        initial_vector = expand_fit_vector(coarse.x)
        optimizer_initial = initial_vector

    check_cancelled()
    result = least_squares(
        optimizer_residuals,
        optimizer_initial,
        jac=numerical_jacobian,
        method="trf",
        x_scale=("jac" if derived_initial else optimizer_scale),
        bounds=(optimizer_lower, optimizer_upper),
        loss="soft_l1" if robust_loss else "linear",
        f_scale=float(robust_scale),
        max_nfev=(
            max_nfev if max_nfev is not None
            else 100 if derived_initial else 24
        ),
        verbose=0,
    )
    result_vector = expand_fit_vector(result.x)
    if use_fast_differential_correction:
        # Polish the fast short-arc solution with the complete planetary force
        # model.  Starting in the correct basin needs only a few evaluations.
        result = least_squares(
            lambda vector: residuals_with_engine(vector, full_engine),
            result_vector,
            method="trf", x_scale=fine_scale,
            bounds=(lower_bounds, upper_bounds), loss="linear",
            max_nfev=4 if screening else 8, verbose=0,
        )
        result_vector = result.x
    elif arc_days > 730.0 and not screening and strict_polish:
        # The relaxed century-scale integration above is only a branch finder.
        # Its state can differ imperceptibly at the reference epoch yet miss a
        # remote apparition by arcminutes after many revolutions.  The old
        # code merely *reported* that discrepancy with the strict engine even
        # though the comment promised a strict polish.  Select the better
        # starting state in the exact force model, then really correct it.
        def strict_optimizer_residuals(vector: np.ndarray) -> np.ndarray:
            return residuals_with_engine(expand_fit_vector(vector), full_engine)

        def strict_optimizer_jacobian(vector: np.ndarray) -> np.ndarray:
            return _adaptive_finite_difference_jacobian(
                strict_optimizer_residuals, vector, derivative_steps,
                optimizer_lower, optimizer_upper, observed_ra.size * 2,
            )

        candidate_residual = strict_optimizer_residuals(result.x)
        initial_residual = strict_optimizer_residuals(optimizer_initial)
        candidate_cost = float(np.dot(candidate_residual, candidate_residual))
        initial_cost = float(np.dot(initial_residual, initial_residual))
        strict_start = result.x if candidate_cost <= initial_cost else optimizer_initial
        result = least_squares(
            strict_optimizer_residuals,
            strict_start,
            jac=strict_optimizer_jacobian,
            method="trf", x_scale=optimizer_scale,
            bounds=(optimizer_lower, optimizer_upper),
            loss="soft_l1" if robust_loss else "linear",
            f_scale=float(robust_scale),
            max_nfev=min(8, max_nfev or 8), verbose=0,
        )
        result_vector = expand_fit_vector(result.x)
    if preserve_linkage_branch:
        # Encounter-sensitive century-scale mappings can turn one nominal TRF
        # step into a jump to another phase branch.  Test fractions of that
        # step with the strict force model before allowing it to replace the
        # already verified propagated solution.
        group_ids = {
            id(observation): group_index
            for group_index, group in enumerate(
                _observing_apparition_groups(
                    [item[0] for item in usable], initial,
                )
            )
            for observation in group
        }
        grouped_components: list[list[int]] = []
        for row_index, item in enumerate(usable):
            group_index = group_ids.get(id(item[0]), 0)
            while len(grouped_components) <= group_index:
                grouped_components.append([])
            grouped_components[group_index].extend(
                (2 * row_index, 2 * row_index + 1)
            )
        safe_vector, accepted_scale = _branch_preserving_backtrack(
            optimizer_initial,
            result.x,
            lambda vector: astrometric_residuals(
                expand_fit_vector(vector), exact_full_engine,
            ),
            [np.asarray(indices, dtype=int) for indices in grouped_components],
            astrometric_sigmas,
        )
        if accepted_scale < 1.0:
            result.x = safe_vector
            result_vector = expand_fit_vector(safe_vector)
            result.jac = optimizer_jacobian(safe_vector)
    if not result.success and result.nfev < 2:
        raise OrbitFitError(result.message)
    check_cancelled()
    elements = cartesian_to_elements(result_vector[:3], result_vector[3:6])
    reporting_engine = (
        full_engine if preserve_linkage_branch or not screening
        else optimizer_engine
    )
    try:
        final_astrometric_residual = astrometric_residuals(result_vector, reporting_engine)
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
    if screening or not fit_state:
        uncertainties, state_covariance = None, None
    else:
        uncertainties, state_covariance = _formal_element_uncertainties(
            result_vector, elements, astrometric_jacobian,
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
    # Calculate reportable O-C values for every valid observation, not only for
    # the (at most 120) points retained by the interactive optimizer.
    report_times = [item[1] for item in reportable]
    report_epochs = dynamical_jd(report_times)
    report_offsets = _observation_offsets_gcrs(reportable)
    report_ra = np.array([item[2] for item in reportable])
    report_dec = np.array([item[3] for item in reportable])
    report_state = StateVector(reference_epoch, result_vector[:3], result_vector[3:6])
    check_cancelled()
    try:
        report_positions = propagate(result_vector, report_state, report_epochs, full_engine)
        report_earth = earth_heliocentric_ecliptic(report_epochs)
        report_light_time = np.linalg.norm(
            report_positions[:, :3] - report_earth, axis=1,
        ) / C_AU_PER_DAY
        report_emitted = propagate(
            result_vector, report_state, report_epochs - report_light_time, full_engine
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
        if observation.excluded_from_fit:
            continue
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


def _evaluate_solution_observations(
    observations: list[Observation], solution: OrbitSolution,
    nongravitational: dict[str, float] | None = None,
    cancel_callback: CancelCallback | None = None,
) -> tuple[float, float, int]:
    """Propagate an existing solution and populate O-C without fitting it.

    This deliberately read-only orbit appraisal is important for historical
    comet recoveries.  A catalogue solution may already represent the correct
    revolution count across several apparitions; first fitting only the newest
    return can move it into an excellent *local* minimum which misses an old
    sparse recovery by many arcminutes.
    """
    if solution.position_au is None or solution.velocity_au_per_day is None:
        raise OrbitFitError("Chybí počáteční kartézský stav dráhy.")
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
    if not parsed:
        raise OrbitFitError("Pro kontrolu dráhy nejsou dostupná platná měření.")
    parsed.sort(key=lambda item: item[1])
    if cancel_callback is not None and cancel_callback():
        raise OrbitFitCancelled("Výpočet dráhy byl zrušen uživatelem.")
    epochs = dynamical_jd([item[1] for item in parsed])
    cache_start = min(float(np.min(epochs)), float(solution.epoch_jd)) - 2.0
    cache_end = max(float(np.max(epochs)), float(solution.epoch_jd)) + 2.0
    ephemeris = (
        _InterpolatedPlanetEphemeris(PLANETS, cache_start, cache_end)
        if cache_end - cache_start <= 730.0 else
        SplinePlanetEphemeris(
            PLANETS, _PLANET_NAMES, cache_start, cache_end,
        )
    )
    force_parameters = dict(solution.nongravitational_parameters or {})
    force_parameters.update(nongravitational or {})
    engine = NBodyIntegrator(
        ephemeris, relative_tolerance=3e-11, absolute_tolerance=2e-13,
        nongravitational_parameters=force_parameters,
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
            "Kontrolní propagace katalogové dráhy skončila numerickou chybou."
        ) from exc
    offsets = _observation_offsets_gcrs(parsed)
    computed_ra, computed_dec = _equatorial_direction(
        emitted, epochs, offsets, earth,
    )
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
        sigma_ra, sigma_dec = _observation_sigma_arcsec(observation)
        physical.extend((float(delta_ra), float(delta_dec)))
        normalized.extend((float(delta_ra) / sigma_ra, float(delta_dec) / sigma_dec))
    return (
        float(np.sqrt(np.mean(np.square(physical)))),
        float(np.sqrt(np.mean(np.square(normalized)))),
        len(parsed),
    )


def _propagate_solution_to_epoch(
    solution: OrbitSolution,
    target_jd: float,
    nongravitational: dict[str, float] | None = None,
) -> OrbitSolution:
    """Return the heliocentric J2000-ecliptic state at ``target_jd``.

    This is used only to construct an identification seed between independently
    fitted apparitions.  It does not fit observations and it deliberately
    clears formal uncertainties, which do not transform by merely copying the
    original covariance to another epoch.
    """
    if solution.position_au is None or solution.velocity_au_per_day is None:
        raise OrbitFitError("Chybí počáteční kartézský stav dráhy.")
    epoch = float(target_jd)
    cache_start = min(epoch, float(solution.epoch_jd)) - 2.0
    cache_end = max(epoch, float(solution.epoch_jd)) + 2.0
    ephemeris = (
        _InterpolatedPlanetEphemeris(PLANETS, cache_start, cache_end)
        if cache_end - cache_start <= 730.0 else
        SplinePlanetEphemeris(PLANETS, _PLANET_NAMES, cache_start, cache_end)
    )
    force = dict(nongravitational or {})
    engine = NBodyIntegrator(
        ephemeris, relative_tolerance=3e-11, absolute_tolerance=2e-13,
        nongravitational_parameters=force,
    )
    source = StateVector(
        solution.epoch_jd,
        np.asarray(solution.position_au, dtype=float),
        np.asarray(solution.velocity_au_per_day, dtype=float),
    )
    try:
        state = engine.propagate(source, np.asarray([epoch], dtype=float))[0]
        elements = cartesian_to_elements(state[:3], state[3:6])
    except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
        raise OrbitFitError(
            "Identifikační propagace mezi návraty skončila numerickou chybou."
        ) from exc
    return replace(
        solution,
        epoch_jd=epoch,
        semi_major_axis_au=elements[0],
        eccentricity=elements[1],
        inclination_deg=elements[2],
        node_deg=elements[3],
        arg_peri_deg=elements[4],
        mean_anomaly_deg=elements[5],
        position_au=tuple(float(value) for value in state[:3]),
        velocity_au_per_day=tuple(float(value) for value in state[3:6]),
        state_covariance=None,
        element_uncertainties=None,
        nongravitational_parameters=force or None,
    )


def _apparition_orbit_similarity(
    propagated_new: OrbitSolution,
    independently_old: OrbitSolution,
) -> tuple[bool, dict[str, float]]:
    """Compare two independent heliocentric J2000-ecliptic orbit hypotheses.

    This is an identification test, not an astrometric acceptance test.  Lost
    comet recoveries can miss the old sky position substantially after many
    revolutions while retaining a recognisably similar perihelion distance,
    eccentricity, orbital plane and apsidal direction.  A subsequent joint
    differential correction must still fit every claimed apparition.
    """
    if abs(propagated_new.epoch_jd - independently_old.epoch_jd) > 1e-7:
        raise OrbitFitError(
            "Identifikační dráhy nejsou vztaženy ke stejné epoše."
        )
    states = []
    for solution in (propagated_new, independently_old):
        if solution.position_au is None or solution.velocity_au_per_day is None:
            raise OrbitFitError("Identifikační dráze chybí kartézský stav.")
        position = np.asarray(solution.position_au, dtype=float)
        velocity = np.asarray(solution.velocity_au_per_day, dtype=float)
        angular_momentum = np.cross(position, velocity)
        h_norm = float(np.linalg.norm(angular_momentum))
        radius = float(np.linalg.norm(position))
        if h_norm <= 0.0 or radius <= 0.0:
            raise OrbitFitError("Identifikační dráha má degenerovaný stav.")
        eccentricity_vector = (
            (
                float(np.dot(velocity, velocity)) - MU_SUN / radius
            ) * position
            - float(np.dot(position, velocity)) * velocity
        ) / MU_SUN
        e_norm = float(np.linalg.norm(eccentricity_vector))
        if e_norm <= 1e-12:
            raise OrbitFitError(
                "Téměř kruhovou dráhu nelze porovnat podle směru perihelu."
            )
        states.append((
            angular_momentum / h_norm,
            eccentricity_vector / e_norm,
        ))
    plane_angle = float(np.degrees(np.arccos(np.clip(
        np.dot(states[0][0], states[1][0]), -1.0, 1.0,
    ))))
    apsidal_angle = float(np.degrees(np.arccos(np.clip(
        np.dot(states[0][1], states[1][1]), -1.0, 1.0,
    ))))
    q_new = propagated_new.semi_major_axis_au * (
        1.0 - propagated_new.eccentricity
    )
    q_old = independently_old.semi_major_axis_au * (
        1.0 - independently_old.eccentricity
    )
    q_fraction = abs(q_new - q_old) / max(abs(q_new), abs(q_old), 1e-12)
    eccentricity_difference = abs(
        propagated_new.eccentricity - independently_old.eccentricity
    )
    metrics = {
        "plane_deg": plane_angle,
        "apsidal_deg": apsidal_angle,
        "q_fraction": float(q_fraction),
        "e_difference": float(eccentricity_difference),
    }
    similar = bool(
        plane_angle <= 8.0
        and apsidal_angle <= 20.0
        and q_fraction <= 0.35
        and eccentricity_difference <= 0.25
    )
    return similar, metrics


def _transverse_period_candidate(
    solution: OrbitSolution,
    fractional_transverse_change: float,
) -> OrbitSolution:
    """Perturb mean motion while preserving epoch position and radial speed.

    The dimensionless change applies only to the transverse velocity in the
    heliocentric J2000-ecliptic orbital plane.  It is a branch-finding seed for
    a period/phase search, not a fitted solution.
    """
    if solution.position_au is None or solution.velocity_au_per_day is None:
        raise OrbitFitError("Periodové korekci chybí kartézský stav.")
    position = np.asarray(solution.position_au, dtype=float)
    velocity = np.asarray(solution.velocity_au_per_day, dtype=float)
    radius = float(np.linalg.norm(position))
    angular_momentum = np.cross(position, velocity)
    h_norm = float(np.linalg.norm(angular_momentum))
    if radius <= 0.0 or h_norm <= 0.0:
        raise OrbitFitError("Periodová korekce dostala degenerovaný stav.")
    radial_direction = position / radius
    normal_direction = angular_momentum / h_norm
    transverse_direction = np.cross(normal_direction, radial_direction)
    radial_speed = float(np.dot(velocity, radial_direction))
    transverse_speed = float(np.dot(velocity, transverse_direction))
    corrected_velocity = (
        radial_speed * radial_direction
        + transverse_speed
        * (1.0 + float(fractional_transverse_change))
        * transverse_direction
    )
    elements = cartesian_to_elements(position, corrected_velocity)
    return replace(
        solution,
        semi_major_axis_au=elements[0],
        eccentricity=elements[1],
        inclination_deg=elements[2],
        node_deg=elements[3],
        arg_peri_deg=elements[4],
        mean_anomaly_deg=elements[5],
        velocity_au_per_day=tuple(
            float(value) for value in corrected_velocity
        ),
        state_covariance=None,
        element_uncertainties=None,
        nongravitational_parameters=None,
    )


def _needs_historical_identification(
    earlier_epoch_jd: float,
    later_epoch_jd: float,
    solution: OrbitSolution,
) -> bool:
    """Return whether the gap spans roughly two or more orbital revolutions."""
    semi_major = float(solution.semi_major_axis_au)
    if not np.isfinite(semi_major) or semi_major <= 0.0:
        return False
    period_days = 2.0 * np.pi * np.sqrt(semi_major ** 3 / MU_SUN)
    gap_days = abs(float(later_epoch_jd) - float(earlier_epoch_jd))
    # Apparition epochs need not coincide exactly with perihelion and the
    # osculating period changes at encounters.  A small margin below two lets
    # a genuinely two-revolution gap enter the identification path without
    # imposing the cost on ordinary adjacent-return corrections.
    return bool(gap_days >= 1.75 * period_days)


def _robust_apparition_appraisal(
    observations: list[Observation],
) -> tuple[float, float, list[Observation], list[Observation]]:
    """Score an apparition while allowing a minority of obvious blunders."""
    scored: list[tuple[float, Observation]] = []
    for observation in observations:
        if (
            observation.excluded_from_fit
            or observation.residual_ra_arcsec is None
            or observation.residual_dec_arcsec is None
        ):
            continue
        sigma_ra, sigma_dec = _observation_sigma_arcsec(observation)
        score = float(np.hypot(
            observation.residual_ra_arcsec / sigma_ra,
            observation.residual_dec_arcsec / sigma_dec,
        ))
        scored.append((score, observation))
    if not scored:
        return np.inf, np.inf, [], []
    scored.sort(key=lambda item: item[0])
    # A three-plate recovery must be supported by at least two plates.  Dense
    # returns may discard only the worst ten percent during this *appraisal*;
    # actual fit exclusions remain explicit and auditable below.
    retained_count = (
        max(2, int(np.ceil(2.0 * len(scored) / 3.0)))
        if len(scored) < 6 else
        max(6, int(np.ceil(0.9 * len(scored))))
    )
    retained = [item[1] for item in scored[:retained_count]]
    rejected = [item[1] for item in scored[retained_count:]]
    physical: list[float] = []
    normalized: list[float] = []
    for observation in retained:
        sigma_ra, sigma_dec = _observation_sigma_arcsec(observation)
        physical.extend((
            float(observation.residual_ra_arcsec),
            float(observation.residual_dec_arcsec),
        ))
        normalized.extend((
            float(observation.residual_ra_arcsec) / sigma_ra,
            float(observation.residual_dec_arcsec) / sigma_dec,
        ))
    return (
        float(np.sqrt(np.mean(np.square(physical)))),
        float(np.sqrt(np.mean(np.square(normalized)))),
        retained, rejected,
    )


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
        now_jd = float(dynamical_jd([datetime.now(timezone.utc)])[0])
        # The first unobserved return is not necessarily still in the future.
        # For an old recovery (for example 206P in 2008), integrate far enough
        # to include the first perihelion after today as well.
        future_reference = max(float(np.max(epochs)), now_jd)
        end = float(future_reference + 1.10 * period_days)
        returns = perihelion_returns_between(solution, start, end)
        latest_observation = float(np.max(epochs))
        # A close planetary encounter can lengthen the next revolution enough
        # that 1.1 times the current osculating period is insufficient. Extend
        # only when necessary, avoiding a second costly integration normally.
        if not any(item.epoch_jd > future_reference for item in returns):
            for multiplier in (2.0, 4.0):
                extended = perihelion_returns_between(
                    solution, start,
                    float(future_reference + multiplier * period_days),
                )
                if extended:
                    returns = extended
                if any(item.epoch_jd > future_reference for item in returns):
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
        "state": tuple(float(value) for value in item.state),
        "future": item.epoch_jd > now_jd,
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
    # A fit consists of many subset/branch trials.  Residuals left by an older
    # user run must never be interpreted as the current trial's prediction for
    # an apparition which has not yet been visited.  OrbitFitWorker restores
    # the complete snapshot if this run fails or is cancelled.
    for observation in observations:
        observation.used_in_fit = False
        observation.residual_ra_arcsec = None
        observation.residual_dec_arcsec = None
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
    secondary_track_exclusions = 0
    catalogue_seed_verified = False
    verified_catalogue_result: tuple[OrbitSolution, float, int] | None = None
    verified_catalogue_report: list[
        tuple[float | None, float | None, bool | None]
    ] | None = None
    verified_catalogue_appraisals: list[tuple[float, float]] | None = None
    catalogue_correction_rejected = False
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
                # Three timed sky positions are the classical minimum for an
                # historical linkage.  They are not sufficient for a robust
                # standalone six-parameter fit, but the common fit also has
                # the well-observed newer apparition and can test whether the
                # propagated branch predicts this sparse return.  Requiring
                # six here silently discarded exactly the three 1951 plates
                # that establish the recovery linkage of 325P/2019 F55.
                if len(group) >= 3
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
            # Per older apparition: cheap A1/A2 prefit, short state correction,
            # and (only on failure) a local-astrometry retry.  One strict fit
            # follows after every return has selected the same branch.
            progress_total = max(
                progress_step + 2,
                4 + 4 * (len(apparition_groups) - 1),
            )
        if (
            multi_apparition and not nongravitational
        ):
            # A catalogue solution can already encode the correct integer-
            # revolution branch across all observed returns.  This is the
            # normal situation for recovered comets such as 206P: JPL's
            # published common solution is gravity-only.  Fitting the latest
            # return first used to destroy that branch and then incorrectly
            # required the user to enable A1/A2/A3 to search for it again.
            # Verify the supplied state before changing it.  When it predicts
            # every apparition, make only a bounded correction on a balanced
            # sample and validate the result independently on every row.
            catalogue_verified = False
            verified_report: list[
                tuple[float | None, float | None, bool | None]
            ] | None = None
            verified_appraisals: list[tuple[float, float]] | None = None
            verified_result: tuple[OrbitSolution, float, int] | None = None
            if initial is not None:
                gravity_initial = replace(
                    initial, nongravitational_parameters=None,
                )
                try:
                    report_progress(
                        "Ověřuji katalogovou gravitační dráhu na všech návratech…",
                        4,
                    )
                    catalogue_rms, catalogue_normalized, catalogue_count = (
                        _evaluate_solution_observations(
                            observations, gravity_initial, None, cancel_callback,
                        )
                    )
                    appraisals = [
                        _robust_apparition_appraisal(group)
                        for group in apparition_groups
                    ]
                    catalogue_verified = bool(appraisals) and all(
                        appraisal[0] <= LINKAGE_ACCEPT_RMS_ARCSEC
                        and appraisal[1] <= LINKAGE_ACCEPT_NORMALIZED_RMS
                        for appraisal in appraisals
                    )
                    if catalogue_verified:
                        gravity_initial.fit_observation_count = catalogue_count
                        gravity_initial.fit_available_observation_count = catalogue_count
                        gravity_initial.fit_rms_arcsec = catalogue_rms
                        gravity_initial.fit_normalized_rms = catalogue_normalized
                        for observation in observations:
                            if (
                                not observation.excluded_from_fit
                                and observation.residual_ra_arcsec is not None
                                and observation.residual_dec_arcsec is not None
                            ):
                                observation.used_in_fit = True
                        verified_result = (
                            gravity_initial, catalogue_rms, catalogue_count,
                        )
                        verified_report = [
                            (
                                observation.residual_ra_arcsec,
                                observation.residual_dec_arcsec,
                                observation.used_in_fit,
                            )
                            for observation in observations
                        ]
                        verified_appraisals = [
                            appraisal[:2] for appraisal in appraisals
                        ]
                except OrbitFitCancelled:
                    raise
                except (
                    OrbitFitError, ValueError, RuntimeError,
                    FloatingPointError, OverflowError,
                ):
                    catalogue_verified = False

            if (
                catalogue_verified
                and verified_result is not None
                and verified_report is not None
                and verified_appraisals is not None
            ):
                balanced = _balanced_linkage_observations(apparition_groups)
                result = verified_result
                correction_accepted = False
                try:
                    report_progress(
                        "Krátce zpřesňuji ověřenou společnou gravitační dráhu…"
                    )
                    corrected = fit_once(
                        balanced, verified_result[0], constraints, truncate_years,
                        None, maximum_observations=None,
                        max_nfev=6, robust_loss=True, robust_scale=8.0,
                    )
                    corrected_rms, corrected_normalized, corrected_count = (
                        _evaluate_solution_observations(
                            observations, corrected[0], None, cancel_callback,
                        )
                    )
                    corrected_appraisals = [
                        _robust_apparition_appraisal(group)[:2]
                        for group in apparition_groups
                    ]
                    materially_worse = any(
                        corrected_group_rms > max(
                            baseline_rms + 1.0, baseline_rms * 1.5,
                        )
                        or corrected_group_normalized > max(
                            baseline_normalized + 1.0,
                            baseline_normalized * 1.5,
                        )
                        for (
                            corrected_group_rms, corrected_group_normalized,
                        ), (
                            baseline_rms, baseline_normalized,
                        ) in zip(corrected_appraisals, verified_appraisals)
                    )
                    all_returns_acceptable = all(
                        rms <= LINKAGE_ACCEPT_RMS_ARCSEC
                        and normalized <= LINKAGE_ACCEPT_NORMALIZED_RMS
                        for rms, normalized in corrected_appraisals
                    )
                    if all_returns_acceptable and not materially_worse:
                        corrected[0].fit_observation_count = len(balanced)
                        corrected[0].fit_available_observation_count = corrected_count
                        corrected[0].fit_rms_arcsec = corrected_rms
                        corrected[0].fit_normalized_rms = corrected_normalized
                        fitted_ids = {id(item) for item in balanced}
                        for observation in observations:
                            observation.used_in_fit = bool(
                                id(observation) in fitted_ids
                                and not observation.excluded_from_fit
                            )
                        result = (
                            corrected[0], corrected_rms, len(balanced),
                        )
                        correction_accepted = True
                except OrbitFitCancelled:
                    raise
                except (
                    OrbitFitError, ValueError, RuntimeError,
                    FloatingPointError, OverflowError,
                ):
                    correction_accepted = False

                if not correction_accepted:
                    result = verified_result
                    for observation, values in zip(observations, verified_report):
                        (
                            observation.residual_ra_arcsec,
                            observation.residual_dec_arcsec,
                            observation.used_in_fit,
                        ) = values
                for record, group in zip(apparition_linkage, apparition_groups):
                    record["status"] = "linked"
                    record["reason"] = (
                        "Ověřená katalogová gravitační větev; krátká společná "
                        "korekce byla potvrzena na všech načtených měřeních."
                        if correction_accepted else
                        "Katalogová gravitační větev byla přímo potvrzena na "
                        "všech načtených měřeních; krátká korekce byla odmítnuta, "
                        "protože zhoršila alespoň jeden návrat."
                    )
                    _update_apparition_linkage_metrics(record, group)
                result[0].fit_model_note = (
                    f"Společná gravitační dráha propojila všech "
                    f"{len(apparition_groups)} pozorovaných návratů. "
                    + (
                        "Ověřená katalogová větev byla krátce zpřesněna na "
                        "vyváženém vzorku a znovu potvrzena na celém oblouku."
                        if correction_accepted else
                        "Zachována je přímo ověřená katalogová větev, protože "
                        "omezená korekce neposkytla bezpečně lepší společné řešení."
                    )
                )
            else:
                # Without a verified common branch, a gravity-only correction
                # across decades can invent a low-quality compromise.  Retain
                # the conservative latest-return behaviour in that case.
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
                    "Bez ověřené společné katalogové větve byl gravitační fit "
                    "určen z posledního pozorovaného návratu; starší a budoucí "
                    "návraty jsou pouze integrované předpovědi."
                )
                for record in apparition_linkage:
                    record["status"] = "integrated_only"
                    record["reason"] = (
                        "Katalogová dráha nepotvrdila společnou gravitační větev; "
                        "návrat nebyl společně fitován."
                    )
                if apparition_linkage:
                    apparition_linkage[-1]["status"] = "seed_only"
                    apparition_linkage[-1]["reason"] = (
                        "Samostatný fit posledního návratu."
                    )
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
            # Sparse historical recoveries (notably the three 1951 plates of
            # 325P) cannot define an independent six-parameter orbit.  Before
            # altering a supplied catalogue state with a latest-return-only
            # fit, test whether that state already propagates onto every
            # observed apparition.  This preserves the correct revolution
            # branch when it is already known.
            if initial is not None and any(
                len(group) < 6 for group in apparition_groups[:-1]
            ):
                try:
                    report_progress(
                        "Ověřuji katalogovou dráhu na všech návratech…"
                    )
                    preflight = _evaluate_solution_observations(
                        observations, initial, seed_nongrav, cancel_callback,
                    )
                    appraisals = [
                        _robust_apparition_appraisal(group)
                        for group in apparition_groups
                    ]
                    catalogue_seed_verified = bool(appraisals) and all(
                        appraisal[0] <= 30.0 and appraisal[1] <= 8.0
                        for appraisal in appraisals
                    )
                    if catalogue_seed_verified:
                        for group, appraisal in zip(apparition_groups, appraisals):
                            if len(group) >= 6 or not appraisal[3]:
                                continue
                            retained_radial = [
                                float(np.hypot(
                                    item.residual_ra_arcsec or 0.0,
                                    item.residual_dec_arcsec or 0.0,
                                ))
                                for item in appraisal[2]
                            ]
                            reference = max(3.0, float(np.median(retained_radial)))
                            for outlier in appraisal[3]:
                                sigma_ra, sigma_dec = _observation_sigma_arcsec(outlier)
                                radial = float(np.hypot(
                                    outlier.residual_ra_arcsec or 0.0,
                                    outlier.residual_dec_arcsec or 0.0,
                                ))
                                normalized = float(np.hypot(
                                    (outlier.residual_ra_arcsec or 0.0) / sigma_ra,
                                    (outlier.residual_dec_arcsec or 0.0) / sigma_dec,
                                ))
                                if radial > max(30.0, 5.0 * reference) and normalized > 8.0:
                                    outlier.excluded_from_fit = True
                                    outlier.raw["automatic_exclusion"] = True
                                    outlier.raw["historical_duplicate_outlier"] = True
                                    outlier.raw["automatic_exclusion_reason"] = (
                                        "Řídký historický návrat: údaj odporuje nejméně "
                                        "dvěma dalším deskám na ověřené společné dráze."
                                    )
                        # Recompute aggregate statistics after the explicit
                        # sparse-return rejection.  The already propagated
                        # O-C values themselves remain valid.
                        physical: list[float] = []
                        normalized: list[float] = []
                        used_count = 0
                        for observation in observations:
                            if (
                                observation.excluded_from_fit
                                or observation.residual_ra_arcsec is None
                                or observation.residual_dec_arcsec is None
                            ):
                                continue
                            used_count += 1
                            sigma_ra, sigma_dec = _observation_sigma_arcsec(
                                observation
                            )
                            physical.extend((
                                float(observation.residual_ra_arcsec),
                                float(observation.residual_dec_arcsec),
                            ))
                            normalized.extend((
                                float(observation.residual_ra_arcsec) / sigma_ra,
                                float(observation.residual_dec_arcsec) / sigma_dec,
                            ))
                            observation.used_in_fit = True
                        verified_rms = float(np.sqrt(np.mean(np.square(physical))))
                        verified_normalized = float(np.sqrt(
                            np.mean(np.square(normalized))
                        ))
                        initial.fit_observation_count = used_count
                        initial.fit_available_observation_count = used_count
                        initial.fit_rms_arcsec = verified_rms
                        initial.fit_normalized_rms = verified_normalized
                        seed = (initial, verified_rms, used_count)
                        verified_catalogue_result = seed
                        verified_catalogue_report = [
                            (
                                observation.residual_ra_arcsec,
                                observation.residual_dec_arcsec,
                                observation.used_in_fit,
                            )
                            for observation in observations
                        ]
                        verified_catalogue_appraisals = [
                            appraisal[:2]
                            for appraisal in (
                                _robust_apparition_appraisal(group)
                                for group in apparition_groups
                            )
                        ]
                    else:
                        # The values belong only to this appraisal; the local
                        # seed below must establish its own complete O-C table.
                        for observation in observations:
                            observation.used_in_fit = False
                            observation.residual_ra_arcsec = None
                            observation.residual_dec_arcsec = None
                except (
                    OrbitFitError, ValueError, RuntimeError,
                    FloatingPointError, OverflowError,
                ):
                    catalogue_seed_verified = False
            if not catalogue_seed_verified:
                report_progress("Určuji lokální dráhu posledního návratu…")
                seed = fit_once(
                    recent_apparition, initial, constraints, truncate_years,
                    None, fixed_nongravitational=seed_nongrav,
                    maximum_observations=120,
                    # A present-day catalogue state can miss a pre-encounter
                    # apparition by several arcminutes when propagated back
                    # through a close planetary encounter (206P/Jupiter 2017).
                    # A narrow robust loss would classify that coherent shift
                    # as outliers and suppress the very gradient needed to
                    # capture the local 2008 branch.  Use a wide capture pass;
                    # the following pass restores normal robust weighting.
                    max_nfev=24, robust_loss=True, robust_scale=100.0,
                )
            # A four-parameter comet model may need more than the quick seed
            # pass even within the modern apparition.  Only pay for a denser
            # polishing pass when the first result is visibly worse than the
            # expected CCD astrometry; this specifically prevents carrying
            # 8–18 arcsecond 1986 residuals into the historical linkage.
            seed_normalized = seed[0].fit_normalized_rms
            if (
                not catalogue_seed_verified
                and (
                seed[1] > 3.0
                or seed_normalized is None
                or not np.isfinite(seed_normalized)
                or seed_normalized > 3.0
                )
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
                        max_nfev=24, robust_loss=True, robust_scale=8.0,
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
            if not _linkage_trial_is_acceptable(seed):
                # The current catalogue epoch can lie beyond a strong
                # encounter, so even a wider differential capture may remain
                # in the wrong pre-encounter basin.  Determine the modern
                # apparition independently from its own angles before trying
                # to identify it with the historical return.
                try:
                    report_progress(
                        "Hledám nezávislé větve posledního návratu…"
                    )
                    local_rows = _normal_place_observations(recent_apparition)
                    preliminary = initial_orbit_from_observations(
                        local_rows, return_candidates=True,
                    )
                    assert isinstance(preliminary, list)
                    local_trials: list[tuple[OrbitSolution, float, int]] = []
                    for local_initial in preliminary[:8]:
                        try:
                            local_trial = fit_once(
                                recent_apparition, local_initial, constraints,
                                truncate_years, None,
                                fixed_nongravitational=seed_nongrav,
                                maximum_observations=120, max_nfev=32,
                                robust_loss=True, robust_scale=100.0,
                            )
                            if np.isfinite(local_trial[1]):
                                local_trials.append(local_trial)
                        except (
                            OrbitFitError, ValueError, RuntimeError,
                            FloatingPointError,
                        ):
                            continue
                    if local_trials:
                        local_seed = min(
                            local_trials,
                            key=lambda item: (
                                item[0].fit_normalized_rms
                                if item[0].fit_normalized_rms is not None
                                and np.isfinite(item[0].fit_normalized_rms)
                                else np.inf,
                                item[1],
                            ),
                        )
                        if local_seed[1] < seed[1]:
                            seed = local_seed
                except (
                    OrbitFitError, ValueError, RuntimeError,
                    FloatingPointError,
                ):
                    pass
            secondary_track = (
                [] if catalogue_seed_verified else
                _coherent_secondary_track_candidates(recent_apparition)
            )
            if secondary_track:
                secondary_track_exclusions = len(secondary_track)
                for observation in secondary_track:
                    observation.excluded_from_fit = True
                    observation.raw["automatic_exclusion"] = True
                    observation.raw["coherent_secondary_track"] = True
                    observation.raw["automatic_exclusion_reason"] = (
                        "Souvislá vedlejší astrometrická stopa; možný fragment "
                        "nebo sekundární jádro."
                    )
                report_progress(
                    "Odděluji souvislou vedlejší astrometrickou stopu…"
                )
                seed = fit_once(
                    recent_apparition, seed[0], constraints, truncate_years,
                    None, fixed_nongravitational=seed_nongrav,
                    maximum_observations=240,
                    max_nfev=16, robust_loss=True,
                )
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
                    (
                        "Publikovaná společná dráha před korekcí ověřena na všech "
                        "návratech; čeká na závěrečný společný fit."
                    )
                    if catalogue_seed_verified else
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
                strict_polish=long_arc,
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

        def failed_apparitions() -> list[tuple[int | str, float, float]]:
            failed: list[tuple[int | str, float, float]] = []
            for index, group in enumerate(apparition_groups):
                metrics: dict[str, object] = {}
                _update_apparition_linkage_metrics(metrics, group)
                if "rms_arcsec" not in metrics or "normalized_rms" not in metrics:
                    continue
                rms = float(metrics.get("rms_arcsec", np.inf))
                normalized = float(metrics.get("normalized_rms", np.inf))
                if rms > 120.0 or normalized > 8.0:
                    year = (
                        apparition_linkage[index].get("return_year", "?")
                        if index < len(apparition_linkage) else "?"
                    )
                    failed.append((year, rms, normalized))
            return failed

        if multi_apparition_seed is not None and seed_report is not None:
            # Link one earlier return at a time.  Jumping directly from the
            # 1986 state to a simultaneous 1835+1910+1986 correction leaves a
            # numerical Jacobian thousands of sigma from its local basin and
            # was the source of the spurious degree-sized Halley residuals.
            last_good = multi_apparition_seed
            last_good_report = seed_report
            linked_count = (
                len(apparition_groups) if catalogue_seed_verified else 1
            )
            linked_parameter_names: set[str] = (
                set(fitted_nongrav(last_good[0]))
                if catalogue_seed_verified else set()
            )
            if catalogue_seed_verified:
                for group_index, group in enumerate(apparition_groups):
                    apparition_linkage[group_index]["status"] = "linked"
                    apparition_linkage[group_index]["reason"] = (
                        "Publikovaná společná N-body větev byla přímo ověřena "
                        "na astrometrii tohoto návratu před diferenciální korekcí."
                    )
                    _update_apparition_linkage_metrics(
                        apparition_linkage[group_index], group,
                    )
            linkage_range = (
                range(len(apparition_groups) - 2, -1, -1)
                if not catalogue_seed_verified else range(0)
            )
            for first_group in linkage_range:
                stage_groups = apparition_groups[first_group:]
                stage_observations = _balanced_linkage_observations(stage_groups)
                stage_count = len(apparition_groups) - first_group
                sparse_linkage = len(apparition_groups[first_group]) < 6
                all_parameters = fitted_nongrav(last_good[0])
                # Two apparitions usually constrain the revolution count and
                # osculating state much better than they constrain secular
                # comet acceleration.  Releasing state and A1/A2 together at
                # this first bridge can produce a deceptively low-RMS branch
                # with wildly displaced A terms; the next older apparition
                # then misses by many arcseconds.  Preserve the requested or
                # catalogue force law for the first bridge.  Once a third
                # return is present the longer time base can determine A1/A2
                # rather than merely trading them against the state vector.
                hold_secular_parameters = bool(
                    stage_count == 2 and any(
                        name in all_parameters
                        for name in ("A1", "A2", "A3", "DT")
                    )
                )
                held_parameters = (
                    dict(all_parameters) if hold_secular_parameters else {}
                )
                # Classical apparition linking did not repeatedly solve all
                # orbital elements over a century.  It first corrected a few
                # secular/non-gravitational terms from normal places, then
                # performed one full differential correction.  A1/A2 provide
                # that cheap phase/period bridge here; A3 and DT are released
                # only in the final all-apparition polish.
                linkage_names = tuple(
                    name for name in ("A1", "A2") if name in all_parameters
                ) or tuple(
                    name for name in ("A3", "DT") if name in all_parameters
                )[:1]
                linkage_parameters = {
                    name: all_parameters[name] for name in linkage_names
                }
                linkage_parameters.update({
                    name: all_parameters[name]
                    for name in ("R_EXPONENT",)
                    if name in all_parameters
                })
                linked_parameter_names.update(
                    held_parameters if hold_secular_parameters
                    else linkage_parameters
                )
                candidate = None
                candidate_source = "propagated"
                best_failed_trial: dict[str, object] | None = None
                failures: list[str] = []
                identification_diagnostic = ""
                return_year = apparition_linkage[first_group].get(
                    "return_year", "?"
                )

                # The newer-return fit can already be the correct common
                # dynamical branch.  In that case an encounter-sensitive
                # differential correction is optional refinement, not a
                # prerequisite for accepting the linkage.  Validate the
                # exact propagation on every original row before launching
                # alternative local/catalogue branches.  This is the 206P
                # case: the propagated 1892 return is about 44 arcsec (3.7σ),
                # comfortably inside the named linkage limits, while an
                # unconstrained correction leaves that basin entirely.
                validation_rows = [
                    observation
                    for group in stage_groups
                    for observation in group
                ]
                validation_report = [
                    (
                        observation.residual_ra_arcsec,
                        observation.residual_dec_arcsec,
                        observation.used_in_fit,
                    )
                    for observation in validation_rows
                ]
                validation_diagnostic = ""
                validation_evaluated_report: list[
                    tuple[float | None, float | None, bool | None]
                ] = []
                try:
                    # Integrate each apparition independently from the same
                    # osculating state.  Besides matching the per-return audit
                    # shown to the user, this avoids letting one problematic
                    # timestamp invalidate an otherwise verified century-long
                    # branch as one opaque all-epochs integration failure.
                    validation_results: list[
                        tuple[float, float, int]
                    ] = []
                    validation_appraisals: list[tuple[float, float]] = []
                    for group in stage_groups:
                        validation_results.append(
                            _evaluate_solution_observations(
                                group, last_good[0], all_parameters,
                                cancel_callback,
                            )
                        )
                        validation_appraisals.append(
                            _robust_apparition_appraisal(group)[:2]
                        )
                    validation_evaluated_report = [
                        (
                            observation.residual_ra_arcsec,
                            observation.residual_dec_arcsec,
                            observation.used_in_fit,
                        )
                        for observation in observations
                    ]
                    validation_count = sum(
                        item[2] for item in validation_results
                    )
                    validation_rms = float(np.sqrt(sum(
                        item[0] ** 2 * item[2]
                        for item in validation_results
                    ) / max(1, validation_count)))
                    validation_normalized = float(np.sqrt(sum(
                        item[1] ** 2 * item[2]
                        for item in validation_results
                    ) / max(1, validation_count)))
                    propagation_is_acceptable = all(
                            rms <= LINKAGE_ACCEPT_RMS_ARCSEC
                            and normalized
                            <= LINKAGE_ACCEPT_NORMALIZED_RMS
                            for rms, normalized in validation_appraisals
                    )
                    validation_diagnostic = (
                        "přímá propagace po návratech: "
                        + ", ".join(
                            f"{apparition_linkage[first_group + offset].get('year_start', '?')} "
                            f"{rms:.2f}″/{normalized:.2f}σ"
                            for offset, (rms, normalized) in enumerate(
                                validation_appraisals
                            )
                        )
                    )
                except (
                    OrbitFitError, ValueError, RuntimeError,
                    FloatingPointError, OverflowError,
                ) as exc:
                    propagation_is_acceptable = False
                    validation_diagnostic = f"přímá propagace selhala: {exc}"
                if not propagation_is_acceptable:
                    # A rejected read-only appraisal must not leak its O-C
                    # values into the subsequent independent branch trials.
                    for observation, report_values in zip(
                        validation_rows, validation_report,
                    ):
                        (
                            observation.residual_ra_arcsec,
                            observation.residual_dec_arcsec,
                            observation.used_in_fit,
                        ) = report_values
                    failures.append(validation_diagnostic)
                if propagation_is_acceptable:
                    validated_solution = replace(
                        last_good[0],
                        fit_rms_arcsec=validation_rms,
                        fit_normalized_rms=validation_normalized,
                        fit_available_observation_count=validation_count,
                    )
                    last_good = (
                        validated_solution, validation_rms, validation_count,
                    )
                    linked_count += 1
                    linked_parameter_names.update(all_parameters)
                    for group_index in range(
                        first_group, len(apparition_groups)
                    ):
                        apparition_linkage[group_index]["status"] = "linked"
                        apparition_linkage[group_index]["reason"] = (
                            "Společná N-body větev byla potvrzena přesnou "
                            "propagací na astrometrii tohoto návratu ještě "
                            "před diferenciální korekcí."
                        )
                        _update_apparition_linkage_metrics(
                            apparition_linkage[group_index],
                            apparition_groups[group_index],
                        )
                    last_good_report = [
                        (
                            observation.residual_ra_arcsec,
                            observation.residual_dec_arcsec,
                            observation.used_in_fit,
                        )
                        for observation in observations
                    ]
                    result = last_good
                    report_progress(
                        f"Návrat {return_year}: přesná propagace společnou "
                        "větev potvrdila; pokračuji bezpečným zpřesněním…"
                    )
                    continue

                def remember_failed_trial(trial_result) -> None:
                    nonlocal best_failed_trial
                    normalized_value = trial_result[0].fit_normalized_rms
                    score = (
                        float(normalized_value)
                        if normalized_value is not None
                        and np.isfinite(normalized_value)
                        else np.inf,
                        float(trial_result[1]),
                    )
                    entry: dict[str, object] = {
                        "score": score,
                        "trial": trial_result,
                        "appraisals": [],
                        "report": [],
                    }
                    if (
                        best_failed_trial is None
                        or score < best_failed_trial["score"]
                    ):
                        best_failed_trial = entry

                def appraise_best_failed_trial() -> None:
                    """Exactly score only the best cheap candidate by return."""
                    if best_failed_trial is None:
                        return
                    trial_result = best_failed_trial["trial"]
                    if (
                        not np.isfinite(trial_result[1])
                        or trial_result[1] > 5000.0
                    ):
                        return
                    previous_report = [
                        (
                            observation.residual_ra_arcsec,
                            observation.residual_dec_arcsec,
                            observation.used_in_fit,
                        )
                        for observation in observations
                    ]
                    try:
                        trial_parameters = fitted_nongrav(trial_result[0])
                        appraisals: list[tuple[float, float]] = []
                        for group in stage_groups:
                            _evaluate_solution_observations(
                                group, trial_result[0], trial_parameters,
                                cancel_callback,
                            )
                            appraisals.append(
                                _robust_apparition_appraisal(group)[:2]
                            )
                        best_failed_trial["appraisals"] = appraisals
                        best_failed_trial["report"] = [
                            (
                                observation.residual_ra_arcsec,
                                observation.residual_dec_arcsec,
                                observation.used_in_fit,
                            )
                            for observation in observations
                        ]
                    except (
                        OrbitFitError, ValueError, RuntimeError,
                        FloatingPointError, OverflowError,
                    ):
                        best_failed_trial["appraisals"] = []
                        best_failed_trial["report"] = []
                    finally:
                        for observation, values in zip(
                            observations, previous_report,
                        ):
                            (
                                observation.residual_ra_arcsec,
                                observation.residual_dec_arcsec,
                                observation.used_in_fit,
                            ) = values

                def attempt_arc_extension():
                    """Grow the old arc before trying unrelated full-fit basins."""
                    older_rows = sorted(
                        (
                            item for item in apparition_groups[first_group]
                            if not item.excluded_from_fit
                        ),
                        key=lambda item: parse_observation_time(item.time),
                    )
                    extension_counts = _arc_extension_counts(len(older_rows))
                    # Always continue from the verified newer-return branch.
                    # A failed whole-arc result may already have crossed to an
                    # adjacent integer-revolution solution.
                    extension_seed = last_good[0]
                    extension_parameters = fitted_nongrav(extension_seed)
                    extension_trial = None
                    try:
                        for count in extension_counts:
                            report_progress(
                                f"Návrat {return_year}: postupně připojuji "
                                f"{count}/{len(older_rows)} měření…"
                            )
                            extension_groups = [
                                older_rows[-count:],
                                *apparition_groups[first_group + 1:],
                            ]
                            # These are branch-finding normal places only.
                            # The exact final correction below still receives
                            # every original enabled observation.
                            extension_observations = (
                                _balanced_linkage_observations(
                                    extension_groups,
                                    maximum_per_group=max(6, min(24, count)),
                                )
                            )
                            # Correct accumulated phase first, with the state
                            # held fixed, and release the Cartesian state only
                            # after that reduced problem has converged.
                            if hold_secular_parameters:
                                extension_prefit = (
                                    extension_seed, np.inf, 0
                                )
                            else:
                                extension_prefit = fit_once(
                                    extension_observations, extension_seed,
                                    constraints, truncate_years,
                                    extension_parameters,
                                    maximum_observations=None,
                                    max_nfev=6 if count > 24 else 10,
                                    robust_loss=True,
                                    robust_scale=(
                                        50.0 if count <= 24 else 12.0
                                    ),
                                    screening=True,
                                    fit_state=False,
                                )
                                extension_parameters = fitted_nongrav(
                                    extension_prefit[0]
                                )
                            extension_trial = fit_once(
                                extension_observations, extension_prefit[0],
                                constraints, truncate_years,
                                None if hold_secular_parameters
                                else extension_parameters,
                                fixed_nongravitational=(
                                    held_parameters
                                    if hold_secular_parameters else None
                                ),
                                maximum_observations=None,
                                max_nfev=10 if count > 24 else 16,
                                robust_loss=True,
                                robust_scale=50.0 if count <= 12 else 12.0,
                                screening=True,
                            )
                            extension_seed = extension_trial[0]
                            extension_parameters = fitted_nongrav(
                                extension_seed
                            )
                            if (
                                not np.isfinite(extension_trial[1])
                                or extension_trial[1]
                                > LINKAGE_DIVERGENCE_RMS_ARCSEC
                            ):
                                raise OrbitFitError(
                                    "postupné rozšíření se vzdálilo od "
                                    "astrometrie"
                                )
                        if extension_trial is None:
                            return None, "postupné rozšíření nemělo žádná měření"
                        extension_normalized = (
                            extension_trial[0].fit_normalized_rms
                        )
                        if _linkage_trial_is_acceptable(extension_trial):
                            linked_parameter_names.update(extension_parameters)
                            return extension_trial, None
                        remember_failed_trial(extension_trial)
                        normalized_text = (
                            "nedostupné"
                            if extension_normalized is None
                            or not np.isfinite(extension_normalized)
                            else f"{extension_normalized:.2f}σ"
                        )
                        return None, (
                            "postupné rozšíření oblouku: RMS "
                            f"{extension_trial[1]:.2f}″; {normalized_text}"
                        )
                    except (
                        OrbitFitError, ValueError, RuntimeError,
                        FloatingPointError,
                    ) as exc:
                        return None, f"postupné rozšíření oblouku: {exc}"
                try:
                    if hold_secular_parameters:
                        report_progress(
                            f"Návrat {return_year}: hledám společný stav při "
                            "pevných katalogových A/DT…"
                        )
                        prefit = last_good
                    else:
                        report_progress(
                            f"Návrat {return_year}: rychlý apariční předfit "
                            f"({'+'.join(linkage_names)})…"
                        )
                        prefit = fit_once(
                            stage_observations, last_good[0], constraints,
                            truncate_years, linkage_parameters,
                            maximum_observations=min(160, 40 * stage_count),
                            max_nfev=8, robust_loss=True, robust_scale=50.0,
                            screening=True, fit_state=False,
                        )
                    report_progress(
                        f"Návrat {return_year}: krátká společná korekce…"
                    )
                    trial = fit_once(
                        stage_observations, prefit[0], constraints,
                        truncate_years,
                        None if hold_secular_parameters else linkage_parameters,
                        fixed_nongravitational=(
                            held_parameters if hold_secular_parameters else None
                        ),
                        maximum_observations=min(240, 60 * stage_count),
                        max_nfev=6, robust_loss=True, robust_scale=25.0,
                        screening=True, preserve_linkage_branch=True,
                    )
                    normalized = trial[0].fit_normalized_rms
                    if (
                        np.isfinite(trial[1]) and trial[1] <= 30.0
                        and normalized is not None and np.isfinite(normalized)
                        and normalized > 8.0
                    ):
                        report_progress(
                            f"Návrat {return_year}: pokračuji v konvergenci "
                            "těsné kandidátní větve…"
                        )
                        trial = fit_once(
                            stage_observations, trial[0], constraints,
                            truncate_years,
                            None if hold_secular_parameters
                            else linkage_parameters,
                            fixed_nongravitational=(
                                held_parameters
                                if hold_secular_parameters else None
                            ),
                            maximum_observations=min(320, 80 * stage_count),
                            max_nfev=24, robust_loss=True,
                            robust_scale=8.0, screening=False,
                        )
                        normalized = trial[0].fit_normalized_rms
                    if _linkage_trial_is_acceptable(trial):
                        candidate = trial
                    else:
                        remember_failed_trial(trial)
                        normalized_text = (
                            "nedostupné" if normalized is None
                            or not np.isfinite(normalized) else f"{normalized:.2f}σ"
                        )
                        failures.append(
                            f"rychlá větev: RMS {trial[1]:.2f}″; {normalized_text}"
                        )
                    # This continuation is the normal path, not a last-resort
                    # recovery after several expensive whole-arc fits.  It is
                    # deliberately attempted before a standalone old-return
                    # orbit, catalogue basin, force grid, or expanded A model.
                    if candidate is None and best_failed_trial is not None:
                        extension_candidate, extension_failure = (
                            attempt_arc_extension()
                        )
                        if extension_candidate is not None:
                            candidate = extension_candidate
                            candidate_source = "arc_extension"
                        elif extension_failure:
                            failures.append(extension_failure)
                    # A propagated solution can be only tens of arcseconds
                    # from an older apparition and still lie outside the
                    # narrow basin in which a century-scale numerical
                    # Jacobian converges.  This is common after one or more
                    # unobserved revolutions.  Determine an independent local
                    # state from that apparition's own astrometry, then retry
                    # the same joint correction from that second branch.  The
                    # local state is never accepted as the answer by itself:
                    # every linked return must still pass the common O-C test.
                    if (
                        candidate is None
                        and len(apparition_groups[first_group]) >= 6
                    ):
                        # Identification must not depend on non-gravitational
                        # coefficients learned only after the recovery.  Meyer
                        # could recognise the 1892 orbit from a backwards
                        # integration of the short 2008 orbit; A1/A2 belong to
                        # the subsequent linked correction.  Retain only force
                        # law metadata here (it has no effect while A=0).
                        identification_force = {
                            name: value for name, value in all_parameters.items()
                            if name == "R_EXPONENT"
                        }
                        try:
                            return_year = apparition_linkage[first_group].get(
                                "return_year", "?"
                            )
                            report_progress(
                                f"Návrat {return_year}: určuji nezávislou "
                                "lokální dráhu z astrometrie…"
                            )
                            local_rows = _normal_place_observations(
                                apparition_groups[first_group]
                            )
                            preliminary = initial_orbit_from_observations(
                                local_rows, return_candidates=True,
                            )
                            assert isinstance(preliminary, list)
                            local_solutions: list[
                                tuple[OrbitSolution, float, int]
                            ] = []
                            # Each hypothesis comes solely from the old
                            # apparition.  This is the identification stage;
                            # no propagated modern state is smuggled in as a
                            # nominally "independent" local orbit.
                            for local_initial in preliminary[:8]:
                                try:
                                    local_trial = fit_once(
                                        local_rows, local_initial, constraints,
                                        truncate_years, None,
                                        fixed_nongravitational=(
                                            identification_force
                                        ),
                                        maximum_observations=None, max_nfev=28,
                                        robust_loss=True, robust_scale=8.0,
                                    )
                                    if np.isfinite(local_trial[1]):
                                        local_solutions.append(local_trial)
                                except (
                                    OrbitFitError, ValueError, RuntimeError,
                                    FloatingPointError,
                                ):
                                    continue
                            if not local_solutions:
                                raise OrbitFitError(
                                    "Starší návrat neposkytl použitelnou "
                                    "samostatnou předběžnou dráhu."
                                )
                            local_solutions.sort(key=lambda item: item[1])
                            failures.append(
                                "nejlepší nezávislá lokální dráha: "
                                f"RMS {local_solutions[0][1]:.2f}″"
                            )
                            # Reproduce the documented recovery sequence: fit
                            # both apparitions independently, integrate the
                            # modern orbit back, and test whether their orbit
                            # geometry identifies the same comet.  The test is
                            # deliberately not an O-C gate; after many missing
                            # revolutions the accumulated phase can be poor
                            # while q, e, the orbital plane and the apsidal
                            # direction remain recognisably related.
                            if (
                                candidate is None
                                and _needs_historical_identification(
                                    local_solutions[0][0].epoch_jd,
                                    last_good[0].epoch_jd,
                                    last_good[0],
                                )
                            ):
                                report_progress(
                                    f"Návrat {return_year}: porovnávám "
                                    "nezávislé gravitační dráhy aparic…"
                                )
                                try:
                                    old_seed = local_solutions[0][0]
                                    modern_at_old = _propagate_solution_to_epoch(
                                        last_good[0], old_seed.epoch_jd,
                                        identification_force,
                                    )
                                    similar, similarity = (
                                        _apparition_orbit_similarity(
                                            modern_at_old, old_seed,
                                        )
                                    )
                                    modern_period = 2.0 * np.pi * np.sqrt(
                                        last_good[0].semi_major_axis_au ** 3
                                        / MU_SUN
                                    )
                                    revolutions = max(1, int(round(
                                        abs(
                                            last_good[0].epoch_jd
                                            - old_seed.epoch_jd
                                        )
                                        / modern_period
                                    )))
                                    similarity_text = (
                                        f"N≈{revolutions}; rovina "
                                        f"{similarity['plane_deg']:.2f}°, "
                                        f"apsidy {similarity['apsidal_deg']:.2f}°, "
                                        f"Δq {100.0 * similarity['q_fraction']:.1f} %, "
                                        f"Δe {similarity['e_difference']:.3f}"
                                    )
                                    if not similar:
                                        identification_diagnostic = (
                                            "nezávislé dráhy nepotvrdily "
                                            f"identitu ({similarity_text})"
                                        )
                                        failures.append(
                                            identification_diagnostic
                                        )
                                    else:
                                        report_progress(
                                            f"Návrat {return_year}: dráhy jsou "
                                            f"dynamicky podobné ({similarity_text}); "
                                            "hledám periodovou větev…"
                                        )
                                        phase_report = [
                                            (
                                                observation.residual_ra_arcsec,
                                                observation.residual_dec_arcsec,
                                                observation.used_in_fit,
                                            )
                                            for observation in observations
                                        ]
                                        # Search the accumulated phase from
                                        # the well-fitted modern apparition.
                                        # Starting from the independent old
                                        # orbit would preserve 1892 while a
                                        # tiny period change could miss 2008
                                        # by tens of degrees.  The old orbit
                                        # is used above to identify the branch,
                                        # not as the phase-search origin.
                                        period_base = last_good[0]
                                        period_seed = period_base
                                        period_diagnostic = ""
                                        try:
                                            phase_groups = [
                                                _balanced_linkage_observations(
                                                    [group],
                                                    maximum_per_group=8,
                                                )
                                                for group in stage_groups
                                            ]
                                            phase_rows = [
                                                observation
                                                for group in phase_groups
                                                for observation in group
                                            ]
                                            period_changes = (
                                                0.0,
                                                -1e-6, 1e-6,
                                                -3e-6, 3e-6,
                                                -1e-5, 1e-5,
                                                -3e-5, 3e-5,
                                                -1e-4, 1e-4,
                                                -3e-4, 3e-4,
                                                -1e-3, 1e-3,
                                                -3e-3, 3e-3,
                                            )
                                            period_trials: list[
                                                tuple[
                                                    tuple[float, float],
                                                    float,
                                                    OrbitSolution,
                                                    list[tuple[float, float]],
                                                ]
                                            ] = []
                                            def evaluate_period_change(
                                                period_change: float,
                                            ) -> None:
                                                try:
                                                    trial_seed = (
                                                        _transverse_period_candidate(
                                                            period_base,
                                                            period_change,
                                                        )
                                                    )
                                                    (
                                                        _, trial_normalized, _,
                                                    ) = (
                                                        _evaluate_solution_observations(
                                                            phase_rows,
                                                            trial_seed,
                                                            identification_force,
                                                            cancel_callback,
                                                        )
                                                    )
                                                    appraisals = [
                                                        _robust_apparition_appraisal(
                                                            group
                                                        )[:2]
                                                        for group in phase_groups
                                                    ]
                                                    score = (
                                                        max(
                                                            item[1]
                                                            for item in appraisals
                                                        ),
                                                        trial_normalized,
                                                    )
                                                    if not np.all(np.isfinite(score)):
                                                        return
                                                    period_trials.append((
                                                        score,
                                                        period_change,
                                                        trial_seed,
                                                        appraisals,
                                                    ))
                                                except OrbitFitCancelled:
                                                    raise
                                                except (
                                                    OrbitFitError, ValueError,
                                                    RuntimeError,
                                                    FloatingPointError,
                                                    OverflowError,
                                                ):
                                                    # A failed period hypothesis
                                                    # is not a failure of the
                                                    # independently verified
                                                    # identity branch.
                                                    return

                                            for period_change in period_changes:
                                                evaluate_period_change(
                                                    period_change
                                                )
                                            if period_trials:
                                                # Refine once on both sides of
                                                # the best logarithmic-grid
                                                # hypothesis.  This resolves
                                                # the small phase correction
                                                # without turning the branch
                                                # search into another costly
                                                # multidimensional fit.
                                                ordered_trials = sorted(
                                                    period_trials,
                                                    key=lambda item: item[1],
                                                )
                                                best_trial = min(
                                                    period_trials,
                                                    key=lambda item: item[0],
                                                )
                                                best_index = ordered_trials.index(
                                                    best_trial
                                                )
                                                if 0 < best_index < (
                                                    len(ordered_trials) - 1
                                                ):
                                                    best_change = best_trial[1]
                                                    evaluate_period_change(
                                                        0.5 * (
                                                            ordered_trials[
                                                                best_index - 1
                                                            ][1] + best_change
                                                        )
                                                    )
                                                    evaluate_period_change(
                                                        0.5 * (
                                                            best_change
                                                            + ordered_trials[
                                                                best_index + 1
                                                            ][1]
                                                        )
                                                    )
                                            if not period_trials:
                                                raise OrbitFitError(
                                                    "Hledání periodové větve "
                                                    "neposkytlo konečný stav."
                                                )
                                            (
                                                _, best_period_change,
                                                period_seed, period_appraisals,
                                            ) = min(
                                                period_trials,
                                                key=lambda item: item[0],
                                            )
                                            old_period = (
                                                2.0 * np.pi * np.sqrt(
                                                    period_base.semi_major_axis_au
                                                    ** 3 / MU_SUN
                                                )
                                            )
                                            best_period = (
                                                2.0 * np.pi * np.sqrt(
                                                    period_seed.semi_major_axis_au
                                                    ** 3 / MU_SUN
                                                )
                                            )
                                            period_diagnostic = (
                                                "periodová větev "
                                                f"Δvₜ/vₜ={best_period_change:+.1e}, "
                                                f"ΔP={best_period - old_period:+.4f} d: "
                                                + ", ".join(
                                                    f"{apparition_linkage[first_group + offset].get('year_start', '?')} "
                                                    f"{rms:.2f}″/{normalized:.2f}σ"
                                                    for offset, (
                                                        rms, normalized,
                                                    ) in enumerate(
                                                        period_appraisals
                                                    )
                                                )
                                            )
                                        finally:
                                            for observation, values in zip(
                                                observations, phase_report,
                                            ):
                                                (
                                                    observation.residual_ra_arcsec,
                                                    observation.residual_dec_arcsec,
                                                    observation.used_in_fit,
                                                ) = values
                                        report_progress(
                                            f"Návrat {return_year}: "
                                            f"{period_diagnostic}; koriguji "
                                            "společnou dráhu…"
                                        )
                                        # The published A terms are not used
                                        # to establish the identity.  Start
                                        # the linked correction at zero and
                                        # let the two apparitions determine
                                        # them from their accumulated phase.
                                        identity_parameters = {
                                            name: (
                                                value
                                                if name == "R_EXPONENT"
                                                else 0.0
                                            )
                                            for name, value in
                                            linkage_parameters.items()
                                        }
                                        identity_prefit = (
                                            period_seed,
                                            np.inf,
                                            len(stage_observations),
                                        )
                                        if any(
                                            name in identity_parameters
                                            for name in ("A1", "A2", "A3", "DT")
                                        ):
                                            identity_prefit = fit_once(
                                                stage_observations, period_seed,
                                                constraints, truncate_years,
                                                identity_parameters,
                                                maximum_observations=min(
                                                    160, 40 * stage_count
                                                ),
                                                max_nfev=36,
                                                robust_loss=True,
                                                robust_scale=200.0,
                                                screening=True,
                                                fit_state=False,
                                            )
                                            identity_parameters = {
                                                name: value for name, value in
                                                fitted_nongrav(
                                                    identity_prefit[0]
                                                ).items()
                                                if name in identity_parameters
                                            }
                                        identity_trial = fit_once(
                                            stage_observations,
                                            identity_prefit[0],
                                            constraints, truncate_years,
                                            identity_parameters or None,
                                            maximum_observations=min(
                                                320, 80 * stage_count
                                            ),
                                            max_nfev=48,
                                            robust_loss=True,
                                            robust_scale=25.0,
                                            screening=True,
                                        )
                                        if _linkage_trial_is_acceptable(
                                            identity_trial
                                        ):
                                            candidate = identity_trial
                                            candidate_source = (
                                                "historical_identification"
                                            )
                                            linked_parameter_names.update(
                                                identity_parameters
                                            )
                                        else:
                                            remember_failed_trial(
                                                identity_trial
                                            )
                                            identification_diagnostic = (
                                                "společná korekce po potvrzení "
                                                "identity: RMS "
                                                f"{identity_trial[1]:.2f}″; "
                                                f"{similarity_text}; "
                                                f"{period_diagnostic}"
                                            )
                                            failures.append(
                                                identification_diagnostic
                                            )
                                except (
                                    OrbitFitError, ValueError, RuntimeError,
                                    FloatingPointError, OverflowError,
                                ) as bridge_exc:
                                    identification_diagnostic = (
                                        "porovnání nezávislých aparičních drah: "
                                        f"{bridge_exc}"
                                    )
                                    failures.append(identification_diagnostic)
                            for local_seed in local_solutions[:3]:
                                if candidate is not None:
                                    break
                                local_prefit = local_seed
                                if not hold_secular_parameters:
                                    report_progress(
                                        f"Návrat {return_year}: rychlý předfit "
                                        "nezávislé lokální větve…"
                                    )
                                    local_prefit = fit_once(
                                        stage_observations, local_seed[0],
                                        constraints, truncate_years,
                                        linkage_parameters,
                                        maximum_observations=min(
                                            160, 40 * stage_count
                                        ),
                                        max_nfev=8, robust_loss=True,
                                        robust_scale=50.0, screening=True,
                                        fit_state=False,
                                    )
                                trial = fit_once(
                                    stage_observations, local_prefit[0],
                                    constraints, truncate_years,
                                    None if hold_secular_parameters
                                    else linkage_parameters,
                                    fixed_nongravitational=(
                                        held_parameters
                                        if hold_secular_parameters else None
                                    ),
                                    maximum_observations=min(
                                        240, 60 * stage_count
                                    ),
                                    max_nfev=8, robust_loss=True,
                                    robust_scale=25.0, screening=True,
                                )
                                normalized = trial[0].fit_normalized_rms
                                if (
                                    np.isfinite(trial[1])
                                    and trial[1] <= 30.0
                                    and normalized is not None
                                    and np.isfinite(normalized)
                                    and normalized > 8.0
                                ):
                                    report_progress(
                                        f"Návrat {return_year}: pokračuji v "
                                        "konvergenci lokální větve…"
                                    )
                                    trial = fit_once(
                                        stage_observations, trial[0],
                                        constraints, truncate_years,
                                        None if hold_secular_parameters
                                        else linkage_parameters,
                                        fixed_nongravitational=(
                                            held_parameters
                                            if hold_secular_parameters else None
                                        ),
                                        maximum_observations=min(
                                            320, 80 * stage_count
                                        ),
                                        max_nfev=24, robust_loss=True,
                                        robust_scale=8.0, screening=False,
                                    )
                                if _linkage_trial_is_acceptable(trial):
                                    candidate = trial
                                    candidate_source = "local_astrometry"
                                else:
                                    remember_failed_trial(trial)
                                    failures.append(
                                        "lokální spojovací větev: RMS "
                                        f"{trial[1]:.2f}″"
                                    )
                        except (
                            OrbitFitError, ValueError, RuntimeError,
                            FloatingPointError,
                        ) as exc:
                            failures.append(
                                f"lokální astrometrická větev: {exc}"
                            )
                    elif candidate is None:
                        # Three historical positions can strongly confirm or
                        # reject a propagated common solution, but they do not
                        # warrant presenting an independently determined
                        # six-state orbit.  Keep them solely as constraints on
                        # the branch established by the newer apparitions.
                        failures.append(
                            "samostatná lokální dráha záměrně nepočítána "
                            f"(jen {len(apparition_groups[first_group])} měření)"
                        )
                        # With only three historical positions the useful
                        # information is accumulated orbital phase, not a new
                        # six-state orbit.  Continue from the nearest failed
                        # common branch by correcting only A1/A2 first.  This
                        # is the classical apparition-linking strategy and is
                        # vastly cheaper than a Cartesian-product grid of full
                        # century-scale six-state corrections.
                        if best_failed_trial is not None and linkage_parameters:
                            try:
                                report_progress(
                                    f"Návrat {return_year}: zpřesňuji fázi "
                                    "řídké historické aparice přes A1/A2…"
                                )
                                sparse_trial_record = best_failed_trial["trial"]
                                sparse_seed = sparse_trial_record[0]
                                sparse_parameters = {
                                    name: value for name, value in
                                    fitted_nongrav(sparse_seed).items()
                                    if name in linkage_parameters
                                }
                                sparse_prefit = fit_once(
                                    stage_observations, sparse_seed,
                                    constraints, truncate_years,
                                    sparse_parameters,
                                    maximum_observations=None,
                                    max_nfev=36, robust_loss=True,
                                    robust_scale=25.0, screening=True,
                                    fit_state=False,
                                )
                                sparse_parameters = {
                                    name: value for name, value in
                                    fitted_nongrav(sparse_prefit[0]).items()
                                    if name in linkage_parameters
                                }
                                sparse_trial = fit_once(
                                    stage_observations, sparse_prefit[0],
                                    constraints, truncate_years,
                                    sparse_parameters,
                                    maximum_observations=None,
                                    max_nfev=12, robust_loss=True,
                                    robust_scale=8.0, screening=True,
                                )
                                sparse_normalized = (
                                    sparse_trial[0].fit_normalized_rms
                                )
                                if _linkage_trial_is_acceptable(sparse_trial):
                                    candidate = sparse_trial
                                    candidate_source = "sparse_phase"
                                    linked_parameter_names.update(
                                        sparse_parameters
                                    )
                                else:
                                    remember_failed_trial(sparse_trial)
                                    failures.append(
                                        "fázová korekce řídké aparice: RMS "
                                        f"{sparse_trial[1]:.2f}″; "
                                        f"{sparse_normalized:.2f}σ"
                                    )
                            except (
                                OrbitFitError, ValueError, RuntimeError,
                                FloatingPointError,
                            ) as exc:
                                failures.append(
                                    f"fázová korekce řídké aparice: {exc}"
                                )
                        if candidate is None:
                            detail = "; ".join(failures[-3:])
                            raise OrbitFitError(
                                "Řídký historický návrat nepotvrdila přímá "
                                f"ani fázová společná korekce ({detail})."
                            )
                    # The catalogue/MPC state is not accepted as evidence by
                    # itself, but it is a valuable third initial basin for a
                    # known periodic comet.  In particular it can preserve the
                    # correct integer revolution count when both propagation
                    # from the newest return and an isolated older-return fit
                    # converge to adjacent local minima.  The observations in
                    # this application still perform and validate the fit.
                    if (
                        candidate is None and initial is not None
                        and initial is not last_good[0]
                    ):
                        catalogue_force = dict(
                            initial.nongravitational_parameters or {}
                        )
                        catalogue_force.update({
                            name: value
                            for name, value in all_parameters.items()
                            if name not in catalogue_force
                        })
                        catalogue_parameters = dict(linkage_parameters)
                        catalogue_parameters.update({
                            name: value
                            for name, value in (
                                initial.nongravitational_parameters or {}
                            ).items()
                            if name in catalogue_parameters
                        })
                        try:
                            report_progress(
                                f"Návrat {return_year}: ověřuji katalogovou "
                                "větev proti načtené astrometrii…"
                            )
                            # First preserve the independently published force
                            # law.  Letting A1/A2 move before the six-state
                            # correction can erase the very integer-revolution
                            # branch for which the catalogue state is useful.
                            catalogue_fixed_trial = fit_once(
                                stage_observations, initial, constraints,
                                truncate_years, None,
                                fixed_nongravitational=catalogue_force,
                                maximum_observations=min(
                                    320, 80 * stage_count
                                ),
                                max_nfev=10 if sparse_linkage else 28,
                                robust_loss=True, robust_scale=12.0,
                                screening=True,
                            )
                            catalogue_fixed_normalized = (
                                catalogue_fixed_trial[0].fit_normalized_rms
                            )
                            if _linkage_trial_is_acceptable(
                                catalogue_fixed_trial
                            ):
                                candidate = catalogue_fixed_trial
                                candidate_source = "catalogue_fixed_force"
                            else:
                                remember_failed_trial(catalogue_fixed_trial)
                                failures.append(
                                    "katalogová větev s pevným silovým "
                                    f"modelem: RMS {catalogue_fixed_trial[1]:.2f}″"
                                )
                            if candidate is not None:
                                catalogue_trial = catalogue_fixed_trial
                            elif hold_secular_parameters:
                                catalogue_prefit = (initial, np.inf, 0)
                                catalogue_trial = fit_once(
                                    stage_observations, catalogue_prefit[0],
                                    constraints, truncate_years, None,
                                    fixed_nongravitational=held_parameters,
                                    maximum_observations=min(
                                        320, 80 * stage_count
                                    ),
                                    max_nfev=10 if sparse_linkage else 28,
                                    robust_loss=True, robust_scale=8.0,
                                    screening=True,
                                )
                            else:
                                catalogue_prefit = fit_once(
                                    stage_observations, initial, constraints,
                                    truncate_years, catalogue_parameters,
                                    maximum_observations=min(
                                        240, 60 * stage_count
                                    ),
                                    max_nfev=6 if sparse_linkage else 12,
                                    robust_loss=True,
                                    robust_scale=25.0, screening=True,
                                    fit_state=False,
                                )
                                catalogue_trial = fit_once(
                                    stage_observations, catalogue_prefit[0],
                                    constraints, truncate_years,
                                    catalogue_parameters,
                                    maximum_observations=min(
                                        320, 80 * stage_count
                                    ),
                                    max_nfev=10 if sparse_linkage else 28,
                                    robust_loss=True, robust_scale=8.0,
                                    screening=True,
                                )
                            catalogue_normalized = (
                                catalogue_trial[0].fit_normalized_rms
                            )
                            if (
                                candidate is None
                                and _linkage_trial_is_acceptable(catalogue_trial)
                            ):
                                candidate = catalogue_trial
                                candidate_source = "catalogue_astrometry"
                            elif candidate is None:
                                remember_failed_trial(catalogue_trial)
                                normalized_text = (
                                    "nedostupné"
                                    if catalogue_normalized is None
                                    or not np.isfinite(catalogue_normalized)
                                    else f"{catalogue_normalized:.2f}σ"
                                )
                                failures.append(
                                    "katalogová astrometrická větev: RMS "
                                    f"{catalogue_trial[1]:.2f}″; "
                                    f"{normalized_text}"
                                )
                        except (
                            OrbitFitError, ValueError, RuntimeError,
                            FloatingPointError,
                        ) as exc:
                            failures.append(
                                f"katalogová astrometrická větev: {exc}"
                            )
                    # A catalogue and the already linked recent arc can each
                    # be right along a different correlated force direction.
                    # Test the small Cartesian product of their A1/A2 values
                    # with the state free but the force law fixed.  This is a
                    # deterministic, bounded analogue of statistical ranging:
                    # it crosses the shallow valley which a local simultaneous
                    # state+A correction cannot cross, then performs a normal
                    # joint correction from the best astrometric branch.
                    if (
                        candidate is None and initial is not None
                        and {"A1", "A2"} <= set(all_parameters)
                        and {"A1", "A2"} <= set(
                            initial.nongravitational_parameters or {}
                        )
                    ):
                        catalogue_force = dict(
                            initial.nongravitational_parameters or {}
                        )
                        force_trials: list[tuple[float, object]] = []
                        current_a1 = float(all_parameters["A1"])
                        current_a2 = float(all_parameters["A2"])
                        catalogue_a1 = float(catalogue_force["A1"])
                        catalogue_a2 = float(catalogue_force["A2"])
                        force_grid = (
                            (catalogue_a1, current_a2),
                            (current_a1, catalogue_a2),
                            (catalogue_a1, catalogue_a2),
                        )
                        try:
                            report_progress(
                                f"Návrat {return_year}: prohledávám "
                                "korelované větve A1/A2…"
                            )
                            for trial_a1, trial_a2 in force_grid:
                                trial_force = dict(all_parameters)
                                trial_force.update({
                                    "A1": trial_a1, "A2": trial_a2,
                                })
                                grid_trial = fit_once(
                                    stage_observations, last_good[0],
                                    constraints, truncate_years, None,
                                    fixed_nongravitational=trial_force,
                                    maximum_observations=min(
                                        240, 60 * stage_count
                                    ),
                                    max_nfev=6 if sparse_linkage else 16,
                                    robust_loss=True,
                                    robust_scale=15.0, screening=True,
                                )
                                grid_normalized = (
                                    grid_trial[0].fit_normalized_rms
                                )
                                grid_score = (
                                    float(grid_normalized)
                                    if grid_normalized is not None
                                    and np.isfinite(grid_normalized)
                                    else np.inf
                                )
                                force_trials.append((grid_score, grid_trial))
                            _, grid_seed = min(
                                force_trials, key=lambda item: item[0]
                            )
                            grid_parameters = {
                                name: value for name, value in
                                fitted_nongrav(grid_seed[0]).items()
                                if name in ("A1", "A2")
                            }
                            grid_polish = fit_once(
                                stage_observations, grid_seed[0], constraints,
                                truncate_years, grid_parameters,
                                maximum_observations=min(
                                    400, 100 * stage_count
                                ),
                                max_nfev=16 if sparse_linkage else 48,
                                robust_loss=True,
                                robust_scale=6.0, screening=False,
                            )
                            grid_normalized = (
                                grid_polish[0].fit_normalized_rms
                            )
                            if _linkage_trial_is_acceptable(grid_polish):
                                candidate = grid_polish
                                candidate_source = "force_grid"
                                linked_parameter_names.update(grid_parameters)
                            else:
                                remember_failed_trial(grid_polish)
                                failures.append(
                                    "mřížka korelovaných A1/A2: RMS "
                                    f"{grid_polish[1]:.2f}″; "
                                    f"{grid_normalized:.2f}σ"
                                )
                        except (
                            OrbitFitError, ValueError, RuntimeError,
                            FloatingPointError,
                        ) as exc:
                            failures.append(
                                f"mřížka korelovaných A1/A2: {exc}"
                            )
                    # A1/A2 are normally the most stable quantities for
                    # selecting an integer-revolution branch.  If that reduced
                    # model reaches the right neighbourhood but cannot satisfy
                    # both apparitions, release the complete model explicitly
                    # requested by the user (typically A3 and/or DT).  This is
                    # deliberately a last resort because these terms are
                    # strongly correlated on one apparition; both passes are
                    # nevertheless fitted and validated against all stage
                    # astrometry, never accepted from a catalogue value alone.
                    if candidate is None and best_failed_trial is not None:
                        appraise_best_failed_trial()
                        best_appraisals = best_failed_trial["appraisals"]
                        if best_appraisals and all(
                            rms <= LINKAGE_ACCEPT_RMS_ARCSEC
                            and normalized <= LINKAGE_ACCEPT_NORMALIZED_RMS
                            for rms, normalized in best_appraisals
                        ):
                            candidate = best_failed_trial["trial"]
                            candidate_source = "balanced_appraisal"
                            for observation, values in zip(
                                observations, best_failed_trial["report"],
                            ):
                                (
                                    observation.residual_ra_arcsec,
                                    observation.residual_dec_arcsec,
                                    observation.used_in_fit,
                                ) = values
                    if candidate is None and best_failed_trial is not None:
                        failed_trial = best_failed_trial["trial"]
                        expanded_parameters = fitted_nongrav(
                            failed_trial[0]
                        )
                        extra_parameters = (
                            set(expanded_parameters) - set(linkage_parameters)
                        )
                        if extra_parameters:
                            try:
                                report_progress(
                                    f"Návrat {return_year}: uvolňuji úplný "
                                    "negravitační model "
                                    f"({'+'.join(sorted(expanded_parameters))})…"
                                )
                                expanded_prefit = fit_once(
                                    stage_observations,
                                    failed_trial[0], constraints,
                                    truncate_years, expanded_parameters,
                                    maximum_observations=min(
                                        320, 80 * stage_count
                                    ),
                                    max_nfev=18, robust_loss=True,
                                    robust_scale=15.0, screening=False,
                                    fit_state=False,
                                )
                                expanded_trial = fit_once(
                                    stage_observations, expanded_prefit[0],
                                    constraints, truncate_years,
                                    expanded_parameters,
                                    maximum_observations=min(
                                        400, 100 * stage_count
                                    ),
                                    max_nfev=40, robust_loss=True,
                                    robust_scale=6.0, screening=False,
                                )
                                expanded_normalized = (
                                    expanded_trial[0].fit_normalized_rms
                                )
                                if _linkage_trial_is_acceptable(expanded_trial):
                                    candidate = expanded_trial
                                    candidate_source = "expanded_nongrav"
                                    linked_parameter_names.update(
                                        expanded_parameters
                                    )
                                else:
                                    remember_failed_trial(expanded_trial)
                                    normalized_text = (
                                        "nedostupné"
                                        if expanded_normalized is None
                                        or not np.isfinite(expanded_normalized)
                                        else f"{expanded_normalized:.2f}σ"
                                    )
                                    failures.append(
                                        "úplný negravitační model: RMS "
                                        f"{expanded_trial[1]:.2f}″; "
                                        f"{normalized_text}"
                                    )
                            except (
                                OrbitFitError, ValueError, RuntimeError,
                                FloatingPointError,
                            ) as exc:
                                failures.append(
                                    f"úplný negravitační model: {exc}"
                                )
                    # A robust loss is essential while choosing a branch, but
                    # once both apparitions are only a few arcseconds apart it
                    # can freeze the older one as a coherent "outlier" and
                    # never finish the phase correction.  Force one balanced,
                    # non-robust continuation from the best near solution.
                    # This is the analogue of including normal places from
                    # every apparition in the final differential correction;
                    # it is attempted only after the safer robust branches.
                    if candidate is None and best_failed_trial is not None:
                        failed_trial = best_failed_trial["trial"]
                        forcing_parameters = fitted_nongrav(
                            failed_trial[0]
                        )
                        try:
                            report_progress(
                                f"Návrat {return_year}: vynucuji vyváženou "
                                "korekci všech aparic…"
                            )
                            forced_trial = fit_once(
                                stage_observations,
                                failed_trial[0], constraints,
                                truncate_years,
                                None if hold_secular_parameters
                                else forcing_parameters,
                                fixed_nongravitational=(
                                    held_parameters
                                    if hold_secular_parameters else None
                                ),
                                maximum_observations=None,
                                max_nfev=48, robust_loss=False,
                                screening=False,
                            )
                            forced_normalized = (
                                forced_trial[0].fit_normalized_rms
                            )
                            if _linkage_trial_is_acceptable(forced_trial):
                                candidate = forced_trial
                                candidate_source = "balanced_forcing"
                                linked_parameter_names.update(
                                    forcing_parameters
                                )
                            else:
                                normalized_text = (
                                    "nedostupné"
                                    if forced_normalized is None
                                    or not np.isfinite(forced_normalized)
                                    else f"{forced_normalized:.2f}σ"
                                )
                                failures.append(
                                    "vyvážená lineární korekce: RMS "
                                    f"{forced_trial[1]:.2f}″; "
                                    f"{normalized_text}"
                                )
                        except (
                            OrbitFitError, ValueError, RuntimeError,
                            FloatingPointError,
                        ) as exc:
                            failures.append(
                                f"vyvážená lineární korekce: {exc}"
                            )
                    if candidate is None:
                        displayed_failures = failures[-2:]
                        best_appraisals = (
                            best_failed_trial["appraisals"]
                            if best_failed_trial is not None else []
                        )
                        best_diagnostic = ""
                        if best_appraisals:
                            best_diagnostic = (
                                "nejlepší mezivětev po návratech: "
                                + ", ".join(
                                    f"{apparition_linkage[first_group + offset].get('year_start', '?')} "
                                    f"{rms:.2f}″/{normalized:.2f}σ"
                                    for offset, (rms, normalized) in enumerate(
                                        best_appraisals
                                    )
                                )
                            )
                        if validation_diagnostic:
                            displayed_failures = [
                                validation_diagnostic, *displayed_failures,
                            ]
                        if (
                            identification_diagnostic
                            and identification_diagnostic
                            not in displayed_failures
                        ):
                            displayed_failures = [
                                *displayed_failures,
                                identification_diagnostic,
                            ]
                        if best_diagnostic:
                            displayed_failures = [
                                *displayed_failures, best_diagnostic,
                            ]
                        detail = (
                            "; ".join(displayed_failures)
                            or "bez použitelného pokusu"
                        )
                        raise OrbitFitError(
                            "Vícefázové připojení staršího návratu neprošlo "
                            f"kontrolou kvality ({detail})."
                        )
                except (OrbitFitError, ValueError, RuntimeError, FloatingPointError) as exc:
                    if apparition_linkage:
                        apparition_linkage[first_group]["status"] = "failed"
                        apparition_linkage[first_group]["reason"] = str(exc)
                        for older_index in range(first_group):
                            apparition_linkage[older_index]["status"] = "not_attempted"
                            apparition_linkage[older_index]["reason"] = (
                                "Nezkoušeno, protože navázání následujícího návratu selhalo."
                            )
                    # Never turn a failed linkage attempt into an apparently
                    # successful replacement orbit when even the retained
                    # newest-apparition solution is grossly inconsistent with
                    # its own astrometry.  Raising here lets OrbitFitWorker
                    # restore the pre-fit catalogue/user solution and O-C
                    # table instead of installing a hundreds-of-arcseconds
                    # fallback such as the former 206P result.
                    if not _linkage_trial_is_acceptable(last_good):
                        raise OrbitFitError(
                            "Výpočet nenašel ani spolehlivou dráhu výchozího "
                            "návratu; původní řešení zůstává beze změny "
                            f"(RMS {last_good[1]:.2f}″)."
                        ) from exc
                    result = last_good
                    for observation, report_values in zip(
                        observations, last_good_report,
                    ):
                        (
                            observation.residual_ra_arcsec,
                            observation.residual_dec_arcsec,
                            observation.used_in_fit,
                        ) = report_values
                    # Recompute O-C for the orbit which is actually returned.
                    # Previously the linkage table retained residuals from the
                    # last rejected trial (44 arcsec for 206P) beside the
                    # restored modern-only orbit whose true 1892 miss was
                    # 2336 arcsec.
                    if validation_evaluated_report:
                        for observation, evaluated, retained in zip(
                            observations, validation_evaluated_report,
                            last_good_report,
                        ):
                            observation.residual_ra_arcsec = evaluated[0]
                            observation.residual_dec_arcsec = evaluated[1]
                            observation.used_in_fit = retained[2]
                    else:
                        for observation, report_values in zip(
                            observations, last_good_report,
                        ):
                            (
                                observation.residual_ra_arcsec,
                                observation.residual_dec_arcsec,
                                observation.used_in_fit,
                            ) = report_values
                    for group_index, group in enumerate(apparition_groups):
                        _update_apparition_linkage_metrics(
                            apparition_linkage[group_index], group,
                        )
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
                        "Společný N-body fit; katalogová větev byla znovu "
                        "určena a potvrzena načtenou astrometrií."
                        if candidate_source == "catalogue_astrometry" else
                        "Společný N-body fit potvrdil katalogovou větev při "
                        "pevném publikovaném negravitačním modelu."
                        if candidate_source == "catalogue_fixed_force" else
                        "Společný N-body fit byl nalezen prohledáním "
                        "korelovaných větví A1/A2 a potvrzen astrometrií."
                        if candidate_source == "force_grid" else
                        "Společný N-body fit byl potvrzen načtenou astrometrií "
                        "po uvolnění celého negravitačního modelu."
                        if candidate_source == "expanded_nongrav" else
                        "Společný N-body fit byl potvrzen vyváženou lineární "
                        "korekcí všech připojených aparic."
                        if candidate_source == "balanced_forcing" else
                        "Společná N-body větev byla přijata podle samostatné "
                        "přesné kontroly každé aparice; souhrnné RMS ji "
                        "původně neprávem skrylo."
                        if candidate_source == "balanced_appraisal" else
                        "Společný N-body fit byl nalezen postupným rozšířením "
                        "pozorovacího oblouku do starší aparice."
                        if candidate_source == "arc_extension" else
                        "Identita byla určena porovnáním nezávislé dráhy "
                        "staršího návratu se zpětně integrovanou dráhou "
                        "novějšího návratu; teprve následná společná korekce "
                        "fitovala A1/A2 od nulových hodnot."
                        if candidate_source == "historical_identification" else
                        "Společný N-body fit řídkého historického návratu byl "
                        "nalezen korekcí nahromaděné fáze přes A1/A2 a následně "
                        "potvrzen jeho astrometrií."
                        if candidate_source == "sparse_phase" else
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
                # Only now pay for the strict DE440 correction with all six
                # state components and every requested A/DT parameter.  The
                # preceding stages merely selected the correct revolution
                # branch with relaxed integration tolerances.
                report_progress(
                    "Zpřesňuji propojenou dráhu na všech návratech…"
                )
                exact_linkage_parameters = {
                    name: value for name, value in fitted_nongrav(last_good[0]).items()
                    if name in linked_parameter_names or name == "R_EXPONENT"
                }
                balanced_final = _balanced_linkage_observations(apparition_groups)
                exact_seed = last_good[0]
                if (
                    not catalogue_seed_verified
                    and len(balanced_final) < sum(map(len, apparition_groups))
                ):
                    report_progress(
                        "Vyvažuji astrometrii jednotlivých návratů před "
                        "závěrečnou korekcí…"
                    )
                    bridge = fit_once(
                        balanced_final, exact_seed, constraints, truncate_years,
                        exact_linkage_parameters,
                        maximum_observations=None,
                        max_nfev=28, robust_loss=True, robust_scale=8.0,
                        preserve_linkage_branch=True,
                    )
                    exact_seed = bridge[0]
                apparition_weights = None
                if catalogue_seed_verified:
                    enabled_group_sizes = [
                        sum(not item.excluded_from_fit for item in group)
                        for group in apparition_groups
                    ]
                    largest_group = max(enabled_group_sizes, default=1)
                    apparition_weights = {
                        id(observation): largest_group / max(1, group_size)
                        for group, group_size in zip(
                            apparition_groups, enabled_group_sizes,
                        )
                        for observation in group
                        if not observation.excluded_from_fit
                    }
                result = fit_once(
                    observations, exact_seed, constraints, truncate_years,
                    exact_linkage_parameters,
                    maximum_observations=None,
                    max_nfev=8 if catalogue_seed_verified else 14,
                    robust_loss=True, robust_scale=8.0,
                    strict_polish=True,
                    observation_weights=apparition_weights,
                    preserve_linkage_branch=True,
                )
                if (
                    catalogue_seed_verified
                    and verified_catalogue_result is not None
                    and verified_catalogue_report is not None
                    and verified_catalogue_appraisals is not None
                ):
                    corrected_appraisals = [
                        _robust_apparition_appraisal(group)[:2]
                        for group in apparition_groups
                    ]
                    materially_worse = any(
                        corrected_rms > max(baseline_rms + 1.0, baseline_rms * 1.5)
                        or corrected_normalized > max(
                            baseline_normalized + 1.0,
                            baseline_normalized * 1.5,
                        )
                        for (
                            corrected_rms, corrected_normalized,
                        ), (
                            baseline_rms, baseline_normalized,
                        ) in zip(
                            corrected_appraisals,
                            verified_catalogue_appraisals,
                        )
                    )
                    if materially_worse:
                        catalogue_correction_rejected = True
                        result = verified_catalogue_result
                        for observation, values in zip(
                            observations, verified_catalogue_report,
                        ):
                            (
                                observation.residual_ra_arcsec,
                                observation.residual_dec_arcsec,
                                observation.used_in_fit,
                            ) = values
                failed = failed_apparitions()
                requested_parameters = fitted_nongrav(result[0])
                extra_names = set(requested_parameters) - set(exact_linkage_parameters)
                if not failed and extra_names:
                    # A3 and DT are accepted only if their larger, strongly
                    # correlated model preserves every already-linked return.
                    baseline = result
                    baseline_score = result[1]
                    baseline_report = [
                        (
                            observation.residual_ra_arcsec,
                            observation.residual_dec_arcsec,
                            observation.used_in_fit,
                        )
                        for observation in observations
                    ]
                    try:
                        expanded = fit_once(
                            observations, result[0], constraints, truncate_years,
                            requested_parameters, maximum_observations=None,
                            max_nfev=8, robust_loss=True, robust_scale=25.0,
                        )
                        if not failed_apparitions() and expanded[1] <= baseline_score * 1.05:
                            result = expanded
                        else:
                            result = baseline
                    except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
                        result = baseline
                    if result is baseline:
                        for observation, values in zip(observations, baseline_report):
                            (
                                observation.residual_ra_arcsec,
                                observation.residual_dec_arcsec,
                                observation.used_in_fit,
                            ) = values
                for group_index, group in enumerate(apparition_groups):
                    _update_apparition_linkage_metrics(
                        apparition_linkage[group_index], group,
                    )
                if failed:
                    failed_years = {str(item[0]) for item in failed}
                    for record in apparition_linkage:
                        if str(record.get("return_year")) in failed_years:
                            record["status"] = "failed"
                            record["reason"] = (
                                "Závěrečná přesná korekce nepotvrdila tento návrat."
                            )
                    detail = "; ".join(
                        f"{year}: {rms:.1f}″/{normalized:.1f}σ"
                        for year, rms, normalized in failed
                    )
                    result[0].fit_model_note = (
                        "Vyvážená apariční předkorekce našla kandidátní větev, ale "
                        "závěrečný přesný fit nepotvrdil všechny návraty "
                        f"({detail}). Společný dynamický model proto není potvrzen; "
                        "to samo o sobě nevyvrací identitu aparic."
                    )
                else:
                    for record in apparition_linkage:
                        record["status"] = "linked"
                    result[0].fit_model_note = (
                        (
                            f"Publikovaná N-body větev přímo propojila všech "
                            f"{len(apparition_groups)} pozorovaných návratů. "
                            "Následná diferenciální korekce byla odmítnuta, "
                            "protože zhoršila alespoň jednu již ověřenou "
                            "aparici; zachována je lépe sedící ověřená větev."
                        )
                        if catalogue_correction_rejected else
                        f"Společný N-body fit propojil všech "
                        f"{len(apparition_groups)} pozorovaných návratů postupnou "
                        "vyváženou apariční předkorekcí a závěrečným přesným "
                        "řešením všech měření s DE440."
                    )
        else:
            report_progress("Zpřesňuji negravitační řešení na celém oblouku…")
            result = fit_once(
                observations, result[0], constraints, truncate_years,
                fitted_nongrav(result[0]),
                maximum_observations=None,
                max_nfev=8 if long_arc else 12, robust_loss=True,
                strict_polish=long_arc,
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
    if not constraints and not long_arc and not multi_apparition:
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
                observation.raw["automatic_exclusion_reason"] = (
                    "Izolované měření s velkým normalizovaným O−C; po "
                    "automatickém vyřazení byla dráha znovu spočtena."
                )
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
    if secondary_track_exclusions:
        note = (
            f"{secondary_track_exclusions} měření v souvislé vedlejší stopě "
            "bylo odděleno od nominálního jádra (možný fragment nebo "
            "sekundární jádro)."
        )
        result[0].fit_model_note = " ".join(filter(None, (
            result[0].fit_model_note, note,
        )))
    _calibrate_fitted_photometry(observations, result[0])
    if apparition_linkage:
        result[0].apparition_linkage = tuple(apparition_linkage)
    _attach_multi_apparition_elements(observations, result[0])
    return result
