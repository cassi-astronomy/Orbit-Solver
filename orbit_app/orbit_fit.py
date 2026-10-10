from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np
from scipy.optimize import least_squares

from .domain import Observation, OrbitSolution
from .mpc_orbit import cartesian_to_elements
from .observatories import (
    dynamical_jd, earth_heliocentric_ecliptic, geodetic_offsets_gcrs,
    station_offsets_gcrs,
)
from .orbit_engine import MU_SUN, NBodyIntegrator, StateVector
from .planetary import PLANETS

C_AU_PER_DAY = 173.1446326846693


class OrbitFitError(RuntimeError):
    pass


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
        a, e, _, _, _, mean_anomaly = elements
        q_gradient = np.array([1.0 - e, -a, 0.0, 0.0, 0.0, 0.0])
        q_sigma = float(np.sqrt(max(0.0, q_gradient @ element_covariance @ q_gradient)))
        mean_motion_rad = 0.01720209895 / abs(a) ** 1.5
        mean_motion_deg = np.degrees(mean_motion_rad)
        mean_motion_sigma = abs(1.5 * mean_motion_deg / a) * sigma[0]
        peri_gradient = np.zeros(6)
        anomaly_for_perihelion = (
            (mean_anomaly + 180.0) % 360.0 - 180.0
            if e < 1.0 else mean_anomaly
        )
        mean_radians = np.radians(anomaly_for_perihelion)
        peri_gradient[0] = -1.5 * mean_radians / (a * mean_motion_rad)
        peri_gradient[5] = -np.pi / (180.0 * mean_motion_rad)
        peri_sigma = float(np.sqrt(max(0.0, peri_gradient @ element_covariance @ peri_gradient)))
        names = ("a", "e", "i", "node", "arg_peri", "mean_anomaly")
        uncertainties = {name: float(value) for name, value in zip(names, sigma)}
        uncertainties.update({
            "q": q_sigma,
            "mean_motion_deg_day": float(mean_motion_sigma),
            "perihelion_jd": peri_sigma,
        })
        for index, name in enumerate(nongrav_names, start=6):
            # A parameters are scaled by 1e-8 inside the optimiser.
            uncertainties[name] = float(
                np.sqrt(max(0.0, parameter_covariance[index, index])) * 1e-8
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
) -> tuple[np.ndarray, np.ndarray]:
    earth = earth_heliocentric_ecliptic(epochs)
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
    """Return ADES astrometric 1-sigma values, with a conservative fallback."""
    lowered = {str(key).casefold(): value for key, value in observation.raw.items()}

    def value_for(key: str) -> float:
        try:
            value = float(lowered.get(key, 1.0))
            return value if np.isfinite(value) and value > 0.0 else 1.0
        except (TypeError, ValueError):
            return 1.0

    return value_for("rmsra"), value_for("rmsdec")


def _observation_sigmas(rows: list[tuple[Observation, datetime, float, float]]) -> np.ndarray:
    return np.asarray([
        value
        for item in rows
        for value in _observation_sigma_arcsec(item[0])
    ], dtype=float)


def initial_orbit_from_observations(
    observations: list[Observation], method: str = "systematic",
    perihelion_distance_au: float | None = None,
    minimum_observations: int = 6,
) -> OrbitSolution:
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
            earth = earth_heliocentric_ecliptic(sample_epochs)
            light_time = np.linalg.norm(propagated[:, :3] - earth, axis=1) / C_AU_PER_DAY
            emitted = engine.propagate(state, sample_epochs - light_time)
            computed_ra, computed_dec = _equatorial_direction(emitted, sample_epochs, sample_offsets)
            delta_ra = (computed_ra - sample_ra + 180.0) % 360.0 - 180.0
            delta_ra *= np.cos(np.radians(sample_dec))
            physical = np.column_stack((delta_ra, computed_dec - sample_dec)).ravel() * 3600.0
            return physical / sample_sigmas
        except (RuntimeError, ValueError, FloatingPointError):
            return np.full(sample_count * 2, 1e9)

    seeds: list[tuple[float, np.ndarray]] = []
    if method == "systematic":
        for distance in np.geomspace(0.03, 60.0, 14):
            for radial_velocity in (-0.04, 0.0, 0.04):
                parameters = np.array([np.log(distance), radial_velocity])
                residual = angular_residual(parameters)
                seeds.append((float(np.median(np.abs(residual))), parameters))
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

    best: tuple[float, np.ndarray] | None = None
    for _, seed in seeds[:4]:
        result = least_squares(
            search_residual, seed, bounds=([np.log(0.005), -0.12], [np.log(120.0), 0.12]),
            loss="soft_l1", f_scale=5.0, max_nfev=24,
        )
        score = float(np.sqrt(np.mean(np.square(angular_residual(result.x)))))
        if np.isfinite(score) and (best is None or score < best[0]):
            best = (score, result.x)
    if best is None or best[0] > 3600.0:
        raise OrbitFitError("Z astrometrie se nepodařilo sestavit použitelný počáteční odhad dráhy.")

    vector = state_for(best[1])
    try:
        elements = cartesian_to_elements(vector[:3], vector[3:])
    except Exception as exc:
        raise OrbitFitError("Počáteční astrometrická dráha nevede na použitelnou kuželosečku.") from exc
    designation = next((item[0].designation for item in parsed if item[0].designation), "Vlastní objekt")
    magnitudes = [item[0].magnitude for item in parsed if item[0].magnitude is not None]
    return OrbitSolution(
        designation=designation, epoch_jd=reference_epoch,
        semi_major_axis_au=elements[0], eccentricity=elements[1], inclination_deg=elements[2],
        node_deg=elements[3], arg_peri_deg=elements[4], mean_anomaly_deg=elements[5],
        absolute_magnitude=float(np.median(magnitudes)) if magnitudes else 18.0,
        provisional=True,
        position_au=tuple(float(value) for value in vector[:3]),
        velocity_au_per_day=tuple(float(value) for value in vector[3:]),
    )


def fit_orbit(
    observations: list[Observation], initial: OrbitSolution | None,
    constraints: dict[str, float] | None = None,
    truncate_years: float | None = None,
    nongravitational: dict[str, float] | None = None,
) -> tuple[OrbitSolution, float, int]:
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
    # Bound interactive run time while retaining the full time baseline.
    if len(usable) > 120:
        indices = np.unique(np.linspace(0, len(usable) - 1, 120).astype(int))
        usable = [usable[index] for index in indices]
    usable_times = [item[1] for item in usable]
    epochs = dynamical_jd(usable_times)
    observer_offsets = _observation_offsets_gcrs(usable)
    observed_ra = np.array([item[2] for item in usable])
    observed_dec = np.array([item[3] for item in usable])
    astrometric_sigmas = _observation_sigmas(usable)
    nongrav_names = tuple(
        name for name in ("A1", "A2", "A3") if name in (nongravitational or {})
    )
    initial_nongrav = dict(initial.nongravitational_parameters or {})
    initial_nongrav.update(nongravitational or {})

    def nongrav_from_vector(vector: np.ndarray) -> dict[str, float]:
        return {
            name: float(vector[6 + index]) * 1e-8
            for index, name in enumerate(nongrav_names)
        }

    def propagate(
        vector: np.ndarray, state: StateVector, output_epochs: np.ndarray,
    ) -> np.ndarray:
        engine = NBodyIntegrator(
            PLANETS, relative_tolerance=3e-11, absolute_tolerance=2e-13,
            nongravitational_parameters=nongrav_from_vector(vector),
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
        np.array([initial_nongrav.get(name, 0.0) / 1e-8 for name in nongrav_names]),
    ))
    initial_vector = np.concatenate((
        propagate(source_parameters, source_state, np.array([reference_epoch]))[0],
        source_parameters[6:],
    ))

    def astrometric_residuals(vector: np.ndarray) -> np.ndarray:
        state = StateVector(reference_epoch, vector[:3], vector[3:6])
        propagated = propagate(vector, state, epochs)
        earth = earth_heliocentric_ecliptic(epochs)
        light_time = np.linalg.norm(propagated[:, :3] - earth, axis=1) / C_AU_PER_DAY
        emitted = propagate(vector, state, epochs - light_time)
        computed_ra, computed_dec = _equatorial_direction(emitted, epochs, observer_offsets)
        delta_ra = (computed_ra - observed_ra + 180.0) % 360.0 - 180.0
        delta_ra *= np.cos(np.radians(observed_dec))
        delta_dec = computed_dec - observed_dec
        return np.column_stack((delta_ra, delta_dec)).ravel() * 3600.0

    def residuals(vector: np.ndarray) -> np.ndarray:
        astrometry = astrometric_residuals(vector) / astrometric_sigmas
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

    fine_scale = np.concatenate((
        np.array([1e-3, 1e-3, 1e-3, 1e-5, 1e-5, 1e-5]),
        np.ones(len(nongrav_names)),
    ))
    lower_bounds = np.concatenate((
        np.full(6, -np.inf), np.full(len(nongrav_names), -1e4),
    ))
    upper_bounds = np.concatenate((
        np.full(6, np.inf), np.full(len(nongrav_names), 1e4),
    ))

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

    result = least_squares(
        residuals,
        initial_vector,
        method="trf",
        x_scale=("jac" if derived_initial else fine_scale),
        bounds=(lower_bounds, upper_bounds),
        loss="soft_l1",
        f_scale=1.5,
        max_nfev=40 if derived_initial else 14,
        verbose=0,
    )
    if not result.success and result.nfev < 2:
        raise OrbitFitError(result.message)
    elements = cartesian_to_elements(result.x[:3], result.x[3:6])
    final_astrometric_residual = astrometric_residuals(result.x)
    rms = float(np.sqrt(np.mean(np.square(final_astrometric_residual))))
    normalized_rms = float(np.sqrt(np.mean(np.square(
        final_astrometric_residual / astrometric_sigmas
    ))))
    astrometric_jacobian = np.asarray(result.jac[:final_astrometric_residual.size, :], dtype=float)
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
        fit_rms_arcsec=rms,
        fit_normalized_rms=normalized_rms,
        display_name=initial.display_name,
        nongravitational_parameters=nongrav_from_vector(result.x) or None,
    )
    # Calculate reportable O-C values for every valid observation, not only for
    # the (at most 120) points retained by the interactive optimizer.
    report_times = [item[1] for item in reportable]
    report_epochs = dynamical_jd(report_times)
    report_offsets = _observation_offsets_gcrs(reportable)
    report_ra = np.array([item[2] for item in reportable])
    report_dec = np.array([item[3] for item in reportable])
    report_state = StateVector(reference_epoch, result.x[:3], result.x[3:6])
    report_positions = propagate(result.x, report_state, report_epochs)
    report_earth = earth_heliocentric_ecliptic(report_epochs)
    report_light_time = np.linalg.norm(report_positions[:, :3] - report_earth, axis=1) / C_AU_PER_DAY
    report_emitted = propagate(result.x, report_state, report_epochs - report_light_time)
    computed_ra, computed_dec = _equatorial_direction(report_emitted, report_epochs, report_offsets)
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
    return solution, rms, len(usable)
