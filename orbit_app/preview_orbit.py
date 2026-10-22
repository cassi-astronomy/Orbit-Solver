from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock

import numpy as np

from .domain import OrbitSolution
from .i18n import tr
from .observatories import dynamical_jd
from .orbit_engine import NBodyIntegrator, StateVector
from .planetary import PLANETS

GAUSSIAN_K = 0.01720209895  # rad/day
J2000_JD = 2451545.0
C_AU_PER_DAY = 173.1446326846693

# Numerical propagation of an old non-gravitational comet to the present can
# take seconds.  All preview panels share the same OrbitSolution instance, so
# retain already integrated states and start later requests from the nearest
# cached epoch instead of repeating the full trip from the fit epoch.
_NBODY_CACHE: OrderedDict[int, tuple[OrbitSolution, np.ndarray, np.ndarray]] = OrderedDict()
_NBODY_CACHE_LOCK = Lock()
_NBODY_CACHE_LIMIT = 8


def _cached_nbody_states(solution: OrbitSolution, epochs: np.ndarray) -> np.ndarray:
    key = id(solution)
    with _NBODY_CACHE_LOCK:
        entry = _NBODY_CACHE.get(key)
        if entry is None or entry[0] is not solution:
            cached_epochs = np.array([solution.epoch_jd], dtype=float)
            cached_states = np.array([np.concatenate((
                np.asarray(solution.position_au, dtype=float),
                np.asarray(solution.velocity_au_per_day, dtype=float),
            ))])
        else:
            cached_epochs = entry[1].copy()
            cached_states = entry[2].copy()

    centre = float((epochs[0] + epochs[-1]) / 2.0)
    base_index = int(np.argmin(np.abs(cached_epochs - centre)))
    engine = NBodyIntegrator(
        PLANETS,
        nongravitational_parameters=solution.nongravitational_parameters,
    )
    states = engine.propagate(
        StateVector(
            float(cached_epochs[base_index]),
            cached_states[base_index, :3], cached_states[base_index, 3:],
        ),
        epochs,
    )
    merged_epochs = np.concatenate((cached_epochs, epochs))
    merged_states = np.vstack((cached_states, states))
    order = np.argsort(merged_epochs)
    merged_epochs = merged_epochs[order]
    merged_states = merged_states[order]
    unique = np.concatenate(([True], np.diff(merged_epochs) != 0.0))
    merged_epochs = merged_epochs[unique]
    merged_states = merged_states[unique]
    # A long ephemeris should not grow a process-wide cache without limit.
    if len(merged_epochs) > 8000:
        keep = np.linspace(0, len(merged_epochs) - 1, 8000).astype(int)
        merged_epochs = merged_epochs[keep]
        merged_states = merged_states[keep]
    with _NBODY_CACHE_LOCK:
        _NBODY_CACHE[key] = (solution, merged_epochs, merged_states)
        _NBODY_CACHE.move_to_end(key)
        while len(_NBODY_CACHE) > _NBODY_CACHE_LIMIT:
            _NBODY_CACHE.popitem(last=False)
    return states


def datetime_to_jd(value: datetime) -> float:
    """Return the TDB Julian date used by the dynamical model.

    UI datetimes are UTC.  A raw Unix-derived Julian date is therefore not a
    suitable argument for DE440 or the numerical integrator; around the
    present epoch UTC and TDB differ by roughly 69 seconds.
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return float(dynamical_jd([value])[0])


def solve_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    mean = (np.asarray(mean_anomaly, dtype=float) + np.pi) % (2.0 * np.pi) - np.pi
    # Near a parabolic orbit Newton's method can jump between remote roots
    # around perihelion.  The elliptic Kepler function is monotonic, therefore
    # vectorised bisection is cheap and guarantees a continuous sky track.
    if eccentricity >= .95:
        lower = np.full_like(mean, -np.pi)
        upper = np.full_like(mean, np.pi)
        for _ in range(56):
            middle = (lower + upper) / 2.0
            residual = middle - eccentricity * np.sin(middle) - mean
            upper = np.where(residual > 0.0, middle, upper)
            lower = np.where(residual <= 0.0, middle, lower)
        return (lower + upper) / 2.0
    eccentric = mean.copy()
    for _ in range(24):
        step = (eccentric - eccentricity * np.sin(eccentric) - mean) / (
            1.0 - eccentricity * np.cos(eccentric)
        )
        eccentric -= step
        if step.size == 0 or np.max(np.abs(step)) < 1e-13:
            break
    return eccentric


def solve_hyperbolic_kepler(mean_anomaly: np.ndarray, eccentricity: float) -> np.ndarray:
    mean = np.asarray(mean_anomaly, dtype=float)
    anomaly = np.arcsinh(mean / max(eccentricity, 1.000001))
    for _ in range(24):
        step = (eccentricity * np.sinh(anomaly) - anomaly - mean) / (
            eccentricity * np.cosh(anomaly) - 1.0
        )
        anomaly -= step
        if step.size == 0 or np.max(np.abs(step)) < 1e-13:
            break
    return anomaly


def _rotate_orbital_plane(solution: OrbitSolution, x_orbit: np.ndarray, y_orbit: np.ndarray) -> np.ndarray:
    """Rotate perifocal coordinates into the J2000 ecliptic frame."""
    node = np.radians(solution.node_deg)
    arg = np.radians(solution.arg_peri_deg)
    inc = np.radians(solution.inclination_deg)
    cos_o, sin_o = np.cos(node), np.sin(node)
    cos_w, sin_w = np.cos(arg), np.sin(arg)
    cos_i, sin_i = np.cos(inc), np.sin(inc)
    x = (cos_o * cos_w - sin_o * sin_w * cos_i) * x_orbit + (
        -cos_o * sin_w - sin_o * cos_w * cos_i
    ) * y_orbit
    y = (sin_o * cos_w + cos_o * sin_w * cos_i) * x_orbit + (
        -sin_o * sin_w + cos_o * cos_w * cos_i
    ) * y_orbit
    z = sin_w * sin_i * x_orbit + cos_w * sin_i * y_orbit
    return np.stack((x, y, z), axis=-1)


def heliocentric_xyz(solution: OrbitSolution, jd: np.ndarray) -> np.ndarray:
    """Preview propagation in the J2000 ecliptic frame.

    Every solution carrying a Cartesian state uses the same numerical force
    model as the fit.  The osculating conic is only a fallback for catalogue
    records that contain elements but no usable state vector.
    """
    a = solution.semi_major_axis_au
    e = solution.eccentricity
    epochs = np.asarray(jd, dtype=float)
    if (
        solution.position_au is not None
        and solution.velocity_au_per_day is not None and epochs.size
    ):
        # A Keplerian preview would discard planetary perturbations and, when
        # present, A1/A2/A3/DT.  Cache the full integration because all preview
        # panels normally ask for overlapping epochs.
        order = np.argsort(epochs)
        sorted_epochs = epochs[order]
        states = _cached_nbody_states(solution, sorted_epochs)
        inverse = np.empty_like(order)
        inverse[order] = np.arange(len(order))
        return states[inverse, :3]
    if e < 1.0 and a > 0.0:
        n = GAUSSIAN_K / (a ** 1.5)
        mean = np.radians(solution.mean_anomaly_deg) + n * (epochs - solution.epoch_jd)
        eccentric = solve_kepler(np.mod(mean + np.pi, 2.0 * np.pi) - np.pi, e)
        x_orbit = a * (np.cos(eccentric) - e)
        y_orbit = a * np.sqrt(max(0.0, 1.0 - e * e)) * np.sin(eccentric)
    elif e > 1.0 and a < 0.0:
        scale = -a
        n = GAUSSIAN_K / (scale ** 1.5)
        mean = np.radians(solution.mean_anomaly_deg) + n * (epochs - solution.epoch_jd)
        hyperbolic = solve_hyperbolic_kepler(mean, e)
        x_orbit = scale * (e - np.cosh(hyperbolic))
        y_orbit = scale * np.sqrt(e * e - 1.0) * np.sinh(hyperbolic)
    else:
        raise ValueError(tr("Téměř parabolická dráha nemá stabilní parametrizaci pomocí a, e."))
    return _rotate_orbital_plane(solution, x_orbit, y_orbit)


def orbital_curve_xyz(
    solution: OrbitSolution, max_radius_au: float | None = None, samples: int = 4000
) -> tuple[np.ndarray, np.ndarray]:
    """Smooth osculating conic sampled by true anomaly, with no time-wrap chords."""
    a = solution.semi_major_axis_au
    e = solution.eccentricity
    p = a * (1.0 - e * e)
    if p <= 0.0:
        raise ValueError(tr("Neplatný parametr kuželosečky."))
    if e < 1.0:
        anomaly = np.linspace(-np.pi, np.pi, samples, endpoint=True)
    else:
        asymptote = np.arccos(-1.0 / e)
        anomaly = np.linspace(-asymptote + 1e-6, asymptote - 1e-6, samples)
    denominator = 1.0 + e * np.cos(anomaly)
    radius = p / denominator
    valid = np.isfinite(radius) & (radius > 0.0)
    if max_radius_au is not None:
        valid &= radius <= max_radius_au
    x_orbit = radius * np.cos(anomaly)
    y_orbit = radius * np.sin(anomaly)
    return _rotate_orbital_plane(solution, x_orbit, y_orbit), valid


def earth_xyz(jd: np.ndarray) -> np.ndarray:
    epochs = np.atleast_1d(np.asarray(jd, dtype=float))
    return np.stack([PLANETS.position("earth", float(epoch)) for epoch in epochs])


def apparent_geometry(
    solution: OrbitSolution, jd: np.ndarray,
    observer_offsets_equatorial: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Return light-time-corrected astrometric J2000 geometry.

    ``jd`` is TDB at reception.  Earth is evaluated at reception and the
    target at the iterated emission time.  The returned RA/Dec are deliberately
    astrometric ICRS/J2000 directions, not apparent-of-date coordinates; the
    latter additionally depend on aberration, precession/nutation and light
    deflection and are applied by Astropy when converting to Alt/Az.
    """
    epochs = np.atleast_1d(np.asarray(jd, dtype=float))
    earth = earth_xyz(epochs)
    observer = earth.copy()
    if observer_offsets_equatorial is not None:
        offsets = np.asarray(observer_offsets_equatorial, dtype=float)
        if offsets.shape != earth.shape:
            raise ValueError(tr("Pole poloh pozorovatele neodpovídá epochám efemeridy."))
        eps = np.radians(23.4392911)
        observer += np.column_stack((
            offsets[:, 0],
            offsets[:, 1] * np.cos(eps) + offsets[:, 2] * np.sin(eps),
            -offsets[:, 1] * np.sin(eps) + offsets[:, 2] * np.cos(eps),
        ))
    target = heliocentric_xyz(solution, epochs)
    light_time = np.linalg.norm(target - observer, axis=1) / C_AU_PER_DAY
    # Two fixed-point iterations leave a sub-millisecond light-time error for
    # ordinary Solar-System velocities while avoiding a costly root solver.
    for _ in range(2):
        target = heliocentric_xyz(solution, epochs - light_time)
        light_time = np.linalg.norm(target - observer, axis=1) / C_AU_PER_DAY
    geo_ecl = target - observer
    eps = np.radians(23.4392911)
    x = geo_ecl[:, 0]
    y = geo_ecl[:, 1] * np.cos(eps) - geo_ecl[:, 2] * np.sin(eps)
    z = geo_ecl[:, 1] * np.sin(eps) + geo_ecl[:, 2] * np.cos(eps)
    delta = np.linalg.norm(geo_ecl, axis=1)
    r = np.linalg.norm(target, axis=1)
    ra = np.mod(np.degrees(np.arctan2(y, x)), 360.0)
    dec = np.degrees(np.arcsin(np.clip(z / delta, -1.0, 1.0)))
    cos_phase = np.sum(target * geo_ecl, axis=1) / np.maximum(r * delta, 1e-12)
    phase = np.degrees(np.arccos(np.clip(cos_phase, -1.0, 1.0)))
    sun_geo = -observer
    cos_elongation = np.sum(geo_ecl * sun_geo, axis=1) / np.maximum(
        delta * np.linalg.norm(sun_geo, axis=1), 1e-12
    )
    elongation = np.degrees(np.arccos(np.clip(cos_elongation, -1.0, 1.0)))
    return {
        "ra": ra, "dec": dec, "delta": delta, "r": r, "phase": phase,
        "elongation": elongation, "xyz": target, "light_time_days": light_time,
    }


def demo_solution() -> OrbitSolution:
    return OrbitSolution(
        designation=tr("Ukázková dráha (neřešená)"),
        epoch_jd=2460950.5,
        semi_major_axis_au=2.2,
        eccentricity=0.46,
        inclination_deg=12.0,
        node_deg=78.0,
        arg_peri_deg=145.0,
        mean_anomaly_deg=35.0,
        absolute_magnitude=18.0,
        provisional=True,
    )

