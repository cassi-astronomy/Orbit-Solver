from __future__ import annotations

from collections import OrderedDict
from datetime import datetime, timezone
from threading import Lock

import numpy as np

from .domain import OrbitSolution
from .i18n import tr
from .observatories import dynamical_jd
from .orbit_engine import ImpactDetected, NBodyIntegrator, StateVector
from .planetary import PLANETS, SplinePlanetEphemeris

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


_PREVIEW_BODIES = (
    "mercury", "venus", "earth", "moon", "mars",
    "jupiter", "saturn", "uranus", "neptune", "pluto",
)
# Interpolated planet positions for the perturbing forces (the observer and
# the Earth in the sky geometry are always read from the kernel directly).
# Same caches as the fit: 3-hour nodes within two years, daily nodes up to
# 400 years; longer spans read the kernel at every step.
_PREVIEW_EPHEMERIS: list = []
_PREVIEW_EPHEMERIS_LOCK = Lock()


def _preview_ephemeris(start_jd: float, end_jd: float):
    span = float(end_jd - start_jd)
    if span > 400.0 * 365.25:
        return PLANETS
    step = .125 if span <= 730.0 else 1.0
    key = (id(PLANETS.jpl), PLANETS.model_name)
    with _PREVIEW_EPHEMERIS_LOCK:
        for item in _PREVIEW_EPHEMERIS:
            if item[0] == key and item[1] <= start_jd and item[2] >= end_jd and item[3] <= step:
                return item[4]
    padding = max(30.0, .25 * span)
    ephemeris = SplinePlanetEphemeris(
        PLANETS, _PREVIEW_BODIES, start_jd - padding, end_jd + padding, step,
    )
    with _PREVIEW_EPHEMERIS_LOCK:
        _PREVIEW_EPHEMERIS.insert(0, (key, start_jd - padding, end_jd + padding, step, ephemeris))
        del _PREVIEW_EPHEMERIS[3:]
    return ephemeris


def _propagate_until_collision(
    engine: NBodyIntegrator, start: StateVector, epochs: np.ndarray,
    contacts: dict[str, float] | None = None,
) -> np.ndarray:
    """States at ``epochs``; rows after a collision with a planet are NaN.

    A body that hits the Earth (2008 TC3) has no position afterwards.  Each
    direction from ``start`` is integrated separately (a collision ends only
    one of them); after a detection the contact time is located once with an
    event integration and the epochs before it are integrated once more.
    """
    from .impact import locate_impact

    states = np.full((len(epochs), 6), np.nan, dtype=float)
    for indices in (
        np.flatnonzero(epochs < start.epoch_jd), np.flatnonzero(epochs >= start.epoch_jd),
    ):
        if not indices.size:
            continue
        try:
            states[indices] = engine.propagate(start, epochs[indices])
            continue
        except ImpactDetected as hit:
            try:
                prediction = locate_impact(engine, start, hit.body, hit.epoch_jd)
            except (RuntimeError, ValueError, FloatingPointError):
                prediction = None
            contact = prediction.surface.epoch_jd if prediction is not None else hit.epoch_jd
        forward = epochs[indices[0]] >= start.epoch_jd
        if contacts is not None:
            contacts["forward" if forward else "backward"] = float(contact)
        usable = indices[
            (epochs[indices] < contact) if forward else (epochs[indices] > contact)
        ]
        if usable.size:
            try:
                states[usable] = engine.propagate(start, epochs[usable])
            except ImpactDetected:
                pass
    return states


# Known collision epochs per solution: the animation asks for one instant per
# frame, and re-detecting the same impact every frame cost a second each.
_COLLISIONS: dict[int, tuple[OrbitSolution, float, float]] = {}


def _known_collisions(solution: OrbitSolution) -> tuple[float, float]:
    """(forward, backward) TDB epochs beyond which the body does not exist."""
    with _NBODY_CACHE_LOCK:
        entry = _COLLISIONS.get(id(solution))
        if entry is not None and entry[0] is solution:
            return entry[1], entry[2]
    forward, backward = np.inf, -np.inf
    prediction = getattr(solution, "impact", None)
    if prediction is not None:
        forward = float(prediction.surface.epoch_jd)
    return forward, backward


def _remember_collisions(solution: OrbitSolution, contacts: dict[str, float]) -> None:
    forward, backward = _known_collisions(solution)
    forward = min(forward, contacts.get("forward", np.inf))
    backward = max(backward, contacts.get("backward", -np.inf))
    with _NBODY_CACHE_LOCK:
        _COLLISIONS[id(solution)] = (solution, forward, backward)
        while len(_COLLISIONS) > _NBODY_CACHE_LIMIT:
            _COLLISIONS.pop(next(iter(_COLLISIONS)))


def _cached_nbody_states(solution: OrbitSolution, epochs: np.ndarray) -> np.ndarray:
    forward_contact, backward_contact = _known_collisions(solution)
    existing = (epochs < forward_contact) & (epochs > backward_contact)
    if not np.all(existing):
        # No position after a known impact: answer without integrating.
        states = np.full((len(epochs), 6), np.nan, dtype=float)
        if np.any(existing):
            states[existing] = _cached_nbody_states_alive(solution, epochs[existing])
        return states
    return _cached_nbody_states_alive(solution, epochs)


def _cached_nbody_states_alive(solution: OrbitSolution, epochs: np.ndarray) -> np.ndarray:
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
    base_epoch = float(cached_epochs[base_index])
    engine = NBodyIntegrator(
        _preview_ephemeris(
            min(base_epoch, float(epochs[0])) - 1.0, max(base_epoch, float(epochs[-1])) + 1.0,
        ),
        nongravitational_parameters=solution.nongravitational_parameters,
    )
    contacts: dict[str, float] = {}
    states = _propagate_until_collision(
        engine,
        StateVector(
            float(cached_epochs[base_index]),
            cached_states[base_index, :3], cached_states[base_index, 3:],
        ),
        epochs, contacts,
    )
    if contacts:
        _remember_collisions(solution, contacts)
    finite = np.all(np.isfinite(states), axis=1)
    merged_epochs = np.concatenate((cached_epochs, epochs[finite]))
    merged_states = np.vstack((cached_states, states[finite]))
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
        # Non-finite epochs (a light-time iteration after an impact) have
        # no position either.
        result = np.full((epochs.size, 3), np.nan, dtype=float)
        finite = np.flatnonzero(np.isfinite(epochs))
        if not finite.size:
            return result
        order = finite[np.argsort(epochs[finite])]
        result[order] = _cached_nbody_states(solution, epochs[order])[:, :3]
        return result
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

