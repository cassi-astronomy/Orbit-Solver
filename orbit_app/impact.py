"""Collisions of an integrated trajectory with a planet or the Moon.

The N-body integrator raises :class:`~orbit_app.orbit_engine.ImpactDetected`
as soon as a force evaluation falls inside a body.  This module locates the
contact precisely with an event integration and, for the Earth, converts it to
geographic coordinates:

* atmospheric entry — the trajectory descends through ``EARTH_ENTRY_ALTITUDE_KM``
  above the WGS84 ellipsoid;
* surface — straight continuation of the vacuum trajectory to the ellipsoid
  (atmospheric drag and fragmentation are not modelled, so this is the
  "impact point" in the Find_Orb sense, not a strewn field).

Uncertainty is estimated by Monte Carlo: states drawn from the fit covariance
are integrated to the same event and the spread of the surface points is
reported as a 1σ ellipse.  Units: AU, days, TDB Julian dates internally;
kilometres, degrees, UTC for the results.
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone

import numpy as np
from scipy.integrate import solve_ivp

from .domain import OrbitSolution
from .i18n import tr
from .orbit_engine import AU_KM, NBodyIntegrator, StateVector

EARTH_EQUATORIAL_RADIUS_KM = 6378.137
EARTH_FLATTENING = 1.0 / 298.257223563
EARTH_POLAR_RADIUS_KM = EARTH_EQUATORIAL_RADIUS_KM * (1.0 - EARTH_FLATTENING)
# Conventional altitude of atmospheric entry for meteoroids and re-entries.
EARTH_ENTRY_ALTITUDE_KM = 100.0
_OBLIQUITY = np.radians(23.4392911)
# Earth's sidereal rotation rate [rad/day] about the (equatorial) z axis.
EARTH_ROTATION_RAD_PER_DAY = 2.0 * np.pi * 86400.0 / 86164.0905
# Monte Carlo budget: clones are added until this many seconds are spent.
MONTE_CARLO_SECONDS = 30.0
MONTE_CARLO_MINIMUM = 40
MONTE_CARLO_MAXIMUM = 400


@dataclass(frozen=True, slots=True)
class ImpactPoint:
    """One event of the impact: entry at altitude, or the surface."""
    epoch_jd: float          # TDB
    utc: datetime | None
    altitude_km: float
    latitude_deg: float | None    # geodetic, Earth only
    longitude_deg: float | None   # east, Earth only
    # Earth: relative to the rotating atmosphere (as fireball data are given);
    # other bodies: relative to the body's centre.
    speed_kms: float
    elevation_angle_deg: float    # path angle below the local horizontal


@dataclass(frozen=True, slots=True)
class ImpactUncertainty:
    clones: int
    impacting: int
    surface_sigma_major_km: float
    surface_sigma_minor_km: float
    major_axis_azimuth_deg: float
    time_sigma_s: float
    latitudes_deg: tuple[float, ...] = ()
    longitudes_deg: tuple[float, ...] = ()

    @property
    def probability(self) -> float:
        return self.impacting / self.clones if self.clones else float("nan")


@dataclass(frozen=True, slots=True)
class ImpactPrediction:
    body: str
    forward: bool                 # False: the backward integration started inside a body
    entry: ImpactPoint | None     # Earth: atmospheric entry
    surface: ImpactPoint
    uncertainty: ImpactUncertainty | None = None


def _ecliptic_to_equatorial(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array((
        x,
        y * np.cos(_OBLIQUITY) - z * np.sin(_OBLIQUITY),
        y * np.sin(_OBLIQUITY) + z * np.cos(_OBLIQUITY),
    ))


def _equatorial_to_ecliptic(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.array((
        x,
        y * np.cos(_OBLIQUITY) + z * np.sin(_OBLIQUITY),
        -y * np.sin(_OBLIQUITY) + z * np.cos(_OBLIQUITY),
    ))


def body_radius_km(body: str, geocentric_equatorial: np.ndarray, radius_km: float) -> float:
    """Local radius of ``body`` towards a body-centred equatorial direction.

    The Earth uses the WGS84 ellipsoid (the geocentric latitude depends only
    on the equatorial z component, so no Earth-rotation model is needed while
    searching for the event); other bodies are spheres of ``radius_km``.
    """
    if body != "earth":
        return float(radius_km)
    norm = float(np.linalg.norm(geocentric_equatorial))
    if norm <= 0.0:
        return EARTH_EQUATORIAL_RADIUS_KM
    sin_lat = float(geocentric_equatorial[2]) / norm
    cos2 = max(0.0, 1.0 - sin_lat * sin_lat)
    a, b = EARTH_EQUATORIAL_RADIUS_KM, EARTH_POLAR_RADIUS_KM
    return a * b / np.sqrt(b * b * cos2 + a * a * sin_lat * sin_lat)


def tdb_to_utc(epoch_jd: float) -> datetime | None:
    from astropy.time import Time
    from erfa import ErfaWarning

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ErfaWarning)
            return Time(float(epoch_jd), format="jd", scale="tdb").utc.to_datetime(
                timezone=timezone.utc,
            )
    except (ValueError, OverflowError):
        return None


def earth_geodetic(epoch_jd: float, geocentric_ecliptic_au: np.ndarray) -> tuple[float, float, float]:
    """(latitude°, east longitude°, height km) of a geocentric J2000 ecliptic vector."""
    import astropy.units as u
    from astropy.coordinates import GCRS, ITRS, CartesianRepresentation, EarthLocation
    from astropy.time import Time
    from astropy.utils.exceptions import AstropyWarning
    from erfa import ErfaWarning

    equatorial_km = _ecliptic_to_equatorial(np.asarray(geocentric_ecliptic_au, dtype=float)) * AU_KM
    with warnings.catch_warnings():
        # Beyond the bundled IERS table astropy uses UT1-UTC = 0 and mean
        # polar motion: well below a kilometre on the ground.
        warnings.simplefilter("ignore", AstropyWarning)
        warnings.simplefilter("ignore", ErfaWarning)
        instant = Time(float(epoch_jd), format="jd", scale="tdb")
        gcrs = GCRS(CartesianRepresentation(*(equatorial_km * u.km)), obstime=instant)
        itrs = gcrs.transform_to(ITRS(obstime=instant))
        location = EarthLocation.from_geocentric(*itrs.cartesian.xyz)
        geodetic = location.to_geodetic("WGS84")
    longitude = (float(geodetic.lon.to_value(u.deg)) + 180.0) % 360.0 - 180.0
    return float(geodetic.lat.to_value(u.deg)), longitude, float(geodetic.height.to_value(u.km))


def _perturber(engine: NBodyIntegrator, body: str):
    for index, item in enumerate(engine.perturbers):
        if item.name == body:
            return index, item
    raise ValueError(f"unknown body {body}")


def _surface_events(
    engine: NBodyIntegrator, initial: StateVector, body: str, stop_jd: float,
) -> tuple[list[tuple[float, np.ndarray]], np.ndarray | None]:
    """Integrate to ``stop_jd`` and return [(jd, state)] of entry/surface events."""
    index, perturber = _perturber(engine, body)
    model = engine.without_collision_detection()
    altitudes = (EARTH_ENTRY_ALTITUDE_KM, 0.0) if body == "earth" else (0.0,)

    def relative(offset: float, state: np.ndarray) -> np.ndarray:
        return state[:3] - model._planet_positions(initial.epoch_jd + offset)[index]  # noqa: SLF001

    def rhs(offset: float, state: np.ndarray) -> np.ndarray:
        return np.concatenate((
            state[3:], model.acceleration(initial.epoch_jd + offset, state[:3], state[3:]),
        ))

    def make_event(altitude: float, terminal: bool):
        def event(offset: float, state: np.ndarray) -> float:
            vector = relative(offset, state)
            radius = body_radius_km(
                body, _ecliptic_to_equatorial(vector) * AU_KM, perturber.radius_km,
            )
            return float(np.linalg.norm(vector)) - (radius + altitude) / AU_KM
        event.terminal = terminal
        event.direction = -1.0
        return event

    events = [make_event(altitude, altitude == 0.0) for altitude in altitudes]
    result = solve_ivp(
        rhs, (0.0, float(stop_jd) - initial.epoch_jd), initial.packed(), method="DOP853",
        rtol=model.rtol, atol=model.atol, events=events,
    )
    found: list[tuple[float, np.ndarray]] = []
    for times, states in zip(result.t_events, result.y_events):
        if len(times):
            found.append((initial.epoch_jd + float(times[0]), np.asarray(states[0], dtype=float)))
        else:
            found.append((float("nan"), None))
    return found, result.y[:, -1] if result.y.size else None


def _impact_point(
    engine: NBodyIntegrator, body: str, epoch_jd: float, state: np.ndarray, altitude: float,
) -> ImpactPoint:
    index, _perturber_item = _perturber(engine, body)
    planet_now = engine._planet_positions(epoch_jd)[index]  # noqa: SLF001
    step = 1e-4
    planet_velocity = (
        engine._planet_positions(epoch_jd + step)[index]  # noqa: SLF001
        - engine._planet_positions(epoch_jd - step)[index]  # noqa: SLF001
    ) / (2.0 * step)
    position = state[:3] - planet_now
    velocity = state[3:] - planet_velocity
    if body == "earth":
        # Air-relative velocity: remove the co-rotating atmosphere ω × r
        # (rotation axis = equatorial z; precession is irrelevant here).
        equatorial_position = _ecliptic_to_equatorial(position)
        rotation = EARTH_ROTATION_RAD_PER_DAY * np.array((
            -equatorial_position[1], equatorial_position[0], 0.0,
        ))
        velocity = velocity - _equatorial_to_ecliptic(rotation)
    speed = float(np.linalg.norm(velocity))
    radial = float(position @ velocity) / max(float(np.linalg.norm(position)), 1e-30)
    angle = float(np.degrees(np.arcsin(np.clip(-radial / max(speed, 1e-30), -1.0, 1.0))))
    latitude = longitude = None
    if body == "earth":
        latitude, longitude, _height = earth_geodetic(epoch_jd, position)
    return ImpactPoint(
        epoch_jd=float(epoch_jd), utc=tdb_to_utc(epoch_jd), altitude_km=float(altitude),
        latitude_deg=latitude, longitude_deg=longitude,
        speed_kms=speed * AU_KM / 86400.0, elevation_angle_deg=angle,
    )


def locate_impact(
    engine: NBodyIntegrator, initial: StateVector, body: str, detected_jd: float,
) -> ImpactPrediction | None:
    """Precise contact with ``body`` near the collision ``detected_jd``.

    Integrates from ``initial`` towards the detection (a little past it, as
    the detection may come from an intermediate Runge–Kutta stage).
    """
    forward = detected_jd >= initial.epoch_jd
    margin = .05 if forward else -.05
    events, _last = _surface_events(engine, initial, body, detected_jd + margin)
    surface_jd, surface_state = events[-1]
    if surface_state is None:
        return None
    entry = None
    if body == "earth":
        entry_jd, entry_state = events[0]
        if entry_state is not None:
            entry = _impact_point(engine, body, entry_jd, entry_state, EARTH_ENTRY_ALTITUDE_KM)
    return ImpactPrediction(
        body=body, forward=forward, entry=entry,
        surface=_impact_point(engine, body, surface_jd, surface_state, 0.0),
    )


def _ground_offsets_km(latitudes: np.ndarray, longitudes: np.ndarray, latitude0: float,
                       longitude0: float) -> np.ndarray:
    radius = EARTH_EQUATORIAL_RADIUS_KM
    east = np.radians((longitudes - longitude0 + 180.0) % 360.0 - 180.0) * radius * np.cos(
        np.radians(latitude0),
    )
    north = np.radians(latitudes - latitude0) * radius
    return np.column_stack((east, north))


def impact_uncertainty(
    solution: OrbitSolution, engine: NBodyIntegrator, nominal: ImpactPrediction,
    *, seconds: float = MONTE_CARLO_SECONDS, seed: int = 20081007,
    cancel_callback=None,
) -> ImpactUncertainty | None:
    """Monte Carlo spread of the impact from the fit covariance (state only)."""
    covariance = solution.state_covariance
    if covariance is None or solution.position_au is None or solution.velocity_au_per_day is None:
        return None
    matrix = np.asarray(covariance, dtype=float)[:6, :6]
    if matrix.shape != (6, 6) or not np.all(np.isfinite(matrix)):
        return None
    values, vectors = np.linalg.eigh(.5 * (matrix + matrix.T))
    factor = vectors * np.sqrt(np.clip(values, 0.0, None))
    mean = np.concatenate((solution.position_au, solution.velocity_au_per_day))
    rng = np.random.default_rng(seed)
    stop = nominal.surface.epoch_jd + (
        max(.1, abs(nominal.surface.epoch_jd - solution.epoch_jd) * .02)
        * (1.0 if nominal.forward else -1.0)
    )
    latitudes: list[float] = []
    longitudes: list[float] = []
    epochs: list[float] = []
    clones = 0
    started = time.monotonic()
    while clones < MONTE_CARLO_MAXIMUM:
        if clones >= MONTE_CARLO_MINIMUM and time.monotonic() - started > seconds:
            break
        if cancel_callback is not None and cancel_callback():
            break
        vector = mean + factor @ rng.standard_normal(6)
        clones += 1
        try:
            events, _last = _surface_events(
                engine, StateVector(solution.epoch_jd, vector[:3], vector[3:]), nominal.body, stop,
            )
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            continue
        surface_jd, surface_state = events[-1]
        if surface_state is None:
            continue
        epochs.append(surface_jd)
        if nominal.body == "earth":
            point = _impact_point(engine, nominal.body, surface_jd, surface_state, 0.0)
            latitudes.append(float(point.latitude_deg))
            longitudes.append(float(point.longitude_deg))
    impacting = len(epochs)
    major = minor = azimuth = float("nan")
    if nominal.body == "earth" and impacting >= 3 and nominal.surface.latitude_deg is not None:
        offsets = _ground_offsets_km(
            np.asarray(latitudes), np.asarray(longitudes),
            nominal.surface.latitude_deg, nominal.surface.longitude_deg,
        )
        spread = np.cov(offsets.T)
        eigenvalues, eigenvectors = np.linalg.eigh(spread)
        minor, major = np.sqrt(np.clip(eigenvalues, 0.0, None))
        east, north = eigenvectors[:, 1]
        azimuth = float(np.degrees(np.arctan2(east, north)) % 180.0)
    return ImpactUncertainty(
        clones=clones, impacting=impacting,
        surface_sigma_major_km=float(major), surface_sigma_minor_km=float(minor),
        major_axis_azimuth_deg=azimuth,
        time_sigma_s=float(np.std(epochs) * 86400.0) if impacting >= 2 else float("nan"),
        latitudes_deg=tuple(latitudes), longitudes_deg=tuple(longitudes),
    )


# A fitted orbit is checked for a collision this many days after its epoch.
# Longer horizons are the task of the orbit-evolution integration.
IMPACT_CHECK_DAYS = 365.25


def imminent_impact(
    solution: OrbitSolution, days: float = IMPACT_CHECK_DAYS, cancel_callback=None,
) -> ImpactPrediction | None:
    """Collision of ``solution`` with a planet or the Moon within ``days``.

    Returns the located impact with its Monte Carlo uncertainty, or None.
    """
    from .orbit_evolution import _evolution_engine
    from .orbit_engine import ImpactDetected

    if solution.position_au is None or solution.velocity_au_per_day is None:
        return None
    end = solution.epoch_jd + float(days)
    engine = _evolution_engine(solution, solution.epoch_jd, end)
    start = StateVector(
        solution.epoch_jd, np.asarray(solution.position_au, dtype=float),
        np.asarray(solution.velocity_au_per_day, dtype=float),
    )
    try:
        engine.propagate(start, np.array([end]))
        return None
    except ImpactDetected as hit:
        prediction = locate_impact(engine, start, hit.body, hit.epoch_jd)
    if prediction is None:
        return None
    try:
        uncertainty = impact_uncertainty(
            solution, engine, prediction, cancel_callback=cancel_callback,
        )
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError):
        uncertainty = None
    return replace(prediction, uncertainty=uncertainty)


def with_uncertainty(prediction: ImpactPrediction, uncertainty: ImpactUncertainty | None) -> ImpactPrediction:
    return replace(prediction, uncertainty=uncertainty)


BODY_LABELS = {"moon": tr("Měsíc")}


def describe_impact(prediction: ImpactPrediction, body_label: str) -> str:
    """Human-readable multi-line summary (translated)."""
    lines = [
        tr("Dopad na těleso {0}", body_label) if prediction.forward
        else tr("Zpětná integrace začíná uvnitř tělesa {0}", body_label),
    ]

    def point_text(point: ImpactPoint, label: str) -> str:
        when = point.utc.strftime("%Y-%m-%d %H:%M:%S UTC") if point.utc else f"JD {point.epoch_jd:.6f} TDB"
        place = ""
        if point.latitude_deg is not None:
            place = tr(
                "; šířka {0:+.3f}°, délka {1:+.3f}°",
                point.latitude_deg, point.longitude_deg,
            )
        return tr(
            "{0}: {1}{2}; rychlost {3:.2f} km/s; sklon dráhy {4:.1f}° pod obzorem",
            label, when, place, point.speed_kms, point.elevation_angle_deg,
        )

    if prediction.entry is not None:
        lines.append(point_text(prediction.entry, tr("Vstup do atmosféry ({0:.0f} km)", EARTH_ENTRY_ALTITUDE_KM)))
    lines.append(point_text(prediction.surface, tr("Povrch (bez odporu atmosféry)")))
    uncertainty = prediction.uncertainty
    if uncertainty is not None:
        lines.append(tr(
            "Monte Carlo: {0} z {1} klonů z kovariance fitu dopadá ({2:.0%}).",
            uncertainty.impacting, uncertainty.clones, uncertainty.probability,
        ))
        if np.isfinite(uncertainty.surface_sigma_major_km):
            lines.append(tr(
                "Nejistota místa dopadu 1σ: {0:.2f} × {1:.2f} km (delší osa azimut {2:.0f}°); "
                "času {3:.2f} s.",
                uncertainty.surface_sigma_major_km, uncertainty.surface_sigma_minor_km,
                uncertainty.major_axis_azimuth_deg, uncertainty.time_sigma_s,
            ))
    return "\n".join(lines)
