"""Relations of a heliocentric orbit to a planet's orbit.

All quantities describe the *geometry and dynamics relative to a planet* of a
body that orbits the Sun, where planetocentric osculating elements would be
meaningless:

* MOID — minimum distance between the two osculating orbits;
* Tisserand parameter T with respect to the planet;
* period ratio P / P_planet (proximity to mean-motion resonances);
* nodal distances — heliocentric distance of the body where it crosses the
  planet's orbital plane minus the planet's orbital radius in that direction.

Inputs are heliocentric ecliptic J2000 states (AU, AU/day) and TDB Julian
dates.  Planet orbits use the JPL mean Keplerian elements with their secular
rates (Standish; the same table drives the analytic planetary model), which is
ample for these orbit-to-orbit quantities.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .orbit_engine import MU_SUN
from .planetary import _ELEMENTS as _MEAN_ELEMENTS
from .planetary import J2000_JD

RELATION_PLANETS = (
    "mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune",
)
JULIAN_CENTURY_DAYS = 36525.0
_BODY_SAMPLES = 360
_PLANET_SAMPLES = 180


@dataclass(frozen=True, slots=True)
class ConicFrame:
    """Orientation and shape of a heliocentric conic.

    ``p`` is the semi-latus rectum [AU]; ``P``/``Q``/``W`` are the unit
    vectors towards pericentre, 90° ahead in the plane, and along the orbit
    normal (ecliptic J2000).
    """
    semi_latus_rectum: float
    eccentricity: float
    reciprocal_a: float
    P: np.ndarray
    Q: np.ndarray
    W: np.ndarray

    @property
    def semi_major_axis(self) -> float:
        return 1.0 / self.reciprocal_a if self.reciprocal_a != 0.0 else float("inf")

    def radius(self, true_anomaly: np.ndarray) -> np.ndarray:
        denominator = 1.0 + self.eccentricity * np.cos(true_anomaly)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                denominator > 0.0, self.semi_latus_rectum / denominator, np.nan,
            )

    def points(self, true_anomaly: np.ndarray) -> np.ndarray:
        radius = self.radius(true_anomaly)
        return (
            (radius * np.cos(true_anomaly))[..., None] * self.P
            + (radius * np.sin(true_anomaly))[..., None] * self.Q
        )

    def true_anomaly_of(self, direction: np.ndarray) -> float:
        return float(np.arctan2(direction @ self.Q, direction @ self.P))

    def true_anomaly_limit(self) -> float:
        """Largest |true anomaly| on the conic (π for an ellipse)."""
        if self.eccentricity < 1.0:
            return float(np.pi)
        return float(np.arccos(-1.0 / self.eccentricity))


def conic_from_state(position: np.ndarray, velocity: np.ndarray, mu: float = MU_SUN) -> ConicFrame:
    """Osculating conic of a heliocentric state (AU, AU/day)."""
    r = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    radius = float(np.sqrt(r @ r))
    h = np.cross(r, v)
    h_norm = float(np.sqrt(h @ h))
    if radius <= 0.0 or h_norm <= 0.0:
        raise ValueError("degenerate state")
    W = h / h_norm
    e_vector = np.cross(v, h) / mu - r / radius
    eccentricity = float(np.sqrt(e_vector @ e_vector))
    P = e_vector / eccentricity if eccentricity > 1e-10 else r / radius
    Q = np.cross(W, P)
    return ConicFrame(
        semi_latus_rectum=h_norm * h_norm / mu, eccentricity=eccentricity,
        reciprocal_a=2.0 / radius - float(v @ v) / mu, P=P, Q=Q, W=W,
    )


def planet_orbit(planet: str, epoch_jd: float) -> ConicFrame:
    """Mean heliocentric orbit of ``planet`` at ``epoch_jd`` (TDB)."""
    base, rate = _MEAN_ELEMENTS[planet.lower()]
    centuries = (float(epoch_jd) - J2000_JD) / JULIAN_CENTURY_DAYS
    a, e, inclination, _mean_longitude, perihelion_longitude, node = (
        value + change * centuries for value, change in zip(base, rate)
    )
    inclination, node, perihelion_longitude = np.radians((inclination, node, perihelion_longitude))
    argument = perihelion_longitude - node
    cos_node, sin_node = np.cos(node), np.sin(node)
    cos_arg, sin_arg = np.cos(argument), np.sin(argument)
    cos_inc, sin_inc = np.cos(inclination), np.sin(inclination)
    P = np.array((
        cos_node * cos_arg - sin_node * sin_arg * cos_inc,
        sin_node * cos_arg + cos_node * sin_arg * cos_inc,
        sin_arg * sin_inc,
    ))
    Q = np.array((
        -cos_node * sin_arg - sin_node * cos_arg * cos_inc,
        -sin_node * sin_arg + cos_node * cos_arg * cos_inc,
        cos_arg * sin_inc,
    ))
    W = np.array((sin_node * sin_inc, -cos_node * sin_inc, cos_inc))
    return ConicFrame(
        semi_latus_rectum=a * (1.0 - e * e), eccentricity=e, reciprocal_a=1.0 / a,
        P=P, Q=Q, W=W,
    )


def planet_period_years(planet: str, epoch_jd: float) -> float:
    return planet_orbit(planet, epoch_jd).semi_major_axis ** 1.5


def tisserand_parameter(body: ConicFrame, planet: ConicFrame) -> float:
    """T = a_p/a + 2 cos i' sqrt(p / a_p), i' measured from the planet's plane.

    Written with 1/a and p = a(1 - e²) it holds for hyperbolic orbits too.
    """
    planet_a = planet.semi_major_axis
    cos_inclination = float(np.clip(body.W @ planet.W, -1.0, 1.0))
    return float(
        planet_a * body.reciprocal_a
        + 2.0 * cos_inclination * np.sqrt(body.semi_latus_rectum / planet_a)
    )


def relative_inclination_deg(body: ConicFrame, planet: ConicFrame) -> float:
    return float(np.degrees(np.arccos(np.clip(body.W @ planet.W, -1.0, 1.0))))


def period_ratio(body: ConicFrame, planet: ConicFrame) -> float:
    """P / P_planet; NaN for an unbound orbit."""
    if body.reciprocal_a <= 0.0:
        return float("nan")
    return float((body.semi_major_axis / planet.semi_major_axis) ** 1.5)


def nodal_distances(body: ConicFrame, planet: ConicFrame) -> tuple[float, float]:
    """(ascending, descending) nodal distances [AU] w.r.t. the planet's plane.

    Positive: the body crosses the plane outside the planet's orbit.  NaN
    where a hyperbolic orbit has no branch in that direction or the planes
    coincide.
    """
    line = np.cross(planet.W, body.W)
    norm = float(np.sqrt(line @ line))
    if norm < 1e-12:
        return float("nan"), float("nan")
    line /= norm
    values = []
    for direction in (line, -line):
        body_radius = float(body.radius(np.array(body.true_anomaly_of(direction))))
        planet_radius = float(planet.radius(np.array(planet.true_anomaly_of(direction))))
        values.append(body_radius - planet_radius)
    return values[0], values[1]


def _planet_point(planet: ConicFrame, eccentric_anomaly: np.ndarray) -> np.ndarray:
    a = planet.semi_major_axis
    e = planet.eccentricity
    x = a * (np.cos(eccentric_anomaly) - e)
    y = a * np.sqrt(1.0 - e * e) * np.sin(eccentric_anomaly)
    return x[..., None] * planet.P + y[..., None] * planet.Q


def minimum_orbit_intersection_distance(body: ConicFrame, planet: ConicFrame) -> float:
    """MOID [AU] between a heliocentric conic and a (near-circular) planet orbit.

    A dense grid in the body's true anomaly and the planet's eccentric anomaly
    locates the candidate minima; each distinct local minimum is then refined
    with a bounded quasi-Newton search.  Two confocal orbits have at most four
    local minima, so the three best grid candidates suffice.
    """
    limit = body.true_anomaly_limit()
    if body.eccentricity < 1.0:
        body_grid = np.linspace(-np.pi, np.pi, _BODY_SAMPLES, endpoint=False)
    else:
        body_grid = np.linspace(-limit, limit, _BODY_SAMPLES + 2)[1:-1] * .999
    planet_grid = np.linspace(-np.pi, np.pi, _PLANET_SAMPLES, endpoint=False)
    body_points = body.points(body_grid)
    planet_points = _planet_point(planet, planet_grid)
    difference = body_points[:, None, :] - planet_points[None, :, :]
    distance2 = np.einsum("ijk,ijk->ij", difference, difference)
    distance2 = np.where(np.isfinite(distance2), distance2, np.inf)
    # Local minima of the (cyclic in the planet index) grid.
    neighbours = [
        np.roll(distance2, shift, axis=1) for shift in (-1, 1)
    ] + [
        np.roll(distance2, shift, axis=0) if body.eccentricity < 1.0
        else np.pad(distance2, ((1, 1), (0, 0)), constant_values=np.inf)[
            1 + shift:distance2.shape[0] + 1 + shift
        ]
        for shift in (-1, 1)
    ]
    local = np.ones_like(distance2, dtype=bool)
    for item in neighbours:
        local &= distance2 <= item
    candidates = np.argwhere(local & np.isfinite(distance2))
    if candidates.size == 0:
        candidates = np.array([np.unravel_index(np.argmin(distance2), distance2.shape)])
    order = np.argsort(distance2[candidates[:, 0], candidates[:, 1]])[:3]

    def objective(values: np.ndarray) -> tuple[float, np.ndarray]:
        nu, eccentric = float(values[0]), float(values[1])
        radius = float(body.radius(np.array(nu)))
        if not np.isfinite(radius):
            return 1e30, np.zeros(2)
        cos_nu, sin_nu = np.cos(nu), np.sin(nu)
        point = radius * (cos_nu * body.P + sin_nu * body.Q)
        # d(point)/d(nu) for r = p / (1 + e cos nu).
        dr = radius * body.eccentricity * sin_nu / (1.0 + body.eccentricity * cos_nu)
        d_point = dr * (cos_nu * body.P + sin_nu * body.Q) + radius * (
            -sin_nu * body.P + cos_nu * body.Q
        )
        a = planet.semi_major_axis
        e = planet.eccentricity
        b = a * np.sqrt(1.0 - e * e)
        planet_point = a * (np.cos(eccentric) - e) * planet.P + b * np.sin(eccentric) * planet.Q
        d_planet = -a * np.sin(eccentric) * planet.P + b * np.cos(eccentric) * planet.Q
        delta = point - planet_point
        return float(delta @ delta), np.array((2.0 * delta @ d_point, -2.0 * delta @ d_planet))

    best = float(np.sqrt(np.min(distance2)))
    span = (limit * .9999) if body.eccentricity >= 1.0 else None
    for row, column in candidates[order]:
        start = np.array((body_grid[row], planet_grid[column]))
        bounds = [(-span, span) if span is not None else (None, None), (None, None)]
        try:
            result = minimize(
                objective, start, jac=True, method="L-BFGS-B", bounds=bounds,
                options={"ftol": 1e-16, "gtol": 1e-14, "maxiter": 200},
            )
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(result.fun) and result.fun >= 0.0:
            best = min(best, float(np.sqrt(result.fun)))
    return best


@dataclass(frozen=True, slots=True)
class PlanetRelation:
    """Relation of one osculating state to a planet at one epoch."""
    epoch_jd: float
    moid_au: float
    tisserand: float
    period_ratio: float
    ascending_node_distance_au: float
    descending_node_distance_au: float
    relative_inclination_deg: float


def planet_relation(
    state: np.ndarray, planet: str, epoch_jd: float, *, include_moid: bool = True,
) -> PlanetRelation:
    """All relation quantities for a heliocentric state (AU, AU/day)."""
    state = np.asarray(state, dtype=float)
    body = conic_from_state(state[:3], state[3:])
    orbit = planet_orbit(planet, epoch_jd)
    ascending, descending = nodal_distances(body, orbit)
    return PlanetRelation(
        epoch_jd=float(epoch_jd),
        moid_au=(
            minimum_orbit_intersection_distance(body, orbit) if include_moid else float("nan")
        ),
        tisserand=tisserand_parameter(body, orbit),
        period_ratio=period_ratio(body, orbit),
        ascending_node_distance_au=ascending,
        descending_node_distance_au=descending,
        relative_inclination_deg=relative_inclination_deg(body, orbit),
    )


def nearby_resonances(
    low: float, high: float, maximum_order: int = 6,
) -> list[tuple[float, str]]:
    """Simple ratios p/q (p, q ≤ maximum_order) within [low, high] as (value, "p:q")."""
    found: dict[float, str] = {}
    for numerator in range(1, maximum_order + 1):
        for denominator in range(1, maximum_order + 1):
            if np.gcd(numerator, denominator) != 1:
                continue
            value = numerator / denominator
            if low <= value <= high:
                found.setdefault(round(value, 12), f"{numerator}:{denominator}")
    return sorted(found.items())
