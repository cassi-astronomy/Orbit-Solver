from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from .domain import OrbitSolution
from .orbit_engine import MU_SUN, NBodyIntegrator, StateVector
from .planetary import PLANETS


BOUNDARY_AU = 250.0
MAX_INTEGRATION_DAYS = 365_250.0


def _planet_velocity(body: str, jd: float, interval_days: float = 0.02) -> np.ndarray:
    return (
        PLANETS.position(body, jd + interval_days)
        - PLANETS.position(body, jd - interval_days)
    ) / (2.0 * interval_days)


def _barycentric_state(
    jd: float, heliocentric_state: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    engine = NBodyIntegrator(PLANETS)
    mass_sum = 1.0
    weighted_position = np.zeros(3, dtype=float)
    weighted_velocity = np.zeros(3, dtype=float)
    for perturber in engine.perturbers:
        ratio = perturber.mu_au3_day2 / MU_SUN
        mass_sum += ratio
        weighted_position += ratio * PLANETS.position(perturber.name, jd)
        weighted_velocity += ratio * _planet_velocity(perturber.name, jd)
    sun_position = -weighted_position / mass_sum
    sun_velocity = -weighted_velocity / mass_sum
    position = np.asarray(heliocentric_state[:3], dtype=float) + sun_position
    velocity = np.asarray(heliocentric_state[3:], dtype=float) + sun_velocity
    mu_total = MU_SUN * mass_sum
    return position, velocity, mu_total


def barycentric_reciprocal_semimajor_axis(jd: float, heliocentric_state: np.ndarray) -> float:
    """Return barycentric 1/a in AU^-1 for a heliocentric Cartesian state."""
    position, velocity, mu_total = _barycentric_state(jd, heliocentric_state)
    return float(2.0 / np.linalg.norm(position) - np.dot(velocity, velocity) / mu_total)


def _formal_reciprocal_a_uncertainty(
    solution: OrbitSolution, state: np.ndarray,
) -> float | None:
    if solution.state_covariance is None:
        return None
    covariance = np.asarray(solution.state_covariance, dtype=float)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        return None
    position, velocity, mu_total = _barycentric_state(solution.epoch_jd, state)
    radius = float(np.linalg.norm(position))
    gradient = np.concatenate((
        -2.0 * position / radius**3,
        -2.0 * velocity / mu_total,
    ))
    variance = float(gradient @ ((covariance + covariance.T) / 2.0) @ gradient)
    return float(np.sqrt(max(0.0, variance)))


def _state_at_outbound_boundary(
    solution: OrbitSolution, state: np.ndarray, direction: int,
    boundary_au: float = BOUNDARY_AU,
) -> tuple[float, np.ndarray]:
    if direction not in (-1, 1):
        raise ValueError("Směr integrace musí být -1 nebo +1.")
    if np.linalg.norm(state[:3]) >= boundary_au:
        return solution.epoch_jd, np.asarray(state, dtype=float)

    engine = NBodyIntegrator(
        PLANETS, nongravitational_parameters=solution.nongravitational_parameters,
    )

    def derivative(offset_days: float, current: np.ndarray) -> np.ndarray:
        return np.concatenate((
            current[3:],
            engine.acceleration(
                solution.epoch_jd + offset_days, current[:3], current[3:],
            ),
        ))

    def reaches_boundary(_offset_days: float, current: np.ndarray) -> float:
        return float(np.linalg.norm(current[:3]) - boundary_au)

    reaches_boundary.terminal = True
    reaches_boundary.direction = 1
    result = solve_ivp(
        derivative,
        (0.0, direction * MAX_INTEGRATION_DAYS),
        np.asarray(state, dtype=float),
        method="DOP853",
        rtol=engine.rtol,
        atol=engine.atol,
        events=reaches_boundary,
        max_step=64.0,
    )
    if not result.success:
        raise RuntimeError(result.message)
    if not result.t_events[0].size:
        raise ValueError(f"Dráha nedosáhla hranice {boundary_au:g} AU.")
    offset = float(result.t_events[0][0])
    return solution.epoch_jd + offset, np.asarray(result.y_events[0][0], dtype=float)


def _boundary_pair(
    solution: OrbitSolution, state: np.ndarray, boundary_au: float,
) -> tuple[float, float]:
    past_jd, past_state = _state_at_outbound_boundary(
        solution, state, -1, boundary_au,
    )
    future_jd, future_state = _state_at_outbound_boundary(
        solution, state, 1, boundary_au,
    )
    return (
        barycentric_reciprocal_semimajor_axis(past_jd, past_state),
        barycentric_reciprocal_semimajor_axis(future_jd, future_state),
    )


def _interpret(value: float, sigma: float | None, incoming: bool) -> str:
    lower = value - 3.0 * sigma if sigma is not None else value
    upper = value + 3.0 * sigma if sigma is not None else value
    if lower <= 0.0 <= upper:
        return "nerozhodné — interval 3σ protíná parabolickou mez"
    if upper < 0.0:
        return (
            "nominálně mezihvězdná příchozí dráha"
            if incoming else "pravděpodobně opustí Sluneční soustavu"
        )
    if incoming and value < 1e-4:
        return "Oortův oblak; kandidát na dynamicky novou kometu"
    if incoming:
        return "vázaná dráha; pravděpodobně již dříve prošla oblastí planet"
    return "zůstává vázaná a měla by se vrátit"


def analyze_original_future_orbit(
    solution: OrbitSolution, boundary_au: float = BOUNDARY_AU,
) -> dict[str, float | str]:
    """Integrate a comet to the inbound/outbound 250-au barycentric orbits."""
    if solution.position_au is None or solution.velocity_au_per_day is None:
        raise ValueError("Dráha nemá kartézský stav.")
    if solution.eccentricity < 1.0 and solution.semi_major_axis_au > 0.0:
        aphelion = solution.semi_major_axis_au * (1.0 + solution.eccentricity)
        if aphelion < boundary_au:
            raise ValueError(
                f"Vázaná dráha má afel jen {aphelion:.1f} AU a hranice "
                f"{boundary_au:g} AU proto není pro tuto kometu použitelná."
            )
    state = StateVector(
        solution.epoch_jd,
        np.asarray(solution.position_au, dtype=float),
        np.asarray(solution.velocity_au_per_day, dtype=float),
    ).packed()
    original, future = _boundary_pair(solution, state, boundary_au)
    reciprocal_sigma = _formal_reciprocal_a_uncertainty(solution, state)
    result: dict[str, float | str] = {
        "boundary_au": float(boundary_au),
        "original": original,
        "future": future,
        "original_interpretation": _interpret(original, reciprocal_sigma, True),
        "future_interpretation": _interpret(future, reciprocal_sigma, False),
        "model": (
            f"N-body model planet ({PLANETS.model_name}); 1σ je lineární formální chyba ze stavové "
            "kovariance; nejde o publikovanou hodnotu MPC/JPL"
        ),
    }
    if reciprocal_sigma is not None:
        result["reciprocal_a_sigma"] = reciprocal_sigma
    return result
