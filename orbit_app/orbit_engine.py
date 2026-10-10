from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp

# Units used throughout the engine: AU, day, solar mass.
GAUSSIAN_K = 0.01720209895
MU_SUN = GAUSSIAN_K * GAUSSIAN_K


def marsden_water_ice_law(radius_au: float | np.ndarray) -> float | np.ndarray:
    """Standard Marsden et al. symmetric water-ice sublimation law, g(1)=1."""
    radius = np.asarray(radius_au, dtype=float)
    scaled = np.maximum(radius, 1e-12) / 2.808
    result = 0.111262 * scaled ** -2.15 * (1.0 + scaled ** 5.093) ** -4.6142
    return float(result) if result.ndim == 0 else result


def nongravitational_acceleration(
    position: np.ndarray, velocity: np.ndarray,
    parameters: dict[str, float] | None,
) -> np.ndarray:
    """Marsden A1/A2/A3 acceleration in the instantaneous RTN frame."""
    if not parameters or not any(float(parameters.get(name, 0.0)) for name in ("A1", "A2", "A3")):
        return np.zeros(3, dtype=float)
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    radius = float(np.linalg.norm(position))
    radial = position / radius
    normal_vector = np.cross(position, velocity)
    normal_norm = float(np.linalg.norm(normal_vector))
    if normal_norm <= 1e-18:
        return np.zeros(3, dtype=float)
    normal = normal_vector / normal_norm
    transverse = np.cross(normal, radial)
    direction = (
        float(parameters.get("A1", 0.0)) * radial +
        float(parameters.get("A2", 0.0)) * transverse +
        float(parameters.get("A3", 0.0)) * normal
    )
    return float(marsden_water_ice_law(radius)) * direction


@dataclass(frozen=True, slots=True)
class StateVector:
    epoch_jd: float
    position_au: np.ndarray
    velocity_au_per_day: np.ndarray

    def packed(self) -> np.ndarray:
        return np.concatenate((self.position_au, self.velocity_au_per_day)).astype(float)


class PlanetEphemeris(Protocol):
    def position(self, body: str, jd: float) -> np.ndarray:
        """Heliocentric position in AU in a consistent inertial frame."""


@dataclass(frozen=True, slots=True)
class Perturber:
    name: str
    mu_au3_day2: float


DEFAULT_PERTURBERS = (
    Perturber("mercury", MU_SUN * 1.660120e-7),
    Perturber("venus", MU_SUN * 2.4478396e-6),
    Perturber("earth", MU_SUN * 3.0034896e-6),
    Perturber("mars", MU_SUN * 3.227151e-7),
    Perturber("jupiter", MU_SUN * 9.547919e-4),
    Perturber("saturn", MU_SUN * 2.858856e-4),
    Perturber("uranus", MU_SUN * 4.366244e-5),
    Perturber("neptune", MU_SUN * 5.151389e-5),
)


class NBodyIntegrator:
    """Heliocentric test-particle integration with planetary perturbations.

    Planet positions will normally come from a JPL DE kernel. The indirect term
    keeps the heliocentric frame dynamically consistent.
    """

    def __init__(
        self,
        ephemeris: PlanetEphemeris,
        perturbers: tuple[Perturber, ...] = DEFAULT_PERTURBERS,
        relative_tolerance: float = 2e-12,
        absolute_tolerance: float = 1e-14,
        nongravitational_parameters: dict[str, float] | None = None,
    ) -> None:
        self.ephemeris = ephemeris
        self.perturbers = perturbers
        self.rtol = relative_tolerance
        self.atol = absolute_tolerance
        self.nongravitational_parameters = dict(nongravitational_parameters or {})

    def acceleration(
        self, jd: float, position: np.ndarray, velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        radius = np.linalg.norm(position)
        if radius == 0:
            raise ValueError("Těleso nemůže být ve středu Slunce.")
        acceleration = -MU_SUN * position / radius**3
        for body in self.perturbers:
            planet = np.asarray(self.ephemeris.position(body.name, jd), dtype=float)
            relative = planet - position
            relative_radius = np.linalg.norm(relative)
            planet_radius = np.linalg.norm(planet)
            if relative_radius == 0 or planet_radius == 0:
                continue
            acceleration += body.mu_au3_day2 * (
                relative / relative_radius**3 - planet / planet_radius**3
            )
        if velocity is not None:
            acceleration += nongravitational_acceleration(
                position, velocity, self.nongravitational_parameters
            )
        return acceleration

    def propagate(self, initial: StateVector, epochs_jd: np.ndarray) -> np.ndarray:
        epochs = np.asarray(epochs_jd, dtype=float)
        if epochs.ndim != 1 or epochs.size == 0:
            raise ValueError("Je potřeba alespoň jeden výstupní čas.")
        if np.any(np.diff(epochs) < 0):
            raise ValueError("Výstupní časy musejí být seřazené vzestupně.")

        def derivative(offset_days: float, state: np.ndarray) -> np.ndarray:
            position = state[:3]
            velocity = state[3:]
            return np.concatenate((
                velocity,
                self.acceleration(initial.epoch_jd + offset_days, position, velocity),
            ))

        offsets = epochs - initial.epoch_jd
        if np.all(offsets == 0.0):
            return np.repeat(initial.packed()[None, :], len(offsets), axis=0)
        start = min(0.0, float(offsets[0]))
        stop = max(0.0, float(offsets[-1]))
        # The common forward case can be integrated in one high-order pass.
        if offsets[0] >= 0:
            result = solve_ivp(
                derivative,
                (0.0, stop),
                initial.packed(),
                method="DOP853",
                t_eval=offsets,
                rtol=self.rtol,
                atol=self.atol,
            )
            if not result.success:
                raise RuntimeError(result.message)
            return result.y.T
        if offsets[-1] <= 0:
            reverse_offsets = offsets[::-1]
            result = solve_ivp(
                derivative,
                (0.0, start),
                initial.packed(),
                method="DOP853",
                t_eval=reverse_offsets,
                rtol=self.rtol,
                atol=self.atol,
            )
            if not result.success:
                raise RuntimeError(result.message)
            return result.y.T[::-1]
        output = np.empty((len(offsets), 6), dtype=float)
        negative_indices = np.flatnonzero(offsets < 0)
        nonnegative_indices = np.flatnonzero(offsets >= 0)
        if negative_indices.size:
            evaluation = offsets[negative_indices][::-1]
            backward = solve_ivp(
                derivative, (0.0, start), initial.packed(), method="DOP853",
                t_eval=evaluation, rtol=self.rtol, atol=self.atol,
            )
            if not backward.success:
                raise RuntimeError(backward.message)
            output[negative_indices] = backward.y.T[::-1]
        if nonnegative_indices.size:
            evaluation = offsets[nonnegative_indices]
            if stop == 0.0:
                output[nonnegative_indices] = initial.packed()
            else:
                forward = solve_ivp(
                    derivative, (0.0, stop), initial.packed(), method="DOP853",
                    t_eval=evaluation, rtol=self.rtol, atol=self.atol,
                )
                if not forward.success:
                    raise RuntimeError(forward.message)
                output[nonnegative_indices] = forward.y.T
        return output

