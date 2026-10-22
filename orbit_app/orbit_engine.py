from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.integrate import solve_ivp
from .i18n import tr

# Units used throughout the engine: AU, day, solar mass.
GAUSSIAN_K = 0.01720209895
MU_SUN = GAUSSIAN_K * GAUSSIAN_K
C_AU_PER_DAY = 173.1446326846693


def solar_post_newtonian_acceleration(
    position: np.ndarray, velocity: np.ndarray,
) -> np.ndarray:
    """First post-Newtonian Schwarzschild correction from the Sun.

    This is the standard heliocentric test-particle form used for small-body
    orbit determination.  It is much smaller than Newtonian gravity, but its
    secular perihelion effect is measurable when apparitions separated by
    decades are fitted together.
    """
    r_vector = np.asarray(position, dtype=float)
    v_vector = np.asarray(velocity, dtype=float)
    radius = float(np.linalg.norm(r_vector))
    if radius <= 0.0:
        raise ValueError(tr("Těleso nemůže být ve středu Slunce."))
    speed2 = float(np.dot(v_vector, v_vector))
    radial_product = float(np.dot(r_vector, v_vector))
    factor = MU_SUN / (C_AU_PER_DAY**2 * radius**3)
    return factor * (
        (4.0 * MU_SUN / radius - speed2) * r_vector
        + 4.0 * radial_product * v_vector
    )


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cross product of two 3-vectors (numpy.cross costs ~20 µs per call)."""
    return np.array((
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ))


def marsden_water_ice_law(radius_au: float | np.ndarray) -> float | np.ndarray:
    """Standard Marsden et al. symmetric water-ice sublimation law, g(1)=1."""
    radius = np.asarray(radius_au, dtype=float)
    scaled = np.maximum(radius, 1e-12) / 2.808
    result = 0.111262 * scaled ** -2.15 * (1.0 + scaled ** 5.093) ** -4.6142
    return float(result) if result.ndim == 0 else result


def _stumpff(z: float) -> tuple[float, float]:
    """Return universal-variable Stumpff C(z), S(z)."""
    if abs(z) < 1e-7:
        return (
            0.5 - z / 24.0 + z * z / 720.0,
            1.0 / 6.0 - z / 120.0 + z * z / 5040.0,
        )
    if z > 0.0:
        root = np.sqrt(z)
        return (float((1.0 - np.cos(root)) / z),
                float((root - np.sin(root)) / root**3))
    root = float(np.sqrt(-z))
    # Beyond this point the hyperbolic functions grow so steeply that a
    # Newton trial is not numerically useful.  Clipping only ``root`` while
    # retaining the original z (the former behaviour) violates the Stumpff
    # identities and can silently manufacture a plausible but wrong radius.
    if root > 50.0:
        raise FloatingPointError(tr("Hyperbolická univerzální anomálie přetekla."))
    return (float((np.cosh(root) - 1.0) / root**2),
            float((np.sinh(root) - root) / root**3))


def delayed_heliocentric_radius(
    position: np.ndarray, velocity: np.ndarray, delay_days: float,
) -> float:
    """Two-body estimate of r(t-DT) for the asymmetric Marsden model.

    The RTN acceleration direction remains tied to the instantaneous state;
    only the water-ice activity law is evaluated at the osculating heliocentric
    distance ``DT`` days earlier. Positive DT therefore moves peak activity
    after perihelion, matching the JPL/Find_Orb convention.
    """
    r0_vector = np.asarray(position, dtype=float)
    v0_vector = np.asarray(velocity, dtype=float)
    r0 = float(np.linalg.norm(r0_vector))
    dt = -float(delay_days)
    if r0 <= 0.0 or abs(dt) < 1e-12:
        return r0
    speed2 = float(np.dot(v0_vector, v0_vector))
    radial_speed = float(np.dot(r0_vector, v0_vector)) / r0
    sqrt_mu = float(np.sqrt(MU_SUN))
    alpha = 2.0 / r0 - speed2 / MU_SUN
    anomaly = sqrt_mu * dt * (abs(alpha) if abs(alpha) > 1e-8 else 1.0 / r0)
    converged = False
    for _ in range(32):
        z = alpha * anomaly * anomaly
        c_value, s_value = _stumpff(z)
        function = (
            r0 * radial_speed / sqrt_mu * anomaly**2 * c_value
            + (1.0 - alpha * r0) * anomaly**3 * s_value
            + r0 * anomaly - sqrt_mu * dt
        )
        derivative = (
            r0 * radial_speed / sqrt_mu * anomaly * (1.0 - z * s_value)
            + (1.0 - alpha * r0) * anomaly**2 * c_value + r0
        )
        if not np.isfinite(function) or not np.isfinite(derivative) or abs(derivative) < 1e-14:
            raise FloatingPointError(tr("Výpočet zpožděné aktivity nekonverguje."))
        correction = function / derivative
        anomaly -= correction
        if abs(correction) < 1e-12:
            converged = True
            break
    if not converged:
        raise FloatingPointError(tr("Výpočet zpožděné aktivity nekonverguje."))
    z = alpha * anomaly * anomaly
    c_value, s_value = _stumpff(z)
    f_value = 1.0 - anomaly * anomaly / r0 * c_value
    g_value = dt - anomaly**3 / sqrt_mu * s_value
    shifted = f_value * r0_vector + g_value * v0_vector
    radius = float(np.linalg.norm(shifted))
    if not np.isfinite(radius) or radius <= 0.0:
        raise FloatingPointError(tr("Výpočet zpožděné heliocentrické vzdálenosti selhal."))
    return radius


def nongravitational_acceleration(
    position: np.ndarray, velocity: np.ndarray,
    parameters: dict[str, float] | None,
) -> np.ndarray:
    """A1/A2/A3 acceleration in RTN using the selected distance law.

    The default is the Marsden water-ice law. ``R_EXPONENT`` selects the
    asteroid convention g(r) = r**(-R_EXPONENT); exponent 2 is the usual
    simple model for radiation pressure (A1) or Yarkovsky drift (A2).
    """
    if not parameters or not any(float(parameters.get(name, 0.0)) for name in ("A1", "A2", "A3")):
        return np.zeros(3, dtype=float)
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    radius = float(np.linalg.norm(position))
    radial = position / radius
    normal_vector = _cross(position, velocity)
    normal_norm = float(np.linalg.norm(normal_vector))
    if normal_norm <= 1e-18:
        return np.zeros(3, dtype=float)
    normal = normal_vector / normal_norm
    transverse = _cross(normal, radial)
    direction = (
        float(parameters.get("A1", 0.0)) * radial +
        float(parameters.get("A2", 0.0)) * transverse +
        float(parameters.get("A3", 0.0)) * normal
    )
    if "R_EXPONENT" in parameters:
        exponent = float(parameters["R_EXPONENT"])
        if not np.isfinite(exponent) or exponent <= 0.0:
            raise FloatingPointError(tr("Exponent negravitačního modelu musí být kladný."))
        distance_factor = radius ** -exponent
    else:
        activity_radius = delayed_heliocentric_radius(
            position, velocity, float(parameters.get(tr("DT"), 0.0)),
        )
        distance_factor = float(marsden_water_ice_law(activity_radius))
    return distance_factor * direction


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
    Perturber("moon", MU_SUN * 3.694303e-8),
    Perturber("mars", MU_SUN * 3.227151e-7),
    Perturber("jupiter", MU_SUN * 9.547919e-4),
    Perturber("saturn", MU_SUN * 2.858856e-4),
    Perturber("uranus", MU_SUN * 4.366244e-5),
    Perturber("neptune", MU_SUN * 5.151389e-5),
    Perturber("pluto", MU_SUN * 7.350478973e-9),
)


# Integration accuracy profiles (relative/absolute tolerance, AU and days).
# A century-long comet linkage crosses many close Jupiter encounters, which
# amplify any local truncation error.  For 206P/Barnard-Boattini (2008 → 1892,
# 20 revolutions, three encounters inside 0.42 AU) DOP853 gives these errors
# in the 1892 sky positions: rtol 1e-8 ≈ 1100″, 3e-11 ≈ 6″, 1e-12 ≈ 0.15″.
# Only the PRECISE profile is therefore admissible for multi-decade arcs; the
# STANDARD profile is ample within a single apparition.
PRECISE_TOLERANCES = (1e-12, 1e-14)
STANDARD_TOLERANCES = (3e-11, 2e-13)
# Branch scanning only has to locate a basin a few arcseconds wide.
SCAN_TOLERANCES = (1e-10, 1e-12)
# Arcs longer than this (days) are integrated with PRECISE_TOLERANCES.
LONG_ARC_DAYS = 730.0


def tolerances_for_span(span_days: float) -> tuple[float, float]:
    """Return the integration tolerances adequate for a time span in days."""
    return PRECISE_TOLERANCES if abs(float(span_days)) > LONG_ARC_DAYS else STANDARD_TOLERANCES


class NBodyIntegrator:
    """Heliocentric test-particle integration with planetary perturbations.

    Planet positions will normally come from a JPL DE kernel. The indirect term
    keeps the heliocentric frame dynamically consistent.  Units: AU, days,
    heliocentric J2000 ecliptic axes, TDB Julian dates.
    """

    def __init__(
        self,
        ephemeris: PlanetEphemeris,
        perturbers: tuple[Perturber, ...] = DEFAULT_PERTURBERS,
        relative_tolerance: float = 2e-12,
        absolute_tolerance: float = 1e-14,
        nongravitational_parameters: dict[str, float] | None = None,
        include_solar_relativity: bool = True,
    ) -> None:
        self.ephemeris = ephemeris
        self.perturbers = perturbers
        self.rtol = relative_tolerance
        self.atol = absolute_tolerance
        self.nongravitational_parameters = dict(nongravitational_parameters or {})
        self.include_solar_relativity = bool(include_solar_relativity)
        self._names = tuple(body.name for body in perturbers)
        self._mu = np.asarray([body.mu_au3_day2 for body in perturbers], dtype=float)
        self._has_nongravitational = any(
            float(self.nongravitational_parameters.get(name, 0.0))
            for name in ("A1", "A2", "A3")
        )

    def with_parameters(
        self, nongravitational_parameters: dict[str, float] | None,
    ) -> "NBodyIntegrator":
        """Return an otherwise identical integrator with another force law."""
        return NBodyIntegrator(
            self.ephemeris, perturbers=self.perturbers,
            relative_tolerance=self.rtol, absolute_tolerance=self.atol,
            nongravitational_parameters=nongravitational_parameters,
            include_solar_relativity=self.include_solar_relativity,
        )

    def _planet_positions(self, jd: float) -> np.ndarray:
        """Heliocentric perturber positions as an ``(N, 3)`` array in AU."""
        if not self._names:
            return np.empty((0, 3), dtype=float)
        array_method = getattr(self.ephemeris, "positions_array", None)
        if callable(array_method):
            return array_method(self._names, jd)
        batch_method = getattr(self.ephemeris, "positions", None)
        if callable(batch_method):
            values = batch_method(self._names, jd)
        else:
            values = tuple(self.ephemeris.position(name, jd) for name in self._names)
        return np.asarray(values, dtype=float).reshape(len(self._names), 3)

    def acceleration(
        self, jd: float, position: np.ndarray, velocity: np.ndarray | None = None,
    ) -> np.ndarray:
        position = np.asarray(position, dtype=float)
        radius2 = float(position @ position)
        if radius2 == 0.0:
            raise ValueError(tr("Těleso nemůže být ve středu Slunce."))
        radius = np.sqrt(radius2)
        acceleration = -MU_SUN / (radius2 * radius) * position
        if velocity is not None and self.include_solar_relativity:
            acceleration = acceleration + solar_post_newtonian_acceleration(position, velocity)
        if self._names:
            planets = self._planet_positions(jd)
            relative = planets - position
            distance = np.sqrt(np.einsum("ij,ij->i", relative, relative))
            planet_radius = np.sqrt(np.einsum("ij,ij->i", planets, planets))
            valid = (distance > 0.0) & (planet_radius > 0.0)
            if not np.all(valid):
                relative = relative[valid]
                planets = planets[valid]
                distance = distance[valid]
                planet_radius = planet_radius[valid]
                mu = self._mu[valid]
            else:
                mu = self._mu
            acceleration = acceleration + (
                mu[:, None] * (
                    relative / distance[:, None] ** 3
                    - planets / planet_radius[:, None] ** 3
                )
            ).sum(axis=0)
        if velocity is not None and self._has_nongravitational:
            acceleration = acceleration + nongravitational_acceleration(
                position, velocity, self.nongravitational_parameters
            )
        return acceleration

    def _acceleration_partials(
        self, jd: float, position: np.ndarray, velocity: np.ndarray,
        parameter_names: tuple[str, ...], parameter_scales: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return da/dr, da/dv and da/dp for variational integration.

        Position derivatives use analytic Newtonian gravity of the Sun and the
        perturbers.  The tiny post-Newtonian and non-gravitational dependences
        on r and v are deliberately omitted from the *derivatives* (they are
        always present in the trajectory itself): a Gauss-Newton Jacobian only
        has to point downhill, while the residuals, which decide convergence,
        are exact.  Derivatives with respect to A1--A3 are analytic; DT is
        differenced.  ``parameter_scales`` maps optimizer units to the physical
        units used by the force model (AU/day² for A1--A3, days for DT).
        """
        identity = np.eye(3)
        radius = float(np.linalg.norm(position))
        position_jacobian = MU_SUN * (
            3.0 * np.outer(position, position) / radius**5
            - identity / radius**3
        )
        if self._names:
            relative = self._planet_positions(jd) - position
            distance = np.sqrt(np.einsum("ij,ij->i", relative, relative))
            valid = distance > 0.0
            relative = relative[valid]
            distance = distance[valid]
            mu = self._mu[valid]
            position_jacobian = position_jacobian + (
                3.0 * np.einsum("k,ki,kj->ij", mu / distance**5, relative, relative)
                - identity * float(np.sum(mu / distance**3))
            )
        # Non-gravitational accelerations are ~1e-8 of solar gravity; their
        # own dependence on r and v is negligible in a Gauss-Newton Jacobian.
        velocity_jacobian = np.zeros((3, 3), dtype=float)
        parameter_jacobian = np.zeros((3, len(parameter_names)), dtype=float)
        if not parameter_names:
            return position_jacobian, velocity_jacobian, parameter_jacobian
        parameters = self.nongravitational_parameters
        radial = position / radius
        normal = _cross(position, velocity)
        normal_norm = float(np.linalg.norm(normal))
        if normal_norm <= 1e-18:
            return position_jacobian, velocity_jacobian, parameter_jacobian
        normal = normal / normal_norm
        transverse = _cross(normal, radial)
        if "R_EXPONENT" in parameters:
            law = radius ** -float(parameters["R_EXPONENT"])
        else:
            law = float(marsden_water_ice_law(delayed_heliocentric_radius(
                position, velocity, float(parameters.get(tr("DT"), 0.0)),
            )))
        directions = {"A1": radial, "A2": transverse, "A3": normal}
        for column, (name, scale) in enumerate(zip(parameter_names, parameter_scales)):
            if name in directions:
                # d(a)/d(A_k) = g(r) * unit vector, exactly.
                parameter_jacobian[:, column] = law * directions[name] * float(scale)
                continue
            physical_value = float(parameters.get(name, 0.0))
            physical_step = max(.01, abs(physical_value) * 1e-5)
            plus_parameters = dict(parameters)
            minus_parameters = dict(parameters)
            plus_parameters[name] = physical_value + physical_step
            minus_parameters[name] = physical_value - physical_step
            parameter_jacobian[:, column] = (
                nongravitational_acceleration(position, velocity, plus_parameters)
                - nongravitational_acceleration(position, velocity, minus_parameters)
            ) / (2.0 * physical_step) * float(scale)
        return position_jacobian, velocity_jacobian, parameter_jacobian

    def propagate_with_partials(
        self, initial: StateVector, epochs_jd: np.ndarray,
        parameter_names: tuple[str, ...] = (),
        parameter_scales: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Propagate states and derivatives with respect to state/force parameters.

        The returned partial array has shape ``(N, 6, 6 + P)``.  Its first
        six columns are the state-transition matrix from ``initial``; the
        remaining columns refer to ``parameter_names`` in optimizer units.
        """
        epochs = np.asarray(epochs_jd, dtype=float)
        if epochs.ndim != 1 or epochs.size == 0 or not np.all(np.isfinite(epochs)):
            raise ValueError(tr("Je potřeba alespoň jeden konečný výstupní čas."))
        scales = np.asarray(
            np.ones(len(parameter_names)) if parameter_scales is None
            else parameter_scales,
            dtype=float,
        )
        if scales.shape != (len(parameter_names),) or not np.all(np.isfinite(scales)):
            raise ValueError(tr("Měřítka variačních parametrů mají chybný rozměr."))

        order = np.argsort(epochs, kind="stable")
        sorted_epochs = epochs[order]
        unique_epochs, inverse = np.unique(sorted_epochs, return_inverse=True)
        if unique_epochs.size != epochs.size or not np.array_equal(
            order, np.arange(epochs.size)
        ):
            unique_states, unique_partials = self.propagate_with_partials(
                initial, unique_epochs, parameter_names, scales,
            )
            states = np.empty((epochs.size, 6), dtype=float)
            partials = np.empty((epochs.size, 6, 6 + len(parameter_names)), dtype=float)
            states[order] = unique_states[inverse]
            partials[order] = unique_partials[inverse]
            return states, partials

        parameter_count = len(parameter_names)
        initial_partials = np.column_stack((
            np.eye(6), np.zeros((6, parameter_count), dtype=float),
        ))
        augmented_initial = np.concatenate((
            initial.packed(), initial_partials.ravel(),
        ))

        def derivative(offset_days: float, augmented: np.ndarray) -> np.ndarray:
            state = augmented[:6]
            position = state[:3]
            velocity = state[3:]
            partials = augmented[6:].reshape(6, 6 + parameter_count)
            acceleration = self.acceleration(
                initial.epoch_jd + offset_days, position, velocity,
            )
            da_dr, da_dv, da_dp = self._acceleration_partials(
                initial.epoch_jd + offset_days, position, velocity,
                parameter_names, scales,
            )
            # d/dt [dr/dp; dv/dp] = [dv/dp; da/dr dr/dp + da/dv dv/dp (+ da/dp)].
            partial_derivative = np.empty_like(partials)
            partial_derivative[:3] = partials[3:]
            partial_derivative[3:] = da_dr @ partials[:3] + da_dv @ partials[3:]
            partial_derivative[3:, 6:] += da_dp
            return np.concatenate((
                velocity, acceleration,
                partial_derivative.ravel(),
            ))

        offsets = unique_epochs - initial.epoch_jd
        output = np.empty((len(offsets), augmented_initial.size), dtype=float)
        if np.all(offsets == 0.0):
            output[:] = augmented_initial
        else:
            start = min(0.0, float(offsets[0]))
            stop = max(0.0, float(offsets[-1]))
            negative = np.flatnonzero(offsets < 0.0)
            nonnegative = np.flatnonzero(offsets >= 0.0)
            if negative.size:
                evaluation = offsets[negative][::-1]
                result = solve_ivp(
                    derivative, (0.0, start), augmented_initial,
                    method="DOP853", t_eval=evaluation,
                    rtol=self.rtol, atol=self.atol,
                )
                if not result.success:
                    raise RuntimeError(result.message)
                output[negative] = result.y.T[::-1]
            if nonnegative.size:
                evaluation = offsets[nonnegative]
                if stop == 0.0:
                    output[nonnegative] = augmented_initial
                else:
                    result = solve_ivp(
                        derivative, (0.0, stop), augmented_initial,
                        method="DOP853", t_eval=evaluation,
                        rtol=self.rtol, atol=self.atol,
                    )
                    if not result.success:
                        raise RuntimeError(result.message)
                    output[nonnegative] = result.y.T
        return output[:, :6], output[:, 6:].reshape(
            len(offsets), 6, 6 + parameter_count,
        )

    def propagate(self, initial: StateVector, epochs_jd: np.ndarray) -> np.ndarray:
        epochs = np.asarray(epochs_jd, dtype=float)
        if epochs.ndim != 1 or epochs.size == 0:
            raise ValueError(tr("Je potřeba alespoň jeden výstupní čas."))
        if not np.all(np.isfinite(epochs)):
            raise ValueError(tr("Výstupní časy musejí být konečná čísla."))
        # solve_ivp requires strictly monotonic t_eval.  Astrometric rows may
        # share one epoch and light-time correction can swap two otherwise
        # adjacent epochs by a tiny amount.  Integrate sorted unique epochs and
        # restore the caller's original order, including duplicates.
        order = np.argsort(epochs, kind="stable")
        sorted_epochs = epochs[order]
        unique_epochs, inverse = np.unique(sorted_epochs, return_inverse=True)
        if unique_epochs.size != epochs.size or not np.array_equal(order, np.arange(epochs.size)):
            unique_states = self.propagate(initial, unique_epochs)
            sorted_states = unique_states[inverse]
            output = np.empty((epochs.size, 6), dtype=float)
            output[order] = sorted_states
            return output

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

