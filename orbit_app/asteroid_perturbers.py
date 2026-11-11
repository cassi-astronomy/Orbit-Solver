"""Optional positions and masses of the 16 SB441 asteroid perturbers.

ASSIST is deliberately optional (and GPL-3.0 licensed).  Orbit Solver keeps
its native integrator and uses ASSIST only as an ephemeris reader when both
``de440.bsp`` and ``sb441-n16.bsp`` are present in the runtime data directory.
The adapter exposes heliocentric J2000-ecliptic positions in AU and GM in
AU³/day², matching :mod:`orbit_app.orbit_engine`.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .orbit_engine import Perturber
from .planetary import de440_path, ephemeris_data_directory

SB441_FILENAME = "sb441-n16.bsp"

_ASTEROIDS = (
    (11, "asteroid_camilla"), (12, "asteroid_ceres"),
    (13, "asteroid_cybele"), (14, "asteroid_davida"),
    (15, "asteroid_eunomia"), (16, "asteroid_euphrosyne"),
    (17, "asteroid_europa"), (18, "asteroid_hygiea"),
    (19, "asteroid_interamnia"), (20, "asteroid_iris"),
    (21, "asteroid_juno"), (22, "asteroid_pallas"),
    (23, "asteroid_psyche"), (24, "asteroid_sylvia"),
    (25, "asteroid_thisbe"), (26, "asteroid_vesta"),
)
ASTEROID_NAMES = tuple(name for _body_id, name in _ASTEROIDS)
_BODY_IDS = {name: body_id for body_id, name in _ASTEROIDS}


def sb441_path() -> Path:
    return ephemeris_data_directory() / SB441_FILENAME


def _equatorial_to_ecliptic(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    epsilon = np.radians(23.4392911)
    return np.array((
        value[0],
        value[1] * np.cos(epsilon) + value[2] * np.sin(epsilon),
        -value[1] * np.sin(epsilon) + value[2] * np.cos(epsilon),
    ))


class AssistAsteroidEphemeris:
    """Planet provider plus SB441 asteroids, for spline precomputation."""

    def __init__(self, planets: Any, assist_ephem: Any) -> None:
        self.planets = planets
        self.assist_ephem = assist_ephem
        self.jd_ref = float(assist_ephem.jd_ref)
        perturbers: list[Perturber] = []
        for body_id, name in _ASTEROIDS:
            particle = assist_ephem.get_particle(body_id, 0.0)
            mu = float(particle.m)
            if not np.isfinite(mu) or mu <= 0.0:
                raise RuntimeError(f"ASSIST returned an invalid GM for {name}.")
            perturbers.append(Perturber(name, mu))
        self.perturbers = tuple(perturbers)

    def position(self, body: str, jd: float) -> np.ndarray:
        name = body.lower()
        body_id = _BODY_IDS.get(name)
        if body_id is None:
            return np.asarray(self.planets.position(name, jd), dtype=float)
        relative_time = float(jd) - self.jd_ref
        asteroid = self.assist_ephem.get_particle(body_id, relative_time)
        sun = self.assist_ephem.get_particle(0, relative_time)
        heliocentric = np.array((
            asteroid.x - sun.x, asteroid.y - sun.y, asteroid.z - sun.z,
        ), dtype=float)
        return _equatorial_to_ecliptic(heliocentric)

    def position_series(self, body: str, epochs: np.ndarray) -> np.ndarray:
        name = body.lower()
        if name not in _BODY_IDS:
            method = getattr(self.planets, "position_series", None)
            if callable(method):
                return np.asarray(method(name, epochs), dtype=float)
        return np.stack([self.position(name, float(epoch)) for epoch in epochs])


@lru_cache(maxsize=1)
def optional_asteroid_ephemeris() -> AssistAsteroidEphemeris | None:
    """Loaded adapter, or ``None`` when package/kernel/model are unavailable."""
    planets_path = de440_path()
    asteroids_path = sb441_path()
    if not planets_path.is_file() or not asteroids_path.is_file():
        return None
    try:
        import assist
        from .planetary import PLANETS

        if PLANETS.jpl is None:
            return None
        ephem = assist.Ephem(str(planets_path), str(asteroids_path))
        return AssistAsteroidEphemeris(PLANETS, ephem)
    except Exception:
        # Optional precision data must never prevent the native planet-only
        # model from starting (bad wheel, damaged kernel, unsupported host).
        return None


def massive_asteroids_active() -> bool:
    return optional_asteroid_ephemeris() is not None
