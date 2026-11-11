from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from orbit_app.asteroid_perturbers import AssistAsteroidEphemeris, ASTEROID_NAMES
from orbit_app.astrometric_debias import catalogue_debias_arcsec, healpix_ring_index
from orbit_app.astrometry import observation_covariance_blocks, observation_sigma_arcsec
from orbit_app.domain import Observation, OrbitSolution
from orbit_app.orbit_fit import (
    _automatic_nongravitational_candidates, _mislinked_tracklet_candidates,
    _nongravitational_significance,
)


def _row(
    seconds: float, ra_deg: float, dec_deg: float = 0.0,
    *, station: str = "500", raw: dict | None = None,
):
    instant = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=seconds)
    observation = Observation(
        time=instant.isoformat(), ra_deg=ra_deg, dec_deg=dec_deg,
        station=station, raw=dict(raw or {}),
    )
    return observation, instant, ra_deg, dec_deg


class AstrometricCovarianceTests(unittest.TestCase):
    def test_ades_position_correlation_is_preserved(self):
        row = _row(0.0, 10.0, raw={"rmsRA": "2", "rmsDec": "3", "rmsCorr": "-0.5"})
        covariance = observation_covariance_blocks([row])[0]
        np.testing.assert_allclose(covariance, ((4.0, -3.0), (-3.0, 9.0)))

    def test_time_uncertainty_is_projected_along_track(self):
        arcsec_per_second = 1.0
        rows = [
            _row(0.0, 10.0, raw={"rmsRA": 1, "rmsDec": 1, "rmsTime": 2}),
            _row(10.0, 10.0 + 10.0 / 3600.0, raw={"rmsRA": 1, "rmsDec": 1, "rmsTime": 2}),
        ]
        covariance = observation_covariance_blocks(rows)
        self.assertAlmostEqual(covariance[0, 0, 0], 1.0 + (2 * arcsec_per_second) ** 2, places=6)
        self.assertAlmostEqual(covariance[0, 1, 1], 1.0, places=6)

    def test_veres_station_fallback_is_used_without_ades_rms(self):
        observation = Observation(
            time="2025-01-01T00:00:00Z", station="G96", catalog="Gaia3",
            raw={"mode": "CCD"},
        )
        self.assertEqual(observation_sigma_arcsec(observation), (0.5, 0.5))


class CatalogueDebiasTests(unittest.TestCase):
    def test_healpix_ring_indices_cover_both_poles(self):
        self.assertEqual(healpix_ring_index(0.0, 90.0), 0)
        self.assertEqual(healpix_ring_index(0.0, -90.0), 12 * 64 * 64 - 4)

    def test_catalogue_bias_includes_proper_motion(self):
        table = np.zeros((1, 26, 4), dtype=float)
        table[0, 0] = (0.5, -0.25, 10.0, -20.0)
        observation = Observation(catalog="USNOA1")
        with (
            patch("orbit_app.astrometric_debias._bias_table", return_value=table),
            patch("orbit_app.astrometric_debias.healpix_ring_index", return_value=0),
        ):
            correction = catalogue_debias_arcsec(observation, 0.0, 0.0, 2010.0)
        self.assertEqual(correction, (0.6, -0.45))


class RobustFitModelTests(unittest.TestCase):
    def test_coherent_incompatible_tracklet_is_rejected_as_a_unit(self):
        observations: list[Observation] = []
        for index in range(6):
            item = _row(index * 20.0, 10.0, station="A01")[0]
            item.residual_ra_arcsec = 0.2
            item.residual_dec_arcsec = -0.1
            observations.append(item)
        for index in range(3):
            item = _row(index * 20.0, 10.0, station="B02", raw={"trkSub": "bad"})[0]
            item.residual_ra_arcsec = 12.0 + index * 0.1
            item.residual_dec_arcsec = -3.0
            observations.append(item)
        selected = _mislinked_tracklet_candidates(observations)
        self.assertEqual({item.station for item in selected}, {"B02"})
        self.assertEqual(len(selected), 3)

    def test_automatic_models_are_conservative_by_object_and_arc(self):
        comet = OrbitSolution("1P", 2451545.0, 1, 0, 0, 0, 0, 0)
        asteroid = OrbitSolution("99942", 2451545.0, 1, 0, 0, 0, 0, 0)
        self.assertEqual(_automatic_nongravitational_candidates(comet, 30), [])
        self.assertEqual(_automatic_nongravitational_candidates(comet, 100), [{"A1": 0.0}])
        self.assertEqual(
            _automatic_nongravitational_candidates(asteroid, 800),
            [{"A2": 0.0, "R_EXPONENT": 2.0}],
        )

    def test_nongravitational_significance_uses_physical_covariance(self):
        solution = OrbitSolution(
            "1P", 2451545.0, 1, 0, 0, 0, 0, 0,
            nongravitational_parameters={"A1": 6e-9},
            parameter_names=("x", "y", "z", "vx", "vy", "vz", "A1"),
            parameter_covariance=tuple(
                tuple(4e-18 if row == column == 6 else (1.0 if row == column else 0.0)
                      for column in range(7))
                for row in range(7)
            ),
        )
        self.assertEqual(_nongravitational_significance(solution, ("A1",)), (3.0,))


class MassiveAsteroidAdapterTests(unittest.TestCase):
    def test_assist_particles_become_heliocentric_perturbers(self):
        class FakeEphem:
            jd_ref = 2451545.0

            def get_particle(self, body, _time):
                if body == 0:
                    return SimpleNamespace(x=1.0, y=2.0, z=3.0, m=1.0)
                return SimpleNamespace(x=2.0, y=2.0, z=3.0, m=body * 1e-15)

        planets = SimpleNamespace(position=lambda _body, _jd: np.zeros(3))
        provider = AssistAsteroidEphemeris(planets, FakeEphem())
        self.assertEqual(len(provider.perturbers), 16)
        self.assertEqual(tuple(item.name for item in provider.perturbers), ASTEROID_NAMES)
        np.testing.assert_allclose(provider.position("asteroid_ceres", 2451545.0), (1.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()
