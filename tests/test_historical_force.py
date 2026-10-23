"""Separate historical non-gravitational parameters and linking with model tension."""
import os
import unittest
from unittest import mock

os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

import numpy as np  # noqa: E402

from orbit_app.domain import Observation, OrbitSolution  # noqa: E402
from orbit_app.linkage import (  # noqa: E402
    LINKAGE_RELAXED_NORMALIZED_RMS, LinkageFailed, _Linker,
)
from orbit_app.orbit_engine import (  # noqa: E402
    NBodyIntegrator, active_nongravitational_coefficients, nongravitational_acceleration,
)
from orbit_app.planetary import PLANETS, SplinePlanetEphemeris  # noqa: E402

POSITION = np.array([.4, .45, .05])
VELOCITY = np.array([-.012, .016, .002])
SPLIT = 2433000.5  # 1949
PARAMETERS = {
    "A1": 1e-8, "A2": 2e-10, "A1H": 3e-8, "A2H": -5e-10, "SPLIT_JD": SPLIT,
}


class HistoricalForceEngineTests(unittest.TestCase):
    def test_coefficients_switch_at_the_split(self):
        self.assertEqual(active_nongravitational_coefficients(PARAMETERS, SPLIT - 1), (3e-8, -5e-10, 0.0))
        self.assertEqual(active_nongravitational_coefficients(PARAMETERS, SPLIT), (1e-8, 2e-10, 0.0))
        # Without an epoch, or without a split, the modern set applies.
        self.assertEqual(active_nongravitational_coefficients(PARAMETERS, None), (1e-8, 2e-10, 0.0))
        self.assertEqual(
            active_nongravitational_coefficients({"A1": 1e-8, "A1H": 3e-8}, 0.0),
            (1e-8, 0.0, 0.0),
        )

    def test_fast_law_matches_reference_on_both_sides(self):
        ephemeris = SplinePlanetEphemeris(PLANETS, ("jupiter",), SPLIT - 20, SPLIT + 20)
        engine = NBodyIntegrator(ephemeris, perturbers=(), nongravitational_parameters=PARAMETERS)
        for jd in (SPLIT - 10.0, SPLIT + 10.0):
            with self.subTest(jd=jd):
                np.testing.assert_allclose(
                    engine._nongravitational_acceleration(POSITION, VELOCITY, jd),  # noqa: SLF001
                    nongravitational_acceleration(POSITION, VELOCITY, PARAMETERS, jd),
                    rtol=1e-13, atol=0.0,
                )
        before = nongravitational_acceleration(POSITION, VELOCITY, PARAMETERS, SPLIT - 1)
        after = nongravitational_acceleration(POSITION, VELOCITY, PARAMETERS, SPLIT + 1)
        self.assertGreater(np.linalg.norm(before), 2.5 * np.linalg.norm(after))

    def test_only_the_active_set_has_partials(self):
        ephemeris = SplinePlanetEphemeris(PLANETS, ("jupiter",), SPLIT - 20, SPLIT + 20)
        engine = NBodyIntegrator(ephemeris, perturbers=(), nongravitational_parameters=PARAMETERS)
        names = ("A1", "A2", "A1H", "A2H")
        scales = np.full(4, 1e-8)
        for jd, active in ((SPLIT - 5.0, (2, 3)), (SPLIT + 5.0, (0, 1))):
            _dr, _dv, dp = engine._acceleration_partials(  # noqa: SLF001
                jd, POSITION, VELOCITY, names, scales,
            )
            for column in range(4):
                norm = np.linalg.norm(dp[:, column])
                if column in active:
                    self.assertGreater(norm, 0.0)
                else:
                    self.assertEqual(norm, 0.0)

    def test_historical_set_alone_enables_the_force(self):
        ephemeris = SplinePlanetEphemeris(PLANETS, ("jupiter",), SPLIT - 20, SPLIT + 20)
        engine = NBodyIntegrator(
            ephemeris, perturbers=(), nongravitational_parameters={"A1H": 1e-8, "SPLIT_JD": SPLIT},
        )
        gravity = NBodyIntegrator(ephemeris, perturbers=())
        difference = (
            engine.acceleration(SPLIT - 1, POSITION, VELOCITY)
            - gravity.acceleration(SPLIT - 1, POSITION, VELOCITY)
        )
        self.assertGreater(np.linalg.norm(difference), 0.0)
        np.testing.assert_allclose(
            engine.acceleration(SPLIT + 1, POSITION, VELOCITY),
            gravity.acceleration(SPLIT + 1, POSITION, VELOCITY), rtol=0.0, atol=0.0,
        )


def _groups(years_and_offsets):
    return [
        [
            Observation(
                time=f"{year}-{month:02d}-15T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
                residual_ra_arcsec=offset, residual_dec_arcsec=offset,
                raw={"rmsRA": 1.0, "rmsDec": 1.0},
            )
            for month in range(1, 7)
        ]
        for year, offset in years_and_offsets
    ]


def _linker(years_and_offsets, nongravitational=None, **options):
    groups = _groups(years_and_offsets)
    observations = [item for group in groups for item in group]
    return _Linker(
        observations, groups, None, None, nongravitational, None, None,
        lambda *a, **k: None, None, **options,
    )


class HistoricalLinkageTests(unittest.TestCase):
    def test_switch_sits_in_the_longest_gap(self):
        linker = _linker(
            [(1896, 1.0), (2008, 1.0), (2015, 1.0), (2021, 1.0)], {"A1": 0.0, "A2": 0.0},
            historical_nongravitational=True,
        )
        from orbit_app.presentation import datetime_from_jd

        self.assertEqual(datetime_from_jd(linker.split_jd).year, 1952)
        self.assertEqual(linker.free_names, ("A1", "A2", "A1H", "A2H"))
        self.assertEqual([linker.historical(index) for index in range(4)], [True, False, False, False])
        # Historical coefficients are released only with a historical return.
        self.assertEqual(linker.free_for([1, 2], linker.free_names), ("A1", "A2"))
        self.assertEqual(linker.free_for([0, 1], linker.free_names), linker.free_names)
        self.assertEqual(linker.historical_free([1, 2]), ())
        self.assertEqual(linker.historical_free([0, 3]), ("A1H", "A2H"))

    def test_historical_set_continues_the_modern_one_until_fitted(self):
        linker = _linker(
            [(1896, 1.0), (2008, 1.0)], {"A1": 0.0, "A2": 0.0},
            historical_nongravitational=True,
        )
        modern = OrbitSolution(
            "205P", 2454700.5, 3.5, .57, 15, 180, 150, 0,
            nongravitational_parameters={"A1": 7e-9, "A2": -3e-10, "SPLIT_JD": linker.split_jd},
        )
        force = linker.current_force(modern)
        self.assertEqual((force["A1H"], force["A2H"]), (7e-9, -3e-10))
        fitted = OrbitSolution(
            "205P", 2454700.5, 3.5, .57, 15, 180, 150, 0,
            nongravitational_parameters={**modern.nongravitational_parameters, "A1H": 2e-8},
        )
        self.assertEqual(linker.current_force(fitted)["A1H"], 2e-8)
        self.assertEqual(linker.current_force(fitted)["SPLIT_JD"], linker.split_jd)

    def test_option_needs_a_requested_force(self):
        gravity = _linker([(1896, 1.0), (2008, 1.0)], None, historical_nongravitational=True)
        self.assertIsNone(gravity.split_jd)
        self.assertEqual(gravity.free_names, ())


class ModelTensionTests(unittest.TestCase):
    def _link_next(self, linker, quality):
        seed = OrbitSolution(
            "205P", 2454700.5, 3.5, .57, 15, 180, 150, 0,
            position_au=(1.0, 2.0, 0.1), velocity_au_per_day=(-.01, .005, 0.0),
        )
        target = mock.Mock(epochs=np.zeros(1))
        with mock.patch.multiple(
            "orbit_app.linkage",
            _sky_target=mock.Mock(return_value=target), _scan_engine=mock.Mock(),
            _normalized_prediction=mock.Mock(return_value=np.zeros(1)),
            _robust_miss=mock.Mock(return_value=10.0),
            _scan_line_of_variations=mock.Mock(return_value=[]),
        ), mock.patch.object(
            linker, "corrected", return_value=(seed, 30.0, 10),
        ), mock.patch.object(
            linker, "group_quality", side_effect=lambda index: (quality[index], 1.0, 6),
        ):
            return linker.link_next([1], seed, 0)

    def test_return_above_the_limit_fails_without_the_option(self):
        linker = _linker([(1896, 1.0), (2008, 1.0)])
        with self.assertRaises(LinkageFailed):
            self._link_next(linker, {0: 10.3, 1: 2.8})
        self.assertEqual(linker.tension, set())

    def test_option_joins_it_and_marks_tension(self):
        linker = _linker([(1896, 1.0), (2008, 1.0)], accept_model_tension=True)
        self._link_next(linker, {0: 10.3, 1: 2.8})
        self.assertEqual(linker.tension, {0})
        # The relaxed limit then holds for this return in later quality gates.
        with mock.patch.object(linker, "group_quality", return_value=(10.3, 31.0, 70)):
            self.assertTrue(linker.accepted([0], new=[]))

    def test_option_never_spoils_the_linked_returns(self):
        linker = _linker([(1896, 1.0), (2008, 1.0)], accept_model_tension=True)
        with self.assertRaises(LinkageFailed):
            self._link_next(linker, {0: 10.3, 1: 6.5})
        with self.assertRaises(LinkageFailed):
            self._link_next(linker, {0: LINKAGE_RELAXED_NORMALIZED_RMS + 1, 1: 1.0})
        self.assertEqual(linker.tension, set())


if __name__ == "__main__":
    unittest.main()
