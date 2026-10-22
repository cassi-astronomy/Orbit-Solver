"""Regression tests for astrometric metadata, time scales and apparition linkage."""
import os

# These suites assert the Czech source texts; English is covered by
# tests/test_i18n.py.  Must be set before orbit_app modules are imported.
os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

import math
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np

from orbit_app.astrometry import (
    is_fit_candidate, mark_duplicate_measurements, measurement_technique,
    normal_place_observations, observation_sigma_arcsec,
)
from orbit_app.domain import Observation, OrbitSolution
from orbit_app.linkage import (
    LINKAGE_ACCEPT_NORMALIZED_RMS, _line_of_variations, link_apparitions,
)
from orbit_app.mpc import parse_mpc80
from orbit_app.mpc_orbit import cartesian_to_elements
from orbit_app.observatories import delta_t_seconds, dynamical_jd
from orbit_app.orbit_engine import (
    MU_SUN, PRECISE_TOLERANCES, STANDARD_TOLERANCES, NBodyIntegrator, StateVector,
    tolerances_for_span,
)
from orbit_app.orbit_fit import (
    _equatorial_direction, _evaluate_solution_observations, _fit_orbit_once,
    _observation_offsets_gcrs, _observing_apparition_groups,
    fit_orbit, parse_observation_time,
)
from orbit_app.observatories import earth_heliocentric_ecliptic
from orbit_app.planetary import PLANETS, SplinePlanetEphemeris
from orbit_app.presentation import apparition_linkage_summary


BARNARD_1892 = (
    "0206PI92T010  M1892 10 14.13824 19 38 57.84 +12 44 37.1                pAJ012662\n"
    "0206PI92T010  M1892 10 14.13824 19 38 57.87 +12 44 37.1                qAJ012662\n"
    "0206PI92T010  M1892 10 16.81152 19 44 46.40 +11 45 39.2                pAN131045\n"
)


class TimeScaleTests(unittest.TestCase):
    def test_historical_delta_t_follows_earth_rotation_record(self):
        # Morrison & Stephenson / Espenak & Meeus: about -6 s in 1892 and
        # +29 s in 1950.  ERFA's pre-UTC fallback would give 32.184 s.
        self.assertAlmostEqual(delta_t_seconds(1892.8), -6.3, delta=0.6)
        self.assertAlmostEqual(delta_t_seconds(1950.0), 29.07, places=2)
        self.assertAlmostEqual(delta_t_seconds(1910.0), 10.4, delta=0.6)

    def test_pre_1960_timestamp_is_universal_time(self):
        instant = datetime(1892, 10, 14, 3, 19, tzinfo=timezone.utc)
        reference = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
        ut_jd = 2451545.0 + (instant - reference).total_seconds() / 86400.0
        tdb_minus_ut = (dynamical_jd([instant])[0] - ut_jd) * 86400.0
        self.assertAlmostEqual(tdb_minus_ut, delta_t_seconds(1892.79), delta=0.01)

    def test_modern_and_historical_instants_keep_their_order(self):
        times = [
            datetime(2008, 10, 7, tzinfo=timezone.utc),
            datetime(1892, 10, 14, tzinfo=timezone.utc),
            datetime(1959, 12, 31, 23, tzinfo=timezone.utc),
            datetime(1960, 1, 1, 1, tzinfo=timezone.utc),
        ]
        values = dynamical_jd(times)
        self.assertGreater(values[0], values[3])
        self.assertLess(values[1], values[2])
        # No discontinuity worth more than about a second at the UTC origin.
        self.assertAlmostEqual((values[3] - values[2]) * 86400.0, 7200.0, delta=1.5)


class AstrometricMetadataTests(unittest.TestCase):
    def test_mpc80_keeps_catalogue_and_technique(self):
        observations = parse_mpc80(BARNARD_1892)
        self.assertEqual([item.catalog for item in observations], ["p", "q", "p"])
        self.assertEqual(measurement_technique(observations[0]), "micrometer")
        self.assertEqual(observation_sigma_arcsec(observations[0]), (3.0, 3.0))

    def test_rereduction_against_better_catalogue_replaces_duplicate(self):
        observations = parse_mpc80(BARNARD_1892)
        self.assertEqual(mark_duplicate_measurements(observations), 1)
        self.assertTrue(observations[0].raw["duplicate_measurement"])
        self.assertFalse(is_fit_candidate(observations[0]))
        self.assertTrue(is_fit_candidate(observations[1]))   # UCAC-4 copy
        self.assertTrue(is_fit_candidate(observations[2]))   # other station

    def test_simultaneous_measurements_of_different_objects_are_not_merged(self):
        first, second = parse_mpc80(BARNARD_1892)[:2]
        second.ra_deg += 0.1  # 6 arcmin: a different position, not a re-reduction
        self.assertEqual(mark_duplicate_measurements([first, second]), 0)

    def test_deprecated_mpc_record_is_not_fitted(self):
        line = BARNARD_1892.splitlines()[2]
        observation = parse_mpc80(line[:14] + "X" + line[15:])[0]
        self.assertFalse(is_fit_candidate(observation))

    def test_ades_uncertainty_has_priority_over_technique(self):
        observation = Observation(
            time="1892-10-16T05:06:00Z", raw={"mode": "MIC", "rmsRA": 1.2, "rmsDec": 1.4},
        )
        self.assertEqual(observation_sigma_arcsec(observation), (1.2, 1.4))

    def test_normal_place_removes_nightly_motion_from_its_uncertainty(self):
        start = datetime(2008, 10, 7, 5, tzinfo=timezone.utc)
        observations = [
            Observation(
                designation="206P", station="703",
                time=(start + timedelta(minutes=10 * index)).isoformat(),
                # 206P moved about a degree per day near its 2008 Earth passage.
                ra_deg=325.5 + 0.007 * index, dec_deg=36.4 - 0.012 * index,
                raw={"rmsRA": .4, "rmsDec": .4},
            )
            for index in range(9)
        ]
        place = normal_place_observations(observations)[0]
        self.assertLess(place.raw["rmsRA"], .5)
        self.assertLess(place.raw["rmsDec"], .5)
        self.assertAlmostEqual(place.ra_deg, 325.5 + 0.007 * 4, places=6)
        self.assertEqual(parse_observation_time(place.time), start + timedelta(minutes=40))

    def test_normal_place_before_1970_does_not_use_unix_timestamps(self):
        start = datetime(1892, 10, 20, 19, tzinfo=timezone.utc)
        observations = [
            Observation(
                station="045", time=(start + timedelta(minutes=5 * index)).isoformat(),
                ra_deg=298.0, dec_deg=10.3,
            )
            for index in range(6)
        ]
        place = normal_place_observations(observations)[0]
        self.assertEqual(parse_observation_time(place.time).year, 1892)


class HistoricalRecordTests(unittest.TestCase):
    LOW_PRECISION = (
        "0078PJ81L010  A1981 06 08.4391  00 36 09    +08 37.6             19.5 T M6070711\n"
        "0078PJ81L010  A1981 08 01.44    01 44 12.2  +15 03 59            19   N M6226711\n"
    )

    def test_low_precision_mpc80_positions_are_parsed(self):
        minute, second = parse_mpc80(self.LOW_PRECISION)
        self.assertAlmostEqual(minute.ra_deg, 15.0 * (36 / 60 + 9 / 3600), places=9)
        self.assertAlmostEqual(minute.dec_deg, 8.0 + 37.6 / 60.0, places=9)
        self.assertAlmostEqual(second.dec_deg, 15.0 + 3 / 60 + 59 / 3600, places=9)

    def test_rounding_sets_a_floor_on_the_uncertainty(self):
        minute, second = parse_mpc80(self.LOW_PRECISION)
        sigma_ra, sigma_dec = observation_sigma_arcsec(minute)
        # 1 s of RA is 15″·cos(δ); 0.1′ of Dec is 6″.
        self.assertAlmostEqual(sigma_ra, 7.5 * math.cos(math.radians(minute.dec_deg)), places=6)
        self.assertAlmostEqual(sigma_dec, 3.0)
        self.assertEqual(observation_sigma_arcsec(second), (2.0, 2.0))

    def test_ades_uncertainty_has_a_systematic_floor(self):
        observation = Observation(
            time="2025-05-09T09:10:00Z", raw={"rmsRA": .08, "rmsDec": .1},
        )
        self.assertEqual(observation_sigma_arcsec(observation), (0.3, 0.3))


class EngineTests(unittest.TestCase):
    def test_long_arcs_use_precise_integration(self):
        self.assertEqual(tolerances_for_span(100.0), STANDARD_TOLERANCES)
        self.assertEqual(tolerances_for_span(116 * 365.25), PRECISE_TOLERANCES)

    def test_vectorised_acceleration_matches_direct_sum(self):
        epoch = 2451545.0
        ephemeris = SplinePlanetEphemeris(
            PLANETS, ("jupiter", "saturn", "earth"), epoch - 5, epoch + 5,
        )
        engine = NBodyIntegrator(ephemeris, include_solar_relativity=False)
        position = np.array([2.5, -1.0, .3])
        expected = -MU_SUN * position / np.linalg.norm(position) ** 3
        for body in engine.perturbers:
            if body.name not in ("jupiter", "saturn", "earth"):
                continue
            planet = ephemeris.position(body.name, epoch)
            relative = planet - position
            expected += body.mu_au3_day2 * (
                relative / np.linalg.norm(relative) ** 3 - planet / np.linalg.norm(planet) ** 3
            )
        engine = NBodyIntegrator(
            ephemeris, perturbers=tuple(
                body for body in engine.perturbers
                if body.name in ("jupiter", "saturn", "earth")
            ), include_solar_relativity=False,
        )
        np.testing.assert_allclose(engine.acceleration(epoch, position), expected, rtol=1e-12)


    def test_fast_nongravitational_law_matches_reference_formula(self):
        from orbit_app.orbit_engine import nongravitational_acceleration

        position = np.array([.4, .45, .05]); velocity = np.array([-.012, .016, .002])
        ephemeris = SplinePlanetEphemeris(PLANETS, ("jupiter",), 2451540.0, 2451550.0)
        for parameters in (
            {"A1": 6.5e-10, "A2": 1.55e-10},
            {"A1": 1e-9, "A2": -2e-10, "A3": 3e-11, "DT": 25.0},
            {"A2": -3e-14, "R_EXPONENT": 2.0},
        ):
            with self.subTest(parameters=parameters):
                engine = NBodyIntegrator(ephemeris, nongravitational_parameters=parameters)
                np.testing.assert_allclose(
                    engine._nongravitational_acceleration(position, velocity),  # noqa: SLF001
                    nongravitational_acceleration(position, velocity, parameters),
                    rtol=1e-13, atol=0.0,
                )

    def test_direct_spline_evaluation_matches_scipy(self):
        ephemeris = SplinePlanetEphemeris(
            PLANETS, ("jupiter", "earth", "moon"), 2400000.5, 2410000.5,
        )
        epochs = np.concatenate((
            ephemeris.epochs[[0, 1, 4567, -2, -1]],
            np.random.default_rng(7).uniform(2400000.5, 2410000.5, 500),
        ))
        for epoch in epochs:
            np.testing.assert_allclose(
                ephemeris.positions_array(("jupiter", "earth", "moon"), epoch),
                ephemeris.spline(epoch), rtol=0.0, atol=1e-14,
            )
        np.testing.assert_allclose(
            ephemeris.positions_array(("moon", "jupiter"), epochs[7]),
            ephemeris.spline(epochs[7])[[2, 0]], rtol=0.0, atol=1e-14,
        )

    def test_gravity_jacobian_matches_numerical_derivative(self):
        epoch = 2451545.0
        ephemeris = SplinePlanetEphemeris(PLANETS, ("jupiter", "earth"), epoch - 5, epoch + 5)
        engine = NBodyIntegrator(ephemeris, perturbers=tuple(
            body for body in NBodyIntegrator(ephemeris).perturbers
            if body.name in ("jupiter", "earth")
        ), include_solar_relativity=False)
        position = np.array([4.9, .8, .1]); velocity = np.zeros(3)
        analytic = engine._acceleration_partials(  # noqa: SLF001
            epoch, position, velocity, (), np.empty(0),
        )[0]
        step = 1e-6
        numeric = np.column_stack([
            (engine.acceleration(epoch, position + step * axis)
             - engine.acceleration(epoch, position - step * axis)) / (2 * step)
            for axis in np.eye(3)
        ])
        np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-14)


class ApparitionGroupingTests(unittest.TestCase):
    def test_gap_of_half_a_period_starts_a_new_apparition(self):
        observations = [
            Observation(time=f"{year}-{month:02d}-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0)
            for year, month in ((2017, 1), (2017, 7), (2020, 9), (2023, 6), (2023, 12))
        ]
        encke = OrbitSolution("2P", 2460220.5, 2.22, .85, 11.3, 334.0, 187.0, 0.0)
        groups = _observing_apparition_groups(observations, encke)
        self.assertEqual([len(group) for group in groups], [2, 1, 2])

    def test_grouping_does_not_extrapolate_a_perturbed_period(self):
        # 206P: osculating period 5.6 yr today, 6.5 yr in 1892.  Grouping by
        # gaps keeps the 1892 apparition together regardless.
        observations = [
            Observation(time=f"1892-{month:02d}-15T00:00:00Z", ra_deg=1.0, dec_deg=2.0)
            for month in (10, 11, 12)
        ] + [
            Observation(time=f"2008-{month:02d}-15T00:00:00Z", ra_deg=1.0, dec_deg=2.0)
            for month in (10, 11, 12)
        ]
        modern = OrbitSolution("206P", 2460000.5, 3.49, .55, 33.6, 202.4, 189.5, 109.2)
        groups = _observing_apparition_groups(observations, modern)
        self.assertEqual([len(group) for group in groups], [3, 3])

    def test_survey_coverage_does_not_merge_consecutive_returns(self):
        # 78P (P = 7.2 yr): followed 2011-2015, then again from 2018.  The
        # 2015-2018 gap is shorter than half a period, but one block may not
        # span most of a revolution.
        observations = [
            Observation(time=f"{year}-{month:02d}-15T00:00:00Z", ra_deg=1.0, dec_deg=2.0)
            for year, month in (
                (2011, 6), (2011, 12), (2012, 6), (2013, 1), (2014, 2), (2015, 4),
                (2018, 5), (2019, 3), (2020, 6), (2021, 4),
            )
        ]
        gehrels = OrbitSolution("78P", 2456000.5, 3.739, .463, 6.26, 210.6, 192.8, 0.3)
        groups = _observing_apparition_groups(observations, gehrels)
        self.assertEqual(
            [(group[0].time[:4], group[-1].time[:4]) for group in groups],
            [("2011", "2015"), ("2018", "2021")],
        )


class QualityGateTests(unittest.TestCase):
    @staticmethod
    def _linker(offsets):
        from orbit_app.linkage import _Linker

        groups = [
            [
                Observation(
                    time=f"{1980 + 10 * index}-10-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
                    residual_ra_arcsec=offset, residual_dec_arcsec=offset,
                    raw={"rmsRA": 1.0, "rmsDec": 1.0},
                )
                for day in range(1, 7)
            ]
            for index, offset in enumerate(offsets)
        ]
        observations = [item for group in groups for item in group]
        return _Linker(
            observations, groups, None, None, None, None, None, lambda *a, **k: None, None,
        )

    def test_each_correction_forgets_o_c_of_earlier_orbits(self):
        # A correction fits a sample and refreshes O-C only on its rows; stale
        # O-C (e.g. from the appraisal of the catalogue seed) must not enter
        # the quality gate of the new apparition.
        linker = self._linker([1.0, 60.0])
        fitted = linker.groups[1][:2]

        def fit_once(rows, *args, **kwargs):
            for observation in rows:
                observation.residual_ra_arcsec = observation.residual_dec_arcsec = 0.5
            return None, 0.5, len(rows)

        linker._fit_once = fit_once  # noqa: SLF001
        linker.fit_once(fitted)
        self.assertEqual(linker.group_quality(1)[2], 2)
        self.assertLess(linker.group_quality(1)[0], 1.0)
        self.assertTrue(all(item.residual_ra_arcsec is None for item in linker.groups[0]))

    def test_new_apparition_must_fit_its_own_astrometry(self):
        linker = self._linker([1.0, 4.0])
        self.assertFalse(linker.accepted([0, 1], new=[1]))

    def test_older_apparition_may_show_reported_model_tension(self):
        linker = self._linker([5.0, 1.0])
        self.assertTrue(linker.accepted([0, 1], new=[1]))
        self.assertFalse(self._linker([7.0, 1.0]).accepted([0, 1], new=[1]))


class LinkageRetryTests(unittest.TestCase):
    def test_failed_return_is_retried_after_another_return_is_linked(self):
        # 1P with a gravity-only start: 1910 misses while only 1986 is linked,
        # but lies between 1835 and 1986 once 1835 has been joined.
        from unittest import mock

        from orbit_app.linkage import LinkageFailed, _Linker

        groups = [
            [Observation(time=f"{year}-{month:02d}-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0)
             for month in range(1, 13)][:size]
            for year, size in ((1835, 3), (1910, 6), (1986, 12))
        ]
        observations = [item for group in groups for item in group]
        linker = _Linker(
            observations, groups, None, None, None, None, None, lambda *a, **k: None, None,
        )
        seed = OrbitSolution("1P", 2446470.5, 17.9, .967, 162.2, 58.9, 111.9, 0.0)
        attempts: list[tuple[int, tuple[int, ...]]] = []

        def link_next(linked, current, index):
            attempts.append((index, tuple(linked)))
            if index == 1 and linked == [2]:
                raise LinkageFailed("miss")
            return seed, 1.0, 10

        with mock.patch.multiple(
            linker, catalogue_shortcut=lambda: False,
            fit_anchor=lambda index: (seed, 1.0, 10), well_predicted=lambda *a: [],
            link_next=link_next, final_fit=lambda linked, current: (seed, 1.0, 10),
            _report_unfitted=lambda *a: None, accepted=lambda *a, **k: True,
            group_quality=lambda index: (1.0, 1.0, 3),
        ):
            _result, records, _note = linker.run()
        self.assertEqual(attempts, [(1, (2,)), (0, (2,)), (1, (0, 2))])
        self.assertEqual([record["status"] for record in records], ["linked"] * 3)


class LineOfVariationsTests(unittest.TestCase):
    def test_step_is_one_sigma_along_the_weakest_direction(self):
        rng = np.random.default_rng(3)
        basis = np.linalg.qr(rng.normal(size=(6, 6)))[0]
        covariance = basis @ np.diag([1e-14, 2e-14, 3e-14, 1e-13, 4e-13, 9e-10]) @ basis.T
        solution = OrbitSolution(
            "X", 2460000.5, 3.0, .5, 10, 20, 30, 40,
            state_covariance=tuple(tuple(row) for row in covariance),
        )
        step = _line_of_variations(solution)
        self.assertAlmostEqual(float(step @ np.linalg.solve(covariance, step)), 1.0, places=6)
        alignment = abs(float(step @ basis[:, -1])) / np.linalg.norm(step)
        self.assertGreater(alignment, .99)

    def test_missing_covariance_gives_no_direction(self):
        self.assertIsNone(_line_of_variations(OrbitSolution("X", 0.0, 1, 0, 0, 0, 0, 0)))


def _state_to_solution(epoch, state, designation="P/LINK"):
    elements = cartesian_to_elements(state[:3], state[3:])
    return OrbitSolution(
        designation, epoch, *elements,
        position_au=tuple(float(value) for value in state[:3]),
        velocity_au_per_day=tuple(float(value) for value in state[3:]),
        photometric_exponent=4.0, phase_correction=False,
    )


class SyntheticLinkageTests(unittest.TestCase):
    """End-to-end linkage of two apparitions several revolutions apart."""

    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(11)
        epoch = 2451545.0
        # q ≈ 1.2 AU, a = 2.6 AU, P ≈ 4.2 yr; the apparitions are 17 yr apart.
        a, e = 2.6, .54
        r = a * (1.0 - e)
        speed = math.sqrt(MU_SUN * (2.0 / r - 1.0 / a))
        incl = math.radians(12.0)
        cls.truth = np.array([
            r, 0.0, 0.0, 0.0, speed * math.cos(incl), speed * math.sin(incl),
        ])
        ephemeris = SplinePlanetEphemeris(
            PLANETS, ("mercury", "venus", "earth", "moon", "mars", "jupiter",
                      "saturn", "uranus", "neptune", "pluto"),
            epoch - 400, epoch + 6600,
        )
        cls.engine = NBodyIntegrator(ephemeris, relative_tolerance=1e-12, absolute_tolerance=1e-14)
        truth_state = StateVector(epoch, cls.truth[:3], cls.truth[3:])
        start = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
        times = [start + timedelta(days=float(day)) for day in np.linspace(-40, 40, 24)]
        times += [start + timedelta(days=float(day)) for day in np.linspace(6201, 6240, 10)]
        rows = [(Observation(time=t.isoformat(), station="500"), t, 0.0, 0.0) for t in times]
        epochs = dynamical_jd(times)
        earth = earth_heliocentric_ecliptic(epochs)
        offsets = _observation_offsets_gcrs(rows)
        received = cls.engine.propagate(truth_state, epochs)
        light_time = np.linalg.norm(received[:, :3] - earth, axis=1) / 173.1446326846693
        emitted = cls.engine.propagate(truth_state, epochs - light_time)
        ra, dec = _equatorial_direction(emitted, epochs, offsets, earth)
        cls.observations = [
            Observation(
                designation="P/LINK", time=t.isoformat().replace("+00:00", "Z"),
                ra_deg=float(alpha + rng.normal(0, .5) / 3600 / math.cos(math.radians(delta))),
                dec_deg=float(delta + rng.normal(0, .5) / 3600), station="500",
                raw={"rmsRA": .5, "rmsDec": .5},
            )
            for t, alpha, delta in zip(times, ra, dec)
        ]

    def test_distant_apparition_is_found_along_line_of_variations(self):
        observations = [replace(item, raw=dict(item.raw)) for item in self.observations]
        # A seed that fits the first apparition but misses the second by many
        # arcminutes: the joint correction alone would leave its basin.
        seed = self.truth.copy()
        seed[3:] *= 1.0 + 4e-5
        initial = _state_to_solution(2451545.0, seed)
        groups = _observing_apparition_groups(observations, initial)
        self.assertEqual(len(groups), 2)
        labels: list[str] = []
        (solution, rms, _count), records, note = link_apparitions(
            observations, groups, initial, constraints=None, nongravitational=None,
            fit_once=_fit_orbit_once, evaluate=_evaluate_solution_observations,
            report_progress=lambda label, total=None: labels.append(label),
        )
        self.assertEqual([record["status"] for record in records], ["linked", "linked"])
        self.assertTrue(any("linie variací" in label for label in labels))
        for record in records:
            self.assertLessEqual(record["normalized_rms"], LINKAGE_ACCEPT_NORMALIZED_RMS)
        self.assertLess(rms, 1.0)
        self.assertIn("NAVÁZÁNO 2/2", apparition_linkage_summary(replace(
            solution, apparition_linkage=tuple(records),
        )))

    def test_fit_orbit_routes_separate_apparitions_to_linkage(self):
        observations = [replace(item, raw=dict(item.raw)) for item in self.observations]
        initial = _state_to_solution(2451545.0, self.truth)
        solution, rms, _count = fit_orbit(observations, initial)
        self.assertEqual(
            [record["status"] for record in solution.apparition_linkage], ["linked", "linked"],
        )
        self.assertLess(rms, 1.0)
        self.assertTrue(all(item.used_in_fit for item in observations))


class ObservedEvolutionPanelTests(unittest.TestCase):
    def test_fit_integration_is_shown_without_rerunning_it(self):
        from PySide6.QtWidgets import QApplication

        from orbit_app.main_window import OrbitEvolutionPanel
        from orbit_app.orbit_evolution import EvolutionResult, EvolutionReturn, PlanetEncounter

        _application = QApplication.instance() or QApplication([])
        returns = tuple(
            EvolutionReturn(index, epoch, (1.0, 0, 0, 0, .02, 0), 3.4, .58, 31.3,
                            204.0, 181.0, 1.43, 6.4, 1 / 3.4)
            for index, epoch in ((-20, 2412420.5), (0, 2454767.5))
        )
        encounters = (
            PlanetEncounter(2423350.0, "jupiter", .252, .71, 5.1, -.02),
            PlanetEncounter(2427670.0, "jupiter", .330, .93, 5.4, .01),
        )
        evolution = EvolutionResult(
            returns, encounters, 2411000.0, 2462000.0, 6.0, "Integrace z fitu.",
        )
        solution = OrbitSolution(
            "206P", 2454767.5, 3.24, .646, 32.9, 204.1, 181.4, 0.0,
            position_au=(1.15, 0.0, 0.0), velocity_au_per_day=(0.0, .02, 0.0),
            observed_evolution=evolution,
        )
        panel = OrbitEvolutionPanel(solution)
        published = []
        panel.evolution_ready.connect(lambda source, result: published.append(result))
        panel.set_solution(solution)
        self.assertIs(panel.result, evolution)
        self.assertEqual(panel.encounters.rowCount(), 2)
        self.assertEqual(panel.returns.rowCount(), 2)
        self.assertEqual(published, [evolution])
        self.assertTrue(panel.calculate.isEnabled())


if __name__ == "__main__":
    unittest.main()
