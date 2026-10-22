"""Orbit relations to planets, collisions and impact locations."""
import os
import unittest
from pathlib import Path
from unittest import mock

os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

import numpy as np  # noqa: E402

from orbit_app import impact  # noqa: E402
from orbit_app.domain import OrbitSolution  # noqa: E402
from orbit_app.orbit_engine import MU_SUN, ImpactDetected, NBodyIntegrator, StateVector  # noqa: E402
from orbit_app.planet_relations import (  # noqa: E402
    conic_from_state, minimum_orbit_intersection_distance, nearby_resonances, nodal_distances,
    period_ratio, planet_orbit, tisserand_parameter,
)
from orbit_app.planetary import PLANETS  # noqa: E402

J2000 = 2451545.0
DATA = Path(__file__).with_name("data")


def _circular(radius: float, P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    return np.concatenate((radius * P, np.sqrt(MU_SUN / radius) * Q))


class PlanetRelationTests(unittest.TestCase):
    def test_moid_of_a_coplanar_circle_outside_earth(self):
        earth = planet_orbit("earth", J2000)
        state = _circular(1.5, earth.P, earth.Q)
        body = conic_from_state(state[:3], state[3:])
        aphelion = earth.semi_major_axis * (1.0 + earth.eccentricity)
        self.assertAlmostEqual(
            minimum_orbit_intersection_distance(body, earth), 1.5 - aphelion, places=7,
        )

    def test_moid_of_an_inclined_crossing_orbit_is_small(self):
        earth = planet_orbit("earth", J2000)
        tilt = np.radians(20.0)
        direction = earth.P
        normal = np.cos(tilt) * earth.Q + np.sin(tilt) * earth.W
        radius = float(earth.radius(np.array(0.0)))  # Earth's perihelion distance
        state = _circular(radius, direction, normal)
        body = conic_from_state(state[:3], state[3:])
        self.assertLess(minimum_orbit_intersection_distance(body, earth), 1e-6)
        ascending, descending = nodal_distances(body, earth)
        self.assertLess(min(abs(ascending), abs(descending)), 1e-9)

    def test_tisserand_and_period_ratio_of_orbits_like_jupiter(self):
        jupiter = planet_orbit("jupiter", J2000)
        same = _circular(jupiter.semi_major_axis, jupiter.P, jupiter.Q)
        body = conic_from_state(same[:3], same[3:])
        self.assertAlmostEqual(tisserand_parameter(body, jupiter), 3.0, places=9)
        self.assertAlmostEqual(period_ratio(body, jupiter), 1.0, places=9)
        inner = _circular(jupiter.semi_major_axis * 0.5 ** (2 / 3), jupiter.P, jupiter.Q)
        self.assertAlmostEqual(period_ratio(conic_from_state(inner[:3], inner[3:]), jupiter), .5)
        self.assertIn((0.5, "1:2"), nearby_resonances(.45, .55))

    def test_hyperbolic_orbit_has_finite_relations(self):
        earth = planet_orbit("earth", J2000)
        state = np.concatenate((.9 * earth.P, .03 * earth.Q + .005 * earth.W))
        body = conic_from_state(state[:3], state[3:])
        self.assertGreater(body.eccentricity, 1.0)
        self.assertTrue(np.isfinite(minimum_orbit_intersection_distance(body, earth)))
        self.assertTrue(np.isfinite(tisserand_parameter(body, earth)))
        self.assertTrue(np.isnan(period_ratio(body, earth)))


class CollisionTests(unittest.TestCase):
    def test_integration_stops_inside_a_planet(self):
        from orbit_app.orbit_fit import _InterpolatedPlanetEphemeris

        epoch = J2000
        engine = NBodyIntegrator(_InterpolatedPlanetEphemeris(PLANETS, epoch - 2, epoch + 3))
        solution = _earth_bound_solution(epoch)
        start = StateVector(
            epoch, np.array(solution.position_au), np.array(solution.velocity_au_per_day),
        )
        with self.assertRaises(ImpactDetected) as caught:
            engine.propagate(start, np.array([epoch + 1.0]))
        self.assertEqual(caught.exception.body, "earth")
        prediction = impact.locate_impact(engine, start, "earth", caught.exception.epoch_jd)
        self.assertIsNotNone(prediction)
        self.assertLess(prediction.entry.epoch_jd, prediction.surface.epoch_jd)
        self.assertAlmostEqual(prediction.surface.elevation_angle_deg, 90.0, delta=3.0)
        self.assertTrue(-90.0 <= prediction.surface.latitude_deg <= 90.0)

    def test_geodetic_conversion_round_trip(self):
        import astropy.units as u
        from astropy.coordinates import EarthLocation
        from astropy.time import Time

        location = EarthLocation.from_geodetic(14.78 * u.deg, 49.91 * u.deg, 500.0 * u.m)
        instant = Time(2454746.6, format="jd", scale="tdb")
        gcrs = location.get_gcrs(obstime=instant).cartesian.xyz.to_value(u.au)
        ecliptic = impact._equatorial_to_ecliptic(gcrs)  # noqa: SLF001
        latitude, longitude, height = impact.earth_geodetic(2454746.6, ecliptic)
        self.assertAlmostEqual(latitude, 49.91, places=4)
        self.assertAlmostEqual(longitude, 14.78, places=4)
        self.assertAlmostEqual(height, .5, delta=.01)

    @unittest.skipUnless(PLANETS.jpl is not None, "needs the JPL DE440 kernel")
    def test_2008_tc3_entry_matches_jpl_prediction(self):
        # State fitted from tests/data/2008TC3_sub.obs (every 8th MPC record).
        # JPL/CNEOS: 2008-10-07 02:45:38.45 UT at 65.4 km, 20.933 N 31.412 E,
        # 12.4 km/s, 20.0 degrees.
        from orbit_app.orbit_evolution import _evolution_engine

        state = (
            0.9706506146008016, 0.24083505321042598, 0.00012874908750975503,
            -0.00832660020357009, 0.01715289686622864, -0.0008135736008060739,
        )
        solution = OrbitSolution(
            "2008 TC3", 2454746.466944425, 1.3, .306, 2.5, 194.1, 234.4, 0.0,
            position_au=state[:3], velocity_au_per_day=state[3:],
        )
        engine = _evolution_engine(solution, solution.epoch_jd, solution.epoch_jd + 1)
        start = StateVector(solution.epoch_jd, np.array(state[:3]), np.array(state[3:]))
        with mock.patch.object(impact, "EARTH_ENTRY_ALTITUDE_KM", 65.4):
            prediction = impact.locate_impact(engine, start, "earth", solution.epoch_jd + .2)
        entry = prediction.entry
        jpl_seconds = 2 * 3600 + 45 * 60 + 38.45
        seconds = entry.utc.hour * 3600 + entry.utc.minute * 60 + entry.utc.second + (
            entry.utc.microsecond / 1e6
        )
        self.assertEqual((entry.utc.year, entry.utc.month, entry.utc.day), (2008, 10, 7))
        self.assertLess(abs(seconds - jpl_seconds), 3.0)
        self.assertAlmostEqual(entry.latitude_deg, 20.933, delta=.05)
        self.assertAlmostEqual(entry.longitude_deg, 31.412, delta=.05)
        self.assertAlmostEqual(entry.speed_kms, 12.4, delta=.1)
        self.assertAlmostEqual(entry.elevation_angle_deg, 20.0, delta=.5)


def _earth_bound_solution(epoch: float = J2000) -> OrbitSolution:
    """A body 50 000 km from the geocentre heading straight in at 12 km/s."""
    earth = PLANETS.position("earth", epoch)
    earth_velocity = (PLANETS.position("earth", epoch + 1e-3) - earth) / 1e-3
    offset = np.array([50_000.0, 0.0, 0.0]) / impact.AU_KM
    velocity = earth_velocity - offset / np.linalg.norm(offset) * 12.0 * 86400 / impact.AU_KM
    return OrbitSolution(
        "IMPACTOR", epoch, 1.0, .1, 1.0, 0.0, 0.0, 0.0,
        position_au=tuple(earth + offset), velocity_au_per_day=tuple(velocity),
    )


class ImpactReportingTests(unittest.TestCase):
    def test_fitted_orbit_is_checked_for_an_imminent_impact(self):
        prediction = impact.imminent_impact(_earth_bound_solution(), days=2.0)
        self.assertIsNotNone(prediction)
        self.assertEqual(prediction.body, "earth")
        self.assertIsNotNone(prediction.entry)
        harmless = OrbitSolution(
            "FAR", J2000, 2.5, .1, 5.0, 0.0, 0.0, 0.0,
            position_au=(2.3, 0.0, .1), velocity_au_per_day=(0.0, .0112, 0.0),
        )
        self.assertIsNone(impact.imminent_impact(harmless, days=30.0))

    def test_preview_positions_end_at_the_impact(self):
        from orbit_app.preview_orbit import heliocentric_xyz

        solution = _earth_bound_solution()
        positions = heliocentric_xyz(solution, solution.epoch_jd + np.array([-.1, .01, .5, 3.0]))
        self.assertTrue(np.all(np.isfinite(positions[:2])))
        self.assertTrue(np.all(np.isnan(positions[2:])))
        self.assertTrue(np.all(np.isnan(heliocentric_xyz(solution, np.array([np.nan])))))

    def test_known_impact_is_not_integrated_again(self):
        # The Solar System animation asks for one instant per frame; after
        # the impact it must not re-integrate and re-locate it every frame.
        from orbit_app import preview_orbit

        solution = _earth_bound_solution()
        preview_orbit.heliocentric_xyz(solution, solution.epoch_jd + np.array([.5]))
        with mock.patch.object(
            preview_orbit.NBodyIntegrator, "propagate", side_effect=AssertionError,
        ):
            later = preview_orbit.heliocentric_xyz(
                solution, solution.epoch_jd + np.array([1.0, 2.0, 30.0]),
            )
        self.assertTrue(np.all(np.isnan(later)))

    def test_elements_tab_announces_the_impact(self):
        from dataclasses import replace

        from PySide6.QtWidgets import QApplication

        from orbit_app.main_window import ElementsPanel

        _application = QApplication.instance() or QApplication([])
        solution = _earth_bound_solution()
        solution = replace(solution, impact=impact.imminent_impact(solution, days=2.0))
        panel = ElementsPanel(solution)
        panel.set_solution(solution)
        self.assertFalse(panel.impact_banner.isHidden())
        self.assertIn("DOPADU", panel.impact_banner.text())
        self.assertIn("PŘEDPOVĚĎ DOPADU", panel.text.toPlainText())
        panel.set_solution(replace(solution, impact=None))
        self.assertTrue(panel.impact_banner.isHidden())


class NearEarthShortArcTests(unittest.TestCase):
    def test_range_search_finds_bound_branch_of_2008_tc3(self):
        # The discovery arc ends an hour before impact, a few Earth radii away:
        # without Earth's attraction and very small ranges every branch was a
        # spurious heliocentric hyperbola.
        from orbit_app.mpc import parse_mpc80
        from orbit_app.orbit_fit import initial_orbit_from_observations

        observations = parse_mpc80((DATA / "2008TC3_sub.obs").read_text(), "2008 TC3")
        best = initial_orbit_from_observations(observations, return_candidates=True)[0]
        self.assertTrue(0.0 < best.eccentricity < 1.0)
        self.assertLess(best.semi_major_axis_au, 3.0)


    def test_range_search_reaches_2024_bx1_a_few_earth_radii_away(self):
        # Observed until minutes before impact; 36 000 km away mid-arc.  With
        # the MPC's ADES uncertainties (0.3") every coarse branch used to miss
        # by degrees: the grid did not reach the range and the finite
        # differences drowned in integration noise.
        from orbit_app.mpc import parse_mpc80
        from orbit_app.orbit_fit import initial_orbit_from_observations

        observations = parse_mpc80((DATA / "2024BX1.obs").read_text(), "2024 BX1")
        for observation in observations:
            observation.raw["rmsRA"] = observation.raw["rmsDec"] = .3
        best = initial_orbit_from_observations(observations, return_candidates=True)[0]
        self.assertTrue(0.0 < best.eccentricity < 1.0)
        self.assertAlmostEqual(best.semi_major_axis_au, 1.39, delta=.25)

class BoundCentreTests(unittest.TestCase):
    def test_planetocentric_centre_only_for_a_bound_body(self):
        from orbit_app.orbit_evolution import bound_centers, planet_velocity

        epoch = J2000
        jupiter = PLANETS.position("jupiter", epoch)
        jupiter_velocity = planet_velocity("jupiter", epoch)
        offset = np.array([.05, 0.0, 0.0])
        circular = np.sqrt(MU_SUN * 9.547919e-4 / .05)
        captured = OrbitSolution(
            "X", epoch, 5.2, .05, 1.0, 0.0, 0.0, 0.0,
            position_au=tuple(jupiter + offset),
            velocity_au_per_day=tuple(jupiter_velocity + np.array([0.0, circular, 0.0])),
        )
        self.assertEqual(bound_centers(captured), ("jupiter",))
        free = OrbitSolution(
            "Y", epoch, 1.3, .3, 2.0, 0.0, 0.0, 0.0,
            position_au=(1.3, 0.0, 0.0), velocity_au_per_day=(0.0, .015, 0.0),
        )
        self.assertEqual(bound_centers(free), ())


class EvolutionViewTests(unittest.TestCase):
    def test_planetocentric_view_is_offered_only_for_captures(self):
        from PySide6.QtWidgets import QApplication

        from orbit_app.main_window import OrbitEvolutionPanel
        from orbit_app.orbit_evolution import CaptureInterval, EvolutionResult

        _application = QApplication.instance() or QApplication([])
        solution = OrbitSolution(
            "X", J2000, 5.2, .05, 1.0, 0.0, 0.0, 0.0,
            position_au=(5.2, 0.0, 0.0), velocity_au_per_day=(0.0, .0075, 0.0),
        )
        panel = OrbitEvolutionPanel(solution)
        views = [panel.element_center.itemData(i) for i in range(panel.element_center.count())]
        self.assertIn("relation:earth", views)
        self.assertFalse(any(item.startswith("captured:") for item in views))
        result = EvolutionResult(
            (), (), J2000, J2000 + 100, 1.0, "",
            captures=(CaptureInterval("jupiter", J2000 + 10, J2000 + 40, .02, .05),),
        )
        with mock.patch("orbit_app.main_window.QMessageBox.warning"):
            panel._completed(result)  # noqa: SLF001
        views = [panel.element_center.itemData(i) for i in range(panel.element_center.count())]
        self.assertIn("captured:jupiter", views)
        panel._relation_series = lambda planet: None  # no background thread in a unit test
        panel.element_center.setCurrentIndex(panel.element_center.findData("relation:earth"))
        self.assertEqual(panel.graph_metric.itemData(0), "distance")


if __name__ == "__main__":
    unittest.main()
