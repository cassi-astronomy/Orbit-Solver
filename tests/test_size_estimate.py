"""Asteroid diameter range implied by the absolute magnitude H."""
import os
import unittest
from dataclasses import replace

os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

from orbit_app.domain import OrbitSolution  # noqa: E402
from orbit_app.photometry import diameter_from_absolute_magnitude_km  # noqa: E402
from orbit_app.presentation import asteroid_size_lines, format_diameter  # noqa: E402

ASTEROID = OrbitSolution("2024 PT5", 2460600.5, 1.0, .02, 1.5, 0.0, 0.0, 0.0, absolute_magnitude=27.6)


class DiameterTests(unittest.TestCase):
    def test_standard_conversion(self):
        # D = 1329 km / sqrt(p) * 10^(-H/5): H = 15, p = 0.25 gives 2.658 km.
        self.assertAlmostEqual(diameter_from_absolute_magnitude_km(15.0, .25), 2.658, places=3)
        # Five magnitudes are a factor of ten; a quarter of the albedo doubles D.
        self.assertAlmostEqual(
            diameter_from_absolute_magnitude_km(20.0, .25) * 10.0,
            diameter_from_absolute_magnitude_km(15.0, .25),
        )
        self.assertAlmostEqual(
            diameter_from_absolute_magnitude_km(15.0, .0625),
            2.0 * diameter_from_absolute_magnitude_km(15.0, .25),
        )
        with self.assertRaises(ValueError):
            diameter_from_absolute_magnitude_km(15.0, 0.0)

    def test_units(self):
        self.assertEqual(format_diameter(2.658), "2.7 km")
        self.assertEqual(format_diameter(946.0), "946 km")
        self.assertEqual(format_diameter(.0083), "8.3 m")
        self.assertEqual(format_diameter(.4), "400 m")
        self.assertEqual(format_diameter(float("nan")), "—")


class ReportTests(unittest.TestCase):
    def test_asteroid_gets_typical_and_extreme_ranges(self):
        lines = asteroid_size_lines(ASTEROID)
        self.assertEqual(len(lines), 4)
        # H = 27.6: 8 m for p = 0.25 … 18 m for p = 0.05.
        self.assertIn("8.0 m – 18 m", lines[1])
        self.assertIn("0.25", lines[1])
        self.assertIn("0.05", lines[1])
        self.assertIn("p_V = 0.50 až 0.03", lines[2])

    def test_no_estimate_for_comets_or_a_placeholder_h(self):
        for comet in ("205P", "C/2023 A3", "P/2008 R6", "73P-C", "1I/ʻOumuamua"):
            with self.subTest(comet=comet):
                self.assertEqual(asteroid_size_lines(replace(ASTEROID, designation=comet)), [])
        self.assertEqual(asteroid_size_lines(replace(
            ASTEROID, designation="2024 AB", display_name="C/2024 AB (comet)",
        )), [])
        self.assertEqual(asteroid_size_lines(replace(ASTEROID, absolute_magnitude_known=False)), [])

    def test_demo_orbit_has_only_a_placeholder_h(self):
        from orbit_app.preview_orbit import demo_solution

        self.assertFalse(demo_solution().absolute_magnitude_known)


if __name__ == "__main__":
    unittest.main()
