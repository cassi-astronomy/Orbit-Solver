"""Small numerical/presentation smoke suite that does not import PySide6."""

import os

# These suites assert the Czech source texts; English is covered by
# tests/test_i18n.py.  Must be set before orbit_app modules are imported.
os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from orbit_app.domain import OrbitSolution
from orbit_app.mpc import parse_mpc80
from orbit_app.orbit_engine import MU_SUN, NBodyIntegrator, StateVector
from orbit_app.presentation import (
    datetime_from_jd, format_dec_sexagesimal, format_ra_sexagesimal,
    project_ecliptic_camera,
)


class _NoPlanets:
    def position(self, body: str, jd: float) -> np.ndarray:
        raise AssertionError("Sun-only test must not request a planet.")


class HeadlessSmokeTests(unittest.TestCase):
    def test_presentation_import_does_not_load_qt(self):
        code = (
            "import sys; import orbit_app.presentation; "
            "assert 'PySide6' not in sys.modules"
        )
        subprocess.run([sys.executable, "-c", code], check=True)

    def test_bundled_mpc_fixture_is_parseable(self):
        fixture = Path(__file__).with_name("data") / "ec0001.obs"
        self.assertEqual(len(parse_mpc80(fixture.read_text(encoding="utf-8"))), 24)

    def test_presentation_helpers_are_available_without_gui(self):
        self.assertEqual(format_ra_sexagesimal(188.73625), "12 34 56.70")
        self.assertEqual(format_dec_sexagesimal(-12.5824167), "−12 34 56.7")
        self.assertEqual(datetime_from_jd(2451545.0).year, 2000)
        x, y, depth = project_ecliptic_camera(np.eye(3), 0.0, 0.0)
        np.testing.assert_allclose(x, [0.0, 1.0, 0.0])
        np.testing.assert_allclose(y, [0.0, 0.0, 1.0])
        np.testing.assert_allclose(depth, [1.0, 0.0, 0.0])

    def test_sun_only_orbit_propagates_without_gui(self):
        initial = StateVector(
            2451545.0,
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, math.sqrt(MU_SUN), 0.0]),
        )
        states = NBodyIntegrator(_NoPlanets(), perturbers=()).propagate(
            initial, np.array([2451545.0, 2451546.0]),
        )
        self.assertEqual(states.shape, (2, 6))
        self.assertTrue(np.all(np.isfinite(states)))

    def test_domain_model_is_gui_independent(self):
        solution = OrbitSolution("Test", 2451545.0, 2.0, 0.1, 3, 4, 5, 6)
        self.assertEqual(solution.designation, "Test")


if __name__ == "__main__":
    unittest.main()
