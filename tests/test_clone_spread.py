"""Optional spread of covariance clones in the orbit evolution."""
import os
import unittest

import numpy as np

os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

from orbit_app.domain import OrbitSolution
from orbit_app.orbit_evolution import (
    calculate_orbit_evolution_between, clone_spread_summary, evolution_clone_spread,
)
from orbit_app.orbit_fit import _formal_element_uncertainties

try:
    from qt_support import dispose, qt_application
except ImportError:  # run as tests.<module> from the repository root
    from tests.qt_support import dispose, qt_application

EPOCH = 2460000.5


def _solution(with_forces: bool = False) -> OrbitSolution:
    sigma_r, sigma_v = 1e-7, 1e-9
    state_covariance = np.diag([sigma_r ** 2] * 3 + [sigma_v ** 2] * 3)
    kwargs = {}
    if with_forces:
        full = np.zeros((8, 8))
        full[:6, :6] = state_covariance
        full[6, 6] = (2e-9) ** 2
        full[7, 7] = (2e-10) ** 2
        kwargs = {
            "parameter_covariance": tuple(map(tuple, full)),
            "parameter_names": ("x", "y", "z", "vx", "vy", "vz", "A1", "A2"),
            "nongravitational_parameters": {"A1": 1e-8, "A2": -3e-10},
        }
    return OrbitSolution(
        "CLONE TEST", EPOCH, 3.0, .4, 6.0, 210.0, 190.0, 0.0,
        position_au=(1.8, .2, .05), velocity_au_per_day=(-.002, .0135, .0012),
        state_covariance=tuple(map(tuple, state_covariance)), **kwargs,
    )


class CloneSpreadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.solution = _solution()
        cls.result = calculate_orbit_evolution_between(
            cls.solution, EPOCH - 400.0, EPOCH + 1200.0, include_encounters=False,
        )

    def test_clones_follow_the_nominal_timeline(self):
        spread = evolution_clone_spread(self.solution, self.result, 6)
        self.assertEqual(spread.count, 6)
        self.assertEqual(spread.failed, 0)
        self.assertFalse(spread.includes_forces)
        q = spread.elements["perihelion_au"]
        self.assertEqual(q.shape, (6, len(self.result.timeline)))
        nominal = np.array([item.perihelion_au for item in self.result.timeline])
        # Tiny covariance: the clones stay close to the nominal curve, on
        # both sides of the epoch.
        self.assertLess(float(np.nanmax(np.abs(q - nominal))), 1e-4)
        self.assertGreater(float(np.nanmax(np.abs(q - nominal))), 0.0)

    def test_spread_is_reproducible(self):
        first = evolution_clone_spread(self.solution, self.result, 3)
        second = evolution_clone_spread(self.solution, self.result, 3)
        np.testing.assert_array_equal(
            first.elements["eccentricity"], second.elements["eccentricity"],
        )

    def test_force_uncertainty_is_sampled_when_the_fit_has_it(self):
        solution = _solution(with_forces=True)
        result = calculate_orbit_evolution_between(
            solution, EPOCH, EPOCH + 800.0, include_encounters=False,
        )
        spread = evolution_clone_spread(solution, result, 4)
        self.assertTrue(spread.includes_forces)
        lines = clone_spread_summary(spread, result)
        self.assertIn("4 klonů", lines[0])
        self.assertIn("negravitační", lines[0])
        self.assertTrue(any("konec výpočtu" in line for line in lines))

    def test_orbit_without_covariance_is_refused(self):
        from dataclasses import replace

        with self.assertRaises(ValueError):
            evolution_clone_spread(replace(self.solution, state_covariance=None), self.result, 3)


class FullCovarianceTests(unittest.TestCase):
    def test_force_columns_are_returned_in_physical_units(self):
        rng = np.random.default_rng(1)
        jacobian = rng.normal(size=(40, 8))
        residual = rng.normal(size=40)
        state = np.array([1.8, .2, .05, -.002, .0135, .0012])
        out: list = []
        _uncertainties, state_covariance = _formal_element_uncertainties(
            state, (3.0, .4, 6.0, 210.0, 190.0, 0.0), jacobian, residual, 8,
            ("A1", "DT"), covariance_out=out,
        )
        self.assertEqual(len(out), 1)
        full = out[0]
        self.assertEqual(full.shape, (8, 8))
        np.testing.assert_allclose(full[:6, :6], state_covariance)
        # A1 is optimised in units of 1e-8 AU/day², DT in days.
        scaled = np.linalg.pinv(jacobian.T @ jacobian) * float(residual @ residual / 32)
        self.assertAlmostEqual(full[6, 6] / scaled[6, 6], 1e-16, delta=1e-18)
        self.assertAlmostEqual(full[7, 7] / scaled[7, 7], 1.0, delta=1e-6)


class ImpactSamplerTests(unittest.TestCase):
    def test_full_covariance_varies_the_fitted_forces(self):
        from orbit_app.impact import clone_engine, covariance_clone_sampler

        solution = _solution(with_forces=True)
        mean, factor, names = covariance_clone_sampler(solution)
        self.assertEqual(names[6:], ("A1", "A2"))
        self.assertEqual(mean.shape, (8,))
        self.assertAlmostEqual(mean[6], 1e-8)
        vector = mean + factor @ np.ones(8)
        engine = clone_engine(_StubEngine(), solution, names, vector)
        self.assertAlmostEqual(engine.parameters["A1"], vector[6])
        self.assertAlmostEqual(engine.parameters["A2"], vector[7])

    def test_state_only_without_fitted_forces(self):
        from orbit_app.impact import clone_engine, covariance_clone_sampler

        solution = _solution()
        mean, _factor, names = covariance_clone_sampler(solution)
        self.assertEqual(len(names), 6)
        stub = _StubEngine()
        self.assertIs(clone_engine(stub, solution, names, mean), stub)


class _StubEngine:
    parameters: dict = {}

    def with_parameters(self, parameters):
        clone = _StubEngine(); clone.parameters = dict(parameters)
        return clone


class ClonePanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = qt_application()

    def test_panel_draws_and_reports_the_spread(self):
        from orbit_app.main_window import OrbitEvolutionPanel

        solution = _solution()
        result = calculate_orbit_evolution_between(
            solution, EPOCH - 200.0, EPOCH + 900.0, include_encounters=False,
        )
        spread = evolution_clone_spread(solution, result, 4)
        panel = OrbitEvolutionPanel(solution)
        try:
            self.assertFalse(panel.clones_enabled.isChecked())
            self.assertFalse(panel.clone_count.isEnabled())
            panel.clones_enabled.setChecked(True)
            self.assertTrue(panel.clone_count.isEnabled())
            panel._completed((result, spread))  # noqa: SLF001
            self.assertIs(panel.clone_spread, spread)
            self.assertIn("Rozptyl 4 klonů", panel._report_text())  # noqa: SLF001
            for index in range(panel.graph_metric.count()):
                panel.graph_metric.setCurrentIndex(index)   # every metric draws
            # A plain EvolutionResult (the fit's own integration) has no spread.
            panel._completed(result)  # noqa: SLF001
            self.assertIsNone(panel.clone_spread)
        finally:
            dispose(panel)


if __name__ == "__main__":
    unittest.main()
