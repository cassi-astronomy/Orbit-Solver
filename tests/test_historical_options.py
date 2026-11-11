"""Linkage options for historical returns: force switch epoch, historical
uncertainties, and the warning for long arcs without JPL DE440."""
import os
import unittest
from unittest import mock

os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

from orbit_app import orbit_fit
from orbit_app.astrometry import (
    SIGMA_SCALE_KEY, observation_sigma_arcsec, scale_historical_sigmas,
)
from orbit_app.domain import Observation
from orbit_app.linkage import historical_split_epoch

try:
    from qt_support import dispose, qt_application
except ImportError:  # run as tests.<module> from the repository root
    from tests.qt_support import dispose, qt_application

# 205P-like returns (TDB JD): 1896-1897, 2008-2009, 2015-2016.
SPANS = [(2413807.0, 2413930.0), (2454720.0, 2454900.0), (2457260.0, 2457450.0)]
JD_1952 = 0.5 * (2413930.0 + 2454720.0)
JD_1998_END = 2451178.5     # 1998-12-31


def _row(time: str, station: str = "020") -> Observation:
    return Observation(
        designation="205P", time=time, ra_deg=10.0, dec_deg=5.0, station=station,
        raw={"mpc_note2": "M"},   # micrometer: 3″ default
    )


class SplitEpochTests(unittest.TestCase):
    def test_default_is_the_middle_of_the_longest_gap(self):
        self.assertEqual(historical_split_epoch(SPANS), (JD_1952, False))

    def test_requested_epoch_inside_a_gap_is_used(self):
        self.assertEqual(historical_split_epoch(SPANS, JD_1998_END), (JD_1998_END, False))
        # Also in the short gap between the modern returns.
        self.assertEqual(historical_split_epoch(SPANS, 2456000.0), (2456000.0, False))

    def test_requested_epoch_inside_a_return_falls_back(self):
        self.assertEqual(historical_split_epoch(SPANS, 2454800.0), (JD_1952, True))
        self.assertEqual(historical_split_epoch(SPANS, 2400000.0), (JD_1952, True))

    def test_without_a_gap_there_is_no_switch(self):
        self.assertEqual(historical_split_epoch([SPANS[0]]), (None, False))


class HistoricalSigmaTests(unittest.TestCase):
    def test_only_older_rows_are_scaled_and_the_scale_is_cleared(self):
        old, new = _row("1896-10-01T00:00:00Z"), _row("2008-10-01T00:00:00Z")
        self.assertEqual(scale_historical_sigmas([old, new], 1950, 4.0), 1)
        self.assertEqual(observation_sigma_arcsec(old), (12.0, 12.0))
        self.assertNotIn(SIGMA_SCALE_KEY, new.raw)
        # Turning the option off removes the scale from every row.
        self.assertEqual(scale_historical_sigmas([old, new], None, None), 0)
        self.assertEqual(observation_sigma_arcsec(old), (3.0, 3.0))

    def test_decimal_year_boundary(self):
        early, late = _row("1950-03-01T00:00:00Z"), _row("1950-09-01T00:00:00Z")
        self.assertEqual(scale_historical_sigmas([early, late], 1950.5, 2.0), 1)
        self.assertIn(SIGMA_SCALE_KEY, early.raw)
        self.assertNotIn(SIGMA_SCALE_KEY, late.raw)

    def test_factor_of_one_does_nothing(self):
        old = _row("1896-10-01T00:00:00Z")
        self.assertEqual(scale_historical_sigmas([old], 1950, 1.0), 0)
        self.assertEqual(observation_sigma_arcsec(old), (3.0, 3.0))


class ForcePeriodTests(unittest.TestCase):
    PARAMETERS = {
        "A1": 1e-8, "A2": -1e-9, "A1H": 2e-8, "A2H": -2e-9, "A1H2": 3e-8, "A2H2": -3e-9,
        "SPLIT_JD": 2449000.5, "SPLIT_JD2": 2440000.5,
    }

    def test_each_epoch_uses_its_period(self):
        from orbit_app.orbit_engine import (
            active_nongravitational_coefficients, force_period, force_period_splits,
        )

        splits = force_period_splits(self.PARAMETERS)
        self.assertEqual(splits, (2449000.5, 2440000.5))
        self.assertEqual(force_period(splits, 2460000.0), 0)
        self.assertEqual(force_period(splits, 2445000.0), 1)
        self.assertEqual(force_period(splits, 2430000.0), 2)
        self.assertEqual(
            active_nongravitational_coefficients(self.PARAMETERS, 2430000.0), (3e-8, -3e-9, 0.0),
        )
        self.assertEqual(
            active_nongravitational_coefficients(self.PARAMETERS, 2445000.0), (2e-8, -2e-9, 0.0),
        )

    def test_integrator_matches_the_reference_law(self):
        import numpy as np

        from orbit_app.orbit_engine import NBodyIntegrator, nongravitational_acceleration
        from orbit_app.planetary import PLANETS

        engine = NBodyIntegrator(PLANETS, perturbers=(), nongravitational_parameters=self.PARAMETERS)
        position, velocity = np.array([1.5, .2, .1]), np.array([-.002, .014, .001])
        for jd in (2460000.0, 2445000.0, 2430000.0):
            np.testing.assert_allclose(
                engine._nongravitational_acceleration(position, velocity, jd),  # noqa: SLF001
                nongravitational_acceleration(position, velocity, self.PARAMETERS, jd),
                rtol=1e-12, atol=1e-25,
            )

    def test_switch_epochs_must_decrease(self):
        from orbit_app.orbit_engine import force_period_splits

        with self.assertRaises(FloatingPointError):
            force_period_splits({"SPLIT_JD": 2440000.5, "SPLIT_JD2": 2449000.5})

    def test_years_are_parsed_and_formatted(self):
        from orbit_app.main_window import format_split_years, parse_split_years, split_years_to_jds

        self.assertEqual(parse_split_years("2008, 1993"), [1993.0, 2008.0])
        self.assertEqual(parse_split_years("1998,9; 2008"), [1998.9, 2008.0])
        self.assertEqual(parse_split_years(""), [])
        self.assertEqual(format_split_years([1993.0, 2008.5]), "1993, 2008.5")
        self.assertEqual(format_split_years(1998.9), "1998.9")   # older saved value
        self.assertEqual(len(split_years_to_jds([1993.0, 2008.0])), 2)
        self.assertIsNone(split_years_to_jds(None))


class De440WarningTests(unittest.TestCase):
    ROWS = [_row("1892-10-14T03:00:00Z"), _row("2008-10-11T00:00:00Z")]

    def test_long_arc_without_de440_is_flagged(self):
        with mock.patch.object(orbit_fit.PLANETS, "jpl", None):
            warning = orbit_fit.long_arc_ephemeris_warning(self.ROWS)
        self.assertIn("DE440", warning)
        self.assertIn("116 let", warning)

    def test_short_arc_or_de440_is_fine(self):
        short = [_row("2008-10-11T00:00:00Z"), _row("2009-02-01T00:00:00Z")]
        with mock.patch.object(orbit_fit.PLANETS, "jpl", None):
            self.assertEqual(orbit_fit.long_arc_ephemeris_warning(short), "")
        with mock.patch.object(orbit_fit.PLANETS, "jpl", object()):
            self.assertEqual(orbit_fit.long_arc_ephemeris_warning(self.ROWS), "")


class LinkageOptionsDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = qt_application()

    def test_options_round_trip(self):
        from orbit_app.main_window import OrbitConstraintsDialog, decimal_year_to_jd
        from orbit_app.preview_orbit import demo_solution

        options = {
            "historical_nongravitational": True, "historical_split_year": [1993.0, 2008.5],
            "accept_model_tension": False, "historical_sigma": [1950, 4.0],
        }
        dialog = OrbitConstraintsDialog(
            demo_solution(), {}, {"A1": 0.0, "A2": 0.0}, linkage_options=options,
        )
        try:
            self.assertEqual(dialog.selected_linkage_options(), options)
            dialog.historical_sigma.setChecked(False)
            dialog.historical_split_year.setText("")
            selected = dialog.selected_linkage_options()
            self.assertIsNone(selected["historical_sigma"])
            self.assertIsNone(selected["historical_split_year"])
        finally:
            dispose(dialog)
        self.assertAlmostEqual(decimal_year_to_jd(1999.0), 2451179.5, delta=.002)  # TT
        self.assertIsNone(decimal_year_to_jd(None))
        self.assertIsNone(decimal_year_to_jd(0.0))


if __name__ == "__main__":
    unittest.main()
