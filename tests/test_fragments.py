"""Split comets: fragment labels, component selection and the GUI selector."""
import atexit
import os
import unittest
from dataclasses import replace

os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

from orbit_app.astrometry import (  # noqa: E402
    designation_fragment, fragment_counts, fragment_return_counts, is_fit_candidate,
    mark_duplicate_measurements, mpc80_fragment, observation_fragment, primary_fragment,
    select_fragment,
)
from orbit_app.mpc import MpcClient, parse_mpc80  # noqa: E402


# Keep the QApplication wrapper alive for the rest of the test process.  If the
# last Python reference disappears while Qt still owns shared objects created by
# MainWindow, Linux/Qt may abort during interpreter shutdown even though every
# assertion passed.
_QT_APPLICATION = None
_QT_SHUTDOWN_REGISTERED = False


def _shutdown_qt_application() -> None:
    """Destroy Qt explicitly before PySide's interpreter-shutdown cleanup."""
    global _QT_APPLICATION

    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication

    application = QApplication.instance()
    if application is None:
        return
    QApplication.closeAllWindows()
    QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
    application.processEvents()
    application.shutdown()
    _QT_APPLICATION = None

# 205P/Giacobini, split in 2008 (MPC astrometry): undivided comet in 1896 and
# early 2008 under the provisional designation, labelled components A/B/C
# measured on the same exposures, and unlabelled survey rows in 2015.
FIXTURE_205P = """\
0205PI96R020  M1896 09 04.84466 17 16 12.16 -07 36 37.1                pBA014020
0205PK08R060  C2008 09 10.56854 20 39 18.70 -01 17 39.9          13.5 T 63804D89
0205PK08R060  C2008 09 11.13677 20 40 03.41 -01 31 45.2          15.2 No63805H47
0205P      a  C2008 09 24.14478 21 00 11.57 -06 31 35.4          15.8 Nr64084850
0205P      b  C2008 09 24.14478 21 00 07.08 -06 31 43.5          19.6 Nr64084850
0205P      c  C2008 09 24.14478 20 59 28.13 -06 32 33.4          19.6 Nr64084850
0205P      a  C2008 09 26.10629 21 03 42.01 -07 11 27.2          15.8 Nr64084850
0205P      b  C2008 09 26.10629 21 03 37.43 -07 11 35.2          20.0 Nr64084850
0205P      c  C2008 09 26.16076 21 03 04.80 -07 13 22.8          19.8 Nr64084850
0205P      c AC2008 09 23.86209 20 58 58.29 -06 26 42.0                r64084J95
0205P      a  C2008 09 28.10243 21 07 22.93 -07 50 13.8          14.0 To64085673
0205P      a  C2008 09 28.15752 21 07 28.57 -07 51 16.4          15.7 No64085H47
0205P        KC2015 09 18.09005 04 45 54.29 +08 34 37.0          15.6 Nd95370A77
"""

# 73P/Schwassmann-Wachmann (MPC astrometry): parent comet in 1990, fragment A
# well observed but lost after the 1995 split, C followed through later
# returns, a fragment B, a new 2017 fragment BT, and the surviving nucleus
# reported without a letter in 2017.
FIXTURE_73P = """\
0073P         A1990 02 23.43201 15 05 01.82 +02 29 45.5                 16124657
0073P         A1990 02 23.50771 15 05 15.36 +02 29 20.0                 16124657
0073P      a  C1995 12 23.12177 22 04 58.78 -21 21 19.8                 26444693
0073P      a  C1995 12 23.12324 22 04 59.07 -21 21 17.8          15.9 N 26444693
0073P      a  C1995 12 23.12581 22 04 59.58 -21 21 14.6                 26444693
0073P      a  C1995 12 23.22982 22 05 19.53 -21 19 11.6                 26444568
0073P      c  C1996 01 02.02911 22 35 02.21 -18 07 11.2                 26444834
0073P      c  C1996 01 04.45225 22 41 51.36 -17 20 25.8                 26444413
0073P      c  C2000 03 01.86672 07 58 42.84 +33 14 54.1                 43446950
0073P      b  C2000 11 19.18509 14 07 30.00 -06 36 37.4                 41633118
0073P     bt  C2017 02 10.35516 18 05 12.88 -20 01 21.6          12.6 TU@3118W97
0073P        KC2017 01 06.84367 15 30 32.43 -11 35 23.4                u@2331367
0073P        KC2017 01 06.84651 15 30 33.09 -11 35 26.3          13.8 Tu@2331367
"""


class FragmentLabelTests(unittest.TestCase):
    def test_mpc80_columns(self):
        self.assertEqual(mpc80_fragment("0205P      b  C2008"), "B")
        self.assertEqual(mpc80_fragment("0073P     bk  C2006"), "BK")
        self.assertEqual(mpc80_fragment("    PK08R06c  C2008"), "C")
        self.assertEqual(mpc80_fragment("0205PK08R060  C2008"), "")
        self.assertEqual(mpc80_fragment("0205P        KC2015"), "")
        # Note 1 (column 14) is not a fragment letter.
        self.assertEqual(mpc80_fragment("0205P        1C2015"), "")

    def test_unpacked_designations(self):
        for text, expected in (
            ("205P-B", "B"), ("73P-BK", "BK"), ("D/1993 F2-G", "G"),
            ("2008 R6-c", "C"), ("C/2019 Y4-A", "A"),
            ("205P", ""), ("2024 BX1", ""), ("C/2019 Y4", ""),
        ):
            with self.subTest(text=text):
                self.assertEqual(designation_fragment(text), expected)

    def test_ades_rows_carry_the_fragment(self):
        suffixed = MpcClient._to_observation(  # noqa: SLF001
            {"permid": "205P-B", "obstime": "2008-09-24T03:28:29Z", "ra": 315.0,
             "dec": -6.5, "stn": "850"}, "205P",
        )
        self.assertEqual(observation_fragment(suffixed), "B")
        with_obs80 = MpcClient._to_observation(  # noqa: SLF001
            {"permID": "205P", "obs80": "0205P      c  C2008 09 24.14478",
             "obsTime": "2008-09-24T03:28:29Z", "ra": 315.0, "dec": -6.5, "stn": "850"},
            "205P",
        )
        self.assertEqual(observation_fragment(with_obs80), "C")


class FragmentSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = parse_mpc80(FIXTURE_205P)

    def test_counts_and_principal_component(self):
        self.assertEqual(fragment_counts(self.rows), {"A": 4, "B": 2, "C": 3})
        self.assertEqual(primary_fragment(self.rows), "A")

    def test_principal_component_takes_unlabelled_rows(self):
        self.assertEqual(select_fragment(self.rows, "A"), 8)
        fitted = [row for row in self.rows if is_fit_candidate(row)]
        self.assertEqual(len(fitted), 8)
        self.assertTrue(all(observation_fragment(row) in ("", "A") for row in fitted))

    def test_secondary_component_uses_only_its_own_rows(self):
        self.assertEqual(select_fragment(self.rows, "c"), 3)
        fitted = [row for row in self.rows if is_fit_candidate(row)]
        self.assertEqual({observation_fragment(row) for row in fitted}, {"C"})

    def test_clearing_the_selection_restores_all_rows(self):
        select_fragment(self.rows, "B")
        self.assertEqual(select_fragment(self.rows, None), len(self.rows))
        self.assertTrue(all(is_fit_candidate(row) for row in self.rows))

    def test_unlabelled_astrometry_is_never_filtered(self):
        rows = [row for row in self.rows if not observation_fragment(row)]
        self.assertEqual(select_fragment(rows, "A"), len(rows))
        self.assertTrue(all(is_fit_candidate(row) for row in rows))

    def test_manual_exclusion_is_preserved(self):
        self.rows[0].excluded_from_fit = True
        select_fragment(self.rows, "A")
        select_fragment(self.rows, None)
        self.assertTrue(self.rows[0].excluded_from_fit)

    def test_principal_component_without_a_label(self):
        rows = [row for row in self.rows if observation_fragment(row) != "A"]
        rows += parse_mpc80(
            "0205P      c  C2008 09 29.80626 21 09 54.14 -08 22 36.0          20.7 Nd64086A77\n"
        )
        self.assertEqual(primary_fragment(rows), "C")

    def test_surviving_component_is_not_necessarily_a(self):
        rows = parse_mpc80(FIXTURE_73P)
        counts = fragment_counts(rows)
        self.assertGreater(counts["A"], counts["C"])
        self.assertEqual(
            fragment_return_counts(rows), {"A": 1, "C": 2, "B": 1, "BT": 1},
        )
        self.assertEqual(primary_fragment(rows), "C")
        select_fragment(rows, "C")
        fitted = [row for row in rows if is_fit_candidate(row)]
        # Parent comet 1990, labelled C, and the unlabelled survivor in 2017.
        self.assertEqual(len(fitted), 7)
        self.assertEqual({observation_fragment(row) for row in fitted}, {"", "C"})

    def test_components_on_one_exposure_are_not_duplicates(self):
        rows = parse_mpc80(
            "0205P      a  C2008 09 24.14478 21 00 11.57 -06 31 35.4          15.8 Nr64084850\n"
            "0205P      b  C2008 09 24.14478 21 00 10.57 -06 31 43.5          19.6 Nr64084850\n"
        )
        self.assertEqual(mark_duplicate_measurements(rows), 0)
        unlabelled = parse_mpc80(
            "0205P         C2008 09 24.14478 21 00 11.57 -06 31 35.4          15.8 Nr64084850\n"
            "0205P         C2008 09 24.14478 21 00 10.57 -06 31 43.5          19.6 Nr64084850\n"
        )
        self.assertEqual(mark_duplicate_measurements(unlabelled), 1)


def _night(offsets, start=0.0, spacing=0.003, partner=None):
    """Rows of one I41 night with given O-C; ``partner`` adds a track 16 s later."""
    from datetime import datetime, timedelta, timezone

    from orbit_app.domain import Observation

    rows = []
    base = datetime(2021, 10, 30, 2, 0, tzinfo=timezone.utc)
    for index, (ra, dec) in enumerate(offsets):
        tracks = [(0.0, ra, dec)]
        if partner is not None:
            tracks.append((1.9e-4, partner[0] + 0.1 * index, partner[1]))
        for delay, oc_ra, oc_dec in tracks:
            instant = base + timedelta(days=start + index * spacing + delay)
            rows.append(Observation(
                time=instant.isoformat().replace("+00:00", "Z"), ra_deg=263.0, dec_deg=-12.0,
                station="I41", residual_ra_arcsec=oc_ra, residual_dec_arcsec=oc_dec,
                raw={"rmsRA": 0.3, "rmsDec": 0.3},
            ))
    return rows


class DoubleTrackTests(unittest.TestCase):
    """Unlabelled second track measured on the same exposures (205P, ZTF 2021)."""

    def setUp(self):
        from orbit_app.orbit_fit import _simultaneous_double_track_candidates

        self.detect = _simultaneous_double_track_candidates

    def test_secondary_track_is_found_even_when_the_orbit_is_dragged(self):
        rows = _night([(8.2, -0.7), (7.5, -0.1), (8.1, -0.2), (8.9, 0.0), (7.9, 0.0)],
                      partner=(-57.4, 5.3))
        found = self.detect(rows)
        self.assertEqual(len(found), 5)
        self.assertTrue(all(row.residual_ra_arcsec < -50 for row in found))

    def test_tracks_are_judged_against_the_other_stations(self):
        # A gravity-only orbit puts the whole return 35″ east; the second track
        # is 31″ west of it - equally far from the orbit, 65″ from everyone else.
        rows = _night([(35.0, -2.6)] * 5, partner=(-30.5, 2.8))
        for index, station in enumerate(("215", "A77", "Q62", "G96")):
            rows += [
                replace(row, station=station, raw=dict(row.raw))
                for row in _night([(34.0 + 0.3 * index, -2.5)] * 2, start=-3.0 - index)
            ]
        found = self.detect(rows)
        self.assertEqual(len(found), 5)
        self.assertTrue(all(row.residual_ra_arcsec < 0 for row in found))

    def test_comparable_offsets_are_not_decided(self):
        rows = _night([(20.0, 0.0)] * 4, partner=(-25.0, 0.0))
        self.assertEqual(self.detect(rows), [])

    def test_close_re_reductions_and_single_pairs_are_ignored(self):
        self.assertEqual(self.detect(_night([(0.5, 0.2)] * 5, partner=(3.0, 1.0))), [])
        self.assertEqual(self.detect(_night([(0.5, 0.2)] * 2, partner=(-60.0, 5.0))), [])

    def test_inactive_fragment_rows_carry_no_weight(self):
        rows = _night([(0.2, 0.1)] * 4, partner=(-69.0, 4.0))
        for row in rows[1::2]:
            row.raw["inactive_fragment"] = True
        self.assertEqual(self.detect(rows), [])


class SecondaryTrackBudgetTests(unittest.TestCase):
    def test_secondary_track_does_not_use_up_the_rejection_budget(self):
        # 205P in 2021: ten ZTF rows of a second track were excluded first and
        # then blocked rejecting three W62 rows 66″ off, so the return failed.
        from unittest import mock

        from orbit_app.domain import Observation
        from orbit_app.linkage import _Linker

        def row(day, offset):
            return Observation(
                time=f"2021-10-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0, station="I41",
                residual_ra_arcsec=offset, residual_dec_arcsec=0.1,
                raw={"rmsRA": 1.0, "rmsDec": 1.0},
            )

        nominal = [row(1 + index % 28, 0.3) for index in range(30)]
        secondary = [row(30, -66.0) for _ in range(8)]
        blunders = [row(14, 67.0) for _ in range(3)]
        group = nominal + secondary + blunders
        linker = _Linker(group, [group], None, None, None, None, None, lambda *a, **k: None, None)
        with mock.patch(
            "orbit_app.linkage._coherent_secondary_track_candidates", return_value=secondary,
        ):
            linker.reject_outliers([0])
        self.assertTrue(all(item.raw.get("coherent_secondary_track") for item in secondary))
        self.assertTrue(all(item.excluded_from_fit for item in blunders))
        self.assertFalse(any(item.excluded_from_fit for item in nominal))


class FragmentSelectorWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global _QT_APPLICATION, _QT_SHUTDOWN_REGISTERED

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        _QT_APPLICATION = QApplication.instance() or QApplication([])
        if not _QT_SHUTDOWN_REGISTERED:
            atexit.register(_shutdown_qt_application)
            _QT_SHUTDOWN_REGISTERED = True
        cls._application = _QT_APPLICATION

    def setUp(self):
        from unittest import mock

        from orbit_app.main_window import MainWindow

        self.window = MainWindow()
        self.window.mpc_network.observations = mock.Mock()

    def tearDown(self):
        from PySide6.QtCore import QEvent
        from PySide6.QtWidgets import QApplication

        self.window.close()
        self.window.deleteLater()
        QApplication.sendPostedEvents(None, QEvent.DeferredDelete)
        self._application.processEvents()

    def test_split_comet_offers_components_and_fits_the_principal_one(self):
        rows = parse_mpc80(FIXTURE_205P)
        self.window._observations_loaded("205P", rows)  # noqa: SLF001
        selector = self.window.fragment_selector
        self.assertFalse(selector.isHidden())
        self.assertEqual([selector.itemData(i) for i in range(selector.count())], ["A", "B", "C"])
        self.assertEqual(selector.currentData(), "A")
        self.assertIn("8 z 13", self.window.fit_button.text())

        selector.setCurrentIndex(1)
        self.assertIn("2 z 13", self.window.fit_button.text())
        self.assertIn("fragment B", self.window.statusBar().currentMessage())
        column = [name for _label, name in self.window.observation_model.columns].index("fragment")
        index = self.window.observation_model.index(4, column)
        self.assertEqual(self.window.observation_model.data(index), "B")

    def test_surviving_component_is_preselected(self):
        rows = parse_mpc80(FIXTURE_73P)
        self.window._observations_loaded("73P", rows)  # noqa: SLF001
        selector = self.window.fragment_selector
        self.assertEqual(
            [selector.itemData(i) for i in range(selector.count())], ["A", "B", "C", "BT"],
        )
        self.assertEqual(selector.currentData(), "C")
        self.assertIn("7 z 13", self.window.fit_button.text())

    def test_selector_is_hidden_for_an_undivided_object(self):
        rows = parse_mpc80(FIXTURE_205P.splitlines()[0] + "\n")
        self.window._observations_loaded("205P", rows)  # noqa: SLF001
        self.assertTrue(self.window.fragment_selector.isHidden())
        self.assertTrue(all(is_fit_candidate(row) for row in rows))


if __name__ == "__main__":
    unittest.main()
