"""Split comets: component overview, row assignment, track split-off, split epoch."""
import os
import unittest
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from unittest import mock

import numpy as np

os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

from orbit_app import fragments, orbit_fit  # noqa: E402
from orbit_app.astrometry import (  # noqa: E402
    ASSIGNED_FRAGMENT_KEY, MAIN_UNLABELLED, fragment_components, observation_fragment,
    primary_fragment, select_fragment,
)
from orbit_app.domain import Observation, OrbitSolution  # noqa: E402
from orbit_app.fragment_ui import fragment_banner_text  # noqa: E402
from orbit_app.fragments import (  # noqa: E402
    AMBIGUOUS_FRAGMENT_KEY, SPLIT_OFF_KEY, assign_unlabelled_rows, component_summaries,
    describe_split, estimate_split, secondary_tracks, split_off_track,
)
from orbit_app.mpc import parse_mpc80  # noqa: E402

try:
    from qt_support import dispose, qt_application  # noqa: E402
except ImportError:  # run as tests.<module> from the repository root
    from tests.qt_support import dispose, qt_application  # noqa: E402

# 205P (MPC astrometry): A/B/C on the same exposures in 2008 and unlabelled
# rows before and after.
FIXTURE = """\
0205PK08R060  C2008 09 10.56854 20 39 18.70 -01 17 39.9          13.5 T 63804D89
0205P      a  C2008 09 24.14478 21 00 11.57 -06 31 35.4          15.8 Nr64084850
0205P      b  C2008 09 24.14478 21 00 07.08 -06 31 43.5          19.6 Nr64084850
0205P      c  C2008 09 24.14478 20 59 28.13 -06 32 33.4          19.6 Nr64084850
0205P      a  C2008 09 26.10629 21 03 42.01 -07 11 27.2          15.8 Nr64084850
0205P      b  C2008 09 26.10629 21 03 37.43 -07 11 35.2          20.0 Nr64084850
0205P      a  C2008 09 28.10243 21 07 22.93 -07 50 13.8          14.0 To64085673
0205P        KC2008 09 26.10900 21 03 37.50 -07 11 35.0          19.9 Nd95370A77
0205P        KC2015 09 18.09005 04 45 54.29 +08 34 37.0          15.6 Nd95370A77
"""


def _rows():
    return parse_mpc80(FIXTURE, "205P")


def _fake_solution(name: str) -> OrbitSolution:
    return OrbitSolution(
        designation=name, epoch_jd=2454730.5, semi_major_axis_au=3.5,
        eccentricity=.3, inclination_deg=15.0, node_deg=0.0, arg_peri_deg=0.0,
        mean_anomaly_deg=0.0, position_au=(1.5, 0.0, 0.0),
        velocity_au_per_day=(0.0, .015, 0.0),
    )


class ComponentSummaryTests(unittest.TestCase):
    def test_principal_first_with_same_exposure_offsets(self):
        summaries = component_summaries(_rows())
        self.assertEqual([item.component for item in summaries], ["A", "B", "C"])
        principal, second, third = summaries
        self.assertTrue(principal.principal)
        # Unlabelled rows belong to the principal component.
        self.assertEqual(principal.rows, 6)
        self.assertEqual(principal.labelled, 3)
        self.assertIsNone(principal.offset_arcsec)
        # B lies ~67″ west of A on both shared exposures.
        self.assertEqual(second.offset_pairs, 2)
        self.assertAlmostEqual(second.offset_arcsec, 67.4, delta=1.5)
        self.assertGreater(third.offset_arcsec, 600.0)

    def test_undivided_object_has_no_summary(self):
        rows = [row for row in _rows() if not observation_fragment(row)]
        self.assertEqual(component_summaries(rows), [])

    def test_banner_text_lists_components_and_fitted_orbits(self):
        summaries = component_summaries(_rows())
        text = fragment_banner_text(summaries, {"A": _fake_solution("A")})
        self.assertIn("(3)", text)
        self.assertIn("A: 6 měření", text)
        self.assertIn("(hlavní složka) ✓", text)
        self.assertIn("od hlavní složky", text)
        self.assertEqual(fragment_banner_text([], {}), "")


class AssignmentTests(unittest.TestCase):
    """Unlabelled post-split rows go to the orbit they follow."""

    @staticmethod
    def _evaluate(residuals: dict[str, dict[str, float]]):
        # O-C by solution name and observation time: stands in for the propagation.
        def evaluate(rows, solution):
            for row in rows:
                value = residuals[solution.designation].get(row.time[:10], 50.0)
                row.residual_ra_arcsec = value
                row.residual_dec_arcsec = 0.0
        return evaluate

    def test_row_following_the_secondary_orbit_is_assigned(self):
        rows = _rows()
        solutions = {"A": _fake_solution("A"), "B": _fake_solution("B")}
        report = assign_unlabelled_rows(rows, solutions, self._evaluate({
            "A": {"2008-09-26": 40.0}, "B": {"2008-09-26": .5},
        }))
        unlabelled = [
            row for row in rows
            if row.time.startswith("2008-09-26") and not row.raw.get("mpc_fragment")
            and row.station == "A77"
        ]
        self.assertEqual(report.assigned, {"B": 1})
        self.assertEqual(report.compared, 2)  # 2015 lies outside the 30-day window
        self.assertEqual(unlabelled[0].raw[ASSIGNED_FRAGMENT_KEY], "B")
        self.assertEqual(select_fragment(rows, "B"), 3)
        # Re-running with different orbits recomputes the assignment.
        report = assign_unlabelled_rows(rows, solutions, self._evaluate({
            "A": {"2008-09-26": .6}, "B": {"2008-09-26": .5},
        }))
        self.assertEqual(report.assigned, {})
        self.assertEqual(report.unresolved, 1)
        self.assertNotIn(ASSIGNED_FRAGMENT_KEY, unlabelled[0].raw)
        self.assertTrue(unlabelled[0].raw[AMBIGUOUS_FRAGMENT_KEY])

    def test_nothing_is_compared_without_a_fitted_secondary(self):
        rows = _rows()
        report = assign_unlabelled_rows(rows, {"A": _fake_solution("A")}, self._evaluate({}))
        self.assertEqual((report.assigned, report.compared), ({}, 0))


def _ztf_like_rows():
    """Main nucleus by several stations; I41 also measured a second track."""
    start = datetime(2021, 10, 30, 2, 0, tzinfo=timezone.utc)
    rows = []
    for night in range(3):
        for index in range(3):
            time = start + timedelta(days=night, minutes=10 * index)
            for station, offset in (("I41", 0.0), ("I41", 65.0 / 3600.0), ("T05", 0.0)):
                rows.append(Observation(
                    designation="205P", time=time.isoformat().replace("+00:00", "Z"),
                    ra_deg=40.0 + night * .3 + index * .002 - offset,
                    dec_deg=10.0, station=station,
                    raw={"coherent_secondary_track": True, "automatic_exclusion": True}
                    if offset else {},
                    excluded_from_fit=bool(offset),
                ))
    return rows


class SecondaryTrackTests(unittest.TestCase):
    def test_detected_track_becomes_a_selectable_component(self):
        rows = _ztf_like_rows()
        tracks = secondary_tracks(rows)
        self.assertEqual(len(tracks), 1)
        self.assertEqual(len(tracks[0].rows), 9)
        self.assertEqual(tracks[0].stations, ("I41",))
        label = split_off_track(rows, tracks[0])
        self.assertEqual(label, "X1")
        self.assertEqual(primary_fragment(rows), MAIN_UNLABELLED)
        self.assertEqual(fragment_components(rows), {MAIN_UNLABELLED: 18, "X1": 9})
        self.assertEqual(secondary_tracks(rows), [])
        track_rows = [row for row in rows if row.raw.get(SPLIT_OFF_KEY)]
        self.assertTrue(all(not row.excluded_from_fit for row in track_rows))
        self.assertEqual(select_fragment(rows, "X1"), 9)
        # Assignment keeps a split-off track with its component.
        assign_unlabelled_rows(rows, {}, lambda *_: None)
        self.assertEqual(observation_fragment(track_rows[0]), "X1")


class SecondaryTrackRecheckTests(unittest.TestCase):
    def test_recheck_respects_the_secondary_track_budget(self):
        rows = _ztf_like_rows()   # 9 of 27 rows already form a track
        extra = rows[0]
        extra.excluded_from_fit = True
        extra.raw = {"automatic_exclusion": True}
        with mock.patch.object(
            orbit_fit, "_simultaneous_double_track_candidates", return_value=[extra],
        ):
            result = orbit_fit._reclassify_secondary_tracks(  # noqa: SLF001
                rows, ("old", 2.0, 10), {}, None, {}, False,
                lambda *a, **k: self.fail("unexpected refit"), lambda *_: None,
            )
        self.assertEqual(result[0], "old")
        self.assertNotIn("coherent_secondary_track", extra.raw)
        self.assertTrue(extra.excluded_from_fit)

    def test_rows_rejected_as_isolated_blunders_rejoin_their_track(self):
        rows = _ztf_like_rows()
        for row in rows:
            row.raw.pop("coherent_secondary_track", None)
        isolated, weighted = rows[1], rows[4]
        isolated.excluded_from_fit = True
        isolated.raw = {"automatic_exclusion": True, "automatic_exclusion_reason": "x"}
        weighted.excluded_from_fit = False
        weighted.raw = {}
        stray = rows[0]
        stray.excluded_from_fit = True
        stray.raw = {"automatic_exclusion": True}
        refits = []

        def fit_once(*arguments, **keywords):
            refits.append(keywords)
            return ("refit", 1.0, 10)

        with mock.patch.object(
            orbit_fit, "_simultaneous_double_track_candidates",
            return_value=[isolated, weighted],
        ):
            result = orbit_fit._reclassify_secondary_tracks(  # noqa: SLF001
                rows, ("old", 2.0, 10), {}, None, {}, False, fit_once, lambda *_: None,
            )
        self.assertEqual(result[0], "refit")
        self.assertEqual(len(refits), 1)
        for row in (isolated, weighted):
            self.assertTrue(row.excluded_from_fit)
            self.assertTrue(row.raw["coherent_secondary_track"])
        # Other isolated rejections stay rejected and keep their reason.
        self.assertTrue(stray.excluded_from_fit)
        self.assertNotIn("coherent_secondary_track", stray.raw)

    def test_no_refit_when_only_rejected_rows_are_relabelled(self):
        rows = _ztf_like_rows()
        for row in rows:
            row.raw.pop("coherent_secondary_track", None)
        isolated = rows[1]
        isolated.raw = {"automatic_exclusion": True}
        isolated.excluded_from_fit = True
        with mock.patch.object(
            orbit_fit, "_simultaneous_double_track_candidates", return_value=[isolated],
        ):
            result = orbit_fit._reclassify_secondary_tracks(  # noqa: SLF001
                rows, ("old", 2.0, 10), {}, None, {}, False,
                lambda *a, **k: self.fail("unexpected refit"), lambda *_: None,
            )
        self.assertEqual(result[0], "old")
        self.assertTrue(isolated.raw["coherent_secondary_track"])


class SplitEpochTests(unittest.TestCase):
    """A synthetic fragment released with a known impulse is dated correctly."""

    def test_early_edge_and_extreme_a1_are_implausible(self):
        self.assertTrue(fragments._split_epoch_at_window_edge(100.5, 100.0, 1.0))  # noqa: SLF001
        self.assertFalse(fragments._split_epoch_at_window_edge(150.0, 100.0, 1.0))  # noqa: SLF001
        # The late end is the first fragment measurement: a physical limit.
        self.assertFalse(fragments._split_epoch_at_window_edge(199.5, 100.0, 1.0))  # noqa: SLF001
        self.assertTrue(fragments._split_epoch_before_first_observation(  # noqa: SLF001
            199.5, 200.0, 1.0,
        ))
        fresh = fragments.SplitModel(
            "impulse", 199.5, None, (1.0, 0.0, 0.0), None,
            .2, 10.0, 4, False, 2.0, before_first_observation=True,
        )
        self.assertTrue(fresh.plausible)
        self.assertIn("čerstvý", describe_split(
            fragments.SplitEstimate("P", "F", (fresh,), 20, 100.0),
        ))
        extreme = fragments.SplitModel(
            "deceleration", 150.0, None, (0.0, 0.0, 0.0),
            2.0 * fragments.PLAUSIBLE_DIFFERENTIAL_A1_AU_PER_DAY2,
            .1, 1.0, 2, False, 2.0,
        )
        impulse = fragments.SplitModel(
            "impulse", 150.0, None, (1.0, 0.0, 0.0), None,
            .2, 10.0, 4, False, 2.0,
        )
        estimate = fragments.SplitEstimate("P", "F", (extreme, impulse), 20, 100.0)
        self.assertFalse(extreme.plausible)
        self.assertIs(estimate.preferred, impulse)
        self.assertIn("|ΔA1|", describe_split(estimate))

    def test_fresh_split_just_before_discovery_is_kept(self):
        estimate, primary, _rows, delta_v = _synthetic_split(days_before=5.0, lookback=400.0)
        impulse = next(model for model in estimate.models if model.name == "impulse")
        first = primary.epoch_jd + 5.0
        self.assertAlmostEqual(first - impulse.epoch_jd, 5.0, delta=2.5)
        self.assertFalse(impulse.at_window_edge)
        self.assertTrue(impulse.plausible)

    def test_impulse_split_is_recovered(self):
        estimate, primary, _rows, delta_v = _synthetic_split(days_before=None, lookback=300.0)
        impulse = next(model for model in estimate.models if model.name == "impulse")
        self.assertLess(impulse.rms_arcsec, .2)
        self.assertAlmostEqual(impulse.epoch_jd, primary.epoch_jd, delta=8.0)
        self.assertAlmostEqual(impulse.speed_ms, float(np.linalg.norm(delta_v)), delta=.4)
        self.assertIn("Odhad rozpadu: fragment F od složky P", describe_split(estimate))


def _synthetic_split(days_before: float | None, lookback: float):
    """Fragment released from ``primary`` at its epoch with a known Δv.

    ``days_before`` places that epoch before the first fragment measurement
    (default: the fixed epoch 2459400.5, ~105 days before).
    """
    start = datetime(2021, 11, 1, 3, 0, tzinfo=timezone.utc)
    first_jd = 2459519.625  # 2021-11-01 03:00 UTC
    epoch = 2459400.5 if days_before is None else first_jd - days_before
    primary = OrbitSolution(
        designation="P", epoch_jd=epoch, semi_major_axis_au=3.0,
        eccentricity=.4, inclination_deg=10.0, node_deg=0.0, arg_peri_deg=0.0,
        mean_anomaly_deg=0.0, position_au=(1.9, .4, .1),
        velocity_au_per_day=(-.003, .0125, .0012),
    )
    state = np.concatenate((primary.position_au, primary.velocity_au_per_day))
    basis = fragments._rtn_basis(state)  # noqa: SLF001
    delta_v = np.array([1.0, -1.5, .8])   # m/s, R/T/N
    velocity = np.asarray(primary.velocity_au_per_day) + basis.T @ (
        delta_v / fragments.AU_PER_DAY_TO_M_PER_S
    )
    fragment = replace(primary, velocity_au_per_day=tuple(velocity))
    rows = [
        Observation(
            designation="P",
            time=(start + timedelta(days=day, hours=hour)).isoformat().replace("+00:00", "Z"),
            ra_deg=0.0, dec_deg=0.0, station="500",
        )
        for day in range(0, 40, 4) for hour in (0, 3)
    ]
    for _iteration in range(3):  # observed = computed for the fragment orbit
        orbit_fit._evaluate_solution_observations(rows, fragment)  # noqa: SLF001
        for row in rows:
            row.dec_deg -= row.residual_dec_arcsec / 3600.0
            row.ra_deg -= row.residual_ra_arcsec / 3600.0 / np.cos(np.radians(row.dec_deg))
    with mock.patch.multiple(
        fragments, SPLIT_COARSE_POINTS=80, SPLIT_EXACT_EVALUATIONS=15,
    ):
        estimate = estimate_split(
            primary, rows, primary_label_text="P", fragment_label_text="F",
            lookback_days=lookback,
        )
    return estimate, primary, rows, delta_v


class FragmentBannerWindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = qt_application()

    def setUp(self):
        from orbit_app.main_window import MainWindow

        self.window = MainWindow()

    def tearDown(self):
        dispose(self.window)

    def test_banner_announces_fragments_and_offers_split_off(self):
        window = self.window
        window._observations_loaded("205P", _rows())  # noqa: SLF001
        self.assertFalse(window.fragment_banner.isHidden())
        self.assertIn("složek rozpadlé komety", window.fragment_banner.text.text())
        self.assertTrue(window.fragment_banner.split_button.isHidden())
        window._observations_loaded("205P", _ztf_like_rows())  # noqa: SLF001
        self.assertTrue(window.fragment_selector.isHidden())
        self.assertFalse(window.fragment_banner.split_off_button.isHidden())
        window.component_solutions[MAIN_UNLABELLED] = _fake_solution("205P")
        window._split_off_secondary_track()  # noqa: SLF001
        self.assertFalse(window.fragment_selector.isHidden())
        self.assertEqual(window._current_component(), MAIN_UNLABELLED)  # noqa: SLF001
        self.assertGreaterEqual(window.fragment_selector.findData("X1"), 0)
        self.assertTrue(window.fragment_banner.split_off_button.isHidden())

    def test_reloading_the_same_object_keeps_split_off_tracks(self):
        window = self.window
        window._observations_loaded("205P", _ztf_like_rows())  # noqa: SLF001
        window.component_solutions[MAIN_UNLABELLED] = _fake_solution("205P")
        window._split_off_secondary_track()  # noqa: SLF001
        window.component_solutions["X1"] = _fake_solution("X1")
        window.component_solutions["Q"] = _fake_solution("Q")   # no rows any more
        fresh = _ztf_like_rows()
        for row in fresh:   # as downloaded again: no flags from the fit
            row.raw = {}
            row.excluded_from_fit = False
        window._observations_loaded("205P", fresh)  # noqa: SLF001
        self.assertEqual(fragment_components(fresh), {MAIN_UNLABELLED: 18, "X1": 9})
        self.assertGreaterEqual(window.fragment_selector.findData("X1"), 0)
        self.assertEqual(set(window.component_solutions), {MAIN_UNLABELLED, "X1"})

    def test_other_fitted_components_are_companions_in_the_views(self):
        window = self.window
        window._observations_loaded("205P", _rows())  # noqa: SLF001
        window.component_solutions = {"A": _fake_solution("A"), "B": _fake_solution("B")}
        window._apply_solution(window.component_solutions["A"])  # noqa: SLF001
        window._update_companions()  # noqa: SLF001
        self.assertEqual(list(window.sky_panel.companions), ["205P-B"])
        self.assertEqual(list(window.solar_panel.companions), ["205P-B"])
        self.assertEqual(window.sky_panel.field_of_view.currentData(), 60.0)
        self.assertGreaterEqual(window.sky_panel.field_of_view.findData(0.25), 0)
        # Switching components on an open Sky tab redraws it once, with the
        # companions of the newly shown orbit.
        for index in range(window.tabs.count()):
            window.tabs.setTabEnabled(index, True)
        window.tabs.setCurrentWidget(window.sky_panel)
        drawn = []
        with mock.patch.object(
            window.sky_panel, "rebuild",
            side_effect=lambda *_: drawn.append(sorted(window.sky_panel.companions)),
        ):
            window.fragment_selector.setCurrentIndex(window.fragment_selector.findData("B"))
        self.assertEqual(drawn, [["205P-A"]])
        # A new object forgets the components of the previous one.
        window._observations_loaded("141P", _rows()[:1])  # noqa: SLF001
        self.assertEqual(window.component_solutions, {})
        self.assertEqual(window.sky_panel.companions, {})


if __name__ == "__main__":
    unittest.main()
