import math
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np

from orbit_app.domain import Observation, OrbitSolution
from orbit_app.main_window import (
    CONSTELLATION_LABELS, CONSTELLATION_LINES, ElementsPanel, MpcNetworkClient, ObservationModel,
    PlaybackBar, SkyPanel, SolarSystemPanel,
    apparition_linkage_summary, apparition_observation_count_text,
    datetime_from_jd, equatorial_ra_ticks, format_dec_sexagesimal, format_ra_sexagesimal,
    is_comet_designation, observation_residual_score, observations_to_tsv, parse_output_epoch,
    perihelion_jd_from_solution, project_ecliptic_camera,
    SKY_TRACK_HORIZONS, sky_track_sample_count, solar_orbit_clip_radius,
)
from orbit_app.mpc import MpcClient, parse_mpc80
from orbit_app.mpc_orbit import (
    cartesian_to_elements, mpc_display_name, mpc_identifier_choices, parse_mpc_orbit,
)
from orbit_app.observatories import (
    equatorial_to_horizontal, observation_altaz_series, observatory_details,
    station_offsets_gcrs,
)
from orbit_app.orbit_engine import (
    MU_SUN, NBodyIntegrator, StateVector, delayed_heliocentric_radius,
    marsden_water_ice_law,
    nongravitational_acceleration,
)
from orbit_app.orbit_evolution import (
    EvolutionResult, EvolutionReturn, EvolutionSnapshot, PLANET_MU, PlanetEncounter,
    centered_snapshot, evolution_span, format_evolution_mpc_report,
    format_evolution_report, perihelion_returns_between,
)
from orbit_app.planetary import PLANETS
from orbit_app.orbit_fit import (
    OrbitFitCancelled, _automatic_orbit_score, _InterpolatedPlanetEphemeris,
    _attach_multi_apparition_elements, _calibrate_fitted_photometry,
    _is_automatic_rejection_candidate,
    _local_light_time_states,
    _observation_offsets_gcrs, _observation_sigma_arcsec,
    _perihelion_offset_days_from_state,
    _latest_observing_apparition, _prefer_parabolic_short_arc,
    _representative_indices, _truncate_rows_to_recent_years,
    fit_orbit, fit_selection_count, parse_observation_time,
)
from orbit_app.photometry import estimate_absolute_magnitude, power_law_magnitude
from orbit_app.preview_orbit import (
    apparent_geometry, datetime_to_jd, demo_solution, heliocentric_xyz,
    orbital_curve_xyz, solve_kepler,
)


class PhotometryTests(unittest.TestCase):
    def test_unit_distances_equal_absolute_magnitude(self):
        result = power_law_magnitude(12.3, np.array([1.0]), np.array([1.0]), 4.0)
        self.assertAlmostEqual(float(result[0]), 12.3)

    def test_comet_exponent_convention(self):
        # n=4 corresponds to K1=10: doubling r adds 10 log10(2) mag.
        result = power_law_magnitude(10.0, np.array([1.0]), np.array([2.0]), 4.0)
        self.assertAlmostEqual(float(result[0]), 10.0 + 10.0 * math.log10(2.0))

    def test_apparent_measurements_are_reduced_before_becoming_comet_m(self):
        delta = np.array([8.0, 8.2, 8.5])
        radius = np.array([8.7, 8.8, 8.9])
        apparent = power_law_magnitude(6.25, delta, radius, 4.0)
        self.assertAlmostEqual(
            estimate_absolute_magnitude(apparent, delta, radius, 4.0), 6.25
        )


class SkyTrackTests(unittest.TestCase):
    def test_standard_playback_has_intermediate_day_speeds(self):
        self.assertIn(("1 den/s", 1), PlaybackBar.STANDARD_SPEEDS)
        self.assertIn(("2 dny/s", 2), PlaybackBar.STANDARD_SPEEDS)
        self.assertIn(("5 dní/s", 5), PlaybackBar.STANDARD_SPEEDS)
        self.assertIn(("250 dní/s", 250), PlaybackBar.STANDARD_SPEEDS)
        self.assertIn(("500 dní/s", 500), PlaybackBar.STANDARD_SPEEDS)
        self.assertLess(
            PlaybackBar.STANDARD_SPEEDS.index(("2 dny/s", 2)),
            PlaybackBar.STANDARD_SPEEDS.index(("5 dní/s", 5)),
        )

    def test_equatorial_ra_ticks_are_unique_and_anchored_to_the_sky(self):
        positions, labels = equatorial_ra_ticks(349.6, 60.0)
        self.assertEqual(labels, ("22h", "23h", "00h", "01h"))
        self.assertEqual(len(labels), len(set(labels)))
        moved_positions, moved_labels = equatorial_ra_ticks(350.6, 60.0)
        self.assertEqual(moved_labels, labels)
        np.testing.assert_allclose(moved_positions, positions - 1.0)

    def test_equatorial_projection_uses_absolute_dec_and_wraps_ra(self):
        x, y, _ = SkyPanel._project_equatorial(
            np.array([359.0, 1.0, 10.0]), np.array([-5.0, 12.0, 25.0]), 0.0, 10.0
        )
        np.testing.assert_allclose(x, [-1.0, 1.0, 10.0])
        np.testing.assert_allclose(y, [-5.0, 12.0, 25.0])

    def test_constellation_strokes_keep_connections_across_ra_zero(self):
        adjacent_pairs = {
            tuple(np.round(np.asarray(pair).ravel(), 3))
            for line in CONSTELLATION_LINES
            for pair in zip(line[:-1], line[1:])
        }
        # Pegasus: Markab -> Alpheratz, and Algenib -> Scheat.
        self.assertIn((345.944, 28.083, 2.097, 29.09), adjacent_pairs)
        self.assertIn((3.309, 15.184, 346.19, 15.205), adjacent_pairs)
        # Pisces also crosses the zero-hour meridian.
        self.assertIn((12.171, 7.585, 359.828, 6.863), adjacent_pairs)

    def test_constellation_abbreviations_are_available(self):
        labels = {item[0] for item in CONSTELLATION_LABELS}
        self.assertTrue({"Sgr", "Leo", "Sct", "Peg", "Psc"} <= labels)

    def test_fast_horizontal_projection_matches_astropy_altaz(self):
        instant = datetime(2026, 10, 10, 21, 0, tzinfo=timezone.utc)
        ra = np.array([0.0, 120.0, 275.0])
        dec = np.array([20.0, -12.0, 45.0])
        fast = equatorial_to_horizontal("557", instant, ra, dec)
        reference = observation_altaz_series("557", [instant] * 3, ra, dec)
        self.assertIsNotNone(fast)
        self.assertIsNotNone(reference)
        np.testing.assert_allclose(fast[0], reference[0], atol=.01)
        np.testing.assert_allclose(fast[1], reference[1], atol=.01)

    def test_long_horizons_are_available_but_bounded(self):
        self.assertIn(("± 600 dní", 600), SKY_TRACK_HORIZONS)
        self.assertIn(("± 2,5 roku", 913), SKY_TRACK_HORIZONS)
        self.assertEqual(SKY_TRACK_HORIZONS[-1], ("± 10 let", 3652))

    def test_sampling_stays_fine_for_short_tracks_and_capped_for_long_ones(self):
        self.assertEqual(sky_track_sample_count(30), 241)
        self.assertEqual(sky_track_sample_count(300), 1501)
        self.assertEqual(sky_track_sample_count(600), 2401)
        self.assertEqual(sky_track_sample_count(3652), 2401)


class ObservationResidualTests(unittest.TestCase):
    def test_observation_clipboard_uses_sexagesimal_coordinates_and_residuals(self):
        observation = Observation(
            time="2026-10-11T01:02:03Z", ra_deg=272.9875, dec_deg=-17.125,
            magnitude=12.3, band="G", station="557",
            residual_ra_arcsec=-.125, residual_dec_arcsec=.375,
            used_in_fit=True,
        )
        text = observations_to_tsv([observation])
        self.assertIn("18 11 57.00", text)
        self.assertIn("−17 07 30.0", text)
        self.assertIn("-0.125\t+0.375", text)

        model = ObservationModel()
        model.set_rows([observation])
        self.assertEqual(model.data(model.index(0, 1)), "18 11 57.00")
        self.assertEqual(model.data(model.index(0, 2)), "−17 07 30.0")

    def test_score_matches_row_colour_rule(self):
        observation = Observation(
            residual_ra_arcsec=-1.25, residual_dec_arcsec=2.4,
        )
        self.assertEqual(observation_residual_score(observation), 2.4)

    def test_score_requires_both_coordinate_residuals(self):
        observation = Observation(residual_ra_arcsec=3.0)
        self.assertIsNone(observation_residual_score(observation))

    def test_output_epoch_accepts_nineteenth_century_and_year_only(self):
        self.assertEqual(parse_output_epoch("1900"), datetime(1900, 1, 1))
        self.assertEqual(
            parse_output_epoch("1800-06-15 12:30"), datetime(1800, 6, 15, 12, 30)
        )

    def test_numbered_periodic_comet_uses_comet_mpc_elements(self):
        solution = OrbitSolution(
            "2P", 2462500.5, 2.2178179, .8472983, 11.34795,
            334.01955, 187.28396, 286.52227,
            absolute_magnitude=14.36, photometric_exponent=4.0,
        )

        class ReportFixture:
            display_solution = solution
            _packed_epoch = staticmethod(ElementsPanel._packed_epoch)

            @staticmethod
            def _perihelion_jd():
                return perihelion_jd_from_solution(solution)

        line = ElementsPanel._mpc_line(ReportFixture())
        self.assertTrue(is_comet_designation("2P"))
        self.assertRegex(line, r"^\s*2P\s+20\d{2} \d{2} \d{2}\.\d{4}")
        self.assertIn(" 0.338", line)

    def test_defunct_d_comet_designation_is_recognized(self):
        self.assertTrue(is_comet_designation("D/1993 F2"))
        self.assertTrue(is_comet_designation("D/1993 F2-A"))

    def test_asteroid_keeps_mpcorb_epoch_and_mean_anomaly(self):
        solution = OrbitSolution(
            "9841", 2461118.5, 2.18882215, .1600963, 21.9873,
            138.52648, 264.66660, 185.59746671,
            absolute_magnitude=15.0, slope_g=.15,
        )

        class ReportFixture:
            display_solution = solution
            _packed_epoch = staticmethod(ElementsPanel._packed_epoch)

        line = ElementsPanel._mpc_line(ReportFixture())
        self.assertFalse(is_comet_designation("9841"))
        self.assertRegex(line, r"^9841\s+15\.00\s+0\.15\s+K263J\s+185\.59747")


class MpcParsingTests(unittest.TestCase):
    def test_sl9_fragment_ids_skip_unused_i_and_o_letters(self):
        self.assertEqual(MpcNetworkClient._sl9_spk_id("D/1993 F2-A"), 1000181)
        self.assertEqual(MpcNetworkClient._sl9_spk_id("D/1993 F2-J"), 1000189)
        self.assertEqual(MpcNetworkClient._sl9_spk_id("D/1993 F2-W"), 1000201)

    def test_horizons_url_percent_encodes_required_quotes(self):
        encoded = bytes(MpcNetworkClient._horizons_url((
            ("COMMAND", "'DES=1000187;'"),
        )).toEncoded())
        self.assertIn(b"COMMAND=%27DES%3D1000187%3B%27", encoded)

    def test_sl9_horizons_parser_uses_requested_ecliptic_ephemeris_block(self):
        result = """
Target body name: Shoemaker-Levy 9-A
 Initial ICRF cartesian coordinates:
 X = 9.900000000E+01 Y = 9.900000000E+01 Z = 9.900000000E+01
 VX= 9.900000000E+01 VY= 9.900000000E+01 VZ= 9.900000000E+01
$$SOE
 2449480.500000000 = A.D. 1994-May-08 00:00:00.0000 TDB
 X = 5.000000000E+00 Y = 1.000000000E-01 Z = -2.000000000E-02
 VX= -1.000000000E-04 VY= 7.000000000E-03 VZ= 1.000000000E-04
$$EOE
"""
        solution = MpcNetworkClient._parse_sl9_horizons_vector(
            result, "D/1993 F2-A", 2449480.5,
        )
        self.assertEqual(solution.data_source, "JPL Horizons")
        self.assertEqual(solution.display_name, "Shoemaker-Levy 9-A")
        np.testing.assert_allclose(solution.position_au, [5.0, .1, -.02])
        np.testing.assert_allclose(solution.velocity_au_per_day, [-.0001, .007, .0001])

        parent = MpcNetworkClient._parse_sl9_horizons_vector(
            result, "D/1993 F2", 2449480.5, representative_parent=True,
        )
        self.assertEqual(parent.display_name, "D/1993 F2 (Shoemaker–Levy 9)")
        self.assertIn("fragmentu G", parent.data_source)

    def test_ec0001_reference_file_has_24_observations(self):
        path = Path(__file__).with_name("data") / "ec0001.obs"
        observations = parse_mpc80(path.read_text(encoding="utf-8"))
        self.assertEqual(len(observations), 24)
        self.assertEqual(observations[0].station, "273")
        self.assertIn("observer_position_gcrs_au", observations[0].raw)

    def test_ades_row_mapping(self):
        body = [{"ADES_DF": [{
            "provID": "2026 AB", "obsTime": "2026-01-01T00:00:00Z",
            "ra": "12.5", "dec": "-2.25", "mag": "20.1", "band": "G", "stn": "F51"
        }]}]
        rows = MpcClient._extract_rows(body)
        observation = MpcClient._to_observation(rows[0], "fallback")
        self.assertEqual(observation.designation, "2026 AB")
        self.assertEqual(observation.station, "F51")
        self.assertAlmostEqual(observation.dec_deg or 0.0, -2.25)

    def test_ades_dict_response_mapping(self):
        body = {"ADES_DF": [{"provID": "2026 AB", "ra": "1", "dec": "2"}]}
        self.assertEqual(len(MpcClient._extract_rows(body)), 1)

    def test_ades_spacecraft_position_is_preserved(self):
        row = {
            "trkSub": "ec0001", "obsTime": "2026-09-16T13:34:48Z",
            "ra": "272.9875", "dec": "64.3758", "stn": "273",
            "sys": "ICRF_AU", "ctr": "399",
            "pos1": "0.00768793", "pos2": "-0.00332991", "pos3": "-0.00312377",
            "vel1": "0.000071", "vel2": "0.000274", "vel3": "0.000052",
        }
        observation = MpcClient._to_observation(row, "ec0001")
        self.assertEqual(
            observation.raw["observer_position_gcrs_au"],
            (0.00768793, -0.00332991, -0.00312377),
        )
        self.assertEqual(
            observation.raw["observer_velocity_gcrs_au_per_day"],
            (0.000071, 0.000274, 0.000052),
        )

    def test_ades_uncertainties_are_used_as_arcsecond_sigmas(self):
        observation = MpcClient._to_observation({
            "trkSub": "SWAN26Q", "obsTime": "2026-08-24T00:00:00Z",
            "ra": "1", "dec": "2", "stn": "C49",
            "rmsra": "120", "rmsdec": "150",
        }, "SWAN26Q")
        self.assertEqual(_observation_sigma_arcsec(observation), (120.0, 150.0))

    def test_ades_wgs84_roving_observer_is_not_treated_as_geocentre(self):
        observation = MpcClient._to_observation({
            "trkSub": "SWAN26Q", "obstime": "2026-09-03T03:05:26.1Z",
            "ra": "175.8953", "dec": "20.63973", "stn": "247",
            "sys": "WGS84", "pos1": "247.16759", "pos2": "37.63058",
            "pos3": "3190",
        }, "SWAN26Q")
        instant = parse_observation_time(observation.time)
        offsets = _observation_offsets_gcrs([
            (observation, instant, observation.ra_deg or 0.0, observation.dec_deg or 0.0)
        ])
        self.assertGreater(float(np.linalg.norm(offsets[0])), 4.0e-5)
        self.assertLess(float(np.linalg.norm(offsets[0])), 4.4e-5)

    def test_mpc80_mapping_preserves_station(self):
        line = list(" " * 80)
        line[0:12] = f"{'K24Y01C':<12}"
        line[15:32] = f"{'2024 12 25.50000':<17}"
        line[32:44] = f"{'12 34 56.70':<12}"
        line[44:56] = f"{'+12 34 56.7':<12}"
        line[65:70] = f"{18.4:5.1f}"
        line[70] = "G"
        line[77:80] = "M22"
        observation = parse_mpc80("".join(line))[0]
        self.assertEqual(observation.station, "M22")
        self.assertEqual(observation.time[:10], "2024-12-25")
        self.assertAlmostEqual(observation.ra_deg or 0.0, 188.73625)
        self.assertAlmostEqual(observation.dec_deg or 0.0, 12.5824167, places=6)

    def test_mpc80_spacecraft_pair_is_one_observation_with_offset(self):
        first = "     C46K1R1*0S2026 09 23.00099916 09 31.734+50 42 00.90               LNEOCPC53"
        second = "     C46K1R1 0s2026 09 23.0009991 +  505.3577 - 3075.1589 + 6130.6117   NEOCPC53"
        observations = parse_mpc80(first + "\n" + second)
        self.assertEqual(len(observations), 1)
        offset = observations[0].raw["observer_position_gcrs_au"]
        self.assertAlmostEqual(offset[0] * 149_597_870.7, 505.3577, places=4)
        self.assertAlmostEqual(offset[1] * 149_597_870.7, -3075.1589, places=4)
        self.assertAlmostEqual(offset[2] * 149_597_870.7, 6130.6117, places=4)

    def test_spacecraft_offset_overrides_unknown_station(self):
        instant = datetime(2026, 9, 23, tzinfo=timezone.utc)
        observation = parse_mpc80(
            "     C46K1R1 0S2026 09 23.00099916 09 31.734+50 42 00.90               LNEOCPC53\n"
            "     C46K1R1 0s2026 09 23.0009991 +  505.3577 - 3075.1589 + 6130.6117   NEOCPC53"
        )[0]
        offsets = _observation_offsets_gcrs([(observation, instant, 0.0, 0.0)])
        self.assertGreater(float(np.linalg.norm(offsets[0])), 4e-5)

    def test_observatory_offset_is_topocentric(self):
        instant = datetime(2026, 8, 3, tzinfo=timezone.utc)
        offsets = station_offsets_gcrs(["500", "M22"], [instant, instant])
        self.assertTrue(np.allclose(offsets[0], 0.0))
        self.assertGreater(float(np.linalg.norm(offsets[1])), 4e-5)

    def test_station_807_is_in_observatory_catalog(self):
        details = observatory_details("807")
        self.assertIsNotNone(details)
        self.assertIn("Cerro Tololo", str(details["name"]))

    def test_cartesian_orbit_mapping(self):
        speed = math.sqrt(MU_SUN)
        body = [{"mpc_orb": [{
            "CAR": {
                "coefficient_names": ["x", "y", "z", "vx", "vy", "vz"],
                "coefficient_values": [1, 0, 0, 0, speed, 0],
            },
            "epoch_data": {"epoch": 51544.5},
            "magnitude_data": {"H": 12.4, "G": 0.21},
        }]}]
        solution = parse_mpc_orbit(body, "Test")
        self.assertAlmostEqual(solution.epoch_jd, 2451545.0)
        self.assertAlmostEqual(solution.semi_major_axis_au, 1.0)
        self.assertAlmostEqual(solution.absolute_magnitude, 12.4)

    def test_mpc_comet_defaults_to_n4_without_asteroid_phase_law(self):
        speed = math.sqrt(MU_SUN)
        body = [{"mpc_orb": [{
            "CAR": {
                "coefficient_names": ["x", "y", "z", "vx", "vy", "vz"],
                "coefficient_values": [1, 0, 0, 0, speed, 0],
            },
            "epoch_data": {"epoch": 51544.5},
            "magnitude_data": {"H": 10.5, "G": 0.15},
        }]}]
        solution = parse_mpc_orbit(body, "2P")
        self.assertEqual(solution.photometric_exponent, 4.0)
        self.assertFalse(solution.phase_correction)

    def test_mpc_cartesian_comet_keeps_nongravitational_seed(self):
        speed = math.sqrt(MU_SUN)
        body = [{"mpc_orb": [{
            "CAR": {
                "coefficient_names": [
                    "x", "y", "z", "vx", "vy", "vz", "A1", "A2",
                ],
                "coefficient_values": [
                    1, 0, 0, 0, speed, 0, -2.51e-9, -2.47e-10,
                ],
            },
            "epoch_data": {"epoch": 51544.5},
        }]}]
        solution = parse_mpc_orbit(body, "78P")
        self.assertEqual(
            solution.nongravitational_parameters,
            {"A1": -2.51e-9, "A2": -2.47e-10},
        )

    def test_numbered_interstellar_designation_is_treated_as_comet(self):
        self.assertTrue(is_comet_designation("3I"))
        self.assertTrue(is_comet_designation("3I/ATLAS"))

    def test_mpc_display_name_for_named_asteroid(self):
        result = mpc_display_name({
            "iau_designation": "(9841)", "permid": "9841",
            "name": "Mašek", "unpacked_primary_provisional_designation": "1988 UT",
        }, "9841")
        self.assertEqual(result, "(9841) Mašek")

    def test_mpc_display_name_for_named_comet(self):
        result = mpc_display_name({
            "name": "Hale-Bopp",
            "unpacked_primary_provisional_designation": "C/1995 O1",
        }, "C/1995 O1")
        self.assertEqual(result, "C/1995 O1 (Hale-Bopp)")

    def test_numbered_comet_prefers_permanent_name_over_old_apparition(self):
        result = mpc_display_name({
            "name": "Encke", "permid": "2P",
            "unpacked_primary_provisional_designation": "P/1818 W1",
        }, "2P")
        self.assertEqual(result, "2P/Encke")

    def test_ambiguous_masek_identifier_offers_comet_and_asteroid(self):
        choices = mpc_identifier_choices({
            "found": 2,
            "disambiguation_list": [
                {"group": "Comets", "name": "Masek", "permid": None,
                 "unpacked_primary_provisional_designation": "C/2024 Y1"},
                {"group": "Minor Planets", "name": "Mašek", "permid": "9841",
                 "unpacked_primary_provisional_designation": "1988 UT"},
            ],
        }, "Masek")
        self.assertEqual(choices, [
            ("(9841) Mašek — planetka", "9841"),
            ("C/2024 Y1 (Masek) — kometa", "C/2024 Y1"),
        ])

    def test_ambiguous_name_choices_follow_discovery_order(self):
        choices = mpc_identifier_choices({
            "found": 5,
            "disambiguation_list": [
                {"group": "Comets", "name": "Leonard",
                 "unpacked_primary_provisional_designation": "C/2023 A1"},
                {"group": "Comets", "name": "Leonard",
                 "unpacked_primary_provisional_designation": "P/2018 VN2"},
                {"group": "Minor Planets", "name": "Leonard", "permid": "53435",
                 "unpacked_primary_provisional_designation": "1999 ED78"},
                {"group": "Comets", "name": "Leonard",
                 "unpacked_primary_provisional_designation": "P/2017 Y3"},
                {"group": "Comets", "name": "Leonard",
                 "unpacked_primary_provisional_designation": "C/2020 F1"},
            ],
        }, "Leonard")
        self.assertEqual([query for _, query in choices], [
            "53435", "P/2017 Y3", "P/2018 VN2", "C/2020 F1", "C/2023 A1",
        ])

    def test_cartesian_covariance_mapping(self):
        covariance = {
            f"cov{row}{column}": (1e-12 if row == column else 0.0)
            for row in range(6) for column in range(row, 6)
        }
        body = [{"mpc_orb": [{
            "CAR": {
                "coefficient_names": ["x", "y", "z", "vx", "vy", "vz"],
                "coefficient_values": [1, 0, 0, 0, math.sqrt(MU_SUN), 0],
                "covariance": covariance,
            },
            "epoch_data": {"epoch": 51544.5},
        }]}]
        solution = parse_mpc_orbit(body, "Test")
        self.assertIsNotNone(solution.state_covariance)
        self.assertAlmostEqual(solution.state_covariance[2][2], 1e-12)


class PreviewGeometryTests(unittest.TestCase):
    def test_integrated_trajectory_interpolates_animation_position(self):
        epochs = np.array([100.0, 102.0])
        states = np.array([
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [5.0, 8.0, 11.0, 0.0, 0.0, 0.0],
        ])
        position = SolarSystemPanel._trajectory_position((epochs, states), 101.0)
        np.testing.assert_allclose(position, [3.0, 5.0, 7.0])
        self.assertIsNone(SolarSystemPanel._trajectory_position((epochs, states), 99.0))

    def test_julian_date_conversion_does_not_use_windows_timestamp_range(self):
        self.assertEqual(datetime_from_jd(2451545.0), datetime(2000, 1, 1, 12, tzinfo=timezone.utc))
        self.assertIsNone(datetime_from_jd(-1e20))

    def test_mean_anomaly_near_360_uses_upcoming_perihelion(self):
        solution = OrbitSolution(
            designation="ec0001", epoch_jd=2461317.5,
            semi_major_axis_au=3914.52883, eccentricity=0.9977681,
            inclination_deg=85.20095, node_deg=304.03877,
            arg_peri_deg=101.83306, mean_anomaly_deg=359.99866038,
        )
        # Screenshot elements are rounded, so agreement within 0.01 day is the
        # meaningful regression check here.
        self.assertAlmostEqual(perihelion_jd_from_solution(solution), 2461650.38517, delta=0.01)

    def test_sexagesimal_ephemeris_coordinates(self):
        self.assertEqual(format_ra_sexagesimal(188.73625), "12 34 56.70")
        self.assertEqual(format_dec_sexagesimal(-12.5824167), "−12 34 56.7")

    def test_sun_centred_camera_axes_are_orthonormal(self):
        points = np.eye(3)
        x, y, depth = project_ecliptic_camera(points, 0.0, 0.0)
        self.assertTrue(np.allclose(x, [0.0, 1.0, 0.0]))
        self.assertTrue(np.allclose(y, [0.0, 0.0, 1.0]))
        self.assertTrue(np.allclose(depth, [1.0, 0.0, 0.0]))

    def test_camera_azimuth_rotates_only_about_ecliptic_z(self):
        points = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        x, y, depth = project_ecliptic_camera(points, 90.0, 0.0)
        self.assertTrue(np.allclose(x, [-1.0, 0.0], atol=1e-12))
        self.assertTrue(np.allclose(y, [0.0, 1.0], atol=1e-12))
        self.assertTrue(np.allclose(depth, [0.0, 0.0], atol=1e-12))

    def test_preview_geometry_is_finite(self):
        result = apparent_geometry(demo_solution(), np.array([2460950.5, 2460951.5]))
        for values in result.values():
            self.assertTrue(np.all(np.isfinite(values)))
        self.assertTrue(np.all((result["ra"] >= 0) & (result["ra"] < 360)))

    def test_ui_utc_is_converted_to_tdb_for_dynamics(self):
        utc_jd = 2460966.5
        converted = datetime_to_jd(datetime(2025, 10, 18, tzinfo=timezone.utc))
        self.assertGreater(converted, utc_jd + 60.0 / 86400.0)
        self.assertLess(converted, utc_jd + 80.0 / 86400.0)

    def test_cartesian_solution_uses_nbody_even_without_comet_parameters(self):
        solution = OrbitSolution(
            "test", 2460000.5, 2.0, .2, 3.0, 4.0, 5.0, 6.0,
            position_au=(1.0, 0.0, 0.0),
            velocity_au_per_day=(0.0, .01, 0.0),
        )
        propagated = np.array([[2.0, 3.0, 4.0, 0.0, 0.0, 0.0]])
        with patch("orbit_app.preview_orbit._cached_nbody_states", return_value=propagated) as mocked:
            result = heliocentric_xyz(solution, np.array([2460001.5]))
        self.assertTrue(np.array_equal(result, propagated[:, :3]))
        mocked.assert_called_once()

    def test_astrometric_geometry_iterates_target_light_time(self):
        epoch = 2460000.5

        def moving_target(_solution, epochs):
            values = np.asarray(epochs, dtype=float)
            return np.column_stack((10.0 + .02 * (values - epoch), values * 0.0, values * 0.0))

        with (
            patch("orbit_app.preview_orbit.heliocentric_xyz", side_effect=moving_target),
            patch("orbit_app.preview_orbit.earth_xyz", return_value=np.zeros((1, 3))),
        ):
            result = apparent_geometry(demo_solution(), np.array([epoch]))
        self.assertLess(float(result["xyz"][0, 0]), 10.0)
        self.assertAlmostEqual(float(result["light_time_days"][0]), 10.0 / 173.1446, delta=1e-4)

    def test_osculating_curve_has_no_wrap_chords(self):
        curve, valid = orbital_curve_xyz(demo_solution(), 30.0)
        steps = np.linalg.norm(np.diff(curve[valid], axis=0), axis=1)
        self.assertLess(float(np.max(steps)), 0.02)

    def test_closed_orbit_is_not_preclipped_by_solar_viewport(self):
        solution = OrbitSolution(
            designation="9841", epoch_jd=2461118.5,
            semi_major_axis_au=2.18882215, eccentricity=.1600963,
            inclination_deg=2.19873, node_deg=138.52648,
            arg_peri_deg=264.66660, mean_anomaly_deg=185.59746671,
        )
        self.assertIsNone(solar_orbit_clip_radius(solution, 1.9))
        curve, valid = orbital_curve_xyz(solution, max_radius_au=None)
        self.assertTrue(np.all(valid))
        self.assertTrue(np.allclose(curve[0], curve[-1], atol=1e-12))

    def test_hyperbolic_state_and_propagation(self):
        position = np.array([1.0, 0.0, 0.0])
        velocity = np.array([0.0, math.sqrt(2.5 * MU_SUN), 0.0])
        elements = cartesian_to_elements(position, velocity)
        self.assertLess(elements[0], 0.0)
        self.assertGreater(elements[1], 1.0)
        from orbit_app.domain import OrbitSolution
        solution = OrbitSolution("Hyperbola", 2451545.0, *elements)
        states = heliocentric_xyz(solution, np.array([2451535.0, 2451545.0, 2451555.0]))
        self.assertTrue(np.all(np.isfinite(states)))
        self.assertTrue(np.allclose(states[1], position, atol=1e-10))

    def test_near_parabolic_kepler_solution_is_continuous(self):
        mean = np.linspace(-.05, .05, 2001)
        eccentric = solve_kepler(mean, .999999)
        residual = eccentric - .999999 * np.sin(eccentric) - mean
        self.assertLess(float(np.max(np.abs(residual))), 2e-14)
        self.assertTrue(np.all(np.diff(eccentric) >= 0.0))


class OrbitFitSelectionTests(unittest.TestCase):
    def test_fit_can_be_cancelled_before_numerical_work_starts(self):
        with self.assertRaises(OrbitFitCancelled):
            fit_orbit([], None, cancel_callback=lambda: True)

    def test_missing_astrometric_uncertainty_uses_epoch_weight(self):
        historical = Observation(time="1835-10-01T00:00:00Z")
        photographic = Observation(time="1935-10-01T00:00:00Z")
        modern = Observation(time="2026-10-01T00:00:00Z")
        self.assertEqual(_observation_sigma_arcsec(historical), (20.0, 20.0))
        self.assertEqual(_observation_sigma_arcsec(photographic), (6.0, 6.0))
        self.assertEqual(_observation_sigma_arcsec(modern), (1.0, 1.0))

    def test_latest_apparition_is_split_at_long_observing_gap(self):
        observations = [
            Observation(time="1835-10-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0),
            Observation(time="1910-04-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0),
            Observation(time="1985-10-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0),
            Observation(time="2003-01-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0),
        ]
        latest = _latest_observing_apparition(observations)
        self.assertEqual([item.time[:4] for item in latest], ["1985", "2003"])

    def test_short_period_comet_apparitions_follow_its_orbital_period(self):
        observations = [
            Observation(
                time=f"{year}-10-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for year in (2017, 2020, 2023)
            for day in range(1, 7)
        ]
        encke = OrbitSolution(
            "2P", 2460220.5, 2.22, .85, 11.3, 334.0, 187.0, 0.0,
        )
        latest = _latest_observing_apparition(
            observations, encke, minimum_size=6,
        )
        self.assertEqual(len(latest), 6)
        self.assertTrue(all(item.time.startswith("2023-") for item in latest))

    def test_gravity_only_short_period_comet_uses_latest_return(self):
        observations = [
            Observation(
                time=f"{year}-10-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for year in (2017, 2020, 2023)
            for day in range(1, 7)
        ]
        encke = OrbitSolution(
            "2P", 2460220.5, 2.22, .85, 11.3, 334.0, 187.0, 0.0,
            position_au=(.34, 0.0, 0.0), velocity_au_per_day=(0.0, .04, 0.0),
        )
        candidate = (encke, 1.0, 6)
        with (
            patch("orbit_app.orbit_fit._fit_orbit_once", return_value=candidate) as fit_once,
            patch("orbit_app.orbit_fit._attach_multi_apparition_elements"),
        ):
            fit_orbit(observations, encke)
        selected = fit_once.call_args_list[0].args[0]
        self.assertEqual(len(selected), 6)
        self.assertTrue(all(item.time.startswith("2023-") for item in selected))

    def test_long_gap_in_asteroid_astrometry_does_not_select_one_apparition(self):
        observations = [
            Observation(
                time=f"1950-01-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for day in range(1, 7)
        ] + [
            Observation(
                time=f"2025-01-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for day in range(1, 7)
        ]
        asteroid = OrbitSolution("12345", 2460676.5, 2.4, .2, 5.0, 10.0, 20.0, 30.0)
        candidate = (asteroid, 1.0, len(observations))
        with (
            patch("orbit_app.orbit_fit._fit_orbit_once", return_value=candidate) as fit_once,
            patch("orbit_app.orbit_fit._attach_multi_apparition_elements"),
        ):
            fit_orbit(observations, asteroid)
        self.assertEqual(len(fit_once.call_args_list[0].args[0]), len(observations))

    def test_gravity_only_multi_apparition_fit_uses_latest_return(self):
        observations = [
            Observation(
                time=f"1835-10-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for day in range(1, 7)
        ] + [
            Observation(
                time=f"1986-03-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for day in range(1, 11)
        ]
        solution = OrbitSolution(
            "1P", 2446470.5, 17.8, .967, 162.3, 58.4, 111.3, 0.0,
            position_au=(.586, 0.0, 0.0), velocity_au_per_day=(0.0, .03, 0.0),
        )
        candidate = (solution, 1.0, 10)
        with (
            patch("orbit_app.orbit_fit._fit_orbit_once", return_value=candidate) as fit_once,
            patch("orbit_app.orbit_fit._attach_multi_apparition_elements"),
        ):
            fit_orbit(observations, solution)
        self.assertEqual(fit_once.call_count, 1)
        self.assertEqual(len(fit_once.call_args_list[0].args[0]), 10)
        self.assertEqual(fit_once.call_args_list[0].kwargs["maximum_observations"], 120)
        self.assertIn("posledního pozorovaného návratu", solution.fit_model_note)

    def test_long_arc_seed_holds_a_parameters_then_frees_them_progressively(self):
        observations = [
            Observation(
                time=f"1835-10-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for day in range(1, 7)
        ] + [
            Observation(
                time=f"1986-03-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for day in range(1, 11)
        ]
        solution = OrbitSolution(
            "1P", 2446470.5, 17.8, .967, 162.3, 58.4, 111.3, 0.0,
            position_au=(.586, 0.0, 0.0), velocity_au_per_day=(0.0, .03, 0.0),
            fit_normalized_rms=.8,
        )
        calls = []

        def fake_fit(*args, **kwargs):
            calls.append((
                dict(args[4] or {}),
                dict(kwargs.get("fixed_nongravitational") or {}),
                kwargs.get("fit_state", True),
            ))
            if len(calls) == 1:
                solution.nongravitational_parameters = {"A1": 2e-8}
            return solution, 1.0, 10

        with (
            patch("orbit_app.orbit_fit._fit_orbit_once", side_effect=fake_fit),
            patch("orbit_app.orbit_fit._attach_multi_apparition_elements"),
        ):
            fit_orbit(observations, solution, nongravitational={"A1": 0.0, "DT": 30.0})
        self.assertEqual(calls[0], ({}, {"A1": 0.0}, True))
        self.assertEqual(calls[1], ({"A1": 2e-8}, {}, False))
        self.assertTrue(calls[-1][2])

    def test_three_comet_returns_are_linked_progressively(self):
        observations = [
            Observation(
                time=f"{year}-{month:02d}-{day:02d}T00:00:00Z",
                ra_deg=1.0, dec_deg=2.0,
            )
            for year, month, count in ((1835, 10, 6), (1910, 4, 8), (1986, 3, 10))
            for day in range(1, count + 1)
        ]
        solution = OrbitSolution(
            "1P", 2446470.5, 17.8, .967, 162.3, 58.4, 111.3, 0.0,
            position_au=(.586, 0.0, 0.0), velocity_au_per_day=(0.0, .03, 0.0),
            fit_normalized_rms=.8,
        )
        row_counts = []

        def fake_fit(rows, *_args, **_kwargs):
            row_counts.append(len(rows))
            solution.nongravitational_parameters = {"A1": 1e-8}
            solution.fit_normalized_rms = .8
            return solution, 1.0, len(rows)

        with (
            patch("orbit_app.orbit_fit._fit_orbit_once", side_effect=fake_fit),
            patch("orbit_app.orbit_fit._attach_multi_apparition_elements"),
        ):
            result = fit_orbit(
                observations, solution,
                nongravitational={"A1": 0.0, "DT": 0.0},
            )
        # Each older return gets a cheap fixed-state A prefit followed by a
        # short state correction; only one strict all-return polish is done.
        self.assertEqual(row_counts, [10, 18, 18, 24, 24, 24, 24])
        self.assertIn("všech 3", result[0].fit_model_note or "")
        self.assertEqual(
            [item["status"] for item in result[0].apparition_linkage or ()],
            ["linked", "linked", "linked"],
        )
        self.assertTrue(
            (apparition_linkage_summary(result[0]) or "").startswith(
                "NAVÁZÁNO 3/3 návratů"
            )
        )

    def test_any_periodic_comet_can_retry_linkage_from_older_local_astrometry(self):
        observations = [
            Observation(
                time=f"{year}-03-{day:02d}T00:00:00Z",
                ra_deg=10.0, dec_deg=20.0,
            )
            for year in (2010, 2020)
            for day in range(1, 7)
        ]
        seed = OrbitSolution(
            "321P/Test", 2458909.5, 4.6416, .45, 12.0, 30.0, 60.0, 0.0,
            position_au=(2.5, 0.0, 0.0),
            velocity_au_per_day=(0.0, .01, 0.0),
            fit_normalized_rms=.8,
            nongravitational_parameters={"A1": 1e-8},
        )
        bad = OrbitSolution(
            "321P/Test", 2458909.5, 4.6416, .45, 12.0, 30.0, 60.0, 0.0,
            position_au=(2.5, 0.0, 0.0),
            velocity_au_per_day=(0.0, .01, 0.0),
            fit_normalized_rms=100.0,
            nongravitational_parameters={"A1": 1e-8},
        )
        row_counts = []

        def fake_fit(rows, *_args, **_kwargs):
            row_counts.append(len(rows))
            # The fixed-state prefit is call 2; make the subsequent state
            # correction fail so the independent local branch is exercised.
            if len(row_counts) == 3:
                return bad, 100.0, len(rows)
            return seed, 1.0, len(rows)

        with (
            patch("orbit_app.orbit_fit._fit_orbit_once", side_effect=fake_fit),
            patch("orbit_app.orbit_fit._attach_multi_apparition_elements"),
        ):
            result = fit_orbit(
                observations, seed, nongravitational={"A1": 0.0},
            )
        self.assertEqual(row_counts, [6, 12, 12, 6, 12, 12, 12])
        self.assertEqual(
            [item["status"] for item in result[0].apparition_linkage or ()],
            ["linked", "linked"],
        )
        self.assertIn("všech 2", result[0].fit_model_note or "")

    def test_final_exact_fit_cannot_claim_return_with_large_residuals(self):
        observations = [
            Observation(
                time=f"{year}-03-{day:02d}T00:00:00Z",
                ra_deg=10.0, dec_deg=20.0,
            )
            for year in (1892, 2008) for day in range(1, 7)
        ]
        solution = OrbitSolution(
            "206P", 2454751.5, 3.2, .64, 32.0, 204.0, 181.0, 0.0,
            position_au=(1.2, 0.0, 0.0),
            velocity_au_per_day=(0.0, .015, 0.0),
            fit_normalized_rms=.8,
            nongravitational_parameters={"A1": 1e-8},
        )
        calls = 0

        def fake_fit(rows, *_args, **_kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                for observation in observations:
                    old = observation.time.startswith("1892")
                    observation.residual_ra_arcsec = 900.0 if old else .3
                    observation.residual_dec_arcsec = 100.0 if old else -.2
                    observation.used_in_fit = True
            return solution, 1.0, len(rows)

        with (
            patch("orbit_app.orbit_fit._fit_orbit_once", side_effect=fake_fit),
            patch("orbit_app.orbit_fit._attach_multi_apparition_elements"),
        ):
            result = fit_orbit(
                observations, solution, nongravitational={"A1": 0.0},
            )
        self.assertEqual(
            [item["status"] for item in result[0].apparition_linkage or ()],
            ["failed", "linked"],
        )
        self.assertIn("NENÍ označen", result[0].fit_model_note or "")

    def test_multi_apparition_table_marks_future_relative_to_today(self):
        solution = OrbitSolution(
            "1P", 2446470.5, 17.8, .967, 162.3, 58.4, 111.3, 0.0,
            position_au=(.586, 0.0, 0.0), velocity_au_per_day=(0.0, .03, 0.0),
        )
        observations = [
            Observation(time="1835-10-01T00:00:00Z"),
            Observation(time="2003-01-01T00:00:00Z"),
        ]
        past = EvolutionReturn(-1, 2392000.0, (0.0,) * 6, 17.9, .967, 162.3,
                               58.4, 111.3, .59, 75.7, 1 / 17.9)
        after_observations = EvolutionReturn(
            0, 2456900.0, (1.0,) * 6, 3.2, .64, 32.0,
            204.0, 181.0, 1.15, 5.8, 1 / 3.2,
        )
        future = EvolutionReturn(1, 2474000.0, (0.0,) * 6, 18.0, .968, 162.2,
                                 58.3, 111.4, .58, 76.4, 1 / 18.0)
        with patch(
            "orbit_app.orbit_evolution.perihelion_returns_between",
            return_value=(past, after_observations, future),
        ):
            _attach_multi_apparition_elements(observations, solution)
        self.assertEqual(len(solution.apparition_elements or ()), 3)
        self.assertFalse(solution.apparition_elements[0]["future"])
        self.assertFalse(solution.apparition_elements[1]["future"])
        self.assertTrue(solution.apparition_elements[2]["future"])
        self.assertEqual(solution.apparition_elements[1]["state"], (1.0,) * 6)

    def test_fitted_photometry_reduces_apparent_magnitude_to_absolute(self):
        solution = OrbitSolution(
            "PCCP", 2461300.5, 5.0, .3, 17, 227, 72, 4.5,
            position_au=(5.0, 0.0, 0.0), velocity_au_per_day=(0.0, .005, 0.0),
        )
        observation = Observation(
            time="2026-10-12T00:00:00Z", magnitude=19.0,
        )
        geometry = {
            "delta": np.array([4.0]), "r": np.array([5.0]),
            "phase": np.array([0.0]),
        }
        with patch("orbit_app.preview_orbit.apparent_geometry", return_value=geometry):
            _calibrate_fitted_photometry([observation], solution)
        expected = 19.0 - 5.0 * math.log10(4.0) - 5.0 * math.log10(5.0)
        self.assertAlmostEqual(solution.absolute_magnitude, expected)

    def test_nongravitational_fit_uses_subset_as_seed_then_all_observations(self):
        solution = OrbitSolution(
            "C/2023 A3", 2460400.5, -4000.0, 1.0001, 139.0,
            21.5, 308.5, 0.0,
            position_au=(1.0, 0.0, 0.0), velocity_au_per_day=(0.0, .02, 0.0),
        )
        candidate = (solution, .8, 120)
        final = (solution, .7, 6314)
        observations = [Observation() for _ in range(6)]
        with patch("orbit_app.orbit_fit._fit_orbit_once", side_effect=[candidate, final]) as mocked:
            result = fit_orbit(
                observations, solution, constraints={"e": 1.0001},
                nongravitational={"A1": 1e-8},
            )
        self.assertEqual(result[2], 6314)
        self.assertEqual(mocked.call_args_list[0].kwargs["maximum_observations"], 120)
        self.assertIsNone(mocked.call_args_list[1].kwargs["maximum_observations"])
        self.assertTrue(mocked.call_args_list[1].kwargs["robust_loss"])

    def test_nongravitational_fit_still_filters_worst_outlier(self):
        solution = OrbitSolution(
            "C/2023 A3", 2460400.5, -4000.0, 1.0001, 139.0,
            21.5, 308.5, 0.0,
            position_au=(1.0, 0.0, 0.0), velocity_au_per_day=(0.0, .02, 0.0),
        )
        observations = [Observation(ra_deg=1.0, dec_deg=2.0) for _ in range(20)]

        def staged_fit(*_args, **_kwargs):
            if staged_fit.calls == 1:
                for index, observation in enumerate(observations):
                    observation.used_in_fit = True
                    observation.residual_ra_arcsec = 10.0 if index == 0 else .1
                    observation.residual_dec_arcsec = .1
            staged_fit.calls += 1
            return solution, .7, len(observations)

        staged_fit.calls = 0
        with patch("orbit_app.orbit_fit._fit_orbit_once", side_effect=staged_fit) as mocked:
            fit_orbit(observations, solution, nongravitational={"A1": 1e-8})
        self.assertEqual(mocked.call_count, 3)
        self.assertTrue(observations[0].excluded_from_fit)
        self.assertTrue(observations[0].raw["automatic_exclusion"])

    def test_automatic_rejection_needs_a_physical_two_arcsecond_error(self):
        tiny_sigma = Observation(
            residual_ra_arcsec=.4, residual_dec_arcsec=.3,
            raw={"rmsRA": .05, "rmsDec": .05},
        )
        obvious_blunder = Observation(
            residual_ra_arcsec=3.0, residual_dec_arcsec=1.0,
            raw={"rmsRA": .2, "rmsDec": .2},
        )
        self.assertFalse(_is_automatic_rejection_candidate(tiny_sigma, .25))
        self.assertTrue(_is_automatic_rejection_candidate(obvious_blunder, .25))

    def test_four_day_candidate_compares_range_branches_even_with_initial_orbit(self):
        start = datetime(2026, 10, 12, tzinfo=timezone.utc)
        observations = [
            Observation(
                time=(start + timedelta(hours=index * 3.25)).isoformat(),
                ra_deg=10.0 + index / 100.0, dec_deg=-20.0,
            )
            for index in range(29)
        ]
        supplied = OrbitSolution("PCCP", 2461327.5, 3.6, .19, 17, 233, 356, 58)
        wrong = OrbitSolution("PCCP", 2461327.5, 3.5, .18, 17, 233, 350, 60)
        right = OrbitSolution("PCCP", 2461327.5, 5.0, .30, 16.8, 227, 72, 4.5)

        def staged_fit(_observations, candidate, *_args, **kwargs):
            rms = .24 if candidate is right else .40
            for observation in observations:
                observation.residual_ra_arcsec = rms
                observation.residual_dec_arcsec = 0.0
                observation.used_in_fit = True
            return candidate, rms, len(observations)

        with (
            patch(
                "orbit_app.orbit_fit.initial_orbit_from_observations",
                return_value=[wrong, right],
            ) as ranging,
            patch("orbit_app.orbit_fit._fit_orbit_once", side_effect=staged_fit) as fit_once,
            patch("orbit_app.orbit_fit._prefer_parabolic_short_arc", side_effect=lambda _, value, __: value),
        ):
            result = fit_orbit(observations, supplied)
        self.assertIs(result[0], right)
        ranging.assert_called_once_with(observations, return_candidates=True)
        self.assertGreaterEqual(fit_once.call_count, 6)

    def test_interpolated_ephemeris_includes_the_moon(self):
        start = 2451545.0
        cached = _InterpolatedPlanetEphemeris(PLANETS, start, start + 1.0)
        moon = cached.position("moon", start + .5)
        self.assertEqual(moon.shape, (3,))
        self.assertTrue(np.all(np.isfinite(moon)))

    def test_interstellar_solution_is_never_replaced_by_parabolic_trial(self):
        interstellar = OrbitSolution("3I", 2461200.5, -0.26, 6.1, 175, 322, 128, 10)
        interstellar.fit_normalized_rms = 1.0
        observation = Observation(
            designation="3I", time="2025-07-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            residual_ra_arcsec=0.2, residual_dec_arcsec=-0.1, used_in_fit=True,
        )
        candidate = (interstellar, 0.8, 1)
        with patch("orbit_app.orbit_fit._fit_orbit_once") as trial:
            result = _prefer_parabolic_short_arc([observation], candidate, None)
        self.assertIs(result, candidate)
        trial.assert_not_called()

    def test_rejected_parabolic_trial_restores_candidate_residuals(self):
        free = OrbitSolution("X", 2461000.5, -50.0, 1.03, 150, 266, 339, -1)
        free.fit_normalized_rms = 0.5
        parabolic = OrbitSolution("X", 2461000.5, -1e8, 1.0, 150, 266, 342, 0)
        parabolic.fit_normalized_rms = 5.0
        observation = Observation(
            designation="X", time="2026-10-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            residual_ra_arcsec=0.2, residual_dec_arcsec=-0.1, used_in_fit=True,
        )

        def rejected_trial(*args, **kwargs):
            observation.residual_ra_arcsec = 10000.0
            observation.residual_dec_arcsec = -20000.0
            observation.used_in_fit = False
            return parabolic, 10.0, 1

        with patch("orbit_app.orbit_fit._fit_orbit_once", side_effect=rejected_trial):
            result = _prefer_parabolic_short_arc([observation], (free, 0.5, 1), None)
        self.assertIs(result[0], free)
        self.assertEqual(observation.residual_ra_arcsec, 0.2)
        self.assertEqual(observation.residual_dec_arcsec, -0.1)
        self.assertTrue(observation.used_in_fit)

    def test_large_fit_sample_represents_sparse_time_interval(self):
        dense_recent = np.linspace(90.0, 100.0, 900)
        epochs = np.concatenate((np.linspace(0.0, 10.0, 100), dense_recent))
        indices = _representative_indices(epochs, 120)
        selected = epochs[indices]
        self.assertLessEqual(len(indices), 120)
        # Pure rank sampling would retain only about six old observations;
        # the elapsed-time half must add independent coverage of that arc.
        self.assertGreater(np.count_nonzero(selected < 50.0), 6)
        self.assertEqual(selected[0], epochs[0])
        self.assertEqual(selected[-1], epochs[-1])

    def test_clustered_multi_apparition_sample_uses_requested_capacity(self):
        epochs = np.concatenate((
            np.linspace(0.0, 100.0, 500),
            np.linspace(27375.0, 27475.0, 1500),
            np.linspace(54750.0, 54900.0, 6432),
        ))
        indices = _representative_indices(epochs, 480)
        self.assertEqual(len(indices), 480)
        self.assertGreater(np.count_nonzero(epochs[indices] < 1000.0), 15)
        self.assertGreater(np.count_nonzero(
            (epochs[indices] > 27000.0) & (epochs[indices] < 28000.0)
        ), 30)

    def test_catastrophic_linked_comet_fit_falls_back_to_latest_apparition(self):
        observations = [
            Observation(
                time=f"1835-10-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for day in range(1, 7)
        ] + [
            Observation(
                time=f"1986-03-{day:02d}T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
            )
            for day in range(1, 11)
        ]
        seed = OrbitSolution(
            "1P", 2446470.5, 17.8, .967, 162.3, 58.4, 111.3, 0.0,
            position_au=(.586, 0.0, 0.0), velocity_au_per_day=(0.0, .03, 0.0),
            fit_normalized_rms=1.0, nongravitational_parameters={"A1": 1e-8},
        )
        failed = OrbitSolution(
            "1P", 2446470.5, 18.0, .96, 162.0, 58.0, 111.0, 0.0,
            position_au=(.6, 0.0, 0.0), velocity_au_per_day=(0.0, .03, 0.0),
            fit_normalized_rms=5000.0, nongravitational_parameters={"A1": 2e-8},
        )

        def staged_fit(rows, *_args, **_kwargs):
            if staged_fit.calls == 0:
                for observation in rows:
                    observation.residual_ra_arcsec = .2
                    observation.residual_dec_arcsec = -.1
                    observation.used_in_fit = True
                result = (seed, .3, len(rows))
            else:
                for observation in observations:
                    observation.residual_ra_arcsec = 100000.0
                    observation.residual_dec_arcsec = 100000.0
                    observation.used_in_fit = True
                result = (failed, 100000.0, len(rows))
            staged_fit.calls += 1
            return result

        staged_fit.calls = 0
        with (
            patch("orbit_app.orbit_fit._fit_orbit_once", side_effect=staged_fit),
            patch("orbit_app.orbit_fit._attach_multi_apparition_elements"),
        ):
            result = fit_orbit(
                observations, seed, nongravitational={"A1": 0.0},
            )
        self.assertIs(result[0], seed)
        self.assertIsNone(observations[0].residual_ra_arcsec)
        self.assertAlmostEqual(observations[-1].residual_ra_arcsec or 0.0, .2)
        self.assertIn("neprošel kontrolou kvality", seed.fit_model_note or "")
        self.assertEqual(
            [item["status"] for item in seed.apparition_linkage or ()],
            ["failed", "seed_only"],
        )
        self.assertEqual(
            apparition_linkage_summary(seed),
            "NENAVÁZÁNO — společný fit 1/2 návratů",
        )

    def test_representative_count_says_which_arc_it_describes(self):
        solution = OrbitSolution("1P", 2446470.5, 17.8, .967, 162, 58, 111, 0)
        solution.fit_observation_count = 360
        solution.fit_available_observation_count = 8432
        self.assertEqual(
            apparition_observation_count_text(solution),
            "360 reprezentativních z 8432 v propojeném oblouku",
        )

    def test_parabolic_perihelion_offset_is_finite_at_singularity(self):
        q = 3.2
        state = np.array([q, 0.0, 0.0, 0.0, math.sqrt(2.0 * MU_SUN / q), 0.0])
        self.assertAlmostEqual(_perihelion_offset_days_from_state(state), 0.0)
        perturbed = state.copy(); perturbed[0] += 1e-7
        self.assertTrue(np.isfinite(
            _perihelion_offset_days_from_state(perturbed, force_parabolic=True)
        ))

    def test_isolated_latest_station_batch_is_fitted_before_outlier_decision(self):
        start = datetime(2026, 10, 5, tzinfo=timezone.utc)
        observations = [
            Observation(
                time=(start + timedelta(hours=index * 8)).isoformat(),
                ra_deg=1.0, dec_deg=2.0, station=f"A{index % 3}",
            )
            for index in range(10)
        ]
        observations.extend([
            Observation(
                time=(start + timedelta(days=35, minutes=index * 10)).isoformat(),
                ra_deg=1.0, dec_deg=2.0, station="E62",
            )
            for index in range(2)
        ])
        solution = OrbitSolution("X", 2461320.5, -80.0, 1.04, 150, 266, 339, -1)
        solution.fit_normalized_rms = 0.6
        candidate = (solution, 0.3, len(observations))
        with (
            patch("orbit_app.orbit_fit._fit_orbit_once", return_value=candidate) as fit_once,
            patch("orbit_app.orbit_fit._prefer_parabolic_short_arc", return_value=candidate),
        ):
            result = fit_orbit(observations, None)
        self.assertIs(result, candidate)
        fit_once.assert_called_once()
        self.assertIsNone(fit_once.call_args_list[0].args[1])
        self.assertTrue(all(not item.excluded_from_fit for item in observations))

    def test_automatic_score_discourages_strong_short_arc_hyperbola(self):
        free = OrbitSolution("X", 2461000.5, -50.0, 1.07, 150, 266, 339, -1)
        free.fit_normalized_rms = .70
        parabolic = OrbitSolution("X", 2461000.5, -1e8, 1.00000001, 150, 266, 342, 0)
        parabolic.fit_normalized_rms = .72
        self.assertLess(_automatic_orbit_score(parabolic), _automatic_orbit_score(free))

    def test_fit_selection_count_respects_window_and_manual_exclusion(self):
        from orbit_app.domain import Observation
        observations = [
            Observation(time="2022-01-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0),
            Observation(time="2024-06-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0),
            Observation(time="2025-01-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0,
                        excluded_from_fit=True),
            Observation(time="2025-02-01T00:00:00Z", ra_deg=1.0, dec_deg=2.0),
        ]
        self.assertEqual(fit_selection_count(observations, None), 3)
        self.assertEqual(fit_selection_count(observations, 2.0), 2)

    def test_ades_time_with_hundredths_is_accepted(self):
        instant = parse_observation_time("2024-12-29T00:58:51.17Z")
        self.assertEqual(instant.microsecond, 170000)

    def test_two_year_window_is_relative_to_newest_observation(self):
        newest = datetime(2025, 1, 1, tzinfo=timezone.utc)
        rows = [
            (object(), newest - timedelta(days=900), 0.0, 0.0),
            (object(), newest - timedelta(days=700), 0.0, 0.0),
            (object(), newest, 0.0, 0.0),
        ]
        selected = _truncate_rows_to_recent_years(rows, 2.0)
        self.assertEqual(len(selected), 2)
        self.assertEqual(selected[-1][1], newest)


class _NoPlanets:
    def position(self, body: str, jd: float) -> np.ndarray:
        raise AssertionError("Sun-only test must not request a planet.")


class NativeIntegratorTests(unittest.TestCase):
    def test_local_light_time_expansion_matches_short_exact_propagation(self):
        epoch = 2460000.5
        position = np.array([5.0, 1.0, .2])
        velocity = np.array([-.002, .006, .001])
        state = StateVector(epoch, position, velocity)
        engine = NBodyIntegrator(PLANETS, perturbers=())
        reception = state.packed()[None, :]
        local, light_time = _local_light_time_states(reception, np.zeros((1, 3)))
        exact = engine.propagate(state, np.array([epoch - light_time[0]]))
        np.testing.assert_allclose(local[:, :3], exact[:, :3], atol=2e-10, rtol=0.0)

    def test_planet_provider_includes_lunar_perturber(self):
        earth = PLANETS.position("earth", 2451545.0)
        moon = PLANETS.position("moon", 2451545.0)
        distance = float(np.linalg.norm(moon - earth))
        self.assertGreater(distance, .0023)
        self.assertLess(distance, .0028)

    def test_planetocentric_elements_use_selected_planet_mass(self):
        radius = .01
        speed = math.sqrt(PLANET_MU["jupiter"] / radius)
        item = EvolutionSnapshot(
            "", 2451545.0, (radius, 0.0, 0.0, 0.0, speed, 0.0),
            radius, 0.0, 0.0, 0.0, 0.0, radius, None, 1.0 / radius,
        )
        with patch("orbit_app.orbit_evolution.PLANETS.position", return_value=np.zeros(3)), \
                patch("orbit_app.orbit_evolution.planet_velocity", return_value=np.zeros(3)):
            centered = centered_snapshot(item, "jupiter")
        self.assertAlmostEqual(centered.semi_major_axis_au, radius, places=10)
        self.assertAlmostEqual(centered.eccentricity, 0.0, places=10)
        expected_years = 2.0 * math.pi * math.sqrt(
            radius**3 / PLANET_MU["jupiter"]
        ) / 365.25
        self.assertAlmostEqual(centered.period_years, expected_years, places=10)

    def test_short_arc_planet_cache_preserves_cubic_motion(self):
        class PolynomialEphemeris:
            def position(self, body, jd):
                return np.array([jd ** 3, 2.0 * jd ** 2, -jd])

        cached = _InterpolatedPlanetEphemeris(
            PolynomialEphemeris(), 0.0, 2.0, step_days=0.25,
        )
        np.testing.assert_allclose(
            cached.position("earth", 0.37),
            PolynomialEphemeris().position("earth", 0.37),
            atol=1e-12,
        )

    @staticmethod
    def _evolution_export_fixture():
        solution = OrbitSolution(
            "C/2024 Y1", 2460676.5, 60.0, .986, 64.7, 112.5, 238.1, 0.0,
            absolute_magnitude=17.6, photometric_exponent=4.0,
            display_name="C/2024 Y1 (Mašek)",
        )
        returns = (
            EvolutionReturn(-1, 2460600.0, (0, 0, 0, 0, 0, 0), 60.0, .986,
                            64.7, 112.5, 238.1, .84, 464.8, 1 / 60),
            EvolutionReturn(0, 2461600.0, (0, 0, 0, 0, 0, 0), 61.0, .987,
                            64.8, 112.6, 238.2, .793, 476.4, 1 / 61),
        )
        encounters = (
            PlanetEncounter(2461000.0, "jupiter", .2, .563, 12.5, 2e-4),
        )
        result = EvolutionResult(
            returns, encounters, 2460500.0, 2461700.0, 2.0, "Testovací model",
        )
        return solution, result

    def test_readable_evolution_export_is_chronological(self):
        solution, result = self._evolution_export_fixture()
        report = format_evolution_report(solution, result)
        self.assertLess(report.index("PERIHELIUM"), report.index("SETKÁNÍ"))
        self.assertLess(report.index("SETKÁNÍ"), report.rindex("PERIHELIUM"))
        self.assertIn("Jupiter", report)
        self.assertIn("Martin Mašek", report)

    def test_open_orbit_export_uses_inbound_perihelion_outbound_states(self):
        solution, base = self._evolution_export_fixture()
        snapshots = tuple(
            EvolutionSnapshot(
                phase, epoch, (0, 0, 0, 0, 0, 0), -0.25, 6.0, 175.0,
                322.0, 128.0, 1.25, None, -4.0 + offset,
            )
            for phase, epoch, offset in (
                ("vstup", 2460500.0, 0.0),
                ("perihel", 2461000.0, .001),
                ("výstup", 2461700.0, .002),
            )
        )
        result = EvolutionResult(
            base.returns, base.encounters, base.start_jd, base.end_jd,
            base.sample_step_days, base.model_note, snapshots,
        )
        report = format_evolution_report(solution, result)
        self.assertIn("STAV\t", report)
        self.assertIn("\tvstup\t", report)
        self.assertIn("\tperihel\t", report)
        self.assertIn("\tvýstup\t", report)
        self.assertNotIn("PERIHELIUM\t", report)

    def test_mpc_evolution_export_contains_elements_and_commented_encounter(self):
        solution, result = self._evolution_export_fixture()
        report = format_evolution_mpc_report(solution, result)
        self.assertIn("K24Y010", report)
        self.assertIn("# SETKÁNÍ", report)
        element_lines = [line for line in report.splitlines() if "K24Y010" in line]
        self.assertEqual(len(element_lines), 2)
        self.assertRegex(element_lines[0], r"^\s+CK24Y010\s+20\d{2} \d{2} \d{2}\.\d{4}")

    def test_numbered_comet_evolution_export_uses_perihelion_date_and_q(self):
        _, result = self._evolution_export_fixture()
        solution = OrbitSolution(
            "2P", 2462500.5, 2.2178179, .8472983, 11.34795,
            334.01955, 187.28396, 286.52227,
            absolute_magnitude=14.36, photometric_exponent=4.0,
            display_name="2P/Encke",
        )
        report = format_evolution_mpc_report(solution, result)
        element_lines = [
            line for line in report.splitlines()
            if line and not line.startswith("#")
        ]
        self.assertEqual(len(element_lines), 2)
        self.assertRegex(element_lines[0], r"^\s*2P\s+20\d{2} \d{2} \d{2}\.\d{4}")
        self.assertIn(" 0.840000", element_lines[0])
        self.assertNotIn("  0.00000 ", element_lines[0])

    def test_evolution_year_range_is_asymmetric(self):
        solution = OrbitSolution(
            "Test", 2451545.0, 2.0, .2, 0.0, 0.0, 0.0, 0.0,
            position_au=(1.6, 0.0, 0.0), velocity_au_per_day=(0.0, .01, 0.0),
        )
        start, stop = evolution_span(solution, "years", 150, 200)
        self.assertAlmostEqual(solution.epoch_jd - start, 150 * 365.25)
        self.assertAlmostEqual(stop - solution.epoch_jd, 200 * 365.25)

    def test_return_range_rejects_open_orbit(self):
        solution = OrbitSolution("Open", 2451545.0, -10.0, 1.1, 0.0, 0.0, 0.0, 0.0)
        with self.assertRaisesRegex(ValueError, "uzavřenou eliptickou"):
            evolution_span(solution, "returns", 1, 1)

    def test_marsden_law_is_normalized_at_one_au(self):
        self.assertAlmostEqual(marsden_water_ice_law(1.0), 1.0, places=5)

    def test_nongravitational_axes_follow_rtn_frame(self):
        acceleration = nongravitational_acceleration(
            np.array([1.0, 0.0, 0.0]), np.array([0.0, .01, 0.0]),
            {"A1": 1e-8, "A2": 2e-8, "A3": -3e-8},
        )
        np.testing.assert_allclose(acceleration, [1e-8, 2e-8, -3e-8], rtol=2e-5)

    def test_dt_uses_earlier_osculating_distance(self):
        position = np.array([0.5, 0.0, 0.0])
        velocity = np.array([0.0, math.sqrt(MU_SUN * 3.0), 0.0])
        shifted_radius = delayed_heliocentric_radius(position, velocity, 30.0)
        self.assertGreater(shifted_radius, 0.5)
        symmetric = nongravitational_acceleration(
            position, velocity, {"A1": 1e-8},
        )
        delayed = nongravitational_acceleration(
            position, velocity, {"A1": 1e-8, "DT": 30.0},
        )
        self.assertLess(float(np.linalg.norm(delayed)), float(np.linalg.norm(symmetric)))

    def test_dt_preserves_radius_of_circular_orbit(self):
        radius = delayed_heliocentric_radius(
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, math.sqrt(MU_SUN), 0.0]),
            120.0,
        )
        self.assertAlmostEqual(radius, 1.0, places=10)

    def test_unphysical_dt_trial_fails_instead_of_returning_current_radius(self):
        with self.assertRaises(FloatingPointError):
            delayed_heliocentric_radius(
                np.array([1.0, 0.0, 0.0]),
                np.array([0.0, 1.0, 0.0]),
                1000.0,
            )

    def test_zero_length_propagation_returns_initial_state(self):
        state = StateVector(2451545.0, np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.01, 0.0]))
        engine = NBodyIntegrator(PLANETS)
        result = engine.propagate(state, np.array([2451545.0, 2451545.0]))
        self.assertEqual(result.shape, (2, 6))
        self.assertTrue(np.allclose(result[0], state.packed()))
        self.assertTrue(np.allclose(result[1], state.packed()))

    def test_one_year_circular_solar_orbit(self):
        initial = StateVector(
            epoch_jd=2451545.0,
            position_au=np.array([1.0, 0.0, 0.0]),
            velocity_au_per_day=np.array([0.0, math.sqrt(MU_SUN), 0.0]),
        )
        engine = NBodyIntegrator(_NoPlanets(), perturbers=())
        states = engine.propagate(initial, np.array([2451545.0, 2451545.0 + 365.2568983]))
        self.assertLess(float(np.linalg.norm(states[-1, :3] - initial.position_au)), 2e-6)
        self.assertLess(float(np.linalg.norm(states[-1, 3:] - initial.velocity_au_per_day)), 4e-8)

    def test_propagates_on_both_sides_of_epoch(self):
        initial = StateVector(
            epoch_jd=2451545.0,
            position_au=np.array([1.0, 0.0, 0.0]),
            velocity_au_per_day=np.array([0.0, math.sqrt(MU_SUN), 0.0]),
        )
        engine = NBodyIntegrator(_NoPlanets(), perturbers=())
        states = engine.propagate(initial, np.array([2451544.0, 2451545.0, 2451546.0]))
        self.assertEqual(states.shape, (3, 6))
        self.assertTrue(np.all(np.isfinite(states)))
        self.assertTrue(np.allclose(states[1], initial.packed()))

    def test_propagation_accepts_unsorted_duplicate_epochs(self):
        initial = StateVector(
            epoch_jd=2451545.0,
            position_au=np.array([1.0, 0.0, 0.0]),
            velocity_au_per_day=np.array([0.0, math.sqrt(MU_SUN), 0.0]),
        )
        engine = NBodyIntegrator(_NoPlanets(), perturbers=())
        epochs = np.array([2451546.0, 2451544.0, 2451546.0, 2451545.0])
        states = engine.propagate(initial, epochs)
        self.assertEqual(states.shape, (4, 6))
        np.testing.assert_allclose(states[0], states[2])
        np.testing.assert_allclose(states[3], initial.packed())


if __name__ == "__main__":
    unittest.main()

