"""Local cache of MPC astrometry."""
import gzip
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

from orbit_app import mpc_cache  # noqa: E402

try:
    from qt_support import qt_application  # noqa: E402
except ImportError:  # run as tests.<module> from the repository root
    from tests.qt_support import qt_application  # noqa: E402

ROWS = [
    {"permID": "1P", "obsTime": "1835-08-21T00:00:00Z", "ra": 80.1, "dec": 20.2,
     "stn": "000", "astCat": ""},
    {"permID": "1P", "obsTime": "2023-12-11T03:04:05.6Z", "ra": 120.5, "dec": -5.25,
     "mag": 21.3, "band": "G", "stn": "F51", "astCat": "Gaia3E",
     "rmsRA": 0.12, "rmsDec": 0.1},
]


FRAGMENT_LINE = (
    "0205P      b  C2008 09 24.14478 21 00 07.08 -06 31 43.5          19.6 Nr64084850"
)


class _TemporaryDataDirectory(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory()
        self._patch = mock.patch.dict(
            os.environ, {"ORBIT_SOLVER_DATA_DIR": self._directory.name},
        )
        self._patch.start()

    def tearDown(self):
        self._patch.stop()
        self._directory.cleanup()


class MpcCacheTests(_TemporaryDataDirectory):
    def test_round_trip_keeps_raw_rows_and_timestamp(self):
        stamp = datetime(2026, 10, 22, 12, 30, tzinfo=timezone.utc)
        path = mpc_cache.save("1P", ROWS, stamp)
        self.assertTrue(path.is_file())
        self.assertEqual(path.parent, mpc_cache.cache_directory())
        cached = mpc_cache.load("1P")
        self.assertIsNotNone(cached)
        self.assertEqual(cached.rows, ROWS)
        self.assertEqual(cached.downloaded_utc, stamp)
        observations = cached.observations()
        self.assertEqual([item.station for item in observations], ["000", "F51"])
        self.assertAlmostEqual(observations[1].ra_deg, 120.5)
        self.assertEqual(observations[1].catalog, "Gaia3E")

    def test_designation_is_normalized_but_objects_do_not_collide(self):
        mpc_cache.save("C/2020  F3", ROWS)
        self.assertIsNotNone(mpc_cache.load(" c/2020 f3 "))
        self.assertIsNone(mpc_cache.load("C_2020 F3"))
        self.assertNotEqual(mpc_cache.cache_path("C/2020 F3"), mpc_cache.cache_path("C_2020 F3"))

    def test_missing_or_damaged_cache_is_ignored(self):
        self.assertIsNone(mpc_cache.load("2P"))
        self.assertIsNone(mpc_cache.load(""))
        path = mpc_cache.cache_path("2P")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not gzip")
        self.assertIsNone(mpc_cache.load("2P"))
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write('{"format": 999, "designation": "2P", "rows": [{}]}')
        self.assertIsNone(mpc_cache.load("2P"))

    def test_fragment_records_round_trip(self):
        mpc_cache.save("205P", ROWS, mpc80_lines=[FRAGMENT_LINE], fragments_checked=True)
        cached = mpc_cache.load("205P")
        self.assertTrue(cached.fragments_checked)
        self.assertEqual(cached.mpc80_lines, (FRAGMENT_LINE,))
        observations = cached.observations()
        self.assertEqual(len(observations), 3)
        self.assertEqual(observations[-1].raw.get("mpc_fragment") or observations[-1].raw.get("mpc80")[11], "b")
        # An old cache file without the new keys still loads.
        mpc_cache.save("1P", ROWS)
        self.assertFalse(mpc_cache.load("1P").fragments_checked)

    def test_update_overwrites_and_remove_deletes(self):
        mpc_cache.save("1P", ROWS[:1])
        mpc_cache.save("1P", ROWS)
        self.assertEqual(len(mpc_cache.load("1P").rows), 2)
        self.assertEqual(list(mpc_cache.cache_directory().glob(".tmp-*")), [])
        self.assertTrue(mpc_cache.remove("1P"))
        self.assertIsNone(mpc_cache.load("1P"))


class MainWindowCacheTests(_TemporaryDataDirectory):
    def setUp(self):
        super().setUp()
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

        from orbit_app.main_window import MainWindow

        self._application = qt_application()
        self.window = MainWindow()
        self.window.mpc_network.observations = mock.Mock()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        super().tearDown()

    def test_cached_astrometry_is_loaded_without_network(self):
        mpc_cache.save(
            "1P", ROWS, datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc), fragments_checked=True,
        )
        self.window.search.setText("1P")
        self.window.load_observations(False)
        self.window.mpc_network.observations.assert_not_called()
        self.assertEqual(len(self.window.current_observations), 2)
        self.assertIn("2026-01-02 03:04", self.window.statusBar().currentMessage())
        self.assertTrue(self.window.mpc_refresh_button.isEnabled())

    def test_comet_cached_before_fragment_listing_is_downloaded_again(self):
        mpc_cache.save("1P", ROWS)            # older cache: listing never consulted
        self.window.search.setText("1P")
        self.window.load_observations(False)
        self.window.mpc_network.observations.assert_called_once_with("1P", False)

    def test_cached_fragment_records_are_loaded(self):
        principal = FRAGMENT_LINE.replace("0205P      b", "0205P      a")
        mpc_cache.save(
            "205P", ROWS[1:], mpc80_lines=[principal, FRAGMENT_LINE], fragments_checked=True,
        )
        self.window.search.setText("205P")
        self.window.load_observations(False)
        self.window.mpc_network.observations.assert_not_called()
        self.assertEqual(len(self.window.current_observations), 3)
        self.assertFalse(self.window.fragment_selector.isHidden())

    def test_update_and_uncached_objects_go_to_the_mpc(self):
        mpc_cache.save("1P", ROWS)
        self.window.search.setText("1P")
        self.window.load_observations(True)
        self.window.mpc_network.observations.assert_called_once_with("1P", False)
        self.assertFalse(self.window.mpc_refresh_button.isEnabled())
        with mock.patch("orbit_app.main_window.QMessageBox.warning"):
            self.window._observations_failed("offline")  # noqa: SLF001
        self.assertTrue(self.window.mpc_refresh_button.isEnabled())
        self.window.search.setText("2P")
        self.window.load_observations(False)
        self.assertEqual(self.window.mpc_network.observations.call_count, 2)


class FragmentListingTests(unittest.TestCase):
    """get-obs omits the lettered rows of a split comet; the text listing has them."""

    def test_link_and_fragment_lines(self):
        from orbit_app.main_window import (
            fragment_listing_link, fragment_listing_lines, wants_fragment_listing,
        )

        page = '<a href="../tmp2/205P.txt">Observations</a>'
        self.assertEqual(fragment_listing_link(page), "../tmp2/205P.txt")
        self.assertIsNone(fragment_listing_link("<html></html>"))
        unlabelled = "0205P        KC2015 09 18.09005 04 45 54.29 +08 34 37.0          15.6 Nd95370A77"
        lines = fragment_listing_lines("\n".join((unlabelled, FRAGMENT_LINE, "header")))
        self.assertEqual(lines, [FRAGMENT_LINE])
        self.assertTrue(wants_fragment_listing("205P"))
        self.assertTrue(wants_fragment_listing("P/2008 R6"))
        self.assertFalse(wants_fragment_listing("205P-A"))
        self.assertFalse(wants_fragment_listing("2024 AB"))

    def test_network_chain_adds_fragments(self):
        from PySide6.QtCore import QByteArray, QUrl
        from PySide6.QtNetwork import QNetworkReply

        from orbit_app.main_window import MpcNetworkClient

        qt_application()
        client = MpcNetworkClient()
        emitted = []
        client.success.connect(emitted.append)

        def reply(body: bytes, url: str = "https://www.minorplanetcenter.net/db_search/show_object"):
            item = mock.Mock()
            item.error.return_value = QNetworkReply.NoError
            item.readAll.return_value = QByteArray(body)
            item.url.return_value = QUrl(url)
            return item

        requested = []

        def fake_get(request):
            requested.append(request.url().toString())
            return mock.Mock()

        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(
            os.environ, {"ORBIT_SOLVER_DATA_DIR": directory},
        ), mock.patch.object(client.manager, "get", side_effect=fake_get):
            client._request_fragment_listing("205P", ROWS[1:], [])  # noqa: SLF001
            self.assertIn("object_id=205P", requested[0])
            client._listing_page_finished(  # noqa: SLF001
                reply(b'<a href="../tmp2/205P.txt">x</a>'), "205P", ROWS[1:], [],
            )
            self.assertEqual(requested[1], "https://www.minorplanetcenter.net/tmp2/205P.txt")
            client._listing_text_finished(  # noqa: SLF001
                reply(FRAGMENT_LINE.encode()), "205P", ROWS[1:], [],
            )
            self.assertEqual(len(emitted), 1)
            designation, rows, _confirmation, _cached = emitted[0]
            self.assertEqual((designation, len(rows)), ("205P", 1))
            cached = mpc_cache.load("205P")
            self.assertEqual(cached.mpc80_lines, (FRAGMENT_LINE,))
            # A failing page still delivers the get-obs rows.
            failing = reply(b"")
            failing.error.return_value = QNetworkReply.HostNotFoundError
            client._listing_page_finished(failing, "205P", ROWS[1:], ["row"])  # noqa: SLF001
            self.assertEqual(emitted[-1][1], ["row"])
            self.assertFalse(mpc_cache.load("205P").fragments_checked)


if __name__ == "__main__":
    unittest.main()
