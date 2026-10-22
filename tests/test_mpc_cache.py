"""Local cache of MPC astrometry."""
import gzip
import os
import tempfile
import unittest
from datetime import datetime, timezone
from unittest import mock

os.environ["ORBIT_SOLVER_LANGUAGE"] = "cs"

from orbit_app import mpc_cache  # noqa: E402

ROWS = [
    {"permID": "1P", "obsTime": "1835-08-21T00:00:00Z", "ra": 80.1, "dec": 20.2,
     "stn": "000", "astCat": ""},
    {"permID": "1P", "obsTime": "2023-12-11T03:04:05.6Z", "ra": 120.5, "dec": -5.25,
     "mag": 21.3, "band": "G", "stn": "F51", "astCat": "Gaia3E",
     "rmsRA": 0.12, "rmsDec": 0.1},
]


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
        from PySide6.QtWidgets import QApplication

        from orbit_app.main_window import MainWindow

        self._application = QApplication.instance() or QApplication([])
        self.window = MainWindow()
        self.window.mpc_network.observations = mock.Mock()

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        super().tearDown()

    def test_cached_astrometry_is_loaded_without_network(self):
        mpc_cache.save("1P", ROWS, datetime(2026, 1, 2, 3, 4, tzinfo=timezone.utc))
        self.window.search.setText("1P")
        self.window.load_observations(False)
        self.window.mpc_network.observations.assert_not_called()
        self.assertEqual(len(self.window.current_observations), 2)
        self.assertIn("2026-01-02 03:04", self.window.statusBar().currentMessage())
        self.assertTrue(self.window.mpc_refresh_button.isEnabled())

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


if __name__ == "__main__":
    unittest.main()
