from __future__ import annotations

import sys
import os
import json
import re
import hashlib
import html
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import numpy as np

# Matplotlib must not auto-select another installed Qt binding before PySide6.
os.environ.setdefault("QT_API", "pyside6")

from PySide6.QtCore import (
    QAbstractTableModel, QByteArray, QItemSelectionModel, QModelIndex, QObject,
    QProcess, QThread, Qt, QTimer, QUrl, Signal, Slot,
)
from PySide6.QtGui import QColor, QFont, QKeySequence, QShortcut
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkReply, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDateTimeEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QPlainTextEdit,
    QProgressBar,
    QScrollArea,
    QSlider,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import matplotlib.dates as mdates
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.projections import register_projection
from mpl_toolkits.mplot3d import Axes3D

from .astrometry import (
    fragment_counts, observation_fragment, primary_fragment, select_fragment,
)
from .domain import Observation, OrbitSolution, is_comet_designation
from .linkage import LINKAGE_RELAXED_NORMALIZED_RMS
from .orbit_engine import HISTORICAL_NONGRAVITATIONAL, NONGRAVITATIONAL_SPLIT_KEY
from .comet_dynamics import analyze_original_future_orbit
from .i18n import SUPPORTED_LANGUAGES, language, save_language, tr
from . import mpc_cache
from .mpc import MpcClient, MpcError, parse_mpc80
from .mpc_orbit import (
    OrbitParseError, cartesian_to_elements, mpc_identifier_choices, parse_mpc_orbit,
)
from .observatories import (
    observation_altaz_series, observation_geometry,
    observatory_details, station_offsets_gcrs,
)
from .orbit_fit import (
    OrbitFitCancelled, fit_orbit, fit_selection_count, initial_orbit_from_observations,
    parse_observation_time,
)
from .orbit_engine import MU_SUN, NBodyIntegrator, StateVector
from .orbit_evolution import (
    BODY_LABELS, ELEMENT_CENTERS, LONG_EVOLUTION_YEARS, EvolutionResult, HILL_RADII_AU,
    PLANET_LABELS, PLANET_MU, approximate_year, bound_centers, clip_span_to_coverage,
    planet_relation_series, with_impact_uncertainties,
    calculate_orbit_evolution, centered_snapshot, centered_solution, evolution_span,
    usable_coverage,
    format_evolution_mpc_report, format_evolution_report, return_solution,
)
from .impact import IMPACT_CHECK_DAYS, describe_impact, imminent_impact
from .planet_relations import RELATION_PLANETS, nearby_resonances, planet_period_years
from .planetary import DE440_MD5, DE440_URL, PLANETS, de440_path
from .photometry import estimate_absolute_magnitude, hg_phase_correction, power_law_magnitude
from .presentation import (
    LINKAGE_STATUS_LABELS, SKY_TRACK_HORIZONS,
    apparition_linkage_summary, apparition_observation_count_text,
    datetime_from_jd, equatorial_ra_ticks, format_dec_sexagesimal,
    format_ra_sexagesimal, observation_residual_score, observations_to_tsv,
    parse_output_epoch, perihelion_jd_from_solution, project_ecliptic_camera,
    sky_track_sample_count, solar_orbit_clip_radius, trajectory_return_masks,
)
from .preview_orbit import apparent_geometry, datetime_to_jd, demo_solution, heliocentric_xyz, orbital_curve_xyz


APP_STYLE = """
QMainWindow, QWidget { background: #0b1220; color: #dce7f3; font-size: 13px; }
QFrame#sidebar { background: #101a2b; border-right: 1px solid #23324a; }
QFrame#card { background: #111d30; border: 1px solid #243650; border-radius: 10px; }
QLabel#title { font-size: 25px; font-weight: 700; color: #f2f7ff; }
QLabel#muted { color: #8fa5bf; }
QLabel#value { font-size: 18px; font-weight: 650; color: #f5fbff; }
QLineEdit, QComboBox, QDateTimeEdit, QSpinBox, QDoubleSpinBox {
  background: #0b1423; border: 1px solid #30425e; border-radius: 7px;
  padding: 7px; selection-background-color: #2c82ff;
}
QPushButton { background: #1b2b43; border: 1px solid #314764; border-radius: 7px; padding: 8px 12px; }
QPushButton:hover { background: #243a59; }
QPushButton#primary { background: #1677ff; border-color: #348bff; color: white; font-weight: 650; }
QPushButton#primary:hover { background: #2d87ff; }
QPushButton#primary:disabled { color: #64758c; background: #162033; border-color: #26354a; }
QPushButton#solveOrbit {
  background: #176b57; border: 1px solid #2f9d7f; color: white;
  font-weight: 700; padding: 10px 12px;
}
QPushButton#solveOrbit:hover { background: #20836b; border-color: #49b998; }
QPushButton#solveOrbit:disabled { color: #64758c; background: #162033; border-color: #26354a; }
QPushButton:disabled { color: #64758c; background: #162033; border-color: #26354a; }
QTabWidget::pane { border: 1px solid #243650; border-radius: 8px; background: #0e1828; }
QTabBar::tab { background: #101a2b; color: #8fa5bf; padding: 10px 15px; margin-right: 2px; }
QTabBar::tab:selected { color: white; background: #17263c; border-bottom: 2px solid #2c82ff; }
QHeaderView::section { background: #17243a; color: #afc2d8; border: 0; border-right: 1px solid #293a52; padding: 7px; }
QTableView, QTableWidget { background: #0c1524; alternate-background-color: #101c2e; gridline-color: #20314a; border: 0; }
QStatusBar { background: #0d1727; color: #91a8c1; }
"""


class WideAxes3D(Axes3D):
    """Axes3D variant which does not force its viewport to a central square."""
    name = "wide3d"

    def apply_aspect(self, position=None) -> None:
        if position is None:
            position = self.get_position(original=True)
        self._set_position(position, "active")


register_projection(WideAxes3D)


class ObservationModel(QAbstractTableModel):
    columns = (
        (tr("Čas (UTC)"), "time"),
        ("RA [h m s]", "ra_deg"),
        ("Dec [° ′ ″]", "dec_deg"),
        (tr("Mag / pásmo"), "magnitude_band"),
        (tr("Stanice"), "station"),
        (tr("Frag."), "fragment"),
        (tr("Použít"), "excluded_from_fit"),
        ("O−C RA [″]", "residual_ra_arcsec"),
        ("O−C Dec [″]", "residual_dec_arcsec"),
        (tr("Ve fitu"), "used_in_fit"),
    )

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[Observation] = []

    def set_rows(self, rows: list[Observation]) -> None:
        self.beginResetModel()
        self.rows = rows
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.rows)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self.columns)

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or not (0 <= index.row() < len(self.rows)):
            return None
        field = self.columns[index.column()][1]
        observation = self.rows[index.row()]
        if field == "fragment":
            value = observation_fragment(observation)
        else:
            value = None if field == "magnitude_band" else getattr(observation, field)
        inactive_fragment = bool(observation.raw.get("inactive_fragment"))
        if role == Qt.ToolTipRole and field == "fragment":
            return tr(
                "Fragment rozpadlé komety podle MPC (sloupec 12 formátu 80 znaků, resp. "
                "přípona -B v ADES). Řádky bez označení patří hlavní složce.",
            )
        if role == Qt.ToolTipRole and field == "excluded_from_fit" and inactive_fragment:
            return tr(
                "Měření jiného fragmentu, než který se právě fituje; dostane O−C, ale "
                "nemá váhu.",
            )
        if (
            role == Qt.ForegroundRole and inactive_fragment
            and field not in ("residual_ra_arcsec", "residual_dec_arcsec")
        ):
            return QColor("#64758c")
        if role == Qt.CheckStateRole and field == "excluded_from_fit":
            return Qt.Unchecked if observation.excluded_from_fit else Qt.Checked
        if role == Qt.ToolTipRole and field == "excluded_from_fit":
            return observation.raw.get(
                "automatic_exclusion_reason",
                tr("Zaškrtnuto = měření je zahrnuto do dalšího fitu."),
            )
        if role == Qt.BackgroundRole and field in ("residual_ra_arcsec", "residual_dec_arcsec"):
            worst = observation_residual_score(observation)
            if worst is not None:
                if worst > 2.0:
                    return QColor("#ef6a6a")
                if worst > 1.0:
                    return QColor("#f0c85a")
                if worst > .5:
                    return QColor("#a7d98b")
                return QColor("#4fbf78")
        if role == Qt.ForegroundRole and field in ("residual_ra_arcsec", "residual_dec_arcsec"):
            return QColor("#07111f") if value is not None else None
        if role == Qt.ToolTipRole and field in ("residual_ra_arcsec", "residual_dec_arcsec"):
            return tr(
                "Barva podle většího z |O−C RA| a |O−C Dec|: zelená ≤0,5″; světle zelená "
                "≤1″; žlutá ≤2″; červená >2″.",
            )
        if role == Qt.DisplayRole:
            if field == "ra_deg":
                return "—" if value is None else format_ra_sexagesimal(float(value))
            if field == "dec_deg":
                return "—" if value is None else format_dec_sexagesimal(float(value))
            if field == "magnitude_band":
                if observation.magnitude is None:
                    return "—" if not observation.band else f"— {observation.band}"
                return f"{observation.magnitude:.2f} {observation.band}".rstrip()
            if field == "excluded_from_fit":
                return ""
            if field == "used_in_fit":
                return tr("ano") if value is True else (tr("ne") if value is False else "—")
            if isinstance(value, float):
                return f"{value:+.2f}" if field.startswith("residual_") else f"{value:.2f}"
            return "—" if value in (None, "") else str(value)
        if role == Qt.TextAlignmentRole and field in (
            "ra_deg", "dec_deg", "magnitude_band", "residual_ra_arcsec", "residual_dec_arcsec"
        ):
            return Qt.AlignRight | Qt.AlignVCenter
        return None

    def flags(self, index: QModelIndex) -> Qt.ItemFlag:
        flags = super().flags(index)
        if index.isValid() and self.columns[index.column()][1] == "excluded_from_fit":
            flags |= Qt.ItemIsUserCheckable | Qt.ItemIsEditable
        return flags

    def setData(self, index: QModelIndex, value, role: int = Qt.EditRole) -> bool:
        if not index.isValid() or self.columns[index.column()][1] != "excluded_from_fit":
            return False
        if role == Qt.CheckStateRole:
            observation = self.rows[index.row()]
            observation.excluded_from_fit = value != Qt.Checked
            observation.used_in_fit = None
            if value == Qt.Checked:
                observation.raw.pop("automatic_exclusion", None)
                observation.raw.pop("coherent_secondary_track", None)
                observation.raw.pop("automatic_exclusion_reason", None)
            self.dataChanged.emit(index, index, [Qt.CheckStateRole, Qt.DisplayRole])
            return True
        return False

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.columns[section][0]
        return super().headerData(section, orientation, role)

    def sort(self, column: int, order: Qt.SortOrder = Qt.AscendingOrder) -> None:
        if not (0 <= column < len(self.columns)):
            return
        field = self.columns[column][1]
        self.layoutAboutToBeChanged.emit()
        self.rows.sort(
            key=lambda row: (
                row.magnitude is None, row.magnitude
            ) if field == "magnitude_band" else (getattr(row, field) is None, getattr(row, field)),
            reverse=order == Qt.DescendingOrder,
        )
        self.layoutChanged.emit()


class MpcNetworkClient(QObject):
    success = Signal(object)
    error = Signal(str)
    orbit_success = Signal(object)
    orbit_error = Signal(str)
    identifier_success = Signal(object)
    identifier_error = Signal(object)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.manager = QNetworkAccessManager(self)
        self._active: set[QNetworkReply] = set()

    def observations(self, designation: str, confirmation_page: bool) -> None:
        self._request_observations(designation, confirmation_page, allow_fallback=True)

    def _request_observations(
        self, designation: str, confirmation_page: bool, allow_fallback: bool,
    ) -> None:
        endpoint = "/get-obs-neocp" if confirmation_page else "/get-obs"
        key = "trksubs" if confirmation_page else "desigs"
        payload = {
            key: [designation],
            "output_format": ["ADES_DF"],
            "ades_version": "2022",
        }
        request = QNetworkRequest(QUrl(f"https://data.minorplanetcenter.net/api{endpoint}"))
        request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
        request.setRawHeader(b"User-Agent", b"OrbitSolver/1.0")
        request.setTransferTimeout(30_000)
        reply = self.manager.sendCustomRequest(
            request,
            b"GET",
            QByteArray(json.dumps(payload).encode("utf-8")),
        )
        self._active.add(reply)
        reply.finished.connect(
            lambda: self._finished(
                reply, designation, confirmation_page, allow_fallback
            )
        )

    def orbit(self, designation: str) -> None:
        request = QNetworkRequest(QUrl("https://data.minorplanetcenter.net/api/get-orb"))
        request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
        request.setRawHeader(b"User-Agent", b"OrbitSolver/1.0")
        request.setTransferTimeout(30_000)
        payload = QByteArray(json.dumps({"desig": designation}).encode("utf-8"))
        reply = self.manager.sendCustomRequest(request, b"GET", payload)
        self._active.add(reply)
        reply.finished.connect(lambda: self._orbit_finished(reply, designation))

    def identifier(self, designation: str) -> None:
        request = QNetworkRequest(
            QUrl("https://data.minorplanetcenter.net/api/query-identifier")
        )
        request.setHeader(QNetworkRequest.ContentTypeHeader, "application/json")
        request.setRawHeader(b"User-Agent", b"OrbitSolver/1.0")
        request.setTransferTimeout(30_000)
        payload = QByteArray(json.dumps({"ids": [designation]}).encode("utf-8"))
        reply = self.manager.sendCustomRequest(request, b"GET", payload)
        self._active.add(reply)
        reply.finished.connect(lambda: self._identifier_finished(reply, designation))

    def _finished(
        self, reply: QNetworkReply, designation: str, confirmation_page: bool,
        allow_fallback: bool,
    ) -> None:
        self._active.discard(reply)
        try:
            if reply.error() != QNetworkReply.NoError:
                if allow_fallback:
                    self._request_observations(
                        designation, not confirmation_page, allow_fallback=False
                    )
                    return
                self.error.emit(tr("MPC dotaz selhal: {0}", reply.errorString()))
                return
            try:
                body = json.loads(bytes(reply.readAll()).decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if allow_fallback:
                    self._request_observations(
                        designation, not confirmation_page, allow_fallback=False
                    )
                    return
                self.error.emit(tr("MPC vrátil nečitelnou odpověď: {0}", exc))
                return
            raw_rows = MpcClient._extract_rows(body)
            rows = [MpcClient._to_observation(row, designation) for row in raw_rows]
            if not rows:
                if allow_fallback:
                    self._request_observations(
                        designation, not confirmation_page, allow_fallback=False
                    )
                    return
                self.error.emit(
                    tr(
                        "Pro objekt „{0}“ nebyla nalezena pozorování ani v běžném MPC, ani na "
                        "NEOCP/PCCP.",
                        designation,
                    )
                )
                return
            if not confirmation_page:
                try:
                    mpc_cache.save(designation, raw_rows)
                except OSError:
                    pass  # an unwritable data folder must not block the download
            self.success.emit((designation, rows, confirmation_page, None))
        finally:
            reply.deleteLater()

    def _orbit_finished(self, reply: QNetworkReply, designation: str) -> None:
        self._active.discard(reply)
        try:
            if reply.error() != QNetworkReply.NoError:
                self.orbit_error.emit(tr("MPC orbitální dotaz selhal: {0}", reply.errorString()))
                return
            try:
                body = json.loads(bytes(reply.readAll()).decode("utf-8"))
                solution = parse_mpc_orbit(body, designation)
            except (UnicodeDecodeError, json.JSONDecodeError, OrbitParseError) as exc:
                if re.fullmatch(r"D/1993\s+F2-[A-HJ-NP-W]", designation, re.I):
                    self._request_sl9_horizons_metadata(designation)
                    return
                if re.fullmatch(r"D/1993\s+F2", designation, re.I):
                    # MPC has astrometry for the disrupted comet but no single
                    # cartesian orbit.  Fragment G supplies a nearby historical
                    # state from which the aggregate astrometry can be refitted.
                    self._request_sl9_horizons_metadata(
                        designation, self._sl9_spk_id("D/1993 F2-G"),
                    )
                    return
                self.orbit_error.emit(str(exc))
                return
            self.orbit_success.emit(solution)
        finally:
            reply.deleteLater()

    @staticmethod
    def _sl9_spk_id(designation: str) -> int:
        letter = designation.strip().upper().rsplit("-", 1)[-1]
        letters = "ABCDEFGHJKLMNPQRSTUVW"
        return 1000181 + letters.index(letter)

    def _request_sl9_horizons_metadata(
        self, designation: str, spk_id: int | None = None,
    ) -> None:
        spk_id = self._sl9_spk_id(designation) if spk_id is None else spk_id
        url = self._horizons_url((
            ("format", "json"), ("COMMAND", f"'DES={spk_id};'"),
            ("OBJ_DATA", "'YES'"), ("MAKE_EPHEM", "'NO'"),
        ))
        request = QNetworkRequest(url)
        request.setRawHeader(b"User-Agent", b"OrbitSolver/1.0")
        request.setTransferTimeout(30_000)
        reply = self.manager.get(request); self._active.add(reply)
        reply.finished.connect(
            lambda: self._sl9_metadata_finished(reply, designation, spk_id)
        )

    def _sl9_metadata_finished(
        self, reply: QNetworkReply, designation: str, spk_id: int,
    ) -> None:
        self._active.discard(reply)
        try:
            if reply.error() != QNetworkReply.NoError:
                self.orbit_error.emit(tr("JPL Horizons dotaz selhal: {0}", reply.errorString()))
                return
            body = json.loads(bytes(reply.readAll()).decode("utf-8"))
            result = str(body.get("result") or "")
            match = re.search(r"\bEPOCH=\s*([0-9.]+)", result)
            if match is None:
                self.orbit_error.emit(tr("JPL Horizons nevrátil epochu fragmentu SL9."))
                return
            self._request_sl9_horizons_vector(designation, spk_id, float(match.group(1)))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            self.orbit_error.emit(tr("JPL Horizons vrátil nečitelná data: {0}", exc))
        finally:
            reply.deleteLater()

    def _request_sl9_horizons_vector(
        self, designation: str, spk_id: int, epoch_jd: float,
    ) -> None:
        parameters = (
            ("format", "json"), ("COMMAND", f"'DES={spk_id};'"),
            ("OBJ_DATA", "'YES'"), ("MAKE_EPHEM", "'YES'"),
            ("EPHEM_TYPE", "'VECTORS'"), ("CENTER", "'500@10'"),
            ("TLIST", f"'{epoch_jd:.8f}'"), ("TLIST_TYPE", "'JD'"),
            ("OUT_UNITS", "'AU-D'"), ("REF_PLANE", "'ECLIPTIC'"),
            ("VEC_TABLE", "'2'"), ("VEC_CORR", "'NONE'"),
        )
        url = self._horizons_url(parameters)
        request = QNetworkRequest(url)
        request.setRawHeader(b"User-Agent", b"OrbitSolver/1.0")
        request.setTransferTimeout(30_000)
        reply = self.manager.get(request); self._active.add(reply)
        reply.finished.connect(
            lambda: self._sl9_vector_finished(reply, designation, epoch_jd)
        )

    @staticmethod
    def _horizons_url(parameters: tuple[tuple[str, str], ...]) -> QUrl:
        # Horizons requires literal quotes around several values. QUrlQuery
        # leaves apostrophes unescaped, which this API rejects as a bad URL.
        encoded = urlencode(parameters).encode("ascii")
        return QUrl.fromEncoded(
            b"https://ssd.jpl.nasa.gov/api/horizons.api?" + encoded
        )

    def _sl9_vector_finished(
        self, reply: QNetworkReply, designation: str, epoch_jd: float,
    ) -> None:
        self._active.discard(reply)
        try:
            if reply.error() != QNetworkReply.NoError:
                self.orbit_error.emit(tr("JPL Horizons dotaz selhal: {0}", reply.errorString()))
                return
            body = json.loads(bytes(reply.readAll()).decode("utf-8"))
            result = str(body.get("result") or "")
            representative_parent = bool(
                re.fullmatch(r"D/1993\s+F2", designation, re.I)
            )
            self.orbit_success.emit(self._parse_sl9_horizons_vector(
                result, designation, epoch_jd, representative_parent,
            ))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError, OrbitParseError) as exc:
            self.orbit_error.emit(tr("JPL Horizons stav SL9 nelze načíst: {0}", exc))
        finally:
            reply.deleteLater()

    @staticmethod
    def _parse_sl9_horizons_vector(
        result: str, designation: str, epoch_jd: float,
        representative_parent: bool = False,
    ) -> OrbitSolution:
        # OBJ_DATA contains another cartesian state in the ICRF frame.  Only
        # the requested AU/day ecliptic vector between the ephemeris sentinels
        # is suitable for the application's J2000-ecliptic dynamics.
        ephemeris = re.search(r"\$\$SOE(.*?)\$\$EOE", result, re.S)
        if ephemeris is None:
            raise ValueError(tr("vektor není ve výsledku"))
        block = ephemeris.group(1)
        number = r"([+\-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+\-]?\d+)?)"
        position = re.search(
            rf"X\s*=\s*{number}\s+Y\s*=\s*{number}\s+Z\s*=\s*{number}", block
        )
        velocity = re.search(
            rf"VX\s*=\s*{number}\s+VY\s*=\s*{number}\s+VZ\s*=\s*{number}", block
        )
        if position is None or velocity is None:
            raise ValueError(tr("vektor není ve výsledku"))
        state = np.array([float(value) for value in (*position.groups(), *velocity.groups())])
        a, e, inc, node, arg_peri, mean = cartesian_to_elements(state[:3], state[3:])
        name_match = re.search(r"Target body name:\s*([^\r\n{]+)", result)
        display = (
            "D/1993 F2 (Shoemaker–Levy 9)"
            if representative_parent else
            name_match.group(1).strip() if name_match else designation
        )
        return OrbitSolution(
            designation=designation, display_name=display, epoch_jd=epoch_jd,
            semi_major_axis_au=a, eccentricity=e, inclination_deg=inc,
            node_deg=node, arg_peri_deg=arg_peri, mean_anomaly_deg=mean,
            photometric_exponent=4.0, phase_correction=False,
            position_au=tuple(state[:3]), velocity_au_per_day=tuple(state[3:]),
            data_source=(
                tr("JPL Horizons (reprezentativní počáteční stav fragmentu G)")
                if representative_parent else "JPL Horizons"
            ),
        )

    def _identifier_finished(self, reply: QNetworkReply, designation: str) -> None:
        self._active.discard(reply)
        try:
            if reply.error() != QNetworkReply.NoError:
                self.identifier_error.emit((designation, reply.errorString()))
                return
            try:
                body = json.loads(bytes(reply.readAll()).decode("utf-8"))
                if isinstance(body, dict):
                    record = body.get(designation)
                    if record is None and len(body) == 1:
                        record = next(iter(body.values()))
                    record = record or {}
                else:
                    record = {}
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.identifier_error.emit((designation, str(exc)))
                return
            self.identifier_success.emit((designation, record))
        finally:
            reply.deleteLater()


class OrbitFitProgressDialog(QDialog):
    cancel_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Výpočet dráhy"))
        self.setModal(True)
        self.setMinimumWidth(620)
        self._started = time.monotonic()
        self._current = 0
        self._total = 0
        self._running = True

        self.phase = QLabel(tr("Připravuji výpočet…"))
        self.phase.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.timing = QLabel(tr("Uplynulo: 0 s • zbývající čas zatím nelze odhadnout"))
        self.timing.setObjectName("muted")
        self.note = QLabel(
            tr(
                "Odhad je orientační. Dlouhé integrační větve mohou trvat různě dlouho; "
                "zrušení se projeví po návratu právě běžící integrace.",
            )
        )
        self.note.setWordWrap(True)
        self.note.setObjectName("muted")
        self.cancel_button = QPushButton(tr("Zrušit výpočet"))
        self.cancel_button.clicked.connect(self._request_cancel)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)
        layout = QVBoxLayout(self)
        layout.addWidget(self.phase)
        layout.addWidget(self.progress)
        layout.addWidget(self.timing)
        layout.addWidget(self.note)
        layout.addLayout(buttons)

        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self._update_timing)
        self.timer.start()

    @staticmethod
    def _duration(seconds: float) -> str:
        value = max(0, int(round(seconds)))
        hours, remainder = divmod(value, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours} h {minutes:02d} min"
        if minutes:
            return f"{minutes} min {secs:02d} s"
        return f"{secs} s"

    @Slot(str, int, int)
    def update_progress(self, label: str, current: int, total: int) -> None:
        self.phase.setText(label)
        self._current = max(0, int(current))
        self._total = max(0, int(total))
        if self._total > 0:
            self.progress.setRange(0, self._total)
            # ``current`` denotes the step which has just started.
            self.progress.setValue(min(self._total, max(0, self._current - 1)))
        else:
            self.progress.setRange(0, 0)
        self._update_timing()

    @Slot()
    def _update_timing(self) -> None:
        elapsed = time.monotonic() - self._started
        completed = max(0, self._current - 1)
        if self._total > 0 and completed > 0 and completed < self._total:
            remaining = elapsed / completed * (self._total - completed)
            detail = tr("hrubý odhad zbývá: {0}", self._duration(remaining))
            step = tr("krok {0} z nejvýše {1}", self._current, self._total)
        elif self._total > 0 and self._current >= self._total:
            detail = tr("dokončuji poslední krok")
            step = tr("krok {0} z {1}", self._total, self._total)
        else:
            detail = tr("zbývající čas zatím nelze odhadnout")
            step = tr("příprava")
        self.timing.setText(
            tr("{0} • uplynulo: {1} • {2}", step, self._duration(elapsed), detail)
        )

    @Slot()
    def _request_cancel(self) -> None:
        if not self._running:
            return
        self.cancel_button.setEnabled(False)
        self.cancel_button.setText(tr("Ruším…"))
        self.phase.setText(
            tr("Čekám na bezpečné ukončení právě probíhající integrace…")
        )
        self.cancel_requested.emit()

    @Slot()
    def finish(self) -> None:
        self._running = False
        self.timer.stop()
        self.accept()

    def reject(self) -> None:
        if self._running:
            self._request_cancel()
            return
        super().reject()


class OrbitFitWorker(QObject):
    success = Signal(object)
    error = Signal(str)
    progress = Signal(str, int, int)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self, observations: list[Observation], initial: OrbitSolution | None,
        constraints: dict[str, float] | None = None,
        truncate_years: float | None = None,
        nongravitational: dict[str, float] | None = None,
        compute_original_future: bool = False,
        linkage_options: dict[str, bool] | None = None,
    ) -> None:
        super().__init__()
        self.observations = observations
        self.initial = initial
        self.constraints = constraints
        self.truncate_years = truncate_years
        self.nongravitational = nongravitational
        self.compute_original_future = compute_original_future
        self.linkage_options = dict(linkage_options or {})
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    @Slot()
    def run(self) -> None:
        # The table model keeps showing ``self.observations`` while the fit
        # runs.  Fit independent copies and hand them back only on success, so
        # the GUI thread never sees half-written trial O-C values and a failed
        # or cancelled run cannot leave its state behind.
        working = [
            replace(item, raw=dict(item.raw)) for item in self.observations
        ]
        initial = replace(self.initial) if self.initial is not None else None
        try:
            result = fit_orbit(
                working, initial, self.constraints, self.truncate_years,
                self.nongravitational,
                progress_callback=self.progress.emit,
                cancel_callback=self.is_cancel_requested,
                historical_nongravitational=bool(
                    self.linkage_options.get("historical_nongravitational")
                ),
                accept_model_tension=bool(self.linkage_options.get("accept_model_tension")),
            )
            solution = result[0]
            if self.compute_original_future:
                if self.is_cancel_requested():
                    raise OrbitFitCancelled()
                self.progress.emit(
                    tr("Integruji původní a budoucí barycentrickou dráhu do 250 AU…"),
                    0, 0,
                )
                try:
                    analysis = analyze_original_future_orbit(solution)
                except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                    analysis = {"error": str(exc)}
                if self.is_cancel_requested():
                    raise OrbitFitCancelled()
                solution = replace(solution, dynamical_1a=analysis)
                result = (solution, result[1], result[2])
            # Formal collision check: does the fitted orbit hit a planet or
            # the Moon within the next year?  (Longer horizons: Orbit evolution.)
            self.progress.emit(
                tr("Kontroluji, zda dráha nevede ke srážce s planetou…"), 0, 0,
            )
            try:
                prediction = imminent_impact(
                    solution, cancel_callback=self.is_cancel_requested,
                )
            except (ValueError, RuntimeError, FloatingPointError, OverflowError):
                prediction = None
            if self.is_cancel_requested():
                raise OrbitFitCancelled()
            if prediction is not None:
                solution = replace(solution, impact=prediction)
                result = (solution, result[1], result[2])
            if (
                solution.nongravitational_parameters
                and solution.position_au is not None
                and solution.velocity_au_per_day is not None
            ):
                # Warm the shared preview cache while this worker thread is
                # already running.  The GUI therefore stays responsive when
                # an old comet is first displayed at the present epoch.
                try:
                    heliocentric_xyz(
                        solution,
                        np.array([datetime_to_jd(datetime.now(timezone.utc))]),
                    )
                except (ValueError, RuntimeError, FloatingPointError, OverflowError):
                    pass
            self.success.emit((result, working))
        except OrbitFitCancelled:
            self.cancelled.emit()
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


def apply_fitted_observations(
    targets: list[Observation], fitted: list[Observation],
) -> None:
    """Copy fit results from worker copies back onto the displayed rows."""
    for target, source in zip(targets, fitted):
        target.residual_ra_arcsec = source.residual_ra_arcsec
        target.residual_dec_arcsec = source.residual_dec_arcsec
        target.used_in_fit = source.used_in_fit
        target.excluded_from_fit = source.excluded_from_fit
        target.raw.clear()
        target.raw.update(source.raw)


class PreliminaryOrbitWorker(QObject):
    success = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self, observations: list[Observation], method: str,
        perihelion_distance_au: float | None,
    ) -> None:
        super().__init__()
        self.observations = observations
        self.method = method
        self.perihelion_distance_au = perihelion_distance_au

    @Slot()
    def run(self) -> None:
        try:
            if self.method == "systematic" and fit_selection_count(self.observations, None) >= 6:
                solution, _, _ = fit_orbit(self.observations, None)
            else:
                solution = initial_orbit_from_observations(
                    self.observations, self.method, self.perihelion_distance_au,
                    minimum_observations=3,
                )
            self.success.emit((solution, self.method, self.perihelion_distance_au))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class EpochPropagationWorker(QObject):
    success = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, source: OrbitSolution, target_jd: float) -> None:
        super().__init__()
        self.source = source
        self.target_jd = target_jd

    @Slot()
    def run(self) -> None:
        try:
            source = self.source
            if source.position_au is None or source.velocity_au_per_day is None:
                raise ValueError(tr("Dráha nemá kartézský stav."))
            engine = NBodyIntegrator(
                PLANETS,
                nongravitational_parameters=source.nongravitational_parameters,
            )
            state = engine.propagate(
                StateVector(
                    source.epoch_jd, np.asarray(source.position_au, dtype=float),
                    np.asarray(source.velocity_au_per_day, dtype=float),
                ),
                np.array([self.target_jd]),
            )[0]
            elements = cartesian_to_elements(state[:3], state[3:])
            self.success.emit(replace(
                source,
                epoch_jd=self.target_jd,
                semi_major_axis_au=elements[0], eccentricity=elements[1],
                inclination_deg=elements[2], node_deg=elements[3],
                arg_peri_deg=elements[4], mean_anomaly_deg=elements[5],
                position_au=tuple(float(value) for value in state[:3]),
                velocity_au_per_day=tuple(float(value) for value in state[3:]),
                state_covariance=None, element_uncertainties=None,
            ))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class OrbitEvolutionWorker(QObject):
    success = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(
        self, solution: OrbitSolution, mode: str, past: int, future: int,
        span: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        self.solution = solution
        self.mode = mode
        self.past = past
        self.future = future
        self.span = span

    @Slot()
    def run(self) -> None:
        try:
            result = calculate_orbit_evolution(
                self.solution, self.mode, self.past, self.future, span=self.span,
            )
            # Impact location uncertainty from the fit covariance (Monte Carlo).
            self.success.emit(with_impact_uncertainties(self.solution, result))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


class PlanetRelationWorker(QObject):
    """Background computation of MOID/Tisserand/... series for one planet."""
    success = Signal(object)
    error = Signal(str)
    finished = Signal()

    def __init__(self, result: EvolutionResult, planet: str) -> None:
        super().__init__()
        self.result = result
        self.planet = planet

    @Slot()
    def run(self) -> None:
        try:
            self.success.emit(planet_relation_series(self.result, self.planet))
        except Exception as exc:
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


def _ephemeris_coordinates(
    solution: OrbitSolution, jd: np.ndarray, observer_offsets: np.ndarray | None,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    """Return light-time-corrected astrometric coordinates without Qt state."""
    geom = apparent_geometry(solution, jd, observer_offsets)
    ra = np.asarray(geom["ra"], dtype=float)
    dec = np.asarray(geom["dec"], dtype=float)
    delta = np.asarray(geom["delta"], dtype=float)
    return geom, ra, dec, delta


def _ephemeris_positional_uncertainty(
    solution: OrbitSolution, jd: np.ndarray, observer_offsets: np.ndarray | None,
    nominal_dec: np.ndarray,
) -> np.ndarray | None:
    """Propagate the fitted state covariance to a sky-plane 1-sigma radius."""
    if (
        solution.state_covariance is None or solution.position_au is None
        or solution.velocity_au_per_day is None
    ):
        return None
    covariance = np.asarray(solution.state_covariance, dtype=float)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        return None
    covariance = (covariance + covariance.T) / 2.0
    state = np.asarray((*solution.position_au, *solution.velocity_au_per_day), dtype=float)
    jacobian = np.zeros((len(jd), 2, 6), dtype=float)
    try:
        for column in range(6):
            sigma = np.sqrt(max(0.0, float(covariance[column, column])))
            floor = 1e-9 if column < 3 else 1e-11
            step = max(sigma * .1, abs(float(state[column])) * 1e-7, floor)
            directions = []
            for sign in (1.0, -1.0):
                perturbed = state.copy(); perturbed[column] += sign * step
                elements = cartesian_to_elements(perturbed[:3], perturbed[3:])
                trial = OrbitSolution(
                    designation=solution.designation, epoch_jd=solution.epoch_jd,
                    semi_major_axis_au=elements[0], eccentricity=elements[1],
                    inclination_deg=elements[2], node_deg=elements[3],
                    arg_peri_deg=elements[4], mean_anomaly_deg=elements[5],
                    position_au=tuple(perturbed[:3]),
                    velocity_au_per_day=tuple(perturbed[3:]),
                    nongravitational_parameters=solution.nongravitational_parameters,
                )
                _, trial_ra, trial_dec, _ = _ephemeris_coordinates(
                    trial, jd, observer_offsets
                )
                directions.append((trial_ra, trial_dec))
            delta_ra = (directions[0][0] - directions[1][0] + 180.0) % 360.0 - 180.0
            jacobian[:, 0, column] = (
                delta_ra * np.cos(np.radians(nominal_dec)) * 3600.0 / (2.0 * step)
            )
            jacobian[:, 1, column] = (
                (directions[0][1] - directions[1][1]) * 3600.0 / (2.0 * step)
            )
        sky_covariance = np.einsum("nik,kl,njl->nij", jacobian, covariance, jacobian)
        eigenvalues = np.linalg.eigvalsh(sky_covariance)
        return np.sqrt(np.maximum(0.0, eigenvalues[:, -1]))
    except (ValueError, FloatingPointError, np.linalg.LinAlgError, ZeroDivisionError):
        return None


class EphemerisUncertaintyWorker(QObject):
    success = Signal(int, object)
    error = Signal(int, str)
    finished = Signal(int)

    def __init__(
        self, request_id: int, solution: OrbitSolution, jd: np.ndarray,
        observer_offsets: np.ndarray | None, nominal_dec: np.ndarray,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.solution = solution
        self.jd = np.asarray(jd, dtype=float).copy()
        self.observer_offsets = (
            None if observer_offsets is None else np.asarray(observer_offsets, dtype=float).copy()
        )
        self.nominal_dec = np.asarray(nominal_dec, dtype=float).copy()

    @Slot()
    def run(self) -> None:
        try:
            result = _ephemeris_positional_uncertainty(
                self.solution, self.jd, self.observer_offsets, self.nominal_dec,
            )
            # The twelve short-lived covariance variants may displace the
            # nominal comet from the small shared N-body cache.  Reinsert its
            # requested ephemeris while still off the GUI thread, so returning
            # to this or another preview never repeats the long integration.
            _ephemeris_coordinates(self.solution, self.jd, self.observer_offsets)
            self.success.emit(self.request_id, result)
        except Exception as exc:
            self.error.emit(self.request_id, str(exc))
        finally:
            self.finished.emit(self.request_id)


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, projection: str | None = None) -> None:
        self.figure = Figure(facecolor="#0e1828", tight_layout=True)
        self.axes = self.figure.add_subplot(111, projection=projection)
        super().__init__(self.figure)
        self.setMinimumHeight(420)
        self.mpl_connect("scroll_event", self._scroll_zoom)

    def _scroll_zoom(self, event) -> None:
        if event.inaxes is not self.axes:
            return
        factor = 0.78 if event.button == "up" else 1.28
        x_min, x_max = self.axes.get_xlim()
        y_min, y_max = self.axes.get_ylim()
        x_center = event.xdata if event.xdata is not None else (x_min + x_max) / 2
        y_center = event.ydata if event.ydata is not None else (y_min + y_max) / 2
        self.axes.set_xlim(x_center + (x_min - x_center) * factor, x_center + (x_max - x_center) * factor)
        self.axes.set_ylim(y_center + (y_min - y_center) * factor, y_center + (y_max - y_center) * factor)
        if hasattr(self.axes, "get_zlim"):
            z_min, z_max = self.axes.get_zlim()
            z_center = (z_min + z_max) / 2
            self.axes.set_zlim(z_center + (z_min - z_center) * factor, z_center + (z_max - z_center) * factor)
        self.draw_idle()

    def style_axes(self) -> None:
        self.axes.set_facecolor("#0e1828")
        self.axes.tick_params(colors="#91a8c1")
        for spine in self.axes.spines.values():
            spine.set_color("#30425e")
        self.axes.xaxis.label.set_color("#aebfd3")
        self.axes.yaxis.label.set_color("#aebfd3")
        self.axes.title.set_color("#edf5ff")


class OrbitEvolutionPanel(QWidget):
    """Kinoshita-style overview of osculating elements across returns."""

    GRAPH_METRICS = {
        "q": ("perihelion_au", tr("perihelová vzdálenost q [AU]"), tr("perihelové vzdálenosti")),
        "a": ("semi_major_axis_au", tr("velká poloosa a [AU]"), tr("velké poloosy")),
        "e": ("eccentricity", tr("excentricita e"), tr("excentricity")),
        "period": ("period_years", tr("oběžná doba P [roky]"), tr("oběžné doby")),
        "reciprocal_a": ("reciprocal_a", "1/a [AU⁻¹]", tr("reciproké velké poloosy")),
        "inclination": ("inclination_deg", tr("sklon i [°]"), tr("sklonu dráhy")),
        "node": ("node_deg", tr("délka výstupného uzlu Ω [°]"), tr("délky výstupného uzlu")),
        "arg_peri": ("arg_peri_deg", tr("argument perihelu ω [°]"), tr("argumentu perihelu")),
    }

    # Quantities describing a heliocentric orbit relative to a planet:
    # key -> (axis label, title fragment).
    RELATION_METRICS = {
        "distance": (tr("vzdálenost od planety [AU]"), tr("vzdálenosti od planety")),
        "moid": (tr("MOID [AU]"), tr("MOID")),
        "tisserand": (tr("Tisserandův parametr T"), tr("Tisserandova parametru")),
        "period_ratio": (tr("poměr oběžných dob P / P planety"), tr("poměru oběžných dob")),
        "nodes": (
            tr("vzdálenost uzlu od dráhy planety [AU]"), tr("uzlových vzdáleností"),
        ),
    }

    show_return = Signal(object, object)
    evolution_ready = Signal(object, object)

    def __init__(self, solution: OrbitSolution) -> None:
        super().__init__()
        self.solution = solution
        self.result: EvolutionResult | None = None
        self._centered_cache: dict[tuple[str, str], tuple] = {}
        self._relation_cache: dict[str, object] = {}
        self._relation_thread: QThread | None = None
        self._relation_worker: PlanetRelationWorker | None = None
        self._relation_pending: str | None = None
        self.thread: QThread | None = None
        self.worker: OrbitEvolutionWorker | None = None

        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem(tr("Podle počtu návratů"), "returns")
        self.mode.addItem(tr("Podle počtu let"), "years")
        self.past_label = QLabel(tr("Předchozí návraty:"))
        self.past = QSpinBox(); self.past.setRange(0, 500); self.past.setValue(5)
        self.future_label = QLabel(tr("Budoucí návraty:"))
        self.future = QSpinBox(); self.future.setRange(0, 500); self.future.setValue(5)
        self.calculate = QPushButton(tr("Spustit integraci"))
        self.calculate.setObjectName("primary")
        self.calculate.clicked.connect(self.start)
        self.mode.currentIndexChanged.connect(self._mode_changed)
        # Shows the calendar years a choice of returns/years really means, so
        # "200 returns" typed instead of "200 years" is obvious before starting.
        self.span_hint = QLabel()
        self.span_hint.setObjectName("muted")
        self.past.valueChanged.connect(self._update_span_hint)
        self.future.valueChanged.connect(self._update_span_hint)
        controls.addWidget(QLabel(tr("Rozsah:"))); controls.addWidget(self.mode)
        controls.addWidget(self.past_label); controls.addWidget(self.past)
        controls.addWidget(self.future_label); controls.addWidget(self.future)
        controls.addWidget(self.span_hint)
        controls.addWidget(self.calculate); controls.addStretch()
        layout.addLayout(controls)

        self.note = QLabel(
            tr(
                "Výpočet se spouští pouze tlačítkem. Návrat 0 je perihelový průchod "
                "nejbližší epoše dráhy.",
            )
        )
        self.note.setObjectName("muted"); self.note.setWordWrap(True)
        layout.addWidget(self.note)
        self.source_note = QLabel()
        self.source_note.setStyleSheet("color: #bcd4ed; font-weight: 600;")
        self.source_note.setWordWrap(True)
        layout.addWidget(self.source_note)

        output_controls = QHBoxLayout()
        self.graph_metric = QComboBox()
        self.graph_metric.currentIndexChanged.connect(self._draw_result)
        # What the plot and the left table describe: heliocentric elements,
        # the relation to a planet, or planetocentric elements while the body
        # is temporarily captured (offered only when a capture was detected).
        self.element_center = QComboBox()
        self.element_center.setToolTip(
            tr(
                "Vztah k planetě ukazuje vzdálenost, MOID, Tisserandův parametr, poměr "
                "oběžných dob a uzlové vzdálenosti. Planetocentrické elementy se nabízejí "
                "jen pro planetu, která těleso dočasně zachytila.",
            )
        )
        self._populate_views()
        self._populate_metrics("elements")
        self.element_center.currentIndexChanged.connect(self._center_changed)
        self.report_format = QComboBox()
        self.report_format.addItem(tr("TXT: čitelný přehled"), "readable")
        self.report_format.addItem("TXT: MPC 1-line", "mpc")
        self.copy_report = QPushButton(tr("Kopírovat TXT"))
        self.save_report = QPushButton(tr("Uložit TXT…"))
        self.copy_report.setEnabled(False); self.save_report.setEnabled(False)
        self.copy_report.clicked.connect(self._copy_report)
        self.save_report.clicked.connect(self._save_report)
        output_controls.addWidget(self.graph_metric)
        output_controls.addWidget(self.element_center)
        output_controls.addStretch()
        output_controls.addWidget(self.report_format)
        output_controls.addWidget(self.copy_report); output_controls.addWidget(self.save_report)
        layout.addLayout(output_controls)

        splitter = QSplitter(Qt.Vertical)
        self.canvas = PlotCanvas(); self.canvas.setMinimumHeight(280)
        splitter.addWidget(self.canvas)
        tables = QSplitter(Qt.Horizontal)
        self.returns = QTableWidget(0, 9)
        self.returns.setHorizontalHeaderLabels((
            tr("Návrat"), tr("Perihelium (UTC)"), "q [AU]", "e", "a [AU]",
            tr("P [roky]"), "i [°]", "Ω [°]", "ω [°]",
        ))
        self.returns.setSelectionBehavior(QTableWidget.SelectRows)
        self.returns.setEditTriggers(QTableWidget.NoEditTriggers)
        self.returns.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.returns.horizontalHeader().setStretchLastSection(True)
        self.returns.cellDoubleClicked.connect(self._open_return)
        tables.addWidget(self.returns)
        self.encounters = QTableWidget(0, 9)
        self.encounters.setHorizontalHeaderLabels((
            tr("Datum (UTC)"), tr("Planeta"), tr("Vzdálenost [AU]"), tr("Hillovy poloměry"),
            tr("v rel. [km/s]"), "v∞ [km/s]", "Δ(1/a) [AU⁻¹]", "ΔT", "ΔMOID [AU]",
        ))
        self.encounters.horizontalHeaderItem(5).setToolTip(tr(
            "Hyperbolický přebytek rychlosti vůči planetě (dvoutělesový odhad).",
        ))
        self.encounters.horizontalHeaderItem(7).setToolTip(tr(
            "Změna Tisserandova parametru vůči planetě během setkání.",
        ))
        self.encounters.horizontalHeaderItem(8).setToolTip(tr(
            "Změna MOID vůči planetě během setkání.",
        ))
        self.encounters.setEditTriggers(QTableWidget.NoEditTriggers)
        self.encounters.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.encounters.horizontalHeader().setStretchLastSection(True)
        tables.addWidget(self.encounters)
        tables.setSizes([850, 650])
        splitter.addWidget(tables); splitter.setSizes([370, 300])
        layout.addWidget(splitter, 1)
        self._mode_changed()
        self._update_source_note()

    def _update_source_note(self) -> None:
        if self.solution.position_au is None or self.solution.velocity_au_per_day is None:
            text = tr("Zdroj integrace: žádný použitelný kartézský stav ještě není načten.")
        elif self.solution.fit_observation_count is not None:
            available = self.solution.fit_available_observation_count
            count_text = (
                tr(
                    "{0} reprezentativních z {1} dostupných",
                    self.solution.fit_observation_count,
                    available,
                )
                if available is not None
                and available > self.solution.fit_observation_count
                else tr("{0} použitých", self.solution.fit_observation_count)
            )
            text = (
                tr("Zdroj integrace: vlastní fit z {0} pozorování.", count_text)
            )
            linkage = apparition_linkage_summary(self.solution)
            if linkage:
                text += f" {linkage}."
        else:
            text = tr("Zdroj integrace: výchozí kartézská dráha MPC.")
        self.source_note.setText(text)

    def _mode_changed(self, *_args) -> None:
        years = self.mode.currentData() == "years"
        self.past_label.setText(tr("Roky do minulosti:") if years else tr("Předchozí návraty:"))
        self.future_label.setText(tr("Roky do budoucnosti:") if years else tr("Budoucí návraty:"))
        if years and self.past.value() <= 50 and self.future.value() <= 50:
            self.past.setValue(200); self.future.setValue(200)
        elif not years and (self.past.value() > 50 or self.future.value() > 50):
            self.past.setValue(5); self.future.setValue(5)
        self._update_span_hint()

    def _requested_span(self) -> tuple[float, float]:
        return evolution_span(
            self.solution, str(self.mode.currentData()), self.past.value(), self.future.value(),
        )

    def _update_span_hint(self, *_args) -> None:
        try:
            start, end = self._requested_span()
        except (ValueError, TypeError, ZeroDivisionError):
            self.span_hint.setText("")
            return
        limits = usable_coverage(PLANETS.coverage_jd)
        outside = limits is not None and (start < limits[0] or end > limits[1])
        self.span_hint.setText(tr(
            "≈ {0:.0f}–{1:.0f} ({2:.0f} let)",
            approximate_year(start), approximate_year(end), (end - start) / 365.25,
        ))
        self.span_hint.setStyleSheet("color: #f0a03c; font-weight: 600;" if outside else "")
        self.span_hint.setToolTip(
            tr(
                "Rozsah přesahuje pokrytí planetární efemeridy {0} ({1:.0f}–{2:.0f}).",
                PLANETS.model_name, approximate_year(limits[0]), approximate_year(limits[1]),
            ) if outside else ""
        )

    def _confirmed_span(self) -> tuple[float, float] | None:
        """Requested interval, restricted or confirmed by the user; None = cancel."""
        try:
            start, end = self._requested_span()
        except ValueError as exc:
            QMessageBox.warning(self, tr("Vývoj dráhy"), str(exc))
            return None
        limits = usable_coverage(PLANETS.coverage_jd)
        if limits is not None and (start < limits[0] or end > limits[1]):
            try:
                clipped = clip_span_to_coverage(start, end, PLANETS.coverage_jd)
            except ValueError as exc:
                QMessageBox.warning(self, tr("Vývoj dráhy"), str(exc))
                return None
            answer = QMessageBox.question(
                self, tr("Vývoj dráhy"),
                tr(
                    "Požadovaný rozsah {0:.0f}–{1:.0f} ({2:.0f} let) přesahuje pokrytí "
                    "planetární efemeridy {3} ({4:.0f}–{5:.0f}).\n\nIntegrovat jen v rámci "
                    "pokrytí, tedy {6:.0f}–{7:.0f}?",
                    approximate_year(start), approximate_year(end), (end - start) / 365.25,
                    PLANETS.model_name, approximate_year(limits[0]),
                    approximate_year(limits[1]), approximate_year(clipped[0]),
                    approximate_year(clipped[1]),
                ),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
            )
            if answer != QMessageBox.Yes:
                return None
            # The user has just confirmed this interval; the kernel coverage
            # itself bounds its length, so no second question follows.
            return clipped
        years = (end - start) / 365.25
        if years > LONG_EVOLUTION_YEARS:
            answer = QMessageBox.question(
                self, tr("Vývoj dráhy"),
                tr(
                    "Integrace přes {0:.0f} let ({1:.0f}–{2:.0f}) může trvat mnoho minut a "
                    "tak vzdálená extrapolace je velmi nejistá.\n\nPokračovat?",
                    years, approximate_year(start), approximate_year(end),
                ),
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
            )
            if answer != QMessageBox.Yes:
                return None
        return start, end

    def set_solution(self, solution: OrbitSolution) -> None:
        self.solution = solution
        self._update_source_note()
        self._update_span_hint()
        self.result = None
        self._centered_cache.clear()
        self._relation_cache.clear()
        self._populate_views()
        self.returns.setRowCount(0); self.encounters.setRowCount(0)
        self.copy_report.setEnabled(False); self.save_report.setEnabled(False)
        self.canvas.axes.clear(); self.canvas.style_axes(); self.canvas.draw_idle()
        self.calculate.setEnabled(solution.position_au is not None)
        self.note.setText(
            tr(
                "Výpočet se spouští pouze tlačítkem. Návrat 0 je perihelový průchod "
                "nejbližší epoše dráhy.",
            )
        )
        observed = solution.observed_evolution
        if isinstance(observed, EvolutionResult):
            # The fit already integrated the whole observed arc (for example
            # 1892–2008 for 206P).  Show its returns, encounters and element
            # history directly instead of asking for the same integration.
            self._completed(observed)
            self.note.setText(
                tr(
                    "{0}  Návrat 0 je perihelový průchod nejbližší epoše dráhy.  Tlačítkem "
                    "Spustit integraci lze zvolit jiný rozsah.",
                    observed.model_note,
                )
            )

    def start(self) -> None:
        if self.thread is not None:
            return
        if self.solution.position_au is None or self.solution.velocity_au_per_day is None:
            QMessageBox.warning(
                self, tr("Vývoj dráhy"),
                tr("Nejdřív načtěte dráhu s kartézským stavem nebo spočítejte vlastní fit."),
            )
            return
        if self.past.value() == 0 and self.future.value() == 0:
            QMessageBox.warning(self, tr("Vývoj dráhy"), tr(
                "Alespoň jeden směr musí mít nenulový rozsah.",
            ))
            return
        span = self._confirmed_span()
        if span is None:
            return
        self.calculate.setEnabled(False)
        self.calculate.setText(tr("Integruji…"))
        self.note.setText(
            tr(
                "Probíhá numerická integrace; u stovek let může výpočet trvat déle. "
                "Ostatní části programu zůstávají ovladatelné.",
            )
        )
        thread = QThread(self)
        worker = OrbitEvolutionWorker(
            self.solution, str(self.mode.currentData()), self.past.value(), self.future.value(),
            span,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.success.connect(self._completed)
        worker.error.connect(self._failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._finished)
        self.thread = thread; self.worker = worker
        thread.start()

    def _finished(self) -> None:
        self.thread = None; self.worker = None
        self.calculate.setText(tr("Spustit integraci"))
        self.calculate.setEnabled(self.solution.position_au is not None)

    def _failed(self, message: str) -> None:
        self.note.setText(tr("Výpočet se nezdařil: {0}", message))
        QMessageBox.warning(self, tr("Vývoj dráhy"), message)

    def _completed(self, result: EvolutionResult) -> None:
        self.result = result
        self._centered_cache.clear()
        self._relation_cache.clear()
        self._populate_views()
        self.copy_report.setEnabled(True); self.save_report.setEnabled(True)
        self.evolution_ready.emit(self.solution, result)
        phase_note = (
            tr(
                "Vstup a výstup jsou oskulační stavy na hranicích zvoleného intervalu. "
                "Svislé značky v grafu ukazují planetární setkání. Dvojklikem zobrazíte "
                "stav ve Sluneční soustavě.",
            )
            if result.snapshots else
            tr("Dvojklikem na návrat jej zobrazíte ve Sluneční soustavě.")
        )
        self.note.setText(
            tr(
                "{0}  Krok vyhledávání událostí: {1:.2f} dne. {2}",
                result.model_note,
                result.sample_step_days,
                phase_note,
            )
            + (
                tr("  Detekováno planetocentrické zachycení: ")
                + ", ".join(
                    f"{PLANET_LABELS.get(item.planet, item.planet)} "
                    f"{(datetime_from_jd(item.start_jd) or datetime.min.replace(tzinfo=timezone.utc)).year}–"
                    f"{(datetime_from_jd(item.end_jd) or datetime.min.replace(tzinfo=timezone.utc)).year}"
                    for item in result.captures
                )
                + "."
                if result.captures else ""
            )
        )
        self._populate_state_table()
        rows: list[tuple[float, tuple[str, ...], str]] = []
        for item in result.encounters:
            date = datetime_from_jd(item.epoch_jd)
            rows.append((item.epoch_jd, (
                date.strftime("%Y-%m-%d %H:%M") if date else f"JD {item.epoch_jd:.2f}",
                PLANET_LABELS.get(item.planet, item.planet), f"{item.distance_au:.6f}",
                f"{item.hill_radii:.3f}", f"{item.relative_velocity_kms:.2f}",
                "—" if item.v_infinity_kms is None else f"{item.v_infinity_kms:.2f}",
                "—" if item.delta_reciprocal_a is None else f"{item.delta_reciprocal_a:+.3e}",
                "—" if item.delta_tisserand is None else f"{item.delta_tisserand:+.4f}",
                "—" if item.delta_moid_au is None else f"{item.delta_moid_au:+.5f}",
            ), ""))
        for impact in result.impacts:
            point = impact.entry or impact.surface
            label = BODY_LABELS.get(impact.body, impact.body)
            rows.append((point.epoch_jd, (
                point.utc.strftime("%Y-%m-%d %H:%M:%S") if point.utc
                else f"JD {point.epoch_jd:.5f}",
                tr("{0} — DOPAD", label) if impact.forward else tr("{0} — KOLIZE", label),
                "0", "0", f"{point.speed_kms:.2f}", "—", "—", "—", "—",
            ), describe_impact(impact, label)))
        rows.sort(key=lambda row: row[0])
        self.encounters.setRowCount(len(rows))
        for row, (_epoch, values, tooltip) in enumerate(rows):
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                if tooltip:
                    cell.setToolTip(tooltip)
                    cell.setForeground(QColor("#ff8a7a"))
                self.encounters.setItem(row, column, cell)
        if result.impacts:
            self.note.setText(
                self.note.text() + "  " + "  ".join(
                    describe_impact(item, BODY_LABELS.get(item.body, item.body)).replace(
                        "\n", "; ",
                    )
                    for item in result.impacts
                )
            )
        if result.snapshots:
            reciprocal_index = self.graph_metric.findData("reciprocal_a")
            if reciprocal_index >= 0:
                self.graph_metric.setCurrentIndex(reciprocal_index)
        self._draw_result()
        if result.impacts and self.isVisible():
            QMessageBox.warning(
                self, tr("Vývoj dráhy — dopad"),
                "\n\n".join(
                    describe_impact(item, BODY_LABELS.get(item.body, item.body))
                    for item in result.impacts
                ) + tr(
                    "\n\nIntegrace v tomto směru končí okamžikem dopadu. Místo na povrchu "
                    "je průsečík vakuové dráhy s elipsoidem WGS84; brzdění a rozpad v "
                    "atmosféře nejsou modelovány.",
                ),
            )

    def _view(self) -> tuple[str, str]:
        """(kind, body): ('elements', 'sun'), ('relation', planet) or ('captured', planet)."""
        kind, _separator, body = str(self.element_center.currentData() or "elements:sun").partition(":")
        return kind, body or "sun"

    def _populate_views(self) -> None:
        current = str(self.element_center.currentData() or "elements:sun")
        captured = sorted({item.planet for item in self.result.captures}) if self.result else []
        self.element_center.blockSignals(True)
        self.element_center.clear()
        self.element_center.addItem(tr("Pohled: heliocentrické elementy"), "elements:sun")
        for planet in RELATION_PLANETS:
            self.element_center.addItem(
                tr("Pohled: vztah k planetě {0}", PLANET_LABELS.get(planet, planet)),
                f"relation:{planet}",
            )
        for planet in captured:
            self.element_center.addItem(
                tr("Pohled: planetocentrické elementy — {0} (zachycení)",
                   PLANET_LABELS.get(planet, planet)),
                f"captured:{planet}",
            )
        index = self.element_center.findData(current)
        self.element_center.setCurrentIndex(index if index >= 0 else 0)
        self.element_center.blockSignals(False)
        self._populate_metrics(self._view()[0])

    def _populate_metrics(self, kind: str) -> None:
        wanted = "relation" if kind == "relation" else "elements"
        if self.graph_metric.property("metricKind") == wanted and self.graph_metric.count():
            return
        self.graph_metric.blockSignals(True)
        self.graph_metric.clear()
        if wanted == "relation":
            for key, label in (
                ("distance", tr("Graf: vzdálenost od planety")),
                ("moid", tr("Graf: MOID (nejmenší vzdálenost drah)")),
                ("tisserand", tr("Graf: Tisserandův parametr T")),
                ("period_ratio", tr("Graf: poměr oběžných dob (rezonance)")),
                ("nodes", tr("Graf: vzdálenosti uzlů od dráhy planety")),
            ):
                self.graph_metric.addItem(label, key)
        else:
            for key, label in (
                ("q", tr("Graf: perihelová vzdálenost q")),
                ("a", tr("Graf: velká poloosa a")),
                ("e", tr("Graf: excentricita e")),
                ("period", tr("Graf: oběžná doba P")),
                ("reciprocal_a", tr("Graf: reciproká poloosa 1/a")),
                ("inclination", tr("Graf: sklon i")),
                ("node", tr("Graf: výstupný uzel Ω")),
                ("arg_peri", tr("Graf: argument perihelu ω")),
            ):
                self.graph_metric.addItem(label, key)
        self.graph_metric.setProperty("metricKind", wanted)
        self.graph_metric.blockSignals(False)

    def _states_for_center(self, kind: str) -> tuple:
        if self.result is None:
            return ()
        view, body = self._view()
        center = body if view == "captured" else "sun"
        key = (center, kind)
        if key in self._centered_cache:
            return self._centered_cache[key]
        if center == "sun":
            source = (
                (self.result.snapshots or self.result.returns)
                if kind == "events" else self.result.timeline
            )
            converted = tuple(source)
        else:
            # Planetocentric elements only while the planet holds the body.
            intervals = [
                (item.start_jd, item.end_jd) for item in self.result.captures
                if item.planet == center
            ]
            inside = [
                item for item in self.result.timeline
                if any(start <= item.epoch_jd <= end for start, end in intervals)
            ]
            if kind == "events" and len(inside) > 40:
                inside = [inside[index] for index in np.linspace(0, len(inside) - 1, 40).astype(int)]
            values = []
            for item in inside:
                try:
                    values.append(centered_snapshot(item, center))
                except (ValueError, OrbitParseError, FloatingPointError):
                    continue
            converted = tuple(values)
        self._centered_cache[key] = converted
        return converted

    def _center_changed(self, *_args) -> None:
        self._populate_metrics(self._view()[0])
        self._populate_state_table()
        self._draw_result()

    def _relation_series(self, planet: str):
        """Cached relation series; starts a background computation if missing."""
        if planet in self._relation_cache:
            return self._relation_cache[planet]
        if self.result is None:
            return None
        if self._relation_thread is None:
            self._relation_pending = planet
            thread = QThread(self)
            worker = PlanetRelationWorker(self.result, planet)
            worker.moveToThread(thread)
            thread.started.connect(worker.run)
            result_identity = id(self.result)
            worker.success.connect(
                lambda series, identity=result_identity: self._relation_ready(series, identity)
            )
            worker.finished.connect(thread.quit)
            worker.finished.connect(worker.deleteLater)
            thread.finished.connect(thread.deleteLater)
            thread.finished.connect(self._relation_finished)
            self._relation_thread = thread; self._relation_worker = worker
            thread.start()
        return None

    def _relation_ready(self, series, identity: int) -> None:
        if self.result is not None and id(self.result) == identity:
            self._relation_cache[series.planet] = series
            view, body = self._view()
            if view == "relation" and body == series.planet:
                self._populate_state_table()
                self._draw_result()

    def _relation_finished(self) -> None:
        self._relation_thread = None; self._relation_worker = None
        view, body = self._view()
        if self.result is not None and view == "relation" and body not in self._relation_cache:
            self._relation_series(body)

    def _set_state_headers(self, labels: tuple[str, ...]) -> None:
        for column, label in enumerate(labels):
            self.returns.setHorizontalHeaderItem(column, QTableWidgetItem(label))

    def _populate_relation_table(self, planet: str) -> None:
        result = self.result
        first = (tr("Fáze"), tr("Epocha (UTC)")) if result.snapshots else (
            tr("Rok návratu"), tr("Perihelium (UTC)"),
        )
        self._set_state_headers(first + (
            "MOID [AU]", "T", tr("P / P planety"), tr("☊ uzel [AU]"), tr("☋ uzel [AU]"),
            tr("i vůči planetě [°]"), "",
        ))
        series = self._relation_series(planet)
        events = result.snapshots or result.returns
        if series is None:
            self.returns.setRowCount(1)
            self.returns.setItem(0, 0, QTableWidgetItem(tr("počítám…")))
            for column in range(1, 9):
                self.returns.setItem(0, column, QTableWidgetItem(""))
            return
        self.returns.setRowCount(len(events))
        for row, (item, relation) in enumerate(zip(events, series.events)):
            date = datetime_from_jd(item.epoch_jd)

            def number(value: float, pattern: str) -> str:
                return "—" if value is None or not np.isfinite(value) else format(value, pattern)

            values = (
                tr(item.phase) if result.snapshots else (
                    str(date.year) if date else f"JD {item.epoch_jd:.0f}"
                ),
                date.strftime("%Y-%m-%d %H:%M") if date else f"JD {item.epoch_jd:.2f}",
            ) + ((
                number(relation.moid_au, ".6f"), number(relation.tisserand, ".4f"),
                number(relation.period_ratio, ".4f"),
                number(relation.ascending_node_distance_au, "+.5f"),
                number(relation.descending_node_distance_au, "+.5f"),
                number(relation.relative_inclination_deg, ".3f"), "",
            ) if relation is not None else ("—",) * 7)
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setTextAlignment(Qt.AlignCenter if column != 1 else Qt.AlignLeft | Qt.AlignVCenter)
                self.returns.setItem(row, column, cell)

    def _populate_state_table(self) -> None:
        if self.result is None:
            self.returns.setRowCount(0)
            return
        view, body = self._view()
        if view == "relation":
            self._populate_relation_table(body)
            return
        displayed_states = self._states_for_center("events")
        result = self.result
        element_headers = (
            "q [AU]", "e", "a [AU]", tr("P [roky]"), "i [°]", "Ω [°]", "ω [°]",
        )
        if view == "captured":
            self._set_state_headers((tr("Zachycení"), tr("Epocha (UTC)")) + element_headers)
        elif result.snapshots:
            self._set_state_headers((tr("Fáze"), tr("Epocha (UTC)")) + element_headers)
        else:
            self._set_state_headers((tr("Rok návratu"), tr("Perihelium (UTC)")) + element_headers)
        self.returns.setRowCount(len(displayed_states))
        for row, item in enumerate(displayed_states):
            date = datetime_from_jd(item.epoch_jd)
            values = (
                PLANET_LABELS.get(body, body) if view == "captured" else
                tr(item.phase) if result.snapshots else (
                    str(date.year) if date else f"JD {item.epoch_jd:.0f}"
                ),
                date.strftime("%Y-%m-%d %H:%M") if date else f"JD {item.epoch_jd:.2f}",
                f"{item.perihelion_au:.6f}", f"{item.eccentricity:.8f}",
                f"{item.semi_major_axis_au:.6f}",
                "—" if item.period_years is None else f"{item.period_years:.4f}",
                f"{item.inclination_deg:.5f}", f"{item.node_deg:.5f}",
                f"{item.arg_peri_deg:.5f}",
            )
            for column, value in enumerate(values):
                cell = QTableWidgetItem(value)
                cell.setTextAlignment(Qt.AlignCenter if column != 1 else Qt.AlignLeft | Qt.AlignVCenter)
                self.returns.setItem(row, column, cell)

    def _draw_result(self) -> None:
        ax = self.canvas.axes; ax.clear(); self.canvas.style_axes()
        if not self.result or not (self.result.timeline or self.result.returns or self.result.snapshots):
            ax.text(.5, .5, tr("V zadaném rozsahu nejsou dostupné stavy dráhy."),
                    ha="center", va="center", transform=ax.transAxes, color="#afc2d8")
            self.canvas.draw_idle(); return
        view, body = self._view()
        if view == "relation":
            self._draw_relation(body)
            return
        metric_key = str(self.graph_metric.currentData() or "q")
        if metric_key not in self.GRAPH_METRICS:
            metric_key = "q"
        attribute, y_label, title_label = self.GRAPH_METRICS[metric_key]
        displayed_states = self._states_for_center("events")
        timeline = self._states_for_center("timeline") or displayed_states
        x = np.array([2000.0 + (item.epoch_jd - 2451545.0) / 365.25 for item in timeline])
        values = np.array([
            np.nan if getattr(item, attribute) is None else float(getattr(item, attribute))
            for item in timeline
        ])
        if metric_key in {"node", "arg_peri"}:
            values = np.degrees(np.unwrap(np.radians(values)))
        finite = np.isfinite(values)
        # NaN values (notably the period while an orbit is open) deliberately
        # break the curve instead of drawing a misleading bridge.
        ax.plot(x, values, color="#67cfe8", lw=1.45, alpha=.9)
        return_x = np.array([
            2000.0 + (item.epoch_jd - 2451545.0) / 365.25 for item in displayed_states
        ])
        return_y = np.array([
            np.nan if getattr(item, attribute) is None else float(getattr(item, attribute))
            for item in displayed_states
        ])
        if metric_key in {"node", "arg_peri"} and np.any(np.isfinite(values)):
            return_y = np.interp(return_x, x[finite], values[finite])
        return_finite = np.isfinite(return_y)
        ax.scatter(return_x[return_finite], return_y[return_finite],
                   color="#f1f5f9", edgecolor="#55e3e9", s=38, zorder=4)
        source_year = 2000.0 + (self.solution.epoch_jd - 2451545.0) / 365.25
        ax.axvline(source_year, color="#55e3e9", lw=.8, alpha=.4)
        if self.result.snapshots:
            for item, x_value, y_value in zip(displayed_states, return_x, return_y):
                if np.isfinite(y_value):
                    ax.annotate(tr(item.phase), (x_value, y_value), xytext=(5, 7),
                                textcoords="offset points", color="#b9cbe0", fontsize=8)
        else:
            for item, x_value, y_value in zip(displayed_states, return_x, return_y):
                if np.isfinite(y_value):
                    date = datetime_from_jd(item.epoch_jd)
                    ax.annotate(str(date.year) if date else "", (x_value, y_value),
                                xytext=(4, 6), textcoords="offset points",
                                color="#b9cbe0", fontsize=8)
        for encounter in self.result.encounters:
            encounter_x = 2000.0 + (encounter.epoch_jd - 2451545.0) / 365.25
            ax.axvline(encounter_x, color="#f0c85a", lw=.9, alpha=.65, ls="--")
            date = datetime_from_jd(encounter.epoch_jd)
            label = PLANET_LABELS.get(encounter.planet, encounter.planet)
            if date:
                label += f" {date.year}"
            ax.text(encounter_x, .98, label, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", color="#f0c85a", fontsize=8)
        for capture in self.result.captures:
            start_x = 2000.0 + (capture.start_jd - 2451545.0) / 365.25
            end_x = 2000.0 + (capture.end_jd - 2451545.0) / 365.25
            ax.axvspan(start_x, end_x, color="#b07cff", alpha=.12, zorder=.1)
            ax.text((start_x + end_x) / 2.0, .04,
                    tr("zachycení: {0}", PLANET_LABELS.get(capture.planet, capture.planet)),
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    color="#c9a8ff", fontsize=8)
        ax.set_xlabel(tr("rok"))
        ax.set_ylabel(y_label)
        self._draw_impacts(ax)
        ax.set_title(
            tr(
                "{0} — vývoj {1} (heliocentrické elementy)",
                self.solution.display_name or self.solution.designation,
                title_label,
            ) if view != "captured" else tr(
                "{0} — vývoj {1} (planetocentrické elementy během zachycení: {2})",
                self.solution.display_name or self.solution.designation,
                title_label, PLANET_LABELS.get(body, body),
            )
        )
        ax.grid(color="#273750", alpha=.55, lw=.6)
        self.canvas.draw_idle()

    def _draw_impacts(self, ax) -> None:
        for impact in self.result.impacts if self.result else ():
            point = impact.entry or impact.surface
            x_value = 2000.0 + (point.epoch_jd - 2451545.0) / 365.25
            ax.axvline(x_value, color="#ff6b5b", lw=1.6, alpha=.9)
            ax.text(
                x_value, .5, tr("dopad: {0}", BODY_LABELS.get(impact.body, impact.body)),
                transform=ax.get_xaxis_transform(), rotation=90, ha="right", va="center",
                color="#ff8a7a", fontsize=9,
            )

    def _draw_relation(self, planet: str) -> None:
        """Distance, MOID, Tisserand, period ratio or nodes relative to a planet."""
        ax = self.canvas.axes
        result = self.result
        metric = str(self.graph_metric.currentData() or "distance")
        if metric not in self.RELATION_METRICS:
            metric = "distance"
        y_label, title_label = self.RELATION_METRICS[metric]
        label = PLANET_LABELS.get(planet, planet)

        def years(epochs) -> np.ndarray:
            return 2000.0 + (np.asarray(epochs, dtype=float) - 2451545.0) / 365.25

        series = None
        if metric != "distance":
            series = self._relation_series(planet)
            if series is None:
                ax.text(.5, .5, tr("Počítám vztah k planetě {0}…", label), ha="center",
                        va="center", transform=ax.transAxes, color="#afc2d8")
                self.canvas.draw_idle()
                return
        events = result.snapshots or result.returns
        event_x = years([item.epoch_jd for item in events])
        if metric == "distance":
            distances = result.planet_distances.get(planet)
            if distances is None or result.trajectory_epochs is None:
                ax.text(.5, .5, tr("Vzdálenosti od planet nejsou pro tento výpočet uloženy."),
                        ha="center", va="center", transform=ax.transAxes, color="#afc2d8")
                self.canvas.draw_idle()
                return
            ax.semilogy(years(result.trajectory_epochs), distances, color="#67cfe8", lw=1.2)
            hill = HILL_RADII_AU.get(planet)
            if hill:
                ax.axhline(hill, color="#b07cff", lw=.9, ls="--", alpha=.8)
                ax.text(.005, hill, tr("Hillova sféra"), transform=ax.get_yaxis_transform(),
                        va="bottom", color="#c9a8ff", fontsize=8)
            if planet == "earth":
                ax.axhline(.00257, color="#9fb3c8", lw=.8, ls=":", alpha=.8)
                ax.text(.005, .00257, tr("vzdálenost Měsíce"), transform=ax.get_yaxis_transform(),
                        va="bottom", color="#b9cbe0", fontsize=8)
            for encounter in result.encounters:
                if encounter.planet == planet:
                    ax.scatter(years([encounter.epoch_jd]), [encounter.distance_au], color="#f0c85a",
                               s=30, zorder=4)
        elif metric == "moid":
            values = np.where(series.moid_au > 0.0, series.moid_au, np.nan)
            ax.semilogy(years(series.moid_epochs), values, color="#67cfe8", lw=1.3)
            event_values = np.array([
                np.nan if item is None else item.moid_au for item in series.events
            ])
            ax.scatter(event_x, np.where(event_values > 0, event_values, np.nan),
                       color="#f1f5f9", edgecolor="#55e3e9", s=34, zorder=4)
            if planet == "earth":
                ax.axhline(.05, color="#f0a03c", lw=.9, ls="--", alpha=.8)
                ax.text(.005, .05, tr("0,05 AU — mez potenciálně nebezpečných těles"),
                        transform=ax.get_yaxis_transform(), va="bottom", color="#f0c08a",
                        fontsize=8)
        elif metric == "tisserand":
            ax.plot(years(series.epochs), series.tisserand, color="#67cfe8", lw=1.3)
            event_values = np.array([np.nan if item is None else item.tisserand for item in series.events])
            ax.scatter(event_x, event_values, color="#f1f5f9", edgecolor="#55e3e9", s=34, zorder=4)
            finite = series.tisserand[np.isfinite(series.tisserand)]
            near = (
                (lambda level: finite.size and finite.min() - 1.0 <= level <= finite.max() + 1.0)
            )
            if planet == "jupiter":
                for level, text in ((2.0, tr("T = 2: Halleyova typu / Jupiterova rodina")),
                                    (3.0, tr("T = 3: Jupiterova rodina / planetky"))):
                    if not near(level):
                        continue
                    ax.axhline(level, color="#f0a03c", lw=.8, ls="--", alpha=.75)
                    ax.text(.005, level, text, transform=ax.get_yaxis_transform(),
                            va="bottom", color="#f0c08a", fontsize=8)
        elif metric == "period_ratio":
            ax.plot(years(series.epochs), series.period_ratio, color="#67cfe8", lw=1.3)
            event_values = np.array([np.nan if item is None else item.period_ratio for item in series.events])
            ax.scatter(event_x, event_values, color="#f1f5f9", edgecolor="#55e3e9", s=34, zorder=4)
            finite = series.period_ratio[np.isfinite(series.period_ratio)]
            if finite.size:
                low, high = float(finite.min()), float(finite.max())
                margin = max(.02, (high - low) * .15)
                for value, text in nearby_resonances(low - margin, high + margin):
                    ax.axhline(value, color="#f0a03c", lw=.7, ls=":", alpha=.7)
                    ax.text(.995, value, text, transform=ax.get_yaxis_transform(), ha="right",
                            va="bottom", color="#f0c08a", fontsize=8)
                ax.set_ylim(low - margin, high + margin)
            ax.text(.005, .02, tr("P planety = {0:.3f} roku", planet_period_years(planet, self.solution.epoch_jd)),
                    transform=ax.transAxes, color="#b9cbe0", fontsize=8)
        else:
            ax.plot(years(series.epochs), series.ascending_node_au, color="#67cfe8", lw=1.3,
                    label=tr("výstupný uzel ☊"))
            ax.plot(years(series.epochs), series.descending_node_au, color="#f0c85a", lw=1.3,
                    label=tr("sestupný uzel ☋"))
            ax.axhline(0.0, color="#ff6b5b", lw=.8, alpha=.7)
            legend = ax.legend(loc="best", fontsize=8, facecolor="#0e1828", edgecolor="#273750")
            for text in legend.get_texts():
                text.set_color("#d6e2f0")
        for encounter in result.encounters:
            if encounter.planet == planet:
                ax.axvline(years([encounter.epoch_jd])[0], color="#f0c85a", lw=.8, alpha=.5, ls="--")
        self._draw_impacts(ax)
        ax.set_xlabel(tr("rok"))
        ax.set_ylabel(y_label)
        ax.set_title(tr(
            "{0} — vývoj {1} vůči planetě {2}",
            self.solution.display_name or self.solution.designation, title_label, label,
        ))
        ax.grid(color="#273750", alpha=.55, lw=.6)
        self.canvas.draw_idle()

    def _report_text(self) -> str:
        if self.result is None:
            return ""
        if self.report_format.currentData() == "mpc":
            return format_evolution_mpc_report(self.solution, self.result)
        return format_evolution_report(self.solution, self.result)

    def _copy_report(self) -> None:
        report = self._report_text()
        if not report:
            return
        QApplication.clipboard().setText(report)
        self.note.setText(
            tr(
                "Chronologický TXT výstup byl zkopírován do schránky.  {0}",
                self.result.model_note,
            )
        )

    def _save_report(self) -> None:
        report = self._report_text()
        if not report:
            return
        safe_name = re.sub(
            r"[^A-Za-z0-9._-]+", "_",
            self.solution.display_name or self.solution.designation,
        ).strip("_") or "orbit"
        suffix = "mpc-elements" if self.report_format.currentData() == "mpc" else tr(
            "vyvoj-drahy",
        )
        path, _filter = QFileDialog.getSaveFileName(
            self, tr("Uložit vývoj dráhy"), f"{safe_name}-{suffix}.txt",
            tr("Textové soubory (*.txt);;Všechny soubory (*)"),
        )
        if not path:
            return
        try:
            Path(path).write_text(report, encoding="utf-8-sig")
        except OSError as exc:
            QMessageBox.warning(self, tr("Vývoj dráhy"), tr(
                "Soubor se nepodařilo uložit:\n{0}",
                exc,
            ))
            return
        self.note.setText(tr("Chronologický výstup byl uložen do {0}", path))

    def _open_return(self, row: int, _column: int) -> None:
        if not self.result:
            return
        displayed_states = self.result.snapshots or self.result.returns
        if not (0 <= row < len(displayed_states)):
            return
        item = displayed_states[row]
        date = datetime_from_jd(item.epoch_jd)
        if date is not None:
            self.show_return.emit(return_solution(self.solution, item), date)


class PlaybackBar(QWidget):
    time_changed = Signal(object)

    STANDARD_SPEEDS = (
        ("1 min/s", 1 / 1440), ("10 min/s", 10 / 1440),
        ("1 h/s", 1 / 24), ("6 h/s", 6 / 24),
        (tr("1 den/s"), 1), (tr("2 dny/s"), 2), (tr("5 dní/s"), 5),
        (tr("10 dní/s"), 10), (tr("25 dní/s"), 25),
        (tr("50 dní/s"), 50), (tr("100 dní/s"), 100),
        (tr("250 dní/s"), 250), (tr("500 dní/s"), 500),
    )
    VISIBILITY_SPEEDS = (
        (tr("1 den/s"), 1), (tr("2 dny/s"), 2), (tr("3 dny/s"), 3),
        (tr("5 dní/s"), 5), (tr("10 dní/s"), 10), (tr("15 dní/s"), 15),
        (tr("20 dní/s"), 20), (tr("25 dní/s"), 25),
    )

    def __init__(self) -> None:
        super().__init__()
        self.direction = 0
        self._fractional_seconds = 0.0
        self.timer = QTimer(self)
        # Ten frames per second made 100 days/s advance in conspicuous
        # ten-day jumps.  A 40-fps target keeps even the fast astronomical
        # time scales visually continuous.  If a complex plot needs longer
        # to render, Qt coalesces timer events instead of building a backlog.
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.setInterval(25)
        self.timer.timeout.connect(self._tick)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.back = QPushButton(tr("◀ Zpět"))
        self.pause = QPushButton("■ Stop")
        self.forward = QPushButton(tr("Vpřed ▶"))
        self.slower = QPushButton("−")
        self.faster = QPushButton("+")
        self.speed = QComboBox()
        self._speed_profile = "standard"
        for label, days_per_second in self.STANDARD_SPEEDS:
            self.speed.addItem(label, days_per_second)
        self.speed.setCurrentIndex(2)
        self.time = QDateTimeEdit(datetime.now())
        self.time.setDisplayFormat("yyyy-MM-dd  HH:mm")
        self.time.setFixedWidth(175)
        self.time.setStyleSheet("font-weight: 700; color: #f5fbff; background: #13243a;")
        self.time.setCalendarPopup(True)
        self.time.dateTimeChanged.connect(lambda qdt: self.time_changed.emit(qdt.toPython()))
        self.back.clicked.connect(lambda: self._play(-1))
        self.forward.clicked.connect(lambda: self._play(1))
        self.pause.clicked.connect(self._stop)
        self.slower.clicked.connect(lambda: self.speed.setCurrentIndex(max(0, self.speed.currentIndex() - 1)))
        self.faster.clicked.connect(
            lambda: self.speed.setCurrentIndex(min(self.speed.count() - 1, self.speed.currentIndex() + 1))
        )
        for widget in (self.back, self.pause, self.forward, self.slower, self.speed, self.faster, self.time):
            layout.addWidget(widget)
        layout.addStretch()

    def set_speed_profile(self, profile: str) -> None:
        if profile == self._speed_profile:
            return
        options = self.VISIBILITY_SPEEDS if profile == "visibility" else self.STANDARD_SPEEDS
        self.speed.blockSignals(True)
        self.speed.clear()
        for label, days_per_second in options:
            self.speed.addItem(label, days_per_second)
        preferred = 1.0 if profile == "visibility" else 1.0 / 24.0
        index = self.speed.findData(preferred)
        self.speed.setCurrentIndex(max(0, index))
        self.speed.blockSignals(False)
        self._speed_profile = profile

    def _play(self, direction: int) -> None:
        self.direction = direction
        self._fractional_seconds = 0.0
        self.timer.start()

    def _stop(self) -> None:
        self.direction = 0
        self.timer.stop()
        self._fractional_seconds = 0.0
        # Repaint once in full quality.  Animated panels may omit costly
        # secondary labels while the timer is running.
        self.time_changed.emit(self.time.dateTime().toPython())

    def _tick(self) -> None:
        days_per_second = float(self.speed.currentData())
        delta_seconds = (
            self.direction * days_per_second * 86400
            * self.timer.interval() / 1000
        ) + self._fractional_seconds
        whole_seconds = int(delta_seconds)
        self._fractional_seconds = delta_seconds - whole_seconds
        if whole_seconds:
            self.time.setDateTime(
                self.time.dateTime().addSecs(whole_seconds)
            )


class SolarSystemPanel(QWidget):
    def __init__(
        self, solution: OrbitSolution, playback: PlaybackBar, allow_detach: bool = True,
    ) -> None:
        super().__init__()
        self.solution = solution
        self.evolution_source: OrbitSolution | None = None
        self.evolution_result: EvolutionResult | None = None
        self.playback = playback
        self._detached_windows: list[QMainWindow] = []
        self.canvas = PlotCanvas()
        self.canvas.figure.set_layout_engine(None)
        self.canvas.axes.set_position([.01, .01, .98, .96])
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(
            lambda: self.set_time(self.playback.time.dateTime().toPython())
        )
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        notice = QLabel(
            tr(
                "Heliocentrická ekliptikální soustava J2000 • tažení myší otáčí kameru • "
                "kolečko přibližuje",
            )
        )
        notice.setObjectName("muted")
        controls.addWidget(notice); controls.addStretch()
        self.show_orbit_curtain = QCheckBox(tr("Projekční plocha k ekliptice (3D)"))
        self.show_orbit_curtain.setChecked(False)
        self.show_orbit_curtain.setToolTip(
            tr(
                "Zobrazí průsvitnou projekční plochu mezi dráhou tělesa a jejím kolmým "
                "průmětem do roviny ekliptiky.",
            )
        )
        self.show_orbit_curtain.toggled.connect(
            lambda _: self.set_time(self.playback.time.dateTime().toPython())
        )
        controls.addWidget(self.show_orbit_curtain)
        self.evolution_count = QComboBox()
        self.evolution_count.addItem(tr("Integrovaná dráha: vypnuta"), 0)
        self.evolution_count.addItem(tr("Integrovaná dráha: zapnuta"), -1)
        self.evolution_count.setCurrentIndex(1)
        self.evolution_count.setVisible(False)
        self.evolution_count.currentIndexChanged.connect(self._evolution_display_changed)
        controls.addWidget(self.evolution_count)
        self.view_center = QComboBox()
        for key, label in ELEMENT_CENTERS:
            self.view_center.addItem(tr("Střed: {0}", label), key)
        self.view_center.setEnabled(False)
        self.view_center.currentIndexChanged.connect(self._view_center_changed)
        controls.addWidget(self.view_center)
        enlarge = QPushButton(tr("Zvětšit ↗") if allow_detach else tr("Celá obrazovka (F11)"))
        enlarge.setToolTip(
            tr(
                "Otevře dráhu ve zvětšeném samostatném okně. Klávesou F11 přepnete celou "
                "obrazovku.",
            )
        )
        enlarge.clicked.connect(self._open_detached if allow_detach else self._toggle_fullscreen)
        controls.addWidget(enlarge)
        layout.addLayout(controls)
        view_controls = QHBoxLayout()
        self.view_azimuth = QSlider(Qt.Horizontal); self.view_azimuth.setRange(0, 360); self.view_azimuth.setValue(90)
        self.view_elevation = QSlider(Qt.Horizontal); self.view_elevation.setRange(-89, 89); self.view_elevation.setValue(35)
        self.view_zoom = QSlider(Qt.Horizontal); self.view_zoom.setRange(0, 1000); self.view_zoom.setValue(838)
        self.azimuth_readout = QLabel(); self.azimuth_readout.setMinimumWidth(38)
        self.elevation_readout = QLabel(); self.elevation_readout.setMinimumWidth(38)
        self.zoom_readout = QLabel()
        self.zoom_readout.setMinimumWidth(105)
        for label, slider in ((tr("Azimut kamery"), self.view_azimuth), (tr("Výška kamery"), self.view_elevation),
                              (tr("Přiblížení"), self.view_zoom)):
            slider.setMinimumWidth(120)
            slider.valueChanged.connect(lambda _: self.set_time(self.playback.time.dateTime().toPython()))
            view_controls.addWidget(QLabel(label)); view_controls.addWidget(slider, 1)
            if slider is self.view_azimuth:
                view_controls.addWidget(self.azimuth_readout)
            elif slider is self.view_elevation:
                view_controls.addWidget(self.elevation_readout)
            else:
                view_controls.addWidget(self.zoom_readout)
        top_view = QPushButton(tr("Shora"))
        ecliptic_view = QPushButton(tr("Z ekliptiky"))
        top_view.clicked.connect(lambda: self.view_elevation.setValue(89))
        ecliptic_view.clicked.connect(lambda: self.view_elevation.setValue(0))
        view_controls.addWidget(top_view); view_controls.addWidget(ecliptic_view)
        layout.addLayout(view_controls)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # The toolbar's live (x, y) readout was squeezed under the controls in
        # narrow layouts and appeared as an unreadable clipped number.
        for attribute in ("locLabel", "_message_label"):
            label = getattr(self.toolbar, attribute, None)
            if label is not None:
                label.hide()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.info = QLabel(); self.info.setObjectName("value")
        self.info.setStyleSheet("padding: 7px 10px; background: #111d30; border-radius: 6px;")
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.info.setMinimumWidth(0)
        self.info.setFixedHeight(42)
        layout.addWidget(self.info)
        layout.addWidget(playback)
        self._camera_drag: tuple[float, float, int, int] | None = None
        self._center_track_cache: dict[tuple[str, int], np.ndarray] = {}
        # A complete planetary orbit is effectively a static backdrop over
        # an animation.  Recomputing thousands of DE440 positions for every
        # frame was by far the largest source of stutter.
        self._planet_path_cache: dict[tuple[str, str], np.ndarray] = {}
        self.canvas.mpl_connect("button_press_event", self._camera_press)
        self.canvas.mpl_connect("motion_notify_event", self._camera_motion)
        self.canvas.mpl_connect("button_release_event", self._camera_release)
        self.canvas.mpl_connect("scroll_event", self._camera_scroll)
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self._toggle_fullscreen)
        self.escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.escape_shortcut.activated.connect(self._leave_fullscreen)
        playback.time_changed.connect(self.set_time)
        self.set_time(datetime.now())

    def _open_detached(self) -> None:
        window = QMainWindow(self)
        window.setAttribute(Qt.WA_DeleteOnClose)
        window.setWindowTitle(tr("{0} — dráha ve Sluneční soustavě", self.solution.designation))
        playback = PlaybackBar()
        playback.time.setDateTime(self.playback.time.dateTime())
        panel = SolarSystemPanel(self.solution, playback, allow_detach=False)
        panel.view_azimuth.setValue(self.view_azimuth.value())
        panel.view_elevation.setValue(self.view_elevation.value())
        panel.view_zoom.setValue(self.view_zoom.value())
        panel.show_orbit_curtain.setChecked(self.show_orbit_curtain.isChecked())
        if self.evolution_result is not None and self.evolution_source is not None:
            panel.set_evolution(self.evolution_source, self.evolution_result)
            panel.evolution_count.setCurrentIndex(
                panel.evolution_count.findData(self.evolution_count.currentData())
            )
            panel.view_center.setCurrentIndex(
                panel.view_center.findData(self.view_center.currentData())
            )
        window.setCentralWidget(panel)
        window.resize(1500, 950)
        self._detached_windows.append(window)
        window.destroyed.connect(
            lambda _=None, item=window: self._detached_windows.remove(item)
            if item in self._detached_windows else None
        )
        window.showMaximized()

    def _toggle_fullscreen(self) -> None:
        window = self.window()
        window.showNormal() if window.isFullScreen() else window.showFullScreen()

    def _leave_fullscreen(self) -> None:
        if self.window().isFullScreen():
            self.window().showNormal()

    def _camera_press(self, event) -> None:
        if event.button != 1 or event.inaxes is not self.canvas.axes or self.toolbar.mode:
            return
        self._camera_drag = (
            float(event.x), float(event.y), self.view_azimuth.value(), self.view_elevation.value()
        )

    def _camera_motion(self, event) -> None:
        if self._camera_drag is None or event.x is None or event.y is None:
            return
        start_x, start_y, start_azimuth, start_elevation = self._camera_drag
        azimuth = int(round(start_azimuth - (float(event.x) - start_x) * .35)) % 360
        elevation = int(np.clip(
            round(start_elevation + (float(event.y) - start_y) * .25), -89, 89
        ))
        self.view_azimuth.blockSignals(True)
        self.view_elevation.blockSignals(True)
        self.view_azimuth.setValue(azimuth)
        self.view_elevation.setValue(elevation)
        self.view_azimuth.blockSignals(False)
        self.view_elevation.blockSignals(False)
        self.set_time(self.playback.time.dateTime().toPython())

    def _camera_release(self, event) -> None:
        self._camera_drag = None

    def _camera_scroll(self, event) -> None:
        if event.inaxes is not self.canvas.axes or not event.step:
            return
        self.view_zoom.setValue(int(np.clip(
            self.view_zoom.value() - event.step * 28, 0, self.view_zoom.maximum()
        )))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        # Recalculate the physical 3-D box after the tab/window has obtained
        # its real dimensions.  A short debounce keeps live resizing smooth.
        if hasattr(self, "_resize_timer"):
            self._resize_timer.start()

    def set_solution(self, solution: OrbitSolution) -> None:
        self.solution = solution
        self._planet_path_cache.clear()
        self.set_time(datetime.now())

    def set_evolution(
        self, source: OrbitSolution | None, result: EvolutionResult | None,
    ) -> None:
        self.evolution_source = source
        self.evolution_result = result
        self._center_track_cache.clear()
        self.evolution_count.setVisible(bool(
            result and result.trajectory_epochs is not None
            and result.trajectory_states is not None
        ))
        self.view_center.setEnabled(bool(
            result and result.trajectory_epochs is not None
            and result.trajectory_states is not None
        ))
        if result is None:
            self.view_center.setCurrentIndex(self.view_center.findData("sun"))

    def _view_center_changed(self, *_args) -> None:
        center = str(self.view_center.currentData() or "sun")
        if center != "sun":
            enabled = self.evolution_count.findData(-1)
            if enabled >= 0 and int(self.evolution_count.currentData()) == 0:
                self.evolution_count.setCurrentIndex(enabled)
            target = max(.01, HILL_RADII_AU[center] * 1.6)
            minimum, maximum = .0005, 5.0
            fraction = np.log(np.clip(target, minimum, maximum) / minimum) / np.log(maximum / minimum)
            self.view_zoom.setValue(int(round(fraction * self.view_zoom.maximum())))
        else:
            self.view_zoom.setValue(838)
        self.set_time(self.playback.time.dateTime().toPython())

    def _evolution_display_changed(self, *_args) -> None:
        if (
            int(self.evolution_count.currentData()) == 0
            and str(self.view_center.currentData() or "sun") != "sun"
        ):
            self.view_center.setCurrentIndex(self.view_center.findData("sun"))
            return
        self.set_time(self.playback.time.dateTime().toPython())

    def _center_positions(self, center: str, epochs: np.ndarray) -> np.ndarray:
        if center == "sun":
            return np.zeros((len(epochs), 3), dtype=float)
        key = (center, id(epochs))
        cached = self._center_track_cache.get(key)
        if cached is None:
            cached = np.stack([PLANETS.position(center, float(epoch)) for epoch in epochs])
            self._center_track_cache[key] = cached
        return cached

    def _integrated_trajectory(self) -> tuple[np.ndarray, np.ndarray] | None:
        result = self.evolution_result
        if (
            result is None or result.trajectory_epochs is None
            or result.trajectory_states is None
            or int(self.evolution_count.currentData()) == 0
        ):
            return None
        epochs = result.trajectory_epochs
        states = result.trajectory_states
        return epochs, states

    @staticmethod
    def _trajectory_position(
        trajectory: tuple[np.ndarray, np.ndarray] | None, epoch_jd: float,
    ) -> np.ndarray | None:
        if trajectory is None:
            return None
        epochs, states = trajectory
        if epoch_jd < epochs[0] or epoch_jd > epochs[-1]:
            return None
        return np.array([
            np.interp(epoch_jd, epochs, states[:, coordinate]) for coordinate in range(3)
        ])

    def _visible_radius_au(self) -> float:
        # Logarithmic scale: sub-AU changes get fine control while the same
        # slider still reaches well beyond Neptune.
        if str(self.view_center.currentData() or "sun") == "sun":
            minimum, maximum = .06, 120.0
        else:
            minimum, maximum = .0005, 5.0
        fraction = self.view_zoom.value() / self.view_zoom.maximum()
        return minimum * (maximum / minimum) ** fraction

    def set_time(self, time: datetime) -> None:
        ax = self.canvas.axes
        ax.clear()
        ax.set_facecolor("#0e1828")
        azimuth = self.view_azimuth.value() % 360
        elevation = self.view_elevation.value()
        self.azimuth_readout.setText(f"{azimuth}°")
        self.elevation_readout.setText(f"{elevation:+d}°")
        limit = self._visible_radius_au()
        self.zoom_readout.setText(f"± {limit:.3f} AU" if limit < 10 else f"± {limit:.1f} AU")
        plot_limit = limit * 1.06
        viewport_aspect = float(np.clip(
            self.canvas.width() / max(self.canvas.height(), 1), 1.0, 3.0
        ))
        curve_radius = plot_limit * viewport_aspect
        # Never pre-clip a closed ellipse in heliocentric 3-D space.  A point
        # outside that radial sphere can still project inside the rectangular
        # viewport when the camera is tilted; pre-clipping therefore produced
        # apparently broken arcs with their ends floating inside the plot.
        # Open conics still need a finite plotting radius near their asymptote.
        current_jd = datetime_to_jd(time)
        center = str(self.view_center.currentData() or "sun")
        center_label = dict(ELEMENT_CENTERS).get(center, center)
        center_now = (
            np.zeros(3, dtype=float) if center == "sun"
            else PLANETS.position(center, current_jd)
        )
        trajectory = self._integrated_trajectory()
        propagated_current: np.ndarray | None = None
        if trajectory is not None:
            trajectory_epochs, trajectory_states = trajectory
            # Plot at most 6000 points; this is visually continuous while a
            # multi-century integration remains responsive during animation.
            stride = max(1, int(np.ceil(len(trajectory_epochs) / 6000)))
            plotted_epochs = trajectory_epochs[::stride]
            center_track = self._center_positions(center, trajectory_epochs)[::stride]
            xyz = trajectory_states[::stride, :3] - center_track
            # Let the axes clip the projected curve.  Radial pre-clipping can
            # cut a tilted orbit whose distant part still projects into view.
            valid = np.all(np.isfinite(xyz), axis=1)
            valid_indices = np.flatnonzero(valid)
            breaks = np.where(np.diff(valid_indices) > 1)[0] + 1
            orbit_segments = [segment for segment in np.split(valid_indices, breaks)
                              if len(segment) > 1]
        else:
            # The moving marker is N-body propagated.  Drawing the ellipse
            # from elements at the original fit epoch (often a previous
            # apparition) therefore put the marker visibly off its own orbit.
            # Build the displayed osculating conic from the same instantaneous
            # propagated state as the marker.
            curve_solution = self.solution
            try:
                velocity_step = .02
                local_epochs = np.array([
                    current_jd - velocity_step, current_jd,
                    current_jd + velocity_step,
                ])
                local_positions = heliocentric_xyz(self.solution, local_epochs)
                propagated_current = local_positions[1]
                velocity = (
                    local_positions[2] - local_positions[0]
                ) / (2.0 * velocity_step)
                elements = cartesian_to_elements(propagated_current, velocity)
                curve_solution = replace(
                    self.solution, epoch_jd=current_jd,
                    semi_major_axis_au=elements[0], eccentricity=elements[1],
                    inclination_deg=elements[2], node_deg=elements[3],
                    arg_peri_deg=elements[4], mean_anomaly_deg=elements[5],
                    position_au=tuple(float(value) for value in propagated_current),
                    velocity_au_per_day=tuple(float(value) for value in velocity),
                )
            except (ValueError, RuntimeError, FloatingPointError, OrbitParseError):
                pass
            orbit_radius = solar_orbit_clip_radius(curve_solution, curve_radius)
            xyz, valid = orbital_curve_xyz(curve_solution, max_radius_au=orbit_radius)
            plotted_epochs = None
            valid_indices = np.flatnonzero(valid)
            breaks = np.where(np.diff(valid_indices) > 1)[0] + 1
            orbit_segments = [segment for segment in np.split(valid_indices, breaks)
                              if len(segment) > 1]
        switch_encounters = ()
        if self.evolution_result is not None:
            # Limit labels and visual branch changes to the strongest recorded
            # encounters so a centuries-long integration remains legible.
            switch_encounters = tuple(sorted(
                self.evolution_result.encounters,
                key=lambda item: (
                    abs(item.delta_reciprocal_a or 0.0),
                    1.0 / max(item.hill_radii, 1e-9),
                ),
                reverse=True,
            )[:12])
        if orbit_segments:
            if self.show_orbit_curtain.isChecked():
                for segment in orbit_segments:
                    segment_xyz = xyz[segment]
                    # Split at the ascending and descending nodes.  Each
                    # resulting ribbon is a simple translucent surface rather
                    # than one self-intersecting polygon around the full orbit.
                    node_breaks = np.where(
                        np.signbit(segment_xyz[:-1, 2])
                        != np.signbit(segment_xyz[1:, 2])
                    )[0] + 1
                    for arc in np.split(segment_xyz, node_breaks):
                        if len(arc) < 2:
                            continue
                        # Limit the surface geometry so a centuries-long
                        # integrated trajectory remains responsive.
                        surface_stride = max(1, int(np.ceil(len(arc) / 700)))
                        sampled_arc = arc[::surface_stride]
                        if not np.array_equal(sampled_arc[-1], arc[-1]):
                            sampled_arc = np.vstack((sampled_arc, arc[-1]))
                        arc = sampled_arc
                        base = arc.copy()
                        base[:, 2] = 0.0
                        arc_x, arc_y, _ = project_ecliptic_camera(
                            arc, azimuth, elevation
                        )
                        base_x, base_y, _ = project_ecliptic_camera(
                            base, azimuth, elevation
                        )
                        # A muted blue above and violet below the ecliptic
                        # reveal both nodes while remaining distinct from the
                        # cyan/magenta past/future trajectory colours.
                        surface_color = (
                            "#579bd1" if float(np.mean(arc[:, 2])) >= 0.0
                            else "#9a65bd"
                        )
                        height_scale = max(
                            float(np.max(np.abs(arc[:, 2]))), 1e-12
                        )
                        ax.fill(
                            np.concatenate((arc_x, base_x[::-1])),
                            np.concatenate((arc_y, base_y[::-1])),
                            facecolor=surface_color, edgecolor="none",
                            alpha=.16, zorder=.28,
                        )
                        # The footprint plus a handful of very fine vertical
                        # connectors act as depth cues without recreating the
                        # dense striped curtain the user wanted to avoid.
                        ax.plot(
                            base_x, base_y, color="#7895b6", lw=.55,
                            alpha=.28, ls=(0, (2, 5)), zorder=.29,
                        )
                        rib_count = min(10, len(arc))
                        rib_indices = np.unique(np.linspace(
                            0, len(arc) - 1, rib_count, dtype=int
                        ))
                        ribs = [
                            ((base_x[index], base_y[index]),
                             (arc_x[index], arc_y[index]))
                            for index in rib_indices
                            if abs(float(arc[index, 2])) > height_scale * .04
                        ]
                        if ribs:
                            ax.add_collection(LineCollection(
                                ribs, colors="#9bb6d0", linewidths=.42,
                                alpha=.20, zorder=.31,
                            ))
            for segment in orbit_segments:
                if len(segment) > 1:
                    screen_x, screen_y, _ = project_ecliptic_camera(
                        xyz[segment], azimuth, elevation
                    )
                    if plotted_epochs is None:
                        ax.plot(screen_x, screen_y, color="#52a9ff", lw=1.55, alpha=.92)
                    else:
                        segment_epochs = plotted_epochs[segment]
                        return_epochs = np.asarray([
                            item.epoch_jd
                            for item in (self.evolution_result.returns
                                         if self.evolution_result else ())
                        ], dtype=float)
                        encounter_epochs = np.asarray([
                            item.epoch_jd
                            for item in switch_encounters
                        ], dtype=float)
                        older, active, newer = trajectory_return_masks(
                            segment_epochs, return_epochs, current_jd,
                            encounter_epochs,
                        )
                        # Inactive revolutions form a quiet historical/future
                        # context. Draw the currently occupied revolution last
                        # and prominently so a multi-century track remains
                        # readable while playback moves between returns.
                        styles = (
                            (older, "#1ab7ad", .72, .28, 1.1),
                            (newer, "#a85dcc", .72, .28, 1.1),
                            (active, "#71e6ff", 2.35, .98, 2.4),
                        )
                        for mask, colour, width, alpha, zorder in styles:
                            indices = np.flatnonzero(mask)
                            if len(indices) < 2:
                                continue
                            colour_breaks = np.where(np.diff(indices) > 1)[0] + 1
                            for colour_segment in np.split(indices, colour_breaks):
                                if len(colour_segment) > 1:
                                    ax.plot(screen_x[colour_segment], screen_y[colour_segment],
                                            color=colour, lw=width, alpha=alpha,
                                            zorder=zorder)
        if trajectory is not None and switch_encounters:
            # Mark the strongest visible switches.  The integrated path itself
            # stays continuous; the marker explains where a planetary impulse
            # transfers emphasis from the incoming to the outgoing branch.
            for encounter in switch_encounters:
                encounter_position = self._trajectory_position(
                    trajectory, encounter.epoch_jd,
                )
                if encounter_position is None:
                    continue
                encounter_center = (
                    np.zeros(3, dtype=float) if center == "sun"
                    else PLANETS.position(center, encounter.epoch_jd)
                )
                encounter_relative = encounter_position - encounter_center
                if np.linalg.norm(encounter_relative) > curve_radius * 1.15:
                    continue
                switch_x, switch_y, _ = project_ecliptic_camera(
                    encounter_relative, azimuth, elevation,
                )
                ax.scatter(
                    switch_x, switch_y, marker="D", s=25,
                    color="#ffd166", edgecolor="#fff4c2", linewidth=.55,
                    alpha=.9, zorder=5.5,
                )
                encounter_date = datetime_from_jd(encounter.epoch_jd)
                encounter_year = (
                    str(encounter_date.year) if encounter_date is not None else "?"
                )
                ax.annotate(
                    f"{PLANET_LABELS.get(encounter.planet, encounter.planet)} "
                    f"{encounter_year}",
                    (switch_x, switch_y), xytext=(5, -11),
                    textcoords="offset points", color="#e9cb78", fontsize=7,
                    alpha=.85,
                )
        planets = (("mercury", tr("Merkur"), .241, "#aaaeb5"), ("venus", tr("Venuše"), .615, "#e4bd70"),
                   ("earth", tr("Země"), 1.0, "#4ea1ff"), ("mars", "Mars", 1.881, "#e0714c"),
                   ("jupiter", "Jupiter", 11.86, "#d0a17a"), ("saturn", "Saturn", 29.46, "#d8c58b"),
                   ("uranus", tr("Uran"), 84.01, "#71d5de"), ("neptune", tr("Neptun"), 164.8, "#537be8"))
        for key, name, years, color in planets:
            relative_position = PLANETS.position(key, current_jd) - center_now
            if np.linalg.norm(relative_position) > curve_radius * 1.15:
                continue
            if key != center:
                cache_key = (key, center)
                path = self._planet_path_cache.get(cache_key)
                if path is None:
                    planet_jd = np.linspace(
                        current_jd, current_jd + years * 365.256, 360
                    )
                    path = np.stack([
                        PLANETS.position(key, float(epoch)) for epoch in planet_jd
                    ])
                    if center != "sun":
                        path -= np.stack([
                            PLANETS.position(center, float(epoch))
                            for epoch in planet_jd
                        ])
                    self._planet_path_cache[cache_key] = path
                path_x, path_y, _ = project_ecliptic_camera(path, azimuth, elevation)
                ax.plot(path_x, path_y, color=color, alpha=.7, lw=1.15)
            px, py, _ = project_ecliptic_camera(relative_position, azimuth, elevation)
            ax.scatter(px, py, s=30 if key != "jupiter" else 46, color=color,
                       edgecolor="white", linewidth=.45, zorder=5)
            ax.annotate(name, (px, py), xytext=(4, 4), textcoords="offset points",
                        color=color, fontsize=8)
        heliocentric_current = self._trajectory_position(trajectory, current_jd)
        if heliocentric_current is None and propagated_current is not None:
            heliocentric_current = propagated_current
        if heliocentric_current is None:
            heliocentric_current = heliocentric_xyz(self.solution, np.array([current_jd]))[0]
        current = heliocentric_current - center_now
        object_x, object_y, _ = project_ecliptic_camera(current, azimuth, elevation)
        ax.scatter(object_x, object_y, s=42, color="#7fe7ff", edgecolor="white",
                   linewidth=.6, zorder=7)
        ax.annotate(self.solution.designation, (object_x, object_y), xytext=(6, 6),
                    textcoords="offset points", color="#7fe7ff", fontsize=9, fontweight="bold")
        sun_position = -center_now
        sun_x, sun_y, _ = project_ecliptic_camera(sun_position, azimuth, elevation)
        ax.scatter(sun_x, sun_y, s=95, color="#ffd166", edgecolor="#fff0b3",
                   linewidth=.8, zorder=8)
        if center != "sun" and np.linalg.norm(sun_position) <= curve_radius * 1.15:
            ax.annotate(tr("Slunce"), (sun_x, sun_y), xytext=(5, 5), textcoords="offset points",
                        color="#ffd166", fontsize=8)
        axis_limit = plot_limit
        ax.set_xlim(-axis_limit * viewport_aspect, axis_limit * viewport_aspect)
        ax.set_ylim(-axis_limit, axis_limit)
        ax.set_aspect("equal", adjustable="box")
        # Fixed heliocentric J2000 ecliptic reference frame.  Only the camera
        # moves; these axes never rotate relative to the orbital coordinates.
        reference = axis_limit * .88
        axis_vectors = np.array([
            [-reference, 0, 0], [reference, 0, 0],
            [0, -reference, 0], [0, reference, 0],
            [0, 0, -reference], [0, 0, reference],
        ])
        axis_x, axis_y, _ = project_ecliptic_camera(axis_vectors, azimuth, elevation)
        for start, color in ((0, "#b85b62"), (2, "#5ea978"), (4, "#668fd3")):
            ax.plot(axis_x[start:start + 2], axis_y[start:start + 2], color=color, lw=.8, alpha=.7)
        for index, label, color in (
            (1, tr("+X  ♈ jarní bod"), "#d9878d"),
            (3, "+Y  λ=90°", "#83c99a"),
            (5, tr("+Z  sever ekliptiky"), "#8eb0eb"),
        ):
            ax.annotate(label, (axis_x[index], axis_y[index]), xytext=(4, 3),
                        textcoords="offset points", color=color, fontsize=8)
        ax.set_title(
            tr("{0} — {1}-centrický pohled — ", self.solution.designation, center_label)
            + time.strftime("%Y-%m-%d %H:%M UTC"),
            color="#edf5ff", y=.965, pad=0,
        )
        ax.set_axis_off()
        earth = PLANETS.position("earth", current_jd)
        distance_earth = float(np.linalg.norm(heliocentric_current - earth))
        distance_sun = float(np.linalg.norm(heliocentric_current))
        center_distance = float(np.linalg.norm(current))
        center_text = "" if center == "sun" else tr(
            "    od {0} {1:.6f} AU",
            center_label,
            center_distance,
        )
        self.info.setText(
            tr(
                "{0:%Y-%m-%d %H:%M UTC}    Δ od Země {1:.6f} AU    r od Slunce {2:.6f} "
                "AU{3}",
                time,
                distance_earth,
                distance_sun,
                center_text,
            )
        )
        self.canvas.draw_idle()


def _load_sky_catalog() -> tuple[np.ndarray, list[np.ndarray], list[tuple[str, float, float]]]:
    data = Path(__file__).with_name("data")
    stars_json = json.loads((data / "stars.6.json").read_text(encoding="utf-8"))
    lines_json = json.loads((data / "constellations.lines.json").read_text(encoding="utf-8"))
    stars = []
    for feature in stars_json["features"]:
        ra, dec = feature["geometry"]["coordinates"]
        stars.append((float(ra) % 360.0, float(dec), float(feature["properties"]["mag"])))
    lines: list[np.ndarray] = []
    label_points: dict[str, list[np.ndarray]] = {}
    for feature in lines_json["features"]:
        abbreviation = str(feature.get("id") or "").strip()
        for coordinates in feature["geometry"]["coordinates"]:
            # Several real constellation strokes cross RA=0 (notably the
            # Great Square of Pegasus and Pisces).  Keep those adjacent
            # vertices together; each map projection below already splits a
            # stroke only when it crosses that projection's visible seam.
            current = np.asarray([
                (float(ra) % 360.0, float(dec)) for ra, dec in coordinates
            ])
            if len(current) > 1:
                lines.append(current)
                label_points.setdefault(abbreviation, []).append(current)
    labels: list[tuple[str, float, float]] = []
    for abbreviation, groups in label_points.items():
        points = np.vstack(groups)
        ra = np.radians(points[:, 0])
        dec = np.radians(points[:, 1])
        vectors = np.column_stack((
            np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec),
        ))
        center = np.mean(vectors, axis=0)
        center /= np.linalg.norm(center)
        labels.append((
            abbreviation,
            float(np.degrees(np.arctan2(center[1], center[0])) % 360.0),
            float(np.degrees(np.arcsin(center[2]))),
        ))
    return np.asarray(stars), lines, labels


BRIGHT_STARS, CONSTELLATION_LINES, CONSTELLATION_LABELS = _load_sky_catalog()

_reference_ra = np.linspace(0.0, 360.0, 721)
_reference_lambda = np.radians(_reference_ra)
_obliquity = np.radians(23.4392911)
_ecliptic_ra = np.degrees(np.arctan2(
    np.sin(_reference_lambda) * np.cos(_obliquity), np.cos(_reference_lambda)
)) % 360.0
_ecliptic_dec = np.degrees(np.arcsin(np.sin(_reference_lambda) * np.sin(_obliquity)))
SKY_REFERENCE_CURVES = (
    (tr("Nebeský rovník"), _reference_ra, np.zeros_like(_reference_ra), "#788bb0", "--"),
    (tr("Ekliptika"), _ecliptic_ra, _ecliptic_dec, "#c6a45b", "-."),
)


class SkyPanel(QWidget):
    def __init__(
        self, solution: OrbitSolution, playback: PlaybackBar, allow_detach: bool = True,
    ) -> None:
        super().__init__()
        self.solution = solution
        self.playback = playback
        self._detached_windows: list[QMainWindow] = []
        self.free_center_ra: float | None = None
        self.free_center_dec: float | None = None
        self._map_drag: tuple[float, float, float, float] | None = None
        self._sky_track_cache: tuple[
            int, int, float, float, np.ndarray, dict[str, np.ndarray]
        ] | None = None
        self.canvas = PlotCanvas()
        self.canvas.figure.set_layout_engine(None)
        self.canvas.axes.set_position([.055, .075, .935, .86])
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(120)
        self._resize_timer.timeout.connect(
            lambda: self.rebuild(self.playback.time.dateTime().toPython())
        )
        layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        self.projection_mode = QComboBox()
        self.projection_mode.addItem(tr("Rovníkový (RA/Dec)"), "equatorial")
        self.projection_mode.addItem(tr("Středový azimutální"), "azimuthal")
        self.projection_mode.addItem(tr("Obzorový (Az/Alt)"), "horizontal")
        self.horizon = QComboBox()
        for label, days in SKY_TRACK_HORIZONS:
            self.horizon.addItem(label, days)
        self.horizon.setToolTip(
            tr(
                "Dlouhé rozsahy jsou určeny hlavně pro přehled. Přesnost vzdálené "
                "extrapolace závisí na nejistotě dráhy a planetárních perturbacích.",
            )
        )
        self.field_of_view = QComboBox()
        for angle in (15, 30, 60, 120, 180, 240):
            self.field_of_view.addItem(f"{angle}°", float(angle))
        self.field_of_view.addItem(tr("Celá obloha"), 360.0)
        self.field_of_view.setCurrentIndex(2)
        self.horizontal_horizon = QComboBox()
        for label, days in (
            (tr("12 hodin celkem"), .25), (tr("24 hodin celkem"), .5),
            (tr("2 dny celkem"), 1.0), (tr("6 dní celkem"), 3.0),
            (tr("14 dní celkem"), 7.0),
        ):
            self.horizontal_horizon.addItem(label, days)
        self.horizontal_horizon.setCurrentIndex(1)
        self.horizontal_time_label = QLabel(tr("Střed intervalu:"))
        self.horizontal_time = QDateTimeEdit(self.playback.time.dateTime())
        self.horizontal_time.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.horizontal_time.setCalendarPopup(True)
        self.horizontal_time.setFixedWidth(160)
        self.horizontal_time.setToolTip(
            tr("Datum a čas uprostřed zobrazeného intervalu v UTC.")
        )
        self.horizontal_timezone = QComboBox()
        self.horizontal_timezone.addItem("UTC", 0)
        self.horizontal_timezone.addItem(tr("SEČ"), 1)
        self.horizontal_timezone.addItem(tr("SELČ"), 2)
        self.horizontal_timezone.setToolTip(
            tr("Časové pásmo použité pro zadaný čas i popisky grafu.")
        )
        self.station_code = QLineEdit("557")
        self.station_code.setMaximumWidth(68)
        self.station_code.setMaxLength(3)
        self.station_code.setToolTip(
            tr(
                "Kód pozorovací stanice MPC; 557 = Ondřejov. Výška je geometrická, bez "
                "refrakce.",
            )
        )
        self.station_name = QLabel()
        self.station_name.setObjectName("muted")
        self.station_name.setMaximumWidth(190)
        self.show_path = QCheckBox(tr("Zobrazit dráhu")); self.show_path.setChecked(True)
        self.show_labels = QCheckBox(tr("Popisky těles")); self.show_labels.setChecked(True)
        self.lock_target = QCheckBox(tr("Fixovat objekt ve středu")); self.lock_target.setChecked(True)
        self.lock_target.setToolTip(
            tr(
                "Vypnutím se souřadnicové pole uzamkne v aktuálním směru a objekt se po "
                "něm bude pohybovat.",
            )
        )
        self.range_label = QLabel(tr("Rozsah:"))
        self.fov_label = QLabel(tr("Zorné pole:"))
        self.station_label = QLabel(tr("Stanice MPC:"))
        controls.addWidget(QLabel(tr("Pohled:"))); controls.addWidget(self.projection_mode)
        controls.addWidget(self.station_label); controls.addWidget(self.station_code)
        controls.addWidget(self.station_name)
        controls.addStretch(); controls.addWidget(self.range_label)
        controls.addWidget(self.horizon); controls.addWidget(self.horizontal_horizon)
        controls.addWidget(self.horizontal_time_label)
        controls.addWidget(self.horizontal_time)
        controls.addWidget(self.horizontal_timezone)
        controls.addWidget(self.fov_label)
        controls.addWidget(self.field_of_view); controls.addWidget(self.show_path)
        controls.addWidget(self.show_labels); controls.addWidget(self.lock_target)
        enlarge = QPushButton(tr("Zvětšit ↗") if allow_detach else tr("Celá obrazovka (F11)"))
        enlarge.setToolTip(
            tr(
                "Otevře mapu ve zvětšeném samostatném okně. Klávesou F11 přepnete celou "
                "obrazovku.",
            )
        )
        enlarge.clicked.connect(self._open_detached if allow_detach else self._toggle_fullscreen)
        controls.addWidget(enlarge)
        layout.addLayout(controls)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self.info = QLabel(); self.info.setObjectName("value")
        self.info.setStyleSheet("padding: 7px 10px; background: #111d30; border-radius: 6px;")
        self.info.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        self.info.setMinimumWidth(0)
        self.info.setFixedHeight(42)
        layout.addWidget(self.info)
        layout.addWidget(playback)
        playback.time_changed.connect(self.set_time)
        playback.time_changed.connect(self._sync_horizontal_time)
        self.horizontal_time.dateTimeChanged.connect(self._horizontal_time_changed)
        self.horizontal_timezone.currentIndexChanged.connect(self._horizontal_timezone_changed)
        self.horizon.currentIndexChanged.connect(lambda _: self.rebuild(self.playback.time.dateTime().toPython()))
        self.horizontal_horizon.currentIndexChanged.connect(
            lambda _: self.rebuild(self.playback.time.dateTime().toPython())
        )
        self.projection_mode.currentIndexChanged.connect(self._projection_changed)
        self.station_code.editingFinished.connect(self._station_changed)
        self.field_of_view.currentIndexChanged.connect(lambda _: self.rebuild(self.playback.time.dateTime().toPython()))
        self.show_path.toggled.connect(lambda _: self.rebuild(self.playback.time.dateTime().toPython()))
        self.show_labels.toggled.connect(lambda _: self.rebuild(self.playback.time.dateTime().toPython()))
        self.lock_target.toggled.connect(self._tracking_changed)
        self.canvas.mpl_connect("button_press_event", self._map_press)
        self.canvas.mpl_connect("motion_notify_event", self._map_motion)
        self.canvas.mpl_connect("button_release_event", self._map_release)
        self.fullscreen_shortcut = QShortcut(QKeySequence("F11"), self)
        self.fullscreen_shortcut.activated.connect(self._toggle_fullscreen)
        self.escape_shortcut = QShortcut(QKeySequence("Escape"), self)
        self.escape_shortcut.activated.connect(self._leave_fullscreen)
        self._projection_changed()
        self.rebuild(datetime.now())

    def _projection_changed(self, *_args) -> None:
        horizontal = self.projection_mode.currentData() == "horizontal"
        self.horizon.setVisible(not horizontal)
        self.horizontal_horizon.setVisible(horizontal)
        self.station_label.setVisible(horizontal)
        self.station_code.setVisible(horizontal)
        self.station_name.setVisible(horizontal)
        self.horizontal_time_label.setVisible(horizontal)
        self.horizontal_time.setVisible(horizontal)
        self.horizontal_timezone.setVisible(horizontal)
        self.fov_label.setVisible(not horizontal)
        self.field_of_view.setVisible(not horizontal)
        self.lock_target.setVisible(not horizontal)
        self.show_labels.setText(tr("Popisek objektu") if horizontal else tr("Popisky těles"))
        self.playback.set_speed_profile("visibility" if horizontal else "standard")
        if horizontal:
            self._update_station_name()
        self.rebuild(self.playback.time.dateTime().toPython())

    def _sync_horizontal_time(self, _time: datetime) -> None:
        self.horizontal_time.blockSignals(True)
        offset_hours = int(self.horizontal_timezone.currentData())
        self.horizontal_time.setDateTime(
            self.playback.time.dateTime().addSecs(offset_hours * 3600)
        )
        self.horizontal_time.blockSignals(False)

    def _horizontal_time_changed(self, value) -> None:
        offset_hours = int(self.horizontal_timezone.currentData())
        self.playback.time.setDateTime(value.addSecs(-offset_hours * 3600))

    def _horizontal_timezone_changed(self, *_args) -> None:
        zone = self.horizontal_timezone.currentText()
        self.horizontal_time.setToolTip(
            tr("Datum a čas uprostřed zobrazeného intervalu v pásmu {0}.", zone)
        )
        self._sync_horizontal_time(self.playback.time.dateTime().toPython())
        self.rebuild(self.playback.time.dateTime().toPython())

    def _station_changed(self) -> None:
        self.station_code.setText(self.station_code.text().strip().upper())
        self._update_station_name()
        self.rebuild(self.playback.time.dateTime().toPython())

    def _update_station_name(self) -> None:
        details = observatory_details(self.station_code.text())
        name = str(details["name"]) if details and "latitude_deg" in details else tr(
            "neznámá stanice",
        )
        self.station_name.setText(name)
        self.station_name.setToolTip(name)

    def _open_detached(self) -> None:
        window = QMainWindow(self)
        window.setAttribute(Qt.WA_DeleteOnClose)
        window.setWindowTitle(tr("{0} — pohled ze Země", self.solution.designation))
        playback = PlaybackBar()
        playback.time.setDateTime(self.playback.time.dateTime())
        panel = SkyPanel(self.solution, playback, allow_detach=False)
        panel.horizon.setCurrentIndex(panel.horizon.findData(self.horizon.currentData()))
        panel.horizontal_horizon.setCurrentIndex(
            panel.horizontal_horizon.findData(self.horizontal_horizon.currentData())
        )
        panel.horizontal_timezone.setCurrentIndex(
            panel.horizontal_timezone.findData(self.horizontal_timezone.currentData())
        )
        panel.projection_mode.setCurrentIndex(
            panel.projection_mode.findData(self.projection_mode.currentData())
        )
        panel.station_code.setText(self.station_code.text())
        panel._update_station_name()
        panel.field_of_view.setCurrentIndex(
            panel.field_of_view.findData(self.field_of_view.currentData())
        )
        panel.show_path.setChecked(self.show_path.isChecked())
        panel.show_labels.setChecked(self.show_labels.isChecked())
        panel.lock_target.setChecked(self.lock_target.isChecked())
        panel.free_center_ra = self.free_center_ra
        panel.free_center_dec = self.free_center_dec
        panel.rebuild(playback.time.dateTime().toPython())
        window.setCentralWidget(panel)
        window.resize(1500, 950)
        self._detached_windows.append(window)
        window.destroyed.connect(
            lambda _=None, item=window: self._detached_windows.remove(item)
            if item in self._detached_windows else None
        )
        window.showMaximized()

    def _toggle_fullscreen(self) -> None:
        window = self.window()
        window.showNormal() if window.isFullScreen() else window.showFullScreen()

    def _leave_fullscreen(self) -> None:
        if self.window().isFullScreen():
            self.window().showNormal()

    def _map_press(self, event) -> None:
        if (
            self.projection_mode.currentData() == "horizontal" or
            event.button != 1 or event.inaxes is not self.canvas.axes or
            event.xdata is None or event.ydata is None or self.toolbar.mode
        ):
            return
        if self.lock_target.isChecked():
            self.lock_target.setChecked(False)
        if self.free_center_ra is None or self.free_center_dec is None:
            return
        self._map_drag = (
            float(event.xdata), float(event.ydata),
            float(self.free_center_ra), float(self.free_center_dec),
        )

    def _map_motion(self, event) -> None:
        if self._map_drag is None or event.xdata is None or event.ydata is None:
            return
        start_x, start_y, start_ra, start_dec = self._map_drag
        delta_x = float(event.xdata) - start_x
        delta_y = float(event.ydata) - start_y
        if self.projection_mode.currentData() == "equatorial":
            self.free_center_ra = (start_ra - delta_x) % 360.0
        else:
            cosine = max(.15, abs(float(np.cos(np.radians(start_dec)))))
            self.free_center_ra = (start_ra - delta_x / cosine) % 360.0
        self.free_center_dec = float(np.clip(start_dec - delta_y, -89.5, 89.5))
        self.rebuild(self.playback.time.dateTime().toPython())

    def _map_release(self, event) -> None:
        self._map_drag = None

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if hasattr(self, "_resize_timer"):
            self._resize_timer.start()

    def set_solution(self, solution: OrbitSolution) -> None:
        self.solution = solution
        self.free_center_ra = None
        self.free_center_dec = None
        self._sky_track_cache = None
        self.rebuild(self.playback.time.dateTime().toPython())

    def _sky_track_geometry(
        self, jd0: float, horizon: int,
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Reuse an absolute sky track while the animation moves within it."""
        wanted_start = jd0 - horizon
        wanted_end = jd0 + horizon
        solution_key = id(self.solution)
        cache = self._sky_track_cache
        if (
            cache is None or cache[0] != solution_key or cache[1] != horizon
            or wanted_start < cache[2] or wanted_end > cache[3]
        ):
            # A generous guard interval prevents a costly DE440 track rebuild
            # on every frame.  The displayed part is still exactly the chosen
            # interval around the current date.
            margin = max(100.0, min(1000.0, float(horizon) * 2.0))
            cache_start = wanted_start - margin
            cache_end = wanted_end + margin
            base_count = sky_track_sample_count(horizon)
            base_step = 2.0 * horizon / max(base_count - 1, 1)
            cache_count = min(
                6001,
                max(base_count + 2, int(np.ceil(
                    (cache_end - cache_start) / max(base_step, 1e-6)
                )) + 1),
            )
            cached_jd = np.linspace(cache_start, cache_end, cache_count)
            cached_geometry = apparent_geometry(self.solution, cached_jd)
            cache = (
                solution_key, horizon, cache_start, cache_end,
                cached_jd, cached_geometry,
            )
            self._sky_track_cache = cache
        cached_jd = cache[4]
        mask = (cached_jd >= wanted_start) & (cached_jd <= wanted_end)
        jd = cached_jd[mask]
        geometry = {
            key: np.asarray(values)[mask] for key, values in cache[5].items()
        }
        return jd, geometry

    def set_time(self, time: datetime) -> None:
        self.rebuild(time)

    def _tracking_changed(self, locked: bool) -> None:
        if locked:
            self.free_center_ra = None
            self.free_center_dec = None
        else:
            time = self.playback.time.dateTime().toPython()
            now = apparent_geometry(self.solution, np.array([datetime_to_jd(time)]))
            self.free_center_ra = float(now["ra"][0])
            self.free_center_dec = float(now["dec"][0])
        self.rebuild(self.playback.time.dateTime().toPython())

    @staticmethod
    def _project_azimuthal(
        ra_deg: np.ndarray | float, dec_deg: np.ndarray | float,
        center_ra_deg: float, center_dec_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Azimuthal-equidistant projection about the moving target."""
        ra = np.radians(np.asarray(ra_deg, dtype=float))
        dec = np.radians(np.asarray(dec_deg, dtype=float))
        ra0 = np.radians(center_ra_deg)
        dec0 = np.radians(center_dec_deg)
        delta_ra = (ra - ra0 + np.pi) % (2.0 * np.pi) - np.pi
        cos_distance = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(delta_ra)
        distance = np.arccos(np.clip(cos_distance, -1.0, 1.0))
        sin_distance = np.sin(distance)
        scale = np.divide(distance, sin_distance, out=np.ones_like(distance), where=np.abs(sin_distance) > 1e-12)
        x = scale * np.cos(dec) * np.sin(delta_ra)
        y = scale * (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(delta_ra))
        return np.degrees(x), np.degrees(y), np.degrees(distance)

    @staticmethod
    def _project_equatorial(
        ra_deg: np.ndarray | float, dec_deg: np.ndarray | float,
        center_ra_deg: float, center_dec_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Rectangular equatorial map centred in RA, with absolute declination."""
        ra = np.asarray(ra_deg, dtype=float)
        dec = np.asarray(dec_deg, dtype=float)
        delta_ra = (ra - center_ra_deg + 180.0) % 360.0 - 180.0
        ra_rad = np.radians(ra)
        dec_rad = np.radians(dec)
        ra0 = np.radians(center_ra_deg)
        dec0 = np.radians(center_dec_deg)
        cos_distance = (
            np.sin(dec0) * np.sin(dec_rad) +
            np.cos(dec0) * np.cos(dec_rad) * np.cos(ra_rad - ra0)
        )
        distance = np.degrees(np.arccos(np.clip(cos_distance, -1.0, 1.0)))
        return delta_ra, dec, distance

    @staticmethod
    def _project_hammer(
        ra_deg: np.ndarray | float, dec_deg: np.ndarray | float,
        center_ra_deg: float, center_dec_deg: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Target-centred, full-sky Hammer projection for wide displays."""
        ra = np.radians(np.asarray(ra_deg, dtype=float))
        dec = np.radians(np.asarray(dec_deg, dtype=float))
        ra0, dec0 = np.radians((center_ra_deg, center_dec_deg))
        vx = np.cos(dec) * np.cos(ra)
        vy = np.cos(dec) * np.sin(ra)
        vz = np.sin(dec)
        center = np.array([np.cos(dec0) * np.cos(ra0), np.cos(dec0) * np.sin(ra0), np.sin(dec0)])
        east = np.array([-np.sin(ra0), np.cos(ra0), 0.0])
        north = np.cross(center, east)
        forward_component = vx * center[0] + vy * center[1] + vz * center[2]
        east_component = vx * east[0] + vy * east[1] + vz * east[2]
        north_component = vx * north[0] + vy * north[1] + vz * north[2]
        longitude = np.arctan2(east_component, forward_component)
        latitude = np.arcsin(np.clip(north_component, -1.0, 1.0))
        denominator = np.sqrt(np.maximum(1e-15, 1.0 + np.cos(latitude) * np.cos(longitude / 2.0)))
        scale = 180.0 / (2.0 * np.sqrt(2.0))
        x = 2.0 * np.sqrt(2.0) * np.cos(latitude) * np.sin(longitude / 2.0) / denominator * scale
        y = np.sqrt(2.0) * np.sin(latitude) / denominator * scale
        distance = np.degrees(np.arccos(np.clip(forward_component, -1.0, 1.0)))
        return x, y, distance

    @staticmethod
    def _vector_radec(vector: np.ndarray) -> tuple[float, float]:
        eps = np.radians(23.4392911)
        x, y0, z0 = vector
        y = y0 * np.cos(eps) - z0 * np.sin(eps)
        z = y0 * np.sin(eps) + z0 * np.cos(eps)
        radius = np.linalg.norm(vector)
        return float(np.degrees(np.arctan2(y, x)) % 360.0), float(np.degrees(np.arcsin(z / radius)))

    @staticmethod
    def _moon_vector(jd: float) -> np.ndarray:
        days = jd - 2451545.0
        longitude = np.radians((218.316 + 13.176396 * days + 6.289 * np.sin(np.radians(134.963 + 13.064993 * days))) % 360)
        latitude = np.radians(5.128 * np.sin(np.radians(93.272 + 13.229350 * days)))
        return np.array([np.cos(latitude) * np.cos(longitude), np.cos(latitude) * np.sin(longitude), np.sin(latitude)])

    def _body_positions(self, jd: float) -> list[tuple[str, float, float, str, int]]:
        earth = PLANETS.position("earth", jd)
        bodies = [(tr("Slunce"), -earth, "#ffd166", 72), (tr("Měsíc"), self._moon_vector(jd), "#d9e1ec", 42)]
        for key, name, color in (("mercury", tr("Merkur"), "#aaaeb5"), ("venus", tr("Venuše"), "#e4bd70"),
                                 ("mars", "Mars", "#e0714c"), ("jupiter", "Jupiter", "#d0a17a"),
                                 ("saturn", "Saturn", "#d8c58b"), ("uranus", tr("Uran"), "#71d5de"),
                                 ("neptune", tr("Neptun"), "#537be8")):
            bodies.append((name, PLANETS.position(key, jd) - earth, color, 34))
        result = []
        for name, vector, color, size in bodies:
            ra, dec = self._vector_radec(vector)
            result.append((name, ra, dec, color, size))
        return result

    def _set_info(
        self, time: datetime, geometry: dict[str, np.ndarray], extra: str = "",
        display_time: datetime | None = None, zone_label: str = "UTC",
    ) -> None:
        magnitude = power_law_magnitude(
            self.solution.absolute_magnitude, geometry["delta"], geometry["r"],
            self.solution.photometric_exponent,
        )
        if self.solution.phase_correction:
            magnitude += hg_phase_correction(geometry["phase"], self.solution.slope_g)
        shown_time = display_time if display_time is not None else time
        ra_text = format_ra_sexagesimal(float(geometry["ra"][0]))
        dec_text = format_dec_sexagesimal(float(geometry["dec"][0]))
        self.info.setText(
            tr(
                "{0:%Y-%m-%d %H:%M} {1}    RA {2}    Dec {3}    mag {4:.2f}    Δ {5:.5f} "
                "AU    r {6:.5f} AU    elongace {7:.1f}°{8}",
                shown_time,
                zone_label,
                ra_text,
                dec_text,
                magnitude[0],
                geometry['delta'][0],
                geometry['r'][0],
                geometry['elongation'][0],
                extra,
            )
        )

    @staticmethod
    def _topocentric_coordinates(
        solution: OrbitSolution, jd: np.ndarray, observer_offsets: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        """Return orbit geometry with RA/Dec/distance referred to the station."""
        geom = apparent_geometry(solution, jd)
        earth = np.stack([PLANETS.position("earth", float(epoch)) for epoch in jd])
        geo_ecliptic = geom["xyz"] - earth
        eps = np.radians(23.4392911)
        topocentric = np.column_stack((
            geo_ecliptic[:, 0],
            geo_ecliptic[:, 1] * np.cos(eps) - geo_ecliptic[:, 2] * np.sin(eps),
            geo_ecliptic[:, 1] * np.sin(eps) + geo_ecliptic[:, 2] * np.cos(eps),
        )) - observer_offsets
        delta = np.linalg.norm(topocentric, axis=1)
        ra = np.degrees(np.arctan2(topocentric[:, 1], topocentric[:, 0])) % 360.0
        dec = np.degrees(np.arcsin(np.clip(topocentric[:, 2] / delta, -1.0, 1.0)))
        return geom, ra, dec, delta

    def _show_missing_target(self, time: datetime) -> None:
        """The body no longer exists at ``time`` (it hit a planet)."""
        ax = self.canvas.axes
        ax.clear(); self.canvas.style_axes()
        prediction = getattr(self.solution, "impact", None)
        if prediction is not None:
            point = prediction.entry or prediction.surface
            when = point.utc.strftime("%Y-%m-%d %H:%M:%S UTC") if point.utc else ""
            message = tr(
                "Těleso v tomto okamžiku neexistuje: dopadlo na těleso {0} ({1}).\n"
                "Zvolte čas před dopadem.",
                BODY_LABELS.get(prediction.body, prediction.body), when,
            )
        else:
            message = tr(
                "Těleso v tomto okamžiku neexistuje: jeho dráha předtím narazila do planety.\n"
                "Zvolte dřívější čas.",
            )
        ax.text(.5, .5, message, transform=ax.transAxes, ha="center", va="center",
                color="#ffb3a8", fontsize=13)
        ax.set_axis_off()
        self.info.setText(message.replace("\n", " "))
        self.canvas.draw_idle()

    def _rebuild_horizontal(self, time: datetime) -> None:
        ax = self.canvas.axes
        ax.clear(); self.canvas.style_axes()
        # Two title lines explain the otherwise easily misread diurnal arc.
        self.canvas.axes.set_position([.055, .075, .935, .80])
        code = self.station_code.text().strip().upper()
        details = observatory_details(code)
        if not details or "latitude_deg" not in details or code == "500":
            ax.text(
                .5, .5, tr(
                    "Zadejte platný pozemní kód stanice MPC.\nGeocentrický kód 500 nemá místní "
                    "obzor.",
                ),
                transform=ax.transAxes, ha="center", va="center", color="#dce7f3",
                fontsize=13,
            )
            ax.set_axis_off()
            self.info.setText(tr("Obzorový pohled vyžaduje pozemní stanici se známou polohou."))
            self.canvas.draw_idle()
            return

        jd0 = datetime_to_jd(time)
        span = float(self.horizontal_horizon.currentData())
        sample_count = max(241, min(961, int(span * 192) + 1))
        jd = np.linspace(jd0 - span, jd0 + span, sample_count)
        times = [datetime_from_jd(float(epoch)) or time for epoch in jd]
        offsets = station_offsets_gcrs([code] * len(times), times)
        geom, track_ra, track_dec, track_delta = self._topocentric_coordinates(
            self.solution, jd, offsets
        )
        track_horizontal = observation_altaz_series(code, times, track_ra, track_dec)
        now_offsets = station_offsets_gcrs([code], [time])
        now_geom, now_ra, now_dec, now_delta = self._topocentric_coordinates(
            self.solution, np.array([jd0]), now_offsets
        )
        if not np.isfinite(float(now_ra[0])):
            self._show_missing_target(time)
            return
        now_horizontal = observation_altaz_series(code, [time], now_ra, now_dec)
        if track_horizontal is None or now_horizontal is None:
            return
        track_az, track_alt, sun_alt = track_horizontal
        now_az, now_alt = (float(now_horizontal[0][0]), float(now_horizontal[1][0]))
        now_sun_alt = float(now_horizontal[2][0])
        zone_offset = int(self.horizontal_timezone.currentData())
        zone_name = self.horizontal_timezone.currentText()
        display_times = [
            (instant + timedelta(hours=zone_offset)).replace(tzinfo=None)
            for instant in times
        ]
        display_time = (time + timedelta(hours=zone_offset)).replace(tzinfo=None)

        def plot_track(
            mask: np.ndarray, color: str, style: str, alpha: float, label: str,
        ) -> None:
            indices = np.flatnonzero(mask)
            if not indices.size:
                return
            discontinuity = np.diff(indices) > 1
            discontinuity |= np.abs(np.diff(track_az[indices])) > 180.0
            segments = np.split(indices, np.where(discontinuity)[0] + 1)
            labelled = False
            for segment in segments:
                if len(segment) > 1:
                    # Include one neighbour on each side so adjacent daylight
                    # categories meet without tiny gaps between samples.
                    start = int(segment[0])
                    stop = int(segment[-1]) + 1
                    if start > 0 and abs(track_az[start] - track_az[start - 1]) < 180.0:
                        start -= 1
                    if stop < len(track_alt) and abs(track_az[stop] - track_az[stop - 1]) < 180.0:
                        stop += 1
                    extended = np.arange(start, stop)
                    ax.plot(
                        track_az[extended],
                        track_alt[extended], color=color, ls=style, lw=1.45,
                        alpha=alpha, zorder=2,
                        label=label if not labelled else "_nolegend_",
                    )
                    labelled = True

        total_hours = max(
            1.0, (display_times[-1] - display_times[0]).total_seconds() / 3600.0
        )
        if self.show_path.isChecked():
            plot_track(track_alt < 0.0, "#79505a", "--", .52, tr("pod obzorem"))
            visibility_bands = (
                ((track_alt >= 0.0) & (sun_alt >= 0.0), "#e2b85d", tr("den")),
                ((track_alt >= 0.0) & (sun_alt < 0.0) & (sun_alt >= -6.0),
                 "#d8895b", tr("občanský soumrak")),
                ((track_alt >= 0.0) & (sun_alt < -6.0) & (sun_alt >= -12.0),
                 "#a170b4", tr("nautický soumrak")),
                ((track_alt >= 0.0) & (sun_alt < -12.0) & (sun_alt >= -18.0),
                 "#6687c9", tr("astronomický soumrak")),
                ((track_alt >= 0.0) & (sun_alt < -18.0),
                 "#45d7ff", tr("astronomická noc")),
            )
            for mask, color, label in visibility_bands:
                plot_track(mask, color, "-", .92, label)

            first_hour = display_times[0].replace(minute=0, second=0, microsecond=0)
            if first_hour < display_times[0]:
                first_hour += timedelta(hours=1)
            label_step = max(1, int(np.ceil(total_hours / 30.0)))
            hour_marks: list[datetime] = []
            cursor = first_hour
            hour_counter = 0
            while cursor <= display_times[-1]:
                if hour_counter % label_step == 0:
                    hour_marks.append(cursor)
                cursor += timedelta(hours=1)
                hour_counter += 1
            marker_indices = np.array([
                min(range(len(display_times)), key=lambda index: abs(
                    (display_times[index] - marker).total_seconds()
                ))
                for marker in hour_marks
            ], dtype=int)
            if marker_indices.size:
                marker_colors = np.where(
                    track_alt[marker_indices] < 0.0, "#8b5963",
                    np.where(sun_alt[marker_indices] < -18.0, "#65ddff", "#d9aa65"),
                )
                ax.scatter(
                    track_az[marker_indices],
                    track_alt[marker_indices], s=13, c=marker_colors, zorder=3,
                )
                for marker, index in zip(hour_marks, marker_indices):
                    ax.annotate(
                        marker.strftime("%Hh"),
                        (track_az[index], track_alt[index]), xytext=(2, 4),
                        textcoords="offset points", color="#9db1c7", fontsize=7,
                    )

        target_color = "#32d7ff" if now_alt >= 0.0 else "#c56d78"
        ax.scatter(now_az, now_alt, s=72, color=target_color, edgecolor="white",
                   linewidth=.8, zorder=6)
        if self.show_labels.isChecked():
            ax.annotate(
                self.solution.designation, (now_az, now_alt), xytext=(9, -13),
                textcoords="offset points", color=target_color, fontsize=9,
                fontweight="bold", va="top", zorder=7,
                bbox={"facecolor": "#0e1828", "edgecolor": "none", "alpha": .72, "pad": 1.5},
            )

        ax.axhspan(-90.0, 0.0, color="#321d2b", alpha=.22, zorder=-3)
        ax.axhline(0.0, color="#b1becd", lw=1.0, alpha=.75)
        for altitude in (15, 30, 45, 60, 75):
            ax.axhline(altitude, color="#30445f", lw=.5, alpha=.45, ls=(0, (3, 5)))
        minimum_altitude = -10.0 if float(np.max(track_alt)) >= 0.0 else max(
            -90.0, np.floor(float(np.min(track_alt)) / 10.0) * 10.0
        )
        ax.set_xlim(0.0, 360.0)
        ax.set_ylim(minimum_altitude, 90.0)
        ax.set_xticks(np.arange(0.0, 361.0, 45.0))
        ax.set_xticklabels(tuple(f"{degrees}°" for degrees in range(0, 361, 45)))
        ax.set_yticks(np.arange(max(0.0, minimum_altitude), 91.0, 15.0))
        ax.set_xlabel(tr("azimut"))
        ax.set_ylabel(tr("výška nad obzorem [°]"))
        ax.set_title(
            tr(
                "{0} • Graf viditelnosti • {1} {2} • {3}\nstřed {4:%Y-%m-%d %H:%M} {5}",
                self.solution.designation,
                code,
                details['name'],
                self.horizontal_horizon.currentText(),
                display_time,
                zone_name,
            )
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            legend = ax.legend(
                handles, labels, loc="upper right", ncol=min(3, len(handles)),
                fontsize=8, framealpha=.82, facecolor="#111d30",
                edgecolor="#3a506e", labelcolor="#d6e2f0",
                handlelength=2.5, columnspacing=1.4,
            )
            for text_item in legend.get_texts():
                text_item.set_color("#d6e2f0")
        display_geom = dict(now_geom)
        display_geom["ra"] = now_ra
        display_geom["dec"] = now_dec
        display_geom["delta"] = now_delta
        self._set_info(
            time, display_geom,
            tr(
                "    Az {0:.1f}°    výška {1:+.1f}°    Slunce {2:+.1f}°    stanice {3}",
                now_az,
                now_alt,
                now_sun_alt,
                code,
            ),
            display_time=display_time, zone_label=zone_name,
        )
        self.canvas.draw_idle()

    def rebuild(self, time: datetime) -> None:
        if self.projection_mode.currentData() == "horizontal":
            self._rebuild_horizontal(time)
            return
        ax = self.canvas.axes
        ax.clear(); self.canvas.style_axes()
        self.canvas.axes.set_position([.055, .075, .935, .86])
        jd0 = datetime_to_jd(time)
        horizon = int(self.horizon.currentData())
        jd, geom = self._sky_track_geometry(jd0, horizon)
        now = apparent_geometry(self.solution, np.array([jd0]))
        if not np.isfinite(float(now["ra"][0])):
            self._show_missing_target(time)
            return
        if self.lock_target.isChecked() or self.free_center_ra is None or self.free_center_dec is None:
            center_ra = float(now["ra"][0])
            center_dec = float(now["dec"][0])
            if not self.lock_target.isChecked():
                self.free_center_ra, self.free_center_dec = center_ra, center_dec
        else:
            center_ra, center_dec = self.free_center_ra, self.free_center_dec
        selected_field = float(self.field_of_view.currentData())
        full_sky = selected_field >= 359.0
        half_field = selected_field / 2.0
        viewport_aspect = float(np.clip(
            self.canvas.width() / max(self.canvas.height(), 1), 1.0, 3.0
        ))
        # The selected angle is the horizontal field of view.  Its vertical
        # counterpart follows the actual panel aspect, so 60° stays a genuine
        # 60° opening instead of silently becoming ~120° on a wide monitor.
        horizontal_half = half_field
        vertical_half = half_field / viewport_aspect
        equatorial = self.projection_mode.currentData() == "equatorial"
        projection = (
            self._project_equatorial if equatorial else
            (self._project_hammer if full_sky else self._project_azimuthal)
        )

        def visible_mask(
            x: np.ndarray, y: np.ndarray, distance: np.ndarray, margin: float = 1.0,
        ) -> np.ndarray:
            if full_sky:
                return np.ones_like(distance, dtype=bool)
            if equatorial:
                return (
                    (np.abs(x) <= horizontal_half * margin) &
                    (y >= center_dec - vertical_half * margin) &
                    (y <= center_dec + vertical_half * margin)
                )
            return (
                (np.abs(x) <= horizontal_half * margin) &
                (np.abs(y) <= vertical_half * margin) &
                (distance < 179.0)
            )

        star_x, star_y, star_distance = projection(
            BRIGHT_STARS[:, 0], BRIGHT_STARS[:, 1], center_ra, center_dec
        )
        star_visible = visible_mask(star_x, star_y, star_distance, 1.03)
        sizes = np.clip(1.0 + 2.2 * np.power(np.maximum(6.2 - BRIGHT_STARS[:, 2], .1), 1.55), .8, 52)
        ax.scatter(star_x[star_visible], star_y[star_visible], s=sizes[star_visible],
                   color="#d9e7ff", alpha=.76, linewidth=0)
        constellation_segments: list[np.ndarray] = []
        for line in CONSTELLATION_LINES:
            line_x, line_y, line_distance = projection(
                line[:, 0], line[:, 1], center_ra, center_dec
            )
            # Draw complete continuous strokes and let Matplotlib clip them at
            # the viewport. Filtering vertices first made a side disappear as
            # soon as one endpoint moved just outside the map (especially the
            # Great Square of Pegasus). Split only at a true projection seam.
            discontinuity = (
                ~np.isfinite(line_x[:-1]) | ~np.isfinite(line_x[1:]) |
                ~np.isfinite(line_y[:-1]) | ~np.isfinite(line_y[1:])
            )
            if equatorial:
                discontinuity |= np.abs(np.diff(line_x)) > 180.0
            elif full_sky:
                discontinuity |= np.hypot(np.diff(line_x), np.diff(line_y)) > 80.0
            else:
                discontinuity |= (
                    (line_distance[:-1] >= 179.0) |
                    (line_distance[1:] >= 179.0) |
                    (np.hypot(np.diff(line_x), np.diff(line_y)) > 120.0)
                )
            for segment in np.split(
                np.arange(len(line_x)), np.where(discontinuity)[0] + 1
            ):
                if len(segment) > 1:
                    constellation_segments.append(np.column_stack((
                        line_x[segment], line_y[segment],
                    )))
        if constellation_segments:
            ax.add_collection(LineCollection(
                constellation_segments, colors="#526b8d", alpha=.72,
                linewidths=.65, zorder=0, clip_on=True,
            ))
        # Text rendering is relatively expensive and prevents a useful frame
        # rate on large maps.  Hide only the faint constellation abbreviations
        # while the animation is moving; the following stop redraw restores
        # them at the final date.
        if CONSTELLATION_LABELS and not self.playback.timer.isActive():
            label_ra = np.array([item[1] for item in CONSTELLATION_LABELS])
            label_dec = np.array([item[2] for item in CONSTELLATION_LABELS])
            label_x, label_y, label_distance = projection(
                label_ra, label_dec, center_ra, center_dec
            )
            label_visible = visible_mask(label_x, label_y, label_distance, .96)
            for (abbreviation, _, _), x_value, y_value, is_visible in zip(
                CONSTELLATION_LABELS, label_x, label_y, label_visible
            ):
                if is_visible:
                    ax.text(x_value, y_value, abbreviation, color="#7890ad",
                            fontsize=7, alpha=.55, ha="center", va="center", zorder=.2)
        for _, reference_ra, reference_dec, color, style in SKY_REFERENCE_CURVES:
            curve_x, curve_y, curve_distance = projection(
                reference_ra, reference_dec, center_ra, center_dec
            )
            visible_indices = np.flatnonzero(
                visible_mask(curve_x, curve_y, curve_distance, 1.05)
            )
            if visible_indices.size:
                discontinuity = np.diff(visible_indices) > 1
                discontinuity |= np.hypot(
                    np.diff(curve_x[visible_indices]), np.diff(curve_y[visible_indices])
                ) > 40.0
                for segment in np.split(visible_indices, np.where(discontinuity)[0] + 1):
                    if len(segment) > 1:
                        ax.plot(curve_x[segment], curve_y[segment], color=color,
                                ls=style, alpha=.48, lw=.8, zorder=.5)
        track_x, track_y, track_distance = projection(
            geom["ra"], geom["dec"], center_ra, center_dec
        )
        if self.show_path.isChecked():
            visible_indices = np.flatnonzero(
                visible_mask(track_x, track_y, track_distance, 1.05)
            )
            if visible_indices.size:
                discontinuity = np.diff(visible_indices) > 1
                maximum_step = 80.0 if full_sky else max(2.0, selected_field * .08)
                discontinuity |= np.hypot(
                    np.diff(track_x[visible_indices]), np.diff(track_y[visible_indices])
                ) > maximum_step
                breaks = np.where(discontinuity)[0] + 1
                for segment in np.split(visible_indices, breaks):
                    if len(segment) > 1:
                        ax.plot(track_x[segment], track_y[segment], color="#54d3ff", lw=.9, alpha=.62)
            marker_indices = np.linspace(0, len(jd) - 1, 7).astype(int)
            if not full_sky:
                marker_indices = marker_indices[visible_mask(
                    track_x[marker_indices], track_y[marker_indices],
                    track_distance[marker_indices],
                )]
            ax.scatter(track_x[marker_indices], track_y[marker_indices], s=13, color="#79dcff", zorder=3)
            for index in marker_indices:
                label_time = datetime_from_jd(float(jd[index]))
                if label_time is not None:
                    date_format = "%m-%d" if horizon <= 300 else "%Y-%m"
                    ax.annotate(label_time.strftime(date_format), (track_x[index], track_y[index]),
                                xytext=(3, 4), textcoords="offset points", color="#8fb5cb", fontsize=7)
        now_x, now_y, _ = projection(
            float(now["ra"][0]), float(now["dec"][0]), center_ra, center_dec
        )
        # Cyan target marker is deliberately unlike the yellow Sun marker.
        ax.scatter(float(now_x), float(now_y), s=70, color="#32d7ff", edgecolor="white",
                   linewidth=.8, zorder=6)
        if self.show_labels.isChecked():
            ax.annotate(self.solution.designation, (float(now_x), float(now_y)), xytext=(9, -13),
                        textcoords="offset points", color="#67e4ff", fontsize=9,
                        fontweight="bold", va="top", zorder=7,
                        bbox={"facecolor": "#0e1828", "edgecolor": "none", "alpha": .72, "pad": 1.5})

        for name, body_ra, body_dec, color, size in self._body_positions(jd0):
            body_x, body_y, body_distance = projection(
                body_ra, body_dec, center_ra, center_dec
            )
            if bool(visible_mask(
                np.atleast_1d(body_x), np.atleast_1d(body_y),
                np.atleast_1d(body_distance),
            )[0]):
                ax.scatter(float(body_x), float(body_y), s=size, color=color,
                           edgecolor="white", linewidth=.35, zorder=4)
                if self.show_labels.isChecked():
                    ax.annotate(name, (float(body_x), float(body_y)), xytext=(4, 4),
                                textcoords="offset points", color=color, fontsize=8)
        theta = np.linspace(0.0, 2.0 * np.pi, 361)
        if equatorial and full_sky:
            ax.axhline(0, color="#23354d", lw=.45, alpha=.45)
            ax.axvline(0, color="#23354d", lw=.45, alpha=.45)
            ax.set_xlim(180.0, -180.0); ax.set_ylim(-90.0, 90.0)
            ax.set_aspect("auto")
        elif equatorial:
            ax.axhline(0, color="#23354d", lw=.45, alpha=.45)
            ax.axvline(0, color="#23354d", lw=.45, alpha=.45)
            ax.set_xlim(horizontal_half, -horizontal_half)
            ax.set_ylim(
                max(-90.0, center_dec - vertical_half),
                min(90.0, center_dec + vertical_half),
            )
            ax.set_aspect("auto")
        elif full_sky:
            ax.plot(180.0 * np.cos(theta), 90.0 * np.sin(theta), color="#526b8d", lw=.9, alpha=.8)
            ax.axhline(0, color="#23354d", lw=.45, alpha=.45)
            ax.axvline(0, color="#23354d", lw=.45, alpha=.45)
            ax.set_xlim(184.0, -184.0); ax.set_ylim(-94.0, 94.0)
            ax.set_aspect("auto")
        else:
            # Angular-distance circles make the globe-like geometry explicit.
            for radius in (vertical_half / 2.0, vertical_half, horizontal_half):
                ax.plot(radius * np.cos(theta), radius * np.sin(theta), color="#23354d", lw=.6, alpha=.58)
            ax.axhline(0, color="#23354d", lw=.45, alpha=.45)
            ax.axvline(0, color="#23354d", lw=.45, alpha=.45)
            ax.set_xlim(horizontal_half, -horizontal_half); ax.set_ylim(-vertical_half, vertical_half)
            ax.set_aspect("auto")
        if equatorial:
            tick_positions, tick_labels = equatorial_ra_ticks(center_ra, selected_field)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels(tick_labels)
            ax.set_xlabel(tr("RA — východ vlevo"))
            ax.set_ylabel("Dec [°]")
            projection_name = tr("Rovníkový pohled (RA/Dec)")
        else:
            ax.set_xlabel(tr("úhlová vzdálenost — východ vlevo [°]"))
            ax.set_ylabel(tr("úhlová vzdálenost k severu [°]"))
            projection_name = (
                tr("Hammerova projekce celé oblohy") if full_sky
                else tr("Středový azimutální pohled")
            )
        tracking_name = tr("objekt ve středu") if self.lock_target.isChecked() else tr(
            "pevné hvězdné pole",
        )
        ax.set_title(
            tr(
                "{0} • {1} • {2} • trajektorie {3}",
                self.solution.designation,
                projection_name,
                tracking_name,
                self.horizon.currentText(),
            )
        )
        self._set_info(time, now)
        self.canvas.draw_idle()


class PhotometryPanel(QWidget):
    photometry_changed = Signal()

    def __init__(self, solution: OrbitSolution) -> None:
        super().__init__()
        self.solution = solution
        self.observations: list[Observation] = []
        self.canvas = PlotCanvas()
        self.distance_axes = self.canvas.axes.twinx()
        controls = QHBoxLayout()
        self.kind = QComboBox(); self.kind.addItems([tr("Planetka"), tr("Kometa – celková jasnost")])
        self.absolute = QDoubleSpinBox(); self.absolute.setRange(-10, 40); self.absolute.setValue(solution.absolute_magnitude)
        self.absolute.setDecimals(2); self.absolute.setPrefix("H/M = ")
        comet = is_comet_designation(solution.designation) or is_comet_designation(solution.display_name or "")
        if comet and solution.photometric_exponent == 2.0:
            solution.photometric_exponent = 4.0
        if comet:
            solution.phase_correction = False
        self.kind.setCurrentIndex(1 if comet else 0)
        self.exponent = QDoubleSpinBox(); self.exponent.setRange(0, 20)
        self.exponent.setValue(solution.photometric_exponent); self.exponent.setSingleStep(.25)
        self.exponent.setPrefix("n = ")
        self.phase = QCheckBox(tr("H–G fázová korekce"))
        self.phase.setChecked(solution.phase_correction and not comet)
        self.phase.setEnabled(not comet)
        for widget in (self.kind, self.absolute, self.exponent, self.phase): controls.addWidget(widget)
        controls.addStretch()
        date_controls = QHBoxLayout()
        now = datetime.now()
        self.date_from = QDateTimeEdit(now - timedelta(days=180)); self.date_from.setCalendarPopup(True)
        self.date_to = QDateTimeEdit(now + timedelta(days=180)); self.date_to.setCalendarPopup(True)
        self.cursor_time = QDateTimeEdit(now); self.cursor_time.setCalendarPopup(True)
        for editor in (self.date_from, self.date_to, self.cursor_time):
            editor.setDisplayFormat("yyyy-MM-dd HH:mm")
        now_button = QPushButton(tr("Nyní")); now_button.clicked.connect(self._select_now)
        redraw_button = QPushButton(tr("Překreslit")); redraw_button.clicked.connect(self.redraw)
        for widget in (QLabel(tr("Od:")), self.date_from, QLabel(tr("Do:")), self.date_to,
                       QLabel(tr("Kurzor:")), self.cursor_time, now_button, redraw_button):
            date_controls.addWidget(widget)
        date_controls.addStretch()
        layout = QVBoxLayout(self); layout.addLayout(controls); layout.addLayout(date_controls)
        layout.addWidget(NavigationToolbar2QT(self.canvas, self)); layout.addWidget(self.canvas, 1)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.absolute.valueChanged.connect(self._absolute_changed)
        self.exponent.valueChanged.connect(self._photometry_setting_changed)
        self.phase.toggled.connect(self._photometry_setting_changed)
        self.cursor_time.dateTimeChanged.connect(lambda _: self.redraw())
        self.redraw()

    def _select_now(self) -> None:
        self.cursor_time.setDateTime(datetime.now())

    def set_solution(self, solution: OrbitSolution) -> None:
        self.solution = solution
        comet = is_comet_designation(solution.designation) or is_comet_designation(solution.display_name or "")
        if comet and solution.photometric_exponent == 2.0:
            solution.photometric_exponent = 4.0
        if comet:
            solution.phase_correction = False
        for control in (self.kind, self.exponent, self.phase):
            control.blockSignals(True)
        self.kind.setCurrentIndex(1 if comet else 0)
        self.exponent.setValue(solution.photometric_exponent)
        self.phase.setChecked(solution.phase_correction and not comet)
        self.phase.setEnabled(not comet)
        for control in (self.kind, self.exponent, self.phase):
            control.blockSignals(False)
        if not self._calibrate_absolute_from_observations():
            self.absolute.setValue(solution.absolute_magnitude)
        self.redraw()

    def set_observations(self, observations: list[Observation]) -> None:
        self.observations = observations
        self._calibrate_absolute_from_observations()

    def _photometry_setting_changed(self, *_args) -> None:
        if not self._calibrate_absolute_from_observations():
            self.solution.photometric_exponent = self.exponent.value()
            self.solution.phase_correction = (
                self.kind.currentIndex() == 0 and self.phase.isChecked()
            )
            self.photometry_changed.emit()
        self.redraw()

    def _absolute_changed(self, value: float) -> None:
        self.solution.absolute_magnitude = float(value)
        self.photometry_changed.emit()
        self.redraw()

    def _calibrate_absolute_from_observations(self) -> bool:
        measured: list[tuple[datetime, float]] = []
        for observation in self.observations:
            if observation.magnitude is None or not observation.time:
                continue
            try:
                measured.append((parse_observation_time(observation.time), observation.magnitude))
            except ValueError:
                continue
        if not measured:
            return False
        jd = np.array([datetime_to_jd(item[0]) for item in measured])
        geometry = apparent_geometry(self.solution, jd)
        use_phase = self.kind.currentIndex() == 0 and self.phase.isChecked()
        phase = (
            hg_phase_correction(geometry["phase"], self.solution.slope_g)
            if use_phase else None
        )
        estimate = estimate_absolute_magnitude(
            np.array([item[1] for item in measured]), geometry["delta"], geometry["r"],
            self.exponent.value(), phase,
        )
        self.solution.absolute_magnitude = estimate
        self.solution.photometric_exponent = self.exponent.value()
        self.solution.phase_correction = use_phase
        self.absolute.setValue(estimate)
        self.photometry_changed.emit()
        return True

    def _kind_changed(self, index: int) -> None:
        self.exponent.setValue(2.0 if index == 0 else 4.0)
        self.phase.setEnabled(index == 0)
        self.phase.setChecked(index == 0)
        if not self._calibrate_absolute_from_observations():
            self.solution.photometric_exponent = self.exponent.value()
            self.solution.phase_correction = index == 0 and self.phase.isChecked()
            self.photometry_changed.emit()
        self.redraw()

    def redraw(self) -> None:
        ax = self.canvas.axes
        distance_ax = self.distance_axes
        ax.clear(); distance_ax.clear(); self.canvas.style_axes()
        distance_ax.set_facecolor("none")
        distance_ax.tick_params(colors="#aebfd3")
        distance_ax.spines["right"].set_color("#405675")
        distance_ax.yaxis.label.set_color("#aebfd3")
        start = self.date_from.dateTime().toPython().replace(tzinfo=timezone.utc)
        stop = self.date_to.dateTime().toPython().replace(tzinfo=timezone.utc)
        cursor = self.cursor_time.dateTime().toPython().replace(tzinfo=timezone.utc)
        if stop <= start:
            return
        sample_count = max(200, min(1600, int((stop - start).total_seconds() / 86400) + 1))
        dates = [start + (stop - start) * fraction for fraction in np.linspace(0, 1, sample_count)]
        jd = np.array([datetime_to_jd(value) for value in dates])
        geom = apparent_geometry(self.solution, jd)
        magnitude = power_law_magnitude(self.absolute.value(), geom["delta"], geom["r"], self.exponent.value())
        if self.kind.currentIndex() == 0 and self.phase.isChecked():
            magnitude += hg_phase_correction(geom["phase"], self.solution.slope_g)
        magnitude_line, = ax.plot(
            dates, magnitude, color="#55c7ff", lw=2.2, label=tr("zdánlivá magnituda")
        )
        earth_line, = distance_ax.plot(
            dates, geom["delta"], color="#55e3a4", lw=1.35, ls="--",
            label=tr("Δ od Země"),
        )
        sun_line, = distance_ax.plot(
            dates, geom["r"], color="#ffb85c", lw=1.35, ls="-.",
            label=tr("r od Slunce"),
        )
        perihelion_time = datetime_from_jd(perihelion_jd_from_solution(self.solution))
        if perihelion_time is not None and start <= perihelion_time <= stop:
            ax.axvline(perihelion_time, color="#ff7e67", lw=1.15, alpha=.9, ls=":")
            ax.text(
                perihelion_time, .98,
                tr("přísluní\n{0:%Y-%m-%d %H:%M}", perihelion_time),
                transform=ax.get_xaxis_transform(), ha="center", va="top",
                color="#ff9b88", fontsize=8,
            )
        if start <= cursor <= stop:
            cursor_geom = apparent_geometry(self.solution, np.array([datetime_to_jd(cursor)]))
            cursor_mag = power_law_magnitude(
                self.absolute.value(), cursor_geom["delta"], cursor_geom["r"], self.exponent.value()
            )
            if self.kind.currentIndex() == 0 and self.phase.isChecked():
                cursor_mag += hg_phase_correction(cursor_geom["phase"], self.solution.slope_g)
            ax.axvline(cursor, color="#ffca58", lw=1.0, alpha=.8)
            ax.scatter([cursor], cursor_mag, s=48, color="#ffca58", edgecolor="white", zorder=5)
            ax.annotate(f"{cursor:%Y-%m-%d}\nmag {cursor_mag[0]:.2f}", (cursor, cursor_mag[0]),
                        xytext=(8, 8), textcoords="offset points", color="#ffdf8a", fontsize=9)
        ax.invert_yaxis(); ax.grid(color="#23354d", alpha=.55)
        ax.set_ylabel(tr("Zdánlivá magnituda")); ax.set_xlabel(tr("Datum"))
        distance_ax.set_ylabel(tr("Vzdálenost [AU]"))
        legend = ax.legend(
            [magnitude_line, earth_line, sun_line],
            [tr("zdánlivá magnituda"), tr("Δ od Země"), tr("r od Slunce")],
            loc="best", framealpha=.72,
        )
        legend.get_frame().set_facecolor("#111d30")
        legend.get_frame().set_edgecolor("#334760")
        for text_item in legend.get_texts():
            text_item.set_color("#d9e6f5")
        ax.set_title(tr(
            "Předpověď jasnosti — fotometrický exponent n = {0:.2f}",
            self.exponent.value(),
        ))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        self.canvas.figure.autofmt_xdate(rotation=25)
        self.canvas.draw_idle()


class EphemerisPanel(QWidget):
    def __init__(self, solution: OrbitSolution) -> None:
        super().__init__()
        self.solution = solution
        self._uncertainty_request_id = 0
        self._uncertainty_jobs: list[tuple[QThread, EphemerisUncertaintyWorker]] = []
        controls = QHBoxLayout()
        utc_now = datetime.now(timezone.utc).replace(tzinfo=None)
        self.start = QDateTimeEdit(utc_now); self.start.setCalendarPopup(True); self.start.setDisplayFormat("yyyy-MM-dd HH:mm")
        self.step = QComboBox()
        for label, minutes in ((tr("10 minut"), 10), (tr("30 minut"), 30), (tr("1 hodina"), 60), (tr(
            "1 den",
        ), 1440)):
            self.step.addItem(label, minutes)
        self.step.setCurrentIndex(self.step.findData(60))
        self.count = QSpinBox(); self.count.setRange(2, 2000); self.count.setValue(48); self.count.setPrefix(tr(
            "Řádků: ",
        ))
        self.station = QLineEdit("500"); self.station.setMaxLength(3); self.station.setMaximumWidth(62)
        self.station.setToolTip(tr("MPC kód observatoře; 500 znamená geocentrickou efemeridu."))
        self.generate_button = QPushButton(tr("Vygenerovat efemeridu"))
        self.generate_button.setObjectName("primary")
        self.generate_button.clicked.connect(self.generate)
        copy_selected = QPushButton(tr("Kopírovat vybrané"))
        copy_all = QPushButton(tr("Kopírovat vše"))
        copy_selected.clicked.connect(self.copy_selected)
        copy_all.clicked.connect(self.copy_all)
        for widget in (QLabel(tr("Od:")), self.start, QLabel(tr("Krok:")), self.step, self.count,
                       QLabel(tr("MPC stanice:")), self.station, self.generate_button,
                       copy_selected, copy_all):
            controls.addWidget(widget)
        controls.addStretch()
        self.uncertainty_status = QLabel("")
        self.uncertainty_status.setObjectName("muted")
        self.table = QTableWidget(0, 10)
        self.headers = (
            tr("Čas UTC"), tr("RA astrom. J2000 (h m s)"), tr("Dec astrom. J2000 (° ′ ″)"), tr("Azimut"), tr(
                "Výška",
            ),
            tr("Výška Slunce"), tr("Nejistota 1σ [″]"), "Δ [AU]", "r [AU]", "Mag",
        )
        self.table.setHorizontalHeaderLabels(self.headers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionMode(QTableWidget.ExtendedSelection)
        self.copy_shortcut = QShortcut(QKeySequence.Copy, self.table)
        self.copy_shortcut.activated.connect(self.copy_selected)
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.uncertainty_status)
        layout.addWidget(self.table, 1)
        self.generate()

    def set_solution(self, solution: OrbitSolution) -> None:
        self.solution = solution
        self.generate()

    def use_short_arc_preset(self) -> None:
        """Use the dense follow-up cadence appropriate for a discovery arc."""
        self.start.setDateTime(datetime.now(timezone.utc).replace(tzinfo=None))
        index = self.step.findData(10)
        if index >= 0:
            self.step.setCurrentIndex(index)
        # 48 hours at ten-minute spacing, including the starting row.
        self.count.setValue(289)

    @staticmethod
    def _coordinates_for_solution(
        solution: OrbitSolution, jd: np.ndarray, observer_offsets: np.ndarray | None,
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        return _ephemeris_coordinates(solution, jd, observer_offsets)

    def _positional_uncertainty(
        self, jd: np.ndarray, observer_offsets: np.ndarray | None,
        nominal_ra: np.ndarray, nominal_dec: np.ndarray,
    ) -> np.ndarray | None:
        return _ephemeris_positional_uncertainty(
            self.solution, jd, observer_offsets, nominal_dec,
        )

    def generate(self) -> None:
        self._uncertainty_request_id += 1
        request_id = self._uncertainty_request_id
        start = self.start.dateTime().toPython().replace(tzinfo=timezone.utc)
        minutes = int(self.step.currentData())
        count = self.count.value()
        dates = [start + timedelta(minutes=minutes * index) for index in range(count)]
        jd = np.array([datetime_to_jd(value) for value in dates])
        station_code = self.station.text().strip().upper() or "500"
        self.station.setText(station_code)
        if station_code != "500" and observatory_details(station_code) is None:
            QMessageBox.warning(self, tr("Efemerida"), tr(
                "MPC stanice {0} není v katalogu.",
                station_code,
            ))
            return
        observer_offsets = (
            None if station_code == "500" else station_offsets_gcrs([station_code] * count, dates)
        )
        geom, ra, dec, delta = self._coordinates_for_solution(
            self.solution, jd, observer_offsets
        )
        # After an impact the body has no position: those rows are dropped.
        valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(delta)
        impact_note = ""
        if not np.all(valid):
            impact = getattr(self.solution, "impact", None)
            impact_note = tr(
                "Těleso v tomto intervalu dopadlo ({0}); efemerida obsahuje jen okamžiky "
                "před dopadem.",
                (
                    (impact.entry or impact.surface).utc.strftime("%Y-%m-%d %H:%M:%S UTC")
                    if impact is not None and (impact.entry or impact.surface).utc
                    else tr("srážka s planetou")
                ),
            )
            keep = np.flatnonzero(valid)
            dates = [dates[index] for index in keep]
            jd = jd[keep]
            ra, dec, delta = ra[keep], dec[keep], delta[keep]
            geom = {
                key: (np.asarray(value)[keep] if np.ndim(value) and len(value) == len(valid)
                      else value)
                for key, value in geom.items()
            }
            if observer_offsets is not None:
                observer_offsets = observer_offsets[keep]
            count = len(dates)
        self._impact_note = impact_note
        if count == 0:
            self.table.setRowCount(0)
            self.uncertainty_status.setText(impact_note)
            return
        altaz = observation_altaz_series(station_code, dates, ra, dec)
        mag = power_law_magnitude(
            self.solution.absolute_magnitude, delta, geom["r"],
            self.solution.photometric_exponent,
        )
        if self.solution.phase_correction:
            mag += hg_phase_correction(geom["phase"], self.solution.slope_g)
        self.table.setRowCount(count)
        for row, instant in enumerate(dates):
            if altaz is None:
                azimuth, altitude, sun_altitude = "—", "—", "—"
            else:
                azimuth = f"{altaz[0][row]:.1f}°"
                altitude = f"{altaz[1][row]:+.1f}°"
                sun_altitude = f"{altaz[2][row]:+.1f}°"
            uncertainty_text = tr("počítám…") if self._has_positional_covariance() else "—"
            values = (
                instant.strftime("%Y-%m-%d %H:%M"), format_ra_sexagesimal(ra[row]),
                format_dec_sexagesimal(dec[row]), azimuth, altitude, sun_altitude,
                uncertainty_text, f"{delta[row]:.6f}", f"{geom['r'][row]:.6f}", f"{mag[row]:.2f}",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        if self._has_positional_covariance():
            self._start_uncertainty_job(request_id, jd, observer_offsets, dec)
            if impact_note:
                self.uncertainty_status.setText(
                    impact_note + "  " + self.uncertainty_status.text()
                )
        else:
            self.uncertainty_status.setText(impact_note)

    def _has_positional_covariance(self) -> bool:
        covariance = self.solution.state_covariance
        return (
            covariance is not None
            and self.solution.position_au is not None
            and self.solution.velocity_au_per_day is not None
            and np.asarray(covariance).shape == (6, 6)
            and bool(np.all(np.isfinite(np.asarray(covariance, dtype=float))))
        )

    def _start_uncertainty_job(
        self, request_id: int, jd: np.ndarray, observer_offsets: np.ndarray | None,
        nominal_dec: np.ndarray,
    ) -> None:
        self.uncertainty_status.setText(
            tr("Tabulka je hotová; polohová nejistota 1σ se počítá na pozadí…")
        )
        self.generate_button.setEnabled(False)
        thread = QThread(self)
        worker = EphemerisUncertaintyWorker(
            request_id, self.solution, jd, observer_offsets, nominal_dec,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.success.connect(self._uncertainty_ready)
        worker.error.connect(self._uncertainty_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(
            lambda active_thread=thread, active_worker=worker:
            self._uncertainty_job_finished(active_thread, active_worker)
        )
        self._uncertainty_jobs.append((thread, worker))
        thread.start()

    @Slot(int, object)
    def _uncertainty_ready(self, request_id: int, uncertainty: object) -> None:
        if request_id != self._uncertainty_request_id:
            return
        if uncertainty is None:
            self.uncertainty_status.setText(tr("Polohovou nejistotu 1σ se nepodařilo určit."))
            for row in range(self.table.rowCount()):
                self.table.setItem(row, 6, QTableWidgetItem("—"))
            return
        values = np.asarray(uncertainty, dtype=float)
        if len(values) != self.table.rowCount():
            return
        for row, value in enumerate(values):
            text = f"{value:.3f}" if value < 10.0 else f"{value:.2f}"
            self.table.setItem(row, 6, QTableWidgetItem(text))
        self.uncertainty_status.setText(
            (getattr(self, "_impact_note", "") + "  " if getattr(self, "_impact_note", "") else "")
            + tr("Polohová nejistota 1σ byla dopočítána v úhlových vteřinách.")
        )

    @Slot(int, str)
    def _uncertainty_failed(self, request_id: int, detail: str) -> None:
        if request_id != self._uncertainty_request_id:
            return
        self.uncertainty_status.setText(tr("Polohovou nejistotu 1σ se nepodařilo dopočítat."))
        self.uncertainty_status.setToolTip(detail)
        for row in range(self.table.rowCount()):
            self.table.setItem(row, 6, QTableWidgetItem("—"))

    def _uncertainty_job_finished(
        self, thread: QThread, worker: EphemerisUncertaintyWorker,
    ) -> None:
        try:
            self._uncertainty_jobs.remove((thread, worker))
        except ValueError:
            pass
        if not any(
            active_worker.request_id == self._uncertainty_request_id
            for _, active_worker in self._uncertainty_jobs
        ):
            self.generate_button.setEnabled(True)

    def _copy_rows(self, rows: list[int]) -> None:
        lines = ["\t".join(self.headers)]
        for row in rows:
            lines.append("\t".join(
                self.table.item(row, column).text() if self.table.item(row, column) else ""
                for column in range(self.table.columnCount())
            ))
        QApplication.clipboard().setText("\n".join(lines))

    def copy_selected(self) -> None:
        indexes = self.table.selectedIndexes()
        if not indexes:
            return
        selected = {(index.row(), index.column()) for index in indexes}
        rows = range(min(row for row, _ in selected), max(row for row, _ in selected) + 1)
        columns = range(min(column for _, column in selected), max(column for _, column in selected) + 1)
        lines = []
        for row in rows:
            lines.append("\t".join(
                self.table.item(row, column).text()
                if (row, column) in selected and self.table.item(row, column) else ""
                for column in columns
            ))
        QApplication.clipboard().setText("\n".join(lines))

    def copy_all(self) -> None:
        self._copy_rows(list(range(self.table.rowCount())))


class ElementsPanel(QWidget):
    def __init__(self, solution: OrbitSolution) -> None:
        super().__init__()
        self.solution = solution
        self._heliocentric_display_solution = solution
        self.display_solution = solution
        self.epoch_thread: QThread | None = None
        self.epoch_worker: EpochPropagationWorker | None = None
        self._epoch_target_time: datetime | None = None
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text.setStyleSheet("font-family: Consolas, 'Courier New', monospace; font-size: 14px;")
        copy_report = QPushButton(tr("Kopírovat přehled"))
        copy_mpc = QPushButton(tr("Kopírovat MPC 1-line elements"))
        copy_report.clicked.connect(lambda: QApplication.clipboard().setText(self._report()))
        copy_mpc.clicked.connect(lambda: QApplication.clipboard().setText(self._mpc_line()))
        buttons = QHBoxLayout(); buttons.addWidget(copy_report); buttons.addWidget(copy_mpc)
        buttons.addStretch()
        # QDateTimeEdit rejects an initial "1" while a year such as 1900 is
        # being typed because its internal calendar minimum is 1752.  A plain
        # validated text field also lets observers enter a year by itself.
        self.output_epoch = QLineEdit()
        self.output_epoch.setPlaceholderText(tr("YYYY nebo YYYY-MM-DD HH:MM"))
        self.output_epoch.setMinimumWidth(175)
        epoch_time = datetime_from_jd(solution.epoch_jd) or datetime.now()
        self.output_epoch.setText(epoch_time.strftime("%Y-%m-%d %H:%M"))
        self.apply_epoch_button = QPushButton(tr("Převést na epochu"))
        self.apply_epoch_button.clicked.connect(self._apply_output_epoch)
        reset_epoch = QPushButton(tr("Epocha fitu"))
        reset_epoch.clicked.connect(self._reset_output_epoch)
        epoch_controls = QHBoxLayout()
        epoch_controls.addWidget(QLabel(tr("Výstupní epocha:")))
        epoch_controls.addWidget(self.output_epoch)
        epoch_controls.addWidget(self.apply_epoch_button)
        epoch_controls.addWidget(reset_epoch)
        self.element_center = QComboBox()
        for key, label in ELEMENT_CENTERS:
            self.element_center.addItem(tr("Střed elementů: {0}", label), key)
        self.element_center.currentIndexChanged.connect(self._apply_element_center)
        epoch_controls.addWidget(self.element_center)
        epoch_controls.addStretch()
        self.epoch_note = QLabel(
            tr(
                "Převod používá N-body integraci s planetami a případnými A1–A3/DT. "
                "Aktivní planetární efemerida: {0}.",
                PLANETS.model_name,
            )
        )
        self.epoch_note.setObjectName("muted"); self.epoch_note.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addLayout(buttons); layout.addLayout(epoch_controls)
        # Prominent banner for a predicted collision (set after the fit).
        self.impact_banner = QLabel()
        self.impact_banner.setWordWrap(True)
        self.impact_banner.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.impact_banner.setStyleSheet(
            "background: #3a1115; border: 1px solid #ff6b5b; border-radius: 6px; "
            "color: #ffd7d2; padding: 8px 10px; font-size: 14px;"
        )
        self.impact_banner.setVisible(False)
        layout.addWidget(self.impact_banner)
        layout.addWidget(self.epoch_note); layout.addWidget(self.text, 1)
        self.redraw()

    @staticmethod
    def _packed_epoch(jd: float) -> str:
        value = datetime_from_jd(jd)
        if value is None:
            return "?????"
        century = {18: "I", 19: "J", 20: "K", 21: "L"}.get(value.year // 100, "?")
        digit = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        return f"{century}{value.year % 100:02d}{digit[value.month]}{digit[value.day]}"

    def _mpc_line(self) -> str:
        center_control = getattr(self, "element_center", None)
        if center_control is not None and str(center_control.currentData() or "sun") != "sun":
            return (
                tr(
                    "MPC 1-line elements jsou definovány pro heliocentrickou dráhu; pro "
                    "planetocentrické elementy nejsou exportovány.",
                )
            )
        s = self.display_solution
        comet_match = re.match(r"^\s*([ACDPX])/\s*(\d{4})\s+([A-Z])(\d+)", s.designation, re.I)
        numbered_comet = re.match(r"^\s*(\d+)([PD])(?:/|\b)", s.designation, re.I)
        if comet_match or numbered_comet:
            periodic_number = int(numbered_comet.group(1)) if numbered_comet else 0
            packed_provisional = ""
            if comet_match:
                orbit_type, year_text, half_month, sequence_text = comet_match.groups()
                year = int(year_text)
                century = {18: "I", 19: "J", 20: "K", 21: "L"}.get(year // 100, "?")
                packed_provisional = (
                    f"{century}{year % 100:02d}{half_month.upper()}"
                    f"{int(sequence_text):02d}0"
                )
            else:
                orbit_type = numbered_comet.group(2).upper()
            perihelion = self._perihelion_jd()
            peri_time = datetime_from_jd(perihelion)
            epoch = datetime_from_jd(s.epoch_jd)
            if peri_time is None or epoch is None:
                return (
                    tr(
                        "MPC 1-line elements nelze vytvořit: kalendářní datum epochy nebo "
                        "perihelia je mimo podporovaný rozsah.",
                    )
                )
            day_fraction = peri_time.day + (
                peri_time.hour * 3600 + peri_time.minute * 60 + peri_time.second + peri_time.microsecond / 1e6
            ) / 86400.0
            perihelion_distance = s.semi_major_axis_au * (1.0 - s.eccentricity)
            slope_parameter = 2.5 * s.photometric_exponent
            number_field = f"{periodic_number:4d}" if numbered_comet else "    "
            prefix = f"{number_field}{orbit_type.upper()}{packed_provisional:<7}"
            return (
                f"{prefix}  {peri_time.year:04d} {peri_time.month:02d} {day_fraction:07.4f} "
                f"{perihelion_distance:9.6f}  {s.eccentricity:8.6f}  "
                f"{s.arg_peri_deg:8.4f}  {s.node_deg:8.4f}  {s.inclination_deg:8.4f}  "
                f"{epoch:%Y%m%d}  {s.absolute_magnitude:4.1f} {slope_parameter:4.1f}  "
                f"{s.designation}"
            )
        mean_motion = np.degrees(0.01720209895 / abs(s.semi_major_axis_au) ** 1.5)
        packed_designation = s.designation.replace(" ", "")[:7]
        return (
            f"{packed_designation:<7} {s.absolute_magnitude:5.2f} {s.slope_g:5.2f} "
            f"{self._packed_epoch(s.epoch_jd):5s} {s.mean_anomaly_deg:9.5f} "
            f"{s.arg_peri_deg:9.5f} {s.node_deg:9.5f} {s.inclination_deg:9.5f} "
            f"{s.eccentricity:10.7f} {mean_motion:11.8f} {s.semi_major_axis_au:11.7f}"
        )

    def _perihelion_jd(self) -> float:
        center = str(self.element_center.currentData() or "sun")
        mu = MU_SUN if center == "sun" else PLANET_MU[center]
        return perihelion_jd_from_solution(self.display_solution, mu)

    def _uncertainty(self, name: str, digits: int) -> str:
        values = self.display_solution.element_uncertainties or {}
        value = values.get(name)
        return "" if value is None or not np.isfinite(value) else f" +/- {value:.{digits}f}"

    def _report(self) -> str:
        s = self.display_solution
        epoch = datetime_from_jd(s.epoch_jd)
        perihelion_jd = self._perihelion_jd()
        perihelion_time = datetime_from_jd(perihelion_jd)
        perihelion = s.semi_major_axis_au * (1.0 - s.eccentricity)
        aphelion = s.semi_major_axis_au * (1.0 + s.eccentricity) if s.eccentricity < 1.0 else None
        center = str(self.element_center.currentData() or "sun")
        mu = MU_SUN if center == "sun" else PLANET_MU[center]
        mean_motion_rad_day = np.sqrt(mu / abs(s.semi_major_axis_au) ** 3)
        period = (
            2.0 * np.pi / mean_motion_rad_day
            if s.semi_major_axis_au > 0 and s.eccentricity < 1 else None
        )
        mean_motion = np.degrees(mean_motion_rad_day)
        if perihelion_time is None:
            perihelion_line = (
                tr(
                    "Perihelion: kalendářní datum mimo podporovaný rozsah [JD {0:.8f}]",
                    perihelion_jd,
                )
            )
        else:
            peri_day = perihelion_time.day + (
                perihelion_time.hour * 3600 + perihelion_time.minute * 60
                + perihelion_time.second
            ) / 86400.0
            perihelion_line = (
                f"Perihelion {perihelion_time.year:04d} {perihelion_time:%b} {peri_day:08.5f}"
                f"{self._uncertainty('perihelion_jd', 5)} TT = {perihelion_time:%H:%M:%S}"
                f" [JD {perihelion_jd:.8f}]"
            )
        if epoch is None:
            epoch_line = tr(
                "Epoch: kalendářní datum mimo podporovaný rozsah [JDT {0:.8f}]",
                s.epoch_jd,
            )
        else:
            epoch_day = epoch.day + (
                epoch.hour * 3600 + epoch.minute * 60 + epoch.second
            ) / 86400.0
            epoch_line = (
                f"Epoch {epoch.year:04d} {epoch:%b} {epoch_day:08.5f} TT = JDT {s.epoch_jd:.8f}"
            )
        impact_lines: list[str] = []
        prediction = getattr(self.solution, "impact", None)
        if prediction is not None:
            label = BODY_LABELS.get(prediction.body, prediction.body)
            impact_lines = [
                "*** " + tr("PŘEDPOVĚĎ DOPADU: {0}", label).upper() + " ***",
                *describe_impact(prediction, label).split("\n")[1:],
                "",
            ]
        lines = [
            *impact_lines,
            f"Orbital elements: {s.designation}",
            f"Element center: {dict(ELEMENT_CENTERS).get(str(self.element_center.currentData() or 'sun'))}",
            perihelion_line,
            epoch_line,
            f"Planetary ephemeris: {PLANETS.model_name}",
            f"M    {s.mean_anomaly_deg:12.8f}{self._uncertainty('mean_anomaly', 8)}",
            f"n    {mean_motion:12.9f}{self._uncertainty('mean_motion_deg_day', 9)} deg/day",
            f"a    {s.semi_major_axis_au:12.9f}{self._uncertainty('a', 9)} AU",
            f"e    {s.eccentricity:12.9f}{self._uncertainty('e', 9)}",
            f"Peri. {s.arg_peri_deg:12.8f}{self._uncertainty('arg_peri', 8)} deg",
            f"Node  {s.node_deg:12.8f}{self._uncertainty('node', 8)} deg",
            f"Incl. {s.inclination_deg:12.8f}{self._uncertainty('i', 8)} deg",
            f"q    {perihelion:12.9f}{self._uncertainty('q', 9)} AU",
            f"Q    {aphelion:12.9f} AU" if aphelion is not None else "Q               — (open orbit)",
            f"P    {period:.5f} days = {period / 365.2569:.6f} years" if period else "P               —",
            f"H {s.absolute_magnitude:.2f}   G {s.slope_g:.2f}",
        ]
        if s.nongravitational_parameters:
            if "R_EXPONENT" in s.nongravitational_parameters:
                exponent = float(s.nongravitational_parameters["R_EXPONENT"])
                lines.append(
                    "Nongravitational parameters "
                    f"(inverse-power model g(r) = r^-{exponent:g}):"
                )
            else:
                lines.append("Nongravitational parameters (Marsden water-ice model):")
            split = s.nongravitational_parameters.get(NONGRAVITATIONAL_SPLIT_KEY)
            if split is not None:
                switch = datetime_from_jd(float(split))
                switch_text = f"{switch:%Y-%m-%d}" if switch else f"JD {float(split):.1f}"
                lines.append(
                    f"Historical set A1H-A3H applies before {switch_text} (TDB), A1-A3 after it."
                )
            for name in ("A1", "A2", "A3", "DT", *HISTORICAL_NONGRAVITATIONAL.values()):
                if name in s.nongravitational_parameters:
                    value = s.nongravitational_parameters[name]
                    uncertainty = (s.element_uncertainties or {}).get(name)
                    suffix = (
                        f" +/- {uncertainty:.3e}" if uncertainty is not None
                        and np.isfinite(uncertainty) else ""
                    )
                    unit = "days" if name == "DT" else "AU/day^2"
                    value_text = f"{value:+.6f}" if name == "DT" else f"{value:+.6e}"
                    lines.append(f"{name:<4} {value_text}{suffix} {unit}")
        if s.dynamical_1a:
            dynamics = s.dynamical_1a
            lines.extend(("", tr("Původní a budoucí barycentrická dráha:")))
            if "error" in dynamics:
                lines.append(tr("Výpočet se nezdařil: {0}", dynamics['error']))
            else:
                boundary = float(dynamics.get("boundary_au", 250.0))
                lines.append(tr("Hranice výpočtu: {0:g} AU od Slunce", boundary))
                sigma_value = dynamics.get("reciprocal_a_sigma")
                sigma = float(sigma_value) if sigma_value is not None else None
                for prefix, label in (("original", "original"), ("future", "future")):
                    reciprocal = float(dynamics[prefix])
                    uncertainty = f" +/- {sigma:.3e}" if sigma is not None else ""
                    lines.append(
                        f"1/a {label:8s} {reciprocal:+.8e}{uncertainty} AU^-1"
                    )
                    if reciprocal > 0.0:
                        semimajor = 1.0 / reciprocal
                        period_years = semimajor ** 1.5
                        lines.append(
                            tr(
                                "    a = {0:.1f} AU; přibližná perioda {1:.0f} let",
                                semimajor,
                                period_years,
                            )
                        )
                    interpretation = dynamics.get(f"{prefix}_interpretation")
                    if interpretation:
                        lines.append(f"    {interpretation}")
                lines.append(tr("Poznámka: {0}", dynamics.get('model', '')))
        if s.fit_observation_count is not None:
            normalized = (
                f"; normalized RMS {s.fit_normalized_rms:.3f} sigma"
                if s.fit_normalized_rms is not None else ""
            )
            count_text = str(s.fit_observation_count)
            if (
                s.fit_available_observation_count is not None
                and s.fit_available_observation_count > s.fit_observation_count
            ):
                count_text += (
                    f" representative of {s.fit_available_observation_count} available"
                )
            lines.append(
                f"{count_text} observations; RMS residual "
                f"{s.fit_rms_arcsec:.3f} arcsec{normalized}" if s.fit_rms_arcsec is not None
                else f"{count_text} observations"
            )
        linkage_summary = apparition_linkage_summary(s)
        if linkage_summary:
            lines.extend(("", tr("Navázání pozorovaných návratů:"), linkage_summary))
            lines.append(
                tr("Návrat   Pozorovací oblouk  Dostupná  Ve fitu  RMS [\"]  RMS [σ]  Stav")
            )
            for record in s.apparition_linkage or ():
                start = record.get("year_start")
                end = record.get("year_end")
                arc = str(start) if start == end else f"{start}–{end}"
                rms = record.get("rms_arcsec")
                normalized = record.get("normalized_rms")
                rms_text = "—" if rms is None else f"{float(rms):.2f}"
                normalized_text = "—" if normalized is None else f"{float(normalized):.2f}"
                status = str(record.get("status") or "")
                lines.append(
                    f"{str(record.get('return_year') or '?'):>6s}   {arc:<16s}  "
                    f"{int(record.get('available') or 0):8d}  "
                    f"{int(record.get('fitted') or 0):7d}  {rms_text:>7s}  "
                    f"{normalized_text:>7s}  {LINKAGE_STATUS_LABELS.get(status, status)}"
                )
                reason = str(record.get("reason") or "").strip()
                if reason:
                    lines.append(f"         ↳ {reason}")
        if s.fit_model_note:
            lines.append(tr("Poznámka k fitu: {0}", s.fit_model_note))
        if (
            s.apparition_elements
            and str(self.element_center.currentData() or "sun") == "sun"
        ):
            lines.extend((
                "", tr("Oskulační elementy jednotlivých návratů:"),
                tr(
                    "Perihelium (UTC)      q [AU]       e           a [AU]      P [roky]    i "
                    "[deg]    Node [deg]  Peri. [deg]",
                ),
            ))
            for item in s.apparition_elements:
                instant = datetime_from_jd(float(item["epoch_jd"]))
                date_text = (
                    instant.strftime("%Y-%m-%d %H:%M") if instant is not None
                    else f"JD {float(item['epoch_jd']):.3f}"
                )
                period_value = item.get("period_years")
                period_text = (
                    "       —" if period_value is None
                    else f"{float(period_value):8.3f}"
                )
                marker = tr("  budoucí") if bool(item.get("future")) else ""
                lines.append(
                    f"{date_text:19s}  {float(item['q']):10.6f}  "
                    f"{float(item['e']):10.7f}  {float(item['a']):11.6f}  "
                    f"{period_text}  {float(item['i']):9.5f}  "
                    f"{float(item['node']):10.5f}  {float(item['arg_peri']):11.5f}"
                    f"{marker}"
                )
            if not s.nongravitational_parameters and is_comet_designation(s.designation):
                lines.append(
                    tr(
                        "Pozor: návraty jsou propojeny čistě gravitačním modelem; bez A1/A2/A3 "
                        "mohou mít aktivní komety systematicky posunuté perihelové časy.",
                    )
                )
        if s.position_au and s.velocity_au_per_day:
            lines.extend((
                "", tr("Kartézský stav (AU, AU/den):"),
                "r = " + "  ".join(f"{value:+.12f}" for value in s.position_au),
                "v = " + "  ".join(f"{value:+.12f}" for value in s.velocity_au_per_day),
            ))
        lines.extend(("", "MPC 1-line elements:", self._mpc_line()))
        return "\n".join(lines)

    def redraw(self) -> None:
        self.text.setPlainText(self._report())

    def _reset_output_epoch(self) -> None:
        self._heliocentric_display_solution = self.solution
        self._apply_element_center()
        epoch_time = datetime_from_jd(self.solution.epoch_jd)
        if epoch_time is not None:
            self.output_epoch.setText(epoch_time.strftime("%Y-%m-%d %H:%M"))
        self.epoch_note.setText(
            tr(
                "Zobrazena je epocha vlastního fitu. Převod používá N-body integraci; "
                "aktivní planetární efemerida: {0}.",
                PLANETS.model_name,
            )
        )
        self.redraw()

    def _refresh_element_centers(self) -> None:
        """Offer planetocentric elements only where the body is bound.

        Osculating elements about a planet describe a real orbit only inside
        its Hill sphere with negative planetocentric energy (SL9, a captured
        minimoon); for a heliocentric body they would be formal numbers.
        """
        source = getattr(self, "_heliocentric_display_solution", None) or self.solution
        try:
            allowed = ("sun",) + bound_centers(source)
        except (ValueError, RuntimeError):
            allowed = ("sun",)
        present = tuple(
            self.element_center.itemData(index) for index in range(self.element_center.count())
        )
        if present == allowed:
            return
        current = str(self.element_center.currentData() or "sun")
        labels = dict(ELEMENT_CENTERS)
        self.element_center.blockSignals(True)
        self.element_center.clear()
        for key in allowed:
            self.element_center.addItem(tr("Střed elementů: {0}", labels.get(key, key)), key)
        index = self.element_center.findData(current)
        self.element_center.setCurrentIndex(index if index >= 0 else 0)
        self.element_center.blockSignals(False)
        self.element_center.setEnabled(len(allowed) > 1)
        self.element_center.setToolTip(
            "" if len(allowed) > 1 else tr(
                "Těleso obíhá Slunce; planetocentrické elementy se nabízejí jen při "
                "gravitačním zachycení planetou.",
            )
        )

    def _apply_element_center(self, *_args) -> None:
        self._refresh_element_centers()
        center = str(self.element_center.currentData() or "sun")
        try:
            self.display_solution = centered_solution(
                self._heliocentric_display_solution, center
            )
        except (ValueError, OrbitParseError) as exc:
            self.display_solution = self._heliocentric_display_solution
            self.epoch_note.setText(tr("Planetocentrické elementy nelze určit: {0}", exc))
        self.redraw()

    def _apply_output_epoch(self) -> None:
        source = self.solution
        if source.position_au is None or source.velocity_au_per_day is None:
            self.epoch_note.setText(tr(
                "Převod epochy vyžaduje kartézský stav načtené nebo vypočtené dráhy.",
            ))
            return
        value = self.output_epoch.text().strip()
        try:
            target_time = parse_output_epoch(value)
        except ValueError:
            self.epoch_note.setText(
                tr(
                    "Neplatná epocha. Použijte například 1900, 1900-01-01 nebo 1900-01-01 "
                    "00:00.",
                )
            )
            self.output_epoch.setFocus()
            return
        target_jd = datetime_to_jd(target_time)
        if self.epoch_thread is not None:
            return
        self._epoch_target_time = target_time
        self.apply_epoch_button.setEnabled(False)
        self.epoch_note.setText(
            tr(
                "Integruji dráhu na epochu {0:%Y-%m-%d %H:%M}… U vzdálené epochy to může "
                "trvat desítky sekund.",
                target_time,
            )
        )
        thread = QThread(self)
        worker = EpochPropagationWorker(source, target_jd)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.success.connect(self._epoch_ready)
        worker.error.connect(self._epoch_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._epoch_finished)
        self.epoch_thread = thread
        self.epoch_worker = worker
        thread.start()

    def _epoch_ready(self, solution: object) -> None:
        self._heliocentric_display_solution = solution
        self._apply_element_center()
        target_time = self._epoch_target_time or datetime.now()
        self.epoch_note.setText(
            tr(
                "Elementy převedeny N-body integrací na {0:%Y-%m-%d %H:%M}. Formální "
                "nejistoty nebyly do nové epochy propagovány. Planetární efemerida: {1}.",
                target_time,
                PLANETS.model_name,
            )
        )
        self.redraw()

    def _epoch_failed(self, message: str) -> None:
        self.epoch_note.setText(tr("Převod epochy se nezdařil: {0}", message))

    def _epoch_finished(self) -> None:
        self.epoch_thread = None
        self.epoch_worker = None
        self.apply_epoch_button.setEnabled(True)

    def set_solution(self, solution: OrbitSolution) -> None:
        self.solution = solution
        self._update_impact_banner()
        self._reset_output_epoch()

    def _update_impact_banner(self) -> None:
        prediction = getattr(self.solution, "impact", None)
        if prediction is None:
            self.impact_banner.setVisible(False)
            return
        label = BODY_LABELS.get(prediction.body, prediction.body)
        details = describe_impact(prediction, label).split("\n")
        self.impact_banner.setText(
            "<b style='font-size:16px; color:#ff8a7a'>"
            + html.escape(tr("⚠ PŘEDPOVĚĎ DOPADU: {0}", label).upper()) + "</b><br>"
            + "<br>".join(html.escape(line) for line in details[1:])
        )
        self.impact_banner.setVisible(True)


class De440DownloadDialog(QDialog):
    """Download and verify the official NAIF DE440 kernel."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Stažení JPL DE440"))
        self.setMinimumWidth(560)
        self.destination = de440_path()
        self.partial = self.destination.with_suffix(".bsp.part")
        self.reply: QNetworkReply | None = None
        self.output = None
        self.cancelled = False
        self.status = QLabel(
            tr("Stahuji oficiální kernel JPL DE440 z NASA/NAIF…\nCíl: {0}", self.destination)
        )
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.cancel_button = QPushButton(tr("Zrušit"))
        self.cancel_button.clicked.connect(self._cancel_download)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        layout.addWidget(self.progress)
        layout.addWidget(self.cancel_button, 0, Qt.AlignRight)
        self.manager = QNetworkAccessManager(self)
        QTimer.singleShot(0, self._start_download)

    def _start_download(self) -> None:
        try:
            self.destination.parent.mkdir(parents=True, exist_ok=True)
            self.output = self.partial.open("wb")
        except OSError as exc:
            QMessageBox.warning(self, "JPL DE440", tr("Nelze vytvořit soubor:\n{0}", exc))
            self.reject()
            return
        request = QNetworkRequest(QUrl(DE440_URL))
        request.setHeader(QNetworkRequest.UserAgentHeader, "Orbit-Solver/1.0")
        self.reply = self.manager.get(request)
        self.reply.readyRead.connect(self._write_available_data)
        self.reply.downloadProgress.connect(self._update_progress)
        self.reply.finished.connect(self._download_finished)
        self.reply.finished.connect(self.reply.deleteLater)

    def _write_available_data(self) -> None:
        if self.reply is not None and self.output is not None:
            self.output.write(bytes(self.reply.readAll()))

    def _update_progress(self, received: int, total: int) -> None:
        if total > 0:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(received * 1000 / total))
            self.status.setText(
                tr(
                    "Stahuji JPL DE440: {0:.1f} z {1:.1f} MB\nCíl: {2}",
                    received / 1048576,
                    total / 1048576,
                    self.destination,
                )
            )
        else:
            self.progress.setRange(0, 0)

    def _cancel_download(self) -> None:
        self.cancelled = True
        self.cancel_button.setEnabled(False)
        if self.reply is not None:
            self.reply.abort()
        else:
            self.reject()

    def _cleanup_partial(self) -> None:
        if self.output is not None:
            self.output.close()
            self.output = None
        try:
            if self.partial.exists():
                self.partial.unlink()
        except OSError:
            pass

    def _download_finished(self) -> None:
        self._write_available_data()
        if self.output is not None:
            self.output.close()
            self.output = None
        if self.cancelled:
            self._cleanup_partial()
            self.reject()
            return
        if self.reply is None or self.reply.error() != QNetworkReply.NoError:
            message = self.reply.errorString() if self.reply is not None else tr(
                "neznámá chyba",
            )
            self._cleanup_partial()
            QMessageBox.warning(self, "JPL DE440", tr("Stažení selhalo:\n{0}", message))
            self.reject()
            return
        self.status.setText(tr("Ověřuji kontrolní součet staženého souboru…"))
        QApplication.processEvents()
        digest = hashlib.md5()
        try:
            with self.partial.open("rb") as source:
                for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest().lower() != DE440_MD5:
                raise ValueError(
                    tr("Kontrolní součet nesouhlasí. Soubor může být neúplný nebo poškozený.")
                )
            # On Windows an mmap-backed SPK cannot be replaced while open.
            PLANETS.unload()
            self.partial.replace(self.destination)
            if not PLANETS.reload():
                raise ValueError(PLANETS.load_error or tr("Kernel se nepodařilo otevřít."))
        except (OSError, ValueError) as exc:
            self._cleanup_partial()
            PLANETS.reload()
            QMessageBox.warning(self, "JPL DE440", str(exc))
            self.reject()
            return
        self.progress.setRange(0, 1000)
        self.progress.setValue(1000)
        self.status.setText(
            tr(
                "JPL DE440 byla ověřena a aktivována. Nové výpočty budou používat "
                "planetární efemeridu DE440 v rozsahu 1549–2650.",
            )
        )
        self.cancel_button.setText(tr("Zavřít"))
        self.cancel_button.setEnabled(True)
        self.cancel_button.clicked.disconnect()
        self.cancel_button.clicked.connect(self.accept)

    def reject(self) -> None:
        if self.reply is not None and not self.reply.isFinished():
            self.cancelled = True
            self.reply.abort()
            return
        super().reject()


class OrbitConstraintsDialog(QDialog):
    LABELS = (
        ("a", tr("Velká poloosa a [AU]")), ("e", tr("Excentricita e")),
        ("i", tr("Sklon i [°]")), ("node", tr("Uzel Ω [°]")),
        ("arg_peri", tr("Perihelium ω [°]")), ("mean_anomaly", tr("Střední anomálie M [°]")),
    )

    def __init__(
        self, solution: OrbitSolution, constraints: dict[str, float],
        nongravitational: dict[str, float] | None = None,
        compute_original_future: bool = False,
        output_epoch: str = "",
        allow_element_constraints: bool = True, parent=None,
        linkage_options: dict[str, bool] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Vlastní nastavení fitu"))
        linkage_options = dict(linkage_options or {})
        self.controls: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        self.ng_controls: dict[str, tuple[QCheckBox, QDoubleSpinBox]] = {}
        values = {
            "a": solution.semi_major_axis_au, "e": solution.eccentricity,
            "i": solution.inclination_deg, "node": solution.node_deg,
            "arg_peri": solution.arg_peri_deg, "mean_anomaly": solution.mean_anomaly_deg,
        }
        form = QFormLayout()
        for name, label in self.LABELS:
            enabled = QCheckBox(tr("fixovat"))
            editor = QDoubleSpinBox(); editor.setDecimals(9); editor.setSingleStep(.01)
            if name == "a":
                editor.setRange(-1_000_000.0, 1_000_000.0)
            elif name == "e":
                editor.setRange(0.0, 10.0)
            else:
                editor.setRange(-360.0, 360.0)
            editor.setValue(constraints.get(name, values[name]))
            enabled.setChecked(name in constraints)
            enabled.setEnabled(allow_element_constraints)
            editor.setEnabled(allow_element_constraints)
            row = QWidget(); row_layout = QHBoxLayout(row); row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(enabled); row_layout.addWidget(editor, 1)
            form.addRow(label, row)
            self.controls[name] = (enabled, editor)
        self.output_epoch = QLineEdit(output_epoch)
        self.output_epoch.setPlaceholderText(tr("např. 2026 nebo 2026-10-18 00:00"))
        self.output_epoch.setToolTip(
            tr(
                "Fit zůstane kvůli numerické stabilitě v epoše uprostřed pozorování; "
                "výsledné oskulační elementy se po výpočtu převedou na tuto epochu.",
            )
        )
        form.addRow(tr("Výstupní epocha po fitu:"), self.output_epoch)
        ephemeris_heading = QLabel(tr("Planetární efemerida"))
        ephemeris_heading.setStyleSheet("font-weight: 650; margin-top: 10px;")
        form.addRow(ephemeris_heading)
        self.ephemeris_status = QLabel(PLANETS.status_text())
        self.ephemeris_status.setWordWrap(True)
        self.ephemeris_status.setObjectName("muted")
        form.addRow(self.ephemeris_status)
        self.download_de440 = QPushButton(
            tr("Přeinstalovat JPL DE440…") if PLANETS.jpl is not None
            else tr("Stáhnout JPL DE440 (cca 114 MB)…")
        )
        self.download_de440.setToolTip(
            tr("Oficiální planetární kernel NASA/NAIF; pokrytí 1549-12-31 až 2650-01-25.")
        )
        self.download_de440.clicked.connect(self._download_de440)
        form.addRow(self.download_de440)
        comet = (
            is_comet_designation(solution.designation)
            or is_comet_designation(solution.display_name or "")
        )
        self.is_comet = comet
        self.original_future = QCheckBox(
            tr("Po fitu spočítat původní a budoucí barycentrickou dráhu v 250 AU")
        )
        self.original_future.setChecked(comet and compute_original_future)
        self.original_future.setVisible(comet)
        separator = QLabel(tr("Negravitační zrychlení"))
        separator.setStyleSheet("font-weight: 650; margin-top: 10px;")
        form.addRow(separator)
        current_ng = dict(solution.nongravitational_parameters or {})
        current_ng.update(nongravitational or {})
        self.ng_model = QComboBox()
        self.ng_model.addItem(tr("Marsdenův vodní model (komety)"), "marsden")
        self.ng_model.addItem(tr("Mocninný model r⁻ᵐ (planetky / Yarkovsky)"), "power")
        default_model = "power" if "R_EXPONENT" in current_ng or not comet else "marsden"
        self.ng_model.setCurrentIndex(self.ng_model.findData(default_model))
        form.addRow(tr("Závislost na vzdálenosti:"), self.ng_model)
        self.ng_exponent = QDoubleSpinBox()
        self.ng_exponent.setDecimals(3)
        self.ng_exponent.setRange(.1, 10.0)
        self.ng_exponent.setSingleStep(.1)
        self.ng_exponent.setValue(float(current_ng.get("R_EXPONENT", 2.0)))
        self.ng_exponent.setToolTip(
            tr(
                "g(r) = (1 AU / r)^m. Hodnota m = 2 je jednoduchý standardní model pro "
                "tlak záření nebo příčnou Yarkovského složku A2.",
            )
        )
        form.addRow("Exponent m:", self.ng_exponent)
        for name, label in (
            ("A1", tr("A1 radiální [10⁻⁸ AU/den²]")),
            ("A2", tr("A2 příčný [10⁻⁸ AU/den²]")),
            ("A3", tr("A3 normálový [10⁻⁸ AU/den²]")),
            ("DT", tr("DT zpoždění aktivity [dny]")),
        ):
            enabled = QCheckBox(tr("fitovat"))
            editor = QDoubleSpinBox()
            editor.setDecimals(6)
            editor.setRange(
                -1_000.0 if name == "DT" else -10_000.0,
                1_000.0 if name == "DT" else 10_000.0,
            )
            editor.setSingleStep(.5 if name == "DT" else .1)
            scale = 1.0 if name == "DT" else 1e-8
            editor.setValue(float(current_ng.get(name, 0.0)) / scale)
            enabled.setChecked(name in (nongravitational or {}))
            row = QWidget(); row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(enabled); row_layout.addWidget(editor, 1)
            form.addRow(label, row)
            self.ng_controls[name] = (enabled, editor)
        dt_enabled, dt_editor = self.ng_controls["DT"]
        a_checks = [self.ng_controls[name][0] for name in ("A1", "A2", "A3")]

        def sync_ng_model() -> None:
            power_model = self.ng_model.currentData() == "power"
            self.ng_exponent.setEnabled(power_model)
            available_dt = not power_model and any(
                control.isChecked() for control in a_checks
            )
            dt_enabled.setEnabled(available_dt)
            dt_editor.setEnabled(available_dt)
            if not available_dt:
                dt_enabled.setChecked(False)

        linkage_heading = QLabel(tr("Navazování návratů"))
        linkage_heading.setStyleSheet("font-weight: 650; margin-top: 10px;")
        self.historical_force = QCheckBox(
            tr("Zvlášť negravitační parametry pro historické návraty (A1H, A2H, A3H)")
        )
        self.historical_force.setToolTip(
            tr(
                "Před nejdelší mezerou v pozorováních platí vlastní sada A1H–A3H, po ní "
                "A1–A3. Hodí se pro komety, jejichž aktivita se mezi historickým a "
                "moderním návratem změnila (205P: ztracena 1896–2008). Historická sada "
                "se fituje až s připojením historického návratu.",
            )
        )
        self.historical_force.setChecked(
            bool(linkage_options.get("historical_nongravitational"))
        )
        self.model_tension = QCheckBox(
            tr(
                "Připojit návrat i nad limitem kvality (napětí modelu, do {0:g}σ)",
                LINKAGE_RELAXED_NORMALIZED_RMS,
            )
        )
        self.model_tension.setToolTip(
            tr(
                "Běžně se návrat připojí, jen když sedí na vlastní astrometrii do 3σ. "
                "S touto volbou se přijme i horší shoda, pokud ostatní návraty zůstanou "
                "v pořádku – například u vizuální nebo fotografické historické "
                "astrometrie. Takový návrat je ve výsledku vždy označen jako napětí "
                "modelu.",
            )
        )
        self.model_tension.setChecked(bool(linkage_options.get("accept_model_tension")))

        def sync_historical() -> None:
            available = any(control.isChecked() for control in a_checks)
            self.historical_force.setEnabled(available)
            if not available:
                self.historical_force.setChecked(False)

        for control in a_checks:
            control.toggled.connect(sync_ng_model)
            control.toggled.connect(sync_historical)
        self.ng_model.currentIndexChanged.connect(sync_ng_model)
        sync_ng_model()
        sync_historical()
        nongrav_note = QLabel(
            tr(
                "A1–A3 lze fitovat bez ohledu na označení tělesa. Pro planetky je obvyklá "
                "příčná A2 (Yarkovsky); radiální A1 může popsat tlak záření. Marsdenův "
                "model a DT jsou určeny pro kometární odplyňování. Detekce Yarkovského "
                "efektu vyžaduje dlouhý přesný oblouk; samotná nenulová hodnota A2 na "
                "krátkém oblouku není průkazná.",
            )
        )
        nongrav_note.setWordWrap(True)
        nongrav_note.setObjectName("muted")
        form.addRow(nongrav_note)
        form.addRow(linkage_heading)
        form.addRow(self.historical_force)
        form.addRow(self.model_tension)
        if comet:
            dynamics_heading = QLabel(tr("Dlouhodobá dynamika komety"))
            dynamics_heading.setStyleSheet("font-weight: 650; margin-top: 10px;")
            form.addRow(dynamics_heading)
            form.addRow(self.original_future)
            dynamics_note = QLabel(
                tr(
                    "Volba je standardně vypnutá. Výpočet může trvat podstatně déle; proběhne "
                    "na pozadí. Aktivní model planet: {0}.",
                    PLANETS.model_name,
                )
            )
            dynamics_note.setWordWrap(True)
            dynamics_note.setObjectName("muted")
            form.addRow(dynamics_note)
            comparison_note = QLabel(
                tr(
                    "Pro přímé porovnání s publikovanou dráhou zapněte stejnou sadu "
                    "parametrů. Například řešení JPL s A1+A2 nelze přesně porovnávat s fitem "
                    "A1+A2+A3+DT; volné A3 a DT mohou kvůli korelacím posunout i A1 a A2.",
                )
            )
            comparison_note.setWordWrap(True)
            comparison_note.setObjectName("muted")
            form.addRow(comparison_note)
        note = QLabel(
            (
                tr(
                    "Fixace elementů je vynucena během diferenciální korekce. Pro "
                    "parabolickou hypotézu lze nastavit e = 1. ",
                )
                if allow_element_constraints else
                tr(
                    "MPC neposkytlo počáteční dráhu, proto bude nejprve určena z astrometrie; "
                    "fixace elementů se zpřístupní po prvním úspěšném fitu. ",
                )
            )
            + tr(
                "A1–A3 a volitelný časový posun DT se fitují jako volné parametry. DT má "
                "smysl především u více návratů; kladná hodnota znamená maximum aktivity "
                "po perihelu.",
            )
        )
        note.setWordWrap(True); note.setObjectName("muted")
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept); buttons.rejected.connect(self.reject)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.addWidget(note)
        content_layout.addLayout(form)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setWidget(content)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll, 1)
        # Keep the confirmation controls outside the scrolling region so they
        # remain reachable even on a small laptop display.
        layout.addWidget(buttons)
        screen = QApplication.screenAt(parent.mapToGlobal(parent.rect().center())) if parent else None
        if screen is None:
            screen = QApplication.primaryScreen()
        available_height = screen.availableGeometry().height() if screen else 900
        self.resize(720, min(820, max(500, int(available_height * .82))))

    def _download_de440(self) -> None:
        if PLANETS.jpl is None:
            prompt = (
                tr(
                    "Stáhnout oficiální JPL DE440 z NASA/NAIF?\n\nSoubor má přibližně 114 MB a "
                    "bude uložen do složky orbit-data. Pokrytí efemeridy je 1549–2650.",
                )
            )
        else:
            prompt = tr("Stáhnout a znovu ověřit JPL DE440 (přibližně 114 MB)?")
        if QMessageBox.question(
            self, "JPL DE440", prompt,
            QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes,
        ) != QMessageBox.Yes:
            return
        dialog = De440DownloadDialog(self)
        dialog.exec()
        self.ephemeris_status.setText(PLANETS.status_text())
        self.download_de440.setText(
            tr("Přeinstalovat JPL DE440…") if PLANETS.jpl is not None
            else tr("Stáhnout JPL DE440 (cca 114 MB)…")
        )

    def selected_constraints(self) -> dict[str, float]:
        return {name: editor.value() for name, (enabled, editor) in self.controls.items() if enabled.isChecked()}

    def selected_nongravitational(self) -> dict[str, float]:
        selected = {
            name: editor.value() * (1.0 if name == "DT" else 1e-8)
            for name, (enabled, editor) in self.ng_controls.items()
            if enabled.isChecked()
        }
        if "DT" in selected and not any(name in selected for name in ("A1", "A2", "A3")):
            selected.pop("DT")
        if any(name in selected for name in ("A1", "A2", "A3")):
            if self.ng_model.currentData() == "power":
                selected.pop("DT", None)
                selected["R_EXPONENT"] = self.ng_exponent.value()
        return selected

    def calculate_original_future(self) -> bool:
        return self.is_comet and self.original_future.isChecked()

    def selected_linkage_options(self) -> dict[str, bool]:
        return {
            "historical_nongravitational": bool(
                self.historical_force.isEnabled() and self.historical_force.isChecked()
            ),
            "accept_model_tension": self.model_tension.isChecked(),
        }

    def selected_output_epoch(self) -> str:
        return self.output_epoch.text().strip()


class PreliminaryOrbitDialog(QDialog):
    """Settings for a deliberately provisional short-arc orbit."""

    def __init__(self, observation_count: int, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Předběžná dráha z krátkého oblouku"))
        self.setMinimumWidth(560)
        self.method = QComboBox()
        self.method.addItem(tr("Systematické hledání vzdálenosti (doporučeno)"), "systematic")
        self.method.addItem(tr("Väisälä — předpoklad průchodu perihelem"), "vaisala")
        self.perihelion = QDoubleSpinBox()
        self.perihelion.setRange(0.01, 100.0)
        self.perihelion.setDecimals(4)
        self.perihelion.setSingleStep(0.1)
        self.perihelion.setValue(1.0)
        self.perihelion.setSuffix(" AU")
        self.perihelion.setEnabled(False)
        self.method.currentIndexChanged.connect(
            lambda _: self.perihelion.setEnabled(self.method.currentData() == "vaisala")
        )
        description = QLabel(
            tr(
                "Načteno {0} měření. Systematická metoda prohledá vzdálenost a radiální "
                "rychlost. Väisälova varianta vytvoří jednu hypotézu pro zadané q.",
                observation_count,
            )
        )
        description.setWordWrap(True)
        warning = QLabel(
            tr(
                "Výsledek z krátkého oblouku není jednoznačně určená dráha. Použijte jej "
                "jako počáteční odhad a počítejte s rychle rostoucí nejistotou efemeridy.",
            )
        )
        warning.setWordWrap(True)
        warning.setStyleSheet("color: #ffcf70;")
        form = QFormLayout()
        form.addRow(tr("Metoda:"), self.method)
        form.addRow(tr("Perihelová vzdálenost q:"), self.perihelion)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText(tr("Vypočítat předběžnou dráhu"))
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout = QVBoxLayout(self)
        layout.addWidget(description)
        layout.addLayout(form)
        layout.addWidget(warning)
        layout.addWidget(buttons)

    def selected_method(self) -> str:
        return str(self.method.currentData())

    def selected_perihelion_distance(self) -> float | None:
        return self.perihelion.value() if self.selected_method() == "vaisala" else None


class PreliminaryOrbitResultDialog(QDialog):
    USE = 1
    REFINE = 2

    def __init__(
        self, solution: OrbitSolution, method: str, assumed_q: float | None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Předběžná dráha — výsledek"))
        self.setMinimumWidth(600)
        method_text = (
            tr("systematické hledání vzdálenosti a radiální rychlosti")
            if method == "systematic" else tr(
                "Väisälova hypotéza, zadané q = {0:.4f} AU",
                assumed_q,
            )
        )
        actual_q = solution.semi_major_axis_au * (1.0 - solution.eccentricity)
        report = QPlainTextEdit()
        report.setReadOnly(True)
        report.setMinimumHeight(220)
        report.setPlainText(
            tr(
                "PŘEDBĚŽNÉ ŘEŠENÍ — DRÁHA NENÍ JEDNOZNAČNĚ URČENA\n\nMetoda: {0}\nEpocha: JD "
                "{1:.6f}\na = {2:.8g} AU\ne = {3:.8f}\nq = {4:.8f} AU\ni = {5:.6f}°\nΩ = "
                "{6:.6f}°\nω = {7:.6f}°\nM = {8:.6f}°\n\nVolba „Použít a zpřesnit“ spustí "
                "běžnou diferenciální korekci ze všech vybraných měření. Efemerida bude "
                "přednastavena po 10 minutách na 48 hodin.",
                method_text,
                solution.epoch_jd,
                solution.semi_major_axis_au,
                solution.eccentricity,
                actual_q,
                solution.inclination_deg,
                solution.node_deg,
                solution.arg_peri_deg,
                solution.mean_anomaly_deg,
            )
        )
        use = QPushButton(tr("Použít předběžnou dráhu"))
        refine = QPushButton(tr("Použít a zpřesnit"))
        refine.setObjectName("solveOrbit")
        close = QPushButton(tr("Zavřít"))
        use.clicked.connect(lambda: self.done(self.USE))
        refine.clicked.connect(lambda: self.done(self.REFINE))
        close.clicked.connect(self.reject)
        actions = QHBoxLayout()
        actions.addStretch()
        actions.addWidget(use)
        actions.addWidget(refine)
        actions.addWidget(close)
        layout = QVBoxLayout(self)
        layout.addWidget(report)
        layout.addLayout(actions)


def metric_card(title: str, value: str) -> tuple[QFrame, QLabel]:
    frame = QFrame(); frame.setObjectName("card")
    layout = QVBoxLayout(frame)
    caption = QLabel(title); caption.setObjectName("muted")
    number = QLabel(value); number.setObjectName("value")
    layout.addWidget(caption); layout.addWidget(number)
    return frame, number


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(tr("Orbit Solver — určování drah"))
        self.resize(1500, 920)
        self.solution = demo_solution()
        self.mpc_network = MpcNetworkClient(self)
        self.mpc_network.success.connect(
            lambda result: self._observations_loaded(*result)
        )
        self.mpc_network.error.connect(self._observations_failed)
        self.mpc_network.orbit_success.connect(self._orbit_loaded)
        self.mpc_network.orbit_error.connect(self._orbit_failed)
        self.mpc_network.identifier_success.connect(self._identifier_loaded)
        self.mpc_network.identifier_error.connect(self._identifier_failed)
        self.observation_model = ObservationModel()
        self.current_designation = ""
        self.current_display_name = ""
        self.current_observations: list[Observation] = []
        self.nongravitational_fit: dict[str, float] = {}
        self.linkage_options: dict[str, bool] = {}
        self.fit_constraints: dict[str, float] = {}
        self.compute_original_future = False
        self.fit_output_epoch = ""
        self.has_orbit = False
        self._auto_orbit_request = False
        self._fit_after_orbit_load = False
        self._fit_after_observations_load = False
        self.fit_thread: QThread | None = None
        self.fit_worker: OrbitFitWorker | None = None
        self.fit_progress_dialog: OrbitFitProgressDialog | None = None
        self.preliminary_thread: QThread | None = None
        self.preliminary_worker: PreliminaryOrbitWorker | None = None
        self._dirty_solution_tabs: set[int] = set()
        self._build_ui()

    def _build_ui(self) -> None:
        splitter = QSplitter()
        splitter.addWidget(self._sidebar())
        splitter.addWidget(self._workspace())
        splitter.setSizes([300, 1200]); splitter.setCollapsible(0, False)
        self.setCentralWidget(splitter)
        status = QStatusBar(); status.showMessage(tr(
            "Připraveno — zadejte objekt; jeho výchozí dráha se načte z MPC.",
        ))
        self.setStatusBar(status)

    def _sidebar(self) -> QWidget:
        panel = QFrame(); panel.setObjectName("sidebar"); panel.setMinimumWidth(270); panel.setMaximumWidth(370)
        layout = QVBoxLayout(panel); layout.setContentsMargins(20, 24, 20, 20); layout.setSpacing(12)
        title = QLabel("ORBIT SOLVER"); title.setObjectName("title")
        subtitle = QLabel(tr("Výpočet drah z astrometrie MPC")); subtitle.setObjectName("muted")
        self.search = QLineEdit(); self.search.setPlaceholderText("Apophis, 2026 AB, P21…")
        self.search.returnPressed.connect(self.load_object)
        self.load_button = QPushButton(tr("Načíst objekt"))
        self.load_button.setToolTip(
            tr(
                "Načte katalogovou dráhu; u NEOCP/PCCP stáhne astrometrii a spočítá "
                "vlastní dráhu.",
            )
        )
        self.load_button.setObjectName("primary"); self.load_button.clicked.connect(self.load_object)
        self.solve = QPushButton(tr("Znovu načíst výchozí dráhu")); self.solve.setEnabled(False)
        self.solve.setToolTip(
            tr(
                "Načte stav z MPC; u historických fragmentů Shoemaker–Levy 9 použije JPL "
                "Horizons.",
            )
        )
        self.solve.clicked.connect(self.load_orbit)
        self.mpc_observations_button = QPushButton(tr("Načíst astrometrii z MPC"))
        self.mpc_observations_button.setToolTip(
            tr(
                "Použije astrometrii uloženou v místní cache, pokud už byla dříve "
                "stažena; jinak ji stáhne z MPC. Program sám vyzkouší běžný archiv MPC "
                "i aktuální NEOCP/PCCP.",
            )
        )
        self.mpc_observations_button.clicked.connect(lambda: self.load_observations(False))
        self.mpc_refresh_button = QPushButton(tr("Aktualizovat astrometrii z MPC"))
        self.mpc_refresh_button.setToolTip(
            tr(
                "Znovu stáhne kompletní astrometrii z MPC (včetně nových pozorování) "
                "a přepíše místní cache.",
            )
        )
        self.mpc_refresh_button.clicked.connect(lambda: self.load_observations(True))
        self.file_observations_button = QPushButton(tr("Načíst pozorování z TXT…"))
        self.file_observations_button.clicked.connect(self.load_observations_file)
        self.paste_observations_button = QPushButton(tr("Vložit MPC pozorování…"))
        self.paste_observations_button.clicked.connect(self.paste_observations)
        self.fit_button = QPushButton(tr("▶  Spočítat vlastní dráhu")); self.fit_button.setEnabled(False)
        self.fit_button.setObjectName("solveOrbit")
        self.fit_button.setVisible(False)
        self.fit_button.clicked.connect(self.calculate_own_orbit)
        self.preliminary_button = QPushButton(tr("Předběžná dráha z krátkého oblouku…"))
        self.preliminary_button.setToolTip(
            tr(
                "Prohledá možné vzdálenosti a radiální rychlosti, případně vytvoří "
                "Väisälovu hypotézu se zadanou perihelovou vzdáleností.",
            )
        )
        self.preliminary_button.setEnabled(False)
        self.preliminary_button.setVisible(False)
        self.preliminary_button.clicked.connect(self.calculate_preliminary_orbit)
        self.constraints_button = QPushButton(tr("Vlastní nastavení…"))
        self.constraints_button.setEnabled(False)
        self.constraints_button.clicked.connect(self.configure_constraints)
        self.fit_interval = QComboBox()
        self.fit_interval.addItem(tr("Celý pozorovací oblouk"), None)
        self.fit_interval.addItem(tr("Poslední 2 roky"), 2.0)
        self.fit_interval.addItem(tr("Posledních 5 let"), 5.0)
        self.fit_interval.addItem(tr("Posledních 10 let"), 10.0)
        self.fit_interval.setToolTip(
            tr(
                "Časový výřez se počítá zpětně od nejnovějšího pozorování; data v tabulce "
                "zůstanou zachována.",
            )
        )
        self.fit_interval.currentIndexChanged.connect(
            lambda _: self._update_fit_button_label()
        )
        # Split comets: the components are separate bodies and a joint fit of
        # their astrometry is meaningless.  Shown only when MPC labels fragments.
        self.fragment_label = QLabel(tr("Fragment komety:"))
        self.fragment_selector = QComboBox()
        self.fragment_selector.setToolTip(
            tr(
                "Kometa se rozpadla; fituje se jen vybraná složka. Neoznačená měření "
                "(před rozpadem nebo bez rozlišení fragmentů) patří hlavní složce.",
            )
        )
        self.fragment_selector.currentIndexChanged.connect(self._fragment_selection_changed)
        self.fragment_label.setVisible(False)
        self.fragment_selector.setVisible(False)
        hint = QLabel(
            tr(
                "MPC dráha je počáteční odhad; vlastní fit používá topocentrické polohy "
                "stanic a automatické planetární perturbace.",
            )
        )
        hint.setWordWrap(True); hint.setObjectName("muted")
        layout.addWidget(title); layout.addWidget(subtitle); layout.addSpacing(18)
        layout.addWidget(QLabel(tr("Objekt"))); layout.addWidget(self.search)
        layout.addWidget(self.load_button); layout.addWidget(self.solve)
        layout.addSpacing(8); layout.addWidget(QLabel(tr("Astrometrie pro vlastní fit")))
        layout.addWidget(self.mpc_observations_button)
        layout.addWidget(self.mpc_refresh_button)
        layout.addWidget(self.file_observations_button)
        layout.addWidget(self.paste_observations_button)
        layout.addSpacing(7)
        layout.addWidget(self.fit_button)
        layout.addWidget(self.preliminary_button)
        layout.addWidget(self.constraints_button); layout.addWidget(QLabel(tr(
            "Rozsah astrometrie:",
        )))
        layout.addWidget(self.fit_interval)
        layout.addWidget(self.fragment_label); layout.addWidget(self.fragment_selector)
        layout.addWidget(hint)
        layout.addStretch()
        language_row = QHBoxLayout()
        language_row.addWidget(QLabel("Language / Jazyk:"))
        self.language_box = QComboBox()
        for code, name in SUPPORTED_LANGUAGES.items():
            self.language_box.addItem(name, code)
        self.language_box.setCurrentIndex(self.language_box.findData(language()))
        self.language_box.currentIndexChanged.connect(self._language_changed)
        language_row.addWidget(self.language_box, 1)
        layout.addLayout(language_row)
        version = QLabel(
            tr("Orbit Solver 1.0\nAutor: Martin Mašek\nVyvíjeno s pomocí OpenAI Codex")
        )
        version.setObjectName("muted")
        version.setWordWrap(True)
        layout.addWidget(version)
        return panel

    def _language_changed(self, *_args) -> None:
        """Store the chosen language and offer an immediate restart.

        Labels are built when the window is created, so a switch needs a new
        process; the choice is remembered for every later start.
        """
        code = str(self.language_box.currentData() or "en")
        if code == language():
            return
        try:
            save_language(code)
        except Exception as exc:  # settings backend unavailable
            QMessageBox.warning(self, "Language / Jazyk", str(exc))
            return
        answer = QMessageBox.question(
            self, "Language / Jazyk",
            "The language will change after a restart. Restart Orbit Solver now?\n\n"
            "Jazyk se změní po restartu. Restartovat Orbit Solver nyní?",
        )
        if answer != QMessageBox.Yes:
            return
        if self.fit_thread is not None:
            QMessageBox.information(
                self, "Language / Jazyk",
                "Wait until the running orbit calculation finishes.\n\n"
                "Počkejte, až doběhne výpočet dráhy.",
            )
            return
        QProcess.startDetached(sys.executable, sys.argv, os.getcwd())
        QApplication.instance().quit()

    def _workspace(self) -> QWidget:
        area = QWidget(); outer = QVBoxLayout(area); outer.setContentsMargins(18, 18, 18, 18)
        header = QHBoxLayout()
        heading = QVBoxLayout(); self.object_title = QLabel(tr("Žádný objekt")); self.object_title.setObjectName("title")
        self.object_label = QLabel(self.solution.designation); self.object_label.setObjectName("muted")
        heading.addWidget(self.object_title); heading.addWidget(self.object_label); header.addLayout(heading); header.addStretch()
        cards = []
        for title, value in ((tr("Pozorování"), "0"), (tr("Časový oblouk"), "—"), ("RMS", "—")):
            card, label = metric_card(title, value); cards.append(label); header.addWidget(card)
        self.obs_metric, self.arc_metric, self.rms_metric = cards
        outer.addLayout(header)

        self.tabs = QTabWidget()
        obs_page = QWidget(); obs_layout = QVBoxLayout(obs_page)
        observation_actions = QHBoxLayout()
        exclude_selected = QPushButton(tr("Vyřadit vybraná z fitu"))
        include_selected = QPushButton(tr("Zařadit vybraná do fitu"))
        copy_selected_observations = QPushButton(tr("Kopírovat vybraná"))
        copy_selected_observations.setToolTip(
            tr("Zkopíruje celé vybrané řádky včetně sexagesimální RA/Dec a obou reziduí O−C.")
        )
        self.exclude_outliers_button = QPushButton(tr("Vyřadit rezidua > 2″"))
        self.exclude_outliers_button.setToolTip(
            tr(
                "Vybere a vyřadí všechna dosud zařazená pozorování, u nichž |O−C RA| nebo "
                "|O−C Dec| přesahuje 2 obloukové sekundy. Poté je potřeba dráhu "
                "přepočítat.",
            )
        )
        self.exclude_outliers_button.setEnabled(False)
        self.worst_observation_button = QPushButton("Worst obs")
        self.worst_observation_button.setToolTip(
            tr(
                "Najde a označí nejhorší dosud nevyřazené pozorování podle většího z "
                "absolutních reziduí RA a Dec.",
            )
        )
        self.worst_observation_button.setEnabled(False)
        exclude_selected.clicked.connect(lambda: self._set_selected_observations_excluded(True))
        include_selected.clicked.connect(lambda: self._set_selected_observations_excluded(False))
        copy_selected_observations.clicked.connect(self._copy_selected_observations)
        self.exclude_outliers_button.clicked.connect(self._exclude_large_residuals)
        self.worst_observation_button.clicked.connect(self._select_worst_observation)
        observation_actions.addWidget(exclude_selected); observation_actions.addWidget(include_selected)
        observation_actions.addWidget(copy_selected_observations)
        observation_actions.addSpacing(12)
        observation_actions.addWidget(self.exclude_outliers_button)
        observation_actions.addWidget(self.worst_observation_button)
        observation_actions.addStretch()
        obs_layout.addLayout(observation_actions)
        self.observation_table = QTableView(); self.observation_table.setModel(self.observation_model)
        self.observation_table.setAlternatingRowColors(True); self.observation_table.setSortingEnabled(True)
        self.observation_table.setSelectionBehavior(QTableView.SelectRows)
        self.observation_table.setSelectionMode(QTableView.ExtendedSelection)
        self.observation_copy_shortcut = QShortcut(QKeySequence.Copy, self.observation_table)
        self.observation_copy_shortcut.activated.connect(self._copy_selected_observations)
        self.observation_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        obs_layout.addWidget(self.observation_table)
        self.observation_detail = QLabel(
            tr("Kliknutím na pozorování zobrazíte observatoř a geometrii v okamžiku měření.")
        )
        self.observation_detail.setObjectName("value")
        self.observation_detail.setWordWrap(True)
        self.observation_detail.setTextInteractionFlags(Qt.TextSelectableByMouse)
        detail_frame = QFrame(); detail_frame.setObjectName("card")
        detail_layout = QVBoxLayout(detail_frame); detail_layout.setContentsMargins(14, 10, 14, 10)
        detail_layout.addWidget(self.observation_detail)
        obs_layout.addWidget(detail_frame)
        self.observation_table.selectionModel().currentRowChanged.connect(
            self._show_observation_detail
        )
        self.observation_model.dataChanged.connect(
            lambda *_: (self._update_fit_button_label(), self._refresh_residual_actions())
        )
        self.tabs.addTab(obs_page, tr("Pozorování"))
        self.elements_panel = ElementsPanel(self.solution)
        self.tabs.addTab(self.elements_panel, tr("Elementy"))
        self.ephemeris_panel = EphemerisPanel(self.solution)
        self.tabs.addTab(self.ephemeris_panel, tr("Efemerida"))
        self.evolution_panel = OrbitEvolutionPanel(self.solution)
        self.evolution_panel.show_return.connect(self._show_evolution_return)
        self.evolution_panel.evolution_ready.connect(self._evolution_ready)
        self.tabs.addTab(self.evolution_panel, tr("Vývoj dráhy"))
        solar_playback = PlaybackBar(); self.solar_panel = SolarSystemPanel(self.solution, solar_playback)
        self.tabs.addTab(self.solar_panel, tr("Sluneční soustava"))
        sky_playback = PlaybackBar(); self.sky_panel = SkyPanel(self.solution, sky_playback)
        self.tabs.addTab(self.sky_panel, tr("Pohled ze Země"))
        self.photometry_panel = PhotometryPanel(self.solution)
        self.photometry_panel.photometry_changed.connect(self._refresh_photometric_dependents)
        self.tabs.addTab(self.photometry_panel, tr("Světelná křivka"))
        for index in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(index, False)
        self.tabs.currentChanged.connect(self._refresh_solution_tab)
        outer.addWidget(self.tabs, 1)
        return area

    def load_object(self) -> None:
        designation = self.search.text().strip()
        if not designation:
            self.search.setFocus(); self.statusBar().showMessage(tr(
                "Nejdřív zadejte označení objektu.",
            ))
            return
        if not any(character.isdigit() for character in designation) and "/" not in designation:
            self.load_button.setEnabled(False)
            self.statusBar().showMessage(tr("Hledám název {0} v MPC…", designation))
            self.mpc_network.identifier(designation)
            return
        self._begin_object_load(designation)

    def _begin_object_load(self, designation: str, display_name: str | None = None) -> None:
        if designation.casefold() != self.current_designation.casefold():
            self.current_observations = []
            self.observation_model.set_rows([])
            self._refresh_residual_actions()
            self.photometry_panel.set_observations([])
            self.fit_button.setText(tr("▶  Spočítat vlastní dráhu"))
            self.fit_button.setVisible(False)
            self.preliminary_button.setVisible(False)
            self.preliminary_button.setEnabled(False)
            self.obs_metric.setText("0")
            self.arc_metric.setText("—")
            self.rms_metric.setText("—")
            self.nongravitational_fit = {}
            self.linkage_options = {}
            self.fit_constraints = {}
            self.compute_original_future = False
            self.fit_output_epoch = ""
        self.current_designation = designation
        self.current_display_name = display_name or designation
        self.search.setText(designation)
        self.object_title.setText(self.current_display_name)
        self.has_orbit = False
        self._fit_after_orbit_load = False
        self._fit_after_observations_load = False
        self.fit_button.setEnabled(False)
        self.constraints_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.load_orbit(auto=True)

    def _identifier_loaded(self, result: object) -> None:
        query, record = result
        choices = mpc_identifier_choices(record, query)
        if not choices:
            # Keep support for names accepted directly by get-orb even if the
            # optional identifier service has no catalogue record for them.
            self._begin_object_load(query)
            return
        if len(choices) == 1:
            caption, designation = choices[0]
        else:
            labels = [caption for caption, _ in choices]
            selected, accepted = QInputDialog.getItem(
                self, tr("Vyberte objekt"),
                tr("Název „{0}“ odpovídá více objektům:", query), labels, 0, False,
            )
            if not accepted:
                self.load_button.setEnabled(True)
                self.statusBar().showMessage(tr("Výběr objektu byl zrušen."))
                return
            caption, designation = choices[labels.index(selected)]
        # The type suffix belongs in the picker, not in the main heading.
        display_name = caption.rsplit(" — ", 1)[0]
        self._begin_object_load(designation, display_name)

    def _identifier_failed(self, result: object) -> None:
        designation, _message = result
        # Failure of name resolution must not prevent the original direct MPC
        # orbit request, which accepts many common names itself.
        self._begin_object_load(designation)

    def load_observations(self, force_refresh: bool = False) -> None:
        designation = self.search.text().strip()
        if not designation:
            self.search.setFocus(); self.statusBar().showMessage(tr(
                "Nejdřív zadejte označení objektu.",
            ))
            return
        self._request_observations(designation, force_refresh)

    def _request_observations(self, designation: str, force_refresh: bool = False) -> None:
        """Load astrometry from the local cache, or download it from the MPC."""
        if not force_refresh:
            cached = mpc_cache.load(designation)
            if cached is not None:
                rows = cached.observations()
                if rows:
                    self._observations_loaded(designation, rows, False, cached.downloaded_utc)
                    return
        self.statusBar().showMessage(tr("Stahuji astrometrii pro {0}…", designation))
        self._set_observation_buttons_enabled(False)
        # Start with the archive for ordinary designations; the network client
        # automatically retries NEOCP/PCCP when the object is not found there.
        self.mpc_network.observations(designation, False)

    def _set_observation_buttons_enabled(self, enabled: bool) -> None:
        self.mpc_observations_button.setEnabled(enabled)
        self.mpc_refresh_button.setEnabled(enabled)

    def _observations_loaded(
        self, designation: str, rows: list[Observation], confirmation_page: bool | None = None,
        cached_utc: datetime | None = None,
    ) -> None:
        self._set_observation_buttons_enabled(True)
        self.load_button.setEnabled(True)
        same_object = designation.strip().casefold() == self.current_designation.strip().casefold()
        self.has_orbit = self.has_orbit and same_object
        self._configure_fragment_selector(rows)
        self.observation_model.set_rows(rows)
        self.observation_detail.setText(
            tr("Kliknutím na pozorování zobrazíte observatoř a geometrii v okamžiku měření.")
        )
        orbit_note = (
            tr("výchozí dráha {0}", self.solution.data_source)
            if self.has_orbit else tr("dráha dosud nenačtena")
        )
        self.object_label.setText(tr("{0} pozorování • {1}", len(rows), orbit_note))
        self.obs_metric.setText(str(len(rows)))
        self.current_designation = designation
        if not same_object:
            self.current_display_name = designation
        self.current_observations = rows
        self._refresh_residual_actions()
        self.photometry_panel.set_observations(rows)
        self.object_title.setText(self.current_display_name or designation)
        self.solve.setEnabled(True)
        self.fit_button.setVisible(True)
        self.preliminary_button.setVisible(True)
        # Astrometry is sufficient to offer the action.  If its required
        # initial Cartesian state has not been loaded yet, calculate_own_orbit
        # obtains that state from MPC before starting the native fit.
        self.fit_button.setEnabled(bool(rows))
        self.preliminary_button.setEnabled(fit_selection_count(rows, None) >= 3)
        self._update_fit_button_label()
        self.constraints_button.setEnabled(bool(rows))
        times = sorted(row.time for row in rows if row.time)
        self.arc_metric.setText(f"{times[0][:10]} → {times[-1][:10]}" if times else "—")
        source_note = (
            tr(
                " z místní cache (staženo z MPC {0:%Y-%m-%d %H:%M} UTC; pro nová data "
                "použijte Aktualizovat astrometrii z MPC)",
                cached_utc,
            ) if cached_utc is not None
            else tr(" z NEOCP/PCCP") if confirmation_page is True
            else tr(" z MPC") if confirmation_page is False else ""
        )
        self.statusBar().showMessage(
            tr(
                "Načteno {0} pozorování{1}. Výchozí dráha se tím nemění.",
                len(rows),
                source_note,
            )
        )
        if self._fit_after_observations_load:
            self._fit_after_observations_load = False
            self.statusBar().showMessage(
                tr("Načteno {0} pozorování{1}; počítám vlastní dráhu…", len(rows), source_note)
            )
            self._start_own_orbit_fit(None)

    def _observations_failed(self, message: str) -> None:
        automatic_fit = self._fit_after_observations_load
        self._fit_after_observations_load = False
        self._set_observation_buttons_enabled(True)
        self.load_button.setEnabled(True)
        self.statusBar().showMessage(
            tr("Automatické načtení objektu selhalo.") if automatic_fit else tr(
                "Načtení MPC selhalo.",
            )
        )
        QMessageBox.warning(self, "MPC", message)

    def load_orbit(self, auto: bool = False) -> None:
        if not self.current_designation:
            return
        self._auto_orbit_request = bool(auto)
        self.solve.setEnabled(False)
        self.statusBar().showMessage(tr(
            "Načítám výchozí kartézský stav pro {0}…",
            self.current_designation,
        ))
        self.mpc_network.orbit(self.current_designation)

    def _orbit_loaded(self, solution: OrbitSolution) -> None:
        self._fit_after_observations_load = False
        continue_with_fit = self._fit_after_orbit_load
        self._fit_after_orbit_load = False
        self._auto_orbit_request = False
        self.has_orbit = True
        self.current_display_name = solution.display_name or solution.designation
        self.load_button.setEnabled(True)
        self.solve.setEnabled(True)
        self.fit_button.setVisible(bool(self.current_observations))
        self.fit_button.setEnabled(bool(self.current_observations))
        self.preliminary_button.setVisible(bool(self.current_observations))
        self.preliminary_button.setEnabled(fit_selection_count(self.current_observations, None) >= 3)
        self.constraints_button.setEnabled(True)
        for index in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(index, True)
        self.tabs.setCurrentIndex(1)
        self._apply_solution(solution)
        self.object_title.setText(self.current_display_name)
        observation_note = tr(" • {0} pozorování", len(self.current_observations)) if self.current_observations else ""
        self.object_label.setText(
            tr("Výchozí kartézský stav {0}{1}", solution.data_source, observation_note)
        )
        if continue_with_fit:
            self.statusBar().showMessage(
                tr(
                    "Výchozí dráha {0} načtena; spouštím vlastní fit z astrometrie…",
                    solution.data_source,
                )
            )
            self.calculate_own_orbit()
        else:
            self.statusBar().showMessage(
                tr("Dráha načtena. Grafy a efemerida se přepočítají při otevření záložky."),
                8000,
            )

    def load_observations_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, tr("Načíst MPC pozorování"), "", tr(
                "Textové soubory (*.txt *.obs);;Všechny soubory (*)",
            )
        )
        if not path:
            return
        try:
            try:
                text = Path(path).read_text(encoding="utf-8-sig")
            except UnicodeDecodeError:
                text = Path(path).read_text(encoding="latin-1")
            rows = parse_mpc80(text, self.search.text().strip())
        except (OSError, MpcError) as exc:
            QMessageBox.warning(self, tr("Pozorování"), str(exc))
            return
        designation = self.search.text().strip() or rows[0].designation
        if not self.search.text().strip():
            self.search.setText(designation)
        self._observations_loaded(designation, rows)

    def paste_observations(self) -> None:
        dialog = QDialog(self); dialog.setWindowTitle(tr("Vložit MPC 80-col pozorování"))
        dialog.resize(900, 520)
        editor = QPlainTextEdit(); editor.setPlaceholderText(
            tr("Vložte jeden nebo více řádků astrometrie ve formátu MPC 80-column…")
        )
        editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept); buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog); layout.addWidget(editor, 1); layout.addWidget(buttons)
        if dialog.exec() != QDialog.Accepted:
            return
        try:
            rows = parse_mpc80(editor.toPlainText(), self.search.text().strip())
        except MpcError as exc:
            QMessageBox.warning(self, tr("Pozorování"), str(exc))
            return
        designation = self.search.text().strip() or rows[0].designation
        if not self.search.text().strip():
            self.search.setText(designation)
        self._observations_loaded(designation, rows)

    def _orbit_failed(self, message: str) -> None:
        requested_for_fit = self._fit_after_orbit_load
        self._fit_after_orbit_load = False
        automatic = self._auto_orbit_request
        self._auto_orbit_request = False
        self.solve.setEnabled(True)
        self.fit_button.setEnabled(bool(self.current_observations))
        if requested_for_fit:
            self.load_button.setEnabled(True)
            self.statusBar().showMessage(
                tr("MPC výchozí dráhu nemá; určuji počáteční stav přímo z astrometrie…")
            )
            self._start_own_orbit_fit(None)
        elif automatic:
            self.statusBar().showMessage(
                tr("Katalogová dráha není dostupná; hledám pozorování v MPC a NEOCP/PCCP…")
            )
            self._fit_after_observations_load = True
            self._request_observations(self.current_designation)
        else:
            self.load_button.setEnabled(True)
            self.statusBar().showMessage(tr("Orbitální data MPC se nepodařilo načíst."))
            QMessageBox.warning(self, tr("Dráha"), message)

    def _apply_solution(self, solution: OrbitSolution) -> None:
        self.solution = solution
        self.elements_panel.set_solution(solution)
        # Hidden tabs used to redraw immediately here.  For an old comet with
        # A1/A2/A3 that meant several duplicate multi-year N-body integrations
        # in the GUI thread and Windows reported the app as not responding.
        # Update their data now and render each panel only when it is opened.
        self.photometry_panel.solution = solution
        self.photometry_panel.observations = self.current_observations
        self.ephemeris_panel.solution = solution
        self.solar_panel.solution = solution
        # Reset before the Evolution panel may publish the fit's integration.
        self.solar_panel.set_evolution(None, None)
        self.evolution_panel.set_solution(solution)
        self.sky_panel.solution = solution
        self.sky_panel.free_center_ra = None
        self.sky_panel.free_center_dec = None
        self._dirty_solution_tabs.update((2, 4, 5, 6))
        self._refresh_solution_tab(self.tabs.currentIndex())

    def _refresh_solution_tab(self, index: int) -> None:
        if index not in self._dirty_solution_tabs:
            return
        self._dirty_solution_tabs.discard(index)
        if index == 2:
            self.ephemeris_panel.generate()
        elif index == 4:
            self.solar_panel.set_time(
                self.solar_panel.playback.time.dateTime().toPython()
            )
        elif index == 5:
            self.sky_panel.rebuild(
                self.sky_panel.playback.time.dateTime().toPython()
            )
        elif index == 6:
            self.photometry_panel.set_solution(self.solution)

    def _show_evolution_return(self, solution: OrbitSolution, time: datetime) -> None:
        self.solar_panel.solution = solution
        self.solar_panel.playback.time.setDateTime(time)
        self.tabs.setCurrentWidget(self.solar_panel)
        self.solar_panel.set_time(time)
        self.statusBar().showMessage(
            tr(
                "Zobrazen je oskulační návrat z Vývoje dráhy; načtená/fituovaná dráha "
                "zůstává beze změny.",
            ),
            9000,
        )

    def _evolution_ready(
        self, source: OrbitSolution, result: EvolutionResult,
    ) -> None:
        self.solar_panel.set_evolution(source, result)
        if self.tabs.currentWidget() is self.solar_panel:
            self.solar_panel.set_time(self.solar_panel.playback.time.dateTime().toPython())
        else:
            self._dirty_solution_tabs.add(4)
        self.statusBar().showMessage(
            tr(
                "Vývoj dráhy dokončen: {0} návratů, {1} významných přiblížení, {2} "
                "planetocentrických zachycení.",
                len(result.returns),
                len(result.encounters),
                len(result.captures),
            ),
            12000,
        )

    def _refresh_photometric_dependents(self) -> None:
        if not hasattr(self, "sky_panel"):
            return
        if self.tabs.currentIndex() == 2:
            self.ephemeris_panel.generate()
        else:
            self._dirty_solution_tabs.add(2)
        if self.tabs.currentIndex() == 5:
            self.sky_panel.rebuild(self.sky_panel.playback.time.dateTime().toPython())
        else:
            self._dirty_solution_tabs.add(5)

    def _set_selected_observations_excluded(self, excluded: bool) -> None:
        rows = {index.row() for index in self.observation_table.selectionModel().selectedRows()}
        if not rows:
            rows = {index.row() for index in self.observation_table.selectedIndexes()}
        for row in rows:
            self.current_observations[row].excluded_from_fit = excluded
            self.current_observations[row].used_in_fit = None
        if rows:
            self.observation_model.set_rows(self.current_observations)
            self._update_fit_button_label()
            self._refresh_residual_actions()
            action = tr("Vyřazeno") if excluded else tr("Zařazeno")
            self.statusBar().showMessage(tr("{0} {1} vybraných pozorování.", action, len(rows)), 6000)

    def _copy_selected_observations(self) -> None:
        selection = self.observation_table.selectionModel()
        rows = sorted({index.row() for index in selection.selectedRows()})
        if not rows:
            rows = sorted({index.row() for index in self.observation_table.selectedIndexes()})
        if not rows and selection.currentIndex().isValid():
            rows = [selection.currentIndex().row()]
        if not rows:
            self.statusBar().showMessage(tr("Nejdřív vyberte alespoň jedno pozorování."), 5000)
            return
        observations = [self.observation_model.rows[row] for row in rows]
        QApplication.clipboard().setText(observations_to_tsv(observations))
        self.statusBar().showMessage(
            tr("Zkopírováno {0} pozorování včetně O−C RA a Dec.", len(observations)), 6000
        )

    def _active_residual_rows(self) -> list[tuple[int, float]]:
        result: list[tuple[int, float]] = []
        for row, observation in enumerate(self.observation_model.rows):
            if observation.excluded_from_fit:
                continue
            score = observation_residual_score(observation)
            if score is not None:
                result.append((row, score))
        return result

    def _refresh_residual_actions(self) -> None:
        if not hasattr(self, "worst_observation_button"):
            return
        active = self._active_residual_rows()
        outlier_count = sum(score > 2.0 for _, score in active)
        self.worst_observation_button.setEnabled(bool(active))
        self.exclude_outliers_button.setText(
            tr("Vyřadit rezidua > 2″ ({0})", outlier_count)
            if outlier_count else tr("Vyřadit rezidua > 2″")
        )
        self.exclude_outliers_button.setEnabled(outlier_count > 0)

    def _select_table_rows(self, rows: list[int]) -> None:
        selection = self.observation_table.selectionModel()
        selection.clearSelection()
        for row in rows:
            index = self.observation_model.index(row, 0)
            selection.select(index, QItemSelectionModel.Select | QItemSelectionModel.Rows)
        if rows:
            first = self.observation_model.index(rows[0], 0)
            selection.setCurrentIndex(first, QItemSelectionModel.NoUpdate)
            self.observation_table.scrollTo(first, QTableView.PositionAtCenter)

    def _exclude_large_residuals(self) -> None:
        rows = [row for row, score in self._active_residual_rows() if score > 2.0]
        if not rows:
            self.statusBar().showMessage(
                tr("Žádné zařazené pozorování nemá reziduum větší než 2″."), 6000
            )
            self._refresh_residual_actions()
            return
        for row in rows:
            observation = self.observation_model.rows[row]
            observation.excluded_from_fit = True
            observation.used_in_fit = None
        self.observation_model.set_rows(self.current_observations)
        self._select_table_rows(rows)
        self._update_fit_button_label()
        self._refresh_residual_actions()
        self.statusBar().showMessage(
            tr("Vyřazeno {0} pozorování s reziduem > 2″; nyní dráhu přepočítejte.", len(rows)),
            10000,
        )

    def _select_worst_observation(self) -> None:
        active = self._active_residual_rows()
        if not active:
            self.statusBar().showMessage(
                tr("Nejhorší pozorování lze určit až po výpočtu reziduí."), 6000
            )
            self._refresh_residual_actions()
            return
        row, score = max(active, key=lambda item: item[1])
        self._select_table_rows([row])
        observation = self.observation_model.rows[row]
        self.statusBar().showMessage(
            tr(
                "Nejhorší nevyřazené pozorování: řádek {0}, {1}, reziduum {2:.2f}″.",
                row + 1,
                observation.time,
                score,
            ),
            10000,
        )

    def _configure_fragment_selector(self, rows: list[Observation]) -> None:
        """Offer the components of a split comet and fit the principal one."""
        counts = fragment_counts(rows)
        primary = primary_fragment(rows)
        unlabelled = sum(1 for row in rows if not observation_fragment(row))
        self.fragment_selector.blockSignals(True)
        self.fragment_selector.clear()
        if len(counts) >= 2:
            for letters, count in counts.items():
                if letters == primary:
                    text = tr("{0} – hlavní složka ({1})", letters, count + unlabelled)
                else:
                    text = f"{letters} ({count})"
                self.fragment_selector.addItem(text, letters)
            self.fragment_selector.setCurrentIndex(list(counts).index(primary))
            select_fragment(rows, primary)
        else:
            select_fragment(rows, None)
        self.fragment_selector.blockSignals(False)
        self.fragment_label.setVisible(len(counts) >= 2)
        self.fragment_selector.setVisible(len(counts) >= 2)

    def _fragment_selection_changed(self, _index: int) -> None:
        fragment = self.fragment_selector.currentData()
        if not fragment or not self.current_observations:
            return
        selected = select_fragment(self.current_observations, fragment)
        for observation in self.current_observations:
            observation.used_in_fit = None
        self.observation_model.set_rows(self.current_observations)
        self._update_fit_button_label()
        self._refresh_residual_actions()
        self.statusBar().showMessage(
            tr(
                "Vybrán fragment {0}: {1} měření; ostatní složky nemají ve fitu váhu. "
                "Spusťte vlastní fit znovu.",
                fragment, selected,
            ),
            10000,
        )

    def _update_fit_button_label(self) -> None:
        total = len(self.current_observations)
        selected = fit_selection_count(
            self.current_observations, self.fit_interval.currentData()
        ) if total else 0
        if hasattr(self, "preliminary_button") and self.preliminary_thread is None:
            self.preliminary_button.setEnabled(fit_selection_count(
                self.current_observations, None,
            ) >= 3)
        if not total:
            self.fit_button.setText(tr("▶  Spočítat vlastní dráhu"))
        elif selected == total:
            self.fit_button.setText(tr("▶  Spočítat vlastní dráhu ({0} měření)", selected))
        else:
            self.fit_button.setText(
                tr("▶  Spočítat vlastní dráhu ({0} z {1} měření)", selected, total)
            )
        self.fit_button.setToolTip(
            tr(
                "Do zvoleného časového rozsahu a ručního výběru spadá {0} z {1} načtených "
                "měření. Bez katalogové dráhy program určí počáteční stav přímo z "
                "astrometrie.",
                selected,
                total,
            )
        )

    def _show_observation_detail(self, current: QModelIndex, previous: QModelIndex) -> None:
        if not current.isValid() or not (0 <= current.row() < len(self.observation_model.rows)):
            return
        observation = self.observation_model.rows[current.row()]
        station = observatory_details(observation.station)
        station_name = str(station["name"]) if station else tr("neznámá observatoř")
        location_text = tr("poloha není v katalogu MPC")
        if station and "latitude_deg" in station:
            latitude = float(station["latitude_deg"])
            longitude = float(station["longitude_deg"])
            location_text = (
                f"{abs(latitude):.4f}° {'N' if latitude >= 0 else 'S'}, "
                f"{abs(longitude):.4f}° {'E' if longitude >= 0 else 'W'}"
            )
        geometry_text = tr("výška a elongace nejsou dostupné")
        if observation.ra_deg is not None and observation.dec_deg is not None and observation.time:
            try:
                instant = parse_observation_time(observation.time)
                geometry = observation_geometry(
                    observation.station, instant, observation.ra_deg, observation.dec_deg
                )
                if geometry:
                    geometry_text = (
                        tr(
                            "výška objektu {0:+.1f}°   •   výška Slunce {1:+.1f}°   •   elongace "
                            "{2:.1f}°",
                            geometry['target_altitude_deg'],
                            geometry['sun_altitude_deg'],
                            geometry['elongation_deg'],
                        )
                    )
            except (ValueError, TypeError):
                pass
        magnitude = "—" if observation.magnitude is None else f"{observation.magnitude:.2f}"
        if observation.band:
            magnitude += f" {observation.band}"
        exclusion_reason = observation.raw.get("automatic_exclusion_reason")
        exclusion_text = (
            tr("\nAutomaticky vyřazeno: {0}", exclusion_reason)
            if exclusion_reason else ""
        )
        self.observation_detail.setText(
            f"{observation.time}   •   {observation.station or '—'} — {station_name}\n"
            f"{location_text}   •   {geometry_text}   •   mag {magnitude}"
            f"{exclusion_text}"
        )

    def configure_constraints(self) -> None:
        settings_solution = self.solution
        if not self.has_orbit:
            settings_solution = replace(
                self.solution,
                designation=self.current_designation or self.solution.designation,
                display_name=self.current_display_name or self.current_designation,
                nongravitational_parameters=None,
                dynamical_1a=None,
            )
        dialog = OrbitConstraintsDialog(
            settings_solution, self.fit_constraints, self.nongravitational_fit,
            self.compute_original_future, self.fit_output_epoch,
            self.has_orbit, self, linkage_options=self.linkage_options,
        )
        if dialog.exec() != QDialog.Accepted:
            return
        self.fit_constraints = dialog.selected_constraints()
        self.nongravitational_fit = dialog.selected_nongravitational()
        self.linkage_options = dialog.selected_linkage_options()
        self.compute_original_future = dialog.calculate_original_future()
        self.fit_output_epoch = dialog.selected_output_epoch()
        if self.fit_output_epoch:
            try:
                parse_output_epoch(self.fit_output_epoch)
            except ValueError:
                QMessageBox.warning(
                    self, tr("Výstupní epocha"),
                    tr(
                        "Neplatná výstupní epocha. Použijte například 2026, 2026-10-18 nebo "
                        "2026-10-18 00:00.",
                    ),
                )
                self.fit_output_epoch = ""
        fitted_ng_names = [
            name for name in ("A1", "A2", "A3", "DT")
            if name in self.nongravitational_fit
        ]
        linkage = [
            text for key, text in (
                ("historical_nongravitational", tr("historické A1H–A3H")),
                ("accept_model_tension", tr("napojení s napětím modelu")),
            ) if self.linkage_options.get(key)
        ]
        count = (
            len(self.fit_constraints) + len(fitted_ng_names)
            + int(self.compute_original_future)
            + int(bool(self.fit_output_epoch)) + len(linkage)
        )
        self.constraints_button.setText(
            tr("Vlastní nastavení… ({0})", count) if count else tr("Vlastní nastavení…")
        )
        if count:
            fixed = [f"{name}={value:g}" for name, value in self.fit_constraints.items()]
            fitted = [f"fit {name}" for name in fitted_ng_names]
            if "R_EXPONENT" in self.nongravitational_fit:
                fitted.append(
                    tr("mocninný model m={0:g}", self.nongravitational_fit['R_EXPONENT'])
                )
            dynamics = ["original/future 1/a"] if self.compute_original_future else []
            epoch = ([tr("výstupní epocha {0}", self.fit_output_epoch)]
                     if self.fit_output_epoch else [])
            summary = ", ".join(fixed + fitted + dynamics + epoch + linkage)
            self.statusBar().showMessage(tr("Aktivní vlastní nastavení: {0}", summary), 10000)
        else:
            self.statusBar().showMessage(tr("Všech šest elementů je ve vlastním fitu volných."), 6000)

    def calculate_preliminary_orbit(self) -> None:
        if fit_selection_count(self.current_observations, None) < 3:
            QMessageBox.information(
                self, tr("Předběžná dráha"),
                tr("Pro předběžný výpočet jsou potřeba alespoň tři platná měření."),
            )
            return
        dialog = PreliminaryOrbitDialog(len(self.current_observations), self)
        if dialog.exec() != QDialog.Accepted:
            return
        method = dialog.selected_method()
        perihelion_distance = dialog.selected_perihelion_distance()
        self.preliminary_button.setEnabled(False)
        self.fit_button.setEnabled(False)
        method_text = tr("systematicky") if method == "systematic" else tr("Väisälovou metodou")
        self.statusBar().showMessage(
            tr("Hledám předběžnou dráhu {0}; výpočet běží na pozadí…", method_text)
        )
        thread = QThread(self)
        worker = PreliminaryOrbitWorker(
            self.current_observations, method, perihelion_distance,
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.success.connect(self._preliminary_orbit_loaded)
        worker.error.connect(self._preliminary_orbit_failed)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._preliminary_finished)
        self.preliminary_thread = thread
        self.preliminary_worker = worker
        thread.start()

    def _preliminary_orbit_loaded(self, result: object) -> None:
        solution, method, assumed_q = result
        solution = replace(
            solution,
            designation=self.current_designation or solution.designation,
            display_name=self.current_display_name or solution.designation,
        )
        dialog = PreliminaryOrbitResultDialog(solution, method, assumed_q, self)
        choice = dialog.exec()
        if choice not in (PreliminaryOrbitResultDialog.USE, PreliminaryOrbitResultDialog.REFINE):
            self.statusBar().showMessage(tr("Předběžné řešení nebylo použito."), 6000)
            return
        self.has_orbit = True
        self.ephemeris_panel.use_short_arc_preset()
        self._apply_solution(solution)
        self.object_title.setText(self.current_display_name or solution.designation)
        method_label = tr("systematický odhad") if method == "systematic" else tr(
            "Väisälova hypotéza",
        )
        self.object_label.setText(
            tr("Předběžná dráha z krátkého oblouku • {0} • není jednoznačná", method_label)
        )
        self.constraints_button.setEnabled(True)
        for index in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(index, True)
        if choice == PreliminaryOrbitResultDialog.REFINE:
            self._start_own_orbit_fit(solution)
        else:
            self.tabs.setCurrentIndex(1)
            self.statusBar().showMessage(
                tr(
                    "Použita předběžná dráha. Efemerida je nastavena po 10 minutách na 48 "
                    "hodin; výsledek může mít velkou a rychle rostoucí nejistotu.",
                ), 15000,
            )

    def _preliminary_orbit_failed(self, message: str) -> None:
        QMessageBox.warning(self, tr("Předběžná dráha"), message)
        self.statusBar().showMessage(tr("Výpočet předběžné dráhy selhal."), 7000)

    def _preliminary_finished(self) -> None:
        self.preliminary_thread = None
        self.preliminary_worker = None
        available = fit_selection_count(self.current_observations, None) >= 3
        self.preliminary_button.setEnabled(available and self.fit_thread is None)
        if self.fit_thread is None:
            self.fit_button.setEnabled(bool(self.current_observations))

    def calculate_own_orbit(self) -> None:
        if not self.current_observations:
            return
        if not self.has_orbit or self.solution.position_au is None:
            self._fit_after_orbit_load = True
            self.fit_button.setEnabled(False)
            self.statusBar().showMessage(
                tr(
                    "Nejprve načítám z MPC počáteční odhad; poté se vlastní fit spustí "
                    "automaticky…",
                )
            )
            self.load_orbit(auto=False)
            return
        self._start_own_orbit_fit(self.solution)

    def _start_own_orbit_fit(self, initial: OrbitSolution | None) -> None:
        self.fit_button.setEnabled(False)
        self.solve.setEnabled(False)
        self.fit_interval.setEnabled(False)
        self.fragment_selector.setEnabled(False)
        truncate_years = self.fit_interval.currentData()
        interval_label = self.fit_interval.currentText().lower()
        mode = f" — {interval_label}"
        extra = tr(" a následně original/future 1/a") if self.compute_original_future else ""
        self.statusBar().showMessage(
            tr("Počítám vlastní dráhu z astrometrie{0}{1}…", mode, extra)
        )
        thread = QThread(self)
        worker = OrbitFitWorker(
            self.current_observations, initial, self.fit_constraints, truncate_years,
            self.nongravitational_fit, self.compute_original_future,
            self.linkage_options,
        )
        progress_dialog = OrbitFitProgressDialog(self)
        # A queued slot on ``worker`` could not run while its thread is inside
        # scipy.  The tiny lambda flips a thread-safe Python flag immediately
        # in the GUI thread; the numerical residual checks it cooperatively.
        progress_dialog.cancel_requested.connect(
            lambda: worker.request_cancel()
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.success.connect(self._own_orbit_loaded)
        worker.error.connect(self._own_orbit_failed)
        worker.progress.connect(progress_dialog.update_progress)
        worker.cancelled.connect(self._own_orbit_cancelled)
        worker.finished.connect(progress_dialog.finish)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._fit_finished)
        self.fit_thread = thread
        self.fit_worker = worker
        self.fit_progress_dialog = progress_dialog
        progress_dialog.show()
        thread.start()

    def _announce_impact(self, prediction) -> None:
        label = BODY_LABELS.get(prediction.body, prediction.body)
        self.statusBar().showMessage(
            tr("POZOR: vypočtená dráha vede k dopadu na těleso {0}.", label), 0,
        )
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle(tr("Předpověď dopadu"))
        box.setText(
            "<b>" + html.escape(tr("Vypočtená dráha vede k dopadu na těleso {0}.", label))
            + "</b>"
        )
        box.setInformativeText(
            describe_impact(prediction, label)
            + tr(
                "\n\nKontroluje se {0:.0f} dní od epochy dráhy. Místo na povrchu je průsečík "
                "vakuové dráhy s elipsoidem WGS84; brzdění a rozpad v atmosféře nejsou "
                "modelovány. Podrobnosti jsou v záložce Elementy.",
                IMPACT_CHECK_DAYS,
            )
        )
        box.exec()

    def _own_orbit_loaded(self, payload: object) -> None:
        result, fitted_observations = payload
        solution, rms, _used = result
        apply_fitted_observations(self.current_observations, fitted_observations)
        self.observation_model.set_rows(self.current_observations)
        self._refresh_residual_actions()
        self.has_orbit = True
        self.tabs.setCurrentIndex(1)
        self._apply_solution(solution)
        if solution.impact is not None:
            QTimer.singleShot(0, lambda prediction=solution.impact: self._announce_impact(prediction))
        if self.fit_output_epoch:
            self.elements_panel.output_epoch.setText(self.fit_output_epoch)
            QTimer.singleShot(0, self.elements_panel._apply_output_epoch)
        self.constraints_button.setEnabled(True)
        for index in range(1, self.tabs.count()):
            self.tabs.setTabEnabled(index, True)
        normalized = solution.fit_normalized_rms
        self.rms_metric.setText(
            f"{rms:.1f}″ / {normalized:.2f}σ" if normalized is not None else f"{rms:.2f}″"
        )
        self.rms_metric.setToolTip(
            tr(
                "První hodnota je nevážené RMS v obloukových sekundách; druhá zohledňuje "
                "nejistoty rmsRA/rmsDec uvedené v ADES a částečnou korelaci série ze "
                "stejné stanice během jedné noci.",
            )
        )
        interval = f" • {self.fit_interval.currentText().lower()}"
        self.object_title.setText(
            self.current_display_name or solution.display_name or solution.designation
        )
        count_text = apparition_observation_count_text(solution)
        linkage_text = apparition_linkage_summary(solution)
        label_parts = [tr("Vlastní fit")]
        if self.fragment_selector.isVisible() and self.fragment_selector.currentData():
            label_parts.append(tr("fragment {0}", self.fragment_selector.currentData()))
        if linkage_text:
            label_parts.append(linkage_text)
        if count_text:
            label_parts.append(count_text)
        self.object_label.setText(" • ".join(label_parts) + interval)
        weighted_note = tr(", vážené RMS {0:.2f}σ", normalized) if normalized is not None else ""
        dynamics_note = ""
        if solution.dynamical_1a:
            dynamics_note = (
                tr("; original/future výpočet se nezdařil")
                if "error" in solution.dynamical_1a
                else tr("; original/future 1/a dopočítáno")
            )
        linkage_note = f"; {linkage_text.lower()}" if linkage_text else ""
        self.statusBar().showMessage(
            tr(
                "Vlastní dráha dokončena; RMS {0:.2f} obloukové sekundy{1}{2}{3}.",
                rms,
                weighted_note,
                linkage_note,
                dynamics_note,
            ), 20000
        )
        failed_linkage = [
            record for record in solution.apparition_linkage or ()
            if record.get("status") in ("failed", "not_attempted")
        ]
        if failed_linkage:
            details = []
            for record in failed_linkage:
                year = record.get("return_year") or "?"
                reason = record.get("reason") or tr("bez podrobností")
                details.append(tr("Návrat {0}: {1}", year, reason))
            QMessageBox.warning(
                self,
                tr("Společný dynamický model nepotvrdil všechny návraty"),
                tr(
                    "Výsledná dráha NENÍ společným řešením všech načtených návratů. To samo o "
                    "sobě nevyvrací jejich identitu: příčinou může být proměnlivá kometární "
                    "aktivita nebo nedostatečný silový model.\n\n",
                )
                + "\n".join(details)
                + (
                    tr(
                        "\n\nTip: návraty byly propojeny čistě gravitačním modelem. Aktivní "
                        "kometa (např. 1P/Halley) se přes více oběhů obvykle navázat nedá bez "
                        "negravitačních parametrů; povolte A1 a A2 ve Vlastním nastavení a "
                        "výpočet zopakujte.",
                    )
                    if not solution.nongravitational_parameters
                    and is_comet_designation(self.current_designation or solution.designation)
                    else ""
                )
                + tr("\n\nPodrobný protokol je v záložce Elementy."),
            )

    def _own_orbit_failed(self, message: str) -> None:
        QMessageBox.warning(self, tr("Výpočet dráhy"), message)
        self.statusBar().showMessage(tr("Vlastní výpočet dráhy selhal."))

    def _own_orbit_cancelled(self) -> None:
        self.statusBar().showMessage(
            tr("Výpočet dráhy byl zrušen; předchozí řešení zůstalo zachováno."),
            10000,
        )

    def _fit_finished(self) -> None:
        self.fit_thread = None
        self.fit_worker = None
        self.fit_progress_dialog = None
        self.fit_button.setEnabled(True)
        self.preliminary_button.setEnabled(
            fit_selection_count(self.current_observations, None) >= 3
        )
        self.solve.setEnabled(True)
        self.fit_interval.setEnabled(True)
        self.fragment_selector.setEnabled(True)


def run() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("Orbit Solver")
    app.setStyle("Fusion")
    app.setStyleSheet(APP_STYLE)
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    window = MainWindow()
    window.show()
    return app.exec()

