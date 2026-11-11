"""Split-comet tools of the main window: banner, overview and split estimate.

The numerical work lives in :mod:`orbit_app.fragments`; this module only
formats it for the observation page and runs the split-epoch fit off the GUI
thread.
"""
from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QObject, Qt, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog, QDialogButtonBox, QFrame, QHBoxLayout, QHeaderView, QLabel,
    QPlainTextEdit, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from .astrometry import is_fit_candidate, primary_fragment, select_fragment
from .domain import Observation, OrbitSolution
from .fragments import (
    AssignmentReport, ComponentSummary, SecondaryTrack, describe_split,
    estimate_split, primary_label,
)
from .i18n import tr
from .orbit_fit import OrbitFitCancelled


def _date(value) -> str:
    return value.strftime("%Y-%m-%d") if value is not None else "—"


def _offset_text(summary: ComponentSummary) -> str:
    if summary.principal:
        return "—"
    if summary.offset_arcsec is None:
        return tr("bez společných snímků")
    return f"{summary.offset_arcsec:.1f}″"


def fragment_banner_text(
    summaries: list[ComponentSummary],
    solutions: dict[str, OrbitSolution],
    report: AssignmentReport | None = None,
    tracks: list[SecondaryTrack] | None = None,
) -> str:
    """Plain-text summary shown above the observation table ('' = no banner)."""
    tracks = list(tracks or ())
    if not summaries and not tracks:
        return ""
    lines: list[str] = []
    if summaries:
        lines.append(tr(
            "Astrometrie obsahuje více složek rozpadlé komety ({0}); fituje se vždy "
            "jen vybraná složka.",
            len(summaries),
        ))
        parts = []
        for summary in summaries:
            name = primary_label(summary.component)
            text = tr(
                "{0}: {1} měření, {2} nocí, {3} → {4}",
                name, summary.rows, summary.nights, _date(summary.first), _date(summary.last),
            )
            if summary.principal:
                text += tr(" (hlavní složka)")
            elif summary.offset_arcsec is not None:
                text += tr(", {0:.1f}″ od hlavní složky", summary.offset_arcsec)
            if summary.component in solutions:
                text += " ✓"
            parts.append(text)
        lines.append(" • ".join(parts))
        fitted = [item.component for item in summaries if item.component in solutions]
        missing = [item.component for item in summaries if item.component not in solutions]
        if fitted and missing:
            lines.append(tr(
                "✓ = složka s vlastní drahou. Další složky spočítáte výběrem fragmentu "
                "vlevo; fit vedlejší složky startuje z dráhy hlavní.",
            ))
    if report is not None and report.compared:
        assigned = sum(report.assigned.values())
        if assigned:
            detail = ", ".join(
                f"{primary_label(letters)} {count}" for letters, count in sorted(report.assigned.items())
            )
            lines.append(tr(
                "Neoznačená měření po rozpadu: {0} z {1} porovnaných přiřazeno vedlejším "
                "složkám ({2}); pro jejich nové rozdělení spusťte fit znovu.",
                assigned, report.compared, detail,
            ))
        else:
            lines.append(tr(
                "Neoznačená měření po rozpadu ({0}) odpovídají hlavní složce.",
                report.compared,
            ))
        if report.unresolved:
            lines.append(tr(
                "{0} měření nelze rozlišit (stopy jsou blíž než chyby měření); "
                "zůstávají hlavní složce.",
                report.unresolved,
            ))
    for track in tracks:
        lines.append(tr(
            "Fit našel souvislou vedlejší stopu bez označení fragmentu: {0} měření "
            "({1}), {2} → {3}. Může jít o nový úlomek.",
            len(track.rows), ", ".join(track.stations) or "?",
            _date(track.first), _date(track.last),
        ))
    return "\n".join(lines)


class FragmentBanner(QFrame):
    """Card above the observation table for split comets."""

    overview_requested = Signal()
    split_requested = Signal()
    split_off_requested = Signal()

    def __init__(self) -> None:
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self); layout.setContentsMargins(14, 8, 14, 8)
        self.text = QLabel()
        self.text.setWordWrap(True)
        self.text.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.text)
        buttons = QHBoxLayout()
        self.overview_button = QPushButton(tr("Přehled fragmentů…"))
        self.split_button = QPushButton(tr("Odhadnout čas rozpadu…"))
        self.split_button.setToolTip(tr(
            "Pro každou vedlejší složku s vlastní drahou najde okamžik, kdy se oddělila "
            "od hlavní složky (rychlost oddělení a/nebo rozdílné zrychlení). Výpočet "
            "trvá zhruba minutu na složku.",
        ))
        self.split_off_button = QPushButton(tr("Vyčlenit stopu jako úlomek"))
        self.split_off_button.setToolTip(tr(
            "Měření vedlejší stopy se stanou samostatnou složkou (X1, X2…), kterou lze "
            "vybrat a fitovat; z dráhy hlavního jádra zůstanou vyřazena.",
        ))
        self.overview_button.clicked.connect(self.overview_requested)
        self.split_button.clicked.connect(self.split_requested)
        self.split_off_button.clicked.connect(self.split_off_requested)
        buttons.addWidget(self.overview_button)
        buttons.addWidget(self.split_button)
        buttons.addWidget(self.split_off_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.setVisible(False)

    def update_state(
        self, text: str, *, can_split: bool, has_track: bool, busy: bool = False,
    ) -> None:
        self.text.setText(text)
        self.split_button.setEnabled(can_split and not busy)
        self.split_button.setVisible(can_split or busy)
        self.split_off_button.setVisible(has_track)
        self.split_off_button.setEnabled(has_track and not busy)
        self.setVisible(bool(text))


def show_text_dialog(parent, title: str, text: str) -> QDialog:
    """Non-modal dialog with selectable, copyable report text."""
    dialog = QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setAttribute(Qt.WA_DeleteOnClose)
    layout = QVBoxLayout(dialog)
    view = QPlainTextEdit(text); view.setReadOnly(True)
    layout.addWidget(view)
    buttons = QDialogButtonBox(QDialogButtonBox.Close)
    copy = buttons.addButton(tr("Kopírovat"), QDialogButtonBox.ActionRole)
    copy.clicked.connect(lambda: QGuiApplication.clipboard().setText(view.toPlainText()))
    buttons.rejected.connect(dialog.close)
    layout.addWidget(buttons)
    dialog.resize(860, 560)
    dialog.show()
    return dialog


class FragmentOverviewDialog(QDialog):
    """Table of components with their fitted orbits and the last split estimates."""

    def __init__(
        self, parent, summaries: list[ComponentSummary],
        solutions: dict[str, OrbitSolution], split_text: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(tr("Přehled fragmentů"))
        self.setAttribute(Qt.WA_DeleteOnClose)
        layout = QVBoxLayout(self)
        headers = [
            tr("Složka"), tr("Měření"), tr("Označeno MPC"), tr("Nocí"), tr("Návratů"),
            tr("První"), tr("Poslední"), tr("Odstup od hlavní"), tr("Vlastní dráha"),
        ]
        self.table = QTableWidget(len(summaries), len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        for row, summary in enumerate(summaries):
            solution = solutions.get(summary.component)
            orbit = "—"
            if solution is not None:
                q = solution.semi_major_axis_au * (1.0 - solution.eccentricity)
                parts = [f"e {solution.eccentricity:.5f}", f"q {q:.5f} AU"]
                if solution.fit_rms_arcsec is not None:
                    parts.append(f"RMS {solution.fit_rms_arcsec:.2f}″")
                orbit = ", ".join(parts)
            name = primary_label(summary.component)
            if summary.principal:
                name += tr(" (hlavní)")
            values = [
                name, str(summary.rows), str(summary.labelled), str(summary.nights),
                str(summary.returns), _date(summary.first), _date(summary.last),
                _offset_text(summary), orbit,
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)
        note = QLabel(tr(
            "Odstup je medián úhlové vzdálenosti od hlavní složky na snímcích téže "
            "stanice (do 2 minut). Neoznačená měření patří hlavní složce, pokud je "
            "fit nepřiřadil vedlejší.",
        ))
        note.setWordWrap(True); note.setObjectName("muted")
        layout.addWidget(note)
        self.split_view = QPlainTextEdit(
            split_text or tr("Odhad času rozpadu zatím nebyl spočítán."),
        )
        self.split_view.setReadOnly(True)
        layout.addWidget(self.split_view, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)
        self.resize(980, 620)


def fragment_fit_rows(observations: list[Observation], component: str) -> list[Observation]:
    """Copies of the fit-ready measurements of one component."""
    copies = [replace(item, raw=dict(item.raw)) for item in observations]
    select_fragment(copies, component)
    return [item for item in copies if is_fit_candidate(item)]


class SplitEstimateWorker(QObject):
    """Runs :func:`estimate_split` for every fitted secondary component."""

    success = Signal(object)          # list[SplitEstimate | str]
    error = Signal(str)
    progress = Signal(str, int, int)
    cancelled = Signal()
    finished = Signal()

    def __init__(
        self, observations: list[Observation], solutions: dict[str, OrbitSolution],
    ) -> None:
        super().__init__()
        self.observations = [replace(item, raw=dict(item.raw)) for item in observations]
        self.solutions = dict(solutions)
        self._cancel_requested = False

    def request_cancel(self) -> None:
        self._cancel_requested = True

    def is_cancel_requested(self) -> bool:
        return self._cancel_requested

    @Slot()
    def run(self) -> None:
        try:
            primary = primary_fragment(self.observations)
            if not primary or primary not in self.solutions:
                raise ValueError(tr("Nejprve spočítejte vlastní dráhu hlavní složky."))
            secondaries = sorted(
                letters for letters in self.solutions if letters != primary
            )
            if not secondaries:
                raise ValueError(tr(
                    "Nejprve spočítejte vlastní dráhu alespoň jedné vedlejší složky.",
                ))
            results: list[object] = []
            for index, letters in enumerate(secondaries):
                if self._cancel_requested:
                    raise OrbitFitCancelled()
                self.progress.emit(
                    tr("Odhaduji čas rozpadu složky {0}…", primary_label(letters)),
                    index, len(secondaries),
                )
                rows = fragment_fit_rows(self.observations, letters)
                try:
                    results.append(estimate_split(
                        self.solutions[primary], rows,
                        primary_label_text=primary_label(primary),
                        fragment_label_text=primary_label(letters),
                        cancel_callback=self.is_cancel_requested,
                    ))
                except OrbitFitCancelled:
                    raise
                except (ValueError, RuntimeError, FloatingPointError, OverflowError) as exc:
                    results.append(tr(
                        "Složka {0}: odhad času rozpadu se nezdařil – {1}",
                        primary_label(letters), exc,
                    ))
            self.success.emit(results)
        except OrbitFitCancelled:
            self.cancelled.emit()
        except Exception as exc:  # noqa: BLE001 - reported to the user
            self.error.emit(str(exc))
        finally:
            self.finished.emit()


def describe_split_results(results: list[object]) -> str:
    return "\n\n".join(
        describe_split(item) if not isinstance(item, str) else item for item in results
    )
