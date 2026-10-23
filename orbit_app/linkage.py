"""Linking astrometry of separate apparitions into one dynamical orbit.

The procedure is the classical one used for recovered periodic comets and
multi-opposition asteroids, independent of the object type:

1. If the supplied (catalogue) orbit already predicts every apparition within
   the direct-correction range, all apparitions are corrected together.
2. Otherwise the best-observed apparition (the *anchor*) is fitted alone, from
   the catalogue state or, if that fails, from its own angles by progressive
   arc extension.
3. The remaining apparitions are added one at a time, nearest in time first.
   When the current orbit does not predict the next apparition closely enough
   for a differential correction, the branch is searched along the *line of
   variations* (LOV): the weakest direction of the current covariance, i.e.
   the family of orbits that still fit the already linked astrometry.  Each
   LOV point is propagated (with planetary perturbations) to the new
   apparition; the robust minimum of the miss, refined by a one-dimensional
   Gauss-Newton step, seeds a joint differential correction of all linked
   apparitions.
4. Every stage is accepted only if each linked apparition fits its own
   astrometry (normalized RMS ≤ ``LINKAGE_ACCEPT_NORMALIZED_RMS``) after a
   bounded, visible rejection of blunders.

All integrations spanning more than ``LONG_ARC_DAYS`` use the PRECISE
tolerance profile; the LOV scan uses SCAN tolerances, whose error (a few
arcseconds per century) is far below the width of the basin being located.
Units: AU, days, TDB Julian dates, heliocentric J2000 ecliptic states.
"""
from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Callable

import numpy as np

from .astrometry import (
    is_fit_candidate, normal_place_observations, observation_sigma_arcsec,
    observing_night_key, parse_observation_time,
)
from .domain import Observation, OrbitSolution
from .i18n import tr
from .observatories import dynamical_jd, earth_heliocentric_ecliptic
from .orbit_engine import MU_SUN, SCAN_TOLERANCES, NBodyIntegrator, StateVector
from .orbit_engine import HISTORICAL_NONGRAVITATIONAL, NONGRAVITATIONAL_SPLIT_KEY
from .orbit_fit import (
    NONGRAVITATIONAL_NAMES, _PLANET_NAMES, OrbitFitCancelled, OrbitFitError,
    _apparition_linkage_record, _coherent_secondary_track_candidates, _equatorial_direction, _local_light_time_states,
    _observation_offsets_gcrs, _representative_indices,
    _automatic_orbit_score, _update_apparition_linkage_metrics,
    initial_orbit_from_observations,
)
from .planetary import PLANETS, SplinePlanetEphemeris

# An apparition is linked when its normalized RMS (per coordinate, after the
# bounded blunder rejection) does not exceed this value.
LINKAGE_ACCEPT_NORMALIZED_RMS = 3.0
# Joining a new apparition may raise already linked ones up to this value:
# the branch is identified by the new apparition fitting its own astrometry,
# while a coherent shift of an older return reveals model tension (a comet's
# variable activity), not a wrong revolution count.  It is reported as such.
LINKAGE_TENSION_NORMALIZED_RMS = 6.0
# Optional relaxed gate (``accept_model_tension``): a return whose best joint
# correction keeps every linked return within the tension limit is joined if
# its own normalized RMS stays below this value.  Meant for historical visual
# or photographic astrometry and comets whose activity a constant force law
# does not describe; such returns are always reported as model tension.
LINKAGE_RELAXED_NORMALIZED_RMS = 15.0
# A predicted miss (median normalized radial residual) below this value is
# first corrected directly (a neighbouring apparition of an active comet can
# easily miss by tens of sigma); the line of variations is searched otherwise
# or when the direct correction fails.
DIRECT_CORRECTION_MEDIAN_SIGMA = 100.0
# Remaining apparitions predicted within this median normalized miss are
# joined together in one correction.
BATCH_MEDIAN_SIGMA = 10.0
# Requested A1/A2 are fitted in a stage as soon as its observations span at
# least this many orbital periods (two perihelia carry the secular signal).
FORCE_RELEASE_PERIODS = 1.0
# Line-of-variations scan, in standard deviations of the current solution.
LOV_SCAN_HALF_WIDTH_SIGMA = 4.0
LOV_SCAN_STEP_SIGMA = 0.25
LOV_SCAN_MAXIMUM_SIGMA = 20.0
LOV_REFINED_CANDIDATES = 3
# Blunder rejection inside a linked apparition.
OUTLIER_NORMALIZED_SIGMA = 5.0
OUTLIER_MINIMUM_ARCSEC = 2.0
# Rejection is attempted only where most rows already fit (median normalized
# radial residual below this), and removes at most this fraction of an
# apparition: a wrong branch must fail, not be rescued by discarding data.
REJECTION_MEDIAN_SIGMA = 2.5
MAXIMUM_REJECTED_FRACTION_DENSE = 0.2
MAXIMUM_REJECTED_FRACTION_SPARSE = 0.5
DENSE_APPARITION_ROWS = 20
# Independent orbit of one apparition: initial windows (days) and the
# normalized RMS a window or arc-extension step must reach to be accepted.
INITIAL_WINDOWS_DAYS = (10.0, 20.0, 40.0)
EXTENSION_ACCEPT_NORMALIZED_RMS = 5.0
# Representative rows per apparition used by stage corrections.
STAGE_ROWS_PER_APPARITION = 240
SCAN_ROWS_PER_APPARITION = 40
FINAL_ROW_LIMIT = 2000
STAGE_MAX_NFEV = 20

FitOnce = Callable[..., tuple[OrbitSolution, float, int]]
Evaluate = Callable[..., tuple[float, float, int]]
ProgressReporter = Callable[..., None]


class LinkageFailed(OrbitFitError):
    """One apparition could not be joined to the already linked ones."""


@dataclass(frozen=True)
class _SkyTarget:
    """Pre-computed geometry of a set of observations (TDB, ICRS, arcsec)."""

    epochs: np.ndarray
    offsets: np.ndarray
    earth: np.ndarray
    ra: np.ndarray
    dec: np.ndarray
    sigmas: np.ndarray  # shape (n, 2)


def _candidates(group: list[Observation]) -> list[Observation]:
    return sorted(
        (item for item in group if is_fit_candidate(item)),
        key=lambda item: parse_observation_time(item.time),
    )


def _sample(rows: list[Observation], maximum: int) -> list[Observation]:
    if len(rows) <= maximum:
        return list(rows)
    epochs = dynamical_jd([parse_observation_time(item.time) for item in rows])
    return [rows[index] for index in _representative_indices(epochs, maximum)]


def _sky_target(observations: list[Observation], maximum: int) -> _SkyTarget:
    rows = _sample(_candidates(observations), maximum)
    if not rows:
        raise LinkageFailed(tr("Návrat nemá žádná použitelná měření."))
    parsed = [
        (item, parse_observation_time(item.time), float(item.ra_deg), float(item.dec_deg))
        for item in rows
    ]
    epochs = dynamical_jd([item[1] for item in parsed])
    return _SkyTarget(
        epochs=epochs,
        offsets=_observation_offsets_gcrs(parsed),
        earth=earth_heliocentric_ecliptic(epochs),
        ra=np.asarray([item[2] for item in parsed]),
        dec=np.asarray([item[3] for item in parsed]),
        sigmas=np.asarray([observation_sigma_arcsec(item) for item in rows], dtype=float),
    )


def _scan_engine(
    epoch_jd: float, target: _SkyTarget, force: dict[str, float],
) -> NBodyIntegrator:
    start = min(float(epoch_jd), float(np.min(target.epochs))) - 2.0
    end = max(float(epoch_jd), float(np.max(target.epochs))) + 2.0
    return NBodyIntegrator(
        SplinePlanetEphemeris(PLANETS, _PLANET_NAMES, start, end),
        relative_tolerance=SCAN_TOLERANCES[0], absolute_tolerance=SCAN_TOLERANCES[1],
        nongravitational_parameters=force,
    )


def _normalized_prediction(
    engine: NBodyIntegrator, epoch_jd: float, state: np.ndarray, target: _SkyTarget,
) -> np.ndarray:
    """(Computed − observed) / sigma for each target row, shape ``(n, 2)``."""
    propagated = engine.propagate(StateVector(epoch_jd, state[:3], state[3:6]), target.epochs)
    emitted, _ = _local_light_time_states(propagated, target.earth)
    ra, dec = _equatorial_direction(emitted, target.epochs, target.offsets, target.earth)
    delta_ra = ((ra - target.ra + 180.0) % 360.0 - 180.0) * np.cos(np.radians(target.dec))
    return np.column_stack((delta_ra, dec - target.dec)) * 3600.0 / target.sigmas


def _robust_miss(normalized: np.ndarray) -> float:
    """Median normalized radial miss: insensitive to up to half blunders."""
    if not np.all(np.isfinite(normalized)):
        return float("inf")
    return float(np.median(np.hypot(normalized[:, 0], normalized[:, 1])))


def _line_of_variations(solution: OrbitSolution) -> np.ndarray | None:
    """One-sigma step along the weakest direction of the state covariance.

    The eigen-decomposition is done on the correlation matrix, so position
    (AU) and velocity (AU/day) components are compared in sigma units; the
    returned vector ``v`` satisfies ``v.T @ inv(C) @ v = 1``.
    """
    if solution.state_covariance is None:
        return None
    covariance = np.asarray(solution.state_covariance, dtype=float)
    if covariance.shape != (6, 6) or not np.all(np.isfinite(covariance)):
        return None
    scale = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    if np.any(scale <= 0.0):
        return None
    correlation = covariance / np.outer(scale, scale)
    values, vectors = np.linalg.eigh((correlation + correlation.T) / 2.0)
    if values[-1] <= 0.0:
        return None
    return scale * vectors[:, -1] * np.sqrt(values[-1])


def _scan_line_of_variations(
    solution: OrbitSolution, target: _SkyTarget, force: dict[str, float],
    check_cancelled: Callable[[], None],
) -> list[tuple[float, float, np.ndarray]]:
    """Return refined LOV candidates ``(robust miss σ, k σ, state)``, best first."""
    direction = _line_of_variations(solution)
    if direction is None:
        raise LinkageFailed(tr("Pro hledání větve chybí kovariance navázané dráhy."))
    origin = np.concatenate((
        np.asarray(solution.position_au, dtype=float),
        np.asarray(solution.velocity_au_per_day, dtype=float),
    ))
    engine = _scan_engine(solution.epoch_jd, target, force)
    evaluated: dict[float, tuple[float, np.ndarray]] = {}

    def evaluate(k: float) -> tuple[float, np.ndarray]:
        key = round(float(k), 6)
        if key not in evaluated:
            check_cancelled()
            try:
                normalized = _normalized_prediction(
                    engine, solution.epoch_jd, origin + key * direction, target,
                )
            except (RuntimeError, ValueError, FloatingPointError, OverflowError):
                normalized = np.full((target.epochs.size, 2), np.inf)
            evaluated[key] = (_robust_miss(normalized), normalized)
        return evaluated[key]

    for k in np.arange(
        -LOV_SCAN_HALF_WIDTH_SIGMA, LOV_SCAN_HALF_WIDTH_SIGMA + 1e-9, LOV_SCAN_STEP_SIGMA,
    ):
        evaluate(k)
    # The minimum may lie beyond the initial window when the linked arc is
    # short; walk outward with a growing step while the miss keeps falling.
    for sign in (-1.0, 1.0):
        edge = sign * LOV_SCAN_HALF_WIDTH_SIGMA
        best_k = min(evaluated, key=lambda key: evaluated[key][0])
        if abs(best_k - edge) > 1e-9:
            continue
        step = LOV_SCAN_STEP_SIGMA
        k = edge
        while abs(k) < LOV_SCAN_MAXIMUM_SIGMA:
            step = min(step * 1.5, 2.0)
            previous = evaluate(k)[0]
            k += sign * step
            if evaluate(k)[0] >= previous:
                break

    keys = sorted(evaluated)
    scores = [evaluated[key][0] for key in keys]
    minima = [
        keys[index] for index in range(len(keys))
        if np.isfinite(scores[index])
        and (index == 0 or scores[index] <= scores[index - 1])
        and (index == len(keys) - 1 or scores[index] <= scores[index + 1])
    ]
    minima.sort(key=lambda key: evaluated[key][0])
    refined: list[tuple[float, float, np.ndarray]] = []
    for start in minima[:LOV_REFINED_CANDIDATES]:
        k = float(start)
        # One-dimensional Gauss-Newton on the trimmed residual vector.  The
        # robust miss is V-shaped, but the residual vector itself varies
        # almost linearly along the LOV near the minimum.
        for _iteration in range(8):
            score, normalized = evaluate(k)
            radial = np.hypot(normalized[:, 0], normalized[:, 1])
            keep = radial <= max(3.0, 2.0 * np.median(radial))
            if np.count_nonzero(keep) < 1:
                break
            step = 0.02
            _, shifted = evaluate(k + step)
            derivative = ((shifted - normalized) / step)[keep].ravel()
            residual = normalized[keep].ravel()
            curvature = float(derivative @ derivative)
            if not np.isfinite(curvature) or curvature <= 0.0:
                break
            change = float(np.clip(-(derivative @ residual) / curvature, -1.0, 1.0))
            if not np.isfinite(change):
                break
            if evaluate(k + change)[0] > score:
                change *= 0.5
                if evaluate(k + change)[0] > score:
                    break
            k += change
            if abs(change) < 1e-3:
                break
        refined.append((evaluate(k)[0], k, origin + k * direction))
    # Far along the LOV the straight-line approximation of the orbit family
    # degrades; between comparable minima prefer the nearer one.
    refined.sort(key=lambda item: item[0] * (1.0 + 0.05 * abs(item[1])))
    return refined


def _densest_windows(
    rows: list[Observation], days: np.ndarray, window_days: float, count: int = 3,
) -> list[float]:
    """Start days of the windows holding the most station-nights (≥ 6 rows)."""
    nights = [observing_night_key(item, parse_observation_time(item.time)) for item in rows]
    scored: list[tuple[int, float]] = []
    for start in np.unique(np.floor(days)):
        inside = (days >= start) & (days <= start + window_days)
        if np.count_nonzero(inside) < 6:
            continue
        distinct = len({nights[index] for index in np.flatnonzero(inside)})
        if distinct >= 3:
            scored.append((distinct, float(start)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    chosen: list[float] = []
    for _distinct, start in scored:
        if all(abs(start - other) > window_days for other in chosen):
            chosen.append(start)
        if len(chosen) >= count:
            break
    return chosen


def _anchor_index(groups: list[list[Observation]]) -> int:
    """Choose the apparition carrying the most independent information."""
    def information(group: list[Observation]) -> float:
        rows = _candidates(group)
        if not rows:
            return 0.0
        instants = [parse_observation_time(item.time) for item in rows]
        nights = {observing_night_key(item, instant) for item, instant in zip(rows, instants)}
        arc = (max(instants) - min(instants)).total_seconds() / 86400.0
        return len(nights) * float(np.log1p(arc))

    scores = [information(group) for group in groups]
    best = max(scores)
    # Prefer the latest apparition among equals: its geometry is usually
    # modern CCD astrometry and the catalogue epoch is close to it.
    return max(index for index, score in enumerate(scores) if score >= best * 0.999)


class _Linker:
    def __init__(
        self, observations, groups, initial, constraints, nongravitational,
        fit_once: FitOnce, evaluate: Evaluate, report_progress: ProgressReporter,
        cancel_callback, historical_nongravitational: bool = False,
        accept_model_tension: bool = False,
    ) -> None:
        self.observations = observations
        self.groups = groups
        self.initial = initial
        self.constraints = constraints
        self.requested = dict(nongravitational or {})
        self.accept_model_tension = bool(accept_model_tension)
        # Returns joined through the relaxed gate (reported as model tension).
        self.tension: set[int] = set()
        self.split_jd: float | None = None
        if historical_nongravitational:
            self._request_historical_force()
        self.free_names = tuple(name for name in NONGRAVITATIONAL_NAMES if name in self.requested)
        # While branches are searched the requested seeds act as a fixed force
        # law (zero A means gravity only).  A zero seed of a parameter that the
        # catalogue orbit carries starts from the catalogue value instead.
        catalogue = (initial.nongravitational_parameters or {}) if initial is not None else {}
        self.fixed_force = {
            name: float(value) if float(value) != 0.0 or name not in catalogue
            else float(catalogue[name])
            for name, value in self.requested.items()
        }
        self._fit_once = fit_once
        self.evaluate = evaluate
        self.report_progress = report_progress
        self.cancel_callback = cancel_callback
        self.records = [_apparition_linkage_record(group, initial) for group in groups]
        self.rejections: dict[int, list[Observation]] = {}
        # Whether any return was joined from a line-of-variations seed (for
        # the provenance note).
        self.used_line_of_variations = False

    def _request_historical_force(self) -> None:
        """Mirror requested A1–A3 as A1H–A3H valid before the longest gap.

        The switch (TDB JD) is the middle of the longest gap between observed
        returns: a lost comet (205P: 1896 → 2008) is far from the Sun there,
        where the water-ice law is negligible, so the force law changes
        without a noticeable jump.  The historical side starts with the
        modern values (``current_force``) and is released only once one of
        its returns takes part in a correction.
        """
        modern = [name for name in HISTORICAL_NONGRAVITATIONAL if name in self.requested]
        if not modern or len(self.groups) < 2:
            return
        spans = sorted((self._start(index), self._end(index)) for index in range(len(self.groups)))
        gaps = [
            (later[0] - earlier[1], 0.5 * (earlier[1] + later[0]))
            for earlier, later in zip(spans, spans[1:])
        ]
        if not gaps:
            return
        _gap, self.split_jd = max(gaps)
        for name in modern:
            self.requested[HISTORICAL_NONGRAVITATIONAL[name]] = float(self.requested[name])
        self.requested[NONGRAVITATIONAL_SPLIT_KEY] = float(self.split_jd)

    def historical(self, index: int) -> bool:
        """Whether a return lies before the force-law switch."""
        return self.split_jd is not None and self._end(index) < self.split_jd

    def free_for(self, indices: list[int], names: tuple[str, ...]) -> tuple[str, ...]:
        """``names`` without historical coefficients that ``indices`` cannot constrain."""
        historical_names = set(HISTORICAL_NONGRAVITATIONAL.values())
        if any(self.historical(index) for index in indices):
            return names
        return tuple(name for name in names if name not in historical_names)

    def historical_free(self, indices: list[int]) -> tuple[str, ...]:
        """A1H/A2H to release when a historical return takes part in a stage."""
        if not any(self.historical(index) for index in indices):
            return ()
        return tuple(
            HISTORICAL_NONGRAVITATIONAL[name] for name in ("A1", "A2")
            if HISTORICAL_NONGRAVITATIONAL[name] in self.free_names
        )

    # -- helpers ----------------------------------------------------------
    def clear_residuals(self) -> None:
        """Forget O-C from earlier trial orbits.

        A staged correction fits a representative sample and refreshes O-C
        only on those rows.  Quality gates and blunder rejection read every
        row that carries O-C, so values left behind by another orbit (for
        example the appraisal of the catalogue seed) would be mixed in and
        could make a correctly joined return look tens of arcseconds off.
        """
        for observation in self.observations:
            observation.residual_ra_arcsec = None
            observation.residual_dec_arcsec = None
            observation.used_in_fit = False

    def fit_once(self, *args, **kwargs) -> tuple[OrbitSolution, float, int]:
        """One correction; afterwards only its own rows carry O-C."""
        self.clear_residuals()
        return self._fit_once(*args, **kwargs)

    def check_cancelled(self) -> None:
        if self.cancel_callback is not None and self.cancel_callback():
            raise OrbitFitCancelled(tr("Výpočet dráhy byl zrušen uživatelem."))

    def label(self, index: int) -> str:
        record = self.records[index]
        start, end = record.get("year_start"), record.get("year_end")
        return str(start) if start == end else f"{start}–{end}"

    def stage_rows(self, indices: list[int], per_group: int) -> list[Observation]:
        rows: list[Observation] = []
        for index in sorted(indices):
            rows.extend(_sample(_candidates(self.groups[index]), per_group))
        return rows

    def group_quality(self, index: int) -> tuple[float, float, int]:
        """(normalized RMS, physical RMS, fitted rows) on rows with O-C."""
        normalized: list[float] = []
        physical: list[float] = []
        for observation in self.groups[index]:
            if (
                not is_fit_candidate(observation)
                or observation.residual_ra_arcsec is None
                or observation.residual_dec_arcsec is None
            ):
                continue
            sigma_ra, sigma_dec = observation_sigma_arcsec(observation)
            normalized.extend((
                observation.residual_ra_arcsec / sigma_ra,
                observation.residual_dec_arcsec / sigma_dec,
            ))
            physical.extend((observation.residual_ra_arcsec, observation.residual_dec_arcsec))
        if not normalized:
            return float("inf"), float("inf"), 0
        return (
            float(np.sqrt(np.mean(np.square(normalized)))),
            float(np.sqrt(np.mean(np.square(physical)))),
            len(normalized) // 2,
        )

    def reject_outliers(self, indices: list[int]) -> int:
        """Exclude clear blunders within the bounds documented above."""
        rejected = 0
        for index in indices:
            enabled = [item for item in self.groups[index] if is_fit_candidate(item)]
            scored: list[tuple[float, Observation]] = []
            radial: list[float] = []
            for observation in enabled:
                if observation.residual_ra_arcsec is None or observation.residual_dec_arcsec is None:
                    continue
                sigma_ra, sigma_dec = observation_sigma_arcsec(observation)
                normalized = float(np.hypot(
                    observation.residual_ra_arcsec / sigma_ra,
                    observation.residual_dec_arcsec / sigma_dec,
                ))
                radial.append(normalized)
                physical = float(np.hypot(
                    observation.residual_ra_arcsec, observation.residual_dec_arcsec,
                ))
                if (
                    normalized > OUTLIER_NORMALIZED_SIGMA
                    and physical > max(OUTLIER_MINIMUM_ARCSEC, 3.0 * max(sigma_ra, sigma_dec))
                ):
                    scored.append((normalized, observation))
            if not radial or float(np.median(radial)) > REJECTION_MEDIAN_SIGMA:
                continue
            # A coherent offset series beside a well-fitted one (a fragment or
            # secondary nucleus reported under the same designation) is not a
            # set of independent blunders; separate it explicitly.
            for observation in _coherent_secondary_track_candidates(enabled):
                observation.excluded_from_fit = True
                observation.raw["automatic_exclusion"] = True
                observation.raw["coherent_secondary_track"] = True
                observation.raw["automatic_exclusion_reason"] = (
                    tr(
                        "Souvislá vedlejší astrometrická stopa; možný fragment nebo sekundární "
                        "jádro.",
                    )
                )
                self.rejections.setdefault(index, []).append(observation)
                rejected += 1
            scored = [item for item in scored if not item[1].excluded_from_fit]
            enabled = [item for item in enabled if not item.excluded_from_fit]
            # A secondary track is another object, not blunders of this one:
            # it neither counts as the object's astrometry nor uses up the
            # bounded rejection budget (205P in 2021: ten ZTF rows of a second
            # track would otherwise block rejecting a 66″-off night).
            blunders = [
                item for item in self.rejections.get(index, [])
                if not item.raw.get("coherent_secondary_track")
            ]
            total = len(enabled) + len(blunders)
            fraction = (
                MAXIMUM_REJECTED_FRACTION_DENSE if total >= DENSE_APPARITION_ROWS
                else MAXIMUM_REJECTED_FRACTION_SPARSE
            )
            allowed = int(np.floor(fraction * total)) - len(blunders)
            allowed = min(allowed, len(enabled) - min(3, len(enabled)))
            for _score, observation in sorted(scored, key=lambda item: -item[0])[:max(0, allowed)]:
                observation.excluded_from_fit = True
                observation.raw["automatic_exclusion"] = True
                observation.raw["automatic_exclusion_reason"] = (
                    tr(
                        "Odlehlé měření při společném fitu návratů (> {0:g}σ od navázané dráhy).",
                        OUTLIER_NORMALIZED_SIGMA,
                    )
                )
                self.rejections.setdefault(index, []).append(observation)
                rejected += 1
        return rejected

    def checkpoint(self) -> dict[int, int]:
        return {index: len(items) for index, items in self.rejections.items()}

    def rollback(self, checkpoint: dict[int, int]) -> None:
        """Undo rejections made after ``checkpoint`` (a failed attempt)."""
        for index in list(self.rejections):
            keep = checkpoint.get(index, 0)
            for observation in self.rejections[index][keep:]:
                observation.excluded_from_fit = False
                for key in (
                    "automatic_exclusion", "automatic_exclusion_reason",
                    "coherent_secondary_track",
                ):
                    observation.raw.pop(key, None)
            del self.rejections[index][keep:]
            if not self.rejections[index]:
                del self.rejections[index]

    def readmit(self, solution: OrbitSolution) -> int:
        """Re-include automatic rejections that fit the final orbit.

        A row rejected while an early stage was still far from convergence
        may be perfectly good; only rows still beyond the outlier limit on
        the final orbit stay excluded.  Coherent secondary tracks are kept.
        """
        rows = [
            item for items in self.rejections.values() for item in items
            if not item.raw.get("coherent_secondary_track")
        ]
        if not rows:
            return 0
        try:
            self.evaluate(rows, solution, self.current_force(solution), self.cancel_callback)
        except OrbitFitCancelled:
            raise
        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError, OverflowError):
            return 0
        readmitted = 0
        for index in list(self.rejections):
            kept: list[Observation] = []
            for observation in self.rejections[index]:
                if observation.residual_ra_arcsec is None or observation.raw.get(
                    "coherent_secondary_track"
                ):
                    kept.append(observation)
                    continue
                sigma_ra, sigma_dec = observation_sigma_arcsec(observation)
                normalized = float(np.hypot(
                    observation.residual_ra_arcsec / sigma_ra,
                    observation.residual_dec_arcsec / sigma_dec,
                ))
                if normalized <= OUTLIER_NORMALIZED_SIGMA:
                    observation.excluded_from_fit = False
                    observation.raw.pop("automatic_exclusion", None)
                    observation.raw.pop("automatic_exclusion_reason", None)
                    readmitted += 1
                else:
                    kept.append(observation)
            if kept:
                self.rejections[index] = kept
            else:
                del self.rejections[index]
        return readmitted

    def restore_rejections(self, index: int) -> None:
        for observation in self.rejections.pop(index, []):
            observation.excluded_from_fit = False
            for key in (
                "automatic_exclusion", "automatic_exclusion_reason", "coherent_secondary_track",
            ):
                observation.raw.pop(key, None)

    def corrected(
        self, indices: list[int], seed: OrbitSolution, free: tuple[str, ...] = (),
        per_group: int = STAGE_ROWS_PER_APPARITION, max_nfev: int = STAGE_MAX_NFEV,
    ) -> tuple[OrbitSolution, float, int]:
        """Least-squares pass, bounded blunder rejection, then robust passes."""
        fitted = {name: self.current_force(seed).get(name, 0.0) for name in free}
        fixed = {
            name: value for name, value in self.current_force(seed).items()
            if name not in fitted
        }

        def run(start: OrbitSolution, robust: bool) -> tuple[OrbitSolution, float, int]:
            return self.fit_once(
                self.stage_rows(indices, per_group), start, self.constraints, None,
                fitted or None, fixed_nongravitational=fixed,
                maximum_observations=None, max_nfev=max_nfev,
                robust_loss=robust, robust_scale=3.0,
            )

        def update_fitted(solution: OrbitSolution) -> None:
            fitted.update({
                name: value for name, value in (solution.nongravitational_parameters or {}).items()
                if name in fitted
            })

        # Ordinary least squares first: with a robust loss a whole apparition
        # that is still tens of sigma away acts as one coherent "outlier" whose
        # pull is suppressed, and the correction stalls beside the solution.
        trial = run(seed, robust=False)
        update_fitted(trial[0])
        trial = run(trial[0], robust=True)
        if self.reject_outliers(indices):
            update_fitted(trial[0])
            trial = run(trial[0], robust=True)
        return trial

    def stage_free(self, indices: list[int], solution: OrbitSolution) -> tuple[str, ...]:
        """Requested A1/A2 that the stage's time span can already determine."""
        names = tuple(name for name in ("A1", "A2") if name in self.free_names)
        if not names or not (0.0 <= solution.eccentricity < 1.0):
            return ()
        period = 2.0 * np.pi * solution.semi_major_axis_au ** 1.5 / np.sqrt(MU_SUN)
        span = max(self._end(index) for index in indices) - min(
            self._start(index) for index in indices
        )
        return names if span >= FORCE_RELEASE_PERIODS * period else ()

    def current_force(self, solution: OrbitSolution | None) -> dict[str, float]:
        force = dict(self.fixed_force)
        reached = (solution.nongravitational_parameters or {}) if solution is not None else {}
        force.update({
            name: float(value) for name, value in reached.items()
            if name in force or name in ("R_EXPONENT", NONGRAVITATIONAL_SPLIT_KEY)
        })
        # Until a historical return has been fitted, the historical
        # coefficients continue the modern ones (one force law).
        for modern, historical in HISTORICAL_NONGRAVITATIONAL.items():
            if historical in force and historical not in reached and modern in force:
                force[historical] = force[modern]
        return force

    def accepted(self, indices: list[int], new: list[int] | None = None) -> bool:
        """Quality gate: ``new`` apparitions ≤ the linkage limit, others ≤ tension."""
        fresh = set(indices if new is None else new)
        for index in indices:
            normalized, _physical, count = self.group_quality(index)
            limit = (
                LINKAGE_RELAXED_NORMALIZED_RMS if index in self.tension
                else LINKAGE_ACCEPT_NORMALIZED_RMS if index in fresh
                else LINKAGE_TENSION_NORMALIZED_RMS
            )
            if count < 1 or normalized > limit:
                return False
        return True

    def quality_text(self, indices: list[int]) -> str:
        parts = []
        for index in indices:
            normalized, physical, _ = self.group_quality(index)
            parts.append(f"{self.label(index)} {physical:.2f}″/{normalized:.2f}σ")
        return ", ".join(parts)

    # -- stages -----------------------------------------------------------
    def catalogue_shortcut(self) -> bool:
        """Whether the seed already predicts every apparition closely enough."""
        initial = self.initial
        if initial is None or initial.position_au is None or initial.velocity_au_per_day is None:
            return False
        self.report_progress(tr("Ověřuji výchozí dráhu na všech návratech…"))
        try:
            # Only the requested force law counts: catalogue A1/A2 that the
            # user did not ask for must not leak into a gravity-only appraisal.
            self.evaluate(
                self.observations, _with_state(initial, np.concatenate((
                    initial.position_au, initial.velocity_au_per_day,
                )), nongravitational=None),
                self.fixed_force, self.cancel_callback,
            )
        except OrbitFitCancelled:
            raise
        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError, OverflowError):
            return False
        for group in self.groups:
            rows = [
                item for item in group
                if is_fit_candidate(item) and item.residual_ra_arcsec is not None
            ]
            if not rows:
                return False
            misses = []
            for item in rows:
                sigma_ra, sigma_dec = observation_sigma_arcsec(item)
                misses.append(np.hypot(
                    item.residual_ra_arcsec / sigma_ra, item.residual_dec_arcsec / sigma_dec,
                ))
            if float(np.median(misses)) > DIRECT_CORRECTION_MEDIAN_SIGMA:
                return False
        return True

    def fit_anchor(self, index: int) -> tuple[OrbitSolution, float, int]:
        trial = self._fit_anchor_gravity(index)
        free = self.stage_free([index], trial[0]) + self.historical_free([index])
        if not free:
            return trial
        self.report_progress(
            tr(
                "Návrat {0}: oblouk pokrývá celý oběh, uvolňuji {1}…",
                self.label(index),
                '+'.join(free),
            )
        )
        report = _snapshot(self.observations)
        try:
            released = self.corrected([index], trial[0], free=free, max_nfev=20)
            if self.accepted([index]):
                return released
        except OrbitFitCancelled:
            raise
        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
            pass
        _restore(self.observations, report)
        return trial

    def _fit_anchor_gravity(self, index: int) -> tuple[OrbitSolution, float, int]:
        label = self.label(index)
        failures: list[str] = []
        if self.initial is not None and self.initial.position_au is not None:
            self.report_progress(tr(
                "Návrat {0}: koriguji výchozí dráhu na jeho astrometrii…",
                label,
            ))
            try:
                trial = self.corrected([index], self.initial, max_nfev=30)
                if self.accepted([index]):
                    return trial
                failures.append(tr("výchozí dráha: {0}", self.quality_text([index])))
            except OrbitFitCancelled:
                raise
            except (OrbitFitError, ValueError, RuntimeError, FloatingPointError) as exc:
                failures.append(tr("výchozí dráha: {0}", exc))
            self.restore_rejections(index)
        self.report_progress(tr("Návrat {0}: určuji samostatnou dráhu z jeho úhlů…", label))
        try:
            trial = self.independent_orbit(index)
            if self.accepted([index]):
                return trial
            failures.append(tr("samostatná dráha: {0}", self.quality_text([index])))
        except OrbitFitCancelled:
            raise
        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError) as exc:
            failures.append(tr("samostatná dráha: {0}", exc))
        self.restore_rejections(index)
        raise OrbitFitError(
            tr(
                "Výchozí návrat {0} nelze spolehlivě napasovat ({1}).",
                label,
                '; '.join(failures),
            )
        )

    def independent_orbit(self, index: int) -> tuple[OrbitSolution, float, int]:
        """Orbit of one apparition from its own angles.

        Angles-only ranging is reliable on a few well-observed nights but can
        fail on a months-long, strongly curved arc (206P passed 0.19 AU from
        the Earth in 2008) or on a handful of faint early detections (78P in
        2011).  Start in the densest short window, then grow the arc in both
        directions, halving the growth whenever a step leaves the astrometry,
        as a human orbit computer would.
        """
        rows = _candidates(self.groups[index])
        days = np.asarray([
            (parse_observation_time(item.time) - parse_observation_time(rows[0].time))
            .total_seconds() / 86400.0
            for item in rows
        ])
        failures: list[str] = []
        current: OrbitSolution | None = None
        low = high = 0.0
        for window_days in INITIAL_WINDOWS_DAYS:
            for start_day in _densest_windows(rows, days, window_days):
                window = [
                    item for item, day in zip(rows, days)
                    if start_day <= day <= start_day + window_days
                ]
                try:
                    candidate = self.preliminary_orbit(window)
                except OrbitFitError as exc:
                    failures.append(tr("okno {0:g} d: {1}", window_days, exc))
                    continue
                if (candidate.fit_normalized_rms or np.inf) <= EXTENSION_ACCEPT_NORMALIZED_RMS:
                    current = candidate
                    low, high = start_day, start_day + window_days
                    break
                failures.append(
                    tr(
                        "okno {0:g} d: {1:.1f}σ",
                        window_days,
                        candidate.fit_normalized_rms or np.inf,
                    )
                )
            if current is not None:
                break
        if current is None:
            raise OrbitFitError(tr("předběžná dráha z krátkého oblouku: ") + "; ".join(failures[-3:]))
        total = float(days[-1])
        growth = 2.0
        while low > 0.0 or high < total:
            self.check_cancelled()
            centre, half = (low + high) / 2.0, max(1.0, (high - low) / 2.0) * growth
            new_low, new_high = max(0.0, centre - half), min(total, centre + half)
            window = [item for item, day in zip(rows, days) if new_low <= day <= new_high]
            try:
                trial = self.fit_once(
                    window, current, self.constraints, None, None,
                    fixed_nongravitational=self.fixed_force,
                    maximum_observations=STAGE_ROWS_PER_APPARITION, max_nfev=30,
                    robust_loss=True, robust_scale=3.0,
                )[0]
                good = (trial.fit_normalized_rms or np.inf) <= EXTENSION_ACCEPT_NORMALIZED_RMS
            except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
                good = False
            if good:
                current, low, high = trial, new_low, new_high
                growth = min(2.0, growth * 1.5)
                continue
            growth = 1.0 + (growth - 1.0) / 2.0
            if growth < 1.05:
                raise OrbitFitError(
                    tr(
                        "rozšíření oblouku se zastavilo na {0:.0f} z {1:.0f} dní",
                        high - low,
                        total,
                    )
                )
        return self.corrected([index], current, max_nfev=30)

    def preliminary_orbit(self, rows: list[Observation]) -> OrbitSolution:
        """Best range-search hypothesis, ranked by residuals and plausibility."""
        places = normal_place_observations(rows)
        candidates = initial_orbit_from_observations(
            places if len(places) >= 6 else rows, return_candidates=True,
        )
        assert isinstance(candidates, list)
        best: tuple[float, OrbitSolution] | None = None
        for candidate in candidates[:8]:
            self.check_cancelled()
            try:
                trial = self.fit_once(
                    rows, candidate, self.constraints, None, None,
                    fixed_nongravitational=self.fixed_force,
                    maximum_observations=STAGE_ROWS_PER_APPARITION, max_nfev=30,
                    robust_loss=True, robust_scale=3.0,
                )[0]
            except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
                continue
            score = _automatic_orbit_score(trial)
            if np.isfinite(score) and (best is None or score < best[0]):
                best = (score, trial)
            # Range candidates frequently converge to the same minimum; one
            # bound orbit fitting the window well ends the search.
            if (
                trial.eccentricity < 1.0
                and (trial.fit_normalized_rms or np.inf) <= LINKAGE_ACCEPT_NORMALIZED_RMS
            ):
                break
        if best is None:
            raise OrbitFitError(tr("Z úhlů krátkého oblouku nelze sestavit předběžnou dráhu."))
        return best[1]

    def well_predicted(self, current: OrbitSolution, indices: list[int]) -> list[int]:
        """Remaining apparitions that the current orbit already predicts well.

        One integration over all of them; joining several such apparitions in
        a single correction saves a full multi-decade correction per return.
        """
        targets = [
            (index, _sky_target(self.groups[index], SCAN_ROWS_PER_APPARITION))
            for index in indices
        ]
        combined = _SkyTarget(*(
            np.concatenate([getattr(target, name) for _index, target in targets])
            for name in ("epochs", "offsets", "earth", "ra", "dec", "sigmas")
        ))
        state = np.concatenate((
            np.asarray(current.position_au, dtype=float),
            np.asarray(current.velocity_au_per_day, dtype=float),
        ))
        self.report_progress(tr("Předpovídám zbývající návraty z navázané dráhy…"))
        try:
            normalized = _normalized_prediction(
                _scan_engine(current.epoch_jd, combined, self.current_force(current)),
                current.epoch_jd, state, combined,
            )
        except (RuntimeError, ValueError, FloatingPointError, OverflowError):
            return []
        selected: list[int] = []
        offset = 0
        for index, target in targets:
            count = target.epochs.size
            if _robust_miss(normalized[offset:offset + count]) <= BATCH_MEDIAN_SIGMA:
                selected.append(index)
            offset += count
        return selected

    def link_batch(
        self, linked: list[int], current: OrbitSolution, batch: list[int],
    ) -> tuple[OrbitSolution, float, int] | None:
        stage = sorted(linked + batch)
        labels = ", ".join(self.label(index) for index in sorted(batch))
        historical = self.historical_free(stage)
        attempts = [self.stage_free(stage, current) + historical]
        requested = tuple(item for item in ("A1", "A2") if item in self.free_names) + historical
        if requested and requested != attempts[0]:
            attempts.append(requested)
        trial = None
        checkpoint = self.checkpoint()
        for free in attempts:
            self.report_progress(tr("Připojuji předpovězené návraty {0} najednou…", labels))
            try:
                trial = self.corrected(stage, trial[0] if trial else current, free=free)
            except OrbitFitCancelled:
                raise
            except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
                trial = None
                continue
            if self.accepted(stage, new=batch):
                return trial
        self.rollback(checkpoint)
        return None

    def link_next(
        self, linked: list[int], current: OrbitSolution, index: int,
    ) -> tuple[OrbitSolution, float, int]:
        label = self.label(index)
        stage = sorted(linked + [index])
        target = _sky_target(self.groups[index], SCAN_ROWS_PER_APPARITION)
        force = self.current_force(current)
        state = np.concatenate((
            np.asarray(current.position_au, dtype=float),
            np.asarray(current.velocity_au_per_day, dtype=float),
        ))
        self.report_progress(tr("Návrat {0}: předpovídám polohy z navázané dráhy…", label))
        try:
            predicted_miss = _robust_miss(_normalized_prediction(
                _scan_engine(current.epoch_jd, target, force), current.epoch_jd, state, target,
            ))
        except (RuntimeError, ValueError, FloatingPointError, OverflowError) as exc:
            raise LinkageFailed(tr("propagace na návrat selhala: {0}", exc)) from exc
        failures = [tr("přímá předpověď: medián {0:.1f}σ", predicted_miss)]
        best: tuple[float, OrbitSolution, tuple[str, ...], bool] | None = None
        seeds: list[tuple[str, OrbitSolution, bool]] = []
        if predicted_miss <= DIRECT_CORRECTION_MEDIAN_SIGMA:
            seeds.append((tr("přímá korekce"), current, False))
        scanned = False

        def scan() -> None:
            self.report_progress(tr("Návrat {0}: hledám větev podél linie variací…", label))
            candidates = _scan_line_of_variations(current, target, force, self.check_cancelled)
            for score, k, vector in candidates:
                failures.append(tr("LOV {0:+.2f}σ: medián {1:.1f}σ", k, score))
                seeds.append((f"LOV {k:+.2f}σ", _with_state(current, vector), True))

        while True:
            if not seeds:
                if scanned:
                    break
                scanned = True
                scan()
                continue
            name, seed, from_line_of_variations = seeds.pop(0)
            # A stage spanning a full revolution determines the requested
            # A1/A2 itself; otherwise the force law stays fixed first and
            # A1/A2 are released only if the fixed law cannot join the return.
            historical = self.historical_free(stage)
            stage_free = self.stage_free(stage, seed) + historical
            attempts = [stage_free]
            requested = tuple(item for item in ("A1", "A2") if item in self.free_names) + historical
            if requested and requested != stage_free:
                attempts.append(requested)
            trial = None
            checkpoint = self.checkpoint()
            for free in attempts:
                label_free = f"{name} + {'+'.join(free)}" if free else name
                self.report_progress(tr(
                    "Návrat {0}: společná korekce ({1})…",
                    label,
                    label_free,
                ))
                try:
                    trial = self.corrected(stage, trial[0] if trial else seed, free=free)
                except OrbitFitCancelled:
                    raise
                except (OrbitFitError, ValueError, RuntimeError, FloatingPointError) as exc:
                    failures.append(f"{label_free}: {exc}")
                    trial = None
                    continue
                if self.accepted(stage, new=[index]):
                    self.used_line_of_variations |= from_line_of_variations
                    return trial
                failures.append(f"{label_free}: {self.quality_text(stage)}")
                own = self.group_quality(index)[0]
                if self.accept_model_tension and self._within_relaxed_gate(stage, index) and (
                    best is None or own < best[0]
                ):
                    best = (own, seed, free, from_line_of_variations)
            self.rollback(checkpoint)
        if best is not None:
            relaxed = self._link_with_tension(stage, index, best)
            if relaxed is not None:
                return relaxed
            failures.append(tr("ani s napětím modelu: {0}", self.quality_text(stage)))
        raise LinkageFailed("; ".join(failures[-4:]))

    def _within_relaxed_gate(self, stage: list[int], index: int) -> bool:
        """New return ≤ relaxed limit, every other one within the tension limit."""
        for item in stage:
            normalized, _physical, count = self.group_quality(item)
            limit = (
                LINKAGE_RELAXED_NORMALIZED_RMS if item == index or item in self.tension
                else LINKAGE_TENSION_NORMALIZED_RMS
            )
            if count < 1 or normalized > limit:
                return False
        return True

    def _link_with_tension(
        self, stage: list[int], index: int,
        best: tuple[float, OrbitSolution, tuple[str, ...], bool],
    ) -> tuple[OrbitSolution, float, int] | None:
        """Repeat the best failed correction and join it through the relaxed gate."""
        _own, seed, free, from_line_of_variations = best
        self.report_progress(tr(
            "Návrat {0}: připojuji s napětím modelu (do {1:g}σ)…",
            self.label(index), LINKAGE_RELAXED_NORMALIZED_RMS,
        ))
        checkpoint = self.checkpoint()
        try:
            trial = self.corrected(stage, seed, free=free)
        except OrbitFitCancelled:
            raise
        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
            self.rollback(checkpoint)
            return None
        if not self._within_relaxed_gate(stage, index):
            self.rollback(checkpoint)
            return None
        self.tension.add(index)
        self.used_line_of_variations |= from_line_of_variations
        return trial

    def final_fit(self, linked: list[int], seed: OrbitSolution) -> tuple[OrbitSolution, float, int]:
        for observation in self.observations:
            observation.residual_ra_arcsec = None
            observation.residual_dec_arcsec = None
            observation.used_in_fit = False
        rows_per_group = max(
            STAGE_ROWS_PER_APPARITION, FINAL_ROW_LIMIT // max(1, len(linked)),
        )
        self.report_progress(tr("Závěrečná přesná korekce všech navázaných návratů…"))
        result, free = self._final_correction(linked, seed, rows_per_group)
        if self.readmit(result[0]):
            self.report_progress(tr("Vracím do fitu měření, která na výslednou dráhu sedí…"))
            for observation in self.observations:
                observation.residual_ra_arcsec = observation.residual_dec_arcsec = None
                observation.used_in_fit = False
            note = result[0].fit_model_note
            result = self.corrected(
                linked, result[0], free=free, per_group=rows_per_group, max_nfev=8,
            )
            result[0].fit_model_note = note
        return result

    def _final_correction(
        self, linked: list[int], seed: OrbitSolution, rows_per_group: int,
    ) -> tuple[tuple[OrbitSolution, float, int], tuple[str, ...]]:
        """Final correction and the force parameters that were left free."""
        free_names = self.free_for(linked, self.free_names)
        if not free_names:
            return self.corrected(linked, seed, per_group=rows_per_group, max_nfev=12), ()
        # The linked branch already carries A1/A2 where the stages could
        # determine them; release every requested parameter directly.
        checkpoint = self.checkpoint()
        report = _snapshot(self.observations)
        try:
            released = self.corrected(
                linked, seed, free=free_names, per_group=rows_per_group, max_nfev=16,
            )
            if self.accepted(linked, new=[]):
                return released, free_names
        except OrbitFitCancelled:
            raise
        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError):
            pass
        self.rollback(checkpoint)
        _restore(self.observations, report)
        fixed = self.corrected(linked, seed, per_group=rows_per_group, max_nfev=12)
        fixed[0].fit_model_note = (
            tr(
                "Uvolnění všech požadovaných negravitačních parametrů některý návrat "
                "zhoršilo; ponechán silový model z navazování.",
            )
        )
        return fixed, ()

    def run(self) -> tuple[tuple[OrbitSolution, float, int], list[dict[str, object]], str]:
        all_indices = list(range(len(self.groups)))
        result: tuple[OrbitSolution, float, int] | None = None
        failed: dict[int, str] = {}
        failed_with: dict[int, tuple[int, ...]] = {}
        retried: set[int] = set()
        if self.catalogue_shortcut():
            self.report_progress(tr(
                "Výchozí dráha předpovídá všechny návraty; koriguji je společně…",
            ))
            try:
                result = self.final_fit(all_indices, self.initial)
            except OrbitFitCancelled:
                raise
            except (OrbitFitError, ValueError, RuntimeError, FloatingPointError, OverflowError):
                result = None
            if result is not None and not self.accepted(all_indices, new=[]):
                result = None
            if result is None:
                for index in all_indices:
                    self.restore_rejections(index)
        if result is not None:
            linked = all_indices
            source = tr("Výchozí dráha předpověděla všechny návraty; byly korigovány společně.")
        else:
            anchor = _anchor_index(self.groups)
            self.records[anchor]["status"] = "seed_only"
            current = self.fit_anchor(anchor)[0]
            linked = [anchor]
            remaining = [index for index in all_indices if index != anchor]
            while True:
                while remaining:
                    first = min(self._start(index) for index in linked)
                    last = max(self._end(index) for index in linked)
                    remaining.sort(key=lambda index: min(
                        abs(self._end(index) - first), abs(self._start(index) - last),
                    ))
                    batch = self.well_predicted(current, remaining)
                    if len(batch) > 1:
                        trial = self.link_batch(linked, current, batch)
                        if trial is not None:
                            linked = sorted(linked + batch)
                            current = trial[0]
                            remaining = [index for index in remaining if index not in batch]
                            continue
                    index = remaining.pop(0)
                    try:
                        trial = self.link_next(linked, current, index)
                    except LinkageFailed as exc:
                        failed[index] = str(exc)
                        failed_with[index] = tuple(linked)
                        self.restore_rejections(index)
                        continue
                    linked = sorted(linked + [index])
                    current = trial[0]
                # A return that failed while fewer returns were linked gets one
                # more attempt: it may now lie between two linked returns and
                # be far better predicted (1P: 1910 between 1835 and 1986).
                remaining = [
                    index for index in sorted(failed)
                    if failed_with.get(index) != tuple(linked) and index not in retried
                ]
                if not remaining:
                    break
                retried.update(remaining)
                for index in remaining:
                    del failed[index]
            result = self.final_fit(linked, current)
            source = (
                tr(
                    "Návraty byly navázány postupně od nejlépe pozorovaného ({0}) hledáním "
                    "větve podél linie variací.",
                    self.label(anchor),
                ) if self.used_line_of_variations else tr(
                    "Návraty byly navázány postupně od nejlépe pozorovaného ({0}); každý "
                    "další byl připojen přímou korekcí navázané dráhy.",
                    self.label(anchor),
                )
            )
        if self.split_jd is not None and not any(self.historical(index) for index in linked):
            # No historical return was joined: the historical coefficients
            # were never constrained, so the result keeps one force law.
            parameters = dict(result[0].nongravitational_parameters or {})
            for name in (*HISTORICAL_NONGRAVITATIONAL.values(), NONGRAVITATIONAL_SPLIT_KEY):
                parameters.pop(name, None)
            result[0].nongravitational_parameters = parameters or None
        # Rows without O-C from the final correction (unlinked apparitions,
        # duplicate reductions, excluded or unsampled rows) are evaluated with
        # the returned orbit; earlier trial values must never remain.
        self._report_unfitted(result[0], [
            item for item in self.observations
            if item.residual_ra_arcsec is None and item.ra_deg is not None
        ])
        for index in list(linked):
            if not self.accepted([index], new=[]):
                failed[index] = (
                    tr("Závěrečná korekce návrat nepotvrdila: ") + self.quality_text([index])
                )
                linked.remove(index)
        strained = [
            index for index in linked
            if self.group_quality(index)[0] > LINKAGE_ACCEPT_NORMALIZED_RMS
        ]
        for index in all_indices:
            record = self.records[index]
            if index in self.tension and index in linked:
                record["status"] = "linked"
                record["reason"] = (
                    tr(
                        "Připojeno nad běžným limitem (volba napětí modelu, do {0:g}σ): {1}; "
                        "historická astrometrie nebo konstantní silový model návrat "
                        "nevystihují přesně.",
                        LINKAGE_RELAXED_NORMALIZED_RMS,
                        self.quality_text([index]),
                    )
                )
            elif index in strained:
                record["status"] = "linked"
                record["reason"] = (
                    tr(
                        "Společný N-body fit, ale s napětím modelu: návrat je jako celek posunut "
                        "({0}); konstantní silový model patrně nevystihuje proměnlivou aktivitu "
                        "komety.",
                        self.quality_text([index]),
                    )
                )
            elif index in linked:
                record["status"] = "linked"
                record["reason"] = (
                    tr(
                        "Společný N-body fit; návrat sedí na vlastní astrometrii (≤ {0:g}σ).",
                        LINKAGE_ACCEPT_NORMALIZED_RMS,
                    )
                )
            else:
                record["status"] = "failed"
                record["reason"] = failed.get(index, tr("Návrat nebyl navázán."))
            _update_apparition_linkage_metrics(record, self.groups[index])
        rejected = sum(len(items) for items in self.rejections.values())
        note_parts = [source]
        if rejected:
            note_parts.append(
                tr(
                    "{0} odlehlých měření v navázaných návratech bylo automaticky vyřazeno.",
                    rejected,
                )
            )
        if strained:
            note_parts.append(
                tr("Napětí modelu (nad {0:g}σ) v návratech ", LINKAGE_ACCEPT_NORMALIZED_RMS)
                + ", ".join(self.label(index) for index in strained) + "."
            )
        if failed:
            note_parts.append(
                tr(
                    "Navázáno {0} z {1} návratů; ostatní jsou jen předpovědí.",
                    len(linked),
                    len(self.groups),
                )
            )
        return result, self.records, " ".join(note_parts)

    def _start(self, index: int) -> float:
        rows = _candidates(self.groups[index])
        return float(dynamical_jd([parse_observation_time(rows[0].time)])[0]) if rows else 0.0

    def _end(self, index: int) -> float:
        rows = _candidates(self.groups[index])
        return float(dynamical_jd([parse_observation_time(rows[-1].time)])[0]) if rows else 0.0

    def _report_unfitted(self, solution: OrbitSolution, rows: list[Observation]) -> None:
        if not rows:
            return
        try:
            self.evaluate(rows, solution, self.current_force(solution), self.cancel_callback)
        except OrbitFitCancelled:
            raise
        except (OrbitFitError, ValueError, RuntimeError, FloatingPointError, OverflowError):
            for item in rows:
                item.residual_ra_arcsec = item.residual_dec_arcsec = None


_KEEP = object()


def _with_state(
    solution: OrbitSolution, vector: np.ndarray, nongravitational=_KEEP,
) -> OrbitSolution:
    """Copy of ``solution`` with another Cartesian state (same epoch)."""
    changes = {
        "position_au": tuple(float(value) for value in vector[:3]),
        "velocity_au_per_day": tuple(float(value) for value in vector[3:6]),
        "state_covariance": None, "element_uncertainties": None,
    }
    if nongravitational is not _KEEP:
        changes["nongravitational_parameters"] = nongravitational
    return replace(solution, **changes)


def _snapshot(observations: list[Observation]) -> list[tuple]:
    return [
        (item.residual_ra_arcsec, item.residual_dec_arcsec, item.used_in_fit)
        for item in observations
    ]


def _restore(observations: list[Observation], report: list[tuple]) -> None:
    for observation, values in zip(observations, report):
        observation.residual_ra_arcsec, observation.residual_dec_arcsec, observation.used_in_fit = values


def link_apparitions(
    observations: list[Observation], groups: list[list[Observation]],
    initial: OrbitSolution | None, *,
    constraints: dict[str, float] | None,
    nongravitational: dict[str, float] | None,
    fit_once: FitOnce, evaluate: Evaluate,
    report_progress: ProgressReporter, cancel_callback=None,
    historical_nongravitational: bool = False, accept_model_tension: bool = False,
) -> tuple[tuple[OrbitSolution, float, int], list[dict[str, object]], str]:
    """Link several observed apparitions; return (fit result, audit records, note).

    ``fit_once`` and ``evaluate`` are :func:`orbit_fit._fit_orbit_once` and
    :func:`orbit_fit._evaluate_solution_observations` (injected so that the
    caller's cancellation callback and test doubles apply).
    """
    linker = _Linker(
        observations, groups, initial, constraints, nongravitational,
        fit_once, evaluate, report_progress, cancel_callback,
        historical_nongravitational=historical_nongravitational,
        accept_model_tension=accept_model_tension,
    )
    return linker.run()
