"""Split comets beyond component selection.

* :func:`component_summaries` – what the astrometry contains (observed span,
  nights, returns, typical sky offset from the principal component).
* :func:`assign_unlabelled_rows` – decide, against the fitted orbits of the
  components, which track an unlabelled measurement made after the split
  actually follows (surveys often report a fragment without its letter).
* :func:`estimate_split` – epoch at which a fragment and the principal
  component were closest, from a backward N-body integration of both orbits.

Units: angles in arcseconds, distances in AU (km where named), velocities in
m/s, epochs as TDB Julian dates unless a field says UTC.
"""
from __future__ import annotations

import time
from bisect import bisect_left
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Callable

import numpy as np

from .astrometry import (
    ASSIGNED_FRAGMENT_KEY, MAIN_UNLABELLED, explicit_fragment, fragment_counts,
    fragment_return_counts, observation_fragment, observation_sigma_arcsec,
    parse_observation_time, primary_fragment, seconds_since_j2000,
)
from .domain import Observation, OrbitSolution
from .i18n import tr
from .observatories import dynamical_jd
from .orbit_engine import AU_KM, StateVector

# Two rows of one station closer than this in time are the same exposure
# (days): their separation is the sky offset between the components.
SAME_EXPOSURE_DAYS = 2.0 / 1440.0
# Unlabelled rows within this many days of a component's labelled rows are
# compared with that component's orbit, where the orbit is well determined.
ASSIGNMENT_WINDOW_DAYS = 30.0
# A row joins a secondary component when it lies within this tolerance of
# that orbit (arcsec, or 3 sigma of the measurement if larger) ...
ASSIGNMENT_TOLERANCE_ARCSEC = 3.0
# ... and the principal orbit is at least this many times farther away.
ASSIGNMENT_CONTRAST = 3.0
AMBIGUOUS_FRAGMENT_KEY = "fragment_unresolved"
# Rows of a secondary track that the user split off as its own component.
SPLIT_OFF_KEY = "split_off_track"
ASSIGNMENT_REASON_KEY = "fragment_assignment_reason"

AU_PER_DAY_TO_M_PER_S = AU_KM * 1000.0 / 86400.0


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    component: str                 # MPC letters or MAIN_UNLABELLED
    rows: int                      # measurements that belong to it
    labelled: int                  # of them with the MPC letter
    first: datetime | None         # UTC
    last: datetime | None          # UTC
    nights: int
    returns: int
    offset_arcsec: float | None    # median offset from the principal one
    offset_pairs: int              # same-exposure pairs behind that median
    principal: bool


def _sky_separation_arcsec(first: Observation, second: Observation) -> float:
    ra1, dec1 = np.radians((first.ra_deg, first.dec_deg))
    ra2, dec2 = np.radians((second.ra_deg, second.dec_deg))
    cosine = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) * 3600.0)


def _day(observation: Observation) -> float | None:
    if not observation.time:
        return None
    try:
        return seconds_since_j2000(parse_observation_time(observation.time)) / 86400.0
    except ValueError:
        return None


def component_summaries(observations: list[Observation]) -> list[ComponentSummary]:
    """One summary per component, principal first; empty for an undivided object."""
    primary = primary_fragment(observations)
    if not primary:
        return []
    groups: dict[str, list[tuple[float, Observation]]] = {}
    for observation in observations:
        day = _day(observation)
        if day is None or observation.ra_deg is None or observation.dec_deg is None:
            continue
        letters = observation_fragment(observation) or primary
        groups.setdefault(letters, []).append((day, observation))
    # Principal rows per station, sorted, for the same-exposure offsets.
    principal_by_station: dict[str, tuple[list[float], list[Observation]]] = {}
    for day, observation in sorted(groups.get(primary, []), key=lambda item: item[0]):
        days, rows = principal_by_station.setdefault(
            (observation.station or "").strip().upper(), ([], []),
        )
        days.append(day); rows.append(observation)
    returns = fragment_return_counts(observations)
    summaries: list[ComponentSummary] = []
    order = sorted(
        groups, key=lambda letters: (letters != primary, len(letters), letters),
    )
    for letters in order:
        items = sorted(groups[letters], key=lambda item: item[0])
        offsets: list[float] = []
        if letters != primary:
            for day, observation in items:
                days, rows = principal_by_station.get(
                    (observation.station or "").strip().upper(), ([], []),
                )
                index = bisect_left(days, day - SAME_EXPOSURE_DAYS)
                best = None
                while index < len(days) and days[index] <= day + SAME_EXPOSURE_DAYS:
                    separation = _sky_separation_arcsec(observation, rows[index])
                    best = separation if best is None else min(best, separation)
                    index += 1
                if best is not None:
                    offsets.append(best)
        first = parse_observation_time(items[0][1].time) if items else None
        last = parse_observation_time(items[-1][1].time) if items else None
        nights = len({
            (observation.station, parse_observation_time(observation.time).date())
            for _day_value, observation in items
        })
        summaries.append(ComponentSummary(
            component=letters, rows=len(items),
            labelled=sum(1 for _d, item in items if explicit_fragment(item) == letters),
            first=first, last=last, nights=nights,
            returns=returns.get(letters, 1),
            offset_arcsec=float(np.median(offsets)) if offsets else None,
            offset_pairs=len(offsets), principal=letters == primary,
        ))
    return summaries


@dataclass(frozen=True, slots=True)
class AssignmentReport:
    assigned: dict[str, int]       # component -> newly assigned rows
    unresolved: int                # rows where the tracks were not separable
    compared: int                  # unlabelled rows compared at all


Evaluate = Callable[[list[Observation], OrbitSolution], object]


def assign_unlabelled_rows(
    observations: list[Observation], solutions: dict[str, OrbitSolution],
    evaluate: Evaluate,
) -> AssignmentReport:
    """Give unlabelled post-split rows to the component whose orbit they follow.

    Only rows within ``ASSIGNMENT_WINDOW_DAYS`` of a secondary component's
    labelled measurements are compared, where both fitted orbits are well
    determined.  A row is assigned when it lies within the tolerance of the
    secondary orbit and the principal orbit is ``ASSIGNMENT_CONTRAST`` times
    farther; where both orbits fit equally (tracks closer than the errors)
    it stays with the principal component and is marked as unresolved.
    Earlier assignments are recomputed; tracks split off with
    :func:`split_off_track` keep their component.  ``evaluate`` populates O-C of the
    rows it gets (``orbit_fit._evaluate_solution_observations``); it is only
    applied to copies, the displayed O-C is not changed.
    """
    for observation in observations:
        if observation.raw.get(SPLIT_OFF_KEY):
            continue
        observation.raw.pop(ASSIGNED_FRAGMENT_KEY, None)
        observation.raw.pop(AMBIGUOUS_FRAGMENT_KEY, None)
        observation.raw.pop(ASSIGNMENT_REASON_KEY, None)
    primary = primary_fragment(observations)
    if not primary or primary not in solutions:
        return AssignmentReport({}, 0, 0)
    windows: dict[str, list[tuple[float, float]]] = {}
    for observation in observations:
        letters = explicit_fragment(observation) or (
            observation_fragment(observation) if observation.raw.get(SPLIT_OFF_KEY) else ""
        )
        day = _day(observation)
        if not letters or letters == primary or letters not in solutions or day is None:
            continue
        windows.setdefault(letters, []).append((day, day))
    spans = {
        letters: (min(item[0] for item in values), max(item[1] for item in values))
        for letters, values in windows.items()
    }
    if not spans:
        return AssignmentReport({}, 0, 0)

    candidates: list[tuple[Observation, list[str]]] = []
    for observation in observations:
        if (
            explicit_fragment(observation) or observation.raw.get(SPLIT_OFF_KEY)
            or observation.ra_deg is None
        ):
            continue
        day = _day(observation)
        if day is None:
            continue
        nearby = [
            letters for letters, (start, end) in spans.items()
            if start - ASSIGNMENT_WINDOW_DAYS <= day <= end + ASSIGNMENT_WINDOW_DAYS
        ]
        if nearby:
            candidates.append((observation, nearby))
    if not candidates:
        return AssignmentReport({}, 0, 0)

    def distances(solution: OrbitSolution, rows: list[Observation]) -> list[float | None]:
        copies = [replace(item, raw=dict(item.raw)) for item in rows]
        try:
            evaluate(copies, solution)
        except (ValueError, RuntimeError, FloatingPointError, OverflowError):
            return [None] * len(rows)
        return [
            None if item.residual_ra_arcsec is None or item.residual_dec_arcsec is None
            else float(np.hypot(item.residual_ra_arcsec, item.residual_dec_arcsec))
            for item in copies
        ]

    rows = [item[0] for item in candidates]
    principal = distances(solutions[primary], rows)
    secondary = {
        letters: distances(solutions[letters], rows)
        for letters in spans
    }
    assigned: dict[str, int] = {}
    unresolved = 0
    for index, (observation, nearby) in enumerate(candidates):
        main = principal[index]
        sigma_ra, sigma_dec = observation_sigma_arcsec(observation)
        tolerance = max(ASSIGNMENT_TOLERANCE_ARCSEC, 3.0 * max(sigma_ra, sigma_dec))
        options = [
            (secondary[letters][index], letters) for letters in nearby
            if secondary[letters][index] is not None
        ]
        if main is None or not options:
            continue
        best, letters = min(options)
        if best <= tolerance and main > max(tolerance, ASSIGNMENT_CONTRAST * best):
            observation.raw[ASSIGNED_FRAGMENT_KEY] = letters
            observation.raw[ASSIGNMENT_REASON_KEY] = tr(
                "Neoznačené měření sedí na dráhu fragmentu {0} ({1:.1f}″), ne na hlavní "
                "složku {2} ({3:.1f}″).",
                letters, best, primary_label(primary), main,
            )
            assigned[letters] = assigned.get(letters, 0) + 1
        elif best <= tolerance and main <= tolerance:
            observation.raw[AMBIGUOUS_FRAGMENT_KEY] = True
            observation.raw[ASSIGNMENT_REASON_KEY] = tr(
                "Neoznačené měření nelze přiřadit: dráhy hlavní složky a fragmentu {0} "
                "jsou zde blíž než chyba měření.",
                letters,
            )
            unresolved += 1
    return AssignmentReport(assigned, unresolved, len(candidates))


# Component labels given to secondary tracks split off by the user: a digit
# never occurs in an MPC fragment designation, so they cannot collide.
DETECTED_TRACK_PREFIX = "X"
# Flagged secondary-track rows closer in time than this form one track (days).
SECONDARY_TRACK_GAP_DAYS = 60.0


@dataclass(frozen=True, slots=True)
class SecondaryTrack:
    rows: tuple[Observation, ...]
    stations: tuple[str, ...]
    first: datetime
    last: datetime


def secondary_tracks(observations: list[Observation]) -> list[SecondaryTrack]:
    """Unlabelled second tracks that a fit separated (``coherent_secondary_track``).

    The rows are grouped into tracks by time; each is a candidate fragment
    the observers did not label (205P in 2021: ZTF measured a secondary
    nucleus 65″ west of the comet, ATel 16104).
    """
    flagged = sorted(
        (
            (day, observation) for observation in observations
            if observation.raw.get("coherent_secondary_track")
            and not explicit_fragment(observation)
            and not observation.raw.get(ASSIGNED_FRAGMENT_KEY)
            and (day := _day(observation)) is not None
        ),
        key=lambda item: item[0],
    )
    groups: list[list[tuple[float, Observation]]] = []
    for item in flagged:
        if groups and item[0] - groups[-1][-1][0] <= SECONDARY_TRACK_GAP_DAYS:
            groups[-1].append(item)
        else:
            groups.append([item])
    return [
        SecondaryTrack(
            rows=tuple(observation for _day_value, observation in group),
            stations=tuple(sorted({(observation.station or "").strip() for _d, observation in group})),
            first=parse_observation_time(group[0][1].time),
            last=parse_observation_time(group[-1][1].time),
        )
        for group in groups
    ]


def split_off_track(observations: list[Observation], track: SecondaryTrack) -> str:
    """Make a secondary track a selectable component; returns its label (X1, X2…)."""
    used = {
        observation_fragment(observation) for observation in observations
    }
    number = 1
    while f"{DETECTED_TRACK_PREFIX}{number}" in used:
        number += 1
    label = f"{DETECTED_TRACK_PREFIX}{number}"
    for observation in track.rows:
        observation.raw[ASSIGNED_FRAGMENT_KEY] = label
        observation.raw[SPLIT_OFF_KEY] = True
        observation.raw[ASSIGNMENT_REASON_KEY] = tr(
            "Vedlejší stopa na stejných snímcích jako hlavní jádro; vyčleněna jako "
            "samostatná složka {0}.",
            label,
        )
        # The measurements are valid astrometry of the new component.
        observation.excluded_from_fit = False
        for key in ("automatic_exclusion", "coherent_secondary_track", "automatic_exclusion_reason"):
            observation.raw.pop(key, None)
    return label


def primary_label(component: str) -> str:
    return tr("hlavní jádro") if component == MAIN_UNLABELLED else component


# Separation models fitted to a fragment's astrometry.  Each value lists the
# fitted quantities: Δv (R, T, N in m/s) and/or ΔA1 (differential radial
# acceleration at 1 AU, falling off as r⁻²).  "deceleration" is Sekanina's classical
# two-parameter model (split epoch and deceleration, no separation velocity).
SPLIT_MODELS = {
    "deceleration": ("A1",),
    "impulse": ("VR", "VT", "VN"),
    "combined": ("VR", "VT", "VN", "A1"),
}
# Fragment separation velocities are typically 0.1-a few m/s; much larger
# values mean the model compensates an unmodelled effect.
PLAUSIBLE_SEPARATION_MS = 10.0
# Soft prior on each Δv component (m/s): observed fragment separation
# velocities are mostly below a few m/s (Sekanina 1982; 7.4 m/s projected for
# the 2021 secondary of 205P, ATel 16104).  Without it a short fragment arc is
# fitted equally well by implausibly fast separations near aphelion, where
# 205P passes close to Jupiter's orbit and the transition matrix is large.
SEPARATION_VELOCITY_PRIOR_MS = 3.0
# Differential A1 values much above a few 10^-8 AU/day² let a short arc mimic
# almost any split epoch.  Use a soft scale in the fit and reject solutions an
# order of magnitude above typical strong cometary accelerations.
DIFFERENTIAL_A1_PRIOR_AU_PER_DAY2 = 3e-8
PLAUSIBLE_DIFFERENTIAL_A1_AU_PER_DAY2 = 1e-7
SOLAR_GRAVITY_1AU = 2.959122082855911e-4   # AU/day² (GM_sun / 1 AU²)


@dataclass(frozen=True, slots=True)
class SplitModel:
    """Best split solution of one separation model."""
    name: str                      # key of SPLIT_MODELS
    epoch_jd: float                # TDB
    sigma_days: float | None       # chi² profile half-width (scaled by fit quality)
    delta_v_rtn_ms: tuple[float, float, float]   # radial, transverse, normal
    delta_a1: float | None         # differential radial A1 (AU/day²), if fitted
    rms_arcsec: float
    chi2: float
    parameters: int                # including the split epoch
    at_window_edge: bool
    heliocentric_distance_au: float
    # Other local minima: (TDB epoch, separation speed m/s, RMS arcsec, Δχ²).
    alternatives: tuple[tuple[float, float, float, float], ...] = ()
    # Minimum at the late end of the window, just before the first fragment
    # measurement.  That end is a physical limit (the fragment existed when it
    # was measured), so this is a valid "split shortly before discovery";
    # only its earliest time is constrained.
    before_first_observation: bool = False

    @property
    def speed_ms(self) -> float:
        return float(np.linalg.norm(self.delta_v_rtn_ms))

    @property
    def plausible(self) -> bool:
        acceleration_ok = (
            self.delta_a1 is None
            or abs(self.delta_a1) <= PLAUSIBLE_DIFFERENTIAL_A1_AU_PER_DAY2
        )
        return (
            self.speed_ms <= PLAUSIBLE_SEPARATION_MS
            and acceleration_ok
            and not self.at_window_edge
        )


@dataclass(frozen=True, slots=True)
class SplitEstimate:
    primary: str
    fragment: str
    models: tuple[SplitModel, ...]
    rows: int                      # fragment measurements used
    window_days: float

    @property
    def preferred(self) -> SplitModel:
        """Plausible model with the best fit per degree of freedom.

        Every extra parameter must lower χ²/dof by 10 %; implausible models
        (excessive separation speed/differential acceleration or a minimum on
        the search boundary) are chosen only when nothing else is available.
        """
        candidates = [model for model in self.models if model.plausible] or list(self.models)

        def score(model: SplitModel) -> float:
            dof = max(1, 2 * self.rows - model.parameters)
            return model.chi2 / dof * (1.1 ** model.parameters)

        return min(candidates, key=score)


# Search window before the first fragment measurement: 1.6 orbital periods
# of the principal component (Sekanina dated fragment C of 205P, first seen
# in 2008, to 1998), bounded below and above (days).
SPLIT_LOOKBACK_PERIODS = 1.6
SPLIT_MINIMUM_LOOKBACK_DAYS = 400.0
SPLIT_MAXIMUM_LOOKBACK_DAYS = 6000.0
SPLIT_MAXIMUM_ROWS = 80
SPLIT_COARSE_POINTS = 300
SPLIT_FINE_STEP_DAYS = 0.1
# Local minima of the coarse χ² profile that are refined with exact
# integrations (separated by this many coarse steps) and the evaluation
# budget of each exact refinement.
SPLIT_CANDIDATES = 3
SPLIT_CANDIDATE_SEPARATION_STEPS = 8
SPLIT_EXACT_EVALUATIONS = 40
# Other minima within this Δχ² are reported as competing solutions.
SPLIT_ALTERNATIVE_DELTA_CHI2 = 25.0


def _split_epoch_at_window_edge(epoch: float, window_start: float, margin: float) -> bool:
    """Whether the minimum lies at the arbitrary early end of the search.

    There the χ² valley continues beyond the window and the split epoch is
    not determined.
    """
    return epoch - window_start < margin


def _split_epoch_before_first_observation(
    epoch: float, window_end: float, margin: float,
) -> bool:
    """Whether the minimum lies at the late end, just before the first measurement.

    Unlike the early end this is a physical boundary: a fragment cannot
    separate after it was measured.  Recently split fragments (found days to
    weeks after the event) legitimately end up here.
    """
    return window_end - epoch < margin


def split_lookback_days(primary: OrbitSolution) -> float:
    """Search window (days) for :func:`estimate_split`."""
    from .orbit_engine import MU_SUN

    a = float(primary.semi_major_axis_au)
    if 0.0 <= float(primary.eccentricity) < 1.0 and np.isfinite(a) and a > 0.0:
        period = 2.0 * np.pi * a ** 1.5 / np.sqrt(MU_SUN)
        return float(np.clip(
            SPLIT_LOOKBACK_PERIODS * period,
            SPLIT_MINIMUM_LOOKBACK_DAYS, SPLIT_MAXIMUM_LOOKBACK_DAYS,
        ))
    return 1500.0


def _rtn_basis(state: np.ndarray) -> np.ndarray:
    """Rows: radial, transverse, normal unit vectors of a heliocentric state."""
    radial = state[:3] / np.linalg.norm(state[:3])
    normal = np.cross(state[:3], state[3:6])
    normal /= np.linalg.norm(normal)
    return np.vstack((radial, np.cross(normal, radial), normal))


def estimate_split(
    primary: OrbitSolution, fragment_rows: list[Observation],
    *, primary_label_text: str = "", fragment_label_text: str = "",
    lookback_days: float | None = None, cancel_callback=None,
) -> SplitEstimate:
    """Fit the split epoch of a fragment for the models in ``SPLIT_MODELS``.

    The fragment leaves the principal component at TDB ``t_s``: its state is
    the principal state plus a velocity impulse Δv, and it may move under a
    different radial non-gravitational parameter ΔA1 afterwards.  Positions
    follow linearly from Δv/ΔA1 through the principal orbit's variational
    equations, integrated once from its epoch ``t0``:
    Φ(t, t_s) = Φ(t, t0) Φ(t_s, t0)⁻¹ and, for a force switched on at
    ``t_s``, ∂r(t)/∂A1 = P(t) − Φ(t, t_s) P(t_s) (P = partials from ``t0``).
    Δv/ΔA1 are fitted to the fragment astrometry (light-time and station
    offsets as in the fit) for every trial ``t_s``; ``t_s`` minimizes the
    weighted χ² on a coarse grid that is then refined.  The principal orbit
    is taken as exact.
    """
    from scipy.optimize import least_squares

    from .astrometry import is_fit_candidate
    from .observatories import earth_heliocentric_ecliptic
    from .orbit_fit import (
        OrbitFitCancelled,
        C_AU_PER_DAY, _dynamics_engine, _equatorial_direction, _observation_offsets_gcrs,
    )

    if primary.position_au is None or primary.velocity_au_per_day is None:
        raise ValueError(tr("Chybí kartézský stav dráhy."))
    lookback = float(lookback_days if lookback_days is not None else split_lookback_days(primary))
    parsed = []
    for observation in fragment_rows:
        if not is_fit_candidate(observation) or not observation.time:
            continue
        try:
            parsed.append((
                observation, parse_observation_time(observation.time),
                float(observation.ra_deg), float(observation.dec_deg),
            ))
        except (TypeError, ValueError):
            continue
    if len(parsed) < 3:
        raise ValueError(tr("K odhadu rozpadu jsou potřeba alespoň 3 měření fragmentu."))
    parsed.sort(key=lambda item: item[1])
    if len(parsed) > SPLIT_MAXIMUM_ROWS:
        indices = np.unique(np.linspace(0, len(parsed) - 1, SPLIT_MAXIMUM_ROWS).round().astype(int))
        parsed = [parsed[index] for index in indices]
    epochs = dynamical_jd([item[1] for item in parsed])
    first = float(np.min(epochs))
    window_start = first - lookback
    window_end = first - 0.05
    sigmas = np.asarray([observation_sigma_arcsec(item[0]) for item in parsed], dtype=float)
    observed_ra = np.asarray([item[2] for item in parsed])
    observed_dec = np.asarray([item[3] for item in parsed])
    offsets = _observation_offsets_gcrs(parsed)
    earth = earth_heliocentric_ecliptic(epochs)
    weights = np.concatenate((sigmas[:, 0], sigmas[:, 1]))

    engine = _dynamics_engine(
        min(window_start, primary.epoch_jd), max(float(np.max(epochs)), primary.epoch_jd),
        dict(primary.nongravitational_parameters or {}),
    ).without_collision_detection()
    source = StateVector(
        primary.epoch_jd, np.asarray(primary.position_au, dtype=float),
        np.asarray(primary.velocity_au_per_day, dtype=float),
    )
    received = engine.propagate(source, epochs)
    emitted = epochs - np.linalg.norm(received[:, :3] - earth, axis=1) / C_AU_PER_DAY
    # Positions come from the plain integration; the variational one only
    # supplies the linear response (as in the fit).  The response engine uses
    # Sekanina's inverse-square law for the fragment's differential radial
    # acceleration (ΔA1 · (1 AU / r)², the standard model of split comets);
    # the principal's own small A1-A3 barely change its transition matrix.
    base_states = engine.propagate(source, emitted)
    response_engine = _dynamics_engine(
        min(window_start, primary.epoch_jd), max(float(np.max(epochs)), primary.epoch_jd),
        {"A1": 0.0, "R_EXPONENT": 2.0},
    ).without_collision_detection()
    cos_dec = np.cos(np.radians(observed_dec))
    a1_scale = 1e-8

    def integrate(split_epochs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Split states, STMs/partials at the split epochs and at the measurements.

        ``split_states`` come from the response engine; they only orient the
        RTN frame and give the heliocentric distance at the split.
        """
        if cancel_callback is not None and cancel_callback():
            raise OrbitFitCancelled(tr("Výpočet byl zrušen."))
        everything = np.concatenate((split_epochs, emitted))
        states, partials = response_engine.propagate_with_partials(
            source, everything, ("A1",), np.array([a1_scale]),
        )
        count = len(split_epochs)
        return states[:count], partials[:count], partials[count:], states[count:]

    observation_partials: np.ndarray | None = None

    def models_at(split_state: np.ndarray, split_partial: np.ndarray) -> dict:
        transition = observation_partials[:, :, :6] @ np.linalg.inv(split_partial[:, :6])
        velocity_partial = transition[:, :3, 3:6]
        a1_partial = (
            observation_partials[:, :3, 6] - transition[:, :3, :] @ split_partial[:, 6]
        )
        basis = _rtn_basis(split_state)
        results: dict = {"heliocentric": float(np.linalg.norm(split_state[:3]))}
        for name, quantities in SPLIT_MODELS.items():

            def residuals(values: np.ndarray, quantities=quantities) -> np.ndarray:
                named = dict(zip(quantities, values))
                # Δv in m/s along R, T, N; ΔA1 in 1e-8 AU/day² (optimizer units).
                rtn = np.array([named.get("VR", 0.0), named.get("VT", 0.0), named.get("VN", 0.0)])
                positions = base_states[:, :3] + velocity_partial @ (
                    basis.T @ (rtn / AU_PER_DAY_TO_M_PER_S)
                )
                positions = positions + a1_partial * named.get("A1", 0.0)
                moved = np.column_stack((positions, base_states[:, 3:]))
                ra, dec = _equatorial_direction(moved, epochs, offsets, earth)
                d_ra = ((observed_ra - ra + 180.0) % 360.0 - 180.0) * cos_dec * 3600.0
                d_dec = (observed_dec - dec) * 3600.0
                priors = []
                if "VR" in named:
                    priors.extend(rtn / SEPARATION_VELOCITY_PRIOR_MS)
                if "A1" in named:
                    priors.append(
                        named["A1"] * a1_scale
                        / DIFFERENTIAL_A1_PRIOR_AU_PER_DAY2
                    )
                return np.concatenate((
                    np.concatenate((d_ra, d_dec)) / weights,
                    np.asarray(priors, dtype=float),
                ))

            fitted = least_squares(residuals, np.zeros(len(quantities)), method="lm", max_nfev=200)
            weighted = residuals(fitted.x)
            astrometric = weighted[:len(weights)] * weights
            results[name] = (
                float(weighted @ weighted), fitted.x,
                float(np.sqrt(np.mean(astrometric ** 2))),
            )
        return results

    profiles: dict[float, dict] = {}

    def scan(split_epochs: np.ndarray) -> None:
        nonlocal observation_partials
        split_states, split_partials, measurement_partials, _states = integrate(split_epochs)
        observation_partials = measurement_partials
        for epoch, state, partial in zip(split_epochs, split_states, split_partials):
            if cancel_callback is not None and cancel_callback():
                raise OrbitFitCancelled(tr("Výpočet byl zrušen."))
            profiles[round(float(epoch), 4)] = models_at(state, partial)

    coarse_step = max(1.0, lookback / SPLIT_COARSE_POINTS)
    coarse = np.arange(window_start, window_end, coarse_step)
    scan(coarse)
    medium_step = max(SPLIT_FINE_STEP_DAYS, coarse_step / 10.0)

    def refine_linear(name: str, centre: float) -> float:
        """Best epoch near ``centre`` on medium, then fine grids (linear model)."""
        best = centre
        for previous, step in ((coarse_step, medium_step), (medium_step, SPLIT_FINE_STEP_DAYS)):
            grid = np.arange(
                max(window_start, best - 1.5 * previous),
                min(window_end, best + 1.5 * previous) + 1e-9, step,
            )
            fresh = np.array([value for value in grid if round(float(value), 4) not in profiles])
            if len(fresh):
                scan(fresh)
            local = [
                epoch for epoch in profiles
                if best - 1.5 * previous - 1e-6 <= epoch <= best + 1.5 * previous + 1e-6
            ]
            best = min(local, key=lambda epoch: profiles[epoch][name][0])
        return best

    # Exact (non-linear) residuals of a split solution: the fragment's
    # displacement from a gravity-only copy of the principal started at t_s,
    # added to the principal's actual positions.  The linear response is
    # accurate near the fragment arc but not for fast separations years
    # earlier (205P at aphelion, 10 m/s: 20″ error).
    reference_cache: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def exact_residuals(epoch: float, quantities: tuple[str, ...], values: np.ndarray) -> np.ndarray:
        if epoch not in reference_cache:
            split_state = response_engine.propagate(source, np.array([epoch]))[0]
            reference = response_engine.propagate(
                StateVector(epoch, split_state[:3], split_state[3:]), emitted,
            )
            reference_cache[epoch] = (split_state, reference)
        split_state, reference = reference_cache[epoch]
        named = dict(zip(quantities, values))
        rtn = np.array([named.get("VR", 0.0), named.get("VT", 0.0), named.get("VN", 0.0)])
        velocity = split_state[3:] + _rtn_basis(split_state).T @ (rtn / AU_PER_DAY_TO_M_PER_S)
        fragment_engine = response_engine.with_parameters(
            {"A1": float(named.get("A1", 0.0)) * a1_scale, "R_EXPONENT": 2.0},
        ).without_collision_detection()
        moved_positions = fragment_engine.propagate(
            StateVector(epoch, split_state[:3], velocity), emitted,
        )[:, :3]
        positions = base_states[:, :3] + (moved_positions - reference[:, :3])
        moved = np.column_stack((positions, base_states[:, 3:]))
        ra, dec = _equatorial_direction(moved, epochs, offsets, earth)
        d_ra = ((observed_ra - ra + 180.0) % 360.0 - 180.0) * cos_dec * 3600.0
        d_dec = (observed_dec - dec) * 3600.0
        priors = []
        if "VR" in named:
            priors.extend(rtn / SEPARATION_VELOCITY_PRIOR_MS)
        if "A1" in named:
            priors.append(
                named["A1"] * a1_scale / DIFFERENTIAL_A1_PRIOR_AU_PER_DAY2
            )
        return np.concatenate((
            np.concatenate((d_ra, d_dec)) / weights,
            np.asarray(priors, dtype=float),
        ))

    def refine_exact(name: str, epoch: float) -> tuple[float, np.ndarray, float]:
        quantities = SPLIT_MODELS[name]
        start_values = np.asarray(profiles[epoch][name][1], dtype=float)
        if cancel_callback is not None and cancel_callback():
            raise OrbitFitCancelled(tr("Výpočet byl zrušen."))
        fitted = least_squares(
            lambda values: exact_residuals(epoch, quantities, values), start_values,
            method="trf", x_scale=np.maximum(np.abs(start_values), 0.1),
            diff_step=1e-3, max_nfev=SPLIT_EXACT_EVALUATIONS,
        )
        weighted = exact_residuals(epoch, quantities, fitted.x)
        astrometric = weighted[:len(weights)] * weights
        return float(weighted @ weighted), fitted.x, float(np.sqrt(np.mean(astrometric ** 2)))

    def local_minima(name: str) -> list[float]:
        values = np.array([profiles[round(float(epoch), 4)][name][0] for epoch in coarse])
        order = np.argsort(values)
        chosen: list[int] = []
        for index in order:
            if all(abs(int(index) - other) > SPLIT_CANDIDATE_SEPARATION_STEPS for other in chosen):
                chosen.append(int(index))
            if len(chosen) == SPLIT_CANDIDATES:
                break
        return [round(float(coarse[index]), 4) for index in chosen]

    models: list[SplitModel] = []
    for name, quantities in SPLIT_MODELS.items():
        parameters = len(quantities) + 1
        dof = max(1, 2 * len(parsed) - parameters)
        trials = []
        for candidate in local_minima(name):
            epoch = refine_linear(name, candidate)
            chi2, vector, rms = refine_exact(name, epoch)
            trials.append((chi2, epoch, vector, rms))
        trials.sort(key=lambda item: item[0])
        chi2, best_epoch, vector, rms = trials[0]
        # Contiguous χ² valley around the minimum (linear profile, threshold
        # one scaled unit): its half-width is the uncertainty of the epoch.
        scale = max(1.0, chi2 / dof)
        ordered = sorted(profiles)
        values = [profiles[epoch][name][0] for epoch in ordered]
        centre = ordered.index(best_epoch)
        limit = profiles[best_epoch][name][0] + scale
        low = high = centre
        while low > 0 and values[low - 1] <= limit:
            low -= 1
        while high < len(ordered) - 1 and values[high + 1] <= limit:
            high += 1
        valley = (ordered[low], ordered[high])
        alternatives = tuple(
            (
                float(epoch),
                float(np.linalg.norm([
                    dict(zip(quantities, values_)).get(key, 0.0) for key in ("VR", "VT", "VN")
                ])),
                float(other_rms), float(other_chi2 - chi2),
            )
            for other_chi2, epoch, values_, other_rms in trials[1:]
            if other_chi2 - chi2 <= SPLIT_ALTERNATIVE_DELTA_CHI2
            and not valley[0] <= epoch <= valley[1]
        )
        named = dict(zip(quantities, vector))
        models.append(SplitModel(
            name=name, epoch_jd=float(best_epoch),
            sigma_days=0.5 * (valley[1] - valley[0]) if valley[1] > valley[0] else None,
            delta_v_rtn_ms=(
                float(named.get("VR", 0.0)), float(named.get("VT", 0.0)), float(named.get("VN", 0.0)),
            ),
            delta_a1=float(named["A1"] * a1_scale) if "A1" in named else None,
            rms_arcsec=rms, chi2=chi2, parameters=parameters,
            at_window_edge=_split_epoch_at_window_edge(best_epoch, window_start, coarse_step),
            heliocentric_distance_au=profiles[best_epoch]["heliocentric"],
            alternatives=alternatives,
            before_first_observation=_split_epoch_before_first_observation(
                best_epoch, window_end, coarse_step,
            ),
        ))
    return SplitEstimate(
        primary=primary_label_text, fragment=fragment_label_text,
        models=tuple(models), rows=len(parsed), window_days=lookback,
    )


SPLIT_MODEL_TITLES = {
    "deceleration": tr("Sekaninův model – rozpad a rozdílné zrychlení, bez rychlosti oddělení"),
    "impulse": tr("jen rychlost oddělení (impulz), bez rozdílného zrychlení"),
    "combined": tr("rychlost oddělení i rozdílné zrychlení"),
}


def describe_split(estimate: SplitEstimate) -> str:
    """Report text of one split estimate (Czech source, translated)."""
    from .presentation import datetime_from_jd

    preferred = estimate.preferred
    lines = [tr(
        "Odhad rozpadu: fragment {0} od složky {1} ({2} měření fragmentu)",
        estimate.fragment, estimate.primary, estimate.rows,
    )]
    for model in estimate.models:
        moment = datetime_from_jd(model.epoch_jd)
        moment_text = f"{moment:%Y-%m-%d %H:%M}" if moment else f"JD {model.epoch_jd:.2f}"
        sigma_text = (
            tr(" ± {0:.1f} dne", model.sigma_days) if model.sigma_days is not None else ""
        )
        marker = "  ▶ " if model is preferred else "    "
        lines.append(marker + SPLIT_MODEL_TITLES.get(model.name, model.name) + ":")
        lines.append(tr(
            "      čas rozpadu {0} UTC{1}; r = {2:.2f} AU; RMS fragmentu {3:.2f}″",
            moment_text, sigma_text, model.heliocentric_distance_au, model.rms_arcsec,
        ))
        if model.name != "deceleration":
            radial, transverse, normal = model.delta_v_rtn_ms
            lines.append(tr(
                "      rychlost oddělení {0:.2f} m/s (radiální {1:+.2f}, příčná {2:+.2f}, "
                "normálová {3:+.2f})",
                model.speed_ms, radial, transverse, normal,
            ))
        if model.delta_a1 is not None:
            lines.append(tr(
                "      rozdílné radiální zrychlení ΔA1 = {0:+.3e} AU/den² v 1 AU, ∝ r⁻² "
                "({1:.1f}·10⁻⁵ sluneční gravitace)",
                model.delta_a1, model.delta_a1 / SOLAR_GRAVITY_1AU * 1e5,
            ))
        for epoch, speed, other_rms, delta in model.alternatives:
            other = datetime_from_jd(epoch)
            lines.append(tr(
                "      jiné řešení: {0}, {1:.2f} m/s, RMS {2:.2f}″ (Δχ² = {3:+.1f})",
                f"{other:%Y-%m-%d}" if other else f"JD {epoch:.1f}", speed, other_rms, delta,
            ))
        if model.speed_ms > PLAUSIBLE_SEPARATION_MS:
            lines.append(tr(
                "      Rychlost nad {0:.0f} m/s je pro rozpad komety nefyzikální; model "
                "vyrovnává jiný, nemodelovaný efekt.",
                PLAUSIBLE_SEPARATION_MS,
            ))
        if (
            model.delta_a1 is not None
            and abs(model.delta_a1) > PLAUSIBLE_DIFFERENTIAL_A1_AU_PER_DAY2
        ):
            lines.append(tr(
                "      |ΔA1| nad {0:.1e} AU/den² je pro rozdílné zrychlení "
                "fragmentu nevěrohodné; model vyrovnává jiný efekt.",
                PLAUSIBLE_DIFFERENTIAL_A1_AU_PER_DAY2,
            ))
        if model.at_window_edge:
            lines.append(tr(
                "      Minimum leží na okraji prohledaného intervalu ({0:.0f} dní před "
                "prvním měřením fragmentu); okamžik rozpadu není určen.",
                estimate.window_days,
            ))
        if model.before_first_observation:
            lines.append(tr(
                "      Rozpad vychází těsně před prvním měřením fragmentu: fragment je "
                "čerstvý, nejvýše asi {0:.0f} dní starý; přesný okamžik uvnitř tohoto "
                "intervalu data neurčí.",
                max(1.0, estimate.window_days / SPLIT_COARSE_POINTS),
            ))
    lines.append(tr(
        "  ▶ = nejpravděpodobnější model (nejlepší shoda na stupeň volnosti s "
        "fyzikálně přijatelnými hodnotami). Dráha hlavní složky se bere jako přesná. "
        "Krátký oblouk fragmentu často připouští několik okamžiků rozpadu (jiná "
        "řešení s malým Δχ²); odhad je orientační.",
    ))
    return "\n".join(lines)
