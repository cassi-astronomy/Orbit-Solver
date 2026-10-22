from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
import re

import numpy as np

from .domain import OrbitSolution, is_comet_designation
from .i18n import tr
from .mpc_orbit import cartesian_to_elements
from .orbit_engine import (
    DEFAULT_PERTURBERS, MU_SUN, NBodyIntegrator, StateVector, tolerances_for_span,
)
from .planetary import PLANETS, SplinePlanetEphemeris


JULIAN_YEAR_DAYS = 365.25

# Approximate Hill radii in AU.  They are used only as a useful encounter
# screening scale; the actual trajectory is integrated with the planet's mass.
HILL_RADII_AU = {
    "mercury": .00147, "venus": .00671, "earth": .00980, "mars": .00658,
    "jupiter": .355, "saturn": .436, "uranus": .468, "neptune": .769,
}
PLANET_LABELS = {
    "mercury": tr("Merkur"), "venus": tr("Venuše"), "earth": tr("Země"), "mars": "Mars",
    "jupiter": "Jupiter", "saturn": "Saturn", "uranus": tr("Uran"), "neptune": tr("Neptun"),
}
PLANET_MU = {item.name: item.mu_au3_day2 for item in DEFAULT_PERTURBERS}
ELEMENT_CENTERS = (("sun", tr("Slunce")),) + tuple(PLANET_LABELS.items())


@dataclass(frozen=True, slots=True)
class EvolutionReturn:
    index: int
    epoch_jd: float
    state: tuple[float, ...]
    semi_major_axis_au: float
    eccentricity: float
    inclination_deg: float
    node_deg: float
    arg_peri_deg: float
    perihelion_au: float
    period_years: float | None
    reciprocal_a: float


@dataclass(frozen=True, slots=True)
class PlanetEncounter:
    epoch_jd: float
    planet: str
    distance_au: float
    hill_radii: float
    relative_velocity_kms: float
    delta_reciprocal_a: float | None


@dataclass(frozen=True, slots=True)
class CaptureInterval:
    planet: str
    start_jd: float
    end_jd: float
    minimum_distance_au: float
    minimum_hill_radii: float


@dataclass(frozen=True, slots=True)
class EvolutionSnapshot:
    """Osculating state at an inbound, perihelion, or outbound open-orbit phase."""
    phase: str
    epoch_jd: float
    state: tuple[float, ...]
    semi_major_axis_au: float
    eccentricity: float
    inclination_deg: float
    node_deg: float
    arg_peri_deg: float
    perihelion_au: float
    period_years: float | None
    reciprocal_a: float


@dataclass(frozen=True, slots=True)
class EvolutionResult:
    returns: tuple[EvolutionReturn, ...]
    encounters: tuple[PlanetEncounter, ...]
    start_jd: float
    end_jd: float
    sample_step_days: float
    model_note: str
    snapshots: tuple[EvolutionSnapshot, ...] = field(default_factory=tuple)
    timeline: tuple[EvolutionSnapshot, ...] = field(default_factory=tuple)
    trajectory_epochs: np.ndarray | None = field(default=None, repr=False, compare=False)
    trajectory_states: np.ndarray | None = field(default=None, repr=False, compare=False)
    captures: tuple[CaptureInterval, ...] = field(default_factory=tuple)


def planet_velocity(body: str, epoch_jd: float, step_days: float = .02) -> np.ndarray:
    """Central-difference heliocentric velocity from the active ephemeris."""
    return (
        PLANETS.position(body, epoch_jd + step_days)
        - PLANETS.position(body, epoch_jd - step_days)
    ) / (2.0 * step_days)


def centered_snapshot(
    item: EvolutionReturn | EvolutionSnapshot, center: str,
) -> EvolutionSnapshot:
    """Express a stored heliocentric state as osculating elements about a body."""
    center = center.lower()
    if center == "sun":
        return _snapshot_from_state(
            getattr(item, "phase", ""), item.epoch_jd, np.asarray(item.state, dtype=float)
        )
    if center not in PLANET_MU:
        raise ValueError(tr("Neznámý střed elementů: {0}", center))
    state = np.asarray(item.state, dtype=float).copy()
    state[:3] -= PLANETS.position(center, item.epoch_jd)
    state[3:] -= planet_velocity(center, item.epoch_jd)
    a, e, inc, node, arg_peri, _mean = cartesian_to_elements(
        state[:3], state[3:], PLANET_MU[center]
    )
    period = 2.0 * np.pi * np.sqrt(a**3 / PLANET_MU[center]) / JULIAN_YEAR_DAYS \
        if a > 0.0 and e < 1.0 else None
    return EvolutionSnapshot(
        phase=getattr(item, "phase", ""), epoch_jd=item.epoch_jd,
        state=tuple(float(value) for value in state), semi_major_axis_au=float(a),
        eccentricity=float(e), inclination_deg=float(inc), node_deg=float(node),
        arg_peri_deg=float(arg_peri), perihelion_au=float(a * (1.0 - e)),
        period_years=None if period is None else float(period),
        reciprocal_a=float(1.0 / a) if a != 0.0 else float("nan"),
    )


def centered_solution(source: OrbitSolution, center: str) -> OrbitSolution:
    """Return display-only osculating elements relative to a selected center."""
    center = center.lower()
    if center == "sun":
        return source
    if source.position_au is None or source.velocity_au_per_day is None:
        raise ValueError(tr("Planetocentrické elementy vyžadují kartézský stav."))
    if center not in PLANET_MU:
        raise ValueError(tr("Neznámý střed elementů: {0}", center))
    position = np.asarray(source.position_au, dtype=float) - PLANETS.position(
        center, source.epoch_jd
    )
    velocity = np.asarray(source.velocity_au_per_day, dtype=float) - planet_velocity(
        center, source.epoch_jd
    )
    a, e, inc, node, arg_peri, mean = cartesian_to_elements(
        position, velocity, PLANET_MU[center]
    )
    return replace(
        source, semi_major_axis_au=float(a), eccentricity=float(e),
        inclination_deg=float(inc), node_deg=float(node), arg_peri_deg=float(arg_peri),
        mean_anomaly_deg=float(mean), position_au=tuple(float(value) for value in position),
        velocity_au_per_day=tuple(float(value) for value in velocity),
        state_covariance=None, element_uncertainties=None, dynamical_1a=None,
    )


def _format_jd_utc(jd: float) -> str:
    try:
        value = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(
            days=float(jd) - 2451545.0
        )
        return value.strftime("%Y-%m-%d %H:%M:%S")
    except (OverflowError, ValueError):
        return f"JD {jd:.8f}"


def format_evolution_report(
    solution: OrbitSolution, result: EvolutionResult,
) -> str:
    """Create a chronological, tab-separated text export of an evolution run."""
    name = solution.display_name or solution.designation
    lines = [
        tr("ORBIT SOLVER — VÝVOJ DRÁHY"),
        tr("Objekt:\t{0}", name),
        tr(
            "Rozsah UTC:\t{0}\t{1}",
            _format_jd_utc(result.start_jd),
            _format_jd_utc(result.end_jd),
        ),
        tr("Krok vyhledávání událostí [dny]:\t{0:.6f}", result.sample_step_days),
        f"Model:\t{result.model_note}",
        "",
        tr("Chronologický přehled (elementy jsou oskulační v uvedené epoše)"),
        tr(
            "Typ\tDatum UTC\tFáze / návrat / planeta\tq [AU]\te\ta [AU]\tP [roky]\t1/a "
            "[AU^-1]\ti [deg]\tOmega [deg]\tomega [deg]\tVzdálenost [AU]\tHillovy "
            "poloměry\tv rel. [km/s]\tDelta(1/a) [AU^-1]",
        ),
    ]
    events: list[tuple[float, int, EvolutionReturn | EvolutionSnapshot | PlanetEncounter]] = []
    orbital_states = result.snapshots or result.returns
    events.extend((item.epoch_jd, 0, item) for item in orbital_states)
    events.extend((item.epoch_jd, 1, item) for item in result.encounters)
    for _epoch, _event_type, item in sorted(events, key=lambda row: (row[0], row[1])):
        if isinstance(item, (EvolutionReturn, EvolutionSnapshot)):
            period = "" if item.period_years is None else f"{item.period_years:.8f}"
            marker = tr(item.phase) if isinstance(item, EvolutionSnapshot) else f"{item.index:+d}"
            lines.append(
                f"{'STAV' if isinstance(item, EvolutionSnapshot) else 'PERIHELIUM'}\t"
                f"{_format_jd_utc(item.epoch_jd)}\t{marker}\t"
                f"{item.perihelion_au:.9f}\t{item.eccentricity:.10f}\t"
                f"{item.semi_major_axis_au:.9f}\t{period}\t{item.reciprocal_a:.10e}\t"
                f"{item.inclination_deg:.7f}\t{item.node_deg:.7f}\t"
                f"{item.arg_peri_deg:.7f}\t\t\t\t"
            )
        else:
            delta = "" if item.delta_reciprocal_a is None else f"{item.delta_reciprocal_a:+.10e}"
            lines.append(
                tr(
                    "SETKÁNÍ\t{0}\t{1}\t\t\t\t\t\t\t\t\t{2:.9f}\t{3:.6f}\t{4:.5f}\t{5}",
                    _format_jd_utc(item.epoch_jd),
                    PLANET_LABELS.get(item.planet, item.planet),
                    item.distance_au,
                    item.hill_radii,
                    item.relative_velocity_kms,
                    delta,
                )
            )
    if not events:
        lines.append(tr("Žádné perihelové průchody ani planetární setkání nebyly nalezeny."))
    if result.captures:
        lines.extend(("", tr("Dočasná planetocentrická zachycení")))
        for capture in result.captures:
            lines.append(
                tr(
                    "ZACHYCENÍ\t{0}\t{1}\t{2}\tminimum {3:.9f} AU\t{4:.5f} Hillova poloměru",
                    _format_jd_utc(capture.start_jd),
                    _format_jd_utc(capture.end_jd),
                    PLANET_LABELS.get(capture.planet, capture.planet),
                    capture.minimum_distance_au,
                    capture.minimum_hill_radii,
                )
            )
    lines.extend(("", tr("Vytvořeno programem Orbit Solver 1.0 — Martin Mašek")))
    return "\n".join(lines) + "\n"


def _packed_mpc_epoch(jd: float) -> str:
    try:
        value = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(
            days=float(jd) - 2451545.0
        )
    except (OverflowError, ValueError):
        return "?????"
    century = {18: "I", 19: "J", 20: "K", 21: "L"}.get(value.year // 100, "?")
    digit = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return f"{century}{value.year % 100:02d}{digit[value.month]}{digit[value.day]}"


def _mpc_return_line(solution: OrbitSolution, item: EvolutionReturn) -> str:
    provisional_comet = re.match(
        r"^\s*([ACDPX])/\s*(\d{4})\s+([A-Z])(\d+)", solution.designation, re.I
    )
    numbered_comet = re.match(r"^\s*(\d+)([PD])(?:/|\b)", solution.designation, re.I)
    if provisional_comet or numbered_comet:
        periodic_number = int(numbered_comet.group(1)) if numbered_comet else 0
        packed_provisional = ""
        if provisional_comet:
            orbit_type, year_text, half_month, sequence_text = provisional_comet.groups()
            year = int(year_text)
            century = {18: "I", 19: "J", 20: "K", 21: "L"}.get(year // 100, "?")
            packed_provisional = (
                f"{century}{year % 100:02d}{half_month.upper()}"
                f"{int(sequence_text):02d}0"
            )
        else:
            orbit_type = numbered_comet.group(2).upper()
        try:
            perihelion = datetime(2000, 1, 1, 12, tzinfo=timezone.utc) + timedelta(
                days=item.epoch_jd - 2451545.0
            )
        except (OverflowError, ValueError):
            return tr(
                "# Návrat {0:+d}: kalendářní datum je mimo podporovaný rozsah.",
                item.index,
            )
        day_fraction = perihelion.day + (
            perihelion.hour * 3600 + perihelion.minute * 60 + perihelion.second
            + perihelion.microsecond / 1e6
        ) / 86400.0
        number_field = f"{periodic_number:4d}" if numbered_comet else "    "
        prefix = f"{number_field}{orbit_type.upper()}{packed_provisional:<7}"
        slope_parameter = 2.5 * solution.photometric_exponent
        return (
            f"{prefix}  {perihelion.year:04d} {perihelion.month:02d} {day_fraction:07.4f} "
            f"{item.perihelion_au:9.6f}  {item.eccentricity:8.6f}  "
            f"{item.arg_peri_deg:8.4f}  {item.node_deg:8.4f}  "
            f"{item.inclination_deg:8.4f}  {perihelion:%Y%m%d}  "
            f"{solution.absolute_magnitude:4.1f} {slope_parameter:4.1f}  "
            f"{solution.display_name or solution.designation}"
        )
    mean_motion = np.degrees(0.01720209895 / abs(item.semi_major_axis_au) ** 1.5)
    packed_designation = solution.designation.replace(" ", "")[:7]
    return (
        f"{packed_designation:<7} {solution.absolute_magnitude:5.2f} {solution.slope_g:5.2f} "
        f"{_packed_mpc_epoch(item.epoch_jd):5s} {0.0:9.5f} "
        f"{item.arg_peri_deg:9.5f} {item.node_deg:9.5f} {item.inclination_deg:9.5f} "
        f"{item.eccentricity:10.7f} {mean_motion:11.8f} {item.semi_major_axis_au:11.7f}"
    )


def format_evolution_mpc_report(
    solution: OrbitSolution, result: EvolutionResult,
) -> str:
    """Return chronological MPC-style element lines plus commented encounters."""
    name = solution.display_name or solution.designation
    lines = [
        tr("# Orbit Solver 1.0 — vývoj dráhy: {0}", name),
        tr(
            "# Rozsah UTC: {0} až {1}",
            _format_jd_utc(result.start_jd),
            _format_jd_utc(result.end_jd),
        ),
        tr("# Elementy jsou oskulační v okamžiku příslušného perihelu."),
        tr("# Řádky začínající # jsou komentáře a planetární setkání."),
        f"# {result.model_note}",
    ]
    events: list[tuple[float, int, EvolutionReturn | EvolutionSnapshot | PlanetEncounter]] = []
    orbital_states = result.snapshots or result.returns
    events.extend((item.epoch_jd, 0, item) for item in orbital_states)
    events.extend((item.epoch_jd, 1, item) for item in result.encounters)
    for _epoch, _kind, item in sorted(events, key=lambda row: (row[0], row[1])):
        if isinstance(item, EvolutionSnapshot):
            lines.append(
                tr(
                    "# STAV {0} — {1} UTC; q={2:.9f} AU; e={3:.10f}; a={4:.9f} AU; "
                    "1/a={5:+.10e} AU^-1; i={6:.7f}; Omega={7:.7f}; omega={8:.7f}",
                    tr(item.phase),
                    _format_jd_utc(item.epoch_jd),
                    item.perihelion_au,
                    item.eccentricity,
                    item.semi_major_axis_au,
                    item.reciprocal_a,
                    item.inclination_deg,
                    item.node_deg,
                    item.arg_peri_deg,
                )
            )
        elif isinstance(item, EvolutionReturn):
            lines.append(tr(
                "# Návrat {0:+d} — {1} UTC",
                item.index,
                _format_jd_utc(item.epoch_jd),
            ))
            lines.append(_mpc_return_line(solution, item))
        else:
            delta = (
                tr("neuvedeno") if item.delta_reciprocal_a is None
                else f"{item.delta_reciprocal_a:+.6e} AU^-1"
            )
            lines.append(
                tr(
                    "# SETKÁNÍ {0} UTC — {1}; d={2:.6f} AU = {3:.3f} Hillova poloměru; "
                    "v_rel={4:.2f} km/s; Delta(1/a)={5}",
                    _format_jd_utc(item.epoch_jd),
                    PLANET_LABELS.get(item.planet, item.planet),
                    item.distance_au,
                    item.hill_radii,
                    item.relative_velocity_kms,
                    delta,
                )
            )
    for capture in result.captures:
        lines.append(
            tr(
                "# ZACHYCENÍ {0}: {1} až {2} UTC; minimum={3:.6f} AU ({4:.3f} Hillova "
                "poloměru)",
                PLANET_LABELS.get(capture.planet, capture.planet),
                _format_jd_utc(capture.start_jd),
                _format_jd_utc(capture.end_jd),
                capture.minimum_distance_au,
                capture.minimum_hill_radii,
            )
        )
    return "\n".join(lines) + "\n"


def _parabolic_vertex_time(times: np.ndarray, values: np.ndarray, index: int) -> float:
    """Refine a sampled local minimum with a three-point parabola."""
    if index <= 0 or index >= len(times) - 1:
        return float(times[index])
    x = times[index - 1:index + 2] - times[index]
    y = values[index - 1:index + 2]
    try:
        coefficient = np.polyfit(x, y, 2)
        if coefficient[0] <= 0.0:
            return float(times[index])
        offset = float(-coefficient[1] / (2.0 * coefficient[0]))
        if offset < x[0] or offset > x[-1]:
            return float(times[index])
        return float(times[index] + offset)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return float(times[index])


def _solution_state(solution: OrbitSolution) -> StateVector:
    if solution.position_au is None or solution.velocity_au_per_day is None:
        raise ValueError(
            tr("Vývoj dráhy vyžaduje kartézský stav z MPC nebo z vlastního fitu.")
        )
    return StateVector(
        solution.epoch_jd,
        np.asarray(solution.position_au, dtype=float),
        np.asarray(solution.velocity_au_per_day, dtype=float),
    )


# Margin kept inside the planetary-kernel coverage (days) for the spline padding
# and the light-time iteration.
COVERAGE_MARGIN_DAYS = 5.0
# Integrations longer than this (years) are confirmed in the GUI first: they
# take long and a distant extrapolation is dominated by orbit uncertainty.
LONG_EVOLUTION_YEARS = 1000.0


def approximate_year(jd: float) -> float:
    """Decimal calendar year of a Julian date (display only)."""
    return 2000.0 + (float(jd) - 2451545.0) / JULIAN_YEAR_DAYS


def usable_coverage(coverage: tuple[float, float] | None) -> tuple[float, float] | None:
    """Planetary-model coverage minus the integration margin (TDB JD)."""
    if coverage is None:
        return None
    return coverage[0] + COVERAGE_MARGIN_DAYS, coverage[1] - COVERAGE_MARGIN_DAYS


def clip_span_to_coverage(
    start_jd: float, end_jd: float, coverage: tuple[float, float] | None,
) -> tuple[float, float]:
    """Restrict an integration interval to the usable planetary coverage."""
    limits = usable_coverage(coverage)
    if limits is None:
        return start_jd, end_jd
    start, end = max(start_jd, limits[0]), min(end_jd, limits[1])
    if end <= start:
        raise ValueError(tr("Zvolený rozsah leží celý mimo pokrytí planetární efemeridy."))
    return start, end


def evolution_span(
    solution: OrbitSolution, mode: str, past: int, future: int,
) -> tuple[float, float]:
    """Translate the UI range choice to absolute Julian dates."""
    if past < 0 or future < 0 or (past == 0 and future == 0):
        raise ValueError(tr("Rozsah integrace musí být větší než nula."))
    if mode == "years":
        return (
            solution.epoch_jd - float(past) * JULIAN_YEAR_DAYS,
            solution.epoch_jd + float(future) * JULIAN_YEAR_DAYS,
        )
    if mode != "returns":
        raise ValueError(tr("Neznámý způsob zadání rozsahu integrace."))
    if not (solution.eccentricity < 1.0 and solution.semi_major_axis_au > 0.0):
        raise ValueError(tr("Počet návratů lze zadat jen pro uzavřenou eliptickou dráhu."))
    period_days = 2.0 * np.pi * solution.semi_major_axis_au ** 1.5 / np.sqrt(MU_SUN)
    # The half-period margin makes the requested outer perihelia fall inside
    # the interval even when the fitted epoch is not itself at perihelion.
    return (
        solution.epoch_jd - (float(past) + .65) * period_days,
        solution.epoch_jd + (float(future) + .65) * period_days,
    )


def _return_from_state(index: int, epoch: float, state: np.ndarray) -> EvolutionReturn:
    a, e, inc, node, arg_peri, _mean = cartesian_to_elements(state[:3], state[3:])
    period = a ** 1.5 if a > 0.0 and e < 1.0 else None
    return EvolutionReturn(
        index=index, epoch_jd=float(epoch), state=tuple(float(value) for value in state),
        semi_major_axis_au=float(a), eccentricity=float(e), inclination_deg=float(inc),
        node_deg=float(node), arg_peri_deg=float(arg_peri),
        perihelion_au=float(a * (1.0 - e)),
        period_years=None if period is None else float(period),
        reciprocal_a=float(1.0 / a) if a != 0.0 else float("nan"),
    )


# Planet positions are cached as a daily spline for spans up to this length
# (days); a longer exploratory integration reads the kernel directly.
SPLINE_CACHE_MAXIMUM_DAYS = 400.0 * JULIAN_YEAR_DAYS
# Tolerance of the direct-kernel engine for exploratory multi-century runs.
EXPLORATORY_TOLERANCES = (2e-11, 2e-13)


def _evolution_engine(
    solution: OrbitSolution, start_jd: float, end_jd: float,
) -> NBodyIntegrator:
    """Force model for an evolution run (heliocentric J2000 ecliptic, TDB).

    Spans that a fit can cover use the fit's planetary cache and accuracy
    profile, so the tables of a linked comet agree with the fitted model across
    the same encounters.  Longer, exploratory spans keep the cheaper historical
    settings because their result is dominated by chaos, not integration error.
    """
    span = max(end_jd, solution.epoch_jd) - min(start_jd, solution.epoch_jd)
    if span <= SPLINE_CACHE_MAXIMUM_DAYS:
        ephemeris = SplinePlanetEphemeris(
            PLANETS, tuple(item.name for item in DEFAULT_PERTURBERS),
            min(start_jd, solution.epoch_jd) - 2.0, max(end_jd, solution.epoch_jd) + 2.0,
        )
        rtol, atol = tolerances_for_span(span)
    else:
        ephemeris, (rtol, atol) = PLANETS, EXPLORATORY_TOLERANCES
    return NBodyIntegrator(
        ephemeris, relative_tolerance=rtol, absolute_tolerance=atol,
        nongravitational_parameters=solution.nongravitational_parameters,
    )


def _states_near_samples(
    engine: NBodyIntegrator, epochs: np.ndarray, states: np.ndarray,
    targets: np.ndarray,
) -> np.ndarray:
    """States at ``targets``, each integrated from the nearest stored sample.

    Refining a perihelion or an encounter needs only a few days of motion; a
    new integration from the solution epoch would repeat the whole span.
    """
    output = np.empty((len(targets), 6), dtype=float)
    for row, target in enumerate(np.asarray(targets, dtype=float)):
        index = int(np.argmin(np.abs(epochs - target)))
        start = StateVector(float(epochs[index]), states[index, :3], states[index, 3:])
        output[row] = engine.propagate(start, np.array([target]))[0]
    return output


def perihelion_returns_between(
    solution: OrbitSolution, start_jd: float, end_jd: float,
    *, maximum_samples: int = 40001,
) -> tuple[EvolutionReturn, ...]:
    """Osculating elements at every perihelion between two TDB Julian dates."""
    return calculate_orbit_evolution_between(
        solution, start_jd, end_jd, maximum_samples=maximum_samples,
        include_encounters=False,
    ).returns


def _snapshot_from_state(phase: str, epoch: float, state: np.ndarray) -> EvolutionSnapshot:
    item = _return_from_state(0, epoch, state)
    return EvolutionSnapshot(
        phase=phase, epoch_jd=item.epoch_jd, state=item.state,
        semi_major_axis_au=item.semi_major_axis_au,
        eccentricity=item.eccentricity, inclination_deg=item.inclination_deg,
        node_deg=item.node_deg, arg_peri_deg=item.arg_peri_deg,
        perihelion_au=item.perihelion_au, period_years=item.period_years,
        reciprocal_a=item.reciprocal_a,
    )


def calculate_orbit_evolution(
    solution: OrbitSolution, mode: str, past: int, future: int,
    *, maximum_samples: int = 80001, span: tuple[float, float] | None = None,
) -> EvolutionResult:
    """Integrate over a UI-selected range of returns or years.

    ``span`` (TDB JD) replaces the interval derived from ``mode`` when the GUI
    has restricted it, e.g. to the coverage of the planetary ephemeris.
    """
    start_jd, end_jd = span if span is not None else evolution_span(
        solution, mode, past, future,
    )
    result = calculate_orbit_evolution_between(
        solution, start_jd, end_jd, maximum_samples=maximum_samples,
    )
    if mode == "returns":
        result = replace(result, returns=tuple(
            item for item in result.returns if -past <= item.index <= future
        ))
    return result


def calculate_orbit_evolution_between(
    solution: OrbitSolution, start_jd: float, end_jd: float,
    *, maximum_samples: int = 80001, include_encounters: bool = True,
) -> EvolutionResult:
    """Integrate once and extract perihelia, planetary encounters and captures.

    ``start_jd``/``end_jd`` are TDB Julian dates.  Return index 0 is the
    perihelion nearest to the solution epoch.  The complete sampled trajectory
    is kept for the Solar System animation.
    """
    initial = _solution_state(solution)
    span_days = end_jd - start_jd
    if span_days <= 0.0:
        raise ValueError(tr("Neplatný rozsah integrace."))
    if solution.eccentricity < 1.0 and solution.semi_major_axis_au > 0.0:
        period_days = 2.0 * np.pi * solution.semi_major_axis_au ** 1.5 / np.sqrt(MU_SUN)
        desired_step = float(np.clip(period_days / 360.0, .25, 5.0))
    else:
        desired_step = 2.0
    sample_count = min(maximum_samples, max(401, int(np.ceil(span_days / desired_step)) + 1))
    sample_epochs = np.linspace(start_jd, end_jd, sample_count)
    sample_step = float(sample_epochs[1] - sample_epochs[0])
    engine = _evolution_engine(solution, start_jd, end_jd)
    states = engine.propagate(initial, sample_epochs)

    radius = np.linalg.norm(states[:, :3], axis=1)
    minima = np.flatnonzero((radius[1:-1] < radius[:-2]) & (radius[1:-1] <= radius[2:])) + 1
    refined_epochs = [_parabolic_vertex_time(sample_epochs, radius, int(index)) for index in minima]
    returns: tuple[EvolutionReturn, ...] = ()
    if refined_epochs:
        refined_states = _states_near_samples(
            engine, sample_epochs, states, np.asarray(refined_epochs, dtype=float),
        )
        zero_position = int(np.argmin(np.abs(np.asarray(refined_epochs) - solution.epoch_jd)))
        returns = tuple(
            _return_from_state(position - zero_position, epoch, state)
            for position, (epoch, state) in enumerate(zip(refined_epochs, refined_states))
        )

    encounters: list[PlanetEncounter] = []
    captures: list[CaptureInterval] = []
    for planet, hill_radius in (HILL_RADII_AU.items() if include_encounters else ()):
        planet_positions = np.asarray(PLANETS.position_series(planet, sample_epochs), dtype=float)
        distances = np.linalg.norm(states[:, :3] - planet_positions, axis=1)
        planet_velocities = np.gradient(planet_positions, sample_epochs, axis=0)
        relative_velocities = states[:, 3:] - planet_velocities
        specific_energy = (
            .5 * np.sum(relative_velocities * relative_velocities, axis=1)
            - PLANET_MU[planet] / np.maximum(distances, 1e-15)
        )
        captured = (specific_energy < 0.0) & (distances <= hill_radius)
        padded = np.concatenate(([False], captured, [False])).astype(np.int8)
        starts = np.flatnonzero(np.diff(padded) == 1)
        stops = np.flatnonzero(np.diff(padded) == -1)
        for first, stop in zip(starts, stops):
            # A single isolated sample is too cadence-dependent to call a
            # capture. Long-lived SL9-like and temporary-moon episodes span
            # multiple samples and are robust under this criterion.
            if stop - first < 2:
                continue
            minimum = float(np.min(distances[first:stop]))
            captures.append(CaptureInterval(
                planet=planet, start_jd=float(sample_epochs[first]),
                end_jd=float(sample_epochs[stop - 1]), minimum_distance_au=minimum,
                minimum_hill_radii=float(minimum / hill_radius),
            ))
        candidate_indices = np.flatnonzero(
            (distances[1:-1] < distances[:-2])
            & (distances[1:-1] <= distances[2:])
            & (distances[1:-1] < 3.0 * hill_radius)
        ) + 1
        for sample_index in candidate_indices:
            encounter_jd = _parabolic_vertex_time(sample_epochs, distances, int(sample_index))
            influence_window = max(30.0, 3.0 * sample_step)
            evaluation = np.array([
                encounter_jd - influence_window, encounter_jd, encounter_jd + influence_window,
            ])
            encounter_states = _states_near_samples(engine, sample_epochs, states, evaluation)
            planet_at_encounter = PLANETS.position(planet, encounter_jd)
            separation = float(np.linalg.norm(encounter_states[1, :3] - planet_at_encounter))
            # Central difference is adequate for the displayed encounter speed
            # and avoids requiring a velocity API from the planet provider.
            velocity_step = .02
            planet_velocity = (
                PLANETS.position(planet, encounter_jd + velocity_step)
                - PLANETS.position(planet, encounter_jd - velocity_step)
            ) / (2.0 * velocity_step)
            relative_velocity = np.linalg.norm(encounter_states[1, 3:] - planet_velocity)
            before = cartesian_to_elements(encounter_states[0, :3], encounter_states[0, 3:])[0]
            after = cartesian_to_elements(encounter_states[2, :3], encounter_states[2, 3:])[0]
            delta_1a = None
            if before != 0.0 and after != 0.0 and np.isfinite(before + after):
                delta_1a = float(1.0 / after - 1.0 / before)
            encounters.append(PlanetEncounter(
                epoch_jd=float(encounter_jd), planet=planet,
                distance_au=separation,
                hill_radii=float(separation / hill_radius),
                relative_velocity_kms=float(relative_velocity * 149597870.7 / 86400.0),
                delta_reciprocal_a=delta_1a,
            ))
    encounters.sort(key=lambda item: item.epoch_jd)
    # Keep the complete integrated trajectory for the Solar System animation,
    # but calculate osculating elements only for a display-sized time series.
    # Extra points around perihelia and encounters preserve the dynamically
    # interesting changes even in integrations spanning several centuries.
    display_indices = set(
        np.linspace(0, sample_count - 1, min(sample_count, 2400), dtype=int).tolist()
    )
    event_epochs = list(refined_epochs) + [item.epoch_jd for item in encounters]
    for event_epoch in event_epochs:
        center = int(np.searchsorted(sample_epochs, event_epoch))
        for offset in range(-5, 6):
            display_indices.add(int(np.clip(center + offset, 0, sample_count - 1)))
    timeline = tuple(
        _snapshot_from_state("", float(sample_epochs[index]), states[index])
        for index in sorted(display_indices)
    )
    snapshots: tuple[EvolutionSnapshot, ...] = ()
    comet = (
        is_comet_designation(solution.designation)
        or is_comet_designation(solution.display_name or "")
    )
    open_source = not (
        solution.eccentricity < 1.0 and solution.semi_major_axis_au > 0.0
    )
    # This view is useful for every single-pass comet, not only an object
    # already classified as interstellar at the source epoch.  A dynamically
    # new Oort-cloud comet may arrive weakly bound and leave hyperbolic after
    # planetary perturbations, while still having exactly one perihelion in a
    # centuries-long integration.
    if len(returns) == 1 and (open_source or comet):
        perihelion = returns[0]
        perihelion_state = np.asarray(perihelion.state, dtype=float)
        perihelion_epoch = perihelion.epoch_jd
        snapshots = (
            _snapshot_from_state("vstup", float(sample_epochs[0]), states[0]),
            _snapshot_from_state("perihel", perihelion_epoch, perihelion_state),
            _snapshot_from_state("výstup", float(sample_epochs[-1]), states[-1]),
        )
    return EvolutionResult(
        returns=returns, encounters=tuple(encounters), start_jd=start_jd, end_jd=end_jd,
        sample_step_days=sample_step,
        model_note=(
            tr(
                "Orientační výpočet: planetární perturbace a zvolený model A1/A2/A3 jsou "
                "zahrnuty, polohy planet používají model {0}.",
                PLANETS.model_name,
            )
        ),
        snapshots=snapshots,
        timeline=timeline,
        trajectory_epochs=np.asarray(sample_epochs, dtype=float),
        trajectory_states=np.asarray(states, dtype=float),
        captures=tuple(sorted(captures, key=lambda item: item.start_jd)),
    )


def return_solution(source: OrbitSolution, item: EvolutionReturn) -> OrbitSolution:
    """Build an osculating solution suitable for the Solar System panel."""
    return OrbitSolution(
        designation=source.designation, display_name=source.display_name,
        epoch_jd=item.epoch_jd, semi_major_axis_au=item.semi_major_axis_au,
        eccentricity=item.eccentricity, inclination_deg=item.inclination_deg,
        node_deg=item.node_deg, arg_peri_deg=item.arg_peri_deg, mean_anomaly_deg=0.0,
        absolute_magnitude=source.absolute_magnitude, slope_g=source.slope_g,
        photometric_exponent=source.photometric_exponent,
        phase_correction=source.phase_correction, provisional=source.provisional,
        position_au=tuple(item.state[:3]), velocity_au_per_day=tuple(item.state[3:]),
        nongravitational_parameters=source.nongravitational_parameters,
    )
