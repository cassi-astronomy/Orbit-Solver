from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import re

import numpy as np

from .domain import OrbitSolution
from .mpc_orbit import cartesian_to_elements
from .orbit_engine import MU_SUN, NBodyIntegrator, StateVector
from .planetary import PLANETS


JULIAN_YEAR_DAYS = 365.25

# Approximate Hill radii in AU.  They are used only as a useful encounter
# screening scale; the actual trajectory is integrated with the planet's mass.
HILL_RADII_AU = {
    "mercury": .00147, "venus": .00671, "earth": .00980, "mars": .00658,
    "jupiter": .355, "saturn": .436, "uranus": .468, "neptune": .769,
}
PLANET_LABELS = {
    "mercury": "Merkur", "venus": "Venuše", "earth": "Země", "mars": "Mars",
    "jupiter": "Jupiter", "saturn": "Saturn", "uranus": "Uran", "neptune": "Neptun",
}


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
class EvolutionResult:
    returns: tuple[EvolutionReturn, ...]
    encounters: tuple[PlanetEncounter, ...]
    start_jd: float
    end_jd: float
    sample_step_days: float
    model_note: str


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
        "ORBIT SOLVER — VÝVOJ DRÁHY",
        f"Objekt:\t{name}",
        f"Rozsah UTC:\t{_format_jd_utc(result.start_jd)}\t{_format_jd_utc(result.end_jd)}",
        f"Krok vyhledávání událostí [dny]:\t{result.sample_step_days:.6f}",
        f"Model:\t{result.model_note}",
        "",
        "Chronologický přehled (hodnoty elementů jsou oskulační v perihelu)",
        "Typ\tDatum UTC\tNávrat / planeta\tq [AU]\te\ta [AU]\tP [roky]\t1/a [AU^-1]"
        "\ti [deg]\tOmega [deg]\tomega [deg]\tVzdálenost [AU]\tHillovy poloměry"
        "\tv rel. [km/s]\tDelta(1/a) [AU^-1]",
    ]
    events: list[tuple[float, int, EvolutionReturn | PlanetEncounter]] = []
    events.extend((item.epoch_jd, 0, item) for item in result.returns)
    events.extend((item.epoch_jd, 1, item) for item in result.encounters)
    for _epoch, _event_type, item in sorted(events, key=lambda row: (row[0], row[1])):
        if isinstance(item, EvolutionReturn):
            period = "" if item.period_years is None else f"{item.period_years:.8f}"
            lines.append(
                "PERIHELIUM\t"
                f"{_format_jd_utc(item.epoch_jd)}\t{item.index:+d}\t"
                f"{item.perihelion_au:.9f}\t{item.eccentricity:.10f}\t"
                f"{item.semi_major_axis_au:.9f}\t{period}\t{item.reciprocal_a:.10e}\t"
                f"{item.inclination_deg:.7f}\t{item.node_deg:.7f}\t"
                f"{item.arg_peri_deg:.7f}\t\t\t\t"
            )
        else:
            delta = "" if item.delta_reciprocal_a is None else f"{item.delta_reciprocal_a:+.10e}"
            lines.append(
                "SETKÁNÍ\t"
                f"{_format_jd_utc(item.epoch_jd)}\t{PLANET_LABELS.get(item.planet, item.planet)}\t"
                "\t\t\t\t\t\t\t\t"
                f"{item.distance_au:.9f}\t{item.hill_radii:.6f}\t"
                f"{item.relative_velocity_kms:.5f}\t{delta}"
            )
    if not events:
        lines.append("Žádné perihelové průchody ani planetární setkání nebyly nalezeny.")
    lines.extend(("", "Vytvořeno programem Orbit Solver 1.0 — Martin Mašek"))
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
            return f"# Návrat {item.index:+d}: kalendářní datum je mimo podporovaný rozsah."
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
        f"# Orbit Solver 1.0 — vývoj dráhy: {name}",
        f"# Rozsah UTC: {_format_jd_utc(result.start_jd)} až {_format_jd_utc(result.end_jd)}",
        "# Elementy jsou oskulační v okamžiku příslušného perihelu.",
        "# Řádky začínající # jsou komentáře a planetární setkání.",
        f"# {result.model_note}",
    ]
    events: list[tuple[float, int, EvolutionReturn | PlanetEncounter]] = []
    events.extend((item.epoch_jd, 0, item) for item in result.returns)
    events.extend((item.epoch_jd, 1, item) for item in result.encounters)
    for _epoch, _kind, item in sorted(events, key=lambda row: (row[0], row[1])):
        if isinstance(item, EvolutionReturn):
            lines.append(f"# Návrat {item.index:+d} — {_format_jd_utc(item.epoch_jd)} UTC")
            lines.append(_mpc_return_line(solution, item))
        else:
            delta = (
                "neuvedeno" if item.delta_reciprocal_a is None
                else f"{item.delta_reciprocal_a:+.6e} AU^-1"
            )
            lines.append(
                f"# SETKÁNÍ {_format_jd_utc(item.epoch_jd)} UTC — "
                f"{PLANET_LABELS.get(item.planet, item.planet)}; "
                f"d={item.distance_au:.6f} AU = {item.hill_radii:.3f} Hillova poloměru; "
                f"v_rel={item.relative_velocity_kms:.2f} km/s; Delta(1/a)={delta}"
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
            "Vývoj dráhy vyžaduje kartézský stav z MPC nebo z vlastního fitu."
        )
    return StateVector(
        solution.epoch_jd,
        np.asarray(solution.position_au, dtype=float),
        np.asarray(solution.velocity_au_per_day, dtype=float),
    )


def evolution_span(
    solution: OrbitSolution, mode: str, past: int, future: int,
) -> tuple[float, float]:
    """Translate the UI range choice to absolute Julian dates."""
    if past < 0 or future < 0 or (past == 0 and future == 0):
        raise ValueError("Rozsah integrace musí být větší než nula.")
    if mode == "years":
        return (
            solution.epoch_jd - float(past) * JULIAN_YEAR_DAYS,
            solution.epoch_jd + float(future) * JULIAN_YEAR_DAYS,
        )
    if mode != "returns":
        raise ValueError("Neznámý způsob zadání rozsahu integrace.")
    if not (solution.eccentricity < 1.0 and solution.semi_major_axis_au > 0.0):
        raise ValueError("Počet návratů lze zadat jen pro uzavřenou eliptickou dráhu.")
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


def calculate_orbit_evolution(
    solution: OrbitSolution, mode: str, past: int, future: int,
    *, maximum_samples: int = 80001,
) -> EvolutionResult:
    """Integrate an orbit and extract perihelion returns and close encounters.

    This first implementation deliberately depends only on the application's
    current planet provider.  Replacing PLANETS with a DE440-backed provider
    later does not change the event/result API or the user interface.
    """
    initial = _solution_state(solution)
    start_jd, end_jd = evolution_span(solution, mode, past, future)
    span_days = end_jd - start_jd
    if span_days <= 0.0:
        raise ValueError("Neplatný rozsah integrace.")

    if solution.eccentricity < 1.0 and solution.semi_major_axis_au > 0.0:
        period_days = 2.0 * np.pi * solution.semi_major_axis_au ** 1.5 / np.sqrt(MU_SUN)
        desired_step = float(np.clip(period_days / 360.0, .25, 5.0))
    else:
        desired_step = 2.0
    sample_count = min(maximum_samples, max(401, int(np.ceil(span_days / desired_step)) + 1))
    sample_epochs = np.linspace(start_jd, end_jd, sample_count)
    sample_step = float(sample_epochs[1] - sample_epochs[0])
    engine = NBodyIntegrator(
        PLANETS, relative_tolerance=2e-11, absolute_tolerance=2e-13,
        nongravitational_parameters=solution.nongravitational_parameters,
    )
    states = engine.propagate(initial, sample_epochs)

    radius = np.linalg.norm(states[:, :3], axis=1)
    minima = np.flatnonzero((radius[1:-1] < radius[:-2]) & (radius[1:-1] <= radius[2:])) + 1
    refined_epochs = [_parabolic_vertex_time(sample_epochs, radius, int(index)) for index in minima]
    if refined_epochs:
        refined_states = engine.propagate(initial, np.asarray(refined_epochs, dtype=float))
        zero_position = int(np.argmin(np.abs(np.asarray(refined_epochs) - solution.epoch_jd)))
        returns = tuple(
            _return_from_state(position - zero_position, epoch, state)
            for position, (epoch, state) in enumerate(zip(refined_epochs, refined_states))
        )
        if mode == "returns":
            returns = tuple(item for item in returns if -past <= item.index <= future)
    else:
        returns = ()

    encounters: list[PlanetEncounter] = []
    for planet, hill_radius in HILL_RADII_AU.items():
        planet_positions = np.stack([
            PLANETS.position(planet, float(epoch)) for epoch in sample_epochs
        ])
        distances = np.linalg.norm(states[:, :3] - planet_positions, axis=1)
        candidate_indices = np.flatnonzero(
            (distances[1:-1] < distances[:-2])
            & (distances[1:-1] <= distances[2:])
            & (distances[1:-1] < 3.0 * hill_radius)
        ) + 1
        for sample_index in candidate_indices:
            encounter_jd = _parabolic_vertex_time(sample_epochs, distances, int(sample_index))
            influence_window = max(30.0, 3.0 * sample_step)
            evaluation = np.array([
                encounter_jd - influence_window, encounter_jd,
                encounter_jd + influence_window,
            ])
            encounter_states = engine.propagate(initial, evaluation)
            planet_at_encounter = PLANETS.position(planet, encounter_jd)
            separation = float(np.linalg.norm(
                encounter_states[1, :3] - planet_at_encounter
            ))
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
    return EvolutionResult(
        returns=returns, encounters=tuple(encounters), start_jd=start_jd, end_jd=end_jd,
        sample_step_days=sample_step,
        model_note=(
            "Orientační výpočet: planetární perturbace a A1/A2/A3 jsou zahrnuty, "
            f"polohy planet používají model {PLANETS.model_name}."
        ),
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
