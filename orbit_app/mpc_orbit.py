from __future__ import annotations

import re

import numpy as np

from .domain import OrbitSolution, is_comet_designation
from .orbit_engine import MU_SUN


class OrbitParseError(ValueError):
    pass


_HALF_MONTH_ORDER = {
    letter: index for index, letter in enumerate("ABCDEFGHJKLMNOPQRSTUVWXY")
}
_SEQUENCE_LETTER_ORDER = {
    letter: index for index, letter in enumerate("ABCDEFGHJKLMNOPQRSTUVWXYZ")
}


def _discovery_sort_key(designation: str) -> tuple[int, int, int, int, str]:
    """Sort MPC provisional designations in approximate discovery order."""
    normalized = designation.strip().upper()
    match = re.search(r"(?<!\d)(\d{4})\s+([A-Z])([A-Z]?)(\d*)", normalized)
    if match is None:
        return (1, 9999, 99, 0, normalized)
    year = int(match.group(1))
    half_month = _HALF_MONTH_ORDER.get(match.group(2), 99)
    second_letter = match.group(3)
    cycle = int(match.group(4) or 0)
    if second_letter:
        # Asteroid-style designations cycle through 25 letters (I omitted).
        sequence = cycle * 25 + _SEQUENCE_LETTER_ORDER.get(second_letter, 99)
    else:
        # Comet designations use the number directly within a half-month.
        sequence = cycle
    return (0, year, half_month, sequence, normalized)


def mpc_display_name(designation_data: object, fallback: str) -> str:
    """Build the familiar MPC caption without changing the orbital identifier."""
    if not isinstance(designation_data, dict):
        return fallback
    name = str(designation_data.get("name") or "").strip()
    permanent = str(designation_data.get("permid") or "").strip()
    primary = str(
        designation_data.get("unpacked_primary_provisional_designation") or ""
    ).strip()
    iau_designation = str(designation_data.get("iau_designation") or "").strip()

    permanent_comet = bool(re.fullmatch(r"\d+[PD]", permanent, re.I))
    is_comet = permanent_comet or "/" in primary
    if permanent_comet and name:
        # MPC can retain the historical discovery designation (for example
        # P/1818 W1) as the primary provisional designation.  Once a comet has
        # a permanent periodic number, its familiar canonical caption is
        # 2P/Encke rather than the designation of one particular apparition.
        return f"{permanent.upper()}/{name}"
    canonical = (
        permanent.upper() if permanent_comet else primary if is_comet else
        iau_designation or (f"({permanent})" if permanent else "") or primary or fallback
    )
    if name:
        if is_comet:
            return f"{canonical} ({name})"
        number = iau_designation or (f"({permanent})" if permanent else "")
        return f"{number} {name}".strip()
    return canonical or fallback


def mpc_identifier_choices(identifier_data: object, fallback: str) -> list[tuple[str, str]]:
    """Return ``(caption, query designation)`` choices from identifier output."""
    if not isinstance(identifier_data, dict):
        return []
    if int(identifier_data.get("found") or 0) == 1:
        primary = str(
            identifier_data.get("unpacked_primary_provisional_designation") or ""
        ).strip()
        permanent = str(identifier_data.get("permid") or "").strip()
        query = permanent or primary or str(identifier_data.get("orbfit_name") or fallback)
        return [(mpc_display_name(identifier_data, fallback), query)]

    choices: list[tuple[tuple[int, int, int, int, str], str, str]] = []
    rows = identifier_data.get("disambiguation_list") or []
    for row in rows:
        if not isinstance(row, dict):
            continue
        primary = str(row.get("unpacked_primary_provisional_designation") or "").strip()
        iau_designation = str(row.get("iau_designation") or "").strip()
        permanent = str(row.get("permid") or "").strip()
        name = str(row.get("name") or "").strip()
        group = str(row.get("group") or "").strip()
        is_comet = "/" in primary or group.casefold() == "comets"
        if is_comet:
            caption = mpc_display_name(row, fallback)
            kind = "kometa"
        else:
            number = f"({permanent})" if permanent else primary
            caption = f"{number} {name}".strip()
            kind = "planetka"
        query = permanent or primary
        if query:
            discovery_designation = primary or iau_designation
            choices.append((
                _discovery_sort_key(discovery_designation),
                f"{caption} — {kind}", query,
            ))
    choices.sort(key=lambda item: (item[0], item[1].casefold()))
    return [(caption, query) for _, caption, query in choices]


def _coefficient_map(section: dict) -> dict[str, float]:
    names = section.get("coefficient_names", [])
    values = section.get("coefficient_values", [])
    return {str(name).lower(): float(value) for name, value in zip(names, values)}


def cartesian_to_elements(
    position: np.ndarray, velocity: np.ndarray, mu: float = MU_SUN,
) -> tuple[float, float, float, float, float, float]:
    r = np.asarray(position, dtype=float)
    v = np.asarray(velocity, dtype=float)
    radius = np.linalg.norm(r)
    speed2 = float(np.dot(v, v))
    h = np.cross(r, v)
    h_norm = np.linalg.norm(h)
    node_vector = np.cross(np.array([0.0, 0.0, 1.0]), h)
    node_norm = np.linalg.norm(node_vector)
    if not np.isfinite(mu) or mu <= 0.0:
        raise OrbitParseError("Gravitační parametr středu musí být kladný.")
    e_vector = ((speed2 - mu / radius) * r - np.dot(r, v) * v) / mu
    eccentricity = float(np.linalg.norm(e_vector))
    energy = speed2 / 2.0 - mu / radius
    semi_major = -mu / (2.0 * energy)
    if not np.isfinite(semi_major) or eccentricity < 0:
        raise OrbitParseError("Kartézský stav nevede na použitelnou kuželosečku.")
    inclination = np.degrees(np.arccos(np.clip(h[2] / h_norm, -1.0, 1.0)))
    node = np.degrees(np.arctan2(node_vector[1], node_vector[0])) % 360.0 if node_norm else 0.0
    if node_norm and eccentricity > 1e-12:
        arg_peri = np.degrees(np.arccos(np.clip(np.dot(node_vector, e_vector) / (node_norm * eccentricity), -1, 1)))
        if e_vector[2] < 0:
            arg_peri = 360.0 - arg_peri
    else:
        arg_peri = 0.0
    true_anomaly = np.arccos(np.clip(np.dot(e_vector, r) / (eccentricity * radius), -1, 1)) if eccentricity > 1e-12 else 0.0
    outbound = np.dot(r, v) >= 0
    if eccentricity < 1.0:
        if not outbound:
            true_anomaly = 2.0 * np.pi - true_anomaly
        eccentric_anomaly = 2.0 * np.arctan2(
            np.sqrt(1.0 - eccentricity) * np.sin(true_anomaly / 2.0),
            np.sqrt(1.0 + eccentricity) * np.cos(true_anomaly / 2.0),
        )
        mean_anomaly = np.degrees(eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly)) % 360.0
    elif eccentricity > 1.0:
        if not outbound:
            true_anomaly = -true_anomaly
        argument = np.sqrt((eccentricity - 1.0) / (eccentricity + 1.0)) * np.tan(true_anomaly / 2.0)
        argument = np.clip(argument, -1.0 + 1e-14, 1.0 - 1e-14)
        hyperbolic_anomaly = 2.0 * np.arctanh(argument)
        mean_anomaly = np.degrees(eccentricity * np.sinh(hyperbolic_anomaly) - hyperbolic_anomaly)
    else:
        raise OrbitParseError("Přesně parabolická dráha vyžaduje perihelový parametr.")
    return semi_major, eccentricity, inclination, node, arg_peri, mean_anomaly


def parse_mpc_orbit(body: object, designation: str) -> OrbitSolution:
    try:
        orbit = body[0]["mpc_orb"][0]  # type: ignore[index]
        car = _coefficient_map(orbit["CAR"])
        def coefficient(*aliases: str) -> float:
            for alias in aliases:
                if alias in car:
                    return car[alias]
            raise KeyError(aliases[0])
        values = [
            coefficient("x"), coefficient("y"), coefficient("z"),
            coefficient("vx", "dx", "xdot", "x_dot"),
            coefficient("vy", "dy", "ydot", "y_dot"),
            coefficient("vz", "dz", "zdot", "z_dot"),
        ]
        epoch = float(orbit["epoch_data"]["epoch"]) + 2400000.5
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise OrbitParseError("MPC nevrátil použitelné kartézské elementy dráhy.") from exc
    elements = cartesian_to_elements(np.array(values[:3]), np.array(values[3:]))
    magnitude = orbit.get("magnitude_data", {})
    display_name = mpc_display_name(orbit.get("designation_data"), designation)
    comet = is_comet_designation(designation) or is_comet_designation(display_name)
    h_value = magnitude.get("H", orbit.get("H", 18.0))
    g_value = magnitude.get("G", orbit.get("G", 0.15))
    covariance_values = orbit.get("CAR", {}).get("covariance", {})
    nongravitational = {
        name.upper(): float(car[name])
        for name in ("a1", "a2", "a3", "dt")
        if name in car and np.isfinite(car[name])
    }
    covariance = None
    if isinstance(covariance_values, dict):
        matrix = np.zeros((6, 6), dtype=float)
        complete = True
        for row in range(6):
            for column in range(row, 6):
                value = covariance_values.get(f"cov{row}{column}")
                if value is None:
                    complete = False
                    break
                matrix[row, column] = matrix[column, row] = float(value)
            if not complete:
                break
        if complete and np.all(np.isfinite(matrix)):
            covariance = tuple(tuple(float(value) for value in row) for row in matrix)
    return OrbitSolution(
        designation=designation,
        epoch_jd=epoch,
        semi_major_axis_au=elements[0],
        eccentricity=elements[1],
        inclination_deg=elements[2],
        node_deg=elements[3],
        arg_peri_deg=elements[4],
        mean_anomaly_deg=elements[5],
        absolute_magnitude=float(h_value) if h_value is not None else 18.0,
        slope_g=float(g_value) if g_value is not None else 0.15,
        photometric_exponent=4.0 if comet else 2.0,
        phase_correction=not comet,
        provisional=False,
        position_au=tuple(float(value) for value in values[:3]),
        velocity_au_per_day=tuple(float(value) for value in values[3:]),
        state_covariance=covariance,
        display_name=display_name,
        nongravitational_parameters=nongravitational or None,
    )


