from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from .domain import Observation


AU_KM = 149_597_870.7


class MpcError(RuntimeError):
    pass


def parse_mpc80(text: str, fallback_designation: str = "") -> list[Observation]:
    """Parse optical astrometry in the MPC 80-column observation format.

    Spacecraft observations occupy two records: an uppercase ``S`` record
    carries the astrometry and a lowercase ``s`` record carries the observer's
    geocentric J2000 equatorial position.  The pair can be out of order, so it
    is joined after all records have been read.
    """
    observations: list[Observation] = []
    satellite_observations: dict[str, tuple[Observation, int]] = {}
    satellite_offsets: dict[str, tuple[tuple[float, float, float], str, int]] = {}
    for line_number, source in enumerate(text.splitlines(), start=1):
        if not source.strip() or source.lstrip().startswith(("#", "!")):
            continue
        line = source.rstrip("\r\n").ljust(80)
        note2 = line[14]
        # Column 13 is the discovery marker.  It is present only on the
        # astrometric ``S`` record (``*``), never on its lowercase ``s``
        # spacecraft-position continuation, and therefore must not be part of
        # the pairing key.  Keep designation, note 1, epoch and station.
        pair_key = line[:12] + line[13:14] + line[15:32] + line[77:80]

        if note2 == "s":
            try:
                unit_code = line[32]
                if unit_code not in ("1", "2"):
                    raise ValueError
                coordinates = tuple(
                    float(line[start:start + 12].replace(" ", ""))
                    for start in (34, 46, 58)
                )
                scale = 1.0 / AU_KM if unit_code == "1" else 1.0
                satellite_offsets[pair_key] = (
                    tuple(value * scale for value in coordinates), source, line_number,
                )
            except (TypeError, ValueError, OverflowError) as exc:
                raise MpcError(
                    f"Řádek {line_number} není platná druhá část satelitního MPC pozorování."
                ) from exc
            continue
        try:
            date_fields = line[15:32].split()
            ra_fields = line[32:44].split()
            dec_fields = line[44:56].split()
            if len(date_fields) != 3 or len(ra_fields) != 3 or len(dec_fields) != 3:
                raise ValueError
            year, month = int(date_fields[0]), int(date_fields[1])
            decimal_day = float(date_fields[2])
            instant = datetime(year, month, 1, tzinfo=timezone.utc) + timedelta(days=decimal_day - 1.0)
            ra_deg = 15.0 * (
                float(ra_fields[0]) + float(ra_fields[1]) / 60.0 + float(ra_fields[2]) / 3600.0
            )
            sign = -1.0 if dec_fields[0].startswith("-") else 1.0
            dec_deg = sign * (
                abs(float(dec_fields[0])) + float(dec_fields[1]) / 60.0 + float(dec_fields[2]) / 3600.0
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise MpcError(f"Řádek {line_number} není platné optické MPC 80-col pozorování.") from exc
        magnitude = None
        try:
            magnitude_text = line[65:70].strip()
            magnitude = float(magnitude_text) if magnitude_text else None
        except ValueError:
            pass
        designation = line[:12].strip() or fallback_designation
        observation = Observation(
            designation=designation,
            time=instant.isoformat().replace("+00:00", "Z"),
            ra_deg=ra_deg,
            dec_deg=dec_deg,
            magnitude=magnitude,
            band=line[70:71].strip(),
            station=line[77:80].strip() or "500",
            raw={"mpc80": source, "mpc_note2": note2},
        )
        observations.append(observation)
        if note2 == "S":
            satellite_observations[pair_key] = (observation, line_number)

    for pair_key, (offset, source, line_number) in satellite_offsets.items():
        paired = satellite_observations.get(pair_key)
        if paired is None:
            raise MpcError(
                f"Řádek {line_number} je poloha družice bez odpovídajícího řádku S s RA/Dec."
            )
        observation, _ = paired
        observation.raw["mpc80_spacecraft"] = source
        observation.raw["observer_position_gcrs_au"] = offset

    for pair_key, (_, line_number) in satellite_observations.items():
        if pair_key not in satellite_offsets:
            raise MpcError(
                f"Satelitní pozorování na řádku {line_number} postrádá druhý řádek s polohou družice."
            )
    if not observations:
        raise MpcError("Text neobsahuje žádné platné MPC 80-col pozorování.")
    return observations


class MpcClient:
    BASE_URL = "https://data.minorplanetcenter.net/api"

    def __init__(self, timeout: float = 30.0) -> None:
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "OrbitSolver/1.0 (+local astronomical application)"}
        )

    def observations(self, designation: str, confirmation_page: bool = False) -> list[Observation]:
        designation = designation.strip()
        if not designation:
            raise MpcError("Zadejte označení objektu.")
        if confirmation_page:
            endpoint = f"{self.BASE_URL}/get-obs-neocp"
            payload = {
                "trksubs": [designation],
                "output_format": ["ADES_DF"],
                "ades_version": "2022",
            }
        else:
            endpoint = f"{self.BASE_URL}/get-obs"
            payload = {
                "desigs": [designation],
                "output_format": ["ADES_DF"],
                "ades_version": "2022",
            }
        try:
            response = self.session.get(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            body = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise MpcError(f"MPC dotaz selhal: {exc}") from exc
        rows = self._extract_rows(body)
        if not rows:
            area = "NEOCP/PCCP" if confirmation_page else "MPC"
            raise MpcError(f"Pro objekt „{designation}“ nebyla v {area} nalezena pozorování.")
        return [self._to_observation(row, designation) for row in rows]

    @staticmethod
    def _extract_rows(body: Any) -> list[dict[str, Any]]:
        if isinstance(body, dict):
            rows = body.get("ADES_DF") or body.get("ades_df") or []
            if isinstance(rows, list):
                return [row for row in rows if isinstance(row, dict)]
        if isinstance(body, list) and body:
            item = body[0]
            if isinstance(item, dict):
                rows = item.get("ADES_DF") or item.get("ades_df") or []
                if isinstance(rows, list):
                    return [row for row in rows if isinstance(row, dict)]
        return []

    @staticmethod
    def _first(row: dict[str, Any], keys: Iterable[str], default: Any = "") -> Any:
        for key in keys:
            value = row.get(key)
            if value not in (None, ""):
                return value
        return default

    @classmethod
    def _to_observation(cls, row: dict[str, Any], fallback: str) -> Observation:
        def number(keys: tuple[str, ...]) -> float | None:
            value = cls._first(row, keys, None)
            try:
                return float(value) if value is not None else None
            except (TypeError, ValueError):
                return None

        raw = dict(row)
        coordinate_system = str(cls._first(row, ("sys", "coordinateSystem"))).upper()
        if coordinate_system in ("ICRF_AU", "ICRF_KM"):
            position = [number((f"pos{axis}",)) for axis in (1, 2, 3)]
            if all(value is not None for value in position):
                scale = 1.0 / AU_KM if coordinate_system == "ICRF_KM" else 1.0
                raw["observer_position_gcrs_au"] = tuple(
                    float(value) * scale for value in position if value is not None
                )
            velocity = [number((f"vel{axis}",)) for axis in (1, 2, 3)]
            if all(value is not None for value in velocity):
                # ADES Cartesian velocities paired with ICRF_KM are km/s;
                # ICRF_AU velocities are AU/day.
                velocity_scale = 86_400.0 / AU_KM if coordinate_system == "ICRF_KM" else 1.0
                raw["observer_velocity_gcrs_au_per_day"] = tuple(
                    float(value) * velocity_scale for value in velocity if value is not None
                )
        elif coordinate_system == "WGS84":
            longitude = number(("pos1",))
            latitude = number(("pos2",))
            height = number(("pos3",))
            if longitude is not None and latitude is not None and height is not None:
                raw["observer_wgs84"] = (longitude, latitude, height)
        return Observation(
            designation=str(cls._first(row, ("permID", "provID", "trkSub"), fallback)),
            time=str(cls._first(row, ("obsTime", "obstime", "obs_time"))),
            ra_deg=number(("ra", "ra_deg")),
            dec_deg=number(("dec", "dec_deg")),
            magnitude=number(("mag", "magnitude")),
            band=str(cls._first(row, ("band", "magBand"))),
            station=str(cls._first(row, ("stn", "station"))),
            catalog=str(cls._first(row, ("astCat", "catalog"))),
            raw=raw,
        )

