from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.coordinates import (
    AltAz, EarthLocation, GCRS, SkyCoord, get_body_barycentric, get_sun,
    solar_system_ephemeris,
)
from astropy.time import Time
from astropy.utils import iers
from astropy.utils.exceptions import AstropyWarning
import astropy.units as u
from erfa import ErfaWarning

from .i18n import tr
from .planetary import PLANETS


# MPC parallax constants are expressed in units of the Earth's equatorial radius.
EARTH_EQUATORIAL_RADIUS_M = 6_378_137.0

# The application must remain usable offline.  Astropy ships an IERS table which
# is accurate enough for astrometric topocentric corrections in this use case.
iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"


@lru_cache(maxsize=1)
def observatory_catalog() -> dict[str, dict[str, object]]:
    path = Path(__file__).with_name("data") / "obscodes.json"
    with path.open("r", encoding="utf-8") as handle:
        body = json.load(handle)
    if isinstance(body, list):
        return {
            str(row.get("Code") or row.get("code") or "").strip(): row
            for row in body if isinstance(row, dict)
        }
    if isinstance(body, dict):
        rows = body.get("data", body)
        if isinstance(rows, list):
            return {
                str(row.get("Code") or row.get("code") or "").strip(): row
                for row in rows if isinstance(row, dict)
            }
        return {str(key).strip(): value for key, value in rows.items() if isinstance(value, dict)}
    return {}


def _number(row: dict[str, object], *names: str) -> float | None:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    return None


@lru_cache(maxsize=4096)
def observatory_location(code: str) -> EarthLocation | None:
    row = observatory_catalog().get((code or "500").strip())
    if not row:
        return None
    longitude = _number(row, "Longitude", "longitude", "Longitude (deg)", "long")
    rho_cos = _number(row, "cos", "rho_cos", "RhoCosPhi", "rhocosphi")
    rho_sin = _number(row, "sin", "rho_sin", "RhoSinPhi", "rhosinphi")
    if longitude is None or rho_cos is None or rho_sin is None:
        return None
    angle = np.radians(longitude)
    return EarthLocation.from_geocentric(
        EARTH_EQUATORIAL_RADIUS_M * rho_cos * np.cos(angle),
        EARTH_EQUATORIAL_RADIUS_M * rho_cos * np.sin(angle),
        EARTH_EQUATORIAL_RADIUS_M * rho_sin,
        unit=u.m,
    )


def observatory_details(code: str) -> dict[str, object] | None:
    station_code = (code or "500").strip()
    row = observatory_catalog().get(station_code)
    location = observatory_location(station_code)
    if not row:
        return None
    result: dict[str, object] = {
        "code": station_code,
        "name": row.get("name_utf8") or row.get("name") or tr("Neznámá observatoř"),
    }
    if location is not None:
        result.update({
            "longitude_deg": float(location.lon.to_value(u.deg)),
            "latitude_deg": float(location.lat.to_value(u.deg)),
            "height_m": float(location.height.to_value(u.m)),
        })
    return result


def observation_geometry(
    code: str, instant: datetime, ra_deg: float, dec_deg: float,
) -> dict[str, float] | None:
    location = observatory_location(code)
    if location is None or (code or "500").strip() == "500":
        return None
    obstime = Time(instant, scale="utc")
    target = SkyCoord(ra=ra_deg * u.deg, dec=dec_deg * u.deg, frame="icrs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        altaz = AltAz(obstime=obstime, location=location)
        target_altitude = target.transform_to(altaz).alt.to_value(u.deg)
        sun = get_sun(obstime)
        sun_altitude = sun.transform_to(altaz).alt.to_value(u.deg)
        target_gcrs = target.transform_to(GCRS(obstime=obstime))
        elongation = target_gcrs.separation(sun).to_value(u.deg)
    return {
        "target_altitude_deg": float(target_altitude),
        "sun_altitude_deg": float(sun_altitude),
        "elongation_deg": float(elongation),
    }


def observation_altaz_series(
    code: str, times: list[datetime], ra_deg: np.ndarray, dec_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Vectorised azimuth/object-altitude/Sun-altitude for an MPC station."""
    location = observatory_location(code)
    if location is None or (code or "500").strip() == "500":
        return None
    obstime = Time(times, scale="utc")
    target = SkyCoord(ra=np.asarray(ra_deg) * u.deg, dec=np.asarray(dec_deg) * u.deg, frame="icrs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        altaz_frame = AltAz(obstime=obstime, location=location)
        target_altaz = target.transform_to(altaz_frame)
        sun_altitude = get_sun(obstime).transform_to(altaz_frame).alt.to_value(u.deg)
    return (
        np.asarray(target_altaz.az.to_value(u.deg), dtype=float),
        np.asarray(target_altaz.alt.to_value(u.deg), dtype=float),
        np.asarray(sun_altitude, dtype=float),
    )


def equatorial_to_horizontal(
    code: str, times: datetime | list[datetime],
    ra_deg: np.ndarray | float, dec_deg: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Convert ICRS RA/Dec to geometric azimuth/altitude for an MPC station.

    The vectorised Astropy transform includes precession, nutation and annual
    aberration.  Atmospheric refraction is intentionally omitted because its
    value would require local pressure and temperature.  ``times`` may contain
    one instant (for an entire star catalogue) or one instant per coordinate.
    """
    station_code = (code or "500").strip().upper()
    location = observatory_location(station_code)
    if location is None or station_code == "500":
        return None
    ra = np.atleast_1d(np.asarray(ra_deg, dtype=float))
    dec = np.atleast_1d(np.asarray(dec_deg, dtype=float))
    if ra.shape != dec.shape:
        raise ValueError(tr("Pole RA a Dec nemají stejný rozměr."))
    time_values = [times] if isinstance(times, datetime) else list(times)
    if len(time_values) not in (1, ra.size):
        raise ValueError(tr("Počet časů neodpovídá počtu souřadnic."))
    obstime = Time(time_values[0] if len(time_values) == 1 else time_values, scale="utc")
    target = SkyCoord(ra=ra * u.deg, dec=dec * u.deg, frame="icrs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        horizontal = target.transform_to(AltAz(obstime=obstime, location=location))
    return (
        np.atleast_1d(np.asarray(horizontal.az.to_value(u.deg), dtype=float)),
        np.atleast_1d(np.asarray(horizontal.alt.to_value(u.deg), dtype=float)),
    )


def station_offsets_gcrs(stations: list[str], times: list[datetime]) -> np.ndarray:
    """Return station vectors from the Earth's centre in GCRS/ICRS axes, in AU."""
    if len(stations) != len(times):
        raise ValueError(tr("Počet stanic neodpovídá počtu časů pozorování."))
    result = np.zeros((len(times), 3), dtype=float)
    active_indices: list[int] = []
    terrestrial_xyz: list[tuple[float, float, float]] = []
    for index, station in enumerate(stations):
        code = (station or "500").strip()
        if code == "500":
            continue
        location = observatory_location(code)
        if location is None:
            continue
        x, y, z = location.to_geocentric()
        active_indices.append(index)
        terrestrial_xyz.append((
            float(x.to_value(u.m)), float(y.to_value(u.m)), float(z.to_value(u.m)),
        ))

    if active_indices:
        # Transform every station/time pair in one vectorised Astropy call.
        # Grouping by observatory looked tidy, but a large historical comet
        # archive can contain hundreds of station codes and made this report
        # step substantially slower than the orbit correction itself.
        xyz = np.asarray(terrestrial_xyz, dtype=float)
        locations = EarthLocation.from_geocentric(
            xyz[:, 0] * u.m, xyz[:, 1] * u.m, xyz[:, 2] * u.m,
        )
        with warnings.catch_warnings():
            # Future dates can be just beyond the bundled IERS table.  Astropy
            # then uses UT1-UTC=0 and mean polar motion, which is still far more
            # accurate than omitting the topocentric correction altogether.
            warnings.simplefilter("ignore", AstropyWarning)
            warnings.simplefilter("ignore", ErfaWarning)
            obstime = Time([times[index] for index in active_indices], scale="utc")
            gcrs = locations.get_gcrs(obstime=obstime).cartesian.xyz.to_value(u.au).T
        result[np.asarray(active_indices)] = gcrs
    return result


def geodetic_offsets_gcrs(
    coordinates: list[tuple[float, float, float]], times: list[datetime],
) -> np.ndarray:
    """Return per-observation WGS84 locations in geocentric celestial axes."""
    if len(coordinates) != len(times):
        raise ValueError(tr("Počet souřadnic neodpovídá počtu časů pozorování."))
    if not coordinates:
        return np.empty((0, 3), dtype=float)
    longitude, latitude, height = np.asarray(coordinates, dtype=float).T
    locations = EarthLocation.from_geodetic(
        longitude * u.deg, latitude * u.deg, height * u.m
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AstropyWarning)
        warnings.simplefilter("ignore", ErfaWarning)
        obstime = Time(times, scale="utc")
        return np.asarray(
            locations.get_gcrs(obstime=obstime).cartesian.xyz.to_value(u.au).T,
            dtype=float,
        )


# UTC is defined by ERFA only from 1960.  Earlier timestamps are Universal
# Time, for which TT - UT must come from the historical Earth-rotation record.
_UTC_START = datetime(1960, 1, 1, tzinfo=timezone.utc)


def delta_t_seconds(year: float) -> float:
    """Return TT - UT in seconds for a decimal year before 1960.

    Polynomial expressions of Espenak & Meeus (NASA Five Millennium Canon of
    Solar Eclipses, 2006).  They follow the Morrison & Stephenson
    reconstruction to about a second in the telescopic era (after 1700).
    """
    y = float(year)
    if y < 500.0:
        u = y / 100.0
        return (10583.6 - 1014.41 * u + 33.78311 * u**2 - 5.952053 * u**3
                - 0.1798452 * u**4 + 0.022174192 * u**5 + 0.0090316521 * u**6)
    if y < 1600.0:
        u = (y - 1000.0) / 100.0
        return (1574.2 - 556.01 * u + 71.23472 * u**2 + 0.319781 * u**3
                - 0.8503463 * u**4 - 0.005050998 * u**5 + 0.0083572073 * u**6)
    if y < 1700.0:
        t = y - 1600.0
        return 120.0 - 0.9808 * t - 0.01532 * t**2 + t**3 / 7129.0
    if y < 1800.0:
        t = y - 1700.0
        return (8.83 + 0.1603 * t - 0.0059285 * t**2 + 0.00013336 * t**3
                - t**4 / 1174000.0)
    if y < 1860.0:
        t = y - 1800.0
        return (13.72 - 0.332447 * t + 0.0068612 * t**2 + 0.0041116 * t**3
                - 0.00037436 * t**4 + 0.0000121272 * t**5
                - 0.0000001699 * t**6 + 0.000000000875 * t**7)
    if y < 1900.0:
        t = y - 1860.0
        return (7.62 + 0.5737 * t - 0.251754 * t**2 + 0.01680668 * t**3
                - 0.0004473624 * t**4 + t**5 / 233174.0)
    if y < 1920.0:
        t = y - 1900.0
        return (-2.79 + 1.494119 * t - 0.0598939 * t**2 + 0.0061966 * t**3
                - 0.000197 * t**4)
    if y < 1941.0:
        t = y - 1920.0
        return 21.20 + 0.84493 * t - 0.076100 * t**2 + 0.0020936 * t**3
    t = y - 1950.0
    return 29.07 + 0.407 * t - t**2 / 233.0 + t**3 / 2547.0


def dynamical_jd(times: list[datetime]) -> np.ndarray:
    """Convert observation instants to the TDB Julian dates used by dynamics.

    From 1960 the instants are UTC and ERFA supplies TT - UTC (including the
    1960-1972 rubber-second era).  Older MPC timestamps are Universal Time;
    ERFA's pre-1960 fallback would silently use TT - UT = 32.184 s, which is
    about 39 s wrong in 1892.  Those instants use :func:`delta_t_seconds`.
    """
    values = list(times)
    if not values:
        return np.empty(0, dtype=float)
    aware = [
        value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        for value in values
    ]
    result = np.empty(len(aware), dtype=float)
    modern = [index for index, value in enumerate(aware) if value >= _UTC_START]
    historical = [index for index, value in enumerate(aware) if value < _UTC_START]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ErfaWarning)
        if modern:
            result[modern] = Time(
                [aware[index] for index in modern], scale="utc",
            ).tdb.jd
        if historical:
            reference = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)
            ut_jd = np.asarray([
                2451545.0 + (aware[index] - reference).total_seconds() / 86400.0
                for index in historical
            ])
            years = 2000.0 + (ut_jd - 2451545.0) / 365.25
            tt_jd = ut_jd + np.asarray([delta_t_seconds(year) for year in years]) / 86400.0
            result[historical] = Time(tt_jd, format="jd", scale="tt").tdb.jd
    return result


def earth_heliocentric_ecliptic(jd_tdb: np.ndarray) -> np.ndarray:
    """Earth position from active DE440, or Astropy's offline fallback."""
    epochs = np.atleast_1d(np.asarray(jd_tdb, dtype=float))
    if PLANETS.jpl is not None:
        return np.vstack([PLANETS.position("earth", float(jd)) for jd in epochs])
    times = Time(epochs, format="jd", scale="tdb")
    with solar_system_ephemeris.set("builtin"):
        earth = get_body_barycentric("earth", times).xyz.to_value(u.au).T
        sun = get_body_barycentric("sun", times).xyz.to_value(u.au).T
    equatorial = earth - sun
    eps = np.radians(23.4392911)
    return np.column_stack((
        equatorial[:, 0],
        equatorial[:, 1] * np.cos(eps) + equatorial[:, 2] * np.sin(eps),
        -equatorial[:, 1] * np.sin(eps) + equatorial[:, 2] * np.cos(eps),
    ))
