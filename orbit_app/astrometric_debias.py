"""Optional Eggl et al. (2020) catalogue debiasing for optical astrometry.

The external ``bias.dat`` table is not bundled.  When it is placed in
``orbit-data/debias_2018/bias.dat`` (or the directory selected by
``ORBIT_SOLVER_DATA_DIR``), corrections are read lazily and applied in the
same HEALPix RING ordering used by the JPL data release.  Returned values are
the RA*cos(Dec), Dec corrections in arcseconds that must be subtracted from an
observation.  Observation epochs are UTC/UT decimal years; the proper-motion
columns are milliarcseconds/year relative to J2000.
"""
from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

import numpy as np

from .domain import Observation

DEBIAS_DIRECTORY = "debias_2018"
DEBIAS_FILENAME = "bias.dat"
DEBIAS_NSIDE = 64

_CATALOGS = (
    "USNOA1", "USNOSA1", "USNOA2", "USNOSA2", "UCAC1", "TYC2",
    "GSC1.1", "GSC1.2", "ACT", "GSCACT", "SDSS8", "USNOB1", "PPM",
    "UCAC4", "UCAC2", "PPMXL", "UCAC3", "NOMAD", "CMC14", "2MASS",
    "SDSS7", "CMC15", "SSTRC4", "URAT1", "GAIA1", "UCAC5",
)
_CATALOG_INDEX = {
    name.replace(".", "").replace("-", ""): index
    for index, name in enumerate(_CATALOGS)
}
# Catalogues tied directly to ICRF/Gaia are intentionally not corrected by
# this pre-Gaia map, matching the JPL reference implementation.
_UNCORRECTED = {"TYC2", "ACT", "GAIA1", "UCAC5"}


def debias_path() -> Path:
    override = os.environ.get("ORBIT_SOLVER_DATA_DIR")
    root = Path(override).expanduser() if override else Path(__file__).resolve().parent.parent / "orbit-data"
    return root / DEBIAS_DIRECTORY / DEBIAS_FILENAME


@lru_cache(maxsize=1)
def _bias_table() -> np.ndarray | None:
    path = debias_path()
    if not path.is_file():
        return None
    try:
        values = np.loadtxt(path, comments="!", dtype=float)
    except (OSError, ValueError):
        return None
    expected_pixels = 12 * DEBIAS_NSIDE * DEBIAS_NSIDE
    if values.shape != (expected_pixels, len(_CATALOGS) * 4):
        return None
    return values.reshape(expected_pixels, len(_CATALOGS), 4)


def _catalog_name(observation: Observation) -> str:
    lowered = {str(key).casefold(): value for key, value in observation.raw.items()}
    value = observation.catalog or lowered.get("astcat") or lowered.get("mpc_catalog") or ""
    return str(value).strip().upper().replace(".", "").replace("-", "")


def healpix_ring_index(ra_deg: float, dec_deg: float, nside: int = DEBIAS_NSIDE) -> int:
    """Zero-based HEALPix RING pixel containing an ICRS direction."""
    phi = np.radians(float(ra_deg) % 360.0)
    z = float(np.sin(np.radians(float(dec_deg))))
    za = abs(z)
    tt = phi / (0.5 * np.pi)
    npix = 12 * nside * nside
    ncap = 2 * nside * (nside - 1)
    if za <= 2.0 / 3.0:
        jp = int(nside * (0.5 + tt - 0.75 * z))
        jm = int(nside * (0.5 + tt + 0.75 * z))
        ring = nside + 1 + jp - jm
        shift = 1 - (ring & 1)
        longitude = (jp + jm - nside + shift + 1) // 2
        longitude %= 4 * nside
        return ncap + (ring - 1) * 4 * nside + longitude
    fractional = tt - np.floor(tt)
    scale = nside * np.sqrt(3.0 * (1.0 - za))
    jp = int(fractional * scale)
    jm = int((1.0 - fractional) * scale)
    ring = jp + jm + 1
    longitude = min(int(tt * ring), 4 * ring - 1)
    if z > 0.0:
        return 2 * ring * (ring - 1) + longitude
    return npix - 2 * ring * (ring + 1) + longitude


def catalogue_debias_arcsec(
    observation: Observation, ra_deg: float, dec_deg: float, decimal_year: float,
) -> tuple[float, float]:
    """Correction to subtract from RA*cos(Dec), Dec, in arcseconds."""
    catalog = _catalog_name(observation)
    if catalog in _UNCORRECTED:
        return 0.0, 0.0
    index = _CATALOG_INDEX.get(catalog)
    table = _bias_table()
    if index is None or table is None:
        return 0.0, 0.0
    row = table[healpix_ring_index(ra_deg, dec_deg), index]
    years = float(decimal_year) - 2000.0
    return float(row[0] + years * row[2] / 1000.0), float(row[1] + years * row[3] / 1000.0)


def debias_available() -> bool:
    return _bias_table() is not None
