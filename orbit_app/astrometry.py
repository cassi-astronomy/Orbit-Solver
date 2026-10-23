"""Astrometric measurement metadata: uncertainties, duplicates and normal places.

Everything here is independent of the dynamical model.  Angles are degrees in
the ICRS/J2000 frame, uncertainties are arcseconds (RA already multiplied by
cos Dec, as in ADES ``rmsRA``), and observation times are UTC (or UT before
1960) ISO strings as stored in :class:`~orbit_app.domain.Observation`.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

import numpy as np

from .domain import Observation

Row = tuple[Observation, datetime, float, float]

# Measurement batches of one station and night share catalogue, timing and
# reduction systematics (Vereš et al. 2017): N > 4 rows are deweighted by
# sqrt(N / 4).
INDEPENDENT_MEASUREMENTS_PER_NIGHT = 4

# Floor of any per-coordinate uncertainty (arcsec).  ADES rmsRA/rmsDec give the
# random part only; catalogue zonal errors, timing and centroiding of a coma
# keep real residuals above this level even for Gaia-reduced CCD astrometry.
ASTROMETRIC_SIGMA_FLOOR_ARCSEC = 0.3

# Two records of one station closer than this in time are re-reductions of the
# same exposure (MPC often publishes a measurement against several catalogues).
DUPLICATE_TIME_TOLERANCE_DAYS = 2e-6
DUPLICATE_POSITION_TOLERANCE_ARCSEC = 30.0

_REFERENCE_EPOCH = datetime(2000, 1, 1, 12, tzinfo=timezone.utc)


def parse_observation_time(value: str) -> datetime:
    """Parse an ISO observation time and return an aware UTC datetime."""
    normalized = value.strip().replace("Z", "+00:00")
    # Python 3.10 accepts only selected ISO fractional-second widths.  ADES
    # legitimately contains values such as ``...51.17Z`` (station 807 in the
    # C/2024 Y1 data), so normalise every fraction to six digits first.
    match = re.match(r"^(.*:\d{2})\.(\d+)([+-]\d{2}:\d{2})$", normalized)
    if match:
        normalized = f"{match.group(1)}.{(match.group(2) + '000000')[:6]}{match.group(3)}"
    parsed = datetime.fromisoformat(normalized)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def seconds_since_j2000(instant: datetime) -> float:
    """Elapsed seconds from J2000 without Unix timestamps (safe before 1970)."""
    return (instant - _REFERENCE_EPOCH).total_seconds()


def instant_from_seconds_since_j2000(seconds: float) -> datetime:
    return _REFERENCE_EPOCH + timedelta(seconds=float(seconds))


def observing_night_key(observation: Observation, instant: datetime) -> tuple[str, object]:
    """Group one station's measurements by astronomical night.

    The night boundary is local mean noon at the station longitude when the
    station is known, so that European and Asian nights are not split at UTC
    midnight-equivalent noon.
    """
    station = (observation.station or "").strip().upper()
    longitude = _station_longitude(station)
    local = instant + timedelta(hours=(longitude / 15.0 if longitude is not None else 0.0))
    return station, (local - timedelta(hours=12)).date()


def _station_longitude(station: str) -> float | None:
    if not station or station == "500":
        return None
    try:
        from .observatories import observatory_catalog
        row = observatory_catalog().get(station)
        value = None if row is None else row.get("longitude", row.get("Longitude"))
        if value in (None, ""):
            return None
        longitude = float(value)
    except (TypeError, ValueError, OSError):
        return None
    return ((longitude + 180.0) % 360.0) - 180.0


# --------------------------------------------------------------------------
# Measurement technique and default uncertainties
# --------------------------------------------------------------------------

_ADES_MODE_TECHNIQUE = {
    "CCD": "ccd", "CMO": "ccd", "VID": "ccd", "PMT": "ccd",
    "PHO": "photographic", "MIC": "micrometer", "MER": "transit",
    "TRA": "transit", "VIS": "visual", "ENC": "encoder", "OCC": "occultation",
}
_MPC_NOTE2_TECHNIQUE = {
    "C": "ccd", "c": "ccd", "B": "ccd", "K": "ccd", "V": "ccd",
    "P": "photographic", "M": "micrometer", "T": "transit",
    "e": "encoder", "E": "occultation", "H": "hipparcos",
}

# Default 1-sigma per coordinate (arcsec) when ADES does not supply rmsRA/rmsDec.
# Historical values follow typical post-fit residuals of re-reduced plates and
# micrometer measurements; the robust fit still identifies individual blunders.
_TECHNIQUE_SIGMA_ARCSEC = {
    "micrometer": 3.0,
    "transit": 2.0,
    "visual": 30.0,
    "encoder": 30.0,
    "hipparcos": 0.1,
}


def measurement_technique(observation: Observation) -> str:
    """Return a coarse measurement technique derived from ADES or MPC80 notes."""
    raw = observation.raw
    mode = str(raw.get("mode") or "").strip().upper()
    if mode in _ADES_MODE_TECHNIQUE:
        return _ADES_MODE_TECHNIQUE[mode]
    note2 = str(raw.get("mpc_note2") or "")
    if note2 in _MPC_NOTE2_TECHNIQUE:
        return _MPC_NOTE2_TECHNIQUE[note2]
    return "unknown"


def _epoch_fallback_sigma(year: int) -> float:
    if year < 1850:
        return 10.0
    if year < 1900:
        return 5.0
    if year < 1950:
        return 3.0
    if year < 1990:
        return 2.0
    if year < 2010:
        return 1.5
    return 1.0


def default_sigma_arcsec(observation: Observation) -> float:
    """Per-coordinate uncertainty from technique and epoch (arcsec)."""
    try:
        year = parse_observation_time(observation.time).year
    except (TypeError, ValueError):
        year = 2000
    technique = measurement_technique(observation)
    if technique in _TECHNIQUE_SIGMA_ARCSEC:
        return _TECHNIQUE_SIGMA_ARCSEC[technique]
    if technique == "photographic":
        return 2.5 if year < 1950 else 1.5
    if technique == "ccd":
        return min(_epoch_fallback_sigma(year), 2.0)
    return _epoch_fallback_sigma(year)


def observation_sigma_arcsec(observation: Observation) -> tuple[float, float]:
    """Return ADES 1-sigma values (RA·cosDec, Dec) or a technique-based default.

    A large part of the historical archive was converted to ADES from legacy
    records and carries no meaningful per-row uncertainty.  The default then
    follows the measurement technique (micrometer, photographic, CCD…) and the
    epoch; a modern CCD record is not trusted merely for being modern because
    robust residual handling can still downweight or reject it.
    """
    fallback = default_sigma_arcsec(observation)
    lowered = {str(key).casefold(): value for key, value in observation.raw.items()}

    def value_for(key: str, quantum_key: str) -> float:
        try:
            value = float(lowered.get(key, fallback))
            value = value if np.isfinite(value) and value > 0.0 else fallback
        except (TypeError, ValueError):
            value = fallback
        # A position published to 1 s of RA (15″) cannot be better than half
        # of its last digit, whatever the nominal technique uncertainty.
        try:
            quantum = float(lowered.get(quantum_key, 0.0) or 0.0)
        except (TypeError, ValueError):
            quantum = 0.0
        floor = max(ASTROMETRIC_SIGMA_FLOOR_ARCSEC, 0.5 * quantum if np.isfinite(quantum) else 0.0)
        return max(value, floor)

    return (
        value_for("rmsra", "mpc_ra_quantum_arcsec"),
        value_for("rmsdec", "mpc_dec_quantum_arcsec"),
    )


def observation_sigmas(rows: list[Row], account_for_batch_correlation: bool = False) -> np.ndarray:
    """Return interleaved (RA, Dec) sigmas, optionally deweighting dense nights."""
    values = np.asarray([
        value for item in rows for value in observation_sigma_arcsec(item[0])
    ], dtype=float)
    if not account_for_batch_correlation or not rows:
        return values
    batch_sizes: dict[tuple[str, object], int] = {}
    batch_keys: list[tuple[str, object]] = []
    for observation, instant, _ra, _dec in rows:
        key = observing_night_key(observation, instant)
        batch_keys.append(key)
        batch_sizes[key] = batch_sizes.get(key, 0) + 1
    factors = np.asarray([
        np.sqrt(max(1.0, batch_sizes[key] / INDEPENDENT_MEASUREMENTS_PER_NIGHT))
        for key in batch_keys
        for _coordinate in range(2)
    ], dtype=float)
    return values * factors


# --------------------------------------------------------------------------
# Duplicate reductions and deprecated records
# --------------------------------------------------------------------------

# Higher is better.  Single characters are MPC80 catalogue codes (column 72);
# longer strings are ADES ``astCat`` values.
_CATALOG_RANK = {
    "W": 100, "X": 100, "V": 98, "U": 95, "Y": 90, "Z": 90, "S": 88, "T": 88,
    "q": 85, "t": 80, "u": 78, "Q": 76, "r": 72, "L": 70, "v": 66, "w": 64,
    "g": 60, "o": 55, "M": 52, "N": 50, "n": 50, "x": 50, "y": 48, "p": 40,
    "m": 35, "l": 35, "c": 30, "d": 30, "a": 25, "b": 25, "h": 20, "i": 20,
    "j": 20, "k": 22, "z": 18, "B": 15, "C": 15, "D": 15, "E": 15, "A": 10,
}
_ADES_CATALOG_RANK = {
    "GAIA3E": 100, "GAIA3": 100, "GAIA2": 98, "GAIA1": 95, "UCAC5": 90,
    "ATLAS2": 90, "URAT1": 88, "URAT2": 88, "UCAC4": 85, "PPMXL": 80,
    "UCAC3": 78, "CMC15": 76, "UCAC2": 72, "2MASS": 70, "NOMAD": 66,
    "CMC14": 64, "TYCHO2": 60, "USNOB1": 55, "GSC2.3": 52, "GSC23": 52,
    "SDSS7": 50, "SDSS8": 50, "HIP2": 50, "PPM": 40, "USNOA2": 30,
    "USNOSA2": 30, "USNOA1": 25, "GSCACT": 20, "GSC1.2": 18, "GSC12": 18,
    "GSC": 15,
}


def catalog_rank(observation: Observation) -> int:
    code = str(observation.catalog or observation.raw.get("mpc_catalog") or "").strip()
    if len(code) == 1:
        return _CATALOG_RANK.get(code, 0)
    return _ADES_CATALOG_RANK.get(code.upper().replace("-", ""), 0)


# --------------------------------------------------------------------------
# Comet fragments
# --------------------------------------------------------------------------

# MPC80 columns 6-12 hold a packed provisional designation (``K08R060`` =
# 2008 R6); a fragment replaces its last character by a lowercase letter
# (``K08R06b``).  A numbered comet leaves columns 6-11 blank and puts the
# fragment letter(s) in columns 11-12 (``0205P      b``, ``0073P     bk``).
_PACKED_COMET_PROVISIONAL = re.compile(r"^[A-L]\d\d[A-Z][0-9A-Za-z]\d[0-9a-z]$")
# Unpacked designations (ADES permID/provID, user input) carry the fragment as
# an uppercase suffix: ``205P-B``, ``73P-BK``, ``D/1993 F2-G``, ``2008 R6-C``.
_UNPACKED_FRAGMENT = re.compile(
    r"^\s*(?:\d+[PDI]|(?:[ACDPXI]/)?\d{4}\s+[A-Z]{1,2}\d+)-([A-Z]{1,2})\s*$", re.I,
)
# ADES rows may carry the original 80-column record.
_OBS80_KEYS = ("mpc80", "obs80", "obs_80")
_DESIGNATION_KEYS = ("permid", "provid", "trksub")


def mpc80_fragment(line: str) -> str:
    """Fragment letter(s) of one MPC 80-column record, upper case, or ``""``."""
    field = (line or "")[5:12].ljust(7)
    if _PACKED_COMET_PROVISIONAL.match(field):
        return field[6].upper() if field[6].islower() else ""
    if not field[:5].strip():
        letters = field[5:7].strip()
        if letters.isalpha() and letters.islower():
            return letters.upper()
    return ""


def designation_fragment(designation: str) -> str:
    """Fragment suffix of an unpacked designation (``205P-B`` → ``B``)."""
    match = _UNPACKED_FRAGMENT.match(designation or "")
    return match.group(1).upper() if match else ""


def observation_fragment(observation: Observation) -> str:
    """Fragment (``A``, ``B``, ``BK``) a measurement refers to; ``""`` if unlabelled.

    Unlabelled rows are the undivided comet (before the split) or a report that
    did not resolve the components.
    """
    raw = {str(key).casefold(): value for key, value in observation.raw.items()}
    for key in _OBS80_KEYS:
        line = raw.get(key)
        if isinstance(line, str) and len(line) >= 12:
            letters = mpc80_fragment(line)
            if letters:
                return letters
    for value in (observation.designation, *(raw.get(key) for key in _DESIGNATION_KEYS)):
        if isinstance(value, str):
            letters = designation_fragment(value)
            if letters:
                return letters
    return ""


def fragment_counts(observations: list[Observation]) -> dict[str, int]:
    """Number of explicitly labelled measurements per fragment, sorted by label."""
    counts: dict[str, int] = {}
    for observation in observations:
        letters = observation_fragment(observation)
        if letters:
            counts[letters] = counts.get(letters, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (len(item[0]), item[0])))


# Labels separated by a longer gap belong to different returns (days); the
# seasonal solar-conjunction gap within one return is shorter.
FRAGMENT_RETURN_GAP_DAYS = 300.0


def fragment_return_counts(observations: list[Observation]) -> dict[str, int]:
    """Number of observed returns in which each fragment is explicitly labelled."""
    days: dict[str, list[float]] = {}
    for observation in observations:
        letters = observation_fragment(observation)
        if not letters or not observation.time:
            continue
        try:
            instant = parse_observation_time(observation.time)
        except ValueError:
            continue
        days.setdefault(letters, []).append(seconds_since_j2000(instant) / 86400.0)
    returns: dict[str, int] = {}
    for letters, values in days.items():
        values.sort()
        returns[letters] = 1 + sum(
            later - earlier > FRAGMENT_RETURN_GAP_DAYS
            for earlier, later in zip(values, values[1:])
        )
    return returns


def primary_fragment(observations: list[Observation]) -> str:
    """Principal (surviving) component of a split comet.

    The MPC letter does not say which nucleus survives: the brightest one at
    the split is ``A``, but 73P/Schwassmann–Wachmann lost A within weeks and
    only C was followed through later returns, after which its astrometry is
    no longer labelled.  The component labelled in the most observed returns
    is therefore principal; the number of labelled measurements breaks ties
    (205P: A, B and C were all seen only in 2008, A is by far the best
    observed).  Short-lived pieces and stray labels rarely span returns.
    """
    counts = fragment_counts(observations)
    if not counts:
        return ""
    returns = fragment_return_counts(observations)
    # ``counts`` is ordered by label, so complete ties go to the earliest one.
    return max(counts, key=lambda letters: (returns.get(letters, 1), counts[letters]))


def select_fragment(observations: list[Observation], fragment: str | None) -> int:
    """Restrict fitting to one fragment of a split comet.

    Measurements of the other components get ``raw['inactive_fragment']`` and
    stop being fit candidates; they still receive O-C against the fitted orbit.
    Unlabelled rows belong to the principal component (:func:`primary_fragment`):
    they are the parent nucleus before the split or reports that did not
    resolve the fragments.  ``fragment`` ``None``/``""`` fits all rows (no
    selection), as does astrometry without any fragment label.  The user's manual inclusion state is not changed.  Returns
    the number of rows belonging to the selected component.
    """
    wanted = (fragment or "").strip().upper()
    primary = primary_fragment(observations) if wanted else ""
    if not primary:
        # Nothing is labelled: the astrometry does not resolve any fragments.
        wanted = ""
    selected = 0
    for observation in observations:
        if not wanted:
            observation.raw.pop("inactive_fragment", None)
            selected += 1
            continue
        letters = observation_fragment(observation) or primary
        if letters == wanted:
            observation.raw.pop("inactive_fragment", None)
            selected += 1
        else:
            observation.raw["inactive_fragment"] = True
    return selected


def is_deprecated(observation: Observation) -> bool:
    """MPC flags replaced/deleted records with note 'X'/'x' or ADES ``deprecated``."""
    if str(observation.raw.get("mpc_note2") or "") in ("X", "x"):
        return True
    deprecated = observation.raw.get("deprecated")
    return bool(deprecated) and str(deprecated).strip().upper() not in ("", "0", "FALSE", "N")


def mark_duplicate_measurements(observations: list[Observation]) -> int:
    """Flag re-reductions of one measurement so only the best one is fitted.

    MPC frequently lists one historical exposure several times, reduced with
    different star catalogues (206P in 1892: PPM and UCAC-4).  Fitting all
    copies would count the same measurement twice.  The copy with the best
    catalogue keeps ``raw['duplicate_measurement'] = False``; the others are
    flagged ``True`` and receive O-C values but no weight.  Returns the number
    of flagged copies.  The user's manual inclusion state is not changed.
    """
    # Components of a split comet are measured on the same exposure; they are
    # different objects, never re-reductions of each other.
    entries: list[tuple[str, float, Observation]] = []
    for observation in observations:
        observation.raw.pop("duplicate_measurement", None)
        if observation.ra_deg is None or observation.dec_deg is None or not observation.time:
            continue
        try:
            instant = parse_observation_time(observation.time)
        except ValueError:
            continue
        entries.append((
            (observation.station or "").strip().upper() + "|" + observation_fragment(observation),
            seconds_since_j2000(instant) / 86400.0,
            observation,
        ))
    entries.sort(key=lambda item: (item[0], item[1]))
    flagged = 0
    index = 0
    while index < len(entries):
        cluster = [entries[index]]
        while (
            index + len(cluster) < len(entries)
            and entries[index + len(cluster)][0] == cluster[0][0]
            and entries[index + len(cluster)][1] - cluster[-1][1]
            <= DUPLICATE_TIME_TOLERANCE_DAYS
        ):
            cluster.append(entries[index + len(cluster)])
        index += len(cluster)
        if len(cluster) < 2:
            continue
        best = max(cluster, key=lambda item: catalog_rank(item[2]))[2]
        for _station, _day, observation in cluster:
            if observation is best:
                continue
            separation = 3600.0 * float(np.hypot(
                ((observation.ra_deg - best.ra_deg + 180.0) % 360.0 - 180.0)
                * np.cos(np.radians(best.dec_deg)),
                observation.dec_deg - best.dec_deg,
            ))
            if separation <= DUPLICATE_POSITION_TOLERANCE_ARCSEC:
                observation.raw["duplicate_measurement"] = True
                flagged += 1
    return flagged


def is_fit_candidate(observation: Observation) -> bool:
    """Valid, enabled, of the selected fragment, not a duplicate and not deprecated."""
    return bool(
        not observation.excluded_from_fit
        and observation.ra_deg is not None and observation.dec_deg is not None
        and observation.time
        and not observation.raw.get("duplicate_measurement")
        and not observation.raw.get("inactive_fragment")
        and not is_deprecated(observation)
    )


# --------------------------------------------------------------------------
# Normal places
# --------------------------------------------------------------------------

def normal_place_observations(
    observations: list[Observation],
    maximum_independent_per_night: int = INDEPENDENT_MEASUREMENTS_PER_NIGHT,
) -> list[Observation]:
    """Build deterministic station-night normal places.

    Batches up to ``maximum_independent_per_night`` are retained unchanged.  A
    larger batch becomes one place at its mean epoch: RA/Dec are evaluated
    from a straight-line fit in time, so the object's own motion during the
    night neither biases the place nor inflates its uncertainty.  The formal
    uncertainty gains at most ``maximum_independent_per_night`` independent
    measurements and never drops below the scatter about the fitted motion.
    The original observations are untouched.
    """
    buckets: dict[tuple[str, object], list[tuple[datetime, Observation]]] = {}
    for observation in observations:
        if not is_fit_candidate(observation):
            continue
        try:
            instant = parse_observation_time(observation.time)
        except ValueError:
            continue
        buckets.setdefault(observing_night_key(observation, instant), []).append(
            (instant, observation)
        )

    output: list[Observation] = []
    for bucket in buckets.values():
        bucket.sort(key=lambda item: item[0])
        chronological = [item[1] for item in bucket]
        if len(chronological) <= maximum_independent_per_night:
            output.extend(chronological)
            continue
        seconds = np.asarray([seconds_since_j2000(item[0]) for item in bucket])
        mean_seconds = float(np.mean(seconds))
        hours = (seconds - mean_seconds) / 3600.0
        reference_ra = float(chronological[0].ra_deg)
        # Unwrap RA around the first measurement and work in a local tangent
        # frame (arcsec) so the 0/360 deg boundary is harmless.
        dec = np.asarray([float(item.dec_deg) for item in chronological])
        mean_dec = float(np.mean(dec))
        x = (
            (np.asarray([float(item.ra_deg) for item in chronological]) - reference_ra + 180.0)
            % 360.0 - 180.0
        ) * np.cos(np.radians(mean_dec)) * 3600.0
        y = (dec - mean_dec) * 3600.0
        degree = 1 if np.ptp(hours) > 0.0 else 0
        x_fit = np.polyfit(hours, x, degree)
        y_fit = np.polyfit(hours, y, degree)
        x_scatter = x - np.polyval(x_fit, hours)
        y_scatter = y - np.polyval(y_fit, hours)
        dof = max(1, len(hours) - (degree + 1))
        scatter = np.sqrt(np.asarray((
            np.sum(np.square(x_scatter)) / dof, np.sum(np.square(y_scatter)) / dof,
        )))
        place_x = float(np.polyval(x_fit, 0.0))
        place_y = float(np.polyval(y_fit, 0.0))
        place_dec = mean_dec + place_y / 3600.0
        place_ra = (
            reference_ra + place_x / 3600.0 / np.cos(np.radians(mean_dec))
        ) % 360.0

        effective_count = min(len(chronological), maximum_independent_per_night)
        sigma_pairs = np.asarray([
            observation_sigma_arcsec(item) for item in chronological
        ], dtype=float)
        nominal = np.sqrt(np.mean(np.square(sigma_pairs), axis=0)) / np.sqrt(effective_count)
        sigma = np.maximum(nominal, scatter / np.sqrt(effective_count))
        template = chronological[len(chronological) // 2]
        raw = dict(template.raw)
        raw.update({
            "rmsRA": float(sigma[0]),
            "rmsDec": float(sigma[1]),
            "normalPlaceCount": len(chronological),
        })
        raw.pop("duplicate_measurement", None)
        output.append(Observation(
            designation=template.designation,
            time=instant_from_seconds_since_j2000(mean_seconds)
            .isoformat().replace("+00:00", "Z"),
            ra_deg=float(place_ra),
            dec_deg=float(place_dec),
            magnitude=template.magnitude,
            band=template.band,
            station=template.station,
            catalog=template.catalog,
            raw=raw,
        ))
    output.sort(key=lambda item: parse_observation_time(item.time))
    return output
