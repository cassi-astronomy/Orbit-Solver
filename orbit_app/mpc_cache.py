"""Local cache of astrometry downloaded from the MPC archive.

Downloading the full astrometry of a long-observed comet (1P, 2P, …) takes a
while, so the raw ADES rows returned by the MPC ``get-obs`` service are stored
in ``orbit-data/mpc-cache`` (one gzip-compressed JSON file per object).  The
GUI loads astrometry from the cache when it is present and contacts the MPC
only when the user explicitly asks for an update.

Raw ADES rows are cached rather than parsed observations, so improvements of
the parser apply to cached data as well.  NEOCP/PCCP astrometry is never
cached: those objects change from hour to hour.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .domain import Observation
from .planetary import ephemeris_data_directory

CACHE_FORMAT = 1
CACHE_DIRECTORY_NAME = "mpc-cache"


@dataclass(frozen=True, slots=True)
class CachedAstrometry:
    designation: str
    downloaded_utc: datetime
    rows: list[dict[str, Any]]
    path: Path

    def observations(self) -> list[Observation]:
        from .mpc import MpcClient

        return [MpcClient._to_observation(row, self.designation) for row in self.rows]


def cache_directory() -> Path:
    return ephemeris_data_directory() / CACHE_DIRECTORY_NAME


def normalized_designation(designation: str) -> str:
    return " ".join(str(designation).split()).upper()


def cache_path(designation: str) -> Path:
    key = normalized_designation(designation)
    if not key:
        raise ValueError("Empty designation")
    slug = re.sub(r"[^A-Z0-9._-]+", "_", key).strip("_")[:60] or "object"
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:10]
    return cache_directory() / f"{slug}-{digest}.json.gz"


def load(designation: str) -> CachedAstrometry | None:
    """Return the cached astrometry of ``designation`` or ``None``.

    A missing, unreadable or foreign file is treated as "not cached".
    """
    try:
        path = cache_path(designation)
    except ValueError:
        return None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, EOFError, ValueError):
        return None
    if not isinstance(payload, dict) or payload.get("format") != CACHE_FORMAT:
        return None
    if normalized_designation(payload.get("designation", "")) != normalized_designation(
        designation
    ):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        return None
    try:
        downloaded = datetime.fromisoformat(str(payload.get("downloaded_utc")))
    except ValueError:
        downloaded = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
    if downloaded.tzinfo is None:
        downloaded = downloaded.replace(tzinfo=timezone.utc)
    return CachedAstrometry(
        designation=str(payload["designation"]),
        downloaded_utc=downloaded,
        rows=[row for row in rows if isinstance(row, dict)],
        path=path,
    )


def save(
    designation: str, rows: list[dict[str, Any]], downloaded_utc: datetime | None = None,
) -> Path:
    """Store raw ADES rows atomically and return the cache file path."""
    path = cache_path(designation)
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = (downloaded_utc or datetime.now(timezone.utc)).astimezone(timezone.utc)
    payload = {
        "format": CACHE_FORMAT,
        "designation": " ".join(str(designation).split()),
        "downloaded_utc": stamp.isoformat(timespec="seconds"),
        "source": "MPC get-obs (ADES_DF 2022)",
        "rows": rows,
    }
    handle, temporary = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as raw, gzip.GzipFile(
            fileobj=raw, mode="wb", mtime=0,
        ) as compressed:
            compressed.write(json.dumps(payload, ensure_ascii=False).encode("utf-8"))
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise
    return path


def remove(designation: str) -> bool:
    try:
        cache_path(designation).unlink()
    except (OSError, ValueError):
        return False
    return True
