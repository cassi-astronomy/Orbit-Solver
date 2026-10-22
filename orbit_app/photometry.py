from __future__ import annotations

import numpy as np
from .i18n import tr


def power_law_magnitude(
    absolute_magnitude: float,
    delta_au: np.ndarray,
    r_au: np.ndarray,
    exponent_n: float,
) -> np.ndarray:
    """m = H + 5 log10(delta) + 2.5 n log10(r)."""
    delta = np.maximum(np.asarray(delta_au, dtype=float), 1e-12)
    radius = np.maximum(np.asarray(r_au, dtype=float), 1e-12)
    return absolute_magnitude + 5.0 * np.log10(delta) + 2.5 * exponent_n * np.log10(radius)


def estimate_absolute_magnitude(
    apparent_magnitude: np.ndarray,
    delta_au: np.ndarray,
    r_au: np.ndarray,
    exponent_n: float,
    phase_correction: np.ndarray | None = None,
) -> float:
    """Robustly reduce measured apparent magnitudes to H or cometary M.

    MPC observation magnitudes are apparent magnitudes.  Treating their median
    directly as H/M applies the distance term twice and can make a distant
    object appear more than ten magnitudes too faint.
    """
    measured = np.asarray(apparent_magnitude, dtype=float)
    reduced = measured - (
        power_law_magnitude(0.0, delta_au, r_au, exponent_n)
    )
    if phase_correction is not None:
        reduced -= np.asarray(phase_correction, dtype=float)
    finite = reduced[np.isfinite(reduced)]
    if finite.size == 0:
        raise ValueError(tr("K odhadu H/M nejsou k dispozici platná fotometrická data."))
    return float(np.median(finite))


def hg_phase_correction(phase_deg: np.ndarray, slope_g: float = 0.15) -> np.ndarray:
    """IAU H-G phase correction (Bowell form) in magnitudes."""
    alpha = np.radians(np.clip(np.asarray(phase_deg, dtype=float), 0.0, 179.9))
    tan_half = np.tan(alpha / 2.0)
    phi1 = np.exp(-3.33 * np.power(tan_half, 0.63))
    phi2 = np.exp(-1.87 * np.power(tan_half, 1.22))
    flux = np.maximum((1.0 - slope_g) * phi1 + slope_g * phi2, 1e-12)
    return -2.5 * np.log10(flux)

