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



# Diameter of a body of absolute magnitude H and geometric albedo p_V:
# D = 1329 km / sqrt(p_V) * 10^(-H/5)  (Fowler & Chillemi 1992; Pravec &
# Harris 2007).  The constant is the diameter of an H = 0, p_V = 1 body.
DIAMETER_CONSTANT_KM = 1329.0
# Geometric albedos for a size estimate without a measured albedo.  Most
# asteroids lie between dark carbonaceous (C/D/P types, p_V ~ 0.05) and
# bright stony (S types, p_V ~ 0.25) surfaces; 0.03 and 0.50 bound the very
# darkest bodies and rare high-albedo E types.
ASTEROID_ALBEDO_TYPICAL = (0.25, 0.05)
ASTEROID_ALBEDO_EXTREME = (0.50, 0.03)


def diameter_from_absolute_magnitude_km(absolute_magnitude: float, albedo: float) -> float:
    """Effective diameter in km for asteroid H (mag) and geometric albedo p_V."""
    albedo = float(albedo)
    if not np.isfinite(albedo) or albedo <= 0.0:
        raise ValueError(tr("Albedo musí být kladné."))
    return float(DIAMETER_CONSTANT_KM / np.sqrt(albedo) * 10.0 ** (-0.2 * float(absolute_magnitude)))
