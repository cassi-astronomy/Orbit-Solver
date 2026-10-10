from __future__ import annotations

import numpy as np

J2000_JD = 2451545.0

# JPL approximate mean elements and rates per Julian century, valid for the
# modern era. Each row: a, e, I, L, longitude of perihelion, longitude of node.
_ELEMENTS = {
    "mercury": ((.38709927, .20563593, 7.00497902, 252.25032350, 77.45779628, 48.33076593),
                (.00000037, .00001906, -.00594749, 149472.67411175, .16047689, -.12534081)),
    "venus": ((.72333566, .00677672, 3.39467605, 181.97909950, 131.60246718, 76.67984255),
              (.00000390, -.00004107, -.00078890, 58517.81538729, .00268329, -.27769418)),
    "earth": ((1.00000261, .01671123, -.00001531, 100.46457166, 102.93768193, 0.0),
              (.00000562, -.00004392, -.01294668, 35999.37244981, .32327364, 0.0)),
    "mars": ((1.52371034, .09339410, 1.84969142, -4.55343205, -23.94362959, 49.55953891),
             (.00001847, .00007882, -.00813131, 19140.30268499, .44441088, -.29257343)),
    "jupiter": ((5.20288700, .04838624, 1.30439695, 34.39644051, 14.72847983, 100.47390909),
                (-.00011607, -.00013253, -.00183714, 3034.74612775, .21252668, .20469106)),
    "saturn": ((9.53667594, .05386179, 2.48599187, 49.95424423, 92.59887831, 113.66242448),
               (-.00125060, -.00050991, .00193609, 1222.49362201, -.41897216, -.28867794)),
    "uranus": ((19.18916464, .04725744, .77263783, 313.23810451, 170.95427630, 74.01692503),
               (-.00196176, -.00004397, -.00242939, 428.48202785, .40805281, .04240589)),
    "neptune": ((30.06992276, .00859048, 1.77004347, -55.12002969, 44.96476227, 131.78422574),
                (.00026291, .00005105, .00035372, 218.45945325, -.32241464, -.00508664)),
}


def _solve_kepler(mean_anomaly: float, eccentricity: float) -> float:
    eccentric_anomaly = mean_anomaly
    for _ in range(15):
        delta = (eccentric_anomaly - eccentricity * np.sin(eccentric_anomaly) - mean_anomaly) / (
            1.0 - eccentricity * np.cos(eccentric_anomaly)
        )
        eccentric_anomaly -= delta
        if abs(delta) < 1e-14:
            break
    return float(eccentric_anomaly)


class AnalyticPlanetEphemeris:
    """Approximate heliocentric J2000-ecliptic planet positions in AU."""

    def position(self, body: str, jd: float) -> np.ndarray:
        base, rate = _ELEMENTS[body.lower()]
        centuries = (jd - J2000_JD) / 36525.0
        a, e, inc, longitude, long_peri, node = np.asarray(base) + np.asarray(rate) * centuries
        mean = np.radians((longitude - long_peri) % 360.0)
        eccentric = _solve_kepler(mean, e)
        x_prime = a * (np.cos(eccentric) - e)
        y_prime = a * np.sqrt(1.0 - e * e) * np.sin(eccentric)
        argument = np.radians(long_peri - node)
        ascending = np.radians(node)
        inclination = np.radians(inc)
        cos_o, sin_o = np.cos(ascending), np.sin(ascending)
        cos_w, sin_w = np.cos(argument), np.sin(argument)
        cos_i, sin_i = np.cos(inclination), np.sin(inclination)
        return np.array([
            (cos_o * cos_w - sin_o * sin_w * cos_i) * x_prime
            + (-cos_o * sin_w - sin_o * cos_w * cos_i) * y_prime,
            (sin_o * cos_w + cos_o * sin_w * cos_i) * x_prime
            + (-sin_o * sin_w + cos_o * cos_w * cos_i) * y_prime,
            sin_w * sin_i * x_prime + cos_w * sin_i * y_prime,
        ])


PLANETS = AnalyticPlanetEphemeris()

