from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline

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
    "pluto": ((39.48211675, .24882730, 17.14001206, 238.92903833, 224.06891629, 110.30393684),
              (-.00031596, .00005170, .00004818, 145.20780515, -.04062942, -.01183482)),
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
        if body.lower() == "moon":
            earth = self.position("earth", jd)
            # Compact fallback for lunar perturbations when DE440 is absent.
            # This is deliberately approximate; DE440 supplies the real lunar
            # state whenever it is installed.
            mean_longitude = np.radians(
                (218.3164477 + 360.0 * (jd - J2000_JD) / 27.321661) % 360.0
            )
            node = np.radians((125.044555 - .05295377 * (jd - J2000_JD)) % 360.0)
            inclination = np.radians(5.145)
            argument = mean_longitude - node
            radius = .00256955529
            relative = np.array((
                radius * (
                    np.cos(node) * np.cos(argument)
                    - np.sin(node) * np.sin(argument) * np.cos(inclination)
                ),
                radius * (
                    np.sin(node) * np.cos(argument)
                    + np.cos(node) * np.sin(argument) * np.cos(inclination)
                ),
                radius * np.sin(argument) * np.sin(inclination),
            ))
            return earth + relative
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

    def positions(self, bodies: tuple[str, ...], jd: float) -> tuple[np.ndarray, ...]:
        return tuple(self.position(body, jd) for body in bodies)

AU_KM = 149_597_870.7
DE440_FILENAME = "de440.bsp"
DE440_URL = (
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp"
)
DE440_MD5 = "c9d581bfd84209dbeee8b1583939b148"
DE440_COVERAGE = (2287184.5, 2688976.5)  # 1549-12-31 through 2650-01-25


def ephemeris_data_directory() -> Path:
    override = os.environ.get("ORBIT_SOLVER_DATA_DIR")
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parent.parent / "orbit-data"


def de440_path() -> Path:
    return ephemeris_data_directory() / DE440_FILENAME


def _equatorial_to_ecliptic(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float)
    epsilon = np.radians(23.4392911)
    return np.array((
        value[0],
        value[1] * np.cos(epsilon) + value[2] * np.sin(epsilon),
        -value[1] * np.sin(epsilon) + value[2] * np.cos(epsilon),
    ))


def _equatorial_to_ecliptic_series(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    epsilon = np.radians(23.4392911)
    return np.column_stack((
        values[:, 0],
        values[:, 1] * np.cos(epsilon) + values[:, 2] * np.sin(epsilon),
        -values[:, 1] * np.sin(epsilon) + values[:, 2] * np.cos(epsilon),
    ))


class JplSpkPlanetEphemeris:
    """Heliocentric J2000-ecliptic positions read from a JPL SPK kernel."""

    _BARYCENTER_IDS = {
        "mercury": 1, "venus": 2, "earth": 3, "mars": 4,
        "jupiter": 5, "saturn": 6, "uranus": 7, "neptune": 8, "pluto": 9,
    }

    def __init__(self, path: Path) -> None:
        try:
            from jplephem.spk import SPK
        except ImportError as exc:
            raise RuntimeError(
                "Pro JPL DE440 chybí knihovna jplephem; nainstalujte requirements.txt."
            ) from exc
        self.path = Path(path)
        self.kernel = SPK.open(str(self.path))

    def _ssb_position(self, center: int, target: int, jd: float) -> np.ndarray:
        return np.asarray(self.kernel[center, target].compute(float(jd)), dtype=float)

    def position(self, body: str, jd: float) -> np.ndarray:
        if not (DE440_COVERAGE[0] <= float(jd) <= DE440_COVERAGE[1]):
            raise ValueError(
                "Požadovaná epocha leží mimo rozsah JPL DE440 "
                "(1549-12-31 až 2650-01-25)."
            )
        name = body.lower()
        if name != "moon" and name not in self._BARYCENTER_IDS:
            raise KeyError(f"Těleso {body!r} není v poskytovateli DE440 podporováno.")
        target = 3 if name == "moon" else self._BARYCENTER_IDS[name]
        barycentric = self._ssb_position(0, target, jd)
        if name == "mercury":
            barycentric += self._ssb_position(1, 199, jd)
        elif name == "venus":
            barycentric += self._ssb_position(2, 299, jd)
        elif name == "earth":
            barycentric += self._ssb_position(3, 399, jd)
        elif name == "moon":
            barycentric += self._ssb_position(3, 301, jd)
        sun = self._ssb_position(0, 10, jd)
        return _equatorial_to_ecliptic((barycentric - sun) / AU_KM)

    def positions(self, bodies: tuple[str, ...], jd: float) -> tuple[np.ndarray, ...]:
        """Return one integrator step while sharing repeated SPK segments."""
        epoch = float(jd)
        if not (DE440_COVERAGE[0] <= epoch <= DE440_COVERAGE[1]):
            raise ValueError(
                "Požadovaná epocha leží mimo rozsah JPL DE440 "
                "(1549-12-31 až 2650-01-25)."
            )
        names = tuple(name.lower() for name in bodies)
        sun = self._ssb_position(0, 10, epoch)
        earth_moon_barycenter = (
            self._ssb_position(0, 3, epoch)
            if "earth" in names or "moon" in names else None
        )
        output: list[np.ndarray] = []
        for name in names:
            if name == "earth":
                assert earth_moon_barycenter is not None
                barycentric = earth_moon_barycenter + self._ssb_position(3, 399, epoch)
            elif name == "moon":
                assert earth_moon_barycenter is not None
                barycentric = earth_moon_barycenter + self._ssb_position(3, 301, epoch)
            else:
                target = self._BARYCENTER_IDS.get(name)
                if target is None:
                    raise KeyError(f"Těleso {name!r} není v poskytovateli DE440 podporováno.")
                barycentric = self._ssb_position(0, target, epoch)
                if name == "mercury":
                    barycentric += self._ssb_position(1, 199, epoch)
                elif name == "venus":
                    barycentric += self._ssb_position(2, 299, epoch)
            output.append(_equatorial_to_ecliptic((barycentric - sun) / AU_KM))
        return tuple(output)

    def position_series(self, body: str, epochs: np.ndarray) -> np.ndarray:
        """Vectorized SPK evaluation used to prepare a long-arc spline."""
        jd = np.asarray(epochs, dtype=float)
        if np.any(jd < DE440_COVERAGE[0]) or np.any(jd > DE440_COVERAGE[1]):
            raise ValueError(
                "Požadovaná epocha leží mimo rozsah JPL DE440 "
                "(1549-12-31 až 2650-01-25)."
            )

        def series(center: int, target: int) -> np.ndarray:
            values = np.asarray(self.kernel[center, target].compute(jd), dtype=float)
            return values.T if values.ndim == 2 else values[None, :]

        name = body.lower()
        sun = series(0, 10)
        if name == "earth":
            barycentric = series(0, 3) + series(3, 399)
        elif name == "moon":
            barycentric = series(0, 3) + series(3, 301)
        else:
            target = self._BARYCENTER_IDS.get(name)
            if target is None:
                raise KeyError(f"Těleso {body!r} není v poskytovateli DE440 podporováno.")
            barycentric = series(0, target)
            if name == "mercury":
                barycentric += series(1, 199)
            elif name == "venus":
                barycentric += series(2, 299)
        return _equatorial_to_ecliptic_series((barycentric - sun) / AU_KM)

    def close(self) -> None:
        self.kernel.close()


class PlanetEphemerisProvider:
    """Stable application-wide provider with optional DE440 activation."""

    def __init__(self) -> None:
        self.analytic = AnalyticPlanetEphemeris()
        self.jpl: JplSpkPlanetEphemeris | None = None
        self.load_error: str | None = None
        self.reload()

    def unload(self) -> None:
        if self.jpl is not None:
            self.jpl.close()
            self.jpl = None

    def reload(self) -> bool:
        self.unload()
        self.load_error = None
        path = de440_path()
        if not path.is_file():
            return False
        try:
            self.jpl = JplSpkPlanetEphemeris(path)
            return True
        except Exception as exc:
            self.load_error = str(exc)
            return False

    def position(self, body: str, jd: float) -> np.ndarray:
        provider = self.jpl or self.analytic
        return provider.position(body, jd)

    def positions(self, bodies: tuple[str, ...], jd: float) -> tuple[np.ndarray, ...]:
        provider = self.jpl or self.analytic
        return provider.positions(bodies, jd)

    def position_series(self, body: str, epochs: np.ndarray) -> np.ndarray:
        provider = self.jpl or self.analytic
        method = getattr(provider, "position_series", None)
        if callable(method):
            return method(body, epochs)
        return np.stack([provider.position(body, float(epoch)) for epoch in epochs])

    @property
    def model_name(self) -> str:
        return "JPL DE440" if self.jpl is not None else "analytický model"

    def status_text(self) -> str:
        if self.jpl is not None:
            size_mb = de440_path().stat().st_size / (1024 * 1024)
            return f"JPL DE440 je aktivní ({size_mb:.0f} MB; 1549–2650)."
        if self.load_error:
            return f"Soubor DE440 nelze použít: {self.load_error}"
        return "JPL DE440 chybí; používá se méně přesný analytický model planet."


PLANETS = PlanetEphemerisProvider()


class SplinePlanetEphemeris:
    """Vectorized, one-time planetary cache for repeated long integrations."""

    def __init__(
        self, source: PlanetEphemerisProvider, bodies: tuple[str, ...],
        start_jd: float, end_jd: float, step_days: float = 1.0,
    ) -> None:
        count = max(5, int(np.ceil((end_jd - start_jd) / step_days)) + 1)
        self.source = source
        self.epochs = np.linspace(float(start_jd), float(end_jd), count)
        self.bodies = tuple(body.lower() for body in bodies)
        self.body_indices = {body: index for index, body in enumerate(self.bodies)}
        # One vector-valued spline is materially faster inside solve_ivp than
        # nine independent Python/CubicSpline calls for every force evaluation.
        values = np.stack([
            source.position_series(body, self.epochs) for body in self.bodies
        ], axis=1)
        self.spline = CubicSpline(self.epochs, values, axis=0)

    def position(self, body: str, jd: float) -> np.ndarray:
        epoch = float(jd)
        if epoch < self.epochs[0] or epoch > self.epochs[-1]:
            return np.asarray(self.source.position(body, epoch), dtype=float)
        return np.asarray(
            self.spline(epoch)[self.body_indices[body.lower()]], dtype=float,
        )

    def positions(self, bodies: tuple[str, ...], jd: float) -> tuple[np.ndarray, ...]:
        epoch = float(jd)
        if epoch < self.epochs[0] or epoch > self.epochs[-1]:
            return tuple(
                np.asarray(self.source.position(body, epoch), dtype=float)
                for body in bodies
            )
        values = np.asarray(self.spline(epoch), dtype=float)
        return tuple(values[self.body_indices[body.lower()]] for body in bodies)

