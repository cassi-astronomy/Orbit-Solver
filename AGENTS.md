# AGENTS Guide

## What This Project Does

- Orbit Solver is a Python 3.10+ desktop application for asteroid and comet orbit work.
- The PySide6 GUI loads astrometry and catalogue orbits from the Minor Planet Center, fits orbits, generates ephemerides, and visualizes trajectories, sky positions, brightness, and long-term orbital evolution.
- The numerical core supports topocentric observations, planetary and lunar perturbations, light-time correction, formal uncertainties, and optional non-gravitational parameters.
- The application UI and user-facing messages are primarily in Czech. Keep new UI text in Czech unless the task explicitly changes the localization strategy.

## Project Map

- `run.py`: application entry point and top-level crash logging.
- `Orbit.cmd`: Windows launcher that runs `run.py` from the repository directory.
- `orbit_app/main_window.py`: Qt models, workers, dialogs, panels, plotting, networking, and main-window orchestration.
- `orbit_app/domain.py`: shared data classes for observations, orbit solutions, and ephemeris rows.
- `orbit_app/mpc.py`: MPC 80-column parser and synchronous MPC/NEOCP client.
- `orbit_app/mpc_orbit.py`: MPC orbit-response parsing and Cartesian-to-element conversion.
- `orbit_app/orbit_fit.py`: preliminary-orbit search, differential correction, weighting, outlier handling, multi-apparition linkage, and fit uncertainties.
- `orbit_app/orbit_engine.py`: state vectors, N-body integration, post-Newtonian solar correction, and non-gravitational acceleration laws.
- `orbit_app/planetary.py`: analytic planetary fallback, optional JPL DE440 provider, and interpolated ephemerides.
- `orbit_app/orbit_evolution.py`: perihelion returns, planetary encounters, captures, centered solutions, and exports.
- `orbit_app/comet_dynamics.py`: original/future barycentric comet-orbit analysis.
- `orbit_app/observatories.py`: MPC observatory catalogue, topocentric geometry, time scales, and horizontal coordinates.
- `orbit_app/presentation.py`: deterministic formatting, report summaries, and plotting projections with no Qt/Matplotlib dependency.
- `orbit_app/preview_orbit.py`: preview propagation and apparent geometry used by plots.
- `orbit_app/photometry.py`: asteroid and comet brightness models.
- `orbit_app/data/`: bundled offline star, constellation, and MPC observatory catalogues plus third-party licence text.
- `tests/test_core.py`: `unittest` coverage for parsers, fitting, geometry, dynamics, exports, and GUI-independent helpers.
- `tests/data/`: small checked-in test fixtures.
- `diagnose_78p.py`: focused developer diagnostic for comet 78P; not part of the normal application entry path.

## Setup, Run, and Test

Create an isolated environment and install the runtime dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Run the application:

```powershell
python run.py
```

Run the complete test suite from the repository root:

```powershell
python -m unittest discover -s tests -v
```

Run a focused test class or method while iterating:

```powershell
python -m unittest tests.test_core.MpcParsingTests -v
python -m unittest tests.test_core.NativeIntegratorTests.test_one_year_circular_solar_orbit -v
```

When PySide6 is unavailable, run the deliberately GUI-free smoke suite:

```powershell
python -m unittest tests.test_headless -v
```

Run the configured correctness-oriented Ruff checks:

```powershell
python -m ruff check .
```

`pyproject.toml` is the authoritative package/tool configuration. Keep the
runtime dependency list synchronized with `requirements.txt`. GitHub Actions
runs Ruff and the complete unittest suite on Python 3.11. There is no configured
formatter or static type checker; do not claim those checks passed.

## Architecture and Change Boundaries

- Keep reusable numerical and domain logic in the focused modules under `orbit_app/`; avoid adding more computation to `main_window.py` when it can be tested independently.
- Keep Qt widgets, signal/slot wiring, dialogs, background workers, and UI orchestration in `main_window.py`.
- Put deterministic formatting, coordinate-display transforms, and report helpers in `presentation.py`; it must remain importable without PySide6 or Matplotlib.
- Long-running fits, propagation, downloads, and uncertainty calculations must not block the GUI thread. Preserve the worker/signal pattern and cancellation checks.
- Keep `OrbitSolution` and `Observation` as the data boundary between parsing, fitting, propagation, and the UI. When adding fields, use safe defaults and update all relevant copy/replace/export paths.
- Prefer small, targeted changes in the large `orbit_fit.py` and `main_window.py` modules. Do not broadly reorganize them unless the task explicitly calls for a refactor.
- Network access is an input boundary, not a numerical dependency. Core calculations and unit tests should remain usable offline with fixtures, mocks, or the analytic planetary fallback.

## Numerical Astronomy Rules

- Document units at every new public boundary. Existing conventions include AU, days, AU/day, degrees, radians, arcseconds, Julian dates, and magnitudes; never infer or silently mix them.
- Be explicit about reference frames and centers: heliocentric vs. barycentric vs. planetocentric, and equatorial ICRS/J2000 vs. J2000 ecliptic.
- Be explicit about time scales. Observation strings are UTC, while integration and planetary ephemeris epochs generally use dynamical Julian dates/TDB. Reuse the existing conversion helpers.
- Preserve light-time treatment, topocentric station offsets, and observer geometry when changing residuals or ephemerides.
- Preserve both planetary modes: the bundled analytic fallback and the optional DE440 kernel. Do not silently fall back outside DE440 coverage when the active precise model is expected to reject the request.
- Treat fit weighting, robust rejection, short-arc branch selection, and multi-apparition linkage as coupled scientific behavior. Any change requires focused regression tests, not only visual inspection.
- Use the named linkage quality limits and `_linkage_trial_is_acceptable()` instead of duplicating literal RMS gates across branch-selection code.
- Do not loosen tolerances merely to make a failing numerical test pass. First identify whether the failure is caused by a model change, units, frame/time conversion, conditioning, or platform-level floating-point variation.
- Prefer vectorized NumPy operations in hot paths, but do not trade away readable unit/frame semantics for compactness.
- Maintain deterministic results for fixed inputs. Avoid hidden dependence on wall-clock time, network state, or unordered sampling in the numerical core.
- Orbit Solver is not an authoritative orbit service. Keep warnings and provenance clear when results are approximate, preliminary, fitted, or sourced from MPC/JPL.

## Coding Conventions

- Follow the existing Python style: four-space indentation, type annotations on public and non-trivial helpers, `snake_case` functions/variables, and `PascalCase` classes.
- Keep imports grouped as standard library, third-party packages, then local modules.
- Use `pathlib.Path` for filesystem paths and UTF-8 for text files.
- Use NumPy arrays for state vectors and sampled numeric series; validate shapes and finiteness at module boundaries where bad data would otherwise propagate.
- Raise domain-specific exceptions such as `MpcError`, `OrbitParseError`, or `OrbitFitError` where callers need to distinguish user/data failures from programming errors.
- User-facing failures should be concise Czech messages. Preserve technical detail in exception chains or `orbit-crash.log` rather than exposing raw tracebacks in dialogs.
- Comments should explain astronomical, numerical, or performance reasoning rather than restating the code.
- Do not add a dependency when the standard library or an already required package solves the problem cleanly. If a dependency is necessary, update `requirements.txt` and README setup information together.

## Testing Expectations

- Add or update tests for every behavioral change in parsers, fitting, dynamics, coordinates, time handling, exports, or helper functions.
- Prefer small synthetic states and observations with known invariants over live MPC/JPL queries.
- Mock network responses and expensive model boundaries. The normal unit suite must not download DE440 or depend on MPC/JPL availability.
- Test scientific invariants where exact reference values are unnecessary: finite outputs, conservation-like behavior, round trips, ordering, frame symmetry, bounds, and graceful rejection of invalid inputs.
- For bug fixes, add a regression test that fails for the original bug and is narrow enough to explain the intended behavior.
- After focused tests pass, run the full `unittest` command above. For GUI-only changes, also launch `python run.py` and smoke-test the affected workflow when the environment supports a display.
- Run `python -m ruff check .` before delivery. Keep tests deterministic and compatible with the CI `QT_QPA_PLATFORM=offscreen` environment.
- Do not commit generated screenshots or diagnostics from manual testing unless the task explicitly requests them as fixtures or documentation.

## Data, Network, and Security

- Treat all MPC, NEOCP/PCCP, JPL Horizons, clipboard, and imported-file content as untrusted input. Validate structure, ranges, missing fields, and error responses before use.
- Keep request timeouts and useful failure messages on external calls. Do not embed credentials or introduce services that require secrets without an explicit project decision.
- The optional `de440.bsp` is a large runtime download. Keep checksum verification and the documented coverage limits intact.
- Preserve the licence file when updating bundled d3-celestial data, and record the source/licence for any new catalogue.
- Avoid casual rewrites of large JSON catalogues. Changes to `orbit_app/data/` should be deliberate, reproducible, and reviewed separately from algorithm changes where practical.

## Generated, Runtime, and Do-Not-Edit Files

- `.venv/`, `__pycache__/`, `*.pyc`, and `.pytest_cache/` are local environment/cache files.
- `orbit-crash.log` is runtime diagnostic output and can contain local paths or user data.
- `orbit-data/` contains runtime-downloaded ephemeris data such as DE440.
- `artifacts/`, `dist/`, `build/`, and `*.spec` are local/generated packaging or investigation outputs.
- `_horizon_check.png` is a local diagnostic image.
- Do not commit or manually patch any of the above unless the task explicitly targets a generated artifact or packaging output.

## Documentation and Delivery

- Update `README.md` when a change affects installation, dependencies, visible features, supported formats, accuracy limits, or the user workflow.
- Keep terminology consistent with the Czech UI and with established MPC/JPL terminology.
- Summarize which tests were run and call out tests that could not be run.
- Keep commits and reviews focused; do not mix catalogue refreshes, numerical-model changes, and GUI restyling unless they are inseparable parts of the requested change.
