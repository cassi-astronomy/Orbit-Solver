"""User-interface language (English by default, Czech optional).

Source texts in the code are Czech; :func:`tr` returns the English
translation from :mod:`orbit_app.translations_en` unless Czech is selected.
Templates use ``str.format`` positional fields (``{0}``, ``{1:.2f}``) and are
always formatted, so literal braces are written as ``{{`` and ``}}``.

The language is fixed for the lifetime of the process: several labels are
built when modules are imported.  ``load_saved_language()`` must therefore run
before the GUI modules are imported (``run.py`` does this); changing the
language in the GUI stores the choice and restarts the application.
The environment variable ``ORBIT_SOLVER_LANGUAGE`` (``en``/``cs``) overrides
the stored choice, which keeps tests and scripts deterministic.
"""
from __future__ import annotations

import os

from .translations_en import CATALOG as _ENGLISH

SUPPORTED_LANGUAGES = {"en": "English", "cs": "Čeština"}
DEFAULT_LANGUAGE = "en"
ENVIRONMENT_VARIABLE = "ORBIT_SOLVER_LANGUAGE"
SETTINGS_KEY = "language"
SETTINGS_ORGANIZATION = "Orbit Solver"
SETTINGS_APPLICATION = "Orbit Solver"


def _normalized(code: object) -> str | None:
    value = str(code or "").strip().lower()[:2]
    return value if value in SUPPORTED_LANGUAGES else None


_language = _normalized(os.environ.get(ENVIRONMENT_VARIABLE)) or DEFAULT_LANGUAGE


def language() -> str:
    """Active language code (``en`` or ``cs``)."""
    return _language


def set_language(code: str) -> None:
    """Switch the language of texts produced from now on."""
    global _language
    normalized = _normalized(code)
    if normalized is None:
        raise ValueError(f"Unsupported language: {code!r}")
    _language = normalized


def tr(template: str, *args: object) -> str:
    """Translate a Czech source template and format it with ``args``."""
    text = _ENGLISH.get(template, template) if _language == "en" else template
    return text.format(*args)


def _settings():
    from PySide6.QtCore import QSettings

    return QSettings(SETTINGS_ORGANIZATION, SETTINGS_APPLICATION)


def load_saved_language() -> str:
    """Apply the language stored by the GUI (environment variable wins)."""
    override = _normalized(os.environ.get(ENVIRONMENT_VARIABLE))
    if override is not None:
        set_language(override)
        return override
    try:
        saved = _normalized(_settings().value(SETTINGS_KEY, DEFAULT_LANGUAGE))
    except Exception:  # Qt unavailable or unreadable settings: keep default
        saved = None
    set_language(saved or DEFAULT_LANGUAGE)
    return language()


def save_language(code: str) -> None:
    """Remember the language for the next start of the application."""
    normalized = _normalized(code)
    if normalized is None:
        raise ValueError(f"Unsupported language: {code!r}")
    settings = _settings()
    settings.setValue(SETTINGS_KEY, normalized)
    settings.sync()
