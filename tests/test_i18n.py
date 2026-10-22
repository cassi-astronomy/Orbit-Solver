"""Interface language: English default, Czech optional, persistent choice."""
import ast
import os
import string
import subprocess
import sys
import unittest
from pathlib import Path

from orbit_app import i18n
from orbit_app.translations_en import CATALOG

ROOT = Path(__file__).resolve().parent.parent


def _fields(template: str) -> list[tuple[str, str, str | None]]:
    return sorted(
        (field, spec, conversion)
        for _text, field, spec, conversion in string.Formatter().parse(template)
        if field is not None
    )


class TranslationCatalogTests(unittest.TestCase):
    def test_every_translation_keeps_the_same_placeholders(self):
        for czech, english in CATALOG.items():
            with self.subTest(czech=czech):
                self.assertEqual(_fields(czech), _fields(english))

    def test_translations_keep_edge_whitespace(self):
        # Fragments are joined with values ("Centre: " + name), so leading
        # and trailing spaces/newlines are part of the template.
        for czech, english in CATALOG.items():
            with self.subTest(czech=czech):
                self.assertEqual(czech[:len(czech) - len(czech.lstrip())],
                                 english[:len(english) - len(english.lstrip())])
                self.assertEqual(czech[len(czech.rstrip()):], english[len(english.rstrip()):])

    def test_translations_contain_no_czech_words(self):
        czech_letters = set("áčďěíňřšťúůýžÁČĎĚÍŇŘŠŤÚŮÝŽ")
        allowed = ("Ondřejov", "Mašek", "Väisälä")
        for english in CATALOG.values():
            stripped = english
            for name in allowed:
                stripped = stripped.replace(name, "")
            self.assertFalse(czech_letters & set(stripped), english)


class SourceCoverageTests(unittest.TestCase):
    def test_every_literal_tr_template_has_an_english_translation(self):
        missing = []
        for path in sorted((ROOT / "orbit_app").glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == "tr" and node.args
                    and isinstance(node.args[0], ast.Constant)
                    and isinstance(node.args[0].value, str)
                    and node.args[0].value not in CATALOG
                ):
                    missing.append(f"{path.name}:{node.lineno}: {node.args[0].value}")
        self.assertEqual(missing, [])


class LanguageSwitchTests(unittest.TestCase):
    def setUp(self):
        self.previous = i18n.language()

    def tearDown(self):
        i18n.set_language(self.previous)

    def test_tr_formats_english_and_czech(self):
        i18n.set_language("en")
        self.assertEqual(i18n.tr("Návrat {0}: {1}", 1892, "ok"), "Return 1892: ok")
        i18n.set_language("cs")
        self.assertEqual(i18n.tr("Návrat {0}: {1}", 1892, "ok"), "Návrat 1892: ok")

    def test_unknown_text_passes_through_formatted(self):
        i18n.set_language("en")
        self.assertEqual(i18n.tr("RMS {0:.1f}″", 1.234), "RMS 1.2″")

    def test_unsupported_language_is_rejected(self):
        with self.assertRaises(ValueError):
            i18n.set_language("de")


class DefaultLanguageTests(unittest.TestCase):
    def _run(self, code: str, language: str | None) -> str:
        environment = dict(os.environ)
        environment.pop(i18n.ENVIRONMENT_VARIABLE, None)
        if language is not None:
            environment[i18n.ENVIRONMENT_VARIABLE] = language
        return subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=environment,
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    def test_english_is_the_default(self):
        code = (
            "from orbit_app.orbit_evolution import PLANET_LABELS;"
            "from orbit_app.i18n import language;"
            "print(language(), PLANET_LABELS['earth'])"
        )
        self.assertEqual(self._run(code, None), "en Earth")

    def test_czech_can_be_selected_before_import(self):
        code = (
            "from orbit_app.orbit_evolution import PLANET_LABELS;"
            "print(PLANET_LABELS['earth'])"
        )
        self.assertEqual(self._run(code, "cs"), "Země")


if __name__ == "__main__":
    unittest.main()
