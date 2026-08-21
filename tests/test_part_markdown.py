from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest

from contraption.catalog.instantiations import PartInstantiationRegistry
from contraption.part_import.agents import ModelingAgent
from contraption.part_import.part_markdown import (
    expression_comments,
    expression_to_latex,
    render_part_markdown,
    write_part_markdown,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_ROOT = PROJECT_ROOT / "model_catalog"
RESISTOR = (
    CATALOG_ROOT
    / "electrical"
    / "resistors"
    / "fixed_resistors"
    / "instantiations"
    / "generic-100ohm-resistor"
)
SERVO = (
    CATALOG_ROOT
    / "electromechanical"
    / "servos"
    / "position_servos"
    / "instantiations"
    / "generic_position_servo"
)
YAGEO_1M = (
    CATALOG_ROOT
    / "electrical"
    / "resistors"
    / "fixed_resistors"
    / "instantiations"
    / "yageo_rc0603_1m"
)


class PartMarkdownTests(unittest.TestCase):
    def test_render_is_deterministic_and_standalone(self):
        first = render_part_markdown(RESISTOR)
        second = render_part_markdown(RESISTOR)

        self.assertEqual(first, second)
        self.assertIn("## How to read this document", first)
        self.assertIn("## Parent interface chain", first)
        self.assertIn("**Programmatically enforced contract**", first)
        self.assertIn("**Text-based explanation or desire", first)
        self.assertIn("## Model hypotheses at a glance", first)
        self.assertIn("Ideal uncertain resistor", first)
        self.assertIn("## Authoritative source manifest", first)
        self.assertIn("each displayed residual is enforced as equal to zero", first)
        self.assertNotIn(str(PROJECT_ROOT), first)

    def test_every_checked_in_part_readme_matches_the_deterministic_renderer(self):
        readmes = tuple(
            path
            for path in sorted(CATALOG_ROOT.rglob("README.md"))
            if path.parent.parent.name == "instantiations"
        )
        self.assertTrue(readmes)
        for readme in readmes:
            with self.subTest(readme=readme.relative_to(CATALOG_ROOT)):
                self.assertEqual(
                    readme.read_text(encoding="utf-8"),
                    render_part_markdown(readme.parent, catalog_root=CATALOG_ROOT),
                )

    def test_every_model_variant_and_human_model_name_is_rendered(self):
        markdown = render_part_markdown(SERVO)

        self.assertIn("| `v1` | Position-controlled hobby servo |", markdown)
        self.assertIn("| `v2` | Position-controlled hobby servo |", markdown)
        self.assertIn(
            "## Model hypothesis v1: Position-controlled hobby servo", markdown
        )
        self.assertIn(
            "## Model hypothesis v2: Position-controlled hobby servo", markdown
        )
        self.assertLess(markdown.index("hypothesis v1"), markdown.index("hypothesis v2"))

    def test_equations_and_inline_comments_are_human_readable(self):
        source = "v_p - resistance * i_p # Ohm's law"

        self.assertEqual(expression_comments(source), ("Ohm's law",))
        latex = expression_to_latex(source)
        self.assertIn(r"\mathrm{v\_p}", latex)
        self.assertIn(r"\mathrm{resistance}", latex)
        self.assertIn(r"\mathrm{i\_p}", latex)

        greek = expression_to_latex("alpha + gamma_i")
        self.assertIn(r"\alpha", greek)
        self.assertIn(r"\gamma_{\mathrm{i}}", greek)

    def test_parameter_table_and_equation_aliases_are_markdown_safe(self):
        markdown = render_part_markdown(YAGEO_1M)
        parameter_section = markdown.split(
            "### Programmatically enforced parameters and constraints", 1
        )[1].split("### Equation symbol key", 1)[0]
        table_rows = [
            line for line in parameter_section.splitlines() if line.startswith("|")
        ]

        self.assertEqual(len(table_rows), 5)
        self.assertTrue(all(line.count("|") == 7 for line in table_rows))
        self.assertIn(
            '`{"correlation_group": "yageo_rc0603_resistance", '
            '"distribution": "normal", "parameters": {"std": 10000.0}}`',
            parameter_section,
        )
        self.assertIn("| $\\theta_{1}$ | `resistance` | parameter |", markdown)
        self.assertIn(
            "| $\\dot{x}_{1}$ | `branch_current_dot` | state derivative |",
            markdown,
        )
        self.assertIn(
            "authoritative identifier is shown here and in the DSL source", markdown
        )
        relation = markdown.split("#### Residual relation: series_branch_voltage", 1)[
            1
        ].split("DSL source:", 1)[0]
        self.assertIn(r"\theta_{2}", relation)
        self.assertIn(r"\dot{x}_{1}", relation)
        self.assertNotIn(r"parasitic_{\mathrm{inductance}}", relation)

    def test_writer_supports_an_explicit_output_without_mutating_catalog(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "part.md"
            written = write_part_markdown(
                RESISTOR, catalog_root=CATALOG_ROOT, output=target
            )

            self.assertEqual(written, target)
            self.assertEqual(target.read_text(encoding="utf-8"), render_part_markdown(RESISTOR))

    def test_registry_accepts_only_the_reserved_derived_markdown_name(self):
        with tempfile.TemporaryDirectory() as temporary:
            catalog = Path(temporary) / "model_catalog"
            shutil.copytree(CATALOG_ROOT, catalog)
            readme = catalog / RESISTOR.relative_to(CATALOG_ROOT) / "README.md"
            readme.write_text("derived\n", encoding="utf-8")
            registry = PartInstantiationRegistry.load_catalog(catalog)
            self.assertIn("generic-100ohm-resistor.v1", registry)
            (readme.parent / "notes.md").write_text("unreserved\n", encoding="utf-8")
            with self.assertRaisesRegex(Exception, "unsupported instantiation files"):
                PartInstantiationRegistry.load_catalog(catalog)

    def test_luna_cannot_supply_the_reserved_readme(self):
        value = {
            "summary": "attempt",
            "artifacts": [
                {
                    "path": "electrical/resistors/fixed_resistors/instantiations/x/README.md",
                    "content": "agent-authored",
                }
            ],
            "assumptions": [],
            "evidence": [],
        }
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(ValueError, "cannot author"):
                ModelingAgent._materialize_artifacts(temporary, value)


if __name__ == "__main__":
    unittest.main()
