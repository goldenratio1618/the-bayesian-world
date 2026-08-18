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
        self.assertIn("v_{\\mathrm{p}}", latex)
        self.assertIn("resistance", latex)
        self.assertIn("i_{\\mathrm{p}}", latex)

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
