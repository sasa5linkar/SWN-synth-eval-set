import csv
import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sentiment_ollama as sa


class FakeOllamaAnalyzer:
    def __init__(self):
        self.positive = [
            {
                "raw": "not json",
                "parsed": None,
                "parsing_error": ValueError("bad json"),
            },
            {
                "raw": '{"value":"ekstremno pozitivan","explanation":"Pojam oznacava najjacu pozitivnu granicu."}',
                "parsed": sa.StructuredSentimentResult(
                    value="ekstremno pozitivan",
                    explanation="Pojam oznacava najjacu pozitivnu granicu.",
                ),
                "parsing_error": None,
            },
        ]
        self.negative = [
            {
                "raw": '{"value":"nije negativan","explanation":"Nema negativnu vrednost."}',
                "parsed": sa.StructuredSentimentResult(
                    value="nije negativan",
                    explanation="Nema negativnu vrednost.",
                ),
                "parsing_error": None,
            }
        ]

    def analyze_positive(self, row):
        return self.positive.pop(0)

    def analyze_negative(self, row):
        return self.negative.pop(0)


class TestOllamaFewShot(unittest.TestCase):
    def test_normalizes_and_accepts_allowed_label(self):
        result = sa.validate_structured_sentiment(
            {"value": " Ekstremno Pozitivan ", "explanation": "Granica."},
            sa.POSITIVE_LABELS,
        )

        self.assertEqual(result.value, "ekstremno pozitivan")
        self.assertEqual(result.explanation, "Granica.")

    def test_rejects_hallucinated_label(self):
        with self.assertRaises(ValueError):
            sa.validate_structured_sentiment(
                {"value": "mnogo dobro", "explanation": "Nije iz skupa."},
                sa.POSITIVE_LABELS,
            )

    def test_process_samples_clones_input_and_appends_ollama_columns(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp = Path(temp_dir)
            input_path = temp / "input.csv"
            output_path = temp / "output.csv"
            original = pd.DataFrame(
                [
                    {
                        "ILI": "ENG30-00000001-n",
                        "definition": "test definicija",
                        "lemma_names": "['test']",
                        "sentiment_SWN": "(0.0, 0.0)",
                    }
                ]
            )
            original.to_csv(input_path, index=False)

            sa.process_samples_ollama_fewshot(
                input_file=input_path,
                output_file=output_path,
                analyzer=FakeOllamaAnalyzer(),
                model_name="test-model",
                retries=1,
            )

            with input_path.open(newline="", encoding="utf-8") as f:
                input_rows = list(csv.DictReader(f))
                input_fields = f.seek(0) or next(csv.reader(f))
            with output_path.open(newline="", encoding="utf-8-sig") as f:
                output_reader = csv.DictReader(f)
                output_rows = list(output_reader)
                output_fields = output_reader.fieldnames

            self.assertEqual(len(output_rows), len(input_rows))
            self.assertEqual(output_rows[0]["ILI"], input_rows[0]["ILI"])
            self.assertEqual(output_rows[0]["definition"], input_rows[0]["definition"])
            self.assertEqual(output_fields[: len(input_fields)], input_fields)
            self.assertEqual(output_rows[0]["ollama_positive_value"], "ekstremno pozitivan")
            self.assertIn("najjacu pozitivnu", output_rows[0]["ollama_positive_explanation"])
            self.assertEqual(output_rows[0]["ollama_negative_value"], "nije negativan")
            self.assertEqual(output_rows[0]["ollama_model"], "test-model")

    def test_balanced_objective_neutral_sample_bundles_objective_and_neutral(self):
        data = pd.DataFrame(
            [
                {"ILI": f"P{i}", "sentiment_sa": "Pozitivan"} for i in range(4)
            ]
            + [
                {"ILI": f"N{i}", "sentiment_sa": "Negativan"} for i in range(4)
            ]
            + [
                {"ILI": f"O{i}", "sentiment_sa": "Objektivan"} for i in range(2)
            ]
            + [
                {"ILI": f"NEU{i}", "sentiment_sa": "Neutralan"} for i in range(2)
            ]
        )

        sample = sa.get_balanced_sample_with_objective_neutral(data, n=3, random_state=7)

        self.assertEqual(len(sample), 9)
        self.assertEqual((sample["sentiment_sa_sample_group"] == "Pozitivan").sum(), 3)
        self.assertEqual((sample["sentiment_sa_sample_group"] == "Negativan").sum(), 3)
        self.assertEqual((sample["sentiment_sa_sample_group"] == "Objektivan/Neutralan").sum(), 3)
        self.assertTrue(
            set(sample[sample["sentiment_sa_sample_group"] == "Objektivan/Neutralan"]["sentiment_sa"])
            <= {"Objektivan", "Neutralan"}
        )


if __name__ == "__main__":
    unittest.main()
