import argparse
import csv
import json
import random

import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm


test_prompt_positive_sr2 = """
    Kao stručnjak za analizu sentimenta, vaš zadatak je da pažljivo procenite dati tekst na srpskom jeziku.
    Na osnovu vaše analize, klasifikujte sentiment teksta koristeći striktno definisane kategorije.
    Ove kategorije uključuju: 'nije pozitivan', 'slabo pozitivan', 'pozitivan', 'veoma pozitivan', i 'ekstremno pozitivan'.
    Važno je naglasiti da su ovo jedine prihvatljive kategorije za klasifikaciju.
    Molimo vas da se držite ovih smernica kako biste osigurali tačnost i konsistentnost u analizi sentimenta.
    Tekst za analizu: {text}.
    Očekujemo da vaša analiza rezultira određivanjem jedne od navedenih kategorija sentimenta.
    Kakav je sentiment teksta?
    """

test_prompt_negative_sr2 = """
    Kao stručnjak za analizu sentimenta, vaš zadatak je da pažljivo procenite dati tekst na srpskom jeziku.
    Na osnovu vaše analize, klasifikujte sentiment teksta koristeći striktno definisane kategorije.
    Ove kategorije uključuju: 'nije negativan', 'slabo negativan', 'negativan', 'veoma negativan', i 'ekstremno negativan'.
    Važno je naglasiti da su ovo jedine prihvatljive kategorije za klasifikaciju.
    Molimo vas da se držite ovih smernica kako biste osigurali tačnost i konsistentnost u analizi sentimenta.
    Tekst za analizu: {text}.
    Očekujemo da vaša analiza rezultira određivanjem jedne od navedenih kategorija sentimenta.
    Kakav je sentiment teksta?
    """

base_prompt_templates = {
    "system": "Vi ste AI asistent koji analizira sentiment teksta. Odgovarajte samo sa 'pozitivan', 'negativan' ili 'neutralan'.",
    "user": "Analiziraj sentiment ovog teksta: {text}",
}

defintion_prompt_templates = {
    "system": "Vi ste AI asistent koji analizira da li pojam opisan definicijom ima pozitivan, negativan ili neutralan sentiment. Odgovarajte samo sa 'pozitivan', 'negativan' ili 'neutralan'.",
    "user": "Analiziraj sentiment pojma opisan definicijom: {text}",
}

test_model_name = "llama3.2"
advanced_model_name = "mistral-small3.2:latest"

POSITIVE_LABELS = (
    "nije pozitivan",
    "slabo pozitivan",
    "pozitivan",
    "veoma pozitivan",
    "ekstremno pozitivan",
)

NEGATIVE_LABELS = (
    "nije negativan",
    "slabo negativan",
    "negativan",
    "veoma negativan",
    "ekstremno negativan",
)

OLLAMA_ANNOTATION_COLUMNS = (
    "ollama_positive_value",
    "ollama_positive_explanation",
    "ollama_positive_raw",
    "ollama_negative_value",
    "ollama_negative_explanation",
    "ollama_negative_raw",
    "ollama_model",
)


class StructuredSentimentResult(BaseModel):
    value: str = Field(description="One sentiment label from the allowed set.")
    explanation: str = Field(description="Short Serbian explanation for the selected value.")


class StructuredSentimentValidationError(ValueError):
    def __init__(self, message, raw=""):
        super().__init__(message)
        self.raw = raw


def normalize_sentiment_label(value):
    return " ".join(str(value).strip().lower().split())


def validate_structured_sentiment(result, allowed_labels):
    if isinstance(result, StructuredSentimentResult):
        value = result.value
        explanation = result.explanation
    elif isinstance(result, dict):
        value = result.get("value")
        explanation = result.get("explanation")
    else:
        raise ValueError("Structured sentiment result must be a dict or StructuredSentimentResult.")

    normalized_value = normalize_sentiment_label(value)
    if normalized_value not in allowed_labels:
        raise ValueError(f"Unexpected sentiment label: {value!r}.")
    if explanation is None or not str(explanation).strip():
        raise ValueError("Missing sentiment explanation.")
    return StructuredSentimentResult(value=normalized_value, explanation=str(explanation).strip())


def load_few_shot_examples(examples_file="few_shot_examples.json"):
    with open(examples_file, "r", encoding="utf-8") as f:
        return json.load(f)


def format_few_shot_examples(examples, polarity):
    annotation_key = "sentiment_sa_positive" if polarity == "positive" else "sentiment_sa_negative"
    lines = []
    for example in examples:
        annotation = example.get("few_shot_annotation", {})
        lines.append(
            "\n".join(
                [
                    f"ID: {example.get('ILI', '')}",
                    f"Literali: {example.get('lemma_names', [])}",
                    f"Definicija: {example.get('definition', '')}",
                    "JSON odgovor:",
                    json.dumps(
                        {
                            "value": annotation.get(annotation_key, ""),
                            "explanation": "Primer je granični kalibracioni slučaj za SWN POS/NEG skalu.",
                        },
                        ensure_ascii=False,
                    ),
                ]
            )
        )
    return "\n\n".join(lines)


def build_ollama_prompt(base_prompt, allowed_labels, few_shot_text, text, retry_message=None):
    schema_text = json.dumps(StructuredSentimentResult.model_json_schema(), ensure_ascii=False)
    retry_text = f"\nPrethodni odgovor nije bio ispravan: {retry_message}\n" if retry_message else ""
    return f"""
{base_prompt}

Ovo nije procena dramskog intenziteta emocije, nego mapiranje na SWN POS/NEG skalu.
Oznaka 'ekstremno pozitivan' predstavlja gornju granicu pozitivne SWN polarizacije.
Oznaka 'ekstremno negativan' predstavlja donju granicu negativne SWN polarizacije.

Dozvoljene vrednosti su isključivo:
{", ".join(allowed_labels)}

Vrati isključivo JSON objekat sa poljima "value" i "explanation".
JSON šema:
{schema_text}

Primeri:
{few_shot_text}
{retry_text}
Tekst za analizu: {text}
""".strip()


def raw_ollama_text(raw):
    if raw is None:
        return ""
    if hasattr(raw, "content"):
        return str(raw.content)
    return str(raw)


class SentimentAnaliserOllamaFewShot:
    def __init__(self, model_name="mistral-small3.2:latest", examples_file="few_shot_examples.json", temperature=0):
        from langchain_ollama import ChatOllama

        self.model_name = model_name
        self.examples = load_few_shot_examples(examples_file)
        llm = ChatOllama(model=model_name, temperature=temperature)
        self.structured_llm = llm.with_structured_output(StructuredSentimentResult, include_raw=True)

    def analyze_positive(self, row, retry_message=None):
        prompt = build_ollama_prompt(
            test_prompt_positive_sr2,
            POSITIVE_LABELS,
            format_few_shot_examples(self.examples, "positive"),
            row.get("definition", ""),
            retry_message=retry_message,
        )
        return self.structured_llm.invoke(prompt)

    def analyze_negative(self, row, retry_message=None):
        prompt = build_ollama_prompt(
            test_prompt_negative_sr2,
            NEGATIVE_LABELS,
            format_few_shot_examples(self.examples, "negative"),
            row.get("definition", ""),
            retry_message=retry_message,
        )
        return self.structured_llm.invoke(prompt)


def coerce_ollama_result(result, allowed_labels):
    parsing_error = None
    raw = ""
    parsed = result
    if isinstance(result, dict) and {"raw", "parsed", "parsing_error"} & set(result):
        raw = raw_ollama_text(result.get("raw"))
        parsed = result.get("parsed")
        parsing_error = result.get("parsing_error")
    if parsing_error:
        raise StructuredSentimentValidationError(f"Structured output parsing failed: {parsing_error}", raw=raw)
    return validate_structured_sentiment(parsed, allowed_labels), raw


def analyze_with_retries(analyze_func, allowed_labels, retries):
    raw = ""
    last_error = None
    for _ in range(retries + 1):
        try:
            parsed, raw = coerce_ollama_result(analyze_func(last_error), allowed_labels)
            return parsed.value, parsed.explanation, raw
        except TypeError:
            try:
                parsed, raw = coerce_ollama_result(analyze_func(), allowed_labels)
                return parsed.value, parsed.explanation, raw
            except Exception as exc:
                last_error = exc
                if hasattr(exc, "raw"):
                    raw = exc.raw
        except Exception as exc:
            last_error = exc
            if hasattr(exc, "raw"):
                raw = exc.raw
    return "", f"Validation failed after retries: {last_error}", raw


def get_balanced_sample_with_objective_neutral(dataframe, n=75, column="sentiment_sa", random_state=42):
    sample_groups = {
        "Pozitivan": ["Pozitivan"],
        "Negativan": ["Negativan"],
        "Objektivan/Neutralan": ["Objektivan", "Neutralan"],
    }
    samples = []
    for group_name, values in sample_groups.items():
        group = dataframe[dataframe[column].isin(values)].copy()
        group = group.sample(min(len(group), n), random_state=random_state)
        group["sentiment_sa_sample_group"] = group_name
        samples.append(group)
    if not samples:
        return dataframe.iloc[0:0].copy()
    return pd.concat(samples, ignore_index=True)


def write_balanced_objective_neutral_sample(
    input_file="sample_synsets3.csv",
    output_file="sample_synsets3_balanced_75.csv",
    n=75,
    random_state=42,
):
    df = pd.read_csv(input_file)
    sample = get_balanced_sample_with_objective_neutral(df, n=n, random_state=random_state)
    sample.to_csv(output_file, index=False)
    return output_file


def process_samples_ollama_fewshot(
    input_file="sample_synsets3.csv",
    output_file="sample_synsets3_ollama_fewshot.csv",
    analyzer=None,
    model_name="mistral-small3.2:latest",
    examples_file="few_shot_examples.json",
    retries=2,
):
    analyzer = analyzer or SentimentAnaliserOllamaFewShot(model_name=model_name, examples_file=examples_file)

    with open(input_file, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        original_fields = list(reader.fieldnames or [])
        rows = list(reader)

    output_fields = original_fields + [col for col in OLLAMA_ANNOTATION_COLUMNS if col not in original_fields]
    with open(output_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        for row in tqdm(rows, desc="Processing Ollama few-shot annotations"):
            output_row = dict(row)

            def positive_call(retry_message=None, row=row):
                try:
                    return analyzer.analyze_positive(row, retry_message)
                except TypeError:
                    return analyzer.analyze_positive(row)

            def negative_call(retry_message=None, row=row):
                try:
                    return analyzer.analyze_negative(row, retry_message)
                except TypeError:
                    return analyzer.analyze_negative(row)

            pos_value, pos_explanation, pos_raw = analyze_with_retries(positive_call, POSITIVE_LABELS, retries)
            neg_value, neg_explanation, neg_raw = analyze_with_retries(negative_call, NEGATIVE_LABELS, retries)
            output_row.update(
                {
                    "ollama_positive_value": pos_value,
                    "ollama_positive_explanation": pos_explanation,
                    "ollama_positive_raw": pos_raw,
                    "ollama_negative_value": neg_value,
                    "ollama_negative_explanation": neg_explanation,
                    "ollama_negative_raw": neg_raw,
                    "ollama_model": model_name,
                }
            )
            writer.writerow(output_row)
    return output_file


def irregular_sinset_senttiment(sample_size, model_name, promp_templates, output_file):
    with open("converted_synsets", "r", encoding="utf-8") as f:
        list_synsets = json.load(f)
    sample_synsets = random.sample(list_synsets, sample_size)
    analyzer = SentimentAnaliserOllamaFewShot(model_name=model_name)
    for synset in tqdm(sample_synsets, desc="Processing synsets"):
        result = analyzer.analyze_positive(synset)
        parsed, _ = coerce_ollama_result(result, POSITIVE_LABELS)
        synset["sentiment_sa"] = parsed.value
    with open(output_file + ".json", "w", encoding="utf-8") as f:
        f.write(json.dumps(sample_synsets, indent=4, ensure_ascii=False))
    pd.DataFrame(sample_synsets).to_csv(output_file + ".csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Ollama Serbian WordNet sentiment utilities.")
    parser.add_argument("--ollama-fewshot", action="store_true", help="Run the Ollama few-shot CSV annotator.")
    parser.add_argument("--input-file", default="sample_synsets3.csv", help="Input CSV for the Ollama few-shot annotator.")
    parser.add_argument("--output-file", default="sample_synsets3_ollama_fewshot.csv", help="Output CSV for the Ollama few-shot annotator.")
    parser.add_argument("--examples-file", default="few_shot_examples.json", help="Few-shot example JSON file.")
    parser.add_argument("--ollama-model", default="mistral-small3.2:latest", help="Local Ollama model name.")
    parser.add_argument("--retries", type=int, default=2, help="Retries for invalid structured output.")
    parser.add_argument("--make-balanced-sample", action="store_true", help="Write a 75/75/75 sample before annotation.")
    parser.add_argument("--balanced-output-file", default="sample_synsets3_balanced_75.csv", help="Output CSV for the balanced sample.")
    parser.add_argument("--sample-size", type=int, default=75, help="Rows per sample group.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for sampling.")
    args = parser.parse_args()

    if args.make_balanced_sample:
        write_balanced_objective_neutral_sample(
            input_file=args.input_file,
            output_file=args.balanced_output_file,
            n=args.sample_size,
            random_state=args.random_state,
        )
        if not args.ollama_fewshot:
            return
        args.input_file = args.balanced_output_file

    if args.ollama_fewshot:
        process_samples_ollama_fewshot(
            input_file=args.input_file,
            output_file=args.output_file,
            model_name=args.ollama_model,
            examples_file=args.examples_file,
            retries=args.retries,
        )
        return

    irregular_sinset_senttiment(2000, advanced_model_name, base_prompt_templates, "sample_sentiment_ollama33")


if __name__ == "__main__":
    main()
