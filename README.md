# I Am AdMan: Automatic Personalized Advertising Imagery

This repository contains a research pipeline for generating personalized
advertising images from consumer/persona data. For each persona and product, the
pipeline builds a persona description, asks a text model to write an image
prompt, generates an edited product image from a reference image, and uses a
judge model to evaluate whether the result satisfies the prompt and product
constraints.

The current pipeline supports two providers:

- `openai`: GPT text/judging models with `gpt-image-1`
- `google`: Gemini text/judging models with Gemini image generation

## Repository Contents

- [01_pipeline.py](01_pipeline.py): end-to-end generation pipeline.
- [02_evaluate_images.py](02_evaluate_images.py): manual review tool for
  generated-image artifacts.
- [03_aggregate_evaluations.py](03_aggregate_evaluations.py): aggregates
  multiple reviewer CSVs and resolves ties interactively.
- [04_calculate_metrics.py](04_calculate_metrics.py): calculates PR, AR, and
  rAR metrics from provider-specific aggregated evaluation data.
- [src/dynamic_config.py](src/dynamic_config.py): shared provider, model,
  product, path, and generation defaults.
- [prompt_templates](prompt_templates): prompt templates used by the generator,
  judge, and selector stages.
- [reference_images](reference_images): product reference images.
- [data/participants.csv](data/participants.csv): included demo persona data.

## Data Availability and Limitations

This repository does not include the original participant study dataset. The
included [data/participants.csv](data/participants.csv) contains celebrity demo
personas so the pipeline can be exercised end to end without private study data.

Evaluation and metrics scripts expect generated outputs and reviewer CSVs under
`data/<provider>/...`. They are useful for the full study workflow, but demo data
results should not be treated as reproducible study metrics.

## Setup

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repository root with the key for the provider you
want to run.

For OpenAI:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

For Google Gemini:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

or:

```bash
GOOGLE_API_KEY=your_google_api_key_here
```

The `.env` file is intentionally not tracked.

## Run the Pipeline

Run a small demo batch with either provider:

```bash
python 01_pipeline.py --provider openai --badge-size 2
```

or:

```bash
python 01_pipeline.py --provider google --badge-size 2
```

`--badge-size` is the number of new personas to process from
`data/participants.csv`. If a persona already has an output directory for the
selected provider, it is skipped.

For each selected persona, the pipeline runs all four configured products:

1. Polestar 4 electric car
2. Rolex Submariner Date Oyster luxury watch
3. Tide Ultra Hygienic Clean laundry detergent
4. Coca-Cola soft drink

For each persona/product pair, the pipeline generates up to three versions. It
stops early when the judge returns `pass`; otherwise, it runs a selector step to
choose the best version from the failed attempts.

## Outputs

Generated files are separated by provider:

```text
data/
  openai/
    customers/<persona_id>/
      <persona_id>.csv
      <persona_id>_cleaned.csv
      <persona_id>_L3.csv
      generated_prompts/
      generated_images/
      judgements/
      prompt_logs/
  google/
    customers/<persona_id>/
      ...
```

Important output types:

- `generated_prompts/*.txt`: generated image prompts and model reasoning logs.
- `generated_images/*.png`: generated ad images.
- `judgements/*_judgement.json`: pass/fail judge output for each version.
- `judgements/*_version_comparison.json`: selector output when no version
  passes within the version limit.
- `prompt_logs/*.txt`: exact prompts sent to generation stages.

## Configuration

Most defaults live in [src/dynamic_config.py](src/dynamic_config.py), including:

- provider defaults
- OpenAI and Google model names
- retry and timeout settings for Google calls
- version limits
- product metadata and reference image filenames
- output path templates

## Manual Evaluation Workflow

After generating images, configure the provider and reviewer initials at the top
of [02_evaluate_images.py](02_evaluate_images.py), then run:

```bash
python 02_evaluate_images.py
```

The script opens generated images and records artifact decisions in:

```text
data/<provider>_<reviewer>_image_evaluation.csv
```

To aggregate multiple reviewers, configure the reviewer names and provider at
the top of [03_aggregate_evaluations.py](03_aggregate_evaluations.py), then run:

```bash
python 03_aggregate_evaluations.py
```

The aggregate output is:

```text
data/<provider>_aggregated_evaluation.csv
```

To calculate metrics from the aggregated file, set `PROVIDER` in
[04_calculate_metrics.py](04_calculate_metrics.py), then run:

```bash
python 04_calculate_metrics.py
```

The metrics script reports:

- `PR`: judge passed rate
- `AR`: artifact rate, including text-only artifacts
- `rAR`: relaxed artifact rate, excluding text-only artifacts

## Sources for Reference Images

- `reference_images/polestar.jpg`: [Source](https://media.polestar.com/ch/de/media/pressreleases/666140)
- `reference_images/rolex.jpg`: [Source](https://www.rolex.com/de-de/watches/submariner/m124060-0001)
- `reference_images/tide.jpg`: [Source](https://www.amazon.co.uk/Tide-Hygienic-Detergent-Original-Compatible/dp/B08QBV9DJ5)
- `reference_images/cola.jpg`: [Source](https://www.coca-cola.com/de/de/brands/brand-coca-cola)

## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE)
file for details.
