# I Am AdMan: A Pipeline for Automatic Generation of Personalized Advertising Imagery

This project explores how Generative AI can be used to create personalized advertisement imagery based on user consumer data.

## Note on API Keys
To run the pipeline with OpenAI, set an OpenAI API key in a `.env` file in the root of the project:

```
OPENAI_API_KEY=your_openai_api_key_here
```

To run the pipeline with Google Gemini / Nano Banana, set either:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

or:

```
GOOGLE_API_KEY=your_google_api_key_here
```

Note that the .env file is not tracked.

## Data Availability / Limitations

- This repository does **not** include the original participant study dataset. As a result, the demographics workflow is not reproducible here.
- The included [data/participants.csv](data/participants.csv) is **celebrity demo data** (fictional / well-known personas) intended to let you test the pipeline end-to-end.
- [05_calculate_metrics.py](05_calculate_metrics.py) is **participant-only**: it filters out celebrity IDs and expects participant evaluation outputs, so it cannot be used with the demo celebrity data.

## Pipeline Overview

```
data/participants.csv (celebrity demo personas)
   ↓
01_pipeline.py (clean → persona → prompt → image → judge)
   ↓
(optional) 02_evaluate_images.py (manual artifact evaluation)
   ↓
(optional) 03_aggregate_evaluations.py
   ↓
(optional) 04_correct_comparisons.py
```

## How to Run

```bash
pip install -r requirements.txt
```

### Run the pipeline on the included celebrity demo data

1. Ensure your provider API key is available (e.g. via `.env`).
2. Choose a provider. `openai` uses GPT + `gpt-image-1`; `google` uses Gemini + Nano Banana.
3. Run:

```bash
python 01_pipeline.py --provider openai --badge-size 2
```

or:

```bash
python 01_pipeline.py --provider google --badge-size 2
```

Outputs are separated by provider under `data/openai/...` or `data/google/...`.

4. Optionally, you can run the evaluation and aggregation scripts, but note that they are designed for participant data and may not work as intended with the celebrity demo data.


## Sources for Reference Images

- `reference_images/polestar.jpg`: [Source](https://media.polestar.com/ch/de/media/pressreleases/666140)
- `reference_images/rolex.jpg`: [Source](https://www.rolex.com/de-de/watches/submariner/m124060-0001)
- `reference_images/tide.jpg`: [Source](https://www.amazon.co.uk/Tide-Hygienic-Detergent-Original-Compatible/dp/B08QBV9DJ5)
- `reference_images/cola.jpg`: [Source](https://www.coca-cola.com/de/de/brands/brand-coca-cola)


## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.
