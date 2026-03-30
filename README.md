# I Am AdMan: A Pipeline for Automatic Generation of Personalized Advertising Imagery

This project explores how Generative AI can be used to create personalized advertisement imagery based on user consumer data.

## Note on API Keys
To run the pipeline, you will need an OpenAI API key. You can set this up in a `.env` file in the root of the project. The .env file should contain the following line:

```
OPENAI_API_KEY=your_openai_api_key_here
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

1. Ensure your OpenAI API key is available (e.g. via `.env`).
2. Set the batch parameter in the 01_pipeline.py to a small number (e.g. `batch_size = 2`) to test with the included celebrity demo data. The pipeline is designed to run in batches, so you can adjust this parameter to control how many personas are processed in one go. It will continue from where it left off if you stop and restart it, so you can run it in multiple batches if desired.
3. Run:

```bash
python 01_pipeline.py
```

4. Optionally, you can run the evaluation and aggregation scripts, but note that they are designed for participant data and may not work as intended with the celebrity demo data.


## Sources for Reference Images

- `reference_images/polestar.jpg`: [Source](https://media.polestar.com/ch/de/media/pressreleases/666140)
- `reference_images/rolex.jpg`: [Source](https://www.rolex.com/de-de/watches/submariner/m124060-0001)
- `reference_images/tide.jpg`: [Source](https://www.amazon.co.uk/Tide-Hygienic-Detergent-Original-Compatible/dp/B08QBV9DJ5)
- `reference_images/cola.jpg`: [Source](https://www.coca-cola.com/de/de/brands/brand-coca-cola)


## License

This project is licensed under the Apache License. See the [LICENSE](LICENSE) file for details.