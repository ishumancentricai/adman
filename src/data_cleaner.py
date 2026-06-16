# src/data_cleaner.py

"""
- Loads raw survey CSV (already using question codes as headers)
- Filters to completed responses:
    - submitdate must be non-empty
    - lastpage must equal 7
  If exactly one of these holds, log an error for that PID indicating what failed.
- Saves cleaned data to new CSV
"""

import pandas as pd
import os
from src.dynamic_config import DynamicConfig


def clean_data(
    config: DynamicConfig | None = None,
    customer_dir=None,
    customer_data_file=None,
    cleaned_customer_data_file=None,
) -> pd.DataFrame:
    config = config or DynamicConfig()
    customer_dir = customer_dir or config.CUSTOMER_DIR
    customer_data_file = customer_data_file or config.CUSTOMER_DATA_FILE
    cleaned_customer_data_file = (
        cleaned_customer_data_file or config.CLEANED_CUSTOMER_DATA_FILE
    )

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, customer_dir, customer_data_file)
    output_path = os.path.join(base_path, customer_dir, cleaned_customer_data_file)

    # Load data (headers are already question codes)
    # df = pd.read_csv(input_path, sep=",", quotechar='"')
    df = pd.read_csv(input_path, sep="\t", quotechar='"')

    # Validate required columns
    required_cols = {"PID", "submitdate", "lastpage"}
    missing = required_cols - set(df.columns)
    print(set(df.columns))
    if missing:
        raise ValueError(
            f"Missing required column(s) in raw data: {', '.join(sorted(missing))}"
        )

    # Normalize checks
    submit_present = df["submitdate"].astype("string").str.strip().notna() & (
        df["submitdate"].astype("string").str.strip() != ""
    )
    lastpage_num = pd.to_numeric(df["lastpage"], errors="coerce")
    lastpage_ok = lastpage_num.eq(7)

    # Log cases where exactly one condition is true
    only_submit_true = submit_present & ~lastpage_ok
    only_lastpage_true = ~submit_present & lastpage_ok

    if only_submit_true.any():
        for pid, lp in df.loc[only_submit_true, ["PID", "lastpage"]].itertuples(
            index=False
        ):
            print(f"Incomplete row for PID='{pid}': lastpage != 7 (lastpage={lp})")

    if only_lastpage_true.any():
        for pid, sd in df.loc[only_lastpage_true, ["PID", "submitdate"]].itertuples(
            index=False
        ):
            print(
                f"Incomplete row for PID='{pid}': submitdate is empty (submitdate='{sd}')"
            )

    # Keep only rows where BOTH conditions are true
    both_ok = submit_present & lastpage_ok
    cleaned = df.loc[both_ok].copy()

    # Ensure output directory exists
    os.makedirs(os.path.join(base_path, config.DATA_DIR), exist_ok=True)

    # Save to CSV
    cleaned.to_csv(output_path, index=False)
    print(
        f"Cleaned data saved to: {output_path} "
        f"(kept {len(cleaned)}/{len(df)} rows; dropped {len(df) - len(cleaned)})"
    )

    return cleaned


# CLI execution
if __name__ == "__main__":
    clean_data()
