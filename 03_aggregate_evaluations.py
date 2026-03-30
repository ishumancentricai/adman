"""
Aggregate Evaluation Script

This script aggregates results from multiple evaluation CSVs.
It performs a majority vote for 'artifact' and 'text_only_artifact'.
If there is no consensus (e.g. tie), it asks the user for a final decision.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# === CONFIGURATION ===
BASE_PATH = Path(__file__).parent.resolve()
DATA_DIR = BASE_PATH / "data"

# List of CSV files to aggregate
# Adjust this list as needed
INPUT_FILES = [
    DATA_DIR / "image_evaluation_paper_rev_1.csv",
    DATA_DIR / "image_evaluation_paper_rev_2.csv",
    DATA_DIR / "image_evaluation_paper_rev_3.csv"
]

OUTPUT_CSV = DATA_DIR / "image_evaluation_aggregated.csv"


def load_data(file_paths):
    """Load and combine data from multiple CSV files."""
    dfs = []
    for p in file_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                # Ensure text_only_artifact exists
                if "text_only_artifact" not in df.columns:
                    df["text_only_artifact"] = 0
                dfs.append(df)
                print(f"Loaded {len(df)} records from {p.name}")
            except Exception as e:
                print(f"Error loading {p.name}: {e}")
        else:
            print(f"Warning: File not found: {p}")

    if not dfs:
        return pd.DataFrame()

    return pd.concat(dfs, ignore_index=True)


def get_majority_vote(values):
    """
    Return the majority value if it exists, otherwise None.
    values: list or series of 0s and 1s.
    """
    # Remove NaNs
    clean_values = [v for v in values if pd.notna(v)]
    if not clean_values:
        return None

    count_1 = clean_values.count(1)
    count_0 = clean_values.count(0)
    total = len(clean_values)

    if count_1 > total / 2:
        return 1
    elif count_0 > total / 2:
        return 0
    else:
        return None     # Tie


def display_image_and_resolve_conflict(
    image_path: Path,
    customer_id: str,
    product_id: int,
    version: int,
    votes_artifact: list,
    votes_text: list
) -> tuple[int, int] | None:
    """
    Display image and ask user to resolve conflict.
    """
    print(f"\n{'='*60}")
    print("CONFLICT RESOLUTION")
    print(f"Customer: {customer_id}")
    print(f"Product ID: {product_id}")
    print(f"Version: {version}")
    # Calculate vote categories
    votes_clean_count = sum(1 for a, t in zip(votes_artifact, votes_text) if a == 0 and t == 0)
    votes_artifact_count = sum(1 for a in votes_artifact if a == 1)
    votes_text_only_count = sum(1 for a, t in zip(votes_artifact, votes_text) if a == 0 and t == 1)
    print(f"Votes Clean: {votes_clean_count}")
    print(f"Votes Artifact: {votes_artifact_count}")
    print(f"Votes Text-Only: {votes_text_only_count}")
    print(f"Image: {image_path.name}")
    print(f"{'='*60}")

    # Result container: (artifact, text_only_artifact)
    result = {"value": None, "done": False}

    def on_artifact(event):
        result["value"] = (1, 0)
        result["done"] = True
        plt.close()

    def on_no_artifact(event):
        result["value"] = (0, 0)
        result["done"] = True
        plt.close()

    def on_text_only(event):
        result["value"] = (0, 1)
        result["done"] = True
        plt.close()

    def on_quit(event):
        result["value"] = "quit"
        result["done"] = True
        plt.close()

    try:
        # Resolve path if it is relative
        if not image_path.is_absolute():
            full_path = BASE_PATH / image_path
        else:
            full_path = image_path

        if not full_path.exists():
            print(f"Image not found at {full_path}")
            return None

        img = Image.open(full_path)
        fig = plt.figure(figsize=(14, 10))

        # Try to maximize
        manager = fig.canvas.manager
        try:
            manager.full_screen_toggle()
        except AttributeError:
            try:
                manager.window.showMaximized()
            except AttributeError:
                pass

        ax_img = fig.add_axes([0.05, 0.15, 0.9, 0.8])
        ax_img.imshow(img)
        ax_img.axis("off")
        # Calculate vote categories for display
        votes_clean_count = sum(1 for a, t in zip(votes_artifact, votes_text) if a == 0 and t == 0)
        votes_artifact_count = sum(1 for a in votes_artifact if a == 1)
        votes_text_only_count = sum(1 for a, t in zip(votes_artifact, votes_text) if a == 0 and t == 1)
        ax_img.set_title(f"CONFLICT: {customer_id} | P{product_id} | v{version}\nClean: {votes_clean_count} | "
                         f"Artifact: {votes_artifact_count} | Text-Only: {votes_text_only_count}", fontsize=12, color='red')

        # Buttons
        btn_width = 0.15
        btn_height = 0.06
        btn_y = 0.03
        gap = 0.02
        total_width = 4 * btn_width + 3 * gap
        start_x = (1 - total_width) / 2

        ax_artifact = fig.add_axes([start_x, btn_y, btn_width, btn_height])
        ax_text_only = fig.add_axes([start_x + btn_width + gap, btn_y, btn_width, btn_height])
        ax_no_artifact = fig.add_axes([start_x + 2 * (btn_width + gap), btn_y, btn_width, btn_height])
        ax_quit = fig.add_axes([start_x + 3 * (btn_width + gap), btn_y, btn_width, btn_height])

        btn_artifact = Button(ax_artifact, 'Artifact (y)', color='#ffcccc', hovercolor='#ff9999')
        btn_text_only = Button(ax_text_only, 'Text Only (t)', color='#ffddcc', hovercolor='#ffbb99')
        btn_no_artifact = Button(ax_no_artifact, 'Clean (n)', color='#ccffcc', hovercolor='#99ff99')
        btn_quit = Button(ax_quit, 'Quit (q)', color='#cccccc', hovercolor='#999999')

        btn_artifact.on_clicked(on_artifact)
        btn_text_only.on_clicked(on_text_only)
        btn_no_artifact.on_clicked(on_no_artifact)
        btn_quit.on_clicked(on_quit)

        # Keyboard shortcuts
        def on_key(event):
            if event.key in ['y', 'Y']:
                on_artifact(event)
            elif event.key in ['t', 'T']:
                on_text_only(event)
            elif event.key in ['n', 'N']:
                on_no_artifact(event)
            elif event.key in ['q', 'Q']:
                on_quit(event)

        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()

        if result["value"] == "quit":
            raise KeyboardInterrupt("User requested quit")

        return result["value"]

    except Exception as e:
        if "User requested quit" in str(e):
            raise
        print(f"Error displaying image: {e}")
        return None


def load_existing_aggregated() -> pd.DataFrame:
    """Load already aggregated results."""
    if OUTPUT_CSV.exists():
        try:
            df = pd.read_csv(OUTPUT_CSV)
            print(f"Loaded {len(df)} existing aggregated records.")
            return df
        except Exception as e:
            print(f"Error loading existing aggregated file: {e}")
    return pd.DataFrame()


def save_aggregated(df: pd.DataFrame) -> None:
    """Save aggregated results to CSV."""
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Progress saved to {OUTPUT_CSV}")


def main():
    print("="*60)
    print("AGGREGATE EVALUATIONS")
    print("="*60)

    # 1. Load Data
    all_data = load_data(INPUT_FILES)
    if all_data.empty:
        print("No data loaded.")
        return

    print(f"Total records loaded: {len(all_data)}")

    # Load existing progress
    final_df = load_existing_aggregated()

    # Create a set of already processed keys
    if not final_df.empty:
        processed_keys = set(zip(final_df['customer_id'], final_df['product_id'], final_df['version']))
    else:
        processed_keys = set()

    # 2. Group by Image
    # We group by customer, product, version
    grouped = all_data.groupby(['customer_id', 'product_id', 'version'])

    conflict_count = 0
    resolved_count = 0
    new_records_count = 0

    print(f"Found {len(grouped)} unique images.")
    print("Processing...")

    try:
        for (cust_id, prod_id, ver), group in grouped:
            # Skip if already processed
            if (cust_id, prod_id, ver) in processed_keys:
                continue

            # Get votes
            votes_artifact = group['artifact'].tolist()
            votes_text = group['text_only_artifact'].tolist()

            # Helper to get majority
            maj_artifact = get_majority_vote(votes_artifact)
            maj_text = get_majority_vote(votes_text)

            final_artifact = maj_artifact
            final_text = maj_text

            # Check if we have a consensus
            # Consensus means we have a majority value for both
            # If any is None (tie), we have a conflict
            is_conflict = (maj_artifact is None) or (maj_text is None)

            if is_conflict:
                conflict_count += 1
                # Take the first record for metadata
                base_record = group.iloc[0].to_dict()

                # Resolve conflict
                result = display_image_and_resolve_conflict(
                    Path(base_record['path_of_image']),
                    cust_id, prod_id, ver,
                    votes_artifact, votes_text
                )

                if result is not None:
                    final_artifact, final_text = result
                    resolved_count += 1
                else:
                    # User skipped? Keep as None or skip record?
                    # If skipped, maybe just skip adding to final for now or use what we have?
                    # Let's assume we skip aggregating this one if user skips resolution
                    print("Skipping record resolution.")
                    continue

            # We have final values (either from majority or resolution)
            if final_artifact is not None and final_text is not None:
                # Construct final record
                # Use metadata from the first row of the group
                # (Assuming metadata like paths, is_final, etc are consistent)
                final_record = group.iloc[0].copy()
                final_record['artifact'] = final_artifact
                final_record['text_only_artifact'] = final_text
                final_record['clean'] = 1 if (final_artifact == 0 and final_text == 0) else 0

                # Add vote counts as separate columns
                final_record['n_evaluators'] = len(votes_artifact)
                # Count votes by category: clean (0,0), artifact (1,x), text_only (0,1)
                votes_clean = sum(1 for a, t in zip(votes_artifact, votes_text) if a == 0 and t == 0)
                votes_artifact_count = sum(1 for a in votes_artifact if a == 1)
                votes_text_only = sum(1 for a, t in zip(votes_artifact, votes_text) if a == 0 and t == 1)
                final_record['votes_clean'] = votes_clean
                final_record['votes_artifact'] = votes_artifact_count
                final_record['votes_text_only'] = votes_text_only

                # Convert Series to DataFrame single row
                new_row = pd.DataFrame([final_record])

                # Append to final_df
                final_df = pd.concat([final_df, new_row], ignore_index=True)
                new_records_count += 1

                # Save periodically (e.g. every time we resolve a conflict or every 10 records)
                if is_conflict or new_records_count % 10 == 0:
                    save_aggregated(final_df)

    except KeyboardInterrupt:
        print("\nAggregation interrupted.")

    # 3. Save Results
    if not final_df.empty:
        save_aggregated(final_df)
        print(f"\nFinal save: {len(final_df)} aggregated records to {OUTPUT_CSV}")
        print(f"New records added this session: {new_records_count}")
        print(f"Conflicts encountered this session: {conflict_count}")
        print(f"Conflicts resolved this session: {resolved_count}")

        # Summary
        print("\n--- Aggregated Summary ---")
        print(f"Total: {len(final_df)}")
        print(f"Artifacts: {final_df['artifact'].sum()}")
        print(f"Text-Only: {final_df['text_only_artifact'].sum()}")
    else:
        print("No records aggregated.")


if __name__ == "__main__":
    main()
