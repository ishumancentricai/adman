"""
Correct Comparisons Script

The judge inconsistently used 0-based (0, 1, 2) or 1-based (1, 2, 3) version
indices in version_comparison.json files. This script lets the user manually
verify which indexing was used and corrects `version_of_final` and `final`
columns in the aggregated evaluation results.

For every (customer_id, product_id) group that has a comparison file, the script
displays all three version images side-by-side together with the comparison
result and asks the user to choose the correct indexing variant:
    - 0-based (0, 1, 2): best_version is already correct
    - 1-based (1, 2, 3): best_version needs to be decremented by 1

The corrected results are saved to a new CSV file.
"""

import json
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

# === CONFIGURATION ===
BASE_PATH = Path(__file__).parent.resolve()
DATA_DIR = BASE_PATH / "data"
CUSTOMERS_DIR = DATA_DIR / "customers"
LEVEL_ID = 3

INPUT_CSV = DATA_DIR / "image_evaluation_aggregated.csv"
OUTPUT_CSV = DATA_DIR / "image_evaluation_corrected.csv"


def load_aggregated() -> pd.DataFrame:
    """Load the aggregated evaluation CSV."""
    if INPUT_CSV.exists():
        df = pd.read_csv(INPUT_CSV)
        print(f"Loaded {len(df)} records from {INPUT_CSV.name}")
        return df
    raise FileNotFoundError(f"Aggregated file not found: {INPUT_CSV}")


def load_existing_corrected() -> pd.DataFrame:
    """Load already corrected results (for resume support)."""
    if OUTPUT_CSV.exists():
        try:
            df = pd.read_csv(OUTPUT_CSV)
            print(f"Loaded {len(df)} existing corrected records.")
            return df
        except Exception as e:
            print(f"Error loading corrected file: {e}")
    return pd.DataFrame()


def save_corrected(df: pd.DataFrame) -> None:
    """Save corrected results to CSV."""
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Progress saved to {OUTPUT_CSV}")


def get_comparison_data(customer_id: str, product_id: int, level_id: int = LEVEL_ID) -> dict | None:
    """
    Load the version_comparison.json for a customer/product combination.

    Returns the parsed JSON dict or None if the file doesn't exist.
    """
    judgements_dir = CUSTOMERS_DIR / customer_id / "judgements"
    comparison_file = judgements_dir / f"{customer_id}_P{product_id}_L{level_id}_version_comparison.json"

    if comparison_file.exists():
        try:
            with open(comparison_file, "r") as f:
                data = json.load(f)
                if data.get("best_version") is not None:
                    return data
        except (json.JSONDecodeError, KeyError):
            pass
    return None


def get_image_path(customer_id: str, product_id: int, version: int, level_id: int = LEVEL_ID) -> Path | None:
    """Get the path to a specific image."""
    images_dir = CUSTOMERS_DIR / customer_id / "generated_images"
    image_file = images_dir / f"image_{customer_id}_P{product_id}_L{level_id}_v{version}.png"
    if image_file.exists():
        return image_file
    return None


def display_comparison_and_choose(
    customer_id: str,
    product_id: int,
    best_version: int,
    explanation: str,
    image_paths: list[Path | None],
) -> str | None:
    """
    Display all version images side-by-side and the comparison result.
    Ask the user to choose the correct indexing variant.

    Returns:
        "0-based" if indices are 0, 1, 2 (no correction needed)
        "1-based" if indices are 1, 2, 3 (best_version -= 1)
        "skip"    if the user wants to skip
        None      if quit requested (raises KeyboardInterrupt)
    """
    print(f"\n{'='*60}")
    print(f"Customer: {customer_id}")
    print(f"Product ID: {product_id}")
    print(f"best_version in JSON: {best_version}")
    print(f"Explanation: {explanation[:120]}...")
    print(f"{'='*60}")

    result = {"value": None, "done": False}

    def on_0based(event):
        result["value"] = "0-based"
        result["done"] = True
        plt.close()

    def on_1based(event):
        result["value"] = "1-based"
        result["done"] = True
        plt.close()

    def on_skip(event):
        result["value"] = "skip"
        result["done"] = True
        plt.close()

    def on_quit(event):
        result["value"] = "quit"
        result["done"] = True
        plt.close()

    try:
        # Count how many images we actually have
        valid_images = [(i, p) for i, p in enumerate(image_paths) if p is not None]
        n_images = len(valid_images)

        if n_images == 0:
            print("No images found for this combination. Skipping.")
            return "skip"

        fig = plt.figure(figsize=(18, 10))

        # Try to maximize the window
        manager = fig.canvas.manager
        try:
            manager.full_screen_toggle()
        except AttributeError:
            try:
                manager.window.showMaximized()
            except AttributeError:
                pass

        # Layout: images on top, buttons at bottom
        # Show images side-by-side
        img_top = 0.95
        img_bottom = 0.28
        img_height = img_top - img_bottom
        gap = 0.02
        total_gap = gap * (n_images - 1)
        img_width = (0.94 - total_gap) / n_images
        start_x = 0.03

        for idx, (version_idx, img_path) in enumerate(valid_images):
            ax = fig.add_axes([
                start_x + idx * (img_width + gap),
                img_bottom,
                img_width,
                img_height
            ])
            img = Image.open(img_path)
            ax.imshow(img)
            ax.axis("off")

            # Highlight which version the judge selected under each indexing
            label_0based = " << SELECTED (0-based)" if version_idx == best_version else ""
            label_1based = " << SELECTED (1-based)" if version_idx == (best_version - 1) else ""
            title_color = "red" if label_0based or label_1based else "black"
            markers = []
            if label_0based:
                markers.append("0-based")
            if label_1based:
                markers.append("1-based")
            marker_str = f" [{', '.join(markers)}]" if markers else ""

            ax.set_title(f"v{version_idx}{marker_str}", fontsize=11,
                         color=title_color if markers else "black", fontweight="bold" if markers else "normal")

        # Add explanation text
        explanation_wrapped = explanation[:300] + ("..." if len(explanation) > 300 else "")
        fig.text(0.5, 0.24, f"best_version = {best_version}  |  {explanation_wrapped}",
                 ha="center", va="top", fontsize=9, wrap=True,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))

        # Suptitle
        fig.suptitle(
            f"{customer_id} | P{product_id} | Judge says best = {best_version}",
            fontsize=14, fontweight="bold"
        )

        # Buttons
        btn_width = 0.18
        btn_height = 0.06
        btn_y = 0.03
        btn_gap = 0.03
        total_btn_width = 4 * btn_width + 3 * btn_gap
        btn_start_x = (1 - total_btn_width) / 2

        ax_0based = fig.add_axes([btn_start_x, btn_y, btn_width, btn_height])
        ax_1based = fig.add_axes([btn_start_x + btn_width + btn_gap, btn_y, btn_width, btn_height])
        ax_skip = fig.add_axes([btn_start_x + 2 * (btn_width + btn_gap), btn_y, btn_width, btn_height])
        ax_quit = fig.add_axes([btn_start_x + 3 * (btn_width + btn_gap), btn_y, btn_width, btn_height])

        btn_0 = Button(ax_0based, f'0-based → v{best_version} (a)', color='#ccffcc', hovercolor='#99ff99')
        corrected = best_version - 1
        btn_1 = Button(ax_1based, f'1-based → v{corrected} (b)', color='#ccccff', hovercolor='#9999ff')
        btn_skip_btn = Button(ax_skip, 'Skip (s)', color='#ffffcc', hovercolor='#ffff99')
        btn_quit_btn = Button(ax_quit, 'Quit (q)', color='#cccccc', hovercolor='#999999')

        btn_0.on_clicked(on_0based)
        btn_1.on_clicked(on_1based)
        btn_skip_btn.on_clicked(on_skip)
        btn_quit_btn.on_clicked(on_quit)

        # Keyboard shortcuts
        def on_key(event):
            if event.key in ['a', 'A']:
                on_0based(event)
            elif event.key in ['b', 'B']:
                on_1based(event)
            elif event.key in ['s', 'S']:
                on_skip(event)
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
        print(f"Error displaying images: {e}")
        return "skip"


def main():
    print("=" * 60)
    print("CORRECT VERSION COMPARISONS")
    print("=" * 60)

    # 1. Load aggregated data
    df = load_aggregated()

    # 2. Load existing corrected data (resume support)
    corrected_df = load_existing_corrected()

    # Build set of already-processed (customer_id, product_id) keys
    if not corrected_df.empty:
        processed_keys = set(zip(corrected_df["customer_id"], corrected_df["product_id"]))
    else:
        processed_keys = set()
        corrected_df = df.copy()

    # 3. Find unique (customer_id, product_id) groups that have a comparison file
    groups = df.groupby(["customer_id", "product_id"])
    pending = []

    for (cust_id, prod_id), group in groups:
        if (cust_id, prod_id) in processed_keys:
            continue
        comparison = get_comparison_data(cust_id, prod_id)
        if comparison is not None:
            pending.append((cust_id, prod_id, comparison))

    print(f"\nFound {len(pending)} product groups with comparison files to review.")
    if not pending:
        print("Nothing to do.")
        # If first run and nothing pending, just copy input to output
        if corrected_df.empty or len(corrected_df) == 0:
            save_corrected(df)
        return

    print(f"Already processed: {len(processed_keys)} groups")
    print("Press 'a' for 0-based, 'b' for 1-based, 's' to skip, 'q' to quit.\n")

    corrected_count = 0

    try:
        for cust_id, prod_id, comparison in pending:
            best_version = comparison["best_version"]
            explanation = comparison.get("explanation", "")

            # Get image paths for versions 0, 1, 2
            image_paths = [
                get_image_path(cust_id, prod_id, v) for v in range(3)
            ]

            choice = display_comparison_and_choose(
                cust_id, prod_id, best_version, explanation, image_paths
            )

            if choice == "skip":
                continue

            if choice is None:
                continue

            # Determine the corrected best_version
            if choice == "0-based":
                corrected_version = best_version
            elif choice == "1-based":
                corrected_version = best_version - 1
            else:
                continue

            # Update all rows for this (customer_id, product_id) in corrected_df
            mask = (
                (corrected_df["customer_id"] == cust_id) &
                (corrected_df["product_id"] == prod_id)
            )
            corrected_df.loc[mask, "version_of_final"] = corrected_version
            corrected_df.loc[mask, "final"] = (
                corrected_df.loc[mask, "version"] == corrected_version
            ).astype(int)

            corrected_count += 1
            print(f"  -> Corrected {cust_id} P{prod_id}: best_version {best_version} -> v{corrected_version} ({choice})")

            # Save periodically
            if corrected_count % 5 == 0:
                save_corrected(corrected_df)

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    # Final save
    save_corrected(corrected_df)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total records: {len(corrected_df)}")
    print(f"Groups corrected this session: {corrected_count}")
    print(f"Output saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
