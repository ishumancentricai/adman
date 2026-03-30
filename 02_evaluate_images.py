"""
Image Evaluation Script

This script allows manual evaluation of generated images for artifacts/glitches.
It iterates through all customers, products, and ALL versions of each image,
displaying each one and recording whether artifacts are present.

Results are stored in a CSV file in the data directory.

CSV Columns:
- customer_id: Customer identifier
- product_id: Product ID (1-4)
- version: Version of the image being evaluated (0, 1, 2, etc.)
- version_of_final: The final selected version from version_comparison.json
- final: 1 if this version is the final selected, 0 otherwise
- passed_judge: 1 if judgement is "pass", 0 if "fail"
- path_of_image: Relative path to the image file
- path_of_judgement: Relative path to the judgement JSON file
- incomplete: 1 if any value couldn't be determined, 0 otherwise
- artifact: Manual evaluation result (1=artifact present, 0=no artifact)
- text_only_artifact: 1 if only text artifacts present (does not count as artifact)
"""

import json
import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# === CONFIGURATION ===
BASE_PATH = Path(__file__).parent.resolve()
DATA_DIR = BASE_PATH / "data"
CUSTOMERS_DIR = DATA_DIR / "customers"
EVALUATION_CSV = DATA_DIR / "image_evaluation_paper_rev_1.csv"

# Product IDs to evaluate
PRODUCT_IDS = [1, 2, 3, 4]
# Level ID (typically L3)
LEVEL_ID = 3

# CSV columns
CSV_COLUMNS = [
    "customer_id",
    "product_id",
    "version",
    "version_of_final",
    "final",
    "passed_judge",
    "path_of_image",
    "path_of_judgement",
    "incomplete",
    "artifact",
    "text_only_artifact"
]


def get_all_customer_ids() -> list[str]:
    """Get all customer IDs from the customers directory."""
    customer_ids = []
    for item in CUSTOMERS_DIR.iterdir():
        if item.is_dir() and item.name != ".DS_Store":
            # Check if this customer has generated images
            images_dir = item / "generated_images"
            if images_dir.exists() and any(images_dir.iterdir()):
                customer_ids.append(item.name)
    return sorted(customer_ids)


def get_all_versions(customer_id: str, product_id: int, level_id: int = LEVEL_ID) -> list[int]:
    """
    Get all available versions for a customer/product combination.

    Returns a sorted list of version numbers.
    """
    images_dir = CUSTOMERS_DIR / customer_id / "generated_images"
    versions = []

    if images_dir.exists():
        for img_file in images_dir.iterdir():
            # Pattern: image_{customer}_P{product}_L{level}_v{version}.png
            if img_file.name.startswith(f"image_{customer_id}_P{product_id}_L{level_id}_v"):
                try:
                    version_str = img_file.stem.split("_v")[-1]
                    versions.append(int(version_str))
                except ValueError:
                    continue

    return sorted(versions)


def get_final_version(customer_id: str, product_id: int, level_id: int = LEVEL_ID) -> int | None:
    """
    Get the final selected version for a customer/product combination.

    Priority:
    1. version_comparison.json (source of truth)
    2. Fallback: if only one version passed the judge, that's the final
    3. Fallback: if only one version exists, that's the final

    Returns None if the final version cannot be determined.
    """
    judgements_dir = CUSTOMERS_DIR / customer_id / "judgements"
    comparison_file = judgements_dir / f"{customer_id}_P{product_id}_L{level_id}_version_comparison.json"

    # Priority 1: Check version_comparison.json
    if comparison_file.exists():
        try:
            with open(comparison_file, "r") as f:
                data = json.load(f)
                best_version = data.get("best_version")
                if best_version is not None:
                    return best_version
        except (json.JSONDecodeError, KeyError):
            pass

    # Get all available versions
    versions = get_all_versions(customer_id, product_id, level_id)

    if not versions:
        return None

    # Fallback: if only one version exists, that's the final
    if len(versions) == 1:
        return versions[0]

    # Fallback: check which versions passed the judge
    passed_versions = []
    for version in versions:
        judgement = get_judgement_result(customer_id, product_id, version, level_id)
        if judgement == 1:  # passed
            passed_versions.append(version)

    # If only one version passed, that's the final
    if len(passed_versions) == 1:
        return passed_versions[0]

    return None


def get_judgement_result(customer_id: str, product_id: int, version: int, level_id: int = LEVEL_ID) -> int | None:
    """
    Get the judgement result for a specific version.
    
    Returns:
        1 if judgement is "pass"
        0 if judgement is "fail"
        None if judgement file doesn't exist or can't be parsed
    """
    judgements_dir = CUSTOMERS_DIR / customer_id / "judgements"
    judgement_file = judgements_dir / f"{customer_id}_P{product_id}_L{level_id}_v{version}_judgement.json"

    if judgement_file.exists():
        try:
            with open(judgement_file, "r") as f:
                data = json.load(f)
                judgement = data.get("judgement", "").lower()
                if judgement == "pass":
                    return 1
                elif judgement == "fail":
                    return 0
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


def get_judgement_path(customer_id: str, product_id: int, version: int, level_id: int = LEVEL_ID) -> Path | None:
    """Get the path to a specific judgement file."""
    judgements_dir = CUSTOMERS_DIR / customer_id / "judgements"
    judgement_file = judgements_dir / f"{customer_id}_P{product_id}_L{level_id}_v{version}_judgement.json"

    if judgement_file.exists():
        return judgement_file
    return None


def to_relative_path(path: Path) -> str:
    """Convert an absolute path to a path relative to BASE_PATH."""
    try:
        return str(path.relative_to(BASE_PATH))
    except ValueError:
        return str(path)


def load_existing_evaluations() -> pd.DataFrame:
    """Load existing evaluations from CSV if it exists."""
    if EVALUATION_CSV.exists():
        df = pd.read_csv(EVALUATION_CSV)
        # Ensure all columns exist
        for col in CSV_COLUMNS:
            if col not in df.columns:
                df[col] = None
        return df
    return pd.DataFrame(columns=CSV_COLUMNS)


def save_evaluations(df: pd.DataFrame) -> None:
    """Save evaluations to CSV."""
    df.to_csv(EVALUATION_CSV, index=False)
    print(f"\nEvaluations saved to: {EVALUATION_CSV}")


def is_already_evaluated(df: pd.DataFrame, customer_id: str, product_id: int, version: int) -> bool:
    """Check if a customer/product/version combination has already been evaluated."""
    if df.empty:
        return False
    return (
        (df["customer_id"] == customer_id) &
        (df["product_id"] == product_id) &
        (df["version"] == version)
    ).any()


def display_image_and_get_evaluation(
    image_path: Path,
    customer_id: str,
    product_id: int,
    version: int,
    version_of_final: int | None,
    is_final: bool,
    passed_judge: int | None
) -> tuple[int, int] | None:
    """
    Display an image and ask the user to evaluate for artifacts using buttons.

    Returns:
        (artifact, text_only_artifact) tuple:
            - (1, 0) if artifact present
            - (0, 0) if no artifact
            - (0, 1) if text-only artifact
        None if user wants to skip
    """
    print(f"\n{'='*60}")
    print(f"Customer: {customer_id}")
    print(f"Product ID: {product_id}")
    print(f"Version: {version}")
    print(f"Final Selected Version: {version_of_final if version_of_final is not None else 'Unknown'}")
    print(f"Is Final: {'Yes' if is_final else 'No'}")
    print(f"Passed Judge: {'Yes' if passed_judge == 1 else 'No' if passed_judge == 0 else 'Unknown'}")
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

    def on_skip(event):
        result["value"] = None
        result["done"] = True
        plt.close()

    def on_quit(event):
        result["value"] = "quit"
        result["done"] = True
        plt.close()

    # Display the image
    try:
        img = Image.open(image_path)
        fig = plt.figure(figsize=(14, 10))

        # Try to maximize/fullscreen the window
        manager = fig.canvas.manager
        try:
            manager.full_screen_toggle()
        except AttributeError:
            try:
                manager.window.showMaximized()
            except AttributeError:
                try:
                    manager.resize(*manager.window.maxsize())
                except AttributeError:
                    pass

        # Create axes for image (leave space at bottom for buttons)
        ax_img = fig.add_axes([0.05, 0.15, 0.9, 0.8])
        ax_img.imshow(img)
        ax_img.axis("off")

        final_str = " [FINAL]" if is_final else ""
        judge_str = " [PASS]" if passed_judge == 1 else " [FAIL]" if passed_judge == 0 else ""
        ax_img.set_title(
            f"Customer: {customer_id} | Product: P{product_id} | Version: v{version}{final_str}{judge_str}",
            fontsize=12
        )

        # Create button axes
        from matplotlib.widgets import Button

        btn_width = 0.15
        btn_height = 0.06
        btn_y = 0.03
        gap = 0.02
        total_width = 5 * btn_width + 4 * gap
        start_x = (1 - total_width) / 2

        ax_artifact = fig.add_axes([start_x, btn_y, btn_width, btn_height])
        ax_text_only = fig.add_axes([start_x + btn_width + gap, btn_y, btn_width, btn_height])
        ax_no_artifact = fig.add_axes([start_x + 2 * (btn_width + gap), btn_y, btn_width, btn_height])
        ax_skip = fig.add_axes([start_x + 3 * (btn_width + gap), btn_y, btn_width, btn_height])
        ax_quit = fig.add_axes([start_x + 4 * (btn_width + gap), btn_y, btn_width, btn_height])

        # Create buttons with colors
        btn_artifact = Button(ax_artifact, 'Artifact (y)', color='#ffcccc', hovercolor='#ff9999')
        btn_text_only = Button(ax_text_only, 'Text Only (t)', color='#ffddcc', hovercolor='#ffbb99')
        btn_no_artifact = Button(ax_no_artifact, 'Clean (n)', color='#ccffcc', hovercolor='#99ff99')
        btn_skip = Button(ax_skip, 'Skip (s)', color='#ffffcc', hovercolor='#ffff99')
        btn_quit = Button(ax_quit, 'Quit (q)', color='#cccccc', hovercolor='#999999')

        # Connect buttons to callbacks
        btn_artifact.on_clicked(on_artifact)
        btn_text_only.on_clicked(on_text_only)
        btn_no_artifact.on_clicked(on_no_artifact)
        btn_skip.on_clicked(on_skip)
        btn_quit.on_clicked(on_quit)

        # Also allow keyboard shortcuts
        def on_key(event):
            if event.key in ['y', 'Y']:
                on_artifact(event)
            elif event.key in ['t', 'T']:
                on_text_only(event)
            elif event.key in ['n', 'N']:
                on_no_artifact(event)
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
        print(f"Error displaying image: {e}")
        return None


def get_pending_evaluations(df: pd.DataFrame, customer_ids: list[str]) -> list[dict]:
    """
    Get list of image records that need evaluation.

    Returns a list of dicts with all metadata for each pending evaluation.
    """
    pending = []

    for customer_id in customer_ids:
        for product_id in PRODUCT_IDS:
            # Get all versions for this customer/product
            versions = get_all_versions(customer_id, product_id)

            if not versions:
                continue

            # Get the final selected version
            version_of_final = get_final_version(customer_id, product_id)

            for version in versions:
                # Check if already evaluated
                if is_already_evaluated(df, customer_id, product_id, version):
                    continue

                # Get image path
                image_path = get_image_path(customer_id, product_id, version)
                if image_path is None:
                    continue

                # Get judgement result and path
                passed_judge = get_judgement_result(customer_id, product_id, version)
                judgement_path = get_judgement_path(customer_id, product_id, version)

                # Determine if this is the final version
                is_final = 1 if (version_of_final is not None and version == version_of_final) else 0

                # Check for incomplete data
                incomplete = 1 if (version_of_final is None or passed_judge is None) else 0

                pending.append({
                    "customer_id": customer_id,
                    "product_id": product_id,
                    "version": version,
                    "version_of_final": version_of_final,
                    "final": is_final,
                    "passed_judge": passed_judge,
                    "path_of_image": to_relative_path(image_path),
                    "path_of_judgement": to_relative_path(judgement_path) if judgement_path else None,
                    "incomplete": incomplete
                })

    return pending


def main():
    """Main evaluation loop."""
    print("="*60)
    print("IMAGE ARTIFACT EVALUATION TOOL")
    print("="*60)

    # Load existing evaluations
    df = load_existing_evaluations()
    print(f"\nLoaded {len(df)} existing evaluations from CSV")

    # Get all customer IDs
    customer_ids = get_all_customer_ids()
    print(f"Found {len(customer_ids)} customers with generated images")

    # Get pending evaluations
    pending = get_pending_evaluations(df, customer_ids)
    print(f"Pending evaluations: {len(pending)}")

    if not pending:
        print("\nAll images have been evaluated!")
        return

    print(f"\nStarting evaluation of {len(pending)} images...")
    print("Press 'q' at any time to quit and save progress.\n")

    evaluated_count = 0

    try:
        for record in pending:
            image_path = BASE_PATH / record["path_of_image"]

            result = display_image_and_get_evaluation(
                image_path=image_path,
                customer_id=record["customer_id"],
                product_id=record["product_id"],
                version=record["version"],
                version_of_final=record["version_of_final"],
                is_final=record["final"] == 1,
                passed_judge=record["passed_judge"]
            )

            if result is not None:
                # Add artifact results to record
                artifact, text_only_artifact = result
                record["artifact"] = artifact
                record["text_only_artifact"] = text_only_artifact

                # Add to dataframe
                new_row = pd.DataFrame([record])
                df = pd.concat([df, new_row], ignore_index=True)
                evaluated_count += 1

                # Save after each evaluation to prevent data loss
                save_evaluations(df)
                pct = 100 * evaluated_count / len(pending)
                remaining = len(pending) - evaluated_count
                print(f"Progress: {evaluated_count}/{len(pending)} ({pct:.1f}%) - {remaining} remaining")

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user.")

    finally:
        # Final save
        save_evaluations(df)
        print(f"\nTotal evaluations in CSV: {len(df)}")
        print(f"New evaluations this session: {evaluated_count}")

        # Show summary statistics
        if not df.empty:
            print("\n--- Summary Statistics ---")
            print(f"Total evaluated: {len(df)}")
            artifact_count = df['artifact'].sum()
            artifact_pct = 100 * df['artifact'].mean()
            print(f"With artifacts: {artifact_count} ({artifact_pct:.1f}%)")
            print(f"Without artifacts: {len(df) - artifact_count} ({100 - artifact_pct:.1f}%)")

            # Additional stats
            if 'incomplete' in df.columns:
                incomplete_count = df['incomplete'].sum()
                print(f"Incomplete records: {incomplete_count}")

            if 'final' in df.columns:
                final_count = df['final'].sum()
                print(f"Final versions evaluated: {final_count}")

            if 'passed_judge' in df.columns:
                passed_count = df['passed_judge'].sum()
                judge_evaluated = df['passed_judge'].notna().sum()
                if judge_evaluated > 0:
                    print(f"Passed judge: {passed_count}/{judge_evaluated} ({100*passed_count/judge_evaluated:.1f}%)")


if __name__ == "__main__":
    main()
