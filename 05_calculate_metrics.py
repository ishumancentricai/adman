"""
Calculate Metrics Script

Computes Passed Rate (PR), Artifact Rate (AR), and Relaxed Artifact Rate (rAR)
across three image sets (I_first, I_passed, I_selected) per product and overall.

Definitions:
  - AR  = (artifact + text_only_artifact) / total   (any defect)
  - rAR = artifact / total                           (excluding text-only artifacts)
"""

import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
BASE_PATH = Path(__file__).parent.resolve()
DATA_DIR = BASE_PATH / "data"
INPUT_CSV = DATA_DIR / "image_evaluation_corrected.csv"

PRODUCT_NAMES = {
    1: "Electric Vehicle (P1)",
    2: "Luxury Watch (P2)",
    3: "Laundry Detergent (P3)",
    4: "Soft Drink (P4)",
}


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    Compute PR, AR, and rAR for a given set of images.

    PR  = passed_judge / total
    AR  = (artifact | text_only_artifact) / total
    rAR = artifact / total
    """
    n = len(df)
    if n == 0:
        return {"n": 0, "PR": 0.0, "AR": 0.0, "rAR": 0.0}

    pr = df["passed_judge"].sum() / n
    ar = ((df["artifact"] == 1) | (df["text_only_artifact"] == 1)).sum() / n
    rar = df["artifact"].sum() / n

    return {"n": n, "PR": pr, "AR": ar, "rAR": rar}


def main():
    print("=" * 60)
    print("CALCULATE METRICS")
    print("=" * 60)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} records from {INPUT_CSV.name}\n")

    # Known celebrities to exclude (though checking customer dir existence helps too)
    CELEBRITIES = [
        "batman", "brucewayne", "hermione", "kevin", "leonardo", "michelle",
        "morpheus", "lee", "don", "rajesh", "rachel"
    ]

    # Filter out any rows with customer_id in CELEBRITIES
    df = df[~df["customer_id"].isin(CELEBRITIES)]

    # Print left number of records after filtering
    print(f"After filtering out celebrities, {len(df)} records remain.\n")

    # --- Define image sets ---
    i_first = df[df["version"] == 0]

    i_passed = df[df["passed_judge"] == 1]

    i_selected = df[df["final"] == 1]

    sets = {
        "I_first": i_first,
        "I_passed": i_passed,
        "I_selected": i_selected,
    }

    # --- Compute per-product and overall ---
    products = sorted(df["product_id"].unique())
    results = {}  # results[set_name][product_id] = metrics dict

    for set_name, subset in sets.items():
        results[set_name] = {}
        for pid in products:
            product_subset = subset[subset["product_id"] == pid]
            results[set_name][pid] = compute_metrics(product_subset)
        # Overall
        results[set_name]["overall"] = compute_metrics(subset)

    # --- Print results ---
    header = (
        f"{'Product':<25s}"
        f"{'':>3s}"
        f"{'PR':>6s} {'AR':>6s} {'rAR':>6s}"
        f"{'':>3s}"
        f"{'PR':>6s} {'AR':>6s} {'rAR':>6s}"
        f"{'':>3s}"
        f"{'PR':>6s} {'AR':>6s} {'rAR':>6s}"
    )

    set_header = (
        f"{'':25s}"
        f"{'':>3s}"
        f"{'--- I_first ---':^20s}"
        f"{'':>1s}"
        f"{'--- I_passed ---':^20s}"
        f"{'':>1s}"
        f"{'--- I_selected ---':^20s}"
    )

    print(set_header)
    print(header)
    print("-" * len(header))

    def fmt(val: float) -> str:
        return f"{val:.2f}"

    for pid in products:
        name = PRODUCT_NAMES.get(pid, f"Product {pid}")
        f = results["I_first"][pid]
        p = results["I_passed"][pid]
        s = results["I_selected"][pid]
        print(
            f"{name:<25s}"
            f"{'':>3s}"
            f"{fmt(f['PR']):>6s} {fmt(f['AR']):>6s} {fmt(f['rAR']):>6s}"
            f"{'':>3s}"
            f"{fmt(p['PR']):>6s} {fmt(p['AR']):>6s} {fmt(p['rAR']):>6s}"
            f"{'':>3s}"
            f"{fmt(s['PR']):>6s} {fmt(s['AR']):>6s} {fmt(s['rAR']):>6s}"
        )

    print("-" * len(header))
    f = results["I_first"]["overall"]
    p = results["I_passed"]["overall"]
    s = results["I_selected"]["overall"]
    print(
        f"{'Overall Mean':<25s}"
        f"{'':>3s}"
        f"{fmt(f['PR']):>6s} {fmt(f['AR']):>6s} {fmt(f['rAR']):>6s}"
        f"{'':>3s}"
        f"{fmt(p['PR']):>6s} {fmt(p['AR']):>6s} {fmt(p['rAR']):>6s}"
        f"{'':>3s}"
        f"{fmt(s['PR']):>6s} {fmt(s['AR']):>6s} {fmt(s['rAR']):>6s}"
    )

    # --- Print set sizes ---
    print("\n--- Set Sizes ---")
    for set_name, subset in sets.items():
        print(f"{set_name}: {len(subset)} images")
        for pid in products:
            n = len(subset[subset["product_id"] == pid])
            print(f"  {PRODUCT_NAMES.get(pid, f'P{pid}')}: {n}")

    # --- Generate LaTeX table ---
    print("\n\n--- LaTeX Table Values ---\n")
    print(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}} l c c c c c c c c c @{}}")
    print(r"\toprule")
    print(
        r"& \multicolumn{3}{c}{\textbf{First Generation ($I_{\text{first}}$)}} "
        r"& \multicolumn{3}{c}{\textbf{Judge Passed ($I_{\text{passed}}$)}} "
        r"& \multicolumn{3}{c}{\textbf{Final Selected ($I_{\text{selected}}$)}} \\"
    )
    print(r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}")
    print(
        r"\textbf{Product} & \textbf{PR} & \textbf{AR} & \textbf{rAR} "
        r"& \textbf{PR} & \textbf{AR} & \textbf{rAR} "
        r"& \textbf{PR} & \textbf{AR} & \textbf{rAR} \\"
    )
    print(r"\midrule")

    latex_names = {
        1: "Electric Vehicle (P1)",
        2: "Luxury Watch (P2)",
        3: "Laundry Detergent (P3)",
        4: "Soft Drink (P4)",
    }

    for pid in products:
        name = latex_names.get(pid, f"Product {pid}")
        f = results["I_first"][pid]
        p = results["I_passed"][pid]
        s = results["I_selected"][pid]
        print(
            f"{name:<25s} & {fmt(f['PR'])} & {fmt(f['AR'])} & {fmt(f['rAR'])} "
            f"& {fmt(p['PR'])} & {fmt(p['AR'])} & {fmt(p['rAR'])} "
            f"& {fmt(s['PR'])} & {fmt(s['AR'])} & {fmt(s['rAR'])} \\\\"
        )

    print(r"\midrule")
    f = results["I_first"]["overall"]
    p = results["I_passed"]["overall"]
    s = results["I_selected"]["overall"]
    print(
        rf"\textbf{{Overall Mean}} & \textbf{{{fmt(f['PR'])}}} & \textbf{{{fmt(f['AR'])}}} & \textbf{{{fmt(f['rAR'])}}} "
        rf"& \textbf{{{fmt(p['PR'])}}} & \textbf{{{fmt(p['AR'])}}} & \textbf{{{fmt(p['rAR'])}}} "
        rf"& \textbf{{{fmt(s['PR'])}}} & \textbf{{{fmt(s['AR'])}}} & \textbf{{{fmt(s['rAR'])}}} \\\\"
    )
    print(r"\bottomrule")
    print(r"\end{tabular*}")


if __name__ == "__main__":
    main()
