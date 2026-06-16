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
PROVIDER = "openai"  # "openai" or "google"
BASE_PATH = Path(__file__).parent.resolve()
DATA_DIR = BASE_PATH / "data"
INPUT_CSV = DATA_DIR / f"{PROVIDER}_aggregated_evaluation.csv"

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

    selected_customers = ['5f6388bf9ae5ad07ba1a0173', '60ff26e2e30862fdb8531934', '66dfa84afd612ac5bf545010', '605f61f6a2529d666698844b', '6654b7668e8642303bbc2fa7', '60b3e6ad448226548d21fc4d', '63d83e6e48dd3f8da9458dd3', '66e0813b79955faf3b6017c3', '588e02b13326830001bc29cd', '5c28feff5ffcb7000128a541', '660b42d7411787277cf4a709', '5fbc56c0293d73121cdc1eb5', '5de1ac810f8b6721970ab78a', '5fadeb340e10471fdeb5fbf5', '5dd396e15a40f2398c38e0e7', '66c0f464e0ff62798027b486', '642b2d8a7b84306d49e2f4b0', '62fbeca46caf4abaaf74b5f1', '5c8db07eb1f89a0016777fe0', '6043001e416c7828f58c6c22', '5815369b94ecc900010b014e', '65de67e5627faa3ffb10dec8', '616b2966d0aee4f9879a656b', '6713eeea64ca1fb59cbcbce4', '66316ac602a513e2e74d5bdc', '663a5cc1f775076b324606c6', '672285407273e65dc89d6f75', '64012a2623351894f2917605', '662945728093113a43ab8077', '5a49d6a06d85f80001c25bc4', '6317511c1dd0acb82cf4b131', '63c06106314da889290931cd', '62349dcdec7c3f8e7e1bc90b', '5959cf88dd044900012cd375', '5a4ff86beedc32000141f232', '59d7c0f750f88f000161eb21', '66e89b61d7709ff7f27a76be', '5eac686d82cc012af1e953c7', '5d30790f326b0f0017a2a828', '65f70aaa07cf9f1befd47e7b', '64360d6189dec3ac8afc9a78', '643480e93c33701dfe923abd', '661f4cfbeabfb4f58a8ecb4c', '63d6873b65a6170d64a652f6', '65615a2b723ed05c242d6331', '5efe6daf6a2551378201893e', '629638416146d56a43e5e028', '5da8edc60f9c9c0017526948', '67215f2922159d42abf932a4', '62fe6312269d0e4ae894297b', '6565205a6f8163d131fa6918', '65ca9c729a13f898ae374552', '6604b10d7328536e5d9bb9db', '62b072828cfef4de9971753b', '62d43cee3d60ac98c1dcacc8', '6562177a48308eebbb6359ea', '66e9a6cf278a9806ebe7bd0b', '57027c1f68c641000e0c8970', '645179915e31c1aeb3dc8129', '63d7f3f795d4674597d533a1', '5a6b53f3e70ca900017ab75a', '5d11e762cc2d290016bc2364', '5ac5260dfa3b4e0001736dd3', '6529648ca1ade4b6c9e876f4', '66eddcfe67e927db6f0a4768', '668ebf594982c57c983cd910', '663d8f0edfd87b4c1f590b99', '66a276d13f40200c70c28109', '6691bc059b0182b0bdd7d682', '5e8caf1b063693041f6ac524', '5c3bdbf548ad2900017fac31', '666e02576e916cd782e9ae12', '638b951008b4843ccdbc4e15', '5c83394a3e93bf0001e143d6', '64345edc0fd9813bdd9ef181', '62cbf51439a76e8fccd8a495', '5d8a29c082fec30001d9c24a', '6422f34be2b8257efb345837', '66661b1242b581f199e49f4c', '62979d6078be475dbbcb0149', '656ea0d1171b158205065fbb', '600e13b23c75bc67ab184745', '5e34325430e5263f0bebb32d', '63bcbf7e0ac7e876387ef720', '629e2781136f152f6592f171', '5d29d6bd30fa6f0015cd4a75', '6478e5fc76f5ecbc4f0ed353', '6643a8837540ee80f2d5be45', '66c7dfc0b974270d47eafde3', '643bd22e67aeded15060e447', '56a914f0119265000c6e13c5', '65ccf0ceb42202f1d53bedb7', '6632404d369dbb28fc6bf5d5', '6630e7089d4946cc85055e34', '66d1fdd400acfdbc181f41e0', '617323e630f3fa36361cb35b', '65fa95fafeda35caf378b6f9', '5da35e177f1f7d001405f60b', '5ccf87bfb46c7000010d8428', '5ac4c721e1099600016a823b']


    # Filter out any rows with customer_id in CELEBRITIES
    df = df[~df["customer_id"].isin(CELEBRITIES)]

    # Only keep records if the customer_id is in the selected_customers list
    df = df[df["customer_id"].isin(selected_customers)]

    # Print left number of records after filtering
    print(f"After filtering out celebrities, {len(df)} records remain.\n")

    # --- Define image sets ---
    i_first = df[df["version"] == 1]

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
