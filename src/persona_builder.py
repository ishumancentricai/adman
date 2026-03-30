# src/pers_build.py

"""
- Parses the CSV
- Builds personalized persona descriptions per level (L1–L3)  (L0 skipped)
- Saves into /persona_descriptions/ as: pers_<level>_<PID>.txt
- Skips rows with missing/blank PID and logs their LimeSurvey run ID to /logs/
"""

import pandas as pd
from config import (
    CUSTOMER,
    CUSTOMER_DIR,
    PERSONA_DESCRIPTION_FILE,
    CUSTOMER_DIR,
    CLEANED_CUSTOMER_DATA_FILE,
    PID_COLUMN,
)


# === DESCRIPTION_COMPOSITION ===
DESCRIPTION_COMPOSITION = [
    "demographics",
    "environment",
    "occupation",
    "origin",
    "lifestyle",
    "pets",
    "relationship_status",
    "children",
    "appearance",
    "style_vibes",
]


# === PERSONA DESCRIPTION SNIPPETS ===
def get_pronouns(gender):
    gender = gender.lower()
    return {
        "male": ("He", "His"),
        "female": ("She", "Her"),
        "diverse": ("They", "Their"),
    }.get(gender, ("They", "Their"))


DESCRIPTION_SNIPPETS = {
    "demographics": lambda row: (
        f"The persona is a {row['gender'].lower()} individual, {int(row['age'])} years old, "
        f"born in {row['birthCountry']}, currently living in {row['residenceCity']}, {row['residenceCountry']}."
    ),
    "occupation": lambda row: (
        f"{get_pronouns(row['gender'])[0]} would describe {get_pronouns(row['gender'])[1].lower()} current occupation as "
        f"'{str(row['occupationType']).lower()}'. "
        f"{get_pronouns(row['gender'])[1]} most recent occupation title is "
        f"'{str(row['occupationTitle']).lower() if pd.notna(row.get('occupationTitle')) else 'not available'}', "
        f"and {get_pronouns(row['gender'])[0].lower()} earns an {str(row['incomeClass']).lower()} income."
    ),
    "environment": lambda row: (
        f"{get_pronouns(row['gender'])[0]} lives in a {row['housingType'].lower()}"
        + (
            f" ({row['housingType[other]']})"
            if row["housingType"] == "Other" and pd.notna(row["housingType[other]"])
            else ""
        )
        + f", in an area that can be described as '{row['residenceArea'].lower()}'. "
        f"The climate there is currently {row['climate'].lower()}."
    ),
    "origin": lambda row: (
        # Ethnicity sentence only if valid
        (
            f"{get_pronouns(row['gender'])[0]} identifies as {row['ethnicity'].lower()}."
            if pd.notna(row.get("ethnicity")) and str(row["ethnicity"]).strip()
            else ""
        )
        + (
            # Add family origin sentence only if applicable
            f" {get_pronouns(row['gender'])[1]} family origin has cultural or national ties to {row['originCountry']}"
            + (
                f" (other: {row['originCountry[other]']})"
                if row["originCountry"] == "Other"
                and pd.notna(row["originCountry[other]"])
                else ""
            )
            + "."
            if row["familyOrigin"] == "Yes"
            else ""
        )
    ),
    "lifestyle": lambda row: (
        f"{get_pronouns(row['gender'])[0]} regularly engages in the following activities: {row['regularActivities'].lower()}."
    ),
    "pets": lambda row: (
        f"{get_pronouns(row['gender'])[0]} owns a pet: {row['petType'].lower()}."
        if row["petOwnership"] == "Yes"
        else ""
    ),
    "relationship_status": lambda row: (
        f"{get_pronouns(row['gender'])[0]} is {row['relationshipStatus'].lower()}"
        + (
            f" (other: {row['relationshipStatus[other]'].lower()})"
            if row["relationshipStatus"] == "Other"
            and pd.notna(row["relationshipStatus[other]"])
            else ""
        )
        + "."
        if (row["relationshipStatus"] != "Single, no children.")
        or (pd.notna(row["childrenNumber"]) and int(row["childrenNumber"]) > 0)
        else ""
    ),
    "children": lambda row: (
        f"{get_pronouns(row['gender'])[1]} {int(row['childrenNumber'])} child(ren), "
        f"is/are in the age groups of "
        + ", ".join(
            [
                label
                for label, code in {
                    "'infants / toddlers (0–3 years)'": "childrenAge[INFA]",
                    "'young children (4–6 years)'": "childrenAge[YCHIL]",
                    "'children (7–12 years)'": "childrenAge[CHIL]",
                    "'teenagers (13–17 years)'": "childrenAge[TEEN]",
                    "'young adults (18–24 years)'": "childrenAge[YADUL]",
                    "'adults (25+ years)'": "childrenAge[ADUL]",
                }.items()
                if row.get(code) == "Yes"
            ]
        )
        + "."
        if pd.notna(row["childrenNumber"]) and int(row["childrenNumber"]) > 0
        else ""
    ),
    "appearance": lambda row: (
        lambda p: (
            f"{p[1]} appearance includes a "
            + (
                f"{row['hairStyle[other]'].lower()}"
                if row["hairStyle"] == "Other" and pd.notna(row["hairStyle[other]"])
                else row["hairStyle"].lower()
            )
            + " hairstyle, "
            + (
                f"{row['hairColor[other]'].lower()}"
                if row["hairColor"] == "Other" and pd.notna(row["hairColor[other]"])
                else row["hairColor"].lower()
            )
            + " hair color, and a(n) "
            + row["bodyType"].lower()
            + " build. "
            + (
                f"{p[1]} facial-hair style is {row['beardStyle'].lower()}."
                if row["gender"] in ["Male", "Diverse"] and pd.notna(row["beardStyle"])
                else ""
            )
            + (
                f"{p[0]} usually wears glasses."
                if row["glassesWearer"] == "Yes"
                else ""
            )
        )
    )(get_pronouns(row["gender"])),
    "style_vibes": lambda row: (
        f"{get_pronouns(row['gender'])[0]} prefers {row['colorPreference'].lower()} colors and likes to dress in "
        + ", ".join(
            [
                style
                for style, code in {
                    "'minimalist (simple cuts, neutral colors, clean look)'": "stylePreference[MIN]",
                    "'casual / streetwear (comfortable, sporty, sneakers, hoodies)'": "stylePreference[STREET]",
                    "'classic / timeless (button shirts, coats, elegant & neat)'": "stylePreference[CLASS]",
                    "'trendy / style-conscious (follows fashion trends, bold combinations)'": "stylePreference[TREND]",
                    "'practical / outdoor-ready (functional gear, jackets, weather-appropriate)'": "stylePreference[OUTD]",
                }.items()
                if row.get(code) == "Yes"
            ]
        )
        + " style."
        if pd.notna(row["colorPreference"])
        else ""
    ),
}


def build_persona_description() -> None:
    # Input data file
    cleaned_persona_data = CUSTOMER_DIR / CLEANED_CUSTOMER_DATA_FILE
    # Sep is tab for LimeSurvey export, comma for generic CSV
    df = pd.read_csv(cleaned_persona_data)

    # Select row for the specific customer
    customer_row = df[df[PID_COLUMN] == CUSTOMER].iloc[0]

    keys = DESCRIPTION_COMPOSITION
    lines = []
    for k in keys:
        fn = DESCRIPTION_SNIPPETS.get(k)
        if not fn:
            continue
        text = (fn(customer_row) or "").strip()
        if text:
            lines.append(text)

    description = "\n".join(lines)
    
    # Output data file
    persona_description_file_path = CUSTOMER_DIR / PERSONA_DESCRIPTION_FILE
    with open(persona_description_file_path, "w", encoding="utf-8") as f:
        f.write(description)


if __name__ == "__main__":
    pass
