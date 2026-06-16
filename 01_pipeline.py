from src.dynamic_config import DynamicConfig
import argparse
import pandas as pd
import os
import re
from dotenv import load_dotenv
from openai import OpenAI
import json
import base64
from PIL import Image
from utils import logger, load_prompt, product_by_id, get_openai_resp, load_and_encode_image_b64
import time

try:
    from google import genai
    from google.genai import errors as google_errors
    from google.genai import types as google_types
    import httpx
except ImportError:
    genai = None
    google_errors = None
    google_types = None
    httpx = None

DYN_CONF = DynamicConfig()

PARTICIPANT_DATA_DF = pd.read_csv(
    DYN_CONF.PARTICIPANT_DATA_FILE,
    sep="\t",
    quotechar='"',
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


def build_persona_description(DYNAMIC_CONFIG) -> None:
    # Input data file
    cleaned_persona_data = DYNAMIC_CONFIG.CUSTOMER_DIR / DYNAMIC_CONFIG.CLEANED_CUSTOMER_DATA_FILE
    # Sep is tab for LimeSurvey export, comma for generic CSV
    df = pd.read_csv(cleaned_persona_data)

    # Select row for the specific customer
    customer_row = df[df[DYNAMIC_CONFIG.PID_COLUMN] == DYNAMIC_CONFIG.CUSTOMER].iloc[0]

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
    persona_description_file_path = DYNAMIC_CONFIG.CUSTOMER_DIR / DYNAMIC_CONFIG.PERSONA_DESCRIPTION_FILE
    with open(persona_description_file_path, "w", encoding="utf-8") as f:
        f.write(description)


def clean_data(DYNAMIC_CONFIG,
               df) -> pd.DataFrame:
    customer_dir = DYNAMIC_CONFIG.CUSTOMER_DIR
    customer_data_file = DYNAMIC_CONFIG.CUSTOMER_DATA_FILE
    cleaned_customer_data_file = DYNAMIC_CONFIG.CLEANED_CUSTOMER_DATA_FILE

    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_path = os.path.join(base_path, customer_dir, customer_data_file)
    output_path = os.path.join(base_path, customer_dir, cleaned_customer_data_file)

    # Make sure customer dir exists, create if not
    os.makedirs(os.path.join(base_path, customer_dir), exist_ok=True)

    # Load data (headers are already question codes)
    # df = pd.read_csv(input_path, sep=",", quotechar='"')
    # df = pd.read_csv(input_path, sep="\t", quotechar='"')

    # Only keep row of the specific customer
    df = df[df["PID"] == DYNAMIC_CONFIG.CUSTOMER]

    # Store as customer_data_file for reference sep="\t", quotechar='"'
    df.to_csv(input_path, sep="\t", quotechar='"', index=False)

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
    os.makedirs(os.path.join(base_path, DYNAMIC_CONFIG.DATA_DIR), exist_ok=True)

    # Save to CSV
    cleaned.to_csv(output_path, index=False)
    print(
        f"Cleaned data saved to: {output_path} "
        f"(kept {len(cleaned)}/{len(df)} rows; dropped {len(df) - len(cleaned)})"
    )

    return cleaned


def _require_google_client(DYNAMIC_CONFIG):
    if genai is None or google_types is None:
        raise ImportError(
            "Google provider requires the 'google-genai' package. "
            "Install it with: pip install google-genai"
        )

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "Google provider requires GEMINI_API_KEY or GOOGLE_API_KEY in the environment."
        )
    return genai.Client(
        api_key=api_key,
        http_options=google_types.HttpOptions(timeout=DYNAMIC_CONFIG.GOOGLE_TIMEOUT_MS),
    )


def _google_response_dict(response) -> dict:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if hasattr(response, "to_json_dict"):
        return response.to_json_dict()
    return {"text": getattr(response, "text", None)}


def _json_safe(value):
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {key: _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    return value


def _dump_json_safe(value, indent: int = 4) -> str:
    return json.dumps(_json_safe(value), indent=indent)


def _parse_model_json(output_text: str) -> dict:
    text = output_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        for index, char in enumerate(text):
            if char != "{":
                continue
            try:
                data, _ = decoder.raw_decode(text[index:])
                break
            except json.JSONDecodeError:
                continue
        else:
            preview = text[:500] if text else "<empty response>"
            raise ValueError(f"No JSON object found in model output: {preview}")

    if not isinstance(data, dict):
        raise ValueError("model JSON output is not a dictionary.")
    return data


def _google_text_response(response) -> tuple[str, str, dict]:
    output_text = getattr(response, "text", None)
    if not output_text:
        raise ValueError("Google API response missing text output.")
    return output_text.strip(), "", _google_response_dict(response)


def _google_image_part(image_b64: str):
    return google_types.Part.from_bytes(
        data=base64.b64decode(image_b64),
        mime_type="image/png",
    )


def _google_thinking_config(DYNAMIC_CONFIG):
    if not DYNAMIC_CONFIG.GOOGLE_THINKING_ENABLED:
        return None

    effort = DYNAMIC_CONFIG.GOOGLE_THINKING_EFFORT.upper()
    try:
        thinking_level = getattr(google_types.ThinkingLevel, effort)
    except AttributeError as exc:
        raise ValueError(
            "GOOGLE_THINKING_EFFORT must be one of: "
            "minimal, low, medium, high"
        ) from exc

    return google_types.ThinkingConfig(
        include_thoughts=True,
        thinking_level=thinking_level,
    )


def _google_generate_config(DYNAMIC_CONFIG, system_prompt: str):
    return google_types.GenerateContentConfig(
        system_instruction=system_prompt,
        thinking_config=_google_thinking_config(DYNAMIC_CONFIG),
    )


def _is_retryable_google_error(exc: Exception) -> bool:
    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return True

    if google_errors is not None and isinstance(exc, google_errors.APIError):
        status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
        return status_code in {429, 500, 502, 503, 504}

    return False


def _google_retry_after_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None

    retry_after = headers.get("retry-after")
    if retry_after is None:
        return None

    try:
        return float(retry_after)
    except ValueError:
        return None


def _google_generate_content(client, DYNAMIC_CONFIG, **kwargs):
    attempts = DYNAMIC_CONFIG.GOOGLE_RETRY_ATTEMPTS
    initial_delay = DYNAMIC_CONFIG.GOOGLE_RETRY_INITIAL_DELAY_SECONDS
    max_delay = DYNAMIC_CONFIG.GOOGLE_RETRY_MAX_DELAY_SECONDS

    for attempt in range(1, attempts + 1):
        try:
            return client.models.generate_content(**kwargs)
        except Exception as exc:
            if attempt >= attempts or not _is_retryable_google_error(exc):
                raise

            retry_after = _google_retry_after_seconds(exc)
            delay = retry_after or min(max_delay, initial_delay * (2 ** (attempt - 1)))
            logger.warning(
                "Google API call failed with a retryable error "
                f"({type(exc).__name__}: {exc}). "
                f"Retrying in {delay:.0f}s ({attempt}/{attempts})."
            )
            time.sleep(delay)

    raise RuntimeError("Google API retry loop exited unexpectedly.")


def ask_text_model(system_prompt, user_prompt, DYNAMIC_CONFIG) -> tuple[str, str, dict]:
    if DYNAMIC_CONFIG.PROVIDER == "google":
        client = _require_google_client(DYNAMIC_CONFIG)
        response = _google_generate_content(
            client,
            DYNAMIC_CONFIG,
            model=DYNAMIC_CONFIG.GOOGLE_TEXT_MODEL,
            contents=user_prompt,
            config=_google_generate_config(DYNAMIC_CONFIG, system_prompt),
        )
        return _google_text_response(response)

    return ask_openai_3(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        DYNAMIC_CONFIG=DYNAMIC_CONFIG,
    )


def ask_text_model_with_judgements(
    system_prompt: str,
    user_prompt: str,
    *judgements,
    DYNAMIC_CONFIG,
) -> tuple[str, str, dict]:
    if DYNAMIC_CONFIG.PROVIDER == "google":
        client = _require_google_client(DYNAMIC_CONFIG)
        joined_judgements = "\n\n".join(
            f"# judgement {i}\n{v}\n\n" for i, v in enumerate(judgements)
        )
        response = _google_generate_content(
            client,
            DYNAMIC_CONFIG,
            model=DYNAMIC_CONFIG.GOOGLE_TEXT_MODEL,
            contents=[user_prompt, joined_judgements],
            config=_google_generate_config(DYNAMIC_CONFIG, system_prompt),
        )
        return _google_text_response(response)

    return ask_openai_2(
        system_prompt,
        user_prompt,
        *judgements,
        DYNAMIC_CONFIG=DYNAMIC_CONFIG,
    )


def ask_judge_model(
    system_prompt: str,
    user_prompt: str,
    image_generation_prompt: str,
    reference_image: str,
    generated_image: str,
    DYNAMIC_CONFIG,
):
    if DYNAMIC_CONFIG.PROVIDER == "google":
        client = _require_google_client(DYNAMIC_CONFIG)
        response = _google_generate_content(
            client,
            DYNAMIC_CONFIG,
            model=DYNAMIC_CONFIG.GOOGLE_TEXT_MODEL,
            contents=[
                user_prompt,
                image_generation_prompt,
                _google_image_part(reference_image),
                _google_image_part(generated_image),
            ],
            config=_google_generate_config(DYNAMIC_CONFIG, system_prompt),
        )
        return _google_text_response(response)

    return ask_openai_4(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_generation_prompt=image_generation_prompt,
        reference_image=reference_image,
        generated_image=generated_image,
        DYNAMIC_CONFIG=DYNAMIC_CONFIG,
    )


def ask_image_model(prompt: str, ref_image_path, DYNAMIC_CONFIG) -> bytes:
    if DYNAMIC_CONFIG.PROVIDER == "google":
        client = _require_google_client(DYNAMIC_CONFIG)
        logger.info(
            "Calling Google image model "
            f"{DYNAMIC_CONFIG.GOOGLE_IMAGE_MODEL} "
            f"({DYNAMIC_CONFIG.GOOGLE_IMAGE_ASPECT_RATIO}, "
            f"{DYNAMIC_CONFIG.GOOGLE_IMAGE_SIZE}); this can take a few minutes"
        )
        t0 = time.time()
        with Image.open(ref_image_path) as ref_image:
            response = _google_generate_content(
                client,
                DYNAMIC_CONFIG,
                model=DYNAMIC_CONFIG.GOOGLE_IMAGE_MODEL,
                contents=[prompt, ref_image],
                config=google_types.GenerateContentConfig(
                    response_modalities=["TEXT", "IMAGE"],
                    image_config=google_types.ImageConfig(
                        aspect_ratio=DYNAMIC_CONFIG.GOOGLE_IMAGE_ASPECT_RATIO,
                        image_size=DYNAMIC_CONFIG.GOOGLE_IMAGE_SIZE,
                    ),
                ),
            )
        logger.info(
            "Google image model returned in "
            f"{time.time() - t0:.2f} seconds"
        )

        for part in response.parts or []:
            if part.text is not None:
                logger.info(f"Google image model text response: {part.text.strip()}")
                continue
            inline_data = getattr(part, "inline_data", None)
            if inline_data and inline_data.data:
                data = inline_data.data
                if isinstance(data, str):
                    return base64.b64decode(data)
                return data
            image = part.as_image()
            if image is not None and image.image_bytes:
                return image.image_bytes
        raise ValueError("Google image response did not include image bytes.")

    return ask_openai(
        prompt=prompt,
        ref_image_path=ref_image_path,
        DYNAMIC_CONFIG=DYNAMIC_CONFIG,
    )


def ask_openai_3(system_prompt, user_prompt, DYNAMIC_CONFIG) -> tuple[str, str, dict]:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    req = {
        "model": DYNAMIC_CONFIG.OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "reasoning": {"effort": DYNAMIC_CONFIG.OPENAI_EFFORT, "summary": DYNAMIC_CONFIG.OPENAI_SUMMARY},
        "text": {"verbosity": DYNAMIC_CONFIG.OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def generate_prompt(version: int, DYNAMIC_CONFIG) -> None:
    # Sanity check
    if not DYNAMIC_CONFIG.SELECTED_LEVEL_ID == 3:
        raise ValueError(f"Only level 3 prompts are valid, but level {DYNAMIC_CONFIG.SELECTED_LEVEL_ID} was selected.")

    # === Generator input paths ===
    # Load system prompt template
    system_prompt_path = DYNAMIC_CONFIG.PROMPT_TEMPLATES_DIR / DYNAMIC_CONFIG.PROMPT_GENERATOR_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = DYNAMIC_CONFIG.PROMPT_TEMPLATES_DIR / DYNAMIC_CONFIG.PROMPT_GENERATOR_USER_PROMPT_FILE
    user_prompt_template = load_prompt(user_prompt_path)
    # Load persona description
    persona_description_path = DYNAMIC_CONFIG.CUSTOMER_DIR / DYNAMIC_CONFIG.PERSONA_DESCRIPTION_FILE
    persona_description = load_prompt(persona_description_path)

    # Load product info
    product_info = product_by_id(DYNAMIC_CONFIG.SELECTED_PRODUCT_ID)

    user_prompt = user_prompt_template.format(
        PRODUCT_NAME=product_info["name"],
        PRODUCT_TYPE=product_info["type"],
        persona_description=persona_description,
    )

    # Log full prompt
    log_prompt = f"{system_prompt}\n\n{user_prompt}"
    prompt_generator_log_path = DYNAMIC_CONFIG.PROMPT_LOG_DIR / DYNAMIC_CONFIG.PROMPT_GENERATOR_LOG
    with open(prompt_generator_log_path, "w", encoding="utf-8") as f:
        f.write(log_prompt.strip())

    output_text, reasoning, _resp_dict = ask_text_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        DYNAMIC_CONFIG=DYNAMIC_CONFIG
    )

    # Save generated prompt
    generated_prompt_path = (
        DYNAMIC_CONFIG.GENERATED_PROMPTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{version}.txt"
    )
    with open(generated_prompt_path, "w", encoding="utf-8") as f:
        f.write(output_text.strip())
    # Save reasoning
    generated_prompt_reasoning_path = (
        DYNAMIC_CONFIG.GENERATED_PROMPTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{version}_reasoning.txt"
    )
    with open(generated_prompt_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save OpenAI full response
    resp_file_path = (
        DYNAMIC_CONFIG.GENERATED_PROMPTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{version}_prompt_resp.json"
    )
    with open(resp_file_path, "w", encoding="utf-8") as f:
        f.write(_dump_json_safe(_resp_dict, indent=4))


def _normalize_comparison(data: dict) -> dict:
    """Normalize comparison output data."""
    if not data:
        raise ValueError("Empty comparison data")

    norm = {}

    def _strip_sec(dat):
        if isinstance(dat, dict):
            return {k: _strip_sec(v) for k, v in dat.items()}
        elif isinstance(dat, list):
            return [_strip_sec(item) for item in dat]
        elif isinstance(dat, str):
            return dat.strip()
        else:
            return dat

    if "best_version" not in data.keys():
        raise ValueError("Comparison data missing 'best_version' field.")
    if "explanation" not in data.keys():
        raise ValueError("Comparison data missing 'explanation' field.")

    for key, value in data.items():
        norm[key] = _strip_sec(value)

    return norm


def ask_openai_2(
    system_prompt: str,
    user_prompt: str,
    *judgements,
    DYNAMIC_CONFIG,
) -> tuple[str, str, dict]:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    req = {
        "model": DYNAMIC_CONFIG.OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_text", "text": "\n\n".join(f"# judgement {i}\n{v}\n\n" for i, v in enumerate(judgements))},
            ]},
        ],
        "reasoning": {"effort": DYNAMIC_CONFIG.OPENAI_EFFORT, "summary": DYNAMIC_CONFIG.OPENAI_SUMMARY},
        "text": {"verbosity": DYNAMIC_CONFIG.OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def compare_versions(DYNAMIC_CONFIG) -> None:
    # Load system prompt template
    system_prompt_path = DYNAMIC_CONFIG.PROMPT_TEMPLATES_DIR / DYNAMIC_CONFIG.SELECTOR_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = DYNAMIC_CONFIG.PROMPT_TEMPLATES_DIR / DYNAMIC_CONFIG.SELECTOR_USER_PROMPT_FILE
    user_prompt = load_prompt(user_prompt_path)

    # Load judgements
    judgements = []
    for v in DYNAMIC_CONFIG.VERSION_IDS:
        judgement_path = (
            DYNAMIC_CONFIG.JUDGEMENTS_DIR /
            f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{v}_judgement.json"
        )
        # Double JSON might be unnecessary - keep for consistent formatting
        with open(judgement_path, "r", encoding="utf-8") as f:
            judgement = json.load(f)
        if not isinstance(judgement, dict):
            raise ValueError(f"Invalid judgement format in {judgement_path}")
        judgements.append(json.dumps(judgement))

    output_text, reasoning, _resp_dict = ask_text_model_with_judgements(
        system_prompt,
        user_prompt,
        *judgements,
        DYNAMIC_CONFIG=DYNAMIC_CONFIG
    )

    # Save comparison output
    comparison_path = (
        DYNAMIC_CONFIG.JUDGEMENTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_version_comparison.json"
    )
    try:
        comparison_json = _parse_model_json(output_text)
        comparison_json = _normalize_comparison(comparison_json)
    except Exception as e:
        raw_comparison_path = comparison_path.with_suffix(".raw.txt")
        with open(raw_comparison_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        raise ValueError(f"Failed to parse comparison output as JSON: {e}")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(comparison_json, indent=4, ensure_ascii=False))
    # Save reasoning
    comparison_reasoning_path = (
        DYNAMIC_CONFIG.JUDGEMENTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_version_comparison_reasoning.txt"
    )
    with open(comparison_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save full response
    resp_path = (
        DYNAMIC_CONFIG.JUDGEMENTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_version_comparison_response.json"
    )
    with open(resp_path, "w", encoding="utf-8") as f:
        f.write(_dump_json_safe(_resp_dict, indent=4))


def get_best_version(DYNAMIC_CONFIG) -> tuple[int, str]:
    comparison_file_path = (
        DYNAMIC_CONFIG.JUDGEMENTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_version_comparison.json"
    )
    with open(comparison_file_path, "r", encoding="utf-8") as f:
        content = json.load(f)
    if not content:
        raise ValueError("Empty comparison file content.")
    if not isinstance(content, dict):
        raise ValueError("Comparison file content is not a dictionary.")

    best_version = content.get("best_version", None)
    if best_version is None:
        raise ValueError("Best version not found in comparison file.")

    explanation = content.get("explanation", None)
    if explanation is None:
        raise ValueError("Explanation not found in comparison file.")

    return best_version, explanation


def ask_openai(prompt: str, ref_image_path, DYNAMIC_CONFIG) -> bytes:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(ref_image_path, "rb") as ref_image:
        result = client.images.edit(
            model=DYNAMIC_CONFIG.IMAGE_MODEL,
            image=[ref_image],
            prompt=prompt,
            quality=DYNAMIC_CONFIG.IMAGE_QUALITY,
            input_fidelity=DYNAMIC_CONFIG.INPUT_FIDELITY,
            size=DYNAMIC_CONFIG.IMAGE_SIZE,
            background=DYNAMIC_CONFIG.IMAGE_BACKGROUND,
            n=DYNAMIC_CONFIG.IMAGE_COUNT,
        )

    image_base64 = result.data[0].b64_json
    return base64.b64decode(image_base64)


def generate_image(version, DYNAMIC_CONFIG) -> None:
    # Load system prompt template
    system_prompt_path = DYNAMIC_CONFIG.PROMPT_TEMPLATES_DIR / DYNAMIC_CONFIG.IMAGE_GENERATOR_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    system_prompt = system_prompt.format(
        PRODUCT_TYPE=product_by_id(DYNAMIC_CONFIG.SELECTED_PRODUCT_ID)["type"],
        PRODUCT_NAME=product_by_id(DYNAMIC_CONFIG.SELECTED_PRODUCT_ID)["name"],
    )
    # Load generated image description
    generated_image_prompt_path = (
        DYNAMIC_CONFIG.CUSTOMER_DIR /
        DYNAMIC_CONFIG.GENERATED_PROMPTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{version}.txt"
    )
    generated_image_prompt = load_prompt(generated_image_prompt_path)

    # Combine system prompt with generated prompt
    full_prompt = f"{system_prompt}\n{generated_image_prompt}"

    # Log full prompt
    image_generator_log_path = (
        DYNAMIC_CONFIG.PROMPT_LOG_DIR /
        DYNAMIC_CONFIG.IMAGE_PROMPT_LOG.format(
            CUSTOMER=DYNAMIC_CONFIG.CUSTOMER,
            SELECTED_PRODUCT_ID=DYNAMIC_CONFIG.SELECTED_PRODUCT_ID,
            SELECTED_LEVEL_ID=DYNAMIC_CONFIG.SELECTED_LEVEL_ID,
            version=version
        )
    )
    with open(image_generator_log_path, "w", encoding="utf-8") as f:
        f.write(full_prompt.strip())

    # Load reference image
    ref_image_name = product_by_id(DYNAMIC_CONFIG.SELECTED_PRODUCT_ID)["ref_image"]
    reference_image_path = (
        DYNAMIC_CONFIG.REF_IMAGE_DIR / ref_image_name
    )

    # Generate image
    image_bytes = ask_image_model(
        prompt=full_prompt,
        ref_image_path=reference_image_path,
        DYNAMIC_CONFIG=DYNAMIC_CONFIG
    )

    # Save generated image
    generated_image_path = (
        DYNAMIC_CONFIG.GENERATED_IMAGES_DIR /
        DYNAMIC_CONFIG.GENERATED_IMAGE_FILE.format(
            CUSTOMER=DYNAMIC_CONFIG.CUSTOMER,
            SELECTED_PRODUCT_ID=DYNAMIC_CONFIG.SELECTED_PRODUCT_ID,
            SELECTED_LEVEL_ID=DYNAMIC_CONFIG.SELECTED_LEVEL_ID,
            version=version
        )
    )
    with open(generated_image_path, "wb") as wf:
        wf.write(image_bytes)


JUDGE_OUTPUT_SECTIONS = [
    "judgement",
    "overall_assessment",
    "inventory",
    "anomalies",
    "reasoning",
]


def _normalize_judgement(data: dict) -> dict:
    """Normalize judgments: 'pass' = PASS, 'fail' = FAIL.
    Any non-'pass / fail' value is treated as error.
    """
    if not data:
        raise ValueError("Empty judgement data")

    sections = JUDGE_OUTPUT_SECTIONS
    norm = {}

    def _strip_sec(dat):
        if isinstance(dat, dict):
            return {k: _strip_sec(v) for k, v in dat.items()}
        elif isinstance(dat, list):
            return [_strip_sec(item) for item in dat]
        elif isinstance(dat, str):
            return dat.strip()
        else:
            return dat

    for sec in sections:
        if sec == "judgement":
            v = data.get("judgement")
            if not v:
                raise ValueError("Missing 'judgement' section in judgement")

            if v.strip().lower() == "pass":
                norm[sec] = "pass"
            elif v.strip().lower() == "fail":
                norm[sec] = "fail"
            else:
                raise ValueError("Invalid value for 'judgement' section in judgement")
            continue

        sec_data = data.get(sec, None)
        sec_norm = _strip_sec(sec_data)
        norm[sec] = sec_norm

    return norm


def check_pass(version: int, DYNAMIC_CONFIG) -> bool:
    judgement_path = (
        DYNAMIC_CONFIG.JUDGEMENTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{version}_judgement.json"
    )
    # Double JSON might be unnecessary - keep for consistent formatting
    with open(judgement_path, "r", encoding="utf-8") as f:
        judgement = json.load(f)
    if not isinstance(judgement, dict):
        raise ValueError(f"Invalid judgement format in {judgement_path}")

    # Check if the judgement passed
    return judgement.get("judgement") == "pass"


def ask_openai_4(
    system_prompt: str,
    user_prompt: str,
    image_generation_prompt: str,
    reference_image: bytes,
    generated_image: bytes,
    DYNAMIC_CONFIG
):
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    req = {
        "model": DYNAMIC_CONFIG.OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_text", "text": image_generation_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{reference_image}"},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{generated_image}"},
                ]
             },
        ],
        "reasoning": {"effort": DYNAMIC_CONFIG.OPENAI_EFFORT, "summary": DYNAMIC_CONFIG.OPENAI_SUMMARY},
        "text": {"verbosity": DYNAMIC_CONFIG.OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def judge(version: int, DYNAMIC_CONFIG) -> None:
    # Load system prompt template
    system_prompt_path = DYNAMIC_CONFIG.PROMPT_TEMPLATES_DIR / DYNAMIC_CONFIG.JUDGE_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = DYNAMIC_CONFIG.PROMPT_TEMPLATES_DIR / DYNAMIC_CONFIG.JUDGE_USER_PROMPT_FILE
    user_prompt = load_prompt(user_prompt_path)

    # Load image generation prompt
    generated_image_prompt_path = (
        DYNAMIC_CONFIG.GENERATED_PROMPTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{version}.txt"
    )
    image_generation_prompt = load_prompt(generated_image_prompt_path)

    # Load reference image
    ref_image_name = product_by_id(DYNAMIC_CONFIG.SELECTED_PRODUCT_ID)["ref_image"]
    reference_image_path = (
        DYNAMIC_CONFIG.REF_IMAGE_DIR / ref_image_name
    )
    ref_image = load_and_encode_image_b64(reference_image_path)

    # Load generated image
    generated_image_path = (
        DYNAMIC_CONFIG.GENERATED_IMAGES_DIR /
        DYNAMIC_CONFIG.GENERATED_IMAGE_FILE.format(
            CUSTOMER=DYNAMIC_CONFIG.CUSTOMER,
            SELECTED_PRODUCT_ID=DYNAMIC_CONFIG.SELECTED_PRODUCT_ID,
            SELECTED_LEVEL_ID=DYNAMIC_CONFIG.SELECTED_LEVEL_ID,
            version=version,
        )
    )
    generated_image = load_and_encode_image_b64(generated_image_path)

    # Generate judgement
    output_text, reasoning, _resp_dict = ask_judge_model(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_generation_prompt=image_generation_prompt,
        reference_image=ref_image,
        generated_image=generated_image,
        DYNAMIC_CONFIG=DYNAMIC_CONFIG
    )

    # Save judgement as JSON
    judgement_path = (
        DYNAMIC_CONFIG.JUDGEMENTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{version}_judgement.json"
    )
    try:
        judgement_json = _parse_model_json(output_text)
        judgement_json = _normalize_judgement(judgement_json)
    except Exception as e:
        raw_judgement_path = judgement_path.with_suffix(".raw.txt")
        with open(raw_judgement_path, "w", encoding="utf-8") as f:
            f.write(output_text)
        raise ValueError(f"Failed to parse judgement output as JSON: {e}")
    with open(judgement_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(judgement_json, indent=4, ensure_ascii=False))

    # Save reasoning
    judgement_reasoning_path = (
        DYNAMIC_CONFIG.JUDGEMENTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{version}_reasoning.txt"
    )
    with open(judgement_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save OpenAI full response
    resp_file_path = (
        DYNAMIC_CONFIG.JUDGEMENTS_DIR /
        f"{DYNAMIC_CONFIG.CUSTOMER}_P{DYNAMIC_CONFIG.SELECTED_PRODUCT_ID}_L{DYNAMIC_CONFIG.SELECTED_LEVEL_ID}_v{version}_prompt_resp.json"
    )
    with open(resp_file_path, "w", encoding="utf-8") as f:
        f.write(_dump_json_safe(_resp_dict, indent=4))


def main(DYNAMIC_CONFIG) -> None:
    start = time.time()

    logger.info("Cleaning data")
    t0 = time.time()
    clean_data(DYNAMIC_CONFIG=DYNAMIC_CONFIG,
               df=PARTICIPANT_DATA_DF)
    logger.info(f"Data cleaning finished in {time.time() - t0:.2f} seconds")

    logger.info("Building persona description")
    t1 = time.time()
    build_persona_description(DYNAMIC_CONFIG=DYNAMIC_CONFIG)
    logger.info(f"Persona build finished in {time.time() - t1:.2f} seconds")

    # Iterative loop: Generate, judge, check pass; until version limit reached or pass criteria met
    VERSION_PASSED = False
    logger.info(
        f"[INFO] Starting iterative loop with versions "
        f"v{DYNAMIC_CONFIG.FIRST_VERSION}-"
        f"v{DYNAMIC_CONFIG.FIRST_VERSION + DYNAMIC_CONFIG.VERSION_LIMIT - 1}"
    )
    for current_version in DYNAMIC_CONFIG.VERSION_IDS:

        logger.info(f"Starting iteration: v{current_version}")

        logger.info(f"[STAGE] 1.{current_version}: Generating prompt v{current_version}")
        t2 = time.time()
        generate_prompt(current_version, DYNAMIC_CONFIG=DYNAMIC_CONFIG)
        logger.info(f"Prompt generation step finished in {time.time() - t2:.2f} seconds")

        logger.info(f"[STAGE] 2.{current_version}: Generating image v{current_version}")
        t3 = time.time()
        generate_image(version=current_version, DYNAMIC_CONFIG=DYNAMIC_CONFIG)
        logger.info(f"Image generation step finished in {time.time() - t3:.2f} seconds")

        logger.info(f"[STAGE] 3.{current_version}: Judging image v{current_version}")
        t4 = time.time()
        judge(version=current_version, DYNAMIC_CONFIG=DYNAMIC_CONFIG)
        logger.info(f"judgement step finished for v{current_version} in {time.time() - t4:.2f} seconds")

        VERSION_PASSED = check_pass(version=current_version, DYNAMIC_CONFIG=DYNAMIC_CONFIG)
        if VERSION_PASSED:
            logger.info(f"Pass criteria met at version {current_version}. Stopping iterations.")
            break

    if not VERSION_PASSED:
        logger.info("[STAGE] 4: No version passed within selection limit. Selecting best version")
        compare_versions(DYNAMIC_CONFIG=DYNAMIC_CONFIG)
        best_version, _ = get_best_version(DYNAMIC_CONFIG=DYNAMIC_CONFIG)
        logger.info(f"Best version selected: v{best_version}.")
        logger.info(f"Pipeline finished in {time.time() - start:.2f} seconds")


def _create_output_dirs(DYNAMIC_CONFIG) -> None:
    DYNAMIC_CONFIG.GENERATED_PROMPTS_DIR.mkdir(parents=True, exist_ok=True)
    DYNAMIC_CONFIG.GENERATED_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    DYNAMIC_CONFIG.JUDGEMENTS_DIR.mkdir(parents=True, exist_ok=True)
    DYNAMIC_CONFIG.PROMPT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the ad generation pipeline.")
    parser.add_argument(
        "--provider",
        choices=["openai", "google"],
        default=DYN_CONF.DEFAULT_PROVIDER,
        help="Model provider to use. 'openai' uses GPT + gpt-image-1; 'google' uses Gemini + Nano Banana.",
    )
    parser.add_argument(
        "--badge-size",
        type=int,
        default=2,
        help="Number of participants to evaluate.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    DYN_CONF.set_provider(args.provider)

    # Number of participants to evaluate:
    badge_size = args.badge_size

    counter = 0
    logger.info(
        f"Starting pipeline with provider '{DYN_CONF.PROVIDER}' "
        f"and data directory {DYN_CONF.DATA_DIR}"
    )

    # Iterate over customers in PARTICIPANT_DATA_DF
    for i, participant_row in PARTICIPANT_DATA_DF.iterrows():

        if counter >= badge_size:
            logger.info(f"Reached badge size limit of {badge_size}. Stopping further processing.")
            break

        customer_id = participant_row["PID"]
        logger.info(f"Starting pipeline for customer: {customer_id}")

        # Set customer in DYNAMIC_CONFIG
        DYN_CONF.set_customer(customer_id)

        # Check if DYNAMIC_CONFIG.CUSTOMER_DIR exists
        if DYN_CONF.CUSTOMER_DIR.exists():
            logger.info(f"Customer directory {DYN_CONF.CUSTOMER_DIR} already exists. Skipping customer.")
            continue

        else:
            counter += 1
            logger.info(f"Processing customer {customer_id} ({counter}/{badge_size})")

        # Create output directories
        _create_output_dirs(DYNAMIC_CONFIG=DYN_CONF)

        # Run main pipeline for each product
        for product_id in range(1, 5):
            DYN_CONF.set_product(product_id)
            logger.info(f"Processing product ID: {product_id}")
            main(DYNAMIC_CONFIG=DYN_CONF)
