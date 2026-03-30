import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    CUSTOMER,
    VERSION_LIMIT,
    SELECTED_LEVEL_ID,
    SELECTED_PRODUCT_ID,
    OPENAI_MODEL,
    OPENAI_EFFORT,
    OPENAI_VERBOSITY,
    OPENAI_SUMMARY,
    PROMPT_TEMPLATES_DIR,
    REF_IMAGE_DIR,
    JUDGE_SELECTOR_SYS_PROMPT_FILE,
    JUDGE_SELECTOR_USER_PROMPT_FILE,
    GENERATED_IMAGE_FILE,
    GENERATED_IMAGES_DIR,
    JUDGEMENTS_DIR,
)
from utils import load_prompt, product_by_id, load_and_encode_image_b64, get_openai_resp


def _normalize_decision(data: dict) -> dict:
    """Normalize decision output data."""
    if not data:
        raise ValueError("Empty decision data")

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

    if not "best_version" in data.keys():
        raise ValueError("Decision data missing 'best_version' field.")
    if not "explanation" in data.keys():
        raise ValueError("Decision data missing 'explanation' field.")

    for key, value in data.items():
        norm[key] = _strip_sec(value)

    return norm


def check_pass(version: int) -> bool:
    judgement_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}_judgement.json"
    )
    # Double JSON might be unnecessary - keep for consistent formatting
    with open(judgement_path, "r", encoding="utf-8") as f:
        judgement = json.load(f)
    if not isinstance(judgement, dict):
        raise ValueError(f"Invalid judgement format in {judgement_path}")

    # Check if the judgement passed
    return judgement.get("judgement") == "pass"


def ask_openai(
    system_prompt: str,
    user_prompt: str,
    reference_image: bytes,
    generated_images: list[bytes],
) -> tuple[str, str, dict]:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    image_content = []
    for i in range(VERSION_LIMIT):
        image_content.append({"type": "input_text", "text": f"Version {i}:"})
        image_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{generated_images[i]}"})

    req = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{reference_image}"},
                ] + image_content,
             },
        ],
        "reasoning": {"effort": OPENAI_EFFORT, "summary": OPENAI_SUMMARY},
        "text": {"verbosity": OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def judge_and_select() -> int:
    # Load system prompt template
    system_prompt_path = PROMPT_TEMPLATES_DIR / JUDGE_SELECTOR_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = PROMPT_TEMPLATES_DIR / JUDGE_SELECTOR_USER_PROMPT_FILE
    user_prompt = load_prompt(user_prompt_path)
    user_prompt = user_prompt.format(
        VERSION_LIMIT=VERSION_LIMIT,
        PRODUCT_TYPE=product_by_id(SELECTED_PRODUCT_ID)["type"],
        PRODUCT_NAME=product_by_id(SELECTED_PRODUCT_ID)["name"],
    )

    # Load generated images
    generated_images = []
    for v in range(VERSION_LIMIT):
        generated_image_path = (
            GENERATED_IMAGES_DIR /
            GENERATED_IMAGE_FILE.format(
                CUSTOMER=CUSTOMER,
                SELECTED_PRODUCT_ID=SELECTED_PRODUCT_ID,
                SELECTED_LEVEL_ID=SELECTED_LEVEL_ID,
                version=v,
            )
        )
        generated_image = load_and_encode_image_b64(generated_image_path)
        generated_images.append(generated_image)

    # Load reference image
    ref_image_name = product_by_id(SELECTED_PRODUCT_ID)["ref_image"]
    reference_image_path = (
        REF_IMAGE_DIR / ref_image_name
    )
    ref_image = load_and_encode_image_b64(reference_image_path)

    # Generate judgement
    output_text, reasoning, _resp_dict = ask_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reference_image=ref_image,
        generated_images=generated_images,
    )

    # Save decision as JSON
    decision_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_decision.json"
    )
    try:
        decision_json = json.loads(output_text)
        if not isinstance(decision_json, dict):
            raise ValueError("Decision output is not a dictionary.")
        decision_json = _normalize_decision(decision_json)
    except Exception as e:
        raise ValueError(f"Failed to parse decision output as JSON: {e}")
    with open(decision_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(decision_json, indent=4, ensure_ascii=False))
    best_version = decision_json["best_version"]
    explanation = decision_json["explanation"]

    # Save reasoning
    decision_reasoning_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_decision_reasoning.txt"
    )
    with open(decision_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save OpenAI full response
    resp_file_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_decision_prompt_resp.json"
    )
    with open(resp_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_resp_dict, indent=4))

    return best_version, explanation


if __name__ == "__main__":
    judge_and_select()
