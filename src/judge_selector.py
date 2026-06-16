import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from src.dynamic_config import DynamicConfig
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


def check_pass(
    version: int,
    config: DynamicConfig | None = None,
) -> bool:
    config = config or DynamicConfig()
    judgement_path = (
        config.JUDGEMENTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_v{version}_judgement.json"
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
    config: DynamicConfig | None = None,
) -> tuple[str, str, dict]:
    config = config or DynamicConfig()
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    image_content = []
    for version, generated_image in zip(config.VERSION_IDS, generated_images):
        image_content.append({"type": "input_text", "text": f"Version {version}:"})
        image_content.append({"type": "input_image", "image_url": f"data:image/png;base64,{generated_image}"})

    req = {
        "model": config.OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user", "content": [
                    {"type": "input_text", "text": user_prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{reference_image}"},
                ] + image_content,
             },
        ],
        "reasoning": {
            "effort": config.OPENAI_EFFORT,
            "summary": config.OPENAI_SUMMARY,
        },
        "text": {"verbosity": config.OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def judge_and_select(config: DynamicConfig | None = None) -> int:
    config = config or DynamicConfig()

    # Load system prompt template
    system_prompt_path = config.PROMPT_TEMPLATES_DIR / config.JUDGE_SELECTOR_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = config.PROMPT_TEMPLATES_DIR / config.JUDGE_SELECTOR_USER_PROMPT_FILE
    user_prompt = load_prompt(user_prompt_path)
    product_info = product_by_id(config.SELECTED_PRODUCT_ID, config.PRODUCTS)
    user_prompt = user_prompt.format(
        VERSION_LIMIT=config.VERSION_LIMIT,
        PRODUCT_TYPE=product_info["type"],
        PRODUCT_NAME=product_info["name"],
    )

    # Load generated images
    generated_images = []
    for v in config.VERSION_IDS:
        generated_image_path = (
            config.GENERATED_IMAGES_DIR /
            config.GENERATED_IMAGE_FILE.format(
                CUSTOMER=config.CUSTOMER,
                SELECTED_PRODUCT_ID=config.SELECTED_PRODUCT_ID,
                SELECTED_LEVEL_ID=config.SELECTED_LEVEL_ID,
                version=v,
            )
        )
        generated_image = load_and_encode_image_b64(generated_image_path)
        generated_images.append(generated_image)

    # Load reference image
    ref_image_name = product_info["ref_image"]
    reference_image_path = (
        config.REF_IMAGE_DIR / ref_image_name
    )
    ref_image = load_and_encode_image_b64(reference_image_path)

    # Generate judgement
    output_text, reasoning, _resp_dict = ask_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        reference_image=ref_image,
        generated_images=generated_images,
        config=config,
    )

    # Save decision as JSON
    decision_path = (
        config.JUDGEMENTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_decision.json"
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
        config.JUDGEMENTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_decision_reasoning.txt"
    )
    with open(decision_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save OpenAI full response
    resp_file_path = (
        config.JUDGEMENTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_decision_prompt_resp.json"
    )
    with open(resp_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_resp_dict, indent=4))

    return best_version, explanation


if __name__ == "__main__":
    judge_and_select()
