import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

from src.dynamic_config import DynamicConfig
from utils import load_prompt, product_by_id, load_and_encode_image_b64, get_openai_resp

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
    image_generation_prompt: str,
    reference_image: bytes,
    generated_image: bytes,
    config: DynamicConfig | None = None,
) -> tuple[Optional[str], str, dict]:
    config = config or DynamicConfig()
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    req = {
        "model": config.OPENAI_MODEL,
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
        "reasoning": {
            "effort": config.OPENAI_EFFORT,
            "summary": config.OPENAI_SUMMARY,
        },
        "text": {"verbosity": config.OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def judge(
    version: int,
    config: DynamicConfig | None = None,
) -> None:
    config = config or DynamicConfig()

    # Load system prompt template
    system_prompt_path = config.PROMPT_TEMPLATES_DIR / config.JUDGE_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = config.PROMPT_TEMPLATES_DIR / config.JUDGE_USER_PROMPT_FILE
    user_prompt = load_prompt(user_prompt_path)

    # Load image generation prompt
    generated_image_prompt_path = (
        config.GENERATED_PROMPTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_v{version}.txt"
    )
    image_generation_prompt = load_prompt(generated_image_prompt_path)

    # Load reference image
    ref_image_name = product_by_id(config.SELECTED_PRODUCT_ID, config.PRODUCTS)["ref_image"]
    reference_image_path = (
        config.REF_IMAGE_DIR / ref_image_name
    )
    ref_image = load_and_encode_image_b64(reference_image_path)

    # Load generated image
    generated_image_path = (
        config.GENERATED_IMAGES_DIR /
        config.GENERATED_IMAGE_FILE.format(
            CUSTOMER=config.CUSTOMER,
            SELECTED_PRODUCT_ID=config.SELECTED_PRODUCT_ID,
            SELECTED_LEVEL_ID=config.SELECTED_LEVEL_ID,
            version=version,
        )
    )
    generated_image = load_and_encode_image_b64(generated_image_path)

    # Generate judgement
    output_text, reasoning, _resp_dict = ask_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        image_generation_prompt=image_generation_prompt,
        reference_image=ref_image,
        generated_image=generated_image,
        config=config,
    )

    # Save judgement as JSON
    judgement_path = (
        config.JUDGEMENTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_v{version}_judgement.json"
    )
    try:
        judgement_json = json.loads(output_text)
        if not isinstance(judgement_json, dict):
            raise ValueError("judgement output is not a dictionary.")
        judgement_json = _normalize_judgement(judgement_json)
    except Exception as e:
        raise ValueError(f"Failed to parse judgement output as JSON: {e}")
    with open(judgement_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(judgement_json, indent=4, ensure_ascii=False))

    # Save reasoning
    judgement_reasoning_path = (
        config.JUDGEMENTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_v{version}_reasoning.txt"
    )
    with open(judgement_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save OpenAI full response
    resp_file_path = (
        config.JUDGEMENTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_v{version}_prompt_resp.json"
    )
    with open(resp_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_resp_dict, indent=4))


if __name__ == "__main__":
    judge(1)
