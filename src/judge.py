import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional

from config import (
    CUSTOMER,
    SELECTED_LEVEL_ID,
    SELECTED_PRODUCT_ID,
    OPENAI_MODEL,
    OPENAI_EFFORT,
    OPENAI_VERBOSITY,
    OPENAI_SUMMARY,
    PROMPT_TEMPLATES_DIR,
    REF_IMAGE_DIR,
    JUDGE_SYS_PROMPT_FILE,
    JUDGE_USER_PROMPT_FILE,
    GENERATED_IMAGE_FILE,
    GENERATED_PROMPTS_DIR,
    GENERATED_IMAGES_DIR,
    JUDGEMENTS_DIR,
)
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
    image_generation_prompt: str,
    reference_image: bytes,
    generated_image: bytes,
) -> tuple[Optional[str], str, dict]:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    req = {
        "model": OPENAI_MODEL,
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
        "reasoning": {"effort": OPENAI_EFFORT, "summary": OPENAI_SUMMARY},
        "text": {"verbosity": OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def judge(version: int) -> None:
    # Load system prompt template
    system_prompt_path = PROMPT_TEMPLATES_DIR / JUDGE_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = PROMPT_TEMPLATES_DIR / JUDGE_USER_PROMPT_FILE
    user_prompt = load_prompt(user_prompt_path)

    # Load image generation prompt
    generated_image_prompt_path = (
        GENERATED_PROMPTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}.txt"
    )
    image_generation_prompt = load_prompt(generated_image_prompt_path)

    # Load reference image
    ref_image_name = product_by_id(SELECTED_PRODUCT_ID)["ref_image"]
    reference_image_path = (
        REF_IMAGE_DIR / ref_image_name
    )
    ref_image = load_and_encode_image_b64(reference_image_path)

    # Load generated image
    generated_image_path = (
        GENERATED_IMAGES_DIR /
        GENERATED_IMAGE_FILE.format(
            CUSTOMER=CUSTOMER,
            SELECTED_PRODUCT_ID=SELECTED_PRODUCT_ID,
            SELECTED_LEVEL_ID=SELECTED_LEVEL_ID,
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
    )

    # Save judgement as JSON
    judgement_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}_judgement.json"
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
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}_reasoning.txt"
    )
    with open(judgement_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save OpenAI full response
    resp_file_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}_prompt_resp.json"
    )
    with open(resp_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_resp_dict, indent=4))


if __name__ == "__main__":
    judge(0)
