import os
import json
from pathlib import Path
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
    JUDGEMENTS_DIR,
    SELECTED_LEVEL_ID,
    SELECTED_PRODUCT_ID,
    SELECTOR_SYS_PROMPT_FILE,
    SELECTOR_USER_PROMPT_FILE,
)
from utils import logger, load_prompt, get_openai_resp


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

    if not "best_version" in data.keys():
        raise ValueError("Comparison data missing 'best_version' field.")
    if not "explanation" in data.keys():
        raise ValueError("Comparison data missing 'explanation' field.")

    for key, value in data.items():
        norm[key] = _strip_sec(value)

    return norm


def ask_openai(
    system_prompt: str,
    user_prompt: str,
    *judgements,
) -> tuple[str, str, dict]:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    req = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "input_text", "text": user_prompt},
                {"type": "input_text", "text": "\n\n".join(f"# judgement {i}\n{v}\n\n" for i, v in enumerate(judgements))},
            ]},
        ],
        "reasoning": {"effort": OPENAI_EFFORT, "summary": OPENAI_SUMMARY},
        "text": {"verbosity": OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def compare_versions():
    # Load system prompt template
    system_prompt_path = PROMPT_TEMPLATES_DIR / SELECTOR_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = PROMPT_TEMPLATES_DIR / SELECTOR_USER_PROMPT_FILE
    user_prompt = load_prompt(user_prompt_path)

    # Load judgements
    judgements = []
    for v in range(VERSION_LIMIT):
        judgement_path = (
            JUDGEMENTS_DIR /
            f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{v}_judgement.json"
        )
        # Double JSON might be unnecessary - keep for consistent formatting
        with open(judgement_path, "r", encoding="utf-8") as f:
            judgement = json.load(f)
        if not isinstance(judgement, dict):
            raise ValueError(f"Invalid judgement format in {judgement_path}")
        judgements.append(json.dumps(judgement))

    output_text, reasoning, _resp_dict = ask_openai(
        system_prompt,
        user_prompt,
        *judgements,
    )

    # Save comparison output
    comparison_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_version_comparison.json"
    )
    try:
        comparison_json = json.loads(output_text)
        if not isinstance(comparison_json, dict):
            raise ValueError("Comparison output is not a dictionary.")
        comparison_json = _normalize_comparison(comparison_json)
    except Exception as e:
        raise ValueError(f"Failed to parse comparison output as JSON: {e}")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(comparison_json, indent=4, ensure_ascii=False))
    # Save reasoning
    comparison_reasoning_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_version_comparison_reasoning.txt"
    )
    with open(comparison_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save full response
    resp_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_version_comparison_response.json"
    )
    with open(resp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_resp_dict, indent=4))


def get_best_version() -> tuple[int, str]:
    comparison_file_path = (
        JUDGEMENTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_version_comparison.json"
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


if __name__ == "__main__":
    compare_versions()
