# src/prompt_gen.py

"""
- Skips rows/files with missing/blank PID to avoid *_nan.txt overwrites
- Saves prompts + reasoning into generated_prompts/ with product_id, level, and (for L1-L3) PID at the end
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    CUSTOMER,
    SELECTED_LEVEL_ID,
    SELECTED_PRODUCT_ID,
    OPENAI_MODEL,
    OPENAI_EFFORT,
    OPENAI_VERBOSITY,
    OPENAI_SUMMARY,
    PROMPT_TEMPLATES_DIR,
    CUSTOMER_DIR,
    PROMPT_GENERATOR_SYS_PROMPT_FILE,
    PROMPT_GENERATOR_USER_PROMPT_FILE,
    PERSONA_DESCRIPTION_FILE,
    GENERATED_PROMPTS_DIR,
    PROMPT_LOG_DIR,
    PROMPT_GENERATOR_LOG,
)
from utils import load_prompt, product_by_id, get_openai_resp


def ask_openai(system_prompt, user_prompt) -> tuple[str, str, dict]:
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    req = {
        "model": OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "reasoning": {"effort": OPENAI_EFFORT, "summary": OPENAI_SUMMARY},
        "text": {"verbosity": OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def generate_prompt(version: int) -> None:
    # Sanity check
    if not SELECTED_LEVEL_ID == 3:
        raise ValueError(f"Only level 3 prompts are valid, but level {SELECTED_LEVEL_ID} was selected.")

    # === Generator input paths ===
    # Load system prompt template
    system_prompt_path = PROMPT_TEMPLATES_DIR / PROMPT_GENERATOR_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = PROMPT_TEMPLATES_DIR / PROMPT_GENERATOR_USER_PROMPT_FILE
    user_prompt_template = load_prompt(user_prompt_path)
    # Load persona description
    persona_description_path = CUSTOMER_DIR / PERSONA_DESCRIPTION_FILE
    persona_description = load_prompt(persona_description_path)

    # Load product info
    product_info = product_by_id(SELECTED_PRODUCT_ID)

    user_prompt = user_prompt_template.format(
        PRODUCT_NAME=product_info["name"],
        PRODUCT_TYPE=product_info["type"],
        persona_description=persona_description,
    )

    # Log full prompt
    log_prompt = f"{system_prompt}\n\n{user_prompt}"
    prompt_generator_log_path = PROMPT_LOG_DIR / PROMPT_GENERATOR_LOG
    with open(prompt_generator_log_path, "w", encoding="utf-8") as f:
        f.write(log_prompt.strip())

    output_text, reasoning, _resp_dict = ask_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    # Save generated prompt
    generated_prompt_path = (
        GENERATED_PROMPTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}.txt"
    )
    with open(generated_prompt_path, "w", encoding="utf-8") as f:
        f.write(output_text.strip())
    # Save reasoning
    generated_prompt_reasoning_path = (
        GENERATED_PROMPTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}_reasoning.txt"
    )
    with open(generated_prompt_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save OpenAI full response
    resp_file_path = (
        GENERATED_PROMPTS_DIR /
        f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}_prompt_resp.json"
    )
    with open(resp_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_resp_dict, indent=4))


if __name__ == "__main__":
    generate_prompt(0)
