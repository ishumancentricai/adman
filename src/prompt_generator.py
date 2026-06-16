# src/prompt_gen.py

"""
- Skips rows/files with missing/blank PID to avoid *_nan.txt overwrites
- Saves prompts + reasoning into generated_prompts/ with product_id, level, and (for L1-L3) PID at the end
"""
import json
import os
from dotenv import load_dotenv
from openai import OpenAI

from src.dynamic_config import DynamicConfig
from utils import load_prompt, product_by_id, get_openai_resp


def ask_openai(
    system_prompt,
    user_prompt,
    config: DynamicConfig | None = None,
) -> tuple[str, str, dict]:
    config = config or DynamicConfig()
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    req = {
        "model": config.OPENAI_MODEL,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "reasoning": {
            "effort": config.OPENAI_EFFORT,
            "summary": config.OPENAI_SUMMARY,
        },
        "text": {"verbosity": config.OPENAI_VERBOSITY}
    }
    return get_openai_resp(client, req)


def generate_prompt(
    version: int,
    config: DynamicConfig | None = None,
) -> None:
    config = config or DynamicConfig()

    # Sanity check
    if not config.SELECTED_LEVEL_ID == 3:
        raise ValueError(f"Only level 3 prompts are valid, but level {config.SELECTED_LEVEL_ID} was selected.")

    # === Generator input paths ===
    # Load system prompt template
    system_prompt_path = config.PROMPT_TEMPLATES_DIR / config.PROMPT_GENERATOR_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    # Load user prompt template
    user_prompt_path = config.PROMPT_TEMPLATES_DIR / config.PROMPT_GENERATOR_USER_PROMPT_FILE
    user_prompt_template = load_prompt(user_prompt_path)
    # Load persona description
    persona_description_path = config.CUSTOMER_DIR / config.PERSONA_DESCRIPTION_FILE
    persona_description = load_prompt(persona_description_path)

    # Load product info
    product_info = product_by_id(config.SELECTED_PRODUCT_ID, config.PRODUCTS)

    user_prompt = user_prompt_template.format(
        PRODUCT_NAME=product_info["name"],
        PRODUCT_TYPE=product_info["type"],
        persona_description=persona_description,
    )

    # Log full prompt
    log_prompt = f"{system_prompt}\n\n{user_prompt}"
    prompt_generator_log_path = config.PROMPT_LOG_DIR / config.PROMPT_GENERATOR_LOG
    with open(prompt_generator_log_path, "w", encoding="utf-8") as f:
        f.write(log_prompt.strip())

    output_text, reasoning, _resp_dict = ask_openai(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        config=config,
    )

    # Save generated prompt
    generated_prompt_path = (
        config.GENERATED_PROMPTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_v{version}.txt"
    )
    with open(generated_prompt_path, "w", encoding="utf-8") as f:
        f.write(output_text.strip())
    # Save reasoning
    generated_prompt_reasoning_path = (
        config.GENERATED_PROMPTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_v{version}_reasoning.txt"
    )
    with open(generated_prompt_reasoning_path, "w", encoding="utf-8") as f:
        f.write(reasoning.strip())
    # Save OpenAI full response
    resp_file_path = (
        config.GENERATED_PROMPTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_v{version}_prompt_resp.json"
    )
    with open(resp_file_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(_resp_dict, indent=4))


if __name__ == "__main__":
    generate_prompt(1)
