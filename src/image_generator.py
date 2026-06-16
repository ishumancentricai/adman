# src/image_gen.py

"""
- Discover prompt files for L0 (no PID) and L1–L3 (PID-suffixed)
- Respect selected products and levels
- Prepend T2I system prompt to every prompt
- Generate images if prompt files exist
"""

import os
import base64
from dotenv import load_dotenv
from openai import OpenAI

from src.dynamic_config import DynamicConfig
from utils import load_prompt, product_by_id


def ask_openai(
    prompt: str,
    ref_image_path,
    config: DynamicConfig | None = None,
) -> bytes:
    config = config or DynamicConfig()
    load_dotenv()
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    with open(ref_image_path, "rb") as ref_image:
        result = client.images.edit(
            model=config.IMAGE_MODEL,
            image=[ref_image],
            prompt=prompt,
            quality=config.IMAGE_QUALITY,
            input_fidelity=config.INPUT_FIDELITY,
            size=config.IMAGE_SIZE,
            background=config.IMAGE_BACKGROUND,
            n=config.IMAGE_COUNT,
        )

    image_base64 = result.data[0].b64_json
    return base64.b64decode(image_base64)


def generate_image(
    version,
    config: DynamicConfig | None = None,
) -> None:
    config = config or DynamicConfig()

    # Load system prompt template
    system_prompt_path = config.PROMPT_TEMPLATES_DIR / config.IMAGE_GENERATOR_SYS_PROMPT_FILE
    system_prompt = load_prompt(system_prompt_path)
    system_prompt = system_prompt.format(
        PRODUCT_TYPE=product_by_id(config.SELECTED_PRODUCT_ID, config.PRODUCTS)["type"],
        PRODUCT_NAME=product_by_id(config.SELECTED_PRODUCT_ID, config.PRODUCTS)["name"],
    )
    # Load generated image description
    generated_image_prompt_path = (
        config.GENERATED_PROMPTS_DIR /
        f"{config.CUSTOMER}_P{config.SELECTED_PRODUCT_ID}_L{config.SELECTED_LEVEL_ID}_v{version}.txt"
    )
    generated_image_prompt = load_prompt(generated_image_prompt_path)

    # Combine system prompt with generated prompt
    full_prompt = f"{system_prompt}\n{generated_image_prompt}"

    # Log full prompt
    image_generator_log_path = (
        config.PROMPT_LOG_DIR /
        config.IMAGE_PROMPT_LOG.format(
            CUSTOMER=config.CUSTOMER,
            SELECTED_PRODUCT_ID=config.SELECTED_PRODUCT_ID,
            SELECTED_LEVEL_ID=config.SELECTED_LEVEL_ID,
            version=version
        )
    )
    with open(image_generator_log_path, "w", encoding="utf-8") as f:
        f.write(full_prompt.strip())

    # Load reference image
    ref_image_name = product_by_id(config.SELECTED_PRODUCT_ID, config.PRODUCTS)["ref_image"]
    reference_image_path = (
        config.REF_IMAGE_DIR / ref_image_name
    )

    # Generate image
    image_bytes = ask_openai(
        prompt=full_prompt,
        ref_image_path=reference_image_path,
        config=config,
    )

    # Save generated image
    generated_image_path = (
        config.GENERATED_IMAGES_DIR /
        config.GENERATED_IMAGE_FILE.format(
            CUSTOMER=config.CUSTOMER,
            SELECTED_PRODUCT_ID=config.SELECTED_PRODUCT_ID,
            SELECTED_LEVEL_ID=config.SELECTED_LEVEL_ID,
            version=version
        )
    )
    with open(generated_image_path, "wb") as wf:
        wf.write(image_bytes)


if __name__ == "__main__":
    generate_image(1)
