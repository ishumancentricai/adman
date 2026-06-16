import logging
import sys
import base64
from pathlib import Path

from src.dynamic_config import DynamicConfig

# === LOGGER ===
logger = logging.getLogger('app')
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '%(asctime)s | %(levelname)s | %(filename)s | %(message)s',
    datefmt='%H:%M:%S'
)
handler.setFormatter(formatter)
logger.addHandler(handler)


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    return txt


def product_by_id(prod_id: int, products: dict | None = None) -> dict:
    products = products or DynamicConfig.PRODUCTS
    for product in products.values():
        if product["id"] == prod_id:
            return product
    raise ValueError(f"Product ID {prod_id} not found in products.")


def load_and_encode_image_b64(image_path: str) -> str:
    """Encode an image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def get_openai_resp(client, req) -> tuple[str, str, dict]:
    """Send request to OpenAI and return output text, reasoning text, and full response."""
    try:
        resp = client.responses.create(**req)
    except Exception as e:
        raise ValueError(f"OpenAI API request failed: {e}")

    output_text = getattr(resp, "output_text", None)
    if not output_text:
        raise ValueError("OpenAI API response missing output_text.")
    output_text = output_text.strip()
    
    resp_dict = resp.model_dump()

    reasoning_text = [
        s.get("text", "")
        for s in resp_dict.get("output", [{}])[0].get("summary", [])
    ]
    reasoning_text = "\n".join(reasoning_text).strip()
    
    return output_text, reasoning_text, resp_dict
