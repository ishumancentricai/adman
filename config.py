# config.py

"""
- Centralized configuration for file paths, models, and parameters
- Keeps all constants used across pipeline scripts
- Enables easy updates and consistent settings management
"""
from pathlib import Path


# === BATCH SELECTION ===
CUSTOMER = "lee"
SELECTED_LEVEL_ID = 3
# Use None or any combination of [0, 1, 2, 3], e.g. [0]
SELECTED_PRODUCT_ID = 4
# Use None or any combination of [1, 2, 3, 4], e.g. [1, 4]
PID_COLUMN = "PID"
# Version limit for image generation and judging (1, 2, 3, ...)
VERSION_LIMIT = 3


# === LLM MODEL CONFIG ===
OPENAI_MODEL = "gpt-5"  # "gpt-5", "gpt-5-mini", "gpt-5-nano"
OPENAI_EFFORT = "high"  # "minimal", "low", "medium", "high"
OPENAI_VERBOSITY = "high"  # "low", "medium", "high"
OPENAI_SUMMARY = "auto"  # "concise", "detailed", "auto" (for selection please check which model supports which paramter) - "detailed for GPT5"
USE_LLM_FOR_L0 = False  # If True  -> prompt_gen will generate L0 prompts (one per product) and image_gen will use them. # If False -> prompt_gen will skip L0; image_gen will generate L0 images using ONLY the T2I system prompt.

# === IMAGE-GEN SETTINGS ===
IMAGE_MODEL = "gpt-image-1"  # "gpt-image-1", "dall-e-3", "dall-e-2"
IMAGE_QUALITY = "high"  #  "low", "medium", "high" (for gpt-image-1); "hd", "standard" (for dall-e-3); "standard" (for dall-e-2)
INPUT_FIDELITY = "high"  # only image 1 supports low, high
PRODUCT_DEP_IMAGE_SIZE = (
    False  # if True, use PRODUCTS[*]["image_size"]; else fallback to IMAGE_SIZE
)
IMAGE_SIZE = "1024x1536"  # "1024x1024", "1536x1024", "1024x1536", "auto" (gpt-image-1); may differ for other models
IMAGE_BACKGROUND = "auto"  # "transparent", "opaque", "auto" (only for gpt-image-1)
IMAGE_MODERATION = "auto"  # "low", "auto" (only for gpt-image-1)
IMAGE_FORMAT = "png"  # "png", "jpeg", "webp" (only for gpt-image-1)
IMAGE_COMPRESSION = 100  # 0–100 (only for gpt-image-1 with jpeg/webp)
IMAGE_COUNT = 1  # Number of images to generate (1–10 for gpt-image-1 and dall-e-2, only 1 for dall-e-3)

# === MARKETING SETTINGS ===
MARKETING_OBJECTIVE = 'increase "purchase intention"'

# === PRODUCTS ===
PRODUCTS = {
    "ev": {
        "id": 1,
        "type": "electric car",
        "name": "Polestar 4",
        "ref_image": "polestar.jpg",
        "image_size": "1536x1024",
    },
    "luxury_watch": {
        "id": 2,
        "type": "luxury watch",
        "name": "Rolex Submariner Date Oyster",
        "ref_image": "rolex.jpg",
        "image_size": "800x1180",
    },
    "detergent": {
        "id": 3,
        "type": "laundry detergent",
        "name": "Tide Ultra Hygienic Clean",
        "ref_image": "tide.jpg",
        "image_size": "1024x1536",
    },
    "soft_drink": {
        "id": 4,
        "type": "soft drink",
        "name": "Coca-Cola",
        "ref_image": "cola.jpg",
        "image_size": "1250x1250",
    },
}


# === DIRECTORIES ===
BASE_PATH = Path(__file__).parent.resolve()

DATA_DIR = BASE_PATH / "data"
PROMPT_TEMPLATES_DIR = BASE_PATH / "prompt_templates"
REF_IMAGE_DIR = BASE_PATH / "reference_images"
CUSTOMER_DIR = DATA_DIR / "customers" / CUSTOMER

# === DATA FILES ===
CUSTOMER_DATA_FILE = f"{CUSTOMER}.csv"
CLEANED_CUSTOMER_DATA_FILE = f"{CUSTOMER}_cleaned.csv"

# === PROMPT TEMPLATE FILES ===
PROMPT_GENERATOR_SYS_PROMPT_FILE = "prompt_generator_system_prompt_revised.txt"
PROMPT_GENERATOR_USER_PROMPT_FILE = "prompt_generator_user_prompt_revised.txt"

JUDGE_SYS_PROMPT_FILE = "judge_system_prompt.txt"
JUDGE_USER_PROMPT_FILE = "judge_user_prompt.txt"

SELECTOR_SYS_PROMPT_FILE = "selector_system_prompt.txt"
SELECTOR_USER_PROMPT_FILE = "selector_user_prompt.txt"

IMAGE_GENERATOR_SYS_PROMPT_FILE = "image_generator_system_prompt_revised.txt"

JUDGE_SELECTOR_SYS_PROMPT_FILE = "judge_selector_system_prompt.txt"
JUDGE_SELECTOR_USER_PROMPT_FILE = "judge_selector_user_prompt.txt"

# === INPUT FILES ===
PERSONA_DESCRIPTION_FILE = f"{CUSTOMER}_L{SELECTED_LEVEL_ID}.csv"

# === OUTPUT===
GENERATED_PROMPTS_DIR = CUSTOMER_DIR / "generated_prompts"
GENERATED_IMAGES_DIR = CUSTOMER_DIR / "generated_images"
JUDGEMENTS_DIR = CUSTOMER_DIR / "judgements"
PROMPT_LOG_DIR = CUSTOMER_DIR / "prompt_logs"
GENERATED_IMAGE_FILE = "image_{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}.png"
PROMPT_GENERATOR_LOG = f"{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_prompt_generator_input.txt"
IMAGE_PROMPT_LOG = "{CUSTOMER}_P{SELECTED_PRODUCT_ID}_L{SELECTED_LEVEL_ID}_v{version}_image_generator_input.txt"
