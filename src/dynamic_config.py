"""Dynamic configuration helper.

This module exposes :class:`DynamicConfig`, the canonical configuration object.
Callers can swap the active customer, product, or level at runtime. Usage
example:

>>> cfg = DynamicConfig(customer="lee", product_id=2)
>>> cfg.CUSTOMER
'lee'
>>> cfg.CUSTOMER_DIR
PosixPath('.../data/customers/lee')

Every attribute is available through the instance (``cfg.VARIABLE_NAME``), and
calling :meth:`update`, :meth:`set_customer`, :meth:`set_product`, or
:meth:`set_level` recomputes all dependent values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional


Provider = Literal["openai", "google"]


class DynamicConfig:
    """Mutable configuration state derived from the static defaults."""

    # --- Defaults for dynamic selectors ---
    DEFAULT_CUSTOMER = "lee"
    DEFAULT_LEVEL_ID = 3
    DEFAULT_PRODUCT_ID = 4
    DEFAULT_PROVIDER: Provider = "google"

    # --- Shared settings ---
    PID_COLUMN = "PID"
    VERSION_LIMIT = 3
    FIRST_VERSION = 1

    OPENAI_MODEL = "gpt-5"
    OPENAI_EFFORT = "high"
    OPENAI_VERBOSITY = "high"
    OPENAI_SUMMARY = "auto"
    USE_LLM_FOR_L0 = False

    IMAGE_MODEL = "gpt-image-1"
    IMAGE_QUALITY = "high"
    INPUT_FIDELITY = "high"
    PRODUCT_DEP_IMAGE_SIZE = False
    IMAGE_SIZE = "1024x1536"
    IMAGE_BACKGROUND = "auto"
    IMAGE_MODERATION = "auto"
    IMAGE_FORMAT = "png"
    IMAGE_COMPRESSION = 100
    IMAGE_COUNT = 1

    GOOGLE_TEXT_MODEL = "gemini-3.1-pro-preview" # "gemini-3.5-flash"
    GOOGLE_IMAGE_MODEL = "gemini-3-pro-image-preview"
    GOOGLE_IMAGE_ASPECT_RATIO = "2:3"
    GOOGLE_IMAGE_SIZE = "1K"  # "512", "1K", "2K", "4K"
    GOOGLE_THINKING_ENABLED = True
    GOOGLE_THINKING_EFFORT = "low"  # "minimal", "low", "medium", "high"
    GOOGLE_TIMEOUT_MS = 400_000
    GOOGLE_RETRY_ATTEMPTS = 10
    GOOGLE_RETRY_INITIAL_DELAY_SECONDS = 15
    GOOGLE_RETRY_MAX_DELAY_SECONDS = 520

    MARKETING_OBJECTIVE = 'increase "purchase intention"'

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

    BASE_PATH = Path(__file__).resolve().parent.parent
    DATA_ROOT_DIR = BASE_PATH / "data"
    DATA_DIR = DATA_ROOT_DIR / DEFAULT_PROVIDER
    PARTICIPANT_DATA_FILE = DATA_ROOT_DIR / "participants.csv"
    PROMPT_TEMPLATES_DIR = BASE_PATH / "prompt_templates"
    REF_IMAGE_DIR = BASE_PATH / "reference_images"

    PROMPT_GENERATOR_SYS_PROMPT_FILE = (
        "prompt_generator_system_prompt_revised.txt"
    )
    PROMPT_GENERATOR_USER_PROMPT_FILE = (
        "prompt_generator_user_prompt_revised.txt"
    )

    JUDGE_SYS_PROMPT_FILE = "judge_system_prompt.txt"
    JUDGE_USER_PROMPT_FILE = "judge_user_prompt.txt"

    SELECTOR_SYS_PROMPT_FILE = "selector_system_prompt.txt"
    SELECTOR_USER_PROMPT_FILE = "selector_user_prompt.txt"

    IMAGE_GENERATOR_SYS_PROMPT_FILE = (
        "image_generator_system_prompt_revised.txt"
    )

    JUDGE_SELECTOR_SYS_PROMPT_FILE = "judge_selector_system_prompt.txt"
    JUDGE_SELECTOR_USER_PROMPT_FILE = "judge_selector_user_prompt.txt"

    def __init__(
        self,
        *,
        customer: Optional[str] = None,
        product_id: Optional[int] = None,
        level_id: Optional[int] = None,
        provider: Provider | str | None = None,
    ) -> None:
        self._customer = customer or self.DEFAULT_CUSTOMER
        self._product_id = product_id or self.DEFAULT_PRODUCT_ID
        self._level_id = level_id or self.DEFAULT_LEVEL_ID
        self._provider = self._normalize_provider(provider or self.DEFAULT_PROVIDER)
        self._sync_dynamic_fields()

    # --- Public mutators -------------------------------------------------
    def set_customer(self, customer: str) -> None:
        if not customer:
            raise ValueError("customer must be a non-empty string")
        self._customer = customer
        self._sync_dynamic_fields()

    def set_product(self, product_id: int) -> None:
        self._validate_product(product_id)
        self._product_id = product_id
        self._sync_dynamic_fields()

    def set_level(self, level_id: int) -> None:
        if level_id is None:
            raise ValueError("level_id must not be None")
        self._level_id = level_id
        self._sync_dynamic_fields()

    def set_provider(self, provider: Provider | str) -> None:
        self._provider = self._normalize_provider(provider)
        self._sync_dynamic_fields()

    def update(
        self,
        *,
        customer: Optional[str] = None,
        product_id: Optional[int] = None,
        level_id: Optional[int] = None,
        provider: Provider | str | None = None,
    ) -> None:
        if customer is not None:
            if not customer:
                raise ValueError("customer must be a non-empty string")
            self._customer = customer
        if product_id is not None:
            self._validate_product(product_id)
            self._product_id = product_id
        if level_id is not None:
            self._level_id = level_id
        if provider is not None:
            self._provider = self._normalize_provider(provider)
        self._sync_dynamic_fields()

    # --- Helpers ---------------------------------------------------------
    def _validate_product(self, product_id: int) -> None:
        if product_id is None:
            raise ValueError("product_id must not be None")
        if not any(
            info.get("id") == product_id for info in self.PRODUCTS.values()
        ):
            raise ValueError(f"Unknown product_id: {product_id}")

    def _normalize_provider(self, provider: Provider | str) -> Provider:
        provider = provider.lower().strip()
        if provider not in {"openai", "google"}:
            raise ValueError("provider must be either 'openai' or 'google'")
        return provider  # type: ignore[return-value]

    def _sync_dynamic_fields(self) -> None:
        # Primary selections
        self.PROVIDER = self._provider
        self.CUSTOMER = self._customer
        self.SELECTED_PRODUCT_ID = self._product_id
        self.SELECTED_LEVEL_ID = self._level_id
        self.VERSION_IDS = range(
            self.FIRST_VERSION,
            self.FIRST_VERSION + self.VERSION_LIMIT,
        )

        # Paths derived from the active customer
        self.DATA_DIR = self.DATA_ROOT_DIR / self.PROVIDER
        self.CUSTOMER_DIR = self.DATA_DIR / "customers" / self.CUSTOMER
        self.CUSTOMER_DATA_FILE = f"{self.CUSTOMER}.csv"
        self.CLEANED_CUSTOMER_DATA_FILE = f"{self.CUSTOMER}_cleaned.csv"
        self.PERSONA_DESCRIPTION_FILE = (
            f"{self.CUSTOMER}_L{self.SELECTED_LEVEL_ID}.csv"
        )

        # Customer scoped output directories
        self.GENERATED_PROMPTS_DIR = self.CUSTOMER_DIR / "generated_prompts"
        self.GENERATED_IMAGES_DIR = self.CUSTOMER_DIR / "generated_images"
        self.JUDGEMENTS_DIR = self.CUSTOMER_DIR / "judgements"
        self.PROMPT_LOG_DIR = self.CUSTOMER_DIR / "prompt_logs"

        # File naming templates reflecting the active selection
        self.GENERATED_IMAGE_FILE = (
            f"image_{self.CUSTOMER}_P{self.SELECTED_PRODUCT_ID}_L"
            f"{self.SELECTED_LEVEL_ID}_v{{version}}.png"
        )
        self.PROMPT_GENERATOR_LOG = (
            f"{self.CUSTOMER}_P{self.SELECTED_PRODUCT_ID}_L"
            f"{self.SELECTED_LEVEL_ID}_prompt_generator_input.txt"
        )
        self.IMAGE_PROMPT_LOG = (
            f"{self.CUSTOMER}_P{self.SELECTED_PRODUCT_ID}_L"
            f"{self.SELECTED_LEVEL_ID}_v{{version}}_image_generator_input.txt"
        )


__all__ = ["DynamicConfig"]
