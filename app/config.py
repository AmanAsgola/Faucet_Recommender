"""Configuration settings for the Faucet Identification POC."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Set

# Root directory of the repository
BASE_DIR = Path(__file__).resolve().parent.parent

# Try to load .env if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass


@dataclass(frozen=True)
class Settings:
    # Base paths
    base_dir: Path = BASE_DIR
    data_dir: Path = BASE_DIR / "data"
    raw_dir: Path = BASE_DIR / "data" / "raw"
    processed_dir: Path = BASE_DIR / "data" / "processed"
    evaluation_dir: Path = BASE_DIR / "data" / "evaluation"
    fixtures_dir: Path = BASE_DIR / "data" / "fixtures"

    # Catalog & manifest paths
    catalog_csv: Path = Path(os.getenv("FAUCET_CATALOG_CSV", str(BASE_DIR / "data" / "products.csv")))
    image_manifest_csv: Path = Path(os.getenv("FAUCET_IMAGE_MANIFEST_CSV", str(BASE_DIR / "data" / "image_manifest.csv")))
    
    # Processed index artifacts
    index_path: Path = Path(os.getenv("FAUCET_INDEX_PATH", str(BASE_DIR / "data" / "processed" / "faucet_index.faiss")))
    metadata_path: Path = Path(os.getenv("FAUCET_METADATA_PATH", str(BASE_DIR / "data" / "processed" / "faucet_metadata.json")))
    manifest_path: Path = Path(os.getenv("FAUCET_MANIFEST_PATH", str(BASE_DIR / "data" / "processed" / "index_manifest.json")))
    sqlite_path: Path = Path(os.getenv("FAUCET_SQLITE_PATH", str(BASE_DIR / "data" / "processed" / "faucet_catalog.db")))

    # OpenCLIP settings
    openclip_model_name: str = os.getenv("OPENCLIP_MODEL_NAME", "ViT-B-32")
    openclip_pretrained: str = os.getenv("OPENCLIP_PRETRAINED", "openai")
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")

    # Image validation parameters
    max_image_file_size_bytes: int = int(os.getenv("MAX_IMAGE_FILE_SIZE_MB", "10")) * 1024 * 1024
    min_image_dimension: int = int(os.getenv("MIN_IMAGE_DIMENSION", "50"))
    max_image_dimension: int = int(os.getenv("MAX_IMAGE_DIMENSION", "10000"))
    allowed_extensions: Set[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
    allowed_mime_types: Set[str] = frozenset({"image/jpeg", "image/png", "image/webp"})

    # Retrieval parameters
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "50"))
    display_top_k: int = int(os.getenv("DISPLAY_TOP_K", "3"))

    # Uncertainty thresholds (Marked uncalibrated baseline)
    similarity_threshold_high: float = float(os.getenv("SIMILARITY_THRESHOLD_HIGH", "0.85"))
    similarity_threshold_low: float = float(os.getenv("SIMILARITY_THRESHOLD_LOW", "0.80"))
    margin_threshold: float = float(os.getenv("MARGIN_THRESHOLD", "0.03"))

    # Hybrid reranking weights (sum to 1.0)
    weight_image: float = float(os.getenv("WEIGHT_IMAGE", "0.60"))
    weight_text: float = float(os.getenv("WEIGHT_TEXT", "0.20"))
    weight_attributes: float = float(os.getenv("WEIGHT_ATTRIBUTES", "0.15"))
    weight_catalog_quality: float = float(os.getenv("WEIGHT_CATALOG_QUALITY", "0.05"))

    # Labels and messages
    label_likely_match: str = "likely_match"
    label_needs_confirmation: str = "needs_confirmation"
    label_no_reliable_match: str = "no_reliable_match"
    repair_part_disclaimer: str = "Visual candidates only. Confirm the model number before ordering repair parts."


settings = Settings()

