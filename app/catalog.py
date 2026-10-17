"""Catalog ingestion, schema validation, and SQLite product metadata store."""

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.config import settings

logger = logging.getLogger(__name__)

# Required and optional columns as defined in spec
REQUIRED_PRODUCT_COLUMNS = {
    "model_id",
    "model_number",
    "product_title",
    "product_url",
}
# At least one image locator is required
IMAGE_LOCATOR_COLUMNS = {"image_url", "local_path"}

OPTIONAL_PRODUCT_COLUMNS = {
    "category",
    "finish",
    "handle_configuration",
    "mounting_type",
    "spout_type",
    "specifications_json",
    "inventory_status",
    "source_name",
    "source_checked_at",
}

REQUIRED_IMAGE_MANIFEST_COLUMNS = {
    "image_id",
    "model_id",
    "local_path",
}

OPTIONAL_IMAGE_MANIFEST_COLUMNS = {
    "image_url",
    "view_type",
    "content_hash",
    "download_status",
}


def normalize_model_key(model_number: str) -> str:
    """Normalize a model number for flexible lookup while preserving original."""
    if not isinstance(model_number, str):
        model_number = str(model_number)
    return "".join(c for c in model_number.upper() if c.isalnum())


@dataclass
class ValidationReport:
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "stats": self.stats,
        }


def validate_catalog_df(df: pd.DataFrame, base_dir: Optional[Path] = None) -> ValidationReport:
    """Validate catalog products DataFrame against specification requirements."""
    report = ValidationReport()
    base_dir = base_dir or settings.base_dir

    if df.empty:
        report.is_valid = False
        report.errors.append("Catalog CSV is empty.")
        return report

    # 1. Required columns
    cols = set(df.columns)
    missing_required = REQUIRED_PRODUCT_COLUMNS - cols
    if missing_required:
        report.is_valid = False
        report.errors.append(f"Missing required columns: {sorted(list(missing_required))}")

    # Check for image locator (at least one of image_url or local_path)
    if not (cols & IMAGE_LOCATOR_COLUMNS):
        report.is_valid = False
        report.errors.append(f"Catalog must contain at least one of: {sorted(list(IMAGE_LOCATOR_COLUMNS))}")

    if not report.is_valid:
        return report

    total_rows = len(df)
    report.stats["total_rows"] = total_rows

    # 2. Reject blank model IDs and model numbers
    blank_ids = df["model_id"].isna() | (df["model_id"].astype(str).str.strip() == "")
    if blank_ids.any():
        report.is_valid = False
        report.errors.append(f"Found {blank_ids.sum()} rows with blank or missing 'model_id'.")

    blank_nums = df["model_number"].isna() | (df["model_number"].astype(str).str.strip() == "")
    if blank_nums.any():
        report.is_valid = False
        report.errors.append(f"Found {blank_nums.sum()} rows with blank or missing 'model_number'.")

    # 3. Duplicate model_id check
    dup_ids = df[df.duplicated(subset=["model_id"], keep=False)]
    if not dup_ids.empty:
        report.is_valid = False
        unique_dups = dup_ids["model_id"].unique().tolist()
        report.errors.append(f"Found duplicate model_id values: {unique_dups[:10]}")

    # 4. Suspicious duplicate model numbers (different model_id with same model_number)
    dup_nums = df[df.duplicated(subset=["model_number"], keep=False)]
    if not dup_nums.empty:
        unique_dup_nums = dup_nums["model_number"].unique().tolist()
        report.warnings.append(f"Found potential duplicate model_number values: {unique_dup_nums[:10]}")

    # 5. Check local image paths if present
    if "local_path" in df.columns:
        missing_images = 0
        checked_images = 0
        for _, row in df.iterrows():
            lp = row.get("local_path")
            if pd.notna(lp) and str(lp).strip():
                checked_images += 1
                img_path = Path(lp)
                if not img_path.is_absolute():
                    img_path = base_dir / img_path
                if not img_path.is_file():
                    missing_images += 1
        report.stats["checked_local_images"] = checked_images
        report.stats["missing_local_images"] = missing_images
        if missing_images > 0:
            report.warnings.append(f"{missing_images} out of {checked_images} referenced local images do not exist on disk.")

    # 6. Report missing metadata coverage
    coverage = {}
    for col in OPTIONAL_PRODUCT_COLUMNS:
        if col in df.columns:
            non_null_count = int(df[col].notna().sum())
            coverage[col] = f"{non_null_count}/{total_rows} ({non_null_count/total_rows:.1%})"
        else:
            coverage[col] = "column not present"
    report.stats["metadata_coverage"] = coverage

    return report


def validate_image_manifest_df(
    manifest_df: pd.DataFrame,
    catalog_df: Optional[pd.DataFrame] = None,
    base_dir: Optional[Path] = None,
) -> ValidationReport:
    """Validate image manifest DataFrame."""
    report = ValidationReport()
    base_dir = base_dir or settings.base_dir

    if manifest_df.empty:
        report.is_valid = False
        report.errors.append("Image manifest CSV is empty.")
        return report

    cols = set(manifest_df.columns)
    missing_required = REQUIRED_IMAGE_MANIFEST_COLUMNS - cols
    if missing_required:
        report.is_valid = False
        report.errors.append(f"Manifest missing required columns: {sorted(list(missing_required))}")
        return report

    total_images = len(manifest_df)
    report.stats["total_manifest_images"] = total_images

    # Check for blank image_id, model_id, local_path
    for col in ["image_id", "model_id", "local_path"]:
        blanks = manifest_df[col].isna() | (manifest_df[col].astype(str).str.strip() == "")
        if blanks.any():
            report.is_valid = False
            report.errors.append(f"Manifest has {blanks.sum()} blank values in '{col}'.")

    # Check duplicate image_ids
    dup_img_ids = manifest_df[manifest_df.duplicated(subset=["image_id"], keep=False)]
    if not dup_img_ids.empty:
        report.is_valid = False
        report.errors.append(f"Manifest has duplicate image_id values: {dup_img_ids['image_id'].unique().tolist()[:10]}")

    # Check image files exist on disk
    missing_files = []
    for _, row in manifest_df.iterrows():
        lp = row["local_path"]
        if pd.notna(lp):
            p = Path(lp)
            if not p.is_absolute():
                p = base_dir / p
            if not p.is_file():
                missing_files.append(str(lp))
    report.stats["missing_image_files_count"] = len(missing_files)
    if missing_files:
        report.warnings.append(f"{len(missing_files)} manifest image files not found on disk.")

    # Cross check model_id with catalog if provided
    if catalog_df is not None and "model_id" in catalog_df.columns:
        valid_model_ids = set(catalog_df["model_id"].dropna().astype(str))
        manifest_model_ids = set(manifest_df["model_id"].dropna().astype(str))
        unmatched = manifest_model_ids - valid_model_ids
        if unmatched:
            report.warnings.append(f"Found {len(unmatched)} model_ids in manifest not present in catalog: {list(unmatched)[:5]}")

    return report


from contextlib import contextmanager


class CatalogStore:
    """SQLite-backed product catalog store for metadata lookup."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.sqlite_path
        self._init_db()

    @contextmanager
    def _get_connection(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    model_id TEXT PRIMARY KEY,
                    model_number TEXT NOT NULL,
                    normalized_model_number TEXT NOT NULL,
                    product_title TEXT NOT NULL,
                    product_url TEXT NOT NULL,
                    image_url TEXT,
                    local_path TEXT,
                    category TEXT,
                    finish TEXT,
                    handle_configuration TEXT,
                    mounting_type TEXT,
                    spout_type TEXT,
                    specifications_json TEXT,
                    inventory_status TEXT,
                    source_name TEXT,
                    source_checked_at TEXT
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_norm_model_number 
                ON products(normalized_model_number)
                """
            )
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS product_images (
                    image_id TEXT PRIMARY KEY,
                    model_id TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    image_url TEXT,
                    view_type TEXT,
                    content_hash TEXT,
                    download_status TEXT,
                    FOREIGN KEY (model_id) REFERENCES products(model_id)
                )
                """
            )
            cursor.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_img_model_id 
                ON product_images(model_id)
                """
            )
            conn.commit()

    def sync_from_dataframes(
        self,
        products_df: pd.DataFrame,
        manifest_df: Optional[pd.DataFrame] = None,
    ) -> int:
        """Populate database from products and optional manifest DataFrame."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            count = 0
            for _, row in products_df.iterrows():
                m_id = str(row["model_id"]).strip()
                m_num = str(row["model_number"]).strip()
                norm_m_num = normalize_model_key(m_num)
                cursor.execute(
                    """
                    INSERT OR REPLACE INTO products (
                        model_id, model_number, normalized_model_number, product_title,
                        product_url, image_url, local_path, category, finish,
                        handle_configuration, mounting_type, spout_type,
                        specifications_json, inventory_status, source_name, source_checked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        m_id,
                        m_num,
                        norm_m_num,
                        str(row.get("product_title", "")),
                        str(row.get("product_url", "")),
                        str(row.get("image_url", "")) if pd.notna(row.get("image_url")) else None,
                        str(row.get("local_path", "")) if pd.notna(row.get("local_path")) else None,
                        str(row.get("category", "")) if pd.notna(row.get("category")) else None,
                        str(row.get("finish", "")) if pd.notna(row.get("finish")) else None,
                        str(row.get("handle_configuration", "")) if pd.notna(row.get("handle_configuration")) else None,
                        str(row.get("mounting_type", "")) if pd.notna(row.get("mounting_type")) else None,
                        str(row.get("spout_type", "")) if pd.notna(row.get("spout_type")) else None,
                        str(row.get("specifications_json", "")) if pd.notna(row.get("specifications_json")) else None,
                        str(row.get("inventory_status", "")) if pd.notna(row.get("inventory_status")) else None,
                        str(row.get("source_name", "")) if pd.notna(row.get("source_name")) else None,
                        str(row.get("source_checked_at", "")) if pd.notna(row.get("source_checked_at")) else None,
                    ),
                )
                count += 1

            if manifest_df is not None and not manifest_df.empty:
                for _, row in manifest_df.iterrows():
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO product_images (
                            image_id, model_id, local_path, image_url, view_type,
                            content_hash, download_status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            str(row["image_id"]),
                            str(row["model_id"]),
                            str(row["local_path"]),
                            str(row.get("image_url", "")) if pd.notna(row.get("image_url")) else None,
                            str(row.get("view_type", "")) if pd.notna(row.get("view_type")) else None,
                            str(row.get("content_hash", "")) if pd.notna(row.get("content_hash")) else None,
                            str(row.get("download_status", "")) if pd.notna(row.get("download_status")) else None,
                        ),
                    )

            conn.commit()
            return count

    def get_product(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Fetch product details by canonical model_id."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM products WHERE model_id = ?", (model_id,))
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_product_by_model_number(self, model_number: str) -> Optional[Dict[str, Any]]:
        """Fetch product by model number (exact or normalized)."""
        norm = normalize_model_key(model_number)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT * FROM products WHERE normalized_model_number = ? LIMIT 1",
                (norm,),
            )
            row = cursor.fetchone()
            if row:
                return dict(row)
            return None

    def get_all_products(self) -> List[Dict[str, Any]]:
        """Fetch all products."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM products")
            return [dict(r) for r in cursor.fetchall()]

    def count_products(self) -> int:
        """Count total products."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM products")
            return cursor.fetchone()[0]

