"""CLI script to validate products CSV and image manifest CSV and sync to SQLite."""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import pandas as pd

from app.catalog import (
    CatalogStore,
    validate_catalog_df,
    validate_image_manifest_df,
)
from app.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_catalog")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate catalog CSV and image manifest CSV.")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=settings.catalog_csv,
        help="Path to products CSV (default: data/products.csv)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=settings.image_manifest_csv,
        help="Path to image manifest CSV (default: data/image_manifest.csv)",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=settings.processed_dir / "catalog_validation_report.json",
        help="Path to output validation report JSON",
    )
    parser.add_argument(
        "--sync-db",
        action="store_true",
        default=True,
        help="Sync validated products to SQLite metadata store",
    )

    args = parser.parse_args()

    print("==================================================")
    print(" Delta Faucet POC — Catalog & Manifest Validation")
    print("==================================================")
    print(f"Catalog file:  {args.catalog}")
    print(f"Manifest file: {args.manifest}")
    print("--------------------------------------------------")

    # 1. Check if files exist
    if not args.catalog.is_file():
        print(f"\n[ERROR] Catalog file not found: {args.catalog}")
        print("Please provide an authorized Delta Faucet catalog CSV at that path or specify --catalog.")
        print("Required columns: model_id, model_number, product_title, product_url, and (local_path or image_url).")
        return 1

    try:
        catalog_df = pd.read_csv(args.catalog)
        print(f"Loaded catalog: {len(catalog_df)} rows, {len(catalog_df.columns)} columns.")
    except Exception as e:
        print(f"\n[ERROR] Failed to read catalog CSV: {e}")
        return 1

    # 2. Validate catalog
    cat_report = validate_catalog_df(catalog_df, base_dir=BASE_DIR)

    manifest_df = None
    man_report = None
    if args.manifest.is_file():
        try:
            manifest_df = pd.read_csv(args.manifest)
            print(f"Loaded manifest: {len(manifest_df)} rows, {len(manifest_df.columns)} columns.")
            man_report = validate_image_manifest_df(manifest_df, catalog_df=catalog_df, base_dir=BASE_DIR)
        except Exception as e:
            print(f"\n[WARNING] Failed to read image manifest CSV: {e}")
    else:
        print(f"[INFO] Image manifest file not found: {args.manifest} (optional if catalog has local_path).")

    # 3. Print Report Summary
    print("\n---------------- Validation Summary ----------------")
    if cat_report.is_valid:
        print("✓ Catalog schema and primary keys: PASSED")
    else:
        print("✕ Catalog validation: FAILED")
        for err in cat_report.errors:
            print(f"   • Error: {err}")

    if cat_report.warnings:
        print(f"\nCatalog Warnings ({len(cat_report.warnings)}):")
        for warn in cat_report.warnings:
            print(f"   ⚠ {warn}")

    if man_report:
        if man_report.is_valid:
            print("\n✓ Image manifest validation: PASSED")
        else:
            print("\n✕ Image manifest validation: FAILED")
            for err in man_report.errors:
                print(f"   • Error: {err}")
        if man_report.warnings:
            print(f"Manifest Warnings ({len(man_report.warnings)}):")
            for warn in man_report.warnings:
                print(f"   ⚠ {warn}")

    print("\nMetadata Coverage Statistics:")
    for k, v in cat_report.stats.get("metadata_coverage", {}).items():
        print(f"   - {k}: {v}")

    # 4. Save validation report JSON
    full_report = {
        "catalog_file": str(args.catalog),
        "manifest_file": str(args.manifest) if manifest_df is not None else None,
        "catalog": cat_report.to_dict(),
        "manifest": man_report.to_dict() if man_report else None,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)
    print(f"\nValidation report saved to: {args.report}")

    # 5. Sync to SQLite if valid
    if cat_report.is_valid and args.sync_db:
        store = CatalogStore(settings.sqlite_path)
        count = store.sync_from_dataframes(catalog_df, manifest_df)
        print(f"✓ Synchronized {count} products into SQLite metadata store ({settings.sqlite_path}).")

    if not cat_report.is_valid or (man_report and not man_report.is_valid):
        return 1

    print("\n==================================================")
    print("Catalog validation complete. Ready for index generation.")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

