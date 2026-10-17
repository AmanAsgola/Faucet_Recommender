"""CLI script to build FAISS vector index from local approved catalog images."""

import argparse
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from PIL import Image

from app.catalog import CatalogStore
from app.config import settings
from app.embeddings import ImageEmbedder
from app.image_utils import ImageValidationError, validate_and_load_image
from app.index_store import IndexManifest, IndexStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("build_index")


def compute_file_sha256(path: Path) -> str:
    """Compute sha256 of a file."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build FAISS vector index for catalog images.")
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
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation",
    )

    args = parser.parse_args()

    print("==================================================")
    print(" Delta Faucet POC — FAISS Index Generator")
    print("==================================================")

    # 1. Check prerequisites
    if not args.catalog.is_file():
        print(f"\n[ERROR] Catalog file not found at: {args.catalog}")
        print("Please ensure 'data/products.csv' exists with approved catalog data.")
        return 1

    try:
        catalog_df = pd.read_csv(args.catalog)
    except Exception as e:
        print(f"[ERROR] Failed to read catalog CSV: {e}")
        return 1

    # Map model metadata for quick lookup
    model_meta_map: Dict[str, Dict[str, Any]] = {}
    for _, row in catalog_df.iterrows():
        m_id = str(row["model_id"])
        model_meta_map[m_id] = {
            "model_id": m_id,
            "model_number": str(row.get("model_number", m_id)),
            "product_title": str(row.get("product_title", "")),
            "product_url": str(row.get("product_url", "")),
            "category": str(row.get("category", "")) if pd.notna(row.get("category")) else None,
            "finish": str(row.get("finish", "")) if pd.notna(row.get("finish")) else None,
        }

    # Determine image items to process
    items_to_process: List[Dict[str, Any]] = []

    if args.manifest.is_file():
        try:
            manifest_df = pd.read_csv(args.manifest)
            for _, row in manifest_df.iterrows():
                items_to_process.append({
                    "image_id": str(row["image_id"]),
                    "model_id": str(row["model_id"]),
                    "local_path": str(row["local_path"]),
                    "view_type": str(row.get("view_type", "primary")),
                })
            print(f"Loaded {len(items_to_process)} image entries from manifest: {args.manifest}")
        except Exception as e:
            print(f"[WARNING] Failed to read manifest CSV ({e}). Falling back to catalog local_path.")

    if not items_to_process and "local_path" in catalog_df.columns:
        print("Using local_path entries directly from products catalog.")
        for idx, row in catalog_df.iterrows():
            lp = row.get("local_path")
            if pd.notna(lp) and str(lp).strip():
                items_to_process.append({
                    "image_id": f"img_{row['model_id']}_{idx}",
                    "model_id": str(row["model_id"]),
                    "local_path": str(lp),
                    "view_type": "primary",
                })

    if not items_to_process:
        print("\n[ERROR] No image entries found to index.")
        print("Please provide a valid image manifest or ensure data/products.csv has a 'local_path' column.")
        return 1

    # 2. Initialize Embedder
    print(f"\nInitializing OpenCLIP Embedder ({settings.openclip_model_name}, device={settings.embedding_device})...")
    try:
        embedder = ImageEmbedder.get_instance()
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenCLIP model: {e}")
        return 1

    # 3. Process and embed images
    valid_vectors: List[np.ndarray] = []
    metadata_list: List[Dict[str, Any]] = []
    failures: List[Dict[str, str]] = []
    skipped_count = 0

    print(f"\nProcessing {len(items_to_process)} catalog images...")

    for item in items_to_process:
        img_id = item["image_id"]
        model_id = item["model_id"]
        rel_path = item["local_path"]

        # Resolve path
        img_path = Path(rel_path)
        if not img_path.is_absolute():
            img_path = BASE_DIR / img_path

        if not img_path.is_file():
            skipped_count += 1
            failures.append({
                "image_id": img_id,
                "model_id": model_id,
                "path": str(rel_path),
                "reason": "File does not exist on disk",
            })
            continue

        # Validate image format, dimensions, corruption
        try:
            pil_img = validate_and_load_image(img_path)
        except ImageValidationError as e:
            skipped_count += 1
            failures.append({
                "image_id": img_id,
                "model_id": model_id,
                "path": str(rel_path),
                "reason": f"Image validation failed: {e}",
            })
            continue
        except Exception as e:
            skipped_count += 1
            failures.append({
                "image_id": img_id,
                "model_id": model_id,
                "path": str(rel_path),
                "reason": f"Unexpected load error: {e}",
            })
            continue

        # Embed image
        try:
            vec = embedder.embed_image(pil_img)
            valid_vectors.append(vec)

            # Metadata entry
            m_info = model_meta_map.get(model_id, {})
            metadata_list.append({
                "vector_index": len(valid_vectors) - 1,
                "image_id": img_id,
                "model_id": model_id,
                "model_number": m_info.get("model_number", model_id),
                "product_title": m_info.get("product_title", ""),
                "product_url": m_info.get("product_url", ""),
                "local_path": str(rel_path),
                "category": m_info.get("category"),
                "finish": m_info.get("finish"),
                "view_type": item.get("view_type", "primary"),
            })
        except Exception as e:
            skipped_count += 1
            failures.append({
                "image_id": img_id,
                "model_id": model_id,
                "path": str(rel_path),
                "reason": f"Embedding failure: {e}",
            })

    if not valid_vectors:
        print("\n[ERROR] Zero valid images could be embedded. Index cannot be created.")
        print(f"Skipped {skipped_count} items. Please inspect failures in logs.")
        return 1

    vectors_arr = np.vstack(valid_vectors)
    print(f"Generated {len(vectors_arr)} embeddings (dim={vectors_arr.shape[1]}). Skipped: {skipped_count}.")

    # 4. Build FAISS index and save
    print("Building FAISS Inner-Product Index...")
    index_store = IndexStore()
    faiss_index = index_store.build_index(vectors_arr)

    # Manifest
    catalog_hash = compute_file_sha256(args.catalog)
    embedder_meta = embedder.get_metadata()
    manifest = IndexManifest(
        build_timestamp=datetime.now(timezone.utc).isoformat(),
        model_name=embedder_meta["model_name"],
        pretrained=embedder_meta["pretrained"],
        embedding_dimension=embedder_meta["dimension"],
        vector_count=len(vectors_arr),
        open_clip_version=embedder_meta["open_clip_version"],
        torch_version=embedder_meta["torch_version"],
        catalog_source=f"{args.catalog} (sha256:{catalog_hash[:12]})",
        skipped_count=skipped_count,
        failures=failures[:100],  # Keep up to 100 failures
    )

    index_store.save(faiss_index, metadata_list, manifest)

    # Sync Catalog SQLite
    cat_store = CatalogStore(settings.sqlite_path)
    man_df = pd.DataFrame(items_to_process) if items_to_process else None
    cat_store.sync_from_dataframes(catalog_df, man_df)

    print("\n==================================================")
    print("✓ FAISS Vector Index Successfully Built!")
    print(f"   - Index file:     {settings.index_path}")
    print(f"   - Metadata file:  {settings.metadata_path}")
    print(f"   - Manifest file:  {settings.manifest_path}")
    print(f"   - Total vectors:  {len(vectors_arr)}")
    print(f"   - Skipped images: {skipped_count}")
    print("==================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())

