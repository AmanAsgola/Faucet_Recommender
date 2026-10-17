"""Extract authentic Delta Faucet products, specifications, and images from the official PDF catalogue."""

import hashlib
import io
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
import pymupdf
from PIL import Image

from app.image_utils import validate_and_load_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("extract_catalogue")

PDF_PATH = BASE_DIR / "Kitchen & Bath Full-Line Catalogue.pdf"
RAW_IMAGES_DIR = BASE_DIR / "data" / "raw" / "images"

# Delta Model number pattern
MODEL_PATTERN = re.compile(
    r"\b(\d{3,5}(?:[A-Z]{1,4})?(?:-[A-Z0-9]{1,8})+|\d{3,5}LF(?:-[A-Z0-9]+)*|\d{3,5}T(?:-[A-Z0-9]+)*)\b"
)

# Known finishes mapping from catalogue suffix or text
FINISH_MAP = {
    "AR": "Arctic Stainless",
    "SS": "Stainless",
    "SP": "SpotShield Stainless",
    "BL": "Matte Black",
    "CZ": "Champagne Bronze",
    "RB": "Venetian Bronze",
    "KS": "Black Stainless",
    "PN": "Polished Nickel",
    "PR": "Lumicoat Chrome",
    "WH": "White",
    "PB": "Brilliance Brass",
    "CH": "Chrome",
}

COLLECTIONS = [
    "Trinsic", "Mateo", "Esque", "Essa", "Allora", "Fuse", "Addison",
    "Marley", "Osler", "Victorian", "Leland", "Cassidy", "Linden",
    "Coranto", "Broadmoor", "Foundations", "Classic", "Pivotal",
    "Stryke", "Dorval", "Kayra", "Saylor", "Brogden", "Tetrao",
    "Ara", "Zura", "Vero", "Lahara", "Ashlyn", "Dryden", "Compel",
    "Tesla", "Roux", "Nicoli", "Portwood", "Windemere", "Peerless"
]


def clean_text(text: str) -> str:
    """Clean whitespace and common ligatures."""
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    return " ".join(text.split()).strip()


def extract_catalogue() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parse PDF catalogue and extract product records and images."""
    if not PDF_PATH.is_file():
        raise FileNotFoundError(f"Catalogue PDF not found: {PDF_PATH}")

    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    doc = pymupdf.open(str(PDF_PATH))
    total_pages = len(doc)
    logger.info(f"Opened catalogue '{PDF_PATH.name}' ({total_pages} pages).")

    products_list: List[Dict[str, Any]] = []
    manifest_list: List[Dict[str, Any]] = []
    seen_model_ids: Set[str] = set()

    current_category = "Kitchen"  # Default initial section
    current_collection = None

    for page_idx in range(total_pages):
        page = doc[page_idx]
        page_num = page_idx + 1
        page_text = page.get_text()

        # Detect section transition (e.g. Bath vs Kitchen)
        if "BATH" in page_text.upper() and page_num > 40:
            current_category = "Bath"
        elif "KITCHEN" in page_text.upper() and page_num < 45:
            current_category = "Kitchen"

        # Check for collection heading
        for coll in COLLECTIONS:
            if re.search(rf"\b{coll}\b", page_text, re.IGNORECASE):
                current_collection = coll
                break

        # Extract text blocks
        blocks = page.get_text("blocks")
        candidate_blocks = []

        for b in blocks:
            b_text = clean_text(b[4])
            matches = MODEL_PATTERN.findall(b_text)
            for m in matches:
                # Exclude replacement parts (RP) or page ranges
                if m.startswith("RP") or re.match(r"^\d{2,3}-\d{2,3}$", m):
                    continue
                candidate_blocks.append({
                    "model_number": m,
                    "x0": b[0],
                    "y0": b[1],
                    "x1": b[2],
                    "y1": b[3],
                    "full_text": b_text,
                })

        if not candidate_blocks:
            continue

        # Extract images on this page
        page_images = []
        for img_info in page.get_images():
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            if not rects:
                continue
            r = rects[0]
            try:
                bi = doc.extract_image(xref)
                w, h = bi["width"], bi["height"]
                # Filter out small swatches, icons, or non-product graphics
                if w >= 60 and h >= 60 and (r.x1 - r.x0) >= 30 and (r.y1 - r.y0) >= 30:
                    page_images.append({
                        "xref": xref,
                        "width": w,
                        "height": h,
                        "image_bytes": bi["image"],
                        "rect": r,
                        "center_x": (r.x0 + r.x1) / 2.0,
                        "center_y": (r.y0 + r.y1) / 2.0,
                    })
            except Exception:
                continue

        # Associate each model candidate with the nearest image on page
        for cb in candidate_blocks:
            raw_model = cb["model_number"]
            canonical_model_id = f"DELTA-{raw_model.upper().replace(' ', '')}"

            if canonical_model_id in seen_model_ids:
                continue

            block_text = cb["full_text"]
            block_cx = (cb["x0"] + cb["x1"]) / 2.0
            block_top = cb["y0"]

            # Derive title
            title_parts = []
            if current_collection:
                title_parts.append(current_collection)
            # Check for faucet type in text
            if "PULL-DOWN" in block_text.upper():
                title_parts.append("Pull-Down")
            elif "PULL-OUT" in block_text.upper():
                title_parts.append("Pull-Out")
            elif "WIDESPREAD" in block_text.upper():
                title_parts.append("Two-Handle Widespread")
            elif "SINGLE-HANDLE" in block_text.upper() or "SINGLE HANDLE" in block_text.upper():
                title_parts.append("Single Handle")
            elif "TWO-HANDLE" in block_text.upper() or "TWO HANDLE" in block_text.upper():
                title_parts.append("Two Handle")

            if "BAR/PREP" in block_text.upper():
                title_parts.append("Bar / Prep Faucet")
            elif "BEVERAGE" in block_text.upper():
                title_parts.append("Beverage Faucet")
            elif "LAVATORY" in block_text.upper() or current_category == "Bath":
                title_parts.append("Bathroom Faucet")
            else:
                title_parts.append("Kitchen Faucet")

            product_title = f"Delta {' '.join(title_parts)}"

            # Derive finish
            finish = "Chrome"  # Delta catalog default base finish
            for code, name in FINISH_MAP.items():
                if f"-{code}-" in raw_model or raw_model.endswith(f"-{code}"):
                    finish = name
                    break

            # Handle configuration & mounting
            handle_cfg = "Single Handle" if "SINGLE" in block_text.upper() else ("Two Handle" if "TWO" in block_text.upper() else "Single Handle")
            mounting = "Single Hole" if "1-hole" in block_text.lower() else ("Deck Mount" if "3-hole" in block_text.lower() or "deckplate" in block_text.lower() else "Deck Mount")
            spout = "High Arc Pull-Down" if "PULL-DOWN" in block_text.upper() else ("Low Arc Pull-Out" if "PULL-OUT" in block_text.upper() else "Standard Spout")

            # Match nearest image above the text block
            best_img = None
            best_dist = float("inf")

            for img in page_images:
                # Faucet images are typically above the text (img.center_y <= block_top + 30)
                # and aligned horizontally
                dx = abs(img["center_x"] - block_cx)
                dy = block_top - img["center_y"]
                
                # Favor images above and vertically close
                if dy >= -40:
                    dist = (dx * 1.5) ** 2 + dy ** 2
                else:
                    dist = (dx * 2.0) ** 2 + (abs(dy) * 3.0) ** 2

                if dist < best_dist:
                    best_dist = dist
                    best_img = img

            local_img_path = None
            if best_img is not None:
                img_filename = f"{canonical_model_id}.jpg"
                dest_path = RAW_IMAGES_DIR / img_filename
                try:
                    # Validate and convert to RGB JPEG
                    pil_img = validate_and_load_image(best_img["image_bytes"])
                    pil_img.save(dest_path, "JPEG", quality=92)
                    rel_path = str(dest_path.relative_to(BASE_DIR)).replace("\\", "/")
                    local_img_path = rel_path

                    # Add to manifest
                    manifest_list.append({
                        "image_id": f"img_{canonical_model_id}",
                        "model_id": canonical_model_id,
                        "local_path": rel_path,
                        "view_type": "catalog_product",
                        "content_hash": hashlib.sha256(best_img["image_bytes"]).hexdigest()[:16],
                        "download_status": "pdf_extracted",
                    })
                except Exception as e:
                    logger.debug(f"Could not convert image for {canonical_model_id}: {e}")

            # Official product URL pattern
            clean_num = raw_model.split()[0]
            product_url = f"https://www.deltafaucet.com/product/{clean_num}"

            products_list.append({
                "model_id": canonical_model_id,
                "model_number": raw_model,
                "product_title": product_title,
                "product_url": product_url,
                "local_path": local_img_path if local_img_path else "",
                "category": current_category,
                "finish": finish,
                "handle_configuration": handle_cfg,
                "mounting_type": mounting,
                "spout_type": spout,
                "specifications_json": json.dumps({
                    "collection": current_collection,
                    "catalog_page": page_num,
                    "catalogue_source": "Delta Kitchen & Bath Full-Line Catalogue",
                }),
                "inventory_status": "Authorized Catalog Item",
                "source_name": "Kitchen & Bath Full-Line Catalogue.pdf",
                "source_checked_at": "2026-10-17T00:00:00Z",
            })
            seen_model_ids.add(canonical_model_id)

    prod_df = pd.DataFrame(products_list)
    man_df = pd.DataFrame(manifest_list)
    return prod_df, man_df


def main():
    print("==================================================")
    print(" Extracting Products from Delta PDF Catalogue")
    print("==================================================")
    prod_df, man_df = extract_catalogue()

    print(f"\nExtraction complete:")
    print(f"  • Extracted Products: {len(prod_df)}")
    print(f"  • Extracted Images:   {len(man_df)}")

    # Only include products that have extracted local images for indexing
    valid_prod_df = prod_df[prod_df["local_path"] != ""].copy()
    print(f"  • Products with Valid Images: {len(valid_prod_df)}")

    # Write to data/products.csv and data/image_manifest.csv
    valid_prod_df.to_csv(BASE_DIR / "data" / "products.csv", index=False)
    man_df.to_csv(BASE_DIR / "data" / "image_manifest.csv", index=False)

    print("\nSaved updated catalog to:")
    print(f"  - {BASE_DIR / 'data' / 'products.csv'}")
    print(f"  - {BASE_DIR / 'data' / 'image_manifest.csv'}")
    print("==================================================")


if __name__ == "__main__":
    main()

